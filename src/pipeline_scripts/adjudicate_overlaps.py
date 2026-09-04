"""
Answer what the measurements and the ontology leave open about a scene's overlaps.

Two things stand between a labelled mesh and a world of bodies, and neither follows from
the geometry:

- **who owns a face two labels both claim**, without which the mesh cannot be split, since
  a face belongs to exactly one body;
- **which whole a part belongs to**, where a door meets more than one cabinet.

Both are asked as few times as they are actually open. Ownership is asked once per set of
claimants rather than once per pair of them, never where the ontology already settled
every relation inside the set, and once per *pattern* of classes rather than once per
occurrence -- a door and a window sharing a pane is one question however many glazed doors
the room has. Membership is asked only where a part meets more than one candidate::

    python -m pipeline_scripts.adjudicate_overlaps pipeline_out_relations

It writes ``adjudications.json``, which the split then reads. It decides nothing itself.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from pipeline_scripts import model_client

OWNERSHIP_PROMPT = """\
You decide whose surface a piece of a scanned room is.

Several objects were labelled onto the same faces, and every face has to be given to
exactly one of them before the room can be cut into objects at all. You are shown one
such piece: each object in its own color, and the faces all of them claim in one more.

Answer with the object whose surface those contested faces are. The one to pick is the
one the faces *are*: a drawer front is the drawer's surface even though the cabinet it
sits in was labelled over it too, and a handle is the handle's even though it was
labelled as the door it is screwed to.

Answer for the kind of situation, not for this one room: the same answer will be used
everywhere these labels meet like this.

Answer with JSON and nothing else:
{"owner": "<one of the labels>", "confidence": 0.0, "reason": "one sentence"}"""

MEMBERSHIP_PROMPT = """\
You decide which object a part belongs to.

A scanned room labelled a part that meets several objects it could belong to. The
ontology already says it is a part of one of them; which one is what the pictures show.
Each candidate has its own color and the part has its own.

Judge by what the pictures show, not by how much surface is shared: a part sits in one of
them and merely touches the others.

Answer with JSON and nothing else:
{"whole": "<one of the candidates>", "confidence": 0.0, "reason": "one sentence"}"""

CAPTIONS = {
    "closeup": "Picture 1 -- the objects alone, with nothing in front of them.",
    "context": "Picture 2 -- where they are in the room, painted the same way.",
    "plain": "Picture 3 -- the same objects, in the colors they were scanned in.",
}
"""
What each render shows, in the order they are shown in.
"""


def pictures(question: Dict[str, Any], images: Path) -> List[Dict[str, Any]]:
    """
    :param question: The question, as ``questions.json`` holds it.
    :param images: The directory holding its renders.
    :return: The renders as parts of a message, captioned.
    """
    named = {
        filename.rsplit("__", 1)[-1].split("_", 1)[0]: filename
        for filename in question["images"]
    }
    content = []
    for kind, caption in CAPTIONS.items():
        if kind in named:
            content.append(model_client.text_part(caption))
            content.append(model_client.image_part(images / named[kind]))
    return content


def painted(question: Dict[str, Any], labels: Dict[str, str]) -> str:
    """
    :param question: The question, as ``questions.json`` holds it.
    :param labels: Per segment, the label it carries.
    :return: What each color in the pictures stands for.
    """
    legend = question.get("legend", {})
    lines = [
        f'{name} (labelled "{labels[name]}") is {legend[name]}'
        for name in question["shown"]
        if name in legend
    ]
    if "contested" in legend:
        lines.append(f"the faces all of them claim are {legend['contested']}")
    return "\n".join(lines)


def ownership_question(
    question: Dict[str, Any], labels: Dict[str, str], images: Path
) -> List[Dict[str, Any]]:
    """
    Build the message asking whose the contested faces are.

    :param question: The question, as ``questions.json`` holds it.
    :param labels: Per segment, the label it carries.
    :param images: The directory holding its renders.
    :return: The message, as :func:`model_client.ask` takes it.
    """
    covered = len(question["covers"])
    return [
        model_client.text_part(
            f"## The labels\n{', '.join(question['pattern'])}\n\n"
            f"## The picture\n{painted(question, labels)}\n\n"
            f"## How often this happens\n"
            f"Objects with these labels are labelled over the same faces "
            f"{covered} time(s) in this room, {question['contested_faces']} faces in "
            f"all. The pictures show the largest of them."
        )
    ] + pictures(question, images)


def membership_question(
    question: Dict[str, Any], labels: Dict[str, str], images: Path
) -> List[Dict[str, Any]]:
    """
    Build the message asking which whole a part belongs to.

    :param question: The question, as ``questions.json`` holds it.
    :param labels: Per segment, the label it carries.
    :param images: The directory holding its renders.
    :return: The message, as :func:`model_client.ask` takes it.
    """
    measured = "\n".join(
        f"{name}: shares {how['shared_faces']} faces with it, and would hold it in "
        f"its {how['field']}"
        for name, how in question["candidates"].items()
    )
    return [
        model_client.text_part(
            f"## The part\n{question['part']}, labelled "
            f"\"{labels[question['part']]}\"\n\n"
            f"## The candidates\n{measured}\n\n"
            f"## The picture\n{painted(question, labels)}"
        )
    ] + pictures(question, images)


def check(question: Dict[str, Any], answer: Dict[str, Any]) -> List[str]:
    """
    Say what is wrong with an answer, if anything.

    :param question: The question it answers.
    :param answer: What the model said.
    :return: One sentence per problem, empty when there are none.
    """
    if question["kind"] == "ownership":
        allowed, given, what = question["pattern"], answer.get("owner"), "owner"
    else:
        allowed, given, what = (
            list(question["candidates"]),
            answer.get("whole"),
            "whole",
        )
    if given not in allowed:
        return [f"{given!r} is not one of the {what}s to choose from: {', '.join(allowed)}"]
    return []


def ask_about(
    question: Dict[str, Any],
    labels: Dict[str, str],
    images: Path,
    answers: Path,
    model: str,
    reuse: bool,
    problems: Sequence[str] = (),
) -> Dict[str, Any]:
    """
    Put one question to the model, or read back what it already said about it.

    :param question: The question, as ``questions.json`` holds it.
    :param labels: Per segment, the label it carries.
    :param images: The directory holding the renders.
    :param answers: The directory raw responses are kept in.
    :param model: Which model to ask.
    :param reuse: Whether to read a kept response rather than ask again.
    :param problems: What was wrong with the previous answer, when there was one.
    :return: The response.
    """
    kept = answers / f"{question['kind']}__{question['name']}.json"
    if reuse and kept.exists() and not problems:
        return json.loads(kept.read_text())

    build = (
        ownership_question if question["kind"] == "ownership" else membership_question
    )
    content = build(question, labels, images)
    if problems:
        content.append(
            model_client.text_part(
                "Your previous answer could not be used:\n- "
                + "\n- ".join(problems)
                + "\nAnswer the same question again, correcting that."
            )
        )
    response = model_client.ask(
        content,
        OWNERSHIP_PROMPT if question["kind"] == "ownership" else MEMBERSHIP_PROMPT,
        model=model,
    )
    answers.mkdir(parents=True, exist_ok=True)
    kept.write_text(json.dumps(response, indent=2))
    return response


def build(arguments: argparse.Namespace) -> None:
    """
    Answer the open questions of an evidence run.

    :param arguments: The command line arguments.
    """
    evidence = arguments.evidence_directory
    questions = json.loads((evidence / "questions.json").read_text())
    relations = json.loads((evidence / "relations.json").read_text())
    labels = {segment["name"]: segment["class"] for segment in relations["segments"]}

    asked = questions["ownership"] + questions["membership"]
    without = [question for question in asked if not question["images"]]
    if without:
        print(
            f"{len(without)} of {len(asked)} questions have no renders yet; run "
            f"build_relation_evidence with --question-renders to make them."
        )
    asked = [question for question in asked if question["images"]]
    asked = asked[: arguments.limit] if arguments.limit else asked

    print(f"asking {arguments.model} about {len(asked)} questions ...")
    answered = []
    for question in asked:
        problems: Sequence[str] = ()
        for attempt in range(1 + arguments.corrections):
            response = ask_about(
                question,
                labels,
                evidence / "questions",
                evidence / "question_answers",
                arguments.model,
                arguments.reuse_answers,
                problems,
            )
            try:
                answer = model_client.parse_json_answer(
                    model_client.answer_text(response)
                )
            except model_client.ModelRefusedError as refusal:
                answer, problems = {}, [str(refusal)]
            else:
                problems = check(question, answer)
            if not problems:
                break
            if attempt + 1 <= arguments.corrections:
                print(f"  {question['name']}: {problems[0]}, asking again ...")

        decided = {
            "kind": question["kind"],
            "name": question["name"],
            "problems": list(problems),
            "confidence": answer.get("confidence"),
            "reason": answer.get("reason"),
        }
        if question["kind"] == "ownership":
            decided.update(
                pattern=question["pattern"],
                owner=answer.get("owner"),
                covers=question["covers"],
            )
            print(f"  {question['name']:<44} -> {decided['owner']}")
        else:
            decided.update(part=question["part"], whole=answer.get("whole"))
            print(f"  {question['name']:<44} in {decided['whole']}")
        for problem in decided["problems"]:
            print(f"      ! {problem}")
        answered.append(decided)

    arguments.output.write_text(
        json.dumps(
            {
                "model": arguments.model,
                "scene": relations["scene"],
                "answered": answered,
                "settled": questions["settled"],
                "forced": questions["forced"],
            },
            indent=2,
        )
    )

    counted = Counter(one["kind"] for one in answered)
    troubled = [one for one in answered if one["problems"]]
    print(
        f"\n{counted['ownership']} patterns and {counted['membership']} memberships "
        f"answered, {len(troubled)} with problems"
    )
    print(f"written to {arguments.output}")


def main() -> None:
    """
    Answer the open questions of the evidence run named on the command line.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "evidence_directory",
        type=Path,
        help="The directory build_relation_evidence wrote.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the decisions, by default adjudications.json beside the "
        "evidence they were made from.",
    )
    parser.add_argument("--model", default=model_client.DEFAULT_MODEL,
                        help="Which model to ask.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Ask only the first so many questions.")
    parser.add_argument(
        "--corrections",
        type=int,
        default=1,
        help="How often an unusable answer is put back to the model with what was "
        "wrong with it.",
    )
    parser.add_argument(
        "--reuse-answers",
        action="store_true",
        help="Read back the kept responses instead of asking again.",
    )
    arguments = parser.parse_args()
    if arguments.output is None:
        arguments.output = arguments.evidence_directory / "adjudications.json"
    build(arguments)


if __name__ == "__main__":
    main()
