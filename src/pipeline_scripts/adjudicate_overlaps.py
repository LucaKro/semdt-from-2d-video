"""
Decide what two overlapping labels are to each other, and whose faces the shared ones are.

Where a scan labels the same faces twice -- a drawer front that is also the cabinet, a
pane that is also the door -- two things stay open after everything measurable has been
measured and the ontology has said what it admits:

- what the two objects are to each other, which decides how the world mounts them,
- which of them the contested faces belong to, which is what lets the mesh be split at
  all, since a face can only be given to one body.

Neither follows from the geometry: a handle 100% inside a drawer and a drawer front 100%
inside a cabinet measure the same and mean different things. This asks a model, pair by
pair, with the measurements, what the ontology admits, and the three pictures the
evidence run produced::

    python -m pipeline_scripts.adjudicate_overlaps pipeline_out_relations

It decides nothing itself and mounts nothing: it writes ``adjudications.json``, which the
split and the mounting then read.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from pipeline_scripts import model_client

PART = "part"
"""
One is a structural part of the other, mounted with ``add()``.
"""

CONTAINS = "contains"
"""
One is merely inside or on the other, mounted with ``add_object()``.
"""

SUPPORTS = "supports"
"""
One is a surface the other rests on, mounted with ``add_supporting_surface()``.
"""

SAME_OBJECT = "same-object"
"""
Two labels were put on one object, so only one body should come out of them.
"""

UNRELATED = "unrelated"
"""
They stand in no relation the world mounts; they merely share a surface.
"""

RELATIONS = (PART, CONTAINS, SUPPORTS, SAME_OBJECT, UNRELATED)
"""
Everything a pair may be answered with.
"""

ANSWER_FIELDS = ("relation", "whole", "field", "owner", "confidence", "reason")
"""
What is kept of an answer, so a model's extra fields do not reach the file.
"""

SYSTEM_PROMPT = """\
You decide what two overlapping objects in a scanned room are to each other.

Both carry a label from the scan, and some faces carry both labels. Two things are open:

1. What they are to each other:
   - "part": one is a structural part of the other, mounted with add(),
   - "contains": one is merely inside or resting on the other, mounted with add_object(),
   - "supports": one is a surface the other rests on,
   - "same-object": the two labels were put on one and the same object,
   - "unrelated": they stand in no relation; they only share a surface.
2. Which of the two the faces they both claim belong to. The mesh is split by giving
   every face to exactly one object, so this has to be answered whatever the relation is.
   A part keeps its own surface; the thing it is part of is what is left over.

You are told which relations the ontology admits between their classes. Do not answer
"part", "contains" or "supports" with anything it does not admit -- if none of them fits
what you see, answer "unrelated".

Nothing in the measurements settles this on its own: a handle wholly inside a drawer and
a drawer front wholly inside a cabinet measure alike and mean different things. Read the
pictures.

Answer with JSON and nothing else:
{"relation": "part", "whole": "<the name of the one that holds>",
 "field": "<the field it would be held in, when the relation is part, else null>",
 "owner": "<the name of the one the shared faces belong to>",
 "confidence": 0.0, "reason": "one sentence"}"""

CAPTIONS = {
    "closeup": "Picture 1 -- the two of them alone, with nothing in front of them.",
    "context": "Picture 2 -- where they are in the room, painted the same way.",
    "plain": "Picture 3 -- the same two, in the colors they were scanned in.",
}
"""
What each of a pair's three renders shows, in the order they are shown in.
"""


def name_of(record: Dict[str, Any], which: str) -> str:
    """
    :param record: The pair's record of ``relations.json``.
    :param which: ``one`` or ``other``.
    :return: The name of that segment.
    """
    return record[which]


def admits(record: Dict[str, Any]) -> str:
    """
    Say what the ontology admits between a pair's classes, as a model reads it.

    :param record: The pair's record of ``relations.json``.
    :return: The admissible mounts, or a sentence saying there are none.
    """
    lines = [
        f"part: {relation['whole']}.{relation['field']} may hold a "
        f"{relation['part']}, mounted with add()"
        + (
            "  (mounting here cuts the part's volume out of the whole)"
            if relation.get("removes_geometry")
            else ""
        )
        for relation in record["admissible"]
    ]
    lines += [
        f"{mount['kind']}: {mount['whole']}.{mount['field']} may hold a "
        f"{mount['target']}, mounted with {mount['mounted_by']}()"
        for mount in record.get("other_mounts", [])
    ]
    if not lines:
        return "Nothing: neither class can hold the other in any way."
    return "\n".join(lines)


def question_for(
    record: Dict[str, Any], images: Path, problems: Sequence[str] = ()
) -> List[Dict[str, Any]]:
    """
    Build the message asking what one pair is.

    :param record: The pair's record of ``relations.json``.
    :param images: The directory holding the pair's renders.
    :param problems: What was wrong with the answer to the same question, when this is
        another attempt at it.
    :return: The message, as :func:`model_client.ask` takes it.
    """
    legend = record["legend"]
    classes = record["classes"]
    described = "\n".join(
        f"{record[which]}, labelled \"{label}\", read as "
        f"{classes.get(label) or 'no class'}, painted {legend[record[which]]}"
        for which, label in zip(("one", "other"), record["classes"])
    )
    content = [
        model_client.text_part(
            f"## The two objects\n{described}\n"
            f"The faces both of them claim are painted {legend['both']}.\n\n"
            f"## What was measured\n{record['prompt_block']}\n\n"
            f"## What the ontology admits\n{admits(record)}"
        )
    ]

    named = {
        filename.rsplit("__", 1)[-1].split("_", 1)[0]: filename
        for filename in record["images"]
    }
    for kind, caption in CAPTIONS.items():
        if kind in named:
            content.append(model_client.text_part(caption))
            content.append(model_client.image_part(images / named[kind]))

    if problems:
        content.append(
            model_client.text_part(
                "Your previous answer could not be used:\n- "
                + "\n- ".join(problems)
                + "\nAnswer the same question again, correcting that."
            )
        )
    return content


def check(answer: Dict[str, Any], record: Dict[str, Any]) -> List[str]:
    """
    Say what is wrong with an answer, if anything.

    An answer naming a relation the ontology does not admit is one the mount would raise
    on, and an answer giving the contested faces to neither object leaves the mesh
    unsplittable, so both are worth catching here rather than three steps later.

    :param answer: What the model said.
    :param record: The pair's record of ``relations.json``.
    :return: One sentence per problem, empty when there are none.
    """
    problems = []
    names = (record["one"], record["other"])
    relation, whole, field = answer["relation"], answer["whole"], answer["field"]

    if relation not in RELATIONS:
        problems.append(f"{relation!r} is not one of {', '.join(RELATIONS)}")
    if answer["owner"] not in names:
        problems.append(
            f"the shared faces were given to {answer['owner']!r}, which is neither "
            f"{names[0]} nor {names[1]}"
        )

    if relation not in (PART, CONTAINS, SUPPORTS):
        return problems

    if whole not in names:
        problems.append(
            f"{whole!r} was named as the one that holds, which is neither "
            f"{names[0]} nor {names[1]}"
        )
        return problems

    holding = record["classes"].get(_label_of(record, whole))
    if relation == PART:
        fields = [
            entry["field"]
            for entry in record["admissible"]
            if entry["whole"] == holding
        ]
    else:
        fields = [
            mount["field"]
            for mount in record.get("other_mounts", [])
            if mount["whole"] == holding and mount["kind"] == relation
        ]

    if not fields:
        problems.append(
            f"the ontology admits no {relation} relation with {holding} holding"
        )
    elif field is None and len(fields) == 1:
        answer["field"] = fields[0]
    elif field not in fields:
        problems.append(
            f"{field!r} is not a field {holding} could hold it in; "
            f"the ontology admits {', '.join(fields)}"
        )
    return problems


def _label_of(record: Dict[str, Any], name: str) -> str:
    """
    :param record: The pair's record of ``relations.json``.
    :param name: The name of one of its segments.
    :return: The label that segment carries.
    """
    which = "one" if record["one"] == name else "other"
    return list(record["classes"])[0 if which == "one" else 1]


def read(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    :param response: What the model answered with.
    :return: The fields of its answer that are kept.
    """
    answered = model_client.parse_json_answer(model_client.answer_text(response))
    return {field: answered.get(field) for field in ANSWER_FIELDS}


def ask_about(
    record: Dict[str, Any],
    images: Path,
    answers: Path,
    model: str,
    reuse: bool,
    problems: Sequence[str] = (),
) -> Dict[str, Any]:
    """
    Put one pair to the model, or read back what it already said about it.

    :param record: The pair's record of ``relations.json``.
    :param images: The directory holding the pair's renders.
    :param answers: The directory raw responses are kept in.
    :param model: Which model to ask.
    :param reuse: Whether to read a kept response rather than ask again.
    :param problems: What was wrong with the previous answer, when there was one.
    :return: The response.
    """
    kept = answers / f"{record['one']}__{record['other']}.json"
    if reuse and kept.exists() and not problems:
        return json.loads(kept.read_text())

    response = model_client.ask(
        question_for(record, images, problems), SYSTEM_PROMPT, model=model
    )
    answers.mkdir(parents=True, exist_ok=True)
    kept.write_text(json.dumps(response, indent=2))
    return response


def build(arguments: argparse.Namespace) -> None:
    """
    Adjudicate the overlapping pairs of an evidence run.

    :param arguments: The command line arguments.
    """
    evidence = arguments.evidence_directory
    relations = json.loads((evidence / "relations.json").read_text())

    overlapping = [pair for pair in relations["pairs"] if pair["shared_faces"]]
    ready = [pair for pair in overlapping if pair.get("images")]
    if len(ready) < len(overlapping):
        print(
            f"{len(overlapping) - len(ready)} of {len(overlapping)} overlapping pairs "
            f"have no renders yet; run build_relation_evidence with "
            f"--adjudication-renders to make them."
        )
    ready = ready[: arguments.limit] if arguments.limit else ready

    print(f"asking {arguments.model} about {len(ready)} pairs ...")
    adjudicated = []
    for record in ready:
        problems: Sequence[str] = ()
        for attempt in range(1 + arguments.corrections):
            response = ask_about(
                record,
                evidence / "adjudications",
                evidence / "adjudication_answers",
                arguments.model,
                arguments.reuse_answers,
                problems,
            )
            try:
                answer = read(response)
            except model_client.ModelRefusedError as refusal:
                answer = {field: None for field in ANSWER_FIELDS}
                answer["problems"] = [str(refusal)]
            else:
                answer["problems"] = check(answer, record)
            problems = answer["problems"]
            if not problems:
                break
            if attempt + 1 <= arguments.corrections:
                print(f"  {record['one']} & {record['other']}: {problems[0]}, asking again ...")

        held = (
            f" {answer['whole']} holds it in {answer['field']}"
            if answer["relation"] in (PART, CONTAINS, SUPPORTS)
            else ""
        )
        print(
            f"  {record['one']:<16} & {record['other']:<16} "
            f"{str(answer['relation']):<12}{held}; faces -> {answer['owner']}"
        )
        for problem in answer["problems"]:
            print(f"      ! {problem}")
        adjudicated.append(
            {
                "one": record["one"],
                "other": record["other"],
                "shared_faces": record["shared_faces"],
                "status": record["status"],
                **answer,
            }
        )

    arguments.output.write_text(
        json.dumps(
            {"model": arguments.model, "scene": relations["scene"], "pairs": adjudicated},
            indent=2,
        )
    )

    counted = Counter(record["relation"] for record in adjudicated)
    troubled = [record for record in adjudicated if record["problems"]]
    print("\nwhat the pairs were answered as:")
    for relation, count in counted.most_common():
        print(f"  {str(relation):<14} {count}")
    print(f"{len(troubled)} with problems; written to {arguments.output}")


def main() -> None:
    """
    Adjudicate the overlaps of the evidence run named on the command line.
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
                        help="Ask about only the first so many pairs.")
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
