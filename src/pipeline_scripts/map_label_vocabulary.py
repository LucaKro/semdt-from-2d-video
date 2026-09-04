"""
Ask a model which class of the taxonomy each of a scene's labels means.

A scanned scene labels its faces with words a human annotator chose: ``cabinet``,
``kitchen_island``, ``jar``. Those are not classes. Some have a class of the same name
that means something else, some have no class at all, and nothing in the mesh says
which is which -- so this is the one question in the pipeline that has to be asked
before any conflict between labels can be resolved, and it is asked once per label
rather than once per object.

It reads what :mod:`pipeline_scripts.build_relation_evidence` wrote::

    python -m pipeline_scripts.map_label_vocabulary pipeline_out_relations

and writes ``vocabulary.json`` beside it, which the same evidence run then takes as
``--vocabulary`` to say what the ontology admits for every pair of overlapping labels.

An answer is either a class of the taxonomy or a new class named by the superclass and
mixins it is composed of, and those decide what it can hold: a ``KitchenIsland``
composed with ``HasDrawers`` admits the drawers overlapping it as parts, one without it
admits nothing. Every answer is checked here -- names against the taxonomy, compositions
by building the class -- so what is written out is known to be usable rather than merely
well spelled.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
    compose_class,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from pipeline_scripts import model_client

SYSTEM_PROMPT = """\
You name the segmentation labels of a scanned room in a robot's ontology.

You are given one label, pictures of one object carrying it, and the ontology as JSON.
Say which class of the ontology that label means. The label is a word an annotator chose:
it may match no class, and a class of the same name may mean something else. Judge from
the pictures what the objects are.

If no class fits, propose a new one by naming the superclass it derives from and any of
the ontology's part_whole_mixins to compose it from. Choose those by what the object can
hold, because they decide it: a class composed with HasDrawers can hold drawers as its
parts and one without it cannot, and the parts of these objects are mounted by exactly
that. The labels measured to share the pictured object's faces are the candidates for
its parts, so a class that cannot hold them will never hold them. Do not propose a new
class where an existing one fits.

Rules:
- "class" is a name from classes[], or the name you propose when "is_new_class" is true.
  If the name you give is not in classes[], "is_new_class" is true.
- A class marked "mixin" is a base to build with, never an answer: it says what
  something can hold, not what it is. Give one as a superclass or a mixin, never as the
  class.
- "superclass" is always a name from classes[], and a class is never its own superclass.
- "mixins" are names from part_whole_mixins[], and [] when none apply.
- Whatever the class should be able to hold as its parts has to be admitted by its
  superclass or by one of its mixins, so read the labels measured to meet the pictured
  object and give the class the mixins that let it hold them.
- Answer for the label as a whole, not only for the one object pictured.
- Answer with "class": null only if the label names nothing the ontology should hold.
- If your reason names a superclass and mixins to build from, then you are proposing a
  new class: give it its own name and set "is_new_class" to true.

Answer with JSON and nothing else, in one of these three shapes.

A class that is already in the taxonomy:
{"class": "<a name from classes[]>", "is_new_class": false,
 "confidence": 0.0, "reason": "one sentence"}

A class you propose:
{"class": "<the name you give it>", "is_new_class": true,
 "superclass": "<a name from classes[]>", "mixins": ["<names from part_whole_mixins[]>"],
 "confidence": 0.0, "reason": "one sentence"}

A label the ontology should hold nothing for:
{"class": null, "is_new_class": false, "confidence": 0.0, "reason": "one sentence"}"""

CAPTIONS = {
    "context": "Picture 1 -- where in the room it is, painted {color}.",
    "plain": "Picture 2 -- the object alone, in the colors it was scanned in.",
    "closeup": "Picture 3 -- the same object alone, painted {color}, which is exactly "
    "the faces the label covers.",
}
"""
What each of an exemplar's three renders shows, in the order they are shown in.

The order is the order the question is answered in: where it stands, what it looks like,
and then which faces are actually being asked about.
"""

ANSWER_FIELDS = ("class", "is_new_class", "superclass", "mixins", "confidence", "reason")
"""
What is kept of an answer, so a model's extra fields do not reach the file.
"""


def taxonomy_of(evidence_directory: Path) -> Dict[str, Any]:
    """
    :param evidence_directory: Where the evidence run wrote its files.
    :return: The taxonomy it exported.
    """
    return json.loads((evidence_directory / "taxonomy.json").read_text())


def meetings(relations: Dict[str, Any], exemplar: str) -> str:
    """
    Say what the pictured object meets, as the scan measures it.

    Which labels cover the same surface is the evidence a composition needs and a picture
    does not carry: an island whose faces are also labelled ``drawer`` has to be given a
    class that can hold drawers, or nothing will ever be mounted into it. It is reported
    as a measurement and said to be one, since sharing a surface is not yet a relation.

    :param relations: What ``relations.json`` holds.
    :param exemplar: The name of the segment being pictured.
    :return: The measurements as the text a model reads, empty when it meets nothing.
    """
    labels = {segment["name"]: segment["class"] for segment in relations["segments"]}
    overlapping: Counter = Counter()
    touching: Counter = Counter()
    for pair in relations["pairs"]:
        if exemplar not in (pair["one"], pair["other"]):
            continue
        other = pair["other"] if pair["one"] == exemplar else pair["one"]
        if pair["shared_faces"]:
            overlapping[labels[other]] += 1
        elif pair["touching_edges"]:
            touching[labels[other]] += 1

    def counted(counts: Counter) -> str:
        return ", ".join(
            f"{label} ({count})" for label, count in counts.most_common()
        )

    lines = []
    if overlapping:
        lines.append(
            f"Objects of these labels are labelled onto some of the same faces: "
            f"{counted(overlapping)}."
        )
    if touching:
        lines.append(
            f"Objects of these labels touch it along edges without sharing faces: "
            f"{counted(touching)}."
        )
    if not lines:
        return ""
    lines.append(
        "That is measurement only: sharing a surface says these labels cover the same "
        "geometry, not which of them holds the other."
    )
    return "## What it meets, measured on the scan\n" + "\n".join(lines)


def question_for(
    label: Dict[str, Any],
    labels: List[str],
    taxonomy: Dict[str, Any],
    images: Path,
    meets: str = "",
) -> List[Dict[str, Any]]:
    """
    Build the message asking what one label means.

    :param label: The label's entry of ``vocabulary_request.json``.
    :param labels: Every label of the scene, since which of them exist alongside a label
        says what it was left to mean: a room that labels handles separately does not
        mean them by ``drawer``.
    :param taxonomy: The exported taxonomy.
    :param images: The directory holding the exemplar renders.
    :param meets: What the pictured object meets, as :func:`meetings` reports it.
    :return: The message, as :func:`model_client.ask` takes it.
    """
    named = {name.split("__")[-1].split("_", 1)[0]: name for name in label["images"]}
    content = [
        model_client.text_part(
            f"## The ontology\n{json.dumps(taxonomy)}\n\n"
            f"## The label\n"
            f'The label is "{label["label"]}". The room carries {label["instances"]} '
            f"objects labelled with it.\n"
            f"The room's labels are: {', '.join(labels)}.\n\n"
            f"{meets}\n\n"
            f"## The pictures\n"
            f'They show one of them, "{label["exemplar"]}", chosen as the one whose '
            f"faces are least shared with other labels."
        )
    ]
    for kind, caption in CAPTIONS.items():
        filename = named.get(kind)
        if filename is None:
            continue
        content.append(model_client.text_part(caption.format(color=label["color"])))
        content.append(model_client.image_part(images / filename))
    return content


def ask_about(
    label: Dict[str, Any],
    labels: List[str],
    taxonomy: Dict[str, Any],
    images: Path,
    meets: str,
    answers: Path,
    model: str,
    reuse: bool,
    problems: Sequence[str] = (),
) -> Dict[str, Any]:
    """
    Put one label to the model, or read back what it already said about it.

    :param label: The label's entry of ``vocabulary_request.json``.
    :param labels: Every label of the scene.
    :param taxonomy: The exported taxonomy.
    :param images: The directory holding the exemplar renders.
    :param meets: What the pictured object meets, as :func:`meetings` reports it.
    :param answers: The directory raw responses are kept in.
    :param model: Which model to ask.
    :param reuse: Whether to read a kept response rather than ask again.
    :param problems: What was wrong with the answer to the same question, when this is
        another attempt at it. An answer naming a class that is not in the taxonomy is
        worth nothing, and saying so is cheaper than either dropping the label or
        letting a person correct it by hand.
    :return: The response.
    """
    kept = answers / f"{label['label']}.json"
    if reuse and kept.exists() and not problems:
        return json.loads(kept.read_text())

    content = question_for(label, labels, taxonomy, images, meets)
    if problems:
        content.append(
            model_client.text_part(
                "Your previous answer could not be used:\n- "
                + "\n- ".join(problems)
                + "\nAnswer the same question again, correcting that."
            )
        )
    response = model_client.ask(content, SYSTEM_PROMPT, model=model)
    answers.mkdir(parents=True, exist_ok=True)
    kept.write_text(json.dumps(response, indent=2))
    return response


def check(
    answer: Dict[str, Any],
    known: Dict[str, type],
    mixins: List[str],
    building_blocks: List[str],
) -> List[str]:
    """
    Say what is wrong with an answer, if anything.

    An answer is only worth as much as it is usable: a class name that is not in the
    taxonomy, or a composition that cannot be built, would come back as an unmapped
    label two steps later, where it would look like the scene's fault rather than the
    answer's.

    :param answer: What the model said, as :func:`read` kept it.
    :param known: The taxonomy's classes by name.
    :param mixins: The names a new class may be composed from.
    :param building_blocks: The names of the classes that exist to be built with, which
        the taxonomy derives concrete classes from -- a floor is a HasSupportingSurface
        -- but which name nothing standing in a room themselves.
    :return: One sentence per problem, empty when there are none.
    """
    problems = []
    name, superclass = answer.get("class"), answer.get("superclass")
    if name is None:
        return problems

    if name in building_blocks:
        problems.append(
            f"{name} is a mixin, so it says what a class can hold rather than what it "
            f"is; derive a class from it instead of answering with it"
        )
    if answer.get("is_new_class"):
        if name in known:
            problems.append(f"{name} is proposed as new but is already in the taxonomy")
        if superclass not in known:
            problems.append(f"the superclass {superclass!r} is not in the taxonomy")
        for mixin in answer.get("mixins") or []:
            if mixin not in building_blocks:
                problems.append(f"{mixin!r} is not one of the taxonomy's mixins")
        if not problems:
            try:
                compose_class(
                    name,
                    known[superclass],
                    [known[mixin] for mixin in answer.get("mixins") or []],
                )
            except TypeError as failure:
                problems.append(f"the composition cannot be built: {failure}")
    elif name not in known:
        problems.append(f"{name} is not in the taxonomy and was not proposed as new")
    return problems


def read(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    :param response: What the model answered with.
    :return: The fields of its answer that are kept, with the shapes they are read as.
    """
    answered = model_client.parse_json_answer(model_client.answer_text(response))
    answer = {field: answered.get(field) for field in ANSWER_FIELDS}
    answer["is_new_class"] = bool(answer["is_new_class"])
    answer["mixins"] = list(answer["mixins"] or [])
    return answer


def report(label: str, answer: Dict[str, Any]) -> None:
    """
    Print what was answered about one label.

    :param label: The label that was asked about.
    :param answer: What came back, as :func:`read` and :func:`check` leave it.
    """
    name = answer["class"] or "-- nothing"
    composition = (
        f" ({', '.join([answer['superclass']] + answer['mixins'])})"
        if answer["is_new_class"]
        else ""
    )
    new = " [new]" if answer["is_new_class"] else ""
    print(f"  {label:<18} -> {name}{composition}{new}")
    for problem in answer["problems"]:
        print(f"      ! {problem}")


def build(arguments: argparse.Namespace) -> None:
    """
    Map every label of an evidence run to a class.

    :param arguments: The command line arguments.
    """
    evidence = arguments.evidence_directory
    request = json.loads((evidence / "vocabulary_request.json").read_text())
    taxonomy = taxonomy_of(evidence)
    known = annotation_classes(SemanticAnnotation)
    mixins = [mixin["name"] for mixin in taxonomy["part_whole_mixins"]]
    building_blocks = [
        node["name"] for node in taxonomy["classes"] if node.get("mixin")
    ]
    relations = json.loads((evidence / "relations.json").read_text())

    entries = [
        entry
        for entry in request["labels"]
        if not arguments.labels or entry["label"] in arguments.labels
    ]
    labels = [entry["label"] for entry in request["labels"]]
    print(f"asking {arguments.model} about {len(entries)} of {len(labels)} labels ...")

    # Asking about a few labels refines a mapping rather than replacing it, so what was
    # answered about the others stays.
    answered: Dict[str, Any] = (
        json.loads(arguments.output.read_text())["labels"]
        if arguments.output.exists()
        else {}
    )
    for entry in entries:
        problems: Sequence[str] = ()
        for attempt in range(1 + arguments.corrections):
            response = ask_about(
                entry,
                labels,
                taxonomy,
                evidence / "exemplars",
                meetings(relations, entry["exemplar"]),
                evidence / "vocabulary_answers",
                arguments.model,
                arguments.reuse_answers,
                problems,
            )
            try:
                answer = read(response)
            except model_client.ModelRefusedError as refusal:
                answer = {field: None for field in ANSWER_FIELDS}
                answer.update(is_new_class=False, mixins=[], problems=[str(refusal)])
            else:
                answer["problems"] = check(answer, known, mixins, building_blocks)
            problems = answer["problems"]
            if not problems:
                break
            if attempt + 1 <= arguments.corrections:
                print(f"  {entry['label']}: {problems[0]}, asking again ...")

        answer["exemplar"] = entry["exemplar"]
        answered[entry["label"]] = answer
        report(entry["label"], answer)

    arguments.output.write_text(
        json.dumps(
            {"model": arguments.model, "scene": request.get("scene"), "labels": answered},
            indent=2,
        )
    )

    mapped = [answer for answer in answered.values() if answer["class"]]
    proposed = [answer for answer in mapped if answer["is_new_class"]]
    troubled = [answer for answer in answered.values() if answer["problems"]]
    print(
        f"\n{len(mapped)} of {len(answered)} labels mapped, "
        f"{len(proposed)} of them to new classes, {len(troubled)} with problems"
    )
    print(f"written to {arguments.output}")


def main() -> None:
    """
    Map the labels of the evidence run named on the command line.
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
        help="Where to write the mapping, by default vocabulary.json beside the "
        "evidence it was built from.",
    )
    parser.add_argument(
        "--model",
        default=model_client.DEFAULT_MODEL,
        help="Which model to ask.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=[],
        help="Ask about only these labels, rather than all of them.",
    )
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
        help="Read back the kept responses instead of asking again, which re-reads a "
        "run without spending anything on it.",
    )
    arguments = parser.parse_args()
    if arguments.output is None:
        arguments.output = arguments.evidence_directory / "vocabulary.json"
    build(arguments)


if __name__ == "__main__":
    main()
