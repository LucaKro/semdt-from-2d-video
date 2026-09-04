"""
Ask whether the taxonomy, rather than the room, is what is wrong -- and mend it.

When objects of two classes are measured to share a surface and no field of either
admits the other as a structural part, one of two things is true: those objects want a
class of their own, or the class they were read as is missing a mixin. This step puts
the second to a model, one class pair at a time, and carries out what it accepts::

    python -m pipeline_scripts.amend_taxonomy pipeline_out_relations
    python -m pipeline_scripts.amend_taxonomy pipeline_out_relations --apply

Without ``--apply`` nothing is written but ``taxonomy_amendments.json``, which is the
proposal to read before anything changes.

An amendment is not a pipeline output. It edits the class in the SDK's own source and
regenerates the ORM, so it holds for every room, every world already in the database and
everything built on the taxonomy afterwards -- which is why the model is asked about the
class rather than about the objects, and why applying it is a separate word on the
command line.

Which mixin would grant a relation is never guessed: it is found by composing the class
with each mixin and asking the same question a mount asks. The model decides only
whether the class should have it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import semantic_digital_twin
from semantic_digital_twin.semantic_annotations.taxonomy_amendment import (
    CannotAmendClass,
    SourceAmendment,
    amend_class_source,
    granting_mixins,
)
from semantic_digital_twin.semantic_annotations.part_whole import part_whole_fields
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
    build_taxonomy,
    relations_of,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from pipeline_scripts import model_client
from pipeline_scripts.build_relation_evidence import classes_of_labels

SYSTEM_PROMPT = """\
You decide whether a class in a robot's ontology is missing a structural part.

Objects of two classes were measured to share a surface in a scanned room, and no field
of either class admits the other as a part. You are told the mixin that would grant it.

Adding it edits the class itself. It holds for every room, every world already stored,
and everything built on the ontology afterwards -- so judge the class, not the objects in
the pictures. Say yes only if a thing of the first class can, in general, have a thing of
the second as one of its structural parts.

Say no if:
- the part can already be reached through something the class holds, in which case the
  paths are listed for you: a field of its own would let it mount beside the thing it is
  really a part of,
- the objects would be better served by a class of their own that has the mixin,
- the overlap is something other than a part: resting on it, stored inside it, or two
  labels covering the same surface,
- the part more plausibly belongs to something else the first class already holds.

Answer with JSON and nothing else:
{"amend": true, "confidence": 0.0, "reason": "one sentence"}"""


@dataclass
class Candidate:
    """
    One mixin a class could be given, and what was measured that raised the question.
    """

    whole: str
    """
    The name of the class that would hold the part.
    """

    mixin: str
    """
    The name of the mixin that would let it.
    """

    part: str
    """
    The name of the class that would be held.
    """

    whole_labels: List[str] = field(default_factory=list)
    """
    The scene's labels that were read as the holding class.
    """

    part_labels: List[str] = field(default_factory=list)
    """
    The scene's labels that were read as the part.
    """

    pairs: int = 0
    """
    How many measured pairs of overlapping objects raised it.
    """

    shared_faces: int = 0
    """
    How many faces those pairs share in total.
    """

    def key(self) -> Tuple[str, str, str]:
        """
        :return: What makes this candidate the same question as another.
        """
        return self.whole, self.mixin, self.part

    def to_json(self) -> Dict[str, Any]:
        """
        :return: The candidate as JSON-ready data.
        """
        return {
            "whole": self.whole,
            "mixin": self.mixin,
            "part": self.part,
            "whole_labels": sorted(self.whole_labels),
            "part_labels": sorted(self.part_labels),
            "measured_pairs": self.pairs,
            "shared_faces": self.shared_faces,
        }


def candidates(
    relations: Dict[str, Any],
    classes: Dict[str, Optional[Type]],
    known: Dict[str, Type],
    mixins: List[Type],
) -> List[Candidate]:
    """
    Find the amendments the scene's measurements raise.

    Only overlapping pairs are considered, and only classes the taxonomy has written
    down: a class proposed for this scene was given its mixins when it was proposed, and
    amending it would mean amending a proposal.

    Whether a relation is already admissible is asked of the classes rather than read
    from the status the measurements were written with, which was decided by whichever
    vocabulary that run was given and may not be the one being read now.

    :param relations: What ``relations.json`` holds.
    :param classes: Per label, the class it was read as.
    :param known: The taxonomy's classes by name, before anything was composed.
    :param mixins: The mixins a class can be given.
    :return: One candidate per class, mixin and part, with what raised it gathered.
    """
    gathered: Dict[Tuple[str, str, str], Candidate] = {}
    for pair in relations["pairs"]:
        if not pair["shared_faces"]:
            continue
        labels = list(pair["classes"])
        for one, other in (labels, labels[::-1]):
            whole, part = classes.get(one), classes.get(other)
            if whole is None or part is None:
                continue
            if known.get(whole.__name__) is not whole:
                continue
            for mixin in granting_mixins(whole, part, mixins):
                candidate = Candidate(whole.__name__, mixin.__name__, part.__name__)
                candidate = gathered.setdefault(candidate.key(), candidate)
                if one not in candidate.whole_labels:
                    candidate.whole_labels.append(one)
                if other not in candidate.part_labels:
                    candidate.part_labels.append(other)
                candidate.pairs += 1
                candidate.shared_faces += pair["shared_faces"]
    return sorted(gathered.values(), key=lambda one: -one.pairs)


def describe(annotation_class: Type) -> str:
    """
    :param annotation_class: The class to describe.
    :return: Its declaration and what it can hold, as a model reads it.
    """
    bases = ", ".join(base.__name__ for base in annotation_class.__bases__)
    lines = [f"{annotation_class.__name__}({bases})"]
    for relation in relations_of(annotation_class):
        many = " (many)" if relation.holds_many else ""
        lines.append(
            f"  {relation.kind} {relation.field_name} -> {relation.target}{many}"
        )
    return "\n".join(lines)


def paths_to(whole: Type, part: Type, maximum_depth: int = 3) -> List[str]:
    """
    Report how a part can already be reached from a class through what it holds.

    This is what makes a proposal redundant rather than wrong: a cabinet holds doors and
    a door holds a handle, so a cabinet reaches a handle without declaring one, and
    giving it a field of its own would let a handle mount onto the carcass and skip the
    door it is actually on. Without these paths the question cannot be answered, since
    the reason to say no is one relation further away than the class itself.

    :param whole: The class to search from.
    :param part: The class to reach.
    :param maximum_depth: How many relations a path may be long.
    :return: One rendered path per way of reaching it, empty when there is none.
    """
    found: List[str] = []
    frontier = [(whole, whole.__name__, {whole})]
    for _ in range(maximum_depth):
        onwards = []
        for current, rendered, seen in frontier:
            for relation in part_whole_fields(current):
                target = relation.part
                if not isinstance(target, type):
                    continue
                path = f"{rendered} -> {relation.field_name} -> {target.__name__}"
                if issubclass(part, target):
                    found.append(path)
                elif target not in seen:
                    onwards.append((target, path, seen | {target}))
        frontier = onwards
    return found


def question_for(
    candidate: Candidate,
    known: Dict[str, Type],
    taxonomy: Dict[str, Any],
    images: Path,
    request: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build the message asking whether a class should be given a mixin.

    :param candidate: The amendment in question.
    :param known: The taxonomy's classes by name.
    :param taxonomy: The exported taxonomy.
    :param images: The directory holding the exemplar renders.
    :param request: What ``vocabulary_request.json`` holds, for the exemplar of a label.
    :return: The message, as :func:`model_client.ask` takes it.
    """
    introduces = next(
        mixin["introduces"]
        for mixin in taxonomy["part_whole_mixins"]
        if mixin["name"] == candidate.mixin
    )
    granted = ", ".join(
        f"{relation['field']} -> {relation['target']}"
        for relation in introduces
        if relation["kind"] == "part"
    )
    # Said only when there is something to say. Reporting that no path was found puts
    # the absence of a reason to refuse where a reason to accept would go, and it gets
    # read as one: it turned four refusals into acceptances, one of them a Handle that
    # would hold Doors.
    reachable = paths_to(known[candidate.whole], known[candidate.part])
    already = (
        f"## How a {candidate.part} can already be reached\n"
        + "\n".join(reachable)
        + "\nA part reachable through something the class already holds needs no field "
        "of its own: mounting it directly onto the class would put it beside the thing "
        "it is really a part of.\n\n"
        if reachable
        else ""
    )
    content = [
        model_client.text_part(
            f"## The proposal\n"
            f"Give {candidate.whole} the mixin {candidate.mixin}, which introduces: "
            f"{granted}.\n\n"
            f"## The class as it stands\n{describe(known[candidate.whole])}\n\n"
            f"## The part\n{describe(known[candidate.part])}\n\n"
            f"{already}"
            f"## What was measured\n"
            f"In one scanned room, objects labelled "
            f"{', '.join(sorted(candidate.whole_labels))} were read as "
            f"{candidate.whole}, and objects labelled "
            f"{', '.join(sorted(candidate.part_labels))} as {candidate.part}. "
            f"They share faces over {candidate.pairs} measured pairs, "
            f"{candidate.shared_faces} shared faces in all."
        )
    ]

    entries = {entry["label"]: entry for entry in request["labels"]}
    for role, labels in (
        (candidate.whole, candidate.whole_labels),
        (candidate.part, candidate.part_labels),
    ):
        label = sorted(labels)[0]
        entry = entries.get(label)
        if entry is None:
            continue
        for filename in entry["images"]:
            if "__plain_" in filename:
                continue
            content.append(
                model_client.text_part(
                    f'An object labelled "{label}", read as {role}, painted '
                    f'{entry["color"]}.'
                )
            )
            content.append(model_client.image_part(images / filename))
    return content


def judge(
    candidate: Candidate,
    known: Dict[str, Type],
    taxonomy: Dict[str, Any],
    images: Path,
    request: Dict[str, Any],
    answers: Path,
    model: str,
    reuse: bool,
) -> Dict[str, Any]:
    """
    Put one amendment to the model.

    :param candidate: The amendment in question.
    :param known: The taxonomy's classes by name.
    :param taxonomy: The exported taxonomy.
    :param images: The directory holding the exemplar renders.
    :param request: What ``vocabulary_request.json`` holds.
    :param answers: The directory raw responses are kept in.
    :param model: Which model to ask.
    :param reuse: Whether to read a kept response rather than ask again.
    :return: What it answered, as ``amend``, ``confidence`` and ``reason``.
    """
    kept = answers / f"{'-'.join(candidate.key())}.json"
    if reuse and kept.exists():
        response = json.loads(kept.read_text())
    else:
        response = model_client.ask(
            question_for(candidate, known, taxonomy, images, request),
            SYSTEM_PROMPT,
            model=model,
        )
        answers.mkdir(parents=True, exist_ok=True)
        kept.write_text(json.dumps(response, indent=2))

    try:
        answered = model_client.parse_json_answer(model_client.answer_text(response))
    except model_client.ModelRefusedError as refusal:
        return {"amend": False, "confidence": None, "reason": str(refusal)}
    return {
        "amend": bool(answered.get("amend")),
        "confidence": answered.get("confidence"),
        "reason": answered.get("reason"),
    }


def verify(applied: List[SourceAmendment]) -> Optional[str]:
    """
    Check that the amended classes really hold what they were amended to hold.

    Asked in a new interpreter, since the classes in this one were built before the
    edit and a dataclass collects its fields once.

    :param applied: The amendments that were written.
    :return: What went wrong, or None when every amendment took effect.
    """
    checks = [
        [amendment.annotation_class.__name__, amendment.mixin.__name__]
        for amendment in applied
    ]
    program = (
        "import json, sys\n"
        "from semantic_digital_twin.semantic_annotations.taxonomy_export import "
        "annotation_classes\n"
        "from semantic_digital_twin.world_description.world_entity import "
        "SemanticAnnotation\n"
        "known = annotation_classes(SemanticAnnotation)\n"
        "for name, mixin in json.loads(sys.argv[1]):\n"
        "    assert issubclass(known[name], known[mixin]), (name, mixin)\n"
    )
    finished = subprocess.run(
        [sys.executable, "-c", program, json.dumps(checks)],
        capture_output=True,
        text=True,
    )
    return None if finished.returncode == 0 else finished.stderr.strip()


def regenerate_orm() -> Optional[str]:
    """
    Rebuild the ORM from the amended classes.

    A field the ORM does not know about cannot be written to the database, so an
    amendment that stops here is one that reads as done and is not.

    :return: What went wrong, or None when it was rebuilt.
    """
    script = (
        Path(semantic_digital_twin.__file__).parent.parent.parent
        / "scripts"
        / "generate_orm.py"
    )
    if not script.exists():
        return f"the ORM generator is not at {script}"
    finished = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True
    )
    return None if finished.returncode == 0 else finished.stderr.strip()[-2000:]


def apply(accepted: List[Tuple[Candidate, SourceAmendment]]) -> None:
    """
    Write the accepted amendments and rebuild the ORM, undoing them if either fails.

    :param accepted: The amendments to carry out.
    """
    written: List[SourceAmendment] = []
    for candidate, amendment in accepted:
        amendment.apply()
        written.append(amendment)
        print(f"  {amendment.path.name}:{amendment.line_number}  {amendment.after}")

    def undo(what: str) -> None:
        for amendment in reversed(written):
            amendment.reverted().apply()
        raise SystemExit(f"{what}\nthe amendments were undone; nothing was changed.")

    failure = verify(written)
    if failure:
        undo(f"the amended classes did not come back as amended:\n{failure}")

    print("regenerating the ORM ...")
    failure = regenerate_orm()
    if failure:
        undo(f"the ORM could not be regenerated:\n{failure}")
    print("the ORM was regenerated; the taxonomy is amended.")


def build(arguments: argparse.Namespace) -> None:
    """
    Judge, and carry out, the amendments a scene's measurements raise.

    :param arguments: The command line arguments.
    """
    evidence = arguments.evidence_directory
    relations = json.loads((evidence / "relations.json").read_text())
    request = json.loads((evidence / "vocabulary_request.json").read_text())
    vocabulary = json.loads((evidence / "vocabulary.json").read_text())

    known = annotation_classes(SemanticAnnotation)
    taxonomy = build_taxonomy(SemanticAnnotation)
    mixins = [known[mixin["name"]] for mixin in taxonomy["part_whole_mixins"]]
    classes = classes_of_labels(vocabulary, known)

    raised = candidates(relations, classes, known, mixins)
    if arguments.only:
        wanted = set(arguments.only)
        raised = [
            candidate
            for candidate in raised
            if f"{candidate.whole}+{candidate.mixin}" in wanted
        ]
    print(f"{len(raised)} amendments are raised by what was measured:")
    for candidate in raised:
        print(
            f"  {candidate.whole} + {candidate.mixin} to hold {candidate.part}"
            f"  ({candidate.pairs} pairs, "
            f"{'/'.join(sorted(candidate.whole_labels))} & "
            f"{'/'.join(sorted(candidate.part_labels))})"
        )

    judged = []
    accepted: List[Tuple[Candidate, SourceAmendment]] = []
    print(f"\nasking {arguments.model} about each ...")
    for candidate in raised:
        answer = judge(
            candidate,
            known,
            taxonomy,
            evidence / "exemplars",
            request,
            evidence / "amendment_answers",
            arguments.model,
            arguments.reuse_answers,
        )
        record = {**candidate.to_json(), **answer}
        print(
            f"  {'yes' if answer['amend'] else 'no ':<3} "
            f"{candidate.whole} + {candidate.mixin}: {answer['reason']}"
        )

        if answer["amend"]:
            try:
                amendment = amend_class_source(known[candidate.whole], known[candidate.mixin])
            except CannotAmendClass as refusal:
                record["blocked"] = str(refusal)
                print(f"      ! {refusal}")
            else:
                if amendment is None:
                    record["blocked"] = "the class already has that mixin"
                else:
                    record["edit"] = {
                        "file": str(amendment.path),
                        "line": amendment.line_number,
                        "before": amendment.before,
                        "after": amendment.after,
                    }
                    accepted.append((candidate, amendment))
        judged.append(record)

    (evidence / "taxonomy_amendments.json").write_text(json.dumps(judged, indent=2))
    print(
        f"\n{len(accepted)} of {len(judged)} accepted; "
        f"written to {evidence / 'taxonomy_amendments.json'}"
    )

    if not accepted:
        return
    if not arguments.apply:
        print("run again with --apply to carry them out.")
        return

    print("\napplying:")
    apply(accepted)


def main() -> None:
    """
    Judge the amendments raised by the evidence run named on the command line.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "evidence_directory",
        type=Path,
        help="The directory build_relation_evidence and map_label_vocabulary wrote.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Edit the classes and regenerate the ORM, rather than only proposing it.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Consider only these amendments, named as <Class>+<Mixin>, rather than "
        "every one the measurements raise.",
    )
    parser.add_argument(
        "--model",
        default=model_client.DEFAULT_MODEL,
        help="Which model to ask.",
    )
    parser.add_argument(
        "--reuse-answers",
        action="store_true",
        help="Read back the kept responses instead of asking again.",
    )
    build(parser.parse_args())


if __name__ == "__main__":
    main()
