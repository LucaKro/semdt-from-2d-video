"""
Measure how a Warsaw scene's labelled objects meet, and say what the ontology makes of it.

Splitting a scene into bodies means deciding what to do where its labels overlap: the
same faces can be a cabinet and its door, a drawer and the handle on it. This gathers
what those decisions need and decides nothing itself:

- the geometry, measured (shared faces, edges touched along, distance apart),
- the taxonomy, in the form that shows what each class can be composed of,
- per pair, which part-whole relations the ontology admits between their classes.

    python -m pipeline_scripts.build_relation_evidence \
        dataset/kitchenlab_new_mesh_agreement_dataset \
        --output-dir pipeline_out_relations --headless

Labels are not classes -- a scene labelling something ``kitchen_island`` says nothing
about which class that is, and some labels have no class at all. That mapping is a
question for a model, and ``vocabulary_request.json`` is what it is asked with. Given the
answer back as ``--vocabulary``, the same run says what the ontology admits for every
pair.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from experiments.warsaw.segment_relations import (
    SegmentRelations,
    claimant_groups,
    segment_evidence,
)
from experiments.warsaw.world_loader import (
    VIEWPOINT_ALONE,
    VIEWPOINT_IN_ROOM,
    LabelSegment,
    WarsawWorldLoader,
)
from semantic_digital_twin.semantic_annotations.part_whole import admissible_relations
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    admissible_mounts,
    annotation_classes,
    compose_class,
    describe_class,
    export_taxonomy,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

CLASS_UNKNOWN = "class-unknown"
"""
Neither label has been mapped to a class yet, so the ontology has nothing to say.
"""

NO_LEGAL_RELATION = "no-legal-relation"
"""
The classes cannot hold one another as a part, so an overlap is something else.
"""

RELATION_KNOWN = "relation-known"
"""
Exactly one part-whole relation is admissible, so only the pair itself is in question.
"""

RELATION_AMBIGUOUS = "relation-ambiguous"
"""
Several relations are admissible, so which field a mount would use is in question too.
"""


def ontology_view(
    one_class: Optional[Type], other_class: Optional[Type]
) -> Dict[str, object]:
    """
    Ask the ontology what two classes may be to one another.

    The status is decided by the part-whole channel alone, because that is the one that
    discriminates: ``contains`` is admissible between almost any two annotations, since
    ``IsStorageSpace.objects`` accepts anything with a root body, so letting it decide
    would report a relation for every pair in the room. The other channels are reported
    beside it instead, since a mug on a counter and a jar in a box do stand in one, and
    an adjudication told only about parts would force "part" onto them.

    ..note:: What comes back is what is *admissible*, never what is the case: whether
        this cabinet holds this drawer is a question about the two objects, which no
        amount of reading the taxonomy answers.

    :param one_class: The class of one segment, or None when its label maps to none.
    :param other_class: The class of the other segment.
    :return: The admissible relations and what that leaves open.
    """
    if one_class is None or other_class is None:
        return {"status": CLASS_UNKNOWN, "admissible": [], "other_mounts": []}

    relations = admissible_relations(one_class, other_class)
    admissible = [
        {
            "whole": relation.whole.__name__,
            "part": relation.part.__name__,
            "field": relation.field_name,
            "many": relation.holds_many,
            "removes_geometry": relation.removes_part_geometry_from_whole,
        }
        for relation in relations
    ]
    if not admissible:
        status = NO_LEGAL_RELATION
    elif len(admissible) == 1:
        status = RELATION_KNOWN
    else:
        status = RELATION_AMBIGUOUS

    other_mounts = [
        {
            "kind": relation.kind,
            "whole": whole.__name__,
            "field": relation.field_name,
            "target": relation.target,
            "mounted_by": relation.mounted_by,
        }
        for whole, relation in admissible_mounts(one_class, other_class)
        if relation.kind != "part"
    ]
    return {"status": status, "admissible": admissible, "other_mounts": other_mounts}


def exemplars(relations: SegmentRelations) -> Dict[str, str]:
    """
    Pick the instance of each label that shows the label best.

    The one with the most surface no other segment claims is the one a viewer can judge
    without judging an overlap at the same time. Ranking by the *share* instead picks
    slivers, which are wholly unclaimed precisely because they are fragments: on this
    scene it offered a strip of 545 faces as the example of a cabinet while a whole
    cabinet front of 960 stood beside it, equally unclaimed.

    :param relations: The measured scene.
    :return: Per label, the name of the segment standing for it.
    """
    best: Dict[str, str] = {}
    for descriptor in relations.descriptors.values():
        standing = best.get(descriptor.class_name)
        if standing is None or (
            descriptor.exclusive_area
            > relations.descriptors[standing].exclusive_area
        ):
            best[descriptor.class_name] = descriptor.name
    return best


def neighbourhood(
    relations: SegmentRelations,
    segments: Dict[str, LabelSegment],
    names: List[str],
) -> List[LabelSegment]:
    """
    Gather what a context view has to show for its subject to be recognisable.

    A picture of the whole room is useless for anything small -- a mug covers none of it.
    Framed on the mug together with the segments measured to stand nearest it, the same
    mug covers a tenth of the picture, and what surrounds it is what says which mug it
    is. The neighbours come from the same measurement the rest of the evidence does, so
    the neighbourhood is not a radius anyone chose.

    :param relations: The measured scene.
    :param segments: The scene's segments by name.
    :param names: The segments the view is about.
    :return: Those segments together with the ones measured to stand near them.
    """
    wanted = set(names)
    for name in names:
        for pair in relations.pairs_of(name):
            wanted.update((pair.one, pair.other))
    return [segments[name] for name in sorted(wanted)]


def classes_of_labels(
    vocabulary: Dict[str, object], known: Dict[str, Type]
) -> Dict[str, Optional[Type]]:
    """
    Turn a mapping of labels onto classes into the classes it names.

    A label mapped to a class of the taxonomy is looked up. One mapped to a class that
    was proposed rather than found is *composed* from the superclass and mixins the
    proposal names, because the class does not exist yet and it is exactly those that
    decide what it admits: composed with ``HasDrawers`` it can hold the drawers
    overlapping it, without it it can hold nothing, and the pairs below turn on that.

    A flat ``{label: class name}`` is read too, so a mapping written by hand to try
    something out needs no more than that.

    :param vocabulary: What :mod:`pipeline_scripts.map_label_vocabulary` wrote, or a
        flat mapping.
    :param known: The taxonomy's classes by name.
    :return: Per label, the class it stands for, or None where it stands for none.
    """
    labels = vocabulary.get("labels", vocabulary)
    classes: Dict[str, Optional[Type]] = {}
    for label, answer in labels.items():
        if answer is None or isinstance(answer, str):
            classes[label] = known.get(answer) if answer else None
            continue

        name = answer.get("class")
        if not name or answer.get("problems"):
            classes[label] = None
            if name:
                print(f"  {label}: leaving unmapped, {answer['problems'][0]}")
        elif not answer.get("is_new_class"):
            classes[label] = known.get(name)
        else:
            classes[label] = compose_class(
                name,
                known[answer["superclass"]],
                [known[mixin] for mixin in answer.get("mixins") or []],
            )
    return classes


def group_highlights(
    segments: List[LabelSegment], contested: np.ndarray
) -> Tuple[List[Tuple[Color, np.ndarray]], Dict[str, Color]]:
    """
    Color a set of objects so that what they disagree about can be seen.

    Each gets a color of its own, and the faces all of them claim get one more, painted
    last. A picture without that last color shows the contested faces as belonging to
    whichever object was painted over them, which is the very thing in question.

    :param segments: The objects to color.
    :param contested: The faces they all claim.
    :return: What to paint, and what each color stands for.
    """
    colors = Color.distinct_colors(len(segments) + 1)
    highlights = [
        (color, segment.faces) for color, segment in zip(colors, segments)
    ]
    highlights.append((colors[-1], contested))
    legend = {str(segment.name): color for color, segment in zip(colors, segments)}
    legend["contested"] = colors[-1]
    return highlights, legend


def ontology_around(
    names: Sequence[str],
    labels: Dict[str, str],
    classes: Dict[str, Optional[Type]],
) -> Dict[str, object]:
    """
    Say what the taxonomy holds about a handful of objects.

    The questions about a set of objects were asked with the pictures and the
    measurements alone, and the ontology already knows things that bear on them: that a
    cabinet can hold a drawer says the drawer is the finer of the two, which is exactly
    what "whose surface is this" turns on. What is reported is the slice about these
    classes rather than the whole taxonomy, since a question about three objects is not
    helped by a hundred and thirty-nine classes.

    :param names: The segments in question.
    :param labels: Per segment, the label it carries.
    :param classes: Per label, the class it was read as.
    :return: Per segment its class, each class written out, and what the ontology admits
        between each pair of them.
    """
    read_as = {name: classes.get(labels[name]) for name in names}
    admits = []
    for one, other in combinations(names, 2):
        for whole, relation in admissible_mounts(read_as[one], read_as[other]) if (
            read_as[one] is not None and read_as[other] is not None
        ) else []:
            admits.append(
                f"{whole.__name__}.{relation.field_name} may hold a "
                f"{relation.target} ({relation.kind}, mounted with "
                f"{relation.mounted_by}())"
            )
    return {
        "read_as": {
            name: getattr(annotation_class, "__name__", None)
            for name, annotation_class in read_as.items()
        },
        "classes": [
            describe_class(annotation_class)
            for annotation_class in dict.fromkeys(
                one for one in read_as.values() if one is not None
            )
        ],
        # The same mount turns up once per pair that could use it, and almost every
        # class has objects -> HasRootBody, so without this the list reads as though
        # containment were being urged six times over.
        "admits": list(dict.fromkeys(admits)),
    }


def measured_of(
    names: Sequence[str], relations: SegmentRelations
) -> Dict[str, Dict[str, object]]:
    """
    :param names: The segments in question.
    :param relations: The measured scene.
    :return: Per segment, what was measured of it on its own.
    """
    return {
        name: {
            "faces": int(relations.descriptors[name].faces),
            "area": round(relations.descriptors[name].area, 3),
            "height": round(relations.descriptors[name].height, 2),
            "pieces": int(relations.descriptors[name].components),
        }
        for name in names
    }


def open_questions(
    loader: WarsawWorldLoader,
    relations: SegmentRelations,
    records: List[Dict[str, object]],
    classes: Dict[str, Optional[Type]],
) -> Dict[str, List[Dict[str, object]]]:
    """
    Work out what is actually left to decide, and how few questions it takes.

    Three things shrink the pile. A face belongs to one object, so the question is asked
    once per set of claimants rather than once per pair of them. A group whose every
    internal pair the ontology settled as a part-whole relation is not a question at all,
    since the part keeps the surface it is made of. And what is left repeats: a door and a
    window sharing a pane is one question however many glazed doors the room has, so the
    groups are gathered by the classes in them and asked once per pattern.

    Which whole a part belongs to is a separate question, and only where a part overlaps
    more than one candidate: a drawer that meets exactly one cabinet has nothing to
    choose between.

    :param loader: The loaded scene.
    :param relations: The measured scene.
    :param records: The pairs, as they were written out.
    :param classes: Per label, the class it was read as.
    :return: The ownership questions, the membership questions, and the groups that need
        neither.
    """
    segments = loader.label_segments
    groups = claimant_groups(
        [segment.faces for segment in segments],
        [str(segment.name) for segment in segments],
        len(loader.scene_mesh.faces),
    )
    status = {
        tuple(sorted((record["one"], record["other"]))): record["status"]
        for record in records
    }
    labels = {name: descriptor.class_name for name, descriptor in relations.descriptors.items()}

    settled, patterns = [], defaultdict(list)
    for group in groups:
        inside = [
            status.get(tuple(sorted(pair)))
            for pair in combinations(group.names, 2)
        ]
        if all(answer == RELATION_KNOWN for answer in inside):
            settled.append(group.to_json())
        else:
            patterns[tuple(sorted(labels[name] for name in group.names))].append(group)

    ownership = []
    for pattern, members in sorted(patterns.items(), key=lambda one: -len(one[1])):
        exemplar = max(members, key=lambda group: len(group.faces))
        ownership.append(
            {
                "name": "__".join(pattern),
                "kind": "ownership",
                "pattern": list(pattern),
                "shown": list(exemplar.names),
                "faces": [int(face) for face in exemplar.faces],
                "covers": [group.to_json() for group in members],
                "contested_faces": sum(len(group.faces) for group in members),
                # How much of each claimant the contested faces are. Without it the
                # picture is all a reader has, and a picture cannot be read when one
                # claimant is twenty times the size of the others: the island label
                # covers the whole block including its drawers, so a drawer front reads
                # as a patch of detail on the island rather than as the drawer.
                "shares": {
                    name: {
                        "faces": int(relations.descriptors[name].faces),
                        "contested_share": round(
                            len(exemplar.faces) / relations.descriptors[name].faces, 4
                        ),
                    }
                    for name in exemplar.names
                },
                "exemplar_faces": int(len(exemplar.faces)),
                "ontology": ontology_around(exemplar.names, labels, classes),
                "measured": measured_of(exemplar.names, relations),
                "images": [],
            }
        )

    # A part is attached to the whole it belongs to, so a candidate has to share faces
    # with it or touch it along an edge. Everything else the measurement reached is
    # merely nearby, and offering it as an alternative is offering a wrong answer: it
    # trebles the questions and none of what it adds could be right.
    candidates = defaultdict(dict)
    for record in records:
        if record["status"] != RELATION_KNOWN:
            continue
        if not record["shared_faces"] and not record["touching_edges"]:
            continue
        admitted = record["admissible"][0]
        whole, part = record["one"], record["other"]
        if labels[whole] not in record["classes"] or record["classes"][labels[whole]] != admitted["whole"]:
            whole, part = part, whole
        candidates[part][whole] = {
            "field": admitted["field"],
            "shared_faces": record["shared_faces"],
            "touching_edges": record["touching_edges"],
            "distance": record["distance"],
        }

    membership = [
        {
            "name": part,
            "kind": "membership",
            "part": part,
            "shown": [part] + sorted(wholes),
            "faces": [],
            "candidates": {name: how for name, how in sorted(wholes.items())},
            "ontology": ontology_around([part] + sorted(wholes), labels, classes),
            "measured": measured_of([part] + sorted(wholes), relations),
            "images": [],
        }
        for part, wholes in sorted(candidates.items())
        if len(wholes) > 1
    ]

    forced = [
        {"part": part, "whole": next(iter(wholes)), **next(iter(wholes.values()))}
        for part, wholes in sorted(candidates.items())
        if len(wholes) == 1
    ]
    return {
        "ownership": ownership,
        "membership": membership,
        "settled": settled,
        "forced": forced,
    }


def carry_over_renders(path: Path, fresh: List[Dict[str, object]]) -> None:
    """
    Keep the renders a previous run made for the same questions.

    A run that re-measures without re-rendering would otherwise write out questions with
    no pictures, which reads as though the pictures were never made and quietly costs the
    next step its evidence. A render is kept only where the question is the same one --
    same name, same objects shown -- and the file is still on disk.

    :param path: The file the previous run wrote.
    :param fresh: The questions this run built, amended in place.
    """
    if not path.exists():
        return
    before = json.loads(path.read_text())
    kept = {
        (question["name"], tuple(question["shown"])): question
        for section in ("ownership", "membership")
        for question in before.get(section, [])
        if question.get("images")
    }
    directory = path.parent / "questions"
    for question in fresh:
        earlier = kept.get((question["name"], tuple(question["shown"])))
        if earlier is None:
            continue
        if all((directory / name).exists() for name in earlier["images"]):
            question["images"] = earlier["images"]
            question["legend"] = earlier.get("legend", {})


def write_images(images: Dict[str, bytes], directory: Path, prefix: str) -> List[str]:
    """
    :param images: The renders to write, by viewpoint.
    :param directory: Where to write them.
    :param prefix: What to name them after.
    :return: The names they were written as.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, image in images.items():
        filename = f"{prefix}__{name}.png"
        (directory / filename).write_bytes(image)
        written.append(filename)
    return sorted(written)


def build(arguments: argparse.Namespace) -> None:
    """
    Gather the evidence for the scene named on the command line.

    :param arguments: The command line arguments.
    """
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)

    loader = WarsawWorldLoader(
        input_directory=arguments.scene_directory,
        render_resolution=tuple(arguments.resolution),
    )
    segments: Dict[str, LabelSegment] = {
        str(segment.name): segment for segment in loader.label_segments
    }
    print(f"{len(segments)} segments; measuring how they meet ...")
    relations = segment_evidence(loader, nearest=arguments.nearest)
    print(f"{len(relations.pairs)} pairs stand in some measurable relation")

    taxonomy = export_taxonomy(SemanticAnnotation, output / "taxonomy.json")
    known = annotation_classes(SemanticAnnotation)
    classes = classes_of_labels(
        json.loads(arguments.vocabulary.read_text()) if arguments.vocabulary else {},
        known,
    )

    records = []
    for pair in relations.pairs:
        one, other = relations.descriptors[pair.one], relations.descriptors[pair.other]
        view = ontology_view(
            classes.get(one.class_name), classes.get(other.class_name)
        )
        records.append(
            {
                **pair.to_json(),
                "classes": {
                    one.class_name: getattr(
                        classes.get(one.class_name), "__name__", None
                    ),
                    other.class_name: getattr(
                        classes.get(other.class_name), "__name__", None
                    ),
                },
                **view,
                "prompt_block": pair.as_prompt_block(relations.descriptors),
            }
        )

    (output / "relations.json").write_text(
        json.dumps(
            {
                "scene": str(loader.scene.mesh_path),
                "segments": relations.to_json()["segments"],
                "pairs": records,
            },
            indent=2,
        )
    )

    standing_for = exemplars(relations)
    request = {
        "scene": str(loader.scene.mesh_path),
        "question": (
            "Each label below names objects in a scanned room. Say which class of the "
            "taxonomy each label is, or propose a new class by naming a superclass and "
            "any mixins it should be composed of. The taxonomy is in taxonomy.json; its "
            "part_whole_mixins list what a new class can be given."
        ),
        "labels": [
            {
                "label": label,
                "instances": sum(
                    1
                    for descriptor in relations.descriptors.values()
                    if descriptor.class_name == label
                ),
                "exemplar": name,
                "exemplar_faces": relations.descriptors[name].faces,
                "exemplar_exclusive_share": round(
                    relations.descriptors[name].exclusive_share, 4
                ),
                "exemplar_exclusive_area": round(
                    relations.descriptors[name].exclusive_area, 4
                ),
                "images": [],
            }
            for label, name in sorted(standing_for.items())
        ],
    }

    if arguments.exemplar_renders:
        print(f"rendering {len(request['labels'])} exemplars ...")
        for entry in request["labels"]:
            segment = segments[entry["exemplar"]]
            color = Color.distinct_colors(1)[0]
            entry["images"] = write_images(
                loader.render_region(
                    [segment],
                    [(color, segment.faces)],
                    viewpoints=arguments.viewpoints,
                    headless=arguments.headless,
                    choose_viewpoint=arguments.best_viewpoint,
                    context_segments=neighbourhood(
                        relations, segments, [entry["exemplar"]]
                    ),
                ),
                output / "exemplars",
                entry["label"],
            )
            entry["color"] = color.closest_css3_name()
            print(f"  {entry['label']}: {entry['exemplar']}")

    (output / "vocabulary_request.json").write_text(json.dumps(request, indent=2))

    questions = open_questions(loader, relations, records, classes)
    print(
        f"\n{len(questions['ownership'])} class patterns and "
        f"{len(questions['membership'])} memberships are open; "
        f"{len(questions['settled'])} groups the ontology settles"
    )

    if arguments.question_renders:
        asked = (
            questions["ownership"][: arguments.question_renders]
            + questions["membership"][: arguments.question_renders]
        )
        print(f"rendering {len(asked)} questions ...")
        for question in asked:
            shown = [segments[name] for name in question["shown"]]
            highlights, legend = group_highlights(
                shown, np.asarray(question["faces"], dtype=np.int64)
            )
            question["images"] = write_images(
                loader.render_region(
                    shown,
                    highlights,
                    viewpoints=arguments.viewpoints,
                    headless=arguments.headless,
                    choose_viewpoint=arguments.best_viewpoint,
                    context_segments=neighbourhood(
                        relations, segments, question["shown"]
                    ),
                ),
                output / "questions",
                question["name"],
            )
            question["legend"] = {
                name: color.closest_css3_name() for name, color in legend.items()
            }
            print(f"  {question['name']}")

    carry_over_renders(
        output / "questions.json", questions["ownership"] + questions["membership"]
    )
    for question in questions["ownership"] + questions["membership"]:
        question.pop("faces", None)
    (output / "questions.json").write_text(
        json.dumps({"scene": str(loader.scene.mesh_path), **questions}, indent=2)
    )

    counted = Counter(record["status"] for record in records)
    print(f"\ntaxonomy: {len(taxonomy['classes'])} classes, "
          f"{len(taxonomy['part_whole_mixins'])} mixins to compose new ones from")
    print("what the ontology makes of the pairs:")
    for status, count in counted.most_common():
        print(f"  {status:<20} {count}")
    print(f"\nwritten to {output}")


def main() -> None:
    """
    Gather the evidence for the scene named on the command line.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "scene_directory",
        type=Path,
        help="Directory holding the scene's labelled mesh.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Where to write the evidence."
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=None,
        help="JSON mapping each label to the name of its class, as answered from "
        "vocabulary_request.json. Without it the ontology has nothing to say.",
    )
    parser.add_argument(
        "--nearest",
        type=int,
        default=5,
        help="How many nearest neighbours each segment reports (default: 5).",
    )
    parser.add_argument(
        "--exemplar-renders",
        action="store_true",
        help="Render one exemplar per label for the vocabulary question.",
    )
    parser.add_argument(
        "--question-renders",
        type=int,
        default=0,
        help="Render at most this many of each kind of open question. One picture per "
        "class pattern and per contested membership, not one per overlapping pair: "
        "there are two hundred of those and a fifth as many questions in them.",
    )
    parser.add_argument(
        "--viewpoints",
        nargs="+",
        default=None,
        help="Which viewpoints to render. Defaults to two of the four when every "
        "render is kept, and to all four with --best-viewpoint, which renders them "
        "only to decide between them and keeps one.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=(1024, 768),
        metavar=("WIDTH", "HEIGHT"),
        help="Size of the rendered images (default: 1024 768).",
    )
    parser.add_argument(
        "--best-viewpoint",
        choices=(VIEWPOINT_IN_ROOM, VIEWPOINT_ALONE),
        default=None,
        help="Keep only the viewpoint that shows the most of what is highlighted, "
        "measured by how much of the picture the highlight accounts for. Renders every "
        "viewpoint to decide, so it costs the renders it then discards: 'in-room' draws "
        "the whole scene each time and takes minutes per region, 'alone' draws only the "
        "highlighted geometry and takes seconds, at the price of not counting what "
        "stands in front of it.",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Render without opening a window."
    )
    arguments = parser.parse_args()
    if arguments.viewpoints is None:
        arguments.viewpoints = (
            None if arguments.best_viewpoint else ["front_left", "back_right"]
        )
    build(arguments)


if __name__ == "__main__":
    main()
