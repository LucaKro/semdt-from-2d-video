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
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Type

from experiments.warsaw.segment_relations import SegmentRelations, segment_evidence
from experiments.warsaw.world_loader import LabelSegment, WarsawWorldLoader
from semantic_digital_twin.semantic_annotations.part_whole import admissible_relations
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
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

    ..note:: What comes back is what is *admissible*, never what is the case: whether
        this cabinet holds this drawer is a question about the two objects, which no
        amount of reading the taxonomy answers.

    :param one_class: The class of one segment, or None when its label maps to none.
    :param other_class: The class of the other segment.
    :return: The admissible relations and what that leaves open.
    """
    if one_class is None or other_class is None:
        return {"status": CLASS_UNKNOWN, "admissible": []}

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
    return {"status": status, "admissible": admissible}


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
    vocabulary = (
        json.loads(arguments.vocabulary.read_text()) if arguments.vocabulary else {}
    )
    classes = {
        label: known.get(class_name) for label, class_name in vocabulary.items()
    }

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
                ),
                output / "exemplars",
                entry["label"],
            )
            entry["color"] = color.closest_css3_name()
            print(f"  {entry['label']}: {entry['exemplar']}")

    (output / "vocabulary_request.json").write_text(json.dumps(request, indent=2))

    if arguments.adjudication_renders:
        undecided = [
            record
            for record in records
            if record["status"] != RELATION_KNOWN and record["shared_faces"]
        ][: arguments.adjudication_renders]
        print(f"rendering {len(undecided)} adjudications ...")
        for record in undecided:
            one, other = segments[record["one"]], segments[record["other"]]
            highlights, legend = loader.pair_highlights(one, other)
            record["images"] = write_images(
                loader.render_region(
                    [one, other],
                    highlights,
                    viewpoints=arguments.viewpoints,
                    headless=arguments.headless,
                    choose_viewpoint=arguments.best_viewpoint,
                ),
                output / "adjudications",
                f"{record['one']}__{record['other']}",
            )
            record["legend"] = {
                name: color.closest_css3_name() for name, color in legend.items()
            }
            print(f"  {record['one']} & {record['other']}")
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
        "--adjudication-renders",
        type=int,
        default=0,
        help="Render this many overlapping pairs the ontology does not settle.",
    )
    parser.add_argument(
        "--viewpoints",
        nargs="+",
        default=["front_left", "back_right"],
        help="Which viewpoints to render (default: front_left back_right).",
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
        action="store_true",
        help="Keep only the viewpoint that shows the most of what is highlighted, "
        "measured by how much of the picture the highlight accounts for. Renders every "
        "viewpoint to decide, so it costs the renders it then discards.",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Render without opening a window."
    )
    build(parser.parse_args())


if __name__ == "__main__":
    main()
