"""
Cut a scene's single body into one body per labelled object.

Everything the cut needs was decided by the steps before it: which class each label
means, who owns the faces several labels claim, and which whole each part belongs to.
This applies those decisions and builds the world::

    python -m pipeline_scripts.split_scene \
        dataset/kitchenlab_new_mesh_agreement_dataset pipeline_out_relations

The world is built even where objects end up with nothing, and every such loss is
reported beside the answer that caused it: an ownership answer is given once per class
pattern, so one wrong answer empties every object of that kind at once, and a report that
merely said ten cabinets are missing would not say why. Nothing is persisted here, so a
wrong answer costs a re-run rather than a bad world in the database.

The pairings are carried out of the split rather than measured again from the bodies,
because the overlap that says a handle is on *this* drawer is gone the moment the faces
stop being shared.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


from experiments.warsaw.scene_split import (
    Ownership,
    Pairing,
    SplitFaces,
    exclusive_faces,
    owner_by_ontology,
    pairings,
    split_world,
)
from experiments.warsaw.segment_relations import ClaimantGroup, claimant_groups
from experiments.warsaw.world_loader import WarsawWorldLoader
from semantic_digital_twin.semantic_annotations.taxonomy_export import annotation_classes
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from pipeline_scripts.build_relation_evidence import classes_of_labels


def resolve_owners(
    groups: List[ClaimantGroup],
    adjudications: Dict[str, object],
    labels: Dict[str, str],
    classes: Dict[str, Optional[type]],
) -> Tuple[List[Ownership], List[ClaimantGroup]]:
    """
    Work out who each set of contested faces belongs to.

    A set the ontology settled is answered from the taxonomy; the rest are answered by the
    class pattern they belong to, since that is the grain the question was asked at.

    :param groups: The sets of faces and who claims them.
    :param adjudications: What the adjudication wrote.
    :param labels: Per segment, the label it carries.
    :param classes: Per label, the class it was read as.
    :return: The ownerships, and the groups no answer reached.
    """
    settled = {tuple(group["claimants"]) for group in adjudications["settled"]}
    answers = {
        tuple(answer["pattern"]): answer["owner"]
        for answer in adjudications["answered"]
        if answer["kind"] == "ownership"
    }

    ownerships, unreached = [], []
    for group in groups:
        by_class = {name: classes.get(labels[name]) for name in group.names}
        if tuple(group.names) in settled:
            owner = owner_by_ontology(group.names, by_class)
            settled_here = True
        else:
            wanted = answers.get(tuple(sorted(labels[name] for name in group.names)))
            claiming = [name for name in group.names if labels[name] == wanted]
            owner = claiming[0] if len(claiming) == 1 else None
            settled_here = False

        if owner is None:
            unreached.append(group)
            continue
        ownerships.append(
            Ownership(
                names=group.names,
                owner=owner,
                faces=group.faces,
                settled_by_ontology=settled_here,
            )
        )
    return ownerships, unreached


def named_pairings(adjudications: Dict[str, object]) -> List[Pairing]:
    """
    :param adjudications: What the adjudication wrote.
    :return: Every mount its answers named, before the split drops any.
    """
    fields = {
        (forced["part"], forced["whole"]): forced["field"]
        for forced in adjudications["forced"]
    }
    carried = [
        Pairing(whole=forced["whole"], part=forced["part"], field_name=forced["field"])
        for forced in adjudications["forced"]
    ]
    carried += [
        Pairing(
            whole=answer["whole"],
            part=answer["part"],
            field_name=fields.get((answer["part"], answer["whole"]), ""),
        )
        for answer in adjudications["answered"]
        if answer["kind"] == "membership" and answer["whole"]
    ]
    return carried


def report(
    split: SplitFaces,
    ownerships: List[Ownership],
    labels: Dict[str, str],
) -> None:
    """
    Say what the split did, and what it cost.

    :param split: What the split left.
    :param ownerships: Who each set of faces was given to.
    :param labels: Per segment, the label it carries.
    """
    print(
        f"{len(split.faces)} bodies, "
        f"{sum(len(faces) for faces in split.faces.values())} faces between them"
    )
    settled = sum(1 for ownership in ownerships if ownership.settled_by_ontology)
    print(
        f"{len(ownerships)} sets of contested faces given away: "
        f"{settled} by the ontology, {len(ownerships) - settled} by an answer"
    )

    if split.contested:
        print(f"\n{len(split.contested)} faces are still claimed twice:")
        for face, names in list(split.contested.items())[:5]:
            print(f"  face {face}: {' & '.join(names)}")

    if not split.emptied:
        return

    # An ownership answer is given once per class pattern, so one wrong answer empties
    # every object of that kind. Grouping the losses by who took them says which answer
    # to look at rather than which objects went missing.
    took = defaultdict(set)
    for name in split.emptied:
        for owner in split.lost_to.get(name, {}):
            took[labels[owner]].add(name)
    print(f"\n{len(split.emptied)} segments lost every face they had:")
    for owner, lost in sorted(took.items(), key=lambda one: -len(one[1])):
        counted = Counter(labels[name] for name in lost)
        wording = ", ".join(f"{count} {label}" for label, count in counted.most_common())
        print(f"  everything went to a {owner}: {wording}")
        print(f"      {', '.join(sorted(lost))}")


def build(arguments: argparse.Namespace) -> None:
    """
    Split the scene named on the command line.

    :param arguments: The command line arguments.
    """
    evidence = arguments.evidence_directory
    adjudications = json.loads((evidence / "adjudications.json").read_text())
    vocabulary = json.loads((evidence / "vocabulary.json").read_text())

    loader = WarsawWorldLoader(input_directory=arguments.scene_directory)
    segments = loader.label_segments
    labels = {str(segment.name): segment.class_name for segment in segments}
    faces = {str(segment.name): segment.faces for segment in segments}
    print(f"{len(segments)} segments over {len(loader.scene_mesh.faces)} faces")

    groups = claimant_groups(
        [segment.faces for segment in segments],
        [str(segment.name) for segment in segments],
        len(loader.scene_mesh.faces),
    )
    classes = classes_of_labels(vocabulary, annotation_classes(SemanticAnnotation))
    ownerships, unreached = resolve_owners(groups, adjudications, labels, classes)
    if unreached:
        print(f"\n{len(unreached)} sets of faces no answer reached:")
        for group in unreached[:5]:
            print(f"  {' & '.join(group.names)} ({len(group.faces)} faces)")

    split = exclusive_faces(faces, ownerships)
    report(split, ownerships, labels)

    carried = pairings(named_pairings(adjudications), split)
    print(f"\n{len(carried)} pairings carried past the split")

    print("\nbuilding the bodies ...")
    world = split_world(
        loader.scene.mesh,
        split.faces,
        WarsawWorldLoader.SOURCE_TO_WORLD,
        directory=arguments.mesh_directory,
    )
    print(f"the world holds {len(world.bodies)} bodies")

    # The faces each body is made of, keyed by the name it carries everywhere else. The
    # world built here dies with the process, so without this the split would have to be
    # derived again from the answers to be used, and a later answer would silently give
    # a different partition than the one that was reported.
    faces_path = arguments.output.with_name(f"{arguments.output.stem}_faces.npz")
    np.savez_compressed(faces_path, **{name: kept for name, kept in split.faces.items()})
    print(f"the bodies' faces written to {faces_path}")

    arguments.output.write_text(
        json.dumps(
            {
                "scene": str(loader.scene.mesh_path),
                "bodies": {
                    name: {"faces": int(len(kept)), "label": labels[name]}
                    for name, kept in sorted(split.faces.items())
                },
                "emptied": {
                    name: split.lost_to.get(name, {}) for name in split.emptied
                },
                "still_contested": len(split.contested),
                "pairings": [pairing.to_json() for pairing in carried],
            },
            indent=2,
        )
    )
    print(f"written to {arguments.output}")


def main() -> None:
    """
    Split the scene named on the command line.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "scene_directory", type=Path, help="Directory holding the scene's mesh."
    )
    parser.add_argument(
        "evidence_directory",
        type=Path,
        help="The directory the evidence and adjudication runs wrote.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write what the split did, by default split.json beside the "
        "evidence it was made from.",
    )
    parser.add_argument(
        "--mesh-directory",
        type=Path,
        default=None,
        help="Where to write the bodies' meshes. Without it they live in a place that "
        "is removed when this process ends, which will not do for a world to be kept.",
    )
    arguments = parser.parse_args()
    if arguments.output is None:
        arguments.output = arguments.evidence_directory / "split.json"
    build(arguments)


if __name__ == "__main__":
    main()
