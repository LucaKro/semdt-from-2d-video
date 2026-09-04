"""
Annotate a split scene's bodies and mount the parts into their wholes.

This is where everything the pipeline decided becomes a world: each body gets an
annotation of the class it was named as, and every pairing the adjudication settled is
carried out with the method the ontology says mounts it::

    python -m pipeline_scripts.annotate_and_mount pipeline_out_relations

The mount is done with ``add()`` rather than by filling constructor fields, because a
whole holds *several* drawers and a constructor slot takes one value: ``add`` routes a
part to the field its type matches and appends where the field holds many.

The classes a scene needs that the taxonomy does not have are generated first, into
``generated_classes.py``, and the ORM is rebuilt so the database knows them. That has to
happen before anything imports the classes, so the world is annotated in a second
process -- the same reason the persist step regenerates at import time.

Nothing here decides anything. Which body is which came from the split, what each one is
came from the classification, and which whole each part belongs to came from the
adjudication; if any of those is wrong, this writes it faithfully into the world.
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import semantic_digital_twin
from semantic_digital_twin.semantic_annotations.in_memory_builder import (
    SemanticAnnotationClassBuilder,
    SemanticAnnotationFilePaths,
)
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

TEMPLATE = "dataclass_template.py.jinja"
"""
What a generated class is written from.
"""


def wanted_classes(classifications: Dict[str, Any]) -> Dict[str, str]:
    """
    :param classifications: What the classification step wrote.
    :return: Per class name a body was given, the superclass proposed for it.
    """
    wanted: Dict[str, str] = {}
    for answer in classifications["bodies"].values():
        name = answer.get("class")
        if name and name not in wanted:
            wanted[name] = answer.get("superclass") or "SemanticAnnotation"
    return wanted


def generate_missing(wanted: Dict[str, str], known: Dict[str, Type]) -> List[str]:
    """
    Write the classes a scene needs that the taxonomy does not have.

    The file is written whole rather than appended to, since it is what the taxonomy
    reset empties between runs: a class one scene needed is not a class the next one
    starts with.

    :param wanted: Per class name, the superclass proposed for it.
    :param known: The taxonomy's classes by name.
    :return: The names that were generated.
    """
    builders, generated = [], []
    for name, superclass_name in sorted(wanted.items()):
        if name in known:
            continue
        superclass = known.get(superclass_name)
        if superclass is None:
            print(
                f"  {name}: {superclass_name} is not in the taxonomy, "
                f"deriving from SemanticAnnotation"
            )
            superclass = SemanticAnnotation
        builders.append(
            SemanticAnnotationClassBuilder(name, template_name=TEMPLATE).add_base(
                superclass
            )
        )
        generated.append(f"{name}({superclass.__name__})")

    if builders:
        SemanticAnnotationClassBuilder.write_classes_to_file(
            builders, Path(SemanticAnnotationFilePaths.GENERATED_CLASSES_FILE.value)
        )
    return generated


def regenerate_orm() -> Optional[str]:
    """
    Rebuild the ORM so the database knows the generated classes.

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


def mount(arguments: argparse.Namespace) -> None:
    """
    Annotate the bodies of a persisted world and mount its parts.

    Imports the ORM and the generated classes, so it runs only once they are written.

    :param arguments: The command line arguments.
    """
    from krrood.ormatic.data_access_objects.helper import to_dao
    from semantic_digital_twin.orm.ormatic_interface import Base, WorldMappingDAO
    from semantic_digital_twin.orm.utils import semantic_digital_twin_sessionmaker
    from sqlalchemy.orm import Session

    evidence = arguments.evidence_directory
    split = json.loads((evidence / "split.json").read_text())
    classifications = json.loads((evidence / "classifications.json").read_text())
    world_db_id = arguments.world_db_id or split.get("world_db_id")
    if world_db_id is None:
        raise SystemExit(
            "the split records no world in the database; run split_scene --persist"
        )

    engine = semantic_digital_twin_sessionmaker()().bind
    # The ORM was regenerated with the classes this scene needed, but a generated class
    # has no table until one is made for it, and a world holding one cannot be written.
    Base.metadata.create_all(bind=engine)
    with Session(engine) as session:
        stored = session.get(WorldMappingDAO, world_db_id)
        if stored is None:
            raise SystemExit(f"no world in the database with id {world_db_id}")
        world = stored.from_dao()
    print(f"world {world_db_id} read back, {len(world.bodies)} bodies")

    known = annotation_classes(SemanticAnnotation)
    bodies = {str(body.name.name): body for body in world.bodies}

    annotations: Dict[str, Any] = {}
    unknown: Counter = Counter()
    with world.modify_world():
        for name, answer in classifications["bodies"].items():
            class_name, body = answer.get("class"), bodies.get(name)
            if body is None or class_name not in known:
                unknown[class_name if body is not None else "no such body"] += 1
                continue
            # An abstract class cannot be instantiated, and one answer naming one should
            # cost that one body rather than every body after it.
            if inspect.isabstract(known[class_name]):
                unknown[f"{class_name} (abstract)"] += 1
                continue
            annotation = known[class_name](root=body, _world=world)
            # Registered as it is made, before anything is mounted into it. A mount
            # records an attribute update against the annotation it changes, and a
            # world replaying its modifications has to have that annotation already:
            # registering them all afterwards puts every update before its own subject
            # and the world cannot be read back at all.
            if annotation not in world.semantic_annotations:
                world.add_semantic_annotation(annotation)
            annotations[name] = annotation
    print(f"{len(annotations)} bodies annotated")
    for missing, count in unknown.most_common():
        print(f"  {count} left alone: {missing}")

    # add() moves the part's branch under the whole, so it modifies the world model and
    # has to be told so.
    mounted, refused = 0, []
    with world.modify_world():
        for pairing in split["pairings"]:
            whole = annotations.get(pairing["whole"])
            part = annotations.get(pairing["part"])
            if whole is None or part is None:
                refused.append((pairing, "one end has no annotation"))
                continue
            try:
                whole.add(part, field_name=pairing["field"] or "")
            except Exception as failure:
                refused.append(
                    (pairing, f"{type(failure).__name__}: {str(failure).splitlines()[0]}")
                )
            else:
                mounted += 1

    print(f"\n{mounted} of {len(split['pairings'])} pairings mounted")
    for pairing, why in refused:
        print(f"  {pairing['whole']} <- {pairing['part']}: {why}")

    # Written as a world of its own rather than merged onto the one it was read from.
    # Merging serialises a world afresh and grafts it onto the stored graph, and what
    # came back could not be read at all: replaying its modifications reached an entity
    # that was not there. Two rows also keep the split world as it was, which is what
    # the classification and the pairings were decided against.
    with Session(engine) as session:
        stored = to_dao(world)
        session.add(stored)
        session.commit()
        annotated_id = stored.database_id
    print(f"written as world {annotated_id}, annotated, beside the split world {world_db_id}")

    record = json.loads((evidence / "split.json").read_text())
    record["annotated_world_db_id"] = annotated_id
    (evidence / "split.json").write_text(json.dumps(record, indent=2))


def build(arguments: argparse.Namespace) -> None:
    """
    Generate what the taxonomy lacks, rebuild the ORM, then annotate and mount.

    :param arguments: The command line arguments.
    """
    evidence = arguments.evidence_directory
    classifications = json.loads((evidence / "classifications.json").read_text())
    wanted = wanted_classes(classifications)
    known = annotation_classes(SemanticAnnotation)
    print(f"{len(wanted)} classes over {len(classifications['bodies'])} bodies")

    generated = generate_missing(wanted, known)
    if generated:
        print(f"generated {len(generated)}: {', '.join(generated)}")
        print("regenerating the ORM ...")
        failure = regenerate_orm()
        if failure:
            raise SystemExit(f"the ORM could not be regenerated:\n{failure}")
    else:
        print("every class the scene needs is already in the taxonomy")

    # The classes were written after this process imported the taxonomy, so the world is
    # annotated in one that starts after they exist.
    print("\nannotating in a new interpreter ...")
    finished = subprocess.run(
        [sys.executable, "-m", "pipeline_scripts.annotate_and_mount",
         str(evidence), "--mount-only"]
        + (["--world-db-id", str(arguments.world_db_id)] if arguments.world_db_id else []),
    )
    if finished.returncode != 0:
        raise SystemExit("annotating failed")


def main() -> None:
    """
    Annotate and mount the split scene named on the command line.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "evidence_directory",
        type=Path,
        help="The directory the split and classification runs wrote.",
    )
    parser.add_argument(
        "--world-db-id",
        type=int,
        default=None,
        help="Which world to annotate, by default the one the split recorded.",
    )
    parser.add_argument(
        "--mount-only",
        action="store_true",
        help="Skip generating classes and rebuilding the ORM, which the first pass "
        "already did. Used when this script calls itself.",
    )
    arguments = parser.parse_args()
    if arguments.mount_only:
        mount(arguments)
        return
    build(arguments)


if __name__ == "__main__":
    main()
