#!/usr/bin/env python3
"""
Persist semantic annotations to the database.

Regenerates the ORM (once) to include any new generated classes, then
loads the world from the database, re-instantiates annotations from
pending_annotations JSON files, and persists the updated world.

Called by run_batch.py after all rooms have been refined so the
expensive ORM regeneration only happens a single time.

Usage:
    python scripts/persist_annotations.py \\
        --world-db-id 7 \\
        --dataset hm3d \\
        batch_output/00800/room1/pending_annotations.json \\
        batch_output/00800/room2/pending_annotations.json ...
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import is_dataclass
from pathlib import Path

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def _is_uuid(value: str) -> bool:
    return bool(_UUID_RE.match(value))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CRAM_DIR = REPO_ROOT.parent / "cognitive_robot_abstract_machine"
SDT_DIR = CRAM_DIR / "semantic_digital_twin" / "src" / "semantic_digital_twin"
ORM_FILE = SDT_DIR / "orm" / "ormatic_interface.py"

# ---------------------------------------------------------------------------
# Step 1: Pre-load generated classes BEFORE any ORM / SDK imports
# ---------------------------------------------------------------------------
from semantic_digital_twin.semantic_annotations.in_memory_builder import (
    SemanticAnnotationFilePaths,
)
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Body,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootKinematicStructureEntity,
)


def _preload_generated_classes():
    generated_file = Path(SemanticAnnotationFilePaths.GENERATED_CLASSES_FILE.value)
    if not generated_file.exists():
        return
    import importlib.util

    module_name = "semantic_digital_twin.semantic_annotations.generated_classes"
    spec = importlib.util.spec_from_file_location(module_name, generated_file)
    generated_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = generated_module
    spec.loader.exec_module(generated_module)
    import semantic_digital_twin.semantic_annotations as parent_module

    setattr(parent_module, "generated_classes", generated_module)

    import semantic_digital_twin.semantic_annotations.in_memory_builder as imb

    for name in dir(generated_module):
        obj = getattr(generated_module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, SemanticAnnotation)
            and obj is not SemanticAnnotation
        ):
            setattr(imb, name, obj)


_preload_generated_classes()

# ---------------------------------------------------------------------------
# Step 2: Regenerate ORM to include generated classes
# ---------------------------------------------------------------------------
# These imports mirror generate_orm.py from the SDK, adding generated classes.
import semantic_digital_twin.adapters.procthor.procthor_resolver
import semantic_digital_twin.callbacks.callback
import semantic_digital_twin.orm.model
import semantic_digital_twin.reasoning.predicates
import semantic_digital_twin.robots.robot_parts
import semantic_digital_twin.robots.hsrb
import semantic_digital_twin.robots.pr2
import semantic_digital_twin.semantic_annotations.semantic_annotations
import semantic_digital_twin.world
import semantic_digital_twin.world_description.degree_of_freedom
import semantic_digital_twin.world_description.geometry
import semantic_digital_twin.world_description.shape_collection
import semantic_digital_twin.world_description.world_entity

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.type_dict import TypeDict
from krrood.ormatic.utils import classes_of_module
from krrood.utils import recursive_subclasses
from semantic_digital_twin.mixin import SimulatorAdditionalProperty
from semantic_digital_twin.orm.model import AlternativeMapping
from semantic_digital_twin.reasoning.predicates import ContainsType
from semantic_digital_twin.semantic_annotations.position_descriptions import (
    SemanticDirection,
)
from semantic_digital_twin.spatial_computations.forward_kinematics import (
    ForwardKinematicsManager,
)
from semantic_digital_twin.world import (
    ResetStateContextManager,
    WorldModelUpdateContextManager,
    WorldModelManager,
)
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    HasUpdateState,
)
from semantic_digital_twin.world_description.world_modification import (
    AttributeUpdateModification,
)
from krrood.adapters.json_serializer import JSONAttributeDiff
import trimesh


def regenerate_orm():
    """Regenerate the ORM including generated classes."""
    logging.info("Regenerating ORM...")

    all_classes = set(
        classes_of_module(semantic_digital_twin.world_description.world_entity)
    )
    all_classes |= set(classes_of_module(semantic_digital_twin.world_description.geometry))
    all_classes |= set(
        classes_of_module(semantic_digital_twin.world_description.shape_collection)
    )
    all_classes |= set(classes_of_module(semantic_digital_twin.world))
    all_classes |= set(
        classes_of_module(semantic_digital_twin.datastructures.prefixed_name)
    )
    all_classes |= set(classes_of_module(semantic_digital_twin.datastructures.joint_state))
    all_classes |= set(
        classes_of_module(semantic_digital_twin.world_description.connections)
    )
    all_classes |= set(
        classes_of_module(semantic_digital_twin.semantic_annotations.semantic_annotations)
    )
    all_classes |= set(
        classes_of_module(semantic_digital_twin.world_description.degree_of_freedom)
    )
    all_classes |= set(classes_of_module(semantic_digital_twin.robots.robot_parts))
    all_classes |= set(classes_of_module(semantic_digital_twin.datastructures.definitions))
    all_classes |= set(classes_of_module(semantic_digital_twin.robots.hsrb))
    all_classes |= set(classes_of_module(semantic_digital_twin.robots.pr2))
    all_classes |= {SimulatorAdditionalProperty}
    all_classes |= set(classes_of_module(semantic_digital_twin.reasoning.predicates))
    all_classes |= set(classes_of_module(semantic_digital_twin.semantic_annotations.mixins))
    all_classes |= set(
        classes_of_module(semantic_digital_twin.adapters.procthor.procthor_resolver)
    )
    all_classes |= set(
        classes_of_module(semantic_digital_twin.world_description.world_modification)
    )
    all_classes |= set(classes_of_module(semantic_digital_twin.callbacks.callback))

    # Add generated classes (the key addition vs generate_orm.py)
    generated_module_name = "semantic_digital_twin.semantic_annotations.generated_classes"
    generated_module = sys.modules.get(generated_module_name)
    if generated_module:
        gen_count = 0
        for name in dir(generated_module):
            obj = getattr(generated_module, name)
            if isinstance(obj, type) and is_dataclass(obj):
                all_classes.add(obj)
                gen_count += 1
        logging.info("Added %d generated classes to ORM", gen_count)

    # Remove classes that should not be mapped
    all_classes -= {
        ResetStateContextManager,
        WorldModelUpdateContextManager,
        HasUpdateState,
        ForwardKinematicsManager,
        WorldModelManager,
        semantic_digital_twin.adapters.procthor.procthor_resolver.ProcthorResolver,
        ContainsType,
        SemanticDirection,
        JSONAttributeDiff,
        AttributeUpdateModification,
    }
    all_classes = {
        c for c in all_classes if is_dataclass(c) and not issubclass(c, AlternativeMapping)
    }
    all_classes |= {am.original_class() for am in recursive_subclasses(AlternativeMapping)}

    alternative_mappings = [
        am
        for am in recursive_subclasses(AlternativeMapping)
        if am.original_class() in all_classes
    ]

    class_diagram = ClassDiagram(
        list(sorted(all_classes, key=lambda c: c.__name__, reverse=True))
    )
    instance = ORMatic(
        class_dependency_graph=class_diagram,
        type_mappings=TypeDict(
            {
                trimesh.Trimesh: semantic_digital_twin.orm.model.TrimeshType,
            }
        ),
        alternative_mappings=alternative_mappings,
    )
    instance.make_all_tables()

    with open(ORM_FILE, "w") as f:
        instance.to_sqlalchemy_file(f)
    logging.info("ORM written to %s", ORM_FILE)


regenerate_orm()

# ---------------------------------------------------------------------------
# Step 3: Import the freshly generated ORM
# ---------------------------------------------------------------------------
from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO, Base
from semantic_digital_twin.semantic_annotations import semantic_annotations as sa_module

from sqlalchemy import select
from sqlalchemy.orm import Session
from krrood.ormatic.utils import create_engine
from krrood.ormatic.data_access_objects.helper import to_dao

DB_NAME = os.getenv("PGDATABASE")
DB_USER = os.getenv("PGUSER")
DB_PASSWORD = os.getenv("PGPASSWORD")
DB_HOST = "localhost"
DB_PORT = os.getenv("PGPORT", 5432)

connection_string = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


def build_class_lookup():
    """Build class_name -> class mapping from sa_module and generated_classes."""
    lookup = {}
    for name in dir(sa_module):
        obj = getattr(sa_module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, SemanticAnnotation)
            and obj is not SemanticAnnotation
        ):
            lookup[name] = obj

    generated_module_name = "semantic_digital_twin.semantic_annotations.generated_classes"
    generated_module = sys.modules.get(generated_module_name)
    if generated_module:
        for name in dir(generated_module):
            obj = getattr(generated_module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, SemanticAnnotation)
                and obj is not SemanticAnnotation
            ):
                lookup[name] = obj

    return lookup


def persist_one_room(
    pending: list[dict],
    world_db_id: int,
    class_lookup: dict,
    engine,
    room_label: str = "",
) -> bool:
    """Load one room's world, instantiate annotations, and persist."""
    tag = f" [{room_label}]" if room_label else ""

    logging.info("%sLoading world from database (id=%d)...", tag, world_db_id)
    with Session(engine) as session:
        queried_dao = session.scalar(
            select(WorldMappingDAO).where(
                WorldMappingDAO.database_id == world_db_id
            )
        )
        if queried_dao is None:
            logging.error(
                "%sNo world found with database_id %d", tag, world_db_id
            )
            return False
        world = queried_dao.from_dao()
    logging.info("%sWorld loaded successfully", tag)

    body_map = {str(b.id): b for b in world.bodies}
    logging.info("%sWorld has %d bodies", tag, len(body_map))

    # Instantiate annotations.  Field assignments may contain UUIDs
    # that reference other annotations (not just bodies), so we
    # iterate until all resolvable annotations are instantiated.
    ann_by_id = {ann["id"]: ann for ann in pending}
    instance_by_id: dict[str, SemanticAnnotation] = {}
    skipped_ids: set[str] = set()

    def _try_instantiate(ann: dict) -> SemanticAnnotation | None:
        """Try to instantiate one annotation.  Returns None if a
        dependency is not yet available (will be retried)."""
        cls_name = ann["class"]
        body_id = ann.get("body_id")

        cls = class_lookup.get(cls_name)
        if cls is None:
            logging.warning("%sClass %s not found in lookup, skipping", tag, cls_name)
            skipped_ids.add(ann["id"])
            return None

        kwargs = {}
        needs_root = issubclass(cls, HasRootKinematicStructureEntity)
        if needs_root and body_id:
            body = body_map.get(body_id)
            if body:
                kwargs["root"] = body
            else:
                skipped_ids.add(ann["id"])
                return None
        elif needs_root and not body_id:
            logging.warning(
                "%s%s needs root but no body_id provided, skipping",
                tag, cls_name,
            )
            skipped_ids.add(ann["id"])
            return None

        # Resolve field assignments: UUIDs can reference bodies or
        # previously instantiated annotations.
        for field_name, value in ann.get("field_assignments", {}).items():
            if isinstance(value, str) and value in body_map:
                kwargs[field_name] = body_map[value]
            elif isinstance(value, str) and value in instance_by_id:
                kwargs[field_name] = instance_by_id[value]
            elif isinstance(value, str) and _is_uuid(value):
                if value in skipped_ids:
                    logging.warning(
                        "%sField %s.%s references skipped annotation %s, skipping",
                        tag, cls_name, field_name, value,
                    )
                    skipped_ids.add(ann["id"])
                    return None
                # Dependency not yet instantiated — retry later
                return None
            else:
                kwargs[field_name] = value

        # SemanticEnvironmentAnnotation.__post_init__ resolves its
        # kinematic branch via self._world. On WorldEntity (the base
        # for SemanticAnnotation) _world is kw_only but init=True, so
        # the dataclass __init__ resets it to its default None unless
        # we pass it explicitly.
        kwargs["_world"] = world
        return cls(**kwargs)

    # Iterate until no more progress (handles dependency ordering)
    remaining = list(ann_by_id.keys())
    max_passes = len(remaining) + 1
    for pass_num in range(max_passes):
        next_remaining = []
        progress = False
        for ann_id in remaining:
            if ann_id in skipped_ids or ann_id in instance_by_id:
                continue
            try:
                instance = _try_instantiate(ann_by_id[ann_id])
            except Exception as e:
                logging.error(
                    "%sFailed to create %s: %s",
                    tag, ann_by_id[ann_id]["class"], e,
                )
                skipped_ids.add(ann_id)
                continue
            if instance is not None:
                instance_by_id[ann_id] = instance
                progress = True
            else:
                if ann_id not in skipped_ids:
                    next_remaining.append(ann_id)
        remaining = next_remaining
        if not remaining or not progress:
            break

    # Anything still remaining has unresolvable dependencies
    for ann_id in remaining:
        if ann_id not in skipped_ids:
            ann = ann_by_id[ann_id]
            logging.warning(
                "%sUnresolvable dependencies for %s (id=%s), skipping",
                tag, ann["class"], ann_id,
            )
            skipped_ids.add(ann_id)

    instances = list(instance_by_id.values())
    skipped = len(skipped_ids)

    logging.info(
        "%sInstantiated %d annotations (%d skipped)", tag, len(instances), skipped
    )

    if not instances:
        logging.warning("%sNo annotations to persist", tag)
        return True

    with world.modify_world():
        for instance in instances:
            world.add_semantic_annotation(instance)

    logging.info("%sAdded %d semantic annotations to world", tag, len(instances))

    logging.info("%sPersisting to database...", tag)
    with Session(engine) as session:
        world_dao = to_dao(world)
        world_dao.database_id = world_db_id
        session.merge(world_dao)
        session.commit()
        logging.info(
            "%sWorld updated (database_id: %d)", tag, world_db_id
        )

    logging.info("%sPersistence complete", tag)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Persist semantic annotations to the database"
    )
    parser.add_argument(
        "pending_annotations_json",
        type=Path,
        nargs="+",
        help="Path(s) to pending_annotations.json files",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--world-db-id",
        type=int,
        help="DB ID of the world (when all files share one world)",
    )
    group.add_argument(
        "--room-db-ids",
        type=Path,
        help="Path to room_db_ids.json mapping room names to DB IDs. "
             "Each pending_annotations file is matched to its room by "
             "its parent directory name.",
    )
    parser.add_argument(
        "--dataset",
        choices=["warsaw", "hm3d"],
        default="warsaw",
    )
    args = parser.parse_args()

    # Build class lookup
    class_lookup = build_class_lookup()
    logging.info("Class lookup has %d classes", len(class_lookup))

    # Connect to database
    engine = create_engine(connection_string, echo=False)

    # Drop only generated DAO tables whose inheritance may have changed
    # between scenes (the VLM can pick different superclasses each time).
    # Core tables (WorldMappingDAO, BodyDAO, etc.) are left intact so
    # data from other scenes is preserved.
    generated_module_name = "semantic_digital_twin.semantic_annotations.generated_classes"
    gen_mod = sys.modules.get(generated_module_name)
    if gen_mod:
        from sqlalchemy import inspect as sa_inspect, text
        inspector = sa_inspect(engine)
        existing_tables = set(inspector.get_table_names())
        tables_to_drop = []
        for name in dir(gen_mod):
            obj = getattr(gen_mod, name)
            if isinstance(obj, type) and is_dataclass(obj):
                dao_table_name = name + "DAO"
                if dao_table_name in existing_tables:
                    tables_to_drop.append(dao_table_name)
        if tables_to_drop:
            logging.info(
                "Dropping %d stale generated DAO tables: %s",
                len(tables_to_drop), tables_to_drop,
            )
            with engine.begin() as conn:
                for table_name in tables_to_drop:
                    conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))

    Base.metadata.create_all(bind=engine)

    # Load room_db_ids mapping if provided
    if args.room_db_ids:
        with open(args.room_db_ids) as f:
            room_db_ids = json.load(f)
    else:
        room_db_ids = None

    # Group annotations by room and persist each room against its own world
    failed = False
    for json_path in args.pending_annotations_json:
        with open(json_path) as f:
            pending = json.load(f)

        room_name = json_path.parent.name
        logging.info("Loaded %d annotations from %s", len(pending), json_path)

        if room_db_ids is not None:
            world_db_id = room_db_ids.get(room_name)
            if world_db_id is None:
                logging.error(
                    "No DB ID for room %s in %s", room_name, args.room_db_ids
                )
                failed = True
                continue
        else:
            world_db_id = args.world_db_id

        ok = persist_one_room(pending, world_db_id, class_lookup, engine, room_name)
        if not ok:
            failed = True

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
