"""
Script to refine class structure from VLM classification summary.

This script handles the complexity of semantic annotations where:
- Multiple SemanticAnnotations can exist per body (e.g., Cabinet + Container for same body)
- Annotations have constructor fields that reference other annotations
- Fields must be recursively resolved before instantiation

Process:
1. Phase 1 - Class Inference: For each object, infer the SemanticAnnotation class
2. Phase 2 - Dependency Resolution: Recursively resolve constructor fields,
   creating new annotations as needed until all dependencies are satisfied
3. Phase 3 - Instantiation: Topologically sort and instantiate all annotations
"""

import argparse
import inspect
import json
import logging
import os
import sys
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Tuple, Type, Optional, Set
from uuid import uuid4
from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.ormatic.data_access_objects import alternative_mappings
from krrood.ormatic.ormatic import ORMatic
import requests

from semantic_digital_twin.adapters.warsaw_world_loader import WarsawWorldLoader
from semantic_digital_twin.semantic_annotations import semantic_annotations as sa_module
from semantic_digital_twin.semantic_annotations.mixins import HasRootKinematicStructureEntity
from semantic_digital_twin.world import World
from semantic_digital_twin.semantic_annotations.in_memory_builder import (
    SemanticAnnotationClassBuilder,
    SemanticAnnotationFilePaths,
)
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Body,
)
from semantic_digital_twin.utils import InheritanceStructureExporter

from semdt_2d_video.hm3d_world_loader import HM3DWorldLoader


# Load generated_classes module BEFORE importing ormatic_interface
# (ormatic_interface may have imports that depend on generated_classes)
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

    # Also expose generated classes via in_memory_builder so that ORM files
    # generated with in-memory classes (whose __module__ was in_memory_builder)
    # can still resolve their references.
    import semantic_digital_twin.semantic_annotations.in_memory_builder as imb

    for name in dir(generated_module):
        obj = getattr(generated_module, name)
        if isinstance(obj, type) and issubclass(obj, SemanticAnnotation) and obj is not SemanticAnnotation:
            setattr(imb, name, obj)


_preload_generated_classes()

from semantic_digital_twin.orm import ormatic_interface
from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO, Base

from sqlalchemy import select
from sqlalchemy.orm import Session
from krrood.ormatic.helper import get_classes_of_ormatic_interface
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

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

_field_exporter = InheritanceStructureExporter(root_class=SemanticAnnotation)


@dataclass
class PendingAnnotation:
    """
    Represents a SemanticAnnotation that needs to be created.
    Tracks its class, associated body, and field dependencies.
    """

    id: str  # Unique identifier for this pending annotation
    class_name: str  # The class to instantiate
    body_id: Optional[str] = None  # The body this annotation relates to (if HasRootBody)
    field_assignments: Dict[str, Any] = field(default_factory=dict)
    # Maps field_name -> either:
    #   - str (pending_annotation_id) for SemanticAnnotation fields
    #   - actual value for primitive fields
    source: str = (
        "detected"  # "detected" (from VLM summary) or "inferred" (created to fill a slot)
    )
    confidence: float = 1.0

    def __hash__(self):
        return hash(self.id)


class AnnotationResolver:
    """
    Manages the resolution of SemanticAnnotation dependencies.
    Handles recursive creation of annotations needed to fill constructor slots.
    """

    def __init__(
        self,
        world: World,
        world_loader: WarsawWorldLoader,
        class_lookup: Dict[str, Type],
        use_vlm: bool = True,
        max_candidates: int = 8,
    ):
        self.world = world
        self.world_loader = world_loader
        self.class_lookup = class_lookup
        self.use_vlm = use_vlm
        self.max_candidates = max_candidates

        # All pending annotations indexed by id
        self.pending_annotations: Dict[str, PendingAnnotation] = {}

        # Index: body_id -> list of pending annotation ids
        self.body_to_annotations: Dict[str, List[str]] = {}

        # Index: class_name -> list of pending annotation ids
        self.class_to_annotations: Dict[str, List[str]] = {}

        # Track resolution state
        self.resolution_log: List[Dict[str, Any]] = []

    def add_pending_annotation(self, annotation: PendingAnnotation) -> str:
        """Add a pending annotation and update indices."""
        self.pending_annotations[annotation.id] = annotation

        if annotation.body_id:
            if annotation.body_id not in self.body_to_annotations:
                self.body_to_annotations[annotation.body_id] = []
            self.body_to_annotations[annotation.body_id].append(annotation.id)

        if annotation.class_name not in self.class_to_annotations:
            self.class_to_annotations[annotation.class_name] = []
        self.class_to_annotations[annotation.class_name].append(annotation.id)

        return annotation.id

    def get_unresolved_fields(
        self, annotation: PendingAnnotation
    ) -> List[Dict[str, Any]]:
        """Get list of fields that still need to be resolved for this annotation."""
        if annotation.class_name not in self.class_lookup:
            return []

        cls = self.class_lookup[annotation.class_name]
        all_fields = _field_exporter.collect_required_public_fields(cls)

        unresolved = []
        for f in all_fields:
            field_name = f["name"]
            field_type = f["type"]

            # Skip body field - handled separately
            if field_name == "body":
                continue

            # Check if this field expects a SemanticAnnotation
            if field_type in self.class_lookup:
                type_cls = self.class_lookup[field_type]
                if isinstance(type_cls, type) and issubclass(
                    type_cls, SemanticAnnotation
                ):
                    # This is a SemanticAnnotation field
                    if field_name not in annotation.field_assignments:
                        unresolved.append(f)

        return unresolved

    def find_compatible_annotation(
        self,
        field_type: str,
        source_annotation: PendingAnnotation,
    ) -> Optional[str]:
        """
        Find an existing pending annotation that can fill a field of the given type.
        Returns the annotation id if found, None otherwise.
        """
        if field_type not in self.class_lookup:
            return None

        base_cls = self.class_lookup[field_type]

        # Look for annotations of compatible types
        for class_name, annotation_ids in self.class_to_annotations.items():
            if class_name not in self.class_lookup:
                continue
            cls = self.class_lookup[class_name]
            if isinstance(cls, type) and issubclass(cls, base_cls):
                for ann_id in annotation_ids:
                    # Don't use the source annotation itself
                    if ann_id != source_annotation.id:
                        return ann_id

        return None

    def resolve_field_with_llm(
        self,
        source_annotation: PendingAnnotation,
        field_info: Dict[str, Any],
    ) -> Optional[str]:
        """
        Use LLM to determine how to fill a field.
        Returns the id of an existing or newly created annotation.
        """
        if not OPENROUTER_API_KEY:
            logging.warning("OPENROUTER_API_KEY not set, cannot query LLM")
            return None

        field_name = field_info["name"]
        field_type = field_info["type"]

        # Build context about available annotations
        available_annotations = []
        for ann_id, ann in self.pending_annotations.items():
            if ann_id == source_annotation.id:
                continue
            available_annotations.append(
                {
                    "annotation_id": ann_id,
                    "class": ann.class_name,
                    "body_id": ann.body_id,
                    "source": ann.source,
                }
            )

        prompt = f"""You are helping to build semantic annotations for a robotic scene understanding system.

## Context
We have a pending annotation:
- Class: {source_annotation.class_name}
- Body ID: {source_annotation.body_id}
- Source: {source_annotation.source}

This annotation needs a value for field `{field_name}` which expects type `{field_type}`.

## Available Annotations
{json.dumps(available_annotations, indent=2)}

## Your Task
Determine how to fill the `{field_name}` field. Options:
1. Use an existing annotation (provide its annotation_id)
2. Create a new annotation (specify the class and optionally a body_id)

Consider semantic relationships:
- A Cabinet's `container` field is typically a Container annotation for the same body
- A Drawer's `container` field is typically a Container for a Cabinet body
- Spatial/containment relationships matter

Respond with valid JSON:
{{
  "action": "use_existing" | "create_new",
  "annotation_id": "string (if use_existing)",
  "new_class": "string (if create_new)",
  "new_body_id": "string | null (if create_new)",
  "reasoning": "brief explanation"
}}
"""

        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "ai.uni-bremen.de",
                    "X-Title": "Uni Bremen",
                },
                data=json.dumps(
                    {
                        "model": "meta-llama/llama-3.3-70b-instruct",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a semantic scene understanding assistant. Respond only with valid JSON.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                    }
                ),
            )

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            content = content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(content)

            logging.info(
                f"LLM response for {source_annotation.class_name}.{field_name}: {parsed}"
            )

            if parsed["action"] == "use_existing":
                ann_id = parsed.get("annotation_id")
                if ann_id in self.pending_annotations:
                    return ann_id
                else:
                    logging.warning(f"LLM suggested non-existent annotation {ann_id}")
                    return None

            elif parsed["action"] == "create_new":
                new_class = parsed.get("new_class")
                if not new_class or new_class not in self.class_lookup:
                    logging.warning(f"LLM suggested unknown class {new_class}")
                    return None

                # Create new pending annotation
                new_ann = PendingAnnotation(
                    id=str(uuid4()),
                    class_name=new_class,
                    body_id=parsed.get("new_body_id"),
                    source="inferred",
                    confidence=0.8,  # Lower confidence for inferred annotations
                )
                self.add_pending_annotation(new_ann)
                logging.info(
                    f"Created inferred annotation: {new_class} (body: {new_ann.body_id})"
                )
                return new_ann.id

        except Exception as e:
            logging.error(f"Failed to query LLM: {e}")
            return None

        return None

    def resolve_field_by_heuristics(
        self,
        source_annotation: PendingAnnotation,
        field_info: Dict[str, Any],
    ) -> Optional[str]:
        """
        Use heuristics to resolve a field without LLM.
        """
        field_name = field_info["name"]
        field_type = field_info["type"]

        # Heuristic 1: For Container field, create a Container for the same body
        if field_type == "Container" and source_annotation.body_id:
            # Check if there's already a Container for this body
            for ann_id in self.body_to_annotations.get(source_annotation.body_id, []):
                ann = self.pending_annotations[ann_id]
                if ann.class_name == "Container":
                    return ann_id

            # Create new Container for the same body
            new_ann = PendingAnnotation(
                id=str(uuid4()),
                class_name="Container",
                body_id=source_annotation.body_id,
                source="inferred",
                confidence=0.9,
            )
            self.add_pending_annotation(new_ann)
            logging.info(
                f"Created Container for body {source_annotation.body_id} (heuristic)"
            )
            return new_ann.id

        # Heuristic 2: Look for compatible type in same body first
        if source_annotation.body_id:
            for ann_id in self.body_to_annotations.get(source_annotation.body_id, []):
                if ann_id == source_annotation.id:
                    continue
                ann = self.pending_annotations[ann_id]
                if self._is_compatible_type(ann.class_name, field_type):
                    return ann_id

        # Heuristic 3: Look for any compatible annotation
        return self.find_compatible_annotation(field_type, source_annotation)

    def _is_compatible_type(self, class_name: str, expected_type: str) -> bool:
        """Check if class_name is a subclass of expected_type."""
        if (
            class_name not in self.class_lookup
            or expected_type not in self.class_lookup
        ):
            return False
        cls = self.class_lookup[class_name]
        expected_cls = self.class_lookup[expected_type]
        return isinstance(cls, type) and issubclass(cls, expected_cls)

    def resolve_all_dependencies(self, max_iterations: int = 10) -> bool:
        """
        Recursively resolve all field dependencies for all pending annotations.
        Returns True if all dependencies were resolved.
        """
        for iteration in range(max_iterations):
            logging.info(f"=== Resolution iteration {iteration + 1} ===")

            # Collect all unresolved fields
            unresolved_count = 0
            progress_made = False

            for ann_id, annotation in list(self.pending_annotations.items()):
                unresolved_fields = self.get_unresolved_fields(annotation)

                for field_info in unresolved_fields:
                    unresolved_count += 1
                    field_name = field_info["name"]
                    field_type = field_info["type"]

                    logging.info(
                        f"Resolving {annotation.class_name}.{field_name} "
                        f"(expects {field_type})"
                    )

                    # Try heuristics first
                    resolved_id = self.resolve_field_by_heuristics(
                        annotation, field_info
                    )

                    # Fall back to LLM if needed and enabled
                    if resolved_id is None and self.use_vlm:
                        resolved_id = self.resolve_field_with_llm(
                            annotation, field_info
                        )

                    if resolved_id:
                        annotation.field_assignments[field_name] = resolved_id
                        progress_made = True
                        self.resolution_log.append(
                            {
                                "source_annotation": ann_id,
                                "source_class": annotation.class_name,
                                "field_name": field_name,
                                "resolved_to": resolved_id,
                                "iteration": iteration,
                            }
                        )
                        logging.info(f"  -> Resolved to annotation {resolved_id}")
                    else:
                        logging.warning(f"  -> Could not resolve")

            if unresolved_count == 0:
                logging.info("All dependencies resolved!")
                return True

            if not progress_made:
                logging.warning(f"No progress made in iteration {iteration + 1}")
                # Continue anyway - might create new annotations in next iteration

        logging.warning(
            f"Max iterations ({max_iterations}) reached with unresolved dependencies"
        )
        return False

    def get_instantiation_order(self) -> List[str]:
        """
        Topologically sort pending annotations based on dependencies.
        Returns list of annotation ids in order they should be instantiated.
        """
        # Build dependency graph
        dependencies: Dict[str, Set[str]] = {
            ann_id: set() for ann_id in self.pending_annotations
        }

        for ann_id, annotation in self.pending_annotations.items():
            for field_name, value in annotation.field_assignments.items():
                if isinstance(value, str) and value in self.pending_annotations:
                    dependencies[ann_id].add(value)

        # Kahn's algorithm for topological sort
        in_degree = {ann_id: len(deps) for ann_id, deps in dependencies.items()}
        queue = [ann_id for ann_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            ann_id = queue.pop(0)
            result.append(ann_id)

            for other_id in self.pending_annotations:
                if ann_id in dependencies[other_id]:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)

        if len(result) != len(self.pending_annotations):
            logging.warning(
                "Circular dependency detected! Some annotations may fail to instantiate."
            )
            # Add remaining annotations
            for ann_id in self.pending_annotations:
                if ann_id not in result:
                    result.append(ann_id)

        return result

    def instantiate_all(self) -> Tuple[List[SemanticAnnotation], List[Dict[str, Any]]]:
        """
        Instantiate all pending annotations.

        Uses multi-pass iteration so that an annotation whose
        SemanticAnnotation-typed kwargs reference other pending
        annotations is retried until those dependencies have been
        instantiated.  When a dependency cannot be instantiated, the
        dependent is also skipped (no partial construction).

        The whole loop runs inside ``world.modify_world()`` because some
        annotations (notably ``SemanticRobotAnnotation`` subclasses such
        as ``Sensor``/``SmokeDetector``) self-register in their
        ``__post_init__`` via ``world.add_semantic_annotation``, which
        is wrapped in ``@atomic_world_modification`` and requires the
        modification context.
        """
        instances: Dict[str, SemanticAnnotation] = {}
        skipped: Dict[str, str] = {}  # ann_id -> failure reason

        body_map = {str(b.id): b for b in self.world.bodies}

        def _try_instantiate(
            annotation: PendingAnnotation,
        ) -> Tuple[Optional[SemanticAnnotation], Optional[str]]:
            """Try to construct one annotation.

            Returns ``(instance, None)`` on success, ``(None, reason)``
            for a permanent skip, or ``(None, None)`` to signal that a
            dependency is not yet instantiated (retry later).
            """
            cls_name = annotation.class_name
            if cls_name not in self.class_lookup:
                return None, f"Class {cls_name} not found"
            cls = self.class_lookup[cls_name]

            kwargs: Dict[str, Any] = {}

            # Handle body field — HasRootKinematicStructureEntity covers the
            # whole mixin family (HasRootBody, HasRootRegion, HasDoors, …), which
            # the SemanticEnvironmentAnnotation branch (Ceiling, Light, …) reaches
            # through HasRootBody.
            needs_root = issubclass(cls, HasRootKinematicStructureEntity)
            if needs_root:
                if not annotation.body_id:
                    return None, (
                        f"{cls_name} requires a root body but the pending "
                        f"annotation has no body_id (likely an inferred "
                        f"placeholder created to fill another annotation's slot)"
                    )
                body = body_map.get(annotation.body_id)
                if body is None:
                    return None, f"Body {annotation.body_id} not found in world"
                kwargs["root"] = body

            # SemanticEnvironmentAnnotation.__post_init__ dereferences
            # self._world to resolve its kinematic branch, so the world
            # backreference must be set at construction time.
            kwargs["_world"] = self.world

            for field_name, value in annotation.field_assignments.items():
                if isinstance(value, str) and value in instances:
                    kwargs[field_name] = instances[value]
                elif isinstance(value, str) and value in skipped:
                    return None, (
                        f"depends on skipped annotation {value} for field "
                        f"{field_name}: {skipped[value]}"
                    )
                elif isinstance(value, str) and value in self.pending_annotations:
                    # Dependency not yet instantiated — retry on a later pass.
                    return None, None
                else:
                    kwargs[field_name] = value

            try:
                instance = cls(**kwargs)
            except Exception as e:
                return None, f"{type(e).__name__}: {e}"
            return instance, None

        with self.world.modify_world():
            remaining = list(self.pending_annotations.keys())
            max_passes = len(remaining) + 1
            for _ in range(max_passes):
                next_remaining: List[str] = []
                progress = False
                for ann_id in remaining:
                    annotation = self.pending_annotations[ann_id]
                    instance, reason = _try_instantiate(annotation)
                    if instance is not None:
                        instances[ann_id] = instance
                        progress = True
                        logging.info(
                            f"Created {annotation.class_name} instance (id: {ann_id})"
                        )
                    elif reason is not None:
                        skipped[ann_id] = reason
                        progress = True
                        logging.error(
                            f"Failed to create {annotation.class_name}: {reason}"
                        )
                    else:
                        next_remaining.append(ann_id)
                remaining = next_remaining
                if not remaining or not progress:
                    break

            for ann_id in remaining:
                annotation = self.pending_annotations[ann_id]
                reason = "unresolvable dependencies after all instantiation passes"
                skipped[ann_id] = reason
                logging.error(
                    f"Failed to create {annotation.class_name}: {reason}"
                )

        results: List[Dict[str, Any]] = []
        for ann_id, annotation in self.pending_annotations.items():
            if ann_id in instances:
                results.append(
                    {
                        "annotation_id": ann_id,
                        "class": annotation.class_name,
                        "body_id": annotation.body_id,
                        "source": annotation.source,
                        "status": "created",
                    }
                )
            else:
                results.append(
                    {
                        "annotation_id": ann_id,
                        "class": annotation.class_name,
                        "body_id": annotation.body_id,
                        "status": "failed",
                        "error": skipped.get(ann_id, "unknown"),
                    }
                )

        return list(instances.values()), results


GENERATED_CLASSES_MODULE_NAME = (
    "semantic_digital_twin.semantic_annotations.generated_classes"
)


def load_generated_classes_module():
    """Load generated_classes.py as a proper module and register it in sys.modules."""
    # Return cached module if already loaded
    if GENERATED_CLASSES_MODULE_NAME in sys.modules:
        return sys.modules[GENERATED_CLASSES_MODULE_NAME]

    generated_file = Path(SemanticAnnotationFilePaths.GENERATED_CLASSES_FILE.value)
    if not generated_file.exists():
        return None

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        GENERATED_CLASSES_MODULE_NAME, generated_file
    )
    generated_module = importlib.util.module_from_spec(spec)
    # Register in sys.modules so ORMatic can import it
    sys.modules[GENERATED_CLASSES_MODULE_NAME] = generated_module
    spec.loader.exec_module(generated_module)

    # Also set it as an attribute on the parent module so `import x.y.z` works
    import semantic_digital_twin.semantic_annotations as parent_module

    setattr(parent_module, "generated_classes", generated_module)

    return generated_module


def _get_existing_dao_names() -> Set[str]:
    """Return the set of DAO class names already defined in the ORM (before any generation).

    This is used to detect name collisions when generating new SemanticAnnotation
    classes whose ``<Name>DAO`` would shadow an existing DAO (e.g. geometry.Box → BoxDAO).
    """
    classes, _, _ = get_classes_of_ormatic_interface(ormatic_interface)
    return {cls.__name__ + "DAO" for cls in classes}


def _to_pascal_case(name: str) -> str:
    """Convert an arbitrary string to a valid PascalCase Python identifier.

    Handles spaces, hyphens, underscores, and leading digits.
    E.g. ``"LED Television"`` -> ``"LEDTelevision"``,
         ``"floor-mat"`` -> ``FloorMat``,
         ``"3d_printer"`` -> ``_3dPrinter``.
    """
    import re as _re
    # Split on non-alphanumeric characters
    parts = _re.split(r"[^a-zA-Z0-9]+", name)
    # Capitalize each part, but preserve fully-uppercase parts (acronyms)
    result = ""
    for part in parts:
        if not part:
            continue
        if part.isupper():
            result += part
        else:
            result += part[0].upper() + part[1:]
    # Ensure valid identifier (can't start with digit)
    if result and result[0].isdigit():
        result = "_" + result
    return result or "UnknownClass"


def _safe_class_name(proposed: str, existing_dao_names: Set[str]) -> str:
    """Return *proposed* if its DAO name is free, otherwise disambiguate.

    For example, if ``BoxDAO`` already exists for ``geometry.Box``, the
    generated semantic-annotation class is renamed to ``HouseholdBox`` (and
    further suffixed if that also collides).
    """
    if proposed + "DAO" not in existing_dao_names:
        return proposed
    # Try common disambiguation prefixes
    for prefix in ("Household", "Semantic", "Annotated"):
        candidate = prefix + proposed
        if candidate + "DAO" not in existing_dao_names:
            logging.warning(
                f"Class name '{proposed}' collides with existing DAO "
                f"'{proposed}DAO' — renamed to '{candidate}'"
            )
            return candidate
    # Fallback: numeric suffix
    for i in range(2, 100):
        candidate = f"{proposed}{i}"
        if candidate + "DAO" not in existing_dao_names:
            logging.warning(
                f"Class name '{proposed}' collides with existing DAO — renamed to '{candidate}'"
            )
            return candidate
    raise RuntimeError(f"Could not find a non-colliding name for '{proposed}'")


def build_class_lookup() -> Dict[str, Type]:
    """Build a name -> class lookup from the semantic_annotations module and generated_classes if it exists.

    Classes that are still abstract (i.e. ``inspect.isabstract`` is
    True — they have unimplemented ``@abstractmethod`` members) are
    excluded so that the VLM/heuristic pipeline can never select one
    as either a target class or a superclass for a generated class.
    Otherwise generated classes like ``Television(SemanticRobotAnnotation)``
    inherit ``assign_to_robot`` without an implementation and crash at
    instantiation with ``Can't instantiate abstract class …``.  When an
    abstract base is filtered, the superclass-resolution code falls
    back to ``SemanticAnnotation`` (see ``main``).
    """
    lookup = {}
    classes, _, _ = get_classes_of_ormatic_interface(ormatic_interface)
    for cls in filter(lambda c: issubclass(c, SemanticAnnotation), classes):
        if inspect.isabstract(cls):
            continue
        lookup[cls.__name__] = cls

    # Also try to import from generated_classes.py if it exists
    try:
        generated_module = load_generated_classes_module()
        if generated_module:
            for name in dir(generated_module):
                obj = getattr(generated_module, name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, SemanticAnnotation)
                    and obj is not SemanticAnnotation
                    and not inspect.isabstract(obj)
                ):
                    lookup[name] = obj
                    logging.info(f"Loaded generated class: {name}")
    except Exception as e:
        logging.warning(f"Failed to load generated_classes.py: {e}")

    return lookup


def encode_image_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load world from database
    logging.info("Loading world from database...")
    engine = create_engine(connection_string, echo=False)
    Base.metadata.create_all(bind=engine)
    with Session(engine) as session:
        queried_dao = session.scalar(
            select(WorldMappingDAO).where(
                WorldMappingDAO.database_id == args.world_database_id
            )
        )
        if queried_dao:
            world: World = queried_dao.from_dao()
        else:
            raise ValueError(
                f"No world found with database_id {args.world_database_id}"
            )

    world_loader = WarsawWorldLoader.from_world(world) if args.dataset == "warsaw" else HM3DWorldLoader.from_world(world)

    # Load summary
    with open(args.summary_json, "r") as f:
        summary: List[Dict[str, Any]] = json.load(f)

    logging.info(f"Loaded {len(summary)} object classifications")

    # Build class lookup
    class_lookup = build_class_lookup()

    # =========================================================================
    # PHASE 1: Class Inference - Create pending annotations from VLM summary
    # =========================================================================
    logging.info("=" * 60)
    logging.info("PHASE 1: Class Inference")
    logging.info("=" * 60)

    resolver = AnnotationResolver(
        world=world,
        world_loader=world_loader,
        class_lookup=class_lookup,
        use_vlm=not args.no_llm,
        max_candidates=args.max_candidates,
    )

    # Collect builders for new classes to write them all to a single file
    new_class_builders: List[SemanticAnnotationClassBuilder] = []

    # Pre-compute existing DAO names so we can detect collisions when the VLM
    # proposes a class name that already maps to a non-annotation DAO
    # (e.g. geometry.Box → BoxDAO).
    existing_dao_names = _get_existing_dao_names()

    for obj in summary:
        if (obj.get("confidence") or 0) < args.min_confidence:
            logging.warning(
                f"Skipping {obj['body_id']} ({obj['class']}) due to low confidence"
            )
            continue

        cls_name = _to_pascal_case(obj["class"])
        superclass_name = _to_pascal_case(obj.get("superclass", "SemanticAnnotation"))

        # Create class if it doesn't exist
        if cls_name not in class_lookup:
            # Guard against DAO name collisions with existing (non-annotation) classes
            safe_name = _safe_class_name(cls_name, existing_dao_names)
            was_renamed = safe_name != cls_name
            if was_renamed:
                logging.info(
                    f"Renamed '{cls_name}' → '{safe_name}' to avoid DAO collision"
                )
                original_name = cls_name
                cls_name = safe_name

            logging.info(
                f"Creating new class: {cls_name} (superclass: {superclass_name})"
            )

            if superclass_name not in class_lookup:
                logging.warning(
                    f"Superclass '{superclass_name}' not found, using SemanticAnnotation"
                )
                superclass_name = "SemanticAnnotation"

            superclass = class_lookup[superclass_name]
            builder = SemanticAnnotationClassBuilder(
                cls_name, template_name="dataclass_template.py.jinja"
            )
            cls = builder.add_base(superclass).build(
                module="semantic_digital_twin.semantic_annotations.generated_classes"
            )
            new_class_builders.append(builder)
            class_lookup[cls_name] = cls
            # Also register under the original VLM name so subsequent summary
            # entries with the same class name resolve without re-creation.
            if was_renamed:
                class_lookup[original_name] = cls
            # Track the new DAO name so subsequent classes don't collide with it either
            existing_dao_names.add(cls_name + "DAO")
            logging.info(f"Created class '{cls_name}'")

        # Resolve cls_name to the actual generated class name.  When a
        # DAO collision caused a rename (e.g. Box → HouseholdBox), the
        # original VLM name is registered in class_lookup as an alias.
        # Use the real class __name__ so pending_annotations.json always
        # references the name that exists in generated_classes.py.
        cls_name = class_lookup[cls_name].__name__

        # Create pending annotation
        pending_ann = PendingAnnotation(
            id=str(uuid4()),
            class_name=cls_name,
            body_id=obj["body_id"],
            source="detected",
            confidence=obj.get("confidence", 1.0),
        )
        resolver.add_pending_annotation(pending_ann)
        logging.info(
            f"Created pending annotation: {cls_name} for body {obj['body_id']}"
        )

    # Write new classes to the generated file, appending to any existing content
    # and ensuring all superclass imports are present
    if new_class_builders:
        generated_file = Path(SemanticAnnotationFilePaths.GENERATED_CLASSES_FILE.value)

        # Collect import requirements from new builders
        from semantic_digital_twin.semantic_annotations.in_memory_builder import (
            _MODULE_TO_IMPORT,
        )
        new_imports_by_source: Dict[str, set] = {}
        for builder in new_class_builders:
            for base in builder.bases:
                key = SemanticAnnotationClassBuilder._get_import_key(base)
                if key not in new_imports_by_source:
                    new_imports_by_source[key] = set()
                new_imports_by_source[key].add(base.__name__)

        # Read existing file content and update imports
        if generated_file.exists():
            existing_content = generated_file.read_text()
        else:
            existing_content = ""

        # Parse existing imports to merge with new ones
        import re as _re
        existing_lines = existing_content.split("\n")
        non_import_lines = []
        existing_imports_by_source: Dict[str, set] = {}
        past_imports = False
        for line in existing_lines:
            # Skip boilerplate imports that we always regenerate
            if line.startswith("from __future__") or line.startswith("from dataclasses"):
                continue
            m = _re.match(r"^from\s+(\S+)\s+import\s+(.+)$", line)
            if m and not past_imports:
                module_path = m.group(1)
                names = {n.strip() for n in m.group(2).split(",")}
                # Map back to source key
                source_key = module_path
                for key, template in _MODULE_TO_IMPORT.items():
                    if key in module_path:
                        source_key = key
                        break
                if source_key not in existing_imports_by_source:
                    existing_imports_by_source[source_key] = set()
                existing_imports_by_source[source_key].update(names)
            elif line.strip() == "":
                if past_imports:
                    non_import_lines.append(line)
            else:
                past_imports = True
                non_import_lines.append(line)

        # Merge imports
        merged_imports = {}
        for source in set(list(existing_imports_by_source.keys()) + list(new_imports_by_source.keys())):
            merged_imports[source] = (
                existing_imports_by_source.get(source, set())
                | new_imports_by_source.get(source, set())
            )

        # Remove locally-defined classes from imports: if a class is defined
        # in generated_classes.py itself, importing it from another module
        # will either fail (it doesn't exist there) or shadow the local def.
        import re as _re_local
        local_class_names = set()
        for line in non_import_lines:
            m = _re_local.match(r"^class\s+(\w+)", line)
            if m:
                local_class_names.add(m.group(1))
        for builder in new_class_builders:
            local_class_names.add(builder.name)
        for source in list(merged_imports):
            merged_imports[source] -= local_class_names
            if not merged_imports[source]:
                del merged_imports[source]

        # Build import lines
        import_lines = [
            "from __future__ import annotations",
            "",
            "from dataclasses import dataclass",
            "",
        ]
        for source, class_names in sorted(merged_imports.items()):
            sorted_names = ", ".join(sorted(class_names))
            if source in _MODULE_TO_IMPORT:
                import_lines.append(
                    _MODULE_TO_IMPORT[source].format(classes=sorted_names)
                )
            else:
                import_lines.append(f"from {source} import {sorted_names}")

        # Render new class definitions
        new_class_defs = []
        for builder in new_class_builders:
            new_class_defs.append(builder.render_source(include_imports=False))

        # Combine existing class definitions with new ones
        existing_class_text = "\n".join(non_import_lines).strip()
        all_class_text = existing_class_text
        if all_class_text and new_class_defs:
            all_class_text += "\n\n"
        all_class_text += "\n\n".join(new_class_defs)

        # Write complete file
        with open(str(generated_file), "w", encoding="utf-8") as f:
            f.write("\n".join(import_lines))
            f.write("\n\n")
            f.write(all_class_text)
            f.write("\n")

        logging.info(
            f"Wrote {len(new_class_builders)} new classes to {generated_file} "
            f"(with updated imports)"
        )

    logging.info(
        f"Phase 1 complete: {len(resolver.pending_annotations)} pending annotations"
    )

    # =========================================================================
    # PHASE 2: Dependency Resolution - Recursively resolve all constructor fields
    # =========================================================================
    logging.info("=" * 60)
    logging.info("PHASE 2: Dependency Resolution")
    logging.info("=" * 60)

    if not args.dry_run:
        all_resolved = resolver.resolve_all_dependencies(
            max_iterations=args.max_iterations
        )

        if not all_resolved:
            logging.warning("Not all dependencies could be resolved")

        # Save resolution log
        resolution_log_file = args.summary_json.parent / "resolution_log.json"
        with open(resolution_log_file, "w") as f:
            json.dump(resolver.resolution_log, f, indent=2)
        logging.info(f"Resolution log saved to {resolution_log_file}")

        # Save pending annotations state
        pending_state = [
            {
                "id": ann.id,
                "class": ann.class_name,
                "body_id": ann.body_id,
                "source": ann.source,
                "confidence": ann.confidence,
                "field_assignments": ann.field_assignments,
                "unresolved_fields": [
                    f["name"] for f in resolver.get_unresolved_fields(ann)
                ],
            }
            for ann in resolver.pending_annotations.values()
        ]
        pending_file = args.summary_json.parent / "pending_annotations.json"
        with open(pending_file, "w") as f:
            json.dump(pending_state, f, indent=2)
        logging.info(f"Pending annotations saved to {pending_file}")
    else:
        logging.info("Dry run - skipping dependency resolution")

    logging.info(
        f"Phase 2 complete: {len(resolver.pending_annotations)} total annotations"
    )

    # =========================================================================
    # PHASE 3: Instantiation - Create actual SemanticAnnotation instances
    # =========================================================================
    logging.info("=" * 60)
    logging.info("PHASE 3: Instantiation")
    logging.info("=" * 60)

    if not args.dry_run:
        instances, instantiation_results = resolver.instantiate_all()

        # Save instantiation results
        instantiation_file = args.summary_json.parent / "instantiation_results.json"
        with open(instantiation_file, "w") as f:
            json.dump(instantiation_results, f, indent=2)
        logging.info(f"Instantiation results saved to {instantiation_file}")

        # Summary
        created = sum(1 for r in instantiation_results if r["status"] == "created")
        failed = sum(1 for r in instantiation_results if r["status"] == "failed")
        logging.info(f"Phase 3 complete: {created} created, {failed} failed")

        # Persistence is handled externally (by run_batch.py or manually
        # via persist_annotations.py) after all rooms have been processed,
        # so the expensive ORM regeneration only happens once.
    else:
        logging.info("Dry run - skipping instantiation")

    logging.info("=" * 60)
    logging.info("Processing complete")
    logging.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refine class structure from VLM classification summary"
    )
    parser.add_argument(
        "summary_json",
        type=Path,
        help="Path to summary JSON file (body_id -> class mapping)",
    )
    parser.add_argument(
        "world_database_id",
        type=int,
        help="DB ID of the world stored in the previous step",
    )
    parser.add_argument(
        "--dataset",
        choices=["warsaw", "hm3d"],
        default="warsaw",
        help="Dataset type: 'warsaw' (default) or 'hm3d'",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for output files (default: same as summary_json)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.9,
        help="Minimum confidence threshold (default: 0.9)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only perform Phase 1 (class inference), skip resolution and instantiation",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM queries, use only heuristics for dependency resolution",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=8,
        help="Maximum candidate objects per slot (default: 8)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum resolution iterations (default: 10)",
    )
    parser.add_argument(
        "--skip-persist",
        action="store_true",
        help="Skip persisting the world with semantic annotations to the database",
    )
    args = parser.parse_args()
    main(args)
