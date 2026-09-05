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
)
from semantic_digital_twin.semantic_annotations.taxonomy_export import (
    annotation_classes,
    in_base_order,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

from pipeline_scripts import run_classes

TEMPLATE = "dataclass_template.py.jinja"
"""
What a generated class is written from.
"""


def wanted_classes(
    classifications: Dict[str, Any], vocabulary: Dict[str, Any]
) -> Dict[str, List[str]]:
    """
    Say what each class a body was given should be built from.

    Two steps proposed compositions and only one of them was asked to. The vocabulary
    step answers what a *label* means and names a superclass and the mixins to compose
    it from, having been shown what objects of that label were measured to meet; the
    classification step answers which class each *object* is, from a picture, and its
    schema carries a superclass as well. Read from the classification alone, a Faucet
    that the vocabulary had composed with HasHandle comes out with no way to hold a
    handle at all, and the pairing measured for it cannot be mounted.

    So the bases come from the vocabulary where it composed that class, and from the
    classification only where it did not.

    :param classifications: What the classification step wrote.
    :param vocabulary: What the vocabulary step wrote.
    :return: Per class name, the names of the classes to derive it from.
    """
    composed = {
        answer["class"]: [answer["superclass"]] + list(answer.get("mixins") or [])
        for answer in vocabulary["labels"].values()
        if answer.get("class") and answer.get("is_new_class") and answer.get("superclass")
    }

    wanted: Dict[str, List[str]] = {}
    for answer in classifications["bodies"].values():
        name = answer.get("class")
        if not name or name in wanted:
            continue
        wanted[name] = composed.get(
            name, [answer.get("superclass") or "SemanticAnnotation"]
        )
    return wanted


def generate_missing(
    wanted: Dict[str, List[str]], known: Dict[str, Type], run: Path
) -> List[str]:
    """
    Write the classes a scene needs that the taxonomy does not have.

    The file is written whole rather than appended to: a class one scene needed is not a
    class the next one starts with, and nothing outside this run should ever import it.

    :param wanted: Per class name, the names of the classes to derive it from.
    :param known: The taxonomy's classes by name.
    :param run: The run's directory, which is where they are written -- they belong to
        the run that proposed them, not to the ontology every later run starts from.
    :return: The names that were generated.
    """
    builders, generated = [], []
    for name, base_names in sorted(wanted.items()):
        if name in known:
            continue
        bases = [known[one] for one in base_names if one in known]
        for missing in [one for one in base_names if one not in known]:
            print(f"  {name}: {missing} is not in the taxonomy, leaving it out")
        if not bases:
            bases = [SemanticAnnotation]

        # Ordered as a class must declare them, since a proposal naming HasRootBody
        # beside IsStorageSpace -- which derives from it -- names them the wrong way
        # round for Python.
        builder = SemanticAnnotationClassBuilder(name, template_name=TEMPLATE)
        for base in in_base_order(bases):
            builder.add_base(base)
        builders.append(builder)
        generated.append(f"{name}({', '.join(base.__name__ for base in in_base_order(bases))})")

    if builders:
        SemanticAnnotationClassBuilder.write_classes_to_file(
            builders, run_classes.path_in(run)
        )
    return generated


def regenerate_orm(run: Path) -> Optional[str]:
    """
    Rebuild the ORM so the database knows the classes this run generated.

    Run in a new interpreter with the run's directory at the front of the annotations
    package's search path: the generator finds classes by walking that path, so the
    run's file is the ``generated_classes`` it maps, without the generator being told
    anything and without the classes ever being written into the ontology's own package.

    :param run: The run's directory.
    :return: What went wrong, or None when it was rebuilt.
    """
    script = (
        Path(semantic_digital_twin.__file__).parent.parent.parent
        / "scripts"
        / "generate_orm.py"
    )
    if not script.exists():
        return f"the ORM generator is not at {script}"

    program = (
        "import importlib.util, sys\n"
        "from pipeline_scripts import run_classes\n"
        "run_classes.use(sys.argv[1])\n"
        "specification = importlib.util.spec_from_file_location("
        "'generate_orm', sys.argv[2])\n"
        "generator = importlib.util.module_from_spec(specification)\n"
        "specification.loader.exec_module(generator)\n"
        "generator.generate_orm()\n"
    )
    finished = subprocess.run(
        [sys.executable, "-c", program, str(Path(run).resolve()), str(script)],
        capture_output=True,
        text=True,
    )
    return None if finished.returncode == 0 else finished.stderr.strip()[-2000:]


def mount(arguments: argparse.Namespace) -> None:
    """
    Annotate the bodies of a persisted world and mount its parts.

    Imports the ORM and the generated classes, so it runs only once they are written.

    :param arguments: The command line arguments.
    """
    # Before the ORM is imported, so that the classes it names are this run's.
    run_classes.use(arguments.evidence_directory)

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

    (evidence / "inspect_world.py").write_text(
        INSPECTOR.format(annotated=annotated_id, split=world_db_id)
    )
    (evidence / "report.md").write_text(report(evidence, annotated_id, world_db_id))
    print(f"written to {evidence / 'report.md'} and {evidence / 'inspect_world.py'}")


INSPECTOR = '''"""
Open the world this run built.

    python inspect_world.py            # what the world holds
    python inspect_world.py --view     # and look at it

Written by the run that built it, so the id is the one it was written under.
"""

import argparse
from collections import Counter
from pathlib import Path

# The classes this run generated live beside this script, and the world names them. This
# has to happen before the ORM is imported, or the ORM looks for them in the ontology's
# own package, where they deliberately are not.
from pipeline_scripts import run_classes, run_database

HERE = Path(__file__).resolve().parent
run_classes.use(HERE)

# The world was written into a schema of this run's own, so this is where to look for
# it. Set before the sessionmaker is called, which is what reads it.
run_database.use(HERE)

from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO  # noqa: E402
from semantic_digital_twin.orm.utils import (  # noqa: E402
    semantic_digital_twin_sessionmaker,
)
from semantic_digital_twin.spatial_computations.raytracer import RayTracer  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

ANNOTATED_WORLD = {annotated}
SPLIT_WORLD = {split}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-db-id", type=int, default=ANNOTATED_WORLD,
                        help=f"Which world to open ({{ANNOTATED_WORLD}} annotated, "
                             f"{{SPLIT_WORLD}} as it was split).")
    parser.add_argument("--view", action="store_true",
                        help="Open it in the viewer; close the window to finish.")
    arguments = parser.parse_args()

    engine = semantic_digital_twin_sessionmaker()().bind
    with Session(engine) as session:
        stored = session.get(WorldMappingDAO, arguments.world_db_id)
        if stored is None:
            raise SystemExit(f"no world in the database with id {{arguments.world_db_id}}")
        world = stored.from_dao()

    annotations = list(world.semantic_annotations)
    print(f"world {{arguments.world_db_id}}: {{len(world.bodies)}} bodies, "
          f"{{len(annotations)}} annotations")
    for name, count in Counter(type(one).__name__ for one in annotations).most_common():
        print(f"  {{count:>4}} {{name}}")

    plumbing = {{"root", "name", "id", "_world", "_semantic_annotations",
                "simulator_additional_properties", "_inference_explanation_"}}

    def holds_something(value) -> bool:
        # Not `value not in (None, [], ())`: comparing an annotation against those
        # sentinels calls its own __eq__, which hashes both sides, and a list is not
        # hashable.
        if value is None:
            return False
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) > 0
        return True

    held = [
        (str(one.root.name.name), field, part)
        for one in annotations
        if getattr(one, "root", None)
        for field, part in vars(one).items()
        if field not in plumbing and holds_something(part)
    ]
    print()
    print(f"{{len(held)}} relations that hold something:")
    # Sorted by name and cut short, the list stopped at the cabinets and never reached
    # the island holding its eight drawers, which is the thing worth seeing. The whole
    # list is one screen, so it is printed.
    for whole, field, part in sorted(held):
        def named(one) -> str:
            root = getattr(one, "root", None)
            return str(root.name.name) if root is not None else str(one)

        shown = (
            [named(one) for one in part]
            if isinstance(part, (list, tuple, set))
            else named(part)
        )
        print(f"  {{whole}}.{{field}} = {{shown}}")

    if arguments.view:
        # Smoothing recomputes vertex normals for a room's worth of geometry before the
        # first frame, which is minutes of an apparently black window.
        RayTracer(world=world).scene.show(smooth=False, resolution=(1280, 960))


if __name__ == "__main__":
    main()
'''
"""
The script a run leaves behind so its world can be opened without knowing anything.
"""


def report(evidence: Path, annotated_id: int, split_id: int) -> str:
    """
    Say what the run made, from what its steps wrote.

    A run's numbers are spread over six files and a terminal that has scrolled away, and
    the question asked of a run afterwards is usually how much of it went through rather
    than what any one step said.

    :param evidence: The run's directory.
    :param annotated_id: The world the annotations were written to.
    :param split_id: The world it was built from.
    :return: The report, as Markdown.
    """

    def held(name: str) -> Dict[str, Any]:
        path = evidence / name
        return json.loads(path.read_text()) if path.exists() else {}

    relations = held("relations.json")
    questions = held("questions.json")
    vocabulary = held("vocabulary.json")
    adjudications = held("adjudications.json")
    split = held("split.json")
    classifications = held("classifications.json")

    overlapping = [
        pair for pair in relations.get("pairs", []) if pair.get("shared_faces")
    ]
    labels = vocabulary.get("labels", {})
    bodies = classifications.get("bodies", {})
    answered = adjudications.get("answered", [])

    lines = [
        f"# {evidence.name}",
        "",
        f"- scene: `{relations.get('scene', 'unknown')}`",
        f"- worlds: **{annotated_id}** annotated, {split_id} as it was split",
        f"- models: {vocabulary.get('model', '?')} (vocabulary), "
        f"{adjudications.get('model', '?')} (adjudication), "
        f"{classifications.get('model', '?')} (classification)",
        "",
        "## What was measured",
        "",
        f"- {len(relations.get('segments', []))} labelled objects over "
        f"{len(relations.get('pairs', []))} measurable pairs, {len(overlapping)} of "
        f"them sharing faces",
        f"- {len(questions.get('settled', []))} sets of contested faces the ontology "
        f"settled, {len(questions.get('forced', []))} memberships with only one "
        f"candidate",
        "",
        "## What was asked",
        "",
        f"- {len(labels)} labels, "
        f"{sum(1 for one in labels.values() if one.get('class'))} mapped to a class, "
        f"{sum(1 for one in labels.values() if one.get('is_new_class'))} of them new",
        f"- {sum(1 for one in answered if one['kind'] == 'ownership')} class patterns "
        f"and {sum(1 for one in answered if one['kind'] == 'membership')} memberships "
        f"adjudicated, {sum(1 for one in answered if one.get('problems'))} with problems",
        f"- {len(bodies)} bodies named, "
        f"{len({one['class'] for one in bodies.values() if one.get('class')})} distinct "
        f"classes",
        "",
        "## What was built",
        "",
        f"- {len(split.get('bodies', {}))} bodies, "
        f"{sum(one['faces'] for one in split.get('bodies', {}).values())} faces between "
        f"them, {split.get('still_contested', 0)} faces still claimed twice",
        f"- {len(split.get('pairings', []))} pairings carried past the split",
    ]

    emptied = split.get("emptied", {})
    if emptied:
        lines += ["", f"### {len(emptied)} objects lost every face", ""]
        for name, took in sorted(emptied.items()):
            whom = ", ".join(f"{who} ({count})" for who, count in took.items())
            lines.append(f"- `{name}` -> {whom}")

    if bodies:
        lines += ["", "### Classes given", ""]
        for name, count in Counter(
            one["class"] for one in bodies.values() if one.get("class")
        ).most_common():
            lines.append(f"- {count} x `{name}`")

    lines += [
        "",
        "## Looking at it",
        "",
        "```",
        "python inspect_world.py          # what the world holds",
        "python inspect_world.py --view   # and look at it",
        "```",
        "",
    ]
    return "\n".join(lines)


def build(arguments: argparse.Namespace) -> None:
    """
    Generate what the taxonomy lacks, rebuild the ORM, then annotate and mount.

    :param arguments: The command line arguments.
    """
    evidence = arguments.evidence_directory
    classifications = json.loads((evidence / "classifications.json").read_text())
    vocabulary = json.loads((evidence / "vocabulary.json").read_text())
    wanted = wanted_classes(classifications, vocabulary)
    known = annotation_classes(SemanticAnnotation)
    print(f"{len(wanted)} classes over {len(classifications['bodies'])} bodies")

    generated = generate_missing(wanted, known, evidence)
    if generated:
        print(f"generated {len(generated)}: {', '.join(generated)}")
        print("regenerating the ORM ...")
        failure = regenerate_orm(evidence)
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
