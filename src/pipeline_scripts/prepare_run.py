"""
Put the ontology back as it is written, read it out, and make a run a directory.

A run must start from the ontology that is checked in, not from what the last run talked
a model into. Classes generated for one scene and mixins one room argued for would
otherwise be in force for the next, where nothing questions them and nobody remembers
they were ever in doubt::

    python -m pipeline_scripts.prepare_run

It prints the directory the run is to write into. Everything the run then produces goes
there and nowhere else; the only thing it reads from outside is what this step extracts
from the ontology, which is a reading of the ontology rather than a conclusion about a
scene and is therefore the same for every run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import semantic_digital_twin
from semantic_digital_twin.semantic_annotations.in_memory_builder import (
    SemanticAnnotationFilePaths,
)

from pipeline_scripts import run_database
from pipeline_scripts.locations import (
    RUNS_DIRECTORY,
    SHARED_DIRECTORY,
    TAXONOMY,
    new_run_directory,
)

WATCHED = ("semantic_annotations.py", "mixins.py")
"""
The files a run may amend and must not leave amended.
"""


def cram_repository() -> Path:
    """
    :return: The checkout the ontology is written in.
    """
    import semantic_digital_twin

    return Path(semantic_digital_twin.__file__).resolve().parents[3]


def hand_written_changes(repository: Path) -> List[str]:
    """
    Report the ontology's own files a run has been left holding changes to.

    :param repository: The checkout to look in.
    :return: The paths that differ from what is committed, empty when none do.
    """
    finished = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    return [
        line[3:]
        for line in finished.stdout.splitlines()
        if any(line.endswith(name) for name in WATCHED)
    ]


def reset_generated_classes(repository: Path) -> None:
    """
    Empty the classes generated for an earlier scene and rebuild the ORM without them.

    :param repository: The checkout holding the reset script.
    :raises SystemExit: If the reset fails.
    """
    script = Path(__file__).resolve().parents[2] / "scripts" / "reset_class_taxonomy.py"
    finished = subprocess.run(
        [sys.executable, str(script), "--simple"], capture_output=True, text=True
    )
    if finished.returncode != 0:
        raise SystemExit(
            f"the generated classes could not be reset:\n{finished.stderr.strip()[-2000:]}"
        )


def regenerate_orm() -> None:
    """
    Rebuild the ORM from the classes that are left.

    The reset writes the stub and stops there -- it says so, and the documented way to
    use it is to regenerate afterwards. Left out, the ORM goes on naming classes that
    were just removed and the first step to import it dies on a class that is not there.
    It is also the only way to check: the generated interface is gitignored, so a clean
    checkout says nothing at all about whether it matches the ontology.

    :raises SystemExit: If the rebuild fails.
    """
    root = Path(semantic_digital_twin.__file__).parent
    script = root.parent.parent / "scripts" / "generate_orm.py"
    interface = root / "orm" / "ormatic_interface.py"

    # The generator reads the interface it is about to replace, and the one standing
    # there still names the classes just removed -- so it cannot be imported and the
    # rebuild dies on the very staleness it was run to cure. Moved aside, the generator
    # builds from the ontology alone; put back if it fails, so a failure costs nothing.
    aside = interface.with_suffix(".py.aside")
    if interface.exists():
        interface.replace(aside)
    finished = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True
    )
    if finished.returncode != 0 or not interface.exists():
        if aside.exists():
            aside.replace(interface)
        raise SystemExit(
            f"the ORM could not be rebuilt:\n{finished.stderr.strip()[-2000:]}"
        )
    aside.unlink(missing_ok=True)


def prepare_the_schema(run: Path) -> str:
    """
    Make the run's own schema and build the ORM's tables inside it.

    Built in a new interpreter for the same reason the export is: the ORM was just
    rewritten and this one is holding the version from before that. The tables are made
    by the ORM that is about to write to them, so a run never meets a table another run
    left standing.

    :param run: The directory this run writes into.
    :return: The schema it writes into.
    :raises SystemExit: If the schema cannot be made.
    """
    schema = run_database.schema_for(run)
    run_database.create(schema)

    program = (
        "from semantic_digital_twin.orm.ormatic_interface import Base\n"
        "from semantic_digital_twin.orm.utils import "
        "semantic_digital_twin_sessionmaker\n"
        "engine = semantic_digital_twin_sessionmaker()().bind\n"
        "Base.metadata.create_all(bind=engine)\n"
        "print(len(Base.metadata.tables))\n"
    )
    finished = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env={**os.environ, run_database.VARIABLE: run_database.uri_in(schema)},
    )
    if finished.returncode != 0:
        raise SystemExit(
            f"the run's schema could not be built:\n{finished.stderr.strip()[-2000:]}"
        )
    print(f"  {finished.stdout.split()[-1]} tables in schema {schema}")
    return schema


def export_ontology() -> None:
    """
    Write out what every run is allowed to read: the taxonomy as a model reads it.

    Exported in a new interpreter, since the ORM and the generated classes were just
    rewritten and this one imported them before that.

    :raises SystemExit: If the export fails.
    """
    SHARED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    program = (
        "import sys\n"
        "from pathlib import Path\n"
        "from semantic_digital_twin.semantic_annotations.taxonomy_export import "
        "export_taxonomy\n"
        "from semantic_digital_twin.world_description.world_entity import "
        "SemanticAnnotation\n"
        "taxonomy = export_taxonomy(SemanticAnnotation, Path(sys.argv[1]))\n"
        "print(len(taxonomy['classes']), len(taxonomy['part_whole_mixins']))\n"
    )
    finished = subprocess.run(
        [sys.executable, "-c", program, str(TAXONOMY)], capture_output=True, text=True
    )
    if finished.returncode != 0:
        raise SystemExit(
            f"the taxonomy could not be exported:\n{finished.stderr.strip()[-2000:]}"
        )
    classes, mixins = finished.stdout.split()[-2:]
    print(f"  {classes} classes, {mixins} mixins -> {TAXONOMY}")


def build(arguments: argparse.Namespace) -> None:
    """
    Prepare a run.

    :param arguments: The command line arguments.
    """
    repository = cram_repository()

    amended = hand_written_changes(repository)
    if amended and not arguments.ignore_amendments:
        raise SystemExit(
            "the ontology's own files are left amended:\n  "
            + "\n  ".join(amended)
            + "\nA run must start from what is committed. Put them back with:\n"
            "    python -m pipeline_scripts.amend_taxonomy <the run that applied "
            "them> --revert\nor pass --ignore-amendments to run against them anyway."
        )
    if amended:
        print("running against an amended ontology, as asked:")
        for path in amended:
            print(f"  {path}")

    print("emptying the classes generated for an earlier scene ...")
    reset_generated_classes(repository)
    print(f"  {SemanticAnnotationFilePaths.GENERATED_CLASSES_FILE.value}")

    print("rebuilding the ORM without them ...")
    regenerate_orm()

    print("reading the ontology out ...")
    export_ontology()

    directory = new_run_directory(arguments.runs_directory)

    print("making the run its own schema in the database ...")
    prepare_the_schema(directory)

    print(f"\nthe run writes into {directory}")


def main() -> None:
    """
    Prepare a run and say where it is to write.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--runs-directory",
        type=Path,
        default=RUNS_DIRECTORY,
        help="Where to make the run's directory.",
    )
    parser.add_argument(
        "--ignore-amendments",
        action="store_true",
        help="Start even though the ontology's own files are left amended, which is "
        "what a run deliberately made against an amended ontology needs.",
    )
    build(parser.parse_args())


if __name__ == "__main__":
    main()
