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
import subprocess
import sys
from pathlib import Path
from typing import List

from semantic_digital_twin.semantic_annotations.in_memory_builder import (
    SemanticAnnotationFilePaths,
)

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

    print("reading the ontology out ...")
    export_ontology()

    directory = new_run_directory(arguments.runs_directory)
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
