"""
Where a run reads from and where it writes to.

A run answers everything from the scene it was given; it never reads what an earlier run
concluded. Two runs of the same scene may disagree, and a later one silently inheriting
half of an earlier one's answers would be neither of them. So every run writes into a
directory of its own and reads nothing outside it -- with one exception.

The exception is what is extracted from the semantic digital twin itself: the taxonomy,
its part-whole relations, which classes are abstract. That is not a conclusion about a
scene, it is a reading of the ontology, and it is the same for every run that starts from
the same ontology. It lives in one place all runs may reach, and is written afresh at the
start of a run, once the ontology has been put back to what is checked in.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
"""
The root of this repository, so paths do not depend on where a run was started from.
"""

SHARED_DIRECTORY = REPOSITORY / "semdt_export"
"""
What was read out of the ontology, which every run may reach.
"""

TAXONOMY = SHARED_DIRECTORY / "taxonomy.json"
"""
The taxonomy as a model reads it: classes, what each can hold, which are abstract.
"""

RUNS_DIRECTORY = REPOSITORY / "pipeline_runs"
"""
Where a run's own directory is made.
"""


def new_run_directory(runs_directory: Path = RUNS_DIRECTORY) -> Path:
    """
    Make a directory for a run to write everything into.

    Named for when it started, so runs sort by age and none can be mistaken for another.

    :param runs_directory: Where to make it.
    :return: The directory, freshly made and empty.
    """
    directory = Path(runs_directory) / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def refuse_to_write_over(directory: Path, overwrite: bool = False) -> None:
    """
    Refuse to write a run's output where output already is, unless told to override it.

    Writing beside it is how a run comes to be part one run and part another: the files
    it does not happen to rewrite stay as the last run left them, and nothing says so.
    Overriding is another matter and is allowed when asked for -- a step run twice in
    one run, the second time knowing something the first did not, writes what it wrote
    again rather than adding to it.

    :param directory: Where the run means to write.
    :param overwrite: Whether what is there is to be written over.
    :raises SystemExit: If anything is there and overwriting was not asked for.
    """
    directory = Path(directory)
    if overwrite or not directory.exists() or not any(directory.iterdir()):
        return
    raise SystemExit(
        f"{directory} already holds output. Write into an empty directory made with:"
        f"\n    python -m pipeline_scripts.prepare_run"
        f"\nor pass --overwrite to write over what is there."
    )
