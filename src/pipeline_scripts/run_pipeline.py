"""
Run the whole pipeline: a labelled scan in, an annotated world out.

    python -m pipeline_scripts.run_pipeline

That is the whole invocation. Everything a run can be told is in the block of settings
below -- change them here and run the file again; nothing needs to be passed on the
command line. Each step is also a script of its own, and this runs them in order, so
anything that can be done here can be done a step at a time when a step needs looking at.

Every step writes into one directory made for this run, named for when it started, and
reads nothing another run concluded. Two runs of the same scene may reach different
answers; neither should quietly inherit half of the other.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from pipeline_scripts.models import Model

# %% ── settings ──────────────────────────────────────────────────────────────────────


@dataclass
class Settings:
    """
    Everything a run can be told. Change these and run the file again.
    """

    scene: Path = Path("dataset/kitchenlab_new_mesh_agreement_dataset")
    """
    The directory holding the scene's labelled mesh.
    """

    model: Model = Model.QWEN3_VL_30B
    """
    Which model every question goes to. See :mod:`pipeline_scripts.models`.
    """

    render_resolution: Tuple[int, int] = (1024, 768)
    """
    How large the pictures a model is shown are drawn.
    """

    deciding_resolution: Optional[Tuple[int, int]] = None
    """
    How large to draw the renders made only to choose a viewpoint and then thrown away.
    ``(256, 192)`` is a sixteenth of the pixels; None draws them full size.
    """

    viewpoint_choice: Optional[str] = "alone"
    """
    How to pick the one viewpoint a question is shown from. ``alone`` measures what is
    visible of the objects by themselves and takes seconds; ``in-room`` measures it in
    the scene, which counts what stands in front of them and takes minutes per region;
    None keeps all four viewpoints.
    """

    group_size: int = 8
    """
    How many bodies are painted and named at once in the classification step.
    """

    nearest: int = 5
    """
    How many nearest neighbours each object's evidence reaches for.
    """

    corrections: int = 1
    """
    How often an unusable answer is put back to the model with what was wrong with it.
    """

    headless: bool = True
    """
    Whether to render without opening a window. False shows the renders as they are made.
    """

    persist: bool = True
    """
    Whether to write the worlds to the database. Without it the run stops at the split's
    report, since everything after it reads a world back.
    """

    ask_about_the_ontology: bool = False
    """
    Whether to ask whether the taxonomy itself is missing a relation -- that a countertop
    can have drawers, say. Off by default: it proposes changes to the ontology every
    later scene would inherit.
    """

    amend_the_ontology: bool = False
    """
    Whether to carry those out for the length of this run. The edits are put back when
    the run ends; they are never committed. Needs
    :attr:`ask_about_the_ontology`.
    """

    runs_directory: Optional[Path] = None
    """
    Where to make the run's directory, defaulting to ``pipeline_runs``.
    """


SETTINGS = Settings()
"""
What this run uses. Edit the defaults above, or assign to this before calling main().
"""

# %% ── the run ───────────────────────────────────────────────────────────────────────


@dataclass
class Step:
    """
    One step of the pipeline, as it is run.
    """

    name: str
    """
    What to call it while it runs.
    """

    arguments: List[str]
    """
    What to run, after ``python -m``.
    """

    optional: bool = field(default=False)
    """
    Whether the run carries on when this step fails.
    """


def steps(settings: Settings, run: Path) -> List[Step]:
    """
    Work out what this run is to do, in order.

    :param settings: What the run was told.
    :param run: The directory it writes into.
    :return: The steps to run.
    """
    scene, model, run = str(settings.scene), str(settings.model), str(run)
    headless = ["--headless"] if settings.headless else []
    resolution = ["--resolution", *map(str, settings.render_resolution)]
    deciding = (
        ["--deciding-resolution", *map(str, settings.deciding_resolution)]
        if settings.deciding_resolution
        else []
    )
    viewpoint = (
        ["--best-viewpoint", settings.viewpoint_choice]
        if settings.viewpoint_choice
        else []
    )
    corrections = ["--corrections", str(settings.corrections)]

    measure = [
        "pipeline_scripts.build_relation_evidence", scene,
        "--output-dir", run, "--nearest", str(settings.nearest),
        *resolution, *deciding, *viewpoint, *headless,
    ]

    planned = [
        Step("measure the scene", [*measure, "--exemplar-renders"]),
        Step("map the labels onto classes",
             ["pipeline_scripts.map_label_vocabulary", run, "--model", model,
              *corrections]),
        Step("measure again, knowing the classes",
             [*measure, "--vocabulary", f"{run}/vocabulary.json",
              "--question-renders", "1000", "--overwrite"]),
    ]

    if settings.ask_about_the_ontology:
        planned.append(
            Step("ask whether the ontology is missing a relation",
                 ["pipeline_scripts.amend_taxonomy", run, "--model", model]
                 + (["--apply"] if settings.amend_the_ontology else []),
                 optional=True)
        )

    planned += [
        Step("adjudicate what is left open",
             ["pipeline_scripts.adjudicate_overlaps", run, "--model", model,
              *corrections]),
        Step("cut the scene into bodies",
             ["pipeline_scripts.split_scene", scene, run]
             + (["--persist"] if settings.persist else [])),
    ]

    if not settings.persist:
        return planned

    planned += [
        Step("name each body",
             ["pipeline_scripts.classify_bodies", scene, run, "--model", model,
              "--group-size", str(settings.group_size), *corrections, *headless]),
        Step("annotate the bodies and mount the parts",
             ["pipeline_scripts.annotate_and_mount", run]),
    ]
    if settings.amend_the_ontology:
        planned.append(
            Step("put the ontology back as it was written",
                 ["pipeline_scripts.amend_taxonomy", run, "--revert"])
        )
    return planned


def run_step(step: Step, number: int, of: int) -> bool:
    """
    Run one step, showing what it says as it says it.

    :param step: The step to run.
    :param number: Which step this is, counting from one.
    :param of: How many there are.
    :return: Whether it succeeded.
    """
    print(f"\n{'─' * 78}\n{number}/{of}  {step.name}\n{'─' * 78}", flush=True)
    finished = subprocess.run([sys.executable, "-m", *step.arguments])
    if finished.returncode == 0:
        return True
    print(
        f"\n{step.name} failed"
        + (", carrying on" if step.optional else ", so the run stops here"),
        flush=True,
    )
    return step.optional


def main(settings: Settings = SETTINGS) -> None:
    """
    Prepare a run and carry it out.

    :param settings: What the run is told.
    :raises SystemExit: If a step the run depends on fails.
    """
    prepare = ["pipeline_scripts.prepare_run"] + (
        ["--runs-directory", str(settings.runs_directory)]
        if settings.runs_directory
        else []
    )
    print(f"{'═' * 78}\npreparing\n{'═' * 78}", flush=True)
    prepared = subprocess.run(
        [sys.executable, "-m", *prepare], capture_output=True, text=True
    )
    print(prepared.stdout.strip() or prepared.stderr.strip()[-2000:], flush=True)
    if prepared.returncode != 0:
        raise SystemExit("the run could not be prepared")

    run = Path(prepared.stdout.split()[-1])
    planned = steps(settings, run)
    print(f"\n{settings.model.name} over {len(planned)} steps into {run.name}", flush=True)

    for number, step in enumerate(planned, start=1):
        if not run_step(step, number, len(planned)):
            raise SystemExit(f"stopped after {number - 1} of {len(planned)} steps")

    print(f"\n{'═' * 78}", flush=True)
    for made in ("report.md", "inspect_world.py"):
        if (run / made).exists():
            print(f"  {run / made}")
    print(f"{'═' * 78}")


if __name__ == "__main__":
    main()
