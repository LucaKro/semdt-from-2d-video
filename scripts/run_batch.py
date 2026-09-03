#!/usr/bin/env python3
"""
Batch runner for the semantic digital twin pipeline.

Discovers all HM3D scenes with semantic annotations and runs
extract_class_structure.py + refine_class_structure.py on each,
with per-scene output isolation, automatic DB ID tracking,
and resumability via a persistent state file.

Output structure:
    batch_output/
      <YYYY-MM-DD_HHMMSS>/           (timestamped experiment directory)
        experiment_metadata.json
        batch_state.json
        batch_results.json
        <scene_id>/
          taxonomy_export/
          taxonomy_snapshot/
            generated_classes.py      (scene-specific generated classes)
            ormatic_interface.py      (scene-specific ORM)
          room<N>/                    (or "all/" when --num-rooms is not used)
            images/
              scene_orig_<pose>.png
              scene_<group>_<pose>__<highlighted bodies>.png
            vlm_responses.json
            vlm_summary.json
            resolution_log.json       (from refine)
            pending_annotations.json  (from refine)
            instantiation_results.json

Usage:
    python scripts/run_batch.py                       # run all scenes
    python scripts/run_batch.py --scenes 00800 00803  # run specific scenes
    python scripts/run_batch.py --resume              # resume latest experiment
    python scripts/run_batch.py --continue-from batch_output/2026-03-30_174927/
    python scripts/run_batch.py --extract-only        # skip refinement
    python scripts/run_batch.py --dry-run             # show what would run
"""

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
DATASET_DIR = REPO_ROOT / "datasets" / "matterport3d"
SEMANTIC_ANNOTS_DIR = DATASET_DIR / "hm3d-minival-semantic-annots-v0.2"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "batch_output"

CRAM_DIR = REPO_ROOT.parent / "cognitive_robot_abstract_machine"
SDT_DIR = CRAM_DIR / "semantic_digital_twin" / "src" / "semantic_digital_twin"
GENERATED_CLASSES_FILE = (
    SDT_DIR / "semantic_annotations" / "generated_classes.py"
)
ORMATIC_INTERFACE_FILE = SDT_DIR / "orm" / "ormatic_interface.py"


def get_git_commit(repo_dir: Path) -> str:
    """Return the current HEAD commit hash for a git repo, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_git_dirty(repo_dir: Path) -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except Exception:
        pass
    return False


def write_experiment_metadata(output_dir: Path, args) -> None:
    """Write a metadata JSON documenting the experiment configuration."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "repos": {
            "semdt_from_2d_video": {
                "path": str(REPO_ROOT),
                "commit": get_git_commit(REPO_ROOT),
                "dirty": get_git_dirty(REPO_ROOT),
            },
            "cognitive_robot_abstract_machine": {
                "path": str(CRAM_DIR),
                "commit": get_git_commit(CRAM_DIR),
                "dirty": get_git_dirty(CRAM_DIR),
            },
        },
        "models": {
            "vlm": {
                "model_id": "qwen/qwen3-vl-30b-a3b-instruct",
                "provider": "openrouter",
                "used_in": "extract_class_structure.py",
            },
            "llm": {
                "model_id": "meta-llama/llama-3.3-70b-instruct",
                "provider": "openrouter",
                "used_in": "refine_class_structure.py (dependency resolution)",
            },
        },
        "command_line": sys.argv,
        "args": {
            "scenes": args.scenes,
            "group_size": args.group_size,
            "num_rooms": args.num_rooms,
            "extract_only": args.extract_only,
            "refine_only": args.refine_only,
            "skip_vlm": args.skip_vlm,
            "render_only": args.render_only,
            "no_prior_labels": getattr(args, "no_prior_labels", False),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = output_dir / "experiment_metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2))
    log.info("Experiment metadata saved to %s", metadata_file)


def save_taxonomy_snapshot(scene_out: Path):
    """Copy generated_classes.py and ormatic_interface.py to the scene output.

    This preserves the taxonomy state for this scene so that the persisted
    DB world can be loaded later with all generated classes available.
    """
    taxonomy_dir = scene_out / "taxonomy_snapshot"
    taxonomy_dir.mkdir(parents=True, exist_ok=True)

    for src in (GENERATED_CLASSES_FILE, ORMATIC_INTERFACE_FILE):
        if src.exists():
            dst = taxonomy_dir / src.name
            shutil.copy2(src, dst)
            log.info("Saved %s -> %s", src.name, dst)


def find_latest_experiment_dir(base_dir: Path) -> Path | None:
    """Find the most recent timestamped experiment directory under base_dir."""
    if not base_dir.exists():
        return None
    candidates = sorted(
        (d for d in base_dir.iterdir()
         if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}_\d{6}", d.name)),
        key=lambda d: d.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def create_experiment_dir(base_dir: Path) -> Path:
    """Create a new timestamped experiment directory under base_dir."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    experiment_dir = base_dir / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def discover_scenes(base_dir: Path) -> list[str]:
    """Return sorted list of scene directory names that have semantic annotations."""
    if not base_dir.exists():
        return []
    return sorted(
        d.name for d in base_dir.iterdir()
        if d.is_dir() and (d / f"{d.name.split('-', 1)[0]}.semantic.glb").exists()
        or list(d.glob("*.semantic.glb"))
    )


def load_state(state_file: Path) -> dict:
    if state_file.exists():
        return json.loads(state_file.read_text())
    return {"scenes": {}}


def save_state(state_file: Path, state: dict):
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2))


def reset_taxonomy() -> bool:
    """Reset generated_classes.py and regenerate ORM to a clean baseline.

    This ensures each scene starts with the hand-authored taxonomy only,
    with no leftover classes from prior scenes.
    """
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "reset_class_taxonomy.py"),
        "--simple",
    ]
    log.info("Resetting taxonomy: %s", " ".join(cmd))
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        log.error("Taxonomy reset failed:\nSTDERR: %s", result.stderr[-2000:])
        return False

    # Regenerate ORM
    generate_orm = CRAM_DIR / "semantic_digital_twin" / "scripts" / "generate_orm.py"
    if generate_orm.exists():
        log.info("Regenerating ORM...")
        result = subprocess.run(
            [sys.executable, str(generate_orm)],
            stdout=sys.stdout, stderr=subprocess.PIPE, text=True,
        )
        if result.returncode != 0:
            log.error("ORM regeneration failed:\nSTDERR: %s", result.stderr[-2000:])
            return False
    else:
        log.warning("ORM generation script not found at %s, skipping", generate_orm)

    log.info("Taxonomy reset complete")
    return True


def reconstruct_state_from_disk(output_dir: Path) -> dict:
    """Rebuild batch state by inspecting the filesystem.

    Used by --continue-from when batch_state.json is missing or incomplete
    (e.g. after a Ctrl-C).  For each scene directory found, checks:
      - Whether rooms have vlm_summary.json  -> extract VLM queries completed
      - Whether a combined vlm_responses file exists
      - Whether rooms have instantiation_results.json -> refine completed
    """
    state: dict = {"scenes": {}}
    if not output_dir.exists():
        return state

    for scene_dir in sorted(output_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_id = scene_dir.name
        # Skip non-scene directories
        if not scene_id.isdigit() and not scene_id[0].isdigit():
            continue

        scene_state: dict = {}

        # Check rooms for extract output
        room_dirs = []
        for d in sorted(scene_dir.iterdir()):
            if d.is_dir() and d.name not in ("taxonomy_export",):
                if (d / "vlm_summary.json").exists():
                    room_dirs.append(d)

        if room_dirs:
            scene_state["extract_vlm_done"] = True
            # Check if DB persistence completed (combined file is written
            # after VLM but before DB persist, so its presence is a hint
            # but not proof of DB persist)
            combined = scene_dir / "vlm_responses_combined.json"
            scene_state["has_combined_json"] = combined.exists()

            # Read per-room DB IDs if available
            room_db_ids_file = scene_dir / "room_db_ids.json"
            if room_db_ids_file.exists():
                with open(room_db_ids_file) as f:
                    scene_state["room_db_ids"] = json.load(f)
                scene_state["extract_status"] = "done"

            # Check per-room refine status
            room_states = {}
            for rd in room_dirs:
                room_name = rd.name
                if (rd / "instantiation_results.json").exists():
                    room_states[room_name] = {"refine_status": "done"}
                else:
                    room_states[room_name] = {"refine_status": "pending"}
            scene_state["rooms"] = room_states

        state["scenes"][scene_id] = scene_state
        log.info("Disk state for %s: vlm_done=%s, combined=%s, rooms=%s",
                 scene_id,
                 scene_state.get("extract_vlm_done", False),
                 scene_state.get("has_combined_json", False),
                 {k: v["refine_status"] for k, v in scene_state.get("rooms", {}).items()})

    return state


def reconstruct_combined_json(scene_out: Path) -> Path:
    """Rebuild the combined vlm_responses JSON from per-room files.

    Needed when extract was interrupted after writing per-room output
    but before writing the combined file.
    """
    combined_path = scene_out / "vlm_responses_combined.json"
    all_responses = []

    for room_dir in sorted(scene_out.iterdir()):
        if not room_dir.is_dir() or room_dir.name == "taxonomy_export":
            continue
        room_responses_file = room_dir / "vlm_responses.json"
        if room_responses_file.exists():
            with open(room_responses_file) as f:
                all_responses.extend(json.load(f))

    with open(combined_path, "w") as f:
        json.dump(all_responses, f, indent=2)

    log.info("Reconstructed combined JSON with %d groups at %s",
             len(all_responses), combined_path)
    return combined_path


def discover_room_dirs(scene_out: Path) -> list[Path]:
    """Find room subdirectories that contain a vlm_summary.json."""
    room_dirs = []
    for d in sorted(scene_out.iterdir()):
        if d.is_dir() and (d / "vlm_summary.json").exists():
            room_dirs.append(d)
    return room_dirs


def run_extract(
    scene_dir: Path,
    scene_out: Path,
    extra_args: list[str],
) -> tuple[bool, dict[str, int] | None]:
    """
    Run extract_class_structure.py for one scene.
    Returns (success, room_db_ids mapping {room_tag: database_id}).
    """
    # The combined output file goes at the scene level;
    # per-room files are written by extract into room subdirs.
    output_json = scene_out / "vlm_responses_combined.json"

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "extract_class_structure.py"),
        str(scene_dir),
        str(output_json),
        "--dataset", "hm3d",
        "--output-dir", str(scene_out),
        "--headless",
        *extra_args,
    ]

    log.info("Running: %s", " ".join(cmd))

    # Stream combined stdout+stderr to terminal. Merging into one pipe
    # avoids a deadlock where a full stderr buffer blocks the child while
    # the parent is only draining stdout.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()

    if proc.returncode != 0:
        log.error("Extract failed for %s (exit %d)", scene_dir.name, proc.returncode)
        return False, None

    # Read per-room database IDs from the mapping file written by extract
    room_db_ids_file = scene_out / "room_db_ids.json"
    room_db_ids = None
    if room_db_ids_file.exists():
        with open(room_db_ids_file) as f:
            room_db_ids = json.load(f)
        log.info("Extract complete for %s (room_db_ids=%s)", scene_dir.name, room_db_ids)
    else:
        log.warning("No room_db_ids.json found for %s", scene_dir.name)

    return True, room_db_ids


def run_refine(
    summary_json: Path,
    world_db_id: int,
    extra_args: list[str],
) -> bool:
    """Run refine_class_structure.py for one room's summary."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "refine_class_structure.py"),
        str(summary_json),
        str(world_db_id),
        "--dataset", "hm3d",
        *extra_args,
    ]

    log.info("Running: %s", " ".join(cmd))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()

    if proc.returncode != 0:
        log.error("Refine failed for %s (exit %d)",
                  summary_json.parent.name, proc.returncode)
        return False

    log.info("Refine complete for %s (world_database_id=%d)",
             summary_json.parent.name, world_db_id)
    return True


def run_persist(
    pending_jsons: list[Path],
    room_db_ids_file: Path,
    dataset: str = "hm3d",
) -> bool:
    """Run persist_annotations.py once with all pending annotation files.

    Uses --room-db-ids so each room's annotations are persisted against
    the correct world, while ORM regeneration happens only once.
    """
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "persist_annotations.py"),
        *[str(p) for p in pending_jsons],
        "--room-db-ids", str(room_db_ids_file),
        "--dataset", dataset,
    ]

    log.info("Running: %s", " ".join(cmd))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()

    if proc.returncode != 0:
        log.error("Persist failed (exit %d)", proc.returncode)
        return False

    log.info("Persist complete")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for the semantic digital twin pipeline"
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        help="Scene IDs to process (e.g. 00800 00803). Default: all available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base output directory (default: batch_output/)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved batch_state.json, skipping already-completed stages",
    )
    parser.add_argument(
        "--continue-from",
        type=Path,
        default=None,
        metavar="DIR",
        help="Resume from a previous output directory by inspecting what exists "
             "on disk. Unlike --resume, does not require batch_state.json. "
             "Reconstructs state from per-room output files.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only run extraction, skip refinement",
    )
    parser.add_argument(
        "--refine-only",
        action="store_true",
        help="Only run refinement (requires prior extraction state)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="Objects per VLM group (passed to extract)",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Pass --skip_vlm to extract (reparse existing output)",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Pass --render-only to extract",
    )
    parser.add_argument(
        "--num-rooms",
        type=int,
        default=None,
        help="Pass --num-rooms N to extract (HM3D room batching)",
    )
    parser.add_argument(
        "--no-prior-labels",
        action="store_true",
        help="Pass --no-prior-labels to extract (don't leak GT labels to VLM)",
    )
    args = parser.parse_args()

    # Discover scenes
    all_scenes = discover_scenes(SEMANTIC_ANNOTS_DIR)
    if not all_scenes:
        log.error("No scenes found in %s", SEMANTIC_ANNOTS_DIR)
        sys.exit(1)

    if args.scenes:
        # Filter to requested scenes (match by prefix)
        selected = []
        for req in args.scenes:
            matches = [s for s in all_scenes if s.startswith(req)]
            if not matches:
                log.error("No scene matching '%s' found. Available: %s", req, all_scenes)
                sys.exit(1)
            selected.extend(matches)
        scenes = sorted(set(selected))
    else:
        scenes = all_scenes

    log.info("Scenes to process: %s", scenes)

    # Determine experiment directory
    base_output_dir = args.output_dir
    if args.continue_from:
        # Use the specified directory directly as the experiment dir
        experiment_dir = args.continue_from
        state = reconstruct_state_from_disk(experiment_dir)
        log.info("Reconstructed state from %s: %d scene(s)",
                 experiment_dir, len(state["scenes"]))
    elif args.resume:
        # Find the latest timestamped experiment dir
        experiment_dir = find_latest_experiment_dir(base_output_dir)
        if experiment_dir is None:
            log.error("No experiment directory found under %s to resume",
                      base_output_dir)
            sys.exit(1)
        log.info("Resuming from %s", experiment_dir)
        state = load_state(experiment_dir / "batch_state.json")
    else:
        # Create a new timestamped experiment dir
        experiment_dir = create_experiment_dir(base_output_dir)
        state = {"scenes": {}}

    state_file = experiment_dir / "batch_state.json"
    log.info("Experiment directory: %s", experiment_dir)

    # Write experiment metadata before any processing
    if not args.dry_run:
        write_experiment_metadata(experiment_dir, args)

    # Build extra args for extract
    extract_extra = ["--group-size", str(args.group_size)]
    if args.skip_vlm:
        extract_extra.append("--skip_vlm")
    if args.render_only:
        extract_extra.append("--render-only")
    if args.num_rooms is not None:
        extract_extra.extend(["--num-rooms", str(args.num_rooms)])
    if args.no_prior_labels:
        extract_extra.append("--no-prior-labels")

    # Build extra args for refine
    # Always skip in-process persistence; we do one combined persist after
    # all rooms finish to avoid regenerating the ORM per room.
    refine_extra = ["--skip-persist"]

    results = {"total": len(scenes), "extracted": 0, "refined": 0, "failed": []}

    for scene_name in scenes:
        scene_dir = SEMANTIC_ANNOTS_DIR / scene_name
        scene_id = scene_name.split("-")[0]  # e.g. "00800"
        scene_out = experiment_dir / scene_id
        scene_out.mkdir(parents=True, exist_ok=True)

        scene_state = state["scenes"].get(scene_id, {})

        log.info("=" * 60)
        log.info("Processing scene: %s", scene_name)
        log.info("=" * 60)

        if args.dry_run:
            log.info("[DRY RUN] Would reset taxonomy")
            log.info("[DRY RUN] Would extract %s -> %s", scene_dir, scene_out)
            if not args.extract_only:
                log.info("[DRY RUN] Would refine each room in %s", scene_out)
            continue

        # --- Reset taxonomy for scene isolation ---
        # Skip reset in refine-only mode: the generated classes from extraction
        # must remain in place so SQLAlchemy can load the persisted world.
        if not args.refine_only:
            if not reset_taxonomy():
                log.error("Failed to reset taxonomy before scene %s, skipping", scene_id)
                results["failed"].append({"scene": scene_id, "stage": "reset"})
                continue

        # --- Extraction ---
        if not args.refine_only:
            already_done = (
                (args.resume or args.continue_from)
                and scene_state.get("extract_status") == "done"
                and scene_state.get("room_db_ids") is not None
            )
            vlm_done_but_no_db = (
                args.continue_from
                and scene_state.get("extract_vlm_done")
                and scene_state.get("room_db_ids") is None
            )

            if already_done:
                log.info("Skipping extraction (already done, room_db_ids=%s)",
                         scene_state.get("room_db_ids"))
            elif vlm_done_but_no_db:
                # VLM output exists on disk but world was never persisted
                # (e.g. Ctrl-C during DB write). Reconstruct combined JSON
                # if needed, then re-run extract with --skip_vlm to persist.
                log.info("VLM output exists for %s but no DB ID — "
                         "re-running extract with --skip_vlm to persist world",
                         scene_id)
                if not scene_state.get("has_combined_json"):
                    reconstruct_combined_json(scene_out)

                replay_args = [
                    a for a in extract_extra
                    if a not in ("--skip_vlm", "--render-only")
                ]
                replay_args.append("--skip_vlm")
                ok, room_db_ids = run_extract(scene_dir, scene_out, replay_args)

                scene_state["extract_status"] = "done" if ok else "failed"
                scene_state["room_db_ids"] = room_db_ids
                scene_state["extract_time"] = datetime.now().isoformat()
                # Clear disk-only keys now that we have proper state
                scene_state.pop("extract_vlm_done", None)
                scene_state.pop("has_combined_json", None)

                state["scenes"][scene_id] = scene_state
                save_state(state_file, state)

                if not ok:
                    results["failed"].append({"scene": scene_id, "stage": "extract"})
                    continue

                results["extracted"] += 1
            else:
                ok, room_db_ids = run_extract(scene_dir, scene_out, extract_extra)

                scene_state["extract_status"] = "done" if ok else "failed"
                scene_state["room_db_ids"] = room_db_ids
                scene_state["extract_time"] = datetime.now().isoformat()

                state["scenes"][scene_id] = scene_state
                save_state(state_file, state)

                if not ok:
                    results["failed"].append({"scene": scene_id, "stage": "extract"})
                    continue

                results["extracted"] += 1

            # Save taxonomy snapshot after extraction so --refine-only can
            # restore it later without needing to re-run extraction.
            save_taxonomy_snapshot(scene_out)
        else:
            extract_done = (
                scene_state.get("extract_status") == "done"
                or scene_state.get("extract_vlm_done")
            )
            if not extract_done:
                log.error("Cannot refine %s: extraction not completed", scene_id)
                results["failed"].append(
                    {"scene": scene_id, "stage": "refine", "reason": "no extraction"})
                continue

        # --- Refinement (per room) ---
        if not args.extract_only:
            room_db_ids = scene_state.get("room_db_ids")
            if not room_db_ids:
                log.error("No room_db_ids for scene %s, cannot refine", scene_id)
                results["failed"].append(
                    {"scene": scene_id, "stage": "refine", "reason": "no db_id"})
                continue

            room_dirs = discover_room_dirs(scene_out)
            if not room_dirs:
                log.error("No room directories with vlm_summary.json found in %s", scene_out)
                results["failed"].append(
                    {"scene": scene_id, "stage": "refine", "reason": "no room summaries"})
                continue

            # Track per-room refine status
            room_states = scene_state.get("rooms", {})
            all_rooms_ok = True

            for room_dir in room_dirs:
                room_name = room_dir.name
                room_state = room_states.get(room_name, {})

                if (args.resume or args.continue_from) and room_state.get("refine_status") == "done":
                    log.info("Skipping refinement for %s/%s (already done)",
                             scene_id, room_name)
                    continue

                # Look up the per-room DB ID; fall back to first available if
                # the mapping uses a single entry (e.g. Warsaw / single-room).
                room_db_id = room_db_ids.get(room_name)
                if room_db_id is None:
                    # Fallback: if only one world was persisted (e.g. "all"),
                    # use that for every room.
                    if len(room_db_ids) == 1:
                        room_db_id = next(iter(room_db_ids.values()))
                    else:
                        log.error("No database_id for %s/%s (available: %s)",
                                  scene_id, room_name, list(room_db_ids.keys()))
                        room_state["refine_status"] = "failed"
                        room_states[room_name] = room_state
                        all_rooms_ok = False
                        continue

                summary_json = room_dir / "vlm_summary.json"
                ok = run_refine(summary_json, room_db_id, refine_extra)

                room_state["refine_status"] = "done" if ok else "failed"
                room_state["refine_time"] = datetime.now().isoformat()
                room_states[room_name] = room_state

                if not ok:
                    all_rooms_ok = False

            scene_state["rooms"] = room_states
            scene_state["refine_status"] = "done" if all_rooms_ok else "partial"
            state["scenes"][scene_id] = scene_state
            save_state(state_file, state)

            if not all_rooms_ok:
                results["failed"].append({"scene": scene_id, "stage": "refine"})
                continue

            # --- Persistence (single process, per-room worlds) ---
            # Collect pending_annotations.json from all rooms that succeeded
            pending_jsons = []
            for room_dir in room_dirs:
                pa = room_dir / "pending_annotations.json"
                if pa.exists():
                    pending_jsons.append(pa)

            room_db_ids_file = scene_out / "room_db_ids.json"

            if pending_jsons and room_db_ids_file.exists():
                log.info("Persisting annotations from %d rooms for scene %s",
                         len(pending_jsons), scene_id)
                ok = run_persist(pending_jsons, room_db_ids_file)
                if ok:
                    results["refined"] += 1
                    scene_state["persist_status"] = "done"
                else:
                    results["failed"].append({"scene": scene_id, "stage": "persist"})
                    scene_state["persist_status"] = "failed"
                state["scenes"][scene_id] = scene_state
                save_state(state_file, state)
            else:
                log.warning("No pending_annotations.json files found for %s", scene_id)

            # Save the accumulated taxonomy (generated_classes.py + ORM)
            # so this scene's DB world can be loaded independently later
            save_taxonomy_snapshot(scene_out)

    # Final summary
    log.info("=" * 60)
    log.info("BATCH COMPLETE")
    log.info("=" * 60)
    log.info("Total scenes: %d", results["total"])
    log.info("Extracted: %d", results["extracted"])
    log.info("Refined: %d", results["refined"])
    if results["failed"]:
        log.warning("Failed: %s", results["failed"])

    # Save final results
    if not args.dry_run:
        results_file = experiment_dir / "batch_results.json"
        results_file.write_text(json.dumps(results, indent=2))
        log.info("Results saved to %s", results_file)


if __name__ == "__main__":
    main()