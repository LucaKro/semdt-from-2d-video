#!/usr/bin/env python3
"""
Compare HM3D ground truth labels against pipeline-predicted semantic classes
using semantic embedding similarity for label normalization and bag-of-labels
metrics (Precision, Recall, F1, per-class IoU, mIoU).

This is an alternative to ``compare_gt_vs_predicted.py`` which uses exact
string matching and simple accuracy.  This version:

- Loads predictions from PostgreSQL (persisted SemanticAnnotation class names)
  rather than VLM summary JSON on disk.
- Loads ground truth directly from the HM3D ``.semantic.txt`` file rather than
  re-instantiating an HM3DWorldLoader.
- Normalizes predicted labels to the GT vocabulary using sentence-transformer
  cosine similarity (all-MiniLM-L6-v2) with a configurable threshold.
- Reports Precision, Recall, F1, per-class IoU, and mIoU.

Usage:
    # Evaluate all rooms from a batch output directory
    python compare_gt_vs_predicted_semantic.py \
        --batch-dir batch_output/2026-04-02_181007/00802 \
        --scene-dir datasets/matterport3d/hm3d-minival-semantic-annots-v0.2/00802-wcojb4TFT35

    # Evaluate specific world DB IDs
    python compare_gt_vs_predicted_semantic.py \
        --world-db-id 46 47 48 \
        --scene-dir datasets/matterport3d/hm3d-minival-semantic-annots-v0.2/00802-wcojb4TFT35

    # Custom similarity threshold
    python compare_gt_vs_predicted_semantic.py \
        --batch-dir batch_output/2026-04-02_181007/00802 \
        --scene-dir datasets/matterport3d/hm3d-minival-semantic-annots-v0.2/00802-wcojb4TFT35 \
        --threshold 0.6

    # Save results to CSV
    python compare_gt_vs_predicted_semantic.py \
        --batch-dir batch_output/2026-04-02_181007/00802 \
        --scene-dir datasets/matterport3d/hm3d-minival-semantic-annots-v0.2/00802-wcojb4TFT35 \
        --csv results.csv
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from krrood.ormatic.utils import create_engine
from semantic_digital_twin.orm.ormatic_interface import WorldMappingDAO


# ---------------------------------------------------------------------------
# Embedding-based label similarity
# ---------------------------------------------------------------------------

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embedding_similarity(a: str, b: str) -> float:
    model = _get_model()
    emb = model.encode([a, b])
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])


def _encode_labels(labels: list[str]) -> dict[str, list[float]]:
    """Batch-encode a list of labels, returning {label: embedding}."""
    model = _get_model()
    unique = list(set(labels))
    print(f"  Encoding {len(unique)} unique labels...")
    embeddings = model.encode(unique, show_progress_bar=False)
    return {label: emb for label, emb in zip(unique, embeddings)}


# ---------------------------------------------------------------------------
# Load predicted labels from PostgreSQL
# ---------------------------------------------------------------------------

def _get_engine():
    db_name = os.getenv("PGDATABASE")
    db_user = os.getenv("PGUSER")
    db_password = os.getenv("PGPASSWORD")
    db_host = os.getenv("PGHOST", "localhost")
    db_port = os.getenv("PGPORT", "5432")
    url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(url)


def load_predicted_labels_by_db_id(db_id: int) -> list[str]:
    """Load predicted semantic labels from a persisted world by database_id."""
    engine = _get_engine()
    session = Session(engine)
    query = select(WorldMappingDAO).where(
        WorldMappingDAO.database_id == db_id
    )
    world_dao = session.scalars(query).first()
    if world_dao is None:
        raise RuntimeError(f"No world found with database_id={db_id}")
    world = world_dao.from_dao()
    labels = []
    for ann in world.semantic_annotations:
        # Split CamelCase -> space-separated, then lowercase
        name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", ann.__class__.__name__)
        label = name.lower().replace("_", " ")
        labels.append(label)
    session.close()
    return labels


def load_predicted_labels_from_batch(batch_dir: Path) -> dict[str, list[str]]:
    """Load predicted labels for all rooms from a batch output directory.

    Reads room_db_ids from batch_state.json (in the parent directory) and
    loads each room's world from the DB.

    Returns {room_name: [labels]}.
    """
    # batch_state.json is in the scene's parent batch dir
    state_file = batch_dir.parent / "batch_state.json"
    if not state_file.exists():
        raise FileNotFoundError(f"No batch_state.json found at {state_file}")

    with open(state_file) as f:
        state = json.load(f)

    # Find the scene entry matching this batch_dir name
    scene_id = batch_dir.name
    if scene_id not in state["scenes"]:
        raise KeyError(f"Scene '{scene_id}' not found in {state_file}")

    room_db_ids = state["scenes"][scene_id].get("room_db_ids", {})
    if not room_db_ids:
        raise RuntimeError(f"No room_db_ids found for scene '{scene_id}'")

    result = {}
    for room_name, db_id in sorted(room_db_ids.items()):
        print(f"  Loading {room_name} (db_id={db_id})...")
        result[room_name] = load_predicted_labels_by_db_id(db_id)
    return result


# ---------------------------------------------------------------------------
# Load ground truth labels from HM3D .semantic.txt
# ---------------------------------------------------------------------------

def load_ground_truth_labels(
    scene_dir: Path, room_id: int | None = None,
) -> list[str]:
    """Parse HM3D semantic.txt and return labels, optionally filtered by room."""
    txt_files = list(Path(scene_dir).glob("*.semantic.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No *.semantic.txt found in {scene_dir}"
        )
    txt_file = txt_files[0]

    rooms: dict[int, list[str]] = defaultdict(list)
    with open(txt_file) as f:
        lines = f.readlines()[1:]  # skip header
        for line in lines:
            parts = line.strip().split(",")
            label = parts[2].strip('"').lower()
            rid = int(parts[3])
            rooms[rid].append(label)

    if room_id is not None:
        if room_id not in rooms:
            raise ValueError(
                f"Room {room_id} not found. Available: {sorted(rooms.keys())}"
            )
        return rooms[room_id]

    # All rooms
    labels = []
    for rid in sorted(rooms.keys()):
        labels.extend(rooms[rid])
    return labels


# ---------------------------------------------------------------------------
# Semantic label normalization
# ---------------------------------------------------------------------------

def _words(label: str) -> set[str]:
    """Split a label into its constituent words."""
    return set(label.lower().split())


def _stem(word: str) -> str:
    """Stem a word using NLTK's Snowball stemmer."""
    from nltk.stem import SnowballStemmer
    return SnowballStemmer("english").stem(word)


def _common_prefix_len(a: str, b: str) -> int:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def _lexically_related(a: str, b: str, min_stem_len: int = 3) -> bool:
    """Check if two tokens are lexically related via substring, stem overlap,
    or shared prefix covering at least 80% of the shorter token."""
    if a == b:
        return True
    # Substring containment
    if len(a) >= min_stem_len and a in b:
        return True
    if len(b) >= min_stem_len and b in a:
        return True
    # Long common prefix (covers shelf/shelving where prefix "shel" is 4/5)
    prefix_len = _common_prefix_len(a, b)
    shorter = min(len(a), len(b))
    if shorter >= min_stem_len and prefix_len / shorter >= 0.8:
        return True
    return False


def _shares_word_stem(a: str, b: str, min_stem_len: int = 3) -> bool:
    """Check whether two labels share a common word stem or substring.

    Catches cases like 'shelving unit'/'shelf', 'household box'/'boxes',
    'cooking container'/'container', 'houseplant'/'decorative plant' where
    the embedding similarity is moderate but the lexical overlap makes the
    match obvious.
    """
    words_a = _words(a)
    words_b = _words(b)
    if words_a & words_b:
        return True
    stems_a = {_stem(w) for w in words_a}
    stems_b = {_stem(w) for w in words_b}
    if stems_a & stems_b:
        return True
    # Cross-check all stems and words for lexical relatedness
    tokens_a = words_a | stems_a
    tokens_b = words_b | stems_b
    for ta in tokens_a:
        for tb in tokens_b:
            if _lexically_related(ta, tb, min_stem_len):
                return True
    return False


def normalize_labels(
    predicted: list[str],
    ground_truth: list[str],
    threshold: float = 0.55,
) -> tuple[list[str], list[str], list[dict]]:
    """Remap predicted labels to GT vocabulary using embedding similarity.

    A match is accepted if:
    - embedding similarity > 0.75 (high confidence, no lexical check needed), OR
    - embedding similarity > threshold AND the labels share a word stem.

    Returns (normalized_predicted, ground_truth, normalization_log).
    """
    gt_unique = list(set(ground_truth))

    # Pre-compute embeddings for all unique labels (batch encode is much
    # faster than encoding pairs one at a time).
    pred_unique = [p for p in set(predicted) if p not in gt_unique]
    print(f"  {len(predicted)} predicted labels, {len(gt_unique)} unique GT labels, "
          f"{len(pred_unique)} predicted labels need matching")
    gt_embeddings = _encode_labels(gt_unique)
    pred_embeddings = _encode_labels(pred_unique) if pred_unique else {}

    # Pre-compute best match for each unique predicted label not in GT
    best_matches: dict[str, tuple[str, float]] = {}
    if pred_unique:
        gt_keys = list(gt_embeddings.keys())
        gt_matrix = np.array([gt_embeddings[g] for g in gt_keys])
        for i, p in enumerate(pred_unique):
            p_emb = np.array([pred_embeddings[p]])
            scores = cosine_similarity(p_emb, gt_matrix)[0]
            best_idx = int(np.argmax(scores))
            best_matches[p] = (gt_keys[best_idx], float(scores[best_idx]))

    normalized = []
    log = []

    n_exact = 0
    n_accepted = 0
    n_rejected = 0

    for i, p in enumerate(predicted):
        if p in gt_unique:
            normalized.append(p)
            n_exact += 1
            continue

        best_label, best_score = best_matches[p]

        accepted = (best_score > 0.75
                    or (best_score > threshold
                        and _shares_word_stem(p, best_label)))

        entry = {
            "original": p,
            "matched_to": best_label,
            "similarity": best_score,
            "accepted": accepted,
        }
        log.append(entry)

        if accepted:
            normalized.append(best_label)
            n_accepted += 1
        else:
            normalized.append(p)
            n_rejected += 1

    print(f"  Normalization done: {n_exact} exact, {n_accepted} matched, "
          f"{n_rejected} unmatched")

    return normalized, ground_truth, log


# ---------------------------------------------------------------------------
# Bag-of-labels evaluation metrics
# ---------------------------------------------------------------------------

def evaluate(
    predicted: list[str], ground_truth: list[str],
) -> dict:
    """Compute bag-of-labels Precision, Recall, F1, per-class IoU, mIoU."""
    pred_counter = Counter(predicted)
    gt_counter = Counter(ground_truth)

    true_positives = 0
    for label in pred_counter:
        true_positives += min(pred_counter[label], gt_counter[label])

    false_positives = sum(pred_counter.values()) - true_positives
    false_negatives = sum(gt_counter.values()) - true_positives

    precision = (true_positives / (true_positives + false_positives)
                 if (true_positives + false_positives) else 0.0)
    recall = (true_positives / (true_positives + false_negatives)
              if (true_positives + false_negatives) else 0.0)
    f1_score = (2 * precision * recall / (precision + recall)
                if (precision + recall) else 0.0)

    all_labels = sorted(set(predicted) | set(ground_truth))
    per_class_iou = {}
    for label in all_labels:
        intersection = min(pred_counter[label], gt_counter[label])
        union = pred_counter[label] + gt_counter[label] - intersection
        per_class_iou[label] = intersection / union if union else 0.0

    miou = (sum(per_class_iou.values()) / len(per_class_iou)
            if per_class_iou else 0.0)

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "per_class_iou": per_class_iou,
        "miou": miou,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(
    metrics: dict, norm_log: list[dict], predicted: list[str],
    ground_truth: list[str],
) -> None:
    sep = "=" * 50

    # Normalization log
    if norm_log:
        print(f"\n{sep}")
        print("SEMANTIC NORMALIZATION")
        print(sep)
        for entry in norm_log:
            status = "accepted" if entry["accepted"] else "rejected"
            print(f"  {entry['original']!r} -> {entry['matched_to']!r}  "
                  f"(sim={entry['similarity']:.3f}, {status})")

    # Counts
    print(f"\n{sep}")
    print("EVALUATION RESULTS")
    print(sep)
    print(f"  Predicted objects:    {len(predicted)}")
    print(f"  Ground truth objects: {len(ground_truth)}")
    print(f"\n  True Positives:  {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")

    # Aggregate metrics
    print(f"\n  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score:  {metrics['f1_score']:.3f}")

    # Per-class IoU
    print(f"\n  Per-class IoU:")
    iou = metrics["per_class_iou"]
    label_w = max((len(l) for l in iou), default=0)
    for label in sorted(iou):
        print(f"    {label:<{label_w}}  {iou[label]:.3f}")

    print(f"\n  mIoU: {metrics['miou']:.3f}")


def save_csv(
    metrics: dict, norm_log: list[dict], path: Path,
) -> None:
    """Save per-class IoU and aggregate metrics to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        # Per-class IoU
        writer.writerow(["class", "iou"])
        for label in sorted(metrics["per_class_iou"]):
            writer.writerow([label, f"{metrics['per_class_iou'][label]:.4f}"])
        writer.writerow([])

        # Aggregate metrics
        writer.writerow(["metric", "value"])
        for key in ("precision", "recall", "f1_score", "miou",
                     "true_positives", "false_positives", "false_negatives"):
            writer.writerow([key, metrics[key]])
        writer.writerow([])

        # Normalization log
        if norm_log:
            writer.writerow(["original", "matched_to", "similarity", "accepted"])
            for entry in norm_log:
                writer.writerow([
                    entry["original"], entry["matched_to"],
                    f"{entry['similarity']:.4f}", entry["accepted"],
                ])

    print(f"\nCSV saved to {path}")


def load_csv(path: Path) -> tuple[dict, list[dict], int, int]:
    """Inverse of save_csv: parse a results CSV back into (metrics, norm_log,
    n_predicted, n_ground_truth). Sections are separated by blank rows and
    identified by their header."""
    per_class_iou: dict[str, float] = {}
    aggregate: dict[str, float] = {}
    norm_log: list[dict] = []
    section: str | None = None

    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all(c == "" for c in row):
                section = None
                continue
            if row == ["class", "iou"]:
                section = "iou"
                continue
            if row == ["metric", "value"]:
                section = "metric"
                continue
            if row == ["original", "matched_to", "similarity", "accepted"]:
                section = "norm"
                continue
            if section == "iou":
                per_class_iou[row[0]] = float(row[1])
            elif section == "metric":
                aggregate[row[0]] = float(row[1])
            elif section == "norm":
                norm_log.append({
                    "original": row[0],
                    "matched_to": row[1],
                    "similarity": float(row[2]),
                    "accepted": row[3] in ("True", "true", "1"),
                })

    tp = int(aggregate.get("true_positives", 0))
    fp = int(aggregate.get("false_positives", 0))
    fn = int(aggregate.get("false_negatives", 0))

    metrics = {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": aggregate.get("precision", 0.0),
        "recall": aggregate.get("recall", 0.0),
        "f1_score": aggregate.get("f1_score", 0.0),
        "miou": aggregate.get("miou", 0.0),
        "per_class_iou": per_class_iou,
    }
    return metrics, norm_log, tp + fp, tp + fn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate semantic annotations against HM3D ground truth "
                    "using embedding similarity and bag-of-labels metrics."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--batch-dir", type=Path,
        help="Batch output directory for one scene (e.g. batch_output/2026-04-02_181007/00802). "
             "Reads room_db_ids from batch_state.json in the parent directory.",
    )
    source.add_argument(
        "--world-db-id", type=int, nargs="+",
        help="Database ID(s) of persisted world(s) to evaluate.",
    )
    source.add_argument(
        "--from-csv", type=Path,
        help="Replay statistics from a CSV previously produced by --csv.",
    )
    parser.add_argument(
        "--scene-dir", type=Path,
        help="HM3D semantic annotation directory containing *.semantic.txt. "
             "Not required with --from-csv.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.55,
        help="Cosine similarity threshold for label normalization (default: 0.55).",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Save results to a CSV file.",
    )
    args = parser.parse_args()

    if args.from_csv:
        metrics, norm_log, n_pred, n_gt = load_csv(args.from_csv)
        print_results(metrics, norm_log, [None] * n_pred, [None] * n_gt)
        return

    if args.scene_dir is None:
        parser.error("--scene-dir is required unless --from-csv is used")

    # Collect all predicted labels across rooms
    print("Loading predicted labels from DB...")
    if args.batch_dir:
        room_labels = load_predicted_labels_from_batch(args.batch_dir)
        predicted = []
        for labels in room_labels.values():
            predicted.extend(labels)
    else:
        predicted = []
        for db_id in args.world_db_id:
            print(f"  Loading world db_id={db_id}...")
            predicted.extend(load_predicted_labels_by_db_id(db_id))

    print(f"Loading ground truth from {args.scene_dir}...")
    ground_truth = load_ground_truth_labels(args.scene_dir)

    print(f"Normalizing labels (threshold={args.threshold})...")
    predicted_norm, gt_norm, norm_log = normalize_labels(
        predicted, ground_truth, threshold=args.threshold,
    )

    metrics = evaluate(predicted_norm, gt_norm)

    print_results(metrics, norm_log, predicted_norm, gt_norm)

    if args.csv:
        save_csv(metrics, norm_log, args.csv)


if __name__ == "__main__":
    main()
