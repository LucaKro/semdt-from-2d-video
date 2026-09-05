# Semantic Digital Twin from 2D Video

Constructs a semantic digital twin from 2D scene imagery using VLM-based object classification and ontology-driven annotation.

What this repository holds is the HM3D flow: render a scene, ask a model what the objects
are, resolve the classes and persist the world.

**The labelled-mesh pipeline lives elsewhere.** Turning one labelled PLY scan of a room
into an annotated, hierarchical world -- measuring how the labels overlap, asking which
class each label means, deciding whose a contested face is, cutting the mesh into bodies
and mounting the parts into their wholes -- moved into the CRAM workspace, at
`cognitive_robot_abstract_machine/experiments/src/experiments/warsaw/pipeline/`. It is run
with `python -m experiments.warsaw.pipeline.pipeline` and takes no arguments; everything a
run can be told is a field of `PipelineSettings`. Its scenes and its output live beside it
and are not committed.

## Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/)
- PostgreSQL (for world persistence)
- Access to the [cognitive_robot_abstract_machine](git@github.com:Sanic/cognitive_robot_abstract_machine.git) repository

## Installation

### 1. Set up PostgreSQL

The pipeline persists worlds and semantic annotations to a PostgreSQL database. Tables are created automatically on first run via SQLAlchemy's `Base.metadata.create_all()`.

**Create the database and user:**

```bash
# Connect as a superuser (or ask your DB admin)
psql -U postgres

# Inside psql:
CREATE USER semdt WITH PASSWORD 'semdt';
CREATE DATABASE semdt_db OWNER semdt;
GRANT ALL PRIVILEGES ON DATABASE semdt_db TO semdt;
\q
```

**Verify the connection:**

```bash
psql -h localhost -U semdt -d semdt_db -c "SELECT 1;"
```

The scripts reach the database through `semantic_digital_twin`'s own session maker, which reads a single connection URI from `SEMANTIC_DIGITAL_TWIN_DATABASE_URI` (see step 4 below).

**Note:** the URI must name the driver as `postgresql+psycopg://`. A bare `postgresql://` selects psycopg2, which this workspace does not install.

### 2. Install the CRAM dependency

This project depends on packages from the CRAM monorepo (`semantic_digital_twin`, `krrood`, etc.). Install it first:

```bash
git clone git@github.com:Sanic/cognitive_robot_abstract_machine.git
cd cognitive_robot_abstract_machine
git checkout semdt-creation-from-video

python3 -m venv cram-env
source cram-env/bin/activate
pip install poetry
poetry install
```

### 3. Install this package

With the same virtual environment activated:

```bash
cd semdt-from-2d-video
pip install -e .
```

### 4. Configure environment variables

The scripts require database credentials and an API key:

```bash
export SEMANTIC_DIGITAL_TWIN_DATABASE_URI=postgresql+psycopg://<user>:<password>@localhost:5432/<database>
export OPENROUTER_API_KEY=<your_key>  # for VLM queries
```

## Dataset: HM3D Semantics v0.2

This project uses the [Habitat-Matterport 3D (HM3D)](https://aihabitat.org/datasets/hm3d-semantics/) dataset. Each scene is provided as three parallel asset bundles:

| Directory | Files | Purpose |
|---|---|---|
| `hm3d-minival-glb-v0.2/` | `<scene>.glb` | Visual mesh (textured) |
| `hm3d-minival-habitat-v0.2/` | `<scene>.basis.glb` + `<scene>.basis.navmesh` | Habitat simulator mesh + navigation mesh |
| `hm3d-minival-semantic-annots-v0.2/` | `<scene>.semantic.glb` + `<scene>.semantic.txt` | Semantic mesh + label lookup table |

You can download the dataset [here](https://nc.uni-bremen.de/index.php/s/mQpn5wWDsmSzZEJ).

### How semantic annotations work

The **`.semantic.glb`** contains the same geometry as the scene mesh, but every face is painted a flat color encoding which object it belongs to. The **`.semantic.txt`** is the lookup table mapping those colors to labels:

```
object_id, hex_color, label,      room_id
1,         97C517,    "ceiling",   1
16,        1C9E7F,    "bed",       1
51,        067FB0,    "toilet",    2
```

### Extracting per-object meshes

Load the semantic GLB, read per-face colors, and map them through the txt file:

```python
import trimesh
from pathlib import Path

scene_id = "TEEsavR23oF"
base = Path("datasets/matterport3d")

# 1. Parse the lookup table
annotations = {}
txt_path = base / f"hm3d-minival-semantic-annots-v0.2/00800-{scene_id}/{scene_id}.semantic.txt"
for line in txt_path.read_text().splitlines()[1:]:  # skip header
    parts = line.split(",")
    obj_id, hex_color, label, room_id = int(parts[0]), parts[1], parts[2].strip('"'), int(parts[3])
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    annotations[(r, g, b)] = {"id": obj_id, "label": label, "room_id": room_id}

# 2. Load the semantic GLB
semantic_glb = base / f"hm3d-minival-semantic-annots-v0.2/00800-{scene_id}/{scene_id}.semantic.glb"
scene = trimesh.load(str(semantic_glb))

# 3. Map face colors to labels
for name, geom in scene.geometry.items():
    if not isinstance(geom, trimesh.Trimesh):
        continue
    face_colors = geom.visual.face_colors[:, :3]
    for face_idx, color in enumerate(face_colors):
        key = tuple(color)
        if key in annotations:
            ann = annotations[key]
            # ann["label"] -> "bed", "wall", "toilet", ...
            # ann["id"]    -> object instance id
            # ann["room_id"] -> room it belongs to
```

Faces sharing the same annotation can be grouped to extract individual object submeshes.

Note: only a subset of minival scenes have semantic annotations (00800, 00802, 00803, 00808). The remaining scenes have visual/navigation meshes but no annotation layer.

## Usage

The pipeline has three stages, run per scene (and per room, when scenes are split by HM3D rooms):

1. **Extract** — render scenes, query the VLM, persist a bare `World` to the DB.
2. **Refine** — for each VLM-classified object, resolve constructor-field dependencies and instantiate `SemanticAnnotation` subclasses, generating new annotation classes on the fly.
3. **Persist** — regenerate the ORM once for all newly generated classes and write the annotated worlds back to the DB.

### Recommended: orchestrated batch runs

`run_batch.py` is the entry point. It discovers all annotated HM3D scenes, runs all three stages with per-scene output isolation, tracks DB IDs across stages, and is resumable.

```bash
# Run every scene with semantic annotations (00800, 00802, 00803, 00808)
python scripts/run_batch.py

# Run specific scenes
python scripts/run_batch.py --scenes 00800 00803

# Resume the latest experiment, skipping completed stages
python scripts/run_batch.py --resume

# Resume a specific experiment dir by inspecting what's on disk
python scripts/run_batch.py --continue-from batch_output/2026-04-28_223640/

# Limit to N rooms per scene (HM3D); useful for quick iteration
python scripts/run_batch.py --num-rooms 2

# Withhold HM3D ground-truth body names from the VLM prompt
python scripts/run_batch.py --no-prior-labels

# Other useful flags
#   --extract-only       skip refine + persist
#   --refine-only        re-run refine (requires prior extract state)
#   --skip-vlm           reparse cached VLM responses
#   --render-only        only render images, no VLM
#   --group-size N       objects per VLM group (default 8)
#   --dry-run            show planned commands without executing
```

Each run creates a timestamped directory `batch_output/<YYYY-MM-DD_HHMMSS>/` containing `experiment_metadata.json` (commit hashes, model IDs, args), `batch_state.json` (resume state), and per-scene subdirectories with `taxonomy_export/`, per-room images, `vlm_summary.json`, `pending_annotations.json`, and `instantiation_results.json`.

### Resetting the taxonomy between experiments

`refine_class_structure.py` modifies the SDK's `generated_classes.py` and ORM. Reset both to a clean baseline before a fresh run:

```bash
./scripts/reset_taxonomy.sh
# or, equivalently:
python scripts/reset_class_taxonomy.py --simple
python ../cognitive_robot_abstract_machine/semantic_digital_twin/scripts/generate_orm.py
```

`run_batch.py` invokes the reset automatically at the start of each new (non-resumed) run.

### Manual single-scene flow

The same three stages can be run by hand. Use this when iterating on a single scene or debugging a stage in isolation.

```bash
# 1. Extract — renders, VLM, writes a bare world to the DB
python scripts/extract_class_structure.py \
    datasets/matterport3d/hm3d-minival-semantic-annots-v0.2/00800-TEEsavR23oF \
    out/vlm_summary.json \
    --dataset hm3d \
    --output-dir out/ \
    --export-dir out/taxonomy_export

# 2. Refine — class inference, dependency resolution, in-memory instantiation.
#    Writes pending_annotations.json. Does NOT regenerate the ORM by itself.
python scripts/refine_class_structure.py \
    out/vlm_summary.json <world_db_id> \
    --dataset hm3d --skip-persist

# 3. Persist — regenerate the ORM once, then write annotations back to the DB.
python scripts/persist_annotations.py \
    --world-db-id <world_db_id> \
    --dataset hm3d \
    out/pending_annotations.json
```

When a scene has been split into rooms, `persist_annotations.py` accepts multiple `pending_annotations.json` files plus a `--room-db-ids room_db_ids.json` mapping so the ORM regeneration runs only once across the entire scene.

### Inspecting persisted worlds

```bash
python scripts/load_and_render_scene.py                 # list worlds in the DB
python scripts/load_and_render_scene.py <world_name>    # render a persisted world
python -m semdt_2d_video.utils.inspect_camera_pose <obj_dir>   # interactive camera pose tuning
```

### Evaluation & paper figures

Scripts in `scripts/paper_graphics/` produce evaluation tables and publication renders.

**`compare_gt_vs_predicted.py`** — exact-match comparison of HM3D ground-truth labels against VLM predictions read from the batch output JSON. Prints a color-coded per-object table.

```bash
python scripts/paper_graphics/compare_gt_vs_predicted.py \
    batch_output/<run>/00800 \
    datasets/matterport3d/hm3d-minival-semantic-annots-v0.2/00800-TEEsavR23oF \
    --rooms 1 2 --csv comparison.csv
```

**`compare_gt_vs_predicted_semantic.py`** — embedding-based comparison. Loads predicted classes from the DB (post-persist), normalizes them to the GT vocabulary using `all-MiniLM-L6-v2` cosine similarity, and reports Precision / Recall / F1 / per-class IoU / mIoU.

```bash
python scripts/paper_graphics/compare_gt_vs_predicted_semantic.py \
    --batch-dir batch_output/<run>/00802 \
    --scene-dir datasets/matterport3d/hm3d-minival-semantic-annots-v0.2/00802-wcojb4TFT35 \
    --threshold 0.6 --csv results.csv

# or by DB ID
python scripts/paper_graphics/compare_gt_vs_predicted_semantic.py \
    --world-db-id 46 47 48 \
    --scene-dir datasets/matterport3d/hm3d-minival-semantic-annots-v0.2/00802-wcojb4TFT35
```

**`render_annotated_scene.py`** — renders a scene from four viewpoints, with objects colored by predicted class, alongside ground-truth coloring. Produces standalone legend images.

```bash
python scripts/paper_graphics/render_annotated_scene.py \
    batch_output/<run>/00800 \
    datasets/matterport3d/hm3d-minival-semantic-annots-v0.2/00800-TEEsavR23oF \
    --headless --output-dir paper_figures \
    --resolution 1920 1080 --render-gt
```

**`render_semantic_segmentation.py`** — loads a persisted world and renders it with each object painted in a distinct color (segmentation-style figure).

```bash
python scripts/paper_graphics/render_semantic_segmentation.py            # list worlds
python scripts/paper_graphics/render_semantic_segmentation.py --id <db_id> --save out.png
```

**`render_exploded_view.py`** — same input, but each body is offset radially from the world's geometric center to expose internal structure.

```bash
python scripts/paper_graphics/render_exploded_view.py --id <db_id> --explosion-factor 1.5 --save exploded.png
```
