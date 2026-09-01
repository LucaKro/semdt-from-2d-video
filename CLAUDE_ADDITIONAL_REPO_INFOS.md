# Project: Semantic Digital Twin from 2D Video

Pipeline for constructing semantic digital twins from 3D scene datasets using VLM-based object classification.

## Architecture

Three-phase pipeline:
1. **extract_class_structure.py** - Loads a 3D scene, renders from 3 camera viewpoints, highlights groups of objects in distinct colors, queries a VLM (Qwen-VL via OpenRouter) to classify each object against a semantic taxonomy.
2. **refine_class_structure.py** - Takes VLM classifications, resolves constructor field dependencies (recursively), instantiates SemanticAnnotation subclasses, persists to PostgreSQL.
3. **load_and_render_scene.py** - Loads a persisted world from DB and renders it.

## Supported Datasets

- **Warsaw** (default): Uses `WarsawWorldLoader` from the `semantic_digital_twin` SDK. Loads from directories of `.obj` files. Camera poses are hardcoded in `create_camera_poses()`.
- **HM3D** (Habitat-Matterport 3D): Uses `HM3DWorldLoader` from `src/hm3d_world_loader.py`. Loads from semantic annotation directories containing `<scene_id>.semantic.glb` + `<scene_id>.semantic.txt`. Camera poses are auto-computed from scene bounding box. Select with `--dataset hm3d`.

HM3D minival data lives in `datasets/matterport3d/` with three parallel bundles per scene:
- `hm3d-minival-glb-v0.2/` - textured visual meshes
- `hm3d-minival-habitat-v0.2/` - Habitat simulator meshes + navmeshes
- `hm3d-minival-semantic-annots-v0.2/` - semantic annotated meshes + color-to-label lookup tables

Scenes with semantic annotations: 00800, 00802, 00803, 00808.

## World Loader Interface

Both `WarsawWorldLoader` and `HM3DWorldLoader` expose:
- `.world` - the `World` object
- `.render_scene_from_camera_pose(camera_transform, output_filepath)` -> PNG bytes
- `._reset_body_colors()` - restore original mesh visuals
- `._apply_highlight_to_group(bodies)` -> `Dict[UUID, Color]`
- `.export_semantic_annotation_inheritance_structure(output_directory)` - exports taxonomy JSON

Bodies are obtained differently:
- Warsaw: `world.bodies_with_enabled_collision`
- HM3D: `world_loader.object_bodies` (all non-root bodies; HM3D bodies lack collision configs)

## External Dependencies

This project depends on the `cognitive_robot_abstract_machine` monorepo at `/home/ben/devel/iai/src/cognitive_robot_abstract_machine/`, which is a **Poetry** workspace (`package-mode = false`) that installs subprojects as path dependencies:
- `semantic_digital_twin` - core world/scene representation, WarsawWorldLoader, Color, Mesh, RayTracer
- `krrood` - ORM (ORMatic), class diagram tools, InheritanceStructureExporter
- `pycram-robotics`, `giskardpy`

**Important**: `semantic_digital_twin` itself uses **setuptools** (not Poetry). Its dependencies are in `requirements.txt` and read via `dynamic = ["dependencies"]` in its `pyproject.toml`. When adding dependencies there, you must `poetry lock && poetry install` at the monorepo level to pick them up.

Key SDK types used: `World`, `Body`, `PrefixedName`, `Color` (has `.distinct_html_colors()`, `.closest_css3_color_name()`), `Mesh`, `ShapeCollection`, `FixedConnection`, `TransformationMatrix`.

## Environment Variables

- `OPENROUTER_API_KEY` - required for VLM queries
- `PGDATABASE`, `PGUSER`, `PGPASSWORD` - PostgreSQL connection (host: localhost, port: 5432)

## Running

```bash
# HM3D extraction
python scripts/extract_class_structure.py \
    datasets/matterport3d/hm3d-minival-semantic-annots-v0.2/00800-TEEsavR23oF \
    model_output/hm3d/hm3d_res.json \
    --dataset hm3d \
    --export-dir model_input/semantic_annotations_exported

# Warsaw extraction (original)
python scripts/extract_class_structure.py /path/to/obj_dir output.json

# Useful flags: --skip_vlm (reparse existing output), --render-only (images only), --group-size N
```