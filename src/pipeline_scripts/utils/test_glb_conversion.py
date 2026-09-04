import os
from pathlib import Path

import trimesh

# Path to the GLB directory, relative to the repository this script lives in.
REPO_ROOT = Path(__file__).resolve().parents[3]
glb_base_dir = REPO_ROOT / "datasets" / "matterport3d" / "hm3d-minival-glb-v0.2"

# Get the first available GLB file
scene_dirs = sorted(os.listdir(glb_base_dir))
first_scene = scene_dirs[0]
glb_path = os.path.join(glb_base_dir, first_scene, f'{first_scene.split("-")[1]}.glb')

print(f'Loading GLB file: {glb_path}')

# Load GLB file
scene = trimesh.load(glb_path)

print(f'Loaded type: {type(scene)}')

# If it's a Scene, get all the geometry
if isinstance(scene, trimesh.Scene):
    print(f'Scene contains {len(scene.geometry)} geometries')

    # meshes = []
    # for name, geom in scene.geometry.items():
    #     print(f'  - {name}: {type(geom)}, vertices: {len(geom.vertices) if hasattr(geom, "vertices") else "N/A"}')
    #     meshes.append(geom)

    scene.show()

    
else:
    print(f'Single mesh: {len(scene.vertices)} vertices, {len(scene.faces)} faces')
    print(f'Bounds: {scene.bounds}')
    scene.show()
