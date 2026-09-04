#!/usr/bin/env python3
"""
Test script for HM3DWorldLoader.

Loads scene 00800-TEEsavR23oF, lists all objects with their IDs and colors,
then renders the world twice:
  1. With original textures (from the visual GLB)
  2. With flat semantic annotation colors (from the semantic GLB / world loader)

Usage:
    python scripts/test_hm3d_world_loader.py
    python scripts/test_hm3d_world_loader.py --save-dir output/
    python scripts/test_hm3d_world_loader.py --no-show          # headless, save only
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh

from semdt_2d_video.hm3d_world_loader import HM3DWorldLoader

DATASETS = Path(__file__).resolve().parents[3] / "datasets" / "matterport3d"

# Default scene: 00800-TEEsavR23oF (has semantic annotations)
DEFAULT_SCENE_DIR = DATASETS / "hm3d-minival-semantic-annots-v0.2" / "00800-TEEsavR23oF"
DEFAULT_VISUAL_GLB = DATASETS / "hm3d-minival-glb-v0.2" / "00800-TEEsavR23oF" / "TEEsavR23oF.glb"


def print_object_table(loader: HM3DWorldLoader) -> None:
    """Print a table of all semantic objects with ID, label, room, and color."""
    # Sort annotations by object_id
    sorted_anns = sorted(loader.annotations.values(), key=lambda a: a.object_id)

    print(f"\n{'ID':>4}  {'Label':<30}  {'Room':>4}  {'Color (hex)':>11}  {'Color (RGB)'}")
    print("-" * 80)
    for ann in sorted_anns:
        r, g, b = ann.rgb
        # ANSI true-color swatch
        swatch = f"\033[48;2;{r};{g};{b}m  \033[0m"
        print(
            f"{ann.object_id:>4}  {ann.label:<30}  {ann.room_id:>4}  "
            f"#{ann.hex_color:>6}  ({r:>3}, {g:>3}, {b:>3}) {swatch}"
        )

    print(f"\nTotal: {len(sorted_anns)} annotated objects")
    print(f"Rooms: {loader.room_ids}")
    print(f"Unique labels: {len(loader.labels)}")


def print_world_summary(loader: HM3DWorldLoader) -> None:
    """Print a summary of the constructed World."""
    world = loader.world
    bodies = list(world.bodies)
    connections = list(world.connections)

    print(f"\nWorld: {world.name}")
    print(f"  Bodies: {len(bodies)} (including root)")
    print(f"  Connections: {len(connections)}")

    # Count faces per body
    total_faces = 0
    for body in bodies:
        if body.name.name == "root":
            continue
        if body.collision and len(body.collision) > 0:
            mesh = body.collision[0].mesh
            if mesh is not None:
                total_faces += len(mesh.faces)
    print(f"  Total mesh faces: {total_faces}")


def render_visual_glb(visual_glb_path: Path, save_path: Path = None, show: bool = True) -> None:
    """Render the original textured scene from the visual GLB."""
    print(f"\nLoading visual GLB: {visual_glb_path}")
    scene = trimesh.load(str(visual_glb_path))

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        png = scene.save_image(resolution=(1920, 1080), visible=True)
        with open(save_path, "wb") as f:
            f.write(png)
        print(f"  Saved to: {save_path}")

    if show:
        print("  Opening original-texture viewer...")
        scene.show()


def render_semantic_world(loader: HM3DWorldLoader, save_path: Path = None, show: bool = True) -> None:
    """Render the world built by HM3DWorldLoader (flat annotation colors)."""
    print("\nBuilding semantic color scene from World...")

    # Build a trimesh.Scene from all bodies in the world
    scene = trimesh.Scene()
    for body in loader.world.bodies:
        if body.name.name == "root":
            continue
        if not body.collision or len(body.collision) == 0:
            continue
        mesh = body.collision[0].mesh
        if mesh is None:
            continue
        scene.add_geometry(mesh, node_name=body.name.name)

    print(f"  Scene has {len(scene.geometry)} geometries")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        png = scene.save_image(resolution=(1920, 1080), visible=True)
        with open(save_path, "wb") as f:
            f.write(png)
        print(f"  Saved to: {save_path}")

    if show:
        print("  Opening semantic-color viewer...")
        scene.show()


def main():
    parser = argparse.ArgumentParser(description="Test the HM3D world loader")
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=DEFAULT_SCENE_DIR,
        help="Path to semantic annotation scene directory",
    )
    parser.add_argument(
        "--visual-glb",
        type=Path,
        default=DEFAULT_VISUAL_GLB,
        help="Path to the visual (textured) GLB file",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save rendered images",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't open interactive viewers (headless mode)",
    )
    args = parser.parse_args()

    show = not args.no_show

    # 1. Load scene
    print(f"Loading HM3D scene from: {args.scene_dir}")
    loader = HM3DWorldLoader(scene_dir=args.scene_dir)

    # 2. Print object table
    print_object_table(loader)
    print_world_summary(loader)

    # 3. Render original textures
    if args.visual_glb.exists():
        render_visual_glb(
            args.visual_glb,
            save_path=args.save_dir / "render_original.png" if args.save_dir else None,
            show=show,
        )
    else:
        print(f"\nVisual GLB not found: {args.visual_glb} — skipping textured render")

    # 4. Render semantic colors
    render_semantic_world(
        loader,
        save_path=args.save_dir / "render_semantic.png" if args.save_dir else None,
        show=show,
    )


if __name__ == "__main__":
    main()
