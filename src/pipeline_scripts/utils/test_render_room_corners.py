#!/usr/bin/env python3
"""
Test script: render one HM3D room from all four top-corner camera poses.

Loads a single room from an HM3D scene and renders it from the four
bounding-box corner viewpoints (front_left, front_right, back_left,
back_right), saving the images to a given output directory.

Usage:
    python scripts/test_render_room_corners.py --output-dir output/room_corners
    python scripts/test_render_room_corners.py --scene-dir datasets/matterport3d/hm3d-minival-semantic-annots-v0.2/00800-TEEsavR23oF --room-id 0
    python scripts/test_render_room_corners.py --headless --output-dir output/room_corners
"""

import argparse
from pathlib import Path

from semdt_2d_video.hm3d_world_loader import HM3DWorldLoader

DATASETS = Path(__file__).resolve().parents[3] / "datasets" / "matterport3d"

DEFAULT_SCENE_DIR = DATASETS / "hm3d-minival-semantic-annots-v0.2" / "00800-TEEsavR23oF"
DEFAULT_VISUAL_GLB = DATASETS / "hm3d-minival-glb-v0.2" / "00800-TEEsavR23oF" / "TEEsavR23oF.glb"


def main():
    parser = argparse.ArgumentParser(
        description="Render one HM3D room from 4 corner camera poses",
    )
    parser.add_argument(
        "--scene-dir", type=Path, default=DEFAULT_SCENE_DIR,
        help="Path to HM3D semantic annotation scene directory",
    )
    parser.add_argument(
        "--visual-glb", type=Path, default=DEFAULT_VISUAL_GLB,
        help="Path to the visual (textured) GLB file",
    )
    parser.add_argument(
        "--room-id", type=int, default=None,
        help="Room ID to render. If omitted, uses the first room in the scene.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/room_corners"),
        help="Directory to save the 4 rendered images",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Use offscreen rendering (no display required)",
    )
    args = parser.parse_args()

    # Determine room ID
    room_ids = HM3DWorldLoader.discover_room_ids(args.scene_dir)
    room_id = args.room_id if args.room_id is not None else room_ids[0]
    print(f"Available rooms: {room_ids}")
    print(f"Rendering room: {room_id}")

    # Resolve visual GLB
    visual_glb = args.visual_glb if args.visual_glb.exists() else None
    if visual_glb:
        print(f"Visual GLB: {visual_glb}")
    else:
        print("No visual GLB found, will render with semantic colors only")

    # Load room
    loader = HM3DWorldLoader(
        scene_dir=args.scene_dir,
        visual_glb_path=visual_glb,
        room_id=room_id,
    )
    bodies = loader.object_bodies
    print(f"Room {room_id}: {len(bodies)} objects")

    # Compute 4 corner camera poses
    poses = loader.compute_camera_poses()
    print(f"Camera poses: {list(poses.keys())}")

    # Render
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for pose_name, camera_pose in poses.items():
        out_path = output_dir / f"room{room_id}_{pose_name}.png"
        print(f"  Rendering {pose_name}...")
        loader.render_scene_from_camera_pose(
            camera_pose, out_path,
            headless=args.headless, use_visual_mesh=True,
        )
        print(f"    Saved: {out_path}")

    print(f"\nDone. {len(poses)} images saved to {output_dir}")


if __name__ == "__main__":
    main()
