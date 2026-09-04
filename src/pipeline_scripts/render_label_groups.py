"""
Render a Warsaw scene with its labelled objects highlighted, a group at a time.

The scene is one mesh whose faces carry, per class, the instance they belong to. This
walks those label segments in groups, colors each group in distinct colors, and renders
the scene from all four viewpoints per group -- the images a VLM is later asked to
classify.

    python -m pipeline_scripts.render_label_groups \
        dataset/kitchenlab_new_mesh_agreement_dataset \
        --output-dir pipeline_out_labels --headless

Beside the images it writes ``groups.json``, recording which segment was colored in
which color in which group, which is what a later stage needs to read the VLM's answers
back onto the scene's objects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from experiments.warsaw.world_loader import RenderedSegmentGroup, WarsawWorldLoader

DEFAULT_GROUP_SIZE = 8
"""
How many of the scene's objects are colored at once by default.
"""


def group_record(group: RenderedSegmentGroup, filenames: Dict[str, str]) -> dict:
    """
    Describe a rendered group, so its images can be read back onto the scene's objects.

    :param group: The rendered group.
    :param filenames: The file each of its images was written as, by viewpoint.
    :return: What the group holds, as JSON-ready data.
    """
    return {
        "index": group.index,
        "images": filenames,
        "segments": [
            {
                "name": str(segment.name),
                "class": segment.class_name,
                "instance": segment.instance,
                "faces": len(segment),
                "color": list(group.colors[segment.name].to_rgba()),
                "color_name": group.colors[segment.name].closest_css3_name(),
            }
            for segment in group.segments
        ],
    }


def render(arguments: argparse.Namespace) -> None:
    """
    Render the scene named on the command line, group by group.

    :param arguments: The command line arguments.
    """
    loader = WarsawWorldLoader(
        input_directory=arguments.scene_directory,
        render_resolution=tuple(arguments.resolution),
        unhighlighted_dimming=arguments.dimming,
    )
    image_directory = arguments.output_dir / "all" / "images"
    image_directory.mkdir(parents=True, exist_ok=True)

    segments = (
        loader.segments_of_classes(arguments.classes)
        if arguments.classes
        else loader.label_segments
    )
    if not segments:
        raise SystemExit(
            f"None of the classes {arguments.classes} is labelled in this scene. "
            f"It labels: {', '.join(loader.scene.class_names)}"
        )
    print(
        f"{len(segments)} segments in groups of {arguments.group_size} "
        f"-> {-(-len(segments) // arguments.group_size)} groups of 4 renders"
    )

    print("rendering the scene in its own colors ...")
    for pose_name, image in loader.render_scene_from_camera_poses(
        loader.compute_camera_poses(), headless=arguments.headless
    ).items():
        (image_directory / f"scene_orig_{pose_name}.png").write_bytes(image)

    records: List[dict] = []
    for group in loader.render_label_segment_groups(
        group_size=arguments.group_size,
        segments=segments,
        headless=arguments.headless,
    ):
        filenames = {}
        for pose_name, image in group.images.items():
            filename = f"scene_{group.index}_{pose_name}__{group.label}.png"
            (image_directory / filename).write_bytes(image)
            filenames[pose_name] = filename
        records.append(group_record(group, filenames))
        print(f"  group {group.index}: {group.label}")

        if arguments.max_groups is not None and len(records) >= arguments.max_groups:
            break

    groups_file = arguments.output_dir / "groups.json"
    groups_file.write_text(
        json.dumps(
            {
                "scene": str(loader.scene.mesh_path),
                "group_size": arguments.group_size,
                "resolution": list(arguments.resolution),
                "groups": records,
            },
            indent=2,
        )
    )
    print(f"\n{len(records)} groups written to {image_directory}")
    print(f"group makeup written to {groups_file}")


def main() -> None:
    """
    Render the scene named on the command line.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "scene_directory",
        type=Path,
        help="Directory holding the scene's labelled mesh.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the images and groups.json into.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=DEFAULT_GROUP_SIZE,
        help=f"How many label segments to color at once (default: {DEFAULT_GROUP_SIZE}).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Only render the segments of these classes, e.g. --classes cabinet drawer.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Stop after this many groups, for a quick look at the output.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=(1024, 768),
        metavar=("WIDTH", "HEIGHT"),
        help="Size of the rendered images (default: 1024 768).",
    )
    parser.add_argument(
        "--dimming",
        type=float,
        default=0.35,
        help="How much of its own color the scene keeps where nothing is highlighted "
        "(default: 0.35, 1.0 to leave it as scanned).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Render without opening a window.",
    )
    render(parser.parse_args())


if __name__ == "__main__":
    main()
