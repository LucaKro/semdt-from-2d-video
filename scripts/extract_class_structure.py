import argparse
import base64
import json
import os
# Import shared functions from load_warsaw_scene
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests

from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from semantic_digital_twin.adapters.warsaw_world_loader import WarsawWorldLoader
from semantic_digital_twin.orm.ormatic_interface import Base, WorldMappingDAO
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from sqlalchemy.orm import Session

sys.path.insert(
    0, str(Path(__file__).parent.parent / "semantic_digital_twin" / "scripts")
)
from semdt_2d_video.hm3d_world_loader import HM3DWorldLoader

DB_NAME = os.getenv("PGDATABASE")
DB_USER = os.getenv("PGUSER")
DB_PASSWORD = os.getenv("PGPASSWORD")

DB_HOST = "localhost"
DB_PORT = os.getenv("PGPORT", 5432)

connection_string = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def encode_image_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def create_camera_poses() -> Dict[str, np.ndarray]:
    """
    Create camera transforms
    """
    return {
        "top": np.array(
            [
                [0.019661, 0.565278, -0.824666, -3.302801],
                [-0.999801, 0.008306, -0.018143, -0.113984],
                [-0.003406, 0.824859, 0.565329, 2.521490],
                [0.000000, 0.000000, 0.000000, 1.000000],
            ]
        ),
        "front_right": np.array(
            [
                [0.493480, -0.501933, 0.710310, 2.967743],
                [0.863728, 0.378807, -0.332385, -1.558169],
                [-0.102235, 0.777540, 0.620467, 2.430472],
                [0.000000, 0.000000, 0.000000, 1.000000],
            ]
        ),
        "front_left": np.array(
            [
                [-0.658764, -0.383959, 0.646997, 2.903789],
                [0.752266, -0.323319, 0.574074, 3.060092],
                [-0.011235, 0.864893, 0.501830, 2.281899],
                [0.000000, 0.000000, 0.000000, 1.000000],
            ]
        ),
    }


def query_vlm(
    original_images: List[bytes],
    highlighted_images: List[bytes],
    object_taxonomy: str,
    color_names: List[str],
    semantic_labels: str,
) -> dict:
    """Query the VLM with original and highlighted images from multiple viewpoints."""
    base64_originals = [encode_image_bytes(img) for img in original_images]
    base64_highlighted = [encode_image_bytes(img) for img in highlighted_images]

    system_prompt = f"""You are a semantic perception system for robotic scene understanding.

## Your Task
Analyze images and classify objects according to a given ontology. You will receive:
1. Four images of a scene with original textures from different viewpoints (diagonal front left, diagonal front right, diagonal back left, diagonal back right)
2. Four corresponding images with specific objects highlighted in distinct colors from the same viewpoints
3. The prior from a previous semantic segmentation step for each highlighted object

Focus ONLY on the highlighted objects. For each:
- Identify its class from the provided taxonomy
- If no suitable class exists, propose a new subclass under the most appropriate parent

### Important rules to keep in mind
1. A class cannot be its own superclass
2. The prior semantic segmentation may be wrong. In such cases, please provide a correct semantic class name.
3. Use all four viewpoints to get a complete understanding of each object's shape and context.

## Output Schema
Respond with valid JSON:
{{
  "objects": [
    {{
      "highlight_color": "string (the color used to highlight this object: one of {color_names})",
      "classification": {{
        "class": "string (class name from taxonomy, or your proposed new class)",
        "superclass": "string (parent class in taxonomy)",
        "is_new_class": "boolean (true if you're proposing a new class)",
        "new_class_justification": "string | null (if is_new_class, explain why existing classes don't fit)"
      }},
      "confidence": "number (0-1)"
    }}
  ],
  "notes": "string | null (any ambiguities or uncertainties)"
}}"""

    user_prompt = f"""## Object Taxonomy (Hierarchy)
{object_taxonomy}

## Prior semantic labels
{semantic_labels}

## Images

Images 1-4: Original scene with natural textures from four viewpoints:
  - Image 1: Diagonal front left view
  - Image 2: Diagonal front right view
  - Image 3: Diagonal back left view
  - Image 4: Diagonal back right view

Images 5-8: Same scene with target objects highlighted in distinct colors from the same viewpoints:
  - Image 5: Diagonal front left view (highlighted)
  - Image 6: Diagonal front right view (highlighted)
  - Image 7: Diagonal back left view (highlighted)
  - Image 8: Diagonal back right view (highlighted)

Identify the highlighted objects using all four viewpoints for a complete understanding.
"""

    # Build content array with text and all images
    content = [{"type": "text", "text": user_prompt}]

    # Add original images
    for base64_img in base64_originals:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_img}"},
            }
        )

    # Add highlighted images
    for base64_img in base64_highlighted:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_img}"},
            }
        )

    payload = json.dumps(
        {
            "model": "qwen/qwen3-vl-30b-a3b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
        }
    )
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "ai.uni-bremen.de",
        "X-Title": "Uni Bremen",
    }

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=payload,
                timeout=120,
            )
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
                print(f"  VLM request failed ({type(e).__name__}), retrying in {wait}s "
                      f"(attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
        except requests.exceptions.HTTPError as e:
            status = response.status_code
            # Retry on 429 (rate limit) and 5xx (server errors)
            if status in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  VLM request returned {status}, retrying in {wait}s "
                      f"(attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise


def extract_summary(all_responses: List[dict]) -> List[dict]:
    """
    Extract a summarized form from the raw VLM responses.
    For each object, returns: body_id, color, class, superclass, is_new_class, confidence.
    """
    summary = []

    for group_data in all_responses:
        body_ids = group_data["body_ids"]
        colors = group_data["colors"]
        vlm_response = group_data["vlm_response"]

        # Parse VLM content (may be wrapped in markdown code block)
        try:
            content = vlm_response["choices"][0]["message"]["content"]
            # Remove markdown code block if present
            content = content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(content)
            objects = parsed.get("objects", [])
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            print(
                f"Warning: Could not parse VLM response for group {group_data['group_index']}: {e}"
            )
            continue

        for i, obj in enumerate(objects):
            color_vlm = obj.get("highlight_color", "")
            body_id = body_ids[colors.index(color_vlm)]

            classification = obj.get("classification", {})

            summary.append(
                {
                    "body_id": body_ids[i],
                    "color": color_vlm,
                    "class": classification.get("class"),
                    "superclass": classification.get("superclass"),
                    "is_new_class": classification.get("is_new_class", False),
                    "confidence": obj.get("confidence"),
                }
            )

    return summary


def main(args):
    obj_dir = args.obj_dir
    output_file = args.output_file
    group_size = args.group_size
    export_path = Path(args.export_dir)

    # Check API key if VLM will be called
    if not args.render_only and not args.skip_vlm:
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is not set. Required for VLM queries."
            )

    # Base output directory for all artifacts (images, JSON, taxonomy)
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = Path(__file__).parent / "model_input"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Create database engine and tables for later persistence of the world
    engine = create_engine(connection_string, echo=True)  # echo=True for debugging SQL
    Base.metadata.create_all(bind=engine)
    print("Database tables created")

    # Load world
    print(f"Loading world from {obj_dir}...")

    if args.dataset == "hm3d":
        # Derive the visual GLB path from the semantic annotations path.
        # e.g. .../hm3d-minival-semantic-annots-v0.2/00800-TEEsavR23oF
        #   -> .../hm3d-minival-glb-v0.2/00800-TEEsavR23oF/<hash>.glb
        visual_glb_path = None
        glb_dir = obj_dir.parent.parent / "hm3d-minival-glb-v0.2" / obj_dir.name
        if glb_dir.exists():
            glb_files = list(glb_dir.glob("*.glb"))
            if glb_files:
                visual_glb_path = glb_files[0]
                print(f"Using visual GLB: {visual_glb_path}")

        # Always process room-by-room for HM3D.
        # --num-rooms limits how many rooms to process (default: all).
        all_room_ids = HM3DWorldLoader.discover_room_ids(obj_dir)
        # room_id=0 in HM3D is typically the "unassigned" bucket: scattered
        # structural elements and unknown objects that don't belong to any
        # well-formed room. Skip it by default.
        if args.include_unassigned_room:
            candidate_room_ids = all_room_ids
        else:
            candidate_room_ids = [r for r in all_room_ids if r != 0]
        if args.num_rooms is not None:
            room_ids = candidate_room_ids[:args.num_rooms]
        else:
            room_ids = candidate_room_ids
        print(f"Will process {len(room_ids)} room(s): {room_ids} "
              f"(out of {len(all_room_ids)} total)")

        room_batches = []
        for rid in room_ids:
            print(f"  Loading room {rid}...")
            loader = HM3DWorldLoader(
                scene_dir=obj_dir, visual_glb_path=visual_glb_path,
                room_id=rid,
            )
            bodies = loader.object_bodies
            poses = loader.compute_camera_poses()
            room_batches.append((f"room{rid}", bodies, poses, loader))
            print(f"  Room {rid}: {len(bodies)} objects")

        # Use the first room's loader for taxonomy export
        world_loader = room_batches[0][3]
        world = world_loader.world
    else:
        world_loader = WarsawWorldLoader(obj_dir)
        world = world_loader.world
        bodies = world.bodies_with_enabled_collision
        camera_poses_dict = create_camera_poses()
        room_batches = [(None, bodies, camera_poses_dict, world_loader)]

    # Export semantic annotations JSON for VLM context
    if args.output_dir and export_path == Path("./vlm_export"):
        # Default export path: put taxonomy inside the output dir
        export_path = base_output_dir / "taxonomy_export"
    world_loader.export_semantic_annotation_inheritance_structure(export_path)

    # Read taxonomy
    object_taxonomy = (export_path / "semantic_annotations.json").read_text()

    if not args.skip_vlm and not args.render_only:
        all_responses = []
        summary = []

        for room_tag, bodies, camera_poses_dict, batch_loader in room_batches:
            room_folder = room_tag if room_tag else "all"
            room_dir = base_output_dir / room_folder
            image_dir = room_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)

            if room_tag:
                print(f"\n{'='*60}")
                print(f"Processing {room_tag} ({len(bodies)} objects)")
                print(f"{'='*60}")

            # Render original scene from 4 viewpoints
            print("Rendering original scene from 4 viewpoints...")
            original_images = []
            for pose_name, camera_pose in camera_poses_dict.items():
                print(f"  Rendering {pose_name} view...")
                image_bytes = batch_loader.render_scene_from_camera_pose(
                    camera_pose,
                    image_dir / f"scene_orig_{pose_name}.png",
                    headless=args.headless, use_visual_mesh=True,
                )
                original_images.append(image_bytes)

            # Process groups
            room_responses = []
            num_groups = (len(bodies) + group_size - 1) // group_size

            for i, start in enumerate(range(0, len(bodies), group_size)):
                group = bodies[start : start + group_size]
                print(f"Processing group {i + 1}/{num_groups} ({len(group)} objects)...")

                # Gray out non-highlighted objects (HM3D) or restore textures (Warsaw)
                if hasattr(batch_loader, '_neutralize_body_colors'):
                    batch_loader._neutralize_body_colors()
                else:
                    batch_loader._reset_body_colors()
                bodies_colors = batch_loader._apply_highlight_to_group(group)
                color_names = list(
                    map(lambda c: c.closest_css3_color_name(), bodies_colors.values())
                )

                # Render highlighted scene from 4 viewpoints
                print(f"  Rendering highlighted scene from 4 viewpoints...")
                highlighted_images = []
                for pose_name, camera_pose in camera_poses_dict.items():
                    image_bytes = batch_loader.render_scene_from_camera_pose(
                        camera_pose,
                        image_dir / f"scene_{i}_{pose_name}.png",
                        headless=args.headless,
                    )
                    highlighted_images.append(image_bytes)

                semantic_labels_dict = {
                    body_id: color_names[i]
                    for i, body_id in enumerate(bodies_colors.keys())
                }
                semantic_labels = ""
                for body_uuid, color_name in semantic_labels_dict.items():
                    body = next(filter(lambda b: b.id == body_uuid, bodies))
                    if args.no_prior_labels:
                        semantic_labels += f"{color_name}: (no prior label)\n"
                    else:
                        semantic_labels += f"{color_name}: {body.name}\n"

                # Query VLM
                print(f"  Querying VLM for group {i + 1}...")
                response = query_vlm(
                    original_images,
                    highlighted_images,
                    object_taxonomy,
                    color_names,
                    semantic_labels,
                )
                print(f"  Response: {response}")

                room_responses.append(
                    {
                        "group_index": i,
                        "room": room_tag,
                        "body_ids": list(map(str, bodies_colors.keys())),
                        "colors": color_names,
                        "vlm_response": response,
                    }
                )

                # Reset for next iteration
                batch_loader._reset_body_colors()

            # Save per-room VLM responses and summary
            room_responses_file = room_dir / "vlm_responses.json"
            with open(room_responses_file, "w") as f:
                json.dump(room_responses, f, indent=2)
            print(f"Room responses saved to {room_responses_file}")

            room_summary = extract_summary(room_responses)
            room_summary_file = room_dir / "vlm_summary.json"
            with open(room_summary_file, "w") as f:
                json.dump(room_summary, f, indent=2)
            print(f"Room summary saved to {room_summary_file}")

            all_responses.extend(room_responses)

        # Also save combined responses to the legacy output_file path
        with open(output_file, "w") as f:
            json.dump(all_responses, f, indent=2)
        print(f"Combined raw results saved to {output_file}")

        summary = extract_summary(all_responses)
        summary_file = output_file.parent / (output_file.stem + "_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Combined summary saved to {summary_file}")

    elif args.render_only:
        # Render-only mode: just save images without calling VLM
        print("Render-only mode: Saving images without VLM queries...")

        for room_tag, bodies, camera_poses_dict, batch_loader in room_batches:
            room_folder = room_tag if room_tag else "all"
            room_dir = base_output_dir / room_folder
            image_dir = room_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)

            if room_tag:
                print(f"\n{'='*60}")
                print(f"Rendering {room_tag} ({len(bodies)} objects)")
                print(f"{'='*60}")

            # Render original scene from 4 viewpoints
            print("Rendering original scene from 4 viewpoints...")
            for pose_name, camera_pose in camera_poses_dict.items():
                print(f"  Rendering {pose_name} view...")
                batch_loader.render_scene_from_camera_pose(
                    camera_pose,
                    image_dir / f"scene_orig_{pose_name}.png",
                    headless=args.headless, use_visual_mesh=True,
                )

            # Process groups
            num_groups = (len(bodies) + group_size - 1) // group_size

            for i, start in enumerate(range(0, len(bodies), group_size)):
                group = bodies[start : start + group_size]
                print(f"Processing group {i + 1}/{num_groups} ({len(group)} objects)...")

                # Gray out non-highlighted objects (HM3D) or restore textures (Warsaw)
                if hasattr(batch_loader, '_neutralize_body_colors'):
                    batch_loader._neutralize_body_colors()
                else:
                    batch_loader._reset_body_colors()
                bodies_colors = batch_loader._apply_highlight_to_group(group)
                color_names = list(
                    map(lambda c: c.closest_css3_color_name(), bodies_colors.values())
                )

                # Render highlighted scene from 4 viewpoints
                print(f"  Rendering highlighted scene from 4 viewpoints...")
                for pose_name, camera_pose in camera_poses_dict.items():
                    batch_loader.render_scene_from_camera_pose(
                        camera_pose,
                        image_dir / f"scene_{i}_{pose_name}.png",
                        headless=args.headless,
                    )

                # Reset for next iteration
                batch_loader._reset_body_colors()

        print(f"All images saved to {base_output_dir}")
        all_responses = []
        summary = []

    elif args.skip_vlm:  # Skipped VLM: Read from file instead
        with open(output_file, "r") as f:
            all_responses = json.load(f)

        # Extract and save summary
        summary = extract_summary(all_responses)
        summary_file = output_file.parent / (output_file.stem + "_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_file}")

    # Convert world(s) to DAO and persist
    # For HM3D room-by-room processing, each room has its own World with
    # unique body UUIDs.  Persist every room's world so that the refinement
    # step can find the bodies referenced in each room's VLM summary.
    room_db_ids: Dict[str, int] = {}

    with Session(engine) as session:
        if args.dataset == "hm3d" and len(room_batches) > 1:
            for room_tag, _bodies, _poses, room_loader in room_batches:
                room_world = room_loader.world
                room_dao: WorldMappingDAO = to_dao(room_world)
                session.add(room_dao)
                session.flush()  # get the database_id
                room_db_ids[room_tag] = room_dao.database_id
                print(f"Room {room_tag} persisted with database_id: {room_dao.database_id}")
            session.commit()
        else:
            world_dao: WorldMappingDAO = to_dao(world)
            session.add(world_dao)
            session.commit()
            room_tag = room_batches[0][0] if room_batches[0][0] else "all"
            room_db_ids[room_tag] = world_dao.database_id
            print(f"World persisted with database_id: {world_dao.database_id}")

    # Write room → database_id mapping for the batch runner
    room_db_ids_file = base_output_dir / "room_db_ids.json"
    with open(room_db_ids_file, "w") as f:
        json.dump(room_db_ids, f, indent=2)
    print(f"Room DB IDs saved to {room_db_ids_file}")

    return all_responses, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query VLM for scene understanding")
    parser.add_argument(
        "obj_dir",
        type=Path,
        help="Path to scene directory (.obj files for warsaw, semantic annotation dir for hm3d)",
    )
    parser.add_argument("output_file", type=Path, help="Path to output JSON file")
    parser.add_argument(
        "--dataset",
        choices=["warsaw", "hm3d"],
        default="warsaw",
        help="Dataset type: 'warsaw' (default) or 'hm3d'",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("./vlm_export"),
        help="Directory for exported metadata",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base output directory for all artifacts (images, JSON, taxonomy). "
             "Per-room subdirectories are created automatically.",
    )
    parser.add_argument(
        "--group-size", type=int, default=8, help="Number of objects per group"
    )
    parser.add_argument(
        "--skip_vlm",
        action="store_true",
        default=False,
        help="Skip VLM queries and read responses from file instead",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        default=False,
        help="Only render images without calling VLM. Images are saved to scripts/model_input/",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Use offscreen rendering (no display required)",
    )
    parser.add_argument(
        "--no-prior-labels",
        action="store_true",
        default=False,
        help="Do not send HM3D ground truth body names as prior semantic labels "
             "to the VLM. Use this for unbiased evaluation.",
    )
    parser.add_argument(
        "--include-unassigned-room",
        action="store_true",
        default=False,
        help="HM3D only: include room_id=0 (typically the 'unassigned' bucket "
             "of scattered structural elements). Skipped by default.",
    )
    parser.add_argument(
        "--num-rooms",
        type=int,
        default=None,
        help="HM3D only: limit processing to the first N rooms. "
             "If omitted, all rooms are processed. Each room is always "
             "processed independently.",
    )
    args = parser.parse_args()
    main(args)
