"""
World loader for HM3D (Habitat-Matterport 3D) Semantics v0.2 scenes.

Parses the semantic annotation GLB + TXT files to produce a World where
each semantically annotated object is a separate Body with a Mesh.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from uuid import UUID

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color, Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


# HM3D GLBs follow the Habitat/OpenGL Y-up convention. All camera math
# in this loader depends on that assumption.
HM3D_UP_AXIS = np.array([0.0, 1.0, 0.0])


def _look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = HM3D_UP_AXIS,
) -> np.ndarray:
    """Compute a camera-to-world 4x4 transform (OpenGL: camera looks along -Z)."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    cam_up = np.cross(right, forward)

    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = cam_up
    mat[:3, 2] = -forward
    mat[:3, 3] = eye
    return mat


@dataclass
class SemanticObject:
    """A single semantically annotated object parsed from HM3D annotations."""

    object_id: int
    label: str
    room_id: int
    hex_color: str
    rgb: Tuple[int, int, int]


@dataclass
class HM3DWorldLoader:
    """
    Load an HM3D scene with semantic annotations into a World.

    Expects a scene directory containing:
      - <scene_id>.semantic.glb  (geometry with per-face annotation colors)
      - <scene_id>.semantic.txt  (color -> label lookup table)

    Optionally accepts a path to the visual GLB for textured rendering.
    Textured rendering is performed by projecting colors sampled from the
    visual GLB onto each face of the room's semantic meshes (nearest-face
    lookup in a KD-tree of visual-GLB face centroids). This avoids the need
    to crop the visual GLB and guarantees the render contains exactly the
    room's geometry.
    """

    scene_dir: Optional[Path] = None
    """Directory containing the semantic GLB and TXT files for one scene."""

    visual_glb_path: Optional[Path] = None
    """Optional path to the textured visual GLB (from hm3d-minival-glb-v0.2/)."""

    room_id: Optional[int] = None
    """If set, only load objects belonging to this room."""

    world: World = field(default=None)
    """The constructed World (built from files, or provided directly via from_world)."""

    annotations: Dict[Tuple[int, int, int], SemanticObject] = field(
        init=False, default_factory=dict
    )
    """Mapping from RGB color tuple to SemanticObject."""

    _scene_id: str = field(init=False, default="")
    _original_visuals: Dict[UUID, np.ndarray] = field(init=False, default_factory=dict)
    """Semantic per-face colors for each body; used to restore after highlighting."""

    _textured_visuals: Dict[UUID, np.ndarray] = field(init=False, default_factory=dict)
    """Per-face RGBA arrays sampled from the visual GLB, keyed by body id.
    Populated only if a visual GLB is available."""

    _shell_body_ids: Set[UUID] = field(init=False, default_factory=set)
    """Bodies loaded as structural shell (room_id=0) when filtering by room.
    Rendered for context/occlusion, excluded from VLM targets and highlighting."""

    def __post_init__(self):
        if self.world is not None:
            # World provided directly (e.g. loaded from DB) — skip file loading
            self._scene_id = self.world.name or ""
            self.annotations = {}
            self._original_visuals = {}
            self._textured_visuals = {}
            self._shell_body_ids = set()
            self._save_original_state()
            return
        self.scene_dir = Path(self.scene_dir)
        self._scene_id = self._detect_scene_id()
        self.annotations = self._parse_semantic_txt()
        self.world = self._build_world()
        self._save_original_state()
        self._project_visual_textures_onto_semantic_meshes()

    @classmethod
    def from_world(cls, world: World) -> "HM3DWorldLoader":
        """Create an HM3DWorldLoader wrapping an existing World (e.g. loaded from DB).

        Bypasses all file-based loading (GLB parsing, semantic TXT parsing).
        The resulting loader supports rendering, highlighting, and camera pose
        computation but will have an empty ``annotations`` dict.
        """
        loader = object.__new__(cls)
        loader.scene_dir = None
        loader.visual_glb_path = None
        loader.room_id = None
        loader.world = world
        loader.annotations = {}
        loader._scene_id = world.name or ""
        loader._original_visuals = {}
        loader._textured_visuals = {}
        loader._shell_body_ids = set()
        loader._save_original_state()
        return loader

    def _detect_scene_id(self) -> str:
        """Infer the scene ID from the .semantic.txt file in the directory."""
        txt_files = list(self.scene_dir.glob("*.semantic.txt"))
        if not txt_files:
            raise FileNotFoundError(
                f"No .semantic.txt file found in {self.scene_dir}"
            )
        return txt_files[0].stem.replace(".semantic", "")

    @staticmethod
    def discover_room_ids(scene_dir: Path) -> List[int]:
        """Return sorted room IDs from the semantic TXT without loading any GLB."""
        scene_dir = Path(scene_dir)
        txt_files = list(scene_dir.glob("*.semantic.txt"))
        if not txt_files:
            raise FileNotFoundError(
                f"No .semantic.txt file found in {scene_dir}"
            )
        room_ids: set = set()
        for line in txt_files[0].read_text().splitlines()[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            room_ids.add(int(parts[3]))
        return sorted(room_ids)

    @property
    def semantic_glb_path(self) -> Path:
        return self.scene_dir / f"{self._scene_id}.semantic.glb"

    @property
    def semantic_txt_path(self) -> Path:
        return self.scene_dir / f"{self._scene_id}.semantic.txt"

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_semantic_txt(self) -> Dict[Tuple[int, int, int], SemanticObject]:
        """Parse the .semantic.txt lookup table."""
        annotations: Dict[Tuple[int, int, int], SemanticObject] = {}
        lines = self.semantic_txt_path.read_text().splitlines()

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            object_id = int(parts[0])
            hex_color = parts[1]
            label = parts[2].strip('"')
            room_id = int(parts[3])

            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)

            annotations[(r, g, b)] = SemanticObject(
                object_id=object_id,
                label=label,
                room_id=room_id,
                hex_color=hex_color,
                rgb=(r, g, b),
            )
        return annotations

    # ------------------------------------------------------------------
    # World construction
    # ------------------------------------------------------------------

    def _build_world(self) -> World:
        """Load the semantic GLB, split by annotation color, and build a World."""
        scene = trimesh.load(str(self.semantic_glb_path))

        meshes: List[trimesh.Trimesh] = []
        if isinstance(scene, trimesh.Scene):
            for geom in scene.geometry.values():
                if isinstance(geom, trimesh.Trimesh):
                    meshes.append(geom)
        elif isinstance(scene, trimesh.Trimesh):
            meshes.append(scene)
        else:
            raise ValueError(
                f"Unexpected type from trimesh.load: {type(scene)}"
            )

        color_groups: Dict[Tuple[int, int, int], List[trimesh.Trimesh]] = {}

        for mesh in meshes:
            # Semantic GLBs may use TextureVisuals (UV-mapped flat color texture)
            # instead of ColorVisuals — convert so we can read per-face colors.
            if hasattr(mesh.visual, "to_color"):
                mesh.visual = mesh.visual.to_color()
            face_colors = mesh.visual.face_colors[:, :3]
            unique_colors = np.unique(face_colors, axis=0)

            for color in unique_colors:
                key = tuple(int(c) for c in color)

                if self.room_id is not None:
                    annotation = self.annotations.get(key)
                    # Keep faces for the target room AND structural shell
                    # (room_id=0) — shells provide walls/floors that would
                    # otherwise be dropped at room boundaries.
                    if annotation is None or annotation.room_id not in (
                        self.room_id, 0,
                    ):
                        continue

                mask = np.all(face_colors == color, axis=1)
                submesh = mesh.submesh([mask], only_watertight=False, append=True)
                if submesh is None or len(submesh.faces) == 0:
                    continue
                color_groups.setdefault(key, []).append(submesh)

        world = World(name=f"hm3d_{self._scene_id}")
        root = Body(name=PrefixedName("root"))

        with world.modify_world():
            world.add_body(root)

            for rgb, submeshes in color_groups.items():
                combined = trimesh.util.concatenate(submeshes)
                annotation = self.annotations.get(rgb)

                if annotation is None:
                    continue

                body_name = f"{annotation.label}_{annotation.object_id}"

                r, g, b = rgb
                mesh_shape = Mesh.from_trimesh(
                    mesh=combined,
                    origin=HomogeneousTransformationMatrix(),
                )
                mesh_shape.dye(Color(R=r / 255.0, G=g / 255.0, B=b / 255.0))
                shape_collection = ShapeCollection([mesh_shape])

                body = Body(
                    name=PrefixedName(body_name),
                    collision=shape_collection,
                    visual=shape_collection,
                )

                connection = FixedConnection(
                    parent=root,
                    child=body,
                    name=PrefixedName(f"root_to_{body_name}"),
                )
                world.add_connection(connection)

                if (
                    self.room_id is not None
                    and annotation.room_id == 0
                    and annotation.room_id != self.room_id
                ):
                    self._shell_body_ids.add(body.id)

        return world

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_bodies_by_label(self, label: str) -> List[Body]:
        return [body for body in self.world.bodies if label in body.name.name]

    def get_bodies_in_room(self, room_id: int) -> List[Body]:
        room_object_names = {
            f"{ann.label}_{ann.object_id}"
            for ann in self.annotations.values()
            if ann.room_id == room_id
        }
        return [
            body for body in self.world.bodies
            if body.name.name in room_object_names
        ]

    @property
    def labels(self) -> List[str]:
        return sorted({ann.label for ann in self.annotations.values()})

    @property
    def room_ids(self) -> List[int]:
        return sorted({ann.room_id for ann in self.annotations.values()})

    @property
    def object_bodies(self) -> List[Body]:
        """Target-room bodies (excludes root and structural shell bodies)."""
        return [
            b for b in self.world.bodies
            if b.name.name != "root" and b.id not in self._shell_body_ids
        ]

    @property
    def _renderable_bodies(self) -> List[Body]:
        """All non-root bodies, including structural shell bodies."""
        return [b for b in self.world.bodies if b.name.name != "root"]

    # ------------------------------------------------------------------
    # Texture projection
    # ------------------------------------------------------------------

    def _project_visual_textures_onto_semantic_meshes(self) -> None:
        """Sample colors from the visual GLB onto each face of each room body.

        For each body, for each face, looks up the nearest face in the visual
        GLB (KD-tree over visual face centroids) and reads that face's color.
        Results are cached in ``self._textured_visuals`` keyed by body id.
        """
        if self.visual_glb_path is None:
            return
        path = Path(self.visual_glb_path)
        if not path.exists():
            print(f"Warning: visual GLB not found at {path}, "
                  f"textured rendering will fall back to semantic colors")
            return

        scene = trimesh.load(str(path))
        visual_meshes: List[trimesh.Trimesh] = []
        if isinstance(scene, trimesh.Scene):
            visual_meshes = [
                g for g in scene.geometry.values()
                if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0
            ]
        elif isinstance(scene, trimesh.Trimesh):
            visual_meshes = [scene]
        if not visual_meshes:
            return

        # Build a KD-tree over all visual face centroids, tracking face colors.
        centroids: List[np.ndarray] = []
        colors: List[np.ndarray] = []
        for m in visual_meshes:
            if hasattr(m.visual, "to_color"):
                m.visual = m.visual.to_color()
            face_rgba = np.asarray(m.visual.face_colors, dtype=np.uint8)
            if face_rgba.shape[0] != len(m.faces):
                # Some visuals return one color — broadcast.
                face_rgba = np.tile(face_rgba.reshape(-1)[:4], (len(m.faces), 1))
            centroids.append(m.triangles.mean(axis=1))
            colors.append(face_rgba)

        all_centroids = np.vstack(centroids)
        all_colors = np.vstack(colors)
        tree = cKDTree(all_centroids)

        for body in self._renderable_bodies:
            mesh = body.collision[0].mesh
            face_centroids = mesh.triangles.mean(axis=1)
            _, idx = tree.query(face_centroids, k=1)
            self._textured_visuals[body.id] = all_colors[idx].astype(np.uint8)

    # ------------------------------------------------------------------
    # Rendering & highlighting
    # ------------------------------------------------------------------

    def _save_original_state(self) -> None:
        """Snapshot each body's semantic face colors for later restoration."""
        for body in self._renderable_bodies:
            mesh = body.collision[0].mesh
            if hasattr(mesh.visual, "to_color"):
                mesh.visual = mesh.visual.to_color()
            self._original_visuals[body.id] = mesh.visual.face_colors.copy()

    def _reset_body_colors(self) -> None:
        """Restore all bodies to their original (semantic) face colors."""
        for body in self._renderable_bodies:
            mesh = body.collision[0].mesh
            mesh.visual.face_colors = self._original_visuals[body.id]

    def _neutralize_body_colors(self) -> None:
        """Set all bodies to a uniform neutral gray.

        Call this before ``_apply_highlight_to_group`` so only the
        highlighted objects carry distinct colors in the rendered image.
        """
        gray = np.array([180, 180, 180, 255], dtype=np.uint8)
        for body in self._renderable_bodies:
            mesh = body.collision[0].mesh
            mesh.visual.face_colors = gray

    @staticmethod
    def _apply_highlight_to_group(bodies: List[Body]) -> Dict[UUID, Color]:
        """Apply distinct highlight colors to a group of bodies."""
        colors = Color.distinct_colors(len(bodies))
        for body, color in zip(bodies, colors):
            body_mesh = body.collision[0]
            body_mesh.dye(color)
        return {body.id: color for body, color in zip(bodies, colors)}

    # Labels whose bodies should be hidden from outside-camera views so
    # they don't occlude the room interior.
    DEFAULT_HIDDEN_LABELS: Set[str] = frozenset({"ceiling"})

    def render_scene_from_camera_pose(
        self,
        camera_transform,
        output_filepath=None,
        headless=False,
        use_visual_mesh=False,
        hide_labels: Optional[Iterable[str]] = None,
    ) -> bytes:
        """Render the world from a single camera pose, return PNG bytes.

        If *use_visual_mesh* is True and texture projection was performed,
        bodies are rendered with their projected textured colors instead of
        the semantic annotation colors. Highlight colors (applied via
        ``_apply_highlight_to_group``) always take precedence because they
        overwrite ``mesh.visual`` directly.

        ``hide_labels`` lists label substrings (e.g. ``"ceiling"``) whose
        bodies should be excluded from the render but kept in the World.
        Defaults to ``{"ceiling"}``.
        """
        resolution = (1024, 768)
        hide_labels = set(
            self.DEFAULT_HIDDEN_LABELS if hide_labels is None else hide_labels
        )

        # Optionally swap in textured face colors for the duration of this render.
        swapped: List[Tuple[object, np.ndarray]] = []
        if use_visual_mesh and self._textured_visuals:
            for body in self._renderable_bodies:
                cached = self._textured_visuals.get(body.id)
                if cached is None:
                    continue
                mesh = body.collision[0].mesh
                swapped.append((mesh, mesh.visual.face_colors.copy()))
                mesh.visual.face_colors = cached

        try:
            scene = trimesh.Scene()
            for body in self._renderable_bodies:
                if any(lbl in body.name.name for lbl in hide_labels):
                    continue
                mesh = body.collision[0].mesh
                if mesh is not None:
                    scene.add_geometry(mesh, node_name=body.name.name)

            if headless:
                png = self._render_offscreen(scene, camera_transform, resolution)
            else:
                scene.graph[scene.camera.name] = camera_transform
                png = scene.save_image(resolution=resolution, visible=True)
        finally:
            for mesh, original in swapped:
                mesh.visual.face_colors = original

        png = self._autocrop_png(png)

        if output_filepath:
            with open(output_filepath, "wb") as f:
                f.write(png)
        return png

    @staticmethod
    def _autocrop_png(png_bytes: bytes, margin: int = 10) -> bytes:
        """Crop whitespace borders from a rendered PNG image."""
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        arr = np.array(img)

        non_white = np.any(arr < 250, axis=2)
        if not non_white.any():
            return png_bytes

        rows = np.any(non_white, axis=1)
        cols = np.any(non_white, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        h, w = arr.shape[:2]
        rmin = max(0, rmin - margin)
        rmax = min(h - 1, rmax + margin)
        cmin = max(0, cmin - margin)
        cmax = min(w - 1, cmax + margin)

        cropped = img.crop((cmin, rmin, cmax + 1, rmax + 1))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _render_offscreen(
        trimesh_scene: trimesh.Scene,
        camera_transform: np.ndarray,
        resolution: Tuple[int, int],
    ) -> bytes:
        """Render a trimesh scene offscreen using pyrender + EGL."""
        import os
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

        import pyrender
        from PIL import Image
        import io

        for geom in trimesh_scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                if hasattr(geom.visual, "to_color"):
                    geom.visual = geom.visual.to_color()
                geom.unmerge_vertices()

        pr_scene = pyrender.Scene()
        for name, geom in trimesh_scene.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                pr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
                pr_scene.add(pr_mesh, name=name)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        pr_scene.add(camera, pose=camera_transform)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        pr_scene.add(light, pose=camera_transform)

        renderer = pyrender.OffscreenRenderer(*resolution)
        color, _ = renderer.render(pr_scene)
        renderer.delete()

        img = Image.fromarray(color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def compute_camera_poses(
        self, bodies: Optional[List[Body]] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute four camera poses from the top corners of the scene's bounding box.

        Auto-detects the up axis from the bounding box (the axis with the
        smallest extent corresponds to floor-to-ceiling height in indoor
        scenes).  Places cameras at all four top corners of the bounding
        cube, each looking at the centroid.  If *bodies* is None, uses all
        object bodies.
        """
        if bodies is None:
            bodies = self.object_bodies
        vertices = np.vstack(
            [body.collision[0].mesh.vertices for body in bodies]
        )
        centroid = vertices.mean(axis=0)

        extent = vertices.max(axis=0) - vertices.min(axis=0)
        max_extent = extent.max()

        # Detect up axis: smallest bbox extent = floor-to-ceiling in indoor scenes
        up_idx = int(np.argmin(extent))
        ground = [i for i in range(3) if i != up_idx]

        distance = max_extent * 1.5
        height_offset = max_extent * 0.5

        up = np.zeros(3)
        up[up_idx] = 1.0

        azimuth_angles = {
            "front_left": np.radians(45),
            "front_right": np.radians(-45),
            "back_left": np.radians(135),
            "back_right": np.radians(-135),
        }

        poses: Dict[str, np.ndarray] = {}
        for name, azimuth in azimuth_angles.items():
            offset = np.zeros(3)
            offset[ground[0]] = distance * np.sin(azimuth)
            offset[ground[1]] = distance * np.cos(azimuth)
            offset[up_idx] = height_offset
            eye = centroid + offset
            poses[name] = _look_at(eye, centroid, up)

        return poses

    def export_semantic_annotation_inheritance_structure(
        self, output_directory: Path
    ) -> None:
        """Export the kinematic structure and SemanticAnnotation taxonomy to JSON."""
        from semantic_digital_twin.semantic_annotations.semantic_annotations import SemanticAnnotation
        from semantic_digital_twin.utils import InheritanceStructureExporter

        output_directory.mkdir(parents=True, exist_ok=True)

        self.world.export_kinematic_structure_tree_to_json(
            output_directory / "kinematic_structure.json",
            include_connections=False,
        )
        InheritanceStructureExporter(
            SemanticAnnotation, output_directory / "semantic_annotations.json"
        ).export()
