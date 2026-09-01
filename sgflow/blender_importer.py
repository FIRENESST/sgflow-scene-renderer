"""Rebuild an SGFlow scene graph in Blender and render it.

The command line after Blender's ``--`` separator is deliberately small:

    blender --background --python sgflow/blender_importer.py -- scene.json render.png
    blender --background --python sgflow/blender_importer.py -- scene.json materials.json render.png

Use ``--full-replace`` after the separator to clear an interactive Blender
scene too. Background invocations replace the scene automatically.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from numbers import Real

import bpy
from mathutils import Matrix, Vector


class UsageError(ValueError):
    """Raised for a malformed Blender bridge command line."""


class MaterialManifestError(ValueError):
    """Raised when a material manifest cannot safely be applied to a scene."""


def _finite_vector(value, width: int, label: str, *, positive: bool = False):
    if not isinstance(value, (list, tuple)) or len(value) != width:
        raise ValueError(f"{label} must contain exactly {width} numbers.")
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
        raise ValueError(f"{label} must contain only numbers.")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains non-finite values.")
    if positive and any(item <= 0 for item in result):
        raise ValueError(f"{label} must contain only positive values.")
    return result


def validate_scene_payload(scene: dict) -> list[dict]:
    """Validate all geometry before background mode mutates the Blender file."""
    if not isinstance(scene, dict):
        raise ValueError("Scene JSON must contain an object.")
    version = scene.get("schema_version")
    if version is not None and (type(version) is not int or version != 1):
        raise ValueError(f"Unsupported scene schema version {version!r}.")
    objects = scene.get("objects")
    if not isinstance(objects, list):
        raise ValueError("Scene JSON must contain an 'objects' list.")
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise ValueError(f"objects[{index}] must be an object.")
        if not isinstance(item.get("category"), str) or not item["category"]:
            raise ValueError(f"objects[{index}].category must be a non-empty string.")
        _finite_vector(item.get("position"), 3, f"objects[{index}].position")
        _finite_vector(item.get("rotation6d"), 6, f"objects[{index}].rotation6d")
        _finite_vector(item.get("scale"), 3, f"objects[{index}].scale", positive=True)
        appearance = item.get("appearance", [])
        if not isinstance(appearance, (list, tuple)):
            raise ValueError(f"objects[{index}].appearance must be a numeric list.")
        _finite_vector(appearance, len(appearance), f"objects[{index}].appearance")
    return objects


@dataclass(frozen=True)
class BridgeArguments:
    scene_path: str
    render_path: str
    materials_path: str | None = None
    full_replace: bool = False


def parse_bridge_argv(argv: list[str]) -> BridgeArguments:
    """Parse Blender's full argv without depending on :mod:`bpy` state."""
    if "--" not in argv:
        raise UsageError(
            "Missing Blender '--' separator. Usage: blender ... -- "
            "scene.json [materials.json] render.png"
        )
    trailing = argv[argv.index("--") + 1:]
    full_replace = False
    if "--full-replace" in trailing:
        trailing.remove("--full-replace")
        full_replace = True
    if any(arg.startswith("-") for arg in trailing):
        raise UsageError("Unknown bridge option. Only --full-replace is supported after '--'.")
    if len(trailing) == 2:
        scene_path, render_path = trailing
        return BridgeArguments(scene_path, render_path, full_replace=full_replace)
    if len(trailing) == 3:
        scene_path, materials_path, render_path = trailing
        return BridgeArguments(scene_path, render_path, materials_path, full_replace)
    raise UsageError(
        "Expected scene.json render.png or scene.json materials.json render.png after '--'."
    )


def resolve_texture_path(path: str, manifest_dir: str, cwd: str | None = None) -> str:
    """Resolve a texture relative to its manifest, falling back to the CWD."""
    if not isinstance(path, str) or not path:
        raise MaterialManifestError("Texture paths must be non-empty strings.")
    if os.path.isabs(path):
        return os.path.abspath(path)
    manifest_candidate = os.path.abspath(os.path.join(manifest_dir, path))
    if os.path.exists(manifest_candidate):
        return manifest_candidate
    return os.path.abspath(os.path.join(cwd or os.getcwd(), path))


def validate_material_manifest(
    manifest: dict,
    scene_objects: list[dict],
    manifest_path: str,
    *,
    check_files: bool = True,
) -> dict[int, dict]:
    """Validate and normalise a manifest before any Blender objects are made."""
    if not isinstance(manifest, dict) or not isinstance(manifest.get("materials"), list):
        raise MaterialManifestError("Material manifest must contain a 'materials' list.")
    version = manifest.get("manifest_version")
    if version is not None and (type(version) is not int or version != 1):
        raise MaterialManifestError(f"Unsupported material manifest version {version!r}.")

    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    by_index: dict[int, dict] = {}
    for position, raw_entry in enumerate(manifest["materials"]):
        if not isinstance(raw_entry, dict):
            raise MaterialManifestError(f"materials[{position}] must be an object.")
        index = raw_entry.get("object_index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(scene_objects):
            raise MaterialManifestError(f"materials[{position}] has invalid object_index {index!r}.")
        if index in by_index:
            raise MaterialManifestError(f"Duplicate material entry for object_index {index}.")
        category = raw_entry.get("category")
        expected_category = scene_objects[index].get("category")
        if category != expected_category:
            raise MaterialManifestError(
                f"materials[{position}] category {category!r} does not match "
                f"scene object {index} category {expected_category!r}."
            )

        entry = dict(raw_entry)
        source = entry.get("source")
        if source not in (None, "library", "generated", "white"):
            raise MaterialManifestError(
                f"materials[{position}] has unsupported source {source!r}."
            )
        textures = entry.get("textures")
        if source == "white" and textures is not None:
            raise MaterialManifestError(
                f"materials[{position}] source='white' requires textures=null."
            )
        if textures is not None:
            if not isinstance(textures, dict):
                raise MaterialManifestError(f"materials[{position}].textures must be an object or null.")
            albedo = textures.get("albedo")
            if not isinstance(albedo, str) or not albedo:
                raise MaterialManifestError(f"materials[{position}] with textures requires an albedo path.")
            resolved = {}
            for kind, texture_path in textures.items():
                if kind not in ("albedo", "roughness", "normal"):
                    continue
                resolved_path = resolve_texture_path(texture_path, manifest_dir)
                if check_files and not os.path.isfile(resolved_path):
                    raise MaterialManifestError(
                        f"materials[{position}] {kind} texture does not exist: {resolved_path}"
                    )
                resolved[kind] = resolved_path
            entry["textures"] = resolved
        by_index[index] = entry
    return by_index


def material_signature(entry: dict) -> tuple | None:
    """Return a stable cache key for a fully textured material."""
    textures = entry.get("textures")
    if not textures:
        return None
    params = entry.get("params") or {}
    return (
        tuple(sorted((kind, os.path.normcase(os.path.abspath(path))) for kind, path in textures.items())),
        tuple(sorted((key, repr(value)) for key, value in params.items())),
    )


def fallback_base_color(entry: dict | None, appearance) -> tuple[float, float, float]:
    """Resolve an untextured material color, preserving the white-model contract."""
    if entry is not None and entry.get("source") == "white":
        return (1.0, 1.0, 1.0)
    values = appearance or []
    return tuple(abs(values[index]) % 1.0 if len(values) > index else 0.5 for index in range(3))


def clear_scene():
    """删除当前场景全部对象。用 data API 而非 bpy.ops，避免依赖 UI context。"""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _stable_rot6d_basis(r6, eps: float = 1e-6):
    """Return the same deterministic 6D rotation basis as ``math3d``.

    Blender runs in its own Python environment, so importing the PyTorch
    implementation here would make the bridge unnecessarily fragile.  Keep
    this small scalar equivalent in sync instead, including its completion
    for zero and collinear input vectors.
    """

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def normalized(v, fallback=None):
        length = dot(v, v) ** 0.5
        if length <= eps:
            return fallback
        return tuple(value / length for value in v)

    def cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    a1 = tuple(float(value) for value in r6[:3])
    a2 = tuple(float(value) for value in r6[3:])
    b1 = normalized(a1, (1.0, 0.0, 0.0))
    projection = dot(b1, a2)
    a2_orth = tuple(value - projection * axis for value, axis in zip(a2, b1))
    b2 = normalized(a2_orth)
    if b2 is None:
        least_aligned = min(range(3), key=lambda index: abs(b1[index]))
        axis = tuple(1.0 if index == least_aligned else 0.0 for index in range(3))
        b2 = normalized(cross(axis, b1))
    b3 = cross(b1, b2)
    return b1, b2, b3


def rot6d_to_mat3(r6):
    """与训练侧一致、可处理退化输入的 Gram-Schmidt 正交化。"""
    return Matrix(_stable_rot6d_basis(r6)).transposed()


def make_proxy(name: str, category: str):
    """单立方体代理（无细节信息时的兜底）。"""
    mesh = bpy.data.meshes.new(f"mesh_{name}")
    verts = [(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5),
             (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5)]
    faces = [(0, 1, 2, 3), (7, 6, 5, 4), (4, 5, 1, 0), (5, 6, 2, 1),
             (6, 7, 3, 2), (7, 4, 0, 3)]
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(f"{name}_{category}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


# ---- 精细等级与参数化部件建模（自包含，不依赖 sgflow 包上下文） ----

_DETAIL_LEVELS = {
    1: {"parts_limit": 1, "subdiv": 0, "smooth": False},
    2: {"parts_limit": 3, "subdiv": 1, "smooth": False},
    3: {"parts_limit": 6, "subdiv": 2, "smooth": True},
    4: {"parts_limit": 12, "subdiv": 2, "smooth": True},
    5: {"parts_limit": None, "subdiv": 3, "smooth": True},
}


def _clamp_level(level):
    try:
        v = int(level)
    except Exception:
        return 3
    return max(1, min(5, v))


def _build_detail(category, detail):
    """归一化 detail dict：确保有 parts 列表。无有效 detail 时返回空 parts（调用方会回退到 make_proxy）。"""
    if isinstance(detail, dict) and isinstance(detail.get("parts"), list) and detail["parts"]:
        return {"parts": list(detail["parts"]), "smooth": bool(detail.get("smooth", True))}
    return {"parts": [], "smooth": True}


def _apply_level(detail, level):
    spec = _DETAIL_LEVELS[_clamp_level(level)]
    parts = detail["parts"]
    limit = spec["parts_limit"]
    if limit is not None:
        parts = parts[:limit]
    return {
        "parts": parts,
        "smooth": bool(detail.get("smooth", True)) and spec["smooth"],
        "subdiv": spec["subdiv"],
    }


def _read_detail_level(scene):
    """从 scene JSON 的 metadata 或根级字段读取精细等级，默认 3。"""
    if not isinstance(scene, dict):
        return 3
    for key in ("detail_level", "geometry_detail_level"):
        for root in (scene, scene.get("metadata", {})):
            val = root.get(key)
            if val is not None:
                return _clamp_level(val)
    return 3


def _part_mesh(name: str, kind: str, offset, size, subdiv: int, smooth: bool):
    """构建单个参数化部件。offset/size 是物体 OBB 归一化局部坐标（±0.5 占满整个 OBB）。"""
    import bmesh
    bm = bmesh.new()
    try:
        if kind == "box":
            bmesh.ops.create_cube(bm, size=1.0)
        elif kind == "sphere":
            seg = 8 * (2 ** subdiv)
            bmesh.ops.create_uvsphere(bm, u_segments=max(8, seg), v_segments=max(4, seg // 2), radius=0.5)
        elif kind == "cylinder":
            seg = 8 * (2 ** subdiv)
            bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=max(8, seg),
                                  radius1=0.5, radius2=0.5, depth=1.0)
        elif kind == "cone":
            seg = 8 * (2 ** subdiv)
            bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=max(8, seg),
                                  radius1=0.5, radius2=0.0, depth=1.0)
        else:
            bmesh.ops.create_cube(bm, size=1.0)
        mesh = bpy.data.meshes.new(f"mesh_{name}")
        bm.to_mesh(mesh)
        mesh.update()
    finally:
        bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(offset)
    obj.scale = Vector(size)
    if smooth:
        for poly in mesh.polygons:
            poly.use_smooth = True
    return obj


def _custom_mesh_object(name: str, mesh_data: dict, smooth: bool = True):
    """从顶点/面数据直接构建 mesh。vertices 是归一化坐标（±0.5 占满 OBB），faces 是顶点索引。"""
    vertices = mesh_data.get("vertices")
    faces = mesh_data.get("faces")
    if not isinstance(vertices, list) or not isinstance(faces, list):
        raise ValueError("custom_mesh must contain vertices and faces lists")
    verts = [Vector(v) for v in vertices]
    # 校验面索引
    n = len(verts)
    for fi, face in enumerate(faces):
        if not isinstance(face, list) or len(face) < 3:
            raise ValueError(f"face {fi} must have at least 3 vertices")
        if any(not isinstance(idx, int) or idx < 0 or idx >= n for idx in face):
            raise ValueError(f"face {fi} has invalid vertex index")
    mesh = bpy.data.meshes.new(f"mesh_{name}")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    if smooth:
        for poly in mesh.polygons:
            poly.use_smooth = True
    return obj


def build_detailed_object(name: str, category: str, detail: dict | None, level: int,
                          custom_mesh: dict | None = None):
    """构建单个物体。L6 且有 custom_mesh 时直接建 mesh；否则按 detail + level 参数化。"""
    level = _clamp_level(level)
    if level >= 6 and custom_mesh is not None:
        return _custom_mesh_object(f"{name}_{category}", custom_mesh, smooth=True)
    spec = _build_detail(category, detail)
    spec = _apply_level(spec, level)
    if not spec["parts"]:
        return None
    parent = bpy.data.objects.new(f"{name}_{category}", None)
    bpy.context.scene.collection.objects.link(parent)
    for idx, part in enumerate(spec["parts"]):
        child = _part_mesh(
            f"{name}_p{idx:02d}", part["kind"], part["offset"], part["size"],
            spec["subdiv"], spec["smooth"],
        )
        child.parent = parent
        bpy.context.scene.collection.objects.link(child)
    return parent


def _principled_bsdf(node_tree):
    """按类型定位 Principled BSDF 节点。

    Blender 的“翻译新数据名称”选项会把默认节点改名为界面语言
    （如“原理化 BSDF”），因此不能按节点名字符串索引。
    """
    for node in node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    bsdf = node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    for node in node_tree.nodes:
        if node.type == "OUTPUT_MATERIAL":
            node_tree.links.new(bsdf.outputs["BSDF"], node.inputs["Surface"])
            break
    return bsdf


def _build_material(name: str, entry: dict | None, fallback_rgb, image_cache: dict[str, object]):
    """Build a node material, sharing already-loaded Blender images."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = _principled_bsdf(mat.node_tree)
    if entry is None or not entry.get("textures"):
        bsdf.inputs["Base Color"].default_value = (*fallback_rgb, 1.0)
        return mat
    paths = entry["textures"]
    params = entry.get("params") or {}
    bsdf.inputs["Roughness"].default_value = params.get("roughness", 0.5)
    bsdf.inputs["Metallic"].default_value = params.get("metallic", 0.0)

    def load_img(path, non_color=False):
        key = os.path.normcase(os.path.abspath(path))
        img = image_cache.get(key)
        if img is None:
            img = bpy.data.images.load(path, check_existing=True)
            image_cache[key] = img
        if non_color:
            img.colorspace_settings.name = "Non-Color"
        return img

    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = load_img(paths["albedo"])
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    if paths.get("roughness"):
        rough_tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
        rough_tex.image = load_img(paths["roughness"], non_color=True)
        mat.node_tree.links.new(rough_tex.outputs["Color"], bsdf.inputs["Roughness"])
    if paths.get("normal"):
        nrm_tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
        nrm_tex.image = load_img(paths["normal"], non_color=True)
        nrm_map = mat.node_tree.nodes.new("ShaderNodeNormalMap")
        nrm_map.inputs["Strength"].default_value = params.get("bump", 1.0)
        mat.node_tree.links.new(nrm_tex.outputs["Color"], nrm_map.inputs["Color"])
        mat.node_tree.links.new(nrm_map.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def build(scene: dict, materials_manifest: dict | None = None, *, manifest_path: str | None = None, replace_scene: bool = False, detail_level: int | None = None):
    objects = validate_scene_payload(scene)
    mat_by_idx = (
        validate_material_manifest(
            materials_manifest, objects,
            manifest_path or os.path.join(os.getcwd(), "materials.json"),
        )
        if materials_manifest is not None else {}
    )
    if replace_scene:
        clear_scene()
    object_details = None
    custom_meshes = None
    if isinstance(scene, dict):
        meta = scene.get("metadata")
        if isinstance(meta, dict):
            if isinstance(meta.get("object_details"), list):
                object_details = meta["object_details"]
            if isinstance(meta.get("custom_meshes"), list):
                custom_meshes = meta["custom_meshes"]
    if detail_level is None:
        detail_level = _read_detail_level(scene)
    image_cache: dict[str, object] = {}
    material_cache: dict[tuple, object] = {}
    for i, o in enumerate(objects):
        detail = None
        custom_mesh = None
        if object_details is not None and i < len(object_details):
            detail = object_details[i]
        if custom_meshes is not None and i < len(custom_meshes):
            custom_mesh = custom_meshes[i]
        obj = build_detailed_object(f"obj{i:03d}", o["category"], detail, detail_level,
                                    custom_mesh=custom_mesh)
        if obj is None:
            obj = make_proxy(f"obj{i:03d}", o["category"])
        t, s, R = Vector(o["position"]), Vector(o["scale"]), rot6d_to_mat3(o["rotation6d"])
        obj.matrix_world = Matrix.Translation(t) @ R.to_4x4() @ Matrix.Diagonal((*s, 1.0))
        entry = mat_by_idx.get(i)
        fallback = fallback_base_color(entry, o.get("appearance"))
        signature = material_signature(entry) if entry else None
        mat = material_cache.get(signature) if signature else None
        if mat is None:
            mat = _build_material(f"mat_{o['category']}_{i:03d}", entry, fallback, image_cache)
            if signature:
                material_cache[signature] = mat
        # 多部件物体：材质挂到所有子部件；单代理：直接挂
        targets = [obj] + list(obj.children) if obj.children else [obj]
        for target in targets:
            if target.data is not None and hasattr(target.data, "materials"):
                target.data.materials.append(mat)
    # 地面 / 相机 / 太阳灯：全部走 data API，后台与 UI 行为一致
    ground_mesh = bpy.data.meshes.new("mesh_ground")
    s = 25.0
    ground_mesh.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    ground_mesh.update()
    ground = bpy.data.objects.new("ground", ground_mesh)
    scn = bpy.context.scene
    scn.collection.objects.link(ground)

    cam_data = bpy.data.cameras.new("camera")
    cam = bpy.data.objects.new("camera", cam_data)
    cam.location = (8, -8, 6)
    scn.collection.objects.link(cam)
    # Track the room center instead of relying on an Euler triple that changes
    # meaning with camera orientation conventions.
    direction = Vector((0.0, 0.0, 1.0)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scn.camera = cam

    sun_data = bpy.data.lights.new("sun", type="SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("sun", sun_data)
    sun.location = (4, 4, 8)
    scn.collection.objects.link(sun)
    configure_render_engine(scn)
    scn.render.resolution_x = scn.render.resolution_y = 512
    scn.render.resolution_percentage = 100
    return scn


def configure_render_engine(scene) -> str:
    """Select the Eevee identifier used by Blender 4.x/5.x or 3.x."""
    failures = []
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = engine
            return engine
        except (AttributeError, TypeError, ValueError) as exc:
            failures.append(f"{engine}: {exc}")
    raise RuntimeError("No supported Eevee render engine (" + "; ".join(failures) + ")")


def main(argv: list[str] | None = None):
    args = parse_bridge_argv(sys.argv if argv is None else argv)
    with open(args.scene_path, encoding="utf-8") as f:
        scene = json.load(f)
    manifest = None
    if args.materials_path:
        with open(args.materials_path, encoding="utf-8") as f:
            manifest = json.load(f)
    replace_scene = args.full_replace or bool(getattr(bpy.app, "background", False))
    scn = build(scene, manifest, manifest_path=args.materials_path, replace_scene=replace_scene)
    scn.render.filepath = os.path.abspath(args.render_path)
    output_dir = os.path.dirname(scn.render.filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    try:
        main()
    except (UsageError, MaterialManifestError, RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"blender_importer: {exc}", file=sys.stderr)
        raise SystemExit(2)
