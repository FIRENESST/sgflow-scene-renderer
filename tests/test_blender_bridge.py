"""Pure bridge checks; Blender itself is intentionally not required."""
import importlib.util
import math
import sys
import types

import pytest


@pytest.fixture(scope="module")
def bridge():
    bpy = types.ModuleType("bpy")
    bpy.app = types.SimpleNamespace(background=False)
    mathutils = types.ModuleType("mathutils")
    mathutils.Matrix = object
    mathutils.Vector = object
    sys.modules.setdefault("bpy", bpy)
    sys.modules.setdefault("mathutils", mathutils)
    spec = importlib.util.spec_from_file_location(
        "blender_importer_under_test", "sgflow/blender_importer.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_cli_two_and_three_argument_contracts(bridge):
    two = bridge.parse_bridge_argv(["blender", "--", "scene.json", "render.png"])
    assert (two.scene_path, two.materials_path, two.render_path) == ("scene.json", None, "render.png")
    three = bridge.parse_bridge_argv(
        ["blender", "--", "--full-replace", "scene.json", "materials.json", "render.png"]
    )
    assert three.materials_path == "materials.json"
    assert three.full_replace is True


@pytest.mark.parametrize("argv", [["blender"], ["blender", "--", "scene.json"], ["blender", "--", "--bad", "a", "b"]])
def test_parse_cli_rejects_invalid_contracts(bridge, argv):
    with pytest.raises(bridge.UsageError):
        bridge.parse_bridge_argv(argv)


def test_manifest_paths_prefer_manifest_dir_then_legacy_cwd(bridge, monkeypatch):
    manifest_dir = "C:/manifest"
    cwd = "C:/legacy"
    monkeypatch.setattr(bridge.os.path, "exists", lambda path: path.endswith("manifest\\albedo.png"))
    assert bridge.resolve_texture_path("albedo.png", manifest_dir, cwd).endswith("manifest\\albedo.png")
    assert bridge.resolve_texture_path("rough.png", manifest_dir, cwd).endswith("legacy\\rough.png")


def test_white_manifest_source_forces_white_fallback(bridge):
    entry = {"source": "white", "textures": None}
    assert bridge.fallback_base_color(entry, [0.0, 0.2, 0.4]) == (1.0, 1.0, 1.0)
    assert bridge.fallback_base_color(None, [0.0, 0.2, 0.4]) == (0.0, 0.2, 0.4)


@pytest.mark.parametrize(
    "rotation",
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ],
)
def test_rot6d_basis_is_orthonormal_for_degenerate_inputs(bridge, rotation):
    basis = bridge._stable_rot6d_basis(rotation)
    for axis in basis:
        assert math.sqrt(sum(value * value for value in axis)) == pytest.approx(1.0)
    for first, second in ((basis[0], basis[1]), (basis[0], basis[2]), (basis[1], basis[2])):
        assert sum(a * b for a, b in zip(first, second)) == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize(
    "materials, message",
    [
        ([{"object_index": 0, "category": "chair", "textures": {"albedo": "a.png"}}, {"object_index": 0, "category": "chair", "textures": None}], "Duplicate"),
        ([{"object_index": 1, "category": "chair", "textures": None}], "invalid object_index"),
        ([{"object_index": 0, "category": "table", "textures": None}], "does not match"),
        ([{"object_index": 0, "category": "chair", "textures": {}}], "requires an albedo"),
        ([{"object_index": 0, "category": "chair", "source": "white", "textures": {"albedo": "a.png"}}], "requires textures=null"),
        ([{"object_index": 0, "category": "chair", "source": "mystery", "textures": None}], "unsupported source"),
    ],
)
def test_manifest_validation_rejects_unsafe_entries(bridge, materials, message):
    with pytest.raises(bridge.MaterialManifestError, match=message):
        bridge.validate_material_manifest(
            {"materials": materials}, [{"category": "chair"}], "C:/manifest/materials.json", check_files=False
        )
