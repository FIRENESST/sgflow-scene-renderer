import json

import pytest
import torch

from sgflow.config import SGFlowConfig
from sgflow.scene_graph import SceneGraph
from sgflow.tex_assets import TexAssetExporter
from sgflow.tex_raster import render_textures
from sgflow.texhead import TexHead


def _cfg():
    return SGFlowConfig(d_model=8, d_appearance=2)


def _scene(cfg, count=2):
    return SceneGraph(
        cfg.categories,
        torch.tensor([3 + i for i in range(count)], dtype=torch.long),
        torch.zeros(count, 3), torch.tensor([[1, 0, 0, 0, 1, 0]] * count, dtype=torch.float32),
        torch.zeros(count, 3), torch.zeros(count, cfg.d_appearance),
    )


def _empty_scene(cfg):
    return SceneGraph(cfg.categories, torch.empty(0, dtype=torch.long), torch.empty(0, 3),
                      torch.empty(0, 6), torch.empty(0, 3), torch.empty(0, cfg.d_appearance))


def test_loss_is_finite_and_all_control_heads_receive_gradients():
    torch.manual_seed(3)
    head = TexHead(8, 4, 2)
    # Coincident palette heads are a legitimate collapsed state; the palette
    # norm must still have a finite backward pass there.
    head.accent_color.load_state_dict(head.base_color.state_dict())
    out = head(torch.tensor([1]), torch.zeros(1, 2), torch.zeros(1, 8))
    for value in out.values():
        value.retain_grad()
    rendered = render_textures(out, size=8, seed=4)
    loss = TexHead.loss_fn(out, rendered)
    assert torch.isfinite(loss)
    loss.backward()
    for value in out.values():
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
    for layer in (
        head.mix_logits, head.base_color, head.accent_color, head.params, head.bump,
    ):
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0


def test_raster_shapes_ranges_and_size_errors():
    head = TexHead(8, 4, 2)
    out = head(torch.tensor([1, 2]), torch.zeros(2, 2), torch.zeros(2, 8))
    albedo, rough, normal = render_textures(out, size=8, seed=12)
    assert albedo.shape == normal.shape == (2, 3, 8, 8)
    assert rough.shape == (2, 1, 8, 8)
    assert all(torch.all((0 <= image) & (image <= 1)) for image in (albedo, rough, normal))
    for bad in (1, 4097, 2.5):
        with pytest.raises(ValueError):
            render_textures(out, size=bad)


def test_generated_export_is_deterministic_and_batched(tmp_path, monkeypatch):
    cfg, sg = _cfg(), _scene(_cfg(), 3)
    calls = []
    from sgflow import tex_assets
    original = tex_assets.render_textures
    monkeypatch.setattr(tex_assets, "render_textures", lambda *args, **kwargs: (calls.append(args[0]["freq"].size(0)) or original(*args, **kwargs)))
    first = TexAssetExporter(cfg, seed=19, batch_size=2).export_scene(sg, out_dir=str(tmp_path / "one"), size=8)
    assert calls == [2, 1]
    second = TexAssetExporter(cfg, seed=19, batch_size=1).export_scene(sg, out_dir=str(tmp_path / "two"), size=8)
    assert first["materials"] == second["materials"]
    for entry in first["materials"]:
        filename = entry["textures"]["albedo"]
        assert (tmp_path / "one" / filename).read_bytes() == (tmp_path / "two" / filename).read_bytes()
    from PIL import Image
    assert Image.open(tmp_path / "one" / first["materials"][0]["textures"]["roughness"]).mode == "L"


def test_empty_scene_and_library_portable_hit_miss(tmp_path):
    cfg = _cfg()
    empty = TexAssetExporter(cfg).export_scene(_empty_scene(cfg), out_dir=str(tmp_path / "empty"), size=8)
    assert empty == {"manifest_version": 1, "materials": []}
    lib = tmp_path / "library"
    (lib / "table").mkdir(parents=True)
    (lib / "table" / "albedo.png").write_bytes(b"not read")
    sg = _scene(cfg, 2)
    manifest = TexAssetExporter(cfg, texture_mode="library", texture_lib=str(lib)).export_scene(sg, out_dir=str(tmp_path / "out"))
    assert manifest["materials"][0]["source"] == "library"
    assert not manifest["materials"][0]["textures"]["albedo"].startswith(str(lib))
    assert manifest["materials"][1]["source"] == "white"
    assert json.loads((tmp_path / "out" / "materials.json").read_text())["manifest_version"] == 1
