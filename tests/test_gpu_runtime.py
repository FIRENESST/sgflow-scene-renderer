import pytest
import torch

from sgflow.config import SGFlowConfig
from sgflow.device import (
    DeviceUnavailableError,
    configure_cuda_runtime,
    device_report,
    read_device_preference,
    resolve_amp_dtype,
    resolve_device,
    validate_device,
)
from sgflow.flow_matching import RectifiedFlow
from sgflow.models import SceneDenoiser
from sgflow.pipeline import ScenePipeline
from sgflow.scene_graph import SceneGraph
from sgflow.tex_assets import TexAssetExporter


def tiny_cfg(**changes):
    values = dict(
        text_dim=8, d_model=8, n_layers=1, n_latents=2, d_state=2,
        expand=1, max_objects=4, d_appearance=2, flow_steps=1,
        categories=["pad", "chair", "table"], use_compile=False,
    )
    values.update(changes)
    return SGFlowConfig(**values)


def test_cpu_remains_explicitly_selectable(monkeypatch, tmp_path):
    from sgflow import device

    config_path = tmp_path / "device.json"
    config_path.write_text('{"device":"cpu"}', encoding="utf-8")
    monkeypatch.setattr(device, "_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("SGFLOW_DEVICE", raising=False)
    assert read_device_preference().source == "config"
    assert resolve_device() == "cpu"
    assert validate_device("cpu") == "cpu"
    assert resolve_amp_dtype(tiny_cfg(), "cpu") == torch.float32


def test_explicit_cuda_is_strict_but_auto_can_fallback(monkeypatch, tmp_path):
    from sgflow import device

    missing_config = tmp_path / "missing.json"
    monkeypatch.setattr(device, "_CONFIG_PATH", str(missing_config))
    monkeypatch.setattr(device.torch.version, "cuda", None)
    monkeypatch.setenv("SGFLOW_DEVICE", "cuda")
    with pytest.raises(DeviceUnavailableError, match="CPU-only"):
        resolve_device()
    monkeypatch.setenv("SGFLOW_DEVICE", "auto")
    assert resolve_device() == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime is unavailable")
def test_cuda_model_forward_backward_and_amp():
    cfg = tiny_cfg(amp_dtype="auto", cuda_allow_tf32=True)
    device = validate_device("cuda", smoke_test=True)
    profile = configure_cuda_runtime(cfg, device)
    dtype = resolve_amp_dtype(cfg, device)
    assert profile["tf32_enabled"] is True
    assert dtype in {torch.float16, torch.bfloat16}

    model = SceneDenoiser(cfg).to(device).train()
    flow = RectifiedFlow(cfg)
    cat = torch.tensor([[1, 2, 0, 0]], device=device)
    mask = cat.ne(0)
    z1 = torch.randn(1, cfg.max_objects, cfg.latent_dim, device=device)
    text = torch.randn(1, 3, cfg.d_model, device=device)
    text_mask = torch.ones(1, 3, dtype=torch.bool, device=device)
    with torch.autocast("cuda", dtype=dtype):
        loss, metrics = flow.loss(model, z1, cat, text, text_mask, mask)
    loss.backward()
    torch.cuda.synchronize()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    report = device_report(smoke_test=True)
    assert report["resolved"].startswith("cuda")
    assert report["cuda"]["smoke_test"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime is unavailable")
def test_cuda_generated_texture_path(tmp_path):
    cfg = tiny_cfg(texture_size_limit=32, texture_train_size=8)
    scene = SceneGraph.from_objects([
        {
            "category": "chair", "position": [0, 0, 0.5], "scale": [1, 1, 1],
            "appearance": [0.0, 0.0],
        }
    ], cfg.categories, cfg.d_appearance)
    exporter = TexAssetExporter(cfg, device="cuda", seed=3, batch_size=1)
    manifest = exporter.export_scene(scene, out_dir=str(tmp_path / "textures"), size=8)
    assert exporter.device.type == "cuda"
    assert manifest["materials"][0]["source"] == "generated"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime is unavailable")
def test_cuda_pipeline_generation_path():
    cfg = tiny_cfg(text_model="unused", use_compile=False)
    pipeline = ScenePipeline(cfg, device="cuda", allow_untrained=True)
    scene = pipeline.generate("a chair and a table", steps=1, refine_steps=1, seed=11)
    assert pipeline.device.type == "cuda"
    assert pipeline.amp_enabled is True
    assert scene.n >= 1
