import torch
import pytest
from pathlib import Path
import shutil
from uuid import uuid4

from sgflow.checkpoint import CheckpointError, read_checkpoint, restore_checkpoint, save_checkpoint
from sgflow.config import SGFlowConfig
from sgflow.models import SceneDenoiser
from sgflow.pipeline import ScenePipeline
from sgflow.text_encoder import TextEncoder


@pytest.fixture
def tmp_path():
    # Python 3.13 on Windows applies restrictive ACLs for pytest's mode=0o700
    # temporary directories in this sandbox. A normal workspace directory keeps
    # this suite portable without changing production behavior.
    path = Path("work/test-artifacts") / uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path.resolve()
    finally:
        shutil.rmtree(path, ignore_errors=True)


def tiny_cfg(**changes):
    values = dict(
        text_model="unused", text_dim=8, d_model=8, n_layers=1, n_latents=2,
        d_state=2, expand=1, max_objects=2, d_appearance=2, flow_steps=1,
        categories=["pad", "chair", "table"], use_compile=False,
    )
    values.update(changes)
    return SGFlowConfig(**values)


def components(cfg):
    return SceneDenoiser(cfg), TextEncoder(
        cfg.d_model, cfg.text_model, cfg.text_dim, backend_kind="hash",
    )


def test_checkpoint_roundtrip_preserves_config_and_text_encoding(tmp_path):
    cfg = tiny_cfg()
    model, encoder = components(cfg)
    expected, expected_mask = encoder(["small oak table"])
    path = tmp_path / "model.pt"
    save_checkpoint(path, cfg=cfg, model=model, encoder=encoder, epoch=3)

    payload = read_checkpoint(path)
    restored_cfg = payload["config_obj"]
    restored_model, restored_encoder = components(restored_cfg)
    restore_checkpoint(payload, model=restored_model, encoder=restored_encoder)
    actual, actual_mask = restored_encoder(["small oak table"])

    assert restored_cfg == cfg
    assert payload["epoch"] == 3
    assert torch.equal(actual_mask, expected_mask)
    assert torch.equal(actual, expected)
    assert not any("hash_table" in key for key in payload["encoder"]["adapter"])


def test_backend_mismatch_is_clear(tmp_path):
    cfg = tiny_cfg()
    model, encoder = components(cfg)
    path = tmp_path / "model.pt"
    save_checkpoint(path, cfg=cfg, model=model, encoder=encoder)
    payload = read_checkpoint(path)
    payload["encoder"]["backend_kind"] = "sentence_transformer"

    with pytest.raises(CheckpointError, match="backend mismatch"):
        restore_checkpoint(payload, model=model, encoder=encoder)


def test_untrained_pipeline_is_explicitly_gated():
    with pytest.raises(ValueError, match="checkpoint is required"):
        ScenePipeline(tiny_cfg(), device="cpu")
    ScenePipeline(tiny_cfg(), device="cpu", allow_untrained=True)


def test_seeded_generation_repeats_without_changing_global_rng(tmp_path):
    cfg = tiny_cfg()
    model, encoder = components(cfg)
    path = tmp_path / "model.pt"
    save_checkpoint(path, cfg=cfg, model=model, encoder=encoder)
    pipeline = ScenePipeline(cfg, device="cpu", checkpoint=path)

    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    first = pipeline.generate("a chair", refine_steps=0, seed=99)
    after = torch.random.get_rng_state().clone()
    second = pipeline.generate("a chair", refine_steps=0, seed=99)

    assert torch.equal(before, after)
    assert first.fingerprint() == second.fingerprint()
    assert 1 <= first.n <= cfg.max_generated_objects


@pytest.mark.parametrize("kwargs", [
    {"d_model": 10}, {"flow_steps": 0}, {"room_size": (8, -1, 4)},
    {"categories": ["chair", "pad"]}, {"texture_mode": "other"},
    {"texture_batch_size": 0}, {"texture_train_size": 1},
])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        tiny_cfg(**kwargs)


def test_device_validation_and_atomic_write(tmp_path, monkeypatch):
    from sgflow import device

    destination = tmp_path / "device.json"
    monkeypatch.setattr(device, "_CONFIG_PATH", str(destination))
    device.set_device("GPU")
    assert destination.read_text(encoding="utf-8") == '{"device": "gpu"}'
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError):
        device.set_device("tpu")


def test_checkpoint_resume_metadata(tmp_path):
    cfg = tiny_cfg()
    model, encoder = components(cfg)
    optimizer = torch.optim.AdamW([*model.parameters(), *encoder.proj.parameters()])
    path = tmp_path / "resume.pt"
    save_checkpoint(
        path, cfg=cfg, model=model, encoder=encoder, optimizer=optimizer,
        epoch=7, metadata={"averaged_logs": {"flow": 1.25}},
    )
    payload = read_checkpoint(path, cfg=cfg)
    new_model, new_encoder = components(cfg)
    new_optimizer = torch.optim.AdamW([*new_model.parameters(), *new_encoder.proj.parameters()])
    epoch = restore_checkpoint(
        payload, model=new_model, encoder=new_encoder, optimizer=new_optimizer,
    )
    assert epoch == 7
    assert payload["metadata"]["averaged_logs"]["flow"] == 1.25
