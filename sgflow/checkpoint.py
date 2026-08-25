"""Versioned, self-describing SGFlow checkpoints."""
from __future__ import annotations

import os
import tempfile
import warnings
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch

from .config import SGFlowConfig

FORMAT_NAME = "sgflow"
FORMAT_VERSION = 1
_ARCH_FIELDS = (
    "text_model", "text_dim", "d_model", "n_layers", "n_latents", "d_state", "expand",
    "max_objects", "d_appearance", "categories",
)


class CheckpointError(ValueError):
    """A checkpoint is malformed or incompatible with the requested runtime."""


def _safe_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        warnings.warn("This PyTorch lacks weights_only loading; using legacy torch.load", RuntimeWarning)
        return torch.load(path, map_location=map_location)


def config_from_dict(data: dict[str, Any]) -> SGFlowConfig:
    if not isinstance(data, dict):
        raise CheckpointError("checkpoint config must be a mapping")
    known = {f.name for f in fields(SGFlowConfig)}
    values = {k: v for k, v in data.items() if k in known}
    if "room_size" in values:
        values["room_size"] = tuple(values["room_size"])
    return SGFlowConfig(**values)


def validate_compatible(expected: SGFlowConfig, stored: SGFlowConfig) -> None:
    mismatches = [
        f"{name}: requested={getattr(expected, name)!r}, checkpoint={getattr(stored, name)!r}"
        for name in _ARCH_FIELDS
        if getattr(expected, name) != getattr(stored, name)
    ]
    if mismatches:
        raise CheckpointError("incompatible SGFlow checkpoint config (" + "; ".join(mismatches) + ")")


def read_checkpoint(path, *, cfg: SGFlowConfig | None = None, map_location="cpu") -> dict[str, Any]:
    payload = _safe_load(path, map_location=map_location)
    if not isinstance(payload, dict) or "model" not in payload:
        raise CheckpointError("checkpoint must be a mapping containing 'model'")
    if payload.get("format") != FORMAT_NAME:
        if "format_version" in payload or "config" in payload:
            raise CheckpointError("unrecognized checkpoint format")
        if cfg is None:
            raise CheckpointError("legacy {'model': ...} checkpoint requires an explicit cfg fallback")
        warnings.warn("Loading legacy model-only checkpoint; optimizer and encoder state are unavailable", RuntimeWarning)
        return {**payload, "config_obj": cfg, "legacy": True}
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise CheckpointError(f"unsupported checkpoint version {version!r}; expected {FORMAT_VERSION}")
    stored_cfg = config_from_dict(payload.get("config"))
    if cfg is not None:
        validate_compatible(cfg, stored_cfg)
    return {**payload, "config_obj": stored_cfg, "legacy": False}


def load_checkpoint(path, *, cfg: SGFlowConfig | None = None, map_location="cpu") -> dict[str, Any]:
    """Load and validate checkpoint metadata without constructing model objects."""
    return read_checkpoint(path, cfg=cfg, map_location=map_location)


def save_checkpoint(
    path, *, cfg: SGFlowConfig, model, encoder, texhead=None, optimizer=None,
    scaler=None, epoch: int = 0, metadata: dict[str, Any] | None = None,
) -> None:
    payload = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "config": asdict(cfg),
        "model": model.state_dict(),
        "encoder": {
            "backend_kind": encoder.backend_kind,
            "adapter": encoder.adapter_state_dict(),
        },
        "texhead": None if texhead is None else texhead.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": int(epoch),
        "metadata": dict(metadata or {}),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def restore_checkpoint(payload, *, model, encoder, texhead=None, optimizer=None, scaler=None) -> int:
    try:
        model.load_state_dict(payload["model"])
    except RuntimeError as exc:
        raise CheckpointError(f"model state is incompatible: {exc}") from exc
    enc = payload.get("encoder")
    if enc is not None:
        saved_backend = enc.get("backend_kind")
        if saved_backend != encoder.backend_kind:
            raise CheckpointError(
                f"text encoder backend mismatch: checkpoint={saved_backend!r}, runtime={encoder.backend_kind!r}"
            )
        encoder.load_adapter_state_dict(enc.get("adapter", {}))
    if payload.get("texhead") is not None:
        if texhead is None:
            raise CheckpointError("checkpoint contains TexHead state but no TexHead was provided")
        texhead.load_state_dict(payload["texhead"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return int(payload.get("epoch", 0))
