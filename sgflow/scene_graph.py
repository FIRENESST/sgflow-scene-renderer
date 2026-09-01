"""Validated, versioned scene-graph interchange for SGFlow."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any

import torch

from .math3d import morton_order, rot6d_to_matrix


SCHEMA_VERSION = 1
PAD_ID = 0


def _validate_categories(categories: list) -> None:
    if not isinstance(categories, list) or not categories:
        raise ValueError("categories must be a non-empty list")
    if any(not isinstance(name, str) or not name for name in categories):
        raise ValueError("categories must contain only non-empty strings")
    if len(set(categories)) != len(categories):
        raise ValueError("categories must not contain duplicate names")
    if categories[PAD_ID] != "pad":
        raise ValueError("categories[0] must be the reserved 'pad' category")


@dataclass
class SceneGraph:
    """A non-PAD set of scene objects and its validated tensor representation."""

    categories: list
    cat: torch.Tensor
    pos: torch.Tensor
    rot6d: torch.Tensor
    log_scale: torch.Tensor
    appearance: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_categories(self.categories)
        for name in ("cat", "pos", "rot6d", "log_scale", "appearance"):
            if not isinstance(getattr(self, name), torch.Tensor):
                raise ValueError(f"{name}: must be a torch.Tensor")
        if self.cat.ndim != 1:
            raise ValueError("cat: must have shape (N,)")
        if self.cat.dtype == torch.bool or self.cat.is_floating_point() or self.cat.is_complex():
            raise ValueError("cat: must use an integer dtype")
        n = self.cat.size(0)
        for name, width in (("pos", 3), ("rot6d", 6), ("log_scale", 3)):
            value = getattr(self, name)
            if value.ndim != 2 or value.shape != (n, width):
                raise ValueError(f"{name}: must have shape ({n}, {width})")
        if self.appearance.ndim != 2 or self.appearance.size(0) != n:
            raise ValueError(f"appearance: must have shape ({n}, A)")
        if self.cat.numel() and (self.cat.lt(1).any() or self.cat.ge(len(self.categories)).any()):
            raise ValueError(f"cat: contains PAD ({PAD_ID}) or an id outside 1..{len(self.categories) - 1}")
        for name in ("pos", "rot6d", "log_scale", "appearance"):
            value = getattr(self, name)
            if value.is_complex() or not torch.is_floating_point(value):
                raise ValueError(f"{name}: must use a real floating-point dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name}: contains non-finite values")
        if not torch.isfinite(self.log_scale.exp()).all():
            raise ValueError("log_scale: must correspond to finite positive scales")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata: must be a mapping")
        try:
            json.dumps(self.metadata, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata: must be finite JSON-serializable data") from exc

    @property
    def n(self) -> int:
        return self.cat.size(0)

    def rotation(self) -> torch.Tensor:
        return rot6d_to_matrix(self.rot6d)

    def scale(self) -> torch.Tensor:
        return self.log_scale.exp()

    def aabb(self):
        half = (self.rotation().abs() @ (self.scale() / 2).unsqueeze(-1)).squeeze(-1)
        return self.pos - half, self.pos + half

    def morton_sorted(self) -> "SceneGraph":
        idx = torch.arange(self.n, device=self.pos.device) if self.n == 0 else morton_order(self.pos)
        metadata = dict(self.metadata)
        order = [int(index) for index in idx.cpu()]
        for key in ("object_ids", "object_details", "custom_meshes"):
            values = metadata.get(key)
            if isinstance(values, list) and len(values) == self.n:
                metadata[key] = [values[i] for i in order]
        return SceneGraph(
            self.categories, self.cat[idx], self.pos[idx], self.rot6d[idx],
            self.log_scale[idx], self.appearance[idx], metadata=metadata,
        )

    def to_latent(self) -> torch.Tensor:
        return torch.cat([self.pos, self.rot6d, self.log_scale, self.appearance], dim=-1)

    @classmethod
    def from_latent(
        cls, z: torch.Tensor, cat: torch.Tensor, categories: list,
        metadata: dict[str, Any] | None = None,
    ) -> "SceneGraph":
        if not isinstance(z, torch.Tensor) or z.ndim != 2 or z.size(1) < 12:
            raise ValueError("z must have shape (N, 12 + A)")
        if not isinstance(cat, torch.Tensor) or cat.ndim != 1 or cat.size(0) != z.size(0):
            raise ValueError("cat must have shape (N,) matching z")
        return cls(
            categories, cat, z[:, :3], z[:, 3:9], z[:, 9:12], z[:, 12:],
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_objects(cls, objects: list, categories: list, d_appearance: int = 16) -> "SceneGraph":
        _validate_categories(categories)
        if not isinstance(objects, list):
            raise ValueError("objects must be a list")
        if not isinstance(d_appearance, int) or d_appearance < 0:
            raise ValueError("d_appearance must be a non-negative integer")
        rows = {"cat": [], "position": [], "rotation6d": [], "scale": [], "appearance": []}
        defaults = {"rotation6d": [1, 0, 0, 0, 1, 0], "scale": [1, 1, 1], "appearance": [0.0] * d_appearance}
        widths = {"position": 3, "rotation6d": 6, "scale": 3, "appearance": d_appearance}
        for index, obj in enumerate(objects):
            if not isinstance(obj, dict):
                raise ValueError(f"object {index}: must be a mapping")
            category = obj.get("category")
            if category not in categories:
                raise ValueError(f"object {index}: unknown category {category!r}")
            category_id = categories.index(category)
            if category_id == PAD_ID:
                raise ValueError(f"object {index}: category {category!r} is reserved for PAD")
            values_by_field = {}
            for field, width in widths.items():
                raw = obj.get(field, defaults.get(field))
                if not isinstance(raw, (list, tuple)) or len(raw) != width:
                    raise ValueError(f"object {index}: {field} must have exactly {width} values")
                if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw):
                    raise ValueError(f"object {index}: {field} must contain numeric values")
                values = [float(value) for value in raw]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"object {index}: {field} contains non-finite values")
                values_by_field[field] = values
            if any(value <= 0 for value in values_by_field["scale"]):
                raise ValueError(f"object {index}: scale must contain only positive values")
            rows["cat"].append(category_id)
            for field in widths:
                rows[field].append(values_by_field[field])
        n = len(objects)
        return cls(
            categories,
            torch.tensor(rows["cat"], dtype=torch.long),
            torch.tensor(rows["position"], dtype=torch.float32).reshape(n, 3),
            torch.tensor(rows["rotation6d"], dtype=torch.float32).reshape(n, 6),
            torch.tensor(rows["scale"], dtype=torch.float32).reshape(n, 3).log(),
            torch.tensor(rows["appearance"], dtype=torch.float32).reshape(n, d_appearance),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {"schema_version": SCHEMA_VERSION, "objects": [
            {"category": self.categories[int(c)], "position": p.detach().cpu().tolist(),
             "rotation6d": r.detach().cpu().tolist(), "scale": s.detach().cpu().exp().tolist(),
             "appearance": a.detach().cpu().tolist()}
            for c, p, r, s, a in zip(self.cat, self.pos, self.rot6d, self.log_scale, self.appearance)
        ]}
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], categories: list, d_appearance: int = 16) -> "SceneGraph":
        if not isinstance(data, dict):
            raise ValueError("scene JSON must contain an object")
        version = data.get("schema_version")
        if version is not None and (type(version) is not int or version != SCHEMA_VERSION):
            raise ValueError(f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION}")
        if "objects" not in data:
            raise ValueError("scene JSON is missing objects")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("scene JSON metadata must be a mapping")
        scene = cls.from_objects(data["objects"], categories, d_appearance)
        return cls(
            scene.categories, scene.cat, scene.pos, scene.rot6d,
            scene.log_scale, scene.appearance, metadata=dict(metadata),
        )

    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def from_json(cls, path: str | Path, categories: list, d_appearance: int = 16) -> "SceneGraph":
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data, categories, d_appearance)
