"""Sparse relation-graph planning and differentiable OBB layout refinement."""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch
import torch.nn.functional as F

from .config import PAD_ID, SGFlowConfig
from .constraints import (
    boundary_penalty,
    obb_collision_penalty,
    world_aabb_half_extents,
)
from .math3d import matrix_to_rot6d
from .scene_graph import SceneGraph


RELATION_TYPES = (
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
    "above",
    "below",
    "near",
    "far",
    "on",
    "aligned_x",
    "aligned_y",
    "facing",
    "parallel_to",
    "perpendicular_to",
    "against_left_wall",
    "against_right_wall",
    "against_front_wall",
    "against_back_wall",
    "center_of_room",
)

STRUCTURAL_CATEGORIES = {"floor", "wall", "window", "door"}


def _finite_vector(value: Any, width: int, label: str, *, positive: bool = False) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != width:
        raise ValueError(f"{label} must contain exactly {width} numbers")
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in value):
        raise ValueError(f"{label} must contain only numbers")
    result = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{label} contains non-finite values")
    if positive and any(v <= 0.0 for v in result):
        raise ValueError(f"{label} must contain only positive values")
    return result


@dataclass(frozen=True)
class PlannedObject:
    object_id: str
    category: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw_degrees: float


@dataclass(frozen=True)
class SpatialRelation:
    subject_id: str
    relation: str
    object_id: str


@dataclass(frozen=True)
class SpatialPlan:
    objects: tuple[PlannedObject, ...]
    relations: tuple[SpatialRelation, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any], cfg: SGFlowConfig) -> "SpatialPlan":
        if not isinstance(data, dict):
            raise ValueError("model response must be a JSON object")
        raw_objects = data.get("objects")
        raw_relations = data.get("relations", [])
        if not isinstance(raw_objects, list) or not raw_objects:
            raise ValueError("model response must contain a non-empty objects list")
        if len(raw_objects) > cfg.max_generated_objects:
            raise ValueError(
                f"model returned {len(raw_objects)} objects; limit is {cfg.max_generated_objects}"
            )
        if not isinstance(raw_relations, list):
            raise ValueError("relations must be a list")

        allowed_categories = set(cfg.categories[PAD_ID + 1:])
        objects: list[PlannedObject] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_objects):
            if not isinstance(raw, dict):
                raise ValueError(f"objects[{index}] must be an object")
            object_id = raw.get("id")
            if not isinstance(object_id, str) or not object_id.strip():
                raise ValueError(f"objects[{index}].id must be a non-empty string")
            object_id = object_id.strip()
            if object_id in seen:
                raise ValueError(f"duplicate object id {object_id!r}")
            seen.add(object_id)
            category = raw.get("category")
            if category not in allowed_categories:
                raise ValueError(f"objects[{index}] has unsupported category {category!r}")
            yaw = raw.get("yaw_degrees", 0.0)
            if isinstance(yaw, bool) or not isinstance(yaw, Real) or not math.isfinite(float(yaw)):
                raise ValueError(f"objects[{index}].yaw_degrees must be a finite number")
            objects.append(PlannedObject(
                object_id=object_id,
                category=category,
                position=_finite_vector(raw.get("position"), 3, f"objects[{index}].position"),
                size=_finite_vector(raw.get("size"), 3, f"objects[{index}].size", positive=True),
                yaw_degrees=float(yaw),
            ))

        relations: list[SpatialRelation] = []
        for index, raw in enumerate(raw_relations):
            if not isinstance(raw, dict):
                raise ValueError(f"relations[{index}] must be an object")
            subject_id = raw.get("subject_id")
            relation = raw.get("relation")
            object_id = raw.get("object_id")
            if subject_id not in seen:
                raise ValueError(f"relations[{index}] has unknown subject_id {subject_id!r}")
            if relation not in RELATION_TYPES:
                raise ValueError(f"relations[{index}] has unsupported relation {relation!r}")
            room_relation = relation.startswith("against_") or relation == "center_of_room"
            if room_relation:
                if object_id != "room":
                    raise ValueError(f"relations[{index}] relation {relation!r} requires object_id='room'")
            elif object_id not in seen:
                raise ValueError(f"relations[{index}] has unknown object_id {object_id!r}")
            if subject_id == object_id:
                raise ValueError(f"relations[{index}] cannot relate an object to itself")
            relations.append(SpatialRelation(subject_id, relation, object_id))
        return cls(tuple(objects), tuple(relations))

    def to_dict(self) -> dict[str, Any]:
        return {
            "objects": [
                {
                    "id": item.object_id,
                    "category": item.category,
                    "position": list(item.position),
                    "size": list(item.size),
                    "yaw_degrees": item.yaw_degrees,
                }
                for item in self.objects
            ],
            "relations": [
                {
                    "subject_id": edge.subject_id,
                    "relation": edge.relation,
                    "object_id": edge.object_id,
                }
                for edge in self.relations
            ],
        }


def yaw_to_matrix(yaw: torch.Tensor) -> torch.Tensor:
    """Convert Z-axis yaw in radians to rotation matrices."""
    c, s = yaw.cos(), yaw.sin()
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([
        c, -s, zero,
        s, c, zero,
        zero, zero, one,
    ], dim=-1).reshape(*yaw.shape, 3, 3)


def _relation_penalty(
    pos: torch.Tensor,
    yaw: torch.Tensor,
    size: torch.Tensor,
    relations: tuple[SpatialRelation, ...],
    id_to_index: dict[str, int],
    room_size: tuple[float, float, float],
) -> torch.Tensor:
    if not relations:
        return pos.sum() * 0.0
    half = world_aabb_half_extents(yaw_to_matrix(yaw), size)
    losses: list[torch.Tensor] = []
    margin = pos.new_tensor(0.10)
    room_x, room_y, _ = (pos.new_tensor(float(v)) for v in room_size)

    for edge in relations:
        a = id_to_index[edge.subject_id]
        relation = edge.relation
        if relation.startswith("against_") or relation == "center_of_room":
            if relation == "against_left_wall":
                loss = (pos[a, 0] - half[a, 0] + room_x / 2).abs()
            elif relation == "against_right_wall":
                loss = (pos[a, 0] + half[a, 0] - room_x / 2).abs()
            elif relation == "against_front_wall":
                loss = (pos[a, 1] + half[a, 1] - room_y / 2).abs()
            elif relation == "against_back_wall":
                loss = (pos[a, 1] - half[a, 1] + room_y / 2).abs()
            else:
                # eps 开方：物体恰在房间中心时 sqrt(0) 的梯度为 Inf。
                loss = (pos[a, :2].square().sum() + 1e-12).sqrt()
            losses.append(loss)
            continue

        b = id_to_index[edge.object_id]
        delta = pos[b] - pos[a]
        # eps 开方替代 .norm()：两物体 XY 完全重叠时 norm 的反向是 0/0 NaN。
        distance_xy = (delta[:2].square().sum() + 1e-12).sqrt()
        if relation == "left_of":
            loss = F.relu(pos[a, 0] + half[a, 0] + margin - (pos[b, 0] - half[b, 0]))
        elif relation == "right_of":
            loss = F.relu(pos[b, 0] + half[b, 0] + margin - (pos[a, 0] - half[a, 0]))
        elif relation == "in_front_of":
            loss = F.relu(pos[b, 1] + half[b, 1] + margin - (pos[a, 1] - half[a, 1]))
        elif relation == "behind":
            loss = F.relu(pos[a, 1] + half[a, 1] + margin - (pos[b, 1] - half[b, 1]))
        elif relation == "above":
            loss = F.relu(pos[b, 2] + half[b, 2] + margin - (pos[a, 2] - half[a, 2]))
        elif relation == "below":
            loss = F.relu(pos[a, 2] + half[a, 2] + margin - (pos[b, 2] - half[b, 2]))
        elif relation == "near":
            surface_gap = distance_xy - half[a, :2].norm() - half[b, :2].norm()
            loss = F.relu(surface_gap - 1.0)
        elif relation == "far":
            surface_gap = distance_xy - half[a, :2].norm() - half[b, :2].norm()
            loss = F.relu(2.0 - surface_gap)
        elif relation == "on":
            vertical = (pos[a, 2] - half[a, 2] - pos[b, 2] - half[b, 2]).abs()
            contain_x = F.relu((pos[a, 0] - pos[b, 0]).abs() + half[a, 0] - half[b, 0])
            contain_y = F.relu((pos[a, 1] - pos[b, 1]).abs() + half[a, 1] - half[b, 1])
            loss = vertical + contain_x + contain_y
        elif relation == "aligned_x":
            loss = (pos[a, 0] - pos[b, 0]).abs()
        elif relation == "aligned_y":
            loss = (pos[a, 1] - pos[b, 1]).abs()
        elif relation == "facing":
            # 手写归一化：F.normalize 内部同样是 norm，零向量反向会产生 NaN。
            target = delta[:2] / distance_xy
            heading = torch.stack([-yaw[a].sin(), yaw[a].cos()])
            loss = 1.0 - (heading * target).sum()
        elif relation == "parallel_to":
            loss = 1.0 - (yaw[a] - yaw[b]).cos().abs()
        elif relation == "perpendicular_to":
            loss = (yaw[a] - yaw[b]).cos().abs()
        else:  # pragma: no cover - SpatialPlan validation makes this unreachable.
            continue
        losses.append(loss)
    return torch.stack(losses).mean()


def refine_spatial_plan(
    plan: SpatialPlan,
    cfg: SGFlowConfig,
    *,
    steps: int = 96,
    device: str = "cpu",
    source: str = "openai-compatible",
    model: str | None = None,
) -> SceneGraph:
    """Turn an LLM plan into a bounded, relation-aware :class:`SceneGraph`."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if not plan.objects:
        raise ValueError("plan must contain at least one object")
    dev = torch.device(device)
    pos = torch.tensor([item.position for item in plan.objects], dtype=torch.float32, device=dev)
    size = torch.tensor([item.size for item in plan.objects], dtype=torch.float32, device=dev)
    room = torch.tensor(cfg.room_size, dtype=torch.float32, device=dev)
    # Keep every box physically capable of fitting inside the configured room.
    size = torch.minimum(size, room[None] * 0.95).clamp_min(0.05)
    yaw = torch.deg2rad(torch.tensor(
        [item.yaw_degrees for item in plan.objects], dtype=torch.float32, device=dev,
    ))
    initial_half = world_aabb_half_extents(yaw_to_matrix(yaw), size)
    fit_ratio = (room[None] * 0.475 / initial_half.clamp_min(1e-6)).amin(-1).clamp(max=1.0)
    size = size * fit_ratio[:, None]
    id_to_index = {item.object_id: index for index, item in enumerate(plan.objects)}
    on_subjects = {edge.subject_id for edge in plan.relations if edge.relation == "on"}
    ground_mask = torch.tensor([
        item.category not in STRUCTURAL_CATEGORIES and item.object_id not in on_subjects
        for item in plan.objects
    ], dtype=torch.bool, device=dev)
    collision_mask = torch.tensor([
        item.category not in STRUCTURAL_CATEGORIES for item in plan.objects
    ], dtype=torch.bool, device=dev)[None]

    # A good deterministic starting manifold substantially reduces optimizer
    # work and prevents normal floor-standing furniture from remaining aloft.
    pos = pos.clone()
    pos[ground_mask, 2] = size[ground_mask, 2] * 0.5
    initial_pos, initial_yaw = pos.clone(), yaw.clone()
    if steps:
        pos = pos.detach().requires_grad_(True)
        yaw = yaw.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([pos, yaw], lr=4e-2)
        all_mask = torch.ones(1, len(plan.objects), dtype=torch.bool, device=dev)
        for _ in range(steps):
            rotation = yaw_to_matrix(yaw)
            half = world_aabb_half_extents(rotation, size)
            collision = obb_collision_penalty(
                pos[None], rotation[None], size[None], collision_mask,
            )
            boundary = boundary_penalty(pos[None], half[None], all_mask, cfg.room_size)
            ground = (
                (pos[:, 2] - half[:, 2]).abs()[ground_mask].mean()
                if ground_mask.any() else pos.sum() * 0.0
            )
            relations = _relation_penalty(
                pos, yaw, size, plan.relations, id_to_index, cfg.room_size,
            )
            regularizer = 0.01 * (pos - initial_pos).square().mean()
            regularizer = regularizer + 0.002 * (1.0 - (yaw - initial_yaw).cos()).mean()
            loss = (
                4.0 * cfg.w_collision * collision
                + 6.0 * cfg.w_boundary * boundary
                + 2.0 * cfg.w_support * ground
                + 4.0 * relations
                + regularizer
            )
            if not torch.isfinite(loss):
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pos, yaw = pos.detach(), yaw.detach()
        if not (torch.isfinite(pos).all() and torch.isfinite(yaw).all()):
            # 任何意外的数值发散都不应逃逸到 SceneGraph 校验：回退到
            # 已经过严格校验的初始布局，保证输出始终有限可用。
            warnings.warn(
                "layout refinement diverged; falling back to the initial layout",
                stacklevel=2,
            )
            pos, yaw = initial_pos.clone(), initial_yaw.clone()

    # Hard final projection makes bounds an invariant even for a zero-step run.
    rotation = yaw_to_matrix(yaw)
    half = world_aabb_half_extents(rotation, size)
    pos[:, 0] = pos[:, 0].clamp(-room[0] / 2 + half[:, 0], room[0] / 2 - half[:, 0])
    pos[:, 1] = pos[:, 1].clamp(-room[1] / 2 + half[:, 1], room[1] / 2 - half[:, 1])
    pos[:, 2] = pos[:, 2].clamp(half[:, 2], room[2] - half[:, 2])
    pos[ground_mask, 2] = half[ground_mask, 2]

    categories = torch.tensor(
        [cfg.categories.index(item.category) for item in plan.objects],
        dtype=torch.long,
    )
    appearance = torch.zeros(len(plan.objects), cfg.d_appearance, dtype=torch.float32)
    metadata = {
        "generator": source,
        "spatial_model": "sparse-relation-graph+obb-sat",
        "room_size": [float(v) for v in cfg.room_size],
        "layout_refine_steps": steps,
        "object_ids": [item.object_id for item in plan.objects],
        "relations": plan.to_dict()["relations"],
    }
    if model:
        metadata["model"] = model
    return SceneGraph(
        list(cfg.categories),
        categories,
        pos.cpu(),
        matrix_to_rot6d(rotation).cpu(),
        size.log().cpu(),
        appearance,
        metadata=metadata,
    )
