from types import SimpleNamespace
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgflow.constraints import (
    collision_penalty,
    obb_collision_penalty,
    scene_penalty,
    world_aabb_half_extents,
)
from sgflow.math3d import matrix_to_rot6d, morton_order, rot6d_to_matrix


def test_rot6d_degenerate_inputs_are_finite_proper_rotations():
    x = torch.tensor([[0.0] * 6, [1.0, 0.0, 0.0, 2.0, 0.0, 0.0]])
    rotation = rot6d_to_matrix(x)
    identity = torch.eye(3).expand_as(rotation)
    assert torch.isfinite(rotation).all()
    assert torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-6)
    assert torch.allclose(torch.linalg.det(rotation), torch.ones(2), atol=1e-6)


def test_rot6d_matrix_roundtrip_for_regular_rotation():
    rotation = torch.linalg.qr(torch.randn(3, 3)).Q
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    assert torch.allclose(rot6d_to_matrix(matrix_to_rot6d(rotation)), rotation, atol=1e-6)


def test_rot6d_is_differentiable_for_regular_inputs():
    x = torch.tensor([[1.0, 2.0, 3.0, -2.0, 1.0, 4.0]], requires_grad=True)
    rot6d_to_matrix(x).square().sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_morton_order_empty_and_per_axis_normalization():
    empty = torch.empty((0, 3))
    assert morton_order(empty).dtype == torch.long
    assert morton_order(empty).numel() == 0

    # y/z still contribute bits despite x having a much larger range.
    pos = torch.tensor([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    assert morton_order(pos).tolist() == [0, 1, 2]


def _cfg(**weights):
    return SimpleNamespace(room_size=(2.0, 2.0, 2.0), **weights)


def test_rotated_extent_contributes_to_boundary_penalty():
    z = torch.zeros((1, 1, 12))
    z[..., 0] = 0.65
    z[..., 3:9] = torch.tensor([0.7071068, 0.7071068, 0.0, -0.7071068, 0.7071068, 0.0])
    z[..., 9:12] = torch.log(torch.tensor([1.0, 0.2, 0.2]))
    penalty = scene_penalty(z, torch.tensor([[True]]), torch.tensor([[False]]),
                            _cfg(w_collision=0.0, w_boundary=1.0, w_support=0.0))
    assert penalty > 0


def test_masked_slots_never_supply_support():
    # Object 0 is suspended. Slot 1 would support it, but is padding.
    z = torch.zeros((1, 2, 12))
    z[..., 3] = 1.0
    z[..., 7] = 1.0
    z[..., 9:12] = torch.log(torch.tensor([1.0, 1.0, 1.0]))
    z[0, 0, 2] = 2.0
    z[0, 1, 2] = 1.0
    cfg = _cfg(w_collision=0.0, w_boundary=0.0, w_support=1.0)
    masked = scene_penalty(z, torch.tensor([[True, False]]), torch.tensor([[True, False]]), cfg)
    removed = scene_penalty(z[:, :1], torch.tensor([[True]]), torch.tensor([[True]]), cfg)
    assert torch.allclose(masked, removed)
    assert masked > 0


def test_obb_sat_avoids_rotated_aabb_false_positive():
    theta = torch.tensor(torch.pi / 4)
    c, s = theta.cos(), theta.sin()
    rotation = torch.tensor([
        [c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0],
    ]).repeat(2, 1, 1)[None]
    scale = torch.tensor([[[2.0, 0.2, 0.2], [2.0, 0.2, 0.2]]])
    pos = torch.tensor([[[0.0, 0.0, 0.1], [-0.2, 0.2, 0.1]]])
    mask = torch.tensor([[True, True]])
    half = world_aabb_half_extents(rotation, scale)

    assert collision_penalty(pos, half, mask) > 0
    assert torch.allclose(
        obb_collision_penalty(pos, rotation, scale, mask), torch.tensor(0.0), atol=1e-6,
    )
