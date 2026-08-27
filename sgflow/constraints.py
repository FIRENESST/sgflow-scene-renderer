"""可微几何约束：碰撞 / 边界 / 支撑。训练期作正则，推理期作梯度精修——无需额外学习模块"""
import torch
import torch.nn.functional as F

from .math3d import rot6d_to_matrix


def unpack_latent(z: torch.Tensor):
    """z: (..., 3+6+3+A) -> 位置, 旋转6D, log尺寸, 外观"""
    return z[..., :3], z[..., 3:9], z[..., 9:12], z[..., 12:]


def world_aabb_half_extents(rotation, scale):
    """Rotated box world-AABB half extents: ``abs(R) @ (scale / 2)``."""
    return torch.matmul(rotation.abs(), (scale * 0.5).unsqueeze(-1)).squeeze(-1)


def collision_penalty(pos, half, mask):
    """World-AABB collision approximation.  ``mask`` is boolean."""
    delta = (pos[:, :, None, :] - pos[:, None, :, :]).abs()
    overlap = F.relu(half[:, :, None, :] + half[:, None, :, :] - delta)
    # A pair intersects only when it overlaps on all three axes.  The minimum
    # overlap is a compact penetration-depth approximation.
    pen = overlap.amin(-1)
    eye = torch.eye(pos.size(1), dtype=torch.bool, device=pos.device)[None]
    pair = (mask[:, :, None] & mask[:, None, :] & ~eye).to(pen.dtype)
    return (pen * pair).sum() / pair.sum().clamp_min(1.0)


def obb_collision_penalty(pos, rotation, scale, mask, eps: float = 1e-7):
    """Exact pairwise OBB overlap penalty using the 15 separating axes.

    The old world-AABB approximation reports collisions for many diagonally
    placed objects.  SAT tests the three local axes of each box plus their nine
    cross products, while remaining differentiable for layout refinement.
    """
    if pos.size(1) < 2:
        return pos.sum() * 0.0
    half = scale * 0.5
    # Matrix columns are local axes in world coordinates.  Store axes as the
    # penultimate dimension to make the pairwise projections explicit.
    local = rotation.transpose(-1, -2)                         # (B,N,3,3)
    n = pos.size(1)
    axes_i = local[:, :, None].expand(-1, -1, n, -1, -1)
    axes_j = local[:, None, :].expand(-1, n, -1, -1, -1)
    cross = torch.cross(
        axes_i[..., :, None, :], axes_j[..., None, :, :], dim=-1,
    ).reshape(*axes_i.shape[:3], 9, 3)
    axes = torch.cat([axes_i, axes_j, cross], dim=-2)           # (B,N,N,15,3)
    axis_norm = axes.norm(dim=-1, keepdim=True)
    valid_axis = axis_norm.squeeze(-1) > eps
    axes = axes / axis_norm.clamp_min(eps)

    delta = pos[:, :, None, :] - pos[:, None, :, :]
    distance = torch.einsum("bijlc,bijc->bijl", axes, delta).abs()
    projection_i = torch.einsum("bijlc,bijac->bijla", axes, axes_i).abs()
    projection_j = torch.einsum("bijlc,bijac->bijla", axes, axes_j).abs()
    radius_i = (projection_i * half[:, :, None, None, :]).sum(-1)
    radius_j = (projection_j * half[:, None, :, None, :]).sum(-1)
    overlap = radius_i + radius_j - distance
    # Parallel-axis cross products carry no separating information.
    overlap = torch.where(valid_axis, overlap, torch.full_like(overlap, torch.inf))
    penetration = F.relu(overlap.amin(-1))

    upper = torch.triu(
        torch.ones(n, n, dtype=torch.bool, device=pos.device), diagonal=1,
    )[None]
    pair = mask[:, :, None] & mask[:, None, :] & upper
    weights = pair.to(penetration.dtype)
    return (penetration * weights).sum() / weights.sum().clamp_min(1.0)


def boundary_penalty(pos, half, mask, room):
    """物体 AABB 须在房间范围内：x,y 居中，z 从地面 0 到层高 h。mask 为 bool"""
    mn = pos.new_tensor([-room[0] / 2, -room[1] / 2, 0.0])
    mx = pos.new_tensor([room[0] / 2, room[1] / 2, room[2]])
    pen = F.relu(mn - (pos - half)) + F.relu((pos + half) - mx)
    m = mask.to(pen.dtype)
    return (pen.sum(-1) * m).sum() / m.sum().clamp_min(1.0)


def support_penalty(pos, half, mask, needs_support):
    """防悬空：物体底面须落在地面或最高可行支撑面（xy 接近且更低的物体顶面）上"""
    bottom = pos[..., 2] - half[..., 2]                        # (B, N)
    top = pos[..., 2] + half[..., 2]
    xy_overlap = (
        (pos[:, :, None, :2] - pos[:, None, :, :2]).abs()
        <= half[:, :, None, :2] + half[:, None, :, :2]
    ).all(-1)
    below = top[:, None, :] <= bottom[:, :, None] + 0.1
    eye = torch.eye(pos.size(1), dtype=torch.bool, device=pos.device)[None]
    # Mask candidate supports as well as objects receiving the penalty.  In
    # particular, a padded slot's arbitrary latent values cannot support one.
    valid = xy_overlap & below & ~eye & mask[:, None, :]
    cands = torch.where(valid, top[:, None, :].expand(valid.shape), top.new_zeros(()))
    support = cands.amax(-1)                                   # 默认地面 0
    pen = F.relu(bottom - support - 0.01)
    supported_objects = mask & needs_support.bool()
    weights = supported_objects.to(pen.dtype)
    return (pen * weights).sum() / weights.sum().clamp_min(1.0)


def scene_penalty(z, mask, needs_support, cfg):
    """组合约束，对潜变量 z 完全可微。mask / needs_support 为 bool"""
    pos, rot6d, log_s, _ = unpack_latent(z)
    scale = torch.exp(log_s.clamp(-4, 2))
    rotation = rot6d_to_matrix(rot6d)
    half = world_aabb_half_extents(rotation, scale)
    collision = (
        obb_collision_penalty(pos, rotation, scale, mask)
        if getattr(cfg, "collision_mode", "obb") == "obb"
        else collision_penalty(pos, half, mask)
    )
    return (
        cfg.w_collision * collision
        + cfg.w_boundary * boundary_penalty(pos, half, mask, cfg.room_size)
        + cfg.w_support * support_penalty(pos, half, mask, needs_support)
    )
