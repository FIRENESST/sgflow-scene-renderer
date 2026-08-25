"""SO(3) 旋转与空间索引工具（纯 PyTorch，全部可微）"""
import torch


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """6D 连续旋转表示 -> 旋转矩阵 (Zhou et al. 2019)

    相比欧拉角/四元数：无万向锁、无双覆盖、处处连续，网络更容易学习。
    x: (..., 6) -> (..., 3, 3)，Gram-Schmidt 正交化。
    """
    a1, a2 = x[..., :3], x[..., 3:]
    # Gram--Schmidt alone is undefined for zero vectors and collinear input.
    # Select a deterministic orthonormal completion only in those degenerate
    # cases; away from the tiny guard band this is the usual differentiable
    # 6D conversion.
    eps = x.new_tensor(1e-6)
    a1_norm = a1.norm(dim=-1, keepdim=True)
    unit_x = torch.zeros_like(a1)
    unit_x[..., 0] = 1
    b1 = torch.where(a1_norm > eps, a1 / a1_norm.clamp_min(eps), unit_x)

    a2_orth = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    a2_orth_norm = a2_orth.norm(dim=-1, keepdim=True)
    # The least-aligned canonical axis guarantees a well-conditioned cross
    # product for every unit b1.
    axis_index = b1.abs().argmin(dim=-1, keepdim=True)
    axis = torch.zeros_like(b1).scatter(-1, axis_index, 1)
    fallback_b2 = torch.cross(axis, b1, dim=-1)
    fallback_b2 = fallback_b2 / fallback_b2.norm(dim=-1, keepdim=True).clamp_min(eps)
    b2 = torch.where(
        a2_orth_norm > eps,
        a2_orth / a2_orth_norm.clamp_min(eps),
        fallback_b2,
    )
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """旋转矩阵 (..., 3, 3) -> 6D 表示（取前两列）"""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def morton_order(pos: torch.Tensor) -> torch.Tensor:
    """三维位置的 Morton(Z-order) 排序置换。

    把空间上相邻的物体排到序列相邻位置，让后面的 SSM
    在线性扫描时获得空间局部性归纳偏置。
    pos: (N, 3) -> (N,) 排序索引
    """
    if pos.size(0) == 0:
        return torch.empty(0, dtype=torch.long, device=pos.device)
    p = pos - pos.min(0, keepdim=True).values
    # Normalize each coordinate independently.  A large range in x should
    # not collapse spatial resolution in y/z.
    p = p / p.max(0, keepdim=True).values.clamp_min(1e-8)
    q = (p * 1023).long().clamp(0, 1023)  # 每轴 10bit
    code = torch.zeros(q.size(0), dtype=torch.long, device=pos.device)
    for i in range(10):
        code |= ((q[:, 0] >> i) & 1) << (3 * i)
        code |= ((q[:, 1] >> i) & 1) << (3 * i + 1)
        code |= ((q[:, 2] >> i) & 1) << (3 * i + 2)
    return torch.argsort(code)
