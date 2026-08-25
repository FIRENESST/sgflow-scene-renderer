"""可微程序纹理光栅器：参数 -> Albedo / 粗糙度 / 法线 三张 PNG

全部用解析核 + 向量化运算（无 Python 循环），O(1) 参数、O(H*W) 像素。
- Gabor 类核做 stripes/dots/checker 的解析表达（带导数，可直接转法线）
- 值噪声用确定性整数 hash（torch 无原生 Perlin，但位运算 hash 同样可微友好）
"""
import hashlib
import math

import torch
import torch.nn.functional as F

from .texhead import TexHead


def _grid(size, device, dtype):
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, size, device=device, dtype=dtype),
        torch.linspace(0, 1, size, device=device, dtype=dtype),
        indexing="ij",
    )
    return xs, ys


def _value_noise(x, y, freq, seed: int):
    """整数格点 hash 值噪声 + 双线性平滑插值（可微）"""
    xf, yf = x * freq, y * freq
    xi, yi = torch.floor(xf), torch.floor(yf)
    fx, fy = xf - xi, yf - yi
    # 平滑插值权重（smoothstep）
    sx = fx * fx * (3 - 2 * fx)
    sy = fy * fy * (3 - 2 * fy)

    def h(ix, iy):
        k = (ix.long() * 374761393 + iy.long() * 668265263 + seed * 2246822519) & 0x7FFFFFFF
        k = (k ^ (k >> 13)) * 1274126177 & 0x7FFFFFFF
        return (k & 0xFFFF).to(x.dtype) / 65535.0

    v00, v10 = h(xi, yi), h(xi + 1, yi)
    v01, v11 = h(xi, yi + 1), h(xi + 1, yi + 1)
    return (v00 * (1 - sx) + v10 * sx) * (1 - sy) + (v01 * (1 - sx) + v11 * sx) * sy


def _gabor(x, y, freq, orient, phase):
    """解析 Gabor 核：cos(2π f (x cosθ + y sinθ) + φ)"""
    u = x * torch.cos(orient) + y * torch.sin(orient)
    return torch.cos(2 * math.pi * freq * u + phase)


def _checker(x, y, freq):
    """解析棋盘格：sign 化的 cos 乘积（软边，可微）"""
    s = torch.sin(math.pi * freq * x) * torch.sin(math.pi * freq * y)
    return torch.tanh(8 * s) * 0.5 + 0.5


def _dots(x, y, freq, orient):
    """解析点阵：两组正交 Gabor 相乘，取正部"""
    g1 = _gabor(x, y, freq, orient, torch.zeros_like(orient))
    g2 = _gabor(x, y, freq, orient + math.pi / 2, torch.zeros_like(orient))
    return F.relu(g1 * g2)


def render_textures(out: dict, size: int = 256, seed: int = 0):
    """out: TexHead.forward 的输出 -> albedo/normal (B,3,H,W), rough (B,1,H,W)。

    全批量矢量化：网格/参数一次性广播，无逐物体 Python 循环，GPU 上单 kernel 完成。
    """
    if not isinstance(size, int) or isinstance(size, bool) or not 2 <= size <= 4096:
        raise ValueError(f"texture size must be an integer in [2, 4096], got {size!r}")
    B = out["freq"].size(0)
    if B == 0:
        device, dtype = out["freq"].device, out["freq"].dtype
        empty_rgb = torch.empty((0, 3, size, size), device=device, dtype=dtype)
        return empty_rgb, torch.empty((0, 1, size, size), device=device, dtype=dtype), empty_rgb.clone()
    device = out["freq"].device
    dtype = out["freq"].dtype
    xs, ys = _grid(size, device, dtype)          # (H, W)

    # 参数 -> (B, 1, 1)，广播到 (B, H, W)
    freq = (2.0 + 18.0 * torch.sigmoid(out["freq"])).view(B, 1, 1)
    orient = (math.pi * torch.sigmoid(out["orient"])).view(B, 1, 1)
    phase = (2 * math.pi * torch.sigmoid(out["phase"])).view(B, 1, 1)
    contrast = torch.sigmoid(out["contrast"]).view(B, 1, 1)

    x = xs.unsqueeze(0)                           # (1, H, W) -> 广播 (B,H,W)
    y = ys.unsqueeze(0)

    # 4 种程序纹理各算一张 (B,H,W)，再按 mix 权重软混合（可微）
    noise = _value_noise_batch(x, y, freq, seed, B)
    stripes = _gabor(x, y, freq, orient, phase) * 0.5 + 0.5
    checker = _checker(x, y, freq)
    dots = _dots(x, y, freq, orient)
    w = F.softmax(out["mix_logits"], dim=-1).view(B, 4, 1, 1)   # (B,4,1,1)
    field = (w[:, 0] * noise + w[:, 1] * stripes + w[:, 2] * checker + w[:, 3] * dots)
    field = (field - field.mean(dim=(1, 2), keepdim=True)) * (0.5 + contrast) + 0.5
    field = field.clamp(0, 1)

    base = torch.sigmoid(out["base_color"]).view(B, 3, 1, 1)
    accent = torch.sigmoid(out["accent_color"]).view(B, 3, 1, 1)
    albedo = base + (accent - base) * field.unsqueeze(1)  # (B,3,H,W)

    rough = 0.2 + 0.6 * torch.sigmoid(out["roughness"]).view(B, 1, 1, 1)
    rough_map = (rough + 0.1 * (field.unsqueeze(1) - 0.5)).clamp(0, 1)

    # 法线：对高度场求梯度（Sobel），bump 控制强度
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    h = field.unsqueeze(1)
    gx = F.conv2d(F.pad(h, (1, 1, 1, 1), mode="reflect"), sobel_x)
    gy = F.conv2d(F.pad(h, (1, 1, 1, 1), mode="reflect"), sobel_y)
    bump = (2.0 * torch.sigmoid(out["bump"])).view(B, 1, 1, 1)
    nz = torch.ones_like(gx)
    n = torch.cat([-gx * bump, -gy * bump, nz], 1)
    normal = (n / n.norm(dim=1, keepdim=True).clamp_min(1e-6)) * 0.5 + 0.5

    return albedo.clamp(0, 1), rough_map.clamp(0, 1), normal.clamp(0, 1)


def _value_noise_batch(x, y, freq, seed: int, B: int):
    """批量值噪声：seed 按 batch 维广播，无 Python 循环"""
    seeds = (seed + torch.arange(B, device=x.device)).view(B, 1, 1).to(x.dtype)
    return _value_noise_v(x, y, freq, seeds)


def _value_noise_v(x, y, freq, seed_b):
    """可向量化值噪声：seed 为 (B,1,1) 张量，逐 batch 不同"""
    xf, yf = x * freq, y * freq
    xi, yi = torch.floor(xf), torch.floor(yf)
    fx, fy = xf - xi, yf - yi
    sx = fx * fx * (3 - 2 * fx)
    sy = fy * fy * (3 - 2 * fy)

    def h(ix, iy):
        k = (ix.long() * 374761393 + iy.long() * 668265263 + seed_b.long() * 2246822519) & 0x7FFFFFFF
        k = (k ^ (k >> 13)) * 1274126177 & 0x7FFFFFFF
        return (k & 0xFFFF).to(x.dtype) / 65535.0

    v00, v10 = h(xi, yi), h(xi + 1, yi)
    v01, v11 = h(xi, yi + 1), h(xi + 1, yi + 1)
    return (v00 * (1 - sx) + v10 * sx) * (1 - sy) + (v01 * (1 - sx) + v11 * sx) * sy
