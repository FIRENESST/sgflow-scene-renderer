"""SGFlow 模型：潜瓶颈交叉注意力 + 选择性 SSM 的线性复杂度去噪网络

GPU 优化：
- SSM 使用 autograd-safe PyTorch 扫描（cuda_ops.ssm_scan）
- 交叉注意力用 F.scaled_dot_product_attention（FlashAttention 后端）
- SceneDenoiser 提供 .compile() 做 torch.compile 图融合
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SGFlowConfig
from .cuda_ops import ssm_scan


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """正弦时间嵌入 (B,) -> (B, dim)"""
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    half = dim // 2
    if half == 0:
        return t[..., None].new_zeros(*t.shape, dim)
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=t.dtype, device=t.device) / half
    )
    args = t[..., None] * freqs
    emb = torch.cat([args.cos(), args.sin()], dim=-1)
    return F.pad(emb, (0, 1)) if dim % 2 else emb


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask[..., None].to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp_min(1.0)


class SSMBlock(nn.Module):
    """选择性状态空间块（Mamba 简化版）

    序列混合复杂度 O(L * d_state)，对物体数线性 —— 替代 O(N^2) 自注意力。
    输入前已按 Morton 序排序，线性扫描即获得空间局部性。
    """

    def __init__(self, d: int, d_state: int = 16, expand: int = 2, d_conv: int = 4, chunk: int = 64):
        super().__init__()
        if chunk <= 0:
            raise ValueError(f"chunk must be a positive integer, got {chunk}")
        self.inner = d * expand
        self.d_state = d_state
        self.chunk = chunk
        self.dt_rank = max(d // 16, 1)
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, self.inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.inner, self.inner, d_conv, padding=d_conv - 1, groups=self.inner)
        self.x_proj = nn.Linear(self.inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.inner, 1))
        )
        self.D = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, d, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        valid = mask[..., None].to(x.dtype) if mask is not None else None
        h = self.norm(x)
        if valid is not None:
            h = h * valid
        u, gate = self.in_proj(h).chunk(2, dim=-1)                # (B, L, I)
        B, L, _ = u.shape
        u = self.conv1d(u.transpose(1, 2))[..., :L].transpose(1, 2)
        u = F.silu(u)
        if valid is not None:
            u = u * valid
        dt, Bm, Cm = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))                      # (B, L, I)
        A = -torch.exp(self.A_log)                                # (I, S)
        Bm = Bm.contiguous()
        Cm = Cm.contiguous()
        # Autograd-safe sequential recurrence on every device.
        y = ssm_scan(u, delta, A, Bm, Cm, chunk=self.chunk)
        y = y + u * self.D
        out = x + self.out_proj(y * F.silu(gate))
        return out * valid if valid is not None else out


class LatentBottleneck(nn.Module):
    """Perceiver 式潜瓶颈：文本条件经 K 个可学潜向量注入场景

    复杂度 O((N + L_text) * K)，K << N —— 替代对文本的全连接 O(N * L_text)。
    注意力用 F.scaled_dot_product_attention（GPU 上自动走 FlashAttention）。
    """

    def __init__(self, d: int, n_latents: int, n_heads: int = 8):
        super().__init__()
        if d % n_heads:
            raise ValueError(f"d ({d}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.latents = nn.Parameter(torch.randn(n_latents, d) * 0.02)
        self.norm_lat = nn.LayerNorm(d)
        self.norm_txt = nn.LayerNorm(d)
        self.norm_obj = nn.LayerNorm(d)
        self.q_in = nn.Linear(d, d)
        self.kv_in = nn.Linear(d, 2 * d)
        self.q_rd = nn.Linear(d, d)
        self.kv_rd = nn.Linear(d, 2 * d)

    def _attend(self, q, kv, key_padding_mask=None):
        """多头注意力，SDPA 后端；kv 为 (K, V) 元组；key_padding_mask: (B, L_kv) bool，True=有效"""
        B, Lq, D = q.shape
        k, v = kv
        # 拆头 -> (B, H, L, d_head)
        def split(t):
            return t.view(B, t.size(1), self.n_heads, self.d_head).transpose(1, 2)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :]  # (B,1,1,L_kv)，True 保留
        return F.scaled_dot_product_attention(
            split(q), split(k), split(v), attn_mask=attn_mask
        ).transpose(1, 2).reshape(B, Lq, D)

    def forward(self, x, cond, cond_mask=None):
        B = x.size(0)
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)
        kpm = cond_mask if cond_mask is not None else None
        # 注入：latents <- text
        kv = self.kv_in(self.norm_txt(cond)).chunk(2, dim=-1)
        h = self._attend(self.q_in(self.norm_lat(lat)), kv, kpm)
        lat = lat + h
        # 读出：objects <- latents
        kv2 = self.kv_rd(lat).chunk(2, dim=-1)
        y = self._attend(self.q_rd(self.norm_obj(x)), kv2)
        return x + y


class MixerBlock(nn.Module):
    """潜瓶颈（读文本）+ SSM（混物体）交替"""

    def __init__(self, cfg: SGFlowConfig):
        super().__init__()
        self.bottleneck = LatentBottleneck(cfg.d_model, cfg.n_latents)
        self.ssm = SSMBlock(cfg.d_model, cfg.d_state, cfg.expand, chunk=cfg.ssm_chunk)

    def forward(self, x, cond, cond_mask, obj_mask):
        x = self.bottleneck(x, cond, cond_mask)
        return self.ssm(x, obj_mask)


class StructureHead(nn.Module):
    """离散结构头：每个槽位预测物体类别（0=PAD 即"无物体"）

    结构（有什么/几个）与几何（在哪/多大）解耦 ——
    离散决策走分类，连续几何走流匹配，各自用最优数学工具。
    """

    def __init__(self, cfg: SGFlowConfig):
        super().__init__()
        d = cfg.d_model
        self.slots = nn.Parameter(torch.randn(cfg.max_objects, d) * 0.02)
        self.ff = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.n_categories)
        )

    def forward(self, pooled):  # (B, d) -> (B, N, C)
        return self.ff(self.slots + pooled[:, None, :])


class SceneDenoiser(nn.Module):
    """整流流匹配速度场网络：预测 v = dz/dt"""

    def __init__(self, cfg: SGFlowConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.z_in = nn.Linear(cfg.latent_dim, d)
        self.cat_embed = nn.Embedding(cfg.n_categories, d)
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([MixerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_out = nn.LayerNorm(d)
        self.v_out = nn.Linear(d, cfg.latent_dim)
        self.struct = StructureHead(cfg)

    def forward(self, z_t, t, cat, text_tokens, text_mask=None, obj_mask=None):
        """
        z_t: (B, N, latent_dim) 含噪潜变量
        t:   (B,)
        cat: (B, N) 类别 id（结构头给出，作为几何条件）
        """
        x = self.z_in(z_t) + self.cat_embed(cat)
        x = x + self.time_mlp(timestep_embedding(t, x.size(-1)))[:, None, :]
        for blk in self.blocks:
            x = blk(x, text_tokens, text_mask, obj_mask)
        v = self.v_out(self.norm_out(x))
        return v * obj_mask[..., None].to(v.dtype) if obj_mask is not None else v

    def compile(self, **kwargs):
        """torch.compile 图融合（GPU 上收益最大）；失败时静默回退 eager"""
        try:
            self.forward = torch.compile(self.forward, **kwargs)
        except Exception:
            pass
        return self
