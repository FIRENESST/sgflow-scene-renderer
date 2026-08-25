"""整流流匹配 (Rectified Flow)：直线路径插值，训练免仿真，采样 8-16 步即可（对比 DDPM 的 1000 步）"""
import torch
import torch.nn.functional as F

from .config import PAD_ID
from .models import masked_mean


class RectifiedFlow:
    """约定 z_t = (1-t)*z0 + t*z1，z0~N(0,I) 噪声，z1 数据，目标速度 v = z1 - z0"""

    def __init__(self, cfg):
        self.cfg = cfg

    def sample_t(self, batch: int, device) -> torch.Tensor:
        # logit-normal 时间采样（SD3）：聚焦信息最密集的中段
        return torch.sigmoid(torch.randn(batch, device=device))

    def loss(self, model, z1, cat, text_tokens, text_mask, obj_mask):
        """Single-step flow loss plus PAD-balanced structure supervision."""
        B = z1.size(0)
        z0 = torch.randn_like(z1)
        valid = obj_mask[..., None].to(z1.dtype)
        z0 = z0 * valid
        z1 = z1 * valid
        t = self.sample_t(B, z1.device)
        z_t = (1.0 - t[:, None, None]) * z0 + t[:, None, None] * z1
        v_pred = model(z_t, t, cat, text_tokens, text_mask, obj_mask)
        per_obj = ((v_pred - (z1 - z0)) ** 2).mean(-1)                       # (B, N)
        m = obj_mask.to(per_obj.dtype)
        flow_loss = (per_obj * m).sum() / m.sum().clamp_min(1.0)
        cat_logits = model.struct(masked_mean(text_tokens, text_mask))       # (B, N, C)
        active = obj_mask.bool() & cat.ne(PAD_ID)
        presence_logits = torch.stack(
            [cat_logits[..., PAD_ID], torch.logsumexp(cat_logits[..., 1:], dim=-1)], dim=-1
        )
        presence_per_slot = F.cross_entropy(
            presence_logits.reshape(-1, 2), active.long().reshape(-1), reduction="none"
        ).view_as(active)
        positives, negatives = active, ~active
        pos_loss = (presence_per_slot * positives).sum() / positives.sum().clamp_min(1)
        neg_loss = (presence_per_slot * negatives).sum() / negatives.sum().clamp_min(1)
        presence_loss = 0.5 * (pos_loss + neg_loss)
        category_loss = (
            F.cross_entropy(cat_logits[active], cat[active])
            if active.any() else cat_logits.sum() * 0.0
        )
        struct_loss = presence_loss + category_loss
        return flow_loss + struct_loss, {
            "flow": flow_loss.detach(), "struct": struct_loss.detach(),
            "presence": presence_loss.detach(), "category": category_loss.detach(),
        }

    @torch.no_grad()
    def sample(self, model, cat, text_tokens, text_mask, obj_mask, steps: int = None, generator=None):
        """欧拉法解 ODE：dz/dt = v_theta(z, t)，从 t=0(纯噪声) 积到 t=1(场景)"""
        steps = self.cfg.flow_steps if steps is None else steps
        if steps <= 0:
            raise ValueError(f"steps must be a positive integer, got {steps}")
        B, N = cat.shape
        z = torch.randn(B, N, self.cfg.latent_dim, device=text_tokens.device, generator=generator)
        valid = obj_mask[..., None].to(z.dtype)
        z = z * valid
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B,), i / steps, device=z.device)
            z = (z + dt * model(z, t, cat, text_tokens, text_mask, obj_mask)) * valid
        return z
