"""纹理参数头：类别嵌入 + 外观 z + 文本池化 -> 4 种可微程序纹理的参数

输出全部是少量标量/向量参数（O(1)），由 tex_raster 中的解析光栅器成图——
替代逐像素卷积解码器（O(H*W*C^2)），训练/推理都快几个数量级。
"""
import torch
import torch.nn as nn


class TexHead(nn.Module):
    TYPES = ("noise", "stripes", "checker", "dots")

    def __init__(self, d_model: int, n_categories: int, d_appearance: int):
        super().__init__()
        self.cat_embed = nn.Embedding(n_categories, d_model)
        self.in_proj = nn.Linear(2 * d_model + d_appearance, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.mix_logits = nn.Linear(d_model, len(self.TYPES))  # 软选择纹理类型（可微）
        self.base_color = nn.Linear(d_model, 3)
        self.accent_color = nn.Linear(d_model, 3)
        self.params = nn.Linear(d_model, 6)  # freq, orient, phase, contrast, rough, metal
        self.bump = nn.Linear(d_model, 1)

    def forward(self, cat, appearance, text_pooled):
        """
        cat: (B,) 外观: (B, A) 文本池化: (B, d_model)
        -> dict of tensors（全部可微，无 argmax/量化）
        """
        h = self.in_proj(
            torch.cat([self.cat_embed(cat), appearance, text_pooled], dim=-1)
        )
        h = h + self.ff(h)
        p = self.params(h)
        return {
            "mix_logits": self.mix_logits(h),                     # (B, 4)
            "base_color": self.base_color(h),                     # (B, 3) 经 sigmoid
            "accent_color": self.accent_color(h),
            "freq": p[:, 0],
            "orient": p[:, 1],
            "phase": p[:, 2],
            "contrast": p[:, 3],
            "roughness": p[:, 4],
            "metallic": p[:, 5],
            "bump": self.bump(h).squeeze(-1),                     # (B,)
        }

    @staticmethod
    def loss_fn(out, rendered=None):
        """无材质标签时的程序纹理训练正则。

        Every one of the 17 emitted controls participates in this objective.
        When ``rendered`` is supplied, lightweight image-space terms also send
        gradients through the differentiable procedural rasterizer.  These are
        priors, not a substitute for supervised texture or perceptual targets.
        """
        mix = torch.softmax(out["mix_logits"], dim=-1)
        colors = torch.cat(
            [torch.sigmoid(out["base_color"]), torch.sigmoid(out["accent_color"])],
            dim=-1,
        )
        scalar_names = (
            "freq", "orient", "phase", "contrast", "roughness", "metallic", "bump",
        )
        scalars = torch.stack([torch.sigmoid(out[name]) for name in scalar_names], dim=-1)
        controls = torch.cat([mix, colors, scalars], dim=-1)

        # Across-object diversity. unbiased=False remains finite for one object;
        # the control prior below keeps every head connected in that case too.
        variance = controls.var(dim=0, unbiased=False).mean()
        diversity = -0.01 * torch.log(variance.clamp_min(1e-6))

        target = controls.new_tensor(
            [0.25] * 4
            + [0.38] * 3 + [0.62] * 3
            + [0.45, 0.50, 0.50, 0.65, 0.50, 0.15, 0.35]
        )
        control_prior = 0.002 * (controls - target).square().mean()
        mix_balance = 0.01 * (mix.mean(dim=0) - 0.25).square().mean()
        palette_distance = (
            torch.sigmoid(out["base_color"]) - torch.sigmoid(out["accent_color"])
        ).square().mean().clamp_min(1e-8).sqrt()
        palette_prior = 0.01 * (palette_distance - 0.25).square()
        loss = diversity + control_prior + mix_balance + palette_prior

        if rendered is not None:
            albedo, roughness, normal = rendered
            albedo_contrast = albedo.flatten(2).std(dim=-1, unbiased=False).mean()
            normal_energy = (normal[:, :2] - 0.5).square().mean()
            roughness_mean = roughness.mean()
            loss = loss + 0.02 * (albedo_contrast - 0.12).square()
            loss = loss + 0.01 * (normal_energy - 0.01).square()
            loss = loss + 0.005 * (roughness_mean - 0.5).square()
        return loss
