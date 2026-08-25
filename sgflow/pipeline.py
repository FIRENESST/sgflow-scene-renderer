"""端到端管线：自然语言 -> 场景图 JSON"""

import torch

from .checkpoint import read_checkpoint, restore_checkpoint
from .config import PAD_ID, SGFlowConfig
from .constraints import scene_penalty
from .flow_matching import RectifiedFlow
from .models import SceneDenoiser, masked_mean
from .scene_graph import SceneGraph
from .text_encoder import TextEncoder
from .texhead import TexHead

NON_SUPPORT_CATS = {"pad", "floor", "wall", "rug", "window", "door"}


class ScenePipeline:
    def __init__(
        self, cfg: SGFlowConfig | None = None, device: str = None,
        checkpoint: str = None, *, allow_untrained: bool = False,
    ):
        if checkpoint is None and not allow_untrained:
            raise ValueError("a trained checkpoint is required; pass allow_untrained=True for smoke tests")
        payload = read_checkpoint(checkpoint, cfg=cfg, map_location="cpu") if checkpoint else None
        self.cfg = payload["config_obj"] if cfg is None and payload is not None else (cfg or SGFlowConfig())
        if device is None:
            from .device import resolve_device
            device = resolve_device(self.cfg)
        self.device = torch.device(device)
        backend = None
        if payload and payload.get("encoder"):
            backend = payload["encoder"].get("backend_kind")
        self.encoder = TextEncoder(
            self.cfg.d_model, self.cfg.text_model, self.cfg.text_dim, backend_kind=backend,
        ).to(self.device)
        self.model = SceneDenoiser(self.cfg).to(self.device)
        self.texhead = None
        if payload and payload.get("texhead") is not None:
            self.texhead = TexHead(
                self.cfg.d_model, self.cfg.n_categories, self.cfg.d_appearance,
            ).to(self.device)
        if payload:
            restore_checkpoint(payload, model=self.model, encoder=self.encoder, texhead=self.texhead)
        if self.device.type == "cuda" and self.cfg.use_compile:
            self.model.compile()
        self.model.eval()
        self.encoder.eval()
        if self.texhead is not None:
            self.texhead.eval()
        self.flow = RectifiedFlow(self.cfg)

    def generate(self, prompt: str, steps: int = None, refine_steps: int = 8, seed: int = None) -> SceneGraph:
        """一句话生成场景：结构采样 -> 几何流生成 -> 约束精修"""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if isinstance(refine_steps, bool) or not isinstance(refine_steps, int) or refine_steps < 0:
            raise ValueError("refine_steps must be a non-negative integer")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or None")
        generator = torch.Generator(device=self.device)
        actual_seed = int(seed) if seed is not None else generator.seed()
        if seed is not None:
            generator.manual_seed(actual_seed)
        with torch.no_grad():
            tok, tmask = self.encoder([prompt])
            tok, tmask = tok.to(self.device), tmask.to(self.device)
            # 1) 结构：逐槽位类别多项式采样（保留多样性）
            probs = self.model.struct(masked_mean(tok, tmask)).softmax(-1)   # (1, N, C)
            cat = torch.multinomial(
                probs.view(-1, probs.size(-1)), 1, generator=generator,
            ).view(1, -1)
            mask = cat != PAD_ID
            if not mask.any():
                non_pad = probs[0, :, 1:]
                flat = int(non_pad.reshape(-1).argmax())
                slot, category = divmod(flat, non_pad.size(-1))
                cat[0, slot] = category + 1
                mask = cat != PAD_ID
            limit = self.cfg.max_generated_objects
            if int(mask.sum()) > limit:
                # Keep the slots with the strongest learned presence evidence.
                presence = 1.0 - probs[0, :, PAD_ID]
                keep_slots = presence.topk(limit).indices
                keep_mask = torch.zeros_like(mask)
                keep_mask[0, keep_slots] = True
                cat = torch.where(keep_mask, cat, torch.full_like(cat, PAD_ID))
                mask = cat != PAD_ID
            # 2) 几何：整流流 ODE 采样
            z = self.flow.sample(
                self.model, cat, tok, tmask, mask, steps, generator=generator,
            )
        # 3) 约束精修（需要梯度，放在 no_grad 外）
        z = self._refine(z, cat, mask, refine_steps)
        # 约束层以这个区间解释 log-scale；导出时保持相同语义，避免 exp 溢出。
        z[..., 9:12] = z[..., 9:12].clamp(-4.0, 2.0)
        keep = mask[0]
        sg = SceneGraph.from_latent(z[0, keep].cpu(), cat[0, keep].cpu(), self.cfg.categories)
        return sg.morton_sorted()

    def _refine(self, z, cat, mask, steps: int):
        """对潜变量做几步梯度下降，消除穿模 / 悬空 / 越界"""
        if steps <= 0:
            return z
        ns_ids = torch.tensor(
            [i for i, n in enumerate(self.cfg.categories) if n in NON_SUPPORT_CATS],
            device=z.device,
        )
        needs_support = mask & ~torch.isin(cat, ns_ids)
        z = z.detach().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=3e-2)
        for _ in range(steps):
            loss = scene_penalty(z, mask, needs_support, self.cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return z.detach()
