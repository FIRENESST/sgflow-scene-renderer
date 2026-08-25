"""训练骨架：python -m sgflow.train --data data/scenes --epochs 100

数据格式：目录下每份 JSON
{"prompt": "一间温馨的卧室...",
 "objects": [{"category": "bed", "position": [x,y,z], "rotation6d": [...], "scale": [x,y,z], "appearance": [...]}]}
"""
import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from .config import PAD_ID, SGFlowConfig
from .flow_matching import RectifiedFlow
from .models import SceneDenoiser, masked_mean
from .scene_graph import SceneGraph
from .text_encoder import TextEncoder
from .tex_raster import render_textures
from .texhead import TexHead


class SceneJsonDataset(Dataset):
    def __init__(self, root: str):
        self.files = sorted(
            os.path.join(root, f) for f in os.listdir(root) if f.endswith(".json")
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with open(self.files[i], encoding="utf-8") as f:
            return json.load(f)


def collate(batch, cfg: SGFlowConfig, device):
    """变长场景 -> 定长张量；按 Morton 序排列（SSM 空间局部性），空槽 = PAD"""
    B, N = len(batch), cfg.max_objects
    z1 = torch.zeros(B, N, cfg.latent_dim, device=device)
    cat = torch.full((B, N), PAD_ID, dtype=torch.long, device=device)
    mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    prompts = []
    for b, item in enumerate(batch):
        sg = SceneGraph.from_objects(item["objects"], cfg.categories, cfg.d_appearance).morton_sorted()
        lat = sg.to_latent()
        lat = F.pad(lat, (0, cfg.latent_dim - lat.size(-1)))[:, : cfg.latent_dim]
        if sg.n > N:
            raise ValueError(
                f"scene contains {sg.n} objects, exceeding max_objects={N}; "
                "increase max_objects or filter the dataset"
            )
        n = sg.n
        z1[b, :n] = lat[:n].to(device)
        cat[b, :n] = sg.cat[:n].to(device)
        mask[b, :n] = True
        prompts.append(item.get("prompt", ""))
    return prompts, z1, cat, mask


def train(
    cfg, data_dir, epochs, lr, batch_size, device, *, output="sgflow_ckpt.pt",
    resume=None, seed=0,
):
    if epochs < 0:
        raise ValueError("epochs must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    random.seed(seed)
    torch.manual_seed(seed)
    device = str(device)
    is_cuda = device.startswith("cuda")
    if is_cuda:
        torch.cuda.manual_seed_all(seed)
    dataset = SceneJsonDataset(data_dir)
    if not dataset:
        raise ValueError(f"dataset is empty: {data_dir}")
    payload = read_checkpoint(resume, cfg=cfg, map_location="cpu") if resume else None
    backend = payload.get("encoder", {}).get("backend_kind") if payload else None
    enc = TextEncoder(
        cfg.d_model, cfg.text_model, cfg.text_dim, backend_kind=backend,
    ).to(device)
    model = SceneDenoiser(cfg).to(device)
    texhead = None
    if cfg.texture_mode == "generated" or (payload and payload.get("texhead") is not None):
        texhead = TexHead(cfg.d_model, cfg.n_categories, cfg.d_appearance).to(device)
    flow = RectifiedFlow(cfg)
    trainable = list(model.parameters()) + list(enc.proj.parameters())
    if texhead is not None:
        trainable += list(texhead.parameters())
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(is_cuda and cfg.use_amp))
    except (AttributeError, TypeError):  # older supported PyTorch releases
        scaler = torch.cuda.amp.GradScaler(enabled=(is_cuda and cfg.use_amp))
    start_epoch = 0
    if payload:
        start_epoch = restore_checkpoint(
            payload, model=model, encoder=enc, texhead=texhead, optimizer=opt, scaler=scaler,
        )
    if is_cuda and cfg.use_compile:
        model.compile()
    amp_ctx = lambda: torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=(is_cuda and cfg.use_amp)
    )
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        # Collate on CPU so pin_memory/non_blocking transfers work correctly on CUDA.
        collate_fn=lambda b: collate(b, cfg, "cpu"),
        num_workers=0,
        pin_memory=is_cuda,
        generator=torch.Generator().manual_seed(seed),
    )
    model.train()
    enc.train()
    if texhead is not None:
        texhead.train()
    for ep in range(start_epoch, epochs):
        totals = {}
        batches = 0
        for prompts, z1, cat, mask in dl:
            z1 = z1.to(device, non_blocking=is_cuda)
            cat = cat.to(device, non_blocking=is_cuda)
            mask = mask.to(device, non_blocking=is_cuda)
            tok, tmask = enc(prompts)
            with amp_ctx():
                loss, logs = flow.loss(model, z1, cat, tok, tmask, mask)
                if texhead is not None and mask.any():
                    pooled = masked_mean(tok, tmask)[:, None, :].expand(-1, mask.size(1), -1)
                    active_cat = cat[mask]
                    active_app = z1[..., 12:][mask]
                    active_text = pooled[mask]
                    tex_out = texhead(active_cat, active_app, active_text)
                    previews = render_textures(
                        tex_out, size=cfg.texture_train_size, seed=seed + ep,
                    )
                    tex_loss = texhead.loss_fn(tex_out, previews)
                    loss = loss + tex_loss
                    logs = {**logs, "texture": tex_loss.detach()}
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt)
            scaler.update()
            batches += 1
            for name, value in logs.items():
                totals[name] = totals.get(name, 0.0) + float(value)
        averages = {name: value / batches for name, value in totals.items()}
        summary = "  ".join(f"{name}={value:.4f}" for name, value in sorted(averages.items()))
        print(f"epoch {ep:03d}  {summary}")
        save_checkpoint(
            output, cfg=cfg, model=model, encoder=enc, texhead=texhead,
            optimizer=opt, scaler=scaler, epoch=ep + 1,
            metadata={"averaged_logs": averages, "seed": seed},
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None, help="留空则自动检测（GPU 优先）")
    p.add_argument("--output", default="sgflow_ckpt.pt")
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    device = a.device
    if device is None:
        from .device import resolve_device
        device = resolve_device()
    train(
        SGFlowConfig(), a.data, a.epochs, a.lr, a.batch_size, device,
        output=a.output, resume=a.resume, seed=a.seed,
    )


if __name__ == "__main__":
    main()
