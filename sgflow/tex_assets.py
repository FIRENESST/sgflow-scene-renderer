"""纹理资源导出器：SGFlow 场景 -> 每物体三张 PNG + 材质清单 JSON

纹理策略二选一（cfg.texture_mode）：
- "generated"：TexHead 模型生成纹理（默认）
- "library"  ：只查纹理库，不跑模型；命中的类别用库贴图，缺失的物体用白模（不贴纹理），
               并在控制台 + texture_report.json 中报告命中/缺失明细

库目录约定：<texture_lib>/<category>/albedo.png（必需）rough.png / normal.png（可选）

用法：
    python -m sgflow.tex_assets scene.json textures_out/ --mode library --lib textures_lib
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import hashlib

import torch

from .config import SGFlowConfig
from .models import SceneDenoiser, masked_mean
from .scene_graph import SceneGraph
from .tex_raster import render_textures
from .texhead import TexHead

logger = logging.getLogger("sgflow.tex")


class TexAssetExporter:
    """批量为场景中的物体生成/查找纹理，并输出 Blender 材质清单"""

    def __init__(self, cfg: SGFlowConfig, device: str = "cpu", texture_mode: str = None,
                 texture_lib: str = None, seed: int = 0, batch_size: int = None):
        self.cfg = cfg
        self.device = torch.device(device)
        self.mode = texture_mode or cfg.texture_mode
        self.lib = texture_lib or cfg.texture_lib
        if self.mode not in ("generated", "library"):
            raise ValueError(f"texture_mode 只能是 generated/library，收到: {self.mode!r}")
        batch_size = cfg.texture_batch_size if batch_size is None else batch_size
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        self.seed = int(seed)
        self.batch_size = batch_size
        # library 模式完全不加载 TexHead —— 模型不参与纹理生成
        if self.mode == "generated":
            # A fresh exporter is reproducible even in a new Python process.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed)
                self.texhead = TexHead(cfg.d_model, cfg.n_categories, cfg.d_appearance).to(self.device)
            self.texhead.eval()
        else:
            self.texhead = None

    def load_from_denoiser(self, model: SceneDenoiser):
        """若 TexHead 权重挂在主模型里，可在此对接；当前为独立模块"""
        return self

    def load_texhead_state_dict(self, state_dict, strict: bool = True):
        """Load externally managed TexHead weights without owning checkpoint I/O."""
        if self.texhead is None:
            raise RuntimeError("TexHead is unavailable in library mode")
        result = self.texhead.load_state_dict(state_dict, strict=strict)
        self.texhead.eval()
        return result

    def load_state_dict(self, state_dict, strict: bool = True):
        """Convenience alias for injecting TexHead weights from a caller-owned checkpoint."""
        return self.load_texhead_state_dict(state_dict, strict=strict)

    # ---------- library 模式 ----------

    def _library_textures(self, category: str):
        """查库：<lib>/<category>/albedo.png 存在才算命中，rough/normal 可选"""
        d = os.path.join(self.lib, category)
        albedo = os.path.join(d, "albedo.png")
        if not os.path.isfile(albedo):
            return None
        paths = {"albedo": albedo}
        for key, fname in (("roughness", "rough.png"), ("normal", "normal.png")):
            p = os.path.join(d, fname)
            if os.path.isfile(p):
                paths[key] = p
        return paths

    # ---------- 主导出 ----------

    @torch.no_grad()
    def export_scene(
        self,
        sg: SceneGraph,
        text_tokens: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        out_dir: str = "textures_out",
        size: int = 256,
        seed: int = None,
        batch_size: int = None,
    ) -> dict:
        """返回材质清单 dict；generated 模式把 PNG 写入 out_dir，library 模式只引用库路径"""
        self._validate_size(size)
        batch_size = self.batch_size if batch_size is None else batch_size
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        os.makedirs(out_dir, exist_ok=True)
        if self.mode == "generated" and sg.n:
            self._require_pillow()
        report = {"mode": self.mode, "total": sg.n, "library_hit": [], "library_miss": [], "generated": []}

        if self.mode == "generated":
            generated = self._generated_entries(sg, text_tokens, text_mask, out_dir, size, seed, batch_size)
        else:
            generated = {}
            logger.info("[纹理] 模式=library：模型不参与纹理生成，直接查库 %s", os.path.abspath(self.lib))

        manifest = {"manifest_version": 1, "materials": []}
        for i in range(sg.n):
            category = sg.categories[int(sg.cat[i])]
            name = f"obj{i:03d}_{category}"
            entry = {"object_index": i, "category": category}

            if self.mode == "library":
                paths = self._library_textures(category)
                if paths is not None:
                    entry["source"] = "library"
                    entry["textures"] = self._relative_paths(paths, out_dir)
                    report["library_hit"].append(name)
                    logger.info("[纹理] 命中  %s <- %s", name, paths["albedo"])
                else:
                    # 缺失 -> 白模：不写贴图，importer 端 entry["textures"] 为 None 即走纯色
                    entry["source"] = "white"
                    entry["textures"] = None
                    report["library_miss"].append(name)
                    logger.warning("[纹理] 缺失  %s：库中无 %s/，使用白模（不贴纹理）", name, category)
            else:
                params, paths = generated[i]
                entry["source"] = "generated"
                entry["textures"] = self._relative_paths(paths, out_dir)
                entry["params"] = params
                report["generated"].append(name)

            manifest["materials"].append(entry)

        with open(os.path.join(out_dir, "materials.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "texture_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(
            "[纹理] 完成：共 %d | 库命中 %d | 缺失(白模) %d | 生成 %d",
            report["total"], len(report["library_hit"]), len(report["library_miss"]), len(report["generated"]),
        )
        return manifest

    def _generated_entries(self, sg, text_tokens, text_mask, out_dir, size, seed, batch_size):
        """Render and save a bounded number of objects at a time."""
        if sg.n == 0:
            return {}
        if text_tokens is None or text_mask is None:
            pooled = torch.zeros((1, self.cfg.d_model), device=self.device)
        else:
            pooled = masked_mean(text_tokens.to(self.device), text_mask.to(self.device))
        scene_seed = self._scene_seed(sg, self.seed if seed is None else int(seed))
        saved = {}
        for start in range(0, sg.n, batch_size):
            stop = min(start + batch_size, sg.n)
            cat = sg.cat[start:stop].to(self.device)
            app = sg.appearance[start:stop].to(self.device)
            out = self.texhead(cat, app, pooled.expand(stop - start, -1))
            albedo, rough, normal = render_textures(out, size=size, seed=scene_seed + start)
            for local, i in enumerate(range(start, stop)):
                category = sg.categories[int(sg.cat[i])]
                name = f"obj{i:03d}_{category}"
                paths = {
                    "albedo": os.path.join(out_dir, f"{name}_albedo.png"),
                    "roughness": os.path.join(out_dir, f"{name}_rough.png"),
                    "normal": os.path.join(out_dir, f"{name}_normal.png"),
                }
                self._save_png(albedo[local], paths["albedo"])
                self._save_png(rough[local], paths["roughness"])
                self._save_png(normal[local], paths["normal"])
                saved[i] = ({
                    "base_color": torch.sigmoid(out["base_color"][local]).tolist(),
                    "roughness": float(torch.sigmoid(out["roughness"][local])),
                    "metallic": float(torch.sigmoid(out["metallic"][local])),
                    "bump": float(2.0 * torch.sigmoid(out["bump"][local])),
                }, paths)
        return saved

    def _validate_size(self, size):
        limit = self.cfg.texture_size_limit
        if not isinstance(size, int) or isinstance(size, bool) or not 2 <= size <= limit:
            raise ValueError(f"texture size must be an integer in [2, {limit}], got {size!r}")

    @staticmethod
    def _require_pillow():
        try:
            import PIL  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Pillow is required to export PNG textures; install it with `pip install Pillow`.") from exc

    @staticmethod
    def _relative_paths(paths, manifest_dir):
        return {key: Path(os.path.relpath(path, manifest_dir)).as_posix() for key, path in paths.items()}

    @staticmethod
    def _scene_seed(sg, explicit_seed):
        """Stable identity seed; prefers a future SceneGraph.fingerprint API."""
        fingerprint = getattr(sg, "fingerprint", None)
        if callable(fingerprint):
            fingerprint = fingerprint()
        if fingerprint is None:
            digest = hashlib.sha256()
            digest.update("\0".join(sg.categories).encode("utf-8"))
            for value in (sg.cat, sg.pos, sg.rot6d, sg.log_scale, sg.appearance):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tuple(tensor.shape)).encode("ascii"))
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(tensor.numpy().tobytes())
            fingerprint = digest.hexdigest()
        digest = hashlib.sha256(str(fingerprint).encode("utf-8")).digest()
        return (int.from_bytes(digest[:8], "big") ^ int(explicit_seed)) & 0x7FFFFFFF

    @staticmethod
    def _save_png(tensor: torch.Tensor, path: str):
        """(1|3,H,W) float [0,1] -> PNG; one channel is saved as grayscale."""
        if tensor.ndim != 3 or tensor.size(0) not in (1, 3):
            raise ValueError(f"expected a (1|3,H,W) texture tensor, got {tuple(tensor.shape)}")
        arr = (tensor.clamp(0, 1) * 255).byte().cpu()
        arr = arr.squeeze(0).numpy() if tensor.size(0) == 1 else arr.permute(1, 2, 0).numpy()
        try:
            from PIL import Image
            Image.fromarray(arr).save(path)
        except ImportError as exc:
            raise RuntimeError("Pillow is required to export PNG textures; install it with `pip install Pillow`.") from exc


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="为场景 JSON 生成/查找纹理资源")
    p.add_argument("scene", help="场景 JSON 路径")
    p.add_argument("out", help="纹理输出目录")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0, help="deterministic generated-texture seed")
    p.add_argument("--batch-size", type=int, default=None, help="objects rendered per generated batch")
    p.add_argument("--checkpoint", default=None, help="SGFlow checkpoint containing encoder/TexHead weights")
    p.add_argument("--allow-untrained", action="store_true",
                   help="explicit smoke-test mode: use deterministic but untrained texture weights")
    p.add_argument("--prompt", default="", help="用于条件化的文本（留空则自动从类别拼）")
    p.add_argument("--mode", choices=["generated", "library"], default=None,
                   help="纹理策略：generated=模型生成 | library=接纹理库（缺失用白模）")
    p.add_argument("--lib", default=None, help="纹理库目录（library 模式）")
    a = p.parse_args()

    payload = None
    if a.checkpoint:
        from .checkpoint import read_checkpoint
        payload = read_checkpoint(a.checkpoint)
        cfg = payload["config_obj"]
    else:
        cfg = SGFlowConfig()

    mode = a.mode or cfg.texture_mode
    sg = SceneGraph.from_json(a.scene, cfg.categories, cfg.d_appearance)
    if mode == "generated":
        if payload is None and not a.allow_untrained:
            p.error("generated mode requires --checkpoint (or explicit --allow-untrained for a smoke test)")
        prompt = a.prompt or " ".join(sg.categories[int(c)] for c in sg.cat[:8])
        from .text_encoder import TextEncoder
        encoder_payload = payload.get("encoder") if payload else None
        backend = encoder_payload.get("backend_kind") if encoder_payload else None
        enc = TextEncoder(cfg.d_model, cfg.text_model, cfg.text_dim, backend_kind=backend)
        if encoder_payload:
            enc.load_adapter_state_dict(encoder_payload.get("adapter", {}))
        enc.eval()
        with torch.no_grad():
            tok, tmask = enc([prompt])
    else:
        tok = tmask = None  # library 模式不需要文本条件

    batch_size = a.batch_size or cfg.texture_batch_size
    exporter = TexAssetExporter(cfg, texture_mode=mode, texture_lib=a.lib,
                                seed=a.seed, batch_size=batch_size)
    if mode == "generated" and payload:
        texhead_state = payload.get("texhead")
        if texhead_state is None:
            p.error("checkpoint does not contain TexHead weights required by generated mode")
        exporter.load_texhead_state_dict(texhead_state)
    manifest = exporter.export_scene(sg, tok, tmask, a.out, size=a.size,
                                     seed=a.seed, batch_size=batch_size)
    print(f"完成：{len(manifest['materials'])} 个材质 -> {a.out}（报告见 texture_report.json）")


if __name__ == "__main__":
    main()
