"""Frozen SentenceTransformer with a compact deterministic offline fallback."""
import hashlib
import warnings

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    HASH_BUCKETS = 8192

    def __init__(
        self, d_model: int, model_name: str, text_dim: int,
        backend_kind: str | None = None,
    ):
        super().__init__()
        if backend_kind not in {None, "sentence_transformer", "hash"}:
            raise ValueError(f"unknown text encoder backend {backend_kind!r}")
        # Linear's default initializer consumes the global RNG. Isolate it as well as
        # choosing a stable seed so constructing an encoder is reproducible.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.proj = nn.Linear(text_dim, d_model)
        self._st = None
        if backend_kind != "hash":
            try:
                from sentence_transformers import SentenceTransformer
                self._st = SentenceTransformer(model_name)
                for p in self._st.parameters():
                    p.requires_grad_(False)
                self._st.eval()
            except Exception as exc:
                if backend_kind == "sentence_transformer":
                    raise RuntimeError(
                        "checkpoint requires the sentence_transformer text backend, "
                        "but it could not be initialized"
                    ) from exc
                warnings.warn(
                    f"SentenceTransformer unavailable ({exc}); using deterministic hash fallback",
                    RuntimeWarning,
                )
        if self._st is None:
            self.backend_kind = "hash"
            g = torch.Generator(device="cpu").manual_seed(0)
            self.register_buffer(
                "hash_table",
                torch.randn(self.HASH_BUCKETS, text_dim, generator=g) / text_dim ** 0.5,
                persistent=False,
            )
        else:
            self.backend_kind = "sentence_transformer"

    def train(self, mode: bool = True):
        super().train(mode)
        if self._st is not None:
            self._st.eval()
        return self

    def adapter_state_dict(self):
        """Return only the trainable adapter; never serialize the frozen backend."""
        return {k: v.detach().clone() for k, v in self.proj.state_dict().items()}

    def load_adapter_state_dict(self, state):
        if not isinstance(state, dict):
            raise ValueError("text encoder adapter state must be a mapping")
        try:
            self.proj.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"text encoder adapter is incompatible: {exc}") from exc

    @property
    def device(self):
        return self.proj.weight.device

    def forward(self, texts: list):
        """texts: list[str] -> (tokens (B,L,d_model), mask (B,L) bool)"""
        if self._st is not None:
            with torch.no_grad():
                feats = self._st.tokenize(texts)
                feats = {k: v.to(self.device) for k, v in feats.items()}
                out = self._st(feats)
                tok = out["token_embeddings"].float()
                mask = feats["attention_mask"].bool()
        else:
            tok, mask = self._hash_encode(texts)
        return self.proj(tok), mask

    def _hash_encode(self, texts):
        rows = []
        for text in texts:
            vecs = []
            for word in text.split():
                grams = [word[i:i + 3] for i in range(max(1, len(word) - 2))]
                idx = [
                    int.from_bytes(hashlib.md5(g.encode()).digest()[:4], "little") % self.HASH_BUCKETS
                    for g in grams
                ]
                vecs.append(self.hash_table[idx].mean(0))
            rows.append(torch.stack(vecs) if vecs else self.hash_table[:1])
        L = max(r.size(0) for r in rows)
        tok = self.hash_table.new_zeros(len(rows), L, self.hash_table.size(1))
        mask = torch.zeros(len(rows), L, dtype=torch.bool, device=self.hash_table.device)
        for i, r in enumerate(rows):
            tok[i, : r.size(0)] = r
            mask[i, : r.size(0)] = True
        return tok, mask
