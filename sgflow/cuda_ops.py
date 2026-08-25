"""Autograd-safe selective SSM scan.

The previous CUDA extension wrote outputs through raw pointers, which bypassed
PyTorch autograd (and only implemented a float32 subset). Correctness is more
important than that experimental fast path, so every device uses this PyTorch
reference implementation.
"""
import torch


def ssm_scan_reference(u, delta, A, Bm, Cm, chunk: int = 64):
    """Run the selective SSM recurrence and return ``(y, final_state)``."""
    if chunk <= 0:
        raise ValueError(f"chunk must be a positive integer, got {chunk}")
    if u.ndim != 3 or delta.shape != u.shape:
        raise ValueError("u and delta must both have shape (batch, length, inner)")
    B, L, I = u.shape
    if A.ndim != 2 or A.shape[0] != I:
        raise ValueError("A must have shape (inner, state)")
    if Bm.shape != (B, L, A.size(1)) or Cm.shape != Bm.shape:
        raise ValueError("Bm and Cm must have shape (batch, length, state)")

    state = u.new_zeros(B, I, A.size(1))
    if L == 0:
        return u.new_empty(B, 0, I), state
    ys = []
    for start in range(0, L, chunk):
        for t in range(start, min(start + chunk, L)):
            dA = torch.exp(delta[:, t, :, None] * A)
            dBx = (delta[:, t] * u[:, t])[:, :, None] * Bm[:, t, None, :]
            state = dA * state + dBx
            ys.append((state * Cm[:, t, None, :]).sum(-1))
    return torch.stack(ys, dim=1), state


def ssm_scan(u, delta, A, Bm, Cm, chunk: int = 64):
    """Return selective SSM outputs on CPU, CUDA, or other PyTorch devices."""
    y, _ = ssm_scan_reference(u, delta, A, Bm, Cm, chunk=chunk)
    return y
