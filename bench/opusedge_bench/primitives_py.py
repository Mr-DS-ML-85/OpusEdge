"""Pure-PyTorch reference implementations of the OpusEdge primitives.

Mirrors the C++ engine one-for-one — same math, same defaults — so the bench
harness can run without building the C++ extension.
"""

from __future__ import annotations
import torch
from torch import Tensor


@torch.no_grad()
def selkv_evict(delta: Tensor, ratio: float) -> tuple[Tensor, Tensor]:
    """Return (keep_mask, evict_mask) that drops the bottom `ratio` fraction by Δ.

    Args:
        delta: [T] importance scores.
        ratio: fraction to evict, e.g. 0.875.
    Returns:
        keep_mask, evict_mask: bool [T] each.
    """
    T = delta.numel()
    n_keep = max(1, int(round(T * (1.0 - ratio))))
    # top-n_keep by δ
    _, topk = torch.topk(delta, k=n_keep, largest=True)
    keep = torch.zeros(T, dtype=torch.bool, device=delta.device)
    keep[topk] = True
    return keep, ~keep


@torch.no_grad()
def smsa_window_mask(seq_len: int, window: int, device=None) -> Tensor:
    """Causal sliding-window mask [T, T] (bool)."""
    device = device or "cpu"
    idx = torch.arange(seq_len, device=device)
    diff = idx[:, None] - idx[None, :]                 # [T, T]
    return (diff >= 0) & (diff < window)


@torch.no_grad()
def delta_ar_topk_indices(delta: Tensor, top_k: int) -> Tensor:
    """For each query t, pick the top-K keys j ≤ t by δ.

    Returns:
        indices: [T, K] long tensor; -1 padded when < K causal keys exist.
    """
    T = delta.numel()
    K = min(top_k, T)
    out = torch.full((T, K), -1, dtype=torch.long, device=delta.device)
    # naive O(T·K log T) via top-k on a running prefix — plenty fast for T ≤ 2048
    for t in range(T):
        window = delta[: t + 1]
        k = min(K, window.numel())
        _, top = torch.topk(window, k=k, largest=True)
        out[t, :k] = top
    return out


@torch.no_grad()
def apply_mask_to_kv(hidden: Tensor, keep_mask: Tensor) -> Tensor:
    """Return hidden restricted to kept positions along the sequence dim.

    Args:
        hidden: [B, T, D] or [T, D].
        keep_mask: bool [T].
    """
    if hidden.dim() == 2:
        return hidden[keep_mask]
    return hidden[:, keep_mask]


@torch.no_grad()
def score_ppl_from_logits(logits: Tensor, targets: Tensor) -> float:
    """Cross-entropy → PPL. `logits`: [B, T, V]. `targets`: [B, T]."""
    B, T, V = logits.shape
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1),
        reduction="mean",
    )
    return float(torch.exp(loss).item())
