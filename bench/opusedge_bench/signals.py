"""Signal extractors that match the paper's definitions.

- proxy_delta_rms   — Eq. Proxy-Δ: per-token RMS drift of hidden state across layers,
                      dim-normalised. Works on any dense transformer.
- native_delta_from_ssm — reads the SSM discretization step Δ (dt) from a Mamba /
                      Falcon-H1 forward pass. O(1) cost — a by-product of the SSM
                      selective scan.
- router_ir_from_moe — normalised Shannon entropy of the router softmax over experts,
                      averaged across layers. Low entropy ↔ specialised routing.
"""

from __future__ import annotations
import math
import torch
from torch import Tensor


@torch.no_grad()
def proxy_delta_rms(hidden_states: list[Tensor]) -> Tensor:
    """Compute Proxy-Δ = mean over layers of ||h_{l,t} - h_{l,t-1}||_2 / sqrt(d_model).

    Args:
        hidden_states: list of L tensors, each [B, T, d_model].
    Returns:
        Tensor [B, T] with the per-token Proxy-Δ. Index 0 mirrors index 1.
    """
    if not hidden_states:
        raise ValueError("empty hidden_states")
    L = len(hidden_states)
    B, T, D = hidden_states[0].shape
    scale = 1.0 / (L * math.sqrt(D))

    delta = torch.zeros(B, T, device=hidden_states[0].device, dtype=torch.float32)
    for h in hidden_states:
        h = h.float()
        # per-layer drift ||h_t - h_{t-1}||_2 along d_model, then accumulate
        diff = h[:, 1:] - h[:, :-1]
        delta[:, 1:] += diff.norm(dim=-1)
    delta = delta * scale
    delta[:, 0] = delta[:, 1]  # mirror the boundary
    return delta


@torch.no_grad()
def native_delta_from_ssm(dt_per_layer: list[Tensor]) -> Tensor:
    """Aggregate the native SSM Δ (dt) across layers.

    Args:
        dt_per_layer: list of L tensors, each [B, T, d_state] or [B, T].
    Returns:
        Tensor [B, T] with mean |Δ| across layers and channels.
    """
    if not dt_per_layer:
        raise ValueError("empty dt_per_layer")
    acc = None
    L = len(dt_per_layer)
    for dt in dt_per_layer:
        dt = dt.float().abs()
        if dt.dim() == 3:
            dt = dt.mean(dim=-1)
        acc = dt if acc is None else acc + dt
    return acc / L


@torch.no_grad()
def router_ir_from_moe(router_logits: list[Tensor]) -> Tensor:
    """Router-Gated Importance = normalised entropy of the per-token router softmax.

    Args:
        router_logits: list of L_moe tensors, each [B, T, n_experts] OR [B*T, n_experts].
    Returns:
        Tensor [B, T] with H / log(E) ∈ [0, 1]. Low = confident / important.
    """
    if not router_logits:
        raise ValueError("empty router_logits")
    ent_sum = None
    total = 0
    E = router_logits[0].shape[-1]
    log_E = math.log(max(2, E))

    for logits in router_logits:
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)  # -> [1, N, E]; safe as pseudo-batch
        p = logits.float().softmax(dim=-1)
        H = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(dim=-1)
        ent_sum = H if ent_sum is None else ent_sum + H
        total += 1
    return (ent_sum / total) / log_E
