"""Residual with the R axis **fixed per window** instead of profiled (ablation B2a).

``physics/residual.py`` profiles ``min over (tau, R, L)``.  B2a replaces the R part by a regressed
value: each window uses only the table slice at one R-grid index, and the minimum runs over
``(tau, L)``.  This file is new; ``residual.py`` is not edited.  The table stays fixed and receives
no gradient (``01_SPEC.md`` section 5): everything here runs on tensors that carry no graph.
"""
from __future__ import annotations

import torch

from fault_localization.physics.grids import NUM_TYPES
from fault_localization.physics.residual import _shift, channel_weights


def slice_table(table: torch.Tensor, r_idx: torch.Tensor, n_r: int) -> torch.Tensor:
    """``[26, H, 75]`` + ``r_idx [B]`` -> ``[B, 26, H', 75]`` with ``H' = n_types * n_L`` (order: tau, L)."""
    n_f, h, c = table.shape
    if h % (NUM_TYPES * n_r):
        raise ValueError(f"H={h} is not a multiple of {NUM_TYPES} * {n_r}")
    n_l = h // (NUM_TYPES * n_r)
    t = table.view(n_f, NUM_TYPES, n_r, n_l, c)
    sel = t[:, :, r_idx.to(table.device)]                      # [26, 11, B, n_L, 75]
    return sel.permute(2, 0, 1, 3, 4).reshape(r_idx.shape[0], n_f, NUM_TYPES * n_l, c)


def raw_residual_sliced(s: torch.Tensor, a: torch.Tensor, table_b: torch.Tensor, sigma_eps,
                        sigma0=0.0, chunk_f: int = 1):
    """``(r [B,26], hyp_slice [B,26])``; same arithmetic as ``raw_residual(method="direct")``.

    ``table_b`` is ``[B, 26, H', 75]`` (one slice per window).  ``hyp_slice`` indexes ``H'``.
    """
    b = s.shape[0]
    s_flat = s.reshape(b, -1)
    w = channel_weights(a, sigma_eps, sigma0).to(s_flat.dtype)
    table_b = table_b.to(device=s_flat.device, dtype=s_flat.dtype)
    n_f = table_b.shape[1]
    r = s_flat.new_empty((b, n_f))
    hyp = torch.empty((b, n_f), dtype=torch.long, device=s_flat.device)
    for start in range(0, n_f, max(1, chunk_f)):
        stop = min(start + max(1, chunk_f), n_f)
        diff = s_flat[:, None, None, :] - table_b[:, start:stop]            # [B, Fc, H', 75]
        chunk_r = (diff * diff * w[:, None, None, :]).sum(dim=-1)
        best = chunk_r.min(dim=-1)
        r[:, start:stop] = best.values
        hyp[:, start:stop] = best.indices
    return r, hyp


def rho_sliced(s, a, table: torch.Tensor, r_idx: torch.Tensor, n_r: int, sigma_eps, sigma0=0.0,
               chunk_f: int = 1):
    """``(rho [B,26], hyp_full [B,26])``; ``hyp_full`` indexes the full ``H = 11 * n_r * n_L`` axis."""
    table_b = slice_table(table, r_idx, n_r)
    r, hyp = raw_residual_sliced(s, a, table_b, sigma_eps, sigma0, chunk_f)
    n_l = table.shape[1] // (NUM_TYPES * n_r)
    return _shift(r), slice_to_full_index(hyp, r_idx, n_r, n_l)


def slice_to_full_index(hyp_slice: torch.Tensor, r_idx: torch.Tensor, n_r: int, n_l: int) -> torch.Tensor:
    """Map an index in the ``(tau, L)`` slice back to the flat ``(tau, R, L)`` index of the full table."""
    tau = hyp_slice // n_l
    l_idx = hyp_slice % n_l
    return tau * (n_r * n_l) + r_idx.to(hyp_slice.device)[:, None] * n_l + l_idx
