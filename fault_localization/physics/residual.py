"""The residual rho -- a fixed, zero-parameter function of the inputs.

``01_SPEC.md`` §4, as amended 2026-09-23.  For each candidate location f the
operator profiles out the nuisance parameters (tau, R, L) and keeps the best
weighted squared error over the available channels::

    r_f(A) = min over (tau, R, L) of  sum_{(i,p) in A} (s_ip - mu[f,tau,R,L,i,p])^2 / (sigma_eps^2 + sigma0^2)
    rho_f  = (r_f - min_g r_g) / 2
    P_phys = softmax(-rho)

Why the shift and the halving
-----------------------------
With L profiled and sigma_eps ~ 1e-3 p.u., a wrong location contributes
per-channel terms of order (0.1 / 1e-3)^2 = 1e4, so a raw ``r_f`` reaches 1e5-1e6
for wrong locations while the true one is ~|A|.  Dividing by |A| (the first
draft's ``r-tilde``) rescales that but does not remove it: lambda_1 would have
needed to be ~1e-5 and any usable beta would have saturated the softmax.
Subtracting the minimum puts the quantity in Gaussian log-likelihood units --
one unit is one nat of evidence -- where lambda and beta of order 1 are
meaningful, and makes it invariant to |A| and therefore comparable across
dropout levels.  ``rho >= 0`` with exactly one zero, by construction.

Numerical note (load-bearing, do not "optimise" away)
-----------------------------------------------------
The obvious speed-up is to expand the square into two matmuls
(``sum w s^2 - 2 (w s) mu^T + w (mu^2)^T``).  In float32 that is **wrong here**:
at the true hypothesis ``(s - mu)^2`` is ~1e-8 while ``s^2`` is ~1, so the
expansion cancels seven significant digits and the surviving error, divided by
``sigma_eps^2 = 1e-6``, swamps ``r_true``.  The default ``method="direct"``
forms the difference first and is accurate in float32.  ``method="gram"`` is the
matmul form and **requires float64**, where the cancellation costs ~1e-8 of
``r`` and is harmless; it is what the offline corpus-wide cache uses.
"""

from __future__ import annotations

import numpy as np
import torch

from fault_localization.physics.grids import (
    NUM_CHANNELS,
    NUM_LOCATIONS,
    NUM_PHASES,
    NUM_PMU_NODES,
    NUM_R,
    NUM_TYPES,
)

NUM_HYPOTHESIS_CLASSES = NUM_LOCATIONS + 1  # 26: no-fault + 25 locations


# ---------------------------------------------------------------------------
# Table preparation
# ---------------------------------------------------------------------------


def flatten_table(mu: np.ndarray, mu0: np.ndarray) -> torch.Tensor:
    """``(mu [25,11,15,nL,25,3], mu0 [nL,25,3])`` -> ``[26, H, 75]``.

    The ``(tau, R, L)`` axes are flattened in C order -- the order
    ``grids.unravel_hypothesis`` inverts.  Row 0 is the no-fault hypothesis,
    broadcast across the tau and R axes so that every class shares one
    hypothesis axis: the minimum over a constant-in-(tau,R) row is the minimum
    over L alone, which is exactly ``r_0`` of the spec, and the uniform shape
    keeps every downstream op a single batched tensor operation.
    """
    mu = np.asarray(mu, dtype=np.float32)
    mu0 = np.asarray(mu0, dtype=np.float32)
    n_l = mu.shape[3]
    if mu.shape != (NUM_LOCATIONS, NUM_TYPES, NUM_R, n_l, NUM_PMU_NODES, NUM_PHASES):
        raise ValueError(f"unexpected mu shape {mu.shape}")
    if mu0.shape != (n_l, NUM_PMU_NODES, NUM_PHASES):
        raise ValueError(f"unexpected mu0 shape {mu0.shape}")

    h = NUM_TYPES * NUM_R * n_l
    flat = np.empty((NUM_HYPOTHESIS_CLASSES, h, NUM_CHANNELS), dtype=np.float32)
    flat[1:] = mu.reshape(NUM_LOCATIONS, h, NUM_CHANNELS)
    flat[0] = np.broadcast_to(
        mu0.reshape(1, 1, n_l, NUM_CHANNELS), (NUM_TYPES, NUM_R, n_l, NUM_CHANNELS)
    ).reshape(h, NUM_CHANNELS)
    return torch.from_numpy(flat)


def channel_weights(a: torch.Tensor, sigma_eps, sigma0: float = 0.0) -> torch.Tensor:
    """``a / (sigma_eps^2 + sigma0^2)``, flattened to ``[B, 75]``.

    ``sigma_eps`` may be a scalar or a per-channel array of length 75.
    ``sigma0`` is the injected measurement-noise level and is 0 on the clean
    corpus; under the noise experiments it **must** be set to the level actually
    injected (``12_CHECKLISTS.md`` §C). It may be a scalar, or a ``[B]`` tensor
    when noise augmentation draws a different level per window.
    """
    a_flat = a.reshape(a.shape[0], -1).to(torch.float32)
    if a_flat.shape[1] != NUM_CHANNELS:
        raise ValueError(f"availability must flatten to {NUM_CHANNELS} channels")
    # PCSTGNN registers sigma_eps as a buffer, so it follows the model's own .to(device)
    # (e.g. onto CUDA), where np.asarray() on a GPU tensor raises -- move it instead.
    eps = (sigma_eps.to(device=a_flat.device) if isinstance(sigma_eps, torch.Tensor)
          else torch.as_tensor(np.asarray(sigma_eps, dtype=np.float32), device=a_flat.device))
    if eps.numel() not in (1, NUM_CHANNELS):
        raise ValueError("sigma_eps must be scalar or per-channel (75)")
    eps2 = eps.to(a_flat.dtype).reshape(-1) ** 2                      # [1] or [75]
    if isinstance(sigma0, torch.Tensor) and sigma0.numel() > 1:
        s0 = sigma0.to(device=a_flat.device, dtype=a_flat.dtype).reshape(-1, 1)
        if s0.shape[0] != a_flat.shape[0]:
            raise ValueError(f"per-window sigma0 must have {a_flat.shape[0]} entries, "
                             f"got {s0.shape[0]}")
        denom = eps2[None, :] + s0 ** 2                               # [B, 1|75]
    else:
        denom = eps2 + float(sigma0) ** 2
    return a_flat / denom


# ---------------------------------------------------------------------------
# r -> rho
# ---------------------------------------------------------------------------


def _shift(r: torch.Tensor) -> torch.Tensor:
    """``(r - min_f r) / 2`` along the class axis."""
    return (r - r.min(dim=-1, keepdim=True).values) * 0.5


def raw_residual(
    s: torch.Tensor,
    a: torch.Tensor,
    table: torch.Tensor,
    sigma_eps,
    sigma0: float = 0.0,
    chunk_f: int = 1,
    method: str = "direct",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unshifted ``r [B, 26]`` and the profiled hypothesis index ``[B, 26]``.

    ``r`` is what S1.6 calibrates ``sigma_eps`` against (divided by ``|A|``);
    everything else in the project uses :func:`rho`.
    """
    b = s.shape[0]
    s_flat = s.reshape(b, -1)
    if s_flat.shape[1] != NUM_CHANNELS:
        raise ValueError(f"observation must flatten to {NUM_CHANNELS} channels")
    w = channel_weights(a, sigma_eps, sigma0).to(s_flat.dtype)
    table = table.to(device=s_flat.device)

    if method == "gram":
        return _raw_residual_gram(s_flat, w, table)
    if method != "direct":
        raise ValueError("method must be 'direct' or 'gram'")

    n_f = table.shape[0]
    r = s_flat.new_empty((b, n_f))
    hyp = torch.empty((b, n_f), dtype=torch.long, device=s_flat.device)
    table = table.to(s_flat.dtype)
    for start in range(0, n_f, max(1, chunk_f)):
        stop = min(start + max(1, chunk_f), n_f)
        # [B, Fc, H, 75] -- the difference is formed *before* squaring, which is
        # what keeps float32 usable (see module docstring).
        diff = s_flat[:, None, None, :] - table[None, start:stop]
        chunk_r = (diff * diff * w[:, None, None, :]).sum(dim=-1)   # [B, Fc, H]
        best = chunk_r.min(dim=-1)
        r[:, start:stop] = best.values
        hyp[:, start:stop] = best.indices
    return r, hyp


def _raw_residual_gram(s_flat: torch.Tensor, w: torch.Tensor, table: torch.Tensor):
    """Matmul form of :func:`raw_residual`.  float64 only -- see module docstring."""
    if s_flat.dtype != torch.float64:
        raise TypeError("method='gram' requires float64 inputs; float32 loses the "
                        "true hypothesis to cancellation")
    n_f, h, _ = table.shape
    tab = table.to(torch.float64).reshape(n_f * h, NUM_CHANNELS)
    quad = (w * s_flat * s_flat).sum(dim=1, keepdim=True)            # [B, 1]
    cross = (w * s_flat) @ tab.T                                     # [B, F*H]
    self_term = w @ (tab * tab).T                                    # [B, F*H]
    r_full = (quad + self_term - 2.0 * cross).reshape(-1, n_f, h)
    best = r_full.min(dim=-1)
    return best.values, best.indices


def rho(
    s: torch.Tensor,
    a: torch.Tensor,
    table: torch.Tensor,
    sigma_eps,
    sigma0: float = 0.0,
    chunk_f: int = 1,
    method: str = "direct",
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rho [B, 26], hyp_star [B, 26])`` -- the quantity everything else uses.

    ``hyp_star`` is the flat ``(tau, R, L)`` index profiled at each class;
    decode it with :func:`grids.unravel_hypothesis`.  For class 0 only the ``L``
    component is meaningful.
    """
    r, hyp = raw_residual(s, a, table, sigma_eps, sigma0, chunk_f, method)
    return _shift(r), hyp


def p_phys(rho_values: torch.Tensor) -> torch.Tensor:
    """``softmax(-rho)`` -- the physics prior over the 26 classes."""
    return torch.softmax(-rho_values, dim=-1)


def psm_predict(rho_values: torch.Tensor) -> torch.Tensor:
    """Physics Signature Matching: ``argmin_f rho_f``."""
    return rho_values.argmin(dim=-1)


def margin(rho_values: torch.Tensor) -> torch.Tensor:
    """``rho_(2) - rho_(1)`` = the second-smallest rho, in nats (``11_METRICS.md`` §2)."""
    return rho_values.topk(2, dim=-1, largest=False).values[..., 1]


# ---------------------------------------------------------------------------
# Held-out residual rho(D) -- the lambda_2 term
# ---------------------------------------------------------------------------


def rho_heldout(
    s_clean: torch.Tensor,
    d_mask: torch.Tensor,
    table: torch.Tensor,
    hyp_star: torch.Tensor,
    sigma_eps,
    sigma0: float = 0.0,
) -> torch.Tensor:
    """Score the hypothesis profiled on A at the **held-out** sensors D.

    ``d_mask`` is ``[B, 25, 3]``, 1 on the channels that were synthetically
    dropped (whose true values are known at training time only), and
    ``hyp_star`` comes from :func:`rho` on the *masked* input -- the profiling
    never sees D, which is what makes this a leave-k-out consistency term rather
    than a reconstruction loss.
    """
    b = s_clean.shape[0]
    s_flat = s_clean.reshape(b, -1)
    w = channel_weights(d_mask, sigma_eps, sigma0).to(s_flat.dtype)
    table = table.to(device=s_flat.device, dtype=s_flat.dtype)
    n_f = table.shape[0]
    r = s_flat.new_empty((b, n_f))
    for f in range(n_f):
        mu_star = table[f][hyp_star[:, f]]            # [B, 75]
        diff = s_flat - mu_star
        r[:, f] = (diff * diff * w).sum(dim=-1)
    return _shift(r)


# ---------------------------------------------------------------------------
# Per-sensor trust scores (PCRG-STGNN, 01_SPEC.md §7)
# ---------------------------------------------------------------------------


def _sensor_error_tensor(
    s: torch.Tensor,
    a: torch.Tensor,
    table: torch.Tensor,
    f_hat: torch.Tensor,
    sigma_eps,
    sigma0: float = 0.0,
) -> torch.Tensor:
    """Per-sensor weighted error at the predicted class: ``[B, H, 25]``.

    Summing the three phase channels of each sensor first is what makes the
    leave-one-*sensor*-out refit a single subtraction later.
    """
    b = s.shape[0]
    s_flat = s.reshape(b, -1)
    w = channel_weights(a, sigma_eps, sigma0).to(s_flat.dtype)
    table = table.to(device=s_flat.device, dtype=s_flat.dtype)
    mu_f = table[f_hat]                                # [B, H, 75]
    diff = s_flat[:, None, :] - mu_f
    e = diff * diff * w[:, None, :]                    # [B, H, 75]
    return e.reshape(b, e.shape[1], NUM_PMU_NODES, NUM_PHASES).sum(dim=-1)


def rho_loo(
    s: torch.Tensor,
    a: torch.Tensor,
    table: torch.Tensor,
    f_hat: torch.Tensor,
    sigma_eps,
    sigma0: float = 0.0,
) -> torch.Tensor:
    """Leave-one-sensor-out trust score ``[B, 25]``: **subtract, then re-minimise**.

    For each sensor i: drop i's channels from the objective, re-profile
    ``(tau, R, L)`` at the predicted class on ``A \\ {i}``, and read i's own
    normalised error *at that refit*.

    The ordering is the whole point (``01_SPEC.md`` §7).  Reading i's term at the
    argmin fitted on all of A -- :func:`rho_insample` -- lets a corrupted sensor
    drag the fit toward itself and so conceal its own error, which would make
    the S5.2 comparison a variant against itself.  ``tests/test_physics_residual.py``
    plants a corrupted sensor and asserts ``rho_loo >> rho_insample`` for it.

    Returned in the same units as ``rho``: half the chi-squared contribution,
    i.e. a negative log-likelihood in nats.
    """
    sens = _sensor_error_tensor(s, a, table, f_hat, sigma_eps, sigma0)   # [B, H, 25]
    total = sens.sum(dim=-1, keepdim=True)                              # [B, H, 1]
    without = total - sens                                              # [B, H, 25]
    h_star = without.argmin(dim=1)                                      # [B, 25]
    own = sens.gather(1, h_star.unsqueeze(1)).squeeze(1)                # [B, 25]
    avail = (a.reshape(a.shape[0], NUM_PMU_NODES, NUM_PHASES).sum(-1) > 0).to(own.dtype)
    return 0.5 * own * avail


def rho_insample(
    s: torch.Tensor,
    a: torch.Tensor,
    table: torch.Tensor,
    f_hat: torch.Tensor,
    sigma_eps,
    sigma0: float = 0.0,
) -> torch.Tensor:
    """The **wrong** score, kept for the S5.2 contrast: i's term at the global argmin."""
    sens = _sensor_error_tensor(s, a, table, f_hat, sigma_eps, sigma0)
    h_star = sens.sum(dim=-1).argmin(dim=1)                             # [B]
    own = sens.gather(1, h_star[:, None, None].expand(-1, 1, NUM_PMU_NODES)).squeeze(1)
    avail = (a.reshape(a.shape[0], NUM_PMU_NODES, NUM_PHASES).sum(-1) > 0).to(own.dtype)
    return 0.5 * own * avail


# ---------------------------------------------------------------------------
# Reference implementation (tests and Gate 3)
# ---------------------------------------------------------------------------


def reference_loop(s, a, table, sigma_eps, sigma0: float = 0.0):
    """Deliberately slow NumPy ``rho``, written straight from the spec.

    Gate 3 requires the batched layer to match this to float tolerance; it is
    the only executable statement of what the equations mean.
    """
    s = np.asarray(s, dtype=np.float64).reshape(len(s), -1)
    a = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    tab = np.asarray(table, dtype=np.float64)
    eps = np.asarray(sigma_eps, dtype=np.float64)
    denom = eps ** 2 + float(sigma0) ** 2
    n_b, n_f, n_h = s.shape[0], tab.shape[0], tab.shape[1]
    r = np.empty((n_b, n_f))
    hyp = np.empty((n_b, n_f), dtype=np.int64)
    for b in range(n_b):
        for f in range(n_f):
            best, best_h = np.inf, -1
            for h in range(n_h):
                acc = 0.0
                for c in range(s.shape[1]):
                    if a[b, c] > 0:
                        d = s[b, c] - tab[f, h, c]
                        acc += d * d / (denom if np.ndim(denom) == 0 else denom[c])
                if acc < best:
                    best, best_h = acc, h
            r[b, f], hyp[b, f] = best, best_h
    return (r - r.min(axis=1, keepdims=True)) / 2.0, hyp
