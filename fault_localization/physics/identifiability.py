"""How far apart are two fault locations, in the units the residual works in?

``04_STAGE_C_gonogo.md`` S2.1.  For a pair of locations the separability is the
*hardest* case -- the closest approach of their two signature manifolds::

    D(f1, f2; A) = min over (tau1,R1,L1), (tau2,R2,L2)
                     sum_{(i,p) in A} (mu_f1 - mu_f2)^2 / (2 (sigma_eps^2 + sigma0^2))

Same units as ``rho``: D is the ``rho`` that a perfect, noise-free observation of
f1 would assign to f2.  ``D < 1`` means the two locations are within one nat --
indistinguishable at the operator's own floor -- and counting those pairs per k
is what says whether 25 magnitude-only uPMUs can support the task at all,
independently of any model.

The pairwise minimum is a cross-distance between two ``[H, 75]`` blocks, so it
is computed with the matmul expansion in **float64** (``residual.py`` explains
why float32 is not an option here).
"""

from __future__ import annotations

import numpy as np

from fault_localization.physics.grids import NUM_CHANNELS, NUM_PHASES, NUM_PMU_NODES


def _weighted_min_distance(m1: np.ndarray, m2: np.ndarray, w: np.ndarray) -> float:
    """min over (h1, h2) of ``sum_c w_c (m1[h1,c] - m2[h2,c])^2``."""
    q1 = (m1 * m1) @ w                      # [H1]
    q2 = (m2 * m2) @ w                      # [H2]
    cross = (m1 * w) @ m2.T                 # [H1, H2]
    d = q1[:, None] + q2[None, :] - 2.0 * cross
    return float(max(d.min(), 0.0))


def separability(
    table: np.ndarray,
    sigma_eps,
    sigma0: float = 0.0,
    availability: np.ndarray | None = None,
    max_hyp: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """``D [26, 26]`` for one availability pattern.

    ``availability`` is ``[25, 3]`` (1 = channel present); ``None`` means every
    instrumented channel.  ``max_hyp`` subsamples the hypothesis axis, which
    turns the exact minimum into an upper bound -- use it for exploration only,
    never for a reported number.
    """
    tab = np.asarray(table, dtype=np.float64)
    n_f, n_h, n_c = tab.shape
    if n_c != NUM_CHANNELS:
        raise ValueError(f"table must carry {NUM_CHANNELS} channels")
    if max_hyp is not None and max_hyp < n_h:
        rng = rng or np.random.default_rng(0)
        pick = rng.choice(n_h, size=max_hyp, replace=False)
        tab = tab[:, pick, :]

    a = np.ones((NUM_PMU_NODES, NUM_PHASES)) if availability is None else np.asarray(availability)
    denom = np.asarray(sigma_eps, dtype=np.float64) ** 2 + float(sigma0) ** 2
    w = a.reshape(-1).astype(np.float64) / (2.0 * denom)   # the 2 puts D in rho units

    d = np.zeros((n_f, n_f))
    for i in range(n_f):
        for j in range(i + 1, n_f):
            value = _weighted_min_distance(tab[i], tab[j], w)
            d[i, j] = d[j, i] = value
    return d


def separability_over_subsets(
    table: np.ndarray,
    sigma_eps,
    sigma0: float = 0.0,
    base_mask: np.ndarray | None = None,
    k: int = 0,
    n_subsets: int = 8,
    seed: int = 0,
    max_hyp: int | None = None,
) -> np.ndarray:
    """Mean ``D [26, 26]`` over ``n_subsets`` random k-sensor dropouts.

    The subsets are drawn the way ``dropout/masking.py`` draws them -- k node
    indices uniformly without replacement -- so the identifiability numbers and
    the measured dropout curves describe the same perturbation.
    """
    base = np.ones((NUM_PMU_NODES, NUM_PHASES)) if base_mask is None else np.asarray(base_mask, dtype=float)
    if k <= 0:
        return separability(table, sigma_eps, sigma0, base, max_hyp=max_hyp)
    rng = np.random.default_rng(seed)
    acc = None
    for _ in range(n_subsets):
        avail = base.copy()
        avail[rng.choice(NUM_PMU_NODES, size=min(k, NUM_PMU_NODES), replace=False)] = 0.0
        d = separability(table, sigma_eps, sigma0, avail, max_hyp=max_hyp)
        acc = d if acc is None else acc + d
    return acc / n_subsets


def indistinguishable_fraction(d: np.ndarray, threshold: float = 1.0) -> float:
    """Fraction of off-diagonal pairs closer than ``threshold`` nats."""
    n = d.shape[0]
    iu = np.triu_indices(n, k=1)
    return float((d[iu] < threshold).mean())


def worst_case_per_location(d: np.ndarray) -> np.ndarray:
    """``min_{f2 != f1} D(f1, f2)`` -- which locations are structurally hard."""
    masked = np.where(np.eye(d.shape[0], dtype=bool), np.inf, d)
    return masked.min(axis=1)


def effective_r_levels(mu: np.ndarray, sigma_eps: float, mask: np.ndarray | None = None,
                       l_index: int | None = None) -> dict:
    """How many of the 15 R grid points are actually *distinguishable*?

    For each ``(f, tau)`` this counts the distinct signatures along the R axis:
    greedily, a grid point opens a new level when its signature differs from
    every level already opened by more than ``sigma_eps`` in max-abs p.u. over
    the instrumented channels.

    Why it matters (``03_STAGE_B_operator.md`` S1.7).  R is profiled, not
    regressed, so the minimiser is free to pick any grid point -- and every
    extra *usable* degree of freedom along that axis improves the fit at **every**
    candidate location, which quietly weakens discrimination rather than
    improving it.  If the axis is effectively 6-valued rather than 15-valued,
    the operator discriminates better than the grid size suggests, and the
    identifiability section (S2.1) should say so instead of quoting 15.

    Parameters
    ----------
    mu:
        ``[f, tau, R, L, 25, 3]``.
    sigma_eps:
        Two signatures closer than this everywhere are one level.
    mask:
        ``[25, 3]`` instrumented channels; all channels if omitted.
    l_index:
        Which L slice to measure on.  Defaults to the middle of the grid.

    Returns
    -------
    dict with the per-``(f, tau)`` level counts and their distribution.
    """
    n_f, n_t, n_r, n_l = mu.shape[:4]
    li = n_l // 2 if l_index is None else int(l_index)
    sel = mu[:, :, :, li]                                   # [f, tau, R, 25, 3]
    if mask is not None:
        m = np.asarray(mask) > 0
        sel = sel[..., m]                                   # [f, tau, R, C]
    else:
        sel = sel.reshape(n_f, n_t, n_r, -1)

    counts = np.zeros((n_f, n_t), dtype=np.int32)
    for fi in range(n_f):
        for ti in range(n_t):
            levels: list[np.ndarray] = []
            for ri in range(n_r):
                sig = sel[fi, ti, ri]
                # ``any``, not ``all``: a grid point is a duplicate when it matches
                # *some* level already opened, not when it matches every one of
                # them.  (With ``all`` a repeat of level 0 still differs from
                # levels 1..n, so every point opened a new level and the count was
                # always n_r.  ``any`` also handles the empty case correctly --
                # ``any([])`` is False, so the first point always opens a level.)
                if any(np.abs(sig - kept).max() <= sigma_eps for kept in levels):
                    continue
                levels.append(sig)
            counts[fi, ti] = max(len(levels), 1)
    flat = counts.reshape(-1)
    return {
        "sigma_eps": float(sigma_eps), "l_index": li, "n_grid_points": int(n_r),
        "counts": counts,
        "mean_levels": float(flat.mean()),
        "median_levels": float(np.median(flat)),
        "min_levels": int(flat.min()), "max_levels": int(flat.max()),
        "effective_fraction": float(flat.mean() / n_r),
    }
