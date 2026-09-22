"""P3 — physics-consistency loss (Type A, physics in the LOSS).

Why this arm exists
-------------------
This is the arm that most people mean by "physics-informed": a residual penalty
added to the objective.  In power systems that residual is normally a power-flow
or Kirchhoff violation, ``Y*dU = dC + delta_ij`` — and that is **not computable
from this corpus**, because the simulator records voltage *magnitudes* and
discards the phase angle (``simulation/monitors.py:78-80``).  The full residual
needs complex voltages; with magnitudes alone it is undefined, not merely
approximate.  See ``PIGNN-SURVEY.md`` §1.

What is computable is the *linearized, magnitude-only* consequence of the same
physics.  On a radial feeder the sag seen at sensor ``i`` from a fault at bus
``b`` is governed to first order by the impedance of the path they share with
the substation, so the sag pattern across the sensor array factorizes as

    dv_i  ~=  alpha * s_{b,i}

(``graph/sensitivity.py``): a scalar severity ``alpha`` absorbing fault
resistance and loading, times a spatial signature ``s_b`` fixed by the network.
The *direction* of ``dv`` in sensor space is therefore a physical fingerprint of
the fault location, while its magnitude is not.  That is what this loss scores.

The residual
------------
For each window, the observed deviation ``d`` over the 25 sensors is compared to
each class signature by a **masked cosine**, and the model is penalized for
placing probability on classes whose signature disagrees::

    L_phys = (1 - p_0) * sum_{c=1..25} p~_c * (1 - cos_a(d, s_c))

where ``p`` is the event-level distribution (mean of the per-node softmaxes,
matching ``voting.soft_vote``), ``p~`` is ``p`` restricted to the 25 fault
classes and renormalized, and ``cos_a`` is computed over *available* sensors
only.  Total objective::

    L = CE + lambda * L_phys

Design choices worth defending
------------------------------
*Cosine, not distance.*  The factorization says severity is a free scalar, so
only the pattern is physical.  A cosine is invariant to ``alpha`` by
construction — which is also why this term is meaningful for high-impedance
faults, where ``alpha`` is ~0.01 p.u. but the pattern is unchanged.

*Deviation measured as a within-window sag depth.*  ``d_i`` is the peak-to-
trough voltage swing at sensor ``i``, taken on the worst-affected phase::

    d_i = max_p [ max_t v_{i,p,t} - min_t v_{i,p,t} ]   over available phases p

This definition is *reference-free* — it needs neither a nominal voltage nor the
corpus mean.  That matters: an earlier version measured deviation against the
corpus channel means (0.83 / 0.60 / 0.76 p.u., dragged far below nominal because
absent phases are stored as 0.0), which zeroed the deviation on 97.6% of node
entries and left the cosine degenerate.  It is also legitimate on *every* window
the label calls a fault: the fault starts at step 40, windows are 20 steps long
with starts 0-39, so every fault-labelled window contains at least one pre-fault
sample to serve as the peak.

The max over phases, rather than the mean, is deliberate: most fault types in
this corpus are not three-phase, so averaging a single-phase sag over three
channels dilutes it threefold.  Measured on 12,800 test windows with all 26
labels present, max-over-phase scores 14.8% top-1 against 13.0% for
mean-over-phase (chance = 4.0%), and both leave all 25 sensors informative with
**no** degenerate all-tie windows -- which a ``max - last`` variant, though
marginally sharper at 16.3%, does not (25.3% degenerate).

*Masked by availability.*  Dropped and uninstrumented sensors contribute to
neither side of the cosine, so the term degrades gracefully under dropout
instead of comparing against zeros the physics never predicted.

*Gated by (1 - p_0).*  The signature model says nothing about a no-fault window,
so the term is switched off exactly when the model believes there is no fault —
and on a genuine no-fault window, where ``d ~ 0`` makes every fault cosine ~0
and every residual ~1, the gradient pushes ``p_0`` up, which is correct.

*Soft, not hard.*  ``sensitivity.py`` is explicit that this is a first-order
model which ignores the driving-point term ``Z_bb + R_f`` and load flow, and
that its templates are "used as an *initialization* for a model that is free to
move them, never as a fixed constraint".  A soft penalty with a tunable
``lambda`` respects that; a hard projection would not.  Published PIGNN work
warns that physics losses are "highly reliant on hyperparameters"
(arXiv 2509.25158), so ``lambda`` sensitivity must be reported, not tuned away.

Cost: **zero** extra parameters and **zero** inference cost — the term exists
only during training.  That is the property that makes it the interesting arm:
the paper's existing headline result is that a training-time fix
(dropout-aware training) beat an architectural one (SDR-STGNN).
"""

from __future__ import annotations

import torch

_EPS = 1e-8


def observed_deviation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sensor sag depth and availability from a normalized window.

    Parameters
    ----------
    x:
        ``[B, N, S, 6]`` — normalized voltages 0-2, availability mask 3-5.

    Returns
    -------
    (deviation, availability)
        ``deviation``    ``[B, N]``, non-negative; the peak-to-trough swing on
        the worst-affected available phase.  Larger = deeper sag.
        ``availability`` ``[B, N]`` in {0, 1}; 0 for a dropped or fully
        uninstrumented sensor.

    Notes
    -----
    Scale is irrelevant downstream — the comparison against the signatures is a
    cosine, matching the ``dv ~= alpha * s_b`` factorization in which severity
    is a free scalar — so no de-normalization is performed.
    """
    volts = x[..., :3]                                  # [B, N, S, 3]
    phase_mask = x[..., 3:].clamp(0, 1)                 # [B, N, S, 3]

    # Per-phase presence is static within a window; a dropped node has all six
    # channels zeroed, so the mask is the authoritative availability signal.
    present = phase_mask.amax(dim=2)                    # [B, N, 3]
    availability = (present.amax(dim=-1) > 0).to(x.dtype)

    # Restrict the temporal extrema to timesteps where the phase is present.
    big = torch.finfo(x.dtype).max
    hi = torch.where(phase_mask > 0, volts, torch.full_like(volts, -big)).amax(dim=2)
    lo = torch.where(phase_mask > 0, volts, torch.full_like(volts, big)).amin(dim=2)

    swing = torch.relu(hi - lo) * present               # [B, N, 3]
    deviation = swing.amax(dim=-1) * availability       # [B, N]
    return deviation, availability


def masked_cosine(d: torch.Tensor, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Cosine between an observed deviation and each class signature.

    Parameters
    ----------
    d: ``[B, N]`` observed deviation.
    s: ``[C, N]`` class signatures (rows need not be unit under masking).
    a: ``[B, N]`` availability in {0, 1}.

    Returns
    -------
    ``[B, C]`` cosine similarity computed over available sensors only.
    """
    da = (d * a).unsqueeze(1)                           # [B, 1, N]
    sa = s.unsqueeze(0) * a.unsqueeze(1)                # [B, C, N]
    num = (da * sa).sum(-1)                             # [B, C]
    den = da.norm(dim=-1) * sa.norm(dim=-1)             # [B, C]
    return num / den.clamp(min=_EPS)


def physics_consistency_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    signatures: torch.Tensor,
) -> torch.Tensor:
    """Sag-pattern consistency penalty (P3).

    Parameters
    ----------
    logits:
        ``[B, N, C]`` per-node class logits, as every model in this repo emits.
    x:
        ``[B, N, S, 6]`` the same normalized window the model consumed.
    signatures:
        ``[25, N]`` L2-normalized fault signatures, row ``label-1`` for class
        ``label``, from ``graph.sensitivity.load_sensitivity()['sensitivity']``.

    Returns
    -------
    Scalar loss in ``[0, 2]``, zero when the model is certain there is no fault.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B, N, C], got {tuple(logits.shape)}")
    if signatures.dim() != 2 or signatures.shape[0] != logits.shape[-1] - 1:
        raise ValueError(
            f"signatures must be [{logits.shape[-1] - 1}, N], "
            f"got {tuple(signatures.shape)}")

    d, a = observed_deviation(x)                        # [B, N], [B, N]

    # Event-level distribution: mean of per-node softmaxes. soft_vote() sums
    # them and argmaxes, so this is the same ranking, normalized.
    probs = torch.softmax(logits, dim=-1).mean(dim=1)   # [B, C]
    p_fault = probs[:, 1:]                              # [B, 25]
    p_none = probs[:, 0]                                # [B]

    total_fault = p_fault.sum(dim=-1)                   # [B]
    p_tilde = p_fault / total_fault.clamp(min=_EPS).unsqueeze(-1)

    cos = masked_cosine(d, signatures.to(x.dtype), a)   # [B, 25]
    residual = (p_tilde * (1.0 - cos)).sum(dim=-1)      # [B]
    return ((1.0 - p_none) * residual).mean()
