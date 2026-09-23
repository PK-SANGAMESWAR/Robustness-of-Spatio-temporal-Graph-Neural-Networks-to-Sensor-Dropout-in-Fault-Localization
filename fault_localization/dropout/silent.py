"""Silent sensor failures: the mask still says 1, the value is wrong.

Every failure mode in Paper 1 announces itself.  Random, persistent, contiguous
and targeted dropout all zero the availability mask, and a dropout-aware model
learns to read ``m = 0`` as "this node is dark".  The failure a utility actually
fears is the other one: a uPMU that keeps reporting while its reading has
drifted, frozen, or swapped phases.  Nothing in the mask distinguishes that from
a healthy sensor, so no masking policy -- and no amount of dropout-aware
training -- can react to it.  That is the gap PCRG-STGNN's trust gate is for,
and this module is the perturbation it has to catch.

Four kinds (``05_STAGE_D_infra.md`` S3.1):

``stuck``
    Freeze the sensor at its value in the **first timestep of the window**.
    Defined this way so it is input-only -- no event metadata, no inception
    offset (``01_SPEC.md`` §5).  For windows 21-39 the first timestep is
    pre-fault, so this is the realistic case: a sensor that stopped updating
    before the fault arrived.  It is also the most realistic of the four; weight
    the discussion toward it.
``bias``
    A multiplicative ratio error of +-2 % or +-5 %, the classic instrument
    transformer / calibration fault.
``gauss``
    Extra white noise at sigma = 0.02 p.u. on the affected sensor only -- a unit
    whose front end has degraded, not the fleet-wide noise of ``noise.py``.
``swap``
    Permute the node's instrumented phase channels (cyclic, so no channel keeps
    its own value).  Requires at least two instrumented phases.

Three invariants, all asserted in ``tests/test_silent.py``:

1. The availability mask is **never** touched.  If it were, this would be
   ordinary dropout under a different name and the whole experiment would be
   circular.
2. Uninstrumented phases (``m = 0``, ``V = 0``) are never written to.  Only 14 of
   the 25 uPMUs are three-phase; manufacturing a reading on a phase that has no
   instrument would be a different, and fictional, fault.
3. Severities are specified in **per-unit**, so corruption is de-normalised,
   applied, and re-normalised -- matching ``dropout/noise.py``, which states the
   same reasoning for measurement noise.

Seeding follows ``masking.py``: callers derive the per-batch seed as
``seed * 10000 + batch_idx`` so that every model sees the identical corruption,
and so that a constant seed cannot silently collapse the number of realisations
to one per batch.
"""

from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch

SilentKind = Literal["stuck", "bias", "gauss", "swap"]
SILENT_KINDS: tuple[str, ...] = ("stuck", "bias", "gauss", "swap")

#: Default severity per kind, in per-unit where the quantity is a voltage.
DEFAULT_SEVERITY: dict[str, float] = {
    "stuck": 0.0,    # unused -- "stuck" has no magnitude knob
    "bias": 0.02,    # relative, i.e. 2 %
    "gauss": 0.02,   # p.u. standard deviation
    "swap": 0.0,     # unused
}


def _denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x[..., :3] * std + mean


def apply_silent_failure_batch(
    x: torch.Tensor,
    n_sensors: int,
    kind: SilentKind,
    seed: int,
    norm_stats: tuple[np.ndarray, np.ndarray] | None = None,
    severity: float | None = None,
) -> tuple[torch.Tensor, list[list[int]]]:
    """Corrupt ``n_sensors`` uPMUs per sample without touching the mask.

    Parameters
    ----------
    x:
        Batch ``[B, N, S, 6]`` as the loader delivers it: z-scored voltages in
        channels 0-2, binary availability in 3-5.
    n_sensors:
        How many uPMUs to corrupt per sample (the study uses 1, 2 or 3).  0
        returns the batch **byte-identical**, which is what makes the
        "no corruption" arm exactly the baseline.
    kind:
        One of :data:`SILENT_KINDS`.
    seed:
        Per-batch seed; callers pass ``seed * 10000 + batch_idx``.
    norm_stats:
        ``(mean, std)`` over the three voltage channels.  Required for every
        kind except ``stuck`` and ``swap``, which are pure rearrangements of
        values already in the tensor and so are exact in normalised space.
    severity:
        Overrides :data:`DEFAULT_SEVERITY`.

    Returns
    -------
    ``(x_corrupted, corrupted_nodes)`` -- the node indices per sample, which are
    the ground-truth labels for the S5.1 detection AUROC.
    """
    if kind not in SILENT_KINDS:
        raise ValueError(f"kind must be one of {SILENT_KINDS}, got {kind!r}")
    if x.dim() != 4 or x.shape[-1] != 6:
        raise ValueError(f"expected [B, N, S, 6], got {tuple(x.shape)}")
    b, n, _, _ = x.shape
    if n_sensors <= 0:
        return x, [[] for _ in range(b)]

    sev = DEFAULT_SEVERITY[kind] if severity is None else float(severity)
    avail = x[..., 3:] > 0.5                                  # [B, N, S, 3]
    node_ok = avail.any(dim=-1).any(dim=-1)                   # [B, N] any phase instrumented
    if kind == "swap":
        # Needs at least two instrumented phases to permute.
        node_ok = node_ok & (avail[:, :, 0, :].sum(dim=-1) >= 2)

    rng = random.Random(seed)
    corrupted: list[list[int]] = []
    for sample in range(b):
        candidates = [i for i in range(n) if bool(node_ok[sample, i])]
        take = min(n_sensors, len(candidates))
        corrupted.append(sorted(rng.sample(candidates, take)) if take else [])

    x_out = x.clone()
    if kind in ("bias", "gauss"):
        if norm_stats is None:
            raise ValueError(f"kind={kind!r} needs norm_stats to work in per-unit")
        mean = torch.as_tensor(np.asarray(norm_stats[0], dtype=np.float32),
                               device=x.device, dtype=x.dtype)
        std = torch.as_tensor(np.asarray(norm_stats[1], dtype=np.float32),
                              device=x.device, dtype=x.dtype)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1)

    for sample, nodes in enumerate(corrupted):
        for node in nodes:
            phase_ok = avail[sample, node, 0, :]               # [3]
            if kind == "stuck":
                frozen = x_out[sample, node, 0:1, :3]
                new = frozen.expand_as(x_out[sample, node, :, :3]).clone()
            elif kind == "swap":
                idx = torch.nonzero(phase_ok, as_tuple=False).flatten()
                rolled = torch.roll(idx, shifts=1)
                new = x_out[sample, node, :, :3].clone()
                new[:, idx] = x_out[sample, node, :, :3][:, rolled]
            else:
                volts = _denormalize(x_out[sample, node], mean, std)   # [S, 3] p.u.
                if kind == "bias":
                    sign = 1.0 if rng.random() < 0.5 else -1.0
                    volts = volts * (1.0 + sign * sev)
                else:  # gauss
                    noise = torch.randn(volts.shape, generator=generator).to(
                        device=volts.device, dtype=volts.dtype)
                    volts = volts + noise * sev
                new = (volts - mean) / std
            # Write back only on instrumented phases -- m = 0 channels keep
            # V = 0, and the mask itself is never written at all.
            x_out[sample, node, :, :3] = torch.where(
                phase_ok.unsqueeze(0), new, x_out[sample, node, :, :3])

    return x_out, corrupted
