"""Axes of the physics signature table, and the one constant the corpus hides.

The operator of ``PC-STGNN/01_SPEC.md`` §2 is a fixed tensor
``mu[f, tau, R, L, i, p]`` plus a no-fault row ``mu0[L, i, p]``.  This module
owns the four hypothesis axes so that the table builder, the residual layer,
the identifiability study and every audit script agree on their order and
values without passing them around.

Axis order is fixed and load-bearing: ``(f, tau, R, L, sensor, phase)``.  The
residual flattens ``(tau, R, L)`` into one hypothesis axis, and
``unravel_hypothesis`` is the only sanctioned way back, so that a reported
``(tau*, R*, L*)`` can never drift out of step with the tensor it came from.
"""

from __future__ import annotations

import numpy as np

from fault_localization.config.constants import (
    FAULT_BUSES,
    FAULT_TYPES,
    NUM_PHASES,
    NUM_PMU_NODES,
    PRE_FAULT_MS,
)

# ---------------------------------------------------------------------------
# The frozen pre-fault load level
# ---------------------------------------------------------------------------

#: The load multiplier actually in force during the corpus's fault period.
#:
#: ``monitors.solve_prefault_timeseries`` rescales every load at the top of each
#: of the 40 pre-fault steps by ``1 + 0.003*sin(2*pi*step/40)``, and the fault
#: loop of ``simulation/runner.py`` (lines 138-150) **never rescales again** --
#: it injects the fault and solves 20 more steps at whatever the loads were left
#: at.  The last rescale is therefore pre-fault step 39, not step 40, so the
#: fault solves run at ``L * (1 + 0.003*sin(2*pi*39/40))`` and the step-40 value
#: of the sinusoid (exactly 1.0) is never applied.
#:
#: Getting this wrong shifts every signature by ~5e-4 p.u., which is the same
#: order as the sigma_eps the operator is calibrated to -- so it is asserted,
#: not assumed, by ``scripts/audit_snapshot_equivalence.py`` (Stage A S0.4).
PREFAULT_FREEZE: float = float(1.0 + 0.003 * np.sin(2.0 * np.pi * (PRE_FAULT_MS - 1) / PRE_FAULT_MS))

# ---------------------------------------------------------------------------
# Hypothesis axes
# ---------------------------------------------------------------------------

#: Fault resistances, 0.05 -> 30 ohm.  ``01_SPEC.md`` §2 originally specified
#: 15 log-spaced points with the corpus's own resistances {0.1, 1, 10} *not*
#: on the grid, on the theory that keeping them off-grid is what makes S1.7's
#: "how far does argmin_R land from R_true" check non-vacuous.
#:
#: ``[Corrected 2026-09-25]`` That theory cost S1.5 (``03_STAGE_B_operator.md``,
#: the decisive load-axis test): with R_true always off-grid, the joint
#: profiled argmin over (tau, R, L) let a nearby-but-wrong grid R trade off
#: against a compensating L shift and out-fit the (off-grid) true point,
#: corrupting L* recovery to 38.4% within one grid step (need >90%) even
#: though R* itself always landed on a sensible neighbour -- see
#: ``results/pcstgnn/audit/s1_4_7/s1_4_7_summary.pre_rgrid_fix.json``.  The
#: S1.7 branch-on-failure's own prescribed remedy (25 log-spaced points,
#: corpus R still off-grid) was tried and does not fix it (58.6%, still fails)
#: -- see ``results/pcstgnn/audit/s1_5_rgrid_diagnostic/rgrid_diagnostic.json``
#: (candidate ``dense25``).  Putting the corpus's own R values exactly
#: on-grid does fix it (97.3% overall, 93.8-99.2% stratified by R -- candidate
#: ``hybrid`` in the same file), so this is now
#: ``sorted(unique(15-pt log grid, {0.1, 1, 10})) = 18 points``, not 15.
#:
#: **Cost, documented rather than hidden**: S1.7's argmin_R-at-corpus-R check
#: is now much less informative at exactly the three values that matter --
#: ``r_star_exact_frac`` rises from 64.8% to 95.2% partly *by construction*
#: (R_true can now literally be the grid point), not from improved operator
#: fidelity. It is not fully vacuous (95.2%, not 100% -- some non-monotone
#: cells still miss it), but any future S1.7 R-interpolation claim should be
#: made against off-corpus R values, not {0.1, 1, 10}.
R_GRID: np.ndarray = np.unique(np.concatenate([np.geomspace(0.05, 30.0, 15), [0.1, 1.0, 10.0]]))

L_MIN: float = 0.5
L_MAX: float = 1.3
DEFAULT_L_STEP: float = 0.05


def l_grid(step: float = DEFAULT_L_STEP) -> np.ndarray:
    """Load-multiplier axis over ``[0.5, 1.3]`` at the given step.

    The step is a *decision*, not a default: S1.2 (``03_STAGE_B_operator.md``)
    chooses the coarsest value whose linear-interpolation error stays under
    1e-3 p.u., and records it in ``operator/grid.json``.  0.05 gives 17 points
    and ~70k solves; 0.025 gives 33 points and ~140k.
    """
    if step <= 0:
        raise ValueError("L step must be positive")
    span = L_MAX - L_MIN
    n = int(round(span / step)) + 1
    if abs((n - 1) * step - span) > 1e-9:
        raise ValueError(
            f"L step {step} does not divide the range [{L_MIN}, {L_MAX}] exactly"
        )
    return np.linspace(L_MIN, L_MAX, n)


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

NUM_LOCATIONS: int = len(FAULT_BUSES)          # 25 candidate fault locations
NUM_TYPES: int = len(FAULT_TYPES)              # 11
NUM_R: int = len(R_GRID)                       # 18 (was 15 pre-S1.5-fix)
NUM_CHANNELS: int = NUM_PMU_NODES * NUM_PHASES  # 75 sensor-phase channels

AXIS_ORDER: tuple[str, ...] = ("f", "tau", "R", "L", "sensor", "phase")


def table_shape(l_step: float = DEFAULT_L_STEP) -> tuple[int, ...]:
    """Shape of ``mu``: ``[25, 11, 15, n_L, 25, 3]``."""
    return (NUM_LOCATIONS, NUM_TYPES, NUM_R, len(l_grid(l_step)), NUM_PMU_NODES, NUM_PHASES)


def nofault_shape(l_step: float = DEFAULT_L_STEP) -> tuple[int, ...]:
    """Shape of ``mu0``: ``[n_L, 25, 3]``."""
    return (len(l_grid(l_step)), NUM_PMU_NODES, NUM_PHASES)


def num_hypotheses(n_l: int) -> int:
    """Size of the flattened ``(tau, R, L)`` hypothesis axis."""
    return NUM_TYPES * NUM_R * n_l


def unravel_hypothesis(index, n_l: int):
    """Flat hypothesis index -> ``(tau_idx, r_idx, l_idx)``.

    Inverse of ``tau_idx * NUM_R * n_l + r_idx * n_l + l_idx`` -- i.e. C order
    over ``(tau, R, L)``, which is the order ``residual.py`` flattens the table
    in and the only order any reported ``(tau*, R*, L*)`` may be decoded with.
    """
    index = np.asarray(index)
    tau_idx, rem = np.divmod(index, NUM_R * n_l)
    r_idx, l_idx = np.divmod(rem, n_l)
    return tau_idx, r_idx, l_idx
