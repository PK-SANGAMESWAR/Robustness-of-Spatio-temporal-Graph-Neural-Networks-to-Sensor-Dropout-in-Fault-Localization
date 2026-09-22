"""PIGNN grafts P1 / P2 — physics injected into the RGSAGE-Max backbone.

Screening design
----------------
The PIGNN literature is not one method: physics enters a graph model at
different *places*, and those places have different data requirements and
different costs (see ``PIGNN-SURVEY.md`` §4).  To find out which place matters
for fault localization we hold the backbone **fixed** — the same GRU-128
temporal encoder and the same 4-hop RGSAGE-Max spatial block that produced every
number in ``results/fullscale_final/`` — and vary only where the physics goes:

    P1  Type C, physics in the GRAPH   -- impedance-weighted messages
    P2  Type D, physics in the PRIOR   -- sensitivity bias on the class logits
    P3  Type A, physics in the LOSS    -- see ``models/physics_loss.py``

P1 and P2 live here because both are model-side; they are independent switches
on one class so the screen is a clean factorial rather than three unrelated
code paths.

Reduction to the baseline (the property that makes the screen fair)
------------------------------------------------------------------
With ``use_impedance_weights=False`` and ``use_sensitivity_bias=False`` this
class is **exactly** :class:`RGSAGE` — same modules, same parameter names, same
arithmetic.  It subclasses RGSAGE rather than reimplementing it, so weights
transfer 1:1 and the equivalence is asserted in ``tests/test_pignn.py`` rather
than assumed.  Any measured difference is therefore attributable to the physics
term and not to an incidental change in capacity or initialization.

P1 — impedance-weighted messages (Type C)
-----------------------------------------
On a feeder, two buses are electrically coupled in inverse proportion to the
impedance between them: a sag propagates strongly across a short, thick line and
weakly across a long, thin one.  The measured-only graph currently throws this
away — every edge carries the same weight, so a 0.05 ohm tie and a 0.88 ohm tie
are treated identically.  ``processing`` already computes ``edge_attr`` =
(R, X, |Z|) per edge and nothing consumes it.

Each message is scaled by a distance kernel on the edge impedance::

    w_ij = exp( -softplus(gamma) * z_ij ),    z_ij = |Z_ij| / median(|Z|)

with ``z = 0`` on self-loops, so ``w = 1`` there.  This is the same functional
form as PPGN's ``exp(-d^2/delta^2)`` neighbourhood kernel (arXiv 2107.02275),
except the distance is *electrical* rather than topological, which is the
sharper quantity on a feeder where line types differ by an order of magnitude.

``gamma`` is a single learned scalar, ``softplus``-constrained so the kernel can
only ever *decay* with impedance (never reward distance), and initialized so the
kernel starts mild.  The model is free to drive ``gamma`` to 0 and recover
uniform weighting — the prior is something it can reject, not a constraint.

P2 — sensitivity bias on the class logits (Type D)
--------------------------------------------------
``graph/sensitivity.py`` derives, for each fault class ``c``, the normalized
spatial signature ``s_c`` over the 25 sensors: how strongly a fault at ``c``
registers at each sensor, from the impedance of the path they share with the
substation.  This says which sensors are *informative* about which classes —
and the baselines never see it.

Node ``i``'s logit for class ``c`` is biased by that entry::

    logits[b, i, c] += beta * S_ext[c, i]

where ``S_ext`` is ``[26, 25]`` with row 0 (no-fault) all zeros — a constant row
would merely shift that logit uniformly and is left at zero deliberately, so the
no-fault class receives no geometric prior rather than an arbitrary one.

The effect is to let each sensor vote more confidently on the classes it can
actually see, which is exactly what the targeted-dropout result says matters:
losing the sensors nearest the fault costs 3.6x more than losing random ones
(0.448 vs 0.736 at k=1), so sensor-class proximity carries the signal.

``beta`` is one learned scalar initialized small; at ``beta = 0`` the bias
vanishes and the model is the baseline again.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .graphsage import RGSAGE


class PIGNNSage(RGSAGE):
    """RGSAGE-Max plus optional physics injections P1 and P2.

    Parameters
    ----------
    edge_index:
        Connectivity ``[2, E]`` of the 25-node measured-only graph.
    edge_attr:
        ``[E, F]`` per-edge features in ``edge_index`` order, as written by
        ``graph/edge_attributes.py``.  Column ``z_col`` is ``|Z|``.  Required
        when ``use_impedance_weights``.
    sensitivity:
        ``[25, 25]`` L2-normalized fault signatures from
        ``graph.sensitivity.load_sensitivity()``; row ``label-1`` is class
        ``label``'s signature over the sensor axis.  Required when
        ``use_sensitivity_bias``.
    use_impedance_weights:
        Enable P1.
    use_sensitivity_bias:
        Enable P2.
    gamma_init, beta_init:
        Initial values for the two learned physics scalars.
    z_col:
        Column of ``edge_attr`` holding ``|Z|`` (default 2, matching
        ``DEFAULT_EDGE_FEATURES = ["R", "X", "Z_mag"]``).
    """

    def __init__(
        self,
        edge_index,
        edge_attr=None,
        sensitivity=None,
        *,
        use_impedance_weights: bool = False,
        use_sensitivity_bias: bool = False,
        gamma_init: float = 1.0,
        beta_init: float = 0.1,
        z_col: int = 2,
        **kwargs,
    ):
        super().__init__(edge_index, **kwargs)
        self.use_impedance_weights = bool(use_impedance_weights)
        self.use_sensitivity_bias = bool(use_sensitivity_bias)
        n_edges = int(self.edge_index.shape[1])

        # ---- P1: normalized edge impedance + learned decay -----------------
        if self.use_impedance_weights:
            if edge_attr is None:
                raise ValueError("use_impedance_weights=True requires edge_attr")
            attr = np.asarray(edge_attr, dtype=np.float64)
            if attr.ndim != 2 or attr.shape[0] != n_edges:
                raise ValueError(
                    f"edge_attr must be [E={n_edges}, F], got {tuple(attr.shape)}")
            if not 0 <= z_col < attr.shape[1]:
                raise ValueError(f"z_col {z_col} out of range for {attr.shape[1]} features")
            z = attr[:, z_col].astype(np.float64)
            if not np.all(np.isfinite(z)) or np.any(z < 0):
                raise ValueError("edge |Z| must be finite and non-negative")
            scale = float(np.median(z))
            if scale <= 0:
                raise ValueError("median edge |Z| is zero; cannot normalize")
            # Self-loops are appended after the real edges by RGSAGE.aggregate
            # and carry z = 0, so their kernel weight is exactly 1.
            z_full = np.concatenate([z / scale, np.zeros(self.edge_index.max().item() + 1)])
            self.register_buffer("edge_z", torch.as_tensor(z_full, dtype=torch.float32))
            self.gamma_raw = nn.Parameter(torch.tensor(float(gamma_init)))
        else:
            self.register_buffer("edge_z", torch.empty(0))

        # ---- P2: class-by-sensor sensitivity prior --------------------------
        if self.use_sensitivity_bias:
            if sensitivity is None:
                raise ValueError("use_sensitivity_bias=True requires sensitivity")
            s = np.asarray(sensitivity, dtype=np.float32)
            if s.shape != (25, 25):
                raise ValueError(f"sensitivity must be [25, 25], got {tuple(s.shape)}")
            # [26, 25]: row 0 = no-fault, deliberately all zeros.
            s_ext = np.zeros((26, 25), dtype=np.float32)
            s_ext[1:] = s
            # Stored transposed to [25 sensors, 26 classes] so it adds directly
            # to the [B, N, C] logits without a transpose at every step.
            self.register_buffer("sens_bias", torch.as_tensor(s_ext.T))
            self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        else:
            self.register_buffer("sens_bias", torch.empty(0))

    # ------------------------------------------------------------------
    # P1 hooks into aggregation
    # ------------------------------------------------------------------

    def edge_weights(self) -> torch.Tensor:
        """Impedance kernel ``exp(-softplus(gamma) * z)`` over edges+self-loops."""
        return torch.exp(-nn.functional.softplus(self.gamma_raw) * self.edge_z)

    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        """Neighbour aggregate, impedance-weighted when P1 is enabled.

        Falls back to :meth:`RGSAGE.aggregate` verbatim when P1 is off, so the
        unweighted path is not merely equivalent but *the same code*.
        """
        if not self.use_impedance_weights:
            return super().aggregate(x)

        b, n, d = x.shape
        src, dst = self.edge_index
        loop = torch.arange(n, device=x.device)
        src = torch.cat([src, loop])
        dst = torch.cat([dst, loop])

        w = self.edge_weights().to(x.dtype)          # [E + N]
        messages = x[:, src] * w.view(1, -1, 1)      # [B, E+N, d]

        if self.aggregator == "mean":
            out = x.new_zeros((b, n, d)).index_add(1, dst, messages)
            denom = w.new_zeros(n).index_add(0, dst, w)
            return out / denom[None, :, None].clamp(min=1e-6)

        dst_expanded = dst.view(1, -1, 1).expand(b, -1, d)
        out = x.new_full((b, n, d), float("-inf"))
        out = out.scatter_reduce(1, dst_expanded, messages, reduce="amax",
                                 include_self=True)
        return torch.where(out.isinf(), torch.zeros_like(out), out)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, edge_index=None) -> torch.Tensor:
        logits = super().forward(x, edge_index)      # [B, N, 26]
        if self.use_sensitivity_bias:
            logits = logits + self.beta * self.sens_bias.unsqueeze(0)
        return logits
