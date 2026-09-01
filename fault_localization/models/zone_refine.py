"""ZoneRefine-STGNN — physics-partitioned hierarchical fault localization.

The idea
--------
Fault localization on a radial feeder decomposes into two questions of very
different character, and the existing models answer both with one flat 26-way
softmax:

1. **Which electrical zone?**  On a radial feeder the sag pattern across the
   sensors factorizes as ``dv_i ~= a * s_{b,i}``: a scalar severity ``a`` that
   absorbs fault resistance and loading, times a spatial signature ``s_b`` fixed
   by the network impedance.  Identifying ``b`` from the *pattern* is a matched-
   filter problem with a closed-form solution, and it is exactly the classical
   impedance-based method — made differentiable.
2. **Which bus inside that zone?**  Some signatures are collinear and no
   amplitude-based method can separate them (see
   :mod:`fault_localization.graph.sensitivity`).  Only temporal shape and
   per-phase asymmetry can, and that genuinely needs a learned model.

ZoneRefine answers each with the machinery suited to it and multiplies the two::

    p(class c)  =  p(zone(c) | physics residual) * p(c | zone(c), learned features)
    p(no fault) =  p(no-fault | deviation energy)

Why this handles missing sensors differently from everything else here
----------------------------------------------------------------------
Stage 1 fits its templates by **masked least squares** — the projection uses
only the coordinates actually observed.  Under a missing-at-random sensor
failure that is the maximum-likelihood estimator, not a heuristic: dropping
sensors removes equations from the fit, it does not inject out-of-distribution
zeros the way feature masking does.  Neither an availability gate
(SDR-STGNN), an augmentation (dropout-aware training), nor imputation has that
property.

Because the zone posterior below divides by the observed deviation energy, it is
**invariant to fault severity**: the same zone is inferred whether the sag is
0.83 p.u. or 0.01 p.u.  Severity is instead the evidence for *no fault*.

Falsifiable prediction this architecture makes
----------------------------------------------
Under sensor dropout, zone accuracy should degrade gracefully while within-zone
accuracy collapses first.  Both are recorded during evaluation, so the claim can
be checked rather than asserted.

Output contract
---------------
Returns **log-probabilities** shaped ``[B, 1, 26]``.  Since they are already
normalized, ``log_softmax`` over them is the identity, so
``expanded_cross_entropy`` is exact and ``soft_vote`` (over a single node) is the
identity — the model is a drop-in for every existing training, evaluation and
dropout-study script.
"""

from __future__ import annotations

import torch
from torch import nn

from .common import TemporalEncoder
from .sdr_stgnn import node_availability

_EPS = 1e-8


class ZoneRefineSTGNN(nn.Module):
    """Physics zone scorer + learned within-zone refinement.

    Parameters
    ----------
    edge_index:
        Connectivity ``[2, E]`` of the 25-node measured-only graph.
    zone_templates:
        ``[Z, 25]`` normalized spatial signatures, one per electrical zone,
        from :func:`fault_localization.graph.sensitivity.load_sensitivity`.
        Used as initialization; learned unless ``freeze_templates``.
    zone_of_class:
        ``[26]`` zone index per class, entry 0 (no-fault) set to -1.
    learn_templates:
        Let gradients move the physics templates (default True).  Setting
        False pins them to the impedance model — ablation Z2.
    use_physics_zones:
        If False the zone stage is replaced by a free linear scorer over the
        pooled features, removing the physics entirely — ablation Z3.  The
        hierarchy is kept so the comparison isolates the *physics*, not the
        two-stage structure.
    """

    def __init__(
        self,
        edge_index,
        zone_templates,
        zone_of_class,
        aggregator: str = "max",
        temporal_hidden_dim: int = 128,
        gnn_hidden_dim: int = 64,
        num_classes: int = 26,
        dropout: float = 0.35,
        num_layers: int = 1,
        learn_templates: bool = True,
        use_physics_zones: bool = True,
    ):
        super().__init__()
        if aggregator not in {"mean", "max"}:
            raise ValueError("aggregator must be 'mean' or 'max'")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.aggregator = aggregator
        self.use_physics_zones = use_physics_zones

        templates = torch.as_tensor(zone_templates, dtype=torch.float32)
        zone_of = torch.as_tensor(zone_of_class, dtype=torch.long)
        if templates.ndim != 2 or templates.shape[1] != 25:
            raise ValueError(f"zone_templates must be [Z, 25], got {tuple(templates.shape)}")
        if zone_of.shape != (num_classes,):
            raise ValueError(f"zone_of_class must be [{num_classes}]")
        self.n_zones = templates.shape[0]

        # Boolean membership [Z, 26] — used to restrict the within-zone softmax.
        membership = torch.zeros(self.n_zones, num_classes, dtype=torch.bool)
        for c in range(1, num_classes):
            membership[int(zone_of[c]), c] = True
        self.register_buffer("membership", membership)
        self.register_buffer("zone_of_class", zone_of)

        self.encoder = TemporalEncoder(6, temporal_hidden_dim)
        self.register_buffer("edge_index", torch.as_tensor(edge_index, dtype=torch.long))

        dims = [temporal_hidden_dim] + [gnn_hidden_dim] * num_layers
        self.spatial_layers = nn.ModuleList(
            nn.Linear(dims[i] * 2, dims[i + 1]) for i in range(num_layers))

        # --- stage 1: physics -------------------------------------------
        # Per-sensor scalar deviation, read off the temporal embedding.
        self.deviation = nn.Linear(temporal_hidden_dim, 1)
        if learn_templates:
            self.templates = nn.Parameter(templates.clone())
        else:
            self.register_buffer("templates", templates.clone())
        # Sharpness of the zone posterior; softplus keeps it positive.
        self.zone_temperature = nn.Parameter(torch.tensor(2.0))
        # No-fault evidence is total deviation energy, not pattern match.
        self.nofault_scale = nn.Parameter(torch.tensor(-1.0))
        self.nofault_bias = nn.Parameter(torch.tensor(0.0))
        # Ablation Z3: a free zone scorer with no physics in it.
        self.free_zone = nn.Linear(gnn_hidden_dim, self.n_zones)

        # --- stage 2: learned within-zone refinement ---------------------
        self.refine = nn.Sequential(
            nn.LayerNorm(gnn_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim, num_classes),
        )

    # ------------------------------------------------------------------
    def _aggregate(self, x: torch.Tensor) -> torch.Tensor:
        """Neighbourhood aggregate (mirrors RGSAGE.aggregate)."""
        b, n, d = x.shape
        src, dst = self.edge_index
        loop = torch.arange(n, device=x.device)
        src = torch.cat([src, loop])
        dst = torch.cat([dst, loop])

        if self.aggregator == "mean":
            out = x.new_zeros((b, n, d)).index_add(1, dst, x[:, src])
            counts = torch.bincount(dst, minlength=n).to(x.dtype)
            return out / counts[None, :, None].clamp(min=1)

        gathered = x[:, src]
        dst_expanded = dst.view(1, -1, 1).expand(b, -1, d)
        out = x.new_full((b, n, d), float("-inf"))
        out = out.scatter_reduce(1, dst_expanded, gathered, reduce="amax",
                                 include_self=True)
        return torch.where(out.isinf(), torch.zeros_like(out), out)

    def _zone_logits(self, deviation: torch.Tensor, avail: torch.Tensor,
                     pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Masked least-squares match of the deviation against each template.

        Returns ``(zone_logits [B, Z], nofault_logit [B, 1])``.

        For each zone the projection of the observed deviation onto the template
        is computed over the available coordinates only::

            energy_z = <M d, M s_z>^2 / ||M s_z||^2

        which is the reduction in squared error from fitting the free amplitude
        ``a_z`` — i.e. exactly the masked least-squares residual, up to the
        constant ``||M d||^2``.  Dividing by that constant makes the score the
        *fraction of observed energy explained*, so the zone posterior does not
        depend on how severe the fault is.
        """
        masked_d = deviation * avail                                 # [B, N]
        masked_t = self.templates.unsqueeze(0) * avail.unsqueeze(1)  # [B, Z, N]

        projection = torch.einsum("bn,bzn->bz", masked_d, masked_t)
        template_energy = (masked_t ** 2).sum(-1)                    # [B, Z]
        observed_energy = (masked_d ** 2).sum(-1, keepdim=True)      # [B, 1]

        explained = projection ** 2 / (template_energy + _EPS)
        fraction = explained / (observed_energy + _EPS)              # [B, Z] in [0,1]

        if self.use_physics_zones:
            zone_logits = nn.functional.softplus(self.zone_temperature) * fraction
        else:
            zone_logits = self.free_zone(pooled)                     # ablation Z3

        # A quiet array is evidence of no fault; a loud one is evidence against.
        nofault = (self.nofault_bias
                   + self.nofault_scale * torch.log1p(observed_energy))
        return zone_logits, nofault

    def _forward(self, x: torch.Tensor) -> dict:
        avail = node_availability(x)                    # [B, N]
        h0 = self.encoder(x)                            # [B, N, temporal_dim]

        h = h0
        for i, layer in enumerate(self.spatial_layers):
            h = layer(torch.cat([h, self._aggregate(h)], dim=-1))
            if i < len(self.spatial_layers) - 1:
                h = torch.relu(h)

        # Availability-weighted mean pool, so dropped sensors do not dilute the
        # graph representation the refinement head reads.
        weights = avail.unsqueeze(-1)
        pooled = (h * weights).sum(1) / weights.sum(1).clamp(min=1.0)

        deviation = self.deviation(h0).squeeze(-1)      # [B, N]
        zone_logits, nofault_logit = self._zone_logits(deviation, avail, pooled)

        # p(zone) over Z zones + the no-fault alternative, one joint softmax.
        joint = torch.cat([nofault_logit, zone_logits], dim=-1)      # [B, 1+Z]
        log_zone = torch.log_softmax(joint, dim=-1)
        log_nofault, log_zone = log_zone[:, :1], log_zone[:, 1:]     # [B,1], [B,Z]

        # p(class | zone): softmax restricted to each zone's member classes.
        fine = self.refine(pooled)                                   # [B, 26]
        neg_inf = torch.finfo(fine.dtype).min
        fine_z = fine.unsqueeze(1).expand(-1, self.n_zones, -1)      # [B,Z,26]
        fine_z = fine_z.masked_fill(~self.membership.unsqueeze(0), neg_inf)
        log_within = torch.log_softmax(fine_z, dim=-1)               # [B,Z,26]

        # log p(c) = log p(zone(c)) + log p(c | zone(c)); singleton zones
        # contribute exactly 0 from the second term.
        log_joint = log_zone.unsqueeze(-1) + log_within              # [B,Z,26]
        log_fault = torch.logsumexp(log_joint, dim=1)                # [B, 26]

        log_probs = torch.cat([log_nofault, log_fault[:, 1:]], dim=-1)
        return {"log_probs": log_probs.unsqueeze(1),
                "log_zone": log_zone, "log_nofault": log_nofault,
                "deviation": deviation}

    def zone_posterior(self, x: torch.Tensor) -> torch.Tensor:
        """Zone posterior ``[B, Z]`` — exposed so the two stages can be scored
        separately during evaluation (the architecture's falsifiable claim)."""
        return self._forward(x)["log_zone"].exp()

    def forward(self, x: torch.Tensor, edge_index=None) -> torch.Tensor:
        return self._forward(x)["log_probs"]
