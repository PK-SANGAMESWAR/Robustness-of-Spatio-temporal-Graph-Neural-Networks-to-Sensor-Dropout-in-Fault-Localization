"""PC-STGNN -- the Paper-1 backbone with a fixed physics operator fused in.

``PC-STGNN/01_SPEC.md`` §6.  The backbone is untouched: the same GRU-128
temporal encoder and the same 4-hop RGSAGE-Max spatial block that produced every
number in ``results/``.  What changes is that each node's class distribution is
reweighted by a physics prior before the soft vote::

    z'[n, f] = z[n, f] - beta * rho_f(A)         same rho broadcast to every node

    sum_n softmax(z'[n])_f  ~  sum_n exp(z[n, f]) * exp(-beta * rho_f) / Z_n

which is a **product of experts per node**, not an additive bias on a
graph-level logit.  Stating it correctly matters for the paper, and implementing
it this way means the per-node cross-entropy of ``models/common.py``, the soft
vote of ``models/voting.py``, ``dropout/evaluator.py`` and every loader stay
exactly as they are.

The whole physics path adds **one** learnable scalar, ``beta``, initialised to 0.
With every switch off this class is byte-for-byte :class:`RGSAGE` with
``aggregator="max"`` -- asserted in ``tests/test_pc_stgnn.py``, and the property
that lets Stage E say "lambda = beta = 0 recovers the Paper-1 model exactly"
(S4.1).

Ablation switches, each isolating one reviewer objection (``06_STAGE_E`` S4.4):

``use_fusion`` (R5)
    The beta fusion above.
``shuffle_seed`` (R6)
    Permute which signature belongs to which location.  The operator keeps
    exactly the same statistics and the same number of parameters, but now
    points at the wrong places -- so any gain that survives is regularisation,
    not physics.  This is the arm that carries the burden of proof.
``use_type_head`` (R1)
    An auxiliary fault-type head.  Never read by the physics: the operator is a
    function of the inputs alone (``01_SPEC.md`` §5), so no network output may
    ever be an argument to mu.
``use_decoder`` (R2)
    A learned decoder that reconstructs the dropped sensors, standing in for
    "any masked autoencoder would have done it".
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from fault_localization.physics import residual as phys
from fault_localization.physics.observation import observe
from .graphsage import RGSAGE
from .voting import soft_vote


class PCSTGNN(RGSAGE):
    """RGSAGE-Max plus the fixed physics operator of ``01_SPEC.md``.

    Parameters
    ----------
    edge_index:
        Connectivity ``[2, E]`` of the 25-node measured-only graph.
    table:
        Flattened signature table ``[26, H, 75]`` from
        :func:`physics.residual.flatten_table`.  Required whenever any physics
        switch is on.
    norm_stats:
        ``(mean, std)`` over the three voltage channels, so the layer can
        de-normalise the batch back to per-unit before it meets ``mu``.
    sigma_eps, sigma0:
        Residual floor and injected-noise level.  ``sigma0`` **must** be set to
        the noise actually injected under the noise experiments
        (``12_CHECKLISTS.md`` §C).
    """

    def __init__(
        self,
        edge_index,
        table=None,
        norm_stats=None,
        *,
        use_fusion: bool = False,
        beta_init: float = 0.0,
        sigma_eps: float = 1e-3,
        sigma0: float = 0.0,
        shuffle_seed: int | None = None,
        use_type_head: bool = False,
        num_fault_types: int = 11,
        use_decoder: bool = False,
        decoder_hidden: int = 64,
        obs_reduce: str = "last",
        chunk_f: int = 1,
        **kwargs,
    ):
        super().__init__(edge_index, **kwargs)
        self.use_fusion = bool(use_fusion)
        self.use_type_head = bool(use_type_head)
        self.use_decoder = bool(use_decoder)
        self.obs_reduce = obs_reduce
        self.chunk_f = int(chunk_f)
        self.sigma0 = float(sigma0)
        self.needs_physics = self.use_fusion

        if table is not None:
            tensor = torch.as_tensor(np.asarray(table, dtype=np.float32))
            if tensor.ndim != 3 or tensor.shape[0] != 26 or tensor.shape[2] != 75:
                raise ValueError(f"table must be [26, H, 75], got {tuple(tensor.shape)}")
            perm = torch.arange(26)
            if shuffle_seed is not None:
                # Permute the 25 *locations* only.  Class 0 (no fault) stays put:
                # shuffling it would change what "no fault" means rather than
                # where a fault is, which is a different -- and uninteresting --
                # ablation.
                g = torch.Generator().manual_seed(int(shuffle_seed))
                perm = torch.cat([torch.zeros(1, dtype=torch.long),
                                  torch.randperm(25, generator=g) + 1])
                tensor = tensor[perm]
            self.register_buffer("phys_table", tensor)
            self.register_buffer("shuffle_map", perm)
        else:
            if self.needs_physics:
                raise ValueError("physics switches require a signature table")
            self.register_buffer("phys_table", torch.empty(0))
            self.register_buffer("shuffle_map", torch.empty(0, dtype=torch.long))

        if norm_stats is not None:
            mean, std = norm_stats
            self.register_buffer("norm_mean", torch.as_tensor(np.asarray(mean, dtype=np.float32)))
            self.register_buffer("norm_std", torch.as_tensor(np.asarray(std, dtype=np.float32)))
        else:
            if self.needs_physics:
                raise ValueError("physics switches require norm_stats to reach per-unit")
            self.register_buffer("norm_mean", torch.empty(0))
            self.register_buffer("norm_std", torch.empty(0))

        self.register_buffer("sigma_eps", torch.as_tensor(float(sigma_eps)))

        # Parameters are created *only* when their switch is on, and only after
        # super().__init__ has drawn the backbone's weights.  Both matter: an
        # unconditional nn.Linear would consume the RNG stream and the all-off
        # model would no longer be byte-identical to RGSAGE under the same seed.
        gnn_dim = self.classifier.linear.in_features
        if self.use_fusion:
            self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        if self.use_type_head:
            self.type_head = nn.Linear(gnn_dim, num_fault_types)
        if self.use_decoder:
            self.decoder = nn.Sequential(
                nn.Linear(gnn_dim, decoder_hidden), nn.ReLU(),
                nn.Linear(decoder_hidden, 3),
            )

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _rho(self, x: torch.Tensor):
        """``(rho [B, 26], hyp_star [B, 26], s, a)`` -- no gradient, ever.

        rho is a function of the inputs and the fixed table (``01_SPEC.md`` §5),
        so it is a constant with respect to the weights.  Computing it under
        ``no_grad`` is therefore not an optimisation that changes semantics; it
        is the semantics, and it also keeps the [B, H, 75] intermediate out of
        the autograd graph.
        """
        with torch.no_grad():
            s, a = observe(x, self.norm_mean, self.norm_std, self.obs_reduce)
            rho, hyp = phys.rho(s, a, self.phys_table, self.sigma_eps,
                                self.sigma0, chunk_f=self.chunk_f)
        return rho, hyp, s, a

    def _hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Node embeddings just before the classifier head."""
        h = self.encoder(x)
        for i, layer in enumerate(self.spatial_layers):
            h = layer(torch.cat([h, self.aggregate(h)], dim=-1))
            if i < len(self.spatial_layers) - 1:
                h = torch.relu(h)
        return h

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, edge_index=None) -> torch.Tensor:
        """Node logits ``[B, N, 26]``, with the per-node physics fusion applied."""
        logits = super().forward(x, edge_index)
        if not self.use_fusion:
            return logits
        rho, _, _, _ = self._rho(x)
        return logits - self.beta * rho[:, None, :]

    def forward_pc(self, x, x_clean=None, drop_mask=None) -> dict:
        """Everything the PC objective needs, from one pass.

        ``x_clean`` and ``drop_mask`` are the pre-masking batch and the dropped
        channels, available at training time from ``DropoutAugmentedLoader``.
        They feed ``rho(D)`` -- the leave-k-out consistency term -- and are never
        needed at inference.
        """
        h = self._hidden(x)
        logits = self.classifier(h)
        out: dict[str, torch.Tensor] = {}

        rho_a = hyp = None
        if self.phys_table.numel():
            rho_a, hyp, _, _ = self._rho(x)
            out["rho_A"] = rho_a
            out["hyp_star"] = hyp

        if self.use_fusion:
            logits = logits - self.beta * rho_a[:, None, :]

        out["logits"] = logits
        scores, _ = soft_vote(logits)
        out["p_graph"] = scores / scores.sum(dim=-1, keepdim=True)

        if x_clean is not None and drop_mask is not None and rho_a is not None:
            with torch.no_grad():
                s_clean, _ = observe(x_clean, self.norm_mean, self.norm_std, self.obs_reduce)
                out["rho_D"] = phys.rho_heldout(s_clean, drop_mask, self.phys_table,
                                                hyp, self.sigma_eps, self.sigma0)

        if self.use_type_head:
            out["type_logits"] = self.type_head(h)
        if self.use_decoder:
            out["recon"] = self.decoder(h)
        return out
