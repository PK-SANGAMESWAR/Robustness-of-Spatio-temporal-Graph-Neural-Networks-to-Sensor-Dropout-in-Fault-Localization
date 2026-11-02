"""B2a -- PC-STGNN with a learned resistance head instead of profiling over R.

``PC-STGNN/01_SPEC.md`` section 5: "A learned R-hat head is not part of the model. It is ablation
B2a, run once to show what a regressed resistance costs on OOD R."  Everything else equals R5:
the network predicts location (z), the fixed table gives rho, ``z' = z - beta * rho``; only rho is
computed on the table slice at one R-grid point per window rather than minimised over R.

Per the 2026-11-02 amendment (``14_PREREGISTRATION.md``, B2a):

* head: ``nn.Linear(64, 1)`` on the node embedding mean-pooled over the 25 nodes; target log10 R;
* training: rho uses the **true** R, snapped to the nearest R-grid point (the corpus R in
  {0.1, 1, 10} is on the grid); evaluation: rho uses the **regressed** R-hat, snapped the same way;
* the head learns only from its own MSE term (rho is computed without gradient);
* the head is created last, so the backbone draws are those of ``PCSTGNN`` (null test N1).

New file.  ``models/pc_stgnn.py`` is not edited (the running queue imports it).  Training this
arm additionally needs a model-name branch in ``experiments/runner.py::_model`` and a call of
:func:`b2a_batch_loss` in the training epoch; both are deferred until the queue has finished
(see ``PC-STGNN/stage_g_plan_draft.md``).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from fault_localization.models.common import pc_loss, unlabelable_window_mask
from fault_localization.models.pc_stgnn import PCSTGNN
from fault_localization.models.voting import soft_vote
from fault_localization.physics import residual as phys
from fault_localization.physics.grids import NUM_R, R_GRID
from fault_localization.physics.observation import observe
from fault_localization.physics.residual_sliced import rho_sliced


class PCSTGNNRHat(PCSTGNN):
    def __init__(self, edge_index, table=None, norm_stats=None, *, use_rhat_head: bool = True, **kwargs):
        super().__init__(edge_index, table, norm_stats, **kwargs)
        self.use_rhat_head = bool(use_rhat_head)
        self.register_buffer("log10_r_grid", torch.log10(torch.as_tensor(np.asarray(R_GRID), dtype=torch.float32)),
                             persistent=False)
        if self.use_rhat_head:
            self.rhat_head = nn.Linear(self.classifier.linear.in_features, 1)   # created last

    # -- R-hat and the slice index -------------------------------------------------------

    def rhat_from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        """``h [B,25,64]`` -> log10 R-hat ``[B]``."""
        return self.rhat_head(h.mean(dim=1)).squeeze(-1)

    def snap_index(self, log10_r: torch.Tensor) -> torch.Tensor:
        """Nearest R-grid index (log distance) for each value ``[B]``."""
        grid = self.log10_r_grid.to(log10_r.device)
        return (log10_r[:, None] - grid[None, :]).abs().argmin(dim=-1)

    def _rho_at(self, x, r_idx, sigma0=None):
        with torch.no_grad():
            s, a = observe(x, self.norm_mean, self.norm_std, self.obs_reduce)
            rho, hyp = rho_sliced(s, a, self.phys_table, r_idx, NUM_R, self.sigma_eps,
                                  self.sigma0 if sigma0 is None else sigma0, chunk_f=self.chunk_f)
        return rho, hyp, s, a

    def _rho(self, x, sigma0=None):
        """Evaluation path used by the scorers: rho on the R-hat slice. ``(rho, hyp, s, a)``."""
        with torch.no_grad():
            r_idx = self.snap_index(self.rhat_from_hidden(self._hidden(x)))
        return self._rho_at(x, r_idx, sigma0)

    def forward(self, x, edge_index=None, r_idx=None):
        """Node logits ``[B,N,26]``; fusion uses the R-hat slice, or ``r_idx`` if given (oracle checks)."""
        logits = RGSAGE_forward(self, x, edge_index)
        if not self.use_fusion:
            return logits
        if r_idx is None:
            rho, _, _, _ = self._rho(x)
        else:
            rho, _, _, _ = self._rho_at(x, r_idx)
        return logits - self.beta * rho[:, None, :]

    # -- training pass ------------------------------------------------------------------

    def forward_pc(self, x, x_clean=None, drop_mask=None, *, r_true=None, use_true_r=None,
                   rho_a_override=None, hyp_override=None, sigma0=None) -> dict:
        """As ``PCSTGNN.forward_pc`` plus ``rhat_log10``. ``r_true`` is the event resistance in ohm [B].

        ``use_true_r`` defaults to ``self.training``: true R in rho while training, R-hat otherwise.
        A cached profiled rho is refused: it would silently put the profiled R back.
        """
        if rho_a_override is not None or hyp_override is not None:
            raise ValueError("B2a computes rho on the R slice; a cached profiled rho must not be passed")
        h = self._hidden(x)
        logits = self.classifier(h)
        rhat = self.rhat_from_hidden(h)
        use_true = self.training if use_true_r is None else use_true_r
        if use_true:
            if r_true is None:
                raise ValueError("training-time rho needs the true resistance r_true [B] (ohm)")
            r_idx = self.snap_index(torch.log10(r_true.to(rhat.dtype).to(rhat.device)))
        else:
            r_idx = self.snap_index(rhat.detach())
        rho_a, hyp, _, _ = self._rho_at(x, r_idx, sigma0)
        out = {"rho_A": rho_a, "hyp_star": hyp, "net_logits": logits, "rhat_log10": rhat, "r_idx": r_idx}
        if self.use_fusion:
            logits = logits - self.beta * rho_a[:, None, :]
        out["logits"] = logits
        scores, _ = soft_vote(logits)
        out["p_graph"] = scores / scores.sum(dim=-1, keepdim=True)
        if x_clean is not None and drop_mask is not None:
            with torch.no_grad():
                s_clean, _ = observe(x_clean, self.norm_mean, self.norm_std, self.obs_reduce)
                out["rho_D"] = phys.rho_heldout(s_clean, drop_mask, self.phys_table, hyp, self.sigma_eps,
                                                self.sigma0 if sigma0 is None else sigma0)
        return out


def RGSAGE_forward(model, x, edge_index=None):
    """The learned backbone's node logits (``RGSAGE.forward``), without any physics term."""
    from fault_localization.models.graphsage import RGSAGE
    return RGSAGE.forward(model, x, edge_index)


# ---------------------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------------------


def rhat_targets_from_metadata(window_labels: torch.Tensor, metadata) -> tuple[torch.Tensor, torch.Tensor]:
    """``(log10 R [B], valid [B])``; ``valid`` is False for no-fault windows (label 0)."""
    r = torch.tensor([float(md["resistance"]) for md in metadata], dtype=torch.float32)
    valid = window_labels.detach().cpu() != 0
    return torch.log10(r), valid


def rhat_loss(rhat_log10: torch.Tensor, target_log10: torch.Tensor, valid: torch.Tensor,
              loss_mask: torch.Tensor | None = None) -> torch.Tensor:
    """MSE on log10 R over fault windows only; ``loss_mask`` also drops w = 20 windows. 0 if none valid."""
    m = valid.to(rhat_log10.device)
    if loss_mask is not None:
        m = m & loss_mask.to(rhat_log10.device).bool()
    if not bool(m.any()):
        return rhat_log10.sum() * 0.0          # keeps the graph; zero gradient
    d = rhat_log10[m] - target_log10.to(rhat_log10.device)[m]
    return (d * d).mean()


def b2a_loss(out: dict, targets: torch.Tensor, *, rhat_target: torch.Tensor, rhat_valid: torch.Tensor,
             rhat_weight: float = 1.0, loss_mask: torch.Tensor | None = None, **pc_kwargs) -> dict:
    """``pc_loss(...) + rhat_weight * MSE(log10 R-hat, log10 R)`` (R5 terms unchanged)."""
    parts = pc_loss(out, targets, loss_mask=loss_mask, **pc_kwargs)
    term = rhat_loss(out["rhat_log10"], rhat_target, rhat_valid, loss_mask)
    parts["rhat_mse"] = term.detach().clone()
    if rhat_weight:
        parts["total"] = parts["total"] + rhat_weight * term
    return parts


def b2a_batch_loss(model: PCSTGNNRHat, batch: dict, device, *, lam1=0.0, lam2=0.1, aux_net_weight=1.0,
                   rhat_weight=1.0) -> dict:
    """One training batch with R5 settings (lambda1 = 0, lambda2 = 0.1, gamma = 1) plus the R-hat term.

    ``batch`` as yielded by ``dropout.training.PhysicsDropoutLoader``: ``x, x_clean, drop_mask, y,
    window_id, metadata`` (and optionally ``sigma0``).
    """
    x = batch["x"].to(device)
    x_clean = batch["x_clean"].to(device)
    drop_mask = batch["drop_mask"].to(device)
    y = batch["y"].to(device)
    loss_mask = unlabelable_window_mask(batch["window_id"]).to(device)
    sigma0 = batch.get("sigma0")
    if sigma0 is not None:
        sigma0 = sigma0.to(device)
    target, valid = rhat_targets_from_metadata(batch["y"], batch["metadata"])
    r_true = torch.tensor([float(md["resistance"]) for md in batch["metadata"]], device=device)
    out = model.forward_pc(x, x_clean, drop_mask, r_true=r_true, sigma0=sigma0)
    return b2a_loss(out, y, rhat_target=target.to(device), rhat_valid=valid.to(device),
                    rhat_weight=rhat_weight, loss_mask=loss_mask, lam1=lam1, lam2=lam2,
                    aux_net_weight=aux_net_weight)
