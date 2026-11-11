"""Training epoch for B2a (learned R-hat head). New file; ``experiments/train.py`` is not edited.

Mirrors ``experiments/train.py::pc_train_epoch`` for the settings of arm R5 (lambda1 = 0, lambda2 = 0.1, gamma = 1) and
adds the R-hat term (``models/pc_stgnn_rhat.py::b2a_batch_loss``).  Differences from ``pc_train_epoch``, all
deliberate: no k = 0 rho cache (it holds the *profiled* rho and would silently put the profiled R back), no type or
decoder terms, and the true resistance of each window goes into rho while training.

It is called from ``experiments/runner.py`` only after the patch ``patches/stage_g/b2a_runner_hooks.patch`` is applied
(the Stage G queue applies it).
"""
from __future__ import annotations

import math
import time

import torch

from fault_localization.models.pc_stgnn_rhat import b2a_batch_loss


def b2a_train_epoch(model, loader, optimizer, device, grad_clip_norm=None, lam1=0.0, lam2=0.0,
                    aux_net_loss_weight=0.0, rhat_loss_weight=1.0, rho_k0_cache=None) -> dict:
    """One epoch; returns ``{"loss", "seconds"}`` plus ``l1``/``l2``/``aux``/``rhat_mse`` means when those terms are on."""
    if rho_k0_cache is not None:
        raise ValueError("B2a must not use the profiled k = 0 rho cache")
    model.train()
    sums = {"loss": 0.0, "l1": 0.0, "l2": 0.0, "aux": 0.0, "rhat_mse": 0.0}
    count = 0
    started = time.perf_counter()
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        parts = b2a_batch_loss(model, batch, device, lam1=lam1, lam2=lam2,
                               aux_net_weight=aux_net_loss_weight, rhat_weight=rhat_loss_weight)
        loss = parts["total"]
        if not math.isfinite(loss.item()):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        n = len(batch["y"])
        sums["loss"] += loss.item() * n
        for key in ("l1", "l2", "aux", "rhat_mse"):
            if key in parts:
                sums[key] += parts[key].item() * n
        count += n
    out = {"loss": sums["loss"] / max(count, 1), "seconds": time.perf_counter() - started}
    if lam1:
        out["l1"] = sums["l1"] / max(count, 1)
    if lam2:
        out["l2"] = sums["l2"] / max(count, 1)
    if aux_net_loss_weight:
        out["aux"] = sums["aux"] / max(count, 1)
    out["rhat_mse"] = sums["rhat_mse"] / max(count, 1)
    return out
