from __future__ import annotations
import math,time
import torch
from fault_localization.models.common import (
    expanded_cross_entropy, pc_loss, type_targets_from_metadata, unlabelable_window_mask,
)
from fault_localization.models.physics_loss import physics_consistency_loss
from fault_localization.physics.observation import observe
from .evaluate import evaluate_loader
def train_epoch(model,loader,optimizer,device,grad_clip_norm=None,physics=None):
    """One training epoch.

    ``physics`` is the P3 arm (Type A, physics in the loss).  When None — the
    default and the behaviour of every run in ``results/`` — the objective is
    exactly the cross-entropy it has always been.  When given it is a dict with
    ``lam`` (float) and ``signatures`` ([25, N] tensor), and the objective
    becomes ``CE + lam * L_phys``.  ``phys_loss`` is reported separately so the
    term can be inspected per epoch rather than inferred from the total.
    """
    model.train(); total=0.; phys_total=0.; count=0; started=time.perf_counter()
    sig=None
    if physics is not None:
        sig=physics["signatures"].to(device)
    for batch in loader:
        x,y=batch["x"].to(device, non_blocking=True),batch["y"].to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True); logits=model(x); loss=expanded_cross_entropy(logits,y)
        if physics is not None:
            p=physics_consistency_loss(logits,x,sig)
            phys_total+=p.item()*len(y)
            loss=loss+physics["lam"]*p
        if not math.isfinite(loss.item()): raise FloatingPointError("Non-finite training loss")
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step(); total+=loss.item()*len(y); count+=len(y)
    out={"loss":total/max(count,1),"seconds":time.perf_counter()-started}
    if physics is not None: out["phys_loss"]=phys_total/max(count,1)
    return out
def validate_epoch(model,loader,device,average): return evaluate_loader(model,loader,device,average)
def pc_train_epoch(model,loader,optimizer,device,grad_clip_norm=None,lam1=0.0,lam2=0.0,rho_k0_cache=None,
                   aux_net_loss_weight=0.0,type_loss_weight=0.0,recon_loss_weight=0.0):
    """PC-STGNN training epoch: ``CE(z',y) + lam1*E_P[rho(A)] + lam2*E_P[rho(D)]``
    (``PC-STGNN/01_SPEC.md`` Sec 6, ``models/common.py::pc_loss``).

    ``loader`` must yield ``x`` (post-dropout), ``x_clean``, ``drop_mask``,
    ``window_id`` and ``dropout_k`` alongside ``y`` -- i.e. a
    ``dropout.training.PhysicsDropoutLoader``, not a plain ``DropoutAugmentedLoader``,
    since the lambda_2 term needs the pre-dropout batch and which channels were
    synthetically zeroed.

    ``rho_k0_cache``, if given, is ``(window_to_row, rho0[W,26], hyp0[W,26])``
    from ``physics.cache.load_rho_k0_lookup``. At k=0 (every window in the batch
    undropped) rho(A) is a fixed function of the window alone, so it is looked up
    rather than recomputed live -- verified equivalent to the live path before use
    (see ``scripts/verify_pcstgnn_rho_cache.py``). k>0 batches are unaffected and
    always take the live path: their masks are fresh per (epoch, batch), so no
    cache built from a fixed set of evaluation-protocol mask seeds can serve them.

    ``aux_net_loss_weight`` (gamma, S4.3d) adds ``gamma * CE(network-only logits, y)``;
    at 0 (default) the objective is unchanged. The unweighted aux CE is reported as ``aux``.

    ``type_loss_weight`` (R1) and ``recon_loss_weight`` (R2) are both 0 by default, so
    every arm without ``use_type_head``/``use_decoder`` is unaffected -- see
    ``models/common.py::pc_loss``. ``batch["metadata"]`` supplies the type label
    (:func:`models.common.type_targets_from_metadata`); ``x_clean``/``drop_mask``
    (already loaded above for lambda_2) supply the reconstruction target/mask.
    """
    model.train(); total=0.; l1_total=0.; l2_total=0.; aux_total=0.; type_total=0.; recon_total=0.
    count=0; started=time.perf_counter()
    for batch in loader:
        x=batch["x"].to(device, non_blocking=True)
        x_clean=batch["x_clean"].to(device, non_blocking=True)
        drop_mask=batch["drop_mask"].to(device, non_blocking=True)
        y=batch["y"].to(device, non_blocking=True)
        loss_mask=unlabelable_window_mask(batch["window_id"]).to(device)
        optimizer.zero_grad(set_to_none=True)
        sigma0=batch.get("sigma0")
        noisy=sigma0 is not None and bool((sigma0>0).any())
        if sigma0 is not None: sigma0=sigma0.to(device, non_blocking=True)
        rho_override=hyp_override=None
        # The k=0 cache was built noise-free: it is exact only for undropped, un-noised windows.
        if rho_k0_cache is not None and batch.get("dropout_k", -1) == 0 and not noisy:
            window_to_row, rho0, hyp0 = rho_k0_cache
            rows=[window_to_row[wid] for wid in batch["window_id"]]
            rho_override=torch.as_tensor(rho0[rows], device=device, dtype=x.dtype)
            hyp_override=torch.as_tensor(hyp0[rows], device=device, dtype=torch.long)
        out=model.forward_pc(x, x_clean, drop_mask, sigma0=sigma0,
                             rho_a_override=rho_override, hyp_override=hyp_override)
        type_targets=recon_target=recon_mask=None
        if type_loss_weight:
            type_targets=type_targets_from_metadata(batch["y"], batch["metadata"]).to(device)
        if recon_loss_weight:
            with torch.no_grad():
                recon_target, _ = observe(x_clean, model.norm_mean, model.norm_std, model.obs_reduce)
            recon_mask=drop_mask
        parts=pc_loss(out, y, lam1=lam1, lam2=lam2, loss_mask=loss_mask,
                      aux_net_weight=aux_net_loss_weight,
                      type_loss_weight=type_loss_weight, type_targets=type_targets,
                      recon_loss_weight=recon_loss_weight, recon_target=recon_target, recon_mask=recon_mask)
        loss=parts["total"]
        if not math.isfinite(loss.item()): raise FloatingPointError("Non-finite training loss")
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step(); total+=loss.item()*len(y); count+=len(y)
        if "l1" in parts: l1_total+=parts["l1"].item()*len(y)
        if "l2" in parts: l2_total+=parts["l2"].item()*len(y)
        if "aux" in parts: aux_total+=parts["aux"].item()*len(y)
        if "type_ce" in parts: type_total+=parts["type_ce"].item()*len(y)
        if "recon_mse" in parts: recon_total+=parts["recon_mse"].item()*len(y)
    out={"loss":total/max(count,1),"seconds":time.perf_counter()-started}
    if lam1: out["l1"]=l1_total/max(count,1)
    if lam2: out["l2"]=l2_total/max(count,1)
    if aux_net_loss_weight: out["aux"]=aux_total/max(count,1)
    if type_loss_weight: out["type_ce"]=type_total/max(count,1)
    if recon_loss_weight: out["recon_mse"]=recon_total/max(count,1)
    return out
