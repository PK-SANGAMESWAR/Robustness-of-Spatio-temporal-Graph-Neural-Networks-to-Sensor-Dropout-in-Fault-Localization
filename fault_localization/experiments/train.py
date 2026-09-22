from __future__ import annotations
import math,time
import torch
from fault_localization.models.common import expanded_cross_entropy
from fault_localization.models.physics_loss import physics_consistency_loss
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
