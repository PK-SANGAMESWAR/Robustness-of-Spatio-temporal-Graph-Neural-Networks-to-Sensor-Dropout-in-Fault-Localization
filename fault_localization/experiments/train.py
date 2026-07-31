from __future__ import annotations
import math,time
import torch
from fault_localization.models.common import expanded_cross_entropy
from .evaluate import evaluate_loader
def train_epoch(model,loader,optimizer,device,grad_clip_norm=None):
    model.train(); total=0.; count=0; started=time.perf_counter()
    for batch in loader:
        x,y=batch["x"].to(device, non_blocking=True),batch["y"].to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True); logits=model(x); loss=expanded_cross_entropy(logits,y)
        if not math.isfinite(loss.item()): raise FloatingPointError("Non-finite training loss")
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step(); total+=loss.item()*len(y); count+=len(y)
    return {"loss":total/max(count,1),"seconds":time.perf_counter()-started}
def validate_epoch(model,loader,device,average): return evaluate_loader(model,loader,device,average)
