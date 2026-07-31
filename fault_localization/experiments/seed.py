from __future__ import annotations
import os, random
import numpy as np
def seed_everything(seed:int, deterministic:bool=True):
    import torch
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"]=str(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True); return {"seed":seed,"deterministic_algorithms":deterministic,"cuda_available":torch.cuda.is_available()}
