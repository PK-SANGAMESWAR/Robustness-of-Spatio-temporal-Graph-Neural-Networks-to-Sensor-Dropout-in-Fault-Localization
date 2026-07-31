from __future__ import annotations
import copy
from pathlib import Path
import yaml

def load_config(path: str | Path) -> dict:
    path=Path(path); config=yaml.safe_load(path.read_text()) or {}
    base=config.pop("base", None)
    if base:
        parent=load_config(path.parent/base)
        return _merge(parent,config)
    return config
def _merge(a,b):
    out=copy.deepcopy(a)
    for k,v in b.items(): out[k]=_merge(out[k],v) if isinstance(v,dict) and isinstance(out.get(k),dict) else v
    return out
def validate_config(c):
    for key in ("benchmark_version","data_root","output_root","model","training","seed","device"):
        if key not in c: raise ValueError(f"Missing config field: {key}")
    if c["training"]["checkpoint_metric"] != "graph_f1": raise ValueError("Checkpoint selection must use graph_f1")
    if c["metrics"]["f1_average"] not in {"macro","weighted","micro"}: raise ValueError("Invalid F1 averaging")
    return c
