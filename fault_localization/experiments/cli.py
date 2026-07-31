from __future__ import annotations
import argparse
from pathlib import Path
from fault_localization.data.manifest import create_manifest,write_manifest
from fault_localization.validation.dataset_checks import strict_benchmark_validation
from .config import load_config,validate_config
from .runner import run
from .reporting import report
def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True)
    for name in ("validate-benchmark","train"):
        q=sub.add_parser(name);q.add_argument("--config",required=True);q.add_argument("--seed",type=int)
        if name=="train":
            q.add_argument("--data-root",help="Override config data_root (e.g. for scaled-dataset trend runs)")
            q.add_argument("--benchmark-version",help="Override config benchmark_version")
            q.add_argument("--disable-edges",action="store_true",help="Ablation: construct the GNN with an empty edge_index (edges-off probe)")
            q.add_argument("--epochs",type=int,help="Diagnostic override: total training epochs (underfitting probe)")
            q.add_argument("--gnn-layers",type=int,help="Diagnostic override: stacked spatial hops for RGSAGE (multi-hop-depth probe)")
            q.add_argument("--grad-clip-norm",type=float,help="Diagnostic override: clip gradient norm to this value each step (training-stability probe)")
            q.add_argument("--warmup-epochs",type=int,help="Diagnostic override: linearly warm up LR from 0 over this many epochs (training-stability probe)")
    q=sub.add_parser("report");q.add_argument("--benchmark",required=True)
    a=p.parse_args()
    if a.command=="report": report(Path("results")/a.benchmark);return
    c=validate_config(load_config(a.config));
    if a.seed is not None:c["seed"]=a.seed
    if a.command=="train":
        if a.data_root:c["data_root"]=a.data_root
        if a.benchmark_version:c["benchmark_version"]=a.benchmark_version
        if a.disable_edges:c.setdefault("ablation",{})["disable_edges"]=True
        if a.epochs is not None:c["training"]["epochs"]=a.epochs
        if a.gnn_layers is not None:c["gnn_layers"]=a.gnn_layers
        if a.grad_clip_norm is not None:c["training"]["grad_clip_norm"]=a.grad_clip_norm
        if a.warmup_epochs is not None:c["training"]["warmup_epochs"]=a.warmup_epochs
    if a.command=="validate-benchmark":
        summary = strict_benchmark_validation(Path(c["data_root"]))
        print({**summary, "fingerprint": create_manifest(c["data_root"],c["benchmark_version"])["fingerprint"]})
    else: print(run(c))
if __name__=="__main__": main()
