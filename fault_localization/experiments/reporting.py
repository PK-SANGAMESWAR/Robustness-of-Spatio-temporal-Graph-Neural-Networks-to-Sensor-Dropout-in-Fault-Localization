from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
def report(benchmark_dir):
    root=Path(benchmark_dir); rows=[]
    for p in root.glob("*/seed_*/metrics.json"):
        d=json.loads(p.read_text()); rows.append({"model":p.parents[1].name,"seed":p.parent.name,"graph_f1":d["test_graph"]["f1"],"precision":d["test_graph"]["precision"],"recall":d["test_graph"]["recall"],"node_f1_diagnostic":d["test_node_diagnostic"]["f1"]})
    frame=pd.DataFrame(rows); summaries=[]
    for model,g in frame.groupby("model"):
        n=len(g); mean=g.graph_f1.mean(); sd=g.graph_f1.std(ddof=1) if n>1 else float("nan"); ci=t.ppf(.95,n-1)*sd/math.sqrt(n) if n>=3 else float("nan"); summaries.append({"model":model,"n":n,"graph_f1_mean":mean,"graph_f1_sample_sd":sd,"graph_f1_90ci_half_width":ci,"ci_formula":"mean ± t_(0.95,n-1)*s/sqrt(n)"})
    frame.to_csv(root/"benchmark_comparison.csv",index=False); pd.DataFrame(summaries).to_csv(root/"benchmark_summary.csv",index=False); return frame
