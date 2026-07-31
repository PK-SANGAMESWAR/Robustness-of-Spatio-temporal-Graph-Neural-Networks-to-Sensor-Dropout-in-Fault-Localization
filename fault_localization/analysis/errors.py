"""Reproducible, descriptive error slices; this module never feeds a model."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def analyze_predictions(predictions_csv: str | Path, output_dir: str | Path) -> Path:
    data=pd.read_csv(predictions_csv); out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    errors=data[data.truth != data.vote_prediction].copy(); errors.to_csv(out/"representative_failures.csv",index=False)
    groups=[x for x in ("fault_bus","fault_type","resistance","load_multiplier") if x in data]
    rows=[]
    for group in groups:
        for value,part in data.groupby(group): rows.append({"slice":group,"value":value,"n":len(part),"accuracy":float((part.truth==part.vote_prediction).mean())})
    pd.DataFrame(rows).to_csv(out/"error_slices.csv",index=False)
    (out/"error_analysis.md").write_text("# Error analysis\n\nResults are descriptive only; they do not establish electrical causality.\n")
    return out
