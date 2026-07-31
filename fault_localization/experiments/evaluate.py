from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from fault_localization.models.voting import soft_vote

def metric_bundle(truth, prediction, average="macro"):
    labels=list(range(26)); p,r,f,s=precision_recall_fscore_support(truth,prediction,labels=labels,average=average,zero_division=0)
    per=precision_recall_fscore_support(truth,prediction,labels=labels,average=None,zero_division=0)
    return {"accuracy":float(accuracy_score(truth,prediction)),"precision":float(p),"recall":float(r),"f1":float(f),"per_class":{"precision":per[0].tolist(),"recall":per[1].tolist(),"f1":per[2].tolist(),"support":per[3].tolist()},"confusion_matrix":confusion_matrix(truth,prediction,labels=labels).tolist()}

def evaluate_loader(model, loader, device, average="macro", records=False):
    import torch
    model.eval(); graph_y=[];graph_p=[];node_y=[];node_p=[]; output=[]
    with torch.no_grad():
        for batch in loader:
            x,y=batch["x"].to(device, non_blocking=True),batch["y"].to(device, non_blocking=True); logits=model(x); _,votes=soft_vote(logits); nodes=logits.argmax(-1)
            graph_y.extend(y.cpu().tolist());graph_p.extend(votes.cpu().tolist());node_y.extend(y[:,None].expand_as(nodes).reshape(-1).cpu().tolist());node_p.extend(nodes.reshape(-1).cpu().tolist())
            if records:
                for i,wid in enumerate(batch["window_id"]): output.append({"window_id":wid,"truth":int(y[i]),"vote_prediction":int(votes[i]),"node_predictions":nodes[i].cpu().tolist(),**batch["metadata"][i]})
    return {"graph":metric_bundle(graph_y,graph_p,average),"node":metric_bundle(node_y,node_p,average),"predictions":output}
