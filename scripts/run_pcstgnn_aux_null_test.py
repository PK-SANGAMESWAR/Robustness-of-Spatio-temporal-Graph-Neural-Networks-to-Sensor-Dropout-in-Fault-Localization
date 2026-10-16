"""S4.3d null test (06_STAGE_E_pcstgnn.md): the auxiliary network-only loss at gamma = 0 must leave
the training objective exactly as it was.

Two phases, run on either side of the code change:

    record   -- run with the OLD code. Builds the S4.3c setup (config/pcstgnn_noisediag/lam1_0p0.yaml,
                seed 42), draws one fixed training batch (noise + dropout augmentation on), and saves
                the batch, the initial weights, the total loss and every parameter gradient. The
                step is computed twice to confirm it is reproducible at all.
    compare  -- run with the NEW code. Loads the same batch and weights, computes the same step with
                aux_net_loss_weight = 0, and compares loss and gradients (tolerance 1e-7).

The step is ``pc_train_epoch`` over a one-batch loader with an lr = 0 SGD optimizer and no grad
clipping, so the real training function is exercised and no parameter moves (no training).

Usage:
    uv run python scripts/run_pcstgnn_aux_null_test.py record
    uv run python scripts/run_pcstgnn_aux_null_test.py compare
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fault_localization.experiments import runner  # noqa: E402  (sets CUBLAS env before torch)
import torch  # noqa: E402

from fault_localization.data.dataset import WindowGraphDataset, make_dataloader  # noqa: E402
from fault_localization.dropout.training import PhysicsDropoutLoader  # noqa: E402
from fault_localization.experiments.config import load_config, validate_config  # noqa: E402
from fault_localization.experiments.seed import seed_everything  # noqa: E402
from fault_localization.experiments.train import pc_train_epoch  # noqa: E402

SEED = 42
CONFIG = REPO / "config" / "pcstgnn_noisediag" / "lam1_0p0.yaml"
OUT = REPO / "results" / "pcstgnn_auxdiag" / "null_test"
TOL = 1e-7


def _setup():
    cfg = validate_config(load_config(str(CONFIG)))
    seed_everything(SEED, True)
    train_ds = WindowGraphDataset(Path(cfg["data_root"]), "train")
    model = runner._model(cfg, train_ds.edge_index).to(cfg["device"])
    return cfg, train_ds, model


def _step(model, batch, cfg, **extra):
    """One pc_train_epoch step on a fixed batch; returns (loss, {name: grad})."""
    torch.manual_seed(SEED)  # same classifier-dropout mask in every call
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    pc = cfg["pc_loss"]
    out = pc_train_epoch(model, [batch], opt, torch.device(cfg["device"]), grad_clip_norm=None,
                         lam1=float(pc["lambda1"]), lam2=float(pc["lambda2"]), **extra)
    grads = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()
             if p.grad is not None}
    return out["loss"], grads


def _diff(loss_a, grads_a, loss_b, grads_b):
    per = {}
    for n in sorted(set(grads_a) | set(grads_b)):
        if n not in grads_a or n not in grads_b:
            per[n] = float("inf")
        else:
            per[n] = float((grads_a[n] - grads_b[n]).abs().max())
    return abs(loss_a - loss_b), per


def record() -> int:
    if "aux_net_loss_weight" in inspect.signature(pc_train_epoch).parameters:
        print("record must run on the OLD code (pc_train_epoch already has aux_net_loss_weight)")
        return 1
    cfg, train_ds, model = _setup()
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    noise = cfg["noise_augmentation"]
    loader = PhysicsDropoutLoader(
        make_dataloader(train_ds, cfg["training"]["batch_size"], shuffle=True, seed=SEED,
                        num_workers=0),
        k_range=cfg["dropout_study"]["train_k_range"], rng_seed=SEED,
        policy=cfg["dropout_study"]["policy"],
        noise_sigmas=list(noise["sigma_grid"]), norm_std=train_ds.std)
    batch = next(iter(loader))
    batch = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}

    loss1, g1 = _step(model, batch, cfg)
    model.load_state_dict(init_state)
    loss2, g2 = _step(model, batch, cfg)
    dl, per = _diff(loss1, g1, loss2, g2)
    print(f"old-vs-old reproducibility: |dloss|={dl:.3e}  max|dgrad|={max(per.values()):.3e}")

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"batch": batch, "init_state": init_state, "loss": loss1, "grads": g1},
               OUT / "old_path.pt")
    sig = batch.get("sigma0")
    print(f"batch: B={len(batch['y'])} dropout_k={batch.get('dropout_k')} "
          f"sigma0 levels={sorted(set(sig.tolist())) if sig is not None else None}")
    print(f"old loss={loss1:.10f}, {len(g1)} parameter grads -> {OUT / 'old_path.pt'}")
    return 0 if dl == 0 and max(per.values()) == 0 else 2


def compare() -> int:
    if "aux_net_loss_weight" not in inspect.signature(pc_train_epoch).parameters:
        print("compare must run on the NEW code (pc_train_epoch has no aux_net_loss_weight)")
        return 1
    old = torch.load(OUT / "old_path.pt", weights_only=False)
    cfg, _, model = _setup()
    model.load_state_dict(old["init_state"])
    loss, grads = _step(model, old["batch"], cfg, aux_net_loss_weight=0.0)
    dl, per = _diff(old["loss"], old["grads"], loss, grads)
    worst = max(per, key=per.get)
    passed = dl <= TOL and per[worst] <= TOL and set(grads) == set(old["grads"])
    result = {"tolerance": TOL, "old_loss": old["loss"], "new_loss_gamma0": loss,
              "abs_loss_diff": dl, "n_param_grads": len(grads),
              "max_abs_grad_diff": per[worst], "worst_param": worst,
              "per_param_max_abs_grad_diff": per, "passed": passed}
    (OUT / "null_test_result.json").write_text(json.dumps(result, indent=2))
    print(f"old loss={old['loss']:.10f}  new loss (gamma=0)={loss:.10f}  |d|={dl:.3e}")
    print(f"{len(grads)} grads, max|dgrad|={per[worst]:.3e} ({worst})")
    print("NULL TEST", "PASSED" if passed else "FAILED")
    return 0 if passed else 3


if __name__ == "__main__":
    sys.exit({"record": record, "compare": compare}[sys.argv[1]]())
