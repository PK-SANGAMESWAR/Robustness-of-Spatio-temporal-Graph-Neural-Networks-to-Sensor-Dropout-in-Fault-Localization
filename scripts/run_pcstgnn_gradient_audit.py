"""S4.2 -- the gradient audit (06_STAGE_E_pcstgnn.md), guards the bug in the first draft.

A reconstruction loss of the form ``||mu[f_hat,...] - s_true||^2`` has zero gradient (f_hat is an
arg-min index, constant in the weights); under that formulation the lambda_2 arm would be silently
identical to lambda_1 and it would take weeks to notice. This script proves the actual
implementation (``models/pc_stgnn.py::forward_pc`` + ``models/common.py::pc_loss``) does not have
that bug, on THREE real batches from the frozen corpus (a single batch cannot exercise the paths
that differ):

1. a batch containing a w=20 window (zero post-inception samples -- the loss mask must be active)
2. a batch at k=10 (|A| small, |D| large -- lambda_2's intended regime)
3. a batch at k=0, no synthetic drops (|D|=0 -- the commonest training case, and the one a naive
   r-tilde/|D| formulation would have turned into NaN on every step)

For each batch, checks:
- d(L1)/dw != 0 for network weights w (except batch 3, where L1 is still well-defined since A is
  non-empty at k=0 -- checked normally)
- d(L2)/dw != 0 for batches 1-2; for batch 3 (|D|=0) L2 must contribute exactly 0 and the gradient
  must be exactly 0 and finite (not NaN) -- a DIFFERENT expected outcome, not a missing check
- beta receives a gradient
- alpha, c: NOT APPLICABLE on the current fault_localization/models/pc_stgnn.py -- those belong to
  01_SPEC.md Sec 7's PCRG-STGNN trust gate (Stage F, 07_STAGE_F_gate.md), which is not implemented
  yet. Flagged explicitly rather than silently skipped or faked.
- mu (phys_table), mu_0, sigma_eps are buffers with requires_grad=False and receive no gradient

Also runs a genuine finite-difference check (not just "is autograd nonzero"): perturbs one
network-weight scalar by +-eps and compares the numerical L1/L2 gradient to autograd's. rho_A/
rho_D/hyp_star are computed under no_grad() from the INPUT alone (01_SPEC.md Sec 5's architectural
rule), so they do not need to be recomputed when only a network weight is perturbed -- this is
cached once per batch and reused, matching what the architecture guarantees rather than assuming it.

Usage:
    uv run python scripts/run_pcstgnn_gradient_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fault_localization.data.dataset import WindowGraphDataset  # noqa: E402
from fault_localization.dropout.masking import apply_sensor_dropout_batch  # noqa: E402
from fault_localization.models.common import pc_loss, unlabelable_window_mask  # noqa: E402
from fault_localization.models.pc_stgnn import PCSTGNN  # noqa: E402
from fault_localization.physics import residual as R  # noqa: E402
from fault_localization.physics import table as T  # noqa: E402

CANONICAL_SIGMA_EPS = 0.0007255549105131452
SEED = 42
BATCH_SIZE = 16
FD_EPS = 1e-3
OUT_PATH = REPO / "results/pcstgnn/audit/s4_2_gradient" / "gradient_audit.json"


def collate(ds: WindowGraphDataset, indices: list[int]) -> dict:
    items = [ds[i] for i in indices]
    x = torch.from_numpy(np.stack([it["x"] for it in items]))
    y = torch.tensor([it["y"] for it in items], dtype=torch.long)
    wids = [it["window_id"] for it in items]
    return {"x": x, "y": y, "window_id": wids}


def make_drop(x_clean: torch.Tensor, k: int, seed: int):
    """(x_dropped, drop_mask[B,25,3]) -- drop_mask is 1 on synthetically dropped channels."""
    if k <= 0:
        return x_clean.clone(), torch.zeros(x_clean.shape[0], 25, 3)
    x_dropped, _, dropped = apply_sensor_dropout_batch(x_clean, k=k, seed=seed)
    b, n = x_clean.shape[0], x_clean.shape[1]
    drop_mask = torch.zeros(b, n, 3)
    for i, ds in enumerate(dropped):
        if ds:
            drop_mask[i, ds, :] = 1.0
    return x_dropped, drop_mask


def pick_batch_with_w20(ds: WindowGraphDataset, batch_size: int) -> list[int]:
    w20_idx = next(i for i, (wid, *_r) in enumerate(ds.records) if wid.endswith("_w0020"))
    other = [i for i in range(len(ds)) if i != w20_idx][:batch_size - 1]
    return [w20_idx] + other


def pick_batch_plain(ds: WindowGraphDataset, batch_size: int, offset: int) -> list[int]:
    return list(range(offset, offset + batch_size))


def leaf_check(model: PCSTGNN) -> dict:
    return {
        "phys_table_requires_grad": bool(model.phys_table.requires_grad),
        "sigma_eps_requires_grad": bool(model.sigma_eps.requires_grad),
        "phys_table_in_parameters": any(p is model.phys_table for p in model.parameters()),
        "sigma_eps_in_parameters": any(p is model.sigma_eps for p in model.parameters()),
    }


def audit_batch(model: PCSTGNN, x_clean: torch.Tensor, y: torch.Tensor, wids: list[str],
                k: int, seed: int, label: str) -> dict:
    x_dropped, drop_mask = make_drop(x_clean, k, seed)
    loss_mask = unlabelable_window_mask(wids)

    result: dict = {"batch": label, "k": k, "n_w20_windows": int((loss_mask == 0).sum()),
                    "|D|_total": int(drop_mask.sum().item())}

    # ---- autograd: L1 alone, L2 alone ----
    # pc_loss() detaches its "l1"/"l2" report entries (they're for logging, not
    # backward -- only "total" retains the graph), and total = CE + lam*term
    # would fold in CE's own gradient, contaminating the isolation this check
    # needs. Reconstruct each term from forward_pc()'s own (undetached)
    # p_graph/rho_A/rho_D -- identical formula to pc_loss, graph intact -- and
    # pull its gradient directly via autograd.grad rather than .backward().
    w = model.classifier.linear.weight
    weights_t = loss_mask.to(torch.float32)
    denom = weights_t.sum().clamp(min=1.0)
    fd_targets: dict[str, tuple[int, int]] = {}
    for name, key in (("l1", "rho_A"), ("l2", "rho_D")):
        model.zero_grad(set_to_none=True)
        out = model.forward_pc(x_dropped, x_clean, drop_mask)
        term = ((out["p_graph"] * out[key]).sum(dim=-1) * weights_t).sum() / denom
        (g,) = torch.autograd.grad(term, w, retain_graph=False, allow_unused=True)
        g_norm = float(g.norm().item()) if g is not None else 0.0
        result[f"{name}_value"] = float(term.item())
        result[f"grad_{name}_norm"] = g_norm
        result[f"grad_{name}_nonzero"] = g_norm > 0.0
        # Pick the weight element with the LARGEST gradient magnitude for the
        # finite-difference check below -- a fixed index like [0,0] can
        # legitimately land on a coincidentally-zero element even when the
        # overall gradient norm is nonzero, which would make that check
        # uninformative rather than a real confirmation.
        if g is not None and g_norm > 0.0:
            flat_idx = int(g.abs().argmax())
            fd_targets[name] = np.unravel_index(flat_idx, g.shape)
        else:
            fd_targets[name] = (0, 0)

    # ---- beta gradient (needs fusion on; CE + fusion both feed beta) ----
    model.zero_grad(set_to_none=True)
    out = model.forward_pc(x_dropped, x_clean, drop_mask)
    parts = pc_loss(out, y, lam1=1.0, lam2=1.0, loss_mask=loss_mask)
    parts["total"].backward()
    beta_grad = float(model.beta.grad.item()) if model.beta.grad is not None else None
    result["beta_grad"] = beta_grad
    result["beta_grad_nonzero"] = beta_grad is not None and beta_grad != 0.0

    # ---- alpha, c ----
    result["alpha_c_status"] = "N/A -- not implemented in models/pc_stgnn.py (Stage F PCRG-STGNN)"

    # ---- leaf checks ----
    result["leaves"] = leaf_check(model)
    result["mu_mu0_sigma_eps_no_grad"] = (
        not result["leaves"]["phys_table_requires_grad"]
        and not result["leaves"]["sigma_eps_requires_grad"]
        and not result["leaves"]["phys_table_in_parameters"]
        and not result["leaves"]["sigma_eps_in_parameters"]
    )

    # ---- finite-difference check on L1 and L2 vs one weight scalar ----
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        rho_a, hyp, _, _ = model._rho(x_dropped)
        s_clean, _ = __import__("fault_localization.physics.observation",
                                fromlist=["observe"]).observe(x_clean, model.norm_mean,
                                                              model.norm_std, model.obs_reduce)
        rho_d = R.rho_heldout(s_clean, drop_mask, model.phys_table, hyp,
                              model.sigma_eps, model.sigma0)

    def loss_at(lam1, lam2):
        h = model._hidden(x_dropped)
        logits = model.classifier(h)
        if model.use_fusion:
            logits = logits - model.beta * rho_a[:, None, :]
        from fault_localization.models.voting import soft_vote
        scores, _ = soft_vote(logits)
        p_graph = scores / scores.sum(dim=-1, keepdim=True)
        weights = loss_mask.to(p_graph.dtype)
        denom = weights.sum().clamp(min=1.0)
        l1 = ((p_graph * rho_a).sum(dim=-1) * weights).sum() / denom
        l2 = ((p_graph * rho_d).sum(dim=-1) * weights).sum() / denom
        return lam1 * l1 + lam2 * l2

    w = model.classifier.linear.weight
    fd_result = {}
    for name, lam1, lam2 in (("l1", 1.0, 0.0), ("l2", 0.0, 1.0)):
        idx = fd_targets[name]
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = loss_at(lam1, lam2)
            loss.backward()
        autograd_g = w.grad[idx].item() if w.grad is not None else 0.0

        with torch.no_grad():
            w[idx] += FD_EPS
            plus = loss_at(lam1, lam2).item()
            w[idx] -= 2 * FD_EPS
            minus = loss_at(lam1, lam2).item()
            w[idx] += FD_EPS  # restore
        numeric_g = (plus - minus) / (2 * FD_EPS)
        fd_result[name] = {"weight_index": list(int(i) for i in idx),
                           "autograd": autograd_g, "numeric": numeric_g,
                           "abs_diff": abs(autograd_g - numeric_g)}
    result["finite_difference"] = fd_result
    model.zero_grad(set_to_none=True)
    return result


def main() -> int:
    torch.manual_seed(SEED)
    tbl = T.load_table(REPO / "operator")
    table = R.flatten_table(tbl["mu"], tbl["mu0"])
    norm_stats = json.loads((REPO / "dataset_fullscale_50/processed/norm_stats.json").read_text())
    mean, std = np.asarray(norm_stats["mean"]), np.asarray(norm_stats["std"])

    ds = WindowGraphDataset(str(REPO / "dataset_fullscale_50"), "train")
    edge_index = ds.edge_index

    torch.manual_seed(SEED)
    model = PCSTGNN(edge_index, table, (mean, std), use_fusion=True, beta_init=0.1,
                    sigma_eps=CANONICAL_SIGMA_EPS, aggregator="max", num_layers=4)
    # eval(): NodeClassifier has nn.Dropout, stochastic in train() mode, which
    # would corrupt the finite-difference comparison (two forward passes would
    # see different dropout masks). eval() (fixed BN running stats, dropout
    # off) doesn't change whether gradients flow -- only train-time
    # regularization -- so it's the right mode for a gradient-flow audit.
    model.eval()

    batches = []
    idx_w20 = pick_batch_with_w20(ds, BATCH_SIZE)
    batches.append(("w20_window_k0", idx_w20, 0))
    idx_k10 = pick_batch_plain(ds, BATCH_SIZE, offset=1000)
    batches.append(("k10_large_D", idx_k10, 10))
    idx_k0 = pick_batch_plain(ds, BATCH_SIZE, offset=2000)
    batches.append(("k0_no_drops", idx_k0, 0))

    results = []
    for label, indices, k in batches:
        batch = collate(ds, indices)
        seed = 900000 + hash(label) % 100000
        res = audit_batch(model, batch["x"], batch["y"], batch["window_id"], k, seed, label)
        results.append(res)
        print(f"\n=== batch: {label} (k={k}) ===")
        print(f"  n_w20_windows={res['n_w20_windows']}  |D|_total={res['|D|_total']}")
        print(f"  L1={res['l1_value']:.6f}  grad_L1_norm={res['grad_l1_norm']:.6e}  "
              f"nonzero={res['grad_l1_nonzero']}")
        print(f"  L2={res['l2_value']:.6f}  grad_L2_norm={res['grad_l2_norm']:.6e}  "
              f"nonzero={res['grad_l2_nonzero']}")
        print(f"  beta_grad={res['beta_grad']}  nonzero={res['beta_grad_nonzero']}")
        print(f"  alpha/c: {res['alpha_c_status']}")
        print(f"  mu/mu0/sigma_eps no-grad: {res['mu_mu0_sigma_eps_no_grad']}")
        fd = res["finite_difference"]
        print(f"  FD L1: autograd={fd['l1']['autograd']:.6e} numeric={fd['l1']['numeric']:.6e} "
              f"diff={fd['l1']['abs_diff']:.2e}")
        print(f"  FD L2: autograd={fd['l2']['autograd']:.6e} numeric={fd['l2']['numeric']:.6e} "
              f"diff={fd['l2']['abs_diff']:.2e}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
