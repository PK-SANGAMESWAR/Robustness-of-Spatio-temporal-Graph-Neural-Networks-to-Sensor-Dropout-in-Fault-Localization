"""B2a training hooks: the runner patch (written, not applied), the training epoch, and the two Stage G configs. CPU only."""
from __future__ import annotations

import copy
import importlib.util
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from fault_localization.experiments.config import load_config, validate_config
from fault_localization.models.pc_stgnn import PCSTGNN
from fault_localization.models.pc_stgnn_rhat import PCSTGNNRHat
from fault_localization.physics import grids, residual as R
from fault_localization.stage_g.b2a_train import b2a_train_epoch

torch.set_num_threads(2)
REPO = Path(__file__).resolve().parents[1]
PATCH = REPO / "patches/stage_g/b2a_runner_hooks.patch"
RUNNER = "fault_localization/experiments/runner.py"


def _git_apply(*args, cwd):
    return subprocess.run(["git", "apply", *args, str(PATCH)], cwd=cwd, capture_output=True, text=True)


def _patched_runner_copy(tmp_path) -> Path:
    """Apply the patch to a copy of runner.py outside the repo; the repo file is never touched."""
    d = tmp_path / "copy"
    (d / "fault_localization/experiments").mkdir(parents=True)
    # base = the committed runner.py, so the helper works whether or not the queue has applied the patch to the working tree
    base = subprocess.run(["git", "show", f"HEAD:{RUNNER}"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    (d / RUNNER).write_text(base)
    if "use_rhat_head" in base:                                   # the hooks are already in the committed runner (commit adca737): nothing to apply
        return d / RUNNER
    r = _git_apply(cwd=d)
    assert r.returncode == 0, r.stderr
    return d / RUNNER


def _load_module(path: Path, name="fault_localization.experiments.runner_patched"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "fault_localization.experiments"          # relative imports resolve to the real package
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- the patch
def test_patch_is_not_applied_to_the_repo_runner_unless_the_queue_applied_it():
    text = (REPO / RUNNER).read_text()
    applied = "use_rhat_head" in text
    check = subprocess.run(["git", "apply", "--check", "-R" if applied else "--check", str(PATCH)], cwd=REPO, capture_output=True)
    assert check.returncode == 0                                 # either cleanly applicable, or cleanly applied


def test_patch_touches_only_runner_py():
    files = [l.split()[1] for l in PATCH.read_text().splitlines() if l.startswith("--- ")]
    assert files == ["a/" + RUNNER]


def test_patched_runner_compiles_and_builds_the_right_model(tmp_path):
    path = _patched_runner_copy(tmp_path)
    compile(path.read_text(), str(path), "exec")
    mod = _load_module(path)
    edge = np.load(REPO / "dataset_fullscale_50/processed/edge_index.npy").astype(np.int64)
    b2a = mod._model(validate_config(load_config(str(REPO / "config/stage_g/b2a.yaml"))), edge)
    assert type(b2a) is PCSTGNNRHat and hasattr(b2a, "rhat_head") and b2a.use_fusion
    r5 = mod._model(validate_config(load_config(str(REPO / "config/s4_4/r5.yaml"))), edge)
    assert type(r5) is PCSTGNN and not hasattr(r5, "rhat_head")             # every other arm is unchanged
    retrain = mod._model(validate_config(load_config(str(REPO / "config/stage_g/retrain_imp5.yaml"))), edge)
    nominal = mod._model(validate_config(load_config(str(REPO / "config/s4_4/r5.yaml"))), edge)
    assert type(retrain) is PCSTGNN and not torch.equal(retrain.phys_table, nominal.phys_table)     # the imp5 table
    assert retrain.phys_table.shape == nominal.phys_table.shape


def test_patched_runner_text_has_the_dispatch_and_the_cache_guard(tmp_path):
    text = _patched_runner_copy(tmp_path).read_text()
    assert "pc_rhat_loss_weight" in text and "b2a_train_epoch" in text and "must not use the profiled k = 0 rho cache" in text
    assert text.count("pc_train_epoch(model, train_loader, opt, device,") == 1          # old path kept for every other arm


# ----------------------------------------------------------------------------- the epoch
class _Loader:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


def _batches(n=3, b=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        x = torch.randn(b, 25, 20, 6, generator=g)
        x[..., 3:] = 1.0
        out.append({"x": x, "x_clean": x.clone(), "drop_mask": torch.zeros(b, 25, 3), "y": torch.tensor([3, 0, 7, 11]),
                    "window_id": [f"e{i}{j}_w00{21 + j}" for j in range(b)],
                    "metadata": [{"resistance": r} for r in ("1.0", "1.0", "10.0", "0.1")]})
    return out


@pytest.fixture(scope="module")
def table():
    rng = np.random.default_rng(5)
    mu = rng.normal(1.0, 0.05, (25, 11, grids.NUM_R, 3, 25, 3)).astype(np.float32)
    mu0 = rng.normal(1.0, 0.05, (3, 25, 3)).astype(np.float32)
    return R.flatten_table(mu, mu0)


def _model(table):
    ei = np.load(REPO / "dataset_fullscale_50/processed/edge_index.npy").astype(np.int64)
    torch.manual_seed(0)
    return PCSTGNNRHat(ei, table, (np.array([0.8285, 0.6046, 0.7576], dtype=np.float32), np.array([0.3752, 0.5012, 0.4369], dtype=np.float32)),
                       use_fusion=True, aggregator="max", num_layers=2)


def test_epoch_runs_updates_the_head_and_leaves_the_table_alone(table):
    m = _model(table)
    before_head = m.rhat_head.weight.detach().clone()
    before_table = m.phys_table.clone()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    out = b2a_train_epoch(m, _Loader(_batches()), opt, torch.device("cpu"), grad_clip_norm=1.0, lam1=0.0, lam2=0.1,
                          aux_net_loss_weight=1.0, rhat_loss_weight=1.0)
    assert np.isfinite(out["loss"]) and {"l2", "aux", "rhat_mse"} <= set(out) and "l1" not in out
    assert not torch.equal(before_head, m.rhat_head.weight)
    assert torch.equal(before_table, m.phys_table)


def test_epoch_refuses_the_profiled_cache(table):
    m = _model(table)
    with pytest.raises(ValueError, match="cache"):
        b2a_train_epoch(m, _Loader(_batches(1)), torch.optim.SGD(m.parameters(), lr=0.1), torch.device("cpu"),
                        rho_k0_cache=object())


# ----------------------------------------------------------------------------- the configs
def _diff(a, b, p=""):
    out = []
    for k in sorted(set(a) | set(b)):
        if isinstance(a.get(k), dict) and isinstance(b.get(k), dict):
            out += _diff(a[k], b[k], p + k + ".")
        elif a.get(k) != b.get(k):
            out.append(p + k)
    return out


def test_stage_g_configs_differ_from_r5_only_where_the_amendments_say():
    r5 = yaml.safe_load((REPO / "config/s4_4/r5.yaml").read_text())
    b2a = yaml.safe_load((REPO / "config/stage_g/b2a.yaml").read_text())
    imp = yaml.safe_load((REPO / "config/stage_g/retrain_imp5.yaml").read_text())
    assert _diff(r5, b2a) == ["benchmark_version", "model.use_rhat_head", "output_root", "pc_loss.rhat_loss_weight"]
    assert _diff(r5, imp) == ["benchmark_version", "model.operator_dir", "output_root"]
    for c in (b2a, imp):                                          # R5 settings kept; guard on; no profiled cache
        assert c["pc_loss"]["lambda1"] == 0.0 and c["pc_loss"]["lambda2"] == 0.1 and c["pc_loss"]["aux_net_loss_weight"] == 1.0
        assert c["model"]["use_fusion"] is True and c["training"]["epochs"] == 30 and c["seed"] == 42
        assert c["training"]["collapse_abort"] == {"epoch": 3, "max_class0_frac": 0.95, "min_net_f1": 0.1}
        assert "rho_k0_cache_path" not in c["pc_loss"]
    assert b2a["pc_loss"]["rhat_loss_weight"] == 1.0
    assert (REPO / imp["model"]["operator_dir"] / "mu.npy").exists()


# ----------------------------------------------------------------------------- the patched runner trains (tiny CPU smoke)
@pytest.mark.slow
@pytest.mark.parametrize("name", ["b2a", "retrain_imp5"])
def test_patched_runner_runs_one_epoch_on_a_tiny_pilot_derived_dataset(tmp_path, name):
    """12 pilot events (never the sealed set), 1 epoch, CPU. Checks the hooks end to end: training epoch, validation, logging."""
    from fault_localization.stage_g import ood_pipeline as P
    raw = tmp_path / "src/raw_sim"
    raw.mkdir(parents=True)
    for f in sorted((REPO / "data/ood_pilot/raw_sim").glob("*.npz"))[:12]:
        shutil.copy2(f, raw / f.name)
    root = tmp_path / "root"
    P.build_eval_root(raw, root, REPO / "dataset_fullscale_50", 12)
    ids = (root / "splits/all.txt").read_text().split()
    events = sorted({i.rsplit("_w", 1)[0] for i in ids})
    pick = lambda evs: "\n".join(i for i in ids if i.rsplit("_w", 1)[0] in evs) + "\n"        # noqa: E731
    (root / "splits/train.txt").write_text(pick(events[:8]))
    (root / "splits/val.txt").write_text(pick(events[8:10]))
    (root / "splits/test.txt").write_text(pick(events[10:]))
    mod = _load_module(_patched_runner_copy(tmp_path))
    cfg = validate_config(load_config(str(REPO / f"config/stage_g/{name}.yaml")))
    cfg.update(data_root=str(root), output_root=str(tmp_path / "out"), device="cpu")
    cfg["training"].update(epochs=1, batch_size=8)
    cfg["training"].pop("collapse_abort")
    if name == "b2a":
        cfg["model"]["operator_dir"] = str(REPO / "operator")
    run_dir = mod.run(cfg)
    rows = (Path(run_dir) / "metrics_by_epoch.csv").read_text().splitlines()
    header = rows[0].split(",")
    assert ("train_rhat_mse" in header) == (name == "b2a") and "train_l2" in header and "beta" in header
    assert (Path(run_dir) / "checkpoint_best.pt").exists()
    state = torch.load(Path(run_dir) / "checkpoint_best.pt", weights_only=True)["model"]
    assert ("rhat_head.weight" in state) == (name == "b2a")
