from __future__ import annotations
import csv, json, os, platform, time
from pathlib import Path

# Must be set before importing torch / creating a CUDA context when
# deterministic CuBLAS kernels are requested by the benchmark config.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from fault_localization.data.dataset import WindowGraphDataset, make_dataloader
from fault_localization.data.manifest import create_manifest, verify_manifest, write_manifest
from fault_localization.models import GRUBaseline, RGSAGE, RGCN, RGATv2
from .config import validate_config
from .evaluate import evaluate_loader
from .seed import seed_everything
from .train import train_epoch, validate_epoch


def _model(config, edge):
    m = config["model"]
    common = dict(
        temporal_hidden_dim=m.get("temporal_hidden_dim", 128),
        num_classes=26,
        dropout=m.get("dropout", 0.35),
    )
    name = m["name"]
    if name == "gru":
        return GRUBaseline(**common)
    if name.startswith("rgsage"):
        return RGSAGE(
            edge, m["aggregator"],
            gnn_hidden_dim=m.get("gnn_hidden_dim", 64),
            num_layers=config.get("gnn_layers", 1),
            **common,
        )
    if name == "rgcn":
        return RGCN(
            edge, gnn_hidden_dim=m.get("gnn_hidden_dim", 64),
            num_layers=config.get("gnn_layers", 1),
            **common,
        )
    if name == "rgatv2":
        return RGATv2(
            edge,
            gnn_hidden_dim=m.get("gnn_hidden_dim", 64),
            heads=m.get("heads", 4),
            attention_dropout=m.get("attention_dropout", 0.3),
            num_layers=config.get("gnn_layers", 1),
            **common,
        )
    raise ValueError(f"Unknown model: {name}")


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run(config, resume=False):
    config = validate_config(config)
    root = Path(config["data_root"])
    output = Path(config["output_root"]) / config["benchmark_version"]
    run_dir = output / config["model"]["name"] / f"seed_{config['seed']:05d}"

    # Dataset manifest versioning
    frozen = output / "manifest.json"
    current = create_manifest(root, config["benchmark_version"])
    if frozen.exists():
        verify_manifest(root, frozen)
    else:
        write_manifest(current, frozen)

    if run_dir.exists() and (run_dir / "metrics.json").exists():
        raise FileExistsError(f"Completed run is immutable: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    import yaml
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    write_manifest(current, run_dir / "dataset_manifest.json")

    device = torch.device(config["device"])
    seed_state = seed_everything(config["seed"], config["training"].get("deterministic", True))
    _json(
        run_dir / "environment.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            **seed_state,
        },
    )

    # Datasets
    train_ds = WindowGraphDataset(root, "train")
    val_ds = WindowGraphDataset(root, "val")
    if not len(train_ds) or not len(val_ds):
        raise RuntimeError("Benchmark requires non-empty train and validation splits")

    # Diagnostic ablation (MEDIUM_DATASET_TREND_TEST.md probe): construct the
    # model with an empty edge_index so every node only self-loops. Isolates
    # whether spatial message passing — not just extra parameters — drives
    # any macro-F1 gain over the GRU floor.
    disable_edges = config.get("ablation", {}).get("disable_edges", False)
    model_edge = np.zeros((2, 0), dtype=np.int64) if disable_edges else train_ds.edge_index
    model = _model(config, model_edge).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # ----------------------------------------------------------------
    # Dropout-aware training (PROBLEM.md §4 novelty)
    # ----------------------------------------------------------------
    ds_cfg = config.get("dropout_study", {})
    dropout_aware = ds_cfg.get("dropout_aware_training", False)
    dropout_policy = ds_cfg.get("policy", "zero_messages")
    train_k_range = ds_cfg.get("train_k_range", [0, 1, 2, 5])

    num_workers = config["training"].get("num_workers", 0)

    def _make_train_loader(epoch: int):
        base_loader = make_dataloader(
            train_ds, config["training"]["batch_size"], shuffle=True,
            seed=config["seed"] + epoch, num_workers=num_workers,
        )
        if dropout_aware:
            from fault_localization.dropout.training import DropoutAugmentedLoader
            return DropoutAugmentedLoader(
                base_loader,
                k_range=train_k_range,
                rng_seed=config["seed"] + epoch,
                policy=dropout_policy,
            )
        return base_loader

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    best = -float("inf")
    history = []
    epochs = config["training"]["epochs"]
    grad_clip_norm = config["training"].get("grad_clip_norm")
    warmup_epochs = config["training"].get("warmup_epochs", 0)
    base_lr = config["training"]["learning_rate"]

    for epoch in range(epochs):
        if warmup_epochs > 0:
            lr_scale = min(1.0, (epoch + 1) / warmup_epochs)
            for group in opt.param_groups:
                group["lr"] = base_lr * lr_scale

        train_loader = _make_train_loader(epoch)
        val_loader = make_dataloader(
            val_ds, config["training"]["batch_size"], shuffle=False, num_workers=num_workers,
        )

        tr = train_epoch(model, train_loader, opt, device, grad_clip_norm=grad_clip_norm)
        va = validate_epoch(
            model, val_loader, device, config["metrics"]["f1_average"],
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": tr["loss"],
            "train_seconds": tr["seconds"],
            "val_graph_f1": va["graph"]["f1"],
            "val_node_f1": va["node"]["f1"],
        }
        history.append(row)

        torch.save(
            {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch},
            run_dir / "checkpoint_last.pt",
        )
        if row["val_graph_f1"] > best:
            best = row["val_graph_f1"]
            torch.save(
                {"model": model.state_dict(), "epoch": epoch},
                run_dir / "checkpoint_best.pt",
            )

        # Rewrite epoch history after every epoch (not just at the end) so a
        # long run's progress is inspectable mid-training.
        with (run_dir / "metrics_by_epoch.csv").open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=history[0])
            w.writeheader()
            w.writerows(history)

    # ----------------------------------------------------------------
    # Test evaluation
    # ----------------------------------------------------------------
    model.load_state_dict(
        torch.load(
            run_dir / "checkpoint_best.pt",
            map_location=device,
            weights_only=True,
        )["model"]
    )
    test_ds = WindowGraphDataset(root, "test")
    test_loader = make_dataloader(
        test_ds, config["training"]["batch_size"], shuffle=False, num_workers=num_workers,
    )
    result = evaluate_loader(
        model, test_loader, device,
        config["metrics"]["f1_average"],
        records=True,
    )

    metrics = {
        "selection_metric": "validation soft-voted graph F1",
        "best_validation_graph_f1": best,
        "test_graph": result["graph"],
        "test_node_diagnostic": result["node"],
        "dropout_aware_training": dropout_aware,
    }
    _json(run_dir / "metrics.json", metrics)

    import pandas as pd
    pd.DataFrame(result["predictions"]).to_csv(
        run_dir / "predictions_test.csv", index=False,
    )
    np.savetxt(
        run_dir / "confusion_matrix.csv",
        np.asarray(result["graph"]["confusion_matrix"]),
        fmt="%d",
        delimiter=",",
    )

    # ----------------------------------------------------------------
    # Sensor-dropout study (PROBLEM.md §4 novelty)
    # Runs AFTER standard test evaluation using the same best checkpoint.
    # KNOWN_ISSUES §3: all models evaluated on identical test data.
    # ----------------------------------------------------------------
    if ds_cfg.get("enabled", False):
        _run_dropout_study(model, test_loader, device, config, ds_cfg, run_dir)

    return run_dir


def _run_dropout_study(model, test_loader, device, config, ds_cfg, run_dir):
    """Run the sensor-dropout degradation curve study."""
    from fault_localization.dropout.evaluator import (
        evaluate_dropout_curve,
        format_degradation_table,
    )

    k_levels = ds_cfg.get("k_levels", [0, 1, 2, 5, 10])
    n_mask_seeds = ds_cfg.get("n_mask_seeds", 3)
    policy = ds_cfg.get("policy", "zero_messages")

    model_name = config["model"]["name"]

    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        "Running dropout study: k=%s, n_mask_seeds=%d, policy=%s",
        k_levels, n_mask_seeds, policy,
    )

    dropout_results = evaluate_dropout_curve(
        model=model,
        loader=test_loader,
        device=device,
        dropout_levels=k_levels,
        n_mask_seeds=n_mask_seeds,
        policy=policy,
        f1_average=config["metrics"]["f1_average"],
        model_name=model_name,
    )

    # Save numeric results
    _json(run_dir / "dropout_curve.json", {
        str(k): v for k, v in dropout_results.items()
    })

    # Save markdown table
    table = format_degradation_table({model_name: dropout_results}, k_levels)
    (run_dir / "dropout_curve.md").write_text(
        f"# Sensor-Dropout Degradation Curve\n\n"
        f"**Model:** {model_name}  \n"
        f"**Seeds per level:** {n_mask_seeds}  \n"
        f"**Policy:** {policy}  \n\n"
        f"{table}\n\n"
        f"*F1 values shown as mean ± std over {n_mask_seeds} mask seeds.*\n"
    )

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Dropout study results saved to %s", run_dir / "dropout_curve.json")
