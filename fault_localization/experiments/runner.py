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
from fault_localization.models import (GRUBaseline, RGSAGE, RGCN, RGATv2,
                                      SDRSTGNN, FaultQuerySTGNN,
                                      ZoneRefineSTGNN, PIGNNSage)
from .config import validate_config
from .evaluate import evaluate_loader
from .seed import seed_everything
from .train import train_epoch, validate_epoch, pc_train_epoch


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
    if name == "sdr_stgnn":
        return SDRSTGNN(
            edge, aggregator=m.get("aggregator", "max"),
            gnn_hidden_dim=m.get("gnn_hidden_dim", 64),
            num_layers=config.get("gnn_layers", 1),
            **common,
        )
    if name == "zone_refine":
        from fault_localization.graph.sensitivity import load_sensitivity
        # Static feeder physics; cached .npz, so no OpenDSS at training time.
        phys = load_sensitivity(
            m.get("node_mapping_path",
                  f"{config['data_root']}/processed/node_mapping.json"))
        return ZoneRefineSTGNN(
            edge, phys["zone_templates"], phys["zone_of_class"],
            aggregator=m.get("aggregator", "max"),
            gnn_hidden_dim=m.get("gnn_hidden_dim", 64),
            num_layers=config.get("gnn_layers", 1),
            learn_templates=m.get("learn_templates", True),
            use_physics_zones=m.get("use_physics_zones", True),
            **common,
        )
    if name == "fault_query":
        from fault_localization.graph.fault_distance import load_fault_hop_distance
        # Static feeder geometry; cached .npy, so no OpenDSS at training time.
        hop_distance = load_fault_hop_distance(
            m.get("node_mapping_path",
                  f"{config['data_root']}/processed/node_mapping.json"))
        return FaultQuerySTGNN(
            edge, hop_distance,
            aggregator=m.get("aggregator", "max"),
            gnn_hidden_dim=m.get("gnn_hidden_dim", 64),
            num_layers=config.get("gnn_layers", 1),
            num_heads=m.get("num_heads", 4),
            alpha_init=m.get("alpha_init", 0.1),
            use_distance_bias=m.get("use_distance_bias", True),
            use_availability_mask=m.get("use_availability_mask", True),
            use_reliability=m.get("use_reliability", True),
            use_local_gnn=m.get("use_local_gnn", True),
            **common,
        )
    if name == "pcstgnn":
        # PC-STGNN (models/pc_stgnn.py), PC-STGNN/01_SPEC.md Sec 6. With every
        # physics switch off (the S4.1 null test) this is byte-for-byte
        # RGSAGE-Max (tests/test_pc_stgnn.py::test_all_switches_off_is_exactly_rgsage),
        # which is what lets S4.1 claim "lambda = beta = 0 recovers Paper-1
        # exactly". Table/norm_stats are only loaded when a physics switch
        # actually needs them, mirroring the pignn branch below.
        from fault_localization.models.pc_stgnn import PCSTGNN
        use_fusion = m.get("use_fusion", False)
        # `[Fixed 2026-10-22, PC-STGNN/S4.4_remaining_arms_plan.md S4.1]` Previously gated only
        # on use_fusion, so a fusion-off arm with lambda1>0 or lambda2>0 (R3/R4: physics as a
        # training constraint, no beta fusion) had no table loaded -- forward_pc() then had no
        # "rho_A"/"rho_D" in its output dict, and pc_loss() raised
        # KeyError("pc_loss needs out['rho_A'] ...") as soon as training started. The pc_loss
        # weights only matter if pc_loss itself is enabled (pc_enabled gates whether
        # pc_train_epoch runs at all, see pc_enabled usage below in run()), so this also checks
        # pc_cfg.enabled -- an R0^n-style run (pc_loss.enabled=true purely for the noise-aug
        # loader plumbing, lambda1=lambda2=0) still correctly loads no table. A buffer
        # (register_buffer, not a Parameter) does not draw from the RNG stream, so this cannot
        # change byte-identity for any arm that doesn't actually need a table (S4.1).
        pc_cfg = config.get("pc_loss", {})
        needs_table = use_fusion or (
            bool(pc_cfg.get("enabled", False))
            and (float(pc_cfg.get("lambda1", 0.0)) > 0 or float(pc_cfg.get("lambda2", 0.0)) > 0)
        )
        # R2 (use_decoder) needs norm_stats to de-normalise x_clean into the recon
        # target (train.py::pc_train_epoch, observation.observe) even though it
        # needs no signature table -- loaded independently of needs_table below.
        needs_norm_stats = needs_table or m.get("use_decoder", False)
        table = None
        norm_stats = None
        if needs_table:
            from fault_localization.physics import residual as _R
            from fault_localization.physics import table as _T
            tbl = _T.load_table(m.get("operator_dir", "operator"))
            table = _R.flatten_table(tbl["mu"], tbl["mu0"])
        if needs_norm_stats:
            from fault_localization.physics.observation import load_norm_stats
            norm_stats = load_norm_stats(config["data_root"])
        return PCSTGNN(
            edge, table, norm_stats,
            use_fusion=use_fusion,
            beta_init=m.get("beta_init", 0.0),
            sigma_eps=m.get("sigma_eps", 1e-3),
            sigma0=m.get("sigma0", 0.0),
            shuffle_seed=m.get("shuffle_seed"),
            use_type_head=m.get("use_type_head", False),
            num_fault_types=m.get("num_fault_types", 11),
            use_decoder=m.get("use_decoder", False),
            decoder_hidden=m.get("decoder_hidden", 64),
            aggregator=m.get("aggregator", "max"),
            gnn_hidden_dim=m.get("gnn_hidden_dim", 64),
            num_layers=config.get("gnn_layers", 1),
            **common,
        )
    if name == "pignn":
        # PIGNN screening arms P1/P2 (models/pignn.py). Both physics artifacts
        # are static feeder properties cached beside the dataset, so no OpenDSS
        # is needed at training time. With both switches off this is byte-for-
        # byte RGSAGE (asserted in tests/test_pignn.py).
        edge_attr = None
        sensitivity = None
        if m.get("use_impedance_weights", False):
            edge_attr = np.load(f"{config['data_root']}/processed/edge_attr.npy")
        if m.get("use_sensitivity_bias", False):
            from fault_localization.graph.sensitivity import load_sensitivity
            sensitivity = load_sensitivity(
                m.get("node_mapping_path",
                      f"{config['data_root']}/processed/node_mapping.json")
            )["sensitivity"]
        return PIGNNSage(
            edge, edge_attr, sensitivity,
            aggregator=m.get("aggregator", "max"),
            gnn_hidden_dim=m.get("gnn_hidden_dim", 64),
            num_layers=config.get("gnn_layers", 1),
            use_impedance_weights=m.get("use_impedance_weights", False),
            use_sensitivity_bias=m.get("use_sensitivity_bias", False),
            gamma_init=m.get("gamma_init", 1.0),
            beta_init=m.get("beta_init", 0.1),
            **common,
        )

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

    # ----------------------------------------------------------------
    # PC-STGNN objective (PC-STGNN/01_SPEC.md Sec 6, models/common.py::pc_loss).
    # Off unless the config asks for it, so the S4.1 null test and every other
    # existing benchmark keep training through the plain train_epoch/CE path
    # unchanged. When on, lambda_2 needs the pre-dropout batch and the dropped-
    # channel mask, so the train loader is a PhysicsDropoutLoader regardless of
    # the generic dropout_aware flag (k_range collapses to [0] if that flag is
    # off, which still yields well-formed, all-zero drop_mask batches).
    # ----------------------------------------------------------------
    pc_cfg = config.get("pc_loss", {})
    pc_enabled = bool(pc_cfg.get("enabled", False))
    pc_lam1 = float(pc_cfg.get("lambda1", 0.0))
    pc_lam2 = float(pc_cfg.get("lambda2", 0.0))
    # S4.3d: gamma * CE on the network-only (pre-fusion) logits. 0 = objective unchanged.
    pc_aux = float(pc_cfg.get("aux_net_loss_weight", 0.0))
    # R1/R2 (models/common.py::pc_loss): both default to 0, so every arm without
    # use_type_head/use_decoder is unaffected.
    pc_type_loss_weight = float(pc_cfg.get("type_loss_weight", 0.0))
    pc_recon_loss_weight = float(pc_cfg.get("recon_loss_weight", 0.0))
    # k=0 rho(A) cache (2026-09-30): at k=0 rho(A) is a fixed function of the window
    # alone, so it is looked up from the offline grid (physics/cache.py, S3.4) instead
    # of recomputed live -- verified equivalent before use, see
    # scripts/verify_pcstgnn_rho_cache.py. k>0 stays live, unaffected. Off unless a
    # cache path is given, so this never changes any run that doesn't ask for it.
    pc_rho_k0_cache_path = pc_cfg.get("rho_k0_cache_path")
    pc_rho_k0_cache = None
    if pc_enabled and pc_rho_k0_cache_path:
        from fault_localization.physics.cache import load_rho_k0_lookup
        pc_rho_k0_cache = load_rho_k0_lookup(pc_rho_k0_cache_path)

    # Measurement-noise augmentation (06_STAGE_E_pcstgnn.md S4.3b): sigma drawn per window
    # from the grid, sigma0 in the residual = the sigma injected. Off unless asked for.
    # Wired only through the PC loader so far; failing loudly beats a silent no-op.
    noise_cfg = config.get("noise_augmentation", {})
    noise_sigmas = list(noise_cfg.get("sigma_grid", [])) if noise_cfg.get("enabled", False) else None
    if noise_sigmas and not pc_enabled:
        raise NotImplementedError("noise_augmentation is only wired through the pc_loss path")
    # S4.3c collapse monitor: network-only metrics on the validation split every epoch.
    log_network_only = bool(pc_cfg.get("log_network_only", False))
    collapse_abort = config["training"].get("collapse_abort")
    if collapse_abort and not log_network_only:
        raise ValueError("training.collapse_abort needs pc_loss.log_network_only")
    collapsed = False

    def _make_train_loader(epoch: int):
        base_loader = make_dataloader(
            train_ds, config["training"]["batch_size"], shuffle=True,
            seed=config["seed"] + epoch, num_workers=num_workers,
        )
        if pc_enabled:
            from fault_localization.dropout.training import PhysicsDropoutLoader
            return PhysicsDropoutLoader(
                base_loader,
                k_range=train_k_range if dropout_aware else [0],
                rng_seed=config["seed"] + epoch,
                policy=dropout_policy,
                noise_sigmas=noise_sigmas,
                norm_std=train_ds.std if noise_sigmas else None,
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

    # ----------------------------------------------------------------
    # P3 — physics-consistency loss (models/physics_loss.py). Off unless the
    # config asks for it, so every existing benchmark keeps a pure-CE objective.
    # ----------------------------------------------------------------
    phys_cfg = config.get("physics_loss", {})
    physics = None
    if phys_cfg.get("enabled", False):
        from fault_localization.graph.sensitivity import load_sensitivity
        signatures = load_sensitivity(
            phys_cfg.get("node_mapping_path",
                         f"{config['data_root']}/processed/node_mapping.json")
        )["sensitivity"]
        physics = {
            "lam": float(phys_cfg.get("lambda", 1.0)),
            "signatures": torch.as_tensor(signatures, dtype=torch.float32),
        }

    for epoch in range(epochs):
        if warmup_epochs > 0:
            lr_scale = min(1.0, (epoch + 1) / warmup_epochs)
            for group in opt.param_groups:
                group["lr"] = base_lr * lr_scale

        train_loader = _make_train_loader(epoch)
        val_loader = make_dataloader(
            val_ds, config["training"]["batch_size"], shuffle=False, num_workers=num_workers,
        )

        if pc_enabled:
            tr = pc_train_epoch(model, train_loader, opt, device,
                                grad_clip_norm=grad_clip_norm, lam1=pc_lam1, lam2=pc_lam2,
                                rho_k0_cache=pc_rho_k0_cache, aux_net_loss_weight=pc_aux,
                                type_loss_weight=pc_type_loss_weight,
                                recon_loss_weight=pc_recon_loss_weight)
        else:
            tr = train_epoch(model, train_loader, opt, device,
                             grad_clip_norm=grad_clip_norm, physics=physics)
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
        if "phys_loss" in tr:
            row["train_phys_loss"] = tr["phys_loss"]
        if "l1" in tr:
            row["train_l1"] = tr["l1"]
        if "l2" in tr:
            row["train_l2"] = tr["l2"]
        if "aux" in tr:
            row["train_aux"] = tr["aux"]
        if "type_ce" in tr:
            row["train_type_ce"] = tr["type_ce"]
        if "recon_mse" in tr:
            row["train_recon_mse"] = tr["recon_mse"]
        if pc_enabled and getattr(model, "use_fusion", False):
            row["beta"] = float(model.beta.item())
        if log_network_only and hasattr(model, "network_logits"):
            row.update(_network_only_eval(model, val_loader, device,
                                          config["metrics"]["f1_average"], train_ds.std))
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

        # S4.3e early stop: a fusion run whose network-only branch is still collapsed at the
        # guard epoch is aborted. Off unless the config asks for it.
        if (collapse_abort and epoch + 1 == collapse_abort["epoch"]
                and (row["val_net_class0_frac"] > collapse_abort["max_class0_frac"]
                     or row["val_net_f1"] <= collapse_abort["min_net_f1"])):
            collapsed = True
            break

    # Validation-only diagnostics (S4.3d) must never touch the test split.
    if config["training"].get("skip_test_eval", False):
        metrics = {
            "selection_metric": "validation soft-voted graph F1",
            "best_validation_graph_f1": best,
            "test_evaluated": False,
            "dropout_aware_training": dropout_aware,
        }
        if collapse_abort:
            metrics["collapse_abort"] = dict(collapse_abort, triggered=collapsed)
        if pc_enabled:
            metrics["pc_loss"] = {"lambda1": pc_lam1, "lambda2": pc_lam2,
                                  "aux_net_loss_weight": pc_aux,
                                  "type_loss_weight": pc_type_loss_weight,
                                  "recon_loss_weight": pc_recon_loss_weight}
            if getattr(model, "use_fusion", False):
                metrics["beta_final"] = float(model.beta.item())
        _json(run_dir / "metrics.json", metrics)
        return run_dir

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
    if pc_enabled:
        metrics["pc_loss"] = {"lambda1": pc_lam1, "lambda2": pc_lam2}
        if getattr(model, "use_fusion", False):
            metrics["beta_final"] = float(model.beta.item())
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


def _network_only_eval(model, loader, device, average, norm_std, noisy_sigma=1e-2):
    """S4.3c collapse monitor: the learned branch alone, physics term removed.

    Clean: macro-F1 and the fraction of windows voted "no fault" (class 0). Noisy: macro-F1
    at sigma = 1e-2 p.u., noise from dropout/noise.py with a per-batch seed.
    """
    from sklearn.metrics import f1_score
    from fault_localization.dropout.noise import apply_measurement_noise_batch
    from fault_localization.models.voting import soft_vote
    model.eval()
    ys, clean, noisy = [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["x"].to(device, non_blocking=True)
            ys.extend(batch["y"].tolist())
            clean.extend(soft_vote(model.network_logits(x))[1].cpu().tolist())
            xn = apply_measurement_noise_batch(x, noisy_sigma, norm_std, seed=880000 + batch_idx)
            noisy.extend(soft_vote(model.network_logits(xn))[1].cpu().tolist())
    clean_np = np.asarray(clean)
    return {
        "val_net_f1": float(f1_score(ys, clean, average=average, zero_division=0)),
        "val_net_class0_frac": float((clean_np == 0).mean()),
        "val_net_f1_sigma1e-2": float(f1_score(ys, noisy, average=average, zero_division=0)),
    }


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
