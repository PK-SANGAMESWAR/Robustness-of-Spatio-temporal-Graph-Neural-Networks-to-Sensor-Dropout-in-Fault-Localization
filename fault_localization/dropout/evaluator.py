"""
Degradation curve evaluator for the sensor-dropout study.

Evaluates all trained baseline models across dropout levels k ∈ {0,1,2,5,10},
reporting macro-F1 mean ± std over multiple seeds.

KNOWN_ISSUES compliance:
  - All models evaluated on IDENTICAL test data (internal fairness).
  - Multiple seeds (≥3) with mean ± std reported.
  - Reduced scale is acceptable; document in reproduction_log.md.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score  # type: ignore

from fault_localization.dropout.masking import apply_sensor_dropout_batch
from fault_localization.models.voting import soft_vote

logger = logging.getLogger(__name__)

# Default dropout levels matching the problem statement and Dataset-check.md §11
DEFAULT_DROPOUT_LEVELS: list[int] = [0, 1, 2, 5, 10]


def evaluate_at_dropout(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    k: int,
    mask_seed: int,
    policy: str = "zero_messages",
    f1_average: str = "macro",
) -> dict[str, float]:
    """Evaluate a model at a single dropout level k.

    Parameters
    ----------
    model:
        Trained model in eval mode.
    loader:
        DataLoader producing batches with ``"x"`` and ``"y"`` keys.
    device:
        Torch device.
    k:
        Number of sensors to drop (0 = no dropout).
    mask_seed:
        Seed for reproducible node selection.
    policy:
        ``"zero_messages"`` or ``"drop_edges"``.
    f1_average:
        Averaging strategy for sklearn f1_score.

    Returns
    -------
    dict
        Keys: ``"f1_macro"``, ``"accuracy"``, ``"n_samples"``.
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    # Get edge_index from model if available
    edge_index = getattr(model, "edge_index", None)

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            # Apply dropout to the batch
            if k > 0:
                x_dropped, ei_dropped, _ = apply_sensor_dropout_batch(
                    x, k=k, seed=mask_seed, policy=policy,
                    edge_index=edge_index,
                )
            else:
                x_dropped = x
                ei_dropped = edge_index

            logits = model(x_dropped)  # [B, N, C]

            # Soft voting: sum node softmax probabilities, then argmax —
            # identical to the training/eval pipeline (models/voting.py) so
            # the degradation curve's k=0 matches the reported test_graph.f1.
            _, graph_pred = soft_vote(logits)  # [B]
            all_preds.extend(graph_pred.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    n = len(all_labels)
    f1 = float(f1_score(all_labels, all_preds, average=f1_average, zero_division=0))
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))

    return {"f1_macro": f1, "accuracy": acc, "n_samples": n}


def evaluate_dropout_curve(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    dropout_levels: list[int] = DEFAULT_DROPOUT_LEVELS,
    n_mask_seeds: int = 3,
    policy: str = "zero_messages",
    f1_average: str = "macro",
    model_name: str = "model",
) -> dict[str, Any]:
    """Evaluate a model across all dropout levels, reporting mean ± std.

    For each k, runs evaluation with ``n_mask_seeds`` different random
    mask selections and reports the mean and std of macro-F1.

    Parameters
    ----------
    model:
        Trained model.
    loader:
        Test DataLoader.
    device:
        Torch device.
    dropout_levels:
        List of k values to evaluate (e.g. [0, 1, 2, 5, 10]).
    n_mask_seeds:
        Number of random mask seeds per level (≥3 recommended).
    policy:
        Message-passing policy for dropped nodes.
    f1_average:
        F1 averaging strategy.
    model_name:
        Label for logging.

    Returns
    -------
    dict
        {
          k: {
            "f1_mean": float,
            "f1_std": float,
            "f1_per_seed": [float, ...],
            "acc_mean": float,
          }
          for k in dropout_levels
        }
    """
    results: dict[str, Any] = {}

    for k in dropout_levels:
        f1_per_seed: list[float] = []
        acc_per_seed: list[float] = []

        for seed_offset in range(n_mask_seeds):
            # Use a deterministic seed per (k, seed_offset) so results are
            # reproducible and comparable across models (internal fairness)
            mask_seed = 10000 * k + seed_offset

            metrics = evaluate_at_dropout(
                model, loader, device, k=k,
                mask_seed=mask_seed, policy=policy,
                f1_average=f1_average,
            )
            f1_per_seed.append(metrics["f1_macro"])
            acc_per_seed.append(metrics["accuracy"])

        f1_mean = float(np.mean(f1_per_seed))
        f1_std = float(np.std(f1_per_seed))

        logger.info(
            "[%s] k=%d: F1 = %.4f ± %.4f (seeds=%d, policy=%s)",
            model_name, k, f1_mean, f1_std, n_mask_seeds, policy,
        )
        results[k] = {
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "f1_per_seed": f1_per_seed,
            "acc_mean": float(np.mean(acc_per_seed)),
            "n_mask_seeds": n_mask_seeds,
            "policy": policy,
        }

    return results


def format_degradation_table(
    results_by_model: dict[str, dict],
    dropout_levels: list[int] = DEFAULT_DROPOUT_LEVELS,
) -> str:
    """Format degradation curve results as a markdown table.

    Parameters
    ----------
    results_by_model:
        {model_name: evaluate_dropout_curve() output}
    dropout_levels:
        k values to include as columns.

    Returns
    -------
    str
        Markdown table string.
    """
    header = "| Model | " + " | ".join(f"k={k}" for k in dropout_levels) + " |"
    sep = "|-------|" + "|".join("-------" for _ in dropout_levels) + "|"
    rows = [header, sep]

    for model_name, results in results_by_model.items():
        cells = []
        for k in dropout_levels:
            if k in results:
                r = results[k]
                cells.append(f"{r['f1_mean']:.3f}±{r['f1_std']:.3f}")
            else:
                cells.append("—")
        rows.append(f"| {model_name} | " + " | ".join(cells) + " |")

    return "\n".join(rows)
