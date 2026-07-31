"""
Sensor-dropout study package — fault_localization.dropout

This package implements the core novelty of the project:
  1. masking.py  — Apply sensor dropout to input tensors.
  2. evaluator.py — Evaluate all baselines across dropout levels.
  3. training.py  — Dropout-aware training augmentation.

KNOWN_ISSUES compliance:
  - All models evaluated on IDENTICAL data/splits (internal fairness).
  - Multi-seed reporting (mean ± std) is enforced.
  - Reduced dataset scale is acceptable; documented in reproduction_log.md.
"""

from fault_localization.dropout.masking import apply_sensor_dropout, apply_sensor_dropout_batch
from fault_localization.dropout.evaluator import evaluate_dropout_curve
from fault_localization.dropout.training import DropoutAugmentedLoader

__all__ = [
    "apply_sensor_dropout",
    "apply_sensor_dropout_batch",
    "evaluate_dropout_curve",
    "DropoutAugmentedLoader",
]
