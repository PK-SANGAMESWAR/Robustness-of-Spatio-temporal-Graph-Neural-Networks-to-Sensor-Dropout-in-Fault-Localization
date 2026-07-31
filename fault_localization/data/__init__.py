"""Frozen-benchmark manifests and lazy window access."""

from .dataset import WindowGraphDataset, make_dataloader
from .manifest import create_manifest, verify_manifest

__all__ = ["WindowGraphDataset", "make_dataloader", "create_manifest", "verify_manifest"]
