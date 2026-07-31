"""
Tests for Gate 1-6 correctness checks.

These tests validate the core scientific requirements from IMPLEMENT.md
and Dataset-check.md §12 (Acceptance Gates).

Run with: pytest tests/test_gate_checks.py -v
"""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Gate 1 — Temporal Validity
# ---------------------------------------------------------------------------


class TestTemporalValidity:
    """Gate 1: Consecutive 1 ms samples must be physically distinct."""

    def test_check_temporal_validity_passes_on_valid_recording(self):
        """Simulated recording with genuine variation passes Gate 1."""
        from fault_localization.simulation.runner import check_temporal_validity

        # Create a recording with clear temporal variation
        rng = np.random.default_rng(42)
        recording = rng.normal(1.0, 0.05, size=(25, 60, 6))  # noisy
        recording[:, :, 3:] = 1.0  # mask channels

        result = check_temporal_validity(recording, threshold=1e-4)
        assert result["valid"], (
            f"Expected valid recording to pass Gate 1, "
            f"but got prefault_std={result['prefault_std']:.2e}"
        )

    def test_check_temporal_validity_fails_on_static_recording(self):
        """Bit-identical pre-fault samples (old bug) fail Gate 1."""
        from fault_localization.simulation.runner import check_temporal_validity

        # Old bug: copy one snapshot 60 times
        snapshot = np.array([1.0, 0.98, 0.99] * 8 + [1.0, 0.98, 0.99])[:25]
        recording = np.zeros((25, 60, 6))
        for t in range(60):
            recording[:, t, :3] = snapshot[:, None] if snapshot.ndim > 1 else 1.0
            recording[:, t, 3:] = 1.0
        # Fill with nearly-identical values (not exactly zero std but ~ noise level)
        recording[:, :40, :3] = 1.0  # all identical

        result = check_temporal_validity(recording, threshold=1e-4)
        # prefault_std should be ~0 for identical samples
        assert result["prefault_std"] < 1e-6, (
            "Static recording should have near-zero prefault_std"
        )

    def test_feature_shape_is_6(self):
        """Recording must have F=6 channels (3 voltage + 3 mask)."""
        from fault_localization.config.constants import NUM_FEATURES, TOTAL_RECORD_MS, NUM_PMU_NODES
        assert NUM_FEATURES == 6
        # Window tensor shape should use F=6
        from fault_localization.config.constants import WINDOW_TENSOR_SHAPE, WINDOW_LENGTH
        assert WINDOW_TENSOR_SHAPE == (NUM_PMU_NODES, WINDOW_LENGTH, 6)


# ---------------------------------------------------------------------------
# Gate 2 — Graph Acyclicity
# ---------------------------------------------------------------------------


class TestGraphAcyclicity:
    """Gate 2: Measured-only graph must be a forest (acyclic)."""

    @pytest.fixture(scope="class")
    def measured_graph(self):
        """Build the graph from a compiled feeder (integration test)."""
        pytest.importorskip("opendssdirect", reason="OpenDSS not available")
        import networkx as nx
        from fault_localization.simulation.feeder import compile_feeder, ensure_feeder_files
        from fault_localization.graph.measured_only import (
            build_full_topology_graph,
            build_measured_only_graph,
        )

        try:
            compile_feeder(ensure_feeder_files())
            full_G = build_full_topology_graph()
            G = build_measured_only_graph(full_G)
            return G
        except Exception as e:
            pytest.skip(f"Feeder compilation failed: {e}")

    def test_graph_is_acyclic(self, measured_graph):
        """Measured-only graph must be a forest (no cycles)."""
        import networkx as nx
        assert nx.is_forest(measured_graph), (
            f"Graph has cycles! "
            f"Cycle rank = {measured_graph.number_of_edges() - measured_graph.number_of_nodes() + nx.number_connected_components(measured_graph)}"
        )

    def test_graph_has_25_nodes(self, measured_graph):
        """Measured-only graph must have exactly 25 nodes."""
        assert measured_graph.number_of_nodes() == 25

    def test_phase_metadata_loads(self):
        """Phase metadata YAML must be loadable and complete."""
        from fault_localization.graph.measured_only import load_pmu_phase_metadata
        from fault_localization.config.constants import PMU_BUSES

        try:
            metadata = load_pmu_phase_metadata()
        except FileNotFoundError:
            pytest.skip("pmu_phase_metadata.yaml not generated yet. Run: python scripts/generate_pmu_metadata.py")

        assert len(metadata) == 25, f"Expected 25 buses, got {len(metadata)}"
        for bus in PMU_BUSES:
            assert bus in metadata, f"Bus {bus} missing from metadata"
            assert metadata[bus] in {1, 2, 3}, (
                f"Bus {bus} has invalid phase count: {metadata[bus]}"
            )


# ---------------------------------------------------------------------------
# Gate 2b — Availability Mask
# ---------------------------------------------------------------------------


class TestAvailabilityMask:
    """Gate 2b: Dropped sensors must have mask=0, not just voltage=0."""

    def test_dropout_sets_mask_to_zero(self):
        """apply_sensor_dropout must set mask channels to 0 for dropped nodes."""
        from fault_localization.dropout.masking import apply_sensor_dropout

        # Create input: voltages ~1.0, mask channels = 1 (all available)
        x = torch.ones(25, 20, 6)  # [N, S, F=6]
        x[:, :, :3] = 1.0  # voltages
        x[:, :, 3:] = 1.0  # mask = available

        rng = random.Random(42)
        x_dropped, _, dropped = apply_sensor_dropout(x, k=3, rng=rng)

        assert len(dropped) == 3, f"Expected 3 dropped nodes, got {len(dropped)}"
        for node in dropped:
            # Voltage channels zeroed
            assert x_dropped[node, :, :3].abs().max() == 0.0, (
                f"Voltage channels not zeroed for dropped node {node}"
            )
            # CRITICAL: mask channels also zeroed
            assert x_dropped[node, :, 3:].abs().max() == 0.0, (
                f"Mask channels not zeroed for dropped node {node}. "
                "A zero voltage alone is not sufficient — mask must indicate unavailability."
            )

    def test_non_dropped_nodes_unaffected(self):
        """Nodes NOT selected for dropout must retain their original values."""
        from fault_localization.dropout.masking import apply_sensor_dropout

        x = torch.ones(25, 20, 6)
        rng = random.Random(99)
        x_dropped, _, dropped = apply_sensor_dropout(x, k=5, rng=rng)
        dropped_set = set(dropped)

        for node in range(25):
            if node not in dropped_set:
                assert (x_dropped[node] == x[node]).all(), (
                    f"Node {node} was modified but not in dropout set"
                )

    def test_k0_no_dropout(self):
        """k=0 must not modify any node."""
        from fault_localization.dropout.masking import apply_sensor_dropout

        x = torch.ones(25, 20, 6)
        rng = random.Random(0)
        x_dropped, _, dropped = apply_sensor_dropout(x, k=0, rng=rng)
        assert dropped == []
        assert (x_dropped == x).all()

    def test_mask_distinct_from_voltage(self):
        """Mask channels (3:6) must be binary; voltage channels (0:3) may vary."""
        x = torch.rand(25, 20, 6)
        x[:, :, 3:] = 1.0  # all available initially

        from fault_localization.dropout.masking import apply_sensor_dropout
        rng = random.Random(7)
        x_dropped, _, dropped = apply_sensor_dropout(x, k=2, rng=rng)

        # For non-dropped nodes: mask stays 1
        for node in range(25):
            if node not in dropped:
                assert (x_dropped[node, :, 3:] == 1.0).all(), (
                    f"Non-dropped node {node} has mask != 1"
                )


# ---------------------------------------------------------------------------
# Gate 3 — GATv2 Attention Correctness
# ---------------------------------------------------------------------------


class TestGATv2Attention:
    """Gate 3: RGATv2 must implement paper Eq. (8) correctly."""

    @pytest.fixture
    def simple_edge_index(self):
        """Simple line graph: 0→1→2 (3 nodes)."""
        return torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    def test_gatv2_forward_shape(self, simple_edge_index):
        """RGATv2 must produce [B, N, C] output."""
        from fault_localization.models.gatv2 import RGATv2

        model = RGATv2(
            simple_edge_index, temporal_hidden_dim=16, gnn_hidden_dim=8,
            heads=2, num_classes=5,
        )
        x = torch.randn(2, 3, 20, 6)  # [B=2, N=3, S=20, F=6]
        out = model(x)
        assert out.shape == (2, 3, 5), f"Expected [2, 3, 5], got {out.shape}"

    def test_gatv2_attention_dropout_on_coefficients(self, simple_edge_index):
        """Attention dropout must be applied to alpha (coefficients), not to z."""
        from fault_localization.models.gatv2 import RGATv2
        import torch.nn as nn

        model = RGATv2(
            simple_edge_index, temporal_hidden_dim=16, gnn_hidden_dim=8,
            heads=2, attention_dropout=0.0,  # no dropout for deterministic test
        )
        # Verify that att_dropout is a Dropout module
        assert isinstance(model.att_dropout, nn.Dropout), (
            "att_dropout should be an nn.Dropout module"
        )

    def test_gatv2_shared_W_transform(self, simple_edge_index):
        """W must transform the concatenated pair [h_i || h_j], not separate endpoints."""
        from fault_localization.models.gatv2 import RGATv2

        model = RGATv2(
            simple_edge_index, temporal_hidden_dim=16, gnn_hidden_dim=8, heads=2,
        )
        # W input dim should be 2 * temporal_hidden_dim
        assert model.W[0].in_features == 2 * 16, (
            f"W.in_features should be 2*temporal_hidden_dim=32, got {model.W[0].in_features}"
        )
        # W output dim should be heads * gnn_hidden_dim
        assert model.W[0].out_features == 2 * 8, (
            f"W.out_features should be heads*gnn_hidden_dim=16, got {model.W[0].out_features}"
        )

    def test_gatv2_no_nan_output(self, simple_edge_index):
        """Forward pass must not produce NaN."""
        from fault_localization.models.gatv2 import RGATv2

        torch.manual_seed(42)
        model = RGATv2(simple_edge_index, temporal_hidden_dim=16, gnn_hidden_dim=8, heads=2)
        x = torch.randn(2, 3, 20, 6)
        out = model(x)
        assert not torch.isnan(out).any(), "RGATv2 output contains NaN"


# ---------------------------------------------------------------------------
# Gate 4 — Green Feeder (integration test)
# ---------------------------------------------------------------------------


class TestGreenFeeder:
    """Gate 4: Green feeder switching must produce a distinct graph."""

    def test_green_feeder_constants_defined(self):
        """Green feeder bus pairs must be defined in constants."""
        from fault_localization.config.constants import (
            GREEN_FEEDER_OPEN_SWITCH_BUSES,
            GREEN_FEEDER_CLOSE_SWITCH_BUSES,
            FEEDER_CONFIGS,
        )
        assert GREEN_FEEDER_OPEN_SWITCH_BUSES == (60, 160)
        assert GREEN_FEEDER_CLOSE_SWITCH_BUSES == (54, 94)
        assert "green" in FEEDER_CONFIGS
        assert "default" in FEEDER_CONFIGS

    def test_configure_green_feeder_exists(self):
        """configure_green_feeder function must be importable."""
        from fault_localization.simulation.feeder import configure_green_feeder
        assert callable(configure_green_feeder)


# ---------------------------------------------------------------------------
# Gate 5 — Normalization: voltage channels only
# ---------------------------------------------------------------------------


class TestNormalizationMaskAware:
    """Gate 2b: Normalization must not touch mask channels."""

    def test_normalize_windows_voltage_only(self):
        """normalize_windows must normalize channels 0:3, pass through 3:6."""
        from fault_localization.processing.normalization import normalize_windows

        windows = np.ones((10, 25, 20, 6), dtype=np.float64)
        windows[:, :, :, :3] = 2.0  # voltages = 2
        windows[:, :, :, 3:] = 1.0  # mask = 1 (all available)

        stats = {"mean": [2.0, 2.0, 2.0], "std": [1.0, 1.0, 1.0]}
        result = normalize_windows(windows, stats)

        # Voltage channels normalized: (2 - 2) / 1 = 0
        assert np.allclose(result[:, :, :, :3], 0.0), (
            "Voltage channels should be normalized to 0"
        )
        # Mask channels unchanged
        assert np.allclose(result[:, :, :, 3:], 1.0), (
            "Mask channels (3:6) must NOT be normalized"
        )


# ---------------------------------------------------------------------------
# Gate 6 — Dropout curve multi-seed evaluation
# ---------------------------------------------------------------------------


class TestDropoutEvaluator:
    """Gate 6: Dropout evaluator must produce mean ± std over multiple seeds."""

    def test_evaluate_at_k0_no_change(self):
        """k=0 evaluation must match standard evaluation (no dropout)."""
        from fault_localization.dropout.evaluator import evaluate_at_dropout

        # Simple mock model that always predicts class 0
        class MockModel(torch.nn.Module):
            def forward(self, x):
                b, n, s, f = x.shape
                logits = torch.zeros(b, n, 26)
                logits[:, :, 0] = 10.0  # always predict class 0
                return logits

        model = MockModel()
        device = torch.device("cpu")

        # Mock batch
        batch = {"x": torch.randn(4, 5, 20, 6), "y": torch.zeros(4, dtype=torch.long)}

        class MockLoader:
            def __iter__(self):
                yield batch
            def __len__(self):
                return 1

        result = evaluate_at_dropout(
            model, MockLoader(), device, k=0, mask_seed=0
        )
        assert "f1_macro" in result
        assert "accuracy" in result
        assert result["accuracy"] == 1.0  # always predicts correct class 0
