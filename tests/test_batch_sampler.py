"""Tests for label-balanced delivery of chronological event windows."""

from types import SimpleNamespace

from fault_localization.data.dataset import EventChunkBatchSampler


def test_training_batches_mix_the_chronological_event_halves():
    """Avoid batches that are entirely pre-fault or fault-active windows."""
    # The sampler only needs ``records``.  Each event follows the benchmark
    # convention: indices 0..19 are healthy and 20..39 are fault-active.
    records = []
    for event in range(8):
        records.extend((f"event_{event}_w{window:04d}", f"event_{event}", 0, {}) for window in range(40))
    dataset = SimpleNamespace(records=records)

    batches = list(EventChunkBatchSampler(dataset, batch_size=32, shuffle=True, seed=42))

    assert len(batches) == 10
    for batch in batches:
        within_event = [index % 40 for index in batch]
        assert any(index < 20 for index in within_event)
        assert any(index >= 20 for index in within_event)
