"""
Constants for the IEEE 123-bus fault localization pipeline.

All values are reproduced from [BASE]:
  Karabulut, B., Manna, C., Develder, C. (2026).
  Robustness of Spatio-temporal Graph Neural Networks for Fault Location
  in Partially Observable Distribution Grids. arXiv:2604.20403.
"""

# ---------------------------------------------------------------------------
# Fault and measurement locations (reproduced from [BASE] Table 1 / Fig. 2)
# ---------------------------------------------------------------------------

FAULT_BUSES: list[int] = [
    7, 13, 18, 21, 25, 29, 35, 42, 47, 51,
    53, 55, 57, 62, 65, 72, 80, 83, 86, 89,
    93, 97, 99, 101, 108,
]
"""25 predefined fault-injection buses (sorted order)."""

PMU_BUSES: list[int] = [
    1, 6, 13, 18, 25, 26, 33, 35, 36, 45,
    47, 54, 60, 69, 75, 76, 81, 84, 91, 97,
    103, 105, 109, 113, 300,
]
"""25 μPMU measurement buses (sorted order). Bus 300 is a load/auxiliary bus."""

assert len(FAULT_BUSES) == 25, "Must have exactly 25 fault buses"
assert len(PMU_BUSES) == 25, "Must have exactly 25 μPMU buses"

# ---------------------------------------------------------------------------
# Fault taxonomy (11 types)
# ---------------------------------------------------------------------------

FAULT_TYPES: list[str] = [
    # Line-to-ground (LG)
    "AG", "BG", "CG",
    # Line-to-line (LL)
    "AB", "BC", "CA",
    # Double line-to-ground (LLG)
    "ABG", "BCG", "CAG",
    # Three-phase
    "ABC", "ABCG",
]
"""11 fault-type codes reproduced from [BASE]."""

assert len(FAULT_TYPES) == 11, "Must have exactly 11 fault types"

# Fault category groups (for reference and validation)
LG_TYPES = ["AG", "BG", "CG"]
LL_TYPES = ["AB", "BC", "CA"]
LLG_TYPES = ["ABG", "BCG", "CAG"]
THREE_PHASE_TYPES = ["ABC", "ABCG"]

# ---------------------------------------------------------------------------
# Fault resistance levels (Ω)
# ---------------------------------------------------------------------------

FAULT_RESISTANCES: list[float] = [0.1, 1.0, 10.0]
"""Three fault-resistance levels (Ohms) reproduced from [BASE]."""

assert len(FAULT_RESISTANCES) == 3, "Must have exactly 3 resistance levels"

# ---------------------------------------------------------------------------
# Scenario counts (derived)
# ---------------------------------------------------------------------------

NUM_FAULT_BUSES: int = len(FAULT_BUSES)         # 25
NUM_FAULT_TYPES: int = len(FAULT_TYPES)         # 11
NUM_RESISTANCES: int = len(FAULT_RESISTANCES)   # 3
NUM_SCENARIOS: int = NUM_FAULT_BUSES * NUM_FAULT_TYPES * NUM_RESISTANCES  # 825

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

FEEDER_NOMINAL_KV: float = 4.16    # kV, line-to-line nominal voltage
FEEDER_FREQUENCY_HZ: float = 60.0  # Hz

LOAD_MULTIPLIER_LOW: float = 0.5
LOAD_MULTIPLIER_HIGH: float = 1.3
LOAD_MULTIPLIER_RANGE: tuple[float, float] = (LOAD_MULTIPLIER_LOW, LOAD_MULTIPLIER_HIGH)

FAULT_DURATION_MS: int = 20         # ms — duration of the injected fault
SAMPLING_RESOLUTION_MS: int = 1     # ms — PMU recording resolution
PRE_FAULT_MS: int = 40              # ms — pre-fault recording window
TOTAL_RECORD_MS: int = PRE_FAULT_MS + FAULT_DURATION_MS  # 60 ms total

FAULT_START_STEP: int = PRE_FAULT_MS // SAMPLING_RESOLUTION_MS  # step index 40

# ---------------------------------------------------------------------------
# Sliding-window parameters
# ---------------------------------------------------------------------------

WINDOW_LENGTH: int = 20   # timesteps (20 ms)
WINDOW_STRIDE: int = 1    # ms

# Approved benchmark protocol: retain the first 40 stride-one windows (starts
# 0..39) and deliberately drop the endpoint window starting at 40.
WINDOWS_PER_EVENT: int = 40
THEORETICAL_WINDOWS_PER_EVENT: int = WINDOWS_PER_EVENT

# ---------------------------------------------------------------------------
# Simulation run count
# ---------------------------------------------------------------------------

RUNS_PER_SCENARIO: int = 100
"""
Full factorial reproduction protocol specified in DATASET.md:
  825 scenarios × 100 independently seeded load draws per scenario.

With the approved 40-window convention this produces 3,300,000 windows.
The paper's reported 2.5 M total is arithmetically incompatible with both
825 scenarios and 100 runs per scenario, so it is retained as a source
discrepancy rather than used to reduce the corpus silently.
"""

TARGET_TOTAL_WINDOWS: int = NUM_SCENARIOS * RUNS_PER_SCENARIO * WINDOWS_PER_EVENT

# ---------------------------------------------------------------------------
# Dataset label space
# ---------------------------------------------------------------------------

NUM_CLASSES: int = 26  # 25 fault-location classes + 1 no-fault class
NO_FAULT_CLASS: int = 0
# Classes 1–25 correspond to sorted FAULT_BUSES index + 1

# Mapping: fault bus → class label (1-indexed; class 0 = no fault)
FAULT_BUS_TO_LABEL: dict[int, int] = {
    bus: idx + 1 for idx, bus in enumerate(sorted(FAULT_BUSES))
}
LABEL_TO_FAULT_BUS: dict[int, int] = {v: k for k, v in FAULT_BUS_TO_LABEL.items()}

# ---------------------------------------------------------------------------
# Feature dimensions
# ---------------------------------------------------------------------------

NUM_PMU_NODES: int = len(PMU_BUSES)   # N = 25
NUM_PHASES: int = 3                    # Number of voltage phase channels (V_A, V_B, V_C)
NUM_MASK_CHANNELS: int = 3             # Binary availability mask channels (one per phase)
NUM_FEATURES: int = NUM_PHASES + NUM_MASK_CHANNELS  # F = 6: voltages (0-2) + mask (3-5)
NUM_TIMESTEPS: int = TOTAL_RECORD_MS   # T = 60 per raw recording

# Per-window tensor shape: [N=25, S=20, F=6]
# Channels 0-2: per-unit RMS voltages; channels 3-5: binary availability mask.
WINDOW_TENSOR_SHAPE: tuple[int, int, int] = (NUM_PMU_NODES, WINDOW_LENGTH, NUM_FEATURES)

# ---------------------------------------------------------------------------
# Feeder configurations
# ---------------------------------------------------------------------------

FEEDER_CONFIGS: list[str] = ["default", "green"]
"""Supported feeder topology configurations."""

# Green (reconfigured) feeder: open tie switch 60-160, close switch 54-94.
# Per paper §4.3.  Exact OpenDSS element names discovered at runtime by
# scripts/generate_pmu_metadata.py and stored in config/pmu_phase_metadata.yaml.
GREEN_FEEDER_OPEN_SWITCH_BUSES: tuple[int, int] = (60, 160)
GREEN_FEEDER_CLOSE_SWITCH_BUSES: tuple[int, int] = (54, 94)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

GLOBAL_RANDOM_SEED: int = 42


def deterministic_seed(fault_bus: int, fault_type: str, resistance: float, run_id: int) -> int:
    """Compute a deterministic integer seed for a (scenario, run) tuple.

    Uses a simple hash to produce a unique seed per (bus, type, resistance, run).
    """
    resistance_idx = FAULT_RESISTANCES.index(resistance)
    type_idx = FAULT_TYPES.index(fault_type)
    bus_idx = FAULT_BUSES.index(fault_bus)
    # Encode as a unique integer — maximum value well within np.random seed range
    return (
        bus_idx * NUM_FAULT_TYPES * NUM_RESISTANCES * 10_000
        + type_idx * NUM_RESISTANCES * 10_000
        + resistance_idx * 10_000
        + run_id
    )
