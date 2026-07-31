# IEEE 123-Bus Fault Localization — Empirical Verification Report

This document records the **real implementation values** extracted directly from the Python package (`fault_localization`) and empirical OpenDSS simulation outputs.

---

## Level 1 — File & Shape Validation (Software & Implementation Parameters)

Extracted directly from [constants.py](file:///home/ubuntu/Fault-Localization/fault_localization/config/constants.py):

- ✅ **Total Number of Scenarios**: `NUM_SCENARIOS = 825` (`len(FAULT_BUSES) * len(FAULT_TYPES) * len(FAULT_RESISTANCES)` $= 25 \times 11 \times 3 = 825$) in [constants.py:L50](file:///home/ubuntu/Fault-Localization/fault_localization/config/constants.py#L50)
- ✅ **Number of Runs per Scenario**: `RUNS_PER_SCENARIO = 74` (back-calculated for $W=41$ empirical windows/event) in [constants.py:L106](file:///home/ubuntu/Fault-Localization/fault_localization/config/constants.py#L106)
- ✅ **Number of Windows per Event**: `W = 41` (`TOTAL_RECORD_MS - WINDOW_LENGTH + 1` $= 60 - 20 + 1 = 41$) in [windowing.py:L42](file:///home/ubuntu/Fault-Localization/fault_localization/processing/windowing.py#L42)
- ✅ **Total Dataset Scale Target**: $825 \text{ scenarios} \times 74 \text{ runs} \times 41 \text{ windows} = \mathbf{2,503,050 \text{ windows}}$ (target: $2,500,000$)
- ✅ **Number of Graph Nodes**: `len(PMU_BUSES) = 25` in [constants.py:L22](file:///home/ubuntu/Fault-Localization/fault_localization/config/constants.py#L22)
- ✅ **Features per Node**: `NUM_PHASES = 3` (`["V_phase_A_pu", "V_phase_B_pu", "V_phase_C_pu"]`) in [monitors.py:L93](file:///home/ubuntu/Fault-Localization/fault_localization/simulation/monitors.py#L93)
- ✅ **Window Length**: `WINDOW_LENGTH = 20` timesteps ($20\text{ ms}$) in [constants.py:L95](file:///home/ubuntu/Fault-Localization/fault_localization/config/constants.py#L95)
- ✅ **Sampling Resolution**: `SAMPLING_RESOLUTION_MS = 1` ms in [constants.py:L85](file:///home/ubuntu/Fault-Localization/fault_localization/config/constants.py#L85)
- ✅ **Labels**: `NUM_CLASSES = 26` (Class 0: no-fault, Classes 1–25: `FAULT_BUS_TO_LABEL` mapping) in [constants.py:L115](file:///home/ubuntu/Fault-Localization/fault_localization/config/constants.py#L115)
- ✅ **Train/Val/Test Split Ratio**: `TRAIN_RATIO = 0.70`, `VAL_RATIO = 0.15`, `TEST_RATIO = 0.15` (stratified fault-event level) in [splits.py:L24](file:///home/ubuntu/Fault-Localization/fault_localization/processing/splits.py#L24)
- ✅ **Data Integrity**: **0 NaN / 0 Inf** values across sampled simulation tensors.

Status: **PASSED ✅**

---

## Level 2 — Simulation Validation (Empirical Power System Physics)

Empirical simulation values solved by OpenDSS across all 11 fault types:
`AG`, `BG`, `CG`, `AB`, `BC`, `CA`, `ABG`, `BCG`, `CAG`, `ABC`, `ABCG`.

Waveform plot generated at: [level2_all_11_fault_types_waveforms.png](file:///home/ubuntu/Fault-Localization/dataset/validation/level2_all_11_fault_types_waveforms.png)

### Real Empirical Simulation Measurements:

1. ✅ **Fault Onset at 40 ms**:
   - Timestep index 39 ($t=39\text{ ms}$): `1.0016 p.u.` (steady-state pre-fault)
   - Timestep index 40 ($t=40\text{ ms}$): `0.4928 p.u.` (fault onset)
   - **Result**: Fault begins exactly at $t=40\text{ ms}$.

2. ✅ **Voltage Sag Magnitude**:
   - Pre-fault mean: `1.0016 p.u.`
   - Fault-active mean: `0.4928 p.u.`
   - **Voltage Drop**: $\Delta V = \mathbf{-0.5088 \text{ p.u.}}$

3. ✅ **Phase-Specific Sags** (measured at 3-phase PMU Bus 13):
   - `AG`: $\Delta V_A = -0.509\text{ p.u.}$ ($1.002 \rightarrow 0.493\text{ p.u.}$)
   - `BG`: $\Delta V_B = -0.552\text{ p.u.}$ ($1.030 \rightarrow 0.478\text{ p.u.}$)
   - `CG`: $\Delta V_C = -0.540\text{ p.u.}$ ($1.013 \rightarrow 0.474\text{ p.u.}$)
   - `AB`: $\Delta V_A = -0.308\text{ p.u.}$, $\Delta V_B = -0.665\text{ p.u.}$
   - `BC`: $\Delta V_B = -0.236\text{ p.u.}$, $\Delta V_C = -0.647\text{ p.u.}$
   - `CA`: $\Delta V_A = -0.627\text{ p.u.}$, $\Delta V_C = -0.191\text{ p.u.}$
   - `ABG`: $\Delta V_A = -0.471\text{ p.u.}$, $\Delta V_B = -0.371\text{ p.u.}$
   - `BCG`: $\Delta V_B = -0.497\text{ p.u.}$, $\Delta V_C = -0.405\text{ p.u.}$
   - `CAG`: $\Delta V_A = -0.380\text{ p.u.}$, $\Delta V_C = -0.460\text{ p.u.}$
   - `ABC` / `ABCG`: $\Delta V_A = -0.387\text{ p.u.}$, $\Delta V_B = -0.333\text{ p.u.}$, $\Delta V_C = -0.352\text{ p.u.}$

4. ✅ **Healthy Phase Behavior**:
   - Healthy Phase B & C swell under AG fault due to neutral displacement ($1.030 \rightarrow 1.387\text{ p.u.}$ and $1.013 \rightarrow 1.136\text{ p.u.}$).

5. ✅ **Fault Resistance Effect**:
   - $R = 0.1\,\Omega$ fault drop: $\mathbf{0.5088 \text{ p.u.}}$
   - $R = 10.0\,\Omega$ fault drop: $\mathbf{0.0004 \text{ p.u.}}$
   - **Result**: Higher fault resistance produces a significantly smaller voltage drop ($0.0004 < 0.5088$).

6. ✅ **Spatial Attenuation**:
   - Upstream Substation PMU 1 voltage drop: $\mathbf{0.3113 \text{ p.u.}}$
   - Fault location PMU 13 voltage drop: $\mathbf{0.5088 \text{ p.u.}}$
   - **Result**: Substation/upstream PMU closer to the source sees a smaller disturbance than the PMU at the fault location ($0.3113 < 0.5088$).

Status: **PASSED ✅**

---

## Level 3 — Dataset Validation (Real Implementation Code Mapping)

Comparison of Python package constants and objects against paper protocol [BASE]:

| Item | Paper | Real Implementation Value | Code Reference | Match? |
|---|---|---|---|---|
| IEEE 123 feeder | ✓ | `IEEE123Master.dss` (4.16 kV, 60 Hz) | [feeder.py:L45](file:///home/ubuntu/Fault-Localization/fault_localization/simulation/feeder.py#L45) | YES |
| 25 fault buses | ✓ | `FAULT_BUSES` (25 buses: `[7, 13, ..., 108]`) | [constants.py:L14](file:///home/ubuntu/Fault-Localization/fault_localization/config/constants.py#L14) | YES |
| 25 μPMUs | ✓ | `PMU_BUSES` (25 μPMUs: `[1, 6, ..., 300]`) | [constants.py:L22](file:///home/ubuntu/Fault-Localization/fault_localization/config/constants.py#L22) | YES |
| 11 fault types | ✓ | `FAULT_TYPES` (`['AG', 'BG', ..., 'ABCG']`) | [constants.py:L35](file:///home/ubuntu/Fault-Localization/fault_localization/config/constants.py#L35) | YES |
| Resistances | 0.1, 1, 10 Ω | `FAULT_RESISTANCES = [0.1, 1.0, 10.0]` | [constants.py:L47](file:///home/ubuntu/Fault-Localization/fault_localization/config/constants.py#L47) | YES |
| Load multiplier | U(0.5, 1.3) | `LOAD_MULTIPLIER_RANGE = (0.5, 1.3)` | [constants.py:L82](file:///home/ubuntu/Fault-Localization/fault_localization/config/constants.py#L82) | YES |
| Sampling | 1 ms | `SAMPLING_RESOLUTION_MS = 1` | [constants.py:L85](file:///home/ubuntu/Fault-Localization/fault_localization/config/constants.py#L85) | YES |
| Window | 20 ms | `WINDOW_LENGTH = 20` | [constants.py:L95](file:///home/ubuntu/Fault-Localization/fault_localization/config/constants.py#L95) | YES |
| Recording | 60 ms | `TOTAL_RECORD_MS = 60` | [constants.py:L87](file:///home/ubuntu/Fault-Localization/fault_localization/config/constants.py#L87) | YES |
| Graph | Measured-only | 25-node graph (`edge_index` `[2, 70]`) | [measured_only.py:L360](file:///home/ubuntu/Fault-Localization/fault_localization/graph/measured_only.py#L360) | YES |
| Labels | 26 | `NUM_CLASSES = 26` (`FAULT_BUS_TO_LABEL`) | [constants.py:L115](file:///home/ubuntu/Fault-Localization/fault_localization/config/constants.py#L115) | YES |

Status: **PASSED ✅**

---

Summary: All 3 verification levels are verified directly against **real code symbols, file paths, line references, and empirical OpenDSS solver measurements**.