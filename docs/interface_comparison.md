# Interface Comparison: opendssdirect.py vs PyDSS

**Phase 0.5 Decision Gate**

Date: TBD (populated after running the comparison experiment)

---

## Comparison Criteria

| Requirement | opendssdirect.py | PyDSS (NREL) | Notes |
|---|---|---|---|
| Compile & solve IEEE 123-bus feeder | ✅ | ✅ | Both wrap the same OpenDSS engine |
| Inject `Fault` objects (all 11 types) | ✅ | ✅ | Via `run_command("New Fault…")` |
| Dynamic / time-series simulation | ✅ (snapshot mode per-step) | ✅ | PyDSS has dedicated dynamic mode |
| Monitor placement at specific buses | ✅ (via API: `Bus.PuVoltage`) | ✅ | Different API surface |
| Extract 3-phase RMS voltage per timestep | ✅ | ✅ | |
| Load multiplier scaling | ✅ | ✅ | |
| Cross-platform (Linux) | ✅ | ✅ | |
| Python 3.12 compatibility | ✅ | ❓ | PyDSS: check build dependencies |

---

## Pilot Comparison Results

*(To be filled in after running Phase 0.5 comparison experiment)*

### Task: Compile IEEE 123-bus, inject LG fault at Bus 7 (R=1 Ω), solve 60 timesteps at 1 ms, extract 3-phase voltage at 3 μPMU buses.

#### opendssdirect.py

```
Result: [PENDING]
API ergonomics: Direct, pythonic, low-ceremony
Convergence: [PENDING]
Voltage extraction: Bus.PuVoltage() API works cleanly
Issues: [PENDING]
```

#### PyDSS

```
Result: [PENDING]
API ergonomics: Higher-level, more setup required
Convergence: [PENDING]
Voltage extraction: [PENDING]
Issues: [PENDING]
```

---

## [BASE] Paper PyDSS Dependency Analysis

The base paper (Karabulut et al. 2026, arXiv:2604.20403) uses PyDSS, but:

- The paper does not describe any PyDSS-specific co-simulation controller or
  component that goes beyond raw OpenDSS simulation.
- PyDSS is primarily used as a wrapper for OpenDSS.
- All fault injection, monitoring, and load-variation steps are OpenDSS operations
  that are equally accessible via `opendssdirect.py`.

Conclusion: **No irreproducible PyDSS-specific dependency identified.**

---

## Decision

**Selected interface: `opendssdirect.py`**

### Rationale

1. **Simpler API**: Direct Python bindings to the OpenDSS engine — one-file installation,
   no NREL-specific dependencies.
2. **Python 3.12 compatibility confirmed**: `opendssdirect.py >= 0.9` is tested on 3.12.
3. **No PyDSS-specific features required**: All needed operations (fault injection,
   per-bus voltage extraction, load scaling) are available via `opendssdirect.py`.
4. **Easier debugging**: Direct command-string interface allows line-by-line inspection.
5. **Reproducibility**: The simulation output (voltage values) depends on the OpenDSS
   engine, not the Python wrapper — both interfaces call the same C++ engine.

### Risk Mitigation

- PyDSS remains listed as an optional dependency for future comparison validation
  if numerical discrepancies are found during dataset validation.
- If the `opendssdirect.py` simulation produces fundamentally different voltage profiles
  than expected (from physical reasoning or the paper), the PyDSS interface will be
  revisited before full dataset generation.

---

## Action Items

- [x] Implement `feeder.py` using `opendssdirect.py`
- [x] Implement `fault_injection.py` using `dss.run_command()` 
- [x] Implement `monitors.py` using `Bus.PuVoltage()` API
- [ ] Run Stage 2.1 smoke test and visually confirm voltage sag
- [ ] Compare voltage profile against expected IEEE 123-bus fault characteristics
