Training — complete ✅
Checkpoints	Seeds present
GRU, RGCN, RGSAGE-Mean, RGSAGE-Max, RGATv2	42, 43, 44 (all 15)
Dropout-aware RGSAGE-Max	42, 43, 44
SDR-STGNN	42, 43, 44

What's actually left — an inference sweep, not training
The only gap is the sensor-dropout degradation sweep (loads a checkpoint, drops k sensors, measures F1 — no gradient updates) for the four secondary baselines at the extra seeds:

Degradation curves	Coverage
RGSAGE-Max, Dropout-aware, SDR-STGNN	✅ seeds 42/43/44 (already 3-seed)
GRU, RGCN, RGSAGE-Mean, RGATv2	⚠️ seed 42 only — missing 43 & 44
So the leftover = 8 model-seed evaluations (4 baselines × seeds 43, 44). This is exactly the run_dropout_study.py --seeds 42 43 44 job you stopped earlier — it was running cleanly, no errors, ~1–2 h total on the T1000.


