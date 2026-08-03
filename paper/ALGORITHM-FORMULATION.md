1. Which algorithm(s) — and there are three candidates
Ranked by value for this paper:

Candidate	What it formalizes	Verdict
A. SDR-STGNN gated forward pass	availability derivation → per-hop gated aggregation → soft-vote	Best single algorithm — captures the L-hop loop that Eqs. (1)–(2) don't show
B. Dropout-aware training loop	per-batch: draw k, mask nodes, forward, weighted-CE step	Strong second — it's the headline contribution and is genuinely procedural
C. Paired degradation-curve evaluation	for each k, for each mask seed: mask → infer → macro-F1 → aggregate	Optional — useful but prose already covers it well
My recommendation: for a Procedia page budget, one algorithm, and make it A (SDR-STGNN forward pass) — it complements the two equations by showing the hop loop + gating + voting end-to-end. If you have room for a second and want to foreground the "training, not architecture" story, add B.

2. Where to put it
A → inside §SDR-STGNN (sec:sdr), right after the gated-aggregation equation (Eq. eq:gate). Reference it in the "Gated aggregation" paragraph before it appears ("…summarized in Algorithm 1").
B → inside §Dropout-Aware Training (sec:dropout-aware), after the "we wrap the training loader…" paragraph.
C → inside §Sensor-Dropout Evaluation Protocol, after the "Dropout sweep" paragraph.
Use [t] (top) float placement; algorithms read best at a column top.

3. How (LaTeX mechanics) — one caution
You must add a package — the preamble currently loads none. Standard choice: \usepackage{algorithm} + \usepackage{algpseudocode} (algorithmicx), or \usepackage[ruled,vlined]{algorithm2e}. This is the one risk: your rules say don't add packages without a compile test, and Overleaf/Procedia (ecrc) occasionally clashes — so add it and do one clean compile before trusting it.
Caption goes above the algorithm body (like tables), with a \label{alg:sdr}; cite as Algorithm~\ref{alg:sdr}.
Keep line count low (~10–14 lines) so it fits a single column.

4. What Algorithm A would contain (steps, so you can decide scope)
Input: window features x ∈ [N,S,6], edge set, hop count L=4
Derive availability a from mask channels (Eq. eq:avail)
h ← GRU-encoder(x)
for l = 1…L: gated max-aggregate over available closed neighbourhood → concat gated-self ‖ agg → ReLU(Wₗ·[…])
logits ← classifier(h); soft-vote over 25 nodes
Output: graph-level class

5. Cautions
Don't duplicate the equations verbatim — the algorithm should add the loop/control flow the equations lack, not restate them. If it would just re-list Eqs. (1)–(2), skip it.
Notation must match the equations exactly (aᵢ, agg_j, 𝒩(j), h_j, Wₗ) — inconsistent symbols between Eq. and Algorithm is a classic reviewer flag.
Page budget: Procedia is tight; one algorithm earns its place, three would read as padding.
Since the paper reframes SDR-STGNN as an ablation, keep the algorithm caption neutral ("SDR-STGNN forward pass"), not "our proposed superior method."
When you want it built, tell me which (A, B, or both) and I'll add the package + algorithm block and do a compile check.