"""Figures for the Phase-2 review deck (PPT2/PPT-CONTENT.md).

Every number is copied from a file in results/ or a closed record in PC-STGNN/; sources are
noted beside each constant. Run:  uv run python PPT2/make_figures.py
Output: PPT2/figures/*.png (300 dpi, 16:9-friendly).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# Palette: dataviz reference instance (light mode).
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"
GOOD, CRIT = "#0ca30c", "#d03b3b"
TINT = {BLUE: "#eaf2fc", ORANGE: "#fdefe9", AQUA: "#e7f6f0", VIOLET: "#edebf7",
        YELLOW: "#fdf5e0", RED: "#fcecec", MUTED: "#f1f1ef"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def box(ax, x, y, w, h, title, body="", color=BLUE, title_size=12, body_size=10, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=TINT.get(color, "#ffffff"), ec=color, lw=1.6,
                                ls="--" if dashed else "-"))
    ty = y + h - 0.03 if body else y + h / 2
    ax.text(x + w / 2, ty, title, ha="center", va="top" if body else "center",
            fontsize=title_size, weight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 0.075, body, ha="center", va="top", fontsize=body_size,
                color=INK2, linespacing=1.35)


def arrow(ax, p, q, color=INK2, lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color=color,
                                 lw=lw, ls=ls, shrinkA=2, shrinkB=2))


def canvas(w=16, h=9):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------------------------------------
# 1. Collapse diagnosis and fix (S4.3 -> S4.3c -> S4.3d)
# Sources: results/pcstgnn_tuning/fusion_decomposition_val_k0.json (S4.3 selected arm),
#          results/pcstgnn_noisediag/diagnostic_summary.json (S4.3c lambda1=0, epoch 5),
#          results/pcstgnn_auxdiag/diagnostic_summary.json (S4.3d gamma=1, epoch 5).
# ---------------------------------------------------------------------------------------------
def fig_collapse():
    stages = ["S4.3\nnoise-free tuning\n(best arm, 15 ep)",
              "S4.3ᶜ\n+ noise augmentation\n(λ₁ = 0, epoch 5)",
              "S4.3ᵈ\n+ auxiliary loss γ = 1\n(λ₁ = 0, epoch 5)"]
    fused = [0.8533, 0.8788, 0.8794]
    net = [0.0256, 0.078, 0.5708]
    nofault = ["100 %", "54.5 %", "53.5 %"]
    fig, ax = plt.subplots(figsize=(12, 6.2))
    x = range(3)
    w = 0.34
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], fused, w, color=BLUE, label="Fused PC-STGNN (network + physics)")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], net, w, color=ORANGE, label="Network branch alone (physics removed)")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.015, f"{r.get_height():.3f}",
                    ha="center", va="bottom", fontsize=11, color=INK, weight="bold")
    ax.axhline(0.10, color=CRIT, lw=1.4, ls="--", label="Collapse guard: network-only F1 must exceed 0.10")
    ax.axhline(0.0256, color=MUTED, lw=1.2, ls=":", label="Trivial always-\"no fault\" classifier = 0.026")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{st}\n\nnetwork votes \"no fault\": {nf}" for st, nf in zip(stages, nofault)],
                       fontsize=10.5, color=INK)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(-0.7, 2.7)
    ax.set_ylabel("Validation macro-F1 (26-class, k = 0)")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", frameon=False, fontsize=10.5, ncol=2)
    ax.set_title("The fused score hid a collapse: the network had handed localization to physics",
                 loc="left", fontsize=13.5, weight="bold", color=INK, pad=14)
    fig.text(0.01, -0.1, "Full validation split (248,000 windows), seed 42, clean inputs. Fused F1 is ~0.85–0.88 in every stage; only the network-only bar shows "
             "whether the network learned anything.",
             fontsize=9, color=MUTED)
    save(fig, "p2_collapse_fix.png")


# ---------------------------------------------------------------------------------------------
# 2. Gantt chart -- actual Phase-2 dates from git, projected bars hatched.
# Projection: S4.3e ~13 GPU-h/arm (measured, 52 min/epoch), S4.4 ~360 GPU-h (budget doc's run
# list, re-priced at the measured noise-on rate), Stages F-H per 13_BUDGET_TIMELINE.md order.
# ---------------------------------------------------------------------------------------------
def fig_gantt():
    D = lambda m, d: date(2026, m, d)  # noqa: E731
    rows = [  # (label, start, end, status, group)
        ("Phase 1: dropout-robust STGNN + paper", D(8, 1), D(9, 21), "done", "P1"),
        ("A  Audit of Phase-1 assets  (G0)", D(9, 22), D(9, 23), "done", "P2"),
        ("B  Physics operator, 84,167 solves  (G1 5/5)", D(9, 23), D(9, 25), "done", "P2"),
        ("C  Training-free go/no-go  (G2: GO)", D(9, 25), D(9, 29), "done", "P2"),
        ("D  Injectors, imputation, ρ cache  (G3 5/5)", D(9, 26), D(9, 29), "done", "P2"),
        ("E  S4.1 null test + S4.2 gradient audit", D(9, 30), D(9, 30), "done", "P2"),
        ("E  S4.3 λ tuning, noise-free (collapse found)", D(9, 30), D(10, 4), "done", "P2"),
        ("E  S4.3ᶜ / S4.3ᵈ collapse diagnostics (fixed)", D(10, 4), D(10, 5), "done", "P2"),
        ("E  S4.3ᵉ λ re-tuning, γ = 1  (running)", D(10, 5), D(10, 9), "run", "P2"),
        ("E  S4.4 core ablation R0ⁿ–R8  → Gate 4", D(10, 10), D(10, 27), "plan", "P2"),
        ("F  Trust gate for silent sensors  → Gate 5", D(10, 28), D(11, 10), "plan", "P2"),
        ("G  Sealed OOD test, noise, model mismatch", D(11, 10), D(11, 25), "plan", "P2"),
        ("H  Analysis, figures, paper, thesis", D(11, 18), D(12, 15), "plan", "P2"),
    ]
    color = {"done": BLUE, "run": ORANGE, "plan": "#c9dcf3"}
    fig, ax = plt.subplots(figsize=(15, 7.2))
    for i, (lab, s, e, st, g) in enumerate(rows):
        y = len(rows) - 1 - i
        start = mdates.date2num(s)
        dur = max(mdates.date2num(e) - start, 0.9)
        ax.barh(y, dur, left=start, height=0.58, color=color[st],
                hatch="///" if st == "plan" else None, edgecolor=BLUE if st == "plan" else color[st],
                lw=0.8)
        tag = {"done": "done", "run": "running", "plan": "planned"}[st]
        ax.text(start + dur + 0.6, y, tag, va="center", fontsize=9,
                color=INK2 if st != "run" else ORANGE, weight="bold" if st == "run" else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=10.5, color=INK)
    today = mdates.date2num(D(10, 5))
    ax.axvline(today, color=CRIT, lw=1.4, ls="--")
    ax.text(today + 0.4, len(rows) - 0.35, "today, 5 Oct 2026", color=CRIT, fontsize=9.5)
    ax.axhline(len(rows) - 1.5, color=GRID, lw=1)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.grid(True, which="minor", color=GRID, lw=0.6)
    ax.xaxis.grid(True, which="major", color=MUTED, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(mdates.date2num(D(7, 28)), mdates.date2num(D(12, 22)))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Patch(color=BLUE, label="completed"), Patch(color=ORANGE, label="running now"),
                       Patch(fc="#c9dcf3", ec=BLUE, hatch="///", label="planned (projected)")],
              loc="lower left", frameon=False, ncol=3, fontsize=10)
    ax.set_title("Semester-3 plan: PC-STGNN, stage-gated (Stages A–D closed; Stage E in progress)",
                 loc="left", fontsize=13.5, weight="bold", color=INK, pad=12)
    fig.text(0.01, -0.03, "Completed dates from git history. Planned dates are projections from measured "
             "GPU rates on one T1000 (S4.4 ≈ 360 GPU-h ≈ 15 days); adjust to the semester calendar.",
             fontsize=9, color=MUTED)
    save(fig, "p2_gantt.png")


# ---------------------------------------------------------------------------------------------
# 3. Why PC-STGNN?  Phase-1 limitation -> evidence -> requirement -> PC-STGNN component
# ---------------------------------------------------------------------------------------------
def fig_why():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    heads = [("Phase-1 limitation", 0.02, RED), ("Measured evidence", 0.265, MUTED),
             ("What the next model needs", 0.51, YELLOW), ("PC-STGNN design response", 0.755, BLUE)]
    for t, x, c in heads:
        ax.text(x + 0.11, 0.955, t, ha="center", fontsize=13.5, weight="bold", color=INK)
    rows = [
        ("Measurement noise", "σ = 1e-2 p.u. (IEEE 1 % TVE):\nF1 0.846 → 0.345.\nDropout-aware training\ngives no benefit (−0.008 to −0.034)",
         "An evidence source whose\nconfidence scales with\nthe noise level",
         "Physics residual ρ with noise\nfloor σε² + σ₀² in denominator;\nnoise-augmented training"),
        ("Unseen fault resistance", "Trained on R ∈ {0.1, 1, 10} Ω.\nOn R ∈ {5, 20} Ω:\nnetwork 0.839 → 0.089;\nphysics PSM 0.772 → 0.539",
         "Knowledge that extends\nbeyond the training\nresistances",
         "Signature table over 18 R values\n(0.05–30 Ω), profiled:\nmin over (type, R, load)"),
        ("Worst-case / heavy\nsensor loss", "Targeted loss: AUDC 0.913 → 0.747.\nk = 10: network 0.674;\nphysics PSM stays at 0.749",
         "Use only the sensors that\nare present; check consistency\non the ones that are missing",
         "ρ computed over available\nsensors A only; λ₂ term scores\nheld-out sensors (leave-k-out)"),
        ("No physical grounding;\nhigh-impedance faults", "10 Ω slice: F1 0.676\n(vs 0.949 at 0.1 Ω).\nA linear model reaches 0.713:\nlittle headroom at k = 0",
         "Inject physics without\nadding capacity, and\nwithout leaking labels",
         "β fusion z − β·ρ: +1 parameter;\nρ reads the input only;\nβ = 0 ⇒ exactly Phase 1"),
    ]
    h, gap, y0 = 0.19, 0.03, 0.92
    for i, (lim, ev, req, resp) in enumerate(rows):
        y = y0 - (i + 1) * h - i * gap + 0.01
        for (x, txt, c, bold) in [(0.02, lim, RED, True), (0.265, ev, MUTED, False),
                                  (0.51, req, YELLOW, False), (0.755, resp, BLUE, True)]:
            ax.add_patch(FancyBboxPatch((x, y), 0.225, h, boxstyle="round,pad=0.004,rounding_size=0.01",
                                        fc=TINT[c], ec=c, lw=1.5))
            ax.text(x + 0.1125, y + h / 2, txt, ha="center", va="center",
                    fontsize=(12.5 if x < 0.1 else 11.2) if bold else 11, weight="bold" if bold else None,
                    color=INK if bold else INK2, linespacing=1.35)
        for x in (0.245, 0.49, 0.735):
            arrow(ax, (x, y + h / 2), (x + 0.02, y + h / 2))
    ax.text(0.02, 0.005, "Evidence: Phase-1 results (test split, 3 seeds) and Phase-2 Gate-2 pilot (PSM, validation / "
            "OOD-pilot). Responses are design decisions; whether they WORK is the open Gate-4 question.",
            fontsize=9.5, color=MUTED)
    save(fig, "p2_why_pcstgnn.png")


# ---------------------------------------------------------------------------------------------
# 4. Phase 1 -> Phase 2 transition
# ---------------------------------------------------------------------------------------------
def fig_transition():
    fig, ax = canvas(16, 7.5)
    ax.text(0.17, 0.93, "PHASE 1 — groundwork (complete)", ha="center", fontsize=15, weight="bold", color=BLUE)
    ax.text(0.83, 0.93, "PHASE 2 — PC-STGNN (in progress)", ha="center", fontsize=15, weight="bold", color=VIOLET)
    p1 = [("Problem", "locate the faulted bus, 25 of\n123 buses measured"),
          ("Dataset + graph", "1.65 M windows, 25-node\nmeasured-only graph"),
          ("5 STGNN baselines", "best RGSAGE-Max 0.846"),
          ("Sensor-dropout study", "k = 10: 0.846 → 0.048"),
          ("Dropout-aware training", "Ret@10 5.7 % → 80.3 %")]
    p2 = [("Physics operator", "84,167 OpenDSS solves ✓"),
          ("Go / no-go test", "physics helps where\nnetwork fails ✓"),
          ("PC-STGNN built", "null test + gradient audit ✓"),
          ("Training / tuning", "collapse found and fixed;\nre-tuning running"),
          ("Evaluation", "ablation, OOD, noise:\nplanned")]
    for i, (t, b) in enumerate(p1):
        y = 0.77 - i * 0.165
        box(ax, 0.04, y, 0.26, 0.13, t, b, BLUE, 12, 10)
        if i:
            arrow(ax, (0.17, y + 0.165 - 0.0), (0.17, y + 0.13))
    for i, (t, b) in enumerate(p2):
        y = 0.77 - i * 0.165
        c = VIOLET if i < 3 else (ORANGE if i == 3 else MUTED)
        box(ax, 0.70, y, 0.26, 0.13, t, b, c, 12, 10, dashed=i == 4)
        if i:
            arrow(ax, (0.83, y + 0.165), (0.83, y + 0.13))
    box(ax, 0.375, 0.36, 0.25, 0.42, "Limitations found in Phase 1",
        "• noise: 0.846 → 0.345,\n   training fix does not help\n\n• unseen resistance:\n   0.839 → 0.089\n\n"
        "• worst-case sensor loss:\n   AUDC 0.913 → 0.747\n\n• network uses no physics", RED, 12.5, 10.5)
    arrow(ax, (0.30, 0.45), (0.375, 0.55), RED, 2.2)
    arrow(ax, (0.625, 0.55), (0.70, 0.77 + 0.065), RED, 2.2)
    ax.text(0.5, 0.25, "Gate 2 evidence: physics (PSM) holds where the network fails —\n"
            "unseen R: 0.539 vs 0.089   ·   k = 10: 0.749 vs 0.674", ha="center", fontsize=11, color=INK2)
    save(fig, "p2_transition.png")


# ---------------------------------------------------------------------------------------------
# 5. PC-STGNN high-level architecture (simplified, incl. the gamma term)
# ---------------------------------------------------------------------------------------------
def fig_arch_simple():
    fig, ax = canvas(16, 8)
    box(ax, 0.02, 0.42, 0.15, 0.22, "µPMU window", "25 nodes × 20 ms\n× 6 channels\n(3 |V| + 3 masks)", MUTED, 12.5, 10.5)
    box(ax, 0.22, 0.60, 0.15, 0.20, "Temporal", "GRU-128 per node\n(shared weights)", ORANGE, 12.5, 10.5)
    box(ax, 0.42, 0.60, 0.15, 0.20, "Spatial", "4 × RGSAGE-Max\n25-node graph", ORANGE, 12.5, 10.5)
    box(ax, 0.62, 0.60, 0.13, 0.20, "Head", "per-node logits z\n[25 × 26]", ORANGE, 12.5, 10.5)
    box(ax, 0.22, 0.17, 0.35, 0.24, "Physics residual ρ  (fixed, no gradient)",
        "match measured voltages against 84,167 simulated\nfault signatures, best (type, R, load) per class\n"
        "only the sensors that are present · noise-aware", BLUE, 12.5, 10.5)
    box(ax, 0.02, 0.03, 0.15, 0.15, "Signature table", "OpenDSS, offline\n26 × 3,366 × 75", BLUE, 11.5, 9.5)
    box(ax, 0.80, 0.42, 0.17, 0.24, "Fusion + vote", "z′ = z − β·ρ\n(β: 1 learned scalar)\nsoft vote over nodes\n→ faulted bus or\n\"no fault\"", VIOLET, 12.5, 10.5)
    arrow(ax, (0.17, 0.56), (0.22, 0.70))
    arrow(ax, (0.17, 0.48), (0.22, 0.31), BLUE, ls="--")
    arrow(ax, (0.37, 0.70), (0.42, 0.70))
    arrow(ax, (0.57, 0.70), (0.62, 0.70))
    arrow(ax, (0.75, 0.70), (0.80, 0.60))
    arrow(ax, (0.57, 0.29), (0.80, 0.47), BLUE, ls="--")
    arrow(ax, (0.095, 0.18), (0.25, 0.17), BLUE, ls="--")
    ax.text(0.5, 0.955, "PC-STGNN = Phase-1 backbone (orange, unchanged) + fixed physics evidence (blue) + 1-parameter fusion (violet)",
            ha="center", fontsize=13.5, weight="bold", color=INK)
    box(ax, 0.62, 0.05, 0.35, 0.21, "Training loss",
        "CE(z′, y)  +  λ₁·E_P[ρ(A)]  +  λ₂·E_P[ρ(D)]  +  γ·CE(z, y)\n"
        "fused       physics on          physics on           network\n"
        "                available A          dropped D            alone (γ = 1)", ORANGE, 12, 10.5)
    ax.text(0.02, 0.88, "solid = learned, gradient flows   ·   dashed = fixed physics, no gradient",
            fontsize=10, color=MUTED)
    save(fig, "p2_pcstgnn_simple.png")


# ---------------------------------------------------------------------------------------------
# 6. Implementation status ladder (Implemented / Executed / Trained / Evaluated / Validated)
# ---------------------------------------------------------------------------------------------
def fig_status():
    cols = ["Implemented", "Executed", "Trained", "Evaluated\n(validation)", "Validated\n(claim-ready)"]
    rows = [  # Y done, ~ in progress, - not yet, N not applicable
        ("Dataset (1.65 M windows, 0 leakage)", "YYNYY"),
        ("Physics operator (Gate 1, 5/5)", "YYNYY"),
        ("Physics-only baselines (B1, PSM)", "YYNY-"),
        ("Physics-fill imputation (3 seeds, test)", "YYNYY"),
        ("Null test + gradient audit", "YYNYY"),
        ("PC-STGNN model + losses", "YY~~-"),
        ("λ re-tuning S4.3ᵉ (γ = 1)", "Y~~~-"),
        ("Core ablation S4.4 → Gate 4", "Y----"),
        ("Trust gate, OOD, mismatch (F–G)", "~----"),
    ]
    sym = {"Y": ("✓", GOOD, 17), "~": ("◐", ORANGE, 17), "-": ("·", MUTED, 20), "N": ("n/a", MUTED, 10)}
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_xlim(-3.6, len(cols))
    ax.set_ylim(-1.2, len(rows) + 0.2)
    ax.axis("off")
    for j, c in enumerate(cols):
        ax.text(j + 0.5, len(rows) - 0.05, c, ha="center", va="bottom", fontsize=11, weight="bold", color=INK)
    for i, (lab, s) in enumerate(rows):
        y = len(rows) - 1 - i
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((-3.6, y - 0.45), 3.6 + len(cols), 0.9, color="#f4f4f2", zorder=0))
        ax.text(-3.5, y, lab, va="center", fontsize=11, color=INK)
        for j, ch in enumerate(s):
            t, c, fs = sym[ch]
            ax.text(j + 0.5, y, t, ha="center", va="center", fontsize=fs, color=c)
    ax.text(-3.5, -1.0, "✓ done   ◐ in progress   · not yet   n/a = not a trained component.   "
            "'Validated' = enough evidence to state as a result (test split, 3 seeds where required).",
            fontsize=9.5, color=INK2)
    ax.set_title("Phase-2 status: implemented ≠ trained ≠ validated", loc="left", fontsize=13.5,
                 weight="bold", color=INK, pad=42)
    save(fig, "p2_status.png")


if __name__ == "__main__":
    fig_collapse()
    fig_gantt()
    fig_why()
    fig_transition()
    fig_arch_simple()
    fig_status()
