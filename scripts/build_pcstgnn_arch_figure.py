"""PC-STGNN architecture figure (paper/slide quality) -> paper/figures/pcstgnn_architecture.{svg,png}.

Hand-laid SVG so every tensor shape, constant and gradient path is exact. Numbers are taken from
the code and closed gates, not invented: operator grid from operator/grid.json, measured-only graph
from dataset_fullscale_50/processed/edge_index.npy, sigma_eps from 03_STAGE_B Gate 1, residual
timing from results/pcstgnn/audit/s3_4_residual_speed/speed_benchmark.json.

Usage:
    uv run python scripts/build_pcstgnn_arch_figure.py
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_SVG = REPO / "paper" / "figures" / "pcstgnn_architecture.svg"
OUT_PNG = REPO / "paper" / "figures" / "pcstgnn_architecture.png"

W, H = 1920, 1172
DY = 22  # vertical gap between the offline row and the rest, for arrow routing
FONT = "DejaVu Sans, Arial, Helvetica, sans-serif"

C = {
    "phys_fill": "#EAF1FB", "phys_stroke": "#2F5D8C",
    "learn_fill": "#FDEFE3", "learn_stroke": "#B5541C",
    "in_fill": "#F4F4F4", "in_stroke": "#555555",
    "fus_fill": "#F1ECF8", "fus_stroke": "#6A4C93",
    "loss_fill": "#EDF6EE", "loss_stroke": "#2E7D32",
    "panel_fill": "#FBFBFB", "panel_stroke": "#BBBBBB",
    "text": "#1A1A1A", "muted": "#555555", "grad": "#222222", "const": "#2F5D8C", "train": "#888888",
}

out: list[str] = []


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


TOKEN = re.compile(r"(_\{[^}]*\}|\^\{[^}]*\})")


def rich(s: str, size: float) -> str:
    """Tiny markup: _{sub} and ^{sup} -> tspans with explicit px baseline shifts."""
    parts, cur = [], 0.0
    for tok in TOKEN.split(s):
        if not tok:
            continue
        if tok.startswith("_{"):
            target, body, fs = 0.32 * size, tok[2:-1], 0.72 * size
        elif tok.startswith("^{"):
            target, body, fs = -0.42 * size, tok[2:-1], 0.72 * size
        else:
            target, body, fs = 0.0, tok, None
        dy = target - cur
        cur = target
        attrs = f' dy="{dy:.2f}"' if abs(dy) > 1e-9 else ""
        if fs:
            attrs += f' font-size="{fs:.1f}"'
        parts.append(f"<tspan{attrs}>{esc(body)}</tspan>")
    return "".join(parts)


def text(x, y, s, size=13, weight="normal", anchor="start", color=None, style="normal"):
    color = color or C["text"]
    out.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
               f'font-weight="{weight}" font-style="{style}" text-anchor="{anchor}" '
               f'fill="{color}">{rich(s, size)}</text>')


def lines(x, y, rows, size=13, gap=None, **kw):
    gap = gap or size * 1.42
    for i, r in enumerate(rows):
        indent, r = r if isinstance(r, tuple) else (0, r)
        if r:
            text(x + indent, y + i * gap, r, size=size, **kw)


def rect(x, y, w, h, fill, stroke, rx=8, dash=False, sw=1.6, opacity=1.0):
    d = ' stroke-dasharray="7 4"' if dash else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
               f'fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def box(x, y, w, h, kind, title, rows=(), size=12.5, dash=False, title_size=15):
    fill, stroke = C[f"{kind}_fill"], C[f"{kind}_stroke"]
    rect(x, y, w, h, fill, stroke, dash=dash)
    text(x + w / 2, y + 22, title, size=title_size, weight="bold", anchor="middle", color=stroke)
    lines(x + 12, y + 44, rows, size=size)


def panel(x, y, w, h, title, color):
    rect(x, y, w, h, C["panel_fill"], C["panel_stroke"], rx=12, sw=1.2)
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="30" rx="12" fill="{color}" opacity="0.12"/>')
    text(x + 14, y + 21, title, size=16, weight="bold", color=color)


def arrow(points, kind="grad", label=None, label_at=None, label_anchor="start", sw=2.0):
    color = {"grad": C["grad"], "const": C["const"], "train": C["train"]}[kind]
    dash = {"grad": "", "const": ' stroke-dasharray="8 5"', "train": ' stroke-dasharray="4 4"'}[kind]
    d = "M " + " L ".join(f"{px},{py}" for px, py in points)
    out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{dash} '
               f'marker-end="url(#arr_{kind})"/>')
    if label:
        lx, ly = label_at or points[len(points) // 2]
        text(lx, ly, label, size=12, anchor=label_anchor, color=color, style="italic")


def defs():
    out.append("<defs>")
    for k, col in (("grad", C["grad"]), ("const", C["const"]), ("train", C["train"])):
        out.append(f'<marker id="arr_{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
                   f'markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
                   f'fill="{col}"/></marker>')
    out.append("</defs>")


# --------------------------------------------------------------------------- drawings

def tensor_cube(x, y, w, h, d, label_front, fill="#DDE6F0", stroke="#455A73"):
    """Isometric block: front face at (x,y,w,h), depth offset d."""
    top = f"{x},{y} {x + d},{y - d} {x + w + d},{y - d} {x + w},{y}"
    side = f"{x + w},{y} {x + w + d},{y - d} {x + w + d},{y + h - d} {x + w},{y + h}"
    out.append(f'<polygon points="{top}" fill="#EEF3F8" stroke="{stroke}" stroke-width="1.3"/>')
    out.append(f'<polygon points="{side}" fill="#C9D6E5" stroke="{stroke}" stroke-width="1.3"/>')
    rect(x, y, w, h, fill, stroke, rx=0, sw=1.3)
    # channel slabs on the front face
    for i in range(1, 6):
        yy = y + h * i / 6
        out.append(f'<line x1="{x}" y1="{yy}" x2="{x + w}" y2="{yy}" stroke="{stroke}" '
                   f'stroke-width="0.6" opacity="0.6"/>')
    text(x + w / 2, y + h / 2 + 5, label_front, size=13, weight="bold", anchor="middle")


def measured_graph(x0, y0, w, h, dropped=(), title=None):
    ei = np.load(REPO / "dataset_fullscale_50/processed/edge_index.npy")
    mapping = json.loads((REPO / "dataset_fullscale_50/processed/node_mapping.json").read_text())
    bus = {v: k for k, v in mapping.items()}
    adj = {i: set() for i in range(25)}
    for a, b in ei.T.tolist():
        adj[a].add(b)
        adj[b].add(a)
    # BFS tree from node 0 (bus 1, nearest the substation)
    parent, order, depth = {0: None}, [], {0: 0}
    q = deque([0])
    while q:
        u = q.popleft()
        order.append(u)
        for v in sorted(adj[u]):
            if v not in parent:
                parent[v], depth[v] = u, depth[u] + 1
                q.append(v)
    children = {u: [v for v in order if parent.get(v) == u] for u in order}
    # tidy layout: leaves get consecutive x slots, parents centred over children
    xs, slot = {}, [0]

    def place(u):
        if not children[u]:
            xs[u] = slot[0]
            slot[0] += 1
        else:
            for c in children[u]:
                place(c)
            xs[u] = (xs[children[u][0]] + xs[children[u][-1]]) / 2

    place(0)
    max_d = max(depth.values())
    n_slots = slot[0] - 1
    px = {u: x0 + 12 + (w - 24) * xs[u] / max(n_slots, 1) for u in xs}
    py = {u: y0 + 14 + (h - 28) * depth[u] / max_d for u in xs}
    for v, u in parent.items():
        if u is not None:
            out.append(f'<line x1="{px[u]:.1f}" y1="{py[u]:.1f}" x2="{px[v]:.1f}" y2="{py[v]:.1f}" '
                       f'stroke="#7A8794" stroke-width="1.2"/>')
    for u in xs:
        dead = u in dropped
        fill = "#FFFFFF" if dead else "#4F7CAC"
        out.append(f'<circle cx="{px[u]:.1f}" cy="{py[u]:.1f}" r="5.2" fill="{fill}" '
                   f'stroke="{"#B00020" if dead else "#2F4F73"}" stroke-width="1.4"/>')
        if dead:
            for s in (1, -1):
                out.append(f'<line x1="{px[u] - 4:.1f}" y1="{py[u] - 4 * s:.1f}" x2="{px[u] + 4:.1f}" '
                           f'y2="{py[u] + 4 * s:.1f}" stroke="#B00020" stroke-width="1.6"/>')
    root = 0
    text(px[root] + 9, py[root] + 4, f"bus {bus[root]}", size=10, color=C["muted"])


def bars(x, y, w, h, values, highlight=None, color="#4F7CAC", hi_color="#B5541C", axis=True):
    n = len(values)
    bw = w / n
    vmax = max(values) or 1
    for i, v in enumerate(values):
        bh = h * v / vmax
        col = hi_color if i == highlight else color
        out.append(f'<rect x="{x + i * bw + 0.6:.1f}" y="{y + h - bh:.1f}" width="{bw - 1.2:.1f}" '
                   f'height="{bh:.1f}" fill="{col}"/>')
    if axis:
        out.append(f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" stroke="#333" stroke-width="1"/>')


def gru_unrolled(x, y):
    cw, ch, gap = 42, 30, 20
    xs = [x, x + cw + gap, x + 2 * (cw + gap) + 26]
    labels = ["t=1", "t=2", "t=20"]
    for i, (cx, lab) in enumerate(zip(xs, labels)):
        rect(cx, y, cw, ch, "#FFFFFF", C["learn_stroke"], rx=5, sw=1.3)
        text(cx + cw / 2, y + 19, "GRU", size=11, weight="bold", anchor="middle", color=C["learn_stroke"])
        text(cx + cw / 2, y + ch + 30, lab, size=10, anchor="middle", color=C["muted"])
        out.append(f'<line x1="{cx + cw / 2}" y1="{y + ch + 18}" x2="{cx + cw / 2}" y2="{y + ch + 3}" '
                   f'stroke="#555" stroke-width="1.2" marker-end="url(#arr_grad)"/>')
    for a, b in ((0, 1),):
        out.append(f'<line x1="{xs[a] + cw}" y1="{y + ch / 2}" x2="{xs[b] - 2}" y2="{y + ch / 2}" '
                   f'stroke="#555" stroke-width="1.2" marker-end="url(#arr_grad)"/>')
    text(xs[1] + cw + 13, y + ch / 2 + 4, "…", size=14, anchor="middle")
    out.append(f'<line x1="{xs[2] + cw}" y1="{y + ch / 2}" x2="{xs[2] + cw + 22}" y2="{y + ch / 2}" '
               f'stroke="#555" stroke-width="1.2" marker-end="url(#arr_grad)"/>')
    text(xs[2] + cw + 26, y + ch / 2 + 4, "h_{20}", size=12)


def sage_inset(x, y):
    """One RGSAGE-Max aggregation: neighbours -> max -> concat with self -> W."""
    cx, cy = x + 30, y + 42
    nb = [(x + 4, y + 8), (x + 4, y + 76), (x + 60, y + 6)]
    for nx_, ny_ in nb:
        out.append(f'<line x1="{nx_}" y1="{ny_}" x2="{cx}" y2="{cy}" stroke="#7A8794" stroke-width="1.1"/>')
        out.append(f'<circle cx="{nx_}" cy="{ny_}" r="6" fill="#F6C9A8" stroke="{C["learn_stroke"]}"/>')
    out.append(f'<circle cx="{cx}" cy="{cy}" r="8" fill="{C["learn_stroke"]}"/>')
    text(cx, cy + 4, "i", size=10, weight="bold", anchor="middle", color="#FFFFFF")
    text(x + 60, y + 30, "j ∈ N(i)", size=10, color=C["muted"])
    rect(x + 82, y + 30, 48, 24, "#FFFFFF", C["learn_stroke"], rx=4, sw=1.2)
    text(x + 106, y + 47, "max", size=11, weight="bold", anchor="middle", color=C["learn_stroke"])
    out.append(f'<line x1="{cx + 9}" y1="{cy}" x2="{x + 80}" y2="{y + 42}" stroke="#555" '
               f'stroke-width="1.1" marker-end="url(#arr_grad)"/>')
    rect(x + 146, y + 30, 26, 24, "#FFFFFF", C["learn_stroke"], rx=4, sw=1.2)
    text(x + 159, y + 47, "‖", size=13, weight="bold", anchor="middle", color=C["learn_stroke"])
    out.append(f'<line x1="{x + 130}" y1="{y + 42}" x2="{x + 144}" y2="{y + 42}" stroke="#555" '
               f'stroke-width="1.1" marker-end="url(#arr_grad)"/>')
    out.append(f'<path d="M {cx} {cy + 9} L {cx} {y + 74} L {x + 159} {y + 74} L {x + 159} {y + 56}" '
               f'fill="none" stroke="#555" stroke-width="1.1" marker-end="url(#arr_grad)"/>')
    text(x + 92, y + 88, "self h_{i}", size=10, color=C["muted"])
    rect(x + 188, y + 30, 40, 24, "#FFFFFF", C["learn_stroke"], rx=4, sw=1.2)
    text(x + 208, y + 47, "W^{(ℓ)}", size=12, weight="bold", anchor="middle", color=C["learn_stroke"])
    out.append(f'<line x1="{x + 172}" y1="{y + 42}" x2="{x + 186}" y2="{y + 42}" stroke="#555" '
               f'stroke-width="1.1" marker-end="url(#arr_grad)"/>')
    text(x + 238, y + 47, "→ h_{i}^{(ℓ)}", size=12)


def stacked_plates(x, y, w, h, n=4):
    for i in range(n - 1, -1, -1):
        rect(x + i * 7, y - i * 7, w, h, "#FFFFFF" if i else "#FFF7F0", C["learn_stroke"], rx=4, sw=1.1)
    text(x + w / 2, y + h / 2 + 4, "RGSAGE-Max  ×4", size=12, weight="bold", anchor="middle",
         color=C["learn_stroke"])


def head_stack(x, y):
    items = ["BatchNorm", "ReLU", "Dropout(0.35)", "Linear 64→26"]
    for i, it in enumerate(items):
        yy = y + i * 34
        rect(x, yy, 150, 24, "#FFFFFF", C["learn_stroke"], rx=4, sw=1.2)
        text(x + 75, yy + 16, it, size=11.5, anchor="middle", color=C["learn_stroke"], weight="bold")
        if i < len(items) - 1:
            out.append(f'<line x1="{x + 75}" y1="{yy + 24}" x2="{x + 75}" y2="{yy + 33}" stroke="#555" '
                       f'stroke-width="1.1" marker-end="url(#arr_grad)"/>')


# --------------------------------------------------------------------------- layout

def build() -> str:
    grid = json.loads((REPO / "operator/grid.json").read_text())
    n_r, n_l = len(grid["r_grid"]), len(grid["l_grid"])
    n_types = len(grid["fault_types"])
    n_hyp = n_types * n_r * n_l
    n_solves = grid["n_solves"]
    n_nonconv = grid["n_non_converged"]
    speed = json.loads((REPO / "results/pcstgnn/audit/s3_4_residual_speed/speed_benchmark.json").read_text())
    rho_ms = speed["rho_A"]["mean_ms"]

    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
               f'viewBox="0 0 {W} {H}">')
    defs()
    out.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')
    text(W / 2, 40, "PC-STGNN: Physics-Consistent Spatio-Temporal Graph Neural Network for "
                    "Sensor-Robust Fault Localization", size=22, weight="bold", anchor="middle")

    # ① OFFLINE ---------------------------------------------------------------
    panel(20, 62, 1880, 178, "① Offline physics operator: fixed, built once, never trained",
          C["phys_stroke"])
    bw, by, bh = 330, 102, 124
    bx = [40, 412, 784, 1156, 1528]
    box(bx[0], by, bw, bh, "phys", "IEEE 123-bus feeder", [
        "OpenDSS model (same as the corpus)",
        "25 candidate fault buses",
        "25 µPMU nodes × 3 phases = 75 channels",
        "6 of 25 fault buses instrumented"])
    box(bx[1], by, bw, bh, "phys", "Hypothesis grid  h = (τ, R, L)", [
        f"τ: {n_types} fault types (AG … ABC, ABCG)",
        f"R: {n_r} resistances, 0.05–30 Ω (log + 0.1,1,10)",
        f"L: {n_l} load levels, 0.5–1.3, step {grid['l_step']}",
        f"{n_types}×{n_r}×{n_l} = {n_hyp:,} hypotheses / location"])
    box(bx[2], by, bw, bh, "phys", "Snapshot power-flow solves", [
        f"{n_solves:,} solves · {n_nonconv} non-converged · 0 NaN",
        "load frozen at L × 0.99953 (pre-fault)",
        "settled post-fault |V| per channel",
        "recompile per solve (no standing circuit)"])
    box(bx[3], by, bw, bh, "phys", "Signature table μ (buffer)", [
        "μ[f, τ, R, L, i, p] p.u.,  μ_{0}[L, i, p] no-fault",
        f"flattened: 26 classes × {n_hyp:,} × 75",
        "σ_{ε} = 7.2555×10^{−4} p.u. (calibrated, train)",
        "requires_grad = False; not a parameter"])
    box(bx[4], by, bw, bh, "phys", "ρ cache (offline, Gram form)", [
        "ρ for every window × k ∈ {0,1,2,5,10} × 3 seeds",
        "train 1,155,000 · val 248,000 · test 247,000",
        "k = 0 training lookup: rel. err 1.6×10^{−5}",
        "keyed by window_id (order-independent)"])
    for i in range(4):
        arrow([(bx[i] + bw, by + bh / 2), (bx[i + 1] - 2, by + bh / 2)], "const")

    out.append(f'<g transform="translate(0,{DY})">')
    # ② INPUT -----------------------------------------------------------------
    panel(20, 262, 330, 650, "② Input", C["in_stroke"])
    tensor_cube(52, 330, 200, 108, 34, "X")
    text(170, 458, "nodes × samples (20 × 1 ms) × 6 channels", size=10.5, anchor="middle", color=C["muted"])
    lines(36, 484, [
        "X ∈ ℝ^{B×25×20×6}",
        "ch 0–2: |V|_{a,b,c} (z-scored on train)",
        "ch 3–5: availability mask m ∈ {0,1}",
        "(static phase instrumentation:",
        "  14 of 25 nodes are 3-phase)",
    ], size=12)
    rect(32, 600, 306, 300, C["in_fill"], C["train"], dash=True)
    text(185, 621, "Dropout augmentation (training only)", size=13.5, weight="bold",
         anchor="middle", color=C["muted"])
    lines(44, 642, [
        "k ~ U{0, 1, 2, 5} per batch;  zero all 6",
        "channels of k random nodes → X̃, mask (0,0,0)",
        "keep clean X and drop mask D ∈ {0,1}^{B×25×3}",
    ], size=11.5, gap=16)
    measured_graph(44, 700, 284, 168, dropped=(7, 15))
    text(185, 892, "measured-only graph: 25 nodes, 24 edges (example k = 2)", size=10.5,
         anchor="middle", color=C["muted"])

    # ④ PHYSICS ---------------------------------------------------------------
    panel(370, 262, 900, 360, "④ Physics branch (input-only, no gradient)",
          C["phys_stroke"])
    box(390, 304, 245, 196, "phys", "Observation", [
        "last timestep, de-normalised:",
        "s_{ip} = x_{i,20,p}·std_{p} + mean_{p}",
        "s ∈ ℝ^{B×25×3} (p.u.)",
        "A = {(i,p) : m_{ip} = 1}",
        "dropped node ⇒ not in A",
        "quasi-static fault period ⇒",
        "  last sample = settled level",
    ], size=12)
    box(655, 304, 370, 196, "phys", "Profiled residual, per class f = 0…25", [
        "r_{f}(A) = min_{(τ,R,L)} Σ_{(i,p)∈A} (s_{ip} − μ_{f,τ,R,L,i,p})^{2}",
        (120, "/ (σ_{ε}^{2} + σ_{0}^{2})"),
        "f = 0 uses μ_{0}[L] (no-fault hypothesis)",
        "hyp*_{f} = argmin hypothesis for class f",
        f"{n_hyp:,} hyp × 26 classes × 75 ch per window",
        f"live: {rho_ms:.1f} ms / batch-32 (T1000, compute-bound)",
        "k = 0 batches: verified cache lookup",
    ], size=12)
    box(1045, 304, 205, 196, "phys", "Physics evidence ρ", [
        "ρ_{f}(A) = ½ (r_{f} − min_{f′} r_{f′})",
        "ρ ∈ ℝ^{B×26}, min = 0",
    ], size=12)
    bars(1062, 384, 170, 82, [6, 9, 4, 7, 0.15, 8, 5, 9, 6, 7, 3, 8, 6, 9, 5, 7, 8, 4, 6, 9, 7, 5, 8, 6,
                              7, 9], highlight=4, color="#8FAFD3", hi_color="#B5541C")
    text(1147, 486, "(illustrative; 0 at best class)", size=9.5, anchor="middle", color=C["muted"])

    rect(655, 515, 595, 92, C["phys_fill"], C["train"], dash=True)
    text(952, 536, "Held-out residual ρ(D): training only", size=14, weight="bold",
         anchor="middle", color=C["phys_stroke"])
    lines(668, 557, [
        "ρ_{f}(D): score hyp*_{f}, fitted on A only, against the clean values on dropped channels D",
        "⇒ leave-k-out physics consistency. Exactly 0 when |D| = 0 (all k = 0 batches).",
        "Not a reconstruction loss: ‖μ[f̂] − s‖^{2} has zero gradient (f̂ is an arg-min index).",
    ], size=11.5, gap=16)

    # ③ BACKBONE --------------------------------------------------------------
    panel(370, 642, 900, 270, "③ Learned backbone: identical to the Phase-1 dropout-aware RGSAGE-Max",
          C["learn_stroke"])
    box(390, 684, 245, 214, "learn", "Temporal encoder", [], size=12)
    gru_unrolled(404, 714)
    lines(402, 812, [
        "1-layer GRU, hidden 128",
        "shared weights across 25 nodes",
        "H^{0} ∈ ℝ^{B×25×128}",
    ], size=12)
    box(655, 684, 370, 214, "learn", "Spatial message passing", [], size=12)
    sage_inset(668, 708)
    stacked_plates(690, 808, 160, 30)
    lines(880, 818, ["hidden 64", "ReLU between", "4 hops"], size=11.5, gap=15)
    text(668, 878, "h_{i}^{(ℓ)} = W^{(ℓ)} [ h_{i}^{(ℓ−1)} ‖ max_{j∈N(i)} h_{j}^{(ℓ−1)} ],  "
                   "H^{4} ∈ ℝ^{B×25×64}", size=11.5)
    box(1045, 684, 205, 214, "learn", "Node classifier", [], size=12)
    head_stack(1072, 712)
    text(1147, 868, "z ∈ ℝ^{B×25×26}", size=12.5, anchor="middle", weight="bold")
    text(1147, 886, "per-node logits", size=11, anchor="middle", color=C["muted"])

    # ⑤ FUSION ----------------------------------------------------------------
    panel(1290, 262, 610, 650, "⑤ Fusion and decision", C["fus_stroke"])
    box(1310, 304, 570, 196, "fus", "Per-node product of experts", [
        "z′_{n,f} = z_{n,f} − β · ρ_{f}(A)",
        "same ρ broadcast to every node n = 1…25",
        "⇔  softmax(z′_{n})_{f} ∝ exp(z_{n,f}) · exp(−β ρ_{f})",
        "β: the only new parameter (scalar, init 0)",
        "β = 0 at init ⇒ model starts exactly at the Phase-1 baseline",
        "params: 95,258 (backbone) + 1 (β) = 95,259",
    ], size=12.5)
    box(1310, 524, 330, 374, "fus", "Soft vote over nodes", [
        "P(f | X) = Σ_{n} softmax(z′_{n})_{f}",
        (52, "/ Σ_{f′} Σ_{n} softmax(z′_{n})_{f′}"),
        "unchanged from Phase-1 voting",
    ], size=12.5)
    bars(1340, 640, 270, 120, [1, 1.2, 0.8, 1, 9, 1.4, 1, 0.9, 1.1, 1, 0.7, 1.2, 1, 0.9, 1.3, 1, 1.1,
                               0.8, 1, 1.2, 0.9, 1, 1.1, 0.8, 1, 1.3],
         highlight=4, color="#B9A6D6", hi_color=C["fus_stroke"])
    text(1475, 782, "class f = 0 … 25 (illustrative)", size=10.5, anchor="middle", color=C["muted"])
    lines(1324, 812, [
        "per window; detection and localization",
        "share one 26-way decision",
    ], size=11.5, gap=16)
    box(1660, 524, 220, 374, "fus", "Output", [
        "ŷ = argmax_{f} P(f|X)",
        "",
        "0  = no fault",
        "1…25 = faulted bus",
        "",
        "inference adds only the",
        "ρ(A) pass. No ρ(D),",
        "no augmentation",
    ], size=12.5)

    # ⑥ LOSS ------------------------------------------------------------------
    panel(370, 932, 1530, 206, "⑥ Training objective: gradient reaches the network only through z and P",
          C["loss_stroke"])
    rect(390, 972, 1490, 150, C["loss_fill"], C["loss_stroke"])
    text(1135, 1010, "ℒ  =  CE(z′, y)   +   λ_{1} · Σ_{f} P(f|X) · ρ_{f}(A)   +   λ_{2} · "
                     "Σ_{f} P(f|X) · ρ_{f}(D)", size=21, anchor="middle", weight="bold")
    lines(412, 1046, [
        "CE: per-node cross-entropy, event label shared by all 25 nodes",
        "λ_{1}: expected physics residual on the available sensors A",
        "λ_{2}: expected residual on held-out sensors D (leave-k-out)",
    ], size=12.5, gap=19)
    lines(1060, 1046, [
        "λ terms masked on w = 20 windows (0 post-fault samples ⇒ physics must prefer f = 0)",
        "λ_{1} = λ_{2} = β = 0  ⇒  byte-identical to Phase-1 (S4.1: max |Δ| = 0.0, 30 epochs)",
        "Gradient audit S4.2 (finite difference): ∂L_{1}/∂w ≠ 0, ∂L_{2}/∂w ≠ 0; μ, σ_{ε} get none",
    ], size=12.5, gap=19)

    # legend ---------------------------------------------------------------------
    rect(20, 932, 330, 206, C["panel_fill"], C["panel_stroke"], rx=12, sw=1.2)
    text(36, 956, "Legend", size=15, weight="bold")
    arrow([(36, 980), (96, 980)], "grad")
    text(106, 984, "differentiable path (gradient flows)", size=11.5)
    arrow([(36, 1004), (96, 1004)], "const")
    text(106, 1008, "constant: input-only physics, no gradient", size=11.5)
    arrow([(36, 1028), (96, 1028)], "train")
    text(106, 1032, "training-only data", size=11.5)
    for i, (fill, stroke, lab) in enumerate((
            (C["phys_fill"], C["phys_stroke"], "fixed physics (buffers)"),
            (C["learn_fill"], C["learn_stroke"], "learned (Phase-1 backbone)"),
            (C["fus_fill"], C["fus_stroke"], "fusion (+1 parameter β)"),
            (C["loss_fill"], C["loss_stroke"], "objective"))):
        yy = 1046 + i * 21
        rect(38, yy, 26, 15, fill, stroke, rx=3, sw=1.3)
        text(76, yy + 12, lab, size=11.5)

    # arrows ----------------------------------------------------------------------
    # table -> residual, cache -> residual (k = 0)
    # input -> physics, input -> backbone, clean X/D -> rho(D)
    arrow([(338, 760), (362, 760), (362, 400), (388, 400)], "const", label="X̃", label_at=(342, 392))
    arrow([(338, 790), (388, 790)], "grad", label="X̃", label_at=(352, 784))
    arrow([(338, 640), (352, 640), (352, 560), (653, 560)], "train", label="clean X, D",
          label_at=(470, 552))
    # physics chain
    arrow([(635, 402), (653, 402)], "const")
    arrow([(1025, 402), (1043, 402)], "const")
    arrow([(1147, 500), (1147, 513)], "const")
    text(1156, 511, "hyp*", size=11, color=C["const"], style="italic")
    # backbone chain
    arrow([(635, 790), (653, 790)], "grad")
    arrow([(1025, 790), (1043, 790)], "grad")
    # fusion inputs
    arrow([(1250, 400), (1308, 400)], "const", label="ρ(A)", label_at=(1256, 392))
    arrow([(1250, 790), (1284, 790), (1284, 470), (1308, 470)], "grad", label="z", label_at=(1290, 640))
    arrow([(1595, 500), (1595, 522)], "grad", label="z′", label_at=(1602, 516))
    arrow([(1640, 711), (1658, 711)], "grad")
    # to the loss
    arrow([(1475, 898), (1475, 970)], "grad", label="P(f|X), z′", label_at=(1484, 950))
    arrow([(645, 607), (645, 970)], "const", label="ρ(A), ρ(D)", label_at=(560, 930))

    out.append("</g>")
    # offline -> physics (absolute coords; targets shifted by DY)
    arrow([(bx[3] + 110, by + bh), (bx[3] + 110, 246), (800, 246), (800, 302 + DY)], "const")
    text(812, 286 + DY, "μ, μ_{0}", size=12, color=C["const"], style="italic")
    arrow([(bx[4] + bw / 2, by + bh), (bx[4] + bw / 2, 256), (905, 256), (905, 302 + DY)], "const")
    text(917, 286 + DY, "cached ρ, k = 0", size=12, color=C["const"], style="italic")
    arrow([(bx[3] + 220, by + bh), (bx[3] + 220, 240), (1276, 240), (1276, 561 + DY),
           (1252, 561 + DY)], "const")
    text(W / 2, H - 8, "B = batch size. Shapes follow fault_localization/models/pc_stgnn.py; "
                       "constants from operator/grid.json and the closed Gate-1/Gate-3 records.",
         size=11, anchor="middle", color=C["muted"], style="italic")
    out.append("</svg>")
    return "\n".join(out)


def main() -> int:
    svg = build()
    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    OUT_SVG.write_text(svg)
    subprocess.run(["rsvg-convert", "-z", "2", "-b", "white", "-o", str(OUT_PNG), str(OUT_SVG)],
                   check=True)
    print(f"wrote {OUT_SVG}\nwrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
