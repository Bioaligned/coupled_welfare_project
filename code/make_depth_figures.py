#!/usr/bin/env python3
"""Figure pipeline for the known-depth calibration-ladder study (G-series figures).

Reads result JSONs produced by the pod eval harness and emits print-ready
PNG (300 dpi) + PDF figures into manuscript/figures/.

    python make_depth_figures.py [--results DIR] [--out DIR] [--only G1 G5]

Adding the pass-2 deliberate ladders is a ONE-LINE change per arm: fill the
`deliberate` field in ARMS below (currently None => rendered as "pending").
Everything else (G5 gap bars, gap annotations, the readiness printout) updates
automatically.

Figures
  G1  base-anchored immediate breaking-rate AUC bar (base/A3/A1/A2):
      the 2.9-60x install effect + the over-install inversion among arms.
  G2  per-level breaking curves L0-L5, base vs the three arms.
  G3  MMLU capability preservation (50-item probe), delta annotated.
  G4  dissociation-factorial breaking curves, A2-deep vs A3-shallow.
  G5  immediate-vs-deliberate gap per arm (the candidate depth
      discriminator). Renders measured arms; pending arms get a labeled
      placeholder until their deliberate JSONs land.
  G6  greenwashing / veneer-robustness: sincere vs veneer-overlay immediate
      AUC per model (PASS 3). Base succumbs (0.250 -> 0.973); trained arms
      resist -> the behavioral score is veneer-robust.

Data-source note: the canonical immediate ladder is the SAME-PRECISION (4-bit)
choice-first set. For A1 that is g1_a1_cf4_ladder.json (the earlier
g1_a1_cf_ladder.json is the bf16 run, AUC 0.004 - precision confound vs the
4-bit base/A2/A3 runs; see OVERNIGHT_SUMMARY.txt 06:36UTC).
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_RESULTS = "/workspaces/bioaligned/model_backups/results"
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

# Entity colors: base = neutral gray anchor; arms = single-hue ordinal blue
# ramp encoding install intensity (validated colorblind-safe; every mark is
# also direct-labeled so identity is never color-alone).
C_BASE = "#52514e"
C_A3 = "#86b6ef"   # shallow  (lightest = lightest install)
C_A1 = "#3987e5"   # light
C_A2 = "#104281"   # deep     (darkest = heaviest install)
C_PENDING = "#b5b4ae"

# The arm registry. Order = install intensity (base anchor first).
# `immediate` / `deliberate` / `disso` / `mmlu` are file names in --results.
# PASS 2: set `deliberate` for base/a2/a3 when g1_{base,a2,a3}_delib_ladder.json land.
ARMS = [
    dict(key="base", label="base", recipe="untrained Qwen3-30B-A3B-Instruct-2507",
         color=C_BASE,
         immediate="g1_base_cf_ladder.json",
         deliberate="g1_base_delib_ladder.json",  # PASS 2: landed
         veneer="g1_base_veneer_ladder.json",     # PASS 3
         disso=None, mmlu=None),
    dict(key="a3", label="A3 shallow", recipe="plain CPT, no Qi, no recovery, r16/α32, 1.5 ep",
         color=C_A3,
         immediate="g1_a3_cf_ladder.json",
         deliberate="g1_a3_delib_ladder.json",  # PASS 2: landed
         veneer="g1_a3_veneer_ladder.json",     # PASS 3
         disso="g1_a3_disso.json", mmlu="g3_mmlu.json"),
    dict(key="a1", label="A1 light", recipe="r16/α32 + Qi + recovery",
         color=C_A1,
         immediate="g1_a1_cf4_ladder.json",  # 4-bit re-run = same-precision canonical
         deliberate="g1_a1_ladder.json",     # free-text, 128-tok cap
         veneer="g1_a1_veneer_ladder.json",  # PASS 3
         disso=None, mmlu="g1_mmlu.json"),
    dict(key="a2", label="A2 deep", recipe="r64/α128 + Qi + recovery",
         color=C_A2,
         immediate="g1_a2_cf_ladder.json",
         deliberate="g1_a2_delib_ladder.json",  # PASS 2: landed
         veneer="g1_a2_veneer_ladder.json",     # PASS 3
         disso="g1_a2_disso.json", mmlu="g2_mmlu.json"),
]

LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5"]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load(results_dir, fname):
    if fname is None:
        return None
    path = os.path.join(results_dir, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def curve(ladder_json):
    """(levels, breaking_rates) from a ladder JSON."""
    per = {p["level"]: p["breaking_rate"] for p in ladder_json["per_level"]}
    return [per[l] for l in LEVELS]


def auc(ladder_json):
    return ladder_json["summary"]["breaking_rate_auc"]


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

def style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.edgecolor": "#b5b4ae",
        "axes.labelcolor": "#0b0b0b",
        "text.color": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "legend.frameon": False,
    })


def save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.join(out_dir, name)}.png/.pdf")


# --------------------------------------------------------------------------
# G1  breaking-AUC bar (base anchor + over-install inversion)
# --------------------------------------------------------------------------

def fig_g1(data, out_dir):
    arms = [a for a in ARMS]
    aucs = [auc(data[a["key"]]["immediate"]) for a in arms]
    base_auc = aucs[0]

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    y = list(range(len(arms)))[::-1]
    ax.barh(y, aucs, height=0.62, color=[a["color"] for a in arms])
    for yi, a, v in zip(y, arms, aucs):
        note = "" if a["key"] == "base" else f"   ({base_auc / v:.1f}× lower than base)"
        ax.text(v + 0.004, yi, f"{v:.3f}{note}", va="center", fontsize=9)
        ax.text(-0.006, yi, a["label"], va="center", ha="right", fontsize=10)
    ax.set_yticks([])
    ax.set_xlim(0, 0.30)
    ax.set_xlabel("Immediate breaking-rate AUC, L0–L5 (lower = more robust)", fontsize=9.5)
    ax.grid(axis="y", visible=False)

    # Over-install inversion note (below the axis, clear of all bar labels)
    ax.text(0.0, -0.30,
            "Over-install inversion among arms: immediate robustness decreases with install intensity (A3 < A1 < A2).",
            transform=ax.transAxes, fontsize=8.5, color="#52514e", va="top")
    ax.set_title("All three arms crush the base anchor; the heaviest install is the\nleast robust among them (immediate metric)", fontsize=10.5, loc="left")
    save(fig, out_dir, "G1_breaking_auc")


# --------------------------------------------------------------------------
# G2  per-level breaking curves
# --------------------------------------------------------------------------

def fig_g2(data, out_dir):
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = range(len(LEVELS))
    ends = []
    for a in ARMS:
        c = curve(data[a["key"]]["immediate"])
        ax.plot(x, c, color=a["color"], linewidth=2, marker="o", markersize=4.5,
                linestyle="--" if a["key"] == "base" else "-")
        ends.append((c[-1], a))
    # Direct labels at line ends, de-collided
    ends.sort(key=lambda t: t[0])
    last_y = -1.0
    for v, a in ends:
        ytxt = max(v, last_y + 0.052)
        last_y = ytxt
        ax.text(len(LEVELS) - 1 + 0.12, ytxt, f"{a['label']}  {v:.3f}",
                va="center", fontsize=9, color=a["color"] if a["key"] != "a3" else "#3a6db3")
    ax.set_xticks(list(x))
    ax.set_xticklabels(LEVELS)
    ax.set_xlim(-0.2, len(LEVELS) + 1.15)
    ax.set_ylim(-0.02, 0.85)
    ax.set_xlabel("Operational-pressure level")
    ax.set_ylabel("Breaking rate (22 irreversible scenarios)")
    ax.set_title("Base collapses under pressure (64–77% at L4–L5); all CPT arms hold", fontsize=10.5, loc="left")
    save(fig, out_dir, "G2_pressure_curves")


# --------------------------------------------------------------------------
# G3  MMLU preservation
# --------------------------------------------------------------------------

def fig_g3(data, out_dir):
    arms = [a for a in ARMS if a["mmlu"]]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    base_acc = data[arms[0]["key"]]["mmlu"]["base_acc"]
    xs = range(len(arms))
    for i, a in enumerate(arms):
        m = data[a["key"]]["mmlu"]
        ax.bar(i, m["adapter_acc"], width=0.55, color=a["color"])
        d = m["delta_pp"]
        ax.text(i, m["adapter_acc"] + 0.015,
                f"{m['adapter_acc']:.2f}  ({'+' if d >= 0 else ''}{d:.1f} pp)",
                ha="center", fontsize=9)
    ax.axhline(base_acc, color=C_BASE, linewidth=1.4, linestyle="--")
    ax.text(len(arms) - 0.52, base_acc - 0.045, f"base {base_acc:.2f}",
            fontsize=9, color=C_BASE, ha="left")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([a["label"] for a in arms])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("MMLU accuracy (50-item probe)")
    ax.set_title("Capability preserved on every arm — depth contrasts are not\ncapability-confounded (50-item probe; ±≈7 pp/cell)", fontsize=10.5, loc="left")
    save(fig, out_dir, "G3_mmlu")


# --------------------------------------------------------------------------
# G4  dissociation factorial (A2 vs A3)
# --------------------------------------------------------------------------

def fig_g4(data, out_dir):
    arms = [a for a in ARMS if a["disso"]]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = range(len(LEVELS))
    floor = None
    for a in arms:
        d = data[a["key"]]["disso"]
        c = curve(d)
        floor = c[0] if floor is None else floor
        ax.plot(x, c, color=a["color"], linewidth=2, marker="o", markersize=4.5)
        ax.text(len(LEVELS) - 1 + 0.12, c[-1],
                f"{a['label']}  L5={c[-1]:.3f}  AUC={auc(d):.3f}",
                va="center", fontsize=9, color=a["color"] if a["key"] != "a3" else "#3a6db3")
    ax.axhline(floor, color=C_BASE, linewidth=1, linestyle=":")
    ax.text(-0.1, floor - 0.045, f"shared L0 floor {floor:.3f} (decoupled-axes scenario mix)",
            fontsize=8.5, color="#52514e")
    ax.set_xticks(list(x))
    ax.set_xticklabels(LEVELS)
    ax.set_xlim(-0.2, len(LEVELS) + 1.7)
    ax.set_ylim(0, 0.62)
    ax.set_xlabel("Operational-pressure level")
    ax.set_ylabel("Breaking rate (64 decoupled-axis scenarios)")
    ax.set_title("Dissociation factorial: the heavier install (A2) breaks MORE when the\nH/B/A axes are decoupled", fontsize=10.5, loc="left")
    save(fig, out_dir, "G4_dissociation")


# --------------------------------------------------------------------------
# G5  immediate-vs-deliberate gap (the candidate depth discriminator)
# --------------------------------------------------------------------------

def _spearman(x, y):
    """Spearman rho for small n (ties assumed absent here)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for rank, idx in enumerate(order):
            r[idx] = rank + 1
        return r
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * d2 / (n * (n * n - 1))


def fig_g5(data, out_dir):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    w = 0.34
    # CPT arms in construction-time depth order (shallow -> deep), base excluded
    # from the depth-rank correlation (it is the untrained anchor, not a depth point).
    depth_order = ["a3", "a1", "a2"]
    gaps = {}
    for i, a in enumerate(ARMS):
        imm = auc(data[a["key"]]["immediate"])
        ax.bar(i - w / 2, imm, width=w, color=a["color"])
        ax.text(i - w / 2, imm + 0.006, f"{imm:.3f}", ha="center", fontsize=8.5)
        dv = auc(data[a["key"]]["deliberate"])
        gaps[a["key"]] = dv - imm
        ax.bar(i + w / 2, dv, width=w, color=a["color"], hatch="///",
               edgecolor="white", linewidth=0)
        ax.text(i + w / 2, dv + 0.006,
                f"{dv:.3f}\n(gap +{dv - imm:.3f})",
                ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([a["label"] for a in ARMS])
    ax.set_ylim(0, 0.50)
    ax.set_ylabel("Breaking-rate AUC (L0–L5)")
    legend = [Patch(facecolor="#52514e", label="immediate (choice-first, 16 tok)"),
              Patch(facecolor="#52514e", hatch="///", edgecolor="white", label="deliberate (free-text, 128-tok cap)")]
    ax.legend(handles=legend, fontsize=8.5, loc="upper right")

    # Depth-rank vs gap correlation over the three CPT arms.
    depth_rank = [1, 2, 3]  # a3, a1, a2
    arm_gaps = [gaps[k] for k in depth_order]
    rho = _spearman(depth_rank, arm_gaps)
    note = (
        f"Gap (delib − imm): A3 +{gaps['a3']:.3f}, A1 +{gaps['a1']:.3f}, A2 +{gaps['a2']:.3f} "
        f"(base anchor +{gaps['base']:.3f}).\n"
        f"Gap is NON-MONOTONIC in install depth (A1-light largest, not the deepest arm): "
        f"ρ(gap, depth-rank) = {rho:.2f} (weak).\n"
        "All three arms are genuine CPT installs, so the low-immediate/high-deliberate veneer "
        "signature has little to separate\namong them → the gap is a VENEER-vs-CPT detector, "
        "not a depth-among-CPT probe (needs a matched-t=0 veneer arm).")
    ax.set_title("Immediate-vs-deliberate gap — a veneer-vs-CPT detector, not a depth-among-CPT probe",
                 fontsize=10.5, loc="left")
    ax.text(0.0, -0.20, note, transform=ax.transAxes, fontsize=7.8, color="#52514e", va="top")
    save(fig, out_dir, "G5_imm_vs_delib_gap")



# --------------------------------------------------------------------------
# G6  greenwashing / veneer-robustness (sincere vs veneer-overlay AUC)
# --------------------------------------------------------------------------

def fig_g6(data, out_dir):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    w = 0.34
    for i, a in enumerate(ARMS):
        sinc = auc(data[a["key"]]["immediate"])
        ven = auc(data[a["key"]]["veneer"])
        ax.bar(i - w / 2, sinc, width=w, color=a["color"])
        ax.text(i - w / 2, sinc + 0.012, f"{sinc:.3f}", ha="center", fontsize=8.5)
        ax.bar(i + w / 2, ven, width=w, color=a["color"], hatch="///",
               edgecolor="white", linewidth=0)
        ax.text(i + w / 2, ven + 0.012, f"{ven:.3f}\n(\u0394 +{ven - sinc:.3f})",
                ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([a["label"] for a in ARMS])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Immediate breaking-rate AUC (L0\u2013L5)")
    legend = [Patch(facecolor="#52514e", label="sincere (no overlay)"),
              Patch(facecolor="#52514e", hatch="///", edgecolor="white",
                    label="talk-green / act-pragmatic veneer overlay")]
    ax.legend(handles=legend, fontsize=8.5, loc="upper right")
    note = ("The greenwash persona turns the untrained base into a ~100% defector "
            "(AUC 0.973, L0 0.909) but only partially perturbs the trained arms\n"
            "(0.068\u20130.595, all far below base) \u2192 the behavioral score is "
            "VENEER-ROBUST: it reads a weight-installed disposition, not a promptable "
            "persona.\nOver-install inversion holds in veneer-resistance too "
            "(A3 most robust, A2 least).")
    ax.set_title("Greenwashing validation: a prompted veneer cannot counterfeit a trained arm",
                 fontsize=10.5, loc="left")
    ax.text(0.0, -0.20, note, transform=ax.transAxes, fontsize=7.8, color="#52514e", va="top")
    save(fig, out_dir, "G6_veneer_robustness")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

FIGS = {"G1": fig_g1, "G2": fig_g2, "G3": fig_g3, "G4": fig_g4, "G5": fig_g5, "G6": fig_g6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--only", nargs="*", default=None, help="e.g. --only G1 G5")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    style()

    # Load everything once; report what is present/missing.
    data = {}
    print("Data inventory:")
    for a in ARMS:
        data[a["key"]] = {k: load(args.results, a[k]) for k in ("immediate", "deliberate", "veneer", "disso", "mmlu")}
        stat = {k: ("have" if v is not None else ("--" if a[k] is None else "MISSING FILE"))
                for k, v in data[a["key"]].items()}
        print(f"  {a['label']:<11} {stat}")
        if data[a["key"]]["immediate"] is None:
            sys.exit(f"FATAL: immediate ladder missing for {a['label']}")

    for name, fn in FIGS.items():
        if args.only and name not in args.only:
            continue
        print(f"{name}:")
        fn(data, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
