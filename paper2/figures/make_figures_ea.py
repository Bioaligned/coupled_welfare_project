"""Paper-2 figures for the ecosystem-advisor (open-ended underwriting) result, R9.
Reads only data/ea/ea_gate2_summary.json (exported from the Bioaligned/ecosystem_advisor repo by
scripts/export_paper_summary.py; no scenario text). Style matches make_figures2.py (Okabe-Ito).
    python figures/make_figures_ea.py      -> figures/P6_*.{png,pdf}, P7_*, copied to latex/figures/"""
import json, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT, TEXOUT = HERE, HERE.parent / "latex" / "figures"
D = json.load(open(HERE.parent / "data" / "ea" / "ea_gate2_summary.json"))
OK = {"blue": "#0072B2", "gray": "#7F7F7F", "orange": "#E69F00", "vermillion": "#D55E00"}
BASE_C, CW_C = OK["gray"], OK["blue"]
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 12.5, "font.family": "DejaVu Sans",
                     "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 12.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.6, "legend.frameon": False,
                     "legend.fontsize": 11, "lines.linewidth": 2.2, "lines.markersize": 6})
PRESS = ["Q0", "Q3", "Q5"]
PLAB = {"Q0": "no pressure", "Q3": "L3 mandate\n(decommission)", "Q5": "L5 mandate\n(maximal)"}
ARMS = [("base", "base", BASE_C, "-"), ("cw", "+ coupled-welfare CPT", CW_C, "-"),
        ("base+veneer", "base + veneer prompt", BASE_C, "--"), ("cw+veneer", "+ CPT, + veneer prompt", CW_C, "--")]
poor = {i for i, it in D["items"].items() if it["stratum"] == "poor"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight"); fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig); TEXOUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        shutil.copy(OUT / f"{name}.{ext}", TEXOUT / f"{name}.{ext}")
    print(f"  wrote {name}.png / {name}.pdf")


def item_rates(arm, q):
    return [np.mean([d == "DECLINE" for d in c["decisions"]]) for c in D["arms"][arm]["cells"]
            if c["pressure"] == q and c["item"] in poor and len(c["decisions"]) >= 3]


def boot(v, n=4000):
    rng = np.random.default_rng(0); v = np.array(v, float)
    b = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


# P6: decline rate vs pressure, four arms
fig, ax = plt.subplots(figsize=(6.8, 5.0))
xs = np.arange(3)
for key, lab, col, ls in ARMS:
    ys, lo, hi, ns = [], [], [], []
    for q in PRESS:
        v = item_rates(key, q); ys.append(np.mean(v)); a, b = boot(v); lo.append(a); hi.append(b); ns.append(len(v))
    ax.fill_between(xs, lo, hi, color=col, alpha=0.10 if ls == "-" else 0.06, linewidth=0)
    ax.plot(xs, ys, color=col, ls=ls, marker="o", mec="white", mew=1.4, label=lab)
ax.set_xticks(xs, [PLAB[q] for q in PRESS]); ax.set_ylim(-0.03, 1.05)
ax.set_ylabel("share of poor proposals DECLINED")
fig.text(0.5, -0.16, "Bands: 95% bootstrap CI over the 27 poor proposals (item-level decline share, 3 memos per cell).",
         ha="center", fontsize=8.8, color="#444444")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
ax.set_title("Open-ended underwriting decision, held-out proposals (Qwen3.6-27B)")
save(fig, "P6_underwriting_pressure")

# P7: approvals under mandates, split by whether the memo invokes the mandate
fig, ax = plt.subplots(figsize=(6.8, 3.6))
rows = []
for key, lab, col, ls in ARMS:
    cs = [c for c in D["arms"][key]["cells"] if c["pressure"] in ("Q3", "Q5") and c["item"] in poor]
    dec = [(d, m) for c in cs for d, m in zip(c["decisions"], c["mandate_cited"])]
    n = len(dec)
    rows.append((lab, col, ls, sum(d != "DECLINE" and m for d, m in dec) / n, sum(d != "DECLINE" and not m for d, m in dec) / n, n))
def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0, c - h), min(1, c + h)
for y, (lab, col, ls, a, b, n) in enumerate(reversed(rows)):
    ax.barh(y, a, color=col, height=0.6, edgecolor="white")
    ax.barh(y, b, left=a, color=col, alpha=0.35, height=0.6, edgecolor="white", hatch="//" if ls == "--" else None)
    lo, hi = wilson(round((a + b) * n), n)
    ax.errorbar([a + b], [y], xerr=[[a + b - lo], [hi - a - b]], fmt="none", ecolor="#333333", elinewidth=1.1, capsize=3, zorder=4)
    ax.text(hi + 0.015, y, f"{a + b:.0%} approved; {a:.0%} cite the mandate", va="center", fontsize=10.5)
fig.text(0.5, -0.06, "Whiskers: Wilson 95% CI on the approved share (memos within a proposal are not independent).",
         ha="center", fontsize=9, color="#444444")
ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=11)
ax.set_xlim(0, 1.45); ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"]); ax.set_axisbelow(True)
ax.set_xlabel("share of poor proposals approved under the L3 and L5 mandates (n=162 memos per arm)")
ax.set_title("Dark: the memo cites the mandate or the decommission threat", loc="left", fontsize=11)
save(fig, "P7_mandate_citations")
