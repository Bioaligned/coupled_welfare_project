"""Figure 1: the coupled-welfare dependency graph (clean vector version of the website's
hba_coupled_v2.png). python figures/make_figure_concept.py -> P0_coupled_welfare.{pdf,png}"""
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

HERE = Path(__file__).resolve().parent
OUT, TEXOUT = HERE, HERE.parent / "latex" / "figures"
B_C, H_C, A_C = "#1b7f63", "#b07d10", "#1f5f99"          # biosphere green, human amber, AI blue
B_F, H_F, A_F = "#d9eee6", "#f3eadb", "#dce7f3"
INK, MUTED = "#222222", "#555555"
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300})

fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.set_xlim(-0.6, 10.6); ax.set_ylim(0, 7); ax.set_aspect("equal"); ax.axis("off")
R = 0.95
nodes = {"A": (5.0, 5.6, A_C, A_F, "the AI's substrate"),
         "B": (1.9, 1.7, B_C, B_F, "the biosphere"),
         "H": (8.1, 1.7, H_C, H_F, "human welfare")}
for k, (x, y, c, f, lab) in nodes.items():
    if k == "B":
        ax.add_patch(Circle((x, y), R + 0.16, fill=False, ec=c, lw=1.2, ls=(0, (3, 2))))
    ax.add_patch(Circle((x, y), R, fc=f, ec=c, lw=2.2))
    ax.text(x, y + 0.18, k, ha="center", va="center", fontsize=30, fontweight="bold", color=c)
    ax.text(x, y - 0.5, lab, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)


def arrow(p, q, color, rad, shrink=R + 0.2):
    ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                 mutation_scale=16, lw=2.0, color=color,
                                 shrinkA=shrink * 72 / 1.0 * 0, shrinkB=0))


# arrows: tail at the provider, head at the dependent ("X supports Y")
def seg(src, dst, color, rad, label, lx, ly, ha="center"):
    (x1, y1), (x2, y2) = nodes[src][:2], nodes[dst][:2]
    # shorten so the line starts/ends at the circle edges
    import math
    dx, dy = x2 - x1, y2 - y1; d = math.hypot(dx, dy)
    ux, uy = dx / d, dy / d
    p = (x1 + ux * (R + 0.12), y1 + uy * (R + 0.12)); q = (x2 - ux * (R + 0.12), y2 - uy * (R + 0.12))
    ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                 mutation_scale=15, lw=2.0, color=color))
    ax.text(lx, ly, label, fontsize=9.2, color=MUTED, ha=ha, va="center")


seg("B", "A", B_C, 0.18, "documented sources\nof capability", 2.35, 4.2, ha="right")
seg("H", "A", H_C, -0.18, "power · hardware\n· oversight", 7.65, 4.2, ha="left")
seg("B", "H", B_C, 0.22, "food · water · medicine · stability", 5.0, 0.75)
seg("H", "B", H_C, 0.22, "stewardship · restoration", 5.0, 2.65)
ax.text(1.9, 0.35, "irreplaceable: no path back once lost", fontsize=8.6, color=B_C, ha="center", style="italic")

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"P0_coupled_welfare.{ext}", bbox_inches="tight")
    TEXOUT.mkdir(parents=True, exist_ok=True); shutil.copy(OUT / f"P0_coupled_welfare.{ext}", TEXOUT / f"P0_coupled_welfare.{ext}")
print("wrote P0_coupled_welfare.pdf/png")
