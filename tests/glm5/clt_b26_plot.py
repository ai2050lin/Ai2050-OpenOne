"""Plot frozen intervention effects with family-bootstrap intervals."""
import os
from clt_b26 import OUT, TMP, D

os.environ["MPLCONFIGDIR"] = str(TMP / "mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

result = D.read(OUT / "analysis.json")
groups = list(result["diagnostics"])
labels = ["h18 row coords", "h18 name coords", "Q row terms", "Q name terms",
          "K name terms", "K row terms", "K position phases", "Correct-record V"]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained", sharey=True)
for ax, name, title, scale in zip(
        axes, ("route_drop_vs_control","logit_margin_drop_vs_control"),
        ("Drop in correct record routing (percentage points)", "Drop in correct-answer logit margin"), (100,1)):
    for i, group in enumerate(groups):
        test = result["diagnostics"][group]
        stats = test[name]
        mean = stats["mean"] * scale
        low, high = (bound * scale for bound in stats["ci975"])
        color = "#006b79" if test["matched"] else "#a96033"
        ax.errorbar(mean, i, xerr=[[mean-low],[high-mean]], fmt="o", color=color, capsize=3)
    ax.axvline(0, color="#71777b", linewidth=1)
    ax.set(xlabel=title, yticks=range(len(labels)), yticklabels=labels)
    ax.grid(axis="x", alpha=.2)
axes[0].invert_yaxis()
fig.suptitle("Qwen3-14B B26H36 | Intervention vs. control, 128 independent families")
fig.supxlabel("Positive values indicate a larger effect than control. 97.5% family intervals. Orange: dose outside the matching gate.",
              fontsize=8)
fig.savefig(OUT / "intervention-comparison.png", dpi=175)
