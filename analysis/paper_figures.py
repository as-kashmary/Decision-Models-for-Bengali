"""Paper figures for an IEEE two-column layout (single column 3.5 in, full width 7.16 in).

A2  fig_transfer_heatmaps   train form x test form accuracy, decision model and TF-IDF
A3  fig_risk_coverage       joint decision model: accuracy vs coverage, one line per form
A5  fig_reliability         joint decision model vs joint TF-IDF reliability diagrams

Reads saved outputs only (no GPU). Run after tfidf_baseline.py:
    python paper_figures.py <project folder>
Needs: pip install matplotlib pandas
"""
import csv, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
OUT = ROOT / "Model Run Outputs"
HERE = Path(__file__).parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)

# ---------- shared style (validated palette: categorical slots 1-4; blue ramp for magnitude) ----------
SERIES = {"English": "#2a78d6", "Bengali": "#eb6834", "Romanized": "#1baf7a", "Code-mixed (13)": "#eda100"}
MARK = {"English": "o", "Bengali": "s", "Romanized": "^", "Code-mixed (13)": "D"}
DASH = {"English": "-", "Bengali": (0, (5, 2)), "Romanized": (0, (1.5, 1.5)), "Code-mixed (13)": (0, (5, 1.5, 1.5, 1.5))}
BLUE_RAMP = ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "legend.fontsize": 7, "pdf.fonttype": 42})

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG / f"{name}.pdf")

def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)

TEST_COLS = ["English", "Bengali", "Romanized", "Code-mixed (13)"]
TEST_KEY = {"English": "EN", "Bengali": "BN", "Romanized": "Banglish", "Code-mixed (13)": "CM"}
DEC_RUNS = {"English": "Outputs EN_FT", "Bengali": "Outputs BN FT", "Romanized": "Outputs_Banglish_FT",
            "Code-mixed": "outputs CM_FT", "Joint (all four)": "Outputs Joined FT"}

# ---------- A2: transfer heatmaps ----------
def dec_acc(folder):
    rows = {r["dataset"]: float(r["accuracy"]) for r in csv.DictReader(open(OUT / folder / "summary_all_datasets.csv"))}
    rows.setdefault("Banglish", rows.get("Romantized"))
    return [rows[TEST_KEY[c]] for c in TEST_COLS]

dec = np.array([dec_acc(f) for f in DEC_RUNS.values()])
tf = pd.read_csv(HERE / "tfidf_baseline.csv")
tf_rows = {"English": "Direct English", "Bengali": "Direct Bengali", "Romanized": "Direct Romanized",
           "Code-mixed": "Direct Code-mixed", "Joint (all four)": "Joint"}
tfm = np.array([[tf[(tf.train == t) & (tf.test == c)].accuracy.iloc[0] for c in TEST_COLS] for t in tf_rows.values()])

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55), constrained_layout=True)
for ax, M, title in zip(axes, (dec, tfm), ("(a) Decision model (Qwen3.5-0.8B)", "(b) Char n-gram TF-IDF baseline")):
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 0.55 else INK)
    ax.axhline(3.5, color="white", lw=3)                       # separate the joint row
    ax.set_xticks(range(4)); ax.set_xticklabels(["English", "Bengali", "Romanized", "Code-mixed\n(13 intents)"])
    ax.set_yticks(range(5)); ax.set_yticklabels(list(DEC_RUNS))
    ax.set_xlabel("Test form"); ax.set_title(title, loc="left", color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
axes[0].set_ylabel("Training data")
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01); cb.set_label("Accuracy"); cb.outline.set_visible(False)
save(fig, "fig_transfer_heatmaps")

# ---------- per-item joint predictions ----------
def preds(folder, form):
    stem = TEST_KEY[form]
    p = OUT / folder / f"{stem}_predictions.csv"
    d = pd.read_csv(p, encoding="utf-8-sig")
    return d.conf.astype(float).values, (d.pred == d.intent).values

def tf_preds(form):
    d = pd.read_csv(HERE / "tfidf_predictions" / f"Joint__{form.split(' ')[0]}.csv", encoding="utf-8-sig")
    return d.conf.astype(float).values, (d.pred == d.intent).values

# ---------- A3: risk-coverage (accuracy vs coverage) ----------
fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
rc_rows = []
for form in TEST_COLS:
    conf, ok = preds(DEC_RUNS["Joint (all four)"], form)
    order = np.argsort(-conf, kind="stable"); okc = ok[order]
    cov = np.arange(1, len(okc) + 1) / len(okc); acc = np.cumsum(okc) / np.arange(1, len(okc) + 1)
    ax.plot(cov * 100, acc * 100, color=SERIES[form], ls=DASH[form], lw=1.6, label=form)
    ax.plot([100], [acc[-1] * 100], marker=MARK[form], color=SERIES[form], ms=4.5, mec="white", mew=0.8)
    for c in (0.5, 0.8, 1.0):
        rc_rows.append(dict(form=form, coverage=c, accuracy=acc[max(0, int(round(c * len(okc))) - 1)]))
ax.axvline(80, color=MUTED, lw=0.8, ls=(0, (3, 2))); ax.text(79, 52, "abstain on\nleast confident 20%", ha="right", fontsize=6.5, color=MUTED)
ax.set_xlim(10, 101); ax.set_ylim(50, 101)
ax.set_xlabel("Coverage: share of inputs answered (%)"); ax.set_ylabel("Accuracy on answered inputs (%)")
ax.legend(frameon=False, loc="lower left"); style(ax)
save(fig, "fig_risk_coverage")
pd.DataFrame(rc_rows).to_csv(FIG / "fig_risk_coverage_data.csv", index=False)

# ---------- A5: reliability diagrams, decision model vs TF-IDF ----------
def reliability(conf, ok, bins=10):
    edges = np.linspace(0, 1, bins + 1); out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 10: out.append(((lo + hi) / 2, conf[m].mean(), ok[m].mean(), int(m.sum())))
    ece = sum(n / len(conf) * abs(a - c) for _, c, a, n in out)
    return out, ece

fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3), sharey=True, constrained_layout=True)
rel_rows = []
for ax, form in zip(axes, ["English", "Bengali", "Romanized"]):
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (3, 2)))
    for label, (conf, ok), color, mk, ls in (
            ("Decision model", preds(DEC_RUNS["Joint (all four)"], form), "#2a78d6", "o", "-"),
            ("TF-IDF", tf_preds(form), "#eb6834", "s", (0, (5, 2)))):
        pts, ece = reliability(conf, ok)
        ax.plot([p[1] for p in pts], [p[2] for p in pts], color=color, marker=mk, ms=3.5, lw=1.4, ls=ls,
                mec="white", mew=0.6, label=f"{label} (ECE {ece:.3f})")
        rel_rows += [dict(form=form, model=label, bin_mid=p[0], mean_conf=p[1], accuracy=p[2], n=p[3]) for p in pts]
    ax.set_title(form, loc="left", color=INK); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence"); ax.legend(frameon=False, loc="upper left", fontsize=6.3); style(ax)
axes[0].set_ylabel("Accuracy")
save(fig, "fig_reliability")
pd.DataFrame(rel_rows).to_csv(FIG / "fig_reliability_data.csv", index=False)
