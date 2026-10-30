"""Label-bias analysis: how training on the 13-intent code-mixed set shifts predictions.

For each model and test form (English, Bengali, Romanized):
  - share of predictions that fall in the 13 code-mixed intents (vs the true share)
  - accuracy on gold items inside the 13 intents, and on the other 47
Writes label_bias.csv and label_bias.png / .pdf. Run:
    python label_bias.py <project folder>
Needs: pip install matplotlib
"""
import csv, sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
OUT = ROOT / "Model Run Outputs"
CM13 = {r["intent"] for r in csv.DictReader(open(ROOT / "Dataset/train_set_codemixed_massive60_fixed.csv", encoding="utf-8-sig"))}
assert len(CM13) == 13

STAGES = [  # x label, folder; sequential chain in order, joint as the reference
    ("1. English", "Outputs EN_FT"),
    ("2. + Bengali", "Outputs plus_Bengali"),
    ("3. + Code-mixed", "Outputs plus_CM"),
    ("4. + Romanized", "Outputs plus_romanized"),
]
REFS = [("Joint", "Outputs Joined FT"), ("Direct code-mixed", "outputs CM_FT")]
FORMS = {"English": ["EN"], "Bengali": ["BN"], "Romanized": ["Banglish", "Romantized"]}

def preds(folder, form):
    for stem in FORMS[form]:
        p = OUT / folder / f"{stem}_predictions.csv"
        if p.exists():
            return list(csv.DictReader(open(p, encoding="utf-8-sig", newline="")))
    raise FileNotFoundError(f"{folder} / {form}")

def measure(rows):
    inn = [r for r in rows if r["intent"] in CM13]; out = [r for r in rows if r["intent"] not in CM13]
    acc = lambda rs: sum(r["pred"] == r["intent"] for r in rs) / len(rs)
    return dict(true_share=len(inn) / len(rows), pred_share=sum(r["pred"] in CM13 for r in rows) / len(rows),
                acc_13=acc(inn), acc_47=acc(out))

table = []
for label, folder in STAGES + REFS:
    for form in FORMS:
        table.append(dict(model=label, form=form, **measure(preds(folder, form))))
with open(Path(__file__).with_name("label_bias.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
for r in table:
    print(f"{r['model']:18s} {r['form']:9s} predicted in 13: {100*r['pred_share']:4.1f}% (true {100*r['true_share']:.1f}%) "
          f"| acc on 13 {r['acc_13']:.3f} | acc on other 47 {r['acc_47']:.3f}")

# ---------- figure: two small multiples, one shared x (stages), joint as a reference mark ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR = {"English": "#2a78d6", "Bengali": "#eb6834", "Romanized": "#1baf7a"}   # categorical slots 1-3, validated
MARK = {"English": "o", "Bengali": "s", "Romanized": "^"}                       # secondary encoding
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

get = lambda model, form, key: next(r[key] for r in table if r["model"] == model and r["form"] == form)
xs = list(range(len(STAGES))); xj = len(STAGES) + 0.6
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75), sharex=True)
panels = [("pred_share", "Predictions falling in the 13 code-mixed intents", "Share of predictions (%)", 100),
          ("acc_47", "Accuracy on the other 47 intents", "Accuracy (%)", 100)]
for ax, (key, title, ylab, k) in zip(axes, panels):
    ax.axvspan(1.5, 2.5, color="#f3f2ee", zorder=0)                     # code-mixed stage
    if key == "pred_share":
        ax.axhline(100 * get("Joint", "English", "true_share"), color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(3.15, 100 * get("Joint", "English", "true_share") - 1.2, "true share 44%", color=MUTED, fontsize=6.5, va="top")
    for form in FORMS:
        ys = [k * get(s, form, key) for s, _ in STAGES]
        ax.plot(xs, ys, color=COLOR[form], lw=2, marker=MARK[form], ms=6, zorder=3,
                markeredgecolor="white", markeredgewidth=1.2, label=form)
        ax.plot([xj], [k * get("Joint", form, key)], marker=MARK[form], ms=6, color=COLOR[form],
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        if key == "acc_47":                                              # direct labels where lines are far apart
            ax.annotate(form, (xs[1], ys[1]), xytext=(0, -11 if form == "Bengali" else 7), textcoords="offset points",
                        fontsize=7, color=INK, ha="center", va="top" if form == "Bengali" else "baseline")
    ax.axvline(len(STAGES) - 0.2, color=GRID, lw=1)
    ax.set_xticks(xs + [xj]); ax.set_xticklabels([s for s, _ in STAGES] + ["Joint"], rotation=20, ha="right")
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    ax.set_ylabel(ylab); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[1].set_ylim(15, 90)
for ax in axes[:1]:
    ax.text(2, ax.get_ylim()[0], "13-intent\nstage", ha="center", va="bottom", fontsize=6.5, color=MUTED)
axes[0].legend(frameon=False, fontsize=7, loc="upper left")
# No in-figure title: IEEE puts the title in the caption.
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(Path(__file__).with_name(f"label_bias.{ext}"), dpi=200, bbox_inches="tight")
print("saved label_bias.csv, label_bias.png, label_bias.pdf")
