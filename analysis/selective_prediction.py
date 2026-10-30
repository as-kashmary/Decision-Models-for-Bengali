"""Selective prediction from saved top-1 confidences: accuracy when the model answers
only its most confident X% of inputs, and AURC (area under the risk-coverage curve,
lower is better). Needs no GPU. Run:
    python selective_prediction.py <project folder>
"""
import csv, sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
OUT = ROOT / "Model Run Outputs"
MODELS = {"Direct English": "Outputs EN_FT", "Direct Bengali": "Outputs BN FT", "Direct Romanized": "Outputs_Banglish_FT",
          "Sequential final": "Outputs plus_romanized", "Joint": "Outputs Joined FT"}
FORMS = {"English": ["EN"], "Bengali": ["BN"], "Romanized": ["Banglish", "Romantized"], "Code-mixed (13)": ["CM"]}

def load(folder, form):
    for s in FORMS[form]:
        p = OUT / folder / f"{s}_predictions.csv"
        if p.exists():
            return [(float(r["conf"]), r["pred"] == r["intent"]) for r in csv.DictReader(open(p, encoding="utf-8-sig"))]

rows = []
for m, folder in MODELS.items():
    for form in FORMS:
        d = sorted(load(folder, form), key=lambda x: -x[0]); n = len(d)
        cum, risks = 0, []
        for k, (_, ok) in enumerate(d, 1):
            cum += ok; risks.append(1 - cum / k)
        acc_at = lambda c: sum(ok for _, ok in d[:max(1, round(c * n))]) / max(1, round(c * n))
        rows.append(dict(model=m, form=form, n=n, acc_full=acc_at(1.0), acc_cov80=acc_at(0.8), acc_cov50=acc_at(0.5),
                         aurc=sum(risks) / n))
with open(Path(__file__).with_name("selective_prediction.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['model']:17s} {r['form']:16s} acc@100% {r['acc_full']:.3f}  acc@80% {r['acc_cov80']:.3f}  acc@50% {r['acc_cov50']:.3f}  AURC {r['aurc']:.3f}")
