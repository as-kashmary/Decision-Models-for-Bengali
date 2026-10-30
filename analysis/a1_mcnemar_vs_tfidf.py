"""A1: exact McNemar tests, joint decision model vs joint TF-IDF baseline, per test form.
Items are paired by (utterance, gold intent); Holm correction across the four forms.
Run after tfidf_baseline.py:  python a1_mcnemar_vs_tfidf.py <project folder>
"""
import csv, sys, math
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
DEC = ROOT / "Model Run Outputs" / "Outputs Joined FT"
TF = Path(__file__).with_name("tfidf_predictions")
FORMS = {"English": ("EN", "English"), "Bengali": ("BN", "Bengali"),
         "Romanized": ("Banglish", "Romanized"), "Code-mixed (13)": ("CM", "Code-mixed")}

def load(p):
    return {(r["utt"], r["intent"]): r["pred"] == r["intent"] for r in csv.DictReader(open(p, encoding="utf-8-sig", newline=""))}

def exact_p(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

rows = []
for form, (dstem, tstem) in FORMS.items():
    A, B = load(DEC / f"{dstem}_predictions.csv"), load(TF / f"Joint__{tstem}.csv")
    keys = A.keys() & B.keys()
    b = sum(A[k] and not B[k] for k in keys); c = sum(B[k] and not A[k] for k in keys)
    rows.append(dict(form=form, n=len(keys), acc_decision=sum(A[k] for k in keys) / len(keys),
                     acc_tfidf=sum(B[k] for k in keys) / len(keys), decision_only=b, tfidf_only=c, p=exact_p(b, c)))
order = sorted(range(len(rows)), key=lambda i: rows[i]["p"]); run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (len(rows) - rank) * rows[i]["p"])); rows[i]["p_holm"] = run
with open(Path(__file__).with_name("a1_mcnemar_vs_tfidf.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['form']:16s} n={r['n']} decision {r['acc_decision']:.4f} vs TF-IDF {r['acc_tfidf']:.4f} "
          f"({100*(r['acc_decision']-r['acc_tfidf']):+.1f} pts) | decision only {r['decision_only']}, TF-IDF only {r['tfidf_only']} "
          f"| Holm p={r['p_holm']:.2e}")
