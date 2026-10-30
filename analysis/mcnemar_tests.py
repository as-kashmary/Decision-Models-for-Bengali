"""Paired McNemar tests on saved per-item predictions.

Two models are compared on the same test items. Only the items where exactly
one model is right carry information:
  b = model A right, model B wrong
  c = model A wrong, model B right
Exact test: two-sided binomial test of b against b + c with p = 0.5.
Items are matched by (utterance, gold intent), so runs that dropped a different
train-overlap row still pair correctly.
"""
import csv, sys, math
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "Model Run Outputs"
RUNS = {
    "Joint": "Outputs Joined FT",
    "Sequential final": "Outputs plus_romanized",
    "Direct Bengali": "Outputs BN FT",
    "Direct English": "Outputs EN_FT",
}
FORM_FILE = {"English": "EN", "Bengali": "BN", "Romanized": "Banglish", "Code-mixed (13 intents)": "CM"}
ALT_NAMES = {"Banglish": ["Banglish", "Romantized"]}   # the EN_FT run used an older file name

def load(run, form):
    for stem in ALT_NAMES.get(FORM_FILE[form], [FORM_FILE[form]]):
        p = ROOT / RUNS[run] / f"{stem}_predictions.csv"
        if p.exists():
            with open(p, encoding="utf-8-sig", newline="") as f:
                return {(r["utt"], r["intent"]): r["pred"] == r["intent"] for r in csv.DictReader(f)}
    raise FileNotFoundError(f"{run} / {form}")

def exact_p(b, c):
    n, k = b + c, min(b, c)
    if n == 0: return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)

def compare(a, b_, form):
    A, B = load(a, form), load(b_, form)
    keys = A.keys() & B.keys()
    b = sum(A[k] and not B[k] for k in keys)
    c = sum(B[k] and not A[k] for k in keys)
    acc_a = sum(A[k] for k in keys) / len(keys); acc_b = sum(B[k] for k in keys) / len(keys)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0   # continuity-corrected, for reference
    return dict(form=form, A=a, B=b_, n=len(keys), acc_A=acc_a, acc_B=acc_b, diff_pts=100 * (acc_a - acc_b),
                A_only=b, B_only=c, chi2=chi2, p=exact_p(b, c))

tests = [compare("Joint", "Sequential final", f) for f in FORM_FILE]
tests.append(compare("Direct Bengali", "Direct English", "Romanized"))

m = len(tests)   # Holm correction across all tests in this family
order = sorted(range(m), key=lambda i: tests[i]["p"])
running = 0.0
for rank, i in enumerate(order):
    running = max(running, min(1.0, (m - rank) * tests[i]["p"]))
    tests[i]["p_holm"] = running

hdr = ["form", "A", "B", "n", "acc_A", "acc_B", "diff_pts", "A_only", "B_only", "chi2", "p", "p_holm"]
out = Path(__file__).with_name("mcnemar_results.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(tests)
for t in tests:
    sig = "significant" if t["p_holm"] < 0.05 else "not significant"
    print(f"{t['form']:24s} {t['A']} vs {t['B']}: n={t['n']} acc {t['acc_A']:.4f} vs {t['acc_B']:.4f} "
          f"({t['diff_pts']:+.1f} pts) | A only {t['A_only']}, B only {t['B_only']} | "
          f"p={t['p']:.2e}, Holm p={t['p_holm']:.2e} -> {sig}")
print("saved", out)
