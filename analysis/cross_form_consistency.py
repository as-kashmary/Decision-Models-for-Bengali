"""Cross-form consistency: does a model give the same intent for the same utterance
in English, Bengali and Romanized form? Uses the parallel test IDs (same 2,000 IDs in
the three test files). Needs no GPU: reads saved predictions only. Run:
    python cross_form_consistency.py <project folder>
"""
import csv, sys, itertools
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
OUT, DATA = ROOT / "Model Run Outputs", ROOT / "Dataset"
TEST = {"EN": "test_en_2000.csv", "BN": "test_bn_2000.csv", "ROM": "banglish_massive_test.csv"}
STEMS = {"EN": ["EN"], "BN": ["BN"], "ROM": ["Banglish", "Romantized"]}
MODELS = {"Base": "output_base", "Direct English": "Outputs EN_FT", "Direct Bengali": "Outputs BN FT",
          "Direct Romanized": "Outputs_Banglish_FT", "Direct code-mixed": "outputs CM_FT",
          "Sequential + Bengali": "Outputs plus_Bengali", "Sequential + code-mixed": "Outputs plus_CM",
          "Sequential final": "Outputs plus_romanized", "Joint": "Outputs Joined FT"}

def read(p): return list(csv.DictReader(open(p, encoding="utf-8-sig", newline="")))
utt2id = {f: {r["utt"]: (r.get("id") or r.get("Id")) for r in read(DATA / n)} for f, n in TEST.items()}

def preds(folder, form):
    for s in STEMS[form]:
        p = OUT / folder / f"{s}_predictions.csv"
        if p.exists():
            return {utt2id[form][r["utt"]]: (r["pred"], r["intent"]) for r in read(p) if r["utt"] in utt2id[form]}
    return None

rows = []
for name, folder in MODELS.items():
    P = {f: preds(folder, f) for f in TEST}
    if any(v is None for v in P.values()): continue
    ids = set.intersection(*(set(v) for v in P.values()))
    same3 = sum(len({P[f][i][0] for f in TEST}) == 1 for i in ids) / len(ids)
    allright = sum(all(P[f][i][0] == P[f][i][1] for f in TEST) for i in ids) / len(ids)
    pair = {f"{a}-{b}": sum(P[a][i][0] == P[b][i][0] for i in ids) / len(ids) for a, b in itertools.combinations(TEST, 2)}
    rows.append(dict(model=name, n=len(ids), agree_all_3=same3, correct_all_3=allright, **pair))

with open(Path(__file__).with_name("cross_form_consistency.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['model']:24s} n={r['n']} same intent in all 3 forms {100*r['agree_all_3']:5.1f}% | correct in all 3 {100*r['correct_all_3']:5.1f}% "
          f"| EN-BN {100*r['EN-BN']:5.1f}% EN-ROM {100*r['EN-ROM']:5.1f}% BN-ROM {100*r['BN-ROM']:5.1f}%")
