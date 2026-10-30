"""A4: error analysis of the joint decision model, Romanized vs Bengali (Bangla script).

Uses the parallel test IDs: the same utterance appears in the Bengali and Romanized
test sets, so differences are due to the written form alone. Writes
a4_error_analysis.md (tables for the paper) and a4_domain_accuracy.csv. Run:
    python a4_error_analysis.py <project folder>
"""
import csv, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
RUN = ROOT / "Model Run Outputs" / "Outputs Joined FT"
DATA = ROOT / "Dataset"
HERE = Path(__file__).parent

def read(p): return list(csv.DictReader(open(p, encoding="utf-8-sig", newline="")))
def ids(name): return {r["utt"]: (r.get("id") or r.get("Id")) for r in read(DATA / name)}
dom = lambda label: label.split("_")[0]

bn_id, rom_id, en_id = ids("test_bn_2000.csv"), ids("banglish_massive_test.csv"), ids("test_en_2000.csv")
BN = {bn_id[r["utt"]]: r for r in read(RUN / "BN_predictions.csv")}
ROM = {rom_id[r["utt"]]: r for r in read(RUN / "Banglish_predictions.csv")}
EN = {en_id[r["utt"]]: r for r in read(RUN / "EN_predictions.csv")}
shared = sorted(BN.keys() & ROM.keys() & EN.keys(), key=int)
ok = lambda R, i: R[i]["pred"] == R[i]["intent"]

# 1. accuracy by domain, same items in each form
by_dom = defaultdict(lambda: Counter())
for i in shared:
    d = dom(BN[i]["intent"]); c = by_dom[d]
    c["n"] += 1; c["en"] += ok(EN, i); c["bn"] += ok(BN, i); c["rom"] += ok(ROM, i)
dom_rows = sorted(({"domain": d, "n": c["n"], "en": c["en"] / c["n"], "bn": c["bn"] / c["n"], "rom": c["rom"] / c["n"],
                    "gap_bn_minus_rom": (c["bn"] - c["rom"]) / c["n"]} for d, c in by_dom.items()),
                  key=lambda r: -r["gap_bn_minus_rom"])
with open(HERE / "a4_domain_accuracy.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(dom_rows[0])); w.writeheader(); w.writerows(dom_rows)

# 2. outcome table on parallel items
both = sum(ok(BN, i) and ok(ROM, i) for i in shared); bn_only = sum(ok(BN, i) and not ok(ROM, i) for i in shared)
rom_only = sum(ok(ROM, i) and not ok(BN, i) for i in shared); neither = len(shared) - both - bn_only - rom_only

# 3. top confusions
def confusions(R, k=8):
    return Counter((R[i]["intent"], R[i]["pred"]) for i in shared if not ok(R, i)).most_common(k)
def cross_share(R):
    err = [i for i in shared if not ok(R, i)]
    return sum(dom(R[i]["intent"]) != dom(R[i]["pred"]) for i in err) / len(err), len(err)

# 4. examples: right in Bangla script, wrong in Romanized
examples = [i for i in shared if ok(BN, i) and not ok(ROM, i)]
examples = sorted(examples, key=lambda i: -float(ROM[i]["conf"]))[:8]

L = ["# A4: Error analysis of the joint decision model (Romanized vs Bengali)", "",
     f"Same {len(shared)} utterances in English, Bengali (Bangla script) and Romanized form; differences come from the written form alone.", "",
     "## Outcome on parallel items (Bengali vs Romanized)", "",
     "| Outcome | Items | Share |", "|---|---|---|",
     f"| Correct in both | {both} | {100*both/len(shared):.1f}% |",
     f"| Correct only in Bangla script | {bn_only} | {100*bn_only/len(shared):.1f}% |",
     f"| Correct only in Romanized | {rom_only} | {100*rom_only/len(shared):.1f}% |",
     f"| Wrong in both | {neither} | {100*neither/len(shared):.1f}% |", "",
     "## Accuracy by domain (largest Bengali-minus-Romanized gap first)", "",
     "| Domain | Items | English | Bengali | Romanized | Gap (BN - ROM, points) |", "|---|---|---|---|---|---|"]
L += [f"| {r['domain']} | {r['n']} | {r['en']:.3f} | {r['bn']:.3f} | {r['rom']:.3f} | {100*r['gap_bn_minus_rom']:+.1f} |" for r in dom_rows]
for name, R in (("Bengali", BN), ("Romanized", ROM)):
    share, n = cross_share(R)
    L += ["", f"## Most frequent confusions, {name} (gold -> predicted)", "",
          f"{n} errors; {100*share:.0f}% cross into a different domain.", "",
          "| Gold intent | Predicted intent | Count |", "|---|---|---|"]
    L += [f"| {g} | {p} | {c} |" for (g, p), c in confusions(R)]
L += ["", "## Examples: correct in Bangla script, wrong in Romanized (most confident errors)", "",
      "| Gold | Bangla script | Romanized | Romanized prediction (confidence) |", "|---|---|---|---|"]
L += [f"| {BN[i]['intent']} | {BN[i]['utt']} | {ROM[i]['utt']} | {ROM[i]['pred']} ({float(ROM[i]['conf']):.2f}) |" for i in examples]
(HERE / "a4_error_analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("saved", HERE / "a4_error_analysis.md")
