"""Tokens per utterance per written form, with the Qwen3.5 tokenizer.

Fertility = tokens per whitespace word. A form the base model saw little of in
pretraining is usually split into more, shorter pieces. Run:
    python token_analysis.py <Dataset folder>
Needs: pip install tokenizers huggingface_hub
"""
import csv, sys, statistics as st
from pathlib import Path
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "Dataset"
tok = Tokenizer.from_file(hf_hub_download("unsloth/Qwen3.5-0.8B", "tokenizer.json"))

FILES = {
    "English":    ("en_3000match.csv", "test_en_2000.csv"),
    "Bengali":    ("bn_3000_clean_unique.csv", "test_bn_2000.csv"),
    "Romanized":  ("banglish_massive_train.csv", "banglish_massive_test.csv"),
    "Code-mixed": ("train_set_codemixed_massive60_fixed.csv", "test_set_codemixed_massive60_fixed.csv"),
}

def rows(name):
    with open(DATA / name, encoding="utf-8-sig", newline="") as f:
        return [(r.get("id") or r.get("Id"), str(r["utt"])) for r in csv.DictReader(f)]

def stats(utts):
    toks = [len(tok.encode(u, add_special_tokens=False).ids) for u in utts]
    words = [max(1, len(u.split())) for u in utts]
    split_words = sum(len(tok.encode(w, add_special_tokens=False).ids) >= 3 for u in utts for w in u.split())
    return dict(n=len(utts), tokens_mean=st.mean(toks), tokens_median=st.median(toks),
                words_mean=st.mean(words), fertility=sum(toks) / sum(words),
                chars_per_token=sum(len(u) for u in utts) / sum(toks),
                pct_words_3plus_tokens=100 * split_words / sum(words)), toks

out, per_id = [], {}
for form, (tr, te) in FILES.items():
    for split, name in (("train", tr), ("test", te)):
        r = rows(name)
        s, toks = stats([u for _, u in r])
        out.append(dict(form=form, split=split, **s))
        if split == "test": per_id[form] = {i: t for (i, _), t in zip(r, toks)}

# Same utterance IDs in English, Bengali and Romanized test sets: paired ratio
shared = per_id["English"].keys() & per_id["Bengali"].keys() & per_id["Romanized"].keys()
ratio = lambda a, b: st.mean(per_id[a][i] / max(1, per_id[b][i]) for i in shared)

res = Path(__file__).with_name("token_stats.csv")
with open(res, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
for r in out:
    print(f"{r['form']:11s} {r['split']:5s} n={r['n']:5d} tokens/utt {r['tokens_mean']:5.1f} (median {r['tokens_median']:4.0f}) "
          f"words/utt {r['words_mean']:4.1f} fertility {r['fertility']:.2f} chars/token {r['chars_per_token']:.2f} "
          f"words split into 3+ tokens {r['pct_words_3plus_tokens']:.1f}%")
print(f"\nPaired test items (n={len(shared)}): Romanized/English tokens {ratio('Romanized','English'):.2f}x, "
      f"Bengali/English {ratio('Bengali','English'):.2f}x, Romanized/Bengali {ratio('Romanized','Bengali'):.2f}x")
print("saved", res)
