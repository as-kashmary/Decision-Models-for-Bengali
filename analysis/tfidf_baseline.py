"""CPU baseline: character n-gram TF-IDF + logistic regression, trained on the same
12,000 joint training rows as the decision model and scored on the same four test
sets (all 60 intents). Trains in seconds on a laptop CPU; no GPU or Kaggle needed.
Also trains one model per form (direct) for the transfer matrix, and saves
per-item predictions to tfidf_predictions/. Run:
    python tfidf_baseline.py <project folder>
Needs: pip install scikit-learn pandas
"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
D = ROOT / "Dataset"
imap = pd.read_csv(D / "massive_intent_mapping.csv")

def load(name, numeric):
    df = pd.read_csv(D / name).rename(columns={"Id": "id"})
    if numeric:
        df = df.merge(imap, left_on="intent", right_on="intent_id").drop(columns=["intent"]).rename(columns={"intent_label": "intent"})
    return df[["utt", "intent"]].dropna()

TRAIN = {"English": load("en_3000match.csv", True), "Bengali": load("bn_3000_clean_unique.csv", True),
         "Code-mixed": load("train_set_codemixed_massive60_fixed.csv", False), "Romanized": load("banglish_massive_train.csv", False)}
TEST = {"English": load("test_en_2000.csv", False), "Bengali": load("test_bn_2000.csv", False),
        "Romanized": load("banglish_massive_test.csv", False), "Code-mixed (13)": load("test_set_codemixed_massive60_fixed.csv", False)}

def model():
    return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), sublinear_tf=True, min_df=2),
                         LogisticRegression(C=10, max_iter=3000))

def run(name, train_df):
    seen = set(train_df.utt)
    m = model(); t = time.perf_counter(); m.fit(train_df.utt, train_df.intent); fit_s = time.perf_counter() - t
    out = []
    for form, te in TEST.items():
        te = te[~te.utt.isin(seen)]                      # same overlap rule as the notebooks
        t = time.perf_counter(); p = m.predict(te.utt); ms = 1000 * (time.perf_counter() - t) / len(te)
        prob = m.predict_proba(te.utt).max(1); ok = (p == te.intent.values)
        pred_dir = Path(__file__).with_name("tfidf_predictions"); pred_dir.mkdir(exist_ok=True)
        pd.DataFrame({"utt": te.utt.values, "intent": te.intent.values, "pred": p, "conf": prob}).to_csv(
            pred_dir / f"{name.replace(' ', '_')}__{form.split(' ')[0]}.csv", index=False, encoding="utf-8-sig")
        bins = np.linspace(0, 1, 11); ece = sum(((prob > lo) & (prob <= hi)).mean() * abs(ok[(prob > lo) & (prob <= hi)].mean() - prob[(prob > lo) & (prob <= hi)].mean())
                                                for lo, hi in zip(bins[:-1], bins[1:]) if ((prob > lo) & (prob <= hi)).any())
        out.append(dict(train=name, test=form, n=len(te), accuracy=accuracy_score(te.intent, p),
                        macro_F1=f1_score(te.intent, p, average="macro", labels=sorted(set(te.intent)), zero_division=0),
                        ECE=ece, ms_per_utt_batch=ms, fit_seconds=fit_s))
    return out

rows = run("Joint", pd.concat(TRAIN.values(), ignore_index=True))
for form, df in TRAIN.items():
    rows += run(f"Direct {form}", df)
res = pd.DataFrame(rows)
res.to_csv(Path(__file__).with_name("tfidf_baseline.csv"), index=False)
print(res.pivot(index="train", columns="test", values="accuracy").round(3).to_string())
print("\nJoint macro-F1:", res[res.train == "Joint"].set_index("test").macro_F1.round(3).to_dict())
print("Joint ECE:", res[res.train == "Joint"].set_index("test").ECE.round(3).to_dict())
print(f"fit {res.fit_seconds.iloc[0]:.1f}s on CPU; prediction {res[res.train=='Joint'].ms_per_utt_batch.mean():.2f} ms/utt (batched, CPU)")
