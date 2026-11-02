"""
16_train_embeddings.py  -  train a small classifier on SaProt embeddings (frozen).

Reads  data/embeddings_db1.json  (from Colab) and data/combined_db1.json
Writes data/embedding_model_report.json

This is the "frozen embeddings + small classifier" step: the protein model is NOT
fine-tuned. We take the vectors it already produces and train a simple classifier
on top. With 13 labels this WILL overfit, so the point is not to win; it is to have
a working, honestly-evaluated trained pipeline ready for when more labels arrive.

Honesty measures, all built in:
  - Leave-one-POSITION-out, never splitting a position across train/test.
  - Every fitted step (scaler, PCA, classifier) is fit on the training fold ONLY,
    inside the loop, so no information leaks from test to train.
  - Dimensionality is cut hard (PCA to a few components) because 1280 features on
    13 points is hopeless otherwise.
  - The result is compared head-to-head with the zero-shot score, which needs no
    training at all.

Feature choices tried: the mutant-context embedding, and the change it makes
(mutant minus wild-type). The second is usually the better variant-effect feature.

Run on Windows, no GPU:
    python scripts\\16_train_embeddings.py

Needs: scikit-learn, numpy
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "embeddings_db1.json"
if not EMB.exists():
    raise SystemExit("data/embeddings_db1.json is missing. Run colab_extract_embeddings.py "
                     "in Colab first and put the file in data\\.")

emb = json.load(open(EMB))
zs = {v["name"]: v["score"] for v in json.load(open(ROOT / "data" / "combined_db1.json"))["variants"]
      if v["kind"] == "single"}

names = list(emb)
print(f"mutations with embeddings: {len(names)}")


def rows_for(task):
    out = []
    for n in names:
        e = emb[n]
        if task == "overall":
            y = 1 if e["label"] == "resistant" else 0
        elif task == "typeI":
            y = 1 if e["type_I"] == "resistant" else (0 if e["type_I"] == "no_effect" else None)
        else:
            y = 1 if e["type_II"] == "resistant" else (0 if e["type_II"] == "no_effect" else None)
        if y is None:
            continue
        mut = np.array(e["emb_mut"]); wt = np.array(e["emb_wt"])
        out.append(dict(name=n, pos=e["position"], y=y,
                        f_mut=mut, f_delta=mut - wt, zs=zs.get(n)))
    return out


def auc(scores, ys):
    pos = [s for s, y in zip(scores, ys) if y == 1]
    neg = [s for s, y in zip(scores, ys) if y == 0]
    if not pos or not neg:
        return None
    pr = [(p, q) for p in pos for q in neg]
    return sum(1.0 if p > q else 0.5 if p == q else 0.0 for p, q in pr) / len(pr)


def lopo(data, feat, n_pca=4):
    by = defaultdict(list)
    for r in data:
        by[r["pos"]].append(r)
    probs, preds, ys = [], [], []
    for held in by:
        tr = [r for p in by if p != held for r in by[p]]
        if len({r["y"] for r in tr}) < 2:
            continue
        X = np.vstack([r[feat] for r in tr]); y = np.array([r["y"] for r in tr])
        sc = StandardScaler().fit(X)
        k = min(n_pca, X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=k).fit(sc.transform(X))
        clf = LogisticRegression(C=0.3, max_iter=2000).fit(pca.transform(sc.transform(X)), y)
        for r in by[held]:
            x = pca.transform(sc.transform(r[feat][None, :]))
            probs.append(clf.predict_proba(x)[0, 1]); preds.append(int(probs[-1] >= 0.5)); ys.append(r["y"])
    if not ys:
        return None
    return dict(auc=auc(probs, ys), acc=sum(p == y for p, y in zip(preds, ys)) / len(ys), n=len(ys))


report = {}
print(f"\n{'task':8} {'model':38} {'ranking':>8} {'accuracy':>9}")
for task, label in (("overall", "overall"), ("typeI", "Type I"), ("typeII", "Type II")):
    data = rows_for(task)
    zs_auc = auc([r["zs"] and -r["zs"] for r in data], [r["y"] for r in data])
    print(f"{label:8} {'zero-shot score (no training)':38} {zs_auc:8.2f} {'--':>9}")
    for feat, fname in (("f_mut", "trained: embedding (mutant)"),
                        ("f_delta", "trained: embedding change (mut - wt)")):
        res = lopo(data, feat)
        if res:
            print(f"{'':8} {fname:38} {res['auc']:8.2f} {res['acc']:8.2f}")
            report[f"{label} | {fname}"] = res
    report[f"{label} | zero-shot"] = dict(auc=zs_auc)
    print()

print("Expectation on 13 labels: the trained models do NOT beat the zero-shot score.")
print("The deliverable is the working, leak-free pipeline, ready for more labels.")
json.dump(report, open(ROOT / "data" / "embedding_model_report.json", "w"), indent=1, default=str)
print("\nwritten: data/embedding_model_report.json")
