"""Evaluate embedding quality on RUHSOLD test set.

Metrics
  Linear probe (logistic regression on frozen embeddings): macro-F1, MCC, ROC-AUC, FNR, FPR
  Cost-sensitive probe: same, with class weight --fn_weight on hate speech
                        (penalises missed hate speech, i.e. false negatives)
  kNN accuracy (k=10, cosine)
  Alignment and uniformity (Wang & Isola, 2020): lower is better for both
  Silhouette score (cosine) of the two classes

Example
  python src/evaluate.py --model models/mnrl --name mnrl --out results
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, os.path.dirname(__file__))
from data import download_ruhsold, load_split  # noqa: E402


def probe_metrics(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "mcc": matthews_corrcoef(y, pred),
        "roc_auc": roc_auc_score(y, score),
        "fnr": fn / max(fn + tp, 1),  # hate speech missed
        "fpr": fp / max(fp + tn, 1),  # normal text flagged
    }


def alignment_uniformity(emb, y, rng, n_pairs=2000, t=2.0):
    """Embeddings must be L2-normalised. Alignment over same-class pairs."""
    e = torch.tensor(emb)
    idx = {c: np.where(y == c)[0] for c in (0, 1)}
    a_list = []
    for c in (0, 1):
        i = rng.choice(idx[c], n_pairs // 2)
        j = rng.choice(idx[c], n_pairs // 2)
        a_list.append((e[i] - e[j]).norm(dim=1).pow(2))
    align = torch.cat(a_list).mean().item()
    sub = e[rng.choice(len(e), min(len(e), 2000), replace=False)]
    unif = torch.pdist(sub, p=2).pow(2).mul(-t).exp().mean().log().item()
    return align, unif


def evaluate(model_path, data_dir="data", fn_weight=2.0, max_rows=None, seed=42, batch_size=64):
    download_ruhsold(data_dir)
    tr = load_split("train", data_dir, max_rows, seed)
    te = load_split("test", data_dir, max_rows, seed)
    model = SentenceTransformer(model_path)
    model.max_seq_length = 64
    enc = lambda t: model.encode(t, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)  # noqa: E731
    Xtr, Xte = enc(tr["text"].tolist()), enc(te["text"].tolist())
    ytr, yte = tr["y_hate"].values, te["y_hate"].values

    res = {}
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    res["probe"] = probe_metrics(yte, clf.predict(Xte), clf.predict_proba(Xte)[:, 1])
    clf_w = LogisticRegression(max_iter=2000, C=1.0, class_weight={0: 1.0, 1: fn_weight}).fit(Xtr, ytr)
    res["probe_cost_sensitive"] = probe_metrics(yte, clf_w.predict(Xte), clf_w.predict_proba(Xte)[:, 1])
    knn = KNeighborsClassifier(n_neighbors=10, metric="cosine").fit(Xtr, ytr)
    res["knn_acc"] = float((knn.predict(Xte) == yte).mean())
    rng = np.random.default_rng(seed)
    res["alignment"], res["uniformity"] = alignment_uniformity(Xte, yte, rng)
    res["silhouette"] = float(silhouette_score(Xte, yte, metric="cosine"))
    res = json.loads(json.dumps(res, default=float))
    return res, Xte, yte


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="trained folder or HF model id (baseline)")
    p.add_argument("--name", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--fn_weight", type=float, default=2.0)
    p.add_argument("--max_rows", type=int, default=None)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res, X, y = evaluate(a.model, a.data_dir, a.fn_weight, a.max_rows)
    with open(os.path.join(a.out, f"{a.name}.json"), "w") as f:
        json.dump(res, f, indent=2)
    np.savez_compressed(os.path.join(a.out, f"{a.name}_test_emb.npz"), X=X, y=y)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
