"""Collect results/*.json into a markdown table and draw t-SNE plots of test embeddings.

  python src/report.py --results results
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

ORDER = ["baseline", "simcse", "contrastive", "mnrl"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    a = p.parse_args()
    runs = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(a.results, "*.json"))}
    names = [n for n in ORDER if n in runs] + sorted(n for n in runs if n not in ORDER)

    head = ("| Model | Macro-F1 | MCC | ROC-AUC | FNR | FPR | FNR (cost-sens.) | kNN acc | "
            "Silhouette | Alignment | Uniformity |")
    lines = [head, "|" + "---|" * 11]
    for n in names:
        r, c = runs[n]["probe"], runs[n]["probe_cost_sensitive"]
        lines.append(f"| {n} | {r['macro_f1']:.3f} | {r['mcc']:.3f} | {r['roc_auc']:.3f} | {r['fnr']:.3f} | "
                     f"{r['fpr']:.3f} | {c['fnr']:.3f} | {runs[n]['knn_acc']:.3f} | {runs[n]['silhouette']:.3f} | "
                     f"{runs[n]['alignment']:.3f} | {runs[n]['uniformity']:.3f} |")
    table = "\n".join(lines)
    open(os.path.join(a.results, "results_table.md"), "w").write(table + "\n")
    print(table)

    embs = [(n, np.load(os.path.join(a.results, f"{n}_test_emb.npz"))) for n in names
            if os.path.exists(os.path.join(a.results, f"{n}_test_emb.npz"))]
    if not embs:
        return
    fig, axes = plt.subplots(1, len(embs), figsize=(4.2 * len(embs), 4), squeeze=False)
    for ax, (n, d) in zip(axes[0], embs):
        X, y = d["X"], d["y"]
        Z = TSNE(n_components=2, metric="cosine", init="pca", random_state=42,
                 perplexity=min(30, len(X) - 1)).fit_transform(X)
        for c, col, lab in [(0, "#4C78A8", "Normal"), (1, "#E45756", "Abusive/Offensive")]:
            ax.scatter(Z[y == c, 0], Z[y == c, 1], s=4, alpha=0.6, c=col, label=lab)
        ax.set_title(n)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0][0].legend(markerscale=3, fontsize=8, loc="lower left")
    fig.suptitle("RUHSOLD test embeddings (t-SNE)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.results, "tsne.png"), dpi=150)
    print("saved", os.path.join(a.results, "tsne.png"))


if __name__ == "__main__":
    main()
