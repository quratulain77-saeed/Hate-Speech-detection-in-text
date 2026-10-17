"""RUHSOLD loading and pair construction for contrastive training.

RUHSOLD Task 1 labels: 0 = Abusive/Offensive, 1 = Normal.
We convert to y_hate (1 = Abusive/Offensive, 0 = Normal) so that the
"positive" class in all metrics (FNR, FPR, ROC-AUC) is hate speech.
"""
import os
import random
import urllib.request

import pandas as pd

RUHSOLD_URL = "https://raw.githubusercontent.com/haroonshakeel/roman_urdu_hate_speech/main/{name}"
SPLITS = {"train": "task_1_train.tsv", "validation": "task_1_validation.tsv", "test": "task_1_test.tsv"}


def download_ruhsold(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    for name in SPLITS.values():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(RUHSOLD_URL.format(name=name), path)
    return data_dir


def load_split(split, data_dir="data", max_rows=None, seed=42):
    path = os.path.join(data_dir, SPLITS[split])
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"],
                     quoting=3, dtype={"text": str}, on_bad_lines="skip")
    df = df.dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]
    df["y_hate"] = 1 - df["label"]
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    return df.reset_index(drop=True)


def simcse_pairs(df):
    """Unsupervised SimCSE: each sentence is its own positive.
    The two views differ only through dropout noise in the encoder."""
    return {"anchor": df["text"].tolist(), "positive": df["text"].tolist()}


def supervised_triplets(df, seed=42):
    """Supervised MNRL: (anchor, positive from same class, hard negative from other class)."""
    rng = random.Random(seed)
    by_class = {c: df.loc[df["y_hate"] == c, "text"].tolist() for c in (0, 1)}
    anchors, positives, negatives = [], [], []
    for text, c in zip(df["text"], df["y_hate"]):
        pos = text
        while pos == text and len(by_class[c]) > 1:
            pos = rng.choice(by_class[c])
        anchors.append(text)
        positives.append(pos)
        negatives.append(rng.choice(by_class[1 - c]))
    return {"anchor": anchors, "positive": positives, "negative": negatives}


def labelled_pairs(df, seed=42):
    """Classic contrastive loss (Hadsell et al., 2006): (s1, s2, label) with
    label 1 if same class, 0 otherwise. One same-class and one cross-class pair per sentence."""
    rng = random.Random(seed)
    by_class = {c: df.loc[df["y_hate"] == c, "text"].tolist() for c in (0, 1)}
    s1, s2, lab = [], [], []
    for text, c in zip(df["text"], df["y_hate"]):
        s1 += [text, text]
        s2 += [rng.choice(by_class[c]), rng.choice(by_class[1 - c])]
        lab += [1, 0]
    return {"sentence1": s1, "sentence2": s2, "label": lab}
