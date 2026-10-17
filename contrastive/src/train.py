"""Contrastive fine-tuning of a multilingual encoder on RUHSOLD.

Methods
  simcse       Unsupervised SimCSE (Gao et al., 2021) using MultipleNegativesRankingLoss
               on (x, x) pairs; dropout provides the augmentation.
  mnrl         Supervised MNRL with an explicit hard negative from the other class
               (anchor, same-class positive, cross-class negative).
  contrastive  Classic margin-based contrastive loss on labelled pairs (OnlineContrastiveLoss).

Example
  python src/train.py --method mnrl --epochs 2 --batch_size 32 --out models/mnrl
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments, losses)
from sentence_transformers.training_args import BatchSamplers

sys.path.insert(0, os.path.dirname(__file__))
from data import download_ruhsold, labelled_pairs, load_split, simcse_pairs, supervised_triplets  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build(method, model, df, seed):
    if method == "simcse":
        return Dataset.from_dict(simcse_pairs(df)), losses.MultipleNegativesRankingLoss(model, scale=20.0)
    if method == "mnrl":
        return Dataset.from_dict(supervised_triplets(df, seed)), losses.MultipleNegativesRankingLoss(model, scale=20.0)
    if method == "contrastive":
        return Dataset.from_dict(labelled_pairs(df, seed)), losses.OnlineContrastiveLoss(model, margin=0.5)
    raise ValueError(method)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["simcse", "mnrl", "contrastive"], required=True)
    p.add_argument("--base_model", default="bert-base-multilingual-cased")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_seq_length", type=int, default=64)
    p.add_argument("--max_train", type=int, default=None, help="subsample for quick tests")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    set_seed(a.seed)
    download_ruhsold(a.data_dir)
    df = load_split("train", a.data_dir, a.max_train, a.seed)

    model = SentenceTransformer(a.base_model)  # mean pooling is added automatically for plain BERT
    model.max_seq_length = a.max_seq_length
    train_ds, loss = build(a.method, model, df, a.seed)
    print(f"[{a.method}] {len(train_ds)} training examples from {len(df)} sentences")

    args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(a.out, "checkpoints"),
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        learning_rate=a.lr,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        # NO_DUPLICATES avoids the same sentence appearing twice in a batch,
        # which would create false in-batch negatives for MNRL.
        batch_sampler=BatchSamplers.NO_DUPLICATES if a.method != "contrastive" else BatchSamplers.BATCH_SAMPLER,
        save_strategy="no",
        logging_steps=20,
        report_to="none",
        seed=a.seed,
    )
    SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds, loss=loss).train()
    model.save(a.out)
    print("saved to", a.out)


if __name__ == "__main__":
    main()
