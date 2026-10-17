# Contrastive Representation Learning for Roman Urdu Hate Speech

An extension of my MPhil thesis (*Hate Speech Detection in Roman Urdu Text*, BERT + CNN-Gram on RUHSOLD).
The thesis fine-tuned `bert-base-multilingual-cased` with cross-entropy. This module asks a different question:
**can contrastive objectives shape the embedding space so that abusive and normal Roman Urdu text separate
more cleanly, and does that reduce missed hate speech (false negatives)?**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quratulain77-saeed/Hate-Speech-detection-in-text/blob/main/contrastive/RUHSOLD_Contrastive_Learning.ipynb)

## Methods

All methods start from `bert-base-multilingual-cased` with mean pooling (via `sentence-transformers`).

| Name | Objective | Training signal |
|---|---|---|
| `baseline` | none (pre-trained mBERT embeddings) | — |
| `simcse` | Unsupervised SimCSE (Gao et al., 2021): MultipleNegativesRankingLoss on (x, x); dropout is the only augmentation | unlabelled text |
| `contrastive` | Margin-based contrastive loss (Hadsell et al., 2006), `OnlineContrastiveLoss`, margin 0.5 | same-class / cross-class pairs |
| `mnrl` | Supervised MultipleNegativesRankingLoss (InfoNCE-style) with an explicit cross-class **hard negative** | (anchor, same-class positive, other-class negative) |

Batches use a no-duplicates sampler so the same sentence never appears twice in a batch as a false in-batch negative.

## Evaluation

Embeddings are frozen and evaluated on the RUHSOLD Task 1 test set (hate = positive class):

- **Linear probe** (logistic regression): macro-F1, MCC, ROC-AUC, FNR (missed hate speech), FPR (normal text flagged)
- **Cost-sensitive probe**: hate class weighted ×2 to penalise false negatives. This is the classifier-side analogue of a
  security-aware penalty, where missing a harmful item costs more than a false alarm.
- **kNN accuracy** (k = 10, cosine)
- **Silhouette** (cosine) between the two classes
- **Alignment and uniformity** (Wang & Isola, 2020): lower is better for both

## Results

_Run the Colab notebook and paste `results/results_table.md` here; add `results/tsne.png` below._

| Model | Macro-F1 | MCC | ROC-AUC | FNR | FPR | FNR (cost-sens.) | kNN acc | Silhouette | Alignment | Uniformity |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | | | | | | | | | | |
| simcse | | | | | | | | | | |
| contrastive | | | | | | | | | | |
| mnrl | | | | | | | | | | |

![t-SNE of test embeddings](results/tsne.png)

## Reproduce

```bash
pip install -r requirements.txt
python src/evaluate.py --model bert-base-multilingual-cased --name baseline
python src/train.py --method simcse      --epochs 1 --batch_size 64 --out models/simcse
python src/train.py --method contrastive --epochs 2 --batch_size 32 --out models/contrastive
python src/train.py --method mnrl        --epochs 2 --batch_size 32 --out models/mnrl
for m in simcse contrastive mnrl; do python src/evaluate.py --model models/$m --name $m; done
python src/report.py
```

RUHSOLD is downloaded automatically from the [official repository](https://github.com/haroonshakeel/roman_urdu_hate_speech)
(Rizwan, Shakeel & Karim, EMNLP 2020). Seed = 42 throughout.

## Files

```
contrastive/
├── RUHSOLD_Contrastive_Learning.ipynb   # end-to-end Colab notebook
├── requirements.txt
└── src/
    ├── data.py       # loading + SimCSE pairs, supervised triplets, labelled pairs
    ├── train.py      # contrastive fine-tuning (simcse | contrastive | mnrl)
    ├── evaluate.py   # probes, kNN, silhouette, alignment/uniformity
    └── report.py     # results table + t-SNE figure
```

## References

- Gao, Yao & Chen (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. EMNLP.
- Hadsell, Chopra & LeCun (2006). Dimensionality Reduction by Learning an Invariant Mapping. CVPR.
- Henderson et al. (2017). Efficient Natural Language Response Suggestion for Smart Reply. (MNRL)
- Wang & Isola (2020). Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere. ICML.
- Rizwan, Shakeel & Karim (2020). Hate-Speech and Offensive Language Detection in Roman Urdu. EMNLP.
