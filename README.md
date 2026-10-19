# Hate Speech Detection in Roman Urdu: BERT+CNN-Gram with Explainability

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23441478.svg)](https://doi.org/10.5281/zenodo.23441478)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0007--2133--4869-a6ce39.svg)](https://orcid.org/0009-0007-2133-4869)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code for the MPhil thesis *Hate Speech Detection in Text* (Department of Information Technology, University of the Punjab, Lahore) and the follow-up paper:

> Saeed, Q. A. (2026). *Stress-Testing Explainable Hate Speech Detection for Roman Urdu: Baselines, Faithfulness and Religious–Political Bias.* Zenodo. https://doi.org/10.5281/zenodo.23441478

## Overview

This repository **re-implements and extends** the BERT+CNN-gram classifier proposed by Rizwan et al. (2020) together with the RUHSOLD dataset, and asks three questions: how does it compare with baselines re-trained under identical conditions, are its explanations faithful, and where does it fail?

- **Model:** `bert-base-multilingual-cased` (all layers fine-tuned) + four parallel CNN-gram filters (kernel sizes 1–4) + residual mean-pooled projection
- **Explanations:** LIME (qualitative), word-level SHAP, Integrated Gradients (Captum), attention maps (illustration only)
- **Faithfulness:** ERASER-style comprehensiveness and sufficiency for an occlusion-based word ranking
- **Extended evaluation:** four re-trained baselines, McNemar tests, paired bootstrap, five-fold cross-validation, threshold sweep, error analysis by fine-grained category and by religious/political content, embedding-space analysis

The project does **not** propose a new architecture; BERT+CNN-gram is the model of Rizwan et al. (2020).

## Results

All numbers below are produced by the notebook in this repository (single run, seed 42, NVIDIA T4). GPU training is not fully deterministic, so a re-run gives slightly different values.

### Official RUHSOLD test set (2,003 posts)

| Metric | Score |
|---|---|
| Accuracy | 0.903 |
| Macro-F1 | 0.902 |
| ROC-AUC | 0.962 |
| PR-AUC | 0.963 |
| MCC | 0.805 |
| Cohen's κ (model vs. gold labels) | 0.804 |
| FNR / FPR (hate = positive class) | 0.100 / 0.095 |

**Five-fold cross-validation** (train + validation pooled, 8,010 posts): macro-F1 **0.872 ± 0.011**, ROC-AUC 0.936 ± 0.003, MCC 0.748 ± 0.023. The single-run test score is at the upper end of what this configuration typically achieves; the cross-validated estimate is the more conservative figure.

### Baselines re-trained under identical conditions

| Model | Macro-F1 | ROC-AUC | MCC | McNemar p vs. ours |
|---|---|---|---|---|
| BERT+CNN-Gram (this work) | 0.902 | **0.962** | 0.805 | – |
| XLM-R + CNN-Gram | **0.907** | 0.951 | **0.815** | 0.505 |
| FastText + CNN-Gram | 0.901 | 0.957 | 0.801 | 0.883 |
| TF-IDF + logistic regression | 0.895 | 0.959 | 0.789 | 0.290 |
| mBERT fine-tuned ([CLS] + linear) | 0.890 | 0.952 | 0.781 | 0.0498 |

The model is **on par with strong baselines, not clearly ahead of them**. Only the comparison with plain mBERT fine-tuning approaches significance, and its bootstrap interval touches zero. Transformer baselines used a shorter early-stopping schedule (patience 3) than the main model.

For reference, Rizwan et al. (2020, Table 4) report macro-F1 0.90 for both their BERT+CNN-gram model and their SVM+RF+AdaBoost ensemble; these published results were not re-computed here.

### Explanations and faithfulness

- **SHAP vs. Integrated Gradients:** 16 of the top-20 words agree (199 test posts; 100 per class sampled, one post with fewer than two words excluded). Both rankings are dominated by explicit slurs and profanity.
- **ERASER-style test (occlusion ranking, 197 posts):** comprehensiveness **0.420** vs. **0.096** for random word removal; sufficiency −0.028. The test validates the occlusion ranking; applying it directly to the SHAP and IG rankings is future work.
- **LIME** (50 posts, 50 perturbation samples each) is used qualitatively only.

### Errors and bias

| Domain (keyword-defined) | Posts | Precision (hate) | FNR | FPR | Macro-F1 |
|---|---|---|---|---|---|
| Religious terms | 118 | 0.667 | 0.190 | 0.224 | 0.778 |
| Political terms | 122 | 0.890 | 0.131 | 0.237 | 0.812 |
| Other posts | 1,763 | 0.906 | 0.092 | 0.080 | 0.914 |

Posts mentioning religious or political terms are wrongly flagged almost three times as often as other posts. The subsets are small, so these figures are indicative. The keyword lists are defined in notebook cell E10.

By fine-grained RUHSOLD category (1,842 test posts matched): profane/untargeted 95.8% detected, sexism 92.9%, religious hate 88.4%, other abusive/offensive 87.4%; 9.2% of normal posts flagged.

**Embedding space:** fine-tuning raises the silhouette score of the two classes from 0.016 (pre-trained mBERT) to 0.52.

**Decision threshold:** lowering the threshold to 0.25 reduces FNR to 0.069 (FPR 0.143); raising it to 0.80 reduces FPR to 0.041 (FNR 0.143).

## Dataset

**RUHSOLD** (Rizwan, Shakeel & Karim, 2020), coarse-grained task, downloaded automatically from the [official repository](https://github.com/haroonshakeel/roman_urdu_hate_speech).

| Split | Posts | Hate | Non-Hate |
|---|---|---|---|
| Train | 7,209 | 3,358 | 3,851 |
| Validation | 801 | 373 | 428 |
| Test | 2,003 | 933 | 1,070 |

- Hate speech is 46.6% of each split. Label convention in the code: **0 = Hate (Abusive/Offensive), 1 = Non-Hate (Normal)**.
- Each post was labelled by three annotators with majority vote. The original paper reports **no** inter-annotator agreement statistic.
- Text is used as provided (no extra cleaning). Training data are augmented: 30% of training posts are duplicated with a random English filler word and a random word swap, giving 9,372 training examples.

## Repository contents

| File / folder | Description |
|---|---|
| `BERT_CNN_Gram_Thesis_code_extended.ipynb` | Main Colab notebook (58 cells): data loading, EDA, training, evaluation, LIME, SHAP, Integrated Gradients, attention, Gradio demo, and the extended evaluation (cells E1–E12) |
| `contrastive/` | Follow-up experiment: contrastive representation learning on RUHSOLD (see its own README) |
| `Hate_Speech_Words_with_IDs.pdf` | Reference list resolving the term identifiers (ID001–ID027) used in the thesis. **Contains explicit offensive language.** |

## How to run

1. Open the main notebook in Google Colab and select a **T4 GPU**.
2. Run the cells from top to bottom. The first cell installs the dependencies (Transformers, LIME, SHAP, Captum, etc.) and the dataset is downloaded automatically.
3. The trained model is saved as `bert_cnn_gram_lime_model.pth`; the SHAP, Integrated Gradients and attention cells reuse it, so run them in the same session.
4. The extended evaluation (cells E1–E12) caches results in Google Drive (`EXT_DIR`) so an interrupted run can resume. It re-trains the baselines and the five cross-validation folds and takes several GPU hours.

## Model and training details

| Component | Setting |
|---|---|
| Encoder | `bert-base-multilingual-cased`, all 12 layers fine-tuned, max length 128 |
| CNN-gram | LayerNorm → 4 × Conv1d (kernels 1–4, 128 filters) + BatchNorm + ReLU + global max-pool → 512-d |
| Residual | attention-masked mean of token states, projected to 512-d and added |
| Head | dropout 0.3 → 512 → 128 (ReLU, BatchNorm) → 2 |
| Loss | cross-entropy with label smoothing 0.1 |
| Optimiser | Adam with layer-wise LR decay: 2×10⁻⁵ top layer, ×0.95 per lower layer; 2×10⁻⁴ for new layers |
| Other | batch 16, mixed precision, gradient clipping 1.0, early stopping on validation loss (patience 10, max 50 epochs), seed 42 |
| Selected checkpoint | epoch 3 (validation loss 0.4114); training stopped after epoch 13 |

## Limitations

- Single training run for the main model; FNR varies noticeably across folds (0.153 ± 0.047).
- Baselines used a shorter early-stopping schedule; no hyperparameter search for any model.
- The faithfulness test covers the occlusion ranking, not SHAP or IG directly; LIME used few perturbation samples.
- Religious and political domains are keyword-defined and small.
- One dataset only; implicit hate, sarcasm and coded language remain difficult.

## Citation

```bibtex
@misc{saeed2026stresstesting,
  author    = {Saeed, Quratul Ain},
  title     = {Stress-Testing Explainable Hate Speech Detection for Roman Urdu: Baselines, Faithfulness and Religious--Political Bias},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23441478}
}

@mastersthesis{saeed2024hate,
  title  = {Hate Speech Detection in Text},
  author = {Saeed, Qurat Ul Ain},
  year   = {2024},
  school = {University of the Punjab, Department of Information Technology}
}
```

Please also cite the dataset and original model:

```bibtex
@inproceedings{rizwan2020hate,
  title     = {Hate-Speech and Offensive Language Detection in Roman {U}rdu},
  author    = {Rizwan, Hammad and Shakeel, Muhammad Haroon and Karim, Asim},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages     = {2512--2522},
  year      = {2020}
}
```

## License

MIT. See [LICENSE](LICENSE). The RUHSOLD dataset is subject to its authors' terms.

## Acknowledgements

Supervisor: Dr. Mian Muhammad Mubasher, Department of Information Technology, University of the Punjab. Dataset: Rizwan, Shakeel & Karim (2020).

## Contact

Quratul Ain Saeed · [ORCID 0009-0007-2133-4869](https://orcid.org/0009-0007-2133-4869) · [GitHub @quratulain77-saeed](https://github.com/quratulain77-saeed)

---

**Disclaimer:** This repository contains offensive language for research purposes only. The model is a research prototype and should not be used for automated moderation without human review, particularly for posts about religion or politics.
