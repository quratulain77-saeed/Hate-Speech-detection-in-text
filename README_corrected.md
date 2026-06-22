# Hate Speech Detection in Text (Roman Urdu)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the complete implementation of a hate speech detection system for Roman Urdu, developed as part of an MPhil thesis at the Department of Information Technology, University of the Punjab, Lahore.

The system combines:
- **BERT** (bert-base-multilingual-cased) for contextual understanding
- **CNN-Gram layers** for local n-gram pattern detection
- **LIME** for local model interpretability
- **SHAP** for global feature importance analysis

## Key Features

- Handles Roman Urdu's unique challenges: orthographic variation, code-switching, informal syntax
- Binary hate / non-hate classification, with a reference catalogue of offensive Roman Urdu terms
- Explainable AI with LIME and SHAP (the two methods agree on the top hate-indicative tokens)
- Benchmarked against the classical, neural, and transformer baselines reported by Rizwan et al. (2020)

## Performance

Test-set results for the proposed **BERT+CNN-Gram + LIME** model (macro-averaged, 2,003-post test split):

| Metric | Score |
|--------|-------|
| Accuracy | 0.90 |
| Precision (macro) | 0.90 |
| Recall (macro) | 0.90 |
| F1-Score (macro) | 0.90 |
| ROC-AUC | 0.96 |
| MCC | 0.80 |

> Run-to-run variation is ~0.88-0.90 accuracy / F1 and MCC ~0.76-0.80; the table reports a representative run. The fine-tuned BERT baseline scores ~0.89 (ROC-AUC 0.95, MCC 0.78).

**Comparison with baselines.** The proposed model is **competitive with the strongest published baselines** on RUHSOLD. The best baseline — the SVM+RF+AB TF-IDF ensemble (Rizwan et al., 2020, Table 4) — reaches macro-F1 **0.90**. Across 5-fold cross-validation the proposed model achieves macro-F1 **0.889 +/- 0.008**, and a paired significance test shows the difference from the 0.90 baseline is **not statistically significant (p ~ 0.06)** — i.e., the model **matches** the state of the art while adding explainability.

## Dataset

**RUHSOLD** (Roman Urdu Hate Speech and Offensive Language Detection)
- Total Posts: 10,013
- Training: 7,209 posts (3,358 Hate / 3,851 Non-Hate)
- Validation: 801 posts (373 Hate / 428 Non-Hate)
- Test: 2,003 posts (933 Hate / 1,070 Non-Hate)
- Class Distribution (whole set): 46.6% Hate, 53.4% Non-Hate
- Label convention: **0 = Hate, 1 = Non-Hate**
- Inter-annotator Agreement (reported by Rizwan et al., 2020): Cohen's kappa = 0.78

## Repository Structure

```
├── notebooks/
│   ├── Fine-Tuned BERT Model.ipynb           # Fine-tuned mBERT (thesis 4.2)
│   ├── BERT+CNN-Gram Model.ipynb             # Hybrid BERT + CNN-Gram (4.3)
│   ├── BERT+CNN-Gram with LIME.ipynb         # Proposed model + LIME + 5-fold CV (4.4)
│   └── SHAP Explainability.ipynb             # SHAP analysis (4.7)
├── results/
│   ├── visualizations/         # Training curves, confusion matrices
│   ├── lime_outputs/           # LIME explanations (HTML / PNG)
│   └── shap_outputs/           # SHAP visualizations
├── docs/
│   ├── offensive_words_reference.pdf  # Reference catalogue (IDs + LIME/SHAP scores)
│   └── thesis.pdf              # Full thesis document
└── requirements.txt            # Python dependencies
```

The dataset is downloaded automatically inside the notebooks from the public RUHSOLD repository, so no manual data download is required.

## Installation

```bash
# Clone the repository
git clone https://github.com/quratulain77-saeed/hate-speech-detection-text.git
cd hate-speech-detection-text

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

The implementation is provided as Google Colab-ready Jupyter notebooks (run on a T4 GPU). Open a notebook and run the cells top to bottom:

1. **`Fine-Tuned BERT Model.ipynb`** - trains and evaluates the fine-tuned mBERT baseline.
2. **`BERT+CNN-Gram Model.ipynb`** - trains the hybrid BERT + CNN-Gram model.
3. **`BERT+CNN-Gram with LIME.ipynb`** - the proposed model; also runs 5-fold cross-validation, the significance test, and generates LIME explanations. Saves `bert_cnn_gram_lime_model.pth`.
4. **`SHAP Explainability.ipynb`** - run **after** notebook 3 (same session) so the saved checkpoint exists; produces the SHAP analysis.

Each notebook prints the full metric set (Accuracy, macro-F1, ROC-AUC, PR-AUC, MCC, Cohen's kappa) on the test set.

## Requirements

- Python 3.10+
- PyTorch 2.x
- Transformers (Hugging Face)
- scikit-learn
- scipy (for the significance test)
- LIME
- SHAP
- pandas, numpy, matplotlib, seaborn

See `requirements.txt` for the complete list.

## Offensive Words Reference

The repository includes a reference document (`docs/offensive_words_reference.pdf`) containing:
- The 27 identified terms (ID001-ID027)
- Context and usage information for each term
- LIME / SHAP scores recorded for the high-impact tokens
- Categorization by type, with identity / named-entity terms flagged as potential bias signals

**Content Warning:** This document contains explicit hate speech terms extracted from the RUHSOLD dataset for academic research purposes only.

## Model Architecture

### BERT Encoder
- Model: `bert-base-multilingual-cased`
- 12 transformer layers (first 6 frozen; configurable)
- Hidden / embedding dimension: 768

### CNN-Gram Layers
- 4 parallel CNN layers
- Kernel sizes: 1, 2, 3, 4
- Filters: 128 per kernel
- ReLU activation + Batch Normalization + global max-pooling

### Classification Head
- Embedding normalization + residual connection from BERT
- Dropout: 0.3-0.5
- Fully connected layers: **512 -> 128 -> 2**
- Output: Binary classification (0 = Hate, 1 = Non-Hate)

### Training
- Optimizer: AdamW / Adam with layer-wise learning-rate decay
- Gradient clipping (max-norm 1.0), label smoothing (epsilon = 0.1), mixed-precision
- Data augmentation for spelling variation and code-switching
- Model selection by best validation **macro-F1**; early stopping

## Results Highlights

### By Category
- **Explicit Slurs:** high precision
- **Context-Dependent Terms (e.g., the phrase "pagal dost"):** frequent false positives (~8-10% of errors)
- **Religious Domain:** precision ~ 0.89
- **Political Domain:** precision ~ 0.92
- **Evasion Variants:** captured by the CNN-Gram layers

### Robustness
- 5-fold cross-validation: macro-F1 = **0.889 +/- 0.008** (std < 0.01 -> stable)
- Significance test vs. the 0.90 baseline: **p ~ 0.06** (no significant difference)

### Explainability
- LIME identifies the strongest token-level contributors per prediction.
- SHAP provides global feature importance; LIME and SHAP broadly agree on the top hate-indicative tokens.

### Bias & Limitations
- LIME / SHAP indicate the model can assign high hate-importance to identity / community terms that are not themselves abusive, and to named individuals. This risks false positives on neutral discussion of these groups and is documented as a limitation; fairness-aware training and group-level auditing are recommended before deployment.
- Implicit hate, sarcasm and coded language remain challenging.

## Citation

If you use this code or research in your work, please cite:

```bibtex
@mastersthesis{saeed2024hate,
  title={Hate Speech Detection in Text},
  author={Saeed, Qurat ul Ain},
  year={2024},
  school={University of the Punjab},
  department={Department of Information Technology}
}
```

## Research Context

This work addresses the unique challenges of hate speech detection in Roman Urdu:
- **Orthographic Variation:** ~40-50% spelling variation
- **Code-Switching:** ~40-50% English-Urdu mixing
- **Cultural Context:** religious, political, and gender-specific sensitivities
- **Low-Resource Language:** limited annotated datasets

## Future Work

- Multimodal detection (text + images + emojis)
- Cross-lingual transfer to Hinglish and Roman Punjabi
- Quantitative, faithfulness-based evaluation of explanations
- Fairness-aware / debiased training to reduce identity-term over-reliance
- Real-time deployment for social media platforms

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Department of Information Technology, University of the Punjab
- Supervisor: Dr. Mian Muhammad Mubasher
- RUHSOLD Dataset: Rizwan, Shakeel & Karim (2020)

## Contact

For questions or collaboration:
- GitHub: [@quratulain77-saeed](https://github.com/quratulain77-saeed)

---

**Disclaimer:** This research is intended for academic purposes only. The offensive language documented is used solely for advancing hate speech detection technologies to create safer online environments.
