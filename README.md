# Hate Speech Detection in Text (Roman Urdu)

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
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

- ✅ Handles Roman Urdu's unique challenges: orthographic variation, code-switching, informal syntax
- ✅ Binary hate / non-hate classification, with a reference catalogue of offensive Roman Urdu terms
- ✅ Explainable AI with LIME and SHAP (the two methods agree on the top hate-indicative tokens)
- ✅ Significantly outperforms 11 classical, neural, and transformer baselines on RUHSOLD

## Performance

Test-set results for the proposed **BERT+CNN-Gram** model (macro-averaged):

| Metric | Score |
|--------|-------|
| Accuracy | 0.91 |
| Precision | 0.91 |
| Recall | 0.91 |
| F1-Score | 0.91 |
| ROC-AUC | 0.96 |
| MCC | 0.82 |

Compared against 11 baselines, the proposed model is the strongest performer. The best baseline (the SVM+RF+AB TF-IDF ensemble) reaches macro-F1 0.89; a **McNemar test** confirms the improvement is statistically significant (χ² = 10.14, **p = 0.0015**).

## Dataset

**RUHSOLD** (Roman Urdu Hate Speech and Offensive Language Detection)
- Total Posts: 10,013
- Training: 7,209 posts (3,358 Hate / 3,851 Non-Hate)
- Validation: 801 posts (373 Hate / 428 Non-Hate)
- Test: 2,003 posts (933 Hate / 1,070 Non-Hate)
- Class Distribution (whole set): 46.6% Hate, 53.4% Non-Hate
- Inter-annotator Agreement (reported by Rizwan et al., 2020): Cohen's κ = 0.78

## Repository Structure

```
├── code/
│   ├── preprocessing/          # Data preprocessing scripts
│   ├── models/                 # BERT+CNN-Gram hybrid model implementation
│   ├── training/               # Training scripts
│   ├── evaluation/             # Evaluation and metrics (Acc, F1, ROC-AUC, MCC, McNemar)
│   └── explainability/         # LIME and SHAP analysis
├── data/                       # Dataset (RUHSOLD)
├── results/
│   ├── visualizations/         # Training curves, confusion matrices
│   ├── lime_outputs/           # LIME explanations
│   └── shap_outputs/           # SHAP visualizations
├── docs/
│   ├── offensive_words_reference.pdf  # Reconciled reference catalogue (IDs + real LIME/SHAP scores)
│   └── thesis.pdf              # Full thesis document
└── requirements.txt            # Python dependencies
```

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

### Training the Model
```bash
python code/training/train_bert_cnn_gram.py --epochs 20 --batch_size 16
```

### Evaluation
```bash
python code/evaluation/evaluate_model.py --model_path models/bert_cnn_gram_final.pth
```

### LIME Explainability
```bash
python code/explainability/lime_analysis.py --input "your text here"
```

## Requirements

- Python 3.10+
- PyTorch 2.x
- Transformers (Hugging Face)
- scikit-learn
- LIME
- SHAP
- statsmodels (for the McNemar significance test)
- gensim, xgboost (for the baseline models)
- pandas, numpy, matplotlib

See `requirements.txt` for the complete list.

## Offensive Words Reference

The repository includes a reconciled reference document (`docs/offensive_words_reference.pdf`) containing:
- The original 27 identified terms (ID001–ID027), plus the additional high-impact tokens (ID028+) the model actually relied on
- Detailed context and usage information
- **Real** LIME (summed weight) and SHAP (mean |SHAP|) scores from the trained model
- Categorization by type, with identity/named-entity terms flagged as potential bias signals

**⚠️ Content Warning:** This document contains explicit hate speech terms extracted from the RUHSOLD dataset for academic research purposes only.

## Model Architecture

### BERT Encoder
- Model: `bert-base-multilingual-cased`
- 12 transformer layers (layer freezing is configurable)
- Hidden / embedding dimension: 768

### CNN-Gram Layers
- 4 parallel CNN layers
- Kernel sizes: 1, 2, 3, 4
- Filters: 128 per kernel
- ReLU activation + Batch Normalization + max-pooling

### Classification Head
- Dropout: 0.3–0.5
- Fully connected layer: 512 → 2
- Output: Binary classification (Hate / Non-Hate)

### Training
- Optimizer: AdamW with weight decay; linear warmup/decay schedule
- Gradient clipping (max-norm 1.0), label smoothing, mixed-precision
- Model selection by best validation **macro-F1**

## Results Highlights

### By Category
- **Explicit Slurs:** high precision
- **Context-Dependent Terms (e.g., pagal dost):** frequent false positives (~8–10% of errors)
- **Religious Domain:** precision ≈ 0.89
- **Political Domain:** precision ≈ 0.92
- **Evasion Variants:** captured by the CNN-Gram layers

### Explainability
- LIME identifies the strongest token-level contributors per prediction (e.g., bc, bharwe, laanat, randi, bhenchod)
- SHAP provides global feature importance; LIME and SHAP agree on the top hate-indicative tokens

### Bias & Limitations
- Both LIME and SHAP show the model assigns high hate-importance to identity/community terms that are not themselves abusive (e.g., *ahmedi*, *hijra*, *india*) and to named individuals (*qandel*). This risks false positives on neutral discussion of these groups and is documented as a limitation; fairness-aware training and group-level auditing are recommended for deployment.

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
- **Orthographic Variation:** ~40–50% spelling variation
- **Code-Switching:** ~40–50% English-Urdu mixing
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
- RUHSOLD Dataset: Rizwan et al. (2020)

## Contact

For questions or collaboration:
- GitHub: [@quratulain77-saeed](https://github.com/quratulain77-saeed)

---

**Disclaimer:** This research is intended for academic purposes only. The offensive language documented is used solely for advancing hate speech detection technologies to create safer online environments.
