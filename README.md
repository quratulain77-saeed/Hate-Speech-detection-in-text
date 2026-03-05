# Hate Speech Detection in Roman Urdu Using Hybrid BERT+CNN-Gram Model with LIME

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the complete implementation of a hate speech detection system for Roman Urdu, developed as part of an MPhil thesis at the Department of Information Technology, University of the Punjab, Lahore.

The system combines:
- **BERT** (bert-base-multilingual-cased) for contextual understanding
- **CNN-Gram layers** for local n-gram pattern detection
- **LIME** for model interpretability and explainability
- **SHAP** for global feature importance analysis

## Key Features

- ✅ Handles Roman Urdu's unique challenges: orthographic variation, code-switching, informal syntax
- ✅ Detects 27 categories of hate speech including explicit slurs, evasion variants, and context-dependent terms
- ✅ Explainable AI with LIME and SHAP visualizations
- ✅ Achieves state-of-the-art performance on RUHSOLD dataset

## Performance

| Metric | Score |
|--------|-------|
| Accuracy | 0.89 |
| Precision | 0.89 |
| Recall | 0.89 |
| F1-Score | 0.89 |
| ROC-AUC | 0.93 |
| MCC | 0.86 |

## Dataset

**RUHSOLD** (Roman Urdu Hate Speech and Offensive Language Detection)
- Total Posts: 10,013
- Training: 7,209 posts
- Validation: 801 posts
- Test: 2,003 posts
- Class Distribution: 46.6% Hate, 53.4% Non-Hate
- Inter-annotator Agreement: Cohen's κ = 0.78

## Repository Structure

```
├── code/
│   ├── preprocessing/          # Data preprocessing scripts
│   ├── models/                 # BERT, CNN-Gram, hybrid model implementations
│   ├── training/               # Training scripts
│   ├── evaluation/             # Evaluation and metrics
│   └── explainability/         # LIME and SHAP analysis
├── data/                       # Dataset (RUHSOLD)
├── results/
│   ├── visualizations/         # Training curves, confusion matrices
│   ├── lime_outputs/           # LIME explanations
│   └── shap_outputs/           # SHAP visualizations
├── docs/
│   ├── offensive_words_reference.pdf  # Complete list with IDs
│   └── thesis.pdf              # Full thesis document
└── requirements.txt            # Python dependencies
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/roman-urdu-hate-speech-detection.git
cd roman-urdu-hate-speech-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model
```bash
python code/training/train_bert_cnn_gram.py --epochs 50 --batch_size 16
```

### Evaluation
```bash
python code/evaluation/evaluate_model.py --model_path models/bert_cnn_gram_lime_model.pth
```

### LIME Explainability
```bash
python code/explainability/lime_analysis.py --input "your text here"
```

## Requirements

- Python 3.10+
- PyTorch 1.9+
- Transformers (Hugging Face)
- scikit-learn
- LIME
- SHAP
- pandas
- numpy
- matplotlib

See `requirements.txt` for complete list.

## Offensive Words Reference

The repository includes a comprehensive reference document (`docs/offensive_words_reference.pdf`) containing:
- 27 identified hate speech terms with unique IDs (ID001-ID027)
- Detailed context and usage information
- LIME/SHAP scores
- Categorization by type

**⚠️ Content Warning:** This document contains explicit hate speech terms extracted from the RUHSOLD dataset for academic research purposes only.

## Model Architecture

### BERT Encoder
- Model: `bert-base-multilingual-cased`
- Layers: 12 transformer layers (first 6 frozen)
- Embedding dimension: 768

### CNN-Gram Layers
- 4 parallel CNN layers
- Kernel sizes: 1, 2, 3, 4
- Filters: 128 per kernel
- Activation: ReLU + Batch Normalization

### Classification Head
- Dropout: 0.5
- Fully connected layers: 512 → 128 → 2
- Output: Binary classification (Hate/Non-Hate)

## Results Highlights

### By Category
- **Explicit Slurs:** Precision ≥ 0.90
- **Context-Dependent Terms:** 8-10% false positive rate
- **Religious Terms:** Domain precision 0.89
- **Political Terms:** Domain precision 0.92
- **Evasion Variants:** Successfully detected by CNN-Gram layers

### Explainability
- LIME identifies key hate-indicative terms with high negative scores
- SHAP provides global feature importance across entire dataset
- Attention visualization shows token-level focus during classification

## Citation

If you use this code or research in your work, please cite:

```bibtex
@mastersthesis{saeed2025hate,
  title={Hate Speech Detection in Text},
  author={Saeed, Qurat ul Ain},
  year={2025},
  school={University of the Punjab},
  department={Department of Information Technology}
}
```

## Research Context

This work addresses the unique challenges of hate speech detection in Roman Urdu:
- **Orthographic Variation:** 40-50% spelling variation
- **Code-Switching:** 40-50% English-Urdu mixing
- **Cultural Context:** Religious, political, and gender-specific sensitivities
- **Low Resource Language:** Limited annotated datasets

## Future Work

- Multimodal detection (text + images + emojis)
- Cross-lingual transfer to Hinglish and Roman Punjabi
- Advanced attention mechanisms for sarcasm detection
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
- Email: quratulainsaeedofficial@gmail.com

---

**Disclaimer:** This research is intended for academic purposes only. The offensive language documented is used solely for advancing hate speech detection technologies to create safer online environments.
