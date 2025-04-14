# Sentiment Analysis with LoRA-Enhanced BERT

This project implements a reproducible pipeline for three-class sentiment analysis (**Negative**, **Neutral**, **Positive**) using a **LoRA-enhanced BERT** model (`bert-base-uncased`). The pipeline includes data preprocessing, model training (documented, pretrained on Kaggle), evaluation, and reporting, all managed with **DVC** for data versioning and **Git** for code versioning.

---

## Project Overview

The goal is to classify text into sentiment categories using a parameter-efficient fine-tuning approach (**LoRA**) on BERT. The pipeline processes raw text data, tokenizes it, evaluates a pretrained LoRA model, and generates detailed metrics and visualizations.

- **Dataset**: Raw text data (`data/raw/train.csv`, `test.csv`) with text and sentiment labels.
- **Model**: LoRA (`r=8`, `lora_alpha=32`) on `bert-base-uncased`, pretrained on Kaggle.
- **Outputs**: Processed data, model weights, metrics (accuracy, precision, recall, F1), confusion matrix, and pipeline report.
- **Tools**: Python, PyTorch, Transformers, PEFT, DVC, Git.

---

## Pipeline Stages

### Preprocessing (`src/preprocess.py`)

- Cleans text (null handling, stopwords, normalization).
- Maps sentiments: `Negative=0`, `Neutral=1`, `Positive=2`.
- Tokenizes with BERT (`max_length=128`).
- **Outputs**:
  - `data/processed/train.csv`, `val.csv`, `test.csv`
  - `train_tokenized.pt`, `val_tokenized.pt`, `test_tokenized.pt`

---

### Training (Documented) (`src/train_lora.py`)

- LoRA fine-tuning code (not run locally, pretrained model used).
- **Settings**: `lr=1e-5`, `batch_size=64 (train)`, `256 (test)`, `5 epochs`, early stopping.
- **Pretrained model**: `models/bert_lora/` (from Kaggle).

---

### Evaluation (`src/evaluate.py`)

- Runs inference on `test_tokenized.pt` using LoRA model.
- Computes accuracy, precision, recall, F1-score.
- **Outputs**:
  - `metrics/test_metrics.json`
  - `plots/test_confusion_matrix.png`

---

### Reporting (`src/report.py`)

- Summarizes pipeline and results.
- **Output**: `reports/pipeline_summary.md`

---

## Repository Structure

```
sentiment-analysis-dvc-pipeline/
├── data/
│   ├── raw/                   # Raw input data (train.csv, test.csv)
│   └── processed/             # Processed CSVs and tokenized .pt files
├── models/
│   └── bert_lora/             # Pretrained LoRA model files
├── metrics/
│   └── test_metrics.json      # Evaluation metrics
├── plots/
│   └── test_confusion_matrix.png  # Confusion matrix
├── reports/
│   └── pipeline_summary.md    # Pipeline summary
├── src/
│   ├── preprocess.py          # Data preprocessing script
│   ├── train_lora.py          # LoRA training code (documented)
│   ├── evaluate.py            # Model evaluation script
│   └── report.py              # Report generation script
├── params.yaml                # Pipeline parameters
├── dvc.yaml                   # DVC pipeline definition
└── README.md                  # This file
```

---

## Prerequisites

- Python 3.8+
- Dependencies: `torch`, `transformers`, `peft`, `scikit-learn`, `seaborn`, `matplotlib`, `tqdm`, `pandas`, `pyyaml`
- DVC for data versioning
- Git for code versioning
- *Optional*: GPU for faster inference

---

## Setup

**Clone the repository:**

```bash
git clone https://github.com/yourusername/sentiment-analysis-dvc-pipeline.git
cd sentiment-analysis-dvc-pipeline
```

**Install dependencies:**

```bash
pip install torch transformers peft scikit-learn seaborn matplotlib tqdm pandas pyyaml dvc
```

**Initialize DVC:**

```bash
dvc init
```

**Pull data and models:**

```bash
# Configure your DVC remote (if needed)
dvc remote add -d myremote /path/to/your/dvc-remote/sentiment-project

# Pull data
dvc pull
```

---

## Usage

Run the pipeline to evaluate the pretrained LoRA model and generate reports:

```bash
dvc repro
```

This executes:

- `evaluate`: Generates `metrics/test_metrics.json` and `plots/test_confusion_matrix.png`.
- `report`: Creates `reports/pipeline_summary.md`.

**View results:**

- **Metrics**: `metrics/test_metrics.json`
- **Plot**: `plots/test_confusion_matrix.png`
- **Report**: `reports/pipeline_summary.md`

---

## Results

The LoRA model achieves competitive performance on the test set:

- **Accuracy**: See `metrics/test_metrics.json`.
- **Precision, Recall, F1-Score**: Detailed in `metrics/test_metrics.json`.
- **Visualization**: Confusion matrix in `plots/test_confusion_matrix.png`.

**Example output:**

```
Test Classification Report:
Overall Score: 69.72%
              precision    recall  f1-score   support

    Negative       0.65      0.78      0.71      1001
     Neutral       0.66      0.61      0.63      1430
    Positive       0.79      0.74      0.77      1103

    accuracy                           0.70      3534
   macro avg       0.70      0.71      0.70      3534
weighted avg       0.70      0.70      0.70      3534
```

---

## Reproducibility

- **DVC**: Tracks data, models, and outputs (`dvc.yaml`, `params.yaml`).
- **Git**: Versions code and pipeline.
- **Params**: Configurable in `params.yaml` (e.g., `max_length=128`, `batch_size=256`).

To reproduce:

```bash
git clone 
dvc pull
dvc repro
```

---

## Notes

- The LoRA model was pretrained on Kaggle for efficiency. Training code is in `src/train_lora.py` for reference.
- Raw data (`data/raw/`) is not public; replace with your own CSV files (columns: `text`, `sentiment`).
- Extend the pipeline by adding inference scripts or evaluating other models (e.g., fine-tuned BERT).

---

## Contributing

Feel free to open issues or submit pull requests for improvements!

---

## License

MIT License

---

## Acknowledgments

- Built with **Transformers** and **PEFT**.
- Managed with **DVC** and **Git**.
- Inspired by sentiment analysis challenges and efficient fine-tuning techniques.
```

