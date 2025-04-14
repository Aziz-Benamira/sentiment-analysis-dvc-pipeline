import json
import yaml
import os

def generate_report():
    # Load metrics
    with open("metrics/test_metrics.json", "r") as f:
        metrics = json.load(f)
    # Load params
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    # Create report
    report = f"""
# Sentiment Analysis Pipeline Summary

## Overview
This pipeline performs three-class sentiment analysis (Negative, Neutral, Positive) using a LoRA-enhanced BERT model (`bert-base-uncased`). The pipeline includes data preprocessing, model training (documented, not run locally), evaluation, and reporting.

## Preprocessing
- **Input**: `data/raw/train.csv`, `test.csv` (text, sentiment).
- **Steps**: Null handling, custom stopwords, text cleaning, sentiment mapping (Negative=0, Neutral=1, Positive=2), BERT tokenization.
- **Outputs**: `data/processed/train.csv`, `val.csv`, `test.csv`, `train_tokenized.pt`, `val_tokenized.pt`, `test_tokenized.pt`.
- **Params**:
  - Max Length: {params['preprocess']['max_length']}
  - Validation Size: {params['preprocess']['val_size']}
  - Missing Threshold: {params['preprocess']['missing_threshold']}

## Training (Kaggle)
- **Model**: LoRA on `bert-base-uncased` (r={params['lora']['rank']}, alpha={params['lora']['lora_alpha']}).
- **Settings**:
  - Learning Rate: {params['lora']['lr']}
  - Epochs: {params['lora']['num_epochs']}
  - Batch Size: {params['lora']['batch_size_train']} (train), {params['lora']['batch_size_test']} (test)
  - Early Stopping Patience: {params['lora']['patience']}
- **Output**: `models/bert_lora/` (pretrained from Kaggle).

## Evaluation
- **Input**: `data/processed/test_tokenized.pt`.
- **Model**: LoRA (`models/bert_lora/`).
- **Metrics**:
  - Accuracy: {metrics['test_accuracy']:.4f}
  - Precision: {metrics['test_precision']:.4f}
  - Recall: {metrics['test_recall']:.4f}
  - F1-Score: {metrics['test_f1_score']:.4f}
- **Plots**: Confusion matrix (`plots/test_confusion_matrix.png`).
- **Batch Size**: {params['eval']['batch_size']}

## Files
- **Code**: `src/preprocess.py`, `src/train_lora.py`, `src/evaluate.py`, `src/report.py`.
- **Data**: `data/processed/*`.
- **Models**: `models/bert_lora/`.
- **Outputs**: `metrics/test_metrics.json`, `plots/test_confusion_matrix.png`, `reports/pipeline_summary.md`.

## Reproducibility
- Versioned with DVC (`dvc.yaml`) and Git.
- Remote: `~/dvc-remote/sentiment-project`.

See `plots/test_confusion_matrix.png` for visualization.
"""
    # Save report
    os.makedirs("reports", exist_ok=True)
    with open("reports/pipeline_summary.md", "w") as f:
        f.write(report)

if __name__ == "__main__":
    generate_report()