import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification
from peft import PeftModel
import yaml
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def evaluate():
    params = load_params()
    # Sentiment mapping
    sentiment_reverse_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    # Load test data
    test_data = torch.load("data/processed/test_tokenized.pt")
    test_dataset = TensorDataset(
        test_data["input_ids"],
        test_data["attention_mask"],
        test_data["labels"]
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=params["eval"]["batch_size"],
        shuffle=False
    )
    # Load LoRA model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3
    )
    model = PeftModel.from_pretrained(base_model, "models/bert_lora")
    model.to(device)
    model.eval()
    # Get predictions
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # Compute metrics
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, 
        target_names=list(sentiment_reverse_mapping.values()), 
        output_dict=True
    )
    accuracy = accuracy_score(all_labels, all_preds)
    metrics = {
        "model_name": "bert_lora",
        "test_accuracy": report["accuracy"],
        "test_precision": report["weighted avg"]["precision"],
        "test_recall": report["weighted avg"]["recall"],
        "test_f1_score": report["weighted avg"]["f1-score"]
    }
    print("Test Classification Report:")
    print(f"Overall Score: {(accuracy * 100):.2f}%")
    print(classification_report(all_labels, all_preds, 
                               target_names=list(sentiment_reverse_mapping.values())))
    # Save metrics
    with open("metrics/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=list(sentiment_reverse_mapping.values()), 
                yticklabels=list(sentiment_reverse_mapping.values()))
    # Highlight diagonal
    for i in range(cm.shape[0]):
        plt.gca().add_patch(patches.Rectangle((i, i), 1, 1, fill=False, edgecolor="Coral", linewidth=2))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Test Confusion Matrix")
    plt.savefig("plots/test_confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    evaluate()