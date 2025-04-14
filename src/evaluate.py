import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification
from peft import PeftModel
import yaml
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import json
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def evaluate():
    params = load_params()
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
    # Evaluate
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
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    metrics = {
        "accuracy": accuracy,
        "f1_score": f1
    }
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    # Save metrics
    with open("metrics/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Negative", "Neutral", "Positive"], 
                yticklabels=["Negative", "Neutral", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("plots/confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    evaluate()