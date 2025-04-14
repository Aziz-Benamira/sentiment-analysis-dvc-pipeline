import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml
import os

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]

def train_lora():
    params = load_params()
    # Load data (from processed CSVs for consistency)
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    train_x = train_df["processed_text"].tolist()
    train_y = train_df["sentiment_class"].tolist()
    val_x = val_df["processed_text"].tolist()
    val_y = val_df["sentiment_class"].tolist()
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Datasets
    train_dataset = TokenizedDataset(
        train_x, train_y, tokenizer, params["lora"]["max_length"]
    )
    val_dataset = TokenizedDataset(
        val_x, val_y, tokenizer, params["lora"]["max_length"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["lora"]["batch_size_train"],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["lora"]["batch_size_test"],
        shuffle=True
    )
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=params["lora"]["rank"],
        lora_alpha=params["lora"]["lora_alpha"],
        lora_dropout=0.1,
        bias="all"
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    # Optimizer and loss
    optimizer = Adam(model.parameters(), lr=params["lora"]["lr"])
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
    # Training loop
    best_val_loss = float("inf")
    epochs_since_improvement = 0
    best_epoch = 0
    checkpoint_folder = "models/bert_lora_checkpoints"
    os.makedirs(checkpoint_folder, exist_ok=True)
    for epoch in range(params["lora"]["num_epochs"]):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['lora']['num_epochs']} - Training"):
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(**inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_accuracy = correct_train / total_train
        print(f"Epoch [{epoch+1}/{params['lora']['num_epochs']}], Training Loss: {epoch_train_loss:.4f}, Training Accuracy: {epoch_train_accuracy:.4f}")
        # Validation
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{params['lora']['num_epochs']} - Validation"):
                inputs, labels = batch
                inputs = {key: val.to(device) for key, val in inputs.items()}
                labels = labels.to(device)
                outputs = model(**inputs).logits
                val_running_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        val_running_loss /= len(val_loader)
        epoch_val_accuracy = correct_val / total_val
        scheduler.step(val_running_loss)
        if val_running_loss < best_val_loss:
            best_val_loss = val_running_loss
            epochs_since_improvement = 0
            best_epoch = epoch + 1
            model.save_pretrained(checkpoint_folder)
            tokenizer.save_pretrained(checkpoint_folder)
            print("Validation loss improved. Saving model checkpoint.")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= params["lora"]["patience"]:
                print(f"Early stopping triggered. No improvement in {params['lora']['patience']} epochs.")
                break
        print(f"Epoch [{epoch+1}/{params['lora']['num_epochs']}], Validation Loss: {val_running_loss:.4f}, Validation Accuracy: {epoch_val_accuracy:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']}")

if __name__ == "__main__":
    train_lora()