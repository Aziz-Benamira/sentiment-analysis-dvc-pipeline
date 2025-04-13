import pandas as pd
import re
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import yaml
from nltk.corpus import stopwords
import nltk

def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def handle_null_values(df, columns_to_keep=None):
    """Handle null values"""
    params = load_params()["preprocess"]
    missing_percentage = df.isnull().mean().mean() * 100
    print(f"Percentage of missing values: {missing_percentage:.4f}%")
    if missing_percentage < params["missing_threshold"] * 100:
        print("Missing values < 5%. Removing rows with missing values...")
        df_cleaned = df.dropna()
        print(f"Rows removed: {len(df) - len(df_cleaned)}")
    else:
        print("Missing values >= 5%. Keeping rows where specified columns are not empty...")
        if columns_to_keep is None:
            print("No columns specified. Returning original DataFrame.")
            df_cleaned = df
        else:
            df_cleaned = df.dropna(subset=columns_to_keep)
            print(f"Rows removed: {len(df) - len(df_cleaned)}")
    return df_cleaned

def preprocess():
    """Preprocess train.csv and test.csv."""
    params = load_params()["preprocess"]
    # Download NLTK stopwords
    nltk.download("stopwords", quiet=True)
    # Custom stopwords
    my_stopwords = stopwords.words("english").copy()
    stopwords_to_keep = {
        "not", "no", "nor", "don't", "isn't", "aren't", "couldn't", "didn't", "doesn't",
        "hadn't", "hasn't", "haven't", "mightn't", "mustn't", "needn't", "shouldn't",
        "wasn't", "weren't", "won't", "wouldn't", "but", "however", "although", "though"
    }
    my_stopwords = list(set(my_stopwords) - stopwords_to_keep)
    # Load datasets
    train_df = pd.read_csv("data/raw/train.csv",encoding='ISO-8859-1')
    test_df = pd.read_csv("data/raw/test.csv",encoding='ISO-8859-1')
    # Handle null values
    train_df = handle_null_values(train_df)
    test_df = handle_null_values(test_df, columns_to_keep=["text", "sentiment"])
    # Clean text
    def clean_text(text):
        text = str(text).lower()
        text = " ".join([word for word in text.split() if word not in my_stopwords])
        text = re.sub(r"https?://[A-Za-z0-9./_?=#]+", " ", text)  # Remove URLs
        text = re.sub(r"[^a-zA-Z0-9\s!?*$]", "", text)  # Remove punctuation
        text = re.sub(r"\s+", " ", text)  # Remove extra spaces
        return text.strip()
    train_df["processed_text"] = train_df["text"].apply(clean_text)
    test_df["processed_text"] = test_df["text"].apply(clean_text)
    # Map sentiments
    sentiment_mapping = {"positive": 2, "neutral": 1, "negative": 0}
    train_df["sentiment_class"] = train_df["sentiment"].map(sentiment_mapping)
    test_df["sentiment_class"] = test_df["sentiment"].map(sentiment_mapping)
    # Split train into train+validation
    train_df, val_df = train_test_split(
        train_df,
        test_size=params["val_size"],
        random_state=42,
        stratify=train_df["sentiment_class"]
    )
    # Save processed CSVs
    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    # Tokenize for BERT
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for split, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        encodings = tokenizer(
            split["processed_text"].tolist(),
            truncation=True,
            padding=True,
            max_length=params["max_length"],
            return_tensors="pt"
        )
        torch.save(
            {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": torch.tensor(split["sentiment_class"].values, dtype=torch.long)
            },
            f"data/processed/{name}_tokenized.pt"
        )

if __name__ == "__main__":
    preprocess()