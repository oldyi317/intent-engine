"""
Download BERT model during Render build step.
This avoids storing the ~440MB model in the git repo.

Run this ONCE during build: python download_model.py
"""
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models/bert_intent_classifier"

def download():
    if os.path.exists(MODEL_DIR):
        print(f"Model already exists at {MODEL_DIR}, skipping download.")
        return

    print("Downloading bert-base-uncased for intent classification...")

    # Download the base model (will be fine-tuned weights if you upload them)
    # If you have a HuggingFace Hub model, change the model name here
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

    os.makedirs(MODEL_DIR, exist_ok=True)
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")

if __name__ == "__main__":
    download()
