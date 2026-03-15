"""
Download BERT model during Render build step.
This avoids storing the ~440MB model in the git repo.

Run this ONCE during build: python download_model.py
"""
import os
from huggingface_hub import snapshot_download

MODEL_DIR = "models/bert_intent"
HF_REPO = "yitommy317/fubon-intent-classifier"

def download():
    if os.path.exists(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, 'model.safetensors')):
        print(f"Model already exists at {MODEL_DIR}, skipping download.")
        return

    print(f"Downloading model from {HF_REPO}...")

    # Use snapshot_download to get ALL files (including label_encoder.pkl)
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=MODEL_DIR,
    )
    print(f"Model saved to {MODEL_DIR}")

if __name__ == "__main__":
    download()
