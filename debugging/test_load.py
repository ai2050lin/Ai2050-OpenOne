
import os

os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import traceback

import torch

from transformer_lens import HookedTransformer

try:
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = HookedTransformer.from_pretrained(
        "gpt2-small", 
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32
    )
    print("Model loaded successfully.")
except Exception:
    with open("debugging/error_log.txt", "w") as f:
        traceback.print_exc(file=f)
    print("Error occurred. Check debugging/error_log.txt")
