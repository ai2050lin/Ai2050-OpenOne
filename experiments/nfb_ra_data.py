import os

# Set environment variables for model loading (copied from server.py)
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import torch
from tqdm import tqdm

from transformer_lens import HookedTransformer

# Configuration
MODEL_NAME = "gpt2-small" # Fallback to smaller model for reliability
# MODEL_NAME = "gpt2-small"   # Fallback
# MODEL_NAME = "Qwen/Qwen3-4B" # Try Qwen first

# If Qwen fails (e.g. OOM or download), we fallback to GPT2 or Synthetic
FALLBACK_MODEL = "gpt2-small"

BATCH_SIZE = 10
N_GENERAL_SAMPLES = 500 # General text to form the "Background Manifold"
SEQ_LEN = 32 # Keep short for speed/memory
CACHE_DIR = "nfb_data"
os.makedirs(CACHE_DIR, exist_ok=True)

# Specific "Concept" prompts to verifying Fiber structure
# We want to capture: King, Man, Woman, Queen, Paris, France, Berlin, Germany etc.
CONCEPT_PROMPTS = [
    # Gender / Royalty
    "The king sat on the throne.",
    "The queen sat on the throne.",
    "A man walks down the street.",
    "A woman walks down the street.",
    "The prince will be king one day.",
    "The princess will be queen one day.",
    
    # Capital / Country
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Rome is the capital of Italy.",
    "London is the capital of England.",
    "Madrid is the capital of Spain.",
    "Tokyo is the capital of Japan.",
    
    # Technology / Coding
    "Python is a coding language.",
    "JavaScript is a coding language.",
    "C++ is a compiled language.",
    "HTML is a markup language.",
    
    # Simple S-V-O for structure
    " The cat ate the fish.",
    " The dog ate the bone.",
    " The boy ate the apple.",
    " The girl ate the orange."
]

def load_model():
    print(f"Attempting to load {MODEL_NAME}...")
    try:
        if torch.cuda.is_available():
            model = HookedTransformer.from_pretrained(
                MODEL_NAME, 
                device="cuda", 
                dtype=torch.float16,
                trust_remote_code=True
            )
        else:
            print("CUDA not available, using CPU (float32)...")
            model = HookedTransformer.from_pretrained(
                MODEL_NAME, 
                device="cpu", 
                dtype=torch.float32,
                trust_remote_code=True
            )
        return model
    except Exception as e:
        print(f"Failed to load {MODEL_NAME}: {e}")
        print(f"Falling back to {FALLBACK_MODEL}...")
        try:
             model = HookedTransformer.from_pretrained(FALLBACK_MODEL)
             return model
        except Exception as e2:
            print(f"Failed fallback: {e2}")
            return None

def collect_data():
    model = load_model()
    if model is None:
        print("Could not load any model. Aborting real data collection.")
        return

    print(f"Model loaded: {model.cfg.model_name}")
    layer = model.cfg.n_layers // 2 # Probe middle layer
    hook_name = f"blocks.{layer}.hook_resid_post"
    print(f"Probing Layer {layer}: {hook_name}")

    all_acts = []
    metadata = [] # To store what text generated this point
    
    # 1. Collect Concept Data (Repeated to ensure visibility)
    print("Collecting Concept Data...")
    for prompt in tqdm(CONCEPT_PROMPTS * 5): # Repeat 5 times
        with torch.no_grad():
            _, cache = model.run_with_cache(prompt, names_filter=lambda x: x == hook_name)
            act = cache[hook_name][0, -1, :].cpu().numpy() # Last token
            all_acts.append(act)
            metadata.append({"text": prompt, "type": "concept"})

    # 2. Collect General Data (if possible, from simple generation or preset list)
    print("Collecting General Background Data...")
    # Use simple generated text seeding
    seeds = ["The", "A", "It", "When", "If", "Why", "Who", "Where"]
    for i in tqdm(range(N_GENERAL_SAMPLES // len(seeds))):
        for seed in seeds:
            # Generate a few tokens
            gen_text = model.generate(seed, max_new_tokens=10, verbose=False)
            with torch.no_grad():
                _, cache = model.run_with_cache(gen_text, names_filter=lambda x: x == hook_name)
                # Take a random token from middle to end to get "average" state
                # or just the last
                act = cache[hook_name][0, -1, :].cpu().numpy()
                all_acts.append(act)
                metadata.append({"text": gen_text, "type": "general"})

    all_acts_np = np.array(all_acts)
    print(f"Collected {all_acts_np.shape} activations.")
    
    np.save(os.path.join(CACHE_DIR, "hidden_states.npy"), all_acts_np)
    # Save metadata as text/json
    import json
    with open(os.path.join(CACHE_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f)
        
    print("Data saved.")

if __name__ == "__main__":
    collect_data()
