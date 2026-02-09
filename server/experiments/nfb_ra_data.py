
import os

# Set environment variables for model loading (copied from server.py)
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json

import numpy as np
import torch
from tqdm import tqdm

from transformer_lens import HookedTransformer

# Configuration
MODEL_NAME = "gpt2"
BATCH_SIZE = 32 # Increased batch size for efficiency
CACHE_DIR = "nfb_data"
CORPUS_FILE = os.path.join(CACHE_DIR, "iso_corpus.json")
OUTPUT_FILE = os.path.join(CACHE_DIR, "trajectory_data.npy")
METADATA_FILE = os.path.join(CACHE_DIR, "metadata.json")

def load_model():
    print(f"Attempting to load {MODEL_NAME}...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = HookedTransformer.from_pretrained(
            MODEL_NAME, 
            device=device, 
            dtype=torch.float16 if device == "cuda" else torch.float32
        )
        return model
    except Exception as e:
        print(f"Failed to load {MODEL_NAME}: {e}")
        return None

def find_token_index(model, prompt, target_word):
    """
    Finds the index of the target word in the prompt.
    Returns the index of the LAST token of the target word.
    """
    try:
        prompt_tokens = model.to_tokens(prompt)[0]
    except:
        return -1
        
    try:
        target_tokens = model.to_tokens(target_word, prepend_bos=False)[0]
    except:
        try:
            target_tokens = model.to_tokens(target_word)[0]
        except:
            return -1

    target_len = len(target_tokens)
    if target_len == 0:
        return -1
        
    prompt_len = len(prompt_tokens)
    
    # Exact match search
    for i in range(prompt_len - target_len, -1, -1):
        if torch.equal(prompt_tokens[i : i + target_len], target_tokens):
            return i + target_len - 1
            
    # Heuristic fallback
    target_clean = target_word.strip().lower()
    for i in range(len(prompt_tokens) - 1, -1, -1):
        token_str = model.to_string(prompt_tokens[i]).strip().lower()
        if target_clean == token_str:
            return i
        if len(token_str) > 2 and target_clean.endswith(token_str):
             return i
    return -1

def collect_data():
    if not os.path.exists(CORPUS_FILE):
        print(f"Corpus file not found: {CORPUS_FILE}. Run generate_iso_corpus.py first.")
        return

    with open(CORPUS_FILE, 'r') as f:
        corpus = json.load(f)

    model = load_model()
    if model is None:
        return

    print(f"Model loaded: {model.cfg.model_name}")
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    
    # Define hooks
    # t=0: Input (Embedding + Pos) -> blocks.0.hook_resid_pre
    # t=1..L: Output of block i -> blocks.i.hook_resid_post
    hook_names = ["blocks.0.hook_resid_pre"] + [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]
    
    trajectories = []
    valid_metadata = []
    
    print("Pre-calculating indices...")
    # Pre-process to find indices and filter valid entries
    prepared_batch = []
    
    for entry in tqdm(corpus):
        prompt = entry["text"]
        target = entry["target_word"]
        idx = find_token_index(model, prompt, target)
        
        if idx != -1:
            entry["token_index"] = idx
            prepared_batch.append(entry)
            
    print(f"Valid entries: {len(prepared_batch)} / {len(corpus)}")
    
    # Process in batches
    # We iterate through prepared_batch in chunks
    
    # Limit for testing/demo if needed, but we can process all 20k
    # prepared_batch = prepared_batch[:1000] 
    
    print(f"Processing {len(prepared_batch)} samples in batches of {BATCH_SIZE}...")
    
    for i in tqdm(range(0, len(prepared_batch), BATCH_SIZE)):
        batch_entries = prepared_batch[i : i + BATCH_SIZE]
        prompts = [e["text"] for e in batch_entries]
        indices = [e["token_index"] for e in batch_entries]
        
        # Run model on batch
        # HookedTransformer handles padding automatically if input is list of strings
        with torch.no_grad():
            # We must use padding_side='left' usually for generation, but for analysis 'right' is often default?
            # HookedTransformer default is left padding for generation, but right padding for validation?
            # Actually run_with_cache handles it.
            # IMPORTANT: token indices 'idx' are relative to the UNPADDED sequence if we calculated them individually.
            # If we batch, token positions shift due to padding!
            
            # To avoid padding index confusion, simplest way is to force left padding and adjust indices
            # OR, process individually if batching logic is complex.
            # But wait, HookedTransformer.to_tokens(prompts) returns padded tensor.
            # If default padding is LEFT, then index `i` becomes `i + padding_len`.
            
            # Let's check padding side.
            model.tokenizer.padding_side = 'left' 
            model.tokenizer.pad_token = model.tokenizer.eos_token
            
            tokens = model.to_tokens(prompts) # [batch, max_len]
            
            # Recalculate indices for the padded batch?
            # If we know the unpadded index `idx` and the length of tokens `L_orig`
            # and padded length `L_pad`, and padding is LEFT, then `idx_new = idx + (L_pad - L_orig)`.
            
            # Let's verify lengths
            attention_mask = (tokens != model.tokenizer.pad_token_id)
            # This logic assumes pad_token is EOS, which might be true for GPT2.
            # Be careful.
            
            # Alternative: Just run individually but loop is fast in C++?
            # No, python loop overhead is high.
            
            # Safer Batched Approach:
            # 1. Tokenize batch.
            # 2. Find target indices IN THE PADDED TENSOR.
            
            # Re-find indices in the actual batch tensor
            batch_indices = []
            valid_batch_entries = [] # Some might fail if max_len truncated? (unlikely)
            
            # Since we already know the target token (last token of target word),
            # we can look for it in the batch tokens.
            # But the specific tokens might change context slightly? No.
            
            # Simple approach: Left padding shift
            for j, entry in enumerate(batch_entries):
                orig_idx = indices[j]
                # Determine padding shift
                # prompt_tokens = model.to_tokens(entry["text"])[0] 
                # ^ This is redundant but safe.
                
                # Length of unpadded:
                l_unpadded = len(model.to_tokens(entry["text"])[0])
                l_padded = tokens.shape[1]
                shift = l_padded - l_unpadded
                
                batch_indices.append(orig_idx + shift)
                valid_batch_entries.append(entry)

            # Run model
            # cache is a Cache object
            _, cache = model.run_with_cache(tokens, names_filter=lambda x: x in hook_names)
            
            # Extract
            for j, batch_idx in enumerate(batch_indices):
                # Check bounds
                if batch_idx >= tokens.shape[1]:
                    continue # Should not happen
                    
                traj_point = []
                
                # Act 0
                act_0 = cache["blocks.0.hook_resid_pre"][j, batch_idx, :].cpu().numpy()
                traj_point.append(act_0)
                
                # Layers
                for layer in range(n_layers):
                    act_i = cache[f"blocks.{layer}.hook_resid_post"][j, batch_idx, :].cpu().numpy()
                    traj_point.append(act_i)
                    
                trajectories.append(np.array(traj_point))
                valid_metadata.append(valid_batch_entries[j])

    # Save
    if not trajectories:
        print("No valid trajectories collected.")
        return

    all_trajectories = np.array(trajectories)
    print(f"Collected shape: {all_trajectories.shape}")

    np.save(OUTPUT_FILE, all_trajectories)
    with open(METADATA_FILE, "w") as f:
        json.dump(valid_metadata, f, indent=2)
        
    print("Trajectory data saved.")

if __name__ == "__main__":
    collect_data()
