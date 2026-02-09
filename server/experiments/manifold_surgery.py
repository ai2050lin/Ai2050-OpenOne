
import os

# Set environment variables for model loading
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys

import torch
import torch.nn.functional as F

from transformer_lens import HookedTransformer

# Redirect stdout to file
sys.stdout = open("clean_output.txt", "w", encoding="utf-8")

# Configuration
MODEL_NAME = "gpt2"
LAYER_ID = 6
ALPHA_VALUES = [0.0, 1.0, 2.0, 5.0, 10.0]
SOURCE_PAIR = ("Man", "Woman")
TARGET_PROMPT = "The King is"
TARGET_TOKEN_INDEX = 1 

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_activation_at_token(model, text, target_word, layer_id):
    # Find the token index of the target word using IDs
    # We assume target_word is a single token with leading space
    try:
        # Get IDs for the full text
        prompt_ids = model.to_tokens(text)[0] # [seq_len]
        
        # Get ID for the target word (with space)
        # Try both with and without space
        target_token_id = -1
        try:
            target_token_id = model.to_single_token(" " + target_word)
        except:
            try:
                target_token_id = model.to_single_token(target_word)
            except:
                pass
        
        idx = -1
        if target_token_id != -1:
            # Search backwards
            for i in range(len(prompt_ids)-1, -1, -1):
                if prompt_ids[i] == target_token_id:
                    idx = i
                    break
        
        # Helper for debug printing (vocab check)
        token_str = model.to_string(prompt_ids[idx]) if idx != -1 else "N/A"
        print(f"DEBUG: text='{text}', target='{target_word}', found_idx={idx}, token_id={target_token_id}, token_str='{token_str}'")

        if idx == -1:
            # Fallback to last token if not found
            idx = -1
            
        cache_name = f"blocks.{layer_id}.hook_resid_post"
        _, cache = model.run_with_cache(text)
        act = cache[cache_name][0, idx, :]
        print(f"DEBUG: Act norm={act.norm().item():.4f}")
        return act
        
    except Exception as e:
        print(f"Error extracting from '{text}': {e}")
        return torch.zeros(model.cfg.d_model).to(model.cfg.device)

def get_transport_vector(model, pair, layer_id):
    print(f"Calculating robust transport vector for {pair}...")
    templates = [
        "The {} is here.",
        "A {} walked in.",
        "I saw a {}.",
        "Hello {}.",
        "The {} sat down."
    ]
    
    vecs = []
    for temp in templates:
        txt_a = temp.format(pair[0])
        txt_b = temp.format(pair[1])
        
        v_a = get_activation_at_token(model, txt_a, pair[0], layer_id)
        v_b = get_activation_at_token(model, txt_b, pair[1], layer_id)
        
        vecs.append(v_b - v_a)
        
    # Average the difference vectors
    avg_vec = torch.stack(vecs).mean(dim=0)
    return avg_vec

def main():
    print("Starting experiment (Robust)...")
    model = load_model()
    if model is None:
        return
    
    transport_vec = get_transport_vector(model, SOURCE_PAIR, LAYER_ID)
    print(f"Transport Vector Norm: {transport_vec.norm().item():.4f}")
    
    print(f"Target Prompt: '{TARGET_PROMPT}'")
    
    hook_name = f"blocks.{LAYER_ID}.hook_resid_post"
    
    # Target words to monitor
    targets = [" man", " woman", " King", " Queen", " husband", " wife", " dead", " alive", " powerful", " beautiful", " strong", " pregnant"]
    target_ids = []
    for t in targets:
        try:
            target_ids.append(model.to_single_token(t))
        except:
            print(f"Warning: Could not get token for '{t}'")
            target_ids.append(-1)
            
    header = f"{'Alpha':<6} | " + " | ".join([f"{t.strip():<9}" for t in targets])
    print(header)
    print("-" * len(header))

    for alpha in ALPHA_VALUES:
        
        def steering_hook(resid, hook):
            # resid: [batch, pos, d_model]
            resid[0, TARGET_TOKEN_INDEX, :] += alpha * transport_vec
            return resid
            
        if alpha == 0.0:
            logits = model(TARGET_PROMPT)
        else:
            with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
                logits = model(TARGET_PROMPT)
            
        probs = F.softmax(logits[0, -1], dim=-1)
        
        row_vals = []
        for tid in target_ids:
            if tid != -1:
                p = probs[tid].item()
            else:
                p = 0.0
            row_vals.append(p)
            
        row_str = f"{alpha:<6} | " + " | ".join([f"{v:.4f}   " for v in row_vals])
        print(row_str)

    print("\nExperiment Complete.")

if __name__ == "__main__":
    main()
