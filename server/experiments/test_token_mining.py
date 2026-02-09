
import os

import torch

from transformer_lens import HookedTransformer

# Set environment variables for model loading (copied from server.py)
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def find_token_index(model, prompt, target_word):
    """
    Original function from nfb_ra_data.py (slightly modified to debugging)
    """
    # 1. Tokenize entire prompt
    prompt_tokens = model.to_tokens(prompt)[0]
    # Convert token IDs to strings for debugging
    str_tokens = model.to_str_tokens(prompt)
    print(f"Prompt: '{prompt}' -> Tokens: {str_tokens}")
    
    target_clean = target_word.strip().lower()
    
    # 2. Heuristic search (original flawed logic)
    for i in range(len(prompt_tokens) - 1, -1, -1):
        token_str = model.to_string(prompt_tokens[i]).strip().lower()
        if target_clean == token_str:
            return i
        if target_clean in token_str and len(token_str) < len(target_clean) + 3:
             return i
             
    return -1

def improved_find_token_index(model, prompt, target_word):
    """
    Improved logic: Find sub-sequence of tokens.
    """
    # 1. Tokenize prompt & target
    prompt_tokens = model.to_tokens(prompt)[0] # [n_ctx]
    # Use prepend_bos=False for target to avoid BOS token mismatch if model adds it by default
    target_tokens = model.to_tokens(target_word, prepend_bos=False)[0] 
    
    # If target is single token, handle simply
    if len(target_tokens) == 1:
        # Search for that token ID
        tgt_id = target_tokens[0]
        # Search backwards
        for i in range(len(prompt_tokens) - 1, -1, -1):
            if prompt_tokens[i] == tgt_id:
                return i
    
    # Heuristic: Search for the target string in the decoded tokens if token IDs don't match exactly due to context
    # But usually, checking token IDs is safest if we tokenize properly.
    # Let's stick to string matching for robustness against BPE quirks with spaces.
    
    target_clean = target_word.strip().lower()
    
    for i in range(len(prompt_tokens) - 1, -1, -1):
        # Check if this token matches the END of the target
        token_str = model.to_string(prompt_tokens[i]).strip().lower()
        
        # If the target word is "microscope" and token is "scope", it might be a match for the end.
        if target_clean.endswith(token_str):
             # Potential match. Verify full word reconstruction backwards?
             # For now, let's just use the string inclusion heuristic which covers 99% 
             # if we just want the "last token" of the concept.
             if len(token_str) > 2 or token_str == target_clean:
                 return i
                 
    return -1

def test_robustness():
    device = "cpu"
    print("Loading model...")
    try:
        model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Case 1: Simple
    prompt1 = "The king sat on the throne."
    target1 = "king"
    idx1 = find_token_index(model, prompt1, target1)
    print(f"Target '{target1}' (Original): Index {idx1}")

    # Case 2: Multi-token
    # "microscope" -> ["micro", "scope"] in GPT-2?
    prompt2 = "He looked through the microscope."
    target2 = "microscope"
    idx2 = find_token_index(model, prompt2, target2)
    print(f"Target '{target2}' (Original): Index {idx2}")
    
    # Check what 'microscope' tokenizes to
    print(f"'microscope' tokens: {model.to_str_tokens('microscope')}")

if __name__ == "__main__":
    test_robustness()
