
import os
import sys

# Set environment variables for model loading BEFORE other imports
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Redirect stdout to file
sys.stdout = open("scale_up_results.txt", "w", encoding="utf-8")

import torch
import torch.nn.functional as F

from transformer_lens import HookedTransformer

# Configuration
MODELS = {
    "gpt2": "gpt2",
    "qwen": "Qwen/Qwen3-4B" # Trying to load the cached model
}

from transformers import AutoConfig, AutoModelForCausalLM


def load_model(model_name):
    print(f"\n--- Loading {model_name} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # Special handling for Qwen3 if direct loading fails
        if "Qwen" in model_name:
            print(" Using robust loading for Qwen...")
            try:
                # 1. Load HF Model
                hf_model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    device_map="auto", 
                    trust_remote_code=True,
                    torch_dtype="auto"
                )
                
                # 2. Patch Config if needed
                # Try class-level patching if instance patching fails
                config_class = type(hf_model.config)
                if not hasattr(config_class, "rope_theta"):
                    print(f" Class-level patching rope_theta on {config_class.__name__}...")
                    setattr(config_class, "rope_theta", 1000000.0)
                
                # Also try instance level again just in case
                try:
                    hf_model.config.rope_theta = 1000000.0
                except:
                    pass

                # 3. Load HookedTransformer from HF model
                model = HookedTransformer.from_pretrained(
                    model_name, 
                    hf_model=hf_model,
                    device=device,
                    fold_ln=False,
                    center_writing_weights=False,
                    center_unembed=False,
                    tokenizer=None # Use default or none to avoid tokenizer issues for now
                )
                return model
            except Exception as e:
                print(f" Robust loading failed: {e}")
                # Fallback to standard loading
                pass

        # Standard loading
        model = HookedTransformer.from_pretrained(
            model_name, 
            device=device,
            trust_remote_code=True
        )
        return model
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

def get_resid_at_last_token(model, text, layer_id):
    # HookedTransformer layer names differ by architecture
    # GPT2: blocks.{layer}.hook_resid_post
    # Qwen typically maps to standard TL names too if supported
    
    try:
        # Check if model has 'blocks' (GPT2 style) or 'layers' (Llama style)
        # TL usually unifies to 'blocks'
        cache_name = f"blocks.{layer_id}.hook_resid_post"
        _, cache = model.run_with_cache(text)
        return cache[cache_name][0, -1, :]
    except Exception as e:
        print(f"Error getting activation: {e}")
        return None

def measure_curvature(model, model_label):
    # Define analogous pairs
    # Pair A: Cat -> Dog
    # Pair B: Girl -> Boy
    # In a flat manifold (Euclidean), Vec(Girl->Boy) should be Parallel to Vec(Cat->Dog)
    # CosSim should be 1.0
    
    n_layers = model.cfg.n_layers
    mid_layer = n_layers // 2
    
    print(f"Model: {model_label}, Layer: {mid_layer} (Middle)")
    
    # 1. Get Vectors
    # Note: Qwen might need different prompting or chat template, but let's try raw text first.
    
    # Pair A: Cat -> Dog
    t_cat = "The cat is here."
    t_dog = "The dog is here."
    
    # Pair B: Girl -> Boy
    t_girl = "The girl is here."
    t_boy = "The boy is here."
    
    h_cat = get_resid_at_last_token(model, t_cat, mid_layer)
    h_dog = get_resid_at_last_token(model, t_dog, mid_layer)
    
    h_girl = get_resid_at_last_token(model, t_girl, mid_layer)
    h_boy = get_resid_at_last_token(model, t_boy, mid_layer)
    
    if h_cat is None or h_girl is None:
        print("Skipping curvature measurement due to activation error.")
        return
        
    v_a = h_dog - h_cat # Direction: Cat -> Dog
    v_b = h_boy - h_girl # Direction: Girl -> Boy
    
    sim = F.cosine_similarity(v_a.unsqueeze(0), v_b.unsqueeze(0)).item()
    print(f"Parallelism (Curvature Metric): {sim:.4f}")
    
    if sim > 0.8:
        print("Conclusion: Manifold is Local-Flat / Low Curvature.")
    else:
        print("Conclusion: Manifold is Curved / High Curvature.")
        
    return sim

def main():
    print("Starting Scale-Up Verification...")
    
    # 1. Test GPT-2 (Baseline)
    gpt2 = load_model(MODELS["gpt2"])
    if gpt2:
        measure_curvature(gpt2, "GPT-2 Small")
        del gpt2
        torch.cuda.empty_cache()
    
    # 2. Test Qwen (Scale Up)
    qwen = load_model(MODELS["qwen"])
    if qwen:
        measure_curvature(qwen, "Qwen-4B")
        del qwen
        torch.cuda.empty_cache()
        
    print("\nExperiment Complete.")

if __name__ == "__main__":
    main()
