
import os
import sys

# Set environment variables
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def debug_loading():
    model_name = "Qwen/Qwen3-4B"
    print(f"Attempting to load config for {model_name}...")
    
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        print("Config loaded successfully.")
        print(f"Config class: {type(config)}")
        
        if hasattr(config, "rope_theta"):
            print(f"rope_theta: {config.rope_theta}")
        else:
            print("rope_theta NOT found in config object.")
            # Print all attributes
            print("Attributes:", dir(config))
            
    except Exception as e:
        print(f"Error loading config: {e}")

if __name__ == "__main__":
    debug_loading()
