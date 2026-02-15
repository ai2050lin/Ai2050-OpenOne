
import torch

from transformer_lens import HookedTransformer

try:
    print("Attempting to load gpt2-small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    print("Success loading gpt2-small")
except Exception as e:
    print(f"Failed loading gpt2-small: {e}")

try:
    print("Attempting to load gpt2...")
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    print("Success loading gpt2")
except Exception as e:
    print(f"Failed loading gpt2: {e}")
