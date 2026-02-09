
import os

os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")
text = "The Man is here."
tokens = model.to_str_tokens(text)
print(f"Text: '{text}'")
print(f"Tokens: {tokens}")
print(f"Length: {len(tokens)}")

ids = model.to_tokens(text)
print(f"IDs: {ids}")
print(f"IDs shape: {ids.shape}")
