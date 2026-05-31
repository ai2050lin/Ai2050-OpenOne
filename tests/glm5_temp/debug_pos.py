"""Debug position calculation"""
import sys
sys.path.insert(0, ".")
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, local_files_only=True)

sent = "the person is happy"
inputs = tok(sent, return_tensors="pt").to(model.device)
print(f"Input IDs shape: {inputs['input_ids'].shape}")
print(f"Input IDs: {inputs['input_ids'][0].tolist()}")
print(f"Decoded: {[tok.decode([x]) for x in inputs['input_ids'][0].tolist()]}")

with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)

hs0 = out.hidden_states[0]
print(f"Hidden states[0] shape: {hs0.shape}")

# The target "happy" should be at position...
target = "happy"
toks_no_bos = tok.encode(sent, add_special_tokens=False)
print(f"Tokens (no BOS): {toks_no_bos}")
decoded_no_bos = [tok.decode([x]) for x in toks_no_bos]
print(f"Decoded (no BOS): {decoded_no_bos}")

# With BOS
toks_with_bos = tok.encode(sent)
print(f"Tokens (with BOS): {toks_with_bos}")
decoded_with_bos = [tok.decode([x]) for x in toks_with_bos]
print(f"Decoded (with BOS): {decoded_with_bos}")

# Find happy position in with-BOS encoding
for prefix in ['', ' ']:
    target_ids = tok.encode(prefix + target, add_special_tokens=False)
    print(f"  target '{prefix}{target}' -> {target_ids}")
    for i in range(len(toks_with_bos) - len(target_ids) + 1):
        if toks_with_bos[i:i+len(target_ids)] == target_ids:
            print(f"    MATCH at position {i} in with-BOS encoding")
            print(f"    Hidden state at pos {i} has shape: {hs0[0, i, :].shape}")

del model
torch.cuda.empty_cache()
