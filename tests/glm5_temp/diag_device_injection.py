"""Quick diagnostic: test injection hook on GLM4 deep layers"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc, torch, numpy as np
from model_utils import get_layers, get_model_info, release_model, MODEL_CONFIGS
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "glm4"
cfg = MODEL_CONFIGS[model_name]

tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True,
                                           local_files_only=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True, local_files_only=True, attn_implementation="sdpa")
model.eval()

info = get_model_info(model, model_name)
layers = get_layers(model)

print(f"n_layers={info.n_layers}")

# Check device map
if hasattr(model, 'hf_device_map'):
    dmap = model.hf_device_map
    layer_devices = {}
    for k, v in dmap.items():
        if k.startswith('model.layers.'):
            lid = k.split('.')[2]
            if lid not in layer_devices:
                layer_devices[lid] = str(v)
    
    gpu_layers = sum(1 for v in layer_devices.values() if 'cuda' in v)
    cpu_layers = sum(1 for v in layer_devices.values() if 'cpu' in v)
    print(f"Layer devices: {gpu_layers} GPU + {cpu_layers} CPU")
    for lid in sorted(layer_devices.keys(), key=int):
        print(f"  Layer {lid}: {layer_devices[lid]}")

# Test forward pass with injection at each layer
text = "The apple is a"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
# Find embed device
embed_device = model.model.embed_tokens.weight.device
input_ids = input_ids.to(embed_device)
attention_mask = torch.ones_like(input_ids)
last_pos = input_ids.shape[1] - 1

# Baseline forward
with torch.no_grad():
    out_base = model(input_ids=input_ids, attention_mask=attention_mask)
    base_logits = out_base.logits[0, -1].float().cpu().numpy()
print(f"Base logits norm: {np.linalg.norm(base_logits):.2f}")

# Test injection at layers 0, 10, 20, 25, 30, 35, 39
test_layers = [0, 10, 20, 25, 30, 35, 39]
inject_vec = np.random.randn(info.d_model).astype(np.float32)
inject_vec = inject_vec / np.linalg.norm(inject_vec)

for li in test_layers:
    if li >= info.n_layers:
        continue
    
    # Get layer device from parameters
    layer = layers[li]
    try:
        param_dev = next(layer.parameters()).device
    except StopIteration:
        param_dev = "unknown"
    
    # Test injection
    inj_tensor = torch.tensor(inject_vec * 2.0, dtype=torch.bfloat16)
    # Don't specify device - let the hook handle it
    
    injection_done = [False]
    injection_success = [False]
    
    def make_hook(vec, _li=li):
        def hook(m, inp, out):
            try:
                if isinstance(out, tuple):
                    new_out = out[0].clone()
                    # Move vec to output device
                    v = vec.to(out[0].device).to(out[0].dtype)
                    new_out[0, last_pos] = new_out[0, last_pos] + v
                    injection_done[0] = True
                    return (new_out,) + out[1:]
            except Exception as e:
                print(f"  L{_li} hook error: {e}")
            return out
        return hook
    
    hook = layers[li].register_forward_hook(make_hook(inj_tensor, li))
    
    try:
        with torch.no_grad():
            out_inj = model(input_ids=input_ids, attention_mask=attention_mask)
            inj_logits = out_inj.logits[0, -1].float().cpu().numpy()
        injection_success[0] = True
        delta = np.linalg.norm(inj_logits - base_logits)
        print(f"  L{li}: param_dev={param_dev}, hook_fired={injection_done[0]}, success={injection_success[0]}, logit_delta={delta:.2f}")
    except Exception as e:
        print(f"  L{li}: param_dev={param_dev}, hook_fired={injection_done[0]}, FAILED: {e}")
    
    hook.remove()

release_model(model)
print("Diagnostic complete!")
