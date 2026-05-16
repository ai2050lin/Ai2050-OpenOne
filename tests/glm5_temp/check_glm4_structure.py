"""Quick check GLM4 model structure for logit lens"""
import sys, gc, torch
sys.path.insert(0, 'tests/glm5')
from model_utils import MODEL_CONFIGS, release_model

model_path = MODEL_CONFIGS["glm4"]["path"]
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True, local_files_only=True)

print(f"Model class: {type(model).__name__}")
print(f"Has model.model: {hasattr(model, 'model')}")
if hasattr(model, 'model'):
    print(f"Has model.model.norm: {hasattr(model.model, 'norm')}")
    norm_attrs = [a for a in dir(model.model) if 'norm' in a.lower()]
    print(f"Norm-related attrs in model.model: {norm_attrs}")

# Check norm weight
if hasattr(model, 'model') and hasattr(model.model, 'norm'):
    norm = model.model.norm
    w = norm.weight
    print(f"norm.weight: is_meta={w.is_meta}, device={w.device}, shape={w.shape}")
    print(f"norm type: {type(norm).__name__}")
    print(f"norm eps: {norm.eps if hasattr(norm, 'eps') else 'N/A'}")

# Check lm_head weight
if hasattr(model, 'lm_head'):
    w = model.lm_head.weight
    print(f"lm_head.weight: is_meta={w.is_meta}, device={w.device}, shape={w.shape}")

# Try to load norm weight from safetensors
if w.is_meta:
    print("lm_head.weight is on meta device, trying safetensors...")
    import glob, os
    from safetensors import safe_open
    sf_files = glob.glob(os.path.join(model_path, '*.safetensors'))
    for sf_file in sf_files:
        with safe_open(sf_file, framework='pt', device='cpu') as sf:
            keys = list(sf.keys())
            norm_keys = [k for k in keys if 'norm' in k and 'layer' not in k]
            lm_keys = [k for k in keys if 'lm_head' in k or 'embed_out' in k]
            if norm_keys:
                print(f"  Norm keys in {os.path.basename(sf_file)}: {norm_keys}")
            if lm_keys:
                print(f"  lm_head keys in {os.path.basename(sf_file)}: {lm_keys}")

release_model(model)
