"""Quick check of GLM4 layer device placement."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "tests/glm5")

from phase530_state_pair_decomposition import load_model_bf16_flash
from model_utils import get_layers, release_model
import torch

model, tokenizer, device, attn = load_model_bf16_flash("glm4")
layers = get_layers(model)
print("GLM4 device_map:")
for i in range(len(layers)):
    layer = layers[i]
    try:
        d = next(layer.parameters()).device
        print(f"  L{i}: {d}")
    except StopIteration:
        print(f"  L{i}: NO PARAMS")

# Check hf_device_map
if hasattr(model, "hf_device_map"):
    print("\nhf_device_map summary:")
    gpu_count = sum(1 for v in model.hf_device_map.values() if "cuda" in str(v))
    cpu_count = sum(1 for v in model.hf_device_map.values() if "cpu" in str(v))
    meta_count = sum(1 for v in model.hf_device_map.values() if "meta" in str(v))
    print(f"  GPU: {gpu_count}, CPU: {cpu_count}, Meta: {meta_count}")
    # Show layers specifically
    for k, v in model.hf_device_map.items():
        if "layers" in k or k.startswith("model."):
            print(f"  {k}: {v}")

release_model(model)
import gc
gc.collect()
torch.cuda.empty_cache()
