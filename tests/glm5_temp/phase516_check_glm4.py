"""Check GLM4 model layer structure"""
import sys, gc
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')
import torch
from model_utils import load_model, get_layers, get_model_info, release_model

# Only load GLM4 briefly
model, tokenizer, device = load_model('glm4')
info = get_model_info(model, 'glm4')

print(f"Model class: {info.model_class}")
print(f"n_layers: {info.n_layers}, d_model: {info.d_model}")

# Check model.model structure
inner = model.model
print(f"\nInner model class: {type(inner).__name__}")
print(f"Inner model attributes: {[a for a in dir(inner) if not a.startswith('_')]}")

# Check for norm
for attr in ['norm', 'final_layernorm']:
    if hasattr(inner, attr):
        obj = getattr(inner, attr)
        print(f"  inner.{attr}: {type(obj).__name__}")

# Check for rotary_emb
if hasattr(inner, 'rotary_emb'):
    print(f"  inner.rotary_emb: {type(inner.rotary_emb).__name__}")

# Check first layer structure
layers = get_layers(model)
layer0 = layers[0]
print(f"\nLayer 0 class: {type(layer0).__name__}")
print(f"Layer 0 attributes: {[a for a in dir(layer0) if not a.startswith('_')]}")

# Check what arguments layer0.forward accepts
import inspect
sig = inspect.signature(layer0.forward)
print(f"\nLayer 0 forward signature: {sig}")

# Try calling layer0 with minimal args
enc = tokenizer("test", return_tensors="pt", truncation=True, max_length=16)
ids = enc["input_ids"].to(device)
mask = enc["attention_mask"].to(device)

with torch.no_grad():
    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    h0 = out.hidden_states[0]  # embedding output
    h1 = out.hidden_states[1]  # after layer 0
    print(f"\nh0 shape: {h0.shape}, norm: {h0.float().norm():.2f}")
    print(f"h1 shape: {h1.shape}, norm: {h1.float().norm():.2f}")
    
    # Try calling layer0 manually
    position_ids = torch.arange(ids.shape[1], device=device).unsqueeze(0)
    try:
        # Compute position embeddings
        if hasattr(inner, 'rotary_emb'):
            pos_emb = inner.rotary_emb(h0, position_ids)
            print(f"pos_emb type: {type(pos_emb)}")
            if isinstance(pos_emb, tuple):
                print(f"  cos shape: {pos_emb[0].shape}, sin shape: {pos_emb[1].shape}")
        
        # Try calling layer0 with position_embeddings
        layer_out = layer0(
            h0,
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=pos_emb if hasattr(inner, 'rotary_emb') else None,
        )
        manual_h1 = layer_out[0]
        diff = (manual_h1 - h1).float().norm().item()
        print(f"\nManual vs auto h1 diff: {diff:.6f}")
    except Exception as e:
        print(f"Manual call failed: {e}")
        # Try without position_embeddings
        try:
            layer_out = layer0(
                h0,
                attention_mask=None,
                position_ids=position_ids,
            )
            manual_h1 = layer_out[0]
            diff = (manual_h1 - h1).float().norm().item()
            print(f"Manual (no pos_emb) vs auto h1 diff: {diff:.6f}")
        except Exception as e2:
            print(f"Manual call (no pos_emb) also failed: {e2}")

release_model(model)
gc.collect()
torch.cuda.empty_cache()
print("\nDone!")
