"""Quick check of model architectures for forward_from_layer compatibility"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/gpt5')

from model_registry import get_model_spec
from transformers import AutoConfig

for name in ['qwen3', 'glm4', 'deepseek7b']:
    spec = get_model_spec(name)
    cfg = AutoConfig.from_pretrained(str(spec.local_dir), local_files_only=True, trust_remote_code=True)
    print(f"\n=== {name} ===")
    print(f"  model_type: {cfg.model_type}")
    print(f"  architectures: {cfg.architectures}")
    print(f"  num_hidden_layers: {cfg.num_hidden_layers}")
    print(f"  hidden_size: {cfg.hidden_size}")
    
    # Check position encoding
    has_rope = hasattr(cfg, 'rope_theta') or hasattr(cfg, 'rope_scaling')
    print(f"  has_rope: {has_rope}")
    
    # GLM-specific
    for attr in ['position_encoding_2d', 'multi_query_attention', 'num_attention_heads']:
        if hasattr(cfg, attr):
            print(f"  {attr}: {getattr(cfg, attr)}")
    
    # Check for any special attributes
    d = cfg.to_dict()
    for k in sorted(d.keys()):
        if any(x in k.lower() for x in ['rope', 'rotary', 'position', 'embed']):
            if k not in ['num_hidden_layers', 'hidden_size']:
                print(f"  {k}: {d[k]}")
