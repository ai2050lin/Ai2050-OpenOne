import sys
sys.path.insert(0, 'tests/glm5')
from model_utils import load_model, get_layers, get_model_info, release_model
m, t, d = load_model('qwen3', use_8bit=False)
layers = get_layers(m)
sa = layers[0].self_attn
print(f'self_attn type: {type(sa).__name__}')
print(f'num_heads: {getattr(sa, "num_heads", "N/A")}')
print(f'head_dim: {getattr(sa, "head_dim", "N/A")}')
print(f'o_proj shape: {sa.o_proj.weight.shape}')
print(f'config: num_attention_heads={m.config.num_attention_heads}, hidden_size={m.config.hidden_size}')
print(f'q_proj shape: {sa.q_proj.weight.shape}')
print(f'v_proj shape: {sa.v_proj.weight.shape}')
# 计算实际的head_dim
if hasattr(sa, 'num_heads') and sa.num_heads:
    actual_head_dim = sa.q_proj.weight.shape[0] // sa.num_heads
    print(f'actual_head_dim = q_out_dim / num_heads = {sa.q_proj.weight.shape[0]} / {sa.num_heads} = {actual_head_dim}')
release_model(m)
