import sys
sys.path.insert(0, 'tests/glm5')
from model_utils import load_model, get_layers, release_model
import torch

m, t, d = load_model('qwen3')
inputs = t('The apple is a fruit', return_tensors='pt')
input_ids = inputs['input_ids'].to(d)
attn_mask = inputs['attention_mask'].to(d)

# Test head_mask with various shapes
n_layers = len(get_layers(m))
n_heads = m.config.num_attention_heads

print(f"n_layers={n_layers}, n_heads={n_heads}")

# Test 1: head_mask shape [n_layers, n_heads]
head_mask = torch.ones(n_layers, n_heads, device=d)
head_mask[0, 0] = 0.0
with torch.no_grad():
    out1 = m(input_ids=input_ids, attention_mask=attn_mask, head_mask=head_mask)
print(f"head_mask [{n_layers}, {n_heads}] works: {out1.logits.shape}")

# Test 2: output_attentions
with torch.no_grad():
    out2 = m(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)
print(f"attentions: {len(out2.attentions)} layers, att[0] shape={out2.attentions[0].shape}")

release_model(m)
