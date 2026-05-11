import sys
sys.path.insert(0, 'tests/glm5')
from model_utils import load_model, get_layers, release_model
import torch

m, t, d = load_model('qwen3')
inputs = t('The apple', return_tensors='pt')
input_ids = inputs['input_ids'].to(d)
attn_mask = inputs['attention_mask'].to(d)

with torch.no_grad():
    out = m(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)
    
print('attentions:', type(out.attentions), len(out.attentions) if out.attentions else 'None')
if out.attentions:
    print('att[0] shape:', out.attentions[0].shape)

# Also test head_mask
try:
    n_layers = len(get_layers(m))
    n_heads = m.config.num_attention_heads
    head_mask = torch.ones(n_layers, n_heads, device=d)
    head_mask[0, 0] = 0.0  # zero out head 0 of layer 0
    with torch.no_grad():
        out2 = m(input_ids=input_ids, attention_mask=attn_mask, head_mask=head_mask)
    print('head_mask works! logits shape:', out2.logits.shape)
except Exception as e:
    print(f'head_mask failed: {e}')

release_model(m)
