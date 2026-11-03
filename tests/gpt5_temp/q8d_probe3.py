# -*- coding: utf-8 -*-
"""Q8-D 探针3：GLM 的 base 模块能否直调 + inputs_embeds / attention_mask 是否被接受。"""
import os
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch, inspect
OUT=[]
def say(*a):
    s=' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)

p='D:/develop/model/ZhipuAI/glm-4-9b-chat-hf'
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained(p)
m=AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, device_map='auto',
        max_memory={0:'15GiB','cpu':'48GiB'}, low_cpu_mem_usage=True)
m.eval()
say('has .model=%s  has .transformer=%s' % (hasattr(m,'model'), hasattr(m,'transformer')))
b = getattr(m,'transformer',None) or m.model
say('base class %s' % type(b).__name__)
sig = inspect.signature(b.forward)
say('forward 参数 %s' % list(sig.parameters))
say('get_input_embeddings -> %s' % type(m.get_input_embeddings()).__name__)
say('get_output_embeddings -> %s' % type(m.get_output_embeddings()).__name__)

ids = tok('苹果、香蕉、橘子、', return_tensors='pt')['input_ids'].cuda()
mask = torch.ones_like(ids)
with torch.no_grad():
    o1 = m(input_ids=ids, attention_mask=mask).logits[:, -1, :].float()
    E = m.get_input_embeddings()(ids)
    h = b(inputs_embeds=E, attention_mask=mask, use_cache=False)[0]
    say('base(inputs_embeds) 输出 h %s' % (tuple(h.shape),))
    W = m.get_output_embeddings().weight
    o2 = (h[:, -1, :].float() @ W.float().T)
    say('手算 head vs 原生 head max|Δ| = %.3e  %s'
        % (float((o1-o2).abs().max()), 'PASS' if float((o1-o2).abs().max())<0.5 else 'FAIL'))
    full = m(inputs_embeds=E, attention_mask=mask).logits[:, -1, :].float()
    say('整模型接 inputs_embeds max|Δ| = %.3e' % float((o1-full).abs().max()))
import pathlib
pathlib.Path('tests/gpt5_temp/q8d_probe3.txt').write_text('\n'.join(OUT)+'\n',
                                                          encoding='utf-8', newline='')
say('WROTE')
