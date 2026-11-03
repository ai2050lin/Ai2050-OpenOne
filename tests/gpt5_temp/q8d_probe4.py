# -*- coding: utf-8 -*-
"""Q8-D 探针4：GLM 的 logits 到底是 h 的什么函数。逐行刷盘，崩了也能看。"""
import os, pathlib
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
F = pathlib.Path('tests/gpt5_temp/q8d_probe4.txt')
F.write_text('', encoding='utf-8')
def say(*a):
    s=' '.join(str(x) for x in a); print(s, flush=True)
    with F.open('a', encoding='utf-8', newline='') as f:
        f.write(s+'\n')

p='D:/develop/model/ZhipuAI/glm-4-9b-chat-hf'
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained(p)
say('tokenizer ok')
m=AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, device_map='auto',
        max_memory={0:'15GiB','cpu':'20GiB'}, low_cpu_mem_usage=True).eval()
say('model ok')
ids = tok('苹果、香蕉、橘子、', return_tensors='pt')['input_ids'].cuda()
mask = torch.ones_like(ids)
W = m.get_output_embeddings().weight
say('lm_head %s %s dev=%s dtype=%s' % (type(m.get_output_embeddings()).__name__,
                                       tuple(W.shape), W.device, W.dtype))
say('vocab_size=%s hidden=%s tie=%s' % (m.config.vocab_size, m.config.hidden_size,
                                        m.config.tie_word_embeddings))
say('emb is head: %s' % (m.get_input_embeddings().weight is W))
with torch.no_grad():
    out = m(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    o1 = out.logits[:, -1, :].float()
    hs = out.hidden_states
    say('logits %s  n_hs=%d  hs[-1] %s' % (tuple(o1.shape), len(hs), tuple(hs[-1].shape)))
    for tag, hv in (('hs[-1]', hs[-1][:, -1, :]), ('hs[-2]', hs[-2][:, -1, :])):
        o2 = hv.float() @ W.float().T
        d = float((o1-o2).abs().max())
        c = float(torch.corrcoef(torch.stack([o1[0], o2[0]]))[0, 1])
        say('%s: max|Δ|=%.4f corr=%.6f  std o1=%.3f o2=%.3f'
            % (tag, d, c, float(o1.std()), float(o2.std())))
    hb = m.model(input_ids=ids, attention_mask=mask, use_cache=False)[0][:, -1, :]
    o3 = hb.float() @ W.float().T
    say('model()[0]: max|Δ|=%.4f' % float((o1-o3).abs().max()))
    say('o1[:5]=%s' % [round(float(x),2) for x in o1[0,:5]])
    say('o2[:5]=%s' % [round(float(x),2) for x in hs[-1][:, -1, :].float() @ W.float().T[:,][:5] if False])
say('DONE')
