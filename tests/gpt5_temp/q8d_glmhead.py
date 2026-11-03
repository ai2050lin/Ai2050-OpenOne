# -*- coding: utf-8 -*-
"""GLM 的 logits 到底怎么来的：逐行刷盘。"""
import os, pathlib, json
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
F=pathlib.Path('tests/gpt5_temp/q8d_glmhead.txt'); F.write_text('',encoding='utf-8')
def say(*a):
    s=' '.join(str(x) for x in a); print(s,flush=True)
    with F.open('a',encoding='utf-8',newline='') as f: f.write(s+'\n')

P='D:/develop/model/ZhipuAI/glm-4-9b-chat-hf'
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained(P)
m=AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, device_map='auto',
      max_memory={0:'15GiB','cpu':'20GiB'}, low_cpu_mem_usage=True).eval()
say('loaded')
W=m.get_output_embeddings().weight
say('W %s dev=%s' % (tuple(W.shape), W.device))
core=m.model
say('core %s  has norm=%s' % (type(core).__name__, hasattr(core,'norm')))
ids=tok('苹果、香蕉、橘子、',return_tensors='pt')['input_ids'].cuda()
mask=torch.ones_like(ids)
with torch.no_grad():
    o=m(input_ids=ids,attention_mask=mask,output_hidden_states=True)
    ref=o.logits[0,-1,:].float()
    hs=o.hidden_states
    say('n_hs=%d hs[-1] %s' % (len(hs), tuple(hs[-1].shape)))
    cands={'hs[-1]':hs[-1][0,-1,:]}
    if hasattr(core,'norm'):
        cands['norm(hs[-1])']=core.norm(hs[-1][0,-1,:].unsqueeze(0))[0]
    # 也试 core 的直接输出
    co=core(input_ids=ids,attention_mask=mask,use_cache=False)
    cands['core[0]']=co[0][0,-1,:]
    for k,h in cands.items():
        o2=(h.float()@W.float().T)
        c=float(torch.corrcoef(torch.stack([ref,o2]))[0,1])
        a=float((ref*o2).sum()/(o2*o2).sum())
        say('%-14s max|Δ|=%.3f corr=%.6f  a=%.4f  |ref|=%.2f |o2|=%.2f'
            % (k, float((ref-o2).abs().max()), c, a, float(ref.norm()), float(o2.norm())))
    say('ref top5 %s' % tok.decode(ref.topk(5).indices.tolist()))
    say('o2  top5 %s' % tok.decode((hs[-1][0,-1,:].float()@W.float().T).topk(5).indices.tolist()))
    say('ref[:5] %s' % [round(float(x),3) for x in ref[:5]])
    say('o2 [:5] %s' % [round(float(x),3) for x in (hs[-1][0,-1,:].float()@W.float().T)[:5]])
say('DONE')
