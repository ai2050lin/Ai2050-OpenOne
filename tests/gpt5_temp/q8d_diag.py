# -*- coding: utf-8 -*-
"""Q8-D 材料诊断：模型在这些 prompt 后面到底预测什么，p_M 为什么只有 0.026。"""
import os, pathlib, sys
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np, torch
from wordfreq import zipf_frequency
F = pathlib.Path('tests/gpt5_temp/q8d_diag.txt'); F.write_text('', encoding='utf-8')
def say(*a):
    s=' '.join(str(x) for x in a); print(s, flush=True)
    with F.open('a', encoding='utf-8', newline='') as f: f.write(s+'\n')

P='D:/develop/model/Qwen/Qwen3-4B'
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained(P)
m=AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, device_map='cuda:0').eval()
V=m.get_output_embeddings().weight.shape[0]
z=np.full(V, np.nan)
for i in range(V):
    s=tok.decode([i])
    if s and '\ufffd' not in s:
        v=zipf_frequency(s,'zh')
        if v>0: z[i]=v
ok=torch.tensor(np.isfinite(z)).cuda(); zt=torch.tensor(np.where(np.isfinite(z), z, 0)).cuda()
say('V=%d 有 zipf %d' % (V, int(ok.sum())))

def show(prompts, tag, mid=False):
    say('--- %s ---' % tag)
    for p in prompts:
        ids = tok(p, add_special_tokens=False)['input_ids']
        t = torch.tensor([ids]).cuda()
        with torch.no_grad():
            lg = m(input_ids=t).logits[0, -1, :].float()
        pr = torch.softmax(lg, -1)
        pM = float(pr[ok].sum())
        top = pr.topk(6)
        items = []
        for v_, i_ in zip(top.values.tolist(), top.indices.tolist()):
            items.append('%r:%.3f(z=%.1f)' % (tok.decode([i_]), v_, z[i_]))
        say('  %-22s pM=%.3f  H=%.2f | %s' % (p, pM, float(-(pr*torch.log(pr.clamp_min(1e-30))).sum()), ' '.join(items)))

A = ['他把苹果', '她每天早上都要喝一杯牛奶', '我们家门口种了一棵苹果',
     '桌子上放着一个苹果', '她很喜欢吃苹果', '山坡上开满了苹果']
show(A, '现用模板（句尾读）')

B = ['他把苹果', '她把香蕉', '桌子上放着橘子', '我昨天买了西瓜', '她很喜欢吃葡萄']
# 中段读：读到名词为止
say('--- 中段读（读到名词末位，看下一个词）---')
for full in ['他把苹果放在桌子上', '她把香蕉放进书包里', '桌子上放着橘子', '我昨天买了西瓜',
             '她很喜欢吃葡萄', '山坡上开满了花朵']:
    ids = tok(full, add_special_tokens=False)['input_ids']
    for cut in range(2, len(ids)):
        pass
    # 用字符串前缀定位：取前 4 个字
    pre = full[:4]
    pi = tok(pre, add_special_tokens=False)['input_ids']
    t = torch.tensor([pi]).cuda()
    with torch.no_grad():
        lg = m(input_ids=t).logits[0, -1, :].float()
    pr = torch.softmax(lg, -1); pM=float(pr[ok].sum()); top=pr.topk(4)
    items = ['%r:%.3f(z=%.1f)' % (tok.decode([i_]), v_, z[i_])
             for v_, i_ in zip(top.values.tolist(), top.indices.tolist())]
    say('  %-14s pM=%.3f | %s' % (pre, pM, ' '.join(items)))
say('DONE')
torch.cuda.empty_cache()
