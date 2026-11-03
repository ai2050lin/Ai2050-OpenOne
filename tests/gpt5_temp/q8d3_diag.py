# -*- coding: utf-8 -*-
"""Q8-D3 诊断：chat 模板包装后末位模型在预测什么。逐行刷盘。"""
import os, pathlib
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
F = pathlib.Path('tests/gpt5_temp/q8d3_diag.txt'); F.write_text('', encoding='utf-8')
def say(*a):
    s=' '.join(str(x) for x in a); print(s, flush=True)
    with F.open('a', encoding='utf-8', newline='') as f: f.write(s+'\n')

P='D:/develop/model/Qwen/Qwen3-4B'
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained(P)
m=AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, device_map='cuda:0').eval()

raw='他把苹果'
for tag, kw in (('thinking默认', {}),
                ('enable_thinking=False', {'enable_thinking': False})):
    try:
        txt = tok.apply_chat_template([{'role':'user','content':raw}],
                                      tokenize=False, add_generation_prompt=True, **kw)
    except Exception as e:
        say('%s: apply 失败 %s' % (tag, e)); continue
    say('%s 末尾 = %r' % (tag, txt[-60:]))
    ids = tok(txt, add_special_tokens=False)['input_ids']
    t = torch.tensor([ids]).cuda()
    with torch.no_grad():
        lg = m(input_ids=t).logits[0,-1,:].float()
    pr = torch.softmax(lg,-1)
    top = pr.topk(5)
    items = ['%r:%.3f' % (tok.decode([i]), v) for v,i in zip(top.values.tolist(), top.indices.tolist())]
    say('  top5: %s' % ' '.join(items))
# GLM 模板对照（仅文本）
from transformers import AutoTokenizer as AT
tg = AT.from_pretrained('D:/develop/model/ZhipuAI/glm-4-9b-chat-hf')
txt = tg.apply_chat_template([{'role':'user','content':raw}], tokenize=False, add_generation_prompt=True)
say('GLM 模板全文 = %r' % txt)
say('DONE')
