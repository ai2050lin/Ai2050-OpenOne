# -*- coding: utf-8 -*-
import os, pathlib
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
F = pathlib.Path('tests/gpt5_temp/q8d3_gdiag.txt'); F.write_text('', encoding='utf-8')
def say(*a):
    s=' '.join(str(x) for x in a); print(s, flush=True)
    with F.open('a', encoding='utf-8', newline='') as f: f.write(s+'\n')
P='D:/develop/model/ZhipuAI/glm-4-9b-chat-hf'
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained(P)
m=AutoModelForCausalLM.from_pretrained(P, dtype=torch.bfloat16, device_map='cuda:0',
      quantization_config=__import__('transformers').BitsAndBytesConfig(load_in_8bit=True,
      llm_int8_skip_modules=['lm_head','embed_tokens'])).eval()
raw='他把苹果'
base = tok.apply_chat_template([{'role':'user','content':raw}], tokenize=False, add_generation_prompt=True)
say('模板 = %r' % base)
for tag, txt in (('模板原样', base), ('模板+\\n', base+'\n')):
    ids = tok(txt, add_special_tokens=False)['input_ids']
    say('%s 末3 token = %s' % (tag, [tok.decode([i]) for i in ids[-3:]]))
    t = torch.tensor([ids]).cuda()
    with torch.no_grad():
        lg = m(input_ids=t).logits[0,-1,:].float()
    pr = torch.softmax(lg,-1); top = pr.topk(5)
    say('  top5: %s' % ' '.join('%r:%.3f' % (tok.decode([i]), v) for v,i in zip(top.values.tolist(), top.indices.tolist())))
say('DONE')
