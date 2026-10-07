# -*- coding: utf-8 -*-
"""N2f: 行为端诊断 —— 三个模板下模型的实际 next-token 分布（确认任务是否真被执行）"""
import os, sys, time
import numpy as np
import torch
ROOT = r'D:\AI2050\Ai2050-OpenOne'
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b'
MDIR = os.path.join(ROOT, 'models', 'hf', MODEL)
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'n2f_report_%s.txt' % MODEL.replace('/', '_'))
lines = []
def w(s=''):
    lines.append(str(s)); print(s); sys.stdout.flush()
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16,
                                            trust_remote_code=True).to('cuda').eval()
def ids_of(s): return tok.encode(s, add_special_tokens=False)
SUP = {'水果': '苹果', '动物': '狗', '交通工具': '汽车', '家具': '桌子', '金属': '铁', '颜色': '红'}
TMPL = {'short': '%s是一种', 'mid': '我说的这个%s是一种', 'long': '我昨天在超市买了一个%s，它其实是一种'}
w('=== N2f 行为端诊断 ===')
w('time %s model=%s' % (time.strftime('%Y-%m-%d %H:%M:%S'), MODEL))
for tag, tmpl in TMPL.items():
    w('')
    w('--- %s ---' % tag)
    for sup, wd in SUP.items():
        p = tmpl % wd
        ii = torch.tensor([ids_of(p)], device='cuda')
        with torch.no_grad():
            z = model(input_ids=ii).logits[0, -1].float()
        lg = z.detach().cpu().numpy()
        top = np.argsort(-lg)[:6]
        tt = tok.convert_ids_to_tokens([int(i) for i in top])
        sid = ids_of(sup)[0]
        pr = float(torch.softmax(z, -1)[sid])
        rk = int((lg > lg[sid]).sum()) + 1
        w('  %-4s -> %-6s top6=%s | P(%s)=%.3f rank=%d' %
          (wd, sup, ' '.join('%s(%.1f)' % (t, lg[i]) for t, i in zip(tt, top)), sup, pr, rk))
        sys.stdout.flush()
w('')
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', OUT)
