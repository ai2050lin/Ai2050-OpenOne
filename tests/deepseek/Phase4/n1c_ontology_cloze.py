# -*- coding: utf-8 -*-
"""
N1c: is-a 关系的"任务依赖"检验
同一批实例词，四类提示：
  neutral  : 这是{W}          （无任务压力）
  cloze_is : {W}是一种        （强制要求上位词）
  cloze_and: {W}和{co}都是    （同级联合 -> 上位词）
  cloze_use: {W}可以吃，它是  （功能 -> 上位词）
每层做 logit lens，测 score_l = logit(真上位) - mean(logit(其他上位)) 与真上位 rank。
判据：若 cloze 类提示的峰值显著高于 neutral，且峰更深 -> 层是"按需计算"is-a，而非"不计算"。
"""
import os, sys, time
import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b'
MDIR = os.path.join(ROOT, 'models', 'hf', MODEL)
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'n1c_report_%s.txt' % MODEL.replace('/', '_'))
lines = []
def w(s=''): lines.append(str(s)); print(s); sys.stdout.flush()

from transformers import AutoTokenizer, AutoModelForCausalLM
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16,
                                            trust_remote_code=True).to('cuda').eval()
_core = getattr(model.model, 'language_model', model.model)
L = len(_core.layers); norm = _core.norm; head = model.lm_head
def ids_of(s): return tok.encode(s, add_special_tokens=False)

GROUPS = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
SUP_ID = {}
for s in GROUPS:
    t = ids_of(s)
    if len(t) == 1: SUP_ID[s] = t[0]
SUPS = [s for s in GROUPS if s in SUP_ID]
w('=== N1c is-a 任务依赖 ===')
w('time %s model=%s layers=%d supers=%s' % (time.strftime('%Y-%m-%d %H:%M:%S'), MODEL, L, ' '.join(SUPS)))

@torch.no_grad()
def runh(text):
    ii = torch.tensor([ids_of(text)], device='cuda')
    out = model(input_ids=ii, output_hidden_states=True)
    return np.stack([h[0][-1].float().cpu().numpy() for h in out.hidden_states], 0)

def lg(h):
    with torch.no_grad():
        z = head(norm(torch.tensor(h, device='cuda').to(torch.bfloat16))).float()
        return z.detach().cpu().numpy()

def score(H, sup, selfid=None):
    s, r = [], []
    for l in range(L + 1):
        v = lg(H[l]).copy()
        if selfid is not None: v[selfid] = -1e9
        own = v[SUP_ID[sup]]
        others = [v[SUP_ID[x]] for x in SUPS if x != sup]
        s.append(float(own - np.mean(others)))
        order = np.argsort(-v)
        r.append(int(np.where(order == SUP_ID[sup])[0][0]) + 1)
    return np.array(s), np.array(r)

RES = {}
for sup, members in GROUPS.items():
    if sup not in SUP_ID: continue
    mem = [x for x in members if len(ids_of(x)) == 1]
    if len(mem) < 4: continue
    for i, wd in enumerate(mem):
        co = mem[(i + 1) % len(mem)]
        prompts = {
            'neutral':   '这是%s' % wd,
            'cloze_is':  '%s是一种' % wd,
            'cloze_and': '%s和%s都是' % (wd, co),
            'cloze_use': '%s可以吃，这东西其实是一种' % wd,
        }
        for k, p in prompts.items():
            H = runh(p)
            s, r = score(H, sup, selfid=ids_of(wd)[0] if len(ids_of(wd)) == 1 else None)
            RES.setdefault(k, []).append((s, r))

for k in ['neutral', 'cloze_is', 'cloze_and', 'cloze_use']:
    if k not in RES: continue
    S = np.mean([x[0] for x in RES[k]], 0)
    R = np.median([x[1] for x in RES[k]], 0)
    w('')
    w('--- %s  (n=%d) ---' % (k, len(RES[k])))
    w('L   score    rank')
    for l in range(L + 1):
        w('%2d  %+7.3f   %6.0f' % (l, S[l], R[l]))
    w('  L0=%.3f  peak @L%d=%.3f  ratio=%.2fx  best_rank=%d @L%d' %
      (S[0], int(np.argmax(S)), S.max(), S.max() / max(abs(S[0]), 1e-9),
       int(R.min()), int(np.argmin(R))))
w('')
w('total %.1fs' % (time.time() - t0))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', OUT)
