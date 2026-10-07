# -*- coding: utf-8 -*-
"""
N1b: 反嵌入段 —— 层次读出探针
问：第 l 层，从实例 W 的状态 h_l(W) 直接经 unembed 读出"它的上位词"，
    成功率随层如何变化？与 L0（纯词嵌入）比是增强还是衰减？
指标 score_l(W) = logit_l(s_true|h_l(W)) - mean_{s!=s_true} logit_l(s|h_l(W))
"""
import os, sys, time
import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b'
MDIR = os.path.join(ROOT, 'models', 'hf', MODEL)
TAG = MODEL.replace('/', '_')
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'n1b_report_%s.txt' % TAG)
lines = []
def w(s=''): lines.append(str(s)); print(s); sys.stdout.flush()

from transformers import AutoTokenizer, AutoModelForCausalLM
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16,
                                            trust_remote_code=True).to('cuda').eval()
_core = getattr(model.model, 'language_model', model.model)
L = len(_core.layers)
norm = _core.norm; head = model.lm_head
NV = int(model.get_input_embeddings().weight.shape[0])
w('=== N1b 反嵌入段：层次读出 ===')
w('time %s model=%s layers=%d vocab=%d' % (time.strftime('%Y-%m-%d %H:%M:%S'), MODEL, L, NV))

def ids_of(s): return tok.encode(s, add_special_tokens=False)

GROUPS = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '樱桃', '柠檬', '橘子'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛', '羊', '鱼'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅', '钢'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白', '紫', '灰'],
}
SUP_ID = {}
for s in GROUPS:
    t = ids_of(s)
    if len(t) == 1: SUP_ID[s] = t[0]
SUPS = [s for s in GROUPS if s in SUP_ID]
w('  superordinates(single-token)=%s' % ' '.join(SUPS))
if len(SUPS) < 3:
    w('  ABORT: too few superordinates'); open(OUT, 'w', encoding='utf-8').write('\n'.join(lines)); sys.exit(1)

CTXS = ['这是%s。', '他说的%s。', '那个%s。']

@torch.no_grad()
def hidden_at(text, pos):
    ii = torch.tensor([ids_of(text)], device='cuda')
    out = model(input_ids=ii, output_hidden_states=True)
    return np.stack([h[0][pos].float().cpu().numpy() for h in out.hidden_states], 0)

def logits_at(hvec_l):
    with torch.no_grad():
        z = head(norm(torch.tensor(hvec_l, device='cuda').to(torch.bfloat16))).float()
        return z.detach().cpu().numpy()

score_inst = {s: [] for s in SUPS}
for sup in SUPS:
    mem = [x for x in GROUPS[sup] if len(ids_of(x)) == 1]
    if len(mem) < 4: continue
    for wd in mem:
        acc = []
        for c in CTXS:
            p = len(ids_of(c.split('%s')[0]))
            acc.append(hidden_at(c % wd, p))
        H = np.mean(acc, 0)
        row = []
        for l in range(L + 1):
            lg = logits_at(H[l])
            own = lg[SUP_ID[sup]]
            others = [lg[SUP_ID[s]] for s in SUPS if s != sup]
            row.append(float(own - np.mean(others)))
        score_inst[sup].append(np.array(row))

rng = np.random.default_rng(3)
rand_rows = []
with torch.no_grad():
    for _ in range(24):
        rid = int(rng.integers(0, NV))
        out = model(input_ids=torch.tensor([[rid]], device='cuda'), output_hidden_states=True)
        H = np.stack([h[0][0].float().cpu().numpy() for h in out.hidden_states], 0)
        sup = SUPS[int(rng.integers(0, len(SUPS)))]
        row = []
        for l in range(L + 1):
            lg = logits_at(H[l])
            row.append(float(lg[SUP_ID[sup]] - np.mean([lg[SUP_ID[s]] for s in SUPS if s != sup])))
        rand_rows.append(np.array(row))

valid = {s: v for s, v in score_inst.items() if v}
allinst = np.mean([np.mean(v, 0) for v in valid.values()], 0)
ninst = sum(len(v) for v in valid.values())
randm = np.mean(rand_rows, 0)
gap = allinst - randm
w('  instances n=%d over %d supers' % (ninst, len(valid)))
w('L   score_inst  score_rand    gap')
for l in range(L + 1):
    w('%2d   %+9.3f   %+9.3f   %+8.3f' % (l, allinst[l], randm[l], gap[l]))
w('  score_inst L0=%.3f peak @L%d=%.3f ratio=%.2fx' %
  (allinst[0], int(np.argmax(allinst)), allinst.max(), allinst.max() / max(abs(allinst[0]), 1e-9)))
w('  gap L0=%.3f peak @L%d=%.3f' % (gap[0], int(np.argmax(gap)), gap.max()))
w('  per-super peak: ' + ', '.join('%s@L%d' % (s, int(np.argmax(np.mean(v, 0)))) for s, v in valid.items()))
w('  total %.1fs' % (time.time() - t0))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', OUT)
