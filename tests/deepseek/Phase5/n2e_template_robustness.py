# -*- coding: utf-8 -*-
"""
N2e: N1c "L31-33 重建窗" 的模板稳健性
=======================================
N2d 发现 L6 写入口只在 2-token 最小句成立（推远主语后 Δscore ~0）——
说明之前的结论可能被退化短句污染。本脚本检验同一现象：is-a rank 轨迹是否也只在短句成立。
"""
import os, sys, time
import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b'
MDIR = os.path.join(ROOT, 'models', 'hf', MODEL)
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'n2e_report_%s.txt' % MODEL.replace('/', '_'))
lines = []
def w(s=''):
    lines.append(str(s)); print(s); sys.stdout.flush()

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
SUP_ID = {s: ids_of(s)[0] for s in GROUPS if len(ids_of(s)) == 1}
SUPS = list(SUP_ID.keys())
CASES = [(wd, sup) for sup, ms in GROUPS.items() if sup in SUP_ID
         for wd in ms if len(ids_of(wd)) == 1]
w('=== N2e 模板稳健性 ===')
w('time %s model=%s L=%d cases=%d' % (time.strftime('%Y-%m-%d %H:%M:%S'), MODEL, L, len(CASES)))
TMPL = {
    'short(T=2)':  '%s是一种',
    'mid(T=5)':    '我说的这个%s是一种',
    'long(T=11)':  '我昨天在超市买了一个%s，它其实是一种',
}
sys.stdout.flush()

def score_rank(v, sup, sid):
    v = v.copy(); v[sid] = -1e9
    own = v[SUP_ID[sup]]
    others = [v[SUP_ID[x]] for x in SUPS if x != sup]
    order = np.argsort(-v)
    return float(own - np.mean(others)), int(np.where(order == SUP_ID[sup])[0][0]) + 1

for tag, tmpl in TMPL.items():
    S = np.zeros(L + 1); R = np.zeros(L + 1)
    for wd, sup in CASES:
        ii = torch.tensor([ids_of(tmpl % wd)], device='cuda')
        out = model(input_ids=ii, output_hidden_states=True)
        for l in range(L + 1):
            hh = out.hidden_states[l][0, -1].to(torch.bfloat16)
            with torch.no_grad():
                v = head(norm(hh.unsqueeze(0))).float().detach()[0].cpu().numpy()
            s, r = score_rank(v, sup, ids_of(wd)[0])
            S[l] += s; R[l] += r
    S /= len(CASES); R /= len(CASES)
    w('')
    w('--- %s ---' % tag)
    w('L: ' + ' '.join('%d' % l for l in range(L + 1)))
    w('R: ' + ' '.join('%.0f' % R[l] for l in range(L + 1)))
    w('  final(L%d) rank=%.1f score=%+.2f | best rank=%.0f @L%d | score peak %.2f @L%d' %
      (L, R[L], S[L], R.min(), int(np.argmin(R)), S.max(), int(np.argmax(S))))
    w('  rank<=10 首次 @L%d ; rank<=3 首次 @L%d' %
      (int(np.argmax(R <= 10)) if (R <= 10).any() else -1,
       int(np.argmax(R <= 3)) if (R <= 3).any() else -1))
    sys.stdout.flush()

w('')
w('total %.1fs' % (time.time() - t0))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', OUT)
