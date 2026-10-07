# -*- coding: utf-8 -*-
"""
N2g: 临界层筛选 + 跨模板存活率
================================
先按 short 模板扫出 |Δscore| 最大的若干 attention 层，再在 short/mid/long 三个模板分别测，
回答："哪些层是上下文无关的必要件？"
"""
import os, sys, time
import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b'
TOPN = int(sys.argv[2]) if len(sys.argv) > 2 else 6
MDIR = os.path.join(ROOT, 'models', 'hf', MODEL)
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'n2g_report_%s.txt' % MODEL.replace('/', '_'))
lines = []
def w(s=''):
    lines.append(str(s)); print(s); sys.stdout.flush()

from transformers import AutoTokenizer, AutoModelForCausalLM
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16,
                                            trust_remote_code=True).to('cuda').eval()
_core = getattr(model.model, 'language_model', model.model)
layers = _core.layers; L = len(layers); norm = _core.norm; head = model.lm_head
def ids_of(s): return tok.encode(s, add_special_tokens=False)
def attn_out_proj(layer):
    a = layer.self_attn
    for nm in ['o_proj', 'dense', 'out_proj']:
        if hasattr(a, nm): return getattr(a, nm)
    raise RuntimeError('no attn out proj')
ATTN = [attn_out_proj(layers[i]) for i in range(L)]
MLP = [layers[i].mlp for i in range(L)]

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
         for wd in ms if len(ids_of(wd)) == 1][:24]
TMPL = {'short': '%s是一种', 'mid': '我说的这个%s是一种',
        'long': '我昨天在超市买了一个%s，它其实是一种'}
w('=== N2g 临界层筛选 ===')
w('time %s model=%s L=%d tie=%s' % (time.strftime('%Y-%m-%d %H:%M:%S'), MODEL, L,
                                     model.config.tie_word_embeddings))
sys.stdout.flush()

def score_rank(v, sup, sid):
    v = v.copy(); v[sid] = -1e9
    own = v[SUP_ID[sup]]
    others = [v[SUP_ID[x]] for x in SUPS if x != sup]
    order = np.argsort(-v)
    return float(own - np.mean(others)), int(np.where(order == SUP_ID[sup])[0][0]) + 1

@torch.no_grad()
def fwd(text, kind=None, l=None):
    ii = torch.tensor([ids_of(text)], device='cuda')
    hs = []
    if kind == 'attn':
        def hook(mod, args):
            y = args[0].clone(); y[:, -1, :] = 0
            return (y,) + tuple(args[1:])
        hs.append(ATTN[l].register_forward_pre_hook(hook))
    elif kind == 'mlp':
        def hook2(mod, args, out):
            y = out.clone(); y[:, -1, :] = 0; return y
        hs.append(MLP[l].register_forward_hook(hook2))
    out = model(input_ids=ii)
    for h in hs: h.remove()
    return out.logits[0, -1].float().detach().cpu().numpy()

# 1) short 模板下逐层扫（attn + mlp）
w('')
w('--- 1) short 模板逐层扫描 ---')
DA = np.zeros(L); DM = np.zeros(L); WA = np.zeros(L)
for wd, sup in CASES:
    p = TMPL['short'] % wd; sid = ids_of(wd)[0]
    v0 = fwd(p); s0, _ = score_rank(v0, sup, sid)
    for l in range(L):
        sa, _ = score_rank(fwd(p, 'attn', l), sup, sid)
        sm, _ = score_rank(fwd(p, 'mlp', l), sup, sid)
        DA[l] += sa - s0; DM[l] += sm - s0
        if sa - s0 < -0.5: WA[l] += 1
DA /= len(CASES); DM /= len(CASES); WA /= len(CASES)
oa = np.argsort(DA)[:TOPN]; om = np.argsort(DM)[:TOPN]
w('  attn top%d: %s' % (TOPN, ' '.join('L%d(%+.2f,w%.2f)' % (l, DA[l], WA[l]) for l in oa)))
w('  mlp  top%d: %s' % (TOPN, ' '.join('L%d(%+.2f)' % (l, DM[l]) for l in om)))
w('  attn 全层 max|Δ|=%.2f ; 次大=%.2f ; 第三=%.2f' %
  (abs(DA).max(), np.sort(np.abs(DA))[::-1][1], np.sort(np.abs(DA))[::-1][2]))
sys.stdout.flush()

# 2) 跨模板存活
w('')
w('--- 2) 跨模板 Δscore（同一部件）---')
w('part      | short   | mid     | long    | 存活(三模板均<-0.3)')
for kind, ls in [('attn', list(oa)), ('mlp', list(om))]:
    for l in ls:
        vals = []
        for tag in ['short', 'mid', 'long']:
            acc = []
            for wd, sup in CASES:
                p = TMPL[tag] % wd; sid = ids_of(wd)[0]
                v0 = fwd(p); s0, _ = score_rank(v0, sup, sid)
                s, _ = score_rank(fwd(p, kind, l), sup, sid)
                acc.append(s - s0)
            vals.append(np.mean(acc))
        surv = all(v < -0.3 for v in vals)
        w('%-4s L%-3d | %+7.3f | %+7.3f | %+7.3f | %s' %
          (kind, l, vals[0], vals[1], vals[2], 'YES' if surv else 'no'))
        sys.stdout.flush()

# 3) 末层 attn 专项（跨模板一致性的唯一候选）
w('')
w('--- 3) 末层 attention 专项 ---')
for l in [L - 1, L - 2, L - 3, L - 4]:
    vals = []
    for tag in ['short', 'mid', 'long']:
        acc = []
        for wd, sup in CASES:
            p = TMPL[tag] % wd; sid = ids_of(wd)[0]
            v0 = fwd(p); s0, _ = score_rank(v0, sup, sid)
            s, _ = score_rank(fwd(p, 'attn', l), sup, sid)
            acc.append(s - s0)
        vals.append(np.mean(acc))
    w('  L%-3d | short %+.3f | mid %+.3f | long %+.3f | min|Δ|=%.3f' %
      (l, vals[0], vals[1], vals[2], min(abs(v) for v in vals)))
sys.stdout.flush()

w('')
w('total %.1fs' % (time.time() - t0))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', OUT)
