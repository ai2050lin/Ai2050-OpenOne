# -*- coding: utf-8 -*-
"""
N2d: L6 写入口的注意力落点 + 距离不变性
=========================================
N2c 证明读位槽在 L6 一次性锁死（L5 移植 dDonor +0.97 -> L6 移植 +10.11，之后恒定）。
本脚本回答两问：
  Q1  L6 在末位的注意力落在哪个位置？（需 eager 才能取权重）
  Q2  L6 是否为"距离无关"的取主语层？——把主语推远，看 (a) 落点是否仍指向主语，(b) L6 消融是否仍致命。
"""
import os, sys, time
import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b'
MDIR = os.path.join(ROOT, 'models', 'hf', MODEL)
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'n2d_report_%s.txt' % MODEL.replace('/', '_'))
lines = []
def w(s=''):
    lines.append(str(s)); print(s); sys.stdout.flush()

from transformers import AutoTokenizer, AutoModelForCausalLM
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16, trust_remote_code=True,
                                            attn_implementation='eager').to('cuda').eval()
_core = getattr(model.model, 'language_model', model.model)
layers = _core.layers; L = len(layers); norm = _core.norm; head = model.lm_head
def ids_of(s): return tok.encode(s, add_special_tokens=False)
def attn_out_proj(layer):
    a = layer.self_attn
    for nm in ['o_proj', 'dense', 'out_proj']:
        if hasattr(a, nm): return getattr(a, nm)
    raise RuntimeError('no attn out proj')
ATTN = [attn_out_proj(layers[i]) for i in range(L)]
n_heads = model.config.num_attention_heads
o_in = ATTN[0].in_features

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
         for wd in ms if len(ids_of(wd)) == 1][:12]

w('=== N2d L6 写入口注意力落点 + 距离不变性 ===')
w('time %s model=%s L=%d heads=%d tie=%s' % (time.strftime('%Y-%m-%d %H:%M:%S'), MODEL, L,
                                             n_heads, model.config.tie_word_embeddings))
TMPL = {
    'short': '%s是一种',
    'mid':   '我说的这个%s是一种',
    'long':  '我昨天在超市买了一个%s，它其实是一种',
}
for k, v in TMPL.items():
    w('  %-6s sample=%r -> %d tokens %r' % (k, v % '苹果', len(ids_of(v % '苹果')),
                                            tok.convert_ids_to_tokens(ids_of(v % '苹果'))))
sys.stdout.flush()

def score_rank(v, sup, sid):
    v = v.copy(); v[sid] = -1e9
    own = v[SUP_ID[sup]]
    others = [v[SUP_ID[x]] for x in SUPS if x != sup]
    order = np.argsort(-v)
    return float(own - np.mean(others)), int(np.where(order == SUP_ID[sup])[0][0]) + 1

@torch.no_grad()
def fwd(text, abl_attn_layer=None):
    ii = torch.tensor([ids_of(text)], device='cuda')
    hs = []
    if abl_attn_layer is not None:
        l = abl_attn_layer
        def hook(mod, args):
            y = args[0].clone(); y[:, -1, :] = 0
            return (y,) + tuple(args[1:])
        hs.append(ATTN[l].register_forward_pre_hook(hook))
    out = model(input_ids=ii)
    for h in hs: h.remove()
    return out.logits[0, -1].float().detach().cpu().numpy()

# ---------------- Q1: 注意力落点 ----------------
w('')
w('--- Q1: 末位注意力落点（eager，跨头平均；列出 attend 质量 >0.15 的位置）---')
LAY = list(range(0, min(12, L)))
for tag, tmpl in TMPL.items():
    R = {}
    for wd, sup in CASES:
        p = tmpl % wd
        ii = torch.tensor([ids_of(p)], device='cuda')
        out = model(input_ids=ii, output_attentions=True)
        for l in LAY:
            A = out.attentions[l][0].float().detach().cpu().numpy()   # [heads, T, T]
            R.setdefault(l, []).append(A[:, -1, :].mean(0))
    w('')
    w('  [%s]' % tag)
    for l in LAY:
        M = np.mean(R[l], 0)
        top = np.argsort(-M)[:3]
        w('    L%-3d  %s' % (l, '  '.join('p%d=%.3f' % (i, M[i]) for i in top)))
sys.stdout.flush()

# ---------------- Q2: 距离不变性 ----------------
w('')
w('--- Q2: L6 attention 消融 @末位（不同距离），Δscore ---')
w('%-8s %3s | dScore_atL6 | dScore_atL5 | dScore_atL9 | dScore_atL34' %
  ('tmpl', 'T'))
for tag, tmpl in TMPL.items():
    rows = {5: [], 6: [], 9: [], 34: []}
    for wd, sup in CASES:
        p = tmpl % wd; sid = ids_of(wd)[0]
        v0 = fwd(p); s0, _ = score_rank(v0, sup, sid)
        for l in rows:
            s, _ = score_rank(fwd(p, abl_attn_layer=l), sup, sid)
            rows[l].append(s - s0)
    w('%-8s %3d | %+12.3f | %+12.3f | %+12.3f | %+13.3f' %
      (tag, len(ids_of(tmpl % '苹果')), np.mean(rows[6]), np.mean(rows[5]),
       np.mean(rows[9]), np.mean(rows[34])))
sys.stdout.flush()

# ---------------- Q3: 扫描哪一层是"写层"（逐层消融最大 |Δ|）----------------
w('')
w('--- Q3: 逐层 attention 消融 @末位（short 模板），Δscore 前 10 ---')
if True:
    ds = np.zeros(L)
    for wd, sup in CASES:
        p = '%s是一种' % wd; sid = ids_of(wd)[0]
        v0 = fwd(p); s0, _ = score_rank(v0, sup, sid)
        for l in range(L):
            s, _ = score_rank(fwd(p, abl_attn_layer=l), sup, sid)
            ds[l] += s - s0
    ds /= len(CASES)
    ordr = np.argsort(ds)[:10]
    w('  %s' % '  '.join('L%d(%+.2f)' % (l, ds[l]) for l in ordr))
    w('  最大层 L%d = %+.3f' % (int(np.argmin(ds)), ds.min()))
sys.stdout.flush()

w('')
w('total %.1fs' % (time.time() - t0))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', OUT)
