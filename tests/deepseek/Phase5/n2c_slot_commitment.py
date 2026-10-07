# -*- coding: utf-8 -*-
"""
N2c: 读位槽的"类别归属承诺曲线" + L6 注意力落点
=================================================
动机：N2/N2b 定位到 L6 attention 是 cloze is-a 读数的早期瓶颈（Δ−6.01，任务特异 23×）。
     若 L6 的职责是"把主语类别写入读位槽"，则可做供体移植实验直接验证。

Part 1  注意力落点：L6 在末位对各前序位置的注意力质量（cloze vs neutral）
Part 2  读位槽移植：把供体句 (donor) 在 layer ℓ 输出的**末位隐状态**移植到受体句 (recipient) 的同一位置，
        看受体句最终的类别分数何时被供体"接管"。
        指标：dRecip = score_recip(patch) - score_recip(base)
              dDonor = score_donor(patch) - score_donor(base_rec)
        若某层后 dDonor 由 ~0 跃升 -> 该层即"槽归属切换点"。
"""
import os, sys, time
import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b'
MDIR = os.path.join(ROOT, 'models', 'hf', MODEL)
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'n2c_report_%s.txt' % MODEL.replace('/', '_'))
lines = []
def w(s=''):
    lines.append(str(s)); print(s); sys.stdout.flush()

from transformers import AutoTokenizer, AutoModelForCausalLM
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16,
                                            trust_remote_code=True).to('cuda').eval()
_core = getattr(model.model, 'language_model', model.model)
layers = _core.layers
L = len(layers); norm = _core.norm; head = model.lm_head
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

w('=== N2c 读位槽承诺曲线 + L6 注意力落点 ===')
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
def fwd(text, patch=None):
    ii = torch.tensor([ids_of(text)], device='cuda')
    hs = []
    if patch is not None:
        l, vec = patch
        def hook(mod, args, out):
            if isinstance(out, tuple):
                h = out[0].clone(); h[0, -1, :] = vec.to(h.dtype)
                return (h,) + tuple(out[1:])
            h = out.clone(); h[0, -1, :] = vec.to(h.dtype); return h
        hs.append(layers[l].register_forward_hook(hook))
    out = model(input_ids=ii)
    for h in hs: h.remove()
    return out.logits[0, -1].float().detach().cpu().numpy()

@torch.no_grad()
def hidden_all(text):
    ii = torch.tensor([ids_of(text)], device='cuda')
    out = model(input_ids=ii, output_hidden_states=True)
    return np.stack([h[0, -1].float().detach().cpu().numpy() for h in out.hidden_states], 0)

# ---------------- Part 1: 已废弃 ----------------
# 注意：v1 此处用 forward_hook 抓 self_attn 输出，拿到的是 attn_output 而非注意力权重（无效量）。
# 正确的注意力落点测量改由 n2d_attn_write.py 以 attn_implementation='eager' + output_attentions 完成。
w('')
w('--- Part 1: (已废弃，见 n2d_attn_write.py Q1 的 eager 版本) ---')
sys.stdout.flush()

# ---------------- Part 2: 读位槽移植 ----------------
w('')
w('--- Part 2: 读位槽归属移植（供体末位隐状态 -> 受体读位）---')
PAIRS = [
    ('苹果', '水果', '桌子', '家具'), ('桌子', '家具', '苹果', '水果'),
    ('狗', '动物', '汽车', '交通工具'), ('汽车', '交通工具', '狗', '动物'),
    ('铁', '金属', '红', '颜色'), ('红', '颜色', '铁', '金属'),
    ('香蕉', '水果', '沙发', '家具'), ('沙发', '家具', '香蕉', '水果'),
]
PATCH_L = [1, 3, 5, 6, 7, 9, 12, 15, 17, 20, 24, 27, 30, 33, 35]
w('donor->recipient  pairs n=%d ; patch layers=%s' % (len(PAIRS), PATCH_L))
HB = {}
for rw, rs, dw, ds in PAIRS:
    HB[rw] = hidden_all('%s是一种' % rw)
    HB[dw] = hidden_all('%s是一种' % dw)
w('L    dRecip     dDonor    (供体类别分数变化 / 受体类别分数变化)')
curve = {}
for l in PATCH_L:
    dr_l = []; dd_l = []
    for rw, rs, dw, ds in PAIRS:
        pr = '%s是一种' % rw
        sid_r = ids_of(rw)[0]; sid_d = ids_of(dw)[0]
        v0 = fwd(pr); sr0, _ = score_rank(v0, rs, sid_r); sd0, _ = score_rank(v0, ds, sid_d)
        vec = torch.tensor(HB[dw][l + 1], device='cuda')   # donor 在 layer l 输出末位
        v1 = fwd(pr, patch=(l, vec))
        sr1, _ = score_rank(v1, rs, sid_r); sd1, _ = score_rank(v1, ds, sid_d)
        dr_l.append(sr1 - sr0); dd_l.append(sd1 - sd0)
    curve[l] = (np.mean(dr_l), np.mean(dd_l))
    w('%2d   %+8.3f   %+8.3f' % (l, np.mean(dr_l), np.mean(dd_l)))
sw = [l for l in PATCH_L if curve[l][1] > 0.5 * abs(curve[1][0]) + 1e-9]
w('')
w('  交叉点：dDonor 首次显著升起（> 受体基线分数一半）@L%s' % (sw[0] if sw else 'not found'))
w('  L6 前 dDonor=%.3f ; L6 后 dDonor=%.3f ; 末层 dDonor=%.3f' %
  (curve.get(5, (0, 0))[1], curve.get(7, (0, 0))[1], curve[35][1]))
w('  L6 前 dRecip=%.3f ; L6 后 dRecip=%.3f ; 末层 dRecip=%.3f' %
  (curve.get(5, (0, 0))[0], curve.get(7, (0, 0))[0], curve[35][0]))
sys.stdout.flush()

w('')
w('total %.1fs' % (time.time() - t0))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', OUT)
