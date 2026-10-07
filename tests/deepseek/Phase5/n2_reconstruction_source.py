# -*- coding: utf-8 -*-
"""
N2: cloze_is 条件下 L24-35 "重建源"定位
============================================
背景（N1c）：`{W}是一种` 提示下，末位真上位词 rank 由 L23 的 ~509 单调爬到 L31-33 的 1。
          层是"按需重建"，但重建由哪些部件完成未知。

方法：零消融（zero ablation）@ 末位置（readout position）
  Stage A  逐层：分别把 layer ℓ 的 (a) 全部 attention 输出、(b) MLP 输出 在末位置零
           -> Δscore_l = score(消融后) - score(基线)，读出用最终层 logits
  Stage B  逐头：在 |Δ| 最大的 K 层内，逐个 head 消融（attn 输出切片置零）
  Stage C  噪声对照：注入与 attn 输出同范数的随机向量，建立"扰动地板"

score = logit(真上位) - mean(logit(其它上位))，rank = 真上位在词表内的名次
判据（预注册）：最强制单头 share = max|Δ_h| / Σ_h |Δ_h|；≤0.30 -> 降级"分布化重构"
"""
import os, sys, time, json
import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b'
TOPK_LAYERS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SEED = 20261001
MDIR = os.path.join(ROOT, 'models', 'hf', MODEL)
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'n2_report_%s.txt' % MODEL.replace('/', '_'))
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
L = len(layers)
norm = _core.norm
head = model.lm_head

def ids_of(s): return tok.encode(s, add_special_tokens=False)

# ---- resolve attention output projection & mlp ----
def attn_out_proj(layer):
    a = layer.self_attn
    for nm in ['o_proj', 'dense', 'out_proj']:
        if hasattr(a, nm):
            return getattr(a, nm)
    raise RuntimeError('no attn out proj: ' + str([n for n, _ in a.named_children()]))

ATTN = [attn_out_proj(layers[i]) for i in range(L)]
MLP = [layers[i].mlp for i in range(L)]

n_heads = model.config.num_attention_heads
o_in = ATTN[0].in_features
assert o_in % n_heads == 0, (o_in, n_heads)
HD = o_in // n_heads

w('=== N2 重建源定位（逐层 + 逐头归因）===')
w('time %s  model=%s  layers=%d  heads=%d  head_dim=%d  o_in=%d  tie=%s' %
  (time.strftime('%Y-%m-%d %H:%M:%S'), MODEL, L, n_heads, HD, o_in,
   model.config.tie_word_embeddings))
sys.stdout.flush()

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
w('supers: %s' % ' '.join(SUPS))

CASES = []   # (word, sup)
for sup, members in GROUPS.items():
    if sup not in SUP_ID: continue
    for wd in members:
        if len(ids_of(wd)) == 1:
            CASES.append((wd, sup))
w('cases n=%d' % len(CASES))

CASES = CASES[:24]          # 控制预算：24 例
w('cases used n=%d (subset for budget)' % len(CASES))
sys.stdout.flush()


def score_rank(v, sup, selfid):
    v = v.copy()
    if selfid is not None: v[selfid] = -1e9
    own = v[SUP_ID[sup]]
    others = [v[SUP_ID[x]] for x in SUPS if x != sup]
    sc = float(own - np.mean(others))
    order = np.argsort(-v)
    rk = int(np.where(order == SUP_ID[sup])[0][0]) + 1
    return sc, rk


@torch.no_grad()
def fwd_logits(text, specs=()):
    """specs: list of (kind, layer, mask_or_None)"""
    ii = torch.tensor([ids_of(text)], device='cuda')
    handles = []
    for kind, l, mask in specs:
        if kind == 'attn':
            def mk(mask):
                def hook(mod, args):
                    y = args[0].clone()
                    if mask is None:
                        y[:, -1, :] = 0
                    else:
                        y[:, -1, :] = y[:, -1, :] * mask.view(1, -1).to(y.dtype)
                    return (y,) + tuple(args[1:])
                return hook
            handles.append(ATTN[l].register_forward_pre_hook(mk(mask)))
        elif kind == 'mlp':
            def hook(mod, args, out):
                y = out.clone(); y[:, -1, :] = 0; return y
            handles.append(MLP[l].register_forward_hook(hook))
    out = model(input_ids=ii)
    for h in handles: h.remove()
    return out.logits[0, -1].float().detach().cpu().numpy()


@torch.no_grad()
def capture_norms(text):
    """记录每层 attn/mlp 在末位的输出范数"""
    ii = torch.tensor([ids_of(text)], device='cuda')
    an = np.zeros(L); mn = np.zeros(L); handles = []
    def mk(i, arr):
        def hook(mod, args, out):
            arr[i] = float(out[0, -1].float().norm())
        return hook
    for i in range(L):
        handles.append(ATTN[i].register_forward_hook(mk(i, an)))
        handles.append(MLP[i].register_forward_hook(mk(i, mn)))
    model(input_ids=ii)
    for h in handles: h.remove()
    return an, mn


# ================= Stage 0: 基线轨迹（logit lens） =================
w('')
w('--- Stage 0: 基线 logit-lens 轨迹 (n=%d, prompt="%s是一种") ---' % (len(CASES), CASES[0][0]))
base_s, base_r = [], []
for wd, sup in CASES:
    ii = torch.tensor([ids_of('%s是一种' % wd)], device='cuda')
    out = model(input_ids=ii, output_hidden_states=True)
    H = out.hidden_states
    sc_l, rk_l = [], []
    for l in range(L + 1):
        h = H[l][0, -1].to(torch.bfloat16)
        with torch.no_grad():
            v = head(norm(h.unsqueeze(0))).float().detach()[0].cpu().numpy()
        s, r = score_rank(v, sup, ids_of(wd)[0] if len(ids_of(wd)) == 1 else None)
        sc_l.append(s); rk_l.append(r)
    base_s.append(sc_l); base_r.append(rk_l)
BASE_S = np.mean(base_s, 0); BASE_R = np.median(base_r, 0)
w('L   score      rank')
for l in range(L + 1):
    w('%2d  %+8.3f  %7.0f' % (l, BASE_S[l], BASE_R[l]))
w('  best rank=%d @L%d ; score peak @L%d=%.3f' %
  (int(BASE_R.min()), int(np.argmin(BASE_R)), int(np.argmax(BASE_S)), BASE_S.max()))
sys.stdout.flush()

# ================= Stage A: 逐层消融 =================
w('')
w('--- Stage A: 逐层零消融 @末位置 (Δscore, Δrank, 部件范数) ---')
w('L    dScore_attn  dRank_attn   ||attn||   dScore_mlp  dRank_mlp    ||mlp||')
A_attn = np.zeros(L); A_mlp = np.zeros(L); AR_attn = np.zeros(L); AR_mlp = np.zeros(L)
N_attn = np.zeros(L); N_mlp = np.zeros(L)
for wd, sup in CASES:
    p = '%s是一种' % wd
    sid = ids_of(wd)[0]
    v0 = fwd_logits(p)
    s0, r0 = score_rank(v0, sup, sid)
    an, mn = capture_norms(p)
    N_attn += an; N_mlp += mn
    for l in range(L):
        va = fwd_logits(p, [('attn', l, None)])
        sa, ra = score_rank(va, sup, sid)
        A_attn[l] += sa - s0; AR_attn[l] += ra - r0
        vm = fwd_logits(p, [('mlp', l, None)])
        sm, rm = score_rank(vm, sup, sid)
        A_mlp[l] += sm - s0; AR_mlp[l] += rm - r0
A_attn /= len(CASES); A_mlp /= len(CASES); AR_attn /= len(CASES); AR_mlp /= len(CASES)
N_attn /= len(CASES); N_mlp /= len(CASES)
for l in range(L):
    w('%2d  %+10.3f  %+10.1f  %9.3f   %+10.3f  %+10.1f  %9.3f' %
      (l, A_attn[l], AR_attn[l], N_attn[l], A_mlp[l], AR_mlp[l], N_mlp[l]))
attn_rank = np.argsort(A_attn)[:8]
mlp_rank = np.argsort(A_mlp)[:8]
w('  attn 最强 8 层(最负=必要): %s' % ' '.join('L%d(%.2f)' % (l, A_attn[l]) for l in attn_rank))
w('  mlp  最强 8 层(最负=必要): %s' % ' '.join('L%d(%.2f)' % (l, A_mlp[l]) for l in mlp_rank))
w('  attn 前 5 层绝对贡献和=%.2f ; mlp 前 5 层绝对贡献和=%.2f' %
  (np.sort(np.abs(A_attn))[::-1][:5].sum(), np.sort(np.abs(A_mlp))[::-1][:5].sum()))
sys.stdout.flush()

# ================= Stage B: 逐头消融 =================
layers_pick = sorted(set(list(np.argsort(A_attn)[:TOPK_LAYERS]) + list(np.argsort(A_mlp)[:TOPK_LAYERS])))
w('')
w('--- Stage B: 逐头消融（层集合 = %s）---' % ' '.join('L%d' % l for l in layers_pick))
w('每个 head 在末位单独置零；share = |Δ_h| / Σ_h|Δ_h| (层内)')
HEAD_D = {}
for l in layers_pick:
    acc = np.zeros(n_heads); accr = np.zeros(n_heads)
    for wd, sup in CASES:
        p = '%s是一种' % wd
        sid = ids_of(wd)[0]
        v0 = fwd_logits(p); s0, r0 = score_rank(v0, sup, sid)
        for hh in range(n_heads):
            mask = torch.ones(o_in, device='cuda')
            mask[hh * HD:(hh + 1) * HD] = 0
            v = fwd_logits(p, [('attn', l, mask)])
            s, r = score_rank(v, sup, sid)
            acc[hh] += s - s0; accr[hh] += r - r0
    acc /= len(CASES); accr /= len(CASES)
    HEAD_D[l] = (acc, accr)
    tot = np.abs(acc).sum()
    order = np.argsort(acc)[:6]
    w('')
    w('L%d: Σ|Δ|=%.3f  max|Δ_h|=%.3f  share_max=%.3f  %s' %
      (l, tot, np.abs(acc).max(), (np.abs(acc).max() / tot) if tot > 0 else -1,
       'DISTRIBUTED' if (tot > 0 and np.abs(acc).max() / tot <= 0.30) else 'CONCENTRATED'))
    w('    top-6 (最负=最必要): %s' % ' '.join('#%d(%+.2f,r%+.0f)' % (i, acc[i], accr[i]) for i in order))
    w('    top-6 (最正=最重要, 去掉反而更好): %s' %
      ' '.join('#%d(%+.2f)' % (i, acc[i]) for i in np.argsort(acc)[::-1][:6]))
sys.stdout.flush()

# ================= Stage C: 噪声对照 =================
w('')
w('--- Stage C: 噪声对照（同范数随机扰动 @末位置, n=%d）---' % len(CASES))
rng = np.random.default_rng(SEED)
NOISE = np.zeros(L); NOISE_R = np.zeros(L)
for wd, sup in CASES:
    p = '%s是一种' % wd
    sid = ids_of(wd)[0]
    v0 = fwd_logits(p); s0, r0 = score_rank(v0, sup, sid)
    for l in range(L):
        nrm = N_attn[l]
        noise = rng.standard_normal(o_in) * (nrm / np.sqrt(o_in))
        nvec = torch.tensor(noise, device='cuda', dtype=torch.bfloat16)
        handles = []
        def hook(mod, args, _n=nvec):
            y = args[0].clone(); y[:, -1, :] = y[:, -1, :] + _n.view(1, -1).to(y.dtype)
            return (y,) + tuple(args[1:])
        handles.append(ATTN[l].register_forward_pre_hook(hook))
        ii = torch.tensor([ids_of(p)], device='cuda')
        out = model(input_ids=ii)
        for h in handles: h.remove()
        v = out.logits[0, -1].float().detach().cpu().numpy()
        s, r = score_rank(v, sup, sid)
        NOISE[l] += s - s0; NOISE_R[l] += r - r0
NOISE /= len(CASES); NOISE_R /= len(CASES)
nmax = np.abs(NOISE).max()
w('  噪声造成的 |Δscore| 最大 = %.3f @L%d ; 中位 = %.3f' % (nmax, int(np.argmax(np.abs(NOISE))), float(np.median(np.abs(NOISE)))))
w('  噪声造成的 |Δrank| 最大 = %.1f' % float(np.max(np.abs(NOISE_R))))
w('  => 真实消融效应需超过该地板才算有效')

# ================= 汇总 =================
w('')
w('--- 汇总 ---')
allh = np.concatenate([HEAD_D[l][0] for l in layers_pick])
w('全部被测 head (n=%d): Σ|Δ|=%.3f  max|Δ_h|=%.3f  全局 share_max=%.3f' %
  (len(allh), np.abs(allh).sum(), np.abs(allh).max(),
   np.abs(allh).max() / np.abs(allh).sum()))
w('单头 share 判据 (≤0.30 -> DISTRIBUTED): %s' %
  ('DISTRIBUTED' if np.abs(allh).max() / np.abs(allh).sum() <= 0.30 else 'CONCENTRATED'))
w('attn 单层最大贡献 L%d=%.3f ; mlp 单层最大贡献 L%d=%.3f' %
  (int(np.argmax(np.abs(A_attn))), A_attn[np.argmax(np.abs(A_attn))],
   int(np.argmax(np.abs(A_mlp))), A_mlp[np.argmax(np.abs(A_mlp))]))
w('')
w('total %.1fs' % (time.time() - t0))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', OUT)
