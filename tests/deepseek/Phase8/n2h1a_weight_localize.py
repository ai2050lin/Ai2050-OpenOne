# -*- coding: utf-8 -*-
"""
Phase 8 / N2h1-alpha : is-a 类别轴「写入算子」的组件预算分解（weight-level localization）
=========================================================================================
预注册：tests/deepseek_temp/Phase8/N2h1a_design_seal.json（观测前冻结）
执行冻结：tests/deepseek_temp/Phase8/execution_phase8.json（面板/配对/层/种子/配置哈希）

事实基础（Phase 6，冻结引用）：
  B_cat 在 L6 一次性锁死：patch@L5 = +0.578 -> patch@L6 = +10.650（相邻最大增量 @L6 = +10.072）
  => 写入算子在**第 6 层内部**（注意力 + MLP 对残差的贡献）。

恒等式（本脚本的分解基础）：
  h6 = h5 + a6 + m6  =>  diff6 = diff5 + sum_h delta_a6_h + delta_m6
  其中 delta_a6_h = W_o[:, h*128:(h+1)*128] @ (donor_head_h - recip_head_h)   （精确加性）

臂：T 搬运（充分性）/ Z 零消融（必要性）/ LOO 留一 / L7 对照 / W 权重容量 / 4 对照
门：G1 分布式（单头 share<=30% 且 MLP<=50%）| G2 局部件 | G3 混合
用法：SMOKE=1 python n2h1a_weight_localize.py   # 冒烟
      python n2h1a_weight_localize.py           # 正式
"""
import os, sys, io, json, time, hashlib
import numpy as np
import torch

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P8T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase8')
EXEC = os.path.join(P8T, 'execution_phase8.json')
SEAL = os.path.join(P8T, 'N2h1a_design_seal.json')
REPORT = os.path.join(P8T, 'n2h1a_report_qwen3-4b.txt')
RESULT = os.path.join(P8T, 'result_phase8.json')
SMOKE = os.environ.get('SMOKE', '0') == '1'

lines = []
def w(s=''):
    lines.append(str(s)); print(s); sys.stdout.flush()

E = json.load(io.open(EXEC, encoding='utf-8'))
S = json.load(io.open(SEAL, encoding='utf-8'))

MODEL = E['model']
MDIR = os.path.join(ROOT, 'models', 'hf', MODEL)
TMPL = E['template']
SUP_ID = {k: int(v) for k, v in E['sup_id'].items()}
SUPS = E['classes']
PRIMARY = E['primary_layer']          # 6
CTRL = E['control_layer']             # 7
PRE = E['pre_layer']                  # 5
NH = E['n_heads']                     # 32
HD = E['head_dim']                    # 128
RNG = np.random.default_rng(E['seed'])

def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()

w('=== Phase 8 / N2h1-alpha : 写入算子组件预算分解 ===')
w('smoke=%s ; time %s' % (SMOKE, time.strftime('%Y-%m-%d %H:%M:%S')))
w('seal sha8 %s ; exec sha8 %s' % (sha(SEAL)[:8], sha(EXEC)[:8]))
w('config_sha256_match %s (expect %s)' % (sha(os.path.join(MDIR, 'config.json')) == E['config_sha256'],
                                          E['config_sha256'][:12]))

from transformers import AutoTokenizer, AutoModelForCausalLM
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16, trust_remote_code=True,
                                            attn_implementation='eager').to('cuda').eval()
_core = getattr(model.model, 'language_model', model.model)
layers = _core.layers; L = len(layers); head_lm = model.lm_head
HID = model.config.hidden_size
CFG = model.config

def ids_of(s):
    return tok.encode(s, add_special_tokens=False)

def attn_out_proj(layer):
    a = layer.self_attn
    for nm in ['o_proj', 'dense', 'out_proj']:
        if hasattr(a, nm):
            return getattr(a, nm)
    raise RuntimeError('no attn out proj')

ATTN = [attn_out_proj(layers[i]) for i in range(L)]
OIN = ATTN[PRIMARY].in_features
MLPS = [layers[i].mlp for i in range(L)]

# ---------------- 0. drift 断言（execution 冻结）----------------
drift = []
if L != E['expected_cfg']['num_hidden_layers']: drift.append('layers %d != %d' % (L, E['expected_cfg']['num_hidden_layers']))
if HID != E['expected_cfg']['hidden_size']: drift.append('hidden')
if CFG.num_attention_heads != NH: drift.append('n_heads')
if bool(CFG.tie_word_embeddings) != bool(E['expected_cfg']['tie_word_embeddings']): drift.append('tie')
if OIN != E['o_proj_in_features']: drift.append('o_proj_in %d != %d' % (OIN, E['o_proj_in_features']))
w('drift: %s' % (drift if drift else 'NONE'))
assert OIN == NH * HD, 'F4 失败：o_proj 输入维 %d != %d' % (OIN, NH * HD)
if drift:
    w('!! DRIFT 非空，按预注册 F4 停止')
    sys.exit(2)

PAIRS_ALL = E['pairs_all']
DISC = [tuple(x) for x in E['discovery']]
CONF = [tuple(x) for x in E['confirmation']]
INST_ALL = [tuple(x) for x in E['instances_all']]
if SMOKE:
    DISC = DISC[:2]
    CONF = []
    _need = set()
    for p in PAIRS_ALL:
        if p[0] in [x[0] for x in DISC]:
            _need.add(p[0]); _need.add(p[2])
    INST_ALL = [(w_, s_) for (w_, s_) in INST_ALL if w_ in _need]
    w('SMOKE panel: discovery=%d ; captured=%s' % (len(DISC), [x[0] for x in INST_ALL]))

w('')
w('model=%s L=%d hid=%d heads=%d head_dim=%d o_proj_in=%d tie=%s' %
  (MODEL, L, HID, NH, HD, OIN, CFG.tie_word_embeddings))
w('primary layer L%d ; control L%d ; pre L%d' % (PRIMARY, CTRL, PRE))
w('panel: discovery=%d confirmation=%d (all=%d) ; seed=%d' % (len(DISC), len(CONF), len(INST_ALL), E['seed']))
sys.stdout.flush()

# ---------------- 1. 采集：hidden / o_proj 输入 / mlp 输出 ----------------
CAP_L = sorted(set([PRE, PRIMARY, CTRL] + [l for l in E['PATCH_L']]))

@torch.no_grad()
def capture(text):
    ii = torch.tensor([ids_of(text)], device='cuda')
    store = {'o': {}, 'm': {}}
    hs = []
    for l in CAP_L:
        def mk_o(l):
            def f(mod, args):
                store['o'][l] = args[0].detach()[0, -1, :].float().cpu().numpy().copy()
            return f
        def mk_m(l):
            def f(mod, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                store['m'][l] = t.detach()[0, -1, :].float().cpu().numpy().copy()
            return f
        hs.append(ATTN[l].register_forward_pre_hook(mk_o(l)))
        hs.append(MLPS[l].register_forward_hook(mk_m(l)))
    out = model(input_ids=ii, output_hidden_states=True)
    for h in hs:
        h.remove()
    HH = np.stack([h[0, -1].float().detach().cpu().numpy() for h in out.hidden_states], 0)
    return HH, store['o'], store['m'], out.logits[0, -1].float().detach().cpu().numpy()

t_cap = time.time()
CAP = {}
for wd, sup in INST_ALL:
    CAP[wd] = capture(TMPL % wd)
w('capture done %d instances in %.1fs' % (len(CAP), time.time() - t_cap))
if SMOKE:
    # 冒烟：形状/维度/NaN 断言
    wd0 = INST_ALL[0][0]
    HH, OO, MM, lg = CAP[wd0]
    w('SMOKE assert: HH%s O_L%d%s M_L%d%s logits%s nan=%s' %
      (HH.shape, PRIMARY, OO[PRIMARY].shape, PRIMARY, MM[PRIMARY].shape, lg.shape,
       bool(np.isnan(HH).any() or np.isnan(OO[PRIMARY]).any() or np.isnan(MM[PRIMARY]).any())))
    assert OO[PRIMARY].shape[0] == OIN, OO[PRIMARY].shape
    assert MM[PRIMARY].shape[0] == HID, MM[PRIMARY].shape
    assert HH.shape[1] == HID
    assert not np.isnan(HH).any()
PAIRS = [tuple(p) for p in PAIRS_ALL if p[0] in CAP and p[2] in CAP]
w('usable pairs (recipient & donor captured) = %d / %d' % (len(PAIRS), len(PAIRS_ALL)))
sys.stdout.flush()

# ---------------- 2. 类子空间 U_l（只用 discovery 估计）----------------
by_class = {}
for wd, sup in DISC:
    by_class.setdefault(sup, []).append(wd)
AVAIL = [s for s in SUPS if s in by_class]
if len(AVAIL) < 4:
    # SMOKE 专用回退：发现集类数不足时用全部已捕获实例估 U（正式运行走不到这里）
    by_class = {}
    for wd, sup in INST_ALL:
        by_class.setdefault(sup, []).append(wd)
    AVAIL = [s for s in SUPS if s in by_class]
    w('!! U 估计回退到全部捕获实例（类数=%d，SMOKE 路径）' % len(AVAIL))
U = {}
SING = {}
for l in CAP_L:
    mus = np.stack([np.mean([CAP[wd][0][l + 1] for wd in by_class[s]], 0) for s in AVAIL], 0).astype(np.float64)
    D = mus - mus.mean(0, keepdims=True)
    _, sv, Vt = np.linalg.svd(D, full_matrices=False)
    U[l] = Vt[:max(len(AVAIL) - 1, 1)].astype(np.float32)
    SING[l] = sv[:max(len(AVAIL) - 1, 1)]
w('U estimated on discovery only (n_classes=%d, n=%d) ; rank=%d ; sing@L%d=%s' %
  (len(AVAIL), len(DISC), U[PRIMARY].shape[0], PRIMARY, ' '.join('%.1f' % x for x in SING[PRIMARY])))
sys.stdout.flush()

def proj(vec, Ub):
    return (vec @ Ub.T) @ Ub

# ---------------- 3. base 分数 ----------------
def score_rank(v, sup, sid):
    v = v.copy(); v[sid] = -1e9
    own = float(v[SUP_ID[sup]])
    others = [float(v[SUP_ID[x]]) for x in SUPS if x != sup]
    return own - float(np.mean(others))

def score_rank_full(v, sup, sid):
    v = v.copy(); v[sid] = -1e9
    own = float(v[SUP_ID[sup]])
    others = [float(v[SUP_ID[x]]) for x in SUPS if x != sup]
    order = np.argsort(-v)
    return own - float(np.mean(others)), int(np.where(order == SUP_ID[sup])[0][0]) + 1

BASE = {}
for (rw, rs, dw, ds, sw) in PAIRS:
    sr0, rr0 = score_rank_full(CAP[rw][3], rs, ids_of(rw)[0])
    sd0, rd0 = score_rank_full(CAP[rw][3], ds, ids_of(dw)[0])
    BASE[rw] = dict(sr0=sr0, sd0=sd0, rr0=rr0, rd0=rd0)
bad_base = [rw for rw, b in BASE.items() if not (b['sr0'] > 0)]
w('base: n=%d ; recipient 类分数 mean=%+.3f ; donor 类分数 mean=%+.3f ; base<=0 的实例=%s (F2)' %
  (len(BASE), np.mean([b['sr0'] for b in BASE.values()]), np.mean([b['sd0'] for b in BASE.values()]),
   bad_base if bad_base else 'NONE'))
sys.stdout.flush()

# ---------------- 4. 前向工具 ----------------
@torch.no_grad()
def fwd_patch_l6(text, vec):
    ii = torch.tensor([ids_of(text)], device='cuda')
    def hook(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        t = t.clone(); t[0, -1, :] = vec.to(t.dtype)
        return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
    h = layers[PRIMARY].register_forward_hook(hook)
    o = model(input_ids=ii)
    h.remove()
    return o.logits[0, -1].float().detach().cpu().numpy()

@torch.no_grad()
def fwd_ablate_l6_head(text, h_idx):
    """零消融 L6 第 h_idx 头在末位的 o_proj 输入块（h_idx=None -> 全部头；'mlp' -> MLP 输出）"""
    ii = torch.tensor([ids_of(text)], device='cuda')
    hs = []
    if h_idx == 'mlp':
        def hm(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            t = t.clone(); t[0, -1, :] = 0
            return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
        hs.append(MLPS[PRIMARY].register_forward_hook(hm))
    else:
        def ha(mod, args):
            y = args[0].clone()
            if h_idx is None:
                y[:, -1, :] = 0
            else:
                y[:, -1, h_idx * HD:(h_idx + 1) * HD] = 0
            return (y,) + tuple(args[1:])
        hs.append(ATTN[PRIMARY].register_forward_pre_hook(ha))
    o = model(input_ids=ii)
    for h in hs:
        h.remove()
    return o.logits[0, -1].float().detach().cpu().numpy()

# ---------------- 5. T 臂（搬运/充分性）@L6 ----------------
AO = U[PRIMARY]                       # [5, HID]
W_O = ATTN[PRIMARY].weight.detach().float().cpu().numpy()      # [HID, OIN]

def comp_deltas(rw, dw):
    """返回 {name: delta_vec}（34 项：diff5 / 32 头 / MLP）"""
    h5r, h5d = CAP[rw][0][PRE + 1].astype(np.float32), CAP[dw][0][PRE + 1].astype(np.float32)
    h6r = CAP[rw][0][PRIMARY + 1].astype(np.float32)
    h6d = CAP[dw][0][PRIMARY + 1].astype(np.float32)
    m6r, m6d = CAP[rw][2][PRIMARY].astype(np.float32), CAP[dw][2][PRIMARY].astype(np.float32)
    o6r, o6d = CAP[rw][1][PRIMARY].astype(np.float32), CAP[dw][1][PRIMARY].astype(np.float32)
    d = {'diff5': h5d - h5r, 'mlp': m6d - m6r}
    head_delta = np.zeros_like(h6r)
    for h in range(NH):
        sl = slice(h * HD, (h + 1) * HD)
        da = W_O[:, sl] @ (o6d[sl] - o6r[sl])         # 头 h 对 L6 输出的贡献增量
        d['head%d' % h] = da
        head_delta += da
    d['attn_all'] = head_delta
    d['diff6'] = h6d - h6r
    return d

def run_T(panel_pairs, want_W_rand=True, want_mismatch=True):
    T = {}
    names = ['head%d' % h for h in range(NH)] + ['mlp', 'diff5', 'attn_all', 'diff6']
    V = {}                        # name -> [sum ||P_U(delta)||, n]  向量预算（精确可加）
    for nm in names:
        T[nm] = [0.0, 0.0, 0]      # dDonor, dRecip, n
        V[nm] = [0.0, 0]
    if want_W_rand: T['V_rand'] = [0.0, 0.0, 0]
    if want_mismatch: T['M_mismatch'] = [0.0, 0.0, 0]
    n = 0
    for (rw, rs, dw, ds, sw) in panel_pairs:
        if rw not in CAP:
            continue
        B = BASE[rw]; sid_r = ids_of(rw)[0]; sid_d = ids_of(dw)[0]
        h6r = CAP[rw][0][PRIMARY + 1].astype(np.float32)
        D = comp_deltas(rw, dw)
        for nm in names:
            pv = proj(D[nm].astype(np.float32), AO)
            V[nm][0] += float(np.linalg.norm(pv)); V[nm][1] += 1
            vec = torch.tensor(h6r + pv, device='cuda')
            v1 = fwd_patch_l6(TMPL % rw, vec)
            T[nm][0] += score_rank(v1, ds, sid_d) - B['sd0']
            T[nm][1] += score_rank(v1, rs, sid_r) - B['sr0']
            T[nm][2] += 1
        if want_mismatch and sw is not None and sw in CAP:
            Dm = comp_deltas(rw, sw)
            vec = torch.tensor(h6r + proj(Dm['attn_all'].astype(np.float32), AO), device='cuda')
            v1 = fwd_patch_l6(TMPL % rw, vec)
            T['M_mismatch'][0] += score_rank(v1, ds, sid_d) - B['sd0']
            T['M_mismatch'][2] += 1
        if want_W_rand:
            nm_abs = np.mean([np.linalg.norm(D['head%d' % h]) for h in range(NH)])
            for r in range(5):
                vr = RNG.standard_normal(HID).astype(np.float32)
                vr = vr / np.linalg.norm(vr) * nm_abs
                vec = torch.tensor(h6r + proj(vr, AO), device='cuda')
                v1 = fwd_patch_l6(TMPL % rw, vec)
                T['V_rand'][0] += score_rank(v1, ds, sid_d) - B['sd0']
                T['V_rand'][2] += 1
        n += 1
        if n % 8 == 0:
            w('    T arm %d/%d @%.0fs' % (n, len(panel_pairs), time.time() - t0)); sys.stdout.flush()
    out = {}
    for k, (sd, sr, c) in T.items():
        out[k] = dict(dDonor=sd / c if c else 0.0, dRecip=sr / c if c else 0.0, n=c)
    VEC = {k: (v[0] / v[1] if v[1] else 0.0) for k, v in V.items()}
    return out, VEC

w('')
w('--- T 臂 @L%d（discovery n=%d）---' % (PRIMARY, len(DISC)))
T6, VECT = run_T([p for p in PAIRS if p[0] in [x[0] for x in DISC]])

# ---------------- 6. Z 臂（零消融/必要性）@L6 ----------------
Z = {}
for h in range(NH):
    Z['head%d' % h] = [0.0, 0]
Z['attn_all'] = [0.0, 0]; Z['mlp'] = [0.0, 0]
n = 0
for (rw, rs, dw, ds, sw) in PAIRS:
    if rw not in [x[0] for x in DISC]:
        continue
    B = BASE[rw]; sid_r = ids_of(rw)[0]
    for h in list(range(NH)) + ['attn_all', 'mlp']:
        key = 'head%d' % h if isinstance(h, int) else h
        idx = h if isinstance(h, int) else (None if h == 'attn_all' else 'mlp')
        v1 = fwd_ablate_l6_head(TMPL % rw, idx)
        Z[key][0] += score_rank(v1, rs, sid_r) - B['sr0']
        Z[key][1] += 1
    n += 1
    if n % 8 == 0:
        w('    Z arm %d @%.0fs' % (n, time.time() - t0)); sys.stdout.flush()
Z6 = {k: dict(dRecip=v[0] / v[1] if v[1] else 0.0, n=v[1]) for k, v in Z.items()}

# ---------------- 7. LOO 搬运留一 ----------------
def run_LOO(topk):
    acc = [0.0, 0]; base_acc = [0.0, 0]
    order = sorted(range(NH), key=lambda h: -abs(T6['head%d' % h]['dDonor']))
    drop = set(order[:topk])
    for (rw, rs, dw, ds, sw) in PAIRS:
        if rw not in CAP or rw not in [x[0] for x in DISC]:
            continue
        B = BASE[rw]; sid_d = ids_of(dw)[0]
        h6r = CAP[rw][0][PRIMARY + 1].astype(np.float32)
        D = comp_deltas(rw, dw)
        dd = np.zeros_like(h6r)
        for h in range(NH):
            if h not in drop:
                dd = dd + D['head%d' % h]
        vec = torch.tensor(h6r + proj(dd.astype(np.float32), AO), device='cuda')
        v1 = fwd_patch_l6(TMPL % rw, vec)
        acc[0] += score_rank(v1, ds, sid_d) - B['sd0']; acc[1] += 1
        vec = torch.tensor(h6r + proj(D['attn_all'].astype(np.float32), AO), device='cuda')
        v1 = fwd_patch_l6(TMPL % rw, vec)
        base_acc[0] += score_rank(v1, ds, sid_d) - B['sd0']; base_acc[1] += 1
    return acc[0] / max(acc[1], 1), base_acc[0] / max(base_acc[1], 1)

w('')
w('--- LOO 留一搬运 @L%d ---' % PRIMARY)
LOO = {}
for k in [1, 3]:
    a, b = run_LOO(k)
    LOO['drop%d' % k] = dict(dDonor=a, retain=a / b if abs(b) > 1e-9 else float('nan'))
    w('  drop top-%d 头: dDonor=%+.3f  retain=%.3f (全头=%.3f)' % (k, a, LOO['drop%d' % k]['retain'], b))
sys.stdout.flush()

# ---------------- 8. L7 对照（写入是否 L6 特异）----------------
w('')
w('--- L7 聚合对照 ---')
L7 = {}
for nm in ['diff6', 'attn_all', 'mlp']:
    acc = [0.0, 0]; n = 0
    for (rw, rs, dw, ds, sw) in PAIRS:
        if rw not in CAP or rw not in [x[0] for x in DISC]:
            continue
        B = BASE[rw]; sid_d = ids_of(dw)[0]
        h7r = CAP[rw][0][CTRL + 1].astype(np.float32)
        if nm == 'diff6':
            dvec = CAP[dw][0][PRIMARY + 1].astype(np.float32) - CAP[rw][0][PRIMARY + 1].astype(np.float32)
        elif nm == 'mlp':
            dvec = CAP[dw][2][CTRL].astype(np.float32) - CAP[rw][2][CTRL].astype(np.float32)
        else:
            o7r, o7d = CAP[rw][1][CTRL].astype(np.float32), CAP[dw][1][CTRL].astype(np.float32)
            W7 = ATTN[CTRL].weight.detach().float().cpu().numpy()
            dvec = W7 @ (o7d - o7r)
        vec = torch.tensor(h7r + proj(dvec.astype(np.float32), U[CTRL]), device='cuda')
        ii = torch.tensor([ids_of(TMPL % rw)], device='cuda')
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            t = t.clone(); t[0, -1, :] = vec.to(t.dtype)
            return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
        hh = layers[CTRL].register_forward_hook(hook)
        o = model(input_ids=ii); hh.remove()
        v1 = o.logits[0, -1].float().detach().cpu().numpy()
        acc[0] += score_rank(v1, ds, sid_d) - B['sd0']; n += 1
    L7[nm] = dict(dDonor=acc[0] / max(n, 1), n=n)
    w('  L7 %-9s dDonor=%+.3f' % (nm, L7[nm]['dDonor']))
sys.stdout.flush()

# ---------------- 9. W 权重空间容量（无 GPU）----------------
Wsp = {}
per_head = []
for h in range(NH):
    sl = slice(h * HD, (h + 1) * HD)
    per_head.append(float(np.linalg.norm(AO @ W_O[:, sl]) ** 2))
per_head = np.array(per_head)
tot = per_head.sum()
Wsp['head_share'] = (per_head / tot).tolist()
W_down = layers[PRIMARY].mlp.down_proj.weight.detach().float().cpu().numpy()   # [HID, inter]
mlp_cap = float(np.linalg.norm(AO @ W_down) ** 2)
Wsp['mlp_share_vs_attn'] = mlp_cap / tot
Wsp['max_head_share'] = float(Wsp['head_share'][int(np.argmax(per_head))])
Wsp['argmax_head'] = int(np.argmax(per_head))

# ---------------- 10. 验证臂：B_cat 曲线（相邻最大增量层）----------------
w('')
w('--- 验证臂：B_cat 曲线（相邻最大增量）---')
curve = {}
PL = [l for l in E['PATCH_L'] if l < L]
for l in PL:
    acc = [0.0, 0]; n = 0
    for (rw, rs, dw, ds, sw) in PAIRS:
        if rw not in CAP or rw not in [x[0] for x in DISC]:
            continue
        B = BASE[rw]; sid_d = ids_of(dw)[0]
        hr = CAP[rw][0][l + 1].astype(np.float32); hd = CAP[dw][0][l + 1].astype(np.float32)
        vec = torch.tensor(hr + proj((hd - hr).astype(np.float32), U.get(l, U[PRIMARY])), device='cuda')
        ii = torch.tensor([ids_of(TMPL % rw)], device='cuda')
        def hook(mod, inp, out, vec=vec, l=l):
            t = out[0] if isinstance(out, tuple) else out
            t = t.clone(); t[0, -1, :] = vec.to(t.dtype)
            return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
        hh = layers[l].register_forward_hook(hook)
        o = model(input_ids=ii); hh.remove()
        v1 = o.logits[0, -1].float().detach().cpu().numpy()
        acc[0] += score_rank(v1, ds, sid_d) - B['sd0']; n += 1
    curve[l] = acc[0] / max(n, 1)
jump = [(PL[i], curve[PL[i]] - curve[PL[i - 1]]) for i in range(1, len(PL))]
lj, jv = max(jump, key=lambda x: x[1])
w('  ' + '  '.join('L%d(%+.2f)' % (l, curve[l]) for l in PL))
w('  相邻最大增量 = @L%d (+%.3f) ; 冻结主层 = L%d ; flag=%s' %
  (lj, jv, PRIMARY, 'OK' if lj == PRIMARY else 'MISMATCH'))
sys.stdout.flush()

# ---------------- 11. 判据 ----------------
def shares_of(dd, denom_names):
    vals = {k: abs(v) for k, v in dd.items() if k in denom_names}
    tot = sum(vals.values())
    return {k: (v / tot if tot > 0 else 0.0) for k, v in vals.items()}, tot

HEAD_NAMES = ['head%d' % h for h in range(NH)]
DENOM = HEAD_NAMES + ['mlp']          # 公平份额分母 = {32 头, MLP}；attn_all/diff5 不入分母（避免重复计数）
compT = {k: T6[k]['dDonor'] for k in T6 if k not in ('V_rand', 'M_mismatch')}
compZ = {k: Z6[k]['dRecip'] for k in Z6}
shT, totT = shares_of(compT, DENOM)
shZ, totZ = shares_of(compZ, DENOM)
maxT_head = max(shT[k] for k in HEAD_NAMES)
argT_head = max(HEAD_NAMES, key=lambda k: shT[k])
maxZ_head = max(shZ[k] for k in HEAD_NAMES)
I_attn = abs(T6['attn_all']['dDonor']) / max(sum(abs(T6[k]['dDonor']) for k in HEAD_NAMES), 1e-9)
floor_V = abs(T6['V_rand']['dDonor'])
floor_M = abs(T6['M_mismatch']['dDonor'])
maxcomp = max(abs(v) for v in compT.values())

gates = {}
gates['G1_distributed'] = bool(maxT_head <= 0.30 and shT.get('mlp', 0) <= 0.50
                               and maxZ_head <= 0.30 and shZ.get('mlp', 0) <= 0.50
                               and LOO['drop1']['retain'] >= 0.70 and I_attn <= 1.50)
gates['G2_localized'] = bool(maxT_head > 0.50 or shT.get('mlp', 0) > 0.75)
gates['G3_mixed'] = (not gates['G1_distributed']) and (not gates['G2_localized'])
floors_ok = bool(floor_V < 0.10 * maxcomp and floor_M < 0.10 * maxcomp and not bad_base)
verdict = ('G1 分布式搬运' if gates['G1_distributed'] else
           ('G2 局部件主导' if gates['G2_localized'] else 'G3 混合'))

# ---- 修正案 1（seal amend1）：第一指标 = 向量预算（精确可加），第二指标 = 效率份额 ----
vec_budget = {k: VECT[k] for k in DENOM}
tot_v = sum(vec_budget.values())
share_v = {k: (v / tot_v if tot_v > 0 else 0.0) for k, v in vec_budget.items()}
maxv_head = max(share_v[k] for k in HEAD_NAMES)
argv_head = max(HEAD_NAMES, key=lambda k: share_v[k])
eff = {k: abs(T6[k]['dDonor']) / max(vec_budget[k], 1e-9) for k in DENOM}
tot_e = sum(eff.values())
share_eff = {k: (v / tot_e if tot_e > 0 else 0.0) for k, v in eff.items()}
maxe_head = max(share_eff[k] for k in HEAD_NAMES)
# 非线性/超可加指数（诊断，非门）
I_nl = abs(T6['diff6']['dDonor']) / max(sum(abs(T6[k]['dDonor']) for k in DENOM), 1e-9)
# 向量预算的 LOO 为解析互补（非独立证据）——显式声明
loo_vec_top1 = 1.0 - maxv_head

gates['G1_distributed'] = bool(maxv_head <= 0.30 and share_v.get('mlp', 0) <= 0.50
                               and maxe_head <= 0.30 and floors_ok)
gates['G2_localized'] = bool(maxv_head > 0.50 or share_v.get('mlp', 0) > 0.75)
gates['G3_mixed'] = (not gates['G1_distributed']) and (not gates['G2_localized'])
verdict = ('G1 分布式搬运' if gates['G1_distributed'] else
           ('G2 局部件主导' if gates['G2_localized'] else 'G3 混合'))

w('')
w('--- 第一指标：向量预算 share_v（P_U(diff6) 的精确可加分解，分母 = {32 头, MLP}）---')
w('  公平份额 = %.2f%% ; 向量预算合计 ||.||_1 = %.2f (mean per pair)' % (100.0 / len(DENOM), tot_v))
rankv = sorted(DENOM, key=lambda k: -share_v[k])
for i, k in enumerate(rankv[:12]):
    w('  %2d. %-9s share_v=%.4f  ||P_U(delta)||=%.2f  dDonor=%+8.3f  效率份额=%.4f' %
      (i + 1, k, share_v[k], vec_budget[k], T6[k]['dDonor'], share_eff[k]))
w('  最大单头 share_v = %s %.4f ; MLP share_v = %.4f' % (argv_head, maxv_head, share_v.get('mlp', 0)))
w('  记账（不入分母）: ||P_U(diff5)||=%.2f (share vs diff6 = %.3f) ; ||P_U(attn_all)||=%.2f ; ||P_U(diff6)||=%.2f' %
  (VECT['diff5'], VECT['diff5'] / max(VECT['diff6'], 1e-9), VECT['attn_all'], VECT['diff6']))
w('  向量预算 LOO(drop top-1) = 1 - share_v(max) = %.3f  【解析互补，非独立证据】' % loo_vec_top1)
w('  第二指标：最大效率份额(单头) = %.4f' % maxe_head)
w('  非线性指数 I_nl = |dDonor(diff6)| / sum_c|dDonor(c)| = %.3f（>>1 = 写入窗超可加/阈值型）' % I_nl)
w('')
w('--- 效应臂（不可加，只作层性质诊断）---')
for k in ['diff5', 'attn_all', 'mlp', 'diff6', 'V_rand', 'M_mismatch']:
    w('  %-11s dDonor=%+8.3f  dRecip=%+8.3f' % (k, T6[k]['dDonor'], T6[k]['dRecip']))
w('  效应和 = %+.3f vs diff6 效应 = %+.3f ⇒ 超可加倍数 %.2f' %
  (sum(T6[k]['dDonor'] for k in DENOM), T6['diff6']['dDonor'], I_nl))
w('')
w('  地板: V_rand=%+.3f (%.3f x max|comp|) ; M_mismatch=%+.3f (%.3f x max|comp|) ; floors_ok=%s' %
  (floor_V, floor_V / max(maxcomp, 1e-9), floor_M, floor_M / max(maxcomp, 1e-9), floors_ok))

w('')
w('--- 组件预算（分母 = {32 头, MLP}，公平份额 = %.2f%%）---' % (100.0 / len(DENOM)))
rank = sorted(((k, compT[k]) for k in DENOM), key=lambda kv: -abs(kv[1]))
for i, (k, v) in enumerate(rank[:12]):
    w('  %2d. %-9s dDonor=%+8.3f T-share=%.3f | 零消融 dRecip=%+8.3f Z-share=%.3f' %
      (i + 1, k, v, shT[k], compZ.get(k, float('nan')), shZ.get(k, float('nan'))))
w('  记账（不入分母）: diff5(上游残留)=%+.3f ; attn_all(32 头之和)=%+.3f ; diff6(全量)=%+.3f' %
  (T6['diff5']['dDonor'], T6['attn_all']['dDonor'], T6['diff6']['dDonor']))
w('  最大单头(T) = %s share=%.3f ; MLP share=%.3f ; 最大单头(Z) share=%.3f ; I_attn=%.3f' %
  (argT_head, maxT_head, shT.get('mlp', 0), maxZ_head, I_attn))
w('  地板: V_rand=%+.3f (%.3f x max) ; M_mismatch=%+.3f (%.3f x max)' %
  (floor_V, floor_V / max(maxcomp, 1e-9), floor_M, floor_M / max(maxcomp, 1e-9)))
_closure = (T6['diff5']['dDonor'] + T6['attn_all']['dDonor'] + T6['mlp']['dDonor'])
w('  预算闭合: diff5+attn_all+mlp = %+.3f vs diff6 = %+.3f (相对残差 %.4f)' %
  (_closure, T6['diff6']['dDonor'], abs(_closure - T6['diff6']['dDonor']) / max(abs(T6['diff6']['dDonor']), 1e-9)))
w('  层内三块占比(以 |diff6| 为 1): 上游 %.3f / 注意力 %.3f / MLP %.3f' %
  (abs(T6['diff5']['dDonor']) / max(abs(T6['diff6']['dDonor']), 1e-9),
   abs(T6['attn_all']['dDonor']) / max(abs(T6['diff6']['dDonor']), 1e-9),
   abs(T6['mlp']['dDonor']) / max(abs(T6['diff6']['dDonor']), 1e-9)))
w('  W 权重容量: 最大头 #%d share=%.3f ; MLP/attn = %.3f' % (Wsp['argmax_head'], Wsp['max_head_share'], Wsp['mlp_share_vs_attn']))
w('')
w('>>> 门判决: G1=%s G2=%s G3=%s ; floors_ok=%s' % (gates['G1_distributed'], gates['G2_localized'], gates['G3_mixed'], floors_ok))
w('>>> 裁决: %s' % verdict)
w('')

# ---------------- 12. 确认集（只测门统计）----------------
conf_out = {}
if CONF:
    w('--- 确认集复核（n=%d，仅门统计）---' % len(CONF))
    Pc = [p for p in PAIRS if p[0] in [x[0] for x in CONF]]
    Tc, VECC = run_T(Pc, want_W_rand=True, want_mismatch=False)
    compTc = {k: Tc[k]['dDonor'] for k in Tc if k not in ('V_rand', 'M_mismatch')}
    shTc, _ = shares_of(compTc, DENOM)
    maxTc = max(shTc[k] for k in HEAD_NAMES)
    I_c = abs(Tc['attn_all']['dDonor']) / max(sum(abs(Tc[k]['dDonor']) for k in HEAD_NAMES), 1e-9)
    # 修正案 1：确认集也用向量预算
    vb_c = {k: VECC[k] for k in DENOM}
    tv_c = sum(vb_c.values())
    share_vc = {k: (v / tv_c if tv_c > 0 else 0.0) for k, v in vb_c.items()}
    maxv_c = max(share_vc[k] for k in HEAD_NAMES)
    eff_c = {k: abs(Tc[k]['dDonor']) / max(vb_c[k], 1e-9) for k in DENOM}
    te_c = sum(eff_c.values())
    share_eff_c = {k: (v / te_c if te_c > 0 else 0.0) for k, v in eff_c.items()}
    maxe_c = max(share_eff_c[k] for k in HEAD_NAMES)
    I_nl_c = abs(Tc['diff6']['dDonor']) / max(sum(abs(Tc[k]['dDonor']) for k in DENOM), 1e-9)
    w('  确认集: 最大单头 share_v=%.4f ; MLP share_v=%.4f ; 最大效率份额=%.4f ; I_nl=%.3f ; V_rand=%+.3f' %
      (maxv_c, share_vc.get('mlp', 0), maxe_c, I_nl_c, abs(Tc['V_rand']['dDonor'])))
    # 确认集 LOO（drop top-1，用确认集自身排序）
    order_c = sorted(range(NH), key=lambda h: -abs(Tc['head%d' % h]['dDonor']))
    acc = [0.0, 0]; bas = [0.0, 0]
    for (rw, rs, dw, ds, sw) in Pc:
        B = BASE[rw]; sid_d = ids_of(dw)[0]
        h6r = CAP[rw][0][PRIMARY + 1].astype(np.float32)
        D = comp_deltas(rw, dw)
        dd = np.zeros_like(h6r)
        for h in range(NH):
            if h != order_c[0]:
                dd = dd + D['head%d' % h]
        v1 = fwd_patch_l6(TMPL % rw, torch.tensor(h6r + proj(dd.astype(np.float32), AO), device='cuda'))
        acc[0] += score_rank(v1, ds, sid_d) - B['sd0']; acc[1] += 1
        v1 = fwd_patch_l6(TMPL % rw, torch.tensor(h6r + proj(D['attn_all'].astype(np.float32), AO), device='cuda'))
        bas[0] += score_rank(v1, ds, sid_d) - B['sd0']; bas[1] += 1
    ret_c = (acc[0] / max(acc[1], 1)) / (bas[0] / max(bas[1], 1)) if abs(bas[0]) > 1e-9 else float('nan')
    w('  确认集 LOO(drop top-1): retain=%.3f' % ret_c)
    conf_out = dict(max_head_share_v=maxv_c, mlp_share_v=share_vc.get('mlp', 0),
                    max_head_share_eff=maxe_c, I_nl=I_nl_c,
                    V_rand=Tc['V_rand']['dDonor'], LOO_retain_top1=ret_c,
                    same_band=bool((maxv_c <= 0.30) == (maxv_head <= 0.30)))
    w('  确认集 LOO(drop top-1): retain=%.3f' % ret_c)
    w('  确认集: 最大单头 share=%.3f ; MLP share=%.3f ; V_rand=%+.3f' %
      (maxTc, shTc.get('mlp', 0), abs(Tc['V_rand']['dDonor'])))
    w('  同带（单头 share_v<=0.30 一致）: %s' % conf_out['same_band'])

el = time.time() - t0
w('')
w('total %.1fs' % el)

# ---------------- 13. 落盘 ----------------
res = dict(phase=8, name='N2h1-alpha', model=MODEL, smoke=SMOKE, elapsed_s=round(el, 1),
           layers=dict(L=L, primary=PRIMARY, control=CTRL, pre=PRE, n_heads=NH, head_dim=HD),
           panel=dict(discovery=len(DISC), confirmation=len(CONF)),
           base_ok=(not bad_base), base_recip_mean=float(np.mean([b['sr0'] for b in BASE.values()])),
           T=T6, Z=Z6, LOO=LOO, L7=L7, W=Wsp, curve=dict((str(k), v) for k, v in curve.items()),
           jump_layer=lj, jump_value=jv, jump_flag=('OK' if lj == PRIMARY else 'MISMATCH'),
           shares_T=shT, shares_Z=shZ, max_head_share_T=maxT_head, argmax_head_T=argT_head,
           I_attn=I_attn, floors=dict(V_rand=floor_V, M_mismatch=floor_M, floors_ok=floors_ok),
           amend1=dict(vec_budget=vec_budget, share_v=share_v, max_head_share_v=maxv_head,
                       argmax_head_v=argv_head, eff=eff, share_eff=share_eff,
                       max_head_share_eff=maxe_head, I_nl=I_nl, loo_vec_top1=loo_vec_top1),
           gates=gates, verdict=verdict, confirmation=conf_out,
           seal_sha8=sha(SEAL)[:8], exec_sha8=sha(EXEC)[:8])
io.open(RESULT, 'w', encoding='utf-8').write(json.dumps(res, ensure_ascii=False, indent=1))
io.open(REPORT, 'w', encoding='utf-8').write('\n'.join(lines))
print('DONE ->', REPORT, '|', RESULT)
