# -*- coding: utf-8 -*-
"""Phase 3263 (G5-H26): P-TRAJ checkpoint training-step trajectory — GPU arm
first execution, per the sealed 3262 protocol draft gpu_trajectory_protocol_draft_v1.json
(b569ce24).

States: base + 6 runs x 4 checkpoints {1,8,32,128} = 25 on qwen3-1.7b
(models/hf/qwen3-1.7b), block model.layers[16].mlp (gate/up/down, 37,748,736
params). Restore rule verbatim phase2755 (a2f3b950): W_t = W_base + delta_t +
residual_t, bit-exact in FP32 parameter space (block upcast to float32 BEFORE
capturing originals, 2755 precedent). Collect: verbatim phase3169 convention
(batch=1, bf16 forward, add_special_tokens=False, last-token h[0,-1],
NH = NL+1 = 29 hidden states, fp16 storage). Chain: verbatim 3190/3252/3253
(ridge LAM=1e-3, 3 seeds split_s1_246, 876 OOV residual rows row-normalized,
seed-averaged, SVD top-R=8).

Readouts (pinned pre-observation, from 3262 draft):
  R1 formation timing: dominance D = ovr(mid,out) / max(ovr(mid,emb), eps);
     band D >= 30 (3253 observed 30-40x). FORMED_DURING_TRAINING iff base
     D < 30 AND >= 5/6 runs reach D >= 30 at some step; PRETRAINING_PROPERTY
     iff base D >= 30; else MIXED.
  R2 track-I stability: TRACK1_STABLE iff layer-0 top1 >= 0.95 for all 25
     states AND |cos(u*_emb(state), u*_emb(base))| >= 0.95 for all 24
     non-base states.
  R3 behavior alignment: spearman across 6 runs between step-128 scalars
     (D_128, sigma1_mid_out_128) and 2755 behavior improvement — n=6,
     characterization, no gate.
Device gates: D1 finite+shape, D2 first-token uniqueness, tail-token '。',
bit-exact restore + base re-collect bitwise assert, ledger zero-write.
"""
import hashlib
import json
import os
import time

import numpy as np

T0 = time.time()
NAME = 'g5h26_p_traj'
SMOKE = os.environ.get('P3263_SMOKE') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913')
OUTDIR = os.path.join(SRC_DIR, 'phase3263', NAME)
LOG = []

MODEL_DIR = os.path.join(ROOT, r'models\hf\qwen3-1.7b')
P2755 = os.path.join(SRC_DIR, 'phase2755', 'qwen25_3b')
AUDIT = os.path.join(SRC_DIR, 'phase3262', 'g5h25_gpu_arm_readiness',
                     'checkpoint_presence_audit_v1.json')
PROTO = os.path.join(SRC_DIR, 'phase3262', 'g5h25_gpu_arm_readiness',
                     'gpu_trajectory_protocol_draft_v1.json')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')

RUNS = ['true_token_2747', 'within_surface_class_permuted_token_2747',
        'surface_class_mass_2747', 'true_token_2748',
        'within_surface_class_permuted_token_2748', 'surface_class_mass_2748']
STEPS = [1, 8, 32, 128]
MAT_KEYS = ['gate_proj.weight', 'up_proj.weight', 'down_proj.weight']
BLOCK = 'model.layers[16].mlp'
BLOCK_PARAMS = 37_748_736
R_BASIS = 8
LAM = 1e-3
SEEDS_S1 = [7, 8, 9]
G_DOM = 30.0
R2_TOP1 = 0.95
R2_COS = 0.95
COND_OF_RUN = {r: ('surface_class_mass' if r.startswith('surface')
                   else ('true_token' if r.startswith('true')
                         else 'within_surface_class_permuted_token'))
               for r in RUNS}

# ---- panel verbatim (3169/3253) ----
CLASSES_SEEN = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
ENT_SEEN = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
CLASSES_OOV = ['乐器', '天气', '运动', '电器']
ENT_OOV_PANEL = {
    '乐器': ['钢琴', '小提琴', '吉他', '鼓', '笛子', '二胡', '琵琶', '口琴'],
    '天气': ['雨', '雪', '雷', '雾', '冰雹', '台风', '露水', '霜'],
    '运动': ['足球', '篮球', '乒乓球', '游泳', '跑步', '体操', '拳击', '围棋'],
    '电器': ['电视', '冰箱', '洗衣机', '空调', '微波炉', '电饭煲', '吸尘器', '风扇'],
}
TPL = {0: '{e}是一种{c}。', 1: '{e}属于{c}这一类。', 2: '{e}，一种常见的{c}。'}
NT = 3
CLASSES = CLASSES_SEEN + CLASSES_OOV
ENT = dict(ENT_SEEN)
ENT.update(ENT_OOV_PANEL)
ENTS = [e for cl in CLASSES for e in ENT[cl]]
NE = len(ENTS)
NC = len(CLASSES)
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
NP_ = len(PAIRS)
assert (NE, NC, NP_) == (73, 10, 730)
PANEL = NT * NP_
assert PANEL == 2190
NCLS_SEEN = len(CLASSES_SEEN)
ALL_OOV_ROWS = [t * NP_ + pi for t in range(NT) for pi in range(NP_)
                if PAIRS[pi][1] >= NCLS_SEEN]
assert len(ALL_OOV_ROWS) == 876
OOV_ROW_CLS = np.array([PAIRS[r % NP_][1] for r in ALL_OOV_ROWS])
STATES = ['base'] + ['%s@%d' % (r, s) for r in RUNS for s in STEPS]
N_STATES = len(STATES)
assert N_STATES == 25

DESIGN = {
    'phase': '3263', 'name': NAME,
    'prereg': '3262 sealed protocol draft P_TRAJ (gpu_trajectory_protocol_draft_v1 '
              'b569ce24), frozen before any observation',
    'model': 'qwen3-1.7b', 'block': BLOCK, 'block_params': BLOCK_PARAMS,
    'states': STATES,
    'restore_rule': 'W_t = W_base + delta_t + residual_t (bit-exact, fp32 '
                    'parameter space, block upcast before capture, 2755 '
                    'precedent a2f3b950)',
    'collect': 'verbatim phase3169: batch=1, bf16, add_special_tokens=False, '
               'last-token h[0,-1], NH=NL+1, fp16 storage; block layers[16].mlp '
               'computes in fp32 via bridged forward (2755 FP32-block precedent), '
               'output cast back to bf16 — identical across all 25 states',
    'chain': 'verbatim 3190/3252/3253: ridge LAM=%g, seeds %s, 876 OOV rows '
             'row-normalized seed-averaged, SVD top-R=%d' % (LAM, SEEDS_S1, R_BASIS),
    'readouts': {
        'R1': 'dominance D = ovr(mid,out)/max(ovr(mid,emb),eps); band D>=%g; '
              'FORMED_DURING_TRAINING iff base D<%g AND >=5/6 runs reach band; '
              'PRETRAINING_PROPERTY iff base D>=%g; else MIXED'
              % (G_DOM, G_DOM, G_DOM),
        'R2': 'TRACK1_STABLE iff layer0 top1 >= %g all 25 states AND '
              '|cos(u*_emb,base)| >= %g all 24 non-base states'
              % (R2_TOP1, R2_COS),
        'R3': 'spearman(6 runs): step-128 D / sigma1_mid_out vs 2755 behavior '
              'improvement (characterization, no gate)',
    },
    'device_gates': ['D1 finite+shape per state', 'D2 first-token uniqueness',
                     'tail token = 。 rate == 100%',
                     'bit-exact restore + base re-collect bitwise',
                     'ledger zero-write'],
    'ledger': {'n': 419, 'post': 'ec0dc42c', 'chain_prev': '4f5ed748'},
    'zero_write': ['research/gpt5/atlas/atlas_ledger.json'],
}
if SMOKE:
    DESIGN['smoke_states'] = ['base', 'true_token_2747@128']
DESIGN_BLOB = json.dumps(DESIGN, ensure_ascii=False, sort_keys=True,
                         indent=1).encode('utf-8')
design_sha8 = hashlib.sha256(DESIGN_BLOB).hexdigest()[:8]


def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    LOG.append(ln)
    try:
        print(ln, flush=True)
    except Exception:
        pass


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


os.makedirs(OUTDIR, exist_ok=True)
exe_p = os.path.join(OUTDIR, 'execution_smoke.json' if SMOKE else 'execution.json')
if os.path.exists(exe_p):
    prev = json.load(open(exe_p, encoding='utf-8'))
    assert prev['design_sha8'] == design_sha8, 'DESIGN DRIFT'
    log('freeze: existing %s OK (%s)'
        % (os.path.basename(exe_p), design_sha8))
else:
    json.dump({'phase': 3263, 'name': NAME, 'design_sha8': design_sha8,
               'design': DESIGN, 'frozen_before': 'any observation',
               'created': time.strftime('%Y-%m-%d %H:%M:%S')},
              open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    log('freeze: execution.json written (%s)' % design_sha8)


def split_s1_246(seed):
    SEEN_PAIR_SET = set((i, c) for i in range(41) for c in range(6))
    pairs246 = sorted(SEEN_PAIR_SET)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(pairs246))
    n_test = int(round(0.2 * len(pairs246)))
    test = set(pairs246[j] for j in idx[:n_test])
    return SEEN_PAIR_SET - test, test


def phi_cols():
    return NE + NC + NT + 1


def phi_row(i, c, t):
    v = np.zeros(phi_cols(), np.float32)
    v[i] = 1.0
    v[NE + c] = 1.0
    v[NE + NC + t] = 1.0
    v[-1] = 1.0
    return v


def unit(v):
    return v / (float(np.linalg.norm(v)) + 1e-18)


# ---- feature matrices (verbatim 3253 precompute) ----
PI_OF_PAIR = {p: j for j, p in enumerate(PAIRS)}
tr_base, Xtr, A_cache, Xo, te_seen_all, Xs = {}, {}, {}, None, {}, {}
for s in SEEDS_S1:
    tr_pairs, te_pairs = split_s1_246(s)
    tr_base[s] = [t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                  for p in PAIRS if p in tr_pairs]
    te_seen_all[s] = [t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                      for p in PAIRS if p in te_pairs]
    Xtr[s] = np.stack([phi_row(r % NP_ // 1 * 0 + PAIRS[r % NP_][0],
                               PAIRS[r % NP_][1], r // NP_)
                       for r in tr_base[s]])
    Xs[s] = np.stack([phi_row(PAIRS[r % NP_][0], PAIRS[r % NP_][1], r // NP_)
                      for r in te_seen_all[s]])
    A_cache[s] = Xtr[s].T @ Xtr[s] + LAM * np.eye(Xtr[s].shape[1], dtype=np.float32)
Xo = np.stack([phi_row(PAIRS[r % NP_][0], PAIRS[r % NP_][1], r // NP_)
               for r in ALL_OOV_ROWS])
log('panel features ready (phi=%d cols, train/seed %s, oov %d rows)'
    % (phi_cols(), [len(tr_base[s]) for s in SEEDS_S1], len(ALL_OOV_ROWS)))


def ustar_chain(H16):
    """Verbatim 3190/3252/3253 chain on an H16 (NT, NP_, NH, D) fp16 array."""
    NTz, NPz, NH, D = H16.shape
    rd = NH - 1
    top1_curve, basis, ustar1 = {}, {}, {}
    for k in range(NH):
        Y = H16[:, :, k, :].reshape(NTz * NPz, D).astype(np.float32)
        assert np.isfinite(Y).all(), ('D1 non-finite', k)
        Q_rows = np.zeros((len(ALL_OOV_ROWS), D), dtype=np.float32)
        for s in SEEDS_S1:
            W = np.linalg.solve(A_cache[s], Xtr[s].T @ Y[tr_base[s]])
            R_oov = Y[ALL_OOV_ROWS] - Xo @ W
            for j in range(len(ALL_OOV_ROWS)):
                rvv = R_oov[j]
                Q_rows[j] += rvv / (float(np.linalg.norm(rvv)) + 1e-18)
        Qn = Q_rows / len(SEEDS_S1)
        Qn = Qn / (np.linalg.norm(Qn, axis=1, keepdims=True) + 1e-18)
        _, sv, Vt = np.linalg.svd(Qn, full_matrices=False)
        energy = sv ** 2
        top1_curve[k] = float(energy[0] / energy.sum())
        basis[k] = Vt[:R_BASIS].copy()
        ustar1[k] = Vt[0].copy()
        del Vt, Y, W, R_oov, Q_rows, Qn
    cands = [k for k in range(1, rd + 1)]
    k_mid = min(cands, key=lambda k: top1_curve[k])
    B_mid, B_out, B_emb = basis[k_mid], basis[rd], basis[0]

    def spect(Ba, Bb):
        return [float(x) for x in np.linalg.svd(Ba @ Bb.T, compute_uv=False)]

    s_mo = spect(B_mid, B_out)
    s_me = spect(B_mid, B_emb)
    sig1 = {'mid_out': s_mo[0], 'mid_emb': s_me[0]}
    ovr = {'mid_out': sum(x * x for x in s_mo) / R_BASIS,
           'mid_emb': sum(x * x for x in s_me) / R_BASIS}
    dom = ovr['mid_out'] / max(ovr['mid_emb'], 1e-9)
    return {'top1_curve': top1_curve, 'k_mid': k_mid,
            'top1_kmid': top1_curve[k_mid], 'sigma1': sig1, 'ovr': ovr,
            'dominance': dom, 'layer0_top1': top1_curve[0],
            'ustar_kout': ustar1[rd], 'ustar_emb': ustar1[0],
            'ustar_all': {k: ustar1[k] for k in range(NH)}}


# ================================================================ main
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

gates = {}

# ---- g1: ledger state (chain recompute) ----
led_raw = open(LED, 'rb').read()
led = json.loads(led_raw.decode('utf-8'))
chk = json.loads(led_raw.decode('utf-8'))
chk['ledger_sha256_8'] = DESIGN['ledger']['chain_prev']
rec = hashlib.sha256(json.dumps(chk, ensure_ascii=False, indent=1)
                     .encode('utf-8')).hexdigest()[:8]
gates['g1_ledger_state'] = (len(led['measurements']) == DESIGN['ledger']['n']
                            and led['ledger_sha256_8'] == DESIGN['ledger']['post']
                            and rec == DESIGN['ledger']['post'])
log('g1 ledger n=%d post=%s recompute=%s'
    % (len(led['measurements']), led['ledger_sha256_8'], rec))

# ---- g2: protocol draft + 24 delta sha8 vs 3262 audit ----
proto_sha = sha8_file(PROTO)
audit = json.load(open(AUDIT, encoding='utf-8'))
audit_deltas = {(d['run'], d['step']): d['sha8']
                for d in audit['qwen3_1_7b_port']['deltas']}
bad = []
for (r, s), ref in sorted(audit_deltas.items()):
    p = os.path.join(P2755, r, 'delta_%03d.npz' % s)
    if sha8_file(p) != ref:
        bad.append('%s@%d' % (r, s))
gates['g2_proto_and_deltas'] = (proto_sha == 'b569ce24' and not bad
                                and len(audit_deltas) == 24)
log('g2 proto %s | deltas 24 sha8 re-verified, bad=%s' % (proto_sha, bad))

# ---- g3: model + tokenizer device gates ----
gates['g3_model_present'] = all(os.path.isfile(os.path.join(
    MODEL_DIR, f)) for f in ['config.json', 'model.safetensors.index.json',
                             'model-00001-of-00002.safetensors',
                             'model-00002-of-00002.safetensors'])
log('g3 model files: %s' % gates['g3_model_present'])

tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
CLS_TOK = [tok(c, add_special_tokens=False)['input_ids'][0] for c in CLASSES]
d2_ok = len(set(CLS_TOK)) == NC
PROMPTS = [TPL[t].format(e=ENTS[i], c=CLASSES[c])
           for t in range(NT) for (i, c) in PAIRS]
TOKIDS = [tok(p, add_special_tokens=False)['input_ids'] for p in PROMPTS]
tail_bad = sum(1 for ids in TOKIDS if tok.decode([ids[-1]]) != '。')
gates['g4_tokenizer_gates'] = (d2_ok and tail_bad == 0 and len(TOKIDS) == PANEL)
log('g4 tokenizer: D2=%s tail_bad=%d/%d' % (d2_ok, tail_bad, PANEL))

# ---- g5: torch cuda ----
import torch  # noqa: E402
gates['g5_torch_cuda'] = bool(torch.cuda.is_available())
log('g5 cuda: %s (%s)' % (gates['g5_torch_cuda'],
                          torch.cuda.get_device_name(0)
                          if gates['g5_torch_cuda'] else 'none'))

# ---- load model, upcast block, capture originals ----
torch.manual_seed(0)
cfg = json.load(open(os.path.join(MODEL_DIR, 'config.json'), encoding='utf-8'))
NL = cfg['num_hidden_layers']
D = cfg['hidden_size']
NH = NL + 1
assert (NL, D) == (28, 2048)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, trust_remote_code=True).to('cuda').eval()
assert model.config.num_hidden_layers == NL
target = getattr(model.model.layers[16], 'mlp')
with torch.no_grad():
    target.float()
named = dict(target.named_parameters())
assert sum(p.numel() for p in named.values()) == BLOCK_PARAMS
original = {k: named[k].detach().cpu().clone() for k in MAT_KEYS}

# FP32-block bridge (2755 precedent): mlp computes in fp32 (weights upcast),
# output cast back to input dtype — required on current torch where F.linear
# rejects mixed bf16 input vs fp32 weights.
import torch.nn.functional as F  # noqa: E402
_ACT = target.act_fn


def _mlp_forward(x):
    xf = x.float()
    g = F.linear(xf, named['gate_proj.weight'])
    u = F.linear(xf, named['up_proj.weight'])
    y = F.linear(_ACT(g) * u, named['down_proj.weight'])
    return y.to(x.dtype)


target.forward = _mlp_forward
log('model loaded; block upcast fp32 + bridged forward; originals captured (%s)'
    % {k: tuple(original[k].shape) for k in MAT_KEYS})


def set_state(state):
    with torch.no_grad():
        for k in MAT_KEYS:
            named[k].copy_(torch.from_numpy(np.ascontiguousarray(state[k]))
                           .to(named[k].device))


def load_state(key):
    """W_t = W0 + delta_t + residual_t (fp32, same op order as phase2755 save
    time -> bit-exact by construction); validates shapes/finite/residual."""
    if key == 'base':
        return None
    run, st = key.rsplit('@', 1)
    z = np.load(os.path.join(P2755, run, 'delta_%03d.npz' % int(st)),
                allow_pickle=False)
    state, res_rel = {}, {}
    for k in MAT_KEYS:
        d = z[k]
        r = z['reconstruction_residual__' + k]
        assert d.shape == original[k].numpy().shape and d.dtype == np.float32, (key, k)
        assert np.isfinite(d).all() and np.isfinite(r).all(), (key, k)
        v = original[k].numpy() + d
        state[k] = (v + r).astype(np.float32)
        res_rel[k] = float(np.abs(r).max() / max(float(np.abs(d).max()), 1e-30))
    return state, res_rel


def collect(key, refresh=False):
    """Verbatim 3169 collect for one state; cached npz; returns H16 fp16.
    refresh=True bypasses the cache (used by the post-restore bitwise
    recheck, which writes collect_base_recheck.npz)."""
    if refresh:
        cache = os.path.join(OUTDIR, 'collect_base_recheck.npz')
    else:
        cache = os.path.join(OUTDIR, 'collect_%s.npz' % key.replace('@', '_s'))
    if os.path.exists(cache) and not refresh:
        z = np.load(cache, allow_pickle=False)
        H16 = z['H']
        assert H16.shape == (NT, NP_, NH, D)
        log('collect cache hit %s H=%s' % (key, H16.shape))
        return H16
    H16 = np.zeros((NT, NP_, NH, D), np.float16)
    if key != 'base':
        state, res_rel = load_state(key)
        set_state(state)
        log('patched %s (max |res|/|delta| per mat: %s)'
            % (key, {k: round(v, 6) for k, v in res_rel.items()}))
    with torch.no_grad():
        for t in range(NT):
            for pi in range(NP_):
                ii = torch.tensor([TOKIDS[t * NP_ + pi]], device='cuda')
                o = model(input_ids=ii, output_hidden_states=True)
                hs = o.hidden_states
                hv = np.stack([h[0, -1].float().detach().cpu().numpy()
                               for h in hs], 0)
                H16[t, pi] = hv.astype(np.float16)
                del o, hs, hv
    np.savez_compressed(cache, H=H16, allow_pickle=False)
    log('collect saved %s (%d MB)' % (key, os.path.getsize(cache) // 1048576))
    return H16


STATE_KEYS = DESIGN.get('smoke_states', STATES) if SMOKE else STATES
log('states to run: %d %s' % (len(STATE_KEYS), STATE_KEYS if SMOKE else ''))

per_state = {}
ustar_store = {}
base_ustar = None
for si, key in enumerate(STATE_KEYS):
    H16 = collect(key)
    met = ustar_chain(H16)
    ustar_store[key] = {'kout': met['ustar_kout'].copy(),
                        'emb': met['ustar_emb'].copy()}
    if key == 'base':
        base_ustar = ustar_store[key]
        met['cos_kout_to_base'] = 1.0
        met['cos_emb_to_base'] = 1.0
    else:
        met['cos_kout_to_base'] = abs(float(met['ustar_kout'] @ base_ustar['kout']))
        met['cos_emb_to_base'] = abs(float(met['ustar_emb'] @ base_ustar['emb']))
    del met['ustar_all'], met['ustar_kout'], met['ustar_emb']
    per_state[key] = met
    log('state %2d/%d %s: top1[kout]=%.4f k_mid=%d D=%.1f '
        'layer0_top1=%.4f cos_out=%.4f cos_emb=%.4f'
        % (si + 1, len(STATE_KEYS), key, met['top1_curve'][NH - 1],
           met['k_mid'], met['dominance'], met['layer0_top1'],
           met['cos_kout_to_base'], met['cos_emb_to_base']))
    del H16

# ---- bit-exact restore + base re-collect (cache-bypassing) ----
set_state({k: original[k].numpy() for k in MAT_KEYS})
with torch.no_grad():
    for k in MAT_KEYS:
        assert torch.equal(named[k].detach().cpu(), original[k]), k
if 'base' in STATE_KEYS:
    H_re = collect('base', refresh=True)
    H_base0 = np.load(os.path.join(OUTDIR, 'collect_base.npz'),
                      allow_pickle=False)['H']
    gates['g6_restore_bitwise'] = bool(np.array_equal(H_re, H_base0))
    del H_re, H_base0
else:
    gates['g6_restore_bitwise'] = None
log('g6 restore bitwise: %s' % gates['g6_restore_bitwise'])

# ---- readouts (formal only) ----
readouts = {}
if not SMOKE:
    # R1 formation timing
    base_dom = per_state['base']['dominance']
    if base_dom >= G_DOM:
        r1 = {'status': 'PRETRAINING_PROPERTY', 'base_dominance': base_dom}
    else:
        formed = {}
        for r in RUNS:
            hit = [s for s in STEPS
                   if per_state['%s@%d' % (r, s)]['dominance'] >= G_DOM]
            formed[r] = hit[0] if hit else None
        n_formed = sum(1 for v in formed.values() if v is not None)
        r1 = {'status': ('FORMED_DURING_TRAINING' if n_formed >= 5 else 'MIXED'),
              'base_dominance': base_dom, 'first_step_in_band': formed,
              'runs_formed': n_formed}
    readouts['R1'] = r1
    # R2 track-I stability
    r2_top1 = all(per_state[k]['layer0_top1'] >= R2_TOP1 for k in STATE_KEYS)
    r2_cos = all(per_state[k]['cos_emb_to_base'] >= R2_COS
                 for k in STATE_KEYS if k != 'base')
    readouts['R2'] = {'status': ('TRACK1_STABLE' if (r2_top1 and r2_cos)
                                 else 'TRACK1_UNSTABLE'),
                      'layer0_top1_min': min(per_state[k]['layer0_top1']
                                             for k in STATE_KEYS),
                      'cos_emb_min': min(per_state[k]['cos_emb_to_base']
                                         for k in STATE_KEYS if k != 'base')}
    # R3 behavior alignment (n=6, characterization)
    bz = np.load(os.path.join(P2755, 'behavior_scores.npz'), allow_pickle=False)
    improve = {}
    for r in RUNS:
        cond = COND_OF_RUN[r]
        improve[r] = float(bz['base__%s__own' % cond].mean()
                           - bz['%s__own' % r].mean())
    runs_at_128 = ['%s@128' % r for r in RUNS]
    x_dom = [per_state[k]['dominance'] for k in runs_at_128]
    x_s1 = [per_state[k]['sigma1']['mid_out'] for k in runs_at_128]
    y = [improve[r] for r in RUNS]

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        return float(np.corrcoef(ra, rb)[0, 1])

    readouts['R3'] = {'improvement': improve,
                      'spearman_D128_vs_improvement': spearman(x_dom, y),
                      'spearman_sigma1_128_vs_improvement': spearman(x_s1, y),
                      'note': 'n=6 characterization, no gate'}
    log('R1: %s | R2: %s | R3: spearman D=%.3f sig1=%.3f'
        % (r1['status'], readouts['R2']['status'],
           readouts['R3']['spearman_D128_vs_improvement'],
           readouts['R3']['spearman_sigma1_128_vs_improvement']))

# ---- zero-write + verdict ----
zero_before = led['ledger_sha256_8']
gates['g7_zero_write'] = (sha8_file(LED) == hashlib.sha256(led_raw).hexdigest()[:8])
log('g7 zero-write: %s' % gates['g7_zero_write'])

gate_all = all(v for v in gates.values() if v is not None)
resolved = bool(readouts) if not SMOKE else None
verdict = ('SMOKE_OK' if SMOKE else
           ('p_traj_executed' if gate_all and resolved else 'BLOCKED'))
res = {'phase': 3263, 'name': NAME, 'line': 'G', 'date': time.strftime('%Y-%m-%d'),
       'design_sha8': design_sha8, 'mode': 'smoke' if SMOKE else 'full',
       'model': 'qwen3-1.7b', 'block': BLOCK, 'NH': NH, 'D': D,
       'states': STATE_KEYS, 'gates': gates, 'gate_all': gate_all,
       'readouts': readouts, 'per_state': per_state,
       'prereg_source': 'phase3262/g5h25_gpu_arm_readiness/'
                        'gpu_trajectory_protocol_draft_v1.json (b569ce24)',
       'verdict': verdict, 'gpu_used': True,
       'note': 'first GPU measurement of the G line; collect caches resumable; '
               'restore bit-exact fp32 (2755 precedent)'}
blob = json.dumps(res, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
res_sha8 = hashlib.sha256(blob).hexdigest()[:8]
res['res_sha8'] = res_sha8
suffix = '_smoke' if SMOKE else ''
json.dump(res, open(os.path.join(OUTDIR, 'result%s.json' % suffix), 'w',
                    encoding='utf-8'), ensure_ascii=False, sort_keys=True, indent=1)
arrs = {}
for key in STATE_KEYS:
    m = per_state[key]
    tag = key.replace('@', '_s').replace('within_surface_class_permuted_token',
                                         'wscp').replace('surface_class_mass', 'scm')
    arrs['%s__top1' % tag] = np.array([m['top1_curve'][k] for k in range(NH)])
    arrs['%s__ustar_kout' % tag] = ustar_store[key]['kout'].astype(np.float32)
    arrs['%s__ustar_emb' % tag] = ustar_store[key]['emb'].astype(np.float32)
np.savez_compressed(os.path.join(OUTDIR, 'trajectory%s.npz' % suffix),
                    allow_pickle=False, **arrs)
log('VERDICT: %s res_sha8=%s' % (verdict, res_sha8))
open(os.path.join(OUTDIR, 'run_log%s.txt' % suffix), 'w',
     encoding='utf-8').write('\n'.join(LOG) + '\n')
print('FORMAL DONE' if not SMOKE else 'SMOKE DONE')
