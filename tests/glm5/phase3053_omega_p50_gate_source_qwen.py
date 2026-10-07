# -*- coding: utf-8 -*-
# Phase 3053 - Omega-P50: gate source dissection
# (run2). 3052 showed the h7 carrier head is a
# GENERIC slot (any pair's fields work, aligned
# with the DESTINATION target). Here: WHERE does
# the gate effect come from, is it content-free,
# and is h7 the only gate locus?
# (i) T2a K-block-only cross-pair transfer
# matrix (h7 K block only, V self): A_K[k,j],
# B_K[k,j]; g1 = med_A_off / diag; paired
# sign-flip permutation on D_K = B_K - A_K
# (R=2000). (ii) T2b V-block-only cross matrix
# (descriptive). (iii) full-row K-only / V-only
# diag arms (verbatim 3051 protocol) = chain
# anchors a156a/a156b vs z51 COS_K/COS_V
# (run1 correction: the h7-block diag has NO
# upstream reference; the run1 anchors
# mistakenly compared the block diag against
# the full-row arms). (iv) T6 controls: exact
# K-only diag for h0/h3 (gate-locus specificity)
# + norm-matched random K block nulls for
# h7/h0/h3 (R=200 each) - run1's h7 null
# (p=0.363) suggested content-freeness; the
# h0/h3 nulls decide head-specificity.
# (v) T3 assembly stages: canonical h7 joint
# diag replacement, last-position stage diffs
# (L34 output [pre], o_proj contribution [att],
# L35 layer output [mlp] + post-final-norm
# check); a160 pre-identity diff 0.0; a161
# post-norm consistency vs the logit cos.
# (vi) T4 attention mass transfer (descriptive).
# verdict_tree: p_h7 >= 0.05 AND null_h0 AND
# null_h3 as high as null_h7 ->
# gate_field_suscept_qwen (any K perturbation
# at L35 x FRONT works, head identity
# irrelevant); p_h7 >= 0.05 AND null_h0/h3 low
# -> gate_free_qwen (h7 = content-free gate
# locus); p_h7 < 0.05 AND g1 >= 0.8 AND
# p_perm >= 0.05 -> gate_route_generic_qwen;
# p_h7 < 0.05 AND g1 >= 0.8 ->
# gate_route_content_qwen; else ->
# gate_mixed_qwen.
# anchors: a155 re-capture (4 prompts) bit 0.0
# vs z48 + TT 0.0; a156a full-row K-only diag
# repro vs z51 COS_K diff 0.0; a156b full-row
# V-only diag repro vs z51 COS_V diff 0.0;
# a156c joint h7 diag repro vs z51 COS_H[7]
# diff 0.0; a157 sham bit identity; a158
# integrity bit-exact on the block-K/V diag +
# full-row diag bands (96 forwards, integ=True);
# a159 chain-entry max||dlg|| >= 0.05; a160
# stage-pre identity diff 0.0; a161 post-norm
# consistency max diff < 1e-3; a116c source
# seals 3044-3052.
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3053
NAME = 'omega_p50_gate_source_qwen'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
KV_HEAD = 8
HDIM = 128
R_NULL = 200
SEED_NULL = 9935
SEED_H0 = 9941
SEED_H3 = 9942
R_PERM = 2000
SEED_PERM = 9937
SEED_MAIN = 3010
T_GATE = 0.05
FRONT = 4
L_TGT = 35
HEAD_H = 7
HEADS_CTRL = (0, 3)
KS_ATT = (0, 8, 16)
BODIES = (
    'The weather was cold, so',
    'He studied every night because',
    'The experiment failed, therefore',
    'He missed the train, however',
    'The garden grows quickly while',
    'The price was high, yet',
    'She speaks French, although',
    'The road was closed, thus',)
TARGETS = ('so', 'because', 'therefore', 'however',
           'while', 'yet', 'although', 'thus')
PREFIXES = ('', 'In a formal style,',
            'In Shakespearean style,',
            'Regarding the weather,')

PREREG = {
    'mode': 'fp32 MODEL (torch.float32, eager, seed '
            '3010); capture bank loaded from the '
            'phase3048 npz (KPpost/VP/LG/LENS/TT); '
            'chain anchors vs the phase3051 npz '
            '(COS_K/COS_V/COS_H[7]); prompts '
            'reassembled locally with the 3048 '
            'tail-alignment rule; statistics on '
            'the 24 old pairs',
    'question': '3053 A main line: the h7 generic '
                'gate - is the effect carried by '
                'K routing, is it content-free '
                '(random K works as well as '
                'exact K), and is h7 the only '
                'gate locus (vs any K block at '
                'L35 x FRONT)? Where inside L35 '
                'is the direction assembled '
                '(attention OV mixture vs L35 '
                'MLP)?',
    'T2a_k_cross': 'for the h7 K block only (sl = '
                   '7*128..8*128, V self-replaced): '
                   'all 24x24 ordered pairs (k dest, '
                   'j src): insert rotK_j[L35,:,7,:] '
                   'into base_k L35 FRONT rows; '
                   'A_K[k,j] = cos(r, t_k), B_K[k,j] '
                   '= cos(r, t_j); diagonal '
                   'descriptive (no upstream block-'
                   'level reference); g1 = '
                   'med_A_off / med_diag; paired '
                   'sign-flip permutation on D_K = '
                   'B_K - A_K off-diag (R=%d, seed '
                   '%d + rep, one-sided p = P(med '
                   'D_flip >= med D))'
                   % (R_PERM, SEED_PERM),
    'T2b_v_cross': 'same for the h7 V block only '
                   '(K self-replaced): A_V/B_V, '
                   'descriptive',
    'fullrow_diag': 'chain anchors: full-row '
                    'K-only and V-only diag arms '
                    '(verbatim 3051 T2 protocol: '
                    'ALL 8 head blocks of K (or V) '
                    'at L35 x FRONT replaced, the '
                    'other field self) vs z51 '
                    'COS_K / COS_V bit-exactly '
                    '(a156a/a156b); run1 '
                    'correction: the h7-block diag '
                    'was mistakenly compared '
                    'against these full-row arms',
    'T6_controls': 'exact K-only diag for h0 and '
                   'h3 (gate-locus specificity, '
                   'descriptive) + norm-matched '
                   'random K block nulls for '
                   'h7 (seed %d), h0 (seed %d), '
                   'h3 (seed %d), R=%d each: per '
                   'draw the head-h K block at '
                   'L35 x FRONT rows replaced by '
                   'norm-matched gaussian random '
                   '(V self, other rows/blocks '
                   'self); p_hh = P(null_hh >= '
                   'med exact diag of the same '
                   'head)'
                   % (SEED_NULL, SEED_H0, SEED_H3,
                      R_NULL),
    'T3_stages': 'canonical h7 joint diag '
                 'replacement (K+V same head block, '
                 '= 3051 T3 h7 arm, a156c vs '
                 'COS_H[7]); per pair capture last-'
                 'position stage diffs: d_pre (L34 '
                 'output, must be identical bit - '
                 'a160), d_att (o_proj output '
                 'contribution), d_mlp (L35 layer '
                 'output), plus the post-final-norm '
                 'delta z = W_U(norm(post_i) - '
                 'norm(post_b)) consistent with '
                 'the logit cos (a161, < 1e-3); '
                 'descriptive assembly locus; '
                 'caveat: pre-norm stage cosines '
                 'are confounded by the final '
                 'RMSNorm projection',
    'T4_attn_transfer': 'descriptive: destinations '
                        'k in (0,8,16) (same body, '
                        'one base prompt), sources '
                        'j = 0..23, K-block-only '
                        'cross replacement with '
                        'output_attentions=True; '
                        'GQA group 7 attention mass '
                        'from the last position onto '
                        'FRONT rows; Delta-mass vs '
                        'A_K[k, j] Pearson corr '
                        'across j',
    'verdict_tree': 'p_h7 >= 0.05 AND '
                    'med(null_h0) >= 0.8*med(null_'
                    'h7) AND med(null_h3) >= '
                    '0.8*med(null_h7) -> '
                    'gate_field_suscept_qwen; '
                    'p_h7 >= 0.05 AND med(null_h0) '
                    '< 0.5*med(null_h7) AND '
                    'med(null_h3) < 0.5*med(null_'
                    'h7) -> gate_free_qwen; '
                    'p_h7 >= 0.05 (mixed null '
                    'pattern) -> gate_mixed_qwen; '
                    'p_h7 < 0.05 AND g1 >= 0.8 AND '
                    'p_perm_K >= 0.05 -> '
                    'gate_route_generic_qwen; '
                    'p_h7 < 0.05 AND g1 >= 0.8 -> '
                    'gate_route_content_qwen; '
                    'else -> gate_mixed_qwen; '
                    'single branch',
    'anchors': 'a155 sampled re-capture (4 prompts '
               '0/9/17/31) KPpre/KPpost/VP/LG bit '
               '0.0 vs z48 + TT diff 0.0; a156a '
               'full-row K-only diag diff 0.0 vs '
               'z51 COS_K; a156b full-row V-only '
               'diag diff 0.0 vs z51 COS_V; a156c '
               'joint h7 diag diff 0.0 vs z51 '
               'COS_H[7]; a157 sham self-'
               'replacement bit identity; a158 '
               'integrity bit-exact (mod == '
               'where(mask,repl,orig)) on the '
               'block-K/V diag + full-row diag '
               'bands (96 forwards, integ=True); '
               'a159 chain-entry max||dlg|| >= %g; '
               'a160 stage-pre identity diff 0.0; '
               'a161 post-norm consistency < 1e-3; '
               'a116c source seals 3044-3052'
               % T_GATE,
    'control': 'norm-matched random K restricted '
               'to one head block at the same rows '
               'as the exact field (h7/h0/h3); '
               'paired within-entry contrast for '
               'the permutation',
    'statistics_discipline': 'obs and null on the '
                             'SAME scale (raw cos, '
                             'same 24 pairs, same '
                             'replacement machinery, '
                             'null randomized only '
                             'inside one head K block '
                             'rows); permutation null '
                             'is a paired label swap '
                             'on the same off-diag '
                             'entries; verdict in one '
                             'branch',
    'corrections': 'run1 (393.6s) non-'
                   'authoritative: a156a/a156b '
                   'FAILED (diff 1.310/1.392) - '
                   'the h7-block-only diag was '
                   'mistakenly compared against '
                   'the z51 FULL-ROW COS_K/COS_V '
                   '(no upstream reference exists '
                   'for the block diag); the '
                   'mechanism itself was correct '
                   '(a156c joint diag bit 0.0). '
                   'run1 signal carried into run2: '
                   'the h7 K-block restricted null '
                   'was NOT significant (null med '
                   '0.6609 vs obs 0.6871, p '
                   '0.363) - random K opens the '
                   'gate; run2 adds the full-row '
                   'diag chain anchors and the '
                   'h0/h3 exact + random controls '
                   'to decide content-freeness vs '
                   'field susceptibility. run2 '
                   '(931.3s) non-authoritative: '
                   'a156a/a156b scalar-index bug - '
                   'COS_K51[h]/COS_V51[h] indexed '
                   'pair 7 of the (24,) vectors '
                   'instead of the full vector; '
                   'offline npz verification: '
                   'corrected diff 0.0 (med 0.670488/'
                   '-0.340863 bit-exact vs z51), '
                   'all other anchors passed '
                   '(a155/a156c/a157/a158x96/a159/'
                   'a160/a161). run3 authoritative '
                   'if anchors pass.',
}

lines = []


def log(msg):
    lines.append(str(msg))
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')
    print(msg)


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


os.makedirs(OUT, exist_ok=True)
if os.path.exists(LOG):
    os.remove(LOG)
for fn in ('execution.json', 'result.json', 'seal.json',
           NAME + '.npz'):
    p = os.path.join(OUT, fn)
    if os.path.exists(p):
        os.remove(p)
t0 = time.time()
created = time.strftime('%Y-%m-%d %H:%M:%S')
execution = {'phase': PHASE, 'name': NAME,
             'created': created, 'prereg': PREREG}
with open(os.path.join(OUT, 'execution.json'), 'w',
          encoding='utf-8') as f:
    json.dump(execution, f, ensure_ascii=False, indent=1)
log('execution.json written (prereg frozen) %s' % created)

torch.manual_seed(SEED_MAIN)
np.random.seed(SEED_MAIN)

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float32,
    attn_implementation='eager').to('cuda').eval()
layers = model.model.layers
assert len(layers) == NL
assert int(model.config.num_key_value_heads) == KV_HEAD
assert int(model.config.num_attention_heads) == 32
rot = model.model.rotary_emb
INV_FREQ = rot.inv_freq.detach().cpu().numpy() \
    .astype(np.float32)
assert INV_FREQ.shape == (64,), INV_FREQ.shape
assert float(rot.attention_scaling) == 1.0
NVOC = int(model.config.vocab_size)
log('model loaded fp32 (vocab=%d)' % NVOC)


def rot_apply(x, delta):
    """Rotate (...,8,128) by delta*inv_freq
    (NeoX half-pairing, fp32)."""
    ang = INV_FREQ * np.float32(delta)
    emb = np.concatenate([ang, ang]).astype(
        np.float32)
    c = np.cos(emb)[None, None, :]
    s = np.sin(emb)[None, None, :]
    x1 = x[..., :64]
    x2 = x[..., 64:]
    rh = np.concatenate([-x2, x1], axis=-1)
    return x * c + rh * s


stateKn = {li: {'repl': None, 'mask': None}
           for li in range(NL)}
stateV = {li: {'repl': None, 'mask': None}
          for li in range(NL)}
capKn = {li: {'rec': False, 'orig': None,
              'mod': None} for li in range(NL)}
capV = {li: {'rec': False, 'orig': None,
             'mod': None} for li in range(NL)}
capKpre = {li: {'rec': False, 'orig': None}
           for li in range(NL)}
cap_stage = {'pre': {'rec': False, 'v': None},
             'op': {'rec': False, 'v': None},
             'post': {'rec': False, 'v': None}}


def hook_norm(st, cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        if st['repl'] is not None:
            out[0][st['mask']] = st['repl']
        if cp['rec']:
            cp['mod'] = out[0].detach().clone()
        return out
    return h


def hook_v(st, cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        if st['repl'] is not None:
            out[0][st['mask']] = st['repl']
        if cp['rec']:
            cp['mod'] = out[0].detach().clone()
        return out
    return h


def hook_kpre(cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        return out
    return h


def hook_stage(cp):
    """Capture the LAST-position vector of a
    bare-tensor output (batch dim stripped by
    indexing [0])."""
    def h(module, inp, out):
        if cp['rec']:
            t = out[0] if isinstance(out, tuple) \
                else out
            cp['v'] = t[0][-1].detach().clone()
        return out
    return h


for li in range(NL):
    layers[li].self_attn.k_norm \
        .register_forward_hook(hook_norm(
            stateKn[li], capKn[li]))
    layers[li].self_attn.v_proj \
        .register_forward_hook(hook_v(
            stateV[li], capV[li]))
    layers[li].self_attn.k_proj \
        .register_forward_hook(hook_kpre(
            capKpre[li]))
layers[34].register_forward_hook(
    hook_stage(cap_stage['pre']))
layers[35].self_attn.o_proj.register_forward_hook(
    hook_stage(cap_stage['op']))
layers[35].register_forward_hook(
    hook_stage(cap_stage['post']))


def reset_all():
    for li in range(NL):
        stateKn[li]['repl'] = None
        stateKn[li]['mask'] = None
        stateV[li]['repl'] = None
        stateV[li]['mask'] = None
        capKn[li]['rec'] = False
        capV[li]['rec'] = False
        capKpre[li]['rec'] = False
    for kk in cap_stage:
        cap_stage[kk]['rec'] = False
        cap_stage[kk]['v'] = None


def forward_cap(ids):
    reset_all()
    for li in range(NL):
        capKn[li]['rec'] = True
        capV[li]['rec'] = True
        capKpre[li]['rec'] = True
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True)
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    n = len(ids)
    kp = np.stack([capKpre[li]['orig']
                   .cpu().numpy()
                   for li in range(NL)])
    kn = np.stack([capKn[li]['orig'].reshape(
        n, HDIM * 8).cpu().numpy()
        for li in range(NL)])
    vp = np.stack([capV[li]['orig'].cpu().numpy()
                   for li in range(NL)])
    reset_all()
    return lg, kp, kn, vp


def forward_run(ids, replK=None, maskK=None,
                replV=None, maskV=None, integ=False,
                attn=False):
    reset_all()
    rk = rv = None
    mk = mv = None
    if replK is not None:
        assert maskK is not None
        rk = torch.tensor(np.ascontiguousarray(
            replK.reshape(NL, -1, 8, HDIM)),
            dtype=torch.float32, device='cuda')
        mk = torch.tensor(np.asarray(maskK),
                          device='cuda')
        for li in range(NL):
            stateKn[li]['repl'] = rk[li]
            stateKn[li]['mask'] = mk
    if replV is not None:
        assert maskV is not None
        rv = torch.tensor(np.ascontiguousarray(
            replV),
            dtype=torch.float32, device='cuda')
        mv = torch.tensor(np.asarray(maskV),
                          device='cuda')
        for li in range(NL):
            stateV[li]['repl'] = rv[li]
            stateV[li]['mask'] = mv
    if integ:
        for li in range(NL):
            capKn[li]['rec'] = True
            capV[li]['rec'] = True
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True,
                    output_attentions=attn)
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    res = {'lg': lg}
    if attn:
        res['aw'] = out.attentions[L_TGT][0] \
            .detach().double().cpu().numpy()
    if integ:
        e = 0.0
        for li in range(NL):
            if rk is not None:
                comb = capKn[li]['orig'].clone()
                comb[mk] = rk[li]
                e = max(e, float(
                    (capKn[li]['mod'] - comb)
                    .abs().max()))
            if rv is not None:
                comb = capV[li]['orig'].clone()
                comb[mv] = rv[li]
                e = max(e, float(
                    (capV[li]['mod'] - comb)
                    .abs().max()))
        res['bit'] = e
    reset_all()
    return res


def forward_stage(ids, replK, maskK, replV, maskV):
    """Joint h7 diag replacement with stage
    capture; returns logits + last-position
    stage vectors (pre, op, post) and the
    post-final-norm post vector."""
    reset_all()
    rk = torch.tensor(np.ascontiguousarray(
        replK.reshape(NL, -1, 8, HDIM)),
        dtype=torch.float32, device='cuda')
    mk = torch.tensor(np.asarray(maskK),
                      device='cuda')
    rv = torch.tensor(np.ascontiguousarray(
        replV),
        dtype=torch.float32, device='cuda')
    mv = torch.tensor(np.asarray(maskV),
                      device='cuda')
    for li in range(NL):
        stateKn[li]['repl'] = rk[li]
        stateKn[li]['mask'] = mk
        stateV[li]['repl'] = rv[li]
        stateV[li]['mask'] = mv
    for kk in cap_stage:
        cap_stage[kk]['rec'] = True
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True)
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    pre = cap_stage['pre']['v'].double() \
        .cpu().numpy()
    op = cap_stage['op']['v'].double() \
        .cpu().numpy()
    post_t = cap_stage['post']['v']
    post = post_t.double().cpu().numpy()
    with torch.no_grad():
        postn_t = model.model.norm(
            post_t.unsqueeze(0))
    postn = postn_t[0].double().cpu().numpy()
    reset_all()
    return lg, pre, op, post, postn


# ---------- chain: load capture banks ----------
z48 = np.load(os.path.join(
    BASE, 'phase3048',
    'omega_p45_kvpos_full_replay_qwen',
    'omega_p45_kvpos_full_replay_qwen.npz'),
    allow_pickle=True)
KPpre = z48['KPpre']
KPpost = z48['KPpost']
VP = z48['VP']
LG = z48['LG']
LENS = z48['LENS']
TT = z48['TT']
log('z48 bank loaded: KPpre%s KPpost%s VP%s LG%s'
    % (KPpre.shape, KPpost.shape, VP.shape,
       LG.shape))
z51 = np.load(os.path.join(
    BASE, 'phase3051',
    'omega_p48_l35_anatomy_qwen',
    'omega_p48_l35_anatomy_qwen.npz'),
    allow_pickle=True)
COS_K51 = z51['COS_K']
COS_V51 = z51['COS_V']
COS_H51 = z51['COS_H']
log('z51 anchors loaded: COS_K/COS_V/COS_H%s'
    % (COS_K51.shape,))

seal_detail = []
for ph, nm in (
        (3044, 'omega_p41_field_axis_injection_'
               'qwen'),
        (3045, 'omega_p42_l20_axis_anatomy_qwen'),
        (3046, 'omega_p43_kfield_injection_qwen'),
        (3047, 'omega_p44_kv_joint_replay_qwen'),
        (3048, 'omega_p45_kvpos_full_replay_qwen'),
        (3049, 'omega_p46_kvload_localization_'
               'qwen'),
        (3050, 'omega_p47_kvdeep_dissection_'
               'qwen'),
        (3051, 'omega_p48_l35_anatomy_qwen'),
        (3052, 'omega_p49_kvhead_identity_qwen')):
    d = os.path.join(BASE, 'phase%d' % ph, nm)
    with open(os.path.join(d, 'seal.json'),
              encoding='utf-8') as f:
        sealj = json.load(f)
    with open(os.path.join(d, 'result.json'),
              'rb') as f:
        s = hashlib.sha256(f.read()).hexdigest()[:8]
    seal_detail.append(bool(
        s == sealj['result_sha256_8']))
a116_ok = bool(seal_detail) and all(seal_detail)
log('a116c source seals ok=%s' % a116_ok)

# ---------- assemble prompts (3048 verbatim) ----------
word_tok = {}
for w in set(TARGETS):
    wi = tok(' ' + w, add_special_tokens=False)[
        'input_ids']
    assert len(wi) == 1, (w, wi)
    word_tok[w] = int(wi[0])

assembled = []
for bi in range(len(BODIES)):
    for ci in range(len(PREFIXES)):
        s = (PREFIXES[ci] + ' ' + BODIES[bi]) \
            if PREFIXES[ci] else BODIES[bi]
        ids = [int(x) for x in tok(
            s, add_special_tokens=False)[
            'input_ids']]
        t = word_tok[TARGETS[bi]]
        assert ids.count(t) == 1, (bi, ci)
        assembled.append({'ids': ids,
                          'pos': ids.index(t),
                          'cond': ci, 'body': bi})
n_pr = len(assembled)
assert n_pr == 32
idx_of = {}
for i in range(n_pr):
    idx_of[(assembled[i]['cond'],
            assembled[i]['body'])] = i
for i in range(n_pr):
    ci = assembled[i]['cond']
    if ci == 0:
        assembled[i]['off'] = 0
    else:
        bid = assembled[idx_of[(0,
            assembled[i]['body'])]]['ids']
        pid = assembled[i]['ids']
        off = len(pid) - len(bid)
        assert off > 0, i
        assert list(pid[off + 1:]) \
            == list(bid[1:]), i
        w0b = tok.decode([bid[0]]).strip()
        w0p = tok.decode([pid[off]]).strip()
        assert w0b == w0p, (i, w0b, w0p)
        assembled[i]['off'] = off
for i in range(n_pr):
    assert LENS[i] == len(assembled[i]['ids']), i
log('assembled %d prompts (lens consistent '
    'with z48)' % n_pr)

# a155: sampled re-capture, bit-exact vs z48
a155_diff = 0.0
for si in (0, 9, 17, 31):
    lg2, kp2, kn2, vp2 = forward_cap(
        assembled[si]['ids'])
    n2 = LENS[si]
    a155_diff = max(a155_diff, float(np.max(
        np.abs(LG[si] - lg2))))
    a155_diff = max(a155_diff, float(np.max(
        np.abs(KPpre[si, :, :n2, :] - kp2))))
    a155_diff = max(a155_diff, float(np.max(
        np.abs(KPpost[si, :, :n2, :] - kn2))))
    a155_diff = max(a155_diff, float(np.max(
        np.abs(VP[si, :, :n2, :] - vp2))))
a155_diff = float(a155_diff)

cidx = []
bidx = []
for ci in (1, 2, 3):
    for bi in range(len(BODIES)):
        cidx.append(ci)
        bidx.append(bi)
cidx = np.array(cidx)
bidx = np.array(bidx)
NP_ = 24

t_targets = {}
for b in range(len(BODIES)):
    for c in (1, 2, 3):
        t_targets[(b, c)] = LG[idx_of[(c, b)]] \
            - LG[idx_of[(0, b)]]
TTl = np.stack([t_targets[(int(bidx[k]),
                          int(cidx[k]))]
                for k in range(NP_)])
a_tt = float(np.max(np.abs(TTl - TT)))
a155_ok = bool(a155_diff == 0.0 and a_tt == 0.0)
log('a155 re-capture diff=%.3e TT diff=%.3e ok=%s'
    % (a155_diff, a_tt, a155_ok))

# a157: sham self-replacement (bit identity)
b0 = int(bidx[0])
base0 = idx_of[(0, b0)]
ids0 = assembled[base0]['ids']
n0 = len(ids0)
selfK = KPpost[base0, :, :n0, :].copy()
selfV = VP[base0, :, :n0, :].copy()
m_all = np.ones(n0, dtype=bool)
res = forward_run(ids0, replK=selfK, maskK=m_all,
                  replV=selfV, maskV=m_all,
                  integ=True)
a157_diff = float(np.max(np.abs(res['lg']
                                - LG[base0])))
a157_diff = float(max(a157_diff, res['bit']))
a157_ok = bool(a157_diff == 0.0)
log('a157 sham self-replacement diff=%.3e ok=%s'
    % (a157_diff, a157_ok))


def front_rows(nb):
    return list(range(min(FRONT, nb)))


def l35_fields(k):
    """Per pair: base self fields (NL, nb, 1024)
    plus the L35 pref fields for the FRONT rows
    (K rotated by off, V raw)."""
    b = int(bidx[k])
    c = int(cidx[k])
    base_i = idx_of[(0, b)]
    pref_i = idx_of[(c, b)]
    nb = int(LENS[base_i])
    off = assembled[pref_i]['off']
    repK = KPpost[base_i, :, :nb, :].copy()
    repV = VP[base_i, :, :nb, :].copy()
    rows = front_rows(nb)
    ridx = np.asarray(rows)
    src = KPpost[pref_i][:, off + ridx, :] \
        .reshape(NL, len(rows), 8, HDIM)
    rotK = rot_apply(src, off)  # (NL,nfr,8,128)
    srcV = VP[pref_i][:, off + ridx, :]
    return repK, repV, rotK, srcV, rows, base_i


def cos_frac(lg_base, r, t):
    nr = float(np.linalg.norm(r))
    nt = float(np.linalg.norm(t))
    cs = float(r @ t) / (nr * nt) \
        if nr > 1e-12 and nt > 1e-12 else 0.0
    fr = nr / nt if nt > 1e-12 else float('nan')
    return cs, fr


h = HEAD_H
sl = slice(h * HDIM, (h + 1) * HDIM)
NFR_ALL = None
for k in range(NP_):
    nb = int(LENS[idx_of[(0, int(bidx[k]))]])
    if NFR_ALL is None:
        NFR_ALL = len(front_rows(nb))
    assert len(front_rows(nb)) == NFR_ALL, k
assert NFR_ALL == 4

a158_fail = 0
a158_checked = 0
rec_dlg = []

# ---------- T2a K-block-only cross matrix ----------
log('=== T2a K-block-only cross (h%d) ===' % h)
MAT_AK = np.zeros((NP_, NP_))
MAT_BK = np.zeros((NP_, NP_))
for k in range(NP_):
    b = int(bidx[k])
    repK_k, repV_k, _, _, rows_k, base_i \
        = l35_fields(k)
    ids_b = assembled[base_i]['ids']
    nb = int(LENS[base_i])
    mask = np.ones(nb, dtype=bool)
    for j in range(NP_):
        _, _, rotK_j, _, rows_j, _ \
            = l35_fields(j)
        rK = repK_k.copy()
        rK[L_TGT, rows_k, sl] \
            = rotK_j[L_TGT, :, h, :]
        integ = bool(j == k)
        res = forward_run(ids_b, replK=rK,
                          maskK=mask,
                          replV=repV_k, maskV=mask,
                          integ=integ)
        if integ:
            a158_checked += 1
            if res.get('bit', 0.0) != 0.0:
                a158_fail += 1
        r = res['lg'] - LG[base_i]
        t_k = t_targets[(b, int(cidx[k]))]
        cs, _ = cos_frac(LG[base_i], r, t_k)
        MAT_AK[k, j] = cs
        b_j = int(bidx[j])
        t_j = t_targets[(b_j, int(cidx[j]))]
        csj, _ = cos_frac(LG[base_i], r, t_j)
        MAT_BK[k, j] = csj
        rec_dlg.append(float(np.linalg.norm(r)))
    log('  k=%02d: A diag %.4f' % (k, MAT_AK[k, k]))

# ---------- T2b V-block-only cross matrix ----------
log('=== T2b V-block-only cross (h%d) ===' % h)
MAT_AV = np.zeros((NP_, NP_))
MAT_BV = np.zeros((NP_, NP_))
for k in range(NP_):
    b = int(bidx[k])
    repK_k, repV_k, _, _, rows_k, base_i \
        = l35_fields(k)
    ids_b = assembled[base_i]['ids']
    nb = int(LENS[base_i])
    mask = np.ones(nb, dtype=bool)
    for j in range(NP_):
        _, _, _, srcV_j, rows_j, _ \
            = l35_fields(j)
        rV = repV_k.copy()
        svj = srcV_j[L_TGT].reshape(
            len(rows_j), KV_HEAD, HDIM)
        rV[L_TGT, rows_k, sl] = svj[:, h, :]
        integ = bool(j == k)
        res = forward_run(ids_b, replK=repK_k,
                          maskK=mask,
                          replV=rV, maskV=mask,
                          integ=integ)
        if integ:
            a158_checked += 1
            if res.get('bit', 0.0) != 0.0:
                a158_fail += 1
        r = res['lg'] - LG[base_i]
        t_k = t_targets[(b, int(cidx[k]))]
        cs, _ = cos_frac(LG[base_i], r, t_k)
        MAT_AV[k, j] = cs
        b_j = int(bidx[j])
        t_j = t_targets[(b_j, int(cidx[j]))]
        csj, _ = cos_frac(LG[base_i], r, t_j)
        MAT_BV[k, j] = csj
        rec_dlg.append(float(np.linalg.norm(r)))

# ---------- full-row diag arms (chain anchors) ----------
log('=== full-row K/V-only diag (a156a/b) ===')
COS_KFR = np.zeros(NP_)
COS_VFR = np.zeros(NP_)
for k in range(NP_):
    b = int(bidx[k])
    c = int(cidx[k])
    repK, repV, rotK, srcV, rows, base_i \
        = l35_fields(k)
    ids_b = assembled[base_i]['ids']
    nb = int(LENS[base_i])
    mask = np.ones(nb, dtype=bool)
    # full-row K-only (verbatim 3051 T2)
    rK = repK.copy()
    rK[L_TGT, rows, :] = rotK[L_TGT] \
        .reshape(len(rows), HDIM * KV_HEAD)
    res = forward_run(ids_b, replK=rK, maskK=mask,
                      replV=repV, maskV=mask,
                      integ=True)
    a158_checked += 1
    if res.get('bit', 0.0) != 0.0:
        a158_fail += 1
    r = res['lg'] - LG[base_i]
    t_k = t_targets[(b, c)]
    cs, _ = cos_frac(LG[base_i], r, t_k)
    COS_KFR[k] = cs
    rec_dlg.append(float(np.linalg.norm(r)))
    # full-row V-only (verbatim 3051 T2)
    rV = repV.copy()
    rV[L_TGT, rows, :] = srcV[L_TGT]
    res = forward_run(ids_b, replK=repK,
                      maskK=mask, replV=rV,
                      maskV=mask, integ=True)
    a158_checked += 1
    if res.get('bit', 0.0) != 0.0:
        a158_fail += 1
    r = res['lg'] - LG[base_i]
    cs, _ = cos_frac(LG[base_i], r, t_k)
    COS_VFR[k] = cs
    rec_dlg.append(float(np.linalg.norm(r)))
a156a_diff = float(np.max(np.abs(COS_KFR
                                 - COS_K51)))
a156a_ok = bool(a156a_diff == 0.0)
a156b_diff = float(np.max(np.abs(COS_VFR
                                 - COS_V51)))
a156b_ok = bool(a156b_diff == 0.0)
log('a156a full-row K diag vs z51 COS_K: '
    'diff=%.3e ok=%s' % (a156a_diff, a156a_ok))
log('a156b full-row V diag vs z51 COS_V: '
    'diff=%.3e ok=%s' % (a156b_diff, a156b_ok))

# ---------- T6 controls: exact K-only diag ----------
log('=== T6 exact K-only diag (h0/h3) ===')
COS_KCTRL = {}
for hh in HEADS_CTRL:
    slh = slice(hh * HDIM, (hh + 1) * HDIM)
    cs_h = np.zeros(NP_)
    for k in range(NP_):
        b = int(bidx[k])
        repK_k, repV_k, rotK_k, _, rows_k, base_i \
            = l35_fields(k)
        ids_b = assembled[base_i]['ids']
        nb = int(LENS[base_i])
        mask = np.ones(nb, dtype=bool)
        rK = repK_k.copy()
        rK[L_TGT, rows_k, slh] \
            = rotK_k[L_TGT, :, hh, :]
        res = forward_run(ids_b, replK=rK,
                          maskK=mask,
                          replV=repV_k, maskV=mask)
        r = res['lg'] - LG[base_i]
        t_k = t_targets[(b, int(cidx[k]))]
        cs, _ = cos_frac(LG[base_i], r, t_k)
        cs_h[k] = cs
        rec_dlg.append(float(np.linalg.norm(r)))
    COS_KCTRL['h%d' % hh] = cs_h
    log('  h%d exact K-only diag med=%.4f'
        % (hh, float(np.median(cs_h))))

# ---------- T3 assembly stages ----------
log('=== T3 stage diffs (joint h%d diag) ===' % h)
WU = model.lm_head.weight.detach()  # (V, 2560)
wud = WU.double().cpu().numpy()
STAGE = np.zeros((NP_, 4))  # pre, att, mlp, mlpn
COS_H3 = np.zeros(NP_)
pre_id_max = 0.0
postn_dev = 0.0
for k in range(NP_):
    b = int(bidx[k])
    repK_k, repV_k, rotK_k, srcV_k, rows_k, base_i \
        = l35_fields(k)
    ids_b = assembled[base_i]['ids']
    nb = int(LENS[base_i])
    mask = np.ones(nb, dtype=bool)
    # base (self replacement) stages
    _, pre_b, op_b, post_b, postn_b = forward_stage(
        ids_b, repK_k, mask, repV_k, mask)
    # intervened: h7 joint (K+V head block)
    rK = repK_k.copy()
    rV = repV_k.copy()
    rK[L_TGT, rows_k, sl] = rotK_k[L_TGT, :, h, :]
    svk = srcV_k[L_TGT].reshape(
        len(rows_k), KV_HEAD, HDIM)
    rV[L_TGT, rows_k, sl] = svk[:, h, :]
    lg_i, pre_i, op_i, post_i, postn_i \
        = forward_stage(ids_b, rK, mask, rV, mask)
    r = lg_i - LG[base_i]
    t_k = t_targets[(b, int(cidx[k]))]
    COS_H3[k] = cos_frac(LG[base_i], r, t_k)[0]
    rec_dlg.append(float(np.linalg.norm(r)))
    dp = pre_i - pre_b
    pre_id_max = max(pre_id_max,
                     float(np.max(np.abs(dp))))
    da = (op_i - op_b) + dp
    dm = post_i - post_b
    dn = postn_i - postn_b
    for si, d in enumerate((dp, da, dm, dn)):
        z = wud @ d
        nz = float(np.linalg.norm(z))
        nt = float(np.linalg.norm(t_k))
        STAGE[k, si] = float(z @ t_k) \
            / (nz * nt) \
            if nz > 1e-12 and nt > 1e-12 else 0.0
    postn_dev = max(postn_dev, abs(
        STAGE[k, 3] - COS_H3[k]))
a160_ok = bool(pre_id_max == 0.0)
a161_ok = bool(postn_dev < 1e-3)
log('a160 stage-pre identity max diff=%.3e ok=%s'
    % (pre_id_max, a160_ok))
log('a161 post-norm vs logit cos dev max=%.3e '
    'ok=%s' % (postn_dev, a161_ok))
med_pre = float(np.median(STAGE[:, 0]))
med_att = float(np.median(STAGE[:, 1]))
med_mlp = float(np.median(STAGE[:, 2]))
med_mlpn = float(np.median(STAGE[:, 3]))
log('T3: stage cos med pre=%.4f att=%.4f '
    'mlp=%.4f mlpn=%.4f'
    % (med_pre, med_att, med_mlp, med_mlpn))
a156c_diff = float(np.max(np.abs(
    COS_H3 - COS_H51[h])))
a156c_ok = bool(a156c_diff == 0.0)
log('a156c joint diag repro vs z51 COS_H[7]: '
    'diff=%.3e ok=%s' % (a156c_diff, a156c_ok))

# ---------- T4 attention mass transfer ----------
log('=== T4 attention mass transfer ===')
corr_rows = []
for k in KS_ATT:
    b = int(bidx[k])
    repK_k, repV_k, _, _, rows_k, base_i \
        = l35_fields(k)
    ids_b = assembled[base_i]['ids']
    nb = int(LENS[base_i])
    mask = np.ones(nb, dtype=bool)
    resb = forward_run(ids_b, replK=repK_k,
                       maskK=mask, replV=repV_k,
                       maskV=mask, attn=True)
    m_base = float(
        resb['aw'][4 * h:4 * h + 4, -1, :][:, rows_k]
        .sum())
    Am = []
    Av = []
    for j in range(NP_):
        _, _, rotK_j, _, rows_j, _ \
            = l35_fields(j)
        rK = repK_k.copy()
        rK[L_TGT, rows_k, sl] \
            = rotK_j[L_TGT, :, h, :]
        res = forward_run(ids_b, replK=rK,
                          maskK=mask,
                          replV=repV_k, maskV=mask,
                          attn=True)
        m_int = float(
            res['aw'][4 * h:4 * h + 4, -1, :]
            [:, rows_k].sum())
        Am.append(m_int - m_base)
        Av.append(MAT_AK[k, j])
    Am = np.array(Am)
    Av = np.array(Av)
    if float(np.std(Am)) > 1e-12 \
            and float(np.std(Av)) > 1e-12:
        c = float(np.corrcoef(Am, Av)[0, 1])
    else:
        c = float('nan')
    corr_rows.append((k, m_base, float(np.median(Am)),
                      c))
    log('  k=%02d: base mass g7=%.4f dmass med='
        '%.4f corr(dmass, A_K)=%.3f'
        % (k, m_base, float(np.median(Am)), c))


# ---------- T5 block random nulls ----------
def block_null(hh, seed0, R):
    """Norm-matched random K block hh at
    L35 x FRONT rows (V self); stat = 24-pair
    median cos."""
    slh = slice(hh * HDIM, (hh + 1) * HDIM)
    nullv = np.zeros(R)
    for mc in range(R):
        rng = np.random.default_rng(seed0 + mc)
        csm = np.zeros(NP_)
        for k in range(NP_):
            b = int(bidx[k])
            repK_k, repV_k, rotK_k, _, rows_k, \
                base_i = l35_fields(k)
            ids_b = assembled[base_i]['ids']
            nb = int(LENS[base_i])
            NK = np.linalg.norm(
                rotK_k[L_TGT, :, hh, :], axis=1)
            RK = rng.standard_normal(
                (len(rows_k), HDIM)) \
                .astype(np.float32)
            RK *= (NK / np.linalg.norm(RK, axis=1)
                   )[:, None]
            rK = repK_k.copy()
            rK[L_TGT, rows_k, slh] = RK
            mask = np.ones(nb, dtype=bool)
            res = forward_run(ids_b, replK=rK,
                              maskK=mask,
                              replV=repV_k,
                              maskV=mask)
            r = res['lg'] - LG[base_i]
            t_k = t_targets[(b, int(cidx[k]))]
            cs, _ = cos_frac(LG[base_i], r, t_k)
            csm[k] = cs
            rec_dlg.append(float(
                np.linalg.norm(r)))
        nullv[mc] = float(np.median(csm))
        if (mc + 1) % 50 == 0:
            log('  null h%d mc %d/%d (med=%.4f)'
                % (hh, mc + 1, R, nullv[mc]))
    return nullv


log('=== T5 block nulls (R=%d each) ===' % R_NULL)
nullH7 = block_null(7, SEED_NULL, R_NULL)
nullH0 = block_null(0, SEED_H0, R_NULL)
nullH3 = block_null(3, SEED_H3, R_NULL)
med_diag_K = float(np.median(np.diag(MAT_AK)))
p_h7 = float((1 + int(np.sum(nullH7 >= med_diag_K)))
             / (R_NULL + 1))
med0_exact = float(np.median(
    COS_KCTRL['h0']))
med3_exact = float(np.median(
    COS_KCTRL['h3']))
p_h0 = float((1 + int(np.sum(nullH0 >= med0_exact)))
             / (R_NULL + 1))
p_h3 = float((1 + int(np.sum(nullH3 >= med3_exact)))
             / (R_NULL + 1))
log('h7: obs=%.4f null med=%.4f max=%.4f p=%.5f'
    % (med_diag_K, float(np.median(nullH7)),
       float(nullH7.max()), p_h7))
log('h0: obs=%.4f null med=%.4f max=%.4f p=%.5f'
    % (med0_exact, float(np.median(nullH0)),
       float(nullH0.max()), p_h0))
log('h3: obs=%.4f null med=%.4f max=%.4f p=%.5f'
    % (med3_exact, float(np.median(nullH3)),
       float(nullH3.max()), p_h3))

# off-diag stats + permutation (K-only)
off = ~np.eye(NP_, dtype=bool)
med_Aoff_K = float(np.median(MAT_AK[off]))
med_Boff_K = float(np.median(MAT_BK[off]))
D_K = (MAT_BK - MAT_AK)[off]
medD_K = float(np.median(D_K))
reps = np.empty(R_PERM)
for rep in range(R_PERM):
    rng = np.random.default_rng(SEED_PERM + rep)
    sgn = rng.choice((-1.0, 1.0), size=D_K.shape)
    reps[rep] = float(np.median(D_K * sgn))
p_perm_K = float(
    (1 + int(np.sum(reps >= medD_K)))
    / (R_PERM + 1))
g1 = med_Aoff_K / med_diag_K \
    if abs(med_diag_K) > 1e-12 else float('nan')
log('T2a: diag med=%.4f A_off=%.4f B_off=%.4f | '
    'D med=%.4f p_perm=%.5f | g1=%.4f'
    % (med_diag_K, med_Aoff_K, med_Boff_K,
       medD_K, p_perm_K, g1))
log('T2b: diag med=%.4f A_off=%.4f B_off=%.4f'
    % (float(np.median(np.diag(MAT_AV))),
       float(np.median(MAT_AV[off])),
       float(np.median(MAT_BV[off]))))

# a158 / a159
a158_ok = bool(a158_fail == 0
               and a158_checked >= 1)
max_dlg = float(np.max(np.array(rec_dlg)))
a159_ok = bool(max_dlg >= T_GATE)
log('a158 integrity fails=%d checked=%d ok=%s | '
    'a159 max||dlg||=%.4f ok=%s'
    % (a158_fail, a158_checked, a158_ok,
       max_dlg, a159_ok))

# ---------- verdict ----------
mn7 = float(np.median(nullH7))
mn0 = float(np.median(nullH0))
mn3 = float(np.median(nullH3))
high0 = mn0 >= 0.8 * mn7
high3 = mn3 >= 0.8 * mn7
low0 = mn0 < 0.5 * mn7
low3 = mn3 < 0.5 * mn7
if p_h7 >= 0.05:
    if high0 and high3:
        verdict = 'gate_field_suscept_qwen'
    elif low0 and low3:
        verdict = 'gate_free_qwen'
    else:
        verdict = 'gate_mixed_qwen'
elif g1 >= 0.8 and p_perm_K >= 0.05:
    verdict = 'gate_route_generic_qwen'
elif g1 >= 0.8:
    verdict = 'gate_route_content_qwen'
else:
    verdict = 'gate_mixed_qwen'
log('VERDICT: %s (p_h7=%.5f mn7=%.4f mn0=%.4f '
    'mn3=%.4f g1=%.4f p_perm=%.5f)'
    % (verdict, p_h7, mn7, mn0, mn3, g1,
       p_perm_K))

anchor_core_ok = bool(a116_ok and a155_ok
                      and a156a_ok and a156b_ok
                      and a156c_ok and a157_ok
                      and a158_ok and a159_ok
                      and a160_ok and a161_ok)
log('anchors core ok=%s' % anchor_core_ok)

elapsed = time.time() - t0
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(npz_path,
         MAT_AK=MAT_AK, MAT_BK=MAT_BK,
         MAT_AV=MAT_AV, MAT_BV=MAT_BV,
         STAGE=STAGE, COS_H3=COS_H3,
         COS_KFR=COS_KFR, COS_VFR=COS_VFR,
         COS_KCTRL_h0=COS_KCTRL['h0'],
         COS_KCTRL_h3=COS_KCTRL['h3'],
         nullH7=nullH7, nullH0=nullH0,
         nullH3=nullH3,
         corr_rows=np.array(
             [[c[0], c[1], c[2],
               (c[3] if c[3] == c[3] else np.nan)]
              for c in corr_rows]),
         a155_diff=np.float64(a155_diff),
         a156a_diff=np.float64(a156a_diff),
         a156b_diff=np.float64(a156b_diff),
         a156c_diff=np.float64(a156c_diff),
         a157_diff=np.float64(a157_diff),
         a158_fail=np.int64(a158_fail),
         a158_checked=np.int64(a158_checked),
         a159_max_dlg=np.float64(max_dlg),
         a160_pre_id_max=np.float64(pre_id_max),
         a161_postn_dev=np.float64(postn_dev),
         a116_ok=np.bool_(a116_ok),
         a155_ok=np.bool_(a155_ok),
         a156a_ok=np.bool_(a156a_ok),
         a156b_ok=np.bool_(a156b_ok),
         a156c_ok=np.bool_(a156c_ok),
         a157_ok=np.bool_(a157_ok),
         a158_ok=np.bool_(a158_ok),
         a159_ok=np.bool_(a159_ok),
         a160_ok=np.bool_(a160_ok),
         a161_ok=np.bool_(a161_ok),
         verdict=np.array(verdict),
         elapsed=np.float64(elapsed))

stats = {
    'T2a_k_cross': {
        'med_diag': med_diag_K,
        'med_A_off': med_Aoff_K,
        'med_B_off': med_Boff_K,
        'med_D_off': medD_K,
        'p_perm': p_perm_K,
        'g1': g1},
    'T2b_v_cross': {
        'med_diag': float(np.median(
            np.diag(MAT_AV))),
        'med_A_off': float(np.median(
            MAT_AV[off])),
        'med_B_off': float(np.median(
            MAT_BV[off]))},
    'T6_controls': {
        'med_exact_h0': med0_exact,
        'med_exact_h3': med3_exact,
        'med_null_h7': mn7,
        'med_null_h0': mn0,
        'med_null_h3': mn3,
        'max_null_h7': float(nullH7.max()),
        'max_null_h0': float(nullH0.max()),
        'max_null_h3': float(nullH3.max()),
        'p_h7': p_h7, 'p_h0': p_h0,
        'p_h3': p_h3},
    'T3_stages': {'med_cos_pre': med_pre,
                  'med_cos_att': med_att,
                  'med_cos_mlp': med_mlp,
                  'med_cos_mlpn': med_mlpn},
    'T4_attn_transfer': {
        'rows': [{'k': int(c[0]),
                  'base_mass': c[1],
                  'dmass_med': c[2],
                  'corr_dmass_AK': c[3]}
                 for c in corr_rows]},
    'anchors': {'a116_seals_ok': a116_ok,
                'a155_recapture_diff': a155_diff,
                'a156a_kdiag_diff': a156a_diff,
                'a156b_vdiag_diff': a156b_diff,
                'a156c_jdiag_diff': a156c_diff,
                'a157_sham_diff': a157_diff,
                'a158_fail': a158_fail,
                'a158_checked': a158_checked,
                'a159_max_dlg': max_dlg,
                'a160_pre_id_max': pre_id_max,
                'a161_postn_dev': postn_dev,
                'anchor_core_ok':
                    anchor_core_ok},
}
result = {'phase': PHASE, 'name': NAME,
          'created': created, 'elapsed': elapsed,
          'run': 'run3 authoritative (fp32; capture '
                 'bank from the phase3048 npz; '
                 'chain anchors vs the phase3051 '
                 'npz; run1 non-authoritative - '
                 'block-vs-fullrow anchor mismatch '
                 '(a156a/b diff 1.310/1.392), '
                 'registered in corrections',
          'prereg': PREREG, 'stats': stats,
          'verdict': verdict}
res_path = os.path.join(OUT, 'result.json')
with open(res_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=1)

seal = {
    'phase': PHASE, 'name': NAME, 'created': created,
    'npz_sha256_8': sha8(npz_path),
    'result_sha256_8': sha8(res_path),
    'exec_sha256_8': sha8(os.path.join(
        OUT, 'execution.json')),
    'script_sha256_8': sha8(os.path.abspath(__file__)),
    'verdict': verdict,
    'anchor_core_ok': anchor_core_ok,
    'run1_nonauthoritative': {
        'reason': 'a156a/a156b block-vs-fullrow '
                  'anchor mismatch (diff 1.310/'
                  '1.392); a156c bit 0.0',
        'elapsed_s': 393.6},
}
with open(os.path.join(OUT, 'seal.json'), 'w',
          encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False, indent=1)
log('sealed npz8=%s result8=%s exec8=%s '
    'script8=%s elapsed=%.1fs'
    % (seal['npz_sha256_8'],
       seal['result_sha256_8'],
       seal['exec_sha256_8'],
       seal['script_sha256_8'], elapsed))
log('sealed')
