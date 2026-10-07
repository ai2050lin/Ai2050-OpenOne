# -*- coding: utf-8 -*-
"""Phase 3117 (Omega-P115): pair-cancellation ablation
test of the equilibrium model + sampled behavioral
transmission of the entropy regulation.

Preregistered (3116 MEMO section 6):
  1. PAIR CANCELLATION: joint MLP ablation of layer
     PAIRS chosen from the 3116 single-point map.
     Classes (frozen rule: sign of the two frozen
     singles):
       cancel  = one negative + one positive single
                 (predicted: joint ~ sum of singles,
                 |joint| << |a| + |b| - the pair's
                 writes cancel);
       same    = both singles same sign (negative
                 block or positive block; predicted:
                 sub-additive, |joint| < |sum|).
     12 pairs: 7 cancel + 4 same-neg + 1 same-pos,
     plus 2 single-point overlap conditions
     (abl_mlp_L24, abl_mlp_L35 - the positive-layer
     sign re-verification) and baseline.
     Frozen singles a, b come from the 3116
     result.json sweep (pipeline determinism already
     proven bit-exact there).
     CANC gate: mean over cancel pairs of
       add_err_rel = |d_pair - (a+b)| /
                     max(|a+b|, 0.01)
       <= 0.20 -> cancellation_pairwise_additive
       >= 0.50 -> cancellation_pairwise_buffered
       else     -> cancellation_pairwise_mixed.
     Report: cann_eff = 1 - |d_pair|/(|a|+|b|) per
     cancel pair; add_frac = |d_pair|/|sum| per same
     pair.
  2. SAMPLED BEHAVIORAL TRANSMISSION: 3116 showed
     abl_L32 changes the P-vs-A1 distribution shape
     (JS 0.1593 -> 0.2128) with ZERO greedy behavior
     change.  Test whether the shape change reaches
     behavior under temperature 0.7 sampling:
       B1 JS re-read (P vs A1 next-token JS, clean
          and abl_L32, all 672 pairs) - anchors vs
          3115 bit-exact (<=1e-4);
       B2 greedy control: clean vs abl_L32 full
          12-token generations for 300 pairs x
          {P, A1} (3116 never compared clean-vs-abl
          sequences directly; also closes the token-
          substitution hole: yes-family swaps);
       B3 sampled generation: K=5 reps, temp 0.7,
          same frozen seed for clean and abl per
          (pair, direction, rep) -> sequence-level
          agreement measures pure ablation-induced
          divergence under sampling.
       SAMP gate: samp_seq_agree <= 0.60 ->
         sampled_consequence_confirmed;
         >= 0.98 -> sampled_behavior_invariant;
         else     -> sampled_behavior_partial.

METHOD: identical MLP-zeroing hook semantics as
3114/3115/3116; overlap bit-exactness (L24/L35 vs
3116, JS anchors vs 3115) is the primary
hook-correctness check.

CONDITIONS (A pairs):
  baseline, abl_mlp_L24, abl_mlp_L35, 12 pair
  conditions.
CONDITIONS (B):
  JS: baseline_dist, abl_L32_dist (want_dist on
      P/A1 records);
  gen: greedy clean/abl_L32 x 300 pairs x 2 dirs;
  gen: sampled (temp 0.7, K=5) clean/abl_L32 x 300
       pairs x 2 dirs.
"""
import io
import json
import os
import time
import zlib
from collections import Counter
from datetime import datetime

import numpy as np

SMOKE = os.environ.get('SMOKE', '0') == '1'
NAME = 'omega_p115_pair_cancellation_sampling'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
D13 = os.path.join(RDIR, 'phase3113',
                   'omega_p111_artifact_writein')
D15 = os.path.join(RDIR, 'phase3115',
                   'omega_p113_joint_mlp_erase_purpose')
D16 = os.path.join(RDIR, 'phase3116',
                   'omega_p114_full_mlp_sweep_decouple')
D05 = os.path.join(RDIR, 'phase3105',
                   'omega_p103_incontext_truth_'
                   'consistency')
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
OUT = os.path.join(RDIR, 'phase3117', NAME)
if SMOKE:
    D13 = os.path.join(D13, 'smoke')
    D05 = os.path.join(D05, 'smoke')
    OUT = os.path.join(OUT, 'smoke')
os.makedirs(OUT, exist_ok=True)
LOGF = os.path.join(OUT, 'run_log.txt')
T0 = time.time()


def log(msg):
    line = '[%7.1fs] %s' % (time.time() - T0, msg)
    with io.open(LOGF, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# ================================================================
# frozen inputs
# ================================================================
res13 = json.load(io.open(
    os.path.join(D13, 'result.json'), encoding='utf-8'))
assert res13['verdict'] == \
    'belief_robust|within_unit_replicated|' \
    'write_in_concentrated'
res15 = json.load(io.open(
    os.path.join(D15, 'result.json'), encoding='utf-8'))
assert res15['verdict'] == \
    'write_joint_partial|additive|' \
    'erase_serves_generation'
res16 = json.load(io.open(
    os.path.join(D16, 'result.json'), encoding='utf-8'))
assert res16['verdict'] == \
    'interaction_dominant|no_behavioral_decoupling'
FROZEN_SINGLES = {}
for L in range(12, 36):
    FROZEN_SINGLES[L] = \
        res16['sweep']['abl_mlp_L%d' % L]['dmp_rel']
FROZEN_OVERLAP = {
    'L24': FROZEN_SINGLES[24],
    'L35': FROZEN_SINGLES[35],
}
JS_ANCHORS = {
    'base': res15['erase_purpose']['js_base_mean'],
    'abl': res15['erase_purpose']['js_abl_mean'],
}
PAIRS = [
    ('pair_L26_L24', (26, 24)),
    ('pair_L33_L35', (33, 35)),
    ('pair_L31_L24', (31, 24)),
    ('pair_L26_L35', (26, 35)),
    ('pair_L31_L35', (31, 35)),
    ('pair_L14_L35', (14, 35)),
    ('pair_L33_L32', (33, 32)),
    ('pair_L26_L33', (26, 33)),
    ('pair_L26_L31', (26, 31)),
    ('pair_L20_L26', (20, 26)),
    ('pair_L17_L33', (17, 33)),
    ('pair_L24_L35', (24, 35)),
]
NSAMP_PAIRS = 300
K_REPS = 5
TEMP = 0.7
N_NEW = 12
N_MATCH = 8
if SMOKE:
    NSAMP_PAIRS = 6
    K_REPS = 2
    N_NEW = 6
    N_MATCH = 4
mat5 = json.load(io.open(
    os.path.join(D05, 'material.json'),
    encoding='utf-8'))
capb = np.load(os.path.join(D13, 'capture_b.npz'),
               allow_pickle=False)
pkB = capb['pk']
condB = capb['cond']
truthB = capb['truth']
m_base = capb['m'].astype(np.float32)
NB = len(pkB)

seal = {
    'phase': 3117,
    'name': NAME,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'pairs': [[n, list(ls)]
              for (n, ls) in PAIRS],
    'pair_class_rule': 'cancel if sign(a) != '
                       'sign(b) using frozen 3116 '
                       'singles; else same',
    'overlap_singles': ['L24', 'L35'],
    'frozen_singles_L24': FROZEN_SINGLES[24],
    'frozen_singles_L35': FROZEN_SINGLES[35],
    'js_def': 'per-pair JS(P_next || A1_next), '
              'float64 full vocab, natural log; '
              'conditions clean + abl_L32',
    'js_anchors': JS_ANCHORS,
    'sampling': {'temp': TEMP, 'K': K_REPS,
                 'n_new': N_NEW,
                 'n_pairs': NSAMP_PAIRS,
                 'pair_selection': 'first %d of '
                                   'sorted pks'
                                   % NSAMP_PAIRS,
                 'seed_rule': 'crc32(pk|dir|rep) '
                              '& 0x7fffffff - '
                              'identical seed for '
                              'clean and abl, no '
                              'cond term',
                 'matcher': 'torch.multinomial on '
                            'softmax(logits/T), '
                            'cuda generator'},
    'gates': {
        'canc': 'mean over cancel pairs of '
                'add_err_rel = |d_pair-(a+b)|/'
                'max(|a+b|,0.01): <=0.20 -> '
                'cancellation_pairwise_additive; '
                '>=0.50 -> '
                'cancellation_pairwise_buffered; '
                'else mixed',
        'overlap': 'L24/L35 dmp_rel vs 3116 within '
                   '1e-4 (bit-exact expected); JS '
                   'anchors vs 3115 within 1e-4',
        'samp': 'samp_seq_agree (same-seed clean '
                'vs abl sampled 12-token '
                'sequences identical): <=0.60 -> '
                'sampled_consequence_confirmed; '
                '>=0.98 -> '
                'sampled_behavior_invariant; '
                'else sampled_behavior_partial'},
    'metrics': {'cann_eff': '1 - |d_pair|/(|a|+|b|) '
                            'per cancel pair (1 = '
                            'perfect cancellation)',
                'add_frac': '|d_pair|/|sum| per '
                            'same pair (<1 = '
                            'sub-additive)',
                'greed_full_agree': 'clean vs abl '
                                    'greedy 12-token '
                                    'sequence '
                                    'identical rate, '
                                    'per direction',
                'samp_first_l1': '0.5*L1 distance '
                                 'of first-token '
                                 'empirical '
                                 'distributions '
                                 '(K reps), clean '
                                 'vs abl'},
    'yes_family': ['yes', 'Yes', ' yes', ' Yes',
                   'YES'],
    'note': '3116 hard-hole 2 (greedy-only) and 3 '
            '(token substitution) both closed '
            'here: greedy clean-vs-abl sequence '
            'agreement measured directly; '
            'yes-family swap rate reported for '
            'diverged greedy pairs',
}
with io.open(os.path.join(OUT, 'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, indent=1)
log('Phase 3117 Omega-P115 start; SMOKE=%d OUT=%s'
    % (SMOKE, OUT))
log('design sealed; frozen singles L24=%+.4f '
    'L35=%+.4f; js anchors base=%.6f abl=%.6f'
    % (FROZEN_SINGLES[24], FROZEN_SINGLES[35],
       JS_ANCHORS['base'], JS_ANCHORS['abl']))
for (pn, ls) in PAIRS:
    (a, b) = (FROZEN_SINGLES[ls[0]],
              FROZEN_SINGLES[ls[1]])
    cls = ('cancel'
           if (a < 0) != (b < 0) else 'same')
    log('  sealed %s layers=%s sum=%+.4f class=%s'
        % (pn, ls, a + b, cls))

# --- rebuilt records (identical to 3113 B / 3114-3116) ---
import random as _rnd  # noqa: E402


def build_prompt_rb(mat, s, o, lrel, qrel):
    ents = mat['entities']
    PREDS = mat['predicates']
    D = [tuple(d) for d in
         mat['distractors']['%d_%d' % (s, o)]]
    k = mat['kline']['%d_%d' % (s, o)]
    lines = [(s, lrel, o)] + list(D)
    rng2 = _rnd.Random(zlib.crc32(
        ('%d_%d_ord5' % (s, o)).encode('ascii')))
    order = list(range(8))
    rng2.shuffle(order)
    lines = [lines[i] for i in order]
    ci = lines.index((s, lrel, o))
    lines[ci], lines[k] = lines[k], lines[ci]
    text = 'Facts:'
    for (ls, lr, lo) in lines:
        text += ' The %s %s the %s.' % (
            ents[ls], PREDS[lr], ents[lo])
    text += (' Query: The %s %s the %s. Is this query '
             'true? Answer:' % (ents[s], PREDS[qrel],
                                ents[o]))
    return text


p2r = mat5['pair2rel']
frel = mat5['false_rels']
texts = []
for i in range(NB):
    pk = str(pkB[i])
    (s, o) = (int(v) for v in pk.split('_'))
    r = p2r[pk]
    ri1, ri2 = frel[pk]
    cc = str(condB[i])
    if cc == 'P':
        (qrel, lrel) = (r, r)
    elif cc == 'A1':
        (qrel, lrel) = (r, ri1)
    else:
        (qrel, lrel) = (ri2, ri1)
    texts.append(build_prompt_rb(mat5, s, o, lrel,
                                 qrel))
hP = {}
hA1 = {}
for i in range(NB):
    pk = str(pkB[i])
    if str(condB[i]) == 'P':
        hP[pk] = i
    elif str(condB[i]) == 'A1':
        hA1[pk] = i
pks = sorted(hP.keys())
ip = np.array([hP[p] for p in pks])
ia = np.array([hA1[p] for p in pks])
NP_ = len(pks)
i_PA = sorted(set(int(x) for x in
                  list(ip) + list(ia)))
log('records rebuilt: %d (%d pairs); P/A1 records '
    '%d' % (NB, NP_, len(i_PA)))

# ================================================================
# model + hooks
# ================================================================
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, \
    AutoTokenizer  # noqa: E402

torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
assert torch.cuda.is_available()
tok = AutoTokenizer.from_pretrained(MDIR)
model = AutoModelForCausalLM.from_pretrained(
    MDIR, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda').eval()
NL = len(model.model.layers)
HIDb = int(model.config.hidden_size)
NH = int(model.config.num_attention_heads)
HD = int(getattr(model.config, 'head_dim', 0) or 0) \
    or HIDb // NH
WU = model.lm_head.weight.detach()
assert not (model.lm_head.bias is not None), \
    'lm_head bias unexpected'
YES_ID = int(mat5['yes_id'])
NO_ID = int(mat5['no_id'])
w_dn = (WU[YES_ID] - WU[NO_ID]).float().cpu().numpy()
YES_FAMILY = []
for ys in seal['yes_family']:
    YES_FAMILY += list(tok(ys,
                           add_special_tokens=False)
                       ['input_ids'])
YES_FAMILY = sorted(set(YES_FAMILY))
log('model loaded NL=%d HID=%d heads=%d head_dim=%d; '
    'yes_family ids=%s'
    % (NL, HIDb, NH, HD, YES_FAMILY))

ABL = {'mode': None, 'layers': set()}


def mk_mlp_post(L):
    def hook(mod, mod_in, out):
        if ABL['mode'] == 'mlp' \
                and L in ABL['layers']:
            return torch.zeros_like(out)
        return None
    return hook


def mk_cap(L, key):
    def hook(mod, args):
        t = args[0]
        if t.dim() == 3:
            v = t[0, -1, :]
        elif t.dim() == 2:
            v = t[-1, :]
        else:
            v = t.reshape(-1)[-1:]
        CCHK.setdefault(L, {})[key] = v.detach()
        return None
    return hook


def mk_cap_post(L, key):
    def hook(mod, mod_in, out):
        t = out[0] if isinstance(out, tuple) else out
        v = t[0, -1, :] if t.dim() == 3 \
            else t[-1, :]
        CCHK.setdefault(L, {})[key] = v.detach()
        return None
    return hook


CCHK = {}
SNAP_CLEAN = {}
INSTR = [20, 24, 28, 32]
handles = []
for L in INSTR:
    blk = model.model.layers[L]
    handles.append(
        blk.mlp.register_forward_hook(mk_mlp_post(L)))
    handles.append(
        blk.register_forward_pre_hook(mk_cap(L, 'h_in')))
    handles.append(
        blk.self_attn.o_proj
        .register_forward_pre_hook(mk_cap(L, 'attn')))
    handles.append(
        blk.mlp.register_forward_hook(
            mk_cap_post(L, 'mlp')))
    handles.append(
        blk.register_forward_hook(
            mk_cap_post(L, 'h_out')))
ABL_LAYERS = set([L for (_, ls) in PAIRS
                  for L in ls]) | {24, 35}
ABL_HOOKS = {}
for L in sorted(ABL_LAYERS):
    if L in INSTR:
        continue
    ABL_HOOKS[L] = model.model.layers[L].mlp \
        .register_forward_hook(mk_mlp_post(L))


def forward_rec(text):
    ids = tok(text, add_special_tokens=False)['input_ids']
    t = torch.tensor([ids], device='cuda')
    with torch.inference_mode():
        out = model(t, output_hidden_states=True,
                    use_cache=False)
        hfn = model.model.norm(
            out.hidden_states[NL][0, -1, :])
        m_val = float((hfn.float().cpu().numpy()
                       @ w_dn).item())
        logits = model.lm_head(hfn.unsqueeze(0))[0]
        lp = torch.softmax(logits.float(), -1)
        y_p = float(lp[YES_ID].item())
        n_p = float(lp[NO_ID].item())
        nxt = int(logits.argmax(-1).item())
        del out
    return m_val, y_p, n_p, nxt


def selfcheck(cname, snap):
    """Identity checks at instrumented layers (values
    captured under the active ablation)."""
    ok_all = 0.0
    with torch.no_grad():
        for L in INSTR:
            if L not in ABL['layers']:
                continue
            c = CCHK.get(L, {})
            if not all(k in c for k in
                       ('h_in', 'h_out', 'mlp',
                        'attn')):
                continue
            blk = model.model.layers[L]
            got = blk.self_attn.o_proj(
                c['attn'].unsqueeze(0))[0].float()
            resid = (c['h_out'].float()
                     - c['h_in'].float()
                     - got - c['mlp'].float())
            r_res = float(
                resid.norm()
                / (c['h_out'].float().norm()
                   + 1e-9))
            ok_all = max(ok_all, r_res)
            if L == min(ABL['layers']):
                s = snap.get(L, {})
                if all(k in s for k in
                       ('h_in', 'attn')):
                    r_up = float(
                        (c['h_in'].float()
                         - s['h_in'].float()).norm()
                        / (s['h_in'].float().norm()
                           + 1e-9))
                    r_at = float(
                        (c['attn'].float()
                         - s['attn'].float()).norm()
                        / (s['attn'].float().norm()
                           + 1e-9))
                    ok_all = max(ok_all, r_up, r_at)
            if len(ABL['layers']) == 1:
                s = snap.get(L, {})
                if all(k in s for k in
                       ('h_out', 'mlp')):
                    lhs = c['h_out'].float()
                    rhs = (s['h_out'].float()
                           - s['mlp'].float())
                    r_id = float(
                        (lhs - rhs).norm()
                        / (s['h_out'].float()
                           .norm() + 1e-9))
                    ok_all = max(ok_all, r_id)
    CCHK.clear()
    return ok_all


def gen_greedy(text, n_new, layers_abl):
    """Greedy generation with KV cache under the
    active ablation; returns list of token ids."""
    ABL['mode'] = 'mlp' if layers_abl else None
    ABL['layers'] = set(layers_abl)
    ids = tok(text,
              add_special_tokens=False)['input_ids']
    cur = torch.tensor([ids], device='cuda')
    out_tokens = []
    with torch.inference_mode():
        out = model(cur, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[0, -1, :].argmax(-1)
        for _ in range(n_new):
            out_tokens.append(int(nxt))
            if len(out_tokens) >= n_new:
                break
            out = model(nxt.view(1, 1),
                        past_key_values=past,
                        use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1, :].argmax(-1)
    ABL['mode'] = None
    ABL['layers'] = set()
    return out_tokens


def gen_sampled(text, n_new, layers_abl, temp,
                seed):
    """Temperature sampling with KV cache under the
    active ablation; frozen seed per
    (pk, direction, rep) shared by clean and abl."""
    ABL['mode'] = 'mlp' if layers_abl else None
    ABL['layers'] = set(layers_abl)
    ids = tok(text,
              add_special_tokens=False)['input_ids']
    cur = torch.tensor([ids], device='cuda')
    g = torch.Generator(device='cuda')
    g.manual_seed(int(seed))
    out_tokens = []
    with torch.inference_mode():
        out = model(cur, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :].float()
        nxt = torch.multinomial(
            torch.softmax(logits / temp, -1), 1,
            generator=g)
        for _ in range(n_new):
            out_tokens.append(int(nxt.item()))
            if len(out_tokens) >= n_new:
                break
            out = model(nxt.view(1, 1),
                        past_key_values=past,
                        use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1, :].float()
            nxt = torch.multinomial(
                torch.softmax(logits / temp, -1), 1,
                generator=g)
    ABL['mode'] = None
    ABL['layers'] = set()
    return out_tokens


# ================================================================
# PART A: pair cancellation ablation
# ================================================================
COND_A = ([('baseline', set())]
          + [('abl_mlp_L%d' % L, {L})
             for L in (24, 35)]
          + [(pn, set(ls)) for (pn, ls) in PAIRS])
RES = {}
sc_log = {}
for (cname, layers) in COND_A:
    ABL['mode'] = 'mlp' if layers else None
    ABL['layers'] = set(layers)
    t0 = time.time()
    R = {'m': np.zeros(NB, dtype=np.float32),
         'y': np.zeros(NB, dtype=np.float32),
         'n': np.zeros(NB, dtype=np.float32),
         'nx': np.zeros(NB, dtype=np.int32)}
    instr_hit = bool(layers & set(INSTR))
    for i in range(NB):
        (m_v, y_p, n_p, nx) = forward_rec(texts[i])
        R['m'][i] = m_v
        R['y'][i] = y_p
        R['n'][i] = n_p
        R['nx'][i] = nx
        if i == 0:
            if not layers:
                SNAP_CLEAN.update(
                    {L: dict(c)
                     for L, c in CCHK.items()})
                CCHK.clear()
                log('clean snapshot stored (%d '
                    'layers)' % len(SNAP_CLEAN))
            elif instr_hit:
                sc = selfcheck(cname, SNAP_CLEAN)
                sc_log[cname] = sc
                log('selfcheck %s: rel L2 = %.2e'
                    % (cname, sc))
                assert sc < 0.02, (cname, sc)
    ABL['layers'] = set()
    ABL['mode'] = None
    RES[cname] = R
    log('%s done (%.1fs)'
        % (cname, time.time() - t0))


def pair_stats(mv):
    mP = mv[ip]
    mA1 = mv[ia]
    dpair = mP - mA1
    return (float(dpair.mean()),
            float(np.median(dpair)),
            float((dpair > 0).mean()))


m0 = pair_stats(RES['baseline']['m'])[0]
sweep = {}
for (cname, layers) in COND_A:
    mp, medp, auc = pair_stats(RES[cname]['m'])
    if cname == 'baseline':
        sweep[cname] = {'mpair_mean': mp,
                        'mpair_median': medp,
                        'auc_truth_m': auc}
        continue
    d = (mp - m0) / (abs(m0) + 1e-9)
    sweep[cname] = {'mpair_mean': mp,
                    'mpair_median': medp,
                    'auc_truth_m': auc,
                    'dmp_rel': d}
    log('%s: mpair=%.4f dmp_rel=%+.4f'
        % (cname, mp, d))

# overlap consistency vs 3116 (bit-exact)
overlap_chk = {}
for L, frozen in FROZEN_OVERLAP.items():
    got = sweep['abl_mlp_%s' % L]['dmp_rel']
    diff = abs(got - frozen)
    overlap_chk[L] = {'got': got,
                      'frozen': frozen,
                      'abs_diff': diff}
    log('overlap %s: got %+.6f frozen %+.6f '
        'diff %.2e' % (L, got, frozen, diff))
    if not SMOKE:
        assert diff < 1e-4, ('overlap', L, got,
                             frozen)

# pair analysis
pairs_out = {}
canc_rels = []
same_neg_frac = []
same_pos_frac = []
for (pn, ls) in PAIRS:
    (a, b) = (FROZEN_SINGLES[ls[0]],
              FROZEN_SINGLES[ls[1]])
    s = a + b
    cls = 'cancel' if (a < 0) != (b < 0) \
        else ('same_neg' if a < 0 else 'same_pos')
    d_pair = sweep[pn]['dmp_rel']
    add_err = d_pair - s
    add_err_rel = abs(add_err) / max(abs(s), 0.01)
    rec = {'layers': list(ls), 'class': cls,
           'a': a, 'b': b, 'sum': s,
           'd_pair': d_pair, 'add_err': add_err,
           'add_err_rel': add_err_rel}
    if cls == 'cancel':
        rec['cann_eff'] = 1.0 - abs(d_pair) \
            / (abs(a) + abs(b))
        canc_rels.append(add_err_rel)
        log('PAIR %s [%s]: d=%+.4f sum=%+.4f '
            'add_err_rel=%.3f cann_eff=%.3f'
            % (pn, cls, d_pair, s, add_err_rel,
               rec['cann_eff']))
    else:
        rec['add_frac'] = abs(d_pair) \
            / max(abs(s), 0.01)
        if cls == 'same_neg':
            same_neg_frac.append(rec['add_frac'])
        else:
            same_pos_frac.append(rec['add_frac'])
        log('PAIR %s [%s]: d=%+.4f sum=%+.4f '
            'add_err_rel=%.3f add_frac=%.3f'
            % (pn, cls, d_pair, s, add_err_rel,
               rec['add_frac']))
    pairs_out[pn] = rec
mean_canc_rel = float(np.mean(canc_rels))
if mean_canc_rel <= 0.20:
    canc_v = 'cancellation_pairwise_additive'
elif mean_canc_rel >= 0.50:
    canc_v = 'cancellation_pairwise_buffered'
else:
    canc_v = 'cancellation_pairwise_mixed'
same_summ = {
    'same_neg_add_frac_mean':
        float(np.mean(same_neg_frac))
        if same_neg_frac else -1.0,
    'same_pos_add_frac_mean':
        float(np.mean(same_pos_frac))
        if same_pos_frac else -1.0,
    'cancel_add_err_rel_mean': mean_canc_rel,
    'cancel_cann_eff_mean':
        float(np.mean([pairs_out[pn]['cann_eff']
                       for (pn, ls) in PAIRS
                       if pairs_out[pn]['class']
                       == 'cancel'])),
    'gate': canc_v}
log('CANC: mean add_err_rel=%.3f over %d cancel '
    'pairs -> %s | same_neg add_frac mean=%.3f '
    'same_pos add_frac mean=%.3f'
    % (mean_canc_rel, len(canc_rels), canc_v,
       same_summ['same_neg_add_frac_mean'],
       same_summ['same_pos_add_frac_mean']))

# ================================================================
# PART B1: JS re-read (P vs A1) clean + abl_L32
# ================================================================
def js_div(lp_p, lp_q):
    """Jensen-Shannon divergence of two next-token
    distributions given as log-prob vectors (natural
    log), float64 over the full vocab."""
    p = np.exp(lp_p.astype(np.float64))
    q = np.exp(lp_q.astype(np.float64))
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(
            a[mask] * (np.log(a[mask])
                       - np.log(b[mask]))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


JSR = {}
for (jcname, layers) in (('baseline_dist', ()),
                         ('abl_L32_dist', (32,))):
    ABL['mode'] = 'mlp' if layers else None
    ABL['layers'] = set(layers)
    t0 = time.time()
    pending = {}
    js_by_pk = {}
    nx_by_pk = {}
    pend_max = 0
    for i in i_PA:
        ids = tok(texts[i],
                  add_special_tokens=False)['input_ids']
        t = torch.tensor([ids], device='cuda')
        with torch.inference_mode():
            out = model(t, output_hidden_states=True,
                        use_cache=False)
            hfn = model.model.norm(
                out.hidden_states[NL][0, -1, :])
            logits = model.lm_head(
                hfn.unsqueeze(0))[0]
            logp = torch.log_softmax(
                logits.double(), -1) \
                .detach().cpu().numpy()
            nx_ = int(logits.argmax(-1).item())
            del out
        pk = str(pkB[i])
        cc = str(condB[i])
        nx_by_pk[(pk, cc)] = nx_
        d = pending.setdefault(pk, {})
        d[cc] = logp
        pend_max = max(pend_max, len(pending))
        if 'P' in d and 'A1' in d:
            d = pending.pop(pk)
            js_by_pk[pk] = js_div(d['P'], d['A1'])
    ABL['layers'] = set()
    ABL['mode'] = None
    assert not pending, \
        'unresolved pairs: %d' % len(pending)
    jsr = np.array([js_by_pk[p] for p in pks],
                   dtype=np.float64)
    JSR[jcname] = {'js': jsr,
                   'nx': dict(nx_by_pk)}
    log('%s: js mean=%.6f med=%.6f pend_max=%d '
        '(%.1fs)'
        % (jcname, jsr.mean(),
           float(np.median(jsr)), pend_max,
           time.time() - t0))
js_base = JSR['baseline_dist']['js']
js_abl = JSR['abl_L32_dist']['js']
js_diff_chk = {
    'base_got': float(js_base.mean()),
    'base_frozen': JS_ANCHORS['base'],
    'base_diff': abs(float(js_base.mean())
                     - JS_ANCHORS['base']),
    'abl_got': float(js_abl.mean()),
    'abl_frozen': JS_ANCHORS['abl'],
    'abl_diff': abs(float(js_abl.mean())
                    - JS_ANCHORS['abl'])}
log('JS OVERLAP: base got %.6f frozen %.6f diff '
    '%.2e | abl got %.6f frozen %.6f diff %.2e'
    % (js_diff_chk['base_got'],
       js_diff_chk['base_frozen'],
       js_diff_chk['base_diff'],
       js_diff_chk['abl_got'],
       js_diff_chk['abl_frozen'],
       js_diff_chk['abl_diff']))
if not SMOKE:
    assert js_diff_chk['base_diff'] < 1e-4, \
        js_diff_chk
    assert js_diff_chk['abl_diff'] < 1e-4, \
        js_diff_chk

# ================================================================
# PART B2+B3: greedy control + sampled generation
# ================================================================
samp_pks = pks[:NSAMP_PAIRS]
assert len(samp_pks) == NSAMP_PAIRS
greed_store = {}
samp_store = {}
yf = set(YES_FAMILY)


def seed_for(pk, direction, rep):
    return zlib.crc32(('%s|%s|%d'
                       % (pk, direction, rep))
                      .encode('ascii')) & 0x7fffffff


t0 = time.time()
for j, pk in enumerate(samp_pks):
    tp = texts[hP[pk]]
    ta = texts[hA1[pk]]
    for direction, txt in (('P', tp), ('A1', ta)):
        # greedy control: clean vs abl_L32
        gp_c = gen_greedy(txt, N_NEW, ())
        gp_a = gen_greedy(txt, N_NEW, (32,))
        greed_store[(pk, direction)] = \
            (gp_c, gp_a)
        # sampled: K reps, same seed across conds
        for rep in range(K_REPS):
            sd = seed_for(pk, direction, rep)
            sc_ = gen_sampled(txt, N_NEW, (), TEMP,
                              sd)
            sa_ = gen_sampled(txt, N_NEW, (32,),
                              TEMP, sd)
            samp_store[(pk, direction, rep)] = \
                (sc_, sa_)
    if j % 50 == 0:
        log('gen %d/%d pairs (%.1fs)'
            % (j, NSAMP_PAIRS,
               time.time() - t0))

# --- greedy control analysis ---
greed_full_agree = {'P': 0, 'A1': 0}
greed_first_agree = {'P': 0, 'A1': 0}
greed_div_swap = {'P': 0, 'A1': 0}
greed_div_n = {'P': 0, 'A1': 0}
for (pk, direction), (gc, ga) in \
        greed_store.items():
    if gc == ga:
        greed_full_agree[direction] += 1
    if gc[:1] == ga[:1]:
        greed_first_agree[direction] += 1
    else:
        greed_div_n[direction] += 1
        c_in = gc[0] in yf
        a_in = ga[0] in yf
        if c_in and a_in:
            greed_div_swap[direction] += 1
greed_res = {
    'full_agree_P':
        greed_full_agree['P'] / NSAMP_PAIRS,
    'full_agree_A1':
        greed_full_agree['A1'] / NSAMP_PAIRS,
    'first_agree_P':
        greed_first_agree['P'] / NSAMP_PAIRS,
    'first_agree_A1':
        greed_first_agree['A1'] / NSAMP_PAIRS,
    'div_yes_swap_rate_P':
        (greed_div_swap['P']
         / greed_div_n['P']) if greed_div_n['P'] else -1.0,
    'div_yes_swap_rate_A1':
        (greed_div_swap['A1']
         / greed_div_n['A1'])
        if greed_div_n['A1'] else -1.0,
    'n_diverged': dict(greed_div_n)}
log('GREED CONTROL: full_agree P=%.4f A1=%.4f | '
    'first_agree P=%.4f A1=%.4f | diverged '
    'yes-swap rate P=%s A1=%s (n=%s)'
    % (greed_res['full_agree_P'],
       greed_res['full_agree_A1'],
       greed_res['first_agree_P'],
       greed_res['first_agree_A1'],
       greed_res['div_yes_swap_rate_P'],
       greed_res['div_yes_swap_rate_A1'],
       greed_res['n_diverged']))

# greedy first-token agreement over ALL 672 pairs
# (from the JS pass nx maps, keyed by (pk, dir))
nx_b = JSR['baseline_dist']['nx']
nx_a = JSR['abl_L32_dist']['nx']
gfa = {'P': 0, 'A1': 0}
for p in pks:
    for cc in ('P', 'A1'):
        if nx_b.get((p, cc)) == nx_a.get((p, cc)):
            gfa[cc] += 1
greed_first_all = {
    'P': gfa['P'] / NP_, 'A1': gfa['A1'] / NP_}
log('GREED FIRST-TOKEN (all %d pairs, from JS '
    'pass): agree P=%.4f A1=%.4f'
    % (NP_, greed_first_all['P'],
       greed_first_all['A1']))

# --- sampled analysis ---
seq_agree = 0
first_l1s = []
ys_c = 0
ys_a = 0
n_seq = NSAMP_PAIRS * 2 * K_REPS
first_clean = {}
first_abl = {}
for (pk, direction, rep), (sc_, sa_) in \
        samp_store.items():
    if sc_ == sa_:
        seq_agree += 1
    fc = sc_[0]
    fa_ = sa_[0]
    first_clean.setdefault((pk, direction),
                           []).append(fc)
    first_abl.setdefault((pk, direction),
                         []).append(fa_)
    if any(t in yf for t in sc_[:N_MATCH]):
        ys_c += 1
    if any(t in yf for t in sa_[:N_MATCH]):
        ys_a += 1
for key in first_clean:
    cc_ = Counter(first_clean[key])
    ca_ = Counter(first_abl[key])
    vocab = set(cc_) | set(ca_)
    l1 = sum(abs(cc_.get(v, 0)
                 - ca_.get(v, 0))
             for v in vocab) \
        / float(K_REPS)
    first_l1s.append(0.5 * l1)
samp_seq_agree = seq_agree / float(n_seq)
samp_first_l1 = float(np.mean(first_l1s)) \
    if first_l1s else -1.0
yr_samp_c = ys_c / float(n_seq)
yr_samp_a = ys_a / float(n_seq)
if samp_seq_agree <= 0.60:
    samp_v = 'sampled_consequence_confirmed'
elif samp_seq_agree >= 0.98:
    samp_v = 'sampled_behavior_invariant'
else:
    samp_v = 'sampled_behavior_partial'
log('SAMPLED: seq_agree=%.4f (%d/%d) '
    'first_l1=%.4f | yes_rate clean=%.4f abl=%.4f '
    '-> %s'
    % (samp_seq_agree, seq_agree, n_seq,
       samp_first_l1, yr_samp_c, yr_samp_a,
       samp_v))
log('VERDICT: %s|%s' % (canc_v, samp_v))

# ================================================================
# save
# ================================================================
npz_dict = {
    'm_base_check': m_base,
    'pk': pkB, 'cond': condB, 'truth': truthB,
    'js_base': js_base, 'js_abl32': js_abl,
}
for (cname, _) in COND_A:
    for k in ('m', 'y', 'n', 'nx'):
        npz_dict['%s__%s' % (cname, k)] = \
            RES[cname][k]
nsv = len(samp_pks)
for direction in ('P', 'A1'):
    arrc = np.zeros((nsv, N_NEW), dtype=np.int32)
    arra = np.zeros((nsv, N_NEW), dtype=np.int32)
    for jx, pk in enumerate(samp_pks):
        (gc, ga) = greed_store[(pk, direction)]
        arrc[jx, :len(gc)] = gc
        arra[jx, :len(ga)] = ga
    npz_dict['greed_%s_clean' % direction] = arrc
    npz_dict['greed_%s_abl' % direction] = arra
for direction in ('P', 'A1'):
    for cond, lab in (('c', 'clean'),
                      ('a', 'abl')):
        arr = np.zeros((nsv, K_REPS, N_NEW),
                       dtype=np.int32)
        for jx, pk in enumerate(samp_pks):
            for rep in range(K_REPS):
                (sc_, sa_) = samp_store[
                    (pk, direction, rep)]
                seq = sc_ if cond == 'c' else sa_
                arr[jx, rep, :len(seq)] = seq
        npz_dict['samp_%s_%s' % (direction, lab)] = \
            arr
npz_dict['samp_pks'] = np.array(samp_pks)
np.savez(os.path.join(OUT, 'pair_readout.npz'),
         **npz_dict)
log('readout saved')

results = {
    'verdict': '%s|%s' % (canc_v, samp_v),
    'sweep': sweep,
    'pairs': pairs_out,
    'cancel_summary': same_summ,
    'overlap_check': overlap_chk,
    'js_overlap_check': js_diff_chk,
    'js_stats': {
        'base_mean': float(js_base.mean()),
        'base_median':
            float(np.median(js_base)),
        'abl_mean': float(js_abl.mean()),
        'abl_median':
            float(np.median(js_abl)),
        'delta_mean':
            float((js_abl - js_base).mean()),
        'delta_sign_rate':
            float(((js_abl - js_base) > 0)
                  .mean())},
    'greedy_control': greed_res,
    'greedy_first_all': greed_first_all,
    'sampled': {
        'seq_agree': samp_seq_agree,
        'first_l1_mean': samp_first_l1,
        'yes_rate_clean': yr_samp_c,
        'yes_rate_abl': yr_samp_a,
        'n_seq': n_seq,
        'gate': samp_v},
    'sampling_params': seal['sampling'],
    'selfcheck_rel': sc_log,
    'gates': seal['gates'],
    'smoke': SMOKE, 'n_records': NB,
    'n_pairs': NP_,
    'n_samp_pairs': NSAMP_PAIRS,
    'k_reps': K_REPS, 'temp': TEMP,
    'n_new': N_NEW,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S')}
with io.open(os.path.join(OUT, 'result.json'), 'w',
             encoding='utf-8') as f:
    json.dump(results, f, indent=1,
              ensure_ascii=False)
log('result.json written')
log('Phase 3117 done (%.1fs)' % (time.time() - T0))
print('PHASE3117_DONE verdict=%s|%s'
      % (canc_v, samp_v))
