# -*- coding: utf-8 -*-
"""Phase 3116 (Omega-P114): full-layer MLP ablation
sweep L12-L35 (localize the ~81% unexplained causal
contribution) + free-generation behavioral test of
belief-generation decoupling.

Preregistered (3115 MEMO section 6):
  1. FULL-LAYER SWEEP: single-point MLP ablation for
     EVERY layer 12..35 (24 conditions) + one
     all-layer anchor abl_all_mlp (L0-L35 MLP zeroed)
     + baseline.  Analysis:
       - per-layer dmp_rel map (which layer carries
         the causal margin);
       - segment sums: early L12-19, write-window
         L20-28, late L29-35;
       - COVERAGE gate: |sum_single - d_all| /
         max(|d_all|, 0.01) <= 0.20 ->
         single_sum_explains_joint (near-additivity
         holds globally); > 0.20 -> interaction_
         dominant.
     Overlap conditions L20 (vs 3115 result), L24/
     L28/L32 (vs 3114 result) must reproduce
     bit-exactly (max |diff| < 1e-4) - the strongest
     hook-correctness check.
  2. BEHAVIORAL DECOUPLING: greedy generation
     (12 new tokens, KV cache) for each P/A1 record
     under clean vs abl_L32_mlp.  The 3115
     distribution result (JS(P||A1) rises when the
     erase is ablated) predicts: with the erase
     active, P and A1 generations AGREE (truth kept
     out of generation); with the erase ablated they
     DIVERGE (truth leaks into behavior).
       agree_x = fraction of pairs whose greedy
                 generations match on the first 8
                 tokens (P vs A1, same condition).
       gate: agree_clean - agree_abl >= 0.05 ->
             behavioral_decoupling_confirmed
             <= 0.02 -> no_behavioral_decoupling
             else -> behavioral_mixed.
     Diagnostic (frozen token family): yes-variants
     {yes, Yes, ' yes', ' Yes', YES} appearance rate
     in the first 8 generated tokens.

METHOD: identical MLP-zeroing hook semantics as
3114/3115 (forward hook returns zeros; per-condition
self-check via captured clean snapshot at the four
instrumented layers 20/24/28/32; bit-exact overlap
reproduction is the primary sweep-wide check).

CONDITIONS (A sweep):
  baseline, abl_mlp_L12 .. abl_mlp_L35 (24),
  abl_all_mlp (L0-L35).
CONDITIONS (B generation):
  clean, abl_L32_mlp  x  {P, A1} x 672 pairs.
"""
import io
import json
import os
import time
import zlib
from datetime import datetime

import numpy as np

SMOKE = os.environ.get('SMOKE', '0') == '1'
NAME = 'omega_p114_full_mlp_sweep_decouple'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
D13 = os.path.join(RDIR, 'phase3113',
                   'omega_p111_artifact_writein')
D14 = os.path.join(RDIR, 'phase3114',
                   'omega_p112_write_erase_ablation')
D15 = os.path.join(RDIR, 'phase3115',
                   'omega_p113_joint_mlp_erase_purpose')
D05 = os.path.join(RDIR, 'phase3105',
                   'omega_p103_incontext_truth_'
                   'consistency')
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
OUT = os.path.join(RDIR, 'phase3116', NAME)
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
res14 = json.load(io.open(
    os.path.join(D14, 'result.json'), encoding='utf-8'))
assert res14['verdict'] == \
    'head_write_not_causal|mlp_write_partial|' \
    'erase_not_active'
res15 = json.load(io.open(
    os.path.join(D15, 'result.json'), encoding='utf-8'))
assert res15['verdict'] == \
    'write_joint_partial|additive|' \
    'erase_serves_generation'
FROZEN_OVERLAP = {
    'L20': res15['dmp_rel']['L20_mlp'],
    'L24': res14['dmp_rel']['L24_mlp'],
    'L28': res14['dmp_rel']['L28_mlp'],
    'L32': res14['dmp_rel']['L32_mlp'],
}
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
    'phase': 3116,
    'name': NAME,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'sweep_layers': list(range(12, 36)),
    'anchor': 'abl_all_mlp = L0-L35 MLP zeroed',
    'conditions_a': ['baseline']
                    + ['abl_mlp_L%d' % L
                       for L in range(12, 36)]
                    + ['abl_all_mlp'],
    'conditions_b': ['clean', 'abl_L32_mlp'],
    'gen_params': 'greedy, 12 new tokens, KV cache, '
                  'batch 1',
    'gates': {
        'coverage': '|sum_single - d_all| / '
                    'max(|d_all|, 0.01) <= 0.20 -> '
                    'single_sum_explains_joint; '
                    'else interaction_dominant',
        'decouple': 'agree_clean - agree_abl >= '
                    '0.05 -> behavioral_decoupling_'
                    'confirmed; <= 0.02 -> '
                    'no_behavioral_decoupling; else '
                    'behavioral_mixed',
        'overlap': 'L20 vs 3115, L24/L28/L32 vs '
                   '3114 dmp_rel must match within '
                   '1e-4 (bit-exact expected)'},
    'agree_def': 'fraction of 672 pairs whose greedy '
                 'generations (P vs A1, same '
                 'intervention) match on all first 8 '
                 'new tokens',
    'yes_family': ['yes', 'Yes', ' yes', ' Yes',
                   'YES'],
    'note': 'overlap bit-exactness doubles as the '
            'sweep-wide hook-correctness check; '
            'segment map early L12-19 / write '
            'L20-28 / late L29-35 reported',
}
with io.open(os.path.join(OUT, 'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, indent=1)
log('Phase 3116 Omega-P114 start; SMOKE=%d OUT=%s'
    % (SMOKE, OUT))
log('design sealed; frozen overlap: %s'
    % json.dumps({k: round(v, 6)
                  for k, v in
                  FROZEN_OVERLAP.items()}))

# --- rebuilt records (identical to 3113 B / 3114/3115) ---
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
log('records rebuilt: %d (%d pairs)'
    % (NB, NP_))

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
SWEEP_HOOKS = {}
for L in range(12, 36):
    if L in INSTR:
        continue
    SWEEP_HOOKS[L] = model.model.layers[L].mlp \
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


# ================================================================
# PART A: full-layer sweep
# ================================================================
COND_A = ([('baseline', set())]
          + [('abl_mlp_L%d' % L, {L})
             for L in range(12, 36)]
          + [('abl_all_mlp',
              set(range(NL)))])
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
for h in handles:
    h.remove()
for h in SWEEP_HOOKS.values():
    h.remove()


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

# overlap consistency vs 3114/3115 (bit-exact)
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

d_all = sweep['abl_all_mlp']['dmp_rel']
singles = {L: sweep['abl_mlp_L%d' % L]['dmp_rel']
           for L in range(12, 36)}
sum_single = float(sum(singles.values()))
cov = abs(sum_single - d_all) / max(abs(d_all),
                                    0.01)
cov_v = ('single_sum_explains_joint' if cov <= 0.20
         else 'interaction_dominant')
seg_e = float(sum(singles[L]
                  for L in range(12, 20)))
seg_w = float(sum(singles[L]
                  for L in range(20, 29)))
seg_l = float(sum(singles[L]
                  for L in range(29, 36)))
top3 = sorted(singles.items(),
              key=lambda kv: kv[1])[:3]
log('COVERAGE: sum_single=%+.4f d_all=%+.4f '
    'cov=%.3f -> %s' % (sum_single, d_all, cov,
                        cov_v))
log('SEGMENTS: early(L12-19)=%+.4f write(L20-28)'
    '=%+.4f late(L29-35)=%+.4f'
    % (seg_e, seg_w, seg_l))
log('TOP3 causal layers: %s'
    % [(k, round(v, 4)) for k, v in top3])

# ================================================================
# PART B: behavioral decoupling (free generation)
# ================================================================
N_NEW = 12
N_MATCH = 8
agree_clean = 0
agree_abl = 0
yes_clean = 0
yes_abl = 0
diverge_pos = []
gen_store = {}
t0 = time.time()
for j, pk in enumerate(pks):
    tp = texts[hP[pk]]
    ta = texts[hA1[pk]]
    for cond, layers in (('clean', ()),
                         ('abl_L32_mlp', (32,))):
        gp = gen_greedy(tp, N_NEW, layers)
        ga = gen_greedy(ta, N_NEW, layers)
        gen_store[(pk, cond)] = (gp, ga)
        if gp[:N_MATCH] == ga[:N_MATCH]:
            if cond == 'clean':
                agree_clean += 1
            else:
                agree_abl += 1
        yf = set(YES_FAMILY)
        if any(t in yf for t in gp[:N_MATCH]):
            if cond == 'clean':
                yes_clean += 1
        if any(t in yf for t in ga[:N_MATCH]):
            if cond == 'clean':
                yes_clean += 1
        if any(t in yf for t in gp[:N_MATCH]):
            if cond == 'abl_L32_mlp':
                yes_abl += 1
        if any(t in yf for t in ga[:N_MATCH]):
            if cond == 'abl_L32_mlp':
                yes_abl += 1
        if gp != ga:
            k = next((idx for idx in range(N_NEW)
                      if gp[idx] != ga[idx]),
                     N_NEW)
            diverge_pos.append(k)
    if j % 100 == 0:
        log('gen %d/%d (%.1fs)'
            % (j, NP_, time.time() - t0))
ag_c = agree_clean / NP_
ag_a = agree_abl / NP_
yr_c = yes_clean / (2.0 * NP_)
yr_a = yes_abl / (2.0 * NP_)
d_agree = ag_c - ag_a
if d_agree >= 0.05:
    dec_v = 'behavioral_decoupling_confirmed'
elif d_agree <= 0.02:
    dec_v = 'no_behavioral_decoupling'
else:
    dec_v = 'behavioral_mixed'
med_div = (float(np.median(diverge_pos))
           if diverge_pos else -1.0)
log('DECOUPLE: agree_clean=%.4f agree_abl=%.4f '
    'diff=%+.4f -> %s | yes_rate clean=%.4f '
    'abl=%.4f | diverge_pos med=%s n=%d'
    % (ag_c, ag_a, d_agree, dec_v, yr_c, yr_a,
       med_div, len(diverge_pos)))
log('VERDICT: %s|%s' % (cov_v, dec_v))

np.savez(os.path.join(OUT, 'sweep_readout.npz'),
         m_base_check=m_base,
         **{('%s__%s' % (c, k)): RES[c][k]
            for (c, _) in COND_A
            for k in ('m', 'y', 'n', 'nx')},
         pk=pkB, cond=condB, truth=truthB)
log('readout saved')

results = {
    'verdict': '%s|%s' % (cov_v, dec_v),
    'sweep': sweep,
    'coverage': {'sum_single': sum_single,
                 'd_all': d_all,
                 'cov': cov, 'gate': cov_v},
    'segments': {'early_L12_19': seg_e,
                 'write_L20_28': seg_w,
                 'late_L29_35': seg_l},
    'top3_causal': [{'layer': int(k),
                     'dmp_rel': v}
                    for k, v in top3],
    'overlap_check': overlap_chk,
    'decouple': {
        'agree_clean': ag_c, 'agree_abl': ag_a,
        'diff': d_agree, 'gate': dec_v,
        'yes_rate_clean': yr_c,
        'yes_rate_abl': yr_a,
        'diverge_median_pos': med_div,
        'n_diverged_pairs': len(diverge_pos)},
    'selfcheck_rel': sc_log,
    'gates': seal['gates'],
    'smoke': SMOKE, 'n_records': NB,
    'n_pairs_gen': NP_,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S')}
with io.open(os.path.join(OUT, 'result.json'), 'w',
             encoding='utf-8') as f:
    json.dump(results, f, indent=1,
              ensure_ascii=False)
log('result.json written')
log('Phase 3116 done (%.1fs)' % (time.time() - T0))
print('PHASE3116_DONE verdict=%s|%s'
      % (cov_v, dec_v))
