# -*- coding: utf-8 -*-
"""Phase 3115 (Omega-P113): joint MLP ablation (total
write capacity upper bound) + erase-purpose readout
change (next-token distribution JS instead of yes/no
probs).

Preregistered (3114 MEMO section 6):
  1. JOINT ablation: zero L20+L24+L28 MLP outputs
     together (and L20+L24+L28+L32) - the correlational
     sum of their 3113 ds values is +4.65 (~56% of the
     margin); 3114 showed single-point ablations are
     non-linear (L24 single ablation RAISES the margin
     +0.346).  Gate (dmp_rel of L20_28_joint):
       <= -0.30 -> write_joint_collapse (redundant
                  distributed writing confirmed)
       <= -0.10 -> write_joint_partial
       else     -> write_elsewhere (L20-28 not the
                   causal carrier)
     Superadditivity check: d_joint28 vs
     d_L20_single + d_L24_single(3114) +
     d_L28_single(3114); <-0.05 more negative ->
     superadditive (chain collapse), else subadditive /
     additive.
  2. ERASE PURPOSE with a resolving readout: yes/no
     probs were ~0 (3114 hard flaw i).  New metric:
     per-pair JS divergence between the P-prompt and
     A1-prompt next-token distributions,
       js = JS(P_next(.|P) || P_next(.|A1)),
     computed clean (baseline) and under abl_L32_mlp.
     The erase hypothesis: L32 MLP erases the truth
     broadcast so it does NOT leak into generation.
     Ablating the erase should INCREASE the truth leak:
       mean(js_abl - js_clean) > 0 AND per-pair sign
       rate >= 0.60 -> erase_serves_generation
       mean <= 0 AND sign rate <= 0.40 ->
           erase_generation_neutral
       else -> erase_mixed.
  3. L20 single-point ablation fills the single-point
     trajectory (3113 ds +0.503 first write).

METHOD: same hook semantics as 3114 (MLP output zeroed
via forward hook; per-condition identity self-checks
against the clean snapshot stored at baseline record 0:
per-layer residual identity under ablation for every
ablated layer + upstream-untouched check at the topmost
ablated layer + exact single-layer identity h_out_abl =
h_out_clean - mlp_clean when exactly one layer is
ablated).

CONDITIONS:
  baseline                 no intervention (dist on)
  abl_L20_mlp              zero L20 MLP output
  abl_L32_mlp              zero L32 MLP output (dist on)
  abl_L20_28_mlp_joint     zero L20+L24+L28 MLP outputs
  abl_L20_32_mlp_joint     zero L20+L24+L28+L32 MLP

READOUT per record: m' = (W_yes-W_no).h_fn(last)
(= yes-minus-no logit margin), yes/no softmax probs,
argnext, and (dist conditions) full next-token
log-softmax for the online pair JS computation.
"""
import io
import json
import os
import time
import zlib
from datetime import datetime

import numpy as np

SMOKE = os.environ.get('SMOKE', '0') == '1'
NAME = 'omega_p113_joint_mlp_erase_purpose'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
D13 = os.path.join(RDIR, 'phase3113',
                   'omega_p111_artifact_writein')
D14 = os.path.join(RDIR, 'phase3114',
                   'omega_p112_write_erase_ablation')
D05 = os.path.join(RDIR, 'phase3105',
                   'omega_p103_incontext_truth_'
                   'consistency')
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
OUT = os.path.join(RDIR, 'phase3115', NAME)
if SMOKE:
    D13 = os.path.join(D13, 'smoke')
    D05 = os.path.join(D05, 'smoke')
    OUT = os.path.join(OUT, 'smoke')
os.makedirs(OUT, exist_ok=True)
LOGF = os.path.join(OUT, 'run_log.txt')
T0 = time.time()
log_lines = []


def log(msg):
    line = '[%7.1fs] %s' % (time.time() - T0, msg)
    log_lines.append(line)
    with io.open(LOGF, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# ================================================================
# frozen inputs: 3113 capture + material, 3114 result
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
D14_SINGLE = {
    'L24_mlp': res14['dmp_rel']['L24_mlp'],
    'L28_mlp': res14['dmp_rel']['L28_mlp'],
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
    'phase': 3115,
    'name': NAME,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'ablations': ['baseline', 'abl_L20_mlp',
                  'abl_L32_mlp', 'abl_L20_28_mlp_joint',
                  'abl_L20_32_mlp_joint'],
    'frozen_single_dmp_rel_3114': D14_SINGLE,
    'gates': {
        'joint': 'abl_L20_28_mlp_joint dmp_rel '
                 '<= -0.30 -> write_joint_collapse; '
                 '<= -0.10 -> write_joint_partial; '
                 'else write_elsewhere',
        'super': 'd_joint28 vs d_L20_single + '
                 'frozen(L24,L28 from 3114); '
                 'more-negative by >0.05 -> '
                 'superadditive else subadditive/'
                 'additive',
        'erase': 'mean(js_abl - js_clean) > 0 AND '
                 'sign_rate >= 0.60 -> '
                 'erase_serves_generation; mean <= 0 '
                 'AND sign_rate <= 0.40 -> '
                 'erase_generation_neutral; else '
                 'erase_mixed'},
    'js_def': 'per-pair JS(P_next(.|P) || '
              'P_next(.|A1)), natural log base, '
              'float64, full vocab; clean = baseline '
              'condition, abl = abl_L32_mlp condition',
    'selfcheck': 'per-layer residual identity under '
                 'ablation (every ablated layer); '
                 'upstream-untouched at topmost '
                 'ablated layer; exact single-layer '
                 'identity h_out_abl = h_out_clean - '
                 'mlp_clean when 1 layer ablated; '
                 'rel L2 < 0.02',
    'note': 'joint gate is the preregistered test of '
            'redundant-distributed writing (3114 '
            'single-point results were non-linear: '
            'L24 single ablation raises margin); '
            'erase gate replaces the unresolvable '
            'yes/no-prob purpose test of 3114',
}
with io.open(os.path.join(OUT, 'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, indent=1)
log('Phase 3115 Omega-P113 start; SMOKE=%d OUT=%s'
    % (SMOKE, OUT))
log('design sealed; frozen singles from 3114: %s'
    % json.dumps(D14_SINGLE))

# --- rebuilt records (identical to 3113 B / 3114) ---
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
log('records rebuilt: %d (%d pairs)'
    % (NB, len(pks)))

# ================================================================
# model + ablation hooks
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
log('model loaded NL=%d HID=%d heads=%d head_dim=%d'
    % (NL, HIDb, NH, HD))

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
handles = []
ABL_LAYERS = [20, 24, 28, 32]
for L in ABL_LAYERS:
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


def forward_rec(text, want_dist=False):
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
        logp = None
        if want_dist:
            logp = torch.log_softmax(
                logits.double(), -1) \
                .detach().cpu().numpy()
        del out
    return m_val, y_p, n_p, nxt, logp


def selfcheck(cond_name, snap):
    """First-record identity checks against the clean
    snapshot (baseline record 0).

    For every ablated layer L (values captured under
    ablation):
      (a) residual identity h_out = h_in +
          o_proj(attn) + mlp (holds per layer).
    At the topmost ablated layer (no intervention
    upstream):
      (b) h_in and o_proj input identical to clean.
    If exactly one layer is ablated:
      (c) exact h_out_abl = h_out_clean - mlp_out_clean.
    """
    ok_all = 0.0
    abl = sorted(ABL['layers'])
    with torch.no_grad():
        for L in abl:
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
            if L == abl[0]:
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
            if len(abl) == 1:
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
            log('  sc L%d: res=%.2e' % (L, r_res))
    CCHK.clear()
    return ok_all


def js_div(lp_p, lp_q):
    """Jensen-Shannon divergence of two next-token
    distributions given as log-prob vectors (natural
    log), computed in float64 over the full vocab."""
    p = np.exp(lp_p.astype(np.float64))
    q = np.exp(lp_q.astype(np.float64))
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(
            a[mask] * (np.log(a[mask])
                       - np.log(b[mask]))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


CONDITIONS = [
    # (name, layers, want_dist)
    ('baseline', set(), True),
    ('abl_L20_mlp', {20}, False),
    ('abl_L32_mlp', {32}, True),
    ('abl_L20_28_mlp_joint', {20, 24, 28}, False),
    ('abl_L20_32_mlp_joint', {20, 24, 28, 32}, False),
]
COND_NAMES = [c[0] for c in CONDITIONS]
DIST_CONDS = {'baseline', 'abl_L32_mlp'}
RES = {c: {'m': np.zeros(NB, dtype=np.float32),
           'y': np.zeros(NB, dtype=np.float32),
           'n': np.zeros(NB, dtype=np.float32),
           'nx': np.zeros(NB, dtype=np.int32)}
       for c in COND_NAMES}
JSR = {}
sc_log = {}
for (cname, layers, want_dist) in CONDITIONS:
    ABL['mode'] = 'mlp' if layers else None
    ABL['layers'] = set(layers)
    t0 = time.time()
    pending = {}
    js_by_pk = {}
    top_by_pk = {}
    pend_max = 0
    for i in range(NB):
        (m_v, y_p, n_p, nx, logp) = forward_rec(
            texts[i], want_dist=want_dist)
        RES[cname]['m'][i] = m_v
        RES[cname]['y'][i] = y_p
        RES[cname]['n'][i] = n_p
        RES[cname]['nx'][i] = nx
        if want_dist and logp is not None:
            pk = str(pkB[i])
            cc = str(condB[i])
            if cc in ('P', 'A1'):
                d = pending.setdefault(pk, {})
                d[cc] = logp
                pend_max = max(pend_max,
                               len(pending))
                if 'P' in d and 'A1' in d:
                    d = pending.pop(pk)
                    js_by_pk[pk] = js_div(d['P'],
                                          d['A1'])
                    tp = np.argpartition(
                        -d['P'], 10)[:10]
                    ta = np.argpartition(
                        -d['A1'], 10)[:10]
                    top_by_pk[pk] = (tp, ta)
        if i == 0:
            if not layers:
                SNAP_CLEAN.update(
                    {L: dict(c)
                     for L, c in CCHK.items()})
                CCHK.clear()
                log('clean snapshot stored (%d layers)'
                    % len(SNAP_CLEAN))
            else:
                sc = selfcheck(cname, SNAP_CLEAN)
                sc_log[cname] = sc
                log('selfcheck %s: rel L2 = %.2e'
                    % (cname, sc))
                assert sc < 0.02, (cname, sc)
    ABL['layers'] = set()
    if want_dist:
        assert not pending, \
            'unresolved pairs: %d' % len(pending)
        jsr = np.array([js_by_pk[p] for p in pks],
                       dtype=np.float64)
        jac = np.array(
            [len(set(top_by_pk[p][0])
                 & set(top_by_pk[p][1])) / 10.0
             for p in pks], dtype=np.float64)
        JSR[cname] = {'js': jsr, 'jac': jac}
        log('%s dist stats: js mean=%.6f med=%.6f '
            'jac mean=%.4f pend_max=%d'
            % (cname, jsr.mean(),
               float(np.median(jsr)), jac.mean(),
               pend_max))
    log('%s done (%.1fs)'
        % (cname, time.time() - t0))
ABL['mode'] = None
for h in handles:
    h.remove()

np.savez(os.path.join(OUT, 'ablation_readout.npz'),
         m_base_check=m_base,
         **{('%s__%s' % (c, k)): RES[c][k]
            for c in COND_NAMES
            for k in ('m', 'y', 'n', 'nx')},
         **{('%s__%s' % (c, k)): JSR[c][k]
            for c in DIST_CONDS
            for k in ('js', 'jac')},
         pk=pkB, cond=condB, truth=truthB)
log('readout saved')

# ================================================================
# analysis
# ================================================================
def pair_stats(mv):
    mP = mv[ip]
    mA1 = mv[ia]
    dpair = mP - mA1
    return (float(dpair.mean()),
            float(np.median(dpair)),
            float((dpair > 0).mean()))


out_conds = {}
for cname in COND_NAMES:
    mp, medp, auc = pair_stats(RES[cname]['m'])
    ym = float(RES[cname]['y'].mean())
    yP = float(RES[cname]['y'][ip].mean())
    yA1 = float(RES[cname]['y'][ia].mean())
    n_flip = int((RES[cname]['nx']
                  != RES['baseline']['nx']).sum())
    out_conds[cname] = {
        'mpair_mean': mp, 'mpair_median': medp,
        'auc_truth_m': auc, 'yes_prob_mean': ym,
        'yes_prob_P': yP, 'yes_prob_A1': yA1,
        'n_argmax_flip_vs_baseline': n_flip}
    log('%s: mpair=%.4f (med %.4f) AUC=%.4f '
        'yes_prob=%.6f flips=%d'
        % (cname, mp, medp, auc, ym, n_flip))

m0 = out_conds['baseline']['mpair_mean']


def drel(cname):
    return (out_conds[cname]['mpair_mean']
            - m0) / (abs(m0) + 1e-9)


d20 = drel('abl_L20_mlp')
d_joint28 = drel('abl_L20_28_mlp_joint')
d_joint32 = drel('abl_L20_32_mlp_joint')
d_l32 = drel('abl_L32_mlp')
d_single_sum = (d20 + D14_SINGLE['L24_mlp']
                + D14_SINGLE['L28_mlp'])
log('dmp_rel: L20=%+.4f L32=%+.4f '
    'joint20_28=%+.4f joint20_32=%+.4f | '
    'single_sum(L20+L24f+L28f)=%+.4f '
    '(L24f=%+.4f L28f=%+.4f)'
    % (d20, d_l32, d_joint28, d_joint32,
       d_single_sum, D14_SINGLE['L24_mlp'],
       D14_SINGLE['L28_mlp']))

joint_v = ('write_joint_collapse'
           if d_joint28 <= -0.30
           else ('write_joint_partial'
                 if d_joint28 <= -0.10
                 else 'write_elsewhere'))
if d_joint28 < d_single_sum - 0.05:
    super_v = 'superadditive'
elif d_joint28 > d_single_sum + 0.05:
    super_v = 'subadditive'
else:
    super_v = 'additive'

js_b = JSR['baseline']['js']
js_a = JSR['abl_L32_mlp']['js']
jac_b = JSR['baseline']['jac']
jac_a = JSR['abl_L32_mlp']['jac']
djs = js_a - js_b
djac = jac_a - jac_b
mean_djs = float(djs.mean())
sr_djs = float((djs > 0).mean())
if mean_djs > 0 and sr_djs >= 0.60:
    erase_v = 'erase_serves_generation'
elif mean_djs <= 0 and sr_djs <= 0.40:
    erase_v = 'erase_generation_neutral'
else:
    erase_v = 'erase_mixed'
log('ERASE PURPOSE: js_base=%.6f js_abl=%.6f '
    'delta=%+.6f sign_rate=%.3f | jac_base=%.4f '
    'jac_abl=%.4f delta=%+.4f -> %s'
    % (js_b.mean(), js_a.mean(), mean_djs, sr_djs,
       jac_b.mean(), jac_a.mean(), float(djac.mean()),
       erase_v))
log('VERDICT: %s|%s|%s' % (joint_v, super_v,
                           erase_v))

m_cons = float(np.abs(
    RES['baseline']['m'] - m_base).max())

results = {
    'verdict': '%s|%s|%s' % (joint_v, super_v,
                             erase_v),
    'conditions': out_conds,
    'dmp_rel': {
        'L20_mlp': d20,
        'L32_mlp': d_l32,
        'joint20_28': d_joint28,
        'joint20_32': d_joint32,
        'frozen_L24_mlp_3114': D14_SINGLE['L24_mlp'],
        'frozen_L28_mlp_3114': D14_SINGLE['L28_mlp'],
        'single_sum_L20_L24_L28': d_single_sum,
        'joint_minus_single_sum':
            d_joint28 - d_single_sum},
    'erase_purpose': {
        'js_base_mean': float(js_b.mean()),
        'js_abl_mean': float(js_a.mean()),
        'js_delta_mean': mean_djs,
        'js_delta_sign_rate': sr_djs,
        'js_base_median':
            float(np.median(js_b)),
        'js_abl_median': float(np.median(js_a)),
        'jac_base_mean': float(jac_b.mean()),
        'jac_abl_mean': float(jac_a.mean()),
        'jac_delta_mean': float(djac.mean())},
    'selfcheck_rel': sc_log,
    'gates': seal['gates'],
    'm_consistency_max_abs_vs_3113': m_cons,
    'smoke': SMOKE, 'n_records': NB,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S')}
with io.open(os.path.join(OUT, 'result.json'), 'w',
             encoding='utf-8') as f:
    json.dump(results, f, indent=1,
              ensure_ascii=False)
log('result.json written (m_consistency=%.3e)'
    % m_cons)
log('Phase 3115 done (%.1fs)' % (time.time() - T0))
print('PHASE3115_DONE verdict=%s|%s|%s'
      % (joint_v, super_v, erase_v))
