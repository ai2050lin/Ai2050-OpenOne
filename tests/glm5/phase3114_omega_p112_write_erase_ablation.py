# -*- coding: utf-8 -*-
"""Phase 3114 (Omega-P112): causal ablation of the write
and erase phases found in 3113.

Preregistered question (3113 MEMO section 6): are the L20-28
MLP write-in, the L28 concentrated head write, and the L32
MLP erase CAUSALLY responsible for the end-of-context truth
margin, and does the erase serve next-token generation?

METHOD (intervention semantics follow the 3101 lesson:
reverse first, then re-create - here the component
contribution is REMOVED at its exact point and the residual
identity is asserted numerically):
  Ablations (forward hooks, batch 1, rebuilt 3105 main
  material, identical texts to 3113 B):
    baseline            no intervention
    abl_L28_top8_head   zero o_proj INPUT columns of the
                        top-8 |ds| heads at layer 28
                        (head ids frozen from 3113 result)
    abl_L24_mlp         zero L24 MLP output
    abl_L28_mlp         zero L28 MLP output
    abl_L32_mlp         zero L32 MLP output (the erase)
  Removing a head block at the o_proj input subtracts
  o_proj(block) from the residual stream - exact linear
  removal; zeroing the MLP output subtracts the MLP write.
  SELF-CHECK per condition (first record): captured h_out
  under ablation must equal h_out_clean - removed_component
  within rel L2 < 0.02 (bf16).

READOUT per record: m' = (W_yes - W_no) . h_fn(last) (this
equals the yes-minus-no logit margin for a bias-free
lm_head), yes/no softmax probabilities, argnext token.

GATES (pre-registered, on the within-pair P-A1 mean margin
difference mpair; relative change vs baseline):
  head gate:  abl_L28_top8_head  dmp <= -0.15 ->
              head_write_causal else head_write_not_causal
  mlp gate:   abl_L28_mlp        dmp <= -0.30 ->
              mlp_write_dominant else mlp_write_partial
  erase gate: abl_L32_mlp         dmp >= +0.30 ->
              erase_active else erase_not_active
  Reported alongside: layer-24 analogues, cross-pair AUC
  of m', yes-probability shifts (erase purposiveness,
  descriptive).

Verdict = head|mlp|erase.
SMOKE=1: smoke material (54 records), same pipeline.

Output: tests/glm5/result/rdc_query_construction_20260913/
        phase3114/omega_p112_write_erase_ablation/
Model: models/hf/qwen3-4b (BF16, eager, batch 1, no cache).
"""
import gc
import hashlib
import io
import json
import os
import random
import time
import zlib
from datetime import datetime

import numpy as np

SMOKE = os.environ.get('SMOKE', '0') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
R13 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913')
D13 = os.path.join(R13, 'phase3113',
                   'omega_p111_artifact_writein')
D05 = os.path.join(R13, 'phase3105',
                   'omega_p103_incontext_truth_consistency')
NAME = 'omega_p112_write_erase_ablation'
OUT = os.path.join(R13, 'phase3114', NAME)
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


res13 = json.load(io.open(
    os.path.join(D13, 'result.json'), encoding='utf-8'))
assert res13['verdict'] == \
    'belief_robust|within_unit_replicated|' \
    'write_in_concentrated'
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
    'phase': 3114,
    'name': NAME,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'top8_head_L28': res13['B']['L28']
                     ['ds_head_order'][:8],
    'ds_head_L28': res13['B']['L28']['ds_head'],
    'ds_mlp_L28': res13['B']['L28']['ds_mlp'],
    'ablations': ['baseline', 'abl_L28_top8_head',
                  'abl_L24_mlp', 'abl_L28_mlp',
                  'abl_L32_mlp'],
    'gates': {
        'head': 'abl_L28_top8_head dmp_rel <= -0.15 -> '
                'head_write_causal',
        'mlp': 'abl_L28_mlp dmp_rel <= -0.30 -> '
               'mlp_write_dominant',
        'erase': 'abl_L32_mlp dmp_rel >= +0.30 -> '
                 'erase_active'},
    'readout': 'm = (W_yes-W_no).h_fn(last) = yes-no '
               'logit margin; yes/no softmax probs; '
               'argnext',
    'selfcheck': 'h_out_ablated = h_out_clean - '
                 'removed_component, rel L2 < 0.02',
    'note': 'mpair = mean over pairs of (m(P) - m(A1)); '
            'heads frozen from 3113 result.json '
            '(pre-registration by freeze); per-head '
            'decomposition exists only for L28 (write '
            'peak) in 3113, so head ablation runs at L28 '
            'only and L24 enters as MLP-only control',
}
with io.open(os.path.join(OUT, 'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, indent=1)
log('Phase 3114 Omega-P112 start; SMOKE=%d OUT=%s'
    % (SMOKE, OUT))
log('design sealed (pre-computation); top8_L28=%s'
    % seal['top8_head_L28'])

# --- rebuilt records (identical to 3113 B) ---
import random as _rnd  # noqa: E402


def line_spans(pos, ents, ls, PREDS, lr, lo):
    a_s = pos + len('The ')
    b_s = a_s + len(ents[ls])
    a_r = b_s + len(' ')
    b_r = a_r + len(PREDS[lr])
    a_o = b_r + len(' the ')
    b_o = a_o + len(ents[lo])
    return a_s, b_o


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
    pos = len(text)
    for (ls, lr, lo) in lines:
        seg = ' The %s %s the %s.' % (ents[ls],
                                      PREDS[lr],
                                      ents[lo])
        text += seg
        pos += len(seg)
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

ABL = {'mode': None, 'layer': None, 'heads': []}


def mk_o_pre(L):
    def hook(mod, args):
        if ABL['mode'] == 'head' \
                and ABL['layer'] == L:
            t = args[0].clone()
            for h in ABL['heads']:
                t[..., h * HD:(h + 1) * HD] = 0
            return t
        return None
    return hook


def mk_mlp_post(L):
    def hook(mod, mod_in, out):
        if ABL['mode'] == 'mlp' \
                and ABL['layer'] == L:
            return torch.zeros_like(out)
        return None
    return hook


# capture hooks for the self-check (last position)
CCHK = {}
SNAP_CLEAN = {}


def mk_cap(L, key):
    def hook(mod, args):
        t = args[0]
        v = t[0, -1, :] if t.dim() == 3 \
            else t.reshape(t.shape[0], -1)[-1]
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


handles = []
ABL_LAYERS = [24, 28, 32]
for L in ABL_LAYERS:
    blk = model.model.layers[L]
    handles.append(
        blk.self_attn.o_proj.register_forward_pre_hook(
            mk_o_pre(L)))
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


def selfcheck(cond_name, snap):
    """First-record identity checks against the clean
    snapshot stored during the baseline condition.

    mlp ablation (exact, linear):
      (i)   h_in and o_proj input identical to clean
            (upstream untouched);
      (ii)  h_out_abl = h_out_clean - mlp_out_clean.
    head ablation (intervention is a linear removal at the
    o_proj input; MLP downstream recomputes on the ablated
    input, so the h_out identity is checked via the exact
    residual identity under ablation):
      (i)   captured o_proj input has exactly zero head
            blocks;
      (ii)  o_proj(attn_abl) = o_proj(attn_clean)
            - o_proj(removed blocks) (linear);
      (iii) h_out_abl = h_in + o_proj(attn_abl)
            + mlp_abl (residual identity under
            ablation).
    """
    ok_all = 0.0
    with torch.no_grad():
        for L in ABL_LAYERS:
            c = CCHK.get(L, {})
            s = snap.get(L, {})
            need = ('h_in', 'h_out', 'mlp', 'attn')
            if not all(k in c for k in need) \
                    or not all(k in s for k in need):
                continue
            blk = model.model.layers[L]
            if ABL['mode'] == 'mlp' \
                    and ABL['layer'] == L:
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
                lhs = c['h_out'].float()
                rhs = (s['h_out'].float()
                       - s['mlp'].float())
                r_id = float((lhs - rhs).norm()
                             / (s['h_out'].float()
                                .norm() + 1e-9))
                ok_all = max(ok_all, r_up, r_at, r_id)
            elif ABL['mode'] == 'head' \
                    and ABL['layer'] == L:
                nb = max(
                    float(c['attn']
                          [h * HD:(h + 1) * HD]
                          .float().norm())
                    for h in ABL['heads'])
                r_zero = 0.0 if nb == 0.0 else 1.0
                ac = s['attn']
                mask = torch.zeros(
                    ac.shape, device=ac.device,
                    dtype=ac.dtype)
                for h in ABL['heads']:
                    mask[h * HD:(h + 1) * HD] = \
                        ac[h * HD:(h + 1) * HD]
                got = blk.self_attn.o_proj(
                    c['attn'].unsqueeze(0)
                )[0].float()
                ref = (blk.self_attn.o_proj(
                    ac.unsqueeze(0))[0].float()
                    - blk.self_attn.o_proj(
                        mask.unsqueeze(0))[0]
                    .float())
                r_lin = float((got - ref).norm()
                              / (ref.norm() + 1e-9))
                resid = (c['h_out'].float()
                         - c['h_in'].float()
                         - got - c['mlp'].float())
                r_res = float(
                    resid.norm()
                    / (c['h_out'].float().norm()
                       + 1e-9))
                log('  sc L%d: zero=%d lin=%.2e '
                    'res=%.2e' % (L, r_zero == 0.0,
                                  r_lin, r_res))
                ok_all = max(ok_all, r_zero, r_lin,
                             r_res)
    CCHK.clear()
    return ok_all


CONDITIONS = [
    ('baseline', None, None, None),
    ('abl_L28_top8_head', 'head', 28,
     seal['top8_head_L28']),
    ('abl_L24_mlp', 'mlp', 24, None),
    ('abl_L28_mlp', 'mlp', 28, None),
    ('abl_L32_mlp', 'mlp', 32, None),
]
COND_NAMES = [c[0] for c in CONDITIONS]
RES = {c: {'m': np.zeros(NB, dtype=np.float32),
           'y': np.zeros(NB, dtype=np.float32),
           'n': np.zeros(NB, dtype=np.float32),
           'nx': np.zeros(NB, dtype=np.int32)}
       for c in COND_NAMES}
sc_log = {}
for (cname, mode, layer, heads) in CONDITIONS:
    ABL['mode'] = mode
    ABL['layer'] = layer
    ABL['heads'] = list(heads) if heads else []
    t0 = time.time()
    for i in range(NB):
        (m_v, y_p, n_p, nx) = forward_rec(texts[i])
        RES[cname]['m'][i] = m_v
        RES[cname]['y'][i] = y_p
        RES[cname]['n'][i] = n_p
        RES[cname]['nx'][i] = nx
        if i == 0:
            if mode is None:
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
    log('%s done (%.1fs)' % (cname, time.time() - t0))
ABL['mode'] = None
for h in handles:
    h.remove()

np.savez(os.path.join(OUT, 'ablation_readout.npz'),
         m_base_check=m_base,
         **{('%s__%s' % (c, k)): RES[c][k]
            for c in COND_NAMES
            for k in ('m', 'y', 'n', 'nx')},
         pk=pkB, cond=condB, truth=truthB)
log('readout saved')

# ================================================================
# analysis
# ================================================================
def pair_stats(mv):
    mP = mv[ip]
    mA1 = mv[ia]
    d = mP - mA1
    return float(d.mean()), float(np.median(d))


def auc_df(y, s):
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    a = float((ranks[y == 1].sum()
               - n1 * (n1 + 1) / 2.0) / (n1 * n0))
    return max(a, 1 - a)


truth = truthB.astype(np.int32)
out_conds = {}
for cname in COND_NAMES:
    mp, medp = pair_stats(RES[cname]['m'])
    auc = auc_df(truth, RES[cname]['m'])
    ym = float(RES[cname]['y'].mean())
    yP = float(RES[cname]['y'][ip].mean())
    yA1 = float(RES[cname]['y'][ia].mean())
    n_argmax_yes = int(sum(
        1 for i in range(NB)
        if RES[cname]['nx'][i] == YES_ID))
    out_conds[cname] = {
        'mpair_mean': mp, 'mpair_median': medp,
        'auc_truth_m': auc,
        'yes_prob_mean': ym,
        'yes_prob_P': yP, 'yes_prob_A1': yA1,
        'n_argmax_yes': n_argmax_yes}
    log('%s: mpair=%.4f (med %.4f) AUC=%.4f '
        'yes_prob=%.4f (P %.4f / A1 %.4f) '
        'argmax_yes=%d'
        % (cname, mp, medp, auc, ym, yP, yA1,
           n_argmax_yes))

m0 = out_conds['baseline']['mpair_mean']


def drel(cname):
    return (out_conds[cname]['mpair_mean']
            - m0) / (abs(m0) + 1e-9)


d_head = drel('abl_L28_top8_head')
d_mlp = drel('abl_L28_mlp')
d_erase = drel('abl_L32_mlp')
d_mlp24 = drel('abl_L24_mlp')
pred_head = -sum(seal['ds_head_L28'][h]
                 for h in seal['top8_head_L28']) \
    / (abs(m0) + 1e-9)
log('dmp_rel: L28top8head=%+.4f (linear pred %+.4f) '
    'L28mlp=%+.4f L32mlp=%+.4f | L24mlp=%+.4f'
    % (d_head, pred_head, d_mlp, d_erase,
       d_mlp24))

head_v = ('head_write_causal' if d_head <= -0.15
          else 'head_write_not_causal')
mlp_v = ('mlp_write_dominant' if d_mlp <= -0.30
         else 'mlp_write_partial')
erase_v = ('erase_active' if d_erase >= 0.30
           else 'erase_not_active')
verdict = '%s|%s|%s' % (head_v, mlp_v, erase_v)
log('VERDICT: %s' % verdict)

results = {
    'verdict': verdict,
    'conditions': out_conds,
    'dmp_rel': {
        'L28_top8_head': d_head,
        'L28_top8_head_linear_pred': pred_head,
        'L28_mlp': d_mlp,
        'L32_mlp': d_erase,
        'L24_mlp': d_mlp24},
    'selfcheck_rel': sc_log,
    'gates': seal['gates'],
    'top8_heads': {'L28': seal['top8_head_L28']},
    'yes_prob_shift_L32_erase':
        float(out_conds['abl_L32_mlp']['yes_prob_mean']
              - out_conds['baseline']
              ['yes_prob_mean']),
    'smoke': SMOKE, 'n_records': NB,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S')}
with io.open(os.path.join(OUT, 'result.json'), 'w',
             encoding='utf-8') as f:
    json.dump(results, f, indent=1,
              ensure_ascii=False)
log('result.json written')
log('Phase 3114 done (%.1fs)' % (time.time() - T0))
print('PHASE3114_DONE verdict=%s' % verdict)
