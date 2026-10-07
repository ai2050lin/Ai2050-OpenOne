# -*- coding: utf-8 -*-
"""Phase 3113 (Omega-P111): artifact separation + write-in
localization for the truth broadcast (3107-3112 readout arc).

TWO preregistered parts, one shared question: is the
record-level global belief broadcast (3111-3112) a real
in-context belief signal, and which components write it?

PART A - ARTIFACT SEPARATION (offline, frozen captures).
  3105 material is ALREADY token-matched at the pair level:
  P vs A1 share pair, facts block, query; they differ in
  EXACTLY 1 token (line-k predicate).  Any single-coordinate
  truth signal that survives WITHIN such pairs cannot be a
  data-construction artifact (pair identity / vocabulary
  differences) - it must track the truth-relevant difference.
  A1 (3105, main verdict): within-pair per-coordinate
    directional consistency frac_i[h_c(P_i) > h_c(A1_i)],
    direction-free = max(frac, 1-frac), median over 2560
    coordinates, per slot (slots = model layers
    4,8,12,16,20,24,28,32,36fn - SLOT INDEX CLARIFICATION
    for 3112: its "L0..L8" are these slots, so its
    "emerge L6" is model layer 28, window L4-L6 = layers
    20-28).
    Gates (slot 8 = final norm, ref slot 6):
      median >= 0.85 -> belief_robust
      0.70-0.85      -> partial_confound
      < 0.70         -> artifact_dominated
  A1b within-pair linear readout: w_slot = mean D(P-A1) over
    train pairs; test-pair sign agreement of D.w (raw and
    pair-centered).  Negative control A1-A2 (same pair, same
    truth=0, differs 1 token in query predicate, carries
    relation-presence cue): D(A1-A2).w sign rate should sit
    at chance; deviation -> cue_leakage.
  A2 (3106 cross-material): within-unit truth=1 vs truth=0
    condition-mean difference, same per-coordinate stat;
    slot-8 median >= 0.70 -> within_unit_replicated.

PART B - WRITE-IN LOCALIZATION (GPU, fresh forward).
  Rebuild 3105 main prompts from frozen material.json
  (structure identical; line ORDER re-randomized with a
  deterministic crc32 seed because the original run used
  salted hash() - registered as a material-variant caveat;
  A6 multiset checks re-asserted on rebuilt texts).
  Hook model layers {12,20,24,28,32} (window 20-28 =
  3112's slot 4-6, plus early/late controls), last position:
    h_in (block input), attn per-head output taken at the
    o_proj INPUT side (the only per-head-legal cut, R55),
    mlp output (down_proj output), h_out (block output).
  Zero-fit truth direction w_dn = W_yes - W_no.
  Per component: s = comp . w_dn; cross-pair direction-free
  AUC; within-pair P-A1 mean difference ds (the WRITTEN
  truth signal); concentration of |ds| over 32 heads at
  layer 28 (top-8 share, Gini, #heads with within-pair
  frac >= 0.70).
  Gates (layer 28):
    top8 share >= 0.50  -> write_in_concentrated
    0.25-0.50           -> moderately_concentrated
    < 0.25              -> write_in_distributed
  Self-checks: A1 determinism (double forward diff = 0);
  residual identity h_out - h_in = o_proj(attn_cat) + mlp
  (relative L2 < 0.02, bf16); A6 multiset on rebuilt texts.

Verdict = A1 | A2 | B.
SMOKE=1: smoke captures for A, smoke material for B.

Output: tests/glm5/result/rdc_query_construction_20260913/
        phase3113/omega_p111_artifact_writein/
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
from collections import Counter
from datetime import datetime

import numpy as np

SMOKE = os.environ.get('SMOKE', '0') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
R13 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913')
D05 = os.path.join(R13, 'phase3105',
                   'omega_p103_incontext_truth_consistency')
D06 = os.path.join(R13, 'phase3106',
                   'omega_p104_composition_dose_depth')
NAME = 'omega_p111_artifact_writein'
OUT = os.path.join(R13, 'phase3113', NAME)
if SMOKE:
    D05 = os.path.join(D05, 'smoke')
    D06 = os.path.join(D06, 'smoke')
    OUT = os.path.join(OUT, 'smoke')
os.makedirs(OUT, exist_ok=True)
LOGF = os.path.join(OUT, 'run_log.txt')
T0 = time.time()
SEED = 31130
SLOTS = 9
SLOT_FN = 8
SLOT_REF = 6
LAST = 5
LAM = 0.01

log_lines = []


def log(msg):
    line = '[%7.1fs] %s' % (time.time() - T0, msg)
    log_lines.append(line)
    with io.open(LOGF, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def sha8(path):
    h = hashlib.sha256()
    with io.open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()[:8]


def auc_score(y, s):
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    srt = s[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) \
                and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = \
                (i + 1 + j + 1) / 2.0
        i = j + 1
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    return float((ranks[y == 1].sum()
                  - n1 * (n1 + 1) / 2.0)
                 / (n1 * n0))


seal = {
    'phase': 3113,
    'name': NAME,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'seed': SEED,
    'A_gate': 'within-pair per-coordinate direction-'
              'free consistency median at slot 8: '
              '>=0.85 belief_robust / 0.70-0.85 '
              'partial_confound / <0.70 artifact_'
              'dominated; slot 6 reference',
    'A_neg': 'A1-A2 (both truth=0, cue-carrying) '
             'projection on w(P-A1): sign rate in '
             '[0.40,0.60] -> cue_independent else '
             'cue_leakage_present',
    'A2_gate': '3106 within-unit slot-8 median >= '
               '0.70 -> within_unit_replicated',
    'B_layers': [12, 20, 24, 28, 32],
    'B_gate': 'layer 28 top-8 |ds| head share >= 0.50 '
              'write_in_concentrated / 0.25-0.50 '
              'moderately_concentrated / <0.25 '
              'write_in_distributed',
    'B_direction': 'zero-fit w_dn = W_yes - W_no; '
                   'lambda %g only if ridge used'
                   % LAM,
    'slot_clarification': '3105/3106 slots map to '
                          'model layers [4,8,12,16,'
                          '20,24,28,32,36fn]; 3112 '
                          '"L6" = model layer 28; '
                          'window L4-L6 = layers '
                          '20-28',
    'material_variant': 'B rebuilds 3105 prompts; '
                        'line order re-randomized '
                        'with crc32 (original hash() '
                        'unsalted, unreproducible); '
                        'structure + A6 multiset '
                        're-asserted',
}
with io.open(os.path.join(OUT, 'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, indent=1)
log('Phase 3113 Omega-P111 start; SMOKE=%d OUT=%s'
    % (SMOKE, OUT))
log('design sealed (pre-computation)')

# ================================================================
# PART A - artifact separation (offline)
# ================================================================
cap5 = np.load(os.path.join(D05, 'capture.npz'),
               allow_pickle=False)
mat5 = json.load(io.open(os.path.join(D05,
                                      'material.json'),
                         encoding='utf-8'))
X5 = cap5['X']            # [N,6,9,2560] fp16
cond5 = cap5['cond']
truth5 = cap5['truth']
tag5 = cap5['tag']
split5 = cap5['split']
N5 = int(X5.shape[0])
HID = int(X5.shape[3])
S5 = int(X5.shape[2])
FN5 = S5 - 1
REF5 = 6 if S5 == 9 else S5 - 2
log('3105 capture loaded N=%d X=%s' % (N5, str(X5.shape)))
log('slots=%d fn=%d ref=%d (formal map: 9/8/6)'
    % (S5, FN5, REF5))

# --- rebuild pair keys for main records (order:
# split train/val/test/testE x pair x (P,A1,A2))
main_idx = np.where(tag5 == 'main')[0]
assert len(main_idx) == 3 * sum(
    len(mat5['splits'][s]) for s in
    ('train', 'val', 'test', 'testE'))
assert list(cond5[main_idx[:3]]) == ['P', 'A1', 'A2']
pair_of = {}
k = 0
pair_list = []
for sname in ('train', 'val', 'test', 'testE'):
    for p in mat5['splits'][sname]:
        pk = '%d_%d' % tuple(p)
        for c in ('P', 'A1', 'A2'):
            assert str(cond5[main_idx[k]]) == c
            pair_of[int(main_idx[k])] = (pk, c, sname)
            k += 1
        pair_list.append((pk, sname))
NPAIR = len(pair_list)
train_set = set(pk for pk, sn in pair_list
                if sn == 'train')
test_set = set(pk for pk, sn in pair_list
               if sn == 'test')
log('pair rebuild ok: %d pairs (train=%d test=%d)'
    % (NPAIR, len(train_set), len(test_set)))

Xl = X5[:, LAST].astype(np.float32)   # [N,9,2560]
hP = {}
hA1 = {}
hA2 = {}
for i, c in pair_of.items():
    (pk, cc, sn) = c
    if cc == 'P':
        hP[pk] = i
    elif cc == 'A1':
        hA1[pk] = i
    else:
        hA2[pk] = i
pks = sorted(hP.keys())
assert set(hA1) == set(pks) and set(hA2) == set(pks)
ip = np.array([hP[p] for p in pks])
ia = np.array([hA1[p] for p in pks])
i2 = np.array([hA2[p] for p in pks])
pk_arr = np.array(pks)
tr_m = np.array([p in train_set for p in pks])
te_m = np.array([p in test_set for p in pks])

D_p_a1 = Xl[ip] - Xl[ia]        # [NPAIR,9,2560]
D_a1_a2 = Xl[ia] - Xl[i2]       # negative control
frac_p = (D_p_a1 > 0).mean(0)   # [9,2560]
cons_p = np.maximum(frac_p, 1.0 - frac_p)
frac_n = (D_a1_a2 > 0).mean(0)
cons_n = np.maximum(frac_n, 1.0 - frac_n)

med_p = np.array([float(np.median(cons_p[sl]))
                  for sl in range(S5)])
med_n = np.array([float(np.median(cons_n[sl]))
                  for sl in range(S5)])
p90_p = np.array([float(np.percentile(cons_p[sl], 90))
                  for sl in range(S5)])
f85_p = np.array([float((cons_p[sl] >= 0.85).mean())
                  for sl in range(S5)])
f70_p = np.array([float((cons_p[sl] >= 0.70).mean())
                  for sl in range(S5)])
for sl in range(S5):
    log('A1 slot%d: within-pair med=%.4f p90=%.4f '
        'frac>=.85=%.3f frac>=.70=%.3f | negctl '
        'med=%.4f'
        % (sl, med_p[sl], p90_p[sl], f85_p[sl],
           f70_p[sl], med_n[sl]))

# global (cross-pair) reference at fn/ref slots
y_main = truth5[main_idx].astype(np.int32)
for sl in (REF5, FN5):
    auc_g = np.zeros(HID, dtype=np.float64)
    Xs = Xl[main_idx, sl]
    for c0 in range(HID):
        a = auc_score(y_main, Xs[:, c0])
        auc_g[c0] = max(a, 1 - a)
    log('A1 slot%d global cross-pair single-coord '
        'median=%.4f (3111 baseline %.4f)'
        % (sl, float(np.median(auc_g)),
           0.9222 if sl == FN5 else -1))

# within-pair linear readout w = mean D (train pairs)
W_sl = D_p_a1[tr_m].mean(0)      # [S5,2560]
proj_raw = np.einsum('psh,sh->ps',
                     D_p_a1[te_m], W_sl)
sign_raw = proj_raw > 0                    # [NTE,S5]
rate_raw = sign_raw.mean(0)
Dc = D_p_a1 - D_p_a1.mean(0, keepdims=True)
Wc = Dc[tr_m].mean(0)
Dte_c = Dc[te_m]
proj_ctr = np.einsum('psh,sh->ps', Dte_c, Wc)
sign_ctr = proj_ctr > 0
rate_ctr = sign_ctr.mean(0)
proj_neg = np.einsum('psh,sh->ps', D_a1_a2, W_sl)
neg_rate = (proj_neg > 0).mean(0)
mean_neg_rate = float(neg_rate[FN5])
log('A1b test-pair sign rate raw=%s'
    % json.dumps([float(x) for x in rate_raw]))
log('A1b test-pair sign rate centered=%s'
    % json.dumps([float(x) for x in rate_ctr]))
log('A1b neg control: A1-A2 proj on w sign rate '
    '(slot8)=%.4f' % mean_neg_rate)
cue_flag = ('cue_independent'
            if 0.40 <= mean_neg_rate <= 0.60
            else 'cue_leakage_present')

m_within = float(med_p[FN5])
m_ref = float(med_p[REF5])
if m_within >= 0.85:
    a1_verdict = 'belief_robust'
elif m_within >= 0.70:
    a1_verdict = 'partial_confound'
else:
    a1_verdict = 'artifact_dominated'
log('A1 VERDICT (slot8 median %.4f, slot6 %.4f): %s'
    % (m_within, m_ref, a1_verdict))

# --- A2: 3106 within-unit
cap6 = np.load(os.path.join(D06, 'capture.npz'),
               allow_pickle=False)
X6 = cap6['X']
unit6 = cap6['unit']
truth6 = cap6['truth']
tag6 = cap6['tag']
Xl6 = X6[:, LAST].astype(np.float32)
S6 = int(X6.shape[2])
assert S6 == S5, (S6, S5)
units = {}
for i in range(len(unit6)):
    if str(tag6[i]) not in ('chain', 'scatter'):
        continue
    units.setdefault(str(unit6[i]), []).append(i)
D6 = []
for u, idxs in units.items():
    idxs = np.array(idxs)
    yt = truth6[idxs].astype(np.int32)
    it = idxs[yt == 1]
    if_ = idxs[yt == 0]
    if len(it) and len(if_):
        D6.append(Xl6[it].mean(0) - Xl6[if_].mean(0))
if len(D6) == 0:
    log('A2: no usable units -> skipped')
    a2_flag = 'within_unit_not_available'
    med6 = np.full(S6, np.nan)
else:
    D6 = np.array(D6)                # [NU,S6,2560]
    frac6 = (D6 > 0).mean(0)
    cons6 = np.maximum(frac6, 1.0 - frac6)
    med6 = np.array([float(np.median(cons6[sl]))
                     for sl in range(S6)])
    for sl in range(S6):
        log('A2 slot%d: within-unit med=%.4f'
            % (sl, med6[sl]))
    a2_flag = ('within_unit_replicated'
               if float(med6[FN5]) >= 0.70
               else 'within_unit_not_replicated')
    log('A2 VERDICT (slot8 median %.4f, n_units=%d):'
        ' %s' % (float(med6[FN5]), len(D6),
                 a2_flag))
del Xl, Xl6, D_p_a1, D_a1_a2, Dc, D6, X5, X6
gc.collect()

# ================================================================
# PART B - write-in localization (GPU)
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
assert NL == 36 and HIDb == HID, (NL, HIDb, HID)
_ow = model.model.layers[0].self_attn.o_proj.weight
assert NH * HD == int(_ow.shape[1]), \
    (NH, HD, int(_ow.shape[1]))
LAY_B = [12, 20, 24, 28, 32]
log('model loaded NL=%d HID=%d heads=%d head_dim=%d '
    '(o_proj in=%d)'
    % (NL, HIDb, NH, HD, int(_ow.shape[1])))

WU = model.lm_head.weight.detach()
YES_ID = int(mat5['yes_id'])
NO_ID = int(mat5['no_id'])
w_dn = (WU[YES_ID] - WU[NO_ID]).float().cpu().numpy()
emb = model.get_input_embeddings().weight.detach()


def line_spans(pos, ents, ls, PREDS, lr, lo):
    a_s = pos + len('The ')
    b_s = a_s + len(ents[ls])
    a_r = b_s + len(' ')
    b_r = a_r + len(PREDS[lr])
    a_o = b_r + len(' the ')
    b_o = a_o + len(ents[lo])
    return a_s, b_o


def build_prompt_rb(mat, s, o, lrel, qrel):
    """Rebuild a 3105 prompt.  Identical structure to
    phase3105 build_prompt; line ORDER from crc32 seed
    (original used salted hash() - unreproducible)."""
    ents = mat['entities']
    PREDS = mat['predicates']
    D = [tuple(d) for d in
         mat['distractors']['%d_%d' % (s, o)]]
    k = mat['kline']['%d_%d' % (s, o)]
    lines = [(s, lrel, o)] + list(D)
    rng2 = random.Random(zlib.crc32(
        ('%d_%d_ord5' % (s, o)).encode('ascii')))
    order = list(range(8))
    rng2.shuffle(order)
    lines = [lines[i] for i in order]
    ci = lines.index((s, lrel, o))
    lines[ci], lines[k] = lines[k], lines[ci]
    text = 'Facts:'
    spans = {}
    pos = len(text)
    for li, (ls, lr, lo) in enumerate(lines):
        seg = ' The %s %s the %s.' % (ents[ls],
                                      PREDS[lr],
                                      ents[lo])
        (a_s, b_o) = line_spans(pos + 1, ents, ls,
                                PREDS, lr, lo)
        text += seg
        pos += len(seg)
        if li == k:
            spans['crit'] = (a_s, b_o)
    qseg = (' Query: The %s %s the %s. Is this query '
            'true? Answer:' % (ents[s], PREDS[qrel],
                               ents[o]))
    text += qseg
    qs = text.rindex('The %s %s the %s.'
                     % (ents[s], PREDS[qrel],
                        ents[o]))
    (a_s, b_o) = line_spans(qs, ents, s, PREDS,
                            qrel, o)
    spans['query'] = (a_s, b_o)
    return text, spans


# --- rebuilt records (main only, 3105 structure)
p2r = mat5['pair2rel']
frel = mat5['false_rels']
recs = []
pair_meta = []
for sname in ('train', 'val', 'test', 'testE'):
    for p in mat5['splits'][sname]:
        (s, o) = p
        pk = '%d_%d' % (s, o)
        r = p2r[pk]
        ri1, ri2 = frel[pk]
        for (cc, qrel, lrel, lab) in (
                ('P', r, r, 1),
                ('A1', r, ri1, 0),
                ('A2', ri2, ri1, 0)):
            recs.append({'pk': pk, 'cond': cc,
                         'truth': lab,
                         'split': sname})
        pair_meta.append((pk, sname))
NB = len(recs)
log('rebuilt records: %d (%d pairs)'
    % (NB, len(pair_meta)))
pk_arrB = np.array([x['pk'] for x in recs])
condB = np.array([x['cond'] for x in recs])
truthB = np.array([x['truth'] for x in recs],
                  dtype=np.int32)
hP_B = {}
hA1_B = {}
hA2_B = {}
for i, x in enumerate(recs):
    if x['cond'] == 'P':
        hP_B[x['pk']] = i
    elif x['cond'] == 'A1':
        hA1_B[x['pk']] = i
    else:
        hA2_B[x['pk']] = i
pksB = sorted(hP_B.keys())
ipB = np.array([hP_B[p] for p in pksB])
iaB = np.array([hA1_B[p] for p in pksB])

# A6 multiset re-assertion on rebuilt texts (20 pairs)
pred_ids = mat5['pred_ids']
n_chk = 0
for pk in pksB[:20]:
    (s, o) = (int(x) for x in pk.split('_'))
    r = p2r[pk]
    ri1, ri2 = frel[pk]
    tP, _ = build_prompt_rb(mat5, s, o, r, r)
    tA1, _ = build_prompt_rb(mat5, s, o, ri1, r)
    tp = Counter(tok.encode(tP, add_special_tokens=False))
    ta = Counter(tok.encode(tA1,
                            add_special_tokens=False))
    rp = pred_ids[r]
    ra = pred_ids[ri1]
    assert (tp - ta) == Counter({rp: 1}), (pk, tp - ta)
    assert (ta - tp) == Counter({ra: 1}), (pk, ta - tp)
    n_chk += 1
log('A6 multiset re-asserted on %d rebuilt pairs'
    % n_chk)

# --- hooks
feats = {}


class Rec:
    pass


cur = Rec()
cur.ti = -1
cur.slot = {L: {} for L in LAY_B}


def mk_pre_layer(L):
    def hook(mod, args):
        t = args[0]
        v = t[0, cur.ti, :] if t.dim() == 3 \
            else t.reshape(t.shape[0], -1)[cur.ti % t.shape[0]]
        feats.setdefault(L, {})['h_in'] = v.detach()
    return hook


def mk_pre_o(L):
    def hook(mod, args):
        t = args[0]
        v = t[0, cur.ti, :] if t.dim() == 3 \
            else t.reshape(t.shape[0], -1)[cur.ti % t.shape[0]]
        feats.setdefault(L, {})['attn'] = v.detach()
    return hook


def mk_post_mlp(L):
    def hook(mod, mod_in, out):
        t = out[0] if isinstance(out, tuple) else out
        v = t[0, cur.ti, :] if t.dim() == 3 \
            else t[cur.ti, :]
        feats.setdefault(L, {})['mlp'] = v.detach()
    return hook


def mk_post_layer(L):
    def hook(mod, mod_in, out):
        t = out[0] if isinstance(out, tuple) else out
        v = t[0, cur.ti, :] if t.dim() == 3 \
            else t[cur.ti, :]
        feats.setdefault(L, {})['h_out'] = v.detach()
    return hook


hs_handle = []
for L in LAY_B:
    blk = model.model.layers[L]
    hs_handle.append(
        blk.register_forward_pre_hook(mk_pre_layer(L)))
    hs_handle.append(
        blk.self_attn.o_proj.register_forward_pre_hook(
            mk_pre_o(L)))
    hs_handle.append(
        blk.mlp.register_forward_hook(mk_post_mlp(L)))
    hs_handle.append(
        blk.register_forward_hook(mk_post_layer(L)))


def forward_rec(text):
    ids = tok(text, add_special_tokens=False)['input_ids']
    cur.ti = len(ids) - 1
    feats.clear()
    t = torch.tensor([ids], device='cuda')
    with torch.inference_mode():
        out = model(t, output_hidden_states=True,
                    use_cache=False)
        hfn = model.model.norm(
            out.hidden_states[NL][0, cur.ti, :])
        m_val = float((hfn.float().cpu().numpy()
                       @ w_dn).item())
        del out
    snap = {L: dict(feats[L]) for L in LAY_B}
    return snap, len(ids), m_val


# --- A1 determinism (double forward)
_s0 = int(pksB[0].split('_')[0])
_o0 = int(pksB[0].split('_')[1])
_r0 = p2r[pksB[0]]
t0b, n1b, m1b = forward_rec(
    build_prompt_rb(mat5, _s0, _o0, _r0, _r0)[0])
t1b, n2b, m2b = forward_rec(
    build_prompt_rb(mat5, _s0, _o0, _r0, _r0)[0])
d_det = 0.0
for L in LAY_B:
    for key in ('h_in', 'attn', 'mlp', 'h_out'):
        d_det = max(d_det, float(
            (t0b[L][key].float()
             - t1b[L][key].float()).abs().max()))
d_det = max(d_det, abs(m1b - m2b))
log('A1 determinism (B): max diff = %.3e' % d_det)
assert d_det == 0.0

# --- residual identity self-check (one record)
# o_proj applied MANUALLY from frozen weights (module call
# would fire its own pre-hook and corrupt the snapshot).
rel_err = 0.0
for L in LAY_B:
    blk = model.model.layers[L]
    Wo = blk.self_attn.o_proj.weight.detach().float()
    bo = blk.self_attn.o_proj.bias
    bo = bo.detach().float() if bo is not None else 0.0
    attn_cat = t0b[L]['attn'].unsqueeze(0).float()
    proj = (attn_cat @ Wo.T + bo)[0]
    mlp_o = t0b[L]['mlp'].float()
    lhs = (t0b[L]['h_out'].float()
           - t0b[L]['h_in'].float())
    rhs = proj + mlp_o
    rel = float((lhs - rhs).norm()
                / (lhs.norm() + 1e-9))
    rel_err = max(rel_err, rel)
log('residual identity check: max rel L2 = %.2e'
    % rel_err)
assert rel_err < 0.02, rel_err

# --- capture loop
A = np.zeros((NB, len(LAY_B), NH, HD),
             dtype=np.float16)
ML = np.zeros((NB, len(LAY_B), HID),
              dtype=np.float16)
HIN = np.zeros((NB, len(LAY_B), HID),
               dtype=np.float16)
HOUT = np.zeros((NB, len(LAY_B), HID),
                dtype=np.float16)
MB = np.zeros(NB, dtype=np.float32)
t_cap = time.time()
for i, x in enumerate(recs):
    (s, o) = (int(v) for v in x['pk'].split('_'))
    r = p2r[x['pk']]
    ri1, ri2 = frel[x['pk']]
    if x['cond'] == 'P':
        (qrel, lrel) = (r, r)
    elif x['cond'] == 'A1':
        (qrel, lrel) = (r, ri1)
    else:
        (qrel, lrel) = (ri2, ri1)
    text, _ = build_prompt_rb(mat5, s, o, lrel, qrel)
    snap, ntok, m_val = forward_rec(text)
    for li, L in enumerate(LAY_B):
        A[i, li] = snap[L]['attn'].float().cpu() \
            .numpy().reshape(NH, HD).astype(np.float16)
        ML[i, li] = snap[L]['mlp'].float().cpu() \
            .numpy().astype(np.float16)
        HIN[i, li] = snap[L]['h_in'].float().cpu() \
            .numpy().astype(np.float16)
        HOUT[i, li] = snap[L]['h_out'].float().cpu() \
            .numpy().astype(np.float16)
    MB[i] = m_val
    if (i + 1) % 200 == 0:
        log('capture %d/%d' % (i + 1, NB))
log('capture done (%.1fs)' % (time.time() - t_cap))
np.savez_compressed(
    os.path.join(OUT, 'capture_b.npz'),
    attn=A, mlp=ML, h_in=HIN, h_out=HOUT, m=MB,
    pk=pk_arrB, cond=condB, truth=truthB,
    layers=np.array(LAY_B))
log('capture_b saved (%.1f MB)'
    % (os.path.getsize(os.path.join(
        OUT, 'capture_b.npz')) / 1e6))

# --- component analysis (zero-fit direction w_dn)
# per-head vectors live at the o_proj INPUT side (head_dim
# each); their residual-stream contribution is the head's
# o_proj column block, so the scalar readout of head h is
# head_vec . (W_o[:, block_h]^T . w_dn).
U_layers = np.zeros((len(LAY_B), NH, HD),
                    dtype=np.float32)
for li, L in enumerate(LAY_B):
    Wo = model.model.layers[L].self_attn \
        .o_proj.weight.detach().float().cpu().numpy()
    for h in range(NH):
        blk_h = Wo[:, h * HD:(h + 1) * HD]
        U_layers[li, h] = blk_h.T @ w_dn
s_attn = np.einsum('nlhd,lhd->nlh',
                   A.astype(np.float32),
                   U_layers)          # [NB,5,NH]
s_mlp = np.einsum('nld,d->nl',
                  ML.astype(np.float32), w_dn)
s_in = np.einsum('nld,d->nl',
                 HIN.astype(np.float32), w_dn)
s_out = np.einsum('nld,d->nl',
                  HOUT.astype(np.float32), w_dn)
mB_auc = auc_score(truthB, MB)
log('B: m rebuilt AUC=%.4f' % mB_auc)

yB = truthB
for li, L in enumerate(LAY_B):
    auc_in = auc_score(yB, s_in[:, li])
    auc_out = auc_score(yB, s_out[:, li])
    auc_mlp = auc_score(yB, s_mlp[:, li])
    # head-level AUCs (direction-free)
    auc_h = np.zeros(NH)
    for h in range(NH):
        a = auc_score(yB, s_attn[:, li, h])
        auc_h[h] = max(a, 1 - a)
    # within-pair ds (P - A1)
    dp = s_attn[ipB, li] - s_attn[iaB, li]
    ds_h = dp.mean(0)
    ds_mlp = float((s_mlp[ipB, li]
                    - s_mlp[iaB, li]).mean())
    ds_blk = float((s_out[ipB, li]
                    - s_in[iaB, li]).mean()
                   - (s_in[ipB, li]
                      - s_in[iaB, li]).mean())
    ds_blk = float(((s_out[ipB, li]
                     - s_out[iaB, li])
                    - (s_in[ipB, li]
                       - s_in[iaB, li])).mean())
    absds = np.abs(ds_h)
    order_h = np.argsort(-absds)
    top8_share = float(absds[order_h[:8]].sum()
                       / (absds.sum() + 1e-12))
    gini = float((np.abs(absds[:, None]
                         - absds[None, :]).sum()
                  / (2.0 * len(absds)
                     * absds.sum() + 1e-12)))
    dp_frac = np.maximum((dp > 0).mean(0),
                         1 - (dp > 0).mean(0))
    n_sig = int((dp_frac >= 0.70).sum())
    log('B L%d: AUC in=%.4f out=%.4f mlp=%.4f '
        'headAUC max=%.4f |ds| top8=%.3f gini=%.3f '
        'n_head(frac>=.70)=%d ds_mlp=%.4f ds_blk=%.4f'
        % (L, auc_in, auc_out, auc_mlp,
           float(auc_h.max()), top8_share, gini,
           n_sig, ds_mlp, ds_blk))
    if L == 28:
        res28 = {
            'auc_in': auc_in, 'auc_out': auc_out,
            'auc_mlp': auc_mlp,
            'auc_head_max': float(auc_h.max()),
            'auc_head': [float(v) for v in auc_h],
            'ds_head': [float(v) for v in ds_h],
            'ds_head_order': [int(v)
                              for v in order_h],
            'ds_mlp': ds_mlp, 'ds_block': ds_blk,
            'top8_share': top8_share,
            'gini': gini, 'n_sig_head': n_sig,
            'within_frac_head':
                [float(v) for v in dp_frac]}

if res28['top8_share'] >= 0.50:
    b_verdict = 'write_in_concentrated'
elif res28['top8_share'] >= 0.25:
    b_verdict = 'moderately_concentrated'
else:
    b_verdict = 'write_in_distributed'
log('B VERDICT (L28 top8 share %.3f): %s'
    % (res28['top8_share'], b_verdict))

for h in hs_handle:
    h.remove()
del model, A, ML, HIN, HOUT
gc.collect()
torch.cuda.empty_cache()

# ================================================================
# Verdict & results
# ================================================================
verdict = '%s|%s|%s' % (a1_verdict, a2_flag, b_verdict)
log('VERDICT: %s' % verdict)
results = {
    'verdict': verdict,
    'A1': {
        'within_pair_median': med_p.tolist(),
        'within_pair_p90': p90_p.tolist(),
        'within_pair_frac85': f85_p.tolist(),
        'within_pair_frac70': f70_p.tolist(),
        'negctl_median': med_n.tolist(),
        'slot8_median': m_within,
        'slot6_median': m_ref,
        'verdict': a1_verdict,
        'sign_rate_raw':
            [float(v) for v in rate_raw],
        'sign_rate_centered':
            [float(v) for v in rate_ctr],
        'neg_proj_sign_rate_slot8': mean_neg_rate,
        'cue_flag': cue_flag,
        'n_pairs': NPAIR},
    'A2': {
        'within_unit_median': med6.tolist(),
        'slot8_median': float(med6[FN5]),
        'n_units': int(len(units)),
        'flag': a2_flag},
    'B': {
        'layers': LAY_B,
        'm_rebuilt_auc': mB_auc,
        'L28': res28,
        'verdict': b_verdict},
    'sanity': {
        'a1_determinism_b': d_det,
        'residual_identity_rel_l2': rel_err,
        'a6_multiset_pairs': n_chk},
    'gates': {
        'A1_thresholds': [0.85, 0.70],
        'B_top8_thresholds': [0.50, 0.25]},
    'smoke': SMOKE,
    'n_records_b': NB,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S')}
with io.open(os.path.join(OUT, 'result.json'), 'w',
             encoding='utf-8') as f:
    json.dump(results, f, indent=1,
              ensure_ascii=False)
log('result.json written')
log('Phase 3113 done (%.1fs)' % (time.time() - T0))
print('PHASE3113_DONE verdict=%s' % verdict)
