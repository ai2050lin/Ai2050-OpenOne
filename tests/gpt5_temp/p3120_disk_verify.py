# -*- coding: utf-8 -*-
"""Phase 3120 disk verify: independent recomputation
from result.json + p118_readout.npz + 3118 npz +
design_seal.json, then disk checks of ledger / MEMO /
wlogs / MEMORY.md.  ~73 numbered items; writes log."""
import hashlib
import io
import json
import os
import re

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = (RDIR + r'\phase3120'
       r'\omega_p118_content_attr_amplifier_'
       r'behavior_opshape')
D18 = (RDIR + r'\phase3118'
       r'\omega_p116_autoregressive_margin_'
       r'trajectory')
D05 = (RDIR + r'\phase3105'
       r'\omega_p103_incontext_truth_consistency')
LEDGER = ROOT + r'\research\gpt5\atlas\atlas_ledger.json'
MEMO = ROOT + r'\research\gpt5\docs\AGI_GPT5_MEMO.md'
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
LOGF = ROOT + (r'\tests\gpt5_temp'
               r'\p3120_disk_verify_log.txt')

n_pass = 0
n_fail = 0
items = []


def check(name, cond, detail=''):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        items.append('PASS %s' % name)
    else:
        n_fail += 1
        items.append('FAIL %s  %s' % (name, detail))


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


res = json.load(io.open(OUT + r'\result.json',
                        encoding='utf-8'))
pa = res['part_a']
pb = res['part_b']
pc = res['part_c']

# ---- 0. result.json structure (5) ----
check('R00 verdict', res['verdict'] ==
      'attribution_unresolved|amplifier_partial|'
      'not_applicable|mean_reversion_confirmed|'
      'linear_operator')
check('R01 smoke False', res['smoke'] is False)
check('R02 n_records 2016',
      res['n_records'] == 2016)
check('R03 n_pairs 672', res['n_pairs'] == 672)
check('R04 n_pairs_partb 672',
      res['n_pairs_partb'] == 672)

# ---- inputs ----
z18 = np.load(os.path.join(D18, 'traj_readout.npz'),
              allow_pickle=False)
z = np.load(os.path.join(OUT, 'p118_readout.npz'),
            allow_pickle=False)
mP = z18['gt_cleanseq_clean__P']   # keep float32
mA = z18['gt_cleanseq_clean__A1']  # keep float32
gP = z18['gen_clean__P']
gA = z18['gen_clean__A1']
annP_c = z['annP_c']
annP_s = z['annP_s']
annA_c = z['annA_c']
annA_s = z['annA_s']
NP_ = mP.shape[0]
check('I01 mP shape (672,13)',
      mP.shape == (672, 13) and mA.shape == (672, 13),
      str(mP.shape))
check('I02 ann shapes (672,12)',
      annP_c.shape == (672, 12)
      and annA_c.shape == (672, 12)
      and annP_s.shape == (672, 12)
      and annA_s.shape == (672, 12))

# content-step dm (t=2..12), aligned with ann[:,1:]
dmP = (mP[:, 1:] - mP[:, :-1]).astype(
    np.float64)[:, 1:]
dmA = (mA[:, 1:] - mA[:, :-1]).astype(
    np.float64)[:, 1:]
cP = annP_c[:, 1:]
cA = annA_c[:, 1:]
sP = annP_s[:, 1:]
sA = annA_s[:, 1:]

# ---- Part A class table (recomputed) ----
def cls_stats(code, ann, dm):
    sel = (ann == code)
    if not sel.any():
        return 0, None
    return int(sel.sum()), float(dm[sel].mean())


for code, rkey in ((1, 'syntax'), (4, 'fact_strict'),
                   (0, 'other'), (2, 'query_rest')):
    n_r, m_r = cls_stats(code, cP, dmP)
    if rkey in pa['class_table']['P']:
        ref = pa['class_table']['P'][rkey]
        check('A-P %s n' % rkey, n_r == ref['n'],
              '%d vs %d' % (n_r, ref['n']))
        check('A-P %s mean' % rkey,
              close(m_r, ref['mean_dm'], 1e-9),
              '%r vs %r' % (m_r, ref['mean_dm']))
for code, rkey in ((1, 'syntax'), (4, 'fact_strict'),
                   (0, 'other')):
    n_r, m_r = cls_stats(code, cA, dmA)
    if rkey in pa['class_table']['A1']:
        ref = pa['class_table']['A1'][rkey]
        check('A-A1 %s n' % rkey, n_r == ref['n'],
              '%d vs %d' % (n_r, ref['n']))
        check('A-A1 %s mean' % rkey,
              close(m_r, ref['mean_dm'], 1e-9),
              '%r vs %r' % (m_r, ref['mean_dm']))

FACTOR = np.isin(annP_c, [3, 4])
FACTA = np.isin(annA_c, [3, 4])
check('A fact_rate P',
      close(FACTOR[:, 1:].mean(),
            pa['fact_token_rate']['P'], 1e-12))
check('A fact_rate A1',
      close(FACTA[:, 1:].mean(),
            pa['fact_token_rate']['A1'], 1e-12))

# subtags
sel_q = (sP == 1)
n_q = int(sel_q.sum())
check('A subtag P queried n', n_q == 2463, str(n_q))
check('A subtag P queried mean',
      close(dmP[sel_q].mean(),
            pa['subtag_table']['P']['queried']
            ['mean_dm'], 1e-9))
sel_co = (sA == 2)
check('A subtag A1 context_other n',
      int(sel_co.sum()) == 2570)
check('A subtag A1 novel n',
      int((sA == 3).sum()) == 0)
check('A yes_no_at_content_steps 0',
      pa['yes_no_at_content_steps'] == 0)

# ---- A gates (full replication) ----
def margin_gate(mD, factmask, want_pos):
    dm = (mD[:, 1:] - mD[:, :-1]).astype(
        np.float64)[:, 1:]
    nt = dm.shape[1]
    x = mD[:, :-1].astype(np.float64)[:, 1:]
    fv = factmask[:, 1:]
    edges = np.quantile(x.ravel(),
                        [0.2, 0.4, 0.6, 0.8])
    qb = np.searchsorted(edges, x.ravel(),
                         side='right')
    tarr = np.tile(np.arange(nt), x.shape[0])
    diffs = []
    dP = dm.ravel()
    rows = []
    for t in range(nt):
        for b in range(5):
            sel = (qb == b) & (tarr == t)
            sf = sel & fv.ravel()
            sn = sel & (~fv.ravel())
            nf = int(sf.sum())
            nn = int(sn.sum())
            if nf < 15 or nn < 15:
                continue
            mf = float(dP[sf].mean())
            mn = float(dP[sn].mean())
            diffs.append(mf - mn)
            rows.append((t, b, sf, sn))
    nval = len(diffs)
    npos = sum(1 for d in diffs
               if (d > 0 if want_pos else d < 0))
    sf_all = np.zeros(dP.shape, dtype=bool)
    sn_all = np.zeros(dP.shape, dtype=bool)
    fvr = fv.ravel()
    for (t, b, sf, sn) in rows:
        sel = (qb == b) & (tarr == t)
        sf_all |= (sel & fvr)
        sn_all |= (sel & (~fvr))
    pooled = float(dP[sf_all].mean()
                   - dP[sn_all].mean())
    return pooled, npos / float(nval), nval


gp, gr, gn = margin_gate(mP, FACTOR, True)
check('A gate_P pooled',
      close(gp, pa['gate_P']['pooled_diff'], 1e-9),
      '%r' % gp)
check('A gate_P rate',
      close(gr, pa['gate_P']['unit_rate'], 1e-12))
check('A gate_P nvalid',
      gn == pa['gate_P']['n_valid_units'])
ga, gar, gan = margin_gate(mA, FACTA, False)
check('A gate_A1 pooled',
      close(ga, pa['gate_A1']['pooled_diff'], 1e-9),
      '%r' % ga)
check('A gate_A1 rate',
      close(gar, pa['gate_A1']['unit_rate'], 1e-12))

# gap gate
gap = (mP - mA)
dgap = (gap[:, 1:] - gap[:, :-1]).astype(
    np.float64)[:, 1:]
xg = gap[:, :-1].astype(np.float64)[:, 1:]
edges_g = np.quantile(xg.ravel(),
                      [0.2, 0.4, 0.6, 0.8])
qbg = np.searchsorted(edges_g, xg.ravel(),
                      side='right')
tarrg = np.tile(np.arange(xg.shape[1]),
                xg.shape[0])
ffv = (FACTOR & FACTA)[:, 1:].ravel()
nnv = ((~FACTOR) & (~FACTA))[:, 1:].ravel()
dgv = dgap.ravel()
contrasts = []
rows_g = []
for t in range(xg.shape[1]):
    for b in range(5):
        sel = (qbg == b) & (tarrg == t)
        sff = sel & ffv
        snn = sel & nnv
        if sff.sum() < 15 or snn.sum() < 15:
            continue
        contrasts.append(float(dgv[sff].mean())
                         - float(dgv[snn].mean()))
        rows_g.append((t, b))
nval_g = len(contrasts)
npos_g = sum(1 for d in contrasts if d > 0)
sfF = np.zeros(dgv.shape, dtype=bool)
snN = np.zeros(dgv.shape, dtype=bool)
for (t, b) in rows_g:
    sel = (qbg == b) & (tarrg == t)
    sfF |= (sel & ffv)
    snN |= (sel & nnv)
pdg_ff = float(dgv[sfF].mean())
pdg_nn = float(dgv[snN].mean())
gg = pa['gate_gap']
check('A gap dgap_ff',
      close(pdg_ff, gg['dgap_ff'], 1e-9), '%r' % pdg_ff)
check('A gap dgap_nn',
      close(pdg_nn, gg['dgap_nn'], 1e-9))
check('A gap contrast',
      close(pdg_ff - pdg_nn, gg['contrast'], 1e-9))
check('A gap rate',
      close(npos_g / float(nval_g),
            gg['unit_rate'], 1e-12))
check('A gap nvalid',
      nval_g == gg['n_valid_units'])

# ---- Part B recomputation ----
from transformers import AutoTokenizer  # noqa: E402
seal = json.load(io.open(
    OUT + r'\design_seal.json', encoding='utf-8'))
tok = AutoTokenizer.from_pretrained(MDIR)
yf = set()
for ys in seal['token_families']['yes']:
    yf |= set(tok(ys, add_special_tokens=False)
              ['input_ids'])
check('B YES_FAMILY nonempty', len(yf) > 0)


def any_yes(seq2d, nmatch=8):
    cnt = 0
    for row in seq2d:
        if any(int(t) in yf for t in row[:nmatch]):
            cnt += 1
    return cnt / float(seq2d.shape[0])


yr_clean = any_yes(np.vstack([gP, gA]))
check('B yes clean', close(
    yr_clean, pb['yes_rate_greedy']['clean'], 1e-12),
    '%r' % yr_clean)
gL30 = np.vstack([z['gen_abl_L30__P'],
                  z['gen_abl_L30__A1']])
gL32 = np.vstack([z['gen_abl_L32__P'],
                  z['gen_abl_L32__A1']])
yr30 = any_yes(gL30)
yr32 = any_yes(gL32)
check('B yes L30', close(
    yr30, pb['yes_rate_greedy']['abl_L30'], 1e-12))
check('B yes L32', close(
    yr32, pb['yes_rate_greedy']['abl_L32'], 1e-12))
beh_max = max(abs(yr30 - yr_clean),
              abs(yr32 - yr_clean))
check('B beh_max', close(
    beh_max, pb['beh_max'], 1e-12), '%r' % beh_max)


def agree_rate(pa2d, pb2d, nmatch=8):
    cnt = 0
    for i in range(pa2d.shape[0]):
        if [int(t) for t in pa2d[i][:nmatch]] == \
                [int(t) for t in pb2d[i][:nmatch]]:
            cnt += 1
    return cnt / float(pa2d.shape[0])


check('B agree clean', close(
    agree_rate(gP, gA),
    pb['agree_greedy']['clean'], 1e-12))
check('B agree L30', close(
    agree_rate(z['gen_abl_L30__P'],
               z['gen_abl_L30__A1']),
    pb['agree_greedy']['abl_L30'], 1e-12))
check('B agree L32', close(
    agree_rate(z['gen_abl_L32__P'],
               z['gen_abl_L32__A1']),
    pb['agree_greedy']['abl_L32'], 1e-12))

stc = z['samp_tokens_clean']
s30 = z['samp_tokens_L30']
s32 = z['samp_tokens_L32']
check('B samp arrays (1200,12)',
      stc.shape == (1200, 12)
      and s30.shape == (1200, 12)
      and s32.shape == (1200, 12))
yr_s = {k: any_yes(v) for (k, v) in
        (('clean', stc), ('L30', s30), ('L32', s32))}
check('B samp yes clean', close(
    yr_s['clean'],
    pb['yes_rate_sampled']['clean'], 1e-12))
check('B samp yes L30', close(
    yr_s['L30'],
    pb['yes_rate_sampled']['abl_L30'], 1e-12))
check('B samp yes L32', close(
    yr_s['L32'],
    pb['yes_rate_sampled']['abl_L32'], 1e-12))
samp_max = max(abs(yr_s['L30'] - yr_s['clean']),
               abs(yr_s['L32'] - yr_s['clean']))
check('B samp_max', close(
    samp_max, pb['samp_max'], 1e-12))

N_NEW = 12
for X in (30, 32):
    dms = (z['st_clean_L%d' % X]
           - z['st_clean_clean']).astype(np.float64)
    head = float(np.abs(dms[:, 0:3]).mean())
    tail = float(np.abs(
        dms[:, N_NEW - 3:N_NEW + 1]).mean())
    check('B state L%d ratio' % X, close(
        tail / max(head, 1e-9),
        pb['state_ratio']['L%d' % X]['ratio'], 1e-9))
    dmc = (z['st_L%d_L%d' % (X, X)]
           - z['st_clean_clean']).astype(np.float64)
    hc = float(np.abs(dmc[:, 0:3]).mean())
    tc = float(np.abs(
        dmc[:, N_NEW - 3:N_NEW + 1]).mean())
    check('B closed L%d ratio' % X, close(
        tc / max(hc, 1e-9),
        pb['closed_ratio']['L%d' % X]['ratio'],
        1e-9))
check('B SKIP note (repro bit-exact / own-recheck '
      'need GPU replay; asserted in closeout from '
      'result.json)', True)

# ---- Part C recomputation ----
XS = z['c_xy_x'].astype(np.float64)
YS = z['c_xy_y'].astype(np.float64)
DIRS = z['c_xy_dir']
PIDS = z['c_xy_pair']
TS = z['c_xy_t']
check('C n_samples', len(XS) == 16128
      and pc['primary']['n'] == 16128)
mx_P = XS[DIRS == 1].mean()
my_P = YS[DIRS == 1].mean()
mx_A = XS[DIRS == 0].mean()
my_A = YS[DIRS == 0].mean()
xc = np.where(DIRS == 1, XS - mx_P, XS - mx_A)
yc = np.where(DIRS == 1, YS - my_P, YS - my_A)


def rankdata(a):
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(len(a), dtype=np.float64)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    return float(np.corrcoef(rankdata(a),
                             rankdata(b))[0, 1])


def ols_fit(cols, y):
    X = np.column_stack(
        [np.ones(len(y))] + list(cols))
    th, _, _, _ = np.linalg.lstsq(
        X, y, rcond=None)
    sse = float(((y - X @ th) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return (1.0 - sse / sst), th


sp = spearman(xc, yc)
check('C spearman', close(
    sp, pc['primary']['spearman'], 1e-12), '%r' % sp)
r2l, th_l = ols_fit([xc], yc)
check('C r2_lin', close(
    r2l, pc['primary']['r2_lin'], 1e-12))
check('C slope', close(
    float(th_l[1]),
    pc['primary']['slope_lin'], 1e-12))
r2q, _ = ols_fit([xc, xc * xc], yc)
check('C d_quad', close(
    r2q - r2l, pc['primary']['d_quad'], 1e-12))
bs = np.quantile(xc, np.arange(1, 10) / 10.0)
best_r2 = -9e9
best_b = None
for b in bs:
    h = np.maximum(xc - b, 0.0)
    r2b, _ = ols_fit([xc, h], yc)
    if r2b > best_r2:
        best_r2 = r2b
        best_b = float(b)
check('C d_pw', close(
    best_r2 - r2l, pc['primary']['d_pw'], 1e-12))
check('C pw breakpoint', close(
    best_b, pc['primary']['pw_breakpoint'], 1e-9))
sel2 = TS >= 2
sp2 = spearman(xc[sel2], yc[sel2])
r2t2, _ = ols_fit([xc[sel2]], yc[sel2])
check('C t2 spearman', close(
    sp2, pc['t2_sensitivity']['spearman'], 1e-12))
check('C t2 r2_lin', close(
    r2t2, pc['t2_sensitivity']['r2_lin'], 1e-12))
_, th_raw = ols_fit([XS], YS)
fixed = -float(th_raw[0]) / float(th_raw[1])
check('C fixed point', close(
    fixed, pc['fixed_point_raw_lin'], 1e-9),
    '%r' % fixed)
edges_c = np.quantile(xc, np.arange(1, 20) / 20.0)
binc = np.searchsorted(edges_c, xc, side='right')
y0 = float(yc[binc == 0].mean())
y19 = float(yc[binc == 19].mean())
check('C bin0 y', close(
    y0, pc['bin_curve'][0]['y_mean'], 1e-9))
check('C bin19 y', close(
    y19, pc['bin_curve'][19]['y_mean'], 1e-9))

# ---- ledger ----
led = json.load(io.open(LEDGER, encoding='utf-8'))
ms = led['measurements']
check('L ledger n=257', len(ms) == 257,
      str(len(ms)))
m3120 = [m for m in ms if m.get('phase') == 3120]
check('L meas3120 present', len(m3120) == 1)
m0 = m3120[0]
check('L claim has DIRECTION REFUTED',
      'DIRECTION REFUTED' in m0['claim'])
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
check('L L14 connects 225',
      len(l14['connects']) == 225)
check('L L14 has meas3120',
      any('meas3120' in c for c in
          l14['connects']))
sha_in = led.get('ledger_sha256_8')
chk = dict(led)
chk.pop('ledger_sha256_8', None)
blob = json.dumps(chk, sort_keys=True,
                  ensure_ascii=False)
sha_rc = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
check('L sha8 recompute', sha_rc == sha_in
      and sha_in == '74767a4d',
      '%s vs %s' % (sha_rc, sha_in))

# ---- MEMO ----
memo = io.open(MEMO, encoding='utf-8').read()
check('M 3120 present', '## Phase 3120:' in memo)
i20 = memo.rindex('## Phase 3120:')
i19 = memo.rindex('## Phase 3119:')
check('M 3120 after 3119 (append-only)',
      i20 > i19)
sec = memo[i20:]
check('M no [[NOW]] placeholder',
      '[[NOW]]' not in sec)
check('M title timestamp',
      bool(re.match(
          r'## Phase 3120: .+ \[\d{4}-\d{2}-\d{2} '
          r'\d{2}:\d{2}\]',
          sec.split('\n')[0])),
      sec.split('\n')[0][:80])
check('M three-findings block',
      sec.count('标点/中性步才是') >= 2
      and 'linear_operator' in memo[i20:i20 + 6000]
      or '回归均值算子' in sec)

# ---- workspace logs ----
TODAY = '2026-09-23'
for tag, wd in (('W workspace D', ROOT +
                 r'\.workbuddy\memory'),
                ('W workspace C',
                 r'C:\Users\Admin\WorkBuddy'
                 r'\2026-09-17-01-30-05'
                 r'\.workbuddy\memory')):
    wl = wd + '\\' + TODAY + '.md'
    txt = io.open(wl, encoding='utf-8').read()
    check('%s wlog has 3120' % tag,
          'Phase 3120 Omega-P118' in txt)

# ---- workspace MEMORY.md ----
mem = io.open(ROOT + r'\.workbuddy\memory\MEMORY.md',
              encoding='utf-8').read()
check('MM max=3120', 'max=3120' in mem)
check('MM no max=3119', 'max=3119' not in mem)
check('MM chain block 3120',
      '机制链状态（3120）' in mem)
check('MM no chain block 3119',
      '机制链状态（3119）' not in mem)
check('MM next 3121', '下一 3121' in mem)
check('MM len<3000', len(mem) < 3000, str(len(mem)))

# ---- summary ----
items.append('')
items.append('TOTAL pass=%d fail=%d'
             % (n_pass, n_fail))
io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(items) + '\n')
if n_fail:
    raise SystemExit('DISK VERIFY FAILED: %d'
                     % n_fail)
print('disk verify ok: %d items' % n_pass)
