# -*- coding: utf-8 -*-
"""Phase 3129 disk verify: independent
re-derivation from real-disk artifacts.
19 partitions A1..A4 B1-B3 C1-C3 D1-D3
E1-E2 F1-F3 G1."""
import hashlib
import io
import json
import os
import re

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUTD = (RDIR + r'\phase3129'
        r'\omega_p127_dose_sweep_gen_'
        r'decouple_s0full_symbolfield')
D25 = (RDIR + r'\phase3125'
       r'\omega_p123_third_comp_qwen_inputstream')
D26 = (RDIR + r'\phase3126'
       r'\omega_p124_glm4_anchoredlast_'
       r'regen_writechain')
D27 = (RDIR + r'\phase3127'
       r'\omega_p125_writechain_port_'
       r'crossmodel_a1closure_fullregen')
D28 = (RDIR + r'\phase3128'
       r'\omega_p126_joint_swap_coord_'
       r'inject_interaction_s0match')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG_D = (ROOT + r'\.workbuddy\memory'
          r'\2026-09-25.md')
WLOG_C = (r'C:\Users\Admin\WorkBuddy'
          r'\2026-09-17-01-30-05'
          r'\.workbuddy\memory\2026-09-25.md')
MEMW = (ROOT + r'\.workbuddy\memory'
        r'\MEMORY.md')

NP = 672
NLQ = 36
NLG = 40
LAY_Q27 = {'write': [26, 28, 30, 32, 34],
           'port': [20, 21],
           'ctrl': [2, 8, 14]}
LAY_G27 = {'write': [8, 9, 13, 29],
           'port': [20],
           'ctrl': [4, 14, 34]}
DIRS = ('P', 'A1')
MID_MIN = 0.05
MID_RATIO = 0.5
A1_MIN = 0.05
GEN_CHANGE_GATE = 0.05
V_GATE = 0.30
V_SEP = 0.15
DOSE_TOL = 0.01
SHA28_LEDGER = '1eed79d9'
EXP_V = ('a_3128_ok|qwen_dose_monotonic'
         '|qwen_midprop_no'
         '|glm4_coord_a1_ineffective'
         '|gen_coupled|s0_fully_deterministic'
         '|interaction_symbolic_not_in_field'
         '|coverage_full')
V28 = ('repl_3127_ok|interaction_not_in_field'
       '|qwen_joint_independent'
       '|qwen_coord_effective'
       '|qwen_coord_dose_monotonic'
       '|glm4_joint_independent'
       '|glm4_coord_effective'
       '|coord_cross_descriptive'
       '|s0_probe_matched|coverage_full')

rep = []
nfail = [0]


def part(name, ok, detail):
    tag = 'PASS' if ok else 'FAIL'
    if not ok:
        nfail[0] += 1
    rep.append('%s %s %s' % (tag, name,
                             detail))


r = json.load(io.open(
    OUTD + r'\result.json', encoding='utf-8'))
z26 = np.load(D26 + r'\p124_readout.npz',
              allow_pickle=False)
z27 = np.load(D27 + r'\p125_readout.npz',
              allow_pickle=False)
z25 = np.load(D25 + r'\p123_readout.npz',
              allow_pickle=False)
res28 = json.load(io.open(
    D28 + r'\result.json', encoding='utf-8'))
zn = np.load(OUTD + r'\p127_readout.npz',
             allow_pickle=False)

# ============ A. result meta ============
ok = (r['phase'] == 3129
      and r['smoke'] is False
      and r['name'] == ('omega_p127_dose_sweep_'
                        'gen_decouple_'
                        's0full_symbolfield')
      and r['verdict'] == EXP_V)
part('A1_result_meta', ok,
     'phase=%s smoke=%s verdict_ok=%s'
     % (r['phase'], r['smoke'],
        r['verdict'] == EXP_V))

ok = (r['runtime_s'] > 10000
      and r['part_d']['np_d'] == NP)
part('A2_runtime_npd', ok,
     'runtime=%.1fs np_d=%s'
     % (r['runtime_s'],
        r['part_d']['np_d']))

exp_shapes = {
    'coords_g_top50': ((50,), 'int64'),
    'm_swap_P': ((NP,), 'float32'),
    'margin_flip_P': ((NP,), 'int64'),
    'gen_same_swap': ((NP,), 'int64'),
    'flip_lab_s1_P': ((NP,), 'int64'),
    'flip_lab_s1_A1': ((NP,), 'int64'),
    'flip_lab_s3_P': ((NP,), 'int64'),
    'flip_lab_s3_A1': ((NP,), 'int64'),
    'phi_s1_P': ((NLG + 1,), 'float64'),
    'phi_s1_A1': ((NLG + 1,), 'float64'),
    'phi_s3_P': ((NLG + 1,), 'float64'),
    'phi_s3_A1': ((NLG + 1,), 'float64')}
bad = []
for k, (sh, dt) in exp_shapes.items():
    if k not in zn.files:
        bad.append(k + ':missing')
        continue
    if zn[k].shape != sh:
        bad.append('%s:shape%s' % (k, zn[k].shape))
    if str(zn[k].dtype) != dt:
        bad.append('%s:dtype%s'
                   % (k, zn[k].dtype))
part('A3_npz_keys', len(bad) == 0,
     '12 keys %s' % ('ok' if not bad
                     else str(bad)))

pa = r['part_a']
pa28 = res28['part_a']
ok = (abs(pa['gates_q']['write']
          - pa28['repl_gates_q']['write'])
      < 1e-12
      and abs(pa['gates_q']['port']
              - pa28['repl_gates_q']['port'])
      < 1e-12
      and abs(pa['gates_g']['write']
              - pa28['repl_gates_g']['write'])
      < 1e-12
      and abs(pa['gates_g']['port']
              - pa28['repl_gates_g']['port'])
      < 1e-12)
part('A4_parta_crossfile', ok,
     'gates q %.6f/%.6f g %.6f/%.6f vs '
     '3128 repl'
     % (pa['gates_q']['write'],
        pa['gates_q']['port'],
        pa['gates_g']['write'],
        pa['gates_g']['port']))


# ============ B. gates + dose ============
def f32_dm(zf, zbase, side, dc, l, NL):
    return (zf['%s_%s_L%02d' % (side, dc, l)]
            [:, -1]
            - zbase['mlg_s0_%s' % dc][:NP, NL,
                                      -1]
            ).astype(np.float64)


def gates27(zf, zbase, LAY, NL, side):
    dmf = {}
    for grp in ('write', 'port', 'ctrl'):
        for l in LAY[grp]:
            for dc in DIRS:
                dmf['%s_L%02d' % (dc, l)] = \
                    f32_dm(zf, zbase, side,
                           dc, l, NL)
    mc = float(np.median(np.abs(
        np.concatenate(
            [dmf['P_L%02d' % l]
             for l in LAY['ctrl']]
            + [dmf['A1_L%02d' % l]
               for l in LAY['ctrl']]))))
    g = {}
    for grp in ('write', 'port'):
        meds = [float(np.median(np.abs(
            dmf['%s_L%02d' % (dc, l)])))
            for l in LAY[grp] for dc in DIRS]
        g[grp] = float(np.median(meds)) / max(
            mc, 1e-12)
    return g, mc


gq_r, mcq_r = gates27(z27, z25, LAY_Q27, NLQ,
                      'dmq')
gg_r, mcg_r = gates27(z27, z26, LAY_G27, NLG,
                      'dmg')
dq = [abs(gq_r['write'] - pa['gates_q']['write']),
      abs(gq_r['port'] - pa['gates_q']['port']),
      abs(gg_r['write'] - pa['gates_g']['write']),
      abs(gg_r['port'] - pa['gates_g']['port']),
      abs(mcq_r - pa['mcq']),
      abs(mcg_r - pa['mcg'])]
part('B1_gates_f32', max(dq) < 1e-12,
     'q %.6f/%.6f g %.6f/%.6f mc %.6f/%.6f'
     ' maxd=%.2e'
     % (gq_r['write'], gq_r['port'],
        gg_r['write'], gg_r['port'],
        mcq_r, mcg_r, max(dq)))

pb = r['part_b']
dose = pb['dose_q']
fracs = ['f05', 'f10', 'f20']
exp_flip = {'f05': 0.08035714285714285,
            'f10': 0.18229166666666666,
            'f20': 0.28422619047619047}
exp_med = {'f05': 0.5132111757993698,
           'f10': 1.0975285470485687,
           'f20': 2.367865651845932}
bad = []
for f in fracs:
    d = dose[f]
    if abs(d['flip'] - exp_flip[f]) >= 1e-12:
        bad.append(f + ':flip')
    if abs(d['med_dm'] - exp_med[f]) >= 1e-12:
        bad.append(f + ':med_dm')
    if abs(d['flip'] - (d['flip_p']
                        + d['flip_m']) / 2.0) \
            >= 1e-12:
        bad.append(f + ':pm_mean')
fr = [dose[f]['flip'] for f in fracs]
mono = all(fr[i + 1] >= fr[i] - DOSE_TOL
           for i in range(2)) and fr[-1] > fr[0]
if not mono:
    bad.append('mono')
part('B2a_dose_values', len(bad) == 0,
     'flip %.6f/%.6f/%.6f mono=%s %s'
     % (fr[0], fr[1], fr[2], mono,
        'ok' if not bad else str(bad)))

cq28 = res28['part_b']['coord_q']
ref_f = cq28['top50']['flip']
ref_dm = cq28['top50']['med_dm']
ok = (abs(dose['f10']['flip'] - ref_f)
      < 1e-9
      and abs(dose['f10']['med_dm'] - ref_dm)
      < 1e-9
      and pb['repro_max_abs'] == 0.0
      and pb['last_flip_ref']
      == dose['f10']['flip'])
part('B2b_dose_3128_repro', ok,
     'f10 %.12f vs ref %.12f d %.10f vs '
     'ref %.10f repro=%s'
     % (dose['f10']['flip'], ref_f,
        dose['f10']['med_dm'], ref_dm,
        pb['repro_max_abs']))

pc = r['part_c']
last_flip = dose['f10']['flip']
mid_ok = (pb['mid_flip'] >= MID_MIN
          and pb['mid_flip']
          >= MID_RATIO * max(last_flip, 1e-9))
a1_ok = pc['a1_flip'] >= A1_MIN
exp_abs = (abs(pb['mid_flip']
               - 0.00818452380952381) < 1e-12
           and abs(pb['mid_dm']
                   - 0.05085253715515137)
           < 1e-12
           and abs(pc['a1_flip']
                   - 0.011904761904761904)
           < 1e-12
           and abs(pc['a1_dm']
                   - 0.34146997332572937)
           < 1e-12)
ok = (exp_abs and not mid_ok and not a1_ok
      and 'qwen_midprop_no' in r['verdict']
      and 'glm4_coord_a1_ineffective'
      in r['verdict'])
part('B3_mid_a1_gates', ok,
     'mid %.6f (gate %.3f) a1 %.6f '
     '(gate %.3f) mid_ok=%s a1_ok=%s'
     % (pb['mid_flip'], MID_MIN,
        pc['a1_flip'], A1_MIN,
        mid_ok, a1_ok))

# ============ C. gen decouple + s0 ============
mfl = zn['margin_flip_P'].astype(bool)
gsame = zn['gen_same_swap'].astype(bool)
mf_n = int(mfl.sum())
chg = 1.0 - float(gsame.mean())
sgf = (float(gsame[mfl].mean())
       if mf_n > 0 else -1.0)
so = float(gsame.mean())
dec_ok = (chg < GEN_CHANGE_GATE
          and mf_n > 0
          and abs(sgf - so)
          < GEN_CHANGE_GATE)
ok = (mf_n == 172
      and abs(pc['margin_flip_n'] - mf_n) == 0
      and abs(chg - pc['gen_chg_rate']) < 1e-12
      and abs(sgf - pc['same_given_flip'])
      < 1e-12
      and abs(so - pc['same_overall'])
      < 1e-12
      and dec_ok == ('gen_decoupled'
                     in r['verdict']))
part('C1_gen_recalc', ok,
     'mf_n=%d chg=%.4f sgf=%.4f so=%.4f '
     'dec_ok=%s' % (mf_n, chg, sgf, so,
                    dec_ok))

ok = (pc['gen_first_chg'] == 178
      and pc['s0_mism'] == 0
      and pc['s0_token_agree'] == 1.0
      and z26['gen_base_P'].shape == (NP, 12)
      and z26['gen_base_A1'].shape == (NP, 12)
      and z26['mlg_s0_P'].shape
      == (NP, NLG + 1, 13))
part('C2_first_s0_shapes', ok,
     'first=%s s0_mism=%s agree=%s z26 %s'
     % (pc['gen_first_chg'], pc['s0_mism'],
        pc['s0_token_agree'],
        z26['mlg_s0_P'].shape))

vd = r['verdict'].split('|')
ok = (vd[4] == 'gen_coupled'
      and vd[5] == 's0_fully_deterministic'
      and vd[7] == 'coverage_full')
part('C3_verdict_segments', ok,
     '|'.join(vd[4:6]))

# ============ D. flip lab + phi ============
pd_ = r['part_d']
fexp = {'s1_P': 'flip_lab_s1_P',
        's1_A1': 'flip_lab_s1_A1',
        's3_P': 'flip_lab_s3_P',
        's3_A1': 'flip_lab_s3_A1'}
bad = []
for k, nk in fexp.items():
    mv = float(zn[nk].mean())
    if abs(mv - pd_['flip_rates'][k]) >= 1e-12:
        bad.append('%s:%.6f vs %.6f'
                   % (k, mv,
                      pd_['flip_rates'][k]))
ok = (len(bad) == 0
      and abs(pd_['flip_rates']['s2_P']
              - 0.8273809523809523) < 1e-12
      and abs(pd_['flip_rates']['s2_A1']
              - 0.34970238095238093) < 1e-12)
part('D1_fliplab_rates', ok,
     's1_P %.6f s1_A1 %.6f s3_P %.6f '
     's3_A1 %.6f %s'
     % (pd_['flip_rates']['s1_P'],
        pd_['flip_rates']['s1_A1'],
        pd_['flip_rates']['s3_P'],
        pd_['flip_rates']['s3_A1'],
        'ok' if not bad else str(bad)))

bad = []
for k in ('s1_P', 's1_A1', 's3_P',
          's3_A1'):
    a = np.asarray(pd_['phi_prof'][k])
    b = zn['phi_' + k]
    if a.shape != b.shape:
        bad.append(k + ':shape')
        continue
    md = float(np.max(np.abs(a - b)))
    if md >= 1e-12:
        bad.append('%s:maxd%.2e' % (k, md))
part('D2_phi_prof_consist', len(bad) == 0,
     '4x41 %s' % ('ok' if not bad
                  else str(bad)))

pp = np.asarray(pd_['phi_prof']['s1_P'])
pq = np.asarray(pd_['phi_prof']['s3_A1'])
sep_prof = np.abs(pp - pq)
pk = int(np.argmax(sep_prof))
sep_v = float(sep_prof[pk])
r_v = float(max(abs(pp[pk]), abs(pq[pk])))
sym_found = (r_v >= V_GATE
             and sep_v >= V_SEP)
ok = (pk == pd_['peak_layer'] == 40
      and abs(sep_v - pd_['peak_sep'])
      < 1e-12
      and abs(r_v - pd_['peak_r']) < 1e-12
      and sym_found == ('interaction_symbolic'
                        '_found'
                        in r['verdict']))
part('D3_peak_recalc', ok,
     'peak L%d sep=%.6f r=%.6f found=%s '
     '(gates r>=%.2f sep>=%.2f)'
     % (pk, sep_v, r_v, sym_found,
        V_GATE, V_SEP))

# ============ E. ledger ============
raw = io.open(OUTD + r'\result.json',
              'rb').read()
sha8 = hashlib.sha256(raw).hexdigest()[:8]
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
stored_sha = led['ledger_sha256_8']
last = led['measurements'][-1]
led['ledger_sha256_8'] = SHA28_LEDGER
blob = json.dumps(led, sort_keys=True,
                  ensure_ascii=False)
sha_re = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
ok = (len(led['measurements']) == 266
      and last['meas_id'].startswith(
          'meas3129_omega_p127')
      and last['verdict'] == r['verdict']
      and last['hashes']['result_sha256_8']
      == sha8
      and sha_re == stored_sha)
part('E1_ledger', ok,
     'n=%d sha8=%s stored=%s recomp=%s'
     % (len(led['measurements']), sha8,
        stored_sha, sha_re))

ok = res28['verdict'] == V28 \
    and res28['smoke'] is False
part('E2_res28_link', ok,
     '3128 verdict+smoke ok=%s' % ok)

# ============ F. memo + wlog ============
memo = io.open(MEMO, encoding='utf-8').read()
mt = re.search(r'(?m)^## Phase 3129:.*$',
               memo)
t_ok = mt is not None and len(
    mt.group(0)) < 100
sec_ok = all(s in memo for s in (
    '### 1. 三大发现', '### 2. 关键数值',
    '### 3. 硬伤', '### 4. 机制拼图更新',
    '### 5. 3130 预注册'))
nums_ok = all(s in memo for s in (
    '0.18229166', '672/672', '172/672',
    'mism 0', 'L40 sep 0.2934', '0.0082',
    'first_chg 178'))
art_ok = ('phase3129/'
          'omega_p127_dose_sweep_gen_'
          'decouple_s0full_symbolfield/'
          in memo)
part('F1_memo', t_ok and sec_ok
     and nums_ok and art_ok,
     'title_len=%s secs=%s nums=%s art=%s'
     % (len(mt.group(0)) if mt else -1,
        sec_ok, nums_ok, art_ok))

wd = io.open(WLOG_D,
             encoding='utf-8').read()
ok = ('Phase 3129 Omega-P127 closeout'
      in wd and stored_sha in wd)
part('F2_wlog_workspace', ok,
     '2026-09-25.md sha8=%s' % stored_sha)

wc = io.open(WLOG_C,
             encoding='utf-8').read()
ok = ('Phase 3129 Omega-P127 closeout'
      in wc and stored_sha in wc)
part('F3_wlog_second', ok,
     'workbuddy copy sha8=%s' % stored_sha)

# ============ G. memory ============
mem = io.open(MEMW,
              encoding='utf-8').read()
ok = ('3129（T4）' in mem
      and 'max=3129' in mem
      and '下一 3130' in mem
      and len(mem) < 3000)
part('G1_memory', ok,
     '3129 line + max/next len=%d ok=%s'
     % (len(mem), ok))

rep.append('')
rep.append('TOTAL FAILS: %d' % nfail[0])
outp = (ROOT + r'\tests\gpt5_temp'
        r'\p3129_verify_out.txt')
with io.open(outp, 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(rep))
print('VERIFY_DONE fails=%d' % nfail[0])
