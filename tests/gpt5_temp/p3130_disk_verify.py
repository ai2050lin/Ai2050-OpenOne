# -*- coding: utf-8 -*-
"""Phase 3130 disk verify: independent
re-derivation from real-disk artifacts.
22 partitions A1-A4 B1-B3 C1-C3 D1-D3
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
OUTD = (RDIR + r'\phase3130'
        r'\omega_p128_positionspectrum_'
        r'a1fit_single_swap_dyn')
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
D29 = (RDIR + r'\phase3129'
       r'\omega_p127_dose_sweep_gen_'
       r'decouple_s0full_symbolfield')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory'
WLOG_C = (r'C:\Users\Admin\WorkBuddy'
          r'\2026-09-17-01-30-05'
          r'\.workbuddy\memory')
MEMW = (ROOT + r'\.workbuddy\memory'
        r'\MEMORY.md')

NP = 672
N_NEW = 12
F10_REF = 0.18229166666666666
MEDDM_REF = 1.0975285470485687
MID6_REF = 0.00818452380952381
A1FIT_REPLAY_REF = 0.011904761904761904
A1_MIN = 0.05
DYN_GATE = 0.5
SINGLE_LAYERS = [8, 9, 13, 29]
SHA29_LEDGER = '7c9e7b23'
MARK = 'Phase 3130 Omega-P128 closeout'

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
z29 = np.load(D29 + r'\p127_readout.npz',
              allow_pickle=False)
z26 = np.load(D26 + r'\p124_readout.npz',
              allow_pickle=False)
res28 = json.load(io.open(
    D28 + r'\result.json', encoding='utf-8'))
res29 = json.load(io.open(
    D29 + r'\result.json', encoding='utf-8'))
zn = np.load(OUTD + r'\p128_readout.npz',
             allow_pickle=False)
seal = json.load(io.open(
    OUTD + r'\design_seal.json',
    encoding='utf-8'))

pb = r['part_b']
pc = r['part_c']
vd = r['verdict'].split('|')
dom = pc['dominance']
fs = pc['first_share']

exp_vd = [
    'a_3129_ok', 'qwen_spec_nonlocal',
    'glm4_pfit_replay_exact',
    'glm4_a1fit_confirms_binding',
    'gen_swap_3129_bitexact',
    ('single_dominant' if dom >= 0.5
     else 'single_distributed'),
    ('dyn_first_concentrated' if fs >= 0.5
     else 'dyn_spread'),
    'coverage_full']

# ============ A. result meta ============
ok = (r['phase'] == 3130
      and r['smoke'] is False
      and r['name'] == ('omega_p128_'
                        'positionspectrum_'
                        'a1fit_single_'
                        'swap_dyn')
      and len(vd) == 8
      and vd == exp_vd)
part('A1_result_meta', ok,
     'phase=%s smoke=%s verdict_ok=%s'
     % (r['phase'], r['smoke'],
        vd == exp_vd))

seal_raw = io.open(
    OUTD + r'\design_seal.json',
    'rb').read()
seal_sha = hashlib.sha256(
    seal_raw).hexdigest()[:8]
ok = (r['runtime_s'] > 7000
      and r['seal_sha8'] == seal_sha)
part('A2_runtime_sealsha', ok,
     'runtime=%.1fs seal_sha8=%s==%s'
     % (r['runtime_s'], r['seal_sha8'],
        seal_sha))

exp_shapes = {
    'spec_ks': ((13,), 'int64'),
    'spec_flip': ((13,), 'float64'),
    'spec_med_dm': ((13,), 'float64'),
    'coords_g_a1_top50': ((50,), 'int64'),
    'rho_a1_top200': ((200,), 'float32'),
    'gen_same_full': ((NP,), 'int64'),
    'gen_tokdiff': ((N_NEW,), 'float64'),
    'gen_first_hist': ((N_NEW + 1,),
                       'int64'),
    'sel': ((256,), 'int64'),
    'single_chg': ((4,), 'float64'),
    'tok_diff_flip': ((N_NEW,), 'float64'),
    'tok_diff_rest': ((N_NEW,),
                      'float64')}
bad = []
for k, (sh, dt) in exp_shapes.items():
    if k not in zn.files:
        bad.append(k + ':missing')
        continue
    if zn[k].shape != sh:
        bad.append('%s:shape%s'
                   % (k, zn[k].shape))
    if str(zn[k].dtype) != dt:
        bad.append('%s:dtype%s'
                   % (k, zn[k].dtype))
part('A3_npz_keys', len(bad) == 0,
     '12 keys %s' % ('ok' if not bad
                     else str(bad)))

pa = r['part_a']
pa29 = res29['part_a']
ok = (abs(pa['gates_q']['write']
          - pa29['gates_q']['write'])
      < 1e-12
      and abs(pa['gates_q']['port']
              - pa29['gates_q']['port'])
      < 1e-12
      and abs(pa['gates_g']['write']
              - pa29['gates_g']['write'])
      < 1e-12
      and abs(pa['gates_g']['port']
              - pa29['gates_g']['port'])
      < 1e-12
      and abs(pa['mcq'] - pa29['mcq'])
      < 1e-12
      and abs(pa['mcg'] - pa29['mcg'])
      < 1e-12)
part('A4_parta_crossfile', ok,
     'gates q %.6f/%.6f g %.6f/%.6f vs '
     '3129 (3127 gate f32)'
     % (pa['gates_q']['write'],
        pa['gates_q']['port'],
        pa['gates_g']['write'],
        pa['gates_g']['port']))


# ============ B. position spectrum =====
sf_r = np.asarray(pb['spec_flip'],
                  dtype=np.float64)
sm_r = np.asarray(pb['spec_med_dm'],
                  dtype=np.float64)
ks_r = np.asarray(pb['spec_ks'],
                  dtype=np.int64)
ok = (np.array_equal(ks_r, zn['spec_ks'])
      and np.max(np.abs(sf_r
                        - zn['spec_flip']))
      < 1e-12
      and np.max(np.abs(sm_r
                        - zn['spec_med_dm']))
      < 1e-12
      and abs(sf_r[0] - F10_REF) < 1e-12
      and abs(sf_r[6] - MID6_REF) < 1e-12
      and abs(sf_r[12] - sf_r[0]) < 1e-12
      and abs(sm_r[0] - MEDDM_REF) < 1e-12)
part('B1_spec_values', ok,
     '13 pts npz==json; k0 %.12f k6 %.12f '
     'k12==k0 med0 %.12f'
     % (sf_r[0], sf_r[6], sm_r[0]))

cq28 = res28['part_b']['coord_q']
ok = (abs(sf_r[0]
          - cq28['top50']['flip']) < 1e-9
      and abs(sm_r[0]
              - cq28['top50']['med_dm'])
      < 1e-9
      and abs(sf_r[6]
              - res29['part_b']['mid_flip'])
      < 1e-9)
part('B2_spec_anchors_3128_3129', ok,
     'k0 %.10f vs 3128 %.10f; med %.8f '
     'vs %.8f; k6 %.10f vs 3129 mid '
     '%.10f'
     % (sf_r[0], cq28['top50']['flip'],
        sm_r[0], cq28['top50']['med_dm'],
        sf_r[6],
        res29['part_b']['mid_flip']))

cliff = float(np.max(sf_r[1:12]))
f0 = float(sf_r[0])
far_dead_recalc = all(
    sf_r[k] < 0.5 * f0
    for k in range(3, 13))
ok = (cliff < 0.035
      and cliff < 0.5 * f0
      and pb['k50'] == 12 and pb['k10'] == 12
      and pb['d50'] == 0 and pb['d10'] == 0
      and 10.694 < pb['delta_q'] < 10.695
      and far_dead_recalc is False
      and 'qwen_spec_nonlocal' in
      r['verdict'])
part('B3_cliff_profile', ok,
     'k1-11 max %.4f (<0.5*f0=%.4f, '
     'ratio %.2f); d50=d10=0; far_dead '
     'recalc False (k12 site-identity '
     'pollutes gate -> verdict word '
     'errata)' % (cliff, 0.5 * f0,
                  cliff / f0))


# ============ C. a1fit + replay ========
ok = (abs(pc['pfit_replay_flip']
          - A1FIT_REPLAY_REF) < 1e-12
      and abs(res29['part_c']['a1_flip']
              - pc['pfit_replay_flip'])
      < 1e-12)
part('C1_pfit_replay_crossphase', ok,
     'pfit_replay %.12f == A1FIT_REF == '
     '3129 a1_flip'
     % pc['pfit_replay_flip'])

inter = int(np.intersect1d(
    zn['coords_g_a1_top50'],
    z29['coords_g_top50']).size)
ok = (pc['coords_a1_overlap_P'] == 0
      and inter == 0)
a1_ok2 = (0.99 < pc['delta_g_a1'] < 1.02
          and abs((pc['a1fit_flip_s1']
                   + pc['a1fit_flip_s-1'])
                  / 2.0
                  - pc['a1fit_flip'])
          < 1e-12
          and 0.014 < pc['a1fit_flip'] < 0.016
          and pc['a1fit_ok'] is False
          and pc['a1fit_flip'] < A1_MIN
          and vd[3] ==
          'glm4_a1fit_confirms_binding')
part('C2_a1fit_coords', ok and a1_ok2,
     'overlap=%s npz_inter=%d delta_g_'
     'a1=%.4f a1fit %.4f (s1 %.4f/s-1 '
     '%.4f) ok=%s'
     % (pc['coords_a1_overlap_P'], inter,
        pc['delta_g_a1'], pc['a1fit_flip'],
        pc['a1fit_flip_s1'],
        pc['a1fit_flip_s-1'],
        pc['a1fit_ok']))

sel_sha = hashlib.sha256(
    zn['sel'].tobytes()).hexdigest()[:8]
ok = (sel_sha == pc['sel256_sha8']
      and len(np.unique(zn['sel'])) == 256
      and int(zn['sel'].min()) >= 0
      and int(zn['sel'].max()) < NP)
part('C3_sel256_sha8', ok,
     'sel256 sha8 %s frozen (PERM_SEED '
     '3130)' % sel_sha)


# ============ D. swap + single + dyn ===
gsame = zn['gen_same_full'].astype(bool)
chg = 1.0 - float(gsame.mean())
fh = np.asarray(pc['first_hist'],
                dtype=np.int64)
ok = (pc['full_swap_bitexact_3129']
      is True
      and pc['full_swap_chg'] == 1.0
      and abs(chg - pc['full_swap_chg'])
      < 1e-12
      and pc['full_swap_first'] == 178
      and int(fh.sum()) == NP
      and int(fh[0]) == 0
      and int(fh[1:].sum()) == NP
      and int(fh[1])
      == pc['full_swap_first'])
part('D1_full_swap_bitexact', ok,
     'chg=%.4f first=%d hist0=%d '
     'hist_sum=%d (vs 3129 bitexact)'
     % (chg, pc['full_swap_first'],
        fh[0], fh.sum()))

sc_arr = zn['single_chg']
bad = []
for i, l in enumerate(SINGLE_LAYERS):
    if abs(float(sc_arr[i])
           - pc['single_chg'][str(l)]
           ) >= 1e-12:
        bad.append('L%d' % l)
dom_re = (max(float(v) for v in sc_arr)
          / chg if chg > 1e-12 else 0.0)
ok = (len(bad) == 0
      and abs(dom_re - dom) < 1e-12
      and vd[5] == ('single_dominant'
                    if dom >= 0.5
                    else 'single_distributed')
      and all(0.0 <= float(v) <= 1.0
              for v in sc_arr))
part('D2_single_dominance', ok,
     'L8 %.3f L9 %.3f L13 %.3f L29 %.3f '
     'dom_re=%.4f==%.4f %s %s'
     % (sc_arr[0], sc_arr[1], sc_arr[2],
        sc_arr[3], dom_re, dom,
        'ok' if not bad else str(bad),
        vd[5]))

mfl = z29['margin_flip_P'].astype(bool)
mfl_n = int(mfl.sum())
tfl = zn['tok_diff_flip']
tfr = zn['tok_diff_rest']
tfa = zn['gen_tokdiff']
wmax = 0.0
for t in range(N_NEW):
    w = (mfl_n * float(tfl[t])
         + (NP - mfl_n) * float(tfr[t]))
    wmax = max(wmax, abs(w - NP
                         * float(tfa[t])))
fs_re = float(fh[1]) / float(
    max(1, int(fh[1:].sum())))
ok = (mfl_n == 172
      and float(tfl.min()) >= 0.0
      and float(tfr.min()) >= 0.0
      and wmax < 1e-6
      and abs(fs_re - fs) < 1e-12
      and vd[6] == ('dyn_first_'
                    'concentrated'
                    if fs >= 0.5
                    else 'dyn_spread'))
part('D3_dynamics', ok,
     'mfl_n=%d weighted==all maxd %.2e; '
     'first_share %.4f==%.4f %s'
     % (mfl_n, wmax, fs_re, fs, vd[6]))


# ============ E. ledger + seal =========
raw = io.open(OUTD + r'\result.json',
              'rb').read()
sha8 = hashlib.sha256(raw).hexdigest()[:8]
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
stored_sha = led['ledger_sha256_8']
last = led['measurements'][-1]
led['ledger_sha256_8'] = SHA29_LEDGER
blob = json.dumps(led, sort_keys=True,
                  ensure_ascii=False)
sha_re = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
ok = (len(led['measurements']) == 267
      and last['meas_id'].startswith(
          'meas3130_omega_p128')
      and last['verdict'] == r['verdict']
      and last['hashes']['result_sha256_8']
      == sha8
      and sha_re == stored_sha)
part('E1_ledger', ok,
     'n=%d result_sha8=%s ledger sha8 '
     'stored=%s recomp=%s'
     % (len(led['measurements']), sha8,
        stored_sha, sha_re))

c = seal['constants']
a = seal['anchors']
ok = (seal['phase'] == 3130
      and seal['smoke'] is False
      and c['NP'] == 672
      and c['N_NEW'] == 12
      and c['INJ_LAYER_Q'] == 24
      and c['K_DOSE'] == 50
      and c['SPEC_FRAC'] == 0.10
      and c['SPEC_KS'] == list(range(13))
      and c['INJ_LAYER_G'] == 20
      and c['HALF_FIT'] == 336
      and c['A1_MIN'] == 0.05
      and c['JOINT_G_write'] == [8, 9, 13,
                                 29]
      and c['SINGLE_LAYERS'] == [8, 9, 13,
                                 29]
      and c['SINGLE_NP'] == 256
      and c['GEN_CHANGE_GATE'] == 0.05
      and c['DYN_GATE'] == 0.5
      and c['PERM_SEED'] == 3130
      and a['f10_flip'] == F10_REF
      and a['f10_med_dm'] == MEDDM_REF
      and a['mid6_flip'] == MID6_REF
      and a['a1fit_pfit_replay']
      == A1FIT_REPLAY_REF)
part('E2_seal_constants', ok,
     'design_seal 15 constants + 4 '
     'anchors frozen pre-observation')


# ============ F. memo + wlog ==========
memo = io.open(MEMO, encoding='utf-8').read()
mt = re.search(r'(?m)^## Phase 3130:.*$',
               memo)
t_ok = mt is not None and len(
    mt.group(0)) < 110
sec_ok = all(s in memo for s in (
    '### 1. 三大发现', '### 2. 关键数值',
    '### 3. 硬伤', '### 4. 机制拼图更新',
    '### 5. errata', '### 6. 3131 预注册'))
nums_ok = all(s in memo for s in (
    '0.1823', 'd50=d10=0', '0/50',
    '0.011905', 'bitexact True',
    'cliff-local', 'qwen_spec_nonlocal',
    'first=178'))
art_ok = ('phase3130/'
          'omega_p128_positionspectrum'
          '_a1fit_single_swap_dyn/'
          in memo)
part('F1_memo', t_ok and sec_ok
     and nums_ok and art_ok,
     'title_len=%s secs=%s nums=%s '
     'art=%s'
     % (len(mt.group(0)) if mt else -1,
        sec_ok, nums_ok, art_ok))


def find_wlog(wdir):
    for fn in sorted(os.listdir(wdir)):
        if not fn.startswith('2026-'):
            continue
        if not fn.endswith('.md'):
            continue
        txt = io.open(
            os.path.join(wdir, fn),
            encoding='utf-8').read()
        if MARK in txt and stored_sha \
                in txt:
            return fn
    return None


fn_d = find_wlog(WLOG_D)
fn_c = find_wlog(WLOG_C)
part('F2_wlog_workspace', fn_d is not None,
     '%s sha8=%s' % (fn_d, stored_sha))
part('F3_wlog_second', fn_c is not None,
     '%s sha8=%s' % (fn_c, stored_sha))

# ============ G. memory ===============
mem = io.open(MEMW,
              encoding='utf-8').read()
ok = ('3130（T4）' in mem
      and 'max=3130' in mem
      and '下一 3131' in mem
      and 'cliff-local' in mem
      and len(mem) < 3000)
part('G1_memory', ok,
     '3130 line + max/next len=%d ok=%s'
     % (len(mem), ok))

rep.append('')
rep.append('TOTAL FAILS: %d' % nfail[0])
outp = (ROOT + r'\tests\gpt5_temp'
        r'\p3130_verify_out.txt')
with io.open(outp, 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(rep))
print('VERIFY_DONE fails=%d' % nfail[0])
