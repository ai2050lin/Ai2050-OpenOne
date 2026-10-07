# -*- coding: utf-8 -*-
"""Phase 3128 disk verify: independent
re-derivation from real-disk artifacts.
19 partitions A1..A4 B1-B2 C1-C2 D1-D3
E1-E2 F1-F2 G1-G4."""
import hashlib
import io
import json
import os
import re

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUTD = (RDIR + r'\phase3128'
        r'\omega_p126_joint_swap_coord_'
        r'inject_interaction_s0match')
D13 = (RDIR + r'\phase3113'
       r'\omega_p111_artifact_writein')
D25 = (RDIR + r'\phase3125'
       r'\omega_p123_third_comp_qwen_inputstream')
D26 = (RDIR + r'\phase3126'
       r'\omega_p124_glm4_anchoredlast_'
       r'regen_writechain')
D27 = (RDIR + r'\phase3127'
       r'\omega_p125_writechain_port_'
       r'crossmodel_a1closure_fullregen')
MDIR_G = (ROOT + r'\models\hf'
          r'\glm4-9b-chat-hf')
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
CB_LAYERS = [12, 20, 24, 28, 32]
INJ_LAYER_Q = 24
INJ_FRAC = 0.10
DIRS = ('P', 'A1')
V_GATE = 0.30
V_SEP = 0.15
EXP_V = ('repl_3127_ok|interaction_not_in_field'
         '|qwen_joint_independent'
         '|qwen_coord_effective'
         '|qwen_coord_dose_monotonic'
         '|glm4_joint_independent'
         '|glm4_coord_effective'
         '|coord_cross_descriptive'
         '|s0_probe_matched|coverage_full')
SHA27 = '1e07b8bd'

r = json.load(io.open(OUTD + r'\result.json',
                      encoding='utf-8'))
seal = json.load(io.open(
    OUTD + r'\design_seal.json',
    encoding='utf-8'))
res27 = json.load(io.open(
    D27 + r'\result.json', encoding='utf-8'))
z = np.load(OUTD + r'\p126_readout.npz',
            allow_pickle=False)
z25 = np.load(D25 + r'\p123_readout.npz',
              allow_pickle=False)
z26 = np.load(D26 + r'\p124_readout.npz',
              allow_pickle=False)
z27 = np.load(D27 + r'\p125_readout.npz',
              allow_pickle=False)
capb = np.load(D13 + r'\capture_b.npz',
               allow_pickle=False)
logtxt = io.open(OUTD + r'\run_log.txt',
                 encoding='utf-8').read()

rep = []
nfail = [0]


def part(pid, ok, detail):
    tag = 'PASS' if ok else 'FAIL'
    if not ok:
        nfail[0] += 1
    rep.append('%s %s %s'
               % (tag, pid, detail))


def close(a, b, tol):
    return bool(abs(float(a) - float(b))
                <= tol)


# ================ A. artifacts ================
pb = r['part_b']
pc = r['part_c']
pd_ = r['part_d']
ok = (r['phase'] == 3128
      and r['name'] == ('omega_p126_joint_swap_'
                        'coord_inject_interaction_'
                        's0match')
      and r['smoke'] is False
      and r['runtime_s'] > 6000
      and r['verdict'] == EXP_V)
part('A1_result_fields', ok,
     'phase=%s smoke=%s rt=%.0f verdict_segs=%d'
     % (r['phase'], r['smoke'], r['runtime_s'],
        len(r['verdict'].split('|'))))

sjb = seal['part_b']['joint_swap']
sci = seal['part_b']['coord_inject']
sdt = seal['part_d']['interaction']
ok = (seal['phase'] == 3128
      and seal['smoke'] is False
      and seal['perm_seed'] == 3128
      and seal['np_a'] == 672
      and seal['np_b'] == 672
      and '26, 28, 30, 32, 34' in sjb
      and '2, 8, 14, 4, 10' in sjb
      and 'ratio >= 2.0' in sjb
      and 'delta=0.1*' in sci
      and 'K in (5, 50, 200)' in sci
      and 'r>=0.3' in sdt
      and 'sep>=0.15' in sdt)
part('A2_seal_frozen', ok,
     'seed=%s np_a=%s joint_w5=%s gate2=%s'
     % (seal['perm_seed'], seal['np_a'],
        '26, 28, 30, 32, 34' in sjb,
        'ratio >= 2.0' in sjb))

exp_keys = sorted([
    'joint_ctrl_P', 'joint_ctrl_A1',
    'joint_write_P', 'joint_write_A1',
    'jointg_ctrl_P', 'jointg_ctrl_A1',
    'jointg_write_P', 'jointg_write_A1',
    'rho_cap_L24', 'rho_g_L20',
    'coords_g_top50', 'flip_lab_s1_P',
    'flip_lab_s3_A1'])
ok = (sorted(z.files) == exp_keys
      and all(z[k].shape == (672,)
              and z[k].dtype == np.float32
              for k in ('joint_ctrl_P',
                        'joint_ctrl_A1',
                        'joint_write_P',
                        'joint_write_A1',
                        'jointg_ctrl_P',
                        'jointg_ctrl_A1',
                        'jointg_write_P',
                        'jointg_write_A1'))
      and z['rho_cap_L24'].shape == (2560,)
      and z['rho_g_L20'].shape == (4096,)
      and z['coords_g_top50'].shape == (50,)
      and z['flip_lab_s1_P'].shape == (672,)
      and z['flip_lab_s3_A1'].dtype == np.int64
      and bool(np.isfinite(
          np.concatenate([
              z['joint_ctrl_P'],
              z['jointg_write_A1']])).all()))
part('A3_npz_schema', ok,
     'keys=%d shapes ok=%s finite=%s'
     % (len(z.files),
        z['rho_cap_L24'].shape == (2560,)
        and z['rho_g_L20'].shape == (4096,),
        bool(np.isfinite(
            z['joint_ctrl_P']).all())))

ok = ('VERDICT: ' + r['verdict'] in logtxt
      and 'P3128 DONE' in logtxt
      and 'B-JOINT' in logtxt
      and 'C-COORD' in logtxt
      and 'C-S0PROBE' in logtxt
      and 'D-INTERACT' in logtxt
      and 'A0 repl gates ok' in logtxt
      and 'seal frozen' in logtxt)
part('A4_run_log', ok,
     'verdict_line=%s done=%s'
     % ('VERDICT: ' + r['verdict'] in logtxt,
        'P3128 DONE' in logtxt))

# ============ B. A0 gates recompute ============


def f32_dm(zf, zbase, side, dc, l, NL):
    return (zf['%s_%s_L%02d' % (side, dc, l)]
            [:, -1]
            - zbase['mlg_s0_%s' % dc][:NP, NL, -1]
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


gq, mcq = gates27(z27, z25, LAY_Q27, NLQ,
                  'dmq')
paq = r['part_a']['repl_gates_q']
ok = (close(gq['write'], paq['write'], 1e-12)
      and close(gq['port'], paq['port'], 1e-12)
      and close(mcq, res27['part_b']
                ['ctrl_median'], 1e-12))
part('B1_repl_gates_qwen', ok,
     'q write %.6f/%.6f port %.6f/%.6f mc %.4f'
     % (gq['write'], paq['write'], gq['port'],
        paq['port'], mcq))

gg, mcg = gates27(z27, z26, LAY_G27, NLG,
                  'dmg')
pag = r['part_a']['repl_gates_g']
ok = (close(gg['write'], pag['write'], 1e-12)
      and close(gg['port'], pag['port'], 1e-12)
      and close(mcg, res27['part_c']
                ['ctrl_median'], 1e-12))
part('B2_repl_gates_glm4', ok,
     'g write %.6f/%.6f port %.6f/%.6f mc %.4f'
     % (gg['write'], pag['write'], gg['port'],
        pag['port'], mcg))

# ========== C. joint swap from npz ===========
mc_npz = float(np.median(np.abs(
    np.concatenate([z['joint_ctrl_P'],
                    z['joint_ctrl_A1']]))))
mw_npz = float(np.median(np.abs(
    np.concatenate([z['joint_write_P'],
                    z['joint_write_A1']]))))
rt_npz = mw_npz / max(mc_npz, 1e-12)
ok = (close(mc_npz, pb['med_ctrl'], 1e-5)
      and close(mw_npz, pb['med_write'], 1e-5)
      and close(rt_npz, pb['joint_ratio_q'],
                1e-5)
      and rt_npz < 2.0
      and r['verdict'].split('|')[2]
      == 'qwen_joint_independent')
part('C1_joint_qwen_npz', ok,
     'q ctrl %.4f/%.4f write %.4f/%.4f '
     'ratio %.4f/%.4f'
     % (mc_npz, pb['med_ctrl'], mw_npz,
        pb['med_write'], rt_npz,
        pb['joint_ratio_q']))

mcg_npz = float(np.median(np.abs(
    np.concatenate([z['jointg_ctrl_P'],
                    z['jointg_ctrl_A1']]))))
mwg_npz = float(np.median(np.abs(
    np.concatenate([z['jointg_write_P'],
                    z['jointg_write_A1']]))))
rtg_npz = mwg_npz / max(mcg_npz, 1e-12)
ok = (close(mcg_npz, pc['med_ctrl'], 1e-5)
      and close(mwg_npz, pc['med_write'],
                1e-5)
      and close(rtg_npz, pc['joint_ratio_g'],
                1e-5)
      and rtg_npz < 2.0
      and r['verdict'].split('|')[5]
      == 'glm4_joint_independent')
part('C2_joint_glm4_npz', ok,
     'g ctrl %.4f/%.4f write %.4f/%.4f '
     'ratio %.4f/%.4f'
     % (mcg_npz, pc['med_ctrl'], mwg_npz,
        pc['med_write'], rtg_npz,
        pc['joint_ratio_g']))

# ========== D. coord injection ===========
h_out = capb['h_out'].astype(np.float32)
m_cap = capb['m'].astype(np.float64)
li = CB_LAYERS.index(INJ_LAYER_Q)
hl = h_out[:, li, :]
hr = np.argsort(np.argsort(hl, axis=0),
                axis=0).astype(np.float64)
mr = np.argsort(np.argsort(m_cap)).astype(
    np.float64)
hc = hr - hr.mean(0, keepdims=True)
mc2 = mr - mr.mean()
den = np.sqrt((hc * hc).sum(0)
              * float((mc2 * mc2).sum()))
rho_cap = (hc * mc2[:, None]).sum(0) / den
norms = np.sqrt((hl.astype(np.float64)
                 * hl.astype(np.float64))
                .sum(1))
delta_q = INJ_FRAC * float(norms.mean())
d_rho = float(np.max(np.abs(
    rho_cap - z['rho_cap_L24'].astype(
        np.float64))))
order_rho = np.argsort(-np.abs(rho_cap))
top200 = [float(rho_cap[i])
          for i in order_rho[:200]]
d_top = float(np.max(np.abs(
    np.array(top200)
    - np.array(pb['rho_top200_L24']))))
ok = (d_rho <= 1e-6
      and close(delta_q, pb['delta_q'], 1e-9)
      and d_top <= 1e-9
      and abs(float(np.min(np.abs(top200))))
      > 0.84)
part('D1_rho_cap_recompute', ok,
     'max|rho-rho_npz|=%.2e delta %.4f/%.4f '
     'top200 diff %.2e'
     % (d_rho, delta_q, pb['delta_q'],
        d_top))

cq = pb['coord_q']
fr = [cq['top%d' % K]['flip']
      for K in (5, 50, 200)]
mono = (fr[2] >= fr[1] - 0.01
        and fr[1] >= fr[0] - 0.01)
eff = (cq['top200']['flip']
       >= 2.0 * max(cq['rand50']['flip'],
                    1e-9)
       and cq['top200']['flip'] >= 0.05)
flip_ok = all(
    close(cq['top%d' % K]['flip'],
          (cq['top%d' % K]['flip_p']
           + cq['top%d' % K]['flip_m']) / 2.0,
          1e-12) for K in (5, 50, 200))
ok = (mono and eff and flip_ok
      and fr[0] < fr[1] < fr[2]
      and cq['top200']['flip']
      > 3.0 * cq['rand50']['flip']
      and r['verdict'].split('|')[3]
      == 'qwen_coord_effective'
      and r['verdict'].split('|')[4]
      == 'qwen_coord_dose_monotonic')
part('D2_coord_gates_qwen', ok,
     'flips %s rand %.4f mono=%s eff=%s'
     % (['%.4f' % v for v in fr],
        cq['rand50']['flip'], mono, eff))

coords_g = z['coords_g_top50']
rg = z['rho_g_L20'].astype(np.float64)
vals_sel = np.sort(np.abs(rg[coords_g]))
vals_top = np.sort(np.abs(rg))[-50:]
cg = pc['coord_g']
ok = (len(set(coords_g.tolist())) == 50
      and int(coords_g.min()) >= 0
      and int(coords_g.max()) < 4096
      and float(np.max(np.abs(
          vals_sel - vals_top))) <= 1e-6
      and close(cg['top50']['flip'], 0.1399,
                1e-3)
      and cg['top50']['flip'] >= 0.05
      and r['verdict'].split('|')[6]
      == 'glm4_coord_effective')
part('D3_coord_glm4', ok,
     'top50 flip %.4f vals_match=%s '
     'uniq=%d'
     % (cg['top50']['flip'],
        float(np.max(np.abs(
            vals_sel - vals_top))) <= 1e-6,
        len(set(coords_g.tolist()))))

# ========== E. interaction ==========
from transformers import AutoTokenizer  # noqa: E402

tok_g = AutoTokenizer.from_pretrained(
    MDIR_G, trust_remote_code=True)


def pol_raw(tok_id):
    if not tok_id:
        return (0, 0)
    t = tok_g.decode([int(tok_id)])
    ts = t.strip()
    tl = ts.lower()
    if tl.startswith('yes'):
        return (1, 0)
    if tl.startswith('no'):
        return (-1,
                1 if ts.startswith('No')
                else 0)
    return (0, 0)


def flip_recompute(c, dc):
    fl = np.zeros(NP, dtype=np.int64)
    reg = z27['regen_%s_%s' % (c, dc)]
    for j in range(NP):
        b12 = [int(v) for v in
               z26['gen_base_%s' % dc][j]]
        bp, _c = pol_raw(b12[0])
        rp, _c2 = pol_raw(reg[j, 0])
        fl[j] = int(rp != bp)
    return fl


fl_all = {}
for c in ('s1', 's2', 's3'):
    for dc in DIRS:
        fl_all[(c, dc)] = flip_recompute(
            c, dc)
e1 = bool(np.array_equal(
    fl_all[('s1', 'P')],
    z['flip_lab_s1_P']))
e3 = bool(np.array_equal(
    fl_all[('s3', 'A1')],
    z['flip_lab_s3_A1']))
part('E1_flip_labels', e1 and e3,
     's1_P=%s s3_A1=%s (rates %.4f/%.4f)'
     % (e1, e3, fl_all[('s1', 'P')].mean(),
        fl_all[('s3', 'A1')].mean()))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(
        np.float64)
    rb = np.argsort(np.argsort(b)).astype(
        np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


dmg_final = {}
for grp in ('write', 'port', 'ctrl'):
    for l in LAY_G27[grp]:
        for dc in DIRS:
            key = '%s_L%02d' % (dc, l)
            dmg_final[key] = (
                z27['dmg_' + key][:, -1]
                - z26['mlg_s0_%s' % dc][:NP,
                                        NLG, -1])
prof = {}
maxd = 0.0
for c in ('s1', 's2', 's3'):
    for dc in DIRS:
        fl = fl_all[(c, dc)]
        for l in (LAY_G27['write']
                  + LAY_G27['port']
                  + LAY_G27['ctrl']):
            key = '%s_L%02d' % (dc, l)
            v = spearman(np.abs(dmg_final[key]),
                         fl)
            k = '%s_%s_L%02d' % (c, dc, l)
            maxd = max(maxd, abs(
                v - pd_['rho_prof'][k]))
            prof[k] = v
pP = np.array([prof['s1_P_L%02d' % l]
               for l in LAY_G27['write']])
pA = np.array([prof['s3_A1_L%02d' % l]
               for l in LAY_G27['write']])
diff = pP - pA
pi = int(np.argmax(np.abs(diff)))
pk_l = LAY_G27['write'][pi]
pk_s = float(np.abs(diff[pi]))
pk_r = float(max(abs(pP[pi]), abs(pA[pi])))
gate_ok = not (pk_r >= V_GATE
               and pk_s >= V_SEP)
ok = (maxd <= 1e-9
      and len(prof) == 48
      and pk_l == pd_['peak_layer'] == 9
      and close(pk_s, pd_['peak_sep'], 1e-9)
      and close(pk_r, pd_['peak_r'], 1e-9)
      and gate_ok
      and r['verdict'].split('|')[1]
      == 'interaction_not_in_field')
part('E2_interaction_recompute', ok,
     'prof48=%d maxdiff %.2e peak L%02d '
     'sep %.4f r %.4f gate=%s'
     % (len(prof), maxd, pk_l, pk_s,
        pk_r, gate_ok))

# ============ F. s0 + ledger ============
ok = (pc['s0_bit_mism'] == 0
      and pc['s0_agree'] == 1.0
      and r['verdict'].split('|')[8]
      == 's0_probe_matched')
part('F1_s0_bitexact', ok,
     'mism=%s agree=%s'
     % (pc['s0_bit_mism'], pc['s0_agree']))

raw = io.open(OUTD + r'\result.json',
              'rb').read()
sha8 = hashlib.sha256(raw).hexdigest()[:8]
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
stored_sha = led['ledger_sha256_8']
last = led['measurements'][-1]
led['ledger_sha256_8'] = SHA27
blob = json.dumps(led, sort_keys=True,
                  ensure_ascii=False)
sha_re = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
ok = (len(led['measurements']) == 265
      and last['meas_id'].startswith(
          'meas3128_omega_p126')
      and last['verdict'] == r['verdict']
      and last['hashes']['result_sha256_8']
      == sha8
      and sha_re == stored_sha)
part('F2_ledger', ok,
     'n=%d sha8=%s stored=%s recomp=%s'
     % (len(led['measurements']), sha8,
        stored_sha, sha_re))

# ============ G. cross-file ============
memo = io.open(MEMO, encoding='utf-8').read()
mt = re.search(r'(?m)^## Phase 3128:.*$',
               memo)
t_ok = mt is not None and len(
    mt.group(0)) < 100
sec_ok = all(s in memo for s in (
    '### 1. 三大发现', '### 2. 关键数值',
    '### 3. 硬伤', '### 4. 机制拼图',
    '### 5. 3129'))
nums_ok = all(s in memo for s in (
    '0.897', '0.896', '0.328', '0.1399',
    'bit_mism 0', 'L09'))
art_ok = ('phase3128/'
          'omega_p126_joint_swap_coord_'
          'inject_interaction_s0match/'
          in memo)
part('G1_memo', t_ok and sec_ok
     and nums_ok and art_ok,
     'title_len=%s secs=%s nums=%s art=%s'
     % (len(mt.group(0)) if mt else -1,
        sec_ok, nums_ok, art_ok))

wd = io.open(WLOG_D,
             encoding='utf-8').read()
ok = ('Phase 3128 Omega-P126 closeout'
      in wd and stored_sha in wd)
part('G2_wlog_workspace', ok,
     '2026-09-25.md sha8=%s' % stored_sha)

wc = io.open(WLOG_C,
             encoding='utf-8').read()
ok = ('Phase 3128 Omega-P126 closeout'
      in wc and stored_sha in wc)
part('G3_wlog_second', ok,
     'workbuddy copy sha8=%s' % stored_sha)

mem = io.open(MEMW,
              encoding='utf-8').read()
ok = ('3128（T4）' in mem
      and '0.897' in mem and '0.896' in mem
      and 'max=3128，下一 3129' in mem)
part('G4_memory', ok,
     '3128 line + max/next ok=%s' % ok)

rep.append('')
rep.append('TOTAL FAILS: %d' % nfail[0])
with io.open(
        r'D:\AI2050\Ai2050-OpenOne\tests'
        r'\gpt5_temp\p3128_verify_out.txt',
        'w', encoding='utf-8') as f:
    f.write('\n'.join(rep))
print('VERIFY_DONE fails=%d' % nfail[0])
