# -*- coding: utf-8 -*-
"""Phase 3140 independent disk verify.
Read-only checks across all persisted
artifacts. Prints PASS/FAIL per item."""
import io
import json
import hashlib
import os
import py_compile

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        + r'\rdc_query_construction_20260913')
D14 = (RDIR + r'\phase3140'
       r'\omega_p138_layerspec_owndecay_'
       r'wrpath_retrattr')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (ROOT + r'\.workbuddy\memory'
        r'\2026-09-29.md')
WMEM = (ROOT + r'\.workbuddy\memory'
        r'\MEMORY.md')
SCRIPT = (ROOT + r'\tests\glm5'
          r'\phase3140_omega_p138_layerspec_'
          r'owndecay_wrpath_retrattr.py')

VERDICT = ('a_3139_ok|spec_repro_3139_ok|'
           'iinj_peak_l19|cinj_peak_l38|'
           'peaks_separated|'
           'spec_dose_monotone|own_below_nb|'
           'own_rank_chance|co_enriched|'
           'wr_active|wr_pcvar_recorded|'
           'retr_v2_flat|retr_v2_below|'
           'retr_v1_repro_ok|xphase_ok|'
           'coverage_full')
RES_SHA = '6d699902'
RETR39 = {'17': 0.05952380952380952,
          '29': 0.03571428571428571,
          '38': 0.05952380952380952}
T39 = {'iinj_l17_d1.0': 0.328125,
       'iinj_l26_d1.0': 0.1015625,
       'iinj_l29_d1.0': 0.109375,
       'iinj_l32_d1.0': 0.078125,
       'cinj_l17_d2.0': 0.1875,
       'cinj_l26_d2.0': 0.2890625,
       'cinj_l29_d1.0': 0.1796875,
       'cinj_l29_d2.0': 0.2578125,
       'cinj_l32_d2.0': 0.2421875}

results = []


def chk(name, cond):
    results.append((name,
                    'PASS' if cond
                    else 'FAIL'))


# 1. result.json
fp = os.path.join(D14, 'result.json')
chk('result.json exists',
    os.path.exists(fp))
raw = io.open(fp, 'rb').read()
sha8 = hashlib.sha256(raw).hexdigest()[:8]
chk('result sha8 == %s (got %s)'
    % (RES_SHA, sha8), sha8 == RES_SHA)
res = json.load(io.open(fp,
                        encoding='utf-8'))
chk('smoke False', res['smoke'] is False)
chk('verdict exact',
    res['verdict'] == VERDICT)
chk('runtime positive',
    res['runtime_s'] > 0)

# 2. seal
fp = os.path.join(D14, 'design_seal.json')
chk('seal exists', os.path.exists(fp))
seal = json.load(io.open(fp,
                         encoding='utf-8'))
chk('seal phase 3140',
    seal['phase'] == 3140)
chk('seal smoke False',
    seal['smoke'] is False)
chk('seal SPECT_L 14 layers',
    seal['constants']['SPECT_L']
    == [17, 19, 21, 23, 25, 26, 27, 29,
        31, 32, 33, 35, 37, 38])

# 3. anchors inside result
pc = res['part_c']
chk('xphase == 1.0', pc['xphase'] == 1.0)
chk('n_repro_match == 9',
    pc['n_repro_match'] == 9)
rep = pc['repro_3139']
chk('repro 9 entries',
    len(rep) == 9)
chk('repro all match',
    all(v['match'] for v in
        rep.values())
    and len(rep) == 9)
chk('repro values == T39 anchors',
    all(abs(rep[k]['want'] - v) < 1e-12
        for k, v in T39.items()))
chk('peak_iinj 19 / peak_cinj 38',
    pc['peak_iinj'] == 19
    and pc['peak_cinj'] == 38)
chk('mono_rate >= 0.9',
    pc['mono_rate'] >= 0.9)
pd_ = res['part_d']
rf = [pd_['own'][k]['rank_frac']
      for k in ('17', '29', '33', '38')]
chk('rank_frac 0.77-0.92 range',
    all(0.75 < v < 0.92 for v in rf))
chk('co36 z>5 / co50 z<1',
    pd_['co_z']['co36']['z'] > 5.0
    and pd_['co_z']['co50']['z'] < 1.0)
pe = res['part_e']
w1 = pe['wr_trials']
chk('WR PC1 dose monotone both L',
    w1['wrpc1_l26_d1.0']
    < w1['wrpc1_l26_d2.0']
    < w1['wrpc1_l26_d4.0']
    and w1['wrpc1_l29_d1.0']
    < w1['wrpc1_l29_d2.0']
    < w1['wrpc1_l29_d4.0'])
chk('WR PC2 both zero',
    w1['wrpc2row_l26_d2.0'] == 0.0
    and w1['wrpc2row_l29_d2.0'] == 0.0)
pf = res['part_f']
chk('retr V1 == 3139 anchors',
    all(abs(pf['retr_v1'][k] - v) < 1e-9
        for k, v in RETR39.items()))
chk('v1_repro_3139 True',
    pf['v1_repro_3139'] is True)
chk('n_pairs 84',
    pf['n_pairs'] == 84)

# 4. npz
import numpy as np
fp = os.path.join(D14, 'p138_readout.npz')
chk('npz exists', os.path.exists(fp))
z = np.load(fp, allow_pickle=False)
need = ['spec_iinj_d1', 'spec_iinj_d2',
        'spec_cinj_d1', 'spec_cinj_d2',
        'spect_l', 'own_eown', 'own_enb',
        'own_eall', 'own_rank', 'co_z',
        'wr_chg', 'retr_v1', 'retr_v2']
chk('npz keys 13',
    all(k in z.files for k in need))
chk('npz spect_l matches',
    list(z['spect_l']) == [17, 19, 21, 23,
                           25, 26, 27, 29, 31,
                           32, 33, 35, 37, 38])
chk('npz co36 z>5',
    float(z['co_z'][0]) > 5.0)

# 5. run_log
fp = os.path.join(D14, 'run_log.txt')
chk('run_log exists',
    os.path.exists(fp))
rl = io.open(fp, encoding='utf-8').read()
chk('run_log VERDICT line',
    'VERDICT: ' + VERDICT in rl)
chk('run_log DONE', 'DONE (' in rl)
chk('run_log 9/9 repro',
    '9/9 bit-match' in rl)
chk('run_log CKPT removed',
    'CKPT removed' in rl)
chk('run_log dumps done',
    'dumps done' in rl)

# 6. CKPT gone (full persistence)
chk('ckpt removed from disk',
    not os.path.exists(os.path.join(
        D14, 'p138_ckpt.pkl')))

# 7. smoke artifacts
fp = os.path.join(D14, 'smoke',
                  'result.json')
chk('smoke result exists',
    os.path.exists(fp))
sres = json.load(io.open(fp,
                         encoding='utf-8'))
chk('smoke result verdict has tags',
    sres['verdict'].startswith(
        'a_3139_ok|'))

# 8. ledger
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
chk('ledger n=277',
    len(led['measurements']) == 277)
last = led['measurements'][-1]
chk('ledger last phase 3140',
    last.get('phase') == 3140)
chk('ledger sha8 match',
    last['hashes']['result_sha256_8']
    == RES_SHA)
chk('ledger verdict match',
    last['verdict'] == VERDICT)
chk('ledger repro 9/9',
    last['anchors']['repro_3139']
    == '9/9')

# 9. MEMO
mt = io.open(MEMO, encoding='utf-8').read()
chk('MEMO 3140 header once',
    mt.count('## Phase 3140') == 1)
chk('MEMO T4 23rd',
    'T4 第23Phase' in mt)
chk('MEMO 3141 prereg',
    '3141（Ω-P139）预注册' in mt)
for key in ('发现 1（×3）', '发现 2（×3）',
            '发现 3（×3）', '发现 4（×3）',
            '发现 5（×3）'):
    chk('MEMO has %s' % key,
        key in mt)
chk('MEMO anchors line',
    'result sha8=6d699902' in mt)

# 10. workspace log + MEMORY
wl = io.open(WLOG, encoding='utf-8').read()
chk('wlog 09-29 has 3140 closeout',
    'Omega-P138) 闭环' in wl)
wm = io.open(WMEM, encoding='utf-8').read()
chk('MEMORY max=3140',
    '- max=3140' in wm)
chk('MEMORY 3140 anchor line',
    '3140（T4）：层位谱分离正式化' in wm)
chk('MEMORY 3141 next',
    '下一 3141' in wm)

# 11. script on disk + compiles
chk('script exists',
    os.path.exists(SCRIPT))
try:
    py_compile.compile(SCRIPT,
                       doraise=True)
    chk('script compiles', True)
except Exception as _e:
    chk('script compiles (%s)' % _e,
        False)
st = io.open(SCRIPT,
             encoding='utf-8').read()
chk('script SPECT_L present',
    'SPECT_L = (17, 19, 21, 23, 25, 26'
    in st)
chk('script rev-3140a fix present',
    'used_rels = {p2r.get(pk)}' in st)
chk('script rev-3140b fix present',
    'RETR39_S[str(l)]' in st)

# summary
n_pass = sum(1 for _, s in results
             if s == 'PASS')
n_fail = len(results) - n_pass
out = ['P3140 DISK VERIFY: %d/%d PASS'
       % (n_pass, len(results)), '']
for name, s in results:
    if s == 'FAIL':
        out.append('FAIL: ' + name)
out.append('')
out.append('ALL PASS' if n_fail == 0
           else '%d FAIL' % n_fail)
io.open(ROOT + r'\tests\gpt5_temp'
        r'\p3140_disk_verify_report.txt',
        'w', encoding='utf-8').write(
    '\n'.join(out))
print('\n'.join(out[-3:]))
