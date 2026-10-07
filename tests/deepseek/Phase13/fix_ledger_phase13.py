# -*- coding: utf-8 -*-
"""Phase 13 修正后补跑 Ledger（295 -> 296）。

① 修 `closeout_phase13.py` 的 `%` 转义（rev_note 里新增的 `9.39%`/`0.73%` 是 `%`-格式串的字面量，
   必须写成 `%%`，否则 `TypeError: not enough arguments for format string`）；
② 只做 Ledger 补登（judgement 与 memo 基线快照已在上一轮正确落盘，不重复执行）。
"""
import io
import os
import json
import time
import shutil
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
T13 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase13')
S13 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase13')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
OUT = os.path.join(S13, 'fix_ledger_phase13.txt')

o = []
def w(s=''):
    o.append(str(s)); print(s)


def full(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def sha8(p):
    return full(p)[:8]


# ---------- ① 修源文件的 % 转义 ----------
CO = os.path.join(S13, 'closeout_phase13.py')
t = io.open(CO, encoding='utf-8').read()
old = "'XH_RANGE margin shrink from 9.39% to 0.73%); 17 adjacent pairs are not independent so no multiplicity-corrected '"
new = "'XH_RANGE margin shrink from 9.39%% to 0.73%%); 17 adjacent pairs are not independent so no multiplicity-corrected '"
assert t.count(old) == 1, 'rev_note % 转义锚点 count=%d' % t.count(old)
io.open(CO, 'w', encoding='utf-8').write(t.replace(old, new))
chk = io.open(CO, encoding='utf-8').read()
assert chk.count('9.39%% to 0.73%%') == 1
w('[1] closeout_phase13.py % 转义已修（count==1 回读通过）')
import py_compile
py_compile.compile(CO, doraise=True)
w('    py_compile OK')

# ---------- ② Ledger 补登 ----------
RESP = os.path.join(T13, 'result_phase13.json')
SEALP = os.path.join(T13, 'N2h1a6_design_seal.json')
EXECP = os.path.join(T13, 'execution_phase13.json')
R = json.load(io.open(RESP, encoding='utf-8'))
A4 = R['A4_concentration']; A5c = R['A5_counts']; PR = R['predictions_check']; EX = R['extra']

bk = os.path.join(T13, 'atlas_ledger_backup_pre_phase13.json')
b_sha = full(LEDGER)
if len(json.load(io.open(LEDGER, encoding='utf-8'))['measurements']) == 295:
    shutil.copy2(LEDGER, bk)
    w('[2] 备份刷新 %s sha8=%s' % (os.path.basename(bk), sha8(bk)))
LG = json.load(io.open(LEDGER, encoding='utf-8'))
n0 = len(LG['measurements'])
assert n0 == 295, 'Ledger 计数不是 295（实际 %d）' % n0

n_rows = (R['panel']['profile_sites'] * R['panel']['BS'] * 2
          + R['panel']['BP'] * 2
          + R['panel']['BS'] * R['panel']['conf_sites'] * 2)
entry = {
 'phase': 13,
 'name': 'n2h1a6_paired_site_bootstrap_discriminability_concentration_coordinate_dependence_qwen3_4b',
 'seal_sha8': sha8(SEALP), 'exec_sha8': sha8(EXECP), 'result_sha8': sha8(RESP),
 'evidence_level': 'statistical', 'model_scope': 'qwen3-4b',
 'n_rows': int(n_rows), 'prereg_id': 'N2h1a6', 'superseded_by': None,
 'verdict': 'paired_%s__%s' % (str(R['disc_verdict']).lower(), str(R['P13_verdict']).lower()),
 'rev_note': (
   'deepseek/N line Phase 13 (N2h1-alpha-6), qwen3-4b, ZERO extra forward passes, CPU only, no torch import. '
   'Pure re-analysis of the per-pair matrices already landed by Phase 12 (E2_pairs 18x14x24, E6_pairs 14x24, '
   'E5_pairs 4x4x17) plus FULL_SWAP_pairs and the frozen seed 20261001. '
   'Device anchor (strongest in the line): replaying the BRNG stream reproduces Phase 12 BIT-IDENTICALLY - '
   'J_ci (18 sites x lo/hi/med), top3_share_x, top3_share_recover, rho_recover, rho_xhalf, R_ci(recover,xhalf), '
   'AND both 2000-value permutation nulls, all max|d| = 0.000e+00; plus the confirmation-set bootstrap band. '
   'Main result 1 (discriminability, extends Phase 11 B3): paired percentile bootstrap on adjacent-site differences '
   'Delta_b(i) = J_b(i) - J_b(i+1) (same idx_b) resolves %d/17 adjacent pairs, versus only %d/17 under the '
   'independent-CI rule Phase 11/12 used; median tightening ratio %.4f; cov>0 for 17/17 pairs (hard assertion '
   '"cov>0 => sd_paired < sd_indep" holds with no counterexample). So the Phase 11 "16/17 overlapping" statement '
   'was a conservative-rule artifact. For xhalf only %d/17 pairs resolve. '
   'Main result 2 (Phase 12 G2_mid re-explained): the concentration statistic top3_share is NOT coordinate-invariant. '
   'xhalf: top3_share = %.4f, band [%.4f, %.4f], P(>=0.60) = %.4f, P(<=0.40) = %.4f, argmax window = %d (deepest, '
   'L28->L34) with bootstrap frequency %.4f. J_swap: top3_share = %.4f, band [%.4f, %.4f], P(>=0.60) = %.4f, '
   'P(<=0.40) = %.4f, argmax window = %d (shallow, L7->L10) with frequency %.4f. Window distance = %d. '
   'Hence verdict %s. The frozen death-line criterion "a few layers carry >= 60%% of the spread" HOLDS in the '
   'J_swap coordinate (79.53%%, P=0.973) and FAILS in the xhalf coordinate (57.45%%, P=0.379). '
   'Deep tail is real not noise: L32->L34 paired band on xhalf = [%+.6f, %+.6f] (excludes 0) and on J = '
   '[%+.6f, %+.6f]; confirmation set (n=17) L20->L34 dX = %+.6f, band [%+.6f, %+.6f], same sign as discovery. '
   'All 3 pre-registered predictions PASS (P1 deep window, P2 shallow window, P3 jump-profile decorrelation '
   'Spearman %.4f <= 0.2). '
   'honesty: paired bootstrap only characterises within-panel resampling uncertainty and does NOT improve '
   'cross-instance/cross-class transferability; N_dec is statistic-dependent (IQR-denominator variant gives '
   '%d/17 instead of %d/17); xhalf bands are limited by the 14-point alpha grid (leave-one-out drops XH_RANGE '
   'from 0.1094 to 0.1007, i.e. the G0 margin shrinks from 9.39%% to 0.73%% but the 0.10 threshold is NOT breached); '
   '17 adjacent pairs are not independent so no multiplicity-corrected significance claim is made.'
   % (A5c['N_dec_J'], A5c['N_dec_indep_J'], A5c['tighten_median'], A5c['N_dec_X'],
      A4['xhalf']['hat'], A4['xhalf']['ci']['lo'], A4['xhalf']['ci']['hi'],
      A4['xhalf']['P_ge_060'], A4['xhalf']['P_le_040'], EX['mode_x'],
      A4['xhalf']['win_hist'][str(EX['mode_x'])] / max(sum(A4['xhalf']['win_hist'].values()), 1),
      A4['J']['hat'], A4['J']['ci']['lo'], A4['J']['ci']['hi'],
      A4['J']['P_ge_060'], A4['J']['P_le_040'], EX['mode_j'],
      A4['J']['win_hist'][str(EX['mode_j'])] / max(sum(A4['J']['win_hist'].values()), 1),
      abs(EX['mode_x'] - EX['mode_j']), R['P13_verdict'],
      R['A6_deep_tail']['L32_L34']['dX']['lo'], R['A6_deep_tail']['L32_L34']['dX']['hi'],
      R['A6_deep_tail']['L32_L34']['dJ']['lo'], R['A6_deep_tail']['L32_L34']['dJ']['hi'],
      R['A7_confirmation']['pairs'][2]['dX']['obs'], R['A7_confirmation']['pairs'][2]['dX']['lo'],
      R['A7_confirmation']['pairs'][2]['dX']['hi'], PR['P3']['got'],
      R['A9_steepness_alt']['N_dec_J_alt'], A5c['N_dec_J'])),
 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
}
LG['measurements'].append(entry)
LG.setdefault('migration_history', []).append({
 'phase': 13,
 'from_version': LG.get('version'), 'to_version': LG.get('version'),
 'backup': 'tests/deepseek_temp/Phase13/atlas_ledger_backup_pre_phase13.json',
 'backup_sha256_8': sha8(bk),
 'note': ('deepseek/N line backfill round 6 (continues Phase 8/9/10/11/12 backfills): appended Phase 13 '
          '(N2h1-alpha-6). Zero-forward re-analysis phase; note this entry was re-appended after a same-session '
          'factual correction (A8 margin, not threshold breach). ledger_sha256_8 remains NOT recomputed '
          '(recipe unknown; marked stale since Phase 8) => use per-file sha8 in this entry instead. '
          'pre-append file sha8 %s.' % b_sha[:8]),
})
io.open(LEDGER, 'w', encoding='utf-8').write(json.dumps(LG, ensure_ascii=False, indent=1))
LG2 = json.load(io.open(LEDGER, encoding='utf-8'))
w('[3] ledger measurements %d -> %d ; file sha8 %s -> %s ; backup sha8 %s' %
  (n0, len(LG2['measurements']), b_sha[:8], full(LEDGER)[:8], sha8(bk)))
w('    tail verdict: %s' % LG2['measurements'][-1]['verdict'])
w('    rev_note 含修正表述: %s' % ('NOT breached' in entry['rev_note']))
assert len(LG2['measurements']) == 296
assert 'NOT breached' in entry['rev_note']
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o) + '\n')
print('DONE ->', OUT)
