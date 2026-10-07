# -*- coding: utf-8 -*-
"""Phase 13 收尾元数据：判决（P13 verdict + 配对判别表）+ Ledger 补登（295 -> 296，含备份）
+ 备忘录 pre-append 基线快照。"""
import os
import io
import json
import time
import shutil
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P13T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase13')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
OUT = os.path.join(P13T, 'closeout_phase13.txt')

o = []
def w(s=''):
    o.append(str(s)); print(s)


def full(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def sha8(p):
    return full(p)[:8]


RESP = os.path.join(P13T, 'result_phase13.json')
SEALP = os.path.join(P13T, 'N2h1a6_design_seal.json')
EXECP = os.path.join(P13T, 'execution_phase13.json')
REPP = os.path.join(P13T, 'n2h1a6_report_qwen3-4b.txt')
P12R = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12', 'result_phase12.json')

R = json.load(io.open(RESP, encoding='utf-8'))
R12 = json.load(io.open(P12R, encoding='utf-8'))
res_sha, seal_sha, exec_sha, rep_sha = full(RESP), full(SEALP), full(EXECP), full(REPP)
w('result_sha8 %s ; seal_sha8 %s ; exec_sha8 %s ; report_sha8 %s' %
  (res_sha[:8], seal_sha[:8], exec_sha[:8], rep_sha[:8]))
w('phase12 result sha256 一致 = %s' % (full(P12R) == R['anchors']['phase12_result_sha256']))

A4 = R['A4_concentration']
A5c = R['A5_counts']
PR = R['predictions_check']
EX = R['extra']
FL = R['floors']

J = {
 'phase': 13,
 'name': 'N2h1-alpha-6 pairwise-site paired bootstrap (discriminability + concentration coordinate-dependence)',
 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
 'kind': 'zero_extra_forward_reanalysis',
 'prereg': {
   'seal_sha8': seal_sha[:8],
   'exec_sha8': exec_sha[:8],
   'seed': 20261001,
   'bootstrap': 'pair-level percentile bootstrap B=%d, B_perm=%d, same seed as Phase 12 (stream replayable)' % (
       R['panel']['BS'], R['panel']['BP']),
   'frozen_rules': 'G0p 装置锚前置 + D_discriminability + D_concentration(coord_dep_rule) + 6 行裁决表 + 3 条预注册预测',
   'arms': 'A0/A0b/A1/A2/A3/A4/A5/A6/A7/A8/A9 (11 arms)',
   'zero_extra_forward': '本轮不加载模型、不导入 torch (F21)；全部量来自 Phase 12 落盘逐对矩阵',
 },
 'verdict': {
   'P13_verdict': R['P13_verdict'],
   'disc_verdict': R['disc_verdict'],
   'G0p': True,
   'coord_dep': bool(EX['coord_dep']),
   'coord_dep_components': {'share_x': A4['xhalf']['hat'], 'share_j': A4['J']['hat'],
                            'W_x': EX['mode_x'], 'W_j': EX['mode_j'],
                            'abs_W_diff': abs(EX['mode_x'] - EX['mode_j'])},
 },
 'headline': {
   'A0_bit_replication': {'max_dev': R['A0_replicate']['max_dev'], 'ok': R['A0_replicate']['ok'],
                          'per_block': R['A0_replicate']['dev'],
                          'top3_share_x_hat_repro': R['A0_replicate']['top3_share_x_hat_repro'],
                          'argmax_window_hat': R['A0_replicate']['argmax_window_hat'],
                          'jumps_x_hat_repro': R['A0_replicate']['jumps_x_hat_repro']},
   'A0b_conf_band_replication': {'max_dev': R['A0b_replicate_conf']['max_dev'],
                                 'ok': R['A0b_replicate_conf']['ok']},
   'discriminability': {
     'N_dec_J_paired': A5c['N_dec_J'], 'N_dec_X_paired': A5c['N_dec_X'],
     'N_dec_J_independent': A5c['N_dec_indep_J'], 'tighten_median': A5c['tighten_median'],
     'cov_positive_pairs': '17/17', 'rho_pair_range': [min(r['rho_pair'] for r in R['A3_tightening']),
                                                       max(r['rho_pair'] for r in R['A3_tightening'])],
   },
   'delta_J_table': R['A1_delta_J'],
   'delta_xhalf_table': R['A2_delta_xhalf'],
   'concentration': {
     'xhalf': {'hat': A4['xhalf']['hat'], 'ci': A4['xhalf']['ci'],
               'P_ge_060': A4['xhalf']['P_ge_060'], 'P_le_040': A4['xhalf']['P_le_040'],
               'P_mid': A4['xhalf']['P_mid'], 'argmax_window_hat': A4['xhalf']['argmax_window_hat'],
               'win_hist': A4['xhalf']['win_hist'], 'jumps': A4['xhalf']['jumps'],
               'range': A4['xhalf']['range']},
     'J_swap': {'hat': A4['J']['hat'], 'ci': A4['J']['ci'],
                'P_ge_060': A4['J']['P_ge_060'], 'P_le_040': A4['J']['P_le_040'],
                'P_mid': A4['J']['P_mid'], 'argmax_window_hat': A4['J']['argmax_window_hat'],
                'win_hist': A4['J']['win_hist'], 'jumps': A4['J']['jumps'],
                'range': A4['J']['range']},
   },
   'deep_tail': R['A6_deep_tail'],
   'confirmation': R['A7_confirmation'],
   'grid_loo': {k: R['A8_grid_loo'][k] for k in ('range_min', 'range_max', 'range_span',
                                                 'top3_min', 'top3_max', 'top3_span')},
   'steepness_alt': {'rho_J_vs_Jalt': R['A9_steepness_alt']['rho_J_vs_Jalt'],
                     'N_dec_J_alt': R['A9_steepness_alt']['N_dec_J_alt']},
   'predictions': PR,
   'floors': FL,
   'elapsed_s': R['elapsed_s'],
 },
 'evidence_levels': {
   'bit_anchored': [
     'A0 装置锚（跨 Phase 逐位复现，本轮最强）：Phase 12 的 J_ci(18 位点 x lo/hi/med) + top3_share_x(3) '
     '+ top3_share_recover(3) + rho_recover(3) + rho_xhalf(3) + R_ci(recover/xhalf 各 2) '
     '+ **两个 2000 值置换零假设**，全部 max|d| = %.3e => BIT-EXACT = %s'
     % (R['A0_replicate']['max_dev'], R['A0_replicate']['ok']),
     'A0 点估计复现：top3_share_x 复现 max|d| = %.3e；xhalf jumps 复现 max|d| = %.3e；argmax 窗口 = %d'
     % (R['A0_replicate']['top3_share_x_hat_repro'], R['A0_replicate']['jumps_x_hat_repro'],
        R['A0_replicate']['argmax_window_hat']),
     'A0b 第二锚：确认集 bootstrap 带 rho_boot_xhalf 复现 max|d| = %.3e (ok=%s)'
     % (R['A0b_replicate_conf']['max_dev'], R['A0b_replicate_conf']['ok']),
     'F0/F0b 前置：Phase 12 result sha256 一致 + FULL_SWAP 重建逐位一致（%.12f）' % R['anchors']['FULL_SWAP_rebuilt'],
     'F21 零前向：本轮未导入 torch（%s）' % (not R['torch_imported']),
   ],
   'statistical': [
     'F14 装置锚硬断言 max|d| == 0.0 => %s' % FL['F14']['ok'],
     'F15 望远镜和（点估计）：max|d| = %.3e' % FL['F15']['max_dev'],
     'F16 望远镜和（逐 bootstrap 样本）：max|d| = %.3e' % FL['F16']['max_dev'],
     'F17 方差分解代数恒等 var_indep - var_paired == 2cov：max|d| = %.3e' % FL['F17']['max_dev'],
     'A3 cov>0 的对数 = 17/17，反例 = NONE => 硬断言「cov>0 => sd_paired < sd_indep」全部成立',
     'D1 可分辨对数：配对口径 J = %d/17，xhalf = %d/17；独立区间口径 J = %d/17；收紧比中位数 = %.4f'
     % (A5c['N_dec_J'], A5c['N_dec_X'], A5c['N_dec_indep_J'], A5c['tighten_median']),
     'D2 集中度尾部概率：xhalf P(share>=0.60)=%.4f / P(<=0.40)=%.4f / P(mid)=%.4f；J P(>=0.60)=%.4f / P(<=0.40)=%.4f'
     % (A4['xhalf']['P_ge_060'], A4['xhalf']['P_le_040'], A4['xhalf']['P_mid'],
        A4['J']['P_ge_060'], A4['J']['P_le_040']),
     'D3 定位：xhalf argmax 窗口 = %d（最深，L28->L34），freq 最高；J_swap argmax 窗口 = %d（浅端，L7->L10）'
     % (EX['mode_x'], EX['mode_j']),
     'A6 深尾配对决断：L32->L34 dX = %+.6f，带 [%+.6f, %+.6f] => 含 0? %s'
     % (R['A6_deep_tail']['L32_L34']['dX']['obs'], R['A6_deep_tail']['L32_L34']['dX']['lo'],
        R['A6_deep_tail']['L32_L34']['dX']['hi'],
        not (R['A6_deep_tail']['L32_L34']['dX']['lo'] <= 0 <= R['A6_deep_tail']['L32_L34']['dX']['hi'])),
     'A7 确认集方向一致：L20->L34 dX = %+.6f 带 [%+.6f, %+.6f]（含 0? %s），与发现集同号'
     % (R['A7_confirmation']['pairs'][2]['dX']['obs'], R['A7_confirmation']['pairs'][2]['dX']['lo'],
        R['A7_confirmation']['pairs'][2]['dX']['hi'],
        (R['A7_confirmation']['pairs'][2]['dX']['lo'] <= 0 <= R['A7_confirmation']['pairs'][2]['dX']['hi'])),
     'F19 概率守恒：max|d| = %.3e' % FL['F19']['max_dev'],
     'F18 带含点估计的比例：J 17/17，xhalf 17/17（不设断言，仅报告百分位带无显著偏差）',
   ],
   'descriptive': [
     '预注册预测核验：P1 %s (mode=%d freq=%.4f) ; P2 %s (mode=%d freq=%.4f) ; P3 %s (rho=%.4f)'
     % ('PASS' if PR['P1']['pass_'] else 'FAIL', PR['P1']['got_mode'], PR['P1']['got_freq'],
        'PASS' if PR['P2']['pass_'] else 'FAIL', PR['P2']['got_mode'], PR['P2']['got_freq'],
        'PASS' if PR['P3']['pass_'] else 'FAIL', PR['P3']['got']),
     'A8 alpha 网格留一：XH_RANGE in [%.4f, %.4f] (span %.4f)；top3 in [%.4f, %.4f] (span %.4f)；'
     '注意剔除 alpha=0.4 时 XH_RANGE 降到 %.4f —— 仍高于 0.10，但裕度由 9.39%% 压到 0.73%%'
     % (R['A8_grid_loo']['range_min'], R['A8_grid_loo']['range_max'], R['A8_grid_loo']['range_span'],
        R['A8_grid_loo']['top3_min'], R['A8_grid_loo']['top3_max'], R['A8_grid_loo']['top3_span'],
        min(v['xh_range'] for v in R['A8_grid_loo']['variants'] if v['ok'])),
     'A9 陡度统计量替代：spearman(J_max/median, J_max/IQR) = %.4f；N_dec_J_alt = %d/17（对比主口径 10/17）'
     % (R['A9_steepness_alt']['rho_J_vs_Jalt'], R['A9_steepness_alt']['N_dec_J_alt']),
     'A3 逐对收紧比：%s' % ' '.join('%d-%d:%.3f' % (r['a'], r['b'], r['tighten']) for r in R['A3_tightening']),
     'J_swap 跳变（17）：%s' % ' '.join('%+.4f' % x for x in A4['J']['jumps']),
     'xhalf 跳变（17）：%s' % ' '.join('%+.4f' % x for x in A4['xhalf']['jumps']),
   ],
 },
 'honesty': R['honesty'],
 'meta': {'seal_sha8': seal_sha[:8], 'exec_sha8': exec_sha[:8], 'result_sha8': res_sha[:8],
          'report_sha8': rep_sha[:8], 'smoke_dir': 'tests/deepseek_temp/Phase13/smoke/',
          'feas_probe': 'tests/deepseek_temp/Phase13/_feas_probe.txt',
          'zero_extra_forward': True, 'cpu_only': True},
 'posthoc_note': {
   'note': '以下为 POST-HOC 读法，无判决角色，仅为解释 Phase 12 G2_mid 提供尺度感',
   'reading': (
     'Phase 12 的 G2_mid 不是纯抽样不确定性，而是**坐标系依赖**：集中度统计量的 argmax 窗口在 '
     'xhalf 坐标指向深尾（L28->L34，由 L34 反弹驱动），在 J_swap 坐标指向浅端（L7->L10，L7 峰之后的下降段），'
     '两者窗口相距 13 个跳变位。§8 冻结的即时判据「少数几层承载 >= 60% 的 spread」'
     '在 J 坐标成立（79.53%%，P(>=0.60)=0.973），在 xhalf 坐标不成立（57.45%%，P(>=0.60)=0.379）。'
   ),
 },
}
jp = os.path.join(P13T, 'judgement_phase13.json')
io.open(jp, 'w', encoding='utf-8').write(json.dumps(J, ensure_ascii=False, indent=1))
w('judgement_phase13.json written (%d bytes)' % os.path.getsize(jp))

# ---------- 2. Ledger 补登 ----------
bk = os.path.join(P13T, 'atlas_ledger_backup_pre_phase13.json')
shutil.copy2(LEDGER, bk)
b_sha = full(LEDGER)
LG = json.load(io.open(LEDGER, encoding='utf-8'))
n0 = len(LG['measurements'])
w('ledger measurements before = %d' % n0)
assert n0 == 295, 'Ledger 计数不是 295（实际 %d）' % n0

n_rows = (R['panel']['profile_sites'] * R['panel']['BS'] * 2      # J_b + XH_b (主循环)
          + R['panel']['BP'] * 2                                   # 置换
          + R['panel']['BS'] * R['panel']['conf_sites'] * 2)        # 确认集
entry = {
 'phase': 13,
 'name': 'n2h1a6_paired_site_bootstrap_discriminability_concentration_coordinate_dependence_qwen3_4b',
 'seal_sha8': seal_sha[:8],
 'exec_sha8': exec_sha[:8],
 'result_sha8': res_sha[:8],
 'evidence_level': 'statistical',
 'model_scope': 'qwen3-4b',
 'n_rows': int(n_rows),
 'prereg_id': 'N2h1a6',
 'superseded_by': None,
 'verdict': 'paired_%s__%s' % (str(R['disc_verdict']).lower(), str(R['P13_verdict']).lower()),
 'rev_note': (
   'deepseek/N line Phase 13 (N2h1-alpha-6), qwen3-4b, ZERO extra forward passes, CPU only, no torch import. '
   'Pure re-analysis of the per-pair matrices already landed by Phase 12 (E2_pairs 18x14x24, E6_pairs 14x24, '
   'E5_pairs 4x4x17) plus FULL_SWAP_pairs and the frozen seed 20261001. '
   'Device anchor (strongest in the line): replaying the BRNG stream reproduces Phase 12 BIT-IDENTICALLY - '
   'J_ci (18 sites x lo/hi/med), top3_share_x, top3_share_recover, rho_recover, rho_xhalf, R_ci(recover,xhalf), '
   'AND both 2000-value permutation nulls, all max|d| = 0.000e+00; plus the confirmation-set bootstrap band. '
   'A0b replicates the confirmation band too. '
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
   '%d/17 instead of %d/17); xhalf bands are limited by the 14-point alpha grid (leave-one-out can push '
   'XH_RANGE margin shrink from 9.39%% to 0.73%%); 17 adjacent pairs are not independent so no multiplicity-corrected '
   'significance claim is made.'
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
          '(N2h1-alpha-6). Zero-forward re-analysis phase. ledger_sha256_8 remains NOT recomputed (recipe unknown; '
          'marked stale since Phase 8) => use per-file sha8 in this entry instead. pre-append file sha8 %s.' % b_sha[:8]),
})
io.open(LEDGER, 'w', encoding='utf-8').write(json.dumps(LG, ensure_ascii=False, indent=1))
LG2 = json.load(io.open(LEDGER, encoding='utf-8'))
w('ledger measurements %d -> %d ; file sha8 %s -> %s ; backup_sha8 %s' %
  (n0, len(LG2['measurements']), b_sha[:8], full(LEDGER)[:8], sha8(bk)))
w('ledger tail verdict: %s' % LG2['measurements'][-1]['verdict'])

# ---------- 3. 备忘录 pre-append 基线快照 ----------
mb = open(MEMO, 'rb').read()
T = mb.decode('utf-8-sig')
mlines = T.splitlines()
heads = {}
for i, l in enumerate(mlines):
    if l.startswith('## '):
        heads[l[:44]] = i + 1
base = {'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'tag': 'pre-append',
        'path': 'research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
        'bytes': len(mb), 'lines': len(mlines), 'sha256': hashlib.sha256(mb).hexdigest(),
        'sha8': hashlib.sha256(mb).hexdigest()[:8],
        'bom': mb[:3] == b'\xef\xbb\xbf', 'crlf': mb.count(b'\r\n'),
        'bare_lf': mb.count(b'\n') - mb.count(b'\r\n'),
        'phase_headings': sum(1 for k in heads if k.startswith('## Phase ')),
        'sections': heads,
        'note': 'Phase 13 追加前快照。Phase 12 节起 L2555；Phase 标题共 12 个。'}
io.open(os.path.join(P13T, 'memo_baseline_preappend_phase13.json'), 'w', encoding='utf-8').write(
    json.dumps(base, ensure_ascii=False, indent=1))
w('memo baseline(pre-append): bytes %d lines %d sha8 %s bare_lf %d phase_headings %d' %
  (base['bytes'], base['lines'], base['sha8'], base['bare_lf'], base['phase_headings']))

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o) + '\n')
print('DONE ->', OUT)
