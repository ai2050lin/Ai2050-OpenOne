# -*- coding: utf-8 -*-
"""Phase 11 收尾元数据：判决三级标签（含 B 族噪声带）+ Ledger 补登（含备份 + 自哈希说明）+ 备忘录基线冻结。"""
import os, io, json, hashlib, shutil, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P11T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase11')
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
OUT = os.path.join(P11T, 'closeout_phase11.txt')

o = []


def w(s=''):
    o.append(str(s)); print(s)


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def full(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


RESP = os.path.join(P11T, 'result_phase11.json')
SEALP = os.path.join(P11T, 'N2h1a4_design_seal.json')
REPP = os.path.join(P11T, 'n2h1a4_report_qwen3-4b.txt')
EXECP = os.path.join(P11T, 'execution_phase11.json')

R = json.load(io.open(RESP, encoding='utf-8'))
res_sha, seal_sha, rep_sha, exec_sha = full(RESP), full(SEALP), full(REPP), full(EXECP)
w('result_sha8 %s ; seal_sha8 %s ; exec_sha8 %s ; report_sha8 %s' %
  (res_sha[:8], seal_sha[:8], exec_sha[:8], rep_sha[:8]))

V = R['verdict']
PA, PR, PRX = R['profile_abs'], R['profile_rel'], R['profile_R_ext']
DEC = R['decisions']
E1S = R['sites']['e1']
OBS = R['sites']['own_basis']
BB = R['bootstrap_band']
E3V = R['E3_verdict']


def tag(s):
    if s == 'R':
        return 'R*'
    try:
        return 'L%d' % int(s)
    except Exception:
        return str(s)


J_abs = {tag(s): PA[str(s)]['jump_ratio'] for s in E1S}
XS_abs = {tag(s): PA[str(s)]['x_star'] for s in E1S}
YS_abs = {tag(s): PA[str(s)]['y_sat'] for s in E1S}
CL_abs = {tag(s): PA[str(s)]['cls'] for s in E1S}
J_own = {tag(k): v for k, v in E3V['J_own'].items()}

# ---------- 1. 判决（三级标签）----------
J = {
 'phase': 11,
 'name': 'N2h1-alpha-4 own_basis_full_profile + noise_band',
 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
 'prereg': {'seal_sha8': seal_sha[:8], 'frozen_rules': 'B1/B2/B3/B4 + F7prime (new); P1-P4 (sealed) and Q1-Q3 (Phase 10 amend1) recomputed for cross-phase reproduction',
            'own_basis_sites': 'full 18 profile sites (Phase 10 covered only [7,12,20,34])',
            'bootstrap': 'pair-level percentile bootstrap B=%d, permutation null B=%d' % (BB['B'], R['permutation_null']['B'])},
 'verdict': V,
 'headline': {
   'full_L6': R['full_L6'], 'full_ref_phase9': R['full_ref_phase9'], 'bit_replication': R['bit_replication'],
   'B_verdict': V['B_verdict'],
   'B1a': V['B1a'], 'B1b': V['B1b'], 'B1': V['B1'], 'B2': V['B2'], 'B4': V['B4'], 'F7prime': V['F7prime'],
   'rho_hat': BB['rho_hat'],
   'band_abs': BB['abs'], 'band_rel': BB['rel'], 'band_own': BB['own'],
   'spread_own': E3V['spread'], 'spread_own_ci': E3V['spread_ci'],
   'permutation_null': R['permutation_null'],
   'n_indistinguishable': BB['n_indistinguishable'],
   'J_abs': J_abs, 'J_own': J_own,
   'x_star_abs': XS_abs, 'y_sat_abs': YS_abs, 'cls_abs': CL_abs,
   'Q_abs': V['Q_abs'], 'Q_rel': V['Q_rel'], 'Q_agreement': V['Q_agreement'],
   'P_family_sealed_recomputed': {k: DEC.get(k) for k in ('P4_once_formed', 'P1_single_source', 'P2_accumulate', 'P3_readout_only')},
   'P2_spearman_recomputed': DEC.get('P2_spearman'),
   'V_abs_recomputed': V['V_abs'],
   'V_ownbasis': V['V_ownbasis'], 'V_ownbasis_agree': V['V_ownbasis_agree'],
   'V_readout': V['V_readout'], 'V_readout_note': V['V_readout_note'],
   'E3_overlap': {tag(k): E3V['overlap'][k] for k in E3V['overlap']},
   'confirmation': R['E5'],
   'floors': R['floors'],
   'n6': R['dose_coord']['mean_n6'], 'n6_ref_phase9': R['dose_coord']['mean_n6_ref_phase9'],
   'n6_drift': R['dose_coord']['n6_drift'],
   'off_manifold_alphas': R['off_manifold_alphas'],
   'elapsed_s': R['elapsed_s'],
 },
 'evidence_levels': {
   'bit_anchored': [
     'E0 (S_L6out, alpha=1) 的 dDonor = %.15f，与 Phase 9 full = 10.574739583333335 逐位相等 = %s' % (R['full_L6'], R['bit_replication']),
     'F3 恒等自检（含 R 位点 norm hook）：%s' % R['floors']['F3_dev'],
     'F8 E3(L6)==E1(L6) 数学恒等检查：ok=%s，max|d|=%.3e' % (R['floors']['F8_ok'], R['floors']['F8_max']),
     'F9 E1 对 result_phase10 逐位复现：ok=%s，max|d|=%.3e（同面板同网格同向量，确定性前向）' % (R['floors']['F9_ok'], R['floors']['F9_max']),
     'F10 逐对数组自洽（mean(per_pair)==dDonor）：ok=%s，max|d|=%.3e' % (R['floors']['F10_ok'], R['floors']['F10_max']),
     'F5 面板 12 字段 x Phase8/Phase10 双向逐元素；F6 逐位复现 = %s' % R['floors']['F6_ok'],
   ],
   'statistical': [
     'B1a（绝对剂量）Spearman 95%% bootstrap 区间 = [%s, %s]（hat=%.4f，B=%d）=> 上界 < -0.6 ? %s' %
     (BB['abs']['lo'], BB['abs']['hi'], BB['rho_hat']['abs'], BB['B'], V['B1a']),
     'B1b（相对剂量）Spearman 95%% bootstrap 区间 = [%s, %s]（hat=%.4f）=> 上界 < -0.6 ? %s' %
     (BB['rel']['lo'], BB['rel']['hi'], BB['rho_hat']['rel'], V['B1b']),
     'B2（自基全剖面 %d 位点）Spearman = %.4f，spread = %.3f（95%% 带 %s）=> <=-0.6 且 spread>=3 ? %s' %
     (len(E3V['L']), (E3V['rho_hat'] if E3V['rho_hat'] is not None else float('nan')),
      (E3V['spread'] if E3V['spread'] is not None else float('nan')), E3V['spread_ci'], V['B2']),
     'F7\'（置换零假设 B=%d）95%% 带 = [%s, %s] => |界| < 0.6 ? %s' %
     (R['permutation_null']['B'], R['permutation_null']['lo'], R['permutation_null']['hi'], V['F7prime']),
     'B4（确认集 n=%d）Spearman 带 = %s => 上界 < 0 ? %s' %
     (len(R['E5'].get('sites', {})) and 17, json.dumps(R['E5'].get('spearman_boot'), ensure_ascii=False), V['B4']),
     'B3 分辨力诊断：相邻位点 J 区间重叠对数 = %d %s' % (BB['n_indistinguishable'], BB['indistinguishable_pairs']),
     '封存 P 族（同代码路径重算）：P4=%s P1=%s P2=%s(rho=%s) P3=%s => V_abs=%s' %
     (DEC.get('P4_once_formed'), DEC.get('P1_single_source'), DEC.get('P2_accumulate'),
      str(DEC.get('P2_spearman')), DEC.get('P3_readout_only'), V['V_abs']),
     'Q 族：Q_abs=%s，Q_rel=%s => %s' % (V['Q_abs'], V['Q_rel'], V['Q_agreement']),
   ],
   'descriptive': [
     'J(l) 固定基剖面：%s' % '  '.join('%s:%s' % (k, ('%.2f' % v) if v is not None else 'n/a') for k, v in J_abs.items()),
     'J(l) 自基剖面：%s' % '  '.join('%s:%s' % (k, ('%.2f' % v) if v is not None and (v == v) else 'n/a') for k, v in J_own.items()),
     'E3 位点 U_l 与 U6 的子空间重叠：%s' % '  '.join('%s:%.4f' % (tag(k), E3V['overlap'][k]) for k in E3V['overlap']),
     'V_ownbasis（E3 全剖面与 E1 同判率）= %s (%s)' % (V['V_ownbasis'], V['V_ownbasis_agree']),
     'V_readout：R* class=%s，J=%s' % (PRX['cls'], ('%.2f' % PRX['jump_ratio']) if PRX['jump_ratio'] is not None else 'n/a'),
     '地板：E4_max=%.4f，maxabs=%.3f，比=%.4f，F1_ok=%s' %
     (R['floors']['E4_max'], R['floors']['maxabs_all'], R['floors']['E4_max'] / max(R['floors']['maxabs_all'], 1e-9), R['floors']['F1_ok']),
     '离流形警告（pert_rel > 0.50 的 alpha）：%s' % R['off_manifold_alphas'],
   ],
 },
 'honesty': [
   '本 Phase 输出的是「同一扰动在不同深度、不同基下的因果剂量-响应」，不是模型内部轨迹，也不是权重级证明。',
   'bootstrap 只覆盖发现集 24 个配对的组成不确定性；不覆盖 alpha 网格离散化、面板选择、基选择、模板选择的不确定性。区间是下界乐观的。',
   'n=24 偏小，percentile bootstrap 在小样本下欠覆盖；B=2000 与固定 seed 保证可复现，但 95% 区间不是严格频率覆盖。',
   '『位点 J 的精度』与『剖面形状的可信度』是两个不同的不确定性，分开列示。',
   '自基臂的 U_ell 由 discovery 估计，与 E1 的 U6 共享同一数据；两者的一致性不等于独立验证。',
   'E3 在 ell=6 与 E1 恒等（同一对象），F8 只检查实现正确性，不提供独立证据。',
   '跨层固定 U6 的固定基剖面递减已被 Phase 10 证明主要是基旋转（overlap 0.6892->0.0298）；本 Phase 把该限界由 4 位点采样升级为全剖面陈述，不改变其性质。',
   '单模型 qwen3-4b、单模板 %s是一种、单语言、41 例全单 token。',
   '本 Phase 不触碰权重级归因（N2h1-alpha-1 仍挂账）。',
 ],
 'meta': {'seal_sha8': seal_sha[:8], 'exec_sha8': exec_sha[:8], 'result_sha8': res_sha[:8],
          'report_sha8': rep_sha[:8], 'smoke_dir': 'tests/deepseek_temp/Phase11/smoke/'},
}
jp = os.path.join(P11T, 'judgement_phase11.json')
io.open(jp, 'w', encoding='utf-8').write(json.dumps(J, ensure_ascii=False, indent=1))
w('judgement_phase11.json written (%d bytes)' % os.path.getsize(jp))

# ---------- 2. Ledger 补登（先备份）----------
bk = os.path.join(P11T, 'atlas_ledger_backup_pre_phase11.json')
shutil.copy2(LEDGER, bk)
b_sha = full(LEDGER)
LG = json.load(io.open(LEDGER, encoding='utf-8'))
n0 = len(LG['measurements'])
n_rows = ((len(E1S) * 7 + len(E1S) * 7 + len(OBS) * 7 + len(R['sites']['floor']) * 3 + 7 + 12) * 24 +
          (len(R['sites']['conf']) * 7) * 17) if R['sites']['conf'] else \
         ((len(E1S) * 7 + len(E1S) * 7 + len(OBS) * 7 + len(R['sites']['floor']) * 3 + 7 + 12) * 24)
entry = {
 'phase': 11,
 'name': 'n2h1a4_own_basis_full_profile_noise_band_qwen3_4b',
 'seal_sha8': seal_sha[:8],
 'result_sha8': res_sha[:8],
 'evidence_level': 'statistical',
 'model_scope': 'qwen3-4b',
 'n_rows': n_rows,
 'prereg_id': 'N2h1a4',
 'superseded_by': None,
 'verdict': 'own_basis_full_profile_%s__band_%s' % (str(V['Q_abs']).lower(), str(V['B_verdict']).lower()),
 'rev_note': (
   'deepseek/N line Phase 11 (N2h1-alpha-4), qwen3-4b, same 41-instance panel inherited bit-for-bit from Phase 8/10 '
   '(discovery 24 / confirmation 17), same U6 (5-dim class-mean subspace at L6 output), same template. '
   'Design: (1) spread the self-basis arm (inject h_ell_recip + alpha*P_U{ell}(diff_ell), U_ell estimated at that very '
   'layer) from 4 sites to all 18 profile sites -> dual-basis J(ell) profile; (2) land per-pair dDonor arrays (n=24) and '
   'run a pair-level percentile bootstrap (B=2000) plus a permutation null (B=2000) to put 95%% bands on J(ell) and on '
   'the across-site Spearman. Zero extra forward passes (E1/E1b/E3 keep the exact Phase 10 grids). '
   'bit_anchored: E0 at S_L6out alpha=1 reproduces Phase 9 full = 10.574739583333335 bit-identically; F8 proves '
   'E3(L6)==E1(L6) to 1e-12 (self-basis is an identity at ell=6); F9 proves every E1 (site,alpha) cell reproduces '
   'result_phase10 bit-for-bit; F10 proves mean(per_pair)==dDonor. '
   'statistical: B1a/B1b (Q2 Spearman band upper bound < -0.6 in absolute / relative dose) = %s / %s; '
   'B2 (self-basis full profile Spearman <= -0.6 and spread >= 3) = %s; F7prime (permutation null band |bound| < 0.6) = %s; '
   'B4 (confirmation-set band upper bound < 0) = %s. Combined verdict = %s. '
   'honesty: bootstrap covers only pair-composition uncertainty, not grid/panel/basis/template selection; n=24 makes the '
   'percentile interval optimistically narrow; the self-basis U_ell shares the discovery data with U6; E3(L6) is an '
   'identity so it is not independent evidence; fixed-basis depth decay is mostly basis rotation (Phase 10).'
   % (V['B1a'], V['B1b'], V['B2'], V['F7prime'], V['B4'], V['B_verdict'])),
 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
}
LG['measurements'].append(entry)
LG.setdefault('migration_history', []).append({
 'phase': 11,
 'from_version': LG.get('version'), 'to_version': LG.get('version'),
 'backup': 'tests/deepseek_temp/Phase11/atlas_ledger_backup_pre_phase11.json',
 'backup_sha256_8': sha8(bk),
 'note': ('deepseek/N line backfill round 4 (continues the Phase 8/9/10 backfills): appended Phase 11 '
          '(N2h1-alpha-4) measurement. ledger_sha256_8 remains NOT recomputed (recipe unknown; marked stale/unverified '
          'since Phase 8) => use per-file sha8 in this entry instead. pre-append file sha8 %s.' % b_sha[:8]),
})
io.open(LEDGER, 'w', encoding='utf-8').write(json.dumps(LG, ensure_ascii=False, indent=1))
LG2 = json.load(io.open(LEDGER, encoding='utf-8'))
w('ledger measurements %d -> %d ; file sha8 %s -> %s ; backup_sha8 %s' %
  (n0, len(LG2['measurements']), b_sha[:8], full(LEDGER)[:8], sha8(bk)))
w('ledger tail verdict: %s' % LG2['measurements'][-1]['verdict'])

# ---------- 3. 备忘录基线（pre-append 快照）----------
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
        'bom': mb[:3] == b'\xef\xbb\xbf', 'crlf': mb.count(b'\r\n'),
        'bare_lf': mb.count(b'\n') - mb.count(b'\r\n'),
        'sections': heads,
        'note': ('Phase 11 追加前快照。Phase 10 节起 L2036；Phase 标题共 10 个。')}
io.open(os.path.join(P11T, 'memo_baseline_preappend_phase11.json'), 'w', encoding='utf-8').write(
    json.dumps(base, ensure_ascii=False, indent=1))
w('memo baseline(pre-append): bytes %d lines %d sha8 %s bare_lf %d' %
  (base['bytes'], base['lines'], base['sha256'][:8], base['bare_lf']))

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
print('DONE ->', OUT)
