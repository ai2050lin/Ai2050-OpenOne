# -*- coding: utf-8 -*-
"""Phase 10 收尾元数据：判决三级标签 + Ledger 补登（含备份 + 自哈希说明）+ 备忘录基线冻结。"""
import os, io, json, hashlib, shutil, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P10T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase10')
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
OUT = os.path.join(P10T, 'closeout_phase10.txt')

o = []


def w(s=''):
    o.append(str(s)); print(s)


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def full(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


RESP = os.path.join(P10T, 'result_phase10.json')
SEALP = os.path.join(P10T, 'N2h1a3_design_seal.json')
AMENDP = os.path.join(P10T, 'N2h1a3_design_seal_amend1.json')
REPP = os.path.join(P10T, 'n2h1a3_report_qwen3-4b.txt')
EXECP = os.path.join(P10T, 'execution_phase10.json')

R = json.load(io.open(RESP, encoding='utf-8'))
res_sha, seal_sha, amend_sha, rep_sha, exec_sha = full(RESP), full(SEALP), full(AMENDP), full(REPP), full(EXECP)
w('result_sha8 %s ; seal_sha8 %s ; amend_sha8 %s ; exec_sha8 %s ; report_sha8 %s' %
  (res_sha[:8], seal_sha[:8], amend_sha[:8], exec_sha[:8], rep_sha[:8]))

V = R['verdict']
PA = R['profile_abs']
PR = R['profile_rel']
PRX = R['profile_R_ext']
DEC = R['decisions']
XR = R['x_star_rel']
E1S = R['sites']['e1']


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

# ---------- 1. 判决（三级标签）----------
J = {
 'phase': 10,
 'name': 'N2h1-alpha-3 soft_threshold_depth_locating',
 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
 'prereg': {'seal_sha8': seal_sha[:8], 'amend1_sha8': amend_sha[:8],
            'frozen_rules': 'P1-P4 (sealed) + Q1-Q3 (amend1, orientation-corrected)',
            'e1b_grid': 'amend1 A1: 7 points',
            'r_site_grid': 'amend1 A5: E6 extended to alpha=16'},
 'verdict': V,
 'headline': {
   'full_L6': R['full_L6'],
   'full_ref_phase9': R['full_ref_phase9'],
   'bit_replication': R['bit_replication'],
   'J_abs': J_abs, 'x_star_abs': XS_abs, 'y_sat_abs': YS_abs, 'cls_abs': CL_abs,
   'x_star_rel': XR,
   'r_ell': R['dose_coord']['rbar_ell'], 'r_R': R['r_R'],
   'Q_family': DEC['Q_family'], 'Q_abs': V['Q_abs'], 'Q_rel': V['Q_rel'],
   'Q_agreement': V['Q_agreement'],
   'V_readout': V['V_readout'], 'V_readout_note': V['V_readout_note'],
   'V_readout_same_grid': DEC.get('V_readout_same_grid_cls'),
   'R_ext': {k: PRX[k] for k in ('cls', 'jump_ratio', 'x_star', 'k_log', 'R2_log', 'R2_lin', 'gamma', 'y_sat')},
   'V_ownbasis': V['V_ownbasis'], 'V_ownbasis_agree': DEC['V_ownbasis_agree'],
   'V_rel': V['V_rel'], 'V_rel_raw': V['V_rel_raw'],
   'P_family_sealed': {'P4': V['P4'], 'P1': V['P1'], 'P1_ell0': V['P1_ell0'],
                       'P2': V['P2'], 'P2_spearman': V['P2_spearman'], 'P3': V['P3']},
   'E3_overlap': {tag(k): R['E3'][k]['overlap'] for k in R['E3']},
   'confirmation': R['E5'],
   'floors': R['floors'],
   'n6': R['dose_coord']['mean_n6'], 'n6_ref_phase9': R['dose_coord']['mean_n6_ref_phase9'],
   'n6_drift': R['dose_coord']['n6_drift'],
   'off_manifold_alphas': R['off_manifold_alphas'],
   'elapsed_s': R['elapsed_s'],
 },
 'evidence_levels': {
   'bit_anchored': [
     'E0 (S_L6out, alpha=1) 的 dDonor = %.15f，与 Phase 9 的 full = 10.574739583333335 逐位相等 = %s => 同一 U6、同一站点口径、同一面板跨 Phase 复现' %
     (R['full_L6'], R['bit_replication']),
     'F3 恒等自检（含 R 位点 norm hook）：%s，全部 0.000e+00' % R['floors']['F3_dev'],
     'F4 o_proj 输入维 4096 = 32x128 通过；F5 面板逐元素继承（12 字段 x Phase8/Phase9 双向一致）；config sha256 匹配',
     'F6 逐位复现 = %s；F6b 锚点 drift = %s' % (R['floors']['F6_ok'], R['floors']['F6b_anchor_drift']),
   ],
   'statistical': [
     'J(l) 绝对剂量剖面：%s => 相邻位点最大跌幅在 %s -> %s（%.2f -> %.2f，比 %.2f）' %
     ('  '.join('%s:%.2f' % (k, v) for k, v in J_abs.items() if v is not None),
      str(DEC['Q_family']['Q3_ell0']), str(DEC['Q_family']['Q3_ell0'] + 1) if DEC['Q_family']['Q3_ell0'] is not None else 'n/a',
      J_abs.get(tag(DEC['Q_family']['Q3_ell0'])) if DEC['Q_family']['Q3_ell0'] is not None else float('nan'),
      J_abs.get(tag(DEC['Q_family']['Q3_ell0'] + 1)) if DEC['Q_family']['Q3_ell0'] is not None else float('nan'),
      (J_abs.get(tag(DEC['Q_family']['Q3_ell0'])) / J_abs.get(tag(DEC['Q_family']['Q3_ell0'] + 1)))
      if (DEC['Q_family']['Q3_ell0'] is not None and J_abs.get(tag(DEC['Q_family']['Q3_ell0'] + 1))) else float('nan')),
     'Q 族（amend1 方向修正）：Q1=%s Q2=%s Q3=%s(ell0=%s)，J 全剖面 spread=%s，Spearman(J,l)=%s => Q_abs=%s' %
     (DEC['Q_family']['Q1'], DEC['Q_family']['Q2'], DEC['Q_family']['Q3'], str(DEC['Q_family']['Q3_ell0']),
      ('%.2f' % DEC['Q_family']['spread']) if DEC['Q_family']['spread'] is not None else 'n/a',
      ('%.3f' % DEC['Q_family']['spearman']) if DEC['Q_family']['spearman'] is not None else 'n/a',
      V['Q_abs']),
   '半饱和点（相对剂量，跨位点可比）x*_rel(l)：%s ；绝对坐标 x*_abs(l)：%s' %
   ('  '.join('%s:%s' % (tag(k), ('%.3f' % v) if v is not None else 'n/a') for k, v in XR.items()),
    '  '.join('%s:%s' % (k, ('%.3f' % v) if v is not None else 'n/a') for k, v in XS_abs.items())),
     '饱和值 y_sat(l)：%s （y = dDonor/full_L6）' %
     '  '.join('%s:%.3f' % (k, v) for k, v in YS_abs.items()),
     'V_readout：R* 位点（扩展网格 alpha<=16）class=%s，J=%s，y_sat=%.3f ；同网格 R class=%s => %s' %
     (PRX['cls'], ('%.2f' % PRX['jump_ratio']) if PRX['jump_ratio'] is not None else 'n/a', PRX['y_sat'],
      DEC.get('V_readout_same_grid_cls'), V['V_readout_note']),
     'V_ownbasis（E3 自基稳健性与 E1 同判率）= %s (%s)' % (V['V_ownbasis'], DEC['V_ownbasis_agree']),
     '确认集验带：%s' % (json.dumps(R['E5'], ensure_ascii=False)[:400] if R['E5'] else 'NONE'),
     '封存 P 族（方向按原文）返回：P4=%s P1=%s(ell0=%s) P2=%s(rho=%s) P3=%s' %
     (V['P4'], V['P1'], str(V['P1_ell0']), V['P2'], str(V['P2_spearman']), V['P3']),
   ],
   'descriptive': [
     '相对剂量剖面（E1b，amend1 A1 七点网格）类标签：%s' %
     '  '.join('%s:%s' % (tag(s), PR[str(s)]['cls']) for s in E1S),
     'E3 位点 U_l 与 U6 的子空间重叠：%s' %
     '  '.join('%s:%.4f' % (tag(k), R['E3'][k]['overlap']) for k in R['E3']),
     '地板：E4_max=%.4f，全臂 maxabs=%.3f，比=%.4f，F1_ok=%s' %
     (R['floors']['E4_max'], R['floors']['maxabs_all'], R['floors']['E4_max'] / max(R['floors']['maxabs_all'], 1e-9),
      R['floors']['F1_ok']),
     '离流形警告（pert_rel > 0.50 的 alpha）：%s' % R['off_manifold_alphas'],
     'n6 口径：本 Phase %.4f vs Phase 9 %.4f（drift=%s）' %
     (R['dose_coord']['mean_n6'], R['dose_coord']['mean_n6_ref_phase9'], R['dose_coord']['n6_drift']),
   ],
 },
 'honesty': [
   '本 Phase 输出的是「同一扰动注入不同深度」的因果剂量-响应剖面，不是模型内部轨迹，也不是权重级证明。',
   '跨层固定 U6 意味着深处注入方向未必是该层「自己的」类别轴；E3 自基臂只覆盖 %d 个位点。' % len(R['sites']['own_basis']),
   '绝对剂量与相对剂量两种坐标可能给出不同类结论（%s vs %s，%s）；两者并列报告，不择一。' %
   (V['V_abs'], V['V_rel'], V['V_rel'] and DEC['V_rel']),
   '深层残差范数与 ||u6|| 之比（r_ell）逐层变化，剖面不能完全排除「有效剂量随深度漂移」这一混淆；E1b 是控制手段，但 E1b 网格只到相对剂量 0.80。',
   'J 与 k_log 在低信噪比处不稳；x* 只在 R2_log 达标时报告。',
   'R 位点（最终 LayerNorm 之后）的扩展网格高端（alpha=8/16 对应相对剂量 1.37/2.73）已远超 0.5 离流形门限，该段只用于确认「是否存在拐点」，不用于定量加权。',
   '单模型 qwen3-4b、单模板 %s是一种、单语言、41 例全单 token；无 seed 噪声带（R1-P8 挂账）。',
   'U6 由 discovery 估计（与 Phase 8/9 同一对象），干预方向与该数据共享；确认集（n=17）不参与任何拟合，只用于验带。',
 ],
 'amend_disclosure': [
   'A1 E1b 网格 4 点 -> 7 点（与 E1 对称，使 V_rel 与 V_abs 的 J 噪声结构同阶）。',
   'A2 E1/E1b 位点加入 L6 输出作为剖面参照点（Phase 9 D1 站点），使剖面有零点。',
   'A3 新增方向修正的次级判据族 Q1/Q2/Q3；封存 P 族原样保留并同时报告（P2 的原文方向与物理设置相反，见 amend1 A3）。',
   'A4 代码层：r_ell 口径声明、Spearman 丢弃无效 J、新增剖面数组行。',
   'A5 新增 E6：R 位点扩展网格（r_R 只有深度位点的约 1/3，同网格下 UNREACH，否证探针会失效）。',
   '全部改动在正式运行前冻结；面板与所有封存阈值未动。',
 ],
 'meta': {'seal_sha8': seal_sha[:8], 'amend_sha8': amend_sha[:8], 'exec_sha8': exec_sha[:8],
          'result_sha8': res_sha[:8], 'report_sha8': rep_sha[:8],
          'smoke_dir': 'tests/deepseek_temp/Phase10/smoke/'},
}
jp = os.path.join(P10T, 'judgement_phase10.json')
io.open(jp, 'w', encoding='utf-8').write(json.dumps(J, ensure_ascii=False, indent=1))
w('judgement_phase10.json written (%d bytes)' % os.path.getsize(jp))

# ---------- 2. Ledger 补登（先备份）----------
bk = os.path.join(P10T, 'atlas_ledger_backup_pre_phase10.json')
shutil.copy2(LEDGER, bk)
b_sha = full(LEDGER)
L = json.load(io.open(LEDGER, encoding='utf-8'))
n0 = len(L['measurements'])
entry = {
 'phase': 10,
 'name': 'n2h1a3_soft_threshold_depth_profile_qwen3_4b',
 'seal_sha8': seal_sha[:8],
 'result_sha8': res_sha[:8],
 'evidence_level': 'statistical',
 'model_scope': 'qwen3-4b',
 'n_rows': (len(R['sites']['e1']) * 7 + len(R['sites']['e1']) * 7 + len(R['sites']['own_basis']) * 7 +
            len(R['sites']['floor']) * 3 + 7 + 1 + 1) * 24 + (len(R['sites']['conf']) * 7) * 17,
 'prereg_id': 'N2h1a3-amend1',
 'superseded_by': None,
 'verdict': 'deep_depth_profile_%s__readout_%s' % (str(V['Q_abs']).lower(), str(V['V_readout']).lower()),
 'rev_note': (
   'deepseek/N line Phase 10 (N2h1-alpha-3), qwen3-4b, same 41-instance panel inherited bit-for-bit from Phase 8 '
   '(discovery 24 / confirmation 17), same U6 (5-dim class-mean subspace at L6 output), same template. '
   'Design: replicate the Phase 9 D1 dose probe (h_recip + alpha*P_U6(diff6)) at every layer output from L6 to L34 '
   'plus a readout site R (after the final LayerNorm), i.e. one fixed perturbation injected at different depths. '
   'bit_anchored: E0 at S_L6out alpha=1 reproduces Phase 9 full = 10.574739583333335 bit-identically; F3 identity '
   'check at 4 sites incl. the norm hook = 0.000e+00. '
   'statistical: J(l) (max adjacent slope / median of the rest) profile = %s. '
   'Relative-dose half-saturation x*_rel(l) = %s. Saturation y_sat(l) = %s. '
   'amend1 A1-A5 were frozen before the formal run (E1b grid 4->7 points; L6 added as the profile reference point; '
   'orientation-corrected secondary rule family Q1/Q2/Q3 added because the sealed P2 direction is inverted relative '
   'to the physics; r_ell caveat; E6 extended grid at R). Sealed P family and every threshold unchanged. '
   'honesty: the probe is a fixed U6 direction at all depths, so deeper sites are not probed along their own class '
   'axis (E3 covers only %d sites); absolute and relative dose coordinates may disagree and both are reported; no seed '
   'noise band; single model/template/language; not a weight-level attribution.'
   % ('  '.join('%s:%.2f' % (k, v) for k, v in J_abs.items() if v is not None),
      '  '.join('%s:%s' % (k, ('%.3f' % v) if v is not None else 'n/a') for k, v in XR.items()),
      '  '.join('%s:%.3f' % (k, v) for k, v in YS_abs.items()),
      len(R['sites']['own_basis']))),
 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
}
L['measurements'].append(entry)
L.setdefault('migration_history', []).append({
 'phase': 10,
 'from_version': L.get('version'), 'to_version': L.get('version'),
 'backup': 'tests/deepseek_temp/Phase10/atlas_ledger_backup_pre_phase10.json',
 'backup_sha256_8': sha8(bk),
 'note': ('deepseek/N line backfill round 3 (continues the Phase 8 and Phase 9 backfills): appended Phase 10 '
          '(N2h1-alpha-3) measurement. ledger_sha256_8 remains NOT recomputed (recipe unknown; marked stale/unverified '
          'since Phase 8) => use per-file sha8 in this entry instead. pre-append file sha8 %s.' % b_sha[:8]),
})
io.open(LEDGER, 'w', encoding='utf-8').write(json.dumps(L, ensure_ascii=False, indent=1))
L2 = json.load(io.open(LEDGER, encoding='utf-8'))
w('ledger measurements %d -> %d ; file sha8 %s -> %s ; backup_sha8 %s' %
  (n0, len(L2['measurements']), b_sha[:8], full(LEDGER)[:8], sha8(bk)))
w('ledger tail verdict: %s' % L2['measurements'][-1]['verdict'])

# ---------- 3. 备忘录基线（pre-append 快照）----------
mb = open(MEMO, 'rb').read()
T = mb.decode('utf-8-sig')
lines = T.splitlines()
heads = {}
for i, l in enumerate(lines):
    if l.startswith('## '):
        heads[l[:44]] = i + 1
base = {'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'tag': 'pre-append',
        'path': 'research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
        'bytes': len(mb), 'lines': len(lines), 'sha256': hashlib.sha256(mb).hexdigest(),
        'bom': mb[:3] == b'\xef\xbb\xbf', 'crlf': mb.count(b'\r\n'),
        'bare_lf': mb.count(b'\n') - mb.count(b'\r\n'),
        'sections': heads,
        'note': ('Phase 10 追加前快照。历史事件：2026-10-01 外部进程曾把 38 处裸 LF 规范化为 CRLF 并插入 1 空行（+40B/+1 行）。'
                 '时钟事件：Phase 8 节标题 [22:05] 晚于其产物 mtime 21:34:50；本机时钟不可作因果排序依据。')}
io.open(os.path.join(P10T, 'memo_baseline_preappend_phase10.json'), 'w', encoding='utf-8').write(
    json.dumps(base, ensure_ascii=False, indent=1))
w('memo baseline(pre-append): bytes %d lines %d sha8 %s bare_lf %d' %
  (base['bytes'], base['lines'], base['sha256'][:8], base['bare_lf']))

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
print('DONE ->', OUT)
