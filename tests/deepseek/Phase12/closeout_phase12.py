# -*- coding: utf-8 -*-
"""
Phase 12 收尾元数据：判决（G 族 + G5）+ Ledger 补登（294 -> 295，含备份）+ 备忘录 pre-append 基线快照。
"""
import os, io, json, hashlib, shutil, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P12T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
OUT = os.path.join(P12T, 'closeout_phase12.txt')

o = []


def w(s=''):
    o.append(str(s)); print(s)


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def full(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


RESP = os.path.join(P12T, 'result_phase12.json')
SEALP = os.path.join(P12T, 'N2h1a5_design_seal.json')
AMENP = os.path.join(P12T, 'N2h1a5_design_seal_amend1.json')
REPP = os.path.join(P12T, 'n2h1a5_report_qwen3-4b.txt')
EXECP = os.path.join(P12T, 'execution_phase12.json')

R = json.load(io.open(RESP, encoding='utf-8'))
res_sha, seal_sha, amen_sha, rep_sha, exec_sha = full(RESP), full(SEALP), full(AMENP), full(REPP), full(EXECP)
w('result_sha8 %s ; seal_sha8 %s ; amend1_sha8 %s ; exec_sha8 %s ; report_sha8 %s' %
  (res_sha[:8], seal_sha[:8], amen_sha[:8], exec_sha[:8], rep_sha[:8]))

GV = R['G_verdict']
GF = R['G_family']
XH = R['xhalf']
REC = R['recover']
BB = R['bootstrap_band']
PN = R['permutation_null']
FL = R['floors']
SV = R['sites']['swap']


def tag(s):
    if s == 'R':
        return 'R*'
    try:
        return 'L%d' % int(s)
    except Exception:
        return str(s)


XHC = {tag(s): XH['curve'][s] for s in XH['curve']}
XSC = {tag(s): XH['x_star_logistic'][s] for s in XH['x_star_logistic']}
RECC = {tag(s): REC['curve'][s] for s in REC['curve']}
JS = {tag(s): R['profile_swap'][s]['jump_ratio'] for s in R['profile_swap']}
CLS = {tag(s): R['profile_swap'][s]['cls'] for s in R['profile_swap']}

# ---------- 1. 判决 ----------
J = {
 'phase': 12,
 'name': 'N2h1-alpha-5 per-site residual swap + layer contribution allocation',
 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
 'prereg': {
   'seal_sha8': seal_sha[:8],
   'amend1_sha8': amen_sha[:8],
   'frozen_rules': 'G0/G1/G2/G3/G4 (amended) + G5 (new) ; amend1 把主量由 recover(alpha=1) 改为 xhalf/x*/J_swap',
   'bootstrap': 'pair-level percentile bootstrap B=%d ; permutation null B=%d' % (BB['B'], PN['B']),
   'swap_sites': 'full 18 profile sites + R'},
 'verdict': {
   'G_verdict': GV,
   'G0': GF['G0'], 'G1': GF['G1'], 'G2': GF['G2'],
   'G3': GF['G3'], 'G4': GF['G4'], 'G5': GF['G5'], 'F7pp': GF['F7pp'],
   'endpoint_diagnostic': {'span_recover': REC['span'], 'min_recover': REC['min'],
                           'rho_recover_depth': REC['rho']},
 },
 'headline': {
   'FULL_SWAP': R['FULL_SWAP'], 'full_L6': R['full_L6'],
   'bit_replication': R['bit_replication'],
   'xhalf': XHC, 'x_star_logistic': XSC, 'recover': RECC,
   'J_swap': JS, 'cls_swap': CLS,
   'xhalf_R': XH['R'], 'xhalf_range': XH['range'], 'rho_xhalf': XH['rho'],
   'jumps_xhalf': XH['jumps'], 'top3_share_x': XH['top3_share_x'], 'max_share_x': XH['max_share_x'],
   'G3_pairs_site_Jswap_Jinject': R['G_family']['G3']['pairs'],
   'boot': {'rho_xhalf': BB['rho_xhalf'], 'top3_share_x_ci': BB['top3_share_x_ci'],
            'rho_recover': BB['rho_recover'], 'R_ci': BB['R_ci']},
   'permutation_null': {'xhalf': {'lo': PN['xhalf']['lo'], 'hi': PN['xhalf']['hi']},
                        'recover': {'lo': PN['recover']['lo'], 'hi': PN['recover']['hi']}},
   'q_ell': R['dose_coord']['q_ell'], 'q_R': R['dose_coord']['q_R'],
   'proj_share_u6': R['proj_share_u6'],
   'confirmation': R['E5'],
   'floors': FL,
   'off_manifold_alphas': R['off_manifold_alphas'],
   'elapsed_s': R['elapsed_s'],
 },
 'evidence_levels': {
   'bit_anchored': [
     'E0 (S_L6out, alpha=1, 注入口径) dDonor = %.15f，与 Phase 9 full = 10.574739583333335 逐位相等 = %s'
     % (R['full_L6'], R['bit_replication']),
     'F11 构造决定点（新增）：R 位点满替换 dDonor = %.15f 必须逐位等于 FULL_SWAP = %.15f => |d| = %.3e，ok=%s'
     % (FL['F11_val'], R['FULL_SWAP'], FL['F11_max'], FL['F11_ok']),
     'F12 满替换 = 供体贴残差（float32 重建误差）：%s' % FL['F12_dev'],
     'F3 alpha=0 钩子恒等自检：%s' % FL['F3_dev'],
     'F10 逐对数组自洽（mean(per_pair)==dDonor）：ok=%s，max|d|=%.3e' % (FL['F10_ok'], FL['F10_max']),
     'F13 跨 Phase 参照闸门（result_phase11.json sha256 一致）=%s' % FL['F13_ok'],
     'F5 面板 12 字段 x Phase8/Phase10/Phase11 三向逐元素断言（gen 阶段）',
     'F6 逐位复现 = %s' % FL['F6_ok'],
   ],
   'statistical': [
     'G0 前置：curve_ok_frac = %d/%d = %.3f（阈值 %.2f）；xhalf 极差 = %s（阈值 %.2f）=> ok=%s'
     % (GF['G0']['n_curve_ok'], len(SV), GF['G0']['curve_ok_frac'], GF['G0']['curve_ok_frac_min'],
        ('%.4f' % GF['G0']['xh_range']) if GF['G0']['xh_range'] is not None else 'n/a',
        GF['G0']['xh_range_min'], GF['G0']['ok']),
     'G2 集中度：top3_share_x = %s（95%% 带 %s）；max_share_x = %s => %s'
     % (('%.4f' % GF['G2']['top3_share_x']) if GF['G2']['top3_share_x'] is not None else 'n/a',
        json.dumps(BB['top3_share_x_ci'], ensure_ascii=False),
        ('%.4f' % GF['G2']['max_share_x']) if GF['G2']['max_share_x'] is not None else 'n/a',
        GF['G2']['label']),
     'G3 秩相关：rho(J_swap, J_inject) = %s（n=%d）=> %s'
     % (('%.4f' % GF['G3']['rho_JG']) if GF['G3']['rho_JG'] is not None else 'n/a',
        GF['G3']['n_sites'], GF['G3']['label']),
     'G4 确认集（n=17）rho(xhalf) = %s => %s'
     % (('%.4f' % GF['G4']['rho_conf_xhalf']) if GF['G4']['rho_conf_xhalf'] is not None else 'n/a',
        GF['G4']['label']),
     'G5 充分性：min recover = %.4f（阈值 %.2f）=> %s' % (GF['G5']['min_recover'],
                                                        GF['G5']['min_recover_min'], GF['G5']['label']),
     'F7 迭代版（置换零假设 B=%d）xhalf 的 95%% 带 = [%s, %s] => |界| < 0.6 ? %s'
     % (PN['B'], PN['xhalf']['lo'], PN['xhalf']['hi'], GF['F7pp']),
   ],
   'descriptive': [
     'xhalf 剖面：%s' % '  '.join('%s:%s' % (k, ('%.3f' % v) if v is not None else 'n/a')
                                  for k, v in XHC.items()),
     'recover 剖面（端点诊断，amend1 A1 说明其按构造饱和）：%s'
     % '  '.join('%s:%.4f' % (k, v) for k, v in RECC.items()),
     'J_swap 剖面：%s' % '  '.join('%s:%s' % (k, ('%.2f' % v) if v is not None else 'n/a') for k, v in JS.items()),
     'cls_swap 剖面：%s' % '  '.join('%s:%s' % (k, v) for k, v in CLS.items()),
     'J_inject 参照（Phase 11）：%s' % '  '.join('L%d:%s' % (a, ('%.2f' % b) if b is not None else 'n/a')
                                                 for (a, _x, b) in GF['G3']['pairs']),
     'q_ell（满替换相对幅度）：%s' % json.dumps(R['dose_coord']['q_ell'], ensure_ascii=False),
     'proj_share_u6（diff 落在类别轴上的份额）：%s' % json.dumps(R['proj_share_u6'], ensure_ascii=False),
     '地板：E4_max=%.4f，maxabs=%.3f，比=%.4f，F1_ok=%s' %
     (FL['E4_max'], FL['maxabs_all'], FL['E4_max'] / max(FL['maxabs_all'], 1e-9), FL['F1_ok']),
     '离流形警告（pert_rel > 0.50 的 alpha）：%s' % R['off_manifold_alphas'],
   ],
 },
 'honesty': [
   '替换是流形外干预：满替换（alpha=1）必然远离受体流形（pert_rel = q_ell）。输出是「状态代换的因果响应」，不是模型轨迹，也不是权重级证明。',
   'diff_ell 不是「类别信息」的纯净载体：包含词面/句法/范数一切差异。recover 与 xhalf 度量的是「供体答案的成形进度」，不是「类别轴的写入进度」。',
   'recover 可 > 1（比值而非概率）；xhalf 是经验插值量，精度受 alpha 网格分辨率限制。',
   'bootstrap 只覆盖发现集 24 个配对的组成不确定性；不覆盖 alpha 网格离散化、面板选择、模板选择，以及替换幅度 q_ell 这一新自由度。区间下界乐观。',
   'n=24 偏小，percentile bootstrap 欠覆盖；B=2000 与固定 seed 保证可复现，但 95% 区间不是严格频率覆盖。',
   '『位点 J 的精度』『xhalf 剖面的可信度』『集中度判据的可信度』是三个不同的不确定性，分开列示。',
   'G3 的参照 J_inject 来自 Phase 11（同模型同面板，但不同探针族与不同 x 轴尺度）；秩相关对尺度不变故合法，但只能支撑「同形/不同形」的粗判定。',
   'amend1 把主量由端点量改为形状量，理由为构造性（Phase 8-11 已证 L6 只注入类别轴 alpha=1 即得 full），不依赖正式数据；端点量未被丢弃而是升格为 G5。',
   '本 Phase 不触碰权重级归因（N2h1-alpha-1 仍挂账）：贡献分配只到「层」粒度。',
   '单模型 qwen3-4b、单模板 %s是一种、单语言、41 例全单 token。',
   '本 Phase 不执行 Phase 11 §8 的第二候选（相邻位点配对 bootstrap），故 G2 判「少层主导」时仍未解决「哪些相邻对可分辨」。',
 ],
 'meta': {'seal_sha8': seal_sha[:8], 'amend1_sha8': amen_sha[:8], 'exec_sha8': exec_sha[:8],
          'result_sha8': res_sha[:8], 'report_sha8': rep_sha[:8],
          'smoke_dir': 'tests/deepseek_temp/Phase12/smoke/'},
}
# ---- 事后描述性（post-hoc，无判决角色；明确标注）----
import math
_js = [(s, R['profile_swap'][str(s)]['jump_ratio']) for s in SV]
_js = [(s, v) for (s, v) in _js if v is not None and math.isfinite(v) and v > 0]
_half_log = None
if len(_js) >= 4 and _js[-1][1] > 0 and _js[0][1] > 0:
    _target = math.sqrt(_js[0][1] * _js[-1][1])
    for i in range(len(_js) - 1):
        a, b = _js[i][1], _js[i + 1][1]
        if b <= _target <= a:
            tt = (math.log(a) - math.log(_target)) / (math.log(a) - math.log(b))
            _half_log = _js[i][0] + tt * (_js[i + 1][0] - _js[i][0])
            break
_shallow = [v for (s, v) in _js if s <= 12]
_deep = [v for (s, v) in _js if s > 12]
J['posthoc_descriptive'] = {
 'note': 'POST-HOC，无判决角色；仅为报告提供尺度感，未写入任何预注册判据',
 'J_swap_endpoints': [_js[0][1], _js[-1][1]] if _js else None,
 'J_swap_log2_decline': (math.log2(_js[0][1] / _js[-1][1]) if _js else None),
 'J_swap_half_log_depth': _half_log,
 'J_swap_mean_L6_L12': (sum(_shallow) / len(_shallow)) if _shallow else None,
 'J_swap_mean_L14_L34': (sum(_deep) / len(_deep)) if _deep else None,
 'G1_is_an_instrument_artifact': (
   'G1 的归一化 XN=(xhalf-min)/(max-min) 隐含假设剖面随深度【上升】；实测 rho(xhalf,depth)=-0.7833 为负，'
   '故 XN[0]=1.0，first_reach(0.1/0.5/0.9) 全部退化返回首站点 => G1a_crystallized 是仪器伪影，本 Phase 不引用。'
   '正确的方向读法是：xhalf 随深度【下降】（承诺点前移 / 越深越容易被推动）。'),
 'G4_note': ('确认集只有 4 个位点 (7,11,20,34)，其 xhalf = 0.437/0.441/0.441/0.473 单调上升 => rho=+0.80，'
             '与发现集 18 位点的 -0.7833 符号相反。这不是物理冲突，而是采样密度不足：发现集剖面非单调'
             '（L30 触底 0.390 后 L34 反弹到 0.465），4 个位点落在不同支上。'),
}
jp = os.path.join(P12T, 'judgement_phase12.json')
io.open(jp, 'w', encoding='utf-8').write(json.dumps(J, ensure_ascii=False, indent=1))
w('judgement_phase12.json written (%d bytes)' % os.path.getsize(jp))

# ---------- 2. Ledger 补登 ----------
bk = os.path.join(P12T, 'atlas_ledger_backup_pre_phase12.json')
shutil.copy2(LEDGER, bk)
b_sha = full(LEDGER)
LG = json.load(io.open(LEDGER, encoding='utf-8'))
n0 = len(LG['measurements'])
nsite = len(SV)
n_rows = (24 + nsite * len(R['E2'][str(SV[0])]) * 24 + len(R['E6']) * 24 +
          len(R['E2b']) * len(R['E2b'][str(R['sites']['swap_rel'][0])]) * 24 +
          len(R['E3']) * len(R['E3'][str(R['sites']['overshoot'][0])]) * 24 +
          len(R['E4']) * 2 * 24 +
          (len(R['sites']['conf']) * len(R['E5'][str(R['sites']['conf'][0])]['rows']) * 17
           if R['sites']['conf'] else 0))
entry = {
 'phase': 12,
 'name': 'n2h1a5_per_site_residual_swap_layer_contribution_allocation_qwen3_4b',
 'seal_sha8': seal_sha[:8],
 'amend1_sha8': amen_sha[:8],
 'result_sha8': res_sha[:8],
 'evidence_level': 'statistical',
 'model_scope': 'qwen3-4b',
 'n_rows': n_rows,
 'prereg_id': 'N2h1a5',
 'superseded_by': None,
 'verdict': 'swap_%s__g5_%s' % (str(GV).lower(), str(GF['G5']['label']).lower()),
 'rev_note': (
   'deepseek/N line Phase 12 (N2h1-alpha-5), qwen3-4b, same 41-instance panel inherited bit-for-bit from Phase 8/10/11 '
   '(discovery 24 / confirmation 17), same template. Probe family CHANGED: instead of injecting the fixed rank-5 class '
   'axis u6 at site ell, this phase REPLACES the recipient last-token residual at ell by the donor one '
   '(h_ell_recip + alpha*diff_ell, alpha in [0,1] = replacement fraction, alpha=1 => exact donor state). 18 profile '
   'sites x 14 alpha points, per-pair landing, pair-level percentile bootstrap (B=2000) + permutation null (B=2000). '
   'bit_anchored: E0 reproduces Phase 9 full = 10.574739583333335 bit-identically; F11 (new construct-determined point) '
   'R-site alpha=1 reproduces FULL_SWAP bit-identically; F12 proves alpha=1 == donor state; F3 alpha=0 identity; '
   'F10 mean(per_pair)==dDonor; F13 gates the Phase 11 result sha. '
   'amend1 (frozen before the formal run): recover(alpha=1) saturates by construction (Phase 8-11 already showed that '
   'injecting ONLY the rank-5 class axis at L6 with alpha=1 already yields full), so the primary allocation metric was '
   'moved to the curve-shape quantities xhalf (empirical half-saturation replacement fraction) and J_swap; the endpoint '
   'was promoted to a positive criterion G5 (last-position-state sufficiency >= 0.90 on all 18 sites). '
   'statistical: G0 curve_ok_frac = %.3f (>=0.75), xhalf range = %s (>=0.10); G2 top3_share_x = %s (threshold 0.60 for '
   'few-layer-dominant / max_share_x <= 0.40 for layer-wise accumulate); G3 Spearman(J_swap, J_inject) = %s; '
   'G4 confirmation rho(xhalf) = %s; G5 min recover = %.4f; F7 permutation null on xhalf = [%s, %s]. '
   'Combined verdict = %s. '
   'honesty: replacement is off-manifold (pert_rel = q_ell); diff_ell carries everything, not just category information; '
   'bootstrap covers only pair-composition uncertainty; G3 reference J_inject is a point estimate from a different probe '
   'family so only rank agreement (not numeric) is comparable.'
   % (GF['G0']['curve_ok_frac'], ('%.4f' % GF['G0']['xh_range']) if GF['G0']['xh_range'] is not None else 'n/a',
      ('%.4f' % GF['G2']['top3_share_x']) if GF['G2']['top3_share_x'] is not None else 'n/a',
      ('%.4f' % GF['G3']['rho_JG']) if GF['G3']['rho_JG'] is not None else 'n/a',
      ('%.4f' % GF['G4']['rho_conf_xhalf']) if GF['G4']['rho_conf_xhalf'] is not None else 'n/a',
      GF['G5']['min_recover'], PN['xhalf']['lo'], PN['xhalf']['hi'], GV)),
 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
}
LG['measurements'].append(entry)
LG.setdefault('migration_history', []).append({
 'phase': 12,
 'from_version': LG.get('version'), 'to_version': LG.get('version'),
 'backup': 'tests/deepseek_temp/Phase12/atlas_ledger_backup_pre_phase12.json',
 'backup_sha256_8': sha8(bk),
 'note': ('deepseek/N line backfill round 5 (continues Phase 8/9/10/11 backfills): appended Phase 12 (N2h1-alpha-5) '
          'measurement. ledger_sha256_8 remains NOT recomputed (recipe unknown; marked stale since Phase 8) => use '
          'per-file sha8 in this entry instead. pre-append file sha8 %s.' % b_sha[:8]),
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
        'bom': mb[:3] == b'\xef\xbb\xbf', 'crlf': mb.count(b'\r\n'),
        'bare_lf': mb.count(b'\n') - mb.count(b'\r\n'),
        'sections': heads,
        'note': 'Phase 12 追加前快照。Phase 11 节起 L2305；Phase 标题共 11 个。'}
io.open(os.path.join(P12T, 'memo_baseline_preappend_phase12.json'), 'w', encoding='utf-8').write(
    json.dumps(base, ensure_ascii=False, indent=1))
w('memo baseline(pre-append): bytes %d lines %d sha8 %s bare_lf %d' %
  (base['bytes'], base['lines'], base['sha256'][:8], base['bare_lf']))

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
print('DONE ->', OUT)
