# -*- coding: utf-8 -*-
"""Phase 9 收尾元数据：判决三级标签 + Ledger 补登（含备份与自哈希说明）+ 备忘录基线冻结。"""
import os, io, json, hashlib, shutil, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P9T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase9')
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
OUT = os.path.join(P9T, 'closeout_phase9.txt')

o = []
def w(s=''):
    o.append(str(s)); print(s)

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

def full(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()

R = json.load(io.open(os.path.join(P9T, 'result_phase9.json'), encoding='utf-8'))
res_sha = full(os.path.join(P9T, 'result_phase9.json'))
seal_sha = full(os.path.join(P9T, 'N2h1a2_design_seal.json'))
amend_sha = full(os.path.join(P9T, 'N2h1a2_design_seal_amend1.json'))
rep_sha = full(os.path.join(P9T, 'n2h1a2_report_qwen3-4b.txt'))
exec_sha = full(os.path.join(P9T, 'execution_phase9.json'))
w('result_sha8 %s ; seal_sha8 %s ; amend_sha8 %s ; exec_sha8 %s ; report_sha8 %s' %
  (res_sha[:8], seal_sha[:8], amend_sha[:8], exec_sha[:8], rep_sha[:8]))

D1 = R['D1']; D2 = R['D2']; amp = R['curves']['amp']
d1a = R['D1a']; D7 = R['D7']; d7d = R['D7_detail']
c1, c2 = R['curves']['D1'], R['curves']['D2']

# ---------- 1. 判决（三级标签）----------
J = {
 'phase': 9,
 'name': 'N2h1-alpha-2 threshold_gain_dose_response',
 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
 'prereg': {'seal_sha8': seal_sha[:8], 'amend1_sha8': amend_sha[:8],
            'frozen_verdict': R['verdict']['D1'] + ' / ' + R['verdict']['D2'] + ' / amp=' + R['verdict']['amp']},
 'verdict': R['verdict'],
 'headline': {
   'full_donor_replication': R['full'],
   'phase8_diff6_arm': 10.574739583333335,
   'replication_bitexact': bool(abs(R['full'] - 10.574739583333335) < 1e-12),
   'amp_min': amp['y'][0], 'amp_max': amp['y'][-1],
   'amp_dose_span_x': [amp['x'][0], amp['x'][-1]],
   'amp_rel_change_pct': 100.0 * (amp['y'][-1] - amp['y'][0]) / amp['y'][0],
   'amp_jump_ratio': amp['jump_ratio'], 'amp_pow_gamma': amp['gamma'],
   'amp_ref_1_over_rbar': amp['amp_ref'],
   'D1_jump_ratio': c1['jump_ratio'], 'D1_gamma': c1['pow']['gamma'], 'D1_R2_pow': c1['pow']['R2'],
   'D1_R2_lin': c1['lin']['R2'], 'D1_x_star': c1['x_star'],
   'D2_jump_ratio': c2['jump_ratio'], 'D2_gamma': c2['pow']['gamma'], 'D2_R2_pow': c2['pow']['R2'],
   'D2_R2_lin': c2['lin']['R2'], 'D2_x_star': c2['x_star'],
   'D2_y_at_natural_upstream': c2['y'][2],
   'D2_y_at_write_magnitude': c2['y'][6],
   'D7_kill_frac': d7d['kill_frac'], 'D7_dD_alpha0': d7d['dD_alpha0'], 'D7_ref_drop': d7d['ref_drop'],
   'D7_donor_rank1_at_identity': d7d['donor_rank1'],
   'rbar': R['dose_coord']['rbar'], 'rbar_phase8': R['dose_coord']['rbar_phase8'],
   'u5_u6_overlap': R['subspace']['overlap'],
   'floors_F1_ok': R['floors']['F1_ok'], 'floors_D4_max': R['floors']['D4_max'],
   'floor_matched': R['floor_matched'],
   'confirmation': R['confirmation'],
   'elapsed_s': R['elapsed_s'],
 },
 'evidence_levels': {
   'bit_anchored': [
     'D1@alpha=1 的 dDonor = %.15f 与 Phase 8 的 T[diff6].dDonor = 10.574739583333335 逐位相同 => U6 子空间与站点口径完全复现' % R['full'],
     'F3 恒等自检：两站点 alpha=0 注入的 |dScore| 均为 0.000e+00（钩子精确还原未干预前向）',
     'D7@alpha=1 恒等偏差 0.000e+00（供体句 + 零撤除 = 原前向）',
     'F4 o_proj 输入维 4096 = 32x128 断言通过；F5 面板逐元素继承断言通过（inherits_panel_sha256 = 45b4641a… == Phase 8 exec sha8）',
     'rbar 0.23676 vs Phase 8 的 0.23162：差 0.0051 < 容忍 0.02，drift=False（口径差见 honesty）',
   ],
   'statistical': [
     'amp 在 12.0x 剂量（x 0.118->1.421）上只从 %.4f 走到 %.4f（%+.1f%%），J=%.2f，幂律 gamma=%.3f（零次幂=常数）=> L6 内部无阈值增益' %
     (amp['y'][0], amp['y'][-1], 100.0 * (amp['y'][-1] - amp['y'][0]) / amp['y'][0], amp['jump_ratio'], amp['gamma']),
     'D1 陡段斜率 2.192 vs 其余斜率中位 0.404（J=%.2f >= 5）；D2 陡段 2.086 vs 中位 0.424（J=%.2f < 5）' % (c1['jump_ratio'], c2['jump_ratio']),
     'D7 必要性：撤除供体末位上游 U 分量，写入只损失 kill_frac=%.4f（dD(0)=%+.3f vs REF %+.3f），donor-class rank-1 比例 0.458->0.458 不变' %
     (d7d['kill_frac'], d7d['dD_alpha0'], d7d['ref_drop']),
     'D2 在自然上游量级 x=%.3f 只给 y=%.3f（与 Phase 8 的 2.7%% 同带）；在写向量量级 x=%.3f 给 y=%.3f（充分性）' %
     (c2['x'][2], c2['y'][2], c2['x'][6], c2['y'][6]),
     '确认集（n=17）：full_ratio=%.3f ; C1=%s(J=%.2f) C2=%s(J=%.2f) ; same_verdict_D1/D2=%s/%s' %
     (R['confirmation']['full_ratio'], R['confirmation']['C1']['cls'], R['confirmation']['C1']['J'],
      R['confirmation']['C2']['cls'], R['confirmation']['C2']['J'],
      R['confirmation']['same_verdict_D1'], R['confirmation']['same_verdict_D2']),
     '预注册判决树三级均为 H0_no_verdict：H1 差"跳变右端 y>=0.50"一条（实测 0.298），H2 差 R2_pow>=0.98（实测 0.9665），H3 差 R2_lin>=0.97（实测 0.9224）',
   ],
   'descriptive': [
     '两条行为曲线均为 S 形：半饱和点 x* = %.3f(D1) / %.3f(D2)，陡度 gamma 2.151 / 1.944，饱和值约 1.0' % (c1['x_star'], c2['x_star']),
     'F1 形式失败（D4_max %.3f = %.3f x 全臂最大），但同剂量地板比 0.098/0.070/0.169 与"随机 5 维方向在写方向上的几何投影"（E|cos|=3/8=0.375）的预测 0.077/0.1665 定量吻合 => 地板不是噪声，响应近似是"注入向量在 P_U6(diff6) 上投影"的函数',
     'U5 与 U6 主角 cos = 0.598/0.526/0.503/0.487/0.453，重叠 0.266 => L5/L6 类均值张的子空间大部分不同；D2b 换基后同带（+0.416/+4.088/+8.511 vs +0.377/+3.458/+10.104）',
     'D3 正交补（norm 约 18.2）在同剂量 alpha=1 给 -0.219（vs 写方向 +10.575）=> 写端特异性成立',
     'I_nl 重新归属：用 D1 曲线作传递函数 f，单头 share_v -> 预测 f(share)*full 为 head14 0.069/head11 0.067/head8 0.059/head15 0.055/head24 0.052，Phase 8 实测 0.037/0.055/0.047/0.032/0.031；MLP 预测 2.83 实测 1.141。要点：全部组件份额远低于 x*≈0.6 => I_nl>>1 是软阈值的算术后果',
   ],
 },
 'honesty': [
   '预注册判决树把 S 形判到 H0：H1 的"跳变区间右端 y>=0.50"隐含阶跃假设，真实曲线是两格陡的 S 形（J=5.41 达标、右端 0.298 未达标）。这是判据设计缺陷而非结论缺陷；本 Phase 不改判（改判=事后挑选），改为把曲线分类器参数化后留给 Phase 10 预注册。',
   'F1 形式失败（0.166 > 0.10 阈值）：按预注册，特异性/份额解释降级为描述性；但失败被几何投影模型定量解释（见 descriptive）。',
   'D1 只能把非线性定位到"L6 之后"，无法定位到 L7..L35 中的具体层。',
   'D2 高位点渐进离流形：alpha=4.32/6 的 pert_rel 0.462/0.642（后者越 0.50 阈值）；饱和结论主要由 x<=0.71 的点支撑。',
   'rbar 口径差：本 Phase 在 41 对上算 0.23676，Phase 8 在 24 个发现集对上算 0.23162；差 2.2% 未触发 drift，但引用时必须写口径。',
   'amp 测的是"注入扰动在 L6 输出上的 U6 投影比"，不是 L6 完整输入->输出算子；P_U6 同时定义干预与测量（自洽但不独立）。',
   '单模型 qwen3-4b、单模板 %s是一种、单语言、41 例全单 token；无 seed 噪声带（R1-P8 挂账）。',
   '注入/撤除均为流形外干预：剂量曲线是"该站点上的因果剂量-响应"，不是实现级证明。',
 ],
 'may_falsify': 'Phase 8 §7 的"L6 = 把上游种子放大到锁定量级的阈值增益"在 L6 内部被否证：amp 平坦（12x 剂量只变 6.6%），上游末位 U 分量不必要（kill 5.8%）。机制改写为"上下文驱动的写入 + L6 之后的软阈值（x*≈0.6）读出"。',
 'meta': {'seal_sha8': seal_sha[:8], 'amend_sha8': amend_sha[:8], 'exec_sha8': exec_sha[:8],
          'result_sha8': res_sha[:8], 'report_sha8': rep_sha[:8],
          'smoke_dir': 'tests/deepseek_temp/Phase9/smoke/'},
}
jp = os.path.join(P9T, 'judgement_phase9.json')
io.open(jp, 'w', encoding='utf-8').write(json.dumps(J, ensure_ascii=False, indent=1))
w('judgement_phase9.json written (%d bytes)' % os.path.getsize(jp))

# ---------- 2. Ledger 补登（先备份）----------
bk = os.path.join(P9T, 'atlas_ledger_backup_pre_phase9.json')
shutil.copy2(LEDGER, bk)
b_sha = full(LEDGER)
L = json.load(io.open(LEDGER, encoding='utf-8'))
n0 = len(L['measurements'])
entry = {
 'phase': 9,
 'name': 'n2h1a2_l6_is_not_a_threshold_gain_nonlinearity_after_l6',
 'seal_sha8': seal_sha[:8],
 'result_sha8': res_sha[:8],
 'evidence_level': 'statistical',
 'model_scope': 'qwen3-4b',
 'n_rows': (7 + 1 + 8 + 3 + 3 + 9 + 2 + 3) * 24 + (7 + 8 + 2) * 17,
 'prereg_id': 'N2h1a2-amend1',
 'superseded_by': None,
 'verdict': 'l6_has_no_threshold_gain_soft_threshold_lies_after_l6_upstream_component_sufficient_not_necessary',
 'rev_note': ('deepseek/N line Phase 9 (N2h1-alpha-2), qwen3-4b dose-response at S_L6out and S_L5out with panel inherited bit-for-bit from Phase 8 '
              '(inherits_panel_sha256 45b4641a). '
              'bit_anchored: D1(alpha=1) dDonor = 10.574739583333335 == Phase 8 T[diff6].dDonor, bit-identical; F3 identity check 0.000e+00 at both sites; D7(alpha=1) identity deviation 0. '
              'statistical: L6 self-gain amp(alpha) = ||P_U6(h6_after-h6_recip)||/(alpha*||P_U6(diff5)||) moves only 1.2624 -> 1.3458 (+6.6%) over a 12.0x dose span, jump ratio 1.42, '
              'power-law gamma 0.030 (constant) => NO threshold gain inside L6; reference 1/rbar = 4.224 is a norm ratio, not a gain. '
              'Behavioural curves are sigmoids: D1 jump ratio 5.41 (steep segment 2.192 vs median 0.404), gamma 2.151, R2_pow 0.9665; D2 jump 4.92 (steep 2.086 vs median 0.424); '
              'half-saturation x* = 0.625 (D1) / 0.592 (D2) in write-vector-fraction units. D2 at natural upstream magnitude (x=0.237) gives y=0.036 (same band as Phase 8 2.7%), at write magnitude (x=1.023) gives y=0.955 (sufficiency). '
              'D7 necessity: removing the donor last-position upstream U-component costs only kill_frac 0.0578 and leaves donor-class rank-1 fraction unchanged (0.458 -> 0.458). '
              'confirmation n=17: full_ratio 0.875, C1/C2 same verdict as discovery. '
              'descriptive: D4 random-in-U6 floor fails F1 formally (0.166) but is quantitatively predicted by the geometric projection of a random 5-dim direction onto the write direction (E|cos| = 3/8); '
              'D3 ortho-complement at matched norm gives -0.219 vs +10.575; U5/U6 subspace overlap only 0.266. '
              'Phase 8 I_nl = 6.85 is re-attributed: with the D1 sigmoid as the transfer function, per-component predictions f(share_v)*full match Phase 8 single-head effect arms within ~2x (higher), i.e. I_nl>>1 is an arithmetic consequence of a soft threshold at x*~0.6, not independent evidence of in-layer superadditivity. '
              'honesty: frozen decision tree returned H0 for all three curves (H1 missed only the "right-end y>=0.50" clause); F1 failed formally => specificity claims downgraded to descriptive; nonlinearity localised only to "after L6".'),
 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
}
L['measurements'].append(entry)
L.setdefault('migration_history', []).append({
 'phase': 9,
 'from_version': L.get('version'), 'to_version': L.get('version'),
 'backup': 'tests/deepseek_temp/Phase9/atlas_ledger_backup_pre_phase9.json',
 'backup_sha256_8': sha8(bk),
 'note': ('deepseek/N line backfill round 2 (continues the Phase 8 backfill): appended Phase 9 (N2h1-alpha-2) measurement. '
          'ledger_sha256_8 remains NOT recomputed (recipe unknown; marked stale/unverified since Phase 8) '
          '=> use per-file sha8 in this entry instead. pre-append file sha8 %s.' % b_sha[:8]),
})
io.open(LEDGER, 'w', encoding='utf-8').write(json.dumps(L, ensure_ascii=False, indent=1))
L2 = json.load(io.open(LEDGER, encoding='utf-8'))
w('ledger measurements %d -> %d ; file sha8 %s -> %s ; backup_sha8 %s' %
  (n0, len(L2['measurements']), b_sha[:8], full(LEDGER)[:8], sha8(bk)))
w('ledger tail verdict: %s' % L2['measurements'][-1]['verdict'])

# ---------- 3. 备忘录基线刷新（append 之后调用，见调用顺序说明）----------
def refresh_baseline(tag):
    mb = open(MEMO, 'rb').read()
    T = mb.decode('utf-8-sig')
    lines = T.splitlines()
    heads = {}
    for i, l in enumerate(lines):
        if l.startswith('## '):
            heads[l[:44]] = i + 1
    base = {'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'tag': tag,
            'path': 'research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
            'bytes': len(mb), 'lines': len(lines), 'sha256': hashlib.sha256(mb).hexdigest(),
            'bom': mb[:3] == b'\xef\xbb\xbf', 'crlf': mb.count(b'\r\n'),
            'bare_lf': mb.count(b'\n') - mb.count(b'\r\n'),
            'phase9_hdr_lines': [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase 9:')],
            'sections': heads,
            'note': ('Phase 9 后刷新。历史事件：2026-10-01 外部进程曾把 38 处裸 LF 规范化为 CRLF 并插入 1 空行（+40B/+1 行）。'
                     '时钟事件：Phase 8 节标题 [22:05] 晚于其产物 mtime 21:34:50，不可作因果排序依据。')}
    io.open(os.path.join(INFRA, 'memo_baseline.json'), 'w', encoding='utf-8').write(
        json.dumps(base, ensure_ascii=False, indent=1))
    return base

base_pre = refresh_baseline('pre-append')
io.open(os.path.join(P9T, 'memo_baseline_preappend_phase9.json'), 'w', encoding='utf-8').write(
    json.dumps(base_pre, ensure_ascii=False, indent=1))
w('memo baseline(pre-append): bytes %d lines %d sha8 %s bare_lf %d' %
  (base_pre['bytes'], base_pre['lines'], base_pre['sha256'][:8], base_pre['bare_lf']))

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
print('DONE ->', OUT)
