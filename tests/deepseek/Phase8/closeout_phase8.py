# -*- coding: utf-8 -*-
"""Phase 8 收尾元数据：判决三级标签 + Ledger 补登（含备份与自哈希说明）+ 备忘录基线冻结。"""
import os, io, json, hashlib, shutil, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P8T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase8')
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
OUT = os.path.join(P8T, 'closeout_phase8.txt')

o = []
def w(s=''):
    o.append(str(s)); print(s)

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

def full(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()

R = json.load(io.open(os.path.join(P8T, 'result_phase8.json'), encoding='utf-8'))
res_sha = full(os.path.join(P8T, 'result_phase8.json'))
seal_sha = full(os.path.join(P8T, 'N2h1a_design_seal.json'))
amend_sha = full(os.path.join(P8T, 'N2h1a_design_seal_amend1.json'))
rep_sha = full(os.path.join(P8T, 'n2h1a_report_qwen3-4b.txt'))
w('result_sha8 %s ; seal_sha8 %s ; amend_sha8 %s ; report_sha8 %s' %
  (res_sha[:8], seal_sha[:8], amend_sha[:8], rep_sha[:8]))

a1 = R['amend1']
# ---------- 1. 判决（三级标签）----------
J = {
 'phase': 8, 'name': 'N2h1-alpha write_operator_weight_level_localization',
 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
 'verdict': R['verdict'],
 'gate': R['gates'],
 'primary_metric': 'share_v（向量预算，精确可加）',
 'headline': {
   'max_head_share_v': a1['max_head_share_v'], 'argmax_head_v': a1['argmax_head_v'],
   'mlp_share_v': a1['share_v']['mlp'],
   'max_head_share_eff': a1['max_head_share_eff'],
   'upstream_share_of_write_vector': None,
   'I_nl_effect_superadditivity': a1['I_nl'],
   'max_head_share_T_effect': R['max_head_share_T'], 'mlp_share_T_effect': R['shares_T']['mlp'],
   'max_head_share_Z': max(v for k, v in R['shares_Z'].items() if k.startswith('head')),
   'mlp_share_Z': R['shares_Z']['mlp'],
   'verification_jump_layer': R['jump_layer'], 'verification_flag': R['jump_flag'],
   'confirmation': R['confirmation'],
 },
 'evidence_levels': {
   'bit_anchored': [
     '写入窗定位：B_cat 曲线相邻最大增量 @L6 = +10.159（与冻结主层一致，flag=OK），L5=+0.42 -> L6=+10.57',
     '向量恒等式 P_U(diff6)=P_U(diff5)+sum_h P_U(delta_a6_h)+P_U(delta_m6)（proj 线性，逐位可复算）',
     'o_proj 输入维 4096 = 32x128（F4 断言通过）；捕获形状 (37,2560)/(4096,)/(2560,) 无 NaN',
     'S_self 对照解析为 0；driver 哈希 seal/exec 与 result 记录一致',
   ],
   'statistical': [
     'share_v 数值：MLP %.4f / 最大单头 %.4f（公平份额 0.0303）' % (a1['share_v']['mlp'], a1['max_head_share_v']),
     '效率份额最大 %.4f；效应份额 MLP %.3f / 最大单头 %.3f' % (a1['max_head_share_eff'], R['shares_T']['mlp'], R['max_head_share_T']),
     '门裕度：MLP share_v 距 0.50 阈值仅 %.4f；最大单头距 0.30 阈值 %.4f' % (0.50 - a1['share_v']['mlp'], 0.30 - a1['max_head_share_v']),
     '确认集（n=17）同带：最大单头 share_v=%.4f ; MLP=%.4f（无需噪声带）' % (R['confirmation']['max_head_share_v'], R['confirmation']['mlp_share_v']),
   ],
   'descriptive': [
     'MLP 是 33 件划分中最大的单一写入方（share_v %.1f%%），但未过半 ⇒ 不是"单组件主导"' % (100 * a1['share_v']['mlp']),
     '写入窗是阈值型/超可加的层：I_nl ≈ %.2f（效应和 2.93 vs 全量 10.58）' % a1['I_nl'],
     '上游残差已承载约 23% 的写入向量却只产生 2.7% 的效应 ⇒ 层在做"放大/锁定"而非"新建"该轴',
   ],
 },
 'honesty': [
   'share_v 用 L1 范数：范数对加法不成立（sum||.||=18.37 vs ||diff6||=18.09，1.5% 三角不等式余项），故"预算"是范式口径不是精确分配。',
   'MLP 距 0.50 阈值仅 0.028 ⇒ 若阈值取 0.45 则门翻转为"MLP 主导"；结论的稳健部分只有"无单头主导"（最大头 0.074，公平份额 3.03% 的 2.4 倍）。',
   '效率份额按 ||P_U(delta)|| 归一，方向特权检验在头间最大仅 0.111，但该检验对范数估计噪声敏感（面板 n=24，无 seed 带）。',
   '单模型（qwen3-4b）、单模板（%s是一种）、单语言（中文）、全单 token 实例；跨模型/跨模板未测。',
   '零消融与注入均在流形外：share 是因果归因的预算分解，不是"模型内部真的这么算"的实现级证明。',
 ],
 'meta': {'seal_sha8': seal_sha[:8], 'amend_sha8': amend_sha[:8], 'result_sha8': res_sha[:8],
          'report_sha8': rep_sha[:8], 'smoke_dir': 'tests/deepseek_temp/Phase8/smoke/'},
}
io.open(os.path.join(P8T, 'judgement_phase8.json'), 'w', encoding='utf-8').write(
    json.dumps(J, ensure_ascii=False, indent=1))
w('judgement_phase8.json written (%d bytes)' % os.path.getsize(os.path.join(P8T, 'judgement_phase8.json')))

# ---------- 2. Ledger 补登（先备份）----------
bk = os.path.join(P8T, 'atlas_ledger_backup_pre_phase8.json')
shutil.copy2(LEDGER, bk)
b_sha = full(LEDGER)
L = json.load(io.open(LEDGER, encoding='utf-8'))
n0 = len(L['measurements'])
entry = {
 'phase': 8,
 'name': 'n2h1a_write_operator_weight_level_localization',
 'seal_sha8': seal_sha[:8],
 'result_sha8': res_sha[:8],
 'evidence_level': 'statistical',
 'model_scope': 'qwen3-4b',
 'n_rows': 24 * 34 * 2 + 41,
 'prereg_id': 'N2h1a-amend1',
 'superseded_by': None,
 'verdict': 'n2h1a_writing_is_distributed_mlp_largest_single_writer',
 'rev_note': ('deepseek/N line Phase 8 (N2h1-alpha), qwen3-4b L6 write-operator component budget. '
              'bit_anchored: write window re-derived by adjacent-max-increment @L6 (+10.159, frozen layer confirmed); '
              'exact vector identity P_U(diff6)=P_U(diff5)+sum_h P_U(delta_a6_h)+P_U(delta_m6). '
              'statistical: vector-budget shares over {32 heads, MLP}: MLP 0.4717, max head 0.0742 (fair share 0.0303), '
              'max head efficiency share 0.1110; gate G1 (<=0.30 head, <=0.50 MLP, <=0.30 eff) -> distributed; '
              'confirmation panel n=17 same band (max head 0.0765 / MLP 0.4511). '
              'descriptive: effect arms are superadditive (I_nl 6.85; sum 2.93 vs diff6 10.58) => write window is threshold-like; '
              'upstream residual already carries 23% of the write vector but only 2.7% of the effect => layer amplifies/locks, does not create the axis. '
              'amend1 (pre-formal-run) moved the primary metric from effect shares to vector budget because SMOKE showed effect shares are non-additive. '
              'honesty: MLP margin to 0.50 gate is 0.028 (robust part = no single-head dominance); single model/template/panel.'),
 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
}
L['measurements'].append(entry)
L.setdefault('migration_history', []).append({
 'phase': 8,
 'from_version': L.get('version'), 'to_version': L.get('version'),
 'backup': 'tests/deepseek_temp/Phase8/atlas_ledger_backup_pre_phase8.json',
 'backup_sha256_8': sha8(bk),
 'note': ('deepseek/N line backfill round 1/1 (R1 挂账项 ⑥ 的执行起点): appended Phase 8 (N2h1-alpha) measurement. '
          'ledger_sha256_8 field NOT recomputed (recipe unknown; stored b76823a6 already did not match file sha8 0d01c25d before this append) '
          '=> treat ledger_sha256_8 as stale/unverified; use per-file sha8 in this entry instead. '
          'pre-append file sha8 %s, post-append recomputed after write.' % b_sha[:8]),
})
io.open(LEDGER, 'w', encoding='utf-8').write(json.dumps(L, ensure_ascii=False, indent=1))
L2 = json.load(io.open(LEDGER, encoding='utf-8'))
w('ledger measurements %d -> %d ; file sha8 %s -> %s ; backup_sha8 %s' %
  (n0, len(L2['measurements']), b_sha[:8], full(LEDGER)[:8], sha8(bk)))
w('ledger tail verdict: %s' % L2['measurements'][-1]['verdict'])

# ---------- 3. 备忘录基线冻结（防外部进程再动）----------
mb = open(MEMO, 'rb').read()
T = mb.decode('utf-8-sig')
lines = T.splitlines()
heads = {}
for i, l in enumerate(lines):
    if l.startswith('## '):
        heads[l[:40]] = i + 1
base = {'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'path': 'research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
        'bytes': len(mb), 'lines': len(lines), 'sha256': hashlib.sha256(mb).hexdigest(),
        'bom': mb[:3] == b'\xef\xbb\xbf', 'crlf': mb.count(b'\r\n'),
        'bare_lf': mb.count(b'\n') - mb.count(b'\r\n'),
        'sections': heads,
        'note': ('2026-10-01 外部进程曾在两次复查间把 38 处裸 LF 规范化为 CRLF 并插入 1 个空行（+40B/+1 行，v2 节标题 L1567->L1568），'
                 '内容无损失。此后以本基线为准核对是否再被改动。')}
io.open(os.path.join(INFRA, 'memo_baseline.json'), 'w', encoding='utf-8').write(
    json.dumps(base, ensure_ascii=False, indent=1))
w('memo baseline: bytes %d lines %d sha8 %s crlf %d bare_lf %d' %
  (base['bytes'], base['lines'], base['sha256'][:8], base['crlf'], base['bare_lf']))

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
print('DONE ->', OUT)
