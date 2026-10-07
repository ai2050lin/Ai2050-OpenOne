# -*- coding: utf-8 -*-
"""Phase 14 文档收尾：当日 wlog 追加 + _infra/memo_baseline.json 刷新（含 history 链）。

铁律 (w)：wlog 正文中的所有数字均从 result_phase14.json / Ledger / MEMO 现场取值渲染，
不做任何手工转录。
"""
import os
import io
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P14T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase14')
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-02.md')
OUT = os.path.join(P14T, 'closeout_docs_phase14.txt')

o = []
def w(s=''):
    o.append(str(s)); print(s)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


R = json.load(io.open(os.path.join(P14T, 'result_phase14.json'), encoding='utf-8'))
JUD = json.load(io.open(os.path.join(P14T, 'judgement_phase14.json'), encoding='utf-8'))
LG = json.load(io.open(LEDGER, encoding='utf-8'))
PRE = json.load(io.open(os.path.join(P14T, 'memo_baseline_preappend_phase14.json'), encoding='utf-8'))
AM1 = json.load(io.open(os.path.join(P14T, 'N2h1a7_design_seal_amend1.json'), encoding='utf-8'))
AM2 = json.load(io.open(os.path.join(P14T, 'N2h1a7_design_seal_amend2.json'), encoding='utf-8'))

V = R['verdict']; A8V = R['A8_verdict']
A6 = R['A6_concentration']; C_A1, C_A8 = A6['A1'], A6['A8']
B_A1, B_A8 = C_A1['bootstrap'], C_A8['bootstrap']
N_A1, N_A8 = C_A1['null'], C_A8['null']
P_A1, P_A8 = C_A1['paired'], C_A8['paired']
PR = R['predictions_check']; FL = R['floors']; PS = R['A3_position_summary']; PD = R['A3b_position_endpoints']
H = JUD['headline']

mb = open(MEMO, 'rb').read()
mt = mb.decode('utf-8-sig').splitlines()
new_sha8 = hashlib.sha256(mb).hexdigest()[:8]
ph_lines = [i + 1 for i, l in enumerate(mt) if l.startswith('## Phase ')]
p14_line = [i + 1 for i, l in enumerate(mt) if l.startswith('## Phase 14')]
tail = LG['measurements'][-1]

# ---------- 1. 当日 wlog 追加 ----------
NEW = not os.path.exists(WLOG)
b0 = open(WLOG, 'rb').read() if not NEW else b''
t0 = b0.decode('utf-8')
if NEW:
    t0 = '# 2026-10-02\n'

L = []
def p(s):
    L.append(s)

p('')
p('## Phase 14 / N2h1-α-7：逐层累积代换 + 双坐标集中度 —— 位置通道近乎为空（字面第三口径不成立）、正确剂量轴是「逐层支撑」（A8）、两坐标 argmax 跨族 %s（%s）'
  % (V['verdict_cross_family'], time.strftime('%H:%M')))
p('')
p('- **死线执行**：Phase 13 §8 最高优先 —— 「逐层累积代换（prefix swap）+ 双坐标集中度报告」，判据必须写成「`(share, argmax_window)` 二元组在两坐标上的一致性」。')
p('- **两条 SMOKE 拦截的装置缺陷（均在任何数据产生前冻结为修正案）**：')
p('  - **amend1（`%s`）= schema_amend，不改设计**：可行性探针把 `head_dim` 写成 `hidden/n_heads`（=80，**错**）；qwen3-4b 是 **GQA**（`head_dim=%s`、`kv_heads=%s`、`o_proj.in_features=%s`），装置前置 drift 断言 `[o_proj_in]` 首次触发。原 seal 字节保留，新增 amend1 承载地面真值。新装置坑 **#47**。'
  % (JUD['prereg']['amend1_sha8'], R['layers']['head_dim'], R['layers']['n_kv_heads'], R['layers']['o_proj_in']))
p('  - **amend2（`%s`）= schema_amend + 加臂，不改假设**：**面板级恒等式 vs 逐对恒等式的作用域混淆** —— 臂内均值（SMOKE 6 对子集）除以 24 对分母 `FULL_SWAP` 必然得 1.147991，是**子集偏置**不是装置缺陷。改为与子集无关的**逐对恒等式** `F30a`（dev `%.3e`，n=%d）+ 明确只在满面板断言的 `F30b`；并冻结 `FULL_SWAP` 的 **24 项口径**。'
  % (JUD['prereg']['amend2_sha8'], A8V['F30a_dev'], R['extra']['F30a_pairs_checked']))
p('  - **同一次 SMOKE 的科学收获**：`median(y0/y1) = %s`、A8 支撑 `i=0` 的 `xhalf` 与 Phase 12 单点 `mask={1}` 值**逐位相等** ⇒ 死线字面意义的「位置前缀」与 Phase 12 单点族**数值重合**，不是独立第三口径 ⇒ amend2 新增 **A8 逐层累积层代换**承担第三口径，A1 降级为**阴性对照**，并新增 P6/P7。' % PS['median_ratio'])
p('- **装置锚（全部 BIT-EXACT，`G0p = %s`）**：F24 `FULL_SWAP` 重建 = `%.15f`（`bit_equal = %s`）；F25 `mean‖P_U6(diff6)‖ = %.12f`（dev `%.3e`）；F26 U6 五奇异值 `max rel dev = %.3e`；F27 41/41 实例 `T = %s`；F28 α=0 全位点 patch `max|dScore| = %.3e`；F35 `max_ℓ |q_ℓ − Phase 12| = %.3e`。'
  % (V['G0p'], R['A0a_full_swap']['rebuilt'], R['A0a_full_swap']['bit_equal'], R['A0b_n6']['mean_n6'],
     R['A0b_n6']['dev'], R['A0c_u6']['dev'], R['A0e_tokenizer']['distinct_T'], R['A0d_noop']['dev'],
     FL['F35']['dev']))
p('- **主结果 1（死线的字面读法是空的）**：T=2 模板下「整段前缀」端点 α=1 ∧ mask={0,1} **按构造**重建整段残差 ⇒ `y01` 是面板级常数（F30b dev `%s`）。更关键的是**位置通道本身近乎为空**：`median(y0/y1) = %s`（pos0 只贡献约 `%.2f%%` 的效应），且 A1 与 Phase 12 单点族的 18 位点剖面对照 **`max|dxhalf| = %.6f`、`max|dJ| = %.6f`** ⇒ **两者是同一族**，不构成独立第三口径。位置判决 `%s`、可加性 `%s`。'
  % (FL['F30']['F30b']['dev'], PS['median_ratio'], 100.0 * float(PS['median_ratio']),
     H['A1_negative_control_position_prefix']['vs_phase12_single_point']['max_abs_d_xhalf'],
     H['A1_negative_control_position_prefix']['vs_phase12_single_point']['max_abs_d_J'],
     PS['verdict_position'], PS['verdict_additivity']))
p('- **主结果 2（正确的剂量轴 = 逐层支撑，A8 臂）**：在**每个已累积层位** `S_i = sites[0..i]` 的末位同时注入，随支撑扩大：形状量单调变软，而端点近饱和（端点曲线单调 `%s`，`y(i=0) = %.6f`）—— 端点饱和是**构造性**的（铁律 r）。**双坐标集中度（A8 主臂）**：`share_x = %.6f` @ 窗口 `%s`、`share_j = %.6f` @ 窗口 `%s`；Phase 13 靶 `mode_x = %s`、`mode_j = %s` ⇒ 同坐标判决 **`%s`**、跨族判决 **`%s`**。阴性对照 A1：`share_x = %.6f` @ `%s`、`share_j = %.6f` @ `%s`。'
  % (PR.get('P6', {}).get('monotone'), PR.get('P6', {}).get('y_at_i0', float('nan')),
     C_A8['top3_x'], C_A8['argmax_w_x'], C_A8['top3_j'], C_A8['argmax_w_j'],
     R['inherits']['MODE_X_13'], R['inherits']['MODE_J_13'],
     V['verdict_same_coordinate'], V['verdict_cross_family'],
     C_A1['top3_x'], C_A1['argmax_w_x'], C_A1['top3_j'], C_A1['argmax_w_j']))
p('- **误差带与零假设校准（铁律 p）**：A8 `P(share≥0.60)` 在 `xhalf` = `%s`、在 `J` = `%s`；置换 95 分位 `null_x = %s`、`null_j = %s`（A8）；A1 相邻位点配对 Δ：`N_dec_J = %s/%s`、`N_dec_X = %s/%s`。'
  % (B_A8.get('P_ge_060_x'), B_A8.get('P_ge_060_j'), N_A8.get('null_x_95'), N_A8.get('null_j_95'),
     P_A1['N_dec_J'], P_A1['n_pairs'], P_A1['N_dec_X'], P_A1['n_pairs']))
p('- **网格与统计量替代（A7）**：`XH_RANGE_p(legacy,%s 点) = %s`、`XH_RANGE_p(dense,%s 点) = %s`、Phase 12 `XH_RANGE = %s`；跨族秩一致 `spearman(J_p, J_swap) = %.4f`、`spearman(xhalf_p, xhalf_12) = %.4f`。'
  % (len(R['dose_coord']['alpha_legacy']), R['A7_range_grid']['legacy']['range'],
     len(R['dose_coord']['alpha_dense']), R['A7_range_grid']['dense']['range'], R['inherits']['XH_RANGE_12'],
     R['A7_steepness_alt']['rho_Jp_vs_Jswap'], R['A7_steepness_alt']['rho_xhalfp_vs_xhalf12']))
p('- **确认集与地板**：确认集面板 %s 对（发现集 %s 对）；A4 `%s`；A5 随机 5 维地板 `%s`。'
  % (R['panel']['confirmation'], R['panel']['discovery'],
     json.dumps(R['A4_confirmation'], ensure_ascii=False)[:160],
     json.dumps(R['A5_floor'], ensure_ascii=False)[:160]))
p('- **预注册预测**：%s。' % ' '.join('%s = %s' % (k, PR[k]['pass_']) for k in sorted(PR)))
p('- **判决**：同坐标 `%s`；跨族 `%s`（第三口径 = A8 逐层累积）；位置 `%s`；可加性 `%s`。'
  % (V['verdict_same_coordinate'], V['verdict_cross_family'], V['verdict_position'], V['verdict_additivity']))
p('- **记录**：deepseek 备忘录新增 `## Phase 14` 节（**L%s** 起），%d → **%d B** / %d → **%d 行**（前缀逐字节未变、BOM/CRLF、`bare_lf 0`、Phase 标题 **%d** 个）；Ledger 补登 N 线第 7 条（296 → **%d**，备份 `atlas_ledger_backup_pre_phase14.json`，verdict `%s`）。'
  % (p14_line[0] if p14_line else '?', PRE['bytes'], len(mb), PRE['lines'], len(mt), len(ph_lines),
     len(LG['measurements']), tail['verdict']))
p('- **新增铁律 (v)(w)(x)(y)**：(v) **「另一个独立口径」必须在 seal 冻结前用 SMOKE 证明其与既有口径**可分**；重合即降级为阴性对照，并另找剂量轴**（本 Phase 两个候选读法都在 SMOKE 阶段就被量出退化，否则会把与 Phase 12 数值重合的臂当独立证据发表）。(w) **面板级恒等式与逐对恒等式的作用域必须分离** —— `y = dDonor_arm / 固定面板均值` 使子集臂**不可能**满足面板级端点恒等式（SMOKE 6 对得 `1.147991` 是子集偏置的算术后果），须写成与子集无关的逐对形式 + 显式 `full_panel` 才断言的旁路。(x) **预注册预测的符号必须与其自身 `rationale` / `falsified_if` 一致** —— P4 的 `desc` 写 `y0 > y1`、`rationale`/`falsified_if` 却都指向 `y1 > y0` ⇒ 一条已被数据满足的预测（18/18、17/18）被机械判成 FAIL。(y) **GQA 模型禁用 `hidden_size / n_heads` 反推 `head_dim`** —— `o_proj.in_features = n_heads × head_dim` 与 `hidden_size` 无必然关系（qwen3-4b：`4096 ≠ 2560`，真值 `head_dim = 128`），配置字段一律直读 `config` 并与投影维度交叉断言。')
p('- **下一步（死线）**：**Phase 15 最高优先 = 跨模型复算「支撑-形状」关系**（qwen3-14b 与 glm4-9b untied 独立复算 `xhalf_A8(i)`/`J_A8(i)` 与两坐标 argmax；判据先冻结，**禁止沿用 L6/U6**）。第二候选：位置通道的量级上界（多模板、多位置、含 3 位置模板）。第三候选：`J` 与 `xhalf` 的联合判据（两坐标 argmax 距离 + 两坐标 top3_share 合成二维判决量 + 零假设分布）。')
p('- **过程备注（非实验内容）**：可行性探针 → seal → SMOKE 前置化 三段拦截了两条真实缺陷（GQA 配置字段 / 端点恒等式作用域），**均在正式运行前修复**；`MEMORY.md` 首次做整编压缩（12,887 → 11,621 B）。')
p('')

sec = '\n'.join(L)
# wlog 与 MEMO 同惯例：CRLF（Phase 14 复核发现本行原用 '\n' ⇒ 混合 EOL，lf 99 / crlf 10）
_sec_n = sec.replace('\r\n', '\n').replace('\n', '\r\n').strip('\r\n')
t1 = t0.rstrip('\r\n') + '\r\n\r\n' + _sec_n + '\r\n'
open(WLOG, 'wb').write(t1.encode('utf-8'))
b1 = open(WLOG, 'rb').read()
w('wlog(%s): bytes %d -> %d (+%d) ; lines %d -> %d' %
  ('new' if NEW else 'append', len(b0), len(b1), len(b1) - len(b0),
   len(b0.split(b'\n')), len(b1.split(b'\n'))))
w('wlog sha256 = %s' % hashlib.sha256(b1).hexdigest())

# ---------- 2. _infra/memo_baseline.json 刷新（带 history 链） ----------
heads = {}
for i, l in enumerate(mt):
    if l.startswith('## '):
        heads[l[:44]] = i + 1
hist = []
bp = os.path.join(P14T, 'memo_baseline_preappend_phase14.json')
if os.path.exists(bp):
    prev = json.load(io.open(bp, encoding='utf-8'))
    hist.append({'tag': 'pre-append-phase14', 'bytes': prev['bytes'], 'lines': prev['lines'],
                 'sha256': prev['sha256']})
old = os.path.join(INFRA, 'memo_baseline.json')
if os.path.exists(old):
    try:
        oh = json.load(io.open(old, encoding='utf-8')).get('history') or []
        for e in oh:
            if e.get('tag') not in [h['tag'] for h in hist]:
                hist.append(e)
    except Exception as e:
        w('warn: old history unreadable: %r' % (e,))
    try:
        ob = json.load(io.open(old, encoding='utf-8'))
        if ob.get('tag') not in [h['tag'] for h in hist]:
            hist.append({'tag': ob.get('tag'), 'bytes': ob.get('bytes'), 'lines': ob.get('lines'),
                         'sha256': ob.get('sha256')})
    except Exception:
        pass
base = {'frozen_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'tag': 'post-append-phase14',
        'path': 'research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
        'bytes': len(mb), 'lines': len(mt), 'sha256': hashlib.sha256(mb).hexdigest(),
        'sha8': new_sha8,
        'bom': mb[:3] == b'\xef\xbb\xbf', 'crlf': mb.count(b'\r\n'),
        'bare_lf': mb.count(b'\n') - mb.count(b'\r\n'),
        'phase_headings': ph_lines,
        'sections': heads, 'history': hist}
io.open(old, 'w', encoding='utf-8').write(json.dumps(base, ensure_ascii=False, indent=1))
w('memo baseline(post-append): bytes %d lines %d sha8 %s bare_lf %d phase_headings=%d history=%d' %
  (base['bytes'], base['lines'], new_sha8, base['bare_lf'], len(base['phase_headings']), len(hist)))

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o) + '\n')
print('DONE ->', OUT)
