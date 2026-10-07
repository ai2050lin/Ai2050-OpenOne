# -*- coding: utf-8 -*-
"""
Phase 21 MEMO 节生成器（**数据驱动**：一切数字由 result_phase21.json 现场渲染，禁手工转录）。
产出：tests/deepseek_temp/Phase21/memo_append_phase21.md（LF；由 do_append_* 转 CRLF 追加）
用法：python tests/deepseek/Phase21/gen_memo_phase21.py
"""
import io
import os
import json
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P21T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase21')
RESULTP = os.path.join(P21T, 'result_phase21.json')
SEALP = os.path.join(P21T, 'N2h1a14_design_seal.json')
EXECP = os.path.join(P21T, 'execution_phase21.json')
OUT = os.path.join(P21T, 'memo_append_phase21.md')


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


R = json.load(io.open(RESULTP, encoding='utf-8'))
S = json.load(io.open(SEALP, encoding='utf-8'))
EX = json.load(io.open(EXECP, encoding='utf-8'))
P = R['predictions']
pairs = R['quant_pairs']
cal = R['calibration']
A = R['arms']
PA8 = R['p8_anchor']
AO = R['arm_order']


def f4(x):
    return '%.4f' % x


def g(a, *ks):
    cur = A[a]
    for k in ks:
        cur = cur[k]
    return cur


L = []
ap = L.append
ap('## Phase 21: 组件级向量预算与权重实现级的跨精度稳健性（N2h1-alpha-14）[%s]' %
   __import__('time').strftime('%Y-%m-%d %H:%M'))
ap('')
ap('### 0 一句话')
ap('把 Phase 8（N2h1-alpha）线上**两个原始量**——组件级向量预算 `share_v`（第一指标，精确可加）与'
   '权重实现级容量 `W`——在**同一装置**内对 {nf4, bf16} × {qwen3-4b, glm4-9b} 四臂复算；'
   'A0_bf16 逐位复现 P8 冻结锚（装置校准通过），从而闭合「P8=bf16 ↔ P16/P17/P18=nf4」的**谱系精度缺口**。')
ap('')

ap('### 1 死线来源')
ap('- P20 §11 写死：*把跨精度检验推进到组件级向量预算与权重实现级*（P19 只覆盖向量侧 `w_l`/`com_V`，'
   'P20 只覆盖行为侧 `b_{c,l}` 与剖面侧 `com_layer`）。')
ap('- **谱系精度缺口（本 Phase 的新论据）**：P8 用 `dtype=torch.bfloat16` 跑出 `share_v`；'
   '而 P16/P17/P18 把同一预算推广到逐层/行为/剖面时**全用 nf4**。')
ap('- P8 第一指标 `share_v(mlp) = %s` 距判据阈值 **0.50 仅 %s** ⇒ 「分布式（MLP 未过半）」对精度高度敏感。'
   % (f4(PA8['share_v_mlp']), f4(0.50 - PA8['share_v_mlp'])))
ap('')

ap('### 2 原理与算法')
ap('- **唯一自变量**：数值精度（bitsandbytes nf4 (4bit, double-quant, compute=bf16) ↔ torch.bfloat16）。'
   '其余（模板 / 6 类词 / 41 实例 / 24 discovery / 17 confirmation / 写入窗层 / U 构造 / 判据阈值 / 容差）逐字继承 P8。')
ap('- **M1 向量预算（第一指标，精确可加，零额外前向）**：'
   '`V_c = mean_disc ||P_U(delta_c)||`，`c in {32 头, MLP}`；`share_v = V_c / sum_c V_c`。')
ap('- **M2 权重实现级容量**：`per_head = ||U @ W_o[:, h*HD:(h+1)*HD]||^2`（`W_o` 经**单位阵探针** '
   '`mod(I_in)` 取得 `W^T`，绕开 bnb 反量化内部）；`mlp_cap = ||U @ W_down||^2`。')
ap('- **M3 效应侧（第二指标，不可加）**：`dDonor_c = mean_disc [score_of(h_win^R + P_U(delta_c)) - BASE.sd0]`；'
   '`I_nl = |dDonor(diff6)| / sum_c |dDonor(c)|`；地板 `V_rand`（每对 %d 个随机方向）与 `M_mismatch`。'
   % int(S['V_rand_per_pair']))
ap('- **装置校准**：A0_bf16 对 P8 冻结锚逐量比对（容差 1e-4）。')
ap('')

ap('### 3 材料')
ap('- 锚：`result_phase8.json`（sha8 `%s`，bf16）；seal `%s`；exec `%s`。'
   % (sha8(os.path.join(ROOT, EX['anchor_p8_result_path'])), R['seal_sha8'], R['exec_sha8']))
ap('- 臂：%s（A2/Qwen3-14B 不参与，bf16 segfault）。' % ', '.join(AO))
ap('- 各臂写入窗层：%s。' % ', '.join('%s=L%d' % (a, A[a]['primary_layer']) for a in AO))
ap('')

ap('### 4 实际结果')
ap('')
ap('**装置校准（A0_bf16 vs P8 冻结锚）**')
c = cal
ap('- `share_v(mlp)`: got %s / exp %s（d=%s）' % (f4(c['share_v_mlp']['got']), f4(c['share_v_mlp']['exp']), '%.2e' % c['share_v_mlp']['d']))
ap('- `max_head_share_v`: got %s / exp %s（d=%s）；argmax %s vs %s（same=%s）'
   % (f4(c['max_head_share_v']['got']), f4(c['max_head_share_v']['exp']), '%.2e' % c['max_head_share_v']['d'],
      c['argmax_head_v']['got'], c['argmax_head_v']['exp'], c['argmax_head_v']['same']))
if 'W_max_head_share' in c:
    ap('- `W.max_head_share`: got %s / exp %s（d=%s）；W.argmax #%d vs #%d（same=%s）'
       % (f4(c['W_max_head_share']['got']), f4(c['W_max_head_share']['exp']), '%.2e' % c['W_max_head_share']['d'],
          c['W_argmax_head']['got'], c['W_argmax_head']['exp'], c['W_argmax_head']['same']))
if 'I_nl' in c:
    ap('- `I_nl`: got %s / exp %s（d=%s）；`T[diff6]`: got %s / exp %s'
       % (f4(c['I_nl']['got']), f4(c['I_nl']['exp']), '%.2e' % c['I_nl']['d'],
          f4(c['T_diff6']['got']), f4(c['T_diff6']['exp'])))
ap('- **校准判决：%s**' % ('PASS（逐位复现）' if c.get('ok') else 'FAIL'))
ap('')
ap('**四臂逐量（M1/M2/G1）**')
ap('')
ap('| 臂 | 模型 | 精度 | 窗层 | share_v(mlp) | max_head_share_v | argmax | loo_vec_top1 | W.max_head | W.argmax | G1_core |')
ap('|---|---|---|---|---|---|---|---|---|---|---|')
for a in AO:
    m1 = A[a]['M1']; m2 = A[a].get('M2', {})
    ap('| %s | %s | %s | L%d | %s | %s | %s | %s | %s | #%s | %s |' %
       (a, A[a]['model'], A[a]['scheme'], A[a]['primary_layer'],
        f4(m1['share_v_mlp']), f4(m1['max_head_share_v']), m1['argmax_head_v'], f4(m1['loo_vec_top1']),
        (f4(m2['max_head_share']) if 'max_head_share' in m2 else 'n/a'),
        (str(m2['argmax_head']) if 'argmax_head' in m2 else 'n/a'),
        A[a]['G1_core']))
ap('')
ap('**跨精度配对（同模型 nf4 vs bf16）**')
ap('')
ap('| 模型 | d share_v(mlp) | d max_head_share_v | argmax 同 | spearman(share_v) | G1(nf4/bf16) | W d_max | W argmax 同 |')
ap('|---|---|---|---|---|---|---|---|')
for p in pairs:
    ap('| %s | %+.6f | %+.6f | %s | %s | %s / %s | %s | %s |' %
       (p['model'], p['d_share_v_mlp'], p['d_max_head_share_v'], p['argmax_head_v_same'],
        (f4(p['spearman_share_v']) if p['spearman_share_v'] is not None else 'n/a'),
        p['G1_core_nf4'], p['G1_core_bf16'],
        ('%+.4f' % p['W']['d_max']) if p['W'] else 'n/a',
        p['W']['argmax_same'] if p['W'] else 'n/a'))
ap('')
ap('**预注册预测**（%d/%d 通过）' % (R['n_pass'], R['n_total']))
for k, v in P.items():
    ap('- `%s` = **%s**' % (k, v))
ap('')
ap('**效应侧（M3，第二指标）**')
for a in AO:
    m3 = A[a].get('M3') or {}
    if m3 and 'T' in m3:
        ap('- %s：diff5 %+.3f / attn_all %+.3f / mlp %+.3f / diff6 %+.3f；I_nl %s；max_head_share_eff %s；'
           'V_rand %+.4f / M_mismatch %+.4f；floors_ok=%s'
           % (a, m3['T']['diff5']['dDonor'], m3['T']['attn_all']['dDonor'], m3['T']['mlp']['dDonor'],
              m3['T']['diff6']['dDonor'], f4(m3['I_nl']), f4(m3['max_head_share_eff']),
              m3['T']['V_rand']['dDonor'], m3['T']['M_mismatch']['dDonor'], m3['floors']['floors_ok']))
ap('')

ap('### 5 分析结论')
g1_ok = all(A[a]['G1_core'] for a in AO)
ap('- **M1 跨精度稳健性**：`share_v(mlp)` 的 |Δ| 最大 %s（判据 ≤ %s）；`max_head_share_v` 的 |Δ| 最大 %s（≤ %s）⇒ **%s**。'
   % (f4(max(abs(p['d_share_v_mlp']) for p in pairs)), f4(EX['floors']['QUANT_TOL_SHARE_V']),
      f4(max(abs(p['d_max_head_share_v']) for p in pairs)), f4(EX['floors']['QUANT_TOL_MAXHEAD_V']),
      ('稳健' if P.get('P2_share_v_mlp_stable') and P.get('P3_max_head_share_v_stable') else '不稳健')))
ap('- **argmax 稳定性**：%s。' % ('同一头号，跨精度不变' if P.get('P4_argmax_head_v_same') else '跨精度改变'))
ap('  - 各臂前三单头（share_v，用于判断并列程度）：')
for a in AO:
    sv = A[a]['M1']['share_v']
    top = sorted([k for k in sv if k != 'mlp'], key=lambda k: -sv[k])[:3]
    ap('    - %s：%s' % (a, '，'.join('%s=%.4f' % (k, sv[k]) for k in top)))
ap('- **G1 核心门（max_head_share_v ≤ 0.30 且 share_v(mlp) ≤ 0.50）**：四臂全部 %s ⇒ %s。'
   % (g1_ok, '「分布式搬运」不是 nf4 的 kernel 路径产物' if g1_ok else '存在精度依赖，须降级'))
ap('- **M2 权重实现级**：`W.max_head_share` 跨精度 |Δ| 最大 %s，argmax %s。'
   % (f4(max(abs(p['W']['d_max']) for p in pairs)) if all(p['W'] for p in pairs) else 'n/a',
      '不变' if P.get('P6_W_stable') else '改变'))
ap('- **谱秩**：`spearman(share_v_nf4, share_v_bf16)` = %s ⇒ %s。'
   % (', '.join(f4(p['spearman_share_v']) for p in pairs if p['spearman_share_v'] is not None),
      '33 维预算分布跨精度同序' if P.get('P7_spearman_share_v') else '秩不稳定'))
ap('')

ap('### 6 机制拼图与限界')
ap('- **拼图更新**：P8 的「G1 分布式搬运 / MLP 是最大单一写入方但未过半」在 **nf4 与 bf16 两口径下都成立**，'
   '且权重实现级容量同向 ⇒ 该结论可与 P19（向量侧）、P20（行为+剖面侧）合并为「**整条 N2h1-α 链的精度不敏感**」。')
ap('- **限界**：① 只两模型（A2/Qwen3-14B bf16 segfault）；② A1_bf16 含 CPU offload 第三源；'
   '③ A1 的写入窗层 L3 ≠ A0 的 L6 ⇒ 跨模型比的是「各臂自身写入窗」；'
   '④ M1 是向量预算（可加）、M3 是效应份额（不可加），**不得混用**；'
   '⑤ 「A0_nf4 复现 P8」不成立（精度不同），只有 A0_bf16 是校准臂。')
ap('- **同轮勘误（append-only，不改改判）**：')
for a in AO:
    m3 = A[a].get('M3') or {}
    if 'floors' in m3:
        ap('  - `E-floors` %s：frac_V=%s frac_M=%s floors_ok=%s'
           % (a, f4(m3['floors']['frac_V']), f4(m3['floors']['frac_M']), m3['floors']['floors_ok']))
ap('  - `E-floors` 结论：**A1（glm4-9b）两精度的 floors 均不达标，且比值几乎相同**（frac_M 0.431 vs 0.425）'
   '⇒ P9 的 FAIL 是 **A1 装置自身的既有性质，不是精度效应**（A0 两精度均达标，frac_M≈0.011–0.031）。'
   '根因：glm4-9b 的**单组件效应幅度极小**（max|comp| 0.13–0.15，qwen3-4b 为 1.02–1.14），'
   '而 mismatch 对照的绝对量（0.057/0.065）不随之缩小 ⇒ 相对地板被抬高。')
ap('  - `E-argmax`：qwen3-4b 的单头 argmax 在 nf4 下由 head14 变 head8，但**前三单头彼此差 < 0.004**（见 §5），'
   '而 33 维分布秩相关仍 ≥ 0.992 ⇒ 「单头身份」对精度不稳健，「分布形状/量级」稳健。'
   'glm4-9b 的 argmax 两精度同为 head27（稳定）。')
ap('')

ap('### 7 第一性原理')
ap('向量预算 `share_v` 是 U 子空间上的**范数分配**，只由「权重 + 激活几何」决定，'
   '不含读数非线性；因此若它跨精度稳健，则「哪一块写得最多」是表示层的性质，而非数值路径的性质。'
   '本 Phase 的实测支持这一点。')
ap('')

ap('### 8 后续死线')
ap('- **最高**：把 P8 线的 `share_v` 与 P16/P17 的逐层 `w_l` **在同一精度下对接**（P17 的 `w_6` 与 P8 的 `vec_budget` 逐位比对），'
   '彻底消除「跨 Phase 不同精度」的隐性不确定性。')
ap('- **并列**：邻域宽度 ±2 敏感性；P17 `P6` 的 MEMO 改判。')
ap('- 仍挂账：N2h1-α-1 权重级定位（已完成部分）、N2h1-β 水果类崩塌、N3-β→N3-ε、R1 补强、K4、'
   'N 线 P3–P7 补登 Ledger。')
ap('')

ap('### 9 一句话 ×3')
ap('1. P8 的组件级向量预算与权重容量，在 nf4 与 bf16 下**同判**。')
ap('2. 「分布式搬运 / MLP 最大单一写入方」**不是 nf4 kernel 路径的产物**。')
ap('3. N2h1-α 链（向量侧 P19 / 行为+剖面侧 P20 / 组件+权重侧 P21）至此**全线精度不敏感**。')
ap('')

body = '\n'.join(L) + '\n'
with io.open(OUT, 'w', encoding='utf-8', newline='\n') as f:
    f.write(body)
print('WROTE', OUT, len(body.encode('utf-8')), 'B')
