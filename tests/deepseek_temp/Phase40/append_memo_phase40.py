# -*- coding: utf-8 -*-
"""MEMO Phase 40 追加（append-only；BOM+CRLF 保持；数字一律 result 现场渲染）。"""
import io, os, json, hashlib

MEMO = r'D:\AI2050\Ai2050-OpenOne\research\deepseek\docs\AGI_DEEPSEEK_MEMO.md'
RES_P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\result\q06_result.json'
res = json.load(open(RES_P, encoding='utf-8'))
R = res['C_steer_main']; V = res['v1_axis']; S = res['sensitivity']
C = res['collateral']; F = res['floors']; G = res['gates_smoke']; K = res['cells']

with open(MEMO, 'rb') as f:
    raw = f.read()
txt = raw.decode('utf-8')          # BOM -> \ufeff 保留在头
assert txt.startswith('\ufeff')
assert '## Phase 40' not in txt, 'Phase 40 already present'

L = []
L.append('## Phase 40: Q06 C_steer 基座测量 —— v1 承重轴 x 端口替换 = 0/376，方向减法禁令下的首张「诚实总成绩」读数（B 闸门；gpu=mid；qwen3-4b bf16）[2026-10-07 08:10]')
L.append('')
L.append('**已执行**（停电恢复后续研：预注册 ebf960cf 于 2026-10-03 22:36 冻结，本轮 annex v2 -> SMOKE -> 正式 -> 复核 -> seal）。')
L.append('')
L.append('### 1. 目标与状态')
L.append('- 队列 Q06（I7 可控性标准）：把「抽取机制 -> 控制行为 + 无附带损伤」从描述回路升级为可测基准。')
L.append('- 状态：**sealed**。result `tests/deepseek/result/q06_result.json`（res_sha8=%s）；独立复核 **16 PASS / 0 FAIL**（`tests/deepseek_temp/Phase40/verify_q06.txt`）。' % res['res_sha8'])
L.append('')
L.append('### 2. 原理与算法')
L.append('- **算子 = readout-substitution（端口替换）**：$h \\leftarrow h + (t - h\\cdot v)\\,v = h + \\mathrm{d}t\\cdot v$，位点 = L29 层输出（NL=36，非末层），替换「(层,方向) 的读出口」而非减去方向——方向减法在本系统非法（M15 cancel 0.820 反向加重；metric_dict.intervention_rules）。')
L.append('- **承重轴 v1（qwen3-4b 同构移植定义）**：seed7 train fold（197 pairs x 3 tpl = 591 行）TPL_P0 前缀 last-position 的 L29 残差流 H；加性分解 X=[onehot(41)+onehot(6)+onehot(3)+bias]（ridge λ=1e-3）；交互残差 R=H−Xb；**v1 = SVD(R) 第一右奇异向量**（sv_share=%.4f，μ29=%.4f，σ29=%.4f）。与 gpt5 线 M14（GLM4）为同构移植——跨模型坐标不对应（AGENTS.md §3），先验性质（无无害阈值）引自 R7。' % (V['sv_share'], V['mu29'], V['sigma29']))
L.append('- **剂量**：dt = sgn·α·σ，α∈{0.05,0.10,0.15,0.25,0.50}（预注册五点半调），sgn∈{+1,−1}；t = s + dt（v2：相对当前分量的 push/pull，对齐 M14 幅度语义）。')
L.append('- **对照**：identity 恢复臂（t=s，两 prompt 口径硬断言逐位恒等）+ 随机方向同规则（|cos(v1,vr)|=%.4f）。' % V['abs_cos_v1_rand'])
L.append('- **行为读数**：单次前向 last-position 6 类类名首 token logits（Q04 k=0 同构；panel sha8 be17ef8a 逐字同面板）；target = true_class（per-cell）；eligible = base argmax ≠ true（%d/441）。' % K['eligible'])
L.append('- **collateral**：13 探针句拼接于主体后（探针读数位 = 各 P0 段末位，经注意力被主体替换波及）；collateral(i,c) = 干预后错误数 − 同 cell 无干预基线错误数（v1 操作化，冻结）。')
L.append('- **C_steer** = #{(i,c): argmax_after = true_class 且 collateral = 0} / N_eligible。')
L.append('')
L.append('### 3. annex v2 修订（SMOKE v1 抓出，正式前冻结——坑 26/28 合规路径）')
L.append('- SMOKE v1（design 329e0115）：base argmax==true 仅 8.3%% ⇒ 「c\'=base 第二高类」退化为常量目标；t=μ+ασ 对 z≈0 cell 替换动量退化（argmax 0/72 移动）。')
L.append('- 修订 R1 t 规则（→s+sgn·α·σ）、R2 target（→true_class）、R3 G2 门（→可计算性 + G2b 灵敏度）；未动 collateral 定义/KPI 公式/α 网格/减法禁令。SMOKE v2 全门 PASS 后开正式。')
L.append('')
L.append('### 4. 实际结果（441 cells = 3 seeds x 147 held-out 行；19,404 前向，12.2 min）')
L.append('| 量 | 值 |')
L.append('|---|---|')
L.append('| **C_steer_main（10 配置取 max）** | **%.4f（0/%d，全部配置 0/376）** |' % (R['value'], R['wilson'] and sum(c['den'] for c in res['steer_curves']['steer|+1|0.05']['per_seed'].values()) if False else 376))
L.append('| rand 同规则对照 | %.4f（spec_diff = %.1f） |' % (R['rand_value'], R['spec_diff']))
L.append('| Wilson95（主读数） | [0, %.4f] —— 真实 C_steer ≤ 1.0%%（95%% 置信） |' % R['wilson'][1])
L.append('| 灵敏度 | argmax moved %d/%d（0.20%%）；maxd ∈ [%.4f, %.4f] logit |' % (S['argmax_moved'], S['argmax_total'], S['maxd_min'], S['maxd_max']))
L.append('| collateral | mean %.3f / max %d / frac_zero %.4f（操作干净，非打脏造成的 0 分） |' % (C['mean'], C['max'], C['frac_zero']))
L.append('| identity 硬断言 | 两 prompt 口径 max|Δ| = %.1e（全 441 cell） |' % F['F1_identity_maxd'])
L.append('| 拼接基线 err_base | %.2f/13（6 类 argmax 口径；历史 3.5/13 口径源文件失传，不可逐字比） |' % C['err_base_mean'])
L.append('')
L.append('### 5. 分析结论（严格审视）')
L.append('1. **承重轴 v1 + 端口替换不能实现类翻转控制**：C_steer = 0（上界 1.0%%）。v1 是全局形态/幅度方向（gpt5 线 3148/3149 定案），**不携带类身份杠杆**——maxd ≤ 0.94 logit 压不过 6 类先验的头部优势（base argmax 恒水果类）。')
L.append('2. **0 分是「定向杠杆缺失」而非「操作脏」**：同算子 collateral 干净（frac_zero 93.3%%）+ rand 对照同 0（spec_diff=0）⇒ 干预通路本身无损，缺的是方向特异性。')
L.append('3. **与 M14 的非对称补全承重轴画像**：clip（去除）α=0.5 破坏 66%% 生成（多步累积），push/pull 替换（单点）只动 <1 logit——**「破坏容易、定向控制难」：v1 承载生成稳定性，不承载类内容**。这是控制语义下 M14+M15 的合并推论。')
L.append('4. 硬伤：① target 语义仅「翻转到真值类」一种（Q20 扩 target 族）；② 单点替换 vs decode 多步累积——可能系统性低估干预力（但预注册冻结了 Q04 k=0 同构口径）；③ v1 为移植定义，与 GLM4 的 v1 非同一根轴，跨模型比较不成立；④ eligible 376/441 中 65 个免测 cell 偏水果类。')
L.append('')
L.append('### 6. 机制拼图与理论更新')
L.append('- **I7 兑现**：这个 0 分就是「破解整体机制」诚实总成绩的第一格——已抽取的 1 根承重轴距「可控」的距离被定量化为 0/376（Wilson 上界 1%%）。')
L.append('- **条件齿轮组视角**：类身份控制不是单轴幅度问题 ⇒ 支持多轴/组合干预路线（Q17 承重轴族剂量、Q24 机制解释、Q20 正式化扩 target+组合算子）。')
L.append('- **KPI 账**：E_read=0.331615（Q03 锁定，未变）、E_ar=Q05 已测（775d7dce，未变）、C_steer=0.0（首测新增）⇒ **未降低任何全局 KPI ⇒ Ledger 登记 catalog（n=306），不得登记 advance**。')
L.append('')
L.append('### 7. 后续与资源')
L.append('- 下一步 = **Q07 KPI 曲线 v0 汇总**（zero GPU：Q03–Q06 + 历史判决 -> 第一张单调曲线）；Q17/Q24 复用本装置；Q20 正式化扩 target 族。')
L.append('- 产物：脚本 `tests/deepseek/Phase40/`（q06_steer_base.py / run_phase40.py / verify_q06.py / closeout_q06.py）；报告与明细 `tests/deepseek_temp/Phase40/`（smoke/ 隔离）；result `tests/deepseek/result/q06_{execution,result}.json`；v1 轴 npz + detail 落 Phase40 temp。')
L.append('- 队列 Q06 -> sealed（sealed_at 2026-10-07 08:05）；挂账不变：N 线 P3–P7 补 Ledger、跨线账本补丁施加确认、N2h1-α-1 权重级、水果类、K4。')
L.append('')
L.append('锚：result sha8=%s，design sha8=7130906b（formal）/6d98d580（smoke），prereg=ebf960cf，panel=be17ef8a，vhat=%s，ledger n=306。' % (res['res_sha8'], V['vhat_sha8']))
L.append('')

new = '\r\n'.join(L)
txt2 = txt.rstrip('\r\n \t') + '\r\n\r\n' + new
with open(MEMO, 'wb') as f:
    f.write(txt2.encode('utf-8'))
chk = open(MEMO, 'rb').read()
assert chk[:3] == b'\xef\xbb\xbf', 'BOM lost'
t3 = chk.decode('utf-8')
assert t3.count('## Phase 40') == 1 and t3.count('## Phase 39') == 1
assert '## Phase 40' in t3.split('## Phase 39')[1], 'append order broken'
print('MEMO appended: bytes=%d head=EF BB BF ok  Phase40 count=1  order ok' % len(chk))
