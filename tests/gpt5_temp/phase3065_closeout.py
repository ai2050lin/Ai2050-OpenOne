# -*- coding: utf-8 -*-
"""Phase 3065 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md (project workspace)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3065'
     r'\omega_p62_v_sign_orchestration')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG_DIR = ROOT + r'\.workbuddy\memory'
LOGF = R + r'\closeout_log.txt'
o = []

res = json.load(io.open(R + r'\result.json',
                        encoding='utf-8'))
seal = json.load(io.open(R + r'\seal.json',
                         encoding='utf-8'))
exe = json.load(io.open(R + r'\execution.json',
                        encoding='utf-8'))
created = exe['created']
verdict = res['verdict']
assert verdict == 'sign_decoupled_all3', verdict
assert seal['verdict'] == verdict
assert seal['setup_ok'] is True
an = res['anchors']
assert an['a4_diff'] == 0.0
assert an['a4_ok'] is True
assert abs(an['a5_val']
           + 0.35579100779974404) < 1e-12
assert an['setup_ok'] is True
st = res['stats']
m1 = st['qwen3_1p7b']
assert m1['b0_ok'] and m1['b1_ok'] and m1['b3_ok']
assert abs(m1['med_c_last']
           - 0.13036215802991347) < 1e-12
assert abs(m1['r']
           - 0.21354441924588854) < 1e-12
assert abs(m1['p_r']
           - 0.28214357128574286) < 1e-12
assert m1['track'] is False
assert m1['late_med_flip'] == 0.0
assert abs(m1['y_tgt_med'] - 18.9375) < 1e-12
m4 = st['qwen3_4b']
assert m4['b0_ok'] and m4['b1_ok'] and m4['b3_ok']
assert abs(m4['med_c_last']
           + 0.35579100779974404) < 1e-12
assert abs(m4['r']
           + 0.4183748544178161) < 1e-12
assert abs(m4['p_r']
           - 0.011197760447910418) < 1e-12
assert m4['track'] is False
assert abs(m4['med_cV_last']
           - 0.8635946020993787) < 1e-12
assert abs(m4['y_tgt_med'] - 4.734375) < 1e-12
m7 = st['ds7b']
assert m7['b0_ok'] and m7['b1_ok'] and m7['b3_ok']
assert abs(m7['med_c_last']
           - 0.9134400687623977) < 1e-12
assert abs(m7['r']
           - 0.2843950837076938) < 1e-12
assert abs(m7['p_r']
           - 0.15136972605478904) < 1e-12
assert m7['track'] is False
assert m7['late_med_flip'] == 1.0
assert abs(m7['med_cV_last']
           - 0.8499320481254675) < 1e-12
assert abs(m7['y_tgt_med'] + 17.8125) < 1e-12
cj = res['cross_jaccard']
assert abs(cj['qwen3_4b|ds7b']['obs']
           - 0.002004008016032064) < 1e-12

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3065
           for m in led['measurements']):
    claim = (
        'Omega-P62 (plan 3065 A) - three-model '
        'bf16 V-arm sign tracing (qwen3-1.7b -> '
        'qwen3-4b -> ds7b sequential, 136.9s, '
        '~2285 forwards; anchors b0/b1 bit 0.0 '
        'x3, b3 finite; a4 DS7B ladder bit-'
        'exact vs 3064 S1 diff 0.0; a5 qwen3-4b '
        'bf16 vs 3051/3063 fp32 -0.3409 diff '
        '0.015 recorded-only). Debug history: 2 '
        'crashes fixed (detach on lm_head '
        'indexing; correlation subset must be '
        'selected on both sides). RESULTS: '
        'verdict sign_decoupled_all3. (1) Sign '
        'matrix at last layer: 1.7b med_c +0.130 '
        'weak positive / 4b -0.356 / ds7b +0.913 '
        '- the V adversarial flip is neither '
        'Qwen3-family (1.7b lacks it) nor '
        'universal: qwen3-4b checkpoint-'
        'specific orchestration. (2) Geometry '
        'tracking FAILS in all 3 (|r| >= 0.6 & '
        'p <= 0.01 all fail; r = 0.21/-0.42/'
        '0.28, p = 0.28/0.011/0.15) - med_cV '
        'stays 0.49-1.00 across layers while '
        'med_c swings -0.36..+0.91: static V '
        'displacement geometry does not '
        'predict effect sign. (3) qwen3-4b '
        'negative lives in TWO bands (L23-29 '
        'and L35 endpoint flip after positive '
        'L30-34 +0.10..+0.53); ds7b only L18; '
        'late-injection propagation median '
        'flips: ds7b 1.0 vs qwen 0.0 - sign is '
        'produced by downstream computation '
        'dynamics, not injection geometry. '
        '(4) E4: y_tgt +18.9/+4.7/-17.8 - '
        'opposite vocab-space behavior modes; '
        'mass256 0.004-0.008 (readout delta '
        'diffuse); cross-model top-256 string '
        'Jaccard 0.002-0.004 null (p 0.28-'
        '0.63) - readout signature model-'
        'specific. Conclusion: structure lives '
        'in weights, sign orchestration lives '
        'in the computation flow; orchestration '
        'carrier search opens (3066).')
    meas = {
        'meas_id': 'meas3065_omega_p62_v_sign_'
                   'orchestration',
        'phase': 3065,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'per model b0 recapture bit '
                   '0.0 (LG+VB+PB, 4 prompts); '
                   'b1 full-field V sham bit '
                   'identity; b3 finite; a4 '
                   'ds7b med_c(27) vs 3064 '
                   'diff 0.0 (bit-exact, in '
                   'setup_ok); a5 qwen3-4b '
                   'med_c(35) -0.3558 vs fp32 '
                   '-0.3409 diff 0.015 '
                   '(recorded-only)',
        'artifacts': {
            'result': 'phase3065/omega_p62_'
                      'v_sign_orchestration/'
                      'result.json',
            'npz': 'phase3065/omega_p62_'
                   'v_sign_orchestration/'
                   'omega_p62_v_sign_'
                   'orchestration.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (three '
                'models bf16 sequential; 2 '
                'crashed launches pre-verdict '
                'registered - detach, subset '
                'selection; both script bugs)',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 204
    l14['connects'].append({
        'meas_id': 'meas3065_omega_p62_v_sign_'
                   'orchestration',
        'phase': 3065,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P62: V-arm sign is '
                        'DECOUPLED from injection-'
                        'layer V geometry in all '
                        'three models '
                        '(sign_decoupled_all3) - '
                        'sign matrix +0.130/-0.356/'
                        '+0.913 (1.7b/4b/ds7b) '
                        'shows the adversarial '
                        'flip is a qwen3-4b '
                        'checkpoint-specific '
                        'computation-flow '
                        'orchestration, not a '
                        'family or universal '
                        'property; qwen3-4b '
                        'negative localized to '
                        'L23-29 band + L35 '
                        'endpoint flip; ds7b '
                        'flips once mid-'
                        'propagation. Orchestration '
                        'carrier search opens '
                        '(3066)'})
    led.pop('ledger_sha256_8')
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w', encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False, indent=1)
    o.append('ledger appended n=%d l14=%d sha=%s'
             % (len(led['measurements']),
                len(l14['connects']),
                led['ledger_sha256_8']))
else:
    o.append('ledger already upserted n=%d l14=%d'
             % (len(led['measurements']),
                len(l14['connects'])))

# ---------- MEMO append ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3065:' not in memo:
    sec = u'''## Phase 3065: Ω-P62 V 臂符号编排溯源——三模型符号矩阵+几何追踪解耦（sign_decoupled_all3） [%(created)s]

**判决：`sign_decoupled_all3`**（三模型 bf16 顺序 136.9s，约 2285 次前向；锚：三模型 b0 recapture bit 0.0、b1 全场 V 自替换 bit 0.0、b3 finite；**a4：DS7B 阶梯 med_c(27)=0.9134400688 与 3064 S1 独立脚本 diff=0.000e+00——跨脚本 bit 级一致**；a5：qwen3-4b med_c(35)=−0.3558 vs 3051/3063 fp32 −0.3409，diff=0.015 跨精度一致）。调试史如实入册：2 次崩溃修复后第 3 次启动成功——①lm_head 裸索引缺 detach（no_grad 不覆盖外部索引）；②相关子集只在一侧选择（med_cV[sel] vs med_c 维度不匹配，须两侧同选）。

### 问题与设计
**问题**（3065 A，3064 菜单）：V 臂符号（qwen3-4b −0.34 vs DS7B +0.913）在**注入层 V 几何**还是在**下游传播动力学**中引入？三模型 bf16（qwen3-1.7b 28L/2048/8kv → qwen3-4b 36L/2560/8kv → DS7B 28L/3584/4kv）同一协议顺序执行：E1 逐层 V-only 阶梯（28/36 层×24 对，FRONT=4 行替换）；E2 逐层 V 位移几何（cos(srcV,VB)、rho=‖srcV−VB‖/‖VB‖，纯库计算）；E3 传播轨迹（E1 前向免费捕获：d_l' 逐层 cos(d,W_U[t_k]) + 翻转计数）；E4 读出签名（Y=D·W_Uᵀ GPU bf16 矩阵乘、top-256 质量、目标 delta、top-16 字符串、跨模型字符串 Jaccard）。track 判据预注册：rho≥0.1 层上 |pearson(med_c,med_cV)|≥0.6 ∧ perm p≤0.01（5000 次层置换 seed 3021）。

### 核心结果（重复三遍）
**① 符号矩阵（一）**：末层 med_c(L−1)：qwen3-1.7b **+0.130** 弱正、qwen3-4b **−0.356**、DS7B **+0.913**——负号既非 Qwen3 家族性（1.7b 无）也非普适（DS7B 反向），是 **qwen3-4b 检查点特异的编排**。**② 几何解耦（二）**：三模型 track 全部失败（r=0.21/−0.42/0.28，p=0.28/0.011/0.15）——med_cV 全层保持 0.49–1.00 高对齐而 med_c 在 −0.36..+0.91 摆动：**静态 V 位移几何不预测效应符号**。**③ 传播动力学（三遍）**：qwen3-4b 负号住两个带——L23-29 中晚负带（−0.08..−0.18）+ **L35 末层翻转**（L30-34 全正 +0.10..+0.53 → L35 −0.36）；DS7B 唯一负层 L18（−0.10）且 late 注入传播中位翻转 1.0（注入后符号在传播中变一次再回正）；两 qwen 模型 late_med_flip=0（晚注入符号传播中稳定）。符号由**下游计算动力学**产生。

### E4 读出签名（词表空间）
y_tgt（末层注入的目标 token logit 变化）：1.7b **+18.9** / 4b **+4.7** / DS7B **−17.8**——DS7B 的 V 臂是"反目标但顺转移方向"（cos TT +0.91 ∧ 目标 logit 降），qwen3-4b 是"顺目标但逆转移方向"（+4.7 ∧ −0.36）——**两模型的 V 臂在词表空间行为模式相反**。mass256=0.004–0.008（top-256 词表方向只承载 <1 pct 的 |Y| 质量——读出增量高度弥散，与 3063"非载荷"图像一致）；跨模型 top-256 字符串 Jaccard 0.002–0.004（null p 0.28–0.63，无显著重叠——读出签名在词表空间也模型特异）。

### 硬伤与边界
- **阈值依赖**：解耦负结果依赖预注册 |r|≥0.6；qwen3-4b r=−0.42 p=0.011 是阈下 suggestive 反相关——不能排除弱几何调制，但即使存在也远不足以决定符号。
- rho 分母用单行 ‖VB‖——层间 V 范数漂移带来系统偏差（已按描述量使用）。
- 单一 prompt 族（8 body×3 前缀）；FRONT=4 行单一注入协议；E4 字符串 null 未校正弦重复（描述性）。
- bf16 三模型：由 a4 bit 级 + a5 0.015 背书。

### 方法论入册
- **裸索引模型权重必须 detach**（tk_rows=Wemb[ids].detach()）——no_grad 只包 forward 不包外部索引。
- **相关/回归子集必须两侧同选**（med_cV[sel] 与 med_c[sel]）——单侧选择是本 phase 第 2 个崩溃。
- **跨脚本 bit 级锚是"新脚本=旧机制"的最强验证**：3064 与 3065 两份独立实现同一阶梯在 DS7B 上 diff=0.0。

### 智能理论洞察（第一性原理）
本期把 3064 的"符号编排层"定位到**传播动力学**而非**写入几何**：三个模型的 V 空间位移几何高度相似（cV 0.49–1.00、rho 同量级），但同样的几何扰动被三个模型的下游计算处理成完全不同的效应符号（+0.13 / −0.36 / +0.91）。这支持更强的分层主张：**结构住在权重里（门区、管道、单秩身份——3064 已证普适），编排住在计算流里（符号、支配——本证模型特异且几何不可预测）**——同一权重几何可经不同动力学编排产生相反行为极性。第一性原理问题升级：编排由什么自由度承载（头组合模式？γ 通道逐层重加权序列？残差流范数调度？）——指向 3066"编排自由度定位"。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3065/omega_p62_v_sign_orchestration/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3066 菜单**——A（主选）**末层翻转机制解剖**：qwen3-4b L34→L35 符号翻转归因（+0.15→−0.36：L35 attention 头组合 vs MLP vs γ 读出的贡献——复用 3051/3058 头级/通道级机械做符号归因）；B **编排自由度定位**（跨模型同几何不同符号的载体：头组合 vs γ 调度 vs 范数调度）；C DS7B 传播翻转溯源（late 注入 mid-propagation 翻转层定位）；D 门区 2D 易感图（沿用）；E 竞争轴源头定位（沿用）。"好的，继续"即进 3066 A。
''' % {'created': created,
           'script8': seal['script_sha256_8'],
           'result8': seal['result_sha256_8'],
           'npz8': seal['npz_sha256_8'],
           'exec8': seal['exec_sha256_8'],
           'n': len(led['measurements']),
           'l14': len(l14['connects'])}
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars' % len(sec))
else:
    o.append('memo already appended')

# ---------- HDMCC audit addendum ----------
aud = io.open(AUDIT, encoding='utf-8').read()
if '## 二十七、3065 增补' not in aud:
    add = u'''

---

## 二十七、3065 增补：三模型 V 臂符号溯源——几何解耦、符号住计算流（Omega-P62，判决 sign_decoupled_all3）

1. **符号矩阵**：末层 V-only 阶梯 med_c：qwen3-1.7b +0.130（弱正）/ qwen3-4b −0.356 / DS7B +0.913——V 臂对抗翻转既非 Qwen3 家族性也非普适，是 qwen3-4b 检查点特异编排；a4（DS7B 与 3064 bit 级 diff=0.0）与 a5（qwen3-4b bf16 vs fp32 diff=0.015）双锚背书。
2. **几何解耦**：三模型阶梯符号均不追踪注入层 V 位移几何（|r|≥0.6 ∧ p≤0.01 全败；med_cV 全层 0.49–1.00 而 med_c 摆动 −0.36..+0.91）——静态写入几何不决定效应符号；qwen3-4b 的 r=−0.42 p=0.011 为阈下 suggestive。
3. **符号住传播动力学**：qwen3-4b 负号=L23-29 带+L35 末层翻转（L30-34 全正）；DS7B 唯一负层 L18 且 late 注入传播中位翻转 1.0（qwen 为 0）；y_tgt 三模型 +18.9/+4.7/−17.8——DS7B"反目标顺转移"、qwen3-4b"顺目标逆转移"，词表空间行为模式相反。
4. HDMCC 修正：Ω-P2 双层结构进一步定位——"拓扑层住权重、编排层住计算流"；跨模型普适性检验必须区分权重几何与动力学编排两个载体；符号类结论（对抗/同盟极性）不得从写入空间几何外推。
'''
    aud += add
    with io.open(AUDIT, 'w', encoding='utf-8') as f:
        f.write(aud)
    o.append('audit addendum +%d chars' % len(add))
else:
    o.append('audit already appended')

# ---------- workspace log ----------
wl = os.path.join(WLOG_DIR, '2026-09-21.md')
try:
    prev = io.open(wl, encoding='utf-8').read()
except IOError:
    prev = ''
if 'Phase 3065' not in prev:
    line = ('- Phase 3065 Omega-P62 three-model '
            'V-sign tracing (1.7b/4b/ds7b bf16 '
            'sequential, 136.9s): verdict '
            'sign_decoupled_all3. Sign matrix '
            'med_c(last): +0.130 / -0.356 / +0.913 '
            '- adversarial flip is qwen3-4b-'
            'specific (1.7b lacks it). Geometry '
            'tracking fails all 3 (r=0.21/-0.42/'
            '0.28) - static V displacement does '
            'not predict sign; qwen3-4b negative '
            '= L23-29 band + L35 endpoint flip '
            '(L30-34 positive); ds7b late-'
            'injection propagation flip 1.0 vs '
            'qwen 0.0; y_tgt +18.9/+4.7/-17.8 '
            '(opposite vocab-space modes); '
            'mass256 <1 pct; cross-model string '
            'Jaccard null. Anchors: a4 ds7b vs '
            '3064 bit-exact diff 0.0; a5 qwen3-4b '
            'bf16 vs fp32 diff 0.015. Structure '
            'in weights, orchestration in '
            'computation flow. 2 crashes fixed '
            '(detach on weight indexing; both-'
            'sides subset selection). Audit 27; '
            'ledger 204/L14 172.\n')
    with io.open(wl, 'a', encoding='utf-8') as f:
        f.write(line)
    o.append('wlog appended')
else:
    o.append('wlog already')

# ---------- MEMORY.md (project workspace) ----------
MEMO_W = os.path.join(WLOG_DIR, 'MEMORY.md')
try:
    mem_cur = io.open(MEMO_W, encoding='utf-8').read()
except IOError:
    mem_cur = ''
if 'max=3065' not in mem_cur:
    mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）、qwen3-1.7b、DS7B=deepseek-r1-distill-qwen-7b（Qwen2 28L GQA 4kv 3584 bf16）。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；closeout tests\\gpt5_temp\\；产物 ...\\phase{N}\\{arm}\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json：measurement+L14.connects（第 14 项 link_id=L14_readout_spectrum_cross_model）；hash=去 ledger_sha256_8 后 dumps(sort_keys, ensure_ascii=False) sha256 前 8。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`（=created）；占位符 %(key)s 风格；裸百分号写 %%。
3. 重跑先删旧 execution.json 与 result.json；负结果/崩溃如实登记；verdict 单分支赋值。
4. 统计纪律：obs/null 同量纲；负对照带符号解读；阈值预注册；随机对照判特异性。

## 标准锚与精度
- bit 级仅限同文件链/同精度；精确 KV 复放：post-norm K+RoPE 偏移旋转；全长 repl+全 True mask。
- **跨脚本 bit 级锚（如 a4：两份独立实现同阶梯 diff=0.0）是"新脚本=旧机制"最强验证**。
- SVD 符号任意性用 |cos|；json 禁 numpy 标量；大数组用后即 del（必须在最后使用点之后）。
- **3063**：wud vocab-major——G=W_U·W_Uᵀ 写成 wud.T @ wud；同源分族级/逐对两层。
- **3064**：transformers 5.14.1 DynamicCache 唯一路径 cache.layers[li].keys；1-based 条件 ID 当位置索引用必须 -1。
- **3065**：裸索引模型权重必须 detach（no_grad 不覆盖外部索引）；相关/回归子集两侧同选；三模型顺序跑必须 del+gc+empty_cache。

## 机制解释审计链（命名前依次检查）
…→KV 阶梯→门位易感→γ 管道→PC1→写入分解→3063 竞争轴→3064 跨模型（拓扑普适/符号特异）→**3065 符号溯源：三模型符号矩阵 +0.130/−0.356/+0.913（1.7b/4b/DS7B）；几何追踪解耦（track 全败）；符号住传播动力学（4b=L23-29+L35 翻转带；DS7B 传播翻转 1.0）；结构住权重、编排住计算流**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径 python.exe；-c stdout 丢→写文件再 Read；长任务 run_in_background。
- cmd.exe 经 bash 损坏（参数截断）——删除/列目录用 Python。
- 关键写入后必须 Grep/Read 复核；幻影 Edit 会再现——Python 补丁 assert count==1 唯一可靠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3065）
Ω-P2（3011-3065）：3045-3048 KV 阶梯；3050 末层；3051 K 场+头；3052-3053 门槽+内容自由门；3054 γ 重整；3055 γ 预对齐；3057 反方差重加权；3060 PC1 身份；3061 写入分解；3062 身份解码 opaque；3063 竞争轴（管道内第三轴）；3064 DS7B chain_fragmented（拓扑普适/符号特异）；**3065 sign_decoupled_all3：符号住传播动力学非注入几何；编排自由度搜索开启**。

## 下一步
- max=3065，下一个 3066（A 主选 **末层翻转机制解剖**：qwen3-4b L34→L35 翻转归因 头组合/MLP/γ；B 编排自由度定位；C DS7B 传播翻转溯源；D 门区 2D 易感图；E 竞争轴源头定位）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars' % len(mem_new))
else:
    o.append('memory already max=3065')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
