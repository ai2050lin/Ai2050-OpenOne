# -*- coding: utf-8 -*-
"""Phase 3067 closeout (idempotent): Ledger -> L14 ->
MEMO append -> HDMCC audit addendum -> workspace log
-> MEMORY.md (project workspace)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3067'
     r'\omega_p64_mlp_conditional_reversal')
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
assert verdict == 'mlp_reversal_localized_shared', \
    verdict
assert seal['verdict'] == verdict
assert seal['setup_ok'] is True
an = res['anchors']
assert an['a1_diff'] == 0.0 and an['a1_ok'] is True
assert an['a2b_diff'] < 0.0002
assert an['a2b_ok'] is True
assert an['b0_diff'] == 0.0 and an['b0_ok'] is True
assert an['b1_diff'] == 0.0 and an['b1_ok'] is True
assert an['b3_ok'] is True
assert an['b4_diff'] == 0.0 and an['b4_ok'] is True
assert an['b5_diff'] == 0.0
assert an['b6_diff'] == 0.0 and an['b6_ok'] is True
assert an['b7_diff'] == 0.0 and an['b7_ok'] is True
assert an['setup_ok'] is True
st = res['stats']
assert st['med_c_34'] == 0.1487826048372403
assert st['med_c_35'] == -0.35579100779974404
assert st['pa34'] == 0.29685845971107483
assert st['pa35'] == 0.19866131991147995
assert st['pf34'] == 0.2371114194393158
assert st['pf35'] == -0.3558848798274994
assert st['c_perm_a35'] == 0.35042549175552035
assert st['c_perm_r35'] == -0.3565880930706864
assert st['c_perm_b34'] == 0.5379552249419755
assert abs(st['recov35']
           - 0.7062164995552644) < 1e-12
assert abs(st['recov_rand']
           + 0.0007970852709423548) < 1e-12
assert abs(st['delta34']
           - 0.3891726201047352) < 1e-12
assert abs(st['jac_cos_act_a']
           - 0.9926168314602393) < 1e-12
assert abs(st['jac_cos_act_b']
           - 0.9557677697038436) < 1e-12
assert abs(st['jac_cos_m_a']
           - 0.9921152415449941) < 1e-12
assert abs(st['jac_cos_m_b']
           - 0.9757333852964698) < 1e-12
assert abs(st['ang_ab']
           - 0.6031670946539858) < 1e-12
assert abs(st['ang_zab']
           - 0.028030945976114604) < 1e-12
assert abs(st['cmread_base']
           + 0.7335929365499825) < 1e-12
assert abs(st['cmread_a']
           + 0.7324421597621176) < 1e-12
assert abs(st['cmread_b']
           + 0.7400100835267196) < 1e-12
assert abs(st['score_max']
           - 13.43401150405407) < 1e-9
assert res['forwards'] == 158

# ---------- Ledger (idempotent) ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
if not any(m.get('phase') == 3067
           for m in led['measurements']):
    claim = (
        'Omega-P64 (plan 3067 A) - qwen3-4b bf16 '
        'L35 MLP conditional reversal anatomy '
        '(14.0s, 158 forwards; anchors: a1 ladder '
        'rows 34/35 bit-exact vs 3066 npz diff '
        '0.0; med_c reference assert passed; '
        'a2b 9.4e-05; b0/b1/b4/b5/b6/b7 bit 0.0; '
        'b3 finite; a2 fp32 lens within 1 bf16 '
        'ulp). Debug history: smoke caught 1 '
        'script bug before the authoritative run '
        '(Jacobian fp32 weight copies not '
        'detached - requires_grad crash). '
        'RESULTS: verdict '
        'mlp_reversal_localized_shared. '
        '(1) LOCALIZED: swapping the top-128 '
        'down_proj-input neurons (S_A scored by '
        'median_k |dact|*||W_down col||, 1.3 pct '
        'of 9728) back to base values at the L35 '
        'last position moves c from -0.356 to '
        '+0.350 (recov35=+0.706 - the negative '
        'effect MORE than fully removed); '
        'random-128 control -0.0008 (perfectly '
        'specific). (2) SHARED: the SAME S_A '
        'swap under state B (inj@34) moves c '
        'from +0.149 to +0.538 (delta34=+0.389) '
        '- the same neurons write negative-'
        'aligned diff contributions in BOTH '
        'states; the state-dependent sign of '
        'the TOTAL emerges from the balance: '
        'rest-of-system positive write (+0.35/'
        '+0.54) vs S_A negative write (stronger '
        'in state A). (3) LINEAR: one Jacobian '
        'linearization around the base state '
        'predicts the MLP diff response in BOTH '
        'states (cos 0.992 A / 0.976 B) - no '
        'nonlinear gate switch; the two input '
        'displacements are nearly ORTHOGONAL '
        '(cos(dz_A,dz_B)=0.028) while outputs '
        'are at 0.603 - conditionality enters '
        'through the input direction, not the '
        'processor. (4) The absolute m-channel '
        'write direction is stable across '
        'states (CMREAD -0.734/-0.732/-0.740) - '
        'the always-on write is stereotyped; '
        'state-dependence lives in the diff. '
        'Conclusion: the L35 conditional '
        'reversal = fixed local linear map '
        '(gear teeth J) x input-direction '
        'selection (drive shaft) x competition '
        'balance in readout space; the sign is '
        'a property of the PAIR (neuron set, '
        'input displacement), not of parameters '
        '- the 3066 (component, state-condition)'
        ' unit resolves at neuron level into '
        '(neuron set x input direction x '
        'balance).')
    meas = {
        'meas_id': 'meas3067_omega_p64_mlp_'
                   'conditional_reversal',
        'phase': 3067,
        'claim': claim,
        'verdict': verdict,
        'anchors': 'a1 rows 34/35 bit 0.0 vs '
                   '3066 npz (hard); med_c '
                   'reference assert; a2b 9.4e-05; '
                   'b0/b1/b4/b5/b6/b7 bit 0.0; '
                   'b3 finite; a2 fp32 lens '
                   '<= 1 bf16 ulp',
        'artifacts': {
            'result': 'phase3067/omega_p64_'
                      'mlp_conditional_reversal/'
                      'result.json',
            'npz': 'phase3067/omega_p64_'
                   'mlp_conditional_reversal/'
                   'omega_p64_mlp_conditional_'
                   'reversal.npz'},
        'hashes': {
            'npz_sha256_8': seal['npz_sha256_8'],
            'result_sha256_8': seal['result_sha256_8'],
            'script_sha256_8': seal['script_sha256_8']},
        'note': 'run1 authoritative (qwen3-4b '
                'single model bf16; smoke mode '
                'caught 1 script bug - undetached '
                'Jacobian weight copies - before '
                'the authoritative launch)',
    }
    led['measurements'].append(meas)
    assert len(led['measurements']) == 206
    l14['connects'].append({
        'meas_id': 'meas3067_omega_p64_mlp_'
                   'conditional_reversal',
        'phase': 3067,
        'axis': 'lang',
        'verdict': verdict,
        'grade_change': 'Omega-P64: L35 MLP '
                        'conditional reversal '
                        'anatomy - localized '
                        '(top-128/9728 neurons '
                        'carry the whole negative '
                        'effect, recov35 +0.706, '
                        'random control -0.001) '
                        'and SHARED (same neurons '
                        'negative-aligned in both '
                        'states, delta34 +0.389; '
                        'total sign = balance of '
                        'S_A-negative vs rest-'
                        'positive writes); linear-'
                        'Jacobian-explainable in '
                        'both states (cos 0.99/0.98)'
                        ' with nearly orthogonal '
                        'input displacements (0.028)'
                        ' - no nonlinear switch; '
                        'the 3066 (component, state)'
                        ' unit resolves into (neuron '
                        'set x input direction x '
                        'competition balance). '
                        'Opens 3068 A: S_B symmetric '
                        'selection + competition '
                        'decomposition'})
    led.pop('ledger_sha256_8')
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w', encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False,
                  indent=1)
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
if '## Phase 3067:' not in memo:
    sec = u'''## Phase 3067: Ω-P64 L35 MLP 条件反转解剖——定位=top-128 神经元、两状态共享、线性雅可比可解释（mlp_reversal_localized_shared） [%(created)s]

**判决：`mlp_reversal_localized_shared`**（qwen3-4b 单模型 bf16 14.0s，158 次前向；锚：**a1：阶梯 34/35 两行与 3066 npz diff=0.000e+00——跨脚本 bit 级一致**；med_c 参考断言通过（0.1487826048372403 / -0.35579100779974404）；a2b lens 校准 9.4e-05；b0/b1/b4/b5/b6/b7 全 bit 0.0；b3 finite；a2 fp32 lens ≤1 bf16 ulp）。调试史如实入册：smoke 模式在权威启动前抓住 1 个脚本 bug（**雅可比 fp32 权重副本未 detach——模型参数默认 requires_grad=True，colnorm 处 numpy() 崩溃**）。无崩溃，smoke→权威一次通过。

### 问题与设计
**问题**（3067 A，3066 菜单）：L35 MLP 条件反转（直接注入 raw +0.199→最终 -0.356；对 L34 传播位移 +0.237→+0.149 保持正）的**参数载体**是什么？设计三步：E1 双状态捕获（l∈{34,35}×24 对，协议 3066 逐字一致→a1 bit 锚；末位捕获 L35 的 z=ln2 输出、g/u=gate/up 输出、act=down_proj 输入、m/a/x）；E2 线性雅可比检验（绕 base 态线性化：`dact_lin = (silu'(g0)⊙u0)⊙(W_gate dz) + silu(g0)⊙(W_up dz)`，`dm_lin = W_down dact_lin`，fp32 权重副本，记录性无阈值；另测两状态输入/输出方向夹角与 CMREAD 绝对写入极性）；E1.5 神经元评分 `score_j = median_k(|dact_k[j]|·||W_down[:,j]||)`→S_A=top-128、S_R=随机 128（seed 3023）；E3 因果换回（**down_proj forward-pre-hook** 在 L35 末位把 act[S] 换成 base 值——L35 更早位置的 MLP 到不了末位读出：L35 attention 读的是块输入 x_35 而非块输出，故末位置换即完备；base 自替换=恒等（b7 验证）→免 matched base）。

### 核心结果（重复三遍）
**① 定位（一）**：top-128/9728（1.3 pct）神经元换回 base，c 从 **-0.356→+0.350**（recov35=**+0.706**，负效应被超过 100 percent 移除）；随机 128 对照 **-0.0008**——特异性完美，评分公式（|dact|·||W_down 列||）一步命中载体。**② 共享（二）**：同一 S_A 在状态 B（inj@34）下换回，c 从 +0.149→**+0.538**（delta34=**+0.389**）——**同一批神经元在两个状态下都写负对齐差分**；总符号的状态依赖来自竞争平衡：其余系统写正（+0.35/+0.54），S_A 写负（状态 A 中更强），A 中负胜、B 中正胜。**③ 线性（三遍）**：绕 base 的**同一个**雅可比线性化在两个状态下都预测 MLP 差分响应（cos_act **0.993/0.956**，cos_m **0.992/0.976**）——**无需非线性开关**；两状态输入位移近乎**正交**（cos(dz_A,dz_B)=**0.028**）而输出夹角 0.603——条件性经输入方向进入，而非处理器改变计算方式。附加：CMREAD 绝对 m 通道写入极性三态稳定（**-0.734/-0.732/-0.740**）——常开写入是刻板的，状态依赖住在差分里；score_max 13.43 vs 全体中位 0.065（约 200 倍）集中。

### 机制综合（Ω-P64 拼图）
3066 的"编排单位=（组件, 状态条件）对"在本 phase 解析到神经元级：**（神经元集 × 输入方向 × 竞争平衡）**。条件齿轮的第一个参数化解答：**J（固定局部线性映射）是齿，输入位移方向是传动轴，读出空间中的向量竞争产生符号**——符号不是任何参数组的属性，而是（参数组, 输入位移）对的属性。这同时解释了为何 3065 符号追踪几何全败（符号住计算流）：符号在 J·dz 与读出投影的夹角里，不在权重矩阵的行列符号里。

### 硬伤与边界
- c 空间读出非线性（final_norm）：recov35 超 100 percent（overshoot）说明恢复量不能当线性贡献份额读，只能读方向与相对强度；S_A 份额是定性的。
- swap=换到 base（差分贡献移除），非消融到零——不说话那些神经元的绝对功能。
- 单 prompt 族（style/connector）、单模型、单层（L35）、单位置（末位）；跨族/跨层/跨模型泛化未测。
- S_A 用状态 A 选择——选择偏置使 delta34=+0.389 是"状态 B 最优集重要性"的下界；对称的 S_B 未测。
- 线性化范围出乎意料地宽（||dz_A||=18.3 约为 ||z0|| 的 36 percent 仍有 cos 0.993）——记录为观察，不外推为普适。

### 方法论入册
- **改模块输入必须用 forward-pre-hook**（forward hook 拿到的 inp 已是计算后值且不可换）；`def h(module, args): a=args[0].clone(); a[mask]=repl; return (a,)`。
- **base 自替换=恒等 → 免 matched base**：换回 base 值的对照在 base 前向上是恒等（b7 bit 0.0 验证），省一半前向。
- **权重副本必须 .detach().float()**（模型参数 requires_grad=True；smoke 抓住）。
- SMOKE 连续第三 phase 在权威启动前抓全 bug（3066 四个、3067 一个）——零崩溃达成。

### 智能理论洞察（第一性原理）
"同一组件对不同状态输出相反符号"（3066）现在有了完整的机制分解：**不是组件变了，是输入位移的方向变了**。Δz_A 与 Δz_B 近乎正交（0.028），同一个 J 把它们映射到 0.603 夹角的两个输出方向，读出投影让它们落在 TT 超平面两侧。这给出"条件化齿轮组"的第一张参数级图纸：**齿轮（J）不变、传动（输入方向）变、输出符号由读出空间中的竞争平衡涌现**。语言能力的"无限组合"在此有了一个可操作的微缩模型：有限参数组（J、读出几何）× 有限类型的状态位移方向 → 组合出任意符号/幅度的响应。下一步的关键问题转为：**输入位移方向的谱系**——dz_A 是否就是注入 V 签名经 ln1/attn/ln2 的像（可线性预测？），dz_B 沿哪条传播路径到达、其方向由什么决定。

### 产物与 hash
`tests/glm5/result/rdc_query_construction_20260913/phase3067/omega_p64_mlp_conditional_reversal/`；script8 %(script8)s、result8 %(result8)s、npz8 %(npz8)s、exec8 %(exec8)s；ledger %(n)d / L14 %(l14)d。

**接续 3068 菜单**——A（主选）**S_B 对称选择+竞争分解**：状态 B 的 dact 选 top-128（S_B），做对称置换（inj@34+S_B、inj@35+S_B）+ 全 9728 置换上界锚，定量分解 S_A/S_B 重叠与竞争 balance；B **跨层推广**：同一 swap-to-base 协议应用到 L30-34 正带 MLP（每层 top-128），检验"MLP 条件反转"是否逐层普遍；C **DS7B 对照**：DS7B 末层 MLP 解剖复刻（符号住传播动力学的跨模型检验，3065 菜单 C 遗留）；D **S_A 神经元身份**：top-128 的 up/gate 权重结构、与 W_U 语义方向关联、跨 prompt 族 dact 稳定性；E **输入方向谱系**：dz_A 是否=注入 V 签名的线性像、dz_B 传播路径分解。"好的，继续"即进 3068 A。
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
if '## 二十九、3067 增补' not in aud:
    add = u'''

---

## 二十九、3067 增补：L35 MLP 条件反转解剖——定位 top-128、两状态共享、线性雅可比可解释（Omega-P64，判决 mlp_reversal_localized_shared）

1. **定位**：top-128/9728（1.3 pct）down_proj 输入神经元（评分 |dact|·||W_down 列||）换回 base 值，c 从 -0.356→+0.350（recov35=+0.706，负效应超 100 percent 移除）；随机 128 对照 -0.0008——载体定位一步命中且完全特异。
2. **共享而非双群**：同一 S_A 在状态 B 下换回使 c +0.149→+0.538（delta34=+0.389）——同一批神经元在两状态都写负对齐差分；总符号=其余系统（写正）与 S_A（写负，状态 A 更强）的竞争平衡。
3. **线性可解释**：绕 base 的单一雅可比线性化预测两状态 MLP 差分响应（cos 0.992/0.976）；输入位移近乎正交（0.028）而输出夹角 0.603——条件性经输入方向进入，处理器计算方式不变；绝对 m 写入极性三态稳定（约 -0.73）。
4. HDMCC 修正：3066 的（组件, 状态条件）对在神经元级解析为（神经元集 × 输入方向 × 竞争平衡）；符号定位实验的标准协议由此确立——评分选择→换回 base 置换→随机对照→雅可比检验四件套。
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
if 'Phase 3067' not in prev:
    line = ('- Phase 3067 Omega-P64 L35 MLP '
            'conditional reversal anatomy '
            '(qwen3-4b bf16 single, 14.0s, 158 '
            'forwards): verdict '
            'mlp_reversal_localized_shared. a1 '
            'rows 34/35 bit-exact vs 3066 npz '
            'diff 0.0; med_c reference assert; '
            'b0/b1/b4/b5/b6/b7 bit 0.0. LOCALIZED:'
            ' top-128/9728 neurons swap-to-base '
            'moves c -0.356 -> +0.350 (recov '
            '+0.706), random control -0.001. '
            'SHARED: same S_A under state B '
            'moves +0.149 -> +0.538 (delta34 '
            '+0.389) - same neurons negative-'
            'aligned in both states; total sign '
            '= competition balance. LINEAR: one '
            'base Jacobian predicts both states '
            '(cos 0.992/0.976); input '
            'displacements near-orthogonal '
            '(0.028) - conditionality enters '
            'through input direction, no '
            'nonlinear switch; absolute m-write '
            'stable (-0.73) across states. '
            '(component,state) unit resolves to '
            '(neuron set x input direction x '
            'balance). Smoke caught 1 bug '
            '(undetached Jacobian weights). '
            'Audit 29; ledger 206/L14 174.\n')
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
if 'max=3067' not in mem_cur:
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
- bit 级仅限同文件链/同精度；跨脚本 bit 级锚（a1/a4：独立实现 diff=0.0）是"新脚本=旧机制"最强验证。
- **3066**：per-k 聚合必须显式累积后 median；cos 探针矩阵乘用 fp32 权重副本；消融变体配 matched base；SMOKE 先行。
- **3067**：改模块输入用 forward-pre-hook；base 自替换=恒等→免 matched base（b7）；权重副本 .detach().float()；符号定位四件套=评分选择→换回 base 置换→随机对照→雅可比检验。

## 机制解释审计链（命名前依次检查）
…→KV 阶梯→门位易感→γ 管道→PC1→写入分解→3063 竞争轴→3064 跨模型（拓扑普适/符号特异）→3065 符号溯源（符号住传播动力学）→3066 翻转解剖（正带=下游分布式正化；L35 负号=MLP 条件反转；编排单位=（组件,状态条件）对）→**3067 神经元级解析：定位=top-128（recov +0.706/随机 -0.001）、两状态共享（delta34 +0.389）、线性雅可比可解释（cos 0.99/0.98）、输入位移近正交（0.028）→（神经元集×输入方向×竞争平衡）**。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：绝对路径用 D:/ 风格（/d/ 失败）；-c stdout 丢→写文件再 Read；长任务超时给足。
- 关键写入后必须 Grep/Read 复核；Python 补丁 assert count==1 唯一可靠；改后必编译检查。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出。

## 机制链状态（2936-3067）
Ω-P2（3011-3067）：3045-3048 KV 阶梯；3050 末层；3051 K 场+头；3052-3053 门槽；3054-3057 γ 重整/预对齐/反方差；3060 PC1 身份；3061 写入分解；3063 竞争轴；3064 chain_fragmented；3065 sign_decoupled_all3；3066 sign_flip_downstream_distributed_heads_dist；**3067 mlp_reversal_localized_shared（J 固定=齿、输入方向=传动、符号=读出空间竞争平衡涌现）**。

## 下一步
- max=3067，下一个 3068（A 主选 **S_B 对称选择+竞争分解**；B 跨层推广 L30-34；C DS7B 末层对照；D S_A 神经元身份；E 输入方向谱系 dz_A/dz_B）。
'''
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory written %d chars' % len(mem_new))
else:
    o.append('memory already max=3067')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('CLOSEOUT_OK')
