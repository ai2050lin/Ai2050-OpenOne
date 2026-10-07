# -*- coding: utf-8 -*-
"""Phase 3115 closeout (idempotent):
Ledger -> MEMO Phase 3115 -> workspace logs -> MEMORY.md."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3115'
        r'\omega_p113_joint_mlp_erase_purpose')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory'
WLOG_C = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
          r'\.workbuddy\memory')
MEMO_W = WLOG_D + r'\MEMORY.md'
LOGF = OUTD + r'\closeout_log.txt'
NOW = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
TODAY = datetime.date.today().isoformat()
o = []

res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
assert res['verdict'] == \
    'write_joint_partial|additive|' \
    'erase_serves_generation'
assert res['n_records'] == 2016
assert res['smoke'] is False
assert -0.30 < res['dmp_rel']['joint20_28'] <= -0.10
assert res['dmp_rel']['L20_mlp'] < -0.30
assert abs(res['dmp_rel']['joint_minus_single_sum']) \
    < 0.05
assert res['erase_purpose']['js_delta_mean'] > 0
assert res['erase_purpose']['js_delta_sign_rate'] \
    >= 0.60
assert res['m_consistency_max_abs_vs_3113'] == 0.0
assert max(res['selfcheck_rel'].values()) < 0.01

# ---------- Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3115
           for m in led['measurements']):
    claim = (
        'Omega-P113 (3115, joint MLP ablation + erase-'
        'purpose JS readout on rebuilt 3105 material, '
        'qwen3-4b, 2016 records x 5 conditions, 364s) - '
        'verdict write_joint_partial|additive|'
        'erase_serves_generation.  (1) L20 SINGLE-POINT '
        'IS THE LARGEST CAUSAL EFFECT (dmp_rel -0.3576, '
        'marginal flip 558/2016 argnext) although 3113 '
        'correlational ds ranked it smallest (+0.503 '
        'vs L28 +2.478) - the correlational climb '
        'peaks late but the causal supply starts '
        'early; CORRELATION != CAUSALITY with a '
        'DIRECTION REVERSAL across the depth axis.  '
        '(2) JOINT L20+L24+L28 MLP ablation only '
        '-0.1879 (gate -0.30 not met) -> the three '
        '"write-peak" MLPs causally carry only ~19% '
        'of the margin; writing is DEEP DISTRIBUTED '
        '(the remaining ~81% lives in other layers/'
        'attention/embedding path).  Joint effect is '
        'ADDITIVE (joint - single_sum = -0.0119): '
        'L24 conditional adversariality (+0.346 '
        'single) cancels inside the joint; no chain '
        'collapse.  (3) ERASE PURPOSE CONFIRMED '
        'STRONG: per-pair JS(P_next|P || P_next|A1) '
        'rises 0.1593 -> 0.2128 under abl_L32_mlp '
        '(delta +0.0535, sign rate 0.976 over 672 '
        'pairs; top-10 Jaccard 0.808 -> 0.781) -> '
        'the L32 MLP erase CAUSALLY SUPPRESSES truth '
        'leakage into the generation distribution: '
        'BELIEF-GENERATION DECOUPLING - the model '
        'keeps the internal belief margin (L20-28 '
        'write) while preventing it from hijacking '
        'next-token generation (L32 erase).  '
        'Cross-phase bit-exactness: baseline m '
        'identical to 3113 capture (max abs diff '
        '0.0).  Self-checks: per-layer residual '
        'identity + upstream-untouched + exact '
        'single-layer identity, all < 2.4e-3.  '
        'CAVEATS: (i) JS may be dominated by frequent '
        'tokens; (ii) only the MLP channel ablated '
        'jointly (attention not); (iii) the ~81% '
        'remainder not localized (no L12-L36 full '
        'scan yet); (iv) single model.  NEXT 3116: '
        'full-layer MLP ablation sweep L12-L36 to '
        'localize the remaining causal contribution '
        '+ behavioral check of belief-generation '
        'decoupling in free generation.')
    meas = {
        'meas_id': 'meas3115_omega_p113_'
                   'joint_mlp_erase_purpose',
        'phase': 3115,
        'claim': claim,
        'verdict': 'write_joint_partial|additive|'
                   'erase_serves_generation',
        'anchors': 'design_seal.json frozen before '
                   'computation: joint gate dmp_rel '
                   '<= -0.30 / -0.10; superadditivity '
                   'band 0.05; erase gate mean>0 AND '
                   'sign_rate>=0.60 (full-vocab '
                   'natural-log JS, float64); frozen '
                   'singles L24 +0.3460 / L28 '
                   '-0.1644 from 3114 result.json',
        'artifacts': {
            'result': 'phase3115/omega_p113_'
                      'joint_mlp_erase_purpose/'
                      'result.json',
            'seal': 'phase3115/omega_p113_'
                    'joint_mlp_erase_purpose/'
                    'design_seal.json',
            'readout': 'phase3115/omega_p113_'
                       'joint_mlp_erase_purpose/'
                       'ablation_readout.npz'},
        'hashes': {},
        'note': 'GPU used (qwen3-4b, BF16, eager, '
                'batch 1, 364s for 5x2016 forwards '
                'incl. 2 full-distribution '
                'conditions); selfchecks '
                '1.3e-3..2.3e-3; m_consistency vs '
                '3113 capture = 0.0 (bit-exact)',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3115_omega_p113_joint_mlp_erase_purpose')
    led.pop('ledger_sha256_8', None)
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
    o.append('ledger already upserted')

# ---------- MEMO Phase 3115 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3115:' not in memo:
    sec = u'''## Phase 3115: Ω-P113 联合消融+擦除目的性——L20 单点因果最大（−0.358，相关 ds 却最小：方向反转）；L20-28 三层 MLP 联合仅 −0.188（写入深度分布式，~81% 在别处）；joint≈单点和（additive）；**L32 擦除目的性强证实：消融后 P/A1 生成分布 JS 0.159→0.213（符号率 0.976）→ 信念-生成解耦** [[NOW]]

**性质**：T3 第 9 Phase。qwen3-4b BF16，重建 3105 材料 2016 记录（672 对）× 5 条件（baseline / abl_L20_mlp / abl_L32_mlp / abl_L20_28_joint / abl_L20_32_joint），363.9s（含 2 个全分布条件）。预注册（design_seal.json 先于计算）：joint 门 dmp_rel ≤−0.30 → write_joint_collapse / ≤−0.10 → partial / else elsewhere；superadditivity 带 0.05；erase 门 = mean(JS_abl−JS_clean)>0 且配对符号率 ≥0.60 → erase_serves_generation（JS=全词表自然对数 float64 配对 P_next(·|P)‖P_next(·|A1)）；冻结 3114 单点 L24 +0.3460 / L28 −0.1644。

### 1. 方法与自检
MLP 输出置零 hook（层集合语义）；selfcheck 三层验证：每消融层残差恒等 + 最上游层上游未污染 + 单层条件精确恒等 h_out_abl=h_out_clean−mlp_clean；4 条件 rel L2 1.3e-03–2.3e-03。跨 Phase 复现：3115 baseline m 与 3113 capture m **max|diff|=0.0（bit-exact）**。JS 在线配对计算（pend_max=1，内存无忧）。

### 2. 结果（2016 记录，672 对）

| 条件 | mpair | dmp_rel | argnext flips |
| --- | --- | --- | --- |
| baseline | 8.2458 | — | 0 |
| abl_L20_mlp | 5.2971 | **−0.3576** | 558 |
| abl_L32_mlp | 9.0950 | +0.1030 | 280 |
| abl_L20_28_joint | 6.6961 | **−0.1879** | 345 |
| abl_L20_32_joint | 7.0157 | −0.1492 | 481 |

AUC_truth(m) 0.9851–0.9896。erase 目的性：js_base 0.1593 → js_abl 0.2128（**+0.0535，符号率 0.976**）；top-10 Jaccard 0.808→0.781（同向）。

### 3. 三大发现（重复三遍）
1. **L20 单点因果效应最大**（−0.358，超过 L28 的 −0.164；flips 558/2016 也是最大），而 3113 相关 ds 中 L20 最小（+0.503）——**相关轨迹峰值在后，因果供血源头在前：相关≠因果且沿深度方向反转**。**L20 是因果主力。相关峰值≠因果主力。**
2. **写入深度分布式**：L20+L24+L28 三个"写入峰"MLP 联合消融仅 −0.188（未过 −0.30 collapse 门）——三层合计因果承载仅 ~19%，其余 ~81% 在其他层/attention/embedding 路径。**联合效应 additive**（joint−single_sum=−0.012）：L24 的条件性对抗（单点 +0.346）在联合中消失，无连锁崩溃。**写入是深度分布的、近可加的。**
3. **擦除目的性强证实（信念-生成解耦）**：消融 L32 擦除后 truth 泄漏到生成分布显著增大（JS +34%，符号率 0.976）→ **L32 擦除的功能是阻止内部信念绑架生成**。模型内部维持 truth margin（L20-28 写入）同时保持输出分布不受其劫持（L32 擦除）——内部信念状态与输出策略在架构上解耦。**信念-生成解耦。L32 擦除=解耦机制。**

### 4. 硬伤
① JS 可能被高频 token 主导（未做 per-pair top-token 分解）；② joint 只覆盖 MLP 通道（3113 的 head 集中未做联合消融）；③ ~81% 剩余贡献位置未定位（无 L12–L36 全层扫描）；④ erase 判决虽强（0.976）但只有分布层面，未验证自由生成文本行为；⑤ 单模型。

### 5. 机制拼图更新
内部响应图谱新增：**因果写入拓扑=深度分布式+首层主导+条件性对抗（近可加）**；**信念-生成解耦**（truth margin 与 next-token 分布的架构级分离，L32 擦除为其执行件）。RDC 更新：模型的"知识状态"与"生成策略"是两个可分别干预的层面——对 AGI 理论：智能系统需要内部信念与输出行为的解耦层，LLM 用一个 MLP 层实现。

### 6. 3116 预注册（观测前冻结）
① **全层 MLP 消融扫描 L12–L36**（单点逐层，~17 条件）定位剩余 ~81% 贡献的真实位置，检验"深度分布式+首层主导"是否全域成立；② **信念-生成解耦行为验证**：自由生成下消融 L32 擦除，检查文本是否被 truth 词元劫持（分布→行为闭环）；③ 之后 T4 多步自回归。

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3115/omega_p113_joint_mlp_erase_purpose/`（result.json、design_seal.json、run_log.txt、ablation_readout.npz）；脚本 `tests/glm5/phase3115_omega_p113_joint_mlp_erase_purpose.py`。
'''
    sec = sec.replace('[[NOW]]', NOW)
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars (Phase 3115)' % len(sec))
else:
    o.append('memo already appended')

# ---------- workspace logs ----------
line_d = ('- Phase 3115 Omega-P113 (joint MLP ablation + '
          'erase-purpose JS readout, qwen3-4b, 5x2016 '
          'forwards, 364s): verdict write_joint_partial|'
          'additive|erase_serves_generation. (1) L20 '
          'single-point dmp_rel -0.3576 = LARGEST causal '
          'effect though smallest 3113 ds (+0.503) - '
          'correlation vs causality REVERSES along '
          'depth. (2) Joint L20+L24+L28 MLP only '
          '-0.1879 (gate -0.30 unmet) -> the three '
          'write-peak MLPs carry ~19% of margin '
          'causally; writing is DEEP DISTRIBUTED; '
          'joint ~= single_sum (-0.0119, additive; '
          'L24 conditional adversariality cancels). '
          '(3) ERASE PURPOSE STRONG: ablating L32 '
          'erase raises per-pair next-token JS(P||A1) '
          '0.1593 -> 0.2128 (sign rate 0.976, top-10 '
          'Jaccard down) -> L32 erase suppresses truth '
          'leak into generation = BELIEF-GENERATION '
          'DECOUPLING. m_consistency vs 3113 = 0.0 '
          '(bit-exact). NEXT 3116: full-layer MLP '
          'sweep L12-L36 to localize the ~81% '
          'remainder + free-generation behavioral '
          'check of decoupling; then T4.\n')
for wdir in (WLOG_D, WLOG_C):
    wl = wdir + '\\' + TODAY + '.md'
    try:
        prev = io.open(wl, encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3115 Omega-P113' not in prev:
        try:
            with io.open(wl, 'a',
                         encoding='utf-8') as f:
                f.write(line_d)
            o.append('wlog appended %s' % wl)
        except Exception as e:
            o.append('wlog fail %s: %r' % (wl, e))
    else:
        o.append('wlog already %s' % wl)

# ---------- MEMORY.md update ----------
mem_old = io.open(MEMO_W, encoding='utf-8').read()
if 'max=3115' not in mem_old:
    mem_new = mem_old.replace(
        '## 机制链状态（3114）\n'
        '- 3114：因果消融 head_write_not_causal|'
        'mlp_write_partial|erase_not_active。相关≠因果：'
        'L28 top8 head 消融 margin 不变（+0.0101）；'
        'L28 MLP 部分（−0.164）；**L24 消融反增 '
        '+0.346=对抗性写入**；L32 擦除 +0.103 半证实。'
        '自检 lin=0 精确。\n'
        '- 3113：伪迹分离+写入端：配对内中位 0.952 ≥ '
        '跨对 0.920（非伪迹）；3106 复现 0.916；MLP '
        '主写 L20–28（峰 +2.48）、L28 head 集中'
        '（0.556）、L32 负写 −3.97=擦除相。',
        '## 机制链状态（3115）\n'
        '- 3115：联合消融 write_joint_partial|additive|'
        'erase_serves_generation。**L20 单点 −0.358='
        '因果最大层**（3113 ds 却最小→相关因果沿深度'
        '反转）；L20-28 三层 MLP 联合仅 −0.188='
        '**写入深度分布式（~81% 在别处）**且 additive；'
        '**L32 擦除目的性强证实：JS 0.159→0.213 '
        '(sign 0.976)=信念-生成解耦**。\n'
        '- 3114：head 消融不变（+0.0101，head=相关'
        '影子）；L28 MLP −0.164 部分；L24 单点 '
        '+0.346=条件性对抗写入。\n'
        '- 3113：配对内中位 0.952≥跨对 0.920（非'
        '伪迹）；MLP 主写 L20–28（相关峰 +2.48）、'
        'L32 负写 −3.97=擦除相。')
    mem_new = mem_new.replace(
        'max=3114', 'max=3115').replace(
        '下一 3115：**L20–28 联合 MLP 消融（写入总能力'
        '上界）+ argnext KL 读出重测擦除目的性**→ '
        '之后 T4 多步自回归。',
        '下一 3116：**全层 MLP 消融扫描 L12–L36（定位 '
        '~81% 剩余贡献）+ 自由生成行为验证解耦**→ '
        '之后 T4 多步自回归。')
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory updated %d chars' % len(mem_new))
else:
    o.append('memory already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
