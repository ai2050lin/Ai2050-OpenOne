# -*- coding: utf-8 -*-
"""Phase 3114 closeout (idempotent):
Ledger -> MEMO Phase 3114 -> workspace logs -> MEMORY.md."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3114'
        r'\omega_p112_write_erase_ablation')
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
    'head_write_not_causal|mlp_write_partial|' \
    'erase_not_active'
assert res['n_records'] == 2016
assert res['smoke'] is False
assert -0.15 < res['dmp_rel']['L28_top8_head'] < 0.15
assert -0.30 < res['dmp_rel']['L28_mlp'] <= 0.0
assert res['dmp_rel']['L32_mlp'] < 0.30
assert res['dmp_rel']['L24_mlp'] > 0.30
assert max(res['selfcheck_rel'].values()) < 0.01
assert res['top8_heads']['L28'] == \
    [30, 22, 31, 4, 20, 1, 21, 23]

# ---------- Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3114
           for m in led['measurements']):
    claim = (
        'Omega-P112 (3114, causal ablation of the 3113 '
        'write/erase structure on rebuilt 3105 material, '
        'qwen3-4b, 2016 records x 5 conditions, 320s) - '
        'verdict head_write_not_causal|mlp_write_partial|'
        'erase_not_active.  CORRELATION IS NOT CAUSALITY '
        'on the write side: zeroing the o_proj input '
        'blocks of the 3113-frozen top-8 |ds| heads at '
        'L28 ([30,22,31,4,20,1,21,23], 55.6% of head '
        'write) leaves the truth margin essentially '
        'UNCHANGED (mpair 8.2458 -> 8.3294, dmp_rel '
        '+0.0101 vs gate -0.15; linear prediction '
        '+0.0316 confirms sign/magnitude of the tiny '
        'effect) -> the concentrated head write is a '
        'correlational structure, not a unique causal '
        'channel.  L28 MLP ablation PARTIALLY causal '
        '(dmp_rel -0.1644, gate -0.30 not met) - about '
        'half of its correlational share (2.478/8.246'
        '=30%).  L32 MLP erase ablation shifts margin '
        '+0.1030 (gate +0.30 not met; erase direction '
        'consistent with 3113 ds -3.970 but only ~1/5 '
        'of the correlational size -> downstream '
        'layers 33-36 partially compensate); purpose '
        'test uninformative (yes/no softmax probs '
        'float near 0; yes_prob shift -1.2e-6).  NEW '
        'PUZZLE - ADVERSARIAL WRITE: ablating the L24 '
        'MLP (3113 ds +1.673 second write peak) '
        'INCREASES the margin +0.346 -> the L24 write '
        'opposes the final margin causally; the '
        'correlational climb L20->L28 is the NET of '
        'multi-layer adversarial writes, not a '
        'monotone causal pipeline.  Self-checks: '
        'o_proj linear identity EXACT (lin=0.0), '
        'residual identity under ablation rel L2 '
        '2.0e-3, all 4 conditions < 0.01.  CAVEATS: '
        '(i) single-point ablations cannot bound '
        'cross-layer redundancy (no L20-28 joint '
        'ablation yet); (ii) erase purpose test has '
        'no resolution at near-zero yes/no probs; '
        '(iii) single model.  NEXT 3115: joint L20-28 '
        'MLP ablation (total write capacity upper '
        'bound) + readout change (argnext '
        'distribution KL instead of yes/no probs) '
        'for the erase purpose test; then T4 '
        'multi-step autoregression.')
    meas = {
        'meas_id': 'meas3114_omega_p112_'
                   'write_erase_ablation',
        'phase': 3114,
        'claim': claim,
        'verdict': 'head_write_not_causal|'
                   'mlp_write_partial|'
                   'erase_not_active',
        'anchors': 'design_seal.json frozen before '
                   'computation: heads frozen from 3113 '
                   'result.json L28 ds_head_order[:8]; '
                   'gates pre-registered head dmp_rel '
                   'le -0.15 / mlp le -0.30 / erase '
                   'ge +0.30; selfcheck identity '
                   'rel L2 < 0.02; L24 head ablation '
                   'dropped (3113 has per-head only '
                   'for L28) - L24 enters as MLP-only '
                   'control',
        'artifacts': {
            'result': 'phase3114/omega_p112_'
                      'write_erase_ablation/'
                      'result.json',
            'seal': 'phase3114/omega_p112_'
                    'write_erase_ablation/'
                    'design_seal.json',
            'readout': 'phase3114/omega_p112_'
                       'write_erase_ablation/'
                       'ablation_readout.npz'},
        'hashes': {},
        'note': 'GPU used (qwen3-4b, BF16, eager, '
                'batch 1, 320s for 5x2016 forward '
                'passes); selfchecks 1.7e-3..2.3e-3; '
                'AUC_truth(m) 0.9885-0.9905 across '
                'conditions (classification '
                'preserved)',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3114_omega_p112_write_erase_ablation')
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

# ---------- MEMO Phase 3114 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3114:' not in memo:
    sec = u'''## Phase 3114: Ω-P112 写/擦两相因果消融——L28 top8 head 消融 margin 不变（+0.0101，head_write_not_causal：相关≠因果）；L28 MLP 部分因果（−0.164，mlp_write_partial）；L32 擦除未激活（+0.103，erase_not_active）；L24 MLP 消融反增 margin +0.346（对抗性写入，新拼图） [[NOW]]

**性质**：T3 第 8 Phase。因果检验 3113 的三点结构（head 集中写入 / MLP 写入 / L32 擦除）。qwen3-4b BF16，重建 3105 材料 2016 记录（672 对）× 5 条件（baseline + L28 top8 head + L24/L28/L32 MLP 消融），319.8s。预注册（design_seal.json 先于计算）：seal 冻结 3113 result.json 的 L28 `ds_head_order[:8]`=[30,22,31,4,20,1,21,23]（3113 仅 L28 有 per-head 分解，L24 以 MLP-only 控制条件进入）；门：head dmp_rel ≤−0.15 → head_write_causal；mlp ≤−0.30 → mlp_write_dominant；erase ≥+0.30 → erase_active。

### 1. 方法与自检
干预语义按 3101 教训（先逆向再复刻）：head 消融=o_proj **输入侧**对应 head 列块置零（经 o_proj 精确线性移除）；MLP 消融=输出置零。读出 m'=(W_yes−W_no)·h_fn(last)（=yes−no logit margin，lm_head 无 bias 已断言）+ yes/no softmax + argnext。**三重恒等自检**（baseline 首记录存 clean 快照，消融条件首记录验证）：①置零块范数=0；②o_proj 线性恒等 o_proj(attn_abl)=o_proj(attn_clean)−o_proj(removed)（实测 **lin=0.0 精确**）；③消融下残差恒等 h_out=h_in+o_proj(attn)+mlp。4 条件 rel L2 全部 1.7e-03–2.3e-03 < 0.01。

### 2. 结果（2016 记录，672 对）

| 条件 | mpair | dmp_rel | 门判决 |
| --- | --- | --- | --- |
| baseline | 8.2458 | — | — |
| abl_L28_top8_head | 8.3294 | **+0.0101**（线性预测 +0.0316，同号同量级）| head_write_not_causal |
| abl_L28_mlp | 6.8898 | **−0.1644** | mlp_write_partial |
| abl_L24_mlp | 11.0991 | **+0.3460** | （对抗性写入）|
| abl_L32_mlp | 9.0950 | **+0.1030** | erase_not_active |

AUC_truth(m) 各条件 0.9885–0.9905（分类能力保持——margin 整体存在，变的是配对差幅度）。

### 3. 三大发现（重复三遍）
1. **相关 ≠ 因果**：3113 的 L28 top8 head 集中（份额 0.556）是观察性结构，不是唯一因果通道——因果移除后 margin 几乎不变（+0.0101 vs 门 −0.15）。**相关 ≠ 因果：head 集中写入被否定为独立机制。相关 ≠ 因果。**
2. **L24 对抗性写入（新拼图）**：3113 中 ds=+1.673 的"L24 第二写入峰"在因果上是**负贡献**——移除后 margin 反升 +0.346。相关轨迹 L20→L28 的"爬升"是**多层对抗性写入的净和**，不是单调因果管线。**L24 写的是对抗方向：移除它 margin 反而增大。L24 对抗性写入。**
3. **擦除半证实**：L32 MLP 消融 margin +0.103（方向与 3113 ds=−3.97 一致）但量级仅相关分解的 ~1/5——下游层 33–36 部分补偿；erase 未过门。**L32 擦除方向对、但因果量级小，下游补偿。擦除半证实。**

### 4. 硬伤
① yes/no softmax 概率全程 ~0（yes_prob shift −1.2e-6），预注册的"擦除服务下一步生成"目的性检验**无分辨力**——需换读出端；② 单点消融不能上界跨层冗余（未做 L20–28 全 MLP 联合消融）；③ 下游补偿路径未直接测量（推断自 +0.103 vs −3.97/8.25=+0.48 相关预期）；④ A2 记录的消融响应在 ablation_readout.npz 中未分析；⑤ 单模型。

### 5. 机制拼图更新
内部响应图谱新增：**写入因果拓扑=对抗性+冗余分布式**（非单点、非单调）。RDC 更新：truth margin 是多层净竞争的结果——L24 写入对抗方向、L28 MLP 写入部分支撑（−16%）、head 集中是相关影子、L32 擦除被下游补偿。观察性 ds 分解（3113）仍有效于"哪里有信号"，但不能预测"移除后发生什么"（3113 硬伤③现已有答案：擦除因果量级 1/5）。

### 6. 3115 预注册（观测前冻结）
① **联合消融**：L20–28 全部 MLP 一次移除 → 写入总能力上界；若 margin 崩溃 → 冗余分布式证实；② **读出端更换**：erase 目的性改用 argnext 分布变化（KL/Top-k 位移）而非 yes/no 概率；③ 之后 T4 多步自回归 + 写入端条件化结构。

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3114/omega_p112_write_erase_ablation/`（result.json、design_seal.json、run_log.txt、ablation_readout.npz）；脚本 `tests/glm5/phase3114_omega_p112_write_erase_ablation.py`。
'''
    sec = sec.replace('[[NOW]]', NOW)
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars (Phase 3114)' % len(sec))
else:
    o.append('memo already appended')

# ---------- workspace logs ----------
line_d = ('- Phase 3114 Omega-P112 (causal ablation, '
          'qwen3-4b, 5x2016 forwards, 320s): verdict '
          'head_write_not_causal|mlp_write_partial|'
          'erase_not_active. CORRELATION IS NOT '
          'CAUSALITY: L28 top8 head ablation (frozen '
          'from 3113, 55.6% head-write share) leaves '
          'margin unchanged (+0.0101 vs gate -0.15; '
          'linear pred +0.0316) -> head write is a '
          'correlational shadow, not a causal channel. '
          'L28 MLP -0.1644 (partial). L24 MLP ablation '
          'INCREASES margin +0.346 -> ADVERSARIAL '
          'WRITE: the L20->L28 correlational climb is '
          'the net of multi-layer opposing writes. '
          'L32 erase +0.1030 (direction consistent '
          'with ds -3.97, size ~1/5 -> downstream '
          'compensation); erase purpose test void '
          '(yes/no probs ~0, shift -1.2e-6). '
          'Self-checks: o_proj linear identity exact '
          '(0.0), residual rel L2 <= 2.3e-3. NEXT '
          '3115: joint L20-28 MLP ablation + argnext '
          'KL readout for erase purpose; then T4.\n')
for wdir in (WLOG_D, WLOG_C):
    wl = wdir + '\\' + TODAY + '.md'
    try:
        prev = io.open(wl, encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3114 Omega-P112' not in prev:
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
if 'max=3114' not in mem_old:
    mem_new = mem_old.replace(
        '## 机制链状态（3113）\n'
        '- 3113：伪迹分离+写入端判决 belief_robust|'
        'within_unit_replicated|write_in_concentrated。'
        '配对内（差 1 token）单坐标中位 0.952 ≥ 跨对 '
        '0.920——广播非伪迹；3106 unit 内 0.916 复现；'
        '写入端=MLP 主力 L20–28（峰 +2.48）+ L28 head '
        '集中（top8 0.556），L32 MLP 负写入 −3.97='
        '擦除相。槽位澄清：3112 L6=模型层 28。',
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
        '（0.556）、L32 负写 −3.97=擦除相。')
    mem_new = mem_new.replace(
        'max=3113', 'max=3114').replace(
        '下一 3114：**写/擦两相因果消融（L28 top8 head + '
        'L28/L32 MLP）**→ 之后 T4 多步自回归。',
        '下一 3115：**L20–28 联合 MLP 消融（写入总能力'
        '上界）+ argnext KL 读出重测擦除目的性**→ '
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
