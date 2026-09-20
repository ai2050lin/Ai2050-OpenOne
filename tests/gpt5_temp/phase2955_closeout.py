# -*- coding: utf-8 -*-
"""Phase 2955 closeout: seal + Ledger + MEMO + worklog + MEMORY.
Idempotent, placeholder replacement."""
import hashlib
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2955', 'qk_source_decomposition')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-19.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2955_qk_source_decomposition.py')

res = json.load(open(os.path.join(OUT, 'result.json'),
                     encoding='utf-8'))
verdict = res['final_verdict']
created = json.load(open(os.path.join(OUT, 'execution.json'),
                         encoding='utf-8'))['created']


def sha8(path):
    return hashlib.sha256(
        open(path, 'rb').read()).hexdigest()[:8]


npz_name = 'qk_source_decomposition.npz'
hashes = {
    'execution.json': sha8(os.path.join(OUT, 'execution.json')),
    'result.json': sha8(os.path.join(OUT, 'result.json')),
    npz_name: sha8(os.path.join(OUT, npz_name)),
    'script': sha8(SCRIPT),
}

# ---------------- Ledger ----------------
led = json.load(open(LEDGER, encoding='utf-8'))
meas_id = 'meas2955_qk_source_decomposition'
if not any(m['meas_id'] == meas_id
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': meas_id,
        'type': 'forward_family_qk_decomposition',
        'verdict': verdict,
        'source': ('phase2955/qk_source_decomposition; '
                   'A11 gain = large logit shift (med|dz|~3, '
                   'same order as |z_b|), cross term dq.ddk '
                   'largest at both layers - not softmax '
                   'steep-region gain, not pure q/k direct'),
    })
    l14 = [lk for lk in led['linkage']
           if lk['link_id'] == 'L14_readout_spectrum_cross_model'][0]
    if meas_id not in l14['connects']:
        l14['connects'].append(meas_id)
led.pop('ledger_sha256_8', None)
new_h = hashlib.sha256(json.dumps(
    led, sort_keys=True, ensure_ascii=False)
    .encode('utf-8')).hexdigest()[:8]
led['ledger_sha256_8'] = new_h
json.dump(led, open(LEDGER, 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)
chk = new_h

# ---------------- MEMO ----------------
memo = open(MEMO, encoding='utf-8').read()
if '## Phase 2955' not in memo:
    section = '''## Phase 2955: q·k 来源分解判决 [@STAMP@]

**判决：`@VERDICT@`** —— 2952 遗留问题（A11 ×7-10 路由增益是注入对 q/k 的直接改动还是 softmax 陡区增益）的决定性回答：**是大 logit 域位移，不是 softmax 增益**；且 logit 位移的最大分量是 **Δq·Δk 交叉项**（纯 q 侧或纯 k 侧线性归属均不成立）。

**设计（一次前向族，runtime @RT@s）**：base×2 + L17@s1.0 + L17@s0.5 + L16@s2.0；剂量层 self_attn pre-hook 捕获层输入残差 pos0+pos1（注入后），fp64 重算链 q_norm/k_norm+RoPE+1/√HD 缩放，a16 输出空间验证链正确（dA_med max 8.2e-04）；精确 logit 分解 Δz = Δq·δ̃ + q̃·Δδ̃ + Δq·Δδ̃（恒等式残差 1.4e-14）。头集口径：全 32 头（教训 24）。锚 **13/13**：a1 2.17e-08（**第 26 次连续前向锚定**）、a11 dsc vs 2952 **bit 0**、a13 sep vs 2945 **bit 0**、a15 A11b vs 2953 **bit 0**、a-iso 因果隔离 **bit 0**（dx0 与上游层 dx1 全零）。

**主检验**：
- T1 源轴（Q/K/X = 逐头中位 |项| 的头中位，全 32 头）：L17 1.82/3.08/**3.88** → qk_mixed；L16 2.12/2.44/**4.06** → qk_mixed——**交叉项在两层均为最大**，路由跳变是非线性 q×k 现象
- T2 域轴：med|z_b| 2.85/3.39，med|Δz| 2.74/3.10——**logit 位移与基底 logit 同阶**（非陡区小位移放大），双层一致 → large_logit
- D1 预测（描述）：spearman(med|Δz|_h, ATT_h@2952) = −0.113（perm p95 0.350，不显著）——**第 18 层解耦：逐头 logit 位移幅度不预测该头的 ATT 承载**

**解剖（D2，L17@s1.0 top-ATT 头 + 早翻转头）**：h20/h21（2947 抵抗头）Δz 高达 **14.0/18.6**（A11 0.030/0.015 → 1.000，完全翻转），且其 Mx（交叉项）11.8/19.3 支配；h0 Δz 8.5（0.013→0.981）；h22 仅部分翻转（0.010→0.370，Δz 4.0）。基底 z 全体强负（−1.8~−4.6：基线注意力读 pos0 功能词），注入把 z 推正。s=0.5 早窗同构（X=1.60 ≥ Q=1.18，med|Δz|=1.29）。**A 域倍率 ×7-10 的分子分母审计（纪律 16）**：增益主要由极小分母驱动（A11_b 0.01-0.14），绝对量 0.01→1.0 的翻转才是本体——"×7-10"是倍率口径陈述。

**结论（重复 3 次）**：**A11 路由跳变由与基底同阶的大 logit 位移驱动（非 softmax 陡区放大），位移由 Δq·Δk 交叉项主导（非线性交互，线性快照归属低估），且逐头 logit 位移与逐头 ATT 承载解耦——路由增益是真实的大重写事件，"增益"倍率是分母伪影。与 2948/2952 合读：W_ov 线性秩序承载读出内容，q·k 非线性交叉承载路由开关，两者是不同模块的不同层级。**

**硬伤与勘误（4 轮运行，1 个真错误 + 2 个判据迭代）**：run1 GQA 广播错误（z einsum 出 KV 轴 8 维，须按 h//4 逐头展开）；run2 撞 a16（0.062>0.05）→ 诊断出**真链错误：遗漏 softmax 1/√HD 缩放**（run3 比值 318.7 抓获，修正后降至 5.8）；v3 比值判据 2.0 失准（逐词逐头 LS 恢复把 r_rec 过拟合至 3e-5，固定比值阈值无意义）→ a16 v4 回归 dA_med<0.05（正链 8e-4，60 倍裕度；错链 0.062，判别力已证）。**教训 26（工程）：attention logits = q·k/√HD 必须缩放；fp64 重算链必须配输出空间验证锚（对 LS 最优底做比值判据会因过拟合失准）；GQA 逐头量必须先展开 KV 轴。** Ledger 94 条 / L14 connects 62 / ledger @LEDHASH@。

**文件+SHA256-8**：execution @HEXE@ / result @HRES@ / npz @HNPZ@ / script @HSCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2955/qk_source_decomposition/`。

**接续**：机制链第十八环（路由来源环）闭合。候选 2956：A（主选）重平衡时间定位（2954 遗留：消融诱发的竞争重平衡在下游哪类模块发生——MLP vs attention 逐层差分，一次前向族）；B 交叉项代数结构（Δq·Δk 主导的低秩/方向结构：交叉项是否可由注入方向预测——零前向+一次前向）；C 承重带跨模型复现（glm4）；D v3 解码器方向重启（2940 遗留）。
'''
    stamp = created.replace('T', ' ')[:16]
    reps = [('@STAMP@', stamp), ('@VERDICT@', verdict),
            ('@RT@', str(res['runtime_s'])),
            ('@LEDHASH@', chk),
            ('@HEXE@', hashes['execution.json']),
            ('@HRES@', hashes['result.json']),
            ('@HNPZ@', hashes[npz_name]),
            ('@HSCR@', hashes['script'])]
    for k, v in reps:
        section = section.replace(k, v)
    memo += section
    open(MEMO, 'w', encoding='utf-8').write(memo)

# ---------------- worklog ----------------
wl = open(WLOG, encoding='utf-8').read()
if 'Phase 2955' not in wl:
    wl += ('- Phase 2955 闭环：qk_mixed_large_logit。A11 ×7-10 '
           '路由增益 = 大 logit 域位移（med|Δz| 2.7-3.1 与基底 '
           '|z_b| 2.8-3.4 同阶），非 softmax 陡区放大；位移最大'
           '分量是 Δq·Δk 交叉项（两层均然，线性归属不成立）；'
           '逐头 logit 位移与 ATT 承载解耦（rho −0.11 ns）；'
           '×7-10 倍率是小分母伪影（A11_b 0.01→1.0 绝对翻转）。'
           '勘误：真链错误 = 遗漏 1/√HD softmax 缩放（a16 输出'
           '空间锚 318.7 抓获）；GQA 逐头展开 + LS 底过拟合教训。'
           'Ledger 94 / hash ' + chk + '。\n')
    open(WLOG, 'w', encoding='utf-8').write(wl)

# ---------------- MEMORY ----------------
mem = open(WMEM, encoding='utf-8').read()
if '当前 max=**2954**' in mem:
    mem = mem.replace(
        '当前 max=**2954**，下一个 **2955**（候选 A q·k 来源分解）',
        '当前 max=**2955**，下一个 **2956**（候选 A 重平衡时间定位）')
if 'attention logits = q·k/√HD' not in mem:
    anchor_line = '- 聚合头集口径门：'
    add = ('- attention 重算链规范（2955）：logits = q·k/√HD 必须'
           '缩放（遗漏→a16 比值 318 抓获）；fp64 重算链（q_norm/'
           'k_norm/RoPE/GQA 逐头展开）必须配输出空间验证锚；对 '
           'LS 最优底做固定比值判据会因过拟合失准（逐词 LS 把 '
           'r_rec 压到 3e-5）——用 dA 中位差判据；科学结论：A11 '
           '路由增益 = 大 logit 位移（Δq·Δk 交叉项主导），非 '
           'softmax 陡区放大，×7-10 是小分母伪影。\n')
    i = mem.find(anchor_line)
    if i >= 0:
        mem = mem[:i] + add + mem[i:]
    else:
        mem = mem.rstrip() + '\n' + add
open(WMEM, 'w', encoding='utf-8').write(mem)

print('closeout done: ledger=%s n_meas=%s' % (
    chk, len(led['measurements'])))
