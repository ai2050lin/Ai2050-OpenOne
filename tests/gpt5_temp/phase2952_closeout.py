# -*- coding: utf-8 -*-
"""Phase 2952 closeout: seal + Ledger + MEMO + worklog + MEMORY.
Idempotent, placeholder replacement."""
import hashlib
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2952', 'amplification_anatomy')
LEDGER = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-19.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2952_amplification_anatomy.py')

res = json.load(open(os.path.join(OUT, 'result.json'),
                     encoding='utf-8'))
verdict = res['final_verdict']
created = json.load(open(os.path.join(OUT, 'execution.json'),
                         encoding='utf-8'))['created']


def sha8(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:8]


npz_name = 'amplification_anatomy.npz'
hashes = {
    'execution.json': sha8(os.path.join(OUT, 'execution.json')),
    'result.json': sha8(os.path.join(OUT, 'result.json')),
    npz_name: sha8(os.path.join(OUT, npz_name)),
    'script': sha8(SCRIPT),
}

# ---------------- Ledger ----------------
led = json.load(open(LEDGER, encoding='utf-8'))
meas_id = 'meas2952_amplification_attention_gain'
if not any(m['meas_id'] == meas_id
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': meas_id,
        'type': 'forward_family_anatomy',
        'verdict': verdict,
        'source': ('phase2952/amplification_anatomy; '
                   'quasi-post-hoc per discipline 9'),
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
if '## Phase 2952' not in memo:
    t1 = res['T1_att_share']
    t2 = res['T2_att_gain_sorted']
    t3 = res['T3_a11_gain']
    d1 = res['D1_anatomy']
    anc = res['anchors']
    section = '''## Phase 2952: 放大来源解剖与注意力增益判决 @STAMP@

**判决：`@VERDICT@`** —— 2951 的 beta~1.5x 放大主体是**注意力自权重增益**，不是值通路 LN 增益：注入使 A11(pos1 自注意) 从 ~0.03-0.05 暴涨至 ~0.27-0.31（x6.8/x9.6），ATT 项承载 90-93% 的逐头位移，且 ATT 谱与 W_ov 增益 g 强排序（rho 0.79-0.81）——同时解释了 2951 的两大谜题（为何 beta>1、为何 Delta 谱 g 排序）。

**设计（一次前向族，runtime @RT@s）**：base + L17@s1.0 + L16@s2.0 注入（无消融）；捕获 v_proj 输出 pos0/pos1 与 o_proj 输入 pos1。2-token 提示下 pos1 只注意 pos0/pos1，故 A11 由线模型 x_h = v0 + A11(v1-v0) 逐词逐头最小二乘恢复（v 无 RoPE，无需复刻 q/k 路径）。精确恒等式 dx_h = dA11(v1n-v0) + A11b(v1n-v1b)，sep 投影后 VAL+ATT == Delta（a7 恒等式 1.85e-04/6.99e-04）。捕获链锚：sc_base vs 2950 sc_B1 = 6.9e-08、delta vs 2951 = 6.9e-08（bit 级）。

**主检验（T1/T2/T3 全过，quasi-post-hoc 纪律 9）**：
- T1 注意力增益主导：|ATT|/(|VAL|+|ATT|) 中位 @S17@ (L17) / @S16@ (L16)，全 > 0.6
- T2 ATT 谱 gain 排序：rho(ATT,g) = @R17@ (L17, p95 @P17@) / @R16@ (L16, p95 @P16@)
- T3 A11 增益：@A17B@ -> @A17N@ (ratio @RAT17@) / @A16B@ -> @A16N@ (ratio @RAT16@)

**解剖（D1）**：mu_val（VAL/g 中位）仅 @MU17@ / @MU16@——值通路（LN Jacobian 门控）只贡献 ~7-12%；rho(ATT,R) = @AR17@ / @AR16@。**机制画像：注入把 pos-1 的 q/k 推向自注意（A11 x7-10），头输出从"几乎纯 pos0（功能词）值"切换为"大量混入 pos1（注入词）值"——放大 = 自注意权重跳变 x 词值差向量，这就是浓度开关（2945-2946）的头级微观载体候选。**

**结论（重复 3 次）**：2951 线性秩序（rho(g,Delta) 0.82-0.85）的物理来源是注意力增益项且该项本身 g 排序（rho 0.79-0.81）——**"开关"在头级是注意力路由跳变：头从读功能词切换到读注入词**。2942 的跨 session 中增益区不稳定（L16@2）与 A11 的 sigmoid 型跳变自洽：路由跳变中点对扰动敏感。2949/2950 组消融不可操作化维持——消融改变的是路由竞争的输入侧。

**硬伤与勘误**：run1 decoder-layer 级 with_kwargs pre-hook 使 transformers forward 崩溃（RoPE 形状错位）——改用 self_attn 级 pre-hook 捕获层输入（2950 verbatim 安全路径）；run3 pass1 忘加 batch 维（1-D 输入 -> 2-D hidden_states）；run4 recover() den 用全词 max 标量（应为逐词）致 A11 低估、a7/a8 锚失败——run5 权威。教训 23 入 MEMORY。锚：a1 2.17e-08（**第 23 次连续前向锚定**）、a2 bit 0、a3 9.95e-14、a9 确定性 0。Ledger 91 条 / L14 connects 59 / ledger @LEDHASH@。

**文件+SHA256-8**：execution @HEXE@ / result @HRES@ / npz @HNPZ@ / script @HSCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2952/amplification_anatomy/`。

**接续**：机制链第十五环（放大来源环）闭合。候选 2953：A 注意力路由跳变的 q·k 来源（零捕获分析：注入对 q/k 的直接改动 vs softmax 增益，一次前向）；B A11 跳变 vs s 的 sigmoid 拟合（细扫 s，预测 2945 阈值曲线的微观重现）；C 承重带跨模型复现（glm4）；D v3 解码器方向重启。
'''
    reps = [('@STAMP@', created), ('@VERDICT@', verdict),
            ('@RT@', str(res['runtime_s'])),
            ('@S17@', str(t1['L17']['att_share_med'])),
            ('@S16@', str(t1['L16']['att_share_med'])),
            ('@R17@', str(t2['L17']['rho_att_g'])),
            ('@P17@', str(t2['L17']['null_p95'])),
            ('@R16@', str(t2['L16']['rho_att_g'])),
            ('@P16@', str(t2['L16']['null_p95'])),
            ('@A17B@', str(t3['L17']['a11_base_med'])),
            ('@A17N@', str(t3['L17']['a11_inj_med'])),
            ('@RAT17@', str(t3['L17']['ratio'])),
            ('@A16B@', str(t3['L16']['a11_base_med'])),
            ('@A16N@', str(t3['L16']['a11_inj_med'])),
            ('@RAT16@', str(t3['L16']['ratio'])),
            ('@MU17@', str(d1['L17']['mu_val'])),
            ('@MU16@', str(d1['L16']['mu_val'])),
            ('@AR17@', str(d1['L17']['rho_att_R'])),
            ('@AR16@', str(d1['L16']['rho_att_R'])),
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
if 'Phase 2952' not in wl:
    wl += ('- Phase 2952 闭环：amplification_attention_gain_sorted'
           '。beta~1.5x 放大主体 = 注意力自权重增益（A11 x7-10，'
           'ATT 项 90-93%，rho(ATT,g) 0.79-0.81），值通路仅 ~7-12%；'
           '机制 = 头读功能词切换到读注入词。勘误：decoder-layer 级 '
           'hook 危险、batch 维包裹、den 逐词。Ledger 91 / hash '
           + chk + '。\n')
    open(WLOG, 'w', encoding='utf-8').write(wl)

# ---------------- MEMORY ----------------
mem = open(WMEM, encoding='utf-8').read()
if '当前 max=2951' in mem:
    mem = mem.replace('当前 max=2951，下一个 2952',
                      '当前 max=2952，下一个 2953')
if '23. **2952（hook 挂点与捕获规范）**' not in mem:
    lesson = ('23. **2952（hook 挂点与捕获规范）**：qwen3+transformers '
              '新版的 decoder-layer 级 with_kwargs pre-hook 会破坏 '
              'forward（RoPE 形状错位）——层输入捕获一律用 self_attn '
              '级 pre-hook（其输入=层输入残差，已验证安全）；单样本前向 '
              '必须保持 batch 维（torch.tensor([toks]) 而非 (toks)）；'
              '逐词最小二乘的 den 禁用跨词聚合标量。科学结论：beta~1.5x '
              '放大 = 注意力自权重增益（A11 x7-10，ATT 项 90-93% 且 g '
              '排序 rho 0.79-0.81），开关的头级微观载体 = 注意力路由'
              '跳变（头从读功能词切到读注入词）。\n')
    i21 = mem.find('22. **2951')
    if i21 >= 0:
        iend = mem.find('\n', mem.find('\n', i21) + 1)
        mem = mem[:iend + 1] + lesson + mem[iend + 1:]
    else:
        mem = mem.rstrip() + '\n' + lesson
open(WMEM, 'w', encoding='utf-8').write(mem)

print('closeout done: ledger=%s n_meas=%s' % (
    chk, len(led['measurements'])))
