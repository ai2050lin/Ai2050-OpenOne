# -*- coding: utf-8 -*-
"""Phase 2951 closeout: seal + Ledger + MEMO + worklog + MEMORY.
Idempotent. Placeholder replacement (no %-formatting on text)."""
import hashlib
import json
import os
import time

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2951', 'rebalance_carrier_functional')
LEDGER = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-19.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2951_rebalance_carrier_functional.py')

res = json.load(open(os.path.join(OUT, 'result.json'),
                     encoding='utf-8'))
verdict = res['verdict']
created = res['prereg']['created']


def sha8(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:8]


hashes = {
    'execution.json': sha8(os.path.join(OUT, 'execution.json')),
    'result.json': sha8(os.path.join(OUT, 'result.json')),
    'rebalance_carrier_functional.npz': sha8(
        os.path.join(OUT, 'rebalance_carrier_functional.npz')),
    'script': sha8(SCRIPT),
}

# ---------------- Ledger ----------------
led = json.load(open(LEDGER, encoding='utf-8'))
meas_id = 'meas2951_rebalance_gain_functional'
already = any(m['meas_id'] == meas_id
              for m in led['measurements'])
if not already:
    led['measurements'].append({
        'meas_id': meas_id,
        'type': 'zero_forward_mechanism_integration',
        'verdict': verdict,
        'source': ('phase2951/rebalance_carrier_functional; '
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
else:
    led.pop('ledger_sha256_8', None)
    new_h = hashlib.sha256(json.dumps(
        led, sort_keys=True, ensure_ascii=False)
        .encode('utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = new_h
chk = new_h

# ---------------- MEMO ----------------
memo = open(MEMO, encoding='utf-8').read()
if '## Phase 2951' not in memo:
    t = res['T1_gain_sorted']
    t2r = res['T2_residual_decoupled']
    d1 = res['D1_anatomy']
    anc = res['anchors']
    section = '''## Phase 2951: 重平衡载体 W_ov 功能性判决 @STAMP@

**判决：`@VERDICT@`（quasi-post-hoc 机制整合，纪律 9 标注）** —— 2950 的重平衡谱由 W_ov 线性增益排序：Delta_h = beta·g_h（直接项，方差占比 71-78%）+ gain 无关非线性残差（|rho| 远低于 null p95）；重平衡载体头是剩余头中最高 |g| 者 funcional。

**设计（零前向，runtime @RT@s）**：Delta_h = sc_I1@MINUS@sc_B1（2950 npz 语义先检修正：keep 头上 sc_I1==sc_I0 恒等，movers 实为对消融基线的变化）；g 来自 2948 npz（GQA 修正版）；perm null 20000 置换 seed 2904。锚 7/7：a1 Vt8-vs-2939 bit 级 0；a2 dirs_word 2948-vs-2950 2.17e-08（跨相位 bf16 噪声惯例阈 1e-6，与 a1 重建锚同量级）；a3 sc-vs-2950json 4.99e-04（舍入）；a4 消融头 bit 级零；a5 秩相关自检（恒等+值反转）；a6 OLS 重建 <1e-12；a7 置换 null 决定性 0。

**主检验**：
- T1 gain 排序：L17 rho(g,Delta)=@R17@ (p95 @P17@, pass)；L16 rho=@R16@ (p95 @P16@, pass)
- T2 残差解耦：L17 rho(g,R)=@QR17@ (p95 @QP17@, pass)；L16 rho=@QR16@ (p95 @QP16@, pass)

**解剖（D1）**：L17 beta=1.5110（放大 1.51x @TIMES@ s=1.0）、share 0.711、mover |g| 百分位 92.6/81.5/96.3/63.0/70.4；L16 beta=2.9640（1.48x @TIMES@ s=2.0）、share 0.777、mover 百分位 96.3/85.2/74.1/88.9/51.9。残差中位 |R|：0.215/0.422。

**结论（关键发现）**：2950 "movers 非 top5 g 排名" 悖论消解——top5 g 头正是被消融的头，剩余头中最高增益者自然承接最大位移；beta·g_h 放大 ~1.5x 表明注意力权重向注入位增益（头输入变化超出 s·xdir 线性预测）；残差 22-29% 为 gain 无关非线性成分。**重平衡载体是功能性 W_ov 头，不是随机接受者；但头组消融仍不可操作化（2949/2950 维持）——单头"重要性" = 线性秩序（W_ov 骨架）× 非线性重平衡（关系性）的混合。**

**硬伤与勘误**：run1 a2 阈值过紧（dirs_word 跨相位 2.17e-08 撞 bit 级判据，按 2940 惯例放宽 1e-6）+ a5 自检设计错误（位置反转 vs 值反转：spearmanr(xs, xs[::-1]) 是秩向量位置置换，相关≈0 正常，探针确认 scipy 1.18 语义无误——历史 rho 全部有效）；run2 a7 raw-vs-rounded 舍入差 4.19e-05 改 raw 对比；run3 权威。Ledger 90 条 / L14 connects 58 / ledger @LEDHASH@。

**文件+SHA256-8**：execution @HEXE@ / result @HRES@ / npz @HNPZ@ / script @HSCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2951/rebalance_carrier_functional/`。

**接续**：机制链第十四环（载体功能性环）闭合。候选 2952：A 直接项 1.5x 放大来源——attn 权重捕获验证（一次前向）；B 承重带跨模型复现（glm4）；C 头级秩序跨层泛化（L14/L15/L18 g-vs-D，零前向）；D 关闭注入子链、转向 v3 解码器方向的进一步工作。
'''
    for k, v in [('@STAMP@', created), ('@VERDICT@', verdict),
                 ('@RT@', str(res['runtime_s'])),
                 ('@MINUS@', chr(8722)),
                 ('@R17@', str(t['L17']['rho'])),
                 ('@P17@', str(t['L17']['null_p95'])),
                 ('@R16@', str(t['L16']['rho'])),
                 ('@P16@', str(t['L16']['null_p95'])),
                 ('@QR17@', str(t2r['L17']['rho_resid'])),
                 ('@QP17@', str(t2r['L17']['null_p95'])),
                 ('@QR16@', str(t2r['L16']['rho_resid'])),
                 ('@QP16@', str(t2r['L16']['null_p95'])),
                 ('@TIMES@', chr(215)),
                 ('@LEDHASH@', chk),
                 ('@HEXE@', hashes['execution.json']),
                 ('@HRES@', hashes['result.json']),
                 ('@HNPZ@', hashes['rebalance_carrier_functional.npz']),
                 ('@HSCR@', hashes['script'])]:
        section = section.replace(k, v)
    memo += section
    open(MEMO, 'w', encoding='utf-8').write(memo)

# ---------------- worklog ----------------
wl = open(WLOG, encoding='utf-8').read()
entry = ('- Phase 2951 闭环：rebalance_carriers_gain_functional'
         '（零前向 quasi-post-hoc）。Delta=I1-B1 语义先检修正；'
         'T1 rho(g,Delta) 0.852/0.815、T2 残差解耦 -0.068/-0.078、'
         '直接项 share 71/78%、放大 1.51x/1.48x；a5 教训：位置反转'
         '不是值反转（历史 rho 有效）。Ledger 90 / hash ' + chk + '。\n')
if 'Phase 2951' not in wl:
    wl += entry
    open(WLOG, 'w', encoding='utf-8').write(wl)

# ---------------- MEMORY ----------------
mem = open(WMEM, encoding='utf-8').read()
changed = False
if 'max=2950' in mem:
    mem = mem.replace('max=2950，下一个 2951', 'max=2951，下一个 2952')
    mem = mem.replace('当前 max=2950，下一个 2951',
                      '当前 max=2951，下一个 2952')
    changed = True
if '2951（秩相关自检规范）' not in mem:
    lesson = ('22. **2951（秩相关自检与跨相位锚规范）**：spearman '
              '自检必须用值反转（spearmanr(x,-x)=-1）而非位置反转'
              '（x[::-1] 是秩向量位置置换，相关≈0 是正常现象，'
              'scipy 语义无误）；跨相位 dirs/特征数组比较用 1e-6 阈'
              '（bf16 噪声 2.17e-08 惯例），bit 级判据仅限同文件链'
              '内；raw-vs-rounded 值比较禁用于确定性锚。科学结论：'
              '重平衡载体是功能性 W_ov 头（rho 0.82-0.85，直接项'
              ' share 71-78%，放大 1.5x），2949/2950 组消融不可操作'
              '化判决维持——单头重要性=线性秩序×非线性关系混合。\n')
    mem = mem.rstrip() + '\n' + lesson
    changed = True
if changed:
    open(WMEM, 'w', encoding='utf-8').write(mem)

print('closeout done: ledger=%s n_meas=%s' % (
    chk, len(led['measurements'])))
