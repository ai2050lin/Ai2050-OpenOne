# -*- coding: utf-8 -*-
"""Phase 2954 closeout: seal + Ledger + MEMO + worklog + MEMORY.
Idempotent, placeholder replacement."""
import hashlib
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2954', 'early_flipper_polarity')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-19.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2954_early_flipper_polarity.py')

res = json.load(open(os.path.join(OUT, 'result.json'),
                     encoding='utf-8'))
verdict = res['final_verdict']
created = json.load(open(os.path.join(OUT, 'execution.json'),
                         encoding='utf-8'))['created']


def sha8(path):
    return hashlib.sha256(
        open(path, 'rb').read()).hexdigest()[:8]


npz_name = 'early_flipper_polarity.npz'
hashes = {
    'execution.json': sha8(os.path.join(OUT, 'execution.json')),
    'result.json': sha8(os.path.join(OUT, 'result.json')),
    npz_name: sha8(os.path.join(OUT, npz_name)),
    'script': sha8(SCRIPT),
}

# ---------------- Ledger ----------------
led = json.load(open(LEDGER, encoding='utf-8'))
meas_id = 'meas2954_early_flipper_polarity'
if not any(m['meas_id'] == meas_id
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': meas_id,
        'type': 'forward_family_polarity_anatomy',
        'verdict': verdict,
        'source': ('phase2954/early_flipper_polarity; '
                   'decisive negative: 2947 promote/resist '
                   'labels are rebalancing-response classes, '
                   'not direct-readout polarities'),
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
if '## Phase 2954' not in memo:
    t3 = res['T3_direct_share']
    d1 = res['D1_flip_vs_dsc']
    section = '''## Phase 2954: 早翻转头极性解耦判决 [@STAMP@]

**判决：`@VERDICT@`** —— 2953 悖论（h20/h21 最早路由翻转却是 2947 抵抗头）的候选解释"它们读出注入词值为正"被决定性否定：抵抗头的快照直接贡献 dsc 实为**负**（L17 10/13 头，h20 −1.844 / h21 −2.905，与其他头同向促进塌缩），其"抵抗"（消融→sep 下降）完全由消融诱发的竞争重平衡承载（comp-share 中位 1.124/1.057 ≈ 1；h1：D=−50.73 vs dsc=+0.175）——**2950 教训 21 从 top5 头推广到全部头类**。

**设计（零前向对齐 + 一次前向族，runtime @RT@s）**：D_h（2947 npz，|D|>2 冻结规则：L17 resist 13 / promote 6 头，L16 10/12 头）× dsc_h（本 Phase，sc_of 2952 verbatim；base + L17@s1.0 + L17@s0.5 + L16@s2.0）× ATT/VAL 分解（2952 恒等式）× 翻转时间 f_h（2953 npz 早窗 s∈{0.5,0.625,0.75} 中位 A11，**全 32 头**——突破 keep 口径，教训 24 的应用）。锚 **13/13**：a1 2.17e-08（**第 25 次连续前向锚定**）、a11 dsc vs 2952 **bit 0** + ATT/VAL **bit 0** + 恒等式 6.99e-04<1e-3、a15 A11b vs 2953 **bit 0**、a13 sep L17@0.5 vs 2945 **bit 0**、a12 sc_I0 vs 2950 6.88e-08。

**主检验（T1/T2/T3 全败，决定性负结果）**：
- T1 极性（resist 头 dsc>0 全体）：fail——L17 仅 h1/h3/h5 弱正（0.14-0.23），其余 10 头负至 −2.99；L16 仅 h6/h31 正
- T2 注意力承载（ATT>0 且 share>0.6）：fail——ATT 份额高（0.79-0.99）但**符号为负**：注意力增益项承载的是促进塌缩方向的读出
- T3 直接项占比（comp-share 中位 <0.5）：fail——1.124/1.057，直接项与 |D| 完全脱钩

**解剖（D1/D2/D3）**：
- D1 翻转时间与极性无显著相关（spearman(f_h, dsc) L17 −0.180 / L16 −0.009，perm p95 0.349）——翻转次序不决定读出极性
- D2 s=0.5 早翻转窗口：resist 头 dsc 已负（h20 −0.926 / h21 −1.330），promote 头反而正（h0 +1.663 / h19 +1.007 / h22 +0.856）——极性在翻转发生时即与 2947 分类**反号**
- D3 逐头表：|重平衡|/|D| = 0.81-1.39，2950 的"直接+竞争"分解模式在全部头类成立

**结论（重复 3 次）**：**2947 的促进/抵抗分类是重平衡响应分类，不是头的直接读出极性分类——两者系统性反号。早翻转头悖论的最终消解：h20/h21 读入注入词值的直接贡献为负（与其他头同向），其"抵抗"完全来自消融诱发的重平衡；头级因果角色（D_h）与头级结构量（dsc、ATT、翻转时间）分属不同描述层，互相不可预测。与 2949/2950 合读：单头消融差分 = 竞争重平衡测量（直接项可忽略），"头级重要性=关系属性"至此覆盖全部头类与全部角色标签。**

**硬伤与勘误**：run1 a11 锚不可达（要求恒等式 VAL+ATT==dsc 达 1e-6，fp 界 ~7e-04；2952 a7 同款判据可达性错误，纪律 10）→ run2 拆三条（dsc bit / 恒等式 1e-3 / ATT-VAL bit）；run2 D2 KeyError（dsc 字典只写两条件、读第三键）+ run3 save 块同族键错误再犯（教训 18 两连违反——**跨代码段共享的字典必须先枚举全部键**）→ run4 权威。Ledger 93 条 / L14 connects 61 / ledger @LEDHASH@。

**文件+SHA256-8**：execution @HEXE@ / result @HRES@ / npz @HNPZ@ / script @HSCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2954/early_flipper_polarity/`。

**接续**：机制链第十七环（极性解耦环）闭合。候选 2955：A q·k 来源分解（2952 遗留：注入对 q/k 的直接改动 vs softmax 增益，一次前向）；B 重平衡时间定位（消融诱发的重平衡在下游哪类模块发生：MLP vs attention，一次前向）；C 承重带跨模型复现（glm4）；D v3 解码器方向重启（2940 遗留）。
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
if 'Phase 2954' not in wl:
    wl += ('- Phase 2954 闭环：early_flipper_not_positive（决定性'
           '负结果）。2947 促进/抵抗分类 = 重平衡响应分类，与直接'
           '读出极性系统性反号：抵抗头 dsc 为负（h20 −1.84/h21 '
           '−2.91），"抵抗"全由消融诱发重平衡承载（comp-share '
           '≈1.1，h1 D=−50.7 vs dsc=+0.18）；翻转时间与极性无相关'
           '（rho −0.18/−0.01）。教训 21 推广到全部头类。勘误：'
           'a11 恒等式锚 1e-6 不可达（fp 界 7e-04）+ 键规范两连'
           '违反。Ledger 93 / hash ' + chk + '。\n')
    open(WLOG, 'w', encoding='utf-8').write(wl)

# ---------------- MEMORY ----------------
mem = open(WMEM, encoding='utf-8').read()
if '当前 max=2953' in mem:
    mem = mem.replace('当前 max=2953，下一个 2954',
                      '当前 max=2954，下一个 2955')
if '25. **2954' not in mem:
    lesson = ('25. **2954（消融差分分类的层级声明）**：2947 的'
              '促进/抵抗头分类是**重平衡响应分类**，不是直接读出'
              '极性分类——两者系统性反号（L17 抵抗头 10/13 dsc 为'
              '负但 D_h<0；促进头 dsc 为正但 D_h>0；h1 D=−50.7 '
              'vs dsc=+0.175，comp-share≈1）；翻转时间与极性也无'
              '相关（rho −0.18/−0.01）。跨相位引用头角色必须声明'
              '所在层级（直接项/消融差分/路由翻转时间三者互不预测'
              '）。科学结论：2953 悖论消解——早翻转头 h20/h21 的'
              '直接贡献为负（读入注入值与其他头同向），"抵抗"完全'
              '来自消融诱发的竞争重平衡；教训 21（D_h=直接+竞争，'
              '可反号）推广到全部头类，"头级重要性=关系属性"最终'
              '形态。\n')
    i24 = mem.find('24. **2953')
    if i24 >= 0:
        iend = mem.find('\n', i24)
        mem = mem[:iend + 1] + lesson + mem[iend + 1:]
    else:
        mem = mem.rstrip() + '\n' + lesson
open(WMEM, 'w', encoding='utf-8').write(mem)

print('closeout done: ledger=%s n_meas=%s' % (
    chk, len(led['measurements'])))
