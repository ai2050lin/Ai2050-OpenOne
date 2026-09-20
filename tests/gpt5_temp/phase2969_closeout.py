# -*- coding: utf-8 -*-
"""Phase 2969 closeout: seal artifacts (SHA256-8), append Ledger
measurement + L14 connect, append MEMO section, append workspace
daily log, update MEMORY.md."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
OUTD = os.path.join(
    BASE, r'tests\glm5\result\rdc_query_construction_20260913'
          r'\phase2969\peak_word_attributes')
SCRIPT = os.path.join(
    BASE, r'tests\glm5\phase2969_peak_word_attributes.py')
LEDGER = os.path.join(BASE, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(BASE, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
WSLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
         r'\.workbuddy\memory\2026-09-20.md')
MEMFILE = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
           r'\.workbuddy\memory\MEMORY.md')

STAMP = '2026-09-20 01:59'


def s8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


# ---------- seal ----------
shas = {
    'execution.json': s8(os.path.join(OUTD, 'execution.json')),
    'result.json': s8(os.path.join(OUTD, 'result.json')),
    'peak_attr.npz': s8(os.path.join(OUTD, 'peak_attr.npz')),
    'script': s8(SCRIPT),
}
print('seal:', shas)

r = json.load(io.open(os.path.join(OUTD, 'result.json'),
                      encoding='utf-8'))
verdict = r['final_verdict']
assert verdict == 'peak_source_lang_within_pair_descriptive'
assert r['anchors']['ok'] is True

# ---------- ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
old_sha = led.get('ledger_sha256_8')
led.pop('ledger_sha256_8', None)
n_before = len(led['measurements'])
meas_id = 'meas2969_peak_word_attributes'
led['measurements'].append({
    'meas_id': meas_id,
    'phase': 2969,
    'name': 'peak_word_attributes',
    'created': '2026-09-20T01:59:21',
    'verdict': verdict,
    'artifacts': {
        'execution.json': shas['execution.json'],
        'result.json': shas['result.json'],
        'peak_attr.npz': shas['peak_attr.npz'],
        'script': shas['script'],
    },
})
l14 = [l for l in led['linkage']
       if l['link_id'] == 'L14_readout_spectrum_cross_model'][0]
assert meas_id not in l14['connects']
l14['connects'].append(meas_id)
new_sha = hashlib.sha256(json.dumps(
    led, sort_keys=True, ensure_ascii=False)
    .encode('utf-8')).hexdigest()[:8]
led['ledger_sha256_8'] = new_sha
json.dump(led, io.open(LEDGER, 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('ledger: %d -> %d, L14 connects %d, sha %s -> %s'
      % (n_before, len(led['measurements']),
         len(l14['connects']), old_sha, new_sha))

# ---------- memo ----------
sec = u"""
## Phase 2969: 峰位分布的词属性解释——配对语言效应硬显著、tid 组内无效 [2026-09-20 01:59]

**性质**：quasi-post-hoc 解释性 Phase（纪律 9 全标注）——峰位派生自 2968 已封存 npz（其 result 已并排展示 n=40/中位/A11 rho），无新前向；全部检验只携带解释性权重，verdict 带 `_descriptive` 尾缀。

**设计**：57 词为 en/fr 词对结构（`en:<cid>:<name>` / `L:<cid>:<name>`，concept id 中字段），1 个 multi-token 词；峰位 verbatim 复用 2968 `peak_loc`（内 argmax→三点抛物线顶点）。锚 5/5：a1 n_peak=40 精确复算、a2 中位 0.6746、a3 A11pk_vs_C15pk_rho 0.9542、a4 C34_A vs 2967 C_all **bit 级 0**、a5 桶直方图精确复算。

**结果**：
- **T1（tid/频率，组内 spearman + maxT family=2）**：en rho 0.118 p 0.61；L rho 0.0515 p 0.84；maxT q 0.856——**组内 tid 律不成立**。Simpson 对账（2963 规范）：全样本 rho 0.5049 正 vs 组内 ~0——全样本正相关是组间结构（L 组 tid 大且峰位晚），非组内频率律，与 2963 承重带同构。
- **T2（配对语言检验，主显著）**：双内峰 concept 对 **13 对**（有效门 ≥10），d = pk_L − pk_en：mean **+0.5132**、中位 +0.5399、**13/13 对符号一致为正**（frac_negative 0.000）、符号翻转置换 **p = 2e-04 硬显著**——fr 词的 h15 瞬态峰位系统性晚于 en 对应词（同 concept 配对，控制了词义）。
- **T3（描述性分解）**：峰位总方差 0.1122；非配对 eta2(lang) 0.404；tid 去趋势后残差方差份额 0.624；残差峰位 vs A11 峰位 rho 0.586。

**判决：`peak_source_lang_within_pair_descriptive`**（run3 权威，锚 5/5）。

**结论**：2968 峰位宽分布（0.29–1.61）的来源**不在词频率/ tid（组内 ns），而在语言身份：同 concept 的 fr 词峰位一律晚于 en 词（13/13，p 2e-4）**——h15 瞬态峰位携带跨语言时序签名，峰位是语言轴的下游读数之一；残差与 A11 增益峰 rho 0.586 表明路由增益解释剩余个体变异的一部分。修正版思路一的签名矩阵再添一行：词类签名在读出坐标缺席、在承重带 function/content 硬显著（2963/2964）、在路由瞬态峰位跨语言系统分化（本 Phase）——**语言与词类在深层带不同自由度上编码：词类在带差分静态量，语言在瞬态峰位时序量**。

**硬伤与勘误（run1→run3）**：① run1 a5 桶键格式错（'%.2f'→'0.60' vs 2968 str(round)→'0.6'）——跨产物复算锚必须先核对源键格式（新教训入 MEMORY）；② 脚本初稿残留一行 numpy 数组 truthiness 坏代码与 maxT 死代码块，冻结前清理；③ run2 漏 execution.json 冻结段（协议缺口）→ 补齐后删产物重跑 run3 权威；三次运行数值完全一致（确定性离线复算）。

**产物**：`phase2969/peak_word_attributes/` execution {a1} / result {b1} / peak_attr.npz {c1} / script {d1}。

**接续（2970 候选）**：A（主选）跨语言峰位延迟的载体定位——fr-vs-en 峰位差的头/层解剖（13 对词重跑全层 C 矩阵 + maxT），判定延迟在 L34 局部还是上游链路；B 语言×词类双因子签名矩阵（合并 2963/2964/2969 词表，n≥60）；C h8/h21 峰位的同款词属性检验（2968 数据离线，成本近零）；D 2961 卡组扩充（补 2962-2969 八行）。
""".format(a1=shas['execution.json'], b1=shas['result.json'],
           c1=shas['peak_attr.npz'], d1=shas['script'])
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(sec)
print('memo appended')

# ---------- workspace daily log ----------
wl = io.open(WSLOG, encoding='utf-8').read()
entry = (u"\n## Phase 2969（2026-09-20）峰位词属性解释\n"
         u"- 判决 peak_source_lang_within_pair_descriptive（quasi-post-hoc，锚 5/5，run3 权威）。\n"
         u"- 57 词 en/fr 词对；T1 tid 组内 ns（maxT q 0.856）；T2 配对语言 p 2e-04、13/13 对 fr 峰位晚于 en；T3 eta2(lang) 0.40。\n"
         u"- 结论：峰位宽分布来源是语言身份而非频率；词类=带差分静态量、语言=瞬态峰位时序量。\n"
         u"- Ledger 108 条 / L14 76 / hash %s；勘误：a5 桶键格式、execution.json 补齐（run2 漏）。\n"
         % new_sha)
if 'Phase 2969' not in wl:
    with io.open(WSLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
    print('wslog appended')

# ---------- MEMORY.md ----------
mem = io.open(MEMFILE, encoding='utf-8').read()
if 'max=**2969**' not in mem:
    mem = mem.replace('max=**2968**', 'max=**2969**')
    anchor = '## 本机环境缺陷与对策（Windows，必读）'
    add = (u"""25. **2969（跨产物复算锚格式规范 + execution.json 完整性）**：跨产物锚（桶直方图/键控复算）必须先核对源产物的键格式（2968 桶键是 str(round(v,2))→'0.6'，不是 '%.2f'→'0.60'）；execution.json 冻结段是主脚本的固定开局块，漏写即协议缺口须删产物重跑。科学结论：h15 瞬态峰位宽分布来源是语言身份（同 concept en/fr 配对 13/13 fr 晚于 en，p 2e-4），tid/频率组内无效（maxT q 0.856）；全样本 rho(tid,pk)=0.50 是组间 Simpson 结构（2963 规范复用）。机制链 24 环：词类在带差分静态量（2963/2964）、语言在瞬态峰位时序量（2969）。
""")
    mem = mem.replace(anchor, add + anchor, 1)
    io.open(MEMFILE, 'w', encoding='utf-8').write(mem)
    print('memory updated')
print('closeout done')
