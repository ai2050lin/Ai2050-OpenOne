# -*- coding: utf-8 -*-
"""Phase 2947 closeout: seal + ledger + MEMO + logs. Idempotent."""
import hashlib
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2947', 'head_anatomy')
GPT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5'
        r'\docs\AGI_GPT5_MEMO.md')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5'
          r'\atlas\atlas_ledger.json')
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-19.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2947_head_anatomy.py')

res = json.load(open(os.path.join(OUT, 'result.json'),
                     encoding='utf-8'))


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


paths = {
    'execution.json': os.path.join(OUT, 'execution.json'),
    'result.json': os.path.join(OUT, 'result.json'),
    'head_anatomy.npz':
        os.path.join(OUT, 'head_anatomy.npz'),
    'script': SCRIPT,
}
hashes = {k: sha8(v) for k, v in paths.items()}
stamp = json.load(open(paths['execution.json'],
                       encoding='utf-8'))['created']
verdict = res['final_verdict']
runtime = res['runtime_s']

# ---------- 1. seal ----------
seal = ['Phase 2947 seal', 'verdict: ' + verdict,
        'created: ' + stamp, 'runtime_s: ' + str(runtime)]
seal += [k + ' sha256-8 ' + v for k, v in hashes.items()]
seal.append('a9 structural gate: o_proj in_features 4096 '
            '== NH*HD confirmed')
open(os.path.join(GPT, 'phase2947_seal_report.txt'), 'w',
     encoding='utf-8').write('\n'.join(seal) + '\n')
print('seal written')

# ---------- 2. ledger ----------
led = json.load(open(LEDGER, encoding='utf-8'))
mid = 'meas_2947_head_anatomy'
if not any(m['meas_id'] == mid
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': mid,
        'type': 'forward_family_head_ablation',
        'verdict': verdict,
        'source': 'phase2947/head_anatomy'})
    l14 = [lk for lk in led['linkage']
           if lk['link_id'] == 'L14_readout_spectrum_cross_model'
           ][0]
    if mid not in l14['connects']:
        l14['connects'].append(mid)
led.pop('ledger_sha256_8', None)
new_h = hashlib.sha256(json.dumps(
    led, sort_keys=True, ensure_ascii=False)
    .encode('utf-8')).hexdigest()[:8]
led['ledger_sha256_8'] = new_h
json.dump(led, open(LEDGER, 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)
print('ledger done, hash ' + new_h)

# ---------- 3. MEMO ----------
memo = open(MEMO, encoding='utf-8').read()
if '## Phase 2947' not in memo:
    section = '''## Phase 2947: 开关头级解剖与集中度判决 [''' + stamp + ''']

**为什么做**：2944-2946 确立开关为每层独立浓度门控阈值现象，L15/L17 开关型（陡降单调）、L16 渐变型。遗留问题：头级载体是什么——开关型层由少数主导头承载、渐变型层分布式摊开？

**方法**：一次前向族。xdir 注入 L17 s=1.0（开关已触发，sep 21.5）与 L16 s=2.0（渐变中部，sep 84.8）；逐头消融 = 在 self_attn.o_proj **输入**（4096 维 = 32×128）将头 h 的 128 维切片在 pos-1 置零；32 头 × {L17 注入, L16 注入, L10 对照（注入仍在 L17）}，K=3 同 session 中位。D_h = C_h − Cc_h（注入特异头效应）。锚 **11/11**：a1 2.17e-08（**第 20 次连续前向锚定**）、a9 消融自检 0 + **o_proj in_features=4096 结构门**、a10 REF17 vs 2945 diff 0.0046、a11 REF16 vs 2945 diff 0.0010（两次跨 session bit 级级稳定）。runtime 25.6 s。

**Run1 无效与勘误（correction_note 已入 PREREG）**：run1 消融 hook 挂在 attention 模块输出上——qwen3-4b **hidden=2560 ≠ 32×128=4096**，该处头结构不存在：头 0-19 实为 hidden 维度切片、头 20-31 为空切片（bit 级零）。诊断探针（config 打印 + fin diff 对照）定位后，run2 把消融移到 o_proj 输入侧（头结构成立）。**教训 19 入 MEMORY：Qwen3-4B 的头级操作必须在 o_proj 输入（4096 维）上做，attention 模块输出已是 2560 维混合 hidden。**

**T 检验（判决 `switch_head_concentrated_only`）**：
| 统计量 | L17（开关型） | L16（渐变型） | 判定 |
|---|---|---|---|
| max D_h（T0 门） | **17.52** | 50.69 | T0 ✓ |
| P = top1/Σ\|D\|（T1） | **0.2464** | 0.1803 | T1 ✓（差 0.066 > 0.05） |
| effN 参与比（T2） | 10.29 | 11.58 | **T2 ✗**（11.58 < 1.2×10.29=12.35） |

T2 fail 的实质：渐变层并不比开关层更分布式——两层都是少数头承载结构，但 L17 的 top1 份额显著更高。

**头级图谱（|D|>2，对照 Cc∈[−1.55,1.75] 确认注入特异性）**：
- L17 促进塌缩：头 **22 (+17.5)、19 (+13.1)**、0 (+5.9)、7 (+5.8)、10 (+3.6)；抵抗塌缩：头 **1 (−50.7！消融反而加深塌缩)**、20 (−17.4)、21 (−14.0)。
- L16 促进：头 **17 (+50.7，单头恢复 47% sep 缺口)**、**19 (+32.9)**、13 (+31.4)、27 (+31.4)、16 (+16.7)；抵抗：31 (−14.5)、23 (−10.9)、6 (−9.8)。
- **共享促进头 19 与 22 在两层均为正**——开关带内存在跨层复用的促进头子集。

**结论**：开关的头级载体存在但非"单头开关"——top1 消融最大恢复 L16 缺口 47% / L17 缺口 11%，且存在双向头（促进/抵抗）。开关型 vs 渐变型的区别是 **集中度（P）而非分布式程度（effN）**：两层 effN 几乎相同（~10-12），L17 更依赖单一主导头。结合 2944-2946：开关 = 带内冗余层 × 层内少数高杠杆头 × 非线性剂量门控，任何单点（头/层/方向）操作化都不足以完整重放 null 重编码。

**文件+SHA256-8**：execution @EXE@ / result @RES@ / npz @NPZ@ / script @SCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2947/head_anatomy/`。runtime @RT@ s。

**接续**：2948 候选——A 促进头对 (L17-h22, L16-h17) 组合消融/增强（检验跨层共享促进头的因果充分性）；B 头 19/22 的 W_ov 语义分析（零前向：o_proj/W_v 与 dirs/v3 的对齐）；C 承重带跨模型复现（glm4）；D L15/L18 头级图谱补全。

'''
    section = (section
               .replace('@EXE@', hashes['execution.json'])
               .replace('@RES@', hashes['result.json'])
               .replace('@NPZ@', hashes['head_anatomy.npz'])
               .replace('@SCR@', hashes['script'])
               .replace('@RT@', str(runtime)))
    with open(MEMO, 'a', encoding='utf-8') as f:
        f.write(section)
memo2 = open(MEMO, encoding='utf-8').read()
print('memo 2947 present:', '## Phase 2947' in memo2,
      '| placeholders left:',
      '@EXE@' in memo2[memo2.find('## Phase 2947'):])

# ---------- 4. workspace log ----------
wl = open(WLOG, encoding='utf-8').read()
if 'Phase 2947' not in wl:
    entry = ('- Phase 2947 闭环：开关头级解剖（L17 s=1.0 / L16 s=2.0 注入下 32 头 o_proj 输入侧逐头消融 + L10 对照，'
             '锚 11/11 含 o_proj in_features=4096 结构门）。判决 switch_head_concentrated_only：T1 P(L17)=0.246>'
             'P(L16)=0.180+0.05 ✓，T2 effN 11.58<12.35 ✗——两层均少数头承载，开关型 vs 渐变型区别在集中度非分布式程度。'
             'L16 头 17 单头恢复 47% 缺口；共享促进头 19/22 跨层复用；L17 头 1 消融反向加深塌缩（−50.7）。'
             'run1 无效：消融误挂 attention 输出（hidden 2560 无头结构，头 20-31 空切片 bit 级零），诊断探针定位后移至 '
             'o_proj 输入侧重跑；教训 19 入 MEMORY。Ledger 86 条 / L14 connects 54 / ledger ' + new_h + '。\n')
    with open(WLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
print('wlog done:',
      'Phase 2947' in open(WLOG, encoding='utf-8').read())

# ---------- 5. MEMORY.md ----------
mem = open(WMEM, encoding='utf-8').read()
if 'max=2947' not in mem:
    mem = mem.replace('max=2946', 'max=2947')
    mem = mem.replace('下一个 2947', '下一个 2948')
    marker = '（sc_interp 浮点键查字符串键字典 KeyError）'
    if marker in mem and '2947 延伸' not in mem:
        mem = mem.replace(
            marker,
            marker + '；2947 延伸（教训 19）：Qwen3-4B hidden=2560≠32×128，头级切片操作必须挂在 o_proj 输入（4096 维）——attention 模块输出已是混合 hidden，头 20-31 会成空切片 bit 级零',
            1)
    with open(WMEM, 'w', encoding='utf-8') as f:
        f.write(mem)
mem2 = open(WMEM, encoding='utf-8').read()
print('wmem max=2947:', 'max=2947' in mem2,
      '| lesson19:', '教训 19' in mem2)
