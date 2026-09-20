# -*- coding: utf-8 -*-
"""Phase 2949 closeout: seal + Ledger + MEMO + logs + MEMORY.

Idempotent: each step checks before writing.
"""
import hashlib
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2949', 'head_dose_sufficiency')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2949_head_dose_sufficiency.py')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-19.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


# ---------- 1. seal ----------
exe = json.load(open(os.path.join(OUT, 'execution.json'),
                     encoding='utf-8'))
stamp = exe['created']
res = json.load(open(os.path.join(OUT, 'result.json'),
                     encoding='utf-8'))
verdict = res['final_verdict']
hashes = {
    'execution.json': sha8(os.path.join(OUT,
                                        'execution.json')),
    'result.json': sha8(os.path.join(OUT, 'result.json')),
    'head_dose_sufficiency.npz': sha8(os.path.join(
        OUT, 'head_dose_sufficiency.npz')),
    'script': sha8(SCRIPT),
}
seal = {'phase': 2949, 'created': stamp,
        'verdict': verdict, 'hashes': hashes}
with open(os.path.join(OUT, 'seal.json'), 'w',
          encoding='utf-8') as f:
    json.dump(seal, f, indent=2, ensure_ascii=False)
print('seal done:', json.dumps(hashes))

# ---------- 2. Ledger ----------
led = json.load(open(LEDGER, encoding='utf-8'))
meas_id = 'meas_2949_head_dose_sufficiency'
if not any(m['meas_id'] == meas_id
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': meas_id,
        'type': 'causal_head_group_ablation_x_injection',
        'verdict': verdict,
        'source': 'phase2949/head_dose_sufficiency'})
    l14 = [lk for lk in led['linkage']
           if lk['link_id'] == 'L14_readout_spectrum_cross_model'][0]
    if meas_id not in l14['connects']:
        l14['connects'].append(meas_id)
    led.pop('ledger_sha256_8', None)
    canonical = json.dumps(led, sort_keys=True,
                           ensure_ascii=False).encode('utf-8')
    led['ledger_sha256_8'] = \
        hashlib.sha256(canonical).hexdigest()[:8]
    json.dump(led, open(LEDGER, 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
led2 = json.load(open(LEDGER, encoding='utf-8'))
chk = led2.pop('ledger_sha256_8')
c2 = json.dumps(led2, sort_keys=True,
                ensure_ascii=False).encode('utf-8')
assert hashlib.sha256(c2).hexdigest()[:8] == chk
print('ledger ok: n=%d last=%s hash=%s connects=%d'
      % (len(led2['measurements']),
         led2['measurements'][-1]['meas_id'], chk,
         len(l14['connects'])))

# ---------- 3. MEMO ----------
memo = open(MEMO, encoding='utf-8').read()
if '## Phase 2949' not in memo:
    d = res['D1_condition_seps']
    t = res['T1_sufficiency_L17']
    t2 = res['T2_necessity_L17']
    t3 = res['T3_sufficiency_L16']
    t4 = res['T4_necessity_L16']
    a = res['anchors']
    block = """## Phase 2949: 头级剂量充分性与组水平反转判决 [@STAMP@]

**判决：@VERDICT@** —— 2947 单头消融秩序在组水平上反转：top-W_ov-gain 头组既不充分也不必要；组消融方向与单头秩序相反，塌缩反而加深。

### 原理与设计
2947 定位开关效应于少数高杠杆头（top1 份额 0.18-0.25），2948 证明头级秩序由线性 W_ov 增益预测（rho 0.38-0.56）。遗留问题：top-g 头是否充分/必要承载开关？本 Phase 在开关剂量（L17 s=1.0 / L16 s=2.0）下做头组消融×注入交叉，消融组（o_proj 输入侧，2947 run2 verbatim）：I0 无消融 / I1 消融 top5_g / I2 消融其余 27 头，加无注入基线 B1/B2（基线门 150）。top5_g 在任何观测前自 2948 result.json D2_top5 冻结并 SHA 断言：L17 [0,7,24,22,19]、L16 [13,16,1,17,6]。

### 锚（12/12）
a1 dirs 重建 2.17e-08（第 21 次连续前向锚定）；a3 Vt8 bit 级 0；a4/a5 7.2e-06/6.3e-06；a9 单切片自检 0.0 + o_proj in_features=4096 结构门；a12 组消融 mask 自检 0.0；a8 同 session 确定性 0.0；a10/a11 I0 vs 2945 跨 session diff 0.0046/0.0010；a6 sep_func 185.6975。

### 主检验（全部方向反转，T1-T4 全败）
| 检验 | 预测（2947 秩序外推） | 实测 | 判定 |
|---|---|---|---|
| T2 必要性 L17 | 消融 top5 促进头 → sep 上升 >+20 | I0 21.5 → I1 **-9.5**（delta **-30.9**） | fail（反转） |
| T4 必要性 L16 | 同上 | I0 84.8 → I1 **48.1**（delta **-36.7**） | fail（反转） |
| T1 充分性 L17 | 仅留 top5_g → 开关仍触发（<100） | I2 **167.9**（B2 门 182.5 通过） | fail |
| T3 充分性 L16 | 同上 | I2 **120.7**（B2 门 185.6 通过） | fail |

B1 基线（无注入消融 top5）189.6/186.9、B2（消融其余 27 头）182.5/185.6——无注入下读出对头组消融稳健（单头贡献小），排除"剩余头自身塌缩"混淆。

### 结论（关键发现，重复 3 次）
**开关不由任何固定头子集承载：2947 的单头 D_h 秩序是局部敏感度，不可外推为组水平操作化——组消融 top5 促进头后剩余头竞争重平衡、塌缩反而加深（L17 至负值 -9.5）；仅留 top5 头时开关几乎不触发。与 2941（单方向注入阻尼）、2942（联合注入陡降过冲）、2946（双层干扰）合读：null 重编码是全层分布式动态的涌现属性，对一切"单点/子集"操作化（注入或消融）保持关闭。**

方法学教训：单头消融秩序（线性可预测，2948）与组消融效应（非线性反转）分属两层描述——"每个头单独重要"与"头组可移除"是不同命题，因果外推必须显式做组水平检验。

**文件+SHA256-8**：execution @HEXE@ / result @HRES@ / npz @HNPZ@ / script @HSCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2949/head_dose_sufficiency/`。runtime @RT@s。Ledger 88 条 / L14 connects 56 / ledger @HLED@。

**硬伤**：无 run 失败（一次通过）。方向反转非实现错误：I0 与 2945 bit 级一致、reps 全同、基线门齐备。

**接续**：2950 候选——A 头级竞争重平衡图谱（I1 配置下逐头 attn 权重 vs I0，定位重平衡机制）；B 承重带跨模型复现（glm4）；C 头级秩序跨层泛化（L14/L15/L18 g vs D，零前向）。
"""
    block = (block
             .replace('@STAMP@', stamp)
             .replace('@VERDICT@', verdict)
             .replace('@HEXE@', hashes['execution.json'])
             .replace('@HRES@', hashes['result.json'])
             .replace('@HNPZ@',
                      hashes['head_dose_sufficiency.npz'])
             .replace('@HSCR@', hashes['script'])
             .replace('@RT@', str(res['runtime_s']))
             .replace('@HLED@', chk))
    with open(MEMO, 'a', encoding='utf-8') as f:
        f.write('\n' + block)
memo2 = open(MEMO, encoding='utf-8').read()
assert '## Phase 2949' in memo2 and chk in memo2
print('memo ok, lines=%d' % memo2.count('\n'))

# ---------- 4. workspace log ----------
wl = open(WLOG, encoding='utf-8').read()
if 'Phase 2949' not in wl:
    wl += (
        '- Phase 2949 闭环：头级剂量充分性判决 '
        'head_dose_not_carried。top5_g 头组消融后塌缩反而加深'
        '（L17 21.5→-9.5、L16 84.8→48.1，方向反转），仅留 '
        'top5_g 开关不触发（167.9/120.7）。2947 单头秩序不可'
        '外推到组水平；锚 12/12（a1 第 21 次连续前向锚定）。'
        'Ledger 88 条 / hash ' + chk + '。产物 sealed。\n')
    open(WLOG, 'w', encoding='utf-8').write(wl)
print('wlog ok:', 'Phase 2949' in open(
    WLOG, encoding='utf-8').read())

# ---------- 5. MEMORY ----------
mem = open(WMEM, encoding='utf-8').read()
changed = False
if 'max=2949' not in mem:
    mem = mem.replace('max=2948', 'max=2949')
    mem = mem.replace('下一个 2949', '下一个 2950')
    changed = True
if '2949（组水平反转）' not in mem:
    anchor_line = '18. **2946 延伸'
    add = ('20. **2949（组水平反转）**：单头消融秩序（2947 '
           'D_h，线性可预测）不可外推为组水平操作化——组消融 '
           'top 促进头后塌缩反而加深（剩余头竞争重平衡），'
           '"每头单独重要"与"头组可移除"是不同命题；'
           '充分性/必要性检验必须带无注入基线门。\n')
    idx = mem.find(anchor_line)
    assert idx > 0
    mem = mem[:idx] + add + mem[idx:]
    changed = True
if changed:
    open(WMEM, 'w', encoding='utf-8').write(mem)
mem2 = open(WMEM, encoding='utf-8').read()
print('memory ok: max=2949 ->', 'max=2949' in mem2,
      '| lesson20 ->', '2949（组水平反转）' in mem2)
