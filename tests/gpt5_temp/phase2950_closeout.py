# -*- coding: utf-8 -*-
"""Phase 2950 closeout: seal + Ledger + MEMO + logs + MEMORY.

Idempotent: each step checks before writing.
"""
import hashlib
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2950', 'rebalance_anatomy')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2950_rebalance_anatomy.py')
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
    'rebalance_anatomy.npz': sha8(os.path.join(
        OUT, 'rebalance_anatomy.npz')),
    'script': sha8(SCRIPT),
}
seal = {'phase': 2950, 'created': stamp,
        'verdict': verdict, 'hashes': hashes}
with open(os.path.join(OUT, 'seal.json'), 'w',
          encoding='utf-8') as f:
    json.dump(seal, f, indent=2, ensure_ascii=False)
print('seal done:', json.dumps(hashes))

# ---------- 2. Ledger ----------
led = json.load(open(LEDGER, encoding='utf-8'))
meas_id = 'meas_2950_rebalance_anatomy'
if not any(m['meas_id'] == meas_id
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': meas_id,
        'type': 'head_contribution_decomposition',
        'verdict': verdict,
        'source': 'phase2950/rebalance_anatomy'})
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
if '## Phase 2950' not in memo:
    block = """## Phase 2950: 头级竞争重平衡与补偿性级联判决 [@STAMP@]

**判决：@VERDICT@** —— 2949 组消融加深塌缩的主体是主动补偿性重平衡（间接项占 70-74%），不是被动丢失；且被消融 top5_g 头的快照线性贡献为正（抵抗塌缩），证明 2947 单头秩序的主体是竞争项而非直接贡献。

### 原理与设计
在注入层 o_proj 输出快照处做精确线性分解：逐头贡献 c_h(w) = u35·(Wo_h @ x_h(w)) 对 proj/sep 线性，故 sep 可逐头分解。D_abl = Σ_{h∈top5_g} sep_c_h(I0)（消融直接损失）；dSep = sep(I1) − sep(I0)（总量，2949 实测反转量）；D_nonlin = dSep + D_abl（间接重平衡 = 剩余头重算 + 下游非线性响应）。判据：D_nonlin < 0 且 |D_nonlin| > |D_abl| 且特异性门 |sep(B1)−sep(B0)| < 10。捕获注入层逐头 o_proj 输入（消融后，a13 验证被消融切片 bit 级零）。

### 锚（13/13）
a1 2.17e-08（第 22 次连续前向锚定）；a3 Vt8 bit 级 0；a9/a12 自检 0.0 + o_proj 结构门；a13 消融捕获零检查 0.0；a8 确定性 0.0；a10/a11 I0 vs 2945 跨 session diff 0.0046/0.0010；六条件 sep 与 2949 bit 级一致。

### 主检验（T1/T2 全过）
| 层 | D_abl（直接） | dSep（实测） | D_nonlin（间接） | 间接占比 | 特异性门 |
|---|---|---|---|---|---|
| L17 | **+9.35** | −30.91 | **−21.56** | 70% | sep(B1)=189.6，gate ✓ |
| L16 | **+9.37** | −36.69 | **−27.31** | 74% | sep(B1)=186.9，gate ✓ |

D_abl 为正：top5_g 头的快照线性贡献整体**推高 sep（抵抗塌缩）**——消融它们本应使 sep 上升 9.4；实测下降 30.9/36.7，差额全部由间接重平衡承载。

### 头级图谱与传播剖面
- 重平衡载体是**非 top5 的新头集**：L17 由 h16(−3.0)/h21(−2.9)/h18/h20/h28 承载；L16 由 **h26(−4.67)**/h8/h2/h12 领衔——与 2947 促进头集、2948 g 排名均不重叠。
- 传播剖面（I1−I0 的逐层 u35 投影中位差）沿深度**单调累积放大**：L17 族从 L18 −1.7 增至 L35 −35.4；L16 族至 −13.0——塌缩加深是全程级联，非单层事件。

### 结论（关键发现，重复 3 次）
**单头消融差分 D_h = 直接贡献 + 竞争重平衡项的混合测量：2947 的"促进头"直接贡献实为抵抗性（+9.4），其正 D_h 由竞争项主导；组消融把竞争项反转为巨大的补偿性加深（−21.6/−27.3，70-74%）。开关的组织原则是全头竞争平衡，头级"重要性"是关系属性而非内在属性——消融任何子集都触发剩余网络的重平衡，使子集操作化不可解释。**

方法学教训：因果归因必须区分快照线性分解（直接贡献）与消融差分（含重平衡）——两者符号可以相反（本次 top5_g：直接 +9.4 vs 消融差分 −17.5 混合）。

**文件+SHA256-8**：execution @HEXE@ / result @HRES@ / npz @HNPZ@ / script @HSCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2950/rebalance_anatomy/`。runtime @RT@s。Ledger 89 条 / L14 connects 57 / ledger @HLED@。

**硬伤**：run1 空 attnin 列表 stack（注入层不捕获但 clear_cap 残留空键）、run2 逐头 sep 贡献索引轴错误（c 为 (32头,57词)，应沿词轴取类均值）——run3 权威；六条件 sep 与 2949 逐轮 bit 级一致证明数据无损。

**接续**：2951 候选——A 重平衡载体的功能性：h26/h16 等新头在 I1 下的 W_ov 增益（g 是否预测重平衡秩，零前向）；B 承重带跨模型复现（glm4）；C 头级秩序跨层泛化（L14/L15/L18）；D 重平衡时间定位：深层 MLP vs attention 贡献分解。
"""
    block = (block
             .replace('@STAMP@', stamp)
             .replace('@VERDICT@', verdict)
             .replace('@HEXE@', hashes['execution.json'])
             .replace('@HRES@', hashes['result.json'])
             .replace('@HNPZ@',
                      hashes['rebalance_anatomy.npz'])
             .replace('@HSCR@', hashes['script'])
             .replace('@RT@', str(res['runtime_s']))
             .replace('@HLED@', chk))
    with open(MEMO, 'a', encoding='utf-8') as f:
        f.write('\n' + block)
memo2 = open(MEMO, encoding='utf-8').read()
assert '## Phase 2950' in memo2 and chk in memo2
print('memo ok, lines=%d' % memo2.count('\n'))

# ---------- 4. workspace log ----------
wl = open(WLOG, encoding='utf-8').read()
if 'Phase 2950' not in wl:
    wl += (
        '- Phase 2950 闭环：头级竞争重平衡判决 '
        'rebalancing_compensatory。分解证明 2949 组消融加深'
        '由间接补偿承载（L17 -21.6 / L16 -27.3，占 70-74%），'
        '被消融 top5_g 头直接贡献为正（+9.4，抵抗塌缩）；'
        '2947 单头秩序主体是竞争项。重平衡载体为非 top5 新头'
        '（h26/h16 等），传播剖面沿深度单调放大（至 L35 '
        '-35.4）。Ledger 89 条 / hash ' + chk + '。\n')
    open(WLOG, 'w', encoding='utf-8').write(wl)
print('wlog ok:', 'Phase 2950' in open(
    WLOG, encoding='utf-8').read())

# ---------- 5. MEMORY ----------
mem = open(WMEM, encoding='utf-8').read()
changed = False
if 'max=2950' not in mem:
    mem = mem.replace('max=2949', 'max=2950')
    mem = mem.replace('下一个 2950', '下一个 2951')
    changed = True
if '2950（消融差分=直接+竞争）' not in mem:
    anchor_line = '20. **2949（组水平反转）**'
    add = ('21. **2950（消融差分=直接+竞争）**：单头消融差分 '
           'D_h 是快照线性贡献与竞争重平衡的混合测量，两者'
           '符号可相反（2950：top5_g 直接 +9.4 抵抗 vs 消融'
           '差分促进）——因果归因必须配快照分解；组消融加深'
           '由间接补偿承载（70-74%）且沿深度级联放大。\n')
    idx = mem.find(anchor_line)
    assert idx > 0
    mem = mem[:idx] + add + mem[idx:]
    changed = True
if changed:
    open(WMEM, 'w', encoding='utf-8').write(mem)
mem2 = open(WMEM, encoding='utf-8').read()
print('memory ok: max=2950 ->', 'max=2950' in mem2,
      '| lesson21 ->', '2950（消融差分=直接+竞争）' in mem2)
