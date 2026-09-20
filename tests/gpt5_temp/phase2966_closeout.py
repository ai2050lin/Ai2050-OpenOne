# -*- coding: utf-8 -*-
"""Phase 2966 closeout: seal + Ledger + MEMO + worklog + MEMORY.
Idempotent, placeholder replacement."""
import hashlib
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2966', 'l34_routing_membership')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-20.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2966_l34_routing_membership.py')

res = json.load(open(os.path.join(OUT, 'result.json'),
                     encoding='utf-8'))
verdict = res['final_verdict']
created = json.load(open(os.path.join(OUT, 'execution.json'),
                         encoding='utf-8'))['created']


def sha8(path):
    return hashlib.sha256(
        open(path, 'rb').read()).hexdigest()[:8]


hashes = {
    'execution.json': sha8(os.path.join(OUT, 'execution.json')),
    'result.json': sha8(os.path.join(OUT, 'result.json')),
    'l34_routing.npz': sha8(
        os.path.join(OUT, 'l34_routing.npz')),
    'script': sha8(SCRIPT),
}

# ---------------- Ledger ----------------
led = json.load(open(LEDGER, encoding='utf-8'))
meas_id = 'meas2966_l34_routing_membership'
if not any(m['meas_id'] == meas_id
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': meas_id,
        'type': 'l34_routing_membership',
        'verdict': verdict,
        'source': ('phase2966/l34_routing_membership; L17 '
                   'xdir injection 2953 GRID17+s0, 57-batch '
                   'verbatim, K=1; anchors 11/11 incl a9 '
                   'A11_L17 vs 2953 npz 1.81e-07 and a11 sep '
                   'shared grid 0.0 bit-level; T1 h15 C-dose '
                   'spearman -0.5909 p 0.059 FAIL (biphasic: '
                   'peak 0.7139 at s=0.625 vs s_c 0.6567, '
                   'then falls to -0.06); T3 rho(s,B) +0.9364 '
                   'p 1e-04 monotone B collapse -1.365->'
                   '-0.019; family top5 h6 rho -1.0'),
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
if '## Phase 2966' not in memo:
    section = '''## Phase 2966: L34 路由成员判定——h15 非单调路由成员（biphasic 峰恰在 s_c），B 随注入单调塌缩 [@STAMP@]

**判决：`@VERDICT@`**（run1 权威一次通过）——2964/2965 类效应载体 h15 是否属于 L17 A11 路由网络：57 词语言 batch verbatim（2887 表），L17 xdir 注入 coef 1.0，s 网格 = 2953 GRID17 + s=0（11 点，K=1），捕获全层 o_proj 输入 / L17+L34 v_proj / final-norm 输入。

**锚 11/11**：a1 2.17e-08 / a2 0.0 / a3 0.0 / a4 7.21e-06 / a5 6.26e-06 / a6 sep_func 185.6975 / a7 9.95e-14 / **a9 A11_L17 全 11 点 vs 2953 npz 1.81e-07（注入机器逐点复现）** / a10 L17 恢复残差 0.162（L34 0.066 描述性） / **a11 sep 共享网格 vs 2953 L17 行 = 0.0（bit 级，10 点）**；in-session s_c(L17) = 0.6567（与 2945 0.656 一致）。

**三检验**：
1. **T1 h15 剂量响应（目标性确认，quasi-post-hoc 登记）**：spearman(s, C15) = −0.5909，置换 p = 0.059 **fail**（门 0.01）——但效应量不小（|ΔC| 0.502 = 0.43σ₀）；**曲线 biphasic**：0.4377 → 峰 **0.7139 @ s=0.625**（恰在 s_c=0.6567 处！）→ 单调回落至 −0.0643 @ s=2。A11_L34_h15 同型 biphasic（峰 0.4529 @ 0.625）。
2. **T2 未触发**（T1 fail，按冻结映射短路）。
3. **T3 描述性（强信号）**：**rho(s, B) = +0.9364，p = 1e-04——B 随注入单调塌缩 −1.3654 → −0.0187（近零）**，载带方向与 L17 开关同步但无自身阈值；全头族响应 top5：**h6 rho = −1.0（完美单调下降）** / h31 +0.98 / h24 +0.98 / h2 +0.97 / h10 +0.96；h8 单调下降（0.415 → −0.297，2965 消融差分第二的因果来源在此显现）。

**结论（重复 3 次）**：**L34/h15 不是 A11 路由网络的单调成员——其类载体的注入响应是 biphasic 瞬态（峰位恰锁定在 L17 开关阈值 s_c=0.6567），不是 sigmoid 状态跟随；但深层带差分 B 本身被语言方向注入单调抹平（rho 0.94，近零），单调塌缩由其他头承载（h6 rho −1.0 / h8 单调降）。词类载体层（L34）与语言路由网络（L17 开关）在阈值水平解耦、在带差分水平强耦合——类效应不是路由态的读数，而是被语言信号调制的独立深层带结构。**

**方法论教训**：spearman 单调门把"无响应"与"非单调响应"混为一谈——T1 fail 不等于无响应（效应 0.43σ、峰位锁定 s_c）；剂量响应类判据必须把单调检验与峰位检验分开预注册（纪律 10 的检验形状版）。

**硬伤与勘误**：无 run 失败；锚-可达性设计生效（L34 恢复残差按纪律 10 降为描述性，未设锚）。

**文件+SHA256-8**：execution @HEXE@ / result @HRES@ / l34_routing.npz @HNpz@ / script @HSCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2966/l34_routing_membership/`。Ledger 105 条 / L14 connects 73 / ledger @LEDHASH@。

**接续**：候选 2967：A（主选）**B(s) 单调塌缩载体解剖**——哪些头/层承载语言注入对深层带差分的抹平（h6/h8/h31/h24 响应族 + 逐层 prof 剖面），打通"语言轴 → 深层带"的因果链路；B h15 biphasic 峰位表征（逐词峰位分布 vs s_c、跨注入方向泛化）；C S1 路由边界带扩容复检（n≥60 加 L34）；D 2961 卡组扩充（补 2962-2966 六行）。
'''
    stamp = created.replace('T', ' ')[:16]
    reps = [('@STAMP@', stamp), ('@VERDICT@', verdict),
            ('@LEDHASH@', chk),
            ('@HEXE@', hashes['execution.json']),
            ('@HRES@', hashes['result.json']),
            ('@HNpz@', hashes['l34_routing.npz']),
            ('@HSCR@', hashes['script'])]
    for k, v in reps:
        section = section.replace(k, v)
    memo += section
    open(MEMO, 'w', encoding='utf-8').write(memo)

# ---------------- worklog ----------------
entry = ('- Phase 2966 闭环：l34_h15_independent_of_routing（run1 一次通过，锚 11/11）。'
         'h15 非单调路由成员：C15 biphasic（峰 0.7139 恰在 s_c=0.6567）rho(s,C15) '
         '-0.59 p 0.059 fail；但 rho(s,B) +0.9364 p 1e-04——B 被语言注入单调抹平'
         '（-1.365→-0.019），塌缩由 h6（rho -1.0）/h8 等承载。词类载体层与语言路由'
         '网络阈值解耦、带差分强耦合。教训：单调门 ≠ 响应存在性，峰位检验须单列。'
         'a9/a11 双 bit 级锚（A11 1.8e-07 / sep 0.0）。Ledger 105 / hash ' + chk + '。\n')
wl = ''
if os.path.exists(WLOG):
    wl = open(WLOG, encoding='utf-8').read()
if 'Phase 2966' not in wl:
    wl += entry
    open(WLOG, 'w', encoding='utf-8').write(wl)

# ---------------- MEMORY ----------------
mem = open(WMEM, encoding='utf-8').read()
mem = mem.replace(
    '当前 max=**2965**，下一个 **2966**（候选 A L34 路由成员判定（h15/h8 剂量响应）；'
    'B S1 扩容复检 n≥60 加 L34；C 旋转轴功能身份；D 卡片扩充）',
    '当前 max=**2966**，下一个 **2967**（候选 A B(s) 塌缩载体解剖；B h15 biphasic '
    '峰位表征；C S1 扩容复检；D 卡片扩充）')
old = '→功能身份：h15 最大单头因果贡献但消融后类效应存活=共享载体(2965)。'
if old in mem:
    mem = mem.replace(
        old,
        old[:-1] + '→路由成员：h15 biphasic 峰锁 s_c 非单调成员，B 被语言注入单调抹平(2966)。')
add = ('- 剂量响应检验形状规范（2966）：spearman 单调门把"无响应"与"非单调响应"'
       '混为一谈——T1 fail 不等于无响应（h15 效应 0.43σ、峰位恰锁 s_c=0.6567）；'
       '剂量响应判据必须把单调检验与峰位检验分开预注册，非单调 biphasic 单独登记。'
       '科学结论：L34/h15 非 A11 路由网络单调成员（biphasic 瞬态放大器），'
       '但深层带差分 B 被语言方向注入单调抹平（rho 0.9364，-1.365→-0.019），'
       '单调塌缩由 h6（rho -1.0）/h8 等响应族承载——词类载体层与语言路由网络'
       '在阈值水平解耦、在带差分水平强耦合。注入机器 bit 级复现 2953'
       '（A11 1.8e-07 / sep 0.0）。\n')
if '剂量响应检验形状规范（2966）' not in mem:
    anchor_line = '- 跨产物引用键核对规范（2965）：'
    i = mem.find(anchor_line)
    if i >= 0:
        mem = mem[:i] + add + mem[i:]
    else:
        mem = mem.rstrip() + '\n' + add
open(WMEM, 'w', encoding='utf-8').write(mem)

print('closeout done: ledger=%s n_meas=%s' % (
    chk, len(led['measurements'])))
