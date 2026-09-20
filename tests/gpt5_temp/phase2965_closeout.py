# -*- coding: utf-8 -*-
"""Phase 2965 closeout: seal + Ledger + MEMO + worklog + MEMORY.
Idempotent, placeholder replacement."""
import hashlib
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2965', 'h15_functional_identity')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-20.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2965_h15_functional_identity.py')

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
    'h15_identity.npz': sha8(
        os.path.join(OUT, 'h15_identity.npz')),
    'script': sha8(SCRIPT),
}

# ---------------- Ledger ----------------
led = json.load(open(LEDGER, encoding='utf-8'))
meas_id = 'meas2965_h15_functional_identity'
if not any(m['meas_id'] == meas_id
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': meas_id,
        'type': 'h15_functional_identity',
        'verdict': verdict,
        'source': ('phase2965/h15_functional_identity; '
                   '990 single forwards (30 intact bit-level '
                   'vs 2964 rel 0.0 + 32 heads x 30 ablated '
                   'at L34); anchors 8/8; T1 cos(c15,u35) '
                   '-0.1301 rank 2/32 (top3 h8/h15/h21); '
                   'T2 rho(C15,B) -0.7953 perm p 1e-04; '
                   'T3a FL abl coef -0.9614 p 2.1e-03 NOT '
                   'abolished; T3b R15 +0.1172 rank 1/32 '
                   '(h8 0.1111 close 2nd), mean|dB| 0.2181 '
                   'vs med 0.0124'),
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
if '## Phase 2965' not in memo:
    section = '''## Phase 2965: L34/h15 功能身份判定——最大单头因果贡献但类效应存活（共享载体） [@STAMP@]

**判决：`@VERDICT@`**（run2 权威）——2964 唯一显著头 h15 的因果身份：词表 verbatim 取自 2964 封存 npz（30 词），2937 pass1 协议 verbatim，990 次单前向 = 30 intact + 32 头 × 30 消融（L34 only，o_proj 输入头切片置零，2932 规范件）。

**锚 8/8**：a1 形状门 / a2 0.0 / **a3 intact B vs 2964 npz rel 0.0（bit 级跨相位恒等）** / a4 30/30 / a5 3.614e-05 / a6 4.14e-16 / **a7 消融效能 max|C_abl| = 0.0（实现门）** / a8 非退化。

**三检验**：
1. **T1 写出身份（描述性）**：cos(c15, u35) = −0.1301，|cos| 秩 **2/32**（top3：h8 −0.1607 / h15 −0.1301 / h21 −0.0943——2953 早翻转头 h21 再次出现于 top3）；c15 在 Vt8 top4 投影均小（|·|≤0.083）——h15 类对比写出方向对齐 u35 但不落入读出 SVD 基。
2. **T2 载体相关**：rho(C[34,w,15], B_w) = **−0.7953**，置换 p = 9.999e-05——h15 贡献与 B 强负相关（贡献越大 B 越负），通过门。
3. **T3 功能消融（主检验）**：gap 1.3738 → 1.2566（仅缩 **8.5%**）；**R15 = +0.1172 秩 1/32**（最大单头因果贡献；h8 0.1111 紧随第二——2964 maxT 未显著但消融差分几乎同等，选拔量与消融量口径分离再证 2950）；消融后 FL 系数 −0.9614、**p = 2.1e-03 类效应存活**（未消去）；mean|dB| h15 0.2181 vs 全头中位 0.0124（17.6×——h15 同时是全局 B 影响最大的头，非类选择性的纯粹开关）。

**结论（重复 3 次）**：**L34/h15 是类效应的最大单头因果载体（消融差分秩 1/32、相关 −0.7953、写出方向 u35 对齐秩 2/32），但类效应在 h15 消融后存活（p 2.1e-3）——功能词 vs 内容词的承重带差距是分布式共享载体（h15/h8/h28 为 top3 贡献者），无单点必要头；2949"单头重要 ≠ 可移除"在类效应域再确认：maxT 选拔显著（2964）与消融必要性（2965）是不同强度的命题，选拔层主张不得代证功能层（纪律 15 分账）。**

**硬伤与勘误（run1→run2）**：run1 a5 从 2964 result.json 取数用了不存在的键 `gap_heads_L34`（实际结构为 `top5_heads` 列表）→ KeyError。教训：从封存产物读数前必须先核对 JSON 实际结构（跨 Phase 引用键不得凭记忆构造）。按纪律 3 删旧产物重跑，run2 一次通过。

**文件+SHA256-8**：execution @HEXE@ / result @HRES@ / h15_identity.npz @HNpz@ / script @HSCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2965/h15_functional_identity/`。Ledger 104 条 / L14 connects 72 / ledger @LEDHASH@。

**接续**：候选 2966：A（主选）**L34 路由成员判定**——h15/h8 双头剂量响应（2953 A11 注入协议挂 L34，k∈{0..1}），判定 L34 头是否属于 A11 路由网络（sigmoid 阈值 vs 渐变）还是独立词类读出层；B S1 路由边界带扩容复检（n≥60，路由层集加 L34，2962 B 候选）；C 旋转轴功能身份（v_rot vs W_ov/u35 代数关系）；D 2961 卡组扩充（补 2962-2965 五行）。
'''
    stamp = created.replace('T', ' ')[:16]
    reps = [('@STAMP@', stamp), ('@VERDICT@', verdict),
            ('@LEDHASH@', chk),
            ('@HEXE@', hashes['execution.json']),
            ('@HRES@', hashes['result.json']),
            ('@HNpz@', hashes['h15_identity.npz']),
            ('@HSCR@', hashes['script'])]
    for k, v in reps:
        section = section.replace(k, v)
    memo += section
    open(MEMO, 'w', encoding='utf-8').write(memo)

# ---------------- worklog ----------------
entry = ('- Phase 2965 闭环：h15_shared_carrier_effect_survives（run2）。L34/h15 功能身份：'
         'T2 rho(C15,B) -0.7953 p 1e-04、T3 消融差分 R15 +0.1172 秩 1/32（h8 0.1111 次之）'
         '但消融后 FL p 2.1e-03 类效应存活（gap 1.374→1.257 仅缩 8.5%）——最大单头因果贡献'
         '+分布式共享载体，无单点必要头；T1 写出方向 u35 对齐秩 2/32（h8/h15/h21 top3）。'
         '勘误：run1 a5 引用键凭记忆构造（gap_heads_L34 不存在）→ KeyError，删旧重跑。'
         'Ledger 104 / hash ' + chk + '。\n')
wl = ''
if os.path.exists(WLOG):
    wl = open(WLOG, encoding='utf-8').read()
if 'Phase 2965' not in wl:
    wl += entry
    open(WLOG, 'w', encoding='utf-8').write(wl)

# ---------------- MEMORY ----------------
mem = open(WMEM, encoding='utf-8').read()
mem = mem.replace(
    '当前 max=**2964**，下一个 **2965**（候选 A L34/h15 功能身份判定；B S1 '
    '扩容复检加 L34；C 旋转轴功能身份；D 卡片扩充）',
    '当前 max=**2965**，下一个 **2966**（候选 A L34 路由成员判定（h15/h8 剂量响应）；'
    'B S1 扩容复检 n≥60 加 L34；C 旋转轴功能身份；D 卡片扩充）')
old = '→载体解剖：L34/h15 单点定位，类效应头≠语言轴头(2964)。'
if old in mem:
    mem = mem.replace(
        old,
        old[:-1] + '→功能身份：h15 最大单头因果贡献但消融后类效应存活=共享载体(2965)。')
add = ('- 跨产物引用键核对规范（2965）：从封存 result.json 读数前必须先核对其'
       '实际 JSON 结构，引用键禁止凭记忆构造（run1 a5 KeyError）。科学结论：'
       'L34/h15 = 最大单头因果载体（R 秩 1/32、rho -0.7953、u35 对齐秩 2/32）'
       '但消融后类效应存活（FL p 2.1e-3，gap 仅缩 8.5%）——类效应共享载体'
       '（h15/h8/h28 top3），无单点必要头；maxT 选拔显著（2964）≠ 消融必要性'
       '（2965），纪律 15 分账再确认；h8 选拔未显著但消融差分 0.1111 近同 h15'
       '——选拔量与消融差分口径分离（2950 再证）。\n')
if '跨产物引用键核对规范（2965）' not in mem:
    anchor_line = '- 协议常量显式重建规范（2964）：'
    i = mem.find(anchor_line)
    if i >= 0:
        mem = mem[:i] + add + mem[i:]
    else:
        mem = mem.rstrip() + '\n' + add
open(WMEM, 'w', encoding='utf-8').write(mem)

print('closeout done: ledger=%s n_meas=%s' % (
    chk, len(led['measurements'])))
