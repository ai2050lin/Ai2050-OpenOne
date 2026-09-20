# -*- coding: utf-8 -*-
"""Phase 2963 closeout: seal + Ledger + MEMO + worklog + MEMORY.
Idempotent, placeholder replacement."""
import hashlib
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2963',
                   'frequency_controlled_band')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-20.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2963_frequency_controlled_band.py')

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
    'freq_band.npz': sha8(
        os.path.join(OUT, 'freq_band.npz')),
    'script': sha8(SCRIPT),
}

# ---------------- Ledger ----------------
led = json.load(open(LEDGER, encoding='utf-8'))
meas_id = 'meas2963_frequency_controlled_band'
if not any(m['meas_id'] == meas_id
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': meas_id,
        'type': 'frequency_controlled_band_recheck',
        'verdict': verdict,
        'source': ('phase2963/frequency_controlled_band; 45 '
                   'FRESH words (F15 function tid 369-3425 / '
                   'C15 common nouns 1251-4627 / R15 rare '
                   'nouns 13551-46118) + 5 anchor words '
                   're-forwarded from 2962; anchors 6/6 '
                   'incl cross-phase protocol anchor B vs '
                   '2962 npz rel 7.76e-16 (bit-level); T1 '
                   'Freedman-Lane group coef -1.0371 p '
                   '2.0e-04 (frequency-controlled, HARD); '
                   'T2 within-content gradient rho -0.16 ns; '
                   'T3 3/5 mixed blocks invalid-registered; '
                   'T4 rho(tokid,B) all-45 -0.6464 is '
                   'between-group Simpson structure; '
                   'function-vs-content band gap ~1.46 is a '
                   'CLASS effect beyond frequency'),
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
if '## Phase 2963' not in memo:
    section = '''## Phase 2963: 承重带 function-vs-content 差距的频率受控复检——类效应确认 [@STAMP@]

**判决：`@VERDICT@`**（一次通过，锚 6/6）——2962 遗留的"function 组承重带 +1.23 可能纯频率驱动"假说在此正检验。设计：**45 个全新未观测词**（F15 功能词 tid 369-3425 / C15 高频名词 1251-4627 与 F 交错 / R15 稀有名词 13551-46118）+ 5 个 2962 词重前向作跨相位协议锚（排除出全部检验）；2937 pass1 协议 verbatim，B = 2962 口径（L6-12 均值 − L28-35 均值）。

**锚 6/6**：a1 Vt8 3.04e-08 / a2 决定性 0.0 / a3 头块-直读 8.13e-16 / a4 单 token 50/50 + 新鲜词 45/45 / **a5 跨相位 B vs 2962 npz rel 7.76e-16（bit 级闭合——协议恒等的直接证明）** / a6 非退化门全过。

**四检验**：
1. **T1 主检验（Freedman-Lane）**：F∪C 上 B ~ rank(tid) + group，残差置换 10000——**组系数 −1.0371，p = 2.0e-04，硬显著**；medB F −1.1416 vs C −2.6059（gap ~1.46，与 2962 的 1.23 同量级复现）。
2. **T2 content 组内频率梯度**：rho(B, tid) = −0.1617，p = 0.393，不显著——名词范围内频率不驱动 B（C medB −2.6059 vs R medB −2.6324 几乎相同）。
3. T3 tid 分层块置换：3/5 块混合 < 4 门，invalid 如实登记（F 低端 tids 过密）。
4. **T4 描述性：rho(tokid, B) 全 45 = −0.6464 强 vs content 组内 −0.16 ns——全样本强相关是组间均值差的 Simpson 结构，不是组内频率律**。

**结论（重复 3 次）**：**承重带 function-vs-content 差距（~1.46）是词类效应而非频率伪影——频率受控后硬显著（p 2e-4），且名词内部无频率梯度；机制签名分层格局定案：读出坐标层词盲（2940/2962）→ 路由头分布仅边界带（2962）→ 承重带 function/content 硬类效应（本 Phase）→ concrete/abstract 无签名（2962）。修正版思路一获得首个硬证据：功能词与内容词在承重带层携带不同机制签名，但类内细分（具体/抽象）不携带。**

**硬伤与勘误**：无 run1 失败；预冻结修正 2 处（T4 预测口径 rank 尺度 → raw tid 尺度；T2 置换循环续行在括号外的语法错）——均在运行前由 ast 检查拦截。工程：内联 python -c 补丁的 `\\` 续行被 shell 双层转义吞掉致 old-string 不匹配（新坑），改 Write 补丁文件 + Grep 复核。T3 块门 3/5 未达是设计现实（function tids 低端过密），Freedman-Lane 主检验不受影响。

**文件+SHA256-8**：execution @HEXE@ / result @HRES@ / freq_band.npz @HNpz@ / script @HSCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2963/frequency_controlled_band/`。Ledger 102 条 / L14 connects 70 / ledger @LEDHASH@。

**接续**：候选 2964：A（主选）**承重带类效应的载体解剖**——function 组 +1.46 差距在哪些层/头聚集（逐层 B 剖面对齐 + 头级 C17 分布 F vs C 的置换检验 + 与 2947 抵抗头/2953 早翻转头集的交集检查）；B S1 边界带功效复检（n≥60，F/C/R 扩容，承重带先验可提功效）；C 旋转轴功能身份（v_rot vs W_ov/u35 代数关系）；D 词类 × 机制签名卡片扩充（2961 卡组补 function/content 带签名行）。
'''
    stamp = created.replace('T', ' ')[:16]
    reps = [('@STAMP@', stamp), ('@VERDICT@', verdict),
            ('@LEDHASH@', chk),
            ('@HEXE@', hashes['execution.json']),
            ('@HRES@', hashes['result.json']),
            ('@HNpz@', hashes['freq_band.npz']),
            ('@HSCR@', hashes['script'])]
    for k, v in reps:
        section = section.replace(k, v)
    memo += section
    open(MEMO, 'w', encoding='utf-8').write(memo)

# ---------------- worklog ----------------
entry = ('- Phase 2963 闭环：band_class_effect_beyond_frequency（一次通过）。45 新词'
         '频率受控复检：T1 Freedman-Lane 组系数 -1.0371 p 2e-04 硬显著、T2 名词组内'
         '频率梯度 ns（C vs R medB 几乎相同）、T4 全样本 rho(tokid,B) -0.65 是组间 '
         'Simpson 结构——承重带 function/content 差距 ~1.46 是类效应非频率伪影。跨相位'
         '协议锚 a5 rel 7.76e-16 bit 级闭合。机制签名分层格局定案：读出词盲→路由边界带'
         '→承重带 function/content 硬签名→concrete/abstract 无。Ledger 102 / hash '
         + chk + '。\n')
wl = ''
if os.path.exists(WLOG):
    wl = open(WLOG, encoding='utf-8').read()
if 'Phase 2963' not in wl:
    wl += entry
    open(WLOG, 'w', encoding='utf-8').write(wl)

# ---------------- MEMORY ----------------
mem = open(WMEM, encoding='utf-8').read()
mem = mem.replace(
    '当前 max=**2962**，下一个 **2963**（候选 A function-vs-content 承重带'
    '频率受控复检；B S1 边界带功效复检 n≥60；C 旋转轴功能身份；D 跨模型'
    '差距清单）',
    '当前 max=**2963**，下一个 **2964**（候选 A 承重带类效应载体解剖；B S1 '
    '扩容功效复检；C 旋转轴功能身份；D 卡片扩充）')
old = '→词类签名矩阵：读出词盲复现+S1 边界带+承重带类内零差(2962)。'
if old in mem:
    mem = mem.replace(
        old,
        old[:-1] + '→频率受控复检：F/C 带差距=类效应 p 2e-4，Simpson 结构识破(2963)。')
add = ('- 跨组相关 Simpson 规范（2963）：全样本协变量相关可由组间均值差制造'
       '（rho(tokid,B) 全 45 = -0.6464 强，content 组内仅 -0.16 ns）——混淆'
       '判定必须"组内梯度 + 协变量控制组系数"双检验，全样本相关既不能证混淆'
       '也不能证无混淆；组均值差型混淆的正确 null = Freedman-Lane 残差置换'
       '（协变量回归 + 残差重排 + 全模型重拟合）。跨相位协议锚新规范：重前向'
       '源 Phase 词表 5 词对 B 比对，单前向同协议下 bit 级（rel 7.76e-16）'
       '闭合即协议恒等直接证明。科学结论：承重带 function/content 差距 ~1.46 '
       '为类效应（频率控制后 p 2e-4），名词内部无频率梯度（C≈R）；机制签名'
       '分层定案：读出坐标盲 → 路由边界带 → 承重带 F/C 硬签名 → 具体抽象无。\n')
if '跨组相关 Simpson 规范（2963）' not in mem:
    anchor_line = '- 词类分组设计规范（2962）：'
    i = mem.find(anchor_line)
    if i >= 0:
        mem = mem[:i] + add + mem[i:]
    else:
        mem = mem.rstrip() + '\n' + add
open(WMEM, 'w', encoding='utf-8').write(mem)

print('closeout done: ledger=%s n_meas=%s' % (
    chk, len(led['measurements'])))
