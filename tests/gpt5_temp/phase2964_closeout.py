# -*- coding: utf-8 -*-
"""Phase 2964 closeout: seal + Ledger + MEMO + worklog + MEMORY.
Idempotent, placeholder replacement."""
import hashlib
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2964', 'carrier_anatomy')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-20.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2964_carrier_anatomy.py')

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
    'carrier_anatomy.npz': sha8(
        os.path.join(OUT, 'carrier_anatomy.npz')),
    'script': sha8(SCRIPT),
}

# ---------------- Ledger ----------------
led = json.load(open(LEDGER, encoding='utf-8'))
meas_id = 'meas2964_carrier_anatomy'
if not any(m['meas_id'] == meas_id
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': meas_id,
        'type': 'carrier_anatomy',
        'verdict': verdict,
        'source': ('phase2964/carrier_anatomy; 30 FRESH '
                   'words (F15 closed-class tid 566-7241 / '
                   'N15 nouns 3241-26752) + 3 anchors; '
                   'anchors 6/6 incl cross-phase B vs 2963 '
                   'rel 3.49e-16; T1 FL replication coef '
                   '-1.0515 p 1.5e-03; T2 unique maxT '
                   'significant layer L34 (gap -4.0, '
                   'top-3 conc 0.461); T3 unique significant '
                   'head L34/h15 (-1.66); T4 rho(gap17, '
                   'D2947) 0.4296, top5 intersects 2953 '
                   'flipper h21 and keep-set 5/5'),
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
if '## Phase 2964' not in memo:
    section = '''## Phase 2964: 承重带类效应载体解剖——L34/h15 单点头定位 [@STAMP@]

**判决：`@VERDICT@`**（run3 权威）——2963 类效应的载体定位：30 个全新词（F15 封闭类功能词 tid 566-7241（12 代词+without/among/unless）/ N15 内容名词 3241-26752）+ 3 个 2963 词重前向锚；2937 pass1 协议 verbatim，全 36 层 o_proj 输入捕获 → C[36,30,32] 头级贡献矩阵。

**锚 6/6**：a1 3.04e-08 / a2 0.0 / a3 2.43e-16 / a4 单 token 33/33 + 新鲜 30/30 / **a5 跨相位 B vs 2963 rel 3.49e-16（bit 级）** / a6 非退化（36/36 层、32/32 头）。

**四检验**：
1. **T1 确认性复制**：Freedman-Lane 组系数 −1.0515，p = 1.5e-03——2963 的类效应在全新词表第二次复制（medB F −1.3763 vs N −2.7501，gap 1.37，两次独立词表 1.23/1.46/1.37 稳定）。
2. **T2 层定位（maxT 族 36）**：唯一显著层 **L34**（gap −4.0）；top5 = L34(−4.0) / L16(−1.91) / L30(−1.88) / L32(−1.21) / L31(−1.19)；top-3 集中度 0.461——**类效应载体是深层 L34 主导 + L16/L30-32 次级的双带结构**，与 2932 承重带（L6-12）不同带。
3. **T3 头定位（maxT 族 32，L34）**：唯一显著头 **h15**（gap −1.66）；top5 = h15/h8/h21/h28/h11。
4. **T4 描述性**：rho(gap17, 2947 D_L17) = 0.4296（中度同构）；top5×2947-top5 **空交集**（头级类效应 ≠ 语言轴头级重要性排序的 top 集）；×2953 早翻转头 **{21}**（h21 入 top5）；×keep_L17 = {8,11,15,21,28} 5/5 全在非退化集内。

**结论（重复 3 次）**：**承重带 function-vs-content 类效应的载体定位为 L34/h15（层/头级 maxT 唯一显著），深层双带（L34 主导 + L16/L30-32 次级）；类效应头与语言轴头部分分离（秩序相关 0.43 但 top 集不交叠），且 2953 早翻转头 h21 重新出现——功能词 vs 内容词的机制差异是少数深层头的专职分工，与语言轴（en/zh）机制在头级部分解耦。**

**硬伤与勘误（run1→run3）**：run1 KeyError——`func_tid='the'` 依赖词表循环副作用而 'the' 不在本 Phase 词表（跨 Phase 协议常量必须显式重建，新教训入 MEMORY）；run2 T3 切片 2D/3D 索引错；run3 T4 set/list 类型错——三连工程 bug 均报错定位 + 补丁文件修复 + 纪律 3 清理重跑。数据结论三次运行完全一致（前向锚 bit 级）。

**文件+SHA256-8**：execution @HEXE@ / result @HRES@ / carrier_anatomy.npz @HNpz@ / script @HSCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2964/carrier_anatomy/`。Ledger 103 条 / L14 connects 71 / ledger @LEDHASH@。

**接续**：候选 2965：A（主选）**L34/h15 功能身份判定**——h15 的 W_ov 头级切片谱（2948 方法）、A11 剂量响应（2953 方法挂 L34）、与 u35/词类轴的代数关系（h15 输出方向 vs dirs_word/词类判别方向的投影），判定它是"词类读出头"还是"词类抑制头"；B S1 路由边界带扩容复检（n≥60，加 L34 进路由层集）；C 旋转轴功能身份；D 词类签名卡片扩充（2961 卡组补 L34/h15 行）。
'''
    stamp = created.replace('T', ' ')[:16]
    reps = [('@STAMP@', stamp), ('@VERDICT@', verdict),
            ('@LEDHASH@', chk),
            ('@HEXE@', hashes['execution.json']),
            ('@HRES@', hashes['result.json']),
            ('@HNpz@', hashes['carrier_anatomy.npz']),
            ('@HSCR@', hashes['script'])]
    for k, v in reps:
        section = section.replace(k, v)
    memo += section
    open(MEMO, 'w', encoding='utf-8').write(memo)

# ---------------- worklog ----------------
entry = ('- Phase 2964 闭环：carrier_localized_layers_heads（run3）。类效应载体定位：'
         '唯一 maxT 显著层 L34（gap -4.0）+ 唯一显著头 L34/h15（-1.66）；深层双带 '
         'L34+L16/L30-32；T1 FL 复制 p 1.5e-03（第三次词表 1.23/1.46/1.37 稳定）；'
         'T4 rho(gap17,D2947) 0.43 但 top5 不交叠、h21 入 top5、5/5 在 keep 集。'
         '勘误：run1 func_tid 副作用依赖 KeyError、run2 切片索引、run3 set 类型——'
         '3 连补丁重跑，数据 bit 级一致。Ledger 103 / hash ' + chk + '。\n')
wl = ''
if os.path.exists(WLOG):
    wl = open(WLOG, encoding='utf-8').read()
if 'Phase 2964' not in wl:
    wl += entry
    open(WLOG, 'w', encoding='utf-8').write(wl)

# ---------------- MEMORY ----------------
mem = open(WMEM, encoding='utf-8').read()
mem = mem.replace(
    '当前 max=**2963**，下一个 **2964**（候选 A 承重带类效应载体解剖；B S1 '
    '扩容功效复检；C 旋转轴功能身份；D 卡片扩充）',
    '当前 max=**2964**，下一个 **2965**（候选 A L34/h15 功能身份判定；B S1 '
    '扩容复检加 L34；C 旋转轴功能身份；D 卡片扩充）')
old = '→频率受控复检：F/C 带差距=类效应 p 2e-4，Simpson 结构识破(2963)。'
if old in mem:
    mem = mem.replace(
        old,
        old[:-1] + '→载体解剖：L34/h15 单点定位，类效应头≠语言轴头(2964)。')
add = ('- 协议常量显式重建规范（2964）：跨 Phase 复用的协议常量（如 '
       "func_tid='the'）必须显式从 tokenizer 重建，禁止依赖词表循环副作用"
       '（run1 KeyError）；运行时 maxT 显著集登记须注明星族口径（层族 36 / '
       '头族 32 分开）。科学结论：F/C 类效应载体 = L34 主导（gap -4.0）'
       '+ h15 唯一显著头，深层双带 L34+L16/L30-32；类效应头与语言轴头'
       '秩序相关 0.43 但 top5 零交叠（部分解耦）；2953 早翻转头 h21 入'
       '类效应 top5；三词表 B gap 稳定（1.23/1.46/1.37）。\n')
if '协议常量显式重建规范（2964）' not in mem:
    anchor_line = '- 跨组相关 Simpson 规范（2963）：'
    i = mem.find(anchor_line)
    if i >= 0:
        mem = mem[:i] + add + mem[i:]
    else:
        mem = mem.rstrip() + '\n' + add
open(WMEM, 'w', encoding='utf-8').write(mem)

print('closeout done: ledger=%s n_meas=%s' % (
    chk, len(led['measurements'])))
