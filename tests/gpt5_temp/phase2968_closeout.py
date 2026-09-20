# -*- coding: utf-8 -*-
"""Phase 2968 closeout: seal + Ledger + MEMO + worklog + MEMORY.
Idempotent, placeholder replacement."""
import hashlib
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2968', 'h15_peak_anatomy')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-20.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2968_h15_peak_anatomy.py')

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
    'h15_peak.npz': sha8(os.path.join(OUT, 'h15_peak.npz')),
    'script': sha8(SCRIPT),
}

# ---------------- Ledger ----------------
led = json.load(open(LEDGER, encoding='utf-8'))
meas_id = 'meas2968_h15_peak_anatomy'
if not any(m['meas_id'] == meas_id
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': meas_id,
        'type': 'h15_peak_anatomy',
        'verdict': verdict,
        'source': ('phase2968/h15_peak_anatomy; two '
                   'families (+xdir anchor, -xdir '
                   'enhancement); anchors 13/13 incl '
                   'a12/a13 vs 2967 npz bit 0; T1 '
                   '40/57 inner-peak words, median peak '
                   '0.6746 vs s_c 0.6567 (diff 0.018), '
                   'boot CI [0.586,0.838]; T2 -xdir rho '
                   '-0.4455 p 0.169 ns (flat, no '
                   'mirror biphasic); T3 A11-peak vs '
                   'C15-peak per-word rho 0.9542'),
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
if '## Phase 2968' not in memo:
    section = '''## Phase 2968: h15 biphasic 峰位表征——个体峰位锁 s_c（分布中心），方向门控确认，峰位与 A11 增益峰同步 [@STAMP@]

**判决：`@VERDICT@`**（run1 权威一次通过）——2966 的 C15(s) 双相曲线是中位数曲线，须判定峰位是个体性质还是异质混合假象，并测反向（−xdir 增强侧）是否镜像。双族设计：family A +xdir（复制锚）+ family B −xdir，协议 2966/2967 verbatim（57 词、GRID17+s0、K=1）。

**锚 13/13**：a1-a11 与 2966/2967 逐项一致；**a12 C34 family A / a13 sep family A vs 2967 npz 全部 = 0.0（bit 级，第四次同实现复制验证）**。设计期拦下一个锚错误：初稿曾把"注入后 A11 相对基线漂移"误设为 a10 门——注入本来就要改变 A11（2966 L34 h15 曲线 0.35→0.45），该门必假失败，冻结前改为描述性 drift 登记（A11_L17 drift 1.00，描述性）。

**主检验**：
1. **T1 逐词峰位（主检验）**：**40/57 词有内峰**（g1 门 ≥20，冻结前预检可达性 40——纪律 10）；0 flat；**中位连续峰位 0.6746 vs s_c 0.6567（差 0.018 < 0.3，g2 过）**；bootstrap 95% CI [0.5862, 0.8375] 含 s_c——**双相性是个体性质：峰位分布的中心锁在 L17 开关阈值**，但个体峰位分布宽（0.29–1.61），是"分布中心锁"而非逐词精确锁。
2. **T2 反向族（−xdir）**：rho(s, C15) = −0.4455，p = 0.169 **ns**——曲线平坦（0.438→0.497→0.304，无镜像双相）；反向注入 sep 仅缓降（185→188→77.4 @ s=2，同剂量下 +xdir 已到 −6.4），B 端点 −0.6201 远未塌尽——**biphasic 是"关闭方向"特有瞬态，方向门控确认**。
3. **T3 描述性**：h8 内峰 48/57、h21（2953 早翻转头）42/57——双相响应不限于 h15；**A11_L34_h15 峰位 vs C15 峰位逐词 rho = 0.9542（26 对）——h15 瞬态峰与路由增益峰逐词同步**，词类瞬态载体与深层路由增益在个体水平耦合。

**结论（重复 3 次）**：**h15 的 biphasic 峰位是个体性质（40/57 词内峰、中位峰位与 s_c 差 0.018、CI 含 s_c），峰位与 A11 路由增益峰逐词同步（rho 0.954）——词类瞬态载体在个体水平被 L17 路由增益驱动；但响应方向门控：反向注入下 h15 平坦无镜像（p 0.169），双相是"语言信号关闭方向"特有的过冲-回落瞬态，不是对称剂量响应。2966 的"独立于路由"判决需精化：h15 非路由状态的单调跟随者，但峰位机制上耦合路由增益——瞬态放大器随开关过冲。**

**硬伤与勘误**：无 run 失败；设计期假锚拦截（A11 drift 误设 a10，冻结前修正为描述性——判据可达性纪律 10 的锚设计应用）。

**文件+SHA256-8**：execution @HEXE@ / result @HRES@ / h15_peak.npz @HNpz@ / script @HSCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2968/h15_peak_anatomy/`。Ledger 107 条 / L14 connects 75 / ledger @LEDHASH@。

**接续**：候选 2969：A（主选）**峰位分布的词属性解释**（个体峰位 0.29-1.61 宽分布——峰位与词的语言标签/tid/token 频率相关？2940-2963 的词类机器可直接套用，判定路由过冲幅度的个体差异来源）；B L34 残差 signature（0.51×gap 头级构成，2967 npz 离线）；C S1 路由边界带扩容复检（n≥60 加 L34）；D 2961 卡组扩充（补 2962-2968 八行）。
'''
    stamp = created.replace('T', ' ')[:16]
    reps = [('@STAMP@', stamp), ('@VERDICT@', verdict),
            ('@LEDHASH@', chk),
            ('@HEXE@', hashes['execution.json']),
            ('@HRES@', hashes['result.json']),
            ('@HNpz@', hashes['h15_peak.npz']),
            ('@HSCR@', hashes['script'])]
    for k, v in reps:
        section = section.replace(k, v)
    memo += section
    open(MEMO, 'w', encoding='utf-8').write(memo)

# ---------------- worklog ----------------
entry = ('- Phase 2968 闭环：biphasic_locked_no_reverse（run1 一次通过，锚 13/13，'
         'a12-13 vs 2967 bit 0）。h15 双相峰位是个体性质：40/57 词内峰，中位峰位 '
         '0.6746 vs s_c 0.6567（差 0.018），CI [0.586,0.838]；峰位与 A11 路由增益峰'
         '逐词 rho 0.954——瞬态峰由路由增益个体驱动；反向 -xdir 侧平坦（rho -0.45 ns）'
         '方向门控：biphasic 是关闭方向特有瞬态。2966 判决精化：非路由跟随者但峰位'
         '耦合路由增益（过冲放大器）。设计期拦截假锚（A11 drift 误设 a10）。'
         'Ledger 107 / hash ' + chk + '。\n')
wl = ''
if os.path.exists(WLOG):
    wl = open(WLOG, encoding='utf-8').read()
if 'Phase 2968' not in wl:
    wl += entry
    open(WLOG, 'w', encoding='utf-8').write(wl)

# ---------------- MEMORY ----------------
mem = open(WMEM, encoding='utf-8').read()
mem = mem.replace(
    '当前 max=**2967**，下一个 **2968**（候选 A h15 biphasic 峰位表征（逐词峰位 vs '
    's_c、跨方向泛化）；B L34 残差 signature（0.51×gap 头级构成，npz 离线）；'
    'C S1 扩容复检；D 卡片扩充）',
    '当前 max=**2968**，下一个 **2969**（候选 A 峰位分布的词属性解释（峰位 vs 语言'
    '标签/tid/频率）；B L34 残差 signature（npz 离线）；C S1 扩容复检；D 卡片扩充）')
old = '→塌缩载体：B 抹平由 L24-33 八层 318 头分布式承载，'\
      'L34 显著响应但端点比 0.51 未塌尽(2967)。'
if old in mem:
    mem = mem.replace(
        old,
        old[:-1] + '→峰位表征：biphasic 个体峰锁 s_c（中位差 0.018，'
                   'A11 峰同步 rho 0.954），方向门控反向平坦(2968)。')
add = ('- 注入族锚的假门陷阱（2968）：把"注入后状态量相对基线的漂移"设为锚门必假失败'
       '——注入的意义就是改变该量；锚只能设恢复残差/确定性/跨相位 bit 级复现类。'
       '科学结论：h15 biphasic 峰位是个体性质（40/57 内峰、中位峰位 0.6746 vs '
       's_c 0.6567、boot CI [0.586,0.838]），峰位与 A11_L34_h15 增益峰逐词 rho '
       '0.954——瞬态过冲由路由增益个体驱动；反向 -xdir 注入下 h15 平坦（rho -0.45 '
       'p 0.169 ns）、sep 仅缓降——双相是关闭方向特有，方向门控。2966 判决精化：'
       'h15 非路由单调跟随者，但峰位机制耦合路由增益（过冲放大器）。\n')
if '注入族锚的假门陷阱（2968）' not in mem:
    anchor_line = '- np.where 广播陷阱（2967）：'
    i = mem.find(anchor_line)
    if i >= 0:
        mem = mem[:i] + add + mem[i:]
    else:
        mem = mem.rstrip() + '\n' + add
open(WMEM, 'w', encoding='utf-8').write(mem)

print('closeout done: ledger=%s n_meas=%s' % (
    chk, len(led['measurements'])))
