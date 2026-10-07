# -*- coding: utf-8 -*-
"""Phase 2979 closeout: seal + Ledger + MEMO + worklog + MEMORY."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
OUTD = os.path.join(
    BASE, r'tests\glm5\result\rdc_query_construction_20260913'
          r'\phase2979\reversal_anatomy')
SCRIPT = os.path.join(
    BASE, r'tests\glm5\phase2979_reversal_anatomy.py')
LEDGER = os.path.join(BASE, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(BASE, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
WSLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
         r'\.workbuddy\memory\2026-09-20.md')
MEMFILE = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
           r'\.workbuddy\memory\MEMORY.md')


def s8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


e = json.load(io.open(os.path.join(OUTD, 'execution.json'),
                      encoding='utf-8'))
STAMP = e['created'].replace('T', ' ')[:16]

shas = {
    'execution.json': s8(os.path.join(OUTD, 'execution.json')),
    'result.json': s8(os.path.join(OUTD, 'result.json')),
    'reversal_anatomy.npz':
        s8(os.path.join(OUTD, 'reversal_anatomy.npz')),
    'script': s8(SCRIPT),
}
print('seal:', shas)

r = json.load(io.open(os.path.join(OUTD, 'result.json'),
                      encoding='utf-8'))
verdict = r['final_verdict']
assert verdict == 'no_head_reversal_registered'
assert r['anchors']['ok'] is True

led = json.load(io.open(LEDGER, encoding='utf-8'))
old_sha = led.get('ledger_sha256_8')
led.pop('ledger_sha256_8', None)
n_before = len(led['measurements'])
meas_id = 'meas2979_reversal_anatomy'
led['measurements'].append({
    'meas_id': meas_id,
    'phase': 2979,
    'name': 'reversal_anatomy',
    'created': e['created'],
    'verdict': verdict,
    'artifacts': {
        'execution.json': shas['execution.json'],
        'result.json': shas['result.json'],
        'reversal_anatomy.npz': shas['reversal_anatomy.npz'],
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

sec = (u"""
## Phase 2979: 高剂量交互反转解剖——头格无正载体，反转是分布式弱项 [%(STAMP)s]

**设计**（2950 快照分解机器注入版，execution.json 先冻结）：Stage1 74 基线单前向（2977/2978 协议 verbatim），新增 per-head 贡献 c_h=u35·Wo17_h·x_h 与 raw L17 头输入切片捕获；轴方向重推导（a5 vs 2977 恒等 0.0）。Stage2 74 词 × 4 条件 {(0.1,0.1) 剂量恒等锚，(0.4,0)，(0,0.4)，(0.4,0.4)}（370 前向）。T0 层级 floor 门；T1 头格交互 I_h=median_w[c_h(11)-c_h(10)-c_h(01)+c_h(00)] 符号翻转 maxT 族 32（rng 2979）；T2 载体命运（S_hi vs 2978 S_lo=[0,4,7,12,13,15,25] + 随机头 null rng 2985）；T3 机制分解 g_h=||x_h(11)-x_h(00)||/(||dx10||+||dx01||)、cos_align=cos(dx11,dx10+dx01)（线性 g=1 cos=1）。

**run1 判决级 bug（对账纪律抓到）**：交互基线误用 (0.1,0.1) 条件而非 sham/00——与 2978 npz 24 词子集对账失配（median -0.0240 vs +0.0127，逐词 max|d| 6.3e-2），2970 恒等门制度生效。修正：stage1 捕获 base per-head + raw 切片，T0/T1/T3 回归 00 基线，删产物重跑（纪律 3）。

**run2 权威，锚 8/8**：a6 (0.1,0.1) vs 2977 prof11 bit 级 4.44e-16 / a7 头和≡profile 恒等 7.15e-16 / a3 0.00 / a5 0.0。

**判决：no_head_reversal_registered**（按冻结映射；负/混合结果如实登记）：
- **T0：median I(0.4,0.4)@L17 = +0.0081**（>0，压 0.008 门）——层级交互在双高剂量确实转正，但量级仅为低剂量峰（-0.0236）的 1/3；2978 报的 +0.0127 是 24 词子集读数（词集差异非协议漂移，(0.1,0.1) 锚 bit 级排除）；
- **T1：头格无正显著头**（sig+ 为空）；负交互仍头级显著：**h9 (-0.0129)、h12 (-0.0120)**——h12 ∈ S_lo（低剂量负载体头），**同头持续承载竞争**，无载体替换；
- **T2**：S_hi=[9,12]，与 S_lo 重叠 obs=1，null p=0.399 ns；
- **T3（全头）**：g=0.710（次线性增益）、cos_align=0.958（高方向对齐）——接近饱和挤压图景（方向保持、增益压缩），但 g 未塌过 0.5 门且无正头，挤压叙事只作描述性。

**结论（重复三遍）**：2978 的"高剂量反转"在权威基线与全集检验下**不是头级集中事件**——层级 +0.0081 转正由分布式弱项承载，竞争负交互仍由 h9/h12 同头持续显著；头输入空间全头 g=0.710/cos=0.958 提示联合响应方向保持、幅度次线性压缩（饱和挤压倾向），反转是"竞争项饱和压缩后剩余分布式项占优"的净效应，而非新正载体接管。2978 MEMO 的 biphasic 叙事补充修正：峰位与反转均来自层级聚合量，头级解剖不支持"正交互头"存在。

**产物**：`phase2979/reversal_anatomy/` execution %(s_exec)s / result %(s_res)s / reversal_anatomy.npz %(s_npz)s / script %(s_scp)s。

**接续（2980 候选）**：A（主选）竞争瓶颈头功能身份——h9/h12 消融（2965 机器：sep 读出 + B 带签名，竞争头因果必要性）；B 2978 dose law 分层修正复测（混桶 min_ab vs 单格剖面，纯离线 2978 npz）；C Omega-C 长上下文调制开题；D 同词跨语言语境对照。
""" % {'STAMP': STAMP,
       's_exec': shas['execution.json'],
       's_res': shas['result.json'],
       's_npz': shas['reversal_anatomy.npz'],
       's_scp': shas['script']})
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(sec)
print('memo appended')

wl = io.open(WSLOG, encoding='utf-8').read()
if 'Phase 2979' not in wl:
    entry = (u"\n## Phase 2979（2026-09-20）高剂量交互反转解剖\n"
             u"- 判决 no_head_reversal_registered（run2 权威，锚 8/8）。\n"
             u"- run1 判决级 bug：交互基线误用 (0.1,0.1) 而非 sham——2978 npz 子集对账失配（-0.0240 vs +0.0127）抓到，2970 恒等门制度生效。\n"
             u"- 核心：层级 +0.0081 转正是分布式弱项（头格 sig+ 空）；负交互 h9/h12 同头持续；全头 g=0.710/cos=0.958 饱和挤压倾向（描述性）。\n"
             u"- Ledger %d 条 / L14 %d / hash %s。\n"
             % (len(led['measurements']),
                len(l14['connects']), new_sha))
    with io.open(WSLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
    print('wslog appended')

mem = io.open(MEMFILE, encoding='utf-8').read()
old_next = (u'max=2978，下一个 **2979**（A 主选：高剂量交互反转快照分解'
            u'（2950 机器，直接项 vs 重平衡）；B 同词跨语言语境对照；'
            u'C Ω-C 长上下文开题；D 第三轴 2³ 扩展）。方案 v3 见')
new_next = (u'max=2979，下一个 **2980**（A 主选：竞争瓶颈头 h9/h12 消融'
            u'功能身份（2965 机器）；B 2978 dose law 分层修正复测（离线）；'
            u'C Ω-C 长上下文开题；D 同词跨语言语境对照）。方案 v3 见')
if old_next in mem:
    mem = mem.replace(old_next, new_next, 1)
else:
    print('WARN: memory next-candidate line not found')
old_chain = u'2973 fr 格塌缩=方向重写（能量占比<6%，与 2967/2970 带重合）。'
new_chain = (u'2973 fr 格塌缩=方向重写（能量占比<6%，与 2967/2970 带重合）→'
             u'2977 双轴注入竞争亚加法（L17 本体）→2978 交互剂量窗（biphasic，'
             u'混桶口径）→2979 权威基线：转正分布式无正头，h9/h12 同头持续，'
             u'g=0.71/cos=0.96 饱和挤压倾向。')
if old_chain in mem:
    mem = mem.replace(old_chain, new_chain, 1)
else:
    print('WARN: chain line not found')
io.open(MEMFILE, 'w', encoding='utf-8').write(mem)
print('memory updated')
print('closeout done')
