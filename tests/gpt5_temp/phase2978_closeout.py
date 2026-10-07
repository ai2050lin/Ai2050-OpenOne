# -*- coding: utf-8 -*-
"""Phase 2978 closeout: seal + Ledger + MEMO + worklog + MEMORY."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
OUTD = os.path.join(
    BASE, r'tests\glm5\result\rdc_query_construction_20260913'
          r'\phase2978\interaction_dose_headgrid')
SCRIPT = os.path.join(
    BASE, r'tests\glm5\phase2978_interaction_dose_headgrid.py')
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
    'interaction_dose_headgrid.npz':
        s8(os.path.join(OUTD,
                        'interaction_dose_headgrid.npz')),
    'script': s8(SCRIPT),
}
print('seal:', shas)

r = json.load(io.open(os.path.join(OUTD, 'result.json'),
                      encoding='utf-8'))
verdict = r['final_verdict']
assert verdict == 'subadditive_dose_windowed'
assert r['anchors']['ok'] is True

led = json.load(io.open(LEDGER, encoding='utf-8'))
old_sha = led.get('ledger_sha256_8')
led.pop('ledger_sha256_8', None)
n_before = len(led['measurements'])
meas_id = 'meas2978_interaction_dose_headgrid'
led['measurements'].append({
    'meas_id': meas_id,
    'phase': 2978,
    'name': 'interaction_dose_headgrid',
    'created': e['created'],
    'verdict': verdict,
    'artifacts': {
        'execution.json': shas['execution.json'],
        'result.json': shas['result.json'],
        'interaction_dose_headgrid.npz':
            shas['interaction_dose_headgrid.npz'],
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
## Phase 2978: Omega-B 交互剂量窗口与头格定位——亚加法呈 biphasic 剂量窗 [%(STAMP)s]

**设计**（方案 v3 Omega-B 第二步，execution.json 先冻结）：Stage1 74 基线单前向（2977 协议 verbatim），轴方向重推导（a5 vs 2977 npz 恒等 0.0）。T1 剂量网格：24 词子集（rng 2980 分层 6/格）× 25 条件 (a,b)∈{0,.05,.1,.2,.4}²（相对剂量 a·||x17_w||·d_lang + b·||x17_w||·d_cls，600 前向），交互 I(a,b)=R(a,b)-R(a,0)-R(0,b)+R(0,0)，主检验挂 L17（2977 唯一显著层），24 非零格符号翻转 maxT（rng 2981）。T2 头格：74 词 × 4 条件权威剂量 0.1（296 前向），per-head o_proj 输入格 I 逐层中位 + 逐层 maxT 族 32（rng 2982），登记性不入主判决（选拔/结构分账）。T3 重叠校准：L17 显著头 vs 2964 T3.sig_heads（随机头 null rng 2983×2000，纪律 11）。

**run1 权威，锚 8/8**：a1 3.04e-08 / a2 0.0 / a3 0.00+0.00 / a4 74/74 / a5 0.0 / a6 sham≡base bit 级 / a7 局部性 / a9 跨相位注入恒等（R(.1,.1)@L17 vs 2977 prof11 bit 级 0.0）。轴 cos=0.1144 复现。

**判决：subadditive_dose_windowed**（按冻结映射）：
- **T1 剂量律呈 biphasic 倒 U（重复三遍）**：交互随剂量先加深后反转——min_ab 剖面 0.05→-0.0118 / **0.1→-0.0144（峰）** / 0.2→-0.0091 / **0.4→+0.0127（符号反转）**；sum_ab 峰在 0.25-0.3（-0.0231）；24 非零格中仅 2 格过 maxT 门（(0.20,0.05) -0.0362 p 门内、(0.20,0.10) -0.0318）——亚加法是**剂量窗口现象**，高剂量下两轴竞争转为协同/饱和挤压；
- **T2 头格**：L17 显著头 [0,4,7,12,13,15,25]（7/32 头），全 36×32 格仅 8 格显著——交互载体头级集中但非单点；
- **T3 重叠校准（纪律 11）**：S17∩2964 载体={h15}，obs=1，随机头 null p=0.218 ns——**重叠不超过机会水平，不能作为载体身份证据**（与 2928 教训一致：选拔量重叠默认机会水平）。

**结论**：2977 的竞争性亚加法不是全域定律而是**剂量窗口现象**（biphasic：低剂量竞争抑制→高剂量反转），与 2958 剂量律/2959 饱和、2968 biphasic 峰位构成第三处 biphasic 证据——开关层非线性资源的竞争结构随注入强度发生质变。头级载体 7 头分布式集中，与 2964 载体头重叠为机会水平。

**产物**：`phase2978/interaction_dose_headgrid/` execution %(s_exec)s / result %(s_res)s / interaction_dose_headgrid.npz %(s_npz)s / script %(s_scp)s。

**接续（2979 候选）**：A（主选）高剂量交互反转的机制解剖——快照分解（2950 机器）：0.4 剂量下直接项 vs 竞争重平衡项符号审计，判定反转是饱和挤压还是方向对齐；B 同词跨语言语境对照（en 词入 fr 语境 vs 反之）；C Omega-C 长上下文调制开题（长度档 {2,16,64,256,1024}）；D Omega-B 第三轴扩展（语法/数量轴 2³ 因子）。
""" % {'STAMP': STAMP,
       's_exec': shas['execution.json'],
       's_res': shas['result.json'],
       's_npz': shas['interaction_dose_headgrid.npz'],
       's_scp': shas['script']})
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(sec)
print('memo appended')

wl = io.open(WSLOG, encoding='utf-8').read()
if 'Phase 2978' not in wl:
    entry = (u"\n## Phase 2978（2026-09-20）Ω-B 交互剂量窗口与头格定位\n"
             u"- 判决 subadditive_dose_windowed（run1 一次通过，锚 8/8，55s）。\n"
             u"- 核心：交互 biphasic 倒 U（min_ab 0.1 峰 -0.0144 → 0.4 反转 +0.0127）——亚加法是剂量窗口现象；L17 显著头 7 个 [0,4,7,12,13,15,25]；与 2964 载体重叠 obs=1 null p 0.218 机会水平。\n"
             u"- 跨相位恒等锚 a9（R(.1,.1) vs 2977 prof11 bit 级）与随机头 null 重叠校准入账。\n"
             u"- Ledger %d 条 / L14 %d / hash %s。\n"
             % (len(led['measurements']),
                len(l14['connects']), new_sha))
    with io.open(WSLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
    print('wslog appended')

mem = io.open(MEMFILE, encoding='utf-8').read()
old_next = (u'max=2977，下一个 **2978**（A 主选：交互剂量网格+头格定位'
            u'I(s_l,s_c)；B 同词跨语言语境对照；C Ω-C 长上下文开题；'
            u'D 配对语言效应确认）。方案 v3 见')
new_next = (u'max=2978，下一个 **2979**（A 主选：高剂量交互反转快照分解'
            u'（2950 机器，直接项 vs 重平衡）；B 同词跨语言语境对照；'
            u'C Ω-C 长上下文开题；D 第三轴 2³ 扩展）。方案 v3 见')
if old_next in mem:
    mem = mem.replace(old_next, new_next, 1)
    io.open(MEMFILE, 'w', encoding='utf-8').write(mem)
    print('memory updated')
else:
    print('WARN: memory next-candidate line not found')
print('closeout done')
