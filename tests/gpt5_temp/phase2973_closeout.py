# -*- coding: utf-8 -*-
"""Phase 2973 closeout: seal + Ledger + MEMO + worklog + MEMORY."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
OUTD = os.path.join(
    BASE, r'tests\glm5\result\rdc_query_construction_20260913'
          r'\phase2973\fr_scale_audit')
SCRIPT = os.path.join(
    BASE, r'tests\glm5\phase2973_fr_scale_audit.py')
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
    'fr_scale_audit.npz':
        s8(os.path.join(OUTD, 'fr_scale_audit.npz')),
    'script': s8(SCRIPT),
}
print('seal:', shas)

r = json.load(io.open(os.path.join(OUTD, 'result.json'),
                      encoding='utf-8'))
verdict = r['final_verdict']
assert verdict == 'language_effect_energy_and_direction'
assert r['anchors']['ok'] is True

led = json.load(io.open(LEDGER, encoding='utf-8'))
old_sha = led.get('ledger_sha256_8')
led.pop('ledger_sha256_8', None)
n_before = len(led['measurements'])
meas_id = 'meas2973_fr_scale_audit'
led['measurements'].append({
    'meas_id': meas_id,
    'phase': 2973,
    'name': 'fr_scale_audit',
    'created': e['created'],
    'verdict': verdict,
    'artifacts': {
        'execution.json': shas['execution.json'],
        'result.json': shas['result.json'],
        'fr_scale_audit.npz': shas['fr_scale_audit.npz'],
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
## Phase 2973: fr 格 B 尺度审计——norm/cos 分解判方向塌缩主导，norm 伪影怀疑被否定 [%(STAMP)s]

**设计**：2972 协议 verbatim 复跑（词表与顺序直接读 2972 execution.json 恒等门，77 单前向 [the, w]，o_proj 输入 pos-1 全 36 层），对每词每层把读出投影 prof[li] = x·M[li] 分解为能量因子 norm[li]=||x|| 与方向因子 coss[li]=cos(x, M[li])（2937 机器）。T1 norm 语言效应（Freedman-Lane reduced 1+cls+cov，残差置换 rng 2973×10000，maxT 族 36，门 p≤0.01）；T2 cos 语言效应（rng 2974）；T3 描述性带内 log 分解（能量/cos 占比，L6-12 与 L28-35）。

**锚 6/6（run1 一次通过）**：a1 3.04e-08 / a2 0.0 / **a3 B 恒等门 vs 2972 npz max rel 3.40e-15（74/74，2970 口径恒等门制度生效）** / a4 77/77 单 token / a5 3.49e-16 / a6 norms>0 且 max|cos| 0.1467。

**判决：language_effect_energy_and_direction**（按冻结映射）：
- **T1 norm**：仅 L35 显著（coef +7.70，p 2e-4）；L30 中位 norm en 15.3-15.8 → fr 10.5-11.4（深层真实下降约 30%%）但族内只 L35 过门。
- **T2 cos**：8 层显著 [0, 24, 26, 27, 30, 31, 32, 34]，系数全负（fr cos 更低），top (31, -0.0822, p 1e-4)——**显著层与 2967 塌缩带（L24-33）、2970 延迟带（L24-34）高度重叠**。
- **T3 带内分解（描述性）**：能量占比仅 **3.9%%（L6-12）/ 5.1%%（L28-35）**——B 塌缩 ~95%% 由 cos 方向因子承载。格中位：cos_L30 F_en 0.0412 → F_fr 0.0011；C_en 0.1021 → C_fr -0.0030（塌到零）。

**结论**：2972 登记的"fr 格 B 趋零疑为 norm 伪影"怀疑**被否定**——主因是深层带的方向塌缩（cos→0），norm 下降真实但次要（占比 <6%%）。机制链新增一环：code-switching 语境下语言效应 = 深带方向重写（与 2937/2938"旋转/重写而非能量丢失"命名一致），且塌缩层集与 2967/2970 塌缩-延迟带重合——语言轴与重编码机制共享同一深层载体带。2972 的语言-词类结构（语言主调制×词类英文特化）由此获得尺度解释：fr 词在英文语境中于 L24-34 被"方向重写"，其 B 读出坐标被清零但能量保留。

**产物**：`phase2973/fr_scale_audit/` execution %(s_exec)s / result %(s_res)s / fr_scale_audit.npz %(s_npz)s / script %(s_scp)s。

**接续（2974 候选）**：A（主选）方案 v3 Ω-A 跨模块子空间对齐（零前向，Attention 输出列空间 × MLP up-proj 行空间主角度 + 随机头 null 三件套，2931 教训强制 null 校准）；B fr 方向塌缩子空间检查（2938 机器：塌缩是否限单方向、子空间对齐是否保留）；C 配对语言效应确认性检验（新 concept 对预注册）；D 方案 v3 Ω-B 多轴融合代数（2^3 因子注入）。
""" % {'STAMP': STAMP,
       's_exec': shas['execution.json'],
       's_res': shas['result.json'],
       's_npz': shas['fr_scale_audit.npz'],
       's_scp': shas['script']})
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(sec)
print('memo appended')

wl = io.open(WSLOG, encoding='utf-8').read()
if 'Phase 2973' not in wl:
    entry = (u"\n## Phase 2973（2026-09-20）fr 格 B 尺度审计\n"
             u"- 判决 language_effect_energy_and_direction（锚 6/6，run1 一次通过；a3 B 恒等门 3.40e-15）。\n"
             u"- norm 伪影怀疑被否定：能量占比仅 3.9%%/5.1%%（L6-12/L28-35），~95%% 由 cos 方向塌缩承载；norm 仅 L35 显著（+7.70，p 2e-4）。\n"
             u"- cos 显著层 [0,24,26,27,30,31,32,34] 与 2967 塌缩带/2970 延迟带高度重叠——语言效应=深带方向重写，共享载体带。\n"
             u"- Ledger %d 条 / L14 %d / hash %s。\n"
             % (len(led['measurements']),
                len(l14['connects']), new_sha))
    with io.open(WSLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
    print('wslog appended')

mem = io.open(MEMFILE, encoding='utf-8').read()
old_next = (u'max=2972，下一个 **2973**（A 主选：fr 格 B 尺度审计 '
            u'norm/cos 分解，2972 npz 离线；B 方案 v3 Ω-A 跨模块'
            u'对齐；C 配对语言效应确认；D Ω-B 多轴融合）')
new_next = (u'max=2973，下一个 **2974**（A 主选：方案 v3 Ω-A 跨模块'
            u'子空间对齐——主角度+随机头 null 三件套；B fr 方向塌缩'
            u'子空间检查（2938 机器）；C 配对语言效应确认；D Ω-B '
            u'多轴融合）')
if old_next in mem:
    mem = mem.replace(old_next, new_next, 1)
    io.open(MEMFILE, 'w', encoding='utf-8').write(mem)
    print('memory updated')
else:
    print('WARN: memory next-candidate line not found')
print('closeout done')
