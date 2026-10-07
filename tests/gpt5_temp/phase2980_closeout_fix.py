# -*- coding: utf-8 -*-
"""Phase 2980 closeout fix: Ledger already written (119/cc4bf46a);
append MEMO + worklog + MEMORY only (percent-escaped)."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
OUTD = os.path.join(
    BASE, r'tests\glm5\result\rdc_query_construction_20260913'
          r'\phase2980\bottleneck_head_identity')
SCRIPT = os.path.join(
    BASE, r'tests\glm5\phase2980_bottleneck_head_identity.py')
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
    'bottleneck_head_identity.npz':
        s8(os.path.join(OUTD, 'bottleneck_head_identity.npz')),
    'script': s8(SCRIPT),
}
r = json.load(io.open(os.path.join(OUTD, 'result.json'),
                      encoding='utf-8'))
verdict = r['final_verdict']
assert verdict == 'bottleneck_heads_carry_interaction'

# Ledger idempotent verify (already written by first closeout run)
led = json.load(io.open(LEDGER, encoding='utf-8'))
stored = led.pop('ledger_sha256_8')
calc = hashlib.sha256(json.dumps(
    led, sort_keys=True, ensure_ascii=False)
    .encode('utf-8')).hexdigest()[:8]
l14 = [l for l in led['linkage']
       if l['link_id'] == 'L14_readout_spectrum_cross_model'][0]
assert 'meas2980_bottleneck_head_identity' in l14['connects']
assert stored == calc, 'ledger hash mismatch'
n_meas = len(led['measurements'])
n14 = len(l14['connects'])
print('ledger verify: n=%d L14=%d sha=%s self-consistent'
      % (n_meas, n14, stored))

sec = (u"""
## Phase 2980: 竞争瓶颈头功能身份——h12 是双轴交互的必要载体 [%(STAMP)s]

**设计**（2965 消融机器 × 2977 注入协议，execution.json 先冻结）：74 词（2979 npz verbatim）；Stage0 intact 74 前向（a3 锚 n17=**L17 层输入** 2560 维范数 vs 2979 npz）；Stage1 剂量 0.1 无消融 4 条件×74（a5 锚 I(0.1,0.1)=prof11-prof10-prof01+prof00@L17 vs 2977 npz **逐词 bit 级 0.00**）；Stage2 消融 {h9, h12, both, rand(rng 29804 抽 h21)} × 4 条件 × 74（1550 前向）。消融 = o_proj 输入切片置零（多头 hook 列表版）；注入 = self_attn pre-hook x[:,1,:] += dt（2977 verbatim，剂量 s_w=0.1·n17，**注入向量=2979 单位方向 d_lang_u/d_cls_u**）。T1 配对符号翻转置换（10000，rng 29800-29802）；T2 随机头对照同统计（描述性校准）；T3 消融主效应 mean|dB| 描述。

**两次 correction（均删产物重跑）**：run1 锚 a2/a3 失败——a2 误把 2979 单位向量 vs 2977 raw 向量做 max|d|（应比方向 cos）；a3 把 n17 算成 o_proj 输入（4096 维）范数，实为 L17 **层输入**（2560 维，self_attn pre-hook）范数（2952 hook 教训再现）。run2 a5 失败（max|d| 0.481）+ UnboundLocalError——根因：**2977 npz 存的 d_lang/d_cls 是脚本内归一化前的 raw mean-diff 向量（范数 4.47/13.90），实际注入的是单位向量**（2979 a5 bit 级复现为证）；run2 误用 raw 向量致剂量放大 4.5-14 倍。run3 = 2979 单位方向 + x17 层输入捕获，锚 7/7。

**判决：bottleneck_heads_carry_interaction**（按冻结映射；锚 7/7，a5/a6 均 0.00）：
- **T1 h12：交互消失 97%%**——|median I| 0.02358 → 0.00056，dI +0.02327（p 1e-4 置换下界）；
- **T1 both：同 h12**（→0.00033）——h9 消融几乎无效应（dI +0.00018，|I| 0.02358→0.02348，效应量微小但方向一致过门）；
- **T2 随机头 h21：dI 仅 −0.00029**（p 1e-4，量级比 h12 小 80 倍）——消融效应头特异，非全局破坏；三 obs |median| 中 0.33 低于 rand（h9 低于、h12/both 远高于），pct=0.33 描述性登记；
- **T3 mean|dB|：h9 0.0102 / h12 0.0134 / both 0.0117 / rand 0.0129**——四条件均衡，h12 消融不是经全局 B 带破坏间接抹掉交互，交互消失是头特异功能通道被切断。

**结论（重复三遍）**：**L17 双轴竞争亚加法交互的因果载体是 h12（h9 仅边际参与）**——消融 h12 使交互消失 97%% 而随机头消融无效应（<2%%），且不伴随全局 B 带破坏（T3 均衡）。2979 "同头持续" 的相关性证据（h12∈S_lo 且高剂量仍显著）由此升级为因果必要性证据；与 2978 S_lo 头级集中、2977 交互生成于注入层本体构成完整链条：**开关层 L17 的竞争瓶颈在头级可操作化定位到 h12**——这是机制链中第一个"单头消融即消除交互效应"的个案（对比 2965 h15 共享载体、2949 组水平反转），说明交互项不同于主效应（主效应分布式、交互项头级集中）。

**产物**：`phase2980/bottleneck_head_identity/` execution %(s_exec)s / result %(s_res)s / bottleneck_head_identity.npz %(s_npz)s / script %(s_scp)s。

**接续（2981 候选）**：A（主选）h12 功能画像——它承载交互的机制分解（g/cos 头级版 + 2965 T1 write identity：c_h 方向、与 u35 关系）；B h12 消融的主效应归宿（交互消失后 B 带签名是否迁移到其他头，重平衡检验）；C Omega-C 长上下文调制开题；D 2978 dose law 分层修正复测（离线）。
""" % {'STAMP': STAMP,
       's_exec': shas['execution.json'],
       's_res': shas['result.json'],
       's_npz': shas['bottleneck_head_identity.npz'],
       's_scp': shas['script']})

memo_txt = io.open(MEMO, encoding='utf-8').read()
if '## Phase 2980:' not in memo_txt:
    with io.open(MEMO, 'a', encoding='utf-8') as f:
        f.write(sec)
    print('memo appended')
else:
    print('memo already present, skip')

wl = io.open(WSLOG, encoding='utf-8').read()
if 'Phase 2980' not in wl:
    entry = (u"\n## Phase 2980（2026-09-20）竞争瓶颈头功能身份\n"
             u"- 判决 bottleneck_heads_carry_interaction（run3 权威，锚 7/7，a5/a6 均 0.00）。\n"
             u"- 两次 correction：run1 a2/a3 锚对象错（单位向量 vs raw 向量；n17 应为 L17 层输入 2560 维范数）；run2 误用 2977 npz raw 向量致剂量放大 4.5-14 倍（根因：2977 npz 存归一化前向量）。\n"
             u"- 核心：消融 h12 使交互消失 97%（0.02358→0.00056），随机头 <2%，T3 mean|dB| 均衡（非全局破坏）——交互项头级集中、主效应分布式的头级分化。\n"
             u"- Ledger %d 条 / L14 %d / hash %s。\n"
             % (n_meas, n14, stored))
    with io.open(WSLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
    print('wslog appended')
else:
    print('wslog already present, skip')

mem = io.open(MEMFILE, encoding='utf-8').read()
changed = False
old_next = (u'max=2979，下一个 **2980**（A 主选：竞争瓶颈头 h9/h12 消融'
            u'功能身份（2965 机器）；B 2978 dose law 分层修正复测（离线）；'
            u'C Ω-C 长上下文开题；D 同词跨语言语境对照）。方案 v3 见')
new_next = (u'max=2980，下一个 **2981**（A 主选：h12 功能画像——write '
            u'identity + 头级 g/cos 机制分解；B h12 消融主效应归宿'
            u'（重平衡检验）；C Ω-C 长上下文开题；D 2978 dose law '
            u'分层修正复测）。方案 v3 见')
if old_next in mem:
    mem = mem.replace(old_next, new_next, 1)
    changed = True
elif 'max=2980' not in mem:
    print('WARN: memory next-candidate line not found')
old_chain = (u'2979 权威：转正分布式无正头，h9/h12 同头持续，'
             u'g=0.71/cos=0.96 饱和挤压倾向。')
new_chain = (u'2979 权威：转正分布式无正头，h9/h12 同头持续，'
             u'g=0.71/cos=0.96 饱和挤压倾向→2980 消融定案：'
             u'h12 是交互必要载体（消融消失 97%，随机头 <2%，'
             u'无全局破坏）——交互项头级集中、主效应分布式。')
if old_chain in mem:
    mem = mem.replace(old_chain, new_chain, 1)
    changed = True
elif '2980 消融定案' not in mem:
    print('WARN: chain line not found')
if changed:
    io.open(MEMFILE, 'w', encoding='utf-8').write(mem)
    print('memory updated')
print('closeout-fix done')
