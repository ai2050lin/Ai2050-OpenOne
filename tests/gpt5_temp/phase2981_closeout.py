# -*- coding: utf-8 -*-
"""Phase 2981 closeout: seal + Ledger + MEMO + worklog + MEMORY."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
OUTD = os.path.join(
    BASE, r'tests\glm5\result\rdc_query_construction_20260913'
          r'\phase2981\h12_functional_portrait')
SCRIPT = os.path.join(
    BASE, r'tests\glm5\phase2981_h12_functional_portrait.py')
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
    'h12_functional_portrait.npz':
        s8(os.path.join(OUTD, 'h12_functional_portrait.npz')),
    'script': s8(SCRIPT),
}
print('seal:', shas)

r = json.load(io.open(os.path.join(OUTD, 'result.json'),
                      encoding='utf-8'))
verdict = r['final_verdict']
assert verdict == 'h12_input_sublinear_carrier', verdict
assert r['anchors']['ok'] is True

led = json.load(io.open(LEDGER, encoding='utf-8'))
old_sha = led.get('ledger_sha256_8')
led.pop('ledger_sha256_8', None)
n_before = len(led['measurements'])
meas_id = 'meas2981_h12_functional_portrait'
led['measurements'].append({
    'meas_id': meas_id,
    'phase': 2981,
    'name': 'h12_functional_portrait',
    'created': e['created'],
    'verdict': verdict,
    'artifacts': {
        'execution.json': shas['execution.json'],
        'result.json': shas['result.json'],
        'h12_functional_portrait.npz':
            shas['h12_functional_portrait.npz'],
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
## Phase 2981: h12 功能画像——交互非线性在 o_proj 输入响应处生成（亚线性载体） [%(STAMP)s]

**设计**（2980 注入协议 verbatim，execution.json 先冻结）：74 词/单位方向/n17 全部 2979 npz verbatim；Stage0 intact 74 前向；Stage1 74×4 条件 {00,10,01,11} dose 0.1（2979 _u dirs，s_w=0.1·n17），捕获 L17 o_proj 输入按头切片 x_h(32×128)。判据逻辑（冻结）：c_h=u35·Wo17_h·x_h 对 x_h 线性 → 非零 I_h 要求 x_h 响应本身非线性——g=||Δx11||/(||Δx10||+||Δx01||) 或 cos(Δx11, Δx10+Δx01)。

**锚 7/7**（a5 层级 I vs 2977 npz **逐词 bit 级 0.00**；a6 头级 I median vs 2978 I_grid_med[17] **bit 级 0.00**——双跨产物恒等门；a3 1.21e-07；a7 分母门 74/74）。

**correction（1 次，删产物重跑）**：run1 T3 崩溃——u35@Wo17[:,h] 是 128 维头输入空间向量，不能与 2560 维层输入轴点积（跨空间 matmul）；T1/T2 判决量未受影响，T3 改同空间量（写核范数排名+通道重叠 cos(dx10,dx01)+响应范数）。

**判决：h12_input_sublinear_carrier**（按冻结映射）：
- T1：I_h12 median -0.023274（p 1e-4），占层交互 median 的 **98.7%%**；I_h9 -0.000182（p 1e-4）占比仅 0.8%%——2980 消融结论的观测面复现（无消融捕获下 h12 承载几乎全部层交互）；
- T2：**h12 输入响应亚线性**——g median 0.8896（vs 1，p 1e-4）、cos 0.9065（p 1e-4）；全头 g 0.6985 / cos 0.9637——h12 的 g 反而高于全体（离线性更近）；
- T3：写核范数 h12 0.3075（rank 7/32）、h9 0.3694（rank 2/32，max 0.4910）；两轴响应通道重叠 cos(dx10,dx01) median 0.5485。

**结论（重复三遍）**：**L17 双轴竞争交互的非线性在 h12 的 o_proj 输入响应处生成——两轴联合注入的响应只有各轴单独响应之和的 89%%（g 0.89，p 1e-4），且方向部分重定向（cos 0.91）**。关键分化：亚线性响应是普遍现象（全头 g 0.70），h12 的独特性不在"最非线性"，而在三要素合取——显著亚线性 × 可观写通道（rank 7/32）× 两轴通道重叠（cos 0.55，竞争的通道级基础）；h9 亚线性更强且写通道更大（rank 2）却不承载交互（share 0.8%%），说明载体身份需要三要素合取，单一要素不充分。

**产物**：`phase2981/h12_functional_portrait/` execution %(s_exec)s / result %(s_res)s / h12_functional_portrait.npz %(s_npz)s / script %(s_scp)s。

**接续（2982 候选）**：A（主选）h12/h9 载体三要素合取分解（写核方向 × 轴响应结构 × 通道重叠逐要素对账——为何 h9 不承载）；B h12 消融主效应归宿（重平衡检验）；C Omega-C 长上下文开题；D 2978 dose law 分层复测（离线）。
""" % {'STAMP': STAMP,
       's_exec': shas['execution.json'],
       's_res': shas['result.json'],
       's_npz': shas['h12_functional_portrait.npz'],
       's_scp': shas['script']})
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(sec)
print('memo appended')

wl = io.open(WSLOG, encoding='utf-8').read()
if 'Phase 2981' not in wl:
    entry = (u"\n## Phase 2981（2026-09-20）h12 功能画像\n"
             u"- 判决 h12_input_sublinear_carrier（run2 权威，锚 7/7，a5/a6 vs 2977/2978 双恒等门 bit 级 0.00）。\n"
             u"- correction：run1 T3 跨空间 matmul 崩溃（u35@Wo_h 是 128 维头空间 vs 2560 维轴向量），T3 改同空间量，删产物重跑。\n"
             u"- 核心：交互非线性在 h12 输入响应生成（g 0.89 亚线性、cos 0.91，均 p 1e-4）；I_h12 占层交互 98.7%%；全头 g 0.70 → h12 独特性=亚线性×写通道 rank7×轴通道重叠 0.55 合取（h9 反例：更亚线性+写通道 rank2 却 share 0.8%%）。\n"
             u"- Ledger %d 条 / L14 %d / hash %s。\n"
             % (len(led['measurements']),
                len(l14['connects']), new_sha))
    with io.open(WSLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
    print('wslog appended')

mem = io.open(MEMFILE, encoding='utf-8').read()
old_next = (u'max=2980，下一个 **2981**（A 主选：h12 功能画像——write '
            u'identity + 头级 g/cos 机制分解；B h12 消融主效应归宿'
            u'（重平衡检验）；C Ω-C 长上下文开题；D 2978 dose law '
            u'分层修正复测）。方案 v3 见')
new_next = (u'max=2981，下一个 **2982**（A 主选：h12/h9 载体三要素'
            u'合取分解——为何 h9 不承载；B h12 消融主效应归宿；'
            u'C Ω-C 长上下文开题；D 2978 dose law 分层复测）。'
            u'方案 v3 见')
if old_next in mem:
    mem = mem.replace(old_next, new_next, 1)
else:
    print('WARN: memory next-candidate line not found')
old_chain = (u'——交互项头级集中、主效应分布式。核心：')
new_chain = (u'——交互项头级集中、主效应分布式→2981 交互非线性在 '
             u'h12 输入响应（g 0.89 亚线性 p 1e-4、cos 0.91；全头 '
             u'g 0.70——独特性=亚线性×写通道 rank7×轴通道重叠 '
             u'0.55 合取，h9 反例）。核心：')
if old_chain in mem:
    mem = mem.replace(old_chain, new_chain, 1)
else:
    print('WARN: chain line not found')
# 字符预算回收（压缩旧条目）
trims = [
    (u'（2970：错误 transpose 产纯噪声假显著，已知效应头被剔除是首证信号）',
     u'（2970 transpose 教训）'),
    (u'跨产物统计管线设"已知量恒等门"（2970 抓住四语言配对漂移：峰词先行过滤 13 对 vs 全词覆盖 11 对）',
     u'跨产物统计管线设"已知量恒等门"（2970 配对漂移例）'),
    (u'消融差分=直接+竞争重平衡（符号可反；2947 分类是重平衡响应分类）',
     u'消融差分=直接+竞争重平衡（符号可反）'),
    (u'bf16 前向 batch 组成敏感性：跨相位复现锚（<1e-4）要求 batch 组成 bit 级一致；按条件独立 batch57，不拼接。',
     u'bf16 batch 组成敏感性：跨相位锚要求 batch 组成 bit 级一致；条件独立 batch，不拼接。'),
]
for o, n in trims:
    if o in mem:
        mem = mem.replace(o, n, 1)
    else:
        print('WARN: trim anchor missing: %s' % o[:20])
io.open(MEMFILE, 'w', encoding='utf-8').write(mem)
print('memory updated, chars=%d' % len(mem))
print('closeout done')
