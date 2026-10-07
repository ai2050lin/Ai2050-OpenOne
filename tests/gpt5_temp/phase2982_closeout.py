# -*- coding: utf-8 -*-
"""Phase 2982 closeout: seal + Ledger + MEMO + worklog + MEMORY."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
OUTD = os.path.join(
    BASE, r'tests\glm5\result\rdc_query_construction_20260913'
          r'\phase2982\carrier_conjunctive_anatomy')
SCRIPT = os.path.join(
    BASE, r'tests\glm5\phase2982_carrier_conjunctive_anatomy.py')
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
    'carrier_conjunctive_anatomy.npz':
        s8(os.path.join(OUTD,
                        'carrier_conjunctive_anatomy.npz')),
    'script': s8(SCRIPT),
}
print('seal:', shas)

r = json.load(io.open(os.path.join(OUTD, 'result.json'),
                      encoding='utf-8'))
verdict = r['final_verdict']
assert verdict == 'three_element_all_void', verdict
assert r['anchors']['ok'] is True

led = json.load(io.open(LEDGER, encoding='utf-8'))
old_sha = led.get('ledger_sha256_8')
led.pop('ledger_sha256_8', None)
n_before = len(led['measurements'])
meas_id = 'meas2982_carrier_conjunctive_anatomy'
led['measurements'].append({
    'meas_id': meas_id,
    'phase': 2982,
    'name': 'carrier_conjunctive_anatomy',
    'created': e['created'],
    'verdict': verdict,
    'artifacts': {
        'execution.json': shas['execution.json'],
        'result.json': shas['result.json'],
        'carrier_conjunctive_anatomy.npz':
            shas['carrier_conjunctive_anatomy.npz'],
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
print('ledger: ' + str(n_before) + ' -> '
      + str(len(led['measurements']))
      + ', L14 connects ' + str(len(l14['connects']))
      + ', sha ' + old_sha + ' -> ' + new_sha)

sec = (
    u"\n## Phase 2982: h12/h9 载体三要素合取分解——群体定律否定"
    u"与 h9 通道反平行发现 [" + STAMP + u"]\n\n"
    u"**设计**（2981 协议 verbatim，execution.json 先冻结）："
    u"74 intact + 74x4 条件 {00,10,01,11} dose 0.1（2979 _u "
    u"dirs，s_w=0.1·n17_ref），370 前向；新量 = 全头通道重叠 "
    u"ov_h = median_w cos(dx10_h, dx01_h)。三要素（正预测方向"
    u"定向）：e_ov=ov_h、e_sub=-g_med、e_wk=写核范数。\n\n"
    u"**锚 10/10**（跨产物恒等四重 bit 级：a5 I_h vs 2981 逐词 "
    u"0.00、a6 g_all 0.00、a7 cos_all 0.00、a8 wk_norm 0.00；"
    u"a9 层级 I vs 2977 0.00；a3 1.21e-07；a10 要素覆盖 "
    u"32/32/32）。\n\n"
    u"**判决：three_element_all_void**（按冻结映射）：\n"
    u"- T1 主检验：Spearman(element, |I_h_med|) 32 头 + 标签"
    u"置换 null——e_ov rho 0.2797（p 0.122 ns）；e_sub rho "
    u"-0.5114（p 4.7e-3 显著但方向相反：越亚线性 |I| 越小，"
    u"与朴素预测相反）；e_wk rho -0.3460（p 0.056 ns）。无一"
    u"要素以预测方向显著 → 冻结映射无分支命中。\n"
    u"- T2 个案对账：h9 = 通道反平行（ov -0.8635，rank 32/32）"
    u"x 极端亚线性（g 0.3943）x 写核最大档（0.3694）→ I_h 仅 "
    u"-0.000182（近乎完全抵消）；h12 = 中间 regime（ov +0.5485 "
    u"rank 8、g 0.890、写核 0.308）→ I_h -0.023274。\n"
    u"- T3（quasi-post-hoc）：S_lo 限域 rho_ov -0.4286（描述"
    u"性）。\n\n"
    u"**结论（重复三遍）**：**2981 的三要素合取不是群体定律——"
    u"无任何单一要素正向预测头级交互承载（all_void），且亚线性"
    u"与承载显著负相关（rho -0.51）**。h9 通道反平行（两轴响应"
    u"在输入空间互相对抗）+ 极端压缩 = 交互自我抵消的新机制型反"
    u"例；h12 的载体身份是中间 regime 现象（重叠适中、增益近线"
    u"性、写通道中等），不是任何要素的极值。载体选择问题保持开"
    u"放：单要素解释与合取解释均被否定。\n\n"
    u"**产物**：`phase2982/carrier_conjunctive_anatomy/` "
    u"execution " + shas['execution.json'] + u" / result "
    + shas['result.json'] + u" / "
    u"carrier_conjunctive_anatomy.npz "
    + shas['carrier_conjunctive_anatomy.npz'] + u" / script "
    + shas['script'] + u"。\n\n"
    u"**接续（2983 候选）**：A（主选）交互项解析分解——I_h 对 "
    u"响应几何 (g/cos/ov/wk/响应范数) 的多变量对账 + "
    u"c11-c10-c01+c00 的响应级来源定位（h12 中间 regime 为何出"
    u"交互）；B h12 消融主效应归宿（重平衡检验）；C Omega-C 长"
    u"上下文开题；D 2978 dose law 分层复测（离线）。\n"
)
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(sec)
print('memo appended')

wl = io.open(WSLOG, encoding='utf-8').read()
if 'Phase 2982' not in wl:
    entry = (u"\n## Phase 2982（2026-09-20）三要素合取分解\n"
             u"- 判决 three_element_all_void（run1 权威，锚 "
             u"10/10，vs 2981 四重 bit 级 0.00 + vs 2977 层级 "
             u"0.00）。\n"
             u"- 核心：三要素合取不是群体定律；亚线性与 |I| 负"
             u"相关（rho -0.51 p 4.7e-3，方向反直觉）；h9 通道"
             u"反平行（ov -0.86 rank 32/32）+ 极端亚线性 = 交"
             u"互自抵消反例；h12 = 中间 regime（ov 0.55 rank "
             u"8）。\n"
             u"- Ledger " + str(len(led['measurements']))
             + u" 条 / L14 " + str(len(l14['connects']))
             + u" / hash " + new_sha + u"。\n")
    with io.open(WSLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
    print('wslog appended')

mem = io.open(MEMFILE, encoding='utf-8').read()
old_next = (u'max=2981，下一个 **2982**（A 主选：h12/h9 载体三要素'
            u'合取分解——为何 h9 不承载；B h12 消融主效应归宿；'
            u'C Ω-C 长上下文开题；D 2978 dose law 分层复测）。'
            u'方案 v3 见')
new_next = (u'max=2982，下一个 **2983**（A 主选：交互项解析分解——'
            u'|I_h| 对响应几何多变量对账+来源定位；B h12 消融主效应'
            u'归宿；C Ω-C 长上下文开题；D dose law 分层复测）。'
            u'方案 v3 见')
if old_next in mem:
    mem = mem.replace(old_next, new_next, 1)
else:
    print('WARN: memory next-candidate line not found')
old_chain = (u'——独特性=亚线性×写通道 rank7×轴通道重叠 '
             u'0.55 合取，h9 反例）。核心：')
new_chain = (u'——独特性=三要素中间 regime 合取，h9 反例）'
             u'→2982 三要素群体否定（无要素正预测；亚线性负相关 '
             u'-0.51 p 4.7e-3；h9 反平行 ov -0.86 自抵消）。'
             u'核心：')
if old_chain in mem:
    mem = mem.replace(old_chain, new_chain, 1)
else:
    print('WARN: chain line not found')
trims = [
    (u'2979 权威：转正分布式无正头，h9/h12 同头持续，'
     u'g=0.71/cos=0.96 饱和挤压倾向',
     u'2979 权威：转正分布式无正头，h9/h12 同头持续'),
    (u'2980 消融定案：h12 是交互必要载体（消融消失 97%，'
     u'随机头 <2%，无全局破坏）',
     u'2980 消融定案：h12 必要载体（消失 97%，随机头无效）'),
    (u'跨产物引用先核对实际 JSON 键/路径与键格式/存储精度',
     u'跨产物引用先核对实际键/路径/精度'),
    (u'前置审计（2936/2937）→2938 子空间',
     u'前置审计→2938 子空间'),
    (u'2977 双轴注入竞争亚加法（L17 本体）',
     u'2977 双轴注入竞争亚加法@L17'),
]
for o, n in trims:
    if o in mem:
        mem = mem.replace(o, n, 1)
    else:
        print('WARN: trim anchor missing: ' + o[:20])
io.open(MEMFILE, 'w', encoding='utf-8').write(mem)
print('memory updated, chars=' + str(len(mem)))
print('closeout done')
