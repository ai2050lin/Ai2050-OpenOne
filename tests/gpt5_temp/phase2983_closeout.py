# -*- coding: utf-8 -*-
"""Phase 2983 closeout: seal + Ledger + MEMO + worklog + MEMORY."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
OUTD = os.path.join(
    BASE, r'tests\glm5\result\rdc_query_construction_20260913'
          r'\phase2983\interaction_analytic_decomposition')
SCRIPT = os.path.join(
    BASE, r'tests\glm5'
          r'\phase2983_interaction_analytic_decomposition.py')
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
    'npz': s8(os.path.join(
        OUTD, 'interaction_analytic_decomposition.npz')),
    'script': s8(SCRIPT),
}
print('seal:', shas)

r = json.load(io.open(os.path.join(OUTD, 'result.json'),
                      encoding='utf-8'))
verdict = r['final_verdict']
assert verdict == 'word_level_deficit_tracks_interaction', \
    verdict
assert r['anchors']['ok'] is True

led = json.load(io.open(LEDGER, encoding='utf-8'))
old_sha = led.get('ledger_sha256_8')
led.pop('ledger_sha256_8', None)
n_before = len(led['measurements'])
meas_id = 'meas2983_interaction_analytic_decomposition'
led['measurements'].append({
    'meas_id': meas_id,
    'phase': 2983,
    'name': 'interaction_analytic_decomposition',
    'created': e['created'],
    'verdict': verdict,
    'artifacts': {
        'execution.json': shas['execution.json'],
        'result.json': shas['result.json'],
        'interaction_analytic_decomposition.npz':
            shas['npz'],
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
    u"\n## Phase 2983: 交互项解析分解——正交重定向主导与词级范数"
    u"赤字锁定 [" + STAMP + u"]\n\n"
    u"**设计**（2982 协议 verbatim，execution.json 先冻结）："
    u"370 前向。利用读出线性恒等式 I_h(w) = Mh·delta_h(w)"
    u"（delta = dx11-(dx10+dx01) 为非线性残差位移，128 维头输"
    u"入空间），把 delta 逐词分解为沿和轴压缩分量 par 与正交"
    u"重定向分量 perp，各自经头读向量 Mh 投影。T1 通道支配度 "
    u"share = sum_h |median par| / 总和，词配对置换 null "
    u"（10000，rng 29830）；T2 h12 词级 Spearman（deficit/"
    u"cosdir，rng 29831/29832）；T3 逐头 ||delta_bar|| vs "
    u"|Mh·delta_bar|（quasi-post-hoc）。\n\n"
    u"**锚 12/12**（六重 bit 级：a5 I_h / a6 g_all / a7 "
    u"cos_all / a8 wk_norm / a11 ov_all vs 2982 全部 0.00，"
    u"a9 层级 I vs 2977 逐词 0.00；a10 代数恒等门 max "
    u"1.13e-16；a3 1.21e-07；a12 覆盖 32/32）。\n\n"
    u"**判决：word_level_deficit_tracks_interaction**（按冻"
    u"结映射）：\n"
    u"- T1 主检验：share_par = 0.1466——正交重定向通道承载 "
    u"~85% 的头级交互；one-sided right 门 p_right = 1.0"
    u"（obs 在 null 分布最低端，等价 p_left ≈ 1e-4）——perp "
    u"主导在 null 校准下结构性显著，但预注册门是 right 方向"
    u"（检验 par 主导），按冻结映射不触发 orthogonal_redirect "
    u"分支，perp 主导作描述性登记（口径已注记）。\n"
    u"- T2 词级（h12）：rho(I, deficit) = 0.9425（p 1e-4，"
    u"正式判决分支）；rho(I, cosdir) = 0.8479（p 1e-4）——范"
    u"数赤字是更强预测子。\n"
    u"- T3：h9 的 median delta 本身近乎为零（dnorm 0.0045，"
    u"proj 0.000183）——自抵消发生在位移向量层面而非仅读出投"
    u"影；h12 dnorm 0.3157 / proj 0.0162 全头最大（ratio "
    u"rank：h12 6/32、h9 21/32）；全头 median ratio 0.0151"
    u"——位移大多不沿读出轴。\n\n"
    u"**结论（重复三遍）**：**L17 双轴竞争交互的物理来源是正"
    u"交重定向通道（perp ~85%），不是沿和轴的饱和压缩；在载体"
    u"头 h12 内，词级交互与范数赤字 ||dx11||-(||dx10||+"
    u"||dx01||) 强锁定（rho 0.94）**。与 2979 全头 g=0.71 的"
    u"饱和挤压图景精细分化：饱和压缩主导全头平均响应幅度，但"
    u"交互项本身由方向重定向承载——亚加法是方向现象而非单纯幅"
    u"度现象。h9 自抵消的机制定位修正：抵消发生在 delta 生成"
    u"层面（两轴响应在输入空间对抗至 median delta≈0），不必诉"
    u"诸读出投影抵消。\n\n"
    u"**产物**：`phase2983/interaction_analytic_decomposition/` "
    u"execution " + shas['execution.json'] + u" / result "
    + shas['result.json'] + u" / "
    u"interaction_analytic_decomposition.npz " + shas['npz']
    + u" / script " + shas['script'] + u"。\n\n"
    u"**接续（2984 候选）**：A（主选）h12 消融主效应归宿（交互"
    u"消失后 B 带签名迁移——重平衡检验，2980 机器 + 2965 读出）；"
    u"B perp 通道的几何身份（SVD 定位重定向子空间，是否词类/语"
    u"言轴共享）；C Omega-C 长上下文调制开题；D 2978 dose law "
    u"分层复测（离线）。\n"
)
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(sec)
print('memo appended')

wl = io.open(WSLOG, encoding='utf-8').read()
if 'Phase 2983' not in wl:
    entry = (u"\n## Phase 2983（2026-09-20）交互解析分解\n"
             u"- 判决 word_level_deficit_tracks_interaction"
             u"（run1 权威，锚 12/12，vs 2982 五重 bit 级 0.00 "
             u"+ vs 2977 0.00 + 恒等门 1.13e-16）。\n"
             u"- 核心：交互由正交重定向通道承载 ~85%（share_par "
             u"0.1466，p_left≈1e-4 描述性）；h12 词级 I 与范数"
             u"赤字 rho 0.94 p 1e-4；h9 自抵消在 delta 生成层"
             u"面（median delta≈0）。\n"
             u"- Ledger " + str(len(led['measurements']))
             + u" 条 / L14 " + str(len(l14['connects']))
             + u" / hash " + new_sha + u"。\n")
    with io.open(WSLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
    print('wslog appended')

mem = io.open(MEMFILE, encoding='utf-8').read()
old_next = (u'max=2982，下一个 **2983**（A 主选：交互项解析分解——'
            u'|I_h| 对响应几何多变量对账+来源定位；B h12 消融主效应'
            u'归宿；C Ω-C 长上下文开题；D dose law 分层复测）。'
            u'方案 v3 见')
new_next = (u'max=2983，下一个 **2984**（A 主选：h12 消融主效应'
            u'归宿——交互消失后 B 带签名迁移/重平衡；B perp 通道'
            u'几何身份 SVD；C Ω-C 长上下文开题；D dose law 分层'
            u'复测）。方案 v3 见')
if old_next in mem:
    mem = mem.replace(old_next, new_next, 1)
else:
    print('WARN: memory next-candidate line not found')
old_chain = (u'2982 三要素群体否定（无要素正预测；亚线性负相关 '
             u'-0.51 p 4.7e-3；h9 反平行 ov -0.86 自抵消）。'
             u'核心：')
new_chain = (u'2982 三要素群体否定（亚线性负相关 -0.51；h9 反平'
             u'行自抵消）→2983 解析分解：交互由正交重定向通道承载'
             u'~85%（非和轴压缩），h12 词级 I 与范数赤字 rho 0.94，'
             u'h9 抵消在 delta 生成层面。核心：')
if old_chain in mem:
    mem = mem.replace(old_chain, new_chain, 1)
else:
    print('WARN: chain line not found')
io.open(MEMFILE, 'w', encoding='utf-8').write(mem)
print('memory updated, chars=' + str(len(mem)))
print('closeout done')
