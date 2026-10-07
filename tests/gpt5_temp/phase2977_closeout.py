# -*- coding: utf-8 -*-
"""Phase 2977 closeout: seal + Ledger + MEMO + worklog + MEMORY."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
OUTD = os.path.join(
    BASE, r'tests\glm5\result\rdc_query_construction_20260913'
          r'\phase2977\two_axis_fusion_injection')
SCRIPT = os.path.join(
    BASE, r'tests\glm5\phase2977_two_axis_fusion.py')
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
    'two_axis_fusion_injection.npz':
        s8(os.path.join(OUTD, 'two_axis_fusion_injection.npz')),
    'script': s8(SCRIPT),
}
print('seal:', shas)

r = json.load(io.open(os.path.join(OUTD, 'result.json'),
                      encoding='utf-8'))
verdict = r['final_verdict']
assert verdict == 'nonlinear_fusion_interaction_detected'
assert r['anchors']['ok'] is True

led = json.load(io.open(LEDGER, encoding='utf-8'))
old_sha = led.get('ledger_sha256_8')
led.pop('ledger_sha256_8', None)
n_before = len(led['measurements'])
meas_id = 'meas2977_two_axis_fusion_injection'
led['measurements'].append({
    'meas_id': meas_id,
    'phase': 2977,
    'name': 'two_axis_fusion_injection',
    'created': e['created'],
    'verdict': verdict,
    'artifacts': {
        'execution.json': shas['execution.json'],
        'result.json': shas['result.json'],
        'two_axis_fusion_injection.npz':
            shas['two_axis_fusion_injection.npz'],
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
## Phase 2977: Omega-B 双轴融合代数——L17 竞争性亚加法交互（2² 因子因果注入） [%(STAMP)s]

**设计**（方案 v3 Omega-B 第一步，execution.json 先冻结）：Stage1 74 基线单前向（2973 协议 verbatim，词表经 2973 npz 恒等门），捕获 o_proj 输入全 36 层 + L17 层输入残差（2560 维），现场推导单位化轴方向 d_lang=mean(fr)-mean(en)、d_cls=mean(C)-mean(F)（quasi-post-hoc 推导已登记，检验对象是新增注入响应）。Stage2 74 词 × 4 条件 {sham, lang, cls, both} 因子注入于 L17 self_attn 输入 pos-1，相对剂量 s_w=0.1·||x17_w||/轴（10%% 自残差范数）。T0 floor 门（max|lang 主效应| ≥ 0.02）；T1 交互项 I=prof11-prof10-prof01+prof00 逐层中位 + 符号翻转 null（rng 2977×10000）+ maxT 族 36；T2 主效应（rng 2978/2979）+ 饱和描述；T3 描述性。

**三次 correction（均删产物重跑）**：run1 self_attn 以 kwargs 传 hidden_states，hook 读 args[0] 落空（2952 教训复现）；run2 cuda 张量 += numpy 数组 TypeError → torch.as_tensor；run3 锚 6/6 全过但全部效应触底（lang 主效应 max 0.0085，B 尺度 ~1.4）——单位范数×0.5 剂量比 2966 效应 regime 低 10-60 倍（判据可达性纪律 10 的注入版），改相对剂量 + 预注册 T0 门。

**run4 权威，锚 6/6 + T0 过**：a1 3.04e-08 / a2 0.0 / a3 恒等门 0.00+0.00（74×36 bit 级）/ a4 74/74 / a5 sham≡base bit 级 0.0 / a6 局部性（L17 前逐层 bit 同、后逐层变）。轴几何：|d_lang|=4.474、|d_cls|=13.901、cos=0.1144（近正交）。T0 max|lang 主效应|=0.0223。

**判决：nonlinear_fusion_interaction_detected**（按冻结映射）：
- **T1 交互项唯一显著层 = L17 注入层本体**：median I_L17 = **-0.0236（p 2.4e-3）**，量级超过任一单轴主效应（lang +0.0223 / cls -0.0156）；下游各层交互 ns（L32 +0.0126 p 0.31 / L18 +0.0121 p 0.32）；
- **T2 主效应**：lang [17] 与 cls [17] 同时显著，符号相反；**L17 饱和描述：observed both = -0.016 vs additive sum = +0.007——符号反转**；
- 融合代数：双轴在开关层竞争同一非线性资源（亚加法/竞争性），交互由 L17 本体生成、下游近似线性传播（逐层稀释）。

**结论（重复三遍）**：多轴融合代数 = **开关层竞争性亚加法 × 下游线性传播**——L17 对语言轴与词类轴的同时注入响应显著偏离线性叠加（交互 -0.0236，p 2.4e-3），且观测组合响应与加和预测符号相反；与 2949/2950（组消融竞争重平衡）、2966（B 被语言注入单调抹平）构成一致图景：开关层是各轴信号竞争进入点的瓶颈。附件空白一的初步答案：融合不是简单线性叠加，交互项在注入层即可测且为抑制型。

**产物**：`phase2977/two_axis_fusion_injection/` execution %(s_exec)s / result %(s_res)s / two_axis_fusion_injection.npz %(s_npz)s / script %(s_scp)s。

**接续（2978 候选）**：A（主选）交互项剂量响应与头格定位——I(s_lang, s_cls) 2D 剂量网格（{0.05,0.1,0.2}×{0.05,0.1,0.2}）+ per-head 交互格定位（交互是否集中于 L17 特定头，衔接 2966 h6/h8/h15 路由头）；B 同词跨语言语境对照（en 词入 fr 语境 vs 反之）；C Ω-C 长上下文调制开题（长度档 {2,16,64,256}）；D 配对语言效应确认（新 concept 对）。
""" % {'STAMP': STAMP,
       's_exec': shas['execution.json'],
       's_res': shas['result.json'],
       's_npz': shas['two_axis_fusion_injection.npz'],
       's_scp': shas['script']})
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(sec)
print('memo appended')

wl = io.open(WSLOG, encoding='utf-8').read()
if 'Phase 2977' not in wl:
    entry = (u"\n## Phase 2977（2026-09-20）Omega-B 双轴融合代数\n"
             u"- 判决 nonlinear_fusion_interaction_detected（run4 权威，锚 6/6+T0；三次 correction：kwargs hook、cuda numpy、剂量 floor）。\n"
             u"- 核心：L17 交互项 -0.0236（p 2.4e-3）超任一单轴主效应；observed both -0.016 vs additive +0.007 符号反转——开关层竞争性亚加法 × 下游线性传播。\n"
             u"- 教训：注入剂量必须用效应 regime 口径（单位范数剂量可低 10-60 倍触底）；T0 floor 门预注册制度化。\n"
             u"- Ledger %d 条 / L14 %d / hash %s。\n"
             % (len(led['measurements']),
                len(l14['connects']), new_sha))
    with io.open(WSLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
    print('wslog appended')

mem = io.open(MEMFILE, encoding='utf-8').read()
old_next = (u'max=2976，下一个 **2977**（A 主选 Ω-B 多轴融合代数：2^3 因子'
            u'注入+快照分解+随机 null；B 配对语言效应确认；C 同词跨语言'
            u'语境对照；D 握手传播）。方案 v3 见')
new_next = (u'max=2977，下一个 **2978**（A 主选：交互剂量网格+头格定位'
            u'I(s_l,s_c)；B 同词跨语言语境对照；C Ω-C 长上下文开题；'
            u'D 配对语言效应确认）。方案 v3 见')
if old_next in mem:
    mem = mem.replace(old_next, new_next, 1)
    io.open(MEMFILE, 'w', encoding='utf-8').write(mem)
    print('memory updated')
else:
    print('WARN: memory next-candidate line not found')
print('closeout done')
