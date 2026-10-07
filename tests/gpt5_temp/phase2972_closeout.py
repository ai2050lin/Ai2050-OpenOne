# -*- coding: utf-8 -*-
"""Phase 2972 closeout: seal + Ledger + MEMO + worklog + MEMORY."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
OUTD = os.path.join(
    BASE, r'tests\glm5\result\rdc_query_construction_20260913'
          r'\phase2972\two_factor_signature')
SCRIPT = os.path.join(
    BASE, r'tests\glm5\phase2972_two_factor_signature.py')
LEDGER = os.path.join(BASE, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(BASE, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
WSLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
         r'\.workbuddy\memory\2026-09-20.md')
MEMFILE = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
           r'.workbuddy\memory\MEMORY.md')
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
    'two_factor_signature.npz':
        s8(os.path.join(OUTD, 'two_factor_signature.npz')),
    'script': s8(SCRIPT),
}
print('seal:', shas)

r = json.load(io.open(os.path.join(OUTD, 'result.json'),
                      encoding='utf-8'))
verdict = r['final_verdict']
assert verdict == 'two_factor_all_void'
assert r['anchors']['ok'] is True

led = json.load(io.open(LEDGER, encoding='utf-8'))
old_sha = led.get('ledger_sha256_8')
led.pop('ledger_sha256_8', None)
n_before = len(led['measurements'])
meas_id = 'meas2972_two_factor_signature'
led['measurements'].append({
    'meas_id': meas_id,
    'phase': 2972,
    'name': 'two_factor_signature',
    'created': e['created'],
    'verdict': verdict,
    'artifacts': {
        'execution.json': shas['execution.json'],
        'result.json': shas['result.json'],
        'two_factor_signature.npz':
            shas['two_factor_signature.npz'],
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

sec = u"""
## Phase 2972: 语言×词类双因子签名矩阵——冻结映射 all_void，语言主导与剖面复制的描述性结构 [%(STAMP)s]

**设计**：2×2 合并双语词表 74 词（F-en 15 = 2964 FUNC verbatim / F-fr 15 冻结表 / C-en 22 + C-fr 22 = 2887 表 22 个 concept 对，按 en tid 排序），协议 2937 pass1 verbatim（[the, w] 单前向，o_proj 输入 pos-1 全 36 层捕获，C[36,74,32]）；协变量 = 语言内 rank(tid) 标准化（跨语言 tid 非共同频率尺度）；T1 双因子 Freedman-Lane（reduced 1+cov，残差置换 rng 2971 联合 10000，full 1+cov+lang+class+lang:class）；T2 四对比逐层 maxT（族 36，rng 2972-2975）；T3 头级 maxT（族 32）；T4 描述性（2964 剖面对齐 / Simpson 检查 / concept 内配对语言差）。

**锚 7/7**（run2 权威）：a1 3.04e-08 / a2 0.0 / a3 2.43e-16 / a4 77/77 单 token / a5 3.49e-16 / a6 非退化 / a7 22 concepts≥20。

**T1（冻结门 p≤0.01）**：lang p 4.32e-2 / class p 1.18e-2 / interaction p 1.15e-2——**三项全部差门未达硬显著，判决 two_factor_all_void（按冻结映射登记）**。格中位 B：F_en -1.3763 / F_fr -0.4750 / C_en -2.5452 / C_fr -0.6821。

**T2**：class_en / class_fr / lang_F 三对比零显著层；**lang_C（内容词内语言差）唯一显著层 L34**——静态带差分上的语言效应同样落在 L34，与 2964 类效应载体、2970 延迟载体同带。T3 无显著头。

**T4 描述性（quasi-post-hoc，只登记不判决）**：
- **剖面复制极强：rho(classgap_en, 2964 gap_layers) = 0.9704**——类效应的空间签名（层剖面）在双语词表上几乎完美复制，尽管量级门未过——空间签名与量级门是两个命题（2946 量级/结构分账再应用）。
- **类效应是语言绑定的**：within-en 类差 1.1689（复制 2964 的 1.37 量级）vs within-fr 0.2070（塌缩）——池化 1.1393 掩盖了这个异质性（Simpson 型结构第 4 次出现）。
- **concept 内配对语言差巨大：22 对中 21 对 fr > en，median d = +1.8156**——语言轴对静态 B 的支配远超词类轴；此为 T4 描述性（配对检验未预注册为门），确认性检验须新词对预注册。

**硬伤与勘误**：① run1 T2 日志行 KeyError（t2 增加 interaction_descriptive 键后日志取 sig_layers 失败）——锚 7/7 与前向已完成后报错，按纪律 3 删产物重跑；② **关键设计局限（登记为下一 Phase 动机）**：协议上下文是英文 ' the'，fr 词处于 code-switching 语境（' the maison'）而非自然法语语境——fr 格 B 值整体趋零（-0.48/-0.68 vs en -1.38/-2.55）可能是**量级（norm）伪影**而非方向效应（2936-2938 审计链：量级塌缩命名前必须 norm/cos 分解）；③ interaction p 1.15e-2 与 class p 1.18e-2 均 0.01 门外边界带，按纪律 11 单独登记不与硬显著混池。

**结论**：合并双语词表上，冻结双因子判据 all_void——**语言轴支配静态带签名（配对 21/22、L34 唯一显著层），词类效应呈语言绑定（en 有 fr 无），但类效应层剖面跨词表复制（rho 0.97）**。机制链新增一环：静态 B 的语言-词类结构 = 语言主调制 × 词类从属（英文特化），"类效应"的普适性主张必须限定语言语境。下一步必须先做 fr 格 B 趋零的 norm/cos 尺度审计（2937 机器），再判定语言效应是"方向重写"还是"能量衰减"。

**产物**：`phase2972/two_factor_signature/` execution {a1} / result {b1} / two_factor_signature.npz {c1} / script {d1}。

**接续（2973 候选）**：A（主选）fr 格 B 尺度审计——norm/cos 分解 + 子空间检查（2937-2939 机器，2972 npz 离线可做）；B 方案 v3 Ω-A 跨模块子空间对齐（零前向 + 随机 null 三件套）；C 配对语言效应确认性检验（新 concept 对 + 预注册配对置换）；D 方案 v3 Ω-B 多轴融合代数（2^3 因子注入）。
""" % {'STAMP': STAMP, 'a1': shas['execution.json'],
       'b1': shas['result.json'],
       'c1': shas['two_factor_signature.npz'],
       'd1': shas['script']}
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(sec)
print('memo appended')

wl = io.open(WSLOG, encoding='utf-8').read()
if 'Phase 2972' not in wl:
    entry = (u"\n## Phase 2972（2026-09-20）语言×词类双因子签名矩阵\n"
             u"- 判决 two_factor_all_void（锚 7/7，run2 权威）：lang p 4.32e-2 / class p 1.18e-2 / inter p 1.15e-2 全部 0.01 门外。\n"
             u"- 描述性：类效应层剖面跨词表复制 rho 0.9704；类效应语言绑定（en 1.17 / fr 0.21）；concept 内配对 fr>en 21/22、median d +1.82；lang_C 唯一显著层 L34。\n"
             u"- 设计局限登记：code-switching 语境 + fr 格 B 趋零疑为 norm 伪影 → 2973 尺度审计。\n"
             u"- Ledger 111 条 / L14 79 / hash %s。\n" % new_sha)
    with io.open(WSLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
    print('wslog appended')

mem = io.open(MEMFILE, encoding='utf-8').read()
if 'max=2971' in mem:
    mem = mem.replace('max=2971，下一个 **2972**（A 主选：语言×词类双因子签名矩阵 n≥60 预注册；B 延迟头群功能身份消融；C h8/h21 峰位词属性离线；D 卡片集跨模型对齐开题）。',
                      'max=2972，下一个 **2973**（A 主选：fr 格 B 尺度审计 norm/cos 分解，2972 npz 离线；B 方案 v3 Ω-A 跨模块对齐；C 配对语言效应确认；D Ω-B 多轴融合）。方案 v3（Omega 伞形）见 research\\gpt5\\docs\\plan_v3_omega_dynamic_manifold.md。')
    io.open(MEMFILE, 'w', encoding='utf-8').write(mem)
    print('memory updated')
print('closeout done')
