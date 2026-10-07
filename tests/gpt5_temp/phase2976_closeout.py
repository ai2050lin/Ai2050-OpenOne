# -*- coding: utf-8 -*-
"""Phase 2976 closeout: seal + Ledger + MEMO + worklog + MEMORY."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
OUTD = os.path.join(
    BASE, r'tests\glm5\result\rdc_query_construction_20260913'
          r'\phase2976\collapse_subspace_check')
SCRIPT = os.path.join(
    BASE, r'tests\glm5\phase2976_collapse_subspace_check.py')
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
    'collapse_subspace_check.npz':
        s8(os.path.join(OUTD, 'collapse_subspace_check.npz')),
    'script': s8(SCRIPT),
}
print('seal:', shas)

r = json.load(io.open(os.path.join(OUTD, 'result.json'),
                      encoding='utf-8'))
verdict = r['final_verdict']
assert verdict == 'subspace_check_mixed_partial'
assert r['anchors']['ok'] is True

led = json.load(io.open(LEDGER, encoding='utf-8'))
old_sha = led.get('ledger_sha256_8')
led.pop('ledger_sha256_8', None)
n_before = len(led['measurements'])
meas_id = 'meas2976_collapse_subspace_check'
led['measurements'].append({
    'meas_id': meas_id,
    'phase': 2976,
    'name': 'collapse_subspace_check',
    'created': e['created'],
    'verdict': verdict,
    'artifacts': {
        'execution.json': shas['execution.json'],
        'result.json': shas['result.json'],
        'collapse_subspace_check.npz':
            shas['collapse_subspace_check.npz'],
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
## Phase 2976: fr 深带塌缩子空间检查——轴限定否定，配对几何无保持（负结果如实登记） [%(STAMP)s]

**设计**（2938 机器第五步应用，execution.json 先冻结）：2973 协议 verbatim 复跑（词表与顺序读 2973 npz 恒等门，74 单前向 [the, w]，o_proj 输入 pos-1 全 36 层 4096 维），22 个 index 对齐多语 concept 对（C_en[i] vs C_fr[i]，词表来自 2972 cells）。带族 = 2973 cos 显著层去掉早层伪影 L0 的 7 深层 [24,26,27,30,31,32,34]。T1 配对位移对齐：delta=x_fr-x_en 与读出轴 M[li] 的 |cos| 中位，fr 标签置换 null（rng 2976×10000）+ maxT 族 7；T2 **M-正交补结构保持（核心）**：x_perp = x-(x.Mhat)Mhat 后 22×22 Gram 余弦阵上三角 Spearman rho（en vs fr），fr 标签置换 null（rng 2977×10000，G_fr 按 pi 重排）+ maxT 族 7；T3 描述性：补能量比 fr/en、带符号 cos(delta,M)、全空间 Gram rho 对照。

**锚 6/6（run1 一次通过）**：a1 3.04e-08 / a2 0.0 / **a3 恒等门 vs 2973 npz：norms rel 0.00 + coss absdiff 0.00（74×36 bit 级贯通）** / a4 74/74 单 token / a5 norms>0 / a6 Gram 对角=1 且对称。

**判决：subspace_check_mixed_partial**（按冻结映射；T1 未达 5/7 显著故轴限定/全局二分支不可判，如实登记）：
- **T1 位移对齐全灭**：|cos(delta,M)| 中位 0.053-0.114，7 层全 ns（min p 3.3e-2 @L30）——配对位移并非沿读出轴定向；
- **T2 正交补保持全灭**：rho 0.010-0.313（L24 0.313 / L27 0.306 / L32 0.176），7 层全 ns vs 置换 null——M-正交补上 en-fr 配对几何**无显著保持**；全空间 rho 与正交补几乎相同（0.009-0.311），正交化不改变图景；
- **T3**：补能量比 fr/en 0.72-1.30（L30/L31 真实下降约 20-28%%，L24/L34 反升）；带符号 cos(delta,M) 全负（-0.05~-0.11）与 2973 方向一致但量级小。

**结论（负结果，重复三遍）**：fr 深带 cos 塌缩**不是** 2938 型"轴限定 + 子空间内旋转重编码"——在读出轴 M 上位移不对齐、在 M-正交补上配对几何也不保持。与 2938/2939（同词跨剂量，几何保持）对比，本案是**跨语言翻译对**：深层带中 en 词与翻译词的 word-level 几何对应本身就近乎不存在（rho~0.1-0.3 且不超置换 null）——深带是"语言重编码区"而非"保几何旋转区"，与 2972 语言轴支配静态带签名、2967 B 塌缩载体带一致。教训延伸：2938 三步审计的"子空间检查"结论依赖比较对象是**同一词**；跨词（翻译对）比较时置换 null 基线本身就近乎饱和，"保持"命题必须先确认基线对应存在。

**产物**：`phase2976/collapse_subspace_check/` execution %(s_exec)s / result %(s_res)s / collapse_subspace_check.npz %(s_npz)s / script %(s_scp)s。

**接续（2977 候选）**：A（主选）方案 v3 Ω-B 多轴融合代数（2^3 因子注入：类别+属性+语法方向同时注入，格级响应张量 + 快照分解 + 随机 null，纪律 2950/2928）；B 配对语言效应确认性检验（新 concept 对预注册，2969 机器）；C 同词跨语言语境对照（en 词入 fr 语境 vs fr 词入 en 语境，区分词身份 vs 语境效应）；D 握手通道层级传播（2975 消融后下游 A-M 对齐矩阵变化）。
""" % {'STAMP': STAMP,
       's_exec': shas['execution.json'],
       's_res': shas['result.json'],
       's_npz': shas['collapse_subspace_check.npz'],
       's_scp': shas['script']})
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(sec)
print('memo appended')

wl = io.open(WSLOG, encoding='utf-8').read()
if 'Phase 2976' not in wl:
    entry = (u"\n## Phase 2976（2026-09-20）fr 深带塌缩子空间检查\n"
             u"- 判决 subspace_check_mixed_partial（锚 6/6，run1 一次通过；a3 恒等门 vs 2973 bit 级 0）。\n"
             u"- 负结果：T1 位移对齐 7 层全 ns（|cos| 0.05-0.11）；T2 M-正交补配对几何保持 7 层全 ns（rho 0.01-0.31）——fr 塌缩非 2938 型轴限定旋转，深带 en-翻译对 word-level 几何对应本身近乎不存在。\n"
             u"- 教训：2938 子空间检查结论依赖同词比较；跨词（翻译对）时置换 null 基线近饱和，须先确认基线对应存在。\n"
             u"- Ledger %d 条 / L14 %d / hash %s。\n"
             % (len(led['measurements']),
                len(l14['connects']), new_sha))
    with io.open(WSLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
    print('wslog appended')

mem = io.open(MEMFILE, encoding='utf-8').read()
old_next = (u'max=2975，下一个 **2976**（A 主选：fr 方向塌缩子空间检查'
            u'（2938 机器，2973 npz）；B 配对语言效应确认；C Ω-B 多轴'
            u'融合；D 握手通道层级传播）')
new_next = (u'max=2976，下一个 **2977**（A 主选：方案 v3 Ω-B 多轴融合'
            u'代数——2^3 因子注入格级张量+快照分解+随机 null；B 配对语言'
            u'效应确认（新 concept 对）；C 同词跨语言语境对照；D 握手通道'
            u'层级传播）')
if old_next in mem:
    mem = mem.replace(old_next, new_next, 1)
    io.open(MEMFILE, 'w', encoding='utf-8').write(mem)
    print('memory updated')
else:
    print('WARN: memory next-candidate line not found')
print('closeout done')
