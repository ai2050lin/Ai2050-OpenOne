# -*- coding: utf-8 -*-
"""Phase 2975 closeout: seal + Ledger + MEMO + worklog + MEMORY."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
OUTD = os.path.join(
    BASE, r'tests\glm5\result\rdc_query_construction_20260913'
          r'\phase2975\handshake_functional_ablation')
SCRIPT = os.path.join(
    BASE, r'tests\glm5\phase2975_handshake_functional_ablation.py')
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
    'handshake_functional_ablation.npz':
        s8(os.path.join(OUTD,
                        'handshake_functional_ablation.npz')),
    'script': s8(SCRIPT),
}
print('seal:', shas)

r = json.load(io.open(os.path.join(OUTD, 'result.json'),
                      encoding='utf-8'))
verdict = r['final_verdict']
assert verdict == 'handshake_subspace_functionally_load_bearing'
assert r['anchors']['ok'] is True

led = json.load(io.open(LEDGER, encoding='utf-8'))
old_sha = led.get('ledger_sha256_8')
led.pop('ledger_sha256_8', None)
n_before = len(led['measurements'])
meas_id = 'meas2975_handshake_functional_ablation'
led['measurements'].append({
    'meas_id': meas_id,
    'phase': 2975,
    'name': 'handshake_functional_ablation',
    'created': e['created'],
    'verdict': verdict,
    'artifacts': {
        'execution.json': shas['execution.json'],
        'result.json': shas['result.json'],
        'handshake_functional_ablation.npz':
            shas['handshake_functional_ablation.npz'],
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
## Phase 2975: 早层握手功能意义——对齐通道消融决定性承重（3x 随机对照） [%(STAMP)s]

**设计**：2974 发现握手只在 L1-3 后，检验该对齐子空间是否功能承重。37 英文词（2972 F_en+C_en 恒等门）× 4 条件单前向（2964 协议 verbatim）：base / abl_L123_align（o_proj 输出前向 hook 沿 QA=Wo top-64 左奇异基投影消融，即关闭"握手写通道"，全位置）/ abl_L123_rand（匹配随机正交基，rng 2975-2977）/ abl_L20_align（层控制）。读出 sep = x_36·u35 与 B 带签名。T1/T2 配对符号翻转置换（rng 2978/2979，10000，单侧 median>0，门 p≤0.01）。

**锚 7/7（run1 一次通过）**：a1 3.04e-08 / a2 0.0 / a3 握手恒等（重算 S(li,li) vs 2974 存储 4dp，max diff 3.61e-05）/ a4 40/40 单 token / a5 6.61e-16 / a6 正交 2.44e-15 + 投影残差 1.67e-07 + 消融-基线差异 1.67e-02 / a7 词集恒等。

**判决：handshake_subspace_functionally_load_bearing**：
- **T1 读出**：median |Δsep| 对齐通道 2.08594 vs 随机通道 0.68609（**3.0x**），delta 1.26222，frac>0 = 0.92，p 9.999e-05。
- **T2 带签名**：median |ΔB| 对齐 0.07636 vs 随机 0.01749（**4.4x**），frac 0.84，p 9.999e-05。
- **T3 描述性**：L20 层控制 median |Δsep| 1.448（深层 top-64 通道本身承重更大——位置在深层），L123/L20 比 1.441；层内对齐 vs 随机的 3x 对比才是握手特异证据。

**工程要点**：o_proj forward-hook（with_kwargs）按层闭包绑定各自的消融基——run 前设计审查抓到初稿"current 键共享基"错误（会把 QA[3] 误用于 L1/L2）与 sep 行残留三元表达式，补丁后 ast+Grep 复核再运行。

**结论**：2974 的早层握手不只是几何巧合——**关闭 L1-3 对齐写通道对下游 u35 读出的扰动是匹配随机通道的 3.0 倍、对带签名是 4.4 倍**（均 p 1e-4，frac 0.92/0.84），握手子空间功能承重判定成立。机制链新增一环：早层 Attention→MLP 交接 = 几何握手通道 × 功能承重（与深层的分布式交接形成拓扑对照）。方案 v3 Omega-A 工作包（2974-2975）两环收官：跨模块交接拓扑 = 早层几何握手（功能承重）+ 深层分布式交接（无几何对齐）。

**产物**：`phase2975/handshake_functional_ablation/` execution %(s_exec)s / result %(s_res)s / handshake_functional_ablation.npz %(s_npz)s / script %(s_scp)s。

**接续（2976 候选）**：A（主选）fr 方向塌缩子空间检查（2938 机器：2973 的 fr 深带 cos 塌缩是否限单方向、子空间对齐是否保留）；B 配对语言效应确认性检验（新 concept 对预注册）；C 方案 v3 Omega-B 多轴融合代数（2^3 因子注入）；D 握手通道层级传播（L1-3 消融后下游层 A-M 对齐矩阵变化，2974 机器复用）。
""" % {'STAMP': STAMP,
       's_exec': shas['execution.json'],
       's_res': shas['result.json'],
       's_npz': shas['handshake_functional_ablation.npz'],
       's_scp': shas['script']})
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(sec)
print('memo appended')

wl = io.open(WSLOG, encoding='utf-8').read()
if 'Phase 2975' not in wl:
    entry = (u"\n## Phase 2975（2026-09-20）早层握手功能意义消融\n"
             u"- 判决 handshake_subspace_functionally_load_bearing（锚 7/7，run1 一次通过）。\n"
             u"- L1-3 对齐通道消融 vs 匹配随机：读出 3.0x（2.086 vs 0.686，p 1e-4）、带签名 4.4x（0.076 vs 0.017，p 1e-4）——握手子空间功能承重成立。\n"
             u"- L20 层控制 1.448（深层通道本身更大），层内 3x 对比是握手特异证据。\n"
             u"- Omega-A 工作包收官：交接拓扑 = 早层几何握手（承重）+ 深层分布式交接。\n"
             u"- Ledger %d 条 / L14 %d / hash %s。\n"
             % (len(led['measurements']),
                len(l14['connects']), new_sha))
    with io.open(WSLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
    print('wslog appended')

mem = io.open(MEMFILE, encoding='utf-8').read()
old_next = (u'max=2974，下一个 **2975**（A 主选：早层握手功能意义——'
            u'L1-3 对齐子空间消融/剂量；B fr 方向塌缩子空间检查（2938 '
            u'机器）；C 配对语言效应确认；D Ω-B 多轴融合）')
new_next = (u'max=2975，下一个 **2976**（A 主选：fr 方向塌缩子空间'
            u'检查（2938 机器，2973 npz）；B 配对语言效应确认；C Ω-B '
            u'多轴融合；D 握手通道层级传播）')
if old_next in mem:
    mem = mem.replace(old_next, new_next, 1)
    io.open(MEMFILE, 'w', encoding='utf-8').write(mem)
    print('memory updated')
else:
    print('WARN: memory next-candidate line not found')
print('closeout done')
