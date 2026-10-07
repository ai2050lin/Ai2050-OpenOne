# -*- coding: utf-8 -*-
"""Phase 2974 closeout: seal + Ledger + MEMO + worklog + MEMORY."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
OUTD = os.path.join(
    BASE, r'tests\glm5\result\rdc_query_construction_20260913'
          r'\phase2974\cross_module_alignment')
SCRIPT = os.path.join(
    BASE, r'tests\glm5\phase2974_cross_module_alignment.py')
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
    'cross_module_alignment.npz':
        s8(os.path.join(OUTD, 'cross_module_alignment.npz')),
    'script': s8(SCRIPT),
}
print('seal:', shas)

r = json.load(io.open(os.path.join(OUTD, 'result.json'),
                      encoding='utf-8'))
verdict = r['final_verdict']
assert verdict == 'cross_module_alignment_partial'
assert r['anchors']['ok'] is True

led = json.load(io.open(LEDGER, encoding='utf-8'))
old_sha = led.get('ledger_sha256_8')
led.pop('ledger_sha256_8', None)
n_before = len(led['measurements'])
meas_id = 'meas2974_cross_module_alignment'
led['measurements'].append({
    'meas_id': meas_id,
    'phase': 2974,
    'name': 'cross_module_alignment',
    'created': e['created'],
    'verdict': verdict,
    'artifacts': {
        'execution.json': shas['execution.json'],
        'result.json': shas['result.json'],
        'cross_module_alignment.npz':
            shas['cross_module_alignment.npz'],
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
## Phase 2974: 方案 v3 Omega-A 跨模块子空间对齐——握手只在早层（L1-3），深层载体无对齐 [%(STAMP)s]

**设计**（零前向，附件空白三的可证伪化）：A 空间 = 注意力写空间（o_proj 权重头切片堆叠的左奇异 top-64，2560 维输出方向）；M 空间 = MLP 读空间（up_proj 右奇异 top-64）；度量 = 主角度 cos 均值（top-16/64）。T1 同层族 36 层（null：N1 随机正交 200 次 maxT + N2 非 对角池 max，双门取大）；T2 载体头画像（2964 L34 top5 头 [15,8,21,28,11]，quasi-post-hoc 登记）vs 匹配随机头 null（200 次 maxT）；T3 描述性 36×36 矩阵 + gate_proj 稳健性。

**锚 6/6（run3 权威，f64）**：a1 结构门（Wo=(2560,4096)、up/gate in=2560）/ a2 正交 1.71e-15 / a3 自对齐 1.000000000000 / a4 确定性 0.00 / a5 载体集恒等 / a6 非退化（min sig ratio 0.247）。

**判决：cross_module_alignment_partial**：
- **T1：显著层 = [L1, L2, L3]**——同层 A-M 对齐 S 0.767/0.829/0.759，同时超过随机正交 null（maxT p95 0.257）与非对角池 max（0.701）。L0-8 剖面 [0.689, 0.767, 0.829, 0.759, 0.632, 0.536, 0.479, 0.594, 0.638]；**L14-35 max 仅 0.543（L18），全部低于门**。
- **T2：载体头门未达**——S(A34c, M) 画像 argmax L35（0.290），但匹配随机头 null maxT 阈 0.418，载体头画像无显著层且低于 null：**载体头的写空间与 MLP 读空间的对齐不比随机 5 头更强**。
- **T3**：36×36 矩阵非对角 top8 全部是早层相邻对（1↔2 0.701、2↔3 0.606、3↔4 0.586、7↔8 0.568、8↔9 0.551）——早层构成平滑的共享子空间连续流形；up vs gate 稳健性 rho 0.255。

**硬伤与勘误**：run1 缺 import io（冻结前，无产物）；run2 锚 a1/a2/a3 失败——a1 形状检查写反（o_proj weight 是 (2560,4096)=(out,in)）+ float32 SVD 正交误差 2.4e-07 超 1e-8 门，修正为 f64 化 + 形状门后按纪律 3 删产物重跑（run3 权威）。

**结论**：附件空白三的"跨模块几何握手协议"主张**只在早层（L1-3）成立**：早层注意力写空间与 MLP 读空间存在超随机 null 的强子空间对齐，且相邻层平滑过渡（共享连续流形）。**在机制承重的深层（L17/L34 区域），对齐不成立**——载体头的写空间对 MLP 读空间的对齐甚至不比随机头强。结合 2965（消融后共享载体）、2967/2970（分布式塌缩/延迟带）：深层 Attention-MLP 交接不是子空间几何对齐机制，而是高维分布式/功能性交接——"路由信号精确落入目标子空间"的具象图景被否定。机制链新增一环：模块间交接拓扑 = 早层几何握手 × 深层分布式交接。

**产物**：`phase2974/cross_module_alignment/` execution %(s_exec)s / result %(s_res)s / cross_module_alignment.npz %(s_npz)s / script %(s_scp)s。

**接续（2975 候选）**：A（主选）早层握手的功能意义——L1-3 对齐子空间是否承载功能读出（对齐方向消融/剂量，2965 机器）；B fr 方向塌缩子空间检查（2938 机器，2973 npz 离线）；C 配对语言效应确认性检验（新 concept 对预注册）；D 方案 v3 Omega-B 多轴融合代数（2^3 因子注入）。
""" % {'STAMP': STAMP,
       's_exec': shas['execution.json'],
       's_res': shas['result.json'],
       's_npz': shas['cross_module_alignment.npz'],
       's_scp': shas['script']})
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(sec)
print('memo appended')

wl = io.open(WSLOG, encoding='utf-8').read()
if 'Phase 2974' not in wl:
    entry = (u"\n## Phase 2974（2026-09-20）方案 v3 Omega-A 跨模块子空间对齐\n"
             u"- 判决 cross_module_alignment_partial（锚 6/6，run3 f64 权威；run1 io 缺失、run2 锚失败 f32+形状门反转已勘误）。\n"
             u"- 握手只在早层：T1 显著层 [L1,L2,L3]（S 0.76-0.83 超 N1 maxT 0.257 与 offdiag max 0.701）；深层 L14-35 max 0.543 低于门。\n"
             u"- 载体头（2964 L34 top5）写空间 vs MLP 读空间对齐低于匹配随机头 null（0.29 vs 阈 0.42）——深层交接非几何对齐，支持分布式交接图景。\n"
             u"- Ledger %d 条 / L14 %d / hash %s。\n"
             % (len(led['measurements']),
                len(l14['connects']), new_sha))
    with io.open(WSLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
    print('wslog appended')

mem = io.open(MEMFILE, encoding='utf-8').read()
old_next = (u'max=2973，下一个 **2974**（A 主选：方案 v3 Ω-A 跨模块'
            u'子空间对齐——主角度+随机头 null 三件套；B fr 方向塌缩'
            u'子空间检查（2938 机器）；C 配对语言效应确认；D Ω-B '
            u'多轴融合）')
new_next = (u'max=2974，下一个 **2975**（A 主选：早层握手功能意义——'
            u'L1-3 对齐子空间消融/剂量；B fr 方向塌缩子空间检查（2938 '
            u'机器）；C 配对语言效应确认；D Ω-B 多轴融合）')
if old_next in mem:
    mem = mem.replace(old_next, new_next, 1)
    io.open(MEMFILE, 'w', encoding='utf-8').write(mem)
    print('memory updated')
else:
    print('WARN: memory next-candidate line not found')
print('closeout done')
