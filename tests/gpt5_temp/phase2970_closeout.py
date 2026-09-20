# -*- coding: utf-8 -*-
"""Phase 2970 closeout: seal + Ledger + MEMO + worklog + MEMORY."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
OUTD = os.path.join(
    BASE, r'tests\glm5\result\rdc_query_construction_20260913'
          r'\phase2970\delay_carrier_localization')
SCRIPT = os.path.join(
    BASE, r'tests\glm5\phase2970_delay_carrier_localization.py')
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
    'delay_carrier.npz': s8(os.path.join(OUTD,
                                         'delay_carrier.npz')),
    'script': s8(SCRIPT),
}
print('seal:', shas)

r = json.load(io.open(os.path.join(OUTD, 'result.json'),
                      encoding='utf-8'))
verdict = r['final_verdict']
assert verdict == 'delay_carrier_heads_and_layers_localized'
assert r['anchors']['ok'] is True

led = json.load(io.open(LEDGER, encoding='utf-8'))
old_sha = led.get('ledger_sha256_8')
led.pop('ledger_sha256_8', None)
n_before = len(led['measurements'])
meas_id = 'meas2970_delay_carrier_localization'
led['measurements'].append({
    'meas_id': meas_id,
    'phase': 2970,
    'name': 'delay_carrier_localization',
    'created': e['created'],
    'verdict': verdict,
    'artifacts': {
        'execution.json': shas['execution.json'],
        'result.json': shas['result.json'],
        'delay_carrier.npz': shas['delay_carrier.npz'],
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
## Phase 2970: 跨语言峰位延迟载体定位——深层带 L24-34 八层 + 分布式头群 [%(STAMP)s]

**设计**：2968 协议 verbatim family A（+xdir，57 词，GRID17+s0，K=1），本次保留全层头级贡献 C[36,11,57,32]；层级（sum_h 曲线）与头级（per (l,h) 曲线）各自 peak_loc → 与 2969 封存口径一致的配对（先过滤峰词再 cidx，2969 教训：2887 是四语言表，同 concept 多 L 词，cidx 覆盖取最后）→ 配对符号翻转置换 + maxT。

**关键结果**（run5 权威，锚 15/15）：
- **锚**：a12/a13 vs 2967 npz bit 0、a14 C15 峰集 40 词恒等、**a15 d[34,15]=0.5132 vs 2969 diff 3.81e-06（口径恒等门，新增）**——a15 在 run4 抓住配对口径漂移（11 对 vs 13 对）。
- **T2 层级（36 层 maxT）**：**八层显著 [24,25,26,27,30,31,32,34]**，d 全正（0.56-0.83）——延迟带与 2967 塌缩载体带 L24-33 高度重叠，L34 亦显著。
- **T1 头级（359/1152 有效头，maxT q<0.05）**：top = L30/h17 +0.864、L32/h30 +0.861、L28/h5 +0.833、L19/h13 −0.815、L22/h29 +0.803——**全部在深层带，符号双向（多数 L 词峰位晚 = 正 d，少数头反向）**。
- **T3 描述性**：rho(延迟, 2967 塌缩响应) = −0.486（L34 头级）——峰位延迟与塌缩响应强度中度负相关。

**判决：`delay_carrier_heads_and_layers_localized`**（run5 权威）。

**结论**：跨语言峰位延迟（2969 的配对语言效应）的载体是**深层带 L24-34 八层 + 分布式头群（359 头），与 B 塌缩载体带 L24-33 重叠**——语言身份对词类瞬态峰位的调制与语言注入对带差分的抹平共享同一深层带载体结构；头级延迟双向（正 d 主导、少数负 d 头），无单点必要载体，与 2965/2967 的分布式结论一致。机制链新增一环：**语言轴在深层带同时调制静态带差分（2967 塌缩）与动态峰位时序（本 Phase 延迟），两者共享载体带**。

**硬伤与勘误（run1→run5，四次 correction）**：① run1 源路径凭记忆构造（layer_dirs/func_ci_arms）FileNotFoundError——2965 键核对教训在路径层复现；② run2 T1 分组 maxT 广播 (250,13) vs (1,139,13)——sg 切片需 [:, None, :L]；③ run3 **transpose 流顺序错误**：C_all (nS,NL,W,H) 要抽 per-(l,h) 的 (s,W) 矩阵，必须 transpose(1,3,0,2)（l→h→s→w 流），transpose(1,0,3,2) 后 reshape 把 NH 轴折进元素流产生纯噪声——run3 的"T1 显著"是噪声上的假显著，d[34,15]=0（h15 被门剔除）是首证信号；④ run4 **配对口径漂移被 a15 抓住**：2887 四语言表同 concept 多 L 词，cidx 全词覆盖（后过滤）11 对 vs 2969 峰词先行过滤 13 对——修复为 2969 口径 verbatim 并新增 a15 恒等门；⑤ run5 权威。教训入 MEMORY（第 26 条：跨产物口径恒等门 + 高维 transpose 流顺序自检必须用已知量锚定）。

**产物**：`phase2970/delay_carrier_localization/` execution {a1} / result {b1} / delay_carrier.npz {c1} / script {d1}。

**接续（2971 候选）**：A（主选）机制链收官卡片扩充——2962-2970 九环入 2961 卡组（纯文档 Phase，含四语言表结构注记）；B 语言×词类双因子签名矩阵（n≥60 合并词表预注册）；C 延迟头群功能身份（top 延迟头消融，2965 机器）；D h8/h21 峰位词属性离线检验。
""" % {'STAMP': STAMP, 'a1': shas['execution.json'],
       'b1': shas['result.json'], 'c1': shas['delay_carrier.npz'],
       'd1': shas['script']}
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(sec)
print('memo appended')

wl = io.open(WSLOG, encoding='utf-8').read()
if 'Phase 2970' not in wl:
    entry = (u"\n## Phase 2970（2026-09-20）跨语言峰位延迟载体定位\n"
             u"- 判决 delay_carrier_heads_and_layers_localized（锚 15/15 含新 a15 口径恒等门，run5 权威）。\n"
             u"- T2 层级八层显著 [24,25,26,27,30,31,32,34] 与 2967 塌缩载体带重叠；T1 359 有效头显著、top 全在 L19-32。\n"
             u"- 四次 correction：源路径、maxT 广播、transpose 流顺序（假显著）、配对口径漂移（a15 抓住）。\n"
             u"- Ledger 109 条 / L14 77 / hash %s。\n" % new_sha)
    with io.open(WSLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
    print('wslog appended')

mem = io.open(MEMFILE, encoding='utf-8').read()
if '跨产物口径恒等门（2970）' not in mem:
    mem = mem.replace('max=**2969**', 'max=**2970**')
    anchor = '- 跨产物复算锚格式规范（2969）：'
    add = (u"""- 跨产物口径恒等门（2970）：跨 Phase 复用统计管线时必须设"已知量恒等门"（如 d[34,15] vs 上游封存值 diff<5.01e-4）——它抓住了配对口径漂移（2887 是四语言表，同 concept 多 L 词，cidx 全词覆盖后过滤=11 对 vs 峰词先行过滤=13 对）；高维 reshape 抽 per-(l,h) 矩阵必须 transpose(1,3,0,2)（l→h→s→w 流），错误 transpose 把 NH 轴折进元素流产生纯噪声且 maxT 会给假显著——d[34,15]=0（h15 被门剔除）是首证信号，任何"top 头不含已知效应头"的结果都应先怀疑索引。科学结论：跨语言峰位延迟载体 = 深层带 L24-34 八层 + 359 头分布式（与 2967 塌缩载体带重叠），语言轴在深层带同时调制静态带差分与动态峰位时序。
""")
    assert anchor in mem, 'memory anchor missing'
    mem = mem.replace(anchor, add + anchor, 1)
    io.open(MEMFILE, 'w', encoding='utf-8').write(mem)
    print('memory updated')
print('closeout done')
