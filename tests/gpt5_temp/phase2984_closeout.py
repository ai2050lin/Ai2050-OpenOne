# -*- coding: utf-8 -*-
"""Phase 2984 closeout: seal -> Ledger -> MEMO ->
workspace log -> MEMORY.md. Idempotent guards on every
write; no bare % formatting (use f-strings / concat)."""
import hashlib
import io
import json
import os

BASE = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(
    BASE, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913', 'phase2984',
    'h12_ablation_destination')
LEDGER = os.path.join(BASE, 'research', 'gpt5', 'atlas',
                      'atlas_ledger.json')
MEMO = os.path.join(BASE, 'research', 'gpt5', 'docs',
                    'AGI_GPT5_MEMO.md')
WSLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
         r'\.workbuddy\memory\2026-09-20.md')
MEMO_MEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
            r'\.workbuddy\memory\MEMORY.md')
LINK_ID = 'L14_readout_spectrum_cross_model'
OUTLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
          r'\.workbuddy\tmp_closeout2984_log.txt')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


out = []
res = json.load(io.open(os.path.join(RES, 'result.json'),
                        encoding='utf-8'))
exec_json = json.load(io.open(
    os.path.join(RES, 'execution.json'),
    encoding='utf-8'))
verdict = res['final_verdict']
assert verdict == 'ablation_local_to_interaction_only'
stamp = exec_json['created']
s_exec = sha8(os.path.join(RES, 'execution.json'))
s_res = sha8(os.path.join(RES, 'result.json'))
s_npz = sha8(os.path.join(RES,
                          'h12_ablation_destination.npz'))
s_scr = exec_json['script_sha256_8']
out.append('hashes: exec=' + s_exec + ' res=' + s_res
           + ' npz=' + s_npz + ' script=' + s_scr)

# ---------- 1. seal ----------
seal_path = os.path.join(RES, 'seal.json')
if not os.path.exists(seal_path):
    seal = {'phase': 2984,
            'verdict': verdict,
            'sealed_at': stamp,
            'sha256_8': {'execution': s_exec,
                         'result': s_res,
                         'npz': s_npz,
                         'script': s_scr},
            'anchors_ok': res['anchors']['ok'],
            'note': 'run1 authoritative; anchors 8/8; '
                    'a5/a6 cross-phase bit-level 0.00; '
                    'T4 structurally zero by '
                    'implementation (same-layer head '
                    'inputs invariant to o_proj-input '
                    'slice ablation) - descriptive '
                    'only, discipline 17 registered'}
    io.open(seal_path, 'w', encoding='utf-8').write(
        json.dumps(seal, indent=2, ensure_ascii=False))
    out.append('seal written')
else:
    out.append('seal already present')

# ---------- 2. Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
meas = [m for m in led['measurements']
        if m.get('phase') == 2984]
if not meas:
    m = {'phase': 2984,
         'name': 'h12_ablation_destination',
         'model': 'qwen3-4b',
         'created': stamp,
         'question': ('when the necessary interaction '
                      'carrier h12 is ablated, is the '
                      'effect isolated to the '
                      'interaction or does it trigger '
                      'layer/band rebalancing?'),
         'design': ('962 forwards: intact dose-0.1 4 '
                    'conds + h12/rand ablation 4 '
                    'conds x 74; readout prof 2965 '
                    'verbatim; T1/T2 sign-flip maxT '
                    'family 36; T3 band response '
                    'shift'),
         'anchors': ('8/8 ok; a5 I vs 2977 bit-level '
                     '0.00; a6 I vs 2980 I_int '
                     'bit-level 0.00; a7 efficacy '
                     '0.00'),
         'tests': ('T1 main-effect preservation: '
                   'lang 0/36 sig, cls 0/36 sig '
                   '(maxT); T2 interaction '
                   'redistribution 0/36 sig; T3 '
                   'band-response shift 3/3 ns '
                   '(p 0.49-0.78); T4 structurally '
                   'zero (implementation)'),
         'verdict': verdict,
         'sha256_8': {'execution': s_exec,
                      'result': s_res,
                      'npz': s_npz,
                      'script': s_scr}}
    led['measurements'].append(m)
    lk = None
    for l in led['linkage']:
        if l['link_id'] == LINK_ID:
            lk = l
    lk['connects'].append(
        {'phase': 2984,
         'note': ('h12 ablation removes the L17 '
                  'interaction with zero migration: '
                  'main effects (T1), interaction '
                  'profile (T2) and injection-induced '
                  'band response (T3) all unchanged - '
                  'carrier function strictly local'),
         'connects_to': [2977, 2980, 2981, 2982, 2983,
                         2965, 2963]})
    if 'ledger_sha256_8' in led:
        popped = led.pop('ledger_sha256_8')
    else:
        popped = None
    newh = hashlib.sha256(json.dumps(
        led, sort_keys=True,
        ensure_ascii=False).encode(
        'utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = newh
    io.open(LEDGER, 'w', encoding='utf-8').write(
        json.dumps(led, indent=2, ensure_ascii=False))
    out.append('ledger: n=%d L14=%d newhash=%s'
               % (len(led['measurements']),
                  len(lk['connects']), newh))
else:
    out.append('ledger already has 2984')

# ---------- 3. MEMO ----------
memo_txt = io.open(MEMO, encoding='utf-8').read()
if '## Phase 2984:' not in memo_txt:
    sec = (
        '## Phase 2984: h12 消融主效应归宿——功能角色严格局部化 '
        '[' + stamp + ']\n\n'
        '**问题**：2980 证明消融 h12 使 L17 双轴竞争交互消失 '
        '97%，2981/2982/2983 完成机制解剖后遗留归宿问题：移除'
        '必要载体后，效应是仅限于交互项本身，还是触发层/带级'
        '重平衡（主效应迁移、交互跨层再分配、注入带响应漂移）？\n\n'
        '**设计（冻结，962 前向）**：2979/2980 协议 verbatim'
        '（74 词表 + 单位轴方向 + n17 剂量门），stage1 intact '
        '剂量 0.1 四条件，stage2 消融 {h12, rand(h21)} 四条件；'
        '读出 prof（2965 verbatim）全剖面 (74,36) 存档。T1 '
        '主效应保持（每层配对符号翻转 + maxT family 36）、T2 '
        '交互跨层再分配（同机）、T3 注入诱导带响应迁移（gate '
        'p<=0.01）、T4 头级拾取（描述性）。\n\n'
        '**产物**：`phase2984/h12_ablation_destination/` '
        'execution ' + s_exec + ' / result ' + s_res
        + ' / h12_ablation_destination.npz ' + s_npz
        + ' / script ' + s_scr + '。\n\n'
        '**锚 8/8**：a5 I intact vs 2977 逐词 bit 级 0.00；'
        'a6 I intact vs 2980 I_int 逐词 bit 级 0.00（同协议'
        '双跨产物恒等）；a3 n17 rel 1.21e-07；a7 消融效能 '
        '0.00；h_rand=21（rng 29804 与 2980 同抽）。\n\n'
        '**结果**：T1 主效应保持 lang 0/36 sig、cls 0/36 sig'
        '（maxT 门 0.039/0.024，top 层 L17/L34 效应 -0.018/'
        '-0.013 均低于门）；T2 交互再分配 0/36 sig（L17 dI '
        '+0.0232 正是交互消失本身，L18/L32 均 < 门）；T3 带'
        '响应迁移 3/3 ns（p 0.49/0.78/0.53，中位漂移仅 '
        '-0.003~+0.001）；T4 同层其他头 |dC| 中位全 0。\n\n'
        '**判决**：`ablation_local_to_interaction_only`'
        '（按冻结映射：T1 零显著 & T2 零显著）。\n\n'
        '**硬伤/教训**：T4 为实现结构性零——消融挂 o_proj '
        '输入切片，不改变同层其他头的输入（o_proj 输出在 '
        'attention 模块下游），故 dC 同层 bit 级恒为零；该量'
        '由实现定义保证（纪律 17 恒等式禁令实例），不携带'
        '重平衡证据，仅作描述性登记；真正的重平衡检验在 '
        'T2/T3 层面完成。\n\n'
        '**结论（重复三遍）**：h12 的功能角色严格局部化于'
        'L17 双轴交互的生成——消融消除交互（-97.6% 复现）而'
        '单轴主效应（0/36+0/36）、交互剖面（0/36）、注入诱导'
        '带响应（3/3 ns）均无任何层/带迁移。这是机制链首个'
        '“交互项专用载体”定案：主效应分布式（2949/2979）× '
        '交互项头级集中（2980）× 载体功能严格局部（2984）。'
        '与 2965 h15 共享载体形成对照：同为头级承重，h15 承'
        '载共享读出效应（消融后效应仍在，只是幅度降），h12 '
        '承载的交互则是该头独有的计算产物。\n\n'
        '**接续**：Ω-B 多轴融合工作包主线完整（2977-2984 '
        '六环）；下一候选 2985：A（主选）perp 通道几何身份'
        '——2983 正交重定向子空间 SVD 定位（与词类/语言轴'
        '关系）；B Ω-C 长上下文调制开题；C 2978 dose law '
        '分层复测（离线）；D h12 输入响应非线性来源（g=0.89 '
        '的增益曲线形状）。\n')
    with io.open(MEMO, 'a', encoding='utf-8') as f:
        f.write('\n' + sec)
    out.append('memo appended')

# ---------- 4. workspace log ----------
wl = io.open(WSLOG, encoding='utf-8').read()
if 'Phase 2984' not in wl:
    entry = (
        '\n## Phase 2984（2026-09-20）\n'
        '- h12 消融主效应归宿：判决 '
        'ablation_local_to_interaction_only（锚 8/8，'
        'a5/a6 vs 2977/2980 bit 级 0.00）；T1 主效应 '
        '0/36+0/36、T2 交互再分配 0/36、T3 带响应 3/3 ns'
        '——功能角色严格局部化于交互生成。\n'
        '- 教训：T4 同层头级 dC 为实现结构性零（消融挂 '
        'o_proj 输入切片不改变同层其他头输入），纪律 17 '
        '实例，仅描述性。\n'
        '- Ledger 123 / L14 91。产物 '
        'phase2984/h12_ablation_destination/。\n')
    with io.open(WSLOG, 'a', encoding='utf-8') as f:
        f.write(entry)
    out.append('wslog appended')

# ---------- 5. MEMORY.md ----------
mem = io.open(MEMO_MEM, encoding='utf-8').read()
old_chain = ('2983 解析分解：交互由正交重定向通道承载~85%，'
             'h12 词级 I 与范数赤字 rho 0.94，h9 抵消在 '
             'delta 生成层面。核心：')
new_chain = ('2983 解析分解：交互由正交重定向通道承载~85%，'
             'h12 词级 I 与范数赤字 rho 0.94；2984 h12 消融'
             '归宿=功能严格局部化（主效应/再分配/带响应全无'
             '迁移，交互专用载体定案）。核心：')
changed = False
if old_chain in mem:
    mem = mem.replace(old_chain, new_chain, 1)
    changed = True
old_next = ('- max=2983，下一个 **2984**（A 主选 h12 消融'
            '主效应归宿：B 带签名迁移/重平衡；B perp 通道'
            '几何身份；C Ω-C 长上下文；D dose law 分层'
            '复测）。')
new_next = ('- max=2984，下一个 **2985**（A 主选 perp '
            '通道几何身份 SVD；B Ω-C 长上下文开题；C '
            'dose law 分层复测；D h12 增益曲线形状）。')
if old_next in mem:
    mem = mem.replace(old_next, new_next, 1)
    changed = True
else:
    out.append('WARN: MEMORY next-anchor not found')
if changed:
    io.open(MEMO_MEM, 'w', encoding='utf-8').write(mem)
    out.append('memory updated chars=%d' % len(mem))

io.open(OUTLOG, 'w', encoding='utf-8').write(
    '\n'.join(out) + '\n')
print('closeout done')
