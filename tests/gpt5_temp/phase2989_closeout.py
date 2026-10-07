# -*- coding: utf-8 -*-
"""Phase 2989 closeout: seal -> ledger 128 -> MEMO -> wslog
-> MEMORY.md (<=3000 chars guard)."""
import hashlib
import json
import os
import time

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2989', 'mlp_neuron_registry')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
WSLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
         r'\.workbuddy\memory\2026-09-20.md')
MEMORY = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
          r'\.workbuddy\memory\MEMORY.md')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2989_mlp_neuron_registry.py')

log_lines = []


def sha8(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


def note(m):
    log_lines.append(m)
    print(m, flush=True)


# ---------- 1. correction_note into result.json ----------
rp = os.path.join(OUT, 'result.json')
res = json.load(open(rp, encoding='utf-8'))
assert res['final_verdict'] == 'snapshot_only_no_causal'
res['correction_note'] = (
    '3 corrections before authoritative run4: '
    '(c1) run1 a10 IndexError - hook act is (1,9728), '
    'neuron indexing needs reshape(-1); '
    '(c2) run2 a9 rel 2.61 - recompute used decoder-layer '
    'input (pre-LN) instead of true MLP input (post-'
    'attention-layernorm), fixed by hooking layers[li].mlp; '
    '(c3) run3 torch tensor indexing with negative-stride '
    'argsort array - ascontiguousarray in forward1. '
    'Branch-name caveat (registered): verdict follows the '
    'frozen prereg branch literally (T1 maxT lang p=0.51 '
    'fails), but T3 group-ablation IS significant at 6/9 '
    'layers (p=0.0476, resolution floor) and T1b share '
    'p<1e-4 - the name no_causal refers only to the '
    'single-neuron maxT criterion, not to causal ablation.')
json.dump(res, open(rp, 'w', encoding='utf-8'), indent=2,
          ensure_ascii=False)
note('correction_note written')

h_exec = sha8(os.path.join(OUT, 'execution.json'))
h_res = sha8(rp)
h_npz = sha8(os.path.join(OUT, 'mlp_neuron_registry.npz'))
h_scr = sha8(SCRIPT)
note('hashes exec=%s res=%s npz=%s scr=%s'
     % (h_exec, h_res, h_npz, h_scr))

# ---------- 2. seal ----------
seal = {
    'phase': 2989,
    'sealed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'verdict': 'snapshot_only_no_causal',
    'anchors': '10/10 (a9 rel 0.00 after c2; five bit-'
               'level 0.00: a2/a3/a6/a7/a8; a10 exact-zero '
               'slices; a5 identity 2.16e-15)',
    'verdict_caveat': res['correction_note'].split(
        'Branch-name caveat (registered): ')[1],
    'key_numbers': {
        'T1_maxT_lang_min_p': 0.51,
        'T1_maxT_cls_L12_p': 0.0,
        'T1b_share_lang_p_min': 0.0,
        'T1b_share_cls_L12': 0.762,
        'T2_cos_max_cls_L12': 0.588,
        'T2_cos_max_lang_L12': 0.1997,
        'sig_count_lang_range': [1587, 2538],
        'chance_486': 486,
        'T3_sig_layers': ['L6', 'L7', 'L9', 'L10', 'L12'],
        'T3_p_floor': 0.0476,
        'direct_share_range': [0.117, 0.446]},
    'files': {'execution.json': h_exec,
              'result.json': h_res,
              'mlp_neuron_registry.npz': h_npz,
              'script': h_scr},
}
with open(os.path.join(OUT, 'seal.json'), 'w',
          encoding='utf-8') as f:
    json.dump(seal, f, indent=2, ensure_ascii=False)
note('seal written')

# ---------- 3. ledger ----------
led = json.load(open(LEDGER, encoding='utf-8'))
stored = led.pop('ledger_sha256_8')
calc = hashlib.sha256(json.dumps(
    led, sort_keys=True,
    ensure_ascii=False).encode('utf-8')).hexdigest()[:8]
assert calc == stored, 'ledger hash mismatch before append'

m2989 = {
    'meas_id': 'M2989_mlp_neuron_registry',
    'phase': 2989,
    'claim': ('MLP neuron registry (9 layers x 9728 x '
              'lang/cls axes): word-class axis has a '
              'single-point salient neuron (L12 maxT p<1e-4, '
              'top-128 share 0.76, down-col cos 0.588); '
              'language axis has NO single-neuron carrier '
              '(maxT p=0.51) but is distributed (1587-2538 '
              'neurons over permutation p95, 3-5x chance) '
              'with top-128 energy concentration (share '
              'p<1e-4, 8/9 layers) and causal group ablation '
              '(6/9 layers p=0.0476, direct share only '
              '0.12-0.45, rebalancing-dominated; L9 sign '
              'flip) - 2982 relational-property conclusion '
              'replicates at neuron granularity'),
    'verdict': 'snapshot_only_no_causal (frozen branch; '
               'single-neuron maxT criterion failed while '
               'share+causal significant - see caveat)',
    'artifacts': {
        'result': 'phase2989/mlp_neuron_registry/result.json',
        'npz': 'phase2989/mlp_neuron_registry/'
               'mlp_neuron_registry.npz'},
    'hashes': {'execution': h_exec, 'result': h_res,
               'npz': h_npz, 'script': h_scr},
    'anchors': 'a1-a10 all pass; five bit-level 0.00 vs '
               '2973/2979/2987; a5 projection identity '
               '2.16e-15; a9 hook-vs-recompute 0.00; a10 '
               'exact-zero ablation slices',
    'corrections': 3,
    'note': ('verdict follows frozen prereg literally; '
             'T3 causal IS significant (branch-name caveat '
             'in seal.json); L12 word-class neuron is the '
             'top single-point lead for 2990'),
}
led['measurements'].append(m2989)
for l in led['linkage']:
    if not isinstance(l, dict):
        continue
    if l.get('link_id') == 'L14_readout_spectrum_cross_model':
        cs = l.get('connects')
        if isinstance(cs, list):
            cs.append({'phase': 2989,
                       'meas': 'M2989_mlp_neuron_registry'})
led['ledger_sha256_8'] = hashlib.sha256(json.dumps(
    led, sort_keys=True,
    ensure_ascii=False).encode('utf-8')).hexdigest()[:8]
json.dump(led, open(LEDGER, 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)
note('ledger n=%d L14=%d hash=%s'
     % (len(led['measurements']),
        len([c for l in led['linkage']
             if isinstance(l, dict)
             and l.get('link_id')
             == 'L14_readout_spectrum_cross_model'
             for c in l.get('connects', [])
             if isinstance(c, dict)]),
        led['ledger_sha256_8']))

# ---------- 4. MEMO append ----------
stamp = '2026-09-20 08:18'
sec = (
    '\n---\n\n'
    '## Phase 2989: MLP 神经元注册表——语言轴分布式、词类轴单点，'
    '重平衡主导消融差分 [%(stamp)s]\n\n'
    '**日期**：2026-09-20。**模型**：qwen3-4b。运行 587.7s'
    '（74 词 L2 + 约 1.4 万消融前向）。\n\n'
    '### 原理\n\n'
    '方案 v4 P1（Omega-G 微观审计开题）：全前链几乎全在 '
    'attention 头+残差读出侧，神经元归因仅 M2906（权重空间）。'
    '本轮建立第一版 MLP 中间神经元注册表：9 层'
    '（L6-12 承重带 + L17/L34 对照）x 9728 神经元 x 双轴'
    '（u_lang/u_cls，2979 单位方向），三镜分账：T1 快照'
    '（轴投影贡献类差的置换 null + 真 family maxT + top-128 '
    '能量份额）、T2 几何（down 列对齐 vs 随机旋转 null）、'
    'T3 因果（top-128 |Delta_lang| 真实消融 vs 20 随机对照，'
    '直接/重平衡分账 2950 口径）。\n\n'
    '### 判决：snapshot_only_no_causal（冻结分支字面；'
    '命名瑕疵见下）\n\n'
    '| 镜 | 语言轴 | 词类轴 |\n'
    '|---|---|---|\n'
    '| T1 maxT（单神经元类差） | 全层不显著 '
    '（p 0.51-0.9995） | **L12 p<1e-4 显著**，'
    '余层不显著 |\n'
    '| T1 sig_count（超 per-neuron null p95） | '
    '1587-2538（3-5x 机会 486） | 2068-4234（L17 最高） |\n'
    '| T1b top-128 能量份额 | 8/9 层 p<1e-4，份额 '
    '0.19-0.31 | L12 份额 **0.762** |\n'
    '| T2 down 列对齐 | max cos 0.11-0.20 = 3-4x null '
    'p95 0.043；n_exceed ≈ 机会 | L12 cos **0.588**、'
    'L17 0.539（近共线列） |\n'
    '| T3 真实消融 | **6/9 层 p=0.0476**（L6/7/9/10/12；'
    'L11 0.095） | —（lang 主判据） |\n\n'
    '### 核心结果（重复三遍）\n\n'
    '**语言身份在 MLP 神经元粒度无单点承载——但不是无结构：'
    '上千神经元弱显著（3-5x 机会）+ top-128 能量集中'
    '（p<1e-4）+ 组消融因果可见（6/9 层达 N=20 分辨率下限），'
    '且直接通道仅占 12-45%%（L9 符号反转），55-88%% 由竞争'
    '重平衡/间接通道承载；词类轴则存在单点显眼神经元'
    '（L12：maxT p<1e-4、份额 0.76、down 列近共线 cos 0.588）。'
    '2982"头级重要性=关系属性，对单点操作化关闭"在神经元'
    '粒度复现，并首次给出轴间分化：lang=分布式关系属性，'
    'cls=集中属性。**\n\n'
    '- 判决分支命名瑕疵（seal 登记）：冻结预注册要求三条件'
    '（maxT+share+causal）同时成立才给 causal；T1 maxT 失败'
    '落入 snapshot_only_no_causal 字面分支，但 T3 组消融'
    '实际显著——"no_causal" 仅指单神经元判据，不指消融'
    '因果。判据未改，命名教训入账（分支命名须覆盖'
    '组合 outcomes）。\n'
    '- T3 p=0.0476 = 1/21 分辨率下限，6 层全部触底——'
    '真实 p 更低；下一轮 N_RAND 应增至 99（p 分辨率 0.01）。\n'
    '- L9 消融符号反转（预测 -0.49 实测 +1.78）：'
    '2950 竞争重平衡在神经元粒度的又一实例。\n\n'
    '### 硬伤与勘误（三次 correction，均删产物重跑）\n\n'
    '1. run1：a10 IndexError——hook act 为 (1,9728)，'
    '神经元索引须 reshape(-1)；\n'
    '2. run2：a9 rel 2.61——复算误用 decoder-layer 输入'
    '（LN 前），真 MLP 输入是 post_attention_layernorm '
    '之后；改挂 layers[li].mlp pre-hook 后 0.00；\n'
    '3. run3：argsort[::-1] 负 stride 数组作 torch 索引'
    '崩溃——ascontiguousarray 修复。\n\n'
    'run4 权威锚 10/10：a2/a3/a6/a7/a8 五重 bit 级 0.00'
    '（vs 2973/2979/2987），a5 投影分解恒等 2.16e-15，'
    'a9 捕获-复算 0.00，a10 消融切片精确零。\n\n'
    '### 文件与 SHA256-8\n\n'
    '| 文件 | sha256_8 |\n|---|---|\n'
    '| phase2989/mlp_neuron_registry/execution.json | '
    '%(h_exec)s |\n'
    '| phase2989/mlp_neuron_registry/result.json | '
    '%(h_res)s |\n'
    '| phase2989/mlp_neuron_registry/'
    'mlp_neuron_registry.npz | %(h_npz)s |\n'
    '| tests/glm5/phase2989_mlp_neuron_registry.py | '
    '%(h_scr)s |\n\n'
    '### 接续（2990 候选）\n\n'
    '- **A（主选）**：L12 词类神经元深挖——该 down 列'
    '（cos 0.588）的身份、Top-1 激活词表扫描、真实单神经元'
    '消融 vs 2964 头载体对照（神经元-头双粒度对账）；\n'
    '- B：P1b 载体迁移图谱（2987 h11/h23 线索 x 本轮注册表'
    '并表）；\n'
    '- C：T3 加密复测（N_RAND=99）+ 消融响应曲线'
    '（k=32..512 剂量）；\n'
    '- D：P2 稀疏字典登记（若 A 证实单点可操作化则提前）。\n\n'
    '*(SHA 与判决以 result.json 为准；本节由 phase2989 '
    '收尾脚本追加。)*\n\n\n'
) % {'stamp': stamp, 'h_exec': h_exec, 'h_res': h_res,
     'h_npz': h_npz, 'h_scr': h_scr}
memo = open(MEMO, encoding='utf-8').read()
assert '## Phase 2989:' not in memo, 'MEMO already has 2989'
memo += sec
open(MEMO, 'w', encoding='utf-8').write(memo)
note('MEMO appended')

# ---------- 5. workspace log ----------
ws = ''
if os.path.exists(WSLOG):
    ws = open(WSLOG, encoding='utf-8').read()
entry = ('- Phase 2989 MLP 神经元注册表（run4 权威 587.7s）：'
         '判决 snapshot_only_no_causal（冻结分支字面，'
         'T3 实显著见 seal 瑕疵注记）。lang 轴无单点承载'
         '（maxT p=0.51）但分布式弱显著 1587-2538 神经元'
         '（3-5x 机会）+ top-128 能量集中 p<1e-4 + 消融因果 '
         '6/9 层 p=0.0476，直接通道仅 12-45%%；cls 轴单点：'
         'L12 maxT p<1e-4、share 0.76、cos 0.588。'
         '三次 correction（reshape/LN 口径/负 stride）。'
         'Ledger 128 / L14 96 / hash %(h)s。'
         '接续 2990A=L12 词类神经元深挖。\n') % {
    'h': led['ledger_sha256_8']}
open(WSLOG, 'a', encoding='utf-8').write(entry)
note('wslog appended')

# ---------- 6. MEMORY.md ----------
mem = open(MEMORY, encoding='utf-8').read()
old_tail = ('2985 perp 通道身份=单方向 86%% 能量但词级不稳定'
            '+非轴锁定（重写非旋转）。')
if old_tail in mem:
    mem = mem.replace(
        old_tail,
        '2985 perp 重写非固定子空间；2986 漂移非单调 L16 峰、'
        '词类签名 context 脆弱；2987 单 token 即坍缩、'
        'h15 载体迁移 h11；2988 峰锁/词盲 L16 复制、'
        '卡片集 v2（稳健5/len2专属3/未测26）；'
        '2989 MLP 神经元注册表：lang 分布式（maxT 失败但'
        'sig 3-5x+集中+消融因果，直接通道仅 12-45%%）、'
        'cls 单点 L12（p<1e-4/share 0.76/cos 0.588）。', 1)
else:
    raise SystemExit('MEMORY anchor not found')
# next line update
import re
mem2 = re.sub(r'max=29\d+，下一个 \*\*\d+\*\*[^。\n]*。',
              'max=2989，下一个 **2990**（A 主选 L12 词类'
              '神经元深挖：down 列身份+Top-1 扫描+单神经元'
              '消融 vs 2964 头载体双粒度对账；B 迁移图谱；'
              'C T3 加密 N_RAND=99；D P2 字典）。', mem)
assert 'max=2989' in mem2, 'next update failed'
if len(mem2) > 3000:
    mem2 = mem2.replace(
        '2986 漂移非单调 L16 峰、词类签名 context 脆弱；',
        '2986 漂移非单调；', 1)
    mem2 = mem2.replace(
        '2987 单 token 即坍缩、h15 载体迁移 h11；',
        '2987 单 token 坍缩、h15 迁移 h11；', 1)
if len(mem2) > 3000:
    raise SystemExit('MEMORY over limit after compress: %d'
                     % len(mem2))
open(MEMORY, 'w', encoding='utf-8').write(mem2)
note('MEMORY chars=%d' % len(mem2))

with open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
          r'\.workbuddy\tmp_closeout2989_log.txt', 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(log_lines) + '\n')
print('CLOSEOUT OK', flush=True)
