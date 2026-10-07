# -*- coding: utf-8 -*-
"""Phase 2990 closeout: seal -> ledger 129 -> MEMO -> wslog
-> MEMORY.md (<=3000 chars guard)."""
import hashlib
import json
import os
import re
import time

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2990', 'neuron2_deep_dive')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
WSLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
         r'\.workbuddy\memory\2026-09-20.md')
MEMORY = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
          r'\.workbuddy\memory\MEMORY.md')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2990_neuron2_deep_dive.py')

log_lines = []


def sha8(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


def note(m):
    log_lines.append(m)
    print(m, flush=True)


# ---------- 1. refresh correction_note in result.json ----------
rp = os.path.join(OUT, 'result.json')
res = json.load(open(rp, encoding='utf-8'))
assert res['final_verdict'] == 'neuron2_not_single_causal'
res['correction_note'] = (
    '3 corrections before authoritative run4: '
    '(c1) run1 TypeError - fin captured with batch dim '
    '(1,2560), float(fin@d) needs reshape(-1) (sweep A '
    'site); '
    '(c2) run2 a7 anchor fail bit 15.9 - missing '
    'condition-dim index on 2987 npz comparison '
    '(prof_all/BL are (5,74,36)/(5,74), L2 condition = '
    '[0] as in 2989 a8); '
    '(c3) run3 same fin-reshape bug at second site '
    '(sep74 closure); run4 authoritative.')
json.dump(res, open(rp, 'w', encoding='utf-8'), indent=2,
          ensure_ascii=False)
note('correction_note refreshed')

h_exec = sha8(os.path.join(OUT, 'execution.json'))
h_res = sha8(rp)
h_npz = sha8(os.path.join(OUT, 'neuron2_deep_dive.npz'))
h_scr = sha8(SCRIPT)
note('hashes exec=%s res=%s npz=%s scr=%s'
     % (h_exec, h_res, h_npz, h_scr))

# ---------- 2. seal ----------
seal = {
    'phase': 2990,
    'sealed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'verdict': 'neuron2_not_single_causal',
    'anchors': '13/13 (six bit-level 0.00: a2/a3/a4/a6/'
               'a7/a9 + a10 exact-zero slices; a5 gap '
               'recompute 4.70e-08 argmax=15; a5b headC30 '
               'vs 2964 C[34] 2.45e-08; a1 3.04e-08)',
    'verdict_caveat': 'none (frozen branches followed '
                      'literally; dose rho=0.80 p=0.0667 '
                      'is directional but below gate - '
                      'registered as descriptive)',
    'key_numbers': {
        'T1_r_cls_74': 0.471,
        'T1_p_rcls_perm': 0.0,
        'T1_act30_mean_F': -23.99,
        'T1_act30_mean_C': 0.19,
        'T1_tail_dominated_by': 'F:fr function words '
                                '(et/sur/le/la <= -41)',
        'T2_dsep_real': 0.295,
        'T2_dsep_rand_median': 0.205,
        'T2_p_rank': 0.42,
        'T2_dose_rho': 0.80,
        'T2_p_dose': 0.0667,
        'T3_dgap_h15': -0.045,
        'T3_p_h15': 0.38,
        'T3_null_h15_p95': 0.102,
        'T4_cos_down_ucls': 0.4695,
        'T4_percentile': 0.9999},
    'files': {'execution.json': h_exec,
              'result.json': h_res,
              'neuron2_deep_dive.npz': h_npz,
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

m2990 = {
    'meas_id': 'M2990_neuron2_deep_dive',
    'phase': 2990,
    'claim': ('L12 neuron #2 (2989 cls top-1) deep-dive: '
              'snapshot-extreme (down-col cos 0.4695, '
              '99.99th percentile of 9728; r_cls 0.471 '
              'p<1e-4; activation tail dominated by French '
              'function words, 30-word set F mean -23.99 '
              'vs C +0.19) yet NOT single causal: k=1 '
              'ablation dsep 0.295 vs rand median 0.205, '
              'p=0.42 (N_RAND=99); dose rho 0.80 p=0.067 '
              'below gate; no coupling to head carrier '
              'L34/h15 (dgap -0.045, p=0.38, null p95 '
              '0.102) - cls concentration is a snapshot '
              'property, causally redundant; single-point '
              'operationalization closed for BOTH axes at '
              'neuron granularity'),
    'verdict': 'neuron2_not_single_causal',
    'artifacts': {
        'result': 'phase2990/neuron2_deep_dive/result.json',
        'npz': 'phase2990/neuron2_deep_dive/'
               'neuron2_deep_dive.npz'},
    'hashes': {'execution': h_exec, 'result': h_res,
               'npz': h_npz, 'script': h_scr},
    'anchors': 'a1-a10+sub all pass; six bit-level 0.00 '
               '(a2/a3/a4/a6/a7/a9) + a10 exact-zero; '
               'a5 4.70e-08 argmax h15; a5b headC30 vs '
               '2964 C[34] 2.45e-08',
    'corrections': 3,
    'note': ('dual word sets: 74 cells (2977) + 2964 '
             '30-word set (stored tids, bit-level head '
             'reproduction); resolves 2989 branch-name '
             'caveat: the cls single-point lead is now '
             'directly tested and negative'),
}
led['measurements'].append(m2990)
for l in led['linkage']:
    if not isinstance(l, dict):
        continue
    if l.get('link_id') == 'L14_readout_spectrum_cross_model':
        cs = l.get('connects')
        if isinstance(cs, list):
            cs.append({'phase': 2990,
                       'meas': 'M2990_neuron2_deep_dive'})
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
stamp = '2026-09-20 08:50'
sec = (
    '\n---\n\n'
    '## Phase 2990: L12 词类神经元深挖——快照极端、因果冗余，'
    '单点操作化双轴关闭 [%(stamp)s]\n\n'
    '**日期**：2026-09-20。**模型**：qwen3-4b。运行 397.7s'
    '（双词集约 1.08 万前向：74 cells + 2964 30 词集 + '
    '99 随机单神经元对照 x 双词集 + 剂量 4 点）。\n\n'
    '### 原理\n\n'
    '2989 发现词类轴单点显眼神经元 L12 #2（c_cls 0.588、'
    'maxT p<1e-4、top-128 份额 0.762、down 列近共线），'
    '但其因果性未直接检验。本轮双词集设计：74 cells '
    '（2977 verbatim）+ 2964 的 30 词集（存档 tids 直接'
    '复用，headC30 vs 2964 C[34] bit 级 2.45e-08——跨产物'
    '锚最强的头级复现之一）。检验四层：T1 身份（激活'
    '剖面 + 点二列相关）、T2 单神经元真实消融 k=1 vs '
    '99 随机对照（N_RAND=99，p 分辨率 0.01，响应 2989 '
    '教训）+ 剂量曲线 f in {0,.25,.5,.75,1}、T3 跨粒度'
    '耦合（消融该神经元对 L34 逐头 gap 的位移，h15 载体'
    '对账）、T4 几何（cos 百分位）。\n\n'
    '### 判决：neuron2_not_single_causal（冻结分支，'
    '无瑕疵）\n\n'
    '**核心结果（重复三遍）：L12 #2 是快照极端但因果'
    '冗余——down 列与 u_cls 近共线 cos 0.4695（9728 '
    '神经元中 99.99 百分位）、激活与词类点二列 r 0.471'
    '（置换 p<1e-4）、30 词集上 function 均值 -23.99 vs '
    'content +0.19 且激活尾完全被法语功能词占据'
    '（et/sur/le/la 全 <= -41，语言x词类合取）；但'
    '单神经元消融对 cls 读出的位移 0.295 落在 99 个'
    '随机单神经元对照分布内（中位 0.205，p=0.42），'
    '剂量曲线 rho 0.80 不达门（p=0.067），且与头级'
    '载体 L34/h15 无显著耦合（dgap_h15 -0.045，p=0.38，'
    'null p95 0.102）。2989 的"cls 集中属性"由此定性：'
    '集中是快照/几何属性，因果上冗余分布式承载——'
    '神经元粒度上单点操作化对双轴（lang 与 cls）'
    '全部关闭，2982 关系属性结论在最强单点候选上'
    '终审。**\n\n'
    '- 身份侧写：激活尾 = 法语功能词（语言x词类合取'
    '检测器样貌），但 down 列语言对齐仅 0.122 vs 词类 '
    '0.469——该神经元的"语义"由读出投影定义而非激活'
    '模式定义，激活空间与读出空间错位是单点判据'
    '失效的机制候选。\n'
    '- T2 rand 中位 0.205 为正：任意单神经元消融'
    '平均也推 cls 分离 0.2 量级（重平衡背景），'
    '真实单点效应需超此背景——#2 未超。\n'
    '- 2989 分支命名瑕疵就此解决：cls 单点线索本轮'
    '直接检验且为负，"snapshot_only" 判读对双轴一致。\n\n'
    '### 硬伤与勘误（三次 correction，均删产物重跑）\n\n'
    '1. run1：TypeError——hook 捕获 fin 含 batch 维 '
    '(1,2560)，float(fin@d) 须 reshape(-1)（sweep A 处）；\n'
    '2. run2：a7 anchor fail bit 15.9——对比 2987 npz '
    '漏条件维索引（prof_all/BL 为 (5,74,36)/(5,74)，'
    'L2 条件 = [0]，2989 a8 原版如此）——**跨产物对比'
    '不仅要核对构造口径，还必须核对数组维度语义**；\n'
    '3. run3：同款 fin reshape 缺口在 sep74 闭包第二处'
    '——同文件内同型代码逐处核查教训。\n\n'
    'run4 权威锚 13/13：六重 bit 级 0.00（a2/a3/a4/a6/'
    'a7/a9）+ a10 精确零切片；a5 4.70e-08 且 argmax=h15；'
    'a5b headC30 vs 2964 C[34] 2.45e-08。\n\n'
    '### 文件与 SHA256-8\n\n'
    '| 文件 | sha256_8 |\n|---|---|\n'
    '| phase2990/neuron2_deep_dive/execution.json | '
    '%(h_exec)s |\n'
    '| phase2990/neuron2_deep_dive/result.json | '
    '%(h_res)s |\n'
    '| phase2990/neuron2_deep_dive/neuron2_deep_dive.npz '
    '| %(h_npz)s |\n'
    '| tests/glm5/phase2990_neuron2_deep_dive.py | '
    '%(h_scr)s |\n\n'
    '### 接续（2991 候选）\n\n'
    '- **A（主选）**：P1b 载体迁移图谱——2987 的 '
    'h11/h23 迁移线索 x 2989 注册表并表，给出协议相对'
    '载体的系统地图；\n'
    '- B：2989 T3 加密复测（N_RAND=99）+ k 剂量曲线'
    '（top-128 规模扫描）；\n'
    '- C：P2 稀疏字典登记（单点可操作化已证伪，'
    '字典路线按 null 校准前置）；\n'
    '- D：30 词集跨域正式化（function/noun 命名协议'
    '下的激活-读出错位机制检验）。\n\n'
    '*(SHA 与判决以 result.json 为准；本节由 phase2990 '
    '收尾脚本追加。)*\n\n\n'
) % {'stamp': stamp, 'h_exec': h_exec, 'h_res': h_res,
     'h_npz': h_npz, 'h_scr': h_scr}
memo = open(MEMO, encoding='utf-8').read()
assert '## Phase 2990:' not in memo, 'MEMO already has 2990'
memo += sec
open(MEMO, 'w', encoding='utf-8').write(memo)
note('MEMO appended')

# ---------- 5. workspace log ----------
ws = ''
if os.path.exists(WSLOG):
    ws = open(WSLOG, encoding='utf-8').read()
entry = ('- Phase 2990 L12 词类神经元深挖（run4 权威 '
         '397.7s）：判决 neuron2_not_single_causal。'
         '快照极端（cos 0.4695 = 99.99 百分位、r_cls 0.471 '
         'p<1e-4、激活尾被法语功能词占据）但单神经元消融 '
         'p=0.42（99 对照）、剂量 rho 0.80 p=0.067 不达门、'
         '与 h15 无耦合（p=0.38）——cls 集中=快照属性，'
         '单点操作化双轴关闭。三次 correction（fin '
         'reshape x2、2987 条件维索引）。Ledger 129 / '
         'L14 97 / hash %(h)s。接续 2991A=P1b 载体迁移'
         '图谱。\n') % {'h': led['ledger_sha256_8']}
open(WSLOG, 'a', encoding='utf-8').write(entry)
note('wslog appended')

# ---------- 6. MEMORY.md ----------
mem = open(MEMORY, encoding='utf-8').read()
old_tail = ('cls 单点 L12（p<1e-4/share 0.76/cos 0.588）。')
if old_tail in mem:
    mem = mem.replace(
        old_tail,
        'cls 单点 L12（p<1e-4/share 0.76/cos 0.588）；'
        '2990 L12#2 深挖：快照极端（cos pct 99.99、'
        'r_cls p<1e-4）但消融 p=0.42+无剂量门+h15 无耦合'
        '——cls 集中也是快照属性，单点操作化双轴关闭。', 1)
else:
    raise SystemExit('MEMORY anchor not found')
mem2 = re.sub(r'max=29\d+，下一个 \*\*\d+\*\*[^。\n]*。',
              'max=2990，下一个 **2991**（A 主选 P1b 载体'
              '迁移图谱：2987 h11/h23 x 2989 注册表并表；'
              'B T3 加密 N_RAND=99+k 剂量；C P2 稀疏字典；'
              'D 30 词集激活-读出错位）。', mem)
assert 'max=2990' in mem2, 'next update failed'
if len(mem2) > 3000:
    mem2 = mem2.replace(
        '2988 峰锁/词盲 L16 复制、卡片集 v2（稳健5/'
        'len2专属3/未测26）；',
        '2988 峰锁/词盲 L16 复制、卡片集 v2；', 1)
if len(mem2) > 3000:
    mem2 = mem2.replace(
        '2990 L12#2 深挖：快照极端（cos pct 99.99、'
        'r_cls p<1e-4）但消融 p=0.42+无剂量门+h15 无耦合'
        '——cls 集中也是快照属性，单点操作化双轴关闭。',
        '2990 L12#2 深挖：快照极端但消融 p=0.42——'
        'cls 集中也是快照属性，单点操作化双轴关闭。', 1)
if len(mem2) > 3000:
    raise SystemExit('MEMORY over limit: %d' % len(mem2))
open(MEMORY, 'w', encoding='utf-8').write(mem2)
note('MEMORY chars=%d' % len(mem2))

with open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
          r'\.workbuddy\tmp_closeout2990_log.txt', 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(log_lines) + '\n')
print('CLOSEOUT OK', flush=True)
