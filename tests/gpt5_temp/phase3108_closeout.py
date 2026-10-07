# -*- coding: utf-8 -*-
"""Phase 3108 closeout (idempotent):
Ledger -> MEMO Phase 3108 -> workspace logs -> MEMORY.md."""
import datetime
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3108'
        r'\omega_p106_degeneracy_subspace')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory'
WLOG_C = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
          r'\.workbuddy\memory')
MEMO_W = WLOG_D + r'\MEMORY.md'
LOGF = OUTD + r'\closeout_log.txt'
NOW = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
TODAY = datetime.date.today().isoformat() \
    if hasattr(datetime, 'date') else ''
import datetime as _dt
TODAY = _dt.date.today().isoformat()
o = []

res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
verb = res['verdict']
assert verb == 'per_material_readout', verb
assert res['sanity'] == 'internal_inconsistency'

# ---------- Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3108
           for m in led['measurements']):
    claim = (
        'Omega-P106 (3108, offline T3 on frozen 3105/3106 '
        'captures, no GPU, 91s) - degeneracy separation '
        'and subspace angles.  Verdict per_material_'
        'readout, with an important correction: the '
        'phenomenon is wider than per-material.  M1 '
        'NOISE FLOOR: same-material random half-split '
        'of train, Top-200 Jaccard mean = 0.118/0.108/'
        '0.108/0.114 (3105 four configs, n_train=1101, '
        'band partially_reproducible) and 0.077/0.066 '
        '(3106, n_train=316, band noise_dominant) - '
        'coordinate identity is NOT reproducible even '
        'within one material; the 3107 G2 cross-material '
        'Jaccard 0.0526 is therefore dominated by '
        'SELECTION NOISE, not by material-specific '
        'coordinate sets.  M2 SUBSPACE ANGLES: all 6 '
        'registered pairs fail - mean cos^2 = 0.0039 '
        '(P1 cross-material last|L8) / 0.0027 (P2) / '
        '0.0032 (P3) / 0.0031 (P4 within-3105) / 0.0060 '
        '(P5 within-3106) / 0.0048 (P6 within-3105 '
        'layers), floors 0.014-0.022 (3105) and '
        '0.081-0.117 (3106); even WITHIN-material pairs '
        'fail (sanity 0/3 -> internal_inconsistency, '
        'pre-registration blind spot, real finding not '
        'a data problem).  M3 SPECTRUM: Top-200 |w| is '
        'nearly flat (PR_norm 0.82-0.88, Gini 0.09-0.10) '
        '- no few dominant coordinates.  PAIRWISE PROBE '
        '(C1, 16 bootstrap solutions): pairwise |cos| '
        'median 0.041 (chance 0.020), vs full-train '
        'solution |cos| median 0.184, participation '
        'ratio 15.33/16 - bootstrap solutions are '
        'near-orthogonal and near-equal: the ridge '
        'truth-readout family is WIDE, FLAT and '
        'DEGENERATE.  KEY THEORETICAL UPDATE (corrects '
        '3107 subspace-level wording): n_train=1101 < '
        'd=2560 makes the linear system UNDERDETERMINED '
        '(solution space >= 1459 dims); L2 minimum-norm '
        'solutions rotate near-orthogonally under '
        'resampling; 3106 (n=316, more underdetermined) '
        'is indeed less stable (floors 0.08-0.12 vs '
        '0.014-0.022) - directional support.  G1 Top-100 '
        'refit must be reinterpreted: |w|-biased '
        'sampling of enough dimensions reads out truth, '
        'not one specific compact subspace.  Readout is '
        'a many-route degenerate solution cone; the '
        'model unembed direction (3107 G3, orthogonal '
        'to probes yet functionally coupled) is one '
        'non-L2 member of the functionally equivalent '
        'family.  NEXT 3109: underdetermination verdict '
        '- (a) n-sweep: subsample 3105 train to n in '
        '{100,200,400,800,1101}, floor/J/PR vs n curve '
        '(prediction: monotone stabilization as n '
        'approaches d); (b) random Top-K control: '
        'random 100/200 coordinates refit AUC vs |w|-'
        'Top-K; (c) functional equivalence: half-A Top-K '
        'refit on half-B.  Decision rule frozen at '
        '3109 seal: random control ~= |w|-Top-K AND '
        'monotone n-sweep -> underdetermined-cone '
        'picture confirmed; random control '
        'significantly worse -> true signal-concentrated '
        'subspace exists (underdetermination only an '
        'amplifier).')
    meas = {
        'meas_id': 'meas3108_omega_p106_'
                   'degeneracy_subspace',
        'phase': 3108,
        'claim': claim,
        'verdict': 'per_material_readout',
        'anchors': 'design_seal.json frozen before '
                   'computation: 6 configs, 6 pairs, '
                   'M1 5 seeds, M2 16 boot r=8 5 '
                   'groupings, thresholds 0.50/0.30/'
                   '+0.15; lambda = recorded values; '
                   'standardization frozen per config '
                   'from full train',
        'artifacts': {
            'result': 'phase3108/omega_p106_'
                      'degeneracy_subspace/'
                      'result.json',
            'seal': 'phase3108/omega_p106_'
                    'degeneracy_subspace/'
                    'design_seal.json',
            'probe': 'tests/gpt5_temp/'
                     'p3108_pairwise_probe_out.txt'},
        'hashes': {},
        'note': 'offline only; SMOKE caught a Gram-'
                'indexing bug (sample indices applied '
                'to coordinate Gram) before the real '
                'run; M3 PR normalization fixed before '
                'real run; probe is diagnostic '
                'appendage, result.json untouched by '
                'it',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3108_omega_p106_degeneracy_subspace')
    import hashlib
    led.pop('ledger_sha256_8', None)
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w', encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False,
                  indent=1)
    o.append('ledger appended n=%d l14=%d sha=%s'
             % (len(led['measurements']),
                len(l14['connects']),
                led['ledger_sha256_8']))
else:
    o.append('ledger already upserted')

# ---------- MEMO Phase 3108 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3108:' not in memo:
    sec = u'''## Phase 3108: Ω-P106 简并分离——同材料两半 Top-200 Jaccard 仅 0.11/0.07（选择噪声主导），解族近正交宽平坦（bootstrap 两两 |cos| 中位 0.041、参与率 15.33/16、|w| 谱 Gini 0.10）→ per_material_readout，且修正 3107：n<d 欠定线性几何（解空间 ≥1459 维）是简并的第一性来源 [[NOW]]

**性质**：T3 第 2 Phase，纯离线（3105/3106 冻结 capture.npz，免 GPU，91s）。预注册（design_seal.json 先于一切统计冻结）：6 配置（3105 last\\|L8 / query_obj\\|L8 / last\\|L5 / last\\|L6；3106 last\\|L8 / query_obj\\|L8，λ=记录值 0.01，标准化冻结自全 train）、6 对（P1 跨材料 last 为主对 + P2/P3 跨材料 + P4/P5/P6 同材料 sanity）、M1 五 seed 半分、M2 16 bootstrap r=8 五分组、阈值 0.50/0.30/+0.15。SMOKE 抓获 Gram 索引 bug（样本索引误用于坐标 Gram）后修复；M3 的 PR 归一化（PR∈[1,200] → PR_norm=PR/200）在正式跑前修正。

### 1. M1 噪声地板：坐标身份在材料内部也不可复现
| 配置 | n_train | 半分 Top-200 Jaccard（5 seeds 均值） | 档位 |
| --- | --- | --- | --- |
| 3105 last\\|L8 | 1101 | 0.118 | partially_reproducible |
| 3105 query_obj\\|L8 | 1101 | 0.108 | partially_reproducible |
| 3105 last\\|L5 | 1101 | 0.108 | partially_reproducible |
| 3105 last\\|L6 | 1101 | 0.114 | partially_reproducible |
| 3106 last\\|L8 | 316 | 0.077 | noise_dominant |
| 3106 query_obj\\|L8 | 316 | 0.066 | noise_dominant |

**同材料、同位置、同层、同 λ，仅随机对半分 train，Top-200 就只重叠 ~11%**（随机期望 ~4.2%，超额交集 ~26 个坐标）。3107 G2 的跨材料 Jaccard 0.0526 因此主要被**坐标选择噪声**解释——"材料特异坐标组"（解释 b）被否定：若真存在，同材料 J 应 ≥0.3。

### 2. M2 主角度：全部 6 对 fail，连同材料 sanity 对也 fail
mean cos² = 0.0039（P1 跨材料 last，主对）/ 0.0027（P2）/ 0.0032（P3）/ 0.0031（P4 同材料跨位置）/ 0.0060（P5）/ 0.0048（P6 同材料跨层），全部 << 0.30。floor：3105 配置 0.014–0.022、3106 配置 0.081–0.117。**预注册 sanity 预期"同材料对 confirmed"被否证（0/3 → internal_inconsistency）**——这是预注册盲点，如实记录：它不是数据问题，而是"16 个 bootstrap 解共享 8 维子空间"假设本身的否证。

### 3. 探针（C1 直接测基本量）+ M3 谱：解族宽、平、简并
bootstrap 16 个 ridge 解两两 |cos|：中位 **0.041**（chance 0.020），与全 train 解 |cos| 中位 0.184，解族参与率 **15.33/16**。Top-200 \\|w\\| 谱：PR_norm 0.82–0.88、Gini 0.09–0.10——**没有少数强坐标主导**。

### 4. 理论更新（修正 3107 的"子空间级对象"表述）
第一性来源是**欠定线性几何**：n_train=1101 < d=2560，约束撑不满坐标空间，线性解空间维度 ≥ 2560−1101 = 1459；L2 最小范数解只是这个巨大解锥的一个代表点，对采样扰动剧烈旋转（bootstrap 近正交是其直接后果）。方向性证据：3106（n=316，更欠定，解空间 ≥2244 维）确实更不稳定（floor 0.08–0.12 vs 3105 的 0.014–0.022；M1 J 0.07 vs 0.11）。**3107 G1 的重新解释**：Top-100 保持 AUC 反映"\\|w\\| 偏置采样 + 维度足够即可读出"，而非"存在特定 ~100 维子空间"。统一图景：真值读出 = **多路简并解锥**（functionally equivalent family），3107 G3 中与探针正交而功能耦合的模型 unembed 方向是这个家族中一个非 L2 成员。这与 AGENTS.md"单坐标不是概念"纪律的定量版本一致，且现在连"固定子空间"版本也被否定。

### 5. 硬伤
① M2 的 r=8 任意；成对 |cos| 只对 C1 做了探针（诊断附件，未进正式 result.json）；② M1 只测坐标身份，未测功能等价（两半 Top-K 交叉 refit 的 AUC，留 3109）；③ 欠定解释的 n-sweep 判决未做（3109 主实验）；④ λ 敏感性未做（λ=0.01 固定为记录值）；⑤ 全部基于 ridge 线性读出。

### 6. 3109 预注册（欠定几何判决，门观测前冻结于 3109 seal）
① n-sweep：3105 train 下采样 n ∈ {100, 200, 400, 800, 1101}，floor/M1-J/PR 随 n 曲线（预测：随 n→d 单调稳定化）；② 随机 Top-K 对照：随机 100/200 坐标 refit AUC vs \\|w\\|-Top-K；③ 功能等价：A 半 Top-K 在 B 半 refit 的 AUC。判决规则：随机对照 ≈ Top-K 且 n-sweep 单调 → **欠定解锥图景确认**（"紧凑子空间"图景否定）；随机对照显著差 → 存在真实信号集中子空间（欠定只是放大器）。之后 3110 T3 收口 → 3111+ T4 多步自回归。

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3108/omega_p106_degeneracy_subspace/`（result.json、design_seal.json、run_log.txt）；探针 `tests/gpt5_temp/p3108_pairwise_probe_out.txt`；脚本 `tests/glm5/phase3108_omega_p106_degeneracy_subspace.py`。
'''
    sec = sec.replace('[[NOW]]', NOW)
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars (Phase 3108)' % len(sec))
else:
    o.append('memo already appended')

# ---------- workspace logs ----------
line_d = ('- Phase 3108 Omega-P106 (offline T3, no GPU): '
          'degeneracy separation -> per_material_readout '
          'with correction. M1: same-material half-split '
          'Top-200 Jaccard only 0.108-0.118 (3105, '
          'n=1101) / 0.066-0.077 (3106, n=316) -> '
          'coordinate selection noise dominates; 3107 G2 '
          'cross-material 0.0526 is noise, not '
          'material-specific sets. M2: all 6 pairs fail '
          '(mean cos^2 0.003-0.006), even within-material '
          'sanity pairs (0/3 -> internal_inconsistency, '
          'pre-reg blind spot). M3+probe: |w| spectrum '
          'flat (Gini 0.10), bootstrap solutions pairwise '
          '|cos| median 0.041, PR 15.33/16 -> wide flat '
          'degenerate solution family. KEY: n_train<d '
          'underdetermined geometry (solution space '
          '>=1459 dims); G1 Top-100 reinterpreted as '
          '|w|-biased sampling + enough dims. NEXT 3109: '
          'n-sweep + random Top-K control + functional '
          'equivalence -> underdetermination verdict.\n')
for wdir in (WLOG_D, WLOG_C):
    wl = wdir + '\\' + TODAY + '.md'
    try:
        prev = io.open(wl, encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3108 Omega-P106' not in prev:
        try:
            with io.open(wl, 'a',
                         encoding='utf-8') as f:
                f.write(line_d)
            o.append('wlog appended %s' % wl)
        except Exception as e:
            o.append('wlog fail %s: %r' % (wl, e))
    else:
        o.append('wlog already %s' % wl)

# ---------- MEMORY.md rewrite ----------
mem_old = io.open(MEMO_W, encoding='utf-8').read()
if 'max=3108' not in mem_old:
    mem_new = mem_old.replace(
        '## 机制链状态（3107）',
        '## 机制链状态（3108）\n'
        '- 3108：简并分离 per_material_readout+修正。同材料'
        '两半 Top-200 J 仅 0.11/0.07（选择噪声主导）；6 对'
        '子空间角全 fail（连 sanity 对 0/3）；解族近正交宽'
        '平坦（bootstrap 两两 cos 0.04、谱 Gini 0.10）。'
        'n<d 欠定几何（解空间 ≥1459 维）是第一性来源；'
        '3107 G1 重解释为 |w| 偏置采样+维度足够。读出='
        '多路简并解锥。\n')
    mem_new = mem_new.replace(
        'max=3107', 'max=3108').replace(
        '下一 3108：**简并分离与子空间角度**（噪声地板+principal angles+谱）→',
        '下一 3109：**欠定几何判决**（n-sweep+随机对照+功能等价）→')
    assert len(mem_new) < 3000, len(mem_new)
    with io.open(MEMO_W, 'w', encoding='utf-8') as f:
        f.write(mem_new)
    o.append('memory updated %d chars' % len(mem_new))
else:
    o.append('memory already')

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
