# -*- coding: utf-8 -*-
"""Phase 2897 closing: ledger (block + measurement + growth +
linkage + errata + L13 refinement) + MEMO append."""
import hashlib
import io
import json
import os

LEDGER = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase2897'
       r'\glm4_readout_spectrum')


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:8]


sha_exec = sha8(os.path.join(OUT, 'execution.json'))
sha_res = sha8(os.path.join(OUT, 'result.json'))
sha_npz = sha8(os.path.join(OUT, 'glm4_readout_spectrum.npz'))
assert sha_exec == '16856795' and sha_res == 'e6291a28' \
    and sha_npz == '9d2a28bc', 'sha drift'

d = json.load(io.open(LEDGER, encoding='utf-8'))
ids_blocks = [b['block_id'] for b in d['blocks']]
ids_meas = [m['meas_id'] for m in d['measurements']]
ids_growth = [g.get('growth_id', '') for g in d['growth_curve']]
ids_link = [l.get('link_id', '') for l in d['linkage']]

if 'B_readout_spectrum_glm4' not in ids_blocks:
    d['blocks'].append({
        'block_id': 'B_readout_spectrum_glm4', 'axis_id': 'language',
        'kind': 'triple_condition_injection', 'model':
        'glm4-9b-chat-hf',
        'shape': [78, 12],
        'src': {'path': 'phase2897/glm4_readout_spectrum/'
                        'glm4_readout_spectrum.npz',
                'sha256_8': sha_npz, 'phase': 2897,
                'key': 'B_attn_orth'},
        'notes': 'W=[28,40); stale(lang_dir)/eigen(dir_q(li))/'
                 'orth(unit(dir_q-(dir_q.lang_dir)lang_dir)) triple '
                 'injection, dual channel mlp+self_attn, 2890 vocab/'
                 'conds verbatim 78 words; eigen+orth attn-positive, '
                 'eigen mlp-positive'})

if 'M2897_glm4_readout_spectrum' not in ids_meas:
    d['measurements'].append({
        'meas_id': 'M2897_glm4_readout_spectrum',
        'type': 'channel_readout_discrimination', 'model':
        'glm4-9b-chat-hf',
        'verdict': 'broad_tolerance_glm4 (frozen mapping: orth+ & '
                   'stale+). attn: eigen acc 0.7308/p95 0.6282 '
                   'margin 0.1213 STRONG; orth acc 0.6795 margin '
                   '0.1222 STRONG; stale acc 0.6795>p95 but margin '
                   '-0.0161 FRAGILE (M2891 same protocol/seed-draw '
                   'gave acc 0.5897<p95, margin -0.0011 -> stale '
                   'cell straddles p95 across null-token draws, '
                   'margin null in BOTH runs). mlp: eigen acc '
                   '0.6795 margin 0.0198; stale 0.5385 negative '
                   '(replicates M2890); orth 0.6282 vs p95 0.6288 '
                   'borderline-negative. Concept controls 0.000-'
                   '0.077. v1 0.0/0.0 shape-matched. Spectrum: '
                   'qwen mlp subspace-tolerant > glm4 attn '
                   'partially-tolerant (eigen+orth) > glm4 mlp '
                   'direction-matched (eigen only)',
        'source': {'path': 'phase2897/glm4_readout_spectrum/'
                           'result.json',
                   'sha256_8': sha_res, 'phase': 2897}})

if 'G_glm4_readout_spectrum' not in ids_growth:
    d['growth_curve'].append({
        'growth_id': 'G_glm4_readout_spectrum', 'phase': 2897,
        'model': 'glm4-9b-chat-hf', 'axis': 'language',
        'finding': 'GLM4 readout is channel-split: attention is '
                   'partially subspace-tolerant (orth component '
                   'carries full retrieval signal, margin 0.1222 '
                   '>> p95 0.0434) while mlp is direction-matched '
                   '(eigen only; orth borderline-negative). 2896 '
                   'binary qwen-vs-glm4 contrast refines to a '
                   'spectrum; the stale-direction cell is null-draw '
                   'fragile (acc straddles p95, margin null in both '
                   'M2891 and M2897) and needs a multi-draw '
                   'robustness check before any strict claim'})

if 'L14_readout_spectrum_cross_model' not in ids_link:
    d['linkage'].append({
        'link_id': 'L14_readout_spectrum_cross_model', 'phase':
        2897, 'type': 'cross_model_spectrum',
        'connects': ['M2896_qwen_readout_type',
                     'M2897_glm4_readout_spectrum', 'M2893_'
                     'language_perlayer_glm4', 'M2890_language_'
                     'axis_glm4', 'M2891_language_attn_glm4'],
        'notes': 'readout tolerance spectrum: qwen mlp all-three '
                 '(margins 0.18-0.22) > glm4 attn eigen+orth '
                 '(margins ~0.12) > glm4 mlp eigen-only; rotation '
                 'dynamics universal (M2895); carrier-law per-model '
                 'variation = readout tolerance gradient'})

errata_exists = any(e.get('corrects', '') ==
                    'M2891_language_attn_glm4' and
                    e.get('phase') == 2897
                    for e in d['errata_ledger'])
if not errata_exists:
    d['errata_ledger'].append({
        'corrects': 'M2891_language_attn_glm4',
        'note': 'M2891 attn_language_signal_absent reading refined '
                'by M2897: the stale-direction attn cell is '
                'null-draw fragile - acc 0.5897 (M2891, SEED=2891) '
                'vs 0.6795 (M2897, SEED=2897) straddles null p95 '
                '~0.63 with margin <= 0 in BOTH runs; the robust '
                'M2891 conclusion (no strong stale signal) stands, '
                'but strict "direction-matched" is untenable for '
                'attn (orth margin 0.1222 positive in M2897); '
                'open candidate: multi-draw robustness check',
        'phase': 2897})

for l in d['linkage']:
    if l.get('link_id') == 'L13_rotation_vs_carrier_refuted':
        if 'M2897' not in l.get('notes', ''):
            l['notes'] = (l.get('notes', '') + ' | refined by M2897: '
                          'glm4 attn is partially tolerant (orth '
                          'margin 0.1222), spectrum not binary; '
                          'stale cell null-draw fragile')

with io.open(LEDGER, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
d2 = json.load(io.open(LEDGER, encoding='utf-8'))
assert len(d2['blocks']) == 22 and len(d2['measurements']) == 36 \
    and len(d2['growth_curve']) == 22 and len(d2['linkage']) == 14 \
    and len(d2['errata_ledger']) == 6, 'ledger count mismatch'

memo_section = """
### Phase 2897 - GLM4 三条件读出谱系判别（broad_tolerance_glm4）

**日期**：2026-09-19。**模型**：glm4-9b-chat-hf。运行 386.5s（零前向
方向 + 窗口 [28,40) 前向，2890 词表/条件 verbatim 78 词，双通道
mlp + self_attn 三条件）。

### 原理与设计

2896 判 qwen = subspace_readout_tolerant，GLM4 侧缺 orth 格。三条件
逐层注入（方向全部零前向自 2890 npz）：

- stale：注入 lang_dir（li=20），投影 lang_dir
- eigen：注入 dir_q(li)（本层本征方向），投影 dir_q(li)
- orth：注入 orth(li) = unit(dir_q(li) − (dir_q(li)·lang_dir)lang_dir)
  （cos(dir_q,lang_dir) 窗口内 0.422 -> 0.078；cos(orth,lang_dir)=0
  构造保证），投影 orth(li)

B = g(same) − 0.5(g(func)+g(null))；loo-NN acc vs null p95
（200 perms, SEED=2897）；概念对照；v1 形状匹配重算 <1e-6（实测
0.0/0.0）。冻结判决映射：orth+ & stale+ => broad_tolerance（errata
vs M2890/M2891）；orth+ & stale- => partial_tolerance；orth- &
eigen+ & stale- => direction_matched_strict。

### 判决：broad_tolerance_glm4（按冻结映射）

| 通道/条件 | acc vs null p95 | margin vs p95 | 概念对照 |
|---|---|---|---|
| mlp/stale | 0.5385 < 0.6282（阴性，复制 M2890） | 0.0390 > 0.0345 | 0.0769 |
| mlp/eigen | **0.6795 > 0.6410**（复制 M2893） | 0.0198 < 0.0402 | 0.0000 |
| mlp/orth | 0.6282 vs 0.6288（边缘阴性） | 0.0098 < 0.0381 | 0.0000 |
| attn/stale | 0.6795 > 0.6282（**与 M2891 冲突**） | −0.0161 < 0.0430 | 0.0000 |
| attn/eigen | **0.7308 > 0.6282** | **0.1213 > 0.0426** | 0.0128 |
| attn/orth | **0.6795 > 0.6667** | **0.1222 > 0.0434** | 0.0000 |

### 冲突诊断与诚实登记（errata 入账）

attn/stale 与 M2891（acc 0.5897 < p95 0.6410）协议完全一致（同窗口/
词表/注入/eps），唯一随机差异是 null-token 抽取（SEED 2891 vs
2897）。两轮 margin 均 <= 0（−0.0011 / −0.0161）——**stale 格是
阈值边缘的 null-draw 脆弱信号**，acc 跨 p95 摆动、margin 双轮皆
空。errata_ledger 登记 corrects=M2891：严格"direction-matched"对
attn 不成立，但"无强 stale 信号"结论维持；开放候选：多重 null-draw
稳健性检验。

### 三个实质发现（重复三遍）

1. **GLM4 读出是通道分裂的**：attention 部分子空间容忍（orth 分量
   单独承载完整检索信号，margin 0.1222 远超 p95 0.0434），mlp 方向
   匹配（仅 eigen；orth 边缘阴性）——2896 的二分对照精化为谱系。
2. **读出容忍谱系成形**：qwen mlp（三条件全阳，margin 0.18–0.22）>
   glm4 attn（eigen+orth，margin ~0.12）> glm4 mlp（仅 eigen）——
   L14 登记；载体定律的 per-model 变异 = 读出容忍梯度，且同一模型
   内部通道间已经不同。
3. **旋转子空间信息跨架构成立**：GLM4 attn 对与 lang_dir 构造正交
   的旋转分量（cos=0）给出强检索响应——旋转扫过的子空间携带语言
   信息不是 qwen 特例，与 2894/2895 旋转动力学普适拼合。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2897/glm4_readout_spectrum/execution.json | 16856795 |
| phase2897/glm4_readout_spectrum/result.json | e6291a28 |
| phase2897/glm4_readout_spectrum/glm4_readout_spectrum.npz | 9d2a28bc |
| tests/glm5/phase2897_glm4_readout_spectrum.py | 420eba53 |

Ledger 更新：B_readout_spectrum_glm4 + M2897_glm4_readout_spectrum +
G_glm4_readout_spectrum + L14_readout_spectrum_cross_model +
errata（corrects M2891）+ L13 notes 精化；blocks 22 / measurements 36
/ growth 22 / linkage 14 / errata 6。

### 接续（2898 候选）

- **A（主选）**：attn stale 格多重 null-draw 稳健性检验——同协议
  N 个 SEED 重复（每次 ~6.5min 或削减词表），acc 分布 vs p95，关闭
  脆弱格
- **B**：norm-matched 对照纯化 2892 L1 归因（零前向）
- **C**：Gemma4/Qwen2-7B 下载决策（待用户确认）

*(SHA 与判决以 result.json 为准；本节由 phase2897 收尾脚本追加。)*
"""

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(memo_section + '\n')

print('ledger: blocks=%d meas=%d growth=%d link=%d errata=%d'
      % (len(d2['blocks']), len(d2['measurements']),
         len(d2['growth_curve']), len(d2['linkage']),
         len(d2['errata_ledger'])))
print('memo appended')
