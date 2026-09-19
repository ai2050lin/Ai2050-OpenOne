# -*- coding: utf-8 -*-
"""Phase 2895 closing: ledger (M2895 + growth + linkage) + MEMO."""
import hashlib
import io
import json
import os

LEDGER = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase2895'
       r'\rotation_structure_qwen')


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:8]


sha_exec = sha8(os.path.join(OUT, 'execution.json'))
sha_res = sha8(os.path.join(OUT, 'result.json'))
sha_npz = sha8(os.path.join(OUT, 'rotation_structure_qwen.npz'))
assert sha_exec == 'b2ad83e1' and sha_res == '1019e3ec' \
    and sha_npz == '5f7bb785', 'sha drift'

d = json.load(io.open(LEDGER, encoding='utf-8'))
ids_blocks = [b['block_id'] for b in d['blocks']]
ids_meas = [m['meas_id'] for m in d['measurements']]
ids_growth = [g.get('growth_id', '') for g in d['growth_curve']]
ids_link = [l.get('link_id', '') for l in d['linkage']]

if 'B_rotation_struct_qwen' not in ids_blocks:
    d['blocks'].append({
        'block_id': 'B_rotation_struct_qwen', 'axis_id': 'language',
        'kind': 'direction_sequence_geometry', 'model': 'qwen3-4b',
        'shape': [36, 2560],
        'src': {'path': 'phase2895/rotation_structure_qwen/'
                        'rotation_structure_qwen.npz',
                'sha256_8': sha_npz, 'phase': 2895, 'key': 'dirs'},
        'notes': '2894 protocol verbatim on 2886 S_last (80,37,2560); '
                 'dir(0) excluded (dnorm(0)=0 exactly, same as glm4)'})

if 'M2895_rotation_structure_qwen' not in ids_meas:
    d['measurements'].append({
        'meas_id': 'M2895_rotation_structure_qwen',
        'type': 'rotation_structure', 'model': 'qwen3-4b',
        'verdict': 'rotation_plane_consistent_manifold_fullrank - '
                   'SAME verdict as glm4 (M2894): R2 top2_share '
                   '0.2037 > null p95 0.1225 (concentrated, weak '
                   'plane); R3 er 23.30 vs null median 35.94 '
                   '(full_rank_like); R4 segment spans 84-89 deg '
                   'near-orthogonal. REFUTES the 2894-anticorrelation '
                   'hypothesis: rotation geometry is model-universal, '
                   'NOT what separates carrier-positive (qwen 0.7719) '
                   'from carrier-negative (glm4 lang_dir injection). '
                   'Sharp consequence: qwen deep mlp responds to the '
                   'stale li=18 direction despite window cos decay '
                   '0.456->0.068, glm4 does not - the carrier '
                   'asymmetry lies in CHANNEL READOUT (fixed-'
                   'direction tolerance) not direction dynamics. '
                   'dnorm(0)=0 exactly in BOTH models - class-mean '
                   'language separation starts from zero at '
                   'embeddings universally. Audit dev 1.6e-15; '
                   'runtime 6.0s zero forward',
        'source': {'path': 'phase2895/rotation_structure_qwen/'
                           'result.json',
                   'sha256_8': sha_res, 'phase': 2895}})

if 'G_rotation_universal' not in ids_growth:
    d['growth_curve'].append({
        'growth_id': 'G_rotation_universal', 'phase': 2895,
        'model': 'qwen3-4b+glm4-9b-chat-hf', 'axis': 'language',
        'finding': 'language-direction rotation structure is '
                   'model-universal: same verdict, similar shares '
                   '(early/mid/deep ~0.39/0.34/0.27 both), similar '
                   'graded concentration, segment-local planes, and '
                   'dnorm(0)=0 exact zero start in both models; '
                   'carrier asymmetry (qwen mlp carries, glm4 does '
                   'not) must be a channel-readout property'})

if 'L13_rotation_vs_carrier_refuted' not in ids_link:
    d['linkage'].append({
        'link_id': 'L13_rotation_vs_carrier_refuted', 'phase': 2895,
        'connects': ['M2894_rotation_structure_glm4',
                     'M2895_rotation_structure_qwen',
                     'M2887_language_axis'],
        'relation': 'refutes_anticorrelation',
        'notes': 'rotation-vs-carrier anticorrelation hypothesis '
                 'refuted: both models rotate equally (top2 0.20 vs '
                 '0.16, er 23.3 vs 26.7); qwen channel reads a '
                 'rotated-away fixed direction (cos 0.14-0.46 at '
                 'window) with 0.7719 accuracy while glm4 needs '
                 'layer-matched directions (M2893)'})

with io.open(LEDGER, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
d2 = json.load(io.open(LEDGER, encoding='utf-8'))
assert len(d2['blocks']) == 20 and len(d2['measurements']) == 34 \
    and len(d2['growth_curve']) == 20 \
    and len(d2['linkage']) == 13, 'ledger count mismatch'

memo_section = """
### Phase 2895 - qwen3-4b 旋转结构对照（假说否定 -> 通道读出归因）

**日期**：2026-09-19。**模型**：qwen3-4b（2886 S_last 复用，
(80, 37, 2560)）。运行 6.0s，零前向，2894 协议 verbatim。

### 原理与设计

2894-A 假说："若 qwen 旋转弱而其语言载体阳性，则方向稳定性与载体
强度反相关"。qwen 同协议（dir(li)=unit(mean_en-mean_fr)，li=1..36
——**dnorm(0)=0 精确为零，与 GLM4 同**；段边界按窗口 [26,36) 适配
冻结：EARLY li 1..12 / MID 13..25 / DEEP 26..35；null 1000 次；
审计 |u|=2sin(theta/2) dev **1.6e-15**）。

### 判决：rotation_plane_consistent_manifold_fullrank（与 GLM4 同判）

| 指标 | qwen3-4b | glm4（2894） |
|---|---|---|
| R2 top2_share vs null p95 | **0.2037 > 0.1225** | 0.1648 > 0.1084 |
| R3 有效秩 vs null median | **23.30** vs 35.94 | 26.66 vs 39.95 |
| R4 分段主夹角 | **84–89°** | 83–89° |
| 窗口 cos(dir(li), ref) | 0.456 -> **0.068** | 0.422 -> 0.078 |
| dnorm(0) | **0（精确）** | 0（精确） |
| 份额 early/mid/deep | 0.392/0.342/0.265 | 0.407/0.342/0.251 |

### 解读（三个实质发现，重复三遍）

1. **反相关假说否定**：qwen（载体阳性 0.7719）与 GLM4（lang_dir
   注入阴性）的旋转几何**同构**——同判决、同份额、同分级集中、
   同分段局部平面。旋转动力学不是载体判别因子。
2. **载体不对称归因于通道读出**：qwen 深层 mlp 对 cos 已衰减至
   0.07–0.46 的陈旧 li=18 方向仍强响应（2887 阳性），GLM4 则需
   层匹配方向（2893 阳性 / 2890-2892 陈旧方向阴性）——**qwen 读出
   是固定方向容忍型，GLM4 是方向匹配型**。这是 L13 登记的核心。
3. **dnorm(0)=0 双模型普适**：类均值语言分离在两模型的 embedding
   处都从零开始——类级语言分离由层构建是架构普适事实（句级身份
   在 embed 是另一层面，2892 已区分）。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2895/rotation_structure_qwen/execution.json | b2ad83e1 |
| phase2895/rotation_structure_qwen/result.json | 1019e3ec |
| phase2895/rotation_structure_qwen/rotation_structure_qwen.npz | 5f7bb785 |
| tests/glm5/phase2895_rotation_structure_qwen.py | 1b0266f3 |

Ledger 更新：B_rotation_struct_qwen + M2895_rotation_structure_qwen
+ G_rotation_universal + **L13_rotation_vs_carrier_refuted**；
blocks 20 / measurements 34 / growth 20 / linkage 13。

### 接续（2896 候选）

- **A（主选）**：通道读出性质判别——qwen 深层 mlp 对"陈旧方向容忍"
  的直接检验：在 qwen 用 li=18 方向注入逐层响应谱（2887 已示阳性），
  再测对旋转后正交分量的响应（固定方向容忍 vs 子空间响应判别，
  单窗口前向）
- **B**：norm-matched 对照纯化 2892 L1 归因（零前向）
- **C**：Gemma4/Qwen2-7B 下载决策（待用户确认）

*(SHA 与判决以 result.json 为准；本节由 phase2895 收尾脚本追加。)*
"""

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(memo_section + '\n')

print('ledger: blocks=%d meas=%d growth=%d linkage=%d'
      % (len(d2['blocks']), len(d2['measurements']),
         len(d2['growth_curve']), len(d2['linkage'])))
print('memo appended')
