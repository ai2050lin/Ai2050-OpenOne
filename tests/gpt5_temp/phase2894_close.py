# -*- coding: utf-8 -*-
"""Phase 2894 closing: ledger (M2894 + growth) + MEMO append."""
import hashlib
import io
import json
import os

LEDGER = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase2894'
       r'\rotation_structure_glm4')


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:8]


sha_exec = sha8(os.path.join(OUT, 'execution.json'))
sha_res = sha8(os.path.join(OUT, 'result.json'))
sha_npz = sha8(os.path.join(OUT, 'rotation_structure_glm4.npz'))
assert sha_exec == 'bb0e7519' and sha_res == 'c5bb65ed' \
    and sha_npz == 'fef03893', 'sha drift'

d = json.load(io.open(LEDGER, encoding='utf-8'))
ids_blocks = [b['block_id'] for b in d['blocks']]
ids_meas = [m['meas_id'] for m in d['measurements']]
ids_growth = [g.get('growth_id', '') for g in d['growth_curve']]

if 'B_rotation_struct_glm4' not in ids_blocks:
    d['blocks'].append({
        'block_id': 'B_rotation_struct_glm4', 'axis_id': 'language',
        'kind': 'direction_sequence_geometry', 'model':
        'glm4-9b-chat-hf', 'shape': [40, 4096],
        'src': {'path': 'phase2894/rotation_structure_glm4/'
                        'rotation_structure_glm4.npz',
                'sha256_8': sha_npz, 'phase': 2894, 'key': 'dirs'},
        'notes': 'dir(li)=unit(mean_en-mean_fr S_last[:,li]) for '
                 'li=1..40 (dir(0) undefined: dnorm(0)=0 exactly); '
                 'theta/tangent/PCA rotation structure'})

if 'M2894_rotation_structure_glm4' not in ids_meas:
    d['measurements'].append({
        'meas_id': 'M2894_rotation_structure_glm4',
        'type': 'rotation_structure', 'model': 'glm4-9b-chat-hf',
        'verdict': 'rotation_plane_consistent_manifold_fullrank '
                   '(zero forward, 2890 S_last): R1 rotation shares '
                   'early 0.407 / mid 0.342 / deep 0.251 (caveat: '
                   'early theta inflated by small dnorm, noise-'
                   'dominated directions at li 1-2); R2 top2 tangent-'
                   'PCA share 0.1648 > null p95 0.1084 (1000 random '
                   'draws) => statistically concentrated but weak '
                   'plane (16 percent, not a tight 2D plane); R3 '
                   'effective rank 26.66 vs null median 39.95 - '
                   'graded concentration, fails pre-registered 0.5x '
                   'low-dim threshold; R4 top-2 tangent PCs nearly '
                   'orthogonal to lang_dir (cos -0.088/-0.148) and '
                   'segment top-2 spans nearly mutually orthogonal '
                   '(principal angles 83-89 deg) => the rotation '
                   'plane ITSELF rotates along depth. Audit: '
                   '|u|=2sin(theta/2) max dev 7.8e-16, PCA recon '
                   'exact; amendment: run 1 crashed pre-statistics '
                   '(PC convention), dir(0) excluded with reason '
                   'documented in execution.json',
        'source': {'path': 'phase2894/rotation_structure_glm4/'
                           'result.json',
                   'sha256_8': sha_res, 'phase': 2894}})

if 'G_rotation_struct_glm4' not in ids_growth:
    d['growth_curve'].append({
        'growth_id': 'G_rotation_struct_glm4', 'phase': 2894,
        'model': 'glm4-9b-chat-hf', 'axis': 'language',
        'finding': 'GLM4 language direction rotates with depth in a '
                   'statistically concentrated but slowly-rotating '
                   'subspace: no single persistent rotation plane '
                   '(segment spans near-orthogonal), no low-dim '
                   'manifold (er 26.7/40); dnorm(0)=0 exactly - '
                   'class-mean language separation starts from zero '
                   'at embeddings and is built entirely by layers '
                   '(per-sentence probe identity at li=1 is a '
                   'different, non-class-level fact)',
        'corrects_refines': 'rotation-plane hypothesis (Unified '
                            'Theory): plane exists statistically but '
                            'is segment-local, not global'})

with io.open(LEDGER, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
d2 = json.load(io.open(LEDGER, encoding='utf-8'))
assert len(d2['blocks']) == 19 and len(d2['measurements']) == 33 \
    and len(d2['growth_curve']) == 19, 'ledger count mismatch'

memo_section = """
### Phase 2894 - GLM4 语言方向旋转结构定量（零前向）

**日期**：2026-09-19。**模型**：glm4-9b-chat-hf。运行 14.6s
（零前向，2890 S_last 复用；随机 null 1000 次 + 代数审计）。

### 原理与设计

dir(li) = unit(mean_en S_last[:,li] - mean_fr S_last[:,li])，
li = 1..40（修正案：dnorm(0)=0.0 精确为零，dir(0) 未定义——首次
运行在 R4 崩溃于 PC 约定错误、统计量未冻结，探针定位后按纪律删旧
execution.json 重跑）。theta(li) = 层间夹角；切向量 u(li) =
dir(li+1) - dir(li)（39 个）。R2/R3 均配随机 null（1000 次同构造
随机单位向量）。审计：|u| = 2 sin(theta/2) max dev **7.8e-16**，
PCA 重构精确。

### 判决：rotation_plane_consistent_manifold_fullrank

| 门线 | 观测 | 结果 |
|---|---|---|
| R1 旋转份额 | early 0.407 / mid 0.342 / deep 0.251 | 描述性（early 受小 dnorm 噪声放大，混杂登记） |
| R2 平面集中度 | top2_share **0.1648** > null p95 0.1084（null mean 0.1067） | plane_consistent（统计显著但仅 16%，非紧密 2D 平面） |
| R3 有效秩 | er_obs **26.66** vs null median 39.95 | full_rank_like（分级集中，未过预注册 0.5x 阈值） |
| R4 平面对齐 | cos(PC1/PC2, lang_dir) = −0.088/−0.148；分段 top-2 主夹角 **83–89°** | 切平面逐段近正交——**旋转平面本身随深度旋转** |

### 解读（三个实质发现，重复三遍）

1. **无全局旋转平面**：方向旋转统计上集中于随机水平之上
   （0.165 vs 0.108），但 top-2 只承载 16% 方差，且 early/mid/deep
   三段切子空间近互正交（主夹角 83–89°）——旋转平面是**分段局部**
   的，不是贯穿深度的固定平面；Unified Theory 旋转平面假说在
   GLM4 语言轴上需修正为"平面序列"。
2. **方向流形非低维**：有效秩 26.7/40，未过预注册低维阈值——
   语言方向沿深度扫过 ~2/3 满秩的子空间，"单轴/单平面"语义
   编码在该轴上不成立。
3. **dnorm(0)=0 精确为零**：GLM4 的类均值语言分离在 embedding
   处**从零开始**、完全由层构建（与 li=1 处逐句探针 1.0 是两个
   层面的事实：句级身份存在于 embed，类级分离由层写入）——与
   2892 L1=deep_write_dominant 一致。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2894/rotation_structure_glm4/execution.json | bb0e7519 |
| phase2894/rotation_structure_glm4/result.json | c5bb65ed |
| phase2894/rotation_structure_glm4/rotation_structure_glm4.npz | fef03893 |
| tests/glm5/phase2894_rotation_structure_glm4.py | 9188fe6b |

Ledger 更新：B_rotation_struct_glm4 + M2894_rotation_structure_glm4
+ G_rotation_struct_glm4；blocks 19 / measurements 33 / growth 19。

### 接续（2895 候选）

- **A（主选）**：qwen3-4b 同协议旋转结构（对照：qwen 语言 mlp
  载体阳性 0.7719 且方向稳定——若 qwen 旋转弱则"旋转 vs 载体"
  建立跨模型反相关，零前向）
- **B**：norm-matched 对照纯化 2892 L1 归因（零前向）
- **C**：Gemma4/Qwen2-7B 下载决策（待用户确认）

*(SHA 与判决以 result.json 为准；本节由 phase2894 收尾脚本追加。)*
"""

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(memo_section + '\n')

print('ledger: blocks=%d meas=%d growth=%d'
      % (len(d2['blocks']), len(d2['measurements']),
         len(d2['growth_curve'])))
print('memo appended')
