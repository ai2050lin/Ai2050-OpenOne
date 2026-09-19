# -*- coding: utf-8 -*-
"""Phase 2896 closing: ledger (M2896 + growth + L13 refinement)
+ MEMO append."""
import hashlib
import io
import json
import os

LEDGER = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase2896'
       r'\qwen_readout_type')


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:8]


sha_exec = sha8(os.path.join(OUT, 'execution.json'))
sha_res = sha8(os.path.join(OUT, 'result.json'))
sha_npz = sha8(os.path.join(OUT, 'qwen_readout_type.npz'))
assert sha_exec == '62a342f3' and sha_res == '2468f496' \
    and sha_npz == '71c5fd16', 'sha drift'

d = json.load(io.open(LEDGER, encoding='utf-8'))
ids_blocks = [b['block_id'] for b in d['blocks']]
ids_meas = [m['meas_id'] for m in d['measurements']]
ids_growth = [g.get('growth_id', '') for g in d['growth_curve']]

if 'B_readout_qwen' not in ids_blocks:
    d['blocks'].append({
        'block_id': 'B_readout_qwen', 'axis_id': 'language',
        'kind': 'triple_condition_injection', 'model': 'qwen3-4b',
        'shape': [57, 10],
        'src': {'path': 'phase2896/qwen_readout_type/'
                        'qwen_readout_type.npz',
                'sha256_8': sha_npz, 'phase': 2896, 'key': 'B_stale'},
        'notes': 'W=[26,36); stale(d18)/eigen(d(li))/orth(d_orth(li)) '
                 'triple injection, 2887 vocab/conds verbatim; all '
                 'three A-positive'})

if 'M2896_qwen_readout_type' not in ids_meas:
    d['measurements'].append({
        'meas_id': 'M2896_qwen_readout_type',
        'type': 'channel_readout_discrimination', 'model':
        'qwen3-4b',
        'verdict': 'subspace_readout_tolerant: all three injection '
                   'conditions A-positive (stale acc 0.8246 / margin '
                   '0.2216 vs p95 0.0292; eigen 0.8421 / 0.1799; '
                   'orth 0.8246 / 0.1831; concept controls 0.000-'
                   '0.070 ~null). qwen deep mlp responds to the '
                   'stale li=18 direction AND per-layer eigen-'
                   'directions AND the rotation-orthogonal '
                   'components (cos(d(li),d18) 0.456->0.068) - '
                   'readout is subspace-tolerant, not direction-'
                   'matched. Cross-model closure: glm4 needs layer-'
                   'matched directions (M2893 positive, stale '
                   'negatives M2890/2891/2892) => carrier asymmetry '
                   '= readout tolerance (qwen) vs direction '
                   'matching (glm4), dynamics universal (M2895). '
                   'v1 recompute 0.0 / hook-vs-call 0.0 (shape-'
                   'matched capture); amendment: run 1 void per '
                   'frozen v1 gate (shape-mismatched hook check, '
                   'bf16 cross-shape kernel err 1.6e-3 = 2877 '
                   'lesson; no verdict used from run 1)',
        'source': {'path': 'phase2896/qwen_readout_type/result.json',
                   'sha256_8': sha_res, 'phase': 2896}})

if 'G_readout_types' not in ids_growth:
    d['growth_curve'].append({
        'growth_id': 'G_readout_types', 'phase': 2896,
        'model': 'qwen3-4b vs glm4-9b-chat-hf', 'axis': 'language',
        'finding': 'two channel-readout types discovered: qwen mlp '
                   'is SUBSPACE-TOLERANT (responds to stale, eigen, '
                   'and orthogonal directions alike, all acc >= '
                   '0.82) while glm4 is DIRECTION-MATCHED (only '
                   'layer-matched eigen-directions work); direction '
                   'rotation dynamics universal (M2895) - the '
                   'carrier-law per-model variation is a readout '
                   'property'})

# refine L13 notes (append only if not already noted)
for l in d['linkage']:
    if l.get('link_id') == 'L13_rotation_vs_carrier_refuted':
        if 'M2896' not in l.get('notes', ''):
            l['notes'] = (l.get('notes', '') + ' | refined by M2896: '
                          'readout types identified - qwen subspace-'
                          'tolerant (stale/eigen/orth all positive '
                          '0.82-0.84), glm4 direction-matched; '
                          'asymmetry is a channel property')

with io.open(LEDGER, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
d2 = json.load(io.open(LEDGER, encoding='utf-8'))
assert len(d2['blocks']) == 21 and len(d2['measurements']) == 35 \
    and len(d2['growth_curve']) == 21, 'ledger count mismatch'

memo_section = """
### Phase 2896 - qwen 通道读出类型判别（subspace_readout_tolerant）

**日期**：2026-09-19。**模型**：qwen3-4b。运行 20.1s（零前向方向 +
窗口 [26,36) 单前向，2887 词表/条件 verbatim 57 词）。

### 原理与设计

L13（2895）留问：qwen 载体对陈旧方向阳性而 GLM4 阴性——读出性质
是什么？三条件逐层注入判别（方向全部零前向提取）：

- stale：注入 d18（li=18 陈旧方向），响应投影 d18
- eigen：注入 d(li)（本层本征方向，2893 qwen 对应），投影 d(li)
- orth：注入 d_orth(li) = unit(d(li) − (d(li)·d18)d18)
  （旋转正交分量；cos(d(li),d18) 在窗口内 0.456 -> 0.068），
  投影 d_orth(li)

B = g(same) − 0.5(g(func)+g(null))；loo-NN acc vs null p95
（200 perms, SEED=2896）；概念对照。v1：重算 + hook-vs-call。

### 执行记录（预注册纪律）

Run 1 按冻结 v1 门线判 void（hook-vs-call 1.6e-3 > 1e-4）：
根因是形状错配 hook 检查（1-D [2560] 重算 vs [1,2,2560] 真实前向，
bf16 跨形状核差异 ~2e-3 = 2877 教训；2887 实际门线只有 recompute，
hook 引用系误读先例）。修正：形状匹配捕获（2890–2893 约定），
修正案入 execution.json，删旧产物重跑——run 1 判决未被使用。
Run 2：v1 recompute **0.0** / hook-vs-call **0.0**。

### 判决：subspace_readout_tolerant（三条件全阳性）

| 条件 | acc vs null p95 | margin vs p95 | 概念对照 |
|---|---|---|---|
| stale（d18 陈旧方向） | **0.8246 > 0.6325** | **0.2216 > 0.0292** | 0.0000 |
| eigen（本层本征方向） | **0.8421 > 0.6491** | **0.1799 > 0.0400** | 0.0702 |
| orth（旋转正交分量） | **0.8246 > 0.6316** | **0.1831 > 0.0413** | 0.0526 |

### 解读（三个实质发现，重复三遍）

1. **qwen 读出是子空间容忍型**：陈旧方向、本层方向、乃至与陈旧
   方向近正交的旋转分量（cos 0.14–0.46）三者在 qwen 深层 mlp 中
   全部承载完整检索信号（acc 0.82–0.84）——响应不挑方向，挑的是
   语言子空间。
2. **载体不对称闭环**：qwen = 子空间容忍读出，GLM4 = 方向匹配读出
   （2893 逐层阳性 + 2890/2891/2892 陈旧方向阴性）；方向旋转动力学
   跨模型普适（2895）——载体定律的 per-model 变异定位于**读出端**，
   G_readout_types 登记。
3. **旋转子空间携带信息**：orth 分量单独即可检索（0.8246）——
   方向旋转不是噪声，旋转扫过的子空间本身携带语言信息；与 2894
   "无全局平面、分段局部旋转"拼合：语言信息在一段局部旋转平面
   序列上分布，qwen 通道对整段子空间开放。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2896/qwen_readout_type/execution.json | 62a342f3 |
| phase2896/qwen_readout_type/result.json | 2468f496 |
| phase2896/qwen_readout_type/qwen_readout_type.npz | 71c5fd16 |
| tests/glm5/phase2896_qwen_readout_type.py | 965dbfd6 |

Ledger 更新：B_readout_qwen + M2896_qwen_readout_type +
G_readout_types + L13 notes 精化；blocks 21 / measurements 35 /
growth 21。

### 接续（2897 候选）

- **A（主选）**：GLM4 读出类型确证——GLM4 orth 分量注入（2893
  协议加 orth 条件；若 orth 阴性则"方向匹配"定性坐实，若阳性则
  GLM4 也是部分容忍，谱系化）
- **B**：norm-matched 对照纯化 2892 L1 归因（零前向）
- **C**：Gemma4/Qwen2-7B 下载决策（待用户确认）

*(SHA 与判决以 result.json 为准；本节由 phase2896 收尾脚本追加。)*
"""

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(memo_section + '\n')

print('ledger: blocks=%d meas=%d growth=%d'
      % (len(d2['blocks']), len(d2['measurements']),
         len(d2['growth_curve'])))
print('memo appended')
