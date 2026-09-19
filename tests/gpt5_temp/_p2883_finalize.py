# -*- coding: utf-8 -*-
"""Phase 2883 finalize: post-hoc per-class + SHA + Ledger v2 + MEMO."""
import hashlib
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(ROOT, r'tests\gpt5_temp\_p2883_finalize.txt')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
BASE = os.path.join(ROOT, r'tests\glm5\result'
                    r'\rdc_query_construction_20260913')
PDIR = os.path.join(BASE, 'phase2883', 'ds7b_class_mlp')

g = io.open(OUT, 'w', encoding='utf-8')


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


files = {
    'exec': os.path.join(PDIR, 'execution.json'),
    'result': os.path.join(PDIR, 'result.json'),
    'npz': os.path.join(PDIR, 'ds7b_class_mlp.npz'),
    'script': os.path.join(ROOT, r'tests\glm5\phase2883_ds7b_class_mlp.py'),
}
shas = {k: sha8(p) for k, p in files.items()}
for k in sorted(shas):
    g.write('SHA %s %s\n' % (k, shas[k]))

# ---------- post-hoc per-class acc from npz (descriptive) ----------
z = np.load(files['npz'], allow_pickle=True)
B3 = z['B3'].astype(np.float64)
labels = z['labels']
tl = [json.loads(str(t)) for t in z['target_list']]
C = B3 @ B3.T


def loo(Cm_, lab):
    Cm_ = Cm_.copy()
    np.fill_diagonal(Cm_, -2.0)
    return float(np.mean([lab[int(np.argmax(Cm_[i]))] == lab[i]
                          for i in range(len(lab))]))


CAT_WORDS = ['fruit', 'animal', 'metal', 'vehicle', 'country', 'food',
             'nature', 'furniture', 'tool', 'clothing']
per_class = {}
for ci, c in enumerate(CAT_WORDS):
    m = labels == ci
    per_class[c] = round(loo(C[np.ix_(m, m)], labels[m]), 4)
g.write('per_class %s\n' % json.dumps(per_class))
with io.open(os.path.join(PDIR, 'per_class_posthoc.txt'), 'w',
             encoding='utf-8') as f:
    f.write('post-hoc descriptive (not gated)\n')
    f.write(json.dumps(per_class, indent=2) + '\n')

# ---------- ledger v2 update ----------
from rdc_atlas_ledger import AtlasLedger
led = AtlasLedger.load(verify_sha=True)
doc = led.doc
assert doc['version'] == 2

MODEL_NAME = 'deepseek-r1-distill-qwen-7b'
if not any(b['block_id'] == 'B3_mlp_ds7b' for b in doc['blocks']):
    doc['blocks'].append({
        'block_id': 'B3_mlp_ds7b', 'axis_id': 'class',
        'kind': 'mlp_response', 'model': MODEL_NAME,
        'shape': [80, 8],
        'src': {'path': 'phase2883/ds7b_class_mlp/ds7b_class_mlp.npz',
                'sha256_8': shas['npz'], 'phase': 2883, 'key': 'B3'},
        'notes': 'DS7B (28L) window 20-27 (scaled from qwen3-4b 26-35); '
                 'same-context g_direct, no patch (D1)'})
if not any(m['meas_id'] == 'M2883_ds7b_carrier'
           for m in doc['measurements']):
    doc['measurements'].append({
        'meas_id': 'M2883_ds7b_carrier', 'type': 'cross_model_carrier',
        'model': MODEL_NAME,
        'verdict': 'V1=true(10/10 classes, 80 words)/W3=true(v1=v2=0.0)/'
                   'W1=mlp_carries_class_axis_ds7b(acc 0.575 vs null p95 '
                   '0.1625, 3.5x)/W2=carrier_deviation(delta -0.30 vs '
                   'qwen3-4b 0.875, outside +/-0.15 band)',
        'source': {'path': 'phase2883/ds7b_class_mlp/result.json',
                   'sha256_8': shas['result'], 'phase': 2883}})
if not any(r['point_id'] == 'G_ds7b_class_mlp'
           for r in doc['growth_curve']):
    doc['growth_curve'].append({
        'point_id': 'G_ds7b_class_mlp', 'axis_id': 'class',
        'model': MODEL_NAME, 'block': 'B3_mlp_ds7b',
        'components': 8, 'acc': 0.575, 'phase': 2883,
        'notes': 'first cross-model growth point; carrier law holds '
                 'qualitatively (3.5x null) but magnitude lower than '
                 'qwen3-4b 0.875 (8-layer vs 10-layer window, distilled '
                 'reasoning model)'})
if not any(l['link_id'] == 'L8_cross_model_carrier'
           for l in doc['linkage']):
    doc['linkage'].append({
        'link_id': 'L8_cross_model_carrier',
        'from': {'axis': 'class', 'model': 'qwen3-4b',
                 'block': 'B3_mlp'},
        'to': {'axis': 'class', 'model': MODEL_NAME,
               'block': 'B3_mlp_ds7b'},
        'evidence': 'same 2806 vocab + scaled window: DS7B W1 acc 0.575 '
                    '(3.5x null p95 0.1625), W3 rebuild exact (v1=v2=0); '
                    'carrier law replicates qualitatively, magnitude '
                    'model-dependent (-0.30 deviation)',
        'phase': 2883, 'status': 'confirmed'})
led.save()

rep = []
led2 = AtlasLedger.load(verify_sha=True)
stale2 = led2.verify(rep)
g.write('ledger v%s: %d blocks / %d measurements / %d growth / '
        '%d linkage / stale=%d\n'
        % (led2.doc['version'], len(led2.doc['blocks']),
           len(led2.doc['measurements']), len(led2.doc['growth_curve']),
           len(led2.doc['linkage']), len(stale2)))

# ---------- MEMO append ----------
before = sum(1 for _ in io.open(MEMO, encoding='utf-8'))
section = u"""

---

## Phase 2883：DS7B class 轴 mlp 载体冷启动（P3 跨模型复制第一站）

**日期**：2026-09-18。**模型**：deepseek-r1-distill-qwen-7b（Qwen2 架构，28 层
× 28 头 × head_dim 128，hidden 3584，untied lm_head，vocab 152064）。

### 原理

把 2861(g_direct)+2867(B3) 协议 verbatim 移植到第二模型，检验 mlp 载体定律的
模型普适性。预注册偏差（冻结）：D1 同上下文单臂（2861 的 L13H30 patch 机制仅
服务残差检查，B3 不需要，且为 qwen3 专属模块手术）；D2 窗口冻结缩放规则
WIN=[floor(26/36·L), L)——qwen3-4b 36 层 → [26,36) 10 层（逐字 2861），
DS7B 28 层 → **[20,28) 8 层**；D3 同一 2806 类词表按各模型 tokenizer 单 token
过滤；D4 E 行取 lm_head.weight（DS7B untied，与 2861 语义一致）。

### 门线（execution.json 先于任何观测冻结）

V1 词表 ≥8/10 类保 5 词且总 ≥50；W3 重构一致性 v1=|LN2(ln2in)−mlpin|<0.05、
v2=|mlp(ln2in)−mlpout|<0.05；W1 检索 LOO-NN acc > null p95（200 置换，
SEED=2883）；W2 描述性复制带 |acc−0.875|≤0.15。

### 结果

**mlp_carrier_replicates=True（V1/W3/W1 全过）**：

- V1=true：10/10 类各保 8 词（80 词，Qwen2 tokenizer 与 Qwen3 高度兼容）
- W3=true：v1=v2=**精确 0.0**（重构一致性无任何 bf16 漂移）
- **W1=true：acc 0.5750 vs null p95 0.1625（3.5×）→ mlp_carries_class_axis_ds7b**
- **W2=carrier_deviation：Δ=−0.30（0.575 vs qwen3-4b 0.875），超出 ±0.15 带**
- 层剖面（mean g_direct，L20-27）：−0.32/−0.32/−0.18/−0.02/+0.20/+0.41/
  −0.45/−0.38——符号跨层翻转，结构性强；g_direct 幅值（至 0.45）≫ g_comp
  （至 0.057），直写臂主导
- 运行时 100.6s（零前向词表 + 前向仅 2 token 序列 × 80 词）

### 解读

1. **载体定律跨模型定性成立**：DS7B 的 mlp 直写响应谱同样承载类身份
  （3.5× null），第二次独立确认"mlp 臂 = 高密度机制载体"。
2. **载体强度是模型依赖量**：−0.30 偏差为真偏差而非噪声（null p95 仅 0.16）。
   候选解释（观测后生成，待检验）：① 8 层窗口 vs 10 层（分辨率差）；②
   reasoning-distill 训练改变 mlp 编码密度；③ untied lm_head 使 E 行几何不同。
   判别实验 = 2884 候选 B（qwen3-4b 在 [26,34) 8 层窗口重算 B3——若 acc 降至
   ~0.58 则窗口长度主因）。
3. 跨模型增长率曲线开通：qwen3-4b class 0.875 / DS7B class 0.575（组件
   10 vs 8，不可直接比较，已在 Ledger 注记）。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2883/ds7b_class_mlp/execution.json | %(exec)s |
| phase2883/ds7b_class_mlp/result.json | %(result)s |
| phase2883/ds7b_class_mlp/ds7b_class_mlp.npz | %(npz)s |
| tests/glm5/phase2883_ds7b_class_mlp.py | %(script)s |

硬伤：lm_head 被 accelerate offload 成 meta tensor（7B > 12GiB 显存帽）——改
safetensors 直读分片（零前向，语义不变）；result.json 漏装 per-class 描述统计，
已由 npz post-hoc 补齐（per_class_posthoc.txt，非门线）。

### 接续（2884 候选）

- **A（主选）**：窗口长度判别——qwen3-4b B3 在 8 层窗口 [26,34) 重算（零新
  前向？否，需重测 g_direct；~15min）：判定 −0.30 偏差归因
- B：DS7B census（2875/2880 协议移植，drop 谱 + attr 轴）——补跨模型第二象限
- C：GLM4 冷启动（glm4-9b-chat-hf，32 层 → 窗口 [23,32)）

*（SHA 与判决以 result.json 为准；本节由 phase2883 收尾脚本追加。）*
""" % shas

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(section)
after = sum(1 for _ in io.open(MEMO, encoding='utf-8'))
g.write('MEMO %d -> %d\n' % (before, after))
g.close()
print('ok')
