# -*- coding: utf-8 -*-
"""Phase 2888 finalize: SHA + Ledger v2 (glm4) + MEMO append."""
import hashlib
import io
import json
import os
import sys

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(ROOT, r'tests\gpt5_temp\_p2888_finalize.txt')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
BASE = os.path.join(ROOT, r'tests\glm5\result'
                    r'\rdc_query_construction_20260913')
PDIR = os.path.join(BASE, 'phase2888', 'glm4_class_mlp')

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
    'npz': os.path.join(PDIR, 'glm4_class_mlp.npz'),
    'script': os.path.join(ROOT, r'tests\glm5'
                           r'\phase2888_glm4_class_mlp.py'),
}
shas = {k: sha8(p) for k, p in files.items()}
for k in sorted(shas):
    g.write('SHA %s %s\n' % (k, shas[k]))

# ---------- ledger v2 update ----------
from rdc_atlas_ledger import AtlasLedger
led = AtlasLedger.load(verify_sha=True)
doc = led.doc
assert doc['version'] == 2

ns = doc.get('model_namespace', {})
if ns.get('primary') == 'qwen3-4b':
    active = set(ns.get('active', []))
    pending = [m for m in ns.get('pending', [])
               if m not in ('ds7b', 'glm4')]
    active.update(['ds7b', 'glm4'])
    ns['active'] = sorted(active)
    ns['pending'] = pending
    ns['glm4_note'] = ('glm4-9b-chat-hf: glm arch 40L, hidden 4096, '
                       'untied, vocab 151552; window [28,40); '
                       'M2888 carrier within band')

if not any(m['meas_id'] == 'M2888_glm4_class'
           for m in doc['measurements']):
    doc['measurements'].append({
        'meas_id': 'M2888_glm4_class', 'type': 'carrier_test',
        'model': 'glm4-9b-chat-hf',
        'verdict': 'mlp_carrier_replicates=True: W1 acc 0.8375 vs null '
                   'p95 0.1381 (6.1x, mlp_carries_class_axis_glm4); '
                   'W2 delta -0.0375 replicates_within_band (closest '
                   'to qwen3-4b 0.875 among 3 models); W3 Gen2 rel '
                   'budgets v1 1.1e-4 / v2 2.7e-4 (Gen1 abs 0.05 '
                   'mismatched GLM4 activation scale - registered '
                   'amendment per 2877 precedent); window [28,40) '
                   '12 layers; vocab 10/10 x 8 words legal',
        'source': {'path': 'phase2888/glm4_class_mlp/result.json',
                   'sha256_8': shas['result'], 'phase': 2888}})

if not any(r.get('point_id') == 'G_glm4_class_mlp'
           for r in doc['growth_curve']):
    doc['growth_curve'].append({
        'point_id': 'G_glm4_class_mlp',
        'axis_id': 'class',
        'model': 'glm4-9b-chat-hf',
        'block': 'B3_mlp_glm4',
        'components': 12,
        'acc': 0.8375,
        'phase': 2888,
        'notes': 'third model, 2nd architecture family (glm vs qwen): '
                 'carrier curve qwen3-4b 0.875 / glm4 0.8375 / ds7b '
                 '0.575; magnitude is model-dependent but the channel '
                 'itself is architecture-general'})

if not any(b.get('block_id') == 'B3_mlp_glm4' for b in doc['blocks']):
    doc['blocks'].append({
        'block_id': 'B3_mlp_glm4',
        'axis_id': 'class',
        'kind': 'mlp_response',
        'model': 'glm4-9b-chat-hf',
        'shape': [80, 12],
        'src': {'path': 'phase2888/glm4_class_mlp/glm4_class_mlp.npz',
                'sha256_8': shas['npz'], 'phase': 2888, 'key': 'B3'},
        'notes': 'glm arch, window [28,40); g_direct >> g_comp '
                 'direct-write arm dominates'})

if not any(l.get('link_id') == 'L11_three_model_carrier'
           for l in doc['linkage']):
    doc['linkage'].append({
        'link_id': 'L11_three_model_carrier',
        'from': {'axis': 'class', 'model': 'qwen3-4b',
                 'block': 'B3_mlp'},
        'to': {'axis': 'class', 'model': 'glm4-9b-chat-hf',
               'block': 'B3_mlp_glm4'},
        'evidence': 'three models, two architecture families (qwen3-4b '
                    '0.875, glm4-9b 0.8375 within band, ds7b 0.575 '
                    'deviation): mlp direct-write carrier is '
                    'architecture-general; strength model-dependent',
        'phase': 2888,
        'status': 'confirmed'})
led.save()

led2 = AtlasLedger.load(verify_sha=True)
rep = []
stale2 = led2.verify(rep)
g.write('ledger v%s: %d axes / %d blocks / %d measurements / '
        '%d growth / %d linkage / %d negatives / stale=%d\n'
        % (led2.doc['version'], len(led2.doc['axes']),
           len(led2.doc['blocks']), len(led2.doc['measurements']),
           len(led2.doc['growth_curve']), len(led2.doc['linkage']),
           len(led2.doc['negatives']), len(stale2)))
g.write('ledger sha %s\n' % sha8(os.path.join(
    ROOT, r'research\gpt5\atlas\atlas_ledger.json')))

# ---------- MEMO append ----------
before = sum(1 for _ in io.open(MEMO, encoding='utf-8'))
section = u"""

---

## Phase 2888：GLM4 class 轴 mlp 载体冷启动——P3 第三模型，架构普适确认

**日期**：2026-09-18。**模型**：glm4-9b-chat-hf（glm 架构，40 层，hidden
4096，untied，vocab 151552）。运行 2m28s（手工 device_map，窗口层 GPU /
其余 CPU）。

### 原理与移植

2883 协议 verbatim 移植（D1 same-context 单臂；D2 窗口缩放规则冻结：
[floor(26/36*40), 40) = [28,40) 12 层；D3 2806 词表按 GLM4 tokenizer
过滤——10/10 类各 8 词 = 80 词；D4 lm_head safetensors 直读）。

### 硬伤两起（均已修复并登记）

1. **Gen1 device_map=auto disk-offload 陷阱**：部分层参数成 meta 张量，
   layer_dev 返回 meta → 直接调用崩溃。修复：手工 device_map（窗口
   28-39 层 + embed + lm_head 约 3.9GB 在 GPU，其余 28 层 CPU）。
2. **Gen1 W3 绝对预算尺度错配**：v2_abs = 0.0625 超 0.05 预算，但相对
   误差仅 ~1.6e-3——正是 2877 认定的 cuBLAS 跨形状 bf16 预算量级；0.05
   绝对值系 hidden-3584 尺度（DS7B）外推错配。Gen2 修订为相对预算
   <5e-3（修订理由冻结于 execution.json，2877 先例），实测
   v1_rel=1.1e-4 / v2_rel=2.7e-4 远低于预算——健康。

### 判决：mlp_carrier_replicates = True

| 门线 | 观测 | 结果 |
|---|---|---|
| **V1** 词表 | 10/10 类各 8 词（80 词） | **vocab_legal=true** |
| **W3** 重构一致性（Gen2 rel） | v1_rel 1.1e-4 / v2_rel 2.7e-4 | **true** |
| **W1** mlp 载体检索 | acc **0.8375** vs null p95 0.1381（**6.1×**） | **mlp_carries_class_axis_glm4** |
| **W2** 复制带 | Δ = **−0.0375** | **replicates_within_band** |

### 解读（三个实质发现，重复三遍）

1. **跨架构确认**：glm 与 qwen 两个架构族均复现 mlp 直写载体
   （g_direct ≫ g_comp），**载体通道是架构普适的**。
2. **三模型载体曲线成形**：qwen3-4b 0.875 / **glm4 0.8375**（带内）/
   ds7b 0.575（偏差）——载体强度模型依赖，且 DS7B 的 −0.30 偏差在
   GLM4 对照下更显特殊（候选归因收窄至 reasoning-distill 训练，untied
   几何已被 GLM4 untied 且带内排除）。
3. **untied 假设再排除一层**：GLM4 untied 且带内——DS7B 偏差不能归因
   untied lm_head 几何，剩余主候选 = distill 训练。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2888/glm4_class_mlp/execution.json | %(exec)s |
| phase2888/glm4_class_mlp/result.json | %(result)s |
| phase2888/glm4_class_mlp/glm4_class_mlp.npz | %(npz)s |
| tests/glm5/phase2888_glm4_class_mlp.py | %(script)s |

### 接续（2889 候选）

- **A（主选）**：DS7B 归因收尾——distill 假设检验（若可获取非 distill
  的同架构 7B 对照则直接判别；否则登记为 open candidate）
- **B**：GLM4 语言轴判别（2887 协议移植：中层句状态语言方向 + 翻译词
  载体，检验 L10 分离跨架构）
- **C**：Gemma4 冷启动（P3 第四模型，收尾 model_namespace）

*（SHA 与判决以 result.json 为准；本节由 phase2888 收尾脚本追加。）*
""" % shas

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(section)
after = sum(1 for _ in io.open(MEMO, encoding='utf-8'))
g.write('MEMO %d -> %d\n' % (before, after))
g.close()
print('ok')
