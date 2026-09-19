# -*- coding: utf-8 -*-
"""Phase 2889 closing: ledger update (4 entries + namespace) + MEMO append."""
import hashlib
import io
import json
import os

LEDGER = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2889', 'qwen25_distill_test')


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:8]


sha_exec = sha8(os.path.join(OUT, 'execution.json'))
sha_res = sha8(os.path.join(OUT, 'result.json'))
sha_npz = sha8(os.path.join(OUT, 'qwen25_distill_test.npz'))
sha_scr = sha8(r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
               r'\phase2889_qwen25_distill_test.py')
assert sha_exec == 'f7c78b49' and sha_res == 'b00c36a0' \
    and sha_npz == '07bec8a4', 'sha drift: %s %s %s' % (sha_exec, sha_res,
                                                        sha_npz)

d = json.load(io.open(LEDGER, encoding='utf-8'))
ids_blocks = [b['block_id'] for b in d['blocks']]
ids_meas = [m['meas_id'] for m in d['measurements']]
ids_growth = [g['point_id'] for g in d['growth_curve']]
ids_link = [l['link_id'] for l in d['linkage']]

if 'B3_mlp_qwen25_3b' not in ids_blocks:
    d['blocks'].append({
        'block_id': 'B3_mlp_qwen25_3b', 'axis_id': 'class',
        'kind': 'mlp_response', 'model': 'qwen2.5-3b-instruct',
        'shape': [80, 10],
        'src': {'path': 'phase2889/qwen25_distill_test/'
                        'qwen25_distill_test.npz',
                'sha256_8': sha_npz, 'phase': 2889, 'key': 'B3'},
        'notes': 'qwen2 arch NON-distill control (distill-hypothesis '
                 'test), window [26,36); g_direct >> g_comp direct-write '
                 'arm dominates; E rows from tied embed_tokens'})
if 'M2889_qwen25_distill' not in ids_meas:
    d['measurements'].append({
        'meas_id': 'M2889_qwen25_distill', 'type': 'attribution_test',
        'model': 'qwen2.5-3b-instruct',
        'verdict': 'mlp_carrier_replicates=True: W1 acc 0.8375 vs null '
                   'p95 0.1500 (5.6x, mlp_carries_class_axis_qwen25_3b); '
                   'W2 delta -0.0375 replicates_within_band; W3 Gen2 rel '
                   'v1 1.6e-4 / v2 2.0e-4; W4 '
                   'distill_hypothesis_strengthened - non-distill '
                   'qwen2-arch control in band exonerates the qwen2 '
                   'architecture at 3B and the size direction is '
                   'reverse (3B < 7B yet in band); DS7B -0.30 deviation '
                   'narrowed to reasoning-distill training; residual '
                   'confound: no 7B non-distill qwen2 control available '
                   'locally (strengthened, not settled)',
        'source': {'path': 'phase2889/qwen25_distill_test/result.json',
                   'sha256_8': sha_res, 'phase': 2889}})
if 'G_qwen25_distill_test' not in ids_growth:
    d['growth_curve'].append({
        'point_id': 'G_qwen25_distill_test', 'axis_id': 'class',
        'model': 'qwen2.5-3b-instruct', 'block': 'B3_mlp_qwen25_3b',
        'components': 10, 'acc': 0.8375, 'phase': 2889,
        'notes': '4th model, 3rd data point on the qwen2 arch family: '
                 'carrier curve qwen3-4b 0.875 / glm4 0.8375 / '
                 'qwen2.5-3b 0.8375 / ds7b 0.575; the channel is '
                 'architecture-general, strength model-dependent'})
if 'L12_distill_attribution' not in ids_link:
    d['linkage'].append({
        'link_id': 'L12_distill_attribution',
        'from': {'axis': 'class',
                 'model': 'deepseek-r1-distill-qwen-7b',
                 'block': 'B3_mlp_ds7b'},
        'to': {'axis': 'class', 'model': 'qwen2.5-3b-instruct',
               'block': 'B3_mlp_qwen25_3b'},
        'evidence': 'same qwen2 architecture family, non-distill '
                    'instruct training, smaller size (3B): carrier in '
                    'band (0.8375 vs anchor 0.875, delta -0.0375; '
                    'delta vs ds7b +0.2625) -> architecture and small '
                    'size exonerated; DS7B deviation attributed to '
                    'reasoning-distill training (7B non-distill control '
                    'unavailable -> strengthened not settled)',
        'phase': 2889, 'status': 'confirmed'})
ns = d['model_namespace']
if 'qwen25_3b' not in ns.get('active', []):
    ns['active'].append('qwen25_3b')
ns['qwen25_note'] = ('qwen2.5-3b-instruct: qwen2 arch 36L, hidden '
                     '2048, tied, vocab 151936; window [26,36); M2889 '
                     'carrier within band (0.8375), distill-hypothesis '
                     'control for ds7b attribution')

with io.open(LEDGER, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)

# round-trip validation
d2 = json.load(io.open(LEDGER, encoding='utf-8'))
assert len(d2['blocks']) == 14 and len(d2['measurements']) == 28 \
    and len(d2['growth_curve']) == 15 and len(d2['linkage']) == 12, \
    'ledger count mismatch'

memo_section = """
### Phase 2889 - DS7B distill 假设检验（qwen2 架构非 distill 对照，候选 A 判别）

**日期**：2026-09-18。**模型**：qwen2.5-3b-instruct（qwen2 架构，36 层，
hidden 2048，tied，vocab 151936）。运行约 1m4s（手工 device_map，窗口
26-35 层 GPU / 其余 26 层 CPU，2888 Gen2 先例）。

### 原理与判别设计

2888 将 DS7B 的 -0.30 载体偏差（0.575 vs 0.875 锚）收窄至
reasoning-distill 训练（窗口/位置 2884 排除、untied 几何 2885+2888
排除）。本机模型目录探针发现 **qwen2 架构非 distill 对照**：
qwen2.5-3b-instruct（同 Qwen2ForCausalLM 架构族、官方 instruct 训练、
3B、tied）——候选 A 从"登记 open candidate"升级为**直接判别实验**。

判别逻辑（预注册冻结）：
- acc 带内（|acc-0.875|<=0.15）=> 架构+小尺寸在 3B 被豁免 =>
  distill_hypothesis_strengthened（残余混杂：3B-vs-7B 尺寸）
- acc 贴近 DS7B（|acc-0.575|<=0.15）=> 偏差跟随架构/尺寸 =>
  distill_hypothesis_weakened
- 否则 inconclusive

移植偏差（全部预冻结）：D4 tied=true 故 E 行取
model.embed_tokens.weight（2884 数学等价先例）；W3 Gen2 相对预算
<5e-3（hidden 2048 尺度）；窗口 [26,36) 10 层（与 qwen3-4b 同）。

### 判决：mlp_carrier_replicates = True；W4 = distill_hypothesis_strengthened

| 门线 | 观测 | 结果 |
|---|---|---|
| **V1** 词表 | 10/10 类各 8 词（80 词） | **vocab_legal=true** |
| **W3** 重构一致性（Gen2 rel） | v1_rel 1.56e-4 / v2_rel 1.98e-4 | **true** |
| **W1** mlp 载体检索 | acc **0.8375** vs null p95 0.1500（**5.6 倍**） | **mlp_carries_class_axis_qwen25_3b** |
| **W2** 复制带 | delta = **-0.0375** | **replicates_within_band** |
| **W4** distill 判别 | vs ds7b +0.2625 | **distill_hypothesis_strengthened** |

### 解读（三个实质发现，重复三遍）

1. **distill 假设增强**：非 distill qwen2 架构对照带内（0.8375）——
   架构在 3B 被豁免，且尺寸方向反向（3B < 7B 却带内，若尺寸致偏差应
   更差）——**DS7B 偏差的最简归因收窄至 reasoning-distill 训练本身**。
2. **载体曲线第四点**：qwen3-4b 0.875 / glm4 0.8375 /
   qwen2.5-3b 0.8375 / ds7b 0.575——四个模型、三个数据点两架构族+
   qwen2 族内对照，mlp 直写载体通道普适，强度模型依赖。
3. **残余混杂如实登记**：7B 级非 distill qwen2 对照本地不可得
   （Qwen2-7B 未下载）=> 归因为 **strengthened 非 settled**；若未来
   获取 Qwen2-7B(-Instruct) 可升级为 settled。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2889/qwen25_distill_test/execution.json | f7c78b49 |
| phase2889/qwen25_distill_test/result.json | b00c36a0 |
| phase2889/qwen25_distill_test/qwen25_distill_test.npz | 07bec8a4 |
| tests/glm5/phase2889_qwen25_distill_test.py | 3c0bdc64 |

Ledger 更新：B3_mlp_qwen25_3b + M2889_qwen25_distill +
G_qwen25_distill_test + L12_distill_attribution；model_namespace
active += qwen25_3b（blocks 14 / measurements 28 / growth 15 /
linkage 12）。

### 接续（2890 候选）

- **A（主选）**：GLM4 语言轴判别（2887 协议移植：中层句状态语言方向
  + 翻译词载体，检验 L10 语言/概念通道分离跨架构）
- **B**：Gemma4 冷启动（model_namespace 收尾，pending_replication
  最后一项）
- **C**：distill 假设升级 settled（需下载 Qwen2-7B-Instruct 对照；
  可选网络获取）

*（SHA 与判决以 result.json 为准；本节由 phase2889 收尾脚本追加。）*
"""

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(memo_section + '\n')

print('ledger: blocks=%d meas=%d growth=%d link=%d'
      % (len(d2['blocks']), len(d2['measurements']),
         len(d2['growth_curve']), len(d2['linkage'])))
print('memo appended')
