# -*- coding: utf-8 -*-
"""Phase 2891 closing: ledger update (negative + N7) + MEMO append."""
import hashlib
import io
import json
import os

LEDGER = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase2891\language_attn_glm4')


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:8]


sha_exec = sha8(os.path.join(OUT, 'execution.json'))
sha_res = sha8(os.path.join(OUT, 'result.json'))
sha_npz = sha8(os.path.join(OUT, 'language_attn_glm4.npz'))
assert sha_exec == '62c040d9' and sha_res == '8ee79eb3' \
    and sha_npz == '37415d8e', 'sha drift: %s %s %s' % (sha_exec, sha_res,
                                                        sha_npz)

d = json.load(io.open(LEDGER, encoding='utf-8'))
ids_blocks = [b['block_id'] for b in d['blocks']]
ids_meas = [m['meas_id'] for m in d['measurements']]
ids_growth = [g['point_id'] for g in d['growth_curve']]
ids_neg = [n['neg_id'] for n in d['negatives']]

if 'B_attn_language_glm4' not in ids_blocks:
    d['blocks'].append({
        'block_id': 'B_attn_language_glm4', 'axis_id': 'language',
        'kind': 'attn_response', 'model': 'glm4-9b-chat-hf',
        'shape': [78, 12],
        'src': {'path': 'phase2891/language_attn_glm4/'
                        'language_attn_glm4.npz',
                'sha256_8': sha_npz, 'phase': 2891, 'key': 'B3_attn'},
        'notes': 'NEGATIVE: g_attn spectrum injected with the same '
                 '2890 lang_dir (target pos 1), window [28,40); '
                 'direct-call path verified rel err 0.0 vs hook; '
                 'registered as measured deviation'})

if 'M2891_language_attn_glm4' not in ids_meas:
    d['measurements'].append({
        'meas_id': 'M2891_language_attn_glm4', 'type': 'carrier_test',
        'model': 'glm4-9b-chat-hf',
        'verdict': 'attn_language_route_glm4=False (NEGATIVE): v1 ok '
                   '(rel err 0.0, kernel-vs-hook noise 0.0); A1 acc '
                   '0.5897 < null p95 0.6410 (null mean 0.5279); A2 '
                   'margin -0.0011 < null p95 0.0340; A3 concept acc '
                   '0.0000 (neither channel content); AC descriptive: '
                   'attn 0.5897 vs own mlp 0.5256 (M2890) vs qwen mlp '
                   '0.7719 - both glm4 channels below null p95; '
                   'closes N6 open candidate: language does NOT route '
                   'via attention direct-write at the deep window '
                   'either',
        'source': {'path': 'phase2891/language_attn_glm4/result.json',
                   'sha256_8': sha_res, 'phase': 2891}})

if 'G_language_attn_glm4' not in ids_growth:
    d['growth_curve'].append({
        'point_id': 'G_language_attn_glm4', 'axis_id': 'language',
        'model': 'glm4-9b-chat-hf', 'block': 'B_attn_language_glm4',
        'components': 12, 'acc': 0.5897, 'phase': 2891,
        'notes': 'ATTENTION channel (negative, below null p95); '
                 'language carrier landscape: qwen mlp 0.7719 / '
                 'glm4 mlp 0.5256 / glm4 attn 0.5897 - glm4 carries '
                 'the direction in NEITHER deep-window channel'})

if 'N7_attention_route_glm4' not in ids_neg:
    d['negatives'].append({
        'neg_id': 'N7_attention_route_glm4',
        'claim': 'glm4 language direction routes through the '
                 'attention channel (direct-write) at the deep '
                 'window, closing the N6 open candidate',
        'evidence': 'M2891: A1 acc 0.5897 < null p95 0.6410; A2 '
                    'margin -0.0011 < null p95 0.0340; A3 concept '
                    '0.0000; v1 rel err 0.0, direct-call verified '
                    'against hooks (noise 0.0)',
        'status': 'settled',
        'implication': 'on glm4 the mid-stream language direction is '
                       'neither mlp- nor attention-written at the '
                       'deep window [28,40): language identity '
                       '(probe >=0.96 from li=1, D0 in M2890) is '
                       'embedding-lexical in origin and passively '
                       'preserved in the residual stream; the 2887 '
                       'mlp language carrier is qwen3-4b-specific - '
                       'the carrier law requires per-model channel '
                       'determination; remaining candidates: early '
                       'layers (< window) write the direction, or it '
                       'is never actively written after embeddings',
        'source': {'phase': 2891, 'meas_id': 'M2891_language_attn_glm4'}})

with io.open(LEDGER, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
d2 = json.load(io.open(LEDGER, encoding='utf-8'))
assert len(d2['blocks']) == 16 and len(d2['measurements']) == 30 \
    and len(d2['growth_curve']) == 17 and len(d2['negatives']) == 7, \
    'ledger count mismatch'

memo_section = """
### Phase 2891 - GLM4 语言 attention 通路判别（N6 收尾，第二阴性）

**日期**：2026-09-18。**模型**：glm4-9b-chat-hf。运行 356.8s。

### 原理与协议

2890 严格平行协议，注入点 mlp -> attention：g_attn[i,q] =
[(self_attn(attnin + eps*cdir) - self_attn(attnin)) . cdir]/eps，
注入仅目标位置（pos 1）；lang_dir 与 78 词表**零前向复用 2890 产物**
（同方向保证 mlp/attn 通道对比内部有效）。直接调用路径先经探针冻结：
self_attn(position_embeddings=model.model.rotary_emb(...),
attention_mask=None) 与 hook 基线 rel err = 0.0（probe_2891_attn）。
窗口 [28,40)，same/func/null 三条件，B3_attn =
g(same) - 0.5(g(func)+g(null))。

一次预注册前崩溃（attnin batch 维索引 bug，无统计量产生），按纪律
删旧 execution.json 后重跑。

### 判决：attn_language_route_glm4 = **False（第二阴性）**

| 门线 | 观测 | 结果 |
|---|---|---|
| v1 确定性 + 调用一致性 | rel err **0.0**；kernel-vs-hook **0.0** | 通过 |
| **A1** 语言检索 | acc **0.5897** < null p95 0.6410（null mean 0.5279） | **attn_language_signal_absent_glm4** |
| **A2** 同语言余量 | -0.0011 < null p95 0.0340 | **margin_absent_attn_glm4** |
| A3 概念对照 | **0.0000**（null p95 0.0641） | 通道两无 |
| AC 通路对比（描述性） | attn 0.5897 / mlp 0.5256 / qwen mlp 0.7719 | 双通道均低于 null p95 |

### 解读（三个实质发现，重复三遍）

1. **N6 候选关闭**：GLM4 语言方向在深层窗口既不经 mlp 直写
   （M2890）也**不经 attention 直写**（M2891）——两通道阴性。
2. **GLM4 语言身份是 embedding-词法起源 + 残差流被动保留**：探针
   li=1 起即 1.0（D0），深层窗口两模块均不主动写入该方向——2887
   语言 mlp 载体是 qwen3-4b 特异的。
3. **载体定律的适用条件收窄**：class/attr/syntax 族跨架构复制，但
   language 族不复制且通道无关——载体定律要求 per-model 通道判定；
   剩余候选：方向由窗口前早期层写入，或 embeddings 之后从未被主动
   写入。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2891/language_attn_glm4/execution.json | 62c040d9 |
| phase2891/language_attn_glm4/result.json | 8ee79eb3 |
| phase2891/language_attn_glm4/language_attn_glm4.npz | 37415d8e |
| tests/glm5/phase2891_language_attn_glm4.py | 6f16c144 |

Ledger 更新：B_attn_language_glm4 + M2891_language_attn_glm4 +
G_language_attn_glm4 + **N7_attention_route_glm4**（settled，关闭
N6 候选）；blocks 16 / measurements 30 / growth 17 / negatives 7。

### 接续（2892 候选）

- **A（主选）**：Gemma4 冷启动（model_namespace 最后一项 pending，
  class 轴 mlp 载体第四模型）
- **B**：GLM4 语言早期层写入检验（窗口 [13,27) 或全层 attn/mlp 响应
  谱，定位语言方向写入层；成本较高）
- **C**：distill 假设升级 settled（Qwen2-7B-Instruct 网络获取）

*（SHA 与判决以 result.json 为准；本节由 phase2891 收尾脚本追加。）*
"""

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(memo_section + '\n')

print('ledger: blocks=%d meas=%d growth=%d neg=%d'
      % (len(d2['blocks']), len(d2['measurements']),
         len(d2['growth_curve']), len(d2['negatives'])))
print('memo appended')
