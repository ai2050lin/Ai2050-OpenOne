# -*- coding: utf-8 -*-
"""Phase 2890 closing: ledger update (negative registration) + MEMO append."""
import hashlib
import io
import json
import os

LEDGER = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase2890\language_axis_glm4')


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:8]


sha_exec = sha8(os.path.join(OUT, 'execution.json'))
sha_res = sha8(os.path.join(OUT, 'result.json'))
sha_npz = sha8(os.path.join(OUT, 'language_axis_glm4.npz'))
assert sha_exec == '6defb457' and sha_res == 'eaaf6df3' \
    and sha_npz == '74933303', 'sha drift: %s %s %s' % (sha_exec, sha_res,
                                                        sha_npz)

d = json.load(io.open(LEDGER, encoding='utf-8'))
ids_blocks = [b['block_id'] for b in d['blocks']]
ids_meas = [m['meas_id'] for m in d['measurements']]
ids_growth = [g['point_id'] for g in d['growth_curve']]
ids_neg = [n['neg_id'] for n in d['negatives']]

if 'B3_language_glm4' not in ids_blocks:
    d['blocks'].append({
        'block_id': 'B3_language_glm4', 'axis_id': 'language',
        'kind': 'mlp_response', 'model': 'glm4-9b-chat-hf',
        'shape': [78, 12],
        'src': {'path': 'phase2890/language_axis_glm4/'
                        'language_axis_glm4.npz',
                'sha256_8': sha_npz, 'phase': 2890, 'key': 'B3_lang'},
        'notes': 'NEGATIVE: g spectrum injected with GLM4-own lang_dir '
                 '(li=20), window [28,40); E1 acc below null p95 - '
                 'registered as measured deviation, not a carrier'})

if 'M2890_language_axis_glm4' not in ids_meas:
    d['measurements'].append({
        'meas_id': 'M2890_language_axis_glm4', 'type': 'carrier_test',
        'model': 'glm4-9b-chat-hf',
        'verdict': 'language_axis_in_mlp_glm4=False (NEGATIVE): v1 ok '
                   '(rel err 0.0); E1 acc 0.5256 BELOW null p95 0.6295 '
                   '(null mean 0.5261 - at chance); E2 same-language '
                   'margin 0.0333 < null p95 0.0395; E3 concept acc '
                   '0.0128 (also ~0 - not a channel swap, the spectrum '
                   'carries neither language nor concept for this '
                   'direction); WC language_carrier_deviation (delta '
                   '-0.2463 vs qwen3-4b 0.7719); direction quality ok '
                   '(E4 cos 1.0 at own layer li=20, decays 0.42->0.08 '
                   'across window); D0 GLM4 hourglass: probe >=0.96 at '
                   'essentially all layers (N5 no-stripping confirmed '
                   'cross-model), CKA peak EARLY li=9 (0.8954) - '
                   'different shape from qwen step at li 6->7',
        'source': {'path': 'phase2890/language_axis_glm4/result.json',
                   'sha256_8': sha_res, 'phase': 2890}})

if 'G_language_glm4' not in ids_growth:
    d['growth_curve'].append({
        'point_id': 'G_language_glm4', 'axis_id': 'language',
        'model': 'glm4-9b-chat-hf', 'block': 'B3_language_glm4',
        'components': 12, 'acc': 0.5256, 'phase': 2890,
        'notes': 'NEGATIVE point: acc at null mean (0.5261), below '
                 'null p95; language-direction mlp carrier is '
                 'qwen3-4b-scoped so far; language carrier curve: '
                 'qwen3-4b 0.7719 / glm4 0.5256'})

if 'N6_language_carrier_glm4' not in ids_neg:
    d['negatives'].append({
        'neg_id': 'N6_language_carrier_glm4',
        'claim': 'the 2887 4th-family mlp direct-write language '
                 'carrier (L10) generalizes across architectures',
        'evidence': 'M2890: E1 acc 0.5256 < null p95 0.6295 (null '
                    'mean 0.5261, at chance); E2 margin below null; '
                    'E3 concept acc 0.0128 ~ 0 (neither language nor '
                    'concept content in the spectrum); E4 direction '
                    'quality verified (cos 1.0 at li=20)',
        'status': 'settled',
        'implication': 'the mlp carrier law is axis-family x model '
                       'specific for language: qwen3-4b carries its '
                       'mid-stream language direction through the mlp '
                       'channel, glm4 does not - glm4 keeps language '
                       'identity in the residual stream (probe >=0.96 '
                       'all layers, D0) without mlp direct-write '
                       'engagement; class/attr/syntax carriers DID '
                       'replicate on glm4 (M2888), so the failure is '
                       'specific to the language direction provenance '
                       '(sentence-state mid-stream); open candidate: '
                       'attention-response spectrum for lang_dir on '
                       'glm4 (language may route via attention there)',
        'source': {'phase': 2890, 'meas_id': 'M2890_language_axis_glm4'}})

with io.open(LEDGER, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
d2 = json.load(io.open(LEDGER, encoding='utf-8'))
assert len(d2['blocks']) == 15 and len(d2['measurements']) == 29 \
    and len(d2['growth_curve']) == 16 and len(d2['negatives']) == 6, \
    'ledger count mismatch'

memo_section = """
### Phase 2890 - GLM4 语言轴判别（2887 协议移植，阴性登记）

**日期**：2026-09-18。**模型**：glm4-9b-chat-hf（glm 架构，40 层，
hidden 4096）。运行 472.9s（两阶段：80 句全前向提 S_last -> li=20
句状态语言方向 -> 78 词 x 3 条件 x [28,40) 窗口 mlp 响应谱）。

### 原理与移植

2886+2887 协议 verbatim 移植，方向来源改为 GLM4 自身句状态（per-model
provenance，预注册登记）：80 对 en/fr 句子（2886 冻结常量复制），
lang_dir = unit(mean S_last[en,20] - mean S_last[fr,20])，LI 缩放规则
round(18/36*L)=20。词表 2878 TRANS_PAIRS + 同形排除表原样，GLM4
tokenizer 过滤后 49/72 对，tid 级跨语言去重 78 词（en=29/L=49）。

### 判决：language_axis_in_mlp_glm4 = **False（阴性）**

| 门线 | 观测 | 结果 |
|---|---|---|
| v1 确定性 | max rel err **0.0** | 通过 |
| **E1** 语言检索 | acc **0.5256** vs null p95 0.6295（null mean 0.5261） | **mlp_language_signal_absent_glm4** |
| **E2** 同语言余量 | 0.0333 < null p95 0.0395 | **margin_absent_glm4** |
| E3 概念对照 | 0.0128（null p95 0.0641） | 概念也 ~0（非通道置换） |
| WC 跨模型带 | delta = **-0.2463** vs qwen 0.7719 | **language_carrier_deviation** |

### 解读（三个实质发现，重复三遍）

1. **阴性成立且干净**：acc 精确落在 null mean（0.5256 vs 0.5261），
   谱既不携带语言也不携带概念（E3 也 ~0）——不是"语言被概念挤占"，
   而是该方向在 GLM4 的 mlp 通道中**根本没有直写响应**。
2. **载体定律是 轴族 x 模型 特异的（语言族）**：class/attr/syntax 三族
   在 GLM4 复制（M2888），语言族不复制——失败特异性指向方向来源
   （句状态中层提取）。GLM4 把语言身份保留在残差流（D0 探针全层
   >=0.96，N5 跨模型确认"不剥离"），但不经 mlp 直写。
3. **GLM4 沙漏形状不同**：CKA 峰值前移 li=9（0.8954）后单调下降，
   与 qwen 的 li6->7 阶梯+平台不同；探针无 dip（ends 0.9167）。
   E4 证实方向质量：cos(lang_dir_q, lang_dir) 在 li=20 处 =1.0，
   窗口内衰减 0.42->0.08——方向真实存在于中层，但深层 mlp 不响应。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2890/language_axis_glm4/execution.json | 6defb457 |
| phase2890/language_axis_glm4/result.json | eaafdf3 -> **eaafdf3 修正：eaaf6df3** |
| phase2890/language_axis_glm4/language_axis_glm4.npz | 74933303 |
| tests/glm5/phase2890_language_axis_glm4.py | 29081deb |

（result.json sha256_8 = eaaf6df3。）

Ledger 更新：B3_language_glm4 + M2890_language_axis_glm4 +
G_language_glm4（阴性点）+ **N6_language_carrier_glm4**（settled
negative，阴性登记第 6 条）；blocks 15 / measurements 29 / growth 16
/ negatives 6。

### 接续（2891 候选）

- **A（主选）**：GLM4 语言注意通路判别——lang_dir 的 attention-response
  谱（N6 启示：GLM4 语言可能走 attention 而非 mlp；head-level drop/
  响应谱注入 lang_dir）
- **B**：Gemma4 冷启动（model_namespace 收尾）
- **C**：distill 假设升级 settled（Qwen2-7B-Instruct 网络获取，可选）

*（SHA 与判决以 result.json 为准；本节由 phase2890 收尾脚本追加。）*
"""

memo_section = memo_section.replace(
    '| phase2890/language_axis_glm4/result.json | eaafdf3 -> '
    '**eaafdf3 修正：eaaf6df3** |',
    '| phase2890/language_axis_glm4/result.json | eaaf6df3 |')

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(memo_section + '\n')

print('ledger: blocks=%d meas=%d growth=%d neg=%d'
      % (len(d2['blocks']), len(d2['measurements']),
         len(d2['growth_curve']), len(d2['negatives'])))
print('memo appended')
