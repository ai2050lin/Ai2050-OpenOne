# -*- coding: utf-8 -*-
"""Phase 2892 closing: ledger (M2892 + errata correcting N7) + MEMO."""
import hashlib
import io
import json
import os

LEDGER = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase2892'
       r'\language_write_locate_glm4')


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:8]


sha_exec = sha8(os.path.join(OUT, 'execution.json'))
sha_res = sha8(os.path.join(OUT, 'result.json'))
sha_npz = sha8(os.path.join(OUT, 'language_write_locate_glm4.npz'))
assert sha_exec == 'b6fd79b7' and sha_res == '9f9c874b' \
    and sha_npz == 'd760c567', 'sha drift'

d = json.load(io.open(LEDGER, encoding='utf-8'))
ids_blocks = [b['block_id'] for b in d['blocks']]
ids_meas = [m['meas_id'] for m in d['measurements']]

if 'B_write_locate_glm4' not in ids_blocks:
    d['blocks'].append({
        'block_id': 'B_write_locate_glm4', 'axis_id': 'language',
        'kind': 'dual_channel_response', 'model': 'glm4-9b-chat-hf',
        'shape': [78, 3],
        'src': {'path': 'phase2892/language_write_locate_glm4/'
                        'language_write_locate_glm4.npz',
                'sha256_8': sha_npz, 'phase': 2892, 'key': 'B_mlp'},
        'notes': 'W2=[37,40) (li*=39 rule-frozen from Stage1 Delta '
                 'argmax); mlp margin weak-positive (0.0510 > p95 '
                 '0.0447) but acc below null p95; attn negative'})

if 'M2892_language_write_locate_glm4' not in ids_meas:
    d['measurements'].append({
        'meas_id': 'M2892_language_write_locate_glm4',
        'type': 'write_location', 'model': 'glm4-9b-chat-hf',
        'verdict': 'Stage1 (zero forward, 2890 S_last): L1='
                   'deep_write_dominant (segment shares of |Delta '
                   'sep|: early 0.048 / mid 0.345 / DEEP 0.608); L2='
                   'False (deep |Delta| 4.18 > 0.10*sep_end 0.69) - '
                   'NOT passive preservation, N7 passive claim '
                   'corrected; sep grows 2.49 (li20) -> 9.61 (li39), '
                   'layer 39 Delta -2.74 retraction; li*=39 -> W2='
                   '[37,40). Stage2 (dual-channel injection, rule-'
                   'frozen window): mlp acc 0.6026 < null p95 0.6282 '
                   '(A1 absent) but margin 0.0510 > p95 0.0447 (A2 '
                   'weak-positive); attn acc 0.4231 absent. '
                   'no_aligned_write_in_W2: the deep-window language '
                   'separation growth is NOT channel direct-write '
                   'along lang_dir - lang_dir (li=20 sentence '
                   'centroid diff) is not the deep-write '
                   'eigen-direction; caveat: Delta sep includes '
                   'norm-growth contribution (dnorm curve rises '
                   'monotonically), norm-matched control = open '
                   'candidate',
        'source': {'path': 'phase2892/language_write_locate_glm4/'
                           'result.json',
                   'sha256_8': sha_res, 'phase': 2892}})

errata_notes = [e.get('note', '')[:60] for e in d['errata_ledger']]
already = any('lang_dir is not the deep-write' in n
              for n in errata_notes)
if not already:
    d['errata_ledger'].append({
        'corrects': 'N7_attention_route_glm4',
        'note': 'passive-preservation claim corrected by M2892: '
                'deep-window language-separation growth is dominant '
                '(60.8 percent of |Delta sep|, L2=False), the write '
                'is real but NOT aligned to lang_dir (mlp/attn '
                'injection negatives at W2 [37,40)); lang_dir is not '
                'the deep-write eigen-direction; embedding-lexical '
                'origin claim also softened (probe 1.0 from li=1 '
                'stands, but the direction keeps growing deep)',
        'phase': 2892})

with io.open(LEDGER, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
d2 = json.load(io.open(LEDGER, encoding='utf-8'))
assert len(d2['blocks']) == 17 and len(d2['measurements']) == 31 \
    and len(d2['errata_ledger']) == 4, 'ledger count mismatch'

memo_section = """
### Phase 2892 - GLM4 语言写入层定位（N7 修正：深层写入但不沿 lang_dir）

**日期**：2026-09-18。**模型**：glm4-9b-chat-hf。运行 476.1s
（Stage 1 零前向定位 + Stage 2 W2=[37,40) 双通道注入判别）。
（候选 A Gemma4 冷启动：本地模型目录探针确认无 gemma4，需 ~8GB
下载，暂缓待用户决策。）

### 原理与设计

Stage 1（零前向，2890 S_last 复用）：sep(li) =
(mean_en - mean_fr) . lang_dir；Delta(li) = sep(li+1) - sep(li) =
层 li 模块沿 lang_dir 的净分离增量。分段规则冻结：
EARLY Delta(0..12) / MID(13..27) / DEEP(28..39)。Stage 2 规则先冻结：
li* = argmax|Delta|，W2 = [max(0,li*-2), min(L,li*+4))（观测只填参）。

### 判决：L1=deep_write_dominant；L2=False；Stage2=no_aligned_write_in_W2

| 门线 | 观测 | 结果 |
|---|---|---|
| L1 分段归因 | shares early **0.048** / mid 0.345 / **DEEP 0.608** | **deep_write_dominant** |
| L2 深层被动 | deep abs 4.18 vs 0.10*sep_end 0.69 | **False（非被动）** |
| sep 曲线 | 2.49 (li20) -> 9.61 (li39)；层 39 Delta **-2.74** 回撤 | 深层主导增长 |
| Stage2 mlp | acc 0.6026 < p95 0.6282；margin 0.0510 > p95 0.0447 | A1 阴性 / A2 弱阳性 |
| Stage2 attn | acc 0.4231；margin -0.0231 | 双阴性 |

### 解读（三个实质发现，重复三遍）

1. **N7"被动保留"被修正**：GLM4 语言分离由深层主导构建（DEEP 占
   60.8%，sep 2.49 -> 9.61），不是 embedding 后被动保留——errata
   入账（corrects N7）。
2. **深层写入与 lang_dir 不对齐**：分离增长真实存在，但直接注入
   lang_dir 三轮全阴性（2890 mlp / 2891 attn / 2892 W2 双通道）——
   **lang_dir（li=20 句质心差）不是深层写入的本征方向**；语言分离
   的深层写入发生在 lang_dir 的旋转/正交子空间。
3. **mlp margin 弱阳性线索**：W2 mlp 同语言余量过 p95（0.0510 vs
   0.0447）而 acc 不过——W2 mlp 有微弱同语言几何组织，方向可能
   需逐层重提取（li=39 的本征方向 != li=20 的 lang_dir）。
   混杂登记：Delta sep 含范数增长贡献（dnorm 单调上升），norm-
   matched 对照 = open candidate。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2892/language_write_locate_glm4/execution.json | b6fd79b7 |
| phase2892/language_write_locate_glm4/result.json | 9f9c874b |
| phase2892/language_write_locate_glm4/language_write_locate_glm4.npz | d760c567 |
| tests/glm5/phase2892_language_write_locate_glm4.py | e3f1a38e |

Ledger 更新：B_write_locate_glm4 + M2892_language_write_locate_glm4 +
**errata（corrects N7）**；blocks 17 / measurements 31 / errata 4。

### 接续（2893 候选）

- **A（主选）**：逐层本征方向提取 + 注入（li 25..39 每层句质心差
  dir_q 注入该层 mlp，检验"方向需逐层重提取"假设；零前向方向 +
  单窗口前向，成本低）
- **B**：norm-matched 对照（Delta sep 减去范数增长期望，纯化 L1
  归因；零前向）
- **C**：Gemma4/Qwen2-7B 下载决策（待用户确认网络获取）

*（SHA 与判决以 result.json 为准；本节由 phase2892 收尾脚本追加。）*
"""

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(memo_section + '\n')

print('ledger: blocks=%d meas=%d errata=%d'
      % (len(d2['blocks']), len(d2['measurements']),
         len(d2['errata_ledger'])))
print('memo appended')
