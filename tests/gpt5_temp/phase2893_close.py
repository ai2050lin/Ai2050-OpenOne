# -*- coding: utf-8 -*-
"""Phase 2893 closing: ledger (M2893 + growth + errata refining N6/N7)
+ MEMO append."""
import hashlib
import io
import json
import os

LEDGER = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase2893'
       r'\language_perlayer_glm4')


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:8]


sha_exec = sha8(os.path.join(OUT, 'execution.json'))
sha_res = sha8(os.path.join(OUT, 'result.json'))
sha_npz = sha8(os.path.join(OUT, 'language_perlayer_glm4.npz'))
assert sha_exec == '51a561e7' and sha_res == 'bb10476a' \
    and sha_npz == '08aae636', 'sha drift'

d = json.load(io.open(LEDGER, encoding='utf-8'))
ids_blocks = [b['block_id'] for b in d['blocks']]
ids_meas = [m['meas_id'] for m in d['measurements']]
ids_growth = [g.get('growth_id', '') for g in d['growth_curve']]

if 'B_perlayer_lang_glm4' not in ids_blocks:
    d['blocks'].append({
        'block_id': 'B_perlayer_lang_glm4', 'axis_id': 'language',
        'kind': 'dual_channel_response_perlayer_dir', 'model':
        'glm4-9b-chat-hf', 'shape': [78, 12],
        'src': {'path': 'phase2893/language_perlayer_glm4/'
                        'language_perlayer_glm4.npz',
                'sha256_8': sha_npz, 'phase': 2893, 'key': 'B_mlp'},
        'notes': 'W=[28,40); per-layer eigen-direction dir_q(li)=unit'
                 '(mean_en S_last[:,li]-mean_fr S_last[:,li]) injected '
                 'into its own layer; both channels A1 positive'})

if 'M2893_language_perlayer_glm4' not in ids_meas:
    d['measurements'].append({
        'meas_id': 'M2893_language_perlayer_glm4',
        'type': 'perlayer_direction_injection', 'model':
        'glm4-9b-chat-hf',
        'verdict': 'perlayer_direction_write_detected_both: per-layer '
                   'eigen-directions (zero-forward from 2890 S_last) '
                   'injected at their own layers in W=[28,40); mlp acc '
                   '0.6923 > null p95 0.6410 (A1 carries, margin 0.0313 '
                   '< p95 0.0349 absent, concept 0.0513 ~null); attn '
                   'acc 0.6667 > p95 0.6288 AND margin 0.0732 > p95 '
                   '0.0384 (both positive, concept 0.0000). Confirms '
                   '2892 hypothesis: deep layers DO write language '
                   'separation but the write direction must be '
                   're-extracted per layer; lang_dir (li=20) negatives '
                   'in 2890/2891/2892 were direction-mismatch '
                   'artifacts. cos(dir_q(li), lang_dir) decays 0.4223 '
                   '(li28) -> 0.0783 (li39). v1 rel err 0.0 both '
                   'channels',
        'source': {'path': 'phase2893/language_perlayer_glm4/'
                           'result.json',
                   'sha256_8': sha_res, 'phase': 2893}})

if 'G_perlayer_lang_glm4' not in ids_growth:
    d['growth_curve'].append({
        'growth_id': 'G_perlayer_lang_glm4', 'phase': 2893,
        'model': 'glm4-9b-chat-hf', 'axis': 'language',
        'finding': 'GLM4 deep-window language write detected with '
                   'layer-matched eigen-directions (mlp 0.6923, attn '
                   '0.6667 both above null p95); 2887 language-carrier '
                   'claim refined: not qwen-exclusive - GLM4 has the '
                   'write but only per-layer directions reveal it; '
                   'direction rotation along depth is intrinsic '
                   '(cos to li=20 lang_dir 0.42 -> 0.08)'})

errata_notes = [e.get('note', '')[:80] for e in d['errata_ledger']]
already = any('per-layer eigen-direction' in n for n in errata_notes)
if not already:
    d['errata_ledger'].append({
        'corrects': 'N6_language_carrier_glm4/N7_attention_route_glm4',
        'note': 'direction-mismatch artifacts corrected by M2893: '
                'layer-matched eigen-directions give A1 positive on '
                'BOTH channels (mlp 0.6923 / attn 0.6667 above null '
                'p95) - GLM4 deep layers do write language separation; '
                '2890/2891 negatives reflect stale li=20 lang_dir, not '
                'absent writes; language write is channel-real but '
                'direction-rotating along depth',
        'phase': 2893})

with io.open(LEDGER, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
d2 = json.load(io.open(LEDGER, encoding='utf-8'))
assert len(d2['blocks']) == 18 and len(d2['measurements']) == 32 \
    and len(d2['errata_ledger']) == 5 \
    and len(d2['growth_curve']) == 18, 'ledger count mismatch'

memo_section = """
### Phase 2893 - GLM4 逐层本征方向注入（2892 假设证实：方向需逐层重提取）

**日期**：2026-09-18。**模型**：glm4-9b-chat-hf。运行 355.9s
（Stage 1 零前向方向提取 + Stage 2 W=[28,40) 逐层本征方向双通道注入）。

### 原理与设计

2892 结论：深层主导写入语言分离（L1=deep_write_dominant），但注入
li=20 的 lang_dir 全阴性——lang_dir 不是深层写入本征方向，方向需
逐层重提取。Phase 2893 直接检验该假设：

- Stage 1（零前向，2890 S_last 复用）：对 W=[28,40) 每层 li 提取
  dir_q(li) = unit(mean_en S_last[:,li] - mean_fr S_last[:,li])。
- Stage 2（预注册冻结）：层 li 注入 **本层自己的** dir_q(li)
  （mlp 与 self_attn 双通道，pos 1，eps=1.0），g 沿 dir_q(li) 投影，
  B = g(same) - 0.5(g(func)+g(null))；词表/条件 verbatim
  2890/2891/2892（78 词零前向）。v1 rel err < 1e-6。
- 对照设计内部有效：方向来源（80 句）与检索域（78 词）分离，
  与 2890/2891 完全一致——唯一变量是"方向是否与注入层匹配"。

### 判决：perlayer_direction_write_detected_both（阳性）

| 门线 | mlp | attn |
|---|---|---|
| v1 rel err | 0.0 | 0.0 |
| A1 acc vs null p95 | **0.6923 > 0.6410**（carries） | **0.6667 > 0.6288**（carries） |
| A2 margin vs p95 | 0.0313 < 0.0349（absent） | **0.0732 > 0.0384**（margin） |
| A3 概念对照 | 0.0513（~null） | 0.0000 |
| 对比 lang_dir 注入（2890/2891） | 0.5256 -> **0.6923** | 0.5897 -> **0.6667** |

cos(dir_q(li), lang_dir) 沿深度衰减：0.4223 (li28) -> 0.0783 (li39)
——方向随层旋转是内禀的。

### 解读（三个实质发现，重复三遍）

1. **2892 假设证实**：同一方向来源、同一窗口、同一协议，仅把
   "全局 lang_dir" 换成 "逐层本征方向"，mlp 0.5256->0.6923、
   attn 0.5897->0.6667，双通道由阴性转显著阳性——**方向需逐层
   重提取**成立。
2. **N6/N7 负结果是方向错配伪影**：GLM4 深层确实在写入语言分离
   （errata 入账 corrects N6/N7）——语言写入通道真实存在，但方向
   沿深度旋转；2887 "语言载体 qwen 特异"结论被精化：非 qwen 独有，
   GLM4 的写入需层匹配方向才可见。
3. **attn 是 GLM4 语言逐层写入的更干净通道**：margin 强阳性
   （0.0732 vs p95 0.0384）且概念 0.0000；mlp acc 过线但 margin
   不过——信号存在但同语言几何组织较弱。方向旋转结构
   （旋转平面假说）成为下一研究对象。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2893/language_perlayer_glm4/execution.json | 51a561e7 |
| phase2893/language_perlayer_glm4/result.json | bb10476a |
| phase2893/language_perlayer_glm4/language_perlayer_glm4.npz | 08aae636 |
| tests/glm5/phase2893_language_perlayer_glm4.py | c629369d |

Ledger 更新：B_perlayer_lang_glm4 + M2893_language_perlayer_glm4 +
G_perlayer_lang_glm4 + **errata（corrects N6/N7）**；blocks 18 /
measurements 32 / growth 18 / errata 5。

### 接续（2894 候选）

- **A（主选）**：方向旋转结构定量——零前向分析 dir_q(li) 的旋转
  平面（相邻层差向量、主旋转平面 PCA、有效秩），对照 Unified
  Theory 旋转平面假说
- **B**：norm-matched 对照（2892 L1 归因混杂，零前向）
- **C**：Gemma4/Qwen2-7B 下载决策（待用户确认网络获取）

*(SHA 与判决以 result.json 为准；本节由 phase2893 收尾脚本追加。)*
"""

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(memo_section + '\n')

print('ledger: blocks=%d meas=%d growth=%d errata=%d'
      % (len(d2['blocks']), len(d2['measurements']),
         len(d2['growth_curve']), len(d2['errata_ledger'])))
print('memo appended')
