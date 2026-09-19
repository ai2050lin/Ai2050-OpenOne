# -*- coding: utf-8 -*-
"""Phase 2885 finalize: SHA + Ledger v2 (M2885 + N4) + MEMO."""
import hashlib
import io
import json
import os
import sys

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(ROOT, r'tests\gpt5_temp\_p2885_finalize.txt')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
BASE = os.path.join(ROOT, r'tests\glm5\result'
                    r'\rdc_query_construction_20260913')
PDIR = os.path.join(BASE, 'phase2885', 'ds7b_trans_axis')

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
    'npz': os.path.join(PDIR, 'ds7b_trans_axis.npz'),
    'script': os.path.join(ROOT, r'tests\glm5'
                           r'\phase2885_ds7b_trans_axis.py'),
}
shas = {k: sha8(p) for k, p in files.items()}
for k in sorted(shas):
    g.write('SHA %s %s\n' % (k, shas[k]))

# ---------- ledger v2 update ----------
from rdc_atlas_ledger import AtlasLedger
led = AtlasLedger.load(verify_sha=True)
doc = led.doc
assert doc['version'] == 2

if not any(m['meas_id'] == 'M2885_ds7b_trans_axis'
           for m in doc['measurements']):
    doc['measurements'].append({
        'meas_id': 'M2885_ds7b_trans_axis', 'type': 'discrimination',
        'model': 'deepseek-r1-distill-qwen-7b',
        'verdict': 'D+_model_general_negative: DS7B UNTIED lm_head '
                   'translation-axis VG2b fr -0.0033 / de -0.0019 / '
                   'es 0.0034 (all <= 0.2, all dead) vs qwen3-4b TIED '
                   'reference -0.0007/-0.0029/0.0011; VG1 3/3 axes pass '
                   '(10/12/13 valid pairs); tied-embedding distortion '
                   'hypothesis REFUTED; "translation != static unembed '
                   'axis" is model-general (tied+untied, 4B+7B)',
        'source': {'path': 'phase2885/ds7b_trans_axis/result.json',
                   'sha256_8': shas['result'], 'phase': 2885}})
if not any(n.get('neg_id') == 'N4_ds7b_translation'
           for n in doc.get('negatives', [])):
    doc['negatives'].append({
        'neg_id': 'N4_ds7b_translation',
        'claim': 'no unified cross-lingual axis direction in the '
                 'UNTIED unembed (lm_head rows) of DS7B',
        'evidence': 'M2885: VG2b fr -0.0033 / de -0.0019 / es 0.0034, '
                    'all axes quarantined; homograph-excluded pools, '
                    '10-13 valid pairs per axis',
        'status': 'settled',
        'implication': 'with N1 (qwen tied), absence of a static '
                       'translation axis at the endpoints is '
                       'model-general; hourglass (mid-stream language '
                       'processing) is the main surviving hypothesis',
        'source': {'phase': 2885, 'meas_id': 'M2885_ds7b_trans_axis'}})
led.save()

rep = []
led2 = AtlasLedger.load(verify_sha=True)
stale2 = led2.verify(rep)
g.write('ledger v%s: %d blocks / %d measurements / %d growth / '
        '%d linkage / %d negatives / stale=%d\n'
        % (led2.doc['version'], len(led2.doc['blocks']),
           len(led2.doc['measurements']), len(led2.doc['growth_curve']),
           len(led2.doc['linkage']), len(led2.doc['negatives']),
           len(stale2)))
g.write('ledger sha %s\n' % sha8(os.path.join(
    ROOT, r'research\gpt5\atlas\atlas_ledger.json')))

# ---------- MEMO append ----------
before = sum(1 for _ in io.open(MEMO, encoding='utf-8'))
section = u"""

---

## Phase 2885：DS7B 翻译轴判别测试——tied 假设否定，阴性升级为模型普适

**日期**：2026-09-18。**模型**：deepseek-r1-distill-qwen-7b（untied lm_head）。
零前向，运行 6.1s。

### 原理

2878 真阴性（qwen3-4b tied unembed 无统一跨语言轴）存在一个未检验混淆变量：
tied-embedding 几何扭曲（附件原因之二）。DS7B 是 untied——把 2878 翻译族
协议 verbatim 移植（同词池、同同形排除表、同门线），E 行改取 lm_head.weight
（2883 safetensors 直读先例），即可判别：
- D- attributed_to_tied：DS7B VG2b>0.2 存活 ≥2 轴 → 2878 阴性归因 tied
- D+ model_general_negative：≥2 轴 VG1 过但全部 VG2b≤0.2 → 阴性与 tying
  无关，"翻译≠静态轴"升级为模型普适事实，沙漏假设获间接支持

### 门线（execution.json 先于任何观测冻结）

T1 翻译族 only（2878 Gen2 词池+同形排除表 verbatim）；T2 E=lm_head 行；
T3 门线 verbatim 2878（VG1 ≥5 对/轴且 ≥3/3；VG2a <0.488；VG2b >0.2 存活；
VG3 <3.0）；T4 零前向 SEED=2885。

### 结果

**D+_model_general_negative**：

| 轴 | 有效对 | VG2b (DS7B untied) | qwen tied 参考 |
|---|---|---|---|
| lang_fr | 10 | **−0.0033** | −0.0007 |
| lang_de | 12 | **−0.0019** | −0.0029 |
| lang_es | 13 | **+0.0034** | +0.0011 |

VG1 3/3 过（DS7B 单 token 率低于 qwen：Qwen2 tokenizer 对法/德/西词切分更
碎，但每轴仍 10-13 对）。三轴 VG2b 全部精确死区（|·|<0.005），与 qwen tied
惊人一致——**tied-embedding 扭曲假设被否定**。

### 解读

1. **"翻译≠端点静态轴"为模型普适事实**（tied/untied × 4B/7B 四格全阴），
   Ledger N1+N4 双登记。
2. **沙漏假设成为主要幸存解释**：语言身份信息若存在统一方向，只可能在
   中层残差流（语言无关语义层两侧），不在端点 E 行——附件武器 1（CKA 倒
   U + 语言探针）优先级提升。
3. 附件理论框架经两轮实测收敛：多义性论证被否（修正一）、tied 论被否
   （本轮）、沙漏论可检验且升级为主假设。**"找不到轴→去中层找流形"成为
   2886 主选的直接依据**。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2885/ds7b_trans_axis/execution.json | %(exec)s |
| phase2885/ds7b_trans_axis/result.json | %(result)s |
| phase2885/ds7b_trans_axis/ds7b_trans_axis.npz | %(npz)s |
| tests/glm5/phase2885_ds7b_trans_axis.py | %(script)s |

### 接续（2886 候选）

- **A（主选）**：CKA 倒 U + 语言探针（附件武器 1，沙漏验证；句级 hidden
  states 全层捕获，~40 句对 × 2 语言）
- B：GLM4 冷启动（P3 第三模型，glm4-9b-chat-hf 32 层 → 窗口 [23,32)）
- C：语言身份方向进 mlp 图谱（中层语言方向 → g_direct 载体检验，第四轴族）

*（SHA 与判决以 result.json 为准；本节由 phase2885 收尾脚本追加。）*
""" % shas

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(section)
after = sum(1 for _ in io.open(MEMO, encoding='utf-8'))
g.write('MEMO %d -> %d\n' % (before, after))
g.close()
print('ok')
