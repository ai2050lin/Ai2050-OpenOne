# -*- coding: utf-8 -*-
"""Phase 2886 finalize: SHA + Ledger v2 (M2886 + N5) + MEMO."""
import hashlib
import io
import json
import os
import sys

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(ROOT, r'tests\gpt5_temp\_p2886_finalize.txt')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
BASE = os.path.join(ROOT, r'tests\glm5\result'
                    r'\rdc_query_construction_20260913')
PDIR = os.path.join(BASE, 'phase2886', 'hourglass_cka')

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
    'npz': os.path.join(PDIR, 'hourglass_cka.npz'),
    'script': os.path.join(ROOT, r'tests\glm5'
                           r'\phase2886_hourglass_cka.py'),
}
shas = {k: sha8(p) for k, p in files.items()}
for k in sorted(shas):
    g.write('SHA %s %s\n' % (k, shas[k]))

from rdc_atlas_ledger import AtlasLedger
led = AtlasLedger.load(verify_sha=True)
doc = led.doc
assert doc['version'] == 2

if not any(m['meas_id'] == 'M2886_hourglass_cka'
           for m in doc['measurements']):
    doc['measurements'].append({
        'meas_id': 'M2886_hourglass_cka', 'type': 'hourglass_test',
        'model': 'qwen3-4b',
        'verdict': 'hourglass_validated=False: H1=false (CKA curve is '
                   'a STEP not inverted-U: last-token cross-language '
                   'CKA low 0.13-0.19 at li 2-6, jumps to plateau '
                   '0.75-0.84 from li=7, slow decline; peak li=12 '
                   '0.8394 but peak-ends contrast 0.086 < 0.10 gate); '
                   'H2=false decisively (language probe LOO '
                   'nearest-centroid = 1.0000 at EVERY layer li 1-36, '
                   'mid NOT lower than ends 0.9146) - language '
                   'identity is never stripped',
        'source': {'path': 'phase2886/hourglass_cka/result.json',
                   'sha256_8': shas['result'], 'phase': 2886}})
if not any(n.get('neg_id') == 'N5_hourglass_stripping'
           for n in doc.get('negatives', [])):
    doc['negatives'].append({
        'neg_id': 'N5_hourglass_stripping',
        'claim': 'hourglass "language stripped mid-stream" (language '
                 'probe dips to ~chance at mid layers)',
        'evidence': 'M2886 H2: probe = 1.0000 at every layer li 1-36 '
                    '(80 sentences, en/fr); H1: CKA step at li 6->7 '
                    'to 0.8 plateau, not inverted-U',
        'status': 'settled',
        'implication': 'language identity coexists with cross-language '
                       'semantic alignment at all depths; the correct '
                       'picture is "aligned semantics WITH persistent '
                       'language identity", mid-layer language '
                       'directions are extractable (probe=1.0) -> '
                       'candidate 4th axis family for the mlp carrier '
                       'law',
        'source': {'phase': 2886, 'meas_id': 'M2886_hourglass_cka'}})
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

before = sum(1 for _ in io.open(MEMO, encoding='utf-8'))
section = u"""

---

## Phase 2886：沙漏验证——"中层剥离语言"否定，CKA 呈阶梯非倒 U

**日期**：2026-09-18。**模型**：qwen3-4b。40 对冻结 en/fr 平行句（与
2878/2885 概念池同源），last-token 全层 hidden states，运行 19.2s。

### 门线（execution.json 先于任何观测冻结）

H1 倒 U：argmax CKA ∈ [12,24] 且峰值−端点均值 ≥0.10；H2 探针凹陷：中层
均值 ≤ 端点均值−0.15（LOO 最近质心，免训练）；H3 分半 ceiling（描述性）。

### 结果

**hourglass_validated=False（H1=False，H2=False 反向）**：

- **CKA 阶梯而非倒 U**：last-token 跨语言 CKA 在 li 2-6 低（0.125-0.192），
  **li 6→7 跳变**至 0.806，此后平台 0.75-0.84 缓降（峰值 li=12=0.8394，
  但峰-端差仅 0.086 < 0.10 门线）
- **语言探针全层完美**：li 1-36 每层 acc=1.0000（端点均值仅 0.9146）——
  **语言身份在任何深度都不被剥离，中层反而最可读**
- 分半 within-en CKA（0.38-0.66）**低于**跨语言 CKA（0.75+）：对齐由共享
  语义内容驱动，跨语言对齐已达内容变异允许的天花板
- mean-pool 变体（描述性）：中层塌至 ~0.06，末层回升至 0.72——词序/长度
  差异主导位置混合表征，末层重新汇聚
- li=0 CKA=0 为退化伪影（last-token 均为"."，中心化后方差≈0），li 1 起有效

### 解读

1. **附件"沙漏"版本被否定（N5）**：语言不被中层剥离——探针全层 1.0。
   附件终章"超越语言的纯粹概念宇宙"不成立：正确图景是**对齐语义与持续
   语言身份共存**。
2. **新结构**：li 6→7 阶梯 = "语言共享语义区"自约 0.20 深度开始，而非倒 U
   峰值。深部缓慢下降（0.84→0.75）是唯一与"重新着装"沾边的信号，但语言
   身份始终可读，非剥离-再穿。
3. **与轴阴性的一致性**：端点无统一轴（N1/N4）+ 语言身份全层在场 ⇒ 翻译
   机制不在"轴"也不在"剥离"，语言信息以**非统一方向的高维形式**全程在
   场——中层语言方向可提取（探针 1.0）→ 直接可测 mlp 载体（第四轴族）。
4. 附件理论框架三轮实测终局：多义论否、tied 论否、剥离论否；幸存核心 =
   "翻译是动态语境重构"（定性正确），但其空间机制需按本轮证据重述。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2886/hourglass_cka/execution.json | %(exec)s |
| phase2886/hourglass_cka/result.json | %(result)s |
| phase2886/hourglass_cka/hourglass_cka.npz | %(npz)s |
| tests/glm5/phase2886_hourglass_cka.py | %(script)s |

硬伤：探针首跑 IndexError（lab[m] 索引错位）→ 修复重跑（清产物纪律）；
last-token 设计使 li 0 退化、端点低 CKA 部分由标点主导——Gen2 若需精确定
位阶梯位置应用 content-token CKA（未列门线）。

### 接续（2887 候选）

- **A（主选）**：语言身份轴进 mlp 图谱（第四轴族）——从 2886 npz 中层
  （li 12-24）提取语言方向（en 质心−fr 质心，或探针权重），g_direct 载体
  检验（~2min 前向）；载体曲线补第四点 class/attr/syntax/language
- B：GLM4 冷启动（P3 第三模型）
- C：content-token CKA 精确定位 li 6→7 阶梯（Gen2，描述性）

*（SHA 与判决以 result.json 为准；本节由 phase2886 收尾脚本追加。）*
""" % shas

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(section)
after = sum(1 for _ in io.open(MEMO, encoding='utf-8'))
g.write('MEMO %d -> %d\n' % (before, after))
g.close()
print('ok')
