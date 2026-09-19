# -*- coding: utf-8 -*-
"""Phase 2884 finalize: SHA + Ledger v2 + MEMO (2884 + attachment analysis)."""
import hashlib
import io
import json
import os
import sys

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(ROOT, r'tests\gpt5_temp\_p2884_finalize.txt')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
BASE = os.path.join(ROOT, r'tests\glm5\result'
                    r'\rdc_query_construction_20260913')
PDIR = os.path.join(BASE, 'phase2884', 'window_discriminate')

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
    'npz': os.path.join(PDIR, 'window_discriminate.npz'),
    'script': os.path.join(ROOT, r'tests\glm5'
                           r'\phase2884_window_discriminate.py'),
}
shas = {k: sha8(p) for k, p in files.items()}
for k in sorted(shas):
    g.write('SHA %s %s\n' % (k, shas[k]))

# ---------- ledger v2 update ----------
from rdc_atlas_ledger import AtlasLedger
led = AtlasLedger.load(verify_sha=True)
doc = led.doc
assert doc['version'] == 2

if not any(m['meas_id'] == 'M2884_window_discriminate'
           for m in doc['measurements']):
    doc['measurements'].append({
        'meas_id': 'M2884_window_discriminate', 'type': 'attribution',
        'model': 'qwen3-4b + deepseek-r1-distill-qwen-7b',
        'verdict': 'window_discriminate=model_factor: C0=true (q_base10 '
                   '0.875 and d_base8 0.575 exactly reproduced); C1=false '
                   '(qwen 8L slices 0.8875/0.8250, within band); C2=false '
                   '(ds7b 10L 0.6375, +0.0625 < 0.15); C3=false; the '
                   '-0.30 carrier deviation is a genuine model factor, '
                   'window length/position excluded',
        'source': {'path': 'phase2884/window_discriminate/result.json',
                   'sha256_8': shas['result'], 'phase': 2884}})
if not any(l['link_id'] == 'L9_carrier_attribution'
           for l in doc['linkage']):
    doc['linkage'].append({
        'link_id': 'L9_carrier_attribution',
        'from': {'axis': 'class', 'model': 'qwen3-4b', 'block': 'B3_mlp'},
        'to': {'axis': 'class',
               'model': 'deepseek-r1-distill-qwen-7b',
               'block': 'B3_mlp_ds7b'},
        'evidence': 'M2884 five-arm window discrimination: qwen '
                    '{[26,36)=0.875, [26,34)=0.8875, [28,36)=0.8250, '
                    '[24,34)=0.9000}, ds7b {[20,28)=0.575, [18,28)=0.6375}; '
                    'no manipulation moves acc by >=0.15 => carrier '
                    'strength is an intrinsic model property (candidate '
                    'causes: distill training, untied lm_head geometry)',
        'phase': 2884, 'status': 'confirmed'})
led.save()

rep = []
led2 = AtlasLedger.load(verify_sha=True)
stale2 = led2.verify(rep)
g.write('ledger v%s: %d blocks / %d measurements / %d growth / '
        '%d linkage / stale=%d\n'
        % (led2.doc['version'], len(led2.doc['blocks']),
           len(led2.doc['measurements']), len(led2.doc['growth_curve']),
           len(led2.doc['linkage']), len(stale2)))
g.write('ledger sha %s\n' % sha8(os.path.join(
    ROOT, r'research\gpt5\atlas\atlas_ledger.json')))

# ---------- MEMO append ----------
before = sum(1 for _ in io.open(MEMO, encoding='utf-8'))
section = u"""

---

## Phase 2884：窗口长度判别——DS7B −0.30 载体偏差归因（P3 判别站）

**日期**：2026-09-18。**模型**：qwen3-4b + deepseek-r1-distill-qwen-7b。

### 原理

2883 发现载体强度跨模型偏差（Δ=−0.30），候选原因：①窗口 8 层 vs 10 层；
②模型因子（distill 训练 / untied lm_head 几何）；③窗口位置。关键效率事实：
g_direct 按**层**定义、窗口=切片——hooks 本就捕获全部层，故每模型只测**一次
全层** g_direct（80 词），五臂窗口全部为切片，一次前向覆盖全部判别臂。

### 门线（execution.json 先于任何观测冻结）

- D1 每模型一次全层测量，窗口=切片（2883 词表/上下文/E 行协议 verbatim）；
  qwen3-4b tied 无 lm_head 行 → E 行取 model.embed_tokens.weight（tie 语义
  数学等价，记为 D1b）
- D2 窗口集冻结：qwen {[26,36), [26,34), [28,36), [24,34)}；ds7b
  {[20,28), [18,28)}
- W3 每模型重构一致性 v1/v2<0.05（违反则全部判决作废）
- **C0 确定性**：acc(q_base10)==0.875 且 acc(d_base8)==0.575 **精确**成立，
  否则作废
- C1 qwen 长度效应：8L 切片偏离基线 ≥0.15；C2 ds7b 长度效应：[18,28) 增益
  ≥+0.15；C3 位置效应：q_early8 与 q_late8 差 ≥0.15
- 归因（优先级冻结）：window_length iff C1∧C2；model_factor iff ¬C1∧¬C2∧¬C3；
  position_matters iff C3∧¬(C1∧C2)；else mixed。null 200 置换，SEED=2884

### 结果

**window_discriminate=model_factor（C0=True，C1/C2/C3 全 False）**：

| 臂 | 窗口 | acc | null p95 |
|---|---|---|---|
| q_base10 | [26,36) | **0.8750**（精确复现 C0） | 0.1625 |
| q_early8 | [26,34) | 0.8875 | 0.1625 |
| q_late8 | [28,36) | 0.8250 | 0.1500 |
| q_end34 | [24,34) | 0.9000 | 0.1506 |
| d_base8 | [20,28) | **0.5750**（精确复现 C0） | 0.1500 |
| d_long10 | [18,28) | 0.6375 | 0.1625 |

全部窗口操作的 acc 变化 <0.15 带。**−0.30 偏差是真实模型因子：载体强度是
模型内在属性，与测量窗口无关**。W3 双模型过（qwen 0.0078 / ds7b 0.0000，
bf16 预算内）。运行 2m20s（一次前向双模型全层）。

### 解读

1. 窗口长度/位置假设排除——qwen 在 8 层切片仍 0.83-0.90，ds7b 加长到 10 层
   仅 +0.0625。
2. 载体强度归入模型因子。可检验候选（观测后）：reasoning-distill 训练改变
   mlp 编码密度；untied lm_head 使 E 行几何不同（2885 附件分析给出判别杠杆）。
3. 跨模型增长率曲线注记修正：组件数差异不解释偏差，Ledger L9 登记。

### 文件与 SHA256-8

| 文件 | sha256_8 |
|---|---|
| phase2884/window_discriminate/execution.json | %(exec)s |
| phase2884/window_discriminate/result.json | %(result)s |
| phase2884/window_discriminate/window_discriminate.npz | %(npz)s |
| tests/glm5/phase2884_window_discriminate.py | %(script)s |

硬伤：qwen3-4b safetensors index 无 lm_head 行（tied）→ 回退
model.embed_tokens.weight（首次运行 KeyError 崩溃后修复，属环境适配非协议
变更）；bash shim 幽灵执行探针纪律再次生效（先探针后重跑）。

### 接续（2885 候选）

- **A（主选）**：DS7B 翻译轴判别测试（untied lm_head）——2878 协议移植：
  若 VG2b>0.2 则 2878 真阴性归因 tied-embedding 扭曲；若仍≈0 则"翻译≠静态轴"
  升级为跨模型事实（附件分析判别杠杆，零前向 ~1min）
- B：CKA 倒 U 曲线（附件武器 1，沙漏模型验证；~50 句对 × 全层捕获）
- C：语言身份轴进入 mlp 图谱（中层语言方向 → g_direct 载体检验，把"语言"
  作为第四轴族）

---

## 附件分析：翻译机制流形理论（"找不到轴反而藏着秘密"）

**日期**：2026-09-18。附件主张：翻译轴不存在 → 翻译不是空间平移而是高维
流形上的上下文重构；沙漏模型（中层语言剥离）；tied embedding 扭曲；
三武器（CKA 倒 U / 语言路由头 / 正交手术）。

### 判定：核心直觉正确，三处需修正，一处给出新判别杠杆

**正确且与实测一致**：
1. "翻译不是静态轴"——2878 实测 VG2b≈0（fr −0.0007/de −0.0029/es 0.0011）
   精确支持。语言恒定分量≈0 是硬数据。
2. "轴不存在是信息而非故障"——真阴性登记（Ledger N1）与附件认识论一致。
3. 沙漏模型（中层语言无关语义层）作为**假设**合理，与文献方向一致，且
   可检验（武器 1 CKA 倒 U + 语言探针，设计可直接采用；预测三段：浅层
   100%%/中层 50%%/深层回升）。

**修正一（证据反例）**：附件把多义词/文化词作为轴不存在的主因。但 2878
的 22 对全是**单义具体名词**（cat/chat、water/eau），仍无统一轴——多义性
不是主因，词形/概念差主导（轴方向 pair-pair cos ≈0）。附件此论证被数据
否定。

**修正二（事实订正）**：2878 词对是 en↔fr/de/es（非附件所称中→英/中→法）；
且空间是 tied unembed 的 E 行差（非"方差极大的高维差"的一般表述——是
**方向一致性**失效，不是方差大）。

**修正三（新判别杠杆，附件遗漏）**：附件把 tied embeddings 几何扭曲列为
原因之二，但未给出检验。本项目现成杠杆：**DS7B 是 untied lm_head**——
2885-A 判别测试：若 untied 模型翻译轴 VG2b>0.2，则 2878 真阴性归因 tied
扭曲；若仍≈0，则"翻译≠静态轴"升级为**跨 tied/untied 的模型普适事实**，
同时沙漏假设获得间接支持（轴信息只可能在中层流形，不在端点 E 行）。

**武器评估**：武器 1（CKA/语言探针）= 低成本可行，直接排期（2885-B）；
武器 2（路由头 + activation patching）= 中成本，qwen3 模块手术经验可复用
（2861 先例），排期靠后；武器 3（正交手术）= 依赖武器 1 先定位语言子空间，
是 P4 因果干预闭环的自然形态，远期。

**结论**：附件理论框架纳入研究主线，作为"翻译轴真阴性"的解释层；其可
证伪子集（沙漏三段预测、tied 判别）转化为 2885 门线实验。

*（本节为研究决策记录，非实验 phase；判别实验 = 2885-A/B。）*
""" % shas

with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(section)
after = sum(1 for _ in io.open(MEMO, encoding='utf-8'))
g.write('MEMO %d -> %d\n' % (before, after))
g.close()
print('ok')
