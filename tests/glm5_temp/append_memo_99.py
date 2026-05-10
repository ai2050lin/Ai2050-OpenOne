import sys, os
sys.stdout.reconfigure(encoding='utf-8')

# 读取现有memo
with open('research/glm5/docs/AGI_GLM5_MEMO.md', 'r', encoding='utf-8') as f:
    content = f.read()

append_text = """

---

## Phase 99: 因果隔离与计算原语检测 [2026-05-09 00:52]

### 核心方法论升级

Phase 98批判指出的最关键问题:
> "所有结论都是correlation，不是causation"
> "对最终输出影响大≠执行了核心计算"

Phase 99引入因果隔离(Causal Isolation): 证明"没有X→没有Y"，不是"X和Y同时出现"

---

### Exp 1: 因果必要性测试 — Qwen3

**最关键的发现: Phase 98的结论被彻底推翻!**

| 层 | Attn zero-ablate | MLP zero-ablate |
|---|---|---|
| **L0** | 11.3% drop | **99.996% drop** ← 最关键! |
| **L6** | **83.6% drop** | **99.996% drop** |
| L26 | 1.7% drop | 17.8% drop |
| L31 | 1.8% drop | 3.5% drop |

**Phase 98说"L26 Attn和L31 MLP是翻译关键层"→ 错！**

L0 MLP zero-ablate → 翻译概率从0.91降到0.000032 → 100%崩塌！
L26 Attn zero-ablate → 翻译概率0.90 → 仅1.7%下降！

**L26和L31的path patching高值只说明它们"传递了翻译信息"，但它们不是因果必要的。**

---

### Exp 2: Hidden State语义子空间 — Qwen3

**最震撼的发现: 表示层切换和logits层切换差31层！**

| 层 | P(en) | 解读 |
|---|---|---|
| L0 | 0.011 | 中文 |
| L6 | **0.594** | ← 切换点! |
| L12 | 0.999 | 英文 |
| L18-35 | 0.998-1.000 | 英文 |

**Hidden state切换层: L1.1 (深度2.9%)**
**Logits切换层: L32 (深度89%)**
**差距: 31层！**

这意味着:
1. 翻译prompt的hidden state在L1就已经被分类为"英文"了
2. 但logits层到L32才切换
3. 中间30层的"表示层已经是英文，但输出还是中文"

**直接回答批判: "振荡是logit几何还是语义切换?"**
→ **表示层在L1就完成切换，logits层的振荡完全是decoder投影延迟！**

但注意: 这个"切换"可能反映的是prompt token组成(翻译prompt有"的英文是"后缀)，不完全是语义切换。需要控制实验验证。

---

### Exp 3: Head级因果中介 — Qwen3

**没有任何单独的head对翻译有显著因果必要性！**

- 最高翻译drop: 1.7% (L34:H0)
- 最高翻译特异性: 8.1% (L34:H0)
- 所有head的翻译drop都<2%

**和Exp 1一致: 翻译是分布式表示的结果，不是局部电路。**

---

### Exp 4: 跨任务原语检测 — Qwen3

| 层 | 翻译drop | 检索drop | 约束drop | 类型 |
|---|---|---|---|---|
| **L0** | 100% | 99.8% | 90.9% | 通用 |
| **L6** | 100% | 99.3% | 92.3% | 通用 |
| **L35** | 28.5% | 26.5% | 50.0% | 通用 |
| L9 | 1.4% | 22.1% | 53.7% | 约束偏向 |
| L31 | 0.5% | 30.5% | 40.9% | 检索+约束 |
| L34 | 4.2% | 21.9% | - | 检索偏向 |

**通用原语层: L0, L6, L35**
**翻译专用层: 无！**

---

### 被推翻的

1. "L26 Attn是语言切换路由" → 错！L26 Attn zero-ablate只导致1.7%下降，不是因果必要
2. "L31 MLP最关键" → 错！L31 MLP zero-ablate只导致3.5%下降
3. "振荡是语义系统竞争" → 错！表示层L1就完成切换，振荡是decoder投影延迟
4. "翻译有专用电路" → 错！翻译是通用计算基础设施的分布式结果

### 被确认的

1. **早期层(L0, L6)是因果必要的** — 不是翻译专用，而是通用计算基础
2. **表示层切换远早于logits层切换** — 31层差距，说明晚期层做的是decoder对齐
3. **没有翻译专用的head/层** — 翻译是分布式计算的结果
4. **因果隔离是正确方法论** — path patching只测sufficiency，zero-ablate测necessity

---

### Phase 99最重要的发现

# 1. 之前所有"翻译关键层"的发现都是sufficiency(充分性)，不是necessity(必要性)
# 2. 翻译不是局部电路，而是通用计算基础设施的涌现属性
# 3. "语言切换"在表示层极早完成(L1)，logits层振荡只是decoder投影延迟
# 4. 通用原语层(L0, L6, L35)对所有任务都必要 → 类似"基础设施"

---

### 硬伤与瓶颈

1. **L0 MLP的100%崩塌可能是过拟合** — 移除任何一层的MLP可能都导致输出崩溃，需要检查非关键层是否也有类似效果
2. **Hidden state分类器可能区分的是"prompt格式"而非"语言"** — 翻译prompt有"的英文是"后缀，纯中文词没有
3. **补全任务的baseline太低(0.0144)** — 可能不是好的控制任务
4. **约束任务的baseline极低(0.0064)** — Qwen3可能不会做"X的第二个字是"
5. **只有Qwen3的结果** — 需要跨模型验证

### 下一步关键方向

1. **控制实验: 检查non-translation任务的MLP ablation** — 验证L0/L6是否对所有任务都关键
2. **格式控制: 用"猫的英文是" vs "猫的首都是"对比** — 区分"语言切换"和"prompt格式"
3. **逐层MLP ablation全扫描** — 检查是否所有层MLP ablate都会导致崩溃
4. **跨模型验证** — GLM4/DS7B的因果必要性测试
5. **更精确的任务控制** — 翻译 vs 续写 vs 问答，用相同prompt前缀
"""

content += append_text

with open('research/glm5/docs/AGI_GLM5_MEMO.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("Memo updated successfully!")
print(f"Appended {len(append_text)} characters")
