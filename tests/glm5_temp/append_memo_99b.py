import sys, os
sys.stdout.reconfigure(encoding='utf-8')

with open('research/glm5/docs/AGI_GLM5_MEMO.md', 'r', encoding='utf-8') as f:
    content = f.read()

append_text = """

---

### 控制实验补充: L6是翻译特异性最强层 [2026-05-09 01:15]

**MLP zero-ablate控制实验 (翻译 vs 补全):**

| 层 | 翻译drop | 补全drop | 翻译特异性 |
|---|---|---|---|
| L0 | 100% | 74.7% | 25.3% |
| L3 | 0.6% | -0.4% | 1.0% |
| **L6** | **100%** | **19.5%** | **80.5%** ← 最强! |
| L9 | 1.4% | 4.7% | -3.3% |
| L24 | 12.3% | -0.2% | 12.5% |
| L35 | 28.5% | -6.4% | 34.9% |

**Attn zero-ablate控制实验 (翻译 vs 补全):**

| 层 | 翻译drop | 补全drop | 翻译特异性 |
|---|---|---|---|
| L0 | 15.5% | 49.8% | -34.3% ← 补全更敏感 |
| **L6** | **79.8%** | **15.3%** | **64.5%** ← 翻译特异性最强! |
| L26 | 1.8% | -1.5% | 3.3% |
| L31 | 1.8% | 11.0% | -9.2% |

**核心结论:**
1. L6是翻译特异性最强的层 — Attn特异性64.5%, MLP特异性80.5%
2. L0是通用基础设施 — 对补全更敏感(Attn)或两者都敏感(MLP)
3. L26-31几乎无因果必要性 — 之前的path patching结论需要大幅修正
4. 翻译不是一个分布式涌现属性 — 有明确的因果必要组件(L6)
"""

content += append_text

with open('research/glm5/docs/AGI_GLM5_MEMO.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("Memo updated!")
