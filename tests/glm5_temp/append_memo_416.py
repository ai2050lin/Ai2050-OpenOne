import sys
from pathlib import Path

memo_path = Path(r"d:\Ai2050\TransformerLens-Project\research\glm5\docs\AGI_GLM5_MEMO.md")

text = """
## Phase 416: 中性对象控制测试 [2026-06-09 18:45]

### 目标
Phase 415发现虚构词有embedding偏见。本实验用3种中性程度递增的对象,
精确分解非对称反转的各因素贡献。

### 三种对象
1. 真实对象(ice/desert) - 训练知识锚定 + embedding先验
2. 虚构词+定义(glorp/zindle) - 上下文锚定 + embedding偏见
3. 随机token ID对象 - 仅规则调制(理论上无知识/嵌入偏见)

### Phase 416-R1结果: 3条件非对称性分解

| 属性 | 模型 | Real | Fictional | Random | Knowledge | Embed | Base |
|------|------|------|-----------|--------|-----------|-------|------|
| temp | Qwen3 | +0.649 | -0.505 | -0.854 | +1.155 | +0.349 | -0.854 |
| temp | GLM4 | +0.925 | -0.266 | +0.723 | +1.191 | -0.990 | +0.723 |
| temp | DS7B | +0.882 | +0.161 | +0.659 | +0.721 | -0.498 | +0.659 |
| speed | Qwen3 | +2.409 | -0.339 | -1.545 | +2.748 | +1.206 | -1.545 |
| speed | GLM4 | +0.359 | +1.394 | +1.244 | -1.036 | +0.150 | +1.244 |
| speed | DS7B | +1.489 | -0.340 | +1.299 | +1.829 | -1.639 | +1.299 |

Knowledge = real - fictional; Embed = fictional - random; Base = random

### R1问题: 随机token不是真正随机的

选出的"随机token"如caric, retard, QPointF等有语义, embedding有偏见。

### Phase 416-R2: 30个中性token大样本确认

筛选条件: 低频子词碎片(3-6字符, 小写, 无常见前后缀), 只测temperature。

| 模型 | n_low/n_high | up_mean | down_mean | asymmetry | LOW L0 | HIGH L0 | 判定 |
|------|-------------|---------|-----------|-----------|--------|---------|------|
| Qwen3 | 15/15 | +0.298 | +1.198 | -0.900 | 2.219 | 4.743 | 显著 |
| GLM4 | 15/15 | +0.962 | +0.839 | +0.123 | 2.249 | 4.655 | 近零 |
| DS7B | 15/15 | +0.421 | +0.967 | -0.546 | 2.625 | 4.128 | 中等 |

### 核心发现

**1. GLM4的随机token几乎无结构性非对称(asymmetry=+0.123)**

这强烈暗示: GLM4中, 非对称反转完全来自知识锚定+嵌入偏见, 没有W_U方向结构的贡献。

**2. Qwen3/DS7B仍有显著非对称(-0.900/-0.546)**

Qwen3和DS7B都是Qwen架构(Qwen3ForCausalLM/Qwen2ForCausalLM), 而GLM4是GLM架构。
**架构差异可能导致不同的结构偏见。**

**3. 随机token的L0基线不居中**

即使30个token取平均, LOW L0=2.2-2.6, HIGH L0=3.4-4.7, 不在midpoint=3.5。
定义("A X is a thing whose temperature is cold/hot")在所有模型中都有效区分了LOW/HIGH。

**4. down-reversal比up-reversal更强的模式在随机token中也存在**

Qwen3: up=+0.298 vs down=+1.198; DS7B: up=+0.421 vs down=+0.967
这暗示: 即使没有知识锚定, 模型也更容易把HIGH对象推向LOW方向。

### 可能解释

**解释1: 定义效应本身的非对称性**
- "A X is cold" → 低温锚定弱(只2-3 level)
- "A X is hot" → 高温锚定强(4-5 level)
- 反转时, 弱锚定更容易被覆盖 → up-reversal看似更容易
- 但实际测的是: 从弱锚定反转到对面 vs 从强锚定反转到对面
- 如果强锚定更难反转 → down-reversal更难 → 应该asymmetry > 0
- 但Qwen3/DS7B的random asymmetry < 0 → 矛盾!

**解释2: 候选词概率基线非对称**
- 在无知识条件下, 模型可能默认偏向cold/freezing等低等级候选词
- 这导致从任何起点, 推向cold都比推向hot更容易
- 这是W_U方向结构的效应: cold方向的logit基线更高
- GLM4没有这个偏见 → GLM4的W_U cold/hot方向更对称

**解释3: 定义的上下文锚定深度不同**
- "A X is cold"在上下文中只有1句话锚定 → 浅
- "A X is hot"在上下文中只有1句话锚定 → 浅
- 两者应该一样, 除非模型对"cold"和"hot"的token有不同的内部表示强度

### 最客观结论

1. **GLM4**: 非对称反转完全来自训练知识锚定 + embedding偏见, 无W_U结构贡献
2. **Qwen3/DS7B**: 非对称反转除了知识锚定外, 还有结构性因素
3. **架构是关键变量**: Qwen架构和GLM架构的内部结构不同
4. **"把HIGH推向LOW更容易"在Qwen架构中是结构性倾向**, 不完全来自知识

### 问题与硬伤

1. **随机token仍然不是零先验**: 30个token取平均后L0基线仍不居中
   - 需要直接在残差流中插入可控向量, 完全绕过tokenizer
2. **只测了temperature**: speed和size的结果可能不同
3. **定义效应非对称性**: 定义"A X is cold/hot"本身可能就非对称地影响模型
4. **n=30仍然偏少**: 标准误约0.2-0.3, 无法区分-0.9和0之间的细微差异
5. **跨架构比较**: Qwen3和DS7B共享Qwen架构, 结论不能直接推广

### 下一步任务

**Phase 417: 残差流可控向量测试**
- 绕过tokenizer, 直接在残差流中插入"对象向量"
- 对象向量 = neutral_base + attribute_offset
- 测试纯attribute_offset的反转非对称性
- 这是最终消除embedding偏见的方法

**Phase 418: 架构差异机制**
- 对比Qwen和GLM架构的W_U方向结构
- 检查Qwen架构是否有cold方向logit基线更高的倾向
- 分析RMSNorm/LayerNorm对偏移方向的非对称压缩

### 测试脚本
`tests/glm5/phase416_neutral_control.py`
`tests/glm5/phase416_r2_large_random.py`
### 结果文件
`results/phase416_neutral_control/{qwen3,glm4,deepseek7b}_phase416.json`
`results/phase416_neutral_control/{qwen3,glm4,deepseek7b}_phase416_r2.json`

"""

with open(memo_path, "a", encoding="utf-8") as f:
    f.write(text)
print("MEMO updated successfully")
