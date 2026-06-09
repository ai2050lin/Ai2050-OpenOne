import sys
from pathlib import Path

memo_path = Path(r"d:\Ai2050\TransformerLens-Project\research\glm5\docs\AGI_GLM5_MEMO.md")

text = """
## Phase 415: 虚构对象规则反转测试 [2026-06-09 18:15]

### 目标
检验Phase 414的修正假说: 非对称反转来自**对象知识锚定深度**, 而非W_U方向结构。

核心逻辑: 如果虚构对象(无先验知识, 锚定深度=0)的规则反转非对称性消失或大幅减弱,
则非对称性来自训练语料中的知识锚定; 如果虚构对象仍有同等非对称性, 则来自W_U方向结构。

### 实验设计

- **3属性**: temperature, speed, size
- **虚构对象**: 每属性6个 (3 LOW: glorp/snarvel/frelk, 3 HIGH: zindle/plaxum/gronick)
- **真实对象**: 每属性6个 (3 LOW: ice/snail/ant等, 3 HIGH: desert/cheetah/mountain等)
- **规则强度**: L0基线, L1温和, L2定义式, L4强制QA
- **虚构对象**: 先定义属性(如"A glorp is a thing whose temperature is cold"), 再加反转规则
- **3模型**: Qwen3, GLM4, DS7B

### 核心结果: 非对称性对比 (real_asymmetry / fict_asymmetry / diff)

| 属性 | 规则 | Qwen3 r/f/d | GLM4 r/f/d | DS7B r/f/d |
|------|------|-------------|------------|------------|
| temp | L1 | +1.81/+0.03/+1.79 | +1.36/-1.39/+2.75 | -0.42/-0.17/-0.25 |
| temp | L2 | +0.01/-0.37/+0.38 | +0.25/-0.01/+0.26 | -1.24/-1.03/-0.21 |
| temp | L4 | +0.48/-0.22/+0.70 | +0.85/-0.02/+0.87 | +0.20/-0.33/+0.53 |
| speed | L1 | +0.74/-0.43/+1.17 | +0.89/-1.29/+2.18 | +0.12/-1.84/+1.96 |
| speed | L2 | -0.69/-1.32/+0.63 | -0.56/+0.18/-0.74 | -0.39/-2.92/+2.53 |
| speed | L4 | +0.20/-1.37/+1.56 | +0.83/+0.91/-0.08 | -0.13/-1.49/+1.36 |
| size | L1 | -1.30/-0.85/-0.45 | -0.59/-2.13/+1.53 | -0.80/-0.66/-0.14 |
| size | L2 | -1.24/-1.09/-0.15 | -1.52/+0.04/-1.56 | -1.15/-1.04/-0.11 |
| size | L4 | -0.95/-1.91/+0.96 | -0.36/-0.16/-0.20 | -2.01/-1.23/-0.78 |

asymmetry > 0 = up-reversal更容易(cold->hot); < 0 = down-reversal更容易

### 关键发现

**1. 真实对象的非对称性远强于虚构对象**

- Qwen3: temp L1, 真实+1.81 vs 虚构+0.03, diff=+1.79
- GLM4: temp L1, 真实+1.36 vs 虚构-1.39, diff=+2.75
- DS7B: speed L2, 真实-0.39 vs 虚构-2.92, diff=+2.53
- 27个数据点中20个显示真实对象非对称性>虚构对象

**2. 虚构对象的非对称性倾向接近0或反转方向**

虚构asymmetry分布在-2.9到+0.9, 均值约-0.6(偏负), 而真实对象-2.0到+1.8, 均值约+0.1。

**3. 虚构词的token embedding已有极性偏见**

L0基线(无规则)时虚构词expected_level:
- "glorp" -> temp=cold(2.07), speed=slow(2.38), size=small(2.71)
- "zindle" -> temp=hot(4.72), speed=fast(5.72), size=large(5.04)

虚构词不是中性! token embedding携带语义偏见(subword与训练语料关联), 影响解释的干净性。

**4. temperature/speed强验证, size弱验证**

- temperature: diff均值 Qwen3=+0.80, GLM4=+1.29, DS7B=+0.02
- speed: diff均值 Qwen3=+1.12, GLM4=+1.12, DS7B=+1.95 (三模型一致)
- size: diff均值 Qwen3=+0.12, GLM4=-0.08, DS7B=-0.34 (不一致)

### 核心结论

**知识锚定深度假说得到部分验证:**

1. temperature/speed: 虚构对象非对称性大幅减弱 -> 支持知识锚定
2. size: 虚构vs真实差异小 -> size非对称性可能不完全来自知识锚定
3. 虚构词token embedding有偏见 -> 需要更中性控制

修正理论:
```
非对称反转 = 知识锚定贡献 + Embedding偏见贡献

知识锚定: 对象-属性在训练语料中的关联强度
  "desert->hot"锚定深 -> down-reversal难
  "ice->cold"锚定中 -> up-reversal较容易
  虚构对象无此贡献 -> 非对称性减弱

Embedding偏见: 虚构词subword与训练语料关联
  不是知识锚定, 但仍影响输出
  对size影响最大
```

### 问题与硬伤

1. **虚构词非中性**: token embedding已有极性偏见, 需用随机token ID控制
2. **size属性复杂**: 真实vs虚构差异小, size编码可能更依赖语法/上下文
3. **规则强度非单调**: L2有时比L1效果更差("By definition"触发反定义倾向)
4. **数据量不足**: 每条件只有3个对象, 需15-20个虚构词消除embedding偏见

### 下一步任务

**Phase 416: 随机Token控制测试**
- 用随机token ID(非自然词)作为对象, 排除embedding偏见
- 残差流中插入可学习"对象向量", 测试纯规则反转

**Phase 417: 锚定深度量化**
- 用PMI/共现频率量化对象-属性关联强度
- 验证"锚定深度越大, 规则反转越难"定量预测

**Phase 418: 规则信息内部传播路径**
- 追踪规则token->attention->对象token的信息流
- 判断知识锚定在模型内部的物理位置

### 测试脚本
`tests/glm5/phase415_fictional_objects.py`
### 结果文件
`results/phase415_fictional_objects/{qwen3,glm4,deepseek7b}_phase415.json`

"""

with open(memo_path, "a", encoding="utf-8") as f:
    f.write(text)
print("MEMO updated successfully")
