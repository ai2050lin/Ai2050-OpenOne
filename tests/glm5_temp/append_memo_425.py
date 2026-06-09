"""
追加Phase 425结果到AGI_GLM5_MEMO.md
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
memo_path = ROOT / "research" / "glm5" / "docs" / "AGI_GLM5_MEMO.md"

now = datetime.now().strftime("%Y-%m-%d %H:%M")

content = f"""

## Phase 425: 词嵌入成分扰动与知识轨道映射 [{{now}}]

### 实验原理

对真实对象(apple, dog, knife, car, desert等)的词嵌入进行可控扰动，观察轨道如何变化。

**核心问题**: 对象词的初始embedding中，类别方向成分是否因果性地决定对象的类别归属？

**扰动类型**:
- add_category: 加上自身类别方向（应增强类别信号）
- remove_category: 减去自身类别方向（应削弱类别信号）
- add_opposing: 加上对立类别方向（应推向对立轨道）
- add_random: 加上随机正交方向（对照，排除范数效应）

**知识槽位任务**:
- category: "A X is a kind of ___" (fruit/animal/tool/vehicle/place)
- property: "The most notable property of a X is that it is ___" (edible/alive/sharp/fast/vast)
- part: "A X has ___" (seeds/fur/blades/wheels/sand)

### R2结果（10对象 × 3任务 × 4扰动 × 3强度）

#### 发现1: Qwen3类别方向有语义特异性

| 扰动 | Qwen3 category |delta| | GLM4 category |delta| | DS7B category |delta| |
|------|-------------|-------------|-------------|
| add_category | 0.016 | 0.502 | 0.138 |
| remove_category | 0.838 | 1.008 | 0.138 |
| add_opposing | 0.674 | 0.957 | 0.151 |
| add_random | **0.026** | **0.937** | 0.176 |

Qwen3: 类别方向扰动 >> 随机方向扰动 (0.838 vs 0.026, ratio=32x)
→ 类别方向是语义特异的

GLM4: 类别方向扰动 ≈ 随机方向扰动 (1.008 vs 0.937, ratio=1.1x)
→ GLM4对任何嵌入扰动都极度敏感，类别方向没有特异性

DS7B: 几乎对所有扰动都不敏感 (all ~0.14)
→ DS7B的类别归属不主要由嵌入成分决定

#### 发现2: 移除类别方向后，对象进入哪个轨道？(跨模型差异)

| 对象 | 原类别 | Qwen3 remove→ | GLM4 remove→ | DS7B remove→ |
|------|-------|--------------|-------------|-------------|
| apple | fruit | animal | **place** | fruit(不变) |
| orange | fruit | animal | **place** | fruit(不变) |
| dog | animal | animal(不变) | **fruit** | animal(不变) |
| horse | animal | animal→fruit(a2) | **fruit** | animal(不变) |
| knife | tool | vehicle | **vehicle** | tool(不变) |
| scissors | tool | vehicle | **vehicle** | tool(不变) |
| car | vehicle | vehicle(不变) | **fruit** | vehicle(不变) |
| bicycle | vehicle | vehicle→tool(a2) | **fruit** | vehicle(不变) |
| desert | place | **animal** | place(不变) | place→fruit(a2) |
| ocean | place | **animal** | place(不变) | place(不变) |

**Qwen3模式**: 移除类别后进入**相邻类别** (fruit→animal, tool→vehicle)
**GLM4模式**: 移除类别后进入**非相邻类别** (fruit→place!, car→fruit!, animal→fruit)
**DS7B模式**: 移除类别后**几乎不变**

#### 发现3: 属性知识(property)不存储在类别方向中

三模型中，property任务对remove_category扰动的delta都接近0:
- Qwen3: property remove_category delta ≈ 0
- GLM4: property remove_category delta ≈ 0 (除alpha=0.5时偶发跳变)
- DS7B: property对扰动更敏感，但不特定于类别方向

**说明**: "edible/alive/sharp"等属性知识不在类别方向的嵌入成分中，而在其他成分或后续层参数中。

#### 发现4: 轨道捕获模式

Qwen3的轨道捕获是**相邻吸引**: fruit↔animal, tool↔vehicle, place→animal
GLM4的轨道捕获是**非邻吸引**: 多个类别直接跳到place或fruit
DS7B几乎没有轨道捕获效应

### 客观现象总结（不加理论）

1. Qwen3的类别方向具有32倍语义特异性（vs随机方向），GLM4没有（1.1倍）
2. Qwen3移除类别方向后对象进入相邻类别轨道，GLM4进入非相邻类别轨道
3. DS7B的类别归属几乎不受嵌入扰动影响
4. 属性知识（edible/alive等）不存储在类别方向的嵌入成分中
5. GLM4对任何嵌入扰动都高度敏感，说明GLM4的内部表示更脆弱
6. 轨道吸引盆结构不同：Qwen3相邻吸引，GLM4非邻吸引

### 问题与硬伤

1. **只修改了第一个token的embedding**: 如果对象词被分为多个token（如"bi"+"cycle"），只修改了第一个token
   - 对多token对象可能低估扰动效果

2. **类别方向用词嵌入均值构造，可能有偏**: 
   - d_fruit = mean(E[apple,banana,...]) - mean(E[dog,cat,...])
   - 这个方向可能不是模型内部真正的类别轴

3. **alpha=1.0对Qwen3已经饱和**:
   - Qwen3中alpha=0.5几乎没有效果，alpha=1.0就完全跳到新轨道
   - 存在临界阈值，需要更精细的alpha扫描

4. **GLM4的随机方向也很强**:
   - 这可能说明GLM4的嵌入空间更"脆"，而非类别方向不特异
   - 需要更小alpha（0.1-0.5）的精细扫描来区分

5. **DS7B的基线就不准确**:
   - DS7B把apple的property判断为"alive"而非"edible"
   - 这说明DS7B的知识表示和其他两模型本质上不同

6. **只测了3个任务，没有颜色/味道等具体属性**:
   - 需要更多属性维度来理解哪些知识在嵌入中，哪些不在

### 测试脚本
tests/glm5/phase425_embedding_perturbation.py
### 结果文件
results/phase425_embedding_perturbation/qwen3_phase425_r1.json (etc.)
results/phase425_embedding_perturbation/qwen3_phase425_r2.json (etc.)
"""

with open(memo_path, "a", encoding="utf-8") as f:
    f.write(content)

print(f"MEMO updated at {now}")
print(f"Path: {memo_path}")
