"""追加Phase 426结果到MEMO"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 项目根目录
memo_path = ROOT / "research" / "glm5" / "docs" / "AGI_GLM5_MEMO.md"

timestamp = "2026-06-09 21:55"

entry = f"""

## Phase 426: 精细Alpha轨道边界扫描 [2026-06-09 21:55]

### 实验目标
精细扫描alpha(扰动强度)从0.02到2.0的范围，定位每个对象的临界跃迁阈值(basin boundary)。
解决Phase 425的核心硬伤：alpha太粗，Qwen3在0.5和1.0之间突然跳变。

### 实验设计
- 只选single-token对象(解决多token问题)
- alpha网格: 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00, 1.25, 1.50, 1.75, 2.00
- 扰动类型: remove_category, add_opposing, add_random, remove_identity
- 任务: category, property, part
- R1: 8对象×粗网格验证; R2: 11对象×19 alpha点

### 核心结果1: 临界Alpha对比

| 对象 | Qwen3 α_c | GLM4 α_c | DS7B α_c |
|------|-----------|----------|----------|
| apple (fruit) | **0.75** | **0.30** | 无跃迁 |
| orange (fruit) | 0.90 | 0.30 | 无 |
| knife (tool) | 0.90 | 0.30 | 无 |
| hammer (tool) | 1.00 | 0.40 | 无 |
| car (vehicle) | 1.00 | 无 | 无 |
| bus (vehicle) | 0.90 | 无 | 0.08(仅property) |

**平均临界Alpha: Qwen3=0.91, GLM4=0.30, DS7B=无稳定category跃迁**

### 核心结果2: 跃迁目标对比(alpha=1.0, remove_category, category任务)

| 对象 | Qwen3目标 | GLM4目标 | DS7B目标 |
|------|-----------|----------|----------|
| apple | fruit→**animal** | fruit→**place** | 不变(fruit) |
| orange | fruit→animal | fruit→tool | 不变 |
| knife | tool→vehicle | tool→place | 不变 |
| hammer | tool→vehicle | tool→place | 不变 |
| car | vehicle→animal | 不变 | 不变 |
| forest | place→animal | 不变 | 不变 |

### 核心结果3: Property任务受影响程度(alpha=1.0, |delta|均值)

| 模型 | mean|Δ| | max|Δ| | 说明 |
|------|---------|---------|------|
| Qwen3 | **0.029** | 0.263 | 几乎不受影响 |
| GLM4 | **0.977** | 2.987 | **强烈受影响!** |
| DS7B | 0.163 | 0.704 | 中等受影响 |

**GLM4中property受类别扰动影响是Qwen3的33倍!**

### 核心结果4: remove_identity效果(alpha=1.0, category, |delta|均值)

| 模型 | mean|Δ| | max|Δ| |
|------|---------|---------|
| Qwen3 | 0.375 | 2.050 |
| GLM4 | 0.427 | 1.863 |
| DS7B | 0.069 | 0.391 |

### 核心结果5: 精细Alpha曲线(apple/category/remove_category)

| Alpha | Qwen3_level | GLM4_level | DS7B_level | Q_top | G_top | D_top |
|-------|-------------|------------|------------|-------|-------|-------|
| 0.00 | 1.00 | 1.00 | 1.01 | fru | fru | fru |
| 0.10 | 1.00 | 1.00 | 1.05 | fru | fru | fru |
| 0.20 | 1.00 | 1.00 | 1.04 | fru | fru | fru |
| **0.30** | 1.00 | **3.89** | 1.03 | fru | **pla** | fru |
| 0.50 | 1.00 | 4.12 | 1.20 | fru | pla | fru |
| **0.75** | **1.78** | 4.21 | 1.07 | **ani** | pla | fru |
| 1.00 | 2.00 | 4.19 | 1.31 | ani | pla | fru |
| 2.00 | 2.00 | 4.17 | 1.23 | ani | pla | fru |

**关键观察: GLM4在alpha=0.3时发生突变(level从1.0直接跳到3.89), Qwen3在alpha=0.75时突变, DS7B永远不跃迁**

### 语义特异性比(alpha=1.0, |category Δ|/|random Δ|, category任务)

| 对象 | Qwen3 | GLM4 | DS7B |
|------|-------|------|------|
| apple | ∞ | ∞ | 2.2 |
| orange | ∞ | ∞ | 5.8 |
| knife | ∞ | ∞ | 20.9 |
| hammer | ∞ | 1.1 | 0.9 |
| car | ∞ | 0.7 | 0.6 |
| forest | ∞ | 0.1 | 1.5 |

**GLM4的特异性比对象依赖极大: fruit/tool极高, 但hammer/car/place极低**

### 客观现象总结

1. Qwen3的临界alpha=0.75-1.0, 跃迁后进入相邻类别(fruit→animal, tool→vehicle)
2. GLM4的临界alpha=0.2-0.3, 远低于Qwen3, 跃迁后进入非相邻类别(fruit→place, tool→place)
3. DS7B几乎不发生category跃迁, 即使alpha=2.0仍留在原类别
4. GLM4中property和category强耦合(delta=0.977), Qwen3中完全解耦(delta=0.029)
5. remove_identity对Qwen3和GLM4有中等效果, 对DS7B几乎无效
6. GLM4的语义特异性比极端对象依赖: fruit/tool极高, 但vehicle/place极低
7. Qwen3的alpha曲线显示硬相变: 0.5无效果, 0.75直接跳变
8. GLM4的alpha曲线也显示硬相变, 但在更小alpha处
9. DS7B的alpha曲线显示渐变: 永远不完全跃迁, 只是概率逐渐偏移

### 与Phase 425的对比

| 特征 | Phase 425 | Phase 426 |
|------|-----------|-----------|
| Alpha网格 | 0.5, 1.0, 2.0 | 0.02-2.0 (19点) |
| 多token问题 | 有 | 无(过滤) |
| Qwen3临界alpha | 0.5-1.0(粗) | **0.75-1.0**(精确) |
| GLM4临界alpha | 0.5-1.0(粗) | **0.2-0.3**(精确!) |
| Property受影响? | Q:无, G:有 | **确认:Q:无(0.029), G:强(0.977)** |
| Remove_identity | 未测 | Q:中(0.375), G:中(0.427), D:弱(0.069) |

### 问题与硬伤

1. **GLM4中某些对象(car, bus, train)不被remove_category影响**: 可能因为这些对象的category方向构造有问题, 或GLM4中这些对象不依赖嵌入类别方向
2. **DS7B的基线不准**: property基线是"alive"而不是"edible", 导致DS7B的结果解读困难
3. **随机方向对象特异性**: GLM4中hammer的random也有大效果, 但apple的random无效。这可能和随机种子有关
4. **只测了category/property/part三个任务**: 还需要颜色、味道、来源等具体属性

### 测试脚本
tests/glm5/phase426_alpha_basin_boundary.py
### 结果文件
results/phase426_alpha_basin_boundary/qwen3_phase426_r2.json
results/phase426_alpha_basin_boundary/glm4_phase426_r2.json
results/phase426_alpha_basin_boundary/deepseek7b_phase426_r2.json
"""

with open(memo_path, "a", encoding="utf-8") as f:
    f.write(entry)

print(f"MEMO updated. Entry added at {timestamp}")
