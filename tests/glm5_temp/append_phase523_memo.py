"""Append Phase 523 results to AGI_GLM5_MEMO.md"""
import time

memo_path = "research/glm5/docs/AGI_GLM5_MEMO.md"
timestamp = time.strftime("%Y-%m-%d %H:%M")

entry = f"""

## Phase 523: Value Subspace Mapping & Cross-Category Generalization (价值子空间测绘与跨类别泛化) [{timestamp}]

### 一、实验目标

Phase 522 发现 random_ortho (seed=42) 在 GLM4 上达 35%，超过 d_plan (25%)。
但那只用了 1 个种子，无法判断 random_ortho 有效性是否稳定。

Phase 523 核心实验：
Exp1: 多种子正交方向测试 — 10 个随机正交方向，比较 d_plan vs random_ortho 分布
Exp2: 跨类别泛化测试 — fruit/vehicle/flower 三类别 3x3 转移矩阵

### 二、跨模型 Exp1: 多种子正交方向测试 (10 seeds, 10 test samples)

| 模型 | d_plan | random mean | random std | effective seeds | z-score | p-value | PCA dims (90%) |
|------|--------|------------|-----------|----------------|---------|---------|----------------|
| qwen3 | 3/10 (30%) | 0.7/10 (7%) | 1.0 | 4/10 | **2.29** | **0.011** | 4 |
| GLM4 | 2/10 (20%) | 0.4/10 (4%) | 0.7 | 3/10 | **2.41** | **0.008** | 3 |
| DS7B | 0/10 (0%) | 0/10 (0%) | 0 | 0/10 | 0.00 | 1.000 | N/A |

**关键结论：**

1. **d_plan 显著优于 random_ortho（qwen3 p=0.011, GLM4 p=0.008）**：这是最重要的发现，直接修正了 Phase 522 的结论。Phase 522 只用了 1 个种子 (seed=42)，该种子恰好命中了有效区域 (35%)。用 10 个种子测试后，random_ortho 的平均效果仅 4-7%，而 d_plan 达到 20-30%。

2. **Phase 522 "random_ortho > d_plan" 是单种子偶然现象**：GLM4 的 10 个 random_ortho 种子中，仅 3/10 有任何效果（最高 20%），而 d_plan 达到 20%。qwen3 的 10 个种子中 4/10 有效（最高 30%），但 d_plan 也达到 30%。少数种子的效果与 d_plan 相当，但大多数种子无效。

3. **价值子空间维度约 3-4 维**：PCA 分析显示有效方向需要 3-4 维解释 90% 方差（qwen3: 4维, GLM4: 3维）。这不是 1 维（单方向），也不是高维（完全分布式），而是一个低秩子空间。

4. **DS7B 完全免疫（0/10 seeds 有效）**：所有 10 个 random_ortho 种子均为 0%，与 d_plan 的 0% 相同。DS7B 的 embedding 层对任何方向的正交扰动都无反应。

5. **cos(d_traj, d_c) 继续确认 ~0.04-0.05**：三个模型一致，d_traj 99.9% 正交于 d_c。

### 三、跨模型 Exp2: 跨类别泛化测试 (fruit/vehicle/flower, 3x3 转移矩阵)

#### qwen3 转移矩阵

| d_traj(d) -> | fruit (within) | vehicle (cross) | flower (cross) |
|---|---|---|---|
| fruit | **38%** | 0% | 12% |
| vehicle | **50%** | 0% | 0% |
| flower | 12% | 0% | 0% |

Within mean: 12%, Cross mean: 12%

#### GLM4 转移矩阵

| d_traj(d) -> | fruit (within) | vehicle (within) | flower (within) |
|---|---|---|---|
| fruit | **25%** | 0% | 0% |
| vehicle | 12% | **25%** | **38%** |
| flower | 0% | 0% | 0% |

Within mean: 17%, Cross mean: 8%

#### DS7B 转移矩阵

全部 0%（所有 within 和 cross 均为 0%）

**关键结论：**

1. **跨类别转移存在但不一致**：qwen3 的 d_traj(vehicle) 对 fruit 失败达 50%（cross > within！），GLM4 的 d_traj(vehicle) 对 flower 失败达 38%（cross > within！）。但 d_traj(flower) 在所有模型上都几乎无效。

2. **within 不总是 > cross**：qwen3 的 within=12%, cross=12%（相等）；GLM4 的 within=17%, cross=8%（within 稍高）。d_traj 不是严格的类别特异方向，但也不是完全通用的。

3. **vehicle 的 d_traj 最强**：在 qwen3 和 GLM4 上，d_traj(vehicle) 都展示了最强的跨类别效果。这可能因为 "vehicle" 类别有更多失败样本（GLM4: 36 fail vs fruit 14 fail），d_traj 从更丰富的失败信号中提取了更强的一般性方向。

4. **所有 d_traj 对所有 d_c 近正交**：余弦矩阵显示 |cos(d_traj_src, d_c_tgt)| < 0.07 对所有 src/tgt 对成立。跨类别的正交性依然成立。

5. **DS7B 跨类别也完全无效**：所有 9 个转移测试均为 0%，进一步确认 embedding 层免疫。

### 四、Phase 523 成功标准评估

| # | 目标 | 状态 | 说明 |
|---|------|------|------|
| 1 | random_ortho 有效性是否跨种子稳定 | 已回答 | 不稳定：4/10 (qwen3), 3/10 (GLM4) 种子有效 |
| 2 | d_plan 是否显著优于 random_ortho | 达成 | qwen3 p=0.011, GLM4 p=0.008 |
| 3 | 价值子空间维度估计 | 达成 | 约 3-4 维 (PCA 90% var) |
| 4 | d_traj 是否类别特异 | 部分回答 | 部分通用：within 略 > cross 但差距不大 |
| 5 | DS7B 失败机制明确 | 已确认 | embedding 层完全免疫 (0% across all) |

### 五、核心发现的理论意义

#### 5.1 修正 Phase 522 的 "random_ortho > d_plan" 结论（最重要）

Phase 522 使用单个种子 (seed=42) 发现 GLM4 的 random_ortho=35% > d_plan=25%，得出 "d_plan 特异性有限" 的结论。

Phase 523 用 10 个种子证明这是一个偶然现象：
- GLM4: d_plan=20% vs random mean=4% (p=0.008)
- qwen3: d_plan=30% vs random mean=7% (p=0.011)

**d_plan 确实是特殊的**，它显著优于随机正交方向。Phase 522 的单种子结论需要修正。

#### 5.2 价值子空间是低秩的（3-4维）

PCA 分析显示有效方向集中在 3-4 维子空间中：
- qwen3: 5 个有效方向，4 维解释 90% 方差
- GLM4: 4 个有效方向，3 维解释 90% 方差

这既不是 1 维（单一方向假说被推翻），也不是高维完全分布式。价值信息编码在一个低秩子空间中，d_traj 是该子空间中的一个有效方向，少数随机方向也能落入该子空间。

#### 5.3 跨类别部分转移

d_traj 不是严格的类别特异方向。在某些情况下（如 vehicle -> fruit, vehicle -> flower），跨类别效果甚至超过 within 效果。这暗示 d_traj 可能编码了某种超越具体类别的通用生成规划信息。

但转移不一致（flower 的 d_traj 几乎无效），说明 d_traj 的有效性可能依赖于失败样本的类型和数量。

#### 5.4 DS7B 确认完全免疫

DS7B 在所有条件（10 seeds, 3 categories, within & cross）下均为 0%。这不是噪声，而是稳定现象。DS7B 的 embedding 层对任何方向的正交扰动都无反应。

### 六、客观现象拼图更新

```
56. d_plan 在 10 种子测试中显著优于 random_ortho（qwen3 p=0.011, GLM4 p=0.008），修正了 Phase 522 单种子结论。
57. 价值子空间是低秩的：约 3-4 维解释 90% 方差，既非 1 维也非高维分布式。
58. 10 个 random_ortho 种子中仅 3-4 个有效（30-40%），说明随机方向命中价值子空间的概率约 30-40%。
59. d_traj 存在部分跨类别转移：vehicle->fruit 达 50%(qwen3)，vehicle->flower 达 38%(GLM4)。
60. 跨类别转移不一致：flower 的 d_traj 几乎无效，说明 d_traj 有效性依赖于失败样本类型。
61. DS7B 在所有 10 seeds + 3 categories 条件下均为 0%，确认 embedding 层完全免疫。
62. 所有 d_traj 对所有 d_c 近正交（|cos| < 0.07），跨类别正交性依然成立。
```

### 七、测试命令记录

```bash
# qwen3 (2.4 min)
python tests/glm5/phase523_subspace_mapping.py qwen3

# GLM4 Exp1 (24 min, 从日志提取结果)
python tests/glm5/phase523_subspace_mapping.py glm4

# GLM4 Exp2 (31 min, --skip-exp1)
python tests/glm5/phase523_subspace_mapping.py glm4 --skip-exp1

# DS7B (22.6 min)
python tests/glm5/phase523_subspace_mapping.py deepseek7b
```

### 八、结果文件

- `results/glm5_phase523_subspace_mapping/phase523_qwen3_subspace_mapping.json`
- `results/glm5_phase523_subspace_mapping/phase523_glm4_subspace_mapping.json`
- `results/glm5_phase523_subspace_mapping/phase523_deepseek7b_subspace_mapping.json`

### 九、下一步：Phase 524

基于 Phase 523 的发现，下一步应聚焦于：

1. **中间层 forward_from_layer 闭环** — DS7B 在 embedding 层完全无效，必须在中间层施加 d_traj。这是从 "embedding steering" 走向 "内部机制因果闭环" 的关键。实现 KV-cache aware 的中间层干预。

2. **路径价值 probe** — 训练 probe 从 hidden state 预测 V_c(y|h)，验证独立于 logit 的价值编码。用 probe 梯度构造 d_value，比较 d_value vs d_traj。

3. **U_plan 子空间构造** — 从多个有效方向（d_plan + 有效 random_ortho 方向）构造 U_plan 子空间，测试 U_plan 投影是否比单个 d_plan 更稳定。

4. **失败模式分类** — 对失败样本聚类，解释为什么 d_traj 对某些失败有效（如 "犹豫型失败"）而对其他无效（如 "知识缺失型失败"）。

5. **跨类别 d_traj 转移机制** — 为什么 vehicle 的 d_traj 比 flower 的更强？是否与失败样本数量/类型有关？

### 十、总结

Phase 523 完成了价值子空间测绘和跨类别泛化测试：

1. **d_plan 显著优于 random_ortho**（qwen3 p=0.011, GLM4 p=0.008），修正了 Phase 522 的单种子结论。d_plan 确实是特殊的正交规划方向。

2. **价值子空间是低秩的**（约 3-4 维），既非 1 维也非高维分布式。少数随机方向（30-40%）能落入该子空间。

3. **跨类别部分转移**：d_traj 不是严格的类别特异方向，某些跨类别效果甚至超过 within 效果（vehicle->fruit 50%, vehicle->flower 38%）。但转移不一致。

4. **DS7B 完全免疫**：所有条件（10 seeds, 3 categories）均为 0%，确认 embedding 层完全无效，需要中间层干预。

最重要的发现是 **d_plan 在多种子测试中显著优于 random_ortho**，这修正了 Phase 522 "d_plan 特异性有限" 的结论。价值子空间是 3-4 维低秩子空间，d_traj 是其中的一个有效方向。

"""

with open(memo_path, "a", encoding="utf-8") as f:
    f.write(entry)

print(f"Appended Phase 523 entry to {memo_path}")
print(f"Timestamp: {timestamp}")
