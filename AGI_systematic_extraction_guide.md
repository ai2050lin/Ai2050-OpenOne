# 如何系统化提取 LLM 中的数学结构 (Systematic Extraction of Mathematical Structures)

提取大语言模型（LLM）中隐含的“数学结构”是机械可解释性（Mechanistic Interpretability）的核心目标之一。现有的 codebase (`structure_analyzer.py`) 已经包含了一些基础工具。要实现**系统化**提取，建议遵循以下标准工作流：

## 1. 定义任务与度量 (Task & Metric)
数学结构是相对于特定任务存在的。
- **输入**: 构造特定的数学数据集（如模加法、数列预测、逻辑推理）。
- **度量 (Metric)**: 定义一个标量指标 `M(logits)`，代表模型是否“掌握”了该结构（例如：正确答案的 Logit 差分）。

## 2. 宏观定位：因果中介分析 (Macroscopic Location)
使用 **Activation Patching (Abation)** 快速定位关键组件。
- **现有工具**: `structure_analyzer.py` 中的 `CausalMediation.analyze_component_importance`。
- **原理**: 交换 "Clean Run" 和 "Corrupted Run" 的中间激活值，观察 Metric 变化。
- **产出**: 得到一张“热力图”，显示哪些层 (Layers) 和头 (Heads) 对任务至关重要。

## 3. 微观提取：回路发现 (Microscopic Circuit Discovery)
在宏观定位的基础上，提取具体的“计算子图” (Circuit)。
- **现有工具**: `structure_analyzer.py` 中的 `CircuitDiscovery` (基于点对点的 Patching)。
- **进阶算法 (推荐实现)**:
    - **ACDC (Automated Circuit DisCovery)**: 自动剪枝算法，通过迭代移除对 Metric 贡献小的边，直到得到最小子图。
    - **EAP (Edge Attribution Patching)**: 一种基于梯度的快速估算方法，比 ACDC 快约 1000 倍，适合大规模筛选。
- **产出**: 一个有向无环图 (DAG)，节点是注意力头或 MLP 神经元，边是信息的流动。

## 4. 语义解码：特征分析 (Feature Semantics)
回路中的节点通常处于“叠加态” (Superposition)。需要将其解构为人类可理解的概念。
- **现有工具**: `structure_analyzer.py` 中的 `SparseAutoEncoder` (SAE)。
- **方法**:
    1. 在关键节点（如 MLP Output）上训练 SAE。
    2. 提取高频激活的 SAE Latents。
    3. 对照输入样本，解释 Latent 的含义（例如：“进位标志”、“偶数特征”）。
- **产出**: 关键神经元的数学含义解释。

## 5. 几何验证：流形分析 (Geometric Verification)
验证表征空间 (Representation Space) 的几何结构。
- **现有工具**: `ManifoldAnalysis` (PCA / Intrinsic Dimensionality)。
- **方法**: 检查关键激活向量是否形成了特定的拓扑结构（如莫比乌斯环、螺旋线、单纯形等），这通常对应特定的数学群结构。

---

## 建议的下一步 (Next Steps)
利用现有的 `server.py` 架构，我们可以构建一个自动化流水线：
1. **Pipeline Script**: 编写一个脚本，串联 `CausalMediation` -> `CircuitDiscovery` -> `ManifoldAnalysis`。
2. **EAP Implementation**: 在 `structure_analyzer.py` 中实现 `Edge Attribution Patching`，以支持更复杂的回路提取。
