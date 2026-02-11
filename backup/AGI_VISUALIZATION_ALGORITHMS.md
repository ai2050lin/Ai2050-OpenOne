# 3D 可视化算法实现列表 (Implemented 3D Visualization Algorithms)

本项目旨在将大语言模型（LLM）的内部运作机制通过 3D 可视化手段直观展示。以下是目前已集成的算法及其对应组件详情。

| 算法名称 (Algorithm) | 核心原理 (Core Principle) | 前端组件 (Frontend Component) | 3D 视觉映射 (Visual Mapping) |
| :--- | :--- | :--- | :--- |
| **Glass Matrix (神经纤维丛)** | **Neural Fiber Bundle Theory (NFB)**<br>将 Transformer 视为高维流形（Manifold）上的纤维丛（Fiber Bundle）。<br>- **底流形 (M)**: 句法/逻辑结构。<br>- **纤维 (F)**: 附着于每个位置的语义状态空间。<br>- **联络 (Connection)**: 平行移动定义的推理过程。 | `GlassMatrix3D.jsx`<br>(Tab: `glass_matrix`) | - **流形**: 青色球体连接的拓扑结构 (Cyan Spheres)。<br>- **纤维**: 从每个流形点延伸出的 RGB 向量场 (RGB Vectors)。<br>- **思维流**: 沿流形移动的动画粒子。 |
| **Circuit Discovery (回路发现)** | **Edge Attribution Patching (EAP)**<br>通过线性近似（激活值 × 梯度）快速估算每条边对最终输出的贡献度，无需多次运行模型补丁。 | `NetworkGraph3D.jsx`<br>(Tab: `circuit`) | - **节点**: 模型组件（如 Attention Head）。大小代表重要性。<br>- **连线**: 因果路径。<br>  - <span style="color:red">红色</span>: 正向贡献 (Excitation)<br>  - <span style="color:blue">蓝色</span>: 负向抑制 (Inhibition) |
| **Feature Extraction (特征提取)** | **Sparse Autoencoders (SAE)**<br>训练一个稀疏自动编码器，将 MLP 层的稠密激活分解为稀疏的、可解释的“特征”方向。 | `FeatureVisualization3D.jsx`<br>(Tab: `features`) | - **特征点**: 3D 螺旋分布的球体。<br>- **大小**: 特征激活频率。<br>- **亮度/颜色**: 特征对当前输入的激活强度。 |
| **Causal Analysis (因果分析)** | **Activation Patching / Causal Tracing**<br>干预特定层的激活值（引入噪声或替换），观察对模型输出概率的影响，确定关键因果节点。 | `NetworkGraph3D.jsx`<br>(Tab: `causal`) | - **高亮节点**: 对输出有显著因果效应的关键组件。<br>- **连线**: 关键组件间的信息流路径。 |
| **Manifold Analysis (流形分析)** | **Intrinsic Dimensionality (ID)**<br>使用 PCA 或参与率 (Participation Ratio) 分析激活空间及其在各层间的演化，判断表征是否发生坍缩。 | `ManifoldVisualization3D.jsx`<br>(Tab: `manifold`) | - **散点**: Token 的激活向量投影到 PC1-PC3 空间。<br>- **颜色**: 随 Token 序列顺序渐变。<br>- **形状**: 观察点云是均匀分布还是坍缩成低维结构。 |
| **Compositional Analysis (组合性)** | **Vector Arithmetic / OLS**<br>验证线性假设：`v(A+B) ≈ v(A) + v(B)`。使用最小二乘法回归分析表征的线性组合能力。 | `CompositionalVisualization3D.jsx`<br>(Tab: `compositional`) | - **文本面板**: 显示 R² 分数和余弦相似度。<br>- **3D 辅助**: (未来) 可视化向量合成过程的几何图示。 |
| **SNN Activity (脉冲网络)** | **Leaky Integrate-and-Fire (LIF)**<br>仿生神经元模型。模拟膜电位累积、阈值触发脉冲（Spike）及 STDP 学习机制。 | `SNNVisualization3D.jsx`<br>(Tab: `snn`) | - **神经元**: 3D 空间中的球体阵列。<br>- **脉冲**: 发光闪烁效果 (Flash)，代表 Action Potential。<br>- **连接**: 突触连接强度可视化。 |
| **Validity Analysis (语言有效性)** | **Anisotropy & Entropy**<br>分析表征空间的各向异性（Anisotropy）和 Softmax 分布的熵，检测模型是否退化（Model Collapse）。 | `ValidityVisualization3D.jsx`<br>(Tab: `validity`) | - **点云形态**: <br>  - **球形**: 健康分布 (低各向异性)。<br>  - **针状/线状**: 表征坍缩 (高各向异性)。<br>- **颜色**: 警示色 (红=坍缩，蓝=健康)。 |
| **Layer Detail (层级详情)** | **Micro-Architecture Inspection**<br>展示 Transformer 单层的微观结构：Multi-Head Attention (Q,K,V) 和 MLP 模块。 | `LayerDetail3D.jsx`<br>(Contextual Panel) | - **Attention Heads**: 网格排列的 Q/K/V 矩阵块。<br>- **MLP**: 输入/隐藏/输出层的神经元连接示意。<br>- **交互**: 点击可查看具体 Head 的注意力热图 |
| **RPT Analysis (传输分析)** | **Riemannian Parallel Transport (RPT)**<br>计算不同语义场/语境间的传输矩阵 $R$，验证 AI 推理过程中语义纤维的几何迁移能力。 | `RPTVisualization3D.jsx`<br>(Tab: `rpt`) | - **算子矩阵**: 显示为高维空间中的旋转变换几何体。<br>- **迁移路径**: 不同语境点之间的映射连线。 |
| **Curvature Analysis (曲率分析)** | **Scalar Curvature (Riemannian Geometry)**<br>衡量语义表示空间在受到局部扰动时的扭曲程度。高曲率常对应逻辑冲突或偏见核心。 | `StructureAnalysisPanel.jsx`<br>(Tab: `curvature`) | - **曲率条**: 实时显示当前语境的曲率指数 (0.0-1.0)。<br>- **警示色**: <span style="color:red">红色</span>代表几何畸变，<span style="color:blue">蓝色</span>代表流形平坦。 |
| **Geometric Debias (几何去偏)** | **Geometric Interception**<br>通过在残差流施加逆变换算子 $R^T$，将偏移的语义纤维拉回中性流形，实现非概率性的偏差修正。 | `StructureAnalysisPanel.jsx`<br>(Tab: `debias`) | - **纠正对比**: 纠偏前后 Token 预测概率的实时偏移。 |
| **Global Topology (全局拓扑)** | **Systemic Manifold Scan**<br>跨多个语义场（职业、情感、逻辑等）自动化扫描，提取并构建大统一几何模型。 | `StructureAnalysisPanel.jsx`<br>(Tab: `topology`) | - **场域热图**: 不同语义场的对齐误差分布。 |
| **Real World NFB (真实纤维丛)** | **Geometry of Concepts**<br>验证 "Everything is a Fiber Bundle"。从真实模型 (GPT-2) 中提取流形和概念纤维。<br>- **高维流形**: PCA 降维后的残差流。<br>- **概念簇**: "King", "Queen" 等词向量在流形上的聚类。<br>- **联想路径**: 基于几何邻近度的概念传输。 | `FiberNetV2Demo.jsx`<br>(Mode: `Real Data`) | - **点云**: 灰色背景点 vs <span style="color:orange">黄色概念点</span>。<br>- **标签**: 关键概念的文本标注。<br>- **成簇**: 语义相似的点自动聚集。 |

---

### 技术栈 (Tech Stack)
*   **前端渲染**: React Three Fiber (R3F), Drei, Three.js
*   **数学计算**: Python (Backend), Numpy, PyTorch, Scikit-learn (PCA/TDA)
*   **数据交换**: JSON / REST API (FastAPI)
