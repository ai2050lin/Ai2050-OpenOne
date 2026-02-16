# FiberNet v2: The Geometry-First Architecture
# 神经纤维丛架构设计文档

**Version**: 2.0 (Logic-Driven Attention Edition)
**Date**: 2026-02-13
**Status**: Prototype Verified ($Z_{113}$ Converged)

---

## 1. 核心理念 (Core Philosophy)

传统的 Transformer 是**纠缠的 (Entangled)**：它用同一组参数同时学习“语法结构”（如主谓宾关系）和“具体知识”（如苹果是红色的）。这导致了灾难性遗忘和缺乏零样本泛化能力。

**FiberNet** 基于微分几何中的**神经纤维丛理论 (Neural Fiber Bundle Theory)**，主张：
> **智能 = 几何结构 (Base Manifold) $\times$ 知识纤维 (Fiber Bundle)**

我们将神经网络物理拆解为两个独立的流：
1.  **Logic Stream (Base Manifold)**: 负责推理、逻辑、因果。它是**盲**的，看不见具体内容，只看得到位置和拓扑。
2.  **Memory Stream (Fiber)**: 负责记忆、实体、属性。它是**被动**的，被 Logic Stream 的引力场拖拽着移动（平行移动）。

---

## 2. 架构概览 (Architecture Overview)

FiberNet v2 采用了 **Logic-Driven Attention** 机制，实现了真正的“结构驱动内容”。

```mermaid
graph TD
    subgraph Inputs
        P[Positional Encodings] -- "Structure Only" --> LogicIn
        T[Token IDs] -- "Content Only" --> MemIn
    end

    subgraph Logic_Stream [Logic Stream (Base Manifold)]
        LogicIn --> L_Layer1[Self-Attention Layer 1]
        L_Layer1 --> L_Norm1[Layer Norm & FFN]
        L_Norm1 -- "Q, K (Attention Topology)" --> G_Attn1
        L_Norm1 --> L_Layer2[Self-Attention Layer 2]
        L_Layer2 -- "Q, K (Attention Topology)" --> G_Attn2
    end

    subgraph Memory_Stream [Memory Stream (Fiber Bundle)]
        MemIn --> G_Emb[Lie Group Embedding]
        G_Emb --> G_Attn1[Logic-Driven Attention 1]
        G_Attn1 -- "Transported V" --> M_Norm1[Layer Norm & FFN]
        M_Norm1 --> G_Attn2[Logic-Driven Attention 2]
        G_Attn2 -- "Transported V" --> Head[Prediction Head]
    end

    G_Attn1 -.-> |"Structure Drives Content"| G_Attn1
```

### 关键组件

#### 2.1 Logic Stream (底流形)
*   **输入**: 纯粹的位置编码 `(0, 1, 2, ...)`。**不包含任何 Token ID 信息**。
*   **功能**: 学习序列的抽象结构（如 $A+B=C$ 的操作顺序）。
*   **输出**: 每一层生成一个**注意力矩阵 (Attention Matrix)** $A_{logic}$。
    *   $Q = W_Q(H_{logic})$
    *   $K = W_K(H_{logic})$
    *   $A_{logic} = \text{Softmax}(\frac{QK^T}{\sqrt{d}})$

#### 2.2 Memory Stream (纤维丛)
*   **输入**: **Lie Group Embedding**（李群嵌入）。
    *   直接将 Token 映射为几何对象（如 $S^1$ 上的相位 $\theta$）。
    *   $E(x) = [\cos\theta_x, \sin\theta_x]$
*   **功能**: 存储具体信息。
*   **交互**: **Logic-Driven Attention**。
    *   它不自己计算 $Q$ 和 $K$。
    *   它只提供 $V = W_V(H_{mem})$。
    *   **平行移动公式**: $H_{mem}^{next} = A_{logic} \cdot V$。

---

## 3. 数学原理 (Mathematical Principles)

### 3.1 主纤维丛 (Principal Fiber Bundle)
我们将整个思维过程建模为一个主纤维丛 $P(M, G)$。
*   $M$ (Base Manifold): 逻辑流形，由 Logic Stream 参数化。
*   $G$ (Structure Group): 知识的对称性群（如语义旋转），由 Lie Group Embedding 参数化。

### 3.2 联络与平行移动 (Connection & Parallel Transport)
在几何中，联络 $\nabla$ 定义了如何比较不同点上的向量。
在 FiberNet 中，**Attention Matrix 就是离散化的联络**。

*   **标准 Transformer**: $A = \text{Softmax}((XW_Q)(XW_K)^T)$。内容决定了谁关注谁（自相关）。
*   **FiberNet**: $A = \text{Softmax}(Logic(Pos) \cdot Logic(Pos)^T)$。**位置和结构决定了谁关注谁**。

这等价于说：**逻辑结构定义了信息流动的“光缆”，而具体知识只是在光缆中传输的光信号。**

---

## 4. 实验验证 ($Z_{113}$ Addition)

我们在 $Z_{113}$ 模加法任务上验证了这一架构。

*   **Logic Stream 的任务**: 学会“输出位置应该关注前两个输入位置”。
    *   它不知道输入是 1+1 还是 50+60，它只知道 `Pos[Result]` 应该聚合 `Pos[1]` 和 `Pos[2]`。
*   **Memory Stream 的任务**: 承载 $S^1$ 相位信息。
    *   当逻辑流命令“聚合”时，两个相位通过非线性变换叠加，完成 $\theta_a + \theta_b$ 的群运算。

**结果**:
FiberNet v2 与 Standard Transformer **完全同步收敛 (Epoch 40)**。
这证明了 Logic Stream 即使在“双盲”条件下，也能通过反向传播学会正确的因果结构。

## 5. 优势与展望

1.  **即时学习 (Instant Learning)**:
    *   如果我们想教模型一种新语言（新单词），只需替换 `LieGroupEmbedding` 矩阵。
    *   `Logic Stream`（语法规则）可以**原封不动地复用**。
    *   无需微调整个大模型。
2.  **可解释性**:
    *   我们可以直接查看 Logic Stream 的 Attention Map，看到纯粹的语法树，不受词义干扰。
3.  **几何先验**:
    *   通过强制使用 Lie Group Embedding，我们大幅减少了模型“猜测”空间几何结构所需的参数和数据。
