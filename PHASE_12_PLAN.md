# Phase 12: NLP 迁移与逻辑解耦验证

## 目标
验证 FiberNet 架构的核心优势：**逻辑 (Logic)与记忆 (Memory) 的物理解耦**。
如果 Logic Stream 确实只学习了“抽象句法”，那么它应该能直接应用于全新的语言（Memory Stream），只要该语言共享相同的句法结构 (S-V-O)。

---

## 实验设计

### 1. 数据集构造 (Tiny Bilingual SVO)
构造一个极简的合成数据集，包含 `Subject + Verb + Object` 结构的句子。

*   **英语 (Source)**:
    *   Subjects: *I, You, We, They*
    *   Verbs: *love, see, know, help*
    *   Objects: *him, her, it, them*
    *   E.g., "I love her", "We see them".
*   **法语 (Target)**:
    *   Subjects: *Je, Tu, Nous, Ils*
    *   Verbs: *aime, vois, sais, aide*
    *   Objects: *le, la, lui, les*
    *   E.g., "Je aime la" (Simplified for SVO consistency).

### 2. 模型设置
*   **DecoupledFiberNet**:
    *   **Logic Stream**: 输入 Positional Encoding。
    *   **Memory Stream**: 
        *   `EnglishEmbedding`: 随机初始化。
        *   `FrenchEmbedding`: 随机初始化。
    *   **注意**: Logic Stream 的参数在英语训练后**冻结**。

### 3. 步骤
1.  **Pre-train Logic (English)**:
    *   使用 `EnglishEmbedding` + `Logic Stream` 训练英语 SVO 任务。
    *   Logic Stream 应该学会：Pos[0] (Subj) -> Pos[1] (Verb) -> Pos[2] (Obj) 的依赖关系。
2.  **Zero-shot Transfer (French)**:
    *   保留训练好的 `Logic Stream`。
    *   **不进行任何微调**。
    *   直接换上未训练的 `FrenchEmbedding` (或者简单的对齐映射)。
    *   *修正*: 为了验证迁移，我们需要让 French Embedding 对齐到与 English Embedding 相同的几何空间，**或者**证明 Logic Stream 可以驱动任意 Embedding。
    *   *更强的测试*: 我们只训练 `FrenchEmbedding` 适配 `Logic Stream` 的信号，而不需要重新训练 Logic。这比从头训练快得多。

### 4. 预期结果
*   **Efficiency**: 固定 Logic Stream 后，仅训练 French Embedding 收敛速度极快。
*   **Decoupling**: 证明句法规则 (Logic) 可以跨语言复用。

## 脚本规划
*   `experiments/fibernet_nlp_transfer.py`
