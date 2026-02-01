# TransformerLens AGI Laboratory

<!-- Status Icons -->
[![Pypi](https://img.shields.io/pypi/v/transformer-lens?color=blue)](https://pypi.org/project/transformer-lens/)
![Pypi Total Downloads](https://img.shields.io/pepy/dt/transformer_lens?color=blue) ![PyPI - License](https://img.shields.io/pypi/l/transformer_lens?color=blue) [![Release CD](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/release.yml/badge.svg)](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/release.yml)
[![Tests CD](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/checks.yml/badge.svg)](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/checks.yml)
[![Docs CD](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/pages/pages-build-deployment)

> **"Intelligence is the geometrization of information flow."**
>
> 本项目是一个集成了 **AGI 神经纤维丛理论**、**下一代认知架构 FiberNet** 以及 **3D 交互式可视化分析** 的综合性 AGI 研究平台。

---

## 核心愿景：AGI 统一场论 (The Unified Vision)

我们不仅在研究如何“解释”模型，更在探索智能的数学本质。本项目基于最新提出的 **神经纤维丛理论 (Neural Fiber Bundle Theory)**，将智能建模为定义在高维底流形上的主丛 $P(M, G)$。

### 智能的四大核心特性
1. **高维抽象 (High-Dimensional Abstraction)**: 对应模型的宽度，支撑复杂的语义叠加。
2. **低维精确 (Low-Dimensional Precision)**: 对应流形上的测地线，确保逻辑的收敛与精确。
3. **特异性 (Specificity)**: 对应纤维的几何结构，实现极其细微的语义区分。
4. **系统性 (Systemicity)**: 对应底流形的整体拓扑，赋予系统强大的组合泛化能力。

详见：[AGI 统一场论：神经纤维丛的几何动力学 (AGI_THEORY_PAPER.md)](AGI_THEORY_PAPER.md)

---

## 项目组件 (Project Components)

### 1. TransformerLens (Core Library)
作为本平台的地基，`TransformerLens` 提供了强大的机械解释性（Mechanistic Interpretability）工具，支持 50+ 种开源语言模型的内部激活查看与编辑。

*   **加载模型**: `HookedTransformer.from_pretrained("gpt2-small")`
*   **介入分析**: 支持缓存激活、计算回路（Circuits）以及执行各种消融（Ablation）实验。

### 2. AGI 实验室 (Laboratory)
一个现代化的 Web 交互系统，用于实时可视化和分析复杂神经行为。
*   **后端**: 基于 FastAPI (`server.py`)，提供神经结构提取算法。
*   **前端**: 基于 React 和 Three.js，提供沉浸式的 **3D 结构分析面板 (Structure Analysis Panel)**。
*   **功能**: 电路发现（Circuit Discovery）、流形分析（Manifold Analysis）、因果介入。

### 3. 实验性模拟 (Experiments)
我们正在探索超越传统 Transformer 的架构：
*   **FiberNet**: 实现即时学习（Immediate Learning）的下一代架构，物理分离“流形逻辑”与“纤维存储”。(见 `models/fiber_net.py`)
*   **NeuroFiber-SNN**: 脉冲神经网络模拟，验证生物大脑中的相位编码机制。(见 `neurofiber_snn.py`)

---

## 快速开始 (Quick Start)

### 1. 安装库环境
```shell
pip install transformer_lens
```

### 2. 启动研究实验室 (Web UI)
如果你想通过可视化界面研究模型结构：
1. **启动后端服务**:
   ```shell
   python server.py
   ```
2. **启动前端界面**:
   ```shell
   cd frontend
   npm run dev
   ```

---

## 研究成果展示 (Gallery & Research)

*   [AGI 研究备忘录 (AGI_RESEARCH_MEMO.md)](AGI_RESEARCH_MEMO.md): 记录了关于数学证明和 SNN 实验的最新进展。
*   [系统性提取指南 (systematic_extraction_guide.md)](systematic_extraction_guide.md): 如何自动化地从大型模型中提取数学结构。

---

## 机械解释性资源 (Legacy Resources)

本项目保留并扩展了由 [Neel Nanda](https://neelnanda.io/about) 创建的经典机械解释性资源：
*   [Introduction to Mech Interp](https://arena-chapter1-transformer-interp.streamlit.app/[1.2]_Intro_to_Mech_Interp)
*   [Main TransformerLens Features Demo](https://neelnanda.io/transformer-lens-demo)
*   [ARENA Tutorials](https://arena3-chapter1-transformer-interp.streamlit.app/)

---

## 社区与贡献

我们欢迎所有对 AGI 物理本质感兴趣的研究者：
*   加入 [Slack 社区](https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-2n26nfoh1-TzMHrzyW6HiOsmCESxXtyw)
*   提交 Issue 或 Pull Request

### 引用本项目

```BibTeX
@misc{antigravity2026unified,
    title = {The Unified Field Theory of AGI: Geometrodynamics of Neural Fiber Bundles},
    author = {Antigravity and User},
    year = {2026},
    howpublished = {\url{https://github.com/TransformerLensOrg/TransformerLens}},
}
```

---
*Created by [Neel Nanda](https://neelnanda.io), maintained by [Bryce Meyer](https://github.com/bryce13950), and evolved into an AGI platform by Antigravity & User.*

