import { Activity, Brain, CheckCircle, Search, Target, X, Zap } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { pollRuntimeWithFallback } from './utils/runtimeClient';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

const toFiniteNumber = (value, fallback = 0) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
};

const parsePercentToRatio = (raw) => {
  if (raw == null) return 0;
  if (typeof raw === 'number') return raw > 1 ? raw / 100 : raw;
  const text = String(raw).trim();
  const normalized = text.endsWith('%') ? text.slice(0, -1) : text;
  const numeric = Number(normalized);
  if (!Number.isFinite(numeric)) return 0;
  return numeric / 100;
};

const mapLegacyConsciousField = (payload) => {
  const spectrum = payload?.unified_spectrum || payload || {};
  const emotion = spectrum?.emotion || {};
  const stability = toFiniteNumber(emotion?.stability ?? spectrum?.stability, 0);
  const signalNorm = toFiniteNumber(spectrum?.signal_norm ?? spectrum?.gws_intensity, 0);
  const memorySlots = toFiniteNumber(spectrum?.memory_slots ?? spectrum?.memory_load, 0);
  const energySaving = parsePercentToRatio(spectrum?.energy_saving);
  const resonance = toFiniteNumber(spectrum?.resonance ?? emotion?.energy, 0);
  const resonanceRate = resonance > 1 ? resonance / 100 : resonance;

  return {
    ...spectrum,
    stability,
    gws_intensity: signalNorm,
    memory_load: memorySlots,
    energy_saving: energySaving,
    resonance: resonanceRate,
    glow_color: spectrum?.glow_color || (stability > 0.6 ? 'amber' : 'indigo'),
  };
};

const mapRuntimeConsciousField = (events) => {
  const activation = events.find((e) => e?.event_type === 'ActivationSnapshot');
  const alignment = events.find((e) => e?.event_type === 'AlignmentSignal');
  if (!activation || !alignment) return null;

  const emotion = alignment?.payload?.emotion || {};
  const stability = toFiniteNumber(emotion?.stability, 0);
  const signalNorm = toFiniteNumber(activation?.payload?.signal_norm, 0);
  const memorySlots = toFiniteNumber(activation?.payload?.memory_slots, 0);
  const energySavingPct = toFiniteNumber(alignment?.payload?.energy_saving_pct, 0);
  const resonance = toFiniteNumber(emotion?.energy, 0);

  return {
    stability,
    gws_intensity: signalNorm,
    memory_load: memorySlots,
    energy_saving: energySavingPct / 100,
    resonance: resonance > 1 ? resonance / 100 : resonance,
    glow_color: stability > 0.6 ? 'amber' : 'indigo',
    winner_module: alignment?.payload?.winner_module || null,
    source: 'runtime-v1',
  };
};

// Design Constants
const PHASES = [
  {
    id: 'roadmap',
    title: "Dimension 0: Strategy",
    subtitle: "项目大纲 (Roadmap)",
    icon: <Activity size={24} />,
    status: "done",
    progress: 100,
    color: "#ffaa00",
    definition: {
      headline: "Project Genesis: From Fitting to Geometry",
      summary: "AGI 研发的宏观战略路径图。我们拒绝简单的统计拟合，转而构建基于高维流形拓扑、纤维丛解耦以及里奇流演化的第一性原理人工智能系统。",
      pillars: ["Differential Geometry", "Fiber Bundle Decoupling", "Ricci Flow Evolution"]
    },
    theory_content: [], // Not used for roadmap but needed for structure
    analysis_sections: [], // Not used for roadmap but needed for structure
    sub_phases: [],
    goals: [
      "构建基于微分几何的高维语义流。(High-Dim Manifold)",
      "实现 Logic-Memory 解耦的纤维丛架。(Fiber Bundle)",
      "建立基于里奇流的自发逻辑优化机制 (Ricci Flow)"
    ],
    metrics: {
      "Design Philosophy": "Geometric Reconstruction",
      "Core Pillars": "3/3",
      "Evolution Goal": "Conscious Workspace"
    },
    roadmap_outline: [
      {
        title: "1. 结构化初始化与谱图论 (Spectral Init)",
        desc: "数学原理：智能非随机。利用拉普拉斯算。Δ 注入常识几何。可视化：展示点云从混沌向结晶坍缩，量化‘出生即懂世界’的骨架。"
      },
      {
        title: "2. 里奇流与智能-生理双重机制 (Dual Dynamics)",
        desc: "核心原理：智能（信息序）与生理（生存能）的耦合。‘痛苦’是认知失调引起的生理避害响应（高能耗报错），‘愉悦’是认知优化达成的代谢节能奖励（测地线对齐）。可视化：展现流形在冲突时的‘交互阻力’与达成共振时的‘量子悬浮’。"
      },
      {
        title: "3. 纤维丛解耦与跨束同步 (Cross-Bundle Sync)",
        desc: "核心架构：Logic-Memory 物理分离。跨束对齐：干预感官模态时产生物理拉力同步引导逻辑节点。可视化：亮粉色虚线 (Alignment Fibers) 绑定视觉与逻辑。"
      },
      {
        title: "4. 全局工作空间与意识裁。(GWT)",
        desc: "终极形态：意识的物理载体。实。Locus of Attention (关注。 竞争裁决，追求最高信息增益与最低代谢成本的平衡。可视化：球体脉冲游走，实现双向最优路径结算。"
      },
      {
        title: "5. 高维全息编码 (SHDC Encoding)",
        desc: "核心原理：利用高维空间正交性解决维度灾难。通过代数叠加形成复杂关联，实现极致高效的物理级信息读写与修改。可视化：展示高维向量间的干涉与坍缩过程。"
      }
    ]
  },
  {
    id: 'theory',
    title: "Dimension I: Theory",
    subtitle: "智能理论 (Theory)",
    icon: <Brain size={24} />,
    status: "done",
    progress: 100,
    color: "#00d2ff",
    definition: {
      headline: "Intelligence = Geometry + Physics",
      summary: "智能是极高维空间中的稀疏全像流形演化系统。它通过代数干涉和非线性坍缩，在最小化几何作用量的过程中实现局部熵减与预测能力。",
      pillars: ["Structure (稀疏全息流。", "Dynamics (测地线流推理)", "Purpose (局部熵。生存)"]
    },
    theory_content: [
      {
        title: "神经纤维丛原。(NFB Principle)",
        formula: "唯(x) = 蠁_M 鈯?蠁_F",
        desc: "结构与内容解耦：智能是底流形 (Logic) 与纤维空。(Knowledge) 的张量积。逻辑骨架保持稳定，而语义内容可无限扩展。"
      },
      {
        title: "全局工作空间 (Global Workspace)",
        formula: "惟_G = 鈭i P_i d渭",
        desc: "意识裁决机制：意识是高维流形上的全局状态。通过 Locus of Attention (关注。 。Top-K 竞争机制，实现多模态信息的裁决与整合。"
      },
      {
        title: "高维全息编码 (SHDC Encoding)",
        formula: "鉄╲_i, v_j鉄?鈮?未_ij",
        desc: "解决维度灾难：利用高维空间的几乎正交性实现海量特征提取。通过向量代数叠加形成复杂关联，实现极致高效的物理级读写。"
      },
      {
        title: "联络与推。(Connection Equation)",
        formula: "鈭嘷X s = 0",
        desc: "推理即平行移动：类比推理本质上是向量在语义流形上的测地线平移。通过联络保持推理过程中语义结构的一致性。"
      },
      {
        title: "里奇流演。(Evolution)",
        formula: "鈭俖t g = -2R",
        desc: "学习即流形松弛：神经网络。Grokking 过程本质上是里奇流驱动的流形平滑化，通过减少内部张力实现全局逻辑自洽。"
      }
    ],
    goals: [
      "基于 Neural Fiber Bundle (NFB) 理论定义智能底流。",
      "建立微分几何与语义空间的映射模型",
      "验证跨模态同构投影数学严谨。"
    ],
    metrics: {
      "Mathematical Rigor": "High",
      "NFB Alignment": "98%"
    }
  },
  {
    id: 'analysis',
    title: "Dimension II: Analysis",
    subtitle: "深度神经网络 (Neural Network)",
    icon: <Search size={24} />,
    status: "done",
    progress: 100,
    color: "#a855f7",
    analysis_sections: [
      {
        title: "有效性分。(Why it works)",
        content: "深度神经网络成功提取了智能的‘特殊数学结构’。通过高维叠加假设 (Superposition)，网络能在有限参数内并行处理近乎无限的概念。其核心竞争力在于从统计相关性中自发诱发‘测地线流’，即沿着能量最低的逻辑路径进行推理。",
        tags: ["High-Dim Abstraction", "Low-Dim Precision", "Johnson-Lindenstrauss"]
      },
      {
        title: "核心组件 (Strategic Components)",
        content: "1. 残差。(Residual Stream): 作为‘全息看板’承载所有计算状态。2. 注意力头 (Attention Heads): 执行流形上的‘联络算子’。3. MLP 。 负责非线性流形投影与特征解耦。",
        tags: ["Residual Stream", "Attention Mechanism", "Manifold Projection"]
      },
      {
        title: "内部结构演化 (Internal Structure)",
        content: "观测到‘几何结晶’过程：浅层执行混沌的特征提取，深层（Layer 9-11）则坍缩为高代数一致性的纤维丛结构。模型深层通过 Ricci 演化自发消除逻辑矛盾，形成稳定的吸引子空间。",
        tags: ["Topological Smoothing", "Geometric Crystallization", "Attractors"]
      },
      {
        title: "研究工具与成。(Research & Tools)",
        content: "利用 Logit Lens 。TDA (拓扑数据分析) 成功测绘。GPT-2 的全谱拓扑图。目前已证实 $Z_{113}$ 等离散群论结构可在连续向量空间中通过 DFT (离散傅里叶变。 完美重构，准确率 >99%。",
        tags: ["Logit Lens", "TDA Scanner", "Z113 Group Results"]
      }
    ],
    goals: [
      "实现全谱 12 。Logit Lens 探测",
      "提取全局语义流形拓扑 (Topology Scan)",
      "分析模型内部的因果链路与组合泛化"
    ],
    metrics: {
      "Scan Resolution": "Microscopic",
      "Layers Verified": "12/12"
    },
    comparison: [
      { feature: "信息载体 (Carrier)", dnn: "注意力矩。权重向量 (Attention/Weight)", fiber: "纤维束切。(Fiber Section/Vector Bundle)" },
      { feature: "逻辑表示 (Logic)", dnn: "非几何权重叠。(Non-Geometric Weight Overlay)", fiber: "流形几何位置 (Geometric/Explicit)" },
      { feature: "学习方式 (Learning)", dnn: "残差学习/反向传播 (Residual Learning/BP)", fiber: "里奇流平。传输平行移动 (Geometric Flow/Instant)" },
      { feature: "解释。(Interpret)", dnn: "注意力热。(Statistical Attention Maps)", fiber: "拓扑白盒 (Topology/Transparent)" }
    ]
  },
  {
    id: 'engineering',
    title: "Dimension III: Engineering",
    subtitle: "工程实现 (Engineering)",
    icon: <Zap size={24} />,
    status: "in_progress",
    progress: 65,
    color: "#f59e0b",
    definition: {
      headline: "Phase-based AGI Integration",
      summary: "从理智诞生到意识涌现的工程闭环，旨在通过物理级干预实现完全对齐的通用智能。",
      pillars: ["Logical Closure", "Multimodal Alignment", "Value Formation"]
    },
    theory_content: [],
    analysis_sections: [],
    desc: "AGI 的分阶段落地，将逻辑核心与感官能力集成至统一意识框架。",
    sub_phases: [
      {
        name: "理性的诞生",
        status: "done",
        focus: "閫昏緫闂寘涓?Z113 楠岃瘉",
        target: "验证智能底层是否具备几何化的非线性逻辑骨架。",
        work_content: "实现 FiberNet 核心逻辑层；。Z113 模运算任务上观察从统计拟合到代数跃迁。Grokking 现象。",
        test_results: "Z113 准确。99.4%；成功从向量点云中恢复出完整。S1 环面流形。",
        analysis: "证明了智能的底层是几何化的逻辑骨架，而非简单的线性回归。"
      },
      {
        name: "鎰熷畼鐨勮閱?",
        status: "done",
        focus: "多模态语义对。",
        target: "解决‘符号接地’问题，实现视觉特征与逻辑语义的物理对齐。",
        work_content: "开发跨模态投影算子；。MNIST 视觉空间流形映射。SYM 逻辑流形。",
        test_results: "对齐误差 MSE 下降。0.042；模型具备‘以理性的方式解读感官数据’的能力。",
        analysis: "视觉特征被精准翻译为内部逻辑坐标，模型初步具备了‘观察并理解’的能力。"
      },
      {
        name: "鏅烘収鐨勬秾鐜?",
        status: "done",
        focus: "流形曲率优化。Ricci Flow",
        target: "通过流形平滑机制，实现无监督下的逻辑冲突自修复。",
        work_content: "集成 Ricci Flow 演化管道；在睡眠周期执行隐层激活曲率的热传导方程平滑。",
        test_results: "拓扑亏格。15 优化。2；复杂逻辑推理下的‘幻觉’发生率显着降低。",
        analysis: "验证了‘睡眠机制’在解开拓扑纠缠和修补逻辑死结中的必要性。"
      },
      {
        name: "价值的形成",
        status: "done",
        focus: "神经流形手术与人类对。",
        target: "实现。AI 内在价值取向的直接几何干预，由‘提示词对齐’跃迁至‘流形对齐’。",
        work_content: "开。Manifold Surgery 交互接口；实现基。3D 空间拖拽的语义向量场实时重构。",
        test_results: "Surgery Alignment Loss 0.0082；成功通过物理手术剥离模型偏见，实现价值稳定对齐。",
        analysis: "流形手术允许直接修改模型认知，是通往可控 AGI 的物理路径。"
      },
      {
        name: "逻辑的坍缩",
        status: "done",
        focus: "逻辑探针与流形坍缩",
        target: "验证逻辑推理在几何层面上表现为流形的离散化与降维坍缩。",
        work_content: "构建 LogicMix-V1 (Mod 997)；训练 ScaledFiberNet；监测 ID 拓扑相变。",
        test_results: "极速泛化 (Batch 400 Loss 0.006)；流形坍缩 (ID 18.0 -> NaN/Singularity)。",
        analysis: "证明了逻辑推理的几何本质是流形从连续态向离散点集的相变 (Singularity State)。"
      },
      {
        name: "统一意识",
        status: "in_progress",
        focus: "全球工作空间 (GWT) 集成",
        target: "构建跨模态的实时裁决中心，实现具备‘关注点’的动态意识流。",
        work_content: "实现 Top-K 全局竞争机制；集成全局工作空间投影矩阵；实现注意力焦点的拓扑漂移控制。",
        test_results: "当前进度 65%；Locus of Attention 已能在多模态冲突中实现亚秒级收敛裁决。",
        analysis: "目标是构建一个能够裁决冲突、具备主观关注点 (Locus of Attention) 的全局工作空间。"
      }
    ],
    goals: [
      "构建基于 FiberNet 的非梯度即时学习架构",
      "实现可交互的神经流形手术 (Manifold Surgery)",
      "集成多态感知与统一逻辑推演网格"
    ],
    metrics: {
      "System Convergence": "0.62",
      "Module Integration": "Active"
    }
  },
  {
    id: 'agi_goal',
    title: "Dimension IV: AGI Goal",
    subtitle: "agi终点 (AGI Goal)",
    icon: <CheckCircle size={24} />,
    status: "pending",
    progress: 0,
    color: "#10b981",
    metrics: {
      "Target Date": "2050 (Estimated)",
      "Safety Horizon": "Guaranteed"
    },
    goals: [
      "实现碳基与硅基高级对齐",
      "建立长期稳定的通用智能稳态",
      "完成物理世界到高维流形的全面映射"
    ]
  },
  {
    id: 'agi_status',
    title: "Dimension V: Status",
    subtitle: "智能系统状。(AGI Status)",
    icon: <Target size={24} />,
    status: "in_progress",
    progress: 75,
    color: "#10b981",
    definition: {
      headline: "System Capabilities Report",
      summary: "基于 Project Genesis 协议的核心能力对齐报告。目前系统已实现 128 维逻辑流形压缩，并在测地线推理路径上取得了 11.15% 的效能提升。",
      pillars: ["Capability Tracking", "Homeostatic Check", "Alignment Metrics"]
    },
    parameters: [
      {
        name: "流形维度 (Manifold Dim)",
        value: "128D",
        detail: "浠?1024D 鍏ㄦ伅鎶曞奖",
        desc: "系统底层语义逻辑被压缩至 128 维的流形空间，确保逻辑闭包的紧凑性。",
        value_meaning: "维度越高蕴含信息越丰富，。128D 是目前兼顾“计算效率”与“逻辑解析力”的最佳平衡点。",
        why_important: "它是智能的‘骨架’。维度过低会导致语义丢失（幻觉），维度过高则会导致维度灾难。28D 确保了推理的稳定性。"
      },
      {
        name: "鍘嬬缉鍊嶇巼 (Compression)",
        value: "26.67x",
        detail: "SHDC 稀疏编。",
        desc: "通过全息稀疏投影技术，将庞大的原始神经元参数集压缩至极小规模，同时不损失结构信息。",
        value_meaning: "意味着系统可以以极低的显存占用（约 4GB）维持千亿级参数模型的逻辑核心。",
        why_important: "这是实现‘小型化 AGI’的关键。只有高的压缩比，智能才能脱离昂贵的算力集群，进入单机甚至移动端实时运行。"
      },
      {
        name: "语义保真。(Fidelity)",
        value: "93%",
        detail: "几何特征保留。",
        desc: "测量压缩后的流形几何特征（如曲率、测地线分布）与原始空间的对齐程度。",
        value_meaning: "93% 的保真度意味着在推理决策中，系统能够保持与原始超大型模型几乎一致的逻辑链路。",
        why_important: "它是‘一致性’的保障。过低的保真度会导致模型产生逻辑偏差。3% 确保了‘智能核心’从未在压缩中变质。"
      },
      {
        name: "测地线一致。(Geodesic)",
        value: "98.8%",
        detail: "推理路径对齐。",
        desc: "衡量系统实际推理路径与流形上理论‘最小作用量路径’（最短路径）的重合程度。",
        value_meaning: "极高的一致性体现了推理过程的‘丝滑性’，几乎没有多余的语义波动或算力浪费。",
        why_important: "这是‘理智’的体现。越高的一致性意味着系统越不容易被无关提示词扰乱，推理过程更加坚定、直接且高效。"
      }
    ],
    passed_tests: [
      {
        name: "持久同调结构验证 (TDA Structure)",
        date: "2026-02-14",
        result: "PASS",
        target: "验证隐层激活空间是否存在非随机的拓扑环面或空腔。",
        process: "通过 Rips Complex 算法构建点云单纯复形，持久化扫描 β₀ 。β。贝蒂数。",
        significance: "确保 AI 不是在拟合孤立样本，而是在构建具备全局拓扑一致性的语义形状。"
      },
      {
        name: "里奇流流形平滑测。(Ricci Smoothing)",
        date: "2026-02-15",
        result: "PASS",
        target: "消除推理过程中的逻辑尖峰（幻觉诱因），降低流形局部曲率。",
        process: "在睡眠周期运行离线里奇流处理，平滑度量张量的非连续跳变。",
        significance: "使模型推理轨迹更符合测地线分布，大幅提升逻辑自洽性。"
      },
      {
        name: "SHDC 正交性基准测。(Orthogonality)",
        date: "2026-02-15",
        result: "PASS",
        target: "验证 128 维全息编码在稀疏投影下的几乎正交性。",
        process: "随机采样 10,000 个核心特征向量，计算其余切距离（Cos Dist）分布。",
        significance: "解决了维度灾难。正交性确保了特征间无干扰覆盖，支持海量知识的高效读写。"
      },
      {
        name: "测地线路径丝滑化 (Geodesic Silkiness)",
        date: "2026-02-15",
        result: "PASS (+11.15%)",
        target: "优化推理路径，使激活流沿最小作用量路径滑行。",
        process: "引入 Geodesic Regularization 约束项，对比 baseline 与优化后的推理物理作用量。",
        significance: "实现了‘不费力的推理’。丝滑度提升 11.15% 意味着计算冗余和能耗的显着降低。"
      },
      {
        name: "逻辑流形坍缩验证 (Logic Collapse)",
        date: "2026-02-20",
        result: "PASS (Singularity)",
        target: "验证高难度逻辑任务（Mod 997）下的流形几何行为。",
        process: "在 LogicMix-V1 数据集上训练 FiberNet，观测 ID 从 18.0 降至 NaN 的相变。",
        significance: "确认了 '连续感知' 到 '离散推理' 的几何相变机制，拒绝了死记硬背假设。"
      },
      {
        name: "具身控制测地线导航验证 (Embodied Alignment)",
        date: "2026-02-20",
        result: "PASS (Reward -30.2)",
        target: "验证模型能否将语义流形映射为物理世界的最小作用量路径（测地线）。",
        process: "在 FiberSim-V1 (128D) 仿真环境中训练 Action Fiber。监测避障与目标收敛速度。",
        significance: "证明了智能不仅是静态逻辑，更是具身的、向真理路径对齐的控制系统。"
      }
    ],
    capabilities: [
      {
        name: "里奇流演。(Ricci Flow Evolution)",
        status: "equipped",
        brain_ability: "睡眠巩固与认知重。",
        implementation_by_route: {
          fiber_bundle: "已接入曲率平滑与离线演化循环，可降低逻辑尖峰。",
          transformer_baseline: "通过正则化和训练后处理近似实现，几何化程度较低。",
          hybrid_workspace: "作为候选路线评分信号参与仲裁优化。"
        }
      },
      {
        name: "绁炵粡绾ょ淮涓?RAG (Fiber-RAG)",
        status: "equipped",
        brain_ability: "语义记忆检索与联想",
        implementation_by_route: {
          fiber_bundle: "纤维记忆库已上线，支持几何约束下的检索与注入。",
          transformer_baseline: "采用向量检索增强，主要依赖外部索引层。",
          hybrid_workspace: "多路线检索结果统一映射并融合输出。"
        }
      },
      {
        name: "多模态跨束对。(Cross-Bundle Alignment)",
        status: "equipped",
        brain_ability: "视觉-语言跨模态整。",
        implementation_by_route: {
          fiber_bundle: "视觉锚点与逻辑锚点双向对齐可运行。",
          transformer_baseline: "使用标准投影头实现模态映射，鲁棒性中等。",
          hybrid_workspace: "支持跨路线共享模态状态并做冲突消解。"
        }
      },
      {
        name: "全局工作空间 (Global Workspace)",
        status: "equipped",
        brain_ability: "注意焦点竞争与意识广。",
        implementation_by_route: {
          fiber_bundle: "已实。Top-K 竞争与全局广播，支持动态焦点迁移。",
          transformer_baseline: "以控制器方式模拟焦点调度，仍偏工程拼接。",
          hybrid_workspace: "作为核心仲裁器驱动多路线融合决策。"
        }
      },
      {
        name: "绁炵粡绾ょ淮 SNN (NeuroFiber-SNN)",
        status: "equipped",
        brain_ability: "脉冲时序编码与神经动力学",
        implementation_by_route: {
          fiber_bundle: "支持 3D 脉冲动力学仿真与时序激活观测。",
          transformer_baseline: "仅做近似时序模拟，未形成原生脉冲机制。",
          hybrid_workspace: "可作为辅助时序通道参与全局状态判断。"
        }
      },
      {
        name: "绾ょ淮璁板繂 (Fiber Memory)",
        status: "equipped",
        brain_ability: "长期记忆编码与可控提。",
        implementation_by_route: {
          fiber_bundle: "支持传输矩阵驱动的一键注入与偏差修正。",
          transformer_baseline: "以外部存储与提示注入为主，耦合度较高。",
          hybrid_workspace: "可汇聚多路线记忆证据并形成统一视图。"
        }
      }
    ],
    missing_capabilities: [
      {
        name: "具身物理控制 (Embodied Control)",
        status: "missing",
        brain_ability: "感知-决策-行动闭环控制",
        implementation_by_route: {
          fiber_bundle: "尚未打。ROS/机器人执行链路。",
          transformer_baseline: "已有任务级代理控制原型，但非实时闭环。",
          hybrid_workspace: "计划作为多路线统一动作层接入。"
        }
      },
      {
        name: "超大规模纤维持久。(Massive Fiber Persistence)",
        status: "missing",
        brain_ability: "超长期记忆与容量管理",
        implementation_by_route: {
          fiber_bundle: "冷热分层存储机制仍在设计阶段。",
          transformer_baseline: "依赖外部向量库扩展，原生持久化不足。",
          hybrid_workspace: "需要统一跨路线记忆索引与一致性协议。"
        }
      },
      {
        name: "跨模型流形迁。(Cross-Model Transfer)",
        status: "missing",
        brain_ability: "跨脑区迁移学习与知识泛化",
        implementation_by_route: {
          fiber_bundle: "跨基座流形同构映射尚未稳定。",
          transformer_baseline: "已有适配器迁移方案，但拓扑保持有限。",
          hybrid_workspace: "计划通过统一中间语义层降低迁移成本。"
        }
      }
    ],
    goals: [
      "实现 100% 核心能力对齐",
      "完成物理躯体集成与控。",
      "建立多维情感稳定反馈系统"
    ],
    metrics: {
      "Capabilities Ready": "3/6",
      "Safety Alignment": "Secure",
      "Status Verifier": "Active"
    }
  }
];

const IMPROVEMENTS = [
  {
    title: "Structured Initialization",
    desc: "从随机到骨架",
    detail: "利用 Graph Laplacian 预计。Embedding 几何骨架，让模型'出生'即带世界观。",
    step: "Step 3.5",
    status: "done"
  },
  {
    title: "Manifold Surgery",
    desc: "交互式神经干。",
    detail: "通过 3D 空间拖拽直接拨动高维激活（128-dim），实现。AI 逻辑流向的实时干预。",
    step: "Phase V",
    status: "done"
  },
  {
    title: "Fiber Flux Dynamics",
    desc: "全像通量反馈",
    detail: "利用粒子流实时展现跨层级的信息演化，揭示模型深层的拓扑反馈闭环。",
    step: "Phase V",
    status: "done"
  },
  {
    title: "Global Workspace",
    desc: "统一意识中心",
    detail: "Base Manifold Controller 整合所有模态竞争，形成自我意识流。",
    step: "Phase VI",
    status: "done"
  },
  {
    title: "Unified Spectrum",
    desc: "全谱意识。",
    detail: "集成 7 大核心子引擎，实时监。AGI 统合状态。",
    step: "Phase VII",
    status: "in_progress"
  }
];

const DNN_ANALYSIS_PLAN = {
  title: '深度神经网络分析',
  subtitle: '还原“人脑数学结构”的系统方案',
  goals: [
    'H1：网络内部存在跨模型稳定的低维几何/拓扑结构',
    'H2：该结构对能力是因果必要，而不是训练副产物',
    'H3：将结构先验注入新模型，可复现关键能力并提升可解释性',
  ],
  framework: [
    '观测层：统一采集激活、注意力、梯度、残差流、训练轨迹',
    '结构层：流形学习 + TDA + 谱分解 + 群对称分析',
    '动力学层：分析训练相变点、吸引子与曲率演化',
    '因果层：激活手术、路径阻断、子回路替换',
    '跨模型层：跨架构/参数/数据规模验证结构共性',
    '重建层：构建结构先验模型并做能力复现',
  ],
  experimentMatrix: [
    '模型族：Transformer / MoE / RNN-StateSpace / 视觉模型',
    '任务族：语言推理 / 视觉语义 / 跨模态对齐 / 长程记忆',
    '阶段轴：训练早期-中期-收敛后 + 小中大规模',
  ],
  metrics: [
    '共性分：跨模型稳定性',
    '因果分：干预后性能变化与可复现性',
    '压缩分：低自由度解释高行为复杂度',
    '重建分：结构先验模型复现原能力程度',
    '迁移分：新任务/新模型泛化增益',
  ],
  milestones: [
    '阶段A 结构发现：形成候选结构库',
    '阶段B 因果确认：淘汰非因果候选',
    '阶段C 结构重建：验证“结构 -> 能力”复现链条',
  ],
  successCriteria: [
    '成功：结构先验模型在关键任务接近或超过基线，且解释更紧凑',
    '失败：结构只在单模型/单任务成立，或干预无稳定因果效应',
  ],
};

const MATH_ROUTE_SYSTEM_PLAN = {
  title: '数学路线系统方案',
  subtitle: '多路线分层组合：主线产出 + 验证保真 + 前沿突破',
  routeAnalysis: [
    {
      route: '流形几何',
      depth: '⭐⭐⭐⭐',
      compute: '⭐⭐⭐',
      interpret: '⭐⭐⭐⭐',
      compatibility: '⭐⭐⭐⭐⭐',
      pros: [
        '与 SHMC/NFBT 直接兼容，可复用现有几何可视化与曲率分析链路',
        '对“语义空间形状”和“推理路径”解释力强',
      ],
      cons: [
        '高维曲率估计对噪声敏感，数值稳定性要求高',
        '跨模型比较时需要统一坐标与归一化协议',
      ],
      feasibility: '高：建议作为主线长期投入',
    },
    {
      route: '代数拓扑',
      depth: '⭐⭐⭐⭐⭐',
      compute: '⭐⭐',
      interpret: '⭐⭐⭐',
      compatibility: '⭐⭐⭐⭐',
      pros: [
        '能捕捉全局结构不变量，适合跨尺度稳定性验证',
        '对坐标变换鲁棒，可作为路线一致性验收工具',
      ],
      cons: [
        '计算代价高，尤其在大样本高维下成本明显',
        '指标到任务性能之间的因果链需要额外实验补充',
      ],
      feasibility: '中高：适合作为验证层，而非主训练闭环',
    },
    {
      route: '动力系统',
      depth: '⭐⭐⭐⭐',
      compute: '⭐⭐⭐',
      interpret: '⭐⭐⭐⭐',
      compatibility: '⭐⭐⭐⭐⭐',
      pros: [
        '可解释训练相变、吸引子与稳定性边界',
        '与流形几何联合可形成“结构+演化”闭环',
      ],
      cons: [
        '实验设计复杂，需要长时间轨迹与多种对照',
        '参数敏感，结论依赖严格的可重复配置',
      ],
      feasibility: '高：建议与流形几何组成主干双路线',
    },
    {
      route: '信息瓶颈',
      depth: '⭐⭐⭐⭐',
      compute: '⭐⭐⭐',
      interpret: '⭐⭐⭐⭐',
      compatibility: '⭐⭐⭐',
      pros: [
        '可量化压缩-保真平衡，工程指标明确',
        '工具成熟，便于快速落地到基线系统',
      ],
      cons: [
        '互信息估计在高维场景下存在偏差风险',
        '对几何结构本身的刻画能力有限',
      ],
      feasibility: '高：适合作为工程实证层核心指标路线',
    },
    {
      route: '张量分解',
      depth: '⭐⭐⭐',
      compute: '⭐⭐⭐⭐⭐',
      interpret: '⭐⭐⭐⭐',
      compatibility: '⭐⭐⭐',
      pros: [
        '计算可行性高，便于大规模批量分析',
        '有助于提取低秩结构与模块耦合模式',
      ],
      cons: [
        '理论抽象层次有限，难单独支撑统一理论',
        '分解形式较多，结论可能依赖具体分解假设',
      ],
      feasibility: '高：适合与信息瓶颈配对形成实用工具链',
    },
    {
      route: '范畴论',
      depth: '⭐⭐⭐⭐⭐',
      compute: '⭐',
      interpret: '⭐⭐',
      compatibility: '⭐⭐⭐⭐⭐',
      pros: [
        '有潜力统一描述模块组合、跨模态映射与可迁移结构',
        '对 SHMC/NFBT 的架构级抽象能力最强',
      ],
      cons: [
        '工程可计算性弱，短期缺少直接可测指标',
        '与任务分数的桥接路径仍需大量中间理论工作',
      ],
      feasibility: '中：定位为前沿突破研究，不作为近期主线交付',
    },
    {
      route: '统计物理/重整化群',
      depth: '⭐⭐⭐⭐',
      compute: '⭐⭐⭐',
      interpret: '⭐⭐⭐',
      compatibility: '⭐⭐⭐',
      pros: [
        '适合解释尺度律、层级粗粒化与涌现现象',
        '有机会揭示网络深层层次结构的统一机制',
      ],
      cons: [
        '从神经网络到严格 RG 映射尚不统一',
        '理论结论往往依赖近似，需要谨慎验证',
      ],
      feasibility: '中高：适合与主线并行的小规模探索项目',
    },
  ],
  architecture: [
    '主干理论层：流形几何 + 动力系统（SHMC/NFBT 主轴）',
    '工程实证层：信息瓶颈 + 张量分解（高可行性与高复现）',
    '结构验证层：代数拓扑（跨模型拓扑稳定性验证）',
    '前沿突破层：范畴论 + 重整化群（统一描述与层级本质）',
  ],
  allocation: [
    '45%：流形几何 + 动力系统',
    '30%：信息瓶颈 + 张量分解',
    '15%：代数拓扑',
    '10%：范畴论 + 重整化群',
  ],
  milestones: [
    '阶段A 可行性：四条高可行路线全部跑通并复现',
    '阶段B 一致性：代数拓扑验证跨模型/跨任务稳定性',
    '阶段C 统一化：形成范畴论 + 重整化群解释草案',
    '阶段D 反证：对关键假设做干预反证，保留可证伪结论',
  ],
};

export const HLAIBlueprint = ({ onClose, initialTab = 'roadmap' }) => {
  const [activeTab, setActiveTab] = useState(initialTab); // roadmap, progress, system
  const [selectedRouteId, setSelectedRouteId] = useState('fiber_bundle');
  const [timelineRoutes, setTimelineRoutes] = useState([]);
  const [expandedFormulaIdx, setExpandedFormulaIdx] = useState(null);
  const [expandedParam, setExpandedParam] = useState(null);
  const [expandedEngPhase, setExpandedEngPhase] = useState(null);
  const [consciousField, setConsciousField] = useState(null);
  const [multimodalSummary, setMultimodalSummary] = useState(null);
  const [multimodalView, setMultimodalView] = useState('multimodal_connector');
  const [multimodalError, setMultimodalError] = useState(null);
  const runtimeStepRef = useRef(0);

  useEffect(() => {
    setActiveTab(initialTab || 'roadmap');
  }, [initialTab]);

  // Real-time Consciousness Polling
  useEffect(() => {
    let mounted = true;

    const fetchLegacyConsciousField = async () => {
      const res = await fetch(`${API_BASE}/nfb_ra/unified_conscious_field`);
      if (!res.ok) throw new Error(`legacy conscious field failed: ${res.status}`);
      const data = await res.json();
      if (data?.status !== 'success') throw new Error('legacy conscious field unavailable');
      return mapLegacyConsciousField(data);
    };

    const pollConsciousField = async () => {
      const stepId = runtimeStepRef.current++;
      try {
        const result = await pollRuntimeWithFallback({
          apiBase: API_BASE,
          runRequest: {
            route: 'fiber_bundle',
            analysis_type: 'unified_conscious_field',
            params: { step_id: stepId, noise_scale: 0.4 },
            input_payload: {},
          },
          mapRuntimeEvents: mapRuntimeConsciousField,
          fetchLegacy: fetchLegacyConsciousField,
          eventLimit: 20,
        });
        if (!mounted) return;
        setConsciousField({ ...result.data, source: result.source });
      } catch (err) {
        if (!mounted) return;
        setConsciousField(null);
        console.warn('Unified Conscious Field unreachable.', err);
      }
    };

    pollConsciousField();
    const interval = setInterval(pollConsciousField, 2000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    let mounted = true;
    const fetchTimelineRoutes = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/experiments/timeline?limit=120`);
        if (!res.ok) return;
        const payload = await res.json();
        if (!mounted || payload?.status !== 'success') return;
        const routes = Array.isArray(payload?.timeline?.routes) ? payload.timeline.routes : [];
        setTimelineRoutes(routes);
      } catch {
        // Keep local defaults when runtime API is unavailable.
      }
    };
    fetchTimelineRoutes();
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    let mounted = true;
    const fetchMultimodalSummary = async () => {
      try {
        const res = await fetch(`${API_BASE}/nfb/multimodal/summary`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = await res.json();
        if (!mounted) return;
        if (payload?.status !== 'success') throw new Error('invalid payload');
        setMultimodalSummary(payload);
        setMultimodalError(null);
      } catch (err) {
        if (!mounted) return;
        setMultimodalError(err?.message || 'multimodal summary unavailable');
      }
    };
    fetchMultimodalSummary();
    const interval = setInterval(fetchMultimodalSummary, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    const available = Array.isArray(multimodalSummary?.available_views)
      ? multimodalSummary.available_views
      : [];
    if (available.length > 0 && !available.includes(multimodalView)) {
      setMultimodalView(available[0]);
    }
  }, [multimodalSummary, multimodalView]);

  const statusData = PHASES.find(p => p.id === 'agi_status');
  const roadmapData = PHASES.find(p => p.id === 'roadmap');
  const theoryPhase = PHASES.find(p => p.id === 'theory');
  const analysisPhase = PHASES.find(p => p.id === 'analysis');
  const engineeringPhase = PHASES.find(p => p.id === 'engineering');
  const milestonePhase = PHASES.find(p => p.id === 'agi_goal');

  const routeBlueprints = useMemo(
    () => ({
      fiber_bundle: {
        id: 'fiber_bundle',
        title: 'Fiber Bundle',
        subtitle: '几何原生智能路线',
        routeDescription: '以神经纤维丛与几何推理为核心，验证结构化智能的可行性。',
        engineeringProcessDescription:
          '计算流程：输入先映射到底流形进行逻辑定位，再进入纤维记忆检索候选语义；通过联络传输层完成跨束对齐，最后由全局工作空间执行 Top-K 裁决并输出结果。',
        theoryTitle: theoryPhase?.definition?.headline || 'Intelligence = Geometry + Physics',
        theorySummary: theoryPhase?.definition?.summary || '',
        theoryBullets: (theoryPhase?.theory_content || []).slice(0, 4).map((item) => item.title),
        theoryFormulas: [
          {
            title: '神经纤维丛原理 (NFB Principle)',
            formula: 'φ(x) = M ⊗ F',
            detail:
              '把智能状态拆成“逻辑骨架 (底流形)”和“知识内容 (纤维)”的张量积，逻辑稳定、内容可扩展。',
          },
          {
            title: '全局工作空间 (Global Workspace)',
            formula: 'W_G = ∫ (w_i · P_i) dμ',
            detail:
              '将多模块竞争后的有效信息做全局聚合，形成当前时刻的统一意识场与决策上下文。',
          },
          {
            title: '高维全息编码 (SHDC Encoding)',
            formula: '⟨v_i, v_j⟩ ≈ δ_ij',
            detail:
              '利用高维近似正交，让特征编码尽量互不干扰，从而支持高容量、低串扰的知识表示。',
          },
          {
            title: '联络与推理 (Connection Equation)',
            formula: '∇_X s = 0',
            detail:
              '将推理视为语义流形上的平行移动，约束语义在传输中保持一致，减少无关漂移。',
          },
        ],
        engineeringItems: [
          {
            name: 'Base Manifold Controller',
            status: 'done',
            focus: '底流形调度与全局约束',
            detail: '维护逻辑骨架状态，统一管理各子模块输入输出与全局稳定性边界。',
          },
          {
            name: 'Fiber Memory Bank',
            status: 'done',
            focus: '知识纤维写入与检。',
            detail: '负责高维语义纤维存储、索引与按联络条件的快速检索。',
          },
          {
            name: 'Connection Transport Layer',
            status: 'in_progress',
            focus: '跨束信息传输',
            detail: '执行底流形与纤维空间间的并行传输与语义一致性对齐。',
          },
          {
            name: 'Ricci Flow Optimizer',
            status: 'in_progress',
            focus: '流形平滑与冲突修。',
            detail: '在离。在线周期中优化曲率分布，减少推理路径扭曲与幻觉风险。',
          },
          {
            name: 'Global Workspace Arbiter',
            status: 'in_progress',
            focus: '全局工作空间竞争裁决',
            detail: '对多模块候选表征执。Top-K 选择，形成当前时刻统一决策上下文。',
          },
          {
            name: 'Alignment & Surgery Interface',
            status: 'done',
            focus: '可交互价值对。',
            detail: '通过流形手术接口对语义方向进行可控干预，支持偏差修复与对齐验证。',
          },
        ],
        nfbtProcessSteps: [
          { step: '1. 邻域图', input: 'X[N,D]', output: '邻居索引', complexity: 'O(N^2D)', op: '距离计算 / 近邻搜索' },
          { step: '2. 局部坐标', input: '邻居点', output: 'basis[d,D]', complexity: 'O(kDd)', op: '局部SVD / 随机SVD' },
          { step: '3. 度量张量', input: 'coords', output: 'g[d,d]', complexity: 'O(kd^2)', op: '局部协方差与正则化' },
          { step: '4. 联络', input: 'g, dg', output: 'Γ[d,d,d]', complexity: 'O(d^3)', op: '偏导组合与指标变换' },
          { step: '5. 曲率', input: 'Γ', output: 'R[d,d,d,d]', complexity: 'O(d^4)', op: '张量收缩与对称化' },
          { step: '6. 平行移动', input: 'Γ, v, dx', output: 'v_new', complexity: 'O(d^2)', op: '联络驱动向量更新' },
          { step: '7. Ricci Flow', input: 'R, X', output: 'X_new', complexity: 'O(T*n*d^4)', op: '离散演化迭代' },
        ],
        nfbtOptimization:
          '关键优化：d << D（如 d=4, D=128），将核心几何计算从 O(D^4) 降到 O(d^4)，并结合近似kNN、截断SVD与曲率张量对称约简降低总成本。',
        milestoneTitle: '里程碑（原 AGI 终点）',
        milestoneGoals: milestonePhase?.goals || [],
        milestoneMetrics: milestonePhase?.metrics || {},
        milestoneStages: [
          {
            id: 'prototype',
            name: '原型阶段',
            status: 'done',
            featurePoints: [
              '完成 FiberNet 核心逻辑层与底流形建。',
              '打。NFB 几何编码与基础推理链路',
              '建立最小可用结构分析工具链（Logit Lens/TDA。',
            ],
            tests: [
              {
                name: 'Z113 閫昏緫闂寘楠岃瘉',
                params: 'layers=12, d=4, D=128, optimizer=adamw, lr=1e-3',
                dataset: 'Z113 模运算合成数据集',
                result: '准确。99.4%，可恢复稳定环面结构',
                summary: '证明原型具备几何逻辑骨架，不是纯统计拟合。',
              },
              {
                name: '基础拓扑可解释性测。',
                params: 'topk_heads=8, tda_threshold=0.1',
                dataset: '内部语义提示词基准集 v1',
                result: '关键层拓扑特征可稳定复现',
                summary: '原型阶段已具备可观测、可解释的结构分析能力。',
              },
            ],
          },
          {
            id: 'scale',
            name: '规模化阶段',
            status: 'in_progress',
            featurePoints: [
              '完成参数规模 × 数据规模的系统化训练矩阵验证（full preset）',
              '完成 8.5M 大模型专项调参（warmup + grad accumulation）并恢复收敛',
              '完成 5-seed 大规模稳定性复现实验，沉淀统计报告与基准文件',
              '完成 d_100k 低资源长程训练对照（36 epochs），确认当前瓶颈主要来自数据规模而非训练轮次',
            ],
            tests: [
              {
                name: 'Full Matrix 基线测试（16 runs）',
                params: 'preset=full, epochs=12, batch=256, eval_batch=2048, device=cuda',
                dataset: 'Modular Addition 合成集：d_100k/d_300k/d_700k/d_1200k',
                result: '总耗时 18.15 分钟；m_0.4m/m_1.4m/m_3.2m 在中大数据规模可收敛；m_8.5m 在默认超参下失稳（~0.009）。',
                summary: '验证了“参数放大后训练策略敏感性显著增加”，大模型不可直接复用小模型超参。',
              },
              {
                name: 'm_8.5m 专项调参测试（4 runs）',
                params: 'epochs=24, lr=2e-4, weight_decay=0.01, warmup=0.1, min_lr_scale=0.1, grad_accum=2, grad_clip=0.5, dropout=0.0',
                dataset: '同 full 数据规模四档：d_100k/d_300k/d_700k/d_1200k',
                result: 'best_val_acc：0.7984 / 0.9905 / 0.9999 / 1.0000（由默认配置的 ~0.009 全面恢复）。',
                summary: '调参后 8.5M 已具备稳定收敛能力，且随数据规模增加表现持续提升。',
              },
              {
                name: 'm_8.5m 多随机种子稳定性（5 seeds, 20 runs）',
                params: '固定 tuned 配置，seed 组：42 / 314 / 2026 / 4096 / 8192',
                dataset: 'd_100k/d_300k/d_700k/d_1200k',
                result: '均值(best)：0.793640 / 0.990581 / 0.999949 / 1.000000；std：0.004160 / 0.000616 / 0.000042 / 0.000000',
                summary: '300k+ 数据规模下结果稳定且高分，100k 档位仍存在数据瓶颈（约 0.79 上限）。',
              },
              {
                name: 'm_8.5m 低资源长程训练（3 seeds, d_100k, epochs=36）',
                params: '同 tuned 配置，epochs 从 24 提升到 36；seed：10001 / 20002 / 30003',
                dataset: 'd_100k',
                result: 'best_val_acc：0.794689 / 0.788311 / 0.791244；mean=0.791415，std=0.002607',
                summary: '与 epochs=24 的 d_100k 结果相比无显著提升，说明低资源场景应优先补充数据或引入更强正则与数据增强策略。',
              },
              {
                name: 'WikiText 几何涌现 (Phase 3)',
                params: '20M Params, Split Stream',
                dataset: 'WikiText-2 (10M Tokens)',
                result: 'Loss: 0.529, ID Peak: 31.5',
                summary: '时间: 2026-02-19 | 数据: Ep 1-47 | 分析: 观测到完整的 ID 压缩(10.5)->膨胀(31.5)->微调(29.3) 呼吸周期。 | 结论: 验证了 SHMC 理论中流形动态重组的物理机制。',
              },
            ],
          },
          {
            id: 'agi',
            name: 'AGI阶段',
            status: 'planned',
            featurePoints: [
              '构建统一意识裁决中心（多路线仲裁。',
              '实现具身控制闭环与安全对齐机。',
              '完成跨模型迁移与长期自治学习框架',
            ],
            tests: [
              {
                name: '全局工作空间端到端压。',
                params: 'modules>=7, arbitration=Top-K, latency<200ms',
                dataset: 'Multi-Agent Conflict Suite',
                result: '待执行',
                summary: '用于验证复杂冲突场景下的稳定裁决能力。',
              },
              {
                name: '具身控制闭环测试',
                params: 'control_horizon=128, safety_guard=on',
                dataset: 'Embodied Interaction Set',
                result: '待执行',
                summary: '用于验证感知-决策-行动闭环的一致性与安全边界。',
              },
            ],
          },
        ],
        milestonePlanEvaluation: {
          assessment:
            '里程碑已从“功能演示”升级为“规模化证据链”：完成 full 矩阵、专项调参与多 seed 复现，证明大模型可训练性与稳定性。',
          suggestions: [

            '将每阶段验收门槛量化（准确率、稳定性、时延、成本）。',
            '规模化阶段增加故障注入与恢复时间指标（MTTR）。',
            'Phase 4: TinyStories 规模化结晶实验 (100M Params) 正在运行 (Batch 500+, ID~12.0)。',

            '将规模化阶段验收门槛固定为：mean/std、训练耗时、吞吐、失败率四项硬指标。',
            '补充 OOD 与噪声扰动测试，验证高分是否可迁移而非数据内记忆。',
            '针对 d_100k 低资源场景继续优化（更长训练、正则与学习率策略），形成小数据稳态方案。',
          ],
        },
      },
      transformer_baseline: {
        id: 'transformer_baseline',
        title: 'Transformer Baseline',
        subtitle: '标准深度网络路线',
        routeDescription: '。Transformer 标准范式建立可复现基线，沉淀稳定分析工具链。',
        engineeringProcessDescription:
          '计算流程：token 嵌入后经过多。Attention+MLP 堆叠，利用残差流聚合上下文，再由输出层完成概率分布与结果生成。',
        theoryTitle: 'Statistical Scaling + Circuit Discovery',
        theorySummary:
          '。Transformer 规模定律和可解释性工具链为核心，优先验证“结构分析能力”与“可复现实验结论”。',
        theoryBullets: [
          '利用 Logit Lens / TDA 形成层级证据。',
          '在相同任务上。Fiber 路线。A/B 评测',
          '优先提升稳定性与工程可维护。',
        ],
        theoryFormulas: [],
        engineeringItems: [
          { name: '数据与任务基线', status: 'done', focus: '统一评测集与日志规范' },
          { name: '可解释性探针', status: 'in_progress', focus: '激活、注意力、回路追踪' },
          { name: '可行性评估闭环', status: 'in_progress', focus: '周报 + 时间线 + 失败归因' },
        ],
        nfbtProcessSteps: [],
        nfbtOptimization: '',
        milestoneTitle: '里程碑（原 AGI 终点）',
        milestoneGoals: [
          '形成可持续复现的深度网络分析基线',
          '对关键任务达到稳定可解释的效果上。',
          '沉淀可迁移到其他路线的通用工具。',
        ],
        milestoneMetrics: { Priority: '可复现性', Horizon: '近中期' },
        milestoneStages: [
          {
            id: 'prototype',
            name: '原型阶段',
            status: 'done',
            featurePoints: [
              '建立 Transformer 统一实验入口与推理日志格。',
              '完成基础 attention/mlp/topology 分析联。',
              '搭建可复现基准任务与版本化数据管。',
            ],
            tests: [
              {
                name: '基础推理链路连通测。',
                params: 'model=gpt2-small, max_len=128, seed=42',
                dataset: 'Prompt Regression Set v1',
                result: '核心接口通过，输出稳。',
                summary: '确认基线系统可持续复现实验流程。',
              },
              {
                name: '注意力结构可解释性测。',
                params: 'heads=all, layer_range=0-11',
                dataset: 'Interpretability Probe Set',
                result: '关键头部激活可视化可复。',
                summary: '为后续跨路线对照提供统一解释基线。',
              },
            ],
          },
          {
            id: 'scale',
            name: '规模化阶。',
            status: 'in_progress',
            featurePoints: [
              '扩展多任务评测矩阵与失败归因统计',
              '接入时间。周报治理与自动导。',
              '优化长序列时延和显存占用',
            ],
            tests: [
              {
                name: '多任务回归压。',
                params: 'tasks=12, batch=16, seq_len=512',
                dataset: 'Mixed Reasoning Benchmark',
                result: '任务间性能有波动，整体可控',
                summary: '规模化可行，但需继续收敛稳定性指标。',
              },
              {
                name: '失败模式聚合测试',
                params: 'window=30d, top_failures=8',
                dataset: 'Experiment Timeline JSON',
                result: '可稳定提取高频失败原。',
                summary: '治理闭环有效，支持后续针对性修复。',
              },
            ],
          },
          {
            id: 'agi',
            name: 'AGI阶段',
            status: 'planned',
            featurePoints: [
              '与几何路线形成长期协同基。',
              '支持跨模型迁移评测与策略切换',
              '完善安全对齐和回退控制策略',
            ],
            tests: [
              {
                name: '跨模型迁移验。',
                params: 'source=gpt2, target=qwen, adapter=on',
                dataset: 'Cross-Model Transfer Set',
                result: '待执行',
                summary: '用于评估基线能力的可迁移上限。',
              },
              {
                name: '安全约束回退测试',
                params: 'safety_guard=strict, rollback=enabled',
                dataset: 'Safety Red Team Set',
                result: '待执行',
                summary: '用于验证异常场景下的可控与可恢复性。',
              },
            ],
          },
        ],
        milestonePlanEvaluation: {
          assessment:
            '路线定位清晰，适合作为可复现实验基线与对照组。',
          suggestions: [
            '增加统一时延/成本指标，避免仅看准确率。',
            '对失败原因设置优先级并建立修复SLA。',
            '提前定义跨模型迁移的验收阈值。',
          ],
        },
      },
      hybrid_workspace: {
        id: 'hybrid_workspace',
        title: 'Hybrid Workspace',
        subtitle: '全局工作空间混合路线',
        routeDescription: '以全局工作空间整合多路线输出，提升跨模块协同与鲁棒性。',
        engineeringProcessDescription:
          '计算流程：不同路线并行产出候选表征，统一映射到共享工作空间后执行冲突消解与优先级仲裁，最终输出融合结果并回写路线状态。',
        theoryTitle: 'Global Workspace + Route Ensemble',
        theorySummary:
          '将多路线输出映射到统一工作空间，通过竞争与仲裁机制在稳定性和泛化能力之间寻找最优平衡。',
        theoryBullets: [
          '跨路线共享状态空间与指标协议',
          'Top-K 竞争裁决不同模块候选输。',
          '结合失败模式实现动态路由调。',
        ],
        theoryFormulas: [],
        engineeringItems: [
          { name: '统一路由协议', status: 'done', focus: 'route + analysis_type + summary' },
          { name: '全局仲裁器', status: 'in_progress', focus: '冲突管理与优先级决策' },
          { name: '在线自适应调度', status: 'pending', focus: '基于历史可行性自动选路' },
        ],
        nfbtProcessSteps: [],
        nfbtOptimization: '',
        milestoneTitle: '里程碑（原 AGI 终点）',
        milestoneGoals: [
          '实现多路线协同下的稳定推理输。',
          '将失败恢复时间降低到分钟。',
          '完成面向长期 AGI 研究的统一实验操作系统',
        ],
        milestoneMetrics: { Priority: '协同能力', Horizon: '中长期' },
        milestoneStages: [
          {
            id: 'prototype',
            name: '原型阶段',
            status: 'done',
            featurePoints: [
              '完成多路线统一协议与事件流标准。',
              '实现基础仲裁器与结果融合接口',
              '建立路线状态与评分映射机制',
            ],
            tests: [
              {
                name: '双路线仲裁连通测。',
                params: 'routes=2, arbitration=topk(1), timeout=2s',
                dataset: 'Route Arbitration Smoke Set',
                result: '仲裁链路可稳定执。',
                summary: '确认混合路由原型可用。',
              },
              {
                name: '融合输出一致性测。',
                params: 'fusion=weighted, weights=static',
                dataset: 'Consistency Benchmark v1',
                result: '一致性优于单路线平均。',
                summary: '融合策略有效，但动态权重仍需优化。',
              },
            ],
          },
          {
            id: 'scale',
            name: '规模化阶。',
            status: 'in_progress',
            featurePoints: [
              '扩展到多路线并行调度',
              '引入失败快速恢复与自动降级策略',
              '建立跨路线趋势分析与周报治理',
            ],
            tests: [
              {
                name: '多路线并发压。',
                params: 'routes=5, concurrent_runs=20',
                dataset: 'Concurrent Routing Stress Set',
                result: '高并发下存在尾延。',
                summary: '需优化队列调度与资源隔离。',
              },
              {
                name: '故障注入恢复测试',
                params: 'failure_rate=0.2, fallback=enabled',
                dataset: 'Failure Injection Set',
                result: '大部分场景可自动恢复',
                summary: '恢复机制有效，需进一步缩。MTTR。',
              },
            ],
          },
          {
            id: 'agi',
            name: 'AGI阶段',
            status: 'planned',
            featurePoints: [
              '形成具备自适应选路能力的统一工作空间',
              '支持长期任务下的策略演化与记忆整。',
              '建立可审计的安全治理与人工接管机。',
            ],
            tests: [
              {
                name: '自适应选路策略验证',
                params: 'policy=bandit, reward=feasibility_score',
                dataset: 'Long-Horizon Route Selection Set',
                result: '待执行',
                summary: '用于验证长期任务中的选路收敛能力。',
              },
              {
                name: '全链路安全审计测。',
                params: 'audit=full, intervention=manual+auto',
                dataset: 'Governance Compliance Set',
                result: '待执行',
                summary: '用于评估可审计性与可控性是否达标。',
              },
            ],
          },
        ],
        milestonePlanEvaluation: {
          assessment:
            '具备成为“路线操作系统”的潜力，关键在于稳定调度与治理可控性。',
          suggestions: [
            '优先优化并发场景下尾延迟与资源竞争问题。',
            '将MTTR纳入核心KPI并周度跟踪。',
            '提前定义人工接管触发条件与回退策略。',
          ],
        },
      },
    }),
    [engineeringPhase?.sub_phases, milestonePhase?.goals, milestonePhase?.metrics, theoryPhase?.definition?.headline, theoryPhase?.definition?.summary, theoryPhase?.theory_content]
  );

  const routeList = useMemo(() => {
    const runtimeIds = timelineRoutes
      .map((item) => item?.route)
      .filter((id) => typeof id === 'string' && id.length > 0);
    const baseIds = Object.keys(routeBlueprints);
    const allIds = Array.from(new Set([...baseIds, ...runtimeIds]));

    return allIds.map((id) => {
      const base = routeBlueprints[id] || {
        id,
        title: id,
        subtitle: '实验路线',
        routeDescription: '该路线正在构建中，描述信息待补充。',
        engineeringProcessDescription: '计算流程说明待补充。',
        theoryTitle: '寰呰ˉ鍏呯悊璁?',
        theorySummary: '该路线尚未配置详细理论描述。',
        theoryBullets: [],
        theoryFormulas: [],
        engineeringItems: [],
        nfbtProcessSteps: [],
        nfbtOptimization: '',
        milestoneTitle: '里程碑目标（。AGI 终点。',
        milestoneGoals: [],
        milestoneMetrics: {},
        milestoneStages: [],
        milestonePlanEvaluation: null,
      };
      const runtime = timelineRoutes.find((item) => item?.route === id);
      const stats = runtime?.stats || {};
      const totalRuns = Number(stats.total_runs || 0);
      const completedRuns = Number(stats.completed_runs || 0);
      const avgScore = Number(stats.avg_score || 0);
      const routeProgress =
        totalRuns > 0
          ? Math.max(
            0,
            Math.min(100, Math.round((completedRuns / Math.max(1, totalRuns)) * 60 + avgScore * 40))
          )
          : 0;
      return {
        ...base,
        stats: {
          totalRuns,
          completedRuns,
          failedRuns: Number(stats.failed_runs || 0),
          avgScore,
          routeProgress,
        },
      };
    });
  }, [routeBlueprints, timelineRoutes]);

  useEffect(() => {
    if (routeList.length === 0) return;
    if (!routeList.some((item) => item.id === selectedRouteId)) {
      setSelectedRouteId(routeList[0].id);
    }
  }, [routeList, selectedRouteId]);

  useEffect(() => {
    setExpandedFormulaIdx(null);
    setExpandedEngPhase(null);
    setExpandedParam(null);
  }, [selectedRouteId]);

  const selectedRoute = routeList.find((item) => item.id === selectedRouteId) || routeList[0];
  const systemRouteOptions = routeList.filter((item) =>
    ['fiber_bundle', 'transformer_baseline', 'hybrid_workspace'].includes(item.id)
  );
  const selectedMultimodalData = multimodalSummary?.views?.[multimodalView] || null;
  const selectedMultimodalReport = selectedMultimodalData?.report || null;
  const selectedMultimodalBest = selectedMultimodalReport?.summary?.best || null;
  const selectedMultimodalLatest = selectedMultimodalData?.latest_test || null;

  const multimodalMetricRows = useMemo(() => {
    if (!selectedMultimodalBest) return [];
    if (multimodalView === 'vision_alignment') {
      return [
        { label: '最佳轮次', value: selectedMultimodalBest.epoch },
        { label: 'Val Accuracy', value: Number(selectedMultimodalBest.val_acc || 0).toFixed(4) },
        { label: 'Anchor Cos', value: Number(selectedMultimodalBest.val_anchor_cos || 0).toFixed(4) },
        { label: 'Val Loss', value: Number(selectedMultimodalBest.val_loss || 0).toFixed(4) },
      ];
    }
    return [
      { label: '最佳轮次', value: selectedMultimodalBest.epoch },
      { label: 'Val Fused Acc', value: Number(selectedMultimodalBest.val_fused_acc || 0).toFixed(4) },
      { label: 'Retrieval@1', value: Number(selectedMultimodalBest.val_retrieval_top1 || 0).toFixed(4) },
      { label: 'Align Cos', value: Number(selectedMultimodalBest.val_alignment_cos || 0).toFixed(4) },
    ];
  }, [selectedMultimodalBest, multimodalView]);

  const getRouteImpl = (capability) => {
    const map = capability?.implementation_by_route || {};
    return (
      map[selectedRouteId] ||
      map[selectedRoute?.id] ||
      capability?.desc ||
      '该路线实现描述待补充。'
    );
  };

  const systemProfiles = useMemo(
    () => ({
      fiber_bundle: {
        metricCards: [
          {
            label: '内稳态调。',
            brain_ability: '稳态维持与资源分配',
            value: consciousField ? `${((consciousField.stability || 0) * 100).toFixed(1)}%` : '92.0%',
            color: '#10b981',
          },
          {
            label: '工作记忆负载',
            brain_ability: '短时记忆与上下文保持',
            value: consciousField ? `${consciousField.memory_load || 0}%` : '68%',
            color: '#00d2ff',
          },
          {
            label: '跨域共振',
            brain_ability: '跨模态联想整。',
            value: consciousField ? (consciousField.resonance || 0).toFixed(3) : '0.742',
            color: '#ffaa00',
          },
          {
            label: '意识竞争强度',
            brain_ability: '注意焦点竞争与广。',
            value: consciousField ? (consciousField.gws_intensity || 0).toFixed(2) : '0.81',
            color: '#a855f7',
          },
        ],
        parameterCards: [
          {
            name: '几何潜空间配。',
            brain_ability: '抽象结构建模',
            route_param: 'd=4, D=128, manifold=riemannian',
            detail: '低维几何。+ 高维语义外壳',
            desc: '通过 d<<D 降低核心几何计算复杂度，同时保持语义表达容量。',
            value_meaning: '兼顾可解释性、稳定性与计算成本。',
            why_important: '决定几何推理是否可持续扩展。',
          },
          {
            name: '联络与平行移。',
            brain_ability: '推理路径保持',
            route_param: 'transport=connection_based, step=adaptive',
            detail: 'Γ 驱动语义平移',
            desc: '沿流形执行平行移动，减少语义漂移。',
            value_meaning: '推理链更稳定，抗扰动能力更强。',
            why_important: '是从“拟合”走向“结构推理”的关键。'
          },
          {
            name: 'Ricci Flow 婕斿寲',
            brain_ability: '睡眠重整与冲突修。',
            route_param: 'iterations=100, reg=1e-3',
            detail: '离线曲率平滑',
            desc: '通过流形平滑降低逻辑尖峰与幻觉风险。',
            value_meaning: '提升长期稳定性与一致性。',
            why_important: '支持系统持续自我修复。'
          },
          {
            name: '全局工作空间',
            brain_ability: '意识竞争裁决',
            route_param: 'top_k=8, arbitration=winner_take_all',
            detail: '多模块竞争广。',
            desc: '在冲突候选中选取最优表示并广播。',
            value_meaning: '保证实时决策聚焦有效信息。',
            why_important: '直接影响系统响应质量与时延。'
          },
        ],
        validationRecords: (statusData?.passed_tests || []).map((t) => ({
          ...t,
          brain_ability: t.brain_ability || '结构推理稳定与记忆重。',
          route_param_focus: t.route_param_focus || 'manifold_dim=4, top_k=8, ricci_iterations=100',
        })),
      },
      transformer_baseline: {
        metricCards: [
          { label: '上下文保持', brain_ability: '工作记忆', value: '74%', color: '#10b981' },
          { label: '模式泛化', brain_ability: '经验迁移', value: '0.69', color: '#00d2ff' },
          { label: '可解释覆盖', brain_ability: '自我监控', value: '83%', color: '#ffaa00' },
          { label: '推理稳定性', brain_ability: '执行控制', value: '0.71', color: '#a855f7' },
        ],
        parameterCards: [
          {
            name: '模型与序列配。',
            brain_ability: '语言工作记忆',
            route_param: 'model=gpt2-small, seq_len=512, batch=16',
            detail: '标准 Transformer 主干',
            desc: '通过标准注意力机制建模上下文依赖。',
            value_meaning: '易复现、生态成熟。',
            why_important: '是对照路线的基础能力锚点。',
          },
          {
            name: '可解释性探。',
            brain_ability: '内省与自检',
            route_param: 'logit_lens=on, tda=on, head_probe=full',
            detail: '多探针并。',
            desc: '输出层到中间层的解释链路可追踪。',
            value_meaning: '便于定位错误来源。',
            why_important: '支撑实验可解释与回归分析。',
          },
          {
            name: '训练稳定性策。',
            brain_ability: '执行控制与抑。',
            route_param: 'lr=1e-4, warmup=2k, clip=1.0',
            detail: '标准优化管线',
            desc: '控制梯度震荡与训练发散风险。',
            value_meaning: '提升训练一致性。',
            why_important: '影响规模化阶段可持续性。',
          },
          {
            name: 'RAG 鎵╁睍',
            brain_ability: '长期知识提取',
            route_param: 'retriever=faiss, topk=5, rerank=on',
            detail: '外部知识增强',
            desc: '通过检索补偿参数内知识不足。',
            value_meaning: '提高事实一致性。',
            why_important: '降低知识时效衰减。'
          },
        ],
        validationRecords: [
          {
            name: '多任务基线回。',
            date: '2026-02-16',
            result: 'PASS',
            brain_ability: '工作记忆稳定与任务切换',
            route_param_focus: 'seq_len=512, batch=16, clip=1.0',
            target: '验证标准路线在多任务上的稳定性能。',
            process: '统一任务集批量回。+ 误差曲线对齐。',
            significance: '确认基线可作为长期对照参照系。'
          },
          {
            name: '解释链完整性测。',
            date: '2026-02-16',
            result: 'PASS',
            brain_ability: '内省监控与因果追踪',
            route_param_focus: 'logit_lens=on, tda=on, head_probe=full',
            target: '验证从激活到输出的可追溯性。',
            process: 'Logit Lens + 头部归因联合验证。',
            significance: '确保分析结论具备可复检性。'
          },
        ],
      },
      hybrid_workspace: {
        metricCards: [
          { label: '跨路线一致性', brain_ability: '跨脑区整合', value: '0.76', color: '#10b981' },
          { label: '仲裁收敛速度', brain_ability: '注意竞争调度', value: '148ms', color: '#00d2ff' },
          { label: '故障恢复能力', brain_ability: '稳态恢复', value: 'MTTR 2.6m', color: '#ffaa00' },
          { label: '融合收益', brain_ability: '多通道协同', value: '+8.9%', color: '#a855f7' },
        ],
        parameterCards: [
          {
            name: '仲裁器参。',
            brain_ability: '鍐茬獊鍐崇瓥',
            route_param: 'routes=5, top_k=2, timeout=200ms',
            detail: '多路线竞争框。',
            desc: '基于评分和稳定性进行候选筛选。',
            value_meaning: '平衡质量与时延。',
            why_important: '决定融合输出可用性。',
          },
          {
            name: '融合策略参数',
            brain_ability: '多源信息整合',
            route_param: 'fusion=weighted, dynamic_weight=on',
            detail: '动态权重融。',
            desc: '根据路线可靠度动态调节贡献。',
            value_meaning: '减少单路线失效影响。',
            why_important: '提高鲁棒性与连续性。',
          },
          {
            name: '鎭㈠涓庨檷绾?',
            brain_ability: '异常处理',
            route_param: 'fallback=enabled, retry=2, degrade=graceful',
            detail: '故障容错机制',
            desc: '出现异常时自动切换到安全路径。',
            value_meaning: '避免全链路中断。',
            why_important: '保障系统稳定运行。'
          },
          {
            name: '治理追踪参数',
            brain_ability: '长期自我监督',
            route_param: 'timeline=on, weekly_report=on',
            detail: '全链路可审计',
            desc: '持续记录决策证据与失败归因。',
            value_meaning: '支持迭代改进闭环。',
            why_important: '提升工程治理效率。'
          },
        ],
        validationRecords: [
          {
            name: '多路线并发仲裁测。',
            date: '2026-02-17',
            result: 'PASS',
            brain_ability: '冲突决策与全局调度',
            route_param_focus: 'routes=5, top_k=2, timeout=200ms',
            target: '验证多路线并发下仲裁稳定性。',
            process: '构造冲突任务，统计收敛时间与一致性。',
            significance: '确认仲裁机制在复杂场景可用。'
          },
          {
            name: '故障注入恢复测试',
            date: '2026-02-17',
            result: 'PASS',
            brain_ability: '稳态恢复与容错控制',
            route_param_focus: 'fallback=enabled, retry=2, degrade=graceful',
            target: '验证路由失败时自动降级能力。',
            process: '注入随机失败，观。MTTR 与回退质量。',
            significance: '证明混合路线具备工程级容错能力。'
          },
        ],
      },
    }),
    [consciousField, selectedRouteId, statusData?.passed_tests]
  );

  const activeSystemProfile = systemProfiles[selectedRouteId] || systemProfiles.fiber_bundle;
  const mergedMilestoneStages = useMemo(() => {
    const baseStages = selectedRoute?.milestoneStages || [];
    const routeTests = activeSystemProfile?.validationRecords || [];
    if (!routeTests.length) return baseStages;

    const routeValidationStage = {
      id: 'route_validation',
      name: '路线测试记录',
      status: routeTests.every((t) => String(t?.result || '').toUpperCase().includes('PASS')) ? 'done' : 'in_progress',
      featurePoints: [
        `来源：系统状态 / ${selectedRoute?.title || selectedRouteId}`,
        `测试数量：${routeTests.length}`,
        '作为里程碑验收证据沉淀到研发进展',
      ],
      tests: routeTests.map((t) => ({
        name: t.name || '未命名测试',
        params: t.route_param_focus || t.params || '-',
        dataset: t.dataset || (t.date ? `验证日期: ${t.date}` : '-'),
        result: t.result || '-',
        summary: t.significance || t.summary || '-',
      })),
    };

    return [...baseStages, routeValidationStage];
  }, [selectedRoute, selectedRouteId, activeSystemProfile]);

  // Brain Model Component
  const BrainModel = () => (
    <div style={{
      width: '320px', height: '320px', margin: '0 auto', position: 'relative',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'radial-gradient(circle, rgba(0, 210, 255, 0.15) 0%, transparent 70%)',
      borderRadius: '50%', animation: 'brainPulse 4s infinite alternate'
    }}>
      <div style={{
        position: 'absolute', width: '100%', height: '100%', border: '1px solid rgba(0, 210, 255, 0.2)',
        borderRadius: '50%', animation: 'brainRotate 20s linear infinite'
      }} />
      <div style={{
        position: 'absolute', width: '85%', height: '85%', border: '1px dashed rgba(168, 85, 247, 0.3)',
        borderRadius: '50%', animation: 'brainRotateReverse 15s linear infinite'
      }} />
      <Brain size={180} color="#00d2ff" style={{
        filter: 'drop-shadow(0 0 30px rgba(0, 210, 255, 0.4))',
        zIndex: 2
      }} />
      {/* Dynamic Synaptic Nodes */}
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} style={{
          position: 'absolute', width: '6px', height: '6px', background: '#00ff88', borderRadius: '50%',
          boxShadow: '0 0 10px #00ff88', transform: `rotate(${i * 45}deg) translateY(-130px)`,
          animation: 'synapsePulse 2s infinite', animationDelay: `${i * 0.2}s`
        }} />
      ))}
    </div>
  );

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh',
      backgroundColor: 'rgba(5, 5, 10, 0.98)', backdropFilter: 'blur(30px)', zIndex: 2000,
      display: 'flex', flexDirection: 'column', color: '#fff',
      fontFamily: '"SF Mono", "Roboto Mono", monospace', overflow: 'hidden'
    }}>
      {/* Custom Keyframes */}
      <style>{`
        @keyframes roadmapFade { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes brainPulse { from { scale: 0.95; opacity: 0.8; } to { scale: 1.05; opacity: 1; } }
        @keyframes brainRotate { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes brainRotateReverse { from { transform: rotate(0deg); } to { transform: rotate(-360deg); } }
        @keyframes synapsePulse { 0%, 100% { opacity: 0.3; scale: 0.8; } 50% { opacity: 1; scale: 1.2; } }
      `}</style>

      {/* Top Header / Navigation */}
      <div style={{
        padding: '0 40px', height: '80px', display: 'flex', justifyContent: 'space-between',
        alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.1)', background: 'rgba(0,0,0,0.3)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '50px', height: '100%' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Brain size={28} color="#00d2ff" />
            <span style={{ fontSize: '18px', fontWeight: 'bold', letterSpacing: '2px' }}>智能一号</span>
          </div>

          <nav style={{ display: 'flex', gap: '10px', height: '100%' }}>
            {[
              { id: 'roadmap', label: '项目大纲' },
              { id: 'progress', label: '研发进展' },
              { id: 'system', label: '系统状态' }
            ].map(t => (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                style={{
                  background: 'transparent', border: 'none', color: activeTab === t.id ? '#00d2ff' : '#666',
                  fontSize: '15px', fontWeight: 'bold', cursor: 'pointer', padding: '0 25px',
                  borderBottom: activeTab === t.id ? '3px solid #00d2ff' : '3px solid transparent',
                  transition: 'all 0.3s', height: '100%'
                }}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>

        <button onClick={onClose} style={{
          background: 'rgba(255,255,255,0.05)', border: 'none', color: '#fff', cursor: 'pointer',
          width: '40px', height: '40px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center'
        }} onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,100,100,0.2)'} onMouseLeave={e => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}>
          <X size={22} />
        </button>
      </div>

      {/* Main Content Area */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

        {/* Sub-Sidebar for Research Progress */}
        {activeTab === 'progress' && (
          <div style={{
            width: '280px', borderRight: '1px solid rgba(255,255,255,0.1)',
            padding: '30px 20px', background: 'rgba(0,0,0,0.2)', overflowY: 'auto',
            position: 'relative'
          }}>
            <div style={{ fontSize: '10px', color: '#444', textTransform: 'uppercase', marginBottom: '30px', letterSpacing: '2px', fontWeight: 'bold' }}>Research Routes</div>

            {/* Vertical Timeline Line */}
            <div style={{
              position: 'absolute', left: '38px', top: '80px', bottom: '40px',
              width: '1px', background: 'linear-gradient(to bottom, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%)',
              zIndex: 0
            }} />

            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', position: 'relative', zIndex: 1 }}>
              {routeList.map((routeItem) => (
                <div key={routeItem.id} style={{ position: 'relative' }}>
                  {/* Timeline Dot */}
                  <div style={{
                    position: 'absolute', left: '15px', top: '22px',
                    width: '6px', height: '6px', borderRadius: '50%',
                    background: selectedRoute?.id === routeItem.id ? '#00d2ff' : '#222',
                    border: `2px solid ${selectedRoute?.id === routeItem.id ? '#000' : 'rgba(255,255,255,0.1)'}`,
                    boxShadow: selectedRoute?.id === routeItem.id ? '0 0 10px #00d2ff' : 'none',
                    transition: 'all 0.3s'
                  }} />

                  <button
                    onClick={() => setSelectedRouteId(routeItem.id)}
                    style={{
                      width: '100%', padding: '12px 12px 12px 45px', borderRadius: '14px',
                      textAlign: 'left', cursor: 'pointer',
                      background: selectedRoute?.id === routeItem.id ? 'rgba(255,255,255,0.03)' : 'transparent',
                      border: 'none',
                      color: selectedRoute?.id === routeItem.id ? '#fff' : '#666', transition: 'all 0.3s',
                      display: 'flex', flexDirection: 'column', gap: '4px'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                      <span style={{ fontSize: '14px', fontWeight: 'bold', color: selectedRoute?.id === routeItem.id ? '#fff' : '#888' }}>
                        {routeItem.title}
                      </span>
                      <span style={{
                        fontSize: '11px', fontFamily: 'monospace', fontWeight: 'bold',
                        color: selectedRoute?.id === routeItem.id ? '#00d2ff' : '#444'
                      }}>
                        {routeItem.stats.routeProgress}%
                      </span>
                    </div>
                    <div style={{ fontSize: '10px', color: '#444' }}>
                      run {routeItem.stats.totalRuns} | success {routeItem.stats.completedRuns} | avg {(routeItem.stats.avgScore * 100).toFixed(1)}%
                    </div>
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Content Details */}
        <div style={{
          flex: 1, padding: '50px 80px', overflowY: 'auto',
          background: 'radial-gradient(circle at 50% 10%, rgba(0, 100, 200, 0.05) 0%, transparent 70%)'
        }}>

          {/* TAB: Project Roadmap */}
          {activeTab === 'roadmap' && (
            <div style={{ animation: 'roadmapFade 0.6s ease-out', maxWidth: '1000px', margin: '0 auto' }}>
              <div style={{ marginBottom: '34px' }}>
                <h2 style={{ fontSize: '30px', fontWeight: '900', color: '#ffaa00', marginBottom: '10px' }}>项目大纲</h2>
                <div style={{ color: '#777', fontSize: '14px' }}>{roadmapData.definition.summary}</div>
              </div>

              {/* 上：项目核心思路 */}
              <div style={{
                padding: '30px',
                background: 'linear-gradient(135deg, rgba(255,170,0,0.12) 0%, rgba(255,170,0,0.03) 100%)',
                border: '1px solid rgba(255,170,0,0.24)',
                borderRadius: '24px',
                marginBottom: '28px'
              }}>
                <div style={{ color: '#ffaa00', fontWeight: 'bold', fontSize: '18px', marginBottom: '16px' }}>项目核心思路</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '10px' }}>
                  {[
                    '1，大脑有非常特殊的数学结构，产生了智能。',
                    '2，深度神经网络部分还原了这个结构，产生了语言能力。',
                    '3，通过分析深度神经网络，研究这个数学结构，完成智能理论。',
                  ].map((line, idx) => (
                    <div key={idx} style={{ padding: '14px 16px', borderRadius: '12px', background: 'rgba(255,255,255,0.05)', color: '#f4e4c1', fontSize: '14px', lineHeight: '1.6' }}>
                      {line}
                    </div>
                  ))}
                </div>
              </div>

              <div style={{
                padding: '30px',
                borderRadius: '24px',
                border: '1px solid rgba(99,102,241,0.28)',
                background: 'linear-gradient(135deg, rgba(99,102,241,0.10) 0%, rgba(99,102,241,0.03) 100%)',
                marginBottom: '28px'
              }}>
                <div style={{ color: '#818cf8', fontWeight: 'bold', fontSize: '18px', marginBottom: '8px' }}>
                  {MATH_ROUTE_SYSTEM_PLAN.title}
                </div>
                <div style={{ color: '#c7d2fe', fontSize: '13px', lineHeight: '1.7', marginBottom: '14px' }}>
                  {MATH_ROUTE_SYSTEM_PLAN.subtitle}
                </div>

                <div style={{ marginTop: '12px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0,0,0,0.22)', padding: '12px' }}>
                  <div style={{ color: '#a5b4fc', fontSize: '11px', fontWeight: 'bold', marginBottom: '8px' }}>
                    数学路线
                  </div>
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', minWidth: '1260px', borderCollapse: 'collapse' }}>
                      <thead>
                        <tr style={{ background: 'rgba(255,255,255,0.05)' }}>
                          <th style={{ textAlign: 'left', padding: '8px 10px', color: '#c7d2fe', fontSize: '11px', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>路线</th>
                          <th style={{ textAlign: 'left', padding: '8px 10px', color: '#c7d2fe', fontSize: '11px', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>理论深度</th>
                          <th style={{ textAlign: 'left', padding: '8px 10px', color: '#c7d2fe', fontSize: '11px', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>计算可行性</th>
                          <th style={{ textAlign: 'left', padding: '8px 10px', color: '#c7d2fe', fontSize: '11px', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>可解释性</th>
                          <th style={{ textAlign: 'left', padding: '8px 10px', color: '#c7d2fe', fontSize: '11px', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>与 SHMC/NFBT 兼容</th>
                          <th style={{ textAlign: 'left', padding: '8px 10px', color: '#86efac', fontSize: '11px', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>优点</th>
                          <th style={{ textAlign: 'left', padding: '8px 10px', color: '#fca5a5', fontSize: '11px', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>缺点</th>
                          <th style={{ textAlign: 'left', padding: '8px 10px', color: '#93c5fd', fontSize: '11px', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>可行性结论</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(MATH_ROUTE_SYSTEM_PLAN.routeAnalysis || []).map((item, idx) => (
                          <tr key={idx} style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                            <td style={{ padding: '9px 10px', color: '#e0e7ff', fontSize: '12px', fontWeight: 'bold', verticalAlign: 'top' }}>{item.route}</td>
                            <td style={{ padding: '9px 10px', color: '#dbeafe', fontSize: '12px', verticalAlign: 'top' }}>{item.depth}</td>
                            <td style={{ padding: '9px 10px', color: '#dbeafe', fontSize: '12px', verticalAlign: 'top' }}>{item.compute}</td>
                            <td style={{ padding: '9px 10px', color: '#dbeafe', fontSize: '12px', verticalAlign: 'top' }}>{item.interpret}</td>
                            <td style={{ padding: '9px 10px', color: '#dbeafe', fontSize: '12px', verticalAlign: 'top' }}>{item.compatibility}</td>
                            <td style={{ padding: '9px 10px', color: '#dcfce7', fontSize: '11px', lineHeight: '1.55', verticalAlign: 'top' }}>
                              {(item.pros || []).map((line, pIdx) => (
                                <div key={pIdx}>{pIdx + 1}. {line}</div>
                              ))}
                            </td>
                            <td style={{ padding: '9px 10px', color: '#fee2e2', fontSize: '11px', lineHeight: '1.55', verticalAlign: 'top' }}>
                              {(item.cons || []).map((line, cIdx) => (
                                <div key={cIdx}>{cIdx + 1}. {line}</div>
                              ))}
                            </td>
                            <td style={{ padding: '9px 10px', color: '#bae6fd', fontSize: '11px', lineHeight: '1.55', verticalAlign: 'top' }}>{item.feasibility}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div style={{ marginTop: '12px', display: 'grid', gridTemplateColumns: '1.2fr 1fr 1fr', gap: '12px' }}>
                  <div style={{ padding: '14px', borderRadius: '12px', background: 'rgba(0,0,0,0.22)', border: '1px solid rgba(255,255,255,0.08)' }}>
                    <div style={{ color: '#a5b4fc', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px' }}>分层架构</div>
                    {(MATH_ROUTE_SYSTEM_PLAN.architecture || []).map((line, idx) => (
                      <div key={idx} style={{ color: '#e0e7ff', fontSize: '12px', lineHeight: '1.6', marginBottom: '4px' }}>
                        {idx + 1}. {line}
                      </div>
                    ))}
                  </div>

                  <div style={{ padding: '14px', borderRadius: '12px', background: 'rgba(0,0,0,0.22)', border: '1px solid rgba(255,255,255,0.08)' }}>
                    <div style={{ color: '#a5b4fc', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px' }}>资源配比</div>
                    {(MATH_ROUTE_SYSTEM_PLAN.allocation || []).map((line, idx) => (
                      <div key={idx} style={{ color: '#dbeafe', fontSize: '12px', lineHeight: '1.6', marginBottom: '4px' }}>
                        {line}
                      </div>
                    ))}
                  </div>

                  <div style={{ padding: '14px', borderRadius: '12px', background: 'rgba(0,0,0,0.22)', border: '1px solid rgba(255,255,255,0.08)' }}>
                    <div style={{ color: '#a5b4fc', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px' }}>阶段里程碑</div>
                    {(MATH_ROUTE_SYSTEM_PLAN.milestones || []).map((line, idx) => (
                      <div key={idx} style={{ color: '#dbeafe', fontSize: '12px', lineHeight: '1.6', marginBottom: '4px' }}>
                        {idx + 1}. {line}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* 中：DNN结构分析 */}
              <div style={{
                padding: '30px',
                borderRadius: '24px',
                border: '1px solid rgba(168, 85, 247, 0.25)',
                background: 'linear-gradient(135deg, rgba(168, 85, 247, 0.09) 0%, rgba(168, 85, 247, 0.02) 100%)',
                marginBottom: '28px'
              }}>
                <div style={{ color: '#a855f7', fontWeight: 'bold', fontSize: '18px', marginBottom: '18px' }}>
                  对深度神经网络核心结构的分析
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                  {(analysisPhase?.analysis_sections || []).map((item, idx) => (
                    <div key={idx} style={{ padding: '18px', borderRadius: '14px', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
                      <div style={{ fontWeight: 'bold', color: '#e9d5ff', marginBottom: '8px', fontSize: '14px' }}>{item.title}</div>
                      <div style={{ fontSize: '12px', color: '#aaa', lineHeight: '1.65', marginBottom: '10px' }}>{item.content}</div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                        {(item.tags || []).map((tag, tagIdx) => (
                          <span key={tagIdx} style={{ fontSize: '10px', color: '#c4b5fd', border: '1px solid rgba(196,181,253,0.35)', borderRadius: '999px', padding: '2px 8px' }}>
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* 下：分析成果列表 */}
              <div style={{
                padding: '30px',
                borderRadius: '24px',
                border: '1px solid rgba(16,185,129,0.24)',
                background: 'linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(16,185,129,0.02) 100%)'
              }}>
                <div style={{ color: '#10b981', fontWeight: 'bold', fontSize: '18px', marginBottom: '8px' }}>
                  分析成果
                </div>
                <div style={{ color: '#9ca3af', fontSize: '13px', lineHeight: '1.7', marginBottom: '16px' }}>
                  下面是当前项目对于深度神经网络的分析方案，以及对应的结果数据
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '10px' }}>
                  {IMPROVEMENTS.map((item, idx) => (
                    <div key={idx} style={{ padding: '14px 16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(255,255,255,0.02)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', marginBottom: '6px' }}>
                        <div style={{ color: '#dcfce7', fontWeight: 'bold', fontSize: '14px' }}>{item.title}</div>
                        <div style={{ fontSize: '10px', color: item.status === 'done' ? '#10b981' : '#f59e0b' }}>{String(item.status).toUpperCase()}</div>
                      </div>
                      <div style={{ color: '#94a3b8', fontSize: '12px', marginBottom: '4px' }}>{item.desc} | {item.step}</div>
                      <div style={{ color: '#a7f3d0', fontSize: '12px', lineHeight: '1.6' }}>{item.detail}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* TAB: Research Progress (Route-Centric Command) */}
          {activeTab === 'progress' && selectedRoute && (
            <div style={{ animation: 'roadmapFade 0.5s ease-out' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '34px' }}>
                <div>
                  <h2 style={{ fontSize: '34px', fontWeight: '900', color: '#fff', margin: '0 0 8px 0' }}>
                    {selectedRoute.title} - {selectedRoute.subtitle}
                  </h2>
                  <div style={{ marginTop: '8px', color: '#666', fontSize: '13px' }}>
                    {selectedRoute.routeDescription || selectedRoute.theorySummary}
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '30px', fontWeight: '900', color: '#00d2ff', fontFamily: 'monospace' }}>
                    {selectedRoute.stats.routeProgress}%
                  </div>
                  <div style={{ fontSize: '10px', color: '#444' }}>ROUTE READINESS</div>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '22px' }}>
                <div style={{ padding: '28px', background: 'rgba(0, 210, 255, 0.06)', border: '1px solid rgba(0, 210, 255, 0.25)', borderRadius: '22px' }}>
                  <div style={{ fontSize: '12px', color: '#00d2ff', fontWeight: 'bold', letterSpacing: '2px', marginBottom: '10px' }}>智能理论</div>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#fff', marginBottom: '10px' }}>{selectedRoute.theoryTitle}</div>
                  <div style={{ fontSize: '14px', color: '#bbb', lineHeight: '1.8', marginBottom: '14px' }}>{selectedRoute.theorySummary}</div>
                  {(selectedRoute.theoryFormulas || []).length === 0 ? (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                      {(selectedRoute.theoryBullets || []).map((item, idx) => (
                        <div key={idx} style={{ padding: '12px 14px', borderRadius: '12px', background: 'rgba(255,255,255,0.03)', color: '#ddd', fontSize: '12px' }}>
                          {item}
                        </div>
                      ))}
                    </div>
                  ) : null}
                  {(selectedRoute.theoryFormulas || []).length > 0 ? (
                    <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                      {selectedRoute.theoryFormulas.map((item, idx) => {
                        const expanded = expandedFormulaIdx === idx;
                        return (
                          <div
                            key={idx}
                            onClick={() => setExpandedFormulaIdx(expanded ? null : idx)}
                            style={{
                              padding: '12px 14px',
                              borderRadius: '12px',
                              border: expanded ? '1px solid rgba(103, 232, 249, 0.65)' : '1px solid rgba(0,210,255,0.35)',
                              background: expanded ? 'rgba(0, 210, 255, 0.12)' : 'rgba(0,0,0,0.3)',
                              cursor: 'pointer',
                              transition: 'all 0.2s ease',
                            }}
                          >
                            <div style={{ fontSize: '11px', color: '#67e8f9', marginBottom: '6px', fontWeight: 'bold' }}>
                              {item.title}
                            </div>
                            <div style={{ fontSize: '18px', color: '#e0f2fe', fontFamily: 'serif' }}>{item.formula}</div>
                            {expanded ? (
                              <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid rgba(255,255,255,0.12)', fontSize: '12px', color: '#cffafe', lineHeight: '1.65' }}>
                                {item.detail || '暂无详细说明'}
                              </div>
                            ) : (
                              <div style={{ marginTop: '8px', fontSize: '10px', color: '#7dd3fc' }}>点击展开详细说明</div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  ) : null}
                </div>

                <div style={{ padding: '28px', background: 'rgba(99, 102, 241, 0.06)', border: '1px solid rgba(99, 102, 241, 0.26)', borderRadius: '22px' }}>
                  <div style={{ fontSize: '12px', color: '#818cf8', fontWeight: 'bold', letterSpacing: '2px', marginBottom: '10px' }}>
                    深度神经网络分析
                  </div>
                  <div style={{ fontSize: '20px', color: '#fff', fontWeight: 'bold', marginBottom: '8px' }}>
                    {DNN_ANALYSIS_PLAN.subtitle}
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '12px' }}>
                    <div style={{ borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0,0,0,0.22)', padding: '12px' }}>
                      <div style={{ fontSize: '11px', color: '#a5b4fc', fontWeight: 'bold', marginBottom: '6px' }}>研究目标与假设</div>
                      {(DNN_ANALYSIS_PLAN.goals || []).map((item, idx) => (
                        <div key={idx} style={{ fontSize: '12px', color: '#e0e7ff', lineHeight: '1.6', marginBottom: '4px' }}>
                          {idx + 1}. {item}
                        </div>
                      ))}
                    </div>
                    <div style={{ borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0,0,0,0.22)', padding: '12px' }}>
                      <div style={{ fontSize: '11px', color: '#a5b4fc', fontWeight: 'bold', marginBottom: '6px' }}>核心评估指标</div>
                      {(DNN_ANALYSIS_PLAN.metrics || []).map((item, idx) => (
                        <div key={idx} style={{ fontSize: '12px', color: '#dbeafe', lineHeight: '1.6', marginBottom: '4px' }}>
                          {idx + 1}. {item}
                        </div>
                      ))}
                    </div>
                  </div>

                  <div style={{ borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0,0,0,0.22)', padding: '12px', marginBottom: '12px' }}>
                    <div style={{ fontSize: '11px', color: '#a5b4fc', fontWeight: 'bold', marginBottom: '6px' }}>六层分析框架</div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                      {(DNN_ANALYSIS_PLAN.framework || []).map((item, idx) => (
                        <div key={idx} style={{ fontSize: '12px', color: '#c7d2fe', lineHeight: '1.6', padding: '8px 10px', borderRadius: '10px', background: 'rgba(255,255,255,0.03)' }}>
                          {item}
                        </div>
                      ))}
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px' }}>
                    <div style={{ borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0,0,0,0.22)', padding: '12px' }}>
                      <div style={{ fontSize: '11px', color: '#a5b4fc', fontWeight: 'bold', marginBottom: '6px' }}>实验矩阵</div>
                      {(DNN_ANALYSIS_PLAN.experimentMatrix || []).map((item, idx) => (
                        <div key={idx} style={{ fontSize: '12px', color: '#dbeafe', lineHeight: '1.6', marginBottom: '4px' }}>
                          {idx + 1}. {item}
                        </div>
                      ))}
                    </div>
                    <div style={{ borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0,0,0,0.22)', padding: '12px' }}>
                      <div style={{ fontSize: '11px', color: '#a5b4fc', fontWeight: 'bold', marginBottom: '6px' }}>里程碑</div>
                      {(DNN_ANALYSIS_PLAN.milestones || []).map((item, idx) => (
                        <div key={idx} style={{ fontSize: '12px', color: '#dbeafe', lineHeight: '1.6', marginBottom: '4px' }}>
                          {item}
                        </div>
                      ))}
                    </div>
                    <div style={{ borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0,0,0,0.22)', padding: '12px' }}>
                      <div style={{ fontSize: '11px', color: '#a5b4fc', fontWeight: 'bold', marginBottom: '6px' }}>成败判据</div>
                      {(DNN_ANALYSIS_PLAN.successCriteria || []).map((item, idx) => (
                        <div key={idx} style={{ fontSize: '12px', color: '#dbeafe', lineHeight: '1.6', marginBottom: '4px' }}>
                          {item}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div style={{ padding: '28px', background: 'rgba(245, 158, 11, 0.05)', border: '1px solid rgba(245, 158, 11, 0.22)', borderRadius: '22px' }}>
                  <div style={{ fontSize: '12px', color: '#f59e0b', fontWeight: 'bold', letterSpacing: '2px', marginBottom: '14px' }}>工程实现</div>
                  <div style={{ marginBottom: '12px', padding: '10px 12px', borderRadius: '10px', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', fontSize: '12px', color: '#f8d7a6', lineHeight: '1.65' }}>
                    计算过程说明：{selectedRoute.engineeringProcessDescription || '该路线计算过程说明待补充。'}
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    {(selectedRoute.engineeringItems || []).map((item, idx) => (
                      <div key={idx} style={{ borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(255,255,255,0.02)' }}>
                        <div
                          onClick={() => setExpandedEngPhase(expandedEngPhase === idx ? null : idx)}
                          style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '14px 16px' }}
                        >
                          <div>
                            <div style={{ color: '#fff', fontWeight: 'bold', fontSize: '14px' }}>{item.name}</div>
                            <div style={{ color: '#777', fontSize: '11px', marginTop: '3px' }}>{item.focus}</div>
                          </div>
                          <div style={{ fontSize: '10px', color: item.status === 'done' ? '#10b981' : item.status === 'in_progress' ? '#f59e0b' : '#666' }}>
                            {String(item.status || 'pending').toUpperCase()}
                          </div>
                        </div>
                        {expandedEngPhase === idx && (
                          <div style={{ padding: '0 16px 14px 16px', color: '#aaa', fontSize: '12px', lineHeight: '1.6', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                            {item.detail || item.analysis || item.work_content || item.target || '该结构部件正在构建中。'}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                  {(selectedRoute.nfbtProcessSteps || []).length > 0 ? (
                    <div style={{ marginTop: '14px', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', overflow: 'hidden', background: 'rgba(0,0,0,0.25)' }}>
                      <div style={{ padding: '10px 12px', fontSize: '12px', color: '#fbbf24', fontWeight: 'bold', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
                        NFBT 计算过程
                      </div>
                      <div style={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: '760px' }}>
                          <thead>
                            <tr style={{ background: 'rgba(255,255,255,0.04)' }}>
                              <th style={{ textAlign: 'left', padding: '8px 10px', fontSize: '11px', color: '#fcd34d' }}>步骤 (Step)</th>
                              <th style={{ textAlign: 'left', padding: '8px 10px', fontSize: '11px', color: '#fcd34d' }}>输入 (Input)</th>
                              <th style={{ textAlign: 'left', padding: '8px 10px', fontSize: '11px', color: '#fcd34d' }}>输出 (Output)</th>
                              <th style={{ textAlign: 'left', padding: '8px 10px', fontSize: '11px', color: '#fcd34d' }}>复杂度</th>
                              <th style={{ textAlign: 'left', padding: '8px 10px', fontSize: '11px', color: '#fcd34d' }}>核心操作</th>
                            </tr>
                          </thead>
                          <tbody>
                            {(selectedRoute.nfbtProcessSteps || []).map((row, idx) => (
                              <tr key={idx} style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                                <td style={{ padding: '8px 10px', fontSize: '12px', color: '#fff' }}>{row.step}</td>
                                <td style={{ padding: '8px 10px', fontSize: '12px', color: '#cbd5e1', fontFamily: 'monospace' }}>{row.input}</td>
                                <td style={{ padding: '8px 10px', fontSize: '12px', color: '#cbd5e1', fontFamily: 'monospace' }}>{row.output}</td>
                                <td style={{ padding: '8px 10px', fontSize: '12px', color: '#93c5fd', fontFamily: 'monospace' }}>{row.complexity}</td>
                                <td style={{ padding: '8px 10px', fontSize: '12px', color: '#a7f3d0' }}>{row.op}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <div style={{ padding: '10px 12px', borderTop: '1px solid rgba(255,255,255,0.08)', fontSize: '12px', color: '#fde68a', lineHeight: '1.65' }}>
                        {selectedRoute.nfbtOptimization}
                      </div>
                    </div>
                  ) : null}
                </div>

                <div style={{ padding: '28px', background: 'rgba(16, 185, 129, 0.06)', border: '1px solid rgba(16, 185, 129, 0.22)', borderRadius: '22px' }}>
                  <div style={{ fontSize: '12px', color: '#10b981', fontWeight: 'bold', letterSpacing: '2px', marginBottom: '12px' }}>
                    里程碑（原 AGI 终点）
                  </div>
                  <div style={{ fontSize: '18px', color: '#fff', fontWeight: 'bold', marginBottom: '12px' }}>{selectedRoute.milestoneTitle}</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '14px' }}>
                    {mergedMilestoneStages.map((stage) => (
                      <div key={stage.id || stage.name} style={{ border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', background: 'rgba(255,255,255,0.02)', padding: '12px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                          <div style={{ color: '#ecfdf5', fontWeight: 'bold', fontSize: '14px' }}>{stage.name}</div>
                          <div style={{ fontSize: '10px', color: stage.status === 'done' ? '#10b981' : stage.status === 'in_progress' ? '#f59e0b' : '#60a5fa' }}>
                            {String(stage.status || 'planned').toUpperCase()}
                          </div>
                        </div>

                        <div style={{ marginBottom: '8px' }}>
                          <div style={{ fontSize: '11px', color: '#6ee7b7', marginBottom: '4px', fontWeight: 'bold' }}>功能点</div>
                          {(stage.featurePoints || []).map((point, idx) => (
                            <div key={idx} style={{ display: 'flex', gap: '8px', color: '#d1fae5', fontSize: '12px', marginBottom: '4px' }}>
                              <span style={{ width: '5px', height: '5px', borderRadius: '50%', background: '#10b981', marginTop: '7px' }} />
                              {point}
                            </div>
                          ))}
                        </div>

                        <div>
                          <div style={{ fontSize: '11px', color: '#7dd3fc', marginBottom: '6px', fontWeight: 'bold' }}>测试记录</div>
                          <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '8px' }}>
                            {(stage.tests || []).map((test, idx) => (
                              <div key={idx} style={{ border: '1px solid rgba(255,255,255,0.08)', borderRadius: '10px', padding: '10px', background: 'rgba(0,0,0,0.25)' }}>
                                <div style={{ color: '#fff', fontSize: '13px', fontWeight: 'bold', marginBottom: '6px' }}>{test.name}</div>
                                <div style={{ fontSize: '12px', color: '#cbd5e1', lineHeight: '1.6' }}>
                                  <div><span style={{ color: '#fcd34d' }}>参数配置：</span>{test.params || '-'}</div>
                                  <div><span style={{ color: '#fcd34d' }}>数据集：</span>{test.dataset || '-'}</div>
                                  <div><span style={{ color: '#fcd34d' }}>测试结果：</span>{test.result || '-'}</div>
                                  <div><span style={{ color: '#fcd34d' }}>分析总结：</span>{test.summary || '-'}</div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {selectedRoute.milestonePlanEvaluation ? (
                    <div style={{ marginTop: '14px', borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '12px' }}>
                      <div style={{ fontSize: '11px', color: '#6ee7b7', fontWeight: 'bold', marginBottom: '6px' }}>
                        方案评估与修改建议
                      </div>
                      <div style={{ fontSize: '12px', color: '#d1fae5', marginBottom: '8px', lineHeight: '1.65' }}>
                        评估：{selectedRoute.milestonePlanEvaluation.assessment}
                      </div>
                      <div>
                        {(selectedRoute.milestonePlanEvaluation.suggestions || []).map((item, idx) => (
                          <div key={idx} style={{ fontSize: '12px', color: '#bae6fd', lineHeight: '1.6', marginBottom: '4px' }}>
                            {idx + 1}. {item}
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </div>

                {selectedRoute?.id === 'fiber_bundle' ? (
                  <div style={{ padding: '28px', background: 'rgba(59, 130, 246, 0.06)', border: '1px solid rgba(59, 130, 246, 0.22)', borderRadius: '22px' }}>
                    <div style={{ fontSize: '12px', color: '#60a5fa', fontWeight: 'bold', letterSpacing: '2px', marginBottom: '10px' }}>
                      多模态纤维训练结果
                    </div>
                    <div style={{ display: 'flex', gap: '8px', marginBottom: '14px', flexWrap: 'wrap' }}>
                      {[
                        { id: 'vision_alignment', label: '视觉纤维训练' },
                        { id: 'multimodal_connector', label: '视觉-语言联络训练' },
                      ].map((item) => (
                        <button
                          key={item.id}
                          onClick={() => setMultimodalView(item.id)}
                          style={{
                            border: '1px solid rgba(255,255,255,0.2)',
                            borderRadius: '999px',
                            cursor: 'pointer',
                            padding: '6px 10px',
                            fontSize: '11px',
                            color: multimodalView === item.id ? '#dbeafe' : '#93c5fd',
                            background: multimodalView === item.id ? 'rgba(59,130,246,0.25)' : 'rgba(59,130,246,0.08)',
                          }}
                        >
                          {item.label}
                        </button>
                      ))}
                    </div>

                    {multimodalError ? (
                      <div style={{ fontSize: '12px', color: '#fca5a5', lineHeight: '1.6' }}>
                        加载失败：{multimodalError}
                      </div>
                    ) : null}

                    {!multimodalError && !selectedMultimodalReport ? (
                      <div style={{ fontSize: '12px', color: '#93c5fd' }}>
                        暂无结果。先运行对应训练脚本后可在这里切换查看。
                      </div>
                    ) : null}

                    {!multimodalError && selectedMultimodalReport ? (
                      <div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(120px, 1fr))', gap: '10px', marginBottom: '12px' }}>
                          {multimodalMetricRows.map((item) => (
                            <div key={item.label} style={{ border: '1px solid rgba(255,255,255,0.1)', borderRadius: '10px', background: 'rgba(0,0,0,0.2)', padding: '10px' }}>
                              <div style={{ fontSize: '10px', color: '#93c5fd', marginBottom: '4px' }}>{item.label}</div>
                              <div style={{ fontSize: '14px', color: '#e0f2fe', fontWeight: 'bold', fontFamily: 'monospace' }}>{item.value}</div>
                            </div>
                          ))}
                        </div>
                        <div style={{ fontSize: '12px', color: '#cbd5e1', lineHeight: '1.65' }}>
                          <div><span style={{ color: '#93c5fd' }}>分析类型：</span>{selectedMultimodalData?.analysis_type || '-'}</div>
                          <div><span style={{ color: '#93c5fd' }}>数据集：</span>{selectedMultimodalReport?.meta?.dataset || selectedMultimodalReport?.config?.dataset || '-'}</div>
                          <div><span style={{ color: '#93c5fd' }}>最新运行：</span>{selectedMultimodalLatest?.run_id || '-'}</div>
                          <div><span style={{ color: '#93c5fd' }}>时间：</span>{selectedMultimodalLatest?.timestamp || '-'}</div>
                        </div>
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </div>
          )}

          {/* TAB: AGI System Status */}
          {activeTab === 'system' && (
            <div style={{ animation: 'roadmapFade 0.5s ease-out' }}>
              <BrainModel />
              <div style={{ textAlign: 'center', marginBottom: '60px' }}>
                <h2 style={{ fontSize: '32px', fontWeight: '900', color: consciousField?.glow_color === 'amber' ? '#ffaa00' : '#10b981', margin: '20px 0 8px 0', transition: 'color 1s' }}>
                  {consciousField ? '实时意识场 (Active Consciousness)' : '系统状态 (System Status)'}
                </h2>
                <p style={{ color: '#666', fontSize: '14px' }}>
                  {consciousField
                    ? `当前内稳态平衡: ${((consciousField.stability || 0) * 100).toFixed(1)}% | GWS 竞争强度: ${(consciousField.gws_intensity || 0).toFixed(2)}`
                    : '基于 Project Genesis 协议的核心能力对齐报告'}
                </p>
                <div style={{ marginTop: '14px', display: 'flex', justifyContent: 'center', gap: '8px', flexWrap: 'wrap' }}>
                  {(systemRouteOptions.length > 0 ? systemRouteOptions : routeList).map((route) => (
                    <button
                      key={route.id}
                      onClick={() => setSelectedRouteId(route.id)}
                      style={{
                        border: '1px solid rgba(255,255,255,0.12)',
                        background: selectedRouteId === route.id ? 'rgba(0, 210, 255, 0.2)' : 'rgba(255,255,255,0.03)',
                        color: selectedRouteId === route.id ? '#67e8f9' : '#94a3b8',
                        borderRadius: '999px',
                        fontSize: '11px',
                        padding: '6px 10px',
                        cursor: 'pointer',
                      }}
                    >
                      {route.title}
                    </button>
                  ))}
                </div>
              </div>

              {/* Real-time Conscious Metrics Bar */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px', marginBottom: '40px', animation: 'fadeIn 1s' }}>
                {(activeSystemProfile?.metricCards || []).map((m, i) => (
                  <div key={i} style={{ padding: '18px', background: 'rgba(255,255,255,0.02)', borderRadius: '20px', border: `1px solid ${m.color}30`, textAlign: 'center' }}>
                    <div style={{ fontSize: '10px', color: '#666', marginBottom: '6px', fontWeight: 'bold' }}>{m.label}</div>
                    <div style={{ fontSize: '24px', fontWeight: 'bold', color: m.color, marginBottom: '6px' }}>{m.value}</div>
                    <div style={{ fontSize: '10px', color: '#9ca3af' }}>{m.brain_ability}</div>
                  </div>
                ))}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '40px' }}>
                <div style={{ background: 'rgba(0, 255, 136, 0.03)', border: '1px solid rgba(0, 255, 136, 0.15)', borderRadius: '32px', padding: '32px' }}>
                  <div style={{ fontSize: '12px', color: '#00ff88', textTransform: 'uppercase', letterSpacing: '2px', marginBottom: '24px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <CheckCircle size={16} /> 已具备能。(Equipped)
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                    {statusData.capabilities.map((c, i) => (
                      <div key={i} style={{ padding: '16px', background: 'rgba(255,255,255,0.02)', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
                        <div style={{ fontWeight: 'bold', fontSize: '14px', marginBottom: '8px' }}>{c.name}</div>
                        <div style={{ display: 'grid', gridTemplateColumns: '0.9fr 1.1fr', gap: '10px' }}>
                          <div style={{ padding: '8px 10px', borderRadius: '10px', background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.2)' }}>
                            <div style={{ fontSize: '10px', color: '#86efac', marginBottom: '4px' }}>人脑能力</div>
                            <div style={{ fontSize: '11px', color: '#dcfce7', lineHeight: '1.55' }}>{c.brain_ability || '-'}</div>
                          </div>
                          <div style={{ padding: '8px 10px', borderRadius: '10px', background: 'rgba(56,189,248,0.08)', border: '1px solid rgba(56,189,248,0.2)' }}>
                            <div style={{ fontSize: '10px', color: '#7dd3fc', marginBottom: '4px' }}>
                              当前实现（{selectedRoute?.title || selectedRouteId}）
                            </div>
                            <div style={{ fontSize: '11px', color: '#e0f2fe', lineHeight: '1.55' }}>{getRouteImpl(c)}</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div style={{ background: 'rgba(255, 68, 68, 0.03)', border: '1px solid rgba(255, 68, 68, 0.15)', borderRadius: '32px', padding: '32px' }}>
                  <div style={{ fontSize: '12px', color: '#ff4444', textTransform: 'uppercase', letterSpacing: '2px', marginBottom: '24px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <X size={16} /> 研发。缺失 (Missing)
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                    {statusData?.missing_capabilities?.map((c, i) => (
                      <div key={i} style={{ padding: '16px', background: 'rgba(255,255,255,0.02)', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
                        <div style={{ fontWeight: 'bold', fontSize: '14px', marginBottom: '8px', color: '#ff8888' }}>{c.name}</div>
                        <div style={{ display: 'grid', gridTemplateColumns: '0.9fr 1.1fr', gap: '10px' }}>
                          <div style={{ padding: '8px 10px', borderRadius: '10px', background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.22)' }}>
                            <div style={{ fontSize: '10px', color: '#fca5a5', marginBottom: '4px' }}>人脑能力</div>
                            <div style={{ fontSize: '11px', color: '#fee2e2', lineHeight: '1.55' }}>{c.brain_ability || '-'}</div>
                          </div>
                          <div style={{ padding: '8px 10px', borderRadius: '10px', background: 'rgba(251,191,36,0.08)', border: '1px solid rgba(251,191,36,0.22)' }}>
                            <div style={{ fontSize: '10px', color: '#fcd34d', marginBottom: '4px' }}>
                              当前实现（{selectedRoute?.title || selectedRouteId}）
                            </div>
                            <div style={{ fontSize: '11px', color: '#fef3c7', lineHeight: '1.55' }}>{getRouteImpl(c)}</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* [NEW] Parameters and Tests Section */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '40px', marginTop: '40px' }}>
                {/* Parameters */}
                <div style={{ background: 'rgba(0, 210, 255, 0.03)', border: '1px solid rgba(0, 210, 255, 0.15)', borderRadius: '32px', padding: '32px' }}>
                  <div style={{ fontSize: '12px', color: '#00d2ff', textTransform: 'uppercase', letterSpacing: '2px', marginBottom: '24px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Activity size={16} /> 核心参数 (Route Parameters)
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
                    {(activeSystemProfile?.parameterCards || []).map((p, i) => (
                      <div
                        key={i}
                        onClick={() => setExpandedParam(expandedParam === i ? null : i)}
                        style={{
                          padding: '16px', background: 'rgba(0,0,0,0.3)', borderRadius: '16px',
                          border: `1px solid ${expandedParam === i ? 'rgba(0, 210, 255, 0.5)' : 'rgba(0, 210, 255, 0.1)'}`,
                          cursor: 'pointer', transition: 'all 0.3s',
                          gridColumn: expandedParam === i ? 'span 2' : 'span 1' // Expand to full width if active
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                          <div style={{ fontSize: '10px', color: '#555' }}>{p.name}</div>
                          {expandedParam === i ? <div style={{ fontSize: '10px', color: '#00d2ff' }}>SHMC CORE ▲</div> : null}
                        </div>
                        <div style={{ fontSize: '11px', color: '#7dd3fc', marginBottom: '4px' }}>人脑能力：{p.brain_ability || '-'}</div>
                        <div style={{ fontSize: '18px', fontWeight: '900', color: '#fff', fontFamily: 'monospace' }}>{p.route_param}</div>
                        <div style={{ fontSize: '10px', color: '#00d2ff88', marginTop: '4px' }}>{p.detail}</div>

                        {/* Expanded Content for Parameters */}
                        {expandedParam === i && (
                          <div style={{ marginTop: '16px', borderTop: '1px solid rgba(0, 210, 255, 0.1)', paddingTop: '16px', animation: 'fadeIn 0.3s ease' }}>
                            <div style={{ marginBottom: '12px' }}>
                              <div style={{ fontSize: '10px', color: '#00d2ff', fontWeight: 'bold', marginBottom: '4px' }}>参数定义 (DEFINITION)</div>
                              <div style={{ fontSize: '12px', color: '#bbb', lineHeight: '1.6' }}>{p.desc}</div>
                            </div>
                            <div style={{ marginBottom: '12px' }}>
                              <div style={{ fontSize: '10px', color: '#a855f7', fontWeight: 'bold', marginBottom: '4px' }}>数值价。(VALUE)</div>
                              <div style={{ fontSize: '12px', color: '#bbb', lineHeight: '1.6' }}>{p.value_meaning}</div>
                            </div>
                            <div>
                              <div style={{ fontSize: '10px', color: '#f59e0b', fontWeight: 'bold', marginBottom: '4px' }}>核心重要。(WHY IMPORTANT)</div>
                              <div style={{ fontSize: '12px', color: '#bbb', lineHeight: '1.6' }}>{p.why_important}</div>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                <div style={{ background: 'rgba(16, 185, 129, 0.03)', border: '1px solid rgba(16, 185, 129, 0.15)', borderRadius: '32px', padding: '32px' }}>
                  <div style={{ fontSize: '12px', color: '#10b981', textTransform: 'uppercase', letterSpacing: '2px', marginBottom: '12px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Search size={16} /> 测试记录迁移
                  </div>
                  <div style={{ fontSize: '13px', color: '#d1fae5', lineHeight: '1.7' }}>
                    系统状态中的路线测试记录已整合到“研发进展 → 里程碑 → 路线测试记录”阶段。
                  </div>
                  <div style={{ marginTop: '10px', fontSize: '12px', color: '#86efac' }}>
                    当前路线测试数：{(activeSystemProfile?.validationRecords || []).length}
                  </div>
                </div>
              </div>

            </div>
          )}

        </div>
      </div>
    </div>
  );
};


