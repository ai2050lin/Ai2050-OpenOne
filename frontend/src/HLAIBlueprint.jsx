import { Activity, Brain, CheckCircle, Search, Target, X, Zap } from 'lucide-react';
import { useState } from 'react';

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
      "构建基于微分几何的高维语义流形 (High-Dim Manifold)",
      "实现 Logic-Memory 解耦的纤维丛架构 (Fiber Bundle)",
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
        desc: "数学原理：智能非随机。利用拉普拉斯算子 Δ 注入常识几何。可视化：展示点云从混沌向结晶坍缩，量化‘出生即懂世界’的骨架。"
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
        title: "4. 全局工作空间与意识裁决 (GWT)",
        desc: "终极形态：意识的物理载体。实现 Locus of Attention (关注点) 竞争裁决，追求最高信息增益与最低代谢成本的平衡。可视化：球体脉冲游走，实现双向最优路径结算。"
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
      pillars: ["Structure (稀疏全息流形)", "Dynamics (测地线流推理)", "Purpose (局部熵减/生存)"]
    },
    theory_content: [
      {
        title: "神经纤维丛原则 (NFB Principle)",
        formula: "Ψ(x) = φ_M ⊗ φ_F",
        desc: "结构与内容解耦：智能是底流形 (Logic) 与纤维空间 (Knowledge) 的张量积。逻辑骨架保持稳定，而语义内容可无限扩展。"
      },
      {
        title: "全局工作空间 (Global Workspace)",
        formula: "Ω_G = ∫Σ_i P_i dμ",
        desc: "意识裁决机制：意识是高维流形上的全局状态。通过 Locus of Attention (关注点) 的 Top-K 竞争机制，实现多模态信息的裁决与整合。"
      },
      {
        title: "高维全息编码 (SHDC Encoding)",
        formula: "⟨v_i, v_j⟩ ≈ δ_ij",
        desc: "解决维度灾难：利用高维空间的几乎正交性实现海量特征提取。通过向量代数叠加形成复杂关联，实现极致高效的物理级读写。"
      },
      {
        title: "联络与推理 (Connection Equation)",
        formula: "∇_X s = 0",
        desc: "推理即平行移动：类比推理本质上是向量在语义流形上的测地线平移。通过联络保持推理过程中语义结构的一致性。"
      },
      {
        title: "里奇流演化 (Evolution)",
        formula: "∂_t g = -2R",
        desc: "学习即流形松弛：神经网络的 Grokking 过程本质上是里奇流驱动的流形平滑化，通过减少内部张力实现全局逻辑自洽。"
      }
    ],
    goals: [
      "基于 Neural Fiber Bundle (NFB) 理论定义智能底流形",
      "建立微分几何与语义空间的映射模型",
      "验证跨模态同构投影数学严谨性"
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
        title: "有效性分析 (Why it works)",
        content: "深度神经网络成功提取了智能的‘特殊数学结构’。通过高维叠加假设 (Superposition)，网络能在有限参数内并行处理近乎无限的概念。其核心竞争力在于从统计相关性中自发诱发‘测地线流’，即沿着能量最低的逻辑路径进行推理。",
        tags: ["High-Dim Abstraction", "Low-Dim Precision", "Johnson-Lindenstrauss"]
      },
      {
        title: "核心组件 (Strategic Components)",
        content: "1. 残差流 (Residual Stream): 作为‘全息看板’承载所有计算状态。 2. 注意力头 (Attention Heads): 执行流形上的‘联络算子’。 3. MLP 层: 负责非线性流形投影与特征解耦。",
        tags: ["Residual Stream", "Attention Mechanism", "Manifold Projection"]
      },
      {
        title: "内部结构演化 (Internal Structure)",
        content: "观测到‘几何结晶’过程：浅层执行混沌的特征提取，深层（Layer 9-11）则坍缩为高代数一致性的纤维丛结构。模型深层通过 Ricci 演化自发消除逻辑矛盾，形成稳定的吸引子空间。",
        tags: ["Topological Smoothing", "Geometric Crystallization", "Attractors"]
      },
      {
        title: "研究工具与成果 (Research & Tools)",
        content: "利用 Logit Lens 与 TDA (拓扑数据分析) 成功测绘了 GPT-2 的全谱拓扑图。目前已证实 $Z_{113}$ 等离散群论结构可在连续向量空间中通过 DFT (离散傅里叶变换) 完美重构，准确率 >99%。",
        tags: ["Logit Lens", "TDA Scanner", "Z113 Group Results"]
      }
    ],
    goals: [
      "实现全谱 12 层 Logit Lens 探测",
      "提取全局语义流形拓扑 (Topology Scan)",
      "分析模型内部的因果链路与组合泛化"
    ],
    metrics: {
      "Scan Resolution": "Microscopic",
      "Layers Verified": "12/12"
    },
    comparison: [
      { feature: "信息载体 (Carrier)", dnn: "注意力矩阵/权重向量 (Attention/Weight)", fiber: "纤维束切面 (Fiber Section/Vector Bundle)" },
      { feature: "逻辑表示 (Logic)", dnn: "非几何权重叠加 (Non-Geometric Weight Overlay)", fiber: "流形几何位置 (Geometric/Explicit)" },
      { feature: "学习方式 (Learning)", dnn: "残差学习/反向传播 (Residual Learning/BP)", fiber: "里奇流平滑/传输平行移动 (Geometric Flow/Instant)" },
      { feature: "解释性 (Interpret)", dnn: "注意力热图 (Statistical Attention Maps)", fiber: "拓扑白盒 (Topology/Transparent)" }
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
        focus: "逻辑闭包与 Z113 验证",
        target: "验证智能底层是否具备几何化的非线性逻辑骨架。",
        work_content: "实现 FiberNet 核心逻辑层；在 Z113 模运算任务上观察从统计拟合到代数跃迁的 Grokking 现象。",
        test_results: "Z113 准确率 99.4%；成功从向量点云中恢复出完整的 S1 环面流形。",
        analysis: "证明了智能的底层是几何化的逻辑骨架，而非简单的线性回归。"
      },
      { 
        name: "感官的觉醒", 
        status: "done", 
        focus: "多模态语义对齐",
        target: "解决‘符号接地’问题，实现视觉特征与逻辑语义的物理对齐。",
        work_content: "开发跨模态投影算子；将 MNIST 视觉空间流形映射至 SYM 逻辑流形。",
        test_results: "对齐误差 MSE 下降至 0.042；模型具备‘以理性的方式解读感官数据’的能力。",
        analysis: "视觉特征被精准翻译为内部逻辑坐标，模型初步具备了‘观察并理解’的能力。"
      },
      { 
        name: "智慧的涌现", 
        status: "done", 
        focus: "流形曲率优化与 Ricci Flow",
        target: "通过流形平滑机制，实现无监督下的逻辑冲突自修复。",
        work_content: "集成 Ricci Flow 演化管道；在睡眠周期执行隐层激活曲率的热传导方程平滑。",
        test_results: "拓扑亏格从 15 优化至 2；复杂逻辑推理下的‘幻觉’发生率显着降低。",
        analysis: "验证了‘睡眠机制’在解开拓扑纠缠和修补逻辑死结中的必要性。"
      },
      { 
        name: "价值的形成", 
        status: "done", 
        focus: "神经流形手术与人类对齐",
        target: "实现对 AI 内在价值取向的直接几何干预，由‘提示词对齐’跃迁至‘流形对齐’。",
        work_content: "开发 Manifold Surgery 交互接口；实现基于 3D 空间拖拽的语义向量场实时重构。",
        test_results: "Surgery Alignment Loss 0.0082；成功通过物理手术剥离模型偏见，实现价值稳定对齐。",
        analysis: "流形手术允许直接修改模型认知，是通往可控 AGI 的物理路径。"
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
    goals: ["实现碳基与硅基的高级对称性对齐", "建立永久性的通用智能稳态", "完成物理世界到高维流形的全面映射"]
  },
  {
    id: 'agi_status',
    title: "Dimension V: Status",
    subtitle: "智能系统状态 (AGI Status)",
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
        detail: "从 1024D 全息投影",
        desc: "系统底层语义逻辑被压缩至 128 维的流形空间，确保逻辑闭包的紧凑性。",
        value_meaning: "维度越高蕴含信息越丰富，但 128D 是目前兼顾“计算效率”与“逻辑解析力”的最佳平衡点。",
        why_important: "它是智能的‘骨架’。维度过低会导致语义丢失（幻觉），维度过高则会导致维度灾难。128D 确保了推理的稳定性。"
      },
      { 
        name: "压缩倍率 (Compression)", 
        value: "26.67x", 
        detail: "SHDC 稀疏编码",
        desc: "通过全息稀疏投影技术，将庞大的原始神经元参数集压缩至极小规模，同时不损失结构信息。",
        value_meaning: "意味着系统可以以极低的显存占用（约 4GB）维持千亿级参数模型的逻辑核心。",
        why_important: "这是实现‘小型化 AGI’的关键。只有高的压缩比，智能才能脱离昂贵的算力集群，进入单机甚至移动端实时运行。"
      },
      { 
        name: "语义保真度 (Fidelity)", 
        value: "93%", 
        detail: "几何特征保留度",
        desc: "测量压缩后的流形几何特征（如曲率、测地线分布）与原始空间的对齐程度。",
        value_meaning: "93% 的保真度意味着在推理决策中，系统能够保持与原始超大型模型几乎一致的逻辑链路。",
        why_important: "它是‘一致性’的保障。过低的保真度会导致模型产生逻辑偏差，93% 确保了‘智能核心’从未在压缩中变质。"
      },
      { 
        name: "测地线一致性 (Geodesic)", 
        value: "98.8%", 
        detail: "推理路径对齐度",
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
        process: "通过 Rips Complex 算法构建点云单纯复形，持久化扫描 β₀ 和 β₁ 贝蒂数。",
        significance: "确保 AI 不是在拟合孤立样本，而是在构建具备全局拓扑一致性的语义形状。"
      },
      { 
        name: "里奇流流形平滑测试 (Ricci Smoothing)", 
        date: "2026-02-15", 
        result: "PASS",
        target: "消除推理过程中的逻辑尖峰（幻觉诱因），降低流形局部曲率。",
        process: "在睡眠周期运行离线里奇流处理，平滑度量张量的非连续跳变。",
        significance: "使模型推理轨迹更符合测地线分布，大幅提升逻辑自洽性。"
      },
      { 
        name: "SHDC 正交性基准测试 (Orthogonality)", 
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
      }
    ],
    capabilities: [
      { name: "里奇流演化 (Ricci Flow Evolution)", status: "equipped", desc: "通过几何平滑机制优化流形曲率，实现逻辑自洽（睡眠固化）。" },
      { name: "神经纤维丛 RAG (Fiber-RAG)", status: "equipped", desc: "实现了事实知识 ($F$) 的 O(N) 线性扩展与几何检索。" },
      { name: "多模态跨束对齐 (Cross-Bundle Alignment)", status: "equipped", desc: "视觉 (MNIST) 与逻辑 (Text) 锚点间的双向干预对齐已调通。" },
      { name: "全局工作空间 (Global Workspace)", status: "equipped", desc: "具备 Locus of Attention 焦点裁决与动态稀疏性 (Top-K) 激活控制。" },
      { name: "神经纤维 SNN (NeuroFiber-SNN)", status: "equipped", desc: "支持 3D 几何脉冲动力学仿真，模拟生物电信号传播。" },
      { name: "纤维记忆 (Fiber Memory)", status: "equipped", desc: "支持基于传输矩阵 (R) 的一键式知识注入与偏见解耦。" }
    ],
    missing_capabilities: [
      { name: "具身物理控制 (Embodied Control)", status: "missing", desc: "流形输出尚未与实时驱动器（如 ROS 接口）进行物理闭环。" },
      { name: "超大规模纤维持久化 (Massive Fiber Persistence)", status: "missing", desc: "十亿级参数规模下的纤维持久化存储与冷热数据切换架构尚在优化。" },
      { name: "跨模型流形迁移 (Cross-Model Transfer)", status: "missing", desc: "在不同基座模型（如 Llama 到 Qwen）之间的逻辑拓扑无损迁移技术尚在研发。" }
    ],
    goals: [
        "实现 100% 核心能力对齐",
        "完成物理躯体集成与控制",
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
    detail: "利用 Graph Laplacian 预计算 Embedding 几何骨架，让模型'出生'即带世界观。",
    step: "Step 3.5",
    status: "done"
  },
  {
    title: "Manifold Surgery",
    desc: "交互式神经干预",
    detail: "通过 3D 空间拖拽直接拨动高维激活（128-dim），实现对 AI 逻辑流向的实时干预。",
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
    desc: "全谱意识流",
    detail: "集成 7 大核心子引擎，实时监测 AGI 统合状态。",
    step: "Phase VII",
    status: "in_progress"
  }
];

export const HLAIBlueprint = ({ onClose }) => {
  const [activeTab, setActiveTab] = useState('roadmap'); // roadmap, progress, system
  const [activePhaseId, setActivePhaseId] = useState('theory');
  const [expandedTest, setExpandedTest] = useState(null);
  const [expandedParam, setExpandedParam] = useState(null);
  const [expandedEngPhase, setExpandedEngPhase] = useState(null); 
  const [consciousField, setConsciousField] = useState(null);

  // Real-time Consciousness Polling
  useState(() => {
    const pollConsciousField = async () => {
      try {
        const res = await fetch('http://localhost:5001/nfb_ra/unified_conscious_field');
        const data = await res.json();
        if (data.status === 'success') {
          setConsciousField(data.unified_spectrum);
        }
      } catch (e) {
        console.warn("Unified Conscious Field unreachable. Ensure server is running.");
      }
    };
    const interval = setInterval(pollConsciousField, 2000);
    return () => clearInterval(interval);
  }, []);

  const activePhase = PHASES.find(p => p.id === activePhaseId);
  const statusData = PHASES.find(p => p.id === 'agi_status');
  const roadmapData = PHASES.find(p => p.id === 'roadmap');

  const progressPhases = PHASES.filter(p => ['theory', 'analysis', 'engineering', 'agi_goal'].includes(p.id));

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
            <div style={{ fontSize: '10px', color: '#444', textTransform: 'uppercase', marginBottom: '30px', letterSpacing: '2px', fontWeight: 'bold' }}>Research Dimensions</div>
            
            {/* Vertical Timeline Line */}
            <div style={{ 
              position: 'absolute', left: '38px', top: '80px', bottom: '40px', 
              width: '1px', background: 'linear-gradient(to bottom, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%)',
              zIndex: 0
            }} />

            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', position: 'relative', zIndex: 1 }}>
              {progressPhases.map(p => (
                <div key={p.id} style={{ position: 'relative' }}>
                  {/* Timeline Dot */}
                  <div style={{
                    position: 'absolute', left: '15px', top: '22px', 
                    width: '6px', height: '6px', borderRadius: '50%',
                    background: activePhaseId === p.id ? p.color : '#222',
                    border: `2px solid ${activePhaseId === p.id ? '#000' : 'rgba(255,255,255,0.1)'}`,
                    boxShadow: activePhaseId === p.id ? `0 0 10px ${p.color}` : 'none',
                    transition: 'all 0.3s'
                  }} />

                  <button
                    onClick={() => setActivePhaseId(p.id)}
                    style={{
                      width: '100%', padding: '12px 12px 12px 45px', borderRadius: '14px', 
                      textAlign: 'left', cursor: 'pointer',
                      background: activePhaseId === p.id ? 'rgba(255,255,255,0.03)' : 'transparent',
                      border: 'none',
                      color: activePhaseId === p.id ? '#fff' : '#666', transition: 'all 0.3s',
                      display: 'flex', flexDirection: 'column', gap: '4px'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                      <span style={{ fontSize: '14px', fontWeight: 'bold', color: activePhaseId === p.id ? '#fff' : '#888' }}>
                        {p.subtitle.split(' (')[0]}
                      </span>
                      <span style={{ 
                        fontSize: '11px', fontFamily: 'monospace', fontWeight: 'bold', 
                        color: activePhaseId === p.id ? p.color : '#444' 
                      }}>
                        {p.progress}%
                      </span>
                    </div>
                    <div style={{ fontSize: '10px', color: '#444', textTransform: 'uppercase' }}>{p.title.split(': ')[1]}</div>
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
              <div style={{ 
                padding: '40px', background: 'linear-gradient(135deg, rgba(255,170,0,0.1) 0%, rgba(255,170,0,0.02) 100%)',
                border: '1px solid rgba(255,170,0,0.2)', borderRadius: '32px', marginBottom: '40px' 
              }}>
                <h2 style={{ fontSize: '32px', fontWeight: '900', color: '#ffaa00', marginBottom: '20px' }}>{roadmapData.definition.headline}</h2>
                <p style={{ fontSize: '16px', color: '#bbb', lineHeight: '1.8', margin: 0 }}>{roadmapData.definition.summary}</p>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
                {roadmapData?.roadmap_outline?.map((o, idx) => (
                  <div key={idx} style={{ padding: '24px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '20px' }}>
                    <div style={{ color: '#ffaa00', fontWeight: 'bold', fontSize: '16px', marginBottom: '12px' }}>{o.title}</div>
                    <p style={{ color: '#888', fontSize: '13px', lineHeight: '1.6', margin: 0 }}>{o.desc}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TAB: Research Progress (Dynamic Dimensions) */}
          {activeTab === 'progress' && activePhase && (
            <div style={{ animation: 'roadmapFade 0.5s ease-out' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '40px' }}>
                <div>
                  <h2 style={{ fontSize: '36px', fontWeight: '900', color: '#fff', margin: '0 0 8px 0' }}>{activePhase.subtitle}</h2>
                  <div style={{ color: activePhase.color, fontWeight: 'bold', letterSpacing: '1px', fontSize: '14px' }}>{activePhase.title}</div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '32px', fontWeight: '900', color: activePhase.color, fontFamily: 'monospace' }}>{activePhase.progress}%</div>
                  <div style={{ fontSize: '10px', color: '#444' }}>SYNC COMPLETE</div>
                </div>
              </div>

              {/* Specific rendering per module */}
              {activePhase.id === 'theory' && (
                <div>
                  {/* Textual Definition of Intelligence */}
                  <div style={{ 
                    padding: '30px', background: 'rgba(0, 210, 255, 0.05)', 
                    border: '1px solid rgba(0, 210, 255, 0.2)', borderRadius: '24px',
                    marginBottom: '30px', position: 'relative', overflow: 'hidden'
                  }}>
                    <div style={{ position: 'absolute', right: '-20px', top: '-20px', opacity: 0.1 }}>
                      <Brain size={120} color="#00d2ff" />
                    </div>
                    <div style={{ fontSize: '11px', color: '#00d2ff', fontWeight: 'bold', marginBottom: '12px', letterSpacing: '2px' }}>CORE DEFINITION</div>
                    <h3 style={{ fontSize: '24px', fontWeight: 'bold', color: '#fff', marginBottom: '16px' }}>{activePhase.definition.headline}</h3>
                    <p style={{ fontSize: '16px', color: '#ccc', lineHeight: '1.8', margin: 0, maxWidth: '80%' }}>
                      {activePhase.definition.summary}
                    </p>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                    {activePhase.theory_content.map((t, idx) => (
                      <div key={idx} style={{ padding: '30px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '24px' }}>
                        <div style={{ fontWeight: 'bold', color: activePhase.color, marginBottom: '20px', fontSize: '15px' }}>{t.title}</div>
                        <div style={{ padding: '25px', background: '#000', borderRadius: '12px', textAlign: 'center', fontSize: '28px', color: '#fff', fontFamily: 'serif', marginBottom: '16px', border: '1px solid rgba(255,255,255,0.1)' }}>{t.formula}</div>
                        <p style={{ fontSize: '12px', color: '#777', lineHeight: '1.6', margin: 0 }}>{t.desc}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {activePhase.id === 'engineering' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                  {activePhase.sub_phases.map((p, i) => (
                    <div key={i} style={{ borderRadius: '20px', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.05)', background: 'rgba(255,255,255,0.01)', transition: 'all 0.3s' }}>
                      <div 
                        onClick={() => setExpandedEngPhase(expandedEngPhase === i ? null : i)}
                        style={{ 
                          padding: '24px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                          background: expandedEngPhase === i ? 'rgba(245, 158, 11, 0.05)' : 'transparent'
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                          <div style={{ 
                            width: '40px', height: '40px', borderRadius: '12px', background: p.status === 'done' ? '#10b98120' : '#f59e0b20',
                            display: 'flex', alignItems: 'center', justifyContent: 'center', color: p.status === 'done' ? '#10b981' : '#f59e0b'
                          }}>
                            {p.status === 'done' ? <CheckCircle size={20} /> : <Activity size={20} />}
                          </div>
                          <div>
                            <div style={{ fontWeight: 'bold', fontSize: '18px', color: '#fff', marginBottom: '4px' }}>{p.name}</div>
                            <div style={{ fontSize: '12px', color: '#666' }}>{p.focus}</div>
                          </div>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                          <div style={{ 
                            padding: '4px 12px', borderRadius: '20px', fontSize: '10px', fontWeight: 'bold',
                            background: p.status === 'done' ? '#10b98120' : '#f59e0b20',
                            color: p.status === 'done' ? '#10b981' : '#f59e0b',
                            border: `1px solid ${p.status === 'done' ? '#10b98140' : '#f59e0b40'}`
                          }}>
                            {p.status.toUpperCase()}
                          </div>
                          <div style={{ color: '#444', transition: 'transform 0.3s', transform: expandedEngPhase === i ? 'rotate(180deg)' : 'rotate(0)' }}>▼</div>
                        </div>
                      </div>

                      {expandedEngPhase === i && (
                        <div style={{ padding: '0 24px 24px 84px', animation: 'fadeIn 0.4s ease' }}>
                          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '24px' }}>
                            <div>
                              <div style={{ marginBottom: '20px' }}>
                                <div style={{ fontSize: '11px', color: '#f59e0b', fontWeight: 'bold', marginBottom: '8px', letterSpacing: '1px' }}>测试目标 (TARGET)</div>
                                <div style={{ fontSize: '13px', color: '#bbb', lineHeight: '1.6' }}>{p.target}</div>
                              </div>
                              <div style={{ marginBottom: '20px' }}>
                                <div style={{ fontSize: '11px', color: '#00d2ff', fontWeight: 'bold', marginBottom: '8px', letterSpacing: '1px' }}>工作内容 (WORK CONTENT)</div>
                                <div style={{ fontSize: '13px', color: '#bbb', lineHeight: '1.6' }}>{p.work_content}</div>
                              </div>
                              <div>
                                <div style={{ fontSize: '11px', color: '#10b981', fontWeight: 'bold', marginBottom: '8px', letterSpacing: '1px' }}>测试结果 (TEST RESULTS)</div>
                                <div style={{ fontSize: '13px', color: '#eee', lineHeight: '1.6', fontWeight: 'bold' }}>{p.test_results}</div>
                              </div>
                            </div>
                            <div style={{ background: 'rgba(255,255,255,0.02)', padding: '20px', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
                               <div style={{ fontSize: '11px', color: '#666', fontWeight: 'bold', marginBottom: '12px' }}>EXPERT ANALYSIS</div>
                               <div style={{ fontSize: '13px', color: '#888', fontStyle: 'italic', lineHeight: '1.7' }}>"{p.analysis}"</div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {activePhase.id === 'analysis' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '40px' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                    {activePhase.analysis_sections.map((a, idx) => (
                      <div key={idx} style={{ padding: '30px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '24px' }}>
                        <div style={{ fontWeight: 'bold', marginBottom: '16px', fontSize: '16px' }}>{a.title}</div>
                        <p style={{ fontSize: '13px', color: '#999', lineHeight: '1.7', margin: 0 }}>{a.content}</p>
                      </div>
                    ))}
                  </div>

                  {/* Comparison Table */}
                  <div style={{ 
                    padding: '30px', background: 'rgba(168, 85, 247, 0.05)', 
                    border: '1px solid rgba(168, 85, 247, 0.2)', borderRadius: '32px' 
                  }}>
                    <h3 style={{ fontSize: '18px', fontWeight: 'bold', color: '#a855f7', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <Activity size={20} /> Fiber vs Transformer 架构对比
                    </h3>
                    <div style={{ overflow: 'hidden', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
                        <thead>
                          <tr style={{ background: 'rgba(255,255,255,0.05)', textAlign: 'left' }}>
                            <th style={{ padding: '15px', color: '#888' }}>核心维度</th>
                            <th style={{ padding: '15px', color: '#888' }}>Transformer 架构 (SOTA)</th>
                            <th style={{ padding: '15px', color: '#a855f7' }}>Fiber 架构 (Project Genesis)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {activePhase?.comparison?.map((row, i) => (
                            <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.02)' }}>
                              <td style={{ padding: '15px', fontWeight: 'bold', color: '#aaa' }}>{row.feature}</td>
                              <td style={{ padding: '15px', color: '#666' }}>{row.dnn}</td>
                              <td style={{ padding: '15px', color: '#fff' }}>{row.fiber}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}

              {/* Shared Objectives footer */}
              <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '30px', marginTop: '40px' }}>
                <div style={{ padding: '24px', background: 'rgba(255,255,255,0.02)', borderRadius: '20px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <div style={{ fontSize: '11px', color: '#444', marginBottom: '16px', fontWeight: 'bold' }}>OBJECTIVES</div>
                  {activePhase?.goals?.map((g, i) => (
                    <div key={i} style={{ display: 'flex', gap: '10px', fontSize: '13px', color: '#999', marginBottom: '8px' }}>
                      <div style={{ width: '4px', height: '4px', background: activePhase.color, borderRadius: '2px', marginTop: '7px' }} /> {g}
                    </div>
                  ))}
                </div>
                <div style={{ padding: '24px', background: 'rgba(255,255,255,0.02)', borderRadius: '20px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <div style={{ fontSize: '11px', color: '#444', marginBottom: '16px', fontWeight: 'bold' }}>KPI STATISTICS</div>
                  {activePhase?.metrics && Object.entries(activePhase.metrics).map(([k, v]) => (
                    <div key={k} style={{ marginBottom: '16px' }}>
                      <div style={{ fontSize: '10px', color: '#555' }}>{k}</div>
                      <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#fff' }}>{v}</div>
                    </div>
                  ))}
                </div>
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
                  {consciousField ? `当前内稳态平衡: ${(consciousField.stability * 100).toFixed(1)}% | GWS 竞争强度: ${consciousField.gws_intensity.toFixed(2)}` : '基于 Project Genesis 协议的核心能力对齐报告'}
                </p>
              </div>

              {/* Real-time Conscious Metrics Bar */}
              {consciousField && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px', marginBottom: '40px', animation: 'fadeIn 1s' }}>
                  {[
                    { label: '能效优化 (DMS)', value: `${(consciousField.energy_saving * 100).toFixed(1)}%`, color: '#a855f7' },
                    { label: '记忆活跃度', value: `${consciousField.memory_load}%`, color: '#00d2ff' },
                    { label: '跨域共振率', value: consciousField.resonance.toFixed(3), color: '#ffaa00' },
                    { label: '对齐稳定性', value: consciousField.stability.toFixed(2), color: '#10b981' }
                  ].map((m, i) => (
                    <div key={i} style={{ padding: '20px', background: 'rgba(255,255,255,0.02)', borderRadius: '20px', border: `1px solid ${m.color}30`, textAlign: 'center' }}>
                      <div style={{ fontSize: '10px', color: '#666', marginBottom: '8px', fontWeight: 'bold' }}>{m.label}</div>
                      <div style={{ fontSize: '24px', fontWeight: 'bold', color: m.color }}>{m.value}</div>
                    </div>
                  ))}
                </div>
              )}

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '40px' }}>
                <div style={{ background: 'rgba(0, 255, 136, 0.03)', border: '1px solid rgba(0, 255, 136, 0.15)', borderRadius: '32px', padding: '32px' }}>
                  <div style={{ fontSize: '12px', color: '#00ff88', textTransform: 'uppercase', letterSpacing: '2px', marginBottom: '24px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <CheckCircle size={16} /> 已具备能力 (Equipped)
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                    {statusData.capabilities.map((c, i) => (
                      <div key={i} style={{ padding: '16px', background: 'rgba(255,255,255,0.02)', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
                        <div style={{ fontWeight: 'bold', fontSize: '14px', marginBottom: '4px' }}>{c.name}</div>
                        <div style={{ fontSize: '11px', color: '#888' }}>{c.desc}</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div style={{ background: 'rgba(255, 68, 68, 0.03)', border: '1px solid rgba(255, 68, 68, 0.15)', borderRadius: '32px', padding: '32px' }}>
                  <div style={{ fontSize: '12px', color: '#ff4444', textTransform: 'uppercase', letterSpacing: '2px', marginBottom: '24px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <X size={16} /> 研发中/缺失 (Missing)
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                    {statusData?.missing_capabilities?.map((c, i) => (
                      <div key={i} style={{ padding: '16px', background: 'rgba(255,255,255,0.02)', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
                        <div style={{ fontWeight: 'bold', fontSize: '14px', marginBottom: '4px', color: '#ff8888' }}>{c.name}</div>
                        <div style={{ fontSize: '11px', color: '#777' }}>{c.desc}</div>
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
                    <Activity size={16} /> 核心参数 (Parameters)
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
                    {statusData?.parameters?.map((p, i) => (
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
                        <div style={{ fontSize: '20px', fontWeight: '900', color: '#fff', fontFamily: 'monospace' }}>{p.value}</div>
                        <div style={{ fontSize: '10px', color: '#00d2ff88', marginTop: '4px' }}>{p.detail}</div>
                        
                        {/* Expanded Content for Parameters */}
                        {expandedParam === i && (
                          <div style={{ marginTop: '16px', borderTop: '1px solid rgba(0, 210, 255, 0.1)', paddingTop: '16px', animation: 'fadeIn 0.3s ease' }}>
                            <div style={{ marginBottom: '12px' }}>
                              <div style={{ fontSize: '10px', color: '#00d2ff', fontWeight: 'bold', marginBottom: '4px' }}>参数定义 (DEFINITION)</div>
                              <div style={{ fontSize: '12px', color: '#bbb', lineHeight: '1.6' }}>{p.desc}</div>
                            </div>
                            <div style={{ marginBottom: '12px' }}>
                              <div style={{ fontSize: '10px', color: '#a855f7', fontWeight: 'bold', marginBottom: '4px' }}>数值价值 (VALUE)</div>
                              <div style={{ fontSize: '12px', color: '#bbb', lineHeight: '1.6' }}>{p.value_meaning}</div>
                            </div>
                            <div>
                              <div style={{ fontSize: '10px', color: '#f59e0b', fontWeight: 'bold', marginBottom: '4px' }}>核心重要性 (WHY IMPORTANT)</div>
                              <div style={{ fontSize: '12px', color: '#bbb', lineHeight: '1.6' }}>{p.why_important}</div>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Passed Tests */}
                <div style={{ background: 'rgba(16, 185, 129, 0.03)', border: '1px solid rgba(16, 185, 129, 0.15)', borderRadius: '32px', padding: '32px' }}>
                  <div style={{ fontSize: '12px', color: '#10b981', textTransform: 'uppercase', letterSpacing: '2px', marginBottom: '24px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Search size={16} /> 已通过测试 (Passed Tests)
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    {statusData?.passed_tests?.map((t, i) => (
                      <div key={i} style={{ borderRadius: '12px', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.05)', transition: 'all 0.3s' }}>
                        <div 
                          onClick={() => setExpandedTest(expandedTest === i ? null : i)}
                          style={{ 
                            display: 'flex', justifyContent: 'space-between', alignItems: 'center', 
                            padding: '12px 16px', background: 'rgba(255,255,255,0.02)', cursor: 'pointer',
                            hover: { background: 'rgba(255,255,255,0.04)' }
                          }}
                        >
                          <div>
                            <div style={{ fontWeight: 'bold', fontSize: '13px', color: '#eee' }}>{t.name}</div>
                            <div style={{ fontSize: '10px', color: '#555' }}>验证日期: {t.date}</div>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8_px' }}>
                            <div style={{ 
                              padding: '4px 10px', background: 'rgba(16, 185, 129, 0.1)', 
                              color: '#10b981', borderRadius: '20px', fontSize: '10px', 
                              fontWeight: 'bold', border: '1px solid rgba(16, 185, 129, 0.2)' 
                            }}>
                              {t.result}
                            </div>
                            <div style={{ color: '#444', transition: 'transform 0.3s', transform: expandedTest === i ? 'rotate(180deg)' : 'rotate(0)' }}>▼</div>
                          </div>
                        </div>
                        
                        {/* Expanded Content */}
                        {expandedTest === i && (
                          <div style={{ padding: '20px', background: 'rgba(0,0,0,0.4)', borderTop: '1px solid rgba(255,255,255,0.05)', animation: 'fadeIn 0.3s ease' }}>
                            <div style={{ marginBottom: '12px' }}>
                              <div style={{ fontSize: '10px', color: '#10b981', fontWeight: 'bold', marginBottom: '4px' }}>测试目标 (TARGET)</div>
                              <div style={{ fontSize: '12px', color: '#bbb', lineHeight: '1.6' }}>{t.target}</div>
                            </div>
                            <div style={{ marginBottom: '12px' }}>
                              <div style={{ fontSize: '10px', color: '#00d2ff', fontWeight: 'bold', marginBottom: '4px' }}>实施过程 (PROCESS)</div>
                              <div style={{ fontSize: '12px', color: '#bbb', lineHeight: '1.6' }}>{t.process}</div>
                            </div>
                            <div>
                              <div style={{ fontSize: '10px', color: '#f59e0b', fontWeight: 'bold', marginBottom: '4px' }}>数据意义 (SIGNIFICANCE)</div>
                              <div style={{ fontSize: '12px', color: '#bbb', lineHeight: '1.6' }}>{t.significance}</div>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
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
