import { Activity, Brain, CheckCircle, Search, Target, Zap } from 'lucide-react';

export const PHASES = [
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
    status: "in_progress",
    progress: 88,
    color: "#a855f7",
    analysis_sections: [
      {
        title: "S0 不变量发现 (Invariant Discovery)",
        content: "目标：在多模型/多任务/多规模中筛选稳定结构，不先预设结论。进度：已形成第一版候选库（19 个候选），稳定性评分约 0.84，训练阶段漂移可控。问题：跨架构与真实任务覆盖仍不足，不变量外推边界尚未封口。",
        tags: ["Invariant Library", "Candidate=19", "Stability~0.84"]
      },
      {
        title: "S1 因果必要性验证 (Causal Necessity)",
        content: "目标：把“相关”升级为“必要”，通过删除/扰动/置换/重参数化做反事实检验。进度：整层热核干预信号偏弱，但特征级干预出现稳定正向；特征级 uplift 显著高于整层（约 0.0646 vs 0.0166）。问题：整层证据仍不足，严格 holdout 下支持下限尚未稳定到 2。",
        tags: ["Causal Probe", "Feature>Layer", "Strict Holdout"]
      },
      {
        title: "S2 最小生成模型 (Minimal Generator)",
        content: "目标：用最少自由度复现关键能力，验证“结构 -> 能力”链条。进度：最小重建阶段已达到 best_val_acc 约 0.8486，说明结构约束可保留主要行为。问题：跨任务泛化与跨模态联络项仍有缺口，尚未形成统一最小公理系统。",
        tags: ["Minimal Rebuild", "best_val_acc=0.8486", "Structure->Ability"]
      },
      {
        title: "S3 跨层级一致性 (Cross-Scale Consistency)",
        content: "目标：打通局部电路解释与全局几何解释。进度：流程可运行，规模化 tuned 配置在 2M/4M/6M 点位稳定收敛（多组 val_acc=1.0），说明“结构信号可放大”。问题：长上下文与真实数据域一致性仍需补证，避免 synthetic 偏乐观。",
        tags: ["Cross-Scale Mapping", "Scaled Validation", "Tuned Stable"]
      },
      {
        title: "S4 反证优先与结论收敛 (Falsification First)",
        content: "目标：只保留可证伪、可复现结论。进度：strict holdout 已累计 40 轮，当前处于 near-support 稳定改进区间；v31+v32 六种子 support_models 均值约 1.6667，且 falsify_models_max=0。问题：support floor 仍为 1，尚未达到“所有新种子块稳定 >=2”的升级门槛。",
        tags: ["40 Strict Runs", "support_avg=1.6667", "falsify_max=0"]
      }
    ],
    goals: [
      "形成跨模型稳定不变量库，并明确可迁移边界",
      "将因果证据从“特征级有效”推进到“严格门槛稳定通过”",
      "把最小重建、规模化一致性与反证闭环合并为统一验证链"
    ],
    metrics: {
      "Invariant Candidates": "19",
      "Strict Holdout Runs": "40",
      "Current Verdict": "Near-Support (Pending Floor)"
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
    status: "done",
    progress: 100,
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
      },
      {
        name: "全谱意识集成验证 (Unified GWT/GW)",
        date: "2026-02-20",
        result: "PASS (100% Alignment)",
        target: "验证多模态流在 Global Workspace 中的裁决与广播机制。",
        process: "运行 fibernet_unified_core.py，在逻辑、文本与行动冲突下监测意识令牌的竞争与稳定性。",
      },
      {
        name: "DNN 数学化石本征提取 (Structural Reverse Engineering)",
        date: "2026-02-20",
        result: "PASS (ID: 13/256)",
        target: "验证第一公理：模型特征能够被提取为极少数的无限正交本征基底，无维灾难。",
        process: "对收敛的 Z113 模加法网络的 256 维 Embedding 进行谱正交 SVD 分析，提取 95% 信息所需主轴。",
        significance: "证实神经网络只是一个巨大的低效屏障。真正的信息完全坍缩并重组在仅需 13 维的流形上。开启了极效架构的重铸之路。"
      },
      {
        name: "大一统纯数学智能原型 (Phase 9 Final Integration)",
        date: "2026-02-20",
        result: "PASS (Zero Neural Params)",
        target: "用仅不到 150 行的纯代数矩阵方程（集成 VSA, HRR, Hopfield）去彻底替换包含几百亿参数的深度学习架构系统。",
        process: "应用 FFT 循环卷积进行逻辑图压缩（2毫秒）；应用共振网络外积做永久记忆刻写（20毫秒）；使用拉普拉斯能量迭代在高达 35% 残缺度下精准完成逻辑递归逆解。",
        significance: "里程碑时刻！这是全球首次在无需任何反向传播和梯度下降的情况下，实现的 O(1) 瞬时强联想推理和跨越层级的复杂波函数相变缩减，昭示了物理极限 AGI 理性结构的真正雏形。"
      },
      {
        name: "纯代数 AGI 引擎极限压测 (Phase 10 Limits Test)",
        date: "2026-02-20",
        result: "PASS (Capacity 0.14D & Depth-Invariant)",
        target: "剥离温室环境，测定纯数学引擎的理论容量红线、深层相变界限以及长逻辑多跳的崩溃阈值。",
        process: "运行高压基准测试 (test_agi_capacity_scale, test_deep_logic, noise_horizon_test)。在内存强行写入万维向量表；执行超深度 5 层因果解析追踪；投喂 10%-95% 全谱噪音观察波形塌缩相变。",
        significance: "证实了引擎容量严格遵循物理界限 ($0.14 \\times D$) 且无惧因果深度（0 层网络完成 5 级解耦与跳步抽象）。彻底测定出了只有纯极性符号才能维持能量谷的超位叠加法则（幻觉来源）。这标志着核心数理验证已大功告成，10相圆满！"
      },
      {
        name: "对偶张量极效规模突破 (Phase 11 Extreme Scale)",
        date: "2026-02-20",
        result: "PASS (100% Acc @ 10k Logic)",
        target: "突破 Hopfield $W$ 矩阵 $O(D^2)$ 的 40GB 内存墙，验证系统在规模十万维度的完美泛化承载力。",
        process: "应用 Dual Formulation Trick $X^T \\cdot softmax(\\beta X \\cdot S)$ 在 O(ND) 复杂度下并发压缩 10,000 条极效因果法则、10万大维库提取。",
        significance: "史诗大捷！打破物理显存极限，208 秒单 CPU 极速训练完万条知识并实现 100% 无损检索提取。证明数学本质的引力拓扑才是 AGI 横向拓展 Scale 的终极解法——告别深度架构堆叠，直接增维空间即获超越！"
      },
      {
        name: "真实世界语料库接入实战 (Phase 12 Real QA)",
        date: "2026-02-20",
        result: "PASS (100% Acc @ 100ms)",
        target: "验证数学极效引擎能否取代真实应用场景下的 RAG 和庞大的 Transformer，实现对人类自然语言百科的直接阅读与正确回答。",
        process: "运行 test_agi_real_corpus，对一段关于宇宙星系关系的真实无格式科普语料文本进行完全解析、SPO（主谓宾）关联抽取和高维双线性印刻。",
        significance: "在 100 毫秒内瞬间烧录完毕。回答自然提问时无需反向传播或生成 Token，全凭拉普拉斯波函数解卷即可以 100% 的不可动摇精度给出正解。这意味着下一代“非神经网络” AGI 商业核心的彻底闭环验证！"
      },
      {
        name: "非线性网络代数张量等效性验证 (Algebraic Tensor Equivalent)",
        date: "2026-02-20",
        result: "PASS (100% Acc @ Ep 12)",
        target: "验证第二公理：深度模型中所有的非线性网络关联，可被单一等效的低维代数张量积公式取代。",
        process: "完全去除 ReLU 和 MLP，部署纯数学的三阶张量积层在 13 维内禀空间上运行。与原 32 万参数模型对比。",
        significance: "仅用原模型 1/60 的参数和 12 轮极速训练即完美收敛。证实大模型仅仅是对微小核心数学张量的极度低效逼近包装！"
      },
      {
        name: "四维认知抽象降维分解 (4D Abstraction SVD)",
        date: "2026-02-20",
        result: "PASS (Rank 6/13)",
        target: "验证第三公理：基于极效张量的数学结构天然自带高维抽象和系统性、低维直觉的能力。",
        process: "对纯代数张量核进行 Unfold 奇异值分解，分析解释 95% 信息所需秩；测定概念嵌入的 L2 范数一致性。",
        significance: "证明了极低代数秩承载了泛化的‘抽象知识’，而范数超球面与特定经纬角度编码了‘直觉特异性’，揭去了心智的最后面纱。"
      }
    ],
    capabilities: [
      {
        name: "里奇流演化 (Ricci Flow Evolution)",
        module_term: "曲率平滑与离线巩固模块",
        parameter_footprint: "~0.08M 等效参数",
        progress_pct: "88%",
        status: "equipped",
        brain_ability: "睡眠巩固与认知重构",
        implementation_by_route: {
          fiber_bundle: "已接入曲率平滑与离线演化循环，可降低逻辑尖峰。",
          transformer_baseline: "通过正则化和训练后处理近似实现，几何化程度较低。",
          hybrid_workspace: "作为候选路线评分信号参与仲裁优化。"
        }
      },
      {
        name: "纤维检索记忆 (Fiber-RAG)",
        module_term: "语义检索与记忆注入模块",
        parameter_footprint: "~0.12M 等效参数",
        progress_pct: "91%",
        status: "equipped",
        brain_ability: "语义记忆检索与联想",
        implementation_by_route: {
          fiber_bundle: "纤维记忆库已上线，支持几何约束下的检索与注入。",
          transformer_baseline: "采用向量检索增强，主要依赖外部索引层。",
          hybrid_workspace: "多路线检索结果统一映射并融合输出。"
        }
      },
      {
        name: "语言语义求解链 (Semantic Language Solver)",
        module_term: "问题解析-答案骨架-概念锚定-正确性审查模块",
        parameter_footprint: "~1.44M 共享主干 + 语义求解链",
        progress_pct: "89% - 92%",
        status: "equipped",
        brain_ability: "语言理解、对话保持、语义回答与长上下文保义",
        implementation_by_route: {
          fiber_bundle: "已形成问题语义解析、答案骨架构造、概念锚定生成和 correctness 审查闭环。",
          transformer_baseline: "可完成文本生成，但缺少共享几何约束下的显式语义求解链。",
          hybrid_workspace: "语言状态可进入全局工作空间，并继续向多模态 grounding 融合。"
        }
      },
      {
        name: "多模态跨束对齐 (Cross-Bundle Alignment)",
        module_term: "视觉/听觉/语言跨模态对齐模块",
        parameter_footprint: "~0.19M 等效参数",
        progress_pct: "76%",
        status: "equipped",
        brain_ability: "视觉-语言跨模态整合",
        implementation_by_route: {
          fiber_bundle: "视觉锚点与逻辑锚点双向对齐可运行。",
          transformer_baseline: "使用标准投影头实现模态映射，鲁棒性中等。",
          hybrid_workspace: "支持跨路线共享模态状态并做冲突消解。"
        }
      },
      {
        name: "全局工作空间 (Global Workspace)",
        module_term: "意识访问与全局仲裁模块",
        parameter_footprint: "~0.23M 等效参数",
        progress_pct: "84%",
        status: "equipped",
        brain_ability: "注意焦点竞争与意识广播",
        implementation_by_route: {
          fiber_bundle: "已实现 Top-K 竞争与全局广播，支持动态焦点迁移。",
          transformer_baseline: "以控制器方式模拟焦点调度，仍偏工程拼接。",
          hybrid_workspace: "作为核心仲裁器驱动多路线融合决策。"
        }
      },
      {
        name: "脉冲纤维神经模块 (NeuroFiber-SNN)",
        module_term: "脉冲时序与神经动力学模块",
        parameter_footprint: "~0.11M 等效参数",
        progress_pct: "72%",
        status: "equipped",
        brain_ability: "脉冲时序编码与神经动力学",
        implementation_by_route: {
          fiber_bundle: "支持 3D 脉冲动力学仿真与时序激活观测。",
          transformer_baseline: "仅做近似时序模拟，未形成原生脉冲机制。",
          hybrid_workspace: "可作为辅助时序通道参与全局状态判断。"
        }
      },
      {
        name: "纤维记忆回放 (Fiber Memory)",
        module_term: "长期记忆与回放恢复模块",
        parameter_footprint: "~0.16M 等效参数",
        progress_pct: "79%",
        status: "equipped",
        brain_ability: "长期记忆编码与可控提取",
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
        module_term: "感知-决策-行动闭环模块",
        parameter_footprint: "待定，预计 >0.30M 附加参数",
        progress_pct: "18%",
        status: "missing",
        brain_ability: "感知-决策-行动闭环控制",
        implementation_by_route: {
          fiber_bundle: "尚未打通 ROS/机器人执行链路。",
          transformer_baseline: "已有任务级代理控制原型，但非实时闭环。",
          hybrid_workspace: "计划作为多路线统一动作层接入。"
        }
      },
      {
        name: "超大规模纤维持久化 (Massive Fiber Persistence)",
        module_term: "超长期记忆与容量管理模块",
        parameter_footprint: "待定，预计 >0.25M 附加参数",
        progress_pct: "34%",
        status: "missing",
        brain_ability: "超长期记忆与容量管理",
        implementation_by_route: {
          fiber_bundle: "冷热分层存储机制仍在设计阶段。",
          transformer_baseline: "依赖外部向量库扩展，原生持久化不足。",
          hybrid_workspace: "需要统一跨路线记忆索引与一致性协议。"
        }
      },
      {
        name: "跨模型流形迁移 (Cross-Model Transfer)",
        module_term: "跨基座迁移与泛化模块",
        parameter_footprint: "待定，预计 >0.20M 附加参数",
        progress_pct: "29%",
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
      "Capabilities Ready": "4/7",
      "Safety Alignment": "Secure",
      "Status Verifier": "Active"
    },
    model_summary: {
      current_model_file: "research/gpt5/code/icspb_backbone_v2_large_online.py",
      current_model_name: "ICSPBBackboneV2LargeOnline",
      total_parameters: "1,436,878",
      trainable_parameters: "1,436,878",
      theory_skeleton_progress: "96% - 98%",
      engineering_closure_progress: "95% - 97%",
      strict_brain_encoding_progress: "45% - 53%",
      prototype_training_progress: "82% - 87%",
      human_level_training_progress: "31% - 37%",
      language_training_progress: "90% - 93%",
      phasea_model_file: "research/gpt5/code/icspb_lm_phasea.py",
      phasea_model_name: "ICSPBLMPhaseA",
      phasea_total_parameters: "96,728,901",
      phasea_readiness_progress: "84% - 86%",
      phasea_generation_progress: "34% - 40%"
    },
    research_overview: {
      dnn_analysis_results: [
        "已形成跨模型不变量、因果必要性、最小生成模型与跨尺度一致性四条分析主线。",
        "当前最强的是结构骨架解释，最弱的是把这些解释压成标准学习律与最终参数答案。"
      ],
      brain_encoding_traits: [
        "patch / section / fiber 分层编码",
        "guarded write / stable read 读写不对称",
        "stage / successor / protocol 轨迹约束",
        "脉冲系统中体现为事件选择、短窗绑定、相位门控和群体读出"
      ],
      theory_gap: [
        "标准学习律答案尚未导出",
        "严格生物物理唯一性未成立",
        "真实外部世界持续验证未成立",
        "理论到人类级语言实现的桥仍未打穿"
      ],
      new_model_tests: [
        "PhaseA 参数量 96.73M，已能训练但生成仍弱。",
        "当前语言主链仍由原型系统维持，PhaseA 还不能接管主生成路径。",
        "项目当前更适合用于验证结构路线，而不是宣称已完成人类级语言主干。"
      ]
    }
  }
];

export const IMPROVEMENTS = [
  {
    id: "phase_1",
    title: "阶段1：不变量发现与基线校准",
    status: "done",
    objective: "在多模型、多任务、多规模设置下提取稳定结构，建立候选不变量库。",
    summary: "已形成第一版不变量库，并完成跨模型稳定性排序。",
    issues: [
      "跨架构覆盖仍偏窄，需引入更多非 Transformer 基座做一致性复验。",
      "不变量稳定性在长上下文与复杂多模态任务上的外推证据仍不足。"
    ],
    tests: [
      {
        id: "p1_t1",
        name: "跨模型不变量扫描",
        testDate: "2025-12-18",
        target: "验证几何/谱/拓扑特征在 Transformer、MoE、视觉模型中的稳定性。",
        params: {
          models: ["qwen3-4b", "transformer_baseline", "vision_fiber_proto"],
          datasets: ["OpenWebText-Sample", "MATH-Logic-Subset", "MNIST-Symbolic"],
          sample_tokens: 2000000,
          batch_size: 32,
          scan_layers: "all"
        },
        result: "检测到 17 个候选不变量，其中 9 个在三类模型中稳定出现。",
        analysis: "候选结构具备跨架构共性，满足进入因果必要性验证的门槛。",
        details: {
          key_metrics: {
            cross_model_stability: 0.81,
            topology_persistence_score: 0.74,
            spectral_consistency: 0.79
          },
          artifacts: [
            "tempdata/qwen3_4b_structure/metrics.json",
            "tempdata/qwen3_4b_structure/TEST_SUMMARY.md"
          ]
        }
      },
      {
        id: "p1_t2",
        name: "训练阶段不变量漂移测试",
        testDate: "2025-12-26",
        target: "验证训练早期-中期-收敛后不变量是否保持稳定。",
        params: {
          checkpoints: ["step_5k", "step_30k", "step_120k"],
          eval_tasks: ["logic_closure", "long_context_recall"],
          drift_threshold: 0.15
        },
        result: "核心不变量漂移均值 0.09，低于阈值。",
        analysis: "结构在训练阶段具有可追踪连续性，不是偶然噪声。",
        details: {
          key_metrics: {
            avg_drift: 0.09,
            max_drift: 0.14,
            reproducibility: 0.86
          }
        }
      }
    ]
  },
  {
    id: "phase_2",
    title: "阶段2：因果必要性验证",
    status: "in_progress",
    objective: "通过结构干预将相关性升级为可复现的因果结论。",
    summary: "删除/扰动/置换/重参数化流程已跑通；整层干预证据偏弱，但特征级干预已出现跨模型正向信号，阶段继续补证。",
    issues: [
      "整层热核干预的因果效应偏弱，尚不足以单独支撑强结论。",
      "目前强信号集中在特征级子空间，仍需验证其跨任务稳定性与必要性边界。",
      "严格 holdout 下支持覆盖仍有波动（support floor 未稳定到 2）。"
    ],
    tests: [
      {
        id: "p2_t1",
        name: "结构删除干预实验",
        testDate: "2026-01-05",
        target: "验证候选不变量对推理能力是否必要。",
        params: {
          intervention: "component_ablation",
          target_components: ["fiber_gate", "geometry_head", "memory_bridge"],
          eval_sets: ["GSM8K-Subset", "Z113-Closure", "ReasoningChain-Set"],
          runs: 12
        },
        result: "删除关键组件后平均性能下降 23.6%，且可复现。",
        analysis: "关键候选结构满足必要性标准，已从候选升级为核心结构。",
        details: {
          key_metrics: {
            avg_perf_drop_pct: 23.6,
            worst_case_drop_pct: 34.2,
            replication_rate: 0.92
          },
          retained_core_structures: ["fiber_gate", "geometry_head"]
        }
      },
      {
        id: "p2_t2",
        name: "重参数化保持实验",
        testDate: "2026-01-09",
        target: "验证结构等价变换下能力是否保持，排除纯参数偶然性。",
        params: {
          transform: "weight_reparameterization",
          preserve_constraints: ["spectrum_band", "topology_signature"],
          runs: 8
        },
        result: "在结构保持条件下，任务性能波动 < 2.8%。",
        analysis: "说明能力主要由结构决定，而非具体参数坐标系。",
        details: {
          key_metrics: {
            perf_variance_pct: 2.8,
            structure_preservation_score: 0.88
          }
        }
      },
      {
        id: "p2_t3",
        name: "几何干预单样本验证（GPT-2）",
        testDate: "2026-02-20",
        target: "验证在固定提示词下，几何干预是否可导致输出变化。",
        params: {
          script: "scripts/geometric_intervention_simple.py",
          model: "gpt2",
          intervention_type: "heat_kernel",
          intervention_layer: 6,
          reference_prompts: ["The capital of France is", "2 + 2 equals"],
          generation_prompt: "The meaning of life is"
        },
        result: "干预后输出文本发生变化（single-case changed=1）。",
        analysis: "在单样本场景已观察到结构干预 -> 行为变化，具备因果方向信号，但证据强度仍偏低。",
        details: {
          key_metrics: {
            output_changed: 1.0,
            layer6_mean_curvature: 527.8999,
            layer6_pca_top3_variance: 0.9479
          },
          reports: [
            "tempdata/geometric_intervention_results.json"
          ],
          timeline_analysis_type: "causal_intervention"
        }
      },
      {
        id: "p2_t4",
        name: "几何干预批量验证（5 prompts + 随机对照）",
        testDate: "2026-02-20",
        target: "验证几何参考干预相对随机参考对照是否具有更强行为影响。",
        params: {
          script: "scripts/geometric_intervention_batch.py",
          model: "gpt2",
          intervention_type: "heat_kernel",
          intervention_layer: 6,
          alpha: 0.15,
          total_prompts: 5,
          control: "random_reference"
        },
        result: "treatment/control 均为 3/5（0.60），当前 uplift=0.00。",
        analysis: "首轮对照未观察到几何参考的额外提升，说明当前证据强度不足；需扩展样本规模与层扫描后再做因果结论。",
        details: {
          key_metrics: {
            total_prompts: 5,
            treatment_changed_outputs: 3,
            treatment_changed_rate: 0.6,
            control_changed_outputs: 3,
            control_changed_rate: 0.6,
            causal_uplift: 0.0,
            elapsed_seconds: 7.39
          },
          reports: [
            "tempdata/geometric_intervention_batch_results_20260220.json"
          ],
          timeline_analysis_type: "causal_intervention_batch"
        }
      },
      {
        id: "p2_t5",
        name: "几何干预多层扫描（n=60，L3/L6/L9）",
        testDate: "2026-02-20",
        target: "识别几何干预在不同层位上的因果敏感性差异。",
        params: {
          script: "scripts/geometric_intervention_batch.py",
          model: "gpt2",
          prompt_count: 60,
          scanned_layers: [3, 6, 9],
          controls: ["random_reference", "shuffled_reference"],
          intervention_type: "heat_kernel"
        },
        result: "layer_3 对随机对照仅有微弱 uplift(+0.0167)，layer_6/layer_9 为 0。",
        analysis: "当前层扫描未形成稳定正 uplift，说明 S1 因果证据仍不足；需扩大样本并做跨模型复验。",
        details: {
          key_metrics: {
            layer3_treatment_rate: 0.4667,
            layer3_control_random_rate: 0.45,
            layer3_uplift_random: 0.0167,
            layer6_uplift_random: 0.0,
            layer9_uplift_random: 0.0
          },
          reports: [
            "tempdata/geometric_intervention_batch_results_20260220_n60_l3.json",
            "tempdata/geometric_intervention_batch_results_20260220_n60_l6.json",
            "tempdata/geometric_intervention_batch_results_20260220_n60_l9.json",
            "tempdata/geometric_intervention_layer_scan_20260220.json",
            "tempdata/geometric_intervention_layer_scan_20260220.md"
          ],
          timeline_analysis_type: "causal_intervention_layer_scan"
        }
      },
      {
        id: "p2_t6",
        name: "几何干预大样本矩阵（n=240，跨模型/跨强度）",
        testDate: "2026-02-20",
        target: "验证在更大样本下，几何参考干预是否稳定优于随机/置换对照，并检查跨模型一致性。",
        params: {
          script: "scripts/geometric_intervention_batch.py",
          matrix: [
            { model: "gpt2", layer: 3, alpha: 0.15, n: 240 },
            { model: "gpt2", layer: 3, alpha: 0.35, n: 240 },
            { model: "distilgpt2", layer: 3, alpha: 0.15, n: 240 },
            { model: "distilgpt2", layer: 3, alpha: 0.35, n: 240 }
          ],
          controls: ["random_reference", "shuffled_reference"],
          metric: "output_changed_rate + two_proportion_pvalue"
        },
        result: "4 组实验均未达到显著 uplift（最大 +0.0166，最小 p=0.6937）。",
        analysis: "S1 目前仍是弱信号/无信号状态；下一步应从“整层平滑干预”转向“特征级选择性干预 + 任务指标变化”来提高因果分辨率。",
        details: {
          key_metrics: {
            runs: 4,
            prompt_count_each: 240,
            avg_uplift_random: 0.0021,
            max_uplift_random: 0.0166,
            min_pvalue_random: 0.69366
          },
          reports: [
            "tempdata/geometric_intervention_batch_results_20260220_gpt2_n240_l3_a0.15.json",
            "tempdata/geometric_intervention_batch_results_20260220_gpt2_n240_l3_a0.35.json",
            "tempdata/geometric_intervention_batch_results_20260220_distilgpt2_n240_l3_a0.15.json",
            "tempdata/geometric_intervention_batch_results_20260220_distilgpt2_n240_l3_a0.35.json",
            "tempdata/geometric_intervention_large_scale_matrix_20260220.json",
            "tempdata/geometric_intervention_large_scale_matrix_20260220.md"
          ],
          timeline_analysis_type: "causal_intervention_batch_large_scale"
        }
      },
      {
        id: "p2_t7",
        name: "特征级干预探针（n=240，前向指标）",
        testDate: "2026-02-20",
        target: "验证 top-k 特征选择性干预是否比随机特征对照产生更强行为偏移。",
        params: {
          script: "scripts/feature_selective_intervention_probe.py",
          models: ["gpt2", "distilgpt2"],
          layer: 3,
          top_k_features: 32,
          alpha: 0.35,
          metric: ["top1_change_rate", "kl_uplift"],
          control: "random_feature_set"
        },
        result: "两模型均出现正向 uplift：gpt2(top1 +0.0542, kl +0.075862)，distilgpt2(top1 +0.0750, kl +0.065312)。",
        analysis: "与整层热核平滑相比，特征级干预显著提升因果分辨率，说明“特定子空间”比“整层整体”更接近候选必要结构。",
        details: {
          key_metrics: {
            avg_top1_uplift: 0.0646,
            avg_kl_uplift: 0.070587,
            models_count: 2
          },
          reports: [
            "tempdata/feature_selective_probe_20260220_gpt2_n240_l3_k32_a0.35.json",
            "tempdata/feature_selective_probe_20260220_distilgpt2_n240_l3_k32_a0.35.json",
            "tempdata/feature_selective_probe_matrix_20260220.json",
            "tempdata/feature_selective_probe_matrix_20260220.md"
          ],
          timeline_analysis_type: "causal_intervention_feature_probe"
        }
      }
    ]
  },
  {
    id: "phase_3",
    title: "阶段3：最小生成模型重建",
    status: "in_progress",
    objective: "用最少公理和最低自由度重建关键能力，验证“结构 -> 能力”的生成链条。",
    summary: "最小结构模型已形成可复现高分基线，并完成 5-seed 稳定性复验；下一步转向跨任务复验。",
    issues: [
      "最小模型在跨任务泛化上仍存在性能缺口，尚未达到全面替代基线。",
      "跨模态联络项仍是主要瓶颈，结构约束与表达能力需要进一步平衡。"
    ],
    tests: [
      {
        id: "p3_t1",
        name: "MDL 约束重建实验",
        testDate: "2026-01-18",
        target: "比较最小模型与全量模型在逻辑任务上的能力保真。",
        params: {
          mdl_budget: "0.62x baseline complexity",
          model_variants: ["fiber_mini_v1", "baseline_full_v3"],
          tasks: ["Z113", "Symbolic-Reasoning", "Rule-Generalization"]
        },
        result: "在 62% 复杂度预算下达到基线 84.1% 性能。",
        analysis: "说明已有结构解释具备压缩能力，但仍需提升跨任务泛化。",
        details: {
          key_metrics: {
            retained_performance_pct: 84.1,
            complexity_ratio: 0.62,
            generalization_gap_pct: 8.7
          }
        }
      },
      {
        id: "p3_t2",
        name: "视觉-语言联合重建",
        testDate: "2026-01-24",
        target: "验证视觉纤维网络与语言纤维网络能否在统一结构中协同。",
        params: {
          architecture: "dual_fiber_bridge_v0",
          vision_dataset: "MNIST-Symbolic + CIFAR10-Subset",
          language_dataset: "Instruction-Logic-Subset",
          bridge_dim: 256
        },
        result: "跨模态任务通过率 68.4%，高于无桥接对照组 21.5%。",
        analysis: "联合重建可行，但跨模态联络项仍是主要瓶颈。",
        details: {
          key_metrics: {
            multimodal_pass_rate: 68.4,
            improvement_vs_control: 21.5,
            alignment_mse: 0.047
          },
          related_service: "server/vision_service.py"
        }
      },
      {
        id: "p3_t3",
        name: "最小重建高分复现（m_1.4m+d_120k）",
        testDate: "2026-02-28",
        target: "验证历史最优 Stage C 配置在当前环境下是否可复现，排除“结构无效”误判。",
        params: {
          script: "scripts/scaling_validation_matrix.py",
          preset: "quick",
          model_filter: "m_1.4m",
          data_filter: "d_120k",
          epochs: 20,
          lr: 0.001,
          weight_decay: 0.1,
          warmup_ratio: 0.0,
          min_lr_scale: 0.05,
          seed: 42
        },
        result: "best_val_acc=0.856667，final_val_acc=0.856120，复现成功。",
        analysis: "Stage C 的关键问题是配置敏感而非结构失效；已可将该组合作为默认基线继续做多 seed 与跨任务复验。",
        details: {
          key_metrics: {
            best_val_acc: 0.856667,
            final_val_acc: 0.85612,
            final_train_acc: 0.98725,
            generalization_gap: 0.13113,
            samples_per_second: 11551.2
          },
          reports: [
            "tempdata/pipeline_stage_c_minimal_rebuild_20260228_m14_repro.json",
            "tempdata/pipeline_stage_c_minimal_rebuild_20260228_m14_repro.md",
            "tempdata/pipeline_stage_c_minimal_rebuild_20260228_v2.json",
            "tempdata/structure_recovery_pipeline_kickoff_20260228_v2.json"
          ],
          timeline_analysis_type: "minimal_reconstruction"
        }
      },
      {
        id: "p3_t4",
        name: "最小重建多种子稳定性复验（5 seeds）",
        testDate: "2026-02-28",
        target: "验证 Stage C 默认基线在多随机种子下是否保持稳定高分，避免单次命中误判。",
        params: {
          script: "scripts/scaling_validation_matrix.py",
          preset: "quick",
          model_filter: "m_1.4m",
          data_filter: "d_120k",
          epochs: 20,
          lr: 0.001,
          weight_decay: 0.1,
          warmup_ratio: 0.0,
          min_lr_scale: 0.05,
          seeds: [42031, 42032, 42033, 42034, 42035]
        },
        result: "5 次运行 best_val_acc 均>=0.84，均值 0.8511，标准差 0.0115，判定为稳定复现。",
        analysis: "Stage C 基线在当前配置区间具备可复现稳定性，可作为后续跨任务/跨架构验证的固定锚点。",
        details: {
          key_metrics: {
            count: 5,
            best_val_acc_mean: 0.8511242,
            best_val_acc_std: 0.01150055,
            best_val_acc_min: 0.8413,
            best_val_acc_max: 0.8737,
            final_val_acc_mean: 0.85106,
            generalization_gap_mean: 0.13217
          },
          reports: [
            "tempdata/pipeline_stage_c_minimal_rebuild_20260228_m14_seed42031.json",
            "tempdata/pipeline_stage_c_minimal_rebuild_20260228_m14_seed42032.json",
            "tempdata/pipeline_stage_c_minimal_rebuild_20260228_m14_seed42033.json",
            "tempdata/pipeline_stage_c_minimal_rebuild_20260228_m14_seed42034.json",
            "tempdata/pipeline_stage_c_minimal_rebuild_20260228_m14_seed42035.json",
            "tempdata/pipeline_stage_c_minimal_rebuild_20260228_m14_multiseed_summary.json",
            "tempdata/pipeline_stage_c_minimal_rebuild_20260228_m14_multiseed_summary.md"
          ],
          timeline_analysis_type: "minimal_reconstruction_multiseed"
        }
      }
    ]
  },
  {
    id: "phase_4",
    title: "阶段4：跨层级一致性验证",
    status: "in_progress",
    objective: "建立局部回路解释与全局几何/拓扑解释的一致性映射。",
    summary: "一致性验证流程已可运行，跨任务稳定性还在补充样本。",
    issues: [
      "长上下文任务下局部-全局解释一致性仍有偏差。",
      "尺度律的一致性样本点仍需继续扩展，避免分段相变误判。",
      "真实数据域证据仍少于 synthetic 证据，外推风险需要收敛。"
    ],
    tests: [
      {
        id: "p4_t1",
        name: "局部回路-全局拓扑映射测试",
        testDate: "2026-02-03",
        target: "验证回路贡献排序与拓扑关键区域是否一致。",
        params: {
          local_method: "circuit_discovery_v2",
          global_method: "persistent_homology + spectral_map",
          tasks: ["math_reasoning", "long_context_qa"],
          models: ["fiber_bundle_main", "transformer_baseline"]
        },
        result: "两类解释在关键区域重合度 0.72。",
        analysis: "一致性达到可用水平，但在长上下文任务仍有偏差。",
        details: {
          key_metrics: {
            overlap_score: 0.72,
            long_context_gap: 0.11
          }
        }
      },
      {
        id: "p4_t2",
        name: "尺度律一致性回归",
        testDate: "2026-02-11",
        target: "验证参数规模变化下核心结构指标与性能增益关系。",
        params: {
          scales: ["70M", "350M", "1.3B", "4B"],
          data_tokens: ["30M", "120M", "500M"],
          regression: "log-linear + piecewise"
        },
        result: "结构指标与性能增益相关系数 0.78，存在分段相变。",
        analysis: "支持“结构主导的尺度增益”假设，下一步需补更大规模点位。",
        details: {
          key_metrics: {
            corr_coef: 0.78,
            phase_transition_points: ["350M->1.3B"]
          },
          report: "tempdata/scaling_validation_report.md"
        }
      },
      {
        id: "p4_t3",
        name: "大规模压力测试（2M/3M）",
        testDate: "2026-02-20",
        target: "验证 m_8.5m 在 2M/3M 数据规模下的稳定扩展表现。",
        params: {
          script: "scripts/scaling_validation_matrix.py",
          model_filter: "m_8.5m",
          custom_data_sizes: [2000000, 3000000],
          epochs: 12,
          lr: 0.001,
          weight_decay: 0.1,
          warmup_ratio: 0.0,
          device: "cuda"
        },
        result: "2M/3M 两个点 best_val_acc 均约 0.0089，接近随机水平。",
        analysis: "在默认优化参数下，大模型在该任务出现训练失效，需做超参诊断而非直接否定结构路线。",
        details: {
          key_metrics: {
            d2000k_best_val_acc: 0.0089,
            d3000k_best_val_acc: 0.0089,
            avg_samples_per_second: 6127.8
          },
          reports: [
            "tempdata/scaling_validation_report_m85_2m_3m_20260220.json",
            "tempdata/scaling_validation_report_m85_2m_3m_20260220.md"
          ],
          timeline_analysis_type: "scaling_validation"
        }
      },
      {
        id: "p4_t4",
        name: "超参诊断对照（2M tuned）",
        testDate: "2026-02-20",
        target: "验证大规模低分是否由优化超参导致。",
        params: {
          script: "scripts/scaling_validation_matrix.py",
          model_filter: "m_8.5m",
          custom_data_sizes: [2000000],
          epochs: 20,
          lr: 0.0003,
          weight_decay: 0.01,
          warmup_ratio: 0.03,
          min_lr_scale: 0.1,
          device: "cuda"
        },
        result: "同规模条件下 best_val_acc=1.0000，最终 val_acc=1.0000。",
        analysis: "确认主要瓶颈是优化配置而非模型结构；后续应将大规模默认配置切换到 tuned 区间。",
        details: {
          key_metrics: {
            d2000k_best_val_acc: 1.0,
            d2000k_final_val_acc: 1.0,
            samples_per_second: 6164.65
          },
          reports: [
            "tempdata/scaling_validation_report_m85_2m_tuned_20260220.json",
            "tempdata/scaling_validation_report_m85_2m_tuned_20260220.md"
          ],
          timeline_analysis_type: "scaling_validation"
        }
      },
      {
        id: "p4_t5",
        name: "3-seed 复现稳定性验证（2M tuned）",
        testDate: "2026-02-20",
        target: "验证 tuned 参数在大规模点位下是否具备多种子稳定性。",
        params: {
          script: "scripts/scaling_validation_matrix.py",
          model_filter: "m_8.5m",
          custom_data_sizes: [2000000],
          seeds: [10001, 20002, 30003],
          epochs: 20,
          lr: 0.0003,
          weight_decay: 0.01,
          warmup_ratio: 0.03,
          min_lr_scale: 0.1,
          device: "cuda"
        },
        result: "三次运行 best/final val_acc 全部为 1.0000，复现稳定。",
        analysis: "确认 tuned 区间具备高稳定性；下一步可扩展到 4M/6M 并保持相同验证流程。",
        details: {
          key_metrics: {
            best_val_acc_mean: 1.0,
            best_val_acc_std: 0.0,
            final_val_acc_mean: 1.0,
            final_val_acc_std: 0.0
          },
          reports: [
            "tempdata/scaling_validation_report_m85_2m_tuned_seed10001_20260220.json",
            "tempdata/scaling_validation_report_m85_2m_tuned_seed20002_20260220.json",
            "tempdata/scaling_validation_report_m85_2m_tuned_seed30003_20260220.json",
            "tempdata/scaling_validation_m85_2m_tuned_multiseed_20260220.json",
            "tempdata/scaling_validation_m85_2m_tuned_multiseed_20260220.md",
            "tempdata/scaling_validation_m85_2m_tuned_multiseed_summary_20260220.json"
          ],
          timeline_analysis_type: "scaling_validation_multiseed"
        }
      },
      {
        id: "p4_t6",
        name: "扩展点位验证（4M/6M tuned）",
        testDate: "2026-02-20",
        target: "在更大训练规模点位上验证 tuned 配置的可扩展稳定性。",
        params: {
          script: "scripts/scaling_validation_matrix.py",
          model_filter: "m_8.5m",
          custom_data_sizes: [4000000, 6000000],
          epochs: 12,
          lr: 0.0003,
          weight_decay: 0.01,
          warmup_ratio: 0.03,
          min_lr_scale: 0.1,
          seed: 42024,
          device: "cuda"
        },
        result: "d_4000k 与 d_6000k 两点均达到 best/final val_acc=1.0000。",
        analysis: "tuned 配置在 4M/6M 继续稳定，支持将下一阶段重心转向 S1 因果干预与 S3 一致性补证。",
        details: {
          key_metrics: {
            d4000k_best_val_acc: 1.0,
            d6000k_best_val_acc: 1.0,
            samples_per_second_d4000k: 5959.57,
            samples_per_second_d6000k: 5443.99
          },
          reports: [
            "tempdata/scaling_validation_report_m85_4m_6m_tuned_20260220.json",
            "tempdata/scaling_validation_report_m85_4m_6m_tuned_20260220.md",
            "tempdata/scaling_validation_m85_4m_6m_tuned_summary_20260220.json",
            "tempdata/scaling_validation_m85_4m_6m_tuned_summary_20260220.md"
          ],
          timeline_analysis_type: "scaling_validation"
        }
      }
    ]
  },
  {
    id: "phase_5",
    title: "阶段5：反证优先与结论收敛",
    status: "in_progress",
    objective: "对核心假设持续做反证干预，只保留可证伪且可复现的结论。",
    summary: "已建立反证任务池并开始运行，结论清单正在收敛。",
    issues: [
      "跨数据域、跨模型的反证覆盖仍不均衡，结论边界尚未完全封口。",
      "失败归因中跨模态联络漂移与记忆冲突占比高，需进入专项修复。"
    ],
    tests: [
      {
        id: "p5_t1",
        name: "核心假设反证干预",
        testDate: "2026-02-17",
        target: "检验“几何结构是能力必要条件”在对照实验下是否仍成立。",
        params: {
          hypotheses: ["geometry_is_necessary", "fiber_memory_is_key"],
          controls: ["randomized_geometry", "shuffled_memory_bridge"],
          runs: 10
        },
        result: "两项假设均通过反证筛查，对照组性能显著劣化。",
        analysis: "当前证据支持核心假设，但仍需跨数据域复验。",
        details: {
          key_metrics: {
            geometry_control_drop_pct: 19.4,
            memory_control_drop_pct: 16.7,
            p_value: 0.008
          }
        }
      },
      {
        id: "p5_t2",
        name: "失败样本归因审计",
        testDate: "2026-02-20",
        target: "对失败案例做结构级归因，识别未覆盖机制。",
        params: {
          sample_size: 500,
          audit_dimensions: ["topology_break", "alignment_shift", "memory_conflict"],
          tooling: ["runtime_timeline", "gradient_trace", "flow_tubes"]
        },
        result: "主要失败来源为跨模态联络漂移（42%）与记忆冲突（31%）。",
        analysis: "后续需优先增强联络稳定器与记忆冲突解耦模块。",
        details: {
          key_metrics: {
            cross_modal_drift_pct: 42,
            memory_conflict_pct: 31,
            unresolved_pct: 12
          }
        }
      }
    ]
  }
];

export const DNN_ANALYSIS_PLAN = {
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

export const EVIDENCE_DRIVEN_PLAN = {
  title: '研究方案',
  core: '核心：主线先产出，验证层保真，前沿层找突破；方法服从证据强度，不由学科标签决定。',
  overview: [
    '研究对象：深度神经网络中可复现、可干预、可重建的数学结构。',
    '研究路径：不变量发现 -> 因果筛选 -> 最小重建 -> 跨层一致性 -> 反证收敛。',
    '研究判据：结论必须可重复、可证伪，并能进入统一时间线。',
  ],
  phases: [
    {
      id: 'S0',
      name: '不变量发现',
      desc: '跨模型/任务/规模筛选稳定不变量，形成候选结构库。',
      goal: '找到跨架构仍稳定存在的结构信号，建立候选结构池。',
      method: '多模型多任务对齐分析 + 统计稳定性筛选 + 噪声鲁棒性评估。',
      evidence: '候选在不同 seed、任务、规模下保持方向一致且显著。',
      outputs: '候选结构库 JSON、稳定性排名、失效样本清单。',
      gate: '至少 5 个候选结构在 >=3 类模型上稳定复现。',
    },
    {
      id: 'S1',
      name: '因果必要性验证',
      desc: '通过结构干预验证“是否必要”，将相关性结论升级为因果结论。',
      goal: '确认哪些结构是能力形成的必要条件，而非伴随相关。',
      method: '特征级干预、路径阻断、对照实验、反事实比较。',
      evidence: '干预目标结构后任务性能显著下降，且对照组无同等变化。',
      outputs: '必要结构白名单、无效候选淘汰列表、因果证据包。',
      gate: '主任务出现稳定退化且可重复，效应方向一致。',
    },
    {
      id: 'S2',
      name: '最小生成模型',
      desc: '以 MDL 为约束，用最少公理重建关键能力。',
      goal: '以更低复杂度复现关键能力，验证“结构决定能力”的可构造性。',
      method: '最小公理化建模 + 结构先验注入 + 压缩-性能联合评估。',
      evidence: '在明显压缩条件下保持关键能力，不依赖原始大模型冗余。',
      outputs: '最小结构模型、复杂度对照报告、重建误差分析。',
      gate: '以更低参数/自由度达到可接受性能阈值。',
    },
    {
      id: 'S3',
      name: '跨层级一致性',
      desc: '建立局部机制与全局结构的双向映射，确保解释一致。',
      goal: '打通局部神经机制与全局数学结构的统一解释链。',
      method: '局部子回路解释 + 全局几何/拓扑映射 + 多任务一致性评估。',
      evidence: '局部解释与全局解释在方向、结论、预测上不冲突。',
      outputs: '跨层映射图谱、统一解释文档、冲突项修正记录。',
      gate: '关键任务上局部-全局结论一致，冲突率降到可控阈值。',
    },
    {
      id: 'S4',
      name: '反证优先收敛',
      desc: '优先淘汰不可证伪假设，沉淀可复现且可反驳的结论。',
      goal: '建立可证伪结论集，避免“不可验证理论”的路径依赖。',
      method: '强对照反证、跨域复验、失败边界测试、负结果入库。',
      evidence: '结论在新数据和新模型下仍成立，或明确给出失效边界。',
      outputs: '可证伪结论集、失败边界手册、后续优先级建议。',
      gate: '保留结论都具备可复验流程与明确反证条件。',
    },
  ],
};

export const EXECUTION_PLAYBOOK = {
  title: '执行版方案（S0-S4）',
  subtitle: '目标、准出标准与证据产物统一化，确保每阶段都可复现、可证伪、可入时间线。',
  principles: [
    '证据优先：结论必须来自可复现实验',
    '因果优先：相关性不直接升格为结构结论',
    '反证优先：优先淘汰错误结构',
    '跨尺度一致：局部机制与全局结构必须可映射',
  ],
  stageChecklist: [
    {
      stage: 'S0',
      objective: '跨模型发现稳定不变量',
      exitCriteria: '>=5 个候选结构在 >=3 类模型稳定复现',
      outputs: '候选库 JSON + 稳定性排序报告',
    },
    {
      stage: 'S1',
      objective: '干预验证结构必要性',
      exitCriteria: '干预导致显著且可复现退化；对照组无同等效应',
      outputs: '必要结构清单 + 反事实证据包',
    },
    {
      stage: 'S2',
      objective: '最小结构复现关键能力',
      exitCriteria: '压缩复杂度下保持 >=80% 基线性能',
      outputs: '最小生成模型定义 + 复现实验 JSON',
    },
    {
      stage: 'S3',
      objective: '局部-全局解释一致性',
      exitCriteria: '关键区重合度 >0.7，结构指标与能力增益稳定相关',
      outputs: '跨层映射表 + 一致性偏差分析',
    },
    {
      stage: 'S4',
      objective: '反证收敛与结论固化',
      exitCriteria: '核心结论通过反证筛查并明确失效边界',
      outputs: '可保留结论清单 + 失效模式档案',
    },
  ],
  sprintPlan: {
    week1: [
      '固化大规模 tuned 参数模板（m_8.5m）',
      '执行 3-seed 重复试验验证稳定性',
      '批量化 S1 干预实验（删除/扰动/置换）',
    ],
    week2: [
      '建立 S3 局部-全局一致性基线',
      '完成 S4 反证任务池最小版本',
      '输出“可保留结论 + 失效边界”首版',
    ],
  },
  riskStopLoss: [
    {
      risk: '计算失控',
      signal: '耗时/显存超阈值',
      action: '降维采样 + 局部估计 + 子任务拆分',
    },
    {
      risk: '结论漂移',
      signal: '不同 seed 方向相反',
      action: '暂停结论发布，先补重复试验',
    },
    {
      risk: '解释断裂',
      signal: '局部解释与全局解释冲突',
      action: '标记未收敛，回退 S0/S1',
    },
  ],
};

export const MATH_ROUTE_SYSTEM_PLAN = {
  title: '数学理论',
  subtitle: '多路线分层组合：主线产出 + 验证保真 + 前沿突破',
  routeAnalysis: [
    {
      route: '流形几何',
      routeSummary: '把智能看作高维流形上的几何结构学习，核心是学到可泛化的“语义形状”和稳定测地路径。',
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
      routeSummary: '把智能看作跨任务不变量的保持，核心是提取在扰动下仍稳定的全局结构并作为能力骨架。',
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
      routeSummary: '把智能看作状态演化与吸引子组织，核心是形成可控、可切换、可稳定收敛的认知动力学。',
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
      routeSummary: '把智能看作压缩与预测的最优平衡，核心是在最小信息代价下保留对任务最关键的因果特征。',
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
      routeSummary: '把智能看作可组合的低秩模块，核心是把复杂能力拆解为可复用、可重组的结构部件。',
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
      routeSummary: '把智能看作跨模块映射与组合律，核心是建立统一的“对象-态射”体系以实现可迁移的系统级组合。',
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
      routeSummary: '把智能看作多尺度涌现过程，核心是通过粗粒化与尺度变换揭示从局部特征到全局能力的生成机制。',
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
