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
        test_result: "Z113 模运算准确率 99.4%, 成功从统计拟合跃迁至离散傅里叶变换 (DFT) 代数表示。",
        analysis: "模型在向量空间构建了完美的 S1 圆环流形。这证明了智能的底层是几何化的逻辑骨架，而非简单的模式识别。"
      },
      { 
        name: "感官的觉醒", 
        status: "done", 
        focus: "多模态语义对齐",
        test_result: "MNIST 视觉锚点与 SYM 逻辑锚点 L2 距离收敛至 0.042 (MSE)，实现跨模态同构投影。",
        analysis: "成功破解了‘符号接地’难题。视觉特征被精准翻译为内部逻辑坐标，模型初步具备了‘观察并理解’的能力。"
      },
      { 
        name: "智慧的涌现", 
        status: "done", 
        focus: "流形曲率优化与 Ricci Flow",
        test_result: "诱发 Grokking 现象，Betti-1 数值趋于稳定，流形拓扑亏格从 15 优化至 2 (Smoother Geometry)。",
        analysis: "通过离线 Ricci Flow 优化，模型在无监督情况下自发修补逻辑死结。这验证了‘睡眠机制’在解开拓扑纠缠中的必要性。"
      },
      { 
        name: "价值的形成", 
        status: "in_progress", 
        focus: "神经流形手术与人类对齐",
        test_result: "Surgery Alignment Loss 0.0082, 成功通过 3D 空间拖拽实现 Concept Steering 实时干预。",
        analysis: "流形手术允许直接修改模型认知，而非依赖提示词工程。这是通往可控 AGI 和价值对齐的物理路径。"
      },
      { 
        name: "统一意识", 
        status: "pending", 
        focus: "全球工作空间集成",
        test_result: "Pending integration with Base Manifold Controller.",
        analysis: "目标是构建一个能够裁决跨模态冲突、具备主观关注点 (Locus of Attention) 的全局工作空间。"
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
        summary: "基于 Project Genesis 协议的核心能力对齐报告，详细展示系统当前已具备与尚在开发的各项核心智能属性。",
        pillars: ["Capability Tracking", "Homeostatic Check", "Alignment Metrics"]
    },
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
    status: "pending"
  }
];

export const HLAIBlueprint = ({ onClose }) => {
  const [activeTab, setActiveTab] = useState('roadmap'); // roadmap, progress, system
  const [activePhaseId, setActivePhaseId] = useState('theory');

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
            <span style={{ fontSize: '18px', fontWeight: 'bold', letterSpacing: '2px' }}>PROJECT GENESIS</span>
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
            padding: '30px 20px', background: 'rgba(0,0,0,0.2)', overflowY: 'auto'
          }}>
            <div style={{ fontSize: '10px', color: '#444', textTransform: 'uppercase', marginBottom: '20px', letterSpacing: '2px', fontWeight: 'bold' }}>Dimensions</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {progressPhases.map(p => (
                <button
                  key={p.id}
                  onClick={() => setActivePhaseId(p.id)}
                  style={{
                    padding: '14px 18px', borderRadius: '12px', textAlign: 'left', cursor: 'pointer',
                    background: activePhaseId === p.id ? 'rgba(255,255,255,0.05)' : 'transparent',
                    border: `1px solid ${activePhaseId === p.id ? p.color + '40' : 'transparent'}`,
                    color: activePhaseId === p.id ? '#fff' : '#666', transition: 'all 0.3s',
                    display: 'flex', alignItems: 'center', gap: '12px'
                  }}
                >
                  <div style={{ color: activePhaseId === p.id ? p.color : '#333' }}>{p.icon}</div>
                  <span style={{ fontSize: '14px', fontWeight: 'bold' }}>{p.subtitle.split(' (')[0]}</span>
                </button>
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
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                  {activePhase.theory_content.map((t, idx) => (
                    <div key={idx} style={{ padding: '30px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '24px' }}>
                      <div style={{ fontWeight: 'bold', color: activePhase.color, marginBottom: '20px', fontSize: '15px' }}>{t.title}</div>
                      <div style={{ padding: '25px', background: '#000', borderRadius: '12px', textAlign: 'center', fontSize: '28px', color: '#fff', fontFamily: 'serif', marginBottom: '16px', border: '1px solid rgba(255,255,255,0.1)' }}>{t.formula}</div>
                      <p style={{ fontSize: '12px', color: '#777', lineHeight: '1.6', margin: 0 }}>{t.desc}</p>
                    </div>
                  ))}
                </div>
              )}

              {activePhase.id === 'engineering' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {activePhase.sub_phases.map((s, idx) => (
                    <div key={idx} style={{ padding: '24px', background: s.status === 'in_progress' ? 'rgba(0,210,255,0.05)' : 'rgba(255,255,255,0.02)', borderRadius: '20px', border: '1px solid rgba(255,255,255,0.05)', display: 'grid', gridTemplateColumns: '1fr 2fr 100px', alignItems: 'center', gap: '30px' }}>
                      <div style={{ fontWeight: 'bold', fontSize: '16px' }}>{s.name}</div>
                      <div style={{ fontSize: '13px', color: '#888' }}>{s.test_result}</div>
                      <div style={{ textAlign: 'right', fontWeight: 'bold', color: s.status === 'done' ? '#00ff88' : (s.status === 'in_progress' ? '#00d2ff' : '#333'), fontSize: '10px' }}>{s.status.toUpperCase()}</div>
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
                <h2 style={{ fontSize: '32px', fontWeight: '900', color: '#10b981', margin: '20px 0 8px 0' }}>系统状态 (System Status)</h2>
                <p style={{ color: '#666', fontSize: '14px' }}>基于 Project Genesis 协议的核心能力对齐报告</p>
              </div>

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
            </div>
          )}

        </div>
      </div>
    </div>
  );
};
