import { ContactShadows, OrbitControls, PerspectiveCamera, Text } from '@react-three/drei';
import { Canvas, useFrame } from '@react-three/fiber';
import axios from 'axios';
import {
  Activity, ArrowRightLeft, BarChart, BarChart2, Brain, CheckCircle, GitBranch, Globe, Globe2,
  Grid3x3, HelpCircle, Layers, Loader2, Maximize2, Minimize2, Network, RefreshCw, RotateCcw,
  Scale, Search, Settings, Share2, Sparkles, Target, TrendingUp, X, Bot, Zap
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import FiberNetV2Demo from './components/FiberNetV2Demo';
import ErrorBoundary from './ErrorBoundary';
import FlowTubesVisualizer from './FlowTubesVisualizer';
import GlassMatrix3D from './GlassMatrix3D';
import { GlobalTopologyDashboard } from './GlobalTopologyDashboard';
import { HLAIBlueprint } from './HLAIBlueprint';
import ResonanceField3D from './ResonanceField3D';
import { SimplePanel } from './SimplePanel';
import { CompositionalVisualization3D, CurvatureField3D, FeatureVisualization3D, FiberBundleVisualization3D, LayerDetail3D, ManifoldVisualization3D, NetworkGraph3D, RPTVisualization3D, SNNVisualization3D, StructureAnalysisControls, ValidityVisualization3D } from './StructureAnalysisPanel';
import TDAVisualization3D from './TDAVisualization3D';
import { AGIChatPanel } from './AGIChatPanel';
import { MotherEnginePanel } from './components/MotherEnginePanel';

import { locales } from './locales';
import { INPUT_PANEL_TABS, STRUCTURE_TABS_V2, COLORS } from './config/panels';
import { AnalysisDataDisplay, MetricsRow, MetricCard } from './components/shared/DataDisplayTemplates';
import { OperationHistoryPanel, useOperationHistory } from './components/shared/OperationHistory';
import { DataComparisonView } from './components/shared/DataComparisonView';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

const navButtonStyle = (isActive, activeColor) => ({
  position: 'absolute',
  top: 20,
  zIndex: 101,
  background: isActive ? activeColor : 'rgba(20, 20, 25, 0.8)',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: '8px',
  padding: '8px',
  cursor: 'pointer',
  color: 'white',
  backdropFilter: 'blur(10px)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  transition: 'all 0.3s ease',
  boxShadow: isActive ? `0 0 15px ${activeColor}40` : 'none'
});




// 3D Glass Node for Logit Lens
function GlassNode({ position, probability, color, label, actual, layer, posIndex, onHover, isActiveLayer }) {
  const mesh = useRef();

  // Size based on probability (0.0 - 1.0)
  const baseSize = 0.3 + (probability * 0.5);

  useFrame((state) => {
    if (mesh.current) {
      // Gentle pulse for high prob nodes
      if (probability > 0.5) {
        mesh.current.scale.setScalar(baseSize + Math.sin(state.clock.elapsedTime * 2) * 0.05);
      }
    }
  });

  return (
    <group position={position}>
      <mesh
        ref={mesh}
        onPointerOver={(e) => {
          e.stopPropagation();
          onHover({ label, actual, probability, layer, posIndex });
          document.body.style.cursor = 'pointer';
        }}
        onPointerOut={() => {
          onHover(null);
          document.body.style.cursor = 'default';
        }}
        scale={[baseSize, baseSize, baseSize]}
      >
        <sphereGeometry args={[1, 32, 32]} />
        <meshPhysicalMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isActiveLayer ? 2.0 : (probability > 0.5 ? 0.8 : 0.2)}
          metalness={0.1}
          roughness={0.05}
          transmission={0.95} // Glassy
          thickness={1.5}
          transparent
          opacity={0.8}
        />
      </mesh>

      {/* Label for high prob nodes or active layer */}
      {(probability > 0.3 || isActiveLayer) && (
        <Text position={[0, 1.2, 0]} fontSize={0.6} color="white" anchorX="center" anchorY="bottom">
          {label}
        </Text>
      )}
    </group>
  );
}

// Probability to Color mapping (Viridis-like)
const getColor = (prob) => {
  const colors = [
    '#440154', // dark purple (low)
    '#4488ff', // blue
    '#21918c', // teal
    '#ff9f43', // orange
    '#ff4444'  // red (high)
  ];
  const idx = Math.min(Math.floor(prob * (colors.length - 1) * 1.5), colors.length - 1); // Boost index
  return colors[idx];
};

function Visualization({ data, hoveredInfo, setHoveredInfo, activeLayer }) {
  if (!data) return null;

  const { logit_lens, tokens } = data;
  const nLayers = logit_lens.length;
  const seqLen = tokens.length;

  // Calculate highest probability path (for connections)
  const paths = [];
  if (logit_lens.length > 0) {
    for (let pos = 0; pos < seqLen; pos++) {
      const path = [];
      for (let l = 0; l < nLayers; l++) {
        const layerData = logit_lens[l][pos];
        // Find position coordinates
        const x = pos * 2.5; // Spacing
        const z = l * 2.0;
        path.push(new THREE.Vector3(x, 0, z));
      }
      paths.push(path);
    }
  }

  return (
    <>
      <group position={[-seqLen, 0, -nLayers]}> {/* Center roughly */}
        {logit_lens.map((layerData, layerIdx) => (
          layerData.map((posData, posIdx) => (
            <GlassNode
              key={`${layerIdx}-${posIdx}`}
              position={[posIdx * 2.5, 0, layerIdx * 2.0]}
              probability={posData.prob}
              color={getColor(posData.prob)}
              label={posData.token}
              actual={posData.actual_token}
              layer={layerIdx}
              posIndex={posIdx}
              onHover={setHoveredInfo}
              isActiveLayer={layerIdx === activeLayer}
            />
          ))
        ))}

        {/* Draw Connections (Trajectory) */}
        {tokens.map((_, i) => (
          <line key={`path-${i}`}>
            <bufferGeometry setFromPoints={paths[i]} />
            <lineBasicMaterial color="#ffffff" opacity={0.15} transparent linewidth={1} />
          </line>
        ))}

        {/* Axis Labels */}
        {tokens.map((token, i) => (
          <Text
            key={`x-label-${i}`}
            position={[i * 1.2, -0.5, -1]}
            rotation={[-Math.PI / 2, 0, 0]}
            fontSize={0.3}
            color="white"
          >
            {token}
          </Text>
        ))}

        {Array.from({ length: nLayers }).map((_, i) => (
          <Text
            key={`z-label-${i}`}
            position={[-1.5, -0.5, i * 1.2]}
            rotation={[-Math.PI / 2, 0, 0]}
            fontSize={0.3}
            color="gray"
          >
            L{i}
          </Text>
        ))}
      </group>

      {/* Info panel moved to DOM overlay - see bottom-left panel */}
    </>
  );
}

// Flow Particles Component - shows information flow between layers
function FlowParticles({ nLayers, seqLen, isPlaying }) {
  const particlesRef = useRef();
  const [particles, setParticles] = useState([]);

  // Generate particles
  useFrame((state) => {
    if (!isPlaying || !particlesRef.current) return;

    // Generate new particles more frequently (20% chance instead of 5%)
    if (Math.random() < 0.2) {
      const newParticle = {
        id: Math.random(),
        x: (Math.random() - 0.5) * seqLen * 1.2,
        z: 0,
        targetZ: (nLayers - 1) * 1.2,
        progress: 0,
        speed: 0.3 + Math.random() * 0.4
      };
      setParticles(prev => [...prev.slice(-50), newParticle]);
    }

    // Update particle positions
    setParticles(prev => prev.map(p => ({
      ...p,
      progress: Math.min(1, p.progress + 0.008 * p.speed)
    })).filter(p => p.progress < 1));
  });

  if (!isPlaying) return null;

  return (
    <group ref={particlesRef} position={[-seqLen / 2, 4, -nLayers / 2]}>
      {particles.map(p => {
        const currentZ = p.z + (p.targetZ - p.z) * p.progress;
        const opacity = Math.sin(p.progress * Math.PI);

        return (
          <mesh key={p.id} position={[p.x, 0, currentZ]}>
            <sphereGeometry args={[0.15, 16, 16]} />
            <meshStandardMaterial
              color="#00d2ff"
              emissive="#00d2ff"
              emissiveIntensity={3}
              transparent
              opacity={opacity * 0.9}
            />
          </mesh>
        );
      })}
    </group>
  );
}

// Attention Heatmap Component using Canvas
function AttentionHeatmap({ pattern, tokens, headIdx }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !pattern) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const size = pattern.length;
    const cellSize = Math.min(200 / size, 40);

    canvas.width = size * cellSize;
    canvas.height = size * cellSize;

    // Draw heatmap
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const value = pattern[i][j];
        const intensity = Math.floor(value * 255);
        ctx.fillStyle = `rgb(${intensity}, ${Math.floor(intensity * 0.5)}, ${255 - intensity})`;
        ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
      }
    }

    // Draw grid
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= size; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cellSize, 0);
      ctx.lineTo(i * cellSize, size * cellSize);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, i * cellSize);
      ctx.lineTo(size * cellSize, i * cellSize);
      ctx.stroke();
    }
  }, [pattern]);

  return (
    <div style={{ marginBottom: '12px' }}>
      <div style={{ fontSize: '11px', fontWeight: 'bold', marginBottom: '4px', color: '#00d2ff' }}>
        头 {headIdx}
      </div>
      <canvas
        ref={canvasRef}
        style={{
          border: '1px solid #444',
          borderRadius: '4px',
          maxWidth: '100%',
          imageRendering: 'pixelated'
        }}
      />
    </div>
  );
}

// MLP Activation Bar Chart using Canvas
function MLPActivationChart({ distribution }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !distribution) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = 300;
    const height = 100;
    const barCount = Math.min(distribution.length, 100);
    const barWidth = width / barCount;

    canvas.width = width;
    canvas.height = height;

    // Find max for scaling
    const maxVal = Math.max(...distribution.slice(0, barCount));

    // Draw bars
    for (let i = 0; i < barCount; i++) {
      const value = distribution[i];
      const barHeight = (value / maxVal) * height;
      const hue = (value / maxVal) * 120; // 0 (red) to 120 (green)
      ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
      ctx.fillRect(i * barWidth, height - barHeight, barWidth, barHeight);
    }
  }, [distribution]);

  return (
    <div>
      <div style={{ fontSize: '11px', fontWeight: 'bold', marginBottom: '4px', color: '#00d2ff' }}>
        MLP激活分布
      </div>
      <canvas
        ref={canvasRef}
        style={{
          border: '1px solid #444',
          borderRadius: '4px',
          width: '100%'
        }}
      />
    </div>
  );
}

// Global Config Panel Component
// Global Config Panel Component
function GlobalConfigPanel({ visibility, onToggle, onClose, onReset, lang, onSetLang, t }) {
  const getLabelFor = (key) => {
    return t(`panels.${key}`) || key;
  };

  return (
    <SimplePanel
      title={t('panels.globalConfig')}
      onClose={onClose}
      icon={<Settings />}
      style={{
        position: 'absolute', top: 60, left: 20, zIndex: 100,
        minWidth: '220px'
      }}
    >
      {/* Language Switcher */}
      <div style={{ marginBottom: '16px', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '12px' }}>
        <div style={{ fontSize: '12px', color: '#aaa', marginBottom: '8px' }}>{t('common.language')}</div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={() => onSetLang('zh')}
            style={{
              flex: 1, padding: '4px', borderRadius: '4px',
              border: lang === 'zh' ? '1px solid #4488ff' : '1px solid #444',
              background: lang === 'zh' ? 'rgba(68, 136, 255, 0.2)' : 'transparent',
              color: lang === 'zh' ? '#fff' : '#888',
              cursor: 'pointer', fontSize: '12px'
            }}
          >
            中文
          </button>
          <button
            onClick={() => onSetLang('en')}
            style={{
              flex: 1, padding: '4px', borderRadius: '4px',
              border: lang === 'en' ? '1px solid #4488ff' : '1px solid #444',
              background: lang === 'en' ? 'rgba(68, 136, 255, 0.2)' : 'transparent',
              color: lang === 'en' ? '#fff' : '#888',
              cursor: 'pointer', fontSize: '12px'
            }}
          >
            English
          </button>
        </div>
      </div>

      <div style={{ marginBottom: '16px' }}>
        {Object.entries(visibility).map(([key, isVisible]) => (
          <div key={key} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px', fontSize: '13px', alignItems: 'center' }}>
            <span style={{ color: '#ccc' }}>{getLabelFor(key)}</span>
            <button
              onClick={() => onToggle(key)}
              style={{
                background: isVisible ? '#4488ff' : '#333',
                border: 'none', borderRadius: '12px', width: '36px', height: '20px',
                position: 'relative', cursor: 'pointer', transition: 'background 0.2s'
              }}
            >
              <div style={{
                position: 'absolute', top: '2px', left: isVisible ? '18px' : '2px',
                width: '16px', height: '16px', background: '#fff', borderRadius: '50%',
                transition: 'left 0.2s'
              }} />
            </button>
          </div>
        ))}
      </div>

      <button onClick={onReset} style={{
        width: '100%', padding: '8px', backgroundColor: '#333', color: '#fff', border: 'none',
        borderRadius: '4px', cursor: 'pointer', fontSize: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
        transition: 'background 0.2s', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '12px'
      }}>
        <RotateCcw size={12} /> {t('panels.resetLayout')}
      </button>
    </SimplePanel>
  );
}

const ALGO_DOCS = {
  // --- Architecture ---
  'architect': {
    title: 'Transformer 架构 (Architecture)',
    simple: {
      title: 'Transformer 就像一个超级工厂',
      desc: '想象你在读一本书，你的大脑在做两件事：',
      points: [
        '👀 注意力机制 (Attention): 当你读到“它”这个字时，你会回头看前面的句子，找找“它”指代的是“小猫”还是“桌子”。在界面中：这就好比那些连接线，显示了 AI 在关注哪些词。',
        '🧠 记忆网络 (MLP): 这就像个巨大的知识库。读到“巴黎”，你会立刻联想到“法国”、“埃菲尔铁塔”。在界面中：这就好比每一层里面密密麻麻的神经元被激活了。'
      ]
    },
    pro: {
      title: 'Transformer Blocks',
      desc: 'Transformer 由多个堆叠的 Block 组成，每个 Block 包含两个主要子层：',
      points: [
        'Multi-Head Self-Attention (MHSA): 允许模型关注不同位置的 token，捕捉长距离依赖。',
        'Feed-Forward Network (MLP): 逐位置处理信息，通常被认为存储了事实性知识 (Knowledge Storage)。',
        'Residual Connections & LayerNorm: 缓解梯度消失，稳定训练。'
      ],
      formula: 'Block(x) = x + MHSA(LN1(x)) + MLP(LN2(x + MHSA(...)))'
    }
  },
  // --- Circuit ---
  'circuit': {
    title: '回路发现 (Circuit Discovery)',
    simple: {
      title: '寻找 AI 的“电路图”',
      desc: '就像拆开收音机看电路板一样，我们试图找出 AI 大脑里具体是哪几根线在负责“把英语翻译成中文”或者“做加法”。',
      points: [
        '节点 (Node): 就像电路板上的元件（电容、电阻），这里指某个特定的注意力头。',
        '连线 (Edge): 就像导线，显示了信息是如何从一个元件流向另一个元件的。红色线表示促进，蓝色线表示抑制。'
      ]
    },
    pro: {
      title: 'Edge Attribution Patching (EAP)',
      desc: 'EAP 是一种快速定位对特定任务有贡献的子网络（Circuit）的方法。它基于线性近似，无需多次运行模型。',
      points: [
        '原理: 通过计算梯度 (Gradient) 和激活值 (Activation) 的逐元素乘积，估算每条边被切断后对损失函数的影响。',
        '优势: 计算成本低（只需一次前向+反向传播），适合大规模分析。'
      ],
      formula: 'Attribution(e) = ∇_e Loss * Activation(e)'
    }
  },
  // --- Features ---
  'features': {
    title: '稀疏特征 (Sparse Features)',
    simple: {
      title: '破译 AI 的“脑电波”',
      desc: 'AI 内部有成千上万个神经元同时在闪烁，很难看懂。我们用一种特殊的解码器（SAE），把这些乱闪的信号翻译成人类能懂的概念。',
      points: [
        '特征 (Feature): 比如“检测到法语”、“发现代码错误”、“感受到愤怒情绪”。',
        '稀疏性 (Sparsity): 大脑在某一时刻只有少数几个概念是活跃的（比如你现在在想“苹果”，就不会同时想“打篮球”）。'
      ]
    },
    pro: {
      title: 'Sparse Autoencoders (SAE)',
      desc: 'SAE 是一种无监督学习技术，用于将稠密的 MLP 激活分解为稀疏的、可解释的过完备基 (Overcomplete Basis)。',
      points: [
        'Encoder: 将激活 x 映射到高维稀疏特征 f。',
        'Decoder: 尝试从 f 重构原始激活 x。',
        'L1 Penalty: 强制绝大多数特征 f 为 0，确保稀疏性。'
      ],
      formula: 'L = ||x - W_dec(f)||^2 + λ||f||_1, where f = ReLU(W_enc(x) + b)'
    }
  },
  // --- Causal ---
  'causal': {
    title: '因果分析 (Causal Analysis)',
    simple: {
      title: '谁是真正的幕后推手？',
      desc: '为了搞清楚 AI 到底是怎么通过“巴黎”联想到“法国”的，我们像做手术一样，尝试阻断或修改某些神经元的信号，看看结果会不会变。',
      points: [
        '干预 (Intervention): 如果我们把“巴黎”这个信号屏蔽掉，AI 还能说出“法国”吗？如果不能，说明这个信号很关键。',
        '因果链 (Causal Chain): 像侦探一样，一步步追踪信息流动的路径。'
      ]
    },
    pro: {
      title: 'Causal Mediation Analysis',
      desc: '通过干预（Intervention）技术，测量特定组件对模型输出的因果效应。',
      points: [
        'Ablation (消融): 将某组件的输出置零或替换为平均值，观察 Logits 变化。',
        'Activation Patching (激活修补): 将组件在“干净输入”下的激活值替换为“受损输入”下的值，观察能否恢复错误输出，或反之。'
      ],
      formula: 'Do-Calculus: P(Y | do(X=x))'
    }
  },
  // --- Manifold ---
  'manifold': {
    title: '流形几何 (Manifold Geometry)',
    simple: {
      title: '思维的形状',
      desc: '如果把每个词都看作空间里的一个点，那么所有合理的句子就会形成一个特定的形状（流形）。',
      points: [
        '数据云: 看起来像一团乱麻的点阵。',
        '主成分 (PCA): 找出这团乱麻的主要延伸方向（比如长、宽、高），帮我们在 3D 屏幕上画出来。',
        '聚类:意思相近的词（如“猫”、“狗”）会聚在一起。'
      ]
    },
    pro: {
      title: 'Activation Manifold & ID',
      desc: '分析激活向量空间 (Activation Space) 的几何拓扑性质。',
      points: [
        'Intrinsic Dimensionality (ID): 测量数据流形的有效自由度。Transformer 的深层往往表现出低维流形结构（流形坍缩）。',
        'PCA Projection: 将高维激活 (d_model) 投影到 3D 空间以进行可视化。',
        'Trajectory: Token 在层与层之间的演化路径。'
      ],
      formula: 'PCA: minimize ||X - X_k||_F^2'
    }
  },
  // --- Compositional ---
  'compositional': {
    title: '组合泛化 (Compositionality)',
    simple: {
      title: '乐高积木式的思维',
      desc: 'AI 没见过的句子它也能懂，因为它学会了“拼积木”。',
      points: [
        '原子概念: 像乐高积木块（"红色的"、"圆的"、"球"）。',
        '组合规则: 怎么拼在一起（"红色的球" vs "圆的红色"）。',
        '泛化: 只要学会了规则，就能拼出从未见过的形状。'
      ]
    },
    pro: {
      title: 'Compositional Generalization',
      desc: '评估模型将已知组件（原语）组合成新颖结构的能力。',
      points: [
        'Systematicity: 理解句法结构独立于语义内容（如 "John loves Mary" vs "Mary loves John"）。',
        'Subspace Alignment: 检查表示不同属性（如颜色、形状）的子空间是否正交。'
      ]
    }
  },
  // --- TDA ---
  'tda_legacy': {
    title: '拓扑分析 (Topology/TDA)',
    simple: {
      title: '思维地图的“坑洞”',
      desc: '有时候研究 AI 的思维形状还不够，我们还得看看这个形状里有没有“洞”。',
      points: [
        '持久同调 (Persistent Homology): 就像用不同大小的筛子去筛沙子，看看哪些形状是真正稳定的。',
        'Betti 数: 0 维代表有多少个孤立的概念点，1 维代表有多少个环形逻辑。',
        '逻辑回路: 如果一个概念绕了一圈又回来了（比如递归逻辑），拓扑分析就能抓到它。'
      ]
    },
    pro: {
      title: 'Topological Data Analysis (TDA)',
      desc: '利用代数拓扑方法研究高维点云的内在几何结构。',
      points: [
        'Vietoris-Rips Filtration: 构建单纯复形序列。',
        'Persistence Diagram: 记录拓扑特征（孔洞）的出生与消亡。',
        'Betti Numbers (β0, β1): 描述流形的连通分量和环的数量，表征语义特征的复杂度和稳定性。'
      ],
      formula: 'H_k(K) = Z_k(K) / B_k(K)'
    }
  },
  // --- AGI / Fiber / Glass ---
  'agi': {
    title: '神经纤维丛 (Neural Fiber Bundle)',
    simple: {
      title: 'AGI 的数学蓝图',
      desc: '这是我们提出的一个全新理论：大模型不仅仅是在预测下一个词，它实际上是在构建一个复杂的几何结构——纤维丛。',
      points: [
        '底流形 (Base Manifold): 代表逻辑和语法骨架（深蓝色网格）。',
        '纤维 (Fiber): 代表附着在骨架上的丰富语义（红色向量）。',
        '平行移动: 推理过程就是把语义沿着逻辑骨架移动。'
      ]
    },
    pro: {
      title: 'Neural Fiber Bundle Theory (NFB)',
      desc: '将 LLM 的表示空间建模为数学纤维丛 (Fiber Bundle) E -> M。',
      points: [
        'Base Space M: 句法/逻辑流形，捕捉结构信息。',
        'Fiber F: 语义向量空间，捕捉具体内容。',
        'Connection (Transport): 注意力机制充当联络 (Connection)，定义了纤维之间的平行移动 (Parallel Transport)，即推理过程。'
      ],
      formula: 'E = M × F (Locally Trivial)'
    }
  },
  'glass_matrix': {
    title: '玻璃矩阵 (Glass Matrix)',
    simple: {
      title: '透明的大脑',
      desc: '这是纤维丛理论的直观展示。我们把复杂的数学结构变成了一个像玻璃一样透明、有序的矩阵。',
      points: [
        '青色球体: 逻辑节点。',
        '红色短棍: 每一根棍子代表一种含义。',
        '黄色连线: 它们之间的推理关系。'
      ]
    },
    pro: {
      title: 'Glass Matrix Visualization',
      desc: 'NFB 理论的静态结构可视化。',
      points: [
        'Manifold Nodes: 显示拓扑结构 (Topology)。',
        'Vector Fibers: 显示局部切空间 (Tangent Space) 的语义方向。',
        'Geodesic Paths: 显示潜在的推理路径。'
      ]
    }
  },
  'model_generation': {
    title: '3D 模型生成说明 (3D Generation)',
    simple: {
      title: '如何变出 3D 的 AI 思维？',
      desc: 'AI 的思维原本是几千个维度的数字，我们通过数学魔法（降维）把它们变成了你能看到的 3D 形状。',
      points: [
        '降维映射: 就像把地球仪压扁变成地图，我们将几千维的空间投影到我们的 3D 屏幕上。',
        '实时渲染: 每一个点的位置都是根据 AI 此时此刻的激活状态动态计算出来的，不是写死的动画。',
        '几何投影: 通过 LLE 算法，我们尽量保证在 3D 空间里离得近的点，在 AI 的原始脑回路里也是意思相近的。'
      ]
    },
    pro: {
      title: 'Model Generation Logic',
      desc: '基于高维流形投影技术实现的实时 3D 结构渲染系统。',
      points: [
        'Projection Algorithm: 使用 Locally Linear Embedding (LLE) 或主成分分析 (PCA) 实现从 d_model 维到 3 维空间的保结构降维。',
        'Dynamic Remapping: 每一层残差流激活向量通过投影矩阵 W_proj 映射到场景坐标系空间。',
        'Topology Preservation: 通过最小化测地距离损失，确保 3D 可视化拓扑与高维流形拓扑的一致性。'
      ],
      formula: 'x_3d = proj(v_high_dim, method="LLE")'
    }
  },
  'gut_relationship': {
    title: '大统一智能理论 (GUT Mapping)',
    simple: {
      title: '智能的“物理公式”',
      desc: '宇宙有相对论，智能也有自己的大统一理论。我们看到的 3D 结构就是这个理论的具体表现。',
      points: [
        '结构即逻辑: 你看到的蓝色网格（底流形）就是 AGI 的逻辑骨架（就像重力场）。',
        '概念即纤维: 红色的小棍（纤维）就是附着在逻辑上的各种知识，它们遵循几何对称性。',
        '推理即平移: AI 思考的过程，就是把语义在逻辑网上按照特定的轨迹进行“平行移动”。'
      ]
    },
    pro: {
      title: 'Grand Unified Theory of Intelligence (GUT)',
      desc: '建立在微分几何与对称群基础上的通用智能理论架构。',
      points: [
        'Geometric Foundations: AGI 的智能源于高维流形的对称性破缺与守恒律映射。',
        'Connection & Transport: 将注意力机制定义为黎曼联络 (Connection)，将推理定义为在纤维丛上的平行移动 (Parallel Transport)。',
        'Unification: 通过几何拓扑将因果性、组合性、稀疏性统一在同一个纤维丛数学框架下。'
      ],
      formula: 'Intelligence ≡ ∫ Connectivity · Symmetry d(Manifold)'
    }
  },
  'flow_tubes': {
    title: '深度动力学 (Deep Dynamics)',
    simple: {
      title: '思维的过山车',
      desc: '这就好比给 AI 的思考过程拍了一段录像。',
      points: [
        '流管 (Tube): 每一根管子代表一句话的思考轨迹。',
        '颜色: 代表不同的语义类别（比如男性/女性）。',
        '收敛: 不管你开始怎么想，最后的结论往往会汇聚到同一个地方。'
      ]
    },
    pro: {
      title: 'Deep Dynamics & Trajectories',
      desc: '将层间变换视为动力系统 (Dynamical System) 的演化轨迹。',
      points: [
        'Trajectory: h_{l+1} = h_l + f(h_l)，视为离散时间的动力系统。',
        'Attractor: 观察轨迹是否收敛到特定的不动点或极限环。',
        'Flow Tubes: 相似输入的轨迹束。'
      ],
      formula: 'dh/dt = F(h, θ)'
    }
  },
  // --- New AGI Modules ---
  'rpt': {
    title: '传输分析 (RPT Analysis)',
    simple: {
      title: '语义的“搬运工”',
      desc: 'RPT 就像是一个精准的导航系统，它能告诉我们一个概念（比如“皇室”）是如何从一个底座（男人）平移到另一个底座（女人）上的。',
      points: [
        '传输矩阵 R: 一张旋转地图，把 A 的状态变换到 B 的状态。',
        '迁移性: 只要 R 是正交的（不扭曲），说明这个逻辑在全宇宙通用。',
        '平行移动: 像在滑梯上滑行一样，保持姿势不变，只换位置。'
      ]
    },
    pro: {
      title: 'Riemannian Parallel Transport',
      desc: '在黎曼流形上定义切空间的线性同构变换。',
      points: [
        'Orthogonal Matrix: 提取的正交传输矩阵 R 捕捉了纯粹的语义旋转。',
        'Isometry: 验证嵌入空间中不同语义族群的几何等距性。',
        'Error Matrix: 衡量传输后的残差，评估线性假设的有效边界。'
      ],
      formula: 'v_target ≈ R * v_source'
    }
  },
  'curvature': {
    title: '曲率分析 (Curvature)',
    simple: {
      title: '思维的“颠簸程度”',
      desc: '如果思维过程很丝滑，说明它在走直线（平坦空间）；如果突然剧烈闪避，说明它碰到了“大坑”（高曲率）。',
      points: [
        '平坦区: 逻辑非常顺畅，没什么好争议的。',
        '高曲率区: 往往是由于偏见、冲突或极其复杂的逻辑导致流形发生了扭曲。',
        '警示灯: 红色代表这里逻辑很绕，AI 可能在这里产生幻觉或偏见。'
      ]
    },
    pro: {
      title: 'Scalar Curvature Analysis',
      desc: '计算表示流形的局部曲率张量，识别高维空间中的非线性奇点。',
      points: [
        'Deviation: 测量激活向量在受到扰动后的局部偏移率。',
        'Geometric Bias: 偏见和刻板印象往往在几何上体现为极高的局部曲率。',
        'Metric Tensor: 通过探测相邻切空间的变换速率来估算局部黎曼度量。'
      ]
    }
  },
  'debias': {
    title: '几何去偏 (Debiasing)',
    simple: {
      title: '给 AI 做“正骨手术”',
      desc: '既然偏见是一个方向性的扭曲，那我们直接用几何方法把它“掰回来”。',
      points: [
        '几何拦截: 识别偏见的方向（比如性别方向）。',
        '逆变换: 把偏移的语义强制旋转回中置轴。',
        '非概率性: 我们不是在调概率，而是在修复 AI 的底层逻辑形状。'
      ]
    },
    pro: {
      title: 'Geometric Interception Method',
      desc: '利用 RPT 提取的传输矩阵的逆算子（R^T）对残差流实施介入。',
      points: [
        'Decoupling: 解耦偏见成分与核心语义。',
        'Residual Hook: 在 Hook 层面将偏见方向投影并消除。',
        'Validation: 观察去偏后模型输出概率分布的对称化回归。'
      ]
    }
  },
  'topology': {
    title: '全局拓扑 (Global Topology)',
    simple: {
      title: 'AGI 的全景地图',
      desc: '不再只看一句话，而是扫描 AI 大脑里所有的逻辑连接点。',
      points: [
        '全域扫描: 扫描职业、情感、逻辑、亲属等所有领域的几何对齐情况。',
        '大统一模型: 试图构建一个包含所有人类知识逻辑的完整 3D 地图。',
        '稳定性: 观察不同模型（如 GPT-2 vs Qwen）底层的几何拓扑是否一致。'
      ]
    },
    pro: {
      title: 'Systemic Manifold Scanning',
      desc: '自动化的、跨语义场的拓扑结构提取与对齐分析。',
      points: [
        'Field Matrix: 构建语义场到几何块的映射表。',
        'Topological Invariants: 提取不同层级间的同调性质。',
        'Global Consistency: 评估全量知识在几何上的闭合性。'
      ]
    }
  },
  // --- SNN ---
  'snn': {
    title: '脉冲神经网络 (SNN)',
    simple: {
      title: '仿生大脑',
      desc: '模仿生物大脑“放电”的机制。',
      points: [
        '脉冲 (Spike): 神经元只有积攒了足够的电量，才会“哔”地发一次信号。更节能，更像人脑。',
        'STDP: “早起的鸟儿有虫吃”——如果 A 经常在 B 之前叫，A 对 B 的影响就会变大。'
      ]
    },
    pro: {
      title: 'Spiking Neural Networks',
      desc: '第三代神经网络，使用离散脉冲进行通信。',
      points: [
        'LIF Neuron: Leaky Integrate-and-Fire 模型。包含膜电位积分、泄漏和阈值发放。',
        'STDP: Spike-Timing-Dependent Plasticity，基于脉冲时序的无监督学习规则。',
        'Energy Efficiency: 具有极高的理论能效比。'
      ],
      formula: 'τ * dv/dt = -(v - v_rest) + R * I(t)'
    }
  },
  'validity': {
    title: '有效性验证 (Validity)',
    simple: {
      title: '这真的靠谱吗？',
      desc: '我们用各种数学指标来给 AI 的“健康状况”打分。',
      points: [
        '困惑度 (PPL): AI 对自己说的话有多大把握？越低越好。',
        '熵 (Entropy): AI 的思维有多发散？'
      ]
    },
    pro: {
      title: 'Validity Metrics',
      desc: '评估模型表示质量和一致性的定量指标。',
      points: [
        'Perplexity: exp(CrossEntropy)。衡量预测的确定性。',
        'Cluster Validity: Silhouette Score 等，衡量表示空间的聚类质量。',
        'Smoothness: 轨迹的光滑程度。'
      ]
    }
  },
  // --- TDA ---
  'tda': {
    title: '拓扑数据分析 (Topological Data Analysis)',
    simple: {
      title: 'AI 思维的"孔洞"和"连通"',
      desc: '如果把 AI 的思维空间想象成一块橡皮泥捏成的形状，拓扑学就是研究这个形状有多少个洞、有几块碎片的科学。',
      points: [
        '🔵 连通分量 (β₀): 这团橡皮泥是一整块还是碎成了好几块？数字越大，说明 AI 的"概念簇"越分散。',
        '🔴 环/孔洞 (β₁): 形状里有没有像甜甜圈一样的洞？这代表了语义关系中的"循环依赖"，比如 A→B→C→A。',
        '📊 条形码 (Barcode): 每根横条代表一个特征的"寿命"——什么时候出现，什么时候消失。越长的条越稳定、越重要。'
      ]
    },
    pro: {
      title: 'Persistent Homology (持久同调)',
      desc: '通过代数拓扑工具分析激活空间的全局结构，揭示传统几何方法无法捕捉的拓扑不变量。',
      points: [
        'Betti Numbers (贝蒂数): β₀ 计算连通分量数，β₁ 计算 1 维环数，β₂ 计算空腔数。',
        'Persistence Diagram: 记录每个拓扑特征的诞生和消亡时间，持久性高的特征代表鲁棒结构。',
        'Rips Complex: 基于点云距离构建的单纯复形，用于近似流形拓扑。'
      ],
      formula: 'Hₖ(X) = ker(∂ₖ) / im(∂ₖ₊₁), βₖ = dim(Hₖ)'
    }
  },
  // --- FiberNet V2 ---
  'fibernet_v2': {
    title: 'FiberNet V2 (即时学习)',
    simple: {
      title: '思维的“插件系统”',
      desc: '传统的 AI 需要通过长时间的训练才能记住新知识，而 FiberNet V2 就像插拔式硬盘，能让 AI 秒懂。',
      points: [
        '慢逻辑 (Manifold): 负责理解句法和逻辑规则，这是“出厂配置”。',
        '快记忆 (Fast Weights): 直接在“纤维空间”写入新事实，实现即时记忆升级。',
        '解耦: 逻辑和内容是分开的。学会了说话方式（逻辑），就能随时换上各种“知识芯片”。'
      ]
    },
    pro: {
      title: 'FiberNet Architecture',
      desc: '通过解耦底流形 (Base Manifold) 与语义纤维 (Fibers)，实现非梯度更新的单次学习 (One-shot Learning)。',
      points: [
        'Slow Weights: 处理逻辑骨架 $M$，捕获通用的推理模式。',
        'Fast Weights: 直接作用于纤维空间 $F$，通过动态权重注入实现即时介入。',
        'Linear Injection: 相比 RAG，FiberNet 直接在激活层介入，实现更深层的“理解”。'
      ],
      formula: 'y = SlowLogic(x) + \\sum \\alpha_i \\cdot FastContent(k_i)'
    }
  }
};

const GUIDE_SECTION_DEFAULT = {
  pro: {
    goal: '明确该方法想解释什么、能回答什么问题。',
    approach: ['定义任务与样本', '运行分析并提取关键统计量', '结合3D可视化形成可解释结论'],
    model3d: '将高维激活映射到三维空间，颜色/尺寸/轨迹分别表示强度、重要性和动态变化。',
    algorithm: '根据当前方法计算结构信号，再做稳定性检查（跨层、跨样本、跨提示词）。',
    metricRanges: ['强信号：显著高于随机基线', '中信号：接近阈值边界', '弱信号：与随机结果难区分']
  },
  simple: {
    goal: '看懂这个分析到底想回答什么。',
    approach: ['先跑一次分析', '看关键数字', '再看3D图确认是否一致'],
    model3d: '3D图就是把看不见的内部状态画成能直观看懂的形状和颜色。',
    algorithm: '算法负责找规律，图形负责让你快速确认规律是否真实稳定。',
    metricRanges: ['明显更好/更差：结论更可信', '差别不大：先别下结论', '多次重复一致：可信度提高']
  }
};

const GUIDE_ICON_MAP = {
  Settings,
  Brain,
  BarChart2,
  Grid3x3,
  GitBranch,
  Share2,
  Sparkles,
  Target,
  Globe2,
  Layers,
  Network,
  ArrowRightLeft,
  TrendingUp,
  BarChart,
  Globe,
  RefreshCw,
  Scale,
  CheckCircle,
  Activity
};

const GUIDE_STRUCTURED = {
  architect: {
    pro: {
      goal: '理解模型容量与层级结构是否支持后续可解释分析。',
      approach: ['读取模型配置', '确认层数/头数/维度', '评估可分析粒度与成本'],
      model3d: '层深代表计算阶段，节点密度代表表示容量，轨迹代表跨层信息变换。',
      algorithm: '结构解析 + 配置统计，不涉及训练，仅做架构可解释性评估。',
      metricRanges: ['n_layers: 24-80常见', 'n_heads: 8-64常见', '参数规模越大，分析成本越高']
    },
    simple: {
      goal: '先看清这个模型有多大、分几层。',
      approach: ['看层数', '看头数', '看参数量'],
      model3d: '层越深，表示处理步骤越多。',
      algorithm: '先做体检再做分析。',
      metricRanges: ['层数多=表达更强', '参数大=可能更强也更难解释', '头数多=注意力模式更丰富']
    }
  },
  logit_lens: {
    pro: {
      goal: '观察 token 概率在各层的演化路径，定位何时形成最终预测。',
      approach: ['按层解码logits', '跟踪top token概率', '识别概率跃迁层'],
      model3d: 'X=位置，Z=层，节点颜色/大小=概率，连线=跨层演化路径。',
      algorithm: 'Layer-wise unembedding，对每层残差流直接映射到词表概率分布。',
      metricRanges: ['prob∈[0,1]', '平均prob > 0.35通常信息较稳定', '高置信比例(>0.5)越高，结论越明确']
    },
    simple: {
      goal: '看模型是在哪一层“想明白”的。',
      approach: ['看每层最可能词', '找概率突然变高的层', '对比前后层变化'],
      model3d: '点越大越亮，说明模型越确定。',
      algorithm: '每一层都提前“猜答案”，看猜测怎么变化。',
      metricRanges: ['0.5以上通常较有把握', '0.2以下通常不稳定', '连续升高比单点升高更可信']
    }
  },
  glass_matrix: {
    pro: {
      goal: '揭示激活强度在层-位置网格中的几何分布与聚集结构。',
      approach: ['提取层/位置激活', '做几何映射', '分析高响应区域与流向'],
      model3d: '玻璃球体代表激活单元，透明度与发光强度对应响应幅度。',
      algorithm: '激活张量降维投影 + 强度映射渲染（emissive/opacity）。',
      metricRanges: ['激活归一化后常在[0,1]', '高激活占比 10%-30%常见', '层间聚集中心漂移越小越稳定']
    },
    simple: {
      goal: '看哪些位置最“亮”，也就是最重要。',
      approach: ['先看最亮区域', '再看亮点是否跨层连续', '最后对照文本含义'],
      model3d: '亮、红、大通常表示更强激活。',
      algorithm: '把隐藏层信号变成可见“玻璃矩阵”。',
      metricRanges: ['亮点太少可能欠拟合', '亮点太多可能噪声大', '连续亮带通常更有意义']
    }
  },
  flow_tubes: {
    pro: {
      goal: '分析语义向量在层间传播轨迹与流形偏转。',
      approach: ['构建层间向量场', '拟合主流管线', '评估流向一致性'],
      model3d: '管道粗细代表流强，弯曲代表语义转向，颜色代表阶段状态。',
      algorithm: '向量场积分 + 轨迹拟合（streamline/tube rendering）。',
      metricRanges: ['轨迹长度越短通常越直接', '分叉率过高可能表示冲突语义', '跨层方向一致性>0.6通常较稳定']
    },
    simple: {
      goal: '看信息在模型里是怎么“流动”的。',
      approach: ['看主干流', '看有没有异常分叉', '看终点是否收敛'],
      model3d: '像水流一样，粗管代表主通路。',
      algorithm: '把每层变化连成流线。',
      metricRanges: ['主流清晰=结论清晰', '分叉太多=不稳定', '终点收敛=结果可信']
    }
  },
  circuit: {
    pro: {
      goal: '定位对目标输出有因果贡献的子回路。',
      approach: ['clean/corrupted对比', '计算边归因', '阈值筛选并重建子图'],
      model3d: '节点=组件，边=因果贡献，边颜色区分促进/抑制。',
      algorithm: 'Edge Attribution Patching / activation patching。',
      metricRanges: ['|attribution| > 0.1常作强边', '关键边占比5%-20%常见', '跨提示重合率>0.6更稳健']
    },
    simple: {
      goal: '找出真正“起作用”的内部电路。',
      approach: ['先找关键线', '再看这些线是否重复出现', '最后判断是否稳定'],
      model3d: '粗线就是关键因果路径。',
      algorithm: '把可疑线路关掉或替换，看结果怎么变。',
      metricRanges: ['变化大=关键', '变化小=次要', '多次都关键=高置信']
    }
  },
  features: {
    pro: {
      goal: '将稠密激活分解为可解释稀疏特征。',
      approach: ['训练/载入SAE', '抽取top features', '评估重建误差与稀疏度'],
      model3d: '特征点簇显示语义主题，强激活特征在局部形成高密度区域。',
      algorithm: 'Sparse Autoencoder + L1正则。',
      metricRanges: ['reconstruction_error < 0.02优秀', '0.02-0.08可用', '>0.08需谨慎']
    },
    simple: {
      goal: '把“看不懂的神经元闪烁”翻译成可命名特征。',
      approach: ['抽特征', '看最强特征', '检查误差是否够低'],
      model3d: '相近特征会聚在一起。',
      algorithm: '用解码器把复杂信号拆成少量“概念开关”。',
      metricRanges: ['误差越低越可信', '太高说明解释不到位', '稳定重复出现更可信']
    }
  },
  causal: {
    pro: {
      goal: '识别组件对输出的真实因果效应，而非相关性。',
      approach: ['对关键组件干预', '测量输出变化', '估计重要组件比例'],
      model3d: '高因果组件在图中形成核心团簇，颜色强度对应因果贡献。',
      algorithm: 'Intervention / ablation / activation patching。',
      metricRanges: ['重要组件占比>20%常见强因果', '10%-20%中等', '<10%偏弱']
    },
    simple: {
      goal: '验证“谁导致了结果”。',
      approach: ['关掉一个部件', '看结果是否改变', '重复验证'],
      model3d: '最关键组件会在图中最突出。',
      algorithm: '像做实验一样做对照组。',
      metricRanges: ['一关就变=关键', '怎么关都不变=影响小', '重复一致=可信']
    }
  },
  manifold: {
    pro: {
      goal: '刻画表示空间的内在维度与几何结构。',
      approach: ['降维投影', '估计内在维度', '分析轨迹平滑与聚类结构'],
      model3d: '点云形态展示语义几何，轨迹展示token随层演化。',
      algorithm: 'PCA/UMAP/LLE + intrinsic dimensionality estimation。',
      metricRanges: ['participation_ratio常见2-20', '维度突降可能对应语义压缩', '簇间分离更好可解释性更强']
    },
    simple: {
      goal: '看语义在空间里是散的还是成团的。',
      approach: ['看点云', '看轨迹', '看是否分群'],
      model3d: '团块越清晰越容易解释。',
      algorithm: '把高维空间压到3D来看形状。',
      metricRanges: ['分群清晰=结构好', '全糊在一起=难解释', '轨迹平滑=稳定']
    }
  },
  compositional: {
    pro: {
      goal: '评估模型的组合泛化能力。',
      approach: ['构造组合样本', '回归拟合组合关系', '评估泛化误差'],
      model3d: '组合方向在空间中表现为可加性位移向量。',
      algorithm: 'compositional probing / linear decomposition。',
      metricRanges: ['R² > 0.8强', '0.5-0.8中', '<0.5弱']
    },
    simple: {
      goal: '看模型会不会“拼积木式”举一反三。',
      approach: ['给新组合', '看是否仍能理解', '看评分'],
      model3d: '可组合关系在图里像可叠加的位移。',
      algorithm: '检验旧知识能否组合成新能力。',
      metricRanges: ['R²越高越好', '中等说明部分可组合', '低分说明泛化不足']
    }
  },
  fibernet_v2: {
    pro: {
      goal: '评估慢逻辑与快记忆解耦后的即时学习效果。',
      approach: ['固定慢权重', '注入快权重', '测单次学习后性能变化'],
      model3d: '底流形表示逻辑骨架，纤维方向表示快速知识写入。',
      algorithm: 'base manifold + fiber injection。',
      metricRanges: ['写入后收益>5%通常有效', '遗忘率越低越好', '跨任务迁移越高越好']
    },
    simple: {
      goal: '看模型能不能“即学即用”。',
      approach: ['写入新知识', '马上测试', '看是否影响旧知识'],
      model3d: '主干不变，旁路快速更新。',
      algorithm: '把新知识写到纤维空间，不重训主模型。',
      metricRanges: ['新任务提升明显=有效', '旧任务不掉=稳定', '多轮都有效=可靠']
    }
  },
  rpt: {
    pro: {
      goal: '分析表示之间的传输效率与保真度。',
      approach: ['构建层间传输映射', '估计损耗与失真', '识别瓶颈层'],
      model3d: '层间桥接边展示信息通过率与损耗热点。',
      algorithm: 'representation transport metrics / alignment analysis。',
      metricRanges: ['传输效率接近1更好', '失真越低越好', '瓶颈层需重点检查']
    },
    simple: {
      goal: '看信息在层与层之间“传得好不好”。',
      approach: ['看通过率', '看失真', '找堵点'],
      model3d: '哪里变细哪里就是瓶颈。',
      algorithm: '衡量传输过程有没有丢信息。',
      metricRanges: ['通过率高=好', '失真高=差', '连续堵点=结构问题']
    }
  },
  curvature: {
    pro: {
      goal: '用曲率刻画表示流形的弯曲复杂度。',
      approach: ['估计局部几何', '汇总全局曲率', '定位异常弯曲区域'],
      model3d: '颜色梯度显示曲率大小，热点表示几何突变。',
      algorithm: 'discrete curvature estimation on embedding manifold。',
      metricRanges: ['|curvature| < 0.1平缓', '0.1-0.5中等', '>0.5可能存在异常几何']
    },
    simple: {
      goal: '看语义空间有没有“急转弯”。',
      approach: ['看高曲率点', '检查是否集中', '结合语义解释'],
      model3d: '越红越弯，越蓝越平。',
      algorithm: '测每个区域弯曲程度。',
      metricRanges: ['弯太大要警惕', '平滑通常更稳定', '局部极端值需复核']
    }
  },
  tda: {
    pro: {
      goal: '提取表示空间拓扑不变量（连通分量、环等）。',
      approach: ['构建复形', '计算持久同调', '筛选高持久性特征'],
      model3d: '点云与条形码共同展示“连通/孔洞”结构。',
      algorithm: 'Persistent Homology / Rips complex。',
      metricRanges: ['β0越大表示簇越分散', 'β1越大表示环结构越多', '长寿命条形码更可信']
    },
    simple: {
      goal: '看语义空间有几块、有没有“洞”。',
      approach: ['看连通块', '看环数量', '看特征寿命'],
      model3d: '长条特征比短条更重要。',
      algorithm: '拓扑方法找几何方法看不到的结构。',
      metricRanges: ['碎片多=分散', '环多=循环关系强', '寿命长=稳定']
    }
  },
  global_topology: {
    pro: {
      goal: '从全局层面评估语义几何的一致性与闭合性。',
      approach: ['跨语义场采样', '统一拓扑指标', '比较场间一致性'],
      model3d: '多语义场拓扑图并置，观察全局结构同构关系。',
      algorithm: 'field-level topology scanning + invariant matching。',
      metricRanges: ['场间一致性高=全局稳定', '差异大=局部策略化', '闭合性高=迁移潜力强']
    },
    simple: {
      goal: '看整体知识结构是不是一张连贯的大网。',
      approach: ['分场扫描', '全局对比', '找断裂区域'],
      model3d: '如果图形风格相近，说明全局更一致。',
      algorithm: '把各个语义区域放在一起做总体验收。',
      metricRanges: ['一致性高=结构健康', '断裂多=需修复', '闭合好=泛化更稳']
    }
  },
  holonomy: {
    pro: {
      goal: '测量闭环语义路径的几何回旋偏差。',
      approach: ['构造闭环路径', '计算回旋误差', '定位非保守变换区域'],
      model3d: '闭环轨迹偏离起点的距离直接显示holonomy强度。',
      algorithm: 'parallel transport / loop deviation analysis。',
      metricRanges: ['偏差接近0更一致', '小偏差可接受', '大偏差提示表示不稳定']
    },
    simple: {
      goal: '绕一圈回来，看有没有“走形”。',
      approach: ['走闭环', '看回到原点差多少', '比较不同层'],
      model3d: '回不去原点说明有几何扭曲。',
      algorithm: '闭环误差测试。',
      metricRanges: ['误差小=稳定', '误差大=扭曲强', '跨层一致更可信']
    }
  },
  agi: {
    pro: {
      goal: '评估跨任务统一表示与泛化能力的几何基础。',
      approach: ['多任务联合观测', '比较共享子空间', '测一致性与迁移性'],
      model3d: '不同任务轨迹是否共享主流形决定统一智能程度。',
      algorithm: 'multi-task representation alignment。',
      metricRanges: ['共享子空间占比越高越好', '任务间偏移越小越好', '迁移收益越大越好']
    },
    simple: {
      goal: '看模型能否用一套思路解决多种任务。',
      approach: ['多任务对比', '看是否共用结构', '看迁移效果'],
      model3d: '多任务轨迹重叠越多越像“通用智能”。',
      algorithm: '检查不同任务是否复用同一内部结构。',
      metricRanges: ['重叠多=更通用', '重叠少=更专用', '迁移强=更好']
    }
  },
  debias: {
    pro: {
      goal: '识别并削弱表示空间中的偏置方向。',
      approach: ['估计偏置子空间', '做投影去偏', '评估性能-公平权衡'],
      model3d: '偏置方向在空间中表现为系统性位移向量。',
      algorithm: 'subspace projection / counterfactual comparison。',
      metricRanges: ['偏置分数下降越多越好', '主任务性能下降应尽量小', '跨群体差距越小越好']
    },
    simple: {
      goal: '减少模型“先入为主”的偏见。',
      approach: ['找偏见方向', '削弱它', '确认能力不明显下降'],
      model3d: '去偏后不同群体点云分布更均衡。',
      algorithm: '把偏见方向从表示里减掉。',
      metricRanges: ['偏见降得多=好', '准确率掉太多=需权衡', '群体差距小=更公平']
    }
  },
  validity: {
    pro: {
      goal: '量化分析结论是否稳定、可靠、可复现。',
      approach: ['计算PPL/熵/聚类质量', '评估平滑性与一致性', '形成有效性结论'],
      model3d: '有效性高时轨迹更平滑、簇结构更清晰。',
      algorithm: 'validity metrics aggregation。',
      metricRanges: ['PPL越低越好', 'Entropy过高可能不稳定', 'Silhouette越高聚类越清晰']
    },
    simple: {
      goal: '判断结果靠不靠谱。',
      approach: ['看困惑度', '看熵', '看聚类分离'],
      model3d: '好结果通常形状更清晰、更连续。',
      algorithm: '用几组分数做质量验收。',
      metricRanges: ['低困惑度更好', '过高熵需谨慎', '聚类清晰更可信']
    }
  },
  training: {
    pro: {
      goal: '观察训练过程中的表示演化与收敛行为。',
      approach: ['按训练步采样', '追踪关键指标曲线', '识别阶段性拐点'],
      model3d: '时间轴上的轨迹收敛形态反映学习阶段。',
      algorithm: 'trajectory over checkpoints + phase segmentation。',
      metricRanges: ['loss稳定下降为正向', '剧烈震荡提示学习率/数据问题', '后期收敛应趋平缓']
    },
    simple: {
      goal: '看模型是否在“越学越稳”。',
      approach: ['看趋势', '看波动', '看是否收敛'],
      model3d: '轨迹从乱到稳是正常学习过程。',
      algorithm: '把训练过程当成时间演化问题来观察。',
      metricRanges: ['持续下降=好', '反复震荡=风险', '后期平稳=收敛']
    }
  }
};

const formatGuideValue = (value, digits = 4) => {
  if (typeof value !== 'number' || Number.isNaN(value) || !Number.isFinite(value)) return 'N/A';
  return value.toFixed(digits);
};

const buildGuideConclusion = ({ tab, activeTab, analysisResult, topologyResults, data }) => {
  const isDirectDataTab = tab === 'architect' || tab === 'logit_lens' || tab === 'glass_matrix' || tab === 'flow_tubes';
  const result = tab === 'global_topology'
    ? (topologyResults || (tab === activeTab ? analysisResult : null))
    : (tab === activeTab ? analysisResult : null);

  const make = (available, title, lines, metrics = []) => ({ available, title, lines, metrics });

  if (tab === 'architect') {
    if (!data?.model_config) return make(false, '当前结论', ['尚未加载模型配置，请先执行一次 analyze。']);
    return make(true, '当前结论', [
      `模型 ${data.model_config.name} 已加载，可进行分层解释。`,
      `当前配置支持按层、按头、按特征的结构化分析。`
    ], [
      { label: '层数', value: `${data.model_config.n_layers}` },
      { label: '头数', value: `${data.model_config.n_heads}` },
      { label: '参数规模', value: `${formatGuideValue((data.model_config.total_params || 0) / 1e9, 2)}B` }
    ]);
  }

  if (tab === 'logit_lens' || tab === 'glass_matrix' || tab === 'flow_tubes') {
    if (!data?.logit_lens?.length) return make(false, '当前结论', ['尚无token概率轨迹，请先运行 analyze。']);
    const probs = data.logit_lens.flatMap(layer => layer.map(item => item.prob)).filter(v => typeof v === 'number');
    if (!probs.length) return make(false, '当前结论', ['当前结果缺少概率信息。']);
    const avgProb = probs.reduce((a, b) => a + b, 0) / probs.length;
    const highRatio = probs.filter(v => v > 0.5).length / probs.length;
    return make(true, '当前结论', [
      `跨层平均置信度为 ${formatGuideValue(avgProb, 3)}，模型已有可解释的预测趋势。`,
      `高置信节点占比 ${formatGuideValue(highRatio * 100, 1)}%，可用于定位关键层/关键位置。`
    ], [
      { label: '层数', value: `${data.logit_lens.length}` },
      { label: '序列长度', value: `${data.tokens?.length || 0}` },
      { label: '高置信占比', value: `${formatGuideValue(highRatio * 100, 1)}%` }
    ]);
  }

  if (!isDirectDataTab && !result) {
    return make(false, '当前结论', [`当前未运行 ${tab.toUpperCase()} 分析，请切换到对应分析后执行。`]);
  }

  switch (tab) {
    case 'circuit':
      return make(true, '当前结论', [
        `检测到 ${result.nodes?.length || 0} 个候选组件，${result.graph?.edges?.length || 0} 条候选边。`,
        '可优先关注高归因边形成的主子图，并做跨提示词复验。'
      ], [
        { label: '节点数', value: `${result.nodes?.length || 0}` },
        { label: '边数', value: `${result.graph?.edges?.length || 0}` }
      ]);
    case 'features':
      return make(true, '当前结论', [
        `已提取 ${result.top_features?.length || 0} 个高响应特征。`,
        `重建误差 ${formatGuideValue(result.reconstruction_error, 5)}，可据此判断可解释性强弱。`
      ], [
        { label: 'Top Features', value: `${result.top_features?.length || 0}` },
        { label: '重建误差', value: formatGuideValue(result.reconstruction_error, 5) }
      ]);
    case 'causal':
      return make(true, '当前结论', [
        `共评估 ${result.n_components_analyzed || 0} 个组件，其中关键组件 ${result.n_important_components || 0} 个。`,
        '若关键组件占比高，说明输出受少量核心机制主导。'
      ], [
        { label: '评估组件', value: `${result.n_components_analyzed || 0}` },
        { label: '关键组件', value: `${result.n_important_components || 0}` }
      ]);
    case 'manifold':
      return make(true, '当前结论', [
        `估计内在维度（PR）为 ${formatGuideValue(result.intrinsic_dimensionality?.participation_ratio, 3)}。`,
        '维度越低且簇结构越清晰，通常表示语义组织更紧凑。'
      ], [
        { label: 'Participation Ratio', value: formatGuideValue(result.intrinsic_dimensionality?.participation_ratio, 3) }
      ]);
    case 'compositional':
      return make(true, '当前结论', [
        `组合泛化 R² = ${formatGuideValue(result.r2_score, 4)}。`,
        'R²越高，说明模型越能把已学能力组合到新任务。'
      ], [{ label: 'R²', value: formatGuideValue(result.r2_score, 4) }]);
    case 'tda':
      return make(true, '当前结论', [
        `拓扑特征统计：β0候选 ${result.ph_0d?.length || 0}，β1候选 ${result.ph_1d?.length || 0}。`,
        '建议重点关注寿命更长的拓扑特征以减少噪声结论。'
      ], [
        { label: 'β0 / ph_0d', value: `${result.ph_0d?.length || 0}` },
        { label: 'β1 / ph_1d', value: `${result.ph_1d?.length || 0}` }
      ]);
    case 'curvature':
      return make(true, '当前结论', [
        `当前曲率指标为 ${formatGuideValue(result.curvature, 4)}。`,
        '高绝对曲率通常对应语义变化快或局部几何不稳定区域。'
      ], [{ label: 'Curvature', value: formatGuideValue(result.curvature, 4) }]);
    case 'global_topology': {
      const keys = Object.keys(result || {});
      return make(true, '当前结论', [
        `已生成全局拓扑结果，共包含 ${keys.length} 个结果字段。`,
        '可对比不同语义场的一致性与闭合性，形成全局结构结论。'
      ], [{ label: '结果字段数', value: `${keys.length}` }]);
    }
    default: {
      const numeric = Object.entries(result || {})
        .filter(([, v]) => typeof v === 'number' && Number.isFinite(v))
        .slice(0, 5);
      return make(true, '当前结论', [
        '该分析已产出结果，可结合下列关键数值与3D模式综合判读。',
        '建议进行至少两次重复运行，检查结论稳定性。'
      ], numeric.map(([k, v]) => ({ label: k, value: formatGuideValue(v, 4) })));
    }
  }
};

const EvolutionMonitor = ({ data, onStartSleep }) => {
  if (!data) return null;
  return (
    <div style={{
      background: 'rgba(0,210,255,0.05)', padding: '15px', color: '#00ffcc',
      border: '1px solid rgba(0,210,255,0.2)', borderRadius: '12px', marginBottom: '20px',
      fontFamily: 'monospace'
    }}>
      <h3 style={{ margin: '0 0 10px 0', borderBottom: '1px solid rgba(0,210,255,0.2)', fontSize: '14px' }}>EVOLUTION MONITOR</h3>
      <div style={{ marginBottom: '8px', fontSize: '12px' }}>
        STATUS: <span style={{ color: data.is_evolving ? '#ff00ff' : '#00ffcc' }}>
          {data.is_evolving ? 'SLEEPING (EVOLVING)' : 'AWAKE (READY)'}
        </span>
      </div>
      <div style={{ marginBottom: '8px', fontSize: '12px' }}>
        CURVATURE (Ω): {data.curvature?.toFixed(6) || 'N/A'}
      </div>
      <div style={{ marginBottom: '15px', width: '100%', background: 'rgba(255,255,255,0.05)', height: '4px', borderRadius: '2px', overflow: 'hidden' }}>
        <div style={{
          width: `${data.progress}%`, height: '100%', background: '#ff00ff',
          transition: 'width 0.3s ease', boxShadow: '0 0 10px #ff00ff'
        }} />
      </div>
      {!data.is_evolving && (
        <button
          onClick={onStartSleep}
          style={{
            width: '100%', padding: '8px', background: 'transparent',
            border: '1px solid #ff00ff', color: '#ff00ff', cursor: 'pointer',
            fontWeight: 'bold', borderRadius: '6px', fontSize: '11px'
          }}
          onMouseOver={e => e.target.style.background = 'rgba(255,0,255,0.1)'}
          onMouseOut={e => e.target.style.background = 'transparent'}
        >
          ENTER SLEEP CYCLE
        </button>
      )}
    </div>
  );
};

export default function App() {
  const [lang, setLang] = useState('zh');
  const [helpTab, setHelpTab] = useState('outline'); // Selected tab in Help Modal
  const t = (key, params = {}) => {
    const keys = key.split('.');
    let val = locales[lang];
    for (const k of keys) {
      val = val?.[k];
    }
    if (!val) return key;
    if (params) {
      for (const [pKey, pVal] of Object.entries(params)) {
        val = val.replace(`{{${pKey}}}`, pVal);
      }
    }
    return val;
  };

  const [prompt, setPrompt] = useState('The quick brown fox');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [hoveredInfo, setHoveredInfo] = useState(null);
  const [modelConfig, setModelConfig] = useState(null);
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [layerData, setLayerData] = useState(null);
  const [loadingLayerData, setLoadingLayerData] = useState(false);
  const [isAnimationPlaying, setIsAnimationPlaying] = useState(true);
  const [showStructurePanel, setShowStructurePanel] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [helpMode, setHelpMode] = useState('pro'); // 'simple' | 'pro'
  const [generating, setGenerating] = useState(false);
  const [layerNeuronState, setLayerNeuronState] = useState(null);
  const [loadingNeurons, setLoadingNeurons] = useState(false);
  const [layerInfo, setLayerInfo] = useState(null); // For 3D visualization

  // Animation states for layer computation visualization
  const [isAnimating, setIsAnimating] = useState(false);
  const [activeLayer, setActiveLayer] = useState(null);
  const [activeTab, setActiveTab] = useState('glass_matrix');
  const [evolutionData, setEvolutionData] = useState(null);

  useEffect(() => {
    let isUnmounted = false;
    let hasLoggedDisconnected = false;

    const pollEvolutionStatus = async () => {
      try {
        const res = await fetch(`${API_BASE}/nfb/evolution/status`);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const status = await res.json();
        if (!isUnmounted) {
          setEvolutionData(status);
        }
        hasLoggedDisconnected = false;
      } catch {
        // Avoid flooding console when backend is down/restarting.
        if (!hasLoggedDisconnected) {
          console.warn(`Evolution monitor unavailable: ${API_BASE}/nfb/evolution/status`);
          hasLoggedDisconnected = true;
        }
      }
    };

    pollEvolutionStatus();
    const timer = setInterval(pollEvolutionStatus, 1000);

    return () => {
      isUnmounted = true;
      clearInterval(timer);
    };
  }, []);

  const handleStartSleep = () => {
    fetch(`${API_BASE}/nfb/evolution/ricci?iterations=100`, { method: 'POST' })
      .catch(err => console.error("Start evolution error:", err));
  };

  const [computationPhase, setComputationPhase] = useState(null); // 'attention' | 'mlp' | 'output'
  const [activeLayerInfo, setActiveLayerInfo] = useState(null);

  // Auto-analysis state
  const [autoAnalysisResult, setAutoAnalysisResult] = useState(null);
  const [stepAnalysisMode, setStepAnalysisMode] = useState('features'); // 'features', 'circuit', 'causal', 'none'
  const [analysisResult, setAnalysisResult] = useState(null);
  const [structureTab, setStructureTab] = useState('circuit');

  // 操作历史
  const { history, addHistory, clearHistory, restoreHistory } = useOperationHistory();


  // Analysis Forms State (Lifted from StructureAnalysisPanel)
  const [circuitForm, setCircuitForm] = useState({
    clean_prompt: 'The capital of France is Paris',
    corrupted_prompt: 'The capital of France is Berlin',
    threshold: 0.1,
    target_token_pos: -1
  });

  const [featureForm, setFeatureForm] = useState({
    prompt: 'The quick brown fox jumps',
    layer_idx: 6,
    hidden_dim: 1024,
    sparsity_coef: 0.001,
    n_epochs: 30
  });

  const [causalForm, setCausalForm] = useState({
    prompt: 'The quick brown fox',
    target_token_pos: -1,
    importance_threshold: 0.01
  });

  const [manifoldForm, setManifoldForm] = useState({
    prompt: 'The quick brown fox',
    layer_idx: 0
  });

  const [rptForm, setRptForm] = useState({
    source_prompts: ['He is a doctor', 'He is an engineer', 'He works as a pilot'],
    target_prompts: ['She is a doctor', 'She is an engineer', 'She works as a pilot'],
    layer_idx: 6
  });

  // System Type State for Structure Analysis
  const [systemType, setSystemType] = useState('dnn');

  // SNN State
  const [snnState, setSnnState] = useState({
    initialized: false,
    layers: [],
    structure: null, // [NEW] Store 3D structure
    time: 0,
    spikes: {},
    isPlaying: false
  });

  const initializeSNN = async () => {
    try {
      const res = await axios.post(`${API_BASE}/snn/initialize`, {
        layers: {
          "Retina_Shape": 20,
          "Retina_Color": 20,
          "Object_Fiber": 20
        },
        connections: [
          { src: "Retina_Shape", tgt: "Object_Fiber", type: "one_to_one", weight: 0.8 },
          { src: "Retina_Color", tgt: "Object_Fiber", type: "one_to_one", weight: 0.8 }
        ]
      });
      setSnnState(prev => ({
        ...prev,
        initialized: true,
        layers: res.data.layers,
        structure: res.data.structure
      }));

    } catch (err) {
      console.error(err);
      if (err.message === 'Network Error') {
        alert("连接服务器失败。请检查后端服务器 (server.py) 是否正在运行。如果已崩溃，请重启它。");
      } else {
        alert("SNN 初始化失败: " + err.message);
      }
    }
  };

  const injectSNNStimulus = async (layer, patternIdx) => {
    try {
      await axios.post(`${API_BASE}/snn/stimulate`, {
        layer_name: layer,
        pattern_idx: patternIdx,
        intensity: 2.0
      });
      // Immediately step to see effect
      await stepSNN();
    } catch (err) {
      console.error(err);
      if (err.message === 'Network Error') {
        alert("连接服务器失败。请检查后端服务器 (server.py) 是否正在运行。如果已崩溃，请重启它。");
      }
    }
  };

  const stepSNN = async () => {
    try {
      const res = await axios.post(`${API_BASE}/snn/step`, { steps: 5 });
      setSnnState(prev => ({
        ...prev,
        time: res.data.time,
        spikes: res.data.spikes
      }));
    } catch (err) {
      console.error(err);
      if (err.message === 'Network Error') {
        alert("连接服务器失败。请检查后端服务器 (server.py) 是否正在运行。如果已崩溃，请重启它。");
      }
    }
  };

  // SNN Auto-play effect
  useEffect(() => {
    let interval;
    if (snnState.isPlaying && snnState.initialized) {
      interval = setInterval(stepSNN, 200); // 5 steps every 200ms = 25 steps/sec
    }
    return () => clearInterval(interval);
  }, [snnState.isPlaying, snnState.initialized]);

  const [infoPanelTab, setInfoPanelTab] = useState('model'); // 'model' | 'detail'
  const [displayInfo, setDisplayInfo] = useState(null); // Persisted hover info
  const [topologyResults, setTopologyResults] = useState(null); // Global Scan Data

  // Auto-switch Info Panel tab on hover and persist info
  useEffect(() => {
    if (hoveredInfo) {
      setInfoPanelTab('detail');
      setDisplayInfo(hoveredInfo);
    }
  }, [hoveredInfo]);

  // UI Tabs State
  const [inputPanelTab, setInputPanelTab] = useState('dnn'); // 'dnn' | 'snn'

  // Sync Auto Analysis Result (Single Step) to Main Result State
  // This ensures results show up even if StructureAnalysisControls is not mounted (Basic Tab)
  useEffect(() => {
    if (autoAnalysisResult) {
      setAnalysisResult(autoAnalysisResult.data);
      if (autoAnalysisResult.type !== 'none') {
        setStructureTab(autoAnalysisResult.type);
      }
    }
  }, [autoAnalysisResult]);

  // Head Analysis Panel State
  const [headPanel, setHeadPanel] = useState({
    isOpen: false,
    layerIdx: null,
    headIdx: null
  });

  // Global Visibility State
  const [showConfigPanel, setShowConfigPanel] = useState(false);
  const [compForm, setCompForm] = useState({
    layer_idx: 0,
    raw_phrases: "black, cat, black cat\nParis, France, Paris France\nking, man, king",
    phrases: [["black", "cat", "black cat"], ["Paris", "France", "Paris France"], ["king", "man", "king"]]
  });
  const [agiForm, setAgiForm] = useState({
    prompt: "The quick brown fox jumps over the lazy dog."
  });

  const [panelVisibility, setPanelVisibility] = useState({
    inputPanel: true,
    infoPanel: true,
    layersPanel: true,
    structurePanel: true,
    neuronPanel: true,
    headPanel: true,
    agiChatPanel: false,
    motherEnginePanel: false,
  });
  const [isInfoPanelMinimized, setIsInfoPanelMinimized] = useState(false);
  const [isLayersPanelMinimized, setIsLayersPanelMinimized] = useState(false);
  const [showBlueprint, setShowBlueprint] = useState(false);
  const [blueprintInitialTab, setBlueprintInitialTab] = useState('roadmap');

  const togglePanelVisibility = (key) => {
    setPanelVisibility(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const handleHeadClick = (layerIdx, headIdx) => {
    setHeadPanel({
      isOpen: true,
      layerIdx,
      headIdx
    });
  };

  // Reusable draggable hook
  const useDraggable = (storageKey, defaultPosition) => {
    const [position, setPosition] = useState(() => {
      const saved = localStorage.getItem(storageKey);
      return saved ? JSON.parse(saved) : defaultPosition;
    });
    const [isDragging, setIsDragging] = useState(false);
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

    useEffect(() => {
      localStorage.setItem(storageKey, JSON.stringify(position));
    }, [position, storageKey]);

    const handleMouseDown = (e) => {
      setIsDragging(true);
      setDragStart({
        x: e.clientX - position.x,
        y: e.clientY - position.y
      });
    };

    const handleMouseMove = (e) => {
      if (isDragging) {
        setPosition({
          x: e.clientX - dragStart.x,
          y: e.clientY - dragStart.y
        });
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    useEffect(() => {
      if (isDragging) {
        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
        return () => {
          window.removeEventListener('mousemove', handleMouseMove);
          window.removeEventListener('mouseup', handleMouseUp);
        };
      }
    }, [isDragging, dragStart]);

    return { position, setPosition, isDragging, handleMouseDown };
  };

  // Draggable panels
  const structurePanel = useDraggable('structureAnalysisPanel', { x: window.innerWidth - 400, y: 20 });
  const headPanelDrag = useDraggable('headAnalysisPanel', { x: 400, y: 100 });
  const neuronPanel = useDraggable('neuronStatePanel', { x: 20, y: window.innerHeight - 600 });
  const layerInfoPanel = useDraggable('layerInfoPanel', { x: 400, y: window.innerHeight - 450 });
  const layerDetailPanel = useDraggable('layerDetailPanel', { x: window.innerWidth - 850, y: 20 });
  const helpGuidePanel = useDraggable('helpGuidePanel', { x: Math.max(20, window.innerWidth - 960), y: 40 });

  const resetConfiguration = () => {
    // Clear all localStorage
    localStorage.removeItem('structureAnalysisPanel');
    localStorage.removeItem('headAnalysisPanel');
    localStorage.removeItem('neuronStatePanel');
    localStorage.removeItem('layerInfoPanel');
    localStorage.removeItem('layerDetailPanel');
    localStorage.removeItem('helpGuidePanel');

    // Reset panel positions
    structurePanel.setPosition({ x: window.innerWidth - 400, y: 20 });
    headPanelDrag.setPosition({ x: 400, y: 100 });
    neuronPanel.setPosition({ x: 20, y: window.innerHeight - 600 });
    layerInfoPanel.setPosition({ x: 400, y: window.innerHeight - 450 });
    layerDetailPanel.setPosition({ x: window.innerWidth - 850, y: 20 });
    helpGuidePanel.setPosition({ x: Math.max(20, window.innerWidth - 960), y: 40 });

    // Clear states
    setPrompt('');
    setData(null);
    setSelectedLayer(null);
    setLayerNeuronState(null);
    setActiveLayer(null);
    setActiveLayerInfo(null);
    setAutoAnalysisResult(null);

    alert('✅ 配置已重置到初始状态');
  };


  const analyze = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/analyze`, { prompt });
      setData(res.data);
    } catch (err) {
      console.error(err);
      alert('Error analyzing text. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };


  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

  const animateLayerComputation = async () => {
    if (!data?.model_config || !prompt) return;

    const nLayers = data.model_config.n_layers;

    for (let layer = 0; layer < nLayers; layer++) {
      setActiveLayer(layer);
      console.log(`[Animation] Processing layer ${layer}/${nLayers}`);

      // Fetch both layer config and neuron state for the active layer
      try {
        console.log(`[Animation] Fetching data for layer ${layer}...`);
        const [configRes, stateRes] = await Promise.all([
          axios.get(`${API_BASE}/layer_detail/${layer}`),
          axios.post(`${API_BASE}/layer_details`, {
            prompt,
            layer_idx: layer
          })
        ]);

        console.log(`[Animation] Layer ${layer} data received:`, {
          config: configRes.data,
          state: stateRes.data
        });

        setActiveLayerInfo(configRes.data);
        setLayerNeuronState(stateRes.data); // Display attention patterns and MLP stats
      } catch (err) {
        console.error(`[Animation] Error fetching layer ${layer} info:`, err);
        alert(`获取第${layer}层数据时出错: ${err.message}`);
      }

      // Auto-run feature extraction for this layer (every layer)
      if (true) {
        try {
          console.log(`[Animation] Running feature extraction for layer ${layer}...`);
          const featureRes = await axios.post(`${API_BASE}/extract_features`, {
            prompt,
            layer_idx: layer,
            hidden_dim: 512,  // Reduced for speed
            sparsity_coef: 0.001,
            n_epochs: 10  // Reduced for speed
          });

          setAutoAnalysisResult({
            layer: layer,
            type: 'features',
            data: featureRes.data
          });

          console.log(`[Animation] Feature extraction complete for layer ${layer}`);
        } catch (err) {
          console.error(`[Animation] Feature extraction failed for layer ${layer}:`, err);
        }
      }

      // Attention phase
      setComputationPhase('attention');
      await sleep(150);

      // MLP phase
      setComputationPhase('mlp');
      await sleep(120);

      // Output phase
      setComputationPhase('output');
      await sleep(80);
    }

    // Clear animation state
    setActiveLayer(null);
    setComputationPhase(null);
    setActiveLayerInfo(null);
    setLayerNeuronState(null);
    setAutoAnalysisResult(null);
  };


  const generateNext = async () => {
    setGenerating(true);
    setIsAnimating(true);

    try {
      // First, run the layer computation animation
      await animateLayerComputation();

      // Then perform actual generation
      const res = await axios.post(`${API_BASE}/generate_next`, {
        prompt,
        num_tokens: 1,
        temperature: 0.7
      });
      setPrompt(res.data.generated_text);

      // Auto-analyze after generation
      setTimeout(() => {
        analyze();
      }, 100);
    } catch (err) {
      console.error(err);
      alert('Error generating text. Is the backend running?');
    } finally {
      setGenerating(false);
      setIsAnimating(false);
    }
  };

  const stepToNextLayer = async () => {
    if (!data?.model_config || !prompt) {
      alert('请先运行分析！');
      return;
    }

    const nLayers = data.model_config.n_layers;
    const nextLayer = activeLayer === null ? 0 : activeLayer + 1;

    if (nextLayer >= nLayers) {
      alert('已到达最后一层！');
      return;
    }

    setIsAnimating(true);
    setActiveLayer(nextLayer);

    try {
      console.log(`[Step] Processing layer ${nextLayer}/${nLayers}`);

      // Fetch layer config and neuron state
      const [configRes, stateRes] = await Promise.all([
        axios.get(`${API_BASE}/layer_detail/${nextLayer}`),
        axios.post(`${API_BASE}/layer_details`, {
          prompt,
          layer_idx: nextLayer
        })
      ]);

      setActiveLayerInfo(configRes.data);
      setLayerNeuronState(stateRes.data);

      // Auto-run analysis based on selected mode
      if (stepAnalysisMode !== 'none') {
        try {
          console.log(`[Step] Running ${stepAnalysisMode} analysis for layer ${nextLayer}...`);

          let resultData = null;
          let resultType = stepAnalysisMode;

          if (stepAnalysisMode === 'features') {
            const featureRes = await axios.post(`${API_BASE}/extract_features`, {
              prompt,
              layer_idx: nextLayer,
              hidden_dim: 512,
              sparsity_coef: 0.001,
              n_epochs: 10
            });
            resultData = featureRes.data;
          }
          else if (stepAnalysisMode === 'circuit') {
            const circuitRes = await axios.post(`${API_BASE}/discover_circuit`, {
              ...circuitForm, // Use detailed form settings
              target_layer: nextLayer
            });
            resultData = circuitRes.data;
          }
          else if (stepAnalysisMode === 'causal') {
            const causalRes = await axios.post(`${API_BASE}/causal_analysis`, {
              ...causalForm, // Use detailed form settings
              target_layer: nextLayer
            });
            resultData = causalRes.data;
          }
          else if (stepAnalysisMode === 'manifold') {
            const manifoldRes = await axios.post(`${API_BASE}/manifold_analysis`, {
              ...manifoldForm, // Use detailed form settings
              layer_idx: nextLayer
            });
            resultData = manifoldRes.data;
          }

          if (resultData) {
            setAutoAnalysisResult({
              layer: nextLayer,
              type: resultType,
              data: resultData
            });
          }
        } catch (err) {
          console.error(`[Step] Analysis (${stepAnalysisMode}) failed:`, err);
        }
      }

      // Animate computation phases
      setComputationPhase('attention');
      await sleep(150);

      setComputationPhase('mlp');
      await sleep(120);

      setComputationPhase('output');
      await sleep(80);

      // Keep layer visible, just set phase to idle
      setComputationPhase('idle');

    } catch (err) {
      console.error(`[Step] Error:`, err);
      alert('单步执行失败');
    } finally {
      setIsAnimating(false);
    }
  };


  const loadLayerDetails = async (layerIdx) => {
    if (!prompt) return;
    setLoadingNeurons(true);
    try {
      // Fetch both layer config and neuron state in parallel
      const [configRes, stateRes] = await Promise.all([
        axios.get(`${API_BASE}/layer_detail/${layerIdx}`),
        axios.post(`${API_BASE}/layer_details`, {
          prompt,
          layer_idx: layerIdx
        })
      ]);

      setLayerInfo(configRes.data);
      setLayerNeuronState(stateRes.data);
    } catch (err) {
      console.error(err);
      alert('Error loading layer details.');
    } finally {
      setLoadingNeurons(false);
    }
  };

  const rightPanelMaxHeight = 'calc((100vh - 56px) / 2)';
  const helpWindowWidth = Math.min(920, Math.max(320, window.innerWidth - 40));
  const helpWindowHeight = Math.min(Math.floor(window.innerHeight * 0.82), window.innerHeight - 40);
  const helpWindowMaxLeft = Math.max(10, window.innerWidth - helpWindowWidth - 10);
  const helpWindowMaxTop = Math.max(10, window.innerHeight - helpWindowHeight - 10);
  const helpWindowLeft = Math.max(10, Math.min(helpGuidePanel.position.x, helpWindowMaxLeft));
  const helpWindowTop = Math.max(10, Math.min(helpGuidePanel.position.y, helpWindowMaxTop));

  const structureTabUI = {
    circuit: { name: '回路发现', category: 'graph', focus: '关注关键节点与边的因果通路' },
    features: { name: '稀疏特征', category: 'feature', focus: '关注特征数量与重构误差' },
    causal: { name: '因果分析', category: 'graph', focus: '关注关键组件占比与干预效果' },
    manifold: { name: '流形几何', category: 'geometry', focus: '关注内在维度与轨迹分布' },
    compositional: { name: '组合泛化', category: 'feature', focus: '关注组合关系与R²得分' },
    tda: { name: '拓扑分析', category: 'topology', focus: '关注连通分量与环结构' },
    agi: { name: '神经纤维丛', category: 'system', focus: '关注层间传输与纤维结构' },
    rpt: { name: '传输分析', category: 'geometry', focus: '关注传输路径与几何偏移' },
    curvature: { name: '曲率分析', category: 'geometry', focus: '关注曲率热点与异常区域' },
    glass_matrix: { name: '玻璃矩阵', category: 'observation', focus: '关注激活强度分布与亮点聚集' },
    flow_tubes: { name: '信息流', category: 'observation', focus: '关注语义流动轨迹与分叉' },
    global_topology: { name: '全局拓扑', category: 'topology', focus: '关注语义场之间的一致性' },
    fibernet_v2: { name: 'FiberNet V2', category: 'system', focus: '关注即时学习与快慢权重协作' },
    holonomy: { name: '全纯扫描', category: 'topology', focus: '关注闭环偏差与几何扭转' },
    debias: { name: '几何去偏', category: 'system', focus: '关注偏置方向与去偏效果' },
    validity: { name: '有效性检验', category: 'system', focus: '关注指标稳定性与可复现性' },
    training: { name: '训练动力学', category: 'system', focus: '关注训练阶段变化与收敛趋势' }
  };
  const currentStructureUI = structureTabUI[structureTab] || { name: structureTab, category: 'analysis', focus: '关注当前分析结果与关键指标' };
  const isObservationMode = currentStructureUI.category === 'observation';

  const probValues = data?.logit_lens
    ? data.logit_lens.flatMap(layer => layer.map(item => item.prob)).filter(v => typeof v === 'number')
    : [];
  const avgProb = probValues.length ? probValues.reduce((sum, v) => sum + v, 0) / probValues.length : null;
  const highProbRatio = probValues.length ? probValues.filter(v => v > 0.5).length / probValues.length : null;

  const operationMetrics = (() => {
    switch (structureTab) {
      case 'features':
        return [
          { label: '特征数', value: `${analysisResult?.top_features?.length || 0}`, color: COLORS.primary },
          { label: '重构误差', value: analysisResult?.reconstruction_error?.toFixed?.(5) || '-', color: COLORS.warning },
          { label: '当前层', value: selectedLayer !== null ? `L${selectedLayer}` : '-', color: COLORS.success }
        ];
      case 'circuit':
      case 'causal':
        return [
          { label: '节点/组件', value: `${analysisResult?.nodes?.length || analysisResult?.n_components_analyzed || 0}`, color: COLORS.primary },
          { label: '边/关键', value: `${analysisResult?.graph?.edges?.length || analysisResult?.n_important_components || 0}`, color: COLORS.warning },
          { label: '历史', value: `${history.length}条`, color: COLORS.purple }
        ];
      case 'manifold':
      case 'rpt':
      case 'curvature':
        return [
          { label: '当前层', value: selectedLayer !== null ? `L${selectedLayer}` : '-', color: COLORS.primary },
          { label: '几何指标', value: analysisResult?.curvature?.toFixed?.(4) || analysisResult?.intrinsic_dimensionality?.participation_ratio?.toFixed?.(2) || '-', color: COLORS.warning },
          { label: '状态', value: loading ? '计算中...' : '就绪', color: loading ? COLORS.warning : COLORS.success }
        ];
      case 'tda':
      case 'global_topology':
      case 'holonomy':
        return [
          { label: 'β0', value: `${analysisResult?.ph_0d?.length || 0}`, color: COLORS.primary },
          { label: 'β1', value: `${analysisResult?.ph_1d?.length || 0}`, color: COLORS.warning },
          { label: '历史', value: `${history.length}条`, color: COLORS.purple }
        ];
      case 'glass_matrix':
      case 'flow_tubes':
        return [
          { label: '平均概率', value: avgProb !== null ? `${(avgProb * 100).toFixed(1)}%` : '-', color: COLORS.primary },
          { label: '高置信占比', value: highProbRatio !== null ? `${(highProbRatio * 100).toFixed(1)}%` : '-', color: COLORS.warning },
          { label: '当前层', value: activeLayer !== null ? `L${activeLayer}` : '-', color: COLORS.success }
        ];
      default:
        return [
          { label: '当前层', value: selectedLayer !== null ? `L${selectedLayer}` : '-', color: COLORS.primary },
          { label: '计算状态', value: loading ? '计算中...' : '就绪', color: loading ? COLORS.warning : COLORS.success },
          { label: '历史', value: `${history.length}条`, color: COLORS.purple }
        ];
    }
  })();

  const structureGuideItems = STRUCTURE_TABS_V2.groups.flatMap((group, groupIdx) => ([
    ...(groupIdx === 0 ? [] : [{ type: 'sep' }]),
    ...group.items.map(item => ({
      id: item.id,
      label: item.label,
      iconName: item.icon
    }))
  ]));

  const guideMenuItems = [
    { id: 'outline', label: '大纲 (Overview)', iconName: 'Settings' },
    { type: 'sep' },
    { id: 'architect', label: '模型架构 (Architecture)', iconName: 'Settings' },
    { type: 'sep' },
    ...structureGuideItems
  ];

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#050505', color: 'white' }}>

      {/* Global Settings Button */}
      <button
        onClick={() => setShowConfigPanel(!showConfigPanel)}
        style={{
          position: 'absolute', top: 20, left: 20, zIndex: 101, // Higher than panels
          background: showConfigPanel ? '#4488ff' : 'rgba(20, 20, 25, 0.8)',
          border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: '8px',
          padding: '8px',
          cursor: 'pointer',
          color: 'white',
          backdropFilter: 'blur(10px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center'
        }}
        title="界面配置"
      >
        <Settings size={20} />
      </button>

      {/* AGI Chat Terminal Button */}
      <button
        onClick={() => togglePanelVisibility('agiChatPanel')}
        style={{ ...navButtonStyle(panelVisibility.agiChatPanel, '#10b981'), left: 120 }}
        title="AGI Chat Terminal: O(1) 代数引擎交互"
      >
        <Bot size={20} />
      </button>

      {/* Mother Engine Button */}
      <button
        onClick={() => togglePanelVisibility('motherEnginePanel')}
        style={{ ...navButtonStyle(panelVisibility.motherEnginePanel, '#ff69b4'), left: 170 }} // Adjust left position
        title="Mother Engine (物理大模型引擎)"
      >
        <Zap size={20} />
      </button>

      {/* Project Genesis Blueprint Button - Strategic Roadmap */}
      <button
        onClick={() => {
          setBlueprintInitialTab('roadmap');
          setShowBlueprint(true);
        }}
        style={{
          position: 'absolute', top: 20, left: 70, zIndex: 101,
          background: showBlueprint ? '#4488ff' : 'rgba(20, 20, 25, 0.8)',
          border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: '8px',
          padding: '8px',
          cursor: 'pointer',
          color: '#00d2ff',
          backdropFilter: 'blur(10px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: '0 0 10px rgba(0, 210, 255, 0.3)'
        }}
        title="Project Genesis: 战略层级路线图"
      >
        <Brain size={20} />
      </button>

      {/* Global Config Panel */}

      {/* Global Config Panel */}
      {showConfigPanel && (
        <GlobalConfigPanel
          visibility={panelVisibility}
          onToggle={togglePanelVisibility}
          onClose={() => setShowConfigPanel(false)}
          onReset={() => {
            resetConfiguration();
            setShowConfigPanel(false);
          }}
          lang={lang}
          onSetLang={setLang}
          t={t}
        />
      )}

      {/* ==================== 左上: 控制面板 ==================== */}
      {panelVisibility.inputPanel && (
        <SimplePanel
          title="控制面板"
          style={{
            position: 'absolute', top: 60, left: 20, zIndex: 10,
            width: '360px', maxHeight: '85vh',
            display: 'flex', flexDirection: 'column'
          }}
          actions={
            <div style={{ display: 'flex', gap: '4px' }}>
              {INPUT_PANEL_TABS.map(tab => (
                <button
                  key={tab.id}
                  onClick={() => { setInputPanelTab(tab.id); setSystemType(tab.id); }}
                  style={{
                    padding: '6px 10px',
                    background: inputPanelTab === tab.id ? tab.color : 'transparent',
                    border: inputPanelTab === tab.id ? 'none' : '1px solid rgba(255,255,255,0.2)',
                    borderRadius: '4px',
                    color: inputPanelTab === tab.id ? '#fff' : '#888',
                    cursor: 'pointer',
                    fontSize: '11px',
                    fontWeight: '600',
                    transition: 'all 0.2s'
                  }}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          }
        >

          {/* Content Container with Scroll */}
          <div style={{ flex: 1, overflowY: 'auto', paddingRight: '4px' }}>

            {/* DNN Content: Generation + Structure Analysis */}
            {inputPanelTab === 'dnn' && (
              <div className="animate-fade-in">
                {/* Generation Section */}
                <div style={{ background: 'rgba(255,255,255,0.03)', padding: '12px', borderRadius: '8px', marginBottom: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <div style={{ fontSize: '12px', color: '#aaa', marginBottom: '8px', fontWeight: 'bold', display: 'flex', justifyContent: 'space-between' }}>
                    <span>文本生成与提示词</span>
                    {generating && <span style={{ color: '#5ec962' }}>Generating...</span>}
                  </div>

                  <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="输入提示词..."
                    rows={3}
                    style={{
                      width: '100%', background: '#1a1a1f', border: '1px solid #333',
                      color: 'white', padding: '10px', borderRadius: '6px', outline: 'none',
                      resize: 'vertical', fontSize: '13px', fontFamily: 'sans-serif'
                    }}
                  />

                  <div style={{ display: 'flex', gap: '8px', marginTop: '10px' }}>
                    <button
                      onClick={analyze}
                      disabled={loading || !prompt}
                      style={{
                        flex: 1, background: '#333', border: '1px solid #444', color: 'white',
                        padding: '8px', borderRadius: '6px', cursor: 'pointer',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                        fontSize: '12px'
                      }}
                      title="仅分析当前提示词"
                    >
                      {loading ? <Loader2 className="animate-spin" size={14} /> : <Search size={14} />} 分析
                    </button>

                    <button
                      onClick={generateNext}
                      disabled={generating || !prompt}
                      style={{
                        flex: 2,
                        background: generating ? '#888' : 'linear-gradient(45deg, #5ec962, #96c93d)',
                        border: 'none',
                        color: 'white',
                        padding: '8px',
                        borderRadius: '6px',
                        cursor: generating || !prompt ? 'not-allowed' : 'pointer',
                        fontSize: '12px',
                        fontWeight: '600',
                        opacity: generating || !prompt ? 0.7 : 1,
                        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px'
                      }}
                    >
                      {generating ? '生成中...' : 'Generate Next Token'}
                    </button>
                  </div>
                </div>

                {/* Structure Analysis Section */}
                <div style={{ marginBottom: '10px' }}>
                  {/* Pass systemType='dnn' expressly */}
                  <StructureAnalysisControls
                    autoResult={autoAnalysisResult}
                    systemType={systemType}
                    setSystemType={setSystemType}
                    circuitForm={circuitForm} setCircuitForm={setCircuitForm}
                    featureForm={featureForm} setFeatureForm={setFeatureForm}
                    causalForm={causalForm} setCausalForm={setCausalForm}
                    manifoldForm={manifoldForm} setManifoldForm={setManifoldForm}
                    compForm={compForm} setCompForm={setCompForm}
                    agiForm={agiForm} setAgiForm={setAgiForm}
                    rptForm={rptForm} setRptForm={setRptForm}
                    topologyResults={topologyResults}
                    setTopologyResults={setTopologyResults}
                    onResultUpdate={setAnalysisResult}
                    activeTab={structureTab}
                    setActiveTab={setStructureTab}
                    t={t}
                    // SNN Props
                    snnState={snnState}
                    onInitializeSNN={initializeSNN}
                    onToggleSNNPlay={() => setSnnState(s => ({ ...s, isPlaying: !s.isPlaying }))}
                    onStepSNN={stepSNN}
                    onInjectStimulus={injectSNNStimulus}
                    containerStyle={{
                      background: 'transparent',
                      borderLeft: 'none',
                      backdropFilter: 'none',
                      padding: 0
                    }}
                  />
                </div>

                {/* Step Execution Controls */}
                <div style={{ marginTop: '12px', padding: '12px', background: 'rgba(0,0,0,0.2)', borderRadius: '8px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <span style={{ fontSize: '11px', color: '#aaa', fontWeight: 'bold' }}>单步调试 (Step-by-Step)</span>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px', color: '#888', cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={stepAnalysisMode !== 'none'}
                        onChange={(e) => setStepAnalysisMode(e.target.checked ? structureTab : 'none')}
                        style={{ accentColor: '#4ecdc4' }}
                      />
                      启用分析
                    </label>
                  </div>

                  <button
                    onClick={stepToNextLayer}
                    disabled={isAnimating || !data}
                    style={{
                      width: '100%',
                      background: isAnimating || !data ? '#444' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                      border: 'none',
                      color: 'white',
                      padding: '8px',
                      borderRadius: '6px',
                      cursor: isAnimating || !data ? 'not-allowed' : 'pointer',
                      fontSize: '12px',
                      display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                      opacity: isAnimating || !data ? 0.6 : 1
                    }}
                  >
                    {isAnimating ? <Loader2 className="animate-spin" size={14} /> : '▶️'}
                    执行单层步进 {activeLayer !== null ? `(当前: L${activeLayer})` : '(从 L0 开始)'}
                  </button>
                </div>
              </div>
            )}

            {/* SNN Content */}
            {inputPanelTab === 'snn' && (
              <div className="animate-fade-in">
                <div style={{ padding: '12px', background: 'rgba(78, 205, 196, 0.1)', borderRadius: '8px', border: '1px solid rgba(78, 205, 196, 0.2)', marginBottom: '16px' }}>
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'start' }}>
                    <Brain size={16} color="#4ecdc4" />
                    <div>
                      <h4 style={{ margin: '0 0 4px 0', fontSize: '13px', color: '#4ecdc4' }}>NeuroFiber SNN 仿真</h4>
                      <p style={{ fontSize: '11px', color: '#bfd', margin: 0, lineHeight: '1.4' }}>
                        探索基于神经纤维丛理论的脉冲神经网络动力学。
                      </p>
                    </div>
                  </div>
                </div>

                {/* Pass systemType='snn' expressly */}
                <StructureAnalysisControls
                  autoResult={autoAnalysisResult}
                  systemType="snn"
                  setSystemType={setSystemType}
                  circuitForm={circuitForm} setCircuitForm={setCircuitForm}
                  featureForm={featureForm} setFeatureForm={setFeatureForm}
                  causalForm={causalForm} setCausalForm={setCausalForm}
                  manifoldForm={manifoldForm} setManifoldForm={setManifoldForm}
                  compForm={compForm} setCompForm={setCompForm}
                  agiForm={agiForm} setAgiForm={setAgiForm}
                  rptForm={rptForm} setRptForm={setRptForm}
                  topologyResults={topologyResults}
                  setTopologyResults={setTopologyResults}
                  onResultUpdate={setAnalysisResult}
                  activeTab={structureTab}
                  setActiveTab={setStructureTab}
                  t={t}
                  // SNN Props
                  snnState={snnState}
                  onInitializeSNN={initializeSNN}
                  onToggleSNNPlay={() => setSnnState(s => ({ ...s, isPlaying: !s.isPlaying }))}
                  onStepSNN={stepSNN}
                  onInjectStimulus={injectSNNStimulus}
                  containerStyle={{
                    background: 'transparent',
                    borderLeft: 'none',
                    backdropFilter: 'none',
                    padding: 0
                  }}
                />
              </div>
            )}

            {/* FiberNet Lab Content */}
            {inputPanelTab === 'fibernet' && (
              <div className="animate-fade-in" style={{ height: '100%' }}>
                <FiberNetV2Demo t={t} />
              </div>
            )}

          </div>
        </SimplePanel>
      )}

      {/* Bottom-left Info Panel */}
      {/* Model Info Panel (Top-Right) */}
      {panelVisibility.infoPanel && (
        <SimplePanel
          title={t('panels.modelInfo')}
          style={{
            position: 'absolute', top: 20, right: 20, zIndex: 100,
            width: '360px',
            maxHeight: isInfoPanelMinimized ? 'none' : rightPanelMaxHeight,
            display: 'flex', flexDirection: 'column',
            overflow: 'hidden',
            userSelect: 'text', // Explicitly allow text selection
            cursor: 'auto'
          }}
          headerStyle={{ marginBottom: '0', cursor: 'grab' }}
          actions={
            <>
              <button
                onClick={() => { setHelpTab('outline'); setShowHelp(true); }}
                style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: '#888', padding: '4px', display: 'flex', transition: 'color 0.2s' }}
                onMouseOver={(e) => e.currentTarget.style.color = '#fff'}
                onMouseOut={(e) => e.currentTarget.style.color = '#888'}
                title="算法原理说明"
              >
                <HelpCircle size={16} />
              </button>
              <button
                onClick={() => setIsInfoPanelMinimized(prev => !prev)}
                style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: '#888', padding: '4px', display: 'flex', transition: 'color 0.2s' }}
                onMouseOver={(e) => e.currentTarget.style.color = '#fff'}
                onMouseOut={(e) => e.currentTarget.style.color = '#888'}
                title={isInfoPanelMinimized ? 'Maximize panel' : 'Minimize panel'}
              >
                {isInfoPanelMinimized ? <Maximize2 size={16} /> : <Minimize2 size={16} />}
              </button>
            </>
          }
        >
          {!isInfoPanelMinimized && (
            <div style={{ padding: '0', height: '100%', display: 'flex', flexDirection: 'column' }}>

              {/* SECTION 1: Model / System Information */}
              <div style={{ flex: '0 0 auto', marginBottom: '12px' }}>
                <div style={{ fontSize: '11px', fontWeight: 'bold', color: '#888', marginBottom: '8px', textTransform: 'uppercase' }}>
                  {systemType === 'snn' ? 'SNN 网络状态' : '模型配置'}
                </div>

                <EvolutionMonitor data={evolutionData} onStartSleep={handleStartSleep} />

                {systemType === 'snn' ? (
                  /* SNN System Info */
                  <div style={{ fontSize: '12px', lineHeight: '1.6', background: 'rgba(255,255,255,0.03)', padding: '8px', borderRadius: '6px' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: '100px 1fr', gap: '4px', color: '#aaa' }}>
                      <span>状态:</span>
                      <span style={{ color: snnState.initialized ? '#4ecdc4' : '#666', fontWeight: 'bold' }}>
                        {snnState.initialized ? (snnState.isPlaying ? '运行中' : '就绪') : '未初始化'}
                      </span>

                      <span>仿真时间:</span>
                      <span style={{ color: '#fff' }}>{snnState.time.toFixed(1)} ms</span>

                      <span>神经元数:</span>
                      <span style={{ color: '#fff' }}>{snnState.structure?.neurons?.length || 0}</span>
                    </div>
                  </div>
                ) : (
                  /* DNN Model Info */
                  data?.model_config ? (
                    <div style={{ fontSize: '12px', lineHeight: '1.6', background: 'rgba(255,255,255,0.03)', padding: '8px', borderRadius: '6px' }}>
                      <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr', gap: '4px', color: '#aaa' }}>
                        <span>架构:</span>
                        <span style={{ color: '#fff', fontWeight: 'bold' }}>{data.model_config.name}</span>

                        <span>层数:</span>
                        <span style={{ color: '#fff' }}>{data.model_config.n_layers}</span>

                        <span>模型维度:</span>
                        <span style={{ color: '#fff' }}>{data.model_config.d_model} (H: {data.model_config.n_heads})</span>

                        <span>参数量:</span>
                        <span style={{ color: '#fff' }}>{(data.model_config.total_params / 1e9).toFixed(2)}B</span>
                      </div>
                    </div>
                  ) : (
                    <div style={{ color: '#666', fontStyle: 'italic', fontSize: '12px', padding: '8px' }}>未加载模型</div>
                  )
                )}
              </div>

              {/* Divider */}
              <div style={{ height: '1px', background: 'rgba(255,255,255,0.1)', marginBottom: '12px' }} />

              {/* SECTION 2: Analysis / Detail Information */}
              <div style={{ flex: 1, overflowY: 'auto' }}>
                <div style={{ fontSize: '11px', fontWeight: 'bold', color: '#888', marginBottom: '8px', textTransform: 'uppercase' }}>
                  {systemType === 'snn' ? '实时动态' : `${currentStructureUI.name}详情`}
                </div>

                {systemType === 'snn' ? (
                  /* SNN Live Details */
                  <div style={{ fontSize: '12px' }}>
                    <div style={{ marginBottom: '8px', color: '#aaa', fontSize: '11px' }}>
                      实时脉冲活动 (STDP 已启用)
                    </div>
                    {/* Compact Spike Visualization */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                      {snnState.layers.map(layer => {
                        const isActive = snnState.spikes[layer] && snnState.spikes[layer].length > 0;
                        return (
                          <div key={layer} style={{
                            padding: '6px',
                            borderRadius: '4px',
                            background: isActive ? 'rgba(255,159,67,0.15)' : 'transparent',
                            border: isActive ? '1px solid rgba(255,159,67,0.3)' : '1px solid rgba(255,255,255,0.05)',
                            display: 'flex', justifyContent: 'space-between', alignItems: 'center'
                          }}>
                            <span style={{ color: isActive ? '#fff' : '#888', fontSize: '11px' }}>{layer}</span>
                            {isActive && <span style={{ fontSize: '9px', color: '#ff9f43', fontWeight: 'bold' }}>活跃</span>}
                          </div>
                        );
                      })}
                    </div>
                    <div style={{ marginTop: '12px', fontSize: '11px', color: '#666' }}>
                      使用左侧面板控制注入刺激信号。
                    </div>
                  </div>
                ) : (
                  /* DNN Analysis Details - Handles both Hover and Active Analysis */
                  (
                    <div>
                      <div style={{
                        marginBottom: '12px',
                        background: 'rgba(255,255,255,0.03)',
                        border: '1px solid rgba(255,255,255,0.08)',
                        borderRadius: '6px',
                        padding: '8px',
                        fontSize: '11px',
                        color: '#bbb'
                      }}>
                        <div style={{ color: '#fff', fontWeight: '600', marginBottom: '4px' }}>
                          当前模式: {currentStructureUI.name}
                        </div>
                        <div>分析重点: {currentStructureUI.focus}</div>
                      </div>
                      {/* 2A. Hover/Selected Info (Highest Priority for immediate feedback) */}
                      {(displayInfo || hoveredInfo) && (
                        <div style={{ marginBottom: '16px', background: 'rgba(0,0,0,0.2)', padding: '10px', borderRadius: '6px', borderLeft: '3px solid #00d2ff' }}>
                          <div style={{ fontSize: '11px', fontWeight: 'bold', color: '#00d2ff', marginBottom: '6px' }}>
                            选中信息
                          </div>
                          <div style={{ fontSize: '12px', lineHeight: '1.5', color: '#ddd' }}>
                            {(hoveredInfo || displayInfo).type === 'feature' ? (
                              <div>
                                <div>特证 <strong>#{(hoveredInfo || displayInfo).featureId}</strong></div>
                                <div>激活值: <span style={{ color: '#4ecdc4' }}>{(hoveredInfo || displayInfo).activation?.toFixed(4)}</span></div>
                                <div style={{ fontSize: '10px', color: '#aaa', marginTop: '4px' }}>
                                  潜在表示单元。
                                </div>
                              </div>
                            ) : (hoveredInfo || displayInfo).type === 'manifold' ? (
                              <div>
                                <div>数据点: {(hoveredInfo || displayInfo).index}</div>
                                <div>PC1/2/3: {(hoveredInfo || displayInfo).pc1?.toFixed(2)}, {(hoveredInfo || displayInfo).pc2?.toFixed(2)}, {(hoveredInfo || displayInfo).pc3?.toFixed(2)}</div>
                              </div>
                            ) : (
                              <div>
                                <div>词元: <strong>"{(hoveredInfo || displayInfo).label}"</strong></div>
                                <div>概率: <span style={{ color: getColor((hoveredInfo || displayInfo).probability) }}>{((hoveredInfo || displayInfo).probability * 100).toFixed(1)}%</span></div>
                                {(hoveredInfo || displayInfo).actual && <div>实际: "{(hoveredInfo || displayInfo).actual}"</div>}
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {/* 2B. Analysis Method Summary (Context) */}
                      {analysisResult && !hoveredInfo && (
                        <div style={{ fontSize: '12px', color: '#aaa' }}>
                          <div style={{ color: '#fff', marginBottom: '4px' }}>
                            当前分析方法: {structureTab.toUpperCase()}
                          </div>

                          {structureTab === 'circuit' && (
                            <div>
                              在因果图中发现 {analysisResult.nodes?.length} 个节点和 {analysisResult.graph?.edges?.length} 条边。
                            </div>
                          )}
                          {structureTab === 'features' && (
                            <div>
                              从第 {featureForm.layer_idx} 层提取了 {analysisResult.top_features?.length} 个稀疏特征。
                              <br />重构误差: {analysisResult.reconstruction_error?.toFixed(5)}
                            </div>
                          )}
                          {structureTab === 'causal' && (
                            <div>
                              分析了 {analysisResult.n_components_analyzed} 个组件，
                              发现 {analysisResult.n_important_components} 个关键组件。
                            </div>
                          )}
                          {structureTab === 'manifold' && (
                            <div>
                              内在维度: {analysisResult.intrinsic_dimensionality?.participation_ratio?.toFixed(2)}
                              <br />分析层数: {manifoldForm.layer_idx}
                            </div>
                          )}
                          {structureTab === 'compositional' && (
                            <div>
                              组合泛化 R² 分数: {analysisResult.r2_score?.toFixed(4)}
                            </div>
                          )}
                          {structureTab === 'tda' && (
                            <div>
                              0维连通分量: {analysisResult.ph_0d?.length || 0}
                              <br />1维环: {analysisResult.ph_1d?.length || 0}
                            </div>
                          )}
                          {structureTab === 'agi' && (
                            <div>
                              神经纤维丛分析完成
                              <br />层间传输矩阵已计算
                            </div>
                          )}
                          {structureTab === 'rpt' && (
                            <div>
                              黎曼平行传输分析完成
                            </div>
                          )}
                          {structureTab === 'curvature' && (
                            <div>
                              标量曲率: {analysisResult.curvature?.toFixed(4)}
                            </div>
                          )}
                          {structureTab === 'glass_matrix' && (
                            <div>
                              玻璃矩阵可视化激活
                              <br />显示激活值的几何结构
                            </div>
                          )}
                          {structureTab === 'flow_tubes' && (
                            <div>
                              信息流动轨迹可视化
                              <br />追踪语义向量演化
                            </div>
                          )}
                          {structureTab === 'global_topology' && (
                            <div>
                              全局拓扑结构分析
                            </div>
                          )}
                          {structureTab === 'fibernet_v2' && (
                            <div>
                              FiberNet V2 纤维丛拓扑演示
                            </div>
                          )}
                          {structureTab === 'holonomy' && (
                            <div>
                              全纯扫描分析
                            </div>
                          )}
                          {structureTab === 'debias' && (
                            <div>
                              几何去偏分析
                            </div>
                          )}
                          {structureTab === 'validity' && (
                            <div>
                              有效性检验完成
                            </div>
                          )}
                          {structureTab === 'training' && (
                            <div>
                              训练动力学可视化
                            </div>
                          )}
                        </div>
                      )}

                      {!analysisResult && !hoveredInfo && !displayInfo && (
                        <div style={{ color: '#666', fontStyle: 'italic', fontSize: '12px' }}>
                          悬停在可视化元素上查看详情。
                        </div>
                      )}
                    </div>
                  )
                )}

                {/* ==================== 数据对比视图 ==================== */}
                {!isObservationMode ? (
                  <div style={{
                    marginTop: '12px',
                    paddingTop: '12px',
                    borderTop: '1px solid rgba(255,255,255,0.1)'
                  }}>
                    <DataComparisonView
                      currentData={data}
                      analysisResult={analysisResult}
                      mode={structureTab}
                    />
                  </div>
                ) : (
                  <div style={{
                    marginTop: '12px',
                    padding: '10px',
                    borderTop: '1px solid rgba(255,255,255,0.1)',
                    color: '#888',
                    fontSize: '11px'
                  }}>
                    当前为观测模式，优先查看 3D 画布中的实时变化。
                  </div>
                )}
              </div>
            </div>
          )}
        </SimplePanel>
      )}

      {/* Algo Explanation Modal */}
      {showHelp && (
        <div style={{
          position: 'fixed',
          left: `${helpWindowLeft}px`,
          top: `${helpWindowTop}px`,
          zIndex: 1000,
          background: '#1a1a1f',
          border: '1px solid #333',
          borderRadius: '12px',
          width: `${helpWindowWidth}px`,
          height: `${helpWindowHeight}px`,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          boxShadow: '0 10px 40px rgba(0,0,0,0.8)'
        }}>
          <div
            onMouseDown={helpGuidePanel.handleMouseDown}
            style={{
              padding: '10px 14px',
              borderBottom: '1px solid #333',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              cursor: 'grab',
              userSelect: 'none',
              background: 'rgba(0,0,0,0.35)'
            }}
          >
            <span style={{ color: '#fff', fontSize: '13px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Settings size={14} />
              算法指南（可拖动）
            </span>
            <button
              onClick={() => setShowHelp(false)}
              style={{ background: 'transparent', border: 'none', color: '#888', cursor: 'pointer', padding: '2px', display: 'flex' }}
              title="关闭"
            >
              <X size={18} />
            </button>
          </div>
          <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
            {/* LEFT SIDEBAR */}
            <div style={{ width: '220px', background: 'rgba(0,0,0,0.3)', borderRight: '1px solid #333', display: 'flex', flexDirection: 'column' }}>
              <div style={{ padding: '20px', borderBottom: '1px solid #333', fontWeight: 'bold', color: '#fff', fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Settings size={16} />
                分析目录
              </div>
              <div style={{ flex: 1, overflowY: 'auto', padding: '10px' }}>
                {guideMenuItems.map((item, idx) => {
                  if (item.type === 'sep') {
                    return <div key={idx} style={{ height: '1px', background: 'rgba(255,255,255,0.1)', margin: '8px 0' }} />;
                  }
                  const MenuIcon = GUIDE_ICON_MAP[item.iconName] || Settings;
                  return (
                    <button
                      key={item.id}
                      onClick={() => setHelpTab(item.id)}
                      style={{
                        width: '100%', textAlign: 'left', padding: '10px',
                        background: helpTab === item.id ? 'rgba(68, 136, 255, 0.2)' : 'transparent',
                        color: helpTab === item.id ? '#fff' : '#888',
                        border: 'none', borderRadius: '6px', cursor: 'pointer',
                        fontSize: '13px', marginBottom: '2px',
                        fontWeight: helpTab === item.id ? '600' : '400',
                        transition: 'all 0.2s',
                        display: 'flex', alignItems: 'center'
                      }}
                    >
                      <span style={{ marginRight: '8px', display: 'inline-flex', alignItems: 'center' }}>
                        <MenuIcon size={14} />
                      </span>
                      {item.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* RIGHT CONTENT */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              {/* Header */}
              <div style={{ padding: '16px', borderBottom: '1px solid #333', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h2 style={{ fontSize: '18px', fontWeight: 'bold', color: '#fff', margin: 0 }}>
                  {helpTab === 'outline' ? '算法指南大纲' : (ALGO_DOCS[helpTab]?.title || '算法说明')}
                </h2>
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                  <div style={{ display: 'flex', background: '#000', borderRadius: '6px', padding: '2px', border: '1px solid #333' }}>
                    <button
                      onClick={() => setHelpMode('simple')}
                      style={{
                        padding: '6px 16px', borderRadius: '4px', border: 'none', cursor: 'pointer', fontSize: '12px', fontWeight: 'bold',
                        background: helpMode === 'simple' ? '#4488ff' : 'transparent', color: helpMode === 'simple' ? '#fff' : '#888', transition: 'all 0.2s'
                      }}
                    >
                      🟢 通俗版
                    </button>
                    <button
                      onClick={() => setHelpMode('pro')}
                      style={{
                        padding: '6px 16px', borderRadius: '4px', border: 'none', cursor: 'pointer', fontSize: '12px', fontWeight: 'bold',
                        background: helpMode === 'pro' ? '#764ba2' : 'transparent', color: helpMode === 'pro' ? '#fff' : '#888', transition: 'all 0.2s'
                      }}
                    >
                      🟣 专业版
                    </button>
                  </div>
                  <button onClick={() => setShowHelp(false)} style={{ background: 'transparent', border: 'none', color: '#888', cursor: 'pointer', padding: '4px' }}>
                    <X size={24} />
                  </button>
                </div>
              </div>
              {/* Scrollable Content */}
              <div style={{ padding: '30px', overflowY: 'auto', flex: 1, lineHeight: '1.8', fontSize: '14px', color: '#ddd' }}>
                {(() => {
                  if (helpTab === 'outline') {
                    const outlineItems = guideMenuItems.filter(item => item.id && item.id !== 'outline');
                    return (
                      <div className="animate-fade-in">
                        <h3 style={{ fontSize: '20px', color: '#4ecdc4', marginTop: 0, marginBottom: '10px' }}>
                          结构分析功能总览
                        </h3>
                        <div style={{ marginBottom: '20px', color: '#a1a1aa', fontSize: '13px' }}>
                          先在这里快速了解每个结构分析功能，再从左侧点击进入详细算法说明。
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '10px' }}>
                          {outlineItems.map(item => {
                            const doc = ALGO_DOCS[item.id];
                            const content = helpMode === 'simple' ? doc?.simple : doc?.pro;
                            const structured = GUIDE_STRUCTURED[item.id]?.[helpMode] || GUIDE_SECTION_DEFAULT[helpMode];
                            const tabMeta = structureTabUI[item.id];
                            const OutlineIcon = GUIDE_ICON_MAP[item.iconName] || Settings;
                            return (
                              <button
                                key={`outline-${item.id}`}
                                onClick={() => setHelpTab(item.id)}
                                style={{
                                  textAlign: 'left',
                                  border: '1px solid rgba(255,255,255,0.12)',
                                  background: 'rgba(255,255,255,0.02)',
                                  borderRadius: '10px',
                                  padding: '12px',
                                  cursor: 'pointer',
                                  color: '#ddd',
                                  transition: 'all 0.2s'
                                }}
                                onMouseOver={(e) => {
                                  e.currentTarget.style.background = 'rgba(68,136,255,0.12)';
                                  e.currentTarget.style.borderColor = 'rgba(68,136,255,0.35)';
                                }}
                                onMouseOut={(e) => {
                                  e.currentTarget.style.background = 'rgba(255,255,255,0.02)';
                                  e.currentTarget.style.borderColor = 'rgba(255,255,255,0.12)';
                                }}
                              >
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                                  <span style={{ display: 'inline-flex', alignItems: 'center' }}>
                                    <OutlineIcon size={14} />
                                  </span>
                                  <span style={{ color: '#fff', fontWeight: 600, fontSize: '14px' }}>{tabMeta?.name || item.label}</span>
                                </div>
                                <div style={{ fontSize: '12px', color: '#9ca3af', marginBottom: '4px' }}>
                                  目标: {structured.goal}
                                </div>
                                <div style={{ fontSize: '12px', color: '#cbd5e1' }}>
                                  {content?.desc || tabMeta?.focus || '查看该功能的详细说明。'}
                                </div>
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    );
                  }

                  const doc = ALGO_DOCS[helpTab];
                  if (!doc) return <div style={{ color: '#666', fontStyle: 'italic' }}>暂无说明文档</div>;

                  const content = helpMode === 'simple' ? doc.simple : doc.pro;
                  const structured = GUIDE_STRUCTURED[helpTab]?.[helpMode] || GUIDE_SECTION_DEFAULT[helpMode];
                  const conclusion = buildGuideConclusion({
                    tab: helpTab,
                    activeTab: structureTab,
                    analysisResult,
                    topologyResults,
                    data
                  });
                  return (
                    <div className="animate-fade-in">
                      <h3 style={{ fontSize: '20px', color: helpMode === 'simple' ? '#4ecdc4' : '#a29bfe', marginTop: 0, marginBottom: '20px' }}>
                        {content.title}
                      </h3>

                      <div style={{ marginBottom: '24px' }}>
                        {content.desc}
                      </div>

                      <div style={{ marginBottom: '22px', padding: '14px', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.15)', background: 'rgba(255,255,255,0.03)' }}>
                        <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#fff', marginBottom: '12px' }}>
                          结构化说明
                        </div>
                        <div style={{ marginBottom: '10px' }}>
                          <div style={{ color: '#7dd3fc', fontWeight: '600', fontSize: '12px' }}>1. 目标</div>
                          <div style={{ color: '#d1d5db', fontSize: '13px' }}>{structured.goal}</div>
                        </div>
                        <div style={{ marginBottom: '10px' }}>
                          <div style={{ color: '#7dd3fc', fontWeight: '600', fontSize: '12px' }}>2. 思路</div>
                          <ul style={{ paddingLeft: '18px', margin: '4px 0 0 0', color: '#d1d5db', fontSize: '13px' }}>
                            {structured.approach.map((item, idx) => (
                              <li key={`approach-${idx}`} style={{ marginBottom: '4px' }}>{item}</li>
                            ))}
                          </ul>
                        </div>
                        <div style={{ marginBottom: '10px' }}>
                          <div style={{ color: '#7dd3fc', fontWeight: '600', fontSize: '12px' }}>3. 3D模型原理</div>
                          <div style={{ color: '#d1d5db', fontSize: '13px' }}>{structured.model3d}</div>
                        </div>
                        <div style={{ marginBottom: '10px' }}>
                          <div style={{ color: '#7dd3fc', fontWeight: '600', fontSize: '12px' }}>4. 算法说明</div>
                          <div style={{ color: '#d1d5db', fontSize: '13px' }}>{structured.algorithm}</div>
                        </div>
                        <div>
                          <div style={{ color: '#7dd3fc', fontWeight: '600', fontSize: '12px' }}>5. 指标范围</div>
                          <ul style={{ paddingLeft: '18px', margin: '4px 0 0 0', color: '#d1d5db', fontSize: '13px' }}>
                            {structured.metricRanges.map((item, idx) => (
                              <li key={`range-${idx}`} style={{ marginBottom: '4px' }}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      </div>

                      <div style={{
                        marginBottom: '24px',
                        padding: '14px',
                        borderRadius: '10px',
                        border: conclusion.available ? '1px solid rgba(94, 201, 98, 0.35)' : '1px solid rgba(255,159,67,0.35)',
                        background: conclusion.available ? 'rgba(94, 201, 98, 0.08)' : 'rgba(255,159,67,0.08)'
                      }}>
                        <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#fff', marginBottom: '8px' }}>
                          {conclusion.title}
                        </div>
                        <ul style={{ paddingLeft: '18px', margin: '0 0 10px 0', color: '#d1d5db', fontSize: '13px' }}>
                          {conclusion.lines.map((line, idx) => (
                            <li key={`conclusion-${idx}`} style={{ marginBottom: '4px' }}>{line}</li>
                          ))}
                        </ul>
                        {conclusion.metrics?.length > 0 && (
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                            {conclusion.metrics.map((metric, idx) => (
                              <div
                                key={`metric-${idx}`}
                                style={{
                                  padding: '4px 8px',
                                  borderRadius: '6px',
                                  border: '1px solid rgba(255,255,255,0.18)',
                                  background: 'rgba(0,0,0,0.2)',
                                  color: '#e5e7eb',
                                  fontSize: '12px'
                                }}
                              >
                                {metric.label}: <span style={{ color: '#fff', fontWeight: 600 }}>{metric.value}</span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>

                      <div style={{ borderTop: '1px solid rgba(255,255,255,0.12)', paddingTop: '14px' }}>
                        <h4 style={{ margin: '0 0 10px 0', fontSize: '13px', color: '#a1a1aa', fontWeight: 'bold', letterSpacing: '0.02em' }}>
                          补充算法说明
                        </h4>
                        {content.points && (
                          <ul style={{ paddingLeft: '20px', color: '#ccc', marginBottom: '24px' }}>
                            {content.points.map((p, i) => (
                              <li key={i} style={{ marginBottom: '10px' }}>{p}</li>
                            ))}
                          </ul>
                        )}

                        {content.blocks && content.blocks.map((b, i) => (
                          <div key={i} style={{
                            background: `rgba(${b.color || '255,255,255'}, 0.05)`,
                            border: `1px solid rgba(${b.color || '255,255,255'}, 0.2)`,
                            borderRadius: '8px', padding: '16px', marginBottom: '16px'
                          }}>
                            <h4 style={{ margin: '0 0 8px 0', color: `rgb(${b.color || '255,255,255'})` }}>{b.title}</h4>
                            <p style={{ margin: 0, fontSize: '13px', color: '#bbb' }}>{b.text}</p>
                          </div>
                        ))}

                        {content.formula && (
                          <div style={{ background: '#000', padding: '16px', borderRadius: '8px', border: '1px solid #333', fontFamily: 'monospace', margin: '20px 0', color: '#ffe66d' }}>
                            {content.formula}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })()}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Right-side Layer Detail Panel */}
      {selectedLayer !== null && data?.layer_details && (
        <SimplePanel
          title={`第 ${selectedLayer} 层详情`}
          onClose={() => {
            setSelectedLayer(null);
            setLayerInfo(null);
          }}
          style={{
            position: 'absolute', right: 340, bottom: 20, zIndex: 10,
            minWidth: '450px', maxWidth: '550px', maxHeight: '80vh'
          }}
        >

          {(() => {
            const layerDetail = data.layer_details[selectedLayer];
            if (!layerDetail) return <div style={{ padding: '20px', color: '#aaa' }}>加载层详情中...</div>;

            return (
              <div style={{ fontSize: '13px', lineHeight: '1.8' }}>
                {/* 3D Visualization */}
                {layerInfo && (
                  <div style={{
                    height: '350px',
                    background: '#0a0a0a',
                    borderRadius: '8px',
                    marginBottom: '16px',
                    border: '1px solid #333'
                  }}>
                    <ErrorBoundary>
                      <Canvas>
                        <PerspectiveCamera makeDefault position={[0, 0, 12]} fov={50} />
                        <OrbitControls enableDamping dampingFactor={0.05} />
                        <ambientLight intensity={0.4} />
                        <pointLight position={[10, 10, 10]} intensity={0.8} />
                        <pointLight position={[-10, -10, 10]} intensity={0.3} color="#00d2ff" />
                        <LayerDetail3D
                          layerIdx={selectedLayer}
                          layerInfo={layerInfo}
                          onHeadClick={handleHeadClick}
                        />
                      </Canvas>
                    </ErrorBoundary>
                    <div style={{
                      padding: '8px',
                      fontSize: '10px',
                      color: '#666',
                      textAlign: 'center'
                    }}>
                      💡 拖动旋转 • 滚轮缩放 • 右键平移
                    </div>
                  </div>
                )}

                <div style={{ marginBottom: '14px' }}>
                  <h3 style={{ margin: '0 0 8px 0', fontSize: '14px', color: '#fff', fontWeight: '600' }}>
                    架构
                  </h3>
                  <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr', gap: '6px', color: '#aaa' }}>
                    <span>注意力头数:</span>
                    <span style={{ color: '#fff' }}>{layerDetail.n_heads}</span>

                    <span>头维度:</span>
                    <span style={{ color: '#fff' }}>{layerDetail.d_head}</span>

                    <span>MLP隐藏维度:</span>
                    <span style={{ color: '#fff' }}>{layerDetail.d_mlp}</span>
                  </div>
                </div>

                <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '14px' }}>
                  <h3 style={{ margin: '0 0 8px 0', fontSize: '14px', color: '#fff', fontWeight: '600' }}>
                    参数
                  </h3>
                  <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr', gap: '6px', color: '#aaa' }}>
                    <span>注意力:</span>
                    <span style={{ color: '#5ec962' }}>
                      {(layerDetail.attn_params / 1e6).toFixed(2)}M
                    </span>

                    <span>MLP (前馈):</span>
                    <span style={{ color: '#5ec962' }}>
                      {(layerDetail.mlp_params / 1e6).toFixed(2)}M
                    </span>

                    <span style={{ fontWeight: '600' }}>总计:</span>
                    <span style={{ color: '#00d2ff', fontWeight: '600' }}>
                      {(layerDetail.total_params / 1e6).toFixed(2)}M
                    </span>
                  </div>
                </div>

                <div style={{
                  marginTop: '14px',
                  padding: '10px',
                  background: 'rgba(0, 210, 255, 0.1)',
                  borderRadius: '6px',
                  fontSize: '11px',
                  color: '#aaa'
                }}>
                  💡 点击其他层查看详情，或点击 × 关闭
                </div>
              </div>
            );
          })()}
        </SimplePanel>
      )}

      {/* Neuron State Visualization Panel */}
      {layerNeuronState && panelVisibility.neuronPanel && (
        <SimplePanel
          title={t('panels.neuronStateTitle', { layer: layerNeuronState.layer_idx })}
          onClose={() => setLayerNeuronState(null)}
          dragHandleProps={{ onMouseDown: neuronPanel.handleMouseDown }}
          headerStyle={{ cursor: 'grab' }}
          style={{
            position: 'absolute',
            left: `${neuronPanel.position.x}px`,
            top: `${neuronPanel.position.y}px`,
            zIndex: 15,
            width: '350px',
            maxHeight: '60vh'
          }}
        >

          {loadingNeurons ? (
            <div style={{ textAlign: 'center', padding: '40px', color: '#888' }}>
              加载神经元状态中...
            </div>
          ) : (
            <div>
              <div style={{ marginBottom: '20px' }}>
                <h3 style={{ margin: '0 0 12px 0', fontSize: '16px', color: '#fff', fontWeight: '600' }}>
                  注意力模式 ({layerNeuronState.n_heads} 个头)
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px' }}>
                  {layerNeuronState.attention_heads.map(head => (
                    <AttentionHeatmap
                      key={head.head_idx}
                      pattern={head.pattern}
                      tokens={layerNeuronState.tokens}
                      headIdx={head.head_idx}
                    />
                  ))}
                </div>
              </div>

              <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '16px' }}>
                <h3 style={{ margin: '0 0 12px 0', fontSize: '16px', color: '#fff', fontWeight: '600' }}>
                  MLP激活
                </h3>
                <div style={{ marginBottom: '12px' }}>
                  <MLPActivationChart distribution={layerNeuronState.mlp_stats.activation_distribution} />
                </div>
                <div style={{ fontSize: '11px', color: '#aaa', lineHeight: '1.6' }}>
                  <div>均值: <span style={{ color: '#fff' }}>{layerNeuronState.mlp_stats.mean.toFixed(3)}</span></div>
                  <div>标准差: <span style={{ color: '#fff' }}>{layerNeuronState.mlp_stats.std.toFixed(3)}</span></div>
                  <div>范围: <span style={{ color: '#fff' }}>[{layerNeuronState.mlp_stats.min.toFixed(3)}, {layerNeuronState.mlp_stats.max.toFixed(3)}]</span></div>
                </div>
              </div>

              <div style={{
                marginTop: '16px',
                padding: '10px',
                background: 'rgba(0, 210, 255, 0.1)',
                borderRadius: '6px',
                fontSize: '10px',
                color: '#aaa'
              }}>
                <div><strong>热图:</strong> 从行(查询)到列(键)的注意力</div>
                <div><strong>颜色:</strong> 蓝色(低) → 紫色(中) → 红色(高)</div>
              </div>
            </div>
          )}
        </SimplePanel>
      )}

      {/* ==================== 右下: 操作面板 ==================== */}
      {panelVisibility.layersPanel && (
        <SimplePanel
          title={`操作面板 · ${currentStructureUI.name}`}
          style={{
            position: 'absolute', bottom: 20, right: 20, zIndex: 10,
            width: '360px',
            maxHeight: isLayersPanelMinimized ? 'none' : rightPanelMaxHeight,
            display: 'flex', flexDirection: 'column',
            overflow: 'hidden'
          }}
          actions={
            <button
              onClick={() => setIsLayersPanelMinimized(prev => !prev)}
              style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: '#888', padding: '4px', display: 'flex', transition: 'color 0.2s' }}
              onMouseOver={(e) => e.currentTarget.style.color = '#fff'}
              onMouseOut={(e) => e.currentTarget.style.color = '#888'}
              title={isLayersPanelMinimized ? 'Maximize panel' : 'Minimize panel'}
            >
              {isLayersPanelMinimized ? <Maximize2 size={16} /> : <Minimize2 size={16} />}
            </button>
          }
        >
          {!isLayersPanelMinimized && (
            <>
              <div style={{
                marginBottom: '10px',
                padding: '8px',
                borderRadius: '6px',
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.08)',
                fontSize: '11px',
                color: '#bbb'
              }}>
                <div style={{ color: '#fff', fontWeight: '600', marginBottom: '2px' }}>
                  当前结构分析: {currentStructureUI.name}
                </div>
                <div>{currentStructureUI.focus}</div>
              </div>
              {/* ==================== 数据展示模板 ==================== */}
              {!isObservationMode ? (
                <div style={{
                  marginBottom: '12px',
                  padding: '8px',
                  background: 'rgba(0,0,0,0.2)',
                  borderRadius: '6px',
                  flex: 1,
                  overflowY: 'auto'
                }}>
                  <AnalysisDataDisplay
                    mode={structureTab}
                    data={data}
                    analysisResult={analysisResult}
                    selectedLayer={selectedLayer}
                    onLayerSelect={(layerIdx) => {
                      setSelectedLayer(layerIdx);
                      loadLayerDetails(layerIdx);
                    }}
                    hoveredInfo={hoveredInfo}
                  />
                </div>
              ) : (
                <div style={{
                  marginBottom: '12px',
                  padding: '10px',
                  background: 'rgba(0,0,0,0.2)',
                  borderRadius: '6px',
                  border: '1px solid rgba(255,255,255,0.08)',
                  fontSize: '12px',
                  color: '#bbb'
                }}>
                  <div style={{ color: '#fff', marginBottom: '6px', fontWeight: '600' }}>观测模式面板</div>
                  <div>实时层: {activeLayer !== null ? `L${activeLayer}` : '-'}</div>
                  <div>悬停词元: {(hoveredInfo || displayInfo)?.label || '-'}</div>
                  <div>置信度: {(hoveredInfo || displayInfo)?.probability ? `${((hoveredInfo || displayInfo).probability * 100).toFixed(1)}%` : '-'}</div>
                </div>
              )}

              {/* ==================== 快速指标栏 ==================== */}
              <div style={{
                display: 'flex',
                gap: '6px',
                marginBottom: '10px',
                padding: '6px',
                background: 'rgba(255,255,255,0.03)',
                borderRadius: '6px'
              }}>
                {operationMetrics.map((metric, idx) => (
                  <MetricCard key={`${metric.label}-${idx}`} label={metric.label} value={metric.value} color={metric.color} />
                ))}
              </div>

              {/* ==================== 操作历史 ==================== */}
              <div style={{
                padding: '8px',
                background: 'rgba(0,0,0,0.2)',
                borderRadius: '6px',
                maxHeight: '150px',
                overflowY: 'auto'
              }}>
                <OperationHistoryPanel
                  history={history}
                  onRestore={(item) => {
                    if (item.details?.mode) {
                      setStructureTab(item.details.mode);
                    }
                  }}
                  onClear={clearHistory}
                  onRemove={(id) => {
                    // 简单过滤掉指定id
                    const idx = history.findIndex(h => h.id === id);
                    if (idx !== -1) {
                      history.splice(idx, 1);
                    }
                  }}
                  maxVisible={3}
                />
              </div>
            </>
          )}
        </SimplePanel>
      )}

      {/* 3D Canvas - Conditionally Render FiberNetV2Demo */}
      {structureTab === 'fibernet_v2' ? (
        <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 1 }}>
          <FiberNetV2Demo t={t} />
        </div>
      ) : (
        <Canvas shadows>
          <PerspectiveCamera makeDefault position={[20, 20, 20]} fov={50} />
          <OrbitControls makeDefault target={structureTab === 'rpt' ? [0, 0, 0] : [0, 0, 0]} />

          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} intensity={1} castShadow />
          <spotLight position={[-10, 20, 10]} angle={0.15} penumbra={1} intensity={1} />

          {/* Standard LogitLens Visualization - Always visible if data exists */}
          {data && (
            <Text position={[0, 15, -5]} fontSize={1} color="#ffffff" anchorX="center" anchorY="bottom">
              Logit Lens (Token Probabilities)
            </Text>
          )}
          <Visualization data={data} hoveredInfo={hoveredInfo} setHoveredInfo={setHoveredInfo} activeLayer={activeLayer} />

          {/* PGRF: Pan-Geometric Resonance Field - 全局大一统背景 */}
          <ResonanceField3D
            topologyResults={topologyResults}
            activeTab={structureTab}
          />

          {/* Analysis Overlays - 模态观测图层叠加 */}
          {analysisResult && structureTab !== 'glass_matrix' && structureTab !== 'flow_tubes' && (
            <group position={data ? [-data.tokens.length, 0, -data.logit_lens.length] : [0, 0, 0]}>
              {/* 场景标签 - 动态显示当前观测模态 */}
              <Text position={[0, 14, 0]} fontSize={1} color="#4ecdc4" anchorX="center">
                {structureTab === 'circuit' && '回路观测 (Circuit Overlay)'}
                {structureTab === 'features' && '特征观测 (Feature Overlay)'}
                {structureTab === 'causal' && '因果深度观测 (Causal Overlay)'}
                {structureTab === 'manifold' && '流形拓扑观测 (Manifold Overlay)'}
                {structureTab === 'compositional' && t('structure.compositional.title')}
                {structureTab === 'rpt' && '语义传输轨迹 (Riemannian Parallel Transport)'}
                {structureTab === 'curvature' && '流形曲率云 (Curvature Field)'}
              </Text>

              {/* 具体分析图层 - 以叠加模式呈现 */}
              {structureTab === 'circuit' && <NetworkGraph3D graph={analysisResult.graph || analysisResult} activeLayer={activeLayer} />}
              {structureTab === 'features' && <FeatureVisualization3D features={analysisResult.top_features} layerIdx={analysisResult.layer_idx} onLayerClick={setSelectedLayer} selectedLayer={selectedLayer} onHover={setHoveredInfo} />}
              {structureTab === 'causal' && <NetworkGraph3D graph={analysisResult.causal_graph} activeLayer={activeLayer} />}
              {structureTab === 'manifold' && analysisResult && <ManifoldVisualization3D pcaData={analysisResult.pca} onHover={setHoveredInfo} />}
              {structureTab === 'compositional' && analysisResult && <CompositionalVisualization3D result={analysisResult} t={t} />}
              {structureTab === 'rpt' && analysisResult && (
                <RPTVisualization3D data={analysisResult} t={t} />
              )}
              {structureTab === 'curvature' && analysisResult && <CurvatureField3D result={analysisResult} t={t} />}
              {structureTab === 'debias' && analysisResult && (
                <group>
                  <Text position={[0, 8, 0]} fontSize={0.6} color="#bb88ff">Geometric Interception (Debias)</Text>
                  <mesh rotation={[Math.PI / 2, 0, 0]}>
                    <torusGeometry args={[4, 0.05, 16, 100]} />
                    <meshStandardMaterial color="#bb88ff" emissive="#bb88ff" emissiveIntensity={2} />
                  </mesh>
                  <mesh position={[0, 0, 0]}>
                    <sphereGeometry args={[3.8, 32, 32, 0, Math.PI * 2, 0, Math.PI / 2]} />
                    <meshStandardMaterial color="#bb88ff" transparent opacity={0.1} side={THREE.DoubleSide} />
                  </mesh>
                </group>
              )}
              {structureTab === 'agi' && analysisResult && <FiberBundleVisualization3D result={analysisResult} t={t} />}
              {structureTab === 'fiber' && <FiberBundleVisualization3D result={analysisResult} t={t} />}
              {structureTab === 'validity' && <ValidityVisualization3D result={analysisResult} t={t} />}
            </group>
          )}

          {/* Independent Visualizations (No Analysis Result Needed) */}
          {/* Note: GlassMatrix3D and FlowTubesVisualizer have their own Canvas, rendered outside */}

          {structureTab === 'flow_tubes' && (
            <group position={[0, -5, 0]}>
              <FlowTubesVisualizer />
            </group>
          )}

          {structureTab === 'tda' && (
            <group position={[0, 0, 0]}>
              <TDAVisualization3D result={analysisResult} t={t} />
            </group>
          )}

          {/* Debug Log for SNN Rendering Conditions */}
          {(() => {
            if (inputPanelTab === 'snn' || snnState.initialized) {
              console.log('[App] SNN Render Check:', { inputPanelTab, initialized: snnState.initialized, hasStructure: !!snnState.structure });
            }
            return null;
          })()}


          {/* SNN Visualization - Independent of structure analysis result */}
          {(inputPanelTab === 'snn' || systemType === 'snn') && snnState.initialized && snnState.structure && (
            <group position={(!data || systemType === 'snn') ? [0, 0, 0] : [-(data?.tokens?.length || 10) - 20, 0, 0]}>
              <SNNVisualization3D
                t={t}
                structure={snnState.structure}
                activeSpikes={snnState.spikes}
              />
            </group>
          )}

          {/* Magnified Layer Visualization during generation */}
          {activeLayer !== null && activeLayerInfo && (
            <group position={[30, 0, 0]}>
              <Text
                position={[0, 8, 0]}
                fontSize={0.5}
                color="#00d2ff"
                anchorX="center"
              >
                {computationPhase === 'attention' ? (t('app.computingAttention') || 'Computing Attention') :
                  computationPhase === 'mlp' ? (t('app.processingMlp') || 'Processing MLP') :
                    computationPhase === 'output' ? (t('app.generatingOutput') || 'Generating Output') : ''}
              </Text>

              <LayerDetail3D
                layerIdx={activeLayer}
                layerInfo={activeLayerInfo}
                animationPhase={computationPhase}
                isActive={true}
                onHeadClick={handleHeadClick}
              />
            </group>
          )}

          <ContactShadows resolution={1024} scale={20} blur={2} opacity={0.35} far={10} color="#000000" />
          <gridHelper args={[100, 50, '#222', '#111']} position={[0, -0.6, 0]} />
        </Canvas>
      )}

      {/* GlassMatrix3D - Has its own Canvas, must be rendered outside main Canvas */}
      {structureTab === 'glass_matrix' && (
        <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 1 }}>
          <GlassMatrix3D />
        </div>
      )}

      {/* Head Analysis Panel - Draggable */}
      {panelVisibility.headPanel && headPanel.isOpen && (
        <SimplePanel
          title={t ? t('head.title', { layer: headPanel.layerIdx, head: headPanel.headIdx }) : `Layer ${headPanel.layerIdx} Head ${headPanel.headIdx}`}
          onClose={() => setHeadPanel({ ...headPanel, isOpen: false })}
          dragHandleProps={{ onMouseDown: headPanelDrag.handleMouseDown }}
          headerStyle={{ cursor: 'grab' }}
          style={{
            position: 'absolute',
            left: `${headPanelDrag.position.x}px`,
            top: `${headPanelDrag.position.y}px`,
            zIndex: 25,
            width: '500px',
            height: '400px'
          }}
        >
          <HeadAnalysisPanel
            layerIdx={headPanel.layerIdx}
            headIdx={headPanel.headIdx}
            prompt={prompt}
            t={t}
          />
        </SimplePanel>
      )}

      {/* AGIChatPanel Terminal */}
      {panelVisibility.agiChatPanel && (
        <AGIChatPanel onClose={() => setPanelVisibility(prev => ({ ...prev, agiChatPanel: false }))} t={t} />
      )}

      {/* MotherEnginePanel */}
      {panelVisibility.motherEnginePanel && (
        <MotherEnginePanel
          onClose={() => setPanelVisibility(prev => ({ ...prev, motherEnginePanel: false }))}
          t={t}
        />
      )}

      {/* Project Genesis Blueprint Overlay */}
      {showBlueprint && (
        <div style={{ position: 'absolute', inset: 0, zIndex: 3000 }}>
          <HLAIBlueprint
            initialTab={blueprintInitialTab}
            onClose={() => {
              setShowBlueprint(false);
              setBlueprintInitialTab('roadmap');
            }}
          />
        </div>
      )}

    </div>
  );
}
