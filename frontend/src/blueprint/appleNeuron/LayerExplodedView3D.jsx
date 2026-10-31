/**
 * LayerExplodedView3D - Transformer Layer 内部结构的3D展开模型
 * 沿 Z 轴竖向布局（与 DNN 整体模型方向一致）
 * 放在 DNN 模型旁边，与 Forward Pass 动画同步
 */
import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Line, Text } from '@react-three/drei';
import { MODEL_CONFIGS } from './constants';

// ── 动画子阶段 ──
const PHASES = [
  { id: 'input',     color: '#94a3b8' },
  { id: 'ln1',       color: '#818cf8' },
  { id: 'qkv',       color: '#60a5fa' },
  { id: 'attn_score',color: '#38bdf8' },
  { id: 'softmax',   color: '#22d3ee' },
  { id: 'attn_out',  color: '#2dd4bf' },
  { id: 'residual1', color: '#a78bfa' },
  { id: 'ln2',       color: '#818cf8' },
  { id: 'ffn_up',    color: '#f59e0b' },
  { id: 'ffn_act',   color: '#fb923c' },
  { id: 'ffn_down',  color: '#f97316' },
  { id: 'residual2', color: '#a78bfa' },
];
const PHASE_DURATIONS = [0.08, 0.07, 0.12, 0.1, 0.08, 0.1, 0.07, 0.07, 0.1, 0.08, 0.08, 0.05];
const TOTAL_DUR = PHASE_DURATIONS.reduce((s, d) => s + d, 0);

function getPhaseBoundaries() {
  let cum = 0;
  return PHASES.map((p, i) => {
    const start = cum / TOTAL_DUR;
    cum += PHASE_DURATIONS[i];
    const end = cum / TOTAL_DUR;
    return { ...p, start, end };
  });
}

// ── 3D 布局常量（沿Z轴竖向，放大尺寸） ──
const BW = 3.8;       // X 方向宽度
const BH = 2.8;       // Y 方向高度
const BD = 0.55;       // Z 方向厚度（每个块的深度，沿Z轴排列）
const GAP_Z = 1.2;    // Z 方向间距
const THIN_D = 0.28;  // LayerNorm 等薄层厚度（Z方向）
const WIDE_D = 0.8;   // FFN Up 等厚块

// ── 辅助组件 ──

/** 3D 块体 - Z轴布局，数据流从底到顶 */
function Block3D({ position, size = [BW, BH, BD], color, label, sublabel, active, emissiveBoost = 0 }) {
  const ref = useRef(null);
  const [w, h, d] = size;

  useFrame(() => {
    if (!ref.current) return;
    ref.current.material.emissiveIntensity = active ? 1.4 + emissiveBoost : 0.15;
  });

  const hw = w / 2, hh = h / 2, hd = d / 2;

  return (
    <group position={position}>
      <mesh ref={ref}>
        <boxGeometry args={size} />
        <meshStandardMaterial
          color={active ? color : '#1e293b'}
          emissive={color}
          emissiveIntensity={active ? 1.4 + emissiveBoost : 0.15}
          transparent
          opacity={active ? 0.92 : 0.55}
          roughness={0.3}
          metalness={0.1}
        />
      </mesh>
      {active && (
        <>
          <Line points={[[-hw,-hh,-hd],[hw,-hh,-hd],[hw,hh,-hd],[-hw,hh,-hd],[-hw,-hh,-hd]]} color={color} transparent opacity={0.85} lineWidth={1.5} />
          <Line points={[[-hw,-hh,hd],[hw,-hh,hd],[hw,hh,hd],[-hw,hh,hd],[-hw,-hh,hd]]} color={color} transparent opacity={0.85} lineWidth={1.5} />
          <Line points={[[-hw,-hh,-hd],[-hw,-hh,hd]]} color={color} transparent opacity={0.85} lineWidth={1.5} />
          <Line points={[[hw,-hh,-hd],[hw,-hh,hd]]} color={color} transparent opacity={0.85} lineWidth={1.5} />
          <Line points={[[hw,hh,-hd],[hw,hh,hd]]} color={color} transparent opacity={0.85} lineWidth={1.5} />
          <Line points={[[-hw,hh,-hd],[-hw,hh,hd]]} color={color} transparent opacity={0.85} lineWidth={1.5} />
        </>
      )}
      {/* 标签 - 右侧 */}
      <Text
        position={[hw + 0.3, 0.15, 0]}
        fontSize={0.32}
        color={active ? '#ffffff' : '#8899bb'}
        anchorX="left"
        anchorY="middle"
        outlineWidth={0.02}
        outlineColor="#0a1022"
      >
        {label}
      </Text>
      {sublabel && (
        <Text
          position={[hw + 0.3, -0.25, 0]}
          fontSize={0.2}
          color={active ? color : '#556688'}
          anchorX="left"
          anchorY="middle"
          outlineWidth={0.012}
          outlineColor="#0a1022"
        >
          {sublabel}
        </Text>
      )}
    </group>
  );
}

/** 残差连接环 */
function ResidualRing3D({ position, active }) {
  const ref = useRef(null);
  useFrame((state) => {
    if (!ref.current) return;
    ref.current.rotation.x = state.clock.elapsedTime * 0.5;
    ref.current.rotation.y = Math.PI / 2;
  });

  return (
    <group position={position}>
      <mesh ref={ref}>
        <torusGeometry args={[0.55, 0.06, 16, 40]} />
        <meshStandardMaterial
          color={active ? '#a78bfa' : '#2d2455'}
          emissive="#a78bfa"
          emissiveIntensity={active ? 1.8 : 0.2}
          transparent
          opacity={active ? 0.9 : 0.4}
        />
      </mesh>
      <Text
        position={[0, 0.8, 0]}
        fontSize={0.26}
        color={active ? '#c4b5fd' : '#556688'}
        anchorX="center"
        anchorY="bottom"
        outlineWidth={0.015}
        outlineColor="#0a1022"
      >
        ⊕ +x
      </Text>
    </group>
  );
}

/** 激活值 → 颜色映射 */
function activationColor(val) {
  if (val > 0.8) return '#ef4444';   // 红色
  if (val > 0.5) return '#fbbf24';   // 黄色
  if (val > 0.3) return '#22c55e';   // 绿色
  return '#60a5fa';                  // 蓝色（低激活）
}

/** Attention Heads 网格 - 小球体（右侧排列）+ 激活值标签 */
function AttentionHeadGrid3D({ position, nHeads, activeHeads = [], headActivations = [], active }) {
  const displayN = Math.min(nHeads, 12);
  const cols = Math.min(nHeads, 4);
  const rows = Math.ceil(displayN / cols);
  const spacing = 0.45;

  return (
    <group position={position}>
      {Array.from({ length: displayN }, (_, i) => {
        const col = i % cols;
        const row = Math.floor(i / cols);
        const isActiveHead = activeHeads.includes(i);
        const act = headActivations[i] ?? (isActiveHead ? 0.6 : 0.15);
        const c = active ? activationColor(act) : '#1a2540';
        const x = BW / 2 + 0.4 + col * spacing;
        const y = ((rows - 1) / 2 - row) * spacing;
        return (
          <group key={`head-${i}`}>
            <mesh position={[x, y, 0]}>
              <sphereGeometry args={[0.16, 12, 12]} />
              <meshStandardMaterial
                color={c}
                emissive={active ? c : '#0a1530'}
                emissiveIntensity={active && act > 0.3 ? 2.0 : 0.15}
                transparent
                opacity={active && act > 0.3 ? 0.95 : 0.3}
              />
            </mesh>
            {active && act > 0.3 && (
              <Text
                position={[x + 0.22, y, 0]}
                fontSize={0.13}
                color={c}
                anchorX="left"
                anchorY="middle"
                outlineWidth={0.008}
                outlineColor="#0a1022"
              >
                {act.toFixed(2)}
              </Text>
            )}
          </group>
        );
      })}
    </group>
  );
}

/** FFN Neuron 柱状条 - 垂直条 + 激活值标签 */
function FFNNeuronBars3D({ position, neurons = [], active }) {
  const display = neurons.slice(0, 8);
  const maxAct = Math.max(0.01, ...display.map(n => n.activation || 0));
  const barSpacing = 0.35;

  return (
    <group position={position}>
      {display.map((n, i) => {
        const h = Math.max(0.2, (n.activation / maxAct) * 2.0);
        const y = (i - (display.length - 1) / 2) * barSpacing;
        const x = BW / 2 + 0.4;
        const c = activationColor(n.activation);
        return (
          <group key={`neuron-${i}`}>
            <mesh position={[x, y, 0]}>
              <boxGeometry args={[0.2, h, 0.2]} />
              <meshStandardMaterial
                color={active ? c : '#1a2540'}
                emissive={active ? c : '#0a1530'}
                emissiveIntensity={active ? 1.2 : 0.1}
                transparent
                opacity={active ? 0.88 : 0.3}
              />
            </mesh>
            {active && n.activation > 0.3 && (
              <Text
                position={[x + 0.22, y, 0]}
                fontSize={0.13}
                color={c}
                anchorX="left"
                anchorY="middle"
                outlineWidth={0.008}
                outlineColor="#0a1022"
              >
                {n.activation.toFixed(2)}
              </Text>
            )}
          </group>
        );
      })}
    </group>
  );
}

/** 连接线 */
function ConnectorLine({ from, to, color, active }) {
  return (
    <Line
      points={[from, to]}
      color={active ? color : '#2a3a55'}
      transparent
      opacity={active ? 0.85 : 0.2}
      lineWidth={active ? 2.5 : 1.0}
    />
  );
}

// ── 主组件 ──

export default function LayerExplodedView3D({
  layerIdx = null,
  modelKey = null,
  layerData = null,
  isActive = false,
  fpSpeed = 800,
  animProgress = 0,
  position = [10, 0, 0],  // DNN 模型旁边
}) {
  const mc = MODEL_CONFIGS[modelKey];
  const phaseBoundaries = useMemo(() => getPhaseBoundaries(), []);

  const currentPhase = useMemo(() => {
    if (!isActive || animProgress == null) return null;
    for (const pb of phaseBoundaries) {
      if (animProgress >= pb.start && animProgress < pb.end) return pb.id;
    }
    return 'residual2';
  }, [isActive, animProgress, phaseBoundaries]);

  const isPhase = (id) => currentPhase === id;
  const isPhaseGroup = (ids) => ids.includes(currentPhase);

  const nHeads = mc?.nHeads || 20;
  const headDim = mc?.headDim || 128;
  const dModel = mc?.dModel || 2560;
  const mlpDim = mc?.mlpDim || 6912;

  const activeHeads = useMemo(() => {
    if (!layerData?.attention) {
      // 默认模拟数据
      return Array.from({ length: Math.min(4, nHeads) }, (_, i) => i);
    }
    const pattern = layerData.attention.pattern;
    if (pattern && pattern.length > 0) {
      return pattern.slice(0, Math.min(nHeads, pattern.length))
        .map((row, i) => ({ idx: i, diag: row[i] || 0 }))
        .filter(h => h.diag > 0.2)
        .map(h => h.idx);
    }
    return Array.from({ length: Math.min(4, nHeads) }, (_, i) => i);
  }, [layerData, nHeads]);

  // Attention Head 激活值数组
  const headActivations = useMemo(() => {
    if (layerData?.attention?.pattern && layerData.attention.pattern.length > 0) {
      return layerData.attention.pattern.slice(0, nHeads).map((row, i) => row[i] || 0);
    }
    // 默认模拟激活值（动画激活时才有意义）
    if (!isActive) return [];
    const mock = [0.85, 0.62, 0.45, 0.73, 0.28, 0.55, 0.91, 0.38, 0.67, 0.15, 0.50, 0.80];
    return Array.from({ length: Math.min(nHeads, mock.length) }, (_, i) => mock[i % mock.length]);
  }, [layerData, nHeads, isActive]);

  // FFN 神经元激活数据（当 layerData 中没有时生成模拟数据）
  const ffnNeurons = useMemo(() => {
    if (layerData?.ffn?.top_neurons?.length) return layerData.ffn.top_neurons;
    if (!isActive) return [];
    // 模拟数据
    return [
      { activation: 0.88 }, { activation: 0.65 }, { activation: 0.42 },
      { activation: 0.78 }, { activation: 0.35 }, { activation: 0.55 },
      { activation: 0.92 }, { activation: 0.20 },
    ];
  }, [layerData, isActive]);

  // ── 沿 Z 轴布局（与 DNN 层方向一致）：从底到顶 ──
  let z = 0;
  const advanceZ = (d = BD) => { const cz = z; z += d + GAP_Z; return cz; };

  const zInput     = advanceZ(BD);
  const zLn1       = advanceZ(THIN_D);
  const zQkv       = advanceZ(BD);
  const zAttnScore = advanceZ(BD);
  const zSoftmax   = advanceZ(THIN_D);
  const zAttnOut   = advanceZ(BD);
  const zRes1      = advanceZ(THIN_D);
  const zLn2       = advanceZ(THIN_D);
  const zFfnUp     = advanceZ(WIDE_D);
  const zFfnAct    = advanceZ(BD);
  const zFfnDown   = advanceZ(BD);
  const zRes2      = advanceZ(THIN_D);
  const zOutput    = advanceZ(BD);

  const attnActive = isPhaseGroup(['qkv', 'attn_score', 'softmax', 'attn_out']);
  const ffnActive  = isPhaseGroup(['ffn_up', 'ffn_act', 'ffn_down']);

  const cx = position[0];
  const cy = position[1];
  const cz = position[2];

  const layerLabel = layerIdx != null ? `Layer ${layerIdx}` : '--';
  const layerFunc = layerData?.label || '';
  const totalZ = z;

  // 居中偏移
  const offsetZ = -totalZ / 2;
  const pz = (rawZ) => rawZ + offsetZ;

  return (
    <group position={[cx, cy, cz]} scale={1.5}>
      {/* 标题 - 顶部 */}
      <Text
        position={[0, BH / 2 + 3.0, pz(zOutput) + 1]}
        fontSize={0.72}
        color={isActive ? '#4facfe' : '#556688'}
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.035}
        outlineColor="#0a1022"
      >
        {`${layerLabel}: ${layerFunc}`}
      </Text>
      {mc && (
        <Text
          position={[0, BH / 2 + 2.2, pz(zOutput) + 1]}
          fontSize={0.36}
          color="#7f95bb"
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.016}
          outlineColor="#0a1022"
        >
          {`${mc.name}  d=${dModel}  h=${nHeads}×${headDim}  mlp=${mlpDim?.toLocaleString()}`}
        </Text>
      )}

      {/* ──── Attention 区域边框 ──── */}
      <Line
        points={[
          [-BW / 2 - 0.5, -BH / 2 - 0.5, pz(zQkv) - BD / 2 - 0.3],
          [BW / 2 + 0.5, -BH / 2 - 0.5, pz(zQkv) - BD / 2 - 0.3],
          [BW / 2 + 0.5, BH / 2 + 0.5, pz(zQkv) - BD / 2 - 0.3],
          [-BW / 2 - 0.5, BH / 2 + 0.5, pz(zQkv) - BD / 2 - 0.3],
          [-BW / 2 - 0.5, -BH / 2 - 0.5, pz(zQkv) - BD / 2 - 0.3],
        ]}
        color={attnActive ? '#60a5fa' : '#1e3050'}
        transparent
        opacity={attnActive ? 0.45 : 0.1}
        lineWidth={attnActive ? 2.0 : 0.8}
      />
      <Line
        points={[
          [-BW / 2 - 0.5, -BH / 2 - 0.5, pz(zAttnOut) + BD / 2 + 0.3],
          [BW / 2 + 0.5, -BH / 2 - 0.5, pz(zAttnOut) + BD / 2 + 0.3],
          [BW / 2 + 0.5, BH / 2 + 0.5, pz(zAttnOut) + BD / 2 + 0.3],
          [-BW / 2 - 0.5, BH / 2 + 0.5, pz(zAttnOut) + BD / 2 + 0.3],
          [-BW / 2 - 0.5, -BH / 2 - 0.5, pz(zAttnOut) + BD / 2 + 0.3],
        ]}
        color={attnActive ? '#60a5fa' : '#1e3050'}
        transparent
        opacity={attnActive ? 0.45 : 0.1}
        lineWidth={attnActive ? 2.0 : 0.8}
      />
      {/* 连接前后框的四条棱 */}
      {[
        [-BW / 2 - 0.5, -BH / 2 - 0.5],
        [BW / 2 + 0.5, -BH / 2 - 0.5],
        [BW / 2 + 0.5, BH / 2 + 0.5],
        [-BW / 2 - 0.5, BH / 2 + 0.5],
      ].map(([px, py], i) => (
        <Line
          key={`attn-edge-${i}`}
          points={[[px, py, pz(zQkv) - BD / 2 - 0.3], [px, py, pz(zAttnOut) + BD / 2 + 0.3]]}
          color={attnActive ? '#60a5fa' : '#1e3050'}
          transparent
          opacity={attnActive ? 0.45 : 0.1}
          lineWidth={attnActive ? 2.0 : 0.8}
        />
      ))}
      <Text
        position={[-BW / 2 - 1.2, 0, (pz(zQkv) + pz(zAttnOut)) / 2]}
        fontSize={0.26}
        color={attnActive ? '#60a5fa' : '#334466'}
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.012}
        outlineColor="#0a1022"
        rotation={[0, -Math.PI / 2, 0]}
      >
        Attention
      </Text>

      {/* ──── FFN 区域边框 ──── */}
      <Line
        points={[
          [-BW / 2 - 0.5, -BH / 2 - 0.5, pz(zFfnUp) - WIDE_D / 2 - 0.3],
          [BW / 2 + 0.5, -BH / 2 - 0.5, pz(zFfnUp) - WIDE_D / 2 - 0.3],
          [BW / 2 + 0.5, BH / 2 + 0.5, pz(zFfnUp) - WIDE_D / 2 - 0.3],
          [-BW / 2 - 0.5, BH / 2 + 0.5, pz(zFfnUp) - WIDE_D / 2 - 0.3],
          [-BW / 2 - 0.5, -BH / 2 - 0.5, pz(zFfnUp) - WIDE_D / 2 - 0.3],
        ]}
        color={ffnActive ? '#f59e0b' : '#2a2510'}
        transparent
        opacity={ffnActive ? 0.45 : 0.1}
        lineWidth={ffnActive ? 2.0 : 0.8}
      />
      <Line
        points={[
          [-BW / 2 - 0.5, -BH / 2 - 0.5, pz(zFfnDown) + BD / 2 + 0.3],
          [BW / 2 + 0.5, -BH / 2 - 0.5, pz(zFfnDown) + BD / 2 + 0.3],
          [BW / 2 + 0.5, BH / 2 + 0.5, pz(zFfnDown) + BD / 2 + 0.3],
          [-BW / 2 - 0.5, BH / 2 + 0.5, pz(zFfnDown) + BD / 2 + 0.3],
          [-BW / 2 - 0.5, -BH / 2 - 0.5, pz(zFfnDown) + BD / 2 + 0.3],
        ]}
        color={ffnActive ? '#f59e0b' : '#2a2510'}
        transparent
        opacity={ffnActive ? 0.45 : 0.1}
        lineWidth={ffnActive ? 2.0 : 0.8}
      />
      {[
        [-BW / 2 - 0.5, -BH / 2 - 0.5],
        [BW / 2 + 0.5, -BH / 2 - 0.5],
        [BW / 2 + 0.5, BH / 2 + 0.5],
        [-BW / 2 - 0.5, BH / 2 + 0.5],
      ].map(([px, py], i) => (
        <Line
          key={`ffn-edge-${i}`}
          points={[[px, py, pz(zFfnUp) - WIDE_D / 2 - 0.3], [px, py, pz(zFfnDown) + BD / 2 + 0.3]]}
          color={ffnActive ? '#f59e0b' : '#2a2510'}
          transparent
          opacity={ffnActive ? 0.45 : 0.1}
          lineWidth={ffnActive ? 2.0 : 0.8}
        />
      ))}
      <Text
        position={[-BW / 2 - 1.2, 0, (pz(zFfnUp) + pz(zFfnDown)) / 2]}
        fontSize={0.26}
        color={ffnActive ? '#f59e0b' : '#554422'}
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.012}
        outlineColor="#0a1022"
        rotation={[0, -Math.PI / 2, 0]}
      >
        FFN
      </Text>

      {/* ──── 数据流连接线（Z轴方向） ──── */}
      <ConnectorLine from={[0, 0, pz(zInput) + BD/2]} to={[0, 0, pz(zLn1) - THIN_D/2]} color="#818cf8" active={isPhase('input') || isPhase('ln1')} />
      <ConnectorLine from={[0, 0, pz(zLn1) + THIN_D/2]} to={[0, 0, pz(zQkv) - BD/2]} color="#60a5fa" active={isPhase('qkv')} />
      <ConnectorLine from={[0, 0, pz(zQkv) + BD/2]} to={[0, 0, pz(zAttnScore) - BD/2]} color="#38bdf8" active={isPhase('attn_score')} />
      <ConnectorLine from={[0, 0, pz(zAttnScore) + BD/2]} to={[0, 0, pz(zSoftmax) - THIN_D/2]} color="#22d3ee" active={isPhase('softmax')} />
      <ConnectorLine from={[0, 0, pz(zSoftmax) + THIN_D/2]} to={[0, 0, pz(zAttnOut) - BD/2]} color="#2dd4bf" active={isPhase('attn_out')} />
      <ConnectorLine from={[0, 0, pz(zAttnOut) + BD/2]} to={[0, 0, pz(zRes1) - THIN_D/2]} color="#a78bfa" active={isPhase('residual1')} />
      <ConnectorLine from={[0, 0, pz(zRes1) + THIN_D/2]} to={[0, 0, pz(zLn2) - THIN_D/2]} color="#818cf8" active={isPhase('ln2')} />
      <ConnectorLine from={[0, 0, pz(zLn2) + THIN_D/2]} to={[0, 0, pz(zFfnUp) - WIDE_D/2]} color="#f59e0b" active={isPhase('ffn_up')} />
      <ConnectorLine from={[0, 0, pz(zFfnUp) + WIDE_D/2]} to={[0, 0, pz(zFfnAct) - BD/2]} color="#fb923c" active={isPhase('ffn_act')} />
      <ConnectorLine from={[0, 0, pz(zFfnAct) + BD/2]} to={[0, 0, pz(zFfnDown) - BD/2]} color="#f97316" active={isPhase('ffn_down')} />
      <ConnectorLine from={[0, 0, pz(zFfnDown) + BD/2]} to={[0, 0, pz(zRes2) - THIN_D/2]} color="#a78bfa" active={isPhase('residual2')} />
      <ConnectorLine from={[0, 0, pz(zRes2) + THIN_D/2]} to={[0, 0, pz(zOutput) - BD/2]} color="#34d399" active={isPhase('residual2')} />

      {/* ──── 残差连接旁路（左侧弧线） ──── */}
      {/* 残差1: Input → Res1 左侧弧线 */}
      <Line
        points={[
          [-BW / 2, 0, pz(zInput)],
          [-BW / 2 - 1.5, 0, pz(zInput)],
          [-BW / 2 - 1.5, 0, pz(zRes1)],
          [-BW / 2, 0, pz(zRes1)],
        ]}
        color={isPhase('residual1') ? '#a78bfa' : '#2a2050'}
        transparent
        opacity={isPhase('residual1') ? 0.8 : 0.15}
        lineWidth={isPhase('residual1') ? 2.5 : 1.0}
      />
      <Text
        position={[-BW / 2 - 1.8, 0, (pz(zInput) + pz(zRes1)) / 2]}
        fontSize={0.22}
        color={isPhase('residual1') ? '#c4b5fd' : '#3a3060'}
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.01}
        outlineColor="#0a1022"
        rotation={[0, -Math.PI / 2, 0]}
      >
        skip
      </Text>

      {/* 残差2: Res1 → Res2 左侧弧线（更远） */}
      <Line
        points={[
          [-BW / 2, 0, pz(zRes1)],
          [-BW / 2 - 2.5, 0, pz(zRes1)],
          [-BW / 2 - 2.5, 0, pz(zRes2)],
          [-BW / 2, 0, pz(zRes2)],
        ]}
        color={isPhase('residual2') ? '#a78bfa' : '#2a2050'}
        transparent
        opacity={isPhase('residual2') ? 0.8 : 0.15}
        lineWidth={isPhase('residual2') ? 2.5 : 1.0}
      />
      <Text
        position={[-BW / 2 - 2.8, 0, (pz(zRes1) + pz(zRes2)) / 2]}
        fontSize={0.22}
        color={isPhase('residual2') ? '#c4b5fd' : '#3a3060'}
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.01}
        outlineColor="#0a1022"
        rotation={[0, -Math.PI / 2, 0]}
      >
        skip
      </Text>

      {/* ──── 各模块3D块体（Z轴排列） ──── */}

      {/* 输入 */}
      <Block3D
        position={[0, 0, pz(zInput)]}
        color="#94a3b8"
        label="Input x"
        sublabel={`d=${dModel}`}
        active={isPhase('input')}
      />

      {/* LayerNorm1 */}
      <Block3D
        position={[0, 0, pz(zLn1)]}
        size={[BW, BH, THIN_D]}
        color="#818cf8"
        label="LN"
        sublabel="Pre-Attn"
        active={isPhase('ln1')}
      />

      {/* Q / K / V 投影 - 三列并排 */}
      <group position={[0, 0, pz(zQkv)]}>
        <Block3D position={[-1.3, BH * 0.65, 0]} size={[0.9, BH * 0.55, BD]} color="#60a5fa" label="Q" sublabel={`${nHeads}×${headDim}`} active={isPhase('qkv')} />
        <Block3D position={[0, BH * 0.65, 0]} size={[0.9, BH * 0.55, BD]} color="#38bdf8" label="K" sublabel={`${nHeads}×${headDim}`} active={isPhase('qkv')} />
        <Block3D position={[1.3, BH * 0.65, 0]} size={[0.9, BH * 0.55, BD]} color="#2dd4bf" label="V" sublabel={`${nHeads}×${headDim}`} active={isPhase('qkv')} />
      </group>

      {/* 注意力分数 + Head Grid */}
      <group position={[0, 0, pz(zAttnScore)]}>
        <Block3D
          position={[0, 0, 0]}
          size={[BW, BH, BD]}
          color="#38bdf8"
          label="Q·Kᵀ/√d"
          sublabel={`${nHeads}×seq²`}
          active={isPhase('attn_score')}
          emissiveBoost={0.5}
        />
        <AttentionHeadGrid3D
          position={[0, 0, 0]}
          nHeads={nHeads}
          activeHeads={activeHeads}
          headActivations={headActivations}
          active={isPhase('attn_score')}
        />
        {/* 激活摘要标签 */}
        {isPhase('attn_score') && headActivations.length > 0 && (() => {
          const maxAct = Math.max(...headActivations);
          const avgAct = headActivations.reduce((a, b) => a + b, 0) / headActivations.length;
          const highCount = headActivations.filter(a => a > 0.8).length;
          return (
            <group position={[-BW / 2 - 0.3, -BH / 2 - 0.3, 0]}>
              <Text
                position={[0, -0.2, 0]}
                fontSize={0.16}
                color={activationColor(maxAct)}
                anchorX="right"
                anchorY="top"
                outlineWidth={0.008}
                outlineColor="#0a1022"
              >
                {`max=${maxAct.toFixed(2)} avg=${avgAct.toFixed(2)}`}
              </Text>
              <Text
                position={[0, -0.5, 0]}
                fontSize={0.14}
                color="#ef4444"
                anchorX="right"
                anchorY="top"
                outlineWidth={0.008}
                outlineColor="#0a1022"
              >
                {`>${'0.8'}: ${highCount}/${headActivations.length} heads`}
              </Text>
            </group>
          );
        })()}
      </group>

      {/* Softmax */}
      <Block3D
        position={[0, 0, pz(zSoftmax)]}
        size={[BW, BH, THIN_D]}
        color="#22d3ee"
        label="Softmax"
        sublabel="row-wise"
        active={isPhase('softmax')}
      />

      {/* Attn·V → Wₒ */}
      <Block3D
        position={[0, 0, pz(zAttnOut)]}
        color="#2dd4bf"
        label="Attn·V → Wₒ"
        sublabel={`→ d=${dModel}`}
        active={isPhase('attn_out')}
      />

      {/* 残差1 */}
      <ResidualRing3D position={[0, 0, pz(zRes1)]} active={isPhase('residual1')} />

      {/* LayerNorm2 */}
      <Block3D
        position={[0, 0, pz(zLn2)]}
        size={[BW, BH, THIN_D]}
        color="#818cf8"
        label="LN"
        sublabel="Pre-FFN"
        active={isPhase('ln2')}
      />

      {/* FFN Up + Neuron Bars */}
      <group position={[0, 0, pz(zFfnUp)]}>
        <Block3D
          position={[0, 0, 0]}
          size={[BW, BH, WIDE_D]}
          color="#f59e0b"
          label="W_up"
          sublabel={`${dModel}→${mlpDim?.toLocaleString()}`}
          active={isPhase('ffn_up')}
          emissiveBoost={0.3}
        />
        {ffnNeurons.length > 0 && (
          <FFNNeuronBars3D
            position={[0, 0, 0]}
            neurons={ffnNeurons}
            active={isPhase('ffn_up')}
          />
        )}
        {/* FFN 激活摘要 */}
        {isPhase('ffn_up') && ffnNeurons.length > 0 && (() => {
          const maxAct = Math.max(...ffnNeurons.map(n => n.activation));
          const avgAct = ffnNeurons.reduce((a, n) => a + n.activation, 0) / ffnNeurons.length;
          const highCount = ffnNeurons.filter(n => n.activation > 0.8).length;
          return (
            <group position={[-BW / 2 - 0.3, -BH / 2 - 0.3, 0]}>
              <Text
                position={[0, -0.2, 0]}
                fontSize={0.16}
                color={activationColor(maxAct)}
                anchorX="right"
                anchorY="top"
                outlineWidth={0.008}
                outlineColor="#0a1022"
              >
                {`max=${maxAct.toFixed(2)} avg=${avgAct.toFixed(2)}`}
              </Text>
              <Text
                position={[0, -0.5, 0]}
                fontSize={0.14}
                color="#ef4444"
                anchorX="right"
                anchorY="top"
                outlineWidth={0.008}
                outlineColor="#0a1022"
              >
                {`>0.8: ${highCount}/${ffnNeurons.length} neurons`}
              </Text>
            </group>
          );
        })()}
      </group>

      {/* SiLU 激活 */}
      {(() => {
        const gateVal = layerData?.ffn?.gate_activation ?? (isActive ? 0.72 : 0);
        const gateColor = activationColor(gateVal);
        return (
          <Block3D
            position={[0, 0, pz(zFfnAct)]}
            color={isPhase('ffn_act') ? gateColor : '#fb923c'}
            label="SiLU"
            sublabel={`gate=${gateVal.toFixed(2)}`}
            active={isPhase('ffn_act')}
          />
        );
      })()}

      {/* FFN Down */}
      <Block3D
        position={[0, 0, pz(zFfnDown)]}
        color="#f97316"
        label="W_down"
        sublabel={`${mlpDim?.toLocaleString()}→${dModel}`}
        active={isPhase('ffn_down')}
      />

      {/* 残差2 */}
      <ResidualRing3D position={[0, 0, pz(zRes2)]} active={isPhase('residual2')} />

      {/* 输出 */}
      <Block3D
        position={[0, 0, pz(zOutput)]}
        color="#34d399"
        label="Output"
        sublabel={`‖r‖=${layerData?.residual_norm?.toFixed(1) || '-'}`}
        active={isActive && !currentPhase}
      />
    </group>
  );
}
