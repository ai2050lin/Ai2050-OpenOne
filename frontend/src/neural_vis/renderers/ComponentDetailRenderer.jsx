/**
 * ComponentDetailRenderer — 层组件详情3D模型 (v2.0)
 * 
 * 功能: 在layer旁边显示独立的3D模型, 展示当前层对应的
 * 内部组件(LN, W_U, W_U⊥, Attn, FFN, Residual)及详细参数
 * 
 * v2.0: 大幅放大尺寸, 使用基础Three.js确保可靠渲染
 */
import React, { useRef, useMemo } from 'react';
import * as THREE from 'three';
import { Text } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import {
  PLANE_SIZE, SUBSPACE_COLORS, COMPONENT_TYPES,
  layerToFuncColor, layerToFuncLabel,
} from '../utils/constants';

const LAYER_GAP = 2.0;
const DETAIL_X_OFFSET = PLANE_SIZE / 2 + 8;
const PANEL_W = 8;
const PANEL_H = 14;
const PANEL_D = 0.3;

// ==================== 组件详情数据生成 ====================
function getLayerComponentData(layer, nLayers = 36) {
  const ratio = layer / (nLayers - 1);
  const lnBeta = 0.98 + 0.02 * Math.sin(ratio * Math.PI);
  const lnLeakage = ratio < 0.33 ? 0.02 : ratio < 0.61 ? 0.08 + 0.04 * ratio : 0.15 - 0.05 * (1 - ratio);
  const wuSignal = ratio < 0.14 ? 0.05 : ratio < 0.61 ? 0.05 + 0.55 * ((ratio - 0.14) / 0.47) : 0.6 + 0.35 * ((ratio - 0.61) / 0.39);
  const wuPerpSignal = ratio < 0.33 ? 0.15 + 0.25 * (ratio / 0.33) : ratio < 0.61 ? 0.40 - 0.15 * ((ratio - 0.33) / 0.28) : 0.25 - 0.20 * ((ratio - 0.61) / 0.39);
  const residualRetention = 0.62 + 0.09 * Math.sin(ratio * Math.PI * 2);
  const attnStrength = ratio < 0.14 ? 0.15 : ratio < 0.61 ? 0.15 + 0.60 * ((ratio - 0.14) / 0.47) : 0.75 + 0.20 * ((ratio - 0.61) / 0.39);
  const ffnGain = ratio < 0.33 ? 0.3 : ratio < 0.61 ? 0.3 + 1.5 * ((ratio - 0.33) / 0.28) : 1.8 + 0.9 * ((ratio - 0.61) / 0.39);
  const darkMatterRatio = 0.86 + 0.06 * Math.sin(ratio * Math.PI);
  const logicSignal = Math.exp(-0.5 * Math.pow((ratio - 0.5) / 0.1, 2)) * 2.7;
  const cosWu = ratio < 0.14 ? 0.01 : ratio < 0.61 ? 0.01 + 0.79 * ((ratio - 0.14) / 0.47) : 0.80 + 0.15 * ((ratio - 0.61) / 0.39);
  const norm = 3.0 + 7.0 * ratio + 2.0 * Math.sin(ratio * Math.PI);

  return {
    layer,
    layerLabel: layerToFuncLabel(layer, nLayers),
    layerColor: layerToFuncColor(layer, nLayers),
    ln: { beta: lnBeta, leakage: lnLeakage },
    wu: { signal: wuSignal, cosWithWu: cosWu },
    wuPerp: { signal: wuPerpSignal },
    residual: { retention: residualRetention },
    attention: { strength: attnStrength, heads: 12, topPattern: ratio < 0.33 ? 'induction' : ratio < 0.61 ? 'semantic' : 'copy' },
    ffn: { gain: ffnGain, direction: -0.3 + 0.7 * ratio },
    darkMatter: { ratio: darkMatterRatio },
    logic: { signal: logicSignal },
    norm,
  };
}

// ==================== 信号条 (3D) ====================
function SignalBar3D({ value, maxValue, color, label, yPos, panelX }) {
  const barMaxWidth = PANEL_W * 0.65;
  const barHeight = 0.35;
  const fillWidth = Math.max(0.1, (value / maxValue) * barMaxWidth);
  const x0 = panelX + PANEL_W * 0.25;
  const z = 0.2;

  return (
    <group>
      {/* 背景条 */}
      <mesh position={[x0 + barMaxWidth / 2, yPos, z]}>
        <boxGeometry args={[barMaxWidth, barHeight, 0.08]} />
        <meshBasicMaterial color="#1e293b" transparent opacity={0.5} />
      </mesh>
      {/* 填充条 */}
      <mesh position={[x0 + fillWidth / 2, yPos, z + 0.02]}>
        <boxGeometry args={[fillWidth, barHeight, 0.12]} />
        <meshBasicMaterial color={color} transparent opacity={0.9} />
      </mesh>
      {/* 标签 */}
      <Text position={[x0 - 0.3, yPos, z + 0.1]} fontSize={0.28} color={color} anchorX="right" anchorY="middle">
        {label}
      </Text>
      {/* 百分比 */}
      <Text position={[x0 + barMaxWidth + 0.4, yPos, z + 0.1]} fontSize={0.24} color="#e2e8f0" anchorX="left" anchorY="middle">
        {(value * 100).toFixed(1)}%
      </Text>
    </group>
  );
}

// ==================== 组件行 ====================
function ComponentRow({ label, value, unit, detail, color, yPos, panelX, active }) {
  const x0 = panelX + 0.5;
  const rowH = 1.0;

  return (
    <group>
      {/* 行背景 */}
      <mesh position={[panelX + PANEL_W / 2, yPos, 0.05]}>
        <boxGeometry args={[PANEL_W - 0.4, rowH * 0.85, 0.1]} />
        <meshStandardMaterial
          color={active ? color : '#1e293b'}
          transparent
          opacity={active ? 0.2 : 0.08}
          emissive={active ? color : '#000000'}
          emissiveIntensity={active ? 0.15 : 0}
        />
      </mesh>
      {/* 组件名 */}
      <Text position={[x0, yPos + 0.15, 0.15]} fontSize={0.35} color={active ? color : '#475569'} anchorX="left" anchorY="middle">
        {label}
      </Text>
      {/* 主值 */}
      <Text position={[panelX + PANEL_W - 0.5, yPos + 0.15, 0.15]} fontSize={0.38} color={active ? '#ffffff' : '#64748b'} anchorX="right" anchorY="middle">
        {value}{unit || ''}
      </Text>
      {/* 详情行 */}
      {detail && (
        <Text position={[panelX + PANEL_W - 0.5, yPos - 0.25, 0.15]} fontSize={0.22} color={active ? '#94a3b8' : '#334155'} anchorX="right" anchorY="middle">
          {detail}
        </Text>
      )}
    </group>
  );
}

// ==================== 脉冲光环 ====================
function PulseRing({ y, color }) {
  const ringRef = useRef();
  useFrame((state) => {
    if (ringRef.current) {
      const t = state.clock.elapsedTime;
      ringRef.current.material.opacity = 0.3 + 0.2 * Math.sin(t * 3);
      const s = 1 + 0.01 * Math.sin(t * 2.5);
      ringRef.current.scale.set(s, s, s);
    }
  });
  return (
    <mesh ref={ringRef} position={[DETAIL_X_OFFSET + PANEL_W / 2, y, 0]} rotation={[0, 0, 0]}>
      <torusGeometry args={[PANEL_W * 0.55, 0.08, 8, 64]} />
      <meshBasicMaterial color={color} transparent opacity={0.4} />
    </mesh>
  );
}

// ==================== 连接线 ====================
function ConnectorBeam({ fromY, color }) {
  const beamRef = useRef();
  useFrame((state) => {
    if (beamRef.current) {
      const t = state.clock.elapsedTime;
      beamRef.current.material.opacity = 0.3 + 0.15 * Math.sin(t * 4);
    }
  });
  const x1 = PLANE_SIZE / 2 + 1;
  const x2 = DETAIL_X_OFFSET - 0.5;
  const midX = (x1 + x2) / 2;
  const len = x2 - x1;

  return (
    <mesh ref={beamRef} position={[midX, fromY, 0]}>
      <boxGeometry args={[len, 0.15, 0.15]} />
      <meshBasicMaterial color={color} transparent opacity={0.4} />
    </mesh>
  );
}

// ==================== 层详情面板 ====================
function LayerDetailPanel({ layerData, y }) {
  const d = layerData;
  const panelX = DETAIL_X_OFFSET;

  const components = [
    { label: 'LayerNorm', value: d.ln.beta.toFixed(3), unit: ' β', detail: `Leakage ${(d.ln.leakage * 100).toFixed(1)}%`, color: COMPONENT_TYPES.layer_norm.color },
    { label: 'W_U', value: (d.wu.signal * 100).toFixed(1), unit: '%', detail: `cos(W_U)=${d.wu.cosWithWu.toFixed(3)}`, color: SUBSPACE_COLORS.w_u },
    { label: 'W_U⊥', value: (d.wuPerp.signal * 100).toFixed(1), unit: '%', detail: '', color: SUBSPACE_COLORS.w_u_perp },
    { label: 'Attention', value: d.attention.strength.toFixed(2), unit: '', detail: `${d.attention.topPattern} H${d.attention.heads}`, color: COMPONENT_TYPES.attention.color },
    { label: 'FFN', value: d.ffn.gain.toFixed(2), unit: 'x', detail: `dir=${d.ffn.direction.toFixed(2)}`, color: COMPONENT_TYPES.ffn.color },
    { label: 'Residual', value: (d.residual.retention * 100).toFixed(1), unit: '%', detail: '', color: COMPONENT_TYPES.residual.color },
    { label: 'DarkMatter', value: (d.darkMatter.ratio * 100).toFixed(1), unit: '%', detail: '', color: SUBSPACE_COLORS.dark_matter },
    { label: 'Logic', value: d.logic.signal.toFixed(2), unit: 'x', detail: d.logic.signal > 2.0 ? '▲ PEAK' : '', color: SUBSPACE_COLORS.logic },
  ];

  const rowH = 1.15;
  const totalH = components.length * rowH + 3.5;

  return (
    <group position={[0, y + totalH / 2 - 3, 0]}>
      {/* 大背景面板 */}
      <mesh position={[panelX + PANEL_W / 2, 0, -0.1]}>
        <boxGeometry args={[PANEL_W + 0.6, totalH + 0.6, PANEL_D]} />
        <meshStandardMaterial color="#0c1222" transparent opacity={0.85} />
      </mesh>
      {/* 边框 */}
      <mesh position={[panelX + PANEL_W / 2, 0, -0.05]}>
        <boxGeometry args={[PANEL_W + 0.8, totalH + 0.8, 0.05]} />
        <meshBasicMaterial color={d.layerColor} transparent opacity={0.35} wireframe />
      </mesh>

      {/* 标题: 层号 + 功能 */}
      <Text position={[panelX + 0.5, totalH / 2 - 0.3, 0.15]} fontSize={0.55} color={d.layerColor} anchorX="left" anchorY="middle">
        L{d.layer}
      </Text>
      <Text position={[panelX + 2.2, totalH / 2 - 0.3, 0.15]} fontSize={0.4} color="#94a3b8" anchorX="left" anchorY="middle">
        {d.layerLabel}
      </Text>

      {/* 分隔线 */}
      <mesh position={[panelX + PANEL_W / 2, totalH / 2 - 0.9, 0.05]}>
        <boxGeometry args={[PANEL_W - 0.6, 0.04, 0.04]} />
        <meshBasicMaterial color={d.layerColor} transparent opacity={0.5} />
      </mesh>

      {/* 组件行 */}
      {components.map((comp, i) => (
        <ComponentRow
          key={comp.label}
          {...comp}
          yPos={totalH / 2 - 1.5 - i * rowH}
          panelX={panelX}
          active={true}
        />
      ))}

      {/* W_U / W_U⊥ 信号条形对比 */}
      <mesh position={[panelX + PANEL_W / 2, -totalH / 2 + 2.8, 0.02]}>
        <boxGeometry args={[PANEL_W - 0.4, 0.04, 0.04]} />
        <meshBasicMaterial color="#334155" transparent opacity={0.3} />
      </mesh>
      <Text position={[panelX + 0.5, -totalH / 2 + 2.2, 0.15]} fontSize={0.3} color="#e2e8f0" anchorX="left" anchorY="middle">
        Subspace Signal
      </Text>
      <SignalBar3D value={d.wu.signal} maxValue={1} color={SUBSPACE_COLORS.w_u} label="W_U" yPos={-totalH / 2 + 1.5} panelX={panelX} />
      <SignalBar3D value={d.wuPerp.signal} maxValue={1} color={SUBSPACE_COLORS.w_u_perp} label="W_U⊥" yPos={-totalH / 2 + 0.9} panelX={panelX} />

      {/* 范数信息 */}
      <Text position={[panelX + 0.5, -totalH / 2 + 0.2, 0.15]} fontSize={0.22} color="#64748b" anchorX="left" anchorY="middle">
        {`||x||=${d.norm.toFixed(1)}  cos(W_U)=${d.wu.cosWithWu.toFixed(3)}`}
      </Text>
    </group>
  );
}

// ==================== 主渲染器 ====================
export default function ComponentDetailRenderer({
  nLayers = 36,
  activeLayerRange = null,
  highlightedLayer = null,
  animProgress = 1,
  activeScenario = null,
}) {
  const groupRef = useRef();

  const defaultLayer = Math.floor(nLayers / 2);
  const targetLayer = useMemo(() => {
    if (activeLayerRange) {
      return Math.floor((activeLayerRange[0] + activeLayerRange[1]) / 2);
    }
    if (highlightedLayer != null) {
      return highlightedLayer;
    }
    return defaultLayer;
  }, [activeLayerRange, highlightedLayer, defaultLayer]);

  const isActive = activeLayerRange != null || highlightedLayer != null;

  const layerData = useMemo(() => {
    return getLayerComponentData(targetLayer, nLayers);
  }, [targetLayer, nLayers]);

  return (
    <group ref={groupRef}>
      {/* 连接光束 */}
      <ConnectorBeam fromY={targetLayer * LAYER_GAP} color={layerData?.layerColor || '#3b82f6'} />

      {/* 脉冲光环 (动画时) */}
      {isActive && (
        <PulseRing y={targetLayer * LAYER_GAP} color={layerData?.layerColor || '#3b82f6'} />
      )}

      {/* 详情面板 */}
      {layerData && (
        <LayerDetailPanel
          layerData={layerData}
          y={targetLayer * LAYER_GAP}
        />
      )}
    </group>
  );
}
