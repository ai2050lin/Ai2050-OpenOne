/**
 * AppleNeuron3D 场景组件
 * 从 AppleNeuron3DTab.jsx 拆分而来
 */

import { Html, Line, OrbitControls, PerspectiveCamera, Text } from '@react-three/drei';
import { Canvas, useFrame } from '@react-three/fiber';
import { useMemo, useRef } from 'react';
import { ANIMATION_SCENARIOS, DIMENSION_VIEWS } from '../../config/panels';
import { LAYER_PARAMETER_STATE_ORDER, LAYER_PARAMETER_STATE_OVERLAY } from '../data/layer_parameter_state_overlay_persisted_v1';

import {
  LAYER_COUNT, DFF, ROLE_COLORS, DIMENSION_LABELS,
  MODE_VISUALS, APPLE_ANIMATION_OPTIONS,
  DEFAULT_LANGUAGE_FOCUS, MODEL_CONFIGS,
} from './constants';

import {
  toSafeNumber, neuronToPosition,
  averagePosition, blendPosition, shiftPosition, normalizeVector,
  buildAnimationSceneProfile,
} from './utils';

import LayerExplodedView3D from './LayerExplodedView3D';
import ComponentDetailPanel3D from './ComponentDetailPanel3D';

function PulsingNeuron({
  node,
  selected,
  onSelect,
  predictionStrength = 0,
  mode = 'static',
  isEffectiveNode = false,
  visibilityEmphasis = 1,
  motionEnabled = false,
  forwardPassReached = true,
  forwardPassActive = false,
}) {
  const ref = useRef(null);
  const modeStyle = MODE_VISUALS[mode] || MODE_VISUALS.static;

  useFrame((state) => {
    if (!ref.current) {
      return;
    }
    if (!motionEnabled) {
      const stableScale = node.size * (selected ? 1.12 : isEffectiveNode ? 1.08 : 1);
      ref.current.scale.set(stableScale, stableScale, stableScale);
      return;
    }
    const pulse = (node.role === 'background' ? 0.04 : 0.14) * modeStyle.nodePulse;
    const speed = (node.role === 'background' ? 1.2 : 2.1) * modeStyle.nodeSpeed;
    const base = node.size;
    const predictionBoost = predictionStrength * (node.role === 'background' ? 0.18 : 0.5) * (0.6 + 0.4 * visibilityEmphasis);
    const modeWave = mode === 'counterfactual' ? Math.sin(state.clock.elapsedTime * speed * 0.7 + node.phase * 1.3) * 0.06 : 0;
    const effectiveBoost = isEffectiveNode ? 0.22 : 0;
    const scale = base * (1 + Math.sin(state.clock.elapsedTime * speed + node.phase) * pulse + predictionBoost + modeWave + effectiveBoost);
    ref.current.scale.set(scale, scale, scale);
  });

  // Forward pass 状态影响
  const fpDimFactor = forwardPassReached ? 1.0 : 0.12;
  const fpActiveBoost = forwardPassActive ? 1.5 : 1.0;
  const fpColor = forwardPassActive ? '#4facfe' : null;

  return (
    <mesh
      ref={ref}
      position={node.position}
      onClick={(e) => {
        e.stopPropagation();
        onSelect(node);
      }}
    >
      <sphereGeometry args={[1, 20, 20]} />
      <meshStandardMaterial
        color={fpColor || (isEffectiveNode ? '#ffffff' : predictionStrength > 0.66 && mode !== 'static' ? modeStyle.accent : node.color)}
        emissive={fpColor || (isEffectiveNode ? '#ffffff' : predictionStrength > 0.5 && mode !== 'static' ? modeStyle.accent : node.color)}
        emissiveIntensity={
          ((selected ? 1.8 : node.role === 'background' ? 0.08 : 0.55)
          + predictionStrength * (node.role === 'background' ? 0.2 : 1.6)
          + (isEffectiveNode ? 0.95 : 0)
          + (mode !== 'static' ? 0.12 : 0)
          + (forwardPassActive ? 1.2 : 0))
          * fpActiveBoost
        }
        roughness={0.2}
        metalness={0.15}
        transparent
        opacity={(isEffectiveNode ? 0.98 : node.role === 'background' ? 0.24 + predictionStrength * 0.08 : 0.92) * visibilityEmphasis * fpDimFactor}
      />
    </mesh>
  );
}


function LayerEffectiveNeuronOverlay({ prediction = null, mode = 'static' }) {
  if (mode !== 'feature_decomposition') {
    return null;
  }
  const layer = Number.isFinite(prediction?.effectiveLayer)
    ? Math.max(0, Math.min(LAYER_COUNT - 1, Math.round(prediction.effectiveLayer)))
    : null;
  if (!Number.isFinite(layer)) {
    return null;
  }
  const rows = Array.isArray(prediction?.effectiveNeurons) ? prediction.effectiveNeurons.slice(0, 6) : [];
  const z = (layer - (LAYER_COUNT - 1) / 2) * 0.92;
  return (
    <group position={[0, 0, z]}>
      <Line points={[[3.8, 0.2, 0], [7.05, 1.95, 0]]} color="#ffffff" transparent opacity={0.82} lineWidth={1.4} />
      <Html position={[7.25, 2.12, 0]} center={false}>
        <div
          style={{
            width: 226,
            borderRadius: 10,
            border: '1px solid rgba(255,255,255,0.58)',
            background: 'rgba(8, 12, 24, 0.86)',
            color: '#e8f2ff',
            padding: '8px 10px',
            fontSize: 11,
            lineHeight: 1.55,
            boxShadow: '0 10px 24px rgba(0,0,0,0.35)',
            pointerEvents: 'none',
          }}
        >
          <div style={{ fontWeight: 700, color: '#ffffff' }}>{`L${layer} 有效神经元 Top-${rows.length}`}</div>
          {rows.length === 0 ? (
            <div style={{ color: '#9bb3de' }}>当前层暂无可显示节点</div>
          ) : (
            rows.map((item, idx) => (
              <div key={`eff-n-${item.id}-${idx}`} style={{ color: '#d4e5ff' }}>
                {`${idx + 1}. N${item.neuron} | ${item.role} | ${(Number(item.score || 0) * 100).toFixed(1)}%`}
              </div>
            ))
          )}
        </div>
      </Html>
    </group>
  );
}

function buildParameterStateSelection(point, position, layerKey, sourceDataPath) {
  return {
    id: `parameter-state-${point.id}`,
    label: point.label,
    role: 'route',
    category: point.category,
    layer: point.layer,
    neuron: point.neuron,
    metric: point.metric,
    value: point.value,
    strength: point.strength,
    source: point.sourceStage,
    sourceStage: point.sourceStage,
    outputDir: point.outputDir,
    parameterIds: point.parameterIds,
    dimIndex: point.dimIndex,
    sourceEntityId: point.sourceEntityId,
    sourceDataPath,
    detailType: 'parameter_state',
    overlayLayer: layerKey,
    position,
  };
}

function ParameterStatePoint({
  point,
  color,
  selected = false,
  active = false,
  onSelect,
  overlayLayer,
  sourceDataPath,
  orderIndex = 0,
  motionEnabled = false,
}) {
  const ref = useRef(null);
  const position = useMemo(
    () => neuronToPosition(point.layer, point.neuron, 0.42 + orderIndex * 0.07),
    [orderIndex, point.layer, point.neuron]
  );
  const layerAnchor = useMemo(() => [-position[0], -position[1], 0], [position]);

  useFrame((state) => {
    if (!ref.current) {
      return;
    }
    if (!motionEnabled) {
      const stableScale = 0.42 + point.strength * 0.22 + (selected ? 0.24 : active ? 0.14 : 0);
      ref.current.scale.setScalar(stableScale);
      return;
    }
    const pulse = 1 + Math.sin(state.clock.elapsedTime * 2.8 + orderIndex * 0.45) * 0.12;
    const boost = selected ? 0.24 : active ? 0.14 : 0;
    const scale = 0.42 + point.strength * 0.22 + boost;
    ref.current.scale.setScalar(scale * pulse);
  });

  return (
    <group position={position}>
      <Line
        points={[layerAnchor, position]}
        color={active ? '#ffffff' : color}
        transparent
        opacity={selected ? 0.95 : active ? 0.8 : 0.5}
        lineWidth={active ? 2.8 : 2}
      />

      <mesh position={[0, 0, -0.02]} scale={[1.4, 1.4, 0.22]} renderOrder={78}>
        <cylinderGeometry args={[1, 1, 1, 16]} />
        <meshBasicMaterial color={active ? '#ffffff' : color} transparent opacity={selected ? 0.36 : 0.22} depthTest={false} toneMapped={false} />
      </mesh>

      <mesh
        ref={ref}
        renderOrder={80}
        onClick={(e) => {
          e.stopPropagation();
          onSelect(buildParameterStateSelection(point, position, overlayLayer, sourceDataPath));
        }}
      >
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial
          color={selected ? '#ffffff' : active ? '#f8fafc' : color}
          transparent
          opacity={0.95}
          depthTest={false}
          toneMapped={false}
        />
      </mesh>

      <mesh scale={[2.8, 2.8, 2.8]} renderOrder={79}>
        <sphereGeometry args={[1, 12, 12]} />
        <meshBasicMaterial color={active ? '#ffffff' : color} transparent opacity={selected ? 0.28 : active ? 0.22 : 0.16} depthTest={false} toneMapped={false} />
      </mesh>

      <Text position={[0, 1.05, 0]} color="#ffffff" fontSize={0.32} anchorX="center" anchorY="bottom" renderOrder={81}>
        {`d${point.dimIndex}`}
      </Text>
      <Text position={[0, -0.92, 0]} color={active ? '#ffffff' : '#dbeafe'} fontSize={0.18} anchorX="center" anchorY="top" renderOrder={81}>
        {`L${point.layer}`}
      </Text>
    </group>
  );
}

function ParameterStateSummaryOverlay({ profile }) {
  if (!profile || !Array.isArray(profile.nodes) || profile.nodes.length === 0) {
    return null;
  }
  return (
    <Html position={[10.8, 8.4, 0]} transform sprite>
      <div
        style={{
          minWidth: 220,
          padding: '10px 12px',
          borderRadius: 12,
          background: 'rgba(7, 12, 25, 0.84)',
          border: `1px solid ${profile.color || '#60a5fa'}`,
          boxShadow: `0 0 18px ${(profile.color || '#60a5fa')}33`,
          color: '#e5eeff',
          pointerEvents: 'none',
        }}
      >
        <div style={{ fontSize: 12, fontWeight: 800, marginBottom: 6 }}>{`${profile.label} 参数节点`}</div>
        <div style={{ display: 'grid', gap: 4 }}>
          {profile.nodes.map((node) => (
            <div key={`param-summary-${node.id}`} style={{ fontSize: 11, lineHeight: 1.5 }}>
              {`L${node.layer} · d${node.dimIndex} · ${Number(node.value || 0).toFixed(4)}`}
            </div>
          ))}
        </div>
      </div>
    </Html>
  );
}

function buildParameterRackPosition(layer, orderIndex = 0, totalInLayer = 1) {
  const safeTotal = Math.max(1, totalInLayer);
  const rowIndex = orderIndex % safeTotal;
  const x = 10.6 + Math.floor(orderIndex / 4) * 0.92;
  const y = 2.2 - rowIndex * 1.08;
  const z = (layer - (LAYER_COUNT - 1) / 2) * 0.92;
  return [x, y, z];
}

function ParameterRackOverlay({ profile, selected = null, onSelect = () => {} }) {
  if (!profile || !Array.isArray(profile.nodes) || profile.nodes.length === 0) {
    return null;
  }

  const groupedCounts = profile.nodes.reduce((acc, node) => {
    const key = Number(node?.layer);
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});

  const seenPerLayer = {};

  return (
    <group>
      <Text position={[10.8, 5.9, 0]} color={profile.color || '#60a5fa'} fontSize={0.24} anchorX="center" anchorY="middle">
        {'参数机架'}
      </Text>
      {profile.nodes.map((node, idx) => {
        const layer = Number(node?.layer) || 0;
        const orderIndex = seenPerLayer[layer] || 0;
        seenPerLayer[layer] = orderIndex + 1;
        const rackPosition = buildParameterRackPosition(layer, orderIndex, groupedCounts[layer] || 1);
        const neuronPosition = neuronToPosition(node.layer, node.neuron, 0.42 + idx * 0.07);
        const isSelected = selected?.id === `parameter-state-${node.id}`;
        const accent = isSelected ? '#ffffff' : (profile.color || '#60a5fa');

        return (
          <group key={`parameter-rack-${node.id}`}>
            <Line
              points={[rackPosition, neuronPosition]}
              color={accent}
              transparent
              opacity={isSelected ? 0.92 : 0.48}
              lineWidth={isSelected ? 2.6 : 1.6}
            />
            <group
              position={rackPosition}
              onClick={(e) => {
                e.stopPropagation();
                onSelect(buildParameterStateSelection(node, rackPosition, profile.layerKey || 'static_encoding', profile.sourceDataPath));
              }}
            >
              <mesh renderOrder={86}>
                <boxGeometry args={[0.72, 0.72, 0.72]} />
                <meshBasicMaterial color={accent} transparent opacity={0.95} depthTest={false} toneMapped={false} />
              </mesh>
              <mesh scale={[1.9, 1.9, 1.9]} renderOrder={85}>
                <sphereGeometry args={[1, 12, 12]} />
                <meshBasicMaterial color={accent} transparent opacity={isSelected ? 0.26 : 0.14} depthTest={false} toneMapped={false} />
              </mesh>
              <Text position={[0, 0.72, 0]} color="#ffffff" fontSize={0.22} anchorX="center" anchorY="bottom" renderOrder={87}>
                {`d${node.dimIndex}`}
              </Text>
              <Text position={[0, -0.68, 0]} color="#dbeafe" fontSize={0.15} anchorX="center" anchorY="top" renderOrder={87}>
                {`L${node.layer}`}
              </Text>
            </group>
          </group>
        );
      })}
    </group>
  );
}

function LayerParameterStateOverlay({
  languageFocus = DEFAULT_LANGUAGE_FOCUS,
  selected = null,
  onSelect = () => {},
  activeIndex = -1,
  isPlaying = false,
}) {
  const layerKey = LAYER_PARAMETER_STATE_ORDER.includes(languageFocus?.researchLayer)
    ? languageFocus.researchLayer
    : 'static_encoding';
  const baseProfile = LAYER_PARAMETER_STATE_OVERLAY[layerKey] || LAYER_PARAMETER_STATE_OVERLAY.static_encoding;
  const profile = useMemo(() => ({ ...baseProfile, layerKey }), [baseProfile, layerKey]);
  const color = profile.color || '#60a5fa';
  const nodePositionMap = useMemo(
    () => Object.fromEntries(
      profile.nodes.map((point, idx) => [
        point.id,
        neuronToPosition(point.layer, point.neuron, 0.42 + idx * 0.07),
      ])
    ),
    [profile.nodes]
  );

  return (
    <group>
      <ParameterStateSummaryOverlay profile={profile} />
      <ParameterRackOverlay profile={profile} selected={selected} onSelect={onSelect} />
      <Text position={[0, 9.6, 0]} color={color} fontSize={0.26} anchorX="center" anchorY="middle">
        {`${profile.label} 参数态`}
      </Text>

      {profile.chains.map(([fromId, toId], chainIndex) => {
        const from = nodePositionMap[fromId];
        const to = nodePositionMap[toId];
        if (!from || !to) {
          return null;
        }
        const fromIndex = profile.nodes.findIndex((item) => item.id === fromId);
        const toIndex = profile.nodes.findIndex((item) => item.id === toId);
        const chainActive = activeIndex >= fromIndex && activeIndex <= toIndex;
        return (
          <group key={`parameter-chain-${fromId}-${toId}`}>
            <Line
              points={[from, to]}
              color={chainActive ? '#ffffff' : color}
              transparent
              opacity={chainActive ? 0.92 : 0.66}
              lineWidth={chainActive ? 2.8 : 1.6}
            />
            {isPlaying ? (
              <TheoryRunner
                path={[from, to]}
                color={chainActive ? '#ffffff' : color}
                size={chainActive ? 0.12 : 0.08}
                speed={0.26 + chainIndex * 0.04}
                phase={chainIndex * 0.18}
              />
            ) : null}
          </group>
        );
      })}

      {profile.nodes.map((point, idx) => (
        <ParameterStatePoint
          key={point.id}
          point={point}
          color={color}
          selected={selected?.id === `parameter-state-${point.id}`}
          active={idx === activeIndex}
          onSelect={onSelect}
          overlayLayer={layerKey}
          sourceDataPath={profile.sourceDataPath}
          orderIndex={idx}
          motionEnabled={isPlaying}
        />
      ))}

      {selected?.detailType === 'parameter_state' && selected?.overlayLayer === layerKey && Array.isArray(selected?.position) && (
        <Html position={[selected.position[0] + 1.2, selected.position[1] + 0.8, selected.position[2]]}>
          <div
            style={{
              width: 240,
              padding: '10px 12px',
              borderRadius: 10,
              background: 'rgba(8, 12, 24, 0.9)',
              border: `1px solid ${color}`,
              color: '#e5eeff',
              fontSize: 11,
              lineHeight: 1.55,
              boxShadow: '0 10px 24px rgba(0,0,0,0.28)',
              pointerEvents: 'none',
            }}
          >
            <div style={{ color: '#ffffff', fontWeight: 700, marginBottom: 6 }}>{selected.label}</div>
            <div>{`层 / 神经元: L${selected.layer} / N${selected.neuron}`}</div>
            <div>{`参数维度: d${selected.dimIndex}`}</div>
            <div>{`来源阶段: ${selected.sourceStage}`}</div>
            <div>{`指标: ${selected.metric} = ${Number(selected.value || 0).toFixed(4)}`}</div>
            <div>{`参数位: ${(selected.parameterIds || []).join(', ')}`}</div>
          </div>
        </Html>
      )}
    </group>
  );
}

function LayerBasicRuntimeControls({
  title = '参数态基础动画',
  onStart = () => {},
  onStop = () => {},
  onReplay = () => {},
  isPlaying = false,
}) {
  const buttonStyle = {
    borderRadius: 8,
    border: '1px solid rgba(148, 163, 184, 0.45)',
    background: 'rgba(15, 23, 42, 0.88)',
    color: '#e2e8f0',
    fontSize: 11,
    padding: '6px 10px',
    cursor: 'pointer',
  };

  return (
    <Html position={[-11.2, 8.9, 0]} transform sprite>
      <div
        style={{
          minWidth: 220,
          padding: '10px 12px',
          borderRadius: 12,
          background: 'rgba(8, 12, 24, 0.88)',
          border: '1px solid rgba(96, 165, 250, 0.45)',
          boxShadow: '0 12px 28px rgba(0,0,0,0.28)',
          color: '#e2e8f0',
          backdropFilter: 'blur(10px)',
        }}
      >
        <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>{title}</div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button type="button" onClick={onStart} style={buttonStyle}>
            开始动画
          </button>
          <button type="button" onClick={onStop} style={buttonStyle}>
            结束动画
          </button>
          <button type="button" onClick={onReplay} style={buttonStyle}>
            重新播放
          </button>
        </div>
        <div style={{ marginTop: 8, fontSize: 10, color: '#93c5fd' }}>
          {isPlaying ? '当前状态：播放中' : '当前状态：静止'}
        </div>
      </div>
    </Html>
  );
}


function ForwardPassLayerHighlight({ forwardPassLayer, forwardPassData, nLayers = LAYER_COUNT }) {
  if (forwardPassLayer == null || forwardPassLayer < 0) return null;
  const z = (forwardPassLayer - (nLayers - 1) / 2) * 0.92;

  // 获取当前层数据
  const layerData = forwardPassData?.[forwardPassLayer];
  const label = layerData?.label || `L${forwardPassLayer}`;

  return (
    <group position={[0, 0, z]}>
      {/* 当前层高亮矩形框 - 脉冲动画 */}
      <Line
        points={[
          [-8.2, -8.2, 0], [8.2, -8.2, 0], [8.2, 8.2, 0], [-8.2, 8.2, 0], [-8.2, -8.2, 0],
        ]}
        color="#4facfe"
        lineWidth={2.5}
        transparent
        opacity={0.8}
      />
      {/* 外层脉冲光环已移除 */}
      {/* 层号标签 */}
      <Text
        position={[-9.5, 0, 0]}
        fontSize={0.5}
        color="#4facfe"
        anchorX="right"
        anchorY="middle"
        outlineWidth={0.02}
        outlineColor="#0a1022"
      >
        L{forwardPassLayer}
      </Text>
      {/* 功能标签 */}
      <Text
        position={[9.5, 0, 0]}
        fontSize={0.42}
        color="#fff"
        anchorX="left"
        anchorY="middle"
        outlineWidth={0.02}
        outlineColor="#0a1022"
      >
        {label}
      </Text>
      {/* 神经元激活值球体 */}
      {layerData?.neuron_activations?.map((n, i) => {
        const act = n.activation || 0;
        const color = act > 0.8 ? '#ff4444' : act > 0.5 ? '#ffcc00' : act > 0.3 ? '#22c55e' : '#3b82f6';
        const size = (act > 0.8 ? 0.45 : act > 0.5 ? 0.35 : act > 0.3 ? 0.25 : 0.16) * 1.4;
        const emissive = act > 0.8 ? 2.2 : act > 0.5 ? 1.4 : act > 0.3 ? 0.7 : 0.25;
        // 神经元映射到层内平面位置
        const nx = (n.x || 0) * 1.2;
        const ny = (n.z || 0) * 1.2;
        return (
          <mesh key={`fpn-${i}`} position={[nx, ny, 0.3]}>
            <sphereGeometry args={[size, 16, 16]} />
            <meshStandardMaterial
              color={color}
              emissive={color}
              emissiveIntensity={emissive}
              transparent
              opacity={0.92}
              toneMapped={false}
            />
          </mesh>
        );
      })}
      {/* 激活值图例 */}
      <group position={[-10, 7, 0]}>
        {[
          { color: '#ff4444', label: '>0.8 强' },
          { color: '#ffcc00', label: '>0.5 中' },
          { color: '#22c55e', label: '>0.3 弱' },
          { color: '#3b82f6', label: '<0.3 微' },
        ].map((item, i) => (
          <group key={`legend-${i}`} position={[0, -i * 0.55, 0]}>
            <mesh>
              <sphereGeometry args={[0.12, 8, 8]} />
              <meshBasicMaterial color={item.color} />
            </mesh>
            <Text position={[0.3, 0, 0]} fontSize={0.2} color={item.color} anchorX="left" anchorY="middle">
              {item.label}
            </Text>
          </group>
        ))}
      </group>
    </group>
  );
}

function LayerGuides({ activeLayer = null, layerCount = LAYER_COUNT }) {
  const layers = useMemo(() => Array.from({ length: layerCount }, (_, i) => i), [layerCount]);
  const hasActiveLayer = Number.isFinite(activeLayer);
  const activeLayerIndex = hasActiveLayer
    ? Math.max(0, Math.min(layerCount - 1, Math.round(activeLayer)))
    : null;
  return (
    <group>
      {layers.map((layer) => {
        const z = (layer - (layerCount - 1) / 2) * 0.92;
        const isMajor = layer % 4 === 0 || layer === layerCount - 1;
        const isActive = activeLayerIndex === layer;
        const lineColor = isActive ? '#ffffff' : isMajor ? '#dbeafe' : '#8ea4c7';
        const lineOpacity = isActive ? 0.8 : isMajor ? 0.2 : 0.1;
        const labelColor = isActive ? '#ffffff' : isMajor ? '#d8ecff' : '#9cb6dc';
        const labelSize = isActive ? 0.38 : isMajor ? 0.3 : 0.22;
        return (
          <group key={`layer-${layer}`}>
            <Line
              points={[
                [-7.5, -7.5, z],
                [7.5, -7.5, z],
                [7.5, 7.5, z],
                [-7.5, 7.5, z],
                [-7.5, -7.5, z],
              ]}
              color={lineColor}
              transparent
              opacity={lineOpacity}
              lineWidth={1}
            />
            <Text
              position={[-8.55, 0, z]}
              color={labelColor}
              fontSize={labelSize}
              anchorX="left"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="#0a1022"
            >
              {`L${layer}`}
            </Text>
            <Text
              position={[8.55, 0, z]}
              color={labelColor}
              fontSize={labelSize}
              anchorX="right"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="#0a1022"
            >
              {`L${layer}`}
            </Text>
            {isActive && (
              <Line
                points={[
                  [-6.2, -6.2, z],
                  [6.2, -6.2, z],
                  [6.2, 6.2, z],
                  [-6.2, 6.2, z],
                  [-6.2, -6.2, z],
                ]}
                color="#ffffff"
                transparent
                opacity={0.58}
                lineWidth={1.6}
              />
            )}
          </group>
        );
      })}
      <Line points={[[0, 0, -13.2], [0, 0, 13.2]]} color="#ffffff" transparent opacity={0.7} lineWidth={1.2} />
      <Text position={[0, 0.95, -13.2]} color="#cde4ff" fontSize={0.28} anchorX="center" anchorY="middle" outlineWidth={0.015} outlineColor="#0a1022">
        Layer 0
      </Text>
      <Text position={[0, 0.95, 13.2]} color="#cde4ff" fontSize={0.28} anchorX="center" anchorY="middle" outlineWidth={0.015} outlineColor="#0a1022">
        Layer 27
      </Text>
    </group>
  );
}

function DimensionLayerImpactGraph({ profile = [], dimension = 'style', suppression = null }) {
  if (!Array.isArray(profile) || profile.length === 0) {
    return null;
  }
  const color = ROLE_COLORS[dimension] || '#84f1ff';
  const points = profile.map((v, layer) => {
    const z = (layer - (LAYER_COUNT - 1) / 2) * 0.92;
    const x = 8.2 + Math.max(0, toSafeNumber(v, 0)) * 4.8;
    const y = -6.45;
    return [x, y, z];
  });
  const diagAdv = suppression?.diagonal_advantage?.[dimension];
  const row = suppression?.suppression_matrix_mean?.[dimension];
  return (
    <group>
      <Line points={[[8.2, -6.45, -13.1], [8.2, -6.45, 13.1]]} color="#7f95bb" transparent opacity={0.28} lineWidth={1} />
      <Line points={points} color={color} transparent opacity={0.95} lineWidth={2.6} />
      {points.map((p, idx) => (
        <mesh key={`impact-${dimension}-${idx}`} position={p}>
          <sphereGeometry args={[0.045, 12, 12]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.9} />
        </mesh>
      ))}
      <Text position={[9.7, -5.8, 13.3]} color={color} fontSize={0.23} anchorX="right" anchorY="middle" outlineWidth={0.015} outlineColor="#0a1022">
        {`${DIMENSION_LABELS[dimension] || dimension} 层影响谱`}
      </Text>
      {Number.isFinite(diagAdv) && (
        <Text position={[9.7, -6.2, 13.3]} color="#cde4ff" fontSize={0.2} anchorX="right" anchorY="middle" outlineWidth={0.012} outlineColor="#0a1022">
          {`对角优势: ${diagAdv.toFixed(4)}`}
        </Text>
      )}
      {row ? (
        <Text position={[9.7, -6.55, 13.3]} color="#9eb4dd" fontSize={0.18} anchorX="right" anchorY="middle" outlineWidth={0.01} outlineColor="#0a1022">
          {`S/L/Y: ${toSafeNumber(row.style, 0).toFixed(3)} / ${toSafeNumber(row.logic, 0).toFixed(3)} / ${toSafeNumber(row.syntax, 0).toFixed(3)}`}
        </Text>
      ) : null}
    </group>
  );
}

function TokenPredictionCarrier({ prediction, mode = 'static' }) {
  const ref = useRef(null);
  const modeStyle = MODE_VISUALS[mode] || MODE_VISUALS.static;
  const movingColor = '#ffffff';

  useFrame((state) => {
    if (!ref.current) {
      return;
    }
    ref.current.rotation.y = state.clock.elapsedTime * (1.1 + modeStyle.nodeSpeed * 0.7);
  });

  if (!prediction?.currentToken || modeStyle.carrier === 'none') {
    return null;
  }

  const z = (prediction.layerProgress - 0.5) * (LAYER_COUNT - 1) * 0.92;
  const radius = 0.5 + prediction.currentToken.prob * 0.75;
  return (
    <group position={[0, 0, z]}>
      {modeStyle.carrier === 'torus' && (
        <mesh ref={ref}>
          <torusGeometry args={[radius, 0.08, 14, 42]} />
          <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.4} transparent opacity={0.75} />
        </mesh>
      )}
      {modeStyle.carrier === 'octa' && (
        <mesh ref={ref}>
          <octahedronGeometry args={[radius * 0.92]} />
          <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.2} transparent opacity={0.72} wireframe />
        </mesh>
      )}
      {modeStyle.carrier === 'plane' && (
        <mesh ref={ref} rotation={[0.55, 0.25, 0.15]}>
          <boxGeometry args={[radius * 1.95, 0.08, radius * 1.1]} />
          <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.0} transparent opacity={0.55} />
        </mesh>
      )}
      {modeStyle.carrier === 'tetra' && (
        <mesh ref={ref}>
          <tetrahedronGeometry args={[radius * 0.95]} />
          <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.2} transparent opacity={0.72} />
        </mesh>
      )}
      {modeStyle.carrier === 'cylinder' && (
        <mesh ref={ref}>
          <cylinderGeometry args={[radius * 0.22, radius * 0.22, radius * 2.0, 16]} />
          <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.15} transparent opacity={0.72} />
        </mesh>
      )}
      {modeStyle.carrier === 'tri_ring' && (
        <group ref={ref}>
          <mesh rotation={[0, 0, 0]}>
            <torusGeometry args={[radius * 0.9, 0.07, 12, 36]} />
            <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.2} transparent opacity={0.72} />
          </mesh>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[radius * 0.7, 0.07, 12, 36]} />
            <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.2} transparent opacity={0.54} />
          </mesh>
          <mesh rotation={[0, Math.PI / 2, 0]}>
            <torusGeometry args={[radius * 0.52, 0.07, 12, 36]} />
            <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.2} transparent opacity={0.4} />
          </mesh>
        </group>
      )}
      {modeStyle.carrier === 'dual_ring' && (
        <group ref={ref}>
          <mesh position={[-0.36, 0, 0]}>
            <torusGeometry args={[radius * 0.6, 0.07, 12, 36]} />
            <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.1} transparent opacity={0.68} />
          </mesh>
          <mesh position={[0.36, 0, 0]}>
            <torusGeometry args={[radius * 0.6, 0.07, 12, 36]} />
            <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.2} transparent opacity={0.78} />
          </mesh>
        </group>
      )}
      {modeStyle.carrier === 'shield' && (
        <mesh ref={ref}>
          <sphereGeometry args={[radius * 0.9, 20, 20]} />
          <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={0.95} transparent opacity={0.2} wireframe />
        </mesh>
      )}
      {modeStyle.carrier === 'hex' && (
        <mesh ref={ref}>
          <cylinderGeometry args={[radius * 0.8, radius * 0.8, radius * 0.95, 6]} />
          <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.2} transparent opacity={0.72} wireframe />
        </mesh>
      )}
      <Text position={[0, 0.9, 0]} color="#dff6ff" fontSize={0.34} anchorX="center" anchorY="middle">
        {`${prediction.currentToken.token} (${(prediction.currentToken.prob * 100).toFixed(1)}%)`}
      </Text>
    </group>
  );
}

function ModeVisualOverlay({ mode = 'static', prediction = null }) {
  const ref = useRef(null);
  const modeStyle = MODE_VISUALS[mode] || MODE_VISUALS.static;

  useFrame((state) => {
    if (!ref.current) {
      return;
    }
    ref.current.rotation.y = state.clock.elapsedTime * (0.25 + modeStyle.nodeSpeed * 0.2);
  });

  if (mode === 'static') {
    return null;
  }

  const z = ((prediction?.layerProgress ?? 0.5) - 0.5) * (LAYER_COUNT - 1) * 0.92;
  return (
    <group ref={ref} position={[0, 0, z]}>
      {mode === 'causal_intervention' && (
        <mesh>
          <torusKnotGeometry args={[1.2, 0.08, 120, 16]} />
          <meshStandardMaterial color={modeStyle.accent} emissive={modeStyle.accent} emissiveIntensity={0.95} transparent opacity={0.45} wireframe />
        </mesh>
      )}
      {mode === 'subspace_geometry' && (
        <mesh rotation={[0.62, 0.15, 0.42]}>
          <boxGeometry args={[3.6, 0.05, 1.6]} />
          <meshStandardMaterial color={modeStyle.accent} emissive={modeStyle.accent} emissiveIntensity={0.8} transparent opacity={0.28} />
        </mesh>
      )}
      {mode === 'feature_decomposition' && (
        <>
          <Line points={[[-1.9, 0, 0], [1.9, 0, 0]]} color="#f59e0b" transparent opacity={0.8} lineWidth={2} />
          <Line points={[[0, -1.9, 0], [0, 1.9, 0]]} color="#38bdf8" transparent opacity={0.8} lineWidth={2} />
          <Line points={[[0, 0, -1.9], [0, 0, 1.9]]} color="#a78bfa" transparent opacity={0.8} lineWidth={2} />
        </>
      )}
      {mode === 'cross_layer_transport' && (
        <>
          <Line points={[[0, 0, -2.8], [0, 0, 2.8]]} color={modeStyle.accent} transparent opacity={0.85} lineWidth={2} />
          <mesh position={[0, 0.2, Math.sin((prediction?.layerProgress || 0) * Math.PI * 2) * 2.2]}>
            <sphereGeometry args={[0.16, 12, 12]} />
            <meshStandardMaterial color={modeStyle.accent} emissive={modeStyle.accent} emissiveIntensity={1.35} />
          </mesh>
        </>
      )}
      {mode === 'compositionality' && (
        <>
          <mesh rotation={[0, 0, 0]}>
            <torusGeometry args={[1.2, 0.05, 12, 42]} />
            <meshStandardMaterial color="#34d399" emissive="#34d399" emissiveIntensity={1.0} transparent opacity={0.62} />
          </mesh>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[1.0, 0.05, 12, 42]} />
            <meshStandardMaterial color="#f59e0b" emissive="#f59e0b" emissiveIntensity={1.0} transparent opacity={0.62} />
          </mesh>
          <mesh rotation={[0, Math.PI / 2, 0]}>
            <torusGeometry args={[0.8, 0.05, 12, 42]} />
            <meshStandardMaterial color="#60a5fa" emissive="#60a5fa" emissiveIntensity={1.0} transparent opacity={0.62} />
          </mesh>
        </>
      )}
      {mode === 'counterfactual' && (
        <>
          <mesh position={[-0.8, 0, 0]}>
            <sphereGeometry args={[0.42, 16, 16]} />
            <meshStandardMaterial color="#fda4af" emissive="#fda4af" emissiveIntensity={1.05} transparent opacity={0.58} />
          </mesh>
          <mesh position={[0.8, 0, 0]}>
            <sphereGeometry args={[0.42, 16, 16]} />
            <meshStandardMaterial color="#fb7185" emissive="#fb7185" emissiveIntensity={1.2} transparent opacity={0.58} />
          </mesh>
          <Line points={[[-0.4, 0, 0], [0.4, 0, 0]]} color="#fda4af" transparent opacity={0.85} lineWidth={2} />
        </>
      )}
      {mode === 'robustness' && (
        <mesh>
          <sphereGeometry args={[1.45, 24, 24]} />
          <meshStandardMaterial color={modeStyle.accent} emissive={modeStyle.accent} emissiveIntensity={0.72} transparent opacity={0.16} wireframe />
        </mesh>
      )}
      {mode === 'minimal_circuit' && (
        <>
          <mesh>
            <cylinderGeometry args={[1.2, 1.2, 1.6, 6]} />
            <meshStandardMaterial color={modeStyle.accent} emissive={modeStyle.accent} emissiveIntensity={0.9} transparent opacity={0.26} wireframe />
          </mesh>
          <Line points={[[0, 0.8, 0], [0, -0.8, 0]]} color={modeStyle.accent} transparent opacity={0.9} lineWidth={2} />
        </>
      )}
    </group>
  );
}


function TheoryBeacon({
  position = [0, 0, 0],
  color = '#ffffff',
  size = 0.14,
  pulse = 0.18,
  speed = 1.2,
  phase = 0,
  opacity = 0.94,
}) {
  const ref = useRef(null);
  useFrame((state) => {
    if (!ref.current) {
      return;
    }
    const s = 1 + Math.sin(state.clock.elapsedTime * speed + phase) * pulse;
    ref.current.scale.set(size * s, size * s, size * s);
  });
  return (
    <mesh ref={ref} position={position}>
      <sphereGeometry args={[1, 14, 14]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.1} transparent opacity={opacity} />
    </mesh>
  );
}

function TheoryRunner({
  path = [],
  color = '#ffffff',
  size = 0.12,
  speed = 0.28,
  phase = 0,
}) {
  const ref = useRef(null);
  useFrame((state) => {
    if (!ref.current || path.length < 2) {
      return;
    }
    const t = (state.clock.elapsedTime * speed + phase) % 1;
    const scaled = t * (path.length - 1);
    const idx = Math.min(path.length - 2, Math.floor(scaled));
    const frac = scaled - idx;
    const pos = blendPosition(path[idx], path[idx + 1], frac);
    ref.current.position.set(pos[0], pos[1], pos[2]);
    const s = 1 + Math.sin(state.clock.elapsedTime * 3.2 + phase * 4) * 0.16;
    ref.current.scale.set(size * s, size * s, size * s);
  });
  return (
    <mesh ref={ref}>
      <sphereGeometry args={[1, 12, 12]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.2} />
    </mesh>
  );
}


function TheoryObjectOverlay({ theoryObjectMeta = null, prediction = null, nodes = [], selected = null }) {
  const ref = useRef(null);
  const accent = theoryObjectMeta?.color || '#7dd3fc';
  const label = theoryObjectMeta?.labelZh || '理论对象';
  const z = ((prediction?.layerProgress ?? 0.5) - 0.5) * (LAYER_COUNT - 1) * 0.92;
  const focusNodeSet = useMemo(() => new Set(prediction?.focusNodeIds || []), [prediction?.focusNodeIds]);
  const coreNodes = useMemo(() => nodes.filter((node) => node.role !== 'background'), [nodes]);
  const focusNodes = useMemo(() => coreNodes.filter((node) => focusNodeSet.has(node.id)), [coreNodes, focusNodeSet]);
  const familyNodes = useMemo(() => coreNodes.filter((node) => ['macro', 'fruitGeneral', 'query'].includes(node.role)), [coreNodes]);
  const sectionNodes = useMemo(() => coreNodes.filter((node) => ['micro', 'query', 'macro'].includes(node.role)), [coreNodes]);
  const routeNodes = useMemo(() => coreNodes.filter((node) => node.role === 'route'), [coreNodes]);
  const attributeNodes = useMemo(() => coreNodes.filter((node) => ['style', 'logic', 'syntax'].includes(node.role)), [coreNodes]);
  const protocolNodes = useMemo(() => coreNodes.filter((node) => ['unifiedDecode', 'route', 'query'].includes(node.role)), [coreNodes]);
  const familyPatchView = useMemo(
    () => buildFamilyPatchViewModel(coreNodes, selected, null),
    [coreNodes, selected]
  );

  const fallbackCenter = useMemo(() => [0, -2.6, z], [z]);
  const familyCenter = useMemo(() => averagePosition(familyNodes, fallbackCenter), [familyNodes, fallbackCenter]);
  const sectionCenter = useMemo(() => averagePosition(focusNodes.length > 0 ? focusNodes : sectionNodes, familyCenter), [focusNodes, sectionNodes, familyCenter]);
  const routeCenter = useMemo(() => averagePosition(routeNodes, shiftPosition(sectionCenter, 0, 0, 1.2)), [routeNodes, sectionCenter]);
  const attributeCenter = useMemo(() => averagePosition(attributeNodes, shiftPosition(sectionCenter, 0, 0.4, 0)), [attributeNodes, sectionCenter]);
  const protocolCenter = useMemo(() => averagePosition(protocolNodes, shiftPosition(routeCenter, 0.8, 0, 0)), [protocolNodes, routeCenter]);
  const selectedCenter = Array.isArray(selected?.position) ? selected.position : sectionCenter;
  const offsetVector = useMemo(() => {
    const raw = [
      selectedCenter[0] - familyCenter[0] + 0.2,
      selectedCenter[1] - familyCenter[1] + 0.1,
      selectedCenter[2] - familyCenter[2],
    ];
    return normalizeVector(raw, 1.45);
  }, [familyCenter, selectedCenter]);
  const offsetTarget = useMemo(
    () => shiftPosition(sectionCenter, offsetVector[0], offsetVector[1], offsetVector[2]),
    [offsetVector, sectionCenter]
  );
  const readoutPort = useMemo(() => shiftPosition(protocolCenter, 5.4, 1.2, 0), [protocolCenter]);
  const bridgePorts = useMemo(
    () => [
      shiftPosition(protocolCenter, 5.2, 2.2, -1.2),
      shiftPosition(protocolCenter, 5.5, 0, 0),
      shiftPosition(protocolCenter, 5.2, -2.2, 1.2),
    ],
    [protocolCenter]
  );
  const stagePath = useMemo(
    () => [
      shiftPosition(familyCenter, -1.5, -0.9, -2.8),
      shiftPosition(sectionCenter, -0.4, 0.15, -0.9),
      shiftPosition(routeCenter, 0.4, -0.1, 1.1),
      shiftPosition(protocolCenter, 1.4, 0.8, 2.8),
    ],
    [familyCenter, protocolCenter, routeCenter, sectionCenter]
  );
  const successorPath = useMemo(
    () => [
      shiftPosition(sectionCenter, -1.25, -0.75, -0.8),
      sectionCenter,
      offsetTarget,
      shiftPosition(offsetTarget, 1.35, 0.82, 0.9),
    ],
    [offsetTarget, sectionCenter]
  );

  useFrame((state) => {
    if (!ref.current) {
      return;
    }
    ref.current.rotation.z = state.clock.elapsedTime * 0.12;
    ref.current.rotation.y = state.clock.elapsedTime * 0.16;
  });

  if (!theoryObjectMeta?.id) {
    return null;
  }

  return (
    <group ref={ref}>
      {theoryObjectMeta.id === 'family_patch' && (
        <>
          <Line points={[familyCenter, sectionCenter]} color={accent} transparent opacity={0.62} lineWidth={1.8} />
          <mesh position={familyPatchView.conceptCenter} rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[0.78, 0.04, 12, 40]} />
            <meshStandardMaterial color="#f8b4ff" emissive="#f8b4ff" emissiveIntensity={0.92} transparent opacity={0.34} />
          </mesh>
          <mesh position={familyPatchView.siblingCenter} rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[0.56, 0.03, 12, 36]} />
            <meshStandardMaterial color="#a7f3d0" emissive="#a7f3d0" emissiveIntensity={0.58} transparent opacity={0.22} />
          </mesh>
          <Line points={[familyCenter, familyPatchView.conceptCenter]} color="#f8b4ff" transparent opacity={0.88} lineWidth={2.2} />
          <Line points={[familyCenter, familyPatchView.siblingCenter]} color="#a7f3d0" transparent opacity={0.4} lineWidth={1.5} />
          {familyPatchView.prototypeWitness.slice(0, 4).map((node, idx) => (
            <Line
              key={`family-proto-${node.id}`}
              points={[familyCenter, node.position]}
              color={idx < 2 ? '#dff6ff' : accent}
              transparent
              opacity={0.56}
              lineWidth={idx < 2 ? 1.8 : 1.2}
            />
          ))}
          {familyPatchView.instanceWitness.slice(0, 4).map((node, idx) => (
            <Line
              key={`family-instance-${node.id}`}
              points={[familyPatchView.conceptCenter, node.position]}
              color={idx < 2 ? '#f8b4ff' : '#fda4af'}
              transparent
              opacity={0.62}
              lineWidth={idx < 2 ? 1.8 : 1.2}
            />
          ))}
          <TheoryBeacon position={familyCenter} color={accent} size={0.18} pulse={0.24} speed={1.1} phase={0.2} />
          <TheoryBeacon position={familyPatchView.conceptCenter} color="#f8b4ff" size={0.12} pulse={0.18} speed={1.3} phase={0.6} />
          <TheoryBeacon position={familyPatchView.siblingCenter} color="#a7f3d0" size={0.09} pulse={0.15} speed={1.05} phase={1.0} />
          <TheoryBeacon position={shiftPosition(familyCenter, 1.25, 0.3, 0.2)} color="#dff6ff" size={0.08} phase={0.7} />
          <TheoryBeacon position={shiftPosition(familyCenter, -1.1, -0.4, -0.1)} color="#dff6ff" size={0.08} phase={1.2} />
          <Text position={shiftPosition(familyCenter, 0, 1.05, 0)} color="#dff6ff" fontSize={0.18} anchorX="center" anchorY="middle">
            {'family prototype'}
          </Text>
          <Text position={shiftPosition(familyPatchView.conceptCenter, 0, 0.86, 0)} color="#f8b4ff" fontSize={0.16} anchorX="center" anchorY="middle">
            {'instance offset'}
          </Text>
        </>
      )}
      {theoryObjectMeta.id === 'concept_section' && (
        <>
          <mesh position={sectionCenter} rotation={[0.4, 0.2, 0.1]}>
            <boxGeometry args={[2.7, 0.08, 1.15]} />
            <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={0.78} transparent opacity={0.28} />
          </mesh>
          <mesh position={offsetTarget} rotation={[0.4, -0.16, -0.12]}>
            <boxGeometry args={[1.25, 0.06, 0.76]} />
            <meshStandardMaterial color="#f8b4ff" emissive="#f8b4ff" emissiveIntensity={0.92} transparent opacity={0.42} />
          </mesh>
          <Line points={[sectionCenter, offsetTarget]} color={accent} transparent opacity={0.88} lineWidth={2} />
          <TheoryBeacon position={sectionCenter} color={accent} size={0.14} phase={0.3} />
          <TheoryBeacon position={offsetTarget} color="#f8b4ff" size={0.12} phase={0.9} />
          <TheoryRunner path={[sectionCenter, offsetTarget]} color="#ffffff" size={0.08} speed={0.45} phase={0.18} />
        </>
      )}
      {theoryObjectMeta.id === 'attribute_fiber' && (
        <>
          <Line points={[shiftPosition(attributeCenter, -1.45, -0.72, 0), shiftPosition(attributeCenter, 1.45, 0.72, 0)]} color="#34d399" transparent opacity={0.88} lineWidth={2} />
          <Line points={[shiftPosition(attributeCenter, -1.42, 0.72, 0), shiftPosition(attributeCenter, 1.42, -0.72, 0)]} color="#60a5fa" transparent opacity={0.76} lineWidth={2} />
          <Line points={[shiftPosition(attributeCenter, 0, -1.0, -0.5), shiftPosition(attributeCenter, 0, 1.0, 0.5)]} color={accent} transparent opacity={0.74} lineWidth={2} />
          <TheoryBeacon position={attributeCenter} color={accent} size={0.12} phase={0.2} />
          <TheoryRunner path={[shiftPosition(attributeCenter, -1.45, -0.72, 0), attributeCenter, shiftPosition(attributeCenter, 1.45, 0.72, 0)]} color="#34d399" size={0.07} speed={0.34} phase={0.12} />
          <TheoryRunner path={[shiftPosition(attributeCenter, -1.42, 0.72, 0), attributeCenter, shiftPosition(attributeCenter, 1.42, -0.72, 0)]} color="#60a5fa" size={0.07} speed={0.32} phase={0.48} />
        </>
      )}
      {theoryObjectMeta.id === 'relation_context_fiber' && (
        <>
          <mesh position={routeCenter} rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[1.25, 0.08, 12, 48]} />
            <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={0.88} transparent opacity={0.36} />
          </mesh>
          <Line points={[sectionCenter, routeCenter, protocolCenter]} color={accent} transparent opacity={0.9} lineWidth={2} />
          <TheoryBeacon position={routeCenter} color={accent} size={0.15} phase={0.4} />
          <TheoryRunner path={[sectionCenter, routeCenter, protocolCenter]} color="#dff6ff" size={0.08} speed={0.38} phase={0.16} />
          <TheoryRunner path={[protocolCenter, routeCenter, sectionCenter]} color="#8be9ff" size={0.07} speed={0.24} phase={0.56} />
        </>
      )}
      {theoryObjectMeta.id === 'admissible_update' && (
        <>
          <mesh position={sectionCenter}>
            <sphereGeometry args={[1.2, 22, 22]} />
            <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={0.65} transparent opacity={0.12} wireframe />
          </mesh>
          <mesh position={sectionCenter}>
            <sphereGeometry args={[0.78, 18, 18]} />
            <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={0.42} transparent opacity={0.08} />
          </mesh>
          <Line points={[shiftPosition(sectionCenter, -0.8, 0, 0), sectionCenter, offsetTarget]} color={accent} transparent opacity={0.88} lineWidth={2} />
          <TheoryBeacon position={sectionCenter} color={accent} size={0.12} phase={0.22} />
          <TheoryBeacon position={offsetTarget} color="#ffffff" size={0.08} phase={0.84} />
          <TheoryRunner path={[sectionCenter, offsetTarget]} color={accent} size={0.07} speed={0.42} phase={0.28} />
        </>
      )}
      {theoryObjectMeta.id === 'restricted_readout' && (
        <>
          {focusNodes.slice(0, 6).map((node, idx) => (
            <Line key={`readout-line-${node.id}`} points={[node.position, readoutPort]} color={idx < 2 ? '#ffffff' : accent} transparent opacity={0.72} lineWidth={idx < 2 ? 2.2 : 1.6} />
          ))}
          <mesh position={readoutPort} rotation={[0, 0, -Math.PI / 2]}>
            <coneGeometry args={[0.85, 1.8, 4]} />
            <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={0.92} transparent opacity={0.24} wireframe />
          </mesh>
          <mesh position={shiftPosition(readoutPort, 0.72, 0, 0)}>
            <sphereGeometry args={[0.18, 12, 12]} />
            <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={1.2} />
          </mesh>
          <TheoryRunner path={[sectionCenter, readoutPort]} color="#ffffff" size={0.08} speed={0.54} phase={0.26} />
        </>
      )}
      {theoryObjectMeta.id === 'stage_conditioned_transport' && (
        <>
          {stagePath.map((pos, idx) => (
            <mesh key={`stage-gate-${idx}`} position={pos} rotation={[Math.PI / 2, 0, 0]}>
              <torusGeometry args={[0.55 + idx * 0.08, 0.04, 12, 36]} />
              <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={0.9} transparent opacity={0.34} />
            </mesh>
          ))}
          <Line points={stagePath} color={accent} transparent opacity={0.88} lineWidth={2} />
          <TheoryRunner path={stagePath} color="#dff6ff" size={0.08} speed={0.36} phase={0.12} />
          <TheoryRunner path={stagePath} color={accent} size={0.07} speed={0.26} phase={0.62} />
        </>
      )}
      {theoryObjectMeta.id === 'successor_aligned_transport' && (
        <>
          <Line points={successorPath} color={accent} transparent opacity={0.9} lineWidth={2.2} />
          <mesh position={successorPath[successorPath.length - 1]} rotation={[0, 0, -Math.PI / 3]}>
            <coneGeometry args={[0.18, 0.42, 12]} />
            <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={1.1} />
          </mesh>
          <TheoryRunner path={successorPath} color="#fff7d6" size={0.08} speed={0.48} phase={0.14} />
          <TheoryRunner path={successorPath} color={accent} size={0.07} speed={0.28} phase={0.52} />
        </>
      )}
      {theoryObjectMeta.id === 'protocol_bridge' && (
        <>
          <mesh position={protocolCenter}>
            <cylinderGeometry args={[1.1, 1.1, 0.26, 6]} />
            <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={0.82} transparent opacity={0.26} wireframe />
          </mesh>
          <Line points={[protocolCenter, readoutPort]} color="#fde68a" transparent opacity={0.88} lineWidth={2} />
          {bridgePorts.map((port, idx) => (
            <group key={`bridge-port-${idx}`}>
              <mesh position={port}>
                <boxGeometry args={[0.36, 0.36, 0.36]} />
                <meshStandardMaterial color="#fde68a" emissive="#fde68a" emissiveIntensity={0.92} transparent opacity={0.78} />
              </mesh>
              <Line points={[protocolCenter, port]} color={accent} transparent opacity={0.82} lineWidth={1.8} />
            </group>
          ))}
          <TheoryRunner path={[sectionCenter, protocolCenter, readoutPort]} color="#ffffff" size={0.08} speed={0.42} phase={0.18} />
          <TheoryRunner path={[protocolCenter, bridgePorts[0], bridgePorts[1], bridgePorts[2]]} color={accent} size={0.07} speed={0.26} phase={0.66} />
        </>
      )}
      <Text position={shiftPosition(protocolCenter, 0, -2.1, 0)} color="#dff6ff" fontSize={0.26} anchorX="center" anchorY="middle">
        {label}
      </Text>
    </group>
  );
}

function averageScenePosition(nodes = []) {
  if (!Array.isArray(nodes) || nodes.length === 0) return [0, 0, 0];
  const total = nodes.reduce((acc, node) => {
    const position = Array.isArray(node?.position) ? node.position : [0, 0, 0];
    acc[0] += position[0] || 0;
    acc[1] += position[1] || 0;
    acc[2] += position[2] || 0;
    return acc;
  }, [0, 0, 0]);
  return total.map((value) => value / nodes.length);
}

function LanguageResearchSceneOverlay({ languageFocus = DEFAULT_LANGUAGE_FOCUS, nodes = [], selected = null }) {
  const overlays = Array.isArray(languageFocus?.structureOverlays) ? languageFocus.structureOverlays : [];
  const sceneCenter = useMemo(() => averageScenePosition(nodes), [nodes]);
  const layerMeta = LANGUAGE_RESEARCH_LAYER_META[languageFocus?.researchLayer] || LANGUAGE_RESEARCH_LAYER_META.static_encoding;
  const selectedPosition = Array.isArray(selected?.position) ? selected.position : sceneCenter;
  const roleLabel = languageFocus?.roleGroup || 'object';
  const taskLabel = languageFocus?.taskGroup || 'translation';
  const riskLabel = LANGUAGE_RISK_META[languageFocus?.riskFocus] || '风险焦点未定义';

  return (
    <group>
      <Html position={[-11.8, 10.2, 0]} transform sprite>
        <div style={{
          minWidth: 260,
          padding: '12px 14px',
          borderRadius: 14,
          background: 'rgba(10, 16, 28, 0.88)',
          border: `1px solid ${layerMeta.color}`,
          boxShadow: `0 0 24px ${layerMeta.color}33`,
          color: '#e8f0ff',
          backdropFilter: 'blur(10px)',
        }}>
          <div style={{ fontSize: 13, fontWeight: 800, color: layerMeta.color, marginBottom: 6 }}>
            {layerMeta.label}
          </div>
          <div style={{ fontSize: 11, lineHeight: 1.6, color: '#b4c4dd', marginBottom: 8 }}>
            {`对象组: ${languageFocus?.objectGroup || 'fruit'} | 任务组: ${taskLabel} | 角色组: ${roleLabel}`}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
            {overlays.map((item) => {
              const meta = LANGUAGE_OVERLAY_META[item] || { label: item, color: '#94a3b8' };
              return (
                <span key={item} style={{
                  padding: '3px 8px',
                  borderRadius: 999,
                  fontSize: 10,
                  color: '#f8fbff',
                  background: `${meta.color}22`,
                  border: `1px solid ${meta.color}`,
                }}>
                  {meta.label}
                </span>
              );
            })}
          </div>
          <div style={{ fontSize: 11, color: '#ffd7d7' }}>
            {`风险焦点: ${riskLabel}`}
          </div>
        </div>
      </Html>

      <Text position={[0, 13.8, 0]} color={layerMeta.color} fontSize={0.42} anchorX="center" anchorY="middle">
        {`${layerMeta.label} / ${LANGUAGE_RISK_META[languageFocus?.riskFocus] || '风险焦点'}`}
      </Text>

      {overlays.includes('shared_base') && (
        <group position={sceneCenter}>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[4.8, 0.08, 24, 120]} />
            <meshStandardMaterial color={LANGUAGE_OVERLAY_META.shared_base.color} emissive={LANGUAGE_OVERLAY_META.shared_base.color} emissiveIntensity={1.1} transparent opacity={0.42} />
          </mesh>
        </group>
      )}

      {overlays.includes('local_delta') && selected && (
        <group position={selectedPosition}>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[0.72, 0.05, 18, 60]} />
            <meshStandardMaterial color={LANGUAGE_OVERLAY_META.local_delta.color} emissive={LANGUAGE_OVERLAY_META.local_delta.color} emissiveIntensity={1.4} transparent opacity={0.72} />
          </mesh>
          <mesh position={[0, 0.45, 0]}>
            <sphereGeometry args={[0.12, 20, 20]} />
            <meshStandardMaterial color="#fff0d6" emissive={LANGUAGE_OVERLAY_META.local_delta.color} emissiveIntensity={1.4} />
          </mesh>
        </group>
      )}

      {overlays.includes('path_amplification') && selected && (
        <>
          <Line
            points={[
              selectedPosition,
              [selectedPosition[0] + 2.4, selectedPosition[1] + 1.4, selectedPosition[2] + 1.1],
              [selectedPosition[0] + 4.2, selectedPosition[1] + 2.6, selectedPosition[2] + 1.8],
            ]}
            color={LANGUAGE_OVERLAY_META.path_amplification.color}
            transparent
            opacity={0.82}
            lineWidth={2.4}
          />
          <Text position={[selectedPosition[0] + 4.6, selectedPosition[1] + 2.9, selectedPosition[2] + 2]} color="#d9ffe8" fontSize={0.22}>
            路径放大
          </Text>
        </>
      )}

      {overlays.includes('semantic_roles') && (
        <group position={[sceneCenter[0], sceneCenter[1] + 3.2, sceneCenter[2]]}>
          <Text position={[-2.4, 0.2, 0]} color="#d8c4ff" fontSize={0.2}>对象</Text>
          <Text position={[-0.8, 0.9, 0]} color="#d8c4ff" fontSize={0.2}>属性</Text>
          <Text position={[0.8, 0.9, 0]} color="#d8c4ff" fontSize={0.2}>位置</Text>
          <Text position={[2.4, 0.2, 0]} color="#d8c4ff" fontSize={0.2}>操作</Text>
          <Text position={[-0.8, -0.7, 0]} color="#d8c4ff" fontSize={0.2}>约束</Text>
          <Text position={[0.8, -0.7, 0]} color="#d8c4ff" fontSize={0.2}>结果</Text>
        </group>
      )}

      {overlays.includes('fidelity') && (
        <group position={[sceneCenter[0], sceneCenter[1] + 0.2, sceneCenter[2]]}>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[5.8, 0.07, 20, 100]} />
            <meshStandardMaterial color={LANGUAGE_OVERLAY_META.fidelity.color} emissive={LANGUAGE_OVERLAY_META.fidelity.color} emissiveIntensity={1.1} transparent opacity={0.34} />
          </mesh>
          <Text position={[0, -2.2, 0]} color="#ffd6de" fontSize={0.24} anchorX="center">
            来源保真风险带
          </Text>
        </group>
      )}
    </group>
  );
}

function ConceptAssociationOverlay({ conceptAssociationState = null }) {
  const layers = conceptAssociationState?.layers || [];
  const relations = conceptAssociationState?.relations || [];
  if (!conceptAssociationState || !layers.length) {
    return null;
  }

  const summaryAnchor = averagePosition(
    layers.map((layer) => ({ position: layer.anchorPosition })),
    [0, 0, 0]
  );

  return (
    <group>
      <Text
        position={shiftPosition(summaryAnchor, 0, 2.2, 0)}
        color="#eff6ff"
        fontSize={0.22}
        anchorX="center"
        anchorY="middle"
      >
        {`${conceptAssociationState.conceptLabel} · ${conceptAssociationState.categoryLabel} · 六层关联`}
      </Text>

      {relations.map((relation, index) => (
        <group key={`concept-association-relation-${relation.id}`}>
          <Line
            points={relation.points}
            color={relation.color}
            transparent
            opacity={0.2 + relation.strength * 0.45}
            lineWidth={1.6 + relation.strength * 1.6}
          />
          <TheoryRunner
            path={relation.points}
            color={relation.color}
            size={0.05 + relation.strength * 0.04}
            speed={0.18 + relation.strength * 0.2}
            phase={index * 0.17}
          />
          <Text
            position={blendPosition(relation.points[0], relation.points[1], 0.5)}
            color="#dbeafe"
            fontSize={0.11}
            anchorX="center"
            anchorY="middle"
          >
            {`${relation.label} ${Math.round(relation.strength * 100)}%`}
          </Text>
        </group>
      ))}

      {layers.map((layer, index) => (
        <group key={`concept-association-layer-${layer.id}`} position={layer.anchorPosition}>
          <PulseColumn
            position={[0, 0.34, 0]}
            color={layer.color}
            height={0.55 + layer.avgSignal * 0.7}
            radius={0.05}
            speed={1.0 + index * 0.06}
            phase={index * 0.19}
            opacity={0.38}
          />
          <TheoryBeacon
            position={[0, 0.78, 0]}
            color={layer.color}
            size={0.05 + Math.max(0.06, layer.avgSignal * 0.08)}
            pulse={0.14}
            speed={1.1 + index * 0.05}
            phase={index * 0.21}
            opacity={0.96}
          />
          <Text position={[0, 1.15, 0]} color={layer.color} fontSize={0.14} anchorX="center" anchorY="middle">
            {layer.label}
          </Text>
          <Text position={[0, -0.62, 0]} color="#dbeafe" fontSize={0.1} anchorX="center" anchorY="middle">
            {`${layer.topNodeLabel} · ${layer.nodeCount} 节点`}
          </Text>
        </group>
      ))}

      {layers.flatMap((layer) => layer.nodes.slice(0, 4).map((node, index) => (
        <group key={`concept-association-node-${layer.id}-${node.id}-${index}`} position={node.position}>
          <mesh>
            <sphereGeometry args={[Math.max(0.12, toSafeNumber(node.size, 0.2) * 0.48), 14, 14]} />
            <meshStandardMaterial
              color={layer.color}
              emissive={layer.color}
              emissiveIntensity={0.92}
              transparent
              opacity={0.08 + Math.max(0.08, layer.avgSignal * 0.12)}
              wireframe
            />
          </mesh>
        </group>
      )))}
    </group>
  );
}


/** Layer模型 → 组件模型 连接光束 */
function LayerToComponentBeam({ animProgress = 0 }) {
  const ref = useRef(null);
  const beamRef = useRef(null);

  // 与 LayerExplodedView3D / ComponentDetailPanel3D 同步的动画阶段
  const PHASES = [
    { id: 'input',      color: '#94a3b8', component: 'input' },
    { id: 'ln1',        color: '#818cf8', component: 'ln' },
    { id: 'qkv',        color: '#60a5fa', component: 'attention' },
    { id: 'attn_score', color: '#38bdf8', component: 'attention' },
    { id: 'softmax',    color: '#22d3ee', component: 'attention' },
    { id: 'attn_out',   color: '#2dd4bf', component: 'attention' },
    { id: 'residual1',  color: '#a78bfa', component: 'residual' },
    { id: 'ln2',        color: '#818cf8', component: 'ln' },
    { id: 'ffn_up',     color: '#f59e0b', component: 'ffn' },
    { id: 'ffn_act',    color: '#fb923c', component: 'ffn' },
    { id: 'ffn_down',   color: '#f97316', component: 'ffn' },
    { id: 'residual2',  color: '#a78bfa', component: 'residual' },
  ];
  const DURATIONS = [0.08, 0.07, 0.12, 0.1, 0.08, 0.1, 0.07, 0.07, 0.1, 0.08, 0.08, 0.05];
  const TOTAL = DURATIONS.reduce((s, d) => s + d, 0);

  const currentPhase = useMemo(() => {
    let cum = 0;
    for (let i = 0; i < PHASES.length; i++) {
      const start = cum / TOTAL;
      cum += DURATIONS[i];
      const end = cum / TOTAL;
      if (animProgress >= start && animProgress < end) return PHASES[i];
    }
    return PHASES[PHASES.length - 1];
  }, [animProgress]);

  const beamColor = currentPhase?.color || '#475569';
  // Layer 模型右边缘 → 组件面板左边缘
  const fromX = 23;  // LayerExplodedView3D at [20,0,0], scale 1.5, half width ~3
  const toX = 28;    // ComponentDetailPanel3D at [32,0,0], half width ~4

  useFrame((state) => {
    if (beamRef.current) {
      beamRef.current.material.opacity = 0.3 + 0.2 * Math.sin(state.clock.elapsedTime * 3);
    }
  });

  return (
    <group>
      {/* 连接光束 */}
      <mesh ref={beamRef} position={[(fromX + toX) / 2, 0, 0]} rotation={[0, 0, 0]}>
        <boxGeometry args={[toX - fromX, 0.12, 0.12]} />
        <meshBasicMaterial color={beamColor} transparent opacity={0.35} />
      </mesh>
      {/* 箭头指向组件面板 */}
      <mesh position={[toX - 0.3, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <coneGeometry args={[0.3, 0.8, 6]} />
        <meshStandardMaterial color={beamColor} emissive={beamColor} emissiveIntensity={0.6} transparent opacity={0.7} />
      </mesh>
      {/* 当前组件标签 */}
      <Text
        position={[(fromX + toX) / 2, 1.2, 0]}
        fontSize={0.4}
        color={beamColor}
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.02}
        outlineColor="#0a1022"
      >
        {currentPhase?.component === 'attention' ? 'Attention' :
         currentPhase?.component === 'ffn' ? 'FFN' :
         currentPhase?.component === 'ln' ? 'LayerNorm' :
         currentPhase?.component === 'residual' ? 'Residual ⊕' : ''}
      </Text>
      {/* 流动粒子 */}
      <LayerToComponentParticle color={beamColor} fromX={fromX} toX={toX} />
    </group>
  );
}

function LayerToComponentParticle({ color, fromX, toX }) {
  const ref = useRef(null);
  useFrame((state) => {
    if (!ref.current) return;
    const t = (state.clock.elapsedTime * 0.8) % 1;
    ref.current.position.x = fromX + t * (toX - fromX);
    ref.current.material.opacity = 0.5 + 0.3 * Math.sin(t * Math.PI);
  });
  return (
    <mesh ref={ref} position={[fromX, 0, 0]}>
      <sphereGeometry args={[0.18, 8, 8]} />
      <meshBasicMaterial color={color} transparent opacity={0.7} />
    </mesh>
  );
}


export function AppleNeuronSceneContent({
  nodes,
  links,
  selected,
  onSelect,
  prediction = null,
  mode = 'static',
  theoryObjectMeta = null,
  dimensionLayerProfile = [],
  activeDimension = 'style',
  dimensionCausal = null,
  nodeDisplayEmphasis = {},
  puzzleCompareState = null,
  conceptAssociationState = null,
  animationMode = 'none',
  scanMechanismData = null,
  languageFocus = DEFAULT_LANGUAGE_FOCUS,
  displayLevels = null,
  basicPlayback = false,
  basicPlaybackStep = 1,
  layerSweepStep = 0,
  showAlgorithmConceptCore = false,
  showAlgorithmStaticEncoding = false,
  onBasicStart = () => {},
  onBasicStop = () => {},
  onBasicReplay = () => {},
  showDNNLayers = true,
  visibleComponents = ['attention', 'ffn', 'layer_norm'],
  animProgress = 1,
  activeScenario = null,
  activeSubView = null,
  forwardPassLayer = null,
  forwardPassData = null,
  modelKey = null,
  layerAnimProgress = 0,
  fpSpeed = 800,
  lang = 'en',
}) {
  const layerCount = MODEL_CONFIGS[modelKey]?.layers || LAYER_COUNT;
  const activationMap = prediction?.activationMap || {};
  const focusNodeIds = prediction?.focusNodeIds || [];
  const focusNodeSet = useMemo(() => new Set(focusNodeIds), [focusNodeIds]);
  const modeStyle = MODE_VISUALS[mode] || MODE_VISUALS.static;
  const animationProfile = useMemo(
    () => buildAnimationSceneProfile(nodes, selected, animationMode, scanMechanismData),
    [animationMode, nodes, scanMechanismData, selected]
  );
  const runtimeLayerKey = LAYER_PARAMETER_STATE_ORDER.includes(languageFocus?.researchLayer)
    ? languageFocus.researchLayer
    : 'static_encoding';
  const runtimeProfile = LAYER_PARAMETER_STATE_OVERLAY[runtimeLayerKey] || LAYER_PARAMETER_STATE_OVERLAY.static_encoding;
  const shouldRenderParameterStateOverlay = Boolean(
    displayLevels?.parameter_state !== false
    && (
      runtimeLayerKey !== 'static_encoding'
      || showAlgorithmStaticEncoding
      || languageFocus?.selectedRepairReplaySlotId
    )
  );
  const predictionActiveLayer = Number.isFinite(prediction?.layerProgress)
    ? prediction.layerProgress * (layerCount - 1)
    : null;
  const combinedNodeEmphasis = useMemo(
    () => Object.fromEntries(
      nodes.map((node) => {
        const baseEmphasis = toSafeNumber(nodeDisplayEmphasis?.[node.id], 1);
        const animationEmphasis = toSafeNumber(animationProfile.emphasisMap?.[node.id], 1);
        return [node.id, baseEmphasis * animationEmphasis];
      })
    ),
    [animationProfile.emphasisMap, nodeDisplayEmphasis, nodes]
  );

  const visibleNodes = useMemo(
    () => nodes.filter((node) => (
      toSafeNumber(combinedNodeEmphasis?.[node.id], 1) > 0.025
      && (showAlgorithmConceptCore || !(node.nodeGroup === 'concept_core' || String(node.id || '').startsWith('apple-core-')))
      && (showAlgorithmStaticEncoding || !['style', 'logic', 'syntax'].includes(node.role))
      && isNodeVisibleByDisplayLevels(node, displayLevels)
      && (showDNNLayers || !['query', 'route', 'unifiedDecode', 'style', 'logic', 'syntax', 'background'].includes(node.role))
      && (visibleComponents.includes(node.role) || !['attention', 'ffn', 'layer_norm', 'residual'].includes(node.role))
    )),
    [combinedNodeEmphasis, displayLevels, nodes, showAlgorithmConceptCore, showAlgorithmStaticEncoding, showDNNLayers, visibleComponents]
  );
  const visibleNodeIdSet = useMemo(() => new Set(visibleNodes.map((n) => n.id)), [visibleNodes]);
  const visibleLinks = useMemo(
    () => links
      .filter((link) => visibleNodeIdSet.has(link?.from) && visibleNodeIdSet.has(link?.to))
      .map((link) => ({
        ...link,
        emphasis: (
          toSafeNumber(combinedNodeEmphasis?.[link?.from], 1)
          + toSafeNumber(combinedNodeEmphasis?.[link?.to], 1)
        ) / 2,
      })),
    [combinedNodeEmphasis, links, visibleNodeIdSet]
  );
  const puzzleCompareVisibleLinks = useMemo(
    () => visibleLinks
      .filter((link) => (puzzleCompareState?.sceneLinkHighlightMap || puzzleCompareState?.linkHighlightMap)?.[link.id])
      .map((link) => ({
        ...link,
        compareMeta: (puzzleCompareState?.sceneLinkHighlightMap || puzzleCompareState?.linkHighlightMap)[link.id],
      })),
    [puzzleCompareState, visibleLinks]
  );
  const puzzleCompareVisibleNodes = useMemo(
    () => visibleNodes
      .filter((node) => (puzzleCompareState?.sceneNodeCategoryMap || puzzleCompareState?.nodeCategoryMap)?.[node.id])
      .map((node) => ({
        ...node,
        compareMeta: (puzzleCompareState?.sceneNodeHighlightMap || puzzleCompareState?.nodeHighlightMap)[node.id],
      })),
    [puzzleCompareState, visibleNodes]
  );


  const activeParameterIndex = basicPlayback
    ? Math.max(0, Math.min(runtimeProfile.nodes.length - 1, basicPlaybackStep - 1))
    : -1;
  const runtimeActiveLayer = activeParameterIndex >= 0
    ? runtimeProfile.nodes[activeParameterIndex]?.layer ?? null
    : null;
  const activeLayer = Number.isFinite(predictionActiveLayer)
    ? predictionActiveLayer
    : Number.isFinite(runtimeActiveLayer)
      ? runtimeActiveLayer
      : layerSweepStep;
  const showAdvancedOverlays = Boolean(displayLevels?.advanced_analysis);
  const motionEnabled = Boolean(
    basicPlayback
    || prediction?.isRunning
    || (showAdvancedOverlays && animationMode !== 'none')
  );

  return (
    <>
      <LayerGuides activeLayer={activeLayer} layerCount={layerCount} />

      {/* DNN 模型名称标签 */}
      {(() => {
        const mc = MODEL_CONFIGS[modelKey];
        const modelName = mc?.name || modelKey || '';
        const modelColor = mc?.color || '#4facfe';
        if (!modelName) return null;
        return (
          <group position={[0, 9.0, 0]}>
            <Text
              position={[0, 0, 0]}
              fontSize={1.0}
              color={modelColor}
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.04}
              outlineColor="#0a1022"
            >
              {modelName}
            </Text>
            <Text
              position={[0, -0.9, 0]}
              fontSize={0.42}
              color="#7f95bb"
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.018}
              outlineColor="#0a1022"
            >
              {mc ? `${mc.layers} Layers · d=${mc.dModel} · ${mc.nHeads} Heads` : ''}
            </Text>
          </group>
        );
      })()}

      {/* Forward Pass 逐层动画叠加 */}
      <ForwardPassLayerHighlight
        forwardPassLayer={forwardPassLayer}
        forwardPassData={forwardPassData}
        nLayers={layerCount}
      />

      {/* Layer 内部结构 3D 展开模型 - DNN 模型旁边，Z轴方向，默认显示 */}
      <LayerExplodedView3D
        layerIdx={forwardPassLayer ?? 0}
        modelKey={modelKey}
        layerData={forwardPassData?.[forwardPassLayer ?? 0] || null}
        isActive={forwardPassLayer != null}
        fpSpeed={fpSpeed}
        animProgress={layerAnimProgress}
        position={[20, 0, 0]}
      />

      {/* 组件详情3D模型 - Layer旁边, 显示组件参数细节 */}
      <ComponentDetailPanel3D
        layerIdx={forwardPassLayer ?? 0}
        modelKey={modelKey}
        layerData={forwardPassData?.[forwardPassLayer ?? 0] || null}
        isActive={forwardPassLayer != null}
        animProgress={layerAnimProgress}
        position={[32, 0, 0]}
        lang={lang}
      />

      {/* Layer模型 → 组件模型 连接光束 (动画运行时显示) */}
      {forwardPassLayer != null && <LayerToComponentBeam animProgress={layerAnimProgress} />}

      {displayLevels?.mechanism_chain !== false && visibleLinks.map((link) => (
        <Line
          key={link.id}
          points={link.points}
          color={mode === 'dynamic_prediction' || mode === 'static' ? link.color : modeStyle.accent}
          transparent
          opacity={(0.24 + (prediction?.isRunning ? 0.18 : 0) + modeStyle.linkOpacityBoost) * toSafeNumber(link.emphasis, 1)}
          lineWidth={(1.1 + modeStyle.linkWidthBoost) * (0.8 + toSafeNumber(link.emphasis, 1) * 0.55)}
        />
      ))}

      {puzzleCompareVisibleLinks.map((link) => (
        <Line
          key={`puzzle-compare-${link.id}`}
          points={link.points}
          color={link.compareMeta.color}
          transparent
          opacity={link.compareMeta.opacity}
          lineWidth={link.compareMeta.lineWidth}
        />
      ))}

      {visibleNodes.map((node) => {
        // Forward pass: 已到达层节点正常显示, 未到达层节点变暗
        const fpReached = forwardPassLayer == null || (node.layer != null && node.layer <= forwardPassLayer);
        const fpActive = forwardPassLayer != null && node.layer === forwardPassLayer;
        return (
          <PulsingNeuron
            key={node.id}
            node={node}
            selected={selected?.id === node.id}
            onSelect={onSelect}
            predictionStrength={activationMap[node.id] || 0}
            mode={mode}
            isEffectiveNode={focusNodeSet.has(node.id)}
            visibilityEmphasis={toSafeNumber(combinedNodeEmphasis?.[node.id], 1)}
            motionEnabled={motionEnabled}
            forwardPassReached={fpReached}
            forwardPassActive={fpActive}
          />
        );
      })}

      {puzzleCompareVisibleNodes.map((node) => (
        <group key={`puzzle-compare-node-${node.id}`} position={node.position}>
          <mesh>
            <sphereGeometry args={[Math.max(0.18, toSafeNumber(node.size, 0.55) * 0.34), 18, 18]} />
            <meshStandardMaterial
              color={node.compareMeta.color}
              emissive={node.compareMeta.color}
              emissiveIntensity={1.15}
              transparent
              opacity={node.compareMeta.opacity * 0.28}
              wireframe
            />
          </mesh>
        </group>
      ))}

      {showAdvancedOverlays ? <ModeVisualOverlay mode={mode} prediction={prediction} /> : null}
      <ConceptAssociationOverlay conceptAssociationState={conceptAssociationState} />
      {showAdvancedOverlays ? <LanguageResearchSceneOverlay languageFocus={languageFocus} nodes={visibleNodes} selected={selected} /> : null}
      {showAdvancedOverlays ? <TheoryObjectOverlay theoryObjectMeta={theoryObjectMeta} prediction={prediction} nodes={visibleNodes} selected={selected} /> : null}
      {showAdvancedOverlays ? (
        <AppleNeuronAnimationOverlay
          animationMode={animationMode}
          nodes={visibleNodes}
          selected={selected}
          prediction={prediction}
          scanMechanismData={scanMechanismData}
        />
      ) : null}

      {showAdvancedOverlays ? <TokenPredictionCarrier prediction={prediction} mode={mode} /> : null}
      {showAdvancedOverlays ? <LayerEffectiveNeuronOverlay prediction={prediction} mode={mode} /> : null}

      {/* 动画场景进度叠加 (始终显示，不依赖 showAdvancedOverlays) */}
      {activeScenario && ANIMATION_SCENARIOS[activeScenario] && (() => {
        const scenario = ANIMATION_SCENARIOS[activeScenario];
        const phase = scenario.phases.find(p => animProgress >= p.start && animProgress < p.end) || scenario.phases[scenario.phases.length - 1];
        if (!phase) return null;
        const layerMin = phase.layerRange[0];
        const layerMax = phase.layerRange[1];
        // 过滤在该层范围内的节点
        const highlightNodes = visibleNodes.filter(n => {
          const layer = n.layer ?? n.position?.[1];
          return layer !== undefined && layer >= layerMin && layer <= layerMax;
        });
        return (
          <group>
            <Text
              position={[0, 12, 0]}
              fontSize={0.8}
              color={modeStyle.accent}
              anchorX="center"
              anchorY="middle"
            >
              {scenario.icon} {phase.label} (L{layerMin}-L{layerMax})
            </Text>
            {/* 动画节点环形高亮已移除 */}
            {/* 进度条可视化 */}
            <group position={[-8, 11, 0]}>
              <mesh>
                <boxGeometry args={[16, 0.15, 0.05]} />
                <meshBasicMaterial color="#333" transparent opacity={0.5} />
              </mesh>
              <mesh position={[animProgress * 8 - 8, 0, 0.01]}>
                <boxGeometry args={[16 * animProgress, 0.15, 0.05]} />
                <meshBasicMaterial color={modeStyle.accent} transparent opacity={0.8} />
              </mesh>
              {/* 阶段分段线 */}
              {scenario.phases.map((p, i) => (
                <mesh key={`phase-line-${i}`} position={[p.end * 16 - 8, 0, 0.02]}>
                  <boxGeometry args={[0.03, 0.25, 0.05]} />
                  <meshBasicMaterial color="#888" transparent opacity={0.6} />
                </mesh>
              ))}
            </group>
          </group>
        );
      })()}



      {shouldRenderParameterStateOverlay ? (
        <LayerParameterStateOverlay
          languageFocus={languageFocus}
          selected={selected}
          onSelect={onSelect}
          activeIndex={activeParameterIndex}
          isPlaying={basicPlayback}
        />
      ) : null}
      {showAdvancedOverlays ? <DimensionLayerImpactGraph profile={dimensionLayerProfile} dimension={activeDimension} suppression={dimensionCausal} /> : null}

      {puzzleCompareState?.summary ? (
        <Html position={[0, 7.9, 0]} center>
          <div
            style={{
              padding: '8px 10px',
              borderRadius: 10,
              background: 'rgba(10, 14, 26, 0.82)',
              border: '1px solid rgba(148, 163, 184, 0.26)',
              color: '#e2e8f0',
              fontSize: 10,
              lineHeight: 1.55,
              whiteSpace: 'nowrap',
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 2 }}>双拼图差异高亮</div>
            <div>{`共享节点 ${puzzleCompareState.summary.sharedNodes} | 主独有 ${puzzleCompareState.summary.primaryOnlyNodes} | 对比独有 ${puzzleCompareState.summary.compareOnlyNodes}`}</div>
            <div>{`局部链路回放 ${puzzleCompareState.summary.localReplayLinks} 条`}</div>
            {puzzleCompareState.replaySlotFocus ? (
              <div>{`回放槽位 ${puzzleCompareState.replaySlotFocus.label} | 阶段 ${puzzleCompareState.replaySlotFocus.activePhaseLabel} | 聚焦链路 ${puzzleCompareState.summary.slotFocusedLinks} 条`}</div>
            ) : null}
            {puzzleCompareState.validation ? (
              <div>{`裁剪验证 ${puzzleCompareState.validation.label} | 最小性 ${Math.round(puzzleCompareState.validation.minimalityScore * 100)}%`}</div>
            ) : null}
          </div>
        </Html>
      ) : null}

      {selected && selected.role !== 'background' && (
        <Html position={[selected.position[0], selected.position[1] + 1.25, selected.position[2]]} center>
          <div
            style={{
              padding: '8px 10px',
              borderRadius: 8,
              background: 'rgba(255,255,255,0.95)',
              border: '1px solid rgba(180, 198, 228, 0.85)',
              color: '#1f2937',
              fontSize: 11,
              whiteSpace: 'nowrap',
              pointerEvents: 'none',
            }}
          >
            <div>
              {selected.detailType === 'apple_switch_unit'
                ? `${selected.label} | ${selected.roleLabel || '-'} | L${selected.actualLayer}`
                : `${selected.label} | L${selected.layer}N${selected.neuron}`}
            </div>
            {selected.detailType === 'parameter_state' ? (
              <div style={{ marginTop: 4, fontSize: 10, color: '#334155', lineHeight: 1.45 }}>
                <div>{`参数维度: d${selected.dimIndex}`}</div>
                <div>{`来源阶段: ${selected.sourceStage}`}</div>
              </div>
            ) : null}
            {selected.detailType === 'apple_switch_unit' ? (
              <div style={{ marginTop: 4, fontSize: 10, color: '#334155', lineHeight: 1.45 }}>
                <div>{`类型: ${selected.unitTypeLabel || '-'}`}</div>
                <div>{`方向: ${selected.directionLabel || '-'}`}</div>
              </div>
            ) : null}
          </div>
        </Html>
      )}
    </>
  );
}

function AppleNeuronScene({
  nodes,
  links,
  selected,
  onSelect,
  prediction,
  mode = 'static',
  theoryObjectMeta = null,
  dimensionLayerProfile = [],
  activeDimension = 'style',
  dimensionCausal = null,
  nodeDisplayEmphasis = {},
  puzzleCompareState = null,
  conceptAssociationState = null,
  animationMode = 'none',
  scanMechanismData = null,
  languageFocus = DEFAULT_LANGUAGE_FOCUS,
  displayLevels = null,
  basicPlayback = false,
  basicPlaybackStep = 1,
  layerSweepStep = 0,
  showAlgorithmConceptCore = false,
  showAlgorithmStaticEncoding = false,
  onBasicStart = () => {},
  onBasicStop = () => {},
  onBasicReplay = () => {},
}) {
  return (
    <Canvas shadows dpr={[1, 1.5]}>
      <color attach="background" args={['#090b15']} />
      <fog attach="fog" args={['#090b15', 14, 42]} />

      <ambientLight intensity={0.5} />
      <pointLight position={[12, 12, 16]} intensity={70} color="#8fc4ff" />
      <pointLight position={[-14, -8, -15]} intensity={30} color="#ff9e6b" />

      <PerspectiveCamera makeDefault position={[16, 12, 26]} fov={42} />
      <OrbitControls enablePan enableZoom minDistance={10} />

      <AppleNeuronSceneContent
        nodes={nodes}
        links={links}
        selected={selected}
        onSelect={onSelect}
        prediction={prediction}
        mode={mode}
        theoryObjectMeta={theoryObjectMeta}
        dimensionLayerProfile={dimensionLayerProfile}
        activeDimension={activeDimension}
        dimensionCausal={dimensionCausal}
        nodeDisplayEmphasis={nodeDisplayEmphasis}
        puzzleCompareState={puzzleCompareState}
        conceptAssociationState={conceptAssociationState}
        animationMode={animationMode}
        scanMechanismData={scanMechanismData}
        languageFocus={languageFocus}
        displayLevels={displayLevels}
        basicPlayback={basicPlayback}
        basicPlaybackStep={basicPlaybackStep}
        layerSweepStep={layerSweepStep}
        showAlgorithmConceptCore={showAlgorithmConceptCore}
        showAlgorithmStaticEncoding={showAlgorithmStaticEncoding}
        onBasicStart={onBasicStart}
        onBasicStop={onBasicStop}
        onBasicReplay={onBasicReplay}
      />
    </Canvas>
  );
}


// ---- Main scene wrapper ----

export function AppleNeuronMainScene({ workspace, sceneHeight = '74vh' }) {
  return (
    <div
      style={{
        height: sceneHeight,
        borderRadius: 18,
        border: '1px solid rgba(122, 162, 255, 0.28)',
        overflow: 'hidden',
        background: 'radial-gradient(circle at 20% 0%, rgba(43, 84, 165, 0.2), rgba(8, 10, 18, 0.95) 55%)',
        boxShadow: '0 18px 44px rgba(0,0,0,0.45)',
      }}
    >
      <AppleNeuronScene
        nodes={workspace.nodes}
        links={workspace.links}
        selected={workspace.selected}
        onSelect={workspace.setSelected}
        prediction={workspace.prediction}
        mode={workspace.analysisMode}
        theoryObjectMeta={workspace.currentTheoryObject}
        dimensionLayerProfile={workspace.multidimLayerProfile}
        activeDimension={workspace.multidimActiveDimension}
        dimensionCausal={workspace.multidimCausalData}
        nodeDisplayEmphasis={workspace.nodeDisplayEmphasis}
        puzzleCompareState={workspace.puzzleCompareState}
        conceptAssociationState={workspace.conceptAssociationState}
        animationMode={workspace.animationMode}
        scanMechanismData={workspace.scanMechanismData}
        languageFocus={workspace.languageFocus}
        displayLevels={workspace.displayLevels}
        basicPlayback={workspace.basicRuntimePlaying}
        basicPlaybackStep={workspace.basicRuntimeStep}
        layerSweepStep={workspace.layerSweepStep}
        showAlgorithmConceptCore={workspace.showAlgorithmConceptCore}
        showAlgorithmStaticEncoding={workspace.showAlgorithmStaticEncoding}
        onBasicStart={workspace.handleBasicRuntimeStart}
        onBasicStop={workspace.handleBasicRuntimeStop}
        onBasicReplay={workspace.handleBasicRuntimeReplay}
      />
    </div>
  );
}
