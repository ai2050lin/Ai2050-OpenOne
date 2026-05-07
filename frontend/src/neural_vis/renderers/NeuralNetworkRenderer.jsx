/**
 * NeuralNetworkRenderer — DNN层结构3D可视化渲染器 (v6.0 Clean)
 * 
 * 清理版: 去除冗余, 保留核心框架 + 逐层前向传播动画
 * - 每层半透明圆盘 + 层号标签
 * - Forward Pass: 逐层高亮动画 (useFrame驱动脉冲光环)
 * - 当前层神经元: 激活值着色 (>0.8红, >0.5黄, >0.3绿, <0.3蓝)
 * - 层间连接线: 已通过层亮蓝色
 */
import React, { useRef, useMemo } from 'react';
import * as THREE from 'three';
import { Text, Line } from '@react-three/drei';
import { useFrame, useThree } from '@react-three/fiber';
import {
  LAYER_GAP as GLOBAL_LAYER_GAP, PLANE_SIZE, SUBSPACE_COLORS,
  layerToFuncColor, layerToFuncLabel,
} from '../utils/constants';

// 本渲染器使用更紧凑的层间距, 36层总高约72
const LAYER_GAP = 2.0;

// ==================== 摄像机跟随前向传播 ====================
function CameraFollow({ targetY, enabled }) {
  const { camera } = useThree();
  const targetRef = useRef(new THREE.Vector3(16, 12, 26));

  useFrame(() => {
    if (!enabled || targetY == null) return;
    // 目标: 看向当前层, 摄像机在侧上方
    const targetPos = new THREE.Vector3(16, targetY + 8, 26);
    camera.position.lerp(targetPos, 0.03);
    const lookAt = new THREE.Vector3(0, targetY, 0);
    const currentTarget = new THREE.Vector3();
    camera.getWorldDirection(currentTarget);
    camera.lookAt(lookAt);
  });

  return null;
}

// ==================== 激活值→颜色 ====================
export function activationToColor(value) {
  if (value > 0.8) return '#ff4444';
  if (value > 0.5) return '#ffcc00';
  if (value > 0.3) return '#22c55e';
  return '#3b82f6';
}

function activationToSize(value) {
  if (value > 0.8) return 0.5;
  if (value > 0.5) return 0.38;
  if (value > 0.3) return 0.28;
  return 0.18;
}

// ==================== 层圆盘 + 脉冲光环 (useFrame动画) ====================
function LayerDisk({ layer, nLayers, y, forwardPassActive, forwardPassReached }) {
  const ringRef = useRef();
  const color = layerToFuncColor(layer, nLayers);
  const label = layerToFuncLabel(layer, nLayers);

  // 脉冲光环动画
  useFrame((state) => {
    if (ringRef.current && forwardPassActive) {
      const t = state.clock.elapsedTime;
      ringRef.current.material.opacity = 0.4 + 0.3 * Math.sin(t * 4);
      ringRef.current.scale.set(1 + 0.02 * Math.sin(t * 3), 1, 1 + 0.02 * Math.sin(t * 3));
    }
  });

  const dimFactor = forwardPassReached === false ? 0.15 : 1.0;

  return (
    <group position={[0, y, 0]}>
      {/* 圆盘主体 */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[PLANE_SIZE / 2, 64]} />
        <meshStandardMaterial
          color={forwardPassActive ? '#ffffff' : color}
          transparent
          opacity={(forwardPassActive ? 0.45 : forwardPassReached === false ? 0.03 : 0.12) * dimFactor}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
      {/* 边缘环 */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[PLANE_SIZE / 2 - 0.15, PLANE_SIZE / 2, 64]} />
        <meshBasicMaterial
          color={forwardPassActive ? '#ffffff' : color}
          transparent
          opacity={(forwardPassActive ? 0.9 : forwardPassReached === false ? 0.06 : 0.3)}
          side={THREE.DoubleSide}
        />
      </mesh>
      {/* Forward Pass 当前层: 脉冲光环 */}
      {forwardPassActive && (
        <mesh ref={ringRef} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[PLANE_SIZE / 2 + 0.3, PLANE_SIZE / 2 + 0.9, 64]} />
          <meshBasicMaterial color="#4facfe" transparent opacity={0.5} side={THREE.DoubleSide} />
        </mesh>
      )}
      {/* 层号 */}
      <Text
        position={[-PLANE_SIZE / 2 - 1.2, 0, 0]}
        fontSize={0.6}
        color={forwardPassActive ? '#fff' : '#64748b'}
        anchorX="right"
        anchorY="middle"
      >
        L{layer}
      </Text>
      {/* 当前层功能标签 */}
      {forwardPassActive && (
        <Text
          position={[PLANE_SIZE / 2 + 1.5, 0, 0]}
          fontSize={0.5}
          color="#4facfe"
          anchorX="left"
          anchorY="middle"
        >
          {label}
        </Text>
      )}
    </group>
  );
}

// ==================== 神经元球体 (当前层+已通过层可见) ====================
function NeuronSpheres({ neurons, y, isCurrentLayer, layerReached }) {
  if (!neurons || neurons.length === 0 || !layerReached) return null;

  return (
    <group position={[0, y, 0]}>
      {neurons.map((n, i) => {
        const color = activationToColor(n.activation || 0);
        const size = activationToSize(n.activation || 0.3) * (isCurrentLayer ? 1.5 : 1.0);
        const emissive = n.activation > 0.8 ? 2.0 : n.activation > 0.5 ? 1.2 : n.activation > 0.3 ? 0.6 : 0.2;
        return (
          <mesh key={i} position={[n.x || 0, 0.3, n.z || 0]}>
            <sphereGeometry args={[size, 16, 16]} />
            <meshStandardMaterial
              color={color}
              emissive={color}
              emissiveIntensity={emissive * (isCurrentLayer ? 1.8 : 0.8)}
              transparent
              opacity={isCurrentLayer ? 0.95 : 0.6}
              toneMapped={false}
            />
          </mesh>
        );
      })}
    </group>
  );
}

// ==================== 层间连接线 ====================
function InterLayerConnections({ nLayers, forwardPassCurrentLayer }) {
  const lines = useMemo(() => {
    const result = [];
    const step = Math.max(1, Math.floor(nLayers / 8));
    for (let l = 0; l < nLayers - step; l += step) {
      result.push({ y1: l * LAYER_GAP, y2: (l + step) * LAYER_GAP, layer: l });
    }
    return result;
  }, [nLayers]);

  return (
    <>
      {lines.map((line, i) => {
        const activated = forwardPassCurrentLayer != null && line.layer < forwardPassCurrentLayer;
        return (
          <Line
            key={i}
            points={[[0, line.y1, 0], [0, line.y2, 0]]}
            color={activated ? '#4facfe' : '#334155'}
            lineWidth={activated ? 2 : 1}
            transparent
            opacity={activated ? 0.5 : 0.1}
          />
        );
      })}
    </>
  );
}

// ==================== 前向传播信号球 (从当前层向上飘) ====================
function ForwardPassSignalBall({ currentLayer }) {
  const meshRef = useRef();

  useFrame((state) => {
    if (meshRef.current && currentLayer != null) {
      const t = state.clock.elapsedTime;
      const yBase = currentLayer * LAYER_GAP;
      meshRef.current.position.y = yBase + 0.5 * Math.sin(t * 5);
      meshRef.current.material.opacity = 0.6 + 0.3 * Math.sin(t * 4);
    }
  });

  if (currentLayer == null) return null;
  return (
    <mesh ref={meshRef} position={[0, currentLayer * LAYER_GAP, 0]}>
      <sphereGeometry args={[0.4, 16, 16]} />
      <meshBasicMaterial color="#4facfe" transparent opacity={0.8} />
    </mesh>
  );
}

// ==================== 激活值图例 ====================
function ActivationLegend({ visible }) {
  if (!visible) return null;
  const items = [
    { color: '#ff4444', label: '>0.8 强激活', y: 1.2 },
    { color: '#ffcc00', label: '>0.5 中激活', y: 0.6 },
    { color: '#22c55e', label: '>0.3 弱激活', y: 0 },
    { color: '#3b82f6', label: '<0.3 微弱', y: -0.6 },
  ];
  return (
    <group position={[-PLANE_SIZE / 2 - 4, 0, PLANE_SIZE / 2]}>
      {items.map((item, i) => (
        <group key={i} position={[0, item.y, 0]}>
          <mesh>
            <sphereGeometry args={[0.15, 8, 8]} />
            <meshBasicMaterial color={item.color} />
          </mesh>
          <Text position={[0.4, 0, 0]} fontSize={0.25} color={item.color} anchorX="left" anchorY="middle">
            {item.label}
          </Text>
        </group>
      ))}
    </group>
  );
}

// ==================== 主渲染器 ====================
export default function NeuralNetworkRenderer({
  nLayers = 36,
  activeLayerRange = null,
  highlightedLayer = null,
  forwardPassLayer = null,
  forwardPassData = null,
  useActivationColor = true,
}) {
  const groupRef = useRef();

  const layers = useMemo(() => Array.from({ length: nLayers }, (_, i) => i), [nLayers]);

  const isActive = (layer) => {
    if (!activeLayerRange) return true;
    return layer >= activeLayerRange[0] && layer <= activeLayerRange[1];
  };

  const isFpActive = (layer) => forwardPassLayer != null && layer === forwardPassLayer;
  const isFpReached = (layer) => forwardPassLayer == null || layer <= forwardPassLayer;

  // 提取某层神经元数据
  const getNeurons = (layerIdx) => {
    return forwardPassData?.[layerIdx]?.neuron_activations || null;
  };

  return (
    <group ref={groupRef}>
      {/* 摄像机跟随前向传播当前层 */}
      <CameraFollow targetY={forwardPassLayer != null ? forwardPassLayer * LAYER_GAP : null} enabled={forwardPassLayer != null} />

      {/* 层圆盘 */}
      {layers.filter(l => isActive(l)).map(l => (
        <LayerDisk
          key={l}
          layer={l}
          nLayers={nLayers}
          y={l * LAYER_GAP}
          forwardPassActive={isFpActive(l)}
          forwardPassReached={isFpReached(l)}
        />
      ))}

      {/* 神经元球体: 已到达层显示 */}
      {layers.filter(l => isActive(l) && isFpReached(l)).map(l => (
        <NeuronSpheres
          key={`n-${l}`}
          neurons={getNeurons(l)}
          y={l * LAYER_GAP}
          isCurrentLayer={isFpActive(l)}
          layerReached={isFpReached(l)}
        />
      ))}

      {/* 层间连接 */}
      <InterLayerConnections nLayers={nLayers} forwardPassCurrentLayer={forwardPassLayer} />

      {/* 前向传播信号球 */}
      <ForwardPassSignalBall currentLayer={forwardPassLayer} />

      {/* 激活值图例 */}
      <ActivationLegend visible={forwardPassLayer != null} />
    </group>
  );
}
