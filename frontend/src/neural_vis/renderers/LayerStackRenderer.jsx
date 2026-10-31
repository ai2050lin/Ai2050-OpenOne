/**
 * LayerStackRenderer — 层堆叠模型
 * 增强: 每层添加W_U/W_U⊥占比条
 */
import React from 'react';
import { Text, Line } from '@react-three/drei';
import * as THREE from 'three';
import { LAYER_GAP, PLANE_SIZE, LAYER_FUNC_COLORS, SUBSPACE_COLORS } from '../utils/constants';

export default function LayerStackRenderer({ layerStack, selectedLayers, trajectoryData }) {
  const layers = (layerStack?.layers || []).filter(
    l => !selectedLayers || selectedLayers.includes(l.layer)
  );

  return (
    <group>
      {layers.map((layer, i) => {
        const yPos = i * LAYER_GAP;
        const color = layer.color || LAYER_FUNC_COLORS[layer.function] || '#4ecdc4';
        const hasSubspace = layer.w_u_ratio !== undefined || layer.w_u_perp_ratio !== undefined;
        const wuRatio = layer.w_u_ratio ?? 0;
        const wuPerpRatio = layer.w_u_perp_ratio ?? (1 - wuRatio);

        return (
          <group key={layer.layer} position={[0, yPos, 0]}>
            {/* 透明层板 */}
            <mesh rotation={[-Math.PI / 2, 0, 0]}>
              <planeGeometry args={[PLANE_SIZE, PLANE_SIZE]} />
              <meshStandardMaterial
                color={color}
                transparent
                opacity={0.06}
                side={THREE.DoubleSide}
                depthWrite={false}
              />
            </mesh>
            {/* 层边框 */}
            <Line
              points={[
                [-PLANE_SIZE/2, 0, -PLANE_SIZE/2],
                [PLANE_SIZE/2, 0, -PLANE_SIZE/2],
                [PLANE_SIZE/2, 0, PLANE_SIZE/2],
                [-PLANE_SIZE/2, 0, PLANE_SIZE/2],
                [-PLANE_SIZE/2, 0, -PLANE_SIZE/2],
              ]}
              color={color}
              lineWidth={1}
              transparent
              opacity={0.3}
            />
            {/* W_U/W_U⊥占比条 */}
            {hasSubspace && (
              <group position={[PLANE_SIZE / 2 + 2.5, 0, 0]}>
                {/* W_U部分 (青绿) */}
                <mesh position={[0, 0, -0.5 + wuRatio * 0.5]}>
                  <boxGeometry args={[0.4, 0.15, Math.max(0.05, wuRatio)]} />
                  <meshStandardMaterial
                    color={SUBSPACE_COLORS.w_u}
                    emissive={SUBSPACE_COLORS.w_u}
                    emissiveIntensity={0.4}
                    transparent
                    opacity={0.8}
                  />
                </mesh>
                {/* W_U⊥部分 (红色) */}
                <mesh position={[0, 0, 0.5 - wuPerpRatio * 0.5]}>
                  <boxGeometry args={[0.4, 0.15, Math.max(0.05, wuPerpRatio)]} />
                  <meshStandardMaterial
                    color={SUBSPACE_COLORS.w_u_perp}
                    emissive={SUBSPACE_COLORS.w_u_perp}
                    emissiveIntensity={0.4}
                    transparent
                    opacity={0.8}
                  />
                </mesh>
              </group>
            )}
            {/* 层标签 */}
            <Text
              position={[-PLANE_SIZE / 2 - 1.5, 0, 0]}
              fontSize={0.45}
              color={color}
              anchorX="right"
              anchorY="middle"
            >
              L{layer.layer} {layer.label || ''}
            </Text>
            {/* 指标摘要 */}
            {layer.metrics && (
              <Text
                position={[PLANE_SIZE / 2 + (hasSubspace ? 4.5 : 0.5), 0, 0]}
                fontSize={0.25}
                color="#94a3b8"
                anchorX="left"
                anchorY="middle"
              >
                {`δ=${(layer.metrics.avg_delta_cos ?? 0).toFixed(2)} sw=${(layer.metrics.switch_rate ?? 0).toFixed(2)}`}
              </Text>
            )}
          </group>
        );
      })}
    </group>
  );
}
