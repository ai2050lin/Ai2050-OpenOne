/**
 * CausalChainRenderer — 因果链追踪可视化
 * 
 * 数据格式 (Schema v2.0):
 * {
 *   type: "causal_chain",
 *   intervention: { layer: int, subspace: str, direction: str },
 *   propagation: [{
 *     layer: int,
 *     kl_divergence: float,
 *     classification_flip: float,
 *     top_token: str,
 *     prob_change: float
 *   }]
 * }
 */
import React from 'react';
import { Line, Text } from '@react-three/drei';
import { LAYER_GAP, CAUSAL_COLORS } from '../utils/constants';

export default function CausalChainRenderer({ causalChain, animated, animationProgress, onHoverToken }) {
  const intervention = causalChain?.intervention || {};
  const propagation = causalChain?.propagation || [];
  const visibleCount = animated
    ? Math.max(1, Math.floor(propagation.length * animationProgress))
    : propagation.length;
  const visibleProp = propagation.slice(0, visibleCount);

  if (visibleProp.length === 0) return null;

  // 计算布局: 层从左到右
  const layerSpacing = 3;
  const offsetX = -(visibleProp.length - 1) * layerSpacing / 2;

  // KL散度线
  const klLinePoints = visibleProp.map((p, i) => [
    i * layerSpacing + offsetX,
    (p.kl_divergence || 0) * 0.5,
    -2
  ]);

  // 翻转率线
  const flipLinePoints = visibleProp.map((p, i) => [
    i * layerSpacing + offsetX,
    (p.classification_flip || 0) * 5,
    2
  ]);

  return (
    <group>
      {/* 干预起点标记 */}
      <group position={[0, 8, 0]}>
        <mesh>
          <sphereGeometry args={[0.5, 16, 16]} />
          <meshStandardMaterial
            color={CAUSAL_COLORS.intervention}
            emissive={CAUSAL_COLORS.intervention}
            emissiveIntensity={1}
          />
        </mesh>
        <Text position={[0, 1, 0]} fontSize={0.35} color={CAUSAL_COLORS.intervention} anchorX="center">
          {`干预: L${intervention.layer} ${intervention.subspace || ''} ${intervention.direction || ''}`}
        </Text>
      </group>

      {/* KL散度线 */}
      {klLinePoints.length > 1 && (
        <Line
          points={klLinePoints}
          color={CAUSAL_COLORS.propagation}
          lineWidth={3}
          transparent
          opacity={0.8}
        />
      )}

      {/* 翻转率线 */}
      {flipLinePoints.length > 1 && (
        <Line
          points={flipLinePoints}
          color={CAUSAL_COLORS.flip}
          lineWidth={3}
          transparent
          opacity={0.8}
        />
      )}

      {/* 逐层节点 */}
      {visibleProp.map((p, i) => {
        const x = i * layerSpacing + offsetX;
        const klY = (p.kl_divergence || 0) * 0.5;
        const flipY = (p.classification_flip || 0) * 5;

        return (
          <group key={i} position={[x, 0, 0]}>
            {/* KL节点 */}
            <mesh position={[0, klY, -2]}
              onPointerOver={(e) => {
                e.stopPropagation();
                onHoverToken?.({
                  layer: p.layer,
                  kl_divergence: p.kl_divergence,
                  classification_flip: p.classification_flip,
                  top_token: p.top_token,
                  type: 'causal_chain',
                });
              }}
            >
              <sphereGeometry args={[0.3, 12, 12]} />
              <meshStandardMaterial
                color={CAUSAL_COLORS.propagation}
                emissive={CAUSAL_COLORS.propagation}
                emissiveIntensity={0.5}
              />
            </mesh>

            {/* 翻转率节点 */}
            <mesh position={[0, flipY, 2]}>
              <sphereGeometry args={[0.25, 12, 12]} />
              <meshStandardMaterial
                color={CAUSAL_COLORS.flip}
                emissive={CAUSAL_COLORS.flip}
                emissiveIntensity={0.5}
              />
            </mesh>

            {/* 层标签 */}
            <Text position={[0, -1, 0]} fontSize={0.25} color="#94a3b8" anchorX="center">
              L{p.layer}
            </Text>

            {/* 数值标注 */}
            <Text position={[0.8, klY, -2]} fontSize={0.18} color={CAUSAL_COLORS.propagation} anchorX="left">
              {`KL=${(p.kl_divergence ?? 0).toFixed(1)}`}
            </Text>
            <Text position={[0.8, flipY, 2]} fontSize={0.18} color={CAUSAL_COLORS.flip} anchorX="left">
              {`flip=${(p.classification_flip ?? 0).toFixed(2)}`}
            </Text>

            {/* 连接虚线 */}
            <Line
              points={[[0, klY, -2], [0, flipY, 2]]}
              color="#334155"
              lineWidth={1}
              transparent
              opacity={0.3}
            />
          </group>
        );
      })}

      {/* 图例 */}
      <group position={[offsetX - 2, 6, 0]}>
        <Text position={[0, 0, 0]} fontSize={0.25} color={CAUSAL_COLORS.propagation} anchorX="left">
          — KL Divergence
        </Text>
        <Text position={[0, -0.5, 0]} fontSize={0.25} color={CAUSAL_COLORS.flip} anchorX="left">
          — Classification Flip Rate
        </Text>
      </group>
    </group>
  );
}
