/**
 * TrajectoryRenderer — Token逐层传播轨迹
 * 增强: 支持cos_with_wu颜色维度
 */
import React from 'react';
import { Line, Text } from '@react-three/drei';
import { SPHERE_BASE_SIZE, TRAJECTORY_LINE_WIDTH, deltaCosToColor, cosWuToColor } from '../utils/constants';

export default function TrajectoryRenderer({ trajectory, animated, animationProgress, onHoverToken }) {
  const points = trajectory.points || [];
  const visibleCount = animated
    ? Math.max(1, Math.floor(points.length * animationProgress))
    : points.length;
  const visiblePoints = points.slice(0, visibleCount);

  if (visiblePoints.length < 2) return null;

  const linePoints = visiblePoints.map(p => [p.x, p.y, p.z]);
  // 选择颜色模式: 如果有cos_with_wu数据则用W_U对齐颜色
  const useWuColor = visiblePoints.some(p => p.cos_with_wu !== undefined);

  return (
    <group>
      {/* 轨迹线 */}
      <Line
        points={linePoints}
        color={trajectory.color || '#ffffff'}
        lineWidth={TRAJECTORY_LINE_WIDTH}
        transparent
        opacity={0.7}
      />
      {/* 逐层标记球 */}
      {visiblePoints.map((pt, i) => {
        const color = useWuColor
          ? cosWuToColor(pt.cos_with_wu ?? 0.5)
          : deltaCosToColor(pt.delta_cos ?? 0.5);
        const size = SPHERE_BASE_SIZE + (pt.norm || 10) * 0.0008;
        const isCorrection = trajectory.correction_layers?.includes(pt.layer);
        return (
          <group key={i} position={[pt.x, pt.y, pt.z]}>
            <mesh
              onPointerOver={(e) => {
                e.stopPropagation();
                onHoverToken?.({
                  token: trajectory.token,
                  source: trajectory.source_token,
                  layer: pt.layer,
                  delta_cos: pt.delta_cos,
                  cos_with_target: pt.cos_with_target,
                  cos_with_wu: pt.cos_with_wu,
                  norm: pt.norm,
                  isCorrection,
                });
              }}
            >
              <sphereGeometry args={[size, 16, 16]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={0.5}
                roughness={0.3}
                metalness={0.6}
              />
            </mesh>
            {/* 纠正层环标记 */}
            {isCorrection && (
              <mesh rotation={[Math.PI / 2, 0, 0]}>
                <torusGeometry args={[size + 0.15, 0.04, 8, 32]} />
                <meshStandardMaterial
                  color="#fbbf24"
                  emissive="#fbbf24"
                  emissiveIntensity={2}
                />
              </mesh>
            )}
            {/* 层标签 (仅关键层显示) */}
            {(i === 0 || i === visiblePoints.length - 1 || isCorrection) && (
              <Text
                position={[0, size + 0.4, 0]}
                fontSize={0.35}
                color="#e2e8f0"
                anchorX="center"
                anchorY="bottom"
                outlineWidth={0.05}
                outlineColor="#000000"
              >
                L{pt.layer}
              </Text>
            )}
          </group>
        );
      })}
    </group>
  );
}
