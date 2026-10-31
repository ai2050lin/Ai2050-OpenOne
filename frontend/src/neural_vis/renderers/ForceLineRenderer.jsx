/**
 * ForceLineRenderer — 语义力线指数增长可视化
 * 
 * 数据格式 (Schema v2.0):
 * {
 *   type: "force_line",
 *   concepts: [{
 *     concept: str,
 *     points: [{
 *       layer: int, norm: float, cos_with_wu: float,
 *       x: float, y: float, z: float, exp_fit: float
 *     }],
 *     growth_rate: float  // exp系数
 *   }]
 * }
 */
import React from 'react';
import { Line, Text } from '@react-three/drei';
import { LAYER_GAP, cosWuToColor, TRAJECTORY_LINE_WIDTH, CATEGORY_COLORS } from '../utils/constants';

const FORCE_LINE_COLORS = ['#4ecdc4', '#ff6b6b', '#ffe66d', '#a855f7', '#f97316', '#34d399'];

export default function ForceLineRenderer({ forceLine, animated, animationProgress, onHoverToken }) {
  const concepts = forceLine?.concepts || [];

  return (
    <group>
      {concepts.map((concept, ci) => {
        const points = concept.points || [];
        const visibleCount = animated
          ? Math.max(1, Math.floor(points.length * animationProgress))
          : points.length;
        const visiblePoints = points.slice(0, visibleCount);
        const lineColor = FORCE_LINE_COLORS[ci % FORCE_LINE_COLORS.length];

        if (visiblePoints.length < 2) return null;

        // 实际轨迹线
        const actualLinePoints = visiblePoints.map(p => [p.x, p.y, p.z]);
        // exp拟合线 (虚线效果用更细线条)
        const fitLinePoints = visiblePoints.map(p => [p.x, (p.exp_fit || p.y) * (p.y / (p.norm || 1)) , p.z]);

        return (
          <group key={ci}>
            {/* 实际力线 (粗) */}
            <Line
              points={actualLinePoints}
              color={lineColor}
              lineWidth={TRAJECTORY_LINE_WIDTH}
              transparent
              opacity={0.8}
            />
            {/* exp拟合线 (细, 半透明) */}
            {visiblePoints.length > 2 && (
              <Line
                points={visiblePoints.map(p => {
                  // 用exp_fit值作为Y坐标
                  const basePt = points[0];
                  const normRatio = basePt ? (p.exp_fit || p.norm) / (basePt.norm || 1) : 1;
                  return [p.x, normRatio * 2, p.z];
                })}
                color={lineColor}
                lineWidth={1}
                transparent
                opacity={0.3}
                dashed
                dashSize={0.5}
                gapSize={0.3}
              />
            )}
            {/* 逐层标记球 — 颜色=cos_with_wu */}
            {visiblePoints.map((pt, pi) => {
              const color = cosWuToColor(pt.cos_with_wu ?? 0.5);
              const size = 0.15 + (pt.norm || 1) * 0.003;
              return (
                <group key={pi} position={[pt.x, pt.y, pt.z]}>
                  <mesh
                    onPointerOver={(e) => {
                      e.stopPropagation();
                      onHoverToken?.({
                        token: concept.concept,
                        layer: pt.layer,
                        norm: pt.norm,
                        cos_with_wu: pt.cos_with_wu,
                        growth_rate: concept.growth_rate,
                      });
                    }}
                  >
                    <sphereGeometry args={[size, 12, 12]} />
                    <meshStandardMaterial
                      color={color}
                      emissive={color}
                      emissiveIntensity={0.5}
                      roughness={0.3}
                      metalness={0.6}
                    />
                  </mesh>
                  {/* 首尾和W_U对齐关键层标注 */}
                  {(pi === 0 || pi === visiblePoints.length - 1 || (pt.cos_with_wu ?? 0) > 0.7) && (
                    <Text
                      position={[0, size + 0.3, 0]}
                      fontSize={0.25}
                      color="#e2e8f0"
                      anchorX="center"
                      anchorY="bottom"
                    >
                      {`L${pt.layer} cos=${(pt.cos_with_wu ?? 0).toFixed(2)}`}
                    </Text>
                  )}
                </group>
              );
            })}
            {/* 概念标签 */}
            {visiblePoints.length > 0 && (
              <Text
                position={[visiblePoints[0].x - 1.5, visiblePoints[0].y, visiblePoints[0].z]}
                fontSize={0.3}
                color={lineColor}
                anchorX="right"
              >
                {concept.concept}
              </Text>
            )}
          </group>
        );
      })}
    </group>
  );
}
