/**
 * DarkMatterFlowRenderer — 暗物质非线性转导可视化
 * 
 * 数据格式 (Schema v2.0):
 * {
 *   type: "dark_matter_flow",
 *   signal_path: [{
 *     layer: int,
 *     w_u_signal: float,         // W_U可见信号占比
 *     w_u_perp_signal: float,    // W_U⊥信号占比
 *     total_norm: float          // 总范数
 *   }],
 *   cascade_transfer: [{         // 可选: 级联转导细节
 *     from_layer: int,
 *     to_layer: int,
 *     transfer_ratio: float,     // 转导比例
 *     nonlinear_component: float // 非线性分量
 *   }]
 * }
 */
import React from 'react';
import { Line, Text } from '@react-three/drei';
import { LAYER_GAP, SUBSPACE_COLORS, CATEGORY_COLORS } from '../utils/constants';

export default function DarkMatterFlowRenderer({ darkMatterFlow, animated, animationProgress, onHoverToken }) {
  const signalPath = darkMatterFlow?.signal_path || [];
  const cascadeTransfer = darkMatterFlow?.cascade_transfer || [];
  const visibleCount = animated
    ? Math.max(1, Math.floor(signalPath.length * animationProgress))
    : signalPath.length;
  const visiblePath = signalPath.slice(0, visibleCount);

  if (visiblePath.length === 0) return null;

  // W_U信号线
  const wuLinePoints = visiblePath.map((p, i) => [
    p.w_u_signal * 6 - 1,
    i * LAYER_GAP,
    2
  ]);

  // W_U⊥信号线 (暗物质)
  const wuPerpLinePoints = visiblePath.map((p, i) => [
    -(p.w_u_perp_signal * 6 - 1),
    i * LAYER_GAP,
    -2
  ]);

  // 总范数线 (中间)
  const normLinePoints = visiblePath.map((p, i) => [
    0,
    i * LAYER_GAP,
    0
  ]);

  return (
    <group>
      {/* W_U信号流 (青绿) */}
      {wuLinePoints.length > 1 && (
        <Line
          points={wuLinePoints}
          color={SUBSPACE_COLORS.w_u}
          lineWidth={3}
          transparent
          opacity={0.7}
        />
      )}

      {/* W_U⊥暗物质流 (红色) */}
      {wuPerpLinePoints.length > 1 && (
        <Line
          points={wuPerpLinePoints}
          color={SUBSPACE_COLORS.w_u_perp}
          lineWidth={3}
          transparent
          opacity={0.7}
        />
      )}

      {/* 逐层节点 */}
      {visiblePath.map((p, i) => {
        const yPos = i * LAYER_GAP;

        return (
          <group key={i} position={[0, yPos, 0]}>
            {/* W_U信号节点 */}
            <mesh position={[p.w_u_signal * 6 - 1, 0, 2]}
              onPointerOver={(e) => {
                e.stopPropagation();
                onHoverToken?.({
                  layer: p.layer,
                  w_u_signal: p.w_u_signal,
                  w_u_perp_signal: p.w_u_perp_signal,
                  total_norm: p.total_norm,
                  type: 'dark_matter_flow',
                });
              }}
            >
              <sphereGeometry args={[0.2 + p.w_u_signal * 0.3, 12, 12]} />
              <meshStandardMaterial
                color={SUBSPACE_COLORS.w_u}
                emissive={SUBSPACE_COLORS.w_u}
                emissiveIntensity={0.5}
              />
            </mesh>

            {/* W_U⊥暗物质节点 */}
            <mesh position={[-(p.w_u_perp_signal * 6 - 1), 0, -2]}>
              <sphereGeometry args={[0.2 + p.w_u_perp_signal * 0.3, 12, 12]} />
              <meshStandardMaterial
                color={SUBSPACE_COLORS.w_u_perp}
                emissive={SUBSPACE_COLORS.w_u_perp}
                emissiveIntensity={0.5}
              />
            </mesh>

            {/* 中心总范数指示 */}
            <mesh position={[0, 0, 0]}>
              <sphereGeometry args={[0.1, 8, 8]} />
              <meshStandardMaterial
                color="#94a3b8"
                emissive="#94a3b8"
                emissiveIntensity={0.3}
              />
            </mesh>

            {/* 连接线 (W_U ↔ 中心 ↔ W_U⊥) */}
            <Line
              points={[
                [p.w_u_signal * 6 - 1, 0, 2],
                [0, 0, 0],
                [-(p.w_u_perp_signal * 6 - 1), 0, -2],
              ]}
              color="#334155"
              lineWidth={1}
              transparent
              opacity={0.3}
            />

            {/* 层标签 + 比例标注 */}
            <Text position={[-7, 0, 0]} fontSize={0.3} color="#94a3b8" anchorX="right">
              {`L${p.layer}`}
            </Text>
            <Text position={[p.w_u_signal * 6 - 1, 0.5, 2]} fontSize={0.18} color={SUBSPACE_COLORS.w_u} anchorX="center">
              {`${(p.w_u_signal * 100).toFixed(0)}%`}
            </Text>
            <Text position={[-(p.w_u_perp_signal * 6 - 1), 0.5, -2]} fontSize={0.18} color={SUBSPACE_COLORS.w_u_perp} anchorX="center">
              {`${(p.w_u_perp_signal * 100).toFixed(0)}%`}
            </Text>
          </group>
        );
      })}

      {/* 级联转导弧线 (如果有) */}
      {cascadeTransfer.map((ct, ci) => {
        const fromY = ct.from_layer * LAYER_GAP;
        const toY = ct.to_layer * LAYER_GAP;
        const midY = (fromY + toY) / 2 + 2;
        const curvePoints = [];
        for (let s = 0; s <= 10; s++) {
          const t = s / 10;
          const y = fromY * (1 - t) + toY * t + Math.sin(t * Math.PI) * 2;
          const z = -4 * Math.sin(t * Math.PI);
          curvePoints.push([0, y, z]);
        }
        return (
          <Line
            key={`cascade${ci}`}
            points={curvePoints}
            color={SUBSPACE_COLORS.dark_matter}
            lineWidth={1 + ct.transfer_ratio * 3}
            transparent
            opacity={0.4 + ct.transfer_ratio * 0.4}
          />
        );
      })}

      {/* 图例 */}
      <group position={[-8, visiblePath.length * LAYER_GAP + 2, 0]}>
        <Text position={[0, 0, 0]} fontSize={0.25} color={SUBSPACE_COLORS.w_u} anchorX="left">
          ● W_U Signal
        </Text>
        <Text position={[0, -0.5, 0]} fontSize={0.25} color={SUBSPACE_COLORS.w_u_perp} anchorX="left">
          ● W_U⊥ Dark Matter
        </Text>
        {cascadeTransfer.length > 0 && (
          <Text position={[0, -1.0, 0]} fontSize={0.25} color={SUBSPACE_COLORS.dark_matter} anchorX="left">
            ─ Cascade Transfer
          </Text>
        )}
      </group>
    </group>
  );
}
