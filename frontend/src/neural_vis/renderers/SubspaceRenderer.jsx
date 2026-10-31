/**
 * SubspaceRenderer — W_U/W_U⊥子空间分解可视化
 * 
 * 数据格式 (Schema v2.0):
 * {
 *   type: "subspace_decomposition",
 *   layers: [{
 *     layer: int,
 *     w_u_ratio: float,           // W_U可见部分占比
 *     w_u_perp_ratio: float,      // W_U⊥部分占比
 *     grammar_in_perp: {          // 各语法角色在W_U⊥中的占比
 *       nsubj: float, dobj: float, amod: float, aux: float
 *     },
 *     semantics_in_w_u: float,    // 语义能量在W_U top10奇异模式占比
 *     concept_points: [{          // 可选: 概念点在两个子空间中的分布
 *       token: str, category: str, subspace: str, x: float, y: float, z: float
 *     }]
 *   }]
 * }
 */
import React from 'react';
import { Text, Line } from '@react-three/drei';
import * as THREE from 'three';
import { LAYER_GAP, SPHERE_BASE_SIZE, SUBSPACE_COLORS, GRAMMAR_ROLE_COLORS, LAYER_FUNC_COLORS } from '../utils/constants';

export default function SubspaceRenderer({ subspaceDecomp, animated, animationProgress, onHoverToken }) {
  const layers = subspaceDecomp?.layers || [];
  const visibleCount = animated
    ? Math.max(1, Math.floor(layers.length * animationProgress))
    : layers.length;

  return (
    <group>
      {layers.slice(0, visibleCount).map((layerData, i) => {
        const yPos = i * LAYER_GAP;
        const wuRatio = layerData.w_u_ratio ?? 0.15;
        const wuPerpRatio = layerData.w_u_perp_ratio ?? 0.85;
        const grammarInPerp = layerData.grammar_in_perp || {};
        const semInWu = layerData.semantics_in_w_u ?? 0;

        return (
          <group key={layerData.layer} position={[0, yPos, 0]}>
            {/* W_U柱体 (青绿, 右侧) */}
            <group position={[3, 0, 0]}>
              <mesh position={[0, wuRatio * 3, 0]}>
                <boxGeometry args={[1.5, Math.max(0.05, wuRatio * 6), 1.5]} />
                <meshStandardMaterial
                  color={SUBSPACE_COLORS.w_u}
                  emissive={SUBSPACE_COLORS.w_u}
                  emissiveIntensity={0.4}
                  transparent
                  opacity={0.7}
                />
              </mesh>
              <Text position={[0, -0.5, 0]} fontSize={0.25} color={SUBSPACE_COLORS.w_u} anchorX="center">
                {`W_U ${(wuRatio * 100).toFixed(0)}%`}
              </Text>
              {/* 语义能量条 */}
              {semInWu > 0 && (
                <mesh position={[0, wuRatio * 3 + 0.2, 0]}>
                  <boxGeometry args={[1.5 * semInWu, 0.15, 1.5]} />
                  <meshStandardMaterial
                    color={SUBSPACE_COLORS.semantic}
                    emissive={SUBSPACE_COLORS.semantic}
                    emissiveIntensity={0.6}
                    transparent
                    opacity={0.5}
                  />
                </mesh>
              )}
            </group>

            {/* W_U⊥柱体 (红色, 左侧) */}
            <group position={[-3, 0, 0]}>
              <mesh position={[0, wuPerpRatio * 3, 0]}>
                <boxGeometry args={[1.5, Math.max(0.05, wuPerpRatio * 6), 1.5]} />
                <meshStandardMaterial
                  color={SUBSPACE_COLORS.w_u_perp}
                  emissive={SUBSPACE_COLORS.w_u_perp}
                  emissiveIntensity={0.4}
                  transparent
                  opacity={0.7}
                />
              </mesh>
              <Text position={[0, -0.5, 0]} fontSize={0.25} color={SUBSPACE_COLORS.w_u_perp} anchorX="center">
                {`W_U⊥ ${(wuPerpRatio * 100).toFixed(0)}%`}
              </Text>

              {/* 语法角色在W_U⊥中的子条 */}
              {Object.entries(grammarInPerp).map(([role, ratio], ri) => {
                const roleColor = GRAMMAR_ROLE_COLORS[role] || '#888888';
                const barY = wuPerpRatio * 3 - ri * 0.3;
                return (
                  <group key={role} position={[0, barY, 0.8]}>
                    <mesh>
                      <boxGeometry args={[1.2 * ratio, 0.12, 0.3]} />
                      <meshStandardMaterial
                        color={roleColor}
                        emissive={roleColor}
                        emissiveIntensity={0.5}
                        transparent
                        opacity={0.8}
                      />
                    </mesh>
                    <Text position={[0.8, 0, 0]} fontSize={0.15} color={roleColor} anchorX="left">
                      {`${role} ${(ratio * 100).toFixed(0)}%`}
                    </Text>
                  </group>
                );
              })}
            </group>

            {/* 层标签 */}
            <Text
              position={[-7, 0, 0]}
              fontSize={0.4}
              color="#94a3b8"
              anchorX="right"
              anchorY="middle"
            >
              L{layerData.layer}
            </Text>

            {/* 概念点 (如果有) */}
            {(layerData.concept_points || []).map((pt, pi) => {
              const color = pt.subspace === 'w_u' ? SUBSPACE_COLORS.w_u : SUBSPACE_COLORS.w_u_perp;
              const xOff = pt.subspace === 'w_u' ? 3 : -3;
              return (
                <mesh
                  key={`cp${pi}`}
                  position={[xOff + (pt.x || 0) * 0.5, (pt.y || 0) * 2 + 2, (pt.z || 0) * 0.5]}
                  onPointerOver={(e) => {
                    e.stopPropagation();
                    onHoverToken?.({
                      token: pt.token,
                      category: pt.category,
                      subspace: pt.subspace,
                      layer: layerData.layer,
                    });
                  }}
                >
                  <sphereGeometry args={[0.15, 8, 8]} />
                  <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.6} />
                </mesh>
              );
            })}
          </group>
        );
      })}
    </group>
  );
}
