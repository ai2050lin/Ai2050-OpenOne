/**
 * GrammarRoleMatrixRenderer — 语法角色余弦矩阵3D可视化
 * 
 * 数据格式 (Schema v2.0):
 * {
 *   type: "grammar_role_matrix",
 *   roles: [str, ...],
 *   cosine_matrix: [[float, ...], ...],
 *   lda_accuracy: [float, ...],
 *   causal_effect: [float, ...],
 *   transfer_kl: [float, ...]
 * }
 */
import React from 'react';
import { Text, Line } from '@react-three/drei';
import { GRAMMAR_ROLE_COLORS, deltaCosToColor } from '../utils/constants';

export default function GrammarRoleMatrixRenderer({ grammarMatrix, onHoverToken }) {
  const roles = grammarMatrix?.roles || [];
  const cosMatrix = grammarMatrix?.cosine_matrix || [];
  const ldaAcc = grammarMatrix?.lda_accuracy || [];
  const causalEff = grammarMatrix?.causal_effect || [];
  const transferKL = grammarMatrix?.transfer_kl || [];

  if (roles.length === 0) return null;

  const cellSize = 2;
  const offset = (roles.length - 1) * cellSize / 2;

  return (
    <group>
      {/* 余弦矩阵3D柱状图 */}
      {cosMatrix.map((row, ri) =>
        row.map((val, ci) => {
          const height = Math.abs(val) * 4;
          const color = val >= 0 ? deltaCosToColor(0.5 + val * 0.5) : '#3b82f6';
          const xPos = ri * cellSize - offset;
          const zPos = ci * cellSize - offset;
          return (
            <group key={`${ri}-${ci}`} position={[xPos, height / 2, zPos]}>
              <mesh
                onPointerOver={(e) => {
                  e.stopPropagation();
                  onHoverToken?.({
                    role_pair: `${roles[ri]} × ${roles[ci]}`,
                    cosine: val,
                    type: 'grammar_matrix',
                  });
                }}
              >
                <boxGeometry args={[cellSize * 0.85, Math.max(0.05, height), cellSize * 0.85]} />
                <meshStandardMaterial
                  color={color}
                  emissive={color}
                  emissiveIntensity={0.3}
                  roughness={0.5}
                  transparent
                  opacity={0.8}
                />
              </mesh>
              {/* 对角线标记 */}
              {ri === ci && (
                <mesh position={[0, height + 0.1, 0]} rotation={[-Math.PI / 2, 0, 0]}>
                  <ringGeometry args={[0.3, 0.4, 16]} />
                  <meshStandardMaterial
                    color={GRAMMAR_ROLE_COLORS[roles[ri]] || '#ffffff'}
                    emissive={GRAMMAR_ROLE_COLORS[roles[ri]] || '#ffffff'}
                    emissiveIntensity={1}
                    side={2}
                  />
                </mesh>
              )}
            </group>
          );
        })
      )}

      {/* 行标签 (角色名) */}
      {roles.map((role, ri) => {
        const roleColor = GRAMMAR_ROLE_COLORS[role] || '#94a3b8';
        return (
          <group key={`role-${ri}`}>
            <Text
              position={[ri * cellSize - offset, -0.5, -(offset + 1.5)]}
              fontSize={0.3}
              color={roleColor}
              anchorX="center"
            >
              {role}
            </Text>
            <Text
              position={[-(offset + 1.5), -0.5, ri * cellSize - offset]}
              fontSize={0.3}
              color={roleColor}
              anchorX="center"
              rotation={[0, 0, -Math.PI / 2]}
            >
              {role}
            </Text>
          </group>
        );
      })}

      {/* LDA准确率条 (右侧) */}
      {ldaAcc.length > 0 && (
        <group position={[offset + 3, 0, 0]}>
          <Text position={[0, 5.5, 0]} fontSize={0.25} color="#94a3b8" anchorX="center">
            LDA Accuracy
          </Text>
          {ldaAcc.map((acc, ri) => {
            const h = acc * 5;
            const color = GRAMMAR_ROLE_COLORS[roles[ri]] || '#888';
            return (
              <group key={`lda${ri}`} position={[0, 0, ri * cellSize - offset]}>
                <mesh position={[0, h / 2, 0]}>
                  <boxGeometry args={[0.5, Math.max(0.05, h), cellSize * 0.7]} />
                  <meshStandardMaterial
                    color={color}
                    emissive={color}
                    emissiveIntensity={0.3}
                    transparent
                    opacity={0.7}
                  />
                </mesh>
                <Text position={[0.5, h, 0]} fontSize={0.2} color={color} anchorX="left">
                  {acc.toFixed(2)}
                </Text>
              </group>
            );
          })}
        </group>
      )}

      {/* 因果效应条 (左侧) */}
      {causalEff.length > 0 && (
        <group position={[-(offset + 3), 0, 0]}>
          <Text position={[0, 5.5, 0]} fontSize={0.25} color="#94a3b8" anchorX="center">
            Causal Effect
          </Text>
          {causalEff.map((eff, ri) => {
            const h = eff * 20; // 放大显示
            const color = GRAMMAR_ROLE_COLORS[roles[ri]] || '#888';
            return (
              <group key={`causal${ri}`} position={[0, 0, ri * cellSize - offset]}>
                <mesh position={[0, h / 2, 0]}>
                  <boxGeometry args={[0.5, Math.max(0.05, h), cellSize * 0.7]} />
                  <meshStandardMaterial
                    color="#f97316"
                    emissive="#f97316"
                    emissiveIntensity={0.3}
                    transparent
                    opacity={0.7}
                  />
                </mesh>
                <Text position={[-0.5, h, 0]} fontSize={0.2} color="#f97316" anchorX="right">
                  {eff.toFixed(3)}
                </Text>
              </group>
            );
          })}
        </group>
      )}

      {/* 迁移KL条 (前方) */}
      {transferKL.length > 0 && (
        <group position={[0, 0, -(offset + 3)]}>
          <Text position={[0, 5.5, 0]} fontSize={0.25} color="#94a3b8" anchorX="center">
            Transfer KL
          </Text>
          {transferKL.map((kl, ri) => {
            const h = kl * 0.3;
            const color = GRAMMAR_ROLE_COLORS[roles[ri]] || '#888';
            return (
              <group key={`kl${ri}`} position={[ri * cellSize - offset, 0, 0]}>
                <mesh position={[0, h / 2, 0]}>
                  <boxGeometry args={[cellSize * 0.7, Math.max(0.05, h), 0.5]} />
                  <meshStandardMaterial
                    color="#a855f7"
                    emissive="#a855f7"
                    emissiveIntensity={0.3}
                    transparent
                    opacity={0.7}
                  />
                </mesh>
                <Text position={[0, h + 0.2, 0]} fontSize={0.2} color="#a855f7" anchorX="center">
                  {kl.toFixed(1)}
                </Text>
              </group>
            );
          })}
        </group>
      )}
    </group>
  );
}
