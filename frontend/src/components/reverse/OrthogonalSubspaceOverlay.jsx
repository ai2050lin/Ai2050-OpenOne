/**
 * 正交视图3D叠加效果
 * 半透明彩色子空间平面 + 差分向量箭头
 */
import { useMemo } from 'react';
import * as THREE from 'three';
import { DIMENSION_COLORS } from '../../config/reverseColorMaps';

const DIM_LABELS = {
  syntax: '语法', semantic: '语义', logic: '逻辑',
  pragmatic: '语用', morphological: '形态',
};

const DIM_ANGLES = {
  syntax: 0, semantic: Math.PI / 3, logic: Math.PI * 2 / 3,
  pragmatic: Math.PI, morphological: Math.PI * 4 / 3,
};

export default function OrthogonalSubspaceOverlay({ activeDimensions, selectedDNNFeature, layerPositions }) {
  const subspaces = useMemo(() => {
    return activeDimensions.map((dimId) => {
      const color = DIMENSION_COLORS[dimId] || '#ffffff';
      const angle = DIM_ANGLES[dimId] || 0;
      const label = DIM_LABELS[dimId] || dimId;
      return { dimId, color, angle, label };
    });
  }, [activeDimensions]);

  if (subspaces.length === 0) return null;

  // Use a fixed set of positions for the subspace planes
  const planePositions = layerPositions.length > 0
    ? layerPositions.filter((_, i) => i % 4 === 0).slice(0, 8)
    : Array.from({ length: 8 }, (_, i) => ({ x: 0, y: i * 2 - 7, z: 0 }));

  return (
    <group>
      {/* Subspace planes for each active dimension */}
      {subspaces.map((sub, idx) => (
        <group key={sub.dimId}>
          {/* Semi-transparent subspace planes at key layers */}
          {planePositions.map((pos, li) => (
            <mesh
              key={`${sub.dimId}-${li}`}
              position={[pos.x + idx * 0.3, pos.y, pos.z]}
              rotation={[0, sub.angle, 0]}
            >
              <planeGeometry args={[3, 1.5]} />
              <meshBasicMaterial
                color={sub.color}
                transparent
                opacity={0.08}
                side={THREE.DoubleSide}
                depthWrite={false}
              />
            </mesh>
          ))}

          {/* Differential vector arrows between layers */}
          {planePositions.slice(0, -1).map((pos, li) => {
            const nextPos = planePositions[li + 1];
            const direction = new THREE.Vector3(
              nextPos.x - pos.x + (idx - subspaces.length / 2) * 0.1,
              nextPos.y - pos.y,
              nextPos.z - pos.z
            ).normalize();

            return (
              <group key={`arrow-${sub.dimId}-${li}`}>
                <arrowHelper
                  args={[
                    direction,
                    new THREE.Vector3(pos.x + idx * 0.3, pos.y, pos.z),
                    1.5,
                    sub.color,
                    0.3,
                    0.15,
                  ]}
                />
              </group>
            );
          })}
        </group>
      ))}
    </group>
  );
}
