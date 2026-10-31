/**
 * PointCloudRenderer — 语义空间点云
 * 增强: 支持subspace分组着色(W_U vs W_U⊥)
 */
import React from 'react';
import { SPHERE_BASE_SIZE, CATEGORY_COLORS, SUBSPACE_COLORS } from '../utils/constants';

export default function PointCloudRenderer({ pointCloud, onHoverToken }) {
  const points = pointCloud.points || [];
  const catColors = pointCloud.categories || CATEGORY_COLORS;
  // 判断是否有子空间分组
  const hasSubspace = points.some(p => p.subspace);

  return (
    <group>
      {points.map((pt, i) => {
        const color = hasSubspace
          ? (SUBSPACE_COLORS[pt.subspace] || '#888888')
          : (catColors[pt.category] || '#888888');
        const size = SPHERE_BASE_SIZE + (pt.norm || 10) * 0.0006;
        return (
          <group key={i} position={[pt.x, pt.y, pt.z]}>
            <mesh
              onPointerOver={(e) => {
                e.stopPropagation();
                onHoverToken?.({
                  token: pt.token,
                  category: pt.category,
                  subspace: pt.subspace,
                  layer: pointCloud.layer,
                  norm: pt.norm,
                  activation: pt.activation,
                });
              }}
            >
              <sphereGeometry args={[size, 12, 12]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={0.3}
                roughness={0.4}
                metalness={0.5}
              />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}
