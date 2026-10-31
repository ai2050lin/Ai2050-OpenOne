/**
 * Heatmap3DRenderer — 层×概念×指标3D柱状图
 * 增强: 支持层×语法角色×指标
 */
import React from 'react';
import { Text } from '@react-three/drei';
import { deltaCosToColor } from '../utils/constants';

export default function Heatmap3DRenderer({ heatmap }) {
  const cells = heatmap?.cells || [];
  const xValues = heatmap?.x_axis?.values || [];
  const yValues = heatmap?.y_axis?.values || [];
  const zRange = heatmap?.z_axis?.range || [0, 1];

  return (
    <group>
      {cells.map((cell, i) => {
        const height = ((cell.value - zRange[0]) / (zRange[1] - zRange[0] + 1e-10)) * 5;
        const xPos = (cell.x - xValues.length / 2) * 1.2;
        const zPos = (cell.y - yValues.length / 2) * 1.2;
        const color = cell.color || deltaCosToColor(cell.value);
        return (
          <group key={i} position={[xPos, height / 2, zPos]}>
            <mesh>
              <boxGeometry args={[0.9, Math.max(0.05, height), 0.9]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={0.3}
                roughness={0.5}
              />
            </mesh>
          </group>
        );
      })}
      {/* X轴标签 */}
      {xValues.map((xv, i) => (
        <Text
          key={`x${i}`}
          position={[(i - xValues.length / 2) * 1.2, -0.5, -(yValues.length / 2 + 1) * 1.2]}
          fontSize={0.3}
          color="#94a3b8"
          anchorX="center"
        >
          {String(xv)}
        </Text>
      ))}
    </group>
  );
}
