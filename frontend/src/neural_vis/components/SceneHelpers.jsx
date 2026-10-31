/**
 * SceneHelpers — 场景辅助(网格/标尺)
 */
import React from 'react';
import { Text, Stars } from '@react-three/drei';
import { LAYER_GAP, PLANE_SIZE } from '../utils/constants';

export default function SceneHelpers({ nLayers = 36 }) {
  return (
    <group>
      {/* 地面网格 */}
      <gridHelper args={[40, 40, '#1e293b', '#0f172a']} position={[0, -1, 0]} />
      {/* Y轴标尺 (层号) */}
      {Array.from({ length: Math.min(nLayers, 37) }, (_, i) => (
        i % 6 === 0 ? (
          <Text
            key={`y${i}`}
            position={[-PLANE_SIZE / 2 - 3, i * LAYER_GAP, 0]}
            fontSize={0.3}
            color="#64748b"
            anchorX="right"
          >
            L{i}
          </Text>
        ) : null
      ))}
    </group>
  );
}
