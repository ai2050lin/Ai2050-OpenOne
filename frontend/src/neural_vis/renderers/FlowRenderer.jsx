/**
 * FlowRenderer — 注意力弧线+粒子流动
 */
import React from 'react';
import { Line, Text } from '@react-three/drei';

export default function FlowRenderer({ flow, animated, animationProgress }) {
  const flows = flow?.flows || [];
  const nodes = flow?.node_positions || [];

  // 构建node位置映射
  const nodeMap = {};
  nodes.forEach(n => { nodeMap[n.id] = n; });

  return (
    <group>
      {/* 节点 */}
      {nodes.map((node, i) => (
        <group key={`node${i}`} position={[node.x, node.y, node.z]}>
          <mesh>
            <sphereGeometry args={[0.4, 16, 16]} />
            <meshStandardMaterial color="#4ecdc4" emissive="#4ecdc4" emissiveIntensity={0.5} />
          </mesh>
          <Text position={[0, 0.7, 0]} fontSize={0.3} color="#e2e8f0" anchorX="center">
            {node.token}
          </Text>
        </group>
      ))}
      {/* 注意力弧线 */}
      {flows.map((f, i) => {
        const src = nodeMap[f.source];
        const tgt = nodeMap[f.target];
        if (!src || !tgt) return null;
        
        const midY = Math.max(src.y || 0, tgt.y || 0) + 1 + f.weight * 2;
        const curvePoints = [];
        const segments = 20;
        for (let s = 0; s <= segments; s++) {
          const t = s / segments;
          const x = (src.x || 0) * (1 - t) + (tgt.x || 0) * t;
          const y = ((src.y || 0) * (1 - t) + (tgt.y || 0) * t) + Math.sin(t * Math.PI) * midY;
          const z = (src.z || 0) * (1 - t) + (tgt.z || 0) * t;
          curvePoints.push([x, y, z]);
        }
        
        return (
          <Line
            key={`flow${i}`}
            points={curvePoints}
            color={f.color || '#4ecdc4'}
            lineWidth={1 + f.weight * 4}
            transparent
            opacity={0.3 + f.weight * 0.5}
          />
        );
      })}
    </group>
  );
}
