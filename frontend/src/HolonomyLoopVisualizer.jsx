import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Line, Html } from '@react-three/drei';
import * as THREE from 'three';

// 语义节点数据 (Man -> King -> Queen -> Woman -> Man)
const NODES = [
  { id: 'man', label: 'Man', pos: [-2, -2, 0], color: '#4a90e2' },
  { id: 'king', label: 'King', pos: [-2, 2, 0], color: '#f5a623' },
  { id: 'queen', label: 'Queen', pos: [2, 2, 0], color: '#e056fd' },
  { id: 'woman', label: 'Woman', pos: [2, -2, 0], color: '#ff6b6b' },
];

// 联络光流组件
const ConnectionFlow = ({ start, end, progress, curvature }) => {
  const points = useMemo(() => {
    return [new THREE.Vector3(...start), new THREE.Vector3(...end)];
  }, [start, end]);

  return (
    <group>
      {/* 基础连线 */}
      <Line points={points} color="#ffffff" opacity={0.3} transparent lineWidth={1} />
      {/* 动态光流粒子 */}
      <mesh position={[
        start[0] + (end[0] - start[0]) * progress,
        start[1] + (end[1] - start[1]) * progress,
        start[2] + (end[2] - start[2]) * progress
      ]}>
        <sphereGeometry args={[0.15, 16, 16]} />
        <meshStandardMaterial color="#00ff00" emissive="#00ff00" emissiveIntensity={2} />
      </mesh>
    </group>
  );
};

// 偏差裂隙组件 (Holonomy Gap)
const HolonomyGap = ({ position, deviation }) => {
  if (deviation < 0.001) return null; // 零曲率时不显示

  return (
    <group position={position}>
      <mesh>
        <torusGeometry args={[0.8, 0.1, 16, 32]} />
        <meshStandardMaterial color="#ff0000" emissive="#ff0000" emissiveIntensity={1 + deviation * 10} />
      </mesh>
      <Html position={[0, 1, 0]}>
        <div style={{ color: 'red', fontWeight: 'bold', background: 'rgba(0,0,0,0.8)', padding: '4px' }}>
          GAP: {deviation.toFixed(6)}
        </div>
      </Html>
    </group>
  );
};

const Scene = ({ layerDeviation }) => {
  const groupRef = useRef();
  
  // 简单的旋转动画
  useFrame((state) => {
    if (groupRef.current) {
        groupRef.current.rotation.y += 0.002;
    }
  });

  // 模拟光流进度
  const [flowProgress, setFlowProgress] = React.useState(0);
  useFrame((state, delta) => {
    setFlowProgress((prev) => (prev + delta * 0.5) % 4); // 4段路径循环
  });

  return (
    <group ref={groupRef}>
      {/* 绘制节点 */}
      {NODES.map((node, idx) => (
        <group key={node.id} position={node.pos}>
          <mesh>
            <sphereGeometry args={[0.3, 32, 32]} />
            <meshStandardMaterial color={node.color} />
          </mesh>
          <Text position={[0, -0.5, 0]} fontSize={0.5} color="white">
            {node.label}
          </Text>
        </group>
      ))}

      {/* 绘制联络线与光流 */}
      {NODES.map((node, idx) => {
        const nextNode = NODES[(idx + 1) % NODES.length];
        const currentSegment = Math.floor(flowProgress);
        const segmentProgress = flowProgress - currentSegment;
        
        // 只在当前激活的段显示光流
        const isActive = idx === currentSegment;
        
        return (
          <ConnectionFlow 
            key={`${node.id}-${nextNode.id}`}
            start={node.pos} 
            end={nextNode.pos} 
            progress={isActive ? segmentProgress : 0}
            curvature={layerDeviation}
          />
        );
      })}

      {/* 起点处的偏差裂隙 (Man 节点) */}
      <HolonomyGap position={NODES[0].pos} deviation={layerDeviation} />
    </group>
  );
};

export default function HolonomyLoopVisualizer({ layer = 0, deviation = 0.0 }) {
  return (
    <div style={{ width: '100%', height: '400px', background: '#111' }}>
      <Canvas camera={{ position: [0, 0, 10], fov: 50 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />
        <Scene layerDeviation={deviation} />
        <OrbitControls />
        <gridHelper args={[20, 20, 0x222222, 0x111111]} rotation={[Math.PI / 2, 0, 0]} />
      </Canvas>
      
      {/* HUD 信息面板 */}
      <div style={{
        position: 'absolute', 
        top: 10, 
        left: 10, 
        color: '#0f0', 
        fontFamily: 'monospace',
        background: 'rgba(0,0,0,0.7)',
        padding: '10px',
        border: '1px solid #0f0'
      }}>
        <div>Running Cycle: Man -> King -> Queen -> Woman</div>
        <div>Layer: {layer}</div>
        <div>Holonomy Deviation: <span style={{color: deviation > 0.001 ? 'red' : '#0f0'}}>{deviation.toFixed(6)}</span></div>
        <div style={{fontSize: '0.8em', color: '#888'}}>
          {deviation < 0.001 ? "✓ CONNECTION INTEGRABLE (FLAT)" : "⚠ CURVATURE DETECTED"}
        </div>
      </div>
    </div>
  );
}
