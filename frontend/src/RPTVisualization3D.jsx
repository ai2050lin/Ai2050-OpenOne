import { Text } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import { useRef } from 'react';
import * as THREE from 'three';

/**
 * RPTVisualization3D
 * 可视化神经网络中的黎曼平行传输 (RPT)
 * 展示两个语境集在流形上的分布、它们的切空间以及它们之间的传输矩阵动画。
 */
export default function RPTVisualization3D({ data, t }) {
  if (!data || !data.source || !data.target) return null;

  const { source, target, transport_matrix, layer_idx } = data;

  return (
    <group>
      <Text position={[0, 10, 0]} fontSize={0.8} color="#fff" anchorX="center">
        {t ? t('structure.rpt.title', 'Riemannian Parallel Transport (RPT)') : 'Riemannian Parallel Transport (RPT)'}
      </Text>
      <Text position={[0, 9, 0]} fontSize={0.4} color="#aaa" anchorX="center">
        Layer: {layer_idx} | Metric Preservation Verification
      </Text>

      {/* Source Context Cloud */}
      <ContextCloud 
        coords={source.coords} 
        center={source.center} 
        color="#4488ff" 
        label="Source Context" 
      />

      {/* Target Context Cloud */}
      <ContextCloud 
        coords={target.coords} 
        center={target.center} 
        color="#ff4444" 
        label="Target Context" 
      />

      {/* Tangent Spaces & Basis (PCA Components) */}
      <TangentSpace 
        center={source.center} 
        basis={source.basis} 
        color="#4488ff" 
        label="T_p M (Source)" 
      />
      <TangentSpace 
        center={target.center} 
        basis={target.basis} 
        color="#ff4444" 
        label="T_q M (Target)" 
      />

      {/* Parallel Transport Animation */}
      <TransportAnimation 
        sourceCenter={source.center} 
        targetCenter={target.center} 
        basis={source.basis} 
        transportMatrix={transport_matrix} 
      />

      <gridHelper args={[20, 20, 0x333333, 0x222222]} position={[0, -2, 0]} />
    </group>
  );
}

function ContextCloud({ coords, center, color, label }) {
  return (
    <group>
      {coords.map((pos, i) => (
        <mesh key={i} position={pos}>
          <sphereGeometry args={[0.1, 8, 8]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.5} />
        </mesh>
      ))}
      {/* Center Marker */}
      <mesh position={center}>
        <sphereGeometry args={[0.2, 16, 16]} />
        <meshStandardMaterial color="#fff" emissive={color} emissiveIntensity={2} />
      </mesh>
      <Text position={[center[0], center[1] + 0.5, center[2]]} fontSize={0.3} color={color}>
        {label}
      </Text>
    </group>
  );
}

function TangentSpace({ center, basis, color, label }) {
  if (!basis) return null;

  return (
    <group position={center}>
      {/* Axes representing the basis vectors */}
      {basis.slice(0, 3).map((v, i) => {
        const dir = new THREE.Vector3(...v).normalize();
        return (
          <primitive 
            key={i}
            object={new THREE.ArrowHelper(dir, new THREE.Vector3(0,0,0), 2, color)} 
          />
        );
      })}
      {/* Local Plane disc approximation */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]}>
        <circleGeometry args={[2.5, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.1} side={THREE.DoubleSide} />
      </mesh>
      <Text position={[0, -0.5, 0]} fontSize={0.25} color={color}>
        {label}
      </Text>
    </group>
  );
}

function TransportAnimation({ sourceCenter, targetCenter, basis, transportMatrix }) {
  const particlesCount = 5;
  const groupRef = useRef();

  useFrame((state) => {
    if (!groupRef.current) return;
    const t = (state.clock.elapsedTime % 4) / 4; // 4 second loop

    groupRef.current.children.forEach((child, i) => {
      const offset = i / particlesCount;
      let progress = (t + offset) % 1;
      
      // Interpolate position along the geodesic (straight line for PCA projection)
      const x = sourceCenter[0] + (targetCenter[0] - sourceCenter[0]) * progress;
      const y = sourceCenter[1] + (targetCenter[1] - sourceCenter[1]) * progress;
      const z = sourceCenter[2] + (targetCenter[2] - sourceCenter[2]) * progress;
      
      child.position.set(x, y, z);
      
      // Fade in/out
      const opacity = Math.sin(progress * Math.PI);
      child.material.opacity = opacity;
      
      // Scale - gentle pulse
      const s = 0.5 + Math.sin(state.clock.elapsedTime * 5 + i) * 0.1;
      child.scale.setScalar(s);
    });
  });

  return (
    <group ref={groupRef}>
      {Array.from({ length: particlesCount }).map((_, i) => (
        <mesh key={i}>
          <octahedronGeometry args={[0.2, 0]} />
          <meshStandardMaterial 
            color="#fff" 
            emissive="#ffff00" 
            emissiveIntensity={2} 
            transparent 
            opacity={0} 
          />
        </mesh>
      ))}
      
      {/* Connection Path */}
      <line>
        <bufferGeometry>
          <bufferAttribute 
            attach="attributes-position"
            count={2}
            array={new Float32Array([...sourceCenter, ...targetCenter])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#ffff00" opacity={0.2} transparent />
      </line>
    </group>
  );
}
