import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, Text, Html, Tube, Box, Sphere, Line } from '@react-three/drei';
import * as THREE from 'three';

const LAYER_COUNT = 5;
const LAYER_HEIGHT = 8;
const TOTAL_HEIGHT = (LAYER_COUNT - 1) * LAYER_HEIGHT;

// Colors for the 5 functional spaces
const FUNC_COLORS = ['#ff2a2a', '#2a80ff', '#2aff80', '#ffea2a', '#b72aff'];
const FUNC_LABELS = ['Syntax', 'Semantic', 'Style', 'Tense', 'Polarity'];

// A component to draw the functional orthogonal axes
const FunctionalAxes = ({ position, rotation }) => {
  const groupRef = useRef();

  // Rotate smoothly over time or just keep the static rotation prop
  useFrame((state) => {
    if (groupRef.current) {
      // Gentle floating/breathing rotation
      groupRef.current.rotation.y = rotation[1] + Math.sin(state.clock.elapsedTime * 0.2) * 0.1;
    }
  });

  return (
    <group position={position} ref={groupRef} rotation={rotation}>
      {/* Inner Core Glow */}
      <Sphere args={[0.8, 32, 32]}>
        <meshBasicMaterial color="#ffffff" transparent opacity={0.15} wireframe />
      </Sphere>
      
      {/* 5 Dimensions (using lines or thin cylinders) */}
      {[...Array(5)].map((_, i) => {
        // Create somewhat orthogonal vectors in 3D for visualization
        // (Since it's 5D in 3D space, we just distribute them evenly)
        const phi = Math.acos(-1 + (2 * i) / 5);
        const theta = Math.sqrt(5 * Math.PI) * phi;
        const x = Math.cos(theta) * Math.sin(phi);
        const y = Math.sin(theta) * Math.sin(phi);
        const z = Math.cos(phi);
        const endPoint = [x * 2.5, y * 2.5, z * 2.5];

        return (
          <group key={i}>
            <Line
              points={[[0, 0, 0], endPoint]}
              color={FUNC_COLORS[i]}
              lineWidth={4}
              transparent
              opacity={0.8}
            />
            <mesh position={endPoint}>
              <sphereGeometry args={[0.15, 16, 16]} />
              <meshStandardMaterial color={FUNC_COLORS[i]} emissive={FUNC_COLORS[i]} emissiveIntensity={2} />
            </mesh>
            {/* Hover Label could go here */}
          </group>
        );
      })}
    </group>
  );
};

// Particles representing the content/non-functional space
const ContentParticles = () => {
  const count = 1000;
  const meshRef = useRef();
  const dummy = useMemo(() => new THREE.Object3D(), []);
  
  const particles = useMemo(() => {
    const temp = [];
    for (let i = 0; i < count; i++) {
      // Cylindrical distribution around the main trunk
      const radius = 3 + Math.random() * 8;
      const angle = Math.random() * Math.PI * 2;
      const y = -TOTAL_HEIGHT / 2 + Math.random() * TOTAL_HEIGHT * 1.2;
      const speed = 0.05 + Math.random() * 0.1;
      temp.push({ radius, angle, y, speed, x: Math.cos(angle) * radius, z: Math.sin(angle) * radius });
    }
    return temp;
  }, [count]);

  useFrame(() => {
    particles.forEach((particle, i) => {
      particle.y += particle.speed;
      if (particle.y > TOTAL_HEIGHT / 2 + 10) {
        particle.y = -TOTAL_HEIGHT / 2 - 5;
      }
      
      // Some rotation around the core
      particle.angle += 0.002;
      particle.x = Math.cos(particle.angle) * particle.radius;
      particle.z = Math.sin(particle.angle) * particle.radius;

      dummy.position.set(particle.x, particle.y, particle.z);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[null, null, count]}>
      <sphereGeometry args={[0.06, 8, 8]} />
      <meshBasicMaterial color="#00d2ff" transparent opacity={0.6} />
    </instancedMesh>
  );
};

const FunctionalParticles = () => {
  const count = 150;
  const meshRef = useRef();
  const dummy = useMemo(() => new THREE.Object3D(), []);
  
  const particles = useMemo(() => {
    const temp = [];
    for (let i = 0; i < count; i++) {
      // Tight cylindrical distribution inside the main trunk (the bypass)
      const radius = Math.random() * 0.8;
      const angle = Math.random() * Math.PI * 2;
      const y = -TOTAL_HEIGHT / 2 + Math.random() * TOTAL_HEIGHT * 1.2;
      const speed = 0.15 + Math.random() * 0.2; // Faster
      temp.push({ radius, angle, y, speed, x: Math.cos(angle) * radius, z: Math.sin(angle) * radius });
    }
    return temp;
  }, [count]);

  useFrame(() => {
    particles.forEach((particle, i) => {
      particle.y += particle.speed;
      if (particle.y > TOTAL_HEIGHT / 2 + 10) {
        particle.y = -TOTAL_HEIGHT / 2 - 5;
      }
      dummy.position.set(particle.x, particle.y, particle.z);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[null, null, count]}>
      <sphereGeometry args={[0.08, 8, 8]} />
      <meshBasicMaterial color="#ffc800" transparent opacity={0.9} />
    </instancedMesh>
  );
}

const AttentionFFNBlock = ({ position, type }) => {
  const isAttn = type === 'attn';
  return (
    <group position={position}>
      <Box args={[4, 2, 4]} radius={0.2} smoothness={4}>
        <meshPhysicalMaterial 
          color={isAttn ? '#0a4b6e' : '#4b0a6e'}
          transmission={0.8}
          transparent
          opacity={0.6}
          roughness={0.2}
          metalness={0.8}
          clearcoat={1}
        />
      </Box>
      <Text position={[0, 1.5, 0]} fontSize={0.6} color="white" anchorX="center" anchorY="middle">
        {isAttn ? 'Self-Attention' : 'MLP / FFN'}
      </Text>
      <Text position={[0, -1.5, 0]} fontSize={0.3} color={isAttn ? "#00ffff" : "#ff00ff"} anchorX="center" anchorY="middle">
        Content Space Only
      </Text>
    </group>
  );
};

const Layer = ({ index, yPos }) => {
  // Simulate the rotation dynamics
  // Early layers: wild rotations. Mid: smooth. Late: distinct.
  const rotX = Math.sin(index * 1.5) * 0.5;
  const rotY = index * 0.6;
  const rotZ = Math.cos(index * 1.5) * 0.3;

  return (
    <group position={[0, yPos, 0]}>
      {/* Functional Core representing h_func */}
      <FunctionalAxes position={[0, 0, 0]} rotation={[rotX, rotY, rotZ]} />
      
      {/* Attn and FFN on the sides */}
      <AttentionFFNBlock position={[-6, 0, 0]} type="attn" />
      <AttentionFFNBlock position={[6, 0, 0]} type="mlp" />

      {/* Cross connections from core to blocks (Content routing) */}
      <Line points={[[-1, 0, 0], [-4, 0, 0]]} color="#00d2ff" lineWidth={2} dashed dashScale={10} dashSize={1} gapSize={1} transparent opacity={0.5} />
      <Line points={[[1, 0, 0], [4, 0, 0]]} color="#00d2ff" lineWidth={2} dashed dashScale={10} dashSize={1} gapSize={1} transparent opacity={0.5} />
      
      <Html position={[0, 3, 0]} center>
        <div style={{ background: 'rgba(0,0,0,0.7)', padding: '4px 10px', borderRadius: '4px', color: '#fff', border: '1px solid #333', fontSize: '12px', whiteSpace: 'nowrap' }}>
          Layer {index * 8} {index === LAYER_COUNT - 1 ? '(Emergence Layer)' : ''}
        </div>
      </Html>
    </group>
  );
}

const DecodingPlane = ({ yPos }) => {
  return (
    <group position={[0, yPos, 0]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[20, 20]} />
        <meshStandardMaterial color="#222" transparent opacity={0.6} metalness={0.8} roughness={0.2} wireframe />
      </mesh>
      <Text position={[0, 0.5, -9]} fontSize={1} color="#00ffcc" anchorX="center">
        W_U Decoding Manifold
      </Text>
      <Text position={[0, -0.5, -9]} fontSize={0.6} color="#aaa" anchorX="center">
        logit_gap = h(L) · ΔW
      </Text>
    </group>
  );
}

export default function LCSVisualization() {
  const startY = -TOTAL_HEIGHT / 2;

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', background: '#050510' }}>
      <Canvas camera={{ position: [15, 5, 20], fov: 45 }}>
        <color attach="background" args={['#050508']} />
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1.5} color="#00d2ff" />
        <pointLight position={[-10, -10, -10]} intensity={1} color="#b72aff" />
        
        {/* The Residual Stream Trunk (h) */}
        <mesh position={[0, 0, 0]}>
          <cylinderGeometry args={[1.5, 1.5, TOTAL_HEIGHT + 10, 32, 1, true]} />
          <meshPhysicalMaterial 
            color="#ffffff" 
            transparent 
            opacity={0.1} 
            transmission={0.9} 
            roughness={0.1} 
            side={THREE.DoubleSide}
            depthWrite={false}
          />
        </mesh>

        {/* Generate Layers */}
        {[...Array(LAYER_COUNT)].map((_, i) => (
          <Layer key={i} index={i} yPos={startY + i * LAYER_HEIGHT} />
        ))}

        {/* Decoding Plane at the top */}
        <DecodingPlane yPos={startY + LAYER_COUNT * LAYER_HEIGHT - 2} />

        <ContentParticles />
        <FunctionalParticles />

        <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={1} />
        <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
      </Canvas>
      
      {/* UI Overlay */}
      <div style={{ position: 'absolute', top: '20px', left: '20px', color: 'white', background: 'rgba(10, 15, 30, 0.8)', padding: '20px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)', backdropFilter: 'blur(10px)', width: '300px' }}>
        <h2 style={{ margin: '0 0 15px 0', fontSize: '18px', color: '#00d2ff' }}>LCS 双层几何分离引擎</h2>
        <p style={{ fontSize: '12px', color: '#aaa', marginBottom: '20px' }}>
          基于 GLM5 理论 (v17.0): 残差流中的功能空间与内容空间完全正交。功能信号绕过 Q/K/V 直接传播。
        </p>
        
        <div style={{ marginBottom: '15px' }}>
          <div style={{ fontSize: '12px', marginBottom: '5px' }}>功能空间能量占比 (h_func)</div>
          <div style={{ width: '100%', height: '8px', background: '#222', borderRadius: '4px', overflow: 'hidden' }}>
            <div style={{ width: '5%', height: '100%', background: '#ffc800' }}></div>
          </div>
          <div style={{ fontSize: '10px', textAlign: 'right', color: '#ffc800', marginTop: '3px' }}>4.2% (W_U能量)</div>
        </div>

        <div style={{ marginBottom: '15px' }}>
          <div style={{ fontSize: '12px', marginBottom: '5px' }}>内容空间能量占比 (h_nonfunc)</div>
          <div style={{ width: '100%', height: '8px', background: '#222', borderRadius: '4px', overflow: 'hidden' }}>
            <div style={{ width: '95%', height: '100%', background: '#00d2ff' }}></div>
          </div>
          <div style={{ fontSize: '10px', textAlign: 'right', color: '#00d2ff', marginTop: '3px' }}>95.8% (W_U能量)</div>
        </div>

        <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '15px' }}>
          <div style={{ fontSize: '12px', marginBottom: '10px' }}>功能子空间维度 (正交)</div>
          {FUNC_LABELS.map((label, i) => (
            <div key={label} style={{ display: 'flex', alignItems: 'center', marginBottom: '5px' }}>
              <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: FUNC_COLORS[i], marginRight: '10px' }}></div>
              <span style={{ fontSize: '11px' }}>{label}</span>
              <span style={{ marginLeft: 'auto', fontSize: '11px', color: '#888' }}>|cos| &lt; 0.12</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
