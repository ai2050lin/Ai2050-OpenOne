import { Text } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

function GlassNode({ position, probability, color, label, actual, layer, posIndex, onHover, isActiveLayer }) {
  const mesh = useRef();
  const baseHeight = 0.4 + (probability * 0.8);

  useFrame((state) => {
    if (mesh.current && probability > 0.5) {
      const pulse = Math.sin(state.clock.elapsedTime * 2) * 0.06;
      mesh.current.scale.set(0.28, baseHeight + pulse, 0.28);
    }
  });

  return (
    <group position={position}>
      <mesh
        ref={mesh}
        onPointerOver={(e) => {
          e.stopPropagation();
          onHover({ label, actual, probability, layer, posIndex });
          document.body.style.cursor = 'pointer';
        }}
        onPointerOut={() => {
          onHover(null);
          document.body.style.cursor = 'default';
        }}
        scale={[0.28, baseHeight, 0.28]}
      >
        <boxGeometry args={[1, 1, 1]} />
        <meshPhysicalMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isActiveLayer ? 2.0 : (probability > 0.5 ? 0.8 : 0.2)}
          metalness={0.1}
          roughness={0.05}
          transmission={0.95}
          thickness={1.5}
          transparent
          opacity={0.8}
        />
      </mesh>

      {(probability > 0.3 || isActiveLayer) && (
        <Text position={[0, 1.2, 0]} fontSize={0.6} color="white" anchorX="center" anchorY="bottom">
          {label}
        </Text>
      )}
    </group>
  );
}

const getColor = (prob) => {
  const colors = [
    '#440154',
    '#4488ff',
    '#21918c',
    '#ff9f43',
    '#ff4444'
  ];
  const idx = Math.min(Math.floor(prob * (colors.length - 1) * 1.5), colors.length - 1);
  return colors[idx];
};

export function Visualization({ data, hoveredInfo, setHoveredInfo, activeLayer }) {
  if (!data) return null;

  const { logit_lens, tokens } = data;
  const nLayers = logit_lens.length;
  const seqLen = tokens.length;

  const paths = [];
  if (logit_lens.length > 0) {
    for (let pos = 0; pos < seqLen; pos++) {
      const path = [];
      for (let l = 0; l < nLayers; l++) {
        const x = pos * 2.5;
        const z = l * 2.0;
        path.push(new THREE.Vector3(x, 0, z));
      }
      paths.push(path);
    }
  }

  return (
    <>
      <group position={[-seqLen, 0, -nLayers]}>
        {logit_lens.map((layerData, layerIdx) => (
          layerData.map((posData, posIdx) => (
            <GlassNode
              key={`${layerIdx}-${posIdx}`}
              position={[posIdx * 2.5, 0, layerIdx * 2.0]}
              probability={posData.prob}
              color={getColor(posData.prob)}
              label={posData.token}
              actual={posData.actual_token}
              layer={layerIdx}
              posIndex={posIdx}
              onHover={setHoveredInfo}
              isActiveLayer={layerIdx === activeLayer}
            />
          ))
        ))}

        {tokens.map((_, i) => (
          <line key={`path-${i}`}>
            <bufferGeometry setFromPoints={paths[i]} />
            <lineBasicMaterial color="#ffffff" opacity={0.15} transparent linewidth={1} />
          </line>
        ))}

        {tokens.map((token, i) => (
          <Text
            key={`x-label-${i}`}
            position={[i * 1.2, -0.5, -1]}
            rotation={[-Math.PI / 2, 0, 0]}
            fontSize={0.3}
            color="white"
          >
            {token}
          </Text>
        ))}

        {Array.from({ length: nLayers }).map((_, i) => (
          <Text
            key={`z-label-${i}`}
            position={[-1.5, -0.5, i * 1.2]}
            rotation={[-Math.PI / 2, 0, 0]}
            fontSize={0.3}
            color="gray"
          >
            L{i}
          </Text>
        ))}
      </group>
    </>
  );
}

export function FlowParticles({ nLayers, seqLen, isPlaying }) {
  const particlesRef = useRef();
  const [particles, setParticles] = useState([]);

  useFrame(() => {
    if (!isPlaying || !particlesRef.current) return;

    if (Math.random() < 0.2) {
      const newParticle = {
        id: Math.random(),
        x: (Math.random() - 0.5) * seqLen * 1.2,
        z: 0,
        targetZ: (nLayers - 1) * 1.2,
        progress: 0,
        speed: 0.3 + Math.random() * 0.4
      };
      setParticles(prev => [...prev.slice(-50), newParticle]);
    }

    setParticles(prev => prev.map(p => ({
      ...p,
      progress: Math.min(1, p.progress + 0.008 * p.speed)
    })).filter(p => p.progress < 1));
  });

  if (!isPlaying) return null;

  return (
    <group ref={particlesRef} position={[-seqLen / 2, 4, -nLayers / 2]}>
      {particles.map(p => {
        const currentZ = p.z + (p.targetZ - p.z) * p.progress;
        const opacity = Math.sin(p.progress * Math.PI);

        return (
          <mesh key={p.id} position={[p.x, 0, currentZ]}>
            <sphereGeometry args={[0.15, 16, 16]} />
            <meshStandardMaterial
              color="#00d2ff"
              emissive="#00d2ff"
              emissiveIntensity={3}
              transparent
              opacity={opacity * 0.9}
            />
          </mesh>
        );
      })}
    </group>
  );
}

export function AttentionHeatmap({ pattern, tokens, headIdx }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !pattern) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const size = pattern.length;
    const cellSize = Math.min(200 / size, 40);

    canvas.width = size * cellSize;
    canvas.height = size * cellSize;

    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const value = pattern[i][j];
        const intensity = Math.floor(value * 255);
        ctx.fillStyle = `rgb(${intensity}, ${Math.floor(intensity * 0.5)}, ${255 - intensity})`;
        ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
      }
    }

    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= size; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cellSize, 0);
      ctx.lineTo(i * cellSize, size * cellSize);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, i * cellSize);
      ctx.lineTo(size * cellSize, i * cellSize);
      ctx.stroke();
    }
  }, [pattern]);

  return (
    <div style={{ marginBottom: '12px' }}>
      <div style={{ fontSize: '11px', fontWeight: 'bold', marginBottom: '4px', color: '#00d2ff' }}>
        头 {headIdx}
      </div>
      <canvas
        ref={canvasRef}
        style={{
          border: '1px solid #444',
          borderRadius: '4px',
          maxWidth: '100%',
          imageRendering: 'pixelated'
        }}
      />
    </div>
  );
}

export function MLPActivationChart({ distribution }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !distribution) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = 300;
    const height = 100;
    const barCount = Math.min(distribution.length, 100);
    const barWidth = width / barCount;

    canvas.width = width;
    canvas.height = height;

    const maxVal = Math.max(...distribution.slice(0, barCount));

    for (let i = 0; i < barCount; i++) {
      const value = distribution[i];
      const barHeight = (value / maxVal) * height;
      const hue = (value / maxVal) * 120;
      ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
      ctx.fillRect(i * barWidth, height - barHeight, barWidth, barHeight);
    }
  }, [distribution]);

  return (
    <div>
      <div style={{ fontSize: '11px', fontWeight: 'bold', marginBottom: '4px', color: '#00d2ff' }}>
        MLP激活分布
      </div>
      <canvas
        ref={canvasRef}
        style={{
          border: '1px solid #444',
          borderRadius: '4px',
          width: '100%'
        }}
      />
    </div>
  );
}
