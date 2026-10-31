/**
 * 因果视图3D叠加效果
 * 因果流线粒子 + 1D流形方向箭头
 */
import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { DIMENSION_COLORS } from '../../config/reverseColorMaps';

function CausalParticle({ start, end, color, speed = 1 }) {
  const meshRef = useRef();
  const direction = useMemo(() => {
    return new THREE.Vector3(
      end[0] - start[0],
      end[1] - start[1],
      end[2] - start[2]
    ).normalize();
  }, [start, end]);

  useFrame((state) => {
    if (meshRef.current) {
      const t = (state.clock.elapsedTime * speed * 0.3) % 1;
      meshRef.current.position.set(
        start[0] + (end[0] - start[0]) * t,
        start[1] + (end[1] - start[1]) * t,
        start[2] + (end[2] - start[2]) * t,
      );
      meshRef.current.material.opacity = 0.6 + Math.sin(t * Math.PI) * 0.4;
    }
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[0.06, 8, 8]} />
      <meshBasicMaterial color={color} transparent opacity={0.8} depthWrite={false} />
    </mesh>
  );
}

function ManifoldArrow({ start, end, color }) {
  const direction = useMemo(() => {
    return new THREE.Vector3(
      end[0] - start[0],
      end[1] - start[1],
      end[2] - start[2]
    ).normalize();
  }, [start, end]);

  const origin = useMemo(() => new THREE.Vector3(...start), [start]);
  const length = useMemo(() => {
    return new THREE.Vector3(
      end[0] - start[0],
      end[1] - start[1],
      end[2] - start[2]
    ).length();
  }, [start, end]);

  return (
    <arrowHelper args={[direction, origin, length, color, 0.3, 0.15]} />
  );
}

export default function CausalFlowOverlay({ activeDimensions, selectedDNNFeature, layerPositions, links }) {
  // Generate flow paths between layers
  const flowPaths = useMemo(() => {
    if (layerPositions.length < 2) {
      // Fallback paths
      return Array.from({ length: 5 }, (_, i) => ({
        start: [i * 0.5 - 1, -4, 0],
        end: [i * 0.5 - 1, 4, 0],
        color: '#ff6b6b',
      }));
    }

    const paths = [];
    const sortedPositions = [...layerPositions].sort((a, b) => a.y - b.y);

    // Create causal flow paths for each active dimension
    activeDimensions.forEach((dimId) => {
      const color = DIMENSION_COLORS[dimId] || '#ff6b6b';
      for (let i = 0; i < sortedPositions.length - 1; i += 2) {
        paths.push({
          start: [sortedPositions[i].x, sortedPositions[i].y, sortedPositions[i].z],
          end: [sortedPositions[i + 1].x, sortedPositions[i + 1].y, sortedPositions[i + 1].z],
          color,
        });
      }
    });

    // Add default causal flow if no dimensions selected
    if (paths.length === 0) {
      for (let i = 0; i < sortedPositions.length - 1; i += 2) {
        paths.push({
          start: [sortedPositions[i].x, sortedPositions[i].y, sortedPositions[i].z],
          end: [sortedPositions[i + 1].x, sortedPositions[i + 1].y, sortedPositions[i + 1].z],
          color: '#ff6b6b',
        });
      }
    }

    return paths;
  }, [layerPositions, activeDimensions]);

  return (
    <group>
      {/* Causal flow particles */}
      {flowPaths.map((path, idx) => (
        <group key={`flow-${idx}`}>
          <CausalParticle
            start={path.start}
            end={path.end}
            color={path.color}
            speed={0.5 + idx * 0.2}
          />
          <CausalParticle
            start={path.start}
            end={path.end}
            color={path.color}
            speed={0.3 + idx * 0.15}
          />
        </group>
      ))}

      {/* 1D manifold direction arrows */}
      {flowPaths.map((path, idx) => (
        <ManifoldArrow
          key={`manifold-${idx}`}
          start={path.start}
          end={path.end}
          color={path.color}
        />
      ))}
    </group>
  );
}
