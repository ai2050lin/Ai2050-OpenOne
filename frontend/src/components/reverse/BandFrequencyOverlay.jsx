/**
 * 频谱视图3D叠加效果
 * 5频段颜色光晕叠加在神经元上
 */
import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { BAND_COLORS } from '../../config/reverseColorMaps';

const BAND_DATA = [
  { band: 1, color: '#38bdf8', label: '极低频' },
  { band: 2, color: '#22c55e', label: '低频' },
  { band: 3, color: '#fbbf24', label: '中频' },
  { band: 4, color: '#f97316', label: '高频' },
  { band: 5, color: '#ef4444', label: '极高频' },
];

function BandHalo({ position, color, size, bandIndex }) {
  const meshRef = useRef();

  useFrame((state) => {
    if (meshRef.current) {
      const t = state.clock.elapsedTime;
      meshRef.current.material.opacity = 0.15 + Math.sin(t * 0.5 + bandIndex) * 0.05;
      meshRef.current.scale.setScalar(1 + Math.sin(t * 0.3 + bandIndex * 0.7) * 0.1);
    }
  });

  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[size, 16, 16]} />
      <meshBasicMaterial
        color={color}
        transparent
        opacity={0.15}
        depthWrite={false}
        side={THREE.BackSide}
      />
    </mesh>
  );
}

export default function BandFrequencyOverlay({ activeDimensions, selectedDNNFeature, layerPositions, nodes }) {
  // Place band halos at layer positions
  const haloPositions = useMemo(() => {
    if (layerPositions.length === 0) {
      // Fallback positions
      return Array.from({ length: 8 }, (_, i) => ({
        position: [0, i * 2 - 7, 0],
        band: (i % 5) + 1,
      }));
    }

    // Assign bands to layers (cycling through 5 bands)
    return layerPositions.slice(0, 16).map((pos, i) => ({
      position: [pos.x, pos.y, pos.z],
      band: (i % 5) + 1,
    }));
  }, [layerPositions]);

  return (
    <group>
      {/* Band halos at each layer */}
      {haloPositions.map((halo, idx) => {
        const bandData = BAND_DATA[halo.band - 1];
        return (
          <BandHalo
            key={`halo-${idx}`}
            position={halo.position}
            color={bandData.color}
            size={0.8}
            bandIndex={halo.band}
          />
        );
      })}

      {/* Band legend spheres */}
      <group position={[-6, 4, 0]}>
        {BAND_DATA.map((band, i) => (
          <group key={band.band} position={[0, -i * 0.6, 0]}>
            <mesh>
              <sphereGeometry args={[0.15, 12, 12]} />
              <meshBasicMaterial color={band.color} transparent opacity={0.8} />
            </mesh>
          </group>
        ))}
      </group>
    </group>
  );
}
