/**
 * 编码视图3D叠加效果
 * 编码方程文字标签 + R²指示器
 */
import { useMemo } from 'react';
import { Text } from '@react-three/drei';
import { DIMENSION_COLORS } from '../../config/reverseColorMaps';

const DIM_LABELS = {
  syntax: '语法', semantic: '语义', logic: '逻辑',
  pragmatic: '语用', morphological: '形态',
};

// Mock R² data per layer range
const MOCK_R2 = [
  { layerRange: 'L0-L4', r2: 0.42, label: '基础编码' },
  { layerRange: 'L4-L8', r2: 0.68, label: '特征提取' },
  { layerRange: 'L8-L16', r2: 0.95, label: '瓶颈层' },
  { layerRange: 'L16-L24', r2: 0.88, label: '因果路由' },
  { layerRange: 'L24-L31', r2: 0.93, label: '读出层' },
];

function R2Indicator({ position, r2, label, color }) {
  const r2Color = r2 > 0.9 ? '#22c55e' : r2 > 0.7 ? '#fbbf24' : '#ef4444';

  return (
    <group position={position}>
      {/* R² value text */}
      <Text
        position={[0, 0.3, 0]}
        fontSize={0.18}
        color={r2Color}
        anchorX="center"
        anchorY="middle"
        fontWeight={700}
      >
        {`R²=${r2.toFixed(2)}`}
      </Text>

      {/* Layer range label */}
      <Text
        position={[0, 0, 0]}
        fontSize={0.12}
        color="#aaa"
        anchorX="center"
        anchorY="middle"
      >
        {label}
      </Text>

      {/* R² bar indicator */}
      <mesh position={[0, -0.25, 0]}>
        <planeGeometry args={[1, 0.06]} />
        <meshBasicMaterial color="#333" transparent opacity={0.5} />
      </mesh>
      <mesh position={[-0.5 + r2 * 0.5, -0.25, 0.01]}>
        <planeGeometry args={[r2, 0.06]} />
        <meshBasicMaterial color={r2Color} transparent opacity={0.8} />
      </mesh>
    </group>
  );
}

export default function EncodingEquationOverlay({ activeDimensions, selectedDNNFeature, layerPositions }) {
  // Position the equation labels and R² indicators at layer positions
  const indicatorPositions = useMemo(() => {
    if (layerPositions.length === 0) {
      return MOCK_R2.map((data, i) => ({
        position: [3, i * 2 - 4, 1],
        ...data,
      }));
    }

    // Spread indicators across available layer positions
    const step = Math.max(1, Math.floor(layerPositions.length / MOCK_R2.length));
    return MOCK_R2.map((data, i) => {
      const pos = layerPositions[i * step] || layerPositions[layerPositions.length - 1];
      return {
        position: [pos.x + 3, pos.y, pos.z + 1],
        ...data,
      };
    });
  }, [layerPositions]);

  // Build encoding equation string
  const equationText = useMemo(() => {
    const dimLabels = activeDimensions.length > 0
      ? activeDimensions.map((d) => DIM_LABELS[d] || d).join('+')
      : '语言维度';
    return `logit(${dimLabels}) = Σα_band × band_logit + β`;
  }, [activeDimensions]);

  return (
    <group>
      {/* Main encoding equation */}
      <Text
        position={[0, 5, 0]}
        fontSize={0.25}
        color="#ffd93d"
        anchorX="center"
        anchorY="middle"
        maxWidth={10}
      >
        {equationText}
      </Text>

      {/* Dimension coefficients */}
      {activeDimensions.map((dimId, idx) => {
        const color = DIMENSION_COLORS[dimId] || '#ffffff';
        const label = DIM_LABELS[dimId] || dimId;
        return (
          <Text
            key={dimId}
            position={[-3 + idx * 1.5, 4.2, 0]}
            fontSize={0.15}
            color={color}
            anchorX="center"
            anchorY="middle"
          >
            {`α_${label}`}
          </Text>
        );
      })}

      {/* R² indicators at layer positions */}
      {indicatorPositions.map((ind, idx) => (
        <R2Indicator
          key={idx}
          position={ind.position}
          r2={ind.r2}
          label={ind.layerRange}
          color={activeDimensions.length > 0 ? DIMENSION_COLORS[activeDimensions[0]] : '#ffffff'}
        />
      ))}
    </group>
  );
}
