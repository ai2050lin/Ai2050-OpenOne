import { Text } from '@react-three/drei';
import { BarChart3, Info } from 'lucide-react';

import { HEATMAP_ROUTE_PREVIEW } from '../../researchKernel/heatmapResearchRoute';

import './ResearchHeatmapRoute.css';

function heatmapColor(value) {
  const clamped = Math.max(0, Math.min(1, Number(value) || 0));
  if (clamped < 0.33) {
    const t = clamped / 0.33;
    return `rgb(${Math.round(20 + 15 * t)}, ${Math.round(70 + 125 * t)}, ${Math.round(180 + 45 * t)})`;
  }
  if (clamped < 0.66) {
    const t = (clamped - 0.33) / 0.33;
    return `rgb(${Math.round(35 + 205 * t)}, ${Math.round(195 + 25 * t)}, ${Math.round(225 - 135 * t)})`;
  }
  const t = (clamped - 0.66) / 0.34;
  return `rgb(${Math.round(240 + 10 * t)}, ${Math.round(220 - 150 * t)}, ${Math.round(90 - 35 * t)})`;
}

export function ResearchHeatmapRouteCard() {
  const preview = HEATMAP_ROUTE_PREVIEW;

  return (
    <section className="research-heatmap-card" aria-label="热力图研究路线预览">
      <header>
        <div><BarChart3 size={13} />热力图效果</div>
        <span>{preview.dataStatus}</span>
      </header>
      <div className="research-heatmap-card__matrix" style={{ '--heatmap-columns': preview.xAxis.length }}>
        {preview.values.flatMap((row, rowIndex) => row.map((value, columnIndex) => (
          <span
            key={`${rowIndex}-${columnIndex}`}
            style={{ background: heatmapColor(value) }}
            title={`${preview.yAxis[rowIndex]} / ${preview.xAxis[columnIndex]}: ${value.toFixed(2)}`}
          >
            {value.toFixed(2)}
          </span>
        )))}
      </div>
      <div className="research-heatmap-card__axis">
        <span>层深度 ↓</span>
        <b>{preview.xAxis.join(' · ')}</b>
      </div>
      <div className="research-heatmap-card__legend">
        <i />
        {preview.legend.map((item) => <span key={item.value}>{item.label} {item.value.toFixed(1)}</span>)}
      </div>
      <p><Info size={11} />{preview.boundary}</p>
    </section>
  );
}

export function ResearchHeatmapPreview3D() {
  const preview = HEATMAP_ROUTE_PREVIEW;
  const spacing = 1.35;
  const xOffset = ((preview.xAxis.length - 1) * spacing) / 2;
  const zOffset = ((preview.yAxis.length - 1) * spacing) / 2;

  return (
    <group position={[0, 1.5, 0]}>
      <Text position={[0, 6.1, 0]} fontSize={0.62} color="#e0f2fe" anchorX="center">
        {preview.title}
      </Text>
      <Text position={[0, 5.45, 0]} fontSize={0.28} color="#fbbf24" anchorX="center">
        {preview.dataStatus}
      </Text>

      {preview.values.flatMap((row, rowIndex) => row.map((value, columnIndex) => {
        const height = 0.18 + value * 3.5;
        const color = heatmapColor(value);
        return (
          <group
            key={`${rowIndex}-${columnIndex}`}
            position={[columnIndex * spacing - xOffset, height / 2, rowIndex * spacing - zOffset]}
          >
            <mesh>
              <boxGeometry args={[1.02, height, 1.02]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={0.28 + value * 0.35}
                roughness={0.42}
              />
            </mesh>
            <Text position={[0, height / 2 + 0.24, 0]} fontSize={0.2} color="#f8fafc" anchorX="center">
              {value.toFixed(2)}
            </Text>
          </group>
        );
      }))}

      {preview.xAxis.map((label, index) => (
        <Text
          key={label}
          position={[index * spacing - xOffset, -0.35, -zOffset - 0.9]}
          fontSize={0.26}
          color="#bae6fd"
          anchorX="center"
        >
          {label}
        </Text>
      ))}
      {preview.yAxis.map((label, index) => (
        <Text
          key={label}
          position={[-xOffset - 1.0, -0.2, index * spacing - zOffset]}
          fontSize={0.24}
          color="#94a3b8"
          anchorX="right"
        >
          {label}
        </Text>
      ))}

      <mesh position={[0, -0.18, 0]}>
        <boxGeometry args={[preview.xAxis.length * spacing + 0.4, 0.08, preview.yAxis.length * spacing + 0.4]} />
        <meshStandardMaterial color="#0f172a" transparent opacity={0.76} />
      </mesh>
      <group position={[xOffset + 2.0, 1.5, 0]}>
        <Layers3DLegend />
      </group>
    </group>
  );
}

function Layers3DLegend() {
  const preview = HEATMAP_ROUTE_PREVIEW;
  return (
    <group>
      <Text position={[0, 1.25, 0]} fontSize={0.28} color="#cbd5e1" anchorX="left">
        强度
      </Text>
      {preview.legend.map((item, index) => (
        <group key={item.value} position={[0, 0.7 - index * 0.65, 0]}>
          <mesh position={[0.18, 0, 0]}>
            <boxGeometry args={[0.36, 0.36, 0.12]} />
            <meshBasicMaterial color={heatmapColor(item.value)} />
          </mesh>
          <Text position={[0.52, 0, 0]} fontSize={0.22} color="#94a3b8" anchorX="left">
            {item.label} {item.value.toFixed(1)}
          </Text>
        </group>
      ))}
    </group>
  );
}
