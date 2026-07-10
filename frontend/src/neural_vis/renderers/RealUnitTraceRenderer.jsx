import { Line, Text } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import { useMemo, useRef } from 'react';

const COLORS = {
  stable: '#fbbf24',
  active: '#22d3ee',
  causalGroup: '#34d399',
  attention: '#60a5fa',
  residual: '#a78bfa',
  readout: '#fb7185',
};

function unitPosition(unit, snapshot) {
  const layerCount = Math.max(1, Number(snapshot?.num_hidden_layers || 1));
  const layer = Math.max(0, Math.min(layerCount - 1, Number(unit.layer || 0)));
  const z = (layer - (layerCount - 1) / 2) * 0.92;
  const kind = unit.unit_kind || 'mlp_product_neuron';
  let radius = 6.2;
  let index = Number(unit.unit_index || 0);
  let count = Math.max(1, Number(snapshot?.intermediate_size || 1));
  if (kind === 'attention_head_channel') {
    const headDim = Math.max(1, Number(snapshot?.head_dim || 1));
    index = Number(unit.head_index || 0) * headDim + Number(unit.unit_index || 0);
    count = Math.max(1, Number(snapshot?.num_attention_heads || 1) * headDim);
    radius = 4.6;
  } else if (kind === 'residual_dimension') {
    count = Math.max(1, Number(snapshot?.hidden_size || 1));
    radius = 3.1;
  } else if (kind === 'unembedding_token') {
    count = Math.max(1, Number(snapshot?.vocab_size || 1));
    radius = 7.5;
  }
  const angle = (index / count) * Math.PI * 2;
  return [Math.cos(angle) * radius, Math.sin(angle) * radius, z];
}

function unitId(unit) {
  return [unit.layer, unit.unit_kind, unit.head_index ?? '', unit.unit_index].join(':');
}

function TraceUnit({ unit, snapshot, active = false, stable = false, onHover }) {
  const ref = useRef(null);
  const position = useMemo(() => unitPosition(unit, snapshot), [snapshot, unit]);
  const causalGroup = unit.causal_scope === 'channel_group_not_single_unit';
  const color = active
    ? COLORS.active
    : causalGroup
      ? COLORS.causalGroup
      : stable
        ? COLORS.stable
        : unit.unit_kind === 'attention_head_channel'
          ? COLORS.attention
          : unit.unit_kind === 'residual_dimension'
            ? COLORS.residual
            : COLORS.readout;

  useFrame((state) => {
    if (!ref.current) return;
    const pulse = active ? 1 + Math.sin(state.clock.elapsedTime * 5 + Number(unit.unit_index || 0) * 0.01) * 0.22 : 1;
    ref.current.scale.setScalar(pulse);
  });

  return (
    <group position={position}>
      <mesh
        ref={ref}
        renderOrder={90}
        onPointerOver={(event) => {
          event.stopPropagation();
          onHover?.({
            token: `${unit.unit_kind} ${unit.head_index != null ? `H${unit.head_index}:` : ''}${unit.unit_index}`,
            label: `${snapshot?.model || unit.model} · L${unit.layer}`,
            layer: unit.layer,
            neuron: unit.unit_index,
            unit_kind: unit.unit_kind,
            head_index: unit.head_index,
            activation: unit.value ?? unit.activation,
            score: unit.candidate_score ?? unit.magnitude,
            evidence_level: unit.evidence_level || (active ? 'L2' : 'L4'),
            causal_scope: unit.causal_scope || 'not_tested',
            source: unit.source_artifact,
            is_real_unit: true,
          });
        }}
        onPointerOut={() => onHover?.(null)}
      >
        <sphereGeometry args={[active ? 0.2 : 0.13, 14, 14]} />
        <meshBasicMaterial color={color} transparent opacity={active ? 1 : 0.72} depthTest={false} toneMapped={false} />
      </mesh>
      {active && (
        <mesh scale={2.4} renderOrder={89}>
          <sphereGeometry args={[0.2, 12, 12]} />
          <meshBasicMaterial color={color} transparent opacity={0.18} depthTest={false} toneMapped={false} />
        </mesh>
      )}
    </group>
  );
}

export default function RealUnitTraceRenderer({
  trace,
  stableUnits = [],
  currentEvent = null,
  currentLayer = null,
  onHover,
}) {
  const snapshot = trace?.model_snapshot;
  const layerCount = Number(snapshot?.num_hidden_layers || 0);
  const stableLayerUnits = useMemo(() => {
    if (currentLayer == null) return [];
    return stableUnits
      .filter((unit) => Number(unit.layer) === Number(currentLayer))
      .sort((a, b) => Number(b.candidate_score || 0) - Number(a.candidate_score || 0))
      .slice(0, 64);
  }, [currentLayer, stableUnits]);
  const activeUnits = useMemo(
    () => (currentEvent?.top_units || []).map((unit) => ({ ...unit, layer: currentEvent.layer, evidence_level: 'L2', source_artifact: currentEvent.source_artifact })),
    [currentEvent]
  );
  const activeIds = useMemo(() => new Set(activeUnits.map(unitId)), [activeUnits]);

  if (!trace || currentLayer == null || !snapshot) return null;
  const z = (Number(currentLayer) - (layerCount - 1) / 2) * 0.92;
  return (
    <group>
      <Line points={[[0, 0, z], [6.2, 0, z]]} color="#334155" transparent opacity={0.36} lineWidth={1} />
      {stableLayerUnits.map((unit) => (
        <TraceUnit
          key={`stable-${unitId(unit)}`}
          unit={unit}
          snapshot={snapshot}
          stable
          active={activeIds.has(unitId(unit))}
          onHover={onHover}
        />
      ))}
      {activeUnits.filter((unit) => !stableLayerUnits.some((stable) => unitId(stable) === unitId(unit))).map((unit) => (
        <TraceUnit key={`active-${unitId(unit)}`} unit={unit} snapshot={snapshot} active onHover={onHover} />
      ))}
      <Text position={[8.4, 5.7, z]} color="#e0f2fe" fontSize={0.3} anchorX="left">
        {`${snapshot.model} · L${currentLayer} · ${currentEvent?.event_type || 'trace'}`}
      </Text>
      <Text position={[8.4, 5.25, z]} color="#94a3b8" fontSize={0.2} anchorX="left">
        {`真实单元 ${activeUnits.length} · 稳定候选 ${stableLayerUnits.length} · ${trace.run_id}`}
      </Text>
    </group>
  );
}
