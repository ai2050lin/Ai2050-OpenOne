import { Line, Text } from '@react-three/drei';

import { useResearchSnapshot } from '../../researchKernel/useResearchSnapshot';

const CHAIN_COLORS = {
  passed: '#34d399',
  blocked: '#f59e0b',
  pending: '#64748b',
};

export function ResearchSpaceOverlay({
  layerVisibility,
  activeFileMeta,
  atlasNodes,
  atlasEdges,
  activeResearchPlugin,
  selectedEvidenceGate,
  onSelectEvidenceGate,
  mechanismMode = 'observe',
}) {
  const { snapshot } = useResearchSnapshot();
  const computationChain = (snapshot?.summaries?.evidence?.latest || []).slice(-7).map((item) => ({
    id: item.id,
    label: item.grade,
    gateId: item.id,
    status: item.polarity === 'positive' ? 'passed' : item.polarity === 'negative' ? 'blocked' : 'pending',
  }));

  const resultTypeLabel = activeFileMeta?.result_type || activeFileMeta?.type || activeResearchPlugin?.resultType || 'Result Type';
  const graphLabel = `${atlasNodes.length || 0} nodes / ${atlasEdges.length || 0} edges`;
  const routeLabel = activeResearchPlugin?.shortName || activeResearchPlugin?.name || 'Research Route';

  const visibleComputationChain = mechanismMode === 'present'
    ? computationChain.filter((stage) => stage.status === 'passed')
    : computationChain;

  const chainTitle = mechanismMode === 'compare'
    ? 'Validation / Compare'
    : mechanismMode === 'present'
      ? 'Observed Trace'
      : 'Evidence Trace';

  return (
    <group>
      <group position={[0, 50, -18]}>
        <Text fontSize={0.62} color="#e0f2fe" anchorX="center">
          {routeLabel}
        </Text>
        <Text position={[0, -0.62, 0]} fontSize={0.3} color="#93c5fd" anchorX="center">
          3D scene focuses on run-level test traces
        </Text>
      </group>

      {layerVisibility.features && (
        <group position={[-14, 7, 12]}>
          <mesh>
            <icosahedronGeometry args={[1.2, 1]} />
            <meshStandardMaterial color="#facc15" emissive="#eab308" emissiveIntensity={0.45} transparent opacity={0.78} />
          </mesh>
          <Text position={[0, 1.75, 0]} fontSize={0.42} color="#fef3c7" anchorX="center">
            Feature Layer
          </Text>
          <Text position={[0, 1.15, 0]} fontSize={0.26} color="#fde68a" anchorX="center">
            SAE / Dictionary / Feature Clusters
          </Text>
        </group>
      )}

      {layerVisibility.causalPath && (
        <group position={[13, 15, -8]}>
          <mesh rotation={[0, 0, Math.PI / 4]}>
            <boxGeometry args={[5.2, 0.08, 0.08]} />
            <meshStandardMaterial color="#22c55e" emissive="#16a34a" emissiveIntensity={0.42} transparent opacity={0.82} />
          </mesh>
          <Text position={[0, 0.9, 0]} fontSize={0.4} color="#bbf7d0" anchorX="center">
            Causal Path
          </Text>
          <Text position={[0, 0.35, 0]} fontSize={0.24} color="#86efac" anchorX="center">
            patch / ablation / restore
          </Text>
        </group>
      )}

      {layerVisibility.dynamics && (
        <group position={[15, 4, 13]}>
          <mesh>
            <torusKnotGeometry args={[1.1, 0.08, 80, 8]} />
            <meshStandardMaterial color="#fb7185" emissive="#e11d48" emissiveIntensity={0.38} transparent opacity={0.72} />
          </mesh>
          <Text position={[0, 1.65, 0]} fontSize={0.4} color="#ffe4e6" anchorX="center">
            Dynamics Layer
          </Text>
          <Text position={[0, 1.1, 0]} fontSize={0.24} color="#fda4af" anchorX="center">
            spike / replay / control state
          </Text>
        </group>
      )}

      {layerVisibility.atlas && atlasNodes.length > 0 && (
        <Text position={[0, 55, -12]} fontSize={0.55} color="#bfdbfe" anchorX="center">
          {resultTypeLabel} / {graphLabel}
        </Text>
      )}

      {layerVisibility.atlas && (
        <group position={[0, -7.2, -7]}>
          <Text position={[0, 1.65, 0]} fontSize={0.42} color="#bae6fd" anchorX="center">
            {chainTitle}
          </Text>
          {visibleComputationChain.length > 1 && (
            <Line
              points={visibleComputationChain.map((_, index) => [index * 4.8 - ((visibleComputationChain.length - 1) * 2.4), 0, 0])}
              color="#334155"
              lineWidth={1}
              transparent
              opacity={0.72}
            />
          )}
          {visibleComputationChain.map((stage, index) => {
            const x = index * 4.8 - ((visibleComputationChain.length - 1) * 2.4);
            const selected = selectedEvidenceGate === stage.gateId;
            const color = CHAIN_COLORS[stage.status] || CHAIN_COLORS.pending;
            return (
              <group
                key={stage.id}
                position={[x, 0, 0]}
                onClick={(event) => {
                  event.stopPropagation();
                  onSelectEvidenceGate?.(stage.gateId);
                }}
                onPointerOver={(event) => {
                  event.stopPropagation();
                  document.body.style.cursor = 'pointer';
                }}
                onPointerOut={() => {
                  document.body.style.cursor = 'default';
                }}
              >
                <mesh scale={selected ? 1.35 : 1}>
                  <sphereGeometry args={[0.28, 18, 18]} />
                  <meshStandardMaterial color={color} emissive={color} emissiveIntensity={selected ? 1.1 : 0.45} />
                </mesh>
                <Text position={[0, -0.68, 0]} fontSize={0.25} color={selected ? '#f8fafc' : '#94a3b8'} anchorX="center">
                  {stage.label}
                </Text>
              </group>
            );
          })}
        </group>
      )}
    </group>
  );
}
