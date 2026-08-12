import { Line, Text } from '@react-three/drei';

import { CURRENT_RESEARCH_STATE, RESEARCH_COMPUTATION_CHAIN } from '../../researchKernel/currentResearchState';

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
  const phaseLabel = activeFileMeta?.phase ? `Phase ${activeFileMeta.phase}` : 'Current phase';
  const graphLabel = `${atlasNodes.length || 0} nodes / ${atlasEdges.length || 0} edges`;
  const routeLabel = activeResearchPlugin?.shortName || activeResearchPlugin?.name || '研究路线';
  const visibleComputationChain = mechanismMode === 'present'
    ? RESEARCH_COMPUTATION_CHAIN.filter((stage) => stage.status === 'passed')
    : RESEARCH_COMPUTATION_CHAIN;
  const chainTitle = mechanismMode === 'compare'
    ? '原始 / 反事实证据链'
    : mechanismMode === 'present'
      ? '已通过成果链'
      : '当前机制观察链';

  return (
    <group>
      <group position={[0, 50, -18]}>
        <Text fontSize={0.62} color="#e0f2fe" anchorX="center">
          {routeLabel}
        </Text>
        <Text position={[0, -0.62, 0]} fontSize={0.3} color="#93c5fd" anchorX="center">
          插件化研究路线 · 共享3D主空间
        </Text>
      </group>

      {layerVisibility.features && (
        <group position={[-14, 7, 12]}>
          <mesh>
            <icosahedronGeometry args={[1.2, 1]} />
            <meshStandardMaterial color="#facc15" emissive="#eab308" emissiveIntensity={0.45} transparent opacity={0.78} />
          </mesh>
          <Text position={[0, 1.75, 0]} fontSize={0.42} color="#fef3c7" anchorX="center">
            特征空间层
          </Text>
          <Text position={[0, 1.15, 0]} fontSize={0.26} color="#fde68a" anchorX="center">
            SAE / dictionary / feature clusters
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
            因果路径层
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
            动力学层
          </Text>
          <Text position={[0, 1.1, 0]} fontSize={0.24} color="#fda4af" anchorX="center">
            spike / replay / control state
          </Text>
        </group>
      )}

      {layerVisibility.atlas && atlasNodes.length > 0 && (
        <Text position={[0, 55, -12]} fontSize={0.55} color="#bfdbfe" anchorX="center">
          {phaseLabel} · {graphLabel}
        </Text>
      )}

      {layerVisibility.atlas && (
        <group position={[0, -7.2, -7]}>
          <Text position={[0, 1.65, 0]} fontSize={0.42} color="#bae6fd" anchorX="center">
            {chainTitle} · Phase {CURRENT_RESEARCH_STATE.phase}
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
                onPointerOut={() => { document.body.style.cursor = 'default'; }}
              >
                <mesh scale={selected ? 1.35 : 1}>
                  <sphereGeometry args={[0.28, 18, 18]} />
                  <meshStandardMaterial
                    color={color}
                    emissive={color}
                    emissiveIntensity={selected ? 1.1 : 0.45}
                  />
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
