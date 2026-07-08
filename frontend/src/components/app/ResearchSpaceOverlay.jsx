import { Text } from '@react-three/drei';

export function ResearchSpaceOverlay({ layerVisibility, activeFileMeta, atlasNodes, atlasEdges, activeResearchPlugin }) {
  const phaseLabel = activeFileMeta?.phase ? `Phase ${activeFileMeta.phase}` : 'Current phase';
  const graphLabel = `${atlasNodes.length || 0} nodes / ${atlasEdges.length || 0} edges`;
  const routeLabel = activeResearchPlugin?.shortName || activeResearchPlugin?.name || '研究路线';

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

      {layerVisibility.boundary && (
        <group position={[0, 3.5, -17]}>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[8.5, 0.04, 12, 128]} />
            <meshStandardMaterial color="#ef4444" emissive="#dc2626" emissiveIntensity={0.38} transparent opacity={0.55} />
          </mesh>
          <Text position={[0, 1.2, 0]} fontSize={0.42} color="#fecaca" anchorX="center">
            失败边界层
          </Text>
          <Text position={[0, 0.65, 0]} fontSize={0.28} color="#fca5a5" anchorX="center">
            weak / null / boundary evidence
          </Text>
        </group>
      )}

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
    </group>
  );
}
