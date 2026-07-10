import { Line, Text } from '@react-three/drei';
import { useLayoutEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';

const COLORS = {
  source: '#94a3b8',
  path: '#38bdf8',
  candidate: '#fbbf24',
  natural: '#22d3ee',
  group: '#fb923c',
  readout: '#fb7185',
  active: '#f8fafc',
};

function filterNodes(nodes, focus) {
  if (focus === 'natural') return nodes.filter((node) => node.natural_observed);
  if (focus === 'group') return nodes.filter((node) => node.group_intervention_supported);
  return nodes;
}

function selectBalancedNodes(nodes, focus, limit) {
  const eligible = filterNodes(nodes, focus)
    .slice()
    .sort((a, b) => Number(b.display_priority || 0) - Number(a.display_priority || 0));
  const grouped = new Map();
  eligible.forEach((node) => {
    const layer = Number(node.layer || 0);
    if (!grouped.has(layer)) grouped.set(layer, []);
    grouped.get(layer).push(node);
  });
  const groups = Array.from(grouped.values());
  if (!groups.length) return [];
  const selected = [];
  const selectedIds = new Set();
  const quota = Math.max(1, Math.floor(limit / groups.length));
  groups.forEach((group) => {
    group.slice(0, quota).forEach((node) => {
      selected.push(node);
      selectedIds.add(node.node_id);
    });
  });
  eligible.forEach((node) => {
    if (selected.length >= limit) return;
    if (!selectedIds.has(node.node_id)) {
      selected.push(node);
      selectedIds.add(node.node_id);
    }
  });
  return selected;
}

function layerY(layer, layerCount) {
  if (layerCount <= 1) return 0;
  return -9.6 + (Number(layer) / (layerCount - 1)) * 19.2;
}

function nodePosition(node, snapshot, rankInLayer) {
  const index = Number(node.unit_index || 0);
  const angle = ((index * 0.618033988749895) % 1) * Math.PI * 2;
  const radius = 3 + (index % 3) * 0.5;
  return [
    Math.cos(angle) * radius,
    layerY(node.layer, Number(snapshot?.num_hidden_layers || 1)) + ((rankInLayer % 5) - 2) * 0.07,
    Math.sin(angle) * radius,
  ];
}

function nodeColor(node) {
  if (node.group_intervention_supported) return COLORS.group;
  if (node.natural_observed) return COLORS.natural;
  return COLORS.candidate;
}

function nodeCategory(node) {
  if (node.group_intervention_supported) return 'group';
  if (node.natural_observed) return 'natural';
  return 'candidate';
}

function toHoverInfo(node) {
  return {
    token: `L${node.layer} · ${node.unit_kind} #${node.unit_index}`,
    label: node.family_name,
    type: '模式族物理单元候选',
    family_id: node.family_id,
    family_name: node.family_name,
    relation: node.relation,
    model: node.model,
    model_revision: node.model_revision,
    layer: node.layer,
    component: node.component,
    unit_kind: node.unit_kind,
    unit_index: node.unit_index,
    activation: node.natural_activation,
    score: node.candidate_score,
    case_count: node.case_count,
    target_labels: node.target_labels,
    evidence_level: node.evidence_level,
    evidence_status: node.evidence_status,
    evidence_boundary: node.evidence_boundary,
    causal_scope: node.causal_scope,
    natural_observed: node.natural_observed,
    group_intervention_supported: node.group_intervention_supported,
    source: node.source_artifacts?.[0],
    source_artifacts: node.source_artifacts,
    node_id: node.node_id,
    is_real_unit: true,
  };
}

function UnitInstances({ items, positions, colorValue, selectedNodeId, onHover, onSelect }) {
  const ref = useRef(null);
  const matrix = useMemo(() => new THREE.Matrix4(), []);

  useLayoutEffect(() => {
    if (!ref.current) return;
    items.forEach((node, index) => {
      const priority = Math.max(0, Number(node.display_priority || 0));
      const scale = 0.82 + Math.min(0.62, priority * 0.72);
      matrix.compose(
        new THREE.Vector3(...positions[index]),
        new THREE.Quaternion(),
        new THREE.Vector3(scale, scale, scale)
      );
      ref.current.setMatrixAt(index, matrix);
    });
    ref.current.instanceMatrix.needsUpdate = true;
  }, [items, matrix, positions]);

  const selectedIndex = items.findIndex((node) => node.node_id === selectedNodeId);
  const selectedPosition = selectedIndex >= 0 ? positions[selectedIndex] : null;

  return (
    <group>
      <instancedMesh
        ref={ref}
        args={[null, null, items.length]}
        onPointerMove={(event) => {
          event.stopPropagation();
          const node = items[event.instanceId];
          if (node) onHover?.(toHoverInfo(node));
        }}
        onPointerOut={() => onHover?.(null)}
        onClick={(event) => {
          event.stopPropagation();
          const node = items[event.instanceId];
          if (node) onSelect?.(toHoverInfo(node));
        }}
      >
        <sphereGeometry args={[0.21, 14, 14]} />
        <meshBasicMaterial color={colorValue} toneMapped={false} fog={false} />
      </instancedMesh>
      {selectedPosition && (
        <mesh position={selectedPosition} scale={1.65} renderOrder={95}>
          <sphereGeometry args={[0.21, 14, 14]} />
          <meshBasicMaterial color={COLORS.active} wireframe transparent opacity={0.92} depthTest={false} />
        </mesh>
      )}
      {items.map((node, index) => (
        <mesh
          key={`hit-${node.node_id}`}
          position={positions[index]}
          renderOrder={110}
          onPointerMove={(event) => {
            event.stopPropagation();
            onHover?.(toHoverInfo(node));
          }}
          onPointerOut={() => onHover?.(null)}
          onClick={(event) => {
            event.stopPropagation();
            onSelect?.(toHoverInfo(node));
          }}
        >
          <sphereGeometry args={[0.34, 10, 10]} />
          <meshBasicMaterial transparent opacity={0.001} depthWrite={false} toneMapped={false} fog={false} />
        </mesh>
      ))}
    </group>
  );
}

function Anchor({ position, color, label, detail, active = false, onHover, onSelect }) {
  return (
    <group position={position}>
      <mesh
        onPointerOver={(event) => { event.stopPropagation(); onHover?.(); }}
        onPointerOut={() => onHover?.(null)}
        onClick={(event) => { event.stopPropagation(); onSelect?.(); }}
      >
        <cylinderGeometry args={[active ? 0.34 : 0.26, active ? 0.34 : 0.26, 0.18, 16]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={active ? 0.62 : 0.3} />
      </mesh>
      <Text position={[-0.55, 0.06, 0]} fontSize={0.24} color={color} anchorX="right">
        {label}
      </Text>
      {detail && (
        <Text position={[-0.55, -0.25, 0]} fontSize={0.14} color="#7f95bb" anchorX="right">
          {detail}
        </Text>
      )}
    </group>
  );
}

function EmptyFamilyScene({ family, model }) {
  return (
    <group>
      <mesh>
        <torusGeometry args={[1.5, 0.035, 10, 64]} />
        <meshBasicMaterial color="#475569" transparent opacity={0.62} />
      </mesh>
      <Text position={[0, 0.1, 0]} fontSize={0.48} color="#cbd5e1" anchorX="center">
        {family?.family_name || '模式族'}
      </Text>
      <Text position={[0, -0.65, 0]} fontSize={0.24} color="#fb7185" anchorX="center">
        {`${model} · 真实物理单元尚未映射`}
      </Text>
    </group>
  );
}

export default function PatternFamilyNeuronAtlasRenderer({
  atlas,
  evidenceFocus = 'key',
  maxUnits = 48,
  currentLayer = null,
  selectedNodeId = '',
  onHover,
  onSelect,
}) {
  const partition = atlas?.partition;
  const snapshot = partition?.model_snapshot;
  const layerCount = Number(snapshot?.num_hidden_layers || 1);
  const selectedNodes = useMemo(
    () => selectBalancedNodes(partition?.nodes || [], evidenceFocus, maxUnits),
    [evidenceFocus, maxUnits, partition?.nodes]
  );
  const rankByLayer = new Map();
  const positions = selectedNodes.map((node) => {
    const layer = Number(node.layer || 0);
    const rank = rankByLayer.get(layer) || 0;
    rankByLayer.set(layer, rank + 1);
    return nodePosition(node, snapshot, rank);
  });
  const positionById = new Map(selectedNodes.map((node, index) => [node.node_id, positions[index]]));
  const instanceGroups = ['candidate', 'natural', 'group'].map((category) => {
    const items = selectedNodes.filter((node) => nodeCategory(node) === category);
    return {
      category,
      items,
      positions: items.map((node) => positionById.get(node.node_id)),
      color: COLORS[category],
    };
  }).filter((group) => group.items.length > 0);
  const sourceY = -11.7;
  const readoutY = 11.7;
  const anchors = partition?.path?.layer_anchors || [];
  const spinePoints = [
    [0, sourceY, 0],
    ...anchors.map((anchor) => [0, layerY(anchor.layer, layerCount), 0]),
    [0, readoutY, 0],
  ];

  if (!partition) return <EmptyFamilyScene family={atlas?.family} model={atlas?.model || ''} />;

  const familyName = partition.family?.family_name || partition.family?.family_id;
  const readout = partition.path?.readout;
  const source = partition.path?.source;

  return (
    <group position={[0, 0, 0]}>
      <Text position={[0, 13.8, 0]} fontSize={0.54} color="#e0f2fe" anchorX="center">
        {familyName}
      </Text>
      <Text position={[0, 13.2, 0]} fontSize={0.24} color="#7dd3fc" anchorX="center">
        {`${partition.model} · 真实证据关键脉络 · ${selectedNodes.length}/${partition.metrics.unique_unit_count} 单元`}
      </Text>

      <Line points={spinePoints} color={COLORS.path} lineWidth={2} transparent opacity={0.5} dashed dashSize={0.3} gapSize={0.18} />

      <Anchor
        position={[0, sourceY, 0]}
        color={COLORS.source}
        label="来源状态"
        detail={`token ${source?.token_position ?? '-'}`}
        onHover={(value) => onHover?.(value === null ? null : {
          token: 'Prompt embedding', label: familyName, type: '自然运行来源事件', family_name: familyName,
          model: partition.model, evidence_level: 'L2', causal_scope: 'observed_not_causal',
          evidence_boundary: '真实运行来源状态；不是因果来源证明', source: partition.source_artifacts?.[1],
        })}
      />

      {anchors.map((anchor) => {
        const y = layerY(anchor.layer, layerCount);
        const active = Number(currentLayer) === Number(anchor.layer);
        const info = {
          token: `L${anchor.layer} · 组件路径锚点`,
          label: familyName,
          type: '观测组件顺序与候选层',
          family_id: partition.family.family_id,
          family_name: familyName,
          model: partition.model,
          layer: anchor.layer,
          evidence_level: anchor.evidence_level,
          causal_scope: 'observed_sequence_not_causal',
          evidence_boundary: anchor.evidence_boundary,
          candidate_count: anchor.candidate_count,
          natural_overlap_count: anchor.natural_overlap_count,
          group_supported_count: anchor.group_supported_count,
          source: partition.source_artifacts?.[1],
        };
        return (
          <Anchor
            key={anchor.anchor_id}
            position={[0, y, 0]}
            color={active ? COLORS.active : COLORS.path}
            label={`L${anchor.layer}`}
            detail={`${anchor.candidate_count} 候选`}
            active={active}
            onHover={(value) => onHover?.(value === null ? null : info)}
            onSelect={() => onSelect?.(info)}
          />
        );
      })}

      <Anchor
        position={[0, readoutY, 0]}
        color={COLORS.readout}
        label="读出结果"
        detail={readout?.global_closed ? '本样本命中' : '本样本未闭合'}
        onHover={(value) => onHover?.(value === null ? null : {
          token: 'Unembedding readout', label: familyName, type: '自然运行读出事件', family_name: familyName, model: partition.model,
          layer: readout?.layer, evidence_level: 'L2', causal_scope: 'observed_not_causal',
          evidence_boundary: '单个自然运行读出；不是跨样本机制闭合', global_closed: readout?.global_closed,
          readout_metrics: readout?.metrics, source: partition.source_artifacts?.[1],
        })}
      />

      {selectedNodes.map((node) => {
        const target = positionById.get(node.node_id);
        const anchor = [0, layerY(node.layer, layerCount), 0];
        return (
          <Line
            key={`edge-${node.node_id}`}
            points={[anchor, target]}
            color={nodeColor(node)}
            lineWidth={node.natural_observed || node.group_intervention_supported ? 1.2 : 0.7}
            transparent
            opacity={node.natural_observed || node.group_intervention_supported ? 0.48 : 0.2}
          />
        );
      })}

      {instanceGroups.map((group) => (
        <UnitInstances
          key={group.category}
          items={group.items}
          positions={group.positions}
          colorValue={group.color}
          selectedNodeId={selectedNodeId}
          onHover={onHover}
          onSelect={onSelect}
        />
      ))}

      <group position={[6.3, -10.8, 0]}>
        {[
          [COLORS.candidate, 'L4 候选'],
          [COLORS.natural, 'L2 自然交叉'],
          [COLORS.group, '组级支持，非单元因果'],
        ].map(([color, label], index) => (
          <group key={label} position={[0, index * 0.48, 0]}>
            <mesh><sphereGeometry args={[0.1, 10, 10]} /><meshBasicMaterial color={color} /></mesh>
            <Text position={[0.25, 0, 0]} fontSize={0.16} color="#a9bdd8" anchorX="left">{label}</Text>
          </group>
        ))}
      </group>
    </group>
  );
}
