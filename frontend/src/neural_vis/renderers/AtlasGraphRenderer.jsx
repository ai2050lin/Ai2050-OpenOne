/**
 * AtlasGraphRenderer — 机制图谱测试结果3D渲染器
 *
 * 输入 Schema: atlas_graph_v1
 * 坐标含义:
 *   x: head/channel/intervention offset
 *   y: layer
 *   z: model/concept lane
 */
import React, { useMemo } from 'react';
import { Line, Text } from '@react-three/drei';

const NODE_COLORS = {
  model: '#94a3b8',
  phase: '#60a5fa',
  layer: '#38bdf8',
  head: '#f97316',
  channel: '#facc15',
  cluster: '#a855f7',
  intervention: '#22c55e',
  concept: '#ec4899',
  task: '#14b8a6',
  failure: '#ef4444',
};

const EDGE_COLORS = {
  contains: '#475569',
  tested_by: '#60a5fa',
  supports_likelihood: '#facc15',
  changes_generation: '#22c55e',
  weak_generation_effect: '#94a3b8',
  negative_effect: '#ef4444',
  shared_by: '#a855f7',
  differs_from: '#fb7185',
  upstream_of: '#38bdf8',
  washed_by: '#64748b',
  candidate_of: '#f97316',
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function asNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function nodeSize(node) {
  const strength = Math.max(
    Math.abs(asNumber(node.score, 0)),
    Math.abs(asNumber(node.mean_logprob_delta, 0)),
    Math.abs(asNumber(node.generation_effect, 0))
  );
  return clamp(0.24 + strength * 0.28, 0.22, 0.95);
}

function edgeWidth(edge) {
  return clamp(1 + Math.abs(asNumber(edge.weight, 0)) * 2.4, 1, 5);
}

function buildPosition(node, index, modelLaneMap) {
  if (Array.isArray(node.position) && node.position.length === 3) {
    return node.position.map((v) => asNumber(v, 0));
  }

  const typeOffset = {
    model: -8,
    phase: -6,
    task: -4,
    concept: -2,
    intervention: 0,
    head: 2,
    cluster: 4,
    channel: 6,
    layer: 8,
    failure: 10,
  }[node.type] ?? 0;

  const layer = node.layer !== undefined ? asNumber(node.layer, 0) : asNumber(node.y, 0);
  const modelLane = modelLaneMap.get(node.model || node.id?.split(':')?.[0] || 'default') ?? 0;
  const head = node.head !== undefined ? asNumber(node.head, 0) : 0;
  const channel = node.channel !== undefined ? asNumber(node.channel, 0) : 0;
  const jitter = ((index % 7) - 3) * 0.28;

  const x = typeOffset + head * 0.24 + channel * 0.045 + jitter;
  const y = layer * 1.75;
  const z = modelLane * 8 + ((index % 3) - 1) * 0.45;

  return [x, y, z];
}

function makeLabel(node) {
  if (node.label) return node.label;
  if (node.layer !== undefined && node.head !== undefined && node.channel !== undefined) {
    return `L${node.layer}H${node.head}C${node.channel}`;
  }
  if (node.layer !== undefined && node.head !== undefined) {
    return `L${node.layer}H${node.head}`;
  }
  return node.id || node.type || 'node';
}

export default function AtlasGraphRenderer({ graph, onHoverNode }) {
  const { nodes, edges, positions, modelLanes } = useMemo(() => {
    const rawNodes = Array.isArray(graph?.nodes) ? graph.nodes : [];
    const rawEdges = Array.isArray(graph?.edges) ? graph.edges : graph?.links || [];
    const models = Array.from(new Set(rawNodes.map((node) => node.model || node.id?.split(':')?.[0] || 'default')));
    const laneMap = new Map(models.map((model, index) => [model, index - (models.length - 1) / 2]));
    const posMap = new Map();

    rawNodes.forEach((node, index) => {
      posMap.set(node.id, buildPosition(node, index, laneMap));
    });

    return {
      nodes: rawNodes,
      edges: rawEdges,
      positions: posMap,
      modelLanes: models.map((model) => ({ model, z: (laneMap.get(model) ?? 0) * 8 })),
    };
  }, [graph]);

  if (!nodes.length) return null;

  return (
    <group position={[0, -8, 0]}>
      <Text position={[0, 66, -10]} fontSize={1.1} color="#e2e8f0" anchorX="center">
        Mechanism Atlas Graph
      </Text>

      {modelLanes.map((lane) => (
        <group key={lane.model}>
          <Line
            points={[[-14, 0, lane.z], [14, 0, lane.z], [14, 62, lane.z], [-14, 62, lane.z], [-14, 0, lane.z]]}
            color="#1e293b"
            lineWidth={1}
            transparent
            opacity={0.5}
          />
          <Text position={[-15, 1, lane.z]} fontSize={0.45} color="#94a3b8" anchorX="right">
            {lane.model}
          </Text>
        </group>
      ))}

      {edges.map((edge, index) => {
        const source = positions.get(edge.source);
        const target = positions.get(edge.target);
        if (!source || !target) return null;
        const relation = edge.relation || edge.type || 'contains';
        const color = EDGE_COLORS[relation] || '#64748b';
        return (
          <Line
            key={`${edge.source}-${edge.target}-${index}`}
            points={[source, target]}
            color={color}
            lineWidth={edgeWidth(edge)}
            transparent
            opacity={relation === 'contains' ? 0.22 : 0.68}
          />
        );
      })}

      {nodes.map((node, index) => {
        const position = positions.get(node.id) || [0, 0, 0];
        const color = node.color || NODE_COLORS[node.type] || '#60a5fa';
        const size = nodeSize(node);
        const evidenceLevel = node.evidence_level || node.evidence || '';
        const opacity = evidenceLevel === 'candidate' ? 0.48 : evidenceLevel === 'likelihood_only' ? 0.72 : 0.9;
        const label = makeLabel(node);

        return (
          <group key={node.id || index} position={position}>
            <mesh
              onPointerOver={(event) => {
                event.stopPropagation();
                onHoverNode?.({
                  token: label,
                  ...node,
                });
              }}
              onPointerOut={() => onHoverNode?.(null)}
            >
              <sphereGeometry args={[size, 18, 18]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={node.type === 'failure' ? 0.6 : 0.35}
                transparent
                opacity={opacity}
                roughness={0.4}
                metalness={0.25}
              />
            </mesh>
            {(node.type === 'model' || node.type === 'head' || node.type === 'cluster' || node.type === 'failure') && (
              <Text position={[0, size + 0.35, 0]} fontSize={0.35} color={color} anchorX="center">
                {label}
              </Text>
            )}
          </group>
        );
      })}
    </group>
  );
}
