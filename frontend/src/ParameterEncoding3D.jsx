import { API_BASE } from './config.js';
import { Line, Text } from '@react-three/drei';
import { useEffect, useMemo, useState } from 'react';
import agi3dClientScene from './blueprint/data/agi_3d_client_scene_v1';


const TEXT_PROPS = {
  color: '#d9f1ff',
  anchorX: 'center',
  anchorY: 'middle',
};

const ROLE_COLORS = {
  shared: '#38bdf8',
  bias: '#f97316',
  amplification: '#ec4899',
  task: '#22c55e',
  compare: '#a78bfa',
  raw: '#e5f6ff',
  band: '#345b7a',
};

function LayerBands() {
  const bands = [
    { id: 'early', label: '早层', y: 1.6 },
    { id: 'middle', label: '中层', y: 4.6 },
    { id: 'late', label: '后层', y: 7.6 },
  ];

  return (
    <group>
      {bands.map((band) => (
        <group key={band.id}>
          <Line
            points={[
              [-18, band.y, -18],
              [18, band.y, -18],
              [18, band.y, 18],
              [-18, band.y, 18],
              [-18, band.y, -18],
            ]}
            color={ROLE_COLORS.band}
            lineWidth={1}
            transparent
            opacity={0.4}
          />
          <Text {...TEXT_PROPS} fontSize={0.36} color="#8fb8d3" position={[-17.2, band.y + 0.25, -17.2]}>
            {band.label}
          </Text>
        </group>
      ))}
    </group>
  );
}

function AxisHint() {
  return (
    <group>
      <Text {...TEXT_PROPS} fontSize={0.34} color="#8fd4ff" position={[0, 10.8, 0]}>
        {`X=${agi3dClientScene.axes.x}  ·  Y=${agi3dClientScene.axes.y}  ·  Z=${agi3dClientScene.axes.z}`}
      </Text>
    </group>
  );
}

function SmallNode({ position, color, size = 0.22, opacity = 0.95 }) {
  return (
    <mesh position={position} renderOrder={3}>
      <sphereGeometry args={[size, 14, 14]} />
      <meshBasicMaterial color={color} transparent opacity={opacity} depthWrite={false} />
    </mesh>
  );
}

function HaloNode({ position, color, size = 0.42, opacity = 0.2 }) {
  return (
    <mesh position={position} renderOrder={1}>
      <sphereGeometry args={[size, 18, 18]} />
      <meshBasicMaterial color={color} transparent opacity={opacity} depthWrite={false} />
    </mesh>
  );
}

function RingNode({ position, color, radius = 0.56, opacity = 0.65 }) {
  return (
    <mesh position={position} rotation={[Math.PI / 2, 0, 0]} renderOrder={2}>
      <torusGeometry args={[radius, 0.04, 10, 36]} />
      <meshBasicMaterial color={color} transparent opacity={opacity} depthWrite={false} />
    </mesh>
  );
}

function SlimPillar({ position, color, height = 1.6, width = 0.3, opacity = 0.84 }) {
  return (
    <mesh position={position} scale={[width, height, width]} renderOrder={2}>
      <boxGeometry args={[1, 1, 1]} />
      <meshBasicMaterial color={color} transparent opacity={opacity} depthWrite={false} />
    </mesh>
  );
}

function Caption({ position, text, color = '#dff7ff', size = 0.28 }) {
  return (
    <Text {...TEXT_PROPS} fontSize={size} color={color} position={position}>
      {text}
    </Text>
  );
}

function createSpread(center, count, xStep = 0.45, zStep = 0.45) {
  return Array.from({ length: count }, (_, index) => {
    const col = index % 4;
    const row = Math.floor(index / 4);
    return [center[0] + (col - 1.5) * xStep, center[1] - 0.35 - row * 0.18, center[2] + ((index % 2 === 0 ? 1 : -1) * zStep)];
  });
}

function createLabeledSpread(center, count, xRadius = 2.6, zRadius = 2.1, yOffset = 0.1) {
  return Array.from({ length: count }, (_, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(1, count);
    return [
      center[0] + Math.cos(angle) * xRadius,
      center[1] + yOffset + (index % 2 === 0 ? 0.42 : -0.12),
      center[2] + Math.sin(angle) * zRadius,
    ];
  });
}

function createParameterRack(center, count, xStep = 1.55, zOffset = -3.4, yOffset = 1.15) {
  const startX = center[0] - ((count - 1) * xStep) / 2;
  return Array.from({ length: count }, (_, index) => [
    startX + index * xStep,
    center[1] + yOffset,
    center[2] + zOffset,
  ]);
}

function NeuronDots({ center, count, color, size = 0.12 }) {
  const positions = useMemo(() => createSpread(center, count), [center, count]);
  return (
    <group>
      {positions.map((position, index) => (
        <group key={`${center.join('-')}-${index}`}>
          <HaloNode position={position} color={color} size={size * 2.2} opacity={0.16} />
          <SmallNode position={position} color={color} size={size} opacity={0.98} />
        </group>
      ))}
    </group>
  );
}

function ParameterNodes({ center, members, color, onHover, onSelect, layerLabel, sourceStage }) {
  const positions = useMemo(() => createParameterRack(center, members?.length || 0), [center, members]);
  if (!members?.length) return null;

  return (
    <group>
      {members.map((member, index) => {
        const position = positions[index];
        return (
          <InteractiveGroup
            key={`${center.join('-')}-member-${member}`}
            info={{
              type: 'encoding3d_parameter_node',
              label: `d${member}`,
              score: member,
              position,
              layerLabel,
              nodeKind: '参数位节点',
              role: '参数级成员',
              detailText: `共享承载成员维度 d${member}`,
              dimIndex: member,
              sourceStage,
            }}
            onHover={onHover}
            onSelect={onSelect}
          >
            <Line
              points={[center, position]}
              color={color}
              lineWidth={2.4}
              transparent
              opacity={0.72}
            />
            <mesh position={[position[0], position[1], position[2]]} renderOrder={2}>
              <boxGeometry args={[0.92, 0.92, 0.92]} />
              <meshBasicMaterial color={color} transparent opacity={0.95} depthWrite={false} />
            </mesh>
            <HaloNode position={position} color={color} size={1.18} opacity={0.26} />
            <Caption
              position={[position[0], position[1] + 0.92, position[2]]}
              text={`d${member}`}
              color={color}
              size={0.34}
            />
            <Caption
              position={[position[0], position[1] - 0.86, position[2]]}
              text="参数位"
              color="#ffffff"
              size={0.18}
            />
          </InteractiveGroup>
        );
      })}
    </group>
  );
}

function InteractiveGroup({ children, info, onHover, onSelect }) {
  return (
    <group
      onPointerOver={(event) => {
        event.stopPropagation();
        onHover?.(info);
        document.body.style.cursor = 'pointer';
      }}
      onPointerOut={() => {
        onHover?.(null);
        document.body.style.cursor = 'default';
      }}
      onClick={(event) => {
        event.stopPropagation();
        onSelect?.(info);
      }}
    >
      {children}
    </group>
  );
}

function RawPointCloud({ points, onHover, onSelect }) {
  if (!points?.length) return null;
  return (
    <group>
      {points.map((point) => (
        <InteractiveGroup
          key={point.id}
          info={{
            type: 'encoding3d_raw_point',
            label: point.label,
            score: point.value,
            position: point.position,
            nodeKind: point.kind || '原始数据点',
            role: point.kind || '原始数据点',
            detailText: `${point.label}：${point.value}`,
          }}
          onHover={onHover}
          onSelect={onSelect}
        >
          <group>
            <HaloNode position={point.position} color={point.color || ROLE_COLORS.raw} size={0.46} opacity={0.2} />
            <SmallNode position={point.position} color={point.color || ROLE_COLORS.raw} size={0.22} opacity={0.98} />
            <Caption
              position={[point.position[0], point.position[1] + 0.34, point.position[2]]}
              text={point.label}
              color={point.color || ROLE_COLORS.raw}
              size={0.14}
            />
          </group>
        </InteractiveGroup>
      ))}
    </group>
  );
}

function RawLinks({ links }) {
  if (!links?.length) return null;
  return (
    <group>
      {links.map((link) => (
        <Line key={link.id} points={[link.from, link.to]} color={link.color || '#9ca3af'} lineWidth={1.2} transparent opacity={0.34} />
      ))}
    </group>
  );
}

function RuntimeProfiles({ profiles, layerLabel, onHover, onSelect }) {
  if (!profiles?.length) return null;
  return (
    <group>
      {profiles.map((profile) => (
        <group key={profile.id}>
          {profile.nodes.map((node, index) => (
            <InteractiveGroup
              key={node.id}
              info={{
                type: 'encoding3d_runtime_node',
                label: `${profile.label} / ${node.label}`,
                score: node.layerIndex,
                position: node.position,
                layerLabel,
                nodeKind: '真实运行节点',
                role: '运行层级',
                detailText: `${node.label}，层号 L${node.layerIndex}`,
              }}
              onHover={onHover}
              onSelect={onSelect}
            >
              <HaloNode position={node.position} color={profile.color} size={0.5} opacity={0.18} />
              <SmallNode position={node.position} color={profile.color} size={0.22} opacity={0.96} />
              <Caption position={[node.position[0], node.position[1] + 0.5, node.position[2]]} text={`L${node.layerIndex}`} color={profile.color} size={0.2} />
            </InteractiveGroup>
          ))}
          {profile.nodes.slice(0, -1).map((node, index) => (
            <Line
              key={`${profile.id}-line-${node.id}`}
              points={[node.position, profile.nodes[index + 1].position]}
              color={profile.color}
              lineWidth={1.4}
              transparent
              opacity={0.6}
            />
          ))}
          <Caption position={[profile.nodes[0].position[0], profile.nodes[0].position[1] - 0.6, profile.nodes[0].position[2]]} text={profile.label} color={profile.color} size={0.18} />
        </group>
      ))}
    </group>
  );
}

function SharedCarrierLayer({ layer, onHover, onSelect }) {
  return (
    <group>
      {layer.clusters.map((cluster) => (
        <InteractiveGroup
          key={cluster.id}
          info={{
            type: 'encoding3d_cluster',
            label: cluster.label,
            score: cluster.stability,
            memberCount: cluster.memberCount,
            members: cluster.members,
            position: cluster.position,
            layerLabel: layer.label,
            nodeKind: '共享承载簇',
            role: '共享承载',
            detailText: `稳定 ${cluster.stability.toFixed(3)}，成员 ${cluster.members.join(', ')}`,
            sourceStage: layer.sourceStage,
            sourceScript: layer.sourceScript,
            sourceOutput: layer.sourceOutput,
          }}
          onHover={onHover}
          onSelect={onSelect}
        >
          <mesh position={[cluster.position[0], cluster.position[1] + 0.2, cluster.position[2] - 0.2]} renderOrder={1}>
            <planeGeometry args={[6.8, 2.3]} />
            <meshBasicMaterial color={cluster.color} transparent opacity={0.08} depthWrite={false} side={2} />
          </mesh>
          <RingNode position={cluster.position} color={cluster.color} radius={cluster.size} opacity={0.18} />
          <NeuronDots center={cluster.position} count={cluster.memberCount} color={cluster.color} size={0.12} />
          <ParameterNodes
            center={cluster.position}
            members={cluster.members}
            color={cluster.color}
            onHover={onHover}
            onSelect={onSelect}
            layerLabel={layer.label}
            sourceStage={layer.sourceStage}
          />
          <Caption position={[cluster.position[0], cluster.position[1] + 0.85, cluster.position[2]]} text={cluster.label} />
        </InteractiveGroup>
      ))}

      {layer.taskBridges.map((bridge) => (
        <InteractiveGroup
          key={bridge.id}
          info={{
            type: 'encoding3d_bridge',
            label: bridge.label,
            score: bridge.strength,
            from: bridge.from,
            to: bridge.to,
            layerLabel: layer.label,
            nodeKind: '共享桥',
            role: '共享承载',
            detailText: `桥接强度 ${bridge.strength.toFixed(3)}`,
            sourceStage: layer.sourceStage,
          }}
          onHover={onHover}
          onSelect={onSelect}
        >
          <Line points={[bridge.from, bridge.to]} color={bridge.color} lineWidth={1.6 + bridge.strength * 1.2} transparent opacity={0.75} />
          <SmallNode position={bridge.to} color={bridge.color} size={0.18} opacity={0.9} />
        </InteractiveGroup>
      ))}

      <RawPointCloud points={layer.rawPoints} onHover={onHover} onSelect={onSelect} />
      <RawLinks links={layer.rawLinks} />
      <RuntimeProfiles profiles={layer.runtimeProfiles} layerLabel={layer.label} onHover={onHover} onSelect={onSelect} />
    </group>
  );
}

function BiasDeflectionLayer({ layer, onHover, onSelect }) {
  return (
    <group>
      {layer.directions.map((direction) => (
        <InteractiveGroup
          key={direction.id}
          info={{
            type: 'encoding3d_direction',
            label: direction.label,
            score: direction.selectivity,
            memberCount: direction.memberCount,
            from: direction.from,
            to: direction.to,
            layerLabel: layer.label,
            nodeKind: '偏转方向',
            role: '偏置偏转',
            detailText: `选择性 ${direction.selectivity.toFixed(3)}，成员数 ${direction.memberCount}`,
            sourceStage: layer.sourceStage,
          }}
          onHover={onHover}
          onSelect={onSelect}
        >
          <Line points={[direction.from, direction.to]} color={direction.color} lineWidth={2.2 + direction.selectivity * 1.2} transparent opacity={0.9} />
          <RingNode position={direction.to} color={direction.color} radius={0.4} opacity={0.32} />
          <NeuronDots center={direction.to} count={direction.memberCount} color={direction.color} size={0.18} />
          <Caption position={[direction.to[0], direction.to[1] + 0.8, direction.to[2]]} text={direction.label} />
        </InteractiveGroup>
      ))}

      {layer.modelBias.map((bias) => (
        <InteractiveGroup
          key={bias.id}
          info={{
            type: 'encoding3d_model_bias',
            label: bias.label,
            score: bias.thickness,
            position: bias.position,
            layerLabel: layer.label,
            nodeKind: '模型偏转柱',
            role: '偏置偏转',
            detailText: `厚度 ${bias.thickness.toFixed(3)}`,
            sourceStage: layer.sourceStage,
          }}
          onHover={onHover}
          onSelect={onSelect}
        >
          <SlimPillar position={bias.position} color={bias.color} height={1.0 + bias.thickness * 1.4} />
          <Caption position={[bias.position[0], bias.position[1] + 1.1, bias.position[2]]} text={bias.label} size={0.24} />
        </InteractiveGroup>
      ))}

      <RawPointCloud points={layer.rawPoints} onHover={onHover} onSelect={onSelect} />
      <RawLinks links={layer.rawLinks} />
      <RuntimeProfiles profiles={layer.runtimeProfiles} layerLabel={layer.label} onHover={onHover} onSelect={onSelect} />
    </group>
  );
}

function AmplificationLayer({ layer, onHover, onSelect }) {
  return (
    <group>
      {layer.bands.map((band) => {
        const start = band.position;
        const end = [band.position[0] + band.length, band.position[1], band.position[2]];
        return (
          <InteractiveGroup
            key={band.id}
            info={{
              type: 'encoding3d_band',
              label: band.label,
              score: band.strength,
              gain: band.gain,
              position: band.position,
              layerLabel: layer.label,
              nodeKind: '放大路径带',
              role: '逐层放大',
              detailText: `强度 ${band.strength.toFixed(3)}，增益 ${band.gain.toFixed(3)}`,
              sourceStage: layer.sourceStage,
            }}
            onHover={onHover}
            onSelect={onSelect}
          >
            <Line points={[start, end]} color={band.color} lineWidth={3.2 + band.gain * 2.4} transparent opacity={0.88} />
            <SmallNode position={start} color={band.color} size={0.18} opacity={0.75} />
            <SmallNode position={end} color={band.color} size={0.22} opacity={0.9} />
            <Caption position={[start[0] + band.length / 2, start[1] + 0.72, start[2]]} text={band.label} size={0.24} />
          </InteractiveGroup>
        );
      })}

      <RawPointCloud points={layer.rawPoints} onHover={onHover} onSelect={onSelect} />
      <RawLinks links={layer.rawLinks} />
      <RuntimeProfiles profiles={layer.runtimeProfiles} layerLabel={layer.label} onHover={onHover} onSelect={onSelect} />
    </group>
  );
}

function MultispaceOperatorLayer({ layer, onHover, onSelect }) {
  const links = useMemo(
    () => [
      [layer.roleNodes[0].position, layer.operatorParts[0].position],
      [layer.roleNodes[1].position, layer.operatorParts[1].position],
      [layer.roleNodes[1].position, layer.operatorParts[2].position],
      [layer.roleNodes[2].position, layer.operatorParts[3].position],
    ],
    [layer]
  );

  return (
    <group>
      {layer.roleNodes.map((node) => (
        <InteractiveGroup
          key={node.id}
          info={{
            type: 'encoding3d_role_node',
            label: node.label,
            score: node.alignment,
            position: node.position,
            layerLabel: layer.label,
            nodeKind: '空间角色节点',
            role: '多空间角色',
            detailText: `对齐 ${node.alignment.toFixed(3)}`,
            sourceStage: layer.sourceStage,
          }}
          onHover={onHover}
          onSelect={onSelect}
        >
          <RingNode position={node.position} color={node.color} radius={0.42} opacity={0.32} />
          <NeuronDots center={node.position} count={4} color={node.color} size={0.18} />
          <Caption position={[node.position[0], node.position[1] + 0.72, node.position[2]]} text={node.label} size={0.24} />
        </InteractiveGroup>
      ))}

      {layer.operatorParts.map((part) => (
        <InteractiveGroup
          key={part.id}
          info={{
            type: 'encoding3d_operator_part',
            label: part.label,
            score: part.value,
            position: part.position,
            layerLabel: layer.label,
            nodeKind: '局部运算元部件',
            role: '多空间角色',
            detailText: `值 ${part.value.toFixed(3)}`,
            sourceStage: layer.sourceStage,
          }}
          onHover={onHover}
          onSelect={onSelect}
        >
          <SlimPillar position={part.position} color={part.color} height={0.8 + part.value * 1.1} width={0.24} opacity={0.8} />
          <Caption position={[part.position[0], part.position[1] + 0.95, part.position[2]]} text={part.label} size={0.22} />
        </InteractiveGroup>
      ))}

      {links.map((points, index) => (
        <Line key={`operator-link-${index}`} points={points} color="#7dd3fc" lineWidth={1.1} transparent opacity={0.42} />
      ))}

      <RawPointCloud points={layer.rawPoints} onHover={onHover} onSelect={onSelect} />
      <RawLinks links={layer.rawLinks} />
      <RuntimeProfiles profiles={layer.runtimeProfiles} layerLabel={layer.label} onHover={onHover} onSelect={onSelect} />
    </group>
  );
}

function CrossModelCompareLayer({ layer, onHover, onSelect }) {
  return (
    <group>
      {layer.models.map((model) => (
        <group key={model.id}>
          <Caption position={[model.position[0], 9.2, model.position[2]]} text={model.label} color={model.color} size={0.34} />
          {model.metrics.map((metric, index) => {
            const x = model.position[0] + index * 2.1 - 3.2;
            const height = 0.8 + metric.value * 2.6;
            const y = height / 2 + 0.8;
            return (
              <InteractiveGroup
                key={`${model.id}-${metric.key}`}
                info={{
                  type: 'encoding3d_model_metric',
                  label: `${model.label} / ${metric.key}`,
                  score: metric.value,
                  model: model.label,
                  metric: metric.key,
                  position: [x, y, model.position[2]],
                  layerLabel: layer.label,
                  nodeKind: '跨模型指标柱',
                  role: '跨模型对照',
                  detailText: `${metric.key} = ${metric.value.toFixed(3)}`,
                  sourceStage: layer.sourceStage,
                }}
                onHover={onHover}
                onSelect={onSelect}
              >
                <SlimPillar position={[x, y, model.position[2]]} color={model.color} height={height} width={0.22} opacity={0.78} />
                <Caption position={[x, y + 0.78, model.position[2]]} text={metric.key} size={0.18} />
              </InteractiveGroup>
            );
          })}
        </group>
      ))}

      <RawPointCloud points={layer.rawPoints} onHover={onHover} onSelect={onSelect} />
      <RawLinks links={layer.rawLinks} />
      <RuntimeProfiles profiles={layer.runtimeProfiles} layerLabel={layer.label} onHover={onHover} onSelect={onSelect} />
    </group>
  );
}

function LiveRuntimeFlow({ mode, onHover, onSelect }) {
  const [flow, setFlow] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    async function loadFlow() {
      try {
        setError('');
        const url = `${API_BASE}/api/runtime/neuron_flow?mode=${encodeURIComponent(mode)}&top_k=8`;
        const response = await fetch(url, { signal: controller.signal });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        if (!cancelled) {
          setFlow(payload.ok ? payload : null);
        }
      } catch (err) {
        if (cancelled || err?.name === 'AbortError') return;
        setFlow(null);
        setError(String(err.message || err));
      }
    }

    loadFlow();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [mode]);

  if (error) {
    return (
      <Caption position={[0, 10.2, 6]} text={`实时流错误: ${error}`} color="#ff8a80" size={0.22} />
    );
  }

  if (!flow?.nodes?.length) {
    return (
      <Caption position={[0, 10.2, 6]} text="实时前向流暂无节点" color="#ffd38a" size={0.22} />
    );
  }

  return (
    <group>
      {flow.links?.map((link) => (
        <Line
          key={link.id}
          points={[link.from, link.to]}
          color="#ffffff"
          lineWidth={1.6}
          transparent
          opacity={0.28}
        />
      ))}
      {flow.nodes.map((node) => (
        <InteractiveGroup
          key={node.id}
          info={{
            type: 'runtime_neuron_node',
            label: node.label,
            score: node.activation_abs,
            position: node.position,
            layerLabel: `L${node.layer_index}`,
            nodeKind: '实时前向神经元',
            role: '实时运行',
            detailText: `${node.token} / d${node.dim_index} / ${node.activation_value.toFixed(4)}`,
            tokenIndex: node.token_index,
            dimIndex: node.dim_index,
            hookName: node.hook_name,
          }}
          onHover={onHover}
          onSelect={onSelect}
        >
          <HaloNode position={node.position} color="#ffffff" size={0.62} opacity={0.12} />
          <SmallNode position={node.position} color="#ffffff" size={0.18} opacity={0.96} />
        </InteractiveGroup>
      ))}
      <Caption
        position={[0, 11.4, 6]}
        text={`实时前向流 · 词元 ${flow.target_token} · Top-${flow.top_k}`}
        color="#ffffff"
        size={0.22}
      />
    </group>
  );
}

const MODE_COMPONENTS = {
  shared_carrier_3d: SharedCarrierLayer,
  bias_deflection_3d: BiasDeflectionLayer,
  layerwise_amplification_3d: AmplificationLayer,
  multispace_operator_3d: MultispaceOperatorLayer,
  cross_model_compare_3d: CrossModelCompareLayer,
};

export default function ParameterEncoding3D({ mode, onHover, onSelect }) {
  const layer = agi3dClientScene.layers[mode];
  const LayerComponent = MODE_COMPONENTS[mode];

  if (!layer || !LayerComponent) return null;

  return (
    <group>
      <LayerBands />
      <AxisHint />
      <Text {...TEXT_PROPS} fontSize={0.76} color="#4ecdc4" position={[0, 11.9, 0]}>
        {layer.label}
      </Text>
      <Text {...TEXT_PROPS} fontSize={0.28} color="#a7dfff" position={[0, 11.1, 0]}>
        {`分数 ${layer.score.toFixed(3)} · 运行机制优先视图`}
      </Text>
      <LayerComponent layer={layer} onHover={onHover} onSelect={onSelect} />
      <LiveRuntimeFlow mode={mode} onHover={onHover} onSelect={onSelect} />
    </group>
  );
}
