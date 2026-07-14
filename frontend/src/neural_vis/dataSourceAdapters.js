const COMPONENT_X = {
  embedding: -8,
  residual: -5,
  norm: -2.5,
  attention: 1,
  mlp: 4.5,
  unembedding: 8,
};

const COMPONENT_COLORS = {
  embedding: '#94a3b8',
  residual: '#38bdf8',
  norm: '#a78bfa',
  attention: '#f97316',
  mlp: '#facc15',
  unembedding: '#22c55e',
};

const FACTOR_COLORS = {
  O: '#ec4899',
  R: '#38bdf8',
  A: '#22c55e',
  C: '#a78bfa',
  F: '#14b8a6',
  M: '#f59e0b',
  K: '#84cc16',
  S_answer: '#10b981',
  B: '#ef4444',
  G: '#fb7185',
  N: '#60a5fa',
  P: '#f97316',
  T: '#eab308',
};

function asNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function stableHash(value) {
  let hash = 2166136261;
  for (const character of String(value || '')) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function basename(filepath) {
  return String(filepath || '').split('/').filter(Boolean).pop() || 'dataset.json';
}

export function joinSourcePath(basePath, filepath) {
  const path = String(filepath || '');
  if (/^(?:https?:)?\/\//.test(path) || path.startsWith('/')) return path;
  const base = String(basePath || '').replace(/\/$/, '');
  return `${base}/${path.replace(/^\//, '')}`;
}

function makeEntry(source, item, index, rawPath, fallbackLabel) {
  const path = joinSourcePath(source.data_base_path, rawPath);
  return {
    ...item,
    id: item.id || item.run_id || `${source.id}:${index}`,
    filename: item.filename || basename(path),
    path,
    label: item.label || fallbackLabel || item.title || basename(path),
    source_id: source.id,
    route_id: source.route_id,
    route_label: source.route_label,
    payload_adapter: source.payload_adapter,
  };
}

export function normalizeManifestEntries(source, manifest) {
  const adapter = source.manifest_adapter;

  if (adapter === 'files') {
    return (Array.isArray(manifest.files) ? manifest.files : []).map((item, index) => {
      const record = typeof item === 'string' ? { filename: item } : item;
      return makeEntry(source, record, index, record.path || record.filename, record.label);
    });
  }

  if (adapter === 'items') {
    return (Array.isArray(manifest.items) ? manifest.items : []).map((item, index) => (
      makeEntry(source, item, index, item.path || item.filename, item.label)
    ));
  }

  if (adapter === 'partitions') {
    return (Array.isArray(manifest.partitions) ? manifest.partitions : []).map((item, index) => {
      const familyLabel = item.family_name || item.family_id || '模式族';
      const modelLabel = item.model || 'mixed';
      return makeEntry(source, item, index, item.path, `${familyLabel} · ${modelLabel}`);
    });
  }

  throw new Error(`Unsupported manifest adapter: ${adapter}`);
}

function sourceContext(source, fileMeta, sourceSchema) {
  return {
    source_id: source?.id || fileMeta?.source_id || 'local',
    source_label: source?.label || fileMeta?.source_label || '本地文件',
    route_id: source?.route_id || fileMeta?.route_id || 'local',
    route_label: source?.route_label || fileMeta?.route_label || '本地文件',
    evidence_scope: source?.evidence_scope || null,
    source_schema_version: sourceSchema,
    dataset_id: fileMeta?.id || fileMeta?.filename || null,
    dataset_label: fileMeta?.label || fileMeta?.filename || null,
  };
}

function canonicalGraph(data, fileMeta, source, graph, extras = {}) {
  return {
    ...extras,
    schema_version: 'atlas_graph_v1',
    source_schema_version: data.schema_version,
    adapted_for_3d: data.schema_version !== 'atlas_graph_v1',
    title: extras.title || data.title || fileMeta?.label || '3D 数据图谱',
    model: extras.model || data.model || fileMeta?.model || null,
    phase: extras.phase ?? data.phase ?? fileMeta?.phase ?? null,
    evidence_boundary: data.evidence_boundary || source?.evidence_scope || null,
    source_context: sourceContext(source, fileMeta, data.schema_version),
    graph,
  };
}

function neuronNodeType(node) {
  if (String(node.node_type || '').includes('anchor')) return 'cluster';
  if (node.unit_kind === 'attention_head') return 'head';
  if (String(node.unit_kind || '').includes('neuron')) return 'channel';
  return node.component === 'mlp' ? 'channel' : 'intervention';
}

function adaptNeuronAtlas(data, fileMeta, source) {
  const familyId = data.family?.family_id || fileMeta?.family_id || 'pattern_family';
  const familyName = data.family?.family_name || familyId;
  const model = data.model || fileMeta?.model || 'unknown';
  const nodes = (Array.isArray(data.nodes) ? data.nodes : []).map((node, index) => {
    const id = node.node_id || node.id || `${familyId}:${model}:node:${index}`;
    const layer = asNumber(node.layer, 0);
    const unit = asNumber(node.unit_index, index);
    const componentOffset = node.component === 'attention' ? -3.5 : node.component === 'mlp' ? 2.5 : 0;
    const relationLane = (stableHash(node.relation || node.node_type) % 9) - 4;
    const score = asNumber(node.display_priority, asNumber(node.candidate_score, 0));
    return {
      ...node,
      id,
      label: node.label || `${familyName} · L${layer} ${node.unit_kind || node.node_type} ${node.unit_index ?? ''}`.trim(),
      type: neuronNodeType(node),
      model,
      layer,
      channel: node.unit_index,
      score,
      size: Math.min(0.42, 0.12 + Math.log1p(Math.abs(score)) * 0.07),
      evidence_level: node.evidence_status || node.evidence_level,
      color: node.natural_observed ? '#22c55e' : node.group_intervention_supported ? '#facc15' : '#60a5fa',
      position: [
        componentOffset + ((unit % 29) - 14) * 0.28,
        layer * 1.72,
        relationLane * 0.72 + ((Math.floor(unit / 29) % 5) - 2) * 0.22,
      ],
      show_label: String(node.node_type || '').includes('anchor'),
    };
  });

  const nodeIds = new Set(nodes.map((node) => node.id));
  const edges = (Array.isArray(data.membership_edges) ? data.membership_edges : []).map((edge, index) => ({
    ...edge,
    id: edge.edge_id || `${familyId}:${model}:edge:${index}`,
    source: edge.source_id || edge.source,
    target: edge.target_id || edge.target,
    relation: edge.relation || 'contains_localized_candidate',
  }));

  for (const edge of edges) {
    for (const endpoint of [edge.source, edge.target]) {
      if (!endpoint || nodeIds.has(endpoint)) continue;
      const layerMatch = String(endpoint).match(/:L(-?\d+):/);
      const layer = asNumber(layerMatch?.[1], 0);
      nodes.push({
        id: endpoint,
        label: `${familyName} · L${layer} 锚点`,
        type: 'cluster',
        model,
        layer,
        color: '#a78bfa',
        size: 0.28,
        position: [0, layer * 1.72, -4.5],
        evidence_level: 'structural_anchor',
        show_label: true,
      });
      nodeIds.add(endpoint);
    }
  }

  return canonicalGraph(data, fileMeta, source, {
    title: `${familyName} · ${model}`,
    nodes,
    edges,
  }, {
    title: `${familyName} · ${model}`,
    model,
    phase: data.scope?.phase,
    model_info: { model, phase: data.scope?.phase, n_layers: Math.max(1, ...nodes.map((node) => asNumber(node.layer, 0))) + 1 },
    family: data.family,
    mapping_status: data.mapping_status,
    metrics: data.metrics,
  });
}

function traceNodeType(component) {
  if (component === 'attention') return 'head';
  if (component === 'mlp') return 'channel';
  if (component === 'residual') return 'layer';
  if (component === 'embedding' || component === 'unembedding') return 'task';
  return 'cluster';
}

function adaptRealComponentTrace(data, fileMeta, source) {
  const events = Array.isArray(data.events) ? data.events : [];
  const model = data.model || fileMeta?.model || 'unknown';
  const nodes = events.map((event, index) => {
    const layer = asNumber(event.layer, -1);
    const component = event.component || event.event_type || 'event';
    const eventIndex = asNumber(event.event_index, index);
    const norm = asNumber(event.norm, 0);
    return {
      ...event,
      id: `${data.run_id || fileMeta?.id || model}:event:${eventIndex}`,
      label: `L${layer} ${component} · ${event.event_type || 'event'}`,
      type: traceNodeType(component),
      model,
      layer,
      score: norm,
      size: Math.min(0.3, 0.11 + Math.log1p(Math.abs(norm)) * 0.035),
      color: COMPONENT_COLORS[component] || '#60a5fa',
      position: [
        COMPONENT_X[component] ?? 0,
        (layer + 1) * 1.72,
        ((stableHash(event.event_type) % 9) - 4) * 0.58 + (eventIndex % 3) * 0.12,
      ],
      evidence_level: 'natural_observation',
      causal: false,
      show_label: eventIndex % 36 === 0,
    };
  });

  const series = new Map();
  nodes.forEach((node) => {
    const key = `${node.component}:${node.event_type}`;
    if (!series.has(key)) series.set(key, []);
    series.get(key).push(node);
  });
  const edges = [];
  series.forEach((seriesNodes, key) => {
    seriesNodes.sort((left, right) => left.event_index - right.event_index);
    for (let index = 1; index < seriesNodes.length; index += 1) {
      edges.push({
        id: `${key}:${index}`,
        source: seriesNodes[index - 1].id,
        target: seriesNodes[index].id,
        relation: 'measured_after',
        causal: false,
        evidence_boundary: 'measurement order only; not a causal edge',
      });
    }
  });

  const maxLayer = Math.max(0, ...events.map((event) => asNumber(event.layer, 0)));
  return canonicalGraph(data, fileMeta, source, {
    title: `${model} · 真实组件轨迹`,
    nodes,
    edges,
  }, {
    title: `${model} · 真实组件轨迹`,
    model,
    model_info: { model, phase: data.phase, n_layers: maxLayer + 1 },
    run_id: data.run_id,
    prompt: data.prompt,
    summary: data.summary,
  });
}

function factorNodeType(factorId) {
  if (['O', 'R', 'A', 'C'].includes(factorId)) return 'concept';
  if (['M', 'G', 'N'].includes(factorId)) return 'intervention';
  if (['P', 'T', 'S_answer'].includes(factorId)) return 'task';
  return 'cluster';
}

function adaptMechanismTrace(data, fileMeta, source) {
  const layers = Array.isArray(data.layers) ? data.layers : [];
  const model = data.model || fileMeta?.model || 'unknown';
  const factorIds = Array.from(new Set(layers.flatMap((layer) => Object.keys(layer.factors || {}))));
  const nodes = [];
  const byFactor = new Map(factorIds.map((factorId) => [factorId, []]));

  layers.forEach((layerRecord) => {
    const layer = asNumber(layerRecord.layer, -1);
    factorIds.forEach((factorId, factorIndex) => {
      const factor = layerRecord.factors?.[factorId];
      if (!factor) return;
      const id = `${fileMeta?.id || model}:L${layer}:factor:${factorId}`;
      const factorValue = asNumber(factor.value, 0);
      const node = {
        ...factor,
        id,
        factor_id: factorId,
        label: `L${layer} ${factor['中文'] || factor.label || factorId}`,
        type: factorNodeType(factorId),
        model,
        layer,
        score: factorValue,
        size: Math.min(0.34, 0.11 + Math.sqrt(Math.abs(factorValue)) * 0.12),
        color: FACTOR_COLORS[factorId] || '#60a5fa',
        position: [
          (factorIndex - (factorIds.length - 1) / 2) * 1.25,
          (layer + 1) * 1.72,
          0,
        ],
        evidence_level: 'observed_factor',
        causal: false,
        show_label: layer === -1 || layer === layers.at(-1)?.layer,
      };
      nodes.push(node);
      byFactor.get(factorId).push(node);
    });
  });

  const edges = [];
  byFactor.forEach((factorNodes, factorId) => {
    factorNodes.sort((left, right) => left.layer - right.layer);
    for (let index = 1; index < factorNodes.length; index += 1) {
      edges.push({
        id: `${fileMeta?.id || model}:${factorId}:${index}`,
        source: factorNodes[index - 1].id,
        target: factorNodes[index].id,
        relation: 'observed_continuity',
        causal: false,
        evidence_boundary: 'same measured factor across adjacent layers; not a causal edge',
      });
    }
  });

  const maxLayer = Math.max(0, ...layers.map((layer) => asNumber(layer.layer, 0)));
  return canonicalGraph(data, fileMeta, source, {
    title: `${model} · 机制因子轨迹`,
    nodes,
    edges,
  }, {
    title: `${model} · 机制因子轨迹`,
    model,
    model_info: { model, phase: data.phase, n_layers: maxLayer + 1 },
    prompt: data.prompt,
    target: data.target,
    summary: data.summary,
    notes: data.notes,
  });
}

export function normalizeVisualizationPayload(data, fileMeta = {}, source = null) {
  const schema = data?.schema_version || '1.0';
  if (schema === 'atlas_graph_v1' || Array.isArray(data?.visualizations)) {
    return {
      ...data,
      source_schema_version: schema,
      adapted_for_3d: false,
      source_context: sourceContext(source, fileMeta, schema),
    };
  }
  if (schema === 'neuron_atlas_partition.v1') return adaptNeuronAtlas(data, fileMeta, source);
  if (schema === 'real_component_trace.v1') return adaptRealComponentTrace(data, fileMeta, source);
  if (schema === 'mechanism_trace_v1') return adaptMechanismTrace(data, fileMeta, source);
  throw new Error(`Unsupported schema: ${schema}`);
}
