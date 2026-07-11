export const PATTERN_ATLAS_EVIDENCE_COLORS = {
  candidate: '#fbbf24',
  natural: '#22d3ee',
  group: '#fb923c',
  confirmed: '#4ade80',
};

export function patternAtlasEvidenceCategory(node) {
  if (node?.expanded_confirmation_pass) return 'confirmed';
  if (node?.group_intervention_supported) return 'group';
  if (node?.natural_observed) return 'natural';
  return 'candidate';
}

export function patternAtlasNodeColor(node) {
  return PATTERN_ATLAS_EVIDENCE_COLORS[patternAtlasEvidenceCategory(node)];
}

export function patternAtlasEvidenceLabel(node) {
  const category = patternAtlasEvidenceCategory(node);
  return {
    candidate: '候选',
    natural: '自然交叉',
    group: '组级支持',
    confirmed: '扩大确认',
  }[category];
}

export function patternAtlasUnitAddressLabel(node) {
  const prefix = node?.unit_kind === 'attention_head'
    ? 'H'
    : node?.unit_kind === 'mlp_product_neuron'
      ? 'N'
      : node?.unit_kind === 'mlp_product_group'
        ? 'G'
        : 'U';
  return `${prefix}#${node?.unit_index ?? '?'}`;
}

export function filterPatternAtlasNodes(nodes, focus) {
  if (focus === 'natural') return nodes.filter((node) => node.natural_observed);
  if (focus === 'group') return nodes.filter((node) => node.group_intervention_supported);
  if (focus === 'confirmed') return nodes.filter((node) => node.expanded_confirmation_pass);
  return nodes;
}

export function sortPatternAtlasNodes(nodes) {
  return nodes.slice().sort((a, b) => Number(b.display_priority || 0) - Number(a.display_priority || 0));
}

export function patternAtlasPhysicalKey(node) {
  return [node?.model, node?.layer, node?.component, node?.unit_kind, node?.unit_index].join(':');
}

export function dedupePatternAtlasUnits(nodes) {
  const byAddress = new Map();
  sortPatternAtlasNodes(nodes).forEach((node) => {
    const key = patternAtlasPhysicalKey(node);
    const existing = byAddress.get(key);
    if (!existing) {
      byAddress.set(key, {
        ...node,
        atlas_membership_count: 1,
        atlas_mechanism_ids: node.mechanism_id ? [node.mechanism_id] : [],
      });
      return;
    }
    existing.atlas_membership_count += 1;
    existing.natural_observed = existing.natural_observed || node.natural_observed;
    existing.group_intervention_supported = existing.group_intervention_supported || node.group_intervention_supported;
    existing.expanded_confirmation_pass = existing.expanded_confirmation_pass || node.expanded_confirmation_pass;
    if (node.mechanism_id && !existing.atlas_mechanism_ids.includes(node.mechanism_id)) {
      existing.atlas_mechanism_ids.push(node.mechanism_id);
    }
  });
  return Array.from(byAddress.values());
}

export function selectBalancedPatternAtlasNodes(nodes, focus, limit) {
  const eligible = sortPatternAtlasNodes(filterPatternAtlasNodes(nodes, focus));
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

export function patternAtlasNodeToInfo(node) {
  const isComponentSet = node.node_type === 'component_set_member';
  return {
    token: `L${node.layer} · ${node.unit_kind} #${node.unit_index}`,
    label: node.family_name,
    type: isComponentSet ? '模式族物理组件集合成员' : '模式族物理单元候选',
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
    expanded_confirmation_pass: node.expanded_confirmation_pass,
    source: node.source_artifacts?.[0],
    source_artifacts: node.source_artifacts,
    node_id: node.node_id,
    is_real_unit: !isComponentSet,
    is_component_set_member: isComponentSet,
    single_unit_causal: false,
    atlas_membership_count: node.atlas_membership_count || 1,
    atlas_mechanism_ids: node.atlas_mechanism_ids || (node.mechanism_id ? [node.mechanism_id] : []),
  };
}
