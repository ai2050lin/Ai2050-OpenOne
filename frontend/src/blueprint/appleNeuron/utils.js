/**
 * AppleNeuron3D 工具函数
 * 从 AppleNeuron3DTab.jsx 拆分而来
 */

import {
  LAYER_COUNT, DFF, IMPORTED_QUERY_NODE_MAX, QUERY_NODE_COUNT,
  HARD_PROBLEM_EXPERIMENT_LABELS,
  ROLE_COLORS, DIMENSION_LABELS,
  APPLE_SWITCH_MECHANISM_SCHEMA, APPLE_SWITCH_MODEL_COLORS, APPLE_SWITCH_ROLE_LABELS,
  TOKEN_TRANSITIONS, TOPIC_FALLBACKS, DEFAULT_CHAIN_TOKENS, PREDICT_CHAIN_LENGTH,
  CONCEPT_ASSOCIATION_LAYER_META, CONCEPT_ALIAS_MAP,
  APPLE_ANIMATION_OPTIONS,
  DEFAULT_LANGUAGE_FOCUS,
} from './constants';

// ---- 3D position helpers ----

function neuronToPosition(layer, neuron, radialJitter = 0) {
  const angle = ((neuron % 4096) / 4096) * Math.PI * 2;
  const radius = 2.7 + ((neuron % 2048) / 2048) * 3.3 + radialJitter;
  const z = (layer - (LAYER_COUNT - 1) / 2) * 0.92;
  const x = Math.cos(angle) * radius;
  const y = Math.sin(angle) * radius;
  return [x, y, z];
}

// ---- Core math helpers ----

function averagePosition(nodes, fallback = [0, 0, 0]) {
  if (!Array.isArray(nodes) || nodes.length === 0) {
    return fallback;
  }
  const sum = nodes.reduce((acc, node) => {
    const pos = Array.isArray(node?.position) ? node.position : fallback;
    return [
      acc[0] + toSafeNumber(pos[0], 0),
      acc[1] + toSafeNumber(pos[1], 0),
      acc[2] + toSafeNumber(pos[2], 0),
    ];
  }, [0, 0, 0]);
  return sum.map((value) => value / nodes.length);
}

function blendPosition(a, b, t) {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

function shiftPosition(a, dx = 0, dy = 0, dz = 0) {
  return [a[0] + dx, a[1] + dy, a[2] + dz];
}

function normalizeVector(vec, scale = 1) {
  const norm = Math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2) || 1;
  return vec.map((value) => (value / norm) * scale);
}

// ---- Safe number & formatting ----

function toSafeNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function normalizeConceptKey(value) {
  return String(value || '').trim().toLowerCase();
}

function nodeSignalStrength(node) {
  return toSafeNumber(node?.strength, 0) + toSafeNumber(node?.value, 0) * 0.35;
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number.isFinite(Number(value)) ? Number(value) : 0));
}

function metricNodeStrength(metricKey, value) {
  const v = Number(value);
  if (!Number.isFinite(v)) {
    return 0.2;
  }
  if (metricKey.includes('error') || metricKey.includes('collision') || metricKey.includes('decay')) {
    return clamp01(1 - v);
  }
  return clamp01(v);
}

function extractMetricScalar(value) {
  if (typeof value === 'number') {
    return value;
  }
  if (value && typeof value === 'object' && typeof value.mean === 'number') {
    return Number(value.mean);
  }
  return NaN;
}

function getMetricByPath(metrics, path) {
  if (!metrics || !path) {
    return undefined;
  }
  if (!String(path).includes('.')) {
    return metrics[path];
  }
  return String(path)
    .split('.')
    .reduce((acc, key) => (acc && typeof acc === 'object' ? acc[key] : undefined), metrics);
}

// ---- Preview & formatting ----

function formatPreviewValue(value) {
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      return '-';
    }
    const abs = Math.abs(value);
    if (abs >= 1000) {
      return value.toFixed(0);
    }
    if (abs >= 1) {
      return value.toFixed(4).replace(/\.?0+$/, '');
    }
    if (abs === 0) {
      return '0';
    }
    return value.toExponential(3);
  }
  if (typeof value === 'boolean') {
    return value ? 'true' : 'false';
  }
  if (value === null || value === undefined || value === '') {
    return '-';
  }
  return String(value);
}

function safeJsonStringify(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch (_err) {
    return String(value ?? '');
  }
}

// ---- Random & chain generation ----

function pseudoRandom(seed) {
  const v = Math.sin(seed * 12.9898) * 43758.5453;
  return v - Math.floor(v);
}

function hashString(value) {
  let h = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    h ^= value.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function extractPromptTokens(prompt) {
  return (prompt || '')
    .toLowerCase()
    .replace(/[，。！？,.!?;:]/g, ' ')
    .split(/\s+/)
    .filter(Boolean);
}

function getFallbackTokens(prompt) {
  const normalized = (prompt || '').toLowerCase();
  const topic = TOPIC_FALLBACKS.find((item) => item.keywords.some((k) => normalized.includes(k.toLowerCase())));
  return topic?.tokens || DEFAULT_CHAIN_TOKENS;
}

function generatePredictChain(prompt) {
  const tokens = extractPromptTokens(prompt);
  const fallback = getFallbackTokens(prompt);
  let context = tokens[tokens.length - 1] || fallback[0];
  const chain = [];
  for (let i = 0; i < PREDICT_CHAIN_LENGTH; i += 1) {
    const candidates = TOKEN_TRANSITIONS[context] || fallback;
    const pickSeed = hashString(`${prompt}|${context}|${i}`);
    const idx = Math.floor(pseudoRandom(pickSeed + i * 17) * candidates.length);
    const token = candidates[idx] || fallback[i % fallback.length];
    const base = Math.exp(-i * 0.18);
    const jitter = 0.04 * pseudoRandom(pickSeed + 101);
    const prob = Math.max(0.06, Math.min(0.96, 0.68 * base + 0.18 + jitter));
    chain.push({ token, prob });
    context = token;
  }
  return chain;
}

// ---- Concept set builders ----

function buildConceptNeuronSet(name, category = '未分类', idx = 0) {
  const normalized = name.trim().toLowerCase();
  const normalizedCategory = category.trim().toLowerCase() || '未分类';
  const baseHash = hashString(`${normalized}-${normalizedCategory}-${idx}`);
  const setId = `query-${normalized.replace(/[^a-z0-9\u4e00-\u9fa5]+/gi, '-')}-${normalizedCategory.replace(/[^a-z0-9\u4e00-\u9fa5]+/gi, '-')}-${baseHash}`;
  const color = `hsl(${baseHash % 360}, 82%, 62%)`;

  const nodes = Array.from({ length: QUERY_NODE_COUNT }, (_, i) => {
    const seed = baseHash + i * 10007;
    const baseLayer = Math.floor((i / QUERY_NODE_COUNT) * LAYER_COUNT);
    const layer = (baseLayer + Math.floor(pseudoRandom(seed + 3) * 5)) % LAYER_COUNT;
    const neuron = Math.floor(pseudoRandom(seed + 17) * DFF);
    const score = 0.35 + pseudoRandom(seed + 29) * 0.65;

    return {
      id: `${setId}-${i}`,
      label: `${name} Query ${i + 1}`,
      role: 'query',
      concept: name,
      category,
      layer,
      neuron,
      metric: 'query_score',
      value: score,
      strength: score,
      source: 'textbox-query-generator',
      color,
      position: neuronToPosition(layer, neuron, 0.18 + i * 0.025),
      size: 0.13 + score * 0.12,
      phase: i * 0.31,
    };
  });

  return {
    id: setId,
    name,
    category,
    normalized,
    normalizedCategory,
    color,
    nodes,
  };
}

// ---- Family patch & association ----

function buildFamilyPatchViewModel(nodes = [], selected = null, scanMechanismData = null) {
  const coreNodes = Array.isArray(nodes) ? nodes.filter((node) => node?.role !== 'background') : [];
  const queryNodes = coreNodes.filter((node) => node?.role === 'query');
  const selectedConceptKey = normalizeConceptKey(selected?.concept || selected?.label);
  const selectedCategoryKey = normalizeConceptKey(selected?.category);

  const conceptNodes = selectedConceptKey
    ? queryNodes.filter((node) => normalizeConceptKey(node?.concept || node?.label) === selectedConceptKey)
    : [];
  const familyNodes = selectedCategoryKey
    ? queryNodes.filter((node) => normalizeConceptKey(node?.category) === selectedCategoryKey)
    : conceptNodes;
  const siblingNodes = familyNodes.filter((node) => normalizeConceptKey(node?.concept || node?.label) !== selectedConceptKey);
  const uniqueSiblingConcepts = Array.from(new Set(siblingNodes.map((node) => String(node?.concept || '').trim()).filter(Boolean)));

  const familyCenter = averagePosition(familyNodes, Array.isArray(selected?.position) ? selected.position : [0, 0, 0]);
  const conceptCenter = averagePosition(conceptNodes, Array.isArray(selected?.position) ? selected.position : familyCenter);
  const siblingCenter = averagePosition(siblingNodes, familyCenter);
  const prototypeWitness = familyNodes
    .slice()
    .sort((a, b) => nodeSignalStrength(b) - nodeSignalStrength(a))
    .slice(0, 6);
  const instanceWitness = conceptNodes
    .slice()
    .sort((a, b) => nodeSignalStrength(b) - nodeSignalStrength(a))
    .slice(0, 6);
  const selectedConceptMinimal = selectedConceptKey ? scanMechanismData?.minimalByNoun?.[selectedConceptKey] || null : null;
  const selectedConceptCounterfactuals = selectedConceptKey ? scanMechanismData?.counterfactualByNoun?.[selectedConceptKey] || [] : [];
  const offsetVector = [
    conceptCenter[0] - familyCenter[0],
    conceptCenter[1] - familyCenter[1],
    conceptCenter[2] - familyCenter[2],
  ];
  const offsetNorm = Math.sqrt(offsetVector[0] ** 2 + offsetVector[1] ** 2 + offsetVector[2] ** 2);

  return {
    selectedConceptKey,
    selectedCategoryKey,
    familyNodes,
    conceptNodes,
    siblingNodes,
    uniqueSiblingConcepts,
    familyCenter,
    conceptCenter,
    siblingCenter,
    prototypeWitness,
    instanceWitness,
    selectedConceptMinimal,
    selectedConceptCounterfactuals,
    offsetNorm,
  };
}

function buildConceptAliasSet(selected = null) {
  const queue = [
    selected?.concept,
    selected?.fruit,
    selected?.category,
    selected?.label,
    selected?.id,
  ];
  const aliases = new Set();
  queue.forEach((value) => {
    const normalized = normalizeConceptKey(value);
    if (!normalized) {
      return;
    }
    aliases.add(normalized);
    (CONCEPT_ALIAS_MAP[normalized] || []).forEach((alias) => {
      const normalizedAlias = normalizeConceptKey(alias);
      if (normalizedAlias) {
        aliases.add(normalizedAlias);
      }
    });
  });
  return aliases;
}

function getAssociationNodeTexts(node = null) {
  if (!node || typeof node !== 'object') {
    return [];
  }
  return [
    node.concept,
    node.fruit,
    node.category,
    node.label,
    node.id,
    node.metric,
    node.source,
  ]
    .map((value) => normalizeConceptKey(value))
    .filter(Boolean);
}

function isTextMatchedByAliases(text = '', aliases = new Set()) {
  const normalized = normalizeConceptKey(text);
  if (!normalized || !aliases?.size) {
    return false;
  }
  if (aliases.has(normalized)) {
    return true;
  }
  return Array.from(aliases).some((alias) => (
    alias
    && normalized.length >= 2
    && (normalized.includes(alias) || alias.includes(normalized))
  ));
}

function isNodeMatchedByAliases(node = null, conceptAliases = new Set(), categoryAliases = new Set()) {
  const texts = getAssociationNodeTexts(node);
  const conceptMatched = texts.some((text) => isTextMatchedByAliases(text, conceptAliases));
  const categoryMatched = texts.some((text) => isTextMatchedByAliases(text, categoryAliases));
  return {
    conceptMatched,
    categoryMatched,
    matched: conceptMatched || categoryMatched,
  };
}

// ---- Association & emphasis ----

function distanceBetweenPositions(a = [0, 0, 0], b = [0, 0, 0]) {
  return Math.sqrt(
    (toSafeNumber(a?.[0], 0) - toSafeNumber(b?.[0], 0)) ** 2
    + (toSafeNumber(a?.[1], 0) - toSafeNumber(b?.[1], 0)) ** 2
    + (toSafeNumber(a?.[2], 0) - toSafeNumber(b?.[2], 0)) ** 2
  );
}

function pickConceptAssociationNodes(
  nodes = [],
  {
    roles = [],
    conceptAliases = new Set(),
    categoryAliases = new Set(),
    referencePosition = [0, 0, 0],
    layerHint = null,
    limit = 6,
  } = {}
) {
  const roleSet = new Set(roles);
  const scored = (Array.isArray(nodes) ? nodes : [])
    .filter((node) => node && node.role !== 'background')
    .map((node) => {
      const roleRank = roleSet.has(node.role) ? roles.indexOf(node.role) : roles.length + 4;
      const matchMeta = isNodeMatchedByAliases(node, conceptAliases, categoryAliases);
      const distance = distanceBetweenPositions(node.position, referencePosition);
      const layerDistance = Number.isFinite(layerHint) ? Math.abs(toSafeNumber(node.layer, 0) - layerHint) : 0;
      const signal = nodeSignalStrength(node);
      const score = (
        (matchMeta.conceptMatched ? 3.2 : 0)
        + (matchMeta.categoryMatched ? 1.2 : 0)
        + (roleSet.has(node.role) ? 1.8 - roleRank * 0.18 : 0)
        + Math.max(0, 1.25 - distance * 0.18)
        + Math.max(0, 0.55 - layerDistance * 0.05)
        + Math.min(1.2, signal * 1.1)
      );
      return {
        node,
        roleRank,
        conceptMatched: matchMeta.conceptMatched,
        categoryMatched: matchMeta.categoryMatched,
        distance,
        layerDistance,
        signal,
        score,
      };
    })
    .sort((left, right) => {
      if (right.score !== left.score) {
        return right.score - left.score;
      }
      if (left.roleRank !== right.roleRank) {
        return left.roleRank - right.roleRank;
      }
      if (left.layerDistance !== right.layerDistance) {
        return left.layerDistance - right.layerDistance;
      }
      return left.distance - right.distance;
    });

  const exactMatches = scored.filter((item) => item.conceptMatched);
  const categoryMatches = scored.filter((item) => !item.conceptMatched && item.categoryMatched);
  const roleMatches = scored.filter((item) => !item.conceptMatched && !item.categoryMatched && roleSet.has(item.node.role));
  const fallbackMatches = scored.filter((item) => !item.conceptMatched && !item.categoryMatched && !roleSet.has(item.node.role));

  return [...exactMatches, ...categoryMatches, ...roleMatches, ...fallbackMatches]
    .slice(0, Math.max(1, limit))
    .map((item) => item.node);
}

function buildConceptAssociationState(nodes = [], links = [], selected = null, languageFocus = DEFAULT_LANGUAGE_FOCUS, scanMechanismData = null) {
  if (!selected || selected.role === 'background') {
    return null;
  }

  const coreNodes = Array.isArray(nodes) ? nodes.filter((node) => node?.role !== 'background') : [];
  if (!coreNodes.length) {
    return null;
  }

  const selectedPosition = Array.isArray(selected?.position) ? selected.position : [0, 0, 0];
  const familyView = buildFamilyPatchViewModel(coreNodes, selected, scanMechanismData);
  const conceptAliases = buildConceptAliasSet(selected);
  const categoryAliases = buildConceptAliasSet({ concept: selected?.category || selected?.fruit || languageFocus?.objectGroup });
  const conceptLabel = selected?.concept || selected?.fruit || selected?.label || '当前概念';
  const categoryLabel = selected?.category || (selected?.fruit ? '水果' : languageFocus?.objectGroup || '未分类');

  const routeNodes = coreNodes.filter((node) => node.role === 'route');
  const resultNodes = coreNodes.filter((node) => ['unifiedDecode', 'hardBinding', 'hardLong', 'hardLocal', 'hardTriplet'].includes(node.role));
  const semanticNodes = coreNodes.filter((node) => ['style', 'logic', 'syntax'].includes(node.role));
  const propagationNodes = coreNodes.filter((node) => node.role === 'query' || node.role === 'macro' || node.role === 'route');

  const referenceByLayer = {
    basic_encoding: blendPosition(selectedPosition, familyView.conceptCenter, 0.18),
    static_encoding: blendPosition(familyView.familyCenter, familyView.conceptCenter, 0.52),
    dynamic_route: averagePosition(routeNodes, shiftPosition(familyView.conceptCenter, 2.2, 0.3, 1.4)),
    result_recovery: averagePosition(resultNodes, shiftPosition(familyView.conceptCenter, 4.4, 0.9, 2.1)),
    propagation_encoding: averagePosition(propagationNodes, shiftPosition(familyView.conceptCenter, 3.2, -0.4, 1.1)),
    semantic_roles: averagePosition(semanticNodes, shiftPosition(familyView.conceptCenter, 1.8, 1.6, -1.4)),
  };

  const layerHints = {
    basic_encoding: familyView.conceptNodes[0]?.layer ?? selected?.layer ?? 4,
    static_encoding: averagePosition(familyView.familyNodes.map((node) => ({ position: [node.layer, 0, 0] })), [selected?.layer ?? 8, 0, 0])[0],
    dynamic_route: averagePosition(routeNodes.map((node) => ({ position: [node.layer, 0, 0] })), [14, 0, 0])[0],
    result_recovery: averagePosition(resultNodes.map((node) => ({ position: [node.layer, 0, 0] })), [20, 0, 0])[0],
    propagation_encoding: averagePosition(propagationNodes.map((node) => ({ position: [node.layer, 0, 0] })), [17, 0, 0])[0],
    semantic_roles: averagePosition(semanticNodes.map((node) => ({ position: [node.layer, 0, 0] })), [22, 0, 0])[0],
  };

  const layers = CONCEPT_ASSOCIATION_LAYER_META.map((meta, index) => {
    const matchedNodes = pickConceptAssociationNodes(coreNodes, {
      roles: meta.roles,
      conceptAliases,
      categoryAliases,
      referencePosition: referenceByLayer[meta.id] || selectedPosition,
      layerHint: layerHints[meta.id],
      limit: meta.id === 'semantic_roles' ? 4 : 6,
    });
    const anchorPosition = averagePosition(matchedNodes, referenceByLayer[meta.id] || selectedPosition);
    const avgSignal = matchedNodes.length
      ? matchedNodes.reduce((sum, node) => sum + nodeSignalStrength(node), 0) / matchedNodes.length
      : 0;
    const topNode = matchedNodes
      .slice()
      .sort((left, right) => nodeSignalStrength(right) - nodeSignalStrength(left))[0] || null;

    return {
      ...meta,
      order: index,
      anchorPosition,
      nodes: matchedNodes,
      nodeIds: matchedNodes.map((node) => node.id),
      nodeCount: matchedNodes.length,
      avgSignal,
      topNodeLabel: topNode?.label || '未命中',
      layerSpanLabel: matchedNodes.length
        ? `${Math.min(...matchedNodes.map((node) => node.layer))} - ${Math.max(...matchedNodes.map((node) => node.layer))}`
        : '未命中',
    };
  });

  const relations = layers.slice(0, -1).map((layer, index) => {
    const nextLayer = layers[index + 1];
    const currentIds = new Set(layer.nodeIds);
    const nextIds = new Set(nextLayer.nodeIds);
    const linkedLinks = (Array.isArray(links) ? links : []).filter((link) => (
      (currentIds.has(link?.from) && nextIds.has(link?.to))
      || (currentIds.has(link?.to) && nextIds.has(link?.from))
    ));
    const relationCoverage = Math.min(
      1,
      linkedLinks.length / Math.max(1, Math.min(layer.nodeCount || 1, nextLayer.nodeCount || 1))
    );
    const distancePenalty = Math.min(1, distanceBetweenPositions(layer.anchorPosition, nextLayer.anchorPosition) / 8);
    const strength = Math.max(
      0.12,
      Math.min(
        1,
        relationCoverage * 0.6
        + ((layer.avgSignal + nextLayer.avgSignal) / 2) * 0.32
        + (1 - distancePenalty) * 0.22
      )
    );
    const label = strength >= 0.72 ? '强关联' : strength >= 0.45 ? '中关联' : '弱关联';
    return {
      id: `${layer.id}->${nextLayer.id}`,
      fromLayerId: layer.id,
      toLayerId: nextLayer.id,
      fromLabel: layer.label,
      toLabel: nextLayer.label,
      color: nextLayer.color,
      strength,
      label,
      linkedCount: linkedLinks.length,
      points: [layer.anchorPosition, nextLayer.anchorPosition],
    };
  });

  const nodeHighlightMap = {};
  layers.forEach((layer) => {
    layer.nodes.forEach((node, index) => {
      nodeHighlightMap[node.id] = {
        color: layer.color,
        opacity: Math.max(0.18, 0.38 - index * 0.03),
        radius: Math.max(0.16, toSafeNumber(node.size, 0.24) * 0.42),
      };
    });
  });

  return {
    conceptLabel,
    categoryLabel,
    selectedNodeId: selected.id,
    layers,
    relations,
    nodeHighlightMap,
    totalLinkedNodes: layers.reduce((sum, layer) => sum + layer.nodeCount, 0),
    totalRelationStrength: relations.length ? relations.reduce((sum, relation) => sum + relation.strength, 0) / relations.length : 0,
  };
}

function buildNodeEmphasisMap(nodes = [], primaryIds = new Set(), secondaryIds = new Set(), tertiaryIds = new Set()) {
  return Object.fromEntries(
    nodes.map((node) => {
      let emphasis = node?.role === 'background' ? 0.04 : 0.08;
      if (tertiaryIds.has(node.id)) {
        emphasis = Math.max(emphasis, 0.22);
      }
      if (secondaryIds.has(node.id)) {
        emphasis = Math.max(emphasis, 0.42);
      }
      if (primaryIds.has(node.id)) {
        emphasis = 1;
      }
      return [node.id, emphasis];
    })
  );
}

// ---- Animation & import builders ----

function buildAnimationSceneProfile(nodes = [], selected = null, animationMode = 'none', scanMechanismData = null) {
  const coreNodes = Array.isArray(nodes) ? nodes.filter((node) => node?.role !== 'background') : [];
  if (animationMode === 'none' || !selected || coreNodes.length === 0) {
    return {
      emphasisMap: {},
      label: APPLE_ANIMATION_OPTIONS.find((opt) => opt.id === animationMode)?.label || '无动画',
    };
  }

  const familyView = buildFamilyPatchViewModel(coreNodes, selected, scanMechanismData);
  const routeNodes = coreNodes.filter((node) => node.role === 'route');
  const attributeNodes = coreNodes.filter((node) => ['style', 'logic', 'syntax'].includes(node.role));
  const protocolNodes = coreNodes.filter((node) => ['unifiedDecode', 'route', 'query'].includes(node.role));
  const layerRelayNodes = coreNodes
    .filter((node) => node.role === 'query')
    .slice()
    .sort((a, b) => a.layer - b.layer)
    .filter((node, idx, arr) => idx === 0 || node.layer !== arr[idx - 1].layer)
    .slice(0, 5);
  const minimalWitness = familyView.selectedConceptMinimal?.subset_flat_indices
    ? familyView.instanceWitness.slice(0, Math.min(5, familyView.selectedConceptMinimal.subset_flat_indices.length))
    : familyView.instanceWitness.slice(0, 4);

  const primaryIds = new Set([selected?.id].filter(Boolean));
  const secondaryIds = new Set();
  const tertiaryIds = new Set();
  const addPrimary = (items = []) => items.forEach((node) => node?.id && primaryIds.add(node.id));
  const addSecondary = (items = []) => items.forEach((node) => node?.id && secondaryIds.add(node.id));
  const addTertiary = (items = []) => items.forEach((node) => node?.id && tertiaryIds.add(node.id));

  switch (animationMode) {
    case 'family_patch_formation':
      addPrimary(familyView.prototypeWitness);
      addSecondary(familyView.familyNodes);
      addTertiary(familyView.siblingNodes);
      break;
    case 'instance_offset':
      addPrimary(familyView.instanceWitness);
      addSecondary(familyView.conceptNodes);
      addTertiary(familyView.familyNodes);
      break;
    case 'attribute_fiber':
      addPrimary(attributeNodes);
      addSecondary(familyView.conceptNodes);
      break;
    case 'successor_transport':
      addPrimary(routeNodes);
      addSecondary(familyView.conceptNodes);
      addTertiary(protocolNodes);
      break;
    case 'protocol_bridge':
      addPrimary(protocolNodes);
      addSecondary(routeNodes);
      addTertiary(familyView.conceptNodes);
      break;
    case 'cross_layer_relay':
      addPrimary(layerRelayNodes);
      addSecondary(routeNodes);
      break;
    case 'ablation_shockwave':
      addPrimary(familyView.instanceWitness);
      addSecondary(familyView.conceptNodes);
      break;
    case 'counterfactual_split':
      addPrimary(familyView.conceptNodes);
      addSecondary(familyView.siblingNodes);
      addTertiary(familyView.familyNodes);
      break;
    case 'minimal_circuit_peeloff':
      addPrimary(minimalWitness);
      addSecondary(familyView.instanceWitness);
      break;
    case 'margin_breathing':
      addPrimary(familyView.familyNodes);
      addSecondary(familyView.siblingNodes);
      break;
    case 'offset_sparsity':
      addPrimary(familyView.instanceWitness.slice(0, 3));
      addSecondary(familyView.instanceWitness.slice(3, 6));
      break;
    case 'prototype_instance_tug':
      addPrimary(familyView.prototypeWitness.slice(0, 3));
      addPrimary(familyView.instanceWitness.slice(0, 3));
      addSecondary(familyView.familyNodes);
      addSecondary(familyView.conceptNodes);
      addTertiary(familyView.siblingNodes);
      break;
    case 'stage_transition':
      addPrimary(familyView.familyNodes);
      addSecondary(routeNodes);
      addTertiary(protocolNodes);
      break;
    default:
      addSecondary(coreNodes);
      break;
  }

  return {
    emphasisMap: buildNodeEmphasisMap(coreNodes, primaryIds, secondaryIds, tertiaryIds),
    label: APPLE_ANIMATION_OPTIONS.find((opt) => opt.id === animationMode)?.label || '动画',
  };
}

function buildConceptNeuronSetFromSignature(name, category = '未分类', signatureIndices = [], idx = 0, dff = DFF, maxNodes = IMPORTED_QUERY_NODE_MAX) {
  const normalized = name.trim().toLowerCase();
  const normalizedCategory = category.trim().toLowerCase() || '未分类';
  const baseHash = hashString(`import-${normalized}-${normalizedCategory}-${idx}`);
  const setId = `import-${normalized.replace(/[^a-z0-9\u4e00-\u9fa5]+/gi, '-')}-${normalizedCategory.replace(/[^a-z0-9\u4e00-\u9fa5]+/gi, '-')}-${baseHash}`;
  const color = `hsl(${baseHash % 360}, 84%, 66%)`;

  const indices = signatureIndices
    .map((v) => toSafeNumber(v, -1))
    .filter((v) => Number.isFinite(v) && v >= 0)
    .slice(0, maxNodes);

  const nodes = indices.map((flatIdx, i) => {
    const layer = Math.floor(flatIdx / dff);
    const neuron = flatIdx % dff;
    const layerClamped = Math.max(0, Math.min(LAYER_COUNT - 1, layer));
    const neuronClamped = Math.max(0, neuron);
    const rank = i + 1;
    const score = Math.max(0.08, 1 - i / Math.max(4, indices.length));
    return {
      id: `${setId}-${i}`,
      label: `${name} Sig ${rank}`,
      role: 'query',
      concept: name,
      category,
      layer: layerClamped,
      neuron: neuronClamped,
      metric: 'signature_rank_score',
      value: score,
      strength: score,
      source: 'mass_noun_encoding_scan_import',
      color,
      position: neuronToPosition(layerClamped, neuronClamped, 0.2 + i * 0.024),
      size: 0.12 + score * 0.16,
      phase: i * 0.28,
    };
  });

  return {
    id: setId,
    name,
    category,
    normalized,
    normalizedCategory,
    color,
    nodes,
  };
}

function buildSharedReuseSet(reusedRecords = [], dff = DFF, maxNodes = IMPORTED_QUERY_NODE_MAX, idx = 0) {
  const list = reusedRecords.slice(0, maxNodes);
  const baseHash = hashString(`shared-reuse-${idx}`);
  const setId = `import-shared-reused-${baseHash}`;
  const color = '#ffd166';
  const nodes = list.map((rec, i) => {
    const layer = Number.isFinite(rec?.layer) ? rec.layer : Math.floor(toSafeNumber(rec?.flat_index, 0) / dff);
    const neuron = Number.isFinite(rec?.neuron) ? rec.neuron : toSafeNumber(rec?.flat_index, 0) % dff;
    const layerClamped = Math.max(0, Math.min(LAYER_COUNT - 1, layer));
    const neuronClamped = Math.max(0, neuron);
    const count = toSafeNumber(rec?.count, 1);
    const score = Math.max(0.1, Math.min(1, count / 12));
    return {
      id: `${setId}-${i}`,
      label: `Shared Reuse ${i + 1}`,
      role: 'query',
      concept: '共享复用神经元',
      category: '共享',
      layer: layerClamped,
      neuron: neuronClamped,
      metric: 'reuse_count_score',
      value: count,
      strength: score,
      source: 'mass_noun_encoding_scan_import',
      color,
      position: neuronToPosition(layerClamped, neuronClamped, 0.24 + i * 0.02),
      size: 0.12 + score * 0.18,
      phase: 0.18 * i,
    };
  });
  return {
    id: setId,
    name: '共享复用神经元',
    category: '共享',
    normalized: '共享复用神经元',
    normalizedCategory: '共享',
    color,
    nodes,
  };
}

// ---- Multidim probe ----

function buildMultidimNodesFromProbe(probeData, visibleDims = { style: true, logic: true, syntax: true }, topN = 64) {
  if (!probeData || !probeData.dimensions) {
    return [];
  }
  const dims = ['style', 'logic', 'syntax'];
  const maxNodes = Math.max(8, Math.min(256, toSafeNumber(topN, 64)));
  const nodes = [];
  dims.forEach((dim, dimIdx) => {
    if (visibleDims[dim] === false) {
      return;
    }
    const rows = probeData?.dimensions?.[dim]?.specific_top_neurons || probeData?.dimensions?.[dim]?.top_neurons || [];
    const color = ROLE_COLORS[dim] || '#84f1ff';
    rows.slice(0, maxNodes).forEach((row, i) => {
      const layer = Math.max(0, Math.min(LAYER_COUNT - 1, toSafeNumber(row?.layer, 0)));
      const neuron = Math.max(0, toSafeNumber(row?.neuron, 0));
      const score = Math.max(0.05, toSafeNumber(row?.specific_score, toSafeNumber(row?.mean_abs_delta, 0.1)));
      nodes.push({
        id: `multidim-${dim}-${i}-l${layer}-n${neuron}`,
        label: `${DIMENSION_LABELS[dim] || dim} ${i + 1}`,
        role: dim,
        dimension: dim,
        concept: DIMENSION_LABELS[dim] || dim,
        category: '多维编码',
        layer,
        neuron,
        metric: 'dimension_specific_score',
        value: score,
        strength: score,
        source: 'multidim_encoding_probe',
        color,
        position: neuronToPosition(layer, neuron, 0.18 + i * 0.018 + dimIdx * 0.06),
        size: 0.11 + Math.min(0.28, Math.abs(score) * 0.08),
        phase: dimIdx * 0.7 + i * 0.22,
      });
    });
  });
  return nodes;
}

// ---- Hard problem & unified decode ----

function buildHardProblemNodes(hardProblemResults = {}) {
  const expEntries = Object.entries(hardProblemResults || {});
  if (expEntries.length === 0) {
    return [];
  }
  const roleByExp = {
    hard_problem_dynamic_binding_v1: 'hardBinding',
    hard_problem_long_horizon_trace_v1: 'hardLong',
    hard_problem_local_credit_assignment_v1: 'hardLocal',
    triplet_targeted_causal_scan_v1: 'hardTriplet',
    triplet_targeted_multiseed_stability_v1: 'hardTriplet',
    hard_problem_variable_binding_verification_v1: 'hardBinding',
    minimal_causal_circuit_search_v1: 'hardLocal',
    unified_coordinate_system_test_v1: 'unifiedDecode',
    concept_family_parallel_scale_v1: 'hardTriplet',
  };
  const metricPriority = {
    hard_problem_dynamic_binding_v1: ['binding_stability_index', 'role_swap_error_rate', 'collision_rate_top1', 'subject_decode_accuracy'],
    hard_problem_long_horizon_trace_v1: ['layer_transport_stability_mean', 'long_horizon_decay', 'hop_recovery_mean'],
    hard_problem_local_credit_assignment_v1: ['local_global_consistency_mean', 'local_sufficiency_mean', 'local_selectivity_mean'],
    hard_problem_variable_binding_verification_v1: [
      'mean_delta',
      'improved_dimension_count',
      'enhanced.rewrite_accuracy',
      'enhanced.role_swap_accuracy',
      'enhanced.cross_sentence_chain_accuracy',
    ],
    minimal_causal_circuit_search_v1: [
      'global.intervention_drop_mean',
      'global.reproducibility_jaccard_mean',
      'global.fidelity_mean',
      'global.min_subset_size_mean',
    ],
    unified_coordinate_system_test_v1: [
      'unified_coordinate_score',
      'probe_orthogonality.orthogonality_index',
      'ablation_coupling.decoupling_score',
      'concept_dim_alignment.concept_dim_coupling_abs_mean',
    ],
    concept_family_parallel_scale_v1: [
      'apple_chain_summary.shared_base_ratio_vs_micro_union.mean',
      'cat_chain_summary.shared_base_ratio_vs_micro_union.mean',
      'apple_vs_cat_shared_base_gap_mean',
    ],
    triplet_targeted_causal_scan_v1: [
      'triplet_minimal_records',
      'triplet_counterfactual_records',
      'axis_specificity_index',
      'triplet_separability_index',
      'global_mean_causal_margin_seq_logprob',
    ],
    triplet_targeted_multiseed_stability_v1: [
      'triplet_counterfactual_records',
      'global_mean_causal_margin_seq_logprob',
      'global_positive_causal_margin_ratio',
      'queen_recovery_ratio_mean',
      'king_recovery_ratio_mean',
    ],
  };

  const nodes = [];
  expEntries.forEach(([expId, payload], expIdx) => {
    const role = roleByExp[expId] || 'hardBinding';
    const color = ROLE_COLORS[role] || '#f97316';
    const title = HARD_PROBLEM_EXPERIMENT_LABELS[expId] || payload?.title || expId;
    const metrics = payload?.metrics || {};
    const preferredKeys = metricPriority[expId] || Object.keys(metrics);
    const keys = preferredKeys.filter((k) => k in metrics).slice(0, 6);
    const resolvedKeys = (keys.length > 0 ? keys : preferredKeys).slice(0, 6);
    resolvedKeys.forEach((k, i) => {
      const rawMetric = getMetricByPath(metrics, k);
      const val = extractMetricScalar(rawMetric);
      const strength = metricNodeStrength(k, val);
      const seed = hashString(`hard|${expId}|${k}|${i}|${expIdx}`);
      const layer = Math.max(0, Math.min(LAYER_COUNT - 1, Math.floor(pseudoRandom(seed + 7) * LAYER_COUNT)));
      const neuron = Math.max(0, Math.floor(pseudoRandom(seed + 13) * DFF));
      nodes.push({
        id: `hard-${expId}-${k}-${i}`,
        label: `${title} ${k}`,
        role,
        concept: title,
        category: '硬伤实验',
        layer,
        neuron,
        metric: k,
        value: Number.isFinite(val) ? val : 0,
        strength: Math.max(0.12, 0.2 + strength * 0.8),
        source: 'agi_research_result_v1',
        color,
        position: neuronToPosition(layer, neuron, 0.28 + i * 0.03 + expIdx * 0.05),
        size: 0.12 + Math.max(0.05, strength) * 0.18,
        phase: expIdx * 0.5 + i * 0.22,
      });
    });
  });
  return nodes;
}

function parseDominantLayers(voteObj) {
  if (!voteObj || typeof voteObj !== 'object') {
    return [];
  }
  const topPattern = Object.entries(voteObj).sort((a, b) => Number(b?.[1] || 0) - Number(a?.[1] || 0))[0]?.[0];
  if (!topPattern || typeof topPattern !== 'string') {
    return [];
  }
  return topPattern
    .split(',')
    .map((x) => Number(x))
    .filter((n) => Number.isFinite(n) && n >= 0 && n < LAYER_COUNT);
}

function buildUnifiedDecodeNodes(unifiedDecodeResult) {
  if (!unifiedDecodeResult) {
    return [];
  }
  const dims = ['style', 'logic', 'syntax'];
  const nodes = [];
  dims.forEach((dim, dimIdx) => {
    const axis = unifiedDecodeResult?.axis_stability?.dimensions?.[dim] || {};
    const causal = unifiedDecodeResult?.causal_separation?.diagonal_advantage?.[dim] || {};
    const profileCos = Number(axis?.profile_cosine_mean);
    const diagAdv = Number(causal?.mean);
    const strength = clamp01((Number.isFinite(profileCos) ? profileCos : 0) * 0.8 + (Number.isFinite(diagAdv) ? Math.max(0, diagAdv) : 0) * 2.0);
    const layers = parseDominantLayers(axis?.dominant_layer_pattern_votes);
    const fallbackLayer = Math.floor((dimIdx / Math.max(1, dims.length - 1)) * (LAYER_COUNT - 1));
    const layerList = layers.length > 0 ? layers.slice(0, 4) : [fallbackLayer];
    layerList.forEach((layer, li) => {
      const seed = hashString(`unified|${dim}|${li}|${layer}`);
      const neuron = Math.max(0, Math.floor(pseudoRandom(seed + 37) * DFF));
      nodes.push({
        id: `unified-${dim}-${li}-l${layer}`,
        label: `统一解码 ${DIMENSION_LABELS[dim] || dim}`,
        role: 'unifiedDecode',
        concept: DIMENSION_LABELS[dim] || dim,
        category: '统一解码',
        layer,
        neuron,
        metric: 'profile_cosine_mean',
        value: Number.isFinite(profileCos) ? profileCos : 0,
        strength: 0.18 + Math.max(0.08, strength) * 0.75,
        source: 'unified_math_structure_decode',
        color: ROLE_COLORS.unifiedDecode,
        position: neuronToPosition(layer, neuron, 0.25 + li * 0.03 + dimIdx * 0.06),
        size: 0.12 + Math.max(0.06, strength) * 0.16,
        phase: dimIdx * 0.7 + li * 0.21,
      });
    });
  });
  return nodes;
}

// ---- Apple switch mechanism ----

function isAppleSwitchMechanismPayload(data) {
  return data?.schema_version === APPLE_SWITCH_MECHANISM_SCHEMA && data?.concept === 'apple' && data?.models;
}

function getAppleSwitchUnitColor(unit = {}) {
  const lateMean = Number(unit?.signed_effect?.late_mean_signed_contrast_switch_coupling || 0);
  const role = String(unit?.role || '');
  const kind = String(unit?.kind || '');
  if (lateMean > 0 || role === 'heldout_booster') {
    return '#fb7185';
  }
  if (kind === 'mlp_neuron' || role === 'anchor_neuron') {
    return '#f59e0b';
  }
  if (role.includes('skeleton') || role.includes('main_booster')) {
    return '#38bdf8';
  }
  if (role.includes('bridge')) {
    return '#a78bfa';
  }
  return '#6ee7b7';
}

function getAppleSwitchUnitRoleLabel(role = '') {
  return APPLE_SWITCH_ROLE_LABELS[role] || role || '未分类';
}

function buildAppleSwitchMechanismNodes(appleSwitchMechanismData) {
  if (!isAppleSwitchMechanismPayload(appleSwitchMechanismData)) {
    return [];
  }
  const nodes = [];
  Object.entries(appleSwitchMechanismData.models || {}).forEach(([modelKey, modelPayload], modelIdx) => {
    const modelColor = APPLE_SWITCH_MODEL_COLORS[modelKey] || '#93c5fd';
    const actualLayerCount = Math.max(1, Number(modelPayload?.actual_layer_count || LAYER_COUNT));
    (modelPayload?.core_units || []).forEach((unit, unitIdx) => {
      const actualLayer = Number(unit?.actual_layer_index || 0);
      const sceneLayer = Number.isFinite(unit?.scene_layer_index)
        ? Number(unit.scene_layer_index)
        : Math.round((actualLayer / Math.max(1, actualLayerCount - 1)) * (LAYER_COUNT - 1));
      const slot = unit?.kind === 'attention_head'
        ? Number(unit?.head_index ?? unitIdx)
        : Number(unit?.neuron_index ?? unitIdx);
      const neuron = Math.max(0, Math.min(DFF - 1, Math.round((slot % 512) * 36 + modelIdx * 240 + unitIdx * 19)));
      const effectiveScore = Number(unit?.scores?.effective_score || 0);
      const causalScore = Number(unit?.scores?.causal_score || 0);
      const lateMean = Number(unit?.signed_effect?.late_mean_signed_contrast_switch_coupling || 0);
      const directionLabel = unit?.signed_effect?.direction_label || (lateMean <= 0 ? '正向支撑' : '反向校正');
      const roleLabel = getAppleSwitchUnitRoleLabel(unit?.role);
      const unitTypeLabel = unit?.kind === 'mlp_neuron' ? 'MLP 神经元' : '注意力头';
      const color = getAppleSwitchUnitColor(unit);
      nodes.push({
        id: `apple-switch-${modelKey}-${unit.unit_id}`,
        label: `${modelKey} ${unit.unit_id}`,
        role: lateMean > 0 ? 'hardBinding' : (unit?.kind === 'mlp_neuron' ? 'micro' : 'route'),
        nodeGroup: 'apple_switch_mechanism',
        detailType: 'apple_switch_unit',
        concept: 'apple',
        category: '苹果切换机制',
        modelKey,
        modelName: modelPayload?.model_name || modelKey,
        unitId: unit.unit_id,
        unitRole: unit.role,
        roleLabel,
        unitKind: unit.kind,
        unitTypeLabel,
        layer: sceneLayer,
        actualLayer,
        sceneLayer,
        neuron,
        metric: 'effective_score',
        value: effectiveScore,
        strength: Math.max(0.18, 0.22 + effectiveScore * 0.72),
        source: 'apple_switch_mechanism_view_v1',
        color,
        position: neuronToPosition(sceneLayer, neuron, 0.22 + effectiveScore * 0.45 + modelIdx * 0.04),
        size: 0.14 + effectiveScore * 0.24,
        phase: modelIdx * 0.8 + unitIdx * 0.24,
        effectiveScore,
        causalScore,
        signedLateMean: lateMean,
        directionLabel,
        modelColor,
        detailText: [
          `${roleLabel}`,
          `${unitTypeLabel}`,
          `真实层 L${actualLayer}`,
          `有效分数 ${effectiveScore.toFixed(3)}`,
        ].join(' | '),
        appleSwitchUnit: unit,
      });
    });
  });
  return nodes;
}

function buildAppleSwitchMechanismLinks(appleSwitchMechanismData, nodes = []) {
  if (!isAppleSwitchMechanismPayload(appleSwitchMechanismData) || !Array.isArray(nodes) || nodes.length === 0) {
    return [];
  }
  const byUnitId = Object.fromEntries(nodes.map((node) => [node.unitId, node]));
  const links = [];
  Object.entries(appleSwitchMechanismData.models || {}).forEach(([modelKey, modelPayload]) => {
    const subsetIds = Array.isArray(modelPayload?.effective_circuit?.final_subset)
      ? modelPayload.effective_circuit.final_subset.map((item) => item?.candidate_id).filter(Boolean)
      : [];
    const subsetNodes = subsetIds
      .map((unitId) => byUnitId[unitId])
      .filter(Boolean)
      .sort((a, b) => a.actualLayer - b.actualLayer);
    for (let idx = 0; idx < subsetNodes.length - 1; idx += 1) {
      const fromNode = subsetNodes[idx];
      const toNode = subsetNodes[idx + 1];
      links.push({
        id: `apple-switch-link-${modelKey}-${fromNode.unitId}-${toNode.unitId}`,
        from: fromNode.id,
        to: toNode.id,
        color: APPLE_SWITCH_MODEL_COLORS[modelKey] || '#93c5fd',
        points: [fromNode.position, toNode.position],
      });
    }
  });
  return links;
}

// ---- Display & puzzle helpers ----

function nodeDisplayGroup(role) {
  if (role === 'background') {
    return 'background';
  }
  if (role === 'query') {
    return 'query';
  }
  if (role === 'style' || role === 'logic' || role === 'syntax') {
    return 'multidim';
  }
  if (role === 'unifiedDecode') {
    return 'unified';
  }
  if (role === 'hardBinding' || role === 'hardLong' || role === 'hardLocal' || role === 'hardTriplet') {
    return 'hard';
  }
  return 'core';
}

function isNodeVisibleByDisplayLevels(node, displayLevels) {
  if (!node || node.role === 'background') {
    return false;
  }
  const levels = displayLevels || {};
  if (node.detailType === 'apple_switch_unit' || node.nodeGroup === 'apple_switch_mechanism') {
    return levels.parameter_state !== false;
  }
  if (node.nodeGroup === 'concept_core' || String(node.id || '').startsWith('apple-core-')) {
    return levels.parameter_state !== false;
  }
  if (node.role === 'fruitGeneral' || node.role === 'fruitSpecific') {
    return levels.object_family !== false;
  }
  if (node.role === 'micro') {
    return levels.parameter_state !== false;
  }
  if (node.role === 'style' || node.role === 'logic' || node.role === 'syntax') {
    return levels.advanced_analysis !== false;
  }
  return levels.basic_neurons !== false;
}

function normalizePuzzleResearchLayer(layerKey = '') {
  if (LAYER_PARAMETER_STATE_ORDER.includes(layerKey)) {
    return layerKey;
  }
  if (layerKey === 'advanced_analysis') {
    return 'result_recovery';
  }
  return 'static_encoding';
}

function buildPuzzleDisplayPreset(puzzleRecord = null) {
  const base = {
    displayLevels: {
      basic_neurons: true,
      object_family: false,
      parameter_state: false,
      mechanism_chain: false,
      advanced_analysis: false,
    },
    showAlgorithmConceptCore: false,
    showAlgorithmStaticEncoding: false,
  };

  if (!puzzleRecord) {
    return base;
  }

  switch (puzzleRecord.layerKey) {
    case 'static_encoding':
      return base;
    case 'dynamic_route':
      return {
        displayLevels: {
          ...base.displayLevels,
          parameter_state: true,
          mechanism_chain: true,
        },
        showAlgorithmConceptCore: false,
        showAlgorithmStaticEncoding: false,
      };
    case 'result_recovery':
      return {
        displayLevels: {
          ...base.displayLevels,
          parameter_state: true,
          mechanism_chain: true,
        },
        showAlgorithmConceptCore: false,
        showAlgorithmStaticEncoding: false,
      };
    case 'advanced_analysis':
      return {
        displayLevels: {
          ...base.displayLevels,
          parameter_state: true,
          mechanism_chain: true,
          advanced_analysis: true,
        },
        showAlgorithmConceptCore: false,
        showAlgorithmStaticEncoding: false,
      };
    default:
      return base;
  }
}

function getPuzzlePreferredRoles(layerKey = '') {
  switch (layerKey) {
    case 'static_encoding':
      return ['fruitGeneral', 'fruitSpecific', 'query'];
    case 'dynamic_route':
      return ['route', 'query', 'macro'];
    case 'result_recovery':
      return ['route', 'macro', 'query'];
    case 'propagation_encoding':
      return ['query', 'macro', 'route'];
    case 'semantic_roles':
      return ['style', 'logic', 'syntax', 'query'];
    case 'advanced_analysis':
      return ['hardBinding', 'hardLong', 'hardLocal', 'hardTriplet', 'unifiedDecode', 'route', 'query'];
    default:
      return ['fruitGeneral', 'fruitSpecific', 'query', 'route', 'macro'];
  }
}

function isNodeMatchedByPuzzle(node, puzzleRecord = null) {
  if (!node || !puzzleRecord || node.role === 'background') {
    return false;
  }
  const [startLayer, endLayer] = Array.isArray(puzzleRecord.layerRange) ? puzzleRecord.layerRange : [null, null];
  if (Number.isFinite(startLayer) && Number.isFinite(endLayer)) {
    return node.layer >= startLayer && node.layer <= endLayer;
  }
  return node.layer >= 0;
}

function findPuzzleSelectionCandidate(nodes = [], puzzleRecord = null) {
  if (!Array.isArray(nodes) || nodes.length === 0 || !puzzleRecord) {
    return null;
  }
  const preferredRoles = getPuzzlePreferredRoles(puzzleRecord.layerKey);
  const preferredRoleSet = new Set(preferredRoles);
  const [startLayer, endLayer] = Array.isArray(puzzleRecord.layerRange) ? puzzleRecord.layerRange : [null, null];
  const middleLayer = Number.isFinite(startLayer) && Number.isFinite(endLayer) ? (startLayer + endLayer) / 2 : null;
  const matched = nodes.filter((node) => isNodeMatchedByPuzzle(node, puzzleRecord));
  if (matched.length === 0) {
    return null;
  }
  return matched
    .slice()
    .sort((left, right) => {
      const leftRolePenalty = preferredRoleSet.has(left.role) ? preferredRoles.indexOf(left.role) : preferredRoles.length + 5;
      const rightRolePenalty = preferredRoleSet.has(right.role) ? preferredRoles.indexOf(right.role) : preferredRoles.length + 5;
      if (leftRolePenalty !== rightRolePenalty) {
        return leftRolePenalty - rightRolePenalty;
      }
      const leftLayerPenalty = Number.isFinite(middleLayer) ? Math.abs(left.layer - middleLayer) : left.layer;
      const rightLayerPenalty = Number.isFinite(middleLayer) ? Math.abs(right.layer - middleLayer) : right.layer;
      if (leftLayerPenalty !== rightLayerPenalty) {
        return leftLayerPenalty - rightLayerPenalty;
      }
      return toSafeNumber(right.strength, 0) - toSafeNumber(left.strength, 0);
    })[0];
}

function getPuzzleVariablePreferredRoles(variables = []) {
  const roleSet = new Set();
  variables.forEach((variable) => {
    switch (variable) {
      case 'a':
        roleSet.add('micro');
        roleSet.add('fruitSpecific');
        roleSet.add('fruitGeneral');
        break;
      case 'r':
        roleSet.add('query');
        roleSet.add('macro');
        break;
      case 'f':
        roleSet.add('macro');
        roleSet.add('route');
        break;
      case 'g':
      case 'q':
        roleSet.add('route');
        roleSet.add('query');
        break;
      case 'b':
        roleSet.add('fruitGeneral');
        roleSet.add('query');
        break;
      case 'p':
      case 'h':
      case 'm':
      case 'c':
        roleSet.add('hardBinding');
        roleSet.add('hardLong');
        roleSet.add('hardLocal');
        roleSet.add('hardTriplet');
        roleSet.add('unifiedDecode');
        roleSet.add('route');
        break;
      default:
        break;
    }
  });
  return Array.from(roleSet);
}

function buildPuzzleNodeEmphasisMap(nodes = [], puzzleRecord = null, selectedId = null) {
  if (!Array.isArray(nodes) || nodes.length === 0 || !puzzleRecord) {
    return null;
  }

  const rolePriority = [
    ...getPuzzlePreferredRoles(puzzleRecord.layerKey),
    ...getPuzzleVariablePreferredRoles(puzzleRecord.mappedVariables),
  ];
  const rolePrioritySet = new Set(rolePriority);
  const [startLayer, endLayer] = Array.isArray(puzzleRecord.layerRange) ? puzzleRecord.layerRange : [null, null];
  const emphasisMap = {};

  nodes.forEach((node) => {
    if (!node || node.role === 'background') {
      emphasisMap[node?.id] = 0.03;
      return;
    }

    const layerMatched = Number.isFinite(startLayer) && Number.isFinite(endLayer)
      ? node.layer >= startLayer && node.layer <= endLayer
      : true;
    const roleMatched = rolePrioritySet.has(node.role);

    let emphasis = 0.06;
    if (layerMatched) {
      emphasis = 0.28;
    }
    if (roleMatched) {
      emphasis = Math.max(emphasis, 0.52);
    }
    if (layerMatched && roleMatched) {
      emphasis = 0.92;
    }
    if (selectedId && node.id === selectedId) {
      emphasis = 1;
    }
    emphasisMap[node.id] = emphasis;
  });

  return emphasisMap;
}

function buildPuzzleFocusNodeIdSet(nodes = [], puzzleRecord = null) {
  if (!Array.isArray(nodes) || nodes.length === 0 || !puzzleRecord) {
    return new Set();
  }

  const rolePriority = [
    ...getPuzzlePreferredRoles(puzzleRecord.layerKey),
    ...getPuzzleVariablePreferredRoles(puzzleRecord.mappedVariables),
  ];
  const rolePrioritySet = new Set(rolePriority);
  const [startLayer, endLayer] = Array.isArray(puzzleRecord.layerRange) ? puzzleRecord.layerRange : [null, null];

  return new Set(
    nodes
      .filter((node) => {
        if (!node || node.role === 'background') {
          return false;
        }
        const layerMatched = Number.isFinite(startLayer) && Number.isFinite(endLayer)
          ? node.layer >= startLayer && node.layer <= endLayer
          : true;
        const roleMatched = rolePrioritySet.size ? rolePrioritySet.has(node.role) : true;
        return layerMatched && roleMatched;
      })
      .map((node) => node.id)
  );
}

// ---- Replay & compare ----

function normalizeReplaySlotHintRoles(hint = '') {
  return String(hint)
    .split('->')
    .map((item) => item.trim())
    .filter(Boolean);
}

function getReplaySlotPhaseMeta(replaySlot = null, replayPhase = null) {
  const phaseSlots = Array.isArray(replaySlot?.phase_slots) ? replaySlot.phase_slots : [];
  if (!phaseSlots.length) {
    return null;
  }
  return phaseSlots.find((phase) => phase.phase === replayPhase)
    || phaseSlots.find((phase) => phase.phase === 'bridge')
    || phaseSlots[0];
}

function getReplayPhaseResearchLayer(phaseId = 'bridge') {
  switch (phaseId) {
    case 'before':
      return 'static_encoding';
    case 'after':
      return 'result_recovery';
    case 'bridge':
    default:
      return 'dynamic_route';
  }
}

function buildRepairReplaySlotFocus(replaySlot = null, sharedSubcircuitCandidates = [], replayPhase = null) {
  if (!replaySlot || !Array.isArray(sharedSubcircuitCandidates) || !sharedSubcircuitCandidates.length) {
    return null;
  }

  const phaseMeta = getReplaySlotPhaseMeta(replaySlot, replayPhase);
  const activePhaseId = phaseMeta?.phase || replayPhase || 'bridge';
  const hintRoles = normalizeReplaySlotHintRoles(replaySlot.shared_subcircuit_hint);
  const hintKey = hintRoles.join(' -> ');
  const sharedVariableSet = new Set(Array.isArray(replaySlot.shared_variable_candidates) ? replaySlot.shared_variable_candidates : []);
  const scoredCandidates = sharedSubcircuitCandidates
    .map((candidate) => {
      const candidateHint = `${candidate.fromRole} -> ${candidate.toRole}`;
      const hasAnchorVariable = Boolean(replaySlot.anchor_variable && candidate.variables.includes(replaySlot.anchor_variable));
      const sharedVariableHits = candidate.variables.filter((variable) => sharedVariableSet.has(variable));
      const fromRoleMatch = hintRoles[0] ? candidate.fromRole === hintRoles[0] : false;
      const toRoleMatch = hintRoles[1] ? candidate.toRole === hintRoles[1] : false;
      const exactHintMatch = Boolean(hintKey && candidateHint === hintKey);
      const phaseBoost = activePhaseId === 'before'
        ? (hasAnchorVariable ? 0.16 : 0) + (fromRoleMatch ? 0.2 : 0)
        : activePhaseId === 'after'
          ? (sharedVariableHits.length ? 0.14 : 0) + (toRoleMatch ? 0.2 : 0)
          : (exactHintMatch ? 0.18 : 0) + (fromRoleMatch ? 0.08 : 0) + (toRoleMatch ? 0.08 : 0);
      const slotScore = Math.max(
        0,
        Math.min(
          1.8,
          candidate.score
          + (hasAnchorVariable ? 0.34 : 0)
          + sharedVariableHits.length * 0.12
          + (fromRoleMatch ? 0.14 : 0)
          + (toRoleMatch ? 0.14 : 0)
          + (exactHintMatch ? 0.22 : 0)
          + phaseBoost
        )
      );

      return {
        ...candidate,
        candidateHint,
        hasAnchorVariable,
        sharedVariableHits,
        slotScore,
      };
    })
    .sort((left, right) => right.slotScore - left.slotScore);

  const matchedCandidates = scoredCandidates
    .filter((candidate, index) => (
      candidate.hasAnchorVariable
      || candidate.sharedVariableHits.length
      || candidate.candidateHint === hintKey
      || index === 0
    ))
    .slice(0, 3);

  if (!matchedCandidates.length) {
    return null;
  }

  return {
    slotId: replaySlot.slot_id,
    label: replaySlot.label,
    sampleLabel: replaySlot.sample_label,
    anchorVariable: replaySlot.anchor_variable || null,
    activePhaseId,
    activePhaseLabel: phaseMeta?.label || activePhaseId,
    activePhaseStatus: phaseMeta?.status || replaySlot.status || 'planned',
    sharedSubcircuitHint: replaySlot.shared_subcircuit_hint || '',
    readiness: toSafeNumber(replaySlot.replay_readiness, 0),
    status: replaySlot.status || 'planned',
    candidateLinkIds: matchedCandidates.map((candidate) => candidate.linkId),
    nodeIds: Array.from(new Set(matchedCandidates.flatMap((candidate) => [candidate.fromId, candidate.toId]).filter(Boolean))),
    strongestCandidate: matchedCandidates[0],
    candidates: matchedCandidates,
  };
}

function buildPuzzleCompareState(nodes = [], links = [], primaryPuzzle = null, comparePuzzle = null, replaySlot = null, replayPhase = null) {
  if (
    !Array.isArray(nodes)
    || !Array.isArray(links)
    || !primaryPuzzle
    || !comparePuzzle
    || primaryPuzzle.id === comparePuzzle.id
  ) {
    return null;
  }

  const primaryNodeIdSet = buildPuzzleFocusNodeIdSet(nodes, primaryPuzzle);
  const compareNodeIdSet = buildPuzzleFocusNodeIdSet(nodes, comparePuzzle);
  const nodeById = Object.fromEntries(nodes.map((node) => [node.id, node]));
  const linkById = Object.fromEntries(links.map((link) => [link.id, link]));
  const sharedVariables = (primaryPuzzle.mappedVariables || []).filter((variable) => (comparePuzzle.mappedVariables || []).includes(variable));
  const categoryMeta = {
    shared: { color: '#f8fafc', opacity: 0.92, lineWidth: 2.8, label: '共享主核' },
    primary_only: { color: '#38bdf8', opacity: 0.86, lineWidth: 2.4, label: '主拼图独有' },
    compare_only: { color: '#f97316', opacity: 0.86, lineWidth: 2.4, label: '对比拼图独有' },
    bridge: { color: '#c084fc', opacity: 0.94, lineWidth: 3.0, label: '拼图差异桥' },
  };

  const nodeCategoryMap = {};
  const nodeHighlightMap = {};
  const nodeCategoryCounts = {
    shared: 0,
    primary_only: 0,
    compare_only: 0,
  };

  nodes.forEach((node) => {
    const inPrimary = primaryNodeIdSet.has(node?.id);
    const inCompare = compareNodeIdSet.has(node?.id);
    if (!inPrimary && !inCompare) {
      return;
    }
    const category = inPrimary && inCompare
      ? 'shared'
      : inPrimary
        ? 'primary_only'
        : 'compare_only';
    nodeCategoryMap[node.id] = category;
    nodeHighlightMap[node.id] = categoryMeta[category];
    nodeCategoryCounts[category] += 1;
  });

  const linkHighlightEntries = links
    .map((link) => {
      const fromPrimary = primaryNodeIdSet.has(link?.from);
      const toPrimary = primaryNodeIdSet.has(link?.to);
      const fromCompare = compareNodeIdSet.has(link?.from);
      const toCompare = compareNodeIdSet.has(link?.to);
      if (!(fromPrimary || toPrimary || fromCompare || toCompare)) {
        return null;
      }

      let category = null;
      if ((fromPrimary && toPrimary && fromCompare && toCompare) || (nodeCategoryMap[link?.from] === 'shared' && nodeCategoryMap[link?.to] === 'shared')) {
        category = 'shared';
      } else if (fromPrimary && toPrimary && !fromCompare && !toCompare) {
        category = 'primary_only';
      } else if (fromCompare && toCompare && !fromPrimary && !toPrimary) {
        category = 'compare_only';
      } else {
        category = 'bridge';
      }

      const fromNode = nodeById[link?.from];
      const toNode = nodeById[link?.to];
      const layerSpan = Math.abs(toSafeNumber(fromNode?.layer, 0) - toSafeNumber(toNode?.layer, 0));
      return [
        link.id,
        {
          ...categoryMeta[category],
          category,
          layerSpan,
        },
      ];
    })
    .filter(Boolean)
    .sort((left, right) => {
      const priority = { bridge: 0, shared: 1, primary_only: 2, compare_only: 3 };
      const leftMeta = left[1];
      const rightMeta = right[1];
      if (priority[leftMeta.category] !== priority[rightMeta.category]) {
        return priority[leftMeta.category] - priority[rightMeta.category];
      }
      return leftMeta.layerSpan - rightMeta.layerSpan;
    });

  const localReplayEntries = linkHighlightEntries.slice(0, 14);
  const localReplayLinkIds = localReplayEntries.map(([id]) => id);
  const localReplayIdSet = new Set(localReplayLinkIds);
  const linkHighlightMap = Object.fromEntries(
    linkHighlightEntries
      .filter(([id]) => localReplayIdSet.has(id))
      .map(([id, meta]) => [id, meta])
  );
  const localReplayCategoryCounts = localReplayEntries.reduce(
    (acc, [, meta]) => {
      acc[meta.category] = (acc[meta.category] || 0) + 1;
      return acc;
    },
    { shared: 0, primary_only: 0, compare_only: 0, bridge: 0 }
  );
  const avgLayerSpan = localReplayEntries.length
    ? localReplayEntries.reduce((sum, [, meta]) => sum + toSafeNumber(meta.layerSpan, 0), 0) / localReplayEntries.length
    : 0;
  const highlightedNodeTotal = nodeCategoryCounts.shared + nodeCategoryCounts.primary_only + nodeCategoryCounts.compare_only;
  const sharedAnchorRate = highlightedNodeTotal ? nodeCategoryCounts.shared / highlightedNodeTotal : 0;
  const bridgeDominance = localReplayEntries.length ? localReplayCategoryCounts.bridge / localReplayEntries.length : 0;
  const compressionRatio = linkHighlightEntries.length ? localReplayEntries.length / linkHighlightEntries.length : 0;
  const minimalityScore = Math.max(
    0,
    Math.min(
      1,
      (1 - compressionRatio) * 0.4
      + sharedAnchorRate * 0.35
      + (1 - bridgeDominance) * 0.25
    )
  );
  let validationLabel = '仍需验证';
  if (minimalityScore >= 0.68 && bridgeDominance <= 0.45) {
    validationLabel = '裁剪较稳';
  } else if (bridgeDominance > 0.55) {
    validationLabel = '差异桥过密';
  } else if (sharedAnchorRate < 0.18) {
    validationLabel = '共享锚点偏弱';
  }

  const sharedSubcircuitCandidates = localReplayEntries
    .map(([id, meta], index) => {
      const link = linkById[id];
      const fromNode = nodeById[link?.from];
      const toNode = nodeById[link?.to];
      if (!link || !fromNode || !toNode) {
        return null;
      }

      const endpointCategories = [nodeCategoryMap[fromNode.id], nodeCategoryMap[toNode.id]].filter(Boolean);
      const endpointSharedCount = endpointCategories.filter((item) => item === 'shared').length;
      const variableHits = sharedVariables.filter((variable) => {
        const roleSet = new Set(getPuzzleVariablePreferredRoles([variable]));
        return roleSet.has(fromNode.role) || roleSet.has(toNode.role);
      });
      const layerSpan = Math.abs(toSafeNumber(fromNode.layer, 0) - toSafeNumber(toNode.layer, 0));
      const compactness = Math.max(0, 1 - Math.min(layerSpan, 12) / 12);
      const score = Math.max(
        0,
        Math.min(
          1,
          (meta.category === 'shared' ? 0.42 : meta.category === 'bridge' ? 0.34 : 0.2)
          + endpointSharedCount * 0.18
          + variableHits.length * 0.12
          + compactness * 0.16
        )
      );

      return {
        id: `shared-subcircuit-${id}`,
        linkId: id,
        rank: index + 1,
        category: meta.category,
        categoryLabel: meta.label,
        title: `${fromNode.label} -> ${toNode.label}`,
        variables: variableHits,
        fromId: fromNode.id,
        toId: toNode.id,
        fromLabel: fromNode.label,
        toLabel: toNode.label,
        fromRole: fromNode.role,
        toRole: toNode.role,
        fromLayer: fromNode.layer,
        toLayer: toNode.layer,
        layerSpan,
        endpointSharedCount,
        score,
        reason:
          endpointSharedCount >= 2
            ? '两端都落在共享主核中，适合优先验证是否为最小共享链。'
            : meta.category === 'bridge'
              ? '当前是共享主核与差异桥之间的过渡链，适合验证是否可继续裁剪。'
              : '当前链路与共享变量发生重叠，适合做最小共享子回路候选。',
      };
    })
    .filter(Boolean)
    .sort((left, right) => right.score - left.score)
    .slice(0, 5);

  const replaySlotFocus = buildRepairReplaySlotFocus(replaySlot, sharedSubcircuitCandidates, replayPhase);
  const sceneLinkIdSet = new Set(
    Array.isArray(replaySlotFocus?.candidateLinkIds) && replaySlotFocus.candidateLinkIds.length
      ? replaySlotFocus.candidateLinkIds
      : localReplayLinkIds
  );
  const sceneNodeIdSet = new Set(
    Array.isArray(replaySlotFocus?.nodeIds) && replaySlotFocus.nodeIds.length
      ? replaySlotFocus.nodeIds
      : Object.keys(nodeCategoryMap)
  );
  const sceneLinkHighlightMap = Object.fromEntries(
    Object.entries(linkHighlightMap)
      .filter(([id]) => sceneLinkIdSet.has(id))
      .map(([id, meta]) => [
        id,
        replaySlotFocus
          ? { ...meta, opacity: Math.max(meta.opacity, 0.96), lineWidth: meta.lineWidth + 0.5, slotFocused: true }
          : meta,
      ])
  );
  const sceneNodeCategoryMap = Object.fromEntries(
    Object.entries(nodeCategoryMap).filter(([id]) => sceneNodeIdSet.has(id))
  );
  const sceneNodeHighlightMap = Object.fromEntries(
    Object.entries(nodeHighlightMap)
      .filter(([id]) => sceneNodeIdSet.has(id))
      .map(([id, meta]) => [
        id,
        replaySlotFocus
          ? { ...meta, opacity: Math.max(meta.opacity, 0.96), slotFocused: true }
          : meta,
      ])
  );

  return {
    primaryPuzzleId: primaryPuzzle.id,
    comparePuzzleId: comparePuzzle.id,
    nodeCategoryMap,
    nodeHighlightMap,
    nodeCategoryCounts,
    linkHighlightMap,
    localReplayLinkIds,
    sharedVariables,
    sharedSubcircuitCandidates,
    replaySlotFocus,
    sceneLinkHighlightMap,
    sceneNodeCategoryMap,
    sceneNodeHighlightMap,
    summary: {
      sharedNodes: nodeCategoryCounts.shared,
      primaryOnlyNodes: nodeCategoryCounts.primary_only,
      compareOnlyNodes: nodeCategoryCounts.compare_only,
      localReplayLinks: localReplayLinkIds.length,
      candidateLinks: linkHighlightEntries.length,
      bridgeLinks: localReplayCategoryCounts.bridge,
      sharedSubcircuits: sharedSubcircuitCandidates.length,
      slotFocusedLinks: sceneLinkIdSet.size,
    },
    validation: {
      label: validationLabel,
      candidateLinks: linkHighlightEntries.length,
      localReplayLinks: localReplayEntries.length,
      bridgeLinks: localReplayCategoryCounts.bridge,
      sharedLinks: localReplayCategoryCounts.shared,
      primaryOnlyLinks: localReplayCategoryCounts.primary_only,
      compareOnlyLinks: localReplayCategoryCounts.compare_only,
      avgLayerSpan,
      compressionRatio,
      sharedAnchorRate,
      bridgeDominance,
      minimalityScore,
    },
  };
}

function buildAutoDisplayProfile(analysisMode) {
  if (['causal_intervention', 'counterfactual', 'robustness', 'minimal_circuit'].includes(analysisMode)) {
    return { core: 0.45, query: 0.65, multidim: 0.5, hard: 1, unified: 0.45, background: 0.08 };
  }
  if (['subspace_geometry', 'feature_decomposition', 'cross_layer_transport', 'compositionality'].includes(analysisMode)) {
    return { core: 0.5, query: 0.7, multidim: 0.95, hard: 0.45, unified: 1, background: 0.08 };
  }
  if (analysisMode === 'dynamic_prediction') {
    return { core: 0.9, query: 1, multidim: 0.85, hard: 0.8, unified: 0.8, background: 0.12 };
  }
  if (analysisMode === 'static') {
    return { core: 0.85, query: 0.85, multidim: 0.85, hard: 0.85, unified: 0.85, background: 0.12 };
  }
  return { core: 0.8, query: 0.8, multidim: 0.8, hard: 0.8, unified: 0.8, background: 0.1 };
}

// ---- Placeholder exports for imports from other modules ----

function shouldShowResearchAssetInTopRight(asset = null) {
  if (!asset) return false;
  return true;
}

function isHardProblemResultPayload(data) {
  return data && typeof data === 'object' && (data.schema_version === 'hard_problem_v1' || data.experiment_id);
}

function isUnifiedDecodePayload(data) {
  return data && typeof data === 'object' && data.schema_version === 'unified_decode_v1';
}

function isBundleManifestPayload(data) {
  return data && typeof data === 'object' && data.schema_version === 'bundle_manifest_v1';
}

function isFourTasksManifestPayload(data) {
  return data && typeof data === 'object' && data.schema_version === 'four_tasks_manifest_v1';
}

function buildArtifactPreview(asset = null) {
  if (!asset) return null;
  return { label: asset.label || asset.id || '未知资产', summary: safeJsonStringify(asset) };
}

// ---- Re-export all public functions ----

export {
  neuronToPosition,
  averagePosition,
  blendPosition,
  shiftPosition,
  normalizeVector,
  toSafeNumber,
  normalizeConceptKey,
  nodeSignalStrength,
  clamp01,
  metricNodeStrength,
  extractMetricScalar,
  getMetricByPath,
  formatPreviewValue,
  safeJsonStringify,
  pseudoRandom,
  hashString,
  extractPromptTokens,
  getFallbackTokens,
  generatePredictChain,
  buildConceptNeuronSet,
  buildFamilyPatchViewModel,
  buildConceptAliasSet,
  getAssociationNodeTexts,
  isTextMatchedByAliases,
  isNodeMatchedByAliases,
  distanceBetweenPositions,
  pickConceptAssociationNodes,
  buildConceptAssociationState,
  buildNodeEmphasisMap,
  buildAnimationSceneProfile,
  buildConceptNeuronSetFromSignature,
  buildSharedReuseSet,
  buildMultidimNodesFromProbe,
  buildHardProblemNodes,
  parseDominantLayers,
  buildUnifiedDecodeNodes,
  isAppleSwitchMechanismPayload,
  getAppleSwitchUnitColor,
  getAppleSwitchUnitRoleLabel,
  buildAppleSwitchMechanismNodes,
  buildAppleSwitchMechanismLinks,
  nodeDisplayGroup,
  isNodeVisibleByDisplayLevels,
  normalizePuzzleResearchLayer,
  buildPuzzleDisplayPreset,
  getPuzzlePreferredRoles,
  isNodeMatchedByPuzzle,
  findPuzzleSelectionCandidate,
  getPuzzleVariablePreferredRoles,
  buildPuzzleNodeEmphasisMap,
  buildPuzzleFocusNodeIdSet,
  normalizeReplaySlotHintRoles,
  getReplaySlotPhaseMeta,
  getReplayPhaseResearchLayer,
  buildRepairReplaySlotFocus,
  buildPuzzleCompareState,
  buildAutoDisplayProfile,
  shouldShowResearchAssetInTopRight,
  isHardProblemResultPayload,
  isUnifiedDecodePayload,
  isBundleManifestPayload,
  isFourTasksManifestPayload,
  buildArtifactPreview,
};

