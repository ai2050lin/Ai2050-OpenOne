import { Brain, Boxes, Layers3, Route, Target } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { PERSISTED_PUZZLE_RECORDS_V1, PERSISTED_PUZZLE_SUMMARY_V1 } from '../blueprint/data/persisted_puzzle_records_v1';
import {
  PERSISTED_REPAIR_REPLAY_SAMPLE_SLOTS_V1,
  PERSISTED_REPAIR_REPLAY_SLOT_META_V1,
} from '../blueprint/data/persisted_repair_replay_sample_slots_v1';
import BasicEncodingPanel from './BasicEncodingPanel';
import DNNAnalysisControlPanel from './DNNAnalysisControlPanel';
import LanguageDimensionSelector from './reverse/LanguageDimensionSelector';
import DNNFeatureTabs from './reverse/DNNFeatureTabs';
import QuickTestPresets from './reverse/QuickTestPresets';
import ViewModeSelect from './reverse/ViewModeSelect';
import { applyPresetToSelection, getDefaultLanguageDimSelection } from '../config/languageDimensions';

const RESEARCH_LAYERS = [
  {
    id: 'static_encoding',
    label: '静态编码层',
    desc: '看基础共享结构和静态编码底盘。',
    theoryObject: 'family_patch',
    analysisMode: 'static',
    animationMode: 'family_patch_formation',
  },
  {
    id: 'dynamic_route',
    label: '动态路径层',
    desc: '看任务词、方向词和路径分流。',
    theoryObject: 'relation_context_fiber',
    analysisMode: 'dynamic_prediction',
    animationMode: 'successor_transport',
  },
  {
    id: 'result_recovery',
    label: '结果回收层',
    desc: '看闭合、修复和结果读出。',
    theoryObject: 'protocol_bridge',
    analysisMode: 'minimal_circuit',
    animationMode: 'protocol_bridge',
  },
  {
    id: 'propagation_encoding',
    label: '传播编码层',
    desc: '看层间传播、接力和保真裂缝。',
    theoryObject: 'stage_conditioned_transport',
    analysisMode: 'cross_layer_transport',
    animationMode: 'cross_layer_relay',
  },
  {
    id: 'semantic_roles',
    label: '语义角色层',
    desc: '看对象、属性、位置、操作和结果角色。',
    theoryObject: 'attribute_fiber',
    analysisMode: 'compositionality',
    animationMode: 'attribute_fiber',
  },
];

const cardStyle = {
  padding: '12px',
  borderRadius: '10px',
  background: 'rgba(255,255,255,0.04)',
  border: '1px solid rgba(255,255,255,0.08)',
};

const sectionTitleStyle = {
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  marginBottom: '10px',
  color: '#dfe8ff',
  fontSize: '13px',
  fontWeight: 700,
};

const actionButtonStyle = {
  padding: '7px 12px',
  borderRadius: '999px',
  border: '1px solid rgba(79, 172, 254, 0.35)',
  background: 'rgba(79, 172, 254, 0.12)',
  color: '#e7f4ff',
  fontSize: '11px',
  fontWeight: 700,
  cursor: 'pointer',
};

const secondaryButtonStyle = {
  ...actionButtonStyle,
  background: 'rgba(255,255,255,0.04)',
  border: '1px solid rgba(255,255,255,0.12)',
  color: '#d8e6ff',
};

const summaryGridStyle = {
  display: 'grid',
  gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
  gap: '8px',
};

const summaryCardStyle = {
  ...cardStyle,
  padding: '10px',
};

const summaryLabelStyle = {
  fontSize: '10px',
  color: '#7f95bb',
  marginBottom: '4px',
};

const summaryValueStyle = {
  fontSize: '16px',
  fontWeight: 800,
  color: '#eef7ff',
};

const detailGridStyle = {
  display: 'grid',
  gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
  gap: '8px',
};

const detailLabelStyle = {
  fontSize: '10px',
  color: '#7f95bb',
  marginBottom: '3px',
};

const detailValueStyle = {
  fontSize: '11px',
  color: '#dbeafe',
  lineHeight: 1.5,
};

const VARIABLE_META = {
  a: { label: 'a', desc: '实体锚点' },
  r: { label: 'r', desc: '关系读取' },
  f: { label: 'f', desc: '特征纤维' },
  g: { label: 'g', desc: '路由门控' },
  q: { label: 'q', desc: '查询压缩' },
  b: { label: 'b', desc: '共享基底' },
  p: { label: 'p', desc: '协议桥接' },
  h: { label: 'h', desc: '高压约束' },
  m: { label: 'm', desc: '记忆修复' },
  c: { label: 'c', desc: '闭合读出' },
};

const VARIABLE_ROLE_MAP = {
  a: ['micro', 'fruitSpecific', 'fruitGeneral'],
  r: ['query', 'macro'],
  f: ['macro', 'route'],
  g: ['route', 'query'],
  q: ['route', 'query'],
  b: ['fruitGeneral', 'query'],
  p: ['hardBinding', 'hardLong', 'hardLocal', 'hardTriplet', 'unifiedDecode', 'route'],
  h: ['hardBinding', 'hardLong', 'hardLocal', 'hardTriplet', 'unifiedDecode', 'route'],
  m: ['hardBinding', 'hardLong', 'hardLocal', 'hardTriplet', 'unifiedDecode', 'route'],
  c: ['hardBinding', 'hardLong', 'hardLocal', 'hardTriplet', 'unifiedDecode', 'route'],
};

const variableGridStyle = {
  display: 'grid',
  gridTemplateColumns: 'minmax(0, 1fr)',
  gap: '8px',
};

const REPAIR_REPLAY_STATUS_LABEL_MAP = Object.fromEntries(
  (PERSISTED_REPAIR_REPLAY_SLOT_META_V1.statusLegend || []).map((item) => [item.id, item.label])
);

function toSafeNumber(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function getDefaultComparePuzzleId(records, activeId) {
  return records.find((item) => item.id !== activeId)?.id || null;
}

function formatLayerRange(layerRange = []) {
  if (!Array.isArray(layerRange) || layerRange.length < 2) return '未定义';
  return `${layerRange[0]} - ${layerRange[1]}`;
}

function buildLayerRelation(primaryRange = [], secondaryRange = []) {
  if (
    !Array.isArray(primaryRange) ||
    !Array.isArray(secondaryRange) ||
    primaryRange.length < 2 ||
    secondaryRange.length < 2
  ) {
    return '层范围未定义';
  }
  const overlapStart = Math.max(primaryRange[0], secondaryRange[0]);
  const overlapEnd = Math.min(primaryRange[1], secondaryRange[1]);
  if (overlapStart <= overlapEnd) {
    return `有重叠层 ${overlapStart} - ${overlapEnd}`;
  }
  if (primaryRange[1] < secondaryRange[0]) {
    return `主拼图更早，间隔 ${secondaryRange[0] - primaryRange[1]} 层`;
  }
  return `副拼图更早，间隔 ${primaryRange[0] - secondaryRange[1]} 层`;
}

function buildVariableDiff(primary = [], secondary = []) {
  const primarySet = new Set(primary);
  const secondarySet = new Set(secondary);
  return {
    shared: primary.filter((item) => secondarySet.has(item)),
    primaryOnly: primary.filter((item) => !secondarySet.has(item)),
    secondaryOnly: secondary.filter((item) => !primarySet.has(item)),
  };
}

function isNodeInRange(node, layerRange = []) {
  if (!Array.isArray(layerRange) || layerRange.length < 2) {
    return true;
  }
  const layer = toSafeNumber(node?.layer, null);
  if (!Number.isFinite(layer)) {
    return false;
  }
  return layer >= layerRange[0] && layer <= layerRange[1];
}

function getNodeSignal(node) {
  const valueSignal = toSafeNumber(node?.value, null);
  if (Number.isFinite(valueSignal)) {
    return Math.max(0, Math.min(1, valueSignal));
  }
  const strengthSignal = toSafeNumber(node?.strength, null);
  if (Number.isFinite(strengthSignal)) {
    return Math.max(0, Math.min(1, strengthSignal * 1000));
  }
  return 0;
}

function getVariableRoles(variable) {
  return VARIABLE_ROLE_MAP[variable] || [];
}

function buildVariableNodeStat(nodes = [], puzzleRecord = null, variable = '') {
  const roles = getVariableRoles(variable);
  const roleSet = new Set(roles);
  const roleNodes = Array.isArray(nodes)
    ? nodes.filter((node) => node && node.role !== 'background' && roleSet.has(node.role))
    : [];
  const matchedNodes = roleNodes.filter((node) => isNodeInRange(node, puzzleRecord?.layerRange));
  const peakNode = [...matchedNodes].sort((left, right) => getNodeSignal(right) - getNodeSignal(left))[0] || null;
  const avgSignal = matchedNodes.length
    ? matchedNodes.reduce((sum, node) => sum + getNodeSignal(node), 0) / matchedNodes.length
    : 0;

  return {
    variable,
    roles,
    matchedCount: matchedNodes.length,
    roleCount: roleNodes.length,
    avgSignal,
    peakLayer: peakNode?.layer ?? null,
    peakLabel: peakNode?.label || '未命中',
  };
}

function buildVariableCompareRows(nodes = [], primaryPuzzle = null, secondaryPuzzle = null) {
  const variables = Array.from(
    new Set([
      ...(Array.isArray(primaryPuzzle?.mappedVariables) ? primaryPuzzle.mappedVariables : []),
      ...(Array.isArray(secondaryPuzzle?.mappedVariables) ? secondaryPuzzle.mappedVariables : []),
    ])
  );

  return variables.map((variable) => {
    const primary = buildVariableNodeStat(nodes, primaryPuzzle, variable);
    const secondary = buildVariableNodeStat(nodes, secondaryPuzzle, variable);
    return {
      variable,
      meta: VARIABLE_META[variable] || { label: variable, desc: '未定义变量' },
      primary,
      secondary,
      deltaCount: primary.matchedCount - secondary.matchedCount,
      deltaSignal: primary.avgSignal - secondary.avgSignal,
    };
  });
}

function isRepairCandidatePuzzle(record = null) {
  return record?.puzzleType === 'repair_candidate' || record?.priorityAxis === 'novelty_generalization';
}

function average(values = []) {
  if (!values.length) return 0;
  return values.reduce((sum, item) => sum + item, 0) / values.length;
}

function buildRepairContrastSummary({
  activePuzzle = null,
  comparePuzzle = null,
  compareVariableDiff = null,
  variableCompareRows = [],
  compareValidation = null,
  compareSceneSummary = null,
}) {
  if (!activePuzzle || !comparePuzzle) {
    return null;
  }

  const activeIsRepair = isRepairCandidatePuzzle(activePuzzle);
  const compareIsRepair = isRepairCandidatePuzzle(comparePuzzle);
  if (!activeIsRepair && !compareIsRepair) {
    return null;
  }

  const afterPuzzle = activeIsRepair ? activePuzzle : comparePuzzle;
  const beforePuzzle = activeIsRepair ? comparePuzzle : activePuzzle;
  const afterSide = activeIsRepair ? 'primary' : 'secondary';
  const beforeSide = activeIsRepair ? 'secondary' : 'primary';
  const sharedRows = variableCompareRows.filter((row) => row.primary.matchedCount > 0 && row.secondary.matchedCount > 0);
  const improvedRows = variableCompareRows.filter((row) => {
    const afterSignal = afterSide === 'primary' ? row.primary.avgSignal : row.secondary.avgSignal;
    const beforeSignal = beforeSide === 'primary' ? row.primary.avgSignal : row.secondary.avgSignal;
    return afterSignal > beforeSignal;
  });
  const regressedRows = variableCompareRows.filter((row) => {
    const afterSignal = afterSide === 'primary' ? row.primary.avgSignal : row.secondary.avgSignal;
    const beforeSignal = beforeSide === 'primary' ? row.primary.avgSignal : row.secondary.avgSignal;
    return afterSignal < beforeSignal;
  });

  const beforeSignals = variableCompareRows.map((row) => (beforeSide === 'primary' ? row.primary.avgSignal : row.secondary.avgSignal));
  const afterSignals = variableCompareRows.map((row) => (afterSide === 'primary' ? row.primary.avgSignal : row.secondary.avgSignal));
  const sharedSignalGain = sharedRows.length
    ? average(
      sharedRows.map((row) => (
        (afterSide === 'primary' ? row.primary.avgSignal : row.secondary.avgSignal)
        - (beforeSide === 'primary' ? row.primary.avgSignal : row.secondary.avgSignal)
      ))
    )
    : 0;

  const afterOnlyVariables = afterSide === 'primary'
    ? compareVariableDiff?.primaryOnly || []
    : compareVariableDiff?.secondaryOnly || [];
  const beforeOnlyVariables = beforeSide === 'primary'
    ? compareVariableDiff?.primaryOnly || []
    : compareVariableDiff?.secondaryOnly || [];

  let verdict = '修复趋势未定';
  if (improvedRows.length > regressedRows.length && sharedSignalGain > 0) {
    verdict = '修复收益更明显';
  } else if (regressedRows.length > improvedRows.length) {
    verdict = '副作用压力偏高';
  } else if (compareValidation?.label) {
    verdict = compareValidation.label;
  }

  return {
    beforePuzzle,
    afterPuzzle,
    verdict,
    beforeAvgSignal: average(beforeSignals),
    afterAvgSignal: average(afterSignals),
    signalGain: average(afterSignals) - average(beforeSignals),
    sharedSignalGain,
    improvedVariables: improvedRows.map((row) => row.variable),
    regressedVariables: regressedRows.map((row) => row.variable),
    afterOnlyVariables,
    beforeOnlyVariables,
    validationLabel: compareValidation?.label || '未验证',
    minimalityScore: compareValidation?.minimalityScore || 0,
    sharedAnchorRate: compareValidation?.sharedAnchorRate || 0,
    bridgeDominance: compareValidation?.bridgeDominance || 0,
    localReplayLinks: compareSceneSummary?.localReplayLinks || 0,
  };
}

function buildRepairReplaySummary(repairContrastSummary = null, sharedSubcircuitCandidates = []) {
  if (!repairContrastSummary || !sharedSubcircuitCandidates.length) {
    return null;
  }

  const anchorCandidate = sharedSubcircuitCandidates[0];
  const bridgeCandidate = sharedSubcircuitCandidates.find((candidate) => candidate.category === 'bridge') || anchorCandidate;
  const strongestVariable = repairContrastSummary.improvedVariables[0]
    || repairContrastSummary.afterOnlyVariables[0]
    || repairContrastSummary.beforeOnlyVariables[0]
    || anchorCandidate.variables?.[0]
    || null;

  const phases = [
    {
      id: 'before',
      label: '修复前',
      title: repairContrastSummary.beforePuzzle.title,
      emphasis: `${Math.round(repairContrastSummary.beforeAvgSignal * 100)}%`,
      note: repairContrastSummary.regressedVariables.length
        ? `当前主要裂缝变量: ${repairContrastSummary.regressedVariables.join(' / ')}`
        : '当前主要裂缝仍待补样本。',
    },
    {
      id: 'bridge',
      label: '共享候选链',
      title: bridgeCandidate.title,
      emphasis: `${Math.round(bridgeCandidate.score * 100)}%`,
      note: bridgeCandidate.reason,
    },
    {
      id: 'after',
      label: '修复后',
      title: repairContrastSummary.afterPuzzle.title,
      emphasis: `${Math.round(repairContrastSummary.afterAvgSignal * 100)}%`,
      note: repairContrastSummary.improvedVariables.length
        ? `当前主要收益变量: ${repairContrastSummary.improvedVariables.join(' / ')}`
        : '当前收益仍需同样本回放确认。',
    },
  ];

  let verdict = '建议先回放共享候选链。';
  if (repairContrastSummary.signalGain > 0 && repairContrastSummary.minimalityScore >= 0.6) {
    verdict = '建议优先验证这条共享链是否承载主要修复收益。';
  } else if (repairContrastSummary.bridgeDominance > 0.55) {
    verdict = '桥链占比偏高，建议先裁剪再回放。';
  }

  return {
    anchorCandidate,
    bridgeCandidate,
    strongestVariable,
    phases,
    verdict,
  };
}

function buildRepairReplaySlotSummary(repairContrastSummary = null, repairReplaySummary = null, replaySlots = []) {
  if (!repairContrastSummary || !repairReplaySummary || !Array.isArray(replaySlots) || !replaySlots.length) {
    return null;
  }

  const repairPuzzleId = repairContrastSummary.afterPuzzle?.id || null;
  const baselinePuzzleId = repairContrastSummary.beforePuzzle?.id || null;
  if (!repairPuzzleId || !baselinePuzzleId) {
    return null;
  }

  const matchedSlots = replaySlots
    .filter((slot) => slot?.repair_puzzle_id === repairPuzzleId)
    .map((slot) => {
      const matchType = slot.baseline_puzzle_id === baselinePuzzleId ? 'exact' : 'repair_only';
      const phaseLabels = Array.isArray(slot.phase_slots)
        ? slot.phase_slots.map((phase) => ({
          id: phase.phase,
          label: `${phase.label}:${REPAIR_REPLAY_STATUS_LABEL_MAP[phase.status] || phase.status || '未定义'}`,
        }))
        : [];
      return {
        ...slot,
        matchType,
        matchTypeLabel: matchType === 'exact' ? '精确匹配' : '修复侧候选',
        readiness: toSafeNumber(slot.replay_readiness, 0),
        missingAssets: Array.isArray(slot.missing_assets) ? slot.missing_assets : [],
        phaseLabels,
      };
    })
    .sort((left, right) => {
      if (left.matchType !== right.matchType) {
        return left.matchType === 'exact' ? -1 : 1;
      }
      return right.readiness - left.readiness;
    });

  if (!matchedSlots.length) {
    return null;
  }

  const uniqueMissingAssets = Array.from(new Set(matchedSlots.flatMap((slot) => slot.missingAssets)));
  const avgReadiness = average(matchedSlots.map((slot) => slot.readiness));

  return {
    slots: matchedSlots,
    exactMatchCount: matchedSlots.filter((slot) => slot.matchType === 'exact').length,
    readyCount: matchedSlots.filter((slot) => slot.status === 'ready').length,
    avgReadiness,
    uniqueMissingAssets,
  };
}

function buildResearchOverviewSummary({
  workspaceSummary = null,
  currentLayerMeta = null,
  currentSceneLabel = 'circuit',
  activePuzzle = null,
  conceptAssociationState = null,
  strongestConceptRelation = null,
  compareValidation = null,
  repairContrastSummary = null,
  repairReplaySlotSummary = null,
}) {
  const currentConcept = conceptAssociationState?.conceptLabel || activePuzzle?.title || '未选中';
  const currentRisk = repairContrastSummary?.verdict
    || compareValidation?.label
    || (repairReplaySlotSummary?.uniqueMissingAssets?.length ? '样本资产未补齐' : '暂无高优先风险');
  const validationState = repairReplaySlotSummary?.exactMatchCount
    ? `已匹配 ${repairReplaySlotSummary.exactMatchCount} 个精确槽位`
    : compareValidation?.label || '尚未进入样本级验证';
  const strongestRelation = strongestConceptRelation
    ? `${strongestConceptRelation.fromLabel} -> ${strongestConceptRelation.toLabel}`
    : '概念跨层关系待形成';
  const dataStatus = repairReplaySlotSummary?.uniqueMissingAssets?.length
    ? `待补 ${repairReplaySlotSummary.uniqueMissingAssets[0]} 等验证资产`
    : (workspaceSummary?.visibleQuerySets ? `当前显示 ${workspaceSummary.visibleQuerySets} 组概念数据` : '当前没有额外数据状态');

  return {
    currentConcept,
    currentLayer: currentLayerMeta?.label || '未定义',
    currentScene: currentSceneLabel || 'circuit',
    activePuzzle: activePuzzle?.title || '未选中',
    strongestRelation,
    currentRisk,
    validationState,
    dataStatus,
    visibleConceptSets: workspaceSummary?.visibleQuerySets ?? 0,
    totalNodes: workspaceSummary?.total ?? 0,
  };
}

function buildPropertySummaryCards({
  activePuzzle = null,
  conceptAssociationState = null,
  compareValidation = null,
  repairReplaySlotSummary = null,
  variableCompareRows = [],
  compareVariableDiff = null,
}) {
  const stability = Math.round(
    (
      toSafeNumber(conceptAssociationState?.totalRelationStrength, 0) * 0.45
      + toSafeNumber(compareValidation?.minimalityScore, 0) * 0.35
      + toSafeNumber(repairReplaySlotSummary?.avgReadiness, 0) * 0.2
    ) * 100
  );
  const separabilityBase = variableCompareRows.length
    ? variableCompareRows.filter((row) => Math.abs(toSafeNumber(row.deltaSignal, 0)) >= 0.12).length / variableCompareRows.length
    : 0;
  const separability = Math.round(
    (
      separabilityBase * 0.65
      + (compareVariableDiff?.shared?.length ? Math.max(0, 1 - compareVariableDiff.shared.length / 10) * 0.35 : 0.2)
    ) * 100
  );
  const coupling = Math.round(
    (
      toSafeNumber(compareValidation?.bridgeDominance, 0) * 0.7
      + toSafeNumber(compareValidation?.sharedAnchorRate, 0) * 0.3
    ) * 100
  );
  const completeness = Math.round(
    (
      ((conceptAssociationState?.layers?.length || 0) / 6) * 0.35
      + toSafeNumber(activePuzzle?.confidence, 0) * 0.25
      + toSafeNumber(repairReplaySlotSummary?.avgReadiness, 0) * 0.4
    ) * 100
  );

  return [
    {
      id: 'stability',
      label: '稳定性',
      value: `${stability}%`,
      desc: '看跨层关系、最小性和样本准备度是否同时站得住。',
    },
    {
      id: 'separability',
      label: '可分性',
      value: `${separability}%`,
      desc: '看变量差异是否足够明显，避免不同概念混成一团。',
    },
    {
      id: 'coupling',
      label: '耦合强度',
      value: `${coupling}%`,
      desc: '看桥链占比和共享锚点率，判断层间绑定是紧还是松。',
    },
    {
      id: 'completeness',
      label: '观测完整度',
      value: `${completeness}%`,
      desc: '看六层覆盖、拼图置信度和回放槽位准备度是否够用。',
    },
  ];
}

function buildValidationEntrySummary({
  repairReplaySlotSummary = null,
  selectedRepairReplaySlot = null,
  resolvedRepairReplayPhase = null,
  compareValidation = null,
}) {
  const slots = Array.isArray(repairReplaySlotSummary?.slots) ? repairReplaySlotSummary.slots : [];
  const phaseCounts = {
    before: 0,
    bridge: 0,
    after: 0,
  };
  const statusCounts = {
    planned: 0,
    partial: 0,
    ready: 0,
  };

  slots.forEach((slot) => {
    if (slot?.status && Object.prototype.hasOwnProperty.call(statusCounts, slot.status)) {
      statusCounts[slot.status] += 1;
    }
    (slot.phase_slots || []).forEach((phase) => {
      if (phase?.phase && Object.prototype.hasOwnProperty.call(phaseCounts, phase.phase)) {
        phaseCounts[phase.phase] += 1;
      }
    });
  });

  const missingAssetCounts = {};
  slots.forEach((slot) => {
    (slot.missingAssets || []).forEach((asset) => {
      missingAssetCounts[asset] = (missingAssetCounts[asset] || 0) + 1;
    });
  });
  const nextPriorityAsset = Object.entries(missingAssetCounts)
    .sort((left, right) => right[1] - left[1])[0]?.[0] || '暂无';

  return {
    totalSlots: slots.length,
    readySlots: statusCounts.ready,
    partialSlots: statusCounts.partial,
    plannedSlots: statusCounts.planned,
    beforeCoverage: phaseCounts.before,
    bridgeCoverage: phaseCounts.bridge,
    afterCoverage: phaseCounts.after,
    nextPriorityAsset,
    currentFocus: selectedRepairReplaySlot
      ? `${selectedRepairReplaySlot.label} / ${resolvedRepairReplayPhase || '待定阶段'}`
      : '未选中验证槽位',
    validationJudge: compareValidation?.label || '尚未形成样本级验证判断',
  };
}

function FocusSummaryItem({ label, value }) {
  return (
    <div style={{ ...cardStyle, padding: '9px 10px' }}>
      <div style={summaryLabelStyle}>{label}</div>
      <div style={{ fontSize: '11px', fontWeight: 700, color: '#eef7ff', lineHeight: 1.5 }}>{value}</div>
    </div>
  );
}

function DetailItem({ label, value }) {
  return (
    <div style={{ ...cardStyle, padding: '8px 9px' }}>
      <div style={detailLabelStyle}>{label}</div>
      <div style={detailValueStyle}>{value}</div>
    </div>
  );
}

export default function LanguageResearchControlPanel({
  workspace,
  structureTab = 'circuit',
  setStructureTab = null,
}) {
  const [legacyOpen, setLegacyOpen] = useState(false);
  const [dnnAnalysisOpen, setDnnAnalysisOpen] = useState(false);
  const [reverseEngineeringOpen, setReverseEngineeringOpen] = useState(false);
  const [fallbackFocus, setFallbackFocus] = useState({
    researchLayer: 'static_encoding',
    puzzleAxisFilter: 'all',
    activePuzzleId: PERSISTED_PUZZLE_RECORDS_V1[0]?.id || null,
    comparePuzzleId: getDefaultComparePuzzleId(PERSISTED_PUZZLE_RECORDS_V1, PERSISTED_PUZZLE_RECORDS_V1[0]?.id || null),
    selectedRepairReplaySlotId: null,
    selectedRepairReplayPhase: null,
  });

  const languageFocus = workspace?.languageFocus || fallbackFocus;
  const workspaceNodes = Array.isArray(workspace?.nodes) ? workspace.nodes : [];
  const compareSceneSummary = workspace?.puzzleCompareState?.summary || null;
  const compareValidation = workspace?.puzzleCompareState?.validation || null;
  const sharedSubcircuitCandidates = workspace?.puzzleCompareState?.sharedSubcircuitCandidates || [];
  const conceptAssociationState = workspace?.conceptAssociationState || null;
  const workspaceSummary = workspace?.summary || null;
  const strongestConceptRelation = useMemo(() => {
    const relations = Array.isArray(conceptAssociationState?.relations) ? conceptAssociationState.relations : [];
    return relations.slice().sort((left, right) => toSafeNumber(right.strength, 0) - toSafeNumber(left.strength, 0))[0] || null;
  }, [conceptAssociationState]);
  const updateLanguageFocus = (patch) => {
    if (workspace?.setLanguageFocus) {
      workspace.setLanguageFocus((prev) => ({ ...prev, ...patch }));
      return;
    }
    setFallbackFocus((prev) => ({ ...prev, ...patch }));
  };

  const researchLayer = languageFocus.researchLayer || 'static_encoding';
  const puzzleAxisFilter = languageFocus.puzzleAxisFilter || 'all';
  const activePuzzleId = languageFocus.activePuzzleId || PERSISTED_PUZZLE_RECORDS_V1[0]?.id || null;
  const selectedRepairReplaySlotId = languageFocus.selectedRepairReplaySlotId || null;
  const selectedRepairReplayPhase = languageFocus.selectedRepairReplayPhase || null;
  const comparePuzzleId = Object.prototype.hasOwnProperty.call(languageFocus || {}, 'comparePuzzleId')
    ? languageFocus.comparePuzzleId
    : getDefaultComparePuzzleId(PERSISTED_PUZZLE_RECORDS_V1, activePuzzleId);

  const currentLayerMeta = useMemo(
    () => RESEARCH_LAYERS.find((item) => item.id === researchLayer) || RESEARCH_LAYERS[0],
    [researchLayer]
  );

  const filteredPuzzleRecords = useMemo(() => {
    if (puzzleAxisFilter === 'all') return PERSISTED_PUZZLE_RECORDS_V1;
    return PERSISTED_PUZZLE_RECORDS_V1.filter((item) => item.priorityAxis === puzzleAxisFilter);
  }, [puzzleAxisFilter]);

  const activePuzzle =
    filteredPuzzleRecords.find((item) => item.id === activePuzzleId) ||
    PERSISTED_PUZZLE_RECORDS_V1.find((item) => item.id === activePuzzleId) ||
    filteredPuzzleRecords[0] ||
    PERSISTED_PUZZLE_RECORDS_V1[0] ||
    null;

  const fallbackCompareIdFromFiltered = getDefaultComparePuzzleId(filteredPuzzleRecords, activePuzzle?.id);
  const fallbackCompareIdFromAll = getDefaultComparePuzzleId(PERSISTED_PUZZLE_RECORDS_V1, activePuzzle?.id);

  const comparePuzzle =
    filteredPuzzleRecords.find((item) => item.id === comparePuzzleId && item.id !== activePuzzle?.id) ||
    PERSISTED_PUZZLE_RECORDS_V1.find((item) => item.id === comparePuzzleId && item.id !== activePuzzle?.id) ||
    (fallbackCompareIdFromFiltered
      ? filteredPuzzleRecords.find((item) => item.id === fallbackCompareIdFromFiltered)
      : null) ||
    (fallbackCompareIdFromAll
      ? PERSISTED_PUZZLE_RECORDS_V1.find((item) => item.id === fallbackCompareIdFromAll)
      : null) ||
    null;

  const compareVariableDiff = useMemo(() => {
    if (!activePuzzle || !comparePuzzle) {
      return {
        shared: [],
        primaryOnly: [],
        secondaryOnly: [],
      };
    }
    return buildVariableDiff(activePuzzle.mappedVariables, comparePuzzle.mappedVariables);
  }, [activePuzzle, comparePuzzle]);

  const variableCompareRows = useMemo(() => {
    if (!activePuzzle || !comparePuzzle || workspaceNodes.length === 0) {
      return [];
    }
    return buildVariableCompareRows(workspaceNodes, activePuzzle, comparePuzzle);
  }, [activePuzzle, comparePuzzle, workspaceNodes]);

  const repairContrastSummary = useMemo(
    () => buildRepairContrastSummary({
      activePuzzle,
      comparePuzzle,
      compareVariableDiff,
      variableCompareRows,
      compareValidation,
      compareSceneSummary,
    }),
    [activePuzzle, comparePuzzle, compareSceneSummary, compareValidation, compareVariableDiff, variableCompareRows]
  );
  const repairReplaySummary = useMemo(
    () => buildRepairReplaySummary(repairContrastSummary, sharedSubcircuitCandidates),
    [repairContrastSummary, sharedSubcircuitCandidates]
  );
  const repairReplaySlotSummary = useMemo(
    () => buildRepairReplaySlotSummary(
      repairContrastSummary,
      repairReplaySummary,
      PERSISTED_REPAIR_REPLAY_SAMPLE_SLOTS_V1
    ),
    [repairContrastSummary, repairReplaySummary]
  );
  const overviewSummary = useMemo(
    () => buildResearchOverviewSummary({
      workspaceSummary,
      currentLayerMeta,
      currentSceneLabel: structureTab || 'circuit',
      activePuzzle,
      conceptAssociationState,
      strongestConceptRelation,
      compareValidation,
      repairContrastSummary,
      repairReplaySlotSummary,
    }),
    [
      activePuzzle,
      compareValidation,
      conceptAssociationState,
      currentLayerMeta,
      repairContrastSummary,
      repairReplaySlotSummary,
      strongestConceptRelation,
      structureTab,
      workspaceSummary,
    ]
  );
  const propertySummaryCards = useMemo(
    () => buildPropertySummaryCards({
      activePuzzle,
      conceptAssociationState,
      compareValidation,
      repairReplaySlotSummary,
      variableCompareRows,
      compareVariableDiff,
    }),
    [
      activePuzzle,
      compareValidation,
      compareVariableDiff,
      conceptAssociationState,
      repairReplaySlotSummary,
      variableCompareRows,
    ]
  );
  const selectedRepairReplaySlot = useMemo(
    () => repairReplaySlotSummary?.slots.find((slot) => slot.slot_id === selectedRepairReplaySlotId) || null,
    [repairReplaySlotSummary, selectedRepairReplaySlotId]
  );
  const resolvedRepairReplayPhase = useMemo(
    () => selectedRepairReplaySlot?.phase_slots?.find((phase) => phase.phase === selectedRepairReplayPhase)?.phase
      || selectedRepairReplaySlot?.phase_slots?.find((phase) => phase.phase === 'bridge')?.phase
      || selectedRepairReplaySlot?.phase_slots?.[0]?.phase
      || null,
    [selectedRepairReplayPhase, selectedRepairReplaySlot]
  );
  const validationEntrySummary = useMemo(
    () => buildValidationEntrySummary({
      repairReplaySlotSummary,
      selectedRepairReplaySlot,
      resolvedRepairReplayPhase,
      compareValidation,
    }),
    [compareValidation, repairReplaySlotSummary, resolvedRepairReplayPhase, selectedRepairReplaySlot]
  );

  useEffect(() => {
    if (!selectedRepairReplaySlotId) {
      return;
    }
    const stillMatched = repairReplaySlotSummary?.slots.some((slot) => slot.slot_id === selectedRepairReplaySlotId);
    if (!stillMatched) {
      updateLanguageFocus({ selectedRepairReplaySlotId: null, selectedRepairReplayPhase: null });
    }
  }, [repairReplaySlotSummary, selectedRepairReplaySlotId]);

  useEffect(() => {
    if (!selectedRepairReplaySlotId) {
      return;
    }
    if (resolvedRepairReplayPhase && selectedRepairReplayPhase !== resolvedRepairReplayPhase) {
      updateLanguageFocus({ selectedRepairReplayPhase: resolvedRepairReplayPhase });
    }
  }, [resolvedRepairReplayPhase, selectedRepairReplayPhase, selectedRepairReplaySlotId]);

  useEffect(() => {
    if (!workspace) return;
    if (currentLayerMeta.animationMode && workspace.setAnimationMode) {
      workspace.setAnimationMode(currentLayerMeta.animationMode);
    }
  }, [currentLayerMeta, workspace]);

  const basicRuntimePlaying = Boolean(workspace?.basicRuntimePlaying);
  const basicRuntimeStep = workspace?.basicRuntimeStep ?? 1;
  const currentSceneLabel = structureTab || 'circuit';

  const handleOpenBasicInfo = () => {
    setLegacyOpen(true);
    updateLanguageFocus({ researchLayer: 'static_encoding' });
    if (typeof setStructureTab === 'function') {
      setStructureTab('circuit');
    }
  };

  const handleAxisFilterChange = (axisId) => {
    const nextRecords =
      axisId === 'all' ? PERSISTED_PUZZLE_RECORDS_V1 : PERSISTED_PUZZLE_RECORDS_V1.filter((item) => item.priorityAxis === axisId);
    const nextActiveId = nextRecords[0]?.id || null;
    const nextCompareId = getDefaultComparePuzzleId(nextRecords, nextActiveId);
    updateLanguageFocus({
      puzzleAxisFilter: axisId,
      activePuzzleId: nextActiveId,
      comparePuzzleId: nextCompareId,
      selectedRepairReplaySlotId: null,
      selectedRepairReplayPhase: null,
    });
  };

  const handlePuzzleSelect = (record) => {
    setLegacyOpen(record.layerKey === 'static_encoding');
    const nextCompareId =
      comparePuzzleId === record.id
        ? getDefaultComparePuzzleId(filteredPuzzleRecords, record.id) ||
          getDefaultComparePuzzleId(PERSISTED_PUZZLE_RECORDS_V1, record.id)
        : comparePuzzleId;
    updateLanguageFocus({
      researchLayer: record.layerKey || researchLayer,
      puzzleAxisFilter,
      activePuzzleId: record.id,
      comparePuzzleId: nextCompareId,
      selectedRepairReplaySlotId: null,
      selectedRepairReplayPhase: null,
    });
    if (typeof setStructureTab === 'function') {
      setStructureTab('circuit');
    }
  };

  const handleComparePuzzleSelect = (recordId) => {
    updateLanguageFocus({
      comparePuzzleId: recordId,
      selectedRepairReplaySlotId: null,
      selectedRepairReplayPhase: null,
    });
  };

  const handleClearCompare = () => {
    updateLanguageFocus({
      comparePuzzleId: null,
      selectedRepairReplaySlotId: null,
      selectedRepairReplayPhase: null,
    });
    if (typeof setStructureTab === 'function') {
      setStructureTab('circuit');
    }
  };

  const handleSelectRepairReplaySlot = (slot) => {
    if (!slot) {
      return;
    }
    const nextPhase = slot.phase_slots?.find((phase) => phase.phase === 'bridge')?.phase
      || slot.phase_slots?.[0]?.phase
      || null;
    updateLanguageFocus({
      selectedRepairReplaySlotId: slot.slot_id,
      selectedRepairReplayPhase: nextPhase,
      researchLayer: nextPhase === 'before' ? 'static_encoding' : nextPhase === 'after' ? 'result_recovery' : 'dynamic_route',
    });
    if (typeof setStructureTab === 'function') {
      setStructureTab('circuit');
    }
  };

  const handleSelectRepairReplayPhase = (phaseId) => {
    if (!phaseId) {
      return;
    }
    updateLanguageFocus({
      selectedRepairReplayPhase: phaseId,
      researchLayer: phaseId === 'before' ? 'static_encoding' : phaseId === 'after' ? 'result_recovery' : 'dynamic_route',
    });
    if (typeof setStructureTab === 'function') {
      setStructureTab('circuit');
    }
  };

  const handleClearRepairReplaySlot = () => {
    updateLanguageFocus({
      selectedRepairReplaySlotId: null,
      selectedRepairReplayPhase: null,
    });
  };

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      

      <div style={cardStyle}>
        <div style={sectionTitleStyle}>
          <Layers3 size={15} color="#8fd4ff" />
          <span>研究入口</span>
        </div>
        <div style={{ display: 'grid', gap: '8px' }}>
          <button
            type="button"
            onClick={handleOpenBasicInfo}
            style={{
              textAlign: 'left',
              padding: '10px 12px',
              borderRadius: '10px',
              border: legacyOpen ? '1px solid rgba(143, 212, 255, 0.55)' : '1px solid rgba(255,255,255,0.08)',
              background: legacyOpen ? 'rgba(143, 212, 255, 0.12)' : 'rgba(255,255,255,0.02)',
              color: legacyOpen ? '#eef7ff' : '#a8b7cc',
              cursor: 'pointer',
            }}
          >
            <div style={{ fontSize: '12px', fontWeight: 700, marginBottom: '3px' }}>基础编码</div>
            <div style={{ fontSize: '10px', lineHeight: 1.5, color: legacyOpen ? '#c8e6ff' : '#7e91ab' }}>
              打开基础信息、研究资产和 3D 映射入口。
            </div>
          </button>

          {RESEARCH_LAYERS.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => updateLanguageFocus({ researchLayer: item.id })}
              style={{
                textAlign: 'left',
                padding: '10px 12px',
                borderRadius: '10px',
                border: researchLayer === item.id ? '1px solid rgba(143, 212, 255, 0.55)' : '1px solid rgba(255,255,255,0.08)',
                background: researchLayer === item.id ? 'rgba(143, 212, 255, 0.12)' : 'rgba(255,255,255,0.02)',
                color: researchLayer === item.id ? '#eef7ff' : '#a8b7cc',
                cursor: 'pointer',
              }}
            >
              <div style={{ fontSize: '12px', fontWeight: 700, marginBottom: '3px' }}>{item.label}</div>
              <div style={{ fontSize: '10px', lineHeight: 1.5, color: researchLayer === item.id ? '#c8e6ff' : '#7e91ab' }}>
                {item.desc}
              </div>
            </button>
          ))}

          {/* Reverse Engineering Entry */}
          <button
            type="button"
            onClick={() => {
              setReverseEngineeringOpen(!reverseEngineeringOpen);
              if (workspace?.setAnalysisMode) {
                workspace.setAnalysisMode(reverseEngineeringOpen ? 'dynamic_prediction' : 'reverse_engineering');
              }
            }}
            style={{
              textAlign: 'left',
              padding: '10px 12px',
              borderRadius: '10px',
              border: reverseEngineeringOpen ? '1px solid rgba(236, 72, 153, 0.55)' : '1px solid rgba(255,255,255,0.08)',
              background: reverseEngineeringOpen ? 'rgba(236, 72, 153, 0.12)' : 'rgba(255,255,255,0.02)',
              color: reverseEngineeringOpen ? '#eef7ff' : '#a8b7cc',
              cursor: 'pointer',
            }}
          >
            <div style={{ fontSize: '12px', fontWeight: 700, marginBottom: '3px', color: reverseEngineeringOpen ? '#ec4899' : undefined }}>
              🔬 逆向工程
            </div>
            <div style={{ fontSize: '10px', lineHeight: 1.5, color: reverseEngineeringOpen ? '#f9a8d4' : '#7e91ab' }}>
              语言×DNN交叉逆向分析，验证编码证据链。
            </div>
          </button>
        </div>
      </div>

      <div style={cardStyle}>        
        {legacyOpen ? (
          <div style={{ marginTop: '10px' }}>
            <BasicEncodingPanel workspace={workspace} />
          </div>
        ) : null}
      </div>


      {repairReplaySlotSummary ? (
        <div style={cardStyle}>
          <div style={sectionTitleStyle}>
            <Route size={15} color="#f87171" />
            <span>样本回放</span>
          </div>

          <div style={{ ...summaryGridStyle, marginBottom: '10px' }}>
            <div style={summaryCardStyle}>
              <div style={summaryLabelStyle}>已匹配槽位</div>
              <div style={summaryValueStyle}>{repairReplaySlotSummary.slots.length}</div>
            </div>
            <div style={summaryCardStyle}>
              <div style={summaryLabelStyle}>精确匹配</div>
              <div style={summaryValueStyle}>{repairReplaySlotSummary.exactMatchCount}</div>
            </div>
            <div style={summaryCardStyle}>
              <div style={summaryLabelStyle}>平均就绪度</div>
              <div style={summaryValueStyle}>{`${Math.round(repairReplaySlotSummary.avgReadiness * 100)}%`}</div>
            </div>
          </div>

          <div style={{ display: 'grid', gap: '8px', maxHeight: '320px', overflowY: 'auto', paddingRight: '2px' }}>
            {repairReplaySlotSummary.slots.map((slot) => (
              <div
                key={slot.slot_id}
                style={{
                  ...cardStyle,
                  padding: '10px',
                  border: selectedRepairReplaySlotId === slot.slot_id
                    ? '1px solid rgba(143, 212, 255, 0.55)'
                    : '1px solid rgba(255,255,255,0.08)',
                  background: selectedRepairReplaySlotId === slot.slot_id
                    ? 'rgba(143, 212, 255, 0.1)'
                    : 'rgba(255,255,255,0.04)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', marginBottom: '6px' }}>
                  <div style={{ fontSize: '12px', fontWeight: 700, color: '#eef7ff' }}>{slot.label}</div>
                  <div style={{ fontSize: '10px', color: '#7f95bb' }}>
                    {`${slot.matchTypeLabel} | ${REPAIR_REPLAY_STATUS_LABEL_MAP[slot.status] || slot.status}`}
                  </div>
                </div>

                <div style={{ ...detailGridStyle, marginBottom: '8px' }}>
                  <DetailItem label="样本" value={slot.sample_label} />
                  <DetailItem label="锚定变量" value={slot.anchor_variable || '待定'} />
                  <DetailItem label="共享候选链" value={slot.shared_subcircuit_hint || '待补'} />
                  <DetailItem label="就绪度" value={`${Math.round(slot.readiness * 100)}%`} />
                </div>

                <div style={{ fontSize: '10px', color: '#9bb3de', lineHeight: 1.6, marginBottom: '6px' }}>
                  {slot.validation_goal}
                </div>

                {selectedRepairReplaySlotId === slot.slot_id && Array.isArray(slot.phase_slots) && slot.phase_slots.length ? (
                  <div style={{ display: 'flex', gap: '6px', marginTop: '8px', marginBottom: '8px', flexWrap: 'wrap' }}>
                    {slot.phase_slots.map((phase) => (
                      <button
                        key={`${slot.slot_id}-${phase.phase}`}
                        type="button"
                        onClick={() => handleSelectRepairReplayPhase(phase.phase)}
                        style={resolvedRepairReplayPhase === phase.phase ? actionButtonStyle : secondaryButtonStyle}
                      >
                        {phase.label}
                      </button>
                    ))}
                  </div>
                ) : null}

                <div style={{ display: 'flex', gap: '8px', marginTop: '8px', flexWrap: 'wrap' }}>
                  <button
                    type="button"
                    onClick={() => handleSelectRepairReplaySlot(slot)}
                    style={selectedRepairReplaySlotId === slot.slot_id ? actionButtonStyle : secondaryButtonStyle}
                  >
                    {selectedRepairReplaySlotId === slot.slot_id ? '已投到场景' : '投到场景'}
                  </button>
                  {selectedRepairReplaySlotId === slot.slot_id ? (
                    <button
                      type="button"
                      onClick={handleClearRepairReplaySlot}
                      style={secondaryButtonStyle}
                    >
                      取消聚焦
                    </button>
                  ) : null}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {(workspace?.handleBasicRuntimeStart || workspace?.handleBasicRuntimeStop || workspace?.handleBasicRuntimeReplay) ? (
        <div style={cardStyle}>
          <div style={sectionTitleStyle}>
            <Route size={15} color="#8fd4ff" />
            <span>动画控制</span>
          </div>
          <div style={{ color: '#9bb3de', fontSize: '11px', lineHeight: 1.6, marginBottom: '10px' }}>
            围绕当前层和当前拼图，控制逐层播放节奏。
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '8px' }}>
            <button type="button" onClick={() => workspace?.handleBasicRuntimeStart?.()} style={actionButtonStyle}>
              开始
            </button>
            <button type="button" onClick={() => workspace?.handleBasicRuntimeStop?.()} style={secondaryButtonStyle}>
              停止
            </button>
            <button type="button" onClick={() => workspace?.handleBasicRuntimeReplay?.()} style={secondaryButtonStyle}>
              重播
            </button>
          </div>
          <div style={{ fontSize: '10px', color: '#7f95bb', lineHeight: 1.6 }}>
            {`状态: ${basicRuntimePlaying ? '播放中' : '静止'} | 研究层 ${currentLayerMeta.label} | 步数: ${basicRuntimeStep}`}
          </div>
        </div>
      ) : null}

      {/* DNN分析控制面板 */}
      <div style={cardStyle}>
        <div style={sectionTitleStyle}>
          <Brain size={15} color="#8fd4ff" />
          <span>DNN分析</span>
        </div>
        <div style={{ color: '#9bb3de', fontSize: '11px', lineHeight: 1.6, marginBottom: '10px' }}>
          深度神经网络多维度分析工具，支持后续数据扩充。
        </div>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '8px' }}>
          <button
            type="button"
            onClick={() => setDnnAnalysisOpen(!dnnAnalysisOpen)}
            style={dnnAnalysisOpen ? actionButtonStyle : secondaryButtonStyle}
          >
            {dnnAnalysisOpen ? '收起面板' : '展开面板'}
          </button>
        </div>
      </div>

      {dnnAnalysisOpen && (
        <div style={cardStyle}>
          <DNNAnalysisControlPanel
            onAnalysisRequest={(data) => {
              console.log('DNN分析请求:', data);
            }}
            onDataSelect={(data) => {
              console.log('DNN数据选择:', data);
            }}
          />
        </div>
      )}

      {/* ==================== 逆向工程控制区 ==================== */}
      {reverseEngineeringOpen && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {/* Language Dimension Selector */}
          <div style={cardStyle}>
            <LanguageDimensionSelector
              selection={workspace?.reverseEngineeringState?.selectedLanguageDims || getDefaultLanguageDimSelection()}
              onSelectionChange={(newSelection) => {
                if (workspace?.setReverseEngineeringState) {
                  workspace.setReverseEngineeringState((prev) => ({
                    ...prev,
                    selectedLanguageDims: newSelection,
                  }));
                }
              }}
            />
          </div>

          {/* DNN Feature Tabs */}
          <div style={cardStyle}>
            <DNNFeatureTabs
              selectedCategory={workspace?.reverseEngineeringState?.selectedDNNCategory || 'activation'}
              selectedFeature={workspace?.reverseEngineeringState?.selectedDNNFeature || 'A3'}
              onCategoryChange={(cat) => {
                if (workspace?.setReverseEngineeringState) {
                  workspace.setReverseEngineeringState((prev) => ({
                    ...prev,
                    selectedDNNCategory: cat,
                  }));
                }
              }}
              onFeatureChange={(feat) => {
                if (workspace?.setReverseEngineeringState) {
                  workspace.setReverseEngineeringState((prev) => ({
                    ...prev,
                    selectedDNNFeature: feat,
                  }));
                }
              }}
            />
          </div>

          {/* View Mode Select */}
          <div style={cardStyle}>
            <ViewModeSelect
              viewMode={workspace?.reverseEngineeringState?.viewMode || 'structure'}
              onViewModeChange={(mode) => {
                if (workspace?.setReverseEngineeringState) {
                  workspace.setReverseEngineeringState((prev) => ({
                    ...prev,
                    viewMode: mode,
                  }));
                }
              }}
            />
          </div>

          {/* Quick Test Presets */}
          <div style={cardStyle}>
            <QuickTestPresets
              activePreset={workspace?.reverseEngineeringState?.activePreset}
              onApplyPreset={(preset) => {
                if (workspace?.setReverseEngineeringState) {
                  const currentSelection = workspace.reverseEngineeringState?.selectedLanguageDims || getDefaultLanguageDimSelection();
                  const newSelection = applyPresetToSelection(preset, currentSelection);
                  workspace.setReverseEngineeringState((prev) => ({
                    ...prev,
                    activePreset: preset.id,
                    selectedLanguageDims: newSelection,
                    selectedDNNFeature: preset.dnnFeature,
                    selectedDNNCategory: preset.dnnFeature?.[0]?.toLowerCase() === 'w' ? 'weight' : preset.dnnFeature?.[0]?.toLowerCase() === 'a' ? 'activation' : preset.dnnFeature?.[0]?.toLowerCase() === 'c' ? 'causal' : preset.dnnFeature?.[0]?.toLowerCase() === 'i' ? 'information' : 'dynamics',
                    viewMode: preset.viewMode,
                  }));
                }
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
