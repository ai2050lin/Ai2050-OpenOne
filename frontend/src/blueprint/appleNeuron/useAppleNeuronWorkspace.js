/**
 * AppleNeuron3D workspace hook
 * 从 AppleNeuron3DTab.jsx 拆分而来
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { AUDIT_3D_FOCUS_EVENT, readPersistedAudit3DFocus } from '../audit3dBridge';
import { LAYER_PARAMETER_STATE_ORDER, LAYER_PARAMETER_STATE_OVERLAY } from '../data/layer_parameter_state_overlay_persisted_v1';
import { PERSISTED_DATA_CATALOG_V1 } from '../data/persisted_data_catalog_v1';
import { PERSISTED_ENTITY_REGISTRY_V1 } from '../data/persisted_entity_registry_v1';
import { PERSISTED_MECHANISM_CHAIN_INDEX_V1 } from '../data/persisted_mechanism_chain_index_v1';
import { PERSISTED_PUZZLE_RECORDS_V1 } from '../data/persisted_puzzle_records_v1';
import { PERSISTED_REPAIR_REPLAY_SAMPLE_SLOTS_V1 } from '../data/persisted_repair_replay_sample_slots_v1';

import {
  LAYER_COUNT, DFF, IMPORTED_QUERY_NODE_MAX, MAIN_API_BASE,
  APPLE_CORE_NEURONS, FRUIT_GENERAL_NEURONS, FRUIT_SPECIFIC_NEURONS, FRUIT_COLORS,
  ROLE_COLORS, DIMENSION_LABELS,
  APPLE_SWITCH_MECHANISM_SCHEMA, APPLE_SWITCH_MODEL_COLORS, APPLE_SWITCH_ROLE_LABELS,
  DEFAULT_PREDICT_PROMPT, PREDICT_CHAIN_LENGTH,
  TOKEN_TRANSITIONS, TOPIC_FALLBACKS, DEFAULT_CHAIN_TOKENS,
  ANALYSIS_MODE_OPTIONS, APPLE_ANIMATION_OPTIONS,
  ICSPB_THEORY_OBJECTS, THEORY_OBJECT_MODE_MAP, FEATURE_AXES,
  DEFAULT_LANGUAGE_FOCUS, LANGUAGE_RESEARCH_LAYER_META,
  CONCEPT_ASSOCIATION_LAYER_META, CONCEPT_ALIAS_MAP,
  HARD_PROBLEM_EXPERIMENT_LABELS,
  MODE_VISUALS,
} from './constants';

import {
  toSafeNumber, hashString, pseudoRandom,
  neuronToPosition, averagePosition, blendPosition, shiftPosition,
  generatePredictChain,
  buildConceptNeuronSet, buildConceptNeuronSetFromSignature,
  buildSharedReuseSet,
  buildMultidimNodesFromProbe,
  buildHardProblemNodes, buildUnifiedDecodeNodes,
  buildAppleSwitchMechanismNodes, buildAppleSwitchMechanismLinks,
  isAppleSwitchMechanismPayload,
  buildArtifactPreview, shouldShowResearchAssetInTopRight,
  nodeDisplayGroup, isNodeVisibleByDisplayLevels,
  normalizePuzzleResearchLayer, buildPuzzleDisplayPreset,
  isNodeMatchedByPuzzle, findPuzzleSelectionCandidate,
  getPuzzleVariablePreferredRoles, getPuzzlePreferredRoles,
  buildPuzzleNodeEmphasisMap, buildPuzzleCompareState,
  buildPuzzleFocusNodeIdSet,
  normalizeReplaySlotHintRoles, getReplaySlotPhaseMeta,
  getReplayPhaseResearchLayer,
  buildRepairReplaySlotFocus,
  buildFamilyPatchViewModel,
  buildConceptAliasSet,
  buildConceptAssociationState,
  buildAnimationSceneProfile,
  nodeSignalStrength, normalizeConceptKey,
  buildAutoDisplayProfile,
  isHardProblemResultPayload, isUnifiedDecodePayload,
  isBundleManifestPayload, isFourTasksManifestPayload,
} from './utils';

function buildFruitSpecificNodes() {
  const nodes = [];
  Object.entries(FRUIT_SPECIFIC_NEURONS).forEach(([fruit, items], fruitIdx) => {
    items.forEach((item, idx) => {
      nodes.push({
        id: `fruit-${fruit}-l${item.layer}-n${item.neuron}`,
        label: `${fruit} specific ${idx + 1}`,
        role: 'fruitSpecific',
        fruit,
        layer: item.layer,
        neuron: item.neuron,
        metric: 'fruit_specific_score',
        value: item.score,
        strength: item.score / 3.2,
        source: 'multi_fruit_20260301_194541',
        color: FRUIT_COLORS[fruit],
        position: neuronToPosition(item.layer, item.neuron, 0.22 + idx * 0.04 + fruitIdx * 0.03),
        size: 0.12 + (item.score / 3.2) * 0.2,
        phase: fruitIdx * 0.6 + idx * 0.35,
      });
    });
  });
  return nodes;
}

function buildFruitGeneralNodes() {
  return FRUIT_GENERAL_NEURONS.map((item, idx) => ({
    id: `fruit-general-l${item.layer}-n${item.neuron}`,
    label: `Fruit General ${idx + 1}`,
    role: 'fruitGeneral',
    layer: item.layer,
    neuron: item.neuron,
    metric: 'fruit_general_score',
    value: item.score,
    strength: item.score / 3.1,
    source: 'multi_fruit_20260301_194541',
    color: ROLE_COLORS.fruitGeneral,
    position: neuronToPosition(item.layer, item.neuron, 0.12 + idx * 0.03),
    size: 0.12 + (item.score / 3.1) * 0.18,
    phase: idx * 0.42,
  }));
}

function buildAppleCoreNodes() {
  return APPLE_CORE_NEURONS.map((n, idx) => {
    const size = 0.16 + Math.sqrt(n.strength / 0.0012) * 0.22;
    return {
      ...n,
      nodeGroup: 'concept_core',
      color: ROLE_COLORS[n.role],
      position: neuronToPosition(n.layer, n.neuron, 0.15 + idx * 0.02),
      size,
      phase: idx * 0.9,
    };
  });
}

function buildBackgroundNodes() {
  return [];
}


export function useAppleNeuronWorkspace() {
  const [analysisMode, setAnalysisMode] = useState('dynamic_prediction');
  const [theoryObject, setTheoryObject] = useState('family_patch');
  const [animationMode, setAnimationMode] = useState('none');
  const [languageFocus, setLanguageFocus] = useState(DEFAULT_LANGUAGE_FOCUS);
  const [showFruitGeneral, setShowFruitGeneral] = useState(true);
  const [showFruit, setShowFruit] = useState(() => Object.fromEntries(Object.keys(FRUIT_COLORS).map((k) => [k, true])));
  const [queryInput, setQueryInput] = useState('');
  const [queryCategoryInput, setQueryCategoryInput] = useState('');
  const [querySets, setQuerySets] = useState([]);
  const [queryVisibility, setQueryVisibility] = useState({});
  const [queryFeedback, setQueryFeedback] = useState('');
  const [scanImportLimit, setScanImportLimit] = useState(20);
  const [scanImportTopK, setScanImportTopK] = useState(IMPORTED_QUERY_NODE_MAX);
  const [scanImportSummary, setScanImportSummary] = useState(null);
  const [selectedScanPath, setSelectedScanPath] = useState('');
  const [scanPreviewData, setScanPreviewData] = useState(null);
  const [scanPreviewLoading, setScanPreviewLoading] = useState(false);
  const [scanPreviewError, setScanPreviewError] = useState('');
  const [scanMechanismData, setScanMechanismData] = useState(null);
  const [appleSwitchMechanismData, setAppleSwitchMechanismData] = useState(null);
  const [multidimProbeData, setMultidimProbeData] = useState(null);
  const [multidimCausalData, setMultidimCausalData] = useState(null);
  const [hardProblemResults, setHardProblemResults] = useState({});
  const [unifiedDecodeResult, setUnifiedDecodeResult] = useState(null);
  const [bundleManifest, setBundleManifest] = useState(null);
  const [fourTasksManifest, setFourTasksManifest] = useState(null);
  const [multidimTopN, setMultidimTopN] = useState(96);
  const [multidimVisible, setMultidimVisible] = useState({ style: true, logic: true, syntax: true });
  const [multidimActiveDimension, setMultidimActiveDimension] = useState('style');
  const [predictPrompt, setPredictPrompt] = useState(DEFAULT_PREDICT_PROMPT);
  const [predictStep, setPredictStep] = useState(0);
  const [predictLayerProgress, setPredictLayerProgress] = useState(0);
  const [predictPlaying, setPredictPlaying] = useState(false);
  const [predictSpeed, setPredictSpeed] = useState(1);
  const [mechanismPlaying, setMechanismPlaying] = useState(false);
  const [mechanismSpeed, setMechanismSpeed] = useState(1);
  const [mechanismTick, setMechanismTick] = useState(0);
  const [interventionSparsity, setInterventionSparsity] = useState(0.45);
  const [featureAxis, setFeatureAxis] = useState(0);
  const [compositionWeights, setCompositionWeights] = useState({
    size: 0.34,
    sweetness: 0.33,
    color: 0.33,
  });
  const [counterfactualPrompt, setCounterfactualPrompt] = useState('');
  const [robustnessTrials, setRobustnessTrials] = useState(6);
  const [minimalSubsetSize, setMinimalSubsetSize] = useState(12);
  const [displayStrategy, setDisplayStrategy] = useState('auto');
  const [externalAuditFocus, setExternalAuditFocus] = useState(null);
  const [displayLevels, setDisplayLevels] = useState({
    basic_neurons: true,
    object_family: false,
    parameter_state: false,
    mechanism_chain: false,
    advanced_analysis: false,
  });
  const [showAlgorithmConceptCore, setShowAlgorithmConceptCore] = useState(false);
  const [showAlgorithmStaticEncoding, setShowAlgorithmStaticEncoding] = useState(false);
  const [showAlgorithmRuntimeChain, setShowAlgorithmRuntimeChain] = useState(false);
  const [manualDisplayGroups, setManualDisplayGroups] = useState({
    core: true,
    query: true,
    multidim: true,
    hard: true,
    unified: true,
    background: false,
  });
  const [basicRuntimePlaying, setBasicRuntimePlaying] = useState(false);
  const [basicRuntimeStep, setBasicRuntimeStep] = useState(1);
  const [layerSweepStep, setLayerSweepStep] = useState(0);

  // Reverse engineering state
  const [reverseEngineeringState, setReverseEngineeringState] = useState({
    selectedLanguageDims: {
      syntax: { S1: false, S2: false, S3: false, S4: false, S5: false, S6: false, S7: false, S8: false },
      semantic: { M1: false, M2: false, M3: false, M4: false, M5: false, M6: false, M7: false, M8: false },
      logic: { L1: false, L2: false, L3: false, L4: false, L5: false, L6: false, L7: false, L8: false },
      pragmatic: { P1: false, P2: false, P3: false, P4: false, P5: false, P6: false },
      morphological: { F1: false, F2: false, F3: false, F4: false, F5: false },
    },
    selectedDNNFeature: 'A3',
    selectedDNNCategory: 'activation',
    viewMode: 'structure',
    activePreset: null,
  });

  const backgroundNodes = useMemo(() => buildBackgroundNodes(), []);
  const appleCoreNodes = useMemo(() => buildAppleCoreNodes(), []);
  const fruitGeneralNodes = useMemo(() => buildFruitGeneralNodes(), []);
  const fruitSpecificNodes = useMemo(() => buildFruitSpecificNodes(), []);
  const queryNodes = useMemo(
    () => querySets
      .filter((set) => queryVisibility[set.id] !== false)
      .flatMap((set) => set.nodes),
    [querySets, queryVisibility]
  );
  const multidimNodes = useMemo(
    () => buildMultidimNodesFromProbe(multidimProbeData, multidimVisible, multidimTopN),
    [multidimProbeData, multidimVisible, multidimTopN]
  );
  const appleSwitchNodes = useMemo(
    () => buildAppleSwitchMechanismNodes(appleSwitchMechanismData),
    [appleSwitchMechanismData]
  );
  const hardProblemNodes = useMemo(() => buildHardProblemNodes(hardProblemResults), [hardProblemResults]);
  const unifiedDecodeNodes = useMemo(() => buildUnifiedDecodeNodes(unifiedDecodeResult), [unifiedDecodeResult]);
  const predictChain = useMemo(() => generatePredictChain(predictPrompt), [predictPrompt]);
  const dynamicEnabled = analysisMode === 'dynamic_prediction';
  const mechanismEnabled = !['static', 'dynamic_prediction'].includes(analysisMode);
  const theoryObjectMetaById = useMemo(
    () => Object.fromEntries(ICSPB_THEORY_OBJECTS.map((item) => [item.id, item])),
    []
  );
  const currentTheoryObject = theoryObjectMetaById[theoryObject] || ICSPB_THEORY_OBJECTS[0];
  const availableModesForTheoryObject = useMemo(
    () => THEORY_OBJECT_MODE_MAP[theoryObject] || THEORY_OBJECT_MODE_MAP.family_patch,
    [theoryObject]
  );

  useEffect(() => {
    setShowAlgorithmConceptCore(false);
    setShowAlgorithmStaticEncoding(false);
    setShowAlgorithmRuntimeChain(false);
    setDisplayLevels((prev) => ({
      ...prev,
      basic_neurons: true,
      object_family: false,
      parameter_state: false,
      mechanism_chain: false,
      advanced_analysis: false,
    }));
  }, []);

  const nodes = useMemo(() => {
    const objectFamilyVisible = displayLevels?.object_family !== false;
    const visibleFruitSpecific = objectFamilyVisible ? fruitSpecificNodes.filter((n) => showFruit[n.fruit]) : [];
    const visibleFruitGeneral = objectFamilyVisible && showFruitGeneral ? fruitGeneralNodes : [];
    const visibleConceptCore = showAlgorithmConceptCore ? appleCoreNodes : [];
    const visibleMultidim = showAlgorithmStaticEncoding ? multidimNodes : [];
    const visibleAppleSwitch = appleSwitchNodes;
    const visibleHardProblem = displayLevels?.advanced_analysis !== false ? hardProblemNodes : [];
    const visibleUnifiedDecode = displayLevels?.advanced_analysis !== false ? unifiedDecodeNodes : [];
    return [
      ...backgroundNodes,
      ...visibleConceptCore,
      ...visibleFruitGeneral,
      ...visibleFruitSpecific,
      ...queryNodes,
      ...visibleMultidim,
      ...visibleAppleSwitch,
      ...visibleHardProblem,
      ...visibleUnifiedDecode,
    ];
  }, [
    appleSwitchNodes,
    backgroundNodes,
    displayLevels?.object_family,
    fruitGeneralNodes,
    fruitSpecificNodes,
    displayLevels?.advanced_analysis,
    hardProblemNodes,
    queryNodes,
    showFruit,
    showFruitGeneral,
    showAlgorithmConceptCore,
    showAlgorithmStaticEncoding,
    unifiedDecodeNodes,
    appleCoreNodes,
    multidimNodes,
  ]);

  const keyNodes = useMemo(() => nodes.filter((n) => n.role !== 'background'), [nodes]);
  const [selected, setSelected] = useState(null);
  const activePuzzleRecord = useMemo(
    () => PERSISTED_PUZZLE_RECORDS_V1.find((item) => item.id === languageFocus?.activePuzzleId) || null,
    [languageFocus?.activePuzzleId]
  );
  const comparePuzzleRecord = useMemo(
    () => PERSISTED_PUZZLE_RECORDS_V1.find((item) => item.id === languageFocus?.comparePuzzleId) || null,
    [languageFocus?.comparePuzzleId]
  );
  const selectedRepairReplayPhase = languageFocus?.selectedRepairReplayPhase || null;
  const selectedRepairReplaySlot = useMemo(
    () => PERSISTED_REPAIR_REPLAY_SAMPLE_SLOTS_V1.find((item) => item.slot_id === languageFocus?.selectedRepairReplaySlotId) || null,
    [languageFocus?.selectedRepairReplaySlotId]
  );
  const lastAppliedPuzzleIdRef = useRef(null);
  const lastAppliedReplaySlotIdRef = useRef(null);

  useEffect(() => {
    const fallbackVisibleNode = nodes.find((node) => node.role !== 'background') || null;
    if (!selected) {
      if (fallbackVisibleNode) {
        setSelected(fallbackVisibleNode);
      }
      return;
    }
    const stillVisible = nodes.some((node) => node.id === selected.id);
    if (!stillVisible) {
      setSelected(fallbackVisibleNode);
    }
  }, [nodes, selected]);

  useEffect(() => {
    if (!activePuzzleRecord) {
      lastAppliedPuzzleIdRef.current = null;
      return;
    }

    const preset = buildPuzzleDisplayPreset(activePuzzleRecord);
    const puzzleChanged = lastAppliedPuzzleIdRef.current !== activePuzzleRecord.id;

    if (puzzleChanged) {
      setDisplayStrategy('auto');
      setPredictPlaying(false);
      setMechanismPlaying(false);
      setBasicRuntimePlaying(false);
      setBasicRuntimeStep(1);
      setDisplayLevels((prev) => ({ ...prev, ...preset.displayLevels }));
      setShowAlgorithmConceptCore(preset.showAlgorithmConceptCore);
      setShowAlgorithmStaticEncoding(preset.showAlgorithmStaticEncoding);
      setShowAlgorithmRuntimeChain(preset.displayLevels.mechanism_chain === true);
      setLanguageFocus((prev) => {
        const nextResearchLayer = normalizePuzzleResearchLayer(activePuzzleRecord.layerKey);
        if (prev?.researchLayer === nextResearchLayer) {
          return prev;
        }
        return { ...prev, researchLayer: nextResearchLayer };
      });
    }

    const selectedMatchesPuzzle = isNodeMatchedByPuzzle(selected, activePuzzleRecord);
    if (!selectedMatchesPuzzle) {
      const nextSelected = findPuzzleSelectionCandidate(nodes, activePuzzleRecord);
      if (nextSelected && nextSelected.id !== selected?.id) {
        setSelected(nextSelected);
      }
    }

    lastAppliedPuzzleIdRef.current = activePuzzleRecord.id;
  }, [
    activePuzzleRecord,
    nodes,
    selected,
    setLanguageFocus,
  ]);
  const puzzleNodeEmphasis = useMemo(
    () => buildPuzzleNodeEmphasisMap(nodes, activePuzzleRecord, selected?.id),
    [activePuzzleRecord, nodes, selected?.id]
  );
  const comparePuzzleNodeEmphasis = useMemo(
    () => buildPuzzleNodeEmphasisMap(nodes, comparePuzzleRecord, selected?.id),
    [comparePuzzleRecord, nodes, selected?.id]
  );
  const replaySlotNodeEmphasis = useMemo(() => {
    if (!selectedRepairReplaySlot) {
      return null;
    }
    const hintRoles = new Set(normalizeReplaySlotHintRoles(selectedRepairReplaySlot.shared_subcircuit_hint));
    getPuzzleVariablePreferredRoles([selectedRepairReplaySlot.anchor_variable]).forEach((role) => {
      hintRoles.add(role);
    });
    const activePhaseId = getReplaySlotPhaseMeta(selectedRepairReplaySlot, selectedRepairReplayPhase)?.phase || selectedRepairReplayPhase || 'bridge';
    if (!hintRoles.size) {
      return null;
    }
    const map = {};
    nodes.forEach((node) => {
      const base = hintRoles.has(node.role) ? 0.9 : 0.08;
      if (activePhaseId === 'before') {
        map[node.id] = node.role === 'micro' || node.role === 'fruitGeneral' || node.role === 'fruitSpecific'
          ? Math.max(base, 0.94)
          : base;
        return;
      }
      if (activePhaseId === 'after') {
        map[node.id] = node.role === 'unifiedDecode' || node.role === 'route'
          ? Math.max(base, 0.94)
          : base;
        return;
      }
      map[node.id] = base;
    });
    return map;
  }, [nodes, selectedRepairReplayPhase, selectedRepairReplaySlot]);
  const nodeDisplayEmphasis = useMemo(() => {
    const map = {};
    const autoProfile = buildAutoDisplayProfile(analysisMode);
    const theoryWeights = currentTheoryObject?.roleWeights || {};
    nodes.forEach((node) => {
      const group = nodeDisplayGroup(node.role);
      let emphasis = 1;
      if (displayStrategy === 'all') {
        emphasis = 1;
      } else if (displayStrategy === 'manual') {
        emphasis = manualDisplayGroups[group] === false ? 0.03 : 1;
      } else {
        emphasis = toSafeNumber(autoProfile[group], 0.8);
      }
      emphasis *= toSafeNumber(theoryWeights[node.role], toSafeNumber(theoryWeights[group], 0.72));
      if (puzzleNodeEmphasis?.[node.id] !== undefined) {
        emphasis *= toSafeNumber(puzzleNodeEmphasis[node.id], 1);
      }
      if (comparePuzzleNodeEmphasis?.[node.id] !== undefined) {
        emphasis = Math.max(emphasis, toSafeNumber(comparePuzzleNodeEmphasis[node.id], 1) * 0.74);
      }
      if (replaySlotNodeEmphasis?.[node.id] !== undefined) {
        emphasis = Math.max(emphasis, toSafeNumber(replaySlotNodeEmphasis[node.id], 1));
      }
      if (selected?.id === node.id) {
        emphasis = Math.max(emphasis, 0.95);
      }
      map[node.id] = Math.max(0, Math.min(1, emphasis));
    });
    return map;
  }, [analysisMode, comparePuzzleNodeEmphasis, currentTheoryObject, displayStrategy, manualDisplayGroups, nodes, puzzleNodeEmphasis, replaySlotNodeEmphasis, selected?.id]);

  useEffect(() => {
    if (analysisMode !== 'dynamic_prediction') {
      setPredictPlaying(false);
    }
    if (!mechanismEnabled) {
      setMechanismPlaying(false);
    }
  }, [analysisMode, mechanismEnabled]);

  useEffect(() => {
    if (!predictChain.length) {
      setPredictPlaying(false);
      return;
    }
    setPredictStep(0);
    setPredictLayerProgress(0);
  }, [predictChain]);

  useEffect(() => {
    if (!predictPlaying || !predictChain.length) {
      return undefined;
    }
    const interval = setInterval(() => {
      setPredictLayerProgress((prev) => {
        const next = prev + 0.038 * predictSpeed;
        if (next >= 1) {
          setPredictStep((s) => (s + 1) % predictChain.length);
          return 0;
        }
        return next;
      });
    }, 40);
    return () => clearInterval(interval);
  }, [predictPlaying, predictChain, predictSpeed]);

  useEffect(() => {
    if (!mechanismPlaying || !mechanismEnabled) {
      return undefined;
    }
    const interval = setInterval(() => {
      setMechanismTick((tick) => tick + 1);
    }, Math.max(30, 80 - mechanismSpeed * 18));
    return () => clearInterval(interval);
  }, [mechanismEnabled, mechanismPlaying, mechanismSpeed]);

  const basicRuntimeLayerKey = LAYER_PARAMETER_STATE_ORDER.includes(languageFocus?.researchLayer)
    ? languageFocus.researchLayer
    : 'static_encoding';
  const basicRuntimeProfile = LAYER_PARAMETER_STATE_OVERLAY[basicRuntimeLayerKey] || LAYER_PARAMETER_STATE_OVERLAY.static_encoding;

  useEffect(() => {
    setBasicRuntimePlaying(false);
    setBasicRuntimeStep(1);
  }, [basicRuntimeLayerKey, basicRuntimeProfile?.nodes?.length]);

  useEffect(() => {
    if (!basicRuntimePlaying) {
      return undefined;
    }
    const total = Array.isArray(basicRuntimeProfile?.nodes) ? basicRuntimeProfile.nodes.length : 0;
    if (total <= 0) {
      return undefined;
    }
    const timer = setInterval(() => {
      setBasicRuntimeStep((prev) => {
        const next = prev + 1;
        if (next > total) {
          return 1;
        }
        return next;
      });
    }, 850);
    return () => clearInterval(timer);
  }, [basicRuntimePlaying, basicRuntimeProfile]);

  useEffect(() => {
    if (basicRuntimePlaying || predictPlaying || mechanismPlaying) {
      return undefined;
    }
    const timer = setInterval(() => {
      setLayerSweepStep((prev) => (prev + 1) % LAYER_COUNT);
    }, 420);
    return () => clearInterval(timer);
  }, [basicRuntimePlaying, mechanismPlaying, predictPlaying]);

  const handleBasicRuntimeStart = () => {
    setBasicRuntimePlaying(true);
  };

  const handleBasicRuntimeStop = () => {
    setBasicRuntimePlaying(false);
  };

  const handleBasicRuntimeReplay = () => {
    setBasicRuntimeStep(1);
    setBasicRuntimePlaying(true);
  };

  useEffect(() => {
    setQueryVisibility((prev) => {
      const next = {};
      querySets.forEach((set) => {
        next[set.id] = prev[set.id] !== false;
      });
      return next;
    });
  }, [querySets]);

  useEffect(() => {
    const applyAuditFocus = (focus) => {
      if (!focus || typeof focus !== 'object') {
        return;
      }
      if (focus.theoryObject) {
        setTheoryObject(focus.theoryObject);
      }
      if (focus.analysisMode) {
        setAnalysisMode(focus.analysisMode);
      }
      if (focus.animationMode) {
        setAnimationMode(focus.animationMode);
      }
      setDisplayStrategy('auto');
      setPredictPlaying(false);
      setMechanismPlaying(false);
      setExternalAuditFocus(focus);
    };

    const persistedFocus = readPersistedAudit3DFocus();
    if (persistedFocus) {
      applyAuditFocus(persistedFocus);
    }

    const handleAuditFocus = (event) => {
      applyAuditFocus(event?.detail || null);
    };

    window.addEventListener(AUDIT_3D_FOCUS_EVENT, handleAuditFocus);
    return () => {
      window.removeEventListener(AUDIT_3D_FOCUS_EVENT, handleAuditFocus);
    };
  }, []);

  const handleGenerateQuery = () => {
    const concept = queryInput.trim();
    const category = queryCategoryInput.trim() || '未分类';
    if (!concept) {
      setQueryFeedback('请输入名称后再生成。');
      return;
    }
    setQuerySets((prev) => {
      const existing = prev.find((set) => set.normalized === concept.toLowerCase() && set.normalizedCategory === category.toLowerCase());
      if (existing) {
        setQueryVisibility((visibilityPrev) => ({ ...visibilityPrev, [existing.id]: true }));
        if (existing.nodes[0]) {
          setSelected(existing.nodes[0]);
        }
        setQueryFeedback(`已存在「${existing.name} [${existing.category}]」，已定位并显示。`);
        return prev;
      }
      const nextSet = buildConceptNeuronSet(concept, category, prev.length);
      if (nextSet.nodes[0]) {
        setSelected(nextSet.nodes[0]);
      }
      setQueryVisibility((visibilityPrev) => ({ ...visibilityPrev, [nextSet.id]: true }));
      setQueryFeedback(`已生成「${nextSet.name} [${nextSet.category}]」神经元集合。`);
      return [...prev, nextSet];
    });
    setQueryInput('');
  };

  const handleImportScanJsonText = (jsonText, sourceName = 'mass_noun_encoding_scan.json') => {
    let parsed;
    try {
      parsed = JSON.parse(jsonText);
    } catch (_err) {
      setQueryFeedback('JSON 解析失败，请确认文件格式正确。');
      return;
    }

    if (isAppleSwitchMechanismPayload(parsed)) {
      setAppleSwitchMechanismData(parsed);
      setDisplayLevels((prev) => ({
        ...prev,
        parameter_state: true,
        mechanism_chain: true,
      }));
      setShowAlgorithmConceptCore(true);
      setQueryFeedback(
        `已导入苹果切换机制资产：${sourceName}，Qwen3 核心单元=${parsed?.models?.qwen3?.core_units?.length || 0}，DeepSeek7B 核心单元=${parsed?.models?.deepseek7b?.core_units?.length || 0}。`
      );
      return;
    }

    if (isHardProblemResultPayload(parsed)) {
      const expId = parsed.experiment_id;
      const expLabel = HARD_PROBLEM_EXPERIMENT_LABELS[expId] || expId;
      setHardProblemResults((prev) => ({ ...prev, [expId]: parsed }));
      if (expId === 'minimal_causal_circuit_search_v1') {
        const targets = parsed?.metrics?.targets || {};
        const minimalByNoun = {};
        Object.entries(targets).forEach(([noun, row]) => {
          const runs = Array.isArray(row?.runs) ? row.runs : [];
          if (runs.length === 0) {
            return;
          }
          const pick = runs
            .slice()
            .sort((a, b) => toSafeNumber(b?.fidelity_ratio, 0) - toSafeNumber(a?.fidelity_ratio, 0))[0];
          const subset = Array.isArray(pick?.minimal_subset) ? pick.minimal_subset : [];
          if (subset.length === 0) {
            return;
          }
          const key = normalizeConceptKey(noun);
          if (!key) {
            return;
          }
          minimalByNoun[key] = {
            noun: key,
            subset_flat_indices: subset,
            subset_size: toSafeNumber(pick?.minimal_size, subset.length),
            recovery_ratio: toSafeNumber(pick?.fidelity_ratio, 0),
            subset_drop_seq_logprob: toSafeNumber(pick?.intervention_drop_after_remove_subset, 0),
          };
        });
        if (Object.keys(minimalByNoun).length > 0) {
          setScanMechanismData((prev) => ({
            dff: Math.max(1, toSafeNumber(prev?.dff, DFF)),
            minimalByNoun: { ...(prev?.minimalByNoun || {}), ...minimalByNoun },
            counterfactualByNoun: { ...(prev?.counterfactualByNoun || {}) },
          }));
        }
      }
      const metricKeys = Object.keys(parsed?.metrics || {});
      setQueryFeedback(`已导入硬伤实验：${expLabel}（${sourceName}），指标数=${metricKeys.length}。`);
      return;
    }

    if (isFourTasksManifestPayload(parsed)) {
      setFourTasksManifest(parsed);
      const allSuccess = Boolean(parsed?.all_success);
      setQueryFeedback(`已导入四任务清单：${allSuccess ? '全部成功' : '存在失败'}，任务数=${Object.keys(parsed?.return_codes || {}).length}。`);
      return;
    }

    if (isUnifiedDecodePayload(parsed)) {
      setUnifiedDecodeResult(parsed);
      const passRatio = toSafeNumber(parsed?.hypothesis_test?.pass_ratio, 0);
      setQueryFeedback(`已导入统一解码结果：${sourceName}，假设通过率=${(passRatio * 100).toFixed(1)}%。`);
      return;
    }

    if (isBundleManifestPayload(parsed)) {
      setBundleManifest(parsed);
      const snap = parsed?.metrics_snapshot || {};
      const dynamic = toSafeNumber(snap?.dynamic_binding?.binding_stability_index, 0);
      const longDecay = toSafeNumber(snap?.long_horizon?.long_horizon_decay, 0);
      const localSel = toSafeNumber(snap?.local_credit?.local_selectivity_mean, 0);
      setQueryFeedback(`已导入批量实验清单：动态稳定=${dynamic.toFixed(3)}，长程衰减=${longDecay.toFixed(3)}，局部选择性=${localSel.toFixed(3)}。`);
      return;
    }

    const hasMultidimProbe = Boolean(
      parsed?.dimensions?.style
      && parsed?.dimensions?.logic
      && parsed?.dimensions?.syntax
      && parsed?.cross_dimension
    );
    const hasMultidimCausal = Boolean(parsed?.suppression_matrix_mean && parsed?.diagonal_advantage);
    const hasMultidimStability = Boolean(parsed?.aggregate?.diag_adv_style && parsed?.aggregate?.specificity_margin_style);
    if (hasMultidimProbe) {
      setMultidimProbeData(parsed);
      const probeSummary = parsed?.runtime_config || {};
      setQueryFeedback(
        `已导入三维编码探针：source=${sourceName}，每维样本=${probeSummary.max_pairs_per_dim || '-'}，top_k=${probeSummary.top_k || '-'}。`
      );
      return;
    }
    if (hasMultidimCausal) {
      setMultidimCausalData(parsed);
      setQueryFeedback(`已导入三维因果消融：source=${sourceName}，top_n=${parsed?.top_n || '-'}。`);
      return;
    }
    if (hasMultidimStability) {
      const runs = toSafeNumber(parsed?.n_runs, 0);
      const style = toSafeNumber(parsed?.aggregate?.specificity_margin_style?.mean, 0);
      const logic = toSafeNumber(parsed?.aggregate?.specificity_margin_logic?.mean, 0);
      const syntax = toSafeNumber(parsed?.aggregate?.specificity_margin_syntax?.mean, 0);
      setQueryFeedback(`已导入三维多seed稳定性汇总：runs=${runs}，specificity(style/logic/syntax)=(${style.toFixed(3)}/${logic.toFixed(3)}/${syntax.toFixed(3)})。`);
      return;
    }

    const nounRecords = Array.isArray(parsed?.noun_records) ? parsed.noun_records : [];
    if (nounRecords.length === 0) {
      setQueryFeedback('未检测到 noun_records，无法导入。');
      return;
    }

    const dff = Math.max(1, toSafeNumber(parsed?.config?.d_ff, DFF));
    const limit = Math.max(1, Math.min(120, toSafeNumber(scanImportLimit, 20)));
    const perConceptTopK = Math.max(4, Math.min(64, toSafeNumber(scanImportTopK, IMPORTED_QUERY_NODE_MAX)));

    const validRecords = nounRecords.filter((rec) => Array.isArray(rec?.signature_top_indices) && rec.signature_top_indices.length > 0);
    if (validRecords.length === 0) {
      setQueryFeedback('该扫描结果没有 signature_top_indices，请先用新版脚本重新导出。');
      return;
    }

    const picked = validRecords.slice(0, limit);
    const importedSets = picked.map((rec, idx) => buildConceptNeuronSetFromSignature(
      String(rec?.noun || `concept_${idx + 1}`),
      String(rec?.category || '未分类'),
      rec?.signature_top_indices || [],
      idx,
      dff,
      perConceptTopK
    ));

    const reused = Array.isArray(parsed?.top_reused_neurons) ? parsed.top_reused_neurons : [];
    if (reused.length > 0) {
      importedSets.push(buildSharedReuseSet(reused, dff, perConceptTopK, picked.length));
    }

    const minimalRecords = Array.isArray(parsed?.causal_ablation?.minimal_circuit?.records)
      ? parsed.causal_ablation.minimal_circuit.records
      : [];
    const counterfactualRecords = Array.isArray(parsed?.causal_ablation?.counterfactual_validation?.records)
      ? parsed.causal_ablation.counterfactual_validation.records
      : [];
    const minimalByNoun = {};
    minimalRecords.forEach((rec) => {
      const key = normalizeConceptKey(rec?.noun);
      if (!key) {
        return;
      }
      minimalByNoun[key] = rec;
    });
    const counterfactualByNoun = {};
    counterfactualRecords.forEach((rec) => {
      const key = normalizeConceptKey(rec?.noun);
      if (!key) {
        return;
      }
      if (!counterfactualByNoun[key]) {
        counterfactualByNoun[key] = [];
      }
      counterfactualByNoun[key].push(rec);
    });
    setScanMechanismData({
      dff,
      minimalByNoun,
      counterfactualByNoun,
    });
    const firstCounterfactual = counterfactualRecords[0];
    if (firstCounterfactual?.counterfactual_noun) {
      setCounterfactualPrompt(String(firstCounterfactual.counterfactual_noun));
    }

    let added = 0;
    let updated = 0;
    setQuerySets((prev) => {
      const next = [...prev];
      importedSets.forEach((set) => {
        const existingIdx = next.findIndex(
          (item) => item.normalized === set.normalized && item.normalizedCategory === set.normalizedCategory
        );
        if (existingIdx >= 0) {
          next[existingIdx] = set;
          updated += 1;
        } else {
          next.push(set);
          added += 1;
        }
      });
      return next;
    });

    setQueryVisibility((prev) => {
      const next = { ...prev };
      importedSets.forEach((set) => {
        next[set.id] = true;
      });
      return next;
    });

    if (importedSets[0]?.nodes?.[0]) {
      setSelected(importedSets[0].nodes[0]);
    }

    const importedCategoryCount = new Set(importedSets.map((set) => set.category)).size;
    setScanImportSummary({
      source: sourceName,
      importedConcepts: importedSets.length,
      importedCategories: importedCategoryCount,
      totalNouns: nounRecords.length,
      minimalCircuitNouns: minimalRecords.length,
      counterfactualPairs: counterfactualRecords.length,
    });
    setQueryFeedback(`已导入扫描结果：新增 ${added}，更新 ${updated}，来源 ${sourceName}。`);
  };

  const removeQuerySet = (setId) => {
    setQuerySets((prev) => prev.filter((set) => set.id !== setId));
    setQueryVisibility((prev) => {
      const next = { ...prev };
      delete next[setId];
      return next;
    });
    setQueryFeedback('已移除该概念集合。');
  };

  const setQuerySetVisible = (setId, visible) => {
    setQueryVisibility((prev) => ({ ...prev, [setId]: visible }));
  };

  const setAllQuerySetVisible = (visible) => {
    setQueryVisibility((prev) => {
      const next = { ...prev };
      querySets.forEach((set) => {
        next[set.id] = visible;
      });
      return next;
    });
  };

  const appleSwitchLinks = useMemo(
    () => buildAppleSwitchMechanismLinks(appleSwitchMechanismData, appleSwitchNodes),
    [appleSwitchMechanismData, appleSwitchNodes]
  );

  const links = useMemo(() => {
    const byId = Object.fromEntries(keyNodes.map((n) => [n.id, n]));
    const linkSpecs = [];

    const fruitLinks = Object.keys(FRUIT_COLORS)
      .flatMap((fruit) => {
        if (!showFruit[fruit]) {
          return [];
        }
        const items = keyNodes.filter((n) => n.role === 'fruitSpecific' && n.fruit === fruit);
        if (items.length < 2) {
          return [];
        }
        return items.slice(1).map((node) => [items[0].id, node.id, FRUIT_COLORS[fruit]]);
      });

    const queryLinks = querySets.flatMap((set) => {
      if (set.nodes.length < 2) {
        return [];
      }
      return set.nodes.slice(1).map((node) => [set.nodes[0].id, node.id, set.color]);
    });

    const multidimLinks = ['style', 'logic', 'syntax'].flatMap((dim) => {
      if (multidimVisible[dim] === false) {
        return [];
      }
      const color = ROLE_COLORS[dim] || '#84f1ff';
      const group = keyNodes
        .filter((n) => n.role === dim)
        .sort((a, b) => Number(b.value || 0) - Number(a.value || 0))
        .slice(0, 16);
      if (group.length < 2) {
        return [];
      }
      return group.slice(1).map((node) => [group[0].id, node.id, color]);
    });

    const baseLinks = [...linkSpecs, ...fruitLinks, ...queryLinks, ...multidimLinks]
      .filter(([from, to]) => byId[from] && byId[to])
      .map(([from, to, color]) => ({
        id: `${from}->${to}`,
        from,
        to,
        color,
        points: [byId[from].position, byId[to].position],
      }));
    return baseLinks.concat(
      appleSwitchLinks.filter((link) => byId[link.from] && byId[link.to])
    );
  }, [appleSwitchLinks, keyNodes, multidimVisible, querySets, showFruit]);

  const puzzleCompareState = useMemo(
    () => buildPuzzleCompareState(
      nodes,
      links,
      activePuzzleRecord,
      comparePuzzleRecord,
      selectedRepairReplaySlot,
      selectedRepairReplayPhase
    ),
    [activePuzzleRecord, comparePuzzleRecord, links, nodes, selectedRepairReplayPhase, selectedRepairReplaySlot]
  );
  const conceptAssociationState = useMemo(
    () => buildConceptAssociationState(nodes, links, selected, languageFocus, scanMechanismData),
    [languageFocus, links, nodes, scanMechanismData, selected]
  );

  useEffect(() => {
    if (!selectedRepairReplaySlot || !puzzleCompareState?.replaySlotFocus) {
      lastAppliedReplaySlotIdRef.current = null;
      return;
    }
    const activePhaseId = puzzleCompareState.replaySlotFocus.activePhaseId || 'bridge';
    const slotFocusKey = `${selectedRepairReplaySlot.slot_id}:${activePhaseId}`;
    if (lastAppliedReplaySlotIdRef.current === slotFocusKey) {
      return;
    }

    setDisplayStrategy('auto');
    setDisplayLevels((prev) => ({
      ...prev,
      parameter_state: true,
      mechanism_chain: activePhaseId !== 'before',
      advanced_analysis: activePhaseId === 'after' ? prev.advanced_analysis : false,
    }));
    setShowAlgorithmRuntimeChain(activePhaseId !== 'before');
    setLanguageFocus((prev) => {
      const nextResearchLayer = getReplayPhaseResearchLayer(activePhaseId);
      if (prev?.researchLayer === nextResearchLayer) {
        return prev;
      }
      return { ...prev, researchLayer: nextResearchLayer };
    });

    const preferredNodeId = puzzleCompareState.replaySlotFocus.nodeIds?.[0] || null;
    const nextSelected = nodes.find((node) => node.id === preferredNodeId) || null;
    if (nextSelected && nextSelected.id !== selected?.id) {
      setSelected(nextSelected);
    }

    lastAppliedReplaySlotIdRef.current = slotFocusKey;
  }, [nodes, puzzleCompareState?.replaySlotFocus, selected?.id, selectedRepairReplaySlot, setLanguageFocus]);

  const currentPredictToken = dynamicEnabled && predictChain.length ? predictChain[predictStep % predictChain.length] : null;
  const predictLayer = predictLayerProgress * (LAYER_COUNT - 1);
  const mechanismPhase = (mechanismTick % 240) / 240;

  const dynamicActivationMap = useMemo(() => {
    if (!currentPredictToken) {
      return {};
    }
    const map = {};
    keyNodes.forEach((node) => {
      const seed = hashString(`${currentPredictToken.token}|${predictStep}|${node.id}`);
      const lexical = 0.25 + pseudoRandom(seed) * 0.75;
      const layerGate = Math.max(0, 1 - Math.abs(node.layer - predictLayer) / 8.2);
      const roleBoost = node.role === 'micro' ? 1.2 : node.role === 'macro' ? 1.08 : node.role === 'route' ? 1.15 : 1;
      map[node.id] = Math.min(1, lexical * layerGate * roleBoost * (0.65 + currentPredictToken.prob));
    });
    return map;
  }, [currentPredictToken, keyNodes, predictLayer, predictStep]);

  const modeOverlay = useMemo(() => {
    const overlay = {
      activationMap: {},
      currentToken: { token: '静态分析', prob: 0 },
      layerProgress: 0,
      focusNodeIds: [],
      effectiveLayer: null,
      effectiveNeurons: [],
      metrics: [],
      statusText: '',
    };

    if (!keyNodes.length) {
      overlay.metrics = [{ label: '节点', value: '0（请先生成或导入概念）' }];
    }
    const selectedConceptKey = normalizeConceptKey(selected?.concept);
    const importedDff = Math.max(1, toSafeNumber(scanMechanismData?.dff, DFF));
    const importedMinimal = selectedConceptKey ? scanMechanismData?.minimalByNoun?.[selectedConceptKey] : null;
    const importedCounterfactualList = selectedConceptKey ? (scanMechanismData?.counterfactualByNoun?.[selectedConceptKey] || []) : [];

    if (analysisMode === 'static') {
      keyNodes.forEach((node) => {
        overlay.activationMap[node.id] = Math.min(0.25, 0.06 + Math.sqrt(Math.max(node.strength, 1e-6)) * 0.5);
      });
      overlay.statusText = '结构分布快照';
      overlay.metrics = [{ label: '模式', value: '静态分析' }];
      return overlay;
    }

    if (analysisMode === 'dynamic_prediction') {
      overlay.activationMap = dynamicActivationMap;
      overlay.currentToken = currentPredictToken || { token: '-', prob: 0 };
      overlay.layerProgress = predictLayerProgress;
      overlay.statusText = 'Autoregressive decoding';
      overlay.metrics = [
        { label: 'Step', value: `${predictStep + 1}/${predictChain.length || 0}` },
        { label: 'Layer', value: `L${predictLayer.toFixed(1)}` },
      ];
      return overlay;
    }

    if (analysisMode === 'causal_intervention') {
      const scores = keyNodes.map((node) => {
        const roleBoost = node.role === 'route' ? 1.25 : node.role === 'macro' ? 1.15 : 1;
        const score = pseudoRandom(hashString(`causal|${predictPrompt}|${node.id}`)) * roleBoost;
        return { id: node.id, score };
      });
      scores.sort((a, b) => b.score - a.score);
      const topCount = Math.max(4, Math.floor(4 + interventionSparsity * 20));
      const focus = scores.slice(0, topCount);
      const focusIds = new Set(focus.map((v) => v.id));
      keyNodes.forEach((node) => {
        const item = scores.find((s) => s.id === node.id);
        overlay.activationMap[node.id] = focusIds.has(node.id) ? 0.55 + item.score * 0.45 : 0.02;
      });
      overlay.focusNodeIds = [...focusIds];
      overlay.currentToken = { token: 'do(intervene)', prob: Math.min(0.99, focus.reduce((a, b) => a + b.score, 0) / topCount) };
      overlay.layerProgress = mechanismPhase;
      overlay.statusText = 'Ablation + patching target set';
      overlay.metrics = [
        { label: 'Top Nodes', value: `${topCount}` },
        { label: 'Sparsity', value: interventionSparsity.toFixed(2) },
      ];
      return overlay;
    }

    if (analysisMode === 'subspace_geometry') {
      const a = pseudoRandom(hashString(`${predictPrompt}|subspace|a`)) * 2 - 1;
      const b = pseudoRandom(hashString(`${predictPrompt}|subspace|b`)) * 2 - 1;
      const c = pseudoRandom(hashString(`${predictPrompt}|subspace|c`)) * 2 - 1;
      keyNodes.forEach((node) => {
        const x = node.layer / (LAYER_COUNT - 1) - 0.5;
        const y = (node.neuron / DFF) * 2 - 1;
        const z = Math.sin((node.layer + 1) * 0.35 + (node.neuron % 97) * 0.02);
        const projection = Math.abs(a * x + b * y + c * z);
        overlay.activationMap[node.id] = Math.min(1, 0.15 + projection * 0.95);
      });
      overlay.currentToken = { token: 'subspace', prob: 0.72 };
      overlay.layerProgress = mechanismPhase;
      overlay.statusText = 'Direction / subspace encoding';
      overlay.metrics = [
        { label: 'Basis', value: `[${a.toFixed(2)}, ${b.toFixed(2)}, ${c.toFixed(2)}]` },
      ];
      return overlay;
    }

    if (analysisMode === 'feature_decomposition') {
      const axisName = FEATURE_AXES[featureAxis] || FEATURE_AXES[0];
      const currentLayer = Math.max(0, Math.min(LAYER_COUNT - 1, Math.round(mechanismPhase * (LAYER_COUNT - 1))));
      const layerEffective = [];
      keyNodes.forEach((node) => {
        const axis = hashString(`feature-axis|${node.id}`) % FEATURE_AXES.length;
        const local = pseudoRandom(hashString(`feature-val|${axisName}|${node.id}`));
        const score = axis === featureAxis ? 0.58 + local * 0.4 : 0.08 + local * 0.2;
        overlay.activationMap[node.id] = score;
        if (node.layer === currentLayer && node.role !== 'background') {
          layerEffective.push({ ...node, score: score * (axis === featureAxis ? 1.15 : 0.72) });
        }
      });
      layerEffective.sort((a, b) => b.score - a.score);
      const topLayerNodes = layerEffective.slice(0, 8);
      overlay.focusNodeIds = topLayerNodes.map((n) => n.id);
      overlay.effectiveLayer = currentLayer;
      overlay.effectiveNeurons = topLayerNodes.map((n) => ({
        id: n.id,
        label: n.label,
        role: n.role,
        layer: n.layer,
        neuron: n.neuron,
        score: n.score,
      }));
      overlay.currentToken = { token: `axis:${axisName}`, prob: 0.78 };
      overlay.layerProgress = mechanismPhase;
      overlay.statusText = `特征分解：定位 L${currentLayer} 有效神经元`;
      overlay.metrics = [
        { label: 'Axis', value: axisName },
        { label: 'Slots', value: `${FEATURE_AXES.length}` },
        { label: '当前层', value: `L${currentLayer}` },
        { label: '有效神经元', value: `${topLayerNodes.length}` },
      ];
      return overlay;
    }

    if (analysisMode === 'cross_layer_transport') {
      const currentLayer = mechanismPhase * (LAYER_COUNT - 1);
      keyNodes.forEach((node) => {
        const layerGate = Math.exp(-Math.abs(node.layer - currentLayer) / 3.4);
        const routeBoost = node.role === 'route' ? 1.2 : 1;
        const lexical = 0.45 + pseudoRandom(hashString(`transport|${node.id}|${Math.floor(currentLayer)}`)) * 0.55;
        overlay.activationMap[node.id] = Math.min(1, layerGate * lexical * routeBoost);
      });
      overlay.currentToken = { token: `transport@L${currentLayer.toFixed(1)}`, prob: 0.75 };
      overlay.layerProgress = mechanismPhase;
      overlay.statusText = 'Layer-wise representational flow';
      overlay.metrics = [{ label: 'Current Layer', value: currentLayer.toFixed(1) }];
      return overlay;
    }

    if (analysisMode === 'compositionality') {
      const total = compositionWeights.size + compositionWeights.sweetness + compositionWeights.color;
      const ws = {
        size: compositionWeights.size / total,
        sweetness: compositionWeights.sweetness / total,
        color: compositionWeights.color / total,
      };
      keyNodes.forEach((node) => {
        const sizeSig = pseudoRandom(hashString(`comp-size|${node.id}`));
        const sweetSig = pseudoRandom(hashString(`comp-sweet|${node.id}`));
        const colorSig = pseudoRandom(hashString(`comp-color|${node.id}`));
        overlay.activationMap[node.id] = Math.min(1, 0.08 + ws.size * sizeSig + ws.sweetness * sweetSig + ws.color * colorSig);
      });
      overlay.currentToken = { token: 'compose(size,sweet,color)', prob: 0.8 };
      overlay.layerProgress = mechanismPhase;
      overlay.statusText = 'Attribute composition';
      overlay.metrics = [
        { label: 'w(size)', value: ws.size.toFixed(2) },
        { label: 'w(sweet)', value: ws.sweetness.toFixed(2) },
        { label: 'w(color)', value: ws.color.toFixed(2) },
      ];
      return overlay;
    }

    if (analysisMode === 'counterfactual') {
      if (importedCounterfactualList.length > 0) {
        const preferred = importedCounterfactualList.find((r) => r?.relation === 'same_category') || importedCounterfactualList[0];
        const cfConcept = normalizeConceptKey(preferred?.counterfactual_noun);
        const focus = [];
        keyNodes.forEach((node) => {
          const nk = normalizeConceptKey(node.concept);
          if (nk === selectedConceptKey) {
            overlay.activationMap[node.id] = 0.88;
            focus.push(node.id);
            return;
          }
          if (cfConcept && nk === cfConcept) {
            overlay.activationMap[node.id] = 0.5;
            focus.push(node.id);
            return;
          }
          overlay.activationMap[node.id] = 0.02 + pseudoRandom(hashString(`cf-import-bg|${node.id}`)) * 0.08;
        });
        overlay.focusNodeIds = focus;
        overlay.currentToken = { token: `CF: ${preferred?.noun || '-'} -> ${preferred?.counterfactual_noun || '-'}`, prob: 0.76 };
        overlay.layerProgress = mechanismPhase;
        overlay.statusText = '反事实特异性（导入）';
        overlay.metrics = [
          { label: '关系', value: preferred?.relation === 'same_category' ? '同类反事实' : '跨类反事实' },
          { label: '特异性边际', value: `${toSafeNumber(preferred?.specificity_margin_seq_logprob, 0).toFixed(6)}` },
          { label: '子集大小', value: `${toSafeNumber(preferred?.subset_size, 0)}` },
        ];
        return overlay;
      }
      keyNodes.forEach((node) => {
        const base = pseudoRandom(hashString(`base|${predictPrompt}|${node.id}`));
        const cf = pseudoRandom(hashString(`cf|${counterfactualPrompt}|${node.id}`));
        overlay.activationMap[node.id] = Math.abs(base - cf);
      });
      overlay.currentToken = { token: 'counterfactual Δ', prob: 0.7 };
      overlay.layerProgress = mechanismPhase;
      overlay.statusText = 'Minimal semantic edit response';
      overlay.metrics = [
        { label: 'Base', value: predictPrompt.slice(0, 16) || '-' },
        { label: 'CF', value: counterfactualPrompt.slice(0, 16) || '-' },
      ];
      return overlay;
    }

    if (analysisMode === 'robustness') {
      const trials = Math.max(2, robustnessTrials);
      keyNodes.forEach((node) => {
        const values = [];
        for (let t = 0; t < trials; t += 1) {
          values.push(pseudoRandom(hashString(`robust|${t}|${node.id}`)));
        }
        const mean = values.reduce((a, b) => a + b, 0) / trials;
        const variance = values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / trials;
        const std = Math.sqrt(variance);
        const stability = Math.max(0, 1 - std * 3.6);
        overlay.activationMap[node.id] = 0.08 + stability * 0.92;
      });
      overlay.currentToken = { token: `robust@${trials}`, prob: 0.76 };
      overlay.layerProgress = mechanismPhase;
      overlay.statusText = 'Noise / paraphrase invariance';
      overlay.metrics = [{ label: 'Trials', value: `${trials}` }];
      return overlay;
    }

    if (analysisMode === 'minimal_circuit') {
      if (importedMinimal && Array.isArray(importedMinimal?.subset_flat_indices)) {
        const subset = new Set(importedMinimal.subset_flat_indices.map((v) => toSafeNumber(v, -1)).filter((v) => v >= 0));
        const focus = [];
        keyNodes.forEach((node) => {
          const flat = node.layer * importedDff + node.neuron;
          if (subset.has(flat)) {
            overlay.activationMap[node.id] = 0.92;
            focus.push(node.id);
          } else {
            overlay.activationMap[node.id] = 0.015;
          }
        });
        overlay.focusNodeIds = focus;
        const subsetSize = toSafeNumber(importedMinimal?.subset_size, subset.size);
        overlay.currentToken = { token: `MCS(import k=${subsetSize})`, prob: Math.min(0.99, toSafeNumber(importedMinimal?.recovery_ratio, 0)) };
        overlay.layerProgress = mechanismPhase;
        overlay.statusText = '最小因果子回路（导入）';
        overlay.metrics = [
          { label: '子集大小', value: `${subsetSize}` },
          { label: '恢复率', value: `${toSafeNumber(importedMinimal?.recovery_ratio, 0).toFixed(3)}` },
          { label: 'Seq Drop', value: `${toSafeNumber(importedMinimal?.subset_drop_seq_logprob, 0).toFixed(6)}` },
        ];
        return overlay;
      }
      const k = Math.max(3, Math.min(minimalSubsetSize, keyNodes.length));
      const scores = keyNodes
        .map((node) => ({ id: node.id, score: pseudoRandom(hashString(`mcs|${predictPrompt}|${node.id}`)) }))
        .sort((a, b) => b.score - a.score);
      const focusIds = new Set(scores.slice(0, k).map((v) => v.id));
      keyNodes.forEach((node) => {
        const s = scores.find((x) => x.id === node.id)?.score || 0;
        overlay.activationMap[node.id] = focusIds.has(node.id) ? 0.6 + s * 0.4 : 0.015;
      });
      overlay.focusNodeIds = [...focusIds];
      overlay.currentToken = { token: `MCS(k=${k})`, prob: Math.min(0.99, scores.slice(0, k).reduce((a, b) => a + b.score, 0) / k) };
      overlay.layerProgress = mechanismPhase;
      overlay.statusText = 'Minimal causal subset';
      overlay.metrics = [{ label: 'Subset Size', value: `${k}` }];
      return overlay;
    }

    return overlay;
  }, [
    analysisMode,
    compositionWeights.color,
    compositionWeights.size,
    compositionWeights.sweetness,
    counterfactualPrompt,
    currentPredictToken,
    dynamicActivationMap,
    featureAxis,
    interventionSparsity,
    keyNodes,
    mechanismPhase,
    minimalSubsetSize,
    predictChain.length,
    predictLayer,
    predictLayerProgress,
    predictPrompt,
    predictStep,
    robustnessTrials,
    scanMechanismData,
    selected,
  ]);

  useEffect(() => {
    const map = modeOverlay.activationMap || {};
    let bestNode = null;
    let bestScore = -1;
    keyNodes.forEach((node) => {
      const score = map[node.id] || 0;
      if (score > bestScore) {
        bestScore = score;
        bestNode = node;
      }
    });
    if (bestNode) {
      setSelected(bestNode);
    }
  }, [keyNodes, modeOverlay.activationMap]);

  const handlePredictReset = () => {
    setPredictPlaying(false);
    setPredictStep(0);
    setPredictLayerProgress(0);
  };

  const handlePredictStepForward = () => {
    if (!predictChain.length) {
      return;
    }
    setPredictPlaying(false);
    setPredictLayerProgress(0);
    setPredictStep((s) => (s + 1) % predictChain.length);
  };

  const handleMechanismReset = () => {
    setMechanismPlaying(false);
    setMechanismTick(0);
  };

  const handleMechanismStepForward = () => {
    setMechanismPlaying(false);
    setMechanismTick((t) => t + 18);
  };

  const multidimLayerProfile = useMemo(() => {
    const arr = multidimProbeData?.dimensions?.[multidimActiveDimension]?.layer_profile_abs_delta_norm;
    return Array.isArray(arr) ? arr : [];
  }, [multidimActiveDimension, multidimProbeData]);

  useEffect(() => {
    let cancelled = false;
    if (!selectedScanPath) {
      setScanPreviewData(null);
      setScanPreviewError('');
      setScanPreviewLoading(false);
      return undefined;
    }
    const loadPreview = async () => {
      setScanPreviewLoading(true);
      setScanPreviewError('');
      try {
        const res = await fetch(`${MAIN_API_BASE}/api/main/scan_file?path=${encodeURIComponent(selectedScanPath)}`);
        const payload = await res.json();
        if (!res.ok) {
          throw new Error(payload?.detail || '读取研究资产失败');
        }
        if (!cancelled) {
          setScanPreviewData(payload?.data || null);
        }
      } catch (err) {
        if (!cancelled) {
          setScanPreviewData(null);
          setScanPreviewError(`研究资产预览失败: ${err?.message || err}`);
        }
      } finally {
        if (!cancelled) {
          setScanPreviewLoading(false);
        }
      }
    };
    loadPreview();
    return () => {
      cancelled = true;
    };
  }, [selectedScanPath]);

  const summary = useMemo(() => {
    const fruitSpecific = keyNodes.filter((n) => n.role === 'fruitSpecific');
    const perFruit = Object.keys(FRUIT_COLORS).reduce((acc, fruit) => {
      acc[fruit] = fruitSpecific.filter((n) => n.fruit === fruit).length;
      return acc;
    }, {});
    const categoryStats = querySets.reduce((acc, set) => {
      const key = set.category || '未分类';
      if (!acc[key]) {
        acc[key] = { concepts: 0, neurons: 0 };
      }
      acc[key].concepts += 1;
      acc[key].neurons += set.nodes.length;
      return acc;
    }, {});

    return {
      micro: keyNodes.filter((n) => n.role === 'micro').length,
      macro: keyNodes.filter((n) => n.role === 'macro').length,
      route: keyNodes.filter((n) => n.role === 'route').length,
      fruitGeneral: keyNodes.filter((n) => n.role === 'fruitGeneral').length,
      fruitSpecific: fruitSpecific.length,
      query: keyNodes.filter((n) => n.role === 'query').length,
      hardProblemNodes: keyNodes.filter(
        (n) => n.role === 'hardBinding' || n.role === 'hardLong' || n.role === 'hardLocal' || n.role === 'hardTriplet'
      ).length,
      unifiedDecodeNodes: keyNodes.filter((n) => n.role === 'unifiedDecode').length,
      appleSwitchUnits: keyNodes.filter((n) => n.detailType === 'apple_switch_unit').length,
      total: keyNodes.length,
      perFruit,
      categoryStats,
      visibleQuerySets: querySets.filter((set) => queryVisibility[set.id] !== false).length,
      hiddenQuerySets: querySets.filter((set) => queryVisibility[set.id] === false).length,
      multidimNodes: keyNodes.filter((n) => n.role === 'style' || n.role === 'logic' || n.role === 'syntax').length,
      multidimActiveDimension,
      hardProblemCount: Object.keys(hardProblemResults || {}).length,
      unifiedDecodeLoaded: Boolean(unifiedDecodeResult),
      appleSwitchLoaded: Boolean(appleSwitchMechanismData),
      bundleLoaded: Boolean(bundleManifest),
      fourTasksLoaded: Boolean(fourTasksManifest),
      currentToken: modeOverlay.currentToken?.token || '-',
      currentTokenProb: modeOverlay.currentToken?.prob || 0,
      analysisMode,
      theoryObject,
      theoryObjectLabel: currentTheoryObject?.labelZh || '',
      theoryObjectDesc: currentTheoryObject?.desc || '',
      animationMode,
      displayStrategy,
      statusText: modeOverlay.statusText || '',
      externalAuditFocus,
    };
  }, [
    analysisMode,
    bundleManifest,
    currentTheoryObject,
    displayStrategy,
    externalAuditFocus,
    fourTasksManifest,
    hardProblemResults,
    keyNodes,
    modeOverlay.currentToken,
    modeOverlay.statusText,
    multidimActiveDimension,
    querySets,
    queryVisibility,
    theoryObject,
    animationMode,
    appleSwitchMechanismData,
    unifiedDecodeResult,
  ]);

  return {
    languageFocus,
    setLanguageFocus,
    analysisMode,
    setAnalysisMode,
    analysisModes: ANALYSIS_MODE_OPTIONS,
    animationMode,
    setAnimationMode,
    animationModes: APPLE_ANIMATION_OPTIONS,
    theoryObject,
    setTheoryObject,
    theoryObjects: ICSPB_THEORY_OBJECTS,
    currentTheoryObject,
    availableModesForTheoryObject,
    showFruitGeneral,
    setShowFruitGeneral,
    showFruit,
    setShowFruit,
    queryInput,
    setQueryInput,
    queryCategoryInput,
    setQueryCategoryInput,
    querySets,
    queryVisibility,
    queryFeedback,
    scanImportLimit,
    setScanImportLimit,
    scanImportTopK,
    setScanImportTopK,
    scanImportSummary,
    selectedScanPath,
    setSelectedScanPath,
    scanPreviewData,
    scanPreviewLoading,
    scanPreviewError,
    scanMechanismData,
    appleSwitchMechanismData,
    multidimProbeData,
    multidimCausalData,
    hardProblemResults,
    unifiedDecodeResult,
    bundleManifest,
    fourTasksManifest,
    multidimTopN,
    setMultidimTopN,
    multidimVisible,
    setMultidimVisible,
    multidimActiveDimension,
    setMultidimActiveDimension,
    multidimLayerProfile,
    handleGenerateQuery,
    handleImportScanJsonText,
    removeQuerySet,
    setQuerySetVisible,
    setAllQuerySetVisible,
    nodes,
    nodeDisplayEmphasis,
    puzzleCompareState,
    conceptAssociationState,
    links,
    selected,
    setSelected,
    summary,
    predictPrompt,
    setPredictPrompt,
    predictChain,
    predictStep,
    predictLayerProgress,
    predictPlaying,
    setPredictPlaying,
    predictSpeed,
    setPredictSpeed,
    handlePredictReset,
    handlePredictStepForward,
    mechanismPlaying,
    setMechanismPlaying,
    mechanismSpeed,
    setMechanismSpeed,
    mechanismTick,
    handleMechanismReset,
    handleMechanismStepForward,
    interventionSparsity,
    setInterventionSparsity,
    featureAxis,
    setFeatureAxis,
    compositionWeights,
    setCompositionWeights,
    counterfactualPrompt,
    setCounterfactualPrompt,
    robustnessTrials,
    setRobustnessTrials,
    minimalSubsetSize,
    setMinimalSubsetSize,
    externalAuditFocus,
    displayLevels,
    setDisplayLevels,
    showAlgorithmConceptCore,
    setShowAlgorithmConceptCore,
    showAlgorithmStaticEncoding,
    setShowAlgorithmStaticEncoding,
    showAlgorithmRuntimeChain,
    setShowAlgorithmRuntimeChain,
    displayStrategy,
    setDisplayStrategy,
    manualDisplayGroups,
    setManualDisplayGroups,
    basicRuntimePlaying,
    basicRuntimeStep,
    layerSweepStep,
    reverseEngineeringState,
    setReverseEngineeringState,
    handleBasicRuntimeStart,
    handleBasicRuntimeStop,
    handleBasicRuntimeReplay,
    modeMetrics: modeOverlay.metrics,
    prediction: analysisMode === 'static'
      ? null
      : {
          isRunning: dynamicEnabled ? predictPlaying : mechanismPlaying,
          currentToken: modeOverlay.currentToken,
          step: dynamicEnabled ? predictStep : mechanismTick,
          layerProgress: modeOverlay.layerProgress,
          activationMap: modeOverlay.activationMap,
          chain: dynamicEnabled ? predictChain : [],
          mode: analysisMode,
          metrics: modeOverlay.metrics,
          statusText: modeOverlay.statusText,
          focusNodeIds: modeOverlay.focusNodeIds,
          effectiveLayer: modeOverlay.effectiveLayer,
          effectiveNeurons: modeOverlay.effectiveNeurons,
        },
  };
}

