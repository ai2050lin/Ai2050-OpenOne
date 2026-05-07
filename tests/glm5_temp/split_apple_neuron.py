"""
Split AppleNeuron3DTab.jsx into multiple modules.
Reads the original file and creates the split files automatically.
"""
import re
import os

BASE_DIR = r'd:\ai2050\TransformerLens-Project\frontend\src\blueprint'
SRC_FILE = os.path.join(BASE_DIR, 'AppleNeuron3DTab.jsx')
OUT_DIR = os.path.join(BASE_DIR, 'appleNeuron')

with open(SRC_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

total = len(lines)
print(f'Total lines: {total}')

# ---- Section boundaries (1-indexed) ----
# Lines 1-13: imports (skip for utils/constants, keep for components)
# Lines 15-603: constants -> constants.js (already created manually)
# Lines 605-660: type check functions
# Lines 666-960: buildMetricRowsFromPaths, buildArtifactPreview
# Lines 961-995: shouldShowResearchAssetInTopRight, MODE_VISUALS
# Lines 997-1086: pseudoRandom, hashString, chain generators, buildConceptNeuronSet
# Lines 1088-2736: utility functions
# Lines 2738-3150: LayerEffectiveNeuronOverlay, LayerParameterStateOverlay
# Lines 3152-3656: ForwardPassLayerHighlight, LayerGuides, etc.
# Lines 3657-4155: TheoryObjectOverlay
# Lines 4157-4583: AppleNeuronSceneContent + AppleNeuronScene
# Lines 4585-4644: buildFruitSpecificNodes, buildBackgroundNodes
# Lines 4646-6097: useAppleNeuronWorkspace
# Lines 6098-6139: AppleNeuronMainScene
# Lines 6141-6183: style constants
# Lines 6184-7437: scan utilities + info panels
# Lines 7438-9363: DEAD CODE (block comment) - REMOVE
# Lines 9365-9386: AppleNeuron3DTab main entry

# ---- Build utils.js ----
# Lines 605-2667 (utility functions) 
# We need to adjust: some functions reference constants from the file header
# and some reference each other. We'll add imports from constants.js.

utils_lines = []

# Add header
utils_lines.append('/**\n')
utils_lines.append(' * AppleNeuron3D 工具函数\n')
utils_lines.append(' * 从 AppleNeuron3DTab.jsx 拆分而来\n')
utils_lines.append(' */\n\n')

# Add imports from constants
utils_lines.append('import {\n')
utils_lines.append('  LAYER_COUNT, DFF, IMPORTED_QUERY_NODE_MAX,\n')
utils_lines.append('  HARD_PROBLEM_EXPERIMENT_LABELS,\n')
utils_lines.append('  ROLE_COLORS, DIMENSION_LABELS,\n')
utils_lines.append('  APPLE_SWITCH_MECHANISM_SCHEMA, APPLE_SWITCH_MODEL_COLORS, APPLE_SWITCH_ROLE_LABELS,\n')
utils_lines.append('  TOKEN_TRANSITIONS, TOPIC_FALLBACKS, DEFAULT_CHAIN_TOKENS, PREDICT_CHAIN_LENGTH,\n')
utils_lines.append('  CONCEPT_ASSOCIATION_LAYER_META, CONCEPT_ALIAS_MAP,\n')
utils_lines.append('  APPLE_ANIMATION_OPTIONS,\n')
utils_lines.append('  DEFAULT_LANGUAGE_FOCUS,\n')
utils_lines.append('} from \'./constants\';\n\n')

# Add neuronToPosition (line 2660-2667) first since other utils depend on it
utils_lines.append('// ---- 3D position helpers ----\n\n')
for i in range(2659, 2667):  # 0-indexed: line 2660
    utils_lines.append(lines[i])

utils_lines.append('\n// ---- Core math helpers ----\n\n')
# Add averagePosition (line 3571-3584), blendPosition, shiftPosition, normalizeVector
for i in range(3570, 3601):  # lines 3571-3601
    utils_lines.append(lines[i])

utils_lines.append('\n// ---- Safe number & formatting ----\n\n')
# Lines 1088-1098: toSafeNumber, normalizeConceptKey, nodeSignalStrength
for i in range(1087, 1100):
    utils_lines.append(lines[i])

# Lines 1675-1710: clamp01, metricNodeStrength, extractMetricScalar, getMetricByPath
for i in range(1674, 1710):
    utils_lines.append(lines[i])

# Lines 632-664: formatPreviewValue, safeJsonStringify
utils_lines.append('\n// ---- Preview & formatting ----\n\n')
for i in range(631, 664):
    utils_lines.append(lines[i])

# Lines 997-1042: pseudoRandom, hashString, chain generators
utils_lines.append('\n// ---- Random & chain generation ----\n\n')
for i in range(996, 1042):
    utils_lines.append(lines[i])

# Lines 1044-1086: buildConceptNeuronSet
utils_lines.append('\n// ---- Concept set builders ----\n\n')
for i in range(1043, 1086):
    utils_lines.append(lines[i])

# Lines 1101-1220: buildFamilyPatchViewModel, buildConceptAliasSet, etc
utils_lines.append('\n// ---- Family patch & association ----\n\n')
for i in range(1100, 1220):
    utils_lines.append(lines[i])

# Lines 1222-1440: distanceBetweenPositions, pickConceptAssociationNodes, buildConceptAssociationState, buildNodeEmphasisMap
utils_lines.append('\n// ---- Association & emphasis ----\n\n')
for i in range(1221, 1440):
    utils_lines.append(lines[i])

# Lines 1442-1633: buildAnimationSceneProfile, buildConceptNeuronSetFromSignature, buildSharedReuseSet
utils_lines.append('\n// ---- Animation & import builders ----\n\n')
for i in range(1441, 1633):
    utils_lines.append(lines[i])

# Lines 1635-1673: buildMultidimNodesFromProbe
utils_lines.append('\n// ---- Multidim probe ----\n\n')
for i in range(1634, 1673):
    utils_lines.append(lines[i])

# Lines 1712-1862: buildHardProblemNodes, parseDominantLayers, buildUnifiedDecodeNodes
utils_lines.append('\n// ---- Hard problem & unified decode ----\n\n')
for i in range(1711, 1862):
    utils_lines.append(lines[i])

# Lines 1864-1987: Apple switch mechanism
utils_lines.append('\n// ---- Apple switch mechanism ----\n\n')
for i in range(1863, 1987):
    utils_lines.append(lines[i])

# Lines 1989-2269: nodeDisplayGroup, isNodeVisibleByDisplayLevels, puzzle functions
utils_lines.append('\n// ---- Display & puzzle helpers ----\n\n')
for i in range(1988, 2269):
    utils_lines.append(lines[i])

# Lines 2271-2657: replay/repair slot functions, buildPuzzleCompareState
utils_lines.append('\n// ---- Replay & compare ----\n\n')
for i in range(2270, 2659):
    utils_lines.append(lines[i])

utils_content = ''.join(utils_lines)

with open(os.path.join(OUT_DIR, 'utils.js'), 'w', encoding='utf-8') as f:
    f.write(utils_content)

print(f'Wrote utils.js: {len(utils_lines)} lines')

# ---- Build SceneComponents.jsx ----
scene_lines = []

scene_lines.append('/**\n')
scene_lines.append(' * AppleNeuron3D 场景组件\n')
scene_lines.append(' * 从 AppleNeuron3DTab.jsx 拆分而来\n')
scene_lines.append(' */\n\n')

scene_lines.append("import { Html, Line, OrbitControls, PerspectiveCamera, Text } from '@react-three/drei';\n")
scene_lines.append("import { Canvas, useFrame } from '@react-three/fiber';\n")
scene_lines.append("import { useMemo, useRef } from 'react';\n")
scene_lines.append("import { ANIMATION_SCENARIOS, DIMENSION_VIEWS } from '../../config/panels';\n")
scene_lines.append("import { LAYER_PARAMETER_STATE_ORDER, LAYER_PARAMETER_STATE_OVERLAY } from '../data/layer_parameter_state_overlay_persisted_v1';\n\n")
scene_lines.append("import {\n")
scene_lines.append("  LAYER_COUNT, DFF, ROLE_COLORS, DIMENSION_LABELS,\n")
scene_lines.append("  MODE_VISUALS, APPLE_ANIMATION_OPTIONS,\n")
scene_lines.append("  DEFAULT_LANGUAGE_FOCUS,\n")
scene_lines.append("} from './constants';\n\n")
scene_lines.append("import {\n")
scene_lines.append("  toSafeNumber, neuronToPosition,\n")
scene_lines.append("  averagePosition, blendPosition, shiftPosition, normalizeVector,\n")
scene_lines.append("} from './utils';\n\n")

# Lines 2669-2736: PulsingNeuron
for i in range(2668, 2736):
    scene_lines.append(lines[i])

scene_lines.append('\n\n')

# Lines 2738-3149: LayerEffectiveNeuronOverlay, LayerParameterStateOverlay
for i in range(2737, 3149):
    scene_lines.append(lines[i])

scene_lines.append('\n\n')

# Lines 3152-3655: ForwardPassLayerHighlight, LayerGuides, DimensionLayerImpactGraph, 
# TokenPredictionCarrier, ModeVisualOverlay, TheoryBeacon, TheoryRunner
for i in range(3151, 3655):
    scene_lines.append(lines[i])

scene_lines.append('\n\n')

# Lines 3657-4155: TheoryObjectOverlay
for i in range(3656, 4155):
    scene_lines.append(lines[i])

scene_lines.append('\n\n')

# Lines 4157-4583: AppleNeuronSceneContent + AppleNeuronScene
for i in range(4156, 4583):
    scene_lines.append(lines[i])

scene_lines.append('\n\n')

# Lines 6098-6139: AppleNeuronMainScene (exported)
# This needs imports from above
scene_lines.append('// ---- Main scene wrapper ----\n\n')
for i in range(6097, 6139):
    scene_lines.append(lines[i])

scene_content = ''.join(scene_lines)

with open(os.path.join(OUT_DIR, 'SceneComponents.jsx'), 'w', encoding='utf-8') as f:
    f.write(scene_content)

print(f'Wrote SceneComponents.jsx: {len(scene_lines)} lines')

# ---- Build useAppleNeuronWorkspace.js ----
workspace_lines = []

workspace_lines.append('/**\n')
workspace_lines.append(' * AppleNeuron3D workspace hook\n')
workspace_lines.append(' * 从 AppleNeuron3DTab.jsx 拆分而来\n')
workspace_lines.append(' */\n\n')

workspace_lines.append("import { useEffect, useMemo, useRef, useState } from 'react';\n")
workspace_lines.append("import { AUDIT_3D_FOCUS_EVENT, readPersistedAudit3DFocus } from '../audit3dBridge';\n")
workspace_lines.append("import { LAYER_PARAMETER_STATE_ORDER, LAYER_PARAMETER_STATE_OVERLAY } from '../data/layer_parameter_state_overlay_persisted_v1';\n")
workspace_lines.append("import { PERSISTED_DATA_CATALOG_V1 } from '../data/persisted_data_catalog_v1';\n")
workspace_lines.append("import { PERSISTED_ENTITY_REGISTRY_V1 } from '../data/persisted_entity_registry_v1';\n")
workspace_lines.append("import { PERSISTED_MECHANISM_CHAIN_INDEX_V1 } from '../data/persisted_mechanism_chain_index_v1';\n")
workspace_lines.append("import { PERSISTED_PUZZLE_RECORDS_V1 } from '../data/persisted_puzzle_records_v1';\n")
workspace_lines.append("import { PERSISTED_REPAIR_REPLAY_SAMPLE_SLOTS_V1 } from '../data/persisted_repair_replay_sample_slots_v1';\n\n")
workspace_lines.append("import {\n")
workspace_lines.append("  LAYER_COUNT, DFF, IMPORTED_QUERY_NODE_MAX, MAIN_API_BASE,\n")
workspace_lines.append("  APPLE_CORE_NEURONS, FRUIT_GENERAL_NEURONS, FRUIT_SPECIFIC_NEURONS, FRUIT_COLORS,\n")
workspace_lines.append("  ROLE_COLORS, DIMENSION_LABELS,\n")
workspace_lines.append("  APPLE_SWITCH_MECHANISM_SCHEMA, APPLE_SWITCH_MODEL_COLORS, APPLE_SWITCH_ROLE_LABELS,\n")
workspace_lines.append("  DEFAULT_PREDICT_PROMPT, PREDICT_CHAIN_LENGTH,\n")
workspace_lines.append("  TOKEN_TRANSITIONS, TOPIC_FALLBACKS, DEFAULT_CHAIN_TOKENS,\n")
workspace_lines.append("  ANALYSIS_MODE_OPTIONS, APPLE_ANIMATION_OPTIONS,\n")
workspace_lines.append("  ICSPB_THEORY_OBJECTS, THEORY_OBJECT_MODE_MAP, FEATURE_AXES,\n")
workspace_lines.append("  DEFAULT_LANGUAGE_FOCUS, LANGUAGE_RESEARCH_LAYER_META,\n")
workspace_lines.append("  CONCEPT_ASSOCIATION_LAYER_META, CONCEPT_ALIAS_MAP,\n")
workspace_lines.append("  HARD_PROBLEM_EXPERIMENT_LABELS,\n")
workspace_lines.append("  MODE_VISUALS,\n")
workspace_lines.append("} from './constants';\n\n")
workspace_lines.append("import {\n")
workspace_lines.append("  toSafeNumber, hashString, pseudoRandom,\n")
workspace_lines.append("  neuronToPosition, averagePosition, blendPosition, shiftPosition,\n")
workspace_lines.append("  generatePredictChain,\n")
workspace_lines.append("  buildConceptNeuronSet, buildConceptNeuronSetFromSignature,\n")
workspace_lines.append("  buildSharedReuseSet,\n")
workspace_lines.append("  buildMultidimNodesFromProbe,\n")
workspace_lines.append("  buildHardProblemNodes, buildUnifiedDecodeNodes,\n")
workspace_lines.append("  buildAppleSwitchMechanismNodes, buildAppleSwitchMechanismLinks,\n")
workspace_lines.append("  isAppleSwitchMechanismPayload,\n")
workspace_lines.append("  buildArtifactPreview, shouldShowResearchAssetInTopRight,\n")
workspace_lines.append("  nodeDisplayGroup, isNodeVisibleByDisplayLevels,\n")
workspace_lines.append("  normalizePuzzleResearchLayer, buildPuzzleDisplayPreset,\n")
workspace_lines.append("  isNodeMatchedByPuzzle, findPuzzleSelectionCandidate,\n")
workspace_lines.append("  getPuzzleVariablePreferredRoles, getPuzzlePreferredRoles,\n")
workspace_lines.append("  buildPuzzleNodeEmphasisMap, buildPuzzleCompareState,\n")
workspace_lines.append("  buildPuzzleFocusNodeIdSet,\n")
workspace_lines.append("  normalizeReplaySlotHintRoles, getReplaySlotPhaseMeta,\n")
workspace_lines.append("  getReplayPhaseResearchLayer,\n")
workspace_lines.append("  buildRepairReplaySlotFocus,\n")
workspace_lines.append("  buildFamilyPatchViewModel,\n")
workspace_lines.append("  buildConceptAliasSet,\n")
workspace_lines.append("  buildConceptAssociationState,\n")
workspace_lines.append("  buildAnimationSceneProfile,\n")
workspace_lines.append("  nodeSignalStrength, normalizeConceptKey,\n")
workspace_lines.append("  buildAutoDisplayProfile,\n")
workspace_lines.append("  isHardProblemResultPayload, isUnifiedDecodePayload,\n")
workspace_lines.append("  isBundleManifestPayload, isFourTasksManifestPayload,\n")
workspace_lines.append("} from './utils';\n\n")

# Lines 4585-4644: buildFruitSpecificNodes, buildBackgroundNodes, buildAppleCoreNodes, buildFruitGeneralNodes
# These are internal helpers for the workspace hook
for i in range(4584, 4644):
    workspace_lines.append(lines[i])

workspace_lines.append('\n\n')

# Lines 4646-6097: useAppleNeuronWorkspace
for i in range(4645, 6097):
    workspace_lines.append(lines[i])

workspace_content = ''.join(workspace_lines)

with open(os.path.join(OUT_DIR, 'useAppleNeuronWorkspace.js'), 'w', encoding='utf-8') as f:
    f.write(workspace_content)

print(f'Wrote useAppleNeuronWorkspace.js: {len(workspace_lines)} lines')

# ---- Build InfoPanels.jsx ----
info_lines = []

info_lines.append('/**\n')
info_lines.append(' * AppleNeuron3D 信息面板组件\n')
info_lines.append(' * 从 AppleNeuron3DTab.jsx 拆分而来\n')
info_lines.append(' */\n\n')

info_lines.append("import { Html } from '@react-three/drei';\n")
info_lines.append("import { useFrame } from '@react-three/fiber';\n")
info_lines.append("import { useMemo, useRef, useState } from 'react';\n\n")
info_lines.append("import {\n")
info_lines.append("  LAYER_COUNT, DFF, IMPORTED_QUERY_NODE_MAX,\n")
info_lines.append("  ROLE_COLORS, DIMENSION_LABELS,\n")
info_lines.append("  FRUIT_COLORS,\n")
info_lines.append("  HARD_PROBLEM_EXPERIMENT_LABELS,\n")
info_lines.append("  THEORY_OBJECT_RESEARCH_MAP,\n")
info_lines.append("  DEFAULT_LANGUAGE_FOCUS,\n")
info_lines.append("} from './constants';\n\n")
info_lines.append("import {\n")
info_lines.append("  toSafeNumber, formatPreviewValue,\n")
info_lines.append("  shouldShowResearchAssetInTopRight,\n")
info_lines.append("  buildAutoDisplayProfile,\n")
info_lines.append("  nodeDisplayGroup,\n")
info_lines.append("} from './utils';\n\n")

# Style constants (lines 6141-6183)
for i in range(6140, 6183):
    info_lines.append(lines[i])

info_lines.append('\n\n')

# Scan utilities (lines 6184-6480)
for i in range(6183, 6480):
    info_lines.append(lines[i])

info_lines.append('\n\n')

# Info panels (lines 6481-7437)
for i in range(6480, 7437):
    info_lines.append(lines[i])

info_content = ''.join(info_lines)

with open(os.path.join(OUT_DIR, 'InfoPanels.jsx'), 'w', encoding='utf-8') as f:
    f.write(info_content)

print(f'Wrote InfoPanels.jsx: {len(info_lines)} lines')

print('\nDone! Now update AppleNeuron3DTab.jsx and bridge files.')
