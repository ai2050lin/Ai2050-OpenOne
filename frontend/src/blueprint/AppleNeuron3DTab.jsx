import { Html, Line, OrbitControls, PerspectiveCamera, Text } from '@react-three/drei';
import { Canvas, useFrame } from '@react-three/fiber';
import { Activity, ArrowRightLeft, BarChart2, Brain, CheckCircle, GitBranch, Network, Scale, Search, Sparkles, Target } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { ANIMATION_SCENARIOS, DIMENSION_VIEWS } from '../config/panels';
import { AUDIT_3D_FOCUS_EVENT, readPersistedAudit3DFocus } from './audit3dBridge';
import LanguageResearchControlPanel from '../components/LanguageResearchControlPanel';
import { LAYER_PARAMETER_STATE_ORDER, LAYER_PARAMETER_STATE_OVERLAY } from './data/layer_parameter_state_overlay_persisted_v1';
import { PERSISTED_DATA_CATALOG_V1 } from './data/persisted_data_catalog_v1';
import { PERSISTED_ENTITY_REGISTRY_V1 } from './data/persisted_entity_registry_v1';
import { PERSISTED_MECHANISM_CHAIN_INDEX_V1 } from './data/persisted_mechanism_chain_index_v1';
import { PERSISTED_PUZZLE_RECORDS_V1 } from './data/persisted_puzzle_records_v1';
import { PERSISTED_REPAIR_REPLAY_SAMPLE_SLOTS_V1 } from './data/persisted_repair_replay_sample_slots_v1';

const LAYER_COUNT = 28;
const DFF = 18944;
const QUERY_NODE_COUNT = 12;
const IMPORTED_QUERY_NODE_MAX = 18;
const MAIN_API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

const APPLE_CORE_NEURONS = [
  { id: 'apple-core-l3-n412', label: '苹果概念核 1', role: 'micro', layer: 3, neuron: 412, metric: 'apple_core_strength', value: 0.83, strength: 0.00086, source: 'seed_layer_neuron_map_v1' },
  { id: 'apple-core-l6-n1284', label: '苹果概念核 2', role: 'micro', layer: 6, neuron: 1284, metric: 'apple_core_strength', value: 0.91, strength: 0.00102, source: 'seed_layer_neuron_map_v1' },
  { id: 'apple-core-l9-n2301', label: '苹果概念核 3', role: 'macro', layer: 9, neuron: 2301, metric: 'apple_core_strength', value: 0.88, strength: 0.00095, source: 'seed_layer_neuron_map_v1' },
  { id: 'apple-core-l12-n3610', label: '苹果概念核 4', role: 'macro', layer: 12, neuron: 3610, metric: 'apple_core_strength', value: 0.79, strength: 0.00073, source: 'seed_layer_neuron_map_v1' },
  { id: 'apple-core-l16-n5180', label: '苹果概念核 5', role: 'route', layer: 16, neuron: 5180, metric: 'apple_route_strength', value: 0.74, strength: 0.00069, source: 'seed_layer_neuron_map_v1' },
  { id: 'apple-core-l21-n7724', label: '苹果概念核 6', role: 'route', layer: 21, neuron: 7724, metric: 'apple_route_strength', value: 0.67, strength: 0.00058, source: 'seed_layer_neuron_map_v1' },
  { id: 'apple-core-l4-n684', label: '苹果概念核 7', role: 'micro', layer: 4, neuron: 684, metric: 'apple_core_strength', value: 0.8, strength: 0.0008, source: 'seed_layer_neuron_map_v2' },
  { id: 'apple-core-l7-n1712', label: '苹果概念核 8', role: 'micro', layer: 7, neuron: 1712, metric: 'apple_core_strength', value: 0.86, strength: 0.0009, source: 'seed_layer_neuron_map_v2' },
  { id: 'apple-core-l10-n2496', label: '苹果概念核 9', role: 'macro', layer: 10, neuron: 2496, metric: 'apple_core_strength', value: 0.84, strength: 0.00087, source: 'seed_layer_neuron_map_v2' },
  { id: 'apple-core-l14-n4288', label: '苹果概念核 10', role: 'macro', layer: 14, neuron: 4288, metric: 'apple_core_strength', value: 0.77, strength: 0.00072, source: 'seed_layer_neuron_map_v2' },
  { id: 'apple-core-l18-n5984', label: '苹果概念核 11', role: 'route', layer: 18, neuron: 5984, metric: 'apple_route_strength', value: 0.71, strength: 0.00064, source: 'seed_layer_neuron_map_v2' },
  { id: 'apple-core-l24-n9120', label: '苹果概念核 12', role: 'route', layer: 24, neuron: 9120, metric: 'apple_route_strength', value: 0.63, strength: 0.00052, source: 'seed_layer_neuron_map_v2' },
];

const FRUIT_GENERAL_NEURONS = [
  { layer: 4, neuron: 902, score: 2.86 },
  { layer: 8, neuron: 2148, score: 2.61 },
  { layer: 13, neuron: 4970, score: 2.48 },
  { layer: 19, neuron: 6804, score: 2.22 },
  { layer: 6, neuron: 1510, score: 2.73 },
  { layer: 11, neuron: 3416, score: 2.57 },
  { layer: 15, neuron: 5522, score: 2.39 },
  { layer: 22, neuron: 8014, score: 2.11 },
];

const FRUIT_SPECIFIC_NEURONS = {
  apple: [
    { layer: 5, neuron: 1180, score: 2.91 },
    { layer: 11, neuron: 3026, score: 2.74 },
    { layer: 18, neuron: 6212, score: 2.33 },
    { layer: 23, neuron: 8564, score: 2.02 },
  ],
  pear: [
    { layer: 5, neuron: 1194, score: 2.55 },
    { layer: 10, neuron: 2876, score: 2.41 },
    { layer: 17, neuron: 6021, score: 2.08 },
    { layer: 22, neuron: 8442, score: 1.96 },
  ],
  banana: [
    { layer: 6, neuron: 1433, score: 2.47 },
    { layer: 12, neuron: 3348, score: 2.28 },
    { layer: 20, neuron: 7118, score: 2.02 },
    { layer: 25, neuron: 9306, score: 1.88 },
  ],
  orange: [
    { layer: 6, neuron: 1468, score: 2.44 },
    { layer: 12, neuron: 3384, score: 2.21 },
    { layer: 19, neuron: 7036, score: 1.97 },
    { layer: 24, neuron: 9012, score: 1.82 },
  ],
  grape: [
    { layer: 7, neuron: 1688, score: 2.32 },
    { layer: 13, neuron: 3520, score: 2.14 },
    { layer: 18, neuron: 6408, score: 1.95 },
    { layer: 23, neuron: 8726, score: 1.79 },
  ],
  peach: [
    { layer: 6, neuron: 1544, score: 2.36 },
    { layer: 12, neuron: 3452, score: 2.17 },
    { layer: 19, neuron: 6892, score: 1.93 },
    { layer: 25, neuron: 9184, score: 1.76 },
  ],
  mango: [
    { layer: 8, neuron: 2016, score: 2.41 },
    { layer: 14, neuron: 4098, score: 2.23 },
    { layer: 20, neuron: 7342, score: 1.98 },
    { layer: 26, neuron: 9788, score: 1.83 },
  ],
};

const FRUIT_COLORS = {
  apple: '#fb7185',
  pear: '#facc15',
  banana: '#fde047',
  orange: '#fb923c',
  grape: '#c084fc',
  peach: '#f9a8d4',
  mango: '#f59e0b',
};

const ROLE_COLORS = {
  micro: '#ff8d3b',
  macro: '#f6d365',
  route: '#39d0ff',
  fruitGeneral: '#6cf7d4',
  style: '#7dd3fc',
  logic: '#fca5a5',
  syntax: '#a7f3d0',
  hardBinding: '#fb7185',
  hardLong: '#38bdf8',
  hardLocal: '#f59e0b',
  hardTriplet: '#a78bfa',
  unifiedDecode: '#22c55e',
  background: '#ffffff',
};

const DIMENSION_LABELS = {
  style: '风格维度',
  logic: '逻辑维度',
  syntax: '句法维度',
};

const APPLE_SWITCH_MECHANISM_SCHEMA = 'apple_switch_mechanism_view.v1';
const APPLE_SWITCH_MODEL_COLORS = {
  qwen3: '#60a5fa',
  deepseek7b: '#34d399',
};

const APPLE_SWITCH_ROLE_LABELS = {
  anchor_neuron: '锚点神经元',
  main_booster_1: '主增强头 1',
  main_booster_2: '主增强头 2',
  skeleton_head_1: '骨架头 1',
  skeleton_head_2: '骨架头 2',
  bridge_head: '桥接头',
  heldout_booster: '校正/补强头',
};

const DEFAULT_PREDICT_PROMPT = '';
const PREDICT_CHAIN_LENGTH = 10;

const TOKEN_TRANSITIONS = {
  概念: ['是', '一种', '结构', '系统', '表达'],
  模型: ['通过', '层级', '编码', '形成', '预测'],
  concept: ['is', 'a', 'structured', 'representation', 'in'],
  model: ['builds', 'multi-layer', 'features', 'for', 'prediction'],
  is: ['a', 'structured', 'mapping', 'inside', 'the'],
  a: ['concept', 'model', 'token', 'signal', 'pattern'],
};

const TOPIC_FALLBACKS = [
  {
    keywords: ['概念', 'concept'],
    tokens: ['是', '一种', '结构', '可以', '在', '层级', '传播'],
  },
  {
    keywords: ['模型', 'model'],
    tokens: ['通过', '多层', '机制', '进行', '编码', '并', '预测'],
  },
];

const DEFAULT_CHAIN_TOKENS = ['is', 'a', 'concept', 'mapped', 'through', 'layers', 'into', 'next', 'token'];

const ANALYSIS_MODE_OPTIONS = [
  { id: 'static', label: '静态分析', desc: '结构分布观察' },
  { id: 'dynamic_prediction', label: '动态预测', desc: 'next-token 动画' },
  { id: 'causal_intervention', label: '因果干预', desc: '必要/充分性打靶' },
  { id: 'subspace_geometry', label: '子空间编码', desc: '方向与子空间表示' },
  { id: 'feature_decomposition', label: '特征分解', desc: '特征簇与可解释轴' },
  { id: 'cross_layer_transport', label: '跨层传输', desc: '层间编码迁移' },
  { id: 'compositionality', label: '组合性测试', desc: '属性组合编码' },
  { id: 'counterfactual', label: '反事实编码', desc: '最小语义改动差分' },
  { id: 'robustness', label: '鲁棒不变性', desc: '扰动下稳定编码' },
  { id: 'minimal_circuit', label: '最小子回路', desc: '最小因果子集' },
  { id: 'reverse_engineering', label: '逆向工程', desc: '语言×DNN交叉逆向分析' },
];


const ANALYSIS_MODE_ICONS = {
  static: Search,
  dynamic_prediction: Sparkles,
  causal_intervention: Target,
  subspace_geometry: Scale,
  feature_decomposition: BarChart2,
  cross_layer_transport: ArrowRightLeft,
  compositionality: Activity,
  counterfactual: GitBranch,
  robustness: CheckCircle,
  minimal_circuit: Network,
  reverse_engineering: Brain,
};

const ANALYSIS_MODE_STAGE_GROUPS = [
  {
    id: 'observation',
    label: '观测',
    icon: Search,
    items: ['static', 'dynamic_prediction', 'cross_layer_transport'],
  },
  {
    id: 'extraction',
    label: '提取',
    icon: Sparkles,
    items: ['subspace_geometry', 'feature_decomposition', 'compositionality'],
  },
  {
    id: 'validation',
    label: '验证',
    icon: CheckCircle,
    items: ['causal_intervention', 'counterfactual', 'robustness'],
  },
  {
    id: 'system',
    label: '系统',
    icon: Network,
    items: ['minimal_circuit'],
  },
  {
    id: 'reverse',
    label: '逆向',
    icon: Brain,
    items: ['reverse_engineering'],
  },
];

const APPLE_ANIMATION_OPTIONS = [
  { id: 'none', label: '无动画', desc: '只看静态结构。' },
  { id: 'family_patch_formation', label: 'Family 成形', desc: '看 family patch 从散点收拢成原型核。' },
  { id: 'instance_offset', label: '实例偏移', desc: '看实例如何从 family 核拉出 offset。' },
  { id: 'attribute_fiber', label: '属性纤维', desc: '看颜色/形状/甜度纤维挂接到概念。' },
  { id: 'successor_transport', label: '后继运输', desc: '看 successor 沿路径运输。' },
  { id: 'protocol_bridge', label: '协议桥接', desc: '看内部编码如何进入读出桥。' },
  { id: 'cross_layer_relay', label: '跨层接力', desc: '看层间 relay 的亮起顺序。' },
  { id: 'ablation_shockwave', label: '消融冲击波', desc: '看打掉局部 witness 后的震荡外扩。' },
  { id: 'counterfactual_split', label: '反事实分叉', desc: '看原轨迹与反事实轨迹分叉。' },
  { id: 'minimal_circuit_peeloff', label: '最小回路剥离', desc: '看回路逐步剥离到最小集合。' },
  { id: 'margin_breathing', label: '边界呼吸', desc: '看 family margin 的呼吸式边界变化。' },
  { id: 'offset_sparsity', label: '偏移稀疏', desc: '看 offset 只点亮少量高权重维。' },
  { id: 'prototype_instance_tug', label: '原型-实例拉扯', desc: '看 prototype 与 instance 两股力的拉扯。' },
  { id: 'stage_transition', label: '阶段切换', desc: '看 observation -> extraction -> validation 的切换。' },
];

const ICSPB_THEORY_OBJECTS = [
  {
    id: 'family_patch',
    label: 'family patch',
    labelZh: '族群底座',
    desc: '看同一概念族是否形成稳定 patch 底座与共享群落。',
    color: '#7dd3fc',
    roleWeights: { macro: 1, fruitGeneral: 0.95, query: 0.82, micro: 0.72, route: 0.68, style: 0.35, logic: 0.35, syntax: 0.35, unifiedDecode: 0.38, hardBinding: 0.28, hardLong: 0.25, hardLocal: 0.25, hardTriplet: 0.25, background: 0.06 },
  },
  {
    id: 'concept_section',
    label: 'concept section',
    labelZh: '概念截面',
    desc: '看概念截面、局部偏移与最小语义改动是否保持局部连续。',
    color: '#c084fc',
    roleWeights: { micro: 1, query: 0.94, macro: 0.82, route: 0.72, fruitGeneral: 0.58, style: 0.52, logic: 0.52, syntax: 0.52, unifiedDecode: 0.4, hardBinding: 0.3, hardLong: 0.3, hardLocal: 0.28, hardTriplet: 0.3, background: 0.06 },
  },
  {
    id: 'attribute_fiber',
    label: 'attribute fiber',
    labelZh: '属性纤维',
    desc: '看颜色、形状、甜度等属性是否沿可组合纤维方向分离。',
    color: '#34d399',
    roleWeights: { style: 1, logic: 0.82, syntax: 0.7, micro: 0.8, query: 0.68, macro: 0.6, route: 0.56, unifiedDecode: 0.56, hardBinding: 0.25, hardLong: 0.24, hardLocal: 0.24, hardTriplet: 0.26, background: 0.05 },
  },
  {
    id: 'relation_context_fiber',
    label: 'relation-context fiber',
    labelZh: '关系/语境纤维',
    desc: '看关系和语境如何沿层间路径传播并重组。',
    color: '#22d3ee',
    roleWeights: { route: 1, query: 0.88, macro: 0.76, micro: 0.64, logic: 0.62, syntax: 0.58, style: 0.42, unifiedDecode: 0.54, hardLong: 0.42, hardTriplet: 0.48, hardBinding: 0.3, hardLocal: 0.3, background: 0.05 },
  },
  {
    id: 'admissible_update',
    label: 'admissible update',
    labelZh: '可容许更新',
    desc: '看什么样的局部改动既能更新知识，又不冲垮旧结构。',
    color: '#a3e635',
    roleWeights: { hardLocal: 1, hardBinding: 0.92, hardLong: 0.84, hardTriplet: 0.84, unifiedDecode: 0.72, route: 0.66, micro: 0.62, macro: 0.58, query: 0.52, style: 0.5, logic: 0.5, syntax: 0.5, background: 0.04 },
  },
  {
    id: 'restricted_readout',
    label: 'restricted readout',
    labelZh: '受限读出',
    desc: '看输出是否主要依赖少数关键节点、局部子回路和读出热点。',
    color: '#fb7185',
    roleWeights: { route: 1, micro: 0.92, macro: 0.84, query: 0.82, hardTriplet: 0.74, hardBinding: 0.62, unifiedDecode: 0.58, logic: 0.46, syntax: 0.46, style: 0.38, background: 0.04 },
  },
  {
    id: 'stage_conditioned_transport',
    label: 'stage-conditioned transport',
    labelZh: '阶段条件运输',
    desc: '看不同计算阶段是否切换不同的运输主路和层间热点。',
    color: '#38bdf8',
    roleWeights: { route: 1, macro: 0.86, micro: 0.78, query: 0.74, hardLong: 0.62, hardBinding: 0.46, unifiedDecode: 0.48, style: 0.34, logic: 0.34, syntax: 0.34, background: 0.04 },
  },
  {
    id: 'successor_aligned_transport',
    label: 'successor-aligned transport',
    labelZh: '后继对齐运输',
    desc: '看后继 token/状态是否沿稳定对齐路径产生，而不是随机跳变。',
    color: '#f59e0b',
    roleWeights: { route: 1, query: 0.9, micro: 0.74, macro: 0.7, hardLong: 0.68, hardTriplet: 0.52, unifiedDecode: 0.5, style: 0.32, logic: 0.38, syntax: 0.38, background: 0.04 },
  },
  {
    id: 'protocol_bridge',
    label: 'protocol bridge',
    labelZh: '协议桥',
    desc: '看内部编码如何进入任务接口、统一解码和最小可用闭环。',
    color: '#f97316',
    roleWeights: { unifiedDecode: 1, hardTriplet: 0.86, route: 0.84, query: 0.78, macro: 0.68, micro: 0.58, style: 0.54, logic: 0.54, syntax: 0.54, hardBinding: 0.48, hardLong: 0.48, hardLocal: 0.48, background: 0.04 },
  },
];

const THEORY_OBJECT_MODE_MAP = {
  family_patch: ['static', 'subspace_geometry', 'feature_decomposition'],
  concept_section: ['static', 'subspace_geometry', 'feature_decomposition', 'counterfactual'],
  attribute_fiber: ['subspace_geometry', 'feature_decomposition', 'compositionality'],
  relation_context_fiber: ['dynamic_prediction', 'cross_layer_transport', 'counterfactual', 'compositionality'],
  admissible_update: ['causal_intervention', 'robustness', 'minimal_circuit'],
  restricted_readout: ['dynamic_prediction', 'causal_intervention', 'minimal_circuit'],
  stage_conditioned_transport: ['dynamic_prediction', 'cross_layer_transport'],
  successor_aligned_transport: ['dynamic_prediction', 'counterfactual', 'causal_intervention'],
  protocol_bridge: ['cross_layer_transport', 'minimal_circuit', 'robustness'],
};
const FEATURE_AXES = ['color', 'taste', 'shape', 'category'];

const DEFAULT_LANGUAGE_FOCUS = {
  researchLayer: 'static_encoding',
  objectGroup: 'fruit',
  taskGroup: 'translation',
  roleGroup: 'object',
  structureOverlays: [],
  modelKey: 'gpt2',
  stageKey: 'stage260',
  compareMode: 'single_model',
  riskFocus: 'fidelity',
  selectedRepairReplaySlotId: null,
  selectedRepairReplayPhase: null,
};

const LANGUAGE_RESEARCH_LAYER_META = {
  static_encoding: { label: '静态编码层', color: '#8fd4ff' },
  dynamic_route: { label: '动态路径层', color: '#5eead4' },
  result_recovery: { label: '结果回收层', color: '#fbbf24' },
  propagation_encoding: { label: '传播编码层', color: '#f87171' },
  semantic_roles: { label: '语义角色层', color: '#c084fc' },
};

const CONCEPT_ASSOCIATION_LAYER_META = [
  { id: 'basic_encoding', label: '基础编码', color: '#93c5fd', roles: ['micro', 'fruitGeneral', 'fruitSpecific', 'query'] },
  { id: 'static_encoding', label: '静态编码层', color: '#8fd4ff', roles: ['fruitGeneral', 'fruitSpecific', 'query', 'macro'] },
  { id: 'dynamic_route', label: '动态路径层', color: '#5eead4', roles: ['route', 'query', 'macro'] },
  { id: 'result_recovery', label: '结果回收层', color: '#fbbf24', roles: ['unifiedDecode', 'route', 'hardTriplet', 'hardBinding'] },
  { id: 'propagation_encoding', label: '传播编码层', color: '#f87171', roles: ['query', 'macro', 'route'] },
  { id: 'semantic_roles', label: '语义角色层', color: '#c084fc', roles: ['style', 'logic', 'syntax', 'query'] },
];

const CONCEPT_ALIAS_MAP = {
  apple: ['苹果'],
  '苹果': ['apple'],
  pear: ['梨'],
  '梨': ['pear'],
  banana: ['香蕉'],
  '香蕉': ['banana'],
  orange: ['橙子', '橘子'],
  '橙子': ['orange'],
  '橘子': ['orange'],
  grape: ['葡萄'],
  '葡萄': ['grape'],
  peach: ['桃子'],
  '桃子': ['peach'],
  mango: ['芒果'],
  '芒果': ['mango'],
  fruit: ['水果'],
  '水果': ['fruit'],
  animal: ['动物'],
  '动物': ['animal'],
  cat: ['猫'],
  '猫': ['cat'],
  dog: ['狗'],
  '狗': ['dog'],
  bird: ['鸟'],
  '鸟': ['bird'],
  tiger: ['老虎'],
  '老虎': ['tiger'],
  lion: ['狮子'],
  '狮子': ['lion'],
  monkey: ['猴子'],
  '猴子': ['monkey'],
};

const LANGUAGE_OVERLAY_META = {
  shared_base: { label: '共享基底', color: '#60a5fa' },
  local_delta: { label: '局部差分', color: '#f97316' },
  path_amplification: { label: '路径放大', color: '#22c55e' },
  semantic_roles: { label: '语义角色', color: '#a78bfa' },
  fidelity: { label: '来源保真', color: '#fb7185' },
};

const LANGUAGE_RISK_META = {
  fidelity: '天然来源保真低',
  competition: '同类高竞争边界脆弱',
  closure: '修复强于原生闭合',
  brand: '品牌义边界弱',
  cross_model: '跨模型硬主核仍少',
};

const DNN_DISPLAY_LEVEL_OPTIONS = [
  {
    id: 'basic_neurons',
    label: '\u57fa\u7840\u795e\u7ecf\u5143',
    desc: '\u663e\u793a\u5f53\u524d 28 \u4e2a layer\uff08\u5c42\uff09\u4e2d\u7684\u57fa\u7840\u6709\u6548\u795e\u7ecf\u5143\u70b9\u3002',
  },
  {
    id: 'object_family',
    label: '\u5bf9\u8c61\u65cf\u6570\u636e',
    desc: '\u663e\u793a\u82f9\u679c\u3001\u6c34\u679c\u7b49\u5bf9\u8c61\u65cf\u76f8\u5173\u795e\u7ecf\u5143\u3002',
  },
  {
    id: 'parameter_state',
    label: '\u53c2\u6570\u4f4d\u6570\u636e',
    desc: '\u663e\u793a\u53c2\u6570\u6001\u8282\u70b9\u3001\u53c2\u6570\u673a\u67b6\u548c\u53c2\u6570\u4f4d\u8be6\u60c5\u3002',
  },
  {
    id: 'mechanism_chain',
    label: '\u8fd0\u884c\u94fe\u8def',
    desc: '\u663e\u793a\u53c2\u6570\u94fe\u8def\u548c\u57fa\u7840\u5c42\u95f4\u8fd0\u884c\u6548\u679c\u3002',
  },
  {
    id: 'advanced_analysis',
    label: '\u9ad8\u7ea7\u5206\u6790',
    desc: '\u663e\u793a\u5171\u4eab\u627f\u8f7d\u3001\u504f\u7f6e\u504f\u8f6c\u3001\u9010\u5c42\u653e\u5927\u7b49\u9ad8\u7ea7\u53e0\u52a0\u5c42\u3002',
  },
];

const DNN_DISPLAY_PRESETS = {
  basic_only: {
    label: '\u53ea\u770b\u57fa\u7840',
    levels: {
      basic_neurons: true,
      object_family: false,
      parameter_state: false,
      mechanism_chain: false,
      advanced_analysis: false,
    },
  },
  parameter_only: {
    label: '\u53ea\u770b\u53c2\u6570',
    levels: {
      basic_neurons: false,
      object_family: false,
      parameter_state: true,
      mechanism_chain: true,
      advanced_analysis: false,
    },
  },
  runtime_focus: {
    label: '\u770b\u8fd0\u884c\u94fe',
    levels: {
      basic_neurons: true,
      object_family: false,
      parameter_state: true,
      mechanism_chain: true,
      advanced_analysis: false,
    },
  },
  all_on: {
    label: '\u5168\u90e8\u6253\u5f00',
    levels: {
      basic_neurons: true,
      object_family: true,
      parameter_state: true,
      mechanism_chain: true,
      advanced_analysis: true,
    },
  },
};

const DNN_RESEARCH_SNAPSHOT = {
  standardizedUnits: 1722,
  exactRealFraction: 0.4884,
  signatureRows: 194,
  uniqueConcepts: 158,
  fullRestorationScore: 0.8704,
  successorTotalUnits: 687,
  successorExactDenseUnits: 96,
  successorProxyUnits: 558,
  successorExactnessFraction: 0.3699,
};

const THEORY_OBJECT_RESEARCH_MAP = {
  family_patch: {
    summary: '当前主看 family patch 是否形成稳定局部图册，而不是松散聚类。',
    metrics: [
      { label: 'family fit strength', value: '0.7846' },
      { label: 'wrong family margin', value: '0.7152' },
      { label: '对应恢复项', value: 'family basis = 75.34%' },
    ],
    sceneHint: '3D 里重点看族群核心团块、共享底座和类别比较面板。',
  },
  concept_section: {
    summary: '当前主看 concept section / offset 是否表现成局部连续偏移，而不是全空间乱跳。',
    metrics: [
      { label: 'concept offset', value: '98.77%' },
      { label: 'specific rows', value: '194' },
      { label: 'unique concepts', value: '158' },
    ],
    sceneHint: '3D 里重点看概念节点相对 family 核心的局部偏移和选中态明细。',
  },
  attribute_fiber: {
    summary: '当前主看属性纤维是否沿稳定方向展开，并能支持组合性与维度切换。',
    metrics: [
      { label: 'topology score', value: '97.32%' },
      { label: 'protocol rows', value: '24' },
      { label: 'topology rows', value: '170' },
    ],
    sceneHint: '3D 里重点看 style / logic / syntax 节点簇和多维探针开关。',
  },
  relation_context_fiber: {
    summary: '当前主看关系与语境纤维如何沿层间路径传播，并在上下文中重组。',
    metrics: [
      { label: 'context operator', value: '87.10%' },
      { label: 'relation topology', value: '已进入真实语料库' },
      { label: 'transport focus', value: 'cross-layer / counterfactual' },
    ],
    sceneHint: '3D 里重点看跨层链路、关系节点和 modeMetrics 中的传输指标。',
  },
  admissible_update: {
    summary: '当前主看哪些局部更新是可容许的，既能写入又不破坏旧结构。',
    metrics: [
      { label: 'hard problem imports', value: '局部信用 / 变量绑定 / 最小回路' },
      { label: '核心风险', value: '局部更新律尚未闭合' },
      { label: '对应动作', value: 'causal / robustness / minimal circuit' },
    ],
    sceneHint: '3D 里重点看硬伤实验节点和因果干预后的局部热点。',
  },
  restricted_readout: {
    summary: '当前主看输出是否主要依赖少数关键读出热点，而不是平均全网读出。',
    metrics: [
      { label: 'minimal circuit', value: '当前系统动作入口' },
      { label: '读出热点', value: 'selected + route 节点' },
      { label: '对应恢复项', value: 'protocol / readout 仍非最终定理' },
    ],
    sceneHint: '3D 里重点看 route 节点、选中热点和最小子回路切换。',
  },
  stage_conditioned_transport: {
    summary: '当前主看不同阶段是否切换不同运输主路，而不是一路直推。',
    metrics: [
      { label: 'transport focus', value: 'dynamic prediction / cross-layer' },
      { label: 'stage rows', value: '20' },
      { label: 'episode-step rows', value: '1920' },
    ],
    sceneHint: '3D 里重点看动态预测和跨层传输下的层级进度与轨迹变化。',
  },
  successor_aligned_transport: {
    summary: '当前主看 successor 是否已经 dense exact 闭合，这也是当前最大硬伤。',
    metrics: [
      { label: 'successor parametric', value: '70.22%' },
      { label: 'exact dense', value: '96 / 687' },
      { label: 'proxy units', value: '558' },
    ],
    sceneHint: '3D 里重点看动态预测链路；后续最值得做 exact vs proxy 双轨迹对照。',
  },
  protocol_bridge: {
    summary: '当前主看内部编码如何进入 protocol field / bridge，而不是停在内部表征。',
    metrics: [
      { label: 'protocol field', value: '95.43%' },
      { label: 'full restoration', value: '87.04%' },
      { label: '主瓶颈', value: 'successor exactness 仍不足' },
    ],
    sceneHint: '3D 里重点看统一解码节点、route 节点与任务闭环线索。',
  },
};

const ANALYSIS_MODE_RESEARCH_NOTES = {
  static: '静态模式适合看 family patch、category compare 和全局编码骨架。',
  dynamic_prediction: '动态预测模式最贴近 successor 与 stage-conditioned transport，是当前最值得盯的缺口。',
  causal_intervention: '因果干预模式适合看 admissible update 与 restricted readout 的真实必要性。',
  subspace_geometry: '子空间编码模式适合看 family basis、concept section 和 attribute fiber 的几何结构。',
  feature_decomposition: '特征分解模式适合看 concept offset、属性轴和有效层的局部解释。',
  cross_layer_transport: '跨层传输模式适合看 relation/context fiber 与 stage-conditioned transport。',
  compositionality: '组合性模式适合看属性纤维是否真能稳定叠加。',
  counterfactual: '反事实模式适合看 concept section 和 successor-aligned transport 的最小改动差分。',
  robustness: '鲁棒模式适合看 admissible update 是否维持稳态。',
  minimal_circuit: '最小子回路模式适合看 restricted readout 与 protocol bridge 的闭环依赖。',
};

const HARD_PROBLEM_EXPERIMENT_LABELS = {
  hard_problem_dynamic_binding_v1: '动态绑定',
  hard_problem_long_horizon_trace_v1: '长程因果链路',
  hard_problem_local_credit_assignment_v1: '局部信用分配',
  triplet_targeted_causal_scan_v1: '三元组定向因果',
  triplet_targeted_multiseed_stability_v1: '三元组多seed稳定性',
  hard_problem_variable_binding_verification_v1: '变量绑定硬验证',
  minimal_causal_circuit_search_v1: '最小因果回路搜索',
  unified_coordinate_system_test_v1: '统一坐标系',
  concept_family_parallel_scale_v1: '规模化概念族',
};

function isHardProblemResultPayload(data) {
  return Boolean(
    data
    && data.schema_version === 'agi_research_result.v1'
    && typeof data.experiment_id === 'string'
    && data.experiment_id in HARD_PROBLEM_EXPERIMENT_LABELS
    && data.metrics
  );
}

function isUnifiedDecodePayload(data) {
  return Boolean(
    data
    && data.axis_stability
    && data.causal_separation
    && data.concept_hierarchy
  );
}

function isBundleManifestPayload(data) {
  return Boolean(data && data.bundle_id === 'agi_research_stage_bundle_v1');
}

function isFourTasksManifestPayload(data) {
  return Boolean(data && data.suite_id === 'agi_four_tasks_suite_v1');
}

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

function buildMetricRowsFromPaths(metrics, paths = [], fallbackLimit = 6) {
  const resolved = paths
    .map((path) => ({ path, raw: getMetricByPath(metrics, path) }))
    .filter((item) => item.raw !== undefined);
  if (resolved.length > 0) {
    return resolved.slice(0, fallbackLimit).map((item) => ({
      label: item.path,
      value: formatPreviewValue(extractMetricScalar(item.raw)),
    }));
  }
  return Object.entries(metrics || {})
    .slice(0, fallbackLimit)
    .map(([key, raw]) => ({
      label: key,
      value: formatPreviewValue(extractMetricScalar(raw)),
    }));
}

function buildArtifactPreview(data, sourcePath = '') {
  const rawJson = safeJsonStringify(data);
  const lowerPath = String(sourcePath || '').toLowerCase();
  const theoryFallback = 'family_patch';
  const base = {
    typeLabel: '未识别研究数据',
    title: String(sourcePath || '未选择文件').replace(/\\/g, '/').split('/').pop() || '未选择文件',
    subtitle: '当前文件没有命中既有 schema，保留原始 JSON 供直接检查。',
    theoryObject: theoryFallback,
    metricRows: [],
    analysisLines: ['直接查看下方原始 JSON，确认字段结构后再决定如何映射到 3D 与理论层。'],
    rawJson,
  };

  if (!data || typeof data !== 'object') {
    return {
      ...base,
      subtitle: '尚未加载到可分析的数据对象。',
      analysisLines: ['当前没有可展示的数据。'],
    };
  }

  if (isAppleSwitchMechanismPayload(data)) {
    const modelEntries = Object.entries(data?.models || {});
    const metricRows = modelEntries.flatMap(([modelKey, modelPayload]) => ([
      {
        label: `${modelKey} 核心单元`,
        value: formatPreviewValue(Array.isArray(modelPayload?.core_units) ? modelPayload.core_units.length : 0),
      },
      {
        label: `${modelKey} 敏感层`,
        value: `L${toSafeNumber(modelPayload?.best_sensitive_layer?.layer_index, 0)}`,
      },
    ]));
    return {
      typeLabel: '苹果切换机制资产',
      title: 'Apple 切换机制统一视图',
      subtitle: '把角色、因果、有符号推进和稳定性压成单一可视化资产，直接服务苹果专题研究界面。',
      theoryObject: 'protocol_bridge',
      metricRows: [
        ...metricRows.slice(0, 6),
        {
          label: '峰值层匹配率',
          value: `${(toSafeNumber(data?.aggregate_stability?.peak_layer_match_rate, 0) * 100).toFixed(1)}%`,
        },
      ],
      analysisLines: [
        '这类资产不是单纯点云，而是苹果切换主线的统一机制视图。',
        '默认应重点看有效单元、敏感层、共享底座层，以及各层的正向推进或反向校正。',
        '它最直接服务 protocol bridge，因为这里开始把内部多层机制压成同一前端阅读接口。',
      ],
      rawJson,
    };
  }

  const hasMultidimProbe = Boolean(
    data?.dimensions?.style
    && data?.dimensions?.logic
    && data?.dimensions?.syntax
    && data?.cross_dimension
  );
  const hasMultidimCausal = Boolean(data?.suppression_matrix_mean && data?.diagonal_advantage);
  const hasMultidimStability = Boolean(data?.aggregate?.diag_adv_style && data?.aggregate?.specificity_margin_style);

  if (isUnifiedDecodePayload(data) || lowerPath.includes('unified_math_structure_decode')) {
    const dims = ['style', 'logic', 'syntax'];
    const metricRows = [
      {
        label: '假设通过率',
        value: `${(toSafeNumber(data?.hypothesis_test?.pass_ratio, 0) * 100).toFixed(1)}%`,
      },
      {
        label: '探针文件数',
        value: formatPreviewValue(toSafeNumber(data?.axis_stability?.n_probe_files, 0)),
      },
      ...dims.map((dim) => ({
        label: `${DIMENSION_LABELS[dim] || dim} profile cosine`,
        value: formatPreviewValue(toSafeNumber(data?.axis_stability?.dimensions?.[dim]?.profile_cosine_mean, 0)),
      })),
    ];
    return {
      typeLabel: '统一解码',
      title: '统一数学结构解码',
      subtitle: '把 style / logic / syntax 的轴稳定性、因果分离和层级结构压成一个统一研究对象。',
      theoryObject: 'protocol_bridge',
      metricRows,
      analysisLines: [
        '数学重点是看三个维度是否能被同一组稳定坐标和层模式统一解释，而不是各自孤立成立。',
        '如果 profile cosine 和 diagonal advantage 同时较高，说明统一坐标系不仅可读，而且具备真实因果分离。',
        '它直接服务于 protocol bridge，因为这里讨论的是内部表征如何收束成统一、可读、可桥接的数学接口。',
      ],
      rawJson,
    };
  }

  if (isBundleManifestPayload(data)) {
    const snap = data?.metrics_snapshot || {};
    return {
      typeLabel: '阶段实验总清单',
      title: 'AGI 阶段实验 bundle',
      subtitle: '这是多项硬伤实验与统一解码的总入口，适合看阶段性闭环是否成立。',
      theoryObject: 'stage_conditioned_transport',
      metricRows: [
        { label: 'seed', value: formatPreviewValue(toSafeNumber(data?.config?.seed, 0)) },
        { label: '动态绑定稳定度', value: formatPreviewValue(toSafeNumber(snap?.dynamic_binding?.binding_stability_index, 0)) },
        { label: '长程衰减', value: formatPreviewValue(toSafeNumber(snap?.long_horizon?.long_horizon_decay, 0)) },
        { label: '局部信用选择性', value: formatPreviewValue(toSafeNumber(snap?.local_credit?.local_selectivity_mean, 0)) },
        { label: '统一解码', value: formatPreviewValue(Boolean(data?.config?.run_unified_decoder)) },
      ],
      analysisLines: [
        '这类清单不是单个现象，而是阶段运输链路是否能同时维持绑定、长程依赖、局部信用和统一解码。',
        '若某个 snapshot 指标明显掉队，说明 stage-conditioned transport 还没有形成稳定主路，理论闭合仍然断裂。',
        '它适合作为“阶段任务是否真正打通”的门槛数据，而不是只看单项实验高分。',
      ],
      rawJson,
    };
  }

  if (isFourTasksManifestPayload(data)) {
    return {
      typeLabel: '四任务套件',
      title: 'AGI 四任务验证',
      subtitle: '变量绑定、最小回路、统一坐标、概念族并行四个任务共同约束内部编码是否可桥接。',
      theoryObject: 'protocol_bridge',
      metricRows: [
        { label: 'all_success', value: formatPreviewValue(Boolean(data?.all_success)) },
        { label: '任务数', value: formatPreviewValue(Object.keys(data?.return_codes || {}).length) },
        { label: 'task1', value: formatPreviewValue(data?.return_codes?.task1) },
        { label: 'task2', value: formatPreviewValue(data?.return_codes?.task2) },
        { label: 'task3', value: formatPreviewValue(data?.return_codes?.task3) },
        { label: 'task4', value: formatPreviewValue(data?.return_codes?.task4) },
      ],
      analysisLines: [
        '四任务同时成功，才说明内部表征不只是局部可解释，而是能穿过多个验证面保持结构一致。',
        '变量绑定和最小回路约束读出必要性，统一坐标和概念族并行约束表征是否可组合、可扩展。',
        '它对应 protocol bridge，因为这里要回答的是“内部结构能否稳定投射到外部任务接口”。',
      ],
      rawJson,
    };
  }

  if (isHardProblemResultPayload(data)) {
    const expId = data?.experiment_id;
    const expLabel = HARD_PROBLEM_EXPERIMENT_LABELS[expId] || expId;
    const theoryObjectMap = {
      hard_problem_dynamic_binding_v1: 'admissible_update',
      hard_problem_long_horizon_trace_v1: 'stage_conditioned_transport',
      hard_problem_local_credit_assignment_v1: 'admissible_update',
      triplet_targeted_causal_scan_v1: 'relation_context_fiber',
      triplet_targeted_multiseed_stability_v1: 'relation_context_fiber',
      hard_problem_variable_binding_verification_v1: 'protocol_bridge',
      minimal_causal_circuit_search_v1: 'restricted_readout',
      unified_coordinate_system_test_v1: 'protocol_bridge',
      concept_family_parallel_scale_v1: 'family_patch',
    };
    const preferredMetricPaths = {
      hard_problem_dynamic_binding_v1: ['binding_stability_index', 'slot_collision_rate', 'binding_gap'],
      hard_problem_long_horizon_trace_v1: ['long_horizon_decay', 'trace_recovery_ratio', 'path_consistency'],
      hard_problem_local_credit_assignment_v1: ['local_selectivity_mean', 'credit_localization_score', 'credit_spread_ratio'],
      triplet_targeted_causal_scan_v1: ['target_hit_rate', 'triplet_margin', 'causal_precision'],
      triplet_targeted_multiseed_stability_v1: ['aggregate.target_hit_rate.mean', 'aggregate.triplet_margin.mean', 'aggregate.causal_precision.mean'],
      hard_problem_variable_binding_verification_v1: ['binding_success_rate', 'swap_error_rate', 'role_consistency'],
      minimal_causal_circuit_search_v1: ['summary.best_fidelity_ratio', 'summary.min_subset_size', 'summary.coverage_ratio'],
      unified_coordinate_system_test_v1: ['alignment_score', 'axis_consistency', 'transfer_success_rate'],
      concept_family_parallel_scale_v1: ['parallel_scale_score', 'family_margin', 'concept_separation'],
    };
    return {
      typeLabel: '硬伤实验',
      title: expLabel,
      subtitle: '这是直接检验编码机制硬伤的实验结果，不是装饰性统计。',
      theoryObject: theoryObjectMap[expId] || 'admissible_update',
      metricRows: buildMetricRowsFromPaths(data?.metrics || {}, preferredMetricPaths[expId] || []),
      analysisLines: [
        '硬伤实验的意义在于强行追问当前编码机制在哪一步会断，而不是只展示已有结构。',
        `当前文件对应实验：${expLabel}。如果关键指标不稳，说明相关理论对象还没有得到足够严格的神经级闭包。`,
        '这类数据优先用于决定下一步补哪条数学机制链路，而不是直接当作最终证明。',
      ],
      rawJson,
    };
  }

  if (hasMultidimProbe) {
    return {
      typeLabel: '多维编码探针',
      title: 'Style / Logic / Syntax 探针',
      subtitle: '用于观察三个维度是否形成稳定分离的子空间结构。',
      theoryObject: 'attribute_fiber',
      metricRows: [
        { label: 'style top_k', value: formatPreviewValue(toSafeNumber(data?.dimensions?.style?.top_neurons?.length, 0)) },
        { label: 'logic top_k', value: formatPreviewValue(toSafeNumber(data?.dimensions?.logic?.top_neurons?.length, 0)) },
        { label: 'syntax top_k', value: formatPreviewValue(toSafeNumber(data?.dimensions?.syntax?.top_neurons?.length, 0)) },
        { label: 'cross margin', value: formatPreviewValue(toSafeNumber(data?.cross_dimension?.margin_mean, 0)) },
        { label: 'probe top_k', value: formatPreviewValue(toSafeNumber(data?.runtime_config?.top_k, 0)) },
      ],
      analysisLines: [
        '数学上看的是属性纤维是否真沿相对稳定的方向展开，而不是把不同维度混进一个不可分的团块。',
        '如果 cross-dimension margin、各维 top neurons 和后续因果消融结果能互相支持，说明子空间结构更接近真实机制。',
        '这类数据优先服务 attribute fiber，也会影响 relation/context fiber 的后续解释。',
      ],
      rawJson,
    };
  }

  if (hasMultidimCausal) {
    return {
      typeLabel: '多维因果消融',
      title: '多维因果分离',
      subtitle: '不是只看探针可分，而是看因果消融后对角优势是否成立。',
      theoryObject: 'attribute_fiber',
      metricRows: ['style', 'logic', 'syntax'].map((dim) => ({
        label: `${DIMENSION_LABELS[dim] || dim} diagonal advantage`,
        value: formatPreviewValue(toSafeNumber(data?.diagonal_advantage?.[dim], 0)),
      })),
      analysisLines: [
        '对角优势高，意味着干预某个维度主要影响自身，而不是把全部维度一起打散。',
        '这类结果比纯探针更接近真实数学结构，因为它包含因果必要性的信息。',
        '它仍然属于 attribute fiber，但比普通探针更接近可用于证明的证据层。',
      ],
      rawJson,
    };
  }

  if (hasMultidimStability) {
    return {
      typeLabel: '多 seed 稳定性',
      title: '多维编码稳定性',
      subtitle: '用于检验同一结构是否能跨 seed 稳定复现，而不是一次性偶然现象。',
      theoryObject: 'attribute_fiber',
      metricRows: [
        { label: 'runs', value: formatPreviewValue(toSafeNumber(data?.n_runs, 0)) },
        { label: 'style specificity', value: formatPreviewValue(toSafeNumber(data?.aggregate?.specificity_margin_style?.mean, 0)) },
        { label: 'logic specificity', value: formatPreviewValue(toSafeNumber(data?.aggregate?.specificity_margin_logic?.mean, 0)) },
        { label: 'syntax specificity', value: formatPreviewValue(toSafeNumber(data?.aggregate?.specificity_margin_syntax?.mean, 0)) },
      ],
      analysisLines: [
        '稳定性是把“看起来成立”推进到“重复实验仍成立”的关键一步。',
        '如果多 seed specificity 波动很大，说明当前提取到的维度可能还只是局部表象，不足以支持通用数学恢复。',
        '这类数据适合作为 attribute fiber 与 protocol bridge 之间的中间证据层。',
      ],
      rawJson,
    };
  }

  const nounRecords = Array.isArray(data?.noun_records) ? data.noun_records : [];
  if (nounRecords.length > 0) {
    const minimalRecords = Array.isArray(data?.causal_ablation?.minimal_circuit?.records)
      ? data.causal_ablation.minimal_circuit.records
      : [];
    const counterfactualRecords = Array.isArray(data?.causal_ablation?.counterfactual_validation?.records)
      ? data.causal_ablation.counterfactual_validation.records
      : [];
    const categoryCount = new Set(nounRecords.map((row) => String(row?.category || '未分类'))).size;
    return {
      typeLabel: '名词扫描',
      title: '名词编码扫描',
      subtitle: '直接展开名词、类别、共享神经元和最小回路，是手动查看 3D 映射最直观的入口。',
      theoryObject: 'family_patch',
      metricRows: [
        { label: '名词总数', value: formatPreviewValue(nounRecords.length) },
        { label: '类别数', value: formatPreviewValue(categoryCount) },
        { label: '共享神经元', value: formatPreviewValue(toSafeNumber(data?.top_reused_neurons?.length, 0)) },
        { label: '最小回路记录', value: formatPreviewValue(minimalRecords.length) },
        { label: '反事实对', value: formatPreviewValue(counterfactualRecords.length) },
        { label: 'd_ff', value: formatPreviewValue(toSafeNumber(data?.config?.d_ff, DFF)) },
      ],
      analysisLines: [
        '名词扫描直接回答 family patch 和 concept section 是否已经形成稳定局部图册，而不是散点式相关性。',
        '最小回路和反事实对把“哪些神经元在表示”推进到“哪些神经元是必要的”。',
        '这类数据最适合和手动输入的 3D 概念观察联动，用来检验局部几何与因果必要性是否一致。',
      ],
      rawJson,
    };
  }

  return base;
}

function shouldShowResearchAssetInTopRight(scanPreview, selectedScanPath = '') {
  const normalizedPath = String(selectedScanPath || '').toLowerCase();
  if (
    normalizedPath.includes('apple_switch_mechanism_view')
    || scanPreview?.typeLabel === '苹果切换机制资产'
  ) {
    return true;
  }
  if (
    normalizedPath.includes('multidim_encoding_probe')
    || normalizedPath.includes('mass_noun')
    || normalizedPath.includes('noun_scan')
    || normalizedPath.includes('encoding_scan')
  ) {
    return true;
  }

  return scanPreview?.title === '名词编码扫描'
    || scanPreview?.title === 'Style / Logic / Syntax 探针'
    || scanPreview?.typeLabel === '名词扫描'
    || scanPreview?.typeLabel === '多维编码探针';
}

const MODE_VISUALS = {
  static: { accent: '#e5e7eb', nodePulse: 0.7, nodeSpeed: 0.85, linkOpacityBoost: 0.02, linkWidthBoost: 0, carrier: 'none' },
  dynamic_prediction: { accent: '#7ee0ff', nodePulse: 1.0, nodeSpeed: 1.0, linkOpacityBoost: 0.18, linkWidthBoost: 0.2, carrier: 'torus' },
  causal_intervention: { accent: '#ff6b6b', nodePulse: 1.3, nodeSpeed: 1.3, linkOpacityBoost: 0.32, linkWidthBoost: 0.45, carrier: 'octa' },
  subspace_geometry: { accent: '#c084fc', nodePulse: 0.95, nodeSpeed: 0.9, linkOpacityBoost: 0.22, linkWidthBoost: 0.25, carrier: 'plane' },
  feature_decomposition: { accent: '#f59e0b', nodePulse: 1.12, nodeSpeed: 1.05, linkOpacityBoost: 0.26, linkWidthBoost: 0.3, carrier: 'tetra' },
  cross_layer_transport: { accent: '#22d3ee', nodePulse: 1.08, nodeSpeed: 1.15, linkOpacityBoost: 0.28, linkWidthBoost: 0.28, carrier: 'cylinder' },
  compositionality: { accent: '#34d399', nodePulse: 1.2, nodeSpeed: 1.1, linkOpacityBoost: 0.26, linkWidthBoost: 0.35, carrier: 'tri_ring' },
  counterfactual: { accent: '#fb7185', nodePulse: 1.22, nodeSpeed: 1.28, linkOpacityBoost: 0.3, linkWidthBoost: 0.35, carrier: 'dual_ring' },
  robustness: { accent: '#a3e635', nodePulse: 0.88, nodeSpeed: 0.82, linkOpacityBoost: 0.14, linkWidthBoost: 0.18, carrier: 'shield' },
  minimal_circuit: { accent: '#f97316', nodePulse: 1.35, nodeSpeed: 1.38, linkOpacityBoost: 0.34, linkWidthBoost: 0.5, carrier: 'hex' },
};

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

function neuronToPosition(layer, neuron, radialJitter = 0) {
  const angle = ((neuron % 4096) / 4096) * Math.PI * 2;
  const radius = 2.7 + ((neuron % 2048) / 2048) * 3.3 + radialJitter;
  const z = (layer - (LAYER_COUNT - 1) / 2) * 0.92;
  const x = Math.cos(angle) * radius;
  const y = Math.sin(angle) * radius;
  return [x, y, z];
}

function PulsingNeuron({
  node,
  selected,
  onSelect,
  predictionStrength = 0,
  mode = 'static',
  isEffectiveNode = false,
  visibilityEmphasis = 1,
  motionEnabled = false,
}) {
  const ref = useRef(null);
  const modeStyle = MODE_VISUALS[mode] || MODE_VISUALS.static;

  useFrame((state) => {
    if (!ref.current) {
      return;
    }
    if (!motionEnabled) {
      const stableScale = node.size * (selected ? 1.12 : isEffectiveNode ? 1.08 : 1);
      ref.current.scale.set(stableScale, stableScale, stableScale);
      return;
    }
    const pulse = (node.role === 'background' ? 0.04 : 0.14) * modeStyle.nodePulse;
    const speed = (node.role === 'background' ? 1.2 : 2.1) * modeStyle.nodeSpeed;
    const base = node.size;
    const predictionBoost = predictionStrength * (node.role === 'background' ? 0.18 : 0.5) * (0.6 + 0.4 * visibilityEmphasis);
    const modeWave = mode === 'counterfactual' ? Math.sin(state.clock.elapsedTime * speed * 0.7 + node.phase * 1.3) * 0.06 : 0;
    const effectiveBoost = isEffectiveNode ? 0.22 : 0;
    const scale = base * (1 + Math.sin(state.clock.elapsedTime * speed + node.phase) * pulse + predictionBoost + modeWave + effectiveBoost);
    ref.current.scale.set(scale, scale, scale);
  });

  return (
    <mesh
      ref={ref}
      position={node.position}
      onClick={(e) => {
        e.stopPropagation();
        onSelect(node);
      }}
    >
      <sphereGeometry args={[1, 20, 20]} />
      <meshStandardMaterial
        color={isEffectiveNode ? '#ffffff' : predictionStrength > 0.66 && mode !== 'static' ? modeStyle.accent : node.color}
        emissive={isEffectiveNode ? '#ffffff' : predictionStrength > 0.5 && mode !== 'static' ? modeStyle.accent : node.color}
        emissiveIntensity={
          (selected ? 1.8 : node.role === 'background' ? 0.08 : 0.55)
          + predictionStrength * (node.role === 'background' ? 0.2 : 1.6)
          + (isEffectiveNode ? 0.95 : 0)
          + (mode !== 'static' ? 0.12 : 0)
        }
        roughness={0.2}
        metalness={0.15}
        transparent
        opacity={(isEffectiveNode ? 0.98 : node.role === 'background' ? 0.24 + predictionStrength * 0.08 : 0.92) * visibilityEmphasis}
      />
    </mesh>
  );
}

function LayerEffectiveNeuronOverlay({ prediction = null, mode = 'static' }) {
  if (mode !== 'feature_decomposition') {
    return null;
  }
  const layer = Number.isFinite(prediction?.effectiveLayer)
    ? Math.max(0, Math.min(LAYER_COUNT - 1, Math.round(prediction.effectiveLayer)))
    : null;
  if (!Number.isFinite(layer)) {
    return null;
  }
  const rows = Array.isArray(prediction?.effectiveNeurons) ? prediction.effectiveNeurons.slice(0, 6) : [];
  const z = (layer - (LAYER_COUNT - 1) / 2) * 0.92;
  return (
    <group position={[0, 0, z]}>
      <Line points={[[3.8, 0.2, 0], [7.05, 1.95, 0]]} color="#ffffff" transparent opacity={0.82} lineWidth={1.4} />
      <Html position={[7.25, 2.12, 0]} center={false}>
        <div
          style={{
            width: 226,
            borderRadius: 10,
            border: '1px solid rgba(255,255,255,0.58)',
            background: 'rgba(8, 12, 24, 0.86)',
            color: '#e8f2ff',
            padding: '8px 10px',
            fontSize: 11,
            lineHeight: 1.55,
            boxShadow: '0 10px 24px rgba(0,0,0,0.35)',
            pointerEvents: 'none',
          }}
        >
          <div style={{ fontWeight: 700, color: '#ffffff' }}>{`L${layer} 有效神经元 Top-${rows.length}`}</div>
          {rows.length === 0 ? (
            <div style={{ color: '#9bb3de' }}>当前层暂无可显示节点</div>
          ) : (
            rows.map((item, idx) => (
              <div key={`eff-n-${item.id}-${idx}`} style={{ color: '#d4e5ff' }}>
                {`${idx + 1}. N${item.neuron} | ${item.role} | ${(Number(item.score || 0) * 100).toFixed(1)}%`}
              </div>
            ))
          )}
        </div>
      </Html>
    </group>
  );
}

function buildParameterStateSelection(point, position, layerKey, sourceDataPath) {
  return {
    id: `parameter-state-${point.id}`,
    label: point.label,
    role: 'route',
    category: point.category,
    layer: point.layer,
    neuron: point.neuron,
    metric: point.metric,
    value: point.value,
    strength: point.strength,
    source: point.sourceStage,
    sourceStage: point.sourceStage,
    outputDir: point.outputDir,
    parameterIds: point.parameterIds,
    dimIndex: point.dimIndex,
    sourceEntityId: point.sourceEntityId,
    sourceDataPath,
    detailType: 'parameter_state',
    overlayLayer: layerKey,
    position,
  };
}

function ParameterStatePoint({
  point,
  color,
  selected = false,
  active = false,
  onSelect,
  overlayLayer,
  sourceDataPath,
  orderIndex = 0,
  motionEnabled = false,
}) {
  const ref = useRef(null);
  const position = useMemo(
    () => neuronToPosition(point.layer, point.neuron, 0.42 + orderIndex * 0.07),
    [orderIndex, point.layer, point.neuron]
  );
  const layerAnchor = useMemo(() => [-position[0], -position[1], 0], [position]);

  useFrame((state) => {
    if (!ref.current) {
      return;
    }
    if (!motionEnabled) {
      const stableScale = 0.42 + point.strength * 0.22 + (selected ? 0.24 : active ? 0.14 : 0);
      ref.current.scale.setScalar(stableScale);
      return;
    }
    const pulse = 1 + Math.sin(state.clock.elapsedTime * 2.8 + orderIndex * 0.45) * 0.12;
    const boost = selected ? 0.24 : active ? 0.14 : 0;
    const scale = 0.42 + point.strength * 0.22 + boost;
    ref.current.scale.setScalar(scale * pulse);
  });

  return (
    <group position={position}>
      <Line
        points={[layerAnchor, position]}
        color={active ? '#ffffff' : color}
        transparent
        opacity={selected ? 0.95 : active ? 0.8 : 0.5}
        lineWidth={active ? 2.8 : 2}
      />

      <mesh position={[0, 0, -0.02]} scale={[1.4, 1.4, 0.22]} renderOrder={78}>
        <cylinderGeometry args={[1, 1, 1, 16]} />
        <meshBasicMaterial color={active ? '#ffffff' : color} transparent opacity={selected ? 0.36 : 0.22} depthTest={false} toneMapped={false} />
      </mesh>

      <mesh
        ref={ref}
        renderOrder={80}
        onClick={(e) => {
          e.stopPropagation();
          onSelect(buildParameterStateSelection(point, position, overlayLayer, sourceDataPath));
        }}
      >
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial
          color={selected ? '#ffffff' : active ? '#f8fafc' : color}
          transparent
          opacity={0.95}
          depthTest={false}
          toneMapped={false}
        />
      </mesh>

      <mesh scale={[2.8, 2.8, 2.8]} renderOrder={79}>
        <sphereGeometry args={[1, 12, 12]} />
        <meshBasicMaterial color={active ? '#ffffff' : color} transparent opacity={selected ? 0.28 : active ? 0.22 : 0.16} depthTest={false} toneMapped={false} />
      </mesh>

      <Text position={[0, 1.05, 0]} color="#ffffff" fontSize={0.32} anchorX="center" anchorY="bottom" renderOrder={81}>
        {`d${point.dimIndex}`}
      </Text>
      <Text position={[0, -0.92, 0]} color={active ? '#ffffff' : '#dbeafe'} fontSize={0.18} anchorX="center" anchorY="top" renderOrder={81}>
        {`L${point.layer}`}
      </Text>
    </group>
  );
}

function ParameterStateSummaryOverlay({ profile }) {
  if (!profile || !Array.isArray(profile.nodes) || profile.nodes.length === 0) {
    return null;
  }
  return (
    <Html position={[10.8, 8.4, 0]} transform sprite>
      <div
        style={{
          minWidth: 220,
          padding: '10px 12px',
          borderRadius: 12,
          background: 'rgba(7, 12, 25, 0.84)',
          border: `1px solid ${profile.color || '#60a5fa'}`,
          boxShadow: `0 0 18px ${(profile.color || '#60a5fa')}33`,
          color: '#e5eeff',
          pointerEvents: 'none',
        }}
      >
        <div style={{ fontSize: 12, fontWeight: 800, marginBottom: 6 }}>{`${profile.label} 参数节点`}</div>
        <div style={{ display: 'grid', gap: 4 }}>
          {profile.nodes.map((node) => (
            <div key={`param-summary-${node.id}`} style={{ fontSize: 11, lineHeight: 1.5 }}>
              {`L${node.layer} · d${node.dimIndex} · ${Number(node.value || 0).toFixed(4)}`}
            </div>
          ))}
        </div>
      </div>
    </Html>
  );
}

function buildParameterRackPosition(layer, orderIndex = 0, totalInLayer = 1) {
  const safeTotal = Math.max(1, totalInLayer);
  const rowIndex = orderIndex % safeTotal;
  const x = 10.6 + Math.floor(orderIndex / 4) * 0.92;
  const y = 2.2 - rowIndex * 1.08;
  const z = (layer - (LAYER_COUNT - 1) / 2) * 0.92;
  return [x, y, z];
}

function ParameterRackOverlay({ profile, selected = null, onSelect = () => {} }) {
  if (!profile || !Array.isArray(profile.nodes) || profile.nodes.length === 0) {
    return null;
  }

  const groupedCounts = profile.nodes.reduce((acc, node) => {
    const key = Number(node?.layer);
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});

  const seenPerLayer = {};

  return (
    <group>
      <Text position={[10.8, 5.9, 0]} color={profile.color || '#60a5fa'} fontSize={0.24} anchorX="center" anchorY="middle">
        {'参数机架'}
      </Text>
      {profile.nodes.map((node, idx) => {
        const layer = Number(node?.layer) || 0;
        const orderIndex = seenPerLayer[layer] || 0;
        seenPerLayer[layer] = orderIndex + 1;
        const rackPosition = buildParameterRackPosition(layer, orderIndex, groupedCounts[layer] || 1);
        const neuronPosition = neuronToPosition(node.layer, node.neuron, 0.42 + idx * 0.07);
        const isSelected = selected?.id === `parameter-state-${node.id}`;
        const accent = isSelected ? '#ffffff' : (profile.color || '#60a5fa');

        return (
          <group key={`parameter-rack-${node.id}`}>
            <Line
              points={[rackPosition, neuronPosition]}
              color={accent}
              transparent
              opacity={isSelected ? 0.92 : 0.48}
              lineWidth={isSelected ? 2.6 : 1.6}
            />
            <group
              position={rackPosition}
              onClick={(e) => {
                e.stopPropagation();
                onSelect(buildParameterStateSelection(node, rackPosition, profile.layerKey || 'static_encoding', profile.sourceDataPath));
              }}
            >
              <mesh renderOrder={86}>
                <boxGeometry args={[0.72, 0.72, 0.72]} />
                <meshBasicMaterial color={accent} transparent opacity={0.95} depthTest={false} toneMapped={false} />
              </mesh>
              <mesh scale={[1.9, 1.9, 1.9]} renderOrder={85}>
                <sphereGeometry args={[1, 12, 12]} />
                <meshBasicMaterial color={accent} transparent opacity={isSelected ? 0.26 : 0.14} depthTest={false} toneMapped={false} />
              </mesh>
              <Text position={[0, 0.72, 0]} color="#ffffff" fontSize={0.22} anchorX="center" anchorY="bottom" renderOrder={87}>
                {`d${node.dimIndex}`}
              </Text>
              <Text position={[0, -0.68, 0]} color="#dbeafe" fontSize={0.15} anchorX="center" anchorY="top" renderOrder={87}>
                {`L${node.layer}`}
              </Text>
            </group>
          </group>
        );
      })}
    </group>
  );
}

function LayerParameterStateOverlay({
  languageFocus = DEFAULT_LANGUAGE_FOCUS,
  selected = null,
  onSelect = () => {},
  activeIndex = -1,
  isPlaying = false,
}) {
  const layerKey = LAYER_PARAMETER_STATE_ORDER.includes(languageFocus?.researchLayer)
    ? languageFocus.researchLayer
    : 'static_encoding';
  const baseProfile = LAYER_PARAMETER_STATE_OVERLAY[layerKey] || LAYER_PARAMETER_STATE_OVERLAY.static_encoding;
  const profile = useMemo(() => ({ ...baseProfile, layerKey }), [baseProfile, layerKey]);
  const color = profile.color || '#60a5fa';
  const nodePositionMap = useMemo(
    () => Object.fromEntries(
      profile.nodes.map((point, idx) => [
        point.id,
        neuronToPosition(point.layer, point.neuron, 0.42 + idx * 0.07),
      ])
    ),
    [profile.nodes]
  );

  return (
    <group>
      <ParameterStateSummaryOverlay profile={profile} />
      <ParameterRackOverlay profile={profile} selected={selected} onSelect={onSelect} />
      <Text position={[0, 9.6, 0]} color={color} fontSize={0.26} anchorX="center" anchorY="middle">
        {`${profile.label} 参数态`}
      </Text>

      {profile.chains.map(([fromId, toId], chainIndex) => {
        const from = nodePositionMap[fromId];
        const to = nodePositionMap[toId];
        if (!from || !to) {
          return null;
        }
        const fromIndex = profile.nodes.findIndex((item) => item.id === fromId);
        const toIndex = profile.nodes.findIndex((item) => item.id === toId);
        const chainActive = activeIndex >= fromIndex && activeIndex <= toIndex;
        return (
          <group key={`parameter-chain-${fromId}-${toId}`}>
            <Line
              points={[from, to]}
              color={chainActive ? '#ffffff' : color}
              transparent
              opacity={chainActive ? 0.92 : 0.66}
              lineWidth={chainActive ? 2.8 : 1.6}
            />
            {isPlaying ? (
              <TheoryRunner
                path={[from, to]}
                color={chainActive ? '#ffffff' : color}
                size={chainActive ? 0.12 : 0.08}
                speed={0.26 + chainIndex * 0.04}
                phase={chainIndex * 0.18}
              />
            ) : null}
          </group>
        );
      })}

      {profile.nodes.map((point, idx) => (
        <ParameterStatePoint
          key={point.id}
          point={point}
          color={color}
          selected={selected?.id === `parameter-state-${point.id}`}
          active={idx === activeIndex}
          onSelect={onSelect}
          overlayLayer={layerKey}
          sourceDataPath={profile.sourceDataPath}
          orderIndex={idx}
          motionEnabled={isPlaying}
        />
      ))}

      {selected?.detailType === 'parameter_state' && selected?.overlayLayer === layerKey && Array.isArray(selected?.position) && (
        <Html position={[selected.position[0] + 1.2, selected.position[1] + 0.8, selected.position[2]]}>
          <div
            style={{
              width: 240,
              padding: '10px 12px',
              borderRadius: 10,
              background: 'rgba(8, 12, 24, 0.9)',
              border: `1px solid ${color}`,
              color: '#e5eeff',
              fontSize: 11,
              lineHeight: 1.55,
              boxShadow: '0 10px 24px rgba(0,0,0,0.28)',
              pointerEvents: 'none',
            }}
          >
            <div style={{ color: '#ffffff', fontWeight: 700, marginBottom: 6 }}>{selected.label}</div>
            <div>{`层 / 神经元: L${selected.layer} / N${selected.neuron}`}</div>
            <div>{`参数维度: d${selected.dimIndex}`}</div>
            <div>{`来源阶段: ${selected.sourceStage}`}</div>
            <div>{`指标: ${selected.metric} = ${Number(selected.value || 0).toFixed(4)}`}</div>
            <div>{`参数位: ${(selected.parameterIds || []).join(', ')}`}</div>
          </div>
        </Html>
      )}
    </group>
  );
}

function LayerBasicRuntimeControls({
  title = '参数态基础动画',
  onStart = () => {},
  onStop = () => {},
  onReplay = () => {},
  isPlaying = false,
}) {
  const buttonStyle = {
    borderRadius: 8,
    border: '1px solid rgba(148, 163, 184, 0.45)',
    background: 'rgba(15, 23, 42, 0.88)',
    color: '#e2e8f0',
    fontSize: 11,
    padding: '6px 10px',
    cursor: 'pointer',
  };

  return (
    <Html position={[-11.2, 8.9, 0]} transform sprite>
      <div
        style={{
          minWidth: 220,
          padding: '10px 12px',
          borderRadius: 12,
          background: 'rgba(8, 12, 24, 0.88)',
          border: '1px solid rgba(96, 165, 250, 0.45)',
          boxShadow: '0 12px 28px rgba(0,0,0,0.28)',
          color: '#e2e8f0',
          backdropFilter: 'blur(10px)',
        }}
      >
        <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>{title}</div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button type="button" onClick={onStart} style={buttonStyle}>
            开始动画
          </button>
          <button type="button" onClick={onStop} style={buttonStyle}>
            结束动画
          </button>
          <button type="button" onClick={onReplay} style={buttonStyle}>
            重新播放
          </button>
        </div>
        <div style={{ marginTop: 8, fontSize: 10, color: '#93c5fd' }}>
          {isPlaying ? '当前状态：播放中' : '当前状态：静止'}
        </div>
      </div>
    </Html>
  );
}

function LayerGuides({ activeLayer = null }) {
  const layers = useMemo(() => Array.from({ length: LAYER_COUNT }, (_, i) => i), []);
  const hasActiveLayer = Number.isFinite(activeLayer);
  const activeLayerIndex = hasActiveLayer
    ? Math.max(0, Math.min(LAYER_COUNT - 1, Math.round(activeLayer)))
    : null;
  return (
    <group>
      {layers.map((layer) => {
        const z = (layer - (LAYER_COUNT - 1) / 2) * 0.92;
        const isMajor = layer % 4 === 0 || layer === LAYER_COUNT - 1;
        const isActive = activeLayerIndex === layer;
        const lineColor = isActive ? '#ffffff' : isMajor ? '#dbeafe' : '#8ea4c7';
        const lineOpacity = isActive ? 0.8 : isMajor ? 0.2 : 0.1;
        const labelColor = isActive ? '#ffffff' : isMajor ? '#d8ecff' : '#9cb6dc';
        const labelSize = isActive ? 0.38 : isMajor ? 0.3 : 0.22;
        return (
          <group key={`layer-${layer}`}>
            <Line
              points={[
                [-7.5, -7.5, z],
                [7.5, -7.5, z],
                [7.5, 7.5, z],
                [-7.5, 7.5, z],
                [-7.5, -7.5, z],
              ]}
              color={lineColor}
              transparent
              opacity={lineOpacity}
              lineWidth={1}
            />
            <Text
              position={[-8.55, 0, z]}
              color={labelColor}
              fontSize={labelSize}
              anchorX="left"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="#0a1022"
            >
              {`L${layer}`}
            </Text>
            <Text
              position={[8.55, 0, z]}
              color={labelColor}
              fontSize={labelSize}
              anchorX="right"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="#0a1022"
            >
              {`L${layer}`}
            </Text>
            {isActive && (
              <Line
                points={[
                  [-6.2, -6.2, z],
                  [6.2, -6.2, z],
                  [6.2, 6.2, z],
                  [-6.2, 6.2, z],
                  [-6.2, -6.2, z],
                ]}
                color="#ffffff"
                transparent
                opacity={0.58}
                lineWidth={1.6}
              />
            )}
          </group>
        );
      })}
      <Line points={[[0, 0, -13.2], [0, 0, 13.2]]} color="#ffffff" transparent opacity={0.7} lineWidth={1.2} />
      <Text position={[0, 0.95, -13.2]} color="#cde4ff" fontSize={0.28} anchorX="center" anchorY="middle" outlineWidth={0.015} outlineColor="#0a1022">
        Layer 0
      </Text>
      <Text position={[0, 0.95, 13.2]} color="#cde4ff" fontSize={0.28} anchorX="center" anchorY="middle" outlineWidth={0.015} outlineColor="#0a1022">
        Layer 27
      </Text>
    </group>
  );
}

function DimensionLayerImpactGraph({ profile = [], dimension = 'style', suppression = null }) {
  if (!Array.isArray(profile) || profile.length === 0) {
    return null;
  }
  const color = ROLE_COLORS[dimension] || '#84f1ff';
  const points = profile.map((v, layer) => {
    const z = (layer - (LAYER_COUNT - 1) / 2) * 0.92;
    const x = 8.2 + Math.max(0, toSafeNumber(v, 0)) * 4.8;
    const y = -6.45;
    return [x, y, z];
  });
  const diagAdv = suppression?.diagonal_advantage?.[dimension];
  const row = suppression?.suppression_matrix_mean?.[dimension];
  return (
    <group>
      <Line points={[[8.2, -6.45, -13.1], [8.2, -6.45, 13.1]]} color="#7f95bb" transparent opacity={0.28} lineWidth={1} />
      <Line points={points} color={color} transparent opacity={0.95} lineWidth={2.6} />
      {points.map((p, idx) => (
        <mesh key={`impact-${dimension}-${idx}`} position={p}>
          <sphereGeometry args={[0.045, 12, 12]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.9} />
        </mesh>
      ))}
      <Text position={[9.7, -5.8, 13.3]} color={color} fontSize={0.23} anchorX="right" anchorY="middle" outlineWidth={0.015} outlineColor="#0a1022">
        {`${DIMENSION_LABELS[dimension] || dimension} 层影响谱`}
      </Text>
      {Number.isFinite(diagAdv) && (
        <Text position={[9.7, -6.2, 13.3]} color="#cde4ff" fontSize={0.2} anchorX="right" anchorY="middle" outlineWidth={0.012} outlineColor="#0a1022">
          {`对角优势: ${diagAdv.toFixed(4)}`}
        </Text>
      )}
      {row ? (
        <Text position={[9.7, -6.55, 13.3]} color="#9eb4dd" fontSize={0.18} anchorX="right" anchorY="middle" outlineWidth={0.01} outlineColor="#0a1022">
          {`S/L/Y: ${toSafeNumber(row.style, 0).toFixed(3)} / ${toSafeNumber(row.logic, 0).toFixed(3)} / ${toSafeNumber(row.syntax, 0).toFixed(3)}`}
        </Text>
      ) : null}
    </group>
  );
}

function TokenPredictionCarrier({ prediction, mode = 'static' }) {
  const ref = useRef(null);
  const modeStyle = MODE_VISUALS[mode] || MODE_VISUALS.static;
  const movingColor = '#ffffff';

  useFrame((state) => {
    if (!ref.current) {
      return;
    }
    ref.current.rotation.y = state.clock.elapsedTime * (1.1 + modeStyle.nodeSpeed * 0.7);
  });

  if (!prediction?.currentToken || modeStyle.carrier === 'none') {
    return null;
  }

  const z = (prediction.layerProgress - 0.5) * (LAYER_COUNT - 1) * 0.92;
  const radius = 0.5 + prediction.currentToken.prob * 0.75;
  return (
    <group position={[0, 0, z]}>
      {modeStyle.carrier === 'torus' && (
        <mesh ref={ref}>
          <torusGeometry args={[radius, 0.08, 14, 42]} />
          <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.4} transparent opacity={0.75} />
        </mesh>
      )}
      {modeStyle.carrier === 'octa' && (
        <mesh ref={ref}>
          <octahedronGeometry args={[radius * 0.92]} />
          <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.2} transparent opacity={0.72} wireframe />
        </mesh>
      )}
      {modeStyle.carrier === 'plane' && (
        <mesh ref={ref} rotation={[0.55, 0.25, 0.15]}>
          <boxGeometry args={[radius * 1.95, 0.08, radius * 1.1]} />
          <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.0} transparent opacity={0.55} />
        </mesh>
      )}
      {modeStyle.carrier === 'tetra' && (
        <mesh ref={ref}>
          <tetrahedronGeometry args={[radius * 0.95]} />
          <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.2} transparent opacity={0.72} />
        </mesh>
      )}
      {modeStyle.carrier === 'cylinder' && (
        <mesh ref={ref}>
          <cylinderGeometry args={[radius * 0.22, radius * 0.22, radius * 2.0, 16]} />
          <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.15} transparent opacity={0.72} />
        </mesh>
      )}
      {modeStyle.carrier === 'tri_ring' && (
        <group ref={ref}>
          <mesh rotation={[0, 0, 0]}>
            <torusGeometry args={[radius * 0.9, 0.07, 12, 36]} />
            <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.2} transparent opacity={0.72} />
          </mesh>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[radius * 0.7, 0.07, 12, 36]} />
            <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.2} transparent opacity={0.54} />
          </mesh>
          <mesh rotation={[0, Math.PI / 2, 0]}>
            <torusGeometry args={[radius * 0.52, 0.07, 12, 36]} />
            <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.2} transparent opacity={0.4} />
          </mesh>
        </group>
      )}
      {modeStyle.carrier === 'dual_ring' && (
        <group ref={ref}>
          <mesh position={[-0.36, 0, 0]}>
            <torusGeometry args={[radius * 0.6, 0.07, 12, 36]} />
            <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.1} transparent opacity={0.68} />
          </mesh>
          <mesh position={[0.36, 0, 0]}>
            <torusGeometry args={[radius * 0.6, 0.07, 12, 36]} />
            <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.2} transparent opacity={0.78} />
          </mesh>
        </group>
      )}
      {modeStyle.carrier === 'shield' && (
        <mesh ref={ref}>
          <sphereGeometry args={[radius * 0.9, 20, 20]} />
          <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={0.95} transparent opacity={0.2} wireframe />
        </mesh>
      )}
      {modeStyle.carrier === 'hex' && (
        <mesh ref={ref}>
          <cylinderGeometry args={[radius * 0.8, radius * 0.8, radius * 0.95, 6]} />
          <meshStandardMaterial color={movingColor} emissive={movingColor} emissiveIntensity={1.2} transparent opacity={0.72} wireframe />
        </mesh>
      )}
      <Text position={[0, 0.9, 0]} color="#dff6ff" fontSize={0.34} anchorX="center" anchorY="middle">
        {`${prediction.currentToken.token} (${(prediction.currentToken.prob * 100).toFixed(1)}%)`}
      </Text>
    </group>
  );
}

function ModeVisualOverlay({ mode = 'static', prediction = null }) {
  const ref = useRef(null);
  const modeStyle = MODE_VISUALS[mode] || MODE_VISUALS.static;

  useFrame((state) => {
    if (!ref.current) {
      return;
    }
    ref.current.rotation.y = state.clock.elapsedTime * (0.25 + modeStyle.nodeSpeed * 0.2);
  });

  if (mode === 'static') {
    return null;
  }

  const z = ((prediction?.layerProgress ?? 0.5) - 0.5) * (LAYER_COUNT - 1) * 0.92;
  return (
    <group ref={ref} position={[0, 0, z]}>
      {mode === 'causal_intervention' && (
        <mesh>
          <torusKnotGeometry args={[1.2, 0.08, 120, 16]} />
          <meshStandardMaterial color={modeStyle.accent} emissive={modeStyle.accent} emissiveIntensity={0.95} transparent opacity={0.45} wireframe />
        </mesh>
      )}
      {mode === 'subspace_geometry' && (
        <mesh rotation={[0.62, 0.15, 0.42]}>
          <boxGeometry args={[3.6, 0.05, 1.6]} />
          <meshStandardMaterial color={modeStyle.accent} emissive={modeStyle.accent} emissiveIntensity={0.8} transparent opacity={0.28} />
        </mesh>
      )}
      {mode === 'feature_decomposition' && (
        <>
          <Line points={[[-1.9, 0, 0], [1.9, 0, 0]]} color="#f59e0b" transparent opacity={0.8} lineWidth={2} />
          <Line points={[[0, -1.9, 0], [0, 1.9, 0]]} color="#38bdf8" transparent opacity={0.8} lineWidth={2} />
          <Line points={[[0, 0, -1.9], [0, 0, 1.9]]} color="#a78bfa" transparent opacity={0.8} lineWidth={2} />
        </>
      )}
      {mode === 'cross_layer_transport' && (
        <>
          <Line points={[[0, 0, -2.8], [0, 0, 2.8]]} color={modeStyle.accent} transparent opacity={0.85} lineWidth={2} />
          <mesh position={[0, 0.2, Math.sin((prediction?.layerProgress || 0) * Math.PI * 2) * 2.2]}>
            <sphereGeometry args={[0.16, 12, 12]} />
            <meshStandardMaterial color={modeStyle.accent} emissive={modeStyle.accent} emissiveIntensity={1.35} />
          </mesh>
        </>
      )}
      {mode === 'compositionality' && (
        <>
          <mesh rotation={[0, 0, 0]}>
            <torusGeometry args={[1.2, 0.05, 12, 42]} />
            <meshStandardMaterial color="#34d399" emissive="#34d399" emissiveIntensity={1.0} transparent opacity={0.62} />
          </mesh>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[1.0, 0.05, 12, 42]} />
            <meshStandardMaterial color="#f59e0b" emissive="#f59e0b" emissiveIntensity={1.0} transparent opacity={0.62} />
          </mesh>
          <mesh rotation={[0, Math.PI / 2, 0]}>
            <torusGeometry args={[0.8, 0.05, 12, 42]} />
            <meshStandardMaterial color="#60a5fa" emissive="#60a5fa" emissiveIntensity={1.0} transparent opacity={0.62} />
          </mesh>
        </>
      )}
      {mode === 'counterfactual' && (
        <>
          <mesh position={[-0.8, 0, 0]}>
            <sphereGeometry args={[0.42, 16, 16]} />
            <meshStandardMaterial color="#fda4af" emissive="#fda4af" emissiveIntensity={1.05} transparent opacity={0.58} />
          </mesh>
          <mesh position={[0.8, 0, 0]}>
            <sphereGeometry args={[0.42, 16, 16]} />
            <meshStandardMaterial color="#fb7185" emissive="#fb7185" emissiveIntensity={1.2} transparent opacity={0.58} />
          </mesh>
          <Line points={[[-0.4, 0, 0], [0.4, 0, 0]]} color="#fda4af" transparent opacity={0.85} lineWidth={2} />
        </>
      )}
      {mode === 'robustness' && (
        <mesh>
          <sphereGeometry args={[1.45, 24, 24]} />
          <meshStandardMaterial color={modeStyle.accent} emissive={modeStyle.accent} emissiveIntensity={0.72} transparent opacity={0.16} wireframe />
        </mesh>
      )}
      {mode === 'minimal_circuit' && (
        <>
          <mesh>
            <cylinderGeometry args={[1.2, 1.2, 1.6, 6]} />
            <meshStandardMaterial color={modeStyle.accent} emissive={modeStyle.accent} emissiveIntensity={0.9} transparent opacity={0.26} wireframe />
          </mesh>
          <Line points={[[0, 0.8, 0], [0, -0.8, 0]]} color={modeStyle.accent} transparent opacity={0.9} lineWidth={2} />
        </>
      )}
    </group>
  );
}

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

function TheoryBeacon({
  position = [0, 0, 0],
  color = '#ffffff',
  size = 0.14,
  pulse = 0.18,
  speed = 1.2,
  phase = 0,
  opacity = 0.94,
}) {
  const ref = useRef(null);
  useFrame((state) => {
    if (!ref.current) {
      return;
    }
    const s = 1 + Math.sin(state.clock.elapsedTime * speed + phase) * pulse;
    ref.current.scale.set(size * s, size * s, size * s);
  });
  return (
    <mesh ref={ref} position={position}>
      <sphereGeometry args={[1, 14, 14]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.1} transparent opacity={opacity} />
    </mesh>
  );
}

function TheoryRunner({
  path = [],
  color = '#ffffff',
  size = 0.12,
  speed = 0.28,
  phase = 0,
}) {
  const ref = useRef(null);
  useFrame((state) => {
    if (!ref.current || path.length < 2) {
      return;
    }
    const t = (state.clock.elapsedTime * speed + phase) % 1;
    const scaled = t * (path.length - 1);
    const idx = Math.min(path.length - 2, Math.floor(scaled));
    const frac = scaled - idx;
    const pos = blendPosition(path[idx], path[idx + 1], frac);
    ref.current.position.set(pos[0], pos[1], pos[2]);
    const s = 1 + Math.sin(state.clock.elapsedTime * 3.2 + phase * 4) * 0.16;
    ref.current.scale.set(size * s, size * s, size * s);
  });
  return (
    <mesh ref={ref}>
      <sphereGeometry args={[1, 12, 12]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.2} />
    </mesh>
  );
}

function TheoryObjectOverlay({ theoryObjectMeta = null, prediction = null, nodes = [], selected = null }) {
  const ref = useRef(null);
  const accent = theoryObjectMeta?.color || '#7dd3fc';
  const label = theoryObjectMeta?.labelZh || '理论对象';
  const z = ((prediction?.layerProgress ?? 0.5) - 0.5) * (LAYER_COUNT - 1) * 0.92;
  const focusNodeSet = useMemo(() => new Set(prediction?.focusNodeIds || []), [prediction?.focusNodeIds]);
  const coreNodes = useMemo(() => nodes.filter((node) => node.role !== 'background'), [nodes]);
  const focusNodes = useMemo(() => coreNodes.filter((node) => focusNodeSet.has(node.id)), [coreNodes, focusNodeSet]);
  const familyNodes = useMemo(() => coreNodes.filter((node) => ['macro', 'fruitGeneral', 'query'].includes(node.role)), [coreNodes]);
  const sectionNodes = useMemo(() => coreNodes.filter((node) => ['micro', 'query', 'macro'].includes(node.role)), [coreNodes]);
  const routeNodes = useMemo(() => coreNodes.filter((node) => node.role === 'route'), [coreNodes]);
  const attributeNodes = useMemo(() => coreNodes.filter((node) => ['style', 'logic', 'syntax'].includes(node.role)), [coreNodes]);
  const protocolNodes = useMemo(() => coreNodes.filter((node) => ['unifiedDecode', 'route', 'query'].includes(node.role)), [coreNodes]);
  const familyPatchView = useMemo(
    () => buildFamilyPatchViewModel(coreNodes, selected, null),
    [coreNodes, selected]
  );

  const fallbackCenter = useMemo(() => [0, -2.6, z], [z]);
  const familyCenter = useMemo(() => averagePosition(familyNodes, fallbackCenter), [familyNodes, fallbackCenter]);
  const sectionCenter = useMemo(() => averagePosition(focusNodes.length > 0 ? focusNodes : sectionNodes, familyCenter), [focusNodes, sectionNodes, familyCenter]);
  const routeCenter = useMemo(() => averagePosition(routeNodes, shiftPosition(sectionCenter, 0, 0, 1.2)), [routeNodes, sectionCenter]);
  const attributeCenter = useMemo(() => averagePosition(attributeNodes, shiftPosition(sectionCenter, 0, 0.4, 0)), [attributeNodes, sectionCenter]);
  const protocolCenter = useMemo(() => averagePosition(protocolNodes, shiftPosition(routeCenter, 0.8, 0, 0)), [protocolNodes, routeCenter]);
  const selectedCenter = Array.isArray(selected?.position) ? selected.position : sectionCenter;
  const offsetVector = useMemo(() => {
    const raw = [
      selectedCenter[0] - familyCenter[0] + 0.2,
      selectedCenter[1] - familyCenter[1] + 0.1,
      selectedCenter[2] - familyCenter[2],
    ];
    return normalizeVector(raw, 1.45);
  }, [familyCenter, selectedCenter]);
  const offsetTarget = useMemo(
    () => shiftPosition(sectionCenter, offsetVector[0], offsetVector[1], offsetVector[2]),
    [offsetVector, sectionCenter]
  );
  const readoutPort = useMemo(() => shiftPosition(protocolCenter, 5.4, 1.2, 0), [protocolCenter]);
  const bridgePorts = useMemo(
    () => [
      shiftPosition(protocolCenter, 5.2, 2.2, -1.2),
      shiftPosition(protocolCenter, 5.5, 0, 0),
      shiftPosition(protocolCenter, 5.2, -2.2, 1.2),
    ],
    [protocolCenter]
  );
  const stagePath = useMemo(
    () => [
      shiftPosition(familyCenter, -1.5, -0.9, -2.8),
      shiftPosition(sectionCenter, -0.4, 0.15, -0.9),
      shiftPosition(routeCenter, 0.4, -0.1, 1.1),
      shiftPosition(protocolCenter, 1.4, 0.8, 2.8),
    ],
    [familyCenter, protocolCenter, routeCenter, sectionCenter]
  );
  const successorPath = useMemo(
    () => [
      shiftPosition(sectionCenter, -1.25, -0.75, -0.8),
      sectionCenter,
      offsetTarget,
      shiftPosition(offsetTarget, 1.35, 0.82, 0.9),
    ],
    [offsetTarget, sectionCenter]
  );

  useFrame((state) => {
    if (!ref.current) {
      return;
    }
    ref.current.rotation.z = state.clock.elapsedTime * 0.12;
    ref.current.rotation.y = state.clock.elapsedTime * 0.16;
  });

  if (!theoryObjectMeta?.id) {
    return null;
  }

  return (
    <group ref={ref}>
      {theoryObjectMeta.id === 'family_patch' && (
        <>
          <mesh position={familyCenter}>
            <ringGeometry args={[0.95, 1.45, 40]} />
            <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={0.8} transparent opacity={0.3} side={2} />
          </mesh>
          <mesh position={familyCenter} rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[1.18, 0.06, 12, 48]} />
            <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={0.7} transparent opacity={0.24} />
          </mesh>
          <mesh position={shiftPosition(familyCenter, 0, 0, 0.08)}>
            <circleGeometry args={[0.72, 30]} />
            <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={0.4} transparent opacity={0.14} side={2} />
          </mesh>
          <Line points={[familyCenter, sectionCenter]} color={accent} transparent opacity={0.62} lineWidth={1.8} />
          <mesh position={familyPatchView.conceptCenter} rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[0.78, 0.04, 12, 40]} />
            <meshStandardMaterial color="#f8b4ff" emissive="#f8b4ff" emissiveIntensity={0.92} transparent opacity={0.34} />
          </mesh>
          <mesh position={familyPatchView.siblingCenter} rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[0.56, 0.03, 12, 36]} />
            <meshStandardMaterial color="#a7f3d0" emissive="#a7f3d0" emissiveIntensity={0.58} transparent opacity={0.22} />
          </mesh>
          <Line points={[familyCenter, familyPatchView.conceptCenter]} color="#f8b4ff" transparent opacity={0.88} lineWidth={2.2} />
          <Line points={[familyCenter, familyPatchView.siblingCenter]} color="#a7f3d0" transparent opacity={0.4} lineWidth={1.5} />
          {familyPatchView.prototypeWitness.slice(0, 4).map((node, idx) => (
            <Line
              key={`family-proto-${node.id}`}
              points={[familyCenter, node.position]}
              color={idx < 2 ? '#dff6ff' : accent}
              transparent
              opacity={0.56}
              lineWidth={idx < 2 ? 1.8 : 1.2}
            />
          ))}
          {familyPatchView.instanceWitness.slice(0, 4).map((node, idx) => (
            <Line
              key={`family-instance-${node.id}`}
              points={[familyPatchView.conceptCenter, node.position]}
              color={idx < 2 ? '#f8b4ff' : '#fda4af'}
              transparent
              opacity={0.62}
              lineWidth={idx < 2 ? 1.8 : 1.2}
            />
          ))}
          <TheoryBeacon position={familyCenter} color={accent} size={0.18} pulse={0.24} speed={1.1} phase={0.2} />
          <TheoryBeacon position={familyPatchView.conceptCenter} color="#f8b4ff" size={0.12} pulse={0.18} speed={1.3} phase={0.6} />
          <TheoryBeacon position={familyPatchView.siblingCenter} color="#a7f3d0" size={0.09} pulse={0.15} speed={1.05} phase={1.0} />
          <TheoryBeacon position={shiftPosition(familyCenter, 1.25, 0.3, 0.2)} color="#dff6ff" size={0.08} phase={0.7} />
          <TheoryBeacon position={shiftPosition(familyCenter, -1.1, -0.4, -0.1)} color="#dff6ff" size={0.08} phase={1.2} />
          <Text position={shiftPosition(familyCenter, 0, 1.05, 0)} color="#dff6ff" fontSize={0.18} anchorX="center" anchorY="middle">
            {'family prototype'}
          </Text>
          <Text position={shiftPosition(familyPatchView.conceptCenter, 0, 0.86, 0)} color="#f8b4ff" fontSize={0.16} anchorX="center" anchorY="middle">
            {'instance offset'}
          </Text>
        </>
      )}
      {theoryObjectMeta.id === 'concept_section' && (
        <>
          <mesh position={sectionCenter} rotation={[0.4, 0.2, 0.1]}>
            <boxGeometry args={[2.7, 0.08, 1.15]} />
            <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={0.78} transparent opacity={0.28} />
          </mesh>
          <mesh position={offsetTarget} rotation={[0.4, -0.16, -0.12]}>
            <boxGeometry args={[1.25, 0.06, 0.76]} />
            <meshStandardMaterial color="#f8b4ff" emissive="#f8b4ff" emissiveIntensity={0.92} transparent opacity={0.42} />
          </mesh>
          <Line points={[sectionCenter, offsetTarget]} color={accent} transparent opacity={0.88} lineWidth={2} />
          <TheoryBeacon position={sectionCenter} color={accent} size={0.14} phase={0.3} />
          <TheoryBeacon position={offsetTarget} color="#f8b4ff" size={0.12} phase={0.9} />
          <TheoryRunner path={[sectionCenter, offsetTarget]} color="#ffffff" size={0.08} speed={0.45} phase={0.18} />
        </>
      )}
      {theoryObjectMeta.id === 'attribute_fiber' && (
        <>
          <Line points={[shiftPosition(attributeCenter, -1.45, -0.72, 0), shiftPosition(attributeCenter, 1.45, 0.72, 0)]} color="#34d399" transparent opacity={0.88} lineWidth={2} />
          <Line points={[shiftPosition(attributeCenter, -1.42, 0.72, 0), shiftPosition(attributeCenter, 1.42, -0.72, 0)]} color="#60a5fa" transparent opacity={0.76} lineWidth={2} />
          <Line points={[shiftPosition(attributeCenter, 0, -1.0, -0.5), shiftPosition(attributeCenter, 0, 1.0, 0.5)]} color={accent} transparent opacity={0.74} lineWidth={2} />
          <TheoryBeacon position={attributeCenter} color={accent} size={0.12} phase={0.2} />
          <TheoryRunner path={[shiftPosition(attributeCenter, -1.45, -0.72, 0), attributeCenter, shiftPosition(attributeCenter, 1.45, 0.72, 0)]} color="#34d399" size={0.07} speed={0.34} phase={0.12} />
          <TheoryRunner path={[shiftPosition(attributeCenter, -1.42, 0.72, 0), attributeCenter, shiftPosition(attributeCenter, 1.42, -0.72, 0)]} color="#60a5fa" size={0.07} speed={0.32} phase={0.48} />
        </>
      )}
      {theoryObjectMeta.id === 'relation_context_fiber' && (
        <>
          <mesh position={routeCenter} rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[1.25, 0.08, 12, 48]} />
            <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={0.88} transparent opacity={0.36} />
          </mesh>
          <Line points={[sectionCenter, routeCenter, protocolCenter]} color={accent} transparent opacity={0.9} lineWidth={2} />
          <TheoryBeacon position={routeCenter} color={accent} size={0.15} phase={0.4} />
          <TheoryRunner path={[sectionCenter, routeCenter, protocolCenter]} color="#dff6ff" size={0.08} speed={0.38} phase={0.16} />
          <TheoryRunner path={[protocolCenter, routeCenter, sectionCenter]} color="#8be9ff" size={0.07} speed={0.24} phase={0.56} />
        </>
      )}
      {theoryObjectMeta.id === 'admissible_update' && (
        <>
          <mesh position={sectionCenter}>
            <sphereGeometry args={[1.2, 22, 22]} />
            <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={0.65} transparent opacity={0.12} wireframe />
          </mesh>
          <mesh position={sectionCenter}>
            <sphereGeometry args={[0.78, 18, 18]} />
            <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={0.42} transparent opacity={0.08} />
          </mesh>
          <Line points={[shiftPosition(sectionCenter, -0.8, 0, 0), sectionCenter, offsetTarget]} color={accent} transparent opacity={0.88} lineWidth={2} />
          <TheoryBeacon position={sectionCenter} color={accent} size={0.12} phase={0.22} />
          <TheoryBeacon position={offsetTarget} color="#ffffff" size={0.08} phase={0.84} />
          <TheoryRunner path={[sectionCenter, offsetTarget]} color={accent} size={0.07} speed={0.42} phase={0.28} />
        </>
      )}
      {theoryObjectMeta.id === 'restricted_readout' && (
        <>
          {focusNodes.slice(0, 6).map((node, idx) => (
            <Line key={`readout-line-${node.id}`} points={[node.position, readoutPort]} color={idx < 2 ? '#ffffff' : accent} transparent opacity={0.72} lineWidth={idx < 2 ? 2.2 : 1.6} />
          ))}
          <mesh position={readoutPort} rotation={[0, 0, -Math.PI / 2]}>
            <coneGeometry args={[0.85, 1.8, 4]} />
            <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={0.92} transparent opacity={0.24} wireframe />
          </mesh>
          <mesh position={shiftPosition(readoutPort, 0.72, 0, 0)}>
            <sphereGeometry args={[0.18, 12, 12]} />
            <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={1.2} />
          </mesh>
          <TheoryRunner path={[sectionCenter, readoutPort]} color="#ffffff" size={0.08} speed={0.54} phase={0.26} />
        </>
      )}
      {theoryObjectMeta.id === 'stage_conditioned_transport' && (
        <>
          {stagePath.map((pos, idx) => (
            <mesh key={`stage-gate-${idx}`} position={pos} rotation={[Math.PI / 2, 0, 0]}>
              <torusGeometry args={[0.55 + idx * 0.08, 0.04, 12, 36]} />
              <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={0.9} transparent opacity={0.34} />
            </mesh>
          ))}
          <Line points={stagePath} color={accent} transparent opacity={0.88} lineWidth={2} />
          <TheoryRunner path={stagePath} color="#dff6ff" size={0.08} speed={0.36} phase={0.12} />
          <TheoryRunner path={stagePath} color={accent} size={0.07} speed={0.26} phase={0.62} />
        </>
      )}
      {theoryObjectMeta.id === 'successor_aligned_transport' && (
        <>
          <Line points={successorPath} color={accent} transparent opacity={0.9} lineWidth={2.2} />
          <mesh position={successorPath[successorPath.length - 1]} rotation={[0, 0, -Math.PI / 3]}>
            <coneGeometry args={[0.18, 0.42, 12]} />
            <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={1.1} />
          </mesh>
          <TheoryRunner path={successorPath} color="#fff7d6" size={0.08} speed={0.48} phase={0.14} />
          <TheoryRunner path={successorPath} color={accent} size={0.07} speed={0.28} phase={0.52} />
        </>
      )}
      {theoryObjectMeta.id === 'protocol_bridge' && (
        <>
          <mesh position={protocolCenter}>
            <cylinderGeometry args={[1.1, 1.1, 0.26, 6]} />
            <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={0.82} transparent opacity={0.26} wireframe />
          </mesh>
          <Line points={[protocolCenter, readoutPort]} color="#fde68a" transparent opacity={0.88} lineWidth={2} />
          {bridgePorts.map((port, idx) => (
            <group key={`bridge-port-${idx}`}>
              <mesh position={port}>
                <boxGeometry args={[0.36, 0.36, 0.36]} />
                <meshStandardMaterial color="#fde68a" emissive="#fde68a" emissiveIntensity={0.92} transparent opacity={0.78} />
              </mesh>
              <Line points={[protocolCenter, port]} color={accent} transparent opacity={0.82} lineWidth={1.8} />
            </group>
          ))}
          <TheoryRunner path={[sectionCenter, protocolCenter, readoutPort]} color="#ffffff" size={0.08} speed={0.42} phase={0.18} />
          <TheoryRunner path={[protocolCenter, bridgePorts[0], bridgePorts[1], bridgePorts[2]]} color={accent} size={0.07} speed={0.26} phase={0.66} />
        </>
      )}
      <Text position={shiftPosition(protocolCenter, 0, -2.1, 0)} color="#dff6ff" fontSize={0.26} anchorX="center" anchorY="middle">
        {label}
      </Text>
    </group>
  );
}

function averageScenePosition(nodes = []) {
  if (!Array.isArray(nodes) || nodes.length === 0) return [0, 0, 0];
  const total = nodes.reduce((acc, node) => {
    const position = Array.isArray(node?.position) ? node.position : [0, 0, 0];
    acc[0] += position[0] || 0;
    acc[1] += position[1] || 0;
    acc[2] += position[2] || 0;
    return acc;
  }, [0, 0, 0]);
  return total.map((value) => value / nodes.length);
}

function LanguageResearchSceneOverlay({ languageFocus = DEFAULT_LANGUAGE_FOCUS, nodes = [], selected = null }) {
  const overlays = Array.isArray(languageFocus?.structureOverlays) ? languageFocus.structureOverlays : [];
  const sceneCenter = useMemo(() => averageScenePosition(nodes), [nodes]);
  const layerMeta = LANGUAGE_RESEARCH_LAYER_META[languageFocus?.researchLayer] || LANGUAGE_RESEARCH_LAYER_META.static_encoding;
  const selectedPosition = Array.isArray(selected?.position) ? selected.position : sceneCenter;
  const roleLabel = languageFocus?.roleGroup || 'object';
  const taskLabel = languageFocus?.taskGroup || 'translation';
  const riskLabel = LANGUAGE_RISK_META[languageFocus?.riskFocus] || '风险焦点未定义';

  return (
    <group>
      <Html position={[-11.8, 10.2, 0]} transform sprite>
        <div style={{
          minWidth: 260,
          padding: '12px 14px',
          borderRadius: 14,
          background: 'rgba(10, 16, 28, 0.88)',
          border: `1px solid ${layerMeta.color}`,
          boxShadow: `0 0 24px ${layerMeta.color}33`,
          color: '#e8f0ff',
          backdropFilter: 'blur(10px)',
        }}>
          <div style={{ fontSize: 13, fontWeight: 800, color: layerMeta.color, marginBottom: 6 }}>
            {layerMeta.label}
          </div>
          <div style={{ fontSize: 11, lineHeight: 1.6, color: '#b4c4dd', marginBottom: 8 }}>
            {`对象组: ${languageFocus?.objectGroup || 'fruit'} | 任务组: ${taskLabel} | 角色组: ${roleLabel}`}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
            {overlays.map((item) => {
              const meta = LANGUAGE_OVERLAY_META[item] || { label: item, color: '#94a3b8' };
              return (
                <span key={item} style={{
                  padding: '3px 8px',
                  borderRadius: 999,
                  fontSize: 10,
                  color: '#f8fbff',
                  background: `${meta.color}22`,
                  border: `1px solid ${meta.color}`,
                }}>
                  {meta.label}
                </span>
              );
            })}
          </div>
          <div style={{ fontSize: 11, color: '#ffd7d7' }}>
            {`风险焦点: ${riskLabel}`}
          </div>
        </div>
      </Html>

      <Text position={[0, 13.8, 0]} color={layerMeta.color} fontSize={0.42} anchorX="center" anchorY="middle">
        {`${layerMeta.label} / ${LANGUAGE_RISK_META[languageFocus?.riskFocus] || '风险焦点'}`}
      </Text>

      {overlays.includes('shared_base') && (
        <group position={sceneCenter}>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[4.8, 0.08, 24, 120]} />
            <meshStandardMaterial color={LANGUAGE_OVERLAY_META.shared_base.color} emissive={LANGUAGE_OVERLAY_META.shared_base.color} emissiveIntensity={1.1} transparent opacity={0.42} />
          </mesh>
        </group>
      )}

      {overlays.includes('local_delta') && selected && (
        <group position={selectedPosition}>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[0.72, 0.05, 18, 60]} />
            <meshStandardMaterial color={LANGUAGE_OVERLAY_META.local_delta.color} emissive={LANGUAGE_OVERLAY_META.local_delta.color} emissiveIntensity={1.4} transparent opacity={0.72} />
          </mesh>
          <mesh position={[0, 0.45, 0]}>
            <sphereGeometry args={[0.12, 20, 20]} />
            <meshStandardMaterial color="#fff0d6" emissive={LANGUAGE_OVERLAY_META.local_delta.color} emissiveIntensity={1.4} />
          </mesh>
        </group>
      )}

      {overlays.includes('path_amplification') && selected && (
        <>
          <Line
            points={[
              selectedPosition,
              [selectedPosition[0] + 2.4, selectedPosition[1] + 1.4, selectedPosition[2] + 1.1],
              [selectedPosition[0] + 4.2, selectedPosition[1] + 2.6, selectedPosition[2] + 1.8],
            ]}
            color={LANGUAGE_OVERLAY_META.path_amplification.color}
            transparent
            opacity={0.82}
            lineWidth={2.4}
          />
          <Text position={[selectedPosition[0] + 4.6, selectedPosition[1] + 2.9, selectedPosition[2] + 2]} color="#d9ffe8" fontSize={0.22}>
            路径放大
          </Text>
        </>
      )}

      {overlays.includes('semantic_roles') && (
        <group position={[sceneCenter[0], sceneCenter[1] + 3.2, sceneCenter[2]]}>
          <Text position={[-2.4, 0.2, 0]} color="#d8c4ff" fontSize={0.2}>对象</Text>
          <Text position={[-0.8, 0.9, 0]} color="#d8c4ff" fontSize={0.2}>属性</Text>
          <Text position={[0.8, 0.9, 0]} color="#d8c4ff" fontSize={0.2}>位置</Text>
          <Text position={[2.4, 0.2, 0]} color="#d8c4ff" fontSize={0.2}>操作</Text>
          <Text position={[-0.8, -0.7, 0]} color="#d8c4ff" fontSize={0.2}>约束</Text>
          <Text position={[0.8, -0.7, 0]} color="#d8c4ff" fontSize={0.2}>结果</Text>
        </group>
      )}

      {overlays.includes('fidelity') && (
        <group position={[sceneCenter[0], sceneCenter[1] + 0.2, sceneCenter[2]]}>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[5.8, 0.07, 20, 100]} />
            <meshStandardMaterial color={LANGUAGE_OVERLAY_META.fidelity.color} emissive={LANGUAGE_OVERLAY_META.fidelity.color} emissiveIntensity={1.1} transparent opacity={0.34} />
          </mesh>
          <Text position={[0, -2.2, 0]} color="#ffd6de" fontSize={0.24} anchorX="center">
            来源保真风险带
          </Text>
        </group>
      )}
    </group>
  );
}

function ConceptAssociationOverlay({ conceptAssociationState = null }) {
  const layers = conceptAssociationState?.layers || [];
  const relations = conceptAssociationState?.relations || [];
  if (!conceptAssociationState || !layers.length) {
    return null;
  }

  const summaryAnchor = averagePosition(
    layers.map((layer) => ({ position: layer.anchorPosition })),
    [0, 0, 0]
  );

  return (
    <group>
      <Text
        position={shiftPosition(summaryAnchor, 0, 2.2, 0)}
        color="#eff6ff"
        fontSize={0.22}
        anchorX="center"
        anchorY="middle"
      >
        {`${conceptAssociationState.conceptLabel} · ${conceptAssociationState.categoryLabel} · 六层关联`}
      </Text>

      {relations.map((relation, index) => (
        <group key={`concept-association-relation-${relation.id}`}>
          <Line
            points={relation.points}
            color={relation.color}
            transparent
            opacity={0.2 + relation.strength * 0.45}
            lineWidth={1.6 + relation.strength * 1.6}
          />
          <TheoryRunner
            path={relation.points}
            color={relation.color}
            size={0.05 + relation.strength * 0.04}
            speed={0.18 + relation.strength * 0.2}
            phase={index * 0.17}
          />
          <Text
            position={blendPosition(relation.points[0], relation.points[1], 0.5)}
            color="#dbeafe"
            fontSize={0.11}
            anchorX="center"
            anchorY="middle"
          >
            {`${relation.label} ${Math.round(relation.strength * 100)}%`}
          </Text>
        </group>
      ))}

      {layers.map((layer, index) => (
        <group key={`concept-association-layer-${layer.id}`} position={layer.anchorPosition}>
          <WaveRing
            position={[0, 0, 0]}
            color={layer.color}
            baseRadius={0.42 + Math.min(0.4, layer.nodeCount * 0.06)}
            speed={0.8 + index * 0.08}
            phase={index * 0.26}
            opacity={0.18}
          />
          <PulseColumn
            position={[0, 0.34, 0]}
            color={layer.color}
            height={0.55 + layer.avgSignal * 0.7}
            radius={0.05}
            speed={1.0 + index * 0.06}
            phase={index * 0.19}
            opacity={0.38}
          />
          <TheoryBeacon
            position={[0, 0.78, 0]}
            color={layer.color}
            size={0.05 + Math.max(0.06, layer.avgSignal * 0.08)}
            pulse={0.14}
            speed={1.1 + index * 0.05}
            phase={index * 0.21}
            opacity={0.96}
          />
          <Text position={[0, 1.15, 0]} color={layer.color} fontSize={0.14} anchorX="center" anchorY="middle">
            {layer.label}
          </Text>
          <Text position={[0, -0.62, 0]} color="#dbeafe" fontSize={0.1} anchorX="center" anchorY="middle">
            {`${layer.topNodeLabel} · ${layer.nodeCount} 节点`}
          </Text>
        </group>
      ))}

      {layers.flatMap((layer) => layer.nodes.slice(0, 4).map((node, index) => (
        <group key={`concept-association-node-${layer.id}-${node.id}-${index}`} position={node.position}>
          <mesh>
            <sphereGeometry args={[Math.max(0.12, toSafeNumber(node.size, 0.2) * 0.48), 14, 14]} />
            <meshStandardMaterial
              color={layer.color}
              emissive={layer.color}
              emissiveIntensity={0.92}
              transparent
              opacity={0.08 + Math.max(0.08, layer.avgSignal * 0.12)}
              wireframe
            />
          </mesh>
        </group>
      )))}
    </group>
  );
}

export function AppleNeuronSceneContent({
  nodes,
  links,
  selected,
  onSelect,
  prediction = null,
  mode = 'static',
  theoryObjectMeta = null,
  dimensionLayerProfile = [],
  activeDimension = 'style',
  dimensionCausal = null,
  nodeDisplayEmphasis = {},
  puzzleCompareState = null,
  conceptAssociationState = null,
  animationMode = 'none',
  scanMechanismData = null,
  languageFocus = DEFAULT_LANGUAGE_FOCUS,
  displayLevels = null,
  basicPlayback = false,
  basicPlaybackStep = 1,
  layerSweepStep = 0,
  showAlgorithmConceptCore = false,
  showAlgorithmStaticEncoding = false,
  onBasicStart = () => {},
  onBasicStop = () => {},
  onBasicReplay = () => {},
  showDNNLayers = true,
  visibleComponents = ['attention', 'ffn', 'layer_norm'],
  animProgress = 1,
  activeScenario = null,
  activeSubView = null,
}) {
  const activationMap = prediction?.activationMap || {};
  const focusNodeIds = prediction?.focusNodeIds || [];
  const focusNodeSet = useMemo(() => new Set(focusNodeIds), [focusNodeIds]);
  const modeStyle = MODE_VISUALS[mode] || MODE_VISUALS.static;
  const animationProfile = useMemo(
    () => buildAnimationSceneProfile(nodes, selected, animationMode, scanMechanismData),
    [animationMode, nodes, scanMechanismData, selected]
  );
  const runtimeLayerKey = LAYER_PARAMETER_STATE_ORDER.includes(languageFocus?.researchLayer)
    ? languageFocus.researchLayer
    : 'static_encoding';
  const runtimeProfile = LAYER_PARAMETER_STATE_OVERLAY[runtimeLayerKey] || LAYER_PARAMETER_STATE_OVERLAY.static_encoding;
  const shouldRenderParameterStateOverlay = Boolean(
    displayLevels?.parameter_state !== false
    && (
      runtimeLayerKey !== 'static_encoding'
      || showAlgorithmStaticEncoding
      || languageFocus?.selectedRepairReplaySlotId
    )
  );
  const predictionActiveLayer = Number.isFinite(prediction?.layerProgress)
    ? prediction.layerProgress * (LAYER_COUNT - 1)
    : null;
  const combinedNodeEmphasis = useMemo(
    () => Object.fromEntries(
      nodes.map((node) => {
        const baseEmphasis = toSafeNumber(nodeDisplayEmphasis?.[node.id], 1);
        const animationEmphasis = toSafeNumber(animationProfile.emphasisMap?.[node.id], 1);
        return [node.id, baseEmphasis * animationEmphasis];
      })
    ),
    [animationProfile.emphasisMap, nodeDisplayEmphasis, nodes]
  );

  const visibleNodes = useMemo(
    () => nodes.filter((node) => (
      toSafeNumber(combinedNodeEmphasis?.[node.id], 1) > 0.025
      && (showAlgorithmConceptCore || !(node.nodeGroup === 'concept_core' || String(node.id || '').startsWith('apple-core-')))
      && (showAlgorithmStaticEncoding || !['style', 'logic', 'syntax'].includes(node.role))
      && isNodeVisibleByDisplayLevels(node, displayLevels)
      && (showDNNLayers || !['query', 'route', 'unifiedDecode', 'style', 'logic', 'syntax', 'background'].includes(node.role))
      && (visibleComponents.includes(node.role) || !['attention', 'ffn', 'layer_norm', 'residual'].includes(node.role))
    )),
    [combinedNodeEmphasis, displayLevels, nodes, showAlgorithmConceptCore, showAlgorithmStaticEncoding, showDNNLayers, visibleComponents]
  );
  const visibleNodeIdSet = useMemo(() => new Set(visibleNodes.map((n) => n.id)), [visibleNodes]);
  const visibleLinks = useMemo(
    () => links
      .filter((link) => visibleNodeIdSet.has(link?.from) && visibleNodeIdSet.has(link?.to))
      .map((link) => ({
        ...link,
        emphasis: (
          toSafeNumber(combinedNodeEmphasis?.[link?.from], 1)
          + toSafeNumber(combinedNodeEmphasis?.[link?.to], 1)
        ) / 2,
      })),
    [combinedNodeEmphasis, links, visibleNodeIdSet]
  );
  const puzzleCompareVisibleLinks = useMemo(
    () => visibleLinks
      .filter((link) => (puzzleCompareState?.sceneLinkHighlightMap || puzzleCompareState?.linkHighlightMap)?.[link.id])
      .map((link) => ({
        ...link,
        compareMeta: (puzzleCompareState?.sceneLinkHighlightMap || puzzleCompareState?.linkHighlightMap)[link.id],
      })),
    [puzzleCompareState, visibleLinks]
  );
  const puzzleCompareVisibleNodes = useMemo(
    () => visibleNodes
      .filter((node) => (puzzleCompareState?.sceneNodeCategoryMap || puzzleCompareState?.nodeCategoryMap)?.[node.id])
      .map((node) => ({
        ...node,
        compareMeta: (puzzleCompareState?.sceneNodeHighlightMap || puzzleCompareState?.nodeHighlightMap)[node.id],
      })),
    [puzzleCompareState, visibleNodes]
  );


  const activeParameterIndex = basicPlayback
    ? Math.max(0, Math.min(runtimeProfile.nodes.length - 1, basicPlaybackStep - 1))
    : -1;
  const runtimeActiveLayer = activeParameterIndex >= 0
    ? runtimeProfile.nodes[activeParameterIndex]?.layer ?? null
    : null;
  const activeLayer = Number.isFinite(predictionActiveLayer)
    ? predictionActiveLayer
    : Number.isFinite(runtimeActiveLayer)
      ? runtimeActiveLayer
      : layerSweepStep;
  const showAdvancedOverlays = Boolean(displayLevels?.advanced_analysis);
  const motionEnabled = Boolean(
    basicPlayback
    || prediction?.isRunning
    || (showAdvancedOverlays && animationMode !== 'none')
  );

  return (
    <>
      <LayerGuides activeLayer={activeLayer} />

      {displayLevels?.mechanism_chain !== false && visibleLinks.map((link) => (
        <Line
          key={link.id}
          points={link.points}
          color={mode === 'dynamic_prediction' || mode === 'static' ? link.color : modeStyle.accent}
          transparent
          opacity={(0.24 + (prediction?.isRunning ? 0.18 : 0) + modeStyle.linkOpacityBoost) * toSafeNumber(link.emphasis, 1)}
          lineWidth={(1.1 + modeStyle.linkWidthBoost) * (0.8 + toSafeNumber(link.emphasis, 1) * 0.55)}
        />
      ))}

      {puzzleCompareVisibleLinks.map((link) => (
        <Line
          key={`puzzle-compare-${link.id}`}
          points={link.points}
          color={link.compareMeta.color}
          transparent
          opacity={link.compareMeta.opacity}
          lineWidth={link.compareMeta.lineWidth}
        />
      ))}

      {visibleNodes.map((node) => (
        <PulsingNeuron
          key={node.id}
          node={node}
          selected={selected?.id === node.id}
          onSelect={onSelect}
          predictionStrength={activationMap[node.id] || 0}
          mode={mode}
          isEffectiveNode={focusNodeSet.has(node.id)}
          visibilityEmphasis={toSafeNumber(combinedNodeEmphasis?.[node.id], 1)}
          motionEnabled={motionEnabled}
        />
      ))}

      {puzzleCompareVisibleNodes.map((node) => (
        <group key={`puzzle-compare-node-${node.id}`} position={node.position}>
          <mesh>
            <sphereGeometry args={[Math.max(0.18, toSafeNumber(node.size, 0.55) * 0.34), 18, 18]} />
            <meshStandardMaterial
              color={node.compareMeta.color}
              emissive={node.compareMeta.color}
              emissiveIntensity={1.15}
              transparent
              opacity={node.compareMeta.opacity * 0.28}
              wireframe
            />
          </mesh>
        </group>
      ))}

      {showAdvancedOverlays ? <ModeVisualOverlay mode={mode} prediction={prediction} /> : null}
      <ConceptAssociationOverlay conceptAssociationState={conceptAssociationState} />
      {showAdvancedOverlays ? <LanguageResearchSceneOverlay languageFocus={languageFocus} nodes={visibleNodes} selected={selected} /> : null}
      {showAdvancedOverlays ? <TheoryObjectOverlay theoryObjectMeta={theoryObjectMeta} prediction={prediction} nodes={visibleNodes} selected={selected} /> : null}
      {showAdvancedOverlays ? (
        <AppleNeuronAnimationOverlay
          animationMode={animationMode}
          nodes={visibleNodes}
          selected={selected}
          prediction={prediction}
          scanMechanismData={scanMechanismData}
        />
      ) : null}

      {showAdvancedOverlays ? <TokenPredictionCarrier prediction={prediction} mode={mode} /> : null}
      {showAdvancedOverlays ? <LayerEffectiveNeuronOverlay prediction={prediction} mode={mode} /> : null}

      {/* 动画场景进度叠加 (始终显示，不依赖 showAdvancedOverlays) */}
      {activeScenario && ANIMATION_SCENARIOS[activeScenario] && (() => {
        const scenario = ANIMATION_SCENARIOS[activeScenario];
        const phase = scenario.phases.find(p => animProgress >= p.start && animProgress < p.end) || scenario.phases[scenario.phases.length - 1];
        if (!phase) return null;
        const layerMin = phase.layerRange[0];
        const layerMax = phase.layerRange[1];
        // 过滤在该层范围内的节点
        const highlightNodes = visibleNodes.filter(n => {
          const layer = n.layer ?? n.position?.[1];
          return layer !== undefined && layer >= layerMin && layer <= layerMax;
        });
        return (
          <group>
            <Text
              position={[0, 12, 0]}
              fontSize={0.8}
              color={modeStyle.accent}
              anchorX="center"
              anchorY="middle"
            >
              {scenario.icon} {phase.label} (L{layerMin}-L{layerMax})
            </Text>
            {highlightNodes.slice(0, 60).map(node => (
              <mesh key={`anim-hl-${node.id}`} position={node.position}>
                <ringGeometry args={[0.55, 0.7, 24]} />
                <meshBasicMaterial color={modeStyle.accent} transparent opacity={0.35} side={2} />
              </mesh>
            ))}
            {/* 进度条可视化 */}
            <group position={[-8, 11, 0]}>
              <mesh>
                <boxGeometry args={[16, 0.15, 0.05]} />
                <meshBasicMaterial color="#333" transparent opacity={0.5} />
              </mesh>
              <mesh position={[animProgress * 8 - 8, 0, 0.01]}>
                <boxGeometry args={[16 * animProgress, 0.15, 0.05]} />
                <meshBasicMaterial color={modeStyle.accent} transparent opacity={0.8} />
              </mesh>
              {/* 阶段分段线 */}
              {scenario.phases.map((p, i) => (
                <mesh key={`phase-line-${i}`} position={[p.end * 16 - 8, 0, 0.02]}>
                  <boxGeometry args={[0.03, 0.25, 0.05]} />
                  <meshBasicMaterial color="#888" transparent opacity={0.6} />
                </mesh>
              ))}
            </group>
          </group>
        );
      })()}

      {/* 维度视角标签叠加 (始终显示) */}
      {activeDimension && DIMENSION_VIEWS[activeDimension] && !activeScenario && (
        <Text
          position={[0, 13, 0]}
          fontSize={0.6}
          color={DIMENSION_VIEWS[activeDimension].color}
          anchorX="center"
          anchorY="middle"
        >
          {DIMENSION_VIEWS[activeDimension].icon} {DIMENSION_VIEWS[activeDimension].label}
          {activeSubView && ` → ${DIMENSION_VIEWS[activeDimension].subViews[activeSubView]?.label || ''}`}
        </Text>
      )}

      {shouldRenderParameterStateOverlay ? (
        <LayerParameterStateOverlay
          languageFocus={languageFocus}
          selected={selected}
          onSelect={onSelect}
          activeIndex={activeParameterIndex}
          isPlaying={basicPlayback}
        />
      ) : null}
      {showAdvancedOverlays ? <DimensionLayerImpactGraph profile={dimensionLayerProfile} dimension={activeDimension} suppression={dimensionCausal} /> : null}

      {puzzleCompareState?.summary ? (
        <Html position={[0, 7.9, 0]} center>
          <div
            style={{
              padding: '8px 10px',
              borderRadius: 10,
              background: 'rgba(10, 14, 26, 0.82)',
              border: '1px solid rgba(148, 163, 184, 0.26)',
              color: '#e2e8f0',
              fontSize: 10,
              lineHeight: 1.55,
              whiteSpace: 'nowrap',
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 2 }}>双拼图差异高亮</div>
            <div>{`共享节点 ${puzzleCompareState.summary.sharedNodes} | 主独有 ${puzzleCompareState.summary.primaryOnlyNodes} | 对比独有 ${puzzleCompareState.summary.compareOnlyNodes}`}</div>
            <div>{`局部链路回放 ${puzzleCompareState.summary.localReplayLinks} 条`}</div>
            {puzzleCompareState.replaySlotFocus ? (
              <div>{`回放槽位 ${puzzleCompareState.replaySlotFocus.label} | 阶段 ${puzzleCompareState.replaySlotFocus.activePhaseLabel} | 聚焦链路 ${puzzleCompareState.summary.slotFocusedLinks} 条`}</div>
            ) : null}
            {puzzleCompareState.validation ? (
              <div>{`裁剪验证 ${puzzleCompareState.validation.label} | 最小性 ${Math.round(puzzleCompareState.validation.minimalityScore * 100)}%`}</div>
            ) : null}
          </div>
        </Html>
      ) : null}

      {selected && selected.role !== 'background' && (
        <Html position={[selected.position[0], selected.position[1] + 1.25, selected.position[2]]} center>
          <div
            style={{
              padding: '8px 10px',
              borderRadius: 8,
              background: 'rgba(255,255,255,0.95)',
              border: '1px solid rgba(180, 198, 228, 0.85)',
              color: '#1f2937',
              fontSize: 11,
              whiteSpace: 'nowrap',
              pointerEvents: 'none',
            }}
          >
            <div>
              {selected.detailType === 'apple_switch_unit'
                ? `${selected.label} | ${selected.roleLabel || '-'} | L${selected.actualLayer}`
                : `${selected.label} | L${selected.layer}N${selected.neuron}`}
            </div>
            {selected.detailType === 'parameter_state' ? (
              <div style={{ marginTop: 4, fontSize: 10, color: '#334155', lineHeight: 1.45 }}>
                <div>{`参数维度: d${selected.dimIndex}`}</div>
                <div>{`来源阶段: ${selected.sourceStage}`}</div>
              </div>
            ) : null}
            {selected.detailType === 'apple_switch_unit' ? (
              <div style={{ marginTop: 4, fontSize: 10, color: '#334155', lineHeight: 1.45 }}>
                <div>{`类型: ${selected.unitTypeLabel || '-'}`}</div>
                <div>{`方向: ${selected.directionLabel || '-'}`}</div>
              </div>
            ) : null}
          </div>
        </Html>
      )}
    </>
  );
}

function AppleNeuronScene({
  nodes,
  links,
  selected,
  onSelect,
  prediction,
  mode = 'static',
  theoryObjectMeta = null,
  dimensionLayerProfile = [],
  activeDimension = 'style',
  dimensionCausal = null,
  nodeDisplayEmphasis = {},
  puzzleCompareState = null,
  conceptAssociationState = null,
  animationMode = 'none',
  scanMechanismData = null,
  languageFocus = DEFAULT_LANGUAGE_FOCUS,
  displayLevels = null,
  basicPlayback = false,
  basicPlaybackStep = 1,
  layerSweepStep = 0,
  showAlgorithmConceptCore = false,
  showAlgorithmStaticEncoding = false,
  onBasicStart = () => {},
  onBasicStop = () => {},
  onBasicReplay = () => {},
}) {
  return (
    <Canvas shadows dpr={[1, 1.5]}>
      <color attach="background" args={['#090b15']} />
      <fog attach="fog" args={['#090b15', 14, 42]} />

      <ambientLight intensity={0.5} />
      <pointLight position={[12, 12, 16]} intensity={70} color="#8fc4ff" />
      <pointLight position={[-14, -8, -15]} intensity={30} color="#ff9e6b" />

      <PerspectiveCamera makeDefault position={[16, 12, 26]} fov={42} />
      <OrbitControls enablePan enableZoom minDistance={10} maxDistance={44} />

      <AppleNeuronSceneContent
        nodes={nodes}
        links={links}
        selected={selected}
        onSelect={onSelect}
        prediction={prediction}
        mode={mode}
        theoryObjectMeta={theoryObjectMeta}
        dimensionLayerProfile={dimensionLayerProfile}
        activeDimension={activeDimension}
        dimensionCausal={dimensionCausal}
        nodeDisplayEmphasis={nodeDisplayEmphasis}
        puzzleCompareState={puzzleCompareState}
        conceptAssociationState={conceptAssociationState}
        animationMode={animationMode}
        scanMechanismData={scanMechanismData}
        languageFocus={languageFocus}
        displayLevels={displayLevels}
        basicPlayback={basicPlayback}
        basicPlaybackStep={basicPlaybackStep}
        layerSweepStep={layerSweepStep}
        showAlgorithmConceptCore={showAlgorithmConceptCore}
        showAlgorithmStaticEncoding={showAlgorithmStaticEncoding}
        onBasicStart={onBasicStart}
        onBasicStop={onBasicStop}
        onBasicReplay={onBasicReplay}
      />
    </Canvas>
  );
}

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

export function AppleNeuronMainScene({ workspace, sceneHeight = '74vh' }) {
  return (
    <div
      style={{
        height: sceneHeight,
        borderRadius: 18,
        border: '1px solid rgba(122, 162, 255, 0.28)',
        overflow: 'hidden',
        background: 'radial-gradient(circle at 20% 0%, rgba(43, 84, 165, 0.2), rgba(8, 10, 18, 0.95) 55%)',
        boxShadow: '0 18px 44px rgba(0,0,0,0.45)',
      }}
    >
      <AppleNeuronScene
        nodes={workspace.nodes}
        links={workspace.links}
        selected={workspace.selected}
        onSelect={workspace.setSelected}
        prediction={workspace.prediction}
        mode={workspace.analysisMode}
        theoryObjectMeta={workspace.currentTheoryObject}
        dimensionLayerProfile={workspace.multidimLayerProfile}
        activeDimension={workspace.multidimActiveDimension}
        dimensionCausal={workspace.multidimCausalData}
        nodeDisplayEmphasis={workspace.nodeDisplayEmphasis}
        puzzleCompareState={workspace.puzzleCompareState}
        conceptAssociationState={workspace.conceptAssociationState}
        animationMode={workspace.animationMode}
        scanMechanismData={workspace.scanMechanismData}
        languageFocus={workspace.languageFocus}
        displayLevels={workspace.displayLevels}
        basicPlayback={workspace.basicRuntimePlaying}
        basicPlaybackStep={workspace.basicRuntimeStep}
        layerSweepStep={workspace.layerSweepStep}
        showAlgorithmConceptCore={workspace.showAlgorithmConceptCore}
        showAlgorithmStaticEncoding={workspace.showAlgorithmStaticEncoding}
        onBasicStart={workspace.handleBasicRuntimeStart}
        onBasicStop={workspace.handleBasicRuntimeStop}
        onBasicReplay={workspace.handleBasicRuntimeReplay}
      />
    </div>
  );
}

const smallActionButtonStyle = {
  borderRadius: 8,
  border: '1px solid rgba(122, 162, 255, 0.5)',
  background: 'rgba(28, 53, 102, 0.75)',
  color: '#dbe9ff',
  fontSize: 12,
  padding: '7px 10px',
  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 6,
};

const panelCardStyle = {
  borderRadius: 14,
  padding: 14,
  border: '1px solid rgba(118, 170, 255, 0.25)',
  background: 'linear-gradient(170deg, rgba(15,24,42,0.94), rgba(7,12,25,0.95))',
};

const inputStyle = {
  width: '100%',
  borderRadius: 8,
  border: '1px solid rgba(122, 162, 255, 0.3)',
  background: 'rgba(7, 12, 25, 0.8)',
  color: '#dbe9ff',
  padding: '8px 10px',
  fontSize: 12,
};

const textAreaStyle = {
  width: '100%',
  borderRadius: 8,
  border: '1px solid rgba(122, 162, 255, 0.3)',
  background: 'rgba(7, 12, 25, 0.8)',
  color: '#dbe9ff',
  padding: '8px 10px',
  fontSize: 12,
  resize: 'vertical',
};

const fixedFileControlWidth = 240;
const inferScanOptionConcept = (fileMeta) => {
  const lower = String(fileMeta?.name || fileMeta?.path || '').toLowerCase();
  if (lower.includes('unified_math_structure_decode')) return '统一解码';
  if (lower.includes('agi_research_stage_bundle_manifest')) return '阶段实验总清单';
  if (lower.includes('agi_four_tasks_suite_manifest')) return '四任务总清单';
  if (lower.includes('multidim_encoding_probe')) return '三维编码探针';
  if (lower.includes('multidim_causal_ablation')) return '三维因果消融';
  if (lower.includes('multidim_multiseed_stability')) return '三维多 Seed 稳定性';
  if (lower.includes('minimal_causal_circuit_search')) return '最小因果子回路';
  if (lower.includes('variable_binding_hard_verification')) return '变量绑定验证';
  if (lower.includes('unified_coordinate_system_test')) return '统一坐标测试';
  if (lower.includes('concept_family_parallel_scale')) return '概念族并行尺度';
  if (lower.includes('dynamic_binding_stress_test')) return '动态绑定压力测试';
  if (lower.includes('long_horizon_causal_trace_test')) return '长程因果链路';
  if (lower.includes('local_credit_assignment_proxy_test')) return '局部信用代理测试';
  if (lower.includes('triplet_targeted_causal_scan')) return '三元组定向因果';
  if (lower.includes('triplet_targeted_multiseed_stability')) return '三元组多 Seed 稳定性';
  if (lower.includes('mass_noun') || lower.includes('noun_scan') || lower.includes('encoding_scan')) return '名词编码扫描';
  return String(fileMeta?.name || '研究资产');
};

function AppleSwitchMechanismInsightsPanel({ workspace, compact = false }) {
  const appleSwitchMechanismData = workspace?.appleSwitchMechanismData || null;
  const selected = workspace?.selected || null;
  const setSelected = workspace?.setSelected || null;
  const nodes = workspace?.nodes || [];
  const [filterModel, setFilterModel] = useState('all');
  const [filterKind, setFilterKind] = useState('all');
  const [filterDirection, setFilterDirection] = useState('all');
  const [filterCircuit, setFilterCircuit] = useState('all');
  const [filterKeyword, setFilterKeyword] = useState('');

  if (!isAppleSwitchMechanismPayload(appleSwitchMechanismData)) {
    return null;
  }

  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;
  const modelEntries = Object.entries(appleSwitchMechanismData.models || {});
  const filteredModelEntries = modelEntries.filter(([modelKey]) => filterModel === 'all' || modelKey === filterModel);
  const normalizedKeyword = String(filterKeyword || '').trim().toLowerCase();
  const activeModelKey = selected?.detailType === 'apple_switch_unit'
    ? selected.modelKey
    : (filterModel !== 'all' ? filterModel : (appleSwitchMechanismData.models?.deepseek7b ? 'deepseek7b' : modelEntries[0]?.[0]));
  const activeModel = appleSwitchMechanismData.models?.[activeModelKey] || filteredModelEntries[0]?.[1] || modelEntries[0]?.[1] || null;
  const selectedUnit = selected?.detailType === 'apple_switch_unit' ? selected.appleSwitchUnit : null;
  const nodeByUnitId = Object.fromEntries(
    (Array.isArray(nodes) ? nodes : [])
      .filter((node) => node?.detailType === 'apple_switch_unit')
      .map((node) => [node.unitId, node])
  );

  const filterUnits = (units = []) => units.filter((unit) => {
    if (filterKind !== 'all' && unit?.kind !== filterKind) {
      return false;
    }
    if (filterDirection !== 'all') {
      const lateMean = Number(unit?.signed_effect?.late_mean_signed_contrast_switch_coupling || 0);
      const resolvedDirection = lateMean > 0 ? 'reverse' : 'forward';
      if (resolvedDirection !== filterDirection) {
        return false;
      }
    }
    if (filterCircuit === 'in' && !unit?.is_final_circuit_member) {
      return false;
    }
    if (filterCircuit === 'out' && unit?.is_final_circuit_member) {
      return false;
    }
    if (normalizedKeyword) {
      const haystack = [
        unit?.unit_id,
        unit?.role,
        getAppleSwitchUnitRoleLabel(unit?.role),
        unit?.kind === 'mlp_neuron' ? 'mlp 神经元' : '注意力头',
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();
      if (!haystack.includes(normalizedKeyword)) {
        return false;
      }
    }
    return true;
  });

  return (
    <div style={{ display: 'grid', gap: 10 }}>
      <div style={cardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>苹果切换机制总览</div>
        <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7, marginBottom: 8 }}>
          <div>{`统一资产: 已导入`}</div>
          <div>{`峰值层匹配率: ${(toSafeNumber(appleSwitchMechanismData?.aggregate_stability?.peak_layer_match_rate, 0) * 100).toFixed(1)}%`}</div>
          <div>{`当前聚焦模型: ${activeModel?.model_name || activeModelKey || '-'}`}</div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: compact ? 'repeat(2, minmax(0, 1fr))' : 'repeat(5, minmax(0, 1fr))', gap: 8, marginBottom: 10 }}>
          <select value={filterModel} onChange={(e) => setFilterModel(e.target.value)} style={inputStyle}>
            <option value="all">全部模型</option>
            {modelEntries.map(([modelKey, modelPayload]) => (
              <option key={`apple-switch-filter-model-${modelKey}`} value={modelKey}>{modelPayload?.model_name || modelKey}</option>
            ))}
          </select>
          <select value={filterKind} onChange={(e) => setFilterKind(e.target.value)} style={inputStyle}>
            <option value="all">全部类型</option>
            <option value="attention_head">注意力头</option>
            <option value="mlp_neuron">MLP 神经元</option>
          </select>
          <select value={filterDirection} onChange={(e) => setFilterDirection(e.target.value)} style={inputStyle}>
            <option value="all">全部方向</option>
            <option value="forward">正向支撑</option>
            <option value="reverse">反向校正</option>
          </select>
          <select value={filterCircuit} onChange={(e) => setFilterCircuit(e.target.value)} style={inputStyle}>
            <option value="all">全部回路成员</option>
            <option value="in">仅最小回路内</option>
            <option value="out">仅最小回路外</option>
          </select>
          <input
            value={filterKeyword}
            onChange={(e) => setFilterKeyword(e.target.value)}
            placeholder="关键词: H:2:26 / 锚点 / 头"
            style={inputStyle}
          />
        </div>
        <div style={{ fontSize: 11, color: '#7ea2c9', lineHeight: 1.7, marginBottom: 8 }}>
          {`筛选方式: 模型 + 类型 + 方向 + 最小回路成员 + 关键词，可同时生效。`}
        </div>
        <div style={{ display: 'grid', gap: 8 }}>
          {filteredModelEntries.map(([modelKey, modelPayload]) => {
            const visibleUnits = filterUnits(modelPayload?.core_units || []);
            return (
            <div
              key={`apple-switch-model-${modelKey}`}
              style={{
                borderRadius: 10,
                border: `1px solid ${APPLE_SWITCH_MODEL_COLORS[modelKey] || '#6ea8ff'}55`,
                padding: '8px 10px',
                background: 'rgba(255,255,255,0.03)',
              }}
            >
              <div style={{ fontSize: 12, fontWeight: 700, color: APPLE_SWITCH_MODEL_COLORS[modelKey] || '#dbeafe' }}>
                {modelPayload?.model_name || modelKey}
              </div>
              <div style={{ marginTop: 4, fontSize: 11, color: '#9bb3de', lineHeight: 1.7 }}>
                <div>{`敏感层: L${toSafeNumber(modelPayload?.best_sensitive_layer?.layer_index, 0)}`}</div>
                <div>{`共享底座最强层: L${toSafeNumber(modelPayload?.best_shared_layer?.layer_index, 0)}`}</div>
                <div>{`核心单元: ${Array.isArray(modelPayload?.core_units) ? modelPayload.core_units.length : 0}`}</div>
                <div>{`筛选后单元: ${visibleUnits.length}`}</div>
                <div>{`最小回路规模: ${Array.isArray(modelPayload?.effective_circuit?.final_subset) ? modelPayload.effective_circuit.final_subset.length : 0}`}</div>
              </div>
              <div style={{ marginTop: 8, display: 'grid', gap: 6 }}>
                {visibleUnits.slice(0, compact ? 4 : 8).map((unit) => {
                  const node = nodeByUnitId[unit.unit_id];
                  const clickable = typeof setSelected === 'function' && node;
                  return (
                    <button
                      key={`apple-switch-unit-${modelKey}-${unit.unit_id}`}
                      type="button"
                      onClick={() => clickable && setSelected(node)}
                      style={{
                        width: '100%',
                        textAlign: 'left',
                        borderRadius: 8,
                        border: '1px solid rgba(255,255,255,0.1)',
                        padding: '6px 8px',
                        background: node?.id === selected?.id ? 'rgba(56,189,248,0.16)' : 'rgba(255,255,255,0.02)',
                        color: '#dbeafe',
                        cursor: clickable ? 'pointer' : 'default',
                      }}
                    >
                      <div style={{ fontSize: 11, fontWeight: 700 }}>{unit.unit_id}</div>
                      <div style={{ marginTop: 2, fontSize: 10, color: '#9bb3de' }}>
                        {`${getAppleSwitchUnitRoleLabel(unit.role)} | 有效 ${toSafeNumber(unit?.scores?.effective_score, 0).toFixed(3)} | ${unit?.signed_effect?.direction_label || '-'}`}
                      </div>
                      <div style={{ marginTop: 2, fontSize: 10, color: '#7ea2c9' }}>
                        {`${unit.kind === 'mlp_neuron' ? 'MLP 神经元' : '注意力头'} | ${unit.is_final_circuit_member ? '最小回路内' : '最小回路外'}`}
                      </div>
                    </button>
                  );
                })}
                {visibleUnits.length === 0 ? (
                  <div style={{ fontSize: 11, color: '#8ea5c5', lineHeight: 1.6 }}>
                    当前筛选条件下没有命中单元，可以放宽一个条件再看。
                  </div>
                ) : null}
              </div>
            </div>
          );
          })}
        </div>
      </div>

      {activeModel ? (
        <div style={cardStyle}>
          <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>
            {selectedUnit ? `${selected.unitId} 逐层过程` : `${activeModel.model_name || activeModelKey} 层过程时间线`}
          </div>
          <div style={{ display: 'grid', gap: 6, maxHeight: compact ? 220 : 320, overflowY: 'auto' }}>
            {selectedUnit ? (
              (selectedUnit?.process_timeline || []).map((row) => {
                const signedValue = toSafeNumber(row?.signed_contrast_switch_coupling, 0);
                const relativeDrop = toSafeNumber(row?.relative_separation_drop, 0);
                const signedWidth = Math.min(100, Math.abs(signedValue) * 650);
                const relativeWidth = Math.min(100, Math.abs(relativeDrop) * 1800);
                return (
                  <div key={`apple-switch-process-${selected.unitId}-${row.layer_index}`} style={{ display: 'grid', gap: 4 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, fontSize: 11, color: '#dbeafe' }}>
                      <span>{`L${row.layer_index}`}</span>
                      <span>{row.direction_label}</span>
                    </div>
                    <div style={{ height: 5, borderRadius: 999, background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
                      <div style={{ width: `${signedWidth}%`, height: '100%', background: signedValue <= 0 ? '#38bdf8' : '#fb7185' }} />
                    </div>
                    <div style={{ height: 4, borderRadius: 999, background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
                      <div style={{ width: `${relativeWidth}%`, height: '100%', background: '#f59e0b' }} />
                    </div>
                    <div style={{ fontSize: 10, color: '#8ea5c5' }}>
                      {`signed=${signedValue.toFixed(4)} | relative_drop=${relativeDrop.toFixed(4)} | pc1=${toSafeNumber(row?.pc1_explained_variance_ratio, 0).toFixed(3)}`}
                    </div>
                  </div>
                );
              })
            ) : (
              (activeModel?.layer_summary || []).map((row) => {
                const sharedWidth = Math.min(100, toSafeNumber(row?.shared_active_neuron_count, 0) * 4);
                const splitWidth = Math.min(100, Math.abs(toSafeNumber(row?.excess_switch_drop, 0)) * 1800);
                return (
                  <div key={`apple-switch-layer-summary-${activeModelKey}-${row.layer_index}`} style={{ display: 'grid', gap: 4 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, fontSize: 11, color: '#dbeafe' }}>
                      <span>{`L${row.layer_index}`}</span>
                      <span>{row.process_label}</span>
                    </div>
                    <div style={{ height: 5, borderRadius: 999, background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
                      <div style={{ width: `${sharedWidth}%`, height: '100%', background: '#34d399' }} />
                    </div>
                    <div style={{ height: 4, borderRadius: 999, background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
                      <div style={{ width: `${splitWidth}%`, height: '100%', background: '#f59e0b' }} />
                    </div>
                    <div style={{ fontSize: 10, color: '#8ea5c5' }}>
                      {`共享=${toSafeNumber(row?.shared_active_neuron_count, 0)} | 分裂强度=${toSafeNumber(row?.excess_switch_drop, 0).toFixed(4)} | Jaccard=${toSafeNumber(row?.active_jaccard, 0).toFixed(3)}`}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}

const buildScanContentLabel = (data, fileMeta) => {
  const nounRecords = Array.isArray(data?.noun_records) ? data.noun_records : [];
  if (nounRecords.length > 0) {
    const pairs = nounRecords
      .slice(0, 2)
      .map((row) => {
        const noun = String(row?.noun || '').trim();
        const category = String(row?.category || '未分类').trim() || '未分类';
        if (!noun) {
          return null;
        }
        return `${noun}-${category}`;
      })
      .filter(Boolean);
    if (pairs.length > 0) {
      return pairs.join(' / ');
    }
  }

  if (data?.experiment_id && HARD_PROBLEM_EXPERIMENT_LABELS[data.experiment_id]) {
    return HARD_PROBLEM_EXPERIMENT_LABELS[data.experiment_id];
  }
  if (data?.suite_id === 'agi_four_tasks_suite_v1') {
    return '四任务验证';
  }
  if (data?.bundle_id === 'agi_research_stage_bundle_v1') {
    return '阶段实验总清单';
  }
  if (isAppleSwitchMechanismPayload(data)) {
    return '苹果切换机制';
  }
  if (isUnifiedDecodePayload(data)) {
    return '风格-逻辑-语法';
  }
  if (data?.dimensions?.style && data?.dimensions?.logic && data?.dimensions?.syntax) {
    return '风格-逻辑-语法';
  }
  return inferScanOptionConcept(fileMeta);
};

const formatScanOptionLabel = (fileMeta, contentLabel = '') => {
  const conceptLabel = contentLabel || inferScanOptionConcept(fileMeta);
  const mtime = String(fileMeta?.mtime_iso || '').slice(0, 19).replace('T', ' ');
  return mtime ? `${conceptLabel} | ${mtime}` : conceptLabel;
};

export function AppleNeuronEncodingInfoPanels({ workspace, compact = false }) {
  const nodes = workspace?.nodes || [];
  const summary = workspace?.summary || {};
  const metrics = workspace?.modeMetrics || [];
  const multidimProbe = workspace?.multidimProbeData || null;
  const multidimCausal = workspace?.multidimCausalData || null;
  const hardProblemResults = workspace?.hardProblemResults || {};
  const unifiedDecodeResult = workspace?.unifiedDecodeResult || null;
  const bundleManifest = workspace?.bundleManifest || null;
  const fourTasksManifest = workspace?.fourTasksManifest || null;
  const activeDim = workspace?.multidimActiveDimension || 'style';

  const layerRows = useMemo(() => {
    const map = new Map();
    nodes.forEach((node) => {
      const key = Number.isFinite(node.layer) ? node.layer : 0;
      const row = map.get(key) || { layer: key, count: 0, strength: 0 };
      row.count += 1;
      row.strength += Number(node.strength || 0);
      map.set(key, row);
    });
    return Array.from(map.values())
      .sort((a, b) => a.layer - b.layer)
      .map((row) => ({
        ...row,
        strength: row.count ? row.strength / row.count : 0,
      }));
  }, [nodes]);

  const maxCount = useMemo(() => Math.max(1, ...layerRows.map((row) => row.count)), [layerRows]);
  const hardProblemRows = useMemo(() => Object.entries(hardProblemResults), [hardProblemResults]);
  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;

  return (
    <div style={{ display: 'grid', gap: 10 }}>
      <AppleSwitchMechanismInsightsPanel workspace={workspace} compact={compact} />

      <div style={cardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>层级编码签名</div>
        <div style={{ display: 'grid', gap: 6, maxHeight: compact ? 130 : 170, overflowY: 'auto' }}>
          {layerRows.map((row) => (
            <div key={`layer-sign-${row.layer}`} style={{ display: 'grid', gridTemplateColumns: '44px 1fr', gap: 8, alignItems: 'center' }}>
              <div style={{ fontSize: 11, color: '#9bb3de' }}>{`L${row.layer}`}</div>
              <div style={{ display: 'grid', gap: 3 }}>
                <div style={{ height: 5, background: 'rgba(255,255,255,0.08)', borderRadius: 8, overflow: 'hidden' }}>
                  <div style={{ width: `${(row.count / maxCount) * 100}%`, height: '100%', background: '#67dfff' }} />
                </div>
                <div style={{ height: 4, background: 'rgba(255,255,255,0.08)', borderRadius: 8, overflow: 'hidden' }}>
                  <div style={{ width: `${Math.min(100, row.strength * 1000000)}%`, height: '100%', background: '#f59e0b' }} />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div style={cardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>数据机制指标</div>
        <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7 }}>
          <div>{`核心神经元: ${(summary.micro || 0) + (summary.macro || 0) + (summary.route || 0)}`}</div>
          <div>{`当前词元: ${summary.currentToken || '-'} (${((summary.currentTokenProb || 0) * 100).toFixed(1)}%)`}</div>
          <div>{`显示策略: ${summary.displayStrategy === 'auto' ? '自动聚焦' : summary.displayStrategy === 'all' ? '全部显示' : '手动筛选'}`}</div>
          <div>{`可见概念集: ${summary.visibleQuerySets || 0} / 隐藏概念集: ${summary.hiddenQuerySets || 0}`}</div>
        </div>
        {metrics.length > 0 ? (
          <div style={{ marginTop: 8, display: 'grid', gap: 4 }}>
            {metrics.map((metric, idx) => (
              <div key={`m-${metric.label}-${idx}`} style={{ fontSize: 11, color: '#9bb3de' }}>{`${metric.label}: ${metric.value}`}</div>
            ))}
          </div>
        ) : null}
        {multidimProbe ? (
          <div style={{ marginTop: 8, fontSize: 11, color: '#9bb3de', lineHeight: 1.6 }}>
            <div>{`多维探针: 已导入`}</div>
            <div>{`当前维度: ${DIMENSION_LABELS[activeDim] || activeDim}`}</div>
            {Number.isFinite(multidimCausal?.diagonal_advantage?.[activeDim]) ? (
              <div>{`对角优势: ${toSafeNumber(multidimCausal?.diagonal_advantage?.[activeDim], 0).toFixed(4)}`}</div>
            ) : (
              <div style={{ color: '#7f95bb' }}>未导入三维因果消融结果</div>
            )}
          </div>
        ) : null}
        {hardProblemRows.length > 0 ? (
          <div style={{ marginTop: 8, fontSize: 11, color: '#9bb3de', lineHeight: 1.6 }}>
            <div>{`硬伤实验导入: ${hardProblemRows.length}`}</div>
            {hardProblemRows.map(([expId, payload]) => {
              const title = HARD_PROBLEM_EXPERIMENT_LABELS[expId] || payload?.title || expId;
              const mm = payload?.metrics || {};
              if (expId === 'hard_problem_dynamic_binding_v1') {
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 稳定性=${toSafeNumber(mm.binding_stability_index, 0).toFixed(3)} | 交换错率=${toSafeNumber(mm.role_swap_error_rate, 0).toFixed(3)}`}
                  </div>
                );
              }
              if (expId === 'hard_problem_long_horizon_trace_v1') {
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 长程衰减=${toSafeNumber(mm.long_horizon_decay, 0).toFixed(3)} | 传输稳定=${toSafeNumber(mm.layer_transport_stability_mean, 0).toFixed(3)}`}
                  </div>
                );
              }
              if (expId === 'hard_problem_local_credit_assignment_v1') {
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 局部充分=${toSafeNumber(mm.local_sufficiency_mean, 0).toFixed(3)} | 局部选择=${toSafeNumber(mm.local_selectivity_mean, 0).toFixed(3)}`}
                  </div>
                );
              }
              if (expId === 'triplet_targeted_causal_scan_v1') {
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 三联分离=${toSafeNumber(mm.triplet_separability_index, 0).toFixed(3)} | 轴特异=${toSafeNumber(mm.axis_specificity_index, 0).toFixed(3)}`}
                  </div>
                );
              }
              if (expId === 'triplet_targeted_multiseed_stability_v1') {
                const seqMargin = toSafeNumber(mm?.global_mean_causal_margin_seq_logprob?.mean, 0);
                const posRatio = toSafeNumber(mm?.global_positive_causal_margin_ratio?.mean, 0);
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: seq边际均值=${seqMargin.toFixed(4)} | 正边际比例=${(posRatio * 100).toFixed(1)}%`}
                  </div>
                );
              }
              if (expId === 'hard_problem_variable_binding_verification_v1') {
                const meanDelta = toSafeNumber(mm?.mean_delta, 0);
                const improvedDims = toSafeNumber(mm?.improved_dimension_count, 0);
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 平均提升=${meanDelta.toFixed(4)} | 提升维度=${improvedDims}`}
                  </div>
                );
              }
              if (expId === 'minimal_causal_circuit_search_v1') {
                const drop = toSafeNumber(mm?.global?.intervention_drop_mean, 0);
                const repr = toSafeNumber(mm?.global?.reproducibility_jaccard_mean, 0);
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 干预下降=${drop.toFixed(4)} | 复现Jaccard=${repr.toFixed(4)}`}
                  </div>
                );
              }
              if (expId === 'unified_coordinate_system_test_v1') {
                const us = toSafeNumber(mm?.unified_coordinate_score, 0);
                const orth = toSafeNumber(mm?.probe_orthogonality?.orthogonality_index, 0);
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 统一分数=${us.toFixed(4)} | 正交性=${orth.toFixed(4)}`}
                  </div>
                );
              }
              if (expId === 'concept_family_parallel_scale_v1') {
                const appleShared = toSafeNumber(mm?.apple_chain_summary?.shared_base_ratio_vs_micro_union?.mean, 0);
                const catShared = toSafeNumber(mm?.cat_chain_summary?.shared_base_ratio_vs_micro_union?.mean, 0);
                return (
                  <div key={`hp-${expId}`}>
                    {`${title}: 苹果共享=${appleShared.toFixed(4)} | 猫共享=${catShared.toFixed(4)}`}
                  </div>
                );
              }
              return <div key={`hp-${expId}`}>{`${title}: 已导入`}</div>;
            })}
          </div>
        ) : null}
        {unifiedDecodeResult ? (
          <div style={{ marginTop: 8, fontSize: 11, color: '#9bb3de', lineHeight: 1.6 }}>
            <div>{`统一解码: 已导入`}</div>
            <div>{`假设通过率: ${(toSafeNumber(unifiedDecodeResult?.hypothesis_test?.pass_ratio, 0) * 100).toFixed(1)}%`}</div>
            <div>{`探针文件数: ${toSafeNumber(unifiedDecodeResult?.axis_stability?.n_probe_files, 0)}`}</div>
          </div>
        ) : null}
        {bundleManifest ? (
          <div style={{ marginTop: 8, fontSize: 11, color: '#9bb3de', lineHeight: 1.6 }}>
            <div>{`批量清单: 已导入`}</div>
            <div>{`seed=${toSafeNumber(bundleManifest?.config?.seed, 0)} | 统一解码=${bundleManifest?.config?.run_unified_decoder ? '开启' : '关闭'}`}</div>
          </div>
        ) : null}
        {fourTasksManifest ? (
          <div style={{ marginTop: 8, fontSize: 11, color: '#9bb3de', lineHeight: 1.6 }}>
            <div>{`四任务清单: 已导入`}</div>
            <div>{`all_success=${fourTasksManifest?.all_success ? 'true' : 'false'} | 任务数=${Object.keys(fourTasksManifest?.return_codes || {}).length}`}</div>
          </div>
        ) : null}
      </div>

      <div style={cardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>硬伤实验与统一编码说明</div>
        <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7, display: 'grid', gap: 6, maxHeight: compact ? 220 : 300, overflowY: 'auto' }}>
          <div style={{ color: '#cfe2ff', fontWeight: 700 }}>硬伤实验节点（红/蓝/橙/紫）</div>
          <div>1. 来源：导入 `agi_research_result.v1` 的实验指标。</div>
          <div>2. 映射：每个关键指标生成一个节点，颜色代表实验类型。</div>
          <div>3. 位置：层号与神经元号由指标哈希映射到主 3D 空间（用于对比，不代表真实单一神经元定位）。</div>
          <div>4. 大小与亮度：由指标强度决定；误差类指标（error/collision/decay）按“越小越好”反向映射。</div>
          <div style={{ color: '#cfe2ff', fontWeight: 700, marginTop: 4 }}>统一编码节点（绿色）</div>
          <div>1. 来源：`unified_math_structure_decode.json` 的融合结果。</div>
          <div>2. 映射：按 style / logic / syntax 三个维度生成节点簇。</div>
          <div>3. 层位：优先使用 dominant layer pattern，把节点放到对应层；无模式时使用回退层。</div>
          <div>4. 强度：综合 `profile_cosine_mean` 与 `diagonal_advantage`，用于显示“轴稳定 + 因果可分离”程度。</div>
          <div style={{ color: '#cfe2ff', fontWeight: 700, marginTop: 4 }}>读图顺序（建议）</div>
          <div>1. 先看图例颜色区分实验类型。</div>
          <div>2. 再看模型说明中的指标数值（均值/通过率）。</div>
          <div>3. 最后点选节点查看 `metric/value/source`，判断该可视化是“证据节点”还是“结构节点”。</div>
        </div>
      </div>
    </div>
  );
}

export function AppleNeuronResearchAssetInfoPanel({ workspace, compact = false }) {
  const selectedScanPath = workspace?.selectedScanPath || '';
  const scanPreviewData = workspace?.scanPreviewData || null;
  const scanPreviewLoading = workspace?.scanPreviewLoading || false;
  const scanPreviewError = workspace?.scanPreviewError || '';
  const languageFocus = workspace?.languageFocus || DEFAULT_LANGUAGE_FOCUS;
  const scanPreview = useMemo(
    () => buildArtifactPreview(scanPreviewData, selectedScanPath),
    [scanPreviewData, selectedScanPath]
  );
  const scanPreviewTheory = useMemo(
    () => THEORY_OBJECT_RESEARCH_MAP[scanPreview?.theoryObject] || THEORY_OBJECT_RESEARCH_MAP.family_patch,
    [scanPreview?.theoryObject]
  );
  const showInTopRight = shouldShowResearchAssetInTopRight(scanPreview, selectedScanPath);
  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;

  if (!showInTopRight) {
    return null;
  }

  return (
    <div style={cardStyle}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'baseline', marginBottom: 8 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff' }}>{`${scanPreview.typeLabel} | ${scanPreview.title}`}</div>
        <div style={{ fontSize: 10, color: '#7ea2c9' }}>{scanPreviewLoading ? '预览加载中...' : '预览就绪'}</div>
      </div>
      <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7 }}>{scanPreview.subtitle}</div>
      {scanPreviewError ? (
        <div style={{ marginTop: 8, fontSize: 11, color: '#ff9fb0' }}>{scanPreviewError}</div>
      ) : null}

      <div style={{ marginTop: 10, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10, display: 'grid', gap: 6 }}>
        <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>理论映射</div>
        <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7 }}>
          <div>{scanPreviewTheory.summary}</div>
          <div>{`3D 关注点：${scanPreviewTheory.sceneHint}`}</div>
          <div>{`当前研究层：${languageFocus?.researchLayer || 'static_encoding'}`}</div>
        </div>
        <div style={{ display: 'grid', gap: 4 }}>
          {scanPreviewTheory.metrics.map((item) => (
            <div key={`artifact-data-${item.label}`} style={{ display: 'grid', gridTemplateColumns: '110px 1fr', gap: 8, fontSize: 11, color: '#9bb3de' }}>
              <span>{item.label}</span>
              <span style={{ color: '#dbe9ff', fontWeight: 700 }}>{item.value}</span>
            </div>
          ))}
        </div>
        {scanPreview.analysisLines.map((line) => (
          <div key={`topright-line-${line}`} style={{ fontSize: 11, color: '#8fd4ff', lineHeight: 1.6 }}>
            {`• ${line}`}
          </div>
        ))}
      </div>
      {scanPreview.metricRows.length > 0 ? (
        <div style={{ marginTop: 10, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10, display: 'grid', gap: 4 }}>
          <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>关键指标</div>
          {scanPreview.metricRows.map((item) => (
            <div key={`preview-metric-${item.label}`} style={{ display: 'grid', gridTemplateColumns: '120px 1fr', gap: 8, fontSize: 11, color: '#9bb3de' }}>
              <span>{item.label}</span>
              <span style={{ color: '#dbe9ff', fontWeight: 700 }}>{item.value}</span>
            </div>
          ))}
        </div>
      ) : null}

      <div style={{ marginTop: 10, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10, display: 'grid', gap: 6 }}>
        <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>原始数据</div>
        <pre
          style={{
            margin: 0,
            maxHeight: compact ? 220 : 320,
            overflow: 'auto',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            fontSize: 11,
            color: '#cfe2ff',
            background: 'rgba(7, 12, 25, 0.82)',
            border: '1px solid rgba(122, 162, 255, 0.22)',
            borderRadius: 10,
            padding: 10,
          }}
        >
          {scanPreview?.rawJson || '暂无原始数据'}
        </pre>
      </div>
    </div>
  );
}

function WaveRing({
  position = [0, 0, 0],
  color = '#ffffff',
  baseRadius = 1,
  thickness = 0.04,
  speed = 1,
  phase = 0,
  opacity = 0.3,
}) {
  const ref = useRef(null);
  useFrame((state) => {
    if (!ref.current) {
      return;
    }
    const t = state.clock.elapsedTime * speed + phase;
    const pulse = 1 + Math.sin(t) * 0.16;
    ref.current.scale.set(pulse, pulse, pulse);
    ref.current.rotation.z = t * 0.18;
  });
  return (
    <mesh ref={ref} position={position} rotation={[Math.PI / 2, 0, 0]}>
      <torusGeometry args={[baseRadius, thickness, 10, 42]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.8} transparent opacity={opacity} />
    </mesh>
  );
}

function PulseColumn({
  position = [0, 0, 0],
  color = '#ffffff',
  height = 1,
  radius = 0.06,
  speed = 1,
  phase = 0,
  opacity = 0.72,
}) {
  const ref = useRef(null);
  useFrame((state) => {
    if (!ref.current) {
      return;
    }
    const t = state.clock.elapsedTime * speed + phase;
    const sy = 0.84 + (Math.sin(t) + 1) * 0.24;
    ref.current.scale.set(1, sy, 1);
  });
  return (
    <mesh ref={ref} position={position}>
      <cylinderGeometry args={[radius, radius, height, 10]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.92} transparent opacity={opacity} />
    </mesh>
  );
}

function AppleNeuronAnimationOverlay({
  animationMode = 'none',
  nodes = [],
  selected = null,
  prediction = null,
  scanMechanismData = null,
}) {
  const coreNodes = useMemo(() => (Array.isArray(nodes) ? nodes.filter((node) => node?.role !== 'background') : []), [nodes]);
  const familyView = useMemo(
    () => buildFamilyPatchViewModel(coreNodes, selected, scanMechanismData),
    [coreNodes, scanMechanismData, selected]
  );

  const familyCenter = familyView.familyCenter;
  const conceptCenter = familyView.conceptCenter;
  const siblingCenter = familyView.siblingCenter;
  const routeNodes = coreNodes.filter((node) => node.role === 'route');
  const routeCenter = averagePosition(routeNodes, shiftPosition(conceptCenter, 0.6, 0.1, 1.4));
  const protocolCenter = shiftPosition(routeCenter, 2.8, 0.6, 1.2);
  const readoutPort = shiftPosition(protocolCenter, 3.2, 0.8, 0);
  const stagePath = [
    shiftPosition(familyCenter, -1.4, -0.8, -2.2),
    shiftPosition(conceptCenter, -0.4, 0.2, -0.6),
    shiftPosition(routeCenter, 0.4, -0.2, 0.9),
    protocolCenter,
  ];
  const successorPath = [
    shiftPosition(conceptCenter, -1.0, -0.5, -0.4),
    conceptCenter,
    shiftPosition(conceptCenter, 1.1, 0.55, 0.45),
    shiftPosition(conceptCenter, 2.2, 0.95, 1.1),
  ];
  const counterfactualPathA = [
    familyCenter,
    conceptCenter,
    shiftPosition(conceptCenter, 1.2, 0.7, 0.8),
    shiftPosition(conceptCenter, 2.2, 1.2, 1.2),
  ];
  const counterfactualPathB = [
    familyCenter,
    conceptCenter,
    shiftPosition(conceptCenter, 1.05, -0.75, -0.6),
    shiftPosition(conceptCenter, 2.0, -1.25, -1.1),
  ];
  const layerRelayNodes = coreNodes
    .filter((node) => node.role === 'query')
    .slice()
    .sort((a, b) => a.layer - b.layer)
    .filter((node, idx, arr) => idx === 0 || node.layer !== arr[idx - 1].layer)
    .slice(0, 5);
  const minimalWitness = familyView.selectedConceptMinimal?.subset_flat_indices
    ? familyView.instanceWitness.slice(0, Math.min(5, familyView.selectedConceptMinimal.subset_flat_indices.length))
    : familyView.instanceWitness.slice(0, 4);
  const siblingLabel = familyView.uniqueSiblingConcepts[0] || 'sibling';
  const animationLabel = APPLE_ANIMATION_OPTIONS.find((opt) => opt.id === animationMode)?.label || '动画';

  if (animationMode === 'none' || !selected) {
    return null;
  }

  return (
    <group>
      <Text position={shiftPosition(conceptCenter, 0, 1.45, -0.1)} color="#f8fafc" fontSize={0.17} anchorX="center" anchorY="middle">
        {animationLabel}
      </Text>
      {animationMode === 'family_patch_formation' && (
        <>
          <WaveRing position={familyCenter} color="#7dd3fc" baseRadius={1.05} speed={1.0} opacity={0.22} />
          <WaveRing position={familyCenter} color="#dff6ff" baseRadius={1.45} speed={1.3} phase={0.4} opacity={0.14} />
          {familyView.prototypeWitness.slice(0, 6).map((node, idx) => (
            <TheoryRunner
              key={`anim-family-form-${node.id}`}
              path={[node.position, blendPosition(node.position, familyCenter, 0.55), familyCenter]}
              color={idx < 2 ? '#ffffff' : '#7dd3fc'}
              size={0.065}
              speed={0.24 + idx * 0.02}
              phase={idx * 0.18}
            />
          ))}
        </>
      )}
      {animationMode === 'instance_offset' && (
        <>
          <Line points={[familyCenter, conceptCenter]} color="#f8b4ff" transparent opacity={0.9} lineWidth={2.4} />
          <WaveRing position={conceptCenter} color="#f8b4ff" baseRadius={0.42} speed={1.4} opacity={0.26} />
          <TheoryRunner path={[familyCenter, conceptCenter]} color="#fff7ff" size={0.09} speed={0.44} phase={0.14} />
          <Text position={shiftPosition(conceptCenter, 0, 0.72, 0)} color="#f8b4ff" fontSize={0.14} anchorX="center" anchorY="middle">
            {'Δ concept'}
          </Text>
        </>
      )}
      {animationMode === 'attribute_fiber' && (
        <>
          <Line points={[shiftPosition(conceptCenter, -1.4, -0.7, 0), shiftPosition(conceptCenter, 1.4, 0.7, 0)]} color="#34d399" transparent opacity={0.88} lineWidth={2} />
          <Line points={[shiftPosition(conceptCenter, -1.4, 0.7, 0), shiftPosition(conceptCenter, 1.4, -0.7, 0)]} color="#60a5fa" transparent opacity={0.82} lineWidth={2} />
          <Line points={[shiftPosition(conceptCenter, 0, -1.0, -0.45), shiftPosition(conceptCenter, 0, 1.0, 0.45)]} color="#f59e0b" transparent opacity={0.8} lineWidth={2} />
          <TheoryRunner path={[shiftPosition(conceptCenter, -1.4, -0.7, 0), conceptCenter, shiftPosition(conceptCenter, 1.4, 0.7, 0)]} color="#34d399" size={0.07} speed={0.38} phase={0.08} />
          <TheoryRunner path={[shiftPosition(conceptCenter, -1.4, 0.7, 0), conceptCenter, shiftPosition(conceptCenter, 1.4, -0.7, 0)]} color="#60a5fa" size={0.07} speed={0.35} phase={0.32} />
          <TheoryBeacon position={conceptCenter} color="#f8b4ff" size={0.1} pulse={0.12} speed={1.5} phase={0.22} />
        </>
      )}
      {animationMode === 'successor_transport' && (
        <>
          <Line points={successorPath} color="#f59e0b" transparent opacity={0.92} lineWidth={2.2} />
          <TheoryRunner path={successorPath} color="#fff7d6" size={0.08} speed={0.46} phase={0.08} />
          <TheoryRunner path={successorPath} color="#f59e0b" size={0.07} speed={0.28} phase={0.42} />
        </>
      )}
      {animationMode === 'protocol_bridge' && (
        <>
          <Line points={[conceptCenter, routeCenter, protocolCenter, readoutPort]} color="#fde68a" transparent opacity={0.88} lineWidth={2.1} />
          <WaveRing position={protocolCenter} color="#fde68a" baseRadius={0.62} speed={1.18} opacity={0.2} />
          <TheoryRunner path={[conceptCenter, routeCenter, protocolCenter, readoutPort]} color="#ffffff" size={0.08} speed={0.42} phase={0.16} />
        </>
      )}
      {animationMode === 'cross_layer_relay' && (
        <>
          {layerRelayNodes.map((node, idx) => (
            <group key={`relay-${node.id}`}>
              <WaveRing position={node.position} color={idx % 2 === 0 ? '#38bdf8' : '#7dd3fc'} baseRadius={0.22 + idx * 0.02} speed={0.9 + idx * 0.16} phase={idx * 0.24} opacity={0.24} />
              {idx < layerRelayNodes.length - 1 ? (
                <Line points={[node.position, layerRelayNodes[idx + 1].position]} color="#38bdf8" transparent opacity={0.42} lineWidth={1.5} />
              ) : null}
            </group>
          ))}
          {layerRelayNodes.length > 1 ? (
            <TheoryRunner path={layerRelayNodes.map((node) => node.position)} color="#dff6ff" size={0.07} speed={0.34} phase={0.12} />
          ) : null}
        </>
      )}
      {animationMode === 'ablation_shockwave' && (
        <>
          <WaveRing position={conceptCenter} color="#fb7185" baseRadius={0.36} speed={1.7} opacity={0.28} />
          <WaveRing position={conceptCenter} color="#fb7185" baseRadius={0.72} speed={1.05} phase={0.4} opacity={0.16} />
          {familyView.instanceWitness.slice(0, 3).map((node) => (
            <Line key={`ablate-${node.id}`} points={[conceptCenter, node.position]} color="#fb7185" transparent opacity={0.6} lineWidth={1.6} />
          ))}
        </>
      )}
      {animationMode === 'counterfactual_split' && (
        <>
          <Line points={counterfactualPathA} color="#7dd3fc" transparent opacity={0.82} lineWidth={2} />
          <Line points={counterfactualPathB} color="#fb7185" transparent opacity={0.82} lineWidth={2} />
          <TheoryRunner path={counterfactualPathA} color="#dff6ff" size={0.07} speed={0.34} phase={0.08} />
          <TheoryRunner path={counterfactualPathB} color="#ffd5df" size={0.07} speed={0.34} phase={0.38} />
          <Text position={shiftPosition(counterfactualPathA[counterfactualPathA.length - 1], 0, 0.45, 0)} color="#dff6ff" fontSize={0.12}>{'actual'}</Text>
          <Text position={shiftPosition(counterfactualPathB[counterfactualPathB.length - 1], 0, -0.45, 0)} color="#ffd5df" fontSize={0.12}>{siblingLabel}</Text>
        </>
      )}
      {animationMode === 'minimal_circuit_peeloff' && (
        <>
          {minimalWitness.map((node, idx) => (
            <group key={`minimal-${node.id}`}>
              <Line points={[conceptCenter, node.position]} color={idx < 2 ? '#ffffff' : '#f97316'} transparent opacity={0.72} lineWidth={idx < 2 ? 2.0 : 1.3} />
              <WaveRing position={node.position} color="#f97316" baseRadius={0.14 + idx * 0.02} speed={0.9 + idx * 0.18} phase={idx * 0.2} opacity={0.18} />
            </group>
          ))}
        </>
      )}
      {animationMode === 'margin_breathing' && (
        <>
          <WaveRing position={familyCenter} color="#7dd3fc" baseRadius={1.05} speed={0.92} opacity={0.22} />
          <WaveRing position={familyCenter} color="#a7f3d0" baseRadius={1.72} speed={0.72} phase={0.35} opacity={0.12} />
          <Line points={[familyCenter, siblingCenter]} color="#a7f3d0" transparent opacity={0.36} lineWidth={1.4} />
        </>
      )}
      {animationMode === 'offset_sparsity' && (
        <>
          {familyView.instanceWitness.slice(0, 6).map((node, idx) => (
            <PulseColumn
              key={`offset-sparse-${node.id}`}
              position={shiftPosition(conceptCenter, -0.7 + idx * 0.28, -1.0, 0)}
              color={idx < 3 ? '#f8b4ff' : '#c084fc'}
              height={0.42 + idx * 0.18}
              radius={0.045}
              speed={0.9 + idx * 0.12}
              phase={idx * 0.2}
              opacity={0.68}
            />
          ))}
        </>
      )}
      {animationMode === 'prototype_instance_tug' && (
        <>
          <Line points={[familyCenter, conceptCenter]} color="#7dd3fc" transparent opacity={0.72} lineWidth={2} />
          <Line points={[siblingCenter, conceptCenter]} color="#f8b4ff" transparent opacity={0.72} lineWidth={2} />
          <TheoryRunner path={[familyCenter, conceptCenter]} color="#dff6ff" size={0.07} speed={0.3} phase={0.12} />
          <TheoryRunner path={[siblingCenter, conceptCenter]} color="#f8b4ff" size={0.07} speed={0.3} phase={0.46} />
          <WaveRing position={conceptCenter} color="#ffffff" baseRadius={0.28} speed={1.45} opacity={0.16} />
        </>
      )}
      {animationMode === 'stage_transition' && (
        <>
          {stagePath.map((pos, idx) => (
            <group key={`stage-transition-${idx}`}>
              <WaveRing position={pos} color={idx === 0 ? '#7dd3fc' : idx === 1 ? '#c084fc' : idx === 2 ? '#34d399' : '#fde68a'} baseRadius={0.26 + idx * 0.08} speed={0.82 + idx * 0.15} phase={idx * 0.26} opacity={0.18} />
            </group>
          ))}
          <Line points={stagePath} color="#dff6ff" transparent opacity={0.38} lineWidth={1.6} />
          <TheoryRunner path={stagePath} color="#ffffff" size={0.07} speed={0.28} phase={0.18} />
        </>
      )}
    </group>
  );
}

function AppleNeuronFamilyPatchInspector({ workspace, compact = false }) {
  const nodes = workspace?.nodes || [];
  const selected = workspace?.selected || null;
  const currentTheoryObject = workspace?.currentTheoryObject || null;
  const scanMechanismData = workspace?.scanMechanismData || null;
  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;
  const familyPatchView = useMemo(
    () => buildFamilyPatchViewModel(nodes, selected, scanMechanismData),
    [nodes, scanMechanismData, selected]
  );

  if (!selected || !['family_patch', 'concept_section'].includes(currentTheoryObject?.id || '')) {
    return null;
  }

  const minimal = familyPatchView.selectedConceptMinimal;
  const counterfactualList = familyPatchView.selectedConceptCounterfactuals || [];
  const firstCounterfactual = counterfactualList[0] || null;

  return (
    <div style={cardStyle}>
      <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>family patch 分解视图</div>
      <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7, display: 'grid', gap: 4 }}>
        <div>{`概念: ${selected?.concept || selected?.label || '-'}`}</div>
        <div>{`类别: ${selected?.category || '未分类'}`}</div>
        <div>{`family 节点数: ${familyPatchView.familyNodes.length}`}</div>
        <div>{`实例节点数: ${familyPatchView.conceptNodes.length}`}</div>
        <div>{`同族兄弟概念: ${familyPatchView.uniqueSiblingConcepts.length}`}</div>
        <div>{`offset 几何长度: ${familyPatchView.offsetNorm.toFixed(3)}`}</div>
      </div>

      <div style={{ marginTop: 10, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10, display: 'grid', gap: 6 }}>
        <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>数学解释</div>
        <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7 }}>
          <div>{`prototype: B_${selected?.category || 'family'}`}</div>
          <div>{`instance: Δ_${selected?.concept || selected?.label || 'c'}`}</div>
          <div>{'state ≈ family prototype + instance offset + attribute/context corrections'}</div>
        </div>
        <div style={{ fontSize: 11, color: '#7ea2c9', lineHeight: 1.6 }}>
          {`蓝色 ring 表示家族原型核，粉色 ring 表示当前概念的实例偏移核，浅绿色 ring 表示同族兄弟概念的相对中心。`}
        </div>
      </div>

      <div style={{ marginTop: 10, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10, display: 'grid', gap: 6 }}>
        <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>神经元见证</div>
        <div style={{ display: 'grid', gap: 4 }}>
          {familyPatchView.prototypeWitness.length > 0 ? (
            familyPatchView.prototypeWitness.slice(0, 4).map((node) => (
              <div key={`family-proto-row-${node.id}`} style={{ display: 'grid', gridTemplateColumns: '88px 1fr', gap: 8, fontSize: 11, color: '#9bb3de' }}>
                <span>prototype</span>
                <span style={{ color: '#dbe9ff', fontWeight: 700 }}>{`${node.label} | L${node.layer} N${node.neuron}`}</span>
              </div>
            ))
          ) : (
            <div style={{ fontSize: 11, color: '#7ea2c9' }}>当前没有可用的 family witness 节点。</div>
          )}
          {familyPatchView.instanceWitness.length > 0 ? (
            familyPatchView.instanceWitness.slice(0, 4).map((node) => (
              <div key={`family-inst-row-${node.id}`} style={{ display: 'grid', gridTemplateColumns: '88px 1fr', gap: 8, fontSize: 11, color: '#9bb3de' }}>
                <span>instance</span>
                <span style={{ color: '#f5d0fe', fontWeight: 700 }}>{`${node.label} | L${node.layer} N${node.neuron}`}</span>
              </div>
            ))
          ) : null}
        </div>
      </div>

      <div style={{ marginTop: 10, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10, display: 'grid', gap: 6 }}>
        <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>因果证据</div>
        <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7 }}>
          <div>{`最小回路: ${minimal ? `subset=${toSafeNumber(minimal?.subset_size, 0)} | recovery=${toSafeNumber(minimal?.recovery_ratio, 0).toFixed(3)}` : '未导入'}`}</div>
          <div>{`反事实对: ${counterfactualList.length}`}</div>
          {firstCounterfactual ? (
            <div>{`首个反事实: ${firstCounterfactual?.noun || '-'} -> ${firstCounterfactual?.counterfactual_noun || '-'} | margin=${toSafeNumber(firstCounterfactual?.specificity_margin_seq_logprob, 0).toFixed(6)}`}</div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

export function AppleNeuronSelectedLegendPanels({ workspace, compact = false }) {
  const selected = workspace?.selected || null;
  const summary = workspace?.summary || {};
  const displayStrategy = workspace?.displayStrategy || 'auto';
  const setDisplayStrategy = workspace?.setDisplayStrategy || (() => {});
  const manualDisplayGroups = workspace?.manualDisplayGroups || {};
  const setManualDisplayGroups = workspace?.setManualDisplayGroups || (() => {});
  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;

  return (
    <div style={{ display: 'grid', gap: 10 }}>
      <AppleNeuronFamilyPatchInspector workspace={workspace} compact={compact} />

      <div style={{ ...cardStyle, minHeight: 160 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>选中神经元</div>
        {selected ? (
          <div style={{ fontSize: 12, color: '#9eb4dd', display: 'grid', gap: 6 }}>
            <div style={{ color: '#e5eeff', fontWeight: 700 }}>{selected.label}</div>
            <div>{`角色: ${selected.role}`}</div>
            {'fruit' in selected ? <div>{`水果: ${selected.fruit}`}</div> : null}
            {'concept' in selected ? <div>{`概念: ${selected.concept}`}</div> : null}
            {'category' in selected ? <div>{`类别: ${selected.category}`}</div> : null}
            <div>{`层 / 神经元: L${selected.layer} / N${selected.neuron}`}</div>
            <div>{`强度: ${selected.strength.toExponential(3)}`}</div>
            <div>{`${selected.metric}: ${selected.value.toExponential(3)}`}</div>
            <div style={{ color: '#6f84ad' }}>{`来源: ${selected.source}`}</div>
          </div>
        ) : (
          <div style={{ fontSize: 12, color: '#7d93bd' }}>请在 3D 场景中点击高亮神经元。</div>
        )}
      </div>

      <div style={{ ...cardStyle, fontSize: 12, color: '#9eb4dd', lineHeight: 1.7 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>显示与降噪策略</div>
        <div style={{ display: 'grid', gridTemplateColumns: compact ? '1fr' : 'repeat(3, 1fr)', gap: 6 }}>
          {[
            { id: 'auto', label: '自动聚焦', desc: '随分析类型切换重点' },
            { id: 'all', label: '全部显示', desc: '不过滤任何节点' },
            { id: 'manual', label: '手动筛选', desc: '按类别开关显示' },
          ].map((opt) => (
            <button
              key={`legend-display-${opt.id}`}
              type="button"
              onClick={() => setDisplayStrategy(opt.id)}
              title={opt.desc}
              style={{
                borderRadius: 8,
                border: `1px solid ${displayStrategy === opt.id ? 'rgba(126, 224, 255, 0.75)' : 'rgba(122, 162, 255, 0.35)'}`,
                background: displayStrategy === opt.id ? 'rgba(24, 101, 134, 0.38)' : 'rgba(7, 12, 25, 0.82)',
                color: '#dbe9ff',
                fontSize: 11,
                padding: '7px 8px',
                cursor: 'pointer',
                textAlign: 'left',
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>
        {displayStrategy === 'manual' ? (
          <div style={{ marginTop: 8, display: 'grid', gap: 6 }}>
            {[
              { id: 'core', label: '核心/基础节点' },
              { id: 'query', label: '输入概念节点' },
              { id: 'multidim', label: '多维编码节点' },
              { id: 'hard', label: '硬伤实验节点' },
              { id: 'unified', label: '统一解码节点' },
              { id: 'background', label: '背景网络节点' },
            ].map((item) => (
              <label key={`legend-manual-group-${item.id}`} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, color: '#9eb4dd' }}>
                <input
                  type="checkbox"
                  checked={manualDisplayGroups[item.id] !== false}
                  onChange={(e) => setManualDisplayGroups((prev) => ({ ...prev, [item.id]: e.target.checked }))}
                />
                <span>{item.label}</span>
              </label>
            ))}
          </div>
        ) : null}
        <div style={{ marginTop: 8, fontSize: 11, color: '#7ea2c9' }}>
          {displayStrategy === 'auto'
            ? '自动模式：因果类分析突出硬伤实验，结构类分析突出统一编码与多维节点。'
            : displayStrategy === 'all'
              ? '全部模式：所有节点同等显示，不做降噪。'
              : '手动模式：按勾选结果控制各类节点显示。'}
        </div>
      </div>

      <div style={{ ...cardStyle, fontSize: 12, color: '#9eb4dd', lineHeight: 1.7 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>图例</div>
        <div><span style={{ color: ROLE_COLORS.micro }}>●</span> 微观神经元</div>
        <div><span style={{ color: ROLE_COLORS.macro }}>●</span> 中观神经元</div>
        <div><span style={{ color: ROLE_COLORS.route }}>●</span> 共享路径神经元</div>
        <div><span style={{ color: ROLE_COLORS.fruitGeneral }}>●</span> 类别通用神经元</div>
        <div><span style={{ color: ROLE_COLORS.style }}>●</span> 风格维度神经元</div>
        <div><span style={{ color: ROLE_COLORS.logic }}>●</span> 逻辑维度神经元</div>
        <div><span style={{ color: ROLE_COLORS.syntax }}>●</span> 句法维度神经元</div>
        <div><span style={{ color: ROLE_COLORS.hardBinding }}>●</span> 硬伤实验-动态绑定</div>
        <div><span style={{ color: ROLE_COLORS.hardLong }}>●</span> 硬伤实验-长程链路</div>
        <div><span style={{ color: ROLE_COLORS.hardLocal }}>●</span> 硬伤实验-局部信用</div>
        <div><span style={{ color: ROLE_COLORS.hardTriplet }}>●</span> 硬伤实验-三元组定向因果</div>
        <div><span style={{ color: ROLE_COLORS.unifiedDecode }}>●</span> 统一解码节点</div>
        <div><span style={{ color: '#84f1ff' }}>●</span> 输入概念神经元</div>
        <div><span style={{ color: ROLE_COLORS.background }}>●</span> 背景网络采样</div>
        <div style={{ color: '#6f84ad', marginTop: 8 }}>
          {`核心集合: ${(summary.micro || 0) + (summary.macro || 0) + (summary.route || 0)} | 多维集合: ${summary.multidimNodes || 0} | 硬伤: ${summary.hardProblemNodes || 0} | 统一解码: ${summary.unifiedDecodeNodes || 0}`}
        </div>
      </div>
    </div>
  );
}

export function AppleNeuronCategoryComparePanel({ workspace, compact = false }) {
  const summary = workspace?.summary || {};
  const querySets = workspace?.querySets || [];
  const categoryStats = summary?.categoryStats || {};
  const categoryRows = Object.entries(categoryStats)
    .map(([name, stat]) => ({ name, ...stat }))
    .sort((a, b) => b.neurons - a.neurons);
  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;

  return (
    <div style={cardStyle}>
      <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>类别神经元比较</div>
      <div style={{ fontSize: 12, color: '#92a6cc', lineHeight: 1.6 }}>
        {`总神经元 ${summary.total || 0} | 查询神经元 ${summary.query || 0} | 类别数 ${categoryRows.length}`}
      </div>
      <div style={{ fontSize: 11, color: '#7ea2c9', marginTop: 4 }}>
        {`当前词元: ${summary.currentToken || '-'} (${((summary.currentTokenProb || 0) * 100).toFixed(1)}%) | 多维节点: ${summary.multidimNodes || 0}`}
      </div>
      <div style={{ marginTop: 8, display: 'grid', gap: 6 }}>
        {categoryRows.length === 0 ? (
          <div style={{ fontSize: 11, color: '#6f84ad' }}>暂无类别数据，请在左侧输入概念和类别后生成。</div>
        ) : (
          categoryRows.map((row) => (
            <div key={`cat-${row.name}`} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#9eb4dd' }}>
              <span>{row.name}</span>
              <span>{`${row.concepts} 概念 / ${row.neurons} 神经元`}</span>
            </div>
          ))
        )}
      </div>
      <div style={{ marginTop: 8, fontSize: 11, color: '#6f84ad' }}>{`已生成概念集: ${querySets.length}`}</div>
    </div>
  );
}

export function AppleNeuronCompareFilterPanel({ workspace, compact = false }) {
  const querySets = workspace?.querySets || [];
  const queryVisibility = workspace?.queryVisibility || {};
  const setQuerySetVisible = workspace?.setQuerySetVisible;
  const setAllQuerySetVisible = workspace?.setAllQuerySetVisible;
  const summary = workspace?.summary || {};
  const visibleCount = querySets.filter((set) => queryVisibility[set.id] !== false).length;
  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;

  return (
    <div style={cardStyle}>
      <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>Compare Filter</div>
      <div style={{ fontSize: 12, color: '#92a6cc', lineHeight: 1.6 }}>
        {`按输入名称筛选：已显示 ${visibleCount}/${querySets.length}`}
      </div>
      <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
        <button type="button" onClick={() => setAllQuerySetVisible?.(true)} style={smallActionButtonStyle}>全选</button>
        <button type="button" onClick={() => setAllQuerySetVisible?.(false)} style={smallActionButtonStyle}>全不选</button>
      </div>
      <div style={{ marginTop: 10, display: 'grid', gap: 8, maxHeight: compact ? 180 : 220, overflowY: 'auto' }}>
        {querySets.length === 0 ? (
          <div style={{ fontSize: 11, color: '#6f84ad' }}>暂无输入名称。请先在左侧生成概念神经元。</div>
        ) : (
          querySets.map((set) => (
            <label key={`qf-${set.id}`} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#9eb4dd' }}>
              <input
                type="checkbox"
                checked={queryVisibility[set.id] !== false}
                onChange={(e) => setQuerySetVisible?.(set.id, e.target.checked)}
              />
              <span style={{ color: set.color }}>●</span>
              <span>{`${set.name} [${set.category}] (${set.nodes.length})`}</span>
            </label>
          ))
        )}
      </div>
      <div style={{ marginTop: 8, fontSize: 11, color: '#6f84ad' }}>
        {`当前词元: ${summary.currentToken || '-'} (${((summary.currentTokenProb || 0) * 100).toFixed(1)}%)`}
      </div>
    </div>
  );
}

export function AppleNeuronGeneratedConceptSetsPanel({ workspace, compact = false }) {
  const querySets = workspace?.querySets || [];
  const queryVisibility = workspace?.queryVisibility || {};
  const setQuerySetVisible = workspace?.setQuerySetVisible;
  const setAllQuerySetVisible = workspace?.setAllQuerySetVisible;
  const removeQuerySet = workspace?.removeQuerySet;
  const visibleCount = querySets.filter((set) => queryVisibility[set.id] !== false).length;
  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;

  return (
    <div style={cardStyle}>
      <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>已生成概念集</div>
      <div style={{ fontSize: 11, color: '#92a6cc', lineHeight: 1.6 }}>
        {`当前共 ${querySets.length} 组概念，其中显示 ${visibleCount} 组。这里统一处理显隐和清理。`}
      </div>
      <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
        <button type="button" onClick={() => setAllQuerySetVisible?.(true)} style={smallActionButtonStyle}>全显示</button>
        <button type="button" onClick={() => setAllQuerySetVisible?.(false)} style={smallActionButtonStyle}>全隐藏</button>
      </div>
      <div style={{ marginTop: 10, display: 'grid', gap: 8, maxHeight: compact ? 220 : 280, overflowY: 'auto' }}>
        {querySets.length === 0 ? (
          <div style={{ fontSize: 11, color: '#6f84ad' }}>暂未生成概念集，请先在左侧输入名词和类别。</div>
        ) : (
          querySets.map((set) => (
            <div key={`generated-set-${set.id}`} style={{ display: 'grid', gridTemplateColumns: '20px 1fr auto', gap: 8, alignItems: 'center', fontSize: 12, color: '#9eb4dd' }}>
              <input
                type="checkbox"
                checked={queryVisibility[set.id] !== false}
                onChange={(e) => setQuerySetVisible?.(set.id, e.target.checked)}
              />
              <span style={{ overflowWrap: 'anywhere' }}>
                <span style={{ color: set.color }}>●</span>
                {` ${set.name} [${set.category}] (${set.nodes.length})`}
              </span>
              <button
                type="button"
                onClick={() => removeQuerySet?.(set.id)}
                style={{ ...smallActionButtonStyle, padding: '2px 8px', fontSize: 11 }}
              >
                删除
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export function AppleNeuronMultidimSettingsPanel({ workspace, compact = false }) {
  const multidimProbeData = workspace?.multidimProbeData || null;
  const multidimCausalData = workspace?.multidimCausalData || null;
  const multidimTopN = workspace?.multidimTopN ?? 96;
  const setMultidimTopN = workspace?.setMultidimTopN;
  const multidimVisible = workspace?.multidimVisible || { style: true, logic: true, syntax: true };
  const setMultidimVisible = workspace?.setMultidimVisible;
  const multidimActiveDimension = workspace?.multidimActiveDimension || 'style';
  const setMultidimActiveDimension = workspace?.setMultidimActiveDimension;
  const multidimLayerProfile = workspace?.multidimLayerProfile || [];
  const cardStyle = compact ? { ...panelCardStyle, padding: 10 } : panelCardStyle;

  return (
    <div style={cardStyle}>
      <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>三维编码设置</div>
      <div style={{ fontSize: 11, color: '#7f95bb', lineHeight: 1.6, marginBottom: 10 }}>
        管理 `style / logic / syntax` 三维探针的可见性、TopN 和当前显示维度。
      </div>
      <div style={{ display: 'grid', gap: 8 }}>
        <div style={{ fontSize: 11, color: '#7ea2c9' }}>
          {multidimProbeData
            ? `已导入探针，当前显示维度: ${DIMENSION_LABELS[multidimActiveDimension] || multidimActiveDimension}，层谱点数: ${multidimLayerProfile?.length || 0}`
            : '未导入三维探针 JSON（multidim_encoding_probe.json）'}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
          <div style={{ fontSize: 12, color: '#9eb4dd' }}>TopN</div>
          <input
            type="number"
            min={16}
            max={256}
            value={multidimTopN}
            onChange={(e) => setMultidimTopN?.(Number(e.target.value))}
            style={inputStyle}
          />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6 }}>
          {['style', 'logic', 'syntax'].map((dim) => (
            <button
              key={`multidim-panel-${dim}`}
              type="button"
              onClick={() => setMultidimActiveDimension?.(dim)}
              style={{
                borderRadius: 8,
                border: `1px solid ${multidimActiveDimension === dim ? ROLE_COLORS[dim] : 'rgba(122,162,255,0.35)'}`,
                background: multidimActiveDimension === dim ? 'rgba(42,71,132,0.82)' : 'rgba(7, 12, 25, 0.82)',
                color: '#dbe9ff',
                fontSize: 11,
                padding: '6px 8px',
                cursor: 'pointer',
              }}
            >
              {DIMENSION_LABELS[dim]}
            </button>
          ))}
        </div>
        <div style={{ display: 'grid', gap: 6 }}>
          {['style', 'logic', 'syntax'].map((dim) => (
            <label key={`multidim-vis-${dim}`} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#9eb4dd' }}>
              <input
                type="checkbox"
                checked={multidimVisible[dim] !== false}
                onChange={(e) => setMultidimVisible?.((prev) => ({ ...prev, [dim]: e.target.checked }))}
              />
              <span style={{ color: ROLE_COLORS[dim] }}>●</span>
              <span>{`${DIMENSION_LABELS[dim]}神经元`}</span>
            </label>
          ))}
        </div>
        <div style={{ fontSize: 11, color: '#7ea2c9', lineHeight: 1.6 }}>
          {multidimCausalData
            ? `对角优势 style=${toSafeNumber(multidimCausalData?.diagonal_advantage?.style, 0).toFixed(4)} / logic=${toSafeNumber(multidimCausalData?.diagonal_advantage?.logic, 0).toFixed(4)} / syntax=${toSafeNumber(multidimCausalData?.diagonal_advantage?.syntax, 0).toFixed(4)}`
            : '未导入三维因果消融 JSON（multidim_causal_ablation.json）'}
        </div>
      </div>
    </div>
  );
}

/* legacy control panels removed from main path on 2026-03-26
  const {
    analysisMode,
    setAnalysisMode,
    analysisModes,
    animationMode,
    setAnimationMode,
    animationModes,
    theoryObject,
    setTheoryObject,
    theoryObjects,
    currentTheoryObject,
    availableModesForTheoryObject,
    nodes,
    setSelected,
    showFruitGeneral,
    setShowFruitGeneral,
    showFruit,
    setShowFruit,
    summary,
    selected,
    queryInput,
    setQueryInput,
    queryCategoryInput,
    setQueryCategoryInput,
    handleGenerateQuery,
    querySets,
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
    multidimProbeData,
    multidimCausalData,
    multidimTopN,
    setMultidimTopN,
    multidimVisible,
    setMultidimVisible,
    multidimActiveDimension,
    setMultidimActiveDimension,
    multidimLayerProfile,
    handleImportScanJsonText,
    removeQuerySet,
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
    languageFocus,
    setLanguageFocus,
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
    handleBasicRuntimeStart,
    handleBasicRuntimeStop,
    handleBasicRuntimeReplay,
    modeMetrics,
    hardProblemResults,
    unifiedDecodeResult,
  } = workspace;
  const [scanFileOptions, setScanFileOptions] = useState([]);
  const [scanFileLoading, setScanFileLoading] = useState(false);
  const [scanFileImporting, setScanFileImporting] = useState(false);
  const [scanFileError, setScanFileError] = useState('');
  const [scanFileFilter, setScanFileFilter] = useState('all');
  const [scanOptionContentLabels, setScanOptionContentLabels] = useState({});
  const [assetPanelTab, setAssetPanelTab] = useState('manual');
  const [appleNeuronPanelTab, setAppleNeuronPanelTab] = useState('core');
  const [dnnPanelTab, setDnnPanelTab] = useState('basic');
  const [basicConceptFocus, setBasicConceptFocus] = useState('fruit_general');
  const modeMetaById = Object.fromEntries(analysisModes.map((mode) => [mode.id, mode]));
  const filteredStageGroups = useMemo(
    () => ANALYSIS_MODE_STAGE_GROUPS
      .map((group) => ({
        ...group,
        items: group.items.filter((id) => availableModesForTheoryObject.includes(id)),
      }))
      .filter((group) => group.items.length > 0),
    [availableModesForTheoryObject]
  );
  const scanFileFilterLabelMap = {
    multidim: '多维编码',
    mass_noun: '名词扫描',
    hard_problem: '硬伤实验',
    four_tasks: '四任务套件',
    unified_decode: '统一解码',
    all: '全部',
  };

  const researchSnapshotRows = useMemo(() => ([
    { label: '标准化单位', value: `${DNN_RESEARCH_SNAPSHOT.standardizedUnits}` },
    { label: '真实精确占比', value: `${(DNN_RESEARCH_SNAPSHOT.exactRealFraction * 100).toFixed(2)}%` },
    { label: '概念签名', value: `${DNN_RESEARCH_SNAPSHOT.signatureRows} / ${DNN_RESEARCH_SNAPSHOT.uniqueConcepts}` },
    { label: '完整还原', value: `${(DNN_RESEARCH_SNAPSHOT.fullRestorationScore * 100).toFixed(2)}%` },
  ]), []);

  const currentTheoryResearch = useMemo(
    () => THEORY_OBJECT_RESEARCH_MAP[theoryObject] || THEORY_OBJECT_RESEARCH_MAP.family_patch,
    [theoryObject]
  );

  const currentLayerProfile = useMemo(() => {
    const key = LAYER_PARAMETER_STATE_ORDER.includes(languageFocus?.researchLayer)
      ? languageFocus.researchLayer
      : 'static_encoding';
    return LAYER_PARAMETER_STATE_OVERLAY[key] || LAYER_PARAMETER_STATE_OVERLAY.static_encoding;
  }, [languageFocus?.researchLayer]);

  const currentLayerProfileStats = useMemo(() => ({
    nodeCount: Array.isArray(currentLayerProfile?.nodes) ? currentLayerProfile.nodes.length : 0,
    chainCount: Array.isArray(currentLayerProfile?.chains) ? currentLayerProfile.chains.length : 0,
    sourceStage: currentLayerProfile?.nodes?.[0]?.sourceStage || '-',
    sourcePath: currentLayerProfile?.sourceDataPath || '-',
  }), [currentLayerProfile]);

  const currentDisplaySummary = useMemo(
    () => DNN_DISPLAY_LEVEL_OPTIONS
      .filter((item) => displayLevels?.[item.id] !== false)
      .map((item) => item.label)
      .join(' / ') || '无',
    [displayLevels]
  );

  const appleNeuronGroups = useMemo(() => {
    const nodeList = Array.isArray(nodes) ? nodes : [];
    const actualAppleCoreNodes = nodeList.filter((node) => node?.role === 'micro' && String(node?.source || '').startsWith('seed_layer_neuron_map'));
    const actualFruitGeneralNodes = nodeList.filter((node) => node?.role === 'fruitGeneral');
    const actualFruitSpecificNodes = nodeList.filter((node) => node?.role === 'fruitSpecific');

    const coreItems = actualAppleCoreNodes.length > 0
      ? actualAppleCoreNodes
      : APPLE_CORE_NEURONS.map((item) => ({
          ...item,
          neuron: item.neuron,
          metric: item.metric,
          source: item.source,
          value: item.value,
          strength: item.strength,
        }));

    const generalItems = actualFruitGeneralNodes.length > 0
      ? actualFruitGeneralNodes
      : FRUIT_GENERAL_NEURONS.map((item, idx) => ({
          id: `fruit-general-seed-${idx}`,
          label: `水果共享神经元 ${idx + 1}`,
          layer: item.layer,
          neuron: item.neuron,
          metric: 'fruit_general_score',
          value: item.score,
          strength: item.score,
          source: 'fruit_general_seed_v1',
          role: 'fruitGeneral',
        }));

    const fruitFamilies = Object.keys(FRUIT_COLORS).map((fruit) => {
      const actualItems = actualFruitSpecificNodes.filter((node) => node?.fruit === fruit);
      const fallbackItems = (FRUIT_SPECIFIC_NEURONS[fruit] || []).map((item, idx) => ({
        id: `${fruit}-specific-seed-${idx}`,
        label: `${fruit} 专属神经元 ${idx + 1}`,
        fruit,
        layer: item.layer,
        neuron: item.neuron,
        metric: 'fruit_specific_score',
        value: item.score,
        strength: item.score,
        source: 'fruit_specific_seed_v1',
        role: 'fruitSpecific',
      }));
      return {
        fruit,
        color: FRUIT_COLORS[fruit],
        visible: showFruit?.[fruit] !== false,
        items: actualItems.length > 0 ? actualItems : fallbackItems,
      };
    });

    return {
      coreItems,
      generalItems,
      fruitFamilies,
      totalCount: coreItems.length + generalItems.length + fruitFamilies.reduce((sum, item) => sum + item.items.length, 0),
    };
  }, [nodes, showFruit]);

  const basicConceptEntries = useMemo(() => {
    const familyEntries = appleNeuronGroups.fruitFamilies.map((family) => ({
      id: family.fruit,
      label: family.fruit,
      desc: `${family.fruit} 的基础神经元`,
      count: family.items.length,
      color: family.color,
      nodes: family.items,
    }));
    return [
      {
        id: 'apple',
        label: 'apple',
        desc: '苹果概念本身的基础神经元',
        count: appleNeuronGroups.coreItems.length,
        color: '#fb7185',
        nodes: appleNeuronGroups.coreItems,
      },
      {
        id: 'fruit_general',
        label: 'fruit',
        desc: '水果对象共同复用的基础神经元',
        count: appleNeuronGroups.generalItems.length,
        color: '#6cf7d4',
        nodes: appleNeuronGroups.generalItems,
      },
      ...familyEntries.filter((entry) => entry.id !== 'apple'),
    ];
  }, [appleNeuronGroups]);

  const currentBasicConceptEntry = useMemo(
    () => basicConceptEntries.find((item) => item.id === basicConceptFocus) || basicConceptEntries[0] || null,
    [basicConceptEntries, basicConceptFocus]
  );

  const importStatusRows = useMemo(() => ([
    { label: '多维探针', value: multidimProbeData ? '已导入' : '未导入' },
    { label: '硬伤实验', value: `${Object.keys(hardProblemResults || {}).length} 项` },
    { label: '统一解码', value: unifiedDecodeResult ? '已导入' : '未导入' },
    { label: '概念集', value: `${querySets.length}` },
  ]), [hardProblemResults, multidimProbeData, querySets.length, unifiedDecodeResult]);

  const researchRiskRows = useMemo(() => {
    const rows = [
      `当前对象: ${currentTheoryObject?.labelZh || '-'}，当前动作: ${modeMetaById[analysisMode]?.label || '-'}`,
      ANALYSIS_MODE_RESEARCH_NOTES[analysisMode] || '当前动作暂无附加说明。',
    ];
    if (theoryObject === 'successor_aligned_transport' || analysisMode === 'dynamic_prediction') {
      rows.push(`successor exact dense 仅 ${DNN_RESEARCH_SNAPSHOT.successorExactDenseUnits} / ${DNN_RESEARCH_SNAPSHOT.successorTotalUnits}`);
      rows.push(`proxy 单位仍有 ${DNN_RESEARCH_SNAPSHOT.successorProxyUnits}，当前 exactness ${(DNN_RESEARCH_SNAPSHOT.successorExactnessFraction * 100).toFixed(2)}%`);
    } else if (theoryObject === 'protocol_bridge') {
      rows.push('protocol field 已接近强候选恢复，但 successor -> protocol 的 exact 闭合仍不足。');
    } else if (theoryObject === 'family_patch' || theoryObject === 'concept_section') {
      rows.push('family basis / concept offset 已较强，但还不是最终唯一数学定理。');
    } else {
      rows.push('当前对象的可视化证据已进入 DNN 主流程，但 dense neuron-level exact closure 仍未完成。');
    }
    return rows;
  }, [analysisMode, currentTheoryObject?.labelZh, modeMetaById, theoryObject]);

  const selectedResearchDetails = useMemo(() => {
    if (!selected) {
      return null;
    }
    if (selected.detailType === 'apple_switch_unit') {
      const unit = selected.appleSwitchUnit || {};
      return {
        sourceTier: '苹果切换机制资产',
        dataGroup: 'core',
        exactness: '角色 + 因果 + 有符号推进 + 稳定性',
        detailRows: [
          { label: '对象', value: selected.label },
          { label: '模型', value: selected.modelName || selected.modelKey || '-' },
          { label: '角色', value: selected.roleLabel || '-' },
          { label: '类型', value: selected.unitTypeLabel || '-' },
          { label: '真实层', value: `L${selected.actualLayer}` },
          { label: '有效分数', value: Number(selected.effectiveScore || 0).toFixed(4) },
          { label: 'search_drop', value: Number(unit?.causal_effect?.search_drop || 0).toFixed(4) },
          { label: '晚层有符号耦合', value: Number(unit?.signed_effect?.late_mean_signed_contrast_switch_coupling || 0).toFixed(4) },
        ],
      };
    }
    const sourceTier = selected.source === 'textbox-query-generator'
      ? '交互生成'
      : selected.source === 'multidim_encoding_probe'
        ? '真实探针'
        : selected.source === 'agi_research_result_v1'
          ? '硬伤实验'
          : selected.source === 'unified_math_structure_decode'
            ? '统一解码'
            : selected.source === 'mass_noun_encoding_scan_import'
              ? '名词扫描导入'
              : '其他';
    const dataGroup = nodeDisplayGroup(selected.role);
    return {
      sourceTier,
      dataGroup,
      exactness:
        selected.source === 'multidim_encoding_probe' || selected.source === 'unified_math_structure_decode'
          ? '较强真实证据'
          : selected.source === 'agi_research_result_v1' || selected.source === 'mass_noun_encoding_scan_import'
            ? '研究资产证据'
            : '交互或背景证据',
      detailRows: [
        { label: '对象', value: selected.label },
        { label: '层 / 神经元', value: `L${selected.layer} / N${selected.neuron}` },
        { label: '来源层级', value: sourceTier },
        { label: '数据分组', value: dataGroup },
        { label: '指标', value: `${selected.metric}: ${selected.value.toExponential(3)}` },
      ],
    };
  }, [selected]);

  const selectedNodeDetails = useMemo(() => {
    if (!selected) {
      return null;
    }
    if (selected.detailType === 'apple_switch_unit') {
      const unit = selected.appleSwitchUnit || {};
      return {
        detailRows: [
          { label: '单元', value: selected.unitId || selected.label },
          { label: '模型', value: selected.modelName || selected.modelKey || '-' },
          { label: '角色', value: selected.roleLabel || '-' },
          { label: '类型', value: selected.unitTypeLabel || '-' },
          { label: '真实层', value: `L${selected.actualLayer}` },
          { label: '顺向峰值层', value: `L${unit?.signed_effect?.forward_peak_layer ?? '-'}` },
          { label: '反向峰值层', value: `L${unit?.signed_effect?.reverse_peak_layer ?? '-'}` },
          { label: '方向', value: unit?.signed_effect?.direction_label || '-' },
          { label: 'utility', value: Number(unit?.causal_effect?.utility || 0).toFixed(4) },
          { label: '稳定性分数', value: Number(unit?.stability?.stability_score || 0).toFixed(4) },
        ],
      };
    }
    if (selected.detailType === 'parameter_state') {
      return {
        detailRows: [
          { label: '节点', value: selected.label },
          { label: '层/神经元', value: `L${selected.layer} / N${selected.neuron}` },
          { label: '参数维度', value: `d${selected.dimIndex}` },
          { label: '参数位', value: Array.isArray(selected.parameterIds) ? selected.parameterIds.join(', ') : '-' },
          { label: '来源阶段', value: selected.sourceStage || '-' },
          { label: '来源实体编号', value: selected.sourceEntityId || '-' },
          { label: '来源数据路径', value: selected.sourceDataPath || '-' },
          { label: '输出目录', value: selected.outputDir || '-' },
          { label: '指标', value: `${selected.metric}: ${Number(selected.value || 0).toFixed(4)}` },
        ],
      };
    }
    if (selected.detailType === 'parameter_state') {
      return {
        detailRows: [
          { label: '节点', value: selected.label },
          { label: '层/神经元', value: `L${selected.layer} / N${selected.neuron}` },
          { label: '参数维度', value: `d${selected.dimIndex}` },
          { label: '参数位', value: Array.isArray(selected.parameterIds) ? selected.parameterIds.join(', ') : '-' },
          { label: '来源阶段', value: selected.sourceStage || '-' },
          { label: '输出目录', value: selected.outputDir || '-' },
          { label: '指标', value: `${selected.metric}: ${Number(selected.value || 0).toFixed(4)}` },
        ],
      };
    }
    const sourceTier = selected.source === 'textbox-query-generator'
      ? '手动输入'
      : selected.source === 'multidim_encoding_probe'
        ? '多维探针'
        : selected.source === 'agi_research_result_v1'
          ? '硬伤实验'
          : selected.source === 'unified_math_structure_decode'
            ? '统一解码'
            : selected.source === 'mass_noun_encoding_scan_import'
              ? '名词扫描导入'
              : '其他';
    return {
      detailRows: [
        { label: '对象', value: selected.label },
        { label: '层 / 神经元', value: `L${selected.layer} / N${selected.neuron}` },
        { label: '来源', value: sourceTier },
        { label: '分组', value: nodeDisplayGroup(selected.role) },
        { label: '指标', value: `${selected.metric}: ${selected.value.toExponential(3)}` },
      ],
    };
  }, [selected]);

  const currentScanMeta = useMemo(
    () => scanFileOptions.find((f) => f.path === selectedScanPath) || null,
    [scanFileOptions, selectedScanPath]
  );

  const scanPreview = useMemo(
    () => buildArtifactPreview(scanPreviewData, selectedScanPath),
    [scanPreviewData, selectedScanPath]
  );

  const scanPreviewTheory = useMemo(
    () => THEORY_OBJECT_RESEARCH_MAP[scanPreview?.theoryObject] || currentTheoryResearch,
    [currentTheoryResearch, scanPreview?.theoryObject]
  );

  const scanPreviewJson = useMemo(
    () => scanPreview?.rawJson || '',
    [scanPreview]
  );
  const showScanPreviewInTopRight = shouldShowResearchAssetInTopRight(scanPreview, selectedScanPath);

  const filteredScanFileOptions = useMemo(() => {
    const rows = Array.isArray(scanFileOptions) ? scanFileOptions : [];
    if (scanFileFilter === 'all') {
      return rows;
    }
    if (scanFileFilter === 'multidim') {
      return rows.filter((f) => {
        const p = String(f?.path || '').toLowerCase();
        return p.includes('multidim_encoding_probe')
          || p.includes('multidim_causal_ablation')
          || p.includes('multidim_multiseed_stability');
      });
    }
    if (scanFileFilter === 'mass_noun') {
      return rows.filter((f) => {
        const p = String(f?.path || '').toLowerCase();
        return p.includes('mass_noun') || p.includes('noun_scan') || p.includes('encoding_scan');
      });
    }
    if (scanFileFilter === 'hard_problem') {
      return rows.filter((f) => {
        const p = String(f?.path || '').toLowerCase();
        return p.includes('dynamic_binding_stress_test')
          || p.includes('long_horizon_causal_trace_test')
          || p.includes('local_credit_assignment_proxy_test')
          || p.includes('triplet_targeted_causal_scan')
          || p.includes('triplet_targeted_multiseed_stability')
          || p.includes('variable_binding_hard_verification')
          || p.includes('minimal_causal_circuit_search')
          || p.includes('unified_coordinate_system_test')
          || p.includes('concept_family_parallel_scale')
          || p.includes('agi_research_stage_bundle_manifest');
      });
    }
    if (scanFileFilter === 'four_tasks') {
      return rows.filter((f) => {
        const p = String(f?.path || '').toLowerCase();
        return p.includes('agi_four_tasks_suite_manifest')
          || p.includes('variable_binding_hard_verification')
          || p.includes('minimal_causal_circuit_search')
          || p.includes('unified_coordinate_system_test')
          || p.includes('concept_family_parallel_scale');
      });
    }
    if (scanFileFilter === 'unified_decode') {
      return rows.filter((f) => {
        const p = String(f?.path || '').toLowerCase();
        return p.includes('unified_math_structure_decode');
      });
    }
    return rows;
  }, [scanFileFilter, scanFileOptions]);

  const refreshScanFileOptions = async () => {
    setScanFileLoading(true);
    setScanFileError('');
    try {
      const res = await fetch(`${MAIN_API_BASE}/api/main/scan_files?limit=200`);
      const payload = await res.json();
      if (!res.ok) {
        throw new Error(payload?.detail || '读取扫描文件列表失败');
      }
      const files = Array.isArray(payload?.files) ? payload.files : [];
      setScanFileOptions(files);
      setSelectedScanPath((prev) => {
        if (prev && files.some((f) => f.path === prev)) {
          return prev;
        }
        return files[0]?.path || '';
      });
      if (files.length === 0) {
        setScanFileError('未发现可导入文件：请先生成名词扫描、多维编码、硬伤实验、四任务套件或统一解码 JSON。');
      }
    } catch (err) {
      setScanFileOptions([]);
      setSelectedScanPath('');
      setScanFileError(`扫描文件列表加载失败: ${err?.message || err}`);
    } finally {
      setScanFileLoading(false);
    }
  };

  useEffect(() => {
    refreshScanFileOptions();
  }, []);

  useEffect(() => {
    setSelectedScanPath((prev) => {
      if (prev && filteredScanFileOptions.some((f) => f.path === prev)) {
        return prev;
      }
      return filteredScanFileOptions[0]?.path || '';
    });
  }, [filteredScanFileOptions]);

  useEffect(() => {
    let cancelled = false;
    const candidates = filteredScanFileOptions
      .map((fileMeta) => ({ path: fileMeta.path, fileMeta }))
      .filter((item) => item.path && !scanOptionContentLabels[item.path])
      .slice(0, 24);

    if (candidates.length === 0) {
      return undefined;
    }

    const loadOptionLabels = async () => {
      const updates = {};
      await Promise.allSettled(candidates.map(async ({ path, fileMeta }) => {
        try {
          const res = await fetch(`${MAIN_API_BASE}/api/main/scan_file?path=${encodeURIComponent(path)}`);
          const payload = await res.json();
          if (!res.ok) {
            throw new Error(payload?.detail || '读取研究资产失败');
          }
          updates[path] = buildScanContentLabel(payload?.data, fileMeta);
        } catch (_err) {
          updates[path] = inferScanOptionConcept(fileMeta);
        }
      }));
      if (!cancelled && Object.keys(updates).length > 0) {
        setScanOptionContentLabels((prev) => ({ ...prev, ...updates }));
      }
    };

    loadOptionLabels();
    return () => {
      cancelled = true;
    };
  }, [filteredScanFileOptions, scanOptionContentLabels]);

  const handleImportSelectedScanFile = async () => {
    if (!selectedScanPath) {
      setScanFileError('请先在下拉框选择一个扫描 JSON 文件。');
      return;
    }
    setScanFileImporting(true);
    setScanFileError('');
    try {
      const res = await fetch(`${MAIN_API_BASE}/api/main/scan_file?path=${encodeURIComponent(selectedScanPath)}`);
      const payload = await res.json();
      if (!res.ok) {
        throw new Error(payload?.detail || '读取扫描文件失败');
      }
      if (!payload?.data) {
        throw new Error('返回数据为空');
      }
      const data = payload.data;
      const sourcePath = payload.path || selectedScanPath;

      const looksLikeBundle = Boolean(data?.bundle_id === 'agi_research_stage_bundle_v1');
      if (looksLikeBundle) {
        handleImportScanJsonText(JSON.stringify(data), sourcePath);
        const artifactPaths = [
          data?.artifacts?.dynamic_binding_json,
          data?.artifacts?.long_horizon_json,
          data?.artifacts?.local_credit_json,
          data?.artifacts?.triplet_targeted_json,
          data?.artifacts?.triplet_multiseed_json,
          data?.artifacts?.unified_decode_json,
        ].filter(Boolean);
        for (const apath of artifactPaths) {
          try {
            const ar = await fetch(`${MAIN_API_BASE}/api/main/scan_file?path=${encodeURIComponent(apath)}`);
            const aj = await ar.json();
            if (ar.ok && aj?.data) {
              handleImportScanJsonText(JSON.stringify(aj.data), aj.path || apath);
            }
          } catch (_err) {
            // Ignore missing optional artifact files.
          }
        }
        return;
      }

      const looksLikeFourTasks = Boolean(data?.suite_id === 'agi_four_tasks_suite_v1');
      if (looksLikeFourTasks) {
        handleImportScanJsonText(JSON.stringify(data), sourcePath);
        const joinPath = (dirPath, fileName) => {
          const base = String(dirPath || '').replace(/[\\/]+$/, '');
          if (!base) {
            return '';
          }
          return `${base}/${fileName}`;
        };
        const taskFiles = [
          joinPath(data?.artifacts?.task1, 'variable_binding_hard_verification.json'),
          joinPath(data?.artifacts?.task2, 'minimal_causal_circuit_search.json'),
          joinPath(data?.artifacts?.task3, 'unified_coordinate_system_test.json'),
          joinPath(data?.artifacts?.task4, 'concept_family_parallel_scale.json'),
        ].filter(Boolean);
        for (const p of taskFiles) {
          try {
            const r = await fetch(`${MAIN_API_BASE}/api/main/scan_file?path=${encodeURIComponent(p)}`);
            const j = await r.json();
            if (r.ok && j?.data) {
              handleImportScanJsonText(JSON.stringify(j.data), j.path || p);
            }
          } catch (_err) {
            // Ignore missing optional task file.
          }
        }
        return;
      }

      // 若导入的是多seed稳定性汇总，则自动加载其中一个 probe + ablation，避免“导入后看不到变化”。
      const looksLikeStability = Boolean(data?.runs && data?.aggregate && data?.aggregate?.specificity_margin_style);
      if (looksLikeStability) {
        const runs = Array.isArray(data?.runs) ? data.runs : [];
        const preferredRun = runs.find((r) => r?.seed === 505) || runs[runs.length - 1] || runs[0];
        const probePath = preferredRun?.probe_json;
        const ablationPath = preferredRun?.ablation_json;

        handleImportScanJsonText(JSON.stringify(data), sourcePath);

        if (probePath) {
          try {
            const probeRes = await fetch(`${MAIN_API_BASE}/api/main/scan_file?path=${encodeURIComponent(probePath)}`);
            const probePayload = await probeRes.json();
            if (probeRes.ok && probePayload?.data) {
              handleImportScanJsonText(JSON.stringify(probePayload.data), probePayload.path || probePath);
            }
          } catch (_e) {
            // ignore, keep stability summary import result
          }
        }
        if (ablationPath) {
          try {
            const abRes = await fetch(`${MAIN_API_BASE}/api/main/scan_file?path=${encodeURIComponent(ablationPath)}`);
            const abPayload = await abRes.json();
            if (abRes.ok && abPayload?.data) {
              handleImportScanJsonText(JSON.stringify(abPayload.data), abPayload.path || ablationPath);
            }
          } catch (_e) {
            // ignore, keep existing imported states
          }
        }
      } else {
        handleImportScanJsonText(JSON.stringify(data), sourcePath);
      }
    } catch (err) {
      setScanFileError(`导入失败: ${err?.message || err}`);
    } finally {
      setScanFileImporting(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {false ? (
      <div style={panelCardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>基础信息</div>
        <div style={{ fontSize: 11, color: '#7f95bb', lineHeight: 1.6, marginBottom: 10 }}>
          用两个入口处理研究数据：手动输入名词与类别，直接观察 3D 模型；或选择测试中的研究资产，直接展开具体数据、数学分析和对应理论。
        </div>
        {externalAuditFocus ? (
          <div
            style={{
              marginBottom: 10,
              borderRadius: 12,
              padding: '10px 12px',
              border: '1px solid rgba(56, 189, 248, 0.28)',
              background: 'rgba(8, 47, 73, 0.35)',
            }}
          >
            <div style={{ fontSize: 11, color: '#67e8f9', fontWeight: 700, marginBottom: 4 }}>
              {`严格审查联动：${externalAuditFocus.stageLabel || externalAuditFocus.stageId || '未知阶段'}`}
            </div>
            <div style={{ fontSize: 11, color: '#dbe9ff', lineHeight: 1.7 }}>
              {externalAuditFocus.summary || '当前 3D 工作台已切换到来自严格审查页的聚焦方案。'}
            </div>
            <div style={{ fontSize: 10, color: '#8fb9d9', marginTop: 6 }}>
              {`数据焦点：研究层 ${languageFocus?.researchLayer || '-'} | 显示策略 ${displayStrategy === 'auto' ? '自动聚焦' : displayStrategy === 'all' ? '全部显示' : '手动筛选'} | 动画 ${animationModes.find((item) => item.id === animationMode)?.label || '无动画'}`}
            </div>
          </div>
        ) : null}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 8, marginBottom: 10 }}>
          {[
            { id: 'manual', label: '手动输入', desc: '输入名词与类别，查看对应 3D 节点' },
            { id: 'artifact', label: '测试数据', desc: '选择实验文件，完整查看数据、差异和验证' },
          ].map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setAssetPanelTab(tab.id)}
              title={tab.desc}
              style={{
                borderRadius: 10,
                border: `1px solid ${assetPanelTab === tab.id ? 'rgba(126, 224, 255, 0.75)' : 'rgba(122, 162, 255, 0.28)'}`,
                background: assetPanelTab === tab.id ? 'rgba(24, 101, 134, 0.36)' : 'rgba(7, 12, 25, 0.82)',
                color: '#dbe9ff',
                padding: '8px 10px',
                cursor: 'pointer',
                textAlign: 'left',
              }}
            >
              <div style={{ fontSize: 12, fontWeight: 700 }}>{tab.label}</div>
              <div style={{ fontSize: 10, color: '#88a6cf', marginTop: 2 }}>{tab.desc}</div>
            </button>
          ))}
        </div>

        {assetPanelTab === 'manual' ? (
          <div style={{ display: 'grid', gap: 10 }}>
            <div style={{ display: 'grid', gap: 8 }}>
              <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
                <div style={{ fontSize: 12, color: '#9eb4dd' }}>名称</div>
                <input
                  value={queryInput}
                  onChange={(e) => setQueryInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      handleGenerateQuery();
                    }
                  }}
                  placeholder="例如：苹果 / 太阳 / 量子"
                  style={inputStyle}
                />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
                <div style={{ fontSize: 12, color: '#9eb4dd' }}>类别</div>
                <input
                  value={queryCategoryInput}
                  onChange={(e) => setQueryCategoryInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      handleGenerateQuery();
                    }
                  }}
                  placeholder="例如：水果 / 天体 / 抽象概念"
                  style={inputStyle}
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
                <div style={{ fontSize: 11, color: '#7ea2c9' }}>{`已生成概念集 ${querySets.length} 个`}</div>
                <button type="button" onClick={handleGenerateQuery} style={smallActionButtonStyle}>
                  生成 3D 模型
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div style={{ display: 'grid', gap: 10 }}>
            <div style={{ display: 'grid', gap: 8 }}>
              <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
                <div style={{ fontSize: 12, color: '#9eb4dd' }}>导入数</div>
                <input
                  type="number"
                  min={1}
                  max={120}
                  value={scanImportLimit}
                  onChange={(e) => setScanImportLimit(Number(e.target.value))}
                  style={inputStyle}
                />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
                <div style={{ fontSize: 12, color: '#9eb4dd' }}>TopK</div>
                <input
                  type="number"
                  min={4}
                  max={64}
                  value={scanImportTopK}
                  onChange={(e) => setScanImportTopK(Number(e.target.value))}
                  style={inputStyle}
                />
              </div>
              <div style={{ display: 'grid', gap: 6 }}>
                <div style={{ fontSize: 12, color: '#9eb4dd' }}>测试数据类型</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6 }}>
                  {[
                    { id: 'multidim', label: '多维编码' },
                    { id: 'mass_noun', label: '名词扫描' },
                    { id: 'hard_problem', label: '硬伤实验' },
                    { id: 'four_tasks', label: '四任务' },
                    { id: 'unified_decode', label: '统一解码' },
                    { id: 'all', label: '全部' },
                  ].map((opt) => (
                    <button
                      key={`scan-filter-${opt.id}`}
                      type="button"
                      onClick={() => setScanFileFilter(opt.id)}
                      style={{
                        borderRadius: 8,
                        border: `1px solid ${scanFileFilter === opt.id ? 'rgba(126, 224, 255, 0.75)' : 'rgba(122, 162, 255, 0.35)'}`,
                        background: scanFileFilter === opt.id ? 'rgba(24, 101, 134, 0.38)' : 'rgba(7, 12, 25, 0.82)',
                        color: '#dbe9ff',
                        fontSize: 11,
                        padding: '6px 8px',
                        cursor: 'pointer',
                      }}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
                <div style={{ fontSize: 12, color: '#9eb4dd' }}>文件</div>
                <select
                  value={selectedScanPath}
                  onChange={(e) => setSelectedScanPath(e.target.value)}
                  style={inputStyle}
                  disabled={scanFileLoading || filteredScanFileOptions.length === 0}
                >
                  {filteredScanFileOptions.length === 0 ? (
                    <option value="">
                      {scanFileLoading
                        ? '扫描中...'
                        : scanFileOptions.length > 0
                          ? '当前筛选下无文件'
                          : '未发现可导入文件'}
                    </option>
                  ) : (
                    filteredScanFileOptions.map((f) => (
                      <option key={f.path} value={f.path}>
                        {formatScanOptionLabel(f, scanOptionContentLabels[f.path])}
                      </option>
                    ))
                  )}
                </select>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
                <div style={{ fontSize: 12, color: '#9eb4dd' }}>动画</div>
                <select
                  value={animationMode}
                  onChange={(e) => setAnimationMode(e.target.value)}
                  style={inputStyle}
                >
                  {animationModes.map((opt) => (
                    <option key={`asset-anim-${opt.id}`} value={opt.id}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
              <div style={{ fontSize: 11, color: '#7ea2c9', lineHeight: 1.6 }}>
                {animationModes.find((opt) => opt.id === animationMode)?.desc || '当前未启用附加动画。'}
              </div>
              <div style={{ fontSize: 11, color: '#7ea2c9' }}>
                {`候选文件 ${filteredScanFileOptions.length}/${scanFileOptions.length}，当前筛选：${scanFileFilterLabelMap[scanFileFilter] || scanFileFilter}`}
              </div>
              {!scanFileLoading && scanFileOptions.length > 0 && filteredScanFileOptions.length === 0 ? (
                <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.6 }}>
                  当前目录里有研究资产，但没有命中这个筛选条件。可以切到“全部”或其它类型查看。
                </div>
              ) : null}
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <button
                  type="button"
                  onClick={refreshScanFileOptions}
                  style={{ ...smallActionButtonStyle, flex: '1 1 120px', minWidth: 120 }}
                  disabled={scanFileLoading}
                >
                  {scanFileLoading ? '刷新中...' : '刷新列表'}
                </button>
                <button
                  type="button"
                  onClick={handleImportSelectedScanFile}
                  style={{ ...smallActionButtonStyle, flex: '1 1 120px', minWidth: 120 }}
                  disabled={scanFileImporting || !selectedScanPath}
                >
                  {scanFileImporting ? '导入中...' : '导入并映射到 3D'}
                </button>
              </div>
            </div>

            {currentScanMeta ? (
              <div style={{ fontSize: 11, color: '#7ea2c9', lineHeight: 1.6, overflowWrap: 'anywhere' }}>
                {`文件：${currentScanMeta.name} | ${(Number(currentScanMeta.size_bytes || 0) / 1024).toFixed(1)} KB | ${currentScanMeta.mtime_iso || ''}`}
              </div>
            ) : selectedScanPath ? (
              <div style={{ fontSize: 11, color: '#7ea2c9', lineHeight: 1.6, overflowWrap: 'anywhere' }}>
                {selectedScanPath}
              </div>
            ) : null}

            {scanFileError ? (
              <div style={{ fontSize: 11, color: '#ff9fb0' }}>{scanFileError}</div>
            ) : null}
            {scanPreviewError ? (
              <div style={{ fontSize: 11, color: '#ff9fb0' }}>{scanPreviewError}</div>
            ) : null}
            {queryFeedback ? (
              <div style={{ fontSize: 11, color: '#8fd4ff', lineHeight: 1.6 }}>{queryFeedback}</div>
            ) : null}
            {scanImportSummary ? (
              <div style={{ fontSize: 11, color: '#7eb8ff', lineHeight: 1.6 }}>
                {`来源：${scanImportSummary.source} | 导入概念集：${scanImportSummary.importedConcepts} | 类别：${scanImportSummary.importedCategories} | 扫描名词总数：${scanImportSummary.totalNouns} | 最小回路名词：${scanImportSummary.minimalCircuitNouns || 0} | 反事实对：${scanImportSummary.counterfactualPairs || 0}`}
              </div>
            ) : null}

            <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10, display: 'grid', gap: 10 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'baseline' }}>
                <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>{`${scanPreview.typeLabel} | ${scanPreview.title}`}</div>
                <div style={{ fontSize: 10, color: '#7ea2c9' }}>{scanPreviewLoading ? '预览加载中...' : '预览就绪'}</div>
              </div>
              <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7 }}>{scanPreview.subtitle}</div>

              {scanPreview.metricRows.length > 0 ? (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 8 }}>
                  {scanPreview.metricRows.map((item) => (
                    <div key={`artifact-metric-${item.label}`} style={{ borderRadius: 10, padding: 10, border: '1px solid rgba(122, 162, 255, 0.24)', background: 'rgba(7, 12, 25, 0.78)' }}>
                      <div style={{ fontSize: 10, color: '#7ea2c9', marginBottom: 3 }}>{item.label}</div>
                      <div style={{ fontSize: 14, fontWeight: 800, color: '#e4f0ff', overflowWrap: 'anywhere' }}>{item.value}</div>
                    </div>
                  ))}
                </div>
              ) : null}

              {!showScanPreviewInTopRight ? (
                <>
                  <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10, display: 'grid', gap: 6 }}>
                    <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>数据摘要与查看提示</div>
                    <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7 }}>
                      <div>{scanPreviewTheory.summary}</div>
                      <div>{`3D 关注点：${scanPreviewTheory.sceneHint}`}</div>
                      <div>{`当前数据视图：${languageFocus?.researchLayer || 'static_encoding'}`}</div>
                    </div>
                    <div style={{ display: 'grid', gap: 4 }}>
                      {scanPreviewTheory.metrics.map((item) => (
                        <div key={`artifact-data-${item.label}`} style={{ display: 'grid', gridTemplateColumns: '110px 1fr', gap: 8, fontSize: 11, color: '#9bb3de' }}>
                          <span>{item.label}</span>
                          <span style={{ color: '#dbe9ff', fontWeight: 700 }}>{item.value}</span>
                        </div>
                      ))}
                    </div>
                      {scanPreview.analysisLines.map((line) => (
                        <div key={line} style={{ fontSize: 11, color: '#8fd4ff', lineHeight: 1.6 }}>
                          {`• ${line}`}
                        </div>
                      ))}
                    </div>
                  </div>

                  <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10, display: 'grid', gap: 6 }}>
                    <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>原始数据</div>
                    <pre
                      style={{
                        margin: 0,
                        maxHeight: 320,
                        overflow: 'auto',
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                        fontSize: 11,
                        color: '#cfe2ff',
                        background: 'rgba(7, 12, 25, 0.82)',
                        border: '1px solid rgba(122, 162, 255, 0.22)',
                        borderRadius: 10,
                        padding: 10,
                      }}
                    >
                      {scanPreviewJson || '暂无原始数据'}
                    </pre>
                  </div>
                </>
              ) : null}
            </div>
          </div>
        )}
      </div>
      ) : (
      <div style={panelCardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 6 }}>理论分析已迁出主界面</div>
        <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7 }}>
          <div>主工作台现在只保留数据观察、差异比较、样本验证和回放。</div>
          <div>理论对象、理论解释和长期战略路线请到“战略层级路线图”中查看。</div>
        </div>
      </div>
      )}

      {false ? (
        <>
      <details style={panelCardStyle} open>
        <summary style={{ cursor: 'pointer', listStyle: 'none', fontSize: 14, fontWeight: 700, color: '#d4e3ff' }}>
          提取语料摘要
        </summary>
        <div style={{ fontSize: 11, color: '#7f95bb', lineHeight: 1.6, marginTop: 10, marginBottom: 10 }}>
          这里不是单独研究页，而是把当前 DNN 提取总量和导入状态直接压进控制面板，避免脱离主流程。
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 8 }}>
          {researchSnapshotRows.map((item) => (
            <div key={`research-snapshot-${item.label}`} style={{ borderRadius: 10, padding: 10, border: '1px solid rgba(122, 162, 255, 0.24)', background: 'rgba(7, 12, 25, 0.78)' }}>
              <div style={{ fontSize: 10, color: '#7ea2c9', marginBottom: 3 }}>{item.label}</div>
              <div style={{ fontSize: 16, fontWeight: 800, color: '#e4f0ff' }}>{item.value}</div>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 10, display: 'grid', gap: 4 }}>
          {importStatusRows.map((item) => (
            <div key={`import-status-${item.label}`} style={{ display: 'flex', justifyContent: 'space-between', gap: 8, fontSize: 11, color: '#9bb3de' }}>
              <span>{item.label}</span>
              <span style={{ color: '#dbe9ff', fontWeight: 700 }}>{item.value}</span>
            </div>
          ))}
        </div>
      </details>

      <details style={panelCardStyle} open>
        <summary style={{ cursor: 'pointer', listStyle: 'none', fontSize: 14, fontWeight: 700, color: '#d4e3ff' }}>
          当前对象的数据映射
        </summary>
        <div style={{ fontSize: 11, color: '#7f95bb', lineHeight: 1.6, marginTop: 10, marginBottom: 10 }}>
          只显示当前理论对象和当前动作最相关的研究数据，避免把所有提取结果一起堆进界面。
        </div>
        <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700, marginBottom: 6 }}>{currentTheoryResearch.summary}</div>
        <div style={{ display: 'grid', gap: 6 }}>
          {currentTheoryResearch.metrics.map((item) => (
            <div key={`theory-metric-${item.label}`} style={{ display: 'grid', gridTemplateColumns: '110px 1fr', gap: 8, fontSize: 11, color: '#9bb3de' }}>
              <span>{item.label}</span>
              <span style={{ color: '#dbe9ff', fontWeight: 700 }}>{item.value}</span>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 10, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10, fontSize: 11, color: '#8fd4ff', lineHeight: 1.7 }}>
          <div>{`当前动作说明：${ANALYSIS_MODE_RESEARCH_NOTES[analysisMode] || '暂无说明'}`}</div>
          <div>{`3D 关注点：${currentTheoryResearch.sceneHint}`}</div>
        </div>
      </details>

      <details style={panelCardStyle} open>
        <summary style={{ cursor: 'pointer', listStyle: 'none', fontSize: 14, fontWeight: 700, color: '#d4e3ff' }}>
          3D 明细与硬伤
        </summary>
        <div style={{ fontSize: 11, color: '#7f95bb', lineHeight: 1.6, marginTop: 10, marginBottom: 10 }}>
          这里把当前选中 3D 节点的研究明细和当前对象的主硬伤放在一起，减少“看数据要切页”的成本。
        </div>
        {selectedResearchDetails ? (
          <div style={{ display: 'grid', gap: 5, marginBottom: 10 }}>
            {selectedResearchDetails.detailRows.map((item) => (
              <div key={`selected-detail-${item.label}`} style={{ display: 'grid', gridTemplateColumns: '90px 1fr', gap: 8, fontSize: 11, color: '#9bb3de' }}>
                <span>{item.label}</span>
                <span style={{ color: '#dbe9ff', fontWeight: 700, overflowWrap: 'anywhere' }}>{item.value}</span>
              </div>
            ))}
            <div style={{ fontSize: 11, color: '#8fd4ff' }}>{`证据强度：${selectedResearchDetails.exactness}`}</div>
          </div>
        ) : (
          <div style={{ fontSize: 11, color: '#9bb3de', marginBottom: 10 }}>
            请先在 3D 场景中选中一个节点，这里会同步显示它的研究明细、来源层级和数据分组。
          </div>
        )}
        <div style={{ display: 'grid', gap: 6, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 10 }}>
          {researchRiskRows.map((item) => (
            <div key={item} style={{ fontSize: 11, color: '#ffcf91', lineHeight: 1.6 }}>
              {`• ${item}`}
            </div>
          ))}
        </div>
      </details>

      <div style={panelCardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>DNN 分层显示</div>
        <div style={{ fontSize: 11, color: '#7f95bb', lineHeight: 1.6, marginBottom: 10 }}>
          所有数据显示都基于当前 28 个 layer（层）主视图。默认先看基础信息；苹果概念核、静态编码层等算法内容，需要手动切到“算法显示”后再打开。
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 8, marginBottom: 10 }}>
          {[
            { id: 'basic', label: '基础信息', desc: '先看基础神经元和对象族数据' },
            { id: 'algorithm', label: '算法显示', desc: '手动打开苹果概念核、静态编码层等内容' },
          ].map((tab) => (
            <button
              key={`dnn-tab-${tab.id}`}
              type="button"
              onClick={() => setDnnPanelTab(tab.id)}
              style={{
                borderRadius: 10,
                border: `1px solid ${dnnPanelTab === tab.id ? 'rgba(126, 224, 255, 0.75)' : 'rgba(122, 162, 255, 0.28)'}`,
                background: dnnPanelTab === tab.id ? 'rgba(24, 101, 134, 0.36)' : 'rgba(7, 12, 25, 0.82)',
                color: '#dbe9ff',
                padding: '8px 10px',
                cursor: 'pointer',
                textAlign: 'left',
              }}
            >
              <div style={{ fontSize: 12, fontWeight: 700 }}>{tab.label}</div>
              <div style={{ fontSize: 10, color: '#88a6cf', marginTop: 2 }}>{tab.desc}</div>
            </button>
          ))}
        </div>
        {dnnPanelTab === 'basic' ? (
          <div style={{ display: 'grid', gap: 8 }}>
            <div
              style={{
                borderRadius: 10,
                border: '1px solid rgba(122, 162, 255, 0.18)',
                background: 'rgba(8, 13, 28, 0.62)',
                padding: '10px 12px',
                display: 'grid',
                gap: 8,
              }}
            >
              <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>{'当前基础摘要'}</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 8 }}>
                <div style={{ borderRadius: 8, background: 'rgba(255,255,255,0.03)', padding: '8px 10px' }}>
                  <div style={{ fontSize: 10, color: '#88a6cf' }}>{'当前概念'}</div>
                  <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>{currentBasicConceptEntry?.label || '-'}</div>
                </div>
                <div style={{ borderRadius: 8, background: 'rgba(255,255,255,0.03)', padding: '8px 10px' }}>
                  <div style={{ fontSize: 10, color: '#88a6cf' }}>{'基础神经元'}</div>
                  <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>{currentBasicConceptEntry?.count || 0}</div>
                </div>
                <div style={{ borderRadius: 8, background: 'rgba(255,255,255,0.03)', padding: '8px 10px' }}>
                  <div style={{ fontSize: 10, color: '#88a6cf' }}>{'当前选中'}</div>
                  <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>{selected?.label || selected?.concept || '未选中'}</div>
                </div>
              </div>
            </div>
            {DNN_DISPLAY_LEVEL_OPTIONS.filter((item) => ['basic_neurons', 'object_family'].includes(item.id)).map((item) => {
              const active = displayLevels?.[item.id] !== false;
              return (
                <label
                  key={`dnn-display-level-${item.id}`}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '18px 1fr',
                    gap: 10,
                    alignItems: 'start',
                    padding: '8px 10px',
                    borderRadius: 10,
                    border: `1px solid ${active ? 'rgba(96, 165, 250, 0.42)' : 'rgba(122, 162, 255, 0.16)'}`,
                    background: active ? 'rgba(17, 24, 39, 0.72)' : 'rgba(7, 12, 25, 0.48)',
                    cursor: 'pointer',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={active}
                    onChange={(e) => setDisplayLevels((prev) => ({ ...prev, [item.id]: e.target.checked }))}
                    style={{ marginTop: 2 }}
                  />
                  <span>
                    <span style={{ display: 'block', fontSize: 12, color: '#dbe9ff', fontWeight: 700 }}>{item.label}</span>
                    <span style={{ display: 'block', fontSize: 10, color: '#8fb0da', marginTop: 2, lineHeight: 1.5 }}>{item.desc}</span>
                  </span>
                </label>
              );
            })}
            <div
              style={{
                borderRadius: 10,
                border: '1px solid rgba(122, 162, 255, 0.18)',
                background: 'rgba(8, 13, 28, 0.62)',
                padding: '10px 12px',
                display: 'grid',
                gap: 8,
              }}
            >
              <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>基础名词 / 概念浏览</div>
              <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.6 }}>
                先看不同名词和概念的基础神经元信息。这里只显示基础层，不主动打开概念核、静态编码层等算法内容。
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {basicConceptEntries.map((entry) => (
                  <button
                    key={`basic-concept-${entry.id}`}
                    type="button"
                    onClick={() => {
                      setBasicConceptFocus(entry.id);
                      if (entry.id === 'fruit_general') {
                        setShowFruitGeneral?.(true);
                      } else if (entry.id !== 'apple') {
                        setShowFruit?.((prev) => ({ ...prev, [entry.id]: true }));
                      }
                    }}
                    style={{
                      borderRadius: 999,
                      border: `1px solid ${basicConceptFocus === entry.id ? entry.color : 'rgba(122, 162, 255, 0.28)'}`,
                      background: basicConceptFocus === entry.id ? `${entry.color}22` : 'rgba(7, 12, 25, 0.78)',
                      color: '#dbe9ff',
                      padding: '6px 10px',
                      cursor: 'pointer',
                      fontSize: 11,
                    }}
                  >
                    {`${entry.label} (${entry.count})`}
                  </button>
                ))}
              </div>
              {currentBasicConceptEntry ? (
                <div style={{ display: 'grid', gap: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, alignItems: 'baseline' }}>
                    <div style={{ fontSize: 12, color: currentBasicConceptEntry.color, fontWeight: 700 }}>{currentBasicConceptEntry.label}</div>
                    <div style={{ fontSize: 10, color: '#9bb3de' }}>{`${currentBasicConceptEntry.count} 个基础神经元`}</div>
                  </div>
                  <div style={{ fontSize: 11, color: '#9bb3de' }}>{currentBasicConceptEntry.desc}</div>
                  <div style={{ display: 'grid', gap: 6 }}>
                    {currentBasicConceptEntry.nodes.map((item, idx) => (
                      <button
                        key={`basic-concept-node-${currentBasicConceptEntry.id}-${item.id || idx}`}
                        type="button"
                        onClick={() => setSelected?.(item)}
                        style={{
                          borderRadius: 8,
                          border: '1px solid rgba(122, 162, 255, 0.18)',
                          background: 'rgba(7, 12, 25, 0.82)',
                          color: '#dbe9ff',
                          padding: '8px 10px',
                          display: 'grid',
                          gap: 2,
                          textAlign: 'left',
                          cursor: 'pointer',
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
                          <span style={{ fontSize: 11, fontWeight: 700 }}>{item.label || `${currentBasicConceptEntry.label} 神经元 ${idx + 1}`}</span>
                          <span style={{ fontSize: 10, color: currentBasicConceptEntry.color }}>{`L${item.layer} / N${item.neuron}`}</span>
                        </div>
                        <div style={{ fontSize: 10, color: '#9bb3de' }}>{`${item.metric || 'score'} = ${Number(item.value || item.strength || 0).toFixed(4)}`}</div>
                      </button>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        ) : (
          <div style={{ display: 'grid', gap: 8 }}>
            <div
              style={{
                borderRadius: 10,
                border: '1px solid rgba(122, 162, 255, 0.18)',
                background: 'rgba(8, 13, 28, 0.62)',
                padding: '10px 12px',
                display: 'grid',
                gap: 8,
              }}
            >
              <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>{'算法显示说明'}</div>
              <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.6 }}>
                {'先看基础信息，再按需要手动打开概念核、编码层和运行链路。下面的每个按钮只负责打开对应层级，不会默认强行叠加。'}
              </div>
            </div>
            {DNN_DISPLAY_LEVEL_OPTIONS.filter((item) => ['parameter_state', 'mechanism_chain', 'advanced_analysis'].includes(item.id)).map((item) => {
              const active = displayLevels?.[item.id] !== false;
              return (
                <label
                  key={`dnn-display-level-${item.id}`}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '18px 1fr',
                    gap: 10,
                    alignItems: 'start',
                    padding: '8px 10px',
                    borderRadius: 10,
                    border: `1px solid ${active ? 'rgba(96, 165, 250, 0.42)' : 'rgba(122, 162, 255, 0.16)'}`,
                    background: active ? 'rgba(17, 24, 39, 0.72)' : 'rgba(7, 12, 25, 0.48)',
                    cursor: 'pointer',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={active}
                    onChange={(e) => setDisplayLevels((prev) => ({ ...prev, [item.id]: e.target.checked }))}
                    style={{ marginTop: 2 }}
                  />
                  <span>
                    <span style={{ display: 'block', fontSize: 12, color: '#dbe9ff', fontWeight: 700 }}>{item.label}</span>
                    <span style={{ display: 'block', fontSize: 10, color: '#8fb0da', marginTop: 2, lineHeight: 1.5 }}>{item.desc}</span>
                  </span>
                </label>
              );
            })}
            <div style={{ fontSize: 12, color: '#e4f0ff', fontWeight: 700 }}>{'算法入口'}</div>
            <div style={{ fontSize: 10, color: '#8fb0da', lineHeight: 1.6 }}>
              {'概念核：查看苹果概念本身；编码层：查看静态编码层；运行链路：查看 layer 之间的参数传播。'}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 8 }}>
              <button
                type="button"
                onClick={() => {
                  setLanguageFocus?.((prev) => ({ ...prev, researchLayer: 'static_encoding' }));
                  setShowAlgorithmConceptCore(false);
                  setShowAlgorithmStaticEncoding(true);
                  setShowAlgorithmRuntimeChain(false);
                  setDisplayLevels((prev) => ({
                    ...prev,
                    basic_neurons: true,
                    object_family: true,
                    parameter_state: true,
                    mechanism_chain: false,
                    advanced_analysis: false,
                  }));
                }}
                style={smallActionButtonStyle}
              >
                显示静态编码层
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowAlgorithmConceptCore(false);
                  setShowAlgorithmStaticEncoding(true);
                  setShowAlgorithmRuntimeChain(true);
                  setDisplayLevels((prev) => ({
                    ...prev,
                    basic_neurons: true,
                    object_family: true,
                    parameter_state: true,
                    mechanism_chain: true,
                    advanced_analysis: false,
                  }));
                }}
                style={smallActionButtonStyle}
              >
                显示运行链路
              </button>
            </div>
          </div>
        )}
        {dnnPanelTab === 'algorithm' ? (
          <div style={{ marginTop: 10, display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 8 }}>
            <button
              type="button"
              onClick={() => {
                setLanguageFocus?.((prev) => ({ ...prev, researchLayer: 'static_encoding' }));
                setShowAlgorithmConceptCore(true);
                setShowAlgorithmStaticEncoding(false);
                setShowAlgorithmRuntimeChain(false);
                setDisplayLevels((prev) => ({
                  ...prev,
                  basic_neurons: true,
                  object_family: true,
                  parameter_state: true,
                  mechanism_chain: false,
                  advanced_analysis: false,
                }));
              }}
              style={smallActionButtonStyle}
            >
              {'\u663e\u793a\u82f9\u679c\u6982\u5ff5\u6838'}
            </button>
            <button
              type="button"
              onClick={() => {
                setLanguageFocus?.((prev) => ({ ...prev, researchLayer: 'static_encoding' }));
                setShowAlgorithmConceptCore(false);
                setShowAlgorithmStaticEncoding(true);
                setShowAlgorithmRuntimeChain(false);
                setDisplayLevels((prev) => ({
                  ...prev,
                  basic_neurons: true,
                  object_family: true,
                  parameter_state: true,
                  mechanism_chain: false,
                  advanced_analysis: false,
                }));
              }}
              style={smallActionButtonStyle}
            >
              {'\u663e\u793a\u9759\u6001\u7f16\u7801\u5c42'}
            </button>
            <button
              type="button"
              onClick={() => {
                setShowAlgorithmConceptCore(false);
                setShowAlgorithmStaticEncoding(true);
                setShowAlgorithmRuntimeChain(true);
                setDisplayLevels((prev) => ({
                  ...prev,
                  basic_neurons: true,
                  object_family: true,
                  parameter_state: true,
                  mechanism_chain: true,
                  advanced_analysis: false,
                }));
              }}
              style={smallActionButtonStyle}
            >
              {'\u663e\u793a\u8fd0\u884c\u94fe\u8def'}
            </button>
          </div>
        ) : null}
      </div>
      <div style={panelCardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 6 }}>理论分析已迁出主界面</div>
        <div style={{ fontSize: 11, color: '#9bb3de', lineHeight: 1.7 }}>
          <div>主工作台现在只保留数据观察、差异比较、样本验证和回放。</div>
          <div>理论对象、理论解释和长期战略路线请到“战略层级路线图”中查看。</div>
        </div>
      </div>

      {false ? (
      <div style={panelCardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 6 }}>实验动作层（第二层）</div>
        <div style={{ fontSize: 11, color: '#7f95bb', lineHeight: 1.6, marginBottom: 10 }}>
          当前对象下只显示相关的观测/提取/验证/系统动作，避免 10 个模式一起堆出相同动画。
        </div>
        <div style={{ padding: 12, border: '1px solid rgba(255,255,255,0.05)', borderRadius: 8 }}>
          {filteredStageGroups.map((group) => {
            const GroupIcon = group.icon || Activity;
            const stageModes = group.items
              .map((id) => modeMetaById[id])
              .filter(Boolean);
            if (stageModes.length === 0) {
              return null;
            }
            return (
              <div key={group.id} style={{ marginBottom: 8 }}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  marginBottom: 6,
                  padding: '4px 8px',
                  background: 'rgba(255,255,255,0.03)',
                  borderRadius: 4,
                }}>
                  <GroupIcon size={14} color="#888" />
                  <span style={{ fontSize: 11, fontWeight: 600, color: '#888' }}>{group.label}</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 4 }}>
                  {stageModes.map((mode) => {
                    const active = analysisMode === mode.id;
                    const ModeIcon = ANALYSIS_MODE_ICONS[mode.id] || Activity;
                    return (
                      <button
                        key={mode.id}
                        type="button"
                        onClick={() => setAnalysisMode(mode.id)}
                        title={mode.desc}
                        style={{
                          padding: '8px 4px',
                          backgroundColor: active ? 'rgba(68, 136, 255, 0.2)' : 'transparent',
                          color: active ? '#4488ff' : '#666',
                          border: active ? '1px solid rgba(68, 136, 255, 0.4)' : '1px solid transparent',
                          borderRadius: 6,
                          cursor: 'pointer',
                          fontSize: 11,
                          fontWeight: 500,
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          gap: 4,
                        }}
                      >
                        <ModeIcon size={14} />
                        <span>{mode.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
        <div style={{ fontSize: 11, color: '#7f95bb', marginTop: 8, lineHeight: 1.6 }}>
          {modeMetaById[analysisMode]?.desc || ''}
        </div>
        <div style={{ fontSize: 11, color: '#9bb3de', marginTop: 6, lineHeight: 1.6 }}>
          {`对象 -> 动作：${currentTheoryObject?.labelZh || '-'} -> ${modeMetaById[analysisMode]?.label || '-'}`}
        </div>
        {summary.statusText ? (
          <div style={{ fontSize: 11, color: '#9bb3de', marginTop: 6 }}>{summary.statusText}</div>
        ) : null}
        {modeMetrics?.length > 0 && (
          <div style={{ marginTop: 8, display: 'grid', gap: 4 }}>
            {modeMetrics.map((metric, idx) => (
              <div key={`${metric.label}-${idx}`} style={{ fontSize: 11, color: '#9bb3de' }}>
                {`${metric.label}: ${metric.value}`}
              </div>
            ))}
          </div>
        )}
      </div>
      ) : null}

      {false && analysisMode === 'dynamic_prediction' && (
        <div style={panelCardStyle}>
          <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>当前模式参数 · 动态预测</div>
          <textarea
            value={predictPrompt}
            onChange={(e) => setPredictPrompt(e.target.value)}
            rows={2}
            placeholder="输入上下文，例如：概念 是 一种 结构"
            style={textAreaStyle}
          />
          <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
            <button type="button" onClick={() => setPredictPlaying((v) => !v)} style={smallActionButtonStyle}>
              {predictPlaying ? 'Pause' : 'Play'}
            </button>
            <button type="button" onClick={handlePredictStepForward} style={smallActionButtonStyle}>Step</button>
            <button type="button" onClick={handlePredictReset} style={smallActionButtonStyle}>Reset</button>
          </div>
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 11, color: '#9eb4dd', marginBottom: 4 }}>
              {`Speed ${predictSpeed.toFixed(2)}x | Step ${predictStep + 1}/${predictChain.length || 0} | Layer ${(predictLayerProgress * 27).toFixed(1)}`}
            </div>
            <input type="range" min={0.4} max={2.4} step={0.1} value={predictSpeed} onChange={(e) => setPredictSpeed(Number(e.target.value))} style={{ width: '100%' }} />
          </div>
        </div>
      )}

      {false && analysisMode !== 'dynamic_prediction' && analysisMode !== 'static' && (
        <div style={panelCardStyle}>
          <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 8 }}>当前模式参数 · 机制实验</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button type="button" onClick={() => setMechanismPlaying((v) => !v)} style={smallActionButtonStyle}>{mechanismPlaying ? 'Pause' : 'Play'}</button>
            <button type="button" onClick={handleMechanismStepForward} style={smallActionButtonStyle}>Step</button>
            <button type="button" onClick={handleMechanismReset} style={smallActionButtonStyle}>Reset</button>
          </div>
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 11, color: '#9eb4dd', marginBottom: 4 }}>{`Mechanism Speed ${mechanismSpeed.toFixed(2)}x | Tick ${mechanismTick}`}</div>
            <input type="range" min={0.4} max={2.4} step={0.1} value={mechanismSpeed} onChange={(e) => setMechanismSpeed(Number(e.target.value))} style={{ width: '100%' }} />
          </div>

          {analysisMode === 'causal_intervention' && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 11, color: '#9eb4dd', marginBottom: 4 }}>{`Intervention Sparsity ${interventionSparsity.toFixed(2)}`}</div>
              <input type="range" min={0.1} max={1} step={0.05} value={interventionSparsity} onChange={(e) => setInterventionSparsity(Number(e.target.value))} style={{ width: '100%' }} />
            </div>
          )}

          {analysisMode === 'feature_decomposition' && (
            <div style={{ marginTop: 8, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
              {FEATURE_AXES.map((axis, idx) => (
                <button
                  key={axis}
                  type="button"
                  onClick={() => setFeatureAxis(idx)}
                  style={{
                    borderRadius: 8,
                    border: `1px solid ${featureAxis === idx ? 'rgba(126, 224, 255, 0.75)' : 'rgba(122, 162, 255, 0.35)'}`,
                    background: featureAxis === idx ? 'rgba(24, 101, 134, 0.38)' : 'rgba(7, 12, 25, 0.82)',
                    color: '#dbe9ff',
                    fontSize: 11,
                    padding: '6px 8px',
                    cursor: 'pointer',
                  }}
                >
                  {axis}
                </button>
              ))}
            </div>
          )}

          {analysisMode === 'compositionality' && (
            <div style={{ marginTop: 8, display: 'grid', gap: 6 }}>
              {['size', 'sweetness', 'color'].map((k) => (
                <div key={k}>
                  <div style={{ fontSize: 11, color: '#9eb4dd', marginBottom: 2 }}>{`${k}: ${compositionWeights[k].toFixed(2)}`}</div>
                  <input
                    type="range"
                    min={0.05}
                    max={1}
                    step={0.01}
                    value={compositionWeights[k]}
                    onChange={(e) => setCompositionWeights((prev) => ({ ...prev, [k]: Number(e.target.value) }))}
                    style={{ width: '100%' }}
                  />
                </div>
              ))}
            </div>
          )}

          {analysisMode === 'counterfactual' && (
            <textarea
              value={counterfactualPrompt}
              onChange={(e) => setCounterfactualPrompt(e.target.value)}
              rows={2}
              placeholder="输入反事实提示词"
              style={{ ...textAreaStyle, marginTop: 8 }}
            />
          )}

          {analysisMode === 'robustness' && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 11, color: '#9eb4dd', marginBottom: 4 }}>{`Perturb Trials ${robustnessTrials}`}</div>
              <input type="range" min={2} max={12} step={1} value={robustnessTrials} onChange={(e) => setRobustnessTrials(Number(e.target.value))} style={{ width: '100%' }} />
            </div>
          )}

          {analysisMode === 'minimal_circuit' && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 11, color: '#9eb4dd', marginBottom: 4 }}>{`Subset Size ${minimalSubsetSize}`}</div>
              <input type="range" min={3} max={32} step={1} value={minimalSubsetSize} onChange={(e) => setMinimalSubsetSize(Number(e.target.value))} style={{ width: '100%' }} />
            </div>
          )}
        </div>
      )}

      <div style={panelCardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>概念生成</div>
        <div style={{ fontSize: 11, color: '#7f95bb', lineHeight: 1.6, marginBottom: 10 }}>
          在主工作台里快速注入一个概念与类别，用来观察编码分布、类别比较和 3D 节点变化。
        </div>
        <div style={{ display: 'grid', gap: 8 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
            <div style={{ fontSize: 12, color: '#9eb4dd' }}>名称</div>
            <input
              value={queryInput}
              onChange={(e) => setQueryInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleGenerateQuery();
                }
              }}
              placeholder="输入名称，例如：猫 / 太阳 / 量子"
              style={inputStyle}
            />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
            <div style={{ fontSize: 12, color: '#9eb4dd' }}>类别</div>
            <input
              value={queryCategoryInput}
              onChange={(e) => setQueryCategoryInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleGenerateQuery();
                }
              }}
              placeholder="输入类别，例如：动物 / 天体 / 抽象概念"
              style={inputStyle}
            />
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <button type="button" onClick={handleGenerateQuery} style={smallActionButtonStyle}>
              Generate
            </button>
          </div>
        </div>
        {queryFeedback ? (
          <div style={{ marginTop: 8, fontSize: 11, color: '#8fd4ff' }}>{queryFeedback}</div>
        ) : null}
      </div>

      <div style={panelCardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>研究资产导入</div>
        <div style={{ fontSize: 11, color: '#7f95bb', lineHeight: 1.6, marginBottom: 10 }}>
          把 mass noun、多维编码、硬伤实验、四任务和统一解码结果当作研究资产单独管理，不和主流程参数混在一起。
        </div>
        <div style={{ display: 'grid', gap: 8 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
            <div style={{ fontSize: 12, color: '#9eb4dd' }}>导入数</div>
            <input
              type="number"
              min={1}
              max={120}
              value={scanImportLimit}
              onChange={(e) => setScanImportLimit(Number(e.target.value))}
              style={inputStyle}
            />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
            <div style={{ fontSize: 12, color: '#9eb4dd' }}>每概念</div>
            <input
              type="number"
              min={4}
              max={64}
              value={scanImportTopK}
              onChange={(e) => setScanImportTopK(Number(e.target.value))}
              style={inputStyle}
            />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: `56px ${fixedFileControlWidth}px`, gap: 8, alignItems: 'center' }}>
            <div style={{ fontSize: 12, color: '#9eb4dd' }}>文件</div>
            <select
              value={selectedScanPath}
              onChange={(e) => setSelectedScanPath(e.target.value)}
              style={{
                ...inputStyle,
                width: fixedFileControlWidth,
                minWidth: fixedFileControlWidth,
                maxWidth: fixedFileControlWidth,
                boxSizing: 'border-box',
                display: 'block',
                whiteSpace: 'nowrap',
                textOverflow: 'ellipsis',
                overflow: 'hidden',
              }}
              disabled={scanFileLoading || filteredScanFileOptions.length === 0}
            >
              {filteredScanFileOptions.length === 0 ? (
                <option value="">{scanFileLoading ? '扫描中...' : '未发现可导入文件'}</option>
              ) : (
                filteredScanFileOptions.map((f) => (
                  <option key={f.path} value={f.path}>
                    {formatScanOptionLabel(f)}
                  </option>
                ))
              )}
            </select>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: `56px ${fixedFileControlWidth}px`, gap: 8, alignItems: 'center' }}>
            <div style={{ fontSize: 12, color: '#9eb4dd' }}>动画</div>
            <select
              value={animationMode}
              onChange={(e) => setAnimationMode(e.target.value)}
              style={{
                ...inputStyle,
                width: fixedFileControlWidth,
                minWidth: fixedFileControlWidth,
                maxWidth: fixedFileControlWidth,
                boxSizing: 'border-box',
                display: 'block',
              }}
            >
              {animationModes.map((opt) => (
                <option key={`bottom-anim-${opt.id}`} value={opt.id}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
          <div style={{ fontSize: 11, color: '#7ea2c9', lineHeight: 1.6 }}>
            {animationModes.find((opt) => opt.id === animationMode)?.desc || '当前未启用附加动画。'}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: `56px ${fixedFileControlWidth}px`, gap: 8, alignItems: 'center' }}>
            <div style={{ fontSize: 12, color: '#9eb4dd' }}>筛选</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6 }}>
              {[
                { id: 'multidim', label: '多维编码' },
                { id: 'mass_noun', label: '名词扫描' },
                { id: 'hard_problem', label: '硬伤实验' },
                { id: 'four_tasks', label: '四任务' },
                { id: 'unified_decode', label: '统一解码' },
                { id: 'all', label: '全部' },
              ].map((opt) => (
                <button
                  key={`scan-filter-${opt.id}`}
                  type="button"
                  onClick={() => setScanFileFilter(opt.id)}
                  style={{
                    borderRadius: 8,
                    border: `1px solid ${scanFileFilter === opt.id ? 'rgba(126, 224, 255, 0.75)' : 'rgba(122, 162, 255, 0.35)'}`,
                    background: scanFileFilter === opt.id ? 'rgba(24, 101, 134, 0.38)' : 'rgba(7, 12, 25, 0.82)',
                    color: '#dbe9ff',
                    fontSize: 11,
                    padding: '6px 8px',
                    cursor: 'pointer',
                  }}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
          <div style={{ fontSize: 11, color: '#7ea2c9' }}>
            {`候选文件: ${filteredScanFileOptions.length}/${scanFileOptions.length}（当前筛选: ${scanFileFilterLabelMap[scanFileFilter] || scanFileFilter}）`}
          </div>
          <div
            style={{
              display: 'flex',
              width: fixedFileControlWidth + 56 + 8,
              maxWidth: fixedFileControlWidth + 56 + 8,
              minWidth: fixedFileControlWidth + 56 + 8,
              gap: 8,
              flexWrap: 'wrap',
            }}
          >
            <button
              type="button"
              onClick={refreshScanFileOptions}
              style={{ ...smallActionButtonStyle, flex: '1 1 120px', minWidth: 120 }}
              disabled={scanFileLoading}
            >
              {scanFileLoading ? '刷新中...' : '刷新列表'}
            </button>
            <button
              type="button"
              onClick={handleImportSelectedScanFile}
              style={{ ...smallActionButtonStyle, flex: '1 1 120px', minWidth: 120 }}
              disabled={scanFileImporting || !selectedScanPath}
            >
              {scanFileImporting ? '导入中...' : '导入选中文件'}
            </button>
          </div>
          {selectedScanPath ? (
            <div
              style={{
                width: fixedFileControlWidth + 56 + 8,
                maxWidth: fixedFileControlWidth + 56 + 8,
                minWidth: fixedFileControlWidth + 56 + 8,
                fontSize: 11,
                color: '#7ea2c9',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
              title={selectedScanPath}
            >
              {(() => {
                const meta = scanFileOptions.find((f) => f.path === selectedScanPath);
                if (!meta) {
                  const fallbackName = String(selectedScanPath).replace(/\\/g, '/').split('/').pop();
                  return `已选: ${fallbackName || 'scan.json'}`;
                }
                const kb = (Number(meta.size_bytes || 0) / 1024).toFixed(1);
                return `已选: ${meta.name} | ${kb} KB | ${meta.mtime_iso || ''}`;
              })()}
            </div>
          ) : null}
          {scanFileError ? (
            <div style={{ fontSize: 11, color: '#ff9fb0' }}>{scanFileError}</div>
          ) : null}
          {scanImportSummary ? (
            <div style={{ fontSize: 11, color: '#7eb8ff', lineHeight: 1.6 }}>
              {`来源: ${scanImportSummary.source} | 导入概念集: ${scanImportSummary.importedConcepts} | 类别: ${scanImportSummary.importedCategories} | 扫描名词总数: ${scanImportSummary.totalNouns} | 最小回路名词: ${scanImportSummary.minimalCircuitNouns || 0} | 反事实对: ${scanImportSummary.counterfactualPairs || 0}`}
            </div>
          ) : null}
        </div>
      </div>
        </>
      ) : null}

      {false ? (
      <div style={panelCardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>三维编码设置</div>
        <div style={{ fontSize: 11, color: '#7f95bb', lineHeight: 1.6, marginBottom: 10 }}>
          管理 `style / logic / syntax` 三维探针的可见性、TopN 和当前显示维度。
        </div>
        <div style={{ display: 'grid', gap: 8 }}>
          <div style={{ fontSize: 11, color: '#7ea2c9' }}>
            {multidimProbeData
              ? `已导入探针，当前显示维度: ${DIMENSION_LABELS[multidimActiveDimension] || multidimActiveDimension}，层谱点数: ${multidimLayerProfile?.length || 0}`
              : '未导入三维探针 JSON（multidim_encoding_probe.json）'}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
            <div style={{ fontSize: 12, color: '#9eb4dd' }}>TopN</div>
            <input
              type="number"
              min={16}
              max={256}
              value={multidimTopN}
              onChange={(e) => setMultidimTopN(Number(e.target.value))}
              style={inputStyle}
            />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6 }}>
            {['style', 'logic', 'syntax'].map((dim) => (
              <button
                key={`dim-active-${dim}`}
                type="button"
                onClick={() => setMultidimActiveDimension(dim)}
                style={{
                  borderRadius: 8,
                  border: `1px solid ${multidimActiveDimension === dim ? ROLE_COLORS[dim] : 'rgba(122,162,255,0.35)'}`,
                  background: multidimActiveDimension === dim ? 'rgba(42,71,132,0.82)' : 'rgba(7, 12, 25, 0.82)',
                  color: '#dbe9ff',
                  fontSize: 11,
                  padding: '6px 8px',
                  cursor: 'pointer',
                }}
              >
                {DIMENSION_LABELS[dim]}
              </button>
            ))}
          </div>
          <div style={{ display: 'grid', gap: 6 }}>
            {['style', 'logic', 'syntax'].map((dim) => (
              <label key={`dim-vis-${dim}`} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: '#9eb4dd' }}>
                <input
                  type="checkbox"
                  checked={multidimVisible[dim] !== false}
                  onChange={(e) => setMultidimVisible((prev) => ({ ...prev, [dim]: e.target.checked }))}
                />
                <span style={{ color: ROLE_COLORS[dim] }}>●</span>
                <span>{`${DIMENSION_LABELS[dim]}神经元`}</span>
              </label>
            ))}
          </div>
          <div style={{ fontSize: 11, color: '#7ea2c9', lineHeight: 1.6 }}>
            {multidimCausalData
              ? `对角优势 style=${toSafeNumber(multidimCausalData?.diagonal_advantage?.style, 0).toFixed(4)} / logic=${toSafeNumber(multidimCausalData?.diagonal_advantage?.logic, 0).toFixed(4)} / syntax=${toSafeNumber(multidimCausalData?.diagonal_advantage?.syntax, 0).toFixed(4)}`
              : '未导入三维因果消融 JSON（multidim_causal_ablation.json）'}
          </div>
        </div>
      </div>
      ) : null}

      {false ? (
      <div style={panelCardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>已生成概念集</div>
        <div style={{ fontSize: 11, color: '#7f95bb', lineHeight: 1.6, marginBottom: 10 }}>
          这里保留当前已经生成并注入到 3D 场景中的概念集合，便于清理、比较和重复实验。
        </div>
        <div style={{ display: 'grid', gap: 6 }}>
          {querySets.length === 0 ? (
            <div style={{ fontSize: 11, color: '#6f84ad' }}>尚未生成概念神经元。</div>
          ) : (
            querySets.map((set) => (
              <div key={set.id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 12, color: '#9eb4dd' }}>
                <span><span style={{ color: set.color }}>●</span>{` ${set.name} [${set.category}] (${set.nodes.length})`}</span>
                <button type="button" onClick={() => removeQuerySet(set.id)} style={{ ...smallActionButtonStyle, padding: '2px 8px', fontSize: 11 }}>Remove</button>
              </div>
            ))
          )}
        </div>
      </div>
      ) : null}

    </div>
  );
}
*/

export function AppleNeuron3DTab({ panelPosition = 'right', sceneHeight = '74vh', workspace: externalWorkspace } = {}) {
  const internalWorkspace = useAppleNeuronWorkspace();
  const workspace = externalWorkspace || internalWorkspace;
  const isPanelLeft = panelPosition === 'left';

  return (
    <div style={{ animation: 'roadmapFade 0.6s ease-out', display: 'grid', gridTemplateColumns: isPanelLeft ? '340px 1fr' : '1fr 340px', gap: 20 }}>
      {isPanelLeft ? (
        <>
          <LanguageResearchControlPanel workspace={workspace} structureTab="circuit" />
          <AppleNeuronMainScene workspace={workspace} sceneHeight={sceneHeight} />
        </>
      ) : (
        <>
          <AppleNeuronMainScene workspace={workspace} sceneHeight={sceneHeight} />
          <LanguageResearchControlPanel workspace={workspace} structureTab="circuit" />
        </>
      )}
    </div>
  );
}
