/**
 * AppleNeuron3D 常量与配置数据
 * 从 AppleNeuron3DTab.jsx 拆分而来
 */

export const LAYER_COUNT = 28;
export const DFF = 18944;

// ---- 模型配置表 ----
export const MODEL_CONFIGS = {
  'qwen3-4b': {
    name: 'Qwen3-4B',
    layers: 36,
    dModel: 2560,
    nHeads: 32,
    headDim: 128,
    mlpDim: 9728,
    vocabSize: 151936,
    type: 'Decoder-Only',
    color: '#60a5fa',
  },
  'glm4-9b': {
    name: 'GLM4-9B-Chat',
    layers: 40,
    dModel: 4096,
    nHeads: 32,
    headDim: 128,
    mlpDim: 13696,
    vocabSize: 151552,
    type: 'Decoder-Only',
    color: '#34d399',
  },
  'ds7b': {
    name: 'DeepSeek-R1-7B',
    layers: 28,
    dModel: 3584,
    nHeads: 28,
    headDim: 128,
    mlpDim: 18944,
    vocabSize: 152064,
    type: 'Dense Decoder-Only',
    color: '#fbbf24',
  },
};
export const QUERY_NODE_COUNT = 12;
export const IMPORTED_QUERY_NODE_MAX = 18;
export const MAIN_API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

export const APPLE_CORE_NEURONS = [
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

export const FRUIT_GENERAL_NEURONS = [
  { layer: 4, neuron: 902, score: 2.86 },
  { layer: 8, neuron: 2148, score: 2.61 },
  { layer: 13, neuron: 4970, score: 2.48 },
  { layer: 19, neuron: 6804, score: 2.22 },
  { layer: 6, neuron: 1510, score: 2.73 },
  { layer: 11, neuron: 3416, score: 2.57 },
  { layer: 15, neuron: 5522, score: 2.39 },
  { layer: 22, neuron: 8014, score: 2.11 },
];

export const FRUIT_SPECIFIC_NEURONS = {
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

export const FRUIT_COLORS = {
  apple: '#fb7185',
  pear: '#facc15',
  banana: '#fde047',
  orange: '#fb923c',
  grape: '#c084fc',
  peach: '#f9a8d4',
  mango: '#f59e0b',
};

export const ROLE_COLORS = {
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

export const DIMENSION_LABELS = {
  style: '风格维度',
  logic: '逻辑维度',
  syntax: '句法维度',
};

export const APPLE_SWITCH_MECHANISM_SCHEMA = 'apple_switch_mechanism_view.v1';
export const APPLE_SWITCH_MODEL_COLORS = {
  qwen3: '#60a5fa',
  deepseek7b: '#34d399',
};

export const APPLE_SWITCH_ROLE_LABELS = {
  anchor_neuron: '锚点神经元',
  main_booster_1: '主增强头 1',
  main_booster_2: '主增强头 2',
  skeleton_head_1: '骨架头 1',
  skeleton_head_2: '骨架头 2',
  bridge_head: '桥接头',
  heldout_booster: '校正/补强头',
};

export const DEFAULT_PREDICT_PROMPT = '';
export const PREDICT_CHAIN_LENGTH = 10;

export const TOKEN_TRANSITIONS = {
  概念: ['是', '一种', '结构', '系统', '表达'],
  模型: ['通过', '层级', '编码', '形成', '预测'],
  concept: ['is', 'a', 'structured', 'representation', 'in'],
  model: ['builds', 'multi-layer', 'features', 'for', 'prediction'],
  is: ['a', 'structured', 'mapping', 'inside', 'the'],
  a: ['concept', 'model', 'token', 'signal', 'pattern'],
};

export const TOPIC_FALLBACKS = [
  { keywords: ['概念', 'concept'], tokens: ['是', '一种', '结构', '可以', '在', '层级', '传播'] },
  { keywords: ['模型', 'model'], tokens: ['通过', '多层', '机制', '进行', '编码', '并', '预测'] },
];

export const DEFAULT_CHAIN_TOKENS = ['is', 'a', 'concept', 'mapped', 'through', 'layers', 'into', 'next', 'token'];

export const ANALYSIS_MODE_OPTIONS = [
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

export const APPLE_ANIMATION_OPTIONS = [
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

export const ICSPB_THEORY_OBJECTS = [
  { id: 'family_patch', label: 'family patch', labelZh: '族群底座', desc: '看同一概念族是否形成稳定 patch 底座与共享群落。', color: '#7dd3fc', roleWeights: { macro: 1, fruitGeneral: 0.95, query: 0.82, micro: 0.72, route: 0.68, style: 0.35, logic: 0.35, syntax: 0.35, unifiedDecode: 0.38, hardBinding: 0.28, hardLong: 0.25, hardLocal: 0.25, hardTriplet: 0.25, background: 0.06 } },
  { id: 'concept_section', label: 'concept section', labelZh: '概念截面', desc: '看概念截面、局部偏移与最小语义改动是否保持局部连续。', color: '#c084fc', roleWeights: { micro: 1, query: 0.94, macro: 0.82, route: 0.72, fruitGeneral: 0.58, style: 0.52, logic: 0.52, syntax: 0.52, unifiedDecode: 0.4, hardBinding: 0.3, hardLong: 0.3, hardLocal: 0.28, hardTriplet: 0.3, background: 0.06 } },
  { id: 'attribute_fiber', label: 'attribute fiber', labelZh: '属性纤维', desc: '看颜色、形状、甜度等属性是否沿可组合纤维方向分离。', color: '#34d399', roleWeights: { style: 1, logic: 0.82, syntax: 0.7, micro: 0.8, query: 0.68, macro: 0.6, route: 0.56, unifiedDecode: 0.56, hardBinding: 0.25, hardLong: 0.24, hardLocal: 0.24, hardTriplet: 0.26, background: 0.05 } },
  { id: 'relation_context_fiber', label: 'relation-context fiber', labelZh: '关系/语境纤维', desc: '看关系和语境如何沿层间路径传播并重组。', color: '#22d3ee', roleWeights: { route: 1, query: 0.88, macro: 0.76, micro: 0.64, logic: 0.62, syntax: 0.58, style: 0.42, unifiedDecode: 0.54, hardLong: 0.42, hardTriplet: 0.48, hardBinding: 0.3, hardLocal: 0.3, background: 0.05 } },
  { id: 'admissible_update', label: 'admissible update', labelZh: '可容许更新', desc: '看什么样的局部改动既能更新知识，又不冲垮旧结构。', color: '#a3e635', roleWeights: { hardLocal: 1, hardBinding: 0.92, hardLong: 0.84, hardTriplet: 0.84, unifiedDecode: 0.72, route: 0.66, micro: 0.62, macro: 0.58, query: 0.52, style: 0.5, logic: 0.5, syntax: 0.5, background: 0.04 } },
  { id: 'restricted_readout', label: 'restricted readout', labelZh: '受限读出', desc: '看输出是否主要依赖少数关键节点、局部子回路和读出热点。', color: '#fb7185', roleWeights: { route: 1, micro: 0.92, macro: 0.84, query: 0.82, hardTriplet: 0.74, hardBinding: 0.62, unifiedDecode: 0.58, logic: 0.46, syntax: 0.46, style: 0.38, background: 0.04 } },
  { id: 'stage_conditioned_transport', label: 'stage-conditioned transport', labelZh: '阶段条件运输', desc: '看不同计算阶段是否切换不同的运输主路和层间热点。', color: '#38bdf8', roleWeights: { route: 1, macro: 0.86, micro: 0.78, query: 0.74, hardLong: 0.62, hardBinding: 0.46, unifiedDecode: 0.48, style: 0.34, logic: 0.34, syntax: 0.34, background: 0.04 } },
  { id: 'successor_aligned_transport', label: 'successor-aligned transport', labelZh: '后继对齐运输', desc: '看后继 token/状态是否沿稳定对齐路径产生，而不是随机跳变。', color: '#f59e0b', roleWeights: { route: 1, query: 0.9, micro: 0.74, macro: 0.7, hardLong: 0.68, hardTriplet: 0.52, unifiedDecode: 0.5, style: 0.32, logic: 0.38, syntax: 0.38, background: 0.04 } },
  { id: 'protocol_bridge', label: 'protocol bridge', labelZh: '协议桥', desc: '看内部编码如何进入任务接口、统一解码和最小可用闭环。', color: '#f97316', roleWeights: { unifiedDecode: 1, hardTriplet: 0.86, route: 0.84, query: 0.78, macro: 0.68, micro: 0.58, style: 0.54, logic: 0.54, syntax: 0.54, hardBinding: 0.48, hardLong: 0.48, hardLocal: 0.48, background: 0.04 } },
];

export const THEORY_OBJECT_MODE_MAP = {
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

export const FEATURE_AXES = ['color', 'taste', 'shape', 'category'];

export const DEFAULT_LANGUAGE_FOCUS = {
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

export const LANGUAGE_RESEARCH_LAYER_META = {
  static_encoding: { label: '静态编码层', color: '#8fd4ff' },
  dynamic_route: { label: '动态路径层', color: '#5eead4' },
  result_recovery: { label: '结果回收层', color: '#fbbf24' },
  propagation_encoding: { label: '传播编码层', color: '#f87171' },
  semantic_roles: { label: '语义角色层', color: '#c084fc' },
};

export const CONCEPT_ASSOCIATION_LAYER_META = [
  { id: 'basic_encoding', label: '基础编码', color: '#93c5fd', roles: ['micro', 'fruitGeneral', 'fruitSpecific', 'query'] },
  { id: 'static_encoding', label: '静态编码层', color: '#8fd4ff', roles: ['fruitGeneral', 'fruitSpecific', 'query', 'macro'] },
  { id: 'dynamic_route', label: '动态路径层', color: '#5eead4', roles: ['route', 'query', 'macro'] },
  { id: 'result_recovery', label: '结果回收层', color: '#fbbf24', roles: ['unifiedDecode', 'route', 'hardTriplet', 'hardBinding'] },
  { id: 'propagation_encoding', label: '传播编码层', color: '#f87171', roles: ['query', 'macro', 'route'] },
  { id: 'semantic_roles', label: '语义角色层', color: '#c084fc', roles: ['style', 'logic', 'syntax', 'query'] },
];

export const CONCEPT_ALIAS_MAP = {
  apple: ['苹果'], '苹果': ['apple'],
  pear: ['梨'], '梨': ['pear'],
  banana: ['香蕉'], '香蕉': ['banana'],
  orange: ['橙子', '橘子'], '橙子': ['orange'], '橘子': ['orange'],
  grape: ['葡萄'], '葡萄': ['grape'],
  peach: ['桃子'], '桃子': ['peach'],
  mango: ['芒果'], '芒果': ['mango'],
  fruit: ['水果'], '水果': ['fruit'],
  animal: ['动物'], '动物': ['animal'],
  cat: ['猫'], '猫': ['cat'],
  dog: ['狗'], '狗': ['dog'],
  bird: ['鸟'], '鸟': ['bird'],
  tiger: ['老虎'], '老虎': ['tiger'],
  lion: ['狮子'], '狮子': ['lion'],
  monkey: ['猴子'], '猴子': ['monkey'],
};

export const LANGUAGE_OVERLAY_META = {
  shared_base: { label: '共享基底', color: '#60a5fa' },
  local_delta: { label: '局部差分', color: '#f97316' },
  path_amplification: { label: '路径放大', color: '#22c55e' },
  semantic_roles: { label: '语义角色', color: '#a78bfa' },
  fidelity: { label: '来源保真', color: '#fb7185' },
};

export const LANGUAGE_RISK_META = {
  fidelity: '天然来源保真低',
  competition: '同类高竞争边界脆弱',
  closure: '修复强于原生闭合',
  brand: '品牌义边界弱',
  cross_model: '跨模型硬主核仍少',
};

export const DNN_DISPLAY_LEVEL_OPTIONS = [
  { id: 'basic_neurons', label: '基础神经元', desc: '显示当前 28 个 layer（层）中的基础有效神经元点。' },
  { id: 'object_family', label: '对象族数据', desc: '显示苹果、水果等对象族相关神经元。' },
  { id: 'parameter_state', label: '参数位数据', desc: '显示参数态节点、参数骨架和参数位详情。' },
  { id: 'mechanism_chain', label: '运行链路', desc: '显示参数链路和基础层间运行效果。' },
  { id: 'advanced_analysis', label: '高级分析', desc: '显示共享承载、偏置偏转、逐层放大等高级叠加层。' },
];

export const DNN_DISPLAY_PRESETS = {
  basic_only: { label: '只看基础', levels: { basic_neurons: true, object_family: false, parameter_state: false, mechanism_chain: false, advanced_analysis: false } },
  parameter_only: { label: '只看参数', levels: { basic_neurons: false, object_family: false, parameter_state: true, mechanism_chain: true, advanced_analysis: false } },
  runtime_focus: { label: '看运行链', levels: { basic_neurons: true, object_family: false, parameter_state: true, mechanism_chain: true, advanced_analysis: false } },
  all_on: { label: '全部打开', levels: { basic_neurons: true, object_family: true, parameter_state: true, mechanism_chain: true, advanced_analysis: true } },
};

export const DNN_RESEARCH_SNAPSHOT = {
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

export const THEORY_OBJECT_RESEARCH_MAP = {
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

export const ANALYSIS_MODE_RESEARCH_NOTES = {
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

export const HARD_PROBLEM_EXPERIMENT_LABELS = {
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

export const MODE_VISUALS = {
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
