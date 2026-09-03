export const RESEARCH_LAYER_DEFINITIONS = {
  atlas: {
    id: 'atlas',
    label: '机制图谱层',
    detail: 'atlas graph nodes / causal edges / evidence links',
  },
  aiOrbit: {
    id: 'aiOrbit',
    label: 'AI研究轨道层',
    detail: '多模型讨论 / 综合 / 脚本 / 运行 / 回写',
  },
  boundary: {
    id: 'boundary',
    label: '失败边界层',
    detail: 'weak / null / falsified / boundary evidence',
  },
  features: {
    id: 'features',
    label: '特征空间层',
    detail: 'SAE feature / dictionary atom / activation cluster',
  },
  causalPath: {
    id: 'causalPath',
    label: '因果路径层',
    detail: 'activation patch / path patch / ablation / restore',
  },
  dynamics: {
    id: 'dynamics',
    label: '动力学层',
    detail: 'spike / replay / control state / temporal stability',
  },
  heatmap: {
    id: 'heatmap',
    label: '热力图层',
    detail: '词嵌入序列与按层 HiddenState 的真实 top-k 状态矩阵',
  },
};

export const RESEARCH_PLUGINS = [
  {
    id: 'language-mechanism',
    name: '语言机制研究',
    shortName: '语言机制',
    routeType: 'mechanistic_interpretability',
    workspaceTab: 'main',
    defaultMode: 'evidence',
    target: '破解语言编码机制',
    focus: '按语言模式族追踪来源状态、关键层、真实神经元候选、残差写入和读出结果，形成可验证物理脉络。',
    status: '主线',
    defaultLayers: ['atlas'],
    layers3D: ['atlas', 'causalPath', 'boundary', 'aiOrbit'],
    panels: [
      { id: 'overview', label: '路线总览', defaultOpen: true },
      { id: 'evidence', label: '证据详情', defaultOpen: false },
      { id: 'experiment', label: '实验矩阵', defaultOpen: false },
    ],
    actions: [
      { id: 'load_latest_atlas', label: '加载最新图谱' },
      { id: 'run_patch', label: '因果验证' },
    ],
    modes: ['configure', 'evidence', 'ai_loop'],
  },
  {
    id: 'sae-features',
    name: 'SAE 特征解释',
    shortName: 'SAE特征',
    routeType: 'dictionary_learning',
    workspaceTab: 'main',
    defaultMode: 'evidence',
    target: '建立可解释特征坐标系',
    focus: '把神经元与通道重新投影到稀疏特征空间，追踪特征与语言任务的因果关系。',
    status: '扩展路线',
    defaultLayers: ['atlas'],
    layers3D: ['features', 'atlas', 'causalPath', 'boundary'],
    panels: [
      { id: 'overview', label: '路线总览', defaultOpen: true },
      { id: 'feature-map', label: '特征字典', defaultOpen: false },
      { id: 'evidence', label: '证据详情', defaultOpen: false },
      { id: 'experiment', label: '实验矩阵', defaultOpen: false },
    ],
    actions: [
      { id: 'set_feature_mode', label: '特征分析' },
      { id: 'run_patch', label: '验证特征' },
    ],
    modes: ['configure', 'evidence', 'ai_loop'],
  },
  {
    id: 'circuit-tracing',
    name: 'Circuit Tracing',
    shortName: 'Circuit',
    routeType: 'attribution_graph',
    workspaceTab: 'main',
    defaultMode: 'evidence',
    target: '追踪跨层计算回路',
    focus: '把候选机制压缩为可审计路径，强调上游因果源、关键边与闭合性验证。',
    status: '验证路线',
    defaultLayers: ['atlas'],
    layers3D: ['atlas', 'causalPath', 'boundary'],
    panels: [
      { id: 'overview', label: '路线总览', defaultOpen: true },
      { id: 'path', label: '因果路径', defaultOpen: false },
      { id: 'experiment', label: '实验矩阵', defaultOpen: false },
      { id: 'evidence', label: '证据详情', defaultOpen: false },
    ],
    actions: [
      { id: 'run_path_patch', label: '路径验证' },
      { id: 'load_latest_atlas', label: '加载图谱' },
    ],
    modes: ['configure', 'evidence'],
  },
  {
    id: 'heatmap-analysis',
    name: '状态热力图（Embedding + HiddenState）',
    shortName: '状态热力图',
    routeType: 'heatmap_matrix',
    workspaceTab: 'main',
    defaultMode: 'configure',
    target: '展示词嵌入序列与 HiddenState 的 top-k 热力图对齐展示',
    focus: '左侧展示词嵌入 top-k 序列，右侧按层展示 HiddenState top-k；以最少噪音、无 Phase 语义的方式观察状态场。',
    status: '可视分析路线',
    defaultLayers: ['heatmap'],
    layers3D: ['heatmap', 'boundary'],
    panels: [
      { id: 'overview', label: '路线总览', defaultOpen: true },
      { id: 'heatmap', label: 'Embedding+HiddenState 热力图', defaultOpen: true },
      { id: 'evidence', label: '数据与证据', defaultOpen: false },
    ],
    actions: [
      { id: 'show_heatmap', label: '显示热力图' },
      { id: 'load_heatmap_data', label: '加载真实数据' },
    ],
    modes: ['configure', 'evidence'],
  },
  {
    id: 'ai-research-loop',
    name: 'AI 自动研究',
    shortName: 'AI研究',
    routeType: 'research_agent',
    workspaceTab: 'main',
    defaultMode: 'ai_loop',
    target: '自动形成假设、实验与图谱',
    focus: '让多个模型讨论，主模型综合，自动编写脚本并把结果回写到机制图谱。',
    status: '自动化',
    defaultLayers: ['atlas', 'aiOrbit'],
    layers3D: ['atlas', 'aiOrbit', 'causalPath', 'boundary'],
    panels: [
      { id: 'overview', label: '路线总览', defaultOpen: true },
      { id: 'agent-log', label: '循环日志', defaultOpen: false },
      { id: 'experiment', label: '实验队列', defaultOpen: false },
      { id: 'evidence', label: '证据回写', defaultOpen: false },
    ],
    actions: [
      { id: 'open_ai_loop', label: 'AI研发窗' },
      { id: 'start_ai_cycle', label: '启动循环' },
    ],
    modes: ['configure', 'ai_loop', 'evidence'],
  },
  {
    id: 'snn-dynamics',
    name: 'SNN 动力学研究',
    shortName: 'SNN',
    routeType: 'spiking_neural_network',
    workspaceTab: 'snn',
    defaultMode: 'evidence',
    target: '观测脉冲神经动力学',
    focus: '分析放电、可塑性、时序传播与语言机制之间的动力学桥接关系。',
    status: '并行路线',
    defaultLayers: ['dynamics'],
    layers3D: ['dynamics', 'boundary'],
    panels: [
      { id: 'overview', label: '路线总览', defaultOpen: true },
      { id: 'dynamics', label: '动力学状态', defaultOpen: false },
      { id: 'experiment', label: '刺激实验', defaultOpen: false },
    ],
    actions: [
      { id: 'switch_snn', label: '进入SNN' },
    ],
    modes: ['configure', 'evidence'],
  },
  {
    id: 'icspb-core',
    name: 'ICSPB 工作台',
    shortName: 'ICSPB',
    routeType: 'icspb',
    workspaceTab: 'icspb',
    defaultMode: 'configure',
    target: '统一语言主干与快速写读分支',
    focus: '围绕语言训练、语义推演、在线写入、记忆回放与稳定性做综合研究。',
    status: '模型路线',
    defaultLayers: ['dynamics'],
    layers3D: ['dynamics', 'boundary'],
    panels: [
      { id: 'overview', label: '路线总览', defaultOpen: true },
      { id: 'control', label: '控制参数', defaultOpen: false },
    ],
    actions: [
      { id: 'switch_icspb', label: '进入ICSPB' },
    ],
    modes: ['configure'],
  },
];

export function getResearchPluginById(pluginId) {
  return RESEARCH_PLUGINS.find((plugin) => plugin.id === pluginId) || RESEARCH_PLUGINS[0];
}

export function makeLayerVisibility(layerIds) {
  return Object.keys(RESEARCH_LAYER_DEFINITIONS).reduce((acc, layerId) => {
    acc[layerId] = layerIds.includes(layerId);
    return acc;
  }, {});
}

export function getPluginWindowState(plugin) {
  return (plugin?.panels || []).reduce((acc, panel) => {
    acc[panel.id] = !!panel.defaultOpen;
    return acc;
  }, {});
}

