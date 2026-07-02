export const RESEARCH_LAYER_DEFINITIONS = {
  network: {
    id: 'network',
    label: '模型结构层',
    detail: 'layer / head / channel / MLP / residual stream',
  },
  atlas: {
    id: 'atlas',
    label: '机制图谱层',
    detail: 'atlas graph nodes / causal edges / evidence links',
  },
  theory: {
    id: 'theory',
    label: '理论连接层',
    detail: '理论假设 / 历史证据 / 反证边界',
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
    focus: '围绕词元、语义角色、注意力头、MLP通道和生成变化，形成可验证机制图谱。',
    status: '主线',
    defaultLayers: ['network', 'atlas'],
    layers3D: ['network', 'atlas', 'causalPath', 'theory', 'boundary', 'aiOrbit'],
    panels: [
      { id: 'overview', label: '路线总览', defaultOpen: true },
      { id: 'evidence', label: '证据详情', defaultOpen: false },
      { id: 'experiment', label: '实验矩阵', defaultOpen: false },
      { id: 'theory', label: '理论记录', defaultOpen: false },
    ],
    actions: [
      { id: 'load_latest_atlas', label: '加载最新图谱' },
      { id: 'run_patch', label: '因果验证' },
      { id: 'open_theory', label: '理论库' },
    ],
    modes: ['configure', 'evidence', 'theory', 'ai_loop'],
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
    defaultLayers: ['network', 'atlas'],
    layers3D: ['network', 'features', 'atlas', 'causalPath', 'boundary'],
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
    defaultLayers: ['network', 'atlas'],
    layers3D: ['network', 'atlas', 'causalPath', 'theory', 'boundary'],
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
    modes: ['configure', 'evidence', 'theory'],
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
    defaultLayers: ['network', 'atlas', 'aiOrbit'],
    layers3D: ['network', 'atlas', 'aiOrbit', 'causalPath', 'theory', 'boundary'],
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
    id: 'theory-audit',
    name: '理论审计',
    shortName: '理论审计',
    routeType: 'theory_system',
    workspaceTab: 'main',
    defaultMode: 'theory',
    target: '把理论变成可反证系统',
    focus: '管理历史研究、最新理论、支持证据、反例、适用边界与下一步验证任务。',
    status: '知识库',
    defaultLayers: ['atlas', 'theory'],
    layers3D: ['atlas', 'theory', 'boundary', 'causalPath'],
    panels: [
      { id: 'overview', label: '路线总览', defaultOpen: true },
      { id: 'theory', label: '理论记录', defaultOpen: false },
      { id: 'boundary', label: '失败边界', defaultOpen: false },
      { id: 'evidence', label: '证据详情', defaultOpen: false },
    ],
    actions: [
      { id: 'open_theory', label: '理论库' },
      { id: 'open_roadmap', label: '路线图' },
    ],
    modes: ['theory', 'evidence', 'configure'],
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
    defaultLayers: ['network'],
    layers3D: ['network', 'dynamics', 'theory', 'boundary'],
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
    defaultLayers: ['network'],
    layers3D: ['network', 'dynamics', 'theory', 'boundary'],
    panels: [
      { id: 'overview', label: '路线总览', defaultOpen: true },
      { id: 'control', label: '控制参数', defaultOpen: false },
      { id: 'theory', label: '理论记录', defaultOpen: false },
    ],
    actions: [
      { id: 'switch_icspb', label: '进入ICSPB' },
    ],
    modes: ['configure', 'theory'],
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

export function getPluginLayerItems(plugin) {
  return (plugin?.layers3D || []).map((layerId) => RESEARCH_LAYER_DEFINITIONS[layerId]).filter(Boolean);
}

export function getPluginWindowState(plugin) {
  return (plugin?.panels || []).reduce((acc, panel) => {
    acc[panel.id] = !!panel.defaultOpen;
    return acc;
  }, {});
}
