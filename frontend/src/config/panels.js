/**
 * 面板配置文件
 * 统一管理所有面板的位置、样式、标签页分组和数据模板
 */

// 面板位置配置
export const PANEL_POSITIONS = {
  inputPanel: {
    position: 'absolute',
    top: 60,
    left: 20,
    zIndex: 10,
    width: '360px',
    maxHeight: '85vh',
  },
  infoPanel: {
    position: 'absolute',
    top: 20,
    right: 20,
    zIndex: 100,
    minWidth: '320px',
    maxWidth: '400px',
    maxHeight: '80vh',
  },
  operationPanel: {
    position: 'absolute',
    bottom: 20,
    right: 20,
    zIndex: 10,
    minWidth: '320px',
    maxWidth: '400px',
    maxHeight: '60vh',
  },
  detailPanel: {
    position: 'absolute',
    bottom: 20,
    left: 20,
    zIndex: 10,
    width: '380px',
    maxHeight: '50vh',
  },
};

// 面板基础样式
export const PANEL_BASE_STYLE = {
  background: 'rgba(20, 20, 25, 0.95)',
  padding: '16px',
  borderRadius: '12px',
  backdropFilter: 'blur(10px)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
  display: 'flex',
  flexDirection: 'column',
  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
};

// 结构分析标签分组（二维菜单）
export const STRUCTURE_TABS_V2 = {
  groups: [
    {
      id: 'observation',
      label: '观测',
      icon: 'Eye',
      color: '#00d2ff',
      description: '层间预测演化与激活可视化',
      items: [
        { id: 'logit_lens', label: '预测演化 (Logit)', desc: '层间预测演化', icon: 'BarChart2' },
      ],
    },
    {
      id: 'analysis',
      label: '分析',
      icon: 'Zap',
      color: '#ff9f43',
      description: '因果回路与特征提取',
      items: [
        { id: 'circuit', label: '回路 (Circuit)', desc: '因果回路发现', icon: 'Share2' },
        { id: 'features', label: '特征 (Features)', desc: 'SAE 稀疏特征', icon: 'Sparkles' },
        { id: 'causal', label: '因果 (Causal)', desc: '因果中介分析', icon: 'Target' },
        { id: 'manifold', label: '流形 (Manifold)', desc: '流形几何分析', icon: 'Globe2' },
        { id: 'compositional', label: '组合 (Compos)', desc: '组合泛化', icon: 'Layers' },
      ],
    },
    {
      id: 'geometry',
      label: '几何',
      icon: 'Hexagon',
      color: '#6c5ce7',
      description: '几何与拓扑分析',
      items: [
        { id: 'rpt', label: '传输 (RPT)', desc: '黎曼平行传输', icon: 'ArrowRightLeft' },
        { id: 'curvature', label: '曲率 (Curv)', desc: '曲率场分析', icon: 'TrendingUp' },
        { id: 'tda', label: '拓扑 (TDA)', desc: '拓扑数据分析', icon: 'BarChart' },
      ],
    },
    {
      id: 'advanced',
      label: '高级',
      icon: 'FlaskConical',
      color: '#ff6b9d',
      description: '可靠性与训练行为分析',
      items: [
        { id: 'validity', label: '有效性 (Valid)', desc: '有效性检验', icon: 'CheckCircle' },
        { id: 'training', label: '训练 (Training)', desc: '训练动力学', icon: 'Activity' },
      ],
    },
    {
      id: 'encoding3d',
      label: '编码 3D',
      icon: 'Layers',
      color: '#22c55e',
      description: '共享承载、偏置偏转与逐层放大的五层 3D 场景',
      items: [
        { id: 'shared_carrier_3d', label: '承载层', desc: '共享承载层 3D 场景', icon: 'Layers' },
        { id: 'bias_deflection_3d', label: '偏转层', desc: '偏置偏转层 3D 场景', icon: 'GitBranch' },
        { id: 'layerwise_amplification_3d', label: '放大层', desc: '逐层放大层 3D 场景', icon: 'TrendingUp' },
        { id: 'multispace_operator_3d', label: '角色层', desc: '多空间角色与局部运算元', icon: 'Grid3x3' },
        { id: 'cross_model_compare_3d', label: '对照层', desc: '跨模型结构对照层', icon: 'ArrowRightLeft' },
      ],
    },
  ],
};

// 扁平化标签（向后兼容）
export const STRUCTURE_TABS = {
  observation: STRUCTURE_TABS_V2.groups.find((g) => g.id === 'observation').items,
  analysis: STRUCTURE_TABS_V2.groups.find((g) => g.id === 'analysis').items,
  geometry: STRUCTURE_TABS_V2.groups.find((g) => g.id === 'geometry').items,
  advanced: STRUCTURE_TABS_V2.groups.find((g) => g.id === 'advanced').items,
  encoding3d: STRUCTURE_TABS_V2.groups.find((g) => g.id === 'encoding3d').items,
};

// 输入面板标签
export const INPUT_PANEL_TABS = [
  { id: 'main', label: 'DNN', color: '#38bdf8', description: 'DNN 主工作台' },
  { id: 'snn', label: 'SNN', color: '#4ecdc4', description: '脉冲神经网络' },
  { id: 'icspb', label: 'ICSPB', color: '#6c5ce7', description: '当前模型工作台' },
];

// 数据展示模板
export const ANALYSIS_DATA_TEMPLATES = {
  logit_lens: {
    title: 'Logit Lens 分析',
    color: '#00d2ff',
    metrics: [
      { key: 'avg_confidence', label: '平均置信度', format: 'percent' },
      { key: 'entropy', label: '熵值', format: 'decimal' },
    ],
    sections: [
      { type: 'layer_list', title: '层间预测', source: 'logit_lens' },
    ],
  },
  circuit: {
    title: '因果回路发现',
    color: '#ff6b6b',
    metrics: [
      { key: 'nodes', label: '节点数', format: 'number' },
      { key: 'edges', label: '边数', format: 'number' },
      { key: 'density', label: '密度', format: 'percent' },
    ],
    sections: [
      { type: 'graph_summary', title: '图结构', source: 'graph' },
    ],
  },
  features: {
    title: 'SAE 特征提取',
    color: '#ffd93d',
    metrics: [
      { key: 'n_features', label: '特征数', format: 'number' },
      { key: 'sparsity', label: '稀疏度', format: 'decimal' },
      { key: 'reconstruction_error', label: '重构误差', format: 'decimal' },
    ],
    sections: [
      { type: 'feature_table', title: 'Top 特征', source: 'top_features' },
    ],
  },
  causal: {
    title: '因果中介分析',
    color: '#6c5ce7',
    metrics: [
      { key: 'n_components_analyzed', label: '分析组件', format: 'number' },
      { key: 'n_important_components', label: '关键组件', format: 'number' },
    ],
    sections: [],
  },
  manifold: {
    title: '流形几何分析',
    color: '#4ecdc4',
    metrics: [
      { key: 'intrinsic_dim', label: '内在维度', format: 'number' },
      { key: 'curvature', label: '曲率', format: 'decimal' },
    ],
    sections: [],
  },
  icspb: {
    title: 'ICSPB 当前模型',
    color: '#6c5ce7',
    metrics: [
      { key: 'stable_read', label: '稳定读取', format: 'decimal' },
      { key: 'guarded_write', label: '受控写入', format: 'decimal' },
    ],
    sections: [],
  },
  tda: {
    title: '拓扑数据分析',
    color: '#e056fd',
    metrics: [
      { key: 'betti_0', label: 'Betti-0 连通分量', format: 'number' },
      { key: 'betti_1', label: 'Betti-1 环', format: 'number' },
      { key: 'betti_2', label: 'Betti-2 空腔', format: 'number' },
    ],
    sections: [],
  },
  rpt: {
    title: '黎曼平行传输',
    color: '#00d2ff',
    metrics: [
      { key: 'transport_distance', label: '传输距离', format: 'decimal' },
      { key: 'alignment', label: '对齐度', format: 'percent' },
    ],
    sections: [],
  },
  curvature: {
    title: '曲率场分析',
    color: '#ff9f43',
    metrics: [
      { key: 'scalar_curvature', label: '标量曲率', format: 'decimal' },
      { key: 'ricci_curvature', label: 'Ricci 曲率', format: 'decimal' },
    ],
    sections: [],
  },
};

// 颜色主题
export const COLORS = {
  primary: '#00d2ff',
  secondary: '#3a7bd5',
  accent: '#4ecdc4',
  warning: '#ff9f43',
  danger: '#ff4444',
  success: '#5ec962',
  purple: '#6c5ce7',
  pink: '#ff6b9d',
  bgDark: 'rgba(20, 20, 25, 0.95)',
  bgLight: 'rgba(255, 255, 255, 0.03)',
  bgBorder: 'rgba(255, 255, 255, 0.1)',
  textPrimary: '#ffffff',
  textSecondary: '#aaaaaa',
  textMuted: '#666666',
};

// 操作历史配置
export const HISTORY_CONFIG = {
  maxItems: 50,
  storageKey: 'transformerlens_history',
};

// 默认面板可见性
export const DEFAULT_PANEL_VISIBILITY = {
  inputPanel: true,
  infoPanel: true,
  operationPanel: true,
  detailPanel: false,
};

// ==================== 维度视角系统 (v3.0, 参考 neural-vis) ====================

/**
 * 5大维度视角 × 3个子视角 = 15种观察角度
 * 对应拼图8大类(KN/LG/GR/MG/SE/WE/TD/UN) + 4层理论框架
 */
export const DIMENSION_VIEWS = {
  semantic: {
    key: 'semantic',
    label: '语义 Semantic',
    icon: '🧠',
    color: '#4ecdc4',
    description: '知识网络层：概念编码 / 属性绑定 / 抽象层级 / 知识拓扑',
    subViews: {
      concept_encoding: {
        key: 'concept_encoding',
        label: '概念编码',
        icon: '🔵',
        description: 'eff_dim=17-22 / 超分散编码 / 同类cos>0异类<0',
        puzzleCells: ['KN-1a', 'KN-1b', 'KN-1c', 'KN-1d'],
        structureTabs: ['manifold', 'logit_lens', 'features'],
        renderers: ['trajectory', 'point_cloud', 'force_line'],
      },
      attribute_binding: {
        key: 'attribute_binding',
        label: '属性绑定',
        icon: '🧬',
        description: '7维正交子空间 / 亚加性压缩 / 交互高秩',
        puzzleCells: ['KN-2a', 'KN-2b', 'KN-2c', 'KN-2d'],
        structureTabs: ['features', 'manifold', 'causal'],
        renderers: ['subspace', 'heatmap'],
      },
      abstraction_chain: {
        key: 'abstraction_chain',
        label: '抽象链',
        icon: '🔺',
        description: '非嵌套非正交 / 层级距离uniform(0.3-0.5)',
        puzzleCells: ['KN-3a', 'KN-3b', 'KN-3c', 'KN-3d'],
        structureTabs: ['manifold', 'features'],
        renderers: ['point_cloud', 'trajectory'],
      },
      knowledge_topology: {
        key: 'knowledge_topology',
        label: '知识拓扑',
        icon: '🌐',
        description: '关系编码 / 图结构映射 / 检索与修改机制',
        puzzleCells: ['KN-4a', 'KN-4b', 'KN-4c', 'KN-4d'],
        structureTabs: ['circuit', 'features', 'causal'],
        renderers: ['causal', 'flow'],
      },
    },
  },
  syntax: {
    key: 'syntax',
    label: '语法 Syntax',
    icon: '📝',
    color: '#ffe66d',
    description: '语法体系层：词性编码 / 句式模板 / 层次结构',
    subViews: {
      pos_encoding: {
        key: 'pos_encoding',
        label: '词性编码',
        icon: '📐',
        description: '名/动/形/介/副在W_U⊥编码 / 高维因果载体',
        puzzleCells: ['GR-1a', 'GR-1b', 'GR-1c', 'GR-1d'],
        structureTabs: ['features', 'manifold', 'causal'],
        renderers: ['grammar', 'subspace'],
      },
      sentence_template: {
        key: 'sentence_template',
        label: '句式模板',
        icon: '📋',
        description: '否定dim=1 / 语法因果逐层增长 / 组合亚加性',
        puzzleCells: ['GR-2a', 'GR-2b', 'GR-2c', 'GR-2d'],
        structureTabs: ['logit_lens', 'causal', 'features'],
        renderers: ['causal', 'flow'],
      },
      hierarchy: {
        key: 'hierarchy',
        label: '层次结构',
        icon: '🏗️',
        description: '词→短语→句子 / 递归vs扁平 / 长距离依赖',
        puzzleCells: ['GR-3a', 'GR-3b', 'GR-3c', 'GR-3d'],
        structureTabs: ['circuit', 'features', 'rpt'],
        renderers: ['heatmap', 'subspace'],
      },
    },
  },
  logic: {
    key: 'logic',
    label: '逻辑 Logic',
    icon: '⚡',
    color: '#a855f7',
    description: '逻辑推理层：条件推理 / 深度思考 / 翻译转换 / 逻辑-知识交互',
    subViews: {
      conditional: {
        key: 'conditional',
        label: '条件推理',
        icon: '📡',
        description: 'A→B因果路径 / L0信号0.70逐步稀释',
        puzzleCells: ['LG-1a', 'LG-1b', 'LG-1c', 'LG-1d'],
        structureTabs: ['causal', 'logit_lens', 'circuit'],
        renderers: ['trajectory', 'heatmap'],
      },
      deep_thinking: {
        key: 'deep_thinking',
        label: '深度思考',
        icon: '🔢',
        description: '多步推理 / 串行vs并行 / 前沿模式',
        puzzleCells: ['LG-2a', 'LG-2b', 'LG-2c', 'LG-2d'],
        structureTabs: ['causal', 'features'],
        renderers: ['causal', 'point_cloud'],
      },
      translation: {
        key: 'translation',
        label: '翻译转换',
        icon: '🌐',
        description: '跨语言编码一致 / 否定cos>0.84 / 时态不一致',
        puzzleCells: ['LG-3a', 'LG-3b', 'LG-3c', 'LG-3d'],
        structureTabs: ['features', 'manifold', 'logit_lens'],
        renderers: ['subspace', 'trajectory'],
      },
      logic_knowledge: {
        key: 'logic_knowledge',
        label: '逻辑-知识交互',
        icon: '🔗',
        description: '知识调用路径 / 交互高秩 / 因果泄漏≈1.0',
        puzzleCells: ['LG-4a', 'LG-4b', 'LG-4c'],
        structureTabs: ['causal', 'circuit', 'features'],
        renderers: ['causal', 'dark_matter'],
      },
    },
  },
  computation: {
    key: 'computation',
    label: '计算 Computation',
    icon: '⚙️',
    color: '#f97316',
    description: '生成控制 / 全局选择 / 系统效率 / 词嵌入数学',
    subViews: {
      generation_control: {
        key: 'generation_control',
        label: '生成控制',
        icon: '🎨',
        description: '风格分化 / 语法约束 / 逻辑连贯 / 多维同时控制',
        puzzleCells: ['MG-1a', 'MG-2a', 'MG-3a', 'MG-3c'],
        structureTabs: ['features', 'logit_lens', 'causal'],
        renderers: ['heatmap', 'trajectory'],
      },
      global_selection: {
        key: 'global_selection',
        label: '全局选择',
        icon: '🎯',
        description: 'W_U映射 / logit唯一性 / SA<1.0约束 / 4000+词选一',
        puzzleCells: ['MG-4a', 'MG-4b', 'MG-4c', 'MG-4d', 'MG-4e'],
        structureTabs: ['logit_lens', 'features', 'circuit'],
        renderers: ['flow', 'trajectory'],
      },
      system_efficiency: {
        key: 'system_efficiency',
        label: '系统效率',
        icon: '⚡',
        description: '分布式编码 / 亚加性压缩 / ICL / 因果泄漏',
        puzzleCells: ['SE-1a', 'SE-1b', 'SE-1c', 'SE-2a', 'SE-3a'],
        structureTabs: ['features', 'causal', 'circuit'],
        renderers: ['heatmap', 'causal'],
      },
      word_math: {
        key: 'word_math',
        label: '词嵌入数学',
        icon: '🔢',
        description: '否定dim=1线性 / 时态dim=15非线性 / 因果效力弱',
        puzzleCells: ['WE-1a', 'WE-1b', 'WE-1c', 'WE-1d'],
        structureTabs: ['features', 'manifold', 'logit_lens'],
        renderers: ['point_cloud', 'subspace'],
      },
    },
  },
  theory: {
    key: 'theory',
    label: '理论 Theory',
    icon: '📐',
    color: '#ec4899',
    description: '4层框架：几何 / 代数 / 动力学 / 信息论',
    subViews: {
      geometry: {
        key: 'geometry',
        label: '几何层',
        icon: '🔷',
        description: '子空间分解 / 流形结构 / 变形单纯形 (30%填充)',
        puzzleCells: ['KN-3a', 'GR-1a', 'KN-1a'],
        structureTabs: ['manifold', 'features', 'rpt'],
        renderers: ['subspace', 'point_cloud', 'force_line'],
      },
      algebra: {
        key: 'algebra',
        label: '代数层',
        icon: '➕',
        description: '组合律 / 变换群 / 交互高秩 (10%空白!)',
        puzzleCells: ['LG-2a', 'UN-1', 'KN-2b'],
        structureTabs: ['causal', 'features'],
        renderers: ['causal', 'grammar'],
      },
      dynamics: {
        key: 'dynamics',
        label: '动力学层',
        icon: '🌊',
        description: '力线指数增长 / Attn主通道 / 残差旋转 (40%填充)',
        puzzleCells: ['TD-3a', 'KN-4a', 'LG-1b'],
        structureTabs: ['rpt', 'curvature', 'circuit'],
        renderers: ['force_line', 'flow', 'dark_matter'],
      },
      information: {
        key: 'information',
        label: '信息论层',
        icon: '📊',
        description: '暗物质转导 / 因果泄漏 / 范数增长100-300x (15%空白!)',
        puzzleCells: ['KN-4a', 'SE-3a', 'SE-3b'],
        structureTabs: ['features', 'manifold', 'causal'],
        renderers: ['dark_matter', 'subspace'],
      },
    },
  },
};

// ==================== 动画场景定义 ====================

export const ANIMATION_SCENARIOS = {
  forward_pass: {
    key: 'forward_pass',
    label: '前向传播',
    icon: '➡️',
    description: 'Token从L0→L35逐层传播的完整过程',
    duration: 15,
    phases: [
      { label: '嵌入', start: 0, end: 0.05, layerRange: [0, 0] },
      { label: '词法断裂', start: 0.05, end: 0.15, layerRange: [1, 5] },
      { label: '语法压缩', start: 0.15, end: 0.30, layerRange: [6, 12] },
      { label: '语义提取', start: 0.30, end: 0.55, layerRange: [13, 22] },
      { label: '逻辑注入', start: 0.45, end: 0.65, layerRange: [18, 24] },
      { label: '决策锁定', start: 0.65, end: 0.85, layerRange: [25, 32] },
      { label: '输出映射', start: 0.85, end: 1.0, layerRange: [33, 35] },
    ],
  },
  subspace_division: {
    key: 'subspace_division',
    label: '子空间分化',
    icon: '🧬',
    description: '语义→W_U / 语法→W_U⊥ / 逻辑→独立子空间',
    duration: 12,
    phases: [
      { label: '初始混合', start: 0, end: 0.15, layerRange: [0, 3] },
      { label: '语法分离', start: 0.15, end: 0.35, layerRange: [4, 10] },
      { label: '语义对齐W_U', start: 0.35, end: 0.60, layerRange: [11, 22] },
      { label: '暗物质形成', start: 0.45, end: 0.70, layerRange: [10, 25] },
      { label: '逻辑独立', start: 0.50, end: 0.75, layerRange: [14, 24] },
      { label: '锁定分化', start: 0.75, end: 1.0, layerRange: [25, 35] },
    ],
  },
  force_line_growth: {
    key: 'force_line_growth',
    label: '语义力线增长',
    icon: '⚡',
    description: '语义信号100-300倍指数增长过程',
    duration: 10,
    phases: [
      { label: '弱信号(L0)', start: 0, end: 0.1, layerRange: [0, 3] },
      { label: '缓慢增长', start: 0.1, end: 0.3, layerRange: [4, 10] },
      { label: '指数加速', start: 0.3, end: 0.6, layerRange: [11, 20] },
      { label: '100x增益', start: 0.6, end: 0.8, layerRange: [21, 30] },
      { label: 'W_U对齐', start: 0.8, end: 1.0, layerRange: [31, 35] },
    ],
  },
  dark_matter_transduction: {
    key: 'dark_matter_transduction',
    label: '暗物质转导',
    icon: '🌑',
    description: 'W_U⊥信号如何"绕过"W_U到达logits',
    duration: 12,
    phases: [
      { label: 'W_U⊥编码', start: 0, end: 0.2, layerRange: [0, 8] },
      { label: '残差直通', start: 0.2, end: 0.4, layerRange: [9, 18] },
      { label: '非线性转导', start: 0.4, end: 0.7, layerRange: [19, 28] },
      { label: '级联衰减', start: 0.55, end: 0.75, layerRange: [15, 28] },
      { label: 'W_U解码', start: 0.75, end: 1.0, layerRange: [29, 35] },
    ],
  },
  attribute_encoding: {
    key: 'attribute_encoding',
    label: '属性编码解码',
    icon: '🧬',
    description: '7维属性子空间 + W_U解码器(SNR×5-9)',
    duration: 12,
    phases: [
      { label: '属性提取', start: 0, end: 0.2, layerRange: [0, 8] },
      { label: '7维子空间形成', start: 0.2, end: 0.4, layerRange: [9, 16] },
      { label: '非线性耦合', start: 0.4, end: 0.6, layerRange: [17, 24] },
      { label: 'W_U投影', start: 0.6, end: 0.8, layerRange: [25, 32] },
      { label: 'SNR放大', start: 0.8, end: 1.0, layerRange: [33, 35] },
    ],
  },
};

// Transformer组件类型 (每个Layer内的子结构)
export const COMPONENT_TYPES = {
  residual: {
    label: '残差连接',
    color: '#60a5fa',
    opacity: 0.3,
    description: '锚定(β≈1.0) + 62-71%保留',
  },
  attention: {
    label: '注意力',
    color: '#4ecdc4',
    opacity: 0.7,
    description: '语义→Logit主通道(21.5x)',
  },
  ffn: {
    label: 'FFN',
    color: '#f97316',
    opacity: 0.7,
    description: '增益调制×内容方向 / 方向反射(-0.5)',
  },
  layer_norm: {
    label: 'LayerNorm',
    color: '#ffe66d',
    opacity: 0.5,
    description: '可能泄露W_U⊥信息到logits',
  },
};

// 层功能分区
export const LAYER_FUNCTIONS = {
  embedding: { label: '嵌入层', color: '#64748b', range: [0, 0] },
  lexical: { label: '词法加工', color: '#ff6b6b', range: [1, 5] },
  syntax_processing: { label: '语法加工', color: '#ffe66d', range: [6, 12] },
  semantic_extraction: { label: '语义提取', color: '#4ecdc4', range: [13, 22] },
  logic_injection: { label: '逻辑注入', color: '#a855f7', range: [18, 24] },
  decision: { label: '输出决策', color: '#f97316', range: [25, 35] },
};

export function layerToFuncColor(layer, nLayers = 36) {
  const ratio = layer / (nLayers - 1);
  if (ratio <= 0.14) return LAYER_FUNCTIONS.lexical.color;
  if (ratio <= 0.33) return LAYER_FUNCTIONS.syntax_processing.color;
  if (ratio <= 0.61) return LAYER_FUNCTIONS.semantic_extraction.color;
  if (ratio <= 0.69) return LAYER_FUNCTIONS.logic_injection.color;
  return LAYER_FUNCTIONS.decision.color;
}

export function layerToFuncLabel(layer, nLayers = 36) {
  const ratio = layer / (nLayers - 1);
  if (layer === 0) return LAYER_FUNCTIONS.embedding.label;
  if (ratio <= 0.14) return LAYER_FUNCTIONS.lexical.label;
  if (ratio <= 0.33) return LAYER_FUNCTIONS.syntax_processing.label;
  if (ratio <= 0.61) return LAYER_FUNCTIONS.semantic_extraction.label;
  if (ratio <= 0.69) return LAYER_FUNCTIONS.logic_injection.label;
  return LAYER_FUNCTIONS.decision.label;
}

// ==================== 渲染器模式定义 ====================

export const RENDERER_MODES = [
  { key: 'all', label: '🔍 全部', icon: '🔍' },
  { key: 'trajectory', label: '📈 轨迹', icon: '📈' },
  { key: 'point_cloud', label: '⚪ 点云', icon: '⚪' },
  { key: 'heatmap', label: '📊 热力图', icon: '📊' },
  { key: 'flow', label: '🔀 信息流', icon: '🔀' },
  { key: 'subspace', label: '🧬 子空间', icon: '🧬' },
  { key: 'force_line', label: '⚡ 力线', icon: '⚡' },
  { key: 'grammar', label: '📝 语法矩阵', icon: '📝' },
  { key: 'causal', label: '🔗 因果链', icon: '🔗' },
  { key: 'dark_matter', label: '🌑 暗物质', icon: '🌑' },
];

// ==================== 颜色系统 ====================

export const CATEGORY_COLORS = {
  fruit: '#ff6b6b', animal: '#4ecdc4', vehicle: '#ffe66d', tool: '#a855f7',
  nature: '#34d399', food: '#f97316', person: '#ec4899', abstract: '#6366f1',
};

export const SUBSPACE_COLORS = {
  w_u: '#4ecdc4',
  w_u_perp: '#ff6b6b',
  grammar: '#ffe66d',
  semantic: '#4ecdc4',
  logic: '#a855f7',
  dark_matter: '#f97316',
};

export const GRAMMAR_ROLE_COLORS = {
  nsubj: '#ff6b6b', dobj: '#4ecdc4', amod: '#ffe66d', aux: '#a855f7',
  iobj: '#34d399', ccomp: '#f97316', xcomp: '#ec4899', mark: '#6366f1',
};

export const CAUSAL_COLORS = {
  intervention: '#ff6b6b',
  propagation: '#4ecdc4',
  decay: '#64748b',
  flip: '#ffe66d',
};

// 布局常量
export const LAYER_GAP = 3.5;
export const PLANE_SIZE = 18;
export const SPHERE_BASE_SIZE = 0.2;
export const TRAJECTORY_LINE_WIDTH = 3;

/**
 * delta_cos → 颜色映射: 1.0=红, 0.5=橙, 0.0=蓝
 */
export function deltaCosToColor(deltaCos) {
  const r = Math.max(0, Math.min(1, deltaCos));
  let red, green, blue;
  if (r > 0.5) {
    const t = (r - 0.5) * 2;
    red = Math.round(239 * t + 245 * (1 - t));
    green = Math.round(68 * t + 158 * (1 - t));
    blue = Math.round(68 * t + 11 * (1 - t));
  } else {
    const t = r * 2;
    red = Math.round(245 * t + 59 * (1 - t));
    green = Math.round(158 * t + 130 * (1 - t));
    blue = Math.round(11 * t + 246 * (1 - t));
  }
  return `#${red.toString(16).padStart(2, '0')}${green.toString(16).padStart(2, '0')}${blue.toString(16).padStart(2, '0')}`;
}

/**
 * cos_with_wu → 颜色映射: 高=青绿(W_U对齐), 低=红色(W_U⊥)
 */
export function cosWuToColor(cosWu) {
  const r = Math.max(0, Math.min(1, cosWu));
  let red, green, blue;
  if (r > 0.7) {
    const t = (r - 0.7) / 0.3;
    red = Math.round(78 * (1 - t) + 34 * t);
    green = Math.round(205 * (1 - t) + 211 * t);
    blue = Math.round(196 * (1 - t) + 153 * t);
  } else if (r > 0.4) {
    const t = (r - 0.4) / 0.3;
    red = Math.round(234 * (1 - t) + 78 * t);
    green = Math.round(179 * (1 - t) + 205 * t);
    blue = Math.round(8 * (1 - t) + 196 * t);
  } else {
    const t = r / 0.4;
    red = Math.round(239 * (1 - t) + 234 * t);
    green = Math.round(68 * (1 - t) + 179 * t);
    blue = Math.round(68 * (1 - t) + 8 * t);
  }
  return `#${red.toString(16).padStart(2, '0')}${green.toString(16).padStart(2, '0')}${blue.toString(16).padStart(2, '0')}`;
}

/**
 * 比例值 → 颜色映射 (0=暗, 1=亮)
 */
export function ratioToColor(ratio, baseColor = [78, 205, 196]) {
  const r = Math.max(0, Math.min(1, ratio));
  const intensity = 0.2 + r * 0.8;
  const red = Math.round(baseColor[0] * intensity);
  const green = Math.round(baseColor[1] * intensity);
  const blue = Math.round(baseColor[2] * intensity);
  return `#${red.toString(16).padStart(2, '0')}${green.toString(16).padStart(2, '0')}${blue.toString(16).padStart(2, '0')}`;
}
