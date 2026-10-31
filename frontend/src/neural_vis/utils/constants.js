/**
 * 常量定义 — v3.0 多维度视角 + DNN层可视化 + 动画系统
 */

// ==================== 布局常量 ====================
export const LAYER_GAP = 3.5;
export const PLANE_SIZE = 18;
export const SPHERE_BASE_SIZE = 0.2;
export const TRAJECTORY_LINE_WIDTH = 3;

// ==================== 类别颜色方案 ====================
export const CATEGORY_COLORS = {
  fruit: '#ff6b6b', animal: '#4ecdc4', vehicle: '#ffe66d', tool: '#a855f7',
  nature: '#34d399', food: '#f97316', person: '#ec4899', abstract: '#6366f1',
};

// 层功能颜色
export const LAYER_FUNC_COLORS = {
  lexical: '#ff6b6b', semantic: '#4ecdc4', syntactic: '#ffe66d', decision: '#a855f7',
};

// 子空间颜色
export const SUBSPACE_COLORS = {
  w_u: '#4ecdc4',         // W_U可见 - 青绿
  w_u_perp: '#ff6b6b',    // W_U⊥ - 红色
  grammar: '#ffe66d',      // 语法 - 黄色
  semantic: '#4ecdc4',     // 语义 - 青绿
  logic: '#a855f7',        // 逻辑 - 紫色
  dark_matter: '#f97316',  // 暗物质 - 橙色
};

// 语法角色颜色
export const GRAMMAR_ROLE_COLORS = {
  nsubj: '#ff6b6b', dobj: '#4ecdc4', amod: '#ffe66d', aux: '#a855f7',
  iobj: '#34d399', ccomp: '#f97316', xcomp: '#ec4899', mark: '#6366f1',
};

// 因果链颜色
export const CAUSAL_COLORS = {
  intervention: '#ff6b6b',  // 干预点 - 红
  propagation: '#4ecdc4',   // 传播 - 青
  decay: '#64748b',         // 衰减 - 灰
  flip: '#ffe66d',          // 翻转 - 黄
};

// ==================== 维度视角定义 (v3.0新增) ====================

/**
 * 维度视角系统: 5个维度 × 每维度3个角度 = 15种观察视角
 * 对应拼图8大类(KN/LG/GR/MG/SE/WE/TD/UN) + 4层理论框架(几何/代数/动力学/信息论)
 */
export const DIMENSION_VIEWS = {
  // ---- 维度1: 语义维度 ----
  semantic: {
    key: 'semantic',
    label: '语义 Semantic',
    icon: '🧠',
    color: '#4ecdc4',
    description: '概念编码 / 属性绑定 / 流形结构',
    subViews: {
      concept_flow: {
        key: 'concept_flow',
        label: '概念流形',
        icon: '🔵',
        description: '变形单纯形 + 7维属性子空间',
        renderers: ['trajectory', 'point_cloud', 'force_line'],
        puzzleCells: ['KN-1a', 'KN-2a', 'KN-3a'],
      },
      attribute_subspace: {
        key: 'attribute_subspace',
        label: '属性子空间',
        icon: '🧬',
        description: '7维正交子空间 + W_U解码器',
        renderers: ['subspace', 'heatmap'],
        puzzleCells: ['KN-2a', 'KN-2b', 'KN-2d'],
      },
      dark_matter: {
        key: 'dark_matter',
        label: '暗物质转导',
        icon: '🌑',
        description: '86-92%范数 + 非线性转导',
        renderers: ['dark_matter', 'subspace'],
        puzzleCells: ['KN-4a', 'SE-3a'],
      },
    },
  },

  // ---- 维度2: 语法维度 ----
  syntax: {
    key: 'syntax',
    label: '语法 Syntax',
    icon: '📝',
    color: '#ffe66d',
    description: 'W_U⊥编码 / 正交补空间 / 角色操控',
    subViews: {
      grammar_subspace: {
        key: 'grammar_subspace',
        label: 'W_U⊥子空间',
        icon: '📐',
        description: '语法角色在W_U⊥编码',
        renderers: ['grammar', 'subspace'],
        puzzleCells: ['GR-1a', 'GR-2a'],
      },
      grammar_control: {
        key: 'grammar_control',
        label: '语法操控',
        icon: '🎮',
        description: 'W_U⊥操控 + 翻转50-68%',
        renderers: ['causal', 'flow'],
        puzzleCells: ['GR-3a', 'GR-3b'],
      },
      grammar_capacity: {
        key: 'grammar_capacity',
        label: '容量与泄露',
        icon: '💧',
        description: 'LayerNorm泄露 + W_U⊥容量极限',
        renderers: ['heatmap', 'subspace'],
        puzzleCells: ['GR-1b', 'SE-2a'],
      },
    },
  },

  // ---- 维度3: 逻辑维度 ----
  logic: {
    key: 'logic',
    label: '逻辑 Logic',
    icon: '⚡',
    color: '#a855f7',
    description: '推理载体 / 组合代数 / 因果路径',
    subViews: {
      logic_signal: {
        key: 'logic_signal',
        label: '逻辑信号',
        icon: '📡',
        description: '2.7x属性 + 正交编码 + L18峰值',
        renderers: ['trajectory', 'heatmap'],
        puzzleCells: ['LG-1a', 'LG-1b'],
      },
      logic_algebra: {
        key: 'logic_algebra',
        label: '组合代数',
        icon: '🔢',
        description: 'not=正交扰动 / 对合性 / 德摩根律',
        renderers: ['causal', 'point_cloud'],
        puzzleCells: ['LG-2a', 'LG-2b'],
      },
      logic_causal: {
        key: 'logic_causal',
        label: '因果推理',
        icon: '🔗',
        description: '传递性崩塌 + 恢复率<4%',
        renderers: ['causal', 'dark_matter'],
        puzzleCells: ['LG-3a', 'LG-3b'],
      },
    },
  },

  // ---- 维度4: 计算维度 ----
  computation: {
    key: 'computation',
    label: '计算 Computation',
    icon: '⚙️',
    color: '#f97316',
    description: 'FFN增益 / Attention主通道 / 层功能分工',
    subViews: {
      ffn_mechanism: {
        key: 'ffn_mechanism',
        label: 'FFN机制',
        icon: '🔧',
        description: '增益调制×内容方向 / 方向反射',
        renderers: ['heatmap', 'trajectory'],
        puzzleCells: ['WE-1a', 'WE-2a'],
      },
      attention_channel: {
        key: 'attention_channel',
        label: 'Attention通道',
        icon: '👁️',
        description: '语义→Logit主通道(21.5x) / 信息流',
        renderers: ['flow', 'trajectory'],
        puzzleCells: ['TD-1a', 'TD-2a'],
      },
      layer_dynamics: {
        key: 'layer_dynamics',
        label: '层间动力学',
        icon: '📈',
        description: '力线指数增长 / 断裂层 / 深层锁定',
        renderers: ['force_line', 'trajectory'],
        puzzleCells: ['TD-3a', 'TD-4a'],
      },
    },
  },

  // ---- 维度5: 理论维度 ----
  theory: {
    key: 'theory',
    label: '理论 Theory',
    icon: '📐',
    color: '#ec4899',
    description: '4层框架 / 候选定律 / 不变量',
    subViews: {
      geometry: {
        key: 'geometry',
        label: '几何层',
        icon: '🔷',
        description: '子空间分解 / 流形结构 (30%填充)',
        renderers: ['subspace', 'point_cloud', 'force_line'],
        puzzleCells: ['KN-3a', 'GR-1a'],
      },
      algebra: {
        key: 'algebra',
        label: '代数层',
        icon: '➕',
        description: '组合律 / 变换群 (10%空白!)',
        renderers: ['causal', 'grammar'],
        puzzleCells: ['LG-2a', 'UN-1'],
      },
      dynamics: {
        key: 'dynamics',
        label: '动力学层',
        icon: '🌊',
        description: '力线增长 / Attn主通道 (40%填充)',
        renderers: ['force_line', 'flow', 'dark_matter'],
        puzzleCells: ['TD-3a', 'KN-4a'],
      },
      information: {
        key: 'information',
        label: '信息论层',
        icon: '📊',
        description: '暗物质转导 / 信息隐藏 (15%空白!)',
        renderers: ['dark_matter', 'subspace'],
        puzzleCells: ['KN-4a', 'SE-3a'],
      },
    },
  },
};

// ==================== DNN层可视化常量 (v3.0新增) ====================

/**
 * Transformer层功能分区
 * 对应R3/R4/R8/INV-26等规律的层功能发现
 */
export const LAYER_FUNCTIONS = {
  embedding: {
    label: '嵌入层',
    color: '#64748b',
    range: [0, 0],
    description: '词嵌入近正交(cos≈0.004)',
  },
  lexical: {
    label: '词法加工',
    color: '#ff6b6b',
    range: [1, 5],
    description: '词频统计 + 语法压缩起始',
  },
  syntax_processing: {
    label: '语法加工',
    color: '#ffe66d',
    range: [6, 12],
    description: 'W_U⊥语法编码 + template热点',
  },
  semantic_extraction: {
    label: '语义提取',
    color: '#4ecdc4',
    range: [13, 22],
    description: '力线指数增长 + 概念流形分化',
  },
  logic_injection: {
    label: '逻辑注入',
    color: '#a855f7',
    range: [18, 24],
    description: '逻辑信号峰值(L18) + 推理加工',
  },
  decision: {
    label: '输出决策',
    color: '#f97316',
    range: [25, 35],
    description: '深层锁定 + W_U对齐 + Logit映射',
  },
};

/**
 * Transformer组件类型 (每个Layer内的子结构)
 */
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

// ==================== 动画场景定义 (v3.0新增) ====================

/**
 * 预设动画场景: 展示DNN工作原理
 */
export const ANIMATION_SCENARIOS = {
  forward_pass: {
    key: 'forward_pass',
    label: '前向传播',
    icon: '➡️',
    description: 'Token从L0→L35逐层传播的完整过程',
    duration: 15, // 秒
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

// ==================== 颜色映射函数 ====================

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

/**
 * 层号 → 层功能颜色
 */
export function layerToFuncColor(layer, nLayers = 36) {
  const ratio = layer / (nLayers - 1);
  if (ratio <= 0.14) return LAYER_FUNCTIONS.lexical.color;
  if (ratio <= 0.33) return LAYER_FUNCTIONS.syntax_processing.color;
  if (ratio <= 0.61) return LAYER_FUNCTIONS.semantic_extraction.color;
  if (ratio <= 0.69) return LAYER_FUNCTIONS.logic_injection.color;
  return LAYER_FUNCTIONS.decision.color;
}

/**
 * 层号 → 层功能标签
 */
export function layerToFuncLabel(layer, nLayers = 36) {
  const ratio = layer / (nLayers - 1);
  if (layer === 0) return LAYER_FUNCTIONS.embedding.label;
  if (ratio <= 0.14) return LAYER_FUNCTIONS.lexical.label;
  if (ratio <= 0.33) return LAYER_FUNCTIONS.syntax_processing.label;
  if (ratio <= 0.61) return LAYER_FUNCTIONS.semantic_extraction.label;
  if (ratio <= 0.69) return LAYER_FUNCTIONS.logic_injection.label;
  return LAYER_FUNCTIONS.decision.label;
}
