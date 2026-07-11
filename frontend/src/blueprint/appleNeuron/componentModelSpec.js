import {
  patternAtlasUnitAddressLabel,
  sortPatternAtlasNodes,
} from '../../researchKernel/patternAtlasEvidence.js';

export const COMPONENT_MODEL_PHASES = {
  input: { component: 'input', shapeId: 'vector_portal' },
  ln1: { component: 'ln', shapeId: 'normalization_tunnel' },
  qkv: { component: 'attention', shapeId: 'multi_head_router' },
  attn_score: { component: 'attention', shapeId: 'multi_head_router' },
  softmax: { component: 'attention', shapeId: 'multi_head_router' },
  attn_out: { component: 'attention', shapeId: 'multi_head_router' },
  residual1: { component: 'residual', shapeId: 'dual_path_merge' },
  ln2: { component: 'ln', shapeId: 'normalization_tunnel' },
  ffn_up: { component: 'ffn', shapeId: 'neuron_expansion_field' },
  ffn_act: { component: 'ffn', shapeId: 'neuron_expansion_field' },
  ffn_down: { component: 'ffn', shapeId: 'neuron_expansion_field' },
  residual2: { component: 'residual', shapeId: 'dual_path_merge' },
};

const COPY = {
  zh: {
    input: {
      title: '层输入向量',
      shape: '形状：隐藏维度束穿过输入门，进入当前 Transformer Layer。',
      mechanism: '作用：承载当前 token 的残差流状态，供归一化、Attention 和 MLP 继续读取与写入。',
      unitType: '对象：隐藏向量维度，不是独立神经元。',
      boundary: '当前阶段没有 H#/N# 单元地址；显示的是结构通道。',
    },
    ln: {
      title: '归一化通道',
      shape: '形状：不等尺度输入束 → 统计与缩放核心 → 稳定尺度输出束。',
      mechanism: '作用：对单个 token 的隐藏向量重标定，稳定后续组件接收到的数值尺度。',
      unitType: '对象：d_model 个隐藏维度以及对应缩放参数，不是独立神经元层。',
      boundary: '归一化目前没有 H#/N# 图谱映射；维度槽只表示结构。',
    },
    attention: {
      title: '多头注意力路由器',
      shape: '形状：Q/K/V 三路投影 → 多头环 → 注意力矩阵 → 输出汇聚。',
      mechanism: '作用：比较当前位置与上文位置，并把被选中的信息沿 Attention head 写回残差流。',
      unitType: '物理单元：Attention head，使用真实 H# 地址。',
      boundary: 'H# 是物理 Head 候选；当前单单元因果闭合仍为 0。',
    },
    ffn: {
      title: 'MLP 神经元扩张场',
      shape: '形状：d_model 窄层 → intermediate_size 宽神经元场 → d_model 回写层。',
      mechanism: '作用：将隐藏状态投影到更宽的特征空间，经门控选择后再写回残差流。',
      unitType: '物理单元：N# 表示 SiLU(gate)⊙up 的 MLP product neuron，G# 表示 MLP 组级对象。',
      boundary: 'N#/G# 是图谱候选；只有单单元干预通过后才能称为有效神经元。',
    },
    residual: {
      title: '残差双路径合流器',
      shape: '形状：原始状态直通路径 + 子模块写入路径，在加法节点合流。',
      mechanism: '作用：保留原状态并叠加组件计算结果，使信息能够跨层连续运输。',
      unitType: '对象：残差向量和加法节点，不是独立神经元。',
      boundary: '当前显示向量流与范数，没有单神经元图谱地址。',
    },
    phases: {
      input: '读取当前层输入状态',
      ln1: 'Attention 前归一化',
      ln2: 'MLP 前归一化',
      qkv: '生成 Query、Key、Value 三路表示',
      attn_score: '计算 Query 与 Key 的匹配分数',
      softmax: '把匹配分数变成位置竞争权重',
      attn_out: '按权重汇聚 Value 并经过输出投影',
      ffn_up: '从 d_model 扩张到 intermediate_size',
      ffn_act: 'SiLU/门控选择参与回写的特征',
      ffn_down: '从宽特征场压回 d_model',
      residual1: '合并 Attention 写入与原残差流',
      residual2: '合并 MLP 写入与原残差流',
    },
    noUnits: '当前层没有对应图谱候选',
    units: '图谱物理单元',
  },
  en: {
    input: {
      title: 'Layer Input Vector',
      shape: 'Shape: hidden-dimension bundle passes through an input portal into this Transformer layer.',
      mechanism: 'Role: carries the current token residual state for normalization, Attention, and MLP reads and writes.',
      unitType: 'Object: hidden-vector dimensions, not independent neurons.',
      boundary: 'No H#/N# address exists at this stage; channels are structural.',
    },
    ln: {
      title: 'Normalization Channel',
      shape: 'Shape: uneven input bundle → statistics and scaling core → stabilized output bundle.',
      mechanism: 'Role: rescales one token hidden vector so the next component receives a stable numerical range.',
      unitType: 'Object: d_model hidden dimensions and scaling parameters, not a neuron layer.',
      boundary: 'Normalization has no H#/N# atlas mapping; dimension slots are structural.',
    },
    attention: {
      title: 'Multi-Head Attention Router',
      shape: 'Shape: Q/K/V projections → head ring → attention matrix → output merge.',
      mechanism: 'Role: compares the current position with context and writes selected information through Attention heads.',
      unitType: 'Physical unit: Attention head, using an exact H# address.',
      boundary: 'H# is a physical-head candidate; single-unit causal closure remains 0.',
    },
    ffn: {
      title: 'MLP Neuron Expansion Field',
      shape: 'Shape: narrow d_model layer → wide intermediate neuron field → d_model write-back layer.',
      mechanism: 'Role: expands hidden state into a wider feature space, gates features, and writes selected features back.',
      unitType: 'Physical unit: N# is a SiLU(gate)⊙up MLP product neuron; G# is an MLP group object.',
      boundary: 'N#/G# remain atlas candidates until single-unit intervention passes.',
    },
    residual: {
      title: 'Residual Dual-Path Merger',
      shape: 'Shape: identity bypass + component write path merge at an addition node.',
      mechanism: 'Role: retains prior state while adding component output so information can travel across layers.',
      unitType: 'Object: residual vector and addition node, not independent neurons.',
      boundary: 'The view shows vector flow and norm, with no single-neuron atlas address.',
    },
    phases: {
      input: 'Read the current layer input state',
      ln1: 'Normalize before Attention',
      ln2: 'Normalize before MLP',
      qkv: 'Build Query, Key, and Value representations',
      attn_score: 'Compute Query-Key matching scores',
      softmax: 'Convert scores into positional competition weights',
      attn_out: 'Aggregate Value vectors and apply output projection',
      ffn_up: 'Expand d_model into intermediate_size',
      ffn_act: 'Use SiLU/gating to select write-back features',
      ffn_down: 'Compress the wide feature field into d_model',
      residual1: 'Merge the Attention write with the residual stream',
      residual2: 'Merge the MLP write with the residual stream',
    },
    noUnits: 'No matching atlas candidate in this layer',
    units: 'Atlas physical units',
  },
};

export function getComponentModelSpec({ component, phaseId, lang = 'en', evidenceUnits = [] }) {
  const locale = String(lang).toLowerCase().startsWith('zh') ? 'zh' : 'en';
  const copy = COPY[locale];
  const componentCopy = copy[component] || copy.input;
  const addresses = sortPatternAtlasNodes(evidenceUnits)
    .slice(0, 8)
    .map(patternAtlasUnitAddressLabel);

  return {
    ...componentCopy,
    phase: copy.phases[phaseId] || phaseId || '',
    shapeId: COMPONENT_MODEL_PHASES[phaseId]?.shapeId || COMPONENT_MODEL_PHASES.input.shapeId,
    unitSummary: component === 'attention' || component === 'ffn'
      ? `${copy.units}: ${addresses.length ? addresses.join(', ') : copy.noUnits}`
      : componentCopy.unitType,
    exactUnitCount: addresses.length,
    exactUnitAddresses: addresses,
  };
}
