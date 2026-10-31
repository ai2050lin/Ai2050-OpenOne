/**
 * 十大核心测试预设配置
 * 用于逆向工程3D可视化客户端的快捷入口
 */

export const TEST_PRESETS = [
  {
    id: 'T1', label: '语法正交验证', priority: 'high',
    description: 'S1-S5 × A3: 验证语法特征是否占据正交子空间',
    languageDims: { syntax: ['S1', 'S2', 'S3', 'S4', 'S5'] },
    dnnFeature: 'A3', viewMode: 'orthogonal',
    expectedResults: { cos: '<0.2', orthoGroup: '3维正交' },
    status: 'partial',
  },
  {
    id: 'T2', label: '1D流形验证', priority: 'high',
    description: 'S1-S8 × C2: 验证DS7B的1D因果流形跨语言维度',
    languageDims: { syntax: ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'] },
    dnnFeature: 'C2', viewMode: 'causal',
    expectedResults: { transferRate: '>0.9(语法), <0.5(逻辑)' },
    status: 'partial',
  },
  {
    id: 'T3', label: '频段分工验证', priority: 'high',
    description: 'S1-S8 × A5: 验证5频段与语法子维度的分工关系',
    languageDims: { syntax: ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'] },
    dnnFeature: 'A5', viewMode: 'spectral',
    expectedResults: { bandAssignment: '5频段×8语法子维度' },
    status: 'pending',
  },
  {
    id: 'T4', label: '语义基底验证', priority: 'medium',
    description: 'M1-M8 × A3: 验证语义类别的正交基底结构',
    languageDims: { semantic: ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'] },
    dnnFeature: 'A3', viewMode: 'orthogonal',
    expectedResults: { baseOrtho: '4-6正交基底' },
    status: 'pending',
  },
  {
    id: 'T5', label: '逻辑正交通道', priority: 'medium',
    description: 'L1-L8 × C2: 验证逻辑操作是否使用独立因果通道',
    languageDims: { logic: ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8'] },
    dnnFeature: 'C2', viewMode: 'causal',
    expectedResults: { channelIndep: '3-5独立通道' },
    status: 'pending',
  },
  {
    id: 'T6', label: '差分向量一致性', priority: 'medium',
    description: 'S2,S3,P1 × A1: 验证差分向量方向的语言学一致性',
    languageDims: { syntax: ['S2', 'S3'], pragmatic: ['P1'] },
    dnnFeature: 'A1', viewMode: 'orthogonal',
    expectedResults: { consistency: '>0.85方向一致' },
    status: 'pending',
  },
  {
    id: 'T7', label: 'PC1主分量验证', priority: 'high',
    description: 'S1-S8,M1-M8 × A2: PC1占比与语言维度的关系',
    languageDims: { syntax: ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'], semantic: ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'] },
    dnnFeature: 'A2', viewMode: 'structure',
    expectedResults: { pc1Range: '40-90%层级变化' },
    status: 'pending',
  },
  {
    id: 'T8', label: '方差陷阱检测', priority: 'medium',
    description: 'P6 × A2: 检测语用方差陷阱现象',
    languageDims: { pragmatic: ['P6'] },
    dnnFeature: 'A2', viewMode: 'structure',
    expectedResults: { trapDepth: '方差>3×均值' },
    status: 'pending',
  },
  {
    id: 'T9', label: '信息瓶颈定位', priority: 'medium',
    description: 'S1-S3,M1-M2 × I3: 定位语言信息压缩的信息瓶颈层',
    languageDims: { syntax: ['S1', 'S2', 'S3'], semantic: ['M1', 'M2'] },
    dnnFeature: 'I3', viewMode: 'encoding',
    expectedResults: { bottleneck: 'L8-L16瓶颈层' },
    status: 'pending',
  },
  {
    id: 'T10', label: '编码方程统一', priority: 'high',
    description: 'S1-S8,M1-M8,L1-L8 × I6: 验证编码方程R²>0.95',
    languageDims: { syntax: ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'], semantic: ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'], logic: ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8'] },
    dnnFeature: 'I6', viewMode: 'encoding',
    expectedResults: { r2: '>0.95统一编码' },
    status: 'pending',
  },
];

// 状态映射
export const STATUS_CONFIG = {
  confirmed: { label: '已确认', icon: '✅', color: '#10b981' },
  partial: { label: '部分验证', icon: '🔶', color: '#f59e0b' },
  pending: { label: '待验证', icon: '⬜', color: '#6b7280' },
  missing: { label: '缺失', icon: '❌', color: '#ef4444' },
};

// 根据预设ID应用语言维度选择
export function applyPresetToSelection(preset, currentSelection) {
  const newSelection = {};
  Object.keys(currentSelection).forEach((dimId) => {
    newSelection[dimId] = { ...currentSelection[dimId] };
    // 先重置该组
    Object.keys(newSelection[dimId]).forEach((subId) => {
      newSelection[dimId][subId] = false;
    });
  });
  // 应用预设
  if (preset.languageDims) {
    Object.entries(preset.languageDims).forEach(([dimId, subIds]) => {
      if (newSelection[dimId]) {
        subIds.forEach((subId) => {
          if (subId in newSelection[dimId]) {
            newSelection[dimId][subId] = true;
          }
        });
      }
    });
  }
  return newSelection;
}

// 获取优先级排序的预设
export function getPresetsByPriority() {
  const order = { high: 0, medium: 1, low: 2 };
  return [...TEST_PRESETS].sort((a, b) => (order[a.priority] || 99) - (order[b.priority] || 99));
}

// 统计完成情况
export function getPuzzleProgressSummary() {
  const counts = { confirmed: 0, partial: 0, pending: 0, missing: 0 };
  TEST_PRESETS.forEach((p) => {
    const s = p.status || 'pending';
    if (s in counts) counts[s]++;
    else counts.pending++;
  });
  return counts;
}
