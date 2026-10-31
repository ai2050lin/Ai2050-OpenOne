/**
 * 逆向工程颜色映射方案
 * 用于3D可视化客户端的神经元颜色映射和维度着色
 */

// 神经元颜色映射函数
export const COLOR_MAPS = {
  // 正交性: 蓝(正交) → 红(对齐)
  orthogonality: (cosValue) => {
    const clamped = Math.max(0, Math.min(1, Math.abs(cosValue)));
    const hue = (1 - clamped) * 240; // 240=蓝 → 0=红
    return `hsl(${hue}, 80%, 55%)`;
  },

  // 频段: 5色离散映射
  bandFrequency: (bandIndex) => ({
    1: '#38bdf8', // 频段1: 蓝
    2: '#22c55e', // 频段2: 绿
    3: '#fbbf24', // 频段3: 黄
    4: '#f97316', // 频段4: 橙
    5: '#ef4444', // 频段5: 红
  }[bandIndex] || '#666666'),

  // 因果效应: 透明(弱) → 亮(强)
  causalEffect: (effectValue) => {
    const intensity = Math.min(Math.max(effectValue, 0), 1);
    return `rgba(255, 107, 107, ${0.2 + intensity * 0.8})`;
  },

  // 子空间: 语言维度类别色
  subspace: (subspaceId) => ({
    syntax: '#4facfe',
    semantic: '#22c55e',
    logic: '#f97316',
    pragmatic: '#a78bfa',
    morphological: '#ec4899',
  }[subspaceId] || '#666666'),

  // 编码R²: 红(低) → 绿(高)
  encodingR2: (r2Value) => {
    const clamped = Math.max(0, Math.min(1, r2Value));
    const hue = clamped * 120; // 0=红 → 120=绿
    return `hsl(${hue}, 75%, 50%)`;
  },

  // 差分向量: 暖色系方向映射
  differentialVector: (angle) => {
    const normalized = ((angle % 360) + 360) % 360;
    const hue = (normalized / 360) * 360;
    return `hsl(${hue}, 70%, 60%)`;
  },

  // 信息量: 暗(低) → 亮(高)
  information: (infoValue) => {
    const clamped = Math.max(0, Math.min(1, infoValue));
    return `rgba(255, 217, 61, ${0.3 + clamped * 0.7})`;
  },
};

// 语言维度主色
export const DIMENSION_COLORS = {
  syntax: '#4facfe',
  semantic: '#22c55e',
  logic: '#f97316',
  pragmatic: '#a78bfa',
  morphological: '#ec4899',
};

// DNN特征主色
export const FEATURE_COLORS = {
  weight: '#38bdf8',
  activation: '#4ecdc4',
  causal: '#ff6b6b',
  information: '#ffd93d',
  dynamics: '#6c5ce7',
};

// 频段颜色
export const BAND_COLORS = {
  1: '#38bdf8',
  2: '#22c55e',
  3: '#fbbf24',
  4: '#f97316',
  5: '#ef4444',
};

// 频段标签
export const BAND_LABELS = {
  1: '极低频 (DC)',
  2: '低频 (空间)',
  3: '中频 (语义)',
  4: '高频 (语法)',
  5: '极高频 (局部)',
};

// 拼图状态色
export const PUZZLE_STATUS_COLORS = {
  confirmed: '#10b981',
  partial: '#f59e0b',
  missing: '#ef4444',
  pending: '#6b7280',
};

// 3D叠加效果颜色透明度
export const OVERLAY_OPACITY = {
  subspacePlane: 0.15,
  differentialArrow: 0.8,
  bandHalo: 0.4,
  causalParticle: 0.7,
  manifoldArrow: 0.9,
  equationLabel: 0.95,
};

// 根据DNN子特征ID获取颜色映射类型
export function getColorMapForFeature(featureId) {
  const featureMap = {
    W1: 'diverging', W2: 'sequential', W3: 'spectral', W4: 'sequential',
    W5: 'sequential', W6: 'sequential', W7: 'sequential', W8: 'diverging',
    A1: 'categorical', A2: 'sequential', A3: 'diverging', A4: 'sequential',
    A5: 'categorical', A6: 'cyclical', A7: 'sequential', A8: 'diverging',
    C1: 'sequential', C2: 'diverging', C3: 'sequential', C4: 'diverging',
    C5: 'diverging', C6: 'sequential', C7: 'sequential', C8: 'categorical',
    I1: 'sequential', I2: 'sequential', I3: 'diverging', I4: 'sequential',
    I5: 'diverging', I6: 'sequential',
    D1: 'categorical', D2: 'sequential', D3: 'diverging', D4: 'categorical',
    D5: 'diverging',
  };
  return featureMap[featureId] || 'sequential';
}

// 热力图颜色插值
export function heatmapColor(value) {
  // value: 0-1, 蓝到红
  const v = Math.max(0, Math.min(1, value));
  if (v < 0.25) {
    const t = v / 0.25;
    return `rgba(${Math.round(t * 255)}, ${Math.round(255 - t * 128)}, ${Math.round(255 - t * 255)}, 0.85)`;
  } else if (v < 0.5) {
    const t = (v - 0.25) / 0.25;
    return `rgba(${Math.round(255)}, ${Math.round(127 - t * 127)}, 0, 0.85)`;
  } else if (v < 0.75) {
    const t = (v - 0.5) / 0.25;
    return `rgba(255, ${Math.round(t * 165)}, 0, 0.85)`;
  } else {
    const t = (v - 0.75) / 0.25;
    return `rgba(255, ${Math.round(165 + t * 90)}, ${Math.round(t * 50)}, 0.85)`;
  }
}
