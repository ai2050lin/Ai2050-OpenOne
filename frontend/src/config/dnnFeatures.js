/**
 * DNN特征维度配置
 * 5大DNN特征维度 × 35子特征定义
 * 用于逆向工程3D可视化客户端
 */

export const DNN_FEATURES = {
  weight: {
    id: 'weight',
    label: '权重结构',
    icon: 'Layers',
    color: '#38bdf8',
    subFeatures: [
      { id: 'W1', label: 'Layer正交程度', colorMap: 'diverging', description: '各层权重矩阵的正交程度' },
      { id: 'W2', label: 'Head极化程度', colorMap: 'sequential', description: '注意力头权重的极化分布' },
      { id: 'W3', label: 'W_U频谱分布', colorMap: 'spectral', description: '输出投影矩阵的频谱特征' },
      { id: 'W4', label: 'W_O泄漏比', colorMap: 'sequential', description: '输出投影的信息泄漏比例' },
      { id: 'W5', label: 'FFN稀疏度', colorMap: 'sequential', description: '前馈网络权重的稀疏程度' },
      { id: 'W6', label: '权重范数分布', colorMap: 'sequential', description: '各层权重范数的分布' },
      { id: 'W7', label: '残差流范数', colorMap: 'sequential', description: '残差连接的范数变化' },
      { id: 'W8', label: 'RMSNorm缩放', colorMap: 'diverging', description: 'RMSNorm缩放因子分布' },
    ],
  },
  activation: {
    id: 'activation',
    label: '激活空间',
    icon: 'Activity',
    color: '#4ecdc4',
    subFeatures: [
      { id: 'A1', label: '差分向量方向', colorMap: 'categorical', description: '对比对的差分向量方向' },
      { id: 'A2', label: 'PC1压缩率', colorMap: 'sequential', description: '第一主成分的方差解释率' },
      { id: 'A3', label: '子空间正交性', colorMap: 'diverging', description: '语言特征子空间的正交程度' },
      { id: 'A4', label: '激活流形维度', colorMap: 'sequential', description: '激活空间的内在维度' },
      { id: 'A5', label: '频段分工', colorMap: 'categorical', description: '5频段与语言特征的对应关系' },
      { id: 'A6', label: '跨层旋转', colorMap: 'cyclical', description: '层间表示的旋转角度' },
      { id: 'A7', label: '激活密度', colorMap: 'sequential', description: '激活值的密度分布' },
      { id: 'A8', label: '流形曲率', colorMap: 'diverging', description: '激活流形的曲率特征' },
    ],
  },
  causal: {
    id: 'causal',
    label: '因果效应',
    icon: 'Zap',
    color: '#ff6b6b',
    subFeatures: [
      { id: 'C1', label: 'Patching效应', colorMap: 'sequential', description: '激活修补的因果效应' },
      { id: 'C2', label: 'Interchange迁移', colorMap: 'diverging', description: '互换干预的因果迁移率' },
      { id: 'C3', label: 'Head贡献度', colorMap: 'sequential', description: '各注意力头的因果贡献' },
      { id: 'C4', label: '回路必要性', colorMap: 'diverging', description: '因果回路的必要性评分' },
      { id: 'C5', label: '回路充分性', colorMap: 'diverging', description: '因果回路的充分性评分' },
      { id: 'C6', label: '信息流瓶颈', colorMap: 'sequential', description: '信息流的关键瓶颈层' },
      { id: 'C7', label: '间接效应', colorMap: 'sequential', description: '间接因果效应的强度' },
      { id: 'C8', label: '1D因果流形', colorMap: 'categorical', description: '1D因果流形的维度' },
    ],
  },
  information: {
    id: 'information',
    label: '信息论',
    icon: 'BarChart2',
    color: '#ffd93d',
    subFeatures: [
      { id: 'I1', label: '编码效率', colorMap: 'sequential', description: '语言特征的编码效率' },
      { id: 'I2', label: '信道容量', colorMap: 'sequential', description: '各层的信道容量' },
      { id: 'I3', label: '信息瓶颈', colorMap: 'diverging', description: '信息瓶颈的位置与程度' },
      { id: 'I4', label: '互信息', colorMap: 'sequential', description: '语言特征与DNN表示的互信息' },
      { id: 'I5', label: 'KL散度', colorMap: 'diverging', description: '层间表示的KL散度' },
      { id: 'I6', label: '编码方程R²', colorMap: 'sequential', description: '编码方程的R²拟合度' },
    ],
  },
  dynamics: {
    id: 'dynamics',
    label: '动力学',
    icon: 'TrendingUp',
    color: '#6c5ce7',
    subFeatures: [
      { id: 'D1', label: '训练阶段', colorMap: 'categorical', description: '训练过程中的阶段划分' },
      { id: 'D2', label: '梯度流', colorMap: 'sequential', description: '梯度流动的范数和方向' },
      { id: 'D3', label: '损失景观', colorMap: 'diverging', description: '损失景观的曲率和平坦度' },
      { id: 'D4', label: '相变检测', colorMap: 'categorical', description: '训练过程中的相变现象' },
      { id: 'D5', label: '紫牛效应', colorMap: 'diverging', description: '罕见但高影响的突变现象' },
    ],
  },
};

// 扁平化所有子特征列表
export const ALL_SUB_FEATURES = Object.values(DNN_FEATURES).flatMap(
  (feat) => feat.subFeatures.map((sub) => ({ ...sub, parentId: feat.id, parentLabel: feat.label, parentColor: feat.color }))
);

// 根据子特征ID查找其父分类
export function findFeatureCategory(subFeatureId) {
  for (const [catId, cat] of Object.entries(DNN_FEATURES)) {
    if (cat.subFeatures.some((s) => s.id === subFeatureId)) {
      return catId;
    }
  }
  return null;
}

// 根据子特征ID获取完整特征对象
export function findSubFeature(subFeatureId) {
  for (const cat of Object.values(DNN_FEATURES)) {
    const found = cat.subFeatures.find((s) => s.id === subFeatureId);
    if (found) return { ...found, parentId: cat.id, parentLabel: cat.label, parentColor: cat.color };
  }
  return null;
}
