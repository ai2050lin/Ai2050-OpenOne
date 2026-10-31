/**
 * 语言维度配置
 * 5大语言维度 × 35子维度定义
 * 用于逆向工程3D可视化客户端
 */

export const LANGUAGE_DIMENSIONS = {
  syntax: {
    id: 'syntax',
    label: '语法维度',
    icon: 'Type',
    color: '#4facfe',
    subDimensions: [
      { id: 'S1', label: '词类编码', color: '#4facfe', testCases: ['cat vs run vs big'], keyMetrics: ['词类间余弦', '聚类紧密度'], dnnMapping: ['A3', 'A1', 'C1'] },
      { id: 'S2', label: '时态编码', color: '#00f2fe', testCases: ['walked vs walks vs will walk'], keyMetrics: ['差分向量方向', '范数比'], dnnMapping: ['A1', 'A3', 'A5'] },
      { id: 'S3', label: '极性编码', color: '#43e97b', testCases: ['good vs bad', 'hot vs cold'], keyMetrics: ['极性差分向量', '对称性'], dnnMapping: ['A3', 'C2'] },
      { id: 'S4', label: '数量编码', color: '#38f9d7', testCases: ['cat vs cats', 'one vs many'], keyMetrics: ['数量轴方向', '正交性'], dnnMapping: ['A1', 'A3'] },
      { id: 'S5', label: '语态编码', color: '#fa709a', testCases: ['active vs passive'], keyMetrics: ['语态差分向量', '一致性'], dnnMapping: ['C1', 'C3'] },
      { id: 'S6', label: '词序编码', color: '#fee140', testCases: ['dog bites man vs man bites dog'], keyMetrics: ['位置编码', '序列效应'], dnnMapping: ['A5', 'C2'] },
      { id: 'S7', label: '句法层级', color: '#a18cd1', testCases: ['nested clauses', 'center embedding'], keyMetrics: ['递归编码', '层级深度'], dnnMapping: ['A3', 'A4'] },
      { id: 'S8', label: '一致性', color: '#fbc2eb', testCases: ['subject-verb agreement'], keyMetrics: ['一致性信号', '干扰抵抗'], dnnMapping: ['A3', 'C1'] },
    ],
  },
  semantic: {
    id: 'semantic',
    label: '语义维度',
    icon: 'Brain',
    color: '#22c55e',
    subDimensions: [
      { id: 'M1', label: '语义类别', color: '#22c55e', testCases: ['animals vs tools vs places'], keyMetrics: ['类别聚类', '边界清晰度'], dnnMapping: ['A3', 'A1', 'W1'] },
      { id: 'M2', label: '多义性', color: '#10b981', testCases: ['bank (river vs money)'], keyMetrics: ['歧义消解方向', '上下文敏感度'], dnnMapping: ['A3', 'A4'] },
      { id: 'M3', label: '上下文消歧', color: '#34d399', testCases: ['river bank vs bank account'], keyMetrics: ['消歧路径', '注意力权重'], dnnMapping: ['C2', 'A1'] },
      { id: 'M4', label: '语义距离', color: '#6ee7b7', testCases: ['king-queen vs man-woman'], keyMetrics: ['距离保持', '各向同性'], dnnMapping: ['A1', 'A2'] },
      { id: 'M5', label: '类比推理', color: '#a7f3d0', testCases: ['king-man+woman=queen'], keyMetrics: ['类比精度', '方向一致性'], dnnMapping: ['A3', 'C2'] },
      { id: 'M6', label: '隐喻映射', color: '#86efac', testCases: ['time is money', 'love is war'], keyMetrics: ['映射方向', '源域激活'], dnnMapping: ['A4', 'A5'] },
      { id: 'M7', label: '语义场', color: '#4ade80', testCases: ['color field', 'emotion field'], keyMetrics: ['场一致性', '邻域结构'], dnnMapping: ['W1', 'A3'] },
      { id: 'M8', label: '原型效应', color: '#bbf7d0', testCases: ['robin vs penguin as bird'], keyMetrics: ['原型距离', '典型性梯度'], dnnMapping: ['A1', 'A3'] },
    ],
  },
  logic: {
    id: 'logic',
    label: '逻辑维度',
    icon: 'GitBranch',
    color: '#f97316',
    subDimensions: [
      { id: 'L1', label: '条件逻辑', color: '#f97316', testCases: ['if-then statements'], keyMetrics: ['条件编码方向', '蕴含信号'], dnnMapping: ['C2', 'C3'] },
      { id: 'L2', label: '否定逻辑', color: '#fb923c', testCases: ['not, never, no'], keyMetrics: ['否定向量', '语义反转'], dnnMapping: ['C1', 'A3'] },
      { id: 'L3', label: '量化逻辑', color: '#fdba74', testCases: ['all, some, none'], keyMetrics: ['量词编码', '范围信号'], dnnMapping: ['A3', 'I1'] },
      { id: 'L4', label: '因果推理', color: '#fed7aa', testCases: ['because, therefore'], keyMetrics: ['因果链强度', '方向性'], dnnMapping: ['C2', 'C4'] },
      { id: 'L5', label: '反事实', color: '#ffedd5', testCases: ['if it had not rained...'], keyMetrics: ['反事实距离', '可能性梯度'], dnnMapping: ['C2', 'C3'] },
      { id: 'L6', label: '逻辑连接词', color: '#f59e0b', testCases: ['and, or, but'], keyMetrics: ['连接词编码', '组合语义'], dnnMapping: ['A3', 'C1'] },
      { id: 'L7', label: '蕴涵关系', color: '#fbbf24', testCases: ['entailment, contradiction'], keyMetrics: ['蕴涵方向', '矛盾信号'], dnnMapping: ['C2', 'I1'] },
      { id: 'L8', label: '选择性稀释', color: '#fcd34d', testCases: ['selective attention decay'], keyMetrics: ['稀释率', '信息损失'], dnnMapping: ['A3', 'A5'] },
    ],
  },
  pragmatic: {
    id: 'pragmatic',
    label: '语用维度',
    icon: 'MessageSquare',
    color: '#a78bfa',
    subDimensions: [
      { id: 'P1', label: '正式度', color: '#a78bfa', testCases: ['formal vs informal speech'], keyMetrics: ['正式度轴', '语域信号'], dnnMapping: ['A1', 'A3'] },
      { id: 'P2', label: '礼貌度', color: '#c4b5fd', testCases: ['please vs give me'], keyMetrics: ['礼貌度轴', '间接程度'], dnnMapping: ['A3', 'C1'] },
      { id: 'P3', label: '间接言语', color: '#ddd6fe', testCases: ['can you pass the salt?'], keyMetrics: ['字面vs意图距离', '间接度'], dnnMapping: ['C2', 'A4'] },
      { id: 'P4', label: '言外之力', color: '#8b5cf6', testCases: ['promise, warn, request'], keyMetrics: ['施为信号', '力度编码'], dnnMapping: ['A3', 'C1'] },
      { id: 'P5', label: '话题聚焦', color: '#7c3aed', testCases: ['topic shift, focus change'], keyMetrics: ['话题向量', '聚焦强度'], dnnMapping: ['A5', 'C2'] },
      { id: 'P6', label: '方差陷阱', color: '#6d28d9', testCases: ['high variance pragmatic features'], keyMetrics: ['方差比', '陷阱深度'], dnnMapping: ['A3', 'A2'] },
    ],
  },
  morphological: {
    id: 'morphological',
    label: '形态维度',
    icon: 'Languages',
    color: '#ec4899',
    subDimensions: [
      { id: 'F1', label: '词根词缀', color: '#ec4899', testCases: ['un-happy-ness'], keyMetrics: ['词缀独立性', '频段分工'], dnnMapping: ['A5', 'W1'] },
      { id: 'F2', label: '屈折变化', color: '#f472b6', testCases: ['walk/walks/walked/walking'], keyMetrics: ['屈折轴方向', '规则vs不规则'], dnnMapping: ['A5', 'A1'] },
      { id: 'F3', label: '派生变化', color: '#f9a8d4', testCases: ['teach/teacher/teachable'], keyMetrics: ['派生路径', '语义偏移'], dnnMapping: ['A4', 'A5'] },
      { id: 'F4', label: '复合构词', color: '#fbcfe8', testCases: ['blackboard, sunflower'], keyMetrics: ['组合编码', '透明度'], dnnMapping: ['A3', 'C2'] },
      { id: 'F5', label: '跨语言频谱', color: '#db2777', testCases: ['multilingual morphology'], keyMetrics: ['跨语言一致性', '频谱差异'], dnnMapping: ['A5', 'I1'] },
    ],
  },
};

// 扁平化所有子维度列表
export const ALL_SUB_DIMENSIONS = Object.values(LANGUAGE_DIMENSIONS).flatMap(
  (dim) => dim.subDimensions.map((sub) => ({ ...sub, parentId: dim.id, parentLabel: dim.label, parentColor: dim.color }))
);

// 获取默认选中状态（全部关闭）
export function getDefaultLanguageDimSelection() {
  const selection = {};
  Object.entries(LANGUAGE_DIMENSIONS).forEach(([dimId, dim]) => {
    selection[dimId] = {};
    dim.subDimensions.forEach((sub) => {
      selection[dimId][sub.id] = false;
    });
  });
  return selection;
}

// 统计选中维度数量
export function countSelectedDims(selection) {
  return Object.values(selection).reduce(
    (total, group) => total + Object.values(group).filter(Boolean).length,
    0
  );
}

// 获取选中的子维度ID列表
export function getSelectedDimIds(selection) {
  return Object.values(selection).flatMap((group) =>
    Object.entries(group).filter(([, v]) => v).map(([id]) => id)
  );
}

// 根据预设应用维度选择
export function applyPresetToSelection(preset, currentSelection) {
  const newSelection = getDefaultLanguageDimSelection();
  if (preset.selectedDims) {
    preset.selectedDims.forEach((dimId) => {
      // dimId format: "S1", "M2", etc. Find parent group
      for (const [groupId, group] of Object.entries(LANGUAGE_DIMENSIONS)) {
        const sub = group.subDimensions.find((s) => s.id === dimId);
        if (sub) {
          newSelection[groupId][dimId] = true;
        }
      }
    });
  }
  return newSelection;
}
