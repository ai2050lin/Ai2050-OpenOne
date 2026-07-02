# DNN逆向工程3D可视化客户端修改方案
# DNN Reverse Engineering 3D Visualization Client Design

> **版本**: v2.0 | **日期**: 2026-04-24 | **核心原则**: 不改变当前风格和整体布局，只改变各个小窗口的内容，3D空间不改变layer模型

---

## 0. 设计原则

**当前布局架构（保持不变）**：
```
┌────────────────────────────────────────────────────────────┐
│ [设置] [蓝图] [AI]                    ← 顶部绝对定位按钮    │
│                                                            │
│ ┌──────────┐              ┌──────────┐                     │
│ │ 左上控制  │              │ 右上信息  │                     │
│ │ 面板     │   全屏3D     │ 面板     │                     │
│ │ 360px   │   Canvas     │ 360px   │                     │
│ │ SimplePanel│ (AppleNeuron│ SimplePanel│                   │
│ └──────────┘  SceneContent)└──────────┘                     │
│                                          ┌──────────┐       │
│                                          │ 右下操作  │       │
│                                          │ 面板     │       │
│                                          │ 360px   │       │
│                                          │ SimplePanel│     │
│                                          └──────────┘       │
└────────────────────────────────────────────────────────────┘
```

**修改策略**：
1. **布局不变**：保持4个浮动SimplePanel + 全屏Canvas的结构
2. **3D场景不变**：AppleNeuronSceneContent保持原样，仅通过props传递新的颜色映射信息
3. **只改面板内容**：在现有SimplePanel内替换为逆向工程相关的内容组件
4. **新增场景叠加层**：在主Canvas内条件渲染5种视角叠加效果（复用现有的structureTab切换机制）

---

## 1. 左上控制面板内容修改

### 现有内容
- `inputPanelTab` 切换：DNN / SNN / ICSPB
- DNN模式下显示 `LanguageResearchControlPanel`
- SNN模式下显示结构分析控件
- ICSPB模式下显示FiberNetPanel

### 修改方案：在DNN标签页内新增"逆向工程"子模式

**LanguageResearchControlPanel内新增子标签**：
```
LanguageResearchControlPanel/
├── [现有标签] 静态观测 / 动态预测 / 子空间几何 / ...
├── [新增] 逆向工程 (analysisMode = 'reverse_engineering')
│   ├── ReverseControlSection          # 逆向工程控制区
│   │   ├── LanguageDimensionAccordion # 语言维度手风琴选择器
│   │   │   ├── DimensionGroup(语法)   # S1-S8 toggle开关
│   │   │   ├── DimensionGroup(语义)   # M1-M8 toggle开关
│   │   │   ├── DimensionGroup(逻辑)   # L1-L8 toggle开关
│   │   │   ├── DimensionGroup(语用)   # P1-P6 toggle开关
│   │   │   └── DimensionGroup(形态)   # F1-F5 toggle开关
│   │   ├── DNNFeatureTabs             # DNN特征标签页
│   │   │   ├── 权重结构(W1-W8)        # radio选择
│   │   │   ├── 激活空间(A1-A8)        # radio选择
│   │   │   ├── 因果效应(C1-C8)        # radio选择
│   │   │   ├── 信息论(I1-I6)          # radio选择
│   │   │   └── 动力学(D1-D5)          # radio选择
│   │   ├── ViewModeSelect             # 3D视角模式选择
│   │   │   ├── 结构视图（默认）
│   │   │   ├── 正交视图
│   │   │   ├── 频谱视图
│   │   │   ├── 因果视图
│   │   │   └── 编码视图
│   │   └── QuickTestPresets            # 十大核心测试快捷入口
│   │       ├── T1 语法正交验证
│   │       ├── T2 1D流形验证
│   │       ├── ...
│   │       └── T10 编码方程统一
│   └── [保留现有其他模式的所有内容]
```

### 交互设计

**语言维度选择器**：
- 5大维度作为可折叠手风琴面板（默认展开语法）
- 每个子维度有独立toggle开关
- 选中后：①3D空间中对应神经元着色 ②右上数据面板联动更新
- 多个子维度可同时选择（颜色编码区分）

**DNN特征选择器**：
- 5大特征维度作为标签页切换
- 每个子特征radio选择（同一时间只激活一种特征映射）
- 切换特征：3D空间中神经元颜色映射改变

**3D视角模式**：
- 5种叠加效果，通过props传递给AppleNeuronSceneContent
- 不修改AppleNeuronSceneContent内部逻辑，只在渲染时叠加效果层

**快捷测试预设**：
- 点击预设自动配置左侧面板所有选项
- 显示预期结果和已有数据状态

---

## 2. 右上信息面板内容修改

### 现有内容
- 模型信息概览
- 编码焦点/概览/细节标签切换
- Apple主视图下显示LanguageResearchDataPanel

### 修改方案：逆向工程模式下替换内容

当 `analysisMode === 'reverse_engineering'` 时，右侧数据面板显示：

```
ReverseEngineeringDataPanel/          # 替换LanguageResearchDataPanel
├── SelectedDimensionSummary          # 选中维度概览
│   ├── DimensionTitle                # 维度名称+图标
│   ├── KeyMetricCards                # 关键指标卡片（4个）
│   │   ├── 正交性: cos=0.14
│   │   ├── PC1占比: 86.49%
│   │   ├── 因果效应: 递增
│   │   └── R²: 0.96
│   └── DimensionDescription          # 维度描述
├── CrossDimensionMatrix              # 语言×DNN交叉矩阵
│   ├── MiniHeatmap                   # 迷你热力图（canvas绘制）
│   │   └── 5语言维度 × 5DNN维度 (缩略)
│   └── 点击展开完整矩阵
├── FeatureDetailView                 # 特征详情视图
│   ├── DifferentialVectorChart       # 差分向量（canvas条形图）
│   ├── BandDecompositionChart        # 频段分解（5色条形图）
│   ├── CausalEffectChart             # 因果效应（信号强度图）
│   └── EncodingEquationDisplay       # 编码方程显示
│       └── logit = Σα_band × band_logit + β
└── PuzzleProgressSummary             # 拼图进度摘要
    ├── CompletionBar                 # 完成度进度条
    └── ConfirmedCount: 3/10          # 已确认数
```

### 数据展示映射

**语言维度 → 数据面板内容**：

| 选择 | 显示指标 |
|------|---------|
| 语法(S) | S1-S8正交矩阵cos值、差分向量方向、频段分布 |
| 语义(M) | 类别基底+偏移、多义性子空间、语义距离vs编码距离 |
| 逻辑(L) | 逻辑正交通道、因果/条件/否定编码、选择性稀释 |
| 语用(P) | 正式度维度、方差陷阱、礼貌度正交性 |
| 形态(F) | 词根词缀频段分工、屈折变化频域编码、跨语言频谱 |

**DNN特征 → 数据面板内容**：

| 选择 | 显示指标 |
|------|---------|
| 权重结构(W) | 正交程度、极化度、频谱分布、FFN稀疏度 |
| 激活空间(A) | 差分向量、PC1压缩率、子空间正交性、频段分工 |
| 因果效应(C) | Patching结果、Interchange迁移、Head贡献、方差陷阱 |
| 信息论(I) | 编码效率、信道容量、信息瓶颈、编码方程 |
| 动力学(D) | 训练动力学、梯度流、损失景观、相变 |

---

## 3. 右下操作面板内容修改

### 现有内容
- 操作面板标题
- 当前算法信息
- 操作按钮（数据模板/结果对比/操作历史）

### 修改方案：逆向工程模式下显示

```
ReverseEngineeringOperationPanel/
├── CurrentConfigSummary              # 当前配置摘要
│   ├── 模型: DeepSeek-7B
│   ├── 层范围: L0-L31
│   ├── 语言维度: S1,S2,S3 (已选3/35)
│   ├── DNN特征: A3-子空间正交性
│   └── 视角: 正交视图
├── ModelComparisonView               # 跨模型对比
│   ├── DS7B vs Qwen3 指标对比
│   └── 关键差异高亮
├── PuzzleProgressView                # 拼图进度视图
│   ├── T1 语法正交验证 ✅
│   ├── T2 1D流形验证   🔶(部分)
│   ├── T3 频段分工验证  ❌
│   ├── ...
│   └── T10 编码方程统一 ❌
├── ExperimentHistory                 # 实验历史
│   └── 最近5次操作记录
└── ActionButtons                     # 操作按钮
    ├── 运行分析
    ├── 导出数据
    └── 重置配置
```

---

## 4. 3D场景叠加效果（不改AppleNeuronSceneContent）

### 4.1 方案：在主Canvas中添加条件渲染的叠加层

**不修改** `AppleNeuronSceneContent` 组件本身。而是通过以下方式实现5种视角：

1. **颜色映射传递**：通过 `nodeDisplayEmphasis` 等现有props传递颜色映射信息
2. **场景叠加层**：在 `App.jsx` 的 Canvas 中，当处于逆向工程模式时，额外渲染叠加3D效果

### 4.2 叠加层组件设计

```jsx
{/* 在 App.jsx 的 Canvas 内，AppleNeuronSceneContent 之后 */}
{isAppleMainView && analysisMode === 'reverse_engineering' && (
  <ReverseEngineeringOverlay
    viewMode={reverseState.viewMode}           // structure|orthogonal|spectral|causal|encoding
    selectedLanguageDims={reverseState.selectedLanguageDims}
    selectedDNNFeature={reverseState.selectedDNNFeature}
    nodes={appleNeuronWorkspace.nodes}
    links={appleNeuronWorkspace.links}
    onHover={setHoveredInfo}
    onSelect={setDisplayInfo}
  />
)}
```

### 4.3 5种叠加效果

| 视角 | 叠加内容 | 不改变的部分 |
|------|---------|------------|
| **结构视图** | 无额外叠加（默认） | 原始AppleNeuronSceneContent |
| **正交视图** | 半透明彩色子空间平面 + 差分向量箭头 | 层级骨架和神经元位置 |
| **频谱视图** | 5频段颜色光晕叠加在神经元上 | 神经元几何形状和位置 |
| **因果视图** | 因果流线粒子 + 1D流形方向箭头 | 层间连接结构 |
| **编码视图** | 编码方程文字标签 + R²指示器 | 整体场景布局 |

### 4.4 ReverseEngineeringOverlay组件结构

```
ReverseEngineeringOverlay/
├── OrthogonalSubspaceOverlay        # 正交视图叠加
│   ├── SubspacePlane (语法,蓝色)
│   ├── SubspacePlane (语义,绿色)
│   ├── SubspacePlane (逻辑,橙色)
│   └── DifferentialVectorArrows     # 差分向量箭头
├── BandFrequencyOverlay             # 频谱视图叠加
│   ├── NeuronBandHalo (5色光晕)
│   └── BandLegend (频段图例)
├── CausalFlowOverlay                # 因果视图叠加
│   ├── CausalFlowParticles          # 因果流粒子
│   └── ManifoldDirectionArrows      # 1D流形方向箭头
└── EncodingEquationOverlay          # 编码视图叠加
    ├── EquationLabels               # 编码方程标签
    └── R2Indicators                 # R²指示器
```

---

## 5. 状态管理

### 5.1 扩展现有appleNeuronWorkspace

在现有的 `useAppleNeuronWorkspace` hook 中新增逆向工程状态：

```javascript
// 在 appleNeuronWorkspace 中新增
const [reverseEngineeringState, setReverseEngineeringState] = useState({
  // 语言维度选择
  selectedLanguageDims: {
    syntax: {},      // { S1: false, S2: true, ... }
    semantic: {},    // { M1: false, ... }
    logic: {},       // { L1: false, ... }
    pragmatic: {},   // { P1: false, ... }
    morphological: {}, // { F1: false, ... }
  },
  // DNN特征选择
  selectedDNNFeature: 'A3',      // 当前选中的DNN子特征ID
  selectedDNNCategory: 'activation', // 当前选中的DNN大类
  // 3D视角
  viewMode: 'structure',  // structure|orthogonal|spectral|causal|encoding
  // 快捷测试
  activePreset: null,     // 当前激活的测试预设ID
});
```

### 5.2 与现有面板的联动

```javascript
// 逆向工程模式下，面板标题和内容自动切换
const isReverseMode = appleNeuronWorkspace.analysisMode === 'reverse_engineering';

// 左上控制面板：使用LanguageResearchControlPanel内的逆向工程子模式
// 右上信息面板：条件渲染ReverseEngineeringDataPanel
// 右下操作面板：条件渲染ReverseEngineeringOperationPanel
// 3D场景：叠加ReverseEngineeringOverlay
```

---

## 6. 新增文件清单

### 6.1 新增组件（放在 src/components/reverse/ 目录下）

| 文件 | 功能 | 复用基础 |
|------|------|---------|
| `LanguageDimensionSelector.jsx` | 语言维度手风琴选择器 | 参考StructureAnalysisControls |
| `DimensionGroup.jsx` | 可折叠维度分组+toggle | 参考DNNAnalysisControlPanel |
| `DNNFeatureTabs.jsx` | DNN特征标签页+radio | 参考DNNAnalysisControlPanel |
| `QuickTestPresets.jsx` | 十大核心测试快捷入口 | 新组件 |
| `ViewModeSelect.jsx` | 3D视角模式选择 | 新组件 |
| `ReverseEngineeringDataPanel.jsx` | 右上数据面板内容 | 参考LanguageResearchDataPanel |
| `CrossDimensionMatrix.jsx` | 迷你交叉矩阵热力图 | 新组件(Canvas2D) |
| `FeatureDetailView.jsx` | 特征详情视图 | 参考DataDisplayTemplates |
| `PuzzleProgressView.jsx` | 拼图进度视图 | 新组件 |
| `ModelComparisonView.jsx` | 跨模型对比 | 新组件 |
| `ReverseEngineeringOperationPanel.jsx` | 右下操作面板内容 | 参考现有操作面板 |
| `ReverseEngineeringOverlay.jsx` | 3D叠加效果容器 | 参考StructureAnalysisPanel |
| `OrthogonalSubspaceOverlay.jsx` | 正交视图3D叠加 | 参考FeatureVisualization3D |
| `BandFrequencyOverlay.jsx` | 频谱视图3D叠加 | 参考ParameterEncoding3D |
| `CausalFlowOverlay.jsx` | 因果视图3D叠加 | 参考FlowTubesVisualizer |
| `EncodingEquationOverlay.jsx` | 编码视图3D叠加 | 新组件 |

### 6.2 新增配置

| 文件 | 功能 |
|------|------|
| `src/config/languageDimensions.js` | 语言5大维度35子维度定义 |
| `src/config/dnnFeatures.js` | DNN 5大维度35子维度定义 |
| `src/config/testPresets.js` | 十大核心测试预设配置 |
| `src/config/reverseColorMaps.js` | 颜色映射方案 |

### 6.3 修改文件

| 文件 | 修改内容 | 改动量 |
|------|---------|--------|
| `src/components/LanguageResearchControlPanel.jsx` | 新增 `reverse_engineering` analysisMode 分支 | 小 |
| `src/components/LanguageResearchDataPanel.jsx` | 新增逆向工程数据面板分支 | 小 |
| `src/App.jsx` | 右上/右下面板条件渲染逆向工程内容 + Canvas中添加叠加层 | 中 |
| `src/blueprint/appleNeuronWorkspaceBridge.js` | 暴露逆向工程状态 | 小 |

---

## 7. 配置数据结构

### 7.1 语言维度 (languageDimensions.js)

```javascript
export const LANGUAGE_DIMENSIONS = {
  syntax: {
    id: 'syntax', label: '语法维度', icon: 'Type', color: '#4facfe',
    subDimensions: [
      { id: 'S1', label: '词类编码', color: '#4facfe',
        testCases: ['cat vs run vs big'], keyMetrics: ['词类间余弦', '聚类紧密度'],
        dnnMapping: ['A3', 'A1', 'C1'] },
      { id: 'S2', label: '时态编码', color: '#00f2fe',
        testCases: ['walked vs walks vs will walk'], keyMetrics: ['差分向量方向', '范数比'],
        dnnMapping: ['A1', 'A3', 'A5'] },
      { id: 'S3', label: '极性编码', color: '#43e97b', dnnMapping: ['A3', 'C2'] },
      { id: 'S4', label: '数量编码', color: '#38f9d7', dnnMapping: ['A1', 'A3'] },
      { id: 'S5', label: '语态编码', color: '#fa709a', dnnMapping: ['C1', 'C3'] },
      { id: 'S6', label: '词序编码', color: '#fee140', dnnMapping: ['A5', 'C2'] },
      { id: 'S7', label: '句法层级', color: '#a18cd1', dnnMapping: ['A3', 'A4'] },
      { id: 'S8', label: '一致性',   color: '#fbc2eb', dnnMapping: ['A3', 'C1'] },
    ],
  },
  semantic: {
    id: 'semantic', label: '语义维度', icon: 'Brain', color: '#22c55e',
    subDimensions: [
      { id: 'M1', label: '语义类别', color: '#22c55e', dnnMapping: ['A3', 'A1', 'W1'] },
      { id: 'M2', label: '多义性',   color: '#10b981', dnnMapping: ['A3', 'A4'] },
      { id: 'M3', label: '上下文消歧', color: '#34d399', dnnMapping: ['C2', 'A1'] },
      { id: 'M4', label: '语义距离', color: '#6ee7b7', dnnMapping: ['A1', 'A2'] },
      { id: 'M5', label: '类比推理', color: '#a7f3d0', dnnMapping: ['A3', 'C2'] },
      { id: 'M6', label: '隐喻映射', color: '#86efac', dnnMapping: ['A4', 'A5'] },
      { id: 'M7', label: '语义场',   color: '#4ade80', dnnMapping: ['W1', 'A3'] },
      { id: 'M8', label: '原型效应', color: '#bbf7d0', dnnMapping: ['A1', 'A3'] },
    ],
  },
  logic: {
    id: 'logic', label: '逻辑维度', icon: 'GitBranch', color: '#f97316',
    subDimensions: [
      { id: 'L1', label: '条件逻辑', color: '#f97316', dnnMapping: ['C2', 'C3'] },
      { id: 'L2', label: '否定逻辑', color: '#fb923c', dnnMapping: ['C1', 'A3'] },
      { id: 'L3', label: '量化逻辑', color: '#fdba74', dnnMapping: ['A3', 'I1'] },
      { id: 'L4', label: '因果推理', color: '#fed7aa', dnnMapping: ['C2', 'C4'] },
      { id: 'L5', label: '反事实',   color: '#ffedd5', dnnMapping: ['C2', 'C3'] },
      { id: 'L6', label: '逻辑连接词', color: '#f59e0b', dnnMapping: ['A3', 'C1'] },
      { id: 'L7', label: '蕴涵关系', color: '#fbbf24', dnnMapping: ['C2', 'I1'] },
      { id: 'L8', label: '选择性稀释', color: '#fcd34d', dnnMapping: ['A3', 'A5'] },
    ],
  },
  pragmatic: {
    id: 'pragmatic', label: '语用维度', icon: 'MessageSquare', color: '#a78bfa',
    subDimensions: [
      { id: 'P1', label: '正式度',   color: '#a78bfa', dnnMapping: ['A1', 'A3'] },
      { id: 'P2', label: '礼貌度',   color: '#c4b5fd', dnnMapping: ['A3', 'C1'] },
      { id: 'P3', label: '间接言语', color: '#ddd6fe', dnnMapping: ['C2', 'A4'] },
      { id: 'P4', label: '言外之力', color: '#8b5cf6', dnnMapping: ['A3', 'C1'] },
      { id: 'P5', label: '话题聚焦', color: '#7c3aed', dnnMapping: ['A5', 'C2'] },
      { id: 'P6', label: '方差陷阱', color: '#6d28d9', dnnMapping: ['A3', 'A2'] },
    ],
  },
  morphological: {
    id: 'morphological', label: '形态维度', icon: 'Languages', color: '#ec4899',
    subDimensions: [
      { id: 'F1', label: '词根词缀', color: '#ec4899', dnnMapping: ['A5', 'W1'] },
      { id: 'F2', label: '屈折变化', color: '#f472b6', dnnMapping: ['A5', 'A1'] },
      { id: 'F3', label: '派生变化', color: '#f9a8d4', dnnMapping: ['A4', 'A5'] },
      { id: 'F4', label: '复合构词', color: '#fbcfe8', dnnMapping: ['A3', 'C2'] },
      { id: 'F5', label: '跨语言频谱', color: '#db2777', dnnMapping: ['A5', 'I1'] },
    ],
  },
};
```

### 7.2 DNN特征 (dnnFeatures.js)

```javascript
export const DNN_FEATURES = {
  weight: {
    id: 'weight', label: '权重结构', icon: 'Layers', color: '#38bdf8',
    subFeatures: [
      { id: 'W1', label: 'Layer正交程度', colorMap: 'diverging' },
      { id: 'W2', label: 'Head极化程度', colorMap: 'sequential' },
      { id: 'W3', label: 'W_U频谱分布', colorMap: 'spectral' },
      { id: 'W4', label: 'W_O泄漏比', colorMap: 'sequential' },
      { id: 'W5', label: 'FFN稀疏度', colorMap: 'sequential' },
      { id: 'W6', label: '权重范数分布', colorMap: 'sequential' },
      { id: 'W7', label: '残差流范数', colorMap: 'sequential' },
      { id: 'W8', label: 'RMSNorm缩放', colorMap: 'diverging' },
    ],
  },
  activation: {
    id: 'activation', label: '激活空间', icon: 'Activity', color: '#4ecdc4',
    subFeatures: [
      { id: 'A1', label: '差分向量方向', colorMap: 'categorical' },
      { id: 'A2', label: 'PC1压缩率', colorMap: 'sequential' },
      { id: 'A3', label: '子空间正交性', colorMap: 'diverging' },
      { id: 'A4', label: '激活流形维度', colorMap: 'sequential' },
      { id: 'A5', label: '频段分工', colorMap: 'categorical' },
      { id: 'A6', label: '跨层旋转', colorMap: 'cyclical' },
      { id: 'A7', label: '激活密度', colorMap: 'sequential' },
      { id: 'A8', label: '流形曲率', colorMap: 'diverging' },
    ],
  },
  causal: {
    id: 'causal', label: '因果效应', icon: 'Zap', color: '#ff6b6b',
    subFeatures: [
      { id: 'C1', label: 'Patching效应', colorMap: 'sequential' },
      { id: 'C2', label: 'Interchange迁移', colorMap: 'diverging' },
      { id: 'C3', label: 'Head贡献度', colorMap: 'sequential' },
      { id: 'C4', label: '回路必要性', colorMap: 'diverging' },
      { id: 'C5', label: '回路充分性', colorMap: 'diverging' },
      { id: 'C6', label: '信息流瓶颈', colorMap: 'sequential' },
      { id: 'C7', label: '间接效应', colorMap: 'sequential' },
      { id: 'C8', label: '1D因果流形', colorMap: 'categorical' },
    ],
  },
  information: {
    id: 'information', label: '信息论', icon: 'BarChart2', color: '#ffd93d',
    subFeatures: [
      { id: 'I1', label: '编码效率', colorMap: 'sequential' },
      { id: 'I2', label: '信道容量', colorMap: 'sequential' },
      { id: 'I3', label: '信息瓶颈', colorMap: 'diverging' },
      { id: 'I4', label: '互信息', colorMap: 'sequential' },
      { id: 'I5', label: 'KL散度', colorMap: 'diverging' },
      { id: 'I6', label: '编码方程R²', colorMap: 'sequential' },
    ],
  },
  dynamics: {
    id: 'dynamics', label: '动力学', icon: 'TrendingUp', color: '#6c5ce7',
    subFeatures: [
      { id: 'D1', label: '训练阶段', colorMap: 'categorical' },
      { id: 'D2', label: '梯度流', colorMap: 'sequential' },
      { id: 'D3', label: '损失景观', colorMap: 'diverging' },
      { id: 'D4', label: '相变检测', colorMap: 'categorical' },
      { id: 'D5', label: '紫牛效应', colorMap: 'diverging' },
    ],
  },
};
```

### 7.3 十大核心测试预设 (testPresets.js)

```javascript
export const TEST_PRESETS = [
  { id: 'T1', label: '语法正交验证', priority: 'high',
    description: 'S1-S5 × A3: 验证语法特征是否占据正交子空间',
    languageDims: { syntax: ['S1','S2','S3','S4','S5'] },
    dnnFeature: 'A3', viewMode: 'orthogonal',
    expectedResults: { cos: '<0.2', orthoGroup: '3维正交' },
    status: 'partial' },
  { id: 'T2', label: '1D流形验证', priority: 'high',
    description: 'S1-S8 × C2: 验证DS7B的1D因果流形跨语言维度',
    languageDims: { syntax: ['S1','S2','S3','S4','S5','S6','S7','S8'] },
    dnnFeature: 'C2', viewMode: 'causal',
    expectedResults: { transferRate: '>0.9(语法), <0.5(逻辑)' },
    status: 'partial' },
  { id: 'T3', label: '频段分工验证', priority: 'high',
    description: 'S1-S8 × A5: 验证5频段与语法子维度的分工关系',
    languageDims: { syntax: ['S1','S2','S3','S4','S5','S6','S7','S8'] },
    dnnFeature: 'A5', viewMode: 'spectral',
    expectedResults: { bandAssignment: '5频段×8语法子维度' },
    status: 'pending' },
  { id: 'T4', label: '语义基底验证', priority: 'medium',
    description: 'M1-M8 × A3: 验证语义类别的正交基底结构',
    languageDims: { semantic: ['M1','M2','M3','M4','M5','M6','M7','M8'] },
    dnnFeature: 'A3', viewMode: 'orthogonal',
    status: 'pending' },
  { id: 'T5', label: '逻辑正交通道', priority: 'medium',
    description: 'L1-L8 × C2: 验证逻辑操作是否使用独立因果通道',
    languageDims: { logic: ['L1','L2','L3','L4','L5','L6','L7','L8'] },
    dnnFeature: 'C2', viewMode: 'causal',
    status: 'pending' },
  { id: 'T6', label: '差分向量一致性', priority: 'medium',
    description: 'S2,S3,P1 × A1: 验证差分向量方向的语言学一致性',
    languageDims: { syntax: ['S2','S3'], pragmatic: ['P1'] },
    dnnFeature: 'A1', viewMode: 'orthogonal',
    status: 'pending' },
  { id: 'T7', label: 'PC1主分量验证', priority: 'high',
    description: '全部35子维度 × A2: PC1占比与语言维度的关系',
    languageDims: { syntax: ['S1','S2','S3','S4','S5','S6','S7','S8'],
                    semantic: ['M1','M2','M3','M4','M5','M6','M7','M8'] },
    dnnFeature: 'A2', viewMode: 'structure',
    status: 'pending' },
  { id: 'T8', label: '方差陷阱检测', priority: 'medium',
    description: 'P6 × A2: 检测语用方差陷阱现象',
    languageDims: { pragmatic: ['P6'] },
    dnnFeature: 'A2', viewMode: 'structure',
    status: 'pending' },
  { id: 'T9', label: '信息瓶颈定位', priority: 'medium',
    description: '全部 × I3: 定位语言信息压缩的信息瓶颈层',
    languageDims: { syntax: ['S1','S2','S3'], semantic: ['M1','M2'] },
    dnnFeature: 'I3', viewMode: 'encoding',
    status: 'pending' },
  { id: 'T10', label: '编码方程统一', priority: 'high',
    description: '全部35子维度 × I6: 验证编码方程R²>0.95',
    languageDims: { syntax: ['S1','S2','S3','S4','S5','S6','S7','S8'],
                    semantic: ['M1','M2','M3','M4','M5','M6','M7','M8'],
                    logic: ['L1','L2','L3','L4','L5','L6','L7','L8'] },
    dnnFeature: 'I6', viewMode: 'encoding',
    status: 'pending' },
];
```

---

## 8. 颜色映射方案 (reverseColorMaps.js)

```javascript
// 神经元颜色映射函数
export const COLOR_MAPS = {
  // 正交性: 蓝(正交) → 红(对齐)
  orthogonality: (cosValue) => {
    const hue = (1 - Math.abs(cosValue)) * 240;
    return `hsl(${hue}, 80%, 55%)`;
  },
  // 频段: 5色离散映射
  bandFrequency: (bandIndex) => ({
    1: '#38bdf8', 2: '#22c55e', 3: '#fbbf24', 4: '#f97316', 5: '#ef4444'
  }[bandIndex]),
  // 因果效应: 透明(弱) → 亮(强)
  causalEffect: (effectValue) => {
    const intensity = Math.min(effectValue * 2, 1);
    return `rgba(255, 107, 107, ${intensity})`;
  },
  // 子空间: 语言维度类别色
  subspace: (subspaceId) => ({
    syntax: '#4facfe', semantic: '#22c55e', logic: '#f97316',
    pragmatic: '#a78bfa', morphological: '#ec4899',
  }[subspaceId]),
};

// 语言维度主色
export const DIMENSION_COLORS = {
  syntax: '#4facfe', semantic: '#22c55e', logic: '#f97316',
  pragmatic: '#a78bfa', morphological: '#ec4899',
};

// DNN特征主色
export const FEATURE_COLORS = {
  weight: '#38bdf8', activation: '#4ecdc4', causal: '#ff6b6b',
  information: '#ffd93d', dynamics: '#6c5ce7',
};
```

---

## 9. 实施步骤

### Phase 1: 配置与基础设施 (1天)
1. 创建 `src/config/languageDimensions.js`
2. 创建 `src/config/dnnFeatures.js`
3. 创建 `src/config/testPresets.js`
4. 创建 `src/config/reverseColorMaps.js`
5. 在 `appleNeuronWorkspaceBridge.js` 中添加逆向工程状态

### Phase 2: 左上控制面板内容 (2-3天)
1. 创建 `src/components/reverse/LanguageDimensionSelector.jsx`
2. 创建 `src/components/reverse/DimensionGroup.jsx`
3. 创建 `src/components/reverse/DNNFeatureTabs.jsx`
4. 创建 `src/components/reverse/QuickTestPresets.jsx`
5. 创建 `src/components/reverse/ViewModeSelect.jsx`
6. 修改 `LanguageResearchControlPanel.jsx` 添加逆向工程分支

### Phase 3: 右上面板内容 (2天)
1. 创建 `src/components/reverse/ReverseEngineeringDataPanel.jsx`
2. 创建 `src/components/reverse/CrossDimensionMatrix.jsx`
3. 创建 `src/components/reverse/FeatureDetailView.jsx`
4. 修改 `LanguageResearchDataPanel.jsx` 添加逆向工程分支

### Phase 4: 右下面板内容 (1-2天)
1. 创建 `src/components/reverse/ReverseEngineeringOperationPanel.jsx`
2. 创建 `src/components/reverse/PuzzleProgressView.jsx`
3. 创建 `src/components/reverse/ModelComparisonView.jsx`
4. 修改 `App.jsx` 右下面板条件渲染

### Phase 5: 3D叠加效果 (2-3天)
1. 创建 `src/components/reverse/ReverseEngineeringOverlay.jsx`
2. 创建 `src/components/reverse/OrthogonalSubspaceOverlay.jsx`
3. 创建 `src/components/reverse/BandFrequencyOverlay.jsx`
4. 创建 `src/components/reverse/CausalFlowOverlay.jsx`
5. 创建 `src/components/reverse/EncodingEquationOverlay.jsx`
6. 修改 `App.jsx` Canvas中添加叠加层

### Phase 6: 集成与优化 (1天)
1. 三面板联动调试
2. 状态持久化
3. 性能优化
4. 样式微调

**预计总工期: 9-12天**

---

## 10. 与现有组件的复用策略

| 现有组件 | 复用方式 |
|---------|---------|
| `SimplePanel` | 直接复用，不改 |
| `AppleNeuronSceneContent` | 不改，仅叠加效果层 |
| `LanguageResearchControlPanel` | 扩展，新增analysisMode分支 |
| `LanguageResearchDataPanel` | 扩展，新增逆向工程数据面板 |
| `DNNAnalysisControlPanel` | 参考样式和交互模式 |
| `DataDisplayTemplates` (MetricCard等) | 直接复用 |
| `ParameterEncoding3D` | 参考频段颜色方案 |
| `FeatureVisualization3D` | 参考正交视图3D渲染 |
| `FlowTubesVisualizer` | 参考因果流线渲染 |
| `colors.js` / `panels.js` | 复用，扩展 |

---

## 11. 设计规范

### 11.1 颜色（复用现有暗色毛玻璃主题）

```css
/* 面板背景 — 与现有SimplePanel一致 */
--panel-bg: rgba(20, 20, 25, 0.8);
--panel-border: rgba(255, 255, 255, 0.1);
--panel-radius: 12px;

/* 语言维度主色 */
--dim-syntax: #4facfe;
--dim-semantic: #22c55e;
--dim-logic: #f97316;
--dim-pragmatic: #a78bfa;
--dim-morphological: #ec4899;

/* DNN特征主色 */
--feat-weight: #38bdf8;
--feat-activation: #4ecdc4;
--feat-causal: #ff6b6b;
--feat-information: #ffd93d;
--feat-dynamics: #6c5ce7;

/* 频段颜色 */
--band-1: #38bdf8;
--band-2: #22c55e;
--band-3: #fbbf24;
--band-4: #f97316;
--band-5: #ef4444;

/* 拼图状态色 */
--status-confirmed: #10b981;
--status-partial: #f59e0b;
--status-missing: #ef4444;
--status-pending: #6b7280;
```

### 11.2 面板尺寸（与现有面板一致）

```css
/* 左上控制面板 */
--input-panel-width: 360px;
--input-panel-max-height: 85vh;

/* 右上信息面板 */
--info-panel-width: 360px;
--info-panel-max-height: 80vh;

/* 右下操作面板 */
--operation-panel-width: 360px;
--operation-panel-max-height: 60vh;
```

### 11.3 字体（与现有风格一致）

```css
--font-panel-title: 18px / bold;   /* SimplePanel标题 */
--font-section-title: 13px / 700;  /* 区域标题 */
--font-label: 12px / 400;          /* 标签 */
--font-metric-value: 16px / 700;   /* 指标值 */
--font-metric-label: 10px / 400;   /* 指标标签 */
```
