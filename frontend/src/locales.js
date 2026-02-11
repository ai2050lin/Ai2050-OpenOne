export const locales = {
  zh: {
    app: {
      title: 'Transformer 结构分析',
      mlpDistribution: 'MLP激活分布',
      computingAttention: '⚡ 计算注意力...',
      processingMlp: '🔄 处理MLP...',
      generatingOutput: '✨ 生成输出...',
    },
    panels: {
      inputPanel: '控制面板 (左上)',
      infoPanel: '信息面板 (左下)',
      layersPanel: '层列表 (右下)',
      structurePanel: '结构分析面板',
      modelInfo: '模型说明',
      neuronPanel: '神经元状态面板',
      neuronStateTitle: '第 {{layer}} 层神经元状态',
      headPanel: '注意力头面板',

      globalConfig: '界面配置',
      resetLayout: '重置所有面板位置',
      resetConfig: '重置布局',
      showSidebar: '显示侧边栏',
      showResults: '显示结果浮窗',
      drag: '拖动'
    },
    validity: {
      title: '语言有效性分析',
      evaluating: '正在评估以下内容的数学有效性:',
      analyze: '分析',
      reanalyze: '重新分析',
      analyzing: '全量分析中...',
      perplexity: '困惑度 (PPL)',
      pplDesc: '越低越可预测 (Low = Good)',
      entropy: '平均熵 (Entropy)',
      entropyDesc: '不确定性 (Uncertainty)',
      entropyStats: '熵统计 (Entropy Statistics)',
      min: '最小',
      max: '最大',
      mean: '均值',
      variance: '方差',
      anisotropy: '层各向异性 (表征坍缩度量)',
      collapseWarning: '值接近 1.0 表示严重的表征坍缩 (Representation Collapse)。',
      clickToAnalyze: '点击“分析”以计算语言有效性指标。',
      layer: '第 {{layer}} 层',
      l: 'L'
    },
    structure: {
      title: '结构分析',
      clear: '清除',
      tabs: {
        circuit: '回路',
        features: '特征',
        causal: '因果',
        manifold: '流形',
        compositional: '组合性',
        agi: 'AGI验证',
        snn: '脉冲仿真',
        validity: '有效性',
        model_generation: '3D模型生成',
        gut_relationship: 'GUT理论关系'
      },
      circuit: {
        title: '回路发现',
        desc: '通过差异激活分析，寻找执行特定任务的最小子网络。',
        cleanPrompt: '干净提示词 (Clean)',
        corruptedPrompt: '损坏提示词 (Corrupted)',
        threshold: '修剪阈值',
        run: '运行回路发现',
        running: '分析中...'
      },
      features: {
        title: '稀疏特征提取',
        desc: '使用稀疏自编码器 (SAE) 提取可解释的神经元特征方向。',
        prompt: '输入提示词',
        layer: '分析层级',
        dim: '隐藏层维度',
        sparsity: '稀疏系数',
        epochs: '训练轮数',
        run: '运行特征提取',
        running: '训练中...'
      },
      causal: {
        title: '因果干预分析',
        desc: '通过激活补丁 (Activation Patching) 定位对输出有因果影响的组件。',
        prompt: '输入提示词',
        targetPos: '目标Token位置 (-1 为最后)',
        threshold: '重要性阈值',
        run: '运行因果分析',
        running: '分析中 (约1-2分钟)...'
      },
      manifold: {
        title: '神经流形几何',
        desc: '分析表示空间的几何结构和固有维度 (Intrinsic Dimensionality)。',
        prompt: '输入提示词',
        layer: '分析层级',
        run: '运行流形分析',
        running: '计算中...'
      },
      compositional: {
        title: '组合性分析',
        desc: '分析短语表示的向量算术性质 (v(AB) ≈ v(A) + v(B))。',
        layer: '分析层级',
        phrases: '测试短语 (CSV格式)',
        format: '格式: 词1, 词2, 组合词 (例如: "black, cat, black cat")',
        run: '运行组合性分析',
        running: '分析中...'
      },
      layer3d: {
        layer: '第 {{layer}} 层',
        heads: '多头注意力 ({{count}} 头)',
        mlp: 'MLP (前馈网络)',
        norm: '层归一化'
      }
    },
    head: {
      title: '第 {{layer}} 层 第 {{head}} 头 分析',
      pattern: '注意力模式',
      qkv: 'Q / K / V / 输出',
      patternDesc: '注意力模式 (源 Token → 目标 Token)',
      q: 'Query (Q)',
      k: 'Key (K)',
      v: 'Value (V)',
      out: 'Output (Z)',
      loading: '正在加载注意力头分析...',
      val: '值'
    },
    common: {
      language: '语言 (Language)',
      loading: '加载中...',
      error: '错误'
    }
  },
  en: {
    app: {
      title: 'Transformer Structure Analysis',
      mlpDistribution: 'MLP Activation Distribution',
      computingAttention: '⚡ Computing Attention...',
      processingMlp: '🔄 Processing MLP...',
      generatingOutput: '✨ Generating Output...',
    },
    panels: {
      inputPanel: 'Control Panel (Top Left)',
      infoPanel: 'Info Panel (Bottom Left)',
      layersPanel: 'Layer List (Bottom Right)',
      structurePanel: 'Structure Analysis',
      modelInfo: 'Model Info',
      neuronPanel: 'Neuron State',
      neuronStateTitle: 'Layer {{layer}} Neuron State',
      headPanel: 'Attention Heads',

      globalConfig: 'Interface Configuration',
      resetLayout: 'Reset All Panel Positions',
      resetConfig: 'Reset Layout',
      showSidebar: 'Show Sidebar',
      showResults: 'Show Results Overlay',
      drag: 'Drag'
    },
    validity: {
      title: 'Language Validity Analysis',
      evaluating: 'Evaluating mathematical validity of:',
      analyze: 'Analyze',
      reanalyze: 'Re-Analyze',
      analyzing: 'Analyzing...',
      perplexity: 'Perplexity (PPL)',
      pplDesc: 'Low = Predictable',
      entropy: 'Avg Entropy',
      entropyDesc: 'Uncertainty',
      entropyStats: 'Entropy Statistics',
      min: 'Min',
      max: 'Max',
      mean: 'Mean',
      variance: 'Variance',
      anisotropy: 'Layer Anisotropy (Collapse)',
      collapseWarning: 'Values near 1.0 indicate severe representation collapse.',
      clickToAnalyze: 'Click "Analyze" to calculate metrics.',
      layer: 'Layer {{layer}}',
      l: 'L'
    },
    structure: {
      title: 'Structure Analysis',
      clear: 'Clear',
      tabs: {
        circuit: 'Circuit',
        features: 'Features',
        causal: 'Causal',
        manifold: 'Manifold',
        compositional: 'Compositional',
        agi: 'AGI Verify',
        snn: 'SNN Sim',
        validity: 'Validity'
      },
      circuit: {
        title: 'Circuit Discovery',
        desc: 'Find minimal subnetworks for specific tasks via differential activation analysis.',
        cleanPrompt: 'Clean Prompt',
        corruptedPrompt: 'Corrupted Prompt',
        threshold: 'Pruning Threshold',
        run: 'Run Circuit Discovery',
        running: 'Analyzing...'
      },
      features: {
        title: 'Sparse Feature Extraction',
        desc: 'Extract interpretable features using Sparse Autoencoders (SAE).',
        prompt: 'Input Prompt',
        layer: 'Layer Index',
        dim: 'Hidden Dimension',
        sparsity: 'Sparsity Coef',
        epochs: 'Epochs',
        run: 'Run Feature Extraction',
        running: 'Training...'
      },
      causal: {
        title: 'Causal Mediation Analysis',
        desc: 'Locate causally important components via Activation Patching.',
        prompt: 'Input Prompt',
        targetPos: 'Target Token Pos (-1 for last)',
        threshold: 'Importance Threshold',
        run: 'Run Causal Analysis',
        running: 'Analyzing (1-2 mins)...'
      },
      manifold: {
        title: 'Neural Manifold Geometry',
        desc: 'Analyze geometry and Intrinsic Dimensionality (ID) of representation space.',
        prompt: 'Input Prompt',
        layer: 'Layer Index',
        run: 'Run Manifold Analysis',
        running: 'Computing...'
      },
      compositional: {
        title: 'Compositional Analysis',
        desc: 'Analyze vector arithmetic properties of phrase representations.',
        layer: 'Layer Index',
        phrases: 'Test Phrases (CSV)',
        format: 'Format: word1, word2, compound (e.g., "black, cat, black cat")',
        run: 'Run Analysis',
        running: 'Analyzing...'
      },
      layer3d: {
        layer: 'Layer {{layer}}',
        heads: 'Attention Heads ({{count}})',
        mlp: 'MLP (Feed Forward)',
        norm: 'Layer Norm'
      }
    },
    head: {
      title: 'Layer {{layer}} Head {{head}} Analysis',
      pattern: 'Attention Pattern',
      qkv: 'Q / K / V / Out',
      patternDesc: 'Attention Pattern (Source -> Dest)',
      q: 'Query (Q)',
      k: 'Key (K)',
      v: 'Value (V)',
      out: 'Output (Z)',
      loading: 'Loading head analysis...',
      val: 'Val'
    },
    common: {
      language: 'Language',
      loading: 'Loading...',
      error: 'Error'
    }
  }
};
