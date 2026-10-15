import { TRAINING_API_BASE } from '../config.js';
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Brain, Zap, Layers, Activity, TrendingUp, Database, Search, ChevronDown, ChevronUp, Play, BarChart3, Box } from 'lucide-react';
import DNNAnalysis3DVisualization from './DNNAnalysis3DVisualization';



/**
 * DNN分析控制面板
 * 为深度神经网络分析提供统一的控制界面，支持后续数据扩充
 */
const DNNAnalysisControlPanel = ({ onAnalysisRequest, onDataSelect }) => {
  // 核心状态
  const [activeSection, setActiveSection] = useState('analysis'); // 当前激活的区域
  const [selectedModel, setSelectedModel] = useState('deepseek7b');
  const [selectedLayerRange, setSelectedLayerRange] = useState({ start: 0, end: 32 });

  // 分析状态
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // 3D可视化状态
  const [show3DVisualization, setShow3DVisualization] = useState(false);

  // 数据源状态
  const [dataSources, setDataSources] = useState([]);
  const [availableModels, setAvailableModels] = useState([]);

  // 搜索状态
  const [searchTerm, setSearchTerm] = useState('');

  // DNN分析维度定义
  const analysisDimensions = [
    {
      id: 'encoding_structure',
      name: '编码结构',
      icon: Layers,
      description: '分析神经元激活模式和编码结构',
      api: '/api/dnn/encoding-structure',
    },
    {
      id: 'attention_patterns',
      name: '注意力模式',
      icon: Activity,
      description: '分析层间注意力流动和路径',
      api: '/api/dnn/attention-patterns',
    },
    {
      id: 'feature_extractions',
      name: '特征提取',
      icon: Zap,
      description: '提取和分类关键特征',
      api: '/api/dnn/feature-extractions',
    },
    {
      id: 'layer_dynamics',
      name: '层间动力学',
      icon: TrendingUp,
      description: '分析层间信息传播和演化',
      api: '/api/dnn/layer-dynamics',
    },
    {
      id: 'neuron_groups',
      name: '神经元分组',
      icon: Brain,
      description: '识别神经元聚类和功能分区',
      api: '/api/dnn/neuron-groups',
    },
    {
      id: 'data_foundation',
      name: '数据基础',
      icon: Database,
      description: '查看和管理基础数据集',
      api: '/api/dnn/data-foundation',
    },
  ];

  // 可用模型列表
  const modelOptions = [
    { id: 'deepseek7b', name: 'DeepSeek 7B', size: '7B', layers: 32 },
    { id: 'qwen3', name: 'Qwen3', size: '14B', layers: 40 },
    { id: 'gpt2', name: 'GPT-2', size: '1.5B', layers: 48 },
    { id: 'llama2', name: 'LLaMA-2', size: '7B', layers: 32 },
  ];

  // 常用概念快捷选择
  const quickConcepts = [
    { id: 'apple', name: '苹果', emoji: '🍎' },
    { id: 'banana', name: '香蕉', emoji: '🍌' },
    { id: 'orange', name: '橙子', emoji: '🍊' },
    { id: 'dog', name: '狗', emoji: '🐕' },
    { id: 'cat', name: '猫', emoji: '🐱' },
  ];

  // 初始化数据
  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      // 加载可用的数据源和模型
      const [sourcesRes, modelsRes] = await Promise.all([
        axios.get(`${TRAINING_API_BASE}/api/dnn/data-sources`),
        axios.get(`${TRAINING_API_BASE}/api/dnn/available-models`),
      ]);

      setDataSources(sourcesRes.data || []);
      setAvailableModels(modelsRes.data || modelOptions);
    } catch (error) {
      console.error('加载初始数据失败:', error);
      // 使用默认数据
      setAvailableModels(modelOptions);
    }
  };

  // 执行DNN分析
  const runDNNAnalysis = async (dimension) => {
    setIsAnalyzing(true);
    try {
      const response = await axios.post(`${TRAINING_API_BASE}${dimension.api}`, {
        model: selectedModel,
        layerRange: [selectedLayerRange.start, selectedLayerRange.end],
        searchTerm: searchTerm || null,
      });

      setAnalysisResults({
        dimension: dimension.id,
        data: response.data,
        timestamp: new Date().toISOString(),
      });

      // 通知父组件
      if (onAnalysisRequest) {
        onAnalysisRequest({
          type: 'dnn_analysis',
          dimension: dimension.id,
          model: selectedModel,
          results: response.data,
        });
      }
    } catch (error) {
      console.log('使用模拟数据（后端API未连接）:', error);
      // 模拟数据（用于演示）
      const mockData = generateMockAnalysisData(dimension.id);
      console.log('模拟数据:', mockData);
      setAnalysisResults({
        dimension: dimension.id,
        data: mockData,
        timestamp: new Date().toISOString(),
      });

      if (onAnalysisRequest) {
        onAnalysisRequest({
          type: 'dnn_analysis',
          dimension: dimension.id,
          model: selectedModel,
          results: mockData,
        });
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  // 生成模拟分析数据
  const generateMockAnalysisData = (dimensionId) => {
    const mockData = {
      encoding_structure: {
        totalNeurons: 2048,
        activeNeurons: 1234,
        encodingEfficiency: 0.87,
        clusters: 15,
        layers: selectedLayerRange,
      },
      attention_patterns: {
        totalAttention: 100,
        activePatterns: 45,
        averagePatternStrength: 0.72,
        dominantLayers: [12, 18, 24],
      },
      feature_extractions: {
        totalFeatures: 512,
        identifiedFeatures: 389,
        confidence: 0.85,
        featureTypes: ['semantic', 'syntactic', 'style'],
      },
      layer_dynamics: {
        propagationSpeed: 0.65,
        informationLoss: 0.12,
        criticalLayers: [8, 16, 24, 30],
        dynamicsPattern: 'stable',
      },
      neuron_groups: {
        totalGroups: 12,
        averageGroupSize: 170,
        groupStability: 0.91,
        functionalRoles: ['encoding', 'routing', 'output'],
      },
      data_foundation: {
        totalSamples: 10000,
        labeledSamples: 8500,
        dataQuality: 0.92,
        categories: 15,
      },
    };
    return mockData[dimensionId] || {};
  };

  // 处理概念搜索
  const handleConceptSearch = async () => {
    if (!searchTerm.trim()) return;

    try {
      const response = await axios.post(`${TRAINING_API_BASE}/api/dnn/concept-analysis`, {
        concept: searchTerm,
        model: selectedModel,
      });

      if (onDataSelect) {
        onDataSelect({
          type: 'concept',
          data: response.data,
        });
      }
    } catch (error) {
      console.error('概念分析失败:', error);
      // 使用模拟数据
      if (onDataSelect) {
        onDataSelect({
          type: 'concept',
          data: {
            concept: searchTerm,
            model: selectedModel,
            neurons: Math.floor(Math.random() * 200) + 50,
            layers: Math.floor(Math.random() * 10) + 5,
            confidence: Math.random() * 0.3 + 0.7,
          },
        });
      }
    }
  };

  return (
    <div className="dnn-analysis-control-panel">
      {/* 模型和层范围选择 */}
      <div className="panel-section model-selection">
        <div className="section-header">
          <Brain size={18} className="section-icon" />
          <span className="section-title">模型配置</span>
        </div>

        <div className="model-config">
          <div className="config-row">
            <label className="config-label">选择模型:</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="model-select"
            >
              {modelOptions.map(model => (
                <option key={model.id} value={model.id}>
                  {model.name} ({model.size})
                </option>
              ))}
            </select>
          </div>

          <div className="config-row">
            <label className="config-label">层范围:</label>
            <div className="layer-range-inputs">
              <input
                type="number"
                value={selectedLayerRange.start}
                onChange={(e) => setSelectedLayerRange({
                  ...selectedLayerRange,
                  start: parseInt(e.target.value)
                })}
                className="layer-input"
                min="0"
              />
              <span className="layer-separator">-</span>
              <input
                type="number"
                value={selectedLayerRange.end}
                onChange={(e) => setSelectedLayerRange({
                  ...selectedLayerRange,
                  end: parseInt(e.target.value)
                })}
                className="layer-input"
                min="0"
              />
            </div>
          </div>
        </div>
      </div>

      {/* 概念搜索 */}
      <div className="panel-section concept-search">
        <div className="section-header">
          <Search size={18} className="section-icon" />
          <span className="section-title">概念搜索</span>
        </div>

        <div className="search-container">
          <div className="search-input-wrapper">
            <input
              type="text"
              className="concept-search-input"
              placeholder="输入概念（如：apple、dog）..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <button
            onClick={handleConceptSearch}
            className="search-button"
            disabled={!searchTerm.trim()}
          >
            <Search size={16} />
            分析
          </button>
        </div>

        {/* 快速概念选择 */}
        <div className="quick-concepts">
          {quickConcepts.map(concept => (
            <button
              key={concept.id}
              className="quick-concept-chip"
              onClick={() => {
                setSearchTerm(concept.id);
                handleConceptSearch();
              }}
            >
              <span className="concept-emoji">{concept.emoji}</span>
              <span className="concept-name">{concept.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* DNN分析维度 */}
      <div className="panel-section analysis-dimensions">
        <div className="section-header">
          <BarChart3 size={18} className="section-icon" />
          <span className="section-title">DNN分析维度</span>
        </div>

        <div className="dimensions-grid">
          {analysisDimensions.map(dimension => (
            <div
              key={dimension.id}
              className="dimension-card"
              style={{
                borderColor: analysisResults?.dimension === dimension.id
                  ? 'rgba(79, 172, 254, 0.6)'
                  : 'rgba(255, 255, 255, 0.08)',
                background: analysisResults?.dimension === dimension.id
                  ? 'rgba(79, 172, 254, 0.15)'
                  : 'rgba(255, 255, 255, 0.04)',
              }}
            >
              <div className="dimension-header">
                <dimension.icon size={20} className="dimension-icon" />
                <div className="dimension-info">
                  <div className="dimension-name">{dimension.name}</div>
                  <div className="dimension-desc">{dimension.description}</div>
                </div>
              </div>

              <div className="dimension-actions">
                <button
                  onClick={() => runDNNAnalysis(dimension)}
                  disabled={isAnalyzing}
                  className="analysis-button"
                >
                  {isAnalyzing && analysisResults?.dimension === dimension.id ? (
                    <>
                      <div className="spinner-small"></div>
                      分析中...
                    </>
                  ) : (
                    <>
                      <Play size={14} />
                      运行分析
                    </>
                  )}
                </button>
              </div>

              {analysisResults?.dimension === dimension.id && analysisResults.data && (
                <div className="analysis-result" style={{ marginTop: '16px' }}>
                  <div className="result-header" style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '8px',
                    marginBottom: '12px',
                    padding: '8px 12px',
                    background: 'rgba(79, 172, 254, 0.1)',
                    borderRadius: '6px',
                    border: '1px solid rgba(79, 172, 254, 0.2)'
                  }}>
                    <TrendingUp size={14} color="#4facfe" />
                    <span style={{ color: '#4facfe', fontWeight: 600, fontSize: '12px' }}>分析结果</span>
                    <button
                      onClick={() => setShow3DVisualization(!show3DVisualization)}
                      style={{
                        marginLeft: 'auto',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        background: show3DVisualization 
                          ? 'rgba(79, 172, 254, 0.2)' 
                          : 'rgba(255, 255, 255, 0.1)',
                        border: show3DVisualization 
                          ? '1px solid rgba(79, 172, 254, 0.4)' 
                          : '1px solid rgba(255, 255, 255, 0.2)',
                        borderRadius: '4px',
                        padding: '6px 12px',
                        color: '#4facfe',
                        cursor: 'pointer',
                        fontSize: '11px',
                        fontWeight: 600,
                        transition: 'all 0.2s ease',
                      }}
                      onMouseEnter={(e) => {
                        e.target.style.background = 'rgba(79, 172, 254, 0.3)';
                        e.target.style.borderColor = 'rgba(79, 172, 254, 0.5)';
                      }}
                      onMouseLeave={(e) => {
                        e.target.style.background = show3DVisualization 
                          ? 'rgba(79, 172, 254, 0.2)' 
                          : 'rgba(255, 255, 255, 0.1)';
                        e.target.style.borderColor = show3DVisualization 
                          ? '1px solid rgba(79, 172, 254, 0.4)' 
                          : '1px solid rgba(255, 255, 255, 0.2)';
                      }}
                    >
                      <Box size={14} />
                      {show3DVisualization ? '切换到2D' : '切换到3D'}
                    </button>
                  </div>
                  
                  {show3DVisualization ? (
                    <div className="result-3d-container" style={{ 
                      marginTop: '12px', 
                      minHeight: '350px',
                      height: '400px',
                      border: '1px solid rgba(79, 172, 254, 0.3)',
                      borderRadius: '8px',
                      overflow: 'hidden',
                      background: 'rgba(7, 12, 25, 0.9)'
                    }}>
                      <DNNAnalysis3DVisualization
                        dimension={dimension.id}
                        data={analysisResults.data}
                        onNodeClick={(label, activation) => console.log('节点点击:', label, activation)}
                      />
                    </div>
                  ) : (
                    <div className="result-content" style={{
                      background: 'rgba(255, 255, 255, 0.03)',
                      borderRadius: '6px',
                      padding: '12px'
                    }}>
                      {Object.entries(analysisResults.data).map(([key, value]) => (
                        <div key={key} className="result-item" style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          padding: '6px 0',
                          borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                          fontSize: '11px'
                        }}>
                          <span className="result-label" style={{ color: '#9bb3de' }}>{key}:</span>
                          <span className="result-value" style={{ color: '#4facfe', fontWeight: 600 }}>
                            {typeof value === 'number' ? value.toFixed(2) : String(value)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* 数据源管理 */}
      <div className="panel-section data-sources">
        <div
          className="section-header clickable"
          onClick={() => setActiveSection(activeSection === 'data' ? null : 'data')}
        >
          <Database size={18} className="section-icon" />
          <span className="section-title">数据源管理</span>
          {activeSection === 'data' ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>

        {activeSection === 'data' && (
          <div className="data-sources-content">
            {dataSources.length > 0 ? (
              dataSources.map(source => (
                <div key={source.id} className="data-source-item">
                  <div className="source-name">{source.name}</div>
                  <div className="source-info">
                    <span className="source-count">{source.count} 样本</span>
                    <span className="source-quality">质量: {source.quality || 'N/A'}</span>
                  </div>
                </div>
              ))
            ) : (
              <div className="empty-state">
                <Database size={24} className="empty-icon" />
                <p>暂无数据源</p>
                <span className="empty-hint">添加数据源以开始分析</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default DNNAnalysisControlPanel;
