import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Search, Layers, Brain, Zap, TrendingUp, ChevronDown, ChevronUp } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

/**
 * 统一数据探索面板
 * 整合所有数据探索功能到一个清晰的界面中
 */
const UnifiedDataExplorer = ({ onSearch, onVisualizationModeChange }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [dataSource, setDataSource] = useState(null);
  const [categories, setCategories] = useState({});
  const [activeSection, setActiveSection] = useState('search');
  const [loading, setLoading] = useState(false);
  const [dataSources, setDataSources] = useState({});

  // 可视化模式
  const visualizationModes = [
    { id: '2d', name: '2D图表', icon: TrendingUp, description: '查看传统图表分析' },
    { id: '3d', name: '3D可视化', icon: Layers, description: '查看多层3D关联' },
    { id: 'neuron', name: '神经元', icon: Brain, description: '查看神经元激活' },
    { id: 'flow', name: '数据流', icon: Zap, description: '查看数据流向' },
  ];

  // 常用概念
  const quickConcepts = [
    { id: 'apple', name: '苹果', emoji: '🍎' },
    { id: 'banana', name: '香蕉', emoji: '🍌' },
    { id: 'orange', name: '橙子', emoji: '🍊' },
    { id: 'dog', name: '狗', emoji: '🐕' },
    { id: 'cat', name: '猫', emoji: '🐱' },
  ];

  // 加载数据源和分类
  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const [sourcesRes, categoriesRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/api/data-sources`),
        axios.get(`${API_BASE_URL}/api/data-puzzle-categories`),
      ]);
      setDataSources(sourcesRes.data);
      setCategories(categoriesRes.data);
      setLoading(false);
    } catch (error) {
      console.error('加载数据失败:', error);
      setLoading(false);
    }
  };

  // 智能搜索处理
  const handleSmartSearch = (e) => {
    e.preventDefault();
    if (searchTerm.trim()) {
      // 智能识别输入类型并自动切换模式
      const term = searchTerm.trim().toLowerCase();
      let mode = '2d';

      if (quickConcepts.some(c => c.id === term)) {
        mode = '3d';
      }

      onSearch({
        term: term,
        type: 'smart',
        mode: mode
      });

      if (onVisualizationModeChange) {
        onVisualizationModeChange(mode);
      }

      setSearchTerm('');
    }
  };

  // 快速选择概念
  const handleQuickSelect = (conceptId) => {
    handleSmartSearch({ preventDefault: () => {} });
    onSearch({
      term: conceptId,
      type: 'quick_select',
      mode: '3d'
    });

    if (onVisualizationModeChange) {
      onVisualizationModeChange('3d');
    }
  };

  // 选择数据源
  const handleDataSourceSelect = (sourceKey) => {
    setDataSource(sourceKey);
    onSearch({
      term: null,
      type: 'data_source',
      dataSource: sourceKey
    });
  };

  return (
    <div className="unified-data-explorer">
      {/* 主搜索区 */}
      <div className="explorer-main-search">
        <div className="search-container">
          <form onSubmit={handleSmartSearch} className="search-form">
            <div className="search-input-wrapper">
              <Search className="search-icon" size={20} />
              <input
                type="text"
                className="unified-search-input"
                placeholder="搜索概念、选择数据源或查看分类..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            <button type="submit" className="search-submit-btn">
              搜索
            </button>
          </form>
        </div>

        {/* 快速概念选择 */}
        <div className="quick-concepts-bar">
          {quickConcepts.map(concept => (
            <button
              key={concept.id}
              className="quick-concept-chip"
              onClick={() => handleQuickSelect(concept.id)}
            >
              <span className="concept-emoji">{concept.emoji}</span>
              <span className="concept-name">{concept.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* 内容展示区 */}
      <div className="explorer-content">
        {loading && (
          <div className="explorer-loading">
            <div className="loading-spinner"></div>
            <span>加载数据中...</span>
          </div>
        )}

        {!loading && (
          <>
            {/* 数据源卡片 */}
            <div className="explorer-card data-sources-card">
              <div
                className="card-header"
                onClick={() => setActiveSection(activeSection === 'sources' ? null : 'sources')}
              >
                <div className="card-title">
                  <Layers size={18} className="card-icon" />
                  <span>数据源</span>
                </div>
                {activeSection === 'sources' ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </div>

              {activeSection === 'sources' && (
                <div className="card-content">
                  <div className="data-sources-grid">
                    {Object.entries(dataSources).map(([key, source]) => (
                      <button
                        key={key}
                        className={`data-source-card ${dataSource === key ? 'active' : ''}`}
                        onClick={() => handleDataSourceSelect(key)}
                      >
                        <div className="source-name">{source.name}</div>
                        <div className="source-count">{source.count} 项</div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* 分类卡片 */}
            <div className="explorer-card categories-card">
              <div
                className="card-header"
                onClick={() => setActiveSection(activeSection === 'categories' ? null : 'categories')}
              >
                <div className="card-title">
                  <Brain size={18} className="card-icon" />
                  <span>分类浏览</span>
                </div>
                {activeSection === 'categories' ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </div>

              {activeSection === 'categories' && (
                <div className="card-content">
                  <div className="categories-list">
                    {Object.entries(categories).map(([key, category]) => (
                      <button
                        key={key}
                        className="category-item"
                        onClick={() => handleDataSourceSelect(key)}
                      >
                        <span className="category-name">{category.name}</span>
                        <span className="category-badge">{category.count}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* 可视化模式卡片 */}
            <div className="explorer-card modes-card">
              <div
                className="card-header"
                onClick={() => setActiveSection(activeSection === 'modes' ? null : 'modes')}
              >
                <div className="card-title">
                  <Zap size={18} className="card-icon" />
                  <span>可视化模式</span>
                </div>
                {activeSection === 'modes' ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </div>

              {activeSection === 'modes' && (
                <div className="card-content">
                  <div className="modes-grid">
                    {visualizationModes.map(mode => (
                      <button
                        key={mode.id}
                        className="mode-card"
                        onClick={() => onVisualizationModeChange && onVisualizationModeChange(mode.id)}
                      >
                        <mode.icon size={24} className="mode-icon" />
                        <div className="mode-info">
                          <div className="mode-name">{mode.name}</div>
                          <div className="mode-desc">{mode.description}</div>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default UnifiedDataExplorer;
