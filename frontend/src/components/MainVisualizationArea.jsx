import { TRAINING_API_BASE } from '../config.js';
import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';
import LCSVisualization from './LCSVisualization';



const MainVisualizationArea = ({
  searchQuery,
  visualizationMode: externalVisualizationMode
}) => {
  const [visualizationData, setVisualizationData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [chartType, setChartType] = useState('heatmap');
  const [viewMode, setViewMode] = useState(externalVisualizationMode || '2d'); // '2d' 或 '3d'
  const [selectedConcept, setSelectedConcept] = useState(null);

  // 同步外部传入的可视化模式
  useEffect(() => {
    if (externalVisualizationMode) {
      setViewMode(externalVisualizationMode);
    }
  }, [externalVisualizationMode]);

  useEffect(() => {
    if (viewMode === '3d' && selectedConcept) {
      // 3D模式不需要加载数据
      return;
    }
    if (searchQuery && searchQuery.term) {
      handleSearch(searchQuery);
    }
  }, [searchQuery, viewMode, selectedConcept]);

  const loadVisualizationData = async () => {
    setLoading(true);
    setError(null);
    try {
      let endpoint;
      let payload = {};

      // 根据选择的类别加载不同的可视化数据
      if (selectedCategory === 'shared_bearing') {
        endpoint = `${TRAINING_API_BASE}/api/visualization/shared-bearing/heatmap`;
        payload = {
          family_type: selectedSubcategory || 'cross_family',
          model: 'deepseek7b'
        };
      } else if (selectedCategory === 'cross_model_validation') {
        endpoint = `${TRAINING_API_BASE}/api/visualization/cross-model/comparison`;
        payload = {
          concept_ids: ['apple', 'banana', 'orange', 'grape', 'pear'],
          models: ['gpt2', 'qwen3', 'deepseek7b']
        };
      } else if (selectedCategory === 'bias_deflection') {
        endpoint = `${TRAINING_API_BASE}/api/visualization/shared-bearing/scatter`;
        payload = {
          family_type: 'cross_family',
          model: 'deepseek7b'
        };
      }

      if (endpoint) {
        const response = await axios.post(endpoint, payload);
        setVisualizationData(response.data);
      }
    } catch (error) {
      console.error('加载可视化数据失败:', error);
      setError('加载数据失败: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const generateDemoData = async () => {
    setLoading(true);
    try {
      await axios.post(`${TRAINING_API_BASE}/api/visualization/demo-data`);
      alert('示例数据已生成！');
      loadVisualizationData();
    } catch (error) {
      console.error('生成示例数据失败:', error);
      setError('生成示例数据失败: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (query) => {
    setLoading(true);
    setError(null);
    try {
      // 如果是快速选择概念，直接设置概念并切换到3D模式
      if (query.type === 'quick_select' || query.mode === '3d') {
        setSelectedConcept(query.term);
        setViewMode('3d');
        return;
      }

      // 否则执行搜索
      const endpoint = `${TRAINING_API_BASE}/api/search`;
      const response = await axios.post(endpoint, {
        term: query.term,
        type: query.type || 'all',
        sortBy: query.sortBy || 'time'
      });
      setVisualizationData(response.data);
    } catch (error) {
      console.error('搜索失败:', error);
      setError('搜索失败: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const getChartTitle = () => {
    if (viewMode === '3d') {
      return `${selectedConcept || '概念'} - 多层3D关联可视化`;
    }
    if (visualizationData && visualizationData.title) {
      return visualizationData.title;
    }
    if (searchQuery && searchQuery.term) {
      return `搜索: ${searchQuery.term}`;
    }
    if (selectedCategory) {
      const categoryNames = {
        shared_bearing: '共享承载机制',
        bias_deflection: '偏置偏转机制',
        layerwise_amplification: '逐层放大机制',
        multi_space_mapping: '多空间映射',
        cross_model_validation: '跨模型验证'
      };
      return categoryNames[selectedCategory] || '数据可视化';
    }
    return '数据可视化';
  };

  return (
    <div className="main-visualization-area">
      <div className="visualization-header">
        <h2>{getChartTitle()}</h2>
        <div className="visualization-controls">
          <button
            className="btn btn-secondary"
            onClick={() => setViewMode(viewMode === '2d' ? '3d' : '2d')}
            disabled={loading}
          >
            {viewMode === '2d' ? '切换到LCS 3D架构' : '切换到2D'}
          </button>
          {viewMode === '2d' && (
            <>
              <button
                className="btn btn-secondary"
                onClick={generateDemoData}
                disabled={loading}
              >
                生成示例数据
              </button>
              <button
                className="btn btn-primary"
                onClick={loadVisualizationData}
                disabled={loading}
              >
                刷新数据
              </button>
            </>
          )}
          {viewMode === '3d' && (
            <span style={{ marginLeft: '15px', color: '#00d2ff', fontWeight: 'bold' }}>
              AGI GLM5 语言数学原理 (v17.0)
            </span>
          )}
        </div>
      </div>

      {loading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <div className="loading-text">加载中...</div>
        </div>
      )}

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {viewMode === '3d' && !loading && !error && (
        <div className="visualization-content" style={{ height: '800px' }}>
          <LCSVisualization />
        </div>
      )}

      {viewMode === '2d' && visualizationData && !loading && !error && (
        <div className="visualization-content">
          <Plot
            data={visualizationData.data}
            layout={{
              ...visualizationData.layout,
              autosize: true,
              margin: { l: 60, r: 20, t: 40, b: 60 }
            }}
            useResizeHandler={true}
            style={{ width: '100%', height: '100%' }}
            config={{
              responsive: true,
              displayModeBar: true,
              displaylogo: false,
              modeBarButtonsToRemove: ['sendDataToCloud']
            }}
          />
        </div>
      )}

      {!visualizationData && viewMode === '2d' && !loading && !error && (
        <div className="empty-state">
          <div className="empty-icon">📊</div>
          <h3>暂无数据</h3>
          <p>请选择数据源或数据拼图类别</p>
          <button
            className="btn btn-primary"
            onClick={generateDemoData}
          >
            生成示例数据
          </button>
        </div>
      )}

      {viewMode === '3d' && loading && !error && (
        <div className="empty-state">
          <div className="empty-icon">🎨</div>
          <h3>LCS 双层几何分离架构加载中...</h3>
        </div>
      )}
    </div>
  );
};

export default MainVisualizationArea;
