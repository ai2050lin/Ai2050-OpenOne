import React, { useState } from 'react';
import UnifiedDataExplorer from './UnifiedDataExplorer';
import MainVisualizationArea from './MainVisualizationArea';
import StatusBar from './StatusBar';

/**
 * AGI研究数据可视化主仪表板
 * 使用统一数据探索面板，提供简洁高效的用户体验
 */
const AGIVisualizationDashboard = () => {
  const [searchQuery, setSearchQuery] = useState(null);
  const [visualizationMode, setVisualizationMode] = useState('2d');

  const handleSearch = (query) => {
    console.log('搜索请求:', query);
    setSearchQuery(query);
  };

  const handleVisualizationModeChange = (mode) => {
    console.log('切换可视化模式:', mode);
    setVisualizationMode(mode);
  };

  return (
    <div className="agi-dashboard">
      {/* 顶部标题栏 */}
      <div className="dashboard-header">
        <h1>AGI研究数据可视化</h1>
        <div className="dashboard-meta">
          <span className="meta-item">智能探索</span>
          <span className="meta-separator">|</span>
          <span className="meta-item">数据驱动研究</span>
        </div>
      </div>

      {/* 主内容区域 */}
      <div className="dashboard-content">
        {/* 左侧统一探索面板 */}
        <div className="dashboard-sidebar">
          <UnifiedDataExplorer
            onSearch={handleSearch}
            onVisualizationModeChange={handleVisualizationModeChange}
          />
        </div>

        {/* 右侧主可视化区域 */}
        <div className="dashboard-main">
          <MainVisualizationArea
            searchQuery={searchQuery}
            visualizationMode={visualizationMode}
          />
        </div>
      </div>

      {/* 底部状态栏 */}
      <StatusBar />
    </div>
  );
};

export default AGIVisualizationDashboard;
