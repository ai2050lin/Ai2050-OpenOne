import React from 'react';
import AGIVisualizationDashboard from './components/AGIVisualizationDashboard';
import './css/AGIDashboard.css';
import './css/UnifiedDataExplorer.css';

function AGIVisualizationApp() {
  return (
    <div className="agi-app">
      <AGIVisualizationDashboard />
    </div>
  );
}

export default AGIVisualizationApp;
