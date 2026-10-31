import React, { useState, useEffect } from 'react';

const StatusBar = () => {
  const [apiStatus, setApiStatus] = useState('unknown');
  const [dataLoaded, setDataLoaded] = useState(false);

  useEffect(() => {
    checkApiStatus();
    const interval = setInterval(checkApiStatus, 30000); // 每30秒检查一次
    return () => clearInterval(interval);
  }, []);

  const checkApiStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/health');
      if (response.ok) {
        setApiStatus('online');
      } else {
        setApiStatus('offline');
      }
    } catch (error) {
      setApiStatus('offline');
    }
  };

  return (
    <div className="status-bar">
      <div className="status-item">
        <span className="status-label">API状态:</span>
        <span className={`status-value ${apiStatus}`}>
          {apiStatus === 'online' ? '🟢 在线' : '🔴 离线'}
        </span>
      </div>
      <div className="status-item">
        <span className="status-label">数据加载:</span>
        <span className={`status-value ${dataLoaded ? 'loaded' : 'pending'}`}>
          {dataLoaded ? '✅ 已加载' : '⏳ 待加载'}
        </span>
      </div>
      <div className="status-item">
        <span className="status-label">内存使用:</span>
        <span className="status-value">正常</span>
      </div>
      <div className="status-item">
        <span className="status-label">系统状态:</span>
        <span className="status-value">运行中</span>
      </div>
    </div>
  );
};

export default StatusBar;
