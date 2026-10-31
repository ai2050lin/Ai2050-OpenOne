/**
 * 操作历史组件
 * 记录和展示用户的分析操作历史
 */
import React, { useState, useEffect, useCallback } from 'react';
import { HISTORY_CONFIG, COLORS } from '../../config/panels';
import { Clock, RotateCcw, Trash2, ChevronDown, ChevronUp } from 'lucide-react';

// 历史记录类型
export const HistoryTypes = {
  ANALYSIS: 'analysis',
  GENERATION: 'generation',
  SELECTION: 'selection',
  CONFIG_CHANGE: 'config',
};

// 创建历史记录
export function createHistoryItem(type, action, details = {}) {
  return {
    id: Date.now(),
    type,
    action,
    details,
    timestamp: new Date().toISOString(),
  };
}

// 历史记录管理 Hook
export function useOperationHistory(maxItems = HISTORY_CONFIG.maxItems) {
  const [history, setHistory] = useState([]);
  
  // 从 localStorage 加载
  useEffect(() => {
    try {
      const saved = localStorage.getItem(HISTORY_CONFIG.storageKey);
      if (saved) {
        setHistory(JSON.parse(saved));
      }
    } catch (e) {
      console.warn('Failed to load history:', e);
    }
  }, []);
  
  // 保存到 localStorage
  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_CONFIG.storageKey, JSON.stringify(history));
    } catch (e) {
      console.warn('Failed to save history:', e);
    }
  }, [history]);
  
  // 添加记录
  const addHistory = useCallback((item) => {
    setHistory(prev => {
      const newHistory = [item, ...prev].slice(0, maxItems);
      return newHistory;
    });
  }, [maxItems]);
  
  // 清除历史
  const clearHistory = useCallback(() => {
    setHistory([]);
  }, []);
  
  // 恢复到某条记录
  const restoreHistory = useCallback((item) => {
    return item.details;
  }, []);
  
  return {
    history,
    addHistory,
    clearHistory,
    restoreHistory,
  };
}

// 时间格式化
function formatTime(isoString) {
  const date = new Date(isoString);
  const now = new Date();
  const diff = now - date;
  
  if (diff < 60000) return '刚刚';
  if (diff < 3600000) return `${Math.floor(diff / 60000)}分钟前`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}小时前`;
  return date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' });
}

// 类型图标和颜色
const typeConfig = {
  [HistoryTypes.ANALYSIS]: { icon: '📊', color: COLORS.primary, label: '分析' },
  [HistoryTypes.GENERATION]: { icon: '✨', color: COLORS.success, label: '生成' },
  [HistoryTypes.SELECTION]: { icon: '🎯', color: COLORS.accent, label: '选择' },
  [HistoryTypes.CONFIG_CHANGE]: { icon: '⚙️', color: COLORS.warning, label: '配置' },
};

// 单条历史记录组件
function HistoryItem({ item, onRestore, onRemove }) {
  const config = typeConfig[item.type] || typeConfig[HistoryTypes.ANALYSIS];
  
  return (
    <div style={{
      display: 'flex',
      alignItems: 'flex-start',
      gap: '8px',
      padding: '8px',
      background: 'rgba(255,255,255,0.03)',
      borderRadius: '6px',
      marginBottom: '4px',
      borderLeft: `2px solid ${config.color}`,
      transition: 'all 0.2s',
    }}>
      <span style={{ fontSize: '14px' }}>{config.icon}</span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ 
          fontSize: '11px', 
          color: COLORS.textPrimary,
          marginBottom: '2px',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap'
        }}>
          {item.action}
        </div>
        <div style={{ fontSize: '9px', color: COLORS.textMuted }}>
          {formatTime(item.timestamp)}
          {item.details?.layer !== undefined && ` · Layer ${item.details.layer}`}
          {item.details?.mode && ` · ${item.details.mode}`}
        </div>
      </div>
      <div style={{ display: 'flex', gap: '4px' }}>
        <button
          onClick={() => onRestore?.(item)}
          title="恢复此状态"
          style={{
            background: 'transparent',
            border: 'none',
            color: COLORS.textMuted,
            cursor: 'pointer',
            padding: '2px',
            display: 'flex',
          }}
        >
          <RotateCcw size={12} />
        </button>
        <button
          onClick={() => onRemove?.(item.id)}
          title="删除此记录"
          style={{
            background: 'transparent',
            border: 'none',
            color: COLORS.textMuted,
            cursor: 'pointer',
            padding: '2px',
            display: 'flex',
          }}
        >
          <Trash2 size={12} />
        </button>
      </div>
    </div>
  );
}

// 操作历史面板组件
export function OperationHistoryPanel({ 
  history, 
  onRestore, 
  onClear,
  onRemove,
  maxVisible = 5 
}) {
  const [expanded, setExpanded] = useState(false);
  const [filter, setFilter] = useState('all');
  
  // 过滤历史记录
  const filteredHistory = history.filter(item => {
    if (filter === 'all') return true;
    return item.type === filter;
  });
  
  const visibleHistory = expanded ? filteredHistory : filteredHistory.slice(0, maxVisible);
  
  if (history.length === 0) {
    return (
      <div style={{ 
        color: COLORS.textMuted, 
        fontSize: '11px', 
        textAlign: 'center',
        padding: '12px',
        fontStyle: 'italic'
      }}>
        暂无操作历史
      </div>
    );
  }
  
  return (
    <div>
      {/* 过滤器和清除按钮 */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        marginBottom: '8px'
      }}>
        <div style={{ display: 'flex', gap: '4px' }}>
          {['all', HistoryTypes.ANALYSIS, HistoryTypes.GENERATION, HistoryTypes.SELECTION].map(type => (
            <button
              key={type}
              onClick={() => setFilter(type)}
              style={{
                padding: '2px 6px',
                fontSize: '9px',
                background: filter === type ? COLORS.primary : 'transparent',
                border: 'none',
                borderRadius: '3px',
                color: filter === type ? '#000' : COLORS.textMuted,
                cursor: 'pointer',
              }}
            >
              {type === 'all' ? '全部' : typeConfig[type]?.label || type}
            </button>
          ))}
        </div>
        <button
          onClick={onClear}
          style={{
            background: 'transparent',
            border: 'none',
            color: COLORS.danger,
            cursor: 'pointer',
            fontSize: '10px',
            padding: '2px 4px',
          }}
        >
          清除全部
        </button>
      </div>
      
      {/* 历史列表 */}
      <div style={{ maxHeight: expanded ? '200px' : 'auto', overflowY: 'auto' }}>
        {visibleHistory.map(item => (
          <HistoryItem 
            key={item.id} 
            item={item} 
            onRestore={onRestore}
            onRemove={onRemove}
          />
        ))}
      </div>
      
      {/* 展开按钮 */}
      {filteredHistory.length > maxVisible && (
        <button
          onClick={() => setExpanded(!expanded)}
          style={{
            width: '100%',
            padding: '6px',
            marginTop: '4px',
            background: 'rgba(255,255,255,0.03)',
            border: 'none',
            borderRadius: '4px',
            color: COLORS.textMuted,
            cursor: 'pointer',
            fontSize: '10px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '4px',
          }}
        >
          {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          {expanded ? '收起' : `展开更多 (${filteredHistory.length - maxVisible})`}
        </button>
      )}
    </div>
  );
}

export default OperationHistoryPanel;
