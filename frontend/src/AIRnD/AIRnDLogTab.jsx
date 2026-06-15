import { useState, useRef, useEffect } from 'react';
import { Filter, Search } from 'lucide-react';

export const AIRnDLogTab = ({ logs }) => {
  const [filter, setFilter] = useState('all');
  const [searchText, setSearchText] = useState('');
  const logEndRef = useRef(null);

  const filteredLogs = logs.filter(log => {
    if (filter !== 'all' && log.type !== filter) return false;
    if (searchText) {
      const text = JSON.stringify(log).toLowerCase();
      if (!text.includes(searchText.toLowerCase())) return false;
    }
    return true;
  });

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [filteredLogs.length]);

  const getLogColor = (type) => {
    const colors = {
      error: '#ff4444', phase_change: '#ffaa00', round_change: '#4488ff',
      finding: '#00ff88', code_generated: '#a78bfa', execution_result: '#00d2ff',
      analysis: '#88aaff', planning: '#ffaa00', generation: '#00ff88', summary: '#aa44ff',
    };
    return colors[type] || '#999';
  };

  const filterOptions = [
    { id: 'all', label: '全部' },
    { id: 'analysis', label: '分析' },
    { id: 'planning', label: '规划' },
    { id: 'generation', label: '生成' },
    { id: 'execution_result', label: '执行' },
    { id: 'finding', label: '发现' },
    { id: 'error', label: '错误' },
  ];

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Filter bar */}
      <div style={{
        padding: '16px 40px', borderBottom: '1px solid rgba(167, 139, 250, 0.15)',
        display: 'flex', gap: 10, alignItems: 'center', background: 'rgba(10, 10, 16, 0.45)', backdropFilter: 'blur(10px)',
      }}>
        <Filter size={14} color="#a78bfa" style={{ filter: 'drop-shadow(0 0 3px #a78bfa)' }} />
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {filterOptions.map(f => {
            const isActive = filter === f.id;
            return (
              <button
                key={f.id}
                onClick={() => setFilter(f.id)}
                style={{
                  padding: '5px 14px', borderRadius: '14px', fontSize: '11px', cursor: 'pointer',
                  background: isActive 
                    ? 'linear-gradient(135deg, rgba(167, 139, 250, 0.18) 0%, rgba(99, 102, 241, 0.06) 100%)' 
                    : 'rgba(255,255,255,0.03)',
                  border: `1px solid ${isActive ? 'rgba(167, 139, 250, 0.35)' : 'rgba(255,255,255,0.06)'}`,
                  color: isActive ? '#a78bfa' : '#9ca3af', 
                  transition: 'all 0.2s ease',
                  fontWeight: isActive ? 'bold' : 'normal',
                  boxShadow: isActive ? '0 0 8px rgba(167, 139, 250, 0.15)' : 'none',
                }}
                onMouseEnter={e => {
                  if (!isActive) {
                    e.currentTarget.style.color = '#c084fc';
                    e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                  }
                }}
                onMouseLeave={e => {
                  if (!isActive) {
                    e.currentTarget.style.color = '#9ca3af';
                    e.currentTarget.style.background = 'rgba(255,255,255,0.03)';
                  }
                }}
              >
                {f.label}
              </button>
            );
          })}
        </div>

        <div style={{ marginLeft: 'auto', position: 'relative' }}>
          <Search size={13} style={{ position: 'absolute', left: 10, top: 11, color: '#a78bfa', filter: 'drop-shadow(0 0 3px #a78bfa)' }} />
          <input
            value={searchText}
            onChange={e => setSearchText(e.target.value)}
            placeholder="搜索日志..."
            className="airnd-input"
            style={{
              padding: '8px 14px 8px 30px', background: 'rgba(0,0,0,0.3)',
              border: '1px solid rgba(255,255,255,0.08)', borderRadius: '8px',
              color: '#f3f4f6', fontSize: '12px', width: 220, outline: 'none',
            }}
          />
        </div>
      </div>

      {/* Log list */}
      <div 
        className="airnd-scroll"
        style={{
          flex: 1, overflowY: 'auto', padding: '24px 40px',
          fontFamily: "'Cascadia Code', 'Fira Code', 'Consolas', monospace",
          fontSize: '12px', lineHeight: '1.75',
          background: 'rgba(10, 10, 16, 0.1)',
        }}
      >
        {filteredLogs.length === 0 ? (
          <div style={{ 
            maxWidth: '400px', margin: '80px auto', textAlign: 'center',
            padding: '30px', borderRadius: '16px',
            border: '1px dashed rgba(255,255,255,0.08)', color: '#6b7280' 
          }}>
            暂无匹配的日志数据
          </div>
        ) : (
          filteredLogs.map((log, i) => (
            <div key={log.id || i} style={{
              padding: '8px 16px', marginBottom: 6, borderRadius: '8px',
              background: log.type === 'error' 
                ? 'linear-gradient(90deg, rgba(239, 68, 68, 0.08), rgba(239, 68, 68, 0.01))' 
                : 'rgba(255,255,255,0.01)',
              border: '1px solid rgba(255,255,255,0.02)',
              borderLeft: `3px solid ${getLogColor(log.type)}`,
              boxShadow: log.type === 'error' ? '0 0 10px rgba(239, 68, 68, 0.05)' : 'none',
            }}>
              <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: '4px' }}>
                <span style={{ color: '#4b5563', minWidth: 70, fontSize: '11px', userSelect: 'none' }}>{log.timestamp || ''}</span>
                <span style={{ 
                  color: getLogColor(log.type), 
                  fontWeight: 'bold', 
                  fontSize: '10px',
                  background: `${getLogColor(log.type)}15`,
                  padding: '1px 6px',
                  borderRadius: '4px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  {log.type}
                </span>
              </div>
              <div style={{ color: '#d1d5db', paddingLeft: 4, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                {log.content || log.message || JSON.stringify(log, null, 2)}
              </div>
            </div>
          ))
        )}
        <div ref={logEndRef} />
      </div>
    </div>
  );
};
