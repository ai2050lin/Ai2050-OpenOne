import { useRef, useEffect, useState } from 'react';
import { Play, Pause, Square, Trash2, RotateCcw, Search, Target, Zap, CheckCircle, Settings, Brain } from 'lucide-react';
import { RESEARCH_PHASES } from './aiRnDConfig';

export const AIRnDConsoleTab = ({
  sessionStatus,
  currentPhase,
  round,
  logs,
  onStart,
  onPause,
  onStop,
  onClear,
}) => {
  const logEndRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);

  useEffect(() => {
    if (autoScroll && logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  const phaseInfo = RESEARCH_PHASES.find(p => p.id === currentPhase);

  const getLogColor = (type) => {
    switch (type) {
      case 'error': return '#ff4444';
      case 'phase_change': return '#ffaa00';
      case 'round_change': return '#4488ff';
      case 'finding': return '#00ff88';
      case 'code_generated': return '#a78bfa';
      case 'execution_result': return '#00d2ff';
      case 'analysis': return '#88aaff';
      case 'planning': return '#ffaa00';
      case 'generation': return '#00ff88';
      case 'summary': return '#aa44ff';
      default: return '#999';
    }
  };

  const formatLogMessage = (log) => {
    if (log.message) return log.message;
    if (log.type === 'phase_change') return `阶段切换 → ${log.phase}`;
    if (log.type === 'round_change') return `进入第 ${log.round} 轮`;
    if (log.type === 'finding') return `发现: ${log.finding?.title || log.finding?.content?.slice(0, 80) || ''}`;
    if (log.type === 'code_generated') return `代码已生成 (${log.code?.length || 0} 字符)`;
    if (log.type === 'execution_result') return `执行完成: ${log.result?.status || ''}`;
    if (log.type === 'error') return `错误: ${log.message || log.detail || ''}`;
    return JSON.stringify(log).slice(0, 100);
  };
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Status Panel */}
      <div style={{
        padding: '24px 40px', borderBottom: '1px solid rgba(167, 139, 250, 0.15)',
        background: 'rgba(10, 10, 16, 0.45)', backdropFilter: 'blur(10px)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 28 }}>
          {/* Round indicator */}
          <div style={{
            width: 80, height: 80, borderRadius: '50%',
            border: `3px double ${phaseInfo?.color || '#3b82f6'}`,
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            background: 'rgba(0,0,0,0.4)',
            boxShadow: sessionStatus === 'running' 
              ? `0 0 24px ${phaseInfo?.color || '#3b82f6'}40, inset 0 0 12px ${phaseInfo?.color || '#3b82f6'}20` 
              : 'none',
            transition: 'all 0.5s ease',
            animation: sessionStatus === 'running' ? 'pulse 2.5s infinite' : 'none',
          }}>
            <span style={{ fontSize: '26px', fontWeight: '900', color: phaseInfo?.color || '#ef4444', fontFamily: 'monospace' }}>{round}</span>
            <span style={{ fontSize: '9px', color: '#9ca3af', fontWeight: 'bold', letterSpacing: '1px' }}>ROUND</span>
          </div>

          {/* Phase info */}
          <div>
            <div style={{ fontSize: '13px', color: '#9ca3af', marginBottom: 6, fontWeight: 'bold' }}>当前研发状态</div>
            <div style={{ 
              fontSize: '20px', fontWeight: '900', 
              color: phaseInfo?.color || '#ef4444',
              display: 'flex', alignItems: 'center', gap: 8,
              textShadow: phaseInfo?.color ? `0 0 10px ${phaseInfo.color}30` : 'none',
            }}>
              {(() => {
                const PHASE_ICONS = {
                  analyze: Search,
                  plan: Target,
                  generate: Zap,
                  execute: Play,
                  summarize: CheckCircle,
                };
                const IconComponent = phaseInfo ? PHASE_ICONS[phaseInfo.id] : Settings;
                return IconComponent ? <IconComponent size={20} style={{ filter: `drop-shadow(0 0 4px ${phaseInfo?.color || '#ef4444'})` }} /> : null;
              })()}
              <span>{phaseInfo ? phaseInfo.label : '引擎待启动'}</span>
            </div>
          </div>

          {/* Quick stats */}
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 16 }}>
            <StatBox label="日志数" value={logs.length} color="#6366f1" />
            <StatBox 
              label="状态" 
              value={sessionStatus === 'running' ? '运行中' : sessionStatus === 'paused' ? '已暂停' : sessionStatus === 'stopped' ? '已停止' : '就绪'} 
              color={sessionStatus === 'running' ? '#00ff88' : sessionStatus === 'paused' ? '#ffaa00' : '#ef4444'}
            />
          </div>
        </div>
      </div>

      {/* Terminal Title Bar */}
      <div style={{
        background: 'rgba(15, 15, 25, 0.45)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
        padding: '8px 20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        fontSize: '11px',
        color: '#6b7280',
        fontFamily: 'monospace',
      }}>
        <div style={{ display: 'flex', gap: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#ef4444' }} />
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#f59e0b' }} />
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#10b981' }} />
        </div>
        <div style={{ letterSpacing: '1px' }}>RESEARCH_ENGINE_STDOUT_LOGGER // v1.2.0</div>
        <div>UTF-8</div>
      </div>

      {/* Log stream */}
      <div 
        className="airnd-scroll"
        style={{
          flex: 1, overflowY: 'auto', padding: '20px 40px',
          fontFamily: "'Cascadia Code', 'Fira Code', 'Consolas', monospace",
          fontSize: '12px', lineHeight: '1.7',
          background: 'rgba(10, 10, 16, 0.2)',
        }}
        onScroll={(e) => {
          const { scrollTop, scrollHeight, clientHeight } = e.target;
          setAutoScroll(scrollHeight - scrollTop - clientHeight < 100);
        }}
      >
        {logs.length === 0 && (
          <div style={{ 
            maxWidth: '520px', margin: '60px auto', textAlign: 'center',
            padding: '40px 30px', 
            borderRadius: '24px',
            border: '1px solid rgba(167, 139, 250, 0.28)',
            background: 'linear-gradient(135deg, rgba(167, 139, 250, 0.10) 0%, rgba(167, 139, 250, 0.03) 100%)',
            animation: 'roadmapFade 0.6s ease-out'
          }}>
            <div style={{ marginBottom: 16, display: 'flex', justifyContent: 'center' }}>
              <Brain size={48} color="#a78bfa" style={{ filter: 'drop-shadow(0 0 12px rgba(167, 139, 250, 0.6))' }} />
            </div>
            <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#a78bfa', marginBottom: 10 }}>AI 自动研发引擎就绪</div>
            <div style={{ fontSize: '13px', color: '#9ca3af', lineHeight: '1.6' }}>
              引擎已处于待命状态。请点击右上方 <span style={{ color: '#00ff88', fontWeight: 'bold' }}>▶</span> 按钮，启动神经网络逆向工程的研究循环。
            </div>
          </div>
        )}

        {logs.map((log, i) => (
          <div key={log.id || i} style={{
            padding: '5px 0', display: 'flex', gap: 12,
            borderBottom: '1px solid rgba(255,255,255,0.02)',
            alignItems: 'flex-start',
          }}>
            <span style={{ color: '#4b5563', minWidth: 70, userSelect: 'none' }}>{log.timestamp || ''}</span>
            <span style={{ color: getLogColor(log.type), minWidth: 16, textAlign: 'center', userSelect: 'none' }}>
              {log.type === 'error' ? '✗' : log.type === 'finding' ? '★' : '●'}
            </span>
            <span style={{ color: getLogColor(log.type), whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
              {formatLogMessage(log)}
            </span>
          </div>
        ))}
        <div ref={logEndRef} />
      </div>

      {/* Bottom bar */}
      <div style={{
        padding: '12px 40px', borderTop: '1px solid rgba(255,255,255,0.06)',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        background: 'rgba(15, 15, 25, 0.3)',
      }}>
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={onClear} style={smallBtnStyle} title="清空日志">
            <Trash2 size={12} style={{ filter: 'drop-shadow(0 0 3px rgba(156, 163, 175, 0.6))' }} /> 清空
          </button>
        </div>
        <span style={{ fontSize: '11px', color: '#4b5563', fontFamily: 'monospace' }}>
          {autoScroll ? 'AUTO SCROLL: ON' : 'AUTO SCROLL: OFF'} | {logs.length} LINES
        </span>
      </div>
    </div>
  );
};

const StatBox = ({ label, value, color }) => (
  <div 
    className="airnd-card"
    style={{
      padding: '10px 20px', background: 'rgba(0,0,0,0.25)',
      borderRadius: '12px', border: '1px solid rgba(255,255,255,0.06)',
      textAlign: 'center',
      position: 'relative',
      overflow: 'hidden',
      minWidth: '110px',
    }}
  >
    <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '2px', background: color }} />
    <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#f3f4f6', fontFamily: 'monospace' }}>{value}</div>
    <div style={{ fontSize: '11px', color: '#9ca3af', marginTop: '4px' }}>{label}</div>
  </div>
);

const smallBtnStyle = {
  background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)',
  color: '#9ca3af', borderRadius: '6px', padding: '6px 12px', cursor: 'pointer',
  fontSize: '11px', display: 'flex', alignItems: 'center', gap: 6,
  transition: 'all 0.2s',
};
