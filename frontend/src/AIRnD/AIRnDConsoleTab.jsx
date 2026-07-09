import { useRef, useEffect, useState } from 'react';
import { Play, Pause, Square, Trash2, Search, Target, Zap, CheckCircle, Brain, FileText, SkipForward } from 'lucide-react';
import { RESEARCH_PHASES } from './aiRnDConfig';

const PHASE_ICONS = {
  analyze: Search,
  plan: Target,
  generate: Zap,
  execute: Play,
  summarize: CheckCircle,
};

const SUB_TABS = [
  { id: 'live', label: '实时日志', icon: FileText },
];

export const AIRnDConsoleTab = ({
  sessionStatus,
  sessionMode,
  currentPhase,
  round,
  logs,
  onStart,
  onPause,
  onResume,
  onStop,
  onToggleMode,
  onStep,
  onClear,
}) => {
  const logEndRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [subTab, setSubTab] = useState('live');
  const [researchInput, setResearchInput] = useState('');

  useEffect(() => {
    if (autoScroll && logEndRef.current && subTab === 'live') {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll, subTab]);

  const phaseInfo = RESEARCH_PHASES.find(p => p.id === currentPhase);
  const phaseIdx = RESEARCH_PHASES.findIndex(p => p.id === currentPhase);
  const isManual = sessionMode === 'manual';
  const isWaiting = sessionStatus === 'waiting_step';
  const isRunning = sessionStatus === 'running';
  const isIdle = sessionStatus === 'idle' || sessionStatus === 'stopped';

  const handleStartWithInput = () => {
    if (!isIdle) return;
    onStart?.(researchInput.trim());
  };

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
      {/* Compact Status Panel */}
      <div style={{
        padding: '10px 16px', borderBottom: '1px solid rgba(167, 139, 250, 0.15)',
        background: 'rgba(10, 10, 16, 0.45)', backdropFilter: 'blur(10px)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, justifyContent: 'space-between' }}>
          {/* Left: Round & Status */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{
              width: 32, height: 32, borderRadius: '50%',
              border: `2px solid ${phaseInfo?.color || '#3b82f6'}`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: 'rgba(0,0,0,0.4)',
              boxShadow: isRunning ? `0 0 10px ${phaseInfo?.color || '#3b82f6'}30` : 'none',
              animation: isRunning ? 'pulse 2.5s infinite' : (isWaiting ? 'pulse 1.5s infinite' : 'none'),
            }}>
              <span style={{ fontSize: '14px', fontWeight: 'bold', color: phaseInfo?.color || '#ef4444', fontFamily: 'monospace' }}>{round}</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <div style={{ fontSize: '10px', color: '#888', fontWeight: 'bold' }}>当前轮次 & 状态</div>
              <div style={{ fontSize: '13px', fontWeight: 'bold', color: phaseInfo?.color || '#aaa', display: 'flex', alignItems: 'center', gap: 6 }}>
                {isWaiting && (
                  <span style={{ fontSize: '9px', color: '#ffaa00', padding: '1px 5px', borderRadius: '4px', background: '#ffaa0015', border: '1px solid #ffaa0030', fontWeight: 'bold' }}>
                    等待
                  </span>
                )}
                <span>{phaseInfo ? phaseInfo.label : '引擎待启动'}</span>
              </div>
            </div>
          </div>

          {/* Right: Actions and Mode toggle */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {/* Step button for Manual mode */}
            {isWaiting && isManual && (
              <button onClick={onStep} style={compactActionBtnStyle('#00d2ff', '#002233')} title={`下一步: ${RESEARCH_PHASES[phaseIdx + 1]?.label || '开始'}`}>
                <SkipForward size={11} /> 下一步
              </button>
            )}

            {/* Start */}
            {isIdle && (
              <button onClick={onStart} style={compactActionBtnStyle('#00ff88', '#003311')} title="启动研发">
                <Play size={11} /> 启动
              </button>
            )}

            {/* Running controls */}
            {isRunning && !isWaiting && (
              <div style={{ display: 'flex', gap: 4 }}>
                <button onClick={onPause} style={compactActionBtnStyle('#ffaa00', '#332200')} title="暂停">
                  <Pause size={11} /> 暂停
                </button>
                <button onClick={onStop} style={compactActionBtnStyle('#ff4444', '#330000')} title="停止">
                  <Square size={11} /> 停止
                </button>
              </div>
            )}

            {/* Paused controls */}
            {sessionStatus === 'paused' && (
              <div style={{ display: 'flex', gap: 4 }}>
                <button onClick={onResume} style={compactActionBtnStyle('#00ff88', '#003311')} title="继续">
                  <Play size={11} /> 继续
                </button>
                <button onClick={onStop} style={compactActionBtnStyle('#ff4444', '#330000')} title="停止">
                  <Square size={11} /> 停止
                </button>
              </div>
            )}

            <div style={{ width: '1px', height: '16px', background: 'rgba(255,255,255,0.08)' }} />

            {/* Mode toggle */}
            <div style={{
              display: 'flex', alignItems: 'center', background: 'rgba(255,255,255,0.03)', borderRadius: '12px',
              border: '1px solid rgba(255,255,255,0.08)', padding: '2px', flexShrink: 0
            }}>
              <button onClick={() => { if (isManual) onToggleMode(); }}
                style={compactModeBtnStyle(!isManual, '#00ff88')}
                title="自动模式">
                自动
              </button>
              <button onClick={() => { if (!isManual) onToggleMode(); }}
                style={compactModeBtnStyle(isManual, '#ffaa00')}
                title="手动模式">
                手动
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Research input */}
      <div style={{
        padding: '12px 16px',
        borderBottom: '1px solid rgba(167, 139, 250, 0.12)',
        background: 'rgba(8, 8, 14, 0.34)',
        display: 'flex',
        gap: 10,
        alignItems: 'stretch',
        flexShrink: 0,
      }}>
        <textarea
          value={researchInput}
          onChange={e => setResearchInput(e.target.value)}
          placeholder="输入本轮研究目标、测试问题或提示词..."
          rows={2}
          style={{
            flex: 1,
            minHeight: 42,
            maxHeight: 96,
            resize: 'vertical',
            boxSizing: 'border-box',
            border: '1px solid rgba(167, 139, 250, 0.22)',
            borderRadius: 8,
            background: 'rgba(0, 0, 0, 0.26)',
            color: '#e5e7eb',
            outline: 'none',
            padding: '9px 11px',
            fontSize: 12,
            lineHeight: 1.45,
            fontFamily: 'inherit',
          }}
        />
        <button
          onClick={handleStartWithInput}
          disabled={!isIdle}
          style={{
            minWidth: 78,
            borderRadius: 8,
            border: `1px solid ${isIdle ? '#00ff8840' : 'rgba(255,255,255,0.08)'}`,
            background: isIdle ? '#003311' : 'rgba(255,255,255,0.04)',
            color: isIdle ? '#00ff88' : '#6b7280',
            cursor: isIdle ? 'pointer' : 'not-allowed',
            fontSize: 12,
            fontWeight: 'bold',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 6,
            flexShrink: 0,
          }}
          title={isIdle ? '开始自动研发' : '当前研发循环运行中'}
        >
          <Play size={13} />
          开始
        </button>
      </div>

      {/* Content area */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {subTab === 'live' && (
          <>
            {/* Terminal Title Bar */}
            <div style={{
              background: 'rgba(15, 15, 25, 0.25)',
              borderBottom: '1px solid rgba(255, 255, 255, 0.03)',
              padding: '6px 16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: '10px',
              color: '#6b7280',
              fontFamily: 'monospace',
              flexShrink: 0
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <div style={{ display: 'flex', gap: 4 }}>
                  <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#ef4444' }} />
                  <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#f59e0b' }} />
                  <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#10b981' }} />
                </div>
                <div style={{ letterSpacing: '1px' }}>LOGGER ({logs.length} Lines)</div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <button onClick={() => setAutoScroll(prev => !prev)} style={{
                  background: 'none', border: 'none', color: autoScroll ? '#00ff88' : '#555',
                  cursor: 'pointer', fontSize: '9px', fontWeight: 'bold'
                }}>
                  自动滚动: {autoScroll ? 'ON' : 'OFF'}
                </button>
                <button onClick={onClear} style={{
                  background: 'none', border: 'none', color: '#ff6666',
                  cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 3
                }} title="清空日志">
                  <Trash2 size={10} /> 清空
                </button>
              </div>
            </div>

            {/* Log stream */}
            <div
              className="airnd-scroll"
              style={{
                flex: 1, overflowY: 'auto', padding: '16px',
                fontFamily: "'Cascadia Code', 'Fira Code', 'Consolas', monospace",
                fontSize: '11px', lineHeight: '1.6',
                background: 'rgba(10, 10, 16, 0.2)',
              }}
              onScroll={(e) => {
                const { scrollTop, scrollHeight, clientHeight } = e.target;
                setAutoScroll(scrollHeight - scrollTop - clientHeight < 100);
              }}
            >
              {logs.length === 0 && (
                <div style={{
                  maxWidth: '380px', margin: '40px auto', textAlign: 'center',
                  padding: '24px 16px',
                  borderRadius: '16px',
                  border: '1px solid rgba(167, 139, 250, 0.2)',
                  background: 'linear-gradient(135deg, rgba(167, 139, 250, 0.08) 0%, rgba(167, 139, 250, 0.02) 100%)',
                  animation: 'roadmapFade 0.6s ease-out'
                }}>
                  <div style={{ marginBottom: 12, display: 'flex', justifyContent: 'center' }}>
                    <Brain size={36} color="#a78bfa" style={{ filter: 'drop-shadow(0 0 10px rgba(167, 139, 250, 0.5))' }} />
                  </div>
                  <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#a78bfa', marginBottom: 8 }}>AI 自动研发引擎就绪</div>
                  <div style={{ fontSize: '12px', color: '#9ca3af', lineHeight: '1.5' }}>
                    请点击上方 <span style={{ color: '#00ff88', fontWeight: 'bold' }}>▶ 启动</span> 按钮开始研究。
                  </div>
                </div>
              )}

              {logs.map((log, i) => (
                <div key={log.id || i} style={{
                  padding: '3px 0', display: 'flex', gap: 8,
                  borderBottom: '1px solid rgba(255,255,255,0.01)',
                  alignItems: 'flex-start',
                }}>
                  <span style={{ color: '#4b5563', minWidth: 60, userSelect: 'none' }}>{log.timestamp || ''}</span>
                  <span style={{ color: getLogColor(log.type), minWidth: 12, textAlign: 'center', userSelect: 'none' }}>
                    {log.type === 'error' ? '✗' : log.type === 'finding' ? '★' : '●'}
                  </span>
                  <span style={{ color: getLogColor(log.type), whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                    {formatLogMessage(log)}
                  </span>
                </div>
              ))}
              <div ref={logEndRef} />
            </div>
          </>
        )}

      </div>
    </div>
  );
};

const compactModeBtnStyle = (active, color) => ({
  padding: '3px 8px', borderRadius: '10px', border: 'none',
  background: active ? `${color}20` : 'transparent', color: active ? color : '#6b7280',
  cursor: 'pointer', fontSize: '10px', fontWeight: 'bold',
  transition: 'all 0.2s ease',
});

const compactActionBtnStyle = (color, bg) => ({
  padding: '5px 10px', borderRadius: '6px', border: `1px solid ${color}40`,
  background: bg, color, cursor: 'pointer',
  fontSize: '11px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: 4,
  transition: 'all 0.2s ease',
  flexShrink: 0
});
