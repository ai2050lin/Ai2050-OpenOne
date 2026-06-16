import { useRef, useEffect, useState } from 'react';
import { Play, Pause, Square, Trash2, Search, Target, Zap, CheckCircle, Settings, Brain, FileText, Lightbulb, SkipForward, Repeat } from 'lucide-react';
import { RESEARCH_PHASES } from './aiRnDConfig';
import { AIRnDLogTab } from './AIRnDLogTab';
import { AIRnDCodeGenTab } from './AIRnDCodeGenTab';
import { AIRnDFindingsTab } from './AIRnDFindingsTab';

const PHASE_ICONS = {
  analyze: Search,
  plan: Target,
  generate: Zap,
  execute: Play,
  summarize: CheckCircle,
};

const SUB_TABS = [
  { id: 'live', label: '实时日志', icon: FileText },
  { id: 'codegen', label: '代码生成', icon: Zap },
  { id: 'findings', label: '发现', icon: Lightbulb },
];

export const AIRnDConsoleTab = ({
  sessionStatus,
  sessionMode,
  currentPhase,
  round,
  logs,
  generatedCode,
  executionResult,
  findings,
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

  useEffect(() => {
    if (autoScroll && logEndRef.current && subTab === 'live') {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll, subTab]);

  // Auto-switch sub-tab when code is generated or findings arrive
  useEffect(() => {
    if (generatedCode && subTab === 'live') {
      setSubTab('codegen');
    }
  }, [generatedCode]);
  useEffect(() => {
    if (findings.length > 0 && subTab !== 'findings') {
      // Don't auto-switch away from codegen
    }
  }, [findings]);

  const phaseInfo = RESEARCH_PHASES.find(p => p.id === currentPhase);
  const phaseIdx = RESEARCH_PHASES.findIndex(p => p.id === currentPhase);
  const isManual = sessionMode === 'manual';
  const isWaiting = sessionStatus === 'waiting_step';
  const isRunning = sessionStatus === 'running';
  const isIdle = sessionStatus === 'idle' || sessionStatus === 'stopped';

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
        padding: '20px 40px', borderBottom: '1px solid rgba(167, 139, 250, 0.15)',
        background: 'rgba(10, 10, 16, 0.45)', backdropFilter: 'blur(10px)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 28 }}>
          {/* Round indicator */}
          <div style={{
            width: 72, height: 72, borderRadius: '50%',
            border: `3px double ${phaseInfo?.color || '#3b82f6'}`,
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            background: 'rgba(0,0,0,0.4)',
            boxShadow: isRunning
              ? `0 0 24px ${phaseInfo?.color || '#3b82f6'}40, inset 0 0 12px ${phaseInfo?.color || '#3b82f6'}20`
              : 'none',
            transition: 'all 0.5s ease',
            animation: isRunning ? 'pulse 2.5s infinite' : (isWaiting ? 'pulse 1.5s infinite' : 'none'),
          }}>
            <span style={{ fontSize: '24px', fontWeight: '900', color: phaseInfo?.color || '#ef4444', fontFamily: 'monospace' }}>{round}</span>
            <span style={{ fontSize: '8px', color: '#9ca3af', fontWeight: 'bold', letterSpacing: '1px' }}>ROUND</span>
          </div>

          {/* Phase info */}
          <div>
            <div style={{ fontSize: '13px', color: '#9ca3af', marginBottom: 4, fontWeight: 'bold' }}>当前状态</div>
            <div style={{ fontSize: '16px', fontWeight: '900', color: phaseInfo?.color || '#ef4444', display: 'flex', alignItems: 'center', gap: 6 }}>
              {isWaiting && (
                <span style={{ fontSize: '11px', color: '#ffaa00', padding: '2px 8px', borderRadius: '10px', background: '#ffaa0015', border: '1px solid #ffaa0030' }}>
                  ⏸ 等待操作
                </span>
              )}
              <span>{phaseInfo ? phaseInfo.label : '引擎待启动'}</span>
            </div>
          </div>

          {/* Mode toggle */}
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8, padding: '4px 4px',
              background: 'rgba(255,255,255,0.03)', borderRadius: '20px',
              border: '1px solid rgba(255,255,255,0.08)',
            }}>
              <button onClick={() => { if (isManual) onToggleMode(); }}
                style={modeBtnStyle(!isManual, '#00ff88')}
                title="自动模式">
                <Repeat size={14} /> 自动
              </button>
              <button onClick={() => { if (!isManual) onToggleMode(); }}
                style={modeBtnStyle(isManual, '#ffaa00')}
                title="手动模式">
                <SkipForward size={14} /> 手动
              </button>
            </div>

            <StatBox label="日志" value={logs.length} color="#6366f1" />
            <StatBox label="发现" value={findings.length} color="#00ff88" />
          </div>
        </div>
      </div>

      {/* Research Pipeline Flow */}
      <div style={{
        padding: '8px 40px', background: 'rgba(10, 10, 18, 0.35)',
        borderBottom: '1px solid rgba(255,255,255,0.04)',
        display: 'flex', alignItems: 'center', gap: 0,
      }}>
        {RESEARCH_PHASES.map((phase, i) => {
          const Icon = PHASE_ICONS[phase.id];
          const isActive = currentPhase === phase.id;
          const isDone = phaseIdx > i;
          const phaseColor = isActive ? phase.color : (isDone ? '#4a5568' : '#2d2d3a');

          return (
            <div key={phase.id} style={{ display: 'flex', alignItems: 'center' }}>
              <div style={{
                display: 'flex', alignItems: 'center', gap: 5,
                padding: '4px 12px', borderRadius: '14px',
                background: isActive ? `${phase.color}15` : (isDone ? 'rgba(74,85,104,0.1)' : 'transparent'),
                border: `1px solid ${isActive ? `${phase.color}40` : 'transparent'}`,
                boxShadow: isActive ? `0 0 10px ${phase.color}25` : 'none',
                transition: 'all 0.3s ease',
              }}>
                {isDone ? (
                  <CheckCircle size={14} color="#4a5568" />
                ) : (
                  <Icon size={14} color={isActive ? phase.color : '#555'} />
                )}
                <span style={{
                  fontSize: '11px', fontWeight: isActive ? 'bold' : 'normal',
                  color: isActive ? phase.color : (isDone ? '#4a5568' : '#555'),
                }}>
                  {phase.label}
                </span>
                {isActive && isWaiting && (
                  <span style={{ width: 5, height: 5, borderRadius: '50%', background: '#ffaa00', animation: 'pulse 1s infinite' }} />
                )}
              </div>

              {i < RESEARCH_PHASES.length - 1 && (
                <div style={{
                  width: 28, height: 2,
                  background: isDone ? '#4a5568' : (isActive ? `linear-gradient(90deg, ${phase.color}, rgba(255,255,255,0.05))` : 'rgba(255,255,255,0.06)'),
                  transition: 'all 0.3s',
                }} />
              )}
            </div>
          );
        })}
        <div style={{ marginLeft: 'auto', fontSize: '10px', color: isManual ? '#ffaa00' : '#6b7280', fontFamily: 'monospace', fontWeight: 'bold' }}>
          {isManual ? '🔬 手动研究' : '🔄 自动研究'} | ROUND {round}
        </div>
      </div>

      {/* Sub-tab bar */}
      <div style={{
        background: 'rgba(15, 15, 25, 0.45)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
        padding: '0 40px',
        display: 'flex',
        alignItems: 'center',
        gap: 2,
      }}>
        {SUB_TABS.map(t => {
          const Icon = t.icon;
          const isActive = subTab === t.id;
          return (
            <button
              key={t.id}
              onClick={() => setSubTab(t.id)}
              style={{
                padding: '10px 20px',
                background: isActive ? 'rgba(167, 139, 250, 0.08)' : 'transparent',
                border: 'none',
                borderBottom: `2px solid ${isActive ? '#a78bfa' : 'transparent'}`,
                color: isActive ? '#a78bfa' : '#6b7280',
                fontSize: '12px', fontWeight: 'bold', cursor: 'pointer',
                display: 'flex', alignItems: 'center', gap: 6,
                transition: 'all 0.2s ease',
              }}
              onMouseEnter={e => { if (!isActive) e.currentTarget.style.color = '#9ca3af'; }}
              onMouseLeave={e => { if (!isActive) e.currentTarget.style.color = '#6b7280'; }}
            >
              <Icon size={13} />
              {t.label}
              {t.id === 'codegen' && generatedCode && (
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#00ff88', display: 'inline-block' }} />
              )}
              {t.id === 'findings' && findings.length > 0 && (
                <span style={{ fontSize: '10px', color: '#a78bfa' }}>({findings.length})</span>
              )}
            </button>
          );
        })}
      </div>

      {/* Content area */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {subTab === 'live' && (
          <>
            {/* Terminal Title Bar */}
            <div style={{
              background: 'rgba(15, 15, 25, 0.25)',
              borderBottom: '1px solid rgba(255, 255, 255, 0.03)',
              padding: '6px 40px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: '10px',
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
                    引擎已处于待命状态。请点击下方 <span style={{ color: '#00ff88', fontWeight: 'bold' }}>▶ 启动</span> 按钮开始研究。
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
          </>
        )}

        {subTab === 'codegen' && (
          <div className="airnd-scroll" style={{ flex: 1, overflow: 'auto' }}>
            <AIRnDCodeGenTab code={generatedCode} result={executionResult} />
          </div>
        )}

        {subTab === 'findings' && (
          <div className="airnd-scroll" style={{ flex: 1, overflow: 'auto' }}>
            <AIRnDFindingsTab findings={findings} round={round} />
          </div>
        )}
      </div>

      {/* Bottom bar */}
      <div style={{
        padding: '12px 40px', borderTop: '1px solid rgba(255,255,255,0.06)',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        background: 'rgba(15, 15, 25, 0.3)',
      }}>
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={onClear} style={smallBtnStyle} title="清空日志">
            <Trash2 size={12} /> 清空
          </button>
        </div>

        {/* Control Buttons */}
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          {/* Manual: Step button */}
          {isWaiting && isManual && (
            <button onClick={onStep} style={actionBtnStyle('#00d2ff', '#002233')}>
              <SkipForward size={16} /> 下一步 ({RESEARCH_PHASES[phaseIdx + 1]?.label || '开始'})
            </button>
          )}

          {/* Start */}
          {isIdle && (
            <button onClick={onStart} style={actionBtnStyle('#00ff88', '#003311')}>
              <Play size={16} /> 启动
            </button>
          )}

          {/* Running: Pause + Stop */}
          {isRunning && !isWaiting && (
            <>
              <button onClick={onPause} style={actionBtnStyle('#ffaa00', '#332200')}>
                <Pause size={16} /> 暂停
              </button>
              <button onClick={onStop} style={actionBtnStyle('#ff4444', '#330000')}>
                <Square size={16} /> 停止
              </button>
            </>
          )}

          {/* Paused: Resume + Stop */}
          {sessionStatus === 'paused' && (
            <>
              <button onClick={onResume} style={actionBtnStyle('#00ff88', '#003311')}>
                <Play size={16} /> 继续
              </button>
              <button onClick={onStop} style={actionBtnStyle('#ff4444', '#330000')}>
                <Square size={16} /> 停止
              </button>
            </>
          )}
        </div>

        <span style={{ fontSize: '11px', color: '#4b5563', fontFamily: 'monospace' }}>
          {autoScroll && subTab === 'live' ? 'AUTO SCROLL: ON' : 'AUTO SCROLL: OFF'} | {logs.length} LINES
        </span>
      </div>
    </div>
  );
};

const StatBox = ({ label, value, color }) => (
  <div
    className="airnd-card"
    style={{
      padding: '8px 16px', background: 'rgba(0,0,0,0.25)',
      borderRadius: '10px', border: '1px solid rgba(255,255,255,0.06)',
      textAlign: 'center', position: 'relative', overflow: 'hidden',
    }}
  >
    <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '2px', background: color }} />
    <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#f3f4f6', fontFamily: 'monospace' }}>{value}</div>
    <div style={{ fontSize: '10px', color: '#9ca3af', marginTop: '2px' }}>{label}</div>
  </div>
);

const modeBtnStyle = (active, color) => ({
  padding: '5px 14px', borderRadius: '16px', border: active ? `1px solid ${color}40` : '1px solid transparent',
  background: active ? `${color}15` : 'transparent', color: active ? color : '#6b7280',
  cursor: 'pointer', fontSize: '11px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: 5,
  transition: 'all 0.2s ease',
});

const smallBtnStyle = {
  background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)',
  color: '#9ca3af', borderRadius: '6px', padding: '6px 12px', cursor: 'pointer',
  fontSize: '11px', display: 'flex', alignItems: 'center', gap: 6,
  transition: 'all 0.2s',
};

const actionBtnStyle = (color, bg) => ({
  padding: '7px 16px', borderRadius: '8px', border: `1px solid ${color}40`,
  background: bg, color, cursor: 'pointer',
  fontSize: '13px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: 6,
  transition: 'all 0.2s ease',
});
