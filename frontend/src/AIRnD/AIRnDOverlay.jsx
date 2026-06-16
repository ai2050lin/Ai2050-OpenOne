import { useState, useEffect, useCallback, useRef } from 'react';
import { X, Brain, Target, Activity, Search, Zap, CheckCircle, Play } from 'lucide-react';
import { AIRnDConfigTab } from './AIRnDConfigTab';
import { AIRnDConsoleTab } from './AIRnDConsoleTab';
import { RESEARCH_PHASES } from './aiRnDConfig';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

const TABS = [
  { id: 'console', label: '控制台', icon: Activity },
  { id: 'config', label: '配置', icon: Target },
];

const PHASE_ICONS = {
  analyze: Search,
  plan: Target,
  generate: Zap,
  execute: Play,
  summarize: CheckCircle,
};

export const AIRnDOverlay = ({ onClose }) => {
  const [activeTab, setActiveTab] = useState('console');
  const [sessionStatus, setSessionStatus] = useState('idle'); // idle | running | paused | stopped | waiting_step
  const [sessionMode, setSessionMode] = useState('auto'); // auto | manual
  const [currentPhase, setCurrentPhase] = useState(null);
  const [round, setRound] = useState(0);
  const [logs, setLogs] = useState([]);
  const [findings, setFindings] = useState([]);
  const [generatedCode, setGeneratedCode] = useState('');
  const [executionResult, setExecutionResult] = useState(null);
  const [researchState, setResearchState] = useState(null);
  const [error, setError] = useState(null);
  const eventSourceRef = useRef(null);

  // Fetch initial session status
  useEffect(() => {
    fetchSessionStatus();
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const fetchSessionStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/ai-rnd/session/status`);
      if (res.ok) {
        const data = await res.json();
        setSessionStatus(data.status || 'idle');
        setSessionMode(data.mode || 'auto');
        setCurrentPhase(data.current_phase || null);
        setRound(data.round || 0);
        setResearchState(data.research_state || null);
      }
    } catch (e) {
      console.warn('AI R&D backend not reachable:', e.message);
    }
  }, []);

  // SSE event stream
  const connectEventStream = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    const es = new EventSource(`${API_BASE}/api/ai-rnd/session/events`);
    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleEvent(data);
      } catch (e) {
        console.warn('Failed to parse SSE event:', e);
      }
    };
    es.onerror = () => {
      es.close();
      eventSourceRef.current = null;
    };
    eventSourceRef.current = es;
  }, []);

  const handleEvent = useCallback((data) => {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = { ...data, timestamp, id: Date.now() + Math.random() };
    setLogs(prev => [...prev.slice(-500), logEntry]);

    if (data.type === 'phase_change') {
      setCurrentPhase(data.phase);
    } else if (data.type === 'round_change') {
      setRound(data.round);
    } else if (data.type === 'status_change') {
      setSessionStatus(data.status);
    } else if (data.type === 'mode_change') {
      setSessionMode(data.mode);
    } else if (data.type === 'finding') {
      setFindings(prev => [...prev.slice(-200), data.finding]);
    } else if (data.type === 'code_generated') {
      setGeneratedCode(data.code || '');
    } else if (data.type === 'execution_result') {
      setExecutionResult(data.result);
    } else if (data.type === 'error') {
      setError(data.message);
    }
  }, []);

  // Control actions
  const startSession = useCallback(async () => {
    try {
      setError(null);
      const res = await fetch(`${API_BASE}/api/ai-rnd/session/start`, { method: 'POST' });
      if (res.ok) {
        setSessionStatus('running');
        connectEventStream();
      } else {
        const err = await res.json();
        setError(err.detail || '启动失败');
      }
    } catch (e) {
      setError(`连接后端失败: ${e.message}`);
    }
  }, [connectEventStream]);

  const pauseSession = useCallback(async () => {
    try {
      await fetch(`${API_BASE}/api/ai-rnd/session/pause`, { method: 'POST' });
      setSessionStatus('paused');
    } catch (e) { setError(e.message); }
  }, []);

  const resumeSession = useCallback(async () => {
    try {
      await fetch(`${API_BASE}/api/ai-rnd/session/start`, { method: 'POST' });
      setSessionStatus('running');
    } catch (e) { setError(e.message); }
  }, []);

  const stopSession = useCallback(async () => {
    try {
      await fetch(`${API_BASE}/api/ai-rnd/session/stop`, { method: 'POST' });
      setSessionStatus('stopped');
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    } catch (e) { setError(e.message); }
  }, []);

  const toggleMode = useCallback(async () => {
    const newMode = sessionMode === 'auto' ? 'manual' : 'auto';
    try {
      const res = await fetch(`${API_BASE}/api/ai-rnd/session/mode`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: newMode }),
      });
      if (res.ok) {
        setSessionMode(newMode);
      }
    } catch (e) { setError(e.message); }
  }, [sessionMode]);

  const stepNext = useCallback(async () => {
    try {
      setError(null);
      const res = await fetch(`${API_BASE}/api/ai-rnd/session/step`, { method: 'POST' });
      if (!res.ok) {
        const err = await res.json();
        setError(err.detail || 'Step failed');
      }
    } catch (e) { setError(e.message); }
  }, []);

  const phaseInfo = RESEARCH_PHASES.find(p => p.id === currentPhase);
  const statusColor = sessionStatus === 'running' ? '#00ff88' : sessionStatus === 'paused' ? '#ffaa00' : '#666';

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 3000,
      background: 'rgba(10, 10, 18, 0.88)',
      backdropFilter: 'blur(20px)',
      display: 'flex', flexDirection: 'column',
      fontFamily: "'Segoe UI', system-ui, -apple-system, sans-serif",
      color: '#e0e0e0',
      animation: 'roadmapFade 0.5s cubic-bezier(0.16, 1, 0.3, 1)',
    }}>
      {/* Global Style Injections */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }
        @keyframes roadmapFade {
          from { opacity: 0; transform: translateY(15px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        /* Custom Scrollbar for AI R&D */
        .airnd-scroll::-webkit-scrollbar {
          width: 6px;
          height: 6px;
        }
        .airnd-scroll::-webkit-scrollbar-track {
          background: rgba(0, 0, 0, 0.1);
        }
        .airnd-scroll::-webkit-scrollbar-thumb {
          background: rgba(167, 139, 250, 0.2);
          border-radius: 3px;
        }
        .airnd-scroll::-webkit-scrollbar-thumb:hover {
          background: rgba(167, 139, 250, 0.4);
        }
        
        /* Input focus glow and transitions */
        .airnd-input {
          transition: border-color 0.25s ease, box-shadow 0.25s ease, background-color 0.25s ease !important;
        }
        .airnd-input:focus {
          border-color: rgba(167, 139, 250, 0.6) !important;
          box-shadow: 0 0 10px rgba(167, 139, 250, 0.25) !important;
          background-color: rgba(0, 0, 0, 0.45) !important;
        }
        
        /* Interactive Cards */
        .airnd-card {
          transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease !important;
        }
        .airnd-card:hover {
          border-color: rgba(167, 139, 250, 0.35) !important;
          box-shadow: 0 8px 24px rgba(167, 139, 250, 0.08) !important;
        }
      `}</style>

      {/* Animated background */}
      <div style={{
        position: 'absolute', inset: 0, zIndex: 0,
        background: 'radial-gradient(ellipse at 20% 50%, rgba(99, 102, 241, 0.05) 0%, transparent 60%), radial-gradient(ellipse at 80% 20%, rgba(167, 139, 250, 0.05) 0%, transparent 60%)',
        pointerEvents: 'none',
      }} />

      {/* Top Header */}
      <div style={{
        padding: '0 40px', height: '80px', display: 'flex', justifyContent: 'space-between',
        alignItems: 'center', borderBottom: '1px solid rgba(167, 139, 250, 0.18)',
        background: 'rgba(10, 10, 16, 0.4)', backdropFilter: 'blur(10px)', position: 'relative', zIndex: 1,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '40px', height: '100%' }}>
          {/* Title */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{
              width: 32, height: 32, borderRadius: '8px',
              background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 0 15px rgba(99, 102, 241, 0.4)',
            }}>
              <Brain size={18} color="#fff" style={{ filter: 'drop-shadow(0 0 5px rgba(255,255,255,0.75))' }} />
            </div>
            <span style={{ fontSize: '18px', fontWeight: 'bold', letterSpacing: '2px', color: '#f3f4f6' }}>AI 自动研发</span>
          </div>

          {/* Tab Navigation */}
          <nav style={{ display: 'flex', gap: '8px', alignItems: 'center', height: '100%', overflowX: 'auto', scrollbarWidth: 'none' }}>
            {TABS.map(t => {
              const Icon = t.icon;
              const isActive = activeTab === t.id;
              return (
                <button
                  key={t.id}
                  onClick={() => setActiveTab(t.id)}
                  style={{
                    background: isActive 
                      ? 'linear-gradient(135deg, rgba(167, 139, 250, 0.15) 0%, rgba(99, 102, 241, 0.05) 100%)' 
                      : 'transparent',
                    border: `1px solid ${isActive ? 'rgba(167, 139, 250, 0.35)' : 'transparent'}`,
                    borderRadius: '8px',
                    color: isActive ? '#a78bfa' : '#9ca3af',
                    fontSize: '14px', fontWeight: 'bold', cursor: 'pointer',
                    padding: '8px 16px',
                    transition: 'all 0.25s ease', display: 'flex', alignItems: 'center', gap: '6px',
                    boxShadow: isActive ? '0 0 10px rgba(167, 139, 250, 0.1)' : 'none',
                  }}
                  onMouseEnter={e => {
                    if (!isActive) {
                      e.currentTarget.style.color = '#c084fc';
                      e.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)';
                    }
                  }}
                  onMouseLeave={e => {
                    if (!isActive) {
                      e.currentTarget.style.color = '#9ca3af';
                      e.currentTarget.style.background = 'transparent';
                    }
                  }}
                >
                  <Icon size={14} style={{ filter: isActive ? 'drop-shadow(0 0 4px #a78bfa)' : 'none' }} />
                  {t.label}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Right controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {/* Session status indicator */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 8, padding: '6px 14px',
            background: 'rgba(255,255,255,0.03)', borderRadius: '20px',
            border: '1px solid rgba(255,255,255,0.06)',
          }}>
            <div style={{
              width: 8, height: 8, borderRadius: '50%', background: statusColor,
              boxShadow: sessionStatus === 'running' ? `0 0 8px ${statusColor}` : 'none',
              animation: sessionStatus === 'running' ? 'pulse 2s infinite' : 'none',
            }} />
            <span style={{ fontSize: '12px', color: '#d1d5db', fontFamily: 'monospace' }}>
              {sessionStatus === 'idle' ? '就绪' : sessionStatus === 'running' ? `R${round} ${phaseInfo?.label || currentPhase || ''}` : sessionStatus === 'paused' ? '已暂停' : '已停止'}
            </span>
          </div>

          {/* Close button */}
          <button onClick={onClose} style={{
            background: 'rgba(255,255,255,0.05)', border: 'none', color: '#fff', cursor: 'pointer',
            width: '40px', height: '40px', borderRadius: '50%',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            transition: 'all 0.25s ease',
          }}
          onMouseEnter={e => e.currentTarget.style.background = 'rgba(239,68,68,0.2)'}
          onMouseLeave={e => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}
          >
            <X size={22} style={{ filter: 'drop-shadow(0 0 4px rgba(255,255,255,0.5))' }} />
          </button>
        </div>
      </div>

      {/* Phase Progress Bar */}
      {sessionStatus === 'running' && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 4, padding: '12px 40px',
          background: 'rgba(10, 10, 16, 0.4)', borderBottom: '1px solid rgba(167, 139, 250, 0.15)',
          position: 'relative', zIndex: 1,
        }}>
          {RESEARCH_PHASES.map((phase, i) => {
            const isCurrent = currentPhase === phase.id;
            const Icon = PHASE_ICONS[phase.id] || Search;
            return (
              <div key={phase.id} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <div style={{
                  padding: '5px 14px', borderRadius: '16px', fontSize: '11px', fontWeight: 'bold',
                  background: isCurrent ? `${phase.color}20` : 'rgba(255,255,255,0.02)',
                  border: `1px solid ${isCurrent ? phase.color : 'rgba(255,255,255,0.08)'}`,
                  color: isCurrent ? phase.color : '#888',
                  transition: 'all 0.3s',
                  boxShadow: isCurrent ? `0 0 12px ${phase.color}40` : 'none',
                  display: 'flex', alignItems: 'center', gap: '6px',
                }}>
                  <Icon size={12} style={{ filter: isCurrent ? `drop-shadow(0 0 3px ${phase.color})` : 'none' }} />
                  <span>{phase.label}</span>
                </div>
                {i < RESEARCH_PHASES.length - 1 && (
                  <div style={{ 
                    width: 40, height: 2, 
                    background: isCurrent ? `linear-gradient(90deg, ${phase.color}, rgba(255,255,255,0.1))` : 'rgba(255,255,255,0.08)',
                    transition: 'all 0.3s'
                  }} />
                )}
              </div>
            );
          })}
          <div style={{ marginLeft: 'auto', fontSize: '11px', color: '#9ca3af', fontFamily: 'monospace', fontWeight: 'bold' }}>
            ROUND {round}
          </div>
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div style={{
          padding: '8px 40px', background: 'rgba(255,50,50,0.1)', borderBottom: '1px solid rgba(255,50,50,0.3)',
          color: '#ff6666', fontSize: '13px', position: 'relative', zIndex: 1,
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}>
          <span>{error}</span>
          <button onClick={() => setError(null)} style={{ background: 'none', border: 'none', color: '#ff6666', cursor: 'pointer' }}>
            <X size={14} />
          </button>
        </div>
      )}

      {/* Main Content */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', position: 'relative', zIndex: 1 }}>
        {activeTab === 'config' && (
          <AIRnDConfigTab />
        )}
        {activeTab === 'console' && (
          <AIRnDConsoleTab
            sessionStatus={sessionStatus}
            sessionMode={sessionMode}
            currentPhase={currentPhase}
            round={round}
            logs={logs}
            generatedCode={generatedCode}
            executionResult={executionResult}
            findings={findings}
            onStart={startSession}
            onPause={pauseSession}
            onResume={resumeSession}
            onStop={stopSession}
            onToggleMode={toggleMode}
            onStep={stepNext}
            onClear={() => setLogs([])}
          />
        )}
      </div>

      {/* Pulse animation */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }
      `}</style>
    </div>
  );
};



