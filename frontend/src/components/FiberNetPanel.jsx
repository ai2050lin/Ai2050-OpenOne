import { API_BASE } from '../config.js';
import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import {
  Activity,
  BarChart3,
  Bot,
  Loader2,
  MessageSquare,
  Moon,
  RefreshCw,
  Send,
  User,
  Waves,
} from 'lucide-react';


const S = {
  panel: {
    padding: 0,
    color: '#eee',
    fontFamily: "'Segoe UI', 'PingFang SC', sans-serif",
    fontSize: 13,
    lineHeight: 1.5,
  },
  card: {
    background: 'rgba(15,15,25,0.9)',
    border: '1px solid #333',
    borderRadius: 10,
    padding: 16,
    marginBottom: 12,
  },
  cardTitle: {
    fontSize: 10,
    fontWeight: 800,
    letterSpacing: '0.15em',
    textTransform: 'uppercase',
    color: '#888',
    marginBottom: 10,
  },
  input: {
    flex: 1,
    padding: '8px 12px',
    background: '#111',
    border: '1px solid #444',
    color: '#fff',
    borderRadius: 6,
    outline: 'none',
    fontSize: 13,
  },
  btn: (color = '#4f46e5', disabled = false) => ({
    padding: '8px 16px',
    background: disabled ? '#333' : color,
    color: '#fff',
    border: 'none',
    borderRadius: 6,
    cursor: disabled ? 'not-allowed' : 'pointer',
    fontWeight: 700,
    fontSize: 12,
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    opacity: disabled ? 0.6 : 1,
  }),
  tabBar: {
    display: 'flex',
    gap: 6,
    padding: 12,
    borderBottom: '1px solid rgba(255,255,255,0.06)',
    background: 'rgba(0,0,0,0.2)',
  },
  tabBtn: (active) => ({
    padding: '8px 12px',
    borderRadius: 8,
    border: active ? '1px solid rgba(99,102,241,0.45)' : '1px solid transparent',
    background: active ? 'rgba(99,102,241,0.18)' : 'transparent',
    color: active ? '#e5e7ff' : '#777',
    cursor: 'pointer',
    fontSize: 11,
    fontWeight: 700,
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  }),
  metricGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: 10,
    marginTop: 12,
  },
  metric: {
    padding: 12,
    background: 'rgba(0,0,0,0.28)',
    borderRadius: 8,
    border: '1px solid #2e2e38',
  },
  metricLabel: {
    fontSize: 9,
    fontWeight: 800,
    letterSpacing: '0.12em',
    textTransform: 'uppercase',
    color: '#888',
    marginBottom: 4,
  },
  metricValue: {
    fontSize: 15,
    fontWeight: 800,
    fontFamily: 'monospace',
    color: '#f4f7ff',
  },
};

const TABS = [
  { id: 'chat', labelZh: '语言对话', labelEn: 'Dialogue', icon: Bot, color: '#4488ff' },
  { id: 'semantic', labelZh: '语义推演', labelEn: 'Semantics', icon: Waves, color: '#8b5cf6' },
  { id: 'memory', labelZh: '回放与固化', labelEn: 'Replay', icon: Moon, color: '#a855f7' },
];

function Metric({ label, value }) {
  return (
    <div style={S.metric}>
      <div style={S.metricLabel}>{label}</div>
      <div style={S.metricValue}>{value}</div>
    </div>
  );
}

const ICSPBPanel = ({ lang = 'zh' }) => {
  const isEn = lang === 'en';
  const [activeTab, setActiveTab] = useState('chat');

  const [chatStatus, setChatStatus] = useState({ is_ready: false, status_msg: '检查中...' });
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const [semanticInput, setSemanticInput] = useState('苹果是什么');
  const [semanticResult, setSemanticResult] = useState(null);
  const [semanticLoading, setSemanticLoading] = useState(false);

  const [sleepIterations, setSleepIterations] = useState(20);
  const [isConsolidating, setIsConsolidating] = useState(false);
  const [memoryStatus, setMemoryStatus] = useState(null);
  const [trainSteps, setTrainSteps] = useState(4);
  const [trainRounds, setTrainRounds] = useState(3);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [benchmarkStatus, setBenchmarkStatus] = useState(null);

  useEffect(() => {
    const check = async () => {
      try {
        const r = await axios.get(`${API_BASE}/api/agi_chat/status`);
        setChatStatus(r.data);
      } catch {
        setChatStatus({ is_ready: false, status_msg: isEn ? 'Engine offline' : '引擎离线' });
      }
    };
    check();
    const timer = setInterval(check, 5000);
    return () => clearInterval(timer);
  }, [isEn]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  useEffect(() => {
    refreshTrainingStatus();
  }, []);

  const refreshMemoryChart = async () => {
    try {
      const r = await axios.get(`${API_BASE}/api/icspb/memory/chart`);
      setMemoryStatus((prev) => ({ ...(prev || {}), curvature_history: r.data.history || [] }));
    } catch {
      // ignore refresh errors
    }
  };

  const refreshTrainingStatus = async () => {
    try {
      const r = await axios.get(`${API_BASE}/api/icspb/train/status`);
      setTrainingStatus({ ...r.data, eval_loss: r.data?.eval_loss ?? r.data?.phasea_last_eval_loss });
      setBenchmarkStatus(r.data?.generation_benchmark || null);
    } catch {
      // ignore refresh errors
    }
  };

  const handleChatSend = async () => {
    if (!chatInput.trim() || !chatStatus.is_ready || isTyping) return;
    const prompt = chatInput.trim();
    setChatInput('');
    setMessages((prev) => [...prev, { role: 'user', content: prompt }]);
    setIsTyping(true);
    try {
      const r = await axios.post(`${API_BASE}/api/agi_chat/generate`, {
        prompt,
        max_new_tokens: 48,
      });
      if (r.data?.generated_text) {
        setMessages((prev) => [...prev, { role: 'agi', content: r.data.generated_text }]);
      }
    } catch (e) {
      setMessages((prev) => [...prev, { role: 'sys', content: `Error: ${e.message}` }]);
    } finally {
      setIsTyping(false);
    }
  };

  const resetChat = async () => {
    try {
      await axios.post(`${API_BASE}/api/agi_chat/reset`);
    } finally {
      setMessages([]);
    }
  };

  const handleSemanticInference = async () => {
    if (!semanticInput.trim() || semanticLoading) return;
    setSemanticLoading(true);
    try {
      const r = await axios.post(`${API_BASE}/api/icspb/semantic`, {
        text: semanticInput,
        lang: isEn ? 'en' : 'zh',
      });
      setSemanticResult(r.data);
    } catch (e) {
      setSemanticResult({ error: e.message });
    } finally {
      setSemanticLoading(false);
    }
  };

  const handleConsolidation = async () => {
    if (isConsolidating) return;
    setIsConsolidating(true);
    try {
      const r = await axios.post(`${API_BASE}/api/icspb/memory/consolidate?iterations=${sleepIterations}&mode=adaptive`);
      setMemoryStatus(r.data);
      await refreshMemoryChart();
    } catch (e) {
      setMemoryStatus({ error: e.message });
    } finally {
      setIsConsolidating(false);
    }
  };

  const handleTraining = async () => {
    if (isTraining) return;
    setIsTraining(true);
    try {
      const r = await axios.post(`${API_BASE}/api/icspb/train`, {
        steps: trainSteps,
        batch_size: 1,
        lr: 1e-4,
        max_texts: Math.max(8, trainSteps * 4),
      });
      setTrainingStatus({ ...r.data, eval_loss: r.data?.eval_loss ?? r.data?.phasea_last_eval_loss });
      const benchResp = await axios.post(`${API_BASE}/api/icspb/train/benchmark`);
      setBenchmarkStatus(benchResp.data);
      await refreshTrainingStatus();
      const statusResp = await axios.get(`${API_BASE}/api/agi_chat/status`);
      setChatStatus(statusResp.data);
    } catch (e) {
      setTrainingStatus({ error: e.message });
    } finally {
      setIsTraining(false);
    }
  };

  const handleTrainingPlan = async () => {
    if (isTraining) return;
    setIsTraining(true);
    try {
      const r = await axios.post(`${API_BASE}/api/icspb/train/plan`, {
        rounds: trainRounds,
        steps_per_round: trainSteps,
        batch_size: 1,
        lr: 1e-4,
        max_texts: Math.max(8, trainSteps * trainRounds * 4),
      });
      setTrainingStatus({
        ...(r.data?.training_status || {}),
        eval_loss: r.data?.training_status?.eval_loss ?? r.data?.training_status?.phasea_last_eval_loss,
      });
      await refreshTrainingStatus();
      const statusResp = await axios.get(`${API_BASE}/api/agi_chat/status`);
      setChatStatus(statusResp.data);
    } catch (e) {
      setTrainingStatus({ error: e.message });
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div style={S.panel}>
      <div style={{ padding: '16px 20px', borderBottom: '1px solid #333', background: 'rgba(0,0,0,0.18)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ background: 'rgba(99,102,241,0.2)', padding: 8, borderRadius: 8, display: 'flex' }}>
            <MessageSquare size={20} color="#a5b4fc" />
          </div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 900, letterSpacing: '0.02em' }}>
              {isEn ? 'ICSPB Current Model Workbench' : 'ICSPB 当前模型工作台'}
            </div>
            <div style={{ fontSize: 10, color: '#666', fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase' }}>
              {isEn ? 'Language Backbone + Online Replay Branch' : '语言主干 + 在线回放分支'}
            </div>
          </div>
        </div>
        <div style={{ marginTop: 12, padding: 12, background: 'rgba(255,255,255,0.03)', borderRadius: 8, border: '1px solid rgba(255,255,255,0.06)', fontSize: 12, color: '#b8c2d4' }}>
          {isEn
            ? 'This panel now routes all active functions through the current ICSPB chain: dialogue, semantic inference, and replay/consolidation.'
            : '这个面板现在把所有活跃功能统一到当前 ICSPB 主链：语言对话、语义推演、回放与固化。'}
        </div>
      </div>

      <div style={S.tabBar}>
        {TABS.map((tab) => {
          const Icon = tab.icon;
          const active = activeTab === tab.id;
          return (
            <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={S.tabBtn(active)}>
              <Icon size={14} color={active ? tab.color : '#888'} />
              <span>{isEn ? tab.labelEn : tab.labelZh}</span>
            </button>
          );
        })}
      </div>

      <div style={{ padding: 16 }}>
        {activeTab === 'chat' && (
          <div style={{ ...S.card, padding: 0, display: 'flex', flexDirection: 'column', height: 520, overflow: 'hidden' }}>
            <div style={{ padding: '10px 12px', background: 'rgba(0,0,0,0.3)', borderBottom: '1px solid #333', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 12 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: chatStatus.is_ready ? '#10b981' : '#f59e0b' }}>
                <Activity size={14} />
                {chatStatus.status_msg}
              </div>
              <button onClick={resetChat} style={{ background: 'transparent', border: 'none', color: '#888', cursor: 'pointer' }} title={isEn ? 'Reset' : '重置'}>
                <RefreshCw size={14} />
              </button>
            </div>

            <div style={{ flex: 1, overflowY: 'auto', padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
              {messages.length === 0 && (
                <div style={{ textAlign: 'center', color: '#666', marginTop: 40, fontSize: 12 }}>
                  <Bot size={32} color="#4488ff33" style={{ margin: '0 auto 12px' }} />
                  <p>{isEn ? 'ICSPB dialogue terminal is ready.' : 'ICSPB 语言对话终端已就绪。'}</p>
                  <p style={{ fontSize: 11, color: '#555' }}>
                    {isEn ? 'Use this area to inspect dialogue quality and runtime state.' : '这里用于检查当前语言对话质量和运行状态。'}
                  </p>
                </div>
              )}

              {messages.map((m, i) => (
                <div
                  key={i}
                  style={{
                    alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
                    maxWidth: '85%',
                    background: m.role === 'user' ? '#4488ff' : m.role === 'agi' ? 'rgba(255,255,255,0.08)' : 'rgba(255,0,0,0.15)',
                    padding: '8px 12px',
                    borderRadius: 8,
                    fontSize: 13,
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                  }}
                >
                  <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.45)', marginBottom: 4, display: 'flex', alignItems: 'center', gap: 4 }}>
                    {m.role === 'user' ? <User size={10} /> : <Bot size={10} />}
                    {m.role === 'user' ? (isEn ? 'You' : '你') : m.role === 'agi' ? 'ICSPB' : (isEn ? 'System' : '系统')}
                  </div>
                  {m.content}
                </div>
              ))}

              {isTyping && (
                <div style={{ color: '#888', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4 }}>
                  <Loader2 size={12} className="animate-spin" />
                  {isEn ? 'Generating...' : '生成中...'}
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <div style={{ padding: 12, borderTop: '1px solid #333', display: 'flex', gap: 8 }}>
              <input
                style={S.input}
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleChatSend()}
                placeholder={chatStatus.is_ready ? (isEn ? 'Enter your message...' : '输入消息...') : (isEn ? 'Waiting for engine...' : '等待引擎就绪...')}
                disabled={!chatStatus.is_ready || isTyping}
              />
              <button onClick={handleChatSend} disabled={!chatStatus.is_ready || isTyping || !chatInput.trim()} style={S.btn('#4488ff', !chatStatus.is_ready || isTyping)}>
                <Send size={14} />
              </button>
            </div>
          </div>
        )}

        {activeTab === 'semantic' && (
          <div style={S.card}>
            <div style={S.cardTitle}>{isEn ? 'ICSPB Semantic Inference' : 'ICSPB 语义推演'}</div>
            <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
              <input
                style={S.input}
                value={semanticInput}
                onChange={(e) => setSemanticInput(e.target.value)}
                placeholder={isEn ? 'Enter text...' : '输入文本...'}
              />
              <button style={S.btn('#6366f1', semanticLoading)} onClick={handleSemanticInference} disabled={semanticLoading}>
                {semanticLoading ? <Loader2 size={14} className="animate-spin" /> : <Waves size={14} />}
                {isEn ? 'Infer' : '推演'}
              </button>
            </div>

            <div style={{ fontSize: 12, color: '#9ea7b7', marginBottom: 12 }}>
              {isEn
                ? 'This view now uses the PhaseA language model directly. Answers come from the neural network rather than explicit semantic rules.'
                : '这里直接走当前 ICSPB 主链，展示问题语义解析、答案骨架、概念锚定和 correctness 审查。'}
            </div>

            {semanticResult?.error && (
              <div style={{ padding: 12, background: 'rgba(255,50,50,0.08)', border: '1px solid #6b1f1f', borderRadius: 8, color: '#ff9090', fontSize: 12 }}>
                {semanticResult.error}
              </div>
            )}

            {semanticResult && !semanticResult.error && (
              <>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12 }}>
                  {(semanticResult.tokens || []).map((token, idx) => (
                    <span key={idx} style={{ padding: '3px 8px', background: '#222', borderRadius: 4, fontSize: 12, color: '#ccc' }}>
                      {token}
                    </span>
                  ))}
                  {semanticResult.next_token && (
                    <span style={{ padding: '3px 8px', background: '#4f46e522', borderRadius: 4, fontSize: 12, color: '#818cf8', fontWeight: 700, border: '1px solid #4f46e544' }}>
                      {'-> '} {semanticResult.next_token}
                    </span>
                  )}
                </div>

                <div style={{ ...S.card, marginBottom: 12 }}>
                  <div style={S.cardTitle}>{isEn ? 'Generated Answer' : '生成结果'}</div>
                  <div style={{ fontSize: 13, color: '#eef2ff', whiteSpace: 'pre-wrap' }}>
                    {semanticResult.generated_text}
                  </div>
                </div>

                <div style={S.metricGrid}>
                  <Metric label={isEn ? 'Quality' : '生成质量'} value={Number(semanticResult.correctness_review?.quality_score || 0).toFixed(2)} />
                  <Metric label={isEn ? 'Unique Ratio' : '唯一比率'} value={Number(semanticResult.correctness_review?.unique_token_ratio || 0).toFixed(2)} />
                  <Metric label={isEn ? 'Theorem' : '定理存活'} value={Number(semanticResult.icspb_metrics?.theorem_survival || 0).toFixed(2)} />
                </div>
              </>
            )}
          </div>
        )}

        {activeTab === 'memory' && (
          <div style={{ ...S.card, borderLeft: '3px solid #a855f7' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <Moon size={18} color="#a855f7" />
                  <span style={{ fontSize: 15, fontWeight: 900 }}>{isEn ? 'Replay & Consolidation' : '回放与固化'}</span>
                </div>
                <p style={{ fontSize: 11, color: '#888', maxWidth: 420, margin: 0 }}>
                  {isEn
                    ? 'Run replay/consolidation on the current ICSPB model and observe whether memory curvature improves.'
                    : '在当前 ICSPB 模型上执行回放与固化，并观察记忆曲率是否下降。'}
                </p>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: '#111', padding: '4px 10px', borderRadius: 6, border: '1px solid #333' }}>
                  <span style={{ fontSize: 9, fontWeight: 700, color: '#888' }}>{isEn ? 'Steps' : '步数'}</span>
                  <input
                    type="range"
                    min="5"
                    max="50"
                    value={sleepIterations}
                    onChange={(e) => setSleepIterations(parseInt(e.target.value, 10))}
                    style={{ width: 80, accentColor: '#a855f7' }}
                    disabled={isConsolidating}
                  />
                  <span style={{ fontSize: 12, fontFamily: 'monospace', color: '#a855f7', width: 20 }}>{sleepIterations}</span>
                </div>
                <button style={S.btn('#7c3aed', isConsolidating)} disabled={isConsolidating} onClick={handleConsolidation}>
                  {isConsolidating ? (
                    <>
                      <Loader2 size={14} className="animate-spin" />
                      {isEn ? 'Consolidating...' : '固化处理中...'}
                    </>
                  ) : (
                    <>
                      <Moon size={14} />
                      {isEn ? 'Run Consolidation' : '执行固化'}
                    </>
                  )}
                </button>
              </div>
            </div>

            <div style={{ ...S.card, marginTop: 0, marginBottom: 12 }}>
              <div style={S.cardTitle}>{isEn ? 'PhaseA Training' : 'PhaseA 继续训练'}</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: '#111', padding: '4px 10px', borderRadius: 6, border: '1px solid #333' }}>
                  <span style={{ fontSize: 9, fontWeight: 700, color: '#888' }}>{isEn ? 'Train Steps' : '训练步数'}</span>
                  <input
                    type="range"
                    min="1"
                    max="12"
                    value={trainSteps}
                    onChange={(e) => setTrainSteps(parseInt(e.target.value, 10))}
                    style={{ width: 80, accentColor: '#2563eb' }}
                    disabled={isTraining}
                  />
                  <span style={{ fontSize: 12, fontFamily: 'monospace', color: '#60a5fa', width: 20 }}>{trainSteps}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: '#111', padding: '4px 10px', borderRadius: 6, border: '1px solid #333' }}>
                  <span style={{ fontSize: 9, fontWeight: 700, color: '#888' }}>{isEn ? 'Rounds' : '训练轮数'}</span>
                  <input
                    type="range"
                    min="1"
                    max="6"
                    value={trainRounds}
                    onChange={(e) => setTrainRounds(parseInt(e.target.value, 10))}
                    style={{ width: 80, accentColor: '#0ea5e9' }}
                    disabled={isTraining}
                  />
                  <span style={{ fontSize: 12, fontFamily: 'monospace', color: '#38bdf8', width: 20 }}>{trainRounds}</span>
                </div>
                <button style={S.btn('#2563eb', isTraining)} disabled={isTraining} onClick={handleTraining}>
                  {isTraining ? (
                    <>
                      <Loader2 size={14} className="animate-spin" />
                      {isEn ? 'Training...' : '训练中...'}
                    </>
                  ) : (
                    <>
                      <RefreshCw size={14} />
                      {isEn ? 'Continue Training' : '继续训练'}
                    </>
                  )}
                </button>
                <button style={S.btn('#0f766e', isTraining)} disabled={isTraining} onClick={handleTrainingPlan}>
                  {isTraining ? (
                    <>
                      <Loader2 size={14} className="animate-spin" />
                      {isEn ? 'Planning...' : '批量训练中...'}
                    </>
                  ) : (
                    <>
                      <BarChart3 size={14} />
                      {isEn ? 'Run Training Plan' : '执行训练计划'}
                    </>
                  )}
                </button>
                {trainingStatus?.semantic_benchmark_score !== undefined && (
                  <div style={{ fontSize: 12, color: '#9fb5d1' }}>
                    {isEn
                      ? `score=${Number(trainingStatus.semantic_benchmark_score).toFixed(3)}, eval_loss=${Number(trainingStatus.eval_loss || 0).toFixed(3)}`
                      : `score=${Number(trainingStatus.semantic_benchmark_score).toFixed(3)}，eval_loss=${Number(trainingStatus.eval_loss || 0).toFixed(3)}`}
                  </div>
                )}
              </div>
              {trainingStatus && !trainingStatus.error && (
                <div style={{ marginTop: 10, display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 8 }}>
                  <div style={{ fontSize: 12, color: '#9fb5d1' }}>
                    {isEn ? 'Rounds' : '轮数'}: {trainingStatus.training_rounds ?? 0}
                  </div>
                  <div style={{ fontSize: 12, color: '#9fb5d1' }}>
                    {isEn ? 'Steps' : '步数'}: {trainingStatus.language_training_steps ?? 0}
                  </div>
                  <div style={{ fontSize: 12, color: '#9fb5d1' }}>
                    {isEn ? 'Gen Score' : '生成分数'}: {Number(benchmarkStatus?.headline_metrics?.benchmark_score || trainingStatus.generation_quality_score || 0).toFixed(3)}
                  </div>
                  <div style={{ fontSize: 12, color: '#9fb5d1' }}>
                    {isEn ? 'History' : '历史点数'}: {trainingStatus.history_count ?? 0}
                  </div>
                </div>
              )}
              {trainingStatus?.best_eval_loss !== undefined && (
                <div style={{ marginTop: 10, display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 8 }}>
                  <div style={{ fontSize: 11, color: '#8ea5c5' }}>
                    {isEn ? 'Best eval loss' : '最佳评估损失'}: {Number(trainingStatus.best_eval_loss || 0).toFixed(3)}
                  </div>
                  <div style={{ fontSize: 11, color: '#8ea5c5' }}>
                    {isEn ? 'Best gen score' : '最佳生成分数'}: {Number(trainingStatus.best_generation_quality_score || 0).toFixed(3)}
                  </div>
                </div>
              )}
              {trainingStatus?.history?.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ fontSize: 11, color: '#8ea5c5', marginBottom: 6 }}>
                    {isEn ? 'Recent training trend' : '最近训练趋势'}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'flex-end', gap: 4, height: 84 }}>
                    {trainingStatus.history.map((row, index) => {
                      const maxLoss = Math.max(...trainingStatus.history.map((item) => Number(item.eval_loss || 0)), 1);
                      const genScore = Number(row.generation_quality_score || 0);
                      const evalLoss = Number(row.eval_loss || 0);
                      return (
                        <div key={`${row.timestamp || index}-${index}`} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3, flex: 1 }}>
                          <div
                            style={{
                              width: '70%',
                              borderRadius: '3px 3px 0 0',
                              background: 'rgba(14,165,233,0.72)',
                              height: `${Math.max(6, genScore * 72)}px`,
                            }}
                            title={`gen=${genScore.toFixed(3)}`}
                          />
                          <div
                            style={{
                              width: '70%',
                              borderRadius: '3px 3px 0 0',
                              background: 'rgba(239,68,68,0.62)',
                              height: `${Math.max(6, (evalLoss / maxLoss) * 72)}px`,
                            }}
                            title={`loss=${evalLoss.toFixed(3)}`}
                          />
                        </div>
                      );
                    })}
                  </div>
                  <div style={{ marginTop: 6, fontSize: 10, color: '#6f88aa' }}>
                    {isEn ? 'Blue: generation score, Red: eval loss' : '蓝色：生成分数，红色：评估损失'}
                  </div>
                </div>
              )}
              {benchmarkStatus?.rows?.length > 0 && (
                <div style={{ marginTop: 10, fontSize: 11, color: '#8ea5c5', lineHeight: 1.6 }}>
                  {isEn ? 'Latest benchmark preview:' : '最近基准预览:'} {benchmarkStatus.rows[0]?.generated_preview || '---'}
                </div>
              )}
              {trainingStatus?.error && (
                <div style={{ marginTop: 10, fontSize: 12, color: '#ff9090' }}>{trainingStatus.error}</div>
              )}
            </div>

            {memoryStatus?.error && (
              <div style={{ padding: 12, background: 'rgba(255,50,50,0.08)', border: '1px solid #6b1f1f', borderRadius: 8, color: '#ff9090', fontSize: 12, marginBottom: 12 }}>
                {memoryStatus.error}
              </div>
            )}

            <div style={S.metricGrid}>
              <Metric label={isEn ? 'Cycles' : '周期数'} value={memoryStatus?.total_sleep_cycles ?? 0} />
              <Metric label={isEn ? 'Pre-Curvature' : '固化前曲率'} value={typeof memoryStatus?.pre_sleep_curvature === 'number' ? memoryStatus.pre_sleep_curvature.toFixed(4) : '---'} />
              <Metric label={isEn ? 'Post-Curvature' : '固化后曲率'} value={typeof memoryStatus?.post_sleep_curvature === 'number' ? memoryStatus.post_sleep_curvature.toFixed(4) : '---'} />
            </div>

            <div style={{ ...S.card, marginBottom: 0, marginTop: 14 }}>
              <div style={{ ...S.cardTitle, display: 'flex', justifyContent: 'space-between' }}>
                <span>{isEn ? 'Curvature Convergence' : '曲率收敛'}</span>
                <button
                  onClick={refreshMemoryChart}
                  style={{ background: 'none', border: 'none', color: '#888', cursor: 'pointer', fontSize: 10, display: 'flex', alignItems: 'center', gap: 4 }}
                >
                  <RefreshCw size={10} /> {isEn ? 'Refresh' : '刷新'}
                </button>
              </div>

              {memoryStatus?.curvature_history?.length > 0 ? (
                <div style={{ display: 'flex', alignItems: 'flex-end', gap: 1, height: 100 }}>
                  {memoryStatus.curvature_history.map((value, index) => {
                    const max = Math.max(...memoryStatus.curvature_history, 0.001);
                    return (
                      <div
                        key={index}
                        style={{
                          flex: 1,
                          background: '#a855f766',
                          borderRadius: '3px 3px 0 0',
                          height: `${Math.max(4, (value / max) * 100)}%`,
                        }}
                        title={`Step ${index}: ${Number(value).toFixed(4)}`}
                      />
                    );
                  })}
                </div>
              ) : (
                <div style={{ height: 100, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#444', fontSize: 10, fontWeight: 700 }}>
                  {isEn ? 'No data yet. Click "Run Consolidation".' : '暂无数据，点击“执行固化”开始。'}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ICSPBPanel;
