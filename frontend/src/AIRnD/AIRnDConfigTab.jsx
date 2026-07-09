import { useState, useEffect, useCallback } from 'react';
import { Plus, Trash2, Save, ChevronDown, ChevronRight, Eye, EyeOff, Brain, Search } from 'lucide-react';
import { DEFAULT_MASTER_MODEL, DEFAULT_ANALYST_MODELS, API_TYPES } from './aiRnDConfig';
import { beginBackendRequest, clearBackendUnavailable, isFetchNetworkError, markBackendUnavailable } from '../utils/backendAvailability';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

export const AIRnDConfigTab = () => {
  const [masterModel, setMasterModel] = useState(DEFAULT_MASTER_MODEL);
  const [analystModels, setAnalystModels] = useState(DEFAULT_ANALYST_MODELS);
  const [expandedModel, setExpandedModel] = useState(null);
  const [showKeys, setShowKeys] = useState({});
  const [saveStatus, setSaveStatus] = useState('');
  const [testResults, setTestResults] = useState({});

  // Load config from backend on mount
  useEffect(() => {
    fetchConfig();
  }, []);

  const fetchConfig = useCallback(async () => {
    if (!beginBackendRequest()) return;
    try {
      const res = await fetch(`${API_BASE}/api/ai-rnd/config`);
      if (res.ok) {
        clearBackendUnavailable();
        const data = await res.json();
        if (data.master_model) setMasterModel(data.master_model);
        if (data.analyst_models) setAnalystModels(data.analyst_models);
      }
    } catch (e) {
      if (isFetchNetworkError(e)) markBackendUnavailable();
      // Use defaults
    }
  }, []);

  const saveConfig = useCallback(async () => {
    if (!beginBackendRequest(undefined, true)) {
      setSaveStatus('后端未连接，请先启动 localhost:5001');
      return;
    }
    try {
      const res = await fetch(`${API_BASE}/api/ai-rnd/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ master_model: masterModel, analyst_models: analystModels }),
      });
      if (res.ok) {
        clearBackendUnavailable();
        setSaveStatus('已保存');
        setTimeout(() => setSaveStatus(''), 2000);
      } else {
        setSaveStatus('保存失败');
      }
    } catch (e) {
      if (isFetchNetworkError(e)) markBackendUnavailable();
      setSaveStatus(`保存失败: ${e.message}`);
    }
  }, [masterModel, analystModels]);

  const addAnalyst = () => {
    const newModel = {
      name: `分析师-${analystModels.length + 1}`,
      model_type: 'analyst',
      api_type: 'openai',
      api_base: 'https://api.openai.com/v1',
      api_key: '',
      model_id: 'gpt-4o-mini',
      analysis_prompt: DEFAULT_ANALYST_MODELS[0].analysis_prompt,
    };
    setAnalystModels(prev => [...prev, newModel]);
    setExpandedModel(`analyst-${analystModels.length}`);
  };

  const removeAnalyst = (idx) => {
    setAnalystModels(prev => prev.filter((_, i) => i !== idx));
    setExpandedModel(prev => {
      if (prev === `analyst-${idx}`) return null;
      if (prev?.startsWith('analyst-')) {
        const currentIdx = Number(prev.replace('analyst-', ''));
        if (Number.isFinite(currentIdx) && currentIdx > idx) return `analyst-${currentIdx - 1}`;
      }
      return prev;
    });
    setTestResults(prev => {
      const next = {};
      Object.entries(prev).forEach(([key, value]) => {
        if (!key.startsWith('analyst-')) {
          next[key] = value;
          return;
        }
        const currentIdx = Number(key.replace('analyst-', ''));
        if (!Number.isFinite(currentIdx) || currentIdx === idx) return;
        next[`analyst-${currentIdx > idx ? currentIdx - 1 : currentIdx}`] = value;
      });
      return next;
    });
  };

  const updateAnalyst = (idx, field, value) => {
    setAnalystModels(prev => prev.map((m, i) => i === idx ? { ...m, [field]: value } : m));
  };

  const updateMaster = (field, value) => {
    setMasterModel(prev => ({ ...prev, [field]: value }));
  };

  const testModelConfig = useCallback(async (key, model) => {
    if (!beginBackendRequest(undefined, true)) {
      setTestResults(prev => ({
        ...prev,
        [key]: {
          status: 'error',
          message: '后端未连接，请先启动 localhost:5001',
          details: {
            endpoint: `${API_BASE}/api/ai-rnd/config/test`,
            error_type: 'BackendUnavailableCooldown',
          },
        },
      }));
      return;
    }
    setTestResults(prev => ({
      ...prev,
      [key]: { status: 'testing', message: '测试中...' },
    }));
    try {
      const res = await fetch(`${API_BASE}/api/ai-rnd/config/test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.ok) {
        clearBackendUnavailable();
        const latency = data.latency_ms != null ? `，${data.latency_ms}ms` : '';
        setTestResults(prev => ({
          ...prev,
          [key]: {
            status: 'success',
            message: `测试成功${latency}`,
            sample: data.sample || '',
          },
        }));
      } else {
        setTestResults(prev => ({
          ...prev,
          [key]: {
            status: 'error',
            message: data.message || `测试失败：HTTP ${res.status}`,
            details: data.details || {
              http_status: res.status,
              response_body: JSON.stringify(data),
            },
          },
        }));
      }
    } catch (e) {
      if (isFetchNetworkError(e)) markBackendUnavailable();
      setTestResults(prev => ({
        ...prev,
        [key]: {
          status: 'error',
          message: `测试失败：${e.message}`,
          details: { error_type: e.name || 'Error' },
        },
      }));
    }
  }, []);

  return (
    <div style={{ flex: 1, padding: '40px clamp(18px, 5vw, 60px)', overflowY: 'auto' }} className="airnd-scroll">
      <h2 style={{ fontSize: '24px', fontWeight: '800', color: '#a78bfa', marginBottom: 8, letterSpacing: '1px' }}>
        模型配置
      </h2>
      <p style={{ fontSize: '13px', color: '#9ca3af', marginBottom: 30 }}>
        配置AI主模型和分析模型的API接口、Key和提示词，控制自动研发循环的行为。
      </p>

      {/* Master Model */}
      <ModelConfigCard
        title="AI主模型（规划 & 代码生成）"
        icon={Brain}
        iconColor="#a78bfa"
        model={masterModel}
        onUpdate={updateMaster}
        expanded={expandedModel === 'master'}
        onToggle={() => setExpandedModel(expandedModel === 'master' ? null : 'master')}
        showKey={showKeys['master']}
        onToggleKey={() => setShowKeys(prev => ({ ...prev, master: !prev.master }))}
        onTest={() => testModelConfig('master', masterModel)}
        testState={testResults.master}
        isMaster={true}
      />

      {/* Analyst Models */}
      <div style={{ marginTop: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
          <h3 style={{ fontSize: '16px', color: '#d1d5db', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Search size={16} color="#a78bfa" style={{ filter: 'drop-shadow(0 0 3px #a78bfa)' }} />
            分析师模型
          </h3>
          <button onClick={addAnalyst} style={{
            background: 'rgba(167, 139, 250, 0.12)', border: '1px solid rgba(167, 139, 250, 0.3)',
            color: '#a78bfa', borderRadius: '8px', padding: '8px 16px', cursor: 'pointer',
            fontSize: '12px', display: 'flex', alignItems: 'center', gap: 6,
            fontWeight: 'bold', transition: 'all 0.2s',
          }}
          onMouseEnter={e => { 
            e.currentTarget.style.background = 'rgba(167, 139, 250, 0.2)'; 
            e.currentTarget.style.borderColor = 'rgba(167, 139, 250, 0.5)'; 
          }}
          onMouseLeave={e => { 
            e.currentTarget.style.background = 'rgba(167, 139, 250, 0.12)'; 
            e.currentTarget.style.borderColor = 'rgba(167, 139, 250, 0.3)'; 
          }}
          >
            <Plus size={14} style={{ filter: 'drop-shadow(0 0 3px #a78bfa)' }} /> 添加分析师
          </button>
        </div>

        {analystModels.map((model, idx) => (
          <div key={idx}>
            <ModelConfigCard
              title={model.name}
              icon={Search}
              iconColor="#38bdf8"
              model={model}
              onUpdate={(field, value) => updateAnalyst(idx, field, value)}
              expanded={expandedModel === `analyst-${idx}`}
              onToggle={() => setExpandedModel(expandedModel === `analyst-${idx}` ? null : `analyst-${idx}`)}
              showKey={showKeys[`analyst-${idx}`]}
              onToggleKey={() => setShowKeys(prev => ({ ...prev, [`analyst-${idx}`]: !prev[`analyst-${idx}`] }))}
              onTest={() => testModelConfig(`analyst-${idx}`, model)}
              testState={testResults[`analyst-${idx}`]}
              onRemove={() => removeAnalyst(idx)}
            />
          </div>
        ))}
      </div>

      {/* Save Button */}
      <div style={{ marginTop: 32, display: 'flex', alignItems: 'center', gap: 16 }}>
        <button onClick={saveConfig} style={{
          background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', border: 'none',
          color: '#fff', borderRadius: '10px', padding: '12px 36px', cursor: 'pointer',
          fontSize: '14px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: 8,
          boxShadow: '0 4px 14px rgba(99, 102, 241, 0.3)',
          transition: 'all 0.25s ease',
        }}
        onMouseEnter={e => { 
          e.currentTarget.style.transform = 'translateY(-1px)'; 
          e.currentTarget.style.boxShadow = '0 6px 20px rgba(99, 102, 241, 0.45)'; 
        }}
        onMouseLeave={e => { 
          e.currentTarget.style.transform = 'translateY(0)'; 
          e.currentTarget.style.boxShadow = '0 4px 14px rgba(99, 102, 241, 0.3)'; 
        }}
        >
          <Save size={16} style={{ filter: 'drop-shadow(0 0 4px rgba(255,255,255,0.5))' }} /> 保存配置
        </button>
        {saveStatus && (
          <span style={{ fontSize: '13px', fontWeight: 'bold', color: saveStatus.includes('失败') ? '#ef4444' : '#00ff88' }}>
            {saveStatus}
          </span>
        )}
      </div>
    </div>
  );
};

const ModelConfigCard = ({
  title,
  icon: Icon,
  iconColor,
  model,
  onUpdate,
  expanded,
  onToggle,
  showKey,
  onToggleKey,
  isMaster,
  onTest,
  testState,
  onRemove,
}) => {
  const promptFields = isMaster
    ? [
        { key: 'analysis_prompt', label: '分析提示词' },
        { key: 'planning_prompt', label: '规划提示词' },
        { key: 'code_gen_prompt', label: '代码生成提示词' },
        { key: 'summary_prompt', label: '总结提示词' },
      ]
    : [{ key: 'analysis_prompt', label: '分析提示词' }];

  return (
    <div 
      className="airnd-card"
      style={{
        background: 'linear-gradient(135deg, rgba(167, 139, 250, 0.08) 0%, rgba(99, 102, 241, 0.02) 100%)', 
        borderRadius: '16px',
        border: '1px solid rgba(167, 139, 250, 0.22)', 
        overflow: 'hidden',
        marginBottom: '16px',
      }}
    >
      {/* Header */}
      <button onClick={onToggle} style={{
        width: '100%', padding: '16px 20px', background: 'none', border: 'none',
        color: '#f3f4f6', cursor: 'pointer', display: 'flex', alignItems: 'center',
        justifyContent: 'space-between', fontSize: '15px', fontWeight: 'bold',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {Icon && <Icon size={16} color={iconColor || '#a78bfa'} style={{ filter: `drop-shadow(0 0 3px ${iconColor || '#a78bfa'})` }} />}
          <span>{title}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: '12px', color: '#9ca3af', fontFamily: 'monospace', background: 'rgba(0,0,0,0.25)', padding: '2px 8px', borderRadius: '4px' }}>{model.model_id}</span>
          {expanded ? <ChevronDown size={16} color="#a78bfa" style={{ filter: 'drop-shadow(0 0 3px #a78bfa)' }} /> : <ChevronRight size={16} color="#a78bfa" style={{ filter: 'drop-shadow(0 0 3px #a78bfa)' }} />}
        </div>
      </button>

      {/* Expanded content */}
      {expanded && (
        <div style={{ padding: '0 20px 20px', display: 'flex', flexDirection: 'column', gap: 14 }}>
          {/* Basic config row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 12 }}>
            <Field label="名称" value={model.name} onChange={v => onUpdate('name', v)} />
            <Field label="模型ID" value={model.model_id} onChange={v => onUpdate('model_id', v)} />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 12 }}>
            <Field
              label="API类型"
              value={model.api_type}
              onChange={v => {
                onUpdate('api_type', v);
                const apiType = API_TYPES.find(a => a.id === v);
                if (apiType) {
                  onUpdate('api_base', apiType.prefix);
                  if (apiType.defaultModel) onUpdate('model_id', apiType.defaultModel);
                }
              }}
              type="select"
              options={API_TYPES.map(a => ({ value: a.id, label: a.label }))}
            />
            <Field label="API地址" value={model.api_base} onChange={v => onUpdate('api_base', v)} />
          </div>

          <div style={{ position: 'relative' }}>
            <Field
              label="API Key"
              value={model.api_key}
              onChange={v => onUpdate('api_key', v)}
              type={showKey ? 'text' : 'password'}
              placeholder="sk-..."
            />
            <button
              onClick={onToggleKey}
              style={{
                position: 'absolute', right: 12, top: 32, background: 'none', border: 'none',
                color: '#9ca3af', cursor: 'pointer',
              }}
            >
              {showKey ? <EyeOff size={15} style={{ filter: 'drop-shadow(0 0 3px rgba(156,163,175,0.6))' }} /> : <Eye size={15} style={{ filter: 'drop-shadow(0 0 3px rgba(156,163,175,0.6))' }} />}
            </button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button
                type="button"
                onClick={onTest}
                disabled={testState?.status === 'testing'}
                style={{
                  background: testState?.status === 'testing' ? 'rgba(167,139,250,0.08)' : 'rgba(167, 139, 250, 0.14)',
                  border: '1px solid rgba(167, 139, 250, 0.35)',
                  color: '#c4b5fd',
                  borderRadius: '8px',
                  padding: '8px 12px',
                  cursor: testState?.status === 'testing' ? 'wait' : 'pointer',
                  fontSize: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  fontWeight: 'bold',
                  opacity: testState?.status === 'testing' ? 0.72 : 1,
                }}
              >
                <Search size={14} />
                {testState?.status === 'testing' ? '测试中' : '测试配置'}
              </button>

              {onRemove && (
                <button
                  type="button"
                  onClick={onRemove}
                  style={{
                    background: 'rgba(239, 68, 68, 0.1)',
                    border: '1px solid rgba(239, 68, 68, 0.28)',
                    color: '#fca5a5',
                    borderRadius: '8px',
                    padding: '8px 12px',
                    cursor: 'pointer',
                    fontSize: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    fontWeight: 'bold',
                  }}
                >
                  <Trash2 size={14} />
                  删除模型
                </button>
              )}
            </div>

            {testState && (
              <div style={{
                padding: '8px 10px',
                borderRadius: '8px',
                background: testState.status === 'success'
                  ? 'rgba(34, 197, 94, 0.08)'
                  : testState.status === 'error'
                    ? 'rgba(239, 68, 68, 0.08)'
                    : 'rgba(167, 139, 250, 0.08)',
                border: `1px solid ${testState.status === 'success'
                  ? 'rgba(34, 197, 94, 0.22)'
                  : testState.status === 'error'
                    ? 'rgba(239, 68, 68, 0.22)'
                    : 'rgba(167, 139, 250, 0.22)'}`,
                color: testState.status === 'success'
                  ? '#86efac'
                  : testState.status === 'error'
                    ? '#fca5a5'
                    : '#c4b5fd',
                fontSize: '12px',
                lineHeight: 1.5,
                wordBreak: 'break-word',
              }}>
                <div>{testState.message}</div>
                {testState.status === 'error' && testState.details && (
                  <div style={{
                    marginTop: 6,
                    paddingTop: 6,
                    borderTop: '1px solid rgba(252, 165, 165, 0.18)',
                    color: '#fca5a5',
                    fontFamily: 'monospace',
                    fontSize: '11px',
                    lineHeight: 1.55,
                    whiteSpace: 'pre-wrap',
                  }}>
                    {[
                      ['API类型', testState.details.api_type],
                      ['接口地址', testState.details.endpoint],
                      ['模型ID', testState.details.model_id],
                      ['HTTP状态', testState.details.http_status],
                      ['错误类型', testState.details.error_type],
                      ['响应内容', testState.details.response_body],
                    ]
                      .filter(([, value]) => value !== undefined && value !== null && value !== '')
                      .map(([label, value]) => `${label}: ${value}`)
                      .join('\n')}
                  </div>
                )}
                {testState.sample && (
                  <div style={{ marginTop: 4, color: '#9ca3af', fontFamily: 'monospace' }}>
                    返回：{testState.sample}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Prompt fields */}
          <div style={{ borderTop: '1px solid rgba(167, 139, 250, 0.15)', paddingTop: 16, marginTop: 8 }}>
            <div style={{ fontSize: '12px', color: '#c084fc', marginBottom: 14, fontWeight: 'bold', letterSpacing: '0.5px' }}>提示词配置</div>
            {promptFields.map(pf => (
              <div key={pf.key} style={{ marginBottom: 14 }}>
                <label style={{ fontSize: '11px', color: '#9ca3af', marginBottom: 6, display: 'block', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{pf.label}</label>
                <textarea
                  value={model[pf.key] || ''}
                  onChange={e => onUpdate(pf.key, e.target.value)}
                  className="airnd-input"
                  style={{
                    width: '100%', minHeight: 90, padding: '10px 14px',
                    background: 'rgba(0,0,0,0.35)', border: '1px solid rgba(255,255,255,0.08)',
                    borderRadius: '8px', color: '#d1d5db', fontSize: '12px', fontFamily: 'monospace',
                    resize: 'vertical', outline: 'none', lineHeight: '1.5', boxSizing: 'border-box',
                  }}
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const Field = ({ label, value, onChange, type = 'text', options, placeholder }) => {
  if (type === 'select') {
    return (
      <div style={{ minWidth: 0 }}>
        <label style={{ fontSize: '11px', color: '#9ca3af', marginBottom: 6, display: 'block', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{label}</label>
        <select
          value={value}
          onChange={e => onChange(e.target.value)}
          className="airnd-input"
          style={{
            width: '100%', padding: '10px 14px', background: 'rgba(0,0,0,0.3)',
            border: '1px solid rgba(255,255,255,0.08)', borderRadius: '8px',
            color: '#e5e7eb', fontSize: '13px', outline: 'none', boxSizing: 'border-box',
          }}
        >
          {options?.map(o => (
            <option key={o.value} value={o.value} style={{ background: '#111827', color: '#e5e7eb' }}>{o.label}</option>
          ))}
        </select>
      </div>
    );
  }
  return (
    <div style={{ minWidth: 0 }}>
      <label style={{ fontSize: '11px', color: '#9ca3af', marginBottom: 6, display: 'block', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{label}</label>
      <input
        type={type}
        value={value || ''}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        className="airnd-input"
        style={{
          width: '100%', padding: '10px 14px', background: 'rgba(0,0,0,0.3)',
          border: '1px solid rgba(255,255,255,0.08)', borderRadius: '8px',
          color: '#e5e7eb', fontSize: '13px', outline: 'none', boxSizing: 'border-box',
        }}
      />
    </div>
  );
};
