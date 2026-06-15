import { useState } from 'react';
import { Copy, Play, CheckCircle, XCircle } from 'lucide-react';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

export const AIRnDCodeGenTab = ({ code, result }) => {
  const [copied, setCopied] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [localResult, setLocalResult] = useState(result);

  const copyCode = () => {
    navigator.clipboard.writeText(code || '');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const executeCode = async () => {
    if (!code) return;
    setExecuting(true);
    setLocalResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/ai-rnd/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code }),
      });
      const data = await res.json();
      setLocalResult(data);
    } catch (e) {
      setLocalResult({ status: 'error', output: e.message });
    }
    setExecuting(false);
  };

  const displayResult = localResult || result;

  return (
    <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
      {/* Code panel */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', borderRight: '1px solid rgba(167, 139, 250, 0.18)' }}>
        <div style={{
          padding: '16px 20px', borderBottom: '1px solid rgba(167, 139, 250, 0.15)',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          background: 'rgba(10, 10, 16, 0.45)', backdropFilter: 'blur(10px)',
        }}>
          <span style={{ fontSize: '13px', color: '#a78bfa', fontWeight: 'bold', letterSpacing: '0.5px' }}>
            生成代码 {code ? `(${code.length} 字符)` : ''}
          </span>
          <div style={{ display: 'flex', gap: 8 }}>
            <button 
              onClick={copyCode} 
              style={toolbarBtnStyle} 
              disabled={!code}
              onMouseEnter={e => { if (code) e.currentTarget.style.color = '#c084fc'; }}
              onMouseLeave={e => { if (code) e.currentTarget.style.color = '#9ca3af'; }}
            >
              <Copy size={12} style={{ filter: 'drop-shadow(0 0 3px rgba(156,163,175,0.6))' }} /> {copied ? '已复制' : '复制'}
            </button>
            <button 
              onClick={executeCode} 
              style={{
                ...toolbarBtnStyle,
                background: code ? 'rgba(16, 185, 129, 0.12)' : 'rgba(255,255,255,0.02)',
                borderColor: code ? 'rgba(16, 185, 129, 0.35)' : 'rgba(255,255,255,0.06)',
                color: code ? '#34d399' : '#4b5563',
                boxShadow: (code && !executing) ? '0 0 10px rgba(16, 185, 129, 0.15)' : 'none',
              }} 
              disabled={!code || executing}
              onMouseEnter={e => { if (code && !executing) { e.currentTarget.style.background = 'rgba(16, 185, 129, 0.2)'; e.currentTarget.style.borderColor = 'rgba(16, 185, 129, 0.5)'; } }}
              onMouseLeave={e => { if (code && !executing) { e.currentTarget.style.background = 'rgba(16, 185, 129, 0.12)'; e.currentTarget.style.borderColor = 'rgba(16, 185, 129, 0.35)'; } }}
            >
              <Play size={12} style={{ filter: code ? 'drop-shadow(0 0 4px #34d399)' : 'none' }} /> {executing ? '执行中...' : '运行代码'}
            </button>
          </div>
        </div>
        <div style={{ flex: 1, overflow: 'auto', padding: '24px 30px', background: 'rgba(10, 10, 16, 0.25)' }} className="airnd-scroll">
          {code ? (
            <pre style={{
              fontFamily: "'Cascadia Code', 'Fira Code', 'Consolas', monospace",
              fontSize: '12px', lineHeight: '1.7', color: '#e5e7eb',
              whiteSpace: 'pre-wrap', wordBreak: 'break-all',
              background: 'rgba(13, 13, 23, 0.65)',
              padding: '20px',
              borderRadius: '12px',
              border: '1px solid rgba(255, 255, 255, 0.05)',
            }}>
              {code}
            </pre>
          ) : (
            <div style={{ 
              maxWidth: '300px', margin: '100px auto', textAlign: 'center',
              padding: '40px 20px', borderRadius: '16px',
              border: '1px dashed rgba(255,255,255,0.08)', color: '#4b5563' 
            }}>
              等待主模型生成测试代码...
            </div>
          )}
        </div>
      </div>

      {/* Result panel */}
      <div style={{ width: '40%', display: 'flex', flexDirection: 'column', background: 'rgba(10, 10, 16, 0.1)' }}>
        <div style={{
          padding: '16px 20px', borderBottom: '1px solid rgba(167, 139, 250, 0.15)',
          background: 'rgba(10, 10, 16, 0.45)', backdropFilter: 'blur(10px)',
          display: 'flex', alignItems: 'center', gap: 8,
        }}>
          {displayResult?.status === 'success' ? (
            <CheckCircle size={15} color="#34d399" style={{ filter: 'drop-shadow(0 0 4px rgba(52, 211, 153, 0.4))' }} />
          ) : displayResult?.status === 'error' ? (
            <XCircle size={15} color="#ef4444" style={{ filter: 'drop-shadow(0 0 4px rgba(239, 68, 68, 0.4))' }} />
          ) : null}
          <span style={{ fontSize: '13px', color: '#9ca3af', fontWeight: 'bold', letterSpacing: '0.5px' }}>执行日志与输出</span>
        </div>
        <div style={{ flex: 1, overflow: 'auto', padding: '24px 20px' }} className="airnd-scroll">
          {displayResult ? (
            <div style={{ fontFamily: "'Cascadia Code', 'Fira Code', monospace", fontSize: '12px' }}>
              <div style={{
                padding: '10px 16px', borderRadius: '12px', marginBottom: 16,
                background: displayResult.status === 'success' 
                  ? 'linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(16, 185, 129, 0.03) 100%)' 
                  : 'linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(239, 68, 68, 0.03) 100%)',
                border: `1px solid ${displayResult.status === 'success' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                color: displayResult.status === 'success' ? '#34d399' : '#ef4444',
                fontWeight: 'bold', fontSize: '13px',
                boxShadow: displayResult.status === 'success' ? '0 4px 12px rgba(16, 185, 129, 0.05)' : '0 4px 12px rgba(239, 68, 68, 0.05)',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center'
              }}>
                <span>{displayResult.status === 'success' ? '✓ 执行完成 (SUCCESS)' : '✗ 执行异常 (ERROR)'}</span>
                {displayResult.duration && <span style={{ fontSize: '11px', opacity: 0.8 }}>耗时: {displayResult.duration}s</span>}
              </div>
              <pre style={{ 
                color: '#d1d5db', whiteSpace: 'pre-wrap', wordBreak: 'break-all',
                background: 'rgba(15, 15, 25, 0.4)', padding: '16px', borderRadius: '10px',
                border: '1px solid rgba(255,255,255,0.04)', lineHeight: '1.6'
              }}>
                {displayResult.output || displayResult.error || '无标准输出'}
              </pre>
            </div>
          ) : (
            <div style={{ 
              maxWidth: '260px', margin: '100px auto', textAlign: 'center',
              padding: '40px 20px', borderRadius: '16px',
              border: '1px dashed rgba(255,255,255,0.08)', color: '#4b5563' 
            }}>
              等待运行生成脚本以捕获输出...
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const toolbarBtnStyle = {
  background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)',
  color: '#9ca3af', borderRadius: '6px', padding: '6px 12px', cursor: 'pointer',
  fontSize: '11px', display: 'flex', alignItems: 'center', gap: 6,
  transition: 'all 0.2s ease',
  fontWeight: 'bold',
};
