import { API_BASE } from './config.js';
import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { Activity, Bot, Loader2, RefreshCw, Send, User } from 'lucide-react';
import { SimplePanel } from './SimplePanel';


export function AGIChatPanel({ onClose }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [status, setStatus] = useState({
    is_ready: false,
    status_msg: '正在检查语言能力服务...',
    model_family: '',
    consistency_mode: '',
  });
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const checkStatus = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/agi_chat/status`);
      setStatus({
        is_ready: !!res.data?.is_ready,
        status_msg: res.data?.status_msg || '服务状态未知',
        model_family: res.data?.model_family || '',
        consistency_mode: res.data?.consistency_mode || '',
      });
    } catch (error) {
      console.error(error);
      setStatus({
        is_ready: false,
        status_msg: '语言能力服务离线',
        model_family: '',
        consistency_mode: '',
      });
    }
  };

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const handleSend = async () => {
    if (!input.trim() || !status.is_ready || isTyping) {
      return;
    }

    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setIsTyping(true);

    try {
      const res = await axios.post(`${API_BASE}/api/agi_chat/generate`, {
        prompt: userMsg,
        max_new_tokens: 64,
      });

      if (res.data?.generated_text) {
        setMessages(prev => [
          ...prev,
          {
            role: 'agi',
            content: res.data.generated_text,
            meta: res.data?.icspb_metrics || null,
          },
        ]);
      } else {
        setMessages(prev => [...prev, { role: 'sys', content: '模型没有返回有效内容。' }]);
      }
    } catch (error) {
      const detail = error.response?.data?.detail || error.message;
      setMessages(prev => [...prev, { role: 'sys', content: `请求失败：${detail}` }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleReset = async () => {
    try {
      await axios.post(`${API_BASE}/api/agi_chat/reset`);
      setMessages([]);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <SimplePanel
      title="语言能力测试窗口"
      onClose={onClose}
      icon={<Bot size={18} />}
      style={{
        position: 'absolute',
        top: 80,
        right: 350,
        zIndex: 100,
        width: '440px',
        height: '560px',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <div
        style={{
          padding: '8px 12px',
          background: 'rgba(0,0,0,0.3)',
          borderBottom: '1px solid #333',
          fontSize: '12px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              color: status.is_ready ? '#10b981' : '#f59e0b',
            }}
          >
            <Activity size={14} />
            {status.status_msg}
          </div>
          {(status.model_family || status.consistency_mode) && (
            <div style={{ color: '#9ca3af', fontSize: '11px' }}>
              {status.model_family}
              {status.model_family && status.consistency_mode ? ' · ' : ''}
              {status.consistency_mode}
            </div>
          )}
        </div>
        <button
          onClick={handleReset}
          style={{ background: 'transparent', border: 'none', color: '#888', cursor: 'pointer' }}
          title="清空测试会话"
        >
          <RefreshCw size={14} />
        </button>
      </div>

      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '12px',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
        }}
      >
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', color: '#888', marginTop: '40px', fontSize: '12px', lineHeight: '1.7' }}>
            <p>当前窗口用于测试原型模型的语言理解、生成和连续对话能力。</p>
            <p>建议先测试：概念解释、改写、比较、短推理、多轮追问。</p>
            <p>如果服务未就绪，请先确认后端语言能力接口已经启动。</p>
          </div>
        )}

        {messages.map((message, index) => (
          <div
            key={index}
            style={{
              alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: '88%',
              background:
                message.role === 'user'
                  ? '#4488ff'
                  : message.role === 'agi'
                    ? 'rgba(255,255,255,0.10)'
                    : 'rgba(255,0,0,0.18)',
              padding: '8px 12px',
              borderRadius: '8px',
              borderBottomRightRadius: message.role === 'user' ? 0 : '8px',
              borderBottomLeftRadius: message.role === 'agi' ? 0 : '8px',
              fontSize: '13px',
              lineHeight: '1.5',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            <div
              style={{
                fontSize: '10px',
                color: 'rgba(255,255,255,0.55)',
                marginBottom: '4px',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
              }}
            >
              {message.role === 'user' ? <User size={10} /> : message.role === 'agi' ? <Bot size={10} /> : <Activity size={10} />}
              {message.role === 'user' ? '用户' : message.role === 'agi' ? '模型' : '系统'}
            </div>
            {message.content}
            {message.meta && (
              <div style={{ marginTop: '6px', fontSize: '10px', color: '#9ca3af' }}>
                CA={Number(message.meta.conscious_access || 0).toFixed(3)} · TS={Number(message.meta.theorem_survival || 0).toFixed(3)}
              </div>
            )}
          </div>
        ))}

        {isTyping && (
          <div
            style={{
              alignSelf: 'flex-start',
              color: '#888',
              fontSize: '12px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
            }}
          >
            <Loader2 size={12} className="animate-spin" />
            正在生成回复...
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div style={{ padding: '12px', borderTop: '1px solid #333', display: 'flex', gap: '8px' }}>
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          placeholder={status.is_ready ? '输入一句话，测试语言理解与生成能力...' : '等待语言能力服务就绪...'}
          disabled={!status.is_ready || isTyping}
          style={{
            flex: 1,
            background: 'rgba(0,0,0,0.2)',
            border: '1px solid #444',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '4px',
            outline: 'none',
            fontSize: '13px',
          }}
        />
        <button
          onClick={handleSend}
          disabled={!status.is_ready || isTyping || !input.trim()}
          style={{
            background: status.is_ready && input.trim() && !isTyping ? '#4488ff' : '#333',
            border: 'none',
            color: 'white',
            padding: '0 12px',
            borderRadius: '4px',
            cursor: status.is_ready && input.trim() && !isTyping ? 'pointer' : 'not-allowed',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Send size={16} />
        </button>
      </div>
    </SimplePanel>
  );
}
