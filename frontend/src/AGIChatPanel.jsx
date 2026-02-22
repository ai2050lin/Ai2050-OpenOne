import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, RefreshCw, Bot, User, Activity, Loader2 } from 'lucide-react';
import { SimplePanel } from './SimplePanel';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

export function AGIChatPanel({ onClose, t }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [status, setStatus] = useState({ is_ready: false, status_msg: 'Checking...' });
    const [isTyping, setIsTyping] = useState(false);
    const messagesEndRef = useRef(null);

    const checkStatus = async () => {
        try {
            const res = await axios.get(`${API_BASE}/api/agi_chat/status`);
            setStatus(res.data);
        } catch (e) {
            console.error(e);
            setStatus({ is_ready: false, status_msg: 'Server Offline' });
        }
    };

    useEffect(() => {
        checkStatus();
        const interval = setInterval(checkStatus, 3000);
        return () => clearInterval(interval);
    }, []);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isTyping]);

    const handleSend = async () => {
        if (!input.trim() || !status.is_ready) return;

        const userMsg = input.trim();
        setInput('');
        setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
        setIsTyping(true);

        try {
            const res = await axios.post(`${API_BASE}/api/agi_chat/generate`, {
                prompt: userMsg,
                max_tokens: 30
            });

            if (res.data && res.data.generated_text) {
                setMessages(prev => [...prev, { role: 'agi', content: res.data.generated_text }]);
            }
        } catch (e) {
            setMessages(prev => [...prev, { role: 'sys', content: `Error: ${e.response?.data?.detail || e.message}` }]);
        } finally {
            setIsTyping(false);
        }
    };

    const handleReset = async () => {
        try {
            await axios.post(`${API_BASE}/api/agi_chat/reset`);
            setMessages([]);
        } catch (e) {
            console.error(e);
        }
    };

    return (
        <SimplePanel
            title="AGI Physics Engine Chat"
            onClose={onClose}
            icon={<Bot size={18} />}
            style={{
                position: 'absolute', top: 80, right: 350, zIndex: 100,
                width: '380px', height: '500px', display: 'flex', flexDirection: 'column'
            }}
        >
            <div style={{ padding: '8px 12px', background: 'rgba(0,0,0,0.3)', borderBottom: '1px solid #333', fontSize: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: status.is_ready ? '#10b981' : '#f59e0b' }}>
                    <Activity size={14} />
                    {status.status_msg}
                </div>
                <button onClick={handleReset} style={{ background: 'transparent', border: 'none', color: '#888', cursor: 'pointer' }} title="Clear Working Memory">
                    <RefreshCw size={14} />
                </button>
            </div>

            <div style={{ flex: 1, overflowY: 'auto', padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {messages.length === 0 && (
                    <div style={{ textAlign: 'center', color: '#666', marginTop: '40px', fontSize: '12px' }}>
                        <p>O(1) 相变坍塌推理引擎交互终端已挂载。</p>
                        <p>基于 50257 维纯数学网络，完全脱离反向传播与注意力机制。请注入前概念刺激脉冲 (Prompt)：</p>
                    </div>
                )}
                {messages.map((m, i) => (
                    <div key={i} style={{
                        alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
                        maxWidth: '85%',
                        background: m.role === 'user' ? '#4488ff' : (m.role === 'agi' ? 'rgba(255,255,255,0.1)' : 'rgba(255,0,0,0.2)'),
                        padding: '8px 12px',
                        borderRadius: '8px',
                        borderBottomRightRadius: m.role === 'user' ? 0 : '8px',
                        borderBottomLeftRadius: m.role === 'agi' ? 0 : '8px',
                        fontSize: '13px',
                        lineHeight: '1.4',
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word'
                    }}>
                        <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.5)', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                            {m.role === 'user' ? <User size={10} /> : (m.role === 'agi' ? <Bot size={10} /> : <Activity size={10} />)}
                            {m.role === 'user' ? 'You' : (m.role === 'agi' ? 'AGI (O(1) Collapse)' : 'System')}
                        </div>
                        {m.content}
                    </div>
                ))}
                {isTyping && (
                    <div style={{ alignSelf: 'flex-start', color: '#888', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Loader2 size={12} className="animate-spin" /> Collapsing Manifold...
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
                    placeholder={status.is_ready ? "Inject stimulation pulse..." : "Waiting for streaming wash..."}
                    disabled={!status.is_ready || isTyping}
                    style={{
                        flex: 1, background: 'rgba(0,0,0,0.2)', border: '1px solid #444',
                        color: 'white', padding: '8px 12px', borderRadius: '4px', outline: 'none',
                        fontSize: '13px'
                    }}
                />
                <button
                    onClick={handleSend}
                    disabled={!status.is_ready || isTyping || !input.trim()}
                    style={{
                        background: (status.is_ready && input.trim() && !isTyping) ? '#4488ff' : '#333',
                        border: 'none', color: 'white', padding: '0 12px', borderRadius: '4px',
                        cursor: (status.is_ready && input.trim() && !isTyping) ? 'pointer' : 'not-allowed',
                        display: 'flex', alignItems: 'center', justifyContent: 'center'
                    }}
                >
                    <Send size={16} />
                </button>
            </div>
        </SimplePanel>
    );
}
