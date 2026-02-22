import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Bot, Play, X, Zap, Activity, Info, AlertTriangle, Maximize2, Minimize2, Cpu, ChevronRight } from 'lucide-react';
import { SimplePanel } from '../SimplePanel';
import { MetricCard } from './shared/DataDisplayTemplates';

// AGI Mother Engine 可视化面板 (纯物理能量坍塌推导)
export function MotherEnginePanel({ onClose, t, theme = 'dark' }) {
    const [promptText, setPromptText] = useState("The artificial");
    const [generateSteps, setGenerateSteps] = useState(15);
    const [isGenerating, setIsGenerating] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [isExpanded, setIsExpanded] = useState(false);

    // 渐进式渲染逻辑 (模拟能量流动的时间感)
    const [renderedTraces, setRenderedTraces] = useState([]);
    const [renderedText, setRenderedText] = useState('');
    const animRef = useRef(null);

    const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

    // 触发能量潮汐推导
    const handleGenerate = async () => {
        if (!promptText.trim() || isGenerating) return;

        setIsGenerating(true);
        setError(null);
        setResult(null);
        setRenderedTraces([]);
        setRenderedText('');

        try {
            const response = await axios.post(`${API_BASE}/api/mother-engine/generate`, {
                prompt: promptText,
                steps: generateSteps
            }, {
                timeout: 30000 // 赋予 30s 物理时间
            });

            if (response.data && response.data.status === 'success') {
                setResult(response.data);
                animateGeneration(response.data.traces, response.data.generated_text);
            } else {
                setError('引擎未返回预期格式。可能脱水过程存在缺失。');
            }
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.detail || err.message || '通信阻断：无法联络物理微观宇宙。');
        } finally {
            setIsGenerating(false);
        }
    };

    // 视觉特效：逐字显示
    const animateGeneration = (traces, fulllText) => {
        if (!traces || traces.length === 0) return;
        let p = 0;

        const tick = () => {
            if (p < traces.length) {
                const currentTrace = traces[p];
                setRenderedTraces(prev => [...prev, currentTrace]);
                setRenderedText(prev => prev + currentTrace.token_str);
                p++;
                animRef.current = setTimeout(tick, 200 + Math.random() * 200); // 模拟推导不均匀节律
            }
        };

        tick();
    };

    useEffect(() => {
        // 清理动画计时器
        return () => {
            if (animRef.current) clearTimeout(animRef.current);
        };
    }, []);


    const panelStyle = isExpanded
        ? { position: 'fixed', top: 20, left: 20, right: 20, bottom: 20, zIndex: 1000 }
        : { position: 'absolute', top: 60, right: 20, width: 420, zIndex: 100, maxHeight: '85vh', overflowY: 'auto' };

    return (
        <SimplePanel
            title={<span style={{ display: 'flex', alignItems: 'center', gap: '8px' }} className="animate-pulse-slow"><Cpu size={16} color="#00ffcc" /> Mother Engine 接入大盘</span>}
            onClose={onClose}
            icon={<Zap color="#00ffcc" />}
            style={{ ...panelStyle, background: 'rgba(10, 15, 25, 0.95)', border: '1px solid #00ffcc44', boxShadow: '0 0 20px rgba(0,255,204,0.1)' }}
        >
            {/* 顶部控制栏扩展 */}
            <div style={{ position: 'absolute', top: 12, right: 40 }}>
                <button
                    onClick={() => setIsExpanded(!isExpanded)}
                    style={{ background: 'transparent', border: 'none', color: '#888', cursor: 'pointer' }}
                >
                    {isExpanded ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
                </button>
            </div>

            {/* 一： 原理科普区 */}
            <div style={{ marginBottom: '20px', padding: '12px', background: 'rgba(0, 255, 204, 0.05)', borderRadius: '8px', borderLeft: '3px solid #00ffcc' }}>
                <h4 style={{ margin: '0 0 8px 0', fontSize: '13px', color: '#00ffcc', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Info size={14} /> 纯代数·反向投影大脑
                </h4>
                <p style={{ fontSize: '12px', color: '#aaa', margin: '0 0 8px 0', lineHeight: 1.5 }}>
                    这是一个真正的物理语言引擎。它在生成文本时 <strong>没有</strong> 调用任何传统的 Transformer Attention 或 Softmax，完全去除了 $W_q, W_k, W_v$ 的概率计算。<br /><br />
                    它的实现仅仅源于两步：<br />
                    1. 将您的输入转化为局部突触放电 $(+1.0)$。<br />
                    2. 根据长出的拓扑引力势能图谱 (P_topology)，让能量像流水一样顺着地层自然滑落，坍塌到哪个神经元就蹦出哪个词。
                </p>

                {result && result.physics_details && (
                    <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
                        <MetricCard title="基础词表受体" value={`${result.physics_details.vocab} 维`} color="#88aaff" />
                        <MetricCard title="常识引力潜空间" value={`${result.physics_details.represent_dim} 维`} color="#ff88aa" />
                    </div>
                )}
            </div>

            {/* 二： 激励输入与推导控制 */}
            <div style={{ marginBottom: '20px' }}>
                <div style={{ fontSize: '12px', color: '#888', marginBottom: '6px' }}>初始点火信标 (Prompt Trigger)</div>
                <div style={{ display: 'flex', gap: '8px' }}>
                    <input
                        type="text"
                        value={promptText}
                        onChange={e => setPromptText(e.target.value)}
                        disabled={isGenerating}
                        style={{ flex: 1, padding: '8px 12px', background: '#111', border: '1px solid #333', color: '#fff', borderRadius: '4px', outline: 'none' }}
                        placeholder="提供一两个词激发势能..."
                    />
                    <button
                        onClick={handleGenerate}
                        disabled={isGenerating || !promptText.trim()}
                        style={{
                            padding: '8px 16px', background: isGenerating ? '#333' : '#00ffcc22',
                            color: isGenerating ? '#888' : '#00ffcc', border: '1px solid ' + (isGenerating ? '#444' : '#00ffcc'),
                            borderRadius: '4px', cursor: isGenerating ? 'not-allowed' : 'pointer',
                            display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 'bold'
                        }}
                    >
                        {isGenerating ? <Activity size={16} className="animate-spin" /> : <Play size={16} />}
                        {isGenerating ? '坍塌推演中...' : '倾泻能量'}
                    </button>
                </div>

                <div style={{ marginTop: '12px', display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <span style={{ fontSize: '11px', color: '#666' }}>推演级数 (Steps): {generateSteps}</span>
                    <input
                        type="range" min="1" max="50" step="1"
                        value={generateSteps}
                        onChange={e => setGenerateSteps(parseInt(e.target.value))}
                        disabled={isGenerating}
                        style={{ flex: 1, accentColor: '#00ffcc' }}
                    />
                </div>
            </div>

            {/* 错误提示 */}
            {error && (
                <div style={{ padding: '12px', background: 'rgba(255,50,50,0.1)', border: '1px solid #ff3333', borderRadius: '6px', color: '#ffaaaa', fontSize: '12px', marginBottom: '16px', display: 'flex', gap: '8px' }}>
                    <AlertTriangle size={16} /> <div>{error}</div>
                </div>
            )}

            {/* 三： 能量瀑布流动过程展示 */}
            {(result || isGenerating || renderedTraces.length > 0) && (
                <div style={{ background: '#0a0a0f', borderRadius: '8px', border: '1px solid #222', overflow: 'hidden' }}>
                    <div style={{ padding: '8px 12px', background: '#151520', borderBottom: '1px solid #222', fontSize: '12px', color: '#88aaff', display: 'flex', justifyContent: 'space-between' }}>
                        <span>高纬能量映射轨迹 (O(1) Matrix Collapse)</span>
                        <span>{renderedTraces.length}/{generateSteps} 步</span>
                    </div>

                    <div style={{ padding: '12px', maxHeight: isExpanded ? '40vh' : '200px', overflowY: 'auto' }}>
                        <div style={{ fontSize: '14px', lineHeight: 1.6, color: '#ccc', marginBottom: '16px', fontFamily: 'monospace' }}>
                            <span style={{ color: '#00ffcc', fontWeight: 'bold' }}>{promptText}</span>
                            {renderedText && <span style={{ color: '#fff', marginLeft: '4px' }}>{renderedText}</span>}
                            {isGenerating && renderedTraces.length < generateSteps && <span className="animate-pulse" style={{ color: '#00ffcc', marginLeft: '4px' }}>_</span>}
                        </div>

                        {/* 剖析每一帧坍塌下来的势能参数 */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                            {renderedTraces.map((t, idx) => (
                                <div key={idx} style={{
                                    display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px',
                                    padding: '4px 8px', background: `rgba(0, ${Math.min(255, 100 + t.resonance_energy / 10)}, 204, 0.1)`,
                                    borderRadius: '4px', borderLeft: '2px solid #55aaff'
                                }}>
                                    <span style={{ color: '#666', width: '24px' }}>#{t.step}</span>
                                    <ChevronRight size={10} color="#55aaff" />
                                    <span style={{ color: '#55aaff', width: '60px' }}>ID: {t.token_id}</span>
                                    <span style={{ flex: 1, color: '#fff', fontWeight: 'bold' }}>'{t.token_str}'</span>
                                    <span style={{ color: '#ff88aa' }}>{t.resonance_energy?.toFixed(2)} eV</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}
        </SimplePanel>
    );
}
