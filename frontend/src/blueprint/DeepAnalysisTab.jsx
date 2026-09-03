import React, { useState } from 'react';
import { BrainCircuit, Cpu, Sparkles } from 'lucide-react';
import { TheoryRouteDashboard } from './TheoryRouteDashboard';
import { useResearchSnapshot } from '../researchKernel/useResearchSnapshot';

const MODEL_TABS = [
    { id: 'GPT5', label: 'GPT5', icon: BrainCircuit },
    { id: 'GLM5', label: 'GLM5', icon: Cpu },
    { id: 'Gemini', label: 'GEMINI', icon: Sparkles },
];

export const DeepAnalysisTab = () => {
    const [activeModelTab, setActiveModelTab] = useState('GPT5');
    const { snapshot, error } = useResearchSnapshot();
    const current = snapshot?.current;

    return (
        <section style={{ maxWidth: '1180px', margin: '0 auto', animation: 'roadmapFade 0.35s ease-out' }}>
            <header style={{ marginBottom: '18px' }}>
                <div style={{ color: '#f8fafc', fontSize: '22px', fontWeight: 750, marginBottom: '6px' }}>
                    智能理论：三路线证据面板
                </div>
                <div style={{ color: '#94a3b8', fontSize: '13px', lineHeight: 1.7 }}>
                    统一问题是恢复语言模式的相对编码、复用差分和条件化状态转移；三条路线分别展示证据等级、已确认边界与下一项可证伪任务。
                </div>
            </header>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 10, marginBottom: 20 }}>
                <div style={{ padding: '15px 17px', borderLeft: '3px solid #38bdf8', background: 'rgba(14,116,144,0.08)' }}>
                    <div style={{ color: '#7dd3fc', fontSize: 10, fontWeight: 800 }}>统一工作框架</div>
                    <div style={{ marginTop: 6, color: '#e0f2fe', fontSize: 13, fontWeight: 750 }}>条件化输出场闭合 · 复用—差分—条件化</div>
                    <div style={{ marginTop: 6, color: '#8fa3bd', fontSize: 11, lineHeight: 1.65 }}>语义不预设为固定向量；真正的机制对象是输入条件、内部状态、组件作用与输出竞争之间可干预的有向转移。</div>
                </div>
                <div style={{ padding: '15px 17px', borderLeft: '3px solid #f59e0b', background: 'rgba(69,26,3,0.15)' }}>
                    <div style={{ color: '#fbbf24', fontSize: 10, fontWeight: 800 }}>路线边界 / 框架视角</div>
                    <div style={{ marginTop: 6, color: '#d6c4a5', fontSize: 11, lineHeight: 1.65 }}>{current?.bottleneck || error || '正在读取 Canonical Snapshot…'}</div>
                </div>
            </div>

            <nav
                aria-label="智能理论研究路线"
                style={{ display: 'flex', gap: '6px', borderBottom: '1px solid rgba(148,163,184,0.2)', marginBottom: '20px' }}
            >
                {MODEL_TABS.map(({ id, label, icon: RouteIcon }) => {
                    const active = id === activeModelTab;
                    return (
                        <button
                            key={id}
                            type="button"
                            onClick={() => setActiveModelTab(id)}
                            aria-pressed={active}
                            style={{
                                minWidth: '120px', minHeight: '42px', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
                                border: 'none', borderBottom: active ? '2px solid #38bdf8' : '2px solid transparent',
                                background: active ? 'rgba(56,189,248,0.09)' : 'transparent', color: active ? '#e0f2fe' : '#94a3b8',
                                cursor: 'pointer', fontWeight: active ? 700 : 500,
                            }}
                        >
                            {React.createElement(RouteIcon, { size: 16, 'aria-hidden': true })}
                            {label}
                        </button>
                    );
                })}
            </nav>

            <TheoryRouteDashboard routeId={activeModelTab} />
        </section>
    );
};
