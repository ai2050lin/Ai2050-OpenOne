import React, { useState } from 'react';
import { BrainCircuit, Cpu, Sparkles } from 'lucide-react';
import { TheoryRouteDashboard } from './TheoryRouteDashboard';

const MODEL_TABS = [
    { id: 'GPT5', label: 'GPT5', icon: BrainCircuit },
    { id: 'GLM5', label: 'GLM5', icon: Cpu },
    { id: 'Gemini', label: 'GEMINI', icon: Sparkles },
];

export const DeepAnalysisTab = () => {
    const [activeModelTab, setActiveModelTab] = useState('GPT5');

    return (
        <section style={{ maxWidth: '1180px', margin: '0 auto', animation: 'roadmapFade 0.35s ease-out' }}>
            <header style={{ marginBottom: '18px' }}>
                <div style={{ color: '#f8fafc', fontSize: '22px', fontWeight: 750, marginBottom: '6px' }}>
                    智能理论：三路线证据面板
                </div>
                <div style={{ color: '#94a3b8', fontSize: '13px', lineHeight: 1.7 }}>
                    分别读取各路线研究记录，统一展示证据等级、已确认边界与下一项可证伪任务。不同路线不共享虚构进度值。
                </div>
            </header>

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
