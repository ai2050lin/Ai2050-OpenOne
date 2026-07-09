import React, { Suspense, lazy, useEffect, useMemo, useState } from 'react';

const GeminiTab = lazy(() => import('./GeminiTab').then((module) => ({ default: module.GeminiTab })));
const GPT5Tab = lazy(() => import('./GPT5Tab').then((module) => ({ default: module.GPT5Tab })));
const GLM5Tab = lazy(() => import('./GLM5Tab').then((module) => ({ default: module.GLM5Tab })));

export const DeepAnalysisTab = ({
    evidenceDrivenPlan,
    improvements,
    expandedImprovementPhase,
    setExpandedImprovementPhase,
    expandedImprovementTest,
    setExpandedImprovementTest,
}) => {
    const [activeModelTab, setActiveModelTab] = useState('Gemini');
    const [showGeminiHistory, setShowGeminiHistory] = useState(false);
    const modelTabs = ['Gemini', 'GPT5', 'GLM5'];
    const ActiveTabComponent = useMemo(() => {
        if (activeModelTab === 'Gemini') return GeminiTab;
        if (activeModelTab === 'GPT5') return GPT5Tab;
        return GLM5Tab;
    }, [activeModelTab]);

    useEffect(() => {
        if (activeModelTab !== 'Gemini') {
            setShowGeminiHistory(false);
        }
    }, [activeModelTab]);

    const activeTabProps = activeModelTab === 'GPT5'
        ? {
            evidenceDrivenPlan,
            improvements,
            expandedImprovementPhase,
            setExpandedImprovementPhase,
            expandedImprovementTest,
            setExpandedImprovementTest,
        }
        : {};

    return (
        <div style={{ animation: 'roadmapFade 0.6s ease-out', maxWidth: '1000px', margin: '0 auto' }}>
            <div
                style={{
                    padding: '30px',
                    borderRadius: '24px',
                    border: '1px solid rgba(244,114,182,0.28)',
                    background: 'linear-gradient(135deg, rgba(244,114,182,0.10) 0%, rgba(168,85,247,0.06) 100%)',
                    marginBottom: '28px',
                }}
            >
                <div style={{ color: '#f472b6', fontWeight: 'bold', fontSize: '24px', marginBottom: '20px' }}>
                    深度分析与模型结构对比
                </div>
                <div style={{ display: 'flex', gap: '0', marginBottom: '24px', borderBottom: '1px solid rgba(255,255,255,0.12)' }}>
                    {modelTabs.map((tab) => (
                        <button
                            key={tab}
                            onClick={() => setActiveModelTab(tab)}
                            style={{
                                padding: '12px 32px',
                                background: activeModelTab === tab
                                    ? 'linear-gradient(135deg, rgba(244,114,182,0.22) 0%, rgba(168,85,247,0.18) 100%)'
                                    : 'transparent',
                                border: 'none',
                                borderBottom: activeModelTab === tab ? '3px solid #f472b6' : '3px solid transparent',
                                color: activeModelTab === tab ? '#f9a8d4' : '#9ca3af',
                                fontSize: '15px',
                                fontWeight: activeModelTab === tab ? 'bold' : 'normal',
                                cursor: 'pointer',
                                transition: 'all 0.2s ease',
                            }}
                        >
                            {tab}
                        </button>
                    ))}
                </div>

                {/* Tab 内容区 */}
                {activeModelTab === 'Gemini' ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
                        <div
                            style={{
                                padding: '22px',
                                borderRadius: '18px',
                                border: '1px solid rgba(244,114,182,0.22)',
                                background: 'rgba(15, 23, 42, 0.38)',
                            }}
                        >
                            <div style={{ color: '#f9a8d4', fontSize: '18px', fontWeight: 900, marginBottom: '8px' }}>
                                Gemini 智能理论
                            </div>
                            <div style={{ color: '#cbd5e1', fontSize: '13px', lineHeight: 1.8, maxWidth: 820 }}>
                                Gemini 路线包含大量历史实验、阶段记录和可视化看板。为保持智能理论首页简洁，历史内容默认收起，需要查看时可在下方展开。
                            </div>
                        </div>

                        <div
                            style={{
                                borderRadius: '18px',
                                border: '1px solid rgba(255,255,255,0.1)',
                                background: 'rgba(2, 6, 23, 0.32)',
                                overflow: 'hidden',
                            }}
                        >
                            <button
                                type="button"
                                onClick={() => setShowGeminiHistory((value) => !value)}
                                style={{
                                    width: '100%',
                                    padding: '18px 22px',
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center',
                                    gap: '16px',
                                    border: 'none',
                                    background: showGeminiHistory ? 'rgba(244,114,182,0.12)' : 'rgba(255,255,255,0.03)',
                                    color: '#fff',
                                    cursor: 'pointer',
                                    fontFamily: 'inherit',
                                    textAlign: 'left',
                                }}
                            >
                                <div>
                                    <div style={{ color: '#f9a8d4', fontSize: '16px', fontWeight: 900 }}>历史记录</div>
                                    <div style={{ color: '#94a3b8', fontSize: '12px', marginTop: '4px' }}>
                                        Gemini 历史实验、阶段记录、完整看板默认不显示
                                    </div>
                                </div>
                                <div style={{ color: '#f9a8d4', fontSize: '12px', fontWeight: 800, whiteSpace: 'nowrap' }}>
                                    {showGeminiHistory ? '收起 ▲' : '展开 ▼'}
                                </div>
                            </button>

                            {showGeminiHistory && (
                                <div style={{ padding: '20px', borderTop: '1px solid rgba(255,255,255,0.08)' }}>
                                    <Suspense
                                        fallback={(
                                            <div
                                                style={{
                                                    padding: '22px',
                                                    borderRadius: '16px',
                                                    border: '1px solid rgba(255,255,255,0.08)',
                                                    background: 'rgba(255,255,255,0.03)',
                                                    color: '#cbd5e1',
                                                    fontSize: '13px',
                                                    lineHeight: '1.7',
                                                }}
                                            >
                                                正在分批加载 Gemini 历史记录...
                                            </div>
                                        )}
                                    >
                                        <GeminiTab />
                                    </Suspense>
                                </div>
                            )}
                        </div>
                    </div>
                ) : (
                    <Suspense
                        fallback={(
                            <div
                                style={{
                                    padding: '22px',
                                    borderRadius: '16px',
                                    border: '1px solid rgba(255,255,255,0.08)',
                                    background: 'rgba(255,255,255,0.03)',
                                    color: '#cbd5e1',
                                    fontSize: '13px',
                                    lineHeight: '1.7',
                                }}
                            >
                                正在分批加载 {activeModelTab} 深度分析内容...
                            </div>
                        )}
                    >
                        <ActiveTabComponent {...activeTabProps} />
                    </Suspense>
                )}
            </div>
        </div>
    );
};
