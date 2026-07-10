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

    const geminiProgressHighlights = [
        '阶段1（done）：完成 MLP 稀疏激活、残差增量 SVD、流形收敛等底层结构解剖，确认“稀疏专家 + 浅层扩张 + 深层收敛”的基本图景。',
        '阶段2（done）：完成 Attention 有效秩、权重矩阵低秩/满秩对照，初步确立“低秩逻辑路由 + 满秩记忆容器”的资源分工。',
        '阶段3（in_progress）：从 Grokking、圆形流形、局部侧抑制、信用分配危机，推进到更严格的泛化相变与局部学习机制分析。',
        '阶段4（in_progress）：把 HRR、张量绑定、同步时间波、四轴正交等思想引入统一编码框架，尝试连接 DNN 与脑式编码机制。',
        '阶段5（in_progress）：扩展到开放世界接地、长期目标、可变规划链和局部脉冲区域机制，测试智能理论能否进入闭环任务。'
    ];

    const geminiRoadmapItems = [
        { id: 'G1', name: '结构解剖', status: 'done', summary: '稀疏激活、残差 SVD、拓扑收敛。' },
        { id: 'G2', name: '组件分工', status: 'done', summary: 'Attention 低秩路由，MLP 满秩记忆。' },
        { id: 'G3', name: '泛化相变', status: 'in_progress', summary: 'Grokking、圆形流形、局部涌现。' },
        { id: 'G4', name: '统一编码', status: 'in_progress', summary: 'HRR、张量绑定、脑式稀疏编码。' },
        { id: 'G5', name: '闭环智能', status: 'in_progress', summary: '接地、规划、记忆、局部脉冲机制。' },
    ];

    const geminiKeyProblems = [
        '历史内容非常多，缺少统一的结论层级：需要把实验看板、理论假设和关键证据重新压缩为少数核心路线。',
        '部分结论仍偏理论推演或演示数据，需要更严格地区分真实模型证据、toy 证据和类脑类比证据。',
        'DNN 内部机制、脑式编码机制、开放世界闭环三条路线跨度很大，需要统一评价指标和失败边界。',
        '信用分配、语义接地、长期规划和概念解绑定仍是关键未闭合问题。',
        '很多模块已经可视化，但还需要沉淀成统一数据格式，方便和 GPT5/GLM5 路线交叉验证。'
    ];

    const geminiNextSteps = [
        'P0：把 Gemini 历史实验整理成“路线-阶段-证据-缺口”结构，减少默认界面的长文本堆叠。',
        'P0：把真实模型证据和 toy/理论演示证据分层标注，避免不同证据等级混用。',
        'P1：选出 3-5 个最关键机制命题，与 GPT5 路线进行同口径复验。',
        'P1：把开放世界接地、长期目标、局部脉冲机制压缩成可比较的闭环指标。',
        'P2：把 Gemini 的类脑理论转化为可执行测试脚本和结构化结果文件，接入统一客户端图谱。'
    ];

    const statusTextMap = {
        done: '已完成',
        in_progress: '进行中',
        pending: '待开始',
    };

    const statusColorMap = {
        done: '#10b981',
        in_progress: '#f59e0b',
        pending: '#94a3b8',
    };

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
                                padding: '30px',
                                borderRadius: '24px',
                                border: '1px solid rgba(244,114,182,0.28)',
                                background: 'linear-gradient(135deg, rgba(244,114,182,0.10) 0%, rgba(168,85,247,0.04) 100%)',
                            }}
                        >
                            <div style={{ color: '#f472b6', fontWeight: 'bold', fontSize: '18px', marginBottom: '8px' }}>
                                深度神经网络逆向与类脑编码路线（Gemini）
                            </div>
                            <div style={{ color: '#fce7f3', fontSize: '13px', lineHeight: '1.7', marginBottom: '20px' }}>
                                Gemini 路线重点研究 DNN 内部结构、类脑编码、概念绑定、开放世界接地和闭环智能。当前页面按照 GPT5 tab 的结构重组：先显示核心框架和进展，再把大量历史看板折叠到历史记录中。
                            </div>

                            <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#fce7f3', marginBottom: '10px', borderBottom: '1px solid rgba(244,114,182,0.35)', paddingBottom: '8px' }}>
                                一、分析框架
                            </div>
                            <div style={{ color: '#fbcfe8', fontSize: '12px', lineHeight: '1.7', marginBottom: '14px' }}>
                                从“结构解剖”进入“机制建模”：先观察网络中知识、路由和状态更新的物理结构，再把这些结构转化为可验证的智能理论。
                            </div>
                            <div style={{ display: 'grid', gap: '6px', marginBottom: '18px' }}>
                                {[
                                    '结构层：识别 MLP 稀疏专家、残差流形、Attention 低秩路由等底层组织方式。',
                                    '机制层：分析概念绑定、正交分离、HRR 压缩、同步时间波等编码机制。',
                                    '任务层：用接地、长期目标、规划链和开放世界闭环验证理论是否能支撑智能行为。',
                                    '图谱层：把历史看板整理为阶段、证据、缺口和下一步任务，便于与 GPT5/GLM5 路线对齐。'
                                ].map((line, idx) => (
                                    <div key={idx} style={{ color: '#fce7f3', fontSize: '12px', lineHeight: '1.6' }}>
                                        {idx + 1}. {line}
                                    </div>
                                ))}
                            </div>

                            <div style={{ color: '#f9a8d4', fontWeight: 'bold', fontSize: '13px', marginBottom: '8px' }}>当前研究进展</div>
                            <div style={{ display: 'grid', gap: '6px', marginBottom: '18px' }}>
                                {geminiProgressHighlights.map((line, idx) => (
                                    <div key={idx} style={{ color: '#fbcfe8', fontSize: '12px', lineHeight: '1.6' }}>
                                        {idx + 1}. {line}
                                    </div>
                                ))}
                            </div>

                            <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#fce7f3', marginBottom: '10px', borderBottom: '1px solid rgba(244,114,182,0.35)', paddingBottom: '8px' }}>
                                二、线路图
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, minmax(120px, 1fr))', gap: '10px', marginBottom: '18px' }}>
                                {geminiRoadmapItems.map((item) => (
                                    <div key={item.id} style={{ padding: '10px', background: 'rgba(0,0,0,0.28)', borderRadius: '10px', borderTop: `2px solid ${statusColorMap[item.status] || '#94a3b8'}` }}>
                                        <div style={{ color: '#fce7f3', fontSize: '12px', fontWeight: 'bold' }}>{item.id}</div>
                                        <div style={{ color: statusColorMap[item.status] || '#94a3b8', fontSize: '11px', marginBottom: '4px' }}>[{statusTextMap[item.status] || '待开始'}]</div>
                                        <div style={{ color: '#fbcfe8', fontSize: '11px', lineHeight: '1.5', fontWeight: 'bold' }}>{item.name}</div>
                                        <div style={{ color: '#d8b4fe', fontSize: '11px', lineHeight: '1.5', marginTop: '4px' }}>{item.summary}</div>
                                    </div>
                                ))}
                            </div>

                            <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#fce7f3', marginBottom: '10px', borderBottom: '1px solid rgba(244,114,182,0.35)', paddingBottom: '8px' }}>
                                三、测试记录
                            </div>
                            <div
                                style={{
                                    padding: '16px',
                                    borderRadius: '14px',
                                    border: '1px solid rgba(168,85,247,0.24)',
                                    background: 'linear-gradient(135deg, rgba(168,85,247,0.08) 0%, rgba(168,85,247,0.02) 100%)',
                                    marginBottom: '18px',
                                }}
                            >
                                <div style={{ color: '#d8b4fe', fontWeight: 'bold', fontSize: '14px', marginBottom: '6px' }}>历史记录与完整看板</div>
                                <div style={{ color: '#c4b5fd', fontSize: '12px', lineHeight: '1.7', marginBottom: '12px' }}>
                                    Gemini 历史实验数量很多，默认不直接渲染。展开后可查看完整实验记录、阶段分析和所有历史可视化看板。
                                </div>
                                <button
                                    type="button"
                                    onClick={() => setShowGeminiHistory((value) => !value)}
                                    style={{
                                        borderRadius: '12px',
                                        border: '1px solid rgba(244,114,182,0.35)',
                                        background: showGeminiHistory ? 'rgba(244,114,182,0.16)' : 'rgba(244,114,182,0.08)',
                                        color: '#fce7f3',
                                        fontSize: '12px',
                                        cursor: 'pointer',
                                        padding: '10px 14px',
                                        fontWeight: 'bold',
                                    }}
                                >
                                    {showGeminiHistory ? '收起 Gemini 历史记录 ▲' : '展开 Gemini 历史记录 ▼'}
                                </button>
                            </div>

                            <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#fca5a5', marginBottom: '10px', borderBottom: '1px solid rgba(248,113,113,0.35)', paddingBottom: '8px' }}>
                                四、存在问题
                            </div>
                            <div style={{ display: 'grid', gap: '6px', marginBottom: '18px' }}>
                                {geminiKeyProblems.map((item, idx) => (
                                    <div key={idx} style={{ color: '#fecaca', fontSize: '12px', lineHeight: '1.6' }}>
                                        {idx + 1}. {item}
                                    </div>
                                ))}
                            </div>

                            <div style={{ fontSize: '15px', fontWeight: 'bold', color: '#86efac', marginBottom: '10px', borderBottom: '1px solid rgba(74,222,128,0.35)', paddingBottom: '8px' }}>
                                五、接下来的核心工作
                            </div>
                            <div style={{ display: 'grid', gap: '6px' }}>
                                {geminiNextSteps.map((item, idx) => (
                                    <div key={idx} style={{ color: '#dcfce7', fontSize: '12px', lineHeight: '1.6' }}>
                                        {idx + 1}. {item}
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div
                            style={{
                                borderRadius: '18px',
                                border: '1px solid rgba(255,255,255,0.1)',
                                background: 'rgba(2, 6, 23, 0.32)',
                                overflow: 'hidden',
                                display: showGeminiHistory ? 'block' : 'none',
                            }}
                        >
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
