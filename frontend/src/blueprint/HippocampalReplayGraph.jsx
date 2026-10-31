import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Database, Zap, Moon, Sun, Shield, Layers } from 'lucide-react';

const HippocampalReplayGraph = () => {
    const [phase, setPhase] = useState('awake'); // 'awake' or 'sleep'

    useEffect(() => {
        const timer = setInterval(() => {
            setPhase(prev => prev === 'awake' ? 'sleep' : 'awake');
        }, 10000);
        return () => clearInterval(timer);
    }, []);

    return (
        <div style={{
            background: 'linear-gradient(145deg, rgba(8, 15, 25, 0.95) 0%, rgba(15, 20, 35, 0.95) 100%)',
            border: `1px solid ${phase === 'awake' ? 'rgba(250, 204, 21, 0.4)' : 'rgba(167, 139, 250, 0.4)'}`,
            borderRadius: '16px',
            padding: '24px',
            marginTop: '24px',
            fontFamily: 'system-ui, -apple-system, sans-serif',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
            transition: 'border-color 1s ease'
        }}>
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div style={{
                        background: phase === 'awake' ? 'rgba(250, 204, 21, 0.2)' : 'rgba(167, 139, 250, 0.2)',
                        padding: '10px', borderRadius: '10px',
                        transition: 'background 1s'
                    }}>
                        {phase === 'awake' ? <Sun size={24} color="#facc15" /> : <Moon size={24} color="#c084fc" />}
                    </div>
                    <div>
                        <h3 style={{ margin: 0, fontSize: '18px', color: '#e2e8f0', fontWeight: '800' }}>
                            上帝大坝：海马体睡眠重放固化 (Hippocampal Offline Replay)
                        </h3>
                        <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                            解开灾难性遗忘的终极方案：前线快缓存 + 绝缘冷库慢刻录
                        </div>
                    </div>
                </div>

                {/* 状态徽章 */}
                <div style={{
                    display: 'flex', alignItems: 'center', gap: '8px',
                    padding: '8px 16px', borderRadius: '20px',
                    background: phase === 'awake' ? 'rgba(250, 204, 21, 0.15)' : 'rgba(167, 139, 250, 0.15)',
                    border: `1px solid ${phase === 'awake' ? 'rgba(250, 204, 21, 0.3)' : 'rgba(167, 139, 250, 0.3)'}`
                }}>
                    <span style={{ fontSize: '13px', fontWeight: 'bold', color: phase === 'awake' ? '#fde047' : '#d8b4fe' }}>
                        当前纪元：{phase === 'awake' ? '白昼 (狂暴浅层缓存冲刷)' : '黑夜 (绝缘脱机梦境注能)'}
                    </span>
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
                {/* 左列：海马体 Hippocampus (短、灵敏) */}
                <div style={{
                    background: 'rgba(0,0,0,0.3)', borderRadius: '12px', padding: '20px',
                    border: '1px solid rgba(255,255,255,0.05)', position: 'relative'
                }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                        <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#fcd34d', display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <Zap size={16} /> 海马体缓存区 (τ=5ms, LR=极高)
                        </div>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                        <div style={{ background: 'rgba(255,255,255,0.03)', padding: '12px', borderRadius: '8px' }}>
                            <div style={{ fontSize: '12px', color: '#94a3b8', marginBottom: '8px' }}>突触印迹矩阵：[外星人] ←→ [狗]</div>
                            <div style={{ height: '6px', background: 'rgba(0,0,0,0.5)', borderRadius: '3px', overflow: 'hidden' }}>
                                <motion.div
                                    animate={{ width: phase === 'awake' ? ['20%', '100%'] : '100%' }}
                                    transition={{ duration: phase === 'awake' ? 2 : 0, ease: 'easeOut' }}
                                    style={{ height: '100%', background: '#fcd34d', borderRadius: '3px' }}
                                />
                            </div>
                            <div style={{ fontSize: '11px', color: '#fbbf24', marginTop: '6px' }}>
                                {phase === 'awake' ? '⚡ 暴露在现实下被瞬间烙印！' : '🔁 根据自底噪发散记忆共振回声 (Sharp Wave Ripples)'}
                            </div>
                        </div>
                    </div>

                    {/* 梦境传输流向指示 (仅晚上显示) */}
                    {phase === 'sleep' && (
                        <motion.div
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 20 }}
                            transition={{ duration: 1.5, repeat: Infinity }}
                            style={{
                                position: 'absolute', right: '-30px', top: '50%', transform: 'translateY(-50%)',
                                color: '#c084fc', fontSize: '24px', zIndex: 10
                            }}
                        >
                            <Layers size={32} />
                        </motion.div>
                    )}
                </div>

                {/* 右列：新皮层 Neocortex (深邃、冷酷、永存) */}
                <div style={{
                    background: 'rgba(0,0,0,0.4)', borderRadius: '12px', padding: '20px',
                    border: `1px solid ${phase === 'awake' ? 'rgba(74, 222, 128, 0.2)' : 'rgba(192, 132, 252, 0.4)'}`,
                    boxShadow: phase === 'sleep' ? '0 0 20px rgba(167, 139, 250, 0.1) inset' : 'none',
                    transition: 'all 1s'
                }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                        <div style={{ fontSize: '14px', fontWeight: 'bold', color: phase === 'awake' ? '#4ade80' : '#c084fc', display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <Database size={16} /> 新皮层深水区 (τ=50ms, LR=极低)
                        </div>
                        {phase === 'awake' && (
                            <div style={{ background: 'rgba(74, 222, 128, 0.1)', color: '#4ade80', padding: '4px 8px', borderRadius: '4px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                <Shield size={12} /> STDP 锁死免疫冲刷
                            </div>
                        )}
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                        {/* 老知识 */}
                        <div style={{ background: 'rgba(255,255,255,0.03)', padding: '12px', borderRadius: '8px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                                <span style={{ fontSize: '12px', color: '#e2e8f0', fontWeight: '600' }}>童年常识：[狗] ←→ [骨头]</span>
                                <span style={{ fontSize: '12px', color: '#4ade80' }}>1.000 磐石稳定</span>
                            </div>
                            <div style={{ height: '6px', background: '#4ade80', borderRadius: '3px', width: '100%' }}></div>
                            {phase === 'awake' && <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '6px' }}>🛡️ 无惧白天知识轰炸引发全局竞争衰减！</div>}
                        </div>

                        {/* 新知识爬升 */}
                        <div style={{ background: 'rgba(255,255,255,0.03)', padding: '12px', borderRadius: '8px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                                <span style={{ fontSize: '12px', color: '#e2e8f0', fontWeight: '600' }}>偷渡新知：[外星人] ←→ [狗]</span>
                                <span style={{ fontSize: '12px', color: phase === 'sleep' ? '#c084fc' : '#64748b' }}>
                                    {phase === 'sleep' ? '梦境重刻爬升中...' : '0.000 拒绝直连'}
                                </span>
                            </div>
                            <div style={{ height: '6px', background: 'rgba(0,0,0,0.5)', borderRadius: '3px', overflow: 'hidden' }}>
                                <motion.div
                                    animate={{ width: phase === 'sleep' ? ['0%', '100%'] : '0%' }}
                                    transition={{ duration: 8, ease: 'linear' }}
                                    style={{ height: '100%', background: '#c084fc', borderRadius: '3px' }}
                                />
                            </div>
                            {phase === 'sleep' && <div style={{ fontSize: '11px', color: '#d8b4fe', marginTop: '6px' }}>✨ 抛开外界杂波，在极低 STDP 和纯净输入下缓慢敲实神级常识墙。</div>}
                        </div>
                    </div>
                </div>
            </div>

            <div style={{ marginTop: '20px', padding: '12px 16px', background: 'rgba(255,255,255,0.03)', borderRadius: '8px', fontSize: '12px', color: '#cbd5e1', lineHeight: '1.6' }}>
                <strong>原理说明：</strong> 如果单一网络暴露于大风暴，其全局竞争（Homeostatic Decay）会将过往最深邃的结晶彻底粉碎。造物主把计算矩阵一刀两断：第一道防线海马体学得快忘得快，而第二大坝新皮层“只有在夜间海马体进行内部重放时”才开放极低速的突触学习之门，此举终极解决了所有大语言模型面临的短长存折叠灾难！
            </div>
        </div>
    );
};

export default HippocampalReplayGraph;
