import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Activity, BrainCircuit, XOctagon, CheckCircle2, Waves } from 'lucide-react';

const GammaSynchronyGraph = () => {
    const [phase, setPhase] = useState('chaos'); // 'chaos' or 'gamma'

    useEffect(() => {
        const timer = setInterval(() => {
            setPhase(prev => prev === 'chaos' ? 'gamma' : 'chaos');
        }, 8000);
        return () => clearInterval(timer);
    }, []);

    // 模拟底层的四路信号
    const bottomSignals = [
        { id: 'red', name: '红色 (A)', color: '#ef4444' },
        { id: 'apple', name: '苹果 (B)', color: '#fca5a5' },
        { id: 'yellow', name: '黄色 (C)', color: '#eab308' },
        { id: 'banana', name: '香蕉 (D)', color: '#fef08a' }
    ];

    return (
        <div style={{
            background: 'linear-gradient(145deg, rgba(12, 10, 22, 0.9) 0%, rgba(20, 15, 35, 0.95) 100%)',
            border: `1px solid ${phase === 'chaos' ? 'rgba(2ef, 68, 68, 0.3)' : 'rgba(56, 189, 248, 0.4)'}`,
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
                        background: phase === 'chaos' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(56, 189, 248, 0.2)',
                        padding: '10px', borderRadius: '10px',
                        transition: 'background 1s'
                    }}>
                        <BrainCircuit size={24} color={phase === 'chaos' ? '#ef4444' : '#38bdf8'} />
                    </div>
                    <div>
                        <h3 style={{ margin: 0, fontSize: '18px', color: '#e2e8f0', fontWeight: '800' }}>
                            Gamma 40Hz 强锁波：消除特征绑定灾难 (Binding-by-Synchrony)
                        </h3>
                        <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
                            展示无高维位置张量下，时间微元如何斩断“红苹果+黄香蕉 {'->'} 红香蕉”的恶性幻觉
                        </div>
                    </div>
                </div>

                {/* 状态切换徽章 */}
                <div style={{
                    display: 'flex', alignItems: 'center', gap: '8px',
                    padding: '8px 16px', borderRadius: '20px',
                    background: phase === 'chaos' ? 'rgba(239, 68, 68, 0.15)' : 'rgba(56, 189, 248, 0.15)',
                    border: `1px solid ${phase === 'chaos' ? 'rgba(239, 68, 68, 0.3)' : 'rgba(56, 189, 248, 0.3)'}`
                }}>
                    <Waves size={16} color={phase === 'chaos' ? '#ef4444' : '#38bdf8'} />
                    <span style={{ fontSize: '13px', fontWeight: 'bold', color: phase === 'chaos' ? '#fca5a5' : '#bae6fd' }}>
                        当前状态：{phase === 'chaos' ? '低频混沌水池 (特征错配)' : '高频 Gamma 硬锁 (毫秒解耦)'}
                    </span>
                </div>
            </div>

            <div style={{ display: 'flex', gap: '30px' }}>
                {/* 左侧：底层特征输入源 (Bottom Layer) */}
                <div style={{ flex: '1', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#94a3b8', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '8px' }}>
                        初级视觉皮层 (输入流)
                    </div>
                    {bottomSignals.map((sig, idx) => (
                        <div key={sig.id} style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                            <div style={{ width: '70px', fontSize: '12px', color: sig.color, fontWeight: '600', textAlign: 'right' }}>
                                {sig.name}
                            </div>
                            <div style={{ flex: 1, height: '30px', background: 'rgba(0,0,0,0.3)', borderRadius: '6px', position: 'relative', overflow: 'hidden' }}>
                                {/* 脉冲动画 */}
                                <motion.div
                                    animate={
                                        phase === 'chaos'
                                            ? { x: ['-20%', '120%'], opacity: [0, 1, 0, 1, 0] } // 混沌状态下随机乱闪
                                            : { x: ['-20%', '120%'], opacity: idx < 2 ? [0, 1, 1, 0, 0, 0] : [0, 0, 0, 1, 1, 0] } // Gamma态严格错时
                                    }
                                    transition={{
                                        duration: phase === 'chaos' ? Math.random() * 2 + 1 : 1.5,
                                        repeat: Infinity,
                                        ease: "linear"
                                    }}
                                    style={{
                                        position: 'absolute', top: '0', bottom: '0', width: '20px',
                                        background: sig.color,
                                        boxShadow: `0 0 10px ${sig.color}`,
                                        opacity: 0.8
                                    }}
                                />
                            </div>
                        </div>
                    ))}
                    <div style={{ fontSize: '11px', color: '#64748b', marginTop: '8px' }}>
                        {phase === 'chaos'
                            ? '🚨 警告：底层电波处于长尾混杂态，没有严格的时间波峰对齐。'
                            : '⚡ 物理制导：底层被强制调入 40Hz 频段。红苹果组与黄香蕉组相位相差 180度！'}
                    </div>
                </div>

                {/* 右侧：高级概念提取区 (Top Layer Concept) */}
                <div style={{ flex: '1.2', background: 'rgba(0,0,0,0.4)', borderRadius: '12px', padding: '20px', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#94a3b8', marginBottom: '20px', textAlign: 'center' }}>
                        高层抽象皮层 (漏电积分突触)
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                        {/* 正确目标概念 */}
                        <div style={{ background: 'rgba(255,255,255,0.03)', padding: '16px', borderRadius: '12px', position: 'relative' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                                <span style={{ color: '#e2e8f0', fontSize: '14px', fontWeight: '600' }}>目标概念：红苹果 (A+B)</span>
                                <span style={{ color: '#4ade80', fontSize: '13px' }}>探测：命中</span>
                            </div>
                            <div style={{ height: '8px', background: 'rgba(0,0,0,0.5)', borderRadius: '4px', overflow: 'hidden' }}>
                                <motion.div
                                    animate={{ width: phase === 'chaos' ? ['20%', '80%', '40%'] : ['0%', '100%', '0%'] }}
                                    transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
                                    style={{ height: '100%', background: '#4ade80', borderRadius: '4px' }}
                                />
                            </div>
                        </div>

                        {/* 灾难性错误概念：红香蕉 */}
                        <div style={{
                            background: phase === 'chaos' ? 'rgba(239,68,68,0.1)' : 'rgba(255,255,255,0.03)',
                            padding: '16px', borderRadius: '12px', position: 'relative',
                            border: phase === 'chaos' ? '1px solid rgba(239,68,68,0.3)' : '1px solid transparent',
                            transition: 'all 0.5s'
                        }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                    {phase === 'chaos' && <XOctagon size={16} color="#ef4444" />}
                                    <span style={{ color: phase === 'chaos' ? '#fca5a5' : '#94a3b8', fontSize: '14px', fontWeight: '600' }}>
                                        多维特征乱伦幻觉：红香蕉 (A+D)
                                    </span>
                                </div>
                                <span style={{ color: phase === 'chaos' ? '#ef4444' : '#64748b', fontSize: '13px', fontWeight: 'bold' }}>
                                    {phase === 'chaos' ? '💥 错误涌现 (29次命中)' : '零唤醒 (0次跨维串流)'}
                                </span>
                            </div>
                            <div style={{ height: '8px', background: 'rgba(0,0,0,0.5)', borderRadius: '4px', overflow: 'hidden' }}>
                                <motion.div
                                    animate={{ width: phase === 'chaos' ? ['30%', '90%', '50%'] : '0%' }}
                                    transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
                                    style={{ height: '100%', background: phase === 'chaos' ? '#ef4444' : '#64748b', borderRadius: '4px' }}
                                />
                            </div>
                            {phase === 'chaos' && (
                                <div style={{ fontSize: '11px', color: '#ef4444', marginTop: '10px', lineHeight: '1.5' }}>
                                    致命错误：缺乏矩阵刻画！由于底层电波的长尾拖拽(τ=15ms)，【红色】与隔壁【香蕉】的突触同时突破阈值！网络严重产生串流！
                                </div>
                            )}
                            {phase === 'gamma' && (
                                <div style={{ fontSize: '11px', color: '#38bdf8', marginTop: '10px', lineHeight: '1.5' }}>
                                    降维打击：高层缩小接收窗口(τ=10ms)，【红色】和【香蕉】哪怕混在同一个房间，由于相差毫秒级的错峰，在这里连一个浪花都翻不起！
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            <div style={{ marginTop: '20px', padding: '12px 16px', background: 'rgba(255,255,255,0.03)', borderRadius: '8px', fontSize: '12px', color: '#cbd5e1', lineHeight: '1.6' }}>
                <strong>原理说明：</strong> 生物脑没有位置编码 (Position Embedding)！当它面对无数并发的局部特征时，利用极高频率的时空电磁切片（Gamma Band）让所属同一主体的细胞【绝对死锁同步开火】。高层只提取共振前沿，彻底从 O(N²) 的复杂度中终结了混合多维度的特征塌缩灾变！
            </div>
        </div>
    );
};

export default GammaSynchronyGraph;
