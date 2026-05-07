/**
 * LayerDetailView - Transformer Layer 内部结构详细视图
 * 显示 Attention (QKV投影、注意力分数、Softmax)、残差连接、FFN (线性+激活)
 * 当 forward pass 动画播放时，高亮当前激活的部分
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { MODEL_CONFIGS } from './constants';

// 动画阶段定义：Forward pass 中的子步骤
const PHASES = [
  { id: 'input', label: '输入', color: '#94a3b8', duration: 0.08 },
  { id: 'ln1', label: 'LayerNorm', color: '#818cf8', duration: 0.07 },
  { id: 'qkv', label: 'QKV 投影', color: '#60a5fa', duration: 0.12 },
  { id: 'attn_score', label: '注意力分数', color: '#38bdf8', duration: 0.1 },
  { id: 'softmax', label: 'Softmax', color: '#22d3ee', duration: 0.08 },
  { id: 'attn_out', label: '注意力输出', color: '#2dd4bf', duration: 0.1 },
  { id: 'residual1', label: '残差连接 ①', color: '#a78bfa', duration: 0.07 },
  { id: 'ln2', label: 'LayerNorm', color: '#818cf8', duration: 0.07 },
  { id: 'ffn_up', label: 'FFN 上投影', color: '#f59e0b', duration: 0.1 },
  { id: 'ffn_act', label: 'SiLU 激活', color: '#fb923c', duration: 0.08 },
  { id: 'ffn_down', label: 'FFN 下投影', color: '#f97316', duration: 0.08 },
  { id: 'residual2', label: '残差连接 ②', color: '#a78bfa', duration: 0.05 },
];

// 计算 phase 累积边界
function getPhaseBoundaries() {
  const total = PHASES.reduce((s, p) => s + p.duration, 0);
  let cum = 0;
  return PHASES.map(p => {
    const start = cum / total;
    cum += p.duration;
    const end = cum / total;
    return { ...p, start, end };
  });
}

function HeadGrid({ nHeads, activeHeads = [], headDim, color }) {
  const maxDisplay = Math.min(nHeads, 12);
  const cols = Math.min(nHeads, 6);
  const rows = Math.ceil(maxDisplay / cols);
  return (
    <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, 1fr)`, gap: 3 }}>
      {Array.from({ length: maxDisplay }, (_, i) => {
        const isActive = activeHeads.includes(i);
        return (
          <div key={i} title={`Head ${i}${headDim ? ` (dim=${headDim})` : ''}`} style={{
            width: 18, height: 14, borderRadius: 3,
            background: isActive ? color : 'rgba(255,255,255,0.06)',
            border: isActive ? `1px solid ${color}` : '1px solid rgba(255,255,255,0.1)',
            boxShadow: isActive ? `0 0 6px ${color}66` : 'none',
            transition: 'all 0.3s',
          }} />
        );
      })}
      {nHeads > maxDisplay && (
        <div style={{ fontSize: 8, color: '#7f95bb', gridColumn: `1 / -1`, textAlign: 'center' }}>
          +{nHeads - maxDisplay} more
        </div>
      )}
    </div>
  );
}

function NeuronBars({ neurons = [], maxDisplay = 8 }) {
  const display = neurons.slice(0, maxDisplay);
  const maxAct = Math.max(0.01, ...display.map(n => n.activation || 0));
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height: 28 }}>
      {display.map((n, i) => {
        const h = Math.max(3, (n.activation / maxAct) * 26);
        const c = n.activation > 0.7 ? '#ef4444' : n.activation > 0.4 ? '#fbbf24' : '#60a5fa';
        return (
          <div key={i} title={`N${n.id} act=${(n.activation || 0).toFixed(2)} [${n.subspace}]`} style={{
            width: 8, height: h, borderRadius: 2,
            background: c, opacity: 0.85,
            transition: 'height 0.3s',
          }} />
        );
      })}
    </div>
  );
}

function BlockBox({ label, sublabel, color, active, highlight, children, width = '100%' }) {
  return (
    <div style={{
      width,
      padding: '6px 8px',
      borderRadius: 8,
      background: active
        ? `linear-gradient(135deg, ${color}18, ${color}08)`
        : 'rgba(255,255,255,0.02)',
      border: active
        ? `1.5px solid ${color}`
        : '1px solid rgba(255,255,255,0.08)',
      boxShadow: highlight ? `0 0 12px ${color}44, inset 0 0 8px ${color}11` : 'none',
      transition: 'all 0.3s',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: children ? 4 : 0 }}>
        <div style={{
          width: 6, height: 6, borderRadius: '50%',
          background: active ? color : 'rgba(255,255,255,0.15)',
          boxShadow: active ? `0 0 4px ${color}` : 'none',
        }} />
        <span style={{
          fontSize: 10, fontWeight: 700, color: active ? color : '#7f95bb',
          transition: 'color 0.3s',
        }}>{label}</span>
        {sublabel && <span style={{ fontSize: 8, color: '#5a6f8e' }}>{sublabel}</span>}
      </div>
      {children}
    </div>
  );
}

function ResidualArrow({ label, active }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 4,
      padding: '2px 6px',
      borderLeft: active ? '2px solid #a78bfa' : '2px solid rgba(255,255,255,0.08)',
      transition: 'all 0.3s',
    }}>
      <span style={{ fontSize: 9, color: active ? '#c4b5fd' : '#5a6f8e' }}>⊕</span>
      <span style={{ fontSize: 8, color: active ? '#a78bfa' : '#5a6f8e' }}>{label}</span>
    </div>
  );
}

export default function LayerDetailView({
  layerIdx = null,
  modelKey = null,
  layerData = null,
  isActive = false,
  fpSpeed = 800, // ms per layer, used to sync animation cycle
}) {
  const mc = MODEL_CONFIGS[modelKey];
  const phaseBoundaries = useMemo(() => getPhaseBoundaries(), []);

  // Internal animation timer: cycles through phases when active
  const [animProgress, setAnimProgress] = useState(0);
  const animRef = useRef(null);
  const startTimeRef = useRef(null);

  useEffect(() => {
    if (!isActive) {
      setAnimProgress(0);
      startTimeRef.current = null;
      return;
    }
    startTimeRef.current = performance.now();
    const cycleMs = fpSpeed || 800;
    const animate = (now) => {
      if (!startTimeRef.current) startTimeRef.current = now;
      const elapsed = now - startTimeRef.current;
      const progress = (elapsed % cycleMs) / cycleMs;
      setAnimProgress(progress);
      animRef.current = requestAnimationFrame(animate);
    };
    animRef.current = requestAnimationFrame(animate);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [isActive, fpSpeed, layerIdx]);

  const phaseProgress = isActive ? animProgress : null;

  // 确定当前活跃 phase
  const currentPhase = useMemo(() => {
    if (phaseProgress == null || !isActive) return null;
    for (const pb of phaseBoundaries) {
      if (phaseProgress >= pb.start && phaseProgress < pb.end) return pb.id;
    }
    return phaseBoundaries[phaseBoundaries.length - 1]?.id || null;
  }, [phaseProgress, isActive, phaseBoundaries]);

  // 激活的 attention heads (基于 layerData 模拟)
  const activeHeads = useMemo(() => {
    if (!isActive || !layerData?.attention) return [];
    const nHeads = mc?.nHeads || 20;
    // 如果有 attention pattern，选择对角线值最高的 heads
    const pattern = layerData.attention.pattern;
    if (pattern && pattern.length > 0) {
      return pattern.slice(0, Math.min(nHeads, pattern.length))
        .map((row, i) => ({ idx: i, diag: row[i] || 0 }))
        .filter(h => h.diag > 0.2)
        .map(h => h.idx);
    }
    // 默认激活前几个
    return Array.from({ length: Math.min(4, nHeads) }, (_, i) => i);
  }, [isActive, layerData, mc]);

  const isPhase = (id) => currentPhase === id;
  const isPhaseGroup = (ids) => ids.includes(currentPhase);

  // Attention 区域是否活跃
  const attnActive = isPhaseGroup(['qkv', 'attn_score', 'softmax', 'attn_out']);
  // FFN 区域是否活跃
  const ffnActive = isPhaseGroup(['ffn_up', 'ffn_act', 'ffn_down']);

  const nHeads = mc?.nHeads || 20;
  const headDim = mc?.headDim || 128;
  const dModel = mc?.dModel || 2560;
  const mlpDim = mc?.mlpDim || 6912;

  // 层标签
  const layerLabel = layerIdx != null ? `L${layerIdx}` : '--';
  const layerFuncLabel = layerData?.label || '';

  return (
    <div style={{
      width: 240,
      minHeight: 420,
      background: 'rgba(8, 12, 24, 0.92)',
      border: isActive ? '1.5px solid #4facfe' : '1px solid rgba(255,255,255,0.08)',
      borderRadius: 14,
      padding: '10px 12px',
      color: '#e8f2ff',
      fontSize: 11,
      lineHeight: 1.5,
      backdropFilter: 'blur(10px)',
      boxShadow: isActive ? '0 4px 20px rgba(79,172,254,0.15)' : '0 2px 8px rgba(0,0,0,0.2)',
      transition: 'all 0.3s',
      overflowY: 'auto',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <div>
          <div style={{ fontSize: 13, fontWeight: 800, color: isActive ? '#4facfe' : '#7f95bb' }}>
            {layerLabel} {layerFuncLabel}
          </div>
          {mc && <div style={{ fontSize: 9, color: '#5a6f8e' }}>{mc.name}</div>}
        </div>
        {isActive && (
          <div style={{
            padding: '2px 6px', borderRadius: 4,
            background: 'rgba(79,172,254,0.15)', border: '1px solid rgba(79,172,254,0.3)',
            fontSize: 9, color: '#4facfe', fontWeight: 600,
            animation: 'pulse 1.5s infinite',
          }}>
            ACTIVE
          </div>
        )}
      </div>

      {/* 流程图主体 */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>

        {/* Input */}
        <BlockBox
          label="输入 x"
          sublabel={`d=${dModel}`}
          color="#94a3b8"
          active={isPhase('input')}
          highlight={isPhase('input')}
        />

        {/* Layer Norm 1 */}
        <BlockBox
          label="LayerNorm"
          sublabel="Pre-Attention"
          color="#818cf8"
          active={isPhase('ln1')}
          highlight={isPhase('ln1')}
        />

        {/* ===== Attention Block ===== */}
        <div style={{
          border: attnActive ? '1px solid rgba(96,165,250,0.3)' : '1px solid rgba(255,255,255,0.04)',
          borderRadius: 10,
          padding: '6px 6px 4px',
          background: attnActive ? 'rgba(96,165,250,0.04)' : 'transparent',
          transition: 'all 0.3s',
        }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: attnActive ? '#60a5fa' : '#5a6f8e', marginBottom: 4 }}>
            Multi-Head Self-Attention
          </div>

          {/* Q K V projections */}
          <div style={{ display: 'flex', gap: 4, marginBottom: 4 }}>
            <BlockBox label="Q" sublabel={`${nHeads}×${headDim}`} color="#60a5fa" active={isPhase('qkv')} highlight={isPhase('qkv')} width="33%" />
            <BlockBox label="K" sublabel={`${nHeads}×${headDim}`} color="#38bdf8" active={isPhase('qkv')} highlight={isPhase('qkv')} width="33%" />
            <BlockBox label="V" sublabel={`${nHeads}×${headDim}`} color="#2dd4bf" active={isPhase('qkv')} highlight={isPhase('qkv')} width="33%" />
          </div>

          {/* Attention Scores */}
          <BlockBox label="Q·Kᵀ / √d" sublabel={`[${nHeads}×seq×seq]`} color="#38bdf8" active={isPhase('attn_score')} highlight={isPhase('attn_score')}>
            <HeadGrid nHeads={nHeads} activeHeads={activeHeads} headDim={headDim} color="#38bdf8" />
          </BlockBox>

          {/* Softmax */}
          <BlockBox label="Softmax" sublabel="按行归一化" color="#22d3ee" active={isPhase('softmax')} highlight={isPhase('softmax')} />

          {/* Attn·V + Output Projection */}
          <BlockBox label="Attn·V → Wₒ" sublabel={`→ d=${dModel}`} color="#2dd4bf" active={isPhase('attn_out')} highlight={isPhase('attn_out')} />
        </div>

        {/* Residual 1 */}
        <ResidualArrow label="残差连接 +x" active={isPhase('residual1')} />

        {/* Layer Norm 2 */}
        <BlockBox
          label="LayerNorm"
          sublabel="Pre-FFN"
          color="#818cf8"
          active={isPhase('ln2')}
          highlight={isPhase('ln2')}
        />

        {/* ===== FFN Block ===== */}
        <div style={{
          border: ffnActive ? '1px solid rgba(245,158,11,0.3)' : '1px solid rgba(255,255,255,0.04)',
          borderRadius: 10,
          padding: '6px 6px 4px',
          background: ffnActive ? 'rgba(245,158,11,0.04)' : 'transparent',
          transition: 'all 0.3s',
        }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: ffnActive ? '#f59e0b' : '#5a6f8e', marginBottom: 4 }}>
            Feed-Forward Network
          </div>

          {/* Up projection */}
          <BlockBox label="W_up" sublabel={`${dModel}→${mlpDim?.toLocaleString()}`} color="#f59e0b" active={isPhase('ffn_up')} highlight={isPhase('ffn_up')}>
            {layerData?.ffn?.top_neurons && <NeuronBars neurons={layerData.ffn.top_neurons} />}
          </BlockBox>

          {/* Activation */}
          <BlockBox
            label="SiLU 激活"
            sublabel={`gate=${layerData?.ffn?.gate_activation?.toFixed(2) || '-'}`}
            color="#fb923c"
            active={isPhase('ffn_act')}
            highlight={isPhase('ffn_act')}
          >
            <div style={{ marginTop: 2 }}>
              <svg width="100%" height="20" viewBox="0 0 100 20">
                <path
                  d="M5,18 C15,18 25,16 35,10 C45,4 55,2 65,2 L95,2"
                  fill="none"
                  stroke={isPhase('ffn_act') ? '#fb923c' : '#3a4a5e'}
                  strokeWidth="1.5"
                />
                <path
                  d="M5,18 C12,16 18,14 25,8 C32,2 38,1 45,1 L95,1"
                  fill="none"
                  stroke={isPhase('ffn_act') ? 'rgba(251,146,60,0.4)' : 'rgba(58,74,94,0.4)'}
                  strokeWidth="1"
                  strokeDasharray="3,2"
                />
              </svg>
            </div>
          </BlockBox>

          {/* Down projection */}
          <BlockBox label="W_down" sublabel={`${mlpDim?.toLocaleString()}→${dModel}`} color="#f97316" active={isPhase('ffn_down')} highlight={isPhase('ffn_down')} />
        </div>

        {/* Residual 2 */}
        <ResidualArrow label="残差连接 +x" active={isPhase('residual2')} />

        {/* Output */}
        <BlockBox
          label="输出"
          sublabel={`‖r‖=${layerData?.residual_norm?.toFixed(1) || '-'}`}
          color="#34d399"
          active={isActive && currentPhase === null}
          highlight={false}
        />
      </div>

      {/* Phase Indicator */}
      {isActive && currentPhase && (
        <div style={{
          marginTop: 8, padding: '4px 8px', borderRadius: 6,
          background: 'rgba(79,172,254,0.08)', border: '1px solid rgba(79,172,254,0.2)',
          fontSize: 9, textAlign: 'center', color: '#4facfe',
        }}>
          ▸ {PHASES.find(p => p.id === currentPhase)?.label || currentPhase}
        </div>
      )}

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.6; }
        }
      `}</style>
    </div>
  );
}
