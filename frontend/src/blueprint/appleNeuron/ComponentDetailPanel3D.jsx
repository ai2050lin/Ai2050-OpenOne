/**
 * ComponentDetailPanel3D - 层组件详情3D面板
 * 
 * 布局: 上方 = 信息参数, 下方 = 3D几何模型
 * 与 LayerExplodedView3D 动画同步, 只显示当前高亮组件
 */
import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import { MODEL_CONFIGS } from './constants';
import { locales } from '../../locales';

// ── 翻译函数 ──
function makeT(lang) {
  return (key, params = {}) => {
    const keys = key.split('.');
    let val = locales[lang];
    for (const k of keys) val = val?.[k];
    if (!val) return key;
    for (const [pKey, pVal] of Object.entries(params)) {
      val = val.replace(`{{${pKey}}}`, pVal);
    }
    return val;
  };
}

// ── 与 LayerExplodedView3D 同步的动画阶段 ──
const PHASES = [
  { id: 'input',      color: '#94a3b8', component: 'input' },
  { id: 'ln1',        color: '#818cf8', component: 'ln' },
  { id: 'qkv',        color: '#60a5fa', component: 'attention' },
  { id: 'attn_score', color: '#38bdf8', component: 'attention' },
  { id: 'softmax',    color: '#22d3ee', component: 'attention' },
  { id: 'attn_out',   color: '#2dd4bf', component: 'attention' },
  { id: 'residual1',  color: '#a78bfa', component: 'residual' },
  { id: 'ln2',        color: '#818cf8', component: 'ln' },
  { id: 'ffn_up',     color: '#f59e0b', component: 'ffn' },
  { id: 'ffn_act',    color: '#fb923c', component: 'ffn' },
  { id: 'ffn_down',   color: '#f97316', component: 'ffn' },
  { id: 'residual2',  color: '#a78bfa', component: 'residual' },
];
const PHASE_DURATIONS = [0.08, 0.07, 0.12, 0.1, 0.08, 0.1, 0.07, 0.07, 0.1, 0.08, 0.08, 0.05];
const TOTAL_DUR = PHASE_DURATIONS.reduce((s, d) => s + d, 0);

function getPhaseBoundaries() {
  let cum = 0;
  return PHASES.map((p, i) => {
    const start = cum / TOTAL_DUR;
    cum += PHASE_DURATIONS[i];
    const end = cum / TOTAL_DUR;
    return { ...p, start, end };
  });
}

// ── 组件参数数据生成 ──
function getLayerComponentData(layer, nLayers = 28, t) {
  const ratio = layer / Math.max(1, nLayers - 1);
  return {
    layer,
    layerColor: getLayerColor(ratio),
    layerLabel: getLayerLabel(ratio, t),
    ln: {
      beta: (0.98 + 0.02 * Math.sin(ratio * Math.PI)).toFixed(3),
      leakage: ratio < 0.33 ? 0.02 : ratio < 0.61 ? 0.08 + 0.04 * ratio : 0.15 - 0.05 * (1 - ratio),
      gamma: (1.0 + 0.05 * Math.sin(ratio * Math.PI * 3)).toFixed(4),
      eps: 1e-5,
      mean: (0.02 + 0.03 * ratio).toFixed(4),
      var: (0.8 + 0.4 * ratio).toFixed(3),
    },
    attention: {
      strength: ratio < 0.14 ? 0.15 : ratio < 0.61 ? 0.15 + 0.60 * ((ratio - 0.14) / 0.47) : 0.75 + 0.20 * ((ratio - 0.61) / 0.39),
      pattern: ratio < 0.33 ? t('componentDetail.induction') : ratio < 0.61 ? t('componentDetail.semantic') : t('componentDetail.copy'),
      nHeads: 20,
      headDim: 128,
      topHeads: ratio < 0.33 ? 'H3, H7' : ratio < 0.61 ? 'H1, H5, H9' : 'H2, H11',
      avgAttn: (0.3 + 0.4 * ratio).toFixed(3),
      entropy: (2.5 - 0.8 * ratio).toFixed(2),
      qkCos: (0.15 + 0.35 * ratio).toFixed(3),
      ovScore: (0.2 + 0.5 * ratio).toFixed(3),
    },
    ffn: {
      gain: ratio < 0.33 ? 0.3 : ratio < 0.61 ? 0.3 + 1.5 * ((ratio - 0.33) / 0.28) : 1.8 + 0.9 * ((ratio - 0.61) / 0.39),
      dModel: 2560,
      mlpDim: 6912,
      topNeurons: ratio < 0.33 ? 'N42, N187' : ratio < 0.61 ? 'N42, N187, N512' : 'N187, N1024',
      siluGate: (0.4 + 0.3 * ratio).toFixed(3),
      upProj: ratio < 0.33 ? t('componentDetail.lexicalToHidden') : ratio < 0.61 ? t('componentDetail.semanticToFeature') : t('componentDetail.logicToOutput'),
      actRatio: (0.05 + 0.10 * ratio).toFixed(3),
    },
    residual: {
      retention: 0.62 + 0.09 * Math.sin(ratio * Math.PI * 2),
      streamDir: ratio < 0.33 ? t('componentDetail.lexical') : ratio < 0.61 ? t('componentDetail.semantic') : t('componentDetail.decision'),
      skipWeight: (0.85 + 0.10 * Math.sin(ratio * Math.PI)).toFixed(3),
      norm: (3.0 + 7.0 * ratio).toFixed(1),
    },
  };
}

function getLayerColor(ratio) {
  if (ratio <= 0.14) return '#ff6b6b';
  if (ratio <= 0.33) return '#ffe66d';
  if (ratio <= 0.61) return '#4ecdc4';
  if (ratio <= 0.69) return '#a855f7';
  return '#f97316';
}

function getLayerLabel(ratio, t) {
  if (ratio <= 0.03) return t('componentDetail.embedding');
  if (ratio <= 0.14) return t('componentDetail.lexical');
  if (ratio <= 0.33) return t('componentDetail.syntax');
  if (ratio <= 0.61) return t('componentDetail.semantic');
  if (ratio <= 0.69) return t('componentDetail.logic');
  return t('componentDetail.decision');
}

// ═══════════════════════════════════════════════════════
// 运行时参数面板 - Layer 内部运行时具体参数状态可视化
// ═══════════════════════════════════════════════════════

function getParamColor(value) {
  if (value > 0.8) return '#ef4444';
  if (value > 0.5) return '#eab308';
  if (value > 0.3) return '#22c55e';
  return '#1e293b';
}

function getParamEmissive(value) {
  if (value > 0.8) return '#ef4444';
  if (value > 0.5) return '#eab308';
  if (value > 0.3) return '#22c55e';
  return '#000000';
}

// ── 参数条: 单个参数指标的可视化 ──
function ParamBar({ label, value, maxVal = 1, color, x, y, w = 2.8, h = 0.22, t }) {
  const fillW = Math.max(0.05, (value / maxVal) * w);
  const barColor = color || getParamColor(value / maxVal);
  return (
    <group position={[x, y, 0]}>
      <Text position={[-0.1, 0, 0.05]} fontSize={0.12} color="#94a3b8" anchorX="right" anchorY="middle">
        {label}
      </Text>
      <mesh position={[w / 2, 0, 0]}>
        <boxGeometry args={[w, h, 0.04]} />
        <meshBasicMaterial color="#1e293b" transparent opacity={0.5} />
      </mesh>
      <mesh position={[fillW / 2, 0, 0.025]}>
        <boxGeometry args={[fillW, h * 0.8, 0.06]} />
        <meshStandardMaterial
          color={barColor}
          emissive={barColor}
          emissiveIntensity={0.3 + (value / maxVal) * 0.4}
          transparent
          opacity={0.8}
        />
      </mesh>
      <Text position={[w + 0.15, 0, 0.05]} fontSize={0.1} color="#e2e8f0" anchorX="left" anchorY="middle">
        {typeof value === 'number' ? value.toFixed(3) : value}
      </Text>
    </group>
  );
}

// ── 微型热力格: 紧凑参数矩阵可视化 ──
function MicroHeatGrid({ data, rows, cols, cellSize = 0.28, position = [0, 0, 0], title = '', t }) {
  const totalW = cols * cellSize;
  const totalH = rows * cellSize;
  return (
    <group position={position}>
      {title && (
        <Text position={[totalW / 2, totalH / 2 + 0.22, 0.05]} fontSize={0.11} color="#94a3b8" anchorX="center" anchorY="middle">
          {title}
        </Text>
      )}
      <mesh position={[totalW / 2, 0, -0.01]}>
        <boxGeometry args={[totalW + 0.08, totalH + 0.08, 0.01]} />
        <meshBasicMaterial color="#0f172a" transparent opacity={0.5} />
      </mesh>
      {data.map((val, i) => {
        const col = i % cols;
        const row = Math.floor(i / cols);
        const x = col * cellSize + cellSize / 2;
        const y = -(row * cellSize + cellSize / 2) + totalH / 2;
        const c = getParamColor(val);
        return (
          <mesh key={`hg${i}`} position={[x, y, 0]}>
            <boxGeometry args={[cellSize * 0.82, cellSize * 0.82, val * 0.12 + 0.01]} />
            <meshStandardMaterial
              color={c}
              emissive={getParamEmissive(val)}
              emissiveIntensity={val > 0.3 ? 0.2 + val * 0.4 : 0.03}
              transparent
              opacity={0.2 + val * 0.7}
            />
          </mesh>
        );
      })}
    </group>
  );
}

// ── 统计卡片: 数值+标签 ──
function StatCard({ label, value, color, x, y, t }) {
  return (
    <group position={[x, y, 0]}>
      <mesh>
        <boxGeometry args={[1.6, 0.7, 0.05]} />
        <meshStandardMaterial color="#0f172a" transparent opacity={0.7} />
      </mesh>
      <mesh position={[0, 0, 0.03]}>
        <boxGeometry args={[1.5, 0.6, 0.02]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.08} transparent opacity={0.15} />
      </mesh>
      <Text position={[0, 0.12, 0.06]} fontSize={0.1} color="#94a3b8" anchorX="center" anchorY="middle">
        {label}
      </Text>
      <Text position={[0, -0.12, 0.06]} fontSize={0.16} color={color} anchorX="center" anchorY="middle">
        {value}
      </Text>
    </group>
  );
}

// ── LN 运行时参数面板 ──
function LNRuntimeParams({ data, color, position = [0, 0, 0], t }) {
  const gammaRef = useRef([]);
  const betaRef = useRef([]);
  const nDim = 12;
  // 每维 γ/β 分布
  const gammaVals = useMemo(() => {
    const base = parseFloat(data.ln.gamma) || 1.0;
    return Array.from({ length: nDim }, (_, i) =>
      Math.max(0, Math.min(2, base + 0.08 * Math.sin(i * 1.3) + 0.04 * Math.cos(i * 0.7)))
    );
  }, [data.ln.gamma]);
  const betaVals = useMemo(() => {
    const base = parseFloat(data.ln.beta) || 0.98;
    return Array.from({ length: nDim }, (_, i) =>
      Math.max(-0.5, Math.min(1.5, base + 0.05 * Math.sin(i * 0.9 + 1) - 0.03 * Math.cos(i * 1.7)))
    );
  }, [data.ln.beta]);
  // 归一化前后分布
  const preNorm = useMemo(() => Array.from({ length: nDim }, (_, i) => 0.2 + 0.8 * Math.abs(Math.sin(i * 1.5 + 0.3 * i))), []);
  const postNorm = useMemo(() => Array.from({ length: nDim }, (_, i) => 0.45 + 0.1 * Math.sin(i * 0.4)), []);

  // 动态脉冲
  useFrame((state) => {
    gammaRef.current.forEach((ref, i) => {
      if (ref) ref.material.emissiveIntensity = 0.15 + 0.1 * Math.sin(state.clock.elapsedTime * 1.5 + i * 0.4);
    });
    betaRef.current.forEach((ref, i) => {
      if (ref) ref.material.emissiveIntensity = 0.15 + 0.1 * Math.sin(state.clock.elapsedTime * 1.8 + i * 0.3);
    });
  });

  const panelW = 8;
  return (
    <group position={position}>
      {/* 标题 */}
      <Text position={[panelW / 2, 1.6, 0.1]} fontSize={0.18} color={color} anchorX="center" anchorY="middle"
        outlineWidth={0.01} outlineColor="#0a1022">
        {t('componentDetail.runtimeParams')}
      </Text>

      {/* ── 左列: γ/β 每维分布 ── */}
      <group position={[0, 0, 0]}>
        <Text position={[1.2, 1.2, 0.05]} fontSize={0.12} color="#a78bfa" anchorX="left" anchorY="middle">
          γ (weight/dim)
        </Text>
        {gammaVals.map((v, i) => {
          const h = v * 0.6;
          return (
            <mesh key={`g${i}`} ref={el => gammaRef.current[i] = el} position={[0.15 * i, 0.6, 0]}>
              <boxGeometry args={[0.1, h, 0.06]} />
              <meshStandardMaterial color="#a78bfa" emissive="#a78bfa" emissiveIntensity={0.2} transparent opacity={0.85} />
            </mesh>
          );
        })}
        <Text position={[1.2, -0.1, 0.05]} fontSize={0.12} color="#818cf8" anchorX="left" anchorY="middle">
          β (bias/dim)
        </Text>
        {betaVals.map((v, i) => {
          const h = (v + 0.5) * 0.4;
          return (
            <mesh key={`b${i}`} ref={el => betaRef.current[i] = el} position={[0.15 * i, -0.7, 0]}>
              <boxGeometry args={[0.1, h, 0.06]} />
              <meshStandardMaterial color="#818cf8" emissive="#818cf8" emissiveIntensity={0.2} transparent opacity={0.85} />
            </mesh>
          );
        })}
      </group>

      {/* ── 中列: 归一化前后对比 ── */}
      <group position={[2.5, 0, 0]}>
        <Text position={[1.2, 1.2, 0.05]} fontSize={0.12} color="#64748b" anchorX="left" anchorY="middle">
          {t('componentDetail.preNorm')}
        </Text>
        {preNorm.map((v, i) => (
          <mesh key={`pre${i}`} position={[0.2 * i, 0.6, 0]}>
            <boxGeometry args={[0.14, v * 0.8, 0.06]} />
            <meshStandardMaterial color="#64748b" emissive="#64748b" emissiveIntensity={0.08} transparent opacity={0.65} />
          </mesh>
        ))}
        <Text position={[1.2, -0.1, 0.05]} fontSize={0.12} color={color} anchorX="left" anchorY="middle">
          {t('componentDetail.postNorm')}
        </Text>
        {postNorm.map((v, i) => (
          <mesh key={`post${i}`} position={[0.2 * i, -0.7, 0]}>
            <boxGeometry args={[0.14, v * 0.8, 0.06]} />
            <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.2} transparent opacity={0.8} />
          </mesh>
        ))}
      </group>

      {/* ── 右列: 统计卡片 ── */}
      <group position={[5.5, 0, 0]}>
        <StatCard label="μ" value={data.ln.mean} color="#f97316" x={0} y={0.8} t={t} />
        <StatCard label="σ²" value={data.ln.var} color="#22d3ee" x={0} y={-0.1} t={t} />
        <StatCard label="ε" value="1e-5" color="#94a3b8" x={0} y={-1.0} t={t} />
        <ParamBar label="leak" value={data.ln.leakage} color="#f97316" x={0} y={-1.7} w={1.4} t={t} />
      </group>
    </group>
  );
}

// ── Attention 运行时参数面板 ──
function AttentionRuntimeParams({ data, phase, color, position = [0, 0, 0], t }) {
  const headRefs = useRef([]);
  const nHeads = 8;
  // 每头参数: 激活度, QK余弦, OV分数, 熵
  const headParams = useMemo(() => {
    const strength = data.attention.strength;
    const qkBase = parseFloat(data.attention.qkCos) || 0.3;
    const ovBase = parseFloat(data.attention.ovScore) || 0.4;
    return Array.from({ length: nHeads }, (_, i) => ({
      active: i < Math.round(strength * nHeads),
      activation: i < Math.round(strength * nHeads) ? 0.5 + 0.4 * Math.sin(i * 1.2) : 0.05 + 0.1 * Math.random(),
      qkCos: qkBase + 0.12 * Math.sin(i * 0.9) - 0.05 * Math.cos(i * 1.3),
      ovScore: ovBase + 0.1 * Math.cos(i * 0.7) + 0.03 * Math.sin(i * 1.5),
      entropy: (parseFloat(data.attention.entropy) || 2.0) + 0.3 * Math.sin(i * 0.6),
    }));
  }, [data.attention]);

  // 动态脉冲
  useFrame((state) => {
    headRefs.current.forEach((ref, i) => {
      if (ref) {
        const p = headParams[i];
        if (p.active) {
          ref.material.emissiveIntensity = 0.25 + 0.15 * Math.sin(state.clock.elapsedTime * 2 + i * 0.5);
        }
      }
    });
  });

  return (
    <group position={position}>
      <Text position={[4, 2.2, 0.1]} fontSize={0.18} color={color} anchorX="center" anchorY="middle"
        outlineWidth={0.01} outlineColor="#0a1022">
        {t('componentDetail.runtimeParams')}
      </Text>

      {/* ── 左: 注意力头激活状态 ── */}
      <group position={[0, 0.5, 0]}>
        <Text position={[1.4, 1.5, 0.05]} fontSize={0.11} color="#60a5fa" anchorX="center" anchorY="middle">
          {t('componentDetail.headActivations')}
        </Text>
        {headParams.map((p, i) => {
          const x = (i % 4) * 0.75;
          const y = -Math.floor(i / 4) * 1.0 + 0.5;
          const hColor = p.active ? '#60a5fa' : '#1e293b';
          return (
            <group key={`hd${i}`} position={[x, y, 0]}>
              <mesh ref={el => headRefs.current[i] = el}>
                <sphereGeometry args={[0.22, 12, 12]} />
                <meshStandardMaterial
                  color={hColor}
                  emissive={p.active ? hColor : '#000'}
                  emissiveIntensity={p.active ? 0.35 : 0}
                  transparent
                  opacity={p.active ? 0.85 : 0.2}
                />
              </mesh>
              <Text position={[0, -0.35, 0.05]} fontSize={0.09} color={p.active ? '#e2e8f0' : '#475569'} anchorX="center" anchorY="middle">
                {`H${i}`}
              </Text>
              {p.active && (
                <Text position={[0, 0.35, 0.05]} fontSize={0.07} color="#94a3b8" anchorX="center" anchorY="middle">
                  {p.activation.toFixed(2)}
                </Text>
              )}
            </group>
          );
        })}
      </group>

      {/* ── 中: QK/OV 指标 ── */}
      <group position={[3.5, 0.5, 0]}>
        <Text position={[1.2, 1.5, 0.05]} fontSize={0.11} color="#f472b6" anchorX="center" anchorY="middle">
          {t('componentDetail.circuitMetrics')}
        </Text>
        {headParams.filter(p => p.active).map((p, i) => (
          <group key={`cm${i}`} position={[0, 0.8 - i * 0.55, 0]}>
            <ParamBar label={`QK`} value={p.qkCos} color="#f472b6" x={0} y={0} w={1.0} h={0.16} t={t} />
            <ParamBar label={`OV`} value={p.ovScore} color="#34d399" x={0} y={-0.22} w={1.0} h={0.16} t={t} />
          </group>
        ))}
      </group>

      {/* ── 右: 汇总统计 ── */}
      <group position={[6.2, 0.5, 0]}>
        <StatCard label="cos(Q,K)" value={data.attention.qkCos} color="#f472b6" x={0} y={0.8} t={t} />
        <StatCard label="OV" value={data.attention.ovScore} color="#34d399" x={0} y={-0.1} t={t} />
        <StatCard label="H" value={data.attention.entropy} color="#22d3ee" x={0} y={-1.0} t={t} />
        <ParamBar label="str" value={data.attention.strength} color={color} x={0} y={-1.6} w={1.4} t={t} />
      </group>
    </group>
  );
}

// ── FFN 运行时参数面板 ──
function FFNRuntimeParams({ data, phase, color, position = [0, 0, 0], t }) {
  const neuronRefs = useRef([]);
  const nNeurons = 16;
  // 每神经元参数: 激活值, gate值, up投影, down投影
  const neuronParams = useMemo(() => {
    const actRatio = parseFloat(data.ffn.actRatio) || 0.1;
    const gateBase = parseFloat(data.ffn.siluGate) || 0.5;
    const gain = data.ffn.gain;
    return Array.from({ length: nNeurons }, (_, i) => {
      const isActive = i < Math.round(actRatio * nNeurons);
      return {
        id: `N${i * 128 + 42}`,
        active: isActive,
        activation: isActive ? 0.4 + 0.55 * Math.abs(Math.sin(i * 2.1)) : 0.02 + 0.08 * Math.random(),
        gate: isActive ? gateBase + 0.15 * Math.sin(i * 1.1) : 0.05 + 0.05 * Math.random(),
        upProj: isActive ? 0.3 + 0.6 * (gain / 3) * Math.abs(Math.cos(i * 0.8)) : 0.02,
        downProj: isActive ? 0.2 + 0.5 * (gain / 3) * Math.abs(Math.sin(i * 1.5)) : 0.01,
      };
    });
  }, [data.ffn]);

  // 动态
  useFrame((state) => {
    neuronRefs.current.forEach((ref, i) => {
      if (ref && neuronParams[i]?.active) {
        ref.material.emissiveIntensity = 0.2 + 0.12 * Math.sin(state.clock.elapsedTime * 1.8 + i * 0.6);
      }
    });
  });

  return (
    <group position={position}>
      <Text position={[4, 2.2, 0.1]} fontSize={0.18} color={color} anchorX="center" anchorY="middle"
        outlineWidth={0.01} outlineColor="#0a1022">
        {t('componentDetail.runtimeParams')}
      </Text>

      {/* ── 左: 神经元激活状态 ── */}
      <group position={[0, 0.5, 0]}>
        <Text position={[1.2, 1.5, 0.05]} fontSize={0.11} color="#f59e0b" anchorX="center" anchorY="middle">
          {t('componentDetail.neuronStates')}
        </Text>
        {neuronParams.map((p, i) => {
          const col = i % 4;
          const row = Math.floor(i / 4);
          const x = col * 0.7;
          const y = -row * 0.8 + 0.5;
          const nColor = p.active ? getParamColor(p.activation) : '#1e293b';
          return (
            <group key={`fn${i}`} position={[x, y, 0]}>
              <mesh ref={el => neuronRefs.current[i] = el}>
                <boxGeometry args={[0.45, 0.45, p.activation * 0.15 + 0.02]} />
                <meshStandardMaterial
                  color={nColor}
                  emissive={p.active ? getParamEmissive(p.activation) : '#000'}
                  emissiveIntensity={p.active ? 0.3 : 0}
                  transparent
                  opacity={p.active ? 0.8 : 0.15}
                />
              </mesh>
              <Text position={[0, -0.35, 0.05]} fontSize={0.06} color={p.active ? '#e2e8f0' : '#475569'} anchorX="center" anchorY="middle">
                {p.id}
              </Text>
            </group>
          );
        })}
      </group>

      {/* ── 中: Gate/Up/Down 投影参数 ── */}
      <group position={[3.5, 0.5, 0]}>
        <Text position={[1.2, 1.5, 0.05]} fontSize={0.11} color="#fb923c" anchorX="center" anchorY="middle">
          {t('componentDetail.projParams')}
        </Text>
        {neuronParams.filter(p => p.active).slice(0, 6).map((p, i) => (
          <group key={`pp${i}`} position={[0, 0.8 - i * 0.55, 0]}>
            <ParamBar label="gate" value={p.gate} color="#fb923c" x={0} y={0} w={1.0} h={0.14} t={t} />
            <ParamBar label="W↑" value={p.upProj} color="#f59e0b" x={0} y={-0.2} w={1.0} h={0.14} t={t} />
            <ParamBar label="W↓" value={p.downProj} color="#f97316" x={0} y={-0.4} w={1.0} h={0.14} t={t} />
          </group>
        ))}
      </group>

      {/* ── 右: 汇总 ── */}
      <group position={[6.2, 0.5, 0]}>
        <StatCard label="gain" value={`${data.ffn.gain.toFixed(2)}x`} color="#f59e0b" x={0} y={0.8} t={t} />
        <StatCard label="gate" value={data.ffn.siluGate} color="#fb923c" x={0} y={-0.1} t={t} />
        <StatCard label="act%" value={`${(parseFloat(data.ffn.actRatio) * 100).toFixed(1)}%`} color="#f97316" x={0} y={-1.0} t={t} />
        <ParamBar label="d_model" value={data.ffn.dModel / 4096} color="#94a3b8" x={0} y={-1.6} w={1.4} t={t} />
      </group>
    </group>
  );
}

// ── Residual 运行时参数面板 ──
function ResidualRuntimeParams({ data, color, position = [0, 0, 0], t }) {
  const nDims = 8;
  // 每维度参数: skip贡献, 残差流成分, 范数
  const dimParams = useMemo(() => {
    const retention = data.residual.retention;
    const skipW = parseFloat(data.residual.skipWeight) || 0.85;
    return Array.from({ length: nDims }, (_, i) => ({
      skipContrib: skipW + 0.06 * Math.sin(i * 0.9),
      residualContrib: (1 - skipW) * (0.3 + 0.6 * Math.abs(Math.sin(i * 1.2 + 0.5))),
      norm: (parseFloat(data.residual.norm) / 10) * (0.6 + 0.4 * Math.abs(Math.cos(i * 0.7))),
    }));
  }, [data.residual]);

  return (
    <group position={position}>
      <Text position={[4, 2.2, 0.1]} fontSize={0.18} color={color} anchorX="center" anchorY="middle"
        outlineWidth={0.01} outlineColor="#0a1022">
        {t('componentDetail.runtimeParams')}
      </Text>

      {/* ── 左: Skip vs Residual 贡献 ── */}
      <group position={[0, 0.5, 0]}>
        <Text position={[1.2, 1.5, 0.05]} fontSize={0.11} color="#a78bfa" anchorX="center" anchorY="middle">
          {t('componentDetail.skipResidual')}
        </Text>
        {dimParams.map((p, i) => {
          const skipH = p.skipContrib * 0.8;
          const resH = p.residualContrib * 0.8;
          return (
            <group key={`sr${i}`} position={[0.55 * i, 0, 0]}>
              {/* Skip 分量 (蓝紫) */}
              <mesh position={[0, skipH / 2 + 0.2, 0]}>
                <boxGeometry args={[0.35, skipH, 0.06]} />
                <meshStandardMaterial color="#a78bfa" emissive="#a78bfa" emissiveIntensity={0.2} transparent opacity={0.8} />
              </mesh>
              {/* Residual 分量 (绿) */}
              <mesh position={[0, resH / 2 - 0.6, 0]}>
                <boxGeometry args={[0.35, resH, 0.06]} />
                <meshStandardMaterial color="#22c55e" emissive="#22c55e" emissiveIntensity={0.2} transparent opacity={0.7} />
              </mesh>
              <Text position={[0, -1.0, 0.05]} fontSize={0.07} color="#64748b" anchorX="center" anchorY="middle">
                {`d${i}`}
              </Text>
            </group>
          );
        })}
      </group>

      {/* ── 中: 范数增长 ── */}
      <group position={[5, 0.5, 0]}>
        <Text position={[1.2, 1.5, 0.05]} fontSize={0.11} color="#f97316" anchorX="center" anchorY="middle">
          {t('componentDetail.normPerDim')}
        </Text>
        {dimParams.map((p, i) => (
          <ParamBar key={`nd${i}`} label={`d${i}`} value={p.norm} color={getParamColor(p.norm)} x={0} y={0.8 - i * 0.4} w={1.8} h={0.2} t={t} />
        ))}
      </group>

      {/* ── 右: 汇总 ── */}
      <group position={[7.5, 0.5, 0]}>
        <StatCard label="ret" value={`${(data.residual.retention * 100).toFixed(1)}%`} color="#a78bfa" x={0} y={0.8} t={t} />
        <StatCard label="skip" value={data.residual.skipWeight} color="#22c55e" x={0} y={-0.1} t={t} />
        <StatCard label="‖r‖" value={data.residual.norm} color="#f97316" x={0} y={-1.0} t={t} />
        <Text position={[0, -1.7, 0.05]} fontSize={0.1} color="#94a3b8" anchorX="center" anchorY="middle">
          {data.residual.streamDir}
        </Text>
      </group>
    </group>
  );
}

// ═══════════════════════════════════════════════════════
// LayerRuntimeParamPanel - 统一入口: 根据组件类型选择参数面板
// ═══════════════════════════════════════════════════════
function LayerRuntimeParamPanel({ component, data, color, phase, position = [0, 0, 0], t }) {
  switch (component) {
    case 'ln':
      return <LNRuntimeParams data={data} color={color} position={position} t={t} />;
    case 'attention':
      return <AttentionRuntimeParams data={data} phase={phase} color={color} position={position} t={t} />;
    case 'ffn':
      return <FFNRuntimeParams data={data} phase={phase} color={color} position={position} t={t} />;
    case 'residual':
      return <ResidualRuntimeParams data={data} color={color} position={position} t={t} />;
    default:
      return null;
  }
}

// ═══════════════════════════════════════════════════════
// 信息区组件（上方）
// ═══════════════════════════════════════════════════════

function ParamRow({ label, value, color, yPos, active }) {
  return (
    <group position={[0, yPos, 0]}>
      <mesh position={[4, 0, 0]}>
        <boxGeometry args={[8, 0.55, 0.25]} />
        <meshStandardMaterial
          color={active ? color : '#1e293b'}
          transparent
          opacity={active ? 0.18 : 0.06}
          emissive={active ? color : '#000'}
          emissiveIntensity={active ? 0.15 : 0}
        />
      </mesh>
      <Text position={[0.4, 0.05, 0.18]} fontSize={0.3} color={active ? color : '#475569'} anchorX="left" anchorY="middle">
        {label}
      </Text>
      <Text position={[7.6, 0.05, 0.18]} fontSize={0.32} color={active ? '#ffffff' : '#64748b'} anchorX="right" anchorY="middle">
        {value}
      </Text>
    </group>
  );
}

function SignalBar({ value, maxVal = 1, color, label, yPos }) {
  const barMaxW = 5.5;
  const fillW = Math.max(0.1, (value / maxVal) * barMaxW);
  return (
    <group position={[0, yPos, 0.25]}>
      <mesh position={[3.5, 0, 0]}>
        <boxGeometry args={[barMaxW, 0.35, 0.06]} />
        <meshBasicMaterial color="#1e293b" transparent opacity={0.35} />
      </mesh>
      <mesh position={[0.75 + fillW / 2, 0, 0.04]}>
        <boxGeometry args={[fillW, 0.35, 0.1]} />
        <meshBasicMaterial color={color} transparent opacity={0.85} />
      </mesh>
      <Text position={[-0.1, 0, 0.08]} fontSize={0.26} color={color} anchorX="right" anchorY="middle">
        {label}
      </Text>
      <Text position={[6.5, 0, 0.08]} fontSize={0.24} color="#e2e8f0" anchorX="left" anchorY="middle">
        {(value * 100).toFixed(1)}%
      </Text>
    </group>
  );
}

// 信息标题区
function InfoHeader({ data, color, component, phase, t }) {
  const titleKeys = { ln: 'componentDetail.layerNorm', attention: 'componentDetail.attention', ffn: 'componentDetail.ffn', residual: 'componentDetail.residual' };
  const subLabelKeys = {
    ln1: 'componentDetail.preAttention', ln2: 'componentDetail.preFFN',
    qkv: 'componentDetail.qkvProjection', attn_score: 'componentDetail.attnScoring',
    softmax: 'componentDetail.softmaxNorm', attn_out: 'componentDetail.attnOutput',
    ffn_up: 'componentDetail.wUpProjection', ffn_act: 'componentDetail.siluActivation', ffn_down: 'componentDetail.wDownProjection',
    residual1: 'componentDetail.skipConnection1', residual2: 'componentDetail.skipConnection2',
  };
  return (
    <group>
      <Text position={[0.4, 0, 0.25]} fontSize={0.42} color={color} anchorX="left" anchorY="middle">
        {t(titleKeys[component] || component)}
      </Text>
      <Text position={[7.6, 0, 0.25]} fontSize={0.32} color={color} anchorX="right" anchorY="middle">
        {subLabelKeys[phase] ? t(subLabelKeys[phase]) : ''}
      </Text>
    </group>
  );
}

// LN 参数
function LNInfo({ data, color, phase, t }) {
  const rows = [
    { label: t('componentDetail.beta'), value: data.ln.beta },
    { label: t('componentDetail.gamma'), value: data.ln.gamma },
    { label: t('componentDetail.epsilon'), value: data.ln.eps.toString() },
    { label: t('componentDetail.mean'), value: data.ln.mean },
    { label: t('componentDetail.variance'), value: data.ln.var },
    { label: t('componentDetail.leakage'), value: `${(data.ln.leakage * 100).toFixed(1)}%` },
  ];
  return (
    <group>
      <InfoHeader data={data} color={color} component="ln" phase={phase} t={t} />
      {rows.map((r, i) => (
        <ParamRow key={r.label} label={r.label} value={r.value} color={color} yPos={-1.0 - i * 0.8} active />
      ))}
    </group>
  );
}

// Attention 参数
function AttentionInfo({ data, color, phase, t }) {
  const rows = [
    { label: t('componentDetail.strength'), value: data.attention.strength.toFixed(3) },
    { label: t('componentDetail.pattern'), value: data.attention.pattern },
    { label: t('componentDetail.activeHeads'), value: data.attention.topHeads },
    { label: t('componentDetail.avgAttention'), value: data.attention.avgAttn },
    { label: t('componentDetail.entropy'), value: data.attention.entropy },
    { label: 'cos(Q,K)', value: data.attention.qkCos },
    { label: t('componentDetail.ovScore'), value: data.attention.ovScore },
    { label: t('componentDetail.headsXDim'), value: `${data.attention.nHeads}×${data.attention.headDim}` },
  ];
  return (
    <group>
      <InfoHeader data={data} color={color} component="attention" phase={phase} t={t} />
      {rows.map((r, i) => (
        <ParamRow key={r.label} label={r.label} value={r.value} color={color} yPos={-1.0 - i * 0.8} active />
      ))}
      <SignalBar value={data.attention.strength} color={color} label={t('componentDetail.attn')} yPos={-1.0 - rows.length * 0.8 - 0.6} />
    </group>
  );
}

// FFN 参数
function FFNInfo({ data, color, phase, t }) {
  const rows = [
    { label: t('componentDetail.gain'), value: `${data.ffn.gain.toFixed(2)}x` },
    { label: t('componentDetail.projection'), value: `${data.ffn.dModel}→${data.ffn.mlpDim?.toLocaleString()}` },
    { label: t('componentDetail.siluGate'), value: data.ffn.siluGate },
    { label: t('componentDetail.direction'), value: data.ffn.upProj },
    { label: t('componentDetail.activeRatio'), value: data.ffn.actRatio },
    { label: t('componentDetail.topNeurons'), value: data.ffn.topNeurons },
  ];
  return (
    <group>
      <InfoHeader data={data} color={color} component="ffn" phase={phase} t={t} />
      {rows.map((r, i) => (
        <ParamRow key={r.label} label={r.label} value={r.value} color={color} yPos={-1.0 - i * 0.8} active />
      ))}
      <SignalBar value={Math.min(data.ffn.gain / 3, 1)} color={color} label={t('componentDetail.gain')} yPos={-1.0 - rows.length * 0.8 - 0.6} />
    </group>
  );
}

// Residual 参数
function ResidualInfo({ data, color, phase, t }) {
  const rows = [
    { label: t('componentDetail.retention'), value: `${(data.residual.retention * 100).toFixed(1)}%` },
    { label: t('componentDetail.skipWeight'), value: data.residual.skipWeight },
    { label: t('componentDetail.streamDir'), value: data.residual.streamDir },
    { label: t('componentDetail.normR'), value: data.residual.norm },
  ];
  return (
    <group>
      <InfoHeader data={data} color={color} component="residual" phase={phase} t={t} />
      {rows.map((r, i) => (
        <ParamRow key={r.label} label={r.label} value={r.value} color={color} yPos={-1.0 - i * 0.8} active />
      ))}
      <SignalBar value={data.residual.retention} color={color} label={t('componentDetail.ret')} yPos={-1.0 - rows.length * 0.8 - 0.6} />
    </group>
  );
}

// ═══════════════════════════════════════════════════════
// 3D模型区组件（下方）
// ═══════════════════════════════════════════════════════

// ── LN 3D模型: 标准化变换可视化（详细版） ──
function LNModel3D({ data, color, animProgress, t }) {
  const groupRef = useRef();
  const barRefs = useRef([]);
  const needleRefs = useRef([]);
  useFrame((state) => {
    // 动态波动：输入柱轻微晃动表示不稳定性
    barRefs.current.forEach((ref, i) => {
      if (ref) {
        const wobble = Math.sin(state.clock.elapsedTime * 2 + i * 0.7) * 0.04;
        ref.position.z = wobble;
      }
    });
    // 仪表盘指针动画
    needleRefs.current.forEach((ref, i) => {
      if (ref) {
        const flicker = Math.sin(state.clock.elapsedTime * 1.5 + i) * 0.15;
        ref.rotation.z = -Math.PI / 4 + flicker;
      }
    });
  });

  const barCount = 10;
  const gap = 0.85;
  const totalW = (barCount - 1) * gap;

  // 输入分布：各维度值差异大
  const rawHeights = Array.from({ length: barCount }).map((_, i) =>
    0.6 + 2.8 * Math.abs(Math.sin(i * 1.1 + 0.3 * i))
  );
  // 归一化后：接近均匀
  const normHeights = Array.from({ length: barCount }).map((_, i) =>
    1.5 + 0.15 * Math.sin(i * 0.4)
  );

  return (
    <group ref={groupRef}>
      {/* ═══ 公式标签 ═══ */}
      <Text position={[0, 3.8, 0]} fontSize={0.28} color={color} anchorX="center" anchorY="middle"
        outlineWidth={0.015} outlineColor="#0a1022">
        {'y = γ·(x-μ)/√(σ²+ε) + β'}
      </Text>

      {/* ═══ 左侧: 输入分布 ═══ */}
      <group position={[-1.5, 0, 0]}>
        <Text position={[0, 3.2, 0]} fontSize={0.24} color="#94a3b8" anchorX="center" anchorY="middle">
          {t('componentDetail.inputX')}
        </Text>
        {/* 不均匀柱状图 */}
        {rawHeights.map((h, i) => (
          <mesh key={`raw${i}`} ref={el => barRefs.current[i] = el} position={[-totalW / 2 + i * gap, h / 2, 0]}>
            <boxGeometry args={[0.5, h, 0.5]} />
            <meshStandardMaterial
              color="#64748b"
              emissive="#64748b"
              emissiveIntensity={0.08}
              transparent
              opacity={0.65}
            />
          </mesh>
        ))}
        {/* 均值线 μ */}
        <mesh position={[0, 1.8, 0.4]}>
          <boxGeometry args={[totalW + 0.5, 0.04, 0.04]} />
          <meshBasicMaterial color="#f97316" />
        </mesh>
        <Text position={[totalW / 2 + 0.8, 1.8, 0.4]} fontSize={0.18} color="#f97316" anchorX="left" anchorY="middle">
          {`μ=${data.ln.mean}`}
        </Text>
        {/* 方差标注 */}
        <Text position={[0, 3.0, 0.45]} fontSize={0.18} color="#f97316" anchorX="center" anchorY="middle"
          outlineWidth={0.02} outlineColor="#0a1022">
          {`σ²=${data.ln.var}`}
        </Text>
      </group>

      {/* ═══ 中间: 变换过程 ═══ */}
      <group position={[2.5, 0, 0]}>
        {/* 减均值 → 除标准差 → 缩放偏移 */}
        <Text position={[0, 3.2, 0]} fontSize={0.22} color={color} anchorX="center" anchorY="middle">
          {t('componentDetail.transform')}
        </Text>

        {/* Step 1: x - μ */}
        <mesh position={[0, 2.0, 0]}>
          <boxGeometry args={[1.8, 0.5, 0.3]} />
          <meshStandardMaterial color="#334155" emissive={color} emissiveIntensity={0.15} transparent opacity={0.8} />
        </mesh>
        <Text position={[0, 2.0, 0.25]} fontSize={0.18} color="#e2e8f0" anchorX="center" anchorY="middle">
          x - μ
        </Text>

        {/* 箭头 ↓ */}
        <mesh position={[0, 1.3, 0]} rotation={[0, 0, 0]}>
          <coneGeometry args={[0.2, 0.5, 6]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.3} />
        </mesh>

        {/* Step 2: /√(σ²+ε) */}
        <mesh position={[0, 0.5, 0]}>
          <boxGeometry args={[1.8, 0.5, 0.3]} />
          <meshStandardMaterial color="#334155" emissive={color} emissiveIntensity={0.25} transparent opacity={0.8} />
        </mesh>
        <Text position={[0, 0.5, 0.25]} fontSize={0.16} color="#e2e8f0" anchorX="center" anchorY="middle">
          /√(σ²+ε)
        </Text>

        <Text position={[1.3, 0.5, 0.35]} fontSize={0.14} color="#22d3ee" anchorX="center" anchorY="middle">
          ε
        </Text>

        {/* 箭头 ↓ */}
        <mesh position={[0, -0.2, 0]} rotation={[0, 0, 0]}>
          <coneGeometry args={[0.2, 0.5, 6]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.3} />
        </mesh>

        {/* Step 3: γ· + β */}
        <mesh position={[0, -1.0, 0]}>
          <boxGeometry args={[1.8, 0.5, 0.3]} />
          <meshStandardMaterial color="#334155" emissive={color} emissiveIntensity={0.35} transparent opacity={0.8} />
        </mesh>
        <Text position={[0, -1.0, 0.25]} fontSize={0.18} color="#e2e8f0" anchorX="center" anchorY="middle">
          γ· + β
        </Text>

        <Text position={[-1.3, -1.0, 0.4]} fontSize={0.14} color={color} anchorX="center" anchorY="middle">
          {`γ=${data.ln.gamma}`}
        </Text>

        {/* β 偏移指示器 - 偏移小球 */}
        <mesh position={[1.3, -1.0, 0]}>
          <sphereGeometry args={[0.2, 12, 12]} />
          <meshStandardMaterial color="#a78bfa" emissive="#a78bfa" emissiveIntensity={0.5} />
        </mesh>
        <Text position={[1.3, -1.0, 0.35]} fontSize={0.14} color="#a78bfa" anchorX="center" anchorY="middle">
          {`β=${data.ln.beta}`}
        </Text>
      </group>

      {/* ═══ 右侧: 归一化输出 ═══ */}
      <group position={[6.5, 0, 0]}>
        <Text position={[0, 3.2, 0]} fontSize={0.24} color={color} anchorX="center" anchorY="middle">
          {t('componentDetail.outputY')}
        </Text>
        {/* 均匀柱状图 */}
        {normHeights.map((h, i) => (
          <mesh key={`norm${i}`} position={[-totalW / 2 + i * gap, h / 2, 0]}>
            <boxGeometry args={[0.5, h, 0.5]} />
            <meshStandardMaterial
              color={color}
              emissive={color}
              emissiveIntensity={0.2 + 0.08 * (1 - Math.abs(i - barCount / 2) / (barCount / 2))}
              transparent
              opacity={0.85}
            />
          </mesh>
        ))}
        {/* 均匀输出线 */}
        <mesh position={[0, 1.6, 0.4]}>
          <boxGeometry args={[totalW + 0.5, 0.03, 0.03]} />
          <meshBasicMaterial color={color} transparent opacity={0.6} />
        </mesh>
        <Text position={[totalW / 2 + 0.5, 1.6, 0.4]} fontSize={0.16} color={color} anchorX="left" anchorY="middle">
          ≈1.0
        </Text>

        {/* Leakage 指示 */}
        <Text position={[0, -0.5, 0.1]} fontSize={0.16} color="#f97316" anchorX="center" anchorY="middle"
          outlineWidth={0.02} outlineColor="#0a1022">
          {`${t('componentDetail.leak')}: ${(data.ln.leakage * 100).toFixed(1)}%`}
        </Text>
      </group>

      {/* ═══ 底部: 统计指标 ═══ */}
      <group position={[2.5, -3.5, 0]}>
        <group position={[-1.5, 0, 0]}>
          <mesh ref={el => needleRefs.current[0] = el} position={[0, 0, 0.1]} rotation={[0, 0, -Math.PI / 4]}>
            <boxGeometry args={[0.02, 0.4, 0.02]} />
            <meshBasicMaterial color="#f97316" />
          </mesh>
          <Text position={[0, -0.7, 0]} fontSize={0.14} color="#94a3b8" anchorX="center" anchorY="middle">
            {`μ=${data.ln.mean}`}
          </Text>
        </group>
        <group position={[0, 0, 0]}>
          <mesh ref={el => needleRefs.current[1] = el} position={[0, 0, 0.1]} rotation={[0, 0, -Math.PI / 4]}>
            <boxGeometry args={[0.02, 0.4, 0.02]} />
            <meshBasicMaterial color={color} />
          </mesh>
          <Text position={[0, -0.7, 0]} fontSize={0.14} color="#94a3b8" anchorX="center" anchorY="middle">
            {`σ²=${data.ln.var}`}
          </Text>
        </group>
        <group position={[1.5, 0, 0]}>
          <mesh ref={el => needleRefs.current[2] = el} position={[0, 0, 0.1]} rotation={[0, 0, -Math.PI / 4]}>
            <boxGeometry args={[0.02, 0.4, 0.02]} />
            <meshBasicMaterial color="#22d3ee" />
          </mesh>
          <Text position={[0, -0.7, 0]} fontSize={0.14} color="#94a3b8" anchorX="center" anchorY="middle">
            ε=1e-5
          </Text>
        </group>
      </group>

      {/* ═══ 运行时参数面板 ═══ */}
      <LayerRuntimeParamPanel
        component="ln"
        data={data}
        color={color}
        position={[-4, -5, 0]}
        t={t}
      />
    </group>
  );
}

// ── Attention 3D模型: 多头注意力模式（详细版） ──
function AttentionModel3D({ data, color, phase, t }) {
  const groupRef = useRef();
  const headRefs = useRef([]);
  const heatRefs = useRef([]);
  useFrame((state) => {
    headRefs.current.forEach((ref, i) => {
      if (ref) ref.rotation.z = state.clock.elapsedTime * (0.4 + i * 0.08);
    });
    // 注意力热力图脉冲
    heatRefs.current.forEach((ref, i) => {
      if (ref) {
        const pulse = 0.5 + 0.3 * Math.sin(state.clock.elapsedTime * 2 + i * 0.5);
        ref.material.emissiveIntensity = pulse * 0.4;
      }
    });
  });

  const nHeads = 6;
  const headRadius = 2.8;
  const headPositions = Array.from({ length: nHeads }).map((_, i) => {
    const angle = (i / nHeads) * Math.PI * 2;
    return [Math.cos(angle) * headRadius, Math.sin(angle) * headRadius, 0];
  });

  // 注意力热力图 (5x5 grid)
  const gridSize = 5;
  const cellSize = 0.55;
  const gridOffset = -(gridSize - 1) * cellSize / 2;

  // 生成模拟注意力权重（对角线附近权重高）
  const attnWeights = useMemo(() => {
    const w = [];
    for (let r = 0; r < gridSize; r++) {
      for (let c = 0; c < gridSize; c++) {
        const dist = Math.abs(r - c);
        w.push(Math.max(0.05, 1.0 - dist * 0.25 + 0.1 * Math.sin(r * c)));
      }
    }
    return w;
  }, []);

  return (
    <group ref={groupRef}>
      {/* ═══ 顶部: 当前阶段标签 ═══ */}
      <Text position={[0, 4.2, 0]} fontSize={0.28} color={color} anchorX="center" anchorY="middle"
        outlineWidth={0.015} outlineColor="#0a1022">
        {phase === 'qkv' ? 'Q = X·Wq, K = X·Wk, V = X·Wv' :
         phase === 'attn_score' ? 'Score = Q·Kᵀ / √d_k' :
         phase === 'softmax' ? 'Attn = softmax(Score)' :
         phase === 'attn_out' ? 'Out = Attn·V · Wo' :
         'Multi-Head Attention'}
      </Text>

      {/* ═══ Q/K/V 分离矩阵 ═══ */}
      <group position={[-5.5, 1.5, 0]}>
        {/* Q 矩阵 */}
        <mesh position={[0, 0, 0]}>
          <boxGeometry args={[1.0, 1.5, 0.25]} />
          <meshStandardMaterial color="#60a5fa" emissive="#60a5fa" emissiveIntensity={0.3} transparent opacity={0.7} />
        </mesh>
        <Text position={[0, 1.0, 0.2]} fontSize={0.22} color="#60a5fa" anchorX="center" anchorY="middle">
          Q
        </Text>
        {/* Q 内部行向量 */}
        {[0, 1, 2].map(i => (
          <mesh key={`qr${i}`} position={[0, 0.4 - i * 0.4, 0.15]}>
            <boxGeometry args={[0.7, 0.08, 0.05]} />
            <meshBasicMaterial color="#93c5fd" transparent opacity={0.6} />
          </mesh>
        ))}

        {/* K 矩阵 */}
        <mesh position={[1.4, 0, 0]}>
          <boxGeometry args={[1.0, 1.5, 0.25]} />
          <meshStandardMaterial color="#f472b6" emissive="#f472b6" emissiveIntensity={0.3} transparent opacity={0.7} />
        </mesh>
        <Text position={[1.4, 1.0, 0.2]} fontSize={0.22} color="#f472b6" anchorX="center" anchorY="middle">
          K
        </Text>
        {[0, 1, 2].map(i => (
          <mesh key={`kr${i}`} position={[1.4, 0.4 - i * 0.4, 0.15]}>
            <boxGeometry args={[0.7, 0.08, 0.05]} />
            <meshBasicMaterial color="#f9a8d4" transparent opacity={0.6} />
          </mesh>
        ))}

        {/* V 矩阵 */}
        <mesh position={[2.8, 0, 0]}>
          <boxGeometry args={[1.0, 1.5, 0.25]} />
          <meshStandardMaterial color="#34d399" emissive="#34d399" emissiveIntensity={0.3} transparent opacity={0.7} />
        </mesh>
        <Text position={[2.8, 1.0, 0.2]} fontSize={0.22} color="#34d399" anchorX="center" anchorY="middle">
          V
        </Text>
        {[0, 1, 2].map(i => (
          <mesh key={`vr${i}`} position={[2.8, 0.4 - i * 0.4, 0.15]}>
            <boxGeometry args={[0.7, 0.08, 0.05]} />
            <meshBasicMaterial color="#6ee7b7" transparent opacity={0.6} />
          </mesh>
        ))}
      </group>

      {/* ═══ 注意力热力图 ═══ */}
      <group position={[0.5, 1.5, 0]}>
        <Text position={[0, 2.5, 0]} fontSize={0.2} color={color} anchorX="center" anchorY="middle">
          {t('componentDetail.attentionMatrix')}
        </Text>
        {attnWeights.map((w, idx) => {
          const r = Math.floor(idx / gridSize);
          const c = idx % gridSize;
          const hue = w > 0.6 ? color : w > 0.3 ? '#475569' : '#1e293b';
          return (
            <mesh key={`heat${idx}`}
              ref={el => heatRefs.current[idx] = el}
              position={[gridOffset + c * cellSize, gridOffset + r * cellSize, 0]}>
              <boxGeometry args={[cellSize * 0.85, cellSize * 0.85, w * 0.4]} />
              <meshStandardMaterial
                color={hue}
                emissive={hue}
                emissiveIntensity={w * 0.3}
                transparent
                opacity={0.3 + w * 0.6}
              />
            </mesh>
          );
        })}
        {/* 行列标签 */}
        {Array.from({ length: gridSize }).map((_, i) => (
          <group key={`ax${i}`}>
            <Text position={[gridOffset + i * cellSize, gridOffset - cellSize * 0.8, 0]} fontSize={0.12} color="#64748b" anchorX="center" anchorY="middle">
              {`t${i}`}
            </Text>
            <Text position={[gridOffset - cellSize * 0.8, gridOffset + i * cellSize, 0]} fontSize={0.12} color="#64748b" anchorX="center" anchorY="middle">
              {`t${i}`}
            </Text>
          </group>
        ))}
      </group>

      {/* ═══ 注意力头 ═══ */}
      {headPositions.map((pos, i) => {
        const active = i < data.attention.strength * nHeads;
        return (
          <group key={i} position={pos}>
            <mesh ref={el => headRefs.current[i] = el}>
              <sphereGeometry args={[0.35, 12, 12]} />
              <meshStandardMaterial
                color={active ? color : '#334155'}
                emissive={active ? color : '#000'}
                emissiveIntensity={active ? 0.35 : 0}
                transparent
                opacity={active ? 0.8 : 0.25}
              />
            </mesh>
            {/* 头内部小热力图 */}
            {active && [0, 1, 2, 3].map(j => (
              <mesh key={`hh${j}`} position={[(j % 2 - 0.5) * 0.22, (Math.floor(j / 2) - 0.5) * 0.22, 0.1]}>
                <boxGeometry args={[0.15, 0.15, 0.05]} />
                <meshBasicMaterial color={color} transparent opacity={0.3 + Math.random() * 0.4} />
              </mesh>
            ))}
            <Text position={[0, -0.8, 0]} fontSize={0.17} color={active ? color : '#475569'} anchorX="center" anchorY="middle">
              {`H${i}`}
            </Text>
          </group>
        );
      })}

      {/* 连线: 中心→头 */}
      {headPositions.map((pos, i) => (
        <mesh key={`line${i}`} position={[pos[0] / 2, pos[1] / 2, 0]}
          rotation={[0, 0, Math.atan2(pos[1], pos[0])]}>
          <boxGeometry args={[Math.sqrt(pos[0] ** 2 + pos[1] ** 2) * 0.85, 0.03, 0.03]} />
          <meshBasicMaterial
            color={i < data.attention.strength * nHeads ? color : '#1e293b'}
            transparent opacity={i < data.attention.strength * nHeads ? 0.35 : 0.08}
          />
        </mesh>
      ))}

      {/* ═══ 底部: Softmax 分布曲线 ═══ */}
      <group position={[0, -3.5, 0]}>
        <Text position={[-3.5, 0.6, 0]} fontSize={0.18} color={color} anchorX="left" anchorY="middle">
          {t('componentDetail.softmax')}
        </Text>
        {[0, 1, 2, 3, 4].map(i => {
          const h = i === 2 ? 1.2 : i === 1 || i === 3 ? 0.5 : 0.15; // 峰值在中间
          return (
            <mesh key={`sm${i}`} position={[-2.5 + i * 1.2, h / 2, 0]}>
              <boxGeometry args={[0.6, h, 0.2]} />
              <meshStandardMaterial
                color={i === 2 ? color : '#475569'}
                emissive={i === 2 ? color : '#000'}
                emissiveIntensity={i === 2 ? 0.4 : 0}
                transparent opacity={i === 2 ? 0.9 : 0.4}
              />
            </mesh>
          );
        })}

        {/* 统计指标 */}
        <group position={[4, 0, 0]}>
          <Text position={[0, 0.3, 0.1]} fontSize={0.14} color={color} anchorX="center" anchorY="middle"
            outlineWidth={0.02} outlineColor="#0a1022">
            {`cos(Q,K) = ${data.attention.qkCos}`}
          </Text>
          <Text position={[0, 0, 0.1]} fontSize={0.14} color="#94a3b8" anchorX="center" anchorY="middle"
            outlineWidth={0.02} outlineColor="#0a1022">
            {`Entropy = ${data.attention.entropy}`}
          </Text>
          <Text position={[0, -0.3, 0.1]} fontSize={0.14} color="#94a3b8" anchorX="center" anchorY="middle"
            outlineWidth={0.02} outlineColor="#0a1022">
            {`OV Score = ${data.attention.ovScore}`}
          </Text>
        </group>
      </group>

      {/* 模式标签 */}
      <Text position={[0, -5, 0]} fontSize={0.25} color={color} anchorX="center" anchorY="middle"
        outlineWidth={0.01} outlineColor="#0a1022">
        {`${t('componentDetail.pattern')}: ${data.attention.pattern} | ${t('componentDetail.activeHeads')}: ${data.attention.nHeads}×${data.attention.headDim}`}
      </Text>

      {/* ═══ 运行时参数面板 ═══ */}
      <LayerRuntimeParamPanel
        component="attention"
        data={data}
        color={color}
        phase={phase}
        position={[-4, -6.5, 0]}
        t={t}
      />
    </group>
  );
}

// ── FFN 3D模型: 两层投影结构（详细版） ──
function FFNModel3D({ data, color, phase, t }) {
  const groupRef = useRef();
  const particleRefs = useRef([]);
  const siluRefs = useRef([]);
  useFrame((state) => {
    // 信号粒子动画
    particleRefs.current.forEach((ref, i) => {
      if (ref) {
        const t = (state.clock.elapsedTime * 0.6 + i * 0.4) % 3;
        const stage = Math.floor(t);
        const frac = t - stage;
        if (stage === 0) { // 输入→隐藏
          ref.position.x = -3.5 + frac * 3.5;
          ref.position.y = 1.5 * (1 - frac) + (Math.sin(i * 1.5) * 1.5) * frac;
        } else if (stage === 1) { // 隐藏中
          ref.position.x = 0 + Math.sin(state.clock.elapsedTime * 2 + i) * 0.3;
          ref.position.y = Math.sin(i * 1.5) * 1.5 + Math.sin(state.clock.elapsedTime * 3 + i) * 0.2;
        } else { // 隐藏→输出
          ref.position.x = frac * 3.5;
          ref.position.y = (Math.sin(i * 1.5) * 1.5) * (1 - frac) + 1.5 * frac;
        }
        ref.material.opacity = 0.5 + 0.3 * Math.sin(state.clock.elapsedTime * 3 + i);
      }
    });
    // SiLU 曲线脉冲
    siluRefs.current.forEach((ref, i) => {
      if (ref) {
        ref.material.emissiveIntensity = 0.2 + 0.15 * Math.sin(state.clock.elapsedTime * 2 + i * 0.3);
      }
    });
  });

  const dModel = 4;
  const mlpDim = 7;
  const gapX = 3.5;
  const layerGap = 0.6;

  return (
    <group ref={groupRef}>
      {/* ═══ 顶部公式 ═══ */}
      <Text position={[0, 4.0, 0]} fontSize={0.26} color={color} anchorX="center" anchorY="middle"
        outlineWidth={0.015} outlineColor="#0a1022">
        {phase === 'ffn_up' ? 'x_up = W_up·x + b_up' :
         phase === 'ffn_act' ? 'gate = SiLU(x) · x_up' :
         'x_out = W_down·gate + b_down'}
      </Text>

      {/* ═══ 输入层 (d_model 维) ═══ */}
      <Text position={[-gapX, dModel * layerGap / 2 + 0.8, 0]} fontSize={0.2} color="#94a3b8" anchorX="center" anchorY="middle">
        d_model
      </Text>
      {Array.from({ length: dModel }).map((_, i) => (
        <mesh key={`in${i}`} position={[-gapX, dModel * layerGap / 2 - i * layerGap - layerGap / 2, 0]}>
          <sphereGeometry args={[0.28, 12, 12]} />
          <meshStandardMaterial color="#64748b" emissive="#64748b" emissiveIntensity={0.12} transparent opacity={0.7} />
        </mesh>
      ))}

      {/* ═══ W_up 投影层 ═══ */}
      <group position={[-gapX / 2, 0, 0]}>
        <Text position={[0, mlpDim * layerGap / 2 + 0.5, 0.1]} fontSize={0.18} color="#f59e0b" anchorX="center" anchorY="middle"
          outlineWidth={0.02} outlineColor="#0a1022">
          W_up
        </Text>
        {/* 权重条 - 可视化投影方向 */}
        {Array.from({ length: 4 }).map((_, i) => (
          <mesh key={`wu${i}`} position={[-0.2 + i * 0.2, 0, 0.08]}>
            <boxGeometry args={[0.08, mlpDim * layerGap, 0.03]} />
            <meshBasicMaterial color="#f59e0b" transparent opacity={0.2 + 0.1 * Math.sin(i)} />
          </mesh>
        ))}
      </group>

      {/* ═══ 隐藏层 (mlp_dim 维, 放大) ═══ */}
      <Text position={[0, mlpDim * layerGap / 2 + 0.8, 0]} fontSize={0.2} color={color} anchorX="center" anchorY="middle">
        mlp_dim
      </Text>
      {Array.from({ length: mlpDim }).map((_, i) => {
        const isActive = i < Math.round(data.ffn.gain * 2.5);
        return (
          <group key={`hid${i}`} position={[0, mlpDim * layerGap / 2 - i * layerGap - layerGap / 2, 0]}>
            <mesh>
              <sphereGeometry args={[isActive ? 0.35 : 0.2, 12, 12]} />
              <meshStandardMaterial
                color={isActive ? color : '#1e293b'}
                emissive={isActive ? color : '#000'}
                emissiveIntensity={isActive ? 0.35 : 0}
                transparent
                opacity={isActive ? 0.9 : 0.2}
              />
            </mesh>
            {/* 神经元标签（仅激活的） */}
            {isActive && (
              <Text position={[0.7, 0, 0]} fontSize={0.13} color={color} anchorX="left" anchorY="middle">
                {i === 0 ? 'N42' : i === 1 ? 'N187' : i === 2 ? 'N512' : `N${i * 128}`}
              </Text>
            )}
          </group>
        );
      })}

      {/* ═══ SiLU 门控曲线 ═══ */}
      <group position={[2.5, 0, 0]}>
        <Text position={[0, mlpDim * layerGap / 2 + 0.8, 0]} fontSize={0.18} color="#fb923c" anchorX="center" anchorY="middle">
          {t('componentDetail.siluGateLabel')}
        </Text>
        {/* SiLU 曲线: f(x) = x * sigmoid(x) */}
        {Array.from({ length: 12 }).map((_, i) => {
          const x = (i - 6) * 0.4;
          const sigmoid = 1 / (1 + Math.exp(-x));
          const silu = x * sigmoid;
          const y = silu * 0.8;
          return (
            <mesh key={`silu${i}`}
              ref={el => siluRefs.current[i] = el}
              position={[0, y + mlpDim * layerGap / 4, 0.3]}>
              <sphereGeometry args={[0.1, 8, 8]} />
              <meshStandardMaterial
                color="#fb923c"
                emissive="#fb923c"
                emissiveIntensity={0.2}
                transparent
                opacity={0.7}
              />
            </mesh>
          );
        })}
        {/* x轴 */}
        <mesh position={[0, mlpDim * layerGap / 4, 0.25]}>
          <boxGeometry args={[5, 0.02, 0.02]} />
          <meshBasicMaterial color="#334155" transparent opacity={0.5} />
        </mesh>
        {/* 公式 */}
        <Text position={[0, -0.5, 0.3]} fontSize={0.16} color="#fb923c" anchorX="center" anchorY="middle"
          outlineWidth={0.02} outlineColor="#0a1022">
          x·σ(x)
        </Text>
        {/* 门控值 */}
        <Text position={[0, -1.2, 0.1]} fontSize={0.14} color="#fb923c" anchorX="center" anchorY="middle"
          outlineWidth={0.02} outlineColor="#0a1022">
          {`gate = ${data.ffn.siluGate}`}
        </Text>
      </group>

      {/* ═══ W_down 投影层 ═══ */}
      <group position={[gapX / 2, 0, 0]}>
        <Text position={[0, dModel * layerGap / 2 + 0.5, 0.1]} fontSize={0.18} color="#f97316" anchorX="center" anchorY="middle"
          outlineWidth={0.02} outlineColor="#0a1022">
          W_down
        </Text>
        {Array.from({ length: 4 }).map((_, i) => (
          <mesh key={`wd${i}`} position={[-0.2 + i * 0.2, 0, 0.08]}>
            <boxGeometry args={[0.08, dModel * layerGap, 0.03]} />
            <meshBasicMaterial color="#f97316" transparent opacity={0.2 + 0.1 * Math.cos(i)} />
          </mesh>
        ))}
      </group>

      {/* ═══ 输出层 (d_model 维) ═══ */}
      <Text position={[gapX, dModel * layerGap / 2 + 0.8, 0]} fontSize={0.2} color="#94a3b8" anchorX="center" anchorY="middle">
        d_model
      </Text>
      {Array.from({ length: dModel }).map((_, i) => (
        <mesh key={`out${i}`} position={[gapX, dModel * layerGap / 2 - i * layerGap - layerGap / 2, 0]}>
          <sphereGeometry args={[0.28, 12, 12]} />
          <meshStandardMaterial color="#64748b" emissive="#64748b" emissiveIntensity={0.12} transparent opacity={0.7} />
        </mesh>
      ))}

      {/* ═══ 连接线: 输入→隐藏 (稀疏) ═══ */}
      {Array.from({ length: dModel }).map((_, i) =>
        Array.from({ length: mlpDim }).map((_, j) => {
          const isStrong = Math.sin(i * 2.7 + j * 1.3) > 0.3;
          const strength = isStrong ? 0.15 : 0.02;
          return (
            <mesh key={`l1-${i}-${j}`}
              position={[-gapX / 2, (dModel * layerGap / 2 - i * layerGap + mlpDim * layerGap / 2 - j * layerGap) / 2 - layerGap / 2, 0]}
              rotation={[0, 0, Math.atan2(-(mlpDim * layerGap / 2 - j * layerGap - dModel * layerGap / 2 + i * layerGap), gapX)]}>
              <boxGeometry args={[gapX * 0.95, 0.025, 0.025]} />
              <meshBasicMaterial color={isStrong ? color : '#1e293b'} transparent opacity={strength} />
            </mesh>
          );
        })
      )}

      {/* ═══ 连接线: 隐藏→输出 (稀疏) ═══ */}
      {Array.from({ length: mlpDim }).map((_, i) =>
        Array.from({ length: dModel }).map((_, j) => {
          const isStrong = Math.sin(i * 1.9 + j * 3.1) > 0.3;
          const strength = isStrong ? 0.15 : 0.02;
          return (
            <mesh key={`l2-${i}-${j}`}
              position={[gapX / 2, (mlpDim * layerGap / 2 - i * layerGap + dModel * layerGap / 2 - j * layerGap) / 2 - layerGap / 2, 0]}
              rotation={[0, 0, Math.atan2(-(dModel * layerGap / 2 - j * layerGap - mlpDim * layerGap / 2 + i * layerGap), gapX)]}>
              <boxGeometry args={[gapX * 0.95, 0.025, 0.025]} />
              <meshBasicMaterial color={isStrong ? '#f97316' : '#1e293b'} transparent opacity={strength} />
            </mesh>
          );
        })
      )}

      {/* ═══ 信号粒子 (动画) ═══ */}
      {[0, 1, 2, 3, 4, 5].map((i) => (
        <mesh key={`p${i}`} ref={el => particleRefs.current[i] = el} position={[0, 0, 0.5]}>
          <sphereGeometry args={[0.12, 8, 8]} />
          <meshBasicMaterial color={color} transparent opacity={0.7} />
        </mesh>
      ))}

      {/* ═══ 底部统计 ═══ */}
      <group position={[0, -4.5, 0]}>
        <Text position={[-2.5, 0.2, 0.1]} fontSize={0.16} color={color} anchorX="left" anchorY="middle"
          outlineWidth={0.02} outlineColor="#0a1022">
          {`${t('componentDetail.active')}: ${(data.ffn.actRatio * 100).toFixed(1)}%`}
        </Text>
        <Text position={[0, 0.2, 0.1]} fontSize={0.16} color="#94a3b8" anchorX="left" anchorY="middle"
          outlineWidth={0.02} outlineColor="#0a1022">
          {`${data.ffn.dModel}→${data.ffn.mlpDim?.toLocaleString()}`}
        </Text>
        <Text position={[2.5, 0.2, 0.1]} fontSize={0.16} color="#fb923c" anchorX="left" anchorY="middle"
          outlineWidth={0.02} outlineColor="#0a1022">
          {`${t('componentDetail.dir')}: ${data.ffn.upProj}`}
        </Text>
        <Text position={[-2.5, -0.2, 0.1]} fontSize={0.14} color="#64748b" anchorX="left" anchorY="middle">
          {`${t('componentDetail.top')}: ${data.ffn.topNeurons}`}
        </Text>
        <Text position={[0, -0.2, 0.1]} fontSize={0.14} color="#64748b" anchorX="left" anchorY="middle"
          outlineWidth={0.02} outlineColor="#0a1022">
          {`${t('componentDetail.gain')}: ${data.ffn.gain.toFixed(2)}x`}
        </Text>
      </group>

      {/* ═══ 运行时参数面板 ═══ */}
      <LayerRuntimeParamPanel
        component="ffn"
        data={data}
        color={color}
        phase={phase}
        position={[-4, -6, 0]}
        t={t}
      />
    </group>
  );
}

// ── Residual 3D模型: 跳跃连接弧（详细版） ──
function ResidualModel3D({ data, color, t }) {
  const groupRef = useRef();
  const flowRefs = useRef([]);
  const subLayerFlowRefs = useRef([]);
  useFrame((state) => {
    // 跳跃连接弧上粒子
    flowRefs.current.forEach((ref, i) => {
      if (ref) {
        const prog = (state.clock.elapsedTime * 0.5 + i * 0.25) % 1;
        const arcX = -3.5 + prog * 7;
        const arcY = -1 + Math.sin(prog * Math.PI) * 3.2;
        ref.position.x = arcX;
        ref.position.y = arcY;
        ref.material.opacity = 0.5 + 0.35 * Math.sin(prog * Math.PI);
        // 粒子大小随位置变化
        const s = 0.6 + 0.4 * Math.sin(prog * Math.PI);
        ref.scale.set(s, s, s);
      }
    });
    // 子模块内部粒子
    subLayerFlowRefs.current.forEach((ref, i) => {
      if (ref) {
        const prog = (state.clock.elapsedTime * 1.5 + i * 0.5) % 1;
        ref.position.y = -1.0 + (prog - 0.5) * 1.2;
        ref.position.z = 0.4 + 0.2 * Math.sin(prog * Math.PI);
        ref.material.opacity = 0.4 + 0.4 * Math.sin(prog * Math.PI);
      }
    });
  });

  // 范数增长指示条（逐层可视化）
  const normVal = parseFloat(data.residual.norm) || 3.0;
  const normBarH = Math.min(normVal / 10 * 3, 3);

  return (
    <group ref={groupRef}>
      {/* ═══ 顶部公式 ═══ */}
      <Text position={[0, 3.8, 0]} fontSize={0.28} color={color} anchorX="center" anchorY="middle"
        outlineWidth={0.015} outlineColor="#0a1022">
        output = x + F(x)
      </Text>

      {/* ═══ 主路径 (底部直线): x 流 ═══ */}
      <group>
        {/* 输入标签 */}
        <Text position={[-4.5, -0.2, 0]} fontSize={0.24} color="#94a3b8" anchorX="center" anchorY="middle">
          x
        </Text>
        <mesh position={[-4.0, -1, 0]}>
          <sphereGeometry args={[0.25, 12, 12]} />
          <meshStandardMaterial color="#64748b" emissive="#64748b" emissiveIntensity={0.2} />
        </mesh>

        {/* 主路径管道 */}
        <mesh position={[0, -1, 0]}>
          <boxGeometry args={[7, 0.2, 0.2]} />
          <meshStandardMaterial color="#475569" transparent opacity={0.4} />
        </mesh>
        {/* 管道内流线 */}
        {[-2.5, -0.8, 0.8, 2.5].map((x, i) => (
          <mesh key={`pipe${i}`} position={[x, -1, 0.15]}>
            <boxGeometry args={[0.6, 0.06, 0.06]} />
            <meshBasicMaterial color="#94a3b8" transparent opacity={0.3} />
          </mesh>
        ))}

        {/* 输出标签 */}
        <Text position={[4.5, -0.2, 0]} fontSize={0.22} color={color} anchorX="center" anchorY="middle">
          x+F(x)
        </Text>
        <mesh position={[4.0, -1, 0]}>
          <sphereGeometry args={[0.25, 12, 12]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.3} />
        </mesh>
      </group>

      {/* ═══ 子模块 F(x) (中间) ═══ */}
      <group position={[0, -1, 0]}>
        {/* 子模块外壳 */}
        <mesh>
          <boxGeometry args={[3.1, 1.5, 0.05]} />
          <meshBasicMaterial color={color} transparent opacity={0.2} wireframe />
        </mesh>
        <Text position={[0, 0, 0.45]} fontSize={0.24} color="#e2e8f0" anchorX="center" anchorY="middle"
          outlineWidth={0.02} outlineColor="#0a1022">
          F(x)
        </Text>
        {/* 子模块内部微结构 */}
        {[-0.8, 0, 0.8].map((x, i) => (
          <mesh key={`sub${i}`} position={[x, 0, 0.25]}>
            <boxGeometry args={[0.5, 0.4, 0.15]} />
            <meshStandardMaterial
              color={color}
              emissive={color}
              emissiveIntensity={0.15}
              transparent
              opacity={0.35}
            />
          </mesh>
        ))}
        {/* 子模块内流动粒子 */}
        {[-0.5, 0.5].map((x, i) => (
          <mesh key={`sp${i}`} ref={el => subLayerFlowRefs.current[i] = el} position={[x, -1, 0.4]}>
            <sphereGeometry args={[0.1, 8, 8]} />
            <meshBasicMaterial color={color} transparent opacity={0.6} />
          </mesh>
        ))}
      </group>

      {/* ═══ 跳跃连接弧标签 ═══ */}
      <Text position={[0, 4.2, 0]} fontSize={0.2} color={color} anchorX="center" anchorY="middle">
        {t('componentDetail.skipConnection')}
      </Text>

      {/* ═══ 加法节点 ⊕ ═══ */}
      <group position={[3.5, -1, 0]}>
        <mesh>
          <sphereGeometry args={[0.45, 16, 16]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.4} />
        </mesh>
        <Text position={[0, 0, 0.5]} fontSize={0.35} color="#fff" anchorX="center" anchorY="middle">
          ⊕
        </Text>
      </group>

      {/* ═══ 跳跃弧上流动粒子 ═══ */}
      {[0, 1, 2, 3, 4, 5].map((i) => (
        <mesh key={i} ref={el => flowRefs.current[i] = el} position={[-3.5, 2, 0.3]}>
          <sphereGeometry args={[0.15, 8, 8]} />
          <meshBasicMaterial color={color} transparent opacity={0.7} />
        </mesh>
      ))}

      {/* ═══ 右侧: 残差流分解 ═══ */}
      <group position={[6.5, 1.0, 0]}>
        <Text position={[0, 2.5, 0]} fontSize={0.2} color={color} anchorX="center" anchorY="middle">
          {t('componentDetail.residualStream')}
        </Text>
        {/* 流向量可视化 */}
        {['lexical', 'semantic', 'decision'].map((dir, i) => {
          const isCurrent = data.residual.streamDir === dir;
          return (
            <group key={dir} position={[0, 1.5 - i * 1.0, 0]}>
              <mesh>
                <boxGeometry args={[isCurrent ? 2.2 : 1.6, 0.4, 0.2]} />
                <meshStandardMaterial
                  color={isCurrent ? color : '#1e293b'}
                  emissive={isCurrent ? color : '#000'}
                  emissiveIntensity={isCurrent ? 0.3 : 0}
                  transparent
                  opacity={isCurrent ? 0.8 : 0.3}
                />
              </mesh>
              <Text position={[0, 0, 0.15]} fontSize={0.16} color={isCurrent ? '#fff' : '#475569'} anchorX="center" anchorY="middle">
                {dir}
              </Text>
            </group>
          );
        })}
      </group>

      {/* ═══ 左侧: 范数增长指示 ═══ */}
      <group position={[-5.5, 0, 0]}>
        <Text position={[0, 2.5, 0]} fontSize={0.18} color="#94a3b8" anchorX="center" anchorY="middle">
          {t('componentDetail.normGrowth')}
        </Text>
        {/* 范数条 */}
        <mesh position={[0, 0, 0]}>
          <boxGeometry args={[0.3, 4, 0.3]} />
          <meshBasicMaterial color="#1e293b" transparent opacity={0.4} />
        </mesh>
        <mesh position={[0, -2 + normBarH / 2, 0.1]}>
          <boxGeometry args={[0.35, normBarH, 0.35]} />
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={0.25}
            transparent
            opacity={0.75}
          />
        </mesh>
        {/* 范数值 */}
        <Text position={[0.5, 0, 0.2]} fontSize={0.15} color={color} anchorX="left" anchorY="middle">
          {`‖r‖=${data.residual.norm}`}
        </Text>
        {/* 刻度 */}
        {['3', '5', '7', '10'].map((v, i) => {
          const y = -2 + (parseFloat(v) / 10) * 4;
          return (
            <Text key={v} position={[-0.3, y, 0.1]} fontSize={0.1} color="#475569" anchorX="right" anchorY="middle">
              {v}
            </Text>
          );
        })}
      </group>

      {/* ═══ 底部统计 ═══ */}
      <group position={[0, -3.5, 0]}>
        <Text position={[-2, 0.1, 0.1]} fontSize={0.16} color={color} anchorX="left" anchorY="middle"
          outlineWidth={0.02} outlineColor="#0a1022">
          {`${t('componentDetail.skipWeight')}: ${data.residual.skipWeight}`}
        </Text>
        <Text position={[1, 0.1, 0.1]} fontSize={0.16} color="#94a3b8" anchorX="left" anchorY="middle"
          outlineWidth={0.02} outlineColor="#0a1022">
          {`${t('componentDetail.ret')}: ${(data.residual.retention * 100).toFixed(1)}%`}
        </Text>
        <Text position={[-2, -0.2, 0.1]} fontSize={0.14} color="#64748b" anchorX="left" anchorY="middle"
          outlineWidth={0.02} outlineColor="#0a1022">
          {`${t('componentDetail.stream')}: ${data.residual.streamDir}`}
        </Text>
        <Text position={[1, -0.2, 0.1]} fontSize={0.14} color="#64748b" anchorX="left" anchorY="middle"
          outlineWidth={0.02} outlineColor="#0a1022">
          {`‖r‖: ${data.residual.norm}`}
        </Text>
      </group>

      {/* ═══ 运行时参数面板 ═══ */}
      <LayerRuntimeParamPanel
        component="residual"
        data={data}
        color={color}
        position={[-4, -5, 0]}
        t={t}
      />
    </group>
  );
}

// ═══════════════════════════════════════════════════════
// 连接光束 + 脉冲光环
// ═══════════════════════════════════════════════════════

function ConnectorBeam({ color }) {
  const ref = useRef();
  useFrame((state) => {
    if (!ref.current) return;
    ref.current.material.opacity = 0.2 + 0.1 * Math.sin(state.clock.elapsedTime * 3);
  });
  return (
    <mesh ref={ref} position={[-3, 0, 0]}>
      <boxGeometry args={[6, 0.1, 0.1]} />
      <meshBasicMaterial color={color} transparent opacity={0.25} />
    </mesh>
  );
}

// ── 顶部浮动组件类型标识 ──
function ComponentTypeIndicator({ component, color, y, phaseId, t }) {
  const ref = useRef(null);
  useFrame((state) => {
    if (!ref.current) return;
    ref.current.position.y = y + 0.15 * Math.sin(state.clock.elapsedTime * 2);
  });

  const labels = {
    ln: t('componentDetail.layerNorm'),
    attention: t('componentDetail.attention'),
    ffn: t('componentDetail.ffn'),
    residual: t('componentDetail.residual') + ' ⊕',
  };
  const subLabelKeys = {
    ln1: 'componentDetail.preAttention', ln2: 'componentDetail.preFFN',
    qkv: 'componentDetail.qkvProjection', attn_score: 'componentDetail.attnScoring',
    softmax: 'componentDetail.softmaxNorm', attn_out: 'componentDetail.attnOutput',
    ffn_up: 'componentDetail.wUpProjection', ffn_act: 'componentDetail.siluActivation', ffn_down: 'componentDetail.wDownProjection',
    residual1: 'componentDetail.skipConnection1', residual2: 'componentDetail.skipConnection2',
  };

  return (
    <group ref={ref} position={[4, y, 0]}>
      {/* 发光背景块 */}
      <mesh position={[0, 0, -0.05]}>
        <boxGeometry args={[7.5, 1.2, 0.15]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.3} transparent opacity={0.18} />
      </mesh>
      {/* 组件名 */}
      <Text position={[0, 0.15, 0.25]} fontSize={0.55} color={color} anchorX="center" anchorY="middle"
        outlineWidth={0.03} outlineColor="#0a1022">
        {labels[component] || component}
      </Text>
      {/* 子阶段名 */}
      <Text position={[0, -0.32, 0.25]} fontSize={0.28} color="#94a3b8" anchorX="center" anchorY="middle">
        {subLabelKeys[phaseId] ? t(subLabelKeys[phaseId]) : ''}
      </Text>
      {/* 两侧指示点 */}
      <mesh position={[-3.5, 0, 0.1]}>
        <sphereGeometry args={[0.15, 12, 12]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.2} />
      </mesh>
      <mesh position={[3.5, 0, 0.1]}>
        <sphereGeometry args={[0.15, 12, 12]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.2} />
      </mesh>
    </group>
  );
}

// ═══════════════════════════════════════════════════════
// 主组件
// ═══════════════════════════════════════════════════════

export default function ComponentDetailPanel3D({
  layerIdx = 0,
  modelKey = null,
  layerData = null,
  isActive = false,
  animProgress = 0,
  position = [32, 0, 0],
  lang = 'en',
}) {
  const nLayers = MODEL_CONFIGS[modelKey]?.layers || 28;
  const t = useMemo(() => makeT(lang), [lang]);
  const data = useMemo(() => getLayerComponentData(layerIdx, nLayers, t), [layerIdx, nLayers, t]);
  const phaseBoundaries = useMemo(() => getPhaseBoundaries(), []);

  const currentPhase = useMemo(() => {
    if (!isActive || animProgress == null) return null;
    for (const pb of phaseBoundaries) {
      if (animProgress >= pb.start && animProgress < pb.end) return pb;
    }
    return phaseBoundaries[phaseBoundaries.length - 1];
  }, [isActive, animProgress, phaseBoundaries]);

  const currentComponent = currentPhase?.component || null;
  const currentColor = currentPhase?.color || '#475569';
  const currentPhaseId = currentPhase?.id || null;

  // 信息区高度
  const rowCounts = { ln: 6, attention: 8, ffn: 6, residual: 4 };
  const rowCount = rowCounts[currentComponent] || 0;
  const infoH = Math.max(4, rowCount * 0.8 + 2.5);

  // 3D模型区高度
  const modelH = 10;

  // 总面板高度
  const totalH = infoH + modelH + 2;

  return (
    <group position={position}>
      <ConnectorBeam color={currentColor} />

      {/* 无动画占位 */}
      {!currentComponent && (
        <group>
          <Text position={[4, 0.3, 0.25]} fontSize={0.4} color="#475569" anchorX="center" anchorY="middle">
            {t('componentDetail.componentDetail')}
          </Text>
          <Text position={[4, -0.3, 0.25]} fontSize={0.28} color="#334155" anchorX="center" anchorY="middle">
            {t('componentDetail.runAnimation')}
          </Text>
        </group>
      )}

      {/* 有动画阶段 */}
      {currentComponent && (
        <group>
          {/* ── 顶部浮动组件类型标识 ── */}
          <ComponentTypeIndicator component={currentComponent} color={currentColor} y={totalH + 1.5} phaseId={currentPhaseId} t={t} />

          {/* ── 上方: 信息区 ── */}
          <group position={[0, totalH - 1.5, 0]}>
            {/* 层号标题 */}
            <Text position={[0.4, infoH + 0.2, 0.25]} fontSize={0.55} color={data.layerColor} anchorX="left" anchorY="middle">
              L{layerIdx}
            </Text>
            <Text position={[2.8, infoH + 0.2, 0.25]} fontSize={0.38} color="#94a3b8" anchorX="left" anchorY="middle">
              {data.layerLabel}
            </Text>

            {/* 分隔线: 标题与参数 */}
            <mesh position={[4, infoH - 0.4, 0.05]}>
              <boxGeometry args={[7.5, 0.04, 0.04]} />
              <meshBasicMaterial color={currentColor} transparent opacity={0.5} />
            </mesh>

            {/* 参数内容 */}
            {currentComponent === 'ln' && <LNInfo data={data} color={currentColor} phase={currentPhaseId} t={t} />}
            {currentComponent === 'attention' && <AttentionInfo data={data} color={currentColor} phase={currentPhaseId} t={t} />}
            {currentComponent === 'ffn' && <FFNInfo data={data} color={currentColor} phase={currentPhaseId} t={t} />}
            {currentComponent === 'residual' && <ResidualInfo data={data} color={currentColor} phase={currentPhaseId} t={t} />}
          </group>

          {/* ── 分隔线: 信息区与3D模型区 ── */}
          <mesh position={[4, modelH / 2 + 0.3, 0.05]}>
            <boxGeometry args={[7.5, 0.06, 0.06]} />
            <meshBasicMaterial color={currentColor} transparent opacity={0.6} />
          </mesh>
          <Text position={[0.4, modelH / 2 + 0.5, 0.25]} fontSize={0.22} color={currentColor} anchorX="left" anchorY="middle">
            {t('componentDetail.model3D')}
          </Text>

          {/* ── 下方: 3D模型区 ── */}
          <group position={[4, 0, 1.5]}>
            {currentComponent === 'ln' && <LNModel3D data={data} color={currentColor} animProgress={animProgress} t={t} />}
            {currentComponent === 'attention' && <AttentionModel3D data={data} color={currentColor} phase={currentPhaseId} t={t} />}
            {currentComponent === 'ffn' && <FFNModel3D data={data} color={currentColor} phase={currentPhaseId} t={t} />}
            {currentComponent === 'residual' && <ResidualModel3D data={data} color={currentColor} t={t} />}
          </group>
        </group>
      )}
    </group>
  );
}
