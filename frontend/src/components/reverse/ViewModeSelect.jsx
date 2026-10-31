/**
 * 3D视角模式选择器
 * 5种叠加效果: 结构/正交/频谱/因果/编码
 */
import { Boxes, Cpu, GitBranch, Radio, Waves } from 'lucide-react';

const VIEW_MODES = [
  { id: 'structure', label: '结构视图', icon: Boxes, color: '#38bdf8', desc: '默认层级骨架视图' },
  { id: 'orthogonal', label: '正交视图', icon: Waves, color: '#4facfe', desc: '半透明子空间平面+差分向量' },
  { id: 'spectral', label: '频谱视图', icon: Radio, color: '#22c55e', desc: '5频段颜色光晕叠加' },
  { id: 'causal', label: '因果视图', icon: GitBranch, color: '#ff6b6b', desc: '因果流线粒子+1D流形' },
  { id: 'encoding', label: '编码视图', icon: Cpu, color: '#ffd93d', desc: '编码方程标签+R²指示器' },
];

export default function ViewModeSelect({ viewMode, onViewModeChange }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      <span style={{ fontSize: '12px', fontWeight: 700, color: '#dfe8ff' }}>
        3D视角模式
      </span>
      <div style={{ display: 'flex', gap: '3px', flexWrap: 'wrap' }}>
        {VIEW_MODES.map((mode) => {
          const IconComp = mode.icon;
          const isActive = viewMode === mode.id;
          return (
            <button
              key={mode.id}
              onClick={() => onViewModeChange(mode.id)}
              title={mode.desc}
              style={{
                display: 'flex', alignItems: 'center', gap: '3px',
                padding: '4px 8px',
                borderRadius: '4px',
                border: `1px solid ${isActive ? mode.color + '60' : 'rgba(255,255,255,0.1)'}`,
                background: isActive ? mode.color + '20' : 'transparent',
                color: isActive ? mode.color : '#888',
                cursor: 'pointer',
                fontSize: '10px',
                fontWeight: isActive ? 700 : 500,
                transition: 'all 0.15s',
              }}
            >
              {IconComp && <IconComp size={11} />}
              {mode.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
