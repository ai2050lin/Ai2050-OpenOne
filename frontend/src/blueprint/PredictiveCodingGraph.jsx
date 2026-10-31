import { useMemo } from 'react';

const LAYERS = [
  { id: 'sensory', label: 'Sensory', x: 60, y: 190, color: '#60a5fa' },
  { id: 'syntax', label: 'Syntax', x: 180, y: 145, color: '#34d399' },
  { id: 'logic', label: 'Logic', x: 300, y: 145, color: '#f59e0b' },
  { id: 'style', label: 'Style', x: 420, y: 145, color: '#f472b6' },
  { id: 'semantic', label: 'Semantic', x: 540, y: 190, color: '#a78bfa' },
];

function arrowPath(x1, y1, x2, y2, bend = 0) {
  const cx = (x1 + x2) / 2 + bend;
  const cy = (y1 + y2) / 2 - Math.abs(bend) * 0.2;
  return `M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`;
}

export default function PredictiveCodingGraph() {
  const connections = useMemo(() => ([
    { from: 0, to: 1, kind: 'feedforward' },
    { from: 1, to: 2, kind: 'feedforward' },
    { from: 2, to: 3, kind: 'feedforward' },
    { from: 3, to: 4, kind: 'feedforward' },
    { from: 4, to: 3, kind: 'feedback' },
    { from: 3, to: 2, kind: 'feedback' },
    { from: 2, to: 1, kind: 'feedback' },
    { from: 1, to: 0, kind: 'feedback' },
  ]), []);

  return (
    <div
      style={{
        borderRadius: 10,
        border: '1px solid rgba(255,255,255,0.1)',
        background: 'linear-gradient(170deg, rgba(14,20,37,0.95), rgba(8,12,24,0.95))',
        padding: 10,
      }}
    >
      <div style={{ fontSize: 12, color: '#dbe9ff', marginBottom: 8, fontWeight: 600 }}>
        Predictive Coding: feedforward + feedback error correction
      </div>
      <svg width="600" height="250" viewBox="0 0 600 250">
        <defs>
          <marker id="arrowFeed" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <polygon points="0 0, 8 4, 0 8" fill="#7dd3fc" />
          </marker>
          <marker id="arrowBack" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <polygon points="0 0, 8 4, 0 8" fill="#fda4af" />
          </marker>
        </defs>

        {connections.map((c, i) => {
          const from = LAYERS[c.from];
          const to = LAYERS[c.to];
          const bend = c.kind === 'feedforward' ? -10 : 10;
          return (
            <path
              key={`edge-${i}`}
              d={arrowPath(from.x, from.y, to.x, to.y, bend)}
              stroke={c.kind === 'feedforward' ? '#7dd3fc' : '#fda4af'}
              strokeWidth="2"
              fill="none"
              markerEnd={c.kind === 'feedforward' ? 'url(#arrowFeed)' : 'url(#arrowBack)'}
              opacity="0.9"
            />
          );
        })}

        {LAYERS.map((node) => (
          <g key={node.id}>
            <circle cx={node.x} cy={node.y} r="24" fill={node.color} fillOpacity="0.2" stroke={node.color} strokeWidth="2" />
            <text x={node.x} y={node.y + 4} fill="#dbe9ff" textAnchor="middle" fontSize="11" fontWeight="600">
              {node.label}
            </text>
          </g>
        ))}

        <text x="300" y="36" fill="#9fb2d5" textAnchor="middle" fontSize="12">
          上下文误差信号在层间往返，联合调节风格/逻辑/句法生成
        </text>
      </svg>
    </div>
  );
}
