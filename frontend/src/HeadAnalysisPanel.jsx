
import axios from 'axios';
import { ArrowRight, Grid } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

const API_BASE = 'http://localhost:5002';

export function HeadAnalysisPanel({ layerIdx, headIdx, prompt, onClose, t }) {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [activeTab, setActiveTab] = useState('pattern'); // 'pattern' or 'qkv'

  useEffect(() => {
    if (layerIdx === null || headIdx === null || !prompt) return;

    const fetchData = async () => {
      setLoading(true);
      try {
        const res = await axios.post(`${API_BASE}/head_details`, {
          prompt,
          layer_idx: layerIdx,
          head_idx: headIdx
        });
        setData(res.data);
      } catch (err) {
        console.error("Failed to fetch head details:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [layerIdx, headIdx, prompt]);

  if (!data && loading) {
    return (
      <div style={{ padding: '20px', color: '#aaa', textAlign: 'center' }}>
        Loading head analysis...
      </div>
    );
  }

  if (!data) return null;

  return (
    <div style={{ padding: '0', color: '#eee', height: '100%', display: 'flex', flexDirection: 'column' }}>
      
      {/* Tabs */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '15px', borderBottom: '1px solid #333' }}>
        <TabButton 
          active={activeTab === 'pattern'} 
          onClick={() => setActiveTab('pattern')} 
          icon={<Grid size={14} />}
          label={t('head.pattern')} 
        />
        <TabButton 
          active={activeTab === 'qkv'} 
          onClick={() => setActiveTab('qkv')} 
          icon={<ArrowRight size={14} />}
          label={t('head.qkv')} 
        />
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>
        {activeTab === 'pattern' && (
          <AttentionPatternView 
            pattern={data.pattern} 
            tokens={data.tokens} 
            t={t}
          />
        )}
        {activeTab === 'qkv' && (
          <QKVView 
            tokens={data.tokens}
            q={data.q}
            k={data.k}
            v={data.v}
            z={data.z}
            t={t}
          />
        )}
      </div>
    </div>
  );
}

function TabButton({ active, onClick, icon, label }) {
  return (
    <button
      onClick={onClick}
      style={{
        background: 'none',
        border: 'none',
        borderBottom: active ? '2px solid #4488ff' : '2px solid transparent',
        color: active ? 'white' : '#888',
        padding: '8px 12px',
        cursor: 'pointer',
        fontSize: '12px',
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        fontWeight: active ? 'bold' : 'normal'
      }}
    >
      {icon}
      {label}
    </button>
  );
}

function AttentionPatternView({ pattern, tokens, t }) {
  // pattern is [n_heads (1), seq_q, seq_k] -> we just have [seq_q, seq_k]
  // Normalize purely for visualization if needed, but attention pattern is already normalized (sum to 1)
  
  return (
    <div>
      <div style={{ fontSize: '12px', color: '#aaa', marginBottom: '10px' }}>
        {t('head.patternDesc')}
      </div>
      <HeatmapMatrix 
        matrix={pattern} 
        xLabels={tokens} 
        yLabels={tokens} 
        valueRange={[0, 1]}
      />
    </div>
  );
}

function QKVView({ tokens, q, k, v, z, t }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <MatrixSection title={t('head.q')} matrix={q} yLabels={tokens} />
      <MatrixSection title={t('head.k')} matrix={k} yLabels={tokens} />
      <MatrixSection title={t('head.v')} matrix={v} yLabels={tokens} />
      <MatrixSection title={t('head.out')} matrix={z} yLabels={tokens} />
    </div>
  );
}

function MatrixSection({ title, matrix, yLabels }) {
  // Matrix is [seq, d_head]
  // We can visualize this. Since d_head is large (e.g. 64), we might simply show it without x-labels, or just simplified
    
  return (
    <div>
      <div style={{ fontSize: '12px', fontWeight: 'bold', color: '#ddd', marginBottom: '5px' }}>{title}</div>
      <HeatmapMatrix 
        matrix={matrix} 
        yLabels={yLabels} 
        xLabel="d_head"
        valueRange={[-1, 1]} // Simple normalization guess
        cellWidth={4}
        cellHeight={16}
      />
    </div>
  );
}

function HeatmapMatrix({ matrix, xLabels, yLabels, xLabel, valueRange, cellWidth = 20, cellHeight = 20 }) {
  const canvasRef = useRef(null);
  const rows = matrix.length;
  const cols = matrix[0].length;
  
  const [hovered, setHovered] = useState(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    // Clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const val = matrix[i][j];
        // Normalize
        let norm = (val - valueRange[0]) / (valueRange[1] - valueRange[0]);
        norm = Math.max(0, Math.min(1, norm));
        
        // Color map: Zero=Black, High=Yellow/Red
        // Simple: Blue (negative) -> Black (zero) -> Red (positive) if range is [-1, 1]
        // Or for Attention (0-1): Black -> White
        
        let color;
        if (valueRange[0] === 0 && valueRange[1] === 1) {
          // 0-1 (Attention)
          const intensity = Math.floor(norm * 255);
          color = `rgb(${intensity}, ${intensity}, ${intensity})`;
          // Highlight high values
          if (norm > 0.5) color = `rgb(255, ${Math.floor((1-norm)*2*255)}, 0)`; 
        } else {
          // -1 to 1 (Activations)
          // -1: Blue, 0: Black, 1: Red
          if (val < 0) {
            const intensity = Math.floor(Math.abs(val) * 255); // simplified assuming approx range
             color = `rgb(0, 0, ${Math.min(255, intensity)})`;
          } else {
            const intensity = Math.floor(Math.abs(val) * 255);
            color = `rgb(${Math.min(255, intensity)}, 0, 0)`;
          }
        }

        ctx.fillStyle = color;
        ctx.fillRect(j * cellWidth, i * cellHeight, cellWidth, cellHeight);
      }
    }
  }, [matrix, valueRange, cellWidth, cellHeight]);

  const handleMouseMove = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const col = Math.floor(x / cellWidth);
    const row = Math.floor(y / cellHeight);
    
    if (col >= 0 && col < cols && row >= 0 && row < rows) {
      setHovered({ row, col, val: matrix[row][col] });
    } else {
      setHovered(null);
    }
  };

  return (
    <div style={{ display: 'flex', gap: '10px' }}>
      {/* Y Labels */}
      {yLabels && (
        <div style={{ display: 'flex', flexDirection: 'column', paddingTop: '0px' }}>
          {yLabels.map((label, i) => (
            <div key={i} style={{ height: cellHeight, fontSize: '10px', lineHeight: `${cellHeight}px`, textAlign: 'right', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', width: '60px', color: hovered?.row === i ? 'white' : '#888' }}>
              {label}
            </div>
          ))}
        </div>
      )}

      {/* Canvas */}
      <div style={{ position: 'relative' }}>
        <canvas 
          ref={canvasRef}
          width={cols * cellWidth}
          height={rows * cellHeight}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHovered(null)}
          style={{ imageRendering: 'pixelated', border: '1px solid #333' }}
        />
        
        {/* Hover Tooltip */}
        {hovered && (
          <div style={{
            position: 'absolute',
            left: hovered.col * cellWidth + cellWidth + 5,
            top: hovered.row * cellHeight,
            background: 'rgba(0,0,0,0.8)',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '11px',
            pointerEvents: 'none',
            zIndex: 10,
            whiteSpace: 'nowrap'
          }}>
            Val: {hovered.val.toFixed(4)}
            {xLabels && <div>→ {xLabels[hovered.col]}</div>}
          </div>
        )}
      </div>

      {/* X Labels (if simple list, mostly for Attention) */}
      {/* For activations, X is just d_head dimension */}
    </div>
  );
}
