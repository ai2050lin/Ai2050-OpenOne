import axios from 'axios';
import { Activity, Play, RotateCcw } from 'lucide-react';
import { useEffect, useState } from 'react';
import { SimplePanel } from './SimplePanel';

const API_BASE = 'http://localhost:5002';

/**
 * Helper to get color for entropy heatmap
 * Low Entropy (Blue) -> Medium (Green) -> High (Red)
 */
const getEntropyColor = (value) => {
  // Typical entropy range 0-10
  // Normalize roughly 0-5 for visual spread
  const norm = Math.min(value / 6, 1.0);
  
  // HSL: Blue (240) -> Green (120) -> Red (0)
  // We want low=blue, high=red.
  // 240 * (1 - norm) would go blue->red
  const hue = 240 * (1 - norm);
  return `hsl(${hue}, 80%, 50%)`;
};

// --- Subcomponents ---

function MetricCard({ title, value, unit, description, color = '#4488ff' }) {
  return (
    <div style={{
      background: 'rgba(255,255,255,0.05)', borderRadius: '8px', padding: '12px',
      borderLeft: `4px solid ${color}`, flex: 1, minWidth: '120px'
    }}>
      <div style={{ fontSize: '12px', color: '#aaa', marginBottom: '4px' }}>{title}</div>
      <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#fff' }}>
        {typeof value === 'number' ? value.toFixed(3) : value}
        {unit && <span style={{ fontSize: '12px', color: '#888', marginLeft: '4px' }}>{unit}</span>}
      </div>
      {description && <div style={{ fontSize: '10px', color: '#666', marginTop: '4px' }}>{description}</div>}
    </div>
  );
}


function EntropyHeatmap({ text, entropyStats, t }) {
  // Note: We don't have per-token entropy from backend yet in the simplest version,
  // but if we did, we would map it here.
  // For now, let's visualize the aggregate stats or assume we update backend to return per-token.
  // The backend `compute_entropy_profile` returns stats, but let's assume valid text is passed.
  
  // Placeholder visualization since we only have aggregation stats in current backend impl
  // We will visualize the stats distribution roughly.
  
  if (!entropyStats) return null;

  return (
    <div style={{ marginTop: '16px', background: 'rgba(0,0,0,0.2)', padding: '12px', borderRadius: '8px' }}>
        <h4 style={{ margin: '0 0 8px 0', fontSize: '14px', color: '#ddd' }}>{t('validity.entropyStats')}</h4>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <div style={{ fontSize: '12px', color: '#aaa' }}>{t('validity.min')}: {entropyStats.min_entropy?.toFixed(2)}</div>
            <div style={{ flex: 1, height: '6px', background: '#333', borderRadius: '3px', position: 'relative' }}>
                <div style={{ 
                    position: 'absolute', 
                    left: '20%', right: '20%', 
                    top: 0, bottom: 0, 
                    background: 'linear-gradient(90deg, #4488ff, #ff4444)',
                    borderRadius: '3px',
                    opacity: 0.5
                }} />
                {/* Mean Marker */}
                <div style={{
                    position: 'absolute',
                    left: '50%', // Placeholder position
                    top: '-4px', bottom: '-4px', width: '2px', background: '#fff'
                }} />
            </div>
            <div style={{ fontSize: '12px', color: '#aaa' }}>{t('validity.max')}: {entropyStats.max_entropy?.toFixed(2)}</div>
        </div>
        <div style={{ textAlign: 'center', fontSize: '11px', color: '#666', marginTop: '4px' }}>
            {t('validity.mean')}: {entropyStats.mean_entropy?.toFixed(2)} | {t('validity.variance')}: {entropyStats.variance_entropy?.toFixed(2)}
        </div>
    </div>
  );
}

function AnisotropyChart({ geometricStats, t }) {
    if (!geometricStats) return null;
    
    // Convert object { "layer_0_anisotropy": 0.1 } to array
    const data = Object.entries(geometricStats)
        .map(([key, val]) => {
            const layer = parseInt(key.split('_')[1]);
            return { layer, value: val };
        })
        .sort((a, b) => a.layer - b.layer);
        
    const maxVal = Math.max(...data.map(d => d.value), 0.1); // Avoid div by zero
    
    return (
        <div style={{ marginTop: '16px' }}>
            <h4 style={{ margin: '0 0 12px 0', fontSize: '14px', color: '#ddd' }}>{t('validity.anisotropy')}</h4>
            <div style={{ display: 'flex', alignItems: 'flex-end', height: '100px', gap: '4px', paddingBottom: '20px', borderBottom: '1px solid #333' }}>
                {data.map((d) => (
                    <div key={d.layer} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
                        <div style={{ 
                            width: '80%', 
                            height: `${(d.value / maxVal) * 100}%`, 
                            background: d.value > 0.9 ? '#ff4444' : '#4488ff', // Red if collapsed
                            borderRadius: '2px 2px 0 0',
                            transition: 'height 0.3s'
                        }} title={`${t('validity.layer', { layer: d.layer })}: ${d.value.toFixed(3)}`} />
                        <div style={{ fontSize: '10px', color: '#666', transform: 'rotate(-45deg)', transformOrigin: 'top left', marginTop: '4px' }}>
                            {t('validity.l')}{d.layer}
                        </div>
                    </div>
                ))}
            </div>
             <div style={{ fontSize: '10px', color: '#888', marginTop: '8px', textAlign: 'center' }}>
                {t('validity.collapseWarning')}
            </div>
        </div>
    );
}


export default function LanguageValidityPanel({ prompt, onClose, t }) {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  const analyze = async () => {
    if (!prompt) return;
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/analyze_validity`, { 
        prompt,
        // Optional: request all layers for full chart
        // target_layers: [0, 1, 2, ..., n] - backend handles null as default
        // Let's rely on backend default or improve later
      });
      setResults(res.data);
    } catch (err) {
      console.error(err);
      alert('分析失败，请检查后端服务。');
    } finally {
      setLoading(false);
    }
  };

  // Auto-analyze on mount
  useEffect(() => {
    analyze();
  }, [prompt]);

  return (
    <SimplePanel
      title={t('validity.title')}
      icon={<Activity />}
      onClose={onClose}
      style={{
        position: 'absolute', top: 20, right: 20, zIndex: 90,
        width: '360px', maxHeight: '90vh'
      }}
    >

      <div style={{ fontSize: '12px', color: '#888', marginBottom: '16px', borderLeft: '2px solid #555', paddingLeft: '8px' }}>
        {t('validity.evaluating')} <br/>
        <span style={{ color: '#ccc', fontStyle: 'italic' }}>
            "{prompt.length > 50 ? prompt.slice(0, 50) + '...' : prompt}"
        </span>
      </div>

      <button 
        onClick={analyze} 
        disabled={loading}
        style={{
          width: '100%', padding: '8px', background: '#333', color: '#fff',
          border: '1px solid #555', borderRadius: '4px', cursor: 'pointer',
          marginBottom: '20px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px'
        }}
      >
        {loading ? <RotateCcw className="spin" size={14} /> : <Play size={14} />}
        {loading ? t('validity.analyzing') : (results ? t('validity.reanalyze') : t('validity.analyze'))}
      </button>

      {results && (
        <div className="fade-in">
          {/* Main Metrics */}
          <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
            <MetricCard 
                title={t('validity.perplexity')} 
                value={results.perplexity} 
                description={t('validity.pplDesc')}
            />
            <MetricCard 
                title={t('validity.entropy')}
                value={results.entropy_stats?.mean_entropy} 
                color="#ffaa00"
                description={t('validity.entropyDesc')}
            />
          </div>

          <EntropyHeatmap entropyStats={results.entropy_stats} text={prompt} t={t} />
          
          <AnisotropyChart geometricStats={results.geometric_stats} t={t} />
          
        </div>
      )}
      
      {!results && !loading && (
        <div style={{ textAlign: 'center', color: '#666', padding: '20px' }}>
          {t('validity.clickToAnalyze')}
        </div>
      )}
    </SimplePanel>
  );
}
