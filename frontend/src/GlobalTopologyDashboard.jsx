
import { Activity, Brain, ShieldCheck, Zap } from 'lucide-react';

export function GlobalTopologyDashboard({ results }) {
  if (!results || !results.results) return (
    <div style={{ color: '#666', fontStyle: 'italic', textAlign: 'center', padding: '40px' }}>
      <Activity size={48} style={{ opacity: 0.2, marginBottom: '16px' }} />
      <p>等待扫描数据分析结果...</p>
    </div>
  );

  const { results: scanData } = results;
  const fields = Object.keys(scanData);

  return (
    <div style={{ padding: '20px', color: '#fff' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '24px', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '12px' }}>
        <Brain color="#4488ff" size={24} />
        <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 'bold' }}>AGI 全局拓扑动力描绘图 (Global Topology Portrait)</h2>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
        {fields.map(field => (
          <div key={field} style={{ background: 'rgba(255,255,255,0.02)', borderRadius: '12px', padding: '20px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <h3 style={{ margin: '0 0 16px 0', fontSize: '14px', color: '#4ecdc4', textTransform: 'uppercase', letterSpacing: '1px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Zap size={14} /> {field.replace('_', ' ')}
            </h3>
            
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
              {Object.entries(scanData[field]).map(([layer, data]) => (
                <div key={layer} style={{ background: 'rgba(0,0,0,0.3)', borderRadius: '8px', padding: '16px', borderLeft: `4px solid ${(data?.ortho_error ?? 0) < 0.1 ? '#44ff88' : '#ffaa44'}` }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                    <span style={{ fontWeight: '600', color: '#fff' }}>{layer}</span>
                    <span style={{ fontSize: '12px', color: (data?.ortho_error ?? 0) < 0.1 ? '#44ff88' : '#ffaa44' }}>
                      {(data?.ortho_error ?? 0) < 0.1 ? 'Logic Pillar ✅' : 'High Curvature ⚠️'}
                    </span>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div style={{ fontSize: '11px', color: '#888' }}>
                      Orthogonality Error: <span style={{ color: '#fff', float: 'right' }}>{(data?.ortho_error ?? 0).toFixed(6)}</span>
                    </div>
                    <div style={{ height: '4px', background: '#222', borderRadius: '2px' }}>
                        <div style={{ width: `${Math.max(0, 100 - (data?.ortho_error ?? 0) * 100)}%`, height: '100%', background: '#4488ff', borderRadius: '2px' }} />
                    </div>
                    
                    <div style={{ fontSize: '11px', color: '#888', marginTop: '4px' }}>
                      Determinant (Det): <span style={{ color: '#fff', float: 'right' }}>{(data?.det ?? 0).toFixed(4)}</span>
                    </div>
                    <div style={{ fontSize: '11px', color: '#888' }}>
                      Scale Property: <span style={{ color: '#fff', float: 'right' }}>{(data?.scale ?? 0).toFixed(4)}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: '30px', padding: '16px', background: 'rgba(68,136,255,0.05)', borderRadius: '8px', border: '1px solid rgba(68,136,255,0.1)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px', color: '#4488ff' }}>
          <ShieldCheck size={16} />
          <span style={{ fontSize: '12px', fontWeight: 'bold' }}>拓扑稳定性分析 (Topology Stability Analysis)</span>
        </div>
        <p style={{ margin: 0, fontSize: '11px', color: '#888', lineHeight: '1.6' }}>
          <b>逻辑支柱 (Logic Pillars):</b> 正交误差低于 0.01 的层级，是模型泛化能力的几何根基。在此区域注入 RPT 算子最安全。<br/>
          <b>高曲率区 (High Curvature):</b> 行列式偏离 1.0 显著的区域，语义纤维发生严重扭曲，通常对应逻辑闭环的转换或偏见坍缩区。
        </p>
      </div>
    </div>
  );
}
