import { Activity, Brain, CheckCircle, Search, X } from 'lucide-react';

const BrainModel = () => (
  <div
    style={{
      width: '320px',
      height: '320px',
      margin: '0 auto',
      position: 'relative',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'radial-gradient(circle, rgba(0, 210, 255, 0.15) 0%, transparent 70%)',
      borderRadius: '50%',
      animation: 'brainPulse 4s infinite alternate',
    }}
  >
    <div
      style={{
        position: 'absolute',
        width: '100%',
        height: '100%',
        border: '1px solid rgba(0, 210, 255, 0.2)',
        borderRadius: '50%',
        animation: 'brainRotate 20s linear infinite',
      }}
    />
    <div
      style={{
        position: 'absolute',
        width: '85%',
        height: '85%',
        border: '1px dashed rgba(168, 85, 247, 0.3)',
        borderRadius: '50%',
        animation: 'brainRotateReverse 15s linear infinite',
      }}
    />
    <Brain
      size={180}
      color="#00d2ff"
      style={{
        filter: 'drop-shadow(0 0 30px rgba(0, 210, 255, 0.4))',
        zIndex: 2,
      }}
    />
    {Array.from({ length: 8 }).map((_, i) => (
      <div
        key={i}
        style={{
          position: 'absolute',
          width: '6px',
          height: '6px',
          background: '#00ff88',
          borderRadius: '50%',
          boxShadow: '0 0 10px #00ff88',
          transform: `rotate(${i * 45}deg) translateY(-130px)`,
          animation: 'synapsePulse 2s infinite',
          animationDelay: `${i * 0.2}s`,
        }}
      />
    ))}
  </div>
);

export const SystemStatusTab = ({
  consciousField,
  systemRouteOptions,
  routeList,
  setSelectedRouteId,
  selectedRouteId,
  activeSystemProfile,
  statusData,
  selectedRoute,
  getRouteImpl,
  expandedParam,
  setExpandedParam,
}) => (
  <div style={{ animation: 'roadmapFade 0.5s ease-out' }}>
    <BrainModel />
    <div style={{ textAlign: 'center', marginBottom: '60px' }}>
      <h2
        style={{
          fontSize: '32px',
          fontWeight: '900',
          color: consciousField?.glow_color === 'amber' ? '#ffaa00' : '#10b981',
          margin: '20px 0 8px 0',
          transition: 'color 1s',
        }}
      >
        {consciousField ? '实时意识场 (Active Consciousness)' : '系统状态 (System Status)'}
      </h2>
      <p style={{ color: '#666', fontSize: '14px' }}>
        {consciousField
          ? `当前内稳态平衡: ${((consciousField.stability || 0) * 100).toFixed(1)}% | GWS 竞争强度: ${(consciousField.gws_intensity || 0).toFixed(2)}`
          : '基于 Project Genesis 协议的核心能力对齐报告'}
      </p>
      <div style={{ marginTop: '14px', display: 'flex', justifyContent: 'center', gap: '8px', flexWrap: 'wrap' }}>
        {(systemRouteOptions.length > 0 ? systemRouteOptions : routeList).map((route) => (
          <button
            key={route.id}
            onClick={() => setSelectedRouteId(route.id)}
            style={{
              border: '1px solid rgba(255,255,255,0.12)',
              background: selectedRouteId === route.id ? 'rgba(0, 210, 255, 0.2)' : 'rgba(255,255,255,0.03)',
              color: selectedRouteId === route.id ? '#67e8f9' : '#94a3b8',
              borderRadius: '999px',
              fontSize: '11px',
              padding: '6px 10px',
              cursor: 'pointer',
            }}
          >
            {route.title}
          </button>
        ))}
      </div>
    </div>

    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px', marginBottom: '40px', animation: 'fadeIn 1s' }}>
      {(activeSystemProfile?.metricCards || []).map((m, i) => (
        <div key={i} style={{ padding: '18px', background: 'rgba(255,255,255,0.02)', borderRadius: '20px', border: `1px solid ${m.color}30`, textAlign: 'center' }}>
          <div style={{ fontSize: '10px', color: '#666', marginBottom: '6px', fontWeight: 'bold' }}>{m.label}</div>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: m.color, marginBottom: '6px' }}>{m.value}</div>
          <div style={{ fontSize: '10px', color: '#9ca3af' }}>{m.brain_ability}</div>
        </div>
      ))}
    </div>

    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '40px' }}>
      <div style={{ background: 'rgba(0, 255, 136, 0.03)', border: '1px solid rgba(0, 255, 136, 0.15)', borderRadius: '32px', padding: '32px' }}>
        <div style={{ fontSize: '12px', color: '#00ff88', textTransform: 'uppercase', letterSpacing: '2px', marginBottom: '24px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <CheckCircle size={16} /> 已具备能力(Equipped)
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          {statusData.capabilities.map((c, i) => (
            <div key={i} style={{ padding: '16px', background: 'rgba(255,255,255,0.02)', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ fontWeight: 'bold', fontSize: '14px', marginBottom: '8px' }}>{c.name}</div>
              <div style={{ display: 'grid', gridTemplateColumns: '0.9fr 1.1fr', gap: '10px' }}>
                <div style={{ padding: '8px 10px', borderRadius: '10px', background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.2)' }}>
                  <div style={{ fontSize: '10px', color: '#86efac', marginBottom: '4px' }}>人脑能力</div>
                  <div style={{ fontSize: '11px', color: '#dcfce7', lineHeight: '1.55' }}>{c.brain_ability || '-'}</div>
                </div>
                <div style={{ padding: '8px 10px', borderRadius: '10px', background: 'rgba(56,189,248,0.08)', border: '1px solid rgba(56,189,248,0.2)' }}>
                  <div style={{ fontSize: '10px', color: '#7dd3fc', marginBottom: '4px' }}>
                    当前实现（{selectedRoute?.title || selectedRouteId}）
                  </div>
                  <div style={{ fontSize: '11px', color: '#e0f2fe', lineHeight: '1.55' }}>{getRouteImpl(c)}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ background: 'rgba(255, 68, 68, 0.03)', border: '1px solid rgba(255, 68, 68, 0.15)', borderRadius: '32px', padding: '32px' }}>
        <div style={{ fontSize: '12px', color: '#ff4444', textTransform: 'uppercase', letterSpacing: '2px', marginBottom: '24px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <X size={16} /> 研发。缺失 (Missing)
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          {statusData?.missing_capabilities?.map((c, i) => (
            <div key={i} style={{ padding: '16px', background: 'rgba(255,255,255,0.02)', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ fontWeight: 'bold', fontSize: '14px', marginBottom: '8px', color: '#ff8888' }}>{c.name}</div>
              <div style={{ display: 'grid', gridTemplateColumns: '0.9fr 1.1fr', gap: '10px' }}>
                <div style={{ padding: '8px 10px', borderRadius: '10px', background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.22)' }}>
                  <div style={{ fontSize: '10px', color: '#fca5a5', marginBottom: '4px' }}>人脑能力</div>
                  <div style={{ fontSize: '11px', color: '#fee2e2', lineHeight: '1.55' }}>{c.brain_ability || '-'}</div>
                </div>
                <div style={{ padding: '8px 10px', borderRadius: '10px', background: 'rgba(251,191,36,0.08)', border: '1px solid rgba(251,191,36,0.22)' }}>
                  <div style={{ fontSize: '10px', color: '#fcd34d', marginBottom: '4px' }}>
                    当前实现（{selectedRoute?.title || selectedRouteId}）
                  </div>
                  <div style={{ fontSize: '11px', color: '#fef3c7', lineHeight: '1.55' }}>{getRouteImpl(c)}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>

    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '40px', marginTop: '40px' }}>
      <div style={{ background: 'rgba(0, 210, 255, 0.03)', border: '1px solid rgba(0, 210, 255, 0.15)', borderRadius: '32px', padding: '32px' }}>
        <div style={{ fontSize: '12px', color: '#00d2ff', textTransform: 'uppercase', letterSpacing: '2px', marginBottom: '24px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Activity size={16} /> 核心参数 (Route Parameters)
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
          {(activeSystemProfile?.parameterCards || []).map((p, i) => (
            <div
              key={i}
              onClick={() => setExpandedParam(expandedParam === i ? null : i)}
              style={{
                padding: '16px',
                background: 'rgba(0,0,0,0.3)',
                borderRadius: '16px',
                border: `1px solid ${expandedParam === i ? 'rgba(0, 210, 255, 0.5)' : 'rgba(0, 210, 255, 0.1)'}`,
                cursor: 'pointer',
                transition: 'all 0.3s',
                gridColumn: expandedParam === i ? 'span 2' : 'span 1',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                <div style={{ fontSize: '10px', color: '#555' }}>{p.name}</div>
                {expandedParam === i ? <div style={{ fontSize: '10px', color: '#00d2ff' }}>SHMC CORE ▲</div> : null}
              </div>
              <div style={{ fontSize: '11px', color: '#7dd3fc', marginBottom: '4px' }}>人脑能力：{p.brain_ability || '-'}</div>
              <div style={{ fontSize: '18px', fontWeight: '900', color: '#fff', fontFamily: 'monospace' }}>{p.route_param}</div>
              <div style={{ fontSize: '10px', color: '#00d2ff88', marginTop: '4px' }}>{p.detail}</div>

              {expandedParam === i && (
                <div style={{ marginTop: '16px', borderTop: '1px solid rgba(0, 210, 255, 0.1)', paddingTop: '16px', animation: 'fadeIn 0.3s ease' }}>
                  <div style={{ marginBottom: '12px' }}>
                    <div style={{ fontSize: '10px', color: '#00d2ff', fontWeight: 'bold', marginBottom: '4px' }}>参数定义 (DEFINITION)</div>
                    <div style={{ fontSize: '12px', color: '#bbb', lineHeight: '1.6' }}>{p.desc}</div>
                  </div>
                  <div style={{ marginBottom: '12px' }}>
                    <div style={{ fontSize: '10px', color: '#a855f7', fontWeight: 'bold', marginBottom: '4px' }}>数值价。(VALUE)</div>
                    <div style={{ fontSize: '12px', color: '#bbb', lineHeight: '1.6' }}>{p.value_meaning}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '10px', color: '#f59e0b', fontWeight: 'bold', marginBottom: '4px' }}>核心重要。(WHY IMPORTANT)</div>
                    <div style={{ fontSize: '12px', color: '#bbb', lineHeight: '1.6' }}>{p.why_important}</div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <div style={{ background: 'rgba(16, 185, 129, 0.03)', border: '1px solid rgba(16, 185, 129, 0.15)', borderRadius: '32px', padding: '32px' }}>
        <div style={{ fontSize: '12px', color: '#10b981', textTransform: 'uppercase', letterSpacing: '2px', marginBottom: '12px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Search size={16} /> 测试记录迁移
        </div>
        <div style={{ fontSize: '13px', color: '#d1fae5', lineHeight: '1.7' }}>
          系统状态中的路线测试记录已整合到“研发进展 → 里程碑 → 路线测试记录”阶段。
        </div>
        <div style={{ marginTop: '10px', fontSize: '12px', color: '#86efac' }}>
          当前路线测试数：{(activeSystemProfile?.validationRecords || []).length}
        </div>
      </div>
    </div>
  </div>
);
