import { useState } from 'react';
import { Tag, Clock, ChevronDown, ChevronRight, Lightbulb, TrendingUp, AlertTriangle, Search, FlaskConical } from 'lucide-react';

export const AIRnDFindingsTab = ({ findings, round }) => {
  const [expandedIdx, setExpandedIdx] = useState(null);

  // Group findings by round
  const groupedByRound = findings.reduce((acc, f, i) => {
    const r = f.round || round || 0;
    if (!acc[r]) acc[r] = [];
    acc[r].push({ ...f, _idx: i });
    return acc;
  }, {});

  const sortedRounds = Object.keys(groupedByRound).sort((a, b) => b - a);

  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: '40px 60px' }} className="airnd-scroll">
      <h2 style={{ fontSize: '24px', fontWeight: '800', color: '#ffaa00', marginBottom: 8, letterSpacing: '1px' }}>
        研究发现
      </h2>
      <p style={{ fontSize: '13px', color: '#9ca3af', marginBottom: 30 }}>
        AI研发引擎自动产出的关键发现 and 结论，汇总神经网络逆向工程的核心洞察。
      </p>

      {findings.length === 0 ? (
        <div style={{ 
          maxWidth: '520px', margin: '80px auto', textAlign: 'center',
          padding: '50px 30px', 
          borderRadius: '24px',
          border: '1px solid rgba(255, 170, 0, 0.28)',
          background: 'linear-gradient(135deg, rgba(255, 170, 0, 0.10) 0%, rgba(255, 170, 0, 0.03) 100%)',
          animation: 'roadmapFade 0.6s ease-out'
        }}>
          <div style={{ marginBottom: 16, display: 'flex', justifyContent: 'center' }}>
            <FlaskConical size={48} color="#ffaa00" style={{ filter: 'drop-shadow(0 0 12px rgba(255, 170, 0, 0.6))' }} />
          </div>
          <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#ffaa00', marginBottom: 10 }}>自动研究发现归档</div>
          <div style={{ fontSize: '13px', color: '#9ca3af', lineHeight: '1.6' }}>
            暂无发现。启动自动研究循环后，引擎在各个阶段产出的核心数学规律、模型机制以及实证发现将自动归档在此。
          </div>
        </div>
      ) : (
        sortedRounds.map(r => (
          <div key={r} style={{ marginBottom: 28 }}>
            <div style={{
              fontSize: '13px', color: '#ffaa00', marginBottom: 12, fontWeight: 'bold',
              display: 'flex', alignItems: 'center', gap: 8,
              borderBottom: '1px solid rgba(255, 170, 0, 0.15)',
              paddingBottom: '8px',
            }}>
              <Clock size={13} style={{ filter: 'drop-shadow(0 0 3px rgba(255, 170, 0, 0.6))' }} /> Round {r}
              <span style={{ color: '#9ca3af', fontSize: '11px', fontWeight: 'normal' }}>({groupedByRound[r].length} 条发现)</span>
            </div>

            {groupedByRound[r].map((finding, i) => (
              <div 
                key={i} 
                className="airnd-card"
                style={{
                  background: 'linear-gradient(135deg, rgba(255, 170, 0, 0.08) 0%, rgba(255, 170, 0, 0.02) 100%)', 
                  borderRadius: '16px',
                  border: '1px solid rgba(255, 170, 0, 0.22)', 
                  marginBottom: 10, overflow: 'hidden',
                }}
              >
                {/* Header */}
                <button
                  onClick={() => setExpandedIdx(expandedIdx === finding._idx ? null : finding._idx)}
                  style={{
                    width: '100%', padding: '14px 20px', background: 'none', border: 'none',
                    color: '#f3f4f6', cursor: 'pointer', display: 'flex', alignItems: 'center',
                    justifyContent: 'space-between', textAlign: 'left',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    {(() => {
                      const getFindingIcon = (emoji) => {
                        switch (emoji) {
                          case '💡': return Lightbulb;
                          case '📈': return TrendingUp;
                          case '⚠️': return AlertTriangle;
                          case '🔍': return Search;
                          case '🧪': return FlaskConical;
                          default: return Lightbulb;
                        }
                      };
                      const IconComponent = getFindingIcon(finding.icon || '💡');
                      return <IconComponent size={16} color="#ffaa00" style={{ filter: 'drop-shadow(0 0 4px rgba(255, 170, 0, 0.6))' }} />;
                    })()}
                    <span style={{ fontSize: '14px', fontWeight: 'bold', color: '#f3f4f6' }}>
                      {finding.title || finding.content?.slice(0, 60) || `发现 #${i + 1}`}
                    </span>
                    {finding.tags?.map(t => (
                      <span key={t} style={{
                        fontSize: '10px', padding: '2px 8px', borderRadius: '4px',
                        background: 'rgba(255, 170, 0, 0.1)', color: '#ffaa00',
                        border: '1px solid rgba(255, 170, 0, 0.25)',
                        fontWeight: 'bold',
                      }}>
                        {t}
                      </span>
                    ))}
                  </div>
                  {expandedIdx === finding._idx 
                    ? <ChevronDown size={16} color="#ffaa00" style={{ filter: 'drop-shadow(0 0 3px rgba(255, 170, 0, 0.6))' }} /> 
                    : <ChevronRight size={16} color="#ffaa00" style={{ filter: 'drop-shadow(0 0 3px rgba(255, 170, 0, 0.6))' }} />
                  }
                </button>

                {/* Expanded content */}
                {expandedIdx === finding._idx && (
                  <div style={{
                    padding: '0 20px 20px', fontSize: '13px', color: '#d1d5db',
                    lineHeight: '1.7', whiteSpace: 'pre-wrap',
                    borderTop: '1px solid rgba(255, 170, 0, 0.15)', paddingTop: 14,
                  }}>
                    <div style={{ color: '#e5e7eb' }}>{finding.content || finding.detail || '无详细内容'}</div>
                    {finding.source && (
                      <div style={{ marginTop: 12, fontSize: '11px', color: '#6b7280', borderTop: '1px dashed rgba(255,255,255,0.04)', paddingTop: '8px', display: 'flex', justifyContent: 'space-between' }}>
                        <span>来源: {finding.source}</span>
                        <span>{finding.timestamp || ''}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        ))
      )}
    </div>
  );
};
