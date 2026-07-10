const panelStyle = {
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 16,
  background: 'rgba(15, 23, 42, 0.42)',
};

const statCard = (label, value, color = '#67e8f9') => (
  <div
    key={label}
    style={{
      ...panelStyle,
      padding: '14px 16px',
      minHeight: 82,
    }}
  >
    <div style={{ color: '#64748b', fontSize: 10, marginBottom: 6 }}>{label}</div>
    <div style={{ color, fontSize: 22, fontWeight: 900, fontFamily: 'monospace' }}>{value}</div>
  </div>
);

const sectionTitle = (title, color = '#67e8f9') => (
  <div style={{ color, fontSize: 12, fontWeight: 900, letterSpacing: 1.5, textTransform: 'uppercase', marginBottom: 12 }}>
    {title}
  </div>
);

const listItems = (items, color = '#cbd5e1') => (
  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 10 }}>
    {(items || []).map((item, idx) => (
      <div key={idx} style={{ color, fontSize: 12, lineHeight: 1.65, padding: '10px 12px', borderRadius: 10, background: 'rgba(255,255,255,0.03)' }}>
        {item}
      </div>
    ))}
  </div>
);

const CollapsibleSection = ({ title, summary, children, color = '#67e8f9' }) => (
  <details
    style={{
      ...panelStyle,
      padding: 0,
      overflow: 'hidden',
    }}
  >
    <summary
      style={{
        padding: '16px 18px',
        cursor: 'pointer',
        color: '#f8fafc',
        fontSize: 14,
        fontWeight: 800,
        listStyle: 'none',
      }}
    >
      <span style={{ color, marginRight: 10 }}>◆</span>
      {title}
      {summary ? <span style={{ color: '#64748b', fontSize: 12, fontWeight: 500, marginLeft: 12 }}>{summary}</span> : null}
    </summary>
    <div style={{ padding: '0 18px 18px', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
      {children}
    </div>
  </details>
);

export const ResearchProgressTab = ({
  selectedRoute,
  expandedFormulaIdx,
  setExpandedFormulaIdx,
  dnnAnalysisPlan,
  expandedEngPhase,
  setExpandedEngPhase,
  mergedMilestoneStages,
  multimodalView,
  setMultimodalView,
  multimodalError,
  selectedMultimodalReport,
  selectedMultimodalData,
  selectedMultimodalLatest,
  multimodalMetricRows,
}) => {
  const stats = selectedRoute?.stats || {};
  const theoryFormulas = selectedRoute?.theoryFormulas || [];
  const theoryBullets = selectedRoute?.theoryBullets || [];
  const engineeringItems = selectedRoute?.engineeringItems || [];
  const milestones = mergedMilestoneStages || [];

  return (
    <div style={{ animation: 'roadmapFade 0.5s ease-out', display: 'flex', flexDirection: 'column', gap: 18 }}>
      <section style={{ ...panelStyle, padding: 24, background: 'linear-gradient(135deg, rgba(8,47,73,0.58), rgba(15,23,42,0.5))' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 20, alignItems: 'flex-start', flexWrap: 'wrap' }}>
          <div style={{ maxWidth: 760 }}>
            <div style={{ color: '#67e8f9', fontSize: 11, fontWeight: 900, letterSpacing: 1.4, marginBottom: 8 }}>
              模型研发路线
            </div>
            <h2 style={{ fontSize: 28, fontWeight: 900, color: '#fff', margin: '0 0 8px' }}>
              {selectedRoute.title}
            </h2>
            <div style={{ color: '#94a3b8', fontSize: 13, lineHeight: 1.75 }}>
              {selectedRoute.routeDescription || selectedRoute.theorySummary || selectedRoute.subtitle}
            </div>
          </div>
          <div style={{ textAlign: 'right', minWidth: 140 }}>
            <div style={{ fontSize: 34, fontWeight: 900, color: '#00d2ff', fontFamily: 'monospace' }}>
              {stats.routeProgress ?? 0}%
            </div>
            <div style={{ fontSize: 10, color: '#64748b' }}>ROUTE READINESS</div>
          </div>
        </div>
      </section>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12 }}>
        {statCard('总运行', stats.totalRuns ?? '-')}
        {statCard('已完成', stats.completedRuns ?? '-', '#86efac')}
        {statCard('平均分', stats.avgScore != null ? `${(stats.avgScore * 100).toFixed(1)}%` : '-', '#fcd34d')}
        {statCard('工程项', engineeringItems.length, '#c4b5fd')}
      </div>

      <section style={{ ...panelStyle, padding: 20 }}>
        {sectionTitle('智能理论')}
        <div style={{ color: '#fff', fontSize: 20, fontWeight: 850, marginBottom: 8 }}>{selectedRoute.theoryTitle}</div>
        <div style={{ color: '#cbd5e1', fontSize: 13, lineHeight: 1.8, marginBottom: 14 }}>{selectedRoute.theorySummary}</div>
        {theoryFormulas.length > 0 ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 10 }}>
            {theoryFormulas.map((item, idx) => {
              const expanded = expandedFormulaIdx === idx;
              return (
                <button
                  key={idx}
                  type="button"
                  onClick={() => setExpandedFormulaIdx(expanded ? null : idx)}
                  style={{
                    textAlign: 'left',
                    padding: '12px 14px',
                    borderRadius: 12,
                    border: expanded ? '1px solid rgba(103,232,249,0.65)' : '1px solid rgba(255,255,255,0.08)',
                    background: expanded ? 'rgba(0,210,255,0.1)' : 'rgba(0,0,0,0.22)',
                    cursor: 'pointer',
                    fontFamily: 'inherit',
                  }}
                >
                  <div style={{ fontSize: 11, color: '#67e8f9', marginBottom: 6, fontWeight: 800 }}>{item.title}</div>
                  <div style={{ fontSize: 16, color: '#e0f2fe', fontFamily: 'serif' }}>{item.formula}</div>
                  {expanded ? <div style={{ marginTop: 10, color: '#cffafe', fontSize: 12, lineHeight: 1.65 }}>{item.detail || '暂无详细说明'}</div> : null}
                </button>
              );
            })}
          </div>
        ) : listItems(theoryBullets)}
      </section>

      <CollapsibleSection title="深度神经网络分析" summary="目标、指标、六层框架、实验矩阵" color="#818cf8">
        {sectionTitle(dnnAnalysisPlan.subtitle || '深度神经网络分析', '#818cf8')}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 14 }}>
          <div>{sectionTitle('研究目标', '#a5b4fc')}{listItems(dnnAnalysisPlan.goals)}</div>
          <div>{sectionTitle('评估指标', '#a5b4fc')}{listItems(dnnAnalysisPlan.metrics)}</div>
          <div>{sectionTitle('六层框架', '#a5b4fc')}{listItems(dnnAnalysisPlan.framework)}</div>
          <div>{sectionTitle('成败判据', '#a5b4fc')}{listItems(dnnAnalysisPlan.successCriteria)}</div>
        </div>
      </CollapsibleSection>

      <CollapsibleSection title="工程实现" summary={`${engineeringItems.length} 个工程项`} color="#f59e0b">
        <div style={{ color: '#f8d7a6', fontSize: 12, lineHeight: 1.65, marginBottom: 12 }}>
          {selectedRoute.engineeringProcessDescription || '该路线计算过程说明待补充。'}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {engineeringItems.map((item, idx) => (
            <div key={idx} style={{ borderRadius: 12, border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(255,255,255,0.02)' }}>
              <button
                type="button"
                onClick={() => setExpandedEngPhase(expandedEngPhase === idx ? null : idx)}
                style={{
                  width: '100%',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: 12,
                  padding: '13px 15px',
                  background: 'transparent',
                  border: 'none',
                  color: '#fff',
                  textAlign: 'left',
                  fontFamily: 'inherit',
                }}
              >
                <div>
                  <div style={{ fontWeight: 800, fontSize: 13 }}>{item.name}</div>
                  <div style={{ color: '#777', fontSize: 11, marginTop: 3 }}>{item.focus}</div>
                </div>
                <div style={{ fontSize: 10, color: item.status === 'done' ? '#10b981' : item.status === 'in_progress' ? '#f59e0b' : '#666' }}>
                  {String(item.status || 'pending').toUpperCase()}
                </div>
              </button>
              {expandedEngPhase === idx ? (
                <div style={{ padding: '0 15px 13px', color: '#aaa', fontSize: 12, lineHeight: 1.65, borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                  {item.detail || item.analysis || item.work_content || item.target || '该结构部件正在构建中。'}
                </div>
              ) : null}
            </div>
          ))}
        </div>
      </CollapsibleSection>

      <CollapsibleSection title="里程碑与测试记录" summary={`${milestones.length} 个阶段`} color="#10b981">
        <div style={{ color: '#fff', fontSize: 18, fontWeight: 850, marginBottom: 12 }}>{selectedRoute.milestoneTitle}</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12 }}>
          {milestones.map((stage) => (
            <div key={stage.id || stage.name} style={{ border: '1px solid rgba(255,255,255,0.08)', borderRadius: 12, background: 'rgba(255,255,255,0.02)', padding: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, marginBottom: 8 }}>
                <div style={{ color: '#ecfdf5', fontWeight: 800, fontSize: 13 }}>{stage.name}</div>
                <div style={{ fontSize: 10, color: stage.status === 'done' ? '#10b981' : stage.status === 'in_progress' ? '#f59e0b' : '#60a5fa' }}>
                  {String(stage.status || 'planned').toUpperCase()}
                </div>
              </div>
              {listItems(stage.featurePoints || [], '#d1fae5')}
            </div>
          ))}
        </div>
      </CollapsibleSection>

      {selectedRoute?.id === 'fiber_bundle' ? (
        <CollapsibleSection title="多模态纤维训练结果" summary="视觉纤维 / 视觉-语言联络" color="#60a5fa">
          <div style={{ display: 'flex', gap: 8, marginBottom: 14, flexWrap: 'wrap' }}>
            {[
              { id: 'vision_alignment', label: '视觉纤维训练' },
              { id: 'multimodal_connector', label: '视觉-语言联络训练' },
            ].map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setMultimodalView(item.id)}
                style={{
                  border: '1px solid rgba(255,255,255,0.2)',
                  borderRadius: 999,
                  cursor: 'pointer',
                  padding: '6px 10px',
                  fontSize: 11,
                  color: multimodalView === item.id ? '#dbeafe' : '#93c5fd',
                  background: multimodalView === item.id ? 'rgba(59,130,246,0.25)' : 'rgba(59,130,246,0.08)',
                }}
              >
                {item.label}
              </button>
            ))}
          </div>
          {multimodalError ? <div style={{ fontSize: 12, color: '#fca5a5' }}>加载失败：{multimodalError}</div> : null}
          {!multimodalError && !selectedMultimodalReport ? <div style={{ fontSize: 12, color: '#93c5fd' }}>暂无结果。</div> : null}
          {!multimodalError && selectedMultimodalReport ? (
            <div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 10, marginBottom: 12 }}>
                {multimodalMetricRows.map((item) => statCard(item.label, item.value, '#e0f2fe'))}
              </div>
              <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.65 }}>
                <div><span style={{ color: '#93c5fd' }}>分析类型：</span>{selectedMultimodalData?.analysis_type || '-'}</div>
                <div><span style={{ color: '#93c5fd' }}>最新运行：</span>{selectedMultimodalLatest?.run_id || '-'}</div>
              </div>
            </div>
          ) : null}
        </CollapsibleSection>
      ) : null}
    </div>
  );
};
