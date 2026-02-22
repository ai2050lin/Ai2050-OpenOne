export const ProjectRoadmapTab = ({
  roadmapData,
  analysisPhase,
  evidenceDrivenPlan,
  mathRouteSystemPlan,
  improvements,
  expandedImprovementPhase,
  setExpandedImprovementPhase,
  expandedImprovementTest,
  setExpandedImprovementTest,
}) => (
  <div style={{ animation: 'roadmapFade 0.6s ease-out', maxWidth: '1000px', margin: '0 auto' }}>
    <div style={{ marginBottom: '34px' }}>
      <h2 style={{ fontSize: '30px', fontWeight: '900', color: '#ffaa00', marginBottom: '10px' }}>项目大纲</h2>
      <div style={{ color: '#777', fontSize: '14px' }}>{roadmapData?.definition?.summary}</div>
    </div>

    <div
      style={{
        padding: '30px',
        background: 'linear-gradient(135deg, rgba(255,170,0,0.12) 0%, rgba(255,170,0,0.03) 100%)',
        border: '1px solid rgba(255,170,0,0.24)',
        borderRadius: '24px',
        marginBottom: '28px',
      }}
    >
      <div style={{ color: '#ffaa00', fontWeight: 'bold', fontSize: '18px', marginBottom: '16px' }}>核心思路</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '10px' }}>
        {[
          '1，大脑有非常特殊的数学结构，产生了智能。',
          '2，深度神经网络部分还原了这个结构，产生了语言能力。',
          '3，通过分析深度神经网络，研究这个数学结构，完成智能理论。',
        ].map((line, idx) => (
          <div
            key={idx}
            style={{
              padding: '14px 16px',
              borderRadius: '12px',
              background: 'rgba(255,255,255,0.05)',
              color: '#f4e4c1',
              fontSize: '14px',
              lineHeight: '1.6',
            }}
          >
            {line}
          </div>
        ))}
      </div>
    </div>

    <div
      style={{
        padding: '30px',
        borderRadius: '24px',
        border: '1px solid rgba(99,102,241,0.28)',
        background: 'linear-gradient(135deg, rgba(99,102,241,0.10) 0%, rgba(99,102,241,0.03) 100%)',
        marginBottom: '28px',
      }}
    >
      <div style={{ color: '#818cf8', fontWeight: 'bold', fontSize: '18px', marginBottom: '8px' }}>
        {mathRouteSystemPlan.title}
      </div>
      <div style={{ color: '#c7d2fe', fontSize: '13px', lineHeight: '1.7', marginBottom: '14px' }}>
        {mathRouteSystemPlan.subtitle}
      </div>

      <div
        style={{
          marginTop: '12px',
          borderRadius: '12px',
          border: '1px solid rgba(255,255,255,0.08)',
          background: 'rgba(0,0,0,0.22)',
          padding: '12px',
        }}
      >
        <div style={{ color: '#a5b4fc', fontSize: '11px', fontWeight: 'bold', marginBottom: '8px' }}>数学路线</div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', minWidth: '1440px', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: 'rgba(255,255,255,0.05)' }}>
                <th
                  style={{
                    textAlign: 'left',
                    padding: '8px 10px',
                    color: '#c7d2fe',
                    fontSize: '11px',
                    borderBottom: '1px solid rgba(255,255,255,0.08)',
                  }}
                >
                  路线
                </th>
                <th
                  style={{
                    textAlign: 'left',
                    padding: '8px 10px',
                    color: '#93c5fd',
                    fontSize: '11px',
                    borderBottom: '1px solid rgba(255,255,255,0.08)',
                  }}
                >
                  路线说明
                </th>
                <th
                  style={{
                    textAlign: 'left',
                    padding: '8px 10px',
                    color: '#86efac',
                    fontSize: '11px',
                    borderBottom: '1px solid rgba(255,255,255,0.08)',
                  }}
                >
                  优点
                </th>
                <th
                  style={{
                    textAlign: 'left',
                    padding: '8px 10px',
                    color: '#fca5a5',
                    fontSize: '11px',
                    borderBottom: '1px solid rgba(255,255,255,0.08)',
                  }}
                >
                  缺点
                </th>
                <th
                  style={{
                    textAlign: 'left',
                    padding: '8px 10px',
                    color: '#93c5fd',
                    fontSize: '11px',
                    borderBottom: '1px solid rgba(255,255,255,0.08)',
                  }}
                >
                  可行性结论
                </th>
                <th
                  style={{
                    textAlign: 'left',
                    padding: '8px 10px',
                    color: '#c7d2fe',
                    fontSize: '11px',
                    borderBottom: '1px solid rgba(255,255,255,0.08)',
                  }}
                >
                  理论深度
                </th>
                <th
                  style={{
                    textAlign: 'left',
                    padding: '8px 10px',
                    color: '#c7d2fe',
                    fontSize: '11px',
                    borderBottom: '1px solid rgba(255,255,255,0.08)',
                  }}
                >
                  计算可行性
                </th>
                <th
                  style={{
                    textAlign: 'left',
                    padding: '8px 10px',
                    color: '#c7d2fe',
                    fontSize: '11px',
                    borderBottom: '1px solid rgba(255,255,255,0.08)',
                  }}
                >
                  可解释性
                </th>
                <th
                  style={{
                    textAlign: 'left',
                    padding: '8px 10px',
                    color: '#c7d2fe',
                    fontSize: '11px',
                    borderBottom: '1px solid rgba(255,255,255,0.08)',
                  }}
                >
                  与 SHMC/NFBT 兼容
                </th>
              </tr>
            </thead>
            <tbody>
              {(mathRouteSystemPlan.routeAnalysis || []).map((item, idx) => (
                <tr key={idx} style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                  <td style={{ padding: '9px 10px', color: '#e0e7ff', fontSize: '12px', fontWeight: 'bold', verticalAlign: 'top' }}>
                    {item.route}
                  </td>
                  <td style={{ padding: '9px 10px', color: '#bfdbfe', fontSize: '11px', lineHeight: '1.55', verticalAlign: 'top' }}>
                    {item.routeSummary || item.description || item.routeDesc || ((item.pros || [])[0] || '—')}
                  </td>
                  <td style={{ padding: '9px 10px', color: '#dcfce7', fontSize: '11px', lineHeight: '1.55', verticalAlign: 'top' }}>
                    {(item.pros || []).map((line, pIdx) => (
                      <div key={pIdx}>{pIdx + 1}. {line}</div>
                    ))}
                  </td>
                  <td style={{ padding: '9px 10px', color: '#fee2e2', fontSize: '11px', lineHeight: '1.55', verticalAlign: 'top' }}>
                    {(item.cons || []).map((line, cIdx) => (
                      <div key={cIdx}>{cIdx + 1}. {line}</div>
                    ))}
                  </td>
                  <td style={{ padding: '9px 10px', color: '#bae6fd', fontSize: '11px', lineHeight: '1.55', verticalAlign: 'top' }}>
                    {item.feasibility}
                  </td>
                  <td style={{ padding: '9px 10px', color: '#dbeafe', fontSize: '12px', verticalAlign: 'top' }}>{item.depth}</td>
                  <td style={{ padding: '9px 10px', color: '#dbeafe', fontSize: '12px', verticalAlign: 'top' }}>{item.compute}</td>
                  <td style={{ padding: '9px 10px', color: '#dbeafe', fontSize: '12px', verticalAlign: 'top' }}>{item.interpret}</td>
                  <td style={{ padding: '9px 10px', color: '#dbeafe', fontSize: '12px', verticalAlign: 'top' }}>{item.compatibility}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div style={{ marginTop: '12px', display: 'grid', gridTemplateColumns: '1.2fr 1fr 1fr', gap: '12px' }}>
        <div style={{ padding: '14px', borderRadius: '12px', background: 'rgba(0,0,0,0.22)', border: '1px solid rgba(255,255,255,0.08)' }}>
          <div style={{ color: '#a5b4fc', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px' }}>分层架构</div>
          {(mathRouteSystemPlan.architecture || []).map((line, idx) => (
            <div key={idx} style={{ color: '#e0e7ff', fontSize: '12px', lineHeight: '1.6', marginBottom: '4px' }}>
              {idx + 1}. {line}
            </div>
          ))}
        </div>

        <div style={{ padding: '14px', borderRadius: '12px', background: 'rgba(0,0,0,0.22)', border: '1px solid rgba(255,255,255,0.08)' }}>
          <div style={{ color: '#a5b4fc', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px' }}>资源配比</div>
          {(mathRouteSystemPlan.allocation || []).map((line, idx) => (
            <div key={idx} style={{ color: '#dbeafe', fontSize: '12px', lineHeight: '1.6', marginBottom: '4px' }}>
              {line}
            </div>
          ))}
        </div>

        <div style={{ padding: '14px', borderRadius: '12px', background: 'rgba(0,0,0,0.22)', border: '1px solid rgba(255,255,255,0.08)' }}>
          <div style={{ color: '#a5b4fc', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px' }}>阶段里程碑</div>
          {(mathRouteSystemPlan.milestones || []).map((line, idx) => (
            <div key={idx} style={{ color: '#dbeafe', fontSize: '12px', lineHeight: '1.6', marginBottom: '4px' }}>
              {idx + 1}. {line}
            </div>
          ))}
        </div>
      </div>
    </div>

    <div
      style={{
        padding: '30px',
        borderRadius: '24px',
        border: '1px solid rgba(56,189,248,0.28)',
        background: 'linear-gradient(135deg, rgba(56,189,248,0.10) 0%, rgba(56,189,248,0.03) 100%)',
        marginBottom: '28px',
      }}
    >
      <div style={{ color: '#38bdf8', fontWeight: 'bold', fontSize: '18px', marginBottom: '8px' }}>
        {evidenceDrivenPlan.title}
      </div>
      <div style={{ color: '#bae6fd', fontSize: '13px', lineHeight: '1.7', marginBottom: '12px' }}>{evidenceDrivenPlan.core}</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '6px', marginBottom: '10px' }}>
        {(evidenceDrivenPlan.overview || []).map((line, idx) => (
          <div key={idx} style={{ color: '#dbeafe', fontSize: '12px', lineHeight: '1.6' }}>
            {idx + 1}. {line}
          </div>
        ))}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '8px' }}>
        {(evidenceDrivenPlan.phases || []).map((item) => (
          <details
            key={item.id}
            style={{
              padding: '12px 14px',
              borderRadius: '12px',
              border: '1px solid rgba(255,255,255,0.10)',
              background: 'rgba(0,0,0,0.18)',
            }}
          >
            <summary style={{ cursor: 'pointer', listStyle: 'none' }}>
              <div style={{ fontSize: '12px', color: '#7dd3fc', fontWeight: 'bold', marginBottom: '4px' }}>
                {item.id} 路 {item.name}
              </div>
              <div style={{ fontSize: '12px', color: '#e0f2fe', lineHeight: '1.6' }}>
                原理说明：{item.desc}
              </div>
              <div style={{ color: '#93c5fd', fontSize: '11px', marginTop: '4px' }}>
                点击展开详细说明
              </div>
            </summary>
            <div style={{ marginTop: '8px' }}>
              <div style={{ color: '#cbd5e1', fontSize: '11px', lineHeight: '1.55' }}>目标：{item.goal}</div>
              <div style={{ color: '#bfdbfe', fontSize: '11px', lineHeight: '1.55' }}>方法：{item.method}</div>
              <div style={{ color: '#a7f3d0', fontSize: '11px', lineHeight: '1.55' }}>证据：{item.evidence}</div>
              <div style={{ color: '#ddd6fe', fontSize: '11px', lineHeight: '1.55' }}>产出：{item.outputs}</div>
              <div style={{ color: '#fcd34d', fontSize: '11px', lineHeight: '1.55' }}>准出：{item.gate}</div>
            </div>
          </details>
        ))}
      </div>
    </div>


    <div
      style={{
        padding: '30px',
        borderRadius: '24px',
        border: '1px solid rgba(16,185,129,0.24)',
        background: 'linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(16,185,129,0.02) 100%)',
      }}
    >
      <div style={{ color: '#10b981', fontWeight: 'bold', fontSize: '18px', marginBottom: '8px' }}>分析进展</div>
      <div style={{ color: '#9ca3af', fontSize: '13px', lineHeight: '1.7', marginBottom: '16px' }}>
        通过五个阶段，尝试完成深度神经网络中数学结构的研究
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '10px' }}>
        {improvements.map((phase) => {
          const isPhaseExpanded = expandedImprovementPhase === phase.id;
          const phaseStatusColor =
            phase.status === 'done' ? '#10b981' : phase.status === 'in_progress' ? '#f59e0b' : '#94a3b8';
          return (
            <div
              key={phase.id}
              style={{
                padding: '14px 16px',
                borderRadius: '12px',
                border: `1px solid ${isPhaseExpanded ? 'rgba(16,185,129,0.45)' : 'rgba(255,255,255,0.08)'}`,
                background: isPhaseExpanded ? 'rgba(16,185,129,0.08)' : 'rgba(255,255,255,0.02)',
              }}
            >
              <button
                onClick={() => {
                  const nextPhase = isPhaseExpanded ? null : phase.id;
                  setExpandedImprovementPhase(nextPhase);
                  setExpandedImprovementTest(null);
                }}
                style={{
                  width: '100%',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  gap: '12px',
                  marginBottom: isPhaseExpanded ? '8px' : 0,
                  background: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  padding: 0,
                  textAlign: 'left',
                }}
              >
                <div style={{ color: '#dcfce7', fontWeight: 'bold', fontSize: '14px' }}>{phase.title}</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div style={{ fontSize: '10px', color: phaseStatusColor }}>{String(phase.status).toUpperCase()}</div>
                  <div style={{ fontSize: '11px', color: '#86efac' }}>{isPhaseExpanded ? '收起' : '展开'}</div>
                </div>
              </button>

              {isPhaseExpanded && (
                <div>
                  <div style={{ color: '#9fe8c7', fontSize: '12px', marginBottom: '6px' }}>阶段目标：{phase.objective}</div>
                  <div style={{ color: '#a7f3d0', fontSize: '12px', lineHeight: '1.6', marginBottom: '10px' }}>
                    阶段总结：{phase.summary}
                  </div>
                  <div style={{ color: '#d1fae5', fontSize: '12px', fontWeight: 'bold', marginBottom: '8px' }}>
                    测试列表（点击查看详细数据）
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '8px' }}>
                    {(phase.tests || []).map((test, testIdx) => {
                      const testKey = `${phase.id}:${test.id}`;
                      const isTestExpanded = expandedImprovementTest === testKey;
                      return (
                        <div
                          key={test.id}
                          style={{
                            borderRadius: '10px',
                            border: `1px solid ${isTestExpanded ? 'rgba(96,165,250,0.5)' : 'rgba(255,255,255,0.08)'}`,
                            background: isTestExpanded ? 'rgba(30,64,175,0.12)' : 'rgba(0,0,0,0.18)',
                            padding: '10px 12px',
                          }}
                        >
                          <button
                            onClick={() => setExpandedImprovementTest(isTestExpanded ? null : testKey)}
                            style={{
                              width: '100%',
                              background: 'transparent',
                              border: 'none',
                              cursor: 'pointer',
                              padding: 0,
                              textAlign: 'left',
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center',
                              gap: '10px',
                            }}
                          >
                            <div style={{ color: '#dbeafe', fontSize: '12px', fontWeight: 'bold' }}>
                              T{testIdx + 1}. {test.name}
                            </div>
                            <div style={{ color: '#93c5fd', fontSize: '11px' }}>{isTestExpanded ? '收起详情' : '查看详情'}</div>
                          </button>

                          <div style={{ color: '#cbd5e1', fontSize: '12px', lineHeight: '1.6', marginTop: '6px' }}>
                            测试目标：{test.target}
                          </div>
                          <div style={{ color: '#93c5fd', fontSize: '12px', lineHeight: '1.6', marginTop: '4px' }}>
                            测试日期：{test.testDate || '未记录'}
                          </div>
                          <div style={{ color: '#94a3b8', fontSize: '12px', lineHeight: '1.6', marginTop: '4px' }}>
                            测试结果：{test.result}
                          </div>
                          <div style={{ color: '#a7f3d0', fontSize: '12px', lineHeight: '1.6', marginTop: '4px' }}>
                            分析总结：{test.analysis}
                          </div>

                          {isTestExpanded && (
                            <div
                              style={{
                                marginTop: '8px',
                                borderRadius: '8px',
                                border: '1px solid rgba(148,163,184,0.35)',
                                background: 'rgba(2,6,23,0.55)',
                                padding: '10px',
                              }}
                            >
                              <div style={{ color: '#bfdbfe', fontSize: '11px', fontWeight: 'bold', marginBottom: '6px' }}>
                                测试参数
                              </div>
                              <pre
                                style={{
                                  margin: 0,
                                  color: '#dbeafe',
                                  fontSize: '11px',
                                  lineHeight: '1.6',
                                  whiteSpace: 'pre-wrap',
                                }}
                              >
                                {JSON.stringify(test.params, null, 2)}
                              </pre>
                              <div
                                style={{
                                  color: '#bfdbfe',
                                  fontSize: '11px',
                                  fontWeight: 'bold',
                                  marginTop: '10px',
                                  marginBottom: '6px',
                                }}
                              >
                                详细测试数据
                              </div>
                              <pre
                                style={{
                                  margin: 0,
                                  color: '#cbd5e1',
                                  fontSize: '11px',
                                  lineHeight: '1.6',
                                  whiteSpace: 'pre-wrap',
                                }}
                              >
                                {JSON.stringify(test.details, null, 2)}
                              </pre>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  </div>
);
