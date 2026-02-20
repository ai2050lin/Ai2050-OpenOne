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
}) => (
  <div style={{ animation: 'roadmapFade 0.5s ease-out' }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '34px' }}>
      <div>
        <h2 style={{ fontSize: '34px', fontWeight: '900', color: '#fff', margin: '0 0 8px 0' }}>
          {selectedRoute.title} - {selectedRoute.subtitle}
        </h2>
        <div style={{ marginTop: '8px', color: '#666', fontSize: '13px' }}>
          {selectedRoute.routeDescription || selectedRoute.theorySummary}
        </div>
      </div>
      <div style={{ textAlign: 'right' }}>
        <div style={{ fontSize: '30px', fontWeight: '900', color: '#00d2ff', fontFamily: 'monospace' }}>
          {selectedRoute.stats.routeProgress}%
        </div>
        <div style={{ fontSize: '10px', color: '#444' }}>ROUTE READINESS</div>
      </div>
    </div>

    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '22px' }}>
      <div
        style={{
          padding: '28px',
          background: 'rgba(0, 210, 255, 0.06)',
          border: '1px solid rgba(0, 210, 255, 0.25)',
          borderRadius: '22px',
        }}
      >
        <div style={{ fontSize: '12px', color: '#00d2ff', fontWeight: 'bold', letterSpacing: '2px', marginBottom: '10px' }}>
          智能理论
        </div>
        <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#fff', marginBottom: '10px' }}>{selectedRoute.theoryTitle}</div>
        <div style={{ fontSize: '14px', color: '#bbb', lineHeight: '1.8', marginBottom: '14px' }}>{selectedRoute.theorySummary}</div>
        {(selectedRoute.theoryFormulas || []).length === 0 ? (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
            {(selectedRoute.theoryBullets || []).map((item, idx) => (
              <div key={idx} style={{ padding: '12px 14px', borderRadius: '12px', background: 'rgba(255,255,255,0.03)', color: '#ddd', fontSize: '12px' }}>
                {item}
              </div>
            ))}
          </div>
        ) : null}
        {(selectedRoute.theoryFormulas || []).length > 0 ? (
          <div style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
            {selectedRoute.theoryFormulas.map((item, idx) => {
              const expanded = expandedFormulaIdx === idx;
              return (
                <div
                  key={idx}
                  onClick={() => setExpandedFormulaIdx(expanded ? null : idx)}
                  style={{
                    padding: '12px 14px',
                    borderRadius: '12px',
                    border: expanded ? '1px solid rgba(103, 232, 249, 0.65)' : '1px solid rgba(0,210,255,0.35)',
                    background: expanded ? 'rgba(0, 210, 255, 0.12)' : 'rgba(0,0,0,0.3)',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                  }}
                >
                  <div style={{ fontSize: '11px', color: '#67e8f9', marginBottom: '6px', fontWeight: 'bold' }}>
                    {item.title}
                  </div>
                  <div style={{ fontSize: '18px', color: '#e0f2fe', fontFamily: 'serif' }}>{item.formula}</div>
                  {expanded ? (
                    <div
                      style={{
                        marginTop: '10px',
                        paddingTop: '10px',
                        borderTop: '1px solid rgba(255,255,255,0.12)',
                        fontSize: '12px',
                        color: '#cffafe',
                        lineHeight: '1.65',
                      }}
                    >
                      {item.detail || '暂无详细说明'}
                    </div>
                  ) : (
                    <div style={{ marginTop: '8px', fontSize: '10px', color: '#7dd3fc' }}>点击展开详细说明</div>
                  )}
                </div>
              );
            })}
          </div>
        ) : null}
      </div>

      <div
        style={{
          padding: '28px',
          background: 'rgba(99, 102, 241, 0.06)',
          border: '1px solid rgba(99, 102, 241, 0.26)',
          borderRadius: '22px',
        }}
      >
        <div style={{ fontSize: '12px', color: '#818cf8', fontWeight: 'bold', letterSpacing: '2px', marginBottom: '10px' }}>
          深度神经网络分析
        </div>
        <div style={{ fontSize: '20px', color: '#fff', fontWeight: 'bold', marginBottom: '8px' }}>{dnnAnalysisPlan.subtitle}</div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '12px' }}>
          <div style={{ borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0,0,0,0.22)', padding: '12px' }}>
            <div style={{ fontSize: '11px', color: '#a5b4fc', fontWeight: 'bold', marginBottom: '6px' }}>研究目标与假设</div>
            {(dnnAnalysisPlan.goals || []).map((item, idx) => (
              <div key={idx} style={{ fontSize: '12px', color: '#e0e7ff', lineHeight: '1.6', marginBottom: '4px' }}>
                {idx + 1}. {item}
              </div>
            ))}
          </div>
          <div style={{ borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0,0,0,0.22)', padding: '12px' }}>
            <div style={{ fontSize: '11px', color: '#a5b4fc', fontWeight: 'bold', marginBottom: '6px' }}>核心评估指标</div>
            {(dnnAnalysisPlan.metrics || []).map((item, idx) => (
              <div key={idx} style={{ fontSize: '12px', color: '#dbeafe', lineHeight: '1.6', marginBottom: '4px' }}>
                {idx + 1}. {item}
              </div>
            ))}
          </div>
        </div>

        <div style={{ borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0,0,0,0.22)', padding: '12px', marginBottom: '12px' }}>
          <div style={{ fontSize: '11px', color: '#a5b4fc', fontWeight: 'bold', marginBottom: '6px' }}>六层分析框架</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
            {(dnnAnalysisPlan.framework || []).map((item, idx) => (
              <div key={idx} style={{ fontSize: '12px', color: '#c7d2fe', lineHeight: '1.6', padding: '8px 10px', borderRadius: '10px', background: 'rgba(255,255,255,0.03)' }}>
                {item}
              </div>
            ))}
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px' }}>
          <div style={{ borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0,0,0,0.22)', padding: '12px' }}>
            <div style={{ fontSize: '11px', color: '#a5b4fc', fontWeight: 'bold', marginBottom: '6px' }}>实验矩阵</div>
            {(dnnAnalysisPlan.experimentMatrix || []).map((item, idx) => (
              <div key={idx} style={{ fontSize: '12px', color: '#dbeafe', lineHeight: '1.6', marginBottom: '4px' }}>
                {idx + 1}. {item}
              </div>
            ))}
          </div>
          <div style={{ borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0,0,0,0.22)', padding: '12px' }}>
            <div style={{ fontSize: '11px', color: '#a5b4fc', fontWeight: 'bold', marginBottom: '6px' }}>里程碑</div>
            {(dnnAnalysisPlan.milestones || []).map((item, idx) => (
              <div key={idx} style={{ fontSize: '12px', color: '#dbeafe', lineHeight: '1.6', marginBottom: '4px' }}>
                {item}
              </div>
            ))}
          </div>
          <div style={{ borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0,0,0,0.22)', padding: '12px' }}>
            <div style={{ fontSize: '11px', color: '#a5b4fc', fontWeight: 'bold', marginBottom: '6px' }}>成败判据</div>
            {(dnnAnalysisPlan.successCriteria || []).map((item, idx) => (
              <div key={idx} style={{ fontSize: '12px', color: '#dbeafe', lineHeight: '1.6', marginBottom: '4px' }}>
                {item}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div
        style={{
          padding: '28px',
          background: 'rgba(245, 158, 11, 0.05)',
          border: '1px solid rgba(245, 158, 11, 0.22)',
          borderRadius: '22px',
        }}
      >
        <div style={{ fontSize: '12px', color: '#f59e0b', fontWeight: 'bold', letterSpacing: '2px', marginBottom: '14px' }}>
          工程实现
        </div>
        <div
          style={{
            marginBottom: '12px',
            padding: '10px 12px',
            borderRadius: '10px',
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.08)',
            fontSize: '12px',
            color: '#f8d7a6',
            lineHeight: '1.65',
          }}
        >
          计算过程说明：{selectedRoute.engineeringProcessDescription || '该路线计算过程说明待补充。'}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {(selectedRoute.engineeringItems || []).map((item, idx) => (
            <div key={idx} style={{ borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)', background: 'rgba(255,255,255,0.02)' }}>
              <div
                onClick={() => setExpandedEngPhase(expandedEngPhase === idx ? null : idx)}
                style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '14px 16px' }}
              >
                <div>
                  <div style={{ color: '#fff', fontWeight: 'bold', fontSize: '14px' }}>{item.name}</div>
                  <div style={{ color: '#777', fontSize: '11px', marginTop: '3px' }}>{item.focus}</div>
                </div>
                <div style={{ fontSize: '10px', color: item.status === 'done' ? '#10b981' : item.status === 'in_progress' ? '#f59e0b' : '#666' }}>
                  {String(item.status || 'pending').toUpperCase()}
                </div>
              </div>
              {expandedEngPhase === idx && (
                <div style={{ padding: '0 16px 14px 16px', color: '#aaa', fontSize: '12px', lineHeight: '1.6', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                  {item.detail || item.analysis || item.work_content || item.target || '该结构部件正在构建中。'}
                </div>
              )}
            </div>
          ))}
        </div>
        {(selectedRoute.nfbtProcessSteps || []).length > 0 ? (
          <div style={{ marginTop: '14px', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', overflow: 'hidden', background: 'rgba(0,0,0,0.25)' }}>
            <div style={{ padding: '10px 12px', fontSize: '12px', color: '#fbbf24', fontWeight: 'bold', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
              NFBT 计算过程
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: '760px' }}>
                <thead>
                  <tr style={{ background: 'rgba(255,255,255,0.04)' }}>
                    <th style={{ textAlign: 'left', padding: '8px 10px', fontSize: '11px', color: '#fcd34d' }}>步骤 (Step)</th>
                    <th style={{ textAlign: 'left', padding: '8px 10px', fontSize: '11px', color: '#fcd34d' }}>输入 (Input)</th>
                    <th style={{ textAlign: 'left', padding: '8px 10px', fontSize: '11px', color: '#fcd34d' }}>输出 (Output)</th>
                    <th style={{ textAlign: 'left', padding: '8px 10px', fontSize: '11px', color: '#fcd34d' }}>复杂度</th>
                    <th style={{ textAlign: 'left', padding: '8px 10px', fontSize: '11px', color: '#fcd34d' }}>核心操作</th>
                  </tr>
                </thead>
                <tbody>
                  {(selectedRoute.nfbtProcessSteps || []).map((row, idx) => (
                    <tr key={idx} style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                      <td style={{ padding: '8px 10px', fontSize: '12px', color: '#fff' }}>{row.step}</td>
                      <td style={{ padding: '8px 10px', fontSize: '12px', color: '#cbd5e1', fontFamily: 'monospace' }}>{row.input}</td>
                      <td style={{ padding: '8px 10px', fontSize: '12px', color: '#cbd5e1', fontFamily: 'monospace' }}>{row.output}</td>
                      <td style={{ padding: '8px 10px', fontSize: '12px', color: '#93c5fd', fontFamily: 'monospace' }}>{row.complexity}</td>
                      <td style={{ padding: '8px 10px', fontSize: '12px', color: '#a7f3d0' }}>{row.op}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ padding: '10px 12px', borderTop: '1px solid rgba(255,255,255,0.08)', fontSize: '12px', color: '#fde68a', lineHeight: '1.65' }}>
              {selectedRoute.nfbtOptimization}
            </div>
          </div>
        ) : null}
      </div>

      <div
        style={{
          padding: '28px',
          background: 'rgba(16, 185, 129, 0.06)',
          border: '1px solid rgba(16, 185, 129, 0.22)',
          borderRadius: '22px',
        }}
      >
        <div style={{ fontSize: '12px', color: '#10b981', fontWeight: 'bold', letterSpacing: '2px', marginBottom: '12px' }}>
          里程碑（原 AGI 终点）
        </div>
        <div style={{ fontSize: '18px', color: '#fff', fontWeight: 'bold', marginBottom: '12px' }}>{selectedRoute.milestoneTitle}</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '14px' }}>
          {mergedMilestoneStages.map((stage) => (
            <div key={stage.id || stage.name} style={{ border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', background: 'rgba(255,255,255,0.02)', padding: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <div style={{ color: '#ecfdf5', fontWeight: 'bold', fontSize: '14px' }}>{stage.name}</div>
                <div style={{ fontSize: '10px', color: stage.status === 'done' ? '#10b981' : stage.status === 'in_progress' ? '#f59e0b' : '#60a5fa' }}>
                  {String(stage.status || 'planned').toUpperCase()}
                </div>
              </div>

              <div style={{ marginBottom: '8px' }}>
                <div style={{ fontSize: '11px', color: '#6ee7b7', marginBottom: '4px', fontWeight: 'bold' }}>功能点</div>
                {(stage.featurePoints || []).map((point, idx) => (
                  <div key={idx} style={{ display: 'flex', gap: '8px', color: '#d1fae5', fontSize: '12px', marginBottom: '4px' }}>
                    <span style={{ width: '5px', height: '5px', borderRadius: '50%', background: '#10b981', marginTop: '7px' }} />
                    {point}
                  </div>
                ))}
              </div>

              <div>
                <div style={{ fontSize: '11px', color: '#7dd3fc', marginBottom: '6px', fontWeight: 'bold' }}>测试记录</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '8px' }}>
                  {(stage.tests || []).map((test, idx) => (
                    <div key={idx} style={{ border: '1px solid rgba(255,255,255,0.08)', borderRadius: '10px', padding: '10px', background: 'rgba(0,0,0,0.25)' }}>
                      <div style={{ color: '#fff', fontSize: '13px', fontWeight: 'bold', marginBottom: '6px' }}>{test.name}</div>
                      <div style={{ fontSize: '12px', color: '#cbd5e1', lineHeight: '1.6' }}>
                        <div><span style={{ color: '#fcd34d' }}>参数配置：</span>{test.params || '-'}</div>
                        <div><span style={{ color: '#fcd34d' }}>数据集：</span>{test.dataset || '-'}</div>
                        <div><span style={{ color: '#fcd34d' }}>测试结果：</span>{test.result || '-'}</div>
                        <div><span style={{ color: '#fcd34d' }}>分析总结：</span>{test.summary || '-'}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>

        {selectedRoute.milestonePlanEvaluation ? (
          <div style={{ marginTop: '14px', borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '12px' }}>
            <div style={{ fontSize: '11px', color: '#6ee7b7', fontWeight: 'bold', marginBottom: '6px' }}>方案评估与修改建议</div>
            <div style={{ fontSize: '12px', color: '#d1fae5', marginBottom: '8px', lineHeight: '1.65' }}>
              评估：{selectedRoute.milestonePlanEvaluation.assessment}
            </div>
            <div>
              {(selectedRoute.milestonePlanEvaluation.suggestions || []).map((item, idx) => (
                <div key={idx} style={{ fontSize: '12px', color: '#bae6fd', lineHeight: '1.6', marginBottom: '4px' }}>
                  {idx + 1}. {item}
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </div>

      {selectedRoute?.id === 'fiber_bundle' ? (
        <div style={{ padding: '28px', background: 'rgba(59, 130, 246, 0.06)', border: '1px solid rgba(59, 130, 246, 0.22)', borderRadius: '22px' }}>
          <div style={{ fontSize: '12px', color: '#60a5fa', fontWeight: 'bold', letterSpacing: '2px', marginBottom: '10px' }}>
            多模态纤维训练结果
          </div>
          <div style={{ display: 'flex', gap: '8px', marginBottom: '14px', flexWrap: 'wrap' }}>
            {[
              { id: 'vision_alignment', label: '视觉纤维训练' },
              { id: 'multimodal_connector', label: '视觉-语言联络训练' },
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => setMultimodalView(item.id)}
                style={{
                  border: '1px solid rgba(255,255,255,0.2)',
                  borderRadius: '999px',
                  cursor: 'pointer',
                  padding: '6px 10px',
                  fontSize: '11px',
                  color: multimodalView === item.id ? '#dbeafe' : '#93c5fd',
                  background: multimodalView === item.id ? 'rgba(59,130,246,0.25)' : 'rgba(59,130,246,0.08)',
                }}
              >
                {item.label}
              </button>
            ))}
          </div>

          {multimodalError ? (
            <div style={{ fontSize: '12px', color: '#fca5a5', lineHeight: '1.6' }}>加载失败：{multimodalError}</div>
          ) : null}

          {!multimodalError && !selectedMultimodalReport ? (
            <div style={{ fontSize: '12px', color: '#93c5fd' }}>暂无结果。先运行对应训练脚本后可在这里切换查看。</div>
          ) : null}

          {!multimodalError && selectedMultimodalReport ? (
            <div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(120px, 1fr))', gap: '10px', marginBottom: '12px' }}>
                {multimodalMetricRows.map((item) => (
                  <div key={item.label} style={{ border: '1px solid rgba(255,255,255,0.1)', borderRadius: '10px', background: 'rgba(0,0,0,0.2)', padding: '10px' }}>
                    <div style={{ fontSize: '10px', color: '#93c5fd', marginBottom: '4px' }}>{item.label}</div>
                    <div style={{ fontSize: '14px', color: '#e0f2fe', fontWeight: 'bold', fontFamily: 'monospace' }}>{item.value}</div>
                  </div>
                ))}
              </div>
              <div style={{ fontSize: '12px', color: '#cbd5e1', lineHeight: '1.65' }}>
                <div><span style={{ color: '#93c5fd' }}>分析类型：</span>{selectedMultimodalData?.analysis_type || '-'}</div>
                <div><span style={{ color: '#93c5fd' }}>数据集：</span>{selectedMultimodalReport?.meta?.dataset || selectedMultimodalReport?.config?.dataset || '-'}</div>
                <div><span style={{ color: '#93c5fd' }}>最新运行：</span>{selectedMultimodalLatest?.run_id || '-'}</div>
                <div><span style={{ color: '#93c5fd' }}>时间：</span>{selectedMultimodalLatest?.timestamp || '-'}</div>
              </div>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  </div>
);
