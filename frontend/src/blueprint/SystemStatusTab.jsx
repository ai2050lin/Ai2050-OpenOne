import { Activity, CheckCircle, Search, X } from 'lucide-react';

const badgeStyle = (bg, border, color) => ({
  padding: '4px 8px',
  borderRadius: '999px',
  background: bg,
  border,
  fontSize: '10px',
  color,
});

const panelStyle = {
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 16,
  background: 'rgba(15, 23, 42, 0.42)',
};

const progressCard = (label, value, color, hint) => (
  <div
    key={label}
    style={{
      ...panelStyle,
      padding: '14px 16px',
      minHeight: 86,
    }}
  >
    <div style={{ fontSize: 10, color: '#64748b', marginBottom: 5 }}>{label}</div>
    <div style={{ fontSize: 20, color, fontWeight: 900, fontFamily: 'monospace' }}>{value}</div>
    {hint ? <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 4 }}>{hint}</div> : null}
  </div>
);

const runtimePill = (label, value, color = '#7dd3fc') => progressCard(label, value, color);

const CollapsibleSection = ({ title, summary, children, color = '#67e8f9', icon = null }) => (
  <details style={{ ...panelStyle, overflow: 'hidden' }}>
    <summary
      style={{
        padding: '16px 18px',
        cursor: 'pointer',
        color: '#f8fafc',
        fontSize: 14,
        fontWeight: 850,
        listStyle: 'none',
      }}
    >
      <span style={{ color, marginRight: 10 }}>{icon || '◆'}</span>
      {title}
      {summary ? <span style={{ color: '#64748b', fontSize: 12, fontWeight: 500, marginLeft: 12 }}>{summary}</span> : null}
    </summary>
    <div style={{ padding: '0 18px 18px', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
      {children}
    </div>
  </details>
);

const summaryBlock = (title, color, items) => (
  <div style={{ ...panelStyle, padding: 16 }}>
    <div style={{ fontSize: 12, color, letterSpacing: 1.2, marginBottom: 12, fontWeight: 850 }}>
      {title}
    </div>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {(items || []).map((item) => (
        <div key={item} style={{ fontSize: 12, lineHeight: 1.65, color: '#cbd5e1', padding: '9px 10px', borderRadius: 10, background: 'rgba(0,0,0,0.18)' }}>
          {item}
        </div>
      ))}
      {(items || []).length === 0 ? <div style={{ color: '#64748b', fontSize: 12 }}>暂无数据</div> : null}
    </div>
  </div>
);

const capabilityList = (items, tone, selectedRoute, selectedRouteId, getRouteImpl) => (
  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12 }}>
    {(items || []).map((c, i) => (
      <div key={i} style={{ ...panelStyle, padding: 14 }}>
        <div style={{ fontWeight: 850, fontSize: 13, marginBottom: 8, color: tone === 'missing' ? '#fca5a5' : '#dcfce7' }}>{c.name}</div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
          <div style={badgeStyle(tone === 'missing' ? 'rgba(239,68,68,0.12)' : 'rgba(16,185,129,0.12)', tone === 'missing' ? '1px solid rgba(239,68,68,0.28)' : '1px solid rgba(16,185,129,0.28)', tone === 'missing' ? '#fca5a5' : '#86efac')}>
            {c.module_term || '-'}
          </div>
          <div style={badgeStyle('rgba(245,158,11,0.12)', '1px solid rgba(245,158,11,0.28)', '#fcd34d')}>
            {c.progress_pct || '-'}
          </div>
        </div>
        <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.55 }}>
          <div style={{ color: '#7dd3fc', marginBottom: 3 }}>当前实现（{selectedRoute?.title || selectedRouteId}）</div>
          {getRouteImpl(c)}
        </div>
      </div>
    ))}
  </div>
);

export const SystemStatusTab = ({
  consciousField,
  activeSystemProfile,
  statusData,
  selectedRoute,
  selectedRouteId,
  getRouteImpl,
}) => {
  const modelSummary = statusData?.model_summary || {};
  const runtimeLanguage = statusData?.runtime_language || {};
  const phaseaRuntime = statusData?.phasea_runtime || {};
  const researchOverview = statusData?.research_overview || {};

  return (
    <div style={{ animation: 'roadmapFade 0.5s ease-out', display: 'flex', flexDirection: 'column', gap: 18 }}>
      <section style={{ ...panelStyle, padding: 22, background: 'linear-gradient(135deg, rgba(16,185,129,0.12), rgba(15,23,42,0.44))' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 20, flexWrap: 'wrap', alignItems: 'flex-start' }}>
          <div style={{ maxWidth: 760 }}>
            <div style={{ color: consciousField?.glow_color === 'amber' ? '#ffaa00' : '#10b981', fontSize: 11, fontWeight: 900, letterSpacing: 1.4, marginBottom: 8 }}>
              系统状态摘要
            </div>
            <h2 style={{ color: '#fff', margin: '0 0 8px', fontSize: 25, fontWeight: 900 }}>
              {consciousField ? '实时意识场与系统状态' : '系统状态总览'}
            </h2>
            <div style={{ color: '#94a3b8', fontSize: 13, lineHeight: 1.75 }}>
              {consciousField
                ? `稳定度 ${(Number(consciousField.stability || 0) * 100).toFixed(1)}%，全局工作空间强度 ${Number(consciousField.gws_intensity || 0).toFixed(2)}。`
                : '聚合训练状态、语言运行时、能力闭合和测试验证。默认只显示核心指标，细节按需展开。'}
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
            <div style={badgeStyle('rgba(34,197,94,0.12)', '1px solid rgba(34,197,94,0.28)', '#86efac')}>
              语义链路：{runtimeLanguage.semantic_pipeline_ready ? '已接通' : '未接通'}
            </div>
            <div style={badgeStyle('rgba(59,130,246,0.12)', '1px solid rgba(59,130,246,0.28)', '#93c5fd')}>
              对话就绪：{runtimeLanguage.dialog_ready ? '是' : '否'}
            </div>
            <div style={badgeStyle('rgba(245,158,11,0.12)', '1px solid rgba(245,158,11,0.28)', '#fcd34d')}>
              开放域：{runtimeLanguage.open_domain_ready ? '是' : '否'}
            </div>
          </div>
        </div>
      </section>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12 }}>
        {progressCard('理论骨架', modelSummary.theory_skeleton_progress || '96% - 98%', '#00d2ff')}
        {progressCard('工程闭合', modelSummary.engineering_closure_progress || '95% - 97%', '#10b981')}
        {progressCard('本体破解', modelSummary.strict_brain_encoding_progress || '45% - 53%', '#f59e0b')}
        {progressCard('语言训练', modelSummary.language_training_progress || '-', '#38bdf8')}
      </div>

      {(activeSystemProfile?.metricCards || []).length > 0 ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12 }}>
          {(activeSystemProfile.metricCards || []).map((m, i) => progressCard(m.label, m.value, m.color, m.brain_ability))}
        </div>
      ) : null}

      <CollapsibleSection title="训练与运行时详情" summary="模型路径、PhaseA、语言运行时" icon={<Activity size={14} />} color="#7dd3fc">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12, marginTop: 16 }}>
          {progressCard('总参数量', modelSummary.total_parameters || '-', '#ffffff', `可训练参数：${modelSummary.trainable_parameters || '-'}`)}
          {progressCard('原型训练闭合', modelSummary.prototype_training_progress || '-', '#22c55e')}
          {progressCard('人类智能训练', modelSummary.human_level_training_progress || '-', '#f59e0b')}
          {progressCard('PhaseA 准备度', modelSummary.phasea_readiness_progress || '-', '#22c55e')}
          {runtimePill('语言等级', phaseaRuntime.language_level || '-', '#86efac')}
          {runtimePill('生成总评', Number(phaseaRuntime.generation_score || 0).toFixed(3), '#fca5a5')}
          {runtimePill('语义基准', Number(runtimeLanguage.semantic_benchmark_score || 0).toFixed(3))}
          {runtimePill('开放域总评', Number(runtimeLanguage.open_domain_assessment_score || 0).toFixed(3), '#fcd34d')}
        </div>
        <div style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.7, marginTop: 14, wordBreak: 'break-all' }}>
          当前在线模型：{modelSummary.current_model_file || '-'}
        </div>
      </CollapsibleSection>

      <CollapsibleSection title="研究概览" summary="DNN、脑编码、理论差距、模型测试" icon={<Search size={14} />} color="#a855f7">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12, marginTop: 16 }}>
          {summaryBlock('深度神经网络分析结果', '#a855f7', researchOverview.dnn_analysis_results || [])}
          {summaryBlock('大脑编码机制特性', '#00d2ff', researchOverview.brain_encoding_traits || [])}
          {summaryBlock('离真正破解还有多远', '#f59e0b', researchOverview.theory_gap || [])}
          {summaryBlock('新网络模型测试效果', '#22c55e', researchOverview.new_model_tests || [])}
        </div>
      </CollapsibleSection>

      <CollapsibleSection title="能力闭合清单" summary="已具备模块 / 未闭合模块" icon={<CheckCircle size={14} />} color="#22c55e">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16, marginTop: 16 }}>
          <div>
            <div style={{ color: '#86efac', fontSize: 12, fontWeight: 850, marginBottom: 10 }}>已具备模块</div>
            {capabilityList(statusData?.capabilities || [], 'done', selectedRoute, selectedRouteId, getRouteImpl)}
          </div>
          <div>
            <div style={{ color: '#fca5a5', fontSize: 12, fontWeight: 850, marginBottom: 10 }}>未闭合模块</div>
            {capabilityList(statusData?.missing_capabilities || [], 'missing', selectedRoute, selectedRouteId, getRouteImpl)}
          </div>
        </div>
      </CollapsibleSection>

      <CollapsibleSection title="参数与验证记录" summary="核心参数、通过验证、待验证" icon={<X size={14} />} color="#f59e0b">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 14, marginTop: 16 }}>
          {summaryBlock('核心参数', '#00d2ff', (statusData?.parameters || []).map((p) => `${p.name}：${p.value}。${p.desc || ''}`))}
          {summaryBlock('已通过验证', '#a855f7', (statusData?.passed_tests || []).map((test) => `${test.name || '-'}：${test.result || test.summary || '-'}`))}
          {summaryBlock('待验证问题', '#f59e0b', (statusData?.pending_tests || []).map((test) => `${test.name || '-'}：${test.target || test.summary || '-'}`))}
        </div>
      </CollapsibleSection>
    </div>
  );
};
