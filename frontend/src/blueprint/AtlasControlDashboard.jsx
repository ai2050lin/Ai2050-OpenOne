import { useEffect, useMemo, useState } from 'react';

const ATLAS_BASE = '/vis_data/pattern_family_atlas/v1';
const ATLAS_REPO = 'frontend/public/vis_data/pattern_family_atlas/v1';
const RESULT_REPO = 'tests/result/pattern_family_atlas/v1';
const MEMO_REPO = 'research/gpt5/docs/AGI_GPT5_MEMO.md';

const pct = (value) => `${Math.round(Number(value || 0) * 100)}%`;
const clamp01 = (value) => Math.max(0, Math.min(1, Number(value || 0)));

const metricRows = [
  {
    id: 'pattern_family_atlas',
    label: '语言模式族谱',
    hint: '九大语言模式族、样本库和路径 schema 的组织完成度。',
    method: '读取 progress.json 顶层 pattern_family_atlas；若缺失则回退到 progress.pattern_family_atlas 或 global_progress.pattern_family_atlas。',
    sourceKeys: ['pattern_family_atlas', 'progress.pattern_family_atlas', 'global_progress.pattern_family_atlas'],
    files: ['progress.json', 'families.jsonl', 'modes.jsonl', 'mode_family_case_bank_v3.jsonl', 'path_schema_rows.jsonl'],
  },
  {
    id: 'physical_path_atlas',
    label: '物理路径图谱',
    hint: '逐层 residual / readout 路径是否已经被稳定追踪。',
    method: '读取 progress.json 顶层 physical_path_atlas，主要对应 Phase267 之后的逐层路径追踪结果。',
    sourceKeys: ['physical_path_atlas'],
    files: ['progress.json', 'phase267_physical_path_rows.jsonl', 'phase267_layerwise_readout_rows.jsonl', 'phase267_family_path_signature_rows.jsonl'],
  },
  {
    id: 'component_path_atlas',
    label: '组件路径图谱',
    hint: 'attention、MLP、residual 等组件级路径的定位进度。',
    method: '读取 progress.json 顶层 component_path_atlas，主要来自 Phase268-269 的组件归因与必要性审计。',
    sourceKeys: ['component_path_atlas'],
    files: ['progress.json', 'phase268_component_physical_path_rows.jsonl', 'phase268_attention_contribution_rows.jsonl', 'phase268_mlp_contribution_rows.jsonl', 'phase269_mlp_necessity_rows.jsonl'],
  },
  {
    id: 'readout_competition_trace',
    label: '读出竞争追踪',
    hint: 'target、stop、continue、wrong token 的竞争是否可解释。',
    method: '读取 progress.json 顶层 readout_competition_trace，结合 Phase266 readout_rows 和后续竞争路径结果。',
    sourceKeys: ['readout_competition_trace', 'progress.readout_competition_trace'],
    files: ['progress.json', 'phase266_readout_rows.jsonl', 'phase267_continue_channel_trace_rows.jsonl'],
  },
  {
    id: 'stepwise_rollout_trace',
    label: 'Rollout 追踪',
    hint: '生成后续、漂移、停止和协议延续是否被记录。',
    method: '读取 progress.json 顶层 stepwise_rollout_trace，结合 Phase266/269 rollout probe/effect rows。',
    sourceKeys: ['stepwise_rollout_trace', 'progress.stepwise_rollout_trace'],
    files: ['progress.json', 'phase266_rollout_probe_rows.jsonl', 'phase269_rollout_effect_rows.jsonl'],
  },
  {
    id: 'causal_closure',
    label: '机制闭合',
    hint: '因果必要性、竞争胜出、无副作用和自然闭合的综合进度。',
    method: '读取 progress.json 顶层 causal_closure。该值是终检项，不等于任一局部 patch 成功率。',
    sourceKeys: ['causal_closure', 'progress.causal_closure'],
    files: ['progress.json', 'phase269_causal_effect_rows.jsonl', 'phase269_mlp_necessity_rows.jsonl'],
  },
];

const routeCards = [
  {
    id: 'family',
    title: '语言模式族谱路线',
    phase: 'Phase264-265',
    status: '框架已成型',
    evidence: '九大模式族、case bank、path schema 已固定。',
    gap: '样本仍需要真实失败分布和跨模型质量校准。',
    files: ['families.jsonl', 'modes.jsonl', 'mode_family_case_bank_v3.jsonl', 'path_schema_rows.jsonl', 'state_factor_design_rows.jsonl'],
    detail: [
      'Phase264 把已有结果从指标表重组为路径表。',
      'Phase265 把 output_protocol 局部样本扩展为九大语言模式族统一样本矩阵。',
      '当前价值是形成可复用数据结构，不是新增闭合证据。',
    ],
  },
  {
    id: 'behavior',
    title: '行为与读出竞争路线',
    phase: 'Phase266',
    status: '三模型基线完成',
    evidence: 'continue 通道在多语言族中强势，stop-vs-continuation 是主要瓶颈。',
    gap: 'answer_correct 仍是代理指标，需要更强语义判分。',
    files: ['phase266_behavior_rows.jsonl', 'phase266_readout_rows.jsonl', 'phase266_rollout_probe_rows.jsonl', 'phase266_quality_calibration_rows.jsonl'],
    detail: [
      'Phase266 对九大语言族做三模型 behavior/readout baseline。',
      '主要作用是挑出高价值失败路径，而不是证明机制闭合。',
      '关键现象是 continue winner 频繁压过 stop 或目标协议。',
    ],
  },
  {
    id: 'physical',
    title: '物理路径追踪路线',
    phase: 'Phase267',
    status: '逐层路径已启动',
    evidence: '高风险样本中 continue path 很早可读出，不是后期偶然失败。',
    gap: '仍是 readout probing，不等于因果路径。',
    files: ['phase267_physical_path_rows.jsonl', 'phase267_layerwise_readout_rows.jsonl', 'phase267_continue_channel_trace_rows.jsonl', 'phase267_family_path_signature_rows.jsonl'],
    detail: [
      'Phase267 把 Phase266 的 continue winner 拆成逐层 residual readout trace。',
      '它回答路径在哪里出现，但还没有回答哪个组件因果必要。',
      '当前应与组件归因和干预审计联动查看。',
    ],
  },
  {
    id: 'component',
    title: '组件路径路线',
    phase: 'Phase268',
    status: 'MLP 主导迹象',
    evidence: 'MLP 对 continue-stop margin 的自然正向贡献强于 attention。',
    gap: 'observational attribution 不能直接升级为 causal closure。',
    files: ['phase268_component_physical_path_rows.jsonl', 'phase268_attention_contribution_rows.jsonl', 'phase268_mlp_contribution_rows.jsonl', 'phase268_residual_accumulation_rows.jsonl', 'phase268_component_summary_rows.jsonl'],
    detail: [
      'Phase268 将 residual layer delta 拆为 attention / MLP 贡献。',
      '当前结论是观测归因：MLP 更强地增加 continue-stop margin。',
      '必须接 Phase269 的 necessity audit 才能判断因果必要性。',
    ],
  },
  {
    id: 'causal',
    title: '因果与补偿路线',
    phase: 'Phase269',
    status: '必要性混合结果',
    evidence: 'qwen3 / DS7B 支持 MLP 必要性，GLM4 暴露补偿路径。',
    gap: '需要跨层 writer set、mean replacement、random same norm 对照。',
    files: ['phase269_mlp_necessity_rows.jsonl', 'phase269_causal_effect_rows.jsonl', 'phase269_rollout_effect_rows.jsonl'],
    detail: [
      'Phase269 检查 strongest MLP 是否真的必要。',
      'qwen3 和 DS7B 出现必要性迹象，GLM4 出现抑制后 margin 反升。',
      '因此 CompensationPath 必须进入机制公式。',
    ],
  },
  {
    id: 'closure',
    title: '闭合验证路线',
    phase: 'Phase260-269',
    status: '后置终检',
    evidence: 'EOS 增强不等于闭合，stop 必须压过结构化 continuation。',
    gap: 'competition、rollout、clean side effect 和 compensation 尚未贯通。',
    files: ['progress.json', 'phase269_causal_effect_rows.jsonl', 'phase269_rollout_effect_rows.jsonl'],
    detail: [
      '闭合不是单点 patch 成功，而是路径、因果、竞争、rollout、副作用和补偿同时过关。',
      '当前 causal_closure 仍低，说明闭合应该作为终检项。',
      '下一步应先补物理路径和补偿路径，再做 clean closure。',
    ],
  },
];

const evidenceLadder = [
  { id: 'case_design', label: '样本设计', status: '强', value: 0.9, files: ['mode_family_case_bank_v3.jsonl', 'path_schema_rows.jsonl'] },
  { id: 'behavior', label: '行为验证', status: '中强', value: 0.68, files: ['phase266_behavior_rows.jsonl', 'phase266_quality_calibration_rows.jsonl'] },
  { id: 'readout', label: '读出竞争', status: '强', value: 0.8, files: ['phase266_readout_rows.jsonl', 'phase267_continue_channel_trace_rows.jsonl'] },
  { id: 'layer_path', label: '层路径追踪', status: '中', value: 0.45, files: ['phase267_layerwise_readout_rows.jsonl'] },
  { id: 'component', label: '组件归因', status: '早期', value: 0.2, files: ['phase268_component_physical_path_rows.jsonl'] },
  { id: 'causal', label: '因果干预', status: '早期', value: 0.18, files: ['phase269_causal_effect_rows.jsonl'] },
  { id: 'rollout', label: 'Rollout 验证', status: '中早期', value: 0.45, files: ['phase266_rollout_probe_rows.jsonl', 'phase269_rollout_effect_rows.jsonl'] },
  { id: 'closure', label: '机制闭合', status: '早期', value: 0.18, files: ['progress.json', 'phase269_mlp_necessity_rows.jsonl'] },
];

const formulaCards = [
  {
    id: 'language_mechanism',
    title: '语言机制路径族',
    body: 'LanguageMechanism = sum_i alpha_i(x,t) P_i(x,t)',
    note: '当前作为图谱组织公式，而不是最终闭合公式。',
    definitions: ['alpha_i 是上下文条件下第 i 条模式路径的权重。', 'P_i 是触发、状态、组件、读出、rollout 和闭合组成的物理路径。'],
    source: 'AGI_GPT5_MEMO.md Phase264-265',
  },
  {
    id: 'continue_path',
    title: '继续路径结构',
    body: 'ContinuePath = B_embed + AttentionRoute + MLPWrite + CompensationPath + ReadoutCompetition',
    note: 'Phase269 后必须加入 CompensationPath，否则解释不了 GLM4 反向结果。',
    definitions: ['B_embed 是输入/模板基底。', 'MLPWrite 是 MLP 写入贡献。', 'CompensationPath 是干预后仍能维持或反向增强目标 margin 的补偿通道。'],
    source: 'AGI_GPT5_MEMO.md Phase268-269',
  },
  {
    id: 'hybrid_distance',
    title: '综合闭合距离',
    body: 'D_hybrid = sum_i lambda_i d_i',
    note: 'd_i 包括 path、causal、competition、rollout、clean、compensation。',
    definitions: ['d_path 衡量路径偏离稳定族中心的距离。', 'd_causal 衡量干预后目标 margin 是否按预期下降。', 'd_competition 衡量 stop/target 是否赢过 continue/blocker。'],
    source: '当前图谱总控公式草案，参考 Phase260-269',
  },
];

const nextActions = [
  {
    id: 'schema',
    text: '冻结 Atlas v2 schema，避免后续 Phase 临时新增互不兼容字段。',
    files: ['schema.json', 'manifest.json', 'client_index.json'],
    source: '当前 v1 文件多且字段分散，下一版需要稳定主表和 detail 表。',
  },
  {
    id: 'phase270',
    text: '把 Phase270 升级为补偿路径与跨层 writer set 批处理，而不是单点测试。',
    files: ['phase269_mlp_necessity_rows.jsonl', 'phase269_causal_effect_rows.jsonl'],
    source: 'Phase269 的 GLM4 反向结果说明单层 MLP 不是完整机制。',
  },
  {
    id: 'client',
    text: '客户端采用 summary/index 首屏加载，case detail 按需读取，避免大 JSONL 卡顿。',
    files: ['client_index.json', 'observations.jsonl', 'metrics.jsonl'],
    source: 'observations.jsonl 很大，不适合总控页首屏加载。',
  },
  {
    id: 'closure',
    text: '闭合继续作为终检：先路径、再因果、再竞争、再 rollout、最后 clean closure。',
    files: ['progress.json', 'phase266_readout_rows.jsonl', 'phase269_rollout_effect_rows.jsonl'],
    source: '当前 causal_closure 仍低，闭合不能前置。',
  },
];

const cardStyle = {
  border: '1px solid rgba(148, 163, 184, 0.16)',
  borderRadius: 8,
  background: 'rgba(15, 23, 42, 0.72)',
  boxShadow: '0 14px 32px rgba(0, 0, 0, 0.22)',
};

function readProgressValue(progress, id) {
  return progress?.[id] ?? progress?.progress?.[id] ?? progress?.global_progress?.[id] ?? 0;
}

function readProgressSource(progress, id) {
  if (progress && Object.prototype.hasOwnProperty.call(progress, id)) return `progress.json 顶层字段 ${id}`;
  if (progress?.progress && Object.prototype.hasOwnProperty.call(progress.progress, id)) return `progress.json progress.${id}`;
  if (progress?.global_progress && Object.prototype.hasOwnProperty.call(progress.global_progress, id)) return `progress.json global_progress.${id}`;
  return `progress.json 未找到 ${id}，显示默认值 0`;
}

function fileLine(name) {
  return `${ATLAS_REPO}/${name}`;
}

function ProgressBar({ value, color = '#22d3ee' }) {
  const width = pct(clamp01(value));
  return (
    <div style={{ height: 6, borderRadius: 999, background: 'rgba(148, 163, 184, 0.16)', overflow: 'hidden' }}>
      <div style={{ width, height: '100%', borderRadius: 999, background: color }} />
    </div>
  );
}

function ClickableCard({ children, active, onClick, style }) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        ...cardStyle,
        ...style,
        width: '100%',
        textAlign: 'left',
        cursor: 'pointer',
        color: 'inherit',
        outline: 'none',
        border: active ? '1px solid rgba(103, 232, 249, 0.55)' : cardStyle.border,
        boxShadow: active ? '0 0 0 1px rgba(103,232,249,0.22), 0 14px 32px rgba(0,0,0,0.25)' : cardStyle.boxShadow,
      }}
    >
      {children}
    </button>
  );
}

function MetricCard({ row, value, active, onClick }) {
  const n = clamp01(value);
  const color = n >= 0.7 ? '#22c55e' : n >= 0.35 ? '#f59e0b' : '#fb7185';
  return (
    <ClickableCard active={active} onClick={onClick} style={{ padding: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 12 }}>
        <div style={{ color: '#e5eefb', fontWeight: 700, fontSize: 13 }}>{row.label}</div>
        <div style={{ color, fontWeight: 800, fontSize: 18 }}>{pct(n)}</div>
      </div>
      <div style={{ color: '#94a3b8', fontSize: 11, lineHeight: 1.55, margin: '8px 0 10px' }}>{row.hint}</div>
      <ProgressBar value={n} color={color} />
      <div style={{ color: '#64748b', fontSize: 10, marginTop: 8 }}>点击查看数据来源</div>
    </ClickableCard>
  );
}

function RouteCard({ route, active, onClick }) {
  return (
    <ClickableCard active={active} onClick={onClick} style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, alignItems: 'center' }}>
        <div style={{ color: '#f8fafc', fontSize: 14, fontWeight: 800 }}>{route.title}</div>
        <span style={{
          color: '#67e8f9',
          border: '1px solid rgba(103, 232, 249, 0.22)',
          background: 'rgba(8, 145, 178, 0.12)',
          borderRadius: 999,
          padding: '3px 8px',
          fontSize: 10,
          whiteSpace: 'nowrap',
        }}>
          {route.phase}
        </span>
      </div>
      <div style={{ color: '#a7f3d0', fontSize: 12, fontWeight: 700 }}>{route.status}</div>
      <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.7 }}>{route.evidence}</div>
      <div style={{ color: '#fbbf24', fontSize: 11, lineHeight: 1.6, borderTop: '1px solid rgba(148,163,184,0.12)', paddingTop: 9 }}>
        缺口：{route.gap}
      </div>
    </ClickableCard>
  );
}

function DetailPanel({ detail }) {
  if (!detail) return null;
  return (
    <section style={{ ...cardStyle, padding: 18, position: 'sticky', top: 0 }}>
      <div style={{ color: '#67e8f9', fontSize: 11, fontWeight: 800, letterSpacing: 1.2, textTransform: 'uppercase' }}>
        Detail / Source
      </div>
      <h3 style={{ color: '#f8fafc', fontSize: 18, lineHeight: 1.35, margin: '8px 0 6px' }}>{detail.title}</h3>
      {detail.subtitle && <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.65 }}>{detail.subtitle}</div>}

      {detail.value != null && (
        <div style={{ marginTop: 14 }}>
          <div style={{ color: '#94a3b8', fontSize: 11 }}>当前数据</div>
          <div style={{ color: '#fff', fontSize: 28, fontWeight: 900 }}>{detail.value}</div>
          {detail.valueHint && <div style={{ color: '#94a3b8', fontSize: 11, lineHeight: 1.55 }}>{detail.valueHint}</div>}
        </div>
      )}

      {(detail.items || []).length > 0 && (
        <div style={{ marginTop: 14 }}>
          <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 8 }}>详细说明</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {detail.items.map((item) => (
              <div key={item} style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.65, padding: '8px 10px', borderRadius: 8, background: 'rgba(2, 6, 23, 0.42)' }}>
                {item}
              </div>
            ))}
          </div>
        </div>
      )}

      {(detail.files || []).length > 0 && (
        <div style={{ marginTop: 14 }}>
          <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 8 }}>来源文件</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {detail.files.map((file) => (
              <code key={file} style={{ color: '#bae6fd', fontSize: 11, lineHeight: 1.55, whiteSpace: 'normal' }}>
                {fileLine(file)}
              </code>
            ))}
          </div>
        </div>
      )}

      {(detail.extraSources || []).length > 0 && (
        <div style={{ marginTop: 14 }}>
          <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 8 }}>研究来源</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {detail.extraSources.map((source) => (
              <code key={source} style={{ color: '#d8b4fe', fontSize: 11, lineHeight: 1.55, whiteSpace: 'normal' }}>
                {source}
              </code>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

export function AtlasControlDashboard() {
  const [progress, setProgress] = useState(null);
  const [manifest, setManifest] = useState(null);
  const [clientIndex, setClientIndex] = useState(null);
  const [error, setError] = useState('');
  const [activeDetail, setActiveDetail] = useState({ type: 'overview' });

  useEffect(() => {
    let mounted = true;
    Promise.all([
      fetch(`${ATLAS_BASE}/progress.json`, { cache: 'no-store' }).then((res) => {
        if (!res.ok) throw new Error(`progress.json ${res.status}`);
        return res.json();
      }),
      fetch(`${ATLAS_BASE}/manifest.json`, { cache: 'no-store' }).then((res) => {
        if (!res.ok) throw new Error(`manifest.json ${res.status}`);
        return res.json();
      }),
      fetch(`${ATLAS_BASE}/client_index.json`, { cache: 'no-store' }).then((res) => (res.ok ? res.json() : null)),
    ])
      .then(([nextProgress, nextManifest, nextClientIndex]) => {
        if (!mounted) return;
        setProgress(nextProgress);
        setManifest(nextManifest);
        setClientIndex(nextClientIndex);
        setError('');
      })
      .catch((err) => {
        if (!mounted) return;
        setError(err?.message || '图谱进度数据读取失败');
      });
    return () => {
      mounted = false;
    };
  }, []);

  const currentSummary = useMemo(() => {
    const pattern = readProgressValue(progress, 'pattern_family_atlas');
    const physical = readProgressValue(progress, 'physical_path_atlas');
    const component = readProgressValue(progress, 'component_path_atlas');
    const closure = readProgressValue(progress, 'causal_closure');
    return {
      headline: pattern >= 0.85
        ? '语言模式族谱基本成型，当前主瓶颈已经转向物理路径、组件因果和补偿路径。'
        : '语言模式族谱仍在成型中，当前应优先补齐样本和路径 schema。',
      overall: (pattern * 0.25) + (physical * 0.25) + (component * 0.2) + (closure * 0.3),
    };
  }, [progress]);

  const detail = useMemo(() => {
    if (activeDetail.type === 'metric') {
      const row = metricRows.find((item) => item.id === activeDetail.id);
      if (!row) return null;
      const value = readProgressValue(progress, row.id);
      return {
        title: row.label,
        subtitle: row.hint,
        value: pct(value),
        valueHint: readProgressSource(progress, row.id),
        items: [row.method, `可用字段：${row.sourceKeys.join(' / ')}`],
        files: row.files,
        extraSources: [MEMO_REPO],
      };
    }
    if (activeDetail.type === 'route') {
      const route = routeCards.find((item) => item.id === activeDetail.id);
      if (!route) return null;
      return {
        title: route.title,
        subtitle: `${route.phase} · ${route.status}`,
        value: route.status,
        valueHint: route.evidence,
        items: [...route.detail, `当前缺口：${route.gap}`],
        files: route.files,
        extraSources: [`${MEMO_REPO} · ${route.phase}`],
      };
    }
    if (activeDetail.type === 'evidence') {
      const row = evidenceLadder.find((item) => item.id === activeDetail.id);
      if (!row) return null;
      return {
        title: row.label,
        subtitle: '证据阶梯用于区分“有数据记录”“有路径信号”“有因果支持”和“真正闭合”。',
        value: `${row.status} · ${pct(row.value)}`,
        valueHint: '该值是总控页证据等级，不等于单个模型准确率。',
        items: ['越往后越接近机制闭合，但所需证据也更严格。', '当前应避免把 readout probing 或 observational attribution 误判为 causal closure。'],
        files: row.files,
        extraSources: [MEMO_REPO],
      };
    }
    if (activeDetail.type === 'formula') {
      const formula = formulaCards.find((item) => item.id === activeDetail.id);
      if (!formula) return null;
      return {
        title: formula.title,
        subtitle: formula.note,
        value: formula.body,
        valueHint: formula.source,
        items: formula.definitions,
        files: ['progress.json', 'summary.md'],
        extraSources: [MEMO_REPO],
      };
    }
    if (activeDetail.type === 'action') {
      const action = nextActions.find((item) => item.id === activeDetail.id);
      if (!action) return null;
      return {
        title: '下一步总控动作',
        subtitle: action.text,
        value: action.id,
        valueHint: action.source,
        items: ['这不是展示项，而是下一阶段工程任务入口。', '后续可以把该动作挂接到脚本运行、报告生成或数据同步流程。'],
        files: action.files,
        extraSources: [RESULT_REPO, MEMO_REPO],
      };
    }
    return {
      title: '图谱系统总览',
      subtitle: currentSummary.headline,
      value: pct(currentSummary.overall),
      valueHint: '综合推进度 = 0.25*语言族谱 + 0.25*物理路径 + 0.20*组件路径 + 0.30*闭合。',
      items: [
        `最新阶段：${progress?.last_phase || progress?.latest_phase || manifest?.phase || '加载中'}`,
        `模型：${(manifest?.models || ['qwen3', 'glm4', 'deepseek7b']).join(' / ')}`,
        `客户端视图数：${clientIndex?.views?.length || 0}`,
      ],
      files: ['progress.json', 'manifest.json', 'client_index.json'],
      extraSources: [MEMO_REPO],
    };
  }, [activeDetail, clientIndex, currentSummary, manifest, progress]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18, maxWidth: 1320 }}>
      <section
        role="button"
        tabIndex={0}
        onClick={() => setActiveDetail({ type: 'overview' })}
        style={{
          ...cardStyle,
          padding: 22,
          background: 'linear-gradient(135deg, rgba(8, 47, 73, 0.92), rgba(15, 23, 42, 0.84))',
          cursor: 'pointer',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 20, flexWrap: 'wrap' }}>
          <div>
            <div style={{ color: '#67e8f9', fontSize: 11, fontWeight: 800, letterSpacing: 1.4, textTransform: 'uppercase' }}>
              Pattern Family Atlas Control
            </div>
            <h2 style={{ margin: '8px 0 8px', color: '#f8fafc', fontSize: 28, lineHeight: 1.2 }}>
              图谱系统总控页
            </h2>
            <div style={{ color: '#cbd5e1', fontSize: 13, lineHeight: 1.75, maxWidth: 760 }}>
              {currentSummary.headline}
            </div>
            <div style={{ color: '#64748b', fontSize: 10, marginTop: 8 }}>点击查看总览计算口径和来源</div>
          </div>
          <div style={{ minWidth: 180 }}>
            <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 6 }}>综合推进度</div>
            <div style={{ color: '#fff', fontSize: 34, fontWeight: 900 }}>{pct(currentSummary.overall)}</div>
            <ProgressBar value={currentSummary.overall} color="#67e8f9" />
          </div>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 18, color: '#94a3b8', fontSize: 11 }}>
          <span>最新阶段：{progress?.last_phase || progress?.latest_phase || manifest?.phase || '加载中'}</span>
          <span>模型：{(manifest?.models || ['qwen3', 'glm4', 'deepseek7b']).join(' / ')}</span>
          <span>当前优先级：{progress?.current_priority || '补齐语言族物理路径图谱'}</span>
        </div>
        {error && (
          <div style={{ marginTop: 14, color: '#fecaca', fontSize: 12 }}>
            {error}
          </div>
        )}
      </section>

      <section style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(320px, 420px)', gap: 16, alignItems: 'start' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 12 }}>
            {metricRows.map((row) => (
              <MetricCard
                key={row.id}
                row={row}
                value={readProgressValue(progress, row.id)}
                active={activeDetail.type === 'metric' && activeDetail.id === row.id}
                onClick={() => setActiveDetail({ type: 'metric', id: row.id })}
              />
            ))}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ color: '#f8fafc', fontSize: 15, fontWeight: 800 }}>关键路线成果</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12 }}>
              {routeCards.map((route) => (
                <RouteCard
                  key={route.id}
                  route={route}
                  active={activeDetail.type === 'route' && activeDetail.id === route.id}
                  onClick={() => setActiveDetail({ type: 'route', id: route.id })}
                />
              ))}
            </div>
          </div>

          <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12 }}>
            <div style={{ ...cardStyle, padding: 16 }}>
              <div style={{ color: '#f8fafc', fontSize: 15, fontWeight: 800, marginBottom: 12 }}>证据阶梯</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {evidenceLadder.map((row) => (
                  <button
                    key={row.id}
                    type="button"
                    onClick={() => setActiveDetail({ type: 'evidence', id: row.id })}
                    style={{
                      background: activeDetail.type === 'evidence' && activeDetail.id === row.id ? 'rgba(8,145,178,0.16)' : 'transparent',
                      border: '1px solid rgba(148,163,184,0.1)',
                      borderRadius: 8,
                      padding: 8,
                      cursor: 'pointer',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 5 }}>
                      <span style={{ color: '#cbd5e1' }}>{row.label}</span>
                      <span style={{ color: '#67e8f9' }}>{row.status}</span>
                    </div>
                    <ProgressBar value={row.value} color={row.value >= 0.7 ? '#22c55e' : row.value >= 0.35 ? '#f59e0b' : '#fb7185'} />
                  </button>
                ))}
              </div>
            </div>

            <div style={{ ...cardStyle, padding: 16 }}>
              <div style={{ color: '#f8fafc', fontSize: 15, fontWeight: 800, marginBottom: 12 }}>核心公式</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {formulaCards.map((formula) => (
                  <button
                    key={formula.id}
                    type="button"
                    onClick={() => setActiveDetail({ type: 'formula', id: formula.id })}
                    style={{
                      textAlign: 'left',
                      border: activeDetail.type === 'formula' && activeDetail.id === formula.id ? '1px solid rgba(103,232,249,0.42)' : '1px solid rgba(148,163,184,0.12)',
                      borderRadius: 8,
                      background: 'rgba(2,6,23,0.38)',
                      padding: 10,
                      cursor: 'pointer',
                    }}
                  >
                    <div style={{ color: '#bae6fd', fontSize: 12, fontWeight: 800 }}>{formula.title}</div>
                    <code style={{ display: 'block', color: '#f8fafc', fontSize: 11, lineHeight: 1.7, margin: '5px 0', whiteSpace: 'normal' }}>
                      {formula.body}
                    </code>
                    <div style={{ color: '#94a3b8', fontSize: 11, lineHeight: 1.55 }}>{formula.note}</div>
                  </button>
                ))}
              </div>
            </div>
          </section>

          <section style={{ ...cardStyle, padding: 16 }}>
            <div style={{ color: '#f8fafc', fontSize: 15, fontWeight: 800, marginBottom: 10 }}>下一步总控动作</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10 }}>
              {nextActions.map((item, idx) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setActiveDetail({ type: 'action', id: item.id })}
                  style={{
                    border: activeDetail.type === 'action' && activeDetail.id === item.id ? '1px solid rgba(103,232,249,0.45)' : '1px solid rgba(148,163,184,0.12)',
                    borderRadius: 8,
                    padding: 12,
                    color: '#cbd5e1',
                    fontSize: 12,
                    lineHeight: 1.65,
                    background: 'rgba(2, 6, 23, 0.42)',
                    textAlign: 'left',
                    cursor: 'pointer',
                  }}
                >
                  <span style={{ color: '#67e8f9', fontWeight: 800 }}>0{idx + 1}. </span>{item.text}
                </button>
              ))}
            </div>
          </section>
        </div>

        <DetailPanel detail={detail} />
      </section>
    </div>
  );
}
