import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Cpu,
  Database,
  Layers3,
  Network,
  ShieldCheck,
} from 'lucide-react';
import { useState } from 'react';

import { MODEL_SYSTEM_SOURCES, summarizeModelEvidence, useModelSystemEvidence } from './useModelSystemEvidence';

const panelStyle = {
  border: '1px solid rgba(148, 163, 184, 0.14)',
  borderRadius: 8,
  background: 'rgba(15, 23, 42, 0.36)',
};

const MODEL_NAMES = {
  qwen3: 'Qwen3-4B',
  glm4: 'GLM4-9B-Chat',
  deepseek7b: 'DeepSeek-R1-Distill-Qwen-7B',
};

const MODEL_BOUNDARIES = {
  qwen3: {
    state: '路径更分布式',
    detail: 'Phase338 的冻结早期 source residual 粗块没有通过 heldout + private 完整模型门；错误深度干预也会损伤答案，暂时不能定位为特异因果块。',
  },
  glm4: {
    state: '模型特异粗块候选',
    detail: 'GLM4 的早期 source MLP 粗块在材料绑定和部分多词元短语复制中出现位置特异必要性，但 Phase344 已关闭“一般复制机制”外推，只保留模型与任务特异边界。',
  },
  deepseek7b: {
    state: '路径更分布式',
    detail: 'Phase338 的冻结早期 source residual 粗块没有通过 heldout + private 完整模型门；结果可能包含分布式路径、冗余和协议敏感性。',
  },
};

const RECENT_STAGES = [
  {
    id: 'protocol',
    phase: 'Phase 337',
    title: '协议资格门',
    value: '7/9 模型-接口合格',
    summary: '答案对齐接口使三个模型在冻结材料关系任务上全部达到 12/12，建立了共同可审计分母。',
    detail: '该结果修复了“模型尚在思考”和“模型不知道答案”的混淆，但答案对齐接口人为跳过思考段，不能视为完整自然聊天机制。',
  },
  {
    id: 'coarse_block',
    phase: 'Phase 338',
    title: '分层粗块因果筛选',
    value: '完整模型门 1/3',
    summary: '三个模型都指向早期 source 区域，但只有 GLM4 的 MLP 粗块通过完整模型门。',
    detail: '功能位置出现弱收敛，物理组件并不一致。Qwen3 和 DeepSeek7B 对错误深度同样敏感，因此跨模型粗块门为 0。',
  },
  {
    id: 'measurement',
    phase: 'Phase 340-342',
    title: '测量路径不变性',
    value: '仅 2 条执行路径稳定',
    summary: '修复 GLM4 批处理异常后，研究进一步发现批量、缓存和执行后端并不天然语义等价。',
    detail: '这属于测量执行层，而不是语言神经机制。后续实验必须冻结执行模式并先通过不变性门，否则同一提示的差异可能来自工具链。',
  },
  {
    id: 'replication',
    phase: 'Phase 343-344',
    title: '复制边界审计',
    value: '跨模型候选 0',
    summary: '全新基线中 38/48 模型任务单元合格，但十三任务粗块审计没有形成跨模型一般复制机制。',
    detail: 'GLM4 候选只保留为多词元短语复制的模型特异、任务特异必要性效应；单神经元因果仍为 0/72。',
  },
];

const formatNumber = (value) => new Intl.NumberFormat('zh-CN').format(Number(value || 0));

function StatusMetric({ label, value, tone = '#e2e8f0', note }) {
  return (
    <div style={{ padding: '12px 14px', borderTop: `2px solid ${tone}`, background: 'rgba(15,23,42,0.28)', minHeight: 78 }}>
      <div style={{ color: '#94a3b8', fontSize: 10 }}>{label}</div>
      <div style={{ color: tone, fontSize: 20, lineHeight: 1.15, fontWeight: 900, fontFamily: 'monospace', marginTop: 5 }}>{value}</div>
      {note && <div style={{ color: '#64748b', fontSize: 10, marginTop: 4 }}>{note}</div>}
    </div>
  );
}

function ModelDetail({ snapshot, summary }) {
  const boundary = MODEL_BOUNDARIES[snapshot.model] || {};
  const rows = [
    ['Architecture', snapshot.architecture],
    ['Layer / hidden / MLP', `${snapshot.num_hidden_layers} / ${snapshot.hidden_size} / ${snapshot.intermediate_size}`],
    ['Attention / KV Head', `${snapshot.num_attention_heads} / ${snapshot.num_key_value_heads}`],
    ['Vocabulary', formatNumber(snapshot.vocab_size)],
    ['模式族覆盖', `${summary.familyCount}/9`],
    ['组件事件', formatNumber(summary.componentEvents)],
    ['路径签名', formatNumber(summary.pathSignatures)],
    ['物理候选', formatNumber(summary.unitCandidates)],
    ['局部读出候选', formatNumber(summary.localReadoutCandidates)],
    ['局部传播通过', formatNumber(summary.localPropagationPasses)],
    ['单神经元因果', formatNumber(summary.singleUnitCausal)],
    ['完整自然链', formatNumber(summary.completeChains)],
  ];
  return (
    <div style={{ display: 'grid', gap: 14 }}>
      <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.7 }}>{boundary.detail}</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: 8 }}>
        {rows.map(([label, value]) => (
          <div key={label} style={{ display: 'flex', justifyContent: 'space-between', gap: 12, padding: '8px 10px', borderBottom: '1px solid rgba(148,163,184,0.12)' }}>
            <span style={{ color: '#94a3b8', fontSize: 11 }}>{label}</span>
            <strong style={{ color: String(value) === '0' ? '#fda4af' : '#dbeafe', fontSize: 11, textAlign: 'right', overflowWrap: 'anywhere' }}>{value}</strong>
          </div>
        ))}
      </div>
      <div style={{ color: '#64748b', fontSize: 10, overflowWrap: 'anywhere' }}>模型修订：{snapshot.model_revision}</div>
      <a href={MODEL_SYSTEM_SOURCES.models} target="_blank" rel="noreferrer" style={{ color: '#7dd3fc', fontSize: 11 }}>查看真实模型注册表</a>
    </div>
  );
}

export const ResearchProgressTab = () => {
  const { data, errors, loading } = useModelSystemEvidence();
  const [activeModel, setActiveModel] = useState('');
  const [activeStage, setActiveStage] = useState('');
  const atlasMetrics = data.atlas?.metrics || {};
  const models = data.models?.models || [];

  const modelRows = models.map((snapshot) => ({
    snapshot,
    summary: summarizeModelEvidence(data.atlas, snapshot.model),
  }));

  const selectedModel = modelRows.find((row) => row.snapshot.model === activeModel);
  const selectedStage = RECENT_STAGES.find((stage) => stage.id === activeStage);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <section style={{ padding: '22px 24px', borderBottom: '1px solid rgba(148,163,184,0.14)', background: 'rgba(15,23,42,0.18)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 18 }}>
          <div style={{ maxWidth: 820 }}>
            <div style={{ color: '#67e8f9', fontSize: 10, fontWeight: 900, letterSpacing: 1.5 }}>MODEL & MECHANISM STATUS</div>
            <h2 style={{ color: '#f8fafc', fontSize: 26, margin: '6px 0 7px' }}>三模型逆向分析状态</h2>
            <div style={{ color: '#94a3b8', fontSize: 13, lineHeight: 1.7 }}>
              三模型运行、九族图谱和大规模组件观测已经建立；最新科学结果仍停留在模型特异粗块与局部传播候选，尚未形成跨模型因果规则、单神经元必要性或完整自然闭合。
            </div>
          </div>
          <div style={{ color: '#fbbf24', fontSize: 11, fontWeight: 800, padding: '7px 10px', border: '1px solid rgba(245,158,11,0.28)', background: 'rgba(245,158,11,0.08)' }}>
            最新客户端证据包 Phase {data.atlas?.phase || '-'} · 最新科学审计 Phase 344
          </div>
        </div>
        {loading && <div style={{ color: '#7dd3fc', fontSize: 11, marginTop: 12 }}>正在读取模型与图谱证据…</div>}
        {errors.length > 0 && <div style={{ color: '#fda4af', fontSize: 11, marginTop: 12 }}>{errors.length} 个数据源读取失败，页面显示部分状态。</div>}
      </section>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(145px, 1fr))', gap: 9 }}>
        <StatusMetric label="真实模型注册" value={`${models.length}/3`} tone="#34d399" note="配置哈希可追溯" />
        <StatusMetric label="模式族物理覆盖" value={`${atlasMetrics.mapped_family_count || 0}/${atlasMetrics.family_count || 0}`} tone="#22d3ee" note="覆盖不等于闭合" />
        <StatusMetric label="注册机制" value={formatNumber(atlasMetrics.registered_mechanism_count)} tone="#60a5fa" note="三模型统一分母" />
        <StatusMetric label="Prompt-model 案例" value={formatNumber(atlasMetrics.prompt_model_case_count)} tone="#a78bfa" />
        <StatusMetric label="组件事件" value={formatNumber(atlasMetrics.component_event_count)} tone="#f59e0b" />
        <StatusMetric label="跨模型因果规则" value="0" tone="#fda4af" note="严格门未通过" />
        <StatusMetric label="单神经元因果" value={formatNumber(atlasMetrics.single_unit_causal_count)} tone="#fda4af" />
        <StatusMetric label="完整自然链" value={formatNumber(atlasMetrics.full_natural_chain_pass_count)} tone="#fda4af" />
      </div>

      <section style={{ display: 'grid', gap: 12 }}>
        <div>
          <div style={{ color: '#f8fafc', fontSize: 17, fontWeight: 800 }}>模型状态</div>
          <div style={{ color: '#94a3b8', fontSize: 12, marginTop: 4 }}>点击模型查看真实架构、图谱分母、当前物理候选和严格边界。</div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 10 }}>
          {modelRows.map(({ snapshot, summary }) => {
            const active = activeModel === snapshot.model;
            const boundary = MODEL_BOUNDARIES[snapshot.model] || {};
            return (
              <button
                key={snapshot.model}
                type="button"
                aria-expanded={active}
                onClick={() => setActiveModel(active ? '' : snapshot.model)}
                style={{ ...panelStyle, padding: 16, color: '#e2e8f0', textAlign: 'left', cursor: 'pointer', fontFamily: 'inherit', borderColor: active ? '#22d3ee' : 'rgba(148,163,184,0.14)' }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
                  <Cpu size={18} color="#22d3ee" />
                  {active ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
                </div>
                <div style={{ color: '#f8fafc', fontSize: 15, fontWeight: 800, marginTop: 10 }}>{MODEL_NAMES[snapshot.model] || snapshot.model}</div>
                <div style={{ color: '#94a3b8', fontSize: 10, marginTop: 4 }}>{snapshot.num_hidden_layers}L · d{snapshot.hidden_size} · MLP {formatNumber(snapshot.intermediate_size)}</div>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, marginTop: 12, fontSize: 10 }}>
                  <span style={{ color: '#bae6fd' }}>{summary.familyCount}/9 模式族</span>
                  <span style={{ color: '#fbbf24' }}>{boundary.state}</span>
                </div>
              </button>
            );
          })}
        </div>
        {selectedModel && (
          <div style={{ padding: 16, borderTop: '2px solid #22d3ee', background: 'rgba(2,6,23,0.4)' }}>
            <ModelDetail snapshot={selectedModel.snapshot} summary={selectedModel.summary} />
          </div>
        )}
      </section>

      <section style={{ display: 'grid', gap: 12 }}>
        <div>
          <div style={{ color: '#f8fafc', fontSize: 17, fontWeight: 800 }}>最近科学进展</div>
          <div style={{ color: '#94a3b8', fontSize: 12, marginTop: 4 }}>从协议资格、粗块筛选、测量校准到一般复制候选关闭，点击查看结论边界。</div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10 }}>
          {RECENT_STAGES.map((stage) => {
            const active = activeStage === stage.id;
            return (
              <button key={stage.id} type="button" aria-expanded={active} onClick={() => setActiveStage(active ? '' : stage.id)} style={{ ...panelStyle, padding: 15, color: '#e2e8f0', textAlign: 'left', cursor: 'pointer', fontFamily: 'inherit', borderColor: active ? '#f59e0b' : 'rgba(148,163,184,0.14)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
                  <span style={{ color: '#fbbf24', fontSize: 10, fontWeight: 800 }}>{stage.phase}</span>
                  {active ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                </div>
                <div style={{ color: '#f8fafc', fontSize: 13, fontWeight: 800, marginTop: 9 }}>{stage.title}</div>
                <div style={{ color: '#7dd3fc', fontSize: 11, fontFamily: 'monospace', marginTop: 5 }}>{stage.value}</div>
                <div style={{ color: '#94a3b8', fontSize: 11, lineHeight: 1.55, marginTop: 8 }}>{stage.summary}</div>
              </button>
            );
          })}
        </div>
        {selectedStage && (
          <div style={{ padding: 15, borderLeft: '3px solid #f59e0b', background: 'rgba(245,158,11,0.05)', color: '#cbd5e1', fontSize: 12, lineHeight: 1.7 }}>
            <strong style={{ color: '#fef3c7' }}>{selectedStage.phase} · {selectedStage.title}：</strong>{selectedStage.detail}
          </div>
        )}
      </section>

      <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: 10 }}>
        <div style={{ ...panelStyle, padding: 16, borderLeft: '3px solid #34d399' }}>
          <CheckCircle2 size={18} color="#34d399" />
          <div style={{ color: '#d1fae5', fontSize: 13, fontWeight: 800, marginTop: 8 }}>已经完成</div>
          <div style={{ color: '#94a3b8', fontSize: 11, lineHeight: 1.7, marginTop: 7 }}>三模型注册与顺序 CUDA 执行、九族统一数据合同、全层组件观测、物理地址图谱、证据内核和客户端分区加载。</div>
        </div>
        <div style={{ ...panelStyle, padding: 16, borderLeft: '3px solid #f59e0b' }}>
          <Network size={18} color="#f59e0b" />
          <div style={{ color: '#fef3c7', fontSize: 13, fontWeight: 800, marginTop: 8 }}>当前研究层级</div>
          <div style={{ color: '#94a3b8', fontSize: 11, lineHeight: 1.7, marginTop: 7 }}>存在模型特异粗块与局部传播候选，但尚未恢复可迁移规则；九族是基准分类，不是已经证明的语言本体。</div>
        </div>
        <div style={{ ...panelStyle, padding: 16, borderLeft: '3px solid #fb7185' }}>
          <AlertTriangle size={18} color="#fb7185" />
          <div style={{ color: '#ffe4e6', fontSize: 13, fontWeight: 800, marginTop: 8 }}>关键硬伤</div>
          <div style={{ color: '#94a3b8', fontSize: 11, lineHeight: 1.7, marginTop: 7 }}>执行路径不完全等价、跨模型候选为零、单神经元因果为零、完整自然链为零，且三个小模型不能直接代表大模型或人脑。</div>
        </div>
        <div style={{ ...panelStyle, padding: 16, borderLeft: '3px solid #60a5fa' }}>
          <ShieldCheck size={18} color="#60a5fa" />
          <div style={{ color: '#dbeafe', fontSize: 13, fontWeight: 800, marginTop: 8 }}>下一证据门</div>
          <div style={{ color: '#94a3b8', fontSize: 11, lineHeight: 1.7, marginTop: 7 }}>先冻结可复现执行路径和模型哈希，再选择跨三模型基线合格的新机制，执行粗块到最小交互集合的分层因果提取。</div>
        </div>
      </section>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, color: '#7dd3fc', fontSize: 10 }}>
        <a href={MODEL_SYSTEM_SOURCES.atlas} target="_blank" rel="noreferrer" style={{ color: 'inherit' }}><Layers3 size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} />物理图谱 manifest</a>
        <a href={MODEL_SYSTEM_SOURCES.kernel} target="_blank" rel="noreferrer" style={{ color: 'inherit' }}><Database size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} />证据内核 manifest</a>
      </div>
    </div>
  );
};
