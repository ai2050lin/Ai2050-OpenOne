import {
  AlertTriangle,
  Boxes,
  BrainCircuit,
  ChevronDown,
  ChevronUp,
  CircleDot,
  Cpu,
  Database,
  FlaskConical,
  GitCompareArrows,
  MonitorCog,
  Network,
  ShieldCheck,
} from 'lucide-react';
import { useMemo, useState } from 'react';

import { MODEL_SYSTEM_SOURCES, useModelSystemEvidence } from './useModelSystemEvidence';

const panelStyle = {
  border: '1px solid rgba(148, 163, 184, 0.14)',
  borderRadius: 8,
  background: 'rgba(15, 23, 42, 0.34)',
};

const STATUS_TONES = {
  ready: { label: '已建立', color: '#34d399' },
  partial: { label: '部分稳定', color: '#f59e0b' },
  candidate: { label: '候选级', color: '#60a5fa' },
  blocked: { label: '未闭合', color: '#fb7185' },
};

const SYSTEM_MODULES = [
  {
    id: 'registry', title: '真实模型注册与哈希', icon: Cpu, status: 'ready',
    summary: 'Qwen3、GLM4、DeepSeek7B 的架构和 config SHA256 已登记。',
    detail: '真实物理地址必须绑定模型修订、层、组件、单元类型和索引。当前三模型 model snapshot 已可供实验、图谱和 3D 客户端共同读取。',
    evidence: 'model_registry.json · 3/3 models',
  },
  {
    id: 'cuda_runner', title: '三模型 CUDA 顺序执行', icon: FlaskConical, status: 'ready',
    summary: '严格按 Qwen3 -> GLM4 -> DeepSeek7B 运行并释放显存。',
    detail: '该执行方式已经支撑九族全量观测和后续粗块实验，避免三个模型同时驻留导致显存溢出。它证明实验管线可执行，不代表模型机制一致。',
    evidence: 'Phase 326-344 execution contract',
  },
  {
    id: 'protocol_gate', title: '协议与基线资格门', icon: ShieldCheck, status: 'partial',
    summary: '答案对齐接口建立共同分母，但不等于完整自然聊天路径。',
    detail: 'Phase337 有 7/9 模型-接口单元合格；答案对齐接口三模型均为 12/12。Phase340 修复批处理异常后，18/27 模型任务单元通过四划分资格。',
    evidence: 'Phase 337 / 340',
  },
  {
    id: 'measurement', title: '测量路径不变性', icon: GitCompareArrows, status: 'partial',
    summary: '缓存、批量与执行后端不能默认视为语义等价。',
    detail: 'Phase342 的十一种执行模式审计中，只有两条路径通过完整不变性门。该问题属于测量系统，必须先冻结执行路径，才能解释模型内部因果差异。',
    evidence: 'Phase 342 · 2 stable execution paths',
  },
  {
    id: 'evidence_kernel', title: '图谱与统一证据内核', icon: Database, status: 'ready',
    summary: '九族、模型分区、Claim、来源和证据边界已经结构化发布。',
    detail: '当前证据包覆盖 9 个模式族、27 个模型族分区、72 个注册机制和数百万组件事件。覆盖率、候选、因果与闭合使用不同计数，不再合并成单一总百分比。',
    evidence: 'pattern_family_neuron_atlas.v1',
  },
  {
    id: 'causal_extractor', title: '分层因果规则提取', icon: BrainCircuit, status: 'blocked',
    summary: '粗块候选存在，但跨模型最小因果集合尚未恢复。',
    detail: 'Phase338 只有 GLM4 的早期 source MLP 粗块通过完整模型门；Phase344 将其收紧为模型与任务特异复制候选。跨模型粗块、单神经元必要性、受控充分性和完整中介链仍为零。',
    evidence: 'Phase 338-344 · cross-model 0',
  },
  {
    id: 'visual_client', title: '3D 物理图谱客户端', icon: MonitorCog, status: 'candidate',
    summary: '可显示真实层、组件和候选地址，但尚无单神经元因果动画。',
    detail: '客户端已经区分 H#、N#、G#、组件组和证据等级，并支持模式族与模型切换。显示坐标是逻辑布局；只有未来干预通过后，候选节点才能升级为因果样式。',
    evidence: 'Phase 324-334 client atlas',
  },
  {
    id: 'auto_research', title: '多 AI 自动研发', icon: Network, status: 'partial',
    summary: '已有研发控制台和多阶段流程，复现性门禁仍需统一。',
    detail: '自动研发可以组织分析、计划、代码、执行和总结，但不能自动把相关性提升为机制。下一步需要把模型哈希、数据切分、执行模式、结果校验和反例审计设为强制门。',
    evidence: 'AI R&D console · reproducibility gate pending',
  },
];

const NEXT_TASKS = [
  { id: 'reproducibility', title: '冻结可复现性门禁', priority: 'P0', detail: '统一模型与 tokenizer 哈希、chat template、缓存、批量、精度、随机种子、数据切分和结果 checksum；任何执行路径不变性失败都阻止机制升级。' },
  { id: 'next_mechanism', title: '选择新的跨模型合格机制', priority: 'P0', detail: '从尚未深审、协议稳定、三模型四划分基线合格的机制中预注册一个对象，避免继续追逐已被 Phase331-344 否定的旧候选。' },
  { id: 'hierarchical_search', title: '粗块到最小交互集合', priority: 'P1', detail: '先验证位置和组件粗块，再执行递归二分、逐成员移除、组合交互、随机同规模、错层和错位置控制；不再按激活或读出 Top-K 选神经元。' },
  { id: 'closure', title: '完整生成闭合', priority: 'P1', detail: '必要性、充分性和中介通过后，继续验证全词表 blocker、完整短语、自然 rollout、副作用和 private heldout；首词元变化不能单独算闭合。' },
];

function SystemModule({ module, active, onClick }) {
  const Icon = module.icon;
  const tone = STATUS_TONES[module.status];
  return (
    <button type="button" aria-expanded={active} onClick={onClick} style={{ ...panelStyle, padding: 14, color: '#e2e8f0', textAlign: 'left', cursor: 'pointer', fontFamily: 'inherit', borderColor: active ? tone.color : 'rgba(148,163,184,0.14)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
        <Icon size={17} color={tone.color} />
        {active ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </div>
      <div style={{ color: '#f8fafc', fontSize: 13, fontWeight: 800, marginTop: 9 }}>{module.title}</div>
      <div style={{ color: tone.color, fontSize: 10, fontWeight: 800, marginTop: 4 }}>{tone.label}</div>
      <div style={{ color: '#94a3b8', fontSize: 11, lineHeight: 1.55, marginTop: 7 }}>{module.summary}</div>
    </button>
  );
}

export const SystemStatusTab = () => {
  const { data } = useModelSystemEvidence();
  const [activeModule, setActiveModule] = useState('measurement');
  const [activeTask, setActiveTask] = useState('reproducibility');
  const metrics = data.atlas?.metrics || {};

  const systemCounts = useMemo(() => {
    const counts = { ready: 0, partial: 0, candidate: 0, blocked: 0 };
    SYSTEM_MODULES.forEach((module) => { counts[module.status] += 1; });
    return counts;
  }, []);

  const selectedModule = SYSTEM_MODULES.find((module) => module.id === activeModule);
  const selectedTask = NEXT_TASKS.find((task) => task.id === activeTask);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      <section style={{ padding: '20px 22px', background: 'rgba(15,23,42,0.2)', borderBottom: '1px solid rgba(148,163,184,0.14)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 18 }}>
          <div style={{ maxWidth: 820 }}>
            <div style={{ color: '#a7f3d0', fontSize: 10, fontWeight: 900, letterSpacing: 1.5 }}>RESEARCH SYSTEM STATUS</div>
            <h2 style={{ color: '#f8fafc', fontSize: 23, margin: '6px 0 7px' }}>逆向工程系统状态</h2>
            <div style={{ color: '#94a3b8', fontSize: 13, lineHeight: 1.7 }}>
              工程基础已经能够稳定生产大规模观测与候选图谱，当前主要瓶颈不再是“能不能运行”，而是测量是否可复现、候选是否具有低副作用因果必要性，以及自然生成能否闭合。
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(78px, 1fr))', gap: 6, minWidth: 210 }}>
            {Object.entries(systemCounts).map(([id, count]) => (
              <div key={id} style={{ padding: '7px 9px', borderTop: `2px solid ${STATUS_TONES[id].color}`, background: 'rgba(15,23,42,0.3)' }}>
                <div style={{ color: '#64748b', fontSize: 9 }}>{STATUS_TONES[id].label}</div>
                <div style={{ color: STATUS_TONES[id].color, fontSize: 16, fontWeight: 900, fontFamily: 'monospace' }}>{count}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section style={{ display: 'grid', gap: 11 }}>
        <div><div style={{ color: '#f8fafc', fontSize: 17, fontWeight: 800 }}>系统模块</div><div style={{ color: '#94a3b8', fontSize: 12, marginTop: 4 }}>点击模块查看它解决的问题、当前证据和严格边界。</div></div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(225px, 1fr))', gap: 9 }}>
          {SYSTEM_MODULES.map((module) => <SystemModule key={module.id} module={module} active={activeModule === module.id} onClick={() => setActiveModule(activeModule === module.id ? '' : module.id)} />)}
        </div>
        {selectedModule && (
          <div style={{ padding: 15, borderLeft: `3px solid ${STATUS_TONES[selectedModule.status].color}`, background: 'rgba(2,6,23,0.42)' }}>
            <div style={{ color: '#e2e8f0', fontSize: 12, lineHeight: 1.7 }}>{selectedModule.detail}</div>
            <div style={{ color: '#7dd3fc', fontSize: 10, marginTop: 8 }}>证据：{selectedModule.evidence}</div>
          </div>
        )}
      </section>

      <section style={{ display: 'grid', gap: 11 }}>
        <div><div style={{ color: '#f8fafc', fontSize: 17, fontWeight: 800 }}>当前证据向量</div><div style={{ color: '#94a3b8', fontSize: 12, marginTop: 4 }}>不使用“系统完成度”平均分，直接展示各个硬门是否通过。</div></div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 8 }}>
          {[
            ['模型注册', `${data.models?.models?.length || 0}/3`, '#34d399'],
            ['模式族映射', `${metrics.mapped_family_count || 0}/${metrics.family_count || 0}`, '#22d3ee'],
            ['跨模型集合读出', metrics.phase330_cross_model_set_readout_specific_mechanism_count || 0, '#60a5fa'],
            ['跨模型行为必要性', metrics.phase330_cross_model_behavior_necessity_mechanism_count || 0, '#fb7185'],
            ['单神经元因果', metrics.single_unit_causal_count || 0, '#fb7185'],
            ['完整自然链', metrics.full_natural_chain_pass_count || 0, '#fb7185'],
          ].map(([label, value, color]) => (
            <div key={label} style={{ padding: '10px 12px', borderTop: `2px solid ${color}`, background: 'rgba(15,23,42,0.25)' }}>
              <div style={{ color: '#94a3b8', fontSize: 10 }}>{label}</div>
              <div style={{ color, fontSize: 18, fontWeight: 900, fontFamily: 'monospace', marginTop: 4 }}>{value}</div>
            </div>
          ))}
        </div>
      </section>

      <section style={{ display: 'grid', gap: 11 }}>
        <div><div style={{ color: '#f8fafc', fontSize: 17, fontWeight: 800 }}>下一阶段任务</div><div style={{ color: '#94a3b8', fontSize: 12, marginTop: 4 }}>先修复测量和证据门，再启动新的机制因果提取。</div></div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: 9 }}>
          {NEXT_TASKS.map((task) => {
            const active = activeTask === task.id;
            return (
              <button key={task.id} type="button" aria-expanded={active} onClick={() => setActiveTask(active ? '' : task.id)} style={{ ...panelStyle, padding: 14, textAlign: 'left', color: '#e2e8f0', cursor: 'pointer', fontFamily: 'inherit', borderColor: active ? '#f59e0b' : 'rgba(148,163,184,0.14)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}><span style={{ color: '#fbbf24', fontSize: 10, fontWeight: 900 }}>{task.priority}</span>{active ? <ChevronUp size={14} /> : <ChevronDown size={14} />}</div>
                <div style={{ color: '#f8fafc', fontSize: 13, fontWeight: 800, marginTop: 8 }}>{task.title}</div>
              </button>
            );
          })}
        </div>
        {selectedTask && <div style={{ padding: 15, borderLeft: '3px solid #f59e0b', background: 'rgba(245,158,11,0.05)', color: '#cbd5e1', fontSize: 12, lineHeight: 1.7 }}><strong style={{ color: '#fef3c7' }}>{selectedTask.priority} · {selectedTask.title}：</strong>{selectedTask.detail}</div>}
      </section>

      <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: 9 }}>
        <div style={{ ...panelStyle, padding: 15, borderLeft: '3px solid #34d399' }}><Boxes size={17} color="#34d399" /><div style={{ color: '#d1fae5', fontSize: 12, fontWeight: 800, marginTop: 7 }}>工程价值</div><div style={{ color: '#94a3b8', fontSize: 11, lineHeight: 1.65, marginTop: 6 }}>统一数据合同、三模型执行、证据内核和 3D 地址图谱已经构成可复用研究平台。</div></div>
        <div style={{ ...panelStyle, padding: 15, borderLeft: '3px solid #f59e0b' }}><AlertTriangle size={17} color="#f59e0b" /><div style={{ color: '#fef3c7', fontSize: 12, fontWeight: 800, marginTop: 7 }}>科学边界</div><div style={{ color: '#94a3b8', fontSize: 11, lineHeight: 1.65, marginTop: 6 }}>平台能稳定产生拼图，不等于已经恢复语言规则；候选图谱、局部必要性和机制闭合必须继续分级。</div></div>
        <div style={{ ...panelStyle, padding: 15, borderLeft: '3px solid #60a5fa' }}><CircleDot size={17} color="#60a5fa" /><div style={{ color: '#dbeafe', fontSize: 12, fontWeight: 800, marginTop: 7 }}>第一性原理方向</div><div style={{ color: '#94a3b8', fontSize: 11, lineHeight: 1.65, marginTop: 6 }}>目标是从重复局部因果规则中还原条件化状态转移，而不是继续寻找一个贯穿所有模型的最大激活神经元或单一线性公式。</div></div>
      </section>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, fontSize: 10 }}>
        <a href={MODEL_SYSTEM_SOURCES.models} target="_blank" rel="noreferrer" style={{ color: '#7dd3fc' }}>模型注册表</a>
        <a href={MODEL_SYSTEM_SOURCES.atlas} target="_blank" rel="noreferrer" style={{ color: '#7dd3fc' }}>物理图谱 manifest</a>
        <a href={MODEL_SYSTEM_SOURCES.progress} target="_blank" rel="noreferrer" style={{ color: '#7dd3fc' }}>证据进度分母</a>
      </div>
    </div>
  );
};
