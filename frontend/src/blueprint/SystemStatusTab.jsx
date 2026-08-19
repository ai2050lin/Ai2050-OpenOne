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

import { MODEL_SYSTEM_SOURCES } from './useModelSystemEvidence';
import { CURRENT_RESEARCH_STATE } from '../researchKernel/currentResearchState';

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
    id: 'evidence_os', title: 'C001 不可变证据系统', icon: Database, status: 'ready',
    summary: '关键证据、假说、拼图、对象、构念与裁决进入机器可校验注册表。',
    detail: 'WP00 已迁移本地可见的关键证据子集，并记录适用域、来源和权限。系统不会伪造完整 K1–K210 总账，也不会让局部正结果自动升级为全局机制。',
    evidence: 'Phase 1236 · C001-WP00',
  },
  {
    id: 'typed_constructs', title: '类型化构念与权限门', icon: ShieldCheck, status: 'ready',
    summary: '内容、格式、自然生成和停止缓存分别定义、分别授权。',
    detail: '四种构念具有各自 observable、负控和闭合等级。内容正确不等于格式服从，候选排序不等于自然生成，第一 token 正确不等于停止缓存闭合。',
    evidence: 'K210 · 4 registered constructs',
  },
  {
    id: 'typed_readout', title: '七读出行为边界', icon: GitCompareArrows, status: 'partial',
    summary: 'Phase1235 的四类门为 1·0·1·1，异质总门失败。',
    detail: '完整候选评分、外部 trie、固定句式和自然内容可以稳定，而严格短字符串合同对模板与协议载体敏感。该结果是行为边界，不是隐藏内容模块证据。',
    evidence: 'Phase 1235 · K210',
  },
  {
    id: 'wp01_contract', title: 'WP01 冻结实验合同', icon: FlaskConical, status: 'blocked',
    summary: 'EXP-C001-WP01-001 已预注册，但 run_ready=false。',
    detail: '当前只授权无模型材料生成、反泄漏负控、环境清单和独立审计器冻结。在机器资格门通过前，不得运行 Qwen3、自动跨模型或采集内部状态。',
    evidence: 'D006 · EXP-C001-WP01-001',
  },
  {
    id: 'causal_extractor', title: '分层因果规则提取', icon: BrainCircuit, status: 'blocked',
    summary: '当前对象尚未获准采集未来响应或执行干预。',
    detail: '只有 WP01 的行为与构念双门通过，才允许测量未来响应张量，并进一步执行必要性、充分性、错误供体、救援、中介和完整生成闭合。',
    evidence: 'C001 permissions · future work package locked',
  },
  {
    id: 'visual_client', title: '3D 物理图谱客户端', icon: MonitorCog, status: 'candidate',
    summary: '可显示真实层、组件和候选地址，但尚无单神经元因果动画。',
    detail: '客户端已经区分 H#、N#、G#、组件组和证据等级，并支持模式族与模型切换。显示坐标是逻辑布局；只有未来干预通过后，候选节点才能升级为因果样式。',
    evidence: 'Phase 324-334 client atlas',
  },
  {
    id: 'cross_model', title: '跨模型功能同构', icon: Network, status: 'blocked',
    summary: '不能把跨模型运行当作 Qwen3 行为失败的救援。',
    detail: 'Qwen3 上的对象必须先通过本模型合同和因果闭合；随后才能按 Qwen3 → GLM4 → DS7B 顺序检验功能关系是否保持。当前没有跨模型统一机制结论。',
    evidence: 'D001 / D006 · cross-model locked',
  },
];

const NEXT_TASKS = [
  { id: 'wp01_preflight', title: 'WP01 无模型预审计', priority: 'P0', detail: '生成全新非双射世界、全新词汇与模板；冻结多参考内容判据、错内容正确格式负控、answer-absent、same-bag swap、query switch 和 alternative-program 上界。' },
  { id: 'freeze_environment', title: '冻结代码与环境清单', priority: 'P0', detail: '登记代码、材料、tokenizer、chat template、模型修订、精度、批次、随机种子、输出预算和全部文件哈希；独立审计器不得导入主实现。' },
  { id: 'behavior_adjudication', title: '一次性 Qwen3 行为裁决', priority: 'P1', detail: '仅在 run_ready=true 后运行 Qwen3。内容、格式、自然生成和停止缓存分别裁决；失败后不得调 prompt 或选择成功子集。' },
  { id: 'response_closure', title: '未来响应与救援闭合', priority: 'P2', detail: '仅在行为与构念双门通过后，采集事件×干预×读出×上下文响应，比较候选机制并执行必要性、错误供体、救援、中介与副作用负控。' },
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
  const [activeModule, setActiveModule] = useState('wp01_contract');
  const [activeTask, setActiveTask] = useState('wp01_preflight');

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
            <h2 style={{ color: '#f8fafc', fontSize: 23, margin: '6px 0 7px' }}>C001 全局结构辨识系统状态</h2>
            <div style={{ color: '#94a3b8', fontSize: 13, lineHeight: 1.7 }}>
              WP00 已把关键证据与权限编译为机器可校验系统；当前主要瓶颈是获得一个跨读出稳定、程序可识别且允许进入内部研究的功能对象。WP01 尚不可运行。
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
            ['候选假说', CURRENT_RESEARCH_STATE.registry.hypotheses, '#60a5fa'],
            ['开放拼图', CURRENT_RESEARCH_STATE.registry.puzzles, '#22d3ee'],
            ['关键证据记录', CURRENT_RESEARCH_STATE.registry.evidence, '#34d399'],
            ['类型化构念', CURRENT_RESEARCH_STATE.registry.constructs, '#a78bfa'],
            ['WP01 Run Ready', String(CURRENT_RESEARCH_STATE.campaign.runReady).toUpperCase(), '#fb7185'],
            ['WP01 模型运行', CURRENT_RESEARCH_STATE.campaign.modelRuns, '#fb7185'],
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
