import { useMemo, useState } from 'react';
import {
  AlertTriangle, BookOpenCheck, Boxes, Database, GitCompareArrows,
  Layers3, Network, Plus, RefreshCw, ScanSearch, Target, X,
} from 'lucide-react';

import { useResearchWorkspace } from './useResearchWorkspace';

const SECTIONS = [
  { id: 'language', label: '语言计算图谱', icon: Network, description: '节点、操作、构式与 Case' },
  { id: 'fields', label: 'HiddenState 条件场', icon: Layers3, description: '完整场、Pair、Probe 与因果结果' },
  { id: 'theory', label: '理论与闭合', icon: BookOpenCheck, description: '主张、关键拼图与人工审核' },
];

const STATUS_LABELS = {
  planned: '待采集', collecting: '采集中', partial: '部分完成', complete: '已完成',
  queued: '待捕获', captured: '已捕获', validated: '已验证', rejected: '已否决',
  open: '开放', hypothesis: '假设', supported: '有支持', challenged: '有冲突', closed: '已闭合',
  in_progress: '进行中', blocked: '受阻', passed: '通过', pending: '待审核',
  untested: '未检验', qualified: '行为合格', failed: '未通过', defined: '已定义', reviewed: '已复核',
  call_only: '仅调用', delete_tested: '已删除检验', rescued: '已救援',
};

const EMPTY_OPERATION = {
  family_type: 'taxonomy', label: '', description: '', language: 'multi', invariants: '',
  changed_factors: '', context_conditions: '', counterfactual_operations: '', expected_outputs: '',
  next_evidence_gap: '', behavior_status: 'untested', evidence_level: 'E0', metadata: {},
};
const EMPTY_NODE = {
  node_type: 'form', label: '', normalized_form: '', language: 'zh', description: '',
  status: 'defined', evidence_level: 'E0', metadata: {},
};
const EMPTY_CASE = {
  operation_id: '', construction_id: '', label: '', input_text: '', variant_text: '',
  semantic_roles: {}, invariants: '', changed_factors: '', split: 'test',
  behavior_status: 'untested', metadata: {},
};
const EMPTY_FIELD = {
  language_object_id: '', model_id: 'qwen3', model_revision: '', case_id: '', run_id: '',
  token_count: 0, layer_count: 0, hidden_size: 0, embedding_parameter_count: 0,
  hiddenstate_parameter_count: 0, embedding_artifact: '', hiddenstate_artifact: '',
  coverage_scope: 'full', status: 'captured', evidence_level: 'E1', metadata: {},
};
const EMPTY_PROBE = {
  operation_id: '', field_record_id: '', run_id: '', source_checkpoint: 'embedding',
  target_checkpoint: 'layer_1', source_token: 0, target_token: 0, source_coordinate: 0,
  target_coordinate: 0, direction_id: '', dose: 0, response_sign: 0,
  response_amplitude: 0, output_effect: 0, artifact_path: '', status: 'captured',
  evidence_level: 'E1', metadata: {},
};
const EMPTY_GEAR = {
  operation_id: '', label: '', condition_domain: '', source_nodes: [], target_nodes: [],
  sign_structure: '', amplitude_model: '', output_effect: '', control_status: 'untested',
  causal_status: 'untested', evidence_level: 'E0', metadata: {},
};
const EMPTY_CLAIM = {
  title: '', statement: '', status: 'open', evidence_level: 'E0', supporting_count: 0,
  contradicting_count: 0, open_puzzle: '', next_test: '', metadata: {},
};

function count(value) {
  return new Intl.NumberFormat('zh-CN').format(Number(value || 0));
}

function splitList(value) {
  return String(value || '').split(/[\n,，]+/).map((item) => item.trim()).filter(Boolean);
}

function Badge({ value }) {
  return <span className={`research-workspace__badge is-${value || 'unknown'}`}>{STATUS_LABELS[value] || value || '未知'}</span>;
}

function Stat({ label, value, detail }) {
  return <article className="research-workspace__stat"><span>{label}</span><strong>{count(value)}</strong><small>{detail}</small></article>;
}

function EmptyState({ children }) {
  return <div className="research-workspace__empty"><Database size={22} />{children}</div>;
}

function ListInput({ label, value, onChange, placeholder = '' }) {
  return <label>{label}<textarea value={value} onChange={(event) => onChange(event.target.value)} placeholder={placeholder} /></label>;
}

function OperationForm({ form, setForm, saving, onCancel, onSubmit }) {
  return (
    <form className="research-workspace__form" onSubmit={onSubmit}>
      <h3>登记可检验语言操作</h3>
      <div className="research-workspace__form-grid">
        <label>操作族<input required value={form.family_type} onChange={(event) => setForm({ ...form, family_type: event.target.value })} placeholder="negation / taxonomy" /></label>
        <label>名称<input required value={form.label} onChange={(event) => setForm({ ...form, label: event.target.value })} /></label>
        <label>语言<input value={form.language} onChange={(event) => setForm({ ...form, language: event.target.value })} /></label>
        <label>行为资格<select value={form.behavior_status} onChange={(event) => setForm({ ...form, behavior_status: event.target.value })}><option value="untested">未检验</option><option value="qualified">行为合格</option><option value="failed">未通过</option></select></label>
      </div>
      <label>操作说明<textarea value={form.description} onChange={(event) => setForm({ ...form, description: event.target.value })} /></label>
      <div className="research-workspace__form-grid">
        <ListInput label="不变量" value={form.invariants} onChange={(value) => setForm({ ...form, invariants: value })} placeholder="逗号或换行分隔" />
        <ListInput label="变化量" value={form.changed_factors} onChange={(value) => setForm({ ...form, changed_factors: value })} />
        <ListInput label="语境条件" value={form.context_conditions} onChange={(value) => setForm({ ...form, context_conditions: value })} />
        <ListInput label="反事实控制" value={form.counterfactual_operations} onChange={(value) => setForm({ ...form, counterfactual_operations: value })} />
      </div>
      <ListInput label="预期输出" value={form.expected_outputs} onChange={(value) => setForm({ ...form, expected_outputs: value })} />
      <label>下一证据缺口<textarea value={form.next_evidence_gap} onChange={(event) => setForm({ ...form, next_evidence_gap: event.target.value })} /></label>
      <div className="research-workspace__form-actions"><button type="button" onClick={onCancel}>取消</button><button className="research-workspace__primary" disabled={saving}>保存操作</button></div>
    </form>
  );
}

function LanguagePanel({ snapshot, saving, onCreate }) {
  const operations = snapshot.operations || [];
  const nodes = useMemo(() => snapshot.language_nodes || [], [snapshot.language_nodes]);
  const cases = snapshot.cases || [];
  const [formType, setFormType] = useState('');
  const [operationForm, setOperationForm] = useState(EMPTY_OPERATION);
  const [nodeForm, setNodeForm] = useState(EMPTY_NODE);
  const [caseForm, setCaseForm] = useState(() => ({ ...EMPTY_CASE, operation_id: operations[0]?.id || '' }));
  const nodeCounts = useMemo(() => ['form', 'concept', 'role', 'context'].map((type) => ({
    type, count: nodes.filter((node) => node.node_type === type).length,
  })), [nodes]);

  const submitOperation = async (event) => {
    event.preventDefault();
    const payload = { ...operationForm };
    ['invariants', 'changed_factors', 'context_conditions', 'counterfactual_operations', 'expected_outputs']
      .forEach((key) => { payload[key] = splitList(payload[key]); });
    if (await onCreate('operations', payload)) { setOperationForm(EMPTY_OPERATION); setFormType(''); }
  };
  const submitNode = async (event) => {
    event.preventDefault();
    if (await onCreate('language-nodes', nodeForm)) { setNodeForm(EMPTY_NODE); setFormType(''); }
  };
  const submitCase = async (event) => {
    event.preventDefault();
    const payload = { ...caseForm, invariants: splitList(caseForm.invariants), changed_factors: splitList(caseForm.changed_factors) };
    if (await onCreate('cases', payload)) { setCaseForm({ ...EMPTY_CASE, operation_id: operations[0]?.id || '' }); setFormType(''); }
  };

  return (
    <div className="research-workspace__panel">
      <section className="research-workspace__notice"><Network size={20} /><div><strong>研究对象是可检验的语言计算图谱</strong><p>区分词面、概念、语义角色和上下文节点；用操作记录不变量、变化量、行为资格与反事实控制。种子条目只是 E0 定义，不算实验结果。</p></div></section>
      <div className="research-workspace__toolbar">
        <div className="research-workspace__inline-stats"><span>操作 <strong>{count(operations.length)}</strong></span><span>节点 <strong>{count(nodes.length)}</strong></span><span>边 <strong>{count(snapshot.language_edges?.length)}</strong></span><span>Case <strong>{count(cases.length)}</strong></span></div>
        <div className="research-workspace__toolbar-actions"><button type="button" onClick={() => setFormType(formType === 'node' ? '' : 'node')}><Plus size={15} />节点</button><button type="button" onClick={() => setFormType(formType === 'case' ? '' : 'case')}><Plus size={15} />Case</button><button type="button" className="research-workspace__primary" onClick={() => setFormType(formType === 'operation' ? '' : 'operation')}><Plus size={15} />语言操作</button></div>
      </div>

      {formType === 'operation' ? <OperationForm form={operationForm} setForm={setOperationForm} saving={saving} onCancel={() => setFormType('')} onSubmit={submitOperation} /> : null}
      {formType === 'node' ? (
        <form className="research-workspace__form" onSubmit={submitNode}>
          <h3>登记类型化语言节点</h3>
          <div className="research-workspace__form-grid"><label>节点类型<select value={nodeForm.node_type} onChange={(event) => setNodeForm({ ...nodeForm, node_type: event.target.value })}><option value="form">词面</option><option value="concept">概念</option><option value="role">语义角色</option><option value="context">上下文</option></select></label><label>名称<input required value={nodeForm.label} onChange={(event) => setNodeForm({ ...nodeForm, label: event.target.value })} /></label><label>标准形式<input value={nodeForm.normalized_form} onChange={(event) => setNodeForm({ ...nodeForm, normalized_form: event.target.value })} /></label><label>语言<input value={nodeForm.language} onChange={(event) => setNodeForm({ ...nodeForm, language: event.target.value })} /></label></div>
          <label>说明<textarea value={nodeForm.description} onChange={(event) => setNodeForm({ ...nodeForm, description: event.target.value })} /></label>
          <div className="research-workspace__form-actions"><button type="button" onClick={() => setFormType('')}>取消</button><button className="research-workspace__primary" disabled={saving}>保存节点</button></div>
        </form>
      ) : null}
      {formType === 'case' ? (
        <form className="research-workspace__form" onSubmit={submitCase}>
          <h3>登记冻结 Case</h3>
          <div className="research-workspace__form-grid"><label>语言操作<select required value={caseForm.operation_id} onChange={(event) => setCaseForm({ ...caseForm, operation_id: event.target.value })}>{operations.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label><label>Case 名称<input required value={caseForm.label} onChange={(event) => setCaseForm({ ...caseForm, label: event.target.value })} /></label><label>数据分区<select value={caseForm.split} onChange={(event) => setCaseForm({ ...caseForm, split: event.target.value })}><option value="train">训练</option><option value="validation">验证</option><option value="test">测试</option><option value="lockbox">锁箱</option></select></label><label>行为资格<select value={caseForm.behavior_status} onChange={(event) => setCaseForm({ ...caseForm, behavior_status: event.target.value })}><option value="untested">未检验</option><option value="qualified">行为合格</option><option value="failed">未通过</option></select></label></div>
          <label>基线输入<textarea required value={caseForm.input_text} onChange={(event) => setCaseForm({ ...caseForm, input_text: event.target.value })} /></label><label>变体输入<textarea value={caseForm.variant_text} onChange={(event) => setCaseForm({ ...caseForm, variant_text: event.target.value })} /></label>
          <div className="research-workspace__form-grid"><ListInput label="不变量" value={caseForm.invariants} onChange={(value) => setCaseForm({ ...caseForm, invariants: value })} /><ListInput label="变化量" value={caseForm.changed_factors} onChange={(value) => setCaseForm({ ...caseForm, changed_factors: value })} /></div>
          <div className="research-workspace__form-actions"><button type="button" onClick={() => setFormType('')}>取消</button><button className="research-workspace__primary" disabled={saving || !caseForm.operation_id}>保存 Case</button></div>
        </form>
      ) : null}

      <section><div className="research-workspace__section-title"><div><GitCompareArrows size={18} /><h2>语言操作族</h2></div><span>先行为合格，再定位内部机制</span></div><div className="research-workspace__operation-grid">{operations.map((operation) => <article key={operation.id}><header><div><small>{operation.family_type} · {operation.evidence_level}</small><h3>{operation.label}</h3></div><Badge value={operation.behavior_status} /></header><p>{operation.description || '尚未登记说明'}</p><dl><div><dt>不变量</dt><dd>{operation.invariants?.join('、') || '待定义'}</dd></div><div><dt>变化量</dt><dd>{operation.changed_factors?.join('、') || '待定义'}</dd></div><div><dt>下一缺口</dt><dd>{operation.next_evidence_gap || '待形成 Case/Pair 与反事实控制'}</dd></div></dl></article>)}</div></section>
      <section className="research-workspace__two-column">
        <div><div className="research-workspace__section-title"><div><Network size={18} /><h2>类型化节点</h2></div><span>{count(snapshot.language_edges?.length)} 条边</span></div><div className="research-workspace__node-types">{nodeCounts.map((item) => <article key={item.type}><span>{item.type}</span><strong>{count(item.count)}</strong></article>)}</div></div>
        <div><div className="research-workspace__section-title"><div><Boxes size={18} /><h2>冻结 Case</h2></div><span>{count(snapshot.constructions?.length)} 个构式</span></div>{cases.length ? <div className="research-workspace__compact-list">{cases.slice(0, 8).map((item) => <article key={item.id}><div><strong>{item.label}</strong><small>{item.split} · {item.input_text}</small></div><Badge value={item.behavior_status} /></article>)}</div> : <EmptyState>尚无 Case。先登记操作，再冻结基线和变体。</EmptyState>}</div>
      </section>
      <p className="research-workspace__compat-note">兼容目录仍保留 {count(snapshot.language_objects?.length)} 个旧 Token/构式对象；新研究优先写入类型化图谱。</p>
    </div>
  );
}

function FieldForm({ form, setForm, objects, saving, onCancel, onSubmit }) {
  const updateNumber = (key, value) => setForm({ ...form, [key]: Number(value || 0) });
  return (
    <form className="research-workspace__form" onSubmit={onSubmit}>
      <h3>登记 Embedding / HiddenState 原始场</h3>
      <div className="research-workspace__form-grid"><label>兼容语言对象<select required value={form.language_object_id} onChange={(event) => setForm({ ...form, language_object_id: event.target.value })}>{objects.map((item) => <option key={item.id} value={item.id}>{item.label} · {item.family}</option>)}</select></label><label>模型<input required value={form.model_id} onChange={(event) => setForm({ ...form, model_id: event.target.value })} /></label><label>Case ID<input value={form.case_id} onChange={(event) => setForm({ ...form, case_id: event.target.value })} /></label><label>Run ID<input value={form.run_id} onChange={(event) => setForm({ ...form, run_id: event.target.value })} /></label><label>Token 数<input type="number" min="0" value={form.token_count} onChange={(event) => updateNumber('token_count', event.target.value)} /></label><label>Layer 数<input type="number" min="0" value={form.layer_count} onChange={(event) => updateNumber('layer_count', event.target.value)} /></label><label>Hidden size<input type="number" min="0" value={form.hidden_size} onChange={(event) => updateNumber('hidden_size', event.target.value)} /></label><label>覆盖范围<select value={form.coverage_scope} onChange={(event) => setForm({ ...form, coverage_scope: event.target.value })}><option value="full">完整参数</option><option value="top_k">仅 Top-K 观察</option><option value="metadata_only">仅元数据</option></select></label><label>Embedding 参数数<input type="number" min="0" value={form.embedding_parameter_count} onChange={(event) => updateNumber('embedding_parameter_count', event.target.value)} /></label><label>HiddenState 参数数<input type="number" min="0" value={form.hiddenstate_parameter_count} onChange={(event) => updateNumber('hiddenstate_parameter_count', event.target.value)} /></label></div>
      <label>Embedding 产物路径<input value={form.embedding_artifact} onChange={(event) => setForm({ ...form, embedding_artifact: event.target.value })} /></label><label>HiddenState 产物路径<input value={form.hiddenstate_artifact} onChange={(event) => setForm({ ...form, hiddenstate_artifact: event.target.value })} /></label>
      <div className="research-workspace__form-actions"><button type="button" onClick={onCancel}>取消</button><button className="research-workspace__primary" disabled={saving || !form.language_object_id}>保存原始场</button></div>
    </form>
  );
}

function FieldPanel({ snapshot, saving, onCreate }) {
  const objects = snapshot.language_objects || [];
  const operations = snapshot.operations || [];
  const records = snapshot.field_records || [];
  const [formType, setFormType] = useState('');
  const [fieldForm, setFieldForm] = useState(() => ({ ...EMPTY_FIELD, language_object_id: objects[0]?.id || '' }));
  const [probeForm, setProbeForm] = useState(() => ({ ...EMPTY_PROBE, operation_id: operations[0]?.id || '' }));
  const [gearForm, setGearForm] = useState(() => ({ ...EMPTY_GEAR, operation_id: operations[0]?.id || '' }));
  const updateProbeNumber = (key, value) => setProbeForm({ ...probeForm, [key]: Number(value || 0) });

  const submitField = async (event) => { event.preventDefault(); if (await onCreate('field-records', fieldForm)) { setFieldForm({ ...EMPTY_FIELD, language_object_id: objects[0]?.id || '' }); setFormType(''); } };
  const submitProbe = async (event) => { event.preventDefault(); if (await onCreate('probe-responses', probeForm)) { setProbeForm({ ...EMPTY_PROBE, operation_id: operations[0]?.id || '' }); setFormType(''); } };
  const submitGear = async (event) => { event.preventDefault(); if (await onCreate('gear-candidates', gearForm)) { setGearForm({ ...EMPTY_GEAR, operation_id: operations[0]?.id || '' }); setFormType(''); } };

  return (
    <div className="research-workspace__panel">
      <section className="research-workspace__notice"><Layers3 size={20} /><div><strong>从静态热力图升级为条件响应场</strong><p>原始数据保留 Embedding 和全部 Layer × Token × HiddenSize；Pair 负责对齐，Probe 记录跨层跨坐标响应，候选齿轮继续接受调用、删除和救援检验。</p></div></section>
      <div className="research-workspace__asset-grid"><article><span>完整原始场</span><strong>{count(snapshot.overview?.full_field_count)}</strong><small>不可用 Top-K 替代</small></article><article><span>Pair 对齐</span><strong>{count(snapshot.overview?.pair_count)}</strong><small>Token + 语义角色</small></article><article><span>Probe 响应</span><strong>{count(snapshot.overview?.probe_response_count)}</strong><small>方向、符号、幅度</small></article><article><span>齿轮候选</span><strong>{count(snapshot.overview?.gear_candidate_count)}</strong><small>尚不是机制结论</small></article><article><span>因果干预</span><strong>{count(snapshot.overview?.intervention_count)}</strong><small>调用 / 删除 / 救援</small></article></div>
      <div className="research-workspace__toolbar"><div className="research-workspace__inline-stats"><span>所有记录都必须引用模型、Case/Run 和不可变产物。</span></div><div className="research-workspace__toolbar-actions"><button type="button" onClick={() => setFormType(formType === 'gear' ? '' : 'gear')}><Plus size={15} />齿轮候选</button><button type="button" onClick={() => setFormType(formType === 'probe' ? '' : 'probe')}><ScanSearch size={15} />Probe</button><button type="button" className="research-workspace__primary" onClick={() => setFormType(formType === 'field' ? '' : 'field')}><Plus size={15} />完整场</button></div></div>
      {formType === 'field' ? <FieldForm form={fieldForm} setForm={setFieldForm} objects={objects} saving={saving} onCancel={() => setFormType('')} onSubmit={submitField} /> : null}
      {formType === 'probe' ? (
        <form className="research-workspace__form" onSubmit={submitProbe}>
          <h3>登记条件 Probe 响应</h3><div className="research-workspace__form-grid"><label>语言操作<select required value={probeForm.operation_id} onChange={(event) => setProbeForm({ ...probeForm, operation_id: event.target.value })}>{operations.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label><label>Run ID<input required value={probeForm.run_id} onChange={(event) => setProbeForm({ ...probeForm, run_id: event.target.value })} /></label><label>源检查点<input required value={probeForm.source_checkpoint} onChange={(event) => setProbeForm({ ...probeForm, source_checkpoint: event.target.value })} /></label><label>目标检查点<input required value={probeForm.target_checkpoint} onChange={(event) => setProbeForm({ ...probeForm, target_checkpoint: event.target.value })} /></label><label>源 Token<input type="number" min="0" value={probeForm.source_token} onChange={(event) => updateProbeNumber('source_token', event.target.value)} /></label><label>目标 Token<input type="number" min="0" value={probeForm.target_token} onChange={(event) => updateProbeNumber('target_token', event.target.value)} /></label><label>源坐标<input type="number" min="0" value={probeForm.source_coordinate} onChange={(event) => updateProbeNumber('source_coordinate', event.target.value)} /></label><label>目标坐标<input type="number" min="0" value={probeForm.target_coordinate} onChange={(event) => updateProbeNumber('target_coordinate', event.target.value)} /></label><label>方向 ID<input value={probeForm.direction_id} onChange={(event) => setProbeForm({ ...probeForm, direction_id: event.target.value })} /></label><label>剂量<input type="number" step="any" value={probeForm.dose} onChange={(event) => updateProbeNumber('dose', event.target.value)} /></label><label>响应幅度<input type="number" step="any" value={probeForm.response_amplitude} onChange={(event) => updateProbeNumber('response_amplitude', event.target.value)} /></label><label>输出效应<input type="number" step="any" value={probeForm.output_effect} onChange={(event) => updateProbeNumber('output_effect', event.target.value)} /></label></div><label>完整产物路径<input value={probeForm.artifact_path} onChange={(event) => setProbeForm({ ...probeForm, artifact_path: event.target.value })} /></label><div className="research-workspace__form-actions"><button type="button" onClick={() => setFormType('')}>取消</button><button className="research-workspace__primary" disabled={saving || !probeForm.operation_id}>保存 Probe</button></div>
        </form>
      ) : null}
      {formType === 'gear' ? (
        <form className="research-workspace__form" onSubmit={submitGear}>
          <h3>登记候选“齿轮”</h3><div className="research-workspace__form-grid"><label>语言操作<select required value={gearForm.operation_id} onChange={(event) => setGearForm({ ...gearForm, operation_id: event.target.value })}>{operations.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label><label>候选名称<input required value={gearForm.label} onChange={(event) => setGearForm({ ...gearForm, label: event.target.value })} /></label></div><label>成立条件<textarea value={gearForm.condition_domain} onChange={(event) => setGearForm({ ...gearForm, condition_domain: event.target.value })} /></label><div className="research-workspace__form-grid"><label>符号结构<textarea value={gearForm.sign_structure} onChange={(event) => setGearForm({ ...gearForm, sign_structure: event.target.value })} /></label><label>幅度模型<textarea value={gearForm.amplitude_model} onChange={(event) => setGearForm({ ...gearForm, amplitude_model: event.target.value })} /></label></div><label>输出效应<textarea value={gearForm.output_effect} onChange={(event) => setGearForm({ ...gearForm, output_effect: event.target.value })} /></label><div className="research-workspace__form-actions"><button type="button" onClick={() => setFormType('')}>取消</button><button className="research-workspace__primary" disabled={saving || !gearForm.operation_id}>保存候选</button></div>
        </form>
      ) : null}
      <section><div className="research-workspace__section-title"><div><Layers3 size={18} /><h2>完整场索引</h2></div><span>{count(records.length)} 份记录</span></div>{records.length ? <div className="research-workspace__record-list">{records.map((record) => <article key={record.id}><header><div><span>{record.model_id}</span><strong>{record.language_object_label}</strong></div><Badge value={record.status} /></header><div className="research-workspace__field-metrics"><span>{count(record.token_count)} tokens</span><span>{count(record.layer_count)} layers</span><span>d={count(record.hidden_size)}</span><span>{record.coverage_scope === 'full' ? '完整参数' : record.coverage_scope}</span></div><dl><div><dt>Embedding</dt><dd>{record.embedding_artifact || '未登记'}</dd></div><div><dt>HiddenState</dt><dd>{record.hiddenstate_artifact || '未登记'}</dd></div></dl><small>{record.run_id || record.case_id || record.id} · {record.evidence_level}</small></article>)}</div> : <EmptyState>尚未登记完整 HiddenState 场。</EmptyState>}</section>
      <section className="research-workspace__two-column"><div><div className="research-workspace__section-title"><div><ScanSearch size={18} /><h2>Probe 响应</h2></div></div>{snapshot.probe_responses?.length ? <div className="research-workspace__compact-list">{snapshot.probe_responses.slice(0, 10).map((item) => <article key={item.id}><div><strong>{item.source_checkpoint} → {item.target_checkpoint}</strong><small>{item.run_id} · 幅度 {item.response_amplitude}</small></div><Badge value={item.status} /></article>)}</div> : <EmptyState>暂无 Probe 响应。</EmptyState>}</div><div><div className="research-workspace__section-title"><div><Target size={18} /><h2>齿轮候选</h2></div></div>{snapshot.gear_candidates?.length ? <div className="research-workspace__compact-list">{snapshot.gear_candidates.slice(0, 10).map((item) => <article key={item.id}><div><strong>{item.label}</strong><small>{item.condition_domain || '成立条件待补充'}</small></div><Badge value={item.causal_status} /></article>)}</div> : <EmptyState>暂无可接受因果检验的候选。</EmptyState>}</div></section>
    </div>
  );
}

function TheoryPanel({ snapshot, saving, onCreate }) {
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState(EMPTY_CLAIM);
  const pendingGateIds = new Set((snapshot.closure_applications || []).filter((item) => item.review_status === 'pending').map((item) => item.gate_id));
  const submit = async (event) => { event.preventDefault(); if (await onCreate('claims', form)) { setForm(EMPTY_CLAIM); setShowForm(false); } };
  const requestClosure = (gate) => onCreate('closure-applications', { gate_id: gate.id, requested_status: 'in_progress', evidence_ids: [], rationale: `请求独立审核“${gate.title}”当前证据；若证据不足，请返回具体缺口。`, requested_by: 'human' });
  return (
    <div className="research-workspace__panel research-workspace__theory">
      <section className="research-workspace__notice is-caution"><BookOpenCheck size={20} /><div><strong>理论提升与闭合必须人工审核</strong><p>界面只提交申请，不直接修改闭合状态。支持证据、反证、负对照、留出预测和因果干预需要同时可追溯。</p></div></section>
      <section><div className="research-workspace__section-title"><div><Target size={18} /><h2>闭合门</h2></div><span>{count(snapshot.overview?.pending_closure_application_count)} 个待审核申请</span></div><div className="research-workspace__gate-grid">{(snapshot.closure_gates || []).map((gate) => <article key={gate.id} className={`is-${gate.status}`}><header><strong>{gate.title}</strong><Badge value={gate.status} /></header><p>{gate.description}</p><footer><span>{count(gate.evidence_count)} 条证据</span>{gate.status !== 'passed' ? <button type="button" disabled={saving || pendingGateIds.has(gate.id)} onClick={() => requestClosure(gate)}>{pendingGateIds.has(gate.id) ? '等待审核' : '提交闭合申请'}</button> : null}</footer></article>)}</div></section>
      {(snapshot.closure_applications || []).length ? <section><div className="research-workspace__section-title"><div><BookOpenCheck size={18} /><h2>闭合审核队列</h2></div></div><div className="research-workspace__compact-list">{snapshot.closure_applications.map((item) => <article key={item.id}><div><strong>{item.gate_id}</strong><small>{item.rationale}</small></div><Badge value={item.review_status} /></article>)}</div></section> : null}
      <section><div className="research-workspace__section-title"><div><BookOpenCheck size={18} /><h2>理论主张与关键拼图</h2></div><button type="button" className="research-workspace__primary" onClick={() => setShowForm((value) => !value)}><Plus size={15} />新增主张</button></div>
        {showForm ? <form className="research-workspace__form" onSubmit={submit}><h3>登记可证伪理论主张</h3><label>标题<input required value={form.title} onChange={(event) => setForm({ ...form, title: event.target.value })} /></label><label>主张<textarea required value={form.statement} onChange={(event) => setForm({ ...form, statement: event.target.value })} /></label><div className="research-workspace__form-grid"><label>证据等级<select value={form.evidence_level} onChange={(event) => setForm({ ...form, evidence_level: event.target.value })}>{['E0', 'E1', 'E2', 'E3', 'E4'].map((item) => <option key={item}>{item}</option>)}</select></label><label>状态<select value={form.status} onChange={(event) => setForm({ ...form, status: event.target.value })}><option value="open">开放</option><option value="hypothesis">假设</option><option value="supported">有支持</option><option value="challenged">有冲突</option><option value="closed">已闭合</option></select></label></div><label>未解拼图<textarea value={form.open_puzzle} onChange={(event) => setForm({ ...form, open_puzzle: event.target.value })} /></label><label>下一验证<textarea value={form.next_test} onChange={(event) => setForm({ ...form, next_test: event.target.value })} /></label><div className="research-workspace__form-actions"><button type="button" onClick={() => setShowForm(false)}>取消</button><button className="research-workspace__primary" disabled={saving}>保存主张</button></div></form> : null}
        <div className="research-workspace__claim-list">{(snapshot.claims || []).map((claim) => <article key={claim.id}><header><div><span>{claim.evidence_level}</span><h3>{claim.title}</h3></div><Badge value={claim.status} /></header><p>{claim.statement}</p><div className="research-workspace__claim-evidence"><span>支持 {count(claim.supporting_count)}</span><span>冲突 {count(claim.contradicting_count)}</span></div><dl><div><dt>关键拼图</dt><dd>{claim.open_puzzle || '尚未登记'}</dd></div><div><dt>下一验证</dt><dd>{claim.next_test || '尚未登记'}</dd></div></dl></article>)}</div>
      </section>
    </div>
  );
}

export function ResearchWorkspace({ mode = 'overlay', onClose }) {
  const [section, setSection] = useState('language');
  const { snapshot, loading, saving, error, reload, create } = useResearchWorkspace();
  const overview = snapshot?.overview || {};
  return (
    <aside className={`simple-research-center research-workspace is-${mode}`} aria-label="理论研究中心">
      <header><div><span>RESEARCH ACCUMULATION</span><h1>理论研究中心</h1><p>语言计算图谱 → HiddenState 条件场 → 可审核理论闭合</p></div><div className="research-workspace__header-actions"><button type="button" onClick={() => reload()} aria-label="刷新"><RefreshCw size={17} className={loading ? 'is-spinning' : ''} /></button><button type="button" onClick={onClose} aria-label="关闭"><X size={18} /></button></div></header>
      {error ? <div className="research-workspace__error"><AlertTriangle size={16} /><span>{error}。请确认后端 5001 端口已启动。</span></div> : null}
      <div className="research-workspace__summary"><Stat label="语言操作" value={overview.operation_count} detail={`${count(overview.case_count)} 个冻结 Case`} /><Stat label="图谱" value={overview.language_node_count} detail={`${count(overview.language_edge_count)} 条边`} /><Stat label="完整场" value={overview.full_field_count} detail={`${count(overview.probe_response_count)} 个 Probe`} /><Stat label="因果拼图" value={overview.gear_candidate_count} detail={`${count(overview.intervention_count)} 次干预`} /><Stat label="闭合审核" value={overview.pending_closure_application_count} detail={`${count(overview.open_claim_count)} 个开放主张`} /></div>
      <nav className="research-workspace__nav" aria-label="研究积累分类">{SECTIONS.map((item) => { const Icon = item.icon; return <button type="button" key={item.id} className={section === item.id ? 'is-active' : ''} onClick={() => setSection(item.id)}><Icon size={18} /><span><strong>{item.label}</strong><small>{item.description}</small></span></button>; })}</nav>
      <main>{loading && !snapshot ? <EmptyState><RefreshCw className="is-spinning" size={20} />正在读取研究数据库…</EmptyState> : null}{snapshot && section === 'language' ? <LanguagePanel snapshot={snapshot} saving={saving} onCreate={create} /> : null}{snapshot && section === 'fields' ? <FieldPanel snapshot={snapshot} saving={saving} onCreate={create} /> : null}{snapshot && section === 'theory' ? <TheoryPanel snapshot={snapshot} saving={saving} onCreate={create} /> : null}</main>
      <footer className="research-workspace__footer"><Boxes size={14} /><span>SQLite 保存可查询索引；完整张量保留在不可变测试产物中；AI 只能提出闭合申请。</span></footer>
    </aside>
  );
}
