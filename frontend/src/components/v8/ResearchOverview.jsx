import { Fragment, useState } from 'react';
import { publishResearchContext, useResearchContext } from '../app/researchContext.js';
import { useApi } from '../app/rdc_fusion/api.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import { ExtraPackPrompt } from '../open/DataBar.jsx';
import { countMeasuredLanguageItems, languageCatalogCount } from './LanguageCatalog.jsx';
import ResearchDisclosure from './ResearchDisclosure.jsx';
import ResearchLibrary from './ResearchLibrary.jsx';
import { EVIDENCE_LABEL } from './researchFragmentUtil.js';
import { Ev, LatestFindings, NextStep, languageSections, relationCounts } from './ClientGuide.jsx';
import { itemLevel, shortModel, workbenchHref } from './v8Util.js';
import { ScoreBoard } from './SpaceGuides.jsx';

const TOPICS = [['KN', '知识网络'], ['RS', '推理与功能'], ['GR', '语法体系']];
const CARDS = [
  ['language', '语言图谱', '语言里有哪些知识、推理和语法现象', '第一部分'],
  ['internal', 'LLM 编码图谱', '模型的参数是什么，运行时信息怎样流动', '第二部分'],
  ['relations', '关联机制', '某种语言能力，由模型内部的什么负责', '第三部分'],
];
const FULL_SNAPSHOT = /^[0-9a-f]{64}$/;
const reportRef = work => work.stages?.registration_validated?.report_sha256;
const graphRef = work => work.stages?.registration_validated?.graph_sha256;
const stageState = stage => stage?.state ?? stage?.status;
// The archived P2794 report tests entity attributes and condition-dependent source choice.
const LEGACY_TOPICS = new Map([['P2794-ATOMIC-ARBITRATION-v4', ['KN', 'RS']]]);

function workTopics(work) {
  const families = work.stages?.freeze?.families;
  const registered = Array.isArray(work.topics) ? work.topics : [];
  const fromFamilies = Array.isArray(families) ? families.map(family => String(family).split(/[-.]/)[0]) : [];
  const tags = new Set([...registered, ...fromFamilies, ...(LEGACY_TOPICS.get(work.run_id) || [])]);
  return TOPICS.filter(([id]) => tags.has(id));
}

function workHref(work, pane, { snapshot, model, dataset }) {
  const q = new URLSearchParams({ run: work.run_id });
  const identity = FULL_SNAPSHOT.test(work.model_snapshot) ? work.model_snapshot : snapshot;
  if (identity) q.set('snapshot', identity);
  if (model) q.set('mdl', model);
  if (dataset) q.set('ds', dataset);
  if (pane === 'internal') {
    q.set('lens', 'spatial');
    q.set('view', 'graph');
    q.set('gview', 'internal');
  } else {
    q.set('lens', 'process');
    q.set('station', pane === 'language' ? 'plan' : 'execution');
  }
  return workbenchHref(q.toString());
}

function WorkList({ works, pane, scope, run }) {
  const [expanded, setExpanded] = useState(false);
  const selected = works.find(work => work.run_id === run);
  const ordered = selected ? [selected, ...works.filter(work => work !== selected)] : works;
  if (!works.length) return <p className="v8-note">当前范围内暂无对应工作包。</p>;
  return <>
    <ul className="v8-overview-worklist">
      {ordered.slice(0, expanded ? ordered.length : 4).map(work => {
        const labels = workTopics(work).map(([, label]) => label);
        return <li key={work.run_id}>
          <a href={workHref(work, pane, scope)}>{work.title || work.run_id}</a>
          {labels.length > 0 && <small>{labels.join(' · ')}</small>}
          {pane === 'relations' && work.stages?.validation && <small>
            留出记录 · {({ done: '已执行', not_supported: '未支持', inconclusive: '尚未证实', not_tested: '未测试' })[stageState(work.stages.validation)] || '查看详情'}
          </small>}
        </li>;
      })}
    </ul>
    {works.length > 4 && <button type="button" className="fw-tbtn" onClick={() => setExpanded(value => !value)}>
      {expanded ? '收起工作包列表' : `查看全部 ${works.length} 个工作包`}
    </button>}
  </>;
}

const countLabel = value => value == null ? '—' : value.toLocaleString('zh-CN');

function coverageLabel(done, total, unit) {
  const ratio = done == null || !total ? null : 100 * done / total;
  const percent = ratio == null ? '' : ratio > 0 && ratio < 0.01 ? ' · <0.01%' : ` · ${ratio.toFixed(2)}%`;
  return `${countLabel(done)} / ${countLabel(total)} ${unit}${percent}`;
}

function NeuronChecks({ rows, skeleton, onOpenUnit }) {
  const [selected, setSelected] = useState(null);
  const byLayer = new Map();
  for (const row of rows) {
    if (!byLayer.has(row.block)) byLayer.set(row.block, []);
    byLayer.get(row.block).push(row);
  }
  const layers = [...byLayer.keys()].sort((a, b) => a - b);
  const block = byLayer.has(selected) ? selected : layers[0];
  if (!layers.length) return <p className="v8-note">该模型暂无已登记的解释检验。</p>;
  return <>
    <label>按块查看 <select aria-label="已检验神经元所在块" value={block} onChange={event => setSelected(Number(event.target.value))}>
      {layers.map(layer => <option key={layer} value={layer}>块 {layer} · {byLayer.get(layer).length} 个</option>)}
    </select></label>
    <ul className="v8-coverage-items" aria-label="已检验神经元">
      {byLayer.get(block).map(row => <li key={row.unit}>
        <button type="button" className="v6-linkbtn" onClick={() => onOpenUnit(row, skeleton.model)}>块 {row.block} · 单元 {row.unit}</button>
        <span className="v8-note">{row.grade}</span>
      </li>)}
    </ul>
  </>;
}

function ModelDetails({ pane, snapshot, runIds, onGo }) {
  const index = useApi('/api/traces/research-space');
  const scoped = (index.data?.objects || []).filter(object =>
    (!snapshot || object.model_snapshot === snapshot) && (!runIds || runIds.has(object.body.run_id)));
  const features = scoped.filter(object => object.type === 'feature');
  const bySnapshot = new Map();
  for (const feature of features) {
    if (!bySnapshot.has(feature.model_snapshot)) bySnapshot.set(feature.model_snapshot, new Set());
    bySnapshot.get(feature.model_snapshot).add(feature.id);
  }
  const levels = {};
  if (pane === 'relations') {
    for (const edge of scoped.filter(object => object.type === 'edge')) {
      const { source, target, evidence } = edge.body;
      const ids = bySnapshot.get(edge.model_snapshot);
      if (ids && ((source?.startsWith('language:') && ids.has(target)) || (target?.startsWith('language:') && ids.has(source)))) {
        const label = EVIDENCE_LABEL[evidence?.level] || '其他登记证据';
        levels[label] = (levels[label] || 0) + 1;
      }
    }
  }
  return <>
    <div className="v6-toolbar">
      {pane === 'internal' ? <>
        <button type="button" className="fw-tbtn" onClick={() => onGo('spatial', 'graph', null, { gview: 'internal' })}>打开内部 3D 图谱</button>
        <button type="button" className="fw-tbtn" onClick={() => onGo('spatial', 'parameters', null, { ptab: 'weights' })}>查看真实参数</button>
      </> : <button type="button" className="fw-tbtn" onClick={() => onGo('route', null, null, { rtab: 'rules' })}>查看成果与证据</button>}
    </div>
    {index.error ? <Empty>{index.error}<ExtraPackPrompt pack={index.missingPack} /></Empty>
      : !index.data ? <p className="v8-note">读取已登记的图谱证据…</p>
        : pane === 'internal' ? <p className="v8-note">当前范围已登记 {features.length} 张特征卡。3D 图谱可查看原生定位和对应样本。</p>
          : Object.keys(levels).length ? <ul className="v8-overview-tags" aria-label="跨图谱证据类型">
            {Object.entries(levels).map(([label, count]) => <li key={label}>{label} {count} 条</li>)}
          </ul> : <p className="v8-note">当前范围暂无连接语言模式与内部特征的登记边。</p>}
  </>;
}

function FrozenTemplates({ snapshot, onGo }) {
  const progress = useApi('/api/research-library/progress');
  const [id, setId] = useState('');
  const buckets = progress.data?.buckets || [];
  const current = buckets.filter(bucket => bucket.snapshot_verified && (!snapshot || bucket.snapshot === snapshot));
  const historical = buckets.filter(bucket => !bucket.snapshot_verified);
  const choices = [...current, ...historical];
  const bucket = choices.find(entry => entry.id === id) || current[0];
  return <div className="v8-overview-fold-body">
    <p className="v8-note">分片进度按冻结模板统计。旧配置记录单列，保留计划量和核验状态。</p>
    {progress.error && <Empty>{progress.error}</Empty>}
    {!progress.error && !progress.data && <p className="v8-note">读取冻结分片记录…</p>}
    {progress.data && !current.length && <p className="v8-note">当前范围暂无完整快照的冻结分片记录。</p>}
    {choices.length > 0 && <label>分片来源 <select value={bucket?.id || ''} onChange={event => setId(event.target.value)}>
      {!current.length && <option value="">选择历史配置记录</option>}
      {choices.map(entry => <option key={entry.id} value={entry.id}>
        {entry.model} · {entry.snapshot.slice(0, 16)} · {entry.precision}{entry.snapshot_verified ? '' : ' · 历史配置'}
      </option>)}
    </select></label>}
    {bucket && <div className="v8-weight-scroll"><table className="v6-table">
      <thead><tr><th>冻结模板</th><th>已核验 / 计划</th></tr></thead>
      <tbody>{(bucket.templates || []).map(template => <tr key={template.tm_id}>
        <td>{template.name}<small>{template.tm_id}</small></td>
        <td>{bucket.snapshot_verified && bucket.precision !== 'unknown' ? `${template.done} / ${template.planned}` : `待核验 · 计划 ${template.planned}`}</td>
      </tr>)}</tbody>
    </table></div>}
    <div className="v6-toolbar"><button type="button" className="fw-tbtn" onClick={() => onGo('process', 'plan')}>查看研究计划与工作包</button></div>
  </div>;
}

export default function ResearchOverview({ onGo, tree, skeletons = {}, model, dataset, open = '', renderLanguage, onOpenUnit }) {
  const { snapshot, run } = useResearchContext();
  const packages = useApi('/api/research-library/workpackages');
  const [topic, setTopic] = useState('');
  const [plan, setPlan] = useState(false);
  const [archive, setArchive] = useState(false);
  const modelList = Object.values(skeletons);
  const matched = snapshot ? modelList.filter(entry => entry.model_snapshot === snapshot) : [];
  const skeleton = snapshot ? matched.find(entry => entry.model === model) || (matched.length === 1 ? matched[0] : null) : skeletons[model];
  const analysisModel = snapshot ? skeleton?.model : model;
  const analyzedLanguageCount = tree?.models?.includes(analysisModel)
    ? countMeasuredLanguageItems(new Set(tree.items.filter(item => itemLevel(item, analysisModel) >= 1).map(item => item.id)))
    : null;
  const measured = new Set((tree?.models?.includes(analysisModel) ? tree.items : []).filter(item => itemLevel(item, analysisModel) >= 1).map(item => item.id));
  const sections = languageSections(measured);
  const space = useApi('/api/traces/research-space');
  const relations = relationCounts(space.data?.objects, skeleton?.model_snapshot || snapshot);
  const layers = skeleton?.n_layers;
  const width = skeleton?.d_ff;
  const neuronTotal = Number.isInteger(layers) && layers > 0 && Number.isInteger(width) && width > 0 ? layers * width : null;
  const censusKey = skeleton?.census_run ? skeleton.census_key : null;
  const checks = useApi(censusKey && neuronTotal ? `/api/neurons/explained?model_key=${encodeURIComponent(censusKey)}` : null);
  const validChecks = Boolean(censusKey) && checks.data?.model_key === censusKey && Array.isArray(checks.data?.results);
  const checked = validChecks ? [...new Map(checks.data.results.filter(row =>
    Number.isInteger(row.block) && row.block >= 0 && row.block < layers &&
    Number.isInteger(row.unit) && row.unit >= 0 && row.unit < width
  ).map(row => [`${row.block}:${row.unit}`, row])).values()] : [];
  const grades = new Map();
  for (const row of checked) grades.set(row.grade, (grades.get(row.grade) || 0) + 1);
  const works = (packages.data?.workpackages || []).map(entry => entry.body);
  const snapshots = [...new Set([...modelList, ...works].map(entry => entry.model_snapshot).filter(value => FULL_SNAPSHOT.test(value)))];
  if (snapshot && !snapshots.includes(snapshot)) snapshots.unshift(snapshot);
  const scoped = works.filter(work => !snapshot || work.model_snapshot === snapshot);
  const internal = scoped.filter(work => FULL_SNAPSHOT.test(work.model_snapshot));
  const reports = scoped.filter(work => reportRef(work) || graphRef(work));
  const summaries = {
    language: coverageLabel(analyzedLanguageCount, languageCatalogCount, '项'),
    internal: coverageLabel(validChecks ? checked.length : null, neuronTotal, '个'),
    relations: space.data ? `${(relations.observation || 0) + (relations.heldout || 0) + (relations.intervention || 0)} 条连接` : '读取连接…',
  };
  const scope = { snapshot, model: analysisModel, dataset };
  const related = (open === 'internal' ? internal : open === 'relations' ? reports : scoped)
    .filter(work => !topic || workTopics(work).some(([id]) => id === topic));
  const relatedRunIds = topic ? new Set(related.map(work => work.run_id)) : null;
  return <div className="v8-route v8-overview">
    <header className="v8-overview-heading">
      <div>
        <h2>我们在破解什么</h2>
        <p>大模型怎样用有限的参数学会语言？我们把问题分成三部分：语言里有什么、模型里有什么、两边怎样对应。</p>
        <p className="v8-note wb-pro">{analysisModel ? `覆盖统计 · ${[shortModel(analysisModel), analysisModel.split('|')[2]].filter(Boolean).join(' · ')}` : '当前范围暂无可核对的统计模型。'}</p>
      </div>
      <details className="v8-overview-scope wb-pro">
        <summary title={snapshot || undefined}>{snapshot ? `模型快照 ${snapshot.slice(0, 12)}` : '全部研究'}</summary>
        <label>模型快照 <select aria-label="总览模型快照" value={snapshot} onChange={event => {
          const next = event.target.value;
          publishResearchContext({ snapshot: next, run: '', trace: '', context_trace: '', pmodel: '' });
        }}>
          <option value="">全部研究</option>
          {snapshots.map(value => <option key={value} value={value}>{modelList.find(entry => entry.model_snapshot === value)?.model.split('|')[0] || '快照'} · {value.slice(0, 12)}</option>)}
        </select></label>
      </details>
    </header>
    {packages.error && <p className="v8-note" role="status">工作包暂不可用。{packages.error}</p>}
    <div className="v8-direction-cards v8-overview-cards">
      {CARDS.map(([id, title, description, part]) => <Fragment key={id}>
        <button type="button" className="v8-card v8-direction v8-overview-card" aria-label={title} aria-expanded={open === id}
          aria-controls={open === id ? 'overview-detail' : undefined} onClick={() => onGo('route', null, null, { rtab: 'overview', opane: open === id ? '' : id })}>
          <small>{part}</small>
          <strong>{title}</strong>
          <span className="v8-note">{description}</span>
          <span className={`v8-overview-metric${id !== 'relations' ? ' v8-overview-ratio' : ''}`}>{summaries[id]}</span>
          {id === 'language' && <span className="v8-note">{sections.map(section => `${section.name} ${section.done} / ${section.total}`).join(' · ')}</span>}
          {id === 'internal' && <span className="v8-note">静态参数和动态机制分别在“空间”查看</span>}
          {id === 'relations' && <span className="v8-overview-evidence"><Ev kind="observation">看到过 {relations.observation || 0}</Ev> <Ev kind="heldout">新材料也成立 {relations.heldout || 0}</Ev> <Ev kind="intervention">改动后输出变了 {relations.intervention || 0}</Ev><br/><small>三类分开计数；另有 {relations.negative || 0} 条未成立</small></span>}
          <span className="v8-overview-action">{open === id ? '收起 ↑' : '查看详情 →'}</span>
        </button>
        {open === id && <section id="overview-detail" className="v8-card v8-overview-detail" aria-label={`${title}详情`}>
          <h3>{title}</h3>
          {id === 'language' && <>
            <p className="v8-note">总览以 {languageCatalogCount} 个目录条目为分母。已有分析记录按当前模型的实际测量去重。</p>
            {renderLanguage?.(analysisModel || '', snapshot)}
          </>}
          {id === 'internal' && <>
            <p className="v8-note">总数按 MLP 层数 × 每层单元数计算{neuronTotal ? `（${layers} × ${countLabel(width)}）` : ''}。解释检验单独统计，包含已检验但尚未得到解释的单元。</p>
            {neuronTotal == null ? <p className="v8-note">当前模型暂无可核对的 MLP 总数。</p>
              : !censusKey ? <p className="v8-note">当前模型暂无可回查的解释检验清单。</p>
                : checks.error ? <p className="v8-note">解释检验记录暂不可用。{checks.error}</p>
                  : !validChecks ? <p className="v8-note">读取解释检验记录…</p>
                    : <>
                      <ul className="v8-overview-tags">{[...grades].map(([grade, count]) => <li key={grade}>{grade} {count} 个</li>)}</ul>
                      <ResearchDisclosure title="查看已检验神经元">
                        <NeuronChecks key={censusKey} rows={checked} skeleton={skeleton} onOpenUnit={onOpenUnit} />
                      </ResearchDisclosure>
                    </>}
            <ResearchDisclosure title="图谱卡片与真实参数">
              <ModelDetails pane={id} snapshot={skeleton?.model_snapshot || snapshot} runIds={relatedRunIds} onGo={onGo} />
            </ResearchDisclosure>
          </>}
          {id === 'relations' && <ModelDetails pane={id} snapshot={snapshot} runIds={relatedRunIds} onGo={onGo} />}
          <details className="v8-overview-packages">
            <summary>相关工作包</summary>
            <div className="v6-toolbar"><label>研究主题 <select value={topic} onChange={event => setTopic(event.target.value)}>
              <option value="">全部主题</option>
              {TOPICS.map(([id, label]) => <option key={id} value={id}>{label}</option>)}
            </select></label></div>
            <p className="v8-note">主题依据工作包的标签与冻结材料。未标注工作包可在全部主题下查看。</p>
            <WorkList key={snapshot + topic} works={related} pane={id} scope={scope} run={run} />
          </details>
        </section>}
      </Fragment>)}
    </div>
    <ScoreBoard/>
    <LatestFindings onGo={onGo}/>
    <NextStep onGo={onGo}/>
    <details className="v8-card v8-overview-fold wb-pro" open={plan} onToggle={event => setPlan(event.currentTarget.open)}>
      <summary>阶段任务与历史分片</summary>
      {plan && <FrozenTemplates key={snapshot} snapshot={snapshot} onGo={onGo} />}
    </details>
    <details className="v8-card v8-overview-fold wb-pro" open={archive} onToggle={event => setArchive(event.currentTarget.open)}>
      <summary>研究归档与原始来源</summary>
      {archive && <ResearchLibrary compact />}
    </details>
  </div>;
}
