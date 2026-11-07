import { CatalogSummary } from './ClientGuide.jsx';
import { useState } from 'react';
import { useApi } from '../app/rdc_fusion/api.js';
import { CatalogItemDetails } from '../app/rdc_fusion/CatalogPanel.jsx';
import { STATUS_META } from '../app/rdc_fusion/catalogUtil.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import { ExtraPackPrompt } from '../open/DataBar.jsx';
import ResearchDisclosure from './ResearchDisclosure.jsx';
import { fragmentHref } from './researchFragmentUtil.js';
import { LockboxTable, PTile, PowerWizard } from './MainlineProgress.jsx';
import { PageHead, PillTools } from './CoreKit.jsx';
import { useCoverage } from './coverage.js';
import { PAIRS, SPLITS } from './data/mainline.js';

const STATUS = { supported_in_scope: '范围内支持', not_supported: '未支持', inconclusive: '尚未确定', not_measured: '未测' };
const METRIC = {
  paired_fixed_completion_log_odds_attenuation: '固定候选的成对偏好变化',
  natural_style_sufficiency_one_way: '自然生成的单向充分性（旧测量）',
  external_ai_natural_register_discovery: '自然生成双 AI 盲评（发现校准）',
};
const ROLE = { controlled_mechanism: '受控机制材料', capability_monitor: '能力监测', natural_discovery: '自然发现语料', counterexample: '反例与复算' };
const REVIEW = {
  ai_double_review_complete: '两种 AI 评分齐全', awaiting_ai_reviews: '等待 AI 评分',
  awaiting_human_reviews: '等待人工评分', awaiting_adjudication: '分歧待裁决', double_review_complete: '双人评分齐全',
};
const FIRST_PAGE = 8;

function SourceNotice({ source, label }) {
  return source.error ? <div role="status"><p className="v8-note">{label} · {source.error}</p><ExtraPackPrompt pack={source.missingPack}/></div>
    : !source.data ? <p className="v8-note">读取{label}…</p> : null;
}

function RawArtifact({ sha }) {
  const source = useApi(`/api/research-library/artifacts/${sha}`);
  return source.data ? <pre className="v8-artifact-json">{JSON.stringify(source.data, null, 2)}</pre>
    : <SourceNotice source={source} label="原始记录"/>;
}

function ClaimDetails({ claim, archive, onGo }) {
  const alias = archive.aliases?.findLast(({ body }) => body.canonical_feature_id === claim.canonical_feature_id &&
    body.run_id === claim.run_id && body.model_snapshot === claim.model_snapshot);
  return <ResearchDisclosure title={`${METRIC[claim.outcome_metric] || claim.outcome_metric} · ${STATUS[claim.status] || claim.status}`}>
    <p>{claim.scope}</p>
    <p className="v8-note">{claim.external_pattern_id} · {claim.canonical_feature_id}</p>
    <div className="v6-toolbar">
      {claim.run_id && <button className="fw-tbtn" type="button" onClick={() => onGo('execution', null, null,
        { run: claim.run_id, snapshot: claim.model_snapshot })}>查看关联实验</button>}
      {alias && <a className="fw-tbtn" href={fragmentHref({ sha: alias.body.artifact_sha256, id: alias.body.legacy_feature_id, body: {} })}>核验并定位真实参数</a>}
    </div>
    {claim.artifact_sha256 && <ResearchDisclosure title="原始主张证据"><RawArtifact sha={claim.artifact_sha256}/></ResearchDisclosure>}
  </ResearchDisclosure>;
}

function ReviewDetails({ review }) {
  const isAi = review.source_type === 'external_ai';
  return <ResearchDisclosure title={`${review.id} · ${REVIEW[review.status] || review.status}`}>
    <p>{isAi ? `外部 AI · ${(review.rater_models ?? []).join(' / ')} · ${review.completed_raters ?? '—'}/2 份评分` : '人工评审'} · {review.items} 条 · 分歧 {review.disagreements ?? '未计算'}</p>
    {isAi && <p className="v8-note">AI 评分与人工标注分别记录，科学证据等级按研究验证结果判断。</p>}
    {(review.note || review.independence) && <p className="v8-note">{review.note || review.independence}</p>}
    {(review.rating_artifacts ?? []).map(artifact => <ResearchDisclosure key={artifact.sha256} title={`查看 ${artifact.rater_model} 逐条评分`}>
      <RawArtifact sha={artifact.sha256}/>
    </ResearchDisclosure>)}
  </ResearchDisclosure>;
}

function DatasetDetails({ entry, archive, onGo }) {
  const dataset = entry.body;
  const claims = archive.claims.filter(({ body }) => body.dataset_versions?.includes(entry.id));
  const reviews = archive.reviews?.filter(review => review.dataset_versions?.includes(entry.id)) ?? [];
  return <>
    <dl className="v8-rnd-meta">
      <div><dt>用途</dt><dd>{ROLE[dataset.role] || dataset.role}</dd></div>
      <div><dt>版本</dt><dd>{dataset.version}</dd></div>
      <div><dt>样本</dt><dd>{dataset.sample_count} 条 / {dataset.source_groups?.length ?? '—'} 个来源组</dd></div>
      <div><dt>来源</dt><dd>{dataset.source}</dd></div>
      <div><dt>许可</dt><dd>{dataset.license}</dd></div>
      <div><dt>标签口径</dt><dd>{dataset.label_status}</dd></div>
    </dl>
    {claims.length > 0 && <ResearchDisclosure title={`关联主张与实验（${claims.length}）`}>
      {claims.map(({ id, body }) => <ClaimDetails key={id} claim={body} archive={archive} onGo={onGo}/>)}
    </ResearchDisclosure>}
    {!claims.length && <p className="v8-note">此材料尚无已登记的机制主张。</p>}
    {dataset.formal_gate_eligible === false && <p className="v8-note">仅供参考，未纳入正式能力门。</p>}
    {reviews.map(review => <ReviewDetails key={review.id} review={review}/>)}
    <ResearchDisclosure title="版本与来源组">
      <p className="v8-note">内容校验值 {dataset.content_sha256}</p>
      <p className="v8-note">{dataset.source_groups?.join('、')}</p>
    </ResearchDisclosure>
  </>;
}

export default function ResearchCatalog({ runFilter = '', template, onGo, onAutotest, onStudio }) {
  const catalog = useApi(runFilter ? null : '/api/catalog');
  const research = useApi('/api/research-library/catalog');
  const skeleton = useApi('/api/model/skeleton');
  const [query, setQuery] = useState('');
  const [type, setType] = useState('');
  const [limit, setLimit] = useState(FIRST_PAGE);
  const coverage = useCoverage();
  const archive = research.data;
  const datasets = archive?.datasets ?? [];
  const linked = new Set((archive?.claims ?? []).filter(({ body }) => body.run_id === runFilter)
    .flatMap(({ body }) => body.dataset_versions ?? []));
  const terms = query.trim().toLowerCase();
  const datasetRows = datasets.filter(entry => !runFilter || linked.has(entry.id)).map(entry => ({
    entry, key: `dataset:${entry.id}`, type: 'dataset', name: entry.body.dataset_id,
    status: entry.body.formal_gate_eligible === false ? '仅参考' : '已登记',
    search: [entry.id, entry.body.dataset_id, entry.body.version, ROLE[entry.body.role]],
  }));
  const templateRows = runFilter ? [] : (catalog.data?.items ?? []).map(entry => ({
    entry, key: `template:${entry.id}`, type: 'template', name: entry.name,
    status: STATUS_META[entry.status]?.[1] || entry.status_label || '已登记',
    search: [entry.id, entry.name, entry.tm_id, entry.pilot_tm_id, entry.category, ...(entry.factors ?? [])],
  }));
  const focused = template ? templateRows.find(({ entry }) => [entry.id, entry.tm_id, entry.pilot_tm_id].includes(template)) : null;
  const byName = (a, b) => a.name.localeCompare(b.name, 'zh-CN');
  datasetRows.sort(byName);
  templateRows.sort(byName);
  const mixed = Array.from({ length: Math.max(datasetRows.length, templateRows.length) },
    (_, index) => [datasetRows[index], templateRows[index]]).flat().filter(Boolean);
  const rows = mixed.filter(row => row !== focused && (!type || type === row.type) &&
    (!terms || row.search.join(' ').toLowerCase().includes(terms)));
  if (focused && (!type || type === 'template') && (!terms || focused.search.join(' ').toLowerCase().includes(terms))) rows.unshift(focused);
  const independentReviews = (archive?.reviews ?? []).filter(review =>
    !review.dataset_versions?.some(id => datasets.some(dataset => dataset.id === id)));

  const [stage, setStage] = useState('');
  const pairsBuilt = PAIRS.filter(p => p.built).length, splitsBuilt = SPLITS.filter(s => s.built).length;
  const stageDetail = {
    lang: <p className="v8-note">语言目录 {coverage.languageTotal} 项里，{coverage.language ?? '—'} 项已有当前模型的分析记录。目录本身在“空间 · 语言图谱”。</p>,
    pairs: <><div className="v8-weight-scroll"><table className="v6-table"><thead><tr><th>因素对</th><th>决策点设计</th><th>已有基础</th><th>主线材料</th></tr></thead>
      <tbody>{PAIRS.map(p => <tr key={p.pair}><td>{p.pair}</td><td>{p.design}</td><td>{p.base}</td><td>{p.built ? '已建' : '未建'}</td></tr>)}</tbody></table></div>
      <p className="v8-note">旧 CF、FU、AM 材料已暴露，只用于方法调试，不能用于确认。</p></>,
    splits: <LockboxTable/>,
    points: <p className="v8-note">决策点库的目标数量在 H1 按功效确定，现在没有分母，所以不显示百分比。</p>,
  }[stage];

  /* 首屏只有一个核心模块：材料流向的四个环；其余收进按钮（Phase 333） */
  return <section className="v8-rnd" aria-label="材料与数据">
    <PageHead title="材料与数据" sub="主线材料准备到哪里。所有实验用到的句子、语料和模型，都能在下方查到来源。" />
    <div className="mp-tiles">
      <PTile title="语言目录" done={coverage.language} total={coverage.languageTotal} unit="项" on={stage === 'lang'} onClick={() => setStage(stage === 'lang' ? '' : 'lang')}/>
      <PTile title="竞争对材料" done={pairsBuilt} total={PAIRS.length} unit="组" on={stage === 'pairs'} onClick={() => setStage(stage === 'pairs' ? '' : 'pairs')}/>
      <PTile title="三个划分" done={splitsBuilt} total={SPLITS.length} unit="份" on={stage === 'splits'} onClick={() => setStage(stage === 'splits' ? '' : 'splits')}/>
      <PTile title="决策点库" empty="未建" on={stage === 'points'} onClick={() => setStage(stage === 'points' ? '' : 'points')}/>
    </div>
    {stageDetail && <div className="mp-detail">{stageDetail}</div>}
    <PillTools label="材料与数据的更多内容" initial={runFilter || template ? 'all' : ''} items={[
      { id: 'power', label: '冻结向导（H1）', render: () => <PowerWizard/> },
      { id: 'lockbox', label: '锁箱看管', render: () => <LockboxTable/> },
      { id: 'all', label: '全部材料（搜索与来源）', render: () => <>
        <CatalogSummary templates={catalog.data?.items?.length} models={skeleton.data?.models?.length} tokens="1,904 万"/>
        <div className="v8-rnd-search">
          <input type="search" aria-label="搜索材料" placeholder="搜索数据集或模板" value={query}
            onChange={event => { setQuery(event.target.value); setLimit(FIRST_PAGE); }}/>
          <select aria-label="材料类型" value={type} onChange={event => { setType(event.target.value); setLimit(FIRST_PAGE); }}>
            <option value="">全部类型</option><option value="dataset">数据集</option><option value="template">模板</option>
          </select>
        </div>
        {runFilter && <div className="v8-rnd-caption"><span>所用材料 · {runFilter}</span>
          <button className="v6-linkbtn" type="button" onClick={() => onGo('catalog', null, null, { mrun: '', tm: '' })}>查看全部材料</button>
        </div>}
        <SourceNotice source={research} label="已登记数据集"/>
        {!runFilter && <SourceNotice source={catalog} label="测试模板"/>}
        {catalog.data?.sandbox_synthetic && <p className="v8-note">沙盒题库含合成模型桶，用于流程验证。</p>}
        {template && catalog.data && !runFilter && !focused && <p className="v8-note">当前题库没有链接指定的模板。</p>}
        {!rows.length && (research.data || (!runFilter && catalog.data)) && <Empty>
          {runFilter && !linked.size ? '此研究尚无通过主张登记关联的材料。' : '没有符合条件的材料。'}
        </Empty>}
        <div className="v8-rnd-list" aria-label="材料清单">
          {rows.slice(0, limit).map(row => <ResearchDisclosure key={row.key} initiallyOpen={row === focused} title={<span className="v8-rnd-row">
            <strong>{row.name}</strong><span className="v8-chip">{row.type === 'dataset' ? '数据集' : '模板'}</span>
            <small>{row.status}</small>
          </span>}>
            {row.type === 'dataset' ? <DatasetDetails entry={row.entry} archive={archive} onGo={onGo}/>
              : <CatalogItemDetails item={row.entry} onGo={onGo} onAutotest={onAutotest} onStudio={onStudio} embedded/>}
          </ResearchDisclosure>)}
        </div>
        {rows.length > limit && <button className="fw-tbtn" type="button" onClick={() => setLimit(limit + FIRST_PAGE)}>
          显示更多材料（还有 {rows.length - limit} 条）
        </button>}
        {!runFilter && independentReviews.length > 0 && <ResearchDisclosure title={`独立评审批次（${independentReviews.length}）`}>
          <p className="v8-note">评审批次保留各自的来源和评分口径。</p>
          {independentReviews.map(review => <ReviewDetails key={review.id} review={review}/>)}
        </ResearchDisclosure>}
      </> },
    ]} />
  </section>;
}
