import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import useCookiePreference from './useCookiePreference.js';

const TARGETS_URL = '/research_data/current/language_graph_targets.v1.json';
const CATALOG_URL = '/research_data/current/language_encoding_catalog.json';
const EMPTY_LIST = Object.freeze([]);
const EVIDENCE_LEVELS = new Set(['E1_repeatable', 'E2_predictive', 'E3_causal_scoped']);
const TASK_STATUS = {
  queued: ['排队中', 'queued'], claimed: ['运行中', 'claimed'], done: ['已回传', 'done'],
  failed: ['失败', 'failed'], expired: ['租约过期', 'expired'],
};
const EVIDENCE_LABEL = {
  E1_repeatable: 'E1 可复现', E2_predictive: 'E2 留出预测', E3_causal_scoped: 'E3 范围内机制证据',
};

function exactTargetId(unit) {
  return unit?.language_graph_target_id || unit?.graph_target_id || '';
}

function isAuditedEvidence(unit) {
  const level = unit?.target_evidence_level || unit?.evidence_level || '';
  return unit?.audit_status === 'passed' && EVIDENCE_LEVELS.has(level);
}

function TargetStatus({ units, aggregate, connected }) {
  if (!connected) return <span className="lg-target-status unknown">进度未接入</span>;
  const workUnitCount = aggregate?.work_units ?? units.length;
  const byStatus = aggregate?.by_status || {};
  if (!workUnitCount) return <span className="lg-target-status planned">未绑定目标任务</span>;
  if (units.some(isAuditedEvidence) || (aggregate?.audit_passed > 0 && aggregate?.evidence_levels?.some(level => EVIDENCE_LEVELS.has(level)))) return <span className="lg-target-status evidence">有审计证据</span>;
  if ((aggregate?.awaiting_audit || 0) > 0 || units.some(unit => unit.evidence_state === 'awaiting_audit' || (unit.result_sha && !unit.audit_status))) {
    return <span className="lg-target-status review">结果待审计</span>;
  }
  if ((byStatus.claimed || 0) > 0 || units.some(unit => unit.task_status === 'claimed')) return <span className="lg-target-status running">运行中</span>;
  if ((byStatus.queued || 0) > 0 || units.some(unit => unit.task_status === 'queued')) return <span className="lg-target-status queued">排队中</span>;
  if ((byStatus.done || 0) > 0 || units.some(unit => unit.task_status === 'done')) return <span className="lg-target-status done">已执行，证据未晋级</span>;
  return <span className="lg-target-status blocked">有失败/过期任务</span>;
}

function SummaryCard({ label, value, note, tone = '' }) {
  return <div className={'lg-target-summary ' + tone}><small>{label}</small><b>{value}</b><span>{note}</span></div>;
}

function TargetDetails({ target, units, detailPage, onLoadMore, onRefresh, aggregate, connected, auditTrackingAvailable, onGo, milestones = [] }) {
  if (!target) return <div className="lg-target-empty">没有符合筛选条件的图谱目标。</div>;
  const matrix = target.test_matrix || {};
  const mapped = aggregate?.work_units ?? units.length;
  const uploaded = aggregate?.result_uploaded ?? units.filter(unit => !!unit.result_sha).length;
  const awaiting = aggregate?.awaiting_audit ?? units.filter(unit => unit.evidence_state === 'awaiting_audit').length;
  const audited = units.filter(isAuditedEvidence);
  const aggregateEvidence = auditTrackingAvailable ? aggregate?.evidence_levels || [] : [];
  const evidenceLevels = [...new Set([...aggregateEvidence, ...audited.map(unit => unit.target_evidence_level || unit.evidence_level)])];
  return (
    <article className="lg-target-detail">
      <div className="lg-target-detail-head">
        <div><span className="at-eyebrow">{target.id} · {target.family_id}</span><h4>{target.label}</h4><p>{target.research_question}</p></div>
        <TargetStatus units={units} connected={connected} />
      </div>

      <div className="lg-target-metrics">
        <div><small>目标级 Work Unit</small><b>{connected ? mapped : '—'}</b></div>
        <div><small>结果已回传</small><b>{connected ? uploaded : '—'}</b></div>
        <div><small>等待独立审计</small><b>{connected ? awaiting : '—'}</b></div>
        <div><small>审计后证据级</small><b>{connected ? (auditTrackingAvailable ? (evidenceLevels.map(level => EVIDENCE_LABEL[level] || level).join(' / ') || '尚无') : '接口未接入') : '—'}</b></div>
      </div>

      <div className="lg-target-graph-schema">
        <section><h5>外部语言图谱节点</h5><div>{(target.graph_nodes || []).map(item => <span key={item}>{item}</span>)}</div></section>
        <section><h5>关系边类型</h5><div>{(target.graph_edges || []).map(item => <code key={item}>{item}</code>)}</div></section>
      </div>

      <div className="lg-target-plan-grid">
        <section><h5>关键对照</h5><ul>{(matrix.contrasts || []).map(item => <li key={item}>{item}</li>)}</ul></section>
        <section><h5>混杂控制</h5><ul>{(matrix.controls || []).map(item => <li key={item}>{item}</li>)}</ul></section>
        <section><h5>留出检验</h5><ul>{(matrix.holdouts || []).map(item => <li key={item}>{item}</li>)}</ul></section>
        <section><h5>分布式切片轴</h5><ul>{(matrix.distributed_shards || []).map(item => <li key={item}>{item}</li>)}</ul></section>
      </div>

      <section className="lg-target-outputs"><h5>每个目标应回收的产物</h5><div>{(matrix.required_outputs || []).map(item => <span key={item}>{item}</span>)}</div></section>
      <div className="lg-target-dispatch-note">分布式派发时使用 <code>--language-graph-target-id {target.id}</code>；冻结模板 adapter 的 <code>language_graph_target_ids</code> 也必须包含此 ID。</div>

      <section className="lg-target-milestones">
        <h5>目标推进门</h5>
        <div className="lg-target-milestone-grid">
          {['可采集', '可复现', '可预测', '范围内机制证据'].map((label, index) => {
            const milestoneId = ['instrumented', 'repeatable', 'predictive', 'mechanism_scoped'][index];
            const count = index === 0 ? mapped : index === 1 ? audited.filter(unit => EVIDENCE_LEVELS.has(unit.target_evidence_level || unit.evidence_level)).length : 0;
            const reached = connected && (index === 0 ? mapped > 0 : index === 1 ? count > 0 || evidenceLevels.some(level => EVIDENCE_LEVELS.has(level)) : evidenceLevels.some(level => index === 2 ? level === 'E2_predictive' || level === 'E3_causal_scoped' : level === 'E3_causal_scoped'));
            const gate = milestones.find(row => row.id === milestoneId);
            return <div className={reached ? 'reached' : ''} key={milestoneId}><b>{label}</b><small>{gate?.rule || '判据需写入冻结实验合同并接受独立审计。'}</small></div>;
          })}
        </div>
      </section>

      {target.existing_records?.length > 0 && (
        <section className="lg-target-history"><h5>已有明确关联记录 · 不等同于目标完成</h5>
          {target.existing_records.map(record => <div key={record.id}><b>{record.id}</b><span>{record.status}</span><p>{record.summary}</p></div>)}
        </section>
      )}

      <section className="lg-target-units"><h5>已关联 Work Unit 明细</h5>
        {!connected && <div className="lg-target-empty">分布式执行视图不可用；此目标的实时进度暂时无法读取。</div>}
        {connected && mapped === 0 && <div className="lg-target-empty">当前没有通过 <code>language_graph_target_id = {target.id}</code> 精确关联的 Work Unit。历史 Puzzle、任务或结果不会按名称猜测归属。</div>}
        {connected && mapped > 0 && units.length === 0 && <div className="lg-target-empty">{detailPage?.loading ? '正在读取该目标的工作单明细…' : detailPage?.error || `该目标共关联 ${mapped} 个 Work Unit，当前页没有返回明细。`}</div>}
        {connected && mapped > 0 && units.length > 0 && <div className="lg-target-unit-list">{units.map(unit => {
          const status = TASK_STATUS[unit.task_status] || [unit.task_status || '未知', ''];
          const level = unit.target_evidence_level || unit.evidence_level;
          return <div key={unit.work_unit_id || unit.task_id}>
            <span className={'lg-target-unit-state ' + status[1]}>{status[0]}</span>
            <b>{unit.work_unit_id || unit.task_id || 'Work Unit'}</b>
            <small>{unit.model_id || '模型未知'} · {unit.split_id || 'split 未知'} · seed {unit.seed ?? '—'}</small>
            {unit.result_sha && <small className="mono">结果 sha256:{String(unit.result_sha).slice(0, 16)} · {unit.artifacts?.length || 0} 个 Artifact</small>}
            <small>证据状态：{unit.evidence_state || '未回报'}{unit.audit_status ? ` · 审计 ${unit.audit_status}` : ''}{level ? ` · ${EVIDENCE_LABEL[level] || level}` : ''}</small>
          </div>;
        })}</div>}
        {connected && mapped > 0 && <div className="lg-target-page-actions">
          <span>{detailPage?.total != null ? `显示 ${units.length} / ${detailPage.total} 条目标工作单` : `目标工作单 ${mapped} 条`}</span>
          <button type="button" disabled={detailPage?.loading} onClick={onRefresh}>{detailPage?.loading ? '读取中…' : '刷新明细'}</button>
          {detailPage?.hasMore && <button type="button" disabled={detailPage?.loading} onClick={onLoadMore}>再加载 50 条</button>}
        </div>}
      </section>

      <div className="lg-target-detail-actions">
        <button type="button" className="fw-tbtn primary" onClick={() => onGo?.('process')}>前往研发工作流 →</button>
        <button type="button" className="fw-tbtn" onClick={() => onGo?.('atlas', 'graph')}>打开研究图谱 →</button>
      </div>
    </article>
  );
}

export default function LanguageGraphTargets({ workView, workViewError, onGo }) {
  const [catalog, setCatalog] = useState(null);
  const [catalogError, setCatalogError] = useState('');
  const [historicalDatasetCount, setHistoricalDatasetCount] = useState(null);
  const [unmappedHistoricalCount, setUnmappedHistoricalCount] = useState(null);
  const [display, setDisplay] = useCookiePreference('rdc-route-targets-v1', { expanded: false, domain: 'all', query: '', selectedId: 'LG-TYPE-GRAPH' });
  const expanded = display?.expanded === true;
  const domain = typeof display?.domain === 'string' ? display.domain : 'all';
  const query = typeof display?.query === 'string' ? display.query : '';
  const selectedId = typeof display?.selectedId === 'string' ? display.selectedId : 'LG-TYPE-GRAPH';
  const setExpanded = value => setDisplay(previous => ({ ...previous, expanded: value }));
  const setDomain = value => setDisplay(previous => ({ ...previous, domain: value }));
  const setQuery = value => setDisplay(previous => ({ ...previous, query: value.slice(0, 120) }));
  const setSelectedId = value => setDisplay(previous => ({ ...previous, selectedId: value }));
  const [detailPage, setDetailPage] = useState({ targetId: '', units: [], total: null, hasMore: false, loading: false, error: '' });
  const detailRequestRef = useRef(0);

  useEffect(() => {
    let dead = false;
    const controller = new AbortController();
    const load = async () => {
      let recognizedTargetIds = new Set();
      try {
        const response = await fetch(TARGETS_URL, { signal: controller.signal });
        if (!response.ok) throw new Error(`目标目录 HTTP ${response.status}`);
        const value = await response.json();
        if (!dead && value.schema_version === 'language-graph-targets.v1') {
          recognizedTargetIds = new Set((value.targets || []).map(row => row.id));
          setCatalog(value); setCatalogError('');
        } else if (!dead) throw new Error('目标目录版本不匹配');
      } catch (reason) {
        if (!dead && reason.name !== 'AbortError') setCatalogError(String(reason.message || reason));
      }
      try {
        const response = await fetch(CATALOG_URL, { signal: controller.signal });
        if (!response.ok) return;
        const value = await response.json();
        if (dead) return;
        const datasets = value.datasets || [];
        const explicitlyMapped = datasets.filter(row => recognizedTargetIds.has(row.language_graph_target_id || row.graph_target_id)).length;
        setHistoricalDatasetCount(datasets.length);
        setUnmappedHistoricalCount(Math.max(0, datasets.length - explicitlyMapped));
      } catch { /* 历史资源计数是辅助信息，不影响目标目录 */ }
    };
    load();
    return () => { dead = true; controller.abort(); };
  }, []);

  const targets = catalog?.targets || EMPTY_LIST;
  const units = workView?.work_units || EMPTY_LIST;
  const targetIds = new Set(targets.map(target => target.id));
  const byTarget = useMemo(() => {
    const grouped = Object.fromEntries(targets.map(target => [target.id, []]));
    for (const unit of units) {
      const id = exactTargetId(unit);
      if (grouped[id]) grouped[id].push(unit);
    }
    return grouped;
  }, [units, targets]);
  const connected = !!workView && !workViewError;
  const targetProgress = workView?.language_graph_target_progress;
  const auditTrackingAvailable = targetProgress?.audit_tracking_available === true || units.some(unit => unit.audit_status !== undefined || unit.target_evidence_level !== undefined || unit.evidence_level !== undefined);
  const aggregateFor = targetId => targetProgress?.by_target?.[targetId] || null;
  const visibleTargets = targets.filter(target => {
    const matchesDomain = domain === 'all' || target.domain_id === domain;
    const haystack = `${target.id} ${target.family_id} ${target.label} ${target.research_question}`.toLowerCase();
    return matchesDomain && haystack.includes(query.trim().toLowerCase());
  });
  const selected = visibleTargets.find(target => target.id === selectedId) || visibleTargets[0] || null;
  const selectedTargetId = selected?.id || '';
  const workViewReady = !!workView;
  const workViewFailed = Boolean(workViewError);
  const mappedTargets = targets.filter(target => (aggregateFor(target.id)?.work_units ?? (byTarget[target.id] || []).length) > 0).length;
  const mappedUnits = targetProgress?.target_bound_work_units ?? units.filter(unit => targetIds.has(exactTargetId(unit))).length;
  const unboundUnits = workView ? (targetProgress ? targetProgress.unmapped_work_units + targetProgress.unrecognized_target_work_units : Math.max(0, (workView.counts?.work_units || units.length) - mappedUnits)) : null;
  const resultUnits = targetProgress?.result_uploaded_total ?? units.filter(unit => targetIds.has(exactTargetId(unit)) && unit.result_sha).length;
  const auditUnits = targetProgress?.awaiting_audit_total ?? units.filter(unit => targetIds.has(exactTargetId(unit)) && unit.evidence_state === 'awaiting_audit').length;
  const evidenceTargets = targets.filter(target => (byTarget[target.id] || []).some(isAuditedEvidence) || (auditTrackingAvailable && aggregateFor(target.id)?.evidence_levels?.some(level => EVIDENCE_LEVELS.has(level)))).length;
  const targetCoverage = targets.length ? Math.round(mappedTargets / targets.length * 100) : 0;

  const loadTargetUnits = useCallback(async (targetId, offset = 0, append = false) => {
    if (!targetId) return;
    const requestId = ++detailRequestRef.current;
    setDetailPage(previous => {
      const keepPage = append && previous.targetId === targetId;
      return { ...previous, targetId, units: keepPage ? previous.units : [], total: keepPage ? previous.total : null, hasMore: keepPage ? previous.hasMore : false, loading: true, error: '' };
    });
    const params = new URLSearchParams({ language_graph_target_id: targetId, offset: String(offset), limit: '50' });
    try {
      const response = await fetch(`${API_BASE}/api/research/work-units?${params.toString()}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const value = await response.json();
      if (value.schema_version !== 'research_distributed_view.v1') throw new Error('执行视图版本不匹配');
      if (requestId !== detailRequestRef.current) return;
      const page = value.work_units_page;
      const received = (value.work_units || []).filter(unit => exactTargetId(unit) === targetId);
      setDetailPage(previous => ({
        targetId,
        units: append && previous.targetId === targetId ? [...previous.units, ...received] : received,
        total: page?.total ?? received.length,
        hasMore: page?.has_more === true,
        loading: false,
        error: '',
      }));
    } catch (reason) {
      if (requestId !== detailRequestRef.current) return;
      setDetailPage(previous => ({ ...previous, targetId, loading: false, error: `目标工作单明细读取失败：${String(reason.message || reason)}` }));
    }
  }, []);

  useEffect(() => {
    if (!expanded || !selectedTargetId || !workViewReady || workViewFailed) return undefined;
    loadTargetUnits(selectedTargetId, 0, false);
    return () => { detailRequestRef.current += 1; };
  }, [expanded, selectedTargetId, workViewReady, workViewFailed, loadTargetUnits]);

  const selectedPage = detailPage.targetId === selectedTargetId ? detailPage : null;
  const selectedUnits = selectedPage?.units ?? (selected ? byTarget[selected.id] || [] : EMPTY_LIST);

  return (
    <section className="lg-target-board" aria-label="语言图谱测试目标清单">
      <header className="lg-target-board-head">
        <div><span className="at-eyebrow">LANGUAGE GRAPH TEST TARGETS · REV {catalog?.revision ?? '—'}</span><h4>语言图谱测试目标</h4>
          <p>{expanded ? '每个目标可拆成冻结的分布式 Work Unit；列表把运行回执与经过审计的研究证据分开显示。' : '查看目标覆盖、结果回传与审计证据；点击展开目标清单和逐项研究设计。'}</p></div>
        <div className={'lg-target-live ' + (connected ? 'live' : workViewError ? 'offline' : '')}>{connected ? '执行视图 LIVE' : workViewError ? '执行视图不可用' : '读取目标进度…'}</div>
        <button type="button" className="lg-target-toggle" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>{expanded ? '收起目标详情' : '展开目标详情'}<span>{expanded ? '−' : '+'}</span></button>
      </header>

      {catalogError && <div className="lg-target-empty">语言图谱目标目录未能加载：{catalogError}</div>}
      {catalog && <>
        {!expanded && <div className="lg-target-compact-summary" aria-label="目标目录摘要">
          <span><b>{targets.length}</b><small>研究目标</small></span>
          <span><b>{connected ? `${mappedTargets} / ${targets.length}` : '—'}</b><small>已绑定目标</small></span>
          <span><b>{connected ? `${resultUnits} / ${auditUnits}` : '—'}</b><small>结果回传 / 待审计</small></span>
          <span><b>{connected && auditTrackingAvailable ? `${evidenceTargets} / ${targets.length}` : '—'}</b><small>有审计级别证据</small></span>
        </div>}
        {expanded && <div className="lg-target-summary-grid">
          <SummaryCard label="图谱目标" value={targets.length} note={`${catalog.domains?.length || 0} 个研究领域 · 目录已登记`} />
          <SummaryCard label="已绑定目标" value={connected ? `${mappedTargets} / ${targets.length}` : '—'} note={connected ? `目标级派发覆盖 ${targetCoverage}%` : '等待 Work Unit 接口'} tone="blue" />
          <SummaryCard label="回传 / 待审计" value={connected ? `${resultUnits} / ${auditUnits}` : '—'} note="仅统计精确绑定目标的 Work Unit" tone="amber" />
          <SummaryCard label="审计后有级别证据" value={connected && auditTrackingAvailable ? `${evidenceTargets} / ${targets.length}` : '—'} note={auditTrackingAvailable ? '要求通过审计并显式登记 E1–E3' : '当前执行投影只跟踪到待审计'} tone="green" />
          <SummaryCard label="目录中的历史数据集" value={historicalDatasetCount ?? '—'} note={unmappedHistoricalCount === null ? '历史资源目录未读取' : `${unmappedHistoricalCount} 项尚无目标 ID 回链，不计作目标完成`} tone="muted" />
        </div>}
        <div className={'lg-target-progress-wrap' + (!expanded ? ' compact' : '')}><div><b>分布式派发覆盖</b><span>{connected ? `${mappedTargets} / ${targets.length} 个目标已关联 Work Unit · ${targetCoverage}%` : '中心服务未提供可用快照'}</span></div><div className="lg-target-progress"><i style={{ width: connected ? `${targetCoverage}%` : '0%' }} /></div>
          {connected && unboundUnits > 0 && <small>另有 {unboundUnits} 个已绑定 Puzzle 的 Work Unit 没有语言图谱目标 ID，保留为未归属任务。</small>}
        </div>
        {expanded && <div className="lg-target-caveat">派发覆盖率只表示目标是否接入分布式任务，不是科研完成百分比。任务成功、历史数据集数量和图谱连线都不会自动升级为复现、预测或机制证据。</div>}

        {expanded && <div className="lg-target-browser">
          <aside className="lg-target-list">
            <div className="lg-target-list-tools"><input value={query} onChange={event => setQuery(event.target.value)} placeholder="搜索目标、关系或问题" aria-label="搜索语言图谱目标" />
              <button type="button" className={domain === 'all' ? 'on' : ''} onClick={() => setDomain('all')}>全部 · {targets.length}</button>
            </div>
            <div className="lg-target-domains">{(catalog.domains || []).map(item => <button type="button" key={item.id} className={domain === item.id ? 'on' : ''} onClick={() => setDomain(item.id)}>{item.label}<span>{targets.filter(target => target.domain_id === item.id).length}</span></button>)}</div>
            <div className="lg-target-list-items">{visibleTargets.map(target => {
              const targetUnits = byTarget[target.id] || [];
              const aggregate = aggregateFor(target.id);
              const targetUnitCount = aggregate?.work_units ?? targetUnits.length;
              const outputCount = aggregate?.result_uploaded ?? targetUnits.filter(unit => !!unit.result_sha).length;
              return <button type="button" key={target.id} className={'lg-target-row ' + (selected?.id === target.id ? 'on' : '')} onClick={() => setSelectedId(target.id)}>
                <span><b>{target.label}</b><small>{target.id} · {target.family_id}</small></span>
                <TargetStatus units={targetUnits} aggregate={aggregate} connected={connected} />
                <small className="lg-target-row-count">{connected ? `${targetUnitCount} 个任务 · ${outputCount} 个回传` : '进度未接入'}</small>
              </button>;
            })}</div>
          </aside>
          <TargetDetails target={selected} units={selectedUnits} detailPage={selectedPage} onRefresh={() => loadTargetUnits(selectedTargetId, 0, false)} onLoadMore={() => loadTargetUnits(selectedTargetId, selectedUnits.length, true)} aggregate={selected ? aggregateFor(selected.id) : null} connected={connected} auditTrackingAvailable={auditTrackingAvailable} onGo={onGo} milestones={catalog.milestones || []} />
        </div>}
      </>}
    </section>
  );
}
