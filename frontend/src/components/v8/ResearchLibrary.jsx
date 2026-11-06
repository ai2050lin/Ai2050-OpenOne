import { useEffect, useRef, useState } from 'react';
import { authJson, useApi, ADMIN_LIVE } from '../app/rdc_fusion/api.js';
import { useAuth } from '../app/rdc_fusion/authContext.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import { ExtraPackPrompt } from '../open/DataBar.jsx';
import ResearchDisclosure from './ResearchDisclosure.jsx';
import ResearchFragments from './ResearchFragments.jsx';
import { EVIDENCE_LABEL } from './researchFragmentUtil.js';

const ARTIFACT_PAGE_SIZE = 50;
const GRAPH_PAGE_SIZE = 100;
const ARTIFACT_KIND = {
  local_report: '研究报告',
  distributed_result: '分片元数据',
  census_result: '普查元数据',
  derived_snapshot: '派生图谱',
  contributed_unreviewed: '未审核贡献',
  registered: '管理员登记',
};

function ArtifactDialog({ sha, detail, onClose }) {
  const ref = useRef(null);
  useEffect(() => {
    const dialog = ref.current;
    if (dialog && !dialog.open) dialog.showModal();
    return () => dialog?.close();
  }, []);

  return <dialog ref={ref} className="v8-artifact-dialog" aria-label={`原始研究产物 ${sha.slice(0, 12)}`}
    onCancel={event => { event.preventDefault(); onClose(); }}
    onClick={event => { if (event.target === event.currentTarget) onClose(); }}>
    <header className="v6-toolbar">
      <h3>原始研究产物</h3>
      <button className="fw-tbtn" autoFocus onClick={onClose}>关闭产物详情</button>
    </header>
    <p className="v8-note">SHA256 {sha}</p>
    {detail.error ? <Empty>{detail.error}<ExtraPackPrompt pack={detail.missingPack} /></Empty>
      : <pre className="v8-artifact-json">{detail.data ? JSON.stringify(detail.data, null, 2) : '读取…'}</pre>}
  </dialog>;
}

function ArtifactTable({ rows, onArtifact }) {
  if (!rows.length) return <Empty>尚无归档产物。</Empty>;
  return <div className="v8-matrix-wrap">
    <table className="v6-table">
      <thead><tr><th>来源</th><th>模型身份</th><th>归档 / 时间</th><th>版本 / 回查</th></tr></thead>
      <tbody>{rows.map(row => <tr key={row.sha}>
        <td>{row.source}</td>
        <td title={row.model}>{row.model?.slice(0, 60) || '未记录'}</td>
        <td>{ARTIFACT_KIND[row.kind] || row.kind}<small>{new Date(row.created * 1000).toLocaleString()}</small></td>
        <td><button className="fw-tbtn" onClick={() => onArtifact(row.sha)}>{row.sha.slice(0, 12)}</button></td>
      </tr>)}</tbody>
    </table>
  </div>;
}

function GraphVersions({ graph, offset, setOffset, onArtifact }) {
  const objects = graph.data?.objects || [];

  return <>
    {graph.error ? <Empty>{graph.error}<ExtraPackPrompt pack={graph.missingPack} /></Empty>
      : !graph.data ? <Empty>读取图谱版本…</Empty> : objects.length ? <div className="v8-matrix-wrap">
      <table className="v6-table">
        <thead><tr><th>对象</th><th>类型 / 关系</th><th>证据声明</th><th>产物版本</th></tr></thead>
        <tbody>{objects.map(object => <tr key={`${object.sha}:${object.id}`}>
          <td>{object.id}</td>
          <td>{object.body.relation || object.type}</td>
          <td>{EVIDENCE_LABEL[object.body.evidence.level] || object.body.evidence.level}</td>
          <td><button className="fw-tbtn" onClick={() => onArtifact(object.sha)}>{object.sha.slice(0, 12)}</button></td>
        </tr>)}</tbody>
      </table>
    </div> : <Empty>暂无图谱对象。</Empty>}
    <div className="v6-toolbar">
      <button className="fw-tbtn" disabled={!offset} onClick={() => setOffset(Math.max(0, offset - GRAPH_PAGE_SIZE))}>上一页</button>
      <button className="fw-tbtn" disabled={Boolean(graph.error) || objects.length < GRAPH_PAGE_SIZE} onClick={() => setOffset(offset + GRAPH_PAGE_SIZE)}>下一页</button>
    </div>
  </>;
}

export default function ResearchLibrary({ compact = false }) {
  const [artifactsVisited, setArtifactsVisited] = useState(false);
  const [versionsVisited, setVersionsVisited] = useState(false);
  const [artifactOffset, setArtifactOffset] = useState(0);
  const [graphOffset, setGraphOffset] = useState(0);
  const [sha, setSha] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const overview = useApi('/api/research-library/overview');
  const listed = useApi(!compact && artifactsVisited ? `/api/research-library/artifacts?offset=${artifactOffset}&limit=${ARTIFACT_PAGE_SIZE}` : null);
  const graph = useApi(!compact && versionsVisited ? `/api/research-library/graph?offset=${graphOffset}&limit=${GRAPH_PAGE_SIZE}` : null);
  const detail = useApi(sha ? `/api/research-library/artifacts/${sha}` : null);
  const { user } = useAuth();
  const canWrite = ADMIN_LIVE && user?.role === 'admin';
  const counts = overview.data;
  const stats = [
    ['归档产物版本', counts?.artifacts],
    ['特征卡版本', counts?.graph_revisions?.feature],
    ['图谱边版本', counts?.graph_revisions?.edge],
  ];

  function refresh() {
    overview.reload();
    if (!compact) {
      listed.reload();
      graph.reload();
    }
  }

  async function ingest(event) {
    const input = event.currentTarget;
    const file = input.files?.[0];
    if (!file) return;
    setError('');
    setBusy(true);
    try {
      if (file.size > 16 * 1024 * 1024) throw Error('登记文件不超过 16 MB。原场应作为独立产物存储。');
      const result = await authJson('/api/research-library/ingest', { method: 'POST', body: JSON.parse(await file.text()) });
      setSha(result.sha256);
      refresh();
    } catch (reason) {
      setError(reason.message);
    } finally {
      setBusy(false);
      input.value = '';
    }
  }

  async function sync() {
    setBusy(true);
    setError('');
    try {
      const result = await authJson('/api/research-library/sync', { method: 'POST' });
      if (result.errors?.length) setError(`${result.errors.length} 项材料未归档。请检查文法与产物格式。`);
      refresh();
    } catch (reason) {
      setError(reason.message);
    } finally {
      setBusy(false);
    }
  }

  return <section className={compact ? 'v8-card' : 'v8-route'}>
    <header className={compact ? 'v6-toolbar' : 'v8-overview-heading'}>
      {compact ? <h3>持续积累的研究库</h3> : <div>
        <h3>研究归档</h3>
        <p className="v8-note">研究库累计版本数</p>
      </div>}
      <button className="fw-tbtn" onClick={refresh}>刷新</button>
    </header>
    {overview.error ? <Empty>{overview.error}</Empty> : !counts ? <Empty>读取研究库…</Empty> :
      compact ? <div className="v8-library-stats">{stats.map(([label, count]) => <span key={label}><b>{count}</b>{label}</span>)}</div>
        : <div className="v8-direction-cards v8-overview-cards" role="group" aria-label="研究库累计版本">
          {stats.map(([label, count]) => <article className="v8-card v8-evidence-stat" key={label}>
            <h3>{label}</h3><strong>{Number.isFinite(count) ? count.toLocaleString('zh-CN') : '—'}</strong>
          </article>)}
        </div>}
    <ResearchDisclosure title={compact ? '来源与最近积累' : '研究产物与来源'} onFirstOpen={() => setArtifactsVisited(true)}>
      {compact ? <ArtifactTable rows={counts?.recent || []} onArtifact={setSha} />
        : <>
            {listed.error ? <Empty>{listed.error}<ExtraPackPrompt pack={listed.missingPack} /></Empty>
              : !listed.data ? <Empty>读取研究产物…</Empty> : <ArtifactTable rows={listed.data.artifacts} onArtifact={setSha} />}
            <div className="v6-toolbar">
              <button className="fw-tbtn" disabled={!artifactOffset} onClick={() => setArtifactOffset(Math.max(0, artifactOffset - ARTIFACT_PAGE_SIZE))}>较新版本</button>
              {listed.data && <span className="v8-note">{listed.data.total ? artifactOffset + 1 : 0}–{Math.min(artifactOffset + ARTIFACT_PAGE_SIZE, listed.data.total)} / {listed.data.total}</span>}
              <button className="fw-tbtn" disabled={!listed.data || Boolean(listed.error) || artifactOffset + ARTIFACT_PAGE_SIZE >= listed.data.total}
                onClick={() => setArtifactOffset(artifactOffset + ARTIFACT_PAGE_SIZE)}>较早版本</button>
            </div>
          </>}
      {counts?.sync && <p className="v8-note">最近同步 {new Date(counts.sync.completed_at * 1000).toLocaleString()} · {counts.sync.errors.length} 项未归档</p>}
    </ResearchDisclosure>
    {!compact && <ResearchDisclosure title="图谱证据">
      {counts?.note && <p className="v8-note">{counts.note}</p>}
      <div className="v8-route">
        <ResearchDisclosure title="特征卡与反例"><ResearchFragments onArtifact={setSha} /></ResearchDisclosure>
        <ResearchDisclosure title="特征与关系的版本记录" onFirstOpen={() => setVersionsVisited(true)}>
          <GraphVersions graph={graph} offset={graphOffset} setOffset={setGraphOffset} onArtifact={setSha} />
        </ResearchDisclosure>
        <ResearchDisclosure title="成果登记说明">
          <p className="v8-note">新增或更新的研究报告与真实分布式结果会自动归档。登记特征或关系时需提供模型、运行、反例与原始证据，历史版本可回查。</p>
        </ResearchDisclosure>
      </div>
    </ResearchDisclosure>}
    {canWrite && !compact && <ResearchDisclosure title="研究库管理">
      <div className="v6-toolbar">
        <button className="fw-tbtn" disabled={busy} onClick={sync}>同步已有成果</button>
        <label className="fw-tbtn">登记研究产物<input type="file" accept=".json" hidden disabled={busy} onChange={ingest} /></label>
      </div>
      <p className="v8-note">同步会归档新增报告与真实分布式结果。登记特征或关系时请提供模型、运行、反例与原始证据。</p>
      {error && <p className="v6-note warn">{error}</p>}
    </ResearchDisclosure>}
    {sha && <ArtifactDialog sha={sha} detail={detail} onClose={() => setSha('')} />}
  </section>;
}
