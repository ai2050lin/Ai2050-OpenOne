/* 数据管理页（网站 ?page=data，管理员）：服务器上的存储占用与磁盘剩余、各类数据计数、按账号汇总的上传，以及逐条上传记录
   （模式族结果、重复上传、普查分片）与详情。只读：本页不删除、不修改任何数据。
   接口 /api/admin/data/*（data_admin.v1）只在打开页面和点击时连接中心；计数只描述服务器上有什么，不表示证据晋级。 */
import { useEffect, useState } from 'react';
import { authJson } from './api.js';
import { useAuth } from './authContext.js';
import { SOURCE_LABEL, entries, fmtBytes, fmtTime, ownerLabel, uploadState, uploadsQuery } from './dataAdminUtil.js';
import { Empty, StatusPill } from './ui.jsx';
import './accounts.css';
import './dataAdmin.css';

const PAGE = 50;
const NO_FILTER = { source: 'all', user: '', kind: '', status: '', q: '' };

function Storage({ s }) {
  const max = Math.max(1, ...s.parts.map(p => p.bytes));
  const disk = s.disk;
  return (
    <section className="dam-card" aria-label="存储占用">
      <h2>存储占用</h2>
      <p className="dam-sub">数据目录 <code>{s.data_dir}</code> 共 <b>{fmtBytes(s.total_bytes)}</b>
        {disk && <>；所在磁盘已用 {fmtBytes(disk.used)} / {fmtBytes(disk.total)}，剩余 <b className={disk.free < disk.total * 0.1 ? 'dam-low' : ''}>{fmtBytes(disk.free)}</b></>}
        <small>（目录统计每 2 分钟最多重算一次，统计于 {fmtTime(s.computed_at)}）</small></p>
      {disk && (
        <div className="dam-disk" role="img" aria-label={`磁盘已用 ${Math.round(disk.used / disk.total * 100)}%`}>
          <span style={{ width: `${disk.used / disk.total * 100}%` }} />
        </div>
      )}
      <table className="dam-parts">
        <tbody>
          {s.parts.map(p => (
            <tr key={p.id}>
              <th scope="row">{p.label}</th>
              <td className="dam-bar-cell"><span className="dam-bar" style={{ width: `${p.bytes / max * 100}%` }} /></td>
              <td className="num">{fmtBytes(p.bytes)}</td>
              <td className="num muted">{p.files.toLocaleString()} 个文件</td>
            </tr>
          ))}
        </tbody>
      </table>
      {s.native && <p className="dam-sub">原生全场与普查合并目录 <code>{s.native.dir}</code>：普查合并结果 {fmtBytes(s.native.census_bytes)}
        {s.native.disk && <>，所在磁盘剩余 {fmtBytes(s.native.disk.free)}</>}</p>}
    </section>
  );
}

function Counts({ c }) {
  const tile = (label, value, sub) => <div><dt>{label}</dt><dd>{value}</dd>{sub && <small>{sub}</small>}</div>;
  const census = c.census;
  return (
    <section className="dam-card" aria-label="数据计数">
      <h2>数据计数</h2>
      <dl className="dam-tiles">
        {tile('账号', c.users.total, `管理员 ${c.users.admin} · 停用 ${c.users.disabled}`)}
        {tile('节点', c.nodes.total, `在线 ${c.nodes.online} · 无主 ${c.nodes.unowned}`)}
        {tile('模板', Object.values(c.templates).reduce((a, b) => a + b, 0), entries(c.templates).map(([k, v]) => `${k} ${v}`).join(' · '))}
        {tile('分片任务', Object.values(c.tasks).reduce((a, b) => a + b, 0), entries(c.tasks).map(([k, v]) => `${k} ${v}`).join(' · '))}
        {tile('模式族结果', c.results.total, `${fmtBytes(c.results.bytes)} · 待审计 ${c.results.unaudited} · 重复上传 ${c.results.replicas}`)}
        {census && tile('普查分片结果', Object.values(census.results).reduce((a, b) => a + b, 0),
          `${fmtBytes(census.bytes)} · ${entries(census.results).map(([k, v]) => `${uploadState(k).label} ${v}`).join(' · ')}`)}
        {census && tile('普查已计入 token', (census.tokens || 0).toLocaleString(), `进行中的租约 ${census.active_leases}`)}
      </dl>
      <p className="dam-sub">结果类型：{entries(c.results.by_kind).map(([k, v]) => `${k} ${v}`).join(' · ') || '—'}
        {census && <>；普查：{entries(census.by_kind).map(([k, v]) => `${k} ${v}`).join(' · ') || '—'}</>}
        。审计与校验只说明执行产物可信、可以合并，不表示证据晋级。</p>
    </section>
  );
}

function ByUser({ rows, onPick, picked }) {
  return (
    <section className="dam-card" aria-label="按账号汇总">
      <h2>按账号汇总</h2>
      <p className="dam-sub">上传按节点的所有者归到账号；没有绑定账号的节点合为「无主节点」。按数据量排序。</p>
      <div className="acm-table-wrap">
        <table className="acm-table">
          <thead><tr><th>账号</th><th className="num">节点</th><th className="num">模式族结果</th><th className="num">重复上传</th>
            <th className="num">普查分片</th><th className="num">数据量</th><th>最近上传</th><th /></tr></thead>
          <tbody>
            {rows.map(u => {
              const key = u.user_id || '-';
              return (
                <tr key={key} className={picked === key ? 'open' : ''}>
                  <th scope="row"><b>{u.display_name || u.username}</b>
                    <small className="mono">{u.user_id ? u.username : '没有绑定账号'}{u.role === 'admin' ? ' · 管理员' : ''}{u.disabled ? ' · 已停用' : ''}</small></th>
                  <td className="num">{u.nodes}{u.online ? <small> · 在线 {u.online}</small> : null}</td>
                  <td className="num">{u.results}</td>
                  <td className="num">{u.replicas}</td>
                  <td className="num" title={`校验通过 ${u.census_accepted} · 驳回 ${u.census_rejected} · 复算不一致 ${u.census_disputed}`}>
                    {u.census}{u.census ? <small> · 通过 {u.census_accepted}</small> : null}</td>
                  <td className="num">{fmtBytes(u.results_bytes + u.census_bytes)}{u.synthetic ? <small className="dam-syn"> · 合成 {u.synthetic}</small> : null}</td>
                  <td>{fmtTime(u.last_upload)}</td>
                  <td><button type="button" className={'fw-tbtn' + (picked === key ? ' on' : '')}
                              disabled={!u.results && !u.census && !u.replicas} onClick={() => onPick(picked === key ? '' : key)}>
                    {picked === key ? '看全部' : '看上传'}</button></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Kv({ rows }) {
  return <dl className="dam-kv">{rows.filter(r => r[1] !== undefined && r[1] !== null && r[1] !== '').map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>;
}

function Detail({ item }) {
  const [d, setD] = useState(null);
  const [err, setErr] = useState('');
  useEffect(() => {
    let dead = false;
    authJson(`/api/admin/data/uploads/${item.source}/${encodeURIComponent(item.id)}`)
      .then(x => { if (!dead) setD(x); }).catch(e => { if (!dead) setErr(e.message); });
    return () => { dead = true; };
  }, [item.source, item.id]);
  if (err) return <div className="acm-err block">{err}</div>;
  if (!d) return <Empty>读取详情…</Empty>;
  const node = d.node;
  const nodeText = node ? `${node.name || node.node_id}（${node.gpu || '未知 GPU'}${node.dtype ? ` · ${node.dtype}` : ''}）` : item.node_id;
  if (d.source === 'census') {
    return (
      <div className="dam-detail">
        <Kv rows={[['作业', `${d.job_id}（${d.job_status || '—'}）`], ['分片', d.shard], ['模型', d.model_key], ['类型', d.kind],
          ['计入 token', d.n_tokens?.toLocaleString()], ['上传字节', fmtBytes(d.bytes)], ['领取', fmtTime(d.claimed_at)], ['提交', fmtTime(d.uploaded_at)],
          ['节点', nodeText], ['结果文件', d.object_present ? '在服务器上' : '没有保存（只保存校验通过的分片）'], ['sha256', <code key="s">{d.sha}</code>],
          ['处理', d.reviewed_by ? `${d.reviewed_by}：${d.review_note || ''}` : undefined]]} />
        {d.validation?.errors?.length > 0 && <div className="acm-err block">校验问题：{d.validation.errors.join('；')}</div>}
        {d.replica_checks.length > 0 && (
          <p className="dam-sub">复算比较：{d.replica_checks.map(p => `${p.result_a} ↔ ${p.result_b} ${p.ok ? '一致' : '不一致'}`).join('；')}</p>
        )}
      </div>
    );
  }
  return (
    <div className="dam-detail">
      <Kv rows={[['模式族', `${d.tm_id}${d.tm_version ? ` v${d.tm_version}` : ''}${d.tm_name ? ` · ${d.tm_name}` : ''}`], ['类型', d.kind],
        ['模型', `${d.model_id}${d.model_rev ? ` @ ${d.model_rev}` : ''}`], ['精度', d.upload_meta?.precision], ['种子', d.seed],
        ['大小', fmtBytes(d.bytes)], ['上传', fmtTime(d.uploaded_at)], ['下载次数', d.downloads], ['节点', nodeText],
        ['金丝雀', d.canary?.status], ['审计', d.audit ? `${uploadState(d.audit.status).label}${d.audit.reviewer ? ` · ${d.audit.reviewer}` : ''}` : '待审计'],
        ['分片任务', d.task ? `${d.task.task_id}${d.task.shard ? ` · ${d.task.shard}` : ''}` : undefined],
        ['原始数据', d.object_present ? '在服务器上' : '缺失'], ['sha256', <code key="s">{d.sha}</code>]]} />
      {d.files.length > 0 && (
        <table className="dam-files">
          <thead><tr><th>文件</th><th className="num">大小</th><th>sha256</th></tr></thead>
          <tbody>{d.files.map(f => <tr key={f.path}><td className="mono">{f.path}</td><td className="num">{fmtBytes(f.size)}</td><td className="mono muted">{String(f.sha256 || '').slice(0, 16)}…</td></tr>)}</tbody>
        </table>
      )}
      {d.replicas.length > 0 && <p className="dam-sub">另有 {d.replicas.length} 次字节相同的重复上传：{d.replicas.map(r => `${r.node_id.slice(0, 8)} ${fmtTime(r.uploaded_at)}（${uploadState(r.note).label}）`).join('；')}</p>}
      {d.audit?.note && <p className="dam-sub">审计备注：{d.audit.note}</p>}
      <details className="dam-json"><summary>摘要字段（{d.summary_keys.length} 个，{d.summary_chars.toLocaleString()} 字符）</summary>
        {d.summary ? <pre>{JSON.stringify(d.summary, null, 2)}</pre> : <p className="dam-sub">摘要超过 20,000 字符，只列字段：{d.summary_keys.join('、')}</p>}
      </details>
    </div>
  );
}

function Uploads({ users, filter, setFilter, nonce }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState('');
  // 翻页与展开的行只属于当前筛选：筛选一变就回到第一页、收起详情
  const fkey = JSON.stringify(filter);
  const [page, setPage] = useState({ fkey, offset: 0, open: '' });
  const cur = page.fkey === fkey ? page : { fkey, offset: 0, open: '' };
  const { offset, open } = cur;
  const setOffset = fn => setPage({ ...cur, offset: fn(offset), open: '' });
  const setOpen = v => setPage({ ...cur, open: v });
  const [q, setQ] = useState(filter.q);
  useEffect(() => {
    let dead = false;
    authJson(`/api/admin/data/uploads?${uploadsQuery(filter, offset, PAGE)}`)
      .then(x => { if (!dead) { setData(x); setErr(''); } }).catch(e => { if (!dead) setErr(e.message); });
    return () => { dead = true; };
  }, [filter, offset, nonce]);
  const up = patch => setFilter(f => ({ ...f, ...patch }));
  const items = data?.items || [];
  return (
    <section className="dam-card" aria-label="上传记录">
      <h2>上传记录</h2>
      <form className="acm-tools" onSubmit={e => { e.preventDefault(); up({ q: q.trim() }); }}>
        <select value={filter.source} aria-label="来源" onChange={e => up({ source: e.target.value })}>
          <option value="all">全部来源</option>
          {Object.entries(SOURCE_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
        <select value={filter.user} aria-label="账号" onChange={e => up({ user: e.target.value })}>
          <option value="">全部账号</option>
          {users.map(u => <option key={u.user_id || '-'} value={u.user_id || '-'}>{u.user_id ? (u.display_name || u.username) : '无主节点'}</option>)}
        </select>
        <select value={filter.kind} aria-label="类型" onChange={e => up({ kind: e.target.value })}>
          <option value="">全部类型</option><option value="real">real</option><option value="synthetic">synthetic</option>
        </select>
        <select value={filter.status} aria-label="状态" onChange={e => up({ status: e.target.value })}>
          <option value="">全部状态</option>
          {entries(data?.statuses).map(([k, v]) => <option key={k} value={k}>{uploadState(k).label}（{v}）</option>)}
          {filter.status && !data?.statuses?.[filter.status] && <option value={filter.status}>{uploadState(filter.status).label}</option>}
        </select>
        <input type="search" value={q} placeholder="搜索 sha、模式族 / 作业、分片、模型、节点、账号，回车" aria-label="搜索上传"
               onChange={e => setQ(e.target.value)} onBlur={() => q.trim() !== filter.q && up({ q: q.trim() })} />
        {fkey !== JSON.stringify(NO_FILTER) && <button type="button" className="fw-tbtn" onClick={() => { setQ(''); setFilter(NO_FILTER); }}>清除筛选</button>}
        {data && <span className="acm-shown">{data.total.toLocaleString()} 条 · {fmtBytes(data.bytes)}</span>}
      </form>
      {err && <div className="acm-err block" role="alert">{err}</div>}
      {!data && !err && <Empty>读取上传记录…</Empty>}
      {data && (
        <div className="acm-table-wrap">
          <table className="acm-table dam-uploads">
            <thead><tr><th>时间</th><th>来源</th><th>内容</th><th>模型</th><th>账号 · 节点</th><th>类型</th><th className="num">大小</th><th>状态</th></tr></thead>
            <tbody>
              {items.map(x => {
                const st = uploadState(x.status);
                const isOpen = open === `${x.source}:${x.id}`;
                return [
                  <tr key={`${x.source}:${x.id}`} className={'dam-row' + (isOpen ? ' open' : '')} tabIndex={0} aria-expanded={isOpen}
                      onClick={() => setOpen(isOpen ? '' : `${x.source}:${x.id}`)}
                      onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpen(isOpen ? '' : `${x.source}:${x.id}`); } }}>
                    <td>{fmtTime(x.at)}</td>
                    <td>{SOURCE_LABEL[x.source]}</td>
                    <td><b>{x.subject || '—'}</b>{x.subject_detail && <small className="mono"> · {x.subject_detail}</small>}</td>
                    <td className="mono">{x.model || '—'}</td>
                    <td>{ownerLabel(x)}<small className="mono"> · {x.node_name || x.node_id.slice(0, 8)}</small></td>
                    <td>{x.kind === 'synthetic' ? <span className="dam-syn">synthetic</span> : x.kind}</td>
                    <td className="num">{fmtBytes(x.bytes)}</td>
                    <td><StatusPill status={st.status} label={st.label} />{x.canary === 'drift' && <StatusPill status="drift" label="金丝雀漂移" />}</td>
                  </tr>,
                  isOpen && <tr key={`${x.source}:${x.id}:d`} className="acm-panel-row"><td colSpan={8}><Detail item={x} /></td></tr>,
                ];
              })}
            </tbody>
          </table>
          {!items.length && <Empty>没有符合条件的上传</Empty>}
        </div>
      )}
      {data && data.total > PAGE && (
        <div className="dam-pager">
          <button type="button" className="fw-tbtn" disabled={offset === 0} onClick={() => setOffset(o => Math.max(0, o - PAGE))}>上一页</button>
          <span>{offset + 1}–{Math.min(offset + PAGE, data.total)} / {data.total.toLocaleString()}</span>
          <button type="button" className="fw-tbtn" disabled={offset + PAGE >= data.total} onClick={() => setOffset(o => o + PAGE)}>下一页</button>
        </div>
      )}
    </section>
  );
}

export default function DataAdmin() {
  const { user, guest, enterAuth } = useAuth();
  const isAdmin = !!user && !guest && user.role === 'admin';
  const [data, setData] = useState(null);
  const [err, setErr] = useState('');
  const [nonce, setNonce] = useState(0);
  const [refresh, setRefresh] = useState(false);
  const [filter, setFilter] = useState(NO_FILTER);

  useEffect(() => {
    if (!isAdmin) return undefined;
    let dead = false;
    authJson(`/api/admin/data/overview${refresh ? '?refresh=true' : ''}`)
      .then(d => { if (!dead) { setData(d); setErr(''); } }).catch(e => { if (!dead) setErr(e.message); });
    return () => { dead = true; };
  }, [nonce, isAdmin, refresh]);

  if (!isAdmin) {
    return (<Empty action={(!user || guest) && <button type="button" className="fw-tbtn primary" onClick={enterAuth}>登录</button>}>
      {user && !guest ? '数据管理只对管理员开放；当前账号是成员。' : '数据管理需要管理员账号登录。'}
    </Empty>);
  }
  return (
    <div className="acm dam">
      <header className="acm-head">
        <div>
          <h1>数据管理</h1>
          <p>服务器上有哪些数据、占了多少空间、每个账号上传了什么。本页只读，不删除、不修改任何数据；审计与复算不一致在桌面客户端的研发工位处理。</p>
        </div>
        <button type="button" className="fw-tbtn" onClick={() => { setRefresh(true); setNonce(n => n + 1); }}>刷新</button>
      </header>
      {err && <div className="acm-err block" role="alert">{err}</div>}
      {!data && !err && <Empty>读取服务器数据…</Empty>}
      {data && (<>
        <Storage s={data.storage} />
        <Counts c={data.counts} />
        <ByUser rows={data.by_user} picked={filter.user} onPick={k => setFilter(f => ({ ...f, user: k }))} />
        <Uploads users={data.by_user} filter={filter} setFilter={setFilter} nonce={nonce} />
      </>)}
    </div>
  );
}
