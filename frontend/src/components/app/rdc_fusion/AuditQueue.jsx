/* 研发 · ④ 审计队列：逐份结果看自动检查（设计一致 / 真实运行 / 产物齐全 / 金丝雀 / 行为门 / 分片 /
   Work Unit 绑定），人工裁决 通过 / 驳回 / 需重跑。自动检查有 fail 时服务端拒绝「通过」。
   审计只决定执行产物能否进入合并统计；Evidence 等级仍在 Research OS 登记。 */
import { useState } from 'react';
import { fmtAgoTs, postJson, useAdminToken, useApi } from './api.js';
import { AdminTokenField, Card, Empty, RefreshButton, StatusPill, Tabs } from './ui.jsx';

const FILTERS = [
  { id: 'pending', label: '待审' }, { id: 'needs_rerun', label: '需重跑' },
  { id: 'rejected', label: '已驳回' }, { id: 'passed', label: '已通过' }, { id: 'all', label: '全部' },
];

export default function AuditQueue() {
  const [filter, setFilter] = useState('pending');
  const queue = useApi(`/api/audit/queue?status=${filter}`, { online: true });
  const [sel, setSel] = useState('');
  const [token, setToken] = useAdminToken();
  const [reviewer, setReviewer] = useState('');
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState('');
  const [message, setMessage] = useState(null);
  const items = queue.data?.items || [];
  const counts = queue.data?.counts || {};
  const item = items.find(i => i.sha === sel) || items[0];

  const decide = async status => {
    if (!item) return;
    setBusy(status); setMessage(null);
    try {
      await postJson(`/api/audit/${item.sha}`, { status, reviewer, note }, { adminToken: token });
      setMessage({ ok: true, text: `${item.sha.slice(0, 10)} → ${status}` });
      setNote('');
      queue.reload();
    } catch (reason) {
      setMessage({ ok: false, text: reason.status === 401 ? '管理令牌无效或未填写' : String(reason.message || reason) });
    }
    setBusy('');
  };

  return (
    <div className="v6-audit">
      <div className="v6-audit-list">
        <Tabs label="审计状态" value={filter} onChange={value => { setFilter(value); setSel(''); }}
          tabs={FILTERS.map(f => ({ ...f, badge: f.id !== 'all' && counts[f.id] ? counts[f.id] : null }))} />
        <div className="v6-toolbar"><RefreshButton apis={[queue]} /></div>
        {!queue.data && <Empty>{queue.error ? `审计队列未接入（${queue.error}）` : '读取…'}</Empty>}
        {queue.data && !items.length && <Empty>该状态下没有结果。</Empty>}
        {items.length > 0 && (
          <table className="v6-mini-table v6-audit-table">
            <thead><tr><th>结果</th><th>模板</th><th>节点</th><th>模型 / 精度</th><th>分片</th><th>自动检查</th><th>上传</th></tr></thead>
            <tbody>
              {items.map(i => (
                <tr key={i.sha} className={item?.sha === i.sha ? 'on' : ''} onClick={() => setSel(i.sha)} tabIndex={0}
                  onKeyDown={event => { if (event.key === 'Enter') setSel(i.sha); }}>
                  <th scope="row" className="mono">{i.sha.slice(0, 10)}</th>
                  <td className="mono">{i.tm_id}</td>
                  <td>{i.node_name || i.node_id}</td>
                  <td className="mono">{i.model_id || '—'} · {i.precision || '—'}</td>
                  <td className="mono">{i.shard || '—'}</td>
                  <td><StatusPill status={i.auto_verdict} /></td>
                  <td>{fmtAgoTs(i.uploaded_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {queue.data && <p className="v6-note">{queue.data.policy}</p>}
      </div>
      <aside className="v6-audit-detail">
        <Card title="审计详情" extra={<AdminTokenField token={token} onChange={setToken} />}>
          {!item && <Empty>选择左侧一份结果。</Empty>}
          {item && (
            <>
              <div className="v6-spec-meta">
                <span>结果 <b className="mono">{item.sha.slice(0, 16)}…</b></span>
                <span>{item.kind === 'real' ? '真实运行' : item.kind}</span>
                <span>状态 <StatusPill status={item.status === 'pending' ? 'pending' : item.status} label={item.status === 'pending' ? '待审' : undefined} /></span>
              </div>
              <ul className="v6-checks">
                {item.checks.map(c => (
                  <li key={c.id}><StatusPill status={c.status} /><b>{c.label}</b><span>{c.detail}</span></li>
                ))}
              </ul>
              {item.audit && <p className="v6-note">上次裁决：{item.audit.status} · {item.audit.reviewer || '匿名'} · {item.audit.note || '无备注'}</p>}
              <div className="v6-decide">
                <input placeholder="审计人" value={reviewer} onChange={event => setReviewer(event.target.value)} />
                <textarea rows={2} placeholder="备注（驳回/重跑请写明原因）" value={note} onChange={event => setNote(event.target.value)} />
                <div>
                  <button type="button" className="fw-tbtn primary" disabled={!!busy || item.auto_verdict === 'fail'}
                    title={item.auto_verdict === 'fail' ? '自动检查有未通过项，不能标记通过' : ''} onClick={() => decide('passed')}>通过</button>
                  <button type="button" className="fw-tbtn" disabled={!!busy} onClick={() => decide('needs_rerun')}>需重跑</button>
                  <button type="button" className="fw-tbtn v6-danger" disabled={!!busy} onClick={() => decide('rejected')}>驳回</button>
                </div>
                {message && <div className={message.ok ? 'v6-ok' : 'fw-ai-err'}>{message.text}</div>}
              </div>
            </>
          )}
        </Card>
      </aside>
    </div>
  );
}
