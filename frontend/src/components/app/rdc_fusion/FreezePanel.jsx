/* 研发 · ② 试跑与冻结：先冻结「试跑版」（每格少量样本、单分片）让节点跑行为门，
   看逐格 top-1 准确率；达标后冻结「正式模板」（完整语料 + 分片）。冻结需管理令牌。
   冻结 = 语料与 runner 写入 design_sha，之后不可变；文法改动后重新冻结会升 version。 */
import { useState } from 'react';
import { postJson, useAdminToken, useApi } from './api.js';
import { AdminTokenField, Card, Empty, GateChain, StatusPill } from './ui.jsx';

function CopyCommand({ cmd }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="v6-cmd">
      <code>{cmd}</code>
      <button type="button" className="fw-tbtn" onClick={() => {
        try { navigator.clipboard.writeText(cmd); } catch { /* 剪贴板不可用 */ }
        setCopied(true); window.setTimeout(() => setCopied(false), 1400);
      }}>{copied ? '✓ 已复制' : '复制'}</button>
    </div>
  );
}

export function NodeCommand() {
  const cmd = 'python deploy/node_agent.py --server http://<center>:5010 run --once --model-path <HF模型目录> --model-id <model_id> --model-rev <快照> --dtype bf16';
  return (
    <div className="v6-node-cmd">
      <small>在 GPU 节点执行（先 register 一次）；中心按「最少完成次数」分配分片，节点把 runner 的 suffstats / behavior / canary 回传：</small>
      <CopyCommand cmd={cmd} />
    </div>
  );
}

function Report({ tm }) {
  const report = useApi(tm ? `/api/grammar/${encodeURIComponent(tm)}/report` : null, { interval: 15000 });
  if (!tm) return null;
  if (!report.data) return <Empty>{report.error ? `报告不可用（${report.error}）` : '读取执行报告…'}</Empty>;
  const r = report.data;
  return (
    <div className="v6-report">
      <div className="v6-spec-meta">
        <span>模板 <b className="mono">{r.tm_id}</b></span><span>{r.mode === 'pilot' ? '试跑版' : '正式版'}</span>
        <span>样本 <b>{r.n_items}</b></span><span>分片 <b>{r.shards}</b></span>
        <span>任务 <b>{Object.entries(r.tasks).map(([k, v]) => `${k} ${v}`).join(' · ') || '无'}</b></span>
        {r.non_real_results > 0 && <span className="warn">非 real 结果 {r.non_real_results} 份（smoke/合成，不进入统计）</span>}
      </div>
      {!r.buckets.length && <Empty>还没有 real 结果。用下方命令在 GPU 节点执行后，这里每 15 秒刷新。</Empty>}
      {r.buckets.map(b => {
        const cells = Object.entries(b.behavior || {});
        const failing = cells.filter(([, rec]) => !rec.pass).length;
        return (
          <div key={b.bucket} className="v6-report-bucket">
            <h5 className="v6-h5"><span className="mono">{b.bucket}</span> · 分片 {b.shards_present.length}/{b.shards_expected} · 行为门 {cells.length - failing}/{cells.length} 格达标（阈值 {Math.round(b.behavior_threshold * 100)}%）</h5>
            {cells.length > 0 && (
              <table className="v6-mini-table">
                <thead><tr><th>条件格</th><th>probe</th><th>准确率</th><th /></tr></thead>
                <tbody>{cells.map(([cell, rec]) => (
                  <tr key={cell}><th scope="row" className="mono">{cell}</th><td className="num">{rec.n}</td>
                    <td className="num">{rec.acc == null ? '—' : `${(rec.acc * 100).toFixed(0)}%`}</td><td><StatusPill status={rec.pass ? 'pass' : 'fail'} /></td></tr>
                ))}</tbody>
              </table>
            )}
            {b.excluded.length > 0 && <p className="v6-note warn">排除：{b.excluded.map(e => `${e.sha.slice(0, 8)} ${e.reason}`).join('；')}</p>}
          </div>
        );
      })}
      <h5 className="v6-h5">证据门链</h5>
      <GateChain gates={r.gates} compact />
    </div>
  );
}

export default function FreezePanel({ draft, onDraft, onGo }) {
  const library = useApi(draft ? null : '/api/grammar/library');
  const [token, setToken] = useAdminToken();
  const [perCell, setPerCell] = useState(4);
  const [busy, setBusy] = useState('');
  const [message, setMessage] = useState(null);
  const [ack, setAck] = useState(false);
  const templates = useApi('/api/templates', { interval: 30000 });
  const v3 = (templates.data?.templates || []).filter(t => t.protocol === 'v3');
  const fam = draft?.family_id;
  const pilotTm = fam ? `TG-${fam}-P` : '';
  const finalTm = fam ? `TG-${fam}` : '';
  const pilotExists = v3.some(t => t.tm_id === pilotTm);
  const finalRow = v3.find(t => t.tm_id === finalTm);

  const freeze = async mode => {
    setBusy(mode); setMessage(null);
    try {
      const out = await postJson('/api/grammar/register', { spec: draft, mode, pilot_per_cell: Number(perCell) || 4 }, { adminToken: token });
      setMessage({ ok: true, text: out.idempotent_replay ? `${out.tm_id} v${out.version} 已是当前冻结版本（未改动）` : `已冻结 ${out.tm_id} v${out.version} · ${out.n_items} 条 · design ${out.design_sha.slice(0, 12)}` });
      templates.reload();
    } catch (reason) {
      setMessage({ ok: false, text: reason.status === 401 ? '管理令牌无效或未填写' : String(reason.message || reason) });
    }
    setBusy('');
  };

  return (
    <div className="v6-freeze">
      <div className="v6-freeze-main">
        <Card title="当前草稿" extra={<AdminTokenField token={token} onChange={setToken} />}>
          {!draft && (
            <Empty action={<button type="button" className="fw-tbtn" onClick={() => onGo('studio')}>去模板工作室 →</button>}>
              <span>还没有选中模板族。直接从模板族库选择：
                <select className="v6-inline-select" value="" onChange={event => {
                  const g = (library.data?.grammars || []).find(x => x.family_id === event.target.value);
                  if (g) onDraft?.(g.spec);
                }}>
                  <option value="">选择…</option>
                  {(library.data?.grammars || []).map(g => <option key={g.file} value={g.family_id}>{g.family_id} · {g.name}</option>)}
                </select>
              </span>
            </Empty>
          )}
          {draft && (
            <div className="v6-freeze-steps">
              <div className="v6-step">
                <span className="n">A</span>
                <div>
                  <b>冻结试跑版 <span className="mono">{pilotTm}</span></b>
                  <p>每个 (划分, 条件格) 最多取 <input type="number" min="1" max="64" value={perCell} onChange={event => setPerCell(event.target.value)} /> 个填充，单分片；节点跑完后在右侧看逐格行为门。</p>
                  <button type="button" className="fw-tbtn" disabled={!!busy} onClick={() => freeze('pilot')}>{busy === 'pilot' ? '冻结中…' : pilotExists ? '重新冻结试跑版' : '冻结试跑版'}</button>
                </div>
              </div>
              <div className="v6-step">
                <span className="n">B</span>
                <div>
                  <b>冻结正式模板 <span className="mono">{finalTm}</span></b>
                  <p>完整语料 {draft.shards || 1} 分片，含金丝雀；冻结后语料、runner 与预注册阈值都不可修改。{finalRow && <> 当前已有 v{finalRow.version}（design {String(finalRow.design_sha).slice(0, 12)}）。</>}</p>
                  <label className="fw-chk"><input type="checkbox" checked={ack} onChange={event => setAck(event.target.checked)} /> 我已查看试跑行为门；未达标的条件格在合并时会被排除</label>
                  <button type="button" className="fw-tbtn primary" disabled={!!busy || !ack} onClick={() => freeze('final')}>{busy === 'final' ? '冻结中…' : '冻结正式模板'}</button>
                </div>
              </div>
              {message && <div className={message.ok ? 'v6-ok' : 'fw-ai-err'}>{message.text}</div>}
            </div>
          )}
        </Card>
        <Card title="已冻结的生成式模板" extra={<button type="button" className="fw-tbtn" onClick={() => onGo('dispatch')}>分发 →</button>}>
          {!templates.data && <Empty>{templates.error ? `未接入（${templates.error}）` : '读取…'}</Empty>}
          {templates.data && !v3.length && <Empty>还没有冻结任何生成式模板。</Empty>}
          {v3.length > 0 && (
            <table className="v6-mini-table">
              <thead><tr><th>模板</th><th>版本</th><th>类型</th><th>样本</th><th>分片</th><th>design_sha</th><th>结果</th></tr></thead>
              <tbody>{v3.map(t => (
                <tr key={t.tm_id}><th scope="row" className="mono">{t.tm_id}</th><td>v{t.version}</td><td>{t.mode === 'pilot' ? '试跑' : '正式'}</td>
                  <td className="num">{t.n_items}</td><td className="num">{t.shards}</td><td className="mono">{String(t.design_sha).slice(0, 12)}</td><td className="num">{t.results}</td></tr>
              ))}</tbody>
            </table>
          )}
        </Card>
      </div>
      <aside className="v6-freeze-side">
        <Card title="试跑报告" sub={pilotTm || '—'}>
          {pilotExists ? <Report tm={pilotTm} /> : <Empty>冻结试跑版后显示逐格行为门。</Empty>}
          <NodeCommand />
        </Card>
      </aside>
    </div>
  );
}
