/* 研发 · 题库（catalog.v1）：所有用户看到的同一份题目清单。
   类别 → 模式族（题目）→ 冻结版本、各模型的测试状态（分片、行为门）、待审数；
   动作：复制领题命令（本地 GPU 测试后回传）、加入自动测试、在空间查看、下载研究包（respack.v1）。
   「已完成」只描述数据（至少一个模型桶分片齐全可合并），不是证据等级。 */
import { useMemo, useState } from 'react';
import { API_BASE, authJson, fmtAgoTs, useApi } from './api.js';
import { useAuth } from './authContext.js';
import { STATUS_META, STATUS_ORDER, claimCommand, packFile, packUrl } from './catalogUtil.js';
import { Card, Empty, SourceTag, StatusPill } from './ui.jsx';

export function CopyButton({ text, label = '复制' }) {
  const [done, setDone] = useState(false);
  return (
    <button type="button" className="fw-tbtn" onClick={() => {
      try { navigator.clipboard.writeText(text); } catch { /* 剪贴板不可用 */ }
      setDone(true); window.setTimeout(() => setDone(false), 1400);
    }}>{done ? '✓ 已复制' : label}</button>
  );
}

function ItemCard({ item, onGo, onAutotest, onStudio }) {
  const [open, setOpen] = useState(false);
  const [icon, label] = STATUS_META[item.status] || ['na', item.status_label];
  const concepts = Object.entries(item.concepts || {});
  return (
    <article className={'v6-cat-item ' + item.status}>
      <header>
        <StatusPill status={icon} label={label} />
        <b><span className="mono">{item.id}</span> {item.name}</b>
        {item.protocol === 'legacy' && <span className="v6-pill muted">legacy</span>}
      </header>
      {item.description && <p className="v6-cat-desc">{item.description}</p>}
      <div className="v6-cat-meta">
        {item.factors?.length > 0 && <span>因子 <b>{item.factors.join(' × ')}</b></span>}
        {item.tm_id && <span>冻结 <b className="mono">{item.tm_id} v{item.version}</b></span>}
        {item.n_items != null && <span>{item.n_items} 条 · {item.shards || 1} 分片</span>}
        {item.tasks_active > 0 && <span className="info">{item.tasks_active} 个分片任务进行中</span>}
        {item.audit_pending > 0 && <span className="warn">{item.audit_pending} 份结果待审</span>}
        {item.last_result_at && <span>最近回传 {fmtAgoTs(item.last_result_at)}</span>}
      </div>
      {concepts.length > 0 && (
        <div className="v6-cat-concepts" title="文法预注册的跨族概念：谱图挖掘检验这些水平是否与其他模式族同概念水平共享编码">
          {concepts.map(([k, v]) => <span key={k}><i>{v}</i>{k}</span>)}
        </div>
      )}
      {item.models?.length > 0 && (
        <table className="v6-mini-table v6-cat-models">
          <thead><tr><th>模型桶</th><th>分片</th><th>行为门</th><th /></tr></thead>
          <tbody>
            {item.models.map(m => (
              <tr key={m.bucket}>
                <th scope="row" className="mono" title={m.bucket}>{m.bucket}</th>
                <td>{m.shards_present != null ? <StatusPill status={m.complete ? 'done' : 'partial'} label={`${m.shards_present}/${m.shards_expected}`} /> : '—'}</td>
                <td className="num">{m.behavior_cells ? `${m.behavior_pass}/${m.behavior_cells} 格` : '—'}</td>
                <td className="v6-cat-act">
                  {item.protocol === 'v3' && m.has_agg && <button type="button" className="fw-tbtn ghost" onClick={() => onGo?.('spatial', null, undefined, { tm: item.tm_id, bucket: m.bucket })}>查看 3D</button>}
                  {m.download && m.has_agg && <a className="fw-tbtn ghost" href={packUrl(item.tm_id, m.bucket)} download={packFile(item.tm_id, m.bucket)}>研究包 ↓</a>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <footer>
        {item.protocol === 'v3' && item.in_library && <button type="button" className="fw-tbtn primary" onClick={() => onAutotest?.(item.id)}>加入自动测试 →</button>}
        {item.claim_command && <button type="button" className="fw-tbtn" aria-expanded={open} onClick={() => setOpen(v => !v)}>{open ? '收起命令' : '本地领题命令'}</button>}
        {item.status === 'draft' && <button type="button" className="fw-tbtn" onClick={() => onStudio?.()}>去模板工作室冻结 →</button>}
      </footer>
      {open && item.claim_command && (
        <div className="v6-cmd">
          <code>{claimCommand(item)}</code>
          <CopyButton text={claimCommand(item)} />
        </div>
      )}
    </article>
  );
}

export function ParticipateSteps({ compact = false }) {
  const { user } = useAuth();
  const [code, setCode] = useState(null);
  const [err, setErr] = useState('');
  const makeCode = async () => {
    setErr('');
    try { setCode(await authJson('/api/auth/node-code', { method: 'POST' })); } catch (e) { setErr(e.message); }
  };
  const bind = code ? ` --bind-code ${code.code}` : '';
  const reg = `python deploy/node_agent.py --server ${API_BASE} register --name <你的节点名> --gpu "<显卡>"${bind}`;
  const run = `python deploy/node_agent.py --server ${API_BASE} run --once --tm-id <题目 tm_id> --model-path <本机HF模型目录> --model-id <model_id> --dtype bf16`;
  return (
    <ol className={'v6-steps-list' + (compact ? ' compact' : '')}>
      <li><b>注册节点</b>
        <span>一次即可，节点令牌保存在本机。{user ? '先生成绑定码，节点会归属你的账号（贡献者署名、在账号菜单「我的节点」查看）。' : ''}</span>
        {user && (
          <span className="v6-bind">
            <button type="button" className="fw-tbtn" onClick={makeCode}>{code ? '重新生成绑定码' : '生成绑定码'}</button>
            {code && <small>绑定码 <b className="mono">{code.code}</b> · {Math.round(code.expires_in / 60)} 分钟内有效 · 只能用一次</small>}
            {err && <small className="warn">{err}</small>}
          </span>
        )}
        <div className="v6-cmd"><code>{reg}</code><CopyButton text={reg} /></div></li>
      <li><b>领题并在本地运行</b><span>在题库中选一个题目的 tm_id；节点领取一个分片，在本机 GPU 上跑 runner，自动回传充分统计量。</span><div className="v6-cmd"><code>{run}</code><CopyButton text={run} /></div></li>
      <li><b>回传后</b><span>结果经自动检查与审计后进入合并；分片齐全即可在 空间 · 实验响应 查看 3D，或下载研究包到本地。</span></li>
    </ol>
  );
}

export default function CatalogPanel({ onGo, onAutotest, onStudio }) {
  const cat = useApi('/api/catalog', { interval: 20000 });
  const [category, setCategory] = useState('');
  const [status, setStatus] = useState('');
  const [query, setQuery] = useState('');
  const [help, setHelp] = useState(false);
  const items = useMemo(() => (cat.data?.items || []).filter(it =>
    (!category || it.category === category) && (!status || it.status === status)
    && (!query || `${it.id} ${it.name} ${it.description} ${(it.factors || []).join(' ')}`.toLowerCase().includes(query.toLowerCase()))
  ), [cat.data, category, status, query]);
  if (!cat.data) return <div className="v6-cat"><Empty>{cat.error ? `中心服务未接入（${cat.error}）。` : '读取题库…'}</Empty></div>;
  const cats = cat.data.categories;
  const total = cats.reduce((s, c) => s + c.total, 0);
  const counts = Object.fromEntries(STATUS_ORDER.map(k => [k, cats.reduce((s, c) => s + (c[k] || 0), 0)]));
  return (
    <div className="v6-cat">
      <header className="v6-at-hd">
        <div>
          <h3>题库 · 所有用户共享的语言测试题</h3>
          <p>每个题目是一个模式族（句式 × 填充词 × 因子）。在这里查看各题在各模型上的测试状态；在本机 GPU 领题测试并回传，或加入自动测试，或下载研究包到本地 3D 观察。</p>
        </div>
        <div className="v6-at-hd-r">
          <SourceTag state="live" label="题库" />
          <button type="button" className="fw-tbtn" aria-expanded={help} onClick={() => setHelp(v => !v)}>{help ? '收起' : '如何参与测试'}</button>
        </div>
      </header>
      {help && <Card title="参与测试（任何用户）"><ParticipateSteps /></Card>}
      {cat.data.sandbox_synthetic && <div className="v6-synthetic"><b>沙盒中心</b><span>题库中包含合成模型桶（「合成·」前缀），只用于流程验证。</span></div>}
      <div className="v6-cat-body">
        <aside className="v6-cat-side">
          <button type="button" className={'v6-cat-cat' + (!category ? ' on' : '')} onClick={() => setCategory('')}>
            <b>全部类别</b><small>{total} 题</small>
          </button>
          {cats.map(c => (
            <button key={c.category} type="button" className={'v6-cat-cat' + (category === c.category ? ' on' : '')} onClick={() => setCategory(c.category)}>
              <b>{c.category}</b><small>{c.done}/{c.total} 已完成{c.running ? ` · ${c.running} 测试中` : ''}</small>
              <span className="v6-cat-bar" aria-hidden="true">
                {STATUS_ORDER.map(k => (c[k] ? <i key={k} className={k} style={{ flex: c[k] }} /> : null))}
              </span>
            </button>
          ))}
          <p className="v6-note">状态只描述数据：已完成 = 至少一个模型桶的分片齐全、可以合并；它不是证据等级。</p>
        </aside>
        <div className="v6-cat-main">
          <div className="v6-cat-filters">
            <div className="v6-chips" role="group" aria-label="按状态筛选">
              <button type="button" className={!status ? 'on' : ''} onClick={() => setStatus('')}>全部 {total}</button>
              {STATUS_ORDER.map(k => (
                <button key={k} type="button" className={status === k ? 'on' : ''} disabled={!counts[k]} onClick={() => setStatus(k)}>
                  {STATUS_META[k][1]} {counts[k]}
                </button>
              ))}
            </div>
            <input className="v6-search" placeholder="搜索题目、因子…" value={query} onChange={e => setQuery(e.target.value)} />
          </div>
          {!items.length && <Empty>没有符合条件的题目。</Empty>}
          <div className="v6-cat-grid">
            {items.map(it => <ItemCard key={it.id} item={it} onGo={onGo} onAutotest={onAutotest} onStudio={onStudio} />)}
          </div>
        </div>
      </div>
    </div>
  );
}
