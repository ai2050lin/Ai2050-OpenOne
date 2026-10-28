/* 研发 · 贡献算力（贡献算力客户端 P2，设计见 design/contributor_client_v1.md）：
   桌面客户端里控制本机节点服务——节点服务 → 设备 → 绑定账号 → 模型 → 资源限制 → 运行 → 日志与我的记录。
   界面只和本机节点服务通信（经 Electron 主进程转发）；只有节点服务访问中心（注册 / 领取 / 续租 / 上传 / 站点配置）。
   唯一的例外是「绑定到我的账号」：向中心 /api/auth/node-code 取一次性绑定码，只在点击时。
   浏览器里的网页不能控制节点，只显示下载入口。上传与中心校验通过只表示结果可以合并，不改变证据等级。 */
import { useEffect, useState } from 'react';

import { downloadKit } from '../../open/openUtil.js';
import { API_BASE, authJson } from './api.js';
import { useAuth } from './authContext.js';
import { NODE_STATE, RESULT_LABEL, SUP_STATE, TASK_KIND_LABEL, fmtTime, nodeCall, useNode } from './contributeUtil.js';
import { Card, Empty } from './ui.jsx';
import './contribute.css';

function Pill({ tone, icon, children, title }) {
  return <span className={'v6-pill ' + tone} title={title}><i aria-hidden="true">{icon}</i>{children}</span>;
}

function WebOnly() {
  const { user, guest } = useAuth();
  const loggedIn = Boolean(user) && !guest;
  const [msg, setMsg] = useState('');
  const get = async () => {
    setMsg('');
    try { await downloadKit(loggedIn); setMsg('已下载 ai2050-node.zip'); } catch (e) { setMsg(e.message); }
  };
  return (
    <Card title="贡献算力" sub="用你的 NVIDIA 显卡运行中心冻结的测试任务，只上传统计结果">
      <p className="ct-lead">浏览器里的网页不能调用显卡运行模型，也不能控制本机节点。请用桌面客户端（研发 → 贡献算力），或下载命令行一键包：</p>
      <div className="v6-toolbar">
        <button type="button" className="fw-tbtn" onClick={get}>下载命令行一键包{loggedIn ? '（内嵌绑定码）' : '（匿名）'}</button>
        {msg && <span className="v6-note">{msg}</span>}
      </div>
      <pre className="ct-cmd">python node_agent.py participate</pre>
    </Card>
  );
}

function HostCard({ node }) {
  const { bridge, sup } = node;
  const [busy, setBusy] = useState(false);
  const run = async fn => { setBusy(true); try { await fn(); } finally { setBusy(false); } };
  if (!sup) return <Card title="本机节点服务"><Empty>读取…</Empty></Card>;
  const bad = ['no-python', 'no-script', 'failed'].includes(sup.state);
  return (
    <Card title="本机节点服务" sub="客户端启动并监护；退出客户端时一并关闭（命令行启动的服务除外）"
          extra={<>
            <button type="button" className="fw-tbtn ghost" disabled={busy} onClick={() => run(bridge.pickPython)}>选择 Python…</button>
            <button type="button" className="fw-tbtn ghost" disabled={busy} onClick={() => run(bridge.restart)}>重启服务</button>
          </>}>
      <dl className="ct-kv">
        <dt>状态</dt><dd><Pill tone={bad ? 'bad' : sup.state === 'starting' ? 'warn' : 'good'} icon={bad ? '✗' : '●'}>{SUP_STATE[sup.state] || sup.state}</Pill>
          {sup.synthetic && <Pill tone="warn" icon="!">synthetic 沙盒联调</Pill>}</dd>
        {sup.error && <><dt>原因</dt><dd className="ct-err">{sup.error}</dd></>}
        <dt>Python</dt><dd className="ct-mono">{sup.python || (sup.state === 'attached' ? '（命令行启动的服务）' : '—')}</dd>
        <dt>服务目录</dt><dd className="ct-mono">{sup.home}</dd>
      </dl>
      {bad && sup.tail?.length > 0 && <pre className="ct-log short">{sup.tail.join('\n')}</pre>}
      {sup.state === 'no-python' && <p className="v6-note">计算组件的一键安装在下一期（P3）实现；现在请选择一个装好 torch（CUDA 版）与 transformers 的 python.exe。</p>}
    </Card>
  );
}

function DeviceCard({ live }) {
  const [dev, setDev] = useState(null);
  const [err, setErr] = useState('');
  const load = refresh => nodeCall('GET', `/device${refresh ? '?refresh=1' : ''}`).then(d => { setDev(d); setErr(''); }).catch(e => setErr(e.message));
  useEffect(() => { if (live) load(false); }, [live]);
  const rt = dev?.runtime;
  return (
    <Card title="设备" sub="nvidia-smi 与计算组件检测（只在本机）"
          extra={<button type="button" className="fw-tbtn ghost" disabled={!live} onClick={() => load(true)}>↻ 重新检测</button>}>
      {err && <Empty>{err}</Empty>}
      {!dev && !err && <Empty>{live ? '检测中…' : '本机节点服务没有运行'}</Empty>}
      {dev && (
        <>
          {dev.gpus.length ? (
            <table className="v6-mini-table">
              <thead><tr><th>设备</th><th>GPU</th><th className="num">显存</th><th className="num">占用</th><th className="num">利用率</th><th>驱动</th></tr></thead>
              <tbody>{dev.gpus.map(g => (
                <tr key={g.index}><td className="ct-mono">cuda:{g.index}</td><td>{g.name}</td><td className="num">{g.vram_gib?.toFixed(1)} GiB</td>
                  <td className="num">{g.used_gib != null ? `${g.used_gib.toFixed(1)} GiB` : '—'}</td><td className="num">{g.util != null ? `${g.util}%` : '—'}</td><td>{g.driver || '—'}</td></tr>
              ))}</tbody>
            </table>
          ) : <Empty>没有检测到 NVIDIA GPU（需要 NVIDIA 显卡与驱动）。</Empty>}
          <dl className="ct-kv">
            <dt>计算组件</dt>
            <dd>{rt?.ok ? <Pill tone="good" icon="✓">可用</Pill> : <Pill tone="bad" icon="✗">不可用</Pill>}
              <span className="ct-mono">torch {rt?.torch || '—'} · CUDA {rt?.cuda || '—'} · transformers {rt?.transformers || '—'}</span>
              {rt && !rt.ok && <small className="ct-err">{rt.error || rt.torch_error || rt.transformers_error || (rt.torch && !rt.cuda_available ? 'torch 检测不到 CUDA（需要 CUDA 版 torch）' : '缺少 torch 或 transformers')}</small>}</dd>
            <dt>磁盘剩余</dt><dd>{dev.disk_free_gib != null ? `${dev.disk_free_gib.toFixed(1)} GiB` : '—'}<small>（服务目录所在盘；模型权重另需空间）</small></dd>
            <dt>电源</dt><dd>{dev.battery === true ? '电池供电' : dev.battery === false ? '外接电源' : '未知'}</dd>
          </dl>
        </>
      )}
    </Card>
  );
}

function AccountCard({ node }) {
  const { status, poll } = node;
  const { user, guest } = useAuth();
  const loggedIn = Boolean(user) && !guest;
  const [code, setCode] = useState('');
  const [msg, setMsg] = useState('');
  const [busy, setBusy] = useState(false);
  const bind = async getCode => {
    setBusy(true); setMsg('');
    try {
      const c = await getCode();
      const r = await nodeCall('POST', '/bind', { bind_code: c });
      setMsg(r.owner_bound ? `已注册 ${r.node_id}，绑定到你的账号` : `已注册 ${r.node_id}（匿名：贡献不署名）`);
      setCode('');
      poll();
    } catch (e) { setMsg(e.message); }
    setBusy(false);
  };
  if (!status) return <Card title="账号与节点"><Empty>读取…</Empty></Card>;
  const center = API_BASE || window.location.origin;
  const mismatch = status.server && center && status.server !== center.replace(/\/+$/, '');
  return (
    <Card title="账号与节点" sub="节点注册后由中心发放节点令牌；绑定码一次性有效">
      <dl className="ct-kv">
        <dt>中心</dt><dd className="ct-mono">{status.server || '—'}</dd>
        <dt>节点</dt><dd>{status.bound
          ? <><span className="ct-mono">{status.node_id}</span> {status.owner_bound ? <Pill tone="good" icon="✓">已绑定账号</Pill> : <Pill tone="muted" icon="○">匿名</Pill>}</>
          : <Pill tone="warn" icon="!">未注册</Pill>}</dd>
      </dl>
      {mismatch && <p className="v6-note warn">节点服务连接的中心与客户端不同：账号绑定码只在客户端连接的中心有效。</p>}
      {!status.bound && (
        <div className="v6-toolbar">
          {loggedIn && <button type="button" className="fw-tbtn" disabled={busy || mismatch}
                               onClick={() => bind(async () => (await authJson('/api/auth/node-code', { method: 'POST' })).code)}>绑定到我的账号（{user.display_name || user.username}）</button>}
          <input className="ct-input" placeholder="或粘贴绑定码 B-XXXX-XXXX" value={code} onChange={e => setCode(e.target.value.trim())} />
          <button type="button" className="fw-tbtn ghost" disabled={busy || !code} onClick={() => bind(async () => code)}>用绑定码注册</button>
          <button type="button" className="fw-tbtn ghost" disabled={busy} onClick={() => bind(async () => '')}>匿名注册</button>
        </div>
      )}
      {msg && <p className="v6-note">{msg}</p>}
    </Card>
  );
}

const MODEL_STATUS = { ok: ['good', '✓', '快照一致'], checking: ['warn', '…', '核对中'], mismatch: ['bad', '✗', '快照不一致'], missing: ['muted', '○', '未指定目录'] };

function ModelsCard({ live }) {
  const [models, setModels] = useState(null);
  const [dirs, setDirs] = useState({});
  const [err, setErr] = useState('');
  const load = refresh => nodeCall('GET', `/models${refresh ? '?refresh=1' : ''}`).then(r => { setModels(r.models); setErr(''); }).catch(e => setErr(e.message));
  useEffect(() => { if (live) load(false); }, [live]);
  useEffect(() => {
    if (!models?.some(m => m.status === 'checking')) return undefined;
    const t = setTimeout(() => load(false), 2000);
    return () => clearTimeout(t);
  }, [models]);
  const check = async m => {
    try { await nodeCall('POST', '/models/check', { snapshot: m.snapshot, dir: dirs[m.snapshot] ?? m.dir }); load(false); } catch (e) { setErr(e.message); }
  };
  return (
    <Card title="模型" sub="中心要求的模型快照；本机目录按权重与 tokenizer 文件的 sha256 核对后才会领取该模型的任务"
          extra={<button type="button" className="fw-tbtn ghost" disabled={!live} onClick={() => load(true)}>↻ 读取中心要求</button>}>
      {err && <p className="v6-note warn">{err}</p>}
      {!models ? <Empty>{live ? '读取…' : '本机节点服务没有运行'}</Empty> : !models.length ? <Empty>中心当前没有列出模型（站点配置的 models 为空）。</Empty> : (
        <table className="v6-mini-table ct-models">
          <thead><tr><th>模型</th><th>快照</th><th>本机目录</th><th className="num">权重</th><th>状态</th><th /></tr></thead>
          <tbody>{models.map(m => {
            const [tone, icon, label] = MODEL_STATUS[m.status] || ['muted', '·', m.status];
            return (
              <tr key={m.snapshot}>
                <td><b>{m.name}</b>{m.hf_repo && <small className="ct-mono">{m.hf_repo}</small>}</td>
                <td className="ct-mono" title={m.snapshot}>{m.snapshot.slice(0, 12)}</td>
                <td><input className="ct-input wide" placeholder="D:\models\…" value={dirs[m.snapshot] ?? m.dir ?? ''} onChange={e => setDirs({ ...dirs, [m.snapshot]: e.target.value })} /></td>
                <td className="num">{m.weight_gib ? `${m.weight_gib} GiB` : '—'}</td>
                <td><Pill tone={tone} icon={icon}>{label}</Pill></td>
                <td><button type="button" className="fw-tbtn ghost" disabled={!(dirs[m.snapshot] ?? m.dir) || m.status === 'checking'} onClick={() => check(m)}>核对</button></td>
              </tr>
            );
          })}</tbody>
        </table>
      )}
      <p className="v6-note">模型下载在下一期（P3）实现；现在请先用 huggingface 等工具下载到本机目录，再填写目录核对。</p>
    </Card>
  );
}

function LimitsCard({ live, gpus }) {
  const [cfg, setCfg] = useState(null);
  const [msg, setMsg] = useState('');
  useEffect(() => { if (live) nodeCall('GET', '/config').then(setCfg).catch(e => setMsg(e.message)); }, [live]);
  if (!cfg) return <Card title="资源限制"><Empty>{msg || (live ? '读取…' : '本机节点服务没有运行')}</Empty></Card>;
  const lim = cfg.limits;
  const setLim = patch => setCfg({ ...cfg, limits: { ...lim, ...patch } });
  const num = v => (v === '' ? 0 : Number(v));
  const save = async () => {
    setMsg('');
    try {
      const { task_kinds, device, node_name, limits } = cfg;
      setCfg(await nodeCall('PUT', '/config', { task_kinds, device, node_name, limits }));
      setMsg('已保存；下一次领取任务前生效');
    } catch (e) { setMsg(e.message); }
  };
  const kinds = Object.keys(TASK_KIND_LABEL);
  const devices = [...new Set([...(gpus || []).map(g => `cuda:${g.index}`), cfg.device])];
  return (
    <Card title="资源限制" sub="每次领取任务前检查；不满足时等待并显示原因" extra={<button type="button" className="fw-tbtn" onClick={save}>保存</button>}>
      <div className="ct-form">
        <label>任务类型<span>{kinds.map(k => (
          <label key={k} className="ct-check"><input type="checkbox" checked={cfg.task_kinds.includes(k)}
            onChange={e => setCfg({ ...cfg, task_kinds: e.target.checked ? [...cfg.task_kinds, k] : cfg.task_kinds.filter(x => x !== k) })} />{TASK_KIND_LABEL[k]}</label>
        ))}</span></label>
        <label>显卡<select value={cfg.device} onChange={e => setCfg({ ...cfg, device: e.target.value })}>{devices.map(d => <option key={d} value={d}>{d}</option>)}</select></label>
        <label>节点名称<input className="ct-input" value={cfg.node_name} placeholder="缺省用计算机名" onChange={e => setCfg({ ...cfg, node_name: e.target.value })} /></label>
        <label>允许的时间段<input className="ct-input" value={lim.hours} placeholder="空 = 全天；如 22:00-08:00" onChange={e => setLim({ hours: e.target.value })} /></label>
        <label className="ct-check"><input type="checkbox" checked={lim.pause_on_battery} onChange={e => setLim({ pause_on_battery: e.target.checked })} />用电池供电时暂停</label>
        <label>其他程序占用 GPU 时让出<span><input className="ct-input num" type="number" min="0" max="100" value={lim.yield_gpu_util} onChange={e => setLim({ yield_gpu_util: num(e.target.value) })} /> %（0 = 不让出）</span></label>
        <label>显存上限<span><input className="ct-input num" type="number" min="0" value={lim.max_vram_gib} onChange={e => setLim({ max_vram_gib: num(e.target.value) })} /> GiB（0 = 显卡总显存；按权重 × 1.2 估计）</span></label>
        <label>最少剩余磁盘<span><input className="ct-input num" type="number" min="0" value={lim.min_free_disk_gib} onChange={e => setLim({ min_free_disk_gib: num(e.target.value) })} /> GiB</span></label>
      </div>
      {msg && <p className="v6-note">{msg}</p>}
    </Card>
  );
}

function RunCard({ node }) {
  const { status, setStatus } = node;
  const [err, setErr] = useState('');
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => { const t = setInterval(() => setNow(Date.now() / 1000), 1000); return () => clearInterval(t); }, []);
  if (!status) return <Card title="运行"><Empty>读取…</Empty></Card>;
  const act = async path => { setErr(''); try { setStatus(await nodeCall('POST', path)); } catch (e) { setErr(e.message); } };
  const meta = NODE_STATE[status.state] || { label: status.state, tone: 'muted', icon: '·' };
  const cur = status.current;
  const p = cur?.progress;
  const btn = (path, label, primary) => <button type="button" className={'fw-tbtn' + (primary ? '' : ' ghost')} onClick={() => act(path)}>{label}</button>;
  return (
    <Card title="运行" sub="暂停：当前任务做完后不再领取；停止：立即结束运行中的任务，中心在租约到期后重新派发"
          extra={<>
            {status.state === 'stopped' || status.state === 'error' ? btn('/start', '开始贡献', true) : null}
            {status.state === 'paused' ? btn('/resume', '继续', true) : null}
            {['running', 'waiting'].includes(status.state) ? btn('/pause', '暂停') : null}
            {status.state !== 'stopped' ? btn('/stop', '停止') : null}
          </>}>
      <dl className="ct-kv">
        <dt>状态</dt><dd><Pill tone={meta.tone} icon={meta.icon}>{meta.label}</Pill>
          {status.next_at && !cur && <small>下次查看：{Math.max(0, Math.round(status.next_at - now))} 秒后</small>}</dd>
        {status.state === 'waiting' && status.reasons?.length > 0 && <><dt>等待原因</dt><dd><ul className="ct-reasons">{status.reasons.map(r => <li key={r}>{r}</li>)}</ul></dd></>}
        <dt>当前任务</dt><dd>{cur ? <>
          <b>{TASK_KIND_LABEL[cur.kind] || cur.kind}</b> · {cur.model || '—'} · 已运行 {Math.round(now - cur.started)} 秒
          {p?.total ? <span className="ct-progress"><span style={{ width: `${Math.min(100, (100 * p.done) / p.total)}%` }} /></span> : null}
          {p?.total ? <small>{p.done}/{p.total} 窗口 · {Number(p.tokens || 0).toLocaleString()} token · {Math.round(p.tok_per_s || 0)} tok/s</small> : null}
        </> : '—'}</dd>
        <dt>本次计数</dt><dd>完成 {status.counts.done} · 拒收 {status.counts.rejected} · 失败 {status.counts.failed}</dd>
        {status.last_error && <><dt>最近错误</dt><dd className="ct-err">{status.last_error}</dd></>}
      </dl>
      {err && <p className="v6-note warn">{err}</p>}
    </Card>
  );
}

function LogCard({ logs }) {
  const [open, setOpen] = useState(true);
  return (
    <Card title="日志" sub="本机节点服务的输出（最近 500 行）" extra={<button type="button" className="fw-tbtn ghost" onClick={() => setOpen(!open)}>{open ? '收起' : '展开'}</button>}>
      {open && (logs.length ? <pre className="ct-log">{logs.slice(-200).map(l => `${fmtTime(l.t)}  ${l.msg}`).join('\n')}</pre> : <Empty>还没有日志。</Empty>)}
    </Card>
  );
}

function HistoryCard({ live, counts }) {
  const [items, setItems] = useState(null);
  const total = counts ? counts.done + counts.rejected + counts.failed : 0;
  useEffect(() => { if (live) nodeCall('GET', '/history?limit=100').then(r => setItems(r.items)).catch(() => setItems([])); }, [live, total]);
  return (
    <Card title="本机完成记录" sub="上传与中心校验通过只表示结果可以合并，不改变证据等级">
      {!items ? <Empty>读取…</Empty> : !items.length ? <Empty>还没有完成的任务。</Empty> : (
        <table className="v6-mini-table">
          <thead><tr><th>时间</th><th>任务</th><th>分片</th><th>结论</th><th>结果 sha256</th></tr></thead>
          <tbody>{items.map((r, i) => {
            const [icon, label, tone] = RESULT_LABEL[r.status] || ['·', r.status, 'muted'];
            return (
              <tr key={i}>
                <td>{fmtTime(r.t)}</td><td>{TASK_KIND_LABEL[r.kind] || r.kind}<small className="ct-mono">{r.job || r.tm_id || ''}</small></td>
                <td className="ct-mono">{r.shard ?? '—'}</td>
                <td><Pill tone={tone} icon={icon}>{label}</Pill>{r.replica ? <small>复算</small> : null}{r.note && <small className="ct-err">{r.note}</small>}</td>
                <td className="ct-mono" title={r.sha}>{r.sha ? r.sha.slice(0, 12) : '—'}</td>
              </tr>
            );
          })}</tbody>
        </table>
      )}
    </Card>
  );
}

export default function ContributePanel() {
  const node = useNode();
  const [gpus, setGpus] = useState(null);
  useEffect(() => { if (node.live) nodeCall('GET', '/device').then(d => setGpus(d.gpus)).catch(() => {}); }, [node.live]);
  if (!node.bridge) return <div className="ct-root"><WebOnly /></div>;
  return (
    <div className="ct-root">
      {node.error && node.live && <p className="v6-note warn">{node.error}</p>}
      <div className="ct-grid">
        <HostCard node={node} />
        <RunCard node={node} />
        <AccountCard node={node} />
        <DeviceCard live={node.live} />
      </div>
      <ModelsCard live={node.live} />
      <LimitsCard live={node.live} gpus={gpus} />
      <LogCard logs={node.logs} />
      <HistoryCard live={node.live} counts={node.status?.counts} />
    </div>
  );
}
