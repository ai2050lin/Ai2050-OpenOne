/* 配置 · 贡献算力（贡献算力客户端 P2，设计见 design/contributor_client_v1.md）：
   桌面客户端里控制本机节点服务——节点服务 → 计算组件（P3）→ 设备 → 绑定账号 → 模型（下载与核对，P3）→ 资源限制 → 运行 → 日志与我的记录。
   界面只和本机节点服务通信（经 Electron 主进程转发）；只有节点服务访问中心（注册 / 领取 / 续租 / 上传 / 站点配置）。
   唯一的例外是「绑定到我的账号」：向中心 /api/auth/node-code 取一次性绑定码，只在点击时。
   浏览器里的网页不能控制节点，只显示下载入口。上传与中心校验通过只表示结果可以合并，不改变证据等级。 */
import { useEffect, useState } from 'react';

import { downloadKit } from '../../open/openUtil.js';
import { API_BASE, authJson } from './api.js';
import { useAuth } from './authContext.js';
import { NODE_STATE, RESULT_LABEL, SUP_STATE, TASK_KIND_LABEL, fmtBytes, fmtTime, nodeBridge, nodeCall, useNode, useRuntime } from './contributeUtil.js';
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
      <p className="ct-lead">浏览器里的网页不能调用显卡运行模型，也不能控制本机节点。请用桌面客户端（配置 → 账号与本机节点），或下载命令行一键包：</p>
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
      {sup.state === 'no-python' && <p className="v6-note">还没有可用的 Python：在「计算组件」里一键安装，或选择一个装好 torch（CUDA 版）与 transformers 的 python.exe。</p>}
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

function RuntimeCard({ onInstalled }) {
  const { bridge, info, setInfo } = useRuntime();
  const [mirror, setMirror] = useState('pypi');
  if (!bridge) return null;
  if (!info) return <Card title="计算组件"><Empty>读取…</Empty></Card>;
  const running = info.state === 'running';
  const ins = info.installed;
  const driverOk = info.driver != null && info.driver >= info.min_driver;
  const install = async () => { const r = await bridge.install(mirror); setInfo(r); if (r.state === 'done') onInstalled?.(); };
  return (
    <Card title="计算组件" sub="Python + torch（CUDA 13.0）+ transformers；版本与开发机验证过的组合相同，每个包按锁文件的 sha256 核对"
          extra={running ? <button type="button" className="fw-tbtn ghost" onClick={() => bridge.cancel().then(setInfo)}>取消</button> : <>
            <button type="button" className="fw-tbtn ghost" onClick={() => bridge.pickDir().then(setInfo)}>换目录…</button>
            <button type="button" className={'fw-tbtn' + (ins && info.current ? ' ghost' : '')} disabled={!driverOk} onClick={install}>{ins ? (info.current ? '重新安装' : '升级') : '安装（约 3 GB 下载）'}</button>
          </>}>
      <dl className="ct-kv">
        <dt>状态</dt><dd>{running ? <Pill tone="warn" icon="…">安装中</Pill>
          : ins ? (info.current ? <Pill tone="good" icon="✓">已安装</Pill> : <Pill tone="warn" icon="!">版本过旧，需要升级</Pill>)
            : <Pill tone="muted" icon="○">未安装</Pill>}
          {ins && <span className="ct-mono">torch {ins.torch} · CUDA {ins.cuda} · transformers {ins.transformers}</span>}</dd>
        <dt>目录</dt><dd className="ct-mono">{info.dir}<small>剩余 {info.free_gib ?? '—'} GiB（需要约 {info.need_gib} GiB）</small></dd>
        <dt>显卡驱动</dt><dd>{info.driver == null ? <Pill tone="bad" icon="✗">没有检测到 NVIDIA 驱动</Pill>
          : driverOk ? <>{info.driver}<small>（≥ {info.min_driver}）</small></> : <Pill tone="bad" icon="✗">{info.driver}：需要 ≥ {info.min_driver}，请先更新驱动</Pill>}</dd>
        {!running && <><dt>下载源</dt><dd><select value={mirror} onChange={e => setMirror(e.target.value)}>
          {Object.entries(info.mirrors).map(([k, label]) => <option key={k} value={k}>{label}</option>)}</select>
          <small>torch 来自 download.pytorch.org；镜像与官方的文件按同一 sha256 核对</small></dd></>}
        {running && <><dt>步骤</dt><dd>{info.step}{info.step.startsWith('下载 uv') && <small>{fmtBytes(info.bytes)} / {fmtBytes(info.uvSize)}</small>}</dd></>}
        {info.state === 'failed' && <><dt>失败</dt><dd className="ct-err">{info.error}</dd></>}
      </dl>
      {(running || info.state === 'failed') && info.log.length > 0 && <pre className="ct-log short">{info.log.slice(-30).join('\n')}</pre>}
    </Card>
  );
}

const MODEL_STATUS = {
  ok: ['good', '✓', '快照一致'], checking: ['warn', '…', '核对中'], mismatch: ['bad', '✗', '快照不一致'], missing: ['muted', '○', '未下载'],
  downloading: ['info', '↓', '下载中'], error: ['bad', '✗', '下载失败'], cancelled: ['muted', '❚❚', '已暂停下载'],
};

function ModelsCard({ live }) {
  const [res, setRes] = useState(null);
  const [dirs, setDirs] = useState({});
  const [err, setErr] = useState('');
  const [now, setNow] = useState(() => Date.now() / 1000);
  const load = refresh => nodeCall('GET', `/models${refresh ? '?refresh=1' : ''}`).then(r => { setRes(r); setErr(''); setNow(Date.now() / 1000); }).catch(e => setErr(e.message));
  useEffect(() => { if (live) load(false); }, [live]);
  const models = res?.models;
  useEffect(() => {
    if (!models?.some(m => m.status === 'checking' || m.status === 'downloading')) return undefined;
    const t = setTimeout(() => load(false), 1500);
    return () => clearTimeout(t);
  }, [models]);
  const act = async (path, body) => { setErr(''); try { await nodeCall('POST', path, body); load(false); } catch (e) { setErr(e.message); } };
  const setCfg = async patch => { setErr(''); try { await nodeCall('PUT', '/config', patch); load(false); } catch (e) { setErr(e.message); } };
  const pickRoot = async () => { const d = await nodeBridge()?.pickModelsDir?.(); if (d) setCfg({ models_root: d }); };
  return (
    <Card title="模型" sub="中心要求的模型快照；下载或指定的目录按权重与 tokenizer 文件的 sha256 核对一致后，才会领取该模型的任务"
          extra={<button type="button" className="fw-tbtn ghost" disabled={!live} onClick={() => load(true)}>↻ 读取中心要求</button>}>
      {err && <p className="v6-note warn">{err}</p>}
      {res && (
        <dl className="ct-kv">
          <dt>下载目录</dt><dd className="ct-mono">{res.models_root}<small>剩余 {res.models_free_gib ?? '—'} GiB</small>
            <button type="button" className="fw-tbtn ghost" onClick={pickRoot}>更改…</button></dd>
          <dt>下载来源</dt><dd><select value={res.hf_endpoint} onChange={e => setCfg({ hf_endpoint: e.target.value })}>
            {res.hf_endpoints.map(u => <option key={u} value={u}>{u.replace('https://', '')}</option>)}</select>
            <small>两处的文件相同；下载完成后按中心给出的模型快照核对</small></dd>
        </dl>
      )}
      {!models ? <Empty>{live ? '读取…' : '本机节点服务没有运行'}</Empty> : !models.length ? <Empty>中心当前没有列出模型（站点配置的 models 为空）。</Empty> : (
        <table className="v6-mini-table ct-models">
          <thead><tr><th>模型</th><th>快照</th><th>本机目录</th><th className="num">权重</th><th>状态</th><th /></tr></thead>
          <tbody>{models.map(m => {
            const [tone, icon, label] = MODEL_STATUS[m.status] || ['muted', '·', m.status];
            const p = m.progress;
            const rate = p && p.started && now > p.started ? p.done / (now - p.started) : 0;
            const busy = m.status === 'checking' || m.status === 'downloading';
            return (
              <tr key={m.snapshot}>
                <td><b>{m.name}</b>{m.hf_repo && <small className="ct-mono">{m.hf_repo}@{(m.hf_revision || 'main').slice(0, 10)}</small>}
                  {m.min_vram_gib ? <small>需要显存约 {m.min_vram_gib} GiB</small> : null}</td>
                <td className="ct-mono" title={m.snapshot}>{m.snapshot.slice(0, 12)}</td>
                <td><input className="ct-input wide" placeholder="已有模型时填目录核对" value={dirs[m.snapshot] ?? m.dir ?? ''} disabled={busy}
                           onChange={e => setDirs({ ...dirs, [m.snapshot]: e.target.value })} /></td>
                <td className="num">{m.weight_gib ? `${m.weight_gib} GiB` : '—'}</td>
                <td><Pill tone={tone} icon={icon}>{label}</Pill>
                  {p?.total ? <>
                    <span className="ct-progress"><span style={{ width: `${Math.min(100, (100 * p.done) / p.total)}%` }} /></span>
                    <small>{fmtBytes(p.done)} / {fmtBytes(p.total)}{rate ? ` · ${fmtBytes(rate)}/s` : ''}{p.file ? ` · ${p.file}` : ''}</small>
                  </> : null}
                  {m.error && <small className="ct-err">{m.error}</small>}</td>
                <td className="ct-actions">
                  {m.status === 'downloading'
                    ? <button type="button" className="fw-tbtn ghost" onClick={() => act('/models/cancel', { snapshot: m.snapshot })}>暂停</button>
                    : <>
                      {m.hf_repo && m.status !== 'ok' && <button type="button" className="fw-tbtn" disabled={busy} onClick={() => act('/models/download', { snapshot: m.snapshot })}>{m.status === 'cancelled' || m.status === 'error' ? '继续下载' : '下载'}</button>}
                      <button type="button" className="fw-tbtn ghost" disabled={!(dirs[m.snapshot] ?? m.dir) || busy} onClick={() => act('/models/check', { snapshot: m.snapshot, dir: dirs[m.snapshot] ?? m.dir })}>核对</button>
                    </>}
                </td>
              </tr>
            );
          })}</tbody>
        </table>
      )}
      <p className="v6-note">下载断点续传；暂停或失败后再点「继续下载」。需要登录授权的模型仓库暂不支持。</p>
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
        <RuntimeCard />
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
