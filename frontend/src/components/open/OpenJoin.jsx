/* 开放平台 · 参与：下载一键包 → 双击运行 → 自动领取、运行、上传。
   登录用户的一键包内嵌一次性绑定码（走 /api/auth/kit，令牌只在请求头里），节点注册后自动署名到该账号。 */
import { useEffect, useState } from 'react';

import { authJson, useApi } from '../app/rdc_fusion/api.js';
import { useAuth } from '../app/rdc_fusion/authContext.js';
import { Empty } from '../app/rdc_fusion/ui.jsx';
import { CensusJobBar } from './OpenHome.jsx';
import { downloadKit, fmtAgo, fmtInt, fmtTokens } from './openUtil.js';

const CONFIG_FIELDS = [
  ['server', '中心地址（已按下载来源填好）'],
  ['bind_code', '一次性绑定码（7 天内有效，注册后自动清除）；为空时节点不署名'],
  ['task_kinds', '参与的任务类型：census_shard（神经元普查分片）、family_test（模式族测试）'],
  ['model_dirs', '模型快照 → 本机目录；留空时程序会询问或帮你下载'],
  ['device', 'cuda、cuda:1 等'],
];

function MyContrib() {
  const [state, setState] = useState({ data: null, error: '' });
  useEffect(() => {
    let dead = false;
    authJson('/api/auth/my-contributions')
      .then(data => { if (!dead) setState({ data, error: '' }); })
      .catch(e => { if (!dead) setState({ data: null, error: e.message }); });
    return () => { dead = true; };
  }, []);
  if (state.error) return <Empty>{state.error}</Empty>;
  if (!state.data) return <Empty>读取我的节点…</Empty>;
  if (!state.data.nodes.length) return <Empty>还没有绑定到你账号的节点：下载一键包运行后，这里会出现你的节点和贡献。</Empty>;
  return (
    <table className="op-table">
      <thead><tr><th>节点</th><th>GPU</th><th>状态</th><th>模式族结果</th><th>普查分片</th><th>最近</th></tr></thead>
      <tbody>
        {state.data.nodes.map(n => {
          const acc = n.census.filter(c => c.status === 'accepted');
          return (
            <tr key={n.node_id}>
              <td><b>{n.name}</b> <span className="op-mono">{n.node_id}</span></td>
              <td>{n.gpu || '—'}</td>
              <td>{n.online ? '● 在线' : '○ 离线'}</td>
              <td>{fmtInt(n.family_results)}</td>
              <td>{acc.length} 通过{n.census.length > acc.length ? ` · ${n.census.length - acc.length} 其他` : ''} · {fmtTokens(acc.reduce((s, c) => s + (c.n_tokens || 0), 0))} token</td>
              <td>{fmtAgo(n.last_seen)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export default function OpenJoin({ config }) {
  const { user, guest, enterAuth } = useAuth();
  const loggedIn = Boolean(user) && !guest;
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');
  const [done, setDone] = useState(false);
  const contrib = useApi('/api/platform/contributors', { interval: 120000 });
  const jobs = useApi('/api/census/jobs', { interval: 60000 });
  const models = config?.models || [];

  const get = async bound => {
    setBusy(true); setErr('');
    try { await downloadKit(bound); setDone(true); } catch (e) { setErr(e.message); }
    setBusy(false);
  };

  return (
    <div className="op-page">
      <section className="op-section">
        <h2>用你的 GPU 参与研究</h2>
        <p className="op-lead">节点从中心领取冻结的任务包，在本机运行后只上传统计结果。任务包按 sha256 核对；中心逐项校验结果，并把一部分分片派给另一个账号复算比较。请不要修改任务包或节点程序，被改过的结果会被拒收。</p>
        <ol className="op-steps">
          <li>
            <b>下载一键包</b>
            {loggedIn ? (
              <span>已登录为 <b>{user.display_name || user.username}</b>：包里内嵌一次性绑定码，贡献自动署名。</span>
            ) : (
              <span>登录后下载，贡献会署名到你的账号；也可以匿名下载。</span>
            )}
            <div className="op-row">
              {loggedIn ? (
                <button type="button" className="op-btn primary" disabled={busy} onClick={() => get(true)}>{busy ? '准备中…' : '下载一键包（绑定我的账号）'}</button>
              ) : (
                <>
                  <button type="button" className="op-btn primary" onClick={enterAuth}>登录 / 注册</button>
                  <button type="button" className="op-btn" disabled={busy} onClick={() => get(false)}>匿名下载</button>
                </>
              )}
            </div>
            {err && <div className="op-err" role="alert">{err}</div>}
            {done && <div className="op-ok">已开始下载 ai2050-node.zip。</div>}
          </li>
          <li><b>解压并运行</b><span>Windows 双击 <code>start_windows.bat</code>；Linux 运行 <code>bash start_linux.sh</code>。第一次会创建 Python 环境并安装依赖（约 3 GB）。</span></li>
          <li><b>指定模型</b><span>程序会询问本机模型目录，或按作业指定的版本从 Hugging Face 下载；快照（权重与 tokenizer 的 sha256）一致才会领取任务。</span></li>
          <li><b>保持窗口打开</b><span>程序自动循环领取、运行、上传；关掉窗口即停止，未完成的分片会在租约到期后重新派发。</span></li>
        </ol>
      </section>

      <section className="op-section op-grid2">
        <div>
          <h2>需要什么</h2>
          <ul className="op-list">
            <li>NVIDIA GPU，显存 12 GB 以上（Qwen3-4B 普查分片实测峰值约 10 GB）</li>
            <li>磁盘约 15 GB（模型约 8 GB、依赖约 3 GB、缓存）</li>
            <li>Python 3.10 以上；可以访问中心服务器与 Hugging Face（或手动放好模型）</li>
          </ul>
          <h3>当前需要的模型</h3>
          {models.length === 0 ? <Empty>中心还没有登记需要的模型。</Empty> : (
            <table className="op-table">
              <thead><tr><th>模型</th><th>来源</th><th>快照</th></tr></thead>
              <tbody>{models.map(m => (
                <tr key={m.snapshot_sha256}><td><b>{m.name}</b></td><td>{m.hf_repo ? `${m.hf_repo}${m.hf_revision ? '@' + m.hf_revision.slice(0, 10) : ''}` : '—'}</td>
                  <td className="op-mono" title={m.snapshot_sha256}>{m.snapshot_sha256.slice(0, 16)}</td></tr>
              ))}</tbody>
            </table>
          )}
        </div>
        <div>
          <h2>隐私与安全</h2>
          <ul className="op-list">
            <li>只上传统计量（直方图、均值、极值位置等），不读取你电脑上的其他文件。</li>
            <li>节点令牌只保存在本机 <code>~/.ai2050_agent</code>；绑定码一次性。</li>
            <li>原生全场只保存在本机 <code>~/.ai2050</code>，不上传。</li>
            <li>节点程序是开源的单个 Python 文件，可以在仓库 <code>deploy/node_agent.py</code> 核对。</li>
          </ul>
        </div>
      </section>

      <section className="op-section">
        <h2>普查作业</h2>
        {jobs.error && <Empty>{jobs.error}</Empty>}
        {jobs.data && (jobs.data.jobs.length ? jobs.data.jobs.map(j => <CensusJobBar key={j.job_id} job={j} />) : <Empty>还没有开放的普查作业。</Empty>)}
      </section>

      {loggedIn && (
        <section className="op-section">
          <h2>我的节点与贡献</h2>
          <MyContrib />
        </section>
      )}

      <section className="op-section">
        <h2>贡献者</h2>
        {contrib.data && (contrib.data.items.length ? (
          <table className="op-table">
            <thead><tr><th>#</th><th>贡献者</th><th>节点</th><th>模式族结果</th><th>普查分片</th><th>普查 token</th></tr></thead>
            <tbody>{contrib.data.items.map((c, i) => (
              <tr key={c.name}><td>{i + 1}</td><td><b>{c.name}</b></td><td>{c.nodes}</td><td>{fmtInt(c.family_results)}</td><td>{fmtInt(c.census_shards)}</td><td>{fmtTokens(c.census_tokens)}</td></tr>
            ))}</tbody>
          </table>
        ) : <Empty>还没有署名的贡献。</Empty>)}
        {contrib.data && <p className="op-note">{contrib.data.note}</p>}
      </section>

      <details className="op-section op-adv">
        <summary>高级：配置与命令行</summary>
        <p className="op-lead">一键包里的 <code>config.json</code>：</p>
        <dl className="op-dl">{CONFIG_FIELDS.map(([k, v]) => <div key={k}><dt><code>{k}</code></dt><dd>{v}</dd></div>)}</dl>
        <p className="op-lead">命令行（在解压目录中）：</p>
        <pre className="op-pre">{`python node_agent.py participate --config config.json            # 一键参与（循环）
python node_agent.py participate --config config.json --once     # 只跑一轮
python node_agent.py status                                       # 本机节点状态`}</pre>
        <p className="op-note">中心：{config?.public_url} · 许可：代码 {config?.license?.code}，数据 {config?.license?.data}</p>
      </details>
    </div>
  );
}
