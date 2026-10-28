/* 研究工作台 · 配置：当前连接的中心与运行形态、账号、被测模型、平台信息。
   纪律：本页只展示「运行时判定（config.js）」与「中心实况（/api/platform/config）」，不摆会被静默丢弃的开关——
   界面上的每个值都能在代码或接口里找到来源。真正的写操作只有「登录 / 注册」与「退出登录」，其余是到对应页面的链接。
   被测模型来自中心 config.models（只认 snapshot_sha256 完整的条目），空时如实说明，不用示例数据填充。 */
import { Fragment } from 'react';

import { RUNTIME_MODE } from '../../config.js';
import { ADMIN_LIVE, API_BASE, DESKTOP, OFFLINE_CLIENT, REMOTE_CENTER, WEB_URL } from './rdc_fusion/api.js';
import { useAuth } from './rdc_fusion/authContext.js';
import { Card, Empty, StatusPill } from './rdc_fusion/ui.jsx';
import { siteHref } from '../v8/v8Util.js';

const MODE = {
  electron: '桌面客户端（安装包）',
  local: '本机开发（连本机中心）',
  remote: '远端客户端（连另一台主机）',
  'same-origin': '网页版（与中心同源托管）',
};

function KV({ rows }) {
  return (
    <dl className="wb-cfg-kv">
      {rows.filter(Boolean).map(([k, v]) => (
        <Fragment key={k}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </Fragment>
      ))}
    </dl>
  );
}

function short(s, n = 12) {
  const v = String(s || '');
  return v.length > n ? `${v.slice(0, n)}…` : v;
}

export default function ConfigView({ cfg, onGo }) {
  const { user, guest, offline, logout, enterAuth } = useAuth();
  const d = cfg?.data || null;
  const models = d?.models || [];
  const center = API_BASE || (typeof window !== 'undefined' ? window.location.origin : '');
  const name = user ? (user.display_name || user.username) : '';
  const role = guest ? ['na', '访客'] : (user?.role === 'admin' ? ['pass', '管理员'] : ['na', '成员']);
  const conn = cfg?.pending ? ['na', '读取中…'] : (cfg?.error ? ['fail', '连接失败'] : ['pass', '已连接']);

  return (
    <div className="wb-cfg">
      <Card title="连接与运行形态" sub="决定哪些界面可读、哪些可写">
        <KV rows={[
          ['中心服务', <span key="c" className="wb-cfg-mono">{center || '（同源，未配置地址）'}</span>],
          ['运行形态', MODE[RUNTIME_MODE] || RUNTIME_MODE || '未知'],
          ['连接状态', <span key="s"><StatusPill status={conn[0]} label={conn[1]} />{cfg?.error ? <small className="wb-cfg-warn"> {cfg.error}</small> : null}</span>],
          ['数据模式', OFFLINE_CLIENT
            ? <>只读数据包<small className="wb-cfg-sub">除登录、下载与配置读取外不联网</small></>
            : <>在线<small className="wb-cfg-sub">读取与写操作实时连接中心</small></>],
          ['研发工位', ADMIN_LIVE
            ? <>可写<small className="wb-cfg-sub">管理工位另需管理员账号</small></>
            : <>只读<small className="wb-cfg-sub">浏览器只能浏览题库；执行测试与管理在桌面客户端</small></>],
          ['本机工具', d?.local_tools
            ? <>可用<small className="wb-cfg-sub">自动测试、读取本机模型权重</small></>
            : <>不可用<small className="wb-cfg-sub">中心未开放本机工具（公开中心没有被测模型）</small></>],
          ['沙盒数据', d?.sandbox_synthetic
            ? <StatusPill status="warn" label="是（合成数据，不可作研究结论）" />
            : <StatusPill status="pass" label="否（未启用合成数据）" />],
          REMOTE_CENTER && ['网页版地址', <a key="w" className="wb-cfg-link" href={WEB_URL} target="_blank" rel="noreferrer">{WEB_URL || center}</a>],
        ]} />
      </Card>

      <Card title="账号" sub={offline ? '中心服务暂不可达，显示的是本机保存的登录信息' : '登录后可领题测试、绑定节点、署名贡献'}
            extra={!user && <button type="button" className="fw-tbtn primary" onClick={enterAuth}>登录 / 注册</button>}>
        {user ? (
          <>
            <KV rows={[
              ['显示名', <b key="n">{name}</b>],
              ['用户名', <span key="u" className="wb-cfg-mono">{user.username}</span>],
              ['角色', <StatusPill key="r" status={role[0]} label={role[1]} />],
            ]} />
            <div className="wb-cfg-act">
              {guest && <button type="button" className="fw-tbtn primary" onClick={enterAuth}>登录 / 注册</button>}
              {!guest && <button type="button" className="fw-tbtn" onClick={logout}>退出登录</button>}
              {user.role === 'admin' && <a className="fw-tbtn" href={siteHref('accounts')}>账号管理</a>}
              {user.role === 'admin' && <a className="fw-tbtn" href={siteHref('data')}>数据管理</a>}
            </div>
          </>
        ) : (
          <Empty>还没有登录信息。登录后可领题测试、绑定节点、署名贡献；不登录也能以访客身份浏览全部内容。</Empty>
        )}
      </Card>

      <Card title="AI 助手" sub="模型配置与 Key 只保存在本机，不经过中心"
            extra={<button type="button" className="fw-tbtn" onClick={() => onGo('ai')}>打开模型配置</button>}>
        <KV rows={[
          ['可用形态', DESKTOP
            ? <StatusPill status="pass" label="桌面客户端（Key 由主进程加密保存）" />
            : <StatusPill status="na" label="当前形态不可用" />],
          ['入口', '研究工作台右下角「问 AI」，可附带当前页数据（语言项、神经元单元、测试报告）'],
        ]} />
        <p className="wb-cfg-note">回答一律标为「未检验」，不是证据；值得验证的想法要写成模板或测试运行。</p>
      </Card>

      <Card title="被测模型" sub="中心登记、且带完整快照校验的模型"
            extra={d && <span className="wb-cfg-sub">{models.length} 个</span>}>
        {!d && (cfg?.error ? <Empty>读不到中心配置：{cfg.error}</Empty> : <Empty>读取中心配置…</Empty>)}
        {d && !models.length && <Empty>中心没有登记被测模型。公开中心不保管权重，测试需要在本机或贡献节点上用本地权重运行。</Empty>}
        {models.length > 0 && (
          <table className="wb-cfg-table">
            <thead><tr><th>模型</th><th>仓库 · 修订</th><th>最低显存</th><th>快照 sha256</th></tr></thead>
            <tbody>
              {models.map(m => (
                <tr key={m.name}>
                  <th scope="row">{m.name}</th>
                  <td className="wb-cfg-mono">{m.hf_repo || '—'}{m.hf_revision ? <small> @{short(m.hf_revision, 8)}</small> : null}</td>
                  <td>{m.min_vram_gib ? `${m.min_vram_gib} GiB` : '—'}</td>
                  <td className="wb-cfg-mono" title={m.snapshot_sha256 || ''}>{m.snapshot_sha256 ? short(m.snapshot_sha256) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      <Card title="平台" sub="中心自报的配置（platform_config.v1）">
        {!d ? <Empty>读取中心配置…</Empty> : (
          <>
            <KV rows={[
              ['标题', d.title || '—'],
              ['版本', <span key="v" className="wb-cfg-mono">{d.version || '—'}{d.schema ? ` · ${d.schema}` : ''}</span>],
              ['注册策略', d.registration === 'open' ? '开放注册' : (d.registration || '—')],
              ['任务类型', (d.task_kinds || []).join(' · ') || '—'],
              ['许可', d.license ? `代码 ${d.license.code || '—'} · 数据 ${d.license.data || '—'}` : '—'],
              d.repo_url && ['仓库', <a key="r" className="wb-cfg-link" href={d.repo_url} target="_blank" rel="noreferrer">{d.repo_url}</a>],
              d.public_url && ['公开地址', <a key="p" className="wb-cfg-link" href={d.public_url} target="_blank" rel="noreferrer">{d.public_url}</a>],
            ]} />
            <p className="wb-cfg-note">这些值由中心下发，本页不做本地覆盖；换中心请改客户端首启填写的服务器地址。</p>
          </>
        )}
      </Card>
    </div>
  );
}
