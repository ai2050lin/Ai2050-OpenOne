/* 本地数据状态条：显示本机数据包的版本与时间，服务器有新数据时提示手动更新。
   只读数据包模式下第一次打开（本机还没有数据包）自动下载一次主数据包（一个静态文件），之后只提示、不自动下载。
   数据包内容见 deploy/datapack.py；本地存储与读取见 rdc_fusion/dataStore.js。 */
import { useEffect, useRef, useState, useSyncExternalStore } from 'react';

import { API_BASE, OFFLINE_CLIENT, useApi } from '../app/rdc_fusion/api.js';
import './databar.css';
import { downloadDataPack, downloadExtra, packInfo, storeTick, subscribe } from '../app/rdc_fusion/dataStore.js';

function fmtTime(ts) {
  if (!ts) return '—';
  const d = new Date(ts * 1000);
  const pad = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function fmtMB(b) {
  return b ? `${(b / 2 ** 20).toFixed(1)} MB` : '';
}

export default function DataBar({ remote }) {
  useSyncExternalStore(subscribe, storeTick);
  const info = packInfo();
  const [busy, setBusy] = useState(null);
  const [err, setErr] = useState('');
  const get = async () => {
    setErr(''); setBusy({ got: 0, total: remote?.bytes || 0 });
    try {
      await downloadDataPack(API_BASE, (got, total) => setBusy({ got, total: total || remote?.bytes || 0 }));
    } catch (e) {
      setErr(e.message);
    }
    setBusy(null);
  };
  const auto = useRef(false);
  useEffect(() => {                                    // 第一次打开：自动下载主数据包（只尝试一次，失败后改为手动）
    if (!OFFLINE_CLIENT || auto.current || !info.ready || info.version || !remote?.version) return;
    auto.current = true;
    window.setTimeout(get, 0);                         // 不在副作用里同步改状态；只触发一次（auto 守卫）
  });
  if (!info.ready) return null;
  const fresh = info.version && remote?.version && info.version === remote.version;
  const label = busy ? `下载中 ${busy.total ? Math.round((100 * busy.got) / busy.total) : 0}%`
    : !info.version ? `下载数据${remote?.bytes ? `（${fmtMB(remote.bytes)}）` : ''}`
      : `更新数据${remote?.bytes ? `（${fmtMB(remote.bytes)}）` : ''}`;
  return (
    <div className={'op-databar' + (!info.version ? ' none' : fresh ? ' fresh' : ' stale')}
         title={OFFLINE_CLIENT ? '页面只读取本机保存的数据包，使用中不连接服务器；服务器有新数据时手动更新'
           : '页面优先读取本机保存的数据包，不会定时访问服务器；单元卡与管理类页面打开时在线读取'}>
      <span className="op-databar-dot" aria-hidden="true" />
      <span className="op-databar-text">
        {!info.version ? (OFFLINE_CLIENT ? (busy ? '正在下载数据包（只下载一次，之后页面都读本机数据）' : '还没有本地数据，下载后各页面显示内容') : '未下载本地数据（页面临时在线读取）')
          : fresh ? `本地数据 · ${fmtTime(info.createdAt)}`
            : remote?.version ? `本地数据 ${fmtTime(info.createdAt)} · 服务器有新数据` : `本地数据 · ${fmtTime(info.createdAt)}`}
      </span>
      {(!fresh || busy) && remote?.version && (
        <button type="button" className="op-databar-btn" onClick={get} disabled={Boolean(busy)}>{label}</button>
      )}
      {err && <span className="op-databar-err" role="alert">{err}</span>}
    </div>
  );
}

/* 全局挂载：读一次站点配置拿到服务器端数据包版本，固定显示在页面左下角（首页与工作台共用）。 */
export function DataBarHost() {
  const cfg = useApi('/api/platform/config');
  return (
    <div className="op-databar-host">
      <DataBar remote={cfg.data?.datapack} />
    </div>
  );
}

/* 附加包下载提示：离线客户端在缺少某个附加包（模式族实验响应、单元卡）时显示，下载后页面自动重读。 */
export function ExtraPackPrompt({ pack }) {
  useSyncExternalStore(subscribe, storeTick);
  const cfg = useApi('/api/platform/config');
  const [busy, setBusy] = useState(null);
  const [err, setErr] = useState('');
  if (!pack) return null;
  const meta = (cfg.data?.datapack?.extras || []).find(e => e.id === pack);
  const have = packInfo().extras[pack];
  const get = async () => {
    setErr(''); setBusy({ got: 0, total: meta?.bytes || 0 });
    try {
      await downloadExtra(API_BASE, pack, (got, total) => setBusy({ got, total: total || meta?.bytes || 0 }));
    } catch (e) {
      setErr(e.message);
    }
    setBusy(null);
  };
  const stale = Boolean(have && meta?.version && have.version !== meta.version);   // 服务器上的附加包已更新
  if (have && !stale && !busy) return null;
  return (
    <div className="op-extra">
      <span>{meta ? `${stale ? '附加包有新版本' : '需要附加包'}「${meta.label}」（${fmtMB(meta.bytes)}）` : cfg.data ? '服务器还没有生成这个附加包，稍后再试' : '读取附加包清单…'}</span>
      {meta && (
        <button type="button" className="op-databar-btn" onClick={get} disabled={Boolean(busy)}>
          {busy ? `下载中 ${busy.total ? Math.round((100 * busy.got) / busy.total) : 0}%` : stale ? '更新' : '下载'}
        </button>
      )}
      {err && <span className="op-databar-err" role="alert">{err}</span>}
    </div>
  );
}
