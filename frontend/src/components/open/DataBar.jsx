/* 本地数据状态条：显示本机数据包的版本与时间，服务器有新数据时提示手动更新（不会自动下载）。
   数据包内容见 deploy/datapack.py；本地存储与读取见 rdc_fusion/dataStore.js。 */
import { useState, useSyncExternalStore } from 'react';

import { API_BASE, useApi } from '../app/rdc_fusion/api.js';
import './databar.css';
import { downloadDataPack, packInfo, storeTick, subscribe } from '../app/rdc_fusion/dataStore.js';

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
  if (!info.ready) return null;
  const fresh = info.version && remote?.version && info.version === remote.version;
  const label = busy ? `下载中 ${busy.total ? Math.round((100 * busy.got) / busy.total) : 0}%`
    : !info.version ? `下载数据${remote?.bytes ? `（${fmtMB(remote.bytes)}）` : ''}`
      : `更新数据${remote?.bytes ? `（${fmtMB(remote.bytes)}）` : ''}`;
  return (
    <div className={'op-databar' + (!info.version ? ' none' : fresh ? ' fresh' : ' stale')}
         title="页面优先读取本机保存的数据包，不会定时访问服务器；单元卡与管理类页面打开时在线读取">
      <span className="op-databar-dot" aria-hidden="true" />
      <span className="op-databar-text">
        {!info.version ? '未下载本地数据（页面临时在线读取）'
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
