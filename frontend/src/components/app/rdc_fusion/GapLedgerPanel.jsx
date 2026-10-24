/* GapLedgerPanel GAP 账本面板 v1（Phase 3255 工程臂；registry v1.34 客户端化）：
   数据 = GET /api/ai-rnd/atlas/gaps（glob 自动发现最新版；v1.12 起含 appendix）。
   渲染：版本链 + supersedes → 缺口状态行（GAP-1..5）→ 附录卡（APPX-1..6，
   APPX-6 = 保真收线链 3239–3250 + u* 词汇解耦，3254 落册，anchor p3250/9d0f1825）。
   一切文字来自端点协议键（id/title/status/note/anchor/anchor_sha8），零实验字面量；
   LIVE/DEMO 范式：服务未启动时空态提示，不伪造数据。 */
import { API_BASE } from '../../../config.js';
import { useEffect, useState } from 'react';

export default function GapLedgerPanel() {
  const [d, setD] = useState(null);
  const [live, setLive] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let dead = false;
    fetch(`${API_BASE}/api/ai-rnd/atlas/gaps`).then(r => {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }).then(j => { if (!dead) { setD(j); setLive(true); } })
      .catch(e => { if (!dead) { setErr(String(e.message || e)); setLive(false); } });
    return () => { dead = true; };
  }, []);

  return (
    <div className="glp-root" data-k="gap_ledger_panel">
      <div className="at-pan-head" style={{ marginTop: 10 }}>
        <b>GAP 账本 · 附录</b>
        <span className="dim2">缺口收线状态 + 附录卡（GET /api/ai-rnd/atlas/gaps，自动发现最新版）</span>
        {live !== null && <span className={'at-badge ' + (live ? 'at-live' : 'at-dead')}>{live ? 'LIVE' : '未连接'}</span>}
      </div>
      {err && <div className="at-err">GAP 账本不可达（{err}）——启动 :5001 后刷新。</div>}
      {!d && !err && <div className="dim2">加载中…</div>}
      {d && (
        <>
          <div className="at-summary">
            <span className="at-kv">gap 账本 v{d.version}（{d.file}）</span>
            <span className="at-kv">updated {d.updated}</span>
            <span className="at-kv mono" title={d.supersedes}>supersedes: {d.supersedes}</span>
            {d.versions && d.versions.length > 1 && (
              <span className="at-kv" title={d.versions.map(v => 'v' + v.version).join(' ← ')}>
                版本链 {d.versions.length} 代（最新 → 最旧）
              </span>
            )}
          </div>
          <div className="glp-gaps">
            {(d.gaps || []).map(g => (
              <span key={g.id} className={'glp-gap glp-' + String(g.status).split('_')[0]}
                title={g.title || ''}>
                <span className="mono">{g.id}</span>
                <span className="glp-status">{g.status}</span>
              </span>
            ))}
          </div>
          <div className="at-grid">
            {(d.appendix || []).map(a => (
              <div key={a.id} className={'at-card glp-appx' + (a.id === 'APPX-6' ? ' glp-newest' : '')}>
                <div className="at-card-head">
                  <span className="mono at-fid">{a.id}</span>
                  <span className="at-fam">{a.anchor}</span>
                  <span className="at-badge at-live">{a.status}</span>
                </div>
                <div className="at-stmt">{a.title}</div>
                <div className="glp-note">{a.note}</div>
                <div className="at-anchors-line mono">anchor_sha8={a.anchor_sha8}</div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
