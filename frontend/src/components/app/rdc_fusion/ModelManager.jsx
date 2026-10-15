/* ModelManager · 本地模型选择模块（空间透镜·实时分析·右侧「模型总览」上方；S9c）：
   模块条 = 当前模型名 + 「模型库 / 下载」按钮 → 弹窗：
   ① 已下载：GET /api/models/local（扫描 AI2050_MODELS_DIR 一级子目录 + HF 缓存快照，
      只收 config.json + *.safetensors 齐备目录）——行内「连接」→ POST /api/live/connect（复用现有连接协议）
   ② 模型目录：catalog 词表（Qwen3-4B / Qwen3-14B / GLM-4-9B / GPT-2）——「下载 ↓」→
      POST /api/models/download（服务端后台 huggingface_hub CLI，只下 safetensors 口径）
   ③ 下载任务：GET /api/models/downloads 轮询（有 downloading 时 2.5s），状态 + tail 回显
   LIVE/DEMO 纪律：端点不可达显示空态与错误，不伪造数据。 */
import { API_BASE } from '../../../config.js';
import { useEffect, useState } from 'react';

const fmtB = n => n >= (1 << 30) ? (n / (1 << 30)).toFixed(2) + ' GB'
  : n >= (1 << 20) ? (n / (1 << 20)).toFixed(1) + ' MB' : (n / 1024).toFixed(1) + ' KB';

export default function ModelManager({ conn, onConn }) {
  const [open, setOpen] = useState(false);
  const [lib, setLib] = useState(null);        // /models/local payload {root, models, catalog}
  const [dls, setDls] = useState([]);          // /models/downloads
  const [err, setErr] = useState('');
  const [connErr, setConnErr] = useState('');
  const [busyPath, setBusyPath] = useState('');

  const loadLib = () => {
    fetch(`${API_BASE}/api/models/local`)
      .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(d => { setLib(d); setErr(''); })
      .catch(e => setErr('模型库服务不可达：' + String(e.message || e)));
  };
  const loadDls = () => {
    fetch(`${API_BASE}/api/models/downloads`).then(r => r.ok ? r.json() : null)
      .then(d => { if (d) setDls(d.downloads || []); }).catch(() => {});
  };

  useEffect(() => { if (open) { loadLib(); loadDls(); } }, [open]);
  useEffect(() => {                             // 有进行中任务时轮询（2.5s），任务刷新本地列表
    if (!open || !dls.some(d => d.status === 'downloading')) return;
    const iv = setInterval(() => { loadDls(); loadLib(); }, 2500);
    return () => clearInterval(iv);
  }, [open, dls]);

  const download = async repoId => {
    setErr('');
    try {
      const r = await fetch(`${API_BASE}/api/models/download`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_id: repoId }),
      });
      if (!r.ok) { const t = await r.json().catch(() => ({})); throw new Error(t.detail || ('HTTP ' + r.status)); }
      loadDls();
    } catch (e) { setErr(String(e.message || e).includes('Failed to fetch') ? '中心节点不可达（:5001）' : String(e.message || e)); }
  };

  const connectPath = async p => {
    setBusyPath(p); setConnErr('');
    try {
      const r = await fetch(`${API_BASE}/api/live/connect`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_path: p }),
      });
      if (!r.ok) { const t = await r.json().catch(() => ({})); throw new Error(t.detail || ('HTTP ' + r.status)); }
      onConn && onConn(await r.json());
      setOpen(false);
    } catch (e) {
      setConnErr(String(e.message || e).includes('Failed to fetch') ? '中心节点不可达（:5001）' : String(e.message || e));
    } finally { setBusyPath(''); }
  };

  const cur = conn ? conn.model_path : null;
  const stOf = id => (dls.find(d => d.repo_id === id) || {}).status;

  return (
    <>
      {/* 模块条：模型总览上方 */}
      <div className="fw-lp-mmodel">
        <div className="hd">
          <b>本地模型</b>
          <span className="mono" title={cur || '未连接——从模型库选择或手填路径连接'}>{cur ? cur.split(/[\\/]/).pop() : '未连接'}</span>
        </div>
        <button type="button" className="fw-tbtn primary" onClick={() => setOpen(true)}>模型库 / 下载 →</button>
      </div>

      {/* 弹窗 */}
      {open && (
        <div className="fw-mmodal" onClick={() => setOpen(false)}>
          <div className="fw-mmodal-card" onClick={e => e.stopPropagation()}>
            <div className="fw-mmodal-hd">
              <b>模型库 · 本地模型管理与下载</b>
              {lib && <span className="dim2 mono" title="下载根目录（env AI2050_MODELS_DIR 可改）">{lib.root}</span>}
              <button type="button" className="fw-tbtn" onClick={() => setOpen(false)}>×</button>
            </div>

            {/* ① 已下载 */}
            <div className="fw-mmodal-sec">
              <h6>已下载 · 本机可连接 <span>{lib ? lib.models.length + ' 个（下载根目录 + HF 缓存快照）' : '扫描中…'}</span></h6>
              {lib && lib.models.map(m => (
                <div key={m.path} className="fw-mmodal-row">
                  <div className="info">
                    <b className="mono">{m.name}</b>
                    <span className="dim2 mono" title={m.path}>{m.path}</span>
                  </div>
                  <span className="meta">{m.arch || '—'} · {m.files} st · {fmtB(m.bytes)}{m.cached ? ' · 缓存' : ''}</span>
                  <button type="button" className={'fw-tbtn' + (cur === m.path ? ' primary' : '')}
                          disabled={busyPath === m.path || cur === m.path}
                          onClick={() => connectPath(m.path)}>
                    {cur === m.path ? '已连接' : (busyPath === m.path ? '连接中…' : '连接')}
                  </button>
                </div>
              ))}
              {lib && !lib.models.length && (
                <div className="dim2" style={{ padding: '4px 2px', fontSize: 10 }}>
                  本机暂无 safetensors 模型——从下方目录下载，或手动放置到下载根目录（config.json + *.safetensors）
                </div>
              )}
              {connErr && <div className="fw-ai-err" style={{ marginTop: 6 }}>{connErr}</div>}
            </div>

            {/* ② 模型目录 */}
            <div className="fw-mmodal-sec">
              <h6>模型目录 · 可下载 <span>后台 huggingface_hub CLI · 只下 safetensors（排除 bin/pth/gguf/onnx）</span></h6>
              {(lib ? lib.catalog : []).map(c => {
                const st = stOf(c.repo_id);
                return (
                  <div key={c.repo_id} className="fw-mmodal-row">
                    <div className="info">
                      <b className="mono">{c.name}</b>
                      <span className="dim2 mono">{c.repo_id}</span>
                    </div>
                    <span className="meta">{c.size} · {c.note}</span>
                    {st === 'downloading'
                      ? <span className="fw-mono" style={{ color: '#0f766e', fontSize: 10.5 }}>下载中…</span>
                      : <button type="button" className="fw-tbtn" disabled={!lib} onClick={() => download(c.repo_id)}>下载 ↓</button>}
                  </div>
                );
              })}
              {!lib && <div className="dim2" style={{ padding: '4px 2px', fontSize: 10 }}>目录加载中…</div>}
              {err && <div className="fw-ai-err" style={{ marginTop: 6 }}>{err}</div>}
            </div>

            {/* ③ 下载任务 */}
            {dls.length > 0 && (
              <div className="fw-mmodal-sec">
                <h6>下载任务 <span>{dls.filter(d => d.status === 'downloading').length} 个进行中</span></h6>
                {dls.map(d => (
                  <div key={d.repo_id} className="fw-mmodal-task">
                    <b className="mono">{d.repo_id}</b>
                    <span className={'st ' + d.status}>
                      {d.status === 'downloading' ? '● 下载中' : d.status === 'done' ? '✓ 完成' : '✗ 失败（exit ' + d.exit + '）'}
                    </span>
                    <span className="dim2">
                      {d.status === 'done' ? '已就绪——在「已下载」列表点击连接'
                        : (d.tail || []).slice(-1)[0] || d.dir}
                    </span>
                  </div>
                ))}
              </div>
            )}

            <p className="fw-lp-note">
              下载由服务端（:5001）后台执行 huggingface_hub CLI，完成后出现在「已下载」并可直接连接；
              需镜像加速时在启动后端前设置 HF_ENDPOINT（如 https://hf-mirror.com）。下载目录可用 env AI2050_MODELS_DIR 指定。
            </p>
          </div>
        </div>
      )}
    </>
  );
}
