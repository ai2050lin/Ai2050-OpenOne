/* 数据透镜 v1：结果浏览器（design/ui_decoupled_plan_v1.md §3「数据 → SummaryBrowser」）
   ─────────────────────────────────────────────────────────────
   与具体测试内容完全解耦：界面只渲染协议键，任何新模板注册后
   无需改本文件即可被筛选、浏览、展开。
   数据源（全部经 :5001 挂载的分布式路由）：
   - GET /api/templates            → 模板筛选器（D1-1 详情按需拉取）
   - GET /api/results              → 结果行（D1-3 summary_digest 平铺）
   - GET /api/results/{sha}        → 展开详情（summary 原文 + cos 矩阵提取）
   离线回退 distributedData.js 的 DEMO_RESULTS（协议结构演示行）。 */
import { useEffect, useState } from 'react';
import { TemplateCard, SummaryBrowser } from './protocolComponents.jsx';
import { TEMPLATES, DEMO_RESULTS } from './distributedData.js';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

export default function LensData({ on }) {
  const [live, setLive] = useState(null);          // {templates, results_total, downloads_total}
  const [rows, setRows] = useState(null);          // 结果行（D1-3 平铺）
  const [filter, setFilter] = useState('');        // tm_id 筛选
  const [tplMeta, setTplMeta] = useState({});      // tm_id → D1-1 meta 缓存
  const [expandedSha, setExpandedSha] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [err, setErr] = useState('');

  useEffect(() => {
    let dead = false;
    (async () => {
      const ctl = new AbortController();
      const t = setTimeout(() => ctl.abort(), 4000);
      try {
        const [tp, rs] = await Promise.all([
          fetch(`${API_BASE}/api/templates`, { signal: ctl.signal }).then(r => r.ok ? r.json() : null).catch(() => null),
          fetch(`${API_BASE}/api/results?limit=100`, { signal: ctl.signal }).then(r => r.ok ? r.json() : null).catch(() => null),
        ]);
        if (dead) return;
        if (tp && Array.isArray(tp.templates)) {
          setLive({ templates: tp.templates });
          setRows(rs && Array.isArray(rs.results) ? rs.results : []);
        }
      } catch { /* 离线 → demo */ }
      clearTimeout(t);
    })();
    return () => { dead = true; };
  }, []);

  const loadRows = async (tm) => {
    setFilter(tm); setExpandedSha(null); setDetail(null);
    if (!live) return;
    try {
      const r = await fetch(`${API_BASE}/api/results?limit=100${tm ? `&tm_id=${encodeURIComponent(tm)}` : ''}`);
      const p = r.ok ? await r.json() : null;
      if (p && Array.isArray(p.results)) setRows(p.results);
    } catch { setErr('结果列表拉取失败'); }
  };

  /* 展开行：拉详情（含 cos 提取）+ 该模板的 D1-1 因子名 */
  const expand = async (sha) => {
    if (!sha) { setExpandedSha(null); setDetail(null); return; }
    setExpandedSha(sha); setDetail(null);
    if (!live) return;
    const row = rows.find(r => r.sha === sha);
    setDetailLoading(true);
    try {
      if (row && !tplMeta[row.tm_id]) {
        const t = await fetch(`${API_BASE}/api/templates/${encodeURIComponent(row.tm_id)}`)
          .then(r => r.ok ? r.json() : null).catch(() => null);
        if (t && t.meta) setTplMeta(m => ({ ...m, [row.tm_id]: t.meta }));
      }
      const d = await fetch(`${API_BASE}/api/results/${encodeURIComponent(sha)}`)
        .then(r => r.ok ? r.json() : null).catch(() => null);
      if (d && d.sha) setDetail(d); else setErr('结果详情不可用');
    } catch { setErr('结果详情拉取失败'); }
    setDetailLoading(false);
  };

  const isLive = Boolean(live);
  const tpls = isLive ? live.templates : TEMPLATES;
  const viewRows = isLive ? rows : DEMO_RESULTS;
  const filtered = filter ? viewRows.filter(r => (r.tm_id) === filter) : viewRows;
  const metaOf = tm => tplMeta[tm] || null;
  const factorsMap = Object.fromEntries(Object.entries(tplMeta).map(([k, m]) => [k, m.factors || []]));
  const resultsTotal = isLive ? viewRows.length : DEMO_RESULTS.length;

  return (
    <section className={'fw-view fw-data' + (on ? ' on' : '')}>
      <div className="fw-rg-inner">
        {/* 顶部：数据源徽标 + 统计 */}
        <div className="fw-pl-stats" style={{ marginBottom: 12 }}>
          <span className={'fw-src-chip ' + (isLive ? 'live' : 'demo')}
                title={isLive ? 'GET /api/results · D1-3 摘要平铺' : '中心节点离线——显示协议结构演示行'}>
            {isLive ? '● LIVE 中心节点结果库' : '○ DEMO（协议结构演示）'}
          </span>
          <div className="fw-pl-stat"><div className="n">{resultsTotal}</div><div className="l">结果条目</div><div className="s">sha256 内容寻址</div></div>
          <div className="fw-pl-stat"><div className="n">{tpls.length}</div><div className="l">模板</div><div className="s">注册表 S2 · 任意句型可扩展</div></div>
          <div className="fw-pl-stat"><div className="n">{new Set((viewRows || []).map(r => r.tm_id)).size}</div><div className="l">有结果模板</div><div className="s">行=一次上传</div></div>
        </div>

        <div className="fw-data-grid">
          {/* 左列：模板筛选器（协议驱动 TemplateCard） */}
          <div className="fw-data-side">
            <div className="fw-pl-card">
              <h6>模板筛选器 <span>每模板=受控语料+固定口径</span></h6>
              <div className="fw-data-tpllist">
                <button type="button" className={'fw-pc-tcard all' + (!filter ? ' on' : '')} onClick={() => loadRows('')}>
                  <span className="fw-pc-tid">全部模板</span>
                  <span className="fw-pc-tname">不过滤——浏览结果库全部条目</span>
                </button>
                {tpls.map(t => (
                  <TemplateCard key={t.tm_id || t.id} tpl={t} detail={metaOf(t.tm_id || t.id)}
                                active={filter === (t.tm_id || t.id)}
                                onClick={() => loadRows(t.tm_id || t.id)}/>
                ))}
              </div>
            </div>
          </div>

          {/* 主体：结果浏览器（SummaryBrowser 协议组件） */}
          <div className="fw-data-main">
            {err && <div className="fw-ai-err" style={{ marginBottom: 8 }}>{err}</div>}
            <div className="fw-pl-card">
              <h6>
                结果浏览器
                <span>{filter ? `筛选 ${filter}` : '全部模板'} · 点击行展开 summary 原文 / η² 分解 / cos 矩阵</span>
              </h6>
              <SummaryBrowser rows={filtered} expandedSha={expandedSha} onExpand={expand}
                              detail={detail} detailLoading={detailLoading} factorsByTm={factorsMap}/>
            </div>
            <div className="fw-rg-note" style={{ marginTop: 10 }}>
              <b>解耦约定</b>：本界面是协议（模板 + 产出 schema）的通用渲染器——只认识
              <span className="fw-mono"> factors / eta2_by_factor / per_layer / fingerprint / cos </span>
              等协议键；加一种句型 = 注册一个新模板（POST /api/templates），此处自动出现并正确渲染，零界面改动。
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
