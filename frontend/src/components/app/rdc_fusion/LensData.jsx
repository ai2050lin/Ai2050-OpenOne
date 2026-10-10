/* 分析技术 → 分析技术视图（design/client_template_tech_plan_v1.md §3，M5-P0 主语反转）
   ─────────────────────────────────────────────────────────────
   与具体测试内容完全解耦：界面只渲染协议键，任何新模板注册后
   无需改本文件即可被筛选、浏览、展开。
   双模式：按技术（默认，主语=技术：选技术 → 契约匹配结果 → 展开自动运行）
          / 按结果（原结果浏览器，行内附属技术面板）。
   数据源（全部经 :5001 挂载的分布式路由）：
   - GET /api/templates            → 模板筛选器（D1-1 详情按需拉取）
   - GET /api/results              → 结果行（D1-3 summary_digest 平铺）
   - GET /api/results/{sha}        → 展开详情（summary 原文 + cos 矩阵提取）
   离线回退 distributedData.js 的 DEMO_RESULTS（协议结构演示行）。 */
import { useEffect, useState } from 'react';
import { TemplateCard, SummaryBrowser, CellCosGrid } from './protocolComponents.jsx';
import { TEMPLATES, DEMO_RESULTS, ANALYSES, DEMO_RDM } from './distributedData.js';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

/* 技术运行内核：对 detail 施加 tech → 输出对象（或 null=缺输入契约）
   rsa-rdm 客户端现算 RDM = 1 − cos；eta2-decompose 直读 summary.eta2_by_factor；
   离线时 rsa-rdm 回退 DEMO_RDM 并显式挂 demo 徽标。 */
function runTech(tech, detail, demo) {
  if (tech.id === 'rsa-rdm') {
    if (detail && detail.cos) {
      return { id: 'rsa-rdm', demo: false,
        rdm: { keys: detail.cos.keys, matrix: detail.cos.matrix.map(row => row.map(v => 1 - v)) } };
    }
    return demo ? { id: 'rsa-rdm', demo: true, rdm: DEMO_RDM } : null;
  }
  if (tech.id === 'eta2-decompose') {
    if (detail && detail.summary && detail.summary.eta2_by_factor) {
      return { id: 'eta2-decompose', demo: false, eta2: detail.summary.eta2_by_factor };
    }
    return null;
  }
  return null;
}

/* 分析技术面板（design/client_analysis_tech_plan_v1.md §2.3，P0/P1-lite）
   注册表 ← ANALYSES（distributedData.js）；可用性按 input 契约对当前 detail 匹配；
   受控模式：techId/setTechId 由视图级技术导航传入；autoRun=true 时详情到位即自动运行；
   hideChips=true 时隐藏面板内 chips（视图级导航已承担选择）。 */
function TechPanel({ detail, isLive, techId, setTechId, autoRun, hideChips }) {
  const [localTech, setLocalTech] = useState('rsa-rdm');
  const [ranTech, setRanTech] = useState(null);
  const cid = techId || localTech;
  const setCid = setTechId || setLocalTech;
  const tech = ANALYSES.find(a => a.id === cid) || ANALYSES[0];

  const contractOk = (key) => {
    if (key === 'result.cos') return Boolean(detail && detail.cos);
    if (key === 'result.summary.eta2_by_factor') return Boolean(detail && detail.summary && detail.summary.eta2_by_factor);
    if (key === 'result2.cos') return false;                    // P2：双结果选择
    return true;
  };
  const avail = tech.input.every(contractOk);
  const demo = !isLive;                                         // 离线 → DEMO 回退
  const runnable = avail || (demo && tech.id === 'rsa-rdm');    // DEMO 仅 rsa-rdm 有演示输出

  const run = () => { setRanTech(runTech(tech, detail, demo)); };

  /* 自动运行：autoRun 模式下详情到位即施加当前技术（按技术视图） */
  useEffect(() => {
    if (!autoRun) return;
    if (isLive && !detail) { setRanTech(null); return; }        // LIVE 详情拉取中
    setRanTech(runTech(tech, detail, demo));
    /* eslint-disable-next-line react-hooks/exhaustive-deps */
  }, [autoRun, detail, cid]);
  return (
    <div className="fw-tech">
      {!hideChips && (
        <>
          <div className="fw-tech-bar">
            <b>分析技术</b>
            <span className="fw-tech-tip">注册表驱动 · 按输入契约过滤可用性 · 口径各自登记，禁止跨量纲平均</span>
            {demo && <span className="fw-src-chip demo" title="离线演示输出——协议结构演示数据">○ DEMO 输出</span>}
          </div>
          <div className="fw-tech-sel">
            {ANALYSES.map(a => {
              const aAvail = a.input.every(contractOk);
              const aRunnable = aAvail || (demo && a.id === 'rsa-rdm');
              return (
                <button key={a.id} type="button" title={a.disabled_reason || a.note}
                        className={'fw-tech-chip' + (cid === a.id ? ' on' : '') + (aRunnable ? '' : ' off')}
                        onClick={() => { if (aRunnable) { setCid(a.id); setRanTech(null); } }}>
                  {a.name}
                  {!aRunnable && <small>·{a.disabled_reason ? '未开放' : '缺输入'}</small>}
                </button>
              );
            })}
          </div>
        </>
      )}
      <div className="fw-tech-meta">
        <span className="fw-mono">{tech.metric_version}</span>
        <span className="fw-mono dim2">证据级 {tech.evidence_level}</span>
        <span className="dim2">{tech.note}</span>
      </div>
      {runnable && ranTech && ranTech.id === 'rsa-rdm' && ranTech.rdm && (
        <TechRdm out={ranTech}/>
      )}
      {runnable && ranTech && ranTech.id === 'eta2-decompose' && ranTech.eta2 && (
        <TechEta2 eta2={ranTech.eta2}/>
      )}
      {runnable && !ranTech && (
        <div className="fw-tech-hint">
          {autoRun
            ? (isLive ? (detail ? '当前结果缺少该技术的输入契约（如 cos 矩阵）' : '拉取详情后自动运行…') : '离线演示仅开放 RSA/RDM（DEMO 输出）')
            : '点击「运行」施加当前技术 → 输出经协议键渲染'}
        </div>
      )}
      {!runnable && <div className="fw-tech-hint">{tech.disabled_reason || (isLive ? '当前结果缺少该技术的输入契约（如 cos 矩阵）' : '离线演示仅开放 RSA/RDM')}</div>}
      {runnable && ranTech && !autoRun && (
        <button type="button" className="fw-tbtn" style={{ marginTop: 6 }} onClick={() => setRanTech(null)}>清除输出</button>
      )}
    </div>
  );
}

/* rsa-rdm 输出：RDM 热图（复用 CellCosGrid mode=rdm）+ 关系摘要（最分离/最相似条件对） */
function TechRdm({ out }) {
  const { keys, matrix } = out.rdm;
  let best = null, worst = null;
  for (let i = 0; i < keys.length; i++) for (let j = i + 1; j < keys.length; j++) {
    const v = matrix[i][j];
    if (!best || v > best.v) best = { a: keys[i], b: keys[j], v };
    if (!worst || v < worst.v) worst = { a: keys[i], b: keys[j], v };
  }
  return (
    <div className="fw-tech-out">
      {out.demo && <div className="fw-tech-demo-note">演示矩阵（协议结构演示）——接入真实结果后此处为该行 means.npz cos 现算值</div>}
      <CellCosGrid cos={{ keys, matrix }} mode="rdm"/>
      <div className="fw-tech-sum">
        <span>{keys.length} 条件</span>
        <span>最分离对 <b className="fw-mono">{best.a} × {best.b}</b>（dissim {best.v.toFixed(2)}）</span>
        <span>最相似对 <b className="fw-mono">{worst.a} × {worst.b}</b>（dissim {worst.v.toFixed(2)}）</span>
      </div>
    </div>
  );
}

/* eta2-decompose 输出：因子条形视图（数据=summary.eta2_by_factor，协议键直读） */
function TechEta2({ eta2 }) {
  const entries = Object.entries(eta2).filter(([, v]) => typeof v === 'number');
  const max = Math.max(0.001, ...entries.map(([, v]) => Math.abs(v)));
  return (
    <div className="fw-tech-out">
      <div className="fw-tech-bars">
        {entries.map(([k, v]) => (
          <div key={k} className="fw-tech-barrow" title={`${k} = ${v.toFixed(4)}`}>
            <span className="fw-mono">{k}</span>
            <span className="fw-tech-bartrack"><i style={{ width: `${(Math.abs(v) / max) * 100}%` }}/></span>
            <b className="fw-mono">{v.toFixed(3)}</b>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function LensData({ on }) {
  const [live, setLive] = useState(null);          // {templates, results_total, downloads_total}
  const [rows, setRows] = useState(null);          // 结果行（D1-3 平铺）
  const [filter, setFilter] = useState('');        // tm_id 筛选
  const [tplMeta, setTplMeta] = useState({});      // tm_id → D1-1 meta 缓存
  const [expandedSha, setExpandedSha] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [err, setErr] = useState('');
  const [viewMode, setViewMode] = useState('tech');   // 按技术（默认，主语反转）/ 按结果
  const [techId, setTechId] = useState('rsa-rdm');    // 视图级技术选择（按技术模式）

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
  const techDef = ANALYSES.find(a => a.id === techId) || ANALYSES[0];
  /* 按技术模式：先按模板筛选，再按「行级可判定的输入契约」预过滤
     （eta2 的输入在 summary_digest 行级即可判定；cos 类契约须展开详情后判定，故不预过滤） */
  const techRows = filtered.filter(r =>
    techDef.id !== 'eta2-decompose' || (r.summary_digest && r.summary_digest.eta2_by_factor));

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

        {/* 视图模式：按技术（主语反转，M5-P0）/ 按结果（原浏览器） */}
        <div className="fw-tech-seg">
          <button type="button" className={viewMode === 'tech' ? 'on' : ''} onClick={() => setViewMode('tech')}>
            按技术<small>选技术 → 契约匹配结果 → 展开自动运行</small>
          </button>
          <button type="button" className={viewMode === 'results' ? 'on' : ''} onClick={() => setViewMode('results')}>
            按结果<small>结果浏览器（展开后行内选技术）</small>
          </button>
        </div>

        {/* 按技术模式：视图级技术导航（注册表驱动，M4 TechPanel 升格） */}
        {viewMode === 'tech' && (
          <div className="fw-tech-nav">
            <div className="fw-tech-sel">
              {ANALYSES.map(a => (
                <button key={a.id} type="button" title={a.disabled_reason || a.note}
                        className={'fw-tech-chip' + (techId === a.id ? ' on' : '') + (a.disabled_reason ? ' off' : '')}
                        onClick={() => { if (!a.disabled_reason) setTechId(a.id); }}>
                  {a.name}{a.disabled_reason && <small>·未开放</small>}
                </button>
              ))}
            </div>
            <div className="fw-tech-meta">
              <span className="fw-mono">{techDef.metric_version}</span>
              <span className="fw-mono dim2">证据级 {techDef.evidence_level}</span>
              <span className="dim2">{techDef.note}</span>
            </div>
          </div>
        )}

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
                {viewMode === 'tech' ? `技术 · ${techDef.name}` : '结果浏览器'}
                <span>{filter ? `筛选 ${filter}` : '全部模板'} · 点击行展开{viewMode === 'tech' ? '，详情到位后自动按输入契约运行当前技术' : ' summary 原文 / η² 分解 / cos 矩阵'}</span>
              </h6>
              <SummaryBrowser rows={viewMode === 'tech' ? techRows : filtered} expandedSha={expandedSha} onExpand={expand}
                              detail={detail} detailLoading={detailLoading} factorsByTm={factorsMap}/>
              {expandedSha && (
                <TechPanel detail={detail} isLive={isLive} detailLoading={detailLoading}
                           techId={viewMode === 'tech' ? techId : undefined}
                           setTechId={viewMode === 'tech' ? setTechId : undefined}
                           autoRun={viewMode === 'tech'} hideChips={viewMode === 'tech'}/>
              )}
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
