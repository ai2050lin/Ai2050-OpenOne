/* 协议驱动 UI 通用组件 v1（design/ui_decoupled_plan_v1.md §2 五件）
   ─────────────────────────────────────────────────────────────
   验收红线（§0）：组件只认识协议键——factors[]、eta2_by_factor{}、per_layer{}、
   fingerprint{}、eta2_by_cond{}、cos{keys,matrix}、topk_jaccard{}。
   禁止出现任何「TM-04 专属」「位置轴专属」的字段名或分支；
   新模板注册后不改任何 JSX，这些组件必须能正确渲染它。 */
import { useState } from 'react';

/* runner two-way 分解协议（deploy/dist_runner_tm.py factorial 分支实测）：
   eta2_by_factor / per_layer 的键 = 因子名（corpus.factors，如 topic/frame）
   + 协议保留名 'interaction'（三者和≈1，维度级平均）；另有 dim_argmax_* 调试键（不渲染） */
const INTERACTION = 'interaction';
const ETA2_COLORS = ['#0284c7', '#6366f1', '#d97706'];

function pct(v) { return (v * 100).toFixed(1) + '%'; }
function f3(v) { return typeof v === 'number' ? (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3)) : String(v); }

/* ── 2-2 FactorEta2View ───────────────────────────────────────
   任意 factorial 模板的 η² 因子分解：条形图 + per_layer 深度切换。
   因子名=协议键本身（factors[0]/factors[1]/interaction），不认识 topic/frame
   还是 entity/position；factors 缺失时从数据键推导（任意因子数通用）。 */
export function FactorEta2View({ eta2, factors, perLayer }) {
  const layers = perLayer ? Object.keys(perLayer) : [];
  const [layer, setLayer] = useState(layers.includes('last') ? 'last' : layers[0]);
  const cur0 = layers.length ? (perLayer[layer] || {}) : (eta2 || {});
  const cur = Object.fromEntries(Object.entries(cur0).filter(([, v]) => typeof v === 'number'));
  if (!Object.keys(cur).length) return null;
  // 因子名序列：interaction 恒在最后；其余按 factors 或数据键序
  const facNames = (factors && factors.length ? factors
    : Object.keys(cur).filter(k => k !== INTERACTION)) || [];
  const names = [...facNames.filter(k => k in cur), ...(INTERACTION in cur ? [INTERACTION] : [])];
  const nameOf = (k) => k === INTERACTION ? '交互项' : k;
  return (
    <div className="fw-pc-eta2">
      {layers.length > 1 && (
        <div className="fw-pc-layers">
          {layers.map(l => (
            <button key={l} type="button" className={'fw-pc-lbtn' + (l === layer ? ' on' : '')}
              onClick={() => setLayer(l)}>{l}</button>
          ))}
        </div>
      )}
      {names.map((k, i) => (
        <div key={k} className="fw-pc-erow" title={`${nameOf(k)} η²=${f3(cur[k])}`}>
          <span className="fw-pc-ename">{nameOf(k)}{k === INTERACTION && <i>{k}</i>}</span>
          <span className="fw-pc-ebar">
            <i style={{ width: pct(cur[k]), background: ETA2_COLORS[i % 3] }}/>
          </span>
          <span className="fw-pc-ev fw-mono">{f3(cur[k])}</span>
        </div>
      ))}
      <div className="fw-pc-enote">η²(因子)+η²(交互)=维度平均占比（+交互=1）· 交互项=组合特异效应（推理特征领地）</div>
    </div>
  );
}

/* ── 2-3 FingerprintBadge ─────────────────────────────────────
   反词嵌入指纹门可视化。runner 实测协议：{类名: {n_within, sep_out, spread_within}}
   ——每类一枚徽标；兼容标量形态 {spread_within, sep_out}。 */
export function FingerprintBadge({ fp }) {
  if (!fp || typeof fp !== 'object') return null;
  const isPerClass = !('spread_within' in fp) && !('sep_out' in fp);
  const fnum = v => (typeof v === 'number' ? v.toFixed(3) : null);
  if (!isPerClass) {
    return (
      <div className="fw-pc-fp">
        <span className="fw-pc-fp-tag">指纹门</span>
        {['spread_within', 'sep_out'].map(k => fp[k] != null && (
          <span key={k} className="fw-pc-fp-cell">{k} <b className="fw-mono">{fnum(fp[k])}</b></span>
        ))}
      </div>
    );
  }
  const cls = Object.entries(fp).slice(0, 6);
  if (!cls.length) return null;
  return (
    <div className="fw-pc-fp">
      <span className="fw-pc-fp-tag">指纹门 · 类内等角</span>
      {cls.map(([name, d]) => (
        <span key={name} className="fw-pc-fp-cell" title={`类内等角 spread（越小=越等角=真特征）· 类外分离 sep`}>
          <i>{name}</i> spread <b className="fw-mono">{fnum(d.spread_within)}</b>
          {' · '}sep <b className="fw-mono">{fnum(d.sep_out)}</b>
        </span>
      ))}
    </div>
  );
}

/* ── 2-4 CellCosGrid ──────────────────────────────────────────
   cell 方向 cos 矩阵热图（源=/results/{sha} 提取的 means.npz cos）。
   行列=任意条件组合（协议键 keys），正=蓝 / 负=橙，强度按 |v|。 */
export function CellCosGrid({ cos }) {
  if (!cos || !Array.isArray(cos.matrix) || !cos.matrix.length) return null;
  const keys = cos.keys || cos.matrix.map((_, i) => 'c' + i);
  const n = keys.length;
  const cellColor = (v) => {
    const a = Math.min(1, Math.abs(v));
    return v >= 0 ? `rgba(2,132,199,${0.08 + 0.75 * a})` : `rgba(217,119,6,${0.08 + 0.75 * a})`;
  };
  const short = (s) => { const t = String(s).replace(/^content=|^prefix=/, ''); return t.length > 10 ? t.slice(0, 9) + '…' : t; };
  return (
    <div className="fw-pc-coswrap">
      <div className="fw-pc-cos" style={{ gridTemplateColumns: `86px repeat(${n},1fr)` }}>
        <span/>
        {keys.map(k => <span key={'h' + k} className="fw-pc-cosh" title={k}>{short(k)}</span>)}
        {keys.flatMap((rk, i) => [
          <span key={'r' + rk} className="fw-pc-cosh l" title={rk}>{short(rk)}</span>,
          ...keys.map((ck, j) => (
            <span key={rk + '|' + ck} className="fw-pc-cosc fw-mono" style={{ background: cellColor(cos.matrix[i][j]) }}
                  title={`${short(rk)} × ${short(ck)} cos=${cos.matrix[i][j].toFixed(3)}`}>
              {n <= 8 ? cos.matrix[i][j].toFixed(2) : ''}
            </span>
          )),
        ])}
      </div>
      <div className="fw-pc-cosnote">条件方向 cos 矩阵 · 蓝=同向 橙=反向 · 悬停看数值</div>
    </div>
  );
}

/* ── 协议数值 chips（oneway eta2_by_cond / heads topk_jaccard / prefix_pull_mean） */
export function StatChips({ title, data, fmt = f3 }) {
  if (!data || typeof data !== 'object') return null;
  const items = Object.entries(data);
  if (!items.length) return null;
  return (
    <div className="fw-pc-chips">
      <span className="fw-pc-chips-t">{title}</span>
      {items.map(([k, v]) => (
        <span key={k} className="fw-pc-chip" title={k}>
          <i>{String(k).replace(/^content=|^prefix=/, '')}</i><b className="fw-mono">{fmt(v)}</b>
        </span>
      ))}
    </div>
  );
}

/* ── 2-5 SummaryBrowser ───────────────────────────────────────
   结果表：行=一次上传（tm/model/kind/seed/sha/摘要），点击展开 summary
   原文与协议可视化。行渲染只用协议键；expandedSha/detail 由父级拉取传入。 */
export function SummaryBrowser({ rows, expandedSha, onExpand, detail, detailLoading, factorsByTm }) {
  if (!rows || !rows.length) {
    return <div className="fw-pc-empty">暂无结果——节点按模板执行并上传后，此处自动出现（协议驱动，无需改界面）。</div>;
  }
  const d = expandedSha === (detail && detail.sha) ? detail : null;
  return (
    <div className="fw-pc-sb">
      <div className="fw-pl-row head">
        <span>结果</span><span>模板</span><span>模型 / 节点</span><span>状态</span><span className="r">摘要</span>
      </div>
      {rows.map(r => {
        const dg = r.summary_digest || {};
        const isOpen = expandedSha === r.sha;
        return (
          <div key={r.sha} className={'fw-pc-srow-wrap' + (isOpen ? ' open' : '')}>
            <button type="button" className="fw-pc-srow" onClick={() => onExpand(isOpen ? null : r.sha)}>
              <span className="fw-mono" title={r.sha}>{r.sha.slice(0, 10)}…</span>
              <span className="fw-mono">{r.tm_id}</span>
              <span className="dim2">{r.model_id || '—'} · {r.node_id}</span>
              <span>
                <span className={'fw-pill ' + (r.kind === 'real' ? 'fw-pill-run' : 'fw-pill-done')}>{r.kind}</span>
                <span className="fw-mono" style={{ marginLeft: 4, fontSize: 9 }}>s{r.seed}</span>
              </span>
              <span className="r dim2">{digestBrief(dg)}</span>
            </button>
            {isOpen && (
              <div className="fw-pc-sdet">
                {detailLoading && <div className="fw-pc-empty">拉取结果详情…</div>}
                {d && <ResultDetail d={d} factors={factorsByTm && factorsByTm[r.tm_id]}/>}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* 摘要列的一句话画像（协议键驱动：eta2 键=因子名，有什么画什么） */
function digestBrief(dg) {
  if (dg.eta2_by_factor) {
    const es = Object.entries(dg.eta2_by_factor).filter(([, v]) => typeof v === 'number');
    return 'η² ' + es.map(([k, v]) => `${k} ${f3(v)}`).join('/');
  }
  if (dg.eta2_by_cond) {
    const es = Object.entries(dg.eta2_by_cond).sort((a, b) => b[1] - a[1]);
    return es.length ? `η² ${es[0][0]}=${f3(es[0][1])}` : '';
  }
  if (dg.topk_jaccard) return 'head 指纹';
  if (dg.status) return dg.status;
  return '—';
}

/* 展开区：summary 原文 + 全部协议可视化（按 digest 里有什么渲染什么） */
export function ResultDetail({ d, factors }) {
  const dg = d.summary || {};
  const s = { ...dg };
  delete s._manifest;
  return (
    <div className="fw-pc-rd">
      <div className="fw-pc-rd-meta fw-mono">
        sha {d.sha.slice(0, 16)}… · {(d.size / 1024).toFixed(1)} KB · 下载 {d.downloads} 次
        {Array.isArray(d.files) && d.files.length ? ' · 文件 ' + d.files.length : ''}
      </div>
      <FactorEta2View eta2={s.eta2_by_factor} factors={factors} perLayer={s.per_layer}/>
      {factors && factors.length >= 2 && s.eta2_by_factor && (
        <div className="fw-pc-fnames">
          因子=A·<b>{factors[0]}</b> / B·<b>{factors[1]}</b>（runner two-way 分解按位对应）
        </div>
      )}
      <FingerprintBadge fp={s.fingerprint}/>
      <StatChips title="单因子 η²（oneway）" data={s.eta2_by_cond}/>
      <StatChips title="head 集合指纹 Jaccard" data={s.topk_jaccard}/>
      <StatChips title="前缀吸引度（目标注意力 → 前缀）" data={s.prefix_pull_mean}/>
      <CellCosGrid cos={d.cos}/>
      <details className="fw-pc-raw">
        <summary>summary.json 原文（协议 schema v2）</summary>
        <pre>{JSON.stringify(s, null, 1)}</pre>
      </details>
    </div>
  );
}

/* ── 2-1 TemplateCard ─────────────────────────────────────────
   渲染任意 tm_id：名称/dim/因子对/结果数/聚合状态 + 可选协议详情徽标。
   detail=GET /templates/{tm_id} 的 meta（协议键），缺省时不渲染徽标区。 */
export function TemplateCard({ tpl, detail, active, onClick }) {
  const m = detail && detail.meta;
  const factors = m && m.factors && m.factors.length ? m.factors.join(' × ') : null;
  return (
    <button type="button" className={'fw-pc-tcard' + (active ? ' on' : '') + (tpl.results > 0 ? ' has' : '')} onClick={onClick}>
      <span className="fw-pc-tid fw-mono">{tpl.tm_id || tpl.id}<i>v{tpl.version || 1}</i></span>
      <span className="fw-pc-tname">{tpl.name}</span>
      <span className="fw-pc-tmeta">
        <em>{tpl.dim}</em>
        {factors && <em>{factors}</em>}
        {m && <em>{m.analysis}</em>}
        {m && m.layers && m.layers.length ? <em>{m.layers.join('/')}</em> : null}
        {m && m.items ? <em>{m.items} items</em> : null}
        {m && m.fingerprint ? <em className="fp">指纹</em> : null}
      </span>
      <span className="fw-pc-tfoot">
        <span className={'fw-tmx-agg ' + ((tpl.results || 0) > 0 ? 'collecting' : 'pending')}>
          {(tpl.results || 0) > 0 ? tpl.results + ' 结果' : '待分发'}
        </span>
        <span className="fw-mono dim2" style={{ fontSize: 9 }}>{tpl.status || 'open'}</span>
      </span>
    </button>
  );
}
