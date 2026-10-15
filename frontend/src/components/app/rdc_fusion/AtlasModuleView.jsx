/* AtlasModuleView · 三模块视图（atlas_tree_v2 驱动；design/atlas_module_view_scheme_v1.md）
   结构：模块选择条（覆盖度徽标）→ 双栏对照（左=外部语言模板 / 右=内部编码机制特征族）
   → 关联边三态（a 实测锚定实线 / c 采集中虚线 / x 跨模块桥橙虚线；缺口红点线）
   → 底部 strip（跨模块交叉边 X1–X3 / 共享机制 MEC / 缺口清单）。
   协议键驱动：一切数字来自 GET /api/ai-rnd/atlas/tree?schema=v2（derive_atlas_tree.py 派生，零界面字面量）；
   DEMO 激活示例富化 = 前端 LANG_TEMPLATES 按 id 匹配（LIVE/DEMO 范式不变）。
   点击特征 → onPickFeature(fid)（与全景四模式共享 FeatureDetail 抽屉）；点击模板 → 本组件详情面板。 */
import { API_BASE } from '../../../config.js';
import { useEffect, useMemo, useRef, useState } from 'react';
import { LANG_TEMPLATES } from './distributedData.js';


const ECLASS = { E2_predictive: 'at-e2', E2_held_out: 'at-e2', E2_behavioral: 'at-e2', E3_causal_scoped: 'at-e3', E1_repeatable: 'at-e1' };
const ELABEL = { E2_predictive: 'E2', E2_held_out: 'E2h', E2_behavioral: 'E2b', E3_causal_scoped: 'E3', E1_repeatable: 'E1' };
const ST_NAME = { measured: '已实测', collecting: '采集中', demo: 'DEMO', pending: '待启动', done: '已完成' };
const KIND_NAME = { a: '实测锚定', c: '采集中', x: '跨模块桥' };

function EdgeSvg({ mod, templates, links, sel, domReady }) {
  const svgRef = useRef(null);
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const wrap = svg.parentElement;
    if (!wrap) return;
    const sr = wrap.getBoundingClientRect();
    if (!sr.width) return;
    svg.setAttribute('width', sr.width);
    svg.setAttribute('height', sr.height);
    let paths = '';
    templates.forEach(t => {
      const a = wrap.querySelector(`[data-nid="tpl-${CSS.escape(t.id)}"]`);
      if (!a) return;
      const ra = a.getBoundingClientRect();
      const x1 = ra.right - sr.left, y1 = ra.top + ra.height / 2 - sr.top;
      links.filter(l => l.tpl === t.id && l.ftrMod === mod.id).forEach(l => {
        const b = wrap.querySelector(`[data-nid="ftr-${CSS.escape(l.ftr)}"]`);
        if (!b) return;
        const rb = b.getBoundingClientRect();
        const x2 = rb.left - sr.left, y2 = rb.top + rb.height / 2 - sr.top;
        const mx = (x1 + x2) / 2;
        const hl = sel && (sel === t.id || sel === l.ftr);
        const dim = sel && !hl;
        const stroke = hl ? mod.color : (l.kind === 'x' ? '#d85a30' : '#94a3b8');
        paths += `<path d="M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}" fill="none" `
          + `stroke="${stroke}" stroke-width="${hl ? 2.2 : 1.2}" ${l.kind === 'a' ? '' : 'stroke-dasharray="5 4"'} `
          + `opacity="${dim ? 0.1 : hl ? 0.95 : 0.5}"/>`;
      });
      if (t.gap) {
        const x2 = x1 + sr.width * 0.3, y2 = y1;
        paths += `<path d="M ${x1} ${y1} C ${x1 + 40} ${y1}, ${x2 - 40} ${y2}, ${x2} ${y2}" fill="none" `
          + `stroke="#dc2626" stroke-width="1.4" stroke-dasharray="2 4" opacity="${sel && sel !== t.id ? 0.1 : 0.7}"/>`
          + `<text x="${x2 + 6}" y="${y2 + 3}" font-size="10" fill="#dc2626" opacity="${sel && sel !== t.id ? 0.15 : 0.9}">缺口</text>`;
      }
    });
    svg.innerHTML = paths;
  }, [mod, templates, links, sel, domReady]);
  return <svg className="amv-edges" ref={svgRef} />;
}

export default function AtlasModuleView({ sel, onPickFeature }) {
  const [tree, setTree] = useState(null);
  const [err, setErr] = useState(null);
  const [modId, setModId] = useState('KN');
  const [tplSel, setTplSel] = useState(null);
  const [domReady, setDomReady] = useState(0);

  useEffect(() => {
    let dead = false;
    fetch(`${API_BASE}/api/ai-rnd/atlas/tree?schema=v2`)
      .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(d => { if (!dead) setTree(d); })
      .catch(e => { if (!dead) setErr(String(e.message || e)); });
    return () => { dead = true; };
  }, []);

  /* links 富化：目标特征归属模块（跨模块引用只在目标模块栏内画线；源模块卡显示跨模块 chip） */
  const links = useMemo(() => {
    if (!tree) return [];
    const famOf = {};
    tree.modules.forEach(m => m.families.forEach(f => f.features.forEach(x => { famOf[x.id] = m.id; })));
    tree.cross_links.forEach(c => c.features.forEach(fid => { if (!famOf[fid]) famOf[fid] = c.domains[0]; }));
    return tree.links.map(l => ({ ...l, ftrMod: famOf[l.ftr] || '' }));
  }, [tree]);

  useEffect(() => { setTplSel(null); setDomReady(v => v + 1); }, [modId]);
  useEffect(() => {
    const t = setTimeout(() => setDomReady(v => v + 1), 30);
    return () => clearTimeout(t);
  }, [tree, tplSel]);

  if (err) return <div className="at-err">模块视图数据不可达（{err}）——运行 python server/derive_atlas_tree.py 后重试（重启 :5001 加载 v2 路由）。</div>;
  if (!tree) return <div className="at-note">加载模块视图（atlas_tree_v2）…</div>;

  const mod = tree.modules.find(m => m.id === modId) || tree.modules[0];
  const demoOf = id => LANG_TEMPLATES.find(t => t.id === id);
  const tplSelData = tplSel ? mod.templates.find(t => t.id === tplSel) : null;
  const linkedFtrs = links.filter(l => l.tpl === tplSel).map(l => l.ftr);
  const hlSel = tplSel || sel;

  return (
    <div className="amv-root">
      {/* 模块选择条 */}
      <div className="amv-mods">
        {tree.modules.map(m => (
          <button key={m.id} type="button" className={'amv-mod' + (m.id === modId ? ' on' : '')}
            style={{ '--mc': m.color }} onClick={() => setModId(m.id)}>
            <span className="amv-mod-h">
              <b>{m.name}</b><i className="en">{m.en}</i>
              <span className="cov">E2+ {m.coverage.e2plus} · E1 {m.coverage.e1} / n {m.coverage.n}</span>
            </span>
            <span className="amv-mod-d">{m.desc}</span>
            <span className="amv-mod-s">
              语言模板 <b>{m.templates.length}</b> · 在册特征 <b>{m.families.reduce((a, f) => a + f.features.length, 0)}</b>
              · 缺口 <b className="gapn">{m.templates.filter(t => t.gap).length}</b>
            </span>
            <span className="amv-mod-bar">
              <i style={{ width: (100 * m.coverage.e2plus / m.coverage.n) + '%', background: 'var(--e2,#10b981)' }} />
              <i style={{ width: (100 * m.coverage.e1 / m.coverage.n) + '%', background: 'var(--e1,#3b82f6)' }} />
            </span>
          </button>
        ))}
      </div>

      {/* 双栏对照 */}
      <div className="amv-stage">
        <div className="amv-cols">
          <div className="amv-col">
            <div className="amv-col-h">外部语言模板<small>外部语言族图谱 · 句式/状态/指标</small></div>
            {mod.templates.map(t => {
              const dim = hlSel && hlSel !== t.id && !links.some(l => l.tpl === hlSel && l.ftr === t.id);
              return (
                <div key={t.id} data-nid={`tpl-${t.id}`}
                  className={'amv-tpl' + (tplSel === t.id ? ' sel' : dim ? ' dim' : '')} style={{ '--mc': mod.color }}
                  onClick={() => setTplSel(cur => (cur === t.id ? null : t.id))}>
                  {t.gap && <span className="amv-gapdot" title={`缺口：${t.gap}`} />}
                  <div className="amv-tpl-h"><b>{t.name}</b><span className="mono tm">{t.tm || '—'}</span></div>
                  <div className="amv-tpl-pat mono">{t.pat}</div>
                  <div className="amv-tpl-ex">{t.ex}</div>
                  <div className="amv-tpl-meta">
                    <span className={'amv-st ' + t.st}>{ST_NAME[t.st]}</span>
                    {t.metric && <span className="mono metric">{t.metric}</span>}
                    {links.filter(l => l.tpl === t.id && l.ftrMod !== mod.id).length > 0 && (
                      <span className="amv-xchip" title="跨模块引用特征">跨模块 →</span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="amv-col mid"><EdgeSvg mod={mod} templates={mod.templates} links={links} sel={hlSel} domReady={domReady} /></div>
          <div className="amv-col">
            <div className="amv-col-h">内部编码机制<small>内部响应图谱 · 特征（E 级）/ 族</small></div>
            {mod.families.map(f => (
              <div key={f.id}>
                <div className="amv-famh">{f.name}</div>
                {f.features.map(ftr => {
                  const dim = hlSel && hlSel !== ftr.id && !linkedFtrs.includes(ftr.id);
                  const fromTpl = mod.templates.some(t => t.id === hlSel && links.some(l => l.tpl === t.id && l.ftr === ftr.id));
                  return (
                    <div key={ftr.id} data-nid={`ftr-${ftr.id}`}
                      className={'amv-ftr' + (sel === ftr.id ? ' sel' : dim ? ' dim' : '')} style={{ '--mc': mod.color }}
                      onClick={() => onPickFeature && onPickFeature(ftr.id)}
                      title={ftr.statement}>
                      <span className={'amv-dot ' + (ECLASS[ftr.evidence_level] || 'at-e1')} />
                      <span className="mono fid">{ftr.id}</span>
                      <span className="fst">{ftr.statement}</span>
                      <span className={'amv-eb ' + (ECLASS[ftr.evidence_level] || 'at-e1')}>{ELABEL[ftr.evidence_level] || '?'}</span>
                      {fromTpl && <span className="amv-xchip">锚定</span>}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        {/* 图例 */}
        <div className="amv-legend">
          <span className="li"><span className="sw solid" />实测锚定（anchor 可回查）</span>
          <span className="li"><span className="sw dash" />采集中 / 跨模块桥</span>
          <span className="li"><span className="sw gap" />缺口（无机制对应）</span>
          <span className="note mono">registry v{tree.derived_from.registry_version} · tree v2 sha8 {tree.tree_sha256_8} · 布局为确定性排布，非模型真实几何</span>
        </div>

        {/* 模板详情面板 */}
        {tplSelData && (
          <div className="amv-detail">
            <div className="amv-detail-h" style={{ '--mc': mod.color }}>
              <b>{tplSelData.name}</b>
              <span className="mono pat">{tplSelData.pat}</span>
              <span className={'amv-st ' + tplSelData.st}>{ST_NAME[tplSelData.st]}</span>
              <span className="amv-st mono">{tplSelData.tm || '—'}</span>
              {tplSelData.metric && <span className="amv-st mono">{tplSelData.metric}</span>}
              {tplSelData.gap && <span className="amv-st gap">缺口</span>}
              <button type="button" className="amv-close" onClick={() => setTplSel(null)}>×</button>
            </div>
            <div className="amv-detail-grid">
              <div className="amv-dsec">
                <h6>测试输入 · 最大激活示例</h6>
                {(() => {
                  const d = demoOf(tplSelData.id);
                  if (d && d.demo && d.demo.act_lines) {
                    return d.demo.act_lines.map((line, i) => (
                      <div key={i} className="amv-tokline">
                        {line.map(([tx, v], j) => (
                          <span key={j} className="amv-tok"
                            style={v > 0 ? { background: `rgba(52,211,153,${v})`, color: v > 0.6 ? '#022c22' : '#064e3b' } : undefined}>{tx}</span>
                        ))}
                        {i === 0 && <span className="amv-toknote">（DEMO 激活示例 · 着色=强度）</span>}
                      </div>
                    ));
                  }
                  return <div className="amv-dempty">示例：{tplSelData.ex}（token 级激活样例待封存 npz 锚定后接入，Phase C 槽位）</div>;
                })()}
                {(() => {
                  const d = demoOf(tplSelData.id);
                  return d && d.demo && d.demo.attn_lookback
                    ? <div className="amv-kv"><span>attention 回看</span><b>{d.demo.attn_lookback.from} → {d.demo.attn_lookback.to}：{d.demo.attn_lookback.note}</b></div>
                    : null;
                })()}
              </div>
              <div className="amv-dsec">
                <h6>生产与采集（{tplSelData.tm || '未分配 TM'}）</h6>
                <div className="amv-kv"><span>归属模块</span><b style={{ color: mod.color }}>{mod.name}</b></div>
                <div className="amv-kv"><span>家族</span><b>{tplSelData.family}</b></div>
                <div className="amv-kv"><span>节点 / cells</span><b className="mono">{tplSelData.prod.nodes} 节点 · {tplSelData.prod.cells} cells</b></div>
                <div className="amv-kv"><span>聚合状态</span><b className="mono">{tplSelData.prod.agg}</b></div>
                {(() => {
                  const d = demoOf(tplSelData.id);
                  return d && d.demo && d.demo.peaks
                    ? <div className="amv-kv"><span>峰位</span><b className="mono">MLP L{d.demo.peaks.mlp} · Res L{d.demo.peaks.res} · Attn L{d.demo.peaks.attn}</b></div>
                    : null;
                })()}
                {tplSelData.gap && <div className="amv-gapbox">⚠ {tplSelData.gap}——建议立项补测</div>}
              </div>
              <div className="amv-dsec">
                <h6>对应 LLM 编码机制（模板 → 特征关联）</h6>
                {(() => {
                  const ls = links.filter(l => l.tpl === tplSelData.id);
                  const ftrById = {};
                  tree.modules.forEach(m => m.families.forEach(f => f.features.forEach(x => { ftrById[x.id] = x; })));
                  tree.cross_links.forEach(c => c.features.forEach(fid => {
                    if (!ftrById[fid]) ftrById[fid] = { id: fid, evidence_level: 'E2_predictive', statement: `交叉边 ${c.edge}（${c.type}）` };
                  }));
                  if (!ls.length) return <div className="amv-dempty">无特征锚定——列入缺口清单</div>;
                  return ls.map(l => {
                    const f = ftrById[l.ftr];
                    const fmod = tree.modules.find(m => m.id === l.ftrMod);
                    return (
                      <div key={l.ftr} className="amv-ftrrow" onClick={() => onPickFeature && onPickFeature(l.ftr)}>
                        <span className={'amv-dot ' + (ECLASS[f.evidence_level] || 'at-e1')} />
                        <span className="mono fid">{l.ftr}</span>
                        <span className={'amv-eb ' + (ECLASS[f.evidence_level] || 'at-e1')}>{ELABEL[f.evidence_level] || '?'}</span>
                        <span className="fst">{f.statement}
                          <span className="kind">{KIND_NAME[l.kind]}{fmod && fmod.id !== mod.id ? ` · 跨模块（${fmod.name}）` : ''}</span>
                        </span>
                      </div>
                    );
                  });
                })()}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 底部 strip：交叉边 / 共享机制 / 缺口清单 */}
      <div className="amv-strips">
        <div className="amv-strip">
          <h5>跨模块交叉边（cross_link 一等边）</h5>
          {tree.cross_links.map(c => (
            <div key={c.edge} className="amv-xrow">
              <span className="mono xid">{c.edge}</span>
              {c.detail.map(f => (
                <span key={f.id} className={'amv-dot sm ' + (ECLASS[f.evidence_level] || 'at-e1')}
                  title={f.statement} />
              ))}
              <span className="mono xf">{c.features.join(' ')}</span>
              <span className="xdom">{c.type} · {c.domains.join(' ↔ ')}</span>
            </div>
          ))}
        </div>
        <div className="amv-strip">
          <h5>共享机制（MEC 横切 · 一机制服务多模块）</h5>
          {tree.mechanisms.map(m => (
            <button key={m.id} type="button" className="amv-mec" title={`成员 ${m.features.map(f => f.id).join('/')} · 底座 ${m.evidence_level_floor || '—'}`}>
              <b>{m.id}</b> {m.name}
            </button>
          ))}
        </div>
        <div className="amv-strip">
          <h5>缺口清单</h5>
          {tree.gaps.templates_without_feature.map(id => {
            const t = tree.modules.flatMap(m => m.templates).find(x => x.id === id);
            const m = tree.modules.find(x => x.id === t.mod);
            return <div key={id} className="amv-gapitem"><i className="r" />{m.name} · {t.name}：{(t.gap || '').split('——')[1] || t.gap || '无特征对应'}</div>;
          })}
          {tree.gaps.features_without_template.map(fid => {
            const f = tree.modules.flatMap(m => m.families.flatMap(x => x.features)).find(x => x.id === fid);
            return <div key={fid} className="amv-gapitem"><i className="o" /><span className="mono">{fid}</span>：特征无模板覆盖（证据来源待登记）</div>;
          })}
          <div className="amv-gapnote">{tree.gaps.note}</div>
        </div>
      </div>
    </div>
  );
}
