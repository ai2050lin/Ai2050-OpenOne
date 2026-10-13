/* AtlasPanorama · 图谱全景（AtlasLens v1.5 首屏可视化；design/atlas_domain_tree_scheme_v1.md）：
   三域泳道 SVG（KN 知识网络 / AR 分析推理 / GR 语法系统 → 语言族 → 特征节点），
   节点按 E 级着色（E2 绿 / E2h 绿 / E3 橙 / E1 蓝），机制横切过滤（MEC-xx 多对多高亮），
   域间交叉 strip（X1–X3 一等边，虚线连回所属域头），点击特征节点 → onPick(fid) 打开详情抽屉。
   布局确定性（无随机）：泳道 x 固定，族纵向堆叠，节点行内排列；非度量几何，图内已标注。
   数据源：GET /api/ai-rnd/atlas/tree（server/derive_atlas_tree.py 派生，零实验字面量）。 */
import { useEffect, useState } from 'react';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');
const ECLASS = { E2_predictive: 'at-e2', E2_held_out: 'at-e2', E3_causal_scoped: 'at-e3', E1_repeatable: 'at-e1' };
const ELABEL = { E2_predictive: 'E2', E2_held_out: 'E2h', E3_causal_scoped: 'E3', E1_repeatable: 'E1' };
const ELEGEND = { E2_predictive: 'E2 可预测', E2_held_out: 'E2h 留出', E3_causal_scoped: 'E3 协议内因果', E1_repeatable: 'E1 可重复' };

/* 确定性布局常量 */
const LANE_X0 = 16, LANE_W = 300, LANE_GAP = 24;
const HEAD_H = 58, FAM_HEAD_H = 22, ROW_H = 32, FAM_PAD = 8;
const CROSS_ROW_H = 44;

const laneX = i => LANE_X0 + i * (LANE_W + LANE_GAP);

/* 单个特征节点（圆 + id 标签；<title> 承载完整陈述） */
function PanNode({ f, x, y, dim, sel, onPick }) {
  return (
    <g className={'at-pan-node' + (dim ? ' dim' : '') + (sel ? ' sel' : '')}
       onClick={() => onPick && onPick(f.id)} style={{ cursor: 'pointer' }}>
      <title>{`[${ELABEL[f.evidence_level] || '?'}] ${f.id} · ${f.statement}${(f.model_scope || []).length ? '（' + f.model_scope.join('/') + '）' : ''}`}</title>
      <circle cx={x} cy={y} r={9} className={'at-node ' + (ECLASS[f.evidence_level] || 'at-e1')} />
      <text x={x + 14} y={y + 3.5} className="at-pan-fid">{f.id}</text>
    </g>
  );
}

export default function AtlasPanorama({ sel, onPick }) {
  const [tree, setTree] = useState(null);
  const [err, setErr] = useState(null);
  const [mech, setMech] = useState(null);

  useEffect(() => {
    let dead = false;
    fetch(`${API_BASE}/api/ai-rnd/atlas/tree`)
      .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(d => { if (!dead) setTree(d); })
      .catch(e => { if (!dead) setErr(String(e.message || e)); });
    return () => { dead = true; };
  }, []);

  if (err) return <div className="at-err">域树派生档不可达（{err}）——在仓库根运行 python server/derive_atlas_tree.py 后重试。</div>;
  if (!tree) return <div className="at-note">加载图谱全景…</div>;

  const domains = tree.domains || [];
  const mechs = tree.mechanisms || [];
  const crosses = tree.cross_links || [];
  const df = tree.derived_from || {};
  const hlIds = mech ? new Set((mechs.find(m => m.id === mech) || { features: [] }).features.map(f => f.id)) : null;

  /* 逐域布局：族纵向堆叠 → 泳道高度（节点坐标在渲染时按同公式现算，保证两处一致） */
  const laneHs = domains.map(d =>
    HEAD_H + d.families.reduce((acc, fam) => acc + FAM_HEAD_H + fam.features.length * ROW_H + FAM_PAD, 0) + 6
  );
  const laneH = Math.max(...laneHs, HEAD_H + 60);
  const crossY0 = laneH + 30;
  const crossH = (crosses.length ? 24 : 0) + crosses.length * CROSS_ROW_H + 8;
  const H = crossY0 + crossH;

  return (
    <div className="at-pan">
      {/* 机制横切过滤（多对多：一机制服务多域） */}
      <div className="at-tree-mechs">
        <span className="lbl">机制过滤</span>
        {mechs.map(m => (
          <button key={m.id} type="button" className={'at-chip mono' + (mech === m.id ? ' on' : '')}
            title={m.name + '（成员 ' + m.features.length + ' · 底座 ' + (m.evidence_level_floor || '—') + '）'}
            onClick={() => setMech(mech === m.id ? null : m.id)}>{m.id}</button>
        ))}
        {mech && <button type="button" className="at-link" onClick={() => setMech(null)}>清除</button>}
      </div>

      <div className="at-pan-wrap">
        <svg viewBox={`0 0 ${LANE_X0 * 2 + domains.length * LANE_W + (domains.length - 1) * LANE_GAP} ${H}`} className="at-map">
          {/* 域泳道背景 + 头部（名称/描述/覆盖度条/计数） */}
          {domains.map((d, i) => {
            const c = (tree.coverage || {})[d.id] || { e2plus: 0, e1: 0, n: 0 };
            const x = laneX(i);
            return (
              <g key={d.id}>
                <rect x={x} y={0} width={LANE_W} height={laneH} rx={10} className="at-pan-lane"/>
                <text x={x + 12} y={22} className="at-pan-dname">{d.name}</text>
                <text x={x + 12} y={37} className="at-pan-ddesc">{d.desc}</text>
                <g title={`${d.name}：E2+ ${c.e2plus} / E1 ${c.e1}，共 ${c.n}（含交叉边计入）`}>
                  <rect x={x + 12} y={44} width={180} height={7} rx={3.5} className="at-pan-bar-bg"/>
                  <rect x={x + 12} y={44} width={180 * (c.e2plus / Math.max(c.n, 1))} height={7} rx={3.5} className="at-pan-bar-e2"/>
                  <rect x={x + 12 + 180 * (c.e2plus / Math.max(c.n, 1))} y={44} width={180 * (c.e1 / Math.max(c.n, 1))} height={7} rx={3.5} className="at-pan-bar-e1"/>
                  <text x={x + LANE_W - 10} y={51} textAnchor="end" className="at-pan-bct">E2+ {c.e2plus} · E1 {c.e1} / n {c.n}</text>
                </g>
              </g>
            );
          })}

          {/* 交叉连线（先画，位于族内容之下）：交叉节点顶边 → 所属域头部底边中点 */}
          {crosses.map((cx, ci) => {
            const ry = crossY0 + 24 + ci * CROSS_ROW_H + 10;   // 与下方 strip 内节点 y 一致
            const li0 = domains.findIndex(d => d.id === cx.domains[0]);
            const nx = (li0 >= 0 ? laneX(li0) : LANE_X0) + 190;
            return cx.domains.map(did => {
              const li = domains.findIndex(d => d.id === did);
              if (li < 0) return null;
              const tx = laneX(li) + LANE_W / 2, ty = HEAD_H;
              return <path key={cx.edge + did}
                d={`M ${nx} ${ry - 9} C ${nx} ${ty + 70}, ${tx} ${ty - 50}, ${tx} ${ty}`}
                className="at-pan-xlink"/>;
            });
          })}

          {/* 族 + 特征节点 */}
          {domains.map((d, i) => {
            const x = laneX(i);
            let y = HEAD_H;
            return (
              <g key={d.id}>
                {d.families.map(fam => {
                  const fy = y;
                  y += FAM_HEAD_H + fam.features.length * ROW_H + FAM_PAD;
                  return (
                    <g key={fam.id}>
                      <text x={x + 10} y={fy + 15} className="at-pan-famh">{fam.name}<tspan className="at-pan-famid"> {fam.id}</tspan></text>
                      <line x1={x + 10} y1={fy + FAM_HEAD_H - 4} x2={x + LANE_W - 10} y2={fy + FAM_HEAD_H - 4} className="at-pan-famln"/>
                      {fam.features.map((f, j) => {
                        const ny = fy + FAM_HEAD_H + j * ROW_H + 16;
                        return <PanNode key={f.id} f={f} x={x + 24} y={ny}
                          dim={hlIds ? !hlIds.has(f.id) : false} sel={sel === f.id} onPick={onPick}/>;
                      })}
                    </g>
                  );
                })}
              </g>
            );
          })}

          {/* 域间交叉 strip（X1–X3 一等边；特征同时计入两侧域覆盖度） */}
          {crosses.length > 0 && (
            <g>
              <text x={LANE_X0} y={crossY0 + 12} className="at-pan-xh">域间交叉（cross_link 一等边）</text>
              {crosses.map((cx, ci) => {
                const ry = crossY0 + 24 + ci * CROSS_ROW_H;
                return (
                  <g key={cx.edge}>
                    <text x={LANE_X0 + 4} y={ry + 14} className="at-pan-xt">{cx.edge} · {cx.type}</text>
                    <text x={LANE_X0 + 4} y={ry + 27} className="at-pan-xd">{cx.domains.join(' ↔ ')}</text>
                    {cx.detail.map(f => {
                      const li = domains.findIndex(d => d.id === cx.domains[0]);
                      const lx = li >= 0 ? laneX(li) : LANE_X0;
                      return <g key={f.id}>
                        <PanNode f={f} x={lx + 190} y={ry + 10} dim={hlIds ? !hlIds.has(f.id) : false} sel={sel === f.id} onPick={onPick}/>
                      </g>;
                    })}
                  </g>
                );
              })}
            </g>
          )}

          <text x={LANE_X0} y={H - 4} className="at-map-note">示意图布局：域→族→特征确定性排布，非模型真实几何 · 点击节点打开特征详情 · 悬停看陈述全文</text>
        </svg>
      </div>

      {/* 图例 */}
      <div className="at-pan-legend">
        {['E2_predictive', 'E2_held_out', 'E3_causal_scoped', 'E1_repeatable'].map(k => (
          <span key={k} className="at-legend-item"><i className={'at-legend-dot ' + (ECLASS[k])}/>{ELEGEND[k]}</span>
        ))}
        <span className="at-legend-item"><i className="at-legend-line"/>交叉边（域间一等边）</span>
        <span className="dim2 mono">registry v{df.registry_version} · sha8 {tree.tree_sha256_8}</span>
      </div>
    </div>
  );
}
