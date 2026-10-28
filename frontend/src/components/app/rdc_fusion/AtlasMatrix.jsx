/* 谱图关系 · 关系矩阵：把「哪些模式族在方向上是同一件事」画成一张族 × 族的表。
   为什么不用 3D 点云：同一层 38 个特征做三维 MDS，前三维只保留一部分距离（实测用账本 cos 校验，
   图上距离反推的 cos 系统性偏高），且 MDS 轴本身没有语义；三族共享的疑问轴会被压进几十像素里。
   矩阵用的是账本原始判据，不做任何降维：格 = 本层达判据的族对里最强的 cos，副标 = 这对关系在几层成立。

   数据全部来自 graph.nodes / graph.edges / graph.axes（atlas.v1），不新增接口。
   纪律：空格 = 未达判据（cos ≥ min_cos 且稳健 z ≥ min_z），不是「没有关系」；
   token_identity 的边（嵌入层即可解释）单独标出，不当机制读。 */

import { useMemo, useState } from 'react';
import { familyPalette } from './atlasUtil.js';
import { Empty } from './ui.jsx';

const famOf = id => String(id || '').split(':')[0];
const leafOf = id => String(id || '').replace(/@[^@]+$/, '').replace(/^[^:]+:/, '');
const n2 = v => (v == null ? '—' : Number(v).toFixed(2));

/* cos → 色阶（单色蓝，四级）；够浅才配深字 */
function tone(v) {
  if (v >= 0.85) return 4;
  if (v >= 0.68) return 3;
  if (v >= 0.52) return 2;
  return 1;
}

export default function AtlasMatrix({ graph, layer, axisId, criteria, onSelectNode }) {
  const [pick, setPick] = useState('');

  const scene = useMemo(() => {
    const fams = graph.families || [];
    const idx = Object.fromEntries(fams.map((f, i) => [f, i]));
    const keyOf = (a, b) => (idx[a] < idx[b] ? `${a}|${b}` : `${b}|${a}`);
    const all = {};   // 全层聚合：这对族在哪些层成立、最强 cos
    const here = {};  // 本层
    for (const e of graph.edges || []) {
      const a = famOf(e.source), b = famOf(e.target);
      if (a === b || idx[a] == null || idx[b] == null) continue;
      const k = keyOf(a, b);
      const c = all[k] || (all[k] = { cos: 0, layers: new Set(), edges: [] });
      c.cos = Math.max(c.cos, e.cos || 0);
      c.layers.add(e.layer);
      c.edges.push(e);
      if (e.layer === layer) {
        const h = here[k] || (here[k] = { cos: 0, edges: [] });
        h.cos = Math.max(h.cos, e.cos || 0);
        h.edges.push(e);
      }
    }
    const ax = (graph.axes || []).find(a => a.id === axisId) || null;
    const axFams = ax ? new Set(ax.families) : null;
    return { fams, idx, keyOf, all, here, ax, axFams, pal: familyPalette(fams) };
  }, [graph, layer, axisId]);

  const { fams, keyOf, all, here, ax, axFams, pal } = scene;
  const minCos = criteria?.min_cos ?? 0.3;
  const minZ = criteria?.min_z ?? 3;

  const detail = useMemo(() => {
    if (!pick) return null;
    const [a, b] = pick.split('|');
    const hits = (all[pick]?.edges || []);
    const at = hits.filter(e => e.layer === layer);
    const axes = (graph.axes || []).filter(x => x.families.includes(a) && x.families.includes(b));
    return { a, b, at, other: hits.filter(e => e.layer !== layer), axes };
  }, [pick, all, graph.axes, layer]);

  if (!fams.length) return <Empty>该模型没有可比的模式族。</Empty>;

  const hereCount = Object.keys(here).length;

  return (
    <div className="v6-mx">
      <div className="v6-mx-hd">
        <b>{layer} 层的族间方向共享</b>
        <span>{fams.length} 个族 · {hereCount} 对达判据{ax ? ` · 高亮轴：${ax.label}` : ''}</span>
      </div>

      {!hereCount && (
        <p className="v6-mx-none">这一层没有达到判据的跨族方向共享（cos ≥ {minCos} 且稳健 z ≥ {minZ}）。
          格内的「另 n 层」表示同一对族在别的层成立。</p>
      )}

      <div className="v6-mx-wrap">
        <table className="v6-mx-t">
          <caption className="sr-only">模式族两两之间在本层的最强方向余弦，以及这条关系在几层成立</caption>
          <thead>
            <tr>
              <th scope="col" className="c0" />
              {fams.map(f => <th key={f} scope="col"><i className="sw" style={{ background: pal[f] }} aria-hidden="true" />{f}</th>)}
            </tr>
          </thead>
          <tbody>
            {fams.map((a, i) => (
              <tr key={a}>
                <th scope="row"><i className="sw" style={{ background: pal[a] }} aria-hidden="true" />{a}</th>
                {fams.map((b, j) => {
                  if (j <= i) return <td key={b} className={j === i ? 'diag' : 'lower'} aria-hidden={j === i ? undefined : 'true'} />;
                  const k = keyOf(a, b);
                  const h = here[k];
                  const o = all[k];
                  const on = axFams ? axFams.has(a) && axFams.has(b) : false;
                  const ti = (h?.edges || []).some(e => e.token_identity === true);
                  if (!h) {
                    return (
                      <td key={b} className={'miss' + (on ? ' on' : '')}>
                        {o ? <span className="other">另 {o.layers.size} 层</span> : <span className="nil">—</span>}
                      </td>
                    );
                  }
                  return (
                    <td key={b} className={`hit t${tone(h.cos)}${on ? ' on' : ''}${ti ? ' ti' : ''}${pick === k ? ' cur' : ''}`}>
                      <button type="button" onClick={() => setPick(prev => (prev === k ? '' : k))}
                        aria-pressed={pick === k}
                        title={`${a} × ${b}：本层最强 cos ${n2(h.cos)}${ti ? '（含 token 身份边）' : ''}`}>
                        <b>{n2(h.cos)}</b>
                        {o.layers.size > 1 && <em>{o.layers.size} 层</em>}
                        {ti && <span className="tok" aria-label="含 token 身份边">T</span>}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {detail && (
        <div className="v6-mx-detail">
          <div className="v6-mx-detail-hd">
            <b>{detail.a} × {detail.b}</b>
            <button type="button" className="fw-tbtn ghost" onClick={() => setPick('')} aria-label="关闭详情">×</button>
          </div>
          {detail.at.length > 0 && (
            <table className="v6-mini-table">
              <thead><tr><th>本层的边</th><th>cos</th><th>留出 cos</th><th>z</th><th>预注册</th></tr></thead>
              <tbody>
                {detail.at.map(e => (
                  <tr key={e.source + e.target} className={e.token_identity ? 'tok' : ''}>
                    <th scope="row">
                      <button type="button" className="v6-mx-lk" onClick={() => onSelectNode?.(e.source)}>{leafOf(e.source)}</button>
                      <span className="muted"> ↔ </span>
                      <button type="button" className="v6-mx-lk" onClick={() => onSelectNode?.(e.target)}>{leafOf(e.target)}</button>
                    </th>
                    <td className="num">{n2(e.cos)}</td>
                    <td className="num">{n2(e.cos_lockbox)}</td>
                    <td className="num">{e.z == null ? '—' : Number(e.z).toFixed(2)}</td>
                    <td>{e.declared ? '是' : <span className="muted">新发现候选</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {detail.at.length === 0 && <p className="v6-mx-none">本层没有这条关系；它只出现在 {detail.other.map(e => e.layer).filter((v, i, s) => s.indexOf(v) === i).join('、')}。</p>}
          {detail.axes.length > 0 && (
            <div className="v6-mx-axes">
              {detail.axes.map(x => (
                <div key={x.id} className={'v6-mx-ax' + (x.supported ? '' : ' weak')}>
                  <span className="mono">{x.layer}</span>
                  <b>{x.concepts.length ? x.concepts.join('/') : '新发现候选'}</b>
                  <span>平均 cos {n2(x.mean_cos)} · 留一族迁移 {n2(x.transfer)}</span>
                  <span>{x.supported ? '达判据' : '未达判据'}{x.replicated_in?.length ? ` · 其他模型复现 ${x.replicated_in.length}` : ''}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      <p className="v6-mx-note">
        对称矩阵只画上三角。格内大字 = 本层达判据的最强 cos，小字 = 这对族一共有几层成立。
        T = 本层这条边含 token 身份即可解释（嵌入层上已对齐），不能当机制读。空格 = 未达判据。
      </p>
    </div>
  );
}
