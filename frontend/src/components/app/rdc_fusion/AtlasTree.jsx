/* AtlasTree · 域树视图（design/atlas_domain_tree_scheme_v1.md §3.1，AtlasLens 第四视图）：
   三域树 KN 知识网络 / AR 分析推理 / GR 语法系统 → 语言族 → 特征行（E 级徽章，点击跳特征登记表详情）
   + 机制横切过滤（MEC-xx 多对多 encoded_by，选中即高亮成员特征）+ cross_link 域间交叉边 + 域覆盖度条。
   数据源：GET /api/ai-rnd/atlas/tree（server/derive_atlas_tree.py 从 registry+ledger 派生，零实验字面量）；
   派生档缺失显示空态提示，不伪造数据。样本层 SAM-xx 为 Phase C 槽位（空列表不渲染）。 */
import { useEffect, useState } from 'react';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');
const ECLASS = { E2_predictive: 'at-e2', E2_held_out: 'at-e2', E3_causal_scoped: 'at-e3', E1_repeatable: 'at-e1' };
const ELABEL = { E2_predictive: 'E2', E2_held_out: 'E2h', E3_causal_scoped: 'E3', E1_repeatable: 'E1' };

function FeatureRow({ f, dim, onPick }) {
  return (
    <button type="button" className={'at-tree-ft' + (dim ? ' dim' : '')}
      title={f.statement}
      onClick={() => onPick && onPick(f.id)}>
      <span className={'at-badge-lv mono ' + (ECLASS[f.evidence_level] || 'at-e1')}>{ELABEL[f.evidence_level] || '?'}</span>
      <b className="mono">{f.id}</b>
      <span className="st">{f.statement}</span>
    </button>
  );
}

export default function AtlasTree({ onPick }) {
  const [tree, setTree] = useState(null);
  const [err, setErr] = useState(null);
  const [mech, setMech] = useState(null);   // 选中的 MEC id（横切过滤）

  useEffect(() => {
    let dead = false;
    fetch(`${API_BASE}/api/ai-rnd/atlas/tree`)
      .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(d => { if (!dead) setTree(d); })
      .catch(e => { if (!dead) setErr(String(e.message || e)); });
    return () => { dead = true; };
  }, []);

  if (err) return (
    <div className="at-tree"><div className="at-err">域树派生档不可达（{err}）——在仓库根运行 python server/derive_atlas_tree.py 后重试。</div></div>
  );
  if (!tree) return <div className="at-tree"><div className="at-err">加载域树…</div></div>;

  const df = tree.derived_from || {};
  const hlIds = mech
    ? new Set(((tree.mechanisms || []).find(m => m.id === mech) || { features: [] }).features.map(f => f.id))
    : null;

  return (
    <div className="at-tree">
      <div className="at-tree-head">
        <span className="at-verdict">
          三域树派生档 · registry v{df.registry_version} + ledger n={df.ledger_measurements}
          {tree.tree_sha256_8 ? ` · sha8 ${tree.tree_sha256_8}` : ''}
        </span>
        {tree.coverage && tree.coverage.mechanisms && (
          <span className="dim2">机制层 {tree.coverage.mechanisms.n} 条（E3 底座 {tree.coverage.mechanisms.e3}）· 交叉边 {(tree.cross_links || []).length}</span>
        )}
      </div>

      {/* 域覆盖度条（缺口驱动：E0/E1 空格 = 候选臂池，禁止为填树降 E 级门） */}
      <div className="at-tree-cov">
        {(tree.domains || []).map(d => {
          const c = (tree.coverage || {})[d.id] || { e2plus: 0, e1: 0, n: 0 };
          return (
            <div key={d.id} className="at-tree-cov-row" title={`${d.name}：E2+ ${c.e2plus} / E1 ${c.e1}，共 ${c.n}（含交叉边计入）`}>
              <span className="nm">{d.name}</span>
              <span className="bar">
                <i className="seg e2" style={{ width: (100 * c.e2plus / Math.max(c.n, 1)) + '%' }}/>
                <i className="seg e1" style={{ width: (100 * c.e1 / Math.max(c.n, 1)) + '%' }}/>
              </span>
              <span className="ct mono">E2+ {c.e2plus} · E1 {c.e1} · n {c.n}</span>
            </div>
          );
        })}
      </div>

      {/* 机制横切过滤（多对多：一机制服务多域，选中共High亮成员特征） */}
      <div className="at-tree-mechs">
        <span className="lbl">机制过滤</span>
        {(tree.mechanisms || []).map(m => (
          <button key={m.id} type="button" className={'at-chip mono' + (mech === m.id ? ' on' : '')}
            title={m.name + '（成员 ' + m.features.length + ' · 底座 ' + (m.evidence_level_floor || '—') + '）'}
            onClick={() => setMech(mech === m.id ? null : m.id)}>{m.id}</button>
        ))}
        {mech && <button type="button" className="at-link" onClick={() => setMech(null)}>清除</button>}
      </div>

      {/* 三域树 */}
      {(tree.domains || []).map(d => (
        <section key={d.id} className="at-tree-domain">
          <h6>{d.name} <span>{d.desc}</span></h6>
          {(d.families || []).map(fam => (
            <div key={fam.id} className="at-tree-fam">
              <div className="at-tree-fam-h">{fam.name}<span className="mono dim2">{fam.id}</span></div>
              {fam.features.map(f => (
                <FeatureRow key={f.id} f={f} dim={hlIds ? !hlIds.has(f.id) : false} onPick={onPick}/>
              ))}
            </div>
          ))}
        </section>
      ))}

      {/* 交叉边（H1 改造：交叉是一等边，树上复现、数据层唯一存储） */}
      {(tree.cross_links || []).length > 0 && (
        <section className="at-tree-domain">
          <h6>域间交叉 <span>cross_link 一等边 · SxK/RxS/RxK 的家</span></h6>
          {tree.cross_links.map(x => (
            <div key={x.edge} className="at-tree-fam">
              <div className="at-tree-fam-h">{x.type}<span className="mono dim2">{x.domains.join(' ↔ ')}</span></div>
              {x.detail.map(f => (
                <FeatureRow key={f.id} f={f} dim={hlIds ? !hlIds.has(f.id) : false} onPick={onPick}/>
              ))}
            </div>
          ))}
        </section>
      )}

      {(tree.evidence_samples || []).length === 0 && (
        <div className="at-note">样本层 SAM-xx 未锚定（Phase C：从封存 npz 提取 token 级证据后此处出现）。</div>
      )}
    </div>
  );
}
