/* 研发 · 模板流水线 ① 建模板：生成式模板（template_grammar.v1）= 句式框架 × 填充词表 × 因子水平。
   左：内置模板族库（shared/dist_protocol/grammars）；中：结构视图 + JSON 编辑；右：生成预览
   （逐格 × 划分计数、功效近似、样例、告警）。预览只计算，不写盘；冻结在「② 试跑与冻结」。 */
import { useMemo, useState } from 'react';
import { postJson, useAdminToken, useApi, fmtNum } from './api.js';
import { Card, Empty, StatusPill } from './ui.jsx';

const SPLITS = ['discovery', 'confirm', 'lockbox', 'lockbox_frame'];

function parseSpec(text) {
  try { return { spec: JSON.parse(text), error: '' }; } catch (reason) { return { spec: null, error: String(reason.message || reason) }; }
}

/* 跨族概念预注册（atlas.v1）：给因子水平标上概念名，声明「它应与其他模式族里同概念的水平共享编码」。
   谱图挖掘检验声明；没声明却对齐的特征记为新发现候选。 */
function ConceptEditor({ spec, onChange, known }) {
  const levels = {};
  for (const it of [...(spec.frames || []), ...(spec.fillers || [])]) {
    for (const [f, v] of Object.entries(it.levels || {})) (levels[f] = levels[f] || new Set()).add(String(v));
  }
  const concepts = spec.concepts || {};
  const rows = (spec.factors || []).flatMap(f => [...(levels[f] || [])].sort().map(lv => `${f}=${lv}`));
  const set = (key, value) => {
    const next = { ...concepts };
    if (value.trim()) next[key] = value.trim(); else delete next[key];
    const out = { ...spec };
    if (Object.keys(next).length) out.concepts = next; else delete out.concepts;
    onChange(out);
  };
  const names = Object.keys(known).sort();
  // 只出现在留出句式里的水平：没有 discovery 数据，谱图里作为「未见句式预测」的检验对象
  const heldIds = new Set(spec.splits?.heldout_frames || []);
  const keysOf = fr => Object.entries(fr.levels || {}).map(([f, v]) => `${f}=${v}`);
  const seenOpen = new Set((spec.frames || []).filter(fr => !heldIds.has(fr.id)).flatMap(keysOf));
  const heldOnly = new Set((spec.frames || []).filter(fr => heldIds.has(fr.id)).flatMap(keysOf).filter(k => !seenOpen.has(k)));
  return (
    <div className="v6-concepts">
      <h5 className="v6-h5">跨族概念预注册（concepts）
        <small>同一概念出现在 ≥ 2 个模式族时，语言谱图检验这些水平是否共享编码；留出句式的水平作为未见句式预测</small>
      </h5>
      <datalist id="v6-concept-names">{names.map(n => <option key={n} value={n} />)}</datalist>
      <table className="v6-mini-table">
        <thead><tr><th>因子 = 水平</th><th>概念</th><th>该概念已出现在</th></tr></thead>
        <tbody>
          {rows.map(key => {
            const c = concepts[key] || '';
            const fams = (known[c] || []).filter(f => f !== spec.family_id);
            const held = heldOnly.has(key);
            return (
              <tr key={key}>
                <th scope="row" className="mono">{key}{held && <em className="v6-tag">留出</em>}</th>
                <td><input className="v6-concept-input" list="v6-concept-names" value={c} placeholder="如 疑问" aria-label={`${key} 的概念`}
                  onChange={e => set(key, e.target.value)} /></td>
                <td className="dim2">{c ? (fams.length ? fams.join('、') : '暂无其他模式族（需要 ≥ 2 个才可检验）') : ''}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function SpecStructure({ spec }) {
  const [fillerQuery, setFillerQuery] = useState('');
  const fillers = (spec.fillers || []).filter(f => !fillerQuery || JSON.stringify(f).includes(fillerQuery));
  return (
    <div className="v6-spec">
      <div className="v6-spec-meta">
        <span>模板族 <b className="mono">{spec.family_id}</b></span>
        <span>模块 <b>{spec.module || '—'}</b></span>
        <span>因子 <b>{(spec.factors || []).join(' × ')}</b></span>
        {/* 文法的 layers 可以是层位列表，也可以是 "all"（全部层，含嵌入层 hs[0]） */}
        <span>层位 <b>{Array.isArray(spec.layers) ? spec.layers.join(' / ') : spec.layers === 'all' ? '全部层（含 hs[0]）' : String(spec.layers ?? '—')}</b></span>
        <span>分片 <b>{spec.shards || 1}</b></span>
        <span>留出句式 <b>{(spec.splits?.heldout_frames || []).join('、') || '无'}</b></span>
      </div>
      {spec.prereg && (
        <div className="v6-spec-meta prereg">
          <span>预注册：</span>
          {Object.entries(spec.prereg).map(([k, v]) => <span key={k}>{k} <b>{String(v)}</b></span>)}
        </div>
      )}
      <h5 className="v6-h5">句式框架（{(spec.frames || []).length}）</h5>
      <table className="v6-mini-table">
        <thead><tr><th>id</th><th>因子水平</th><th>句式</th><th>目标槽</th><th>行为 probe</th></tr></thead>
        <tbody>
          {(spec.frames || []).map(frame => (
            <tr key={frame.id}>
              <th scope="row" className="mono">{frame.id}{(spec.splits?.heldout_frames || []).includes(frame.id) && <em className="v6-tag">留出</em>}</th>
              <td>{Object.entries(frame.levels || {}).map(([k, v]) => `${k}=${v}`).join(' · ')}</td>
              <td>{frame.text}</td>
              <td className="mono">{frame.target}</td>
              <td className="dim2">{frame.probe ? `${frame.probe.prompt} → ${frame.probe.answer}` : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h5 className="v6-h5">填充词表（{(spec.fillers || []).length}）
        <input className="v6-search" placeholder="筛选…" value={fillerQuery} onChange={event => setFillerQuery(event.target.value)} />
      </h5>
      <div className="v6-fillers">
        {fillers.map(filler => (
          <span key={filler.id} title={filler.id}>
            {Object.values(filler.slots || {}).join(' / ')}
            {Object.keys(filler.levels || {}).length > 0 && <small>{Object.values(filler.levels).join('·')}</small>}
          </span>
        ))}
      </div>
    </div>
  );
}

function Preview({ result }) {
  if (!result) return <Empty>点击「生成预览」：按切分规则生成全部样本，统计逐格样本数与功效。</Empty>;
  if (!result.ok) {
    return (
      <div className="v6-errors">
        <b>规格错误（{result.errors.length}）</b>
        <ul>{result.errors.map(error => <li key={error}>{error}</li>)}</ul>
      </div>
    );
  }
  const minCell = result.cells.reduce((m, c) => (c.discovery + c.confirm + c.lockbox > 0 ? Math.min(m, c.discovery) : m), Infinity);
  return (
    <div className="v6-preview">
      <div className="v6-kpis">
        <div><small>样本</small><b>{result.n_items}</b><span>{result.n_frames} 句式 × {result.n_fillers} 填充</span></div>
        {SPLITS.map(sp => <div key={sp}><small>{sp}</small><b>{result.by_split[sp]}</b></div>)}
        <div><small>probe 条目</small><b>{result.probe_items}</b></div>
      </div>
      <h5 className="v6-h5">功效近似（discovery 划分，Bonferroni 到 2560 维）</h5>
      <table className="v6-mini-table">
        <thead><tr><th>因子</th><th>水平数</th><th>discovery N</th><th>最小可检测 η²</th></tr></thead>
        <tbody>{result.power.map(p => <tr key={p.factor}><th scope="row">{p.factor}</th><td>{p.levels}</td><td>{p.n_discovery}</td><td className="num">{fmtNum(p.mde_eta2)}</td></tr>)}</tbody>
      </table>
      <p className="v6-note">{result.note}</p>
      <h5 className="v6-h5">逐格样本数（discovery 最少 {Number.isFinite(minCell) ? minCell : '—'}）</h5>
      <div className="v6-cells-wrap">
        <table className="v6-mini-table">
          <thead><tr><th>条件格</th>{SPLITS.map(sp => <th key={sp}>{sp}</th>)}</tr></thead>
          <tbody>
            {result.cells.map(c => (
              <tr key={c.cell}><th scope="row" className="mono">{c.cell}</th>{SPLITS.map(sp => <td key={sp} className="num">{c[sp] || ''}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
      {result.warnings.length > 0 && (
        <div className="v6-warns"><b>告警</b><ul>{result.warnings.map(w => <li key={w}>{w}</li>)}</ul></div>
      )}
      <h5 className="v6-h5">样例</h5>
      <ul className="v6-samples">
        {result.samples.map(s => (
          <li key={s.id}>
            <StatusPill status="na" label={s.split} />
            <span>{s.text.slice(0, s.target_char)}<mark>{s.target}</mark>{s.text.slice(s.target_char + s.target.length)}</span>
            <small>{Object.entries(s.cond).map(([k, v]) => `${k}=${v}`).join(' · ')}</small>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function TemplateStudio({ draft, onDraft, onNext }) {
  const library = useApi('/api/grammar/library');
  const catalog = useApi('/api/catalog');
  const [token] = useAdminToken();
  const grammars = library.data?.grammars || [];
  const [text, setText] = useState(() => (draft ? JSON.stringify(draft, null, 1) : ''));
  const [showJson, setShowJson] = useState(false);
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const parsed = useMemo(() => (text ? parseSpec(text) : { spec: null, error: '' }), [text]);
  const current = parsed.spec;
  const knownConcepts = useMemo(() => {
    const out = {};
    const add = (fam, cs) => Object.values(cs || {}).forEach(c => { out[c] = [...new Set([...(out[c] || []), fam])]; });
    (catalog.data?.items || []).forEach(it => add(it.id, it.concepts));
    (library.data?.grammars || []).forEach(g => add(g.family_id, g.spec?.concepts));
    return out;
  }, [catalog.data, library.data]);
  const editSpec = spec => { setText(JSON.stringify(spec, null, 1)); setPreview(null); onDraft?.(spec); };

  const pick = grammar => {
    setText(JSON.stringify(grammar.spec, null, 1));
    setPreview(null);
    onDraft?.(grammar.spec);
  };
  const runPreview = async () => {
    if (!current) return;
    setBusy(true); setError('');
    try {
      // preview 有真实计算成本，服务端与管理写操作同级鉴权（管理员会话自动作为令牌回退）
      setPreview(await postJson('/api/grammar/preview', { spec: current }, { adminToken: token }));
      onDraft?.(current);
    } catch (reason) { setError(String(reason.message || reason)); }
    setBusy(false);
  };
  const download = () => {
    const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url; a.download = `${current?.family_id || 'grammar'}.json`; a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="v6-studio">
      <aside className="v6-studio-lib">
        <h5 className="v6-h5">模板族库</h5>
        {!library.data && <Empty>{library.error ? `未接入（${library.error}）` : '读取…'}</Empty>}
        {grammars.map(g => (
          <button key={g.file} type="button" className={'v6-lib-item' + (current?.family_id === g.family_id ? ' on' : '')} onClick={() => pick(g)}>
            <span className="mono">{g.family_id}</span>
            <b>{g.name}</b>
            <small>{g.factors?.join(' × ')} · {g.n_frames} 句式 × {g.n_fillers} 填充</small>
            {g.registered?.length > 0 && <small className="reg">已冻结：{g.registered.map(r => `${r.tm_id} v${r.version}`).join('、')}</small>}
          </button>
        ))}
        <p className="v6-note">新增模板族：在 JSON 中修改 family_id 后预览、下载，并提交到仓库 shared/dist_protocol/grammars/。</p>
      </aside>
      <div className="v6-studio-main">
        {!current && !parsed.error && <Empty>从左侧选择一个模板族开始。</Empty>}
        {parsed.error && <div className="fw-ai-err">JSON 解析失败：{parsed.error}</div>}
        {current && <Card title={current.name || current.family_id} sub={current.description} extra={
          <>
            <button type="button" className="fw-tbtn" onClick={() => setShowJson(v => !v)}>{showJson ? '收起 JSON' : '编辑 JSON'}</button>
            <button type="button" className="fw-tbtn" onClick={download}>下载</button>
            <button type="button" className="fw-tbtn primary" disabled={busy} onClick={runPreview}>{busy ? '生成中…' : '生成预览'}</button>
          </>
        }>
          {showJson && <textarea className="v6-json" spellCheck={false} value={text} onChange={event => { setText(event.target.value); setPreview(null); }} />}
          <SpecStructure spec={current} />
          <ConceptEditor spec={current} onChange={editSpec} known={knownConcepts} />
        </Card>}
      </div>
      <aside className="v6-studio-preview">
        <Card title="生成预览" extra={preview?.ok && <button type="button" className="fw-tbtn primary" onClick={() => onNext?.(current)}>试跑与冻结 →</button>}>
          {error && <div className="fw-ai-err">{error}</div>}
          <Preview result={preview} />
        </Card>
      </aside>
    </div>
  );
}
