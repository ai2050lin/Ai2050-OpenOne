import { useEffect, useRef, useState } from 'react';
import { API_CONFIG } from '../../config/api';

const base = `${API_CONFIG.main.replace(/\/$/, '')}/api/research-assets`;

export default function NativeAtlasHeatmap() {
  const [catalog, setCatalog] = useState([]), [panel, setPanel] = useState(''), [start, setStart] = useState('0');
  const [page, setPage] = useState(null), [error, setError] = useState(''), [busy, setBusy] = useState(false);
  const [gain, setGain] = useState('1'), [coordinate, setCoordinate] = useState('0'), [selectedRow, setSelectedRow] = useState(0);
  const canvas = useRef(null), request = useRef(0);
  async function loadCatalog() {
    const id = ++request.current; setError(''); setBusy(true); setPage(null);
    try {
      const response = await fetch(`${base}/native-atlas-panels?compact=true`); const obj = await response.json();
      if (!response.ok) throw new Error(obj.detail || '热力图类型读取失败');
      if (id === request.current) { setCatalog(obj.panels); setPanel(obj.panels[0]?.key || ''); setStart('0'); }
    } catch (e) { if (id === request.current) setError(e.message); } finally { if (id === request.current) setBusy(false); }
  }
  async function loadRows(event) {
    event.preventDefault(); const id = ++request.current; setBusy(true); setError(''); setPage(null);
    try {
      const response = await fetch(`${base}/native-atlas-rows?${new URLSearchParams({ panel, start, count: '8' })}`); const obj = await response.json();
      if (!response.ok) throw new Error(obj.detail || '坐标读取失败');
      if (id === request.current) { setPage(obj); setSelectedRow(0); setCoordinate('0'); }
    } catch (e) { if (id === request.current) setError(e.message); }
    finally { if (id === request.current) setBusy(false); }
  }
  useEffect(() => {
    if (!page || !canvas.current) return;
    const el = canvas.current; el.width = page.coordinate_count; el.height = page.rows.length * 18;
    const ctx = el.getContext('2d'); const g = Number(gain); let maximum = 0;
    for (const row of page.rows) for (const v of row.values) maximum = Math.max(maximum, Math.abs(v));
    for (let r = 0; r < page.rows.length; r += 1) for (let k = 0; k < page.coordinate_count; k += 1) {
      const v = page.rows[r].values[k]; const strength = maximum ? Math.min(1, Math.abs(v) / maximum * g) : 0;
      const low = Math.round(28 + (1 - strength) * 62), high = Math.round(90 + strength * 165);
      ctx.fillStyle = v >= 0 ? `rgb(${high},${low},${low})` : `rgb(${low},${low},${high})`; ctx.fillRect(k, r * 18, 1, 17);
    }
  }, [page, gain]);
  const k = coordinate.trim() === '' ? NaN : Number(coordinate), selected = page?.rows[selectedRow];
  const value = selected && Number.isInteger(k) && k >= 0 && k < page.coordinate_count ? selected.values[k] : undefined;
  return <details style={{ marginTop: 14, borderTop: '1px solid #537184', paddingTop: 10 }}>
    <summary style={{ cursor: 'pointer' }}>全物理坐标热力图类型（已发布的原生坐标研究）</summary>
    <p>每列一个原始坐标，每次显示最多 8 行；全部列都绘制，可横向滚动。分页不筛坐标，不做 Top-K 或平均分箱。正值红、负值蓝；颜色放大不改变原始数值。</p>
    <p>颜色按当前页的最大绝对值缩放，因此不同页或不同类型的颜色深浅不能直接作数值比较；请使用下方的精确坐标值。颜色增益仅用于查看低幅值纹理。</p>
    <button type="button" onClick={loadCatalog} disabled={busy}>读取热力图类型</button>
    {catalog.length > 0 && <form onSubmit={loadRows} style={{ marginTop: 8 }}>
      <label>热力图类型 <select aria-label="全坐标热力图类型" value={panel} onChange={(e) => { request.current += 1; setPanel(e.target.value); setStart('0'); setPage(null); setBusy(false); }} style={{ maxWidth: '100%' }}>
        {catalog.map((p) => <option key={p.key} value={p.key}>{p.title} · {p.coordinate_count} 坐标</option>)}
      </select></label>
      <label>起始行 <input aria-label="全坐标起始行" type="number" min="0" max={(catalog.find((p) => p.key === panel)?.row_count || 1) - 1} step="1" required value={start} onChange={(e) => { request.current += 1; setStart(e.target.value); setPage(null); setBusy(false); setError(''); }} style={{ width: 72 }} /></label>
      <button type="submit" disabled={busy}>{busy ? '读取中…' : '显示全部坐标'}</button>
    </form>}
    {error && <p role="alert">{error}</p>}
    {page && <>
      <p>{page.title}：{page.total_rows} 行 × {page.coordinate_count} 物理坐标；当前从第 {page.start} 行起（零起始）。</p>
      <label>颜色增益 <select aria-label="全坐标颜色增益" value={gain} onChange={(e) => setGain(e.target.value)}>{['1', '4', '16', '64', '256'].map((v) => <option key={v} value={v}>{v}×</option>)}</select></label>
      <div style={{ maxWidth: '100%', overflowX: 'auto', marginTop: 8 }}>
        <canvas ref={canvas} aria-label="原始物理坐标热力图，所有列按原编号绘制" style={{ display: 'block', width: page.coordinate_count, height: page.rows.length * 18, maxWidth: 'none', imageRendering: 'pixelated' }} onClick={(event) => {
          const rect = event.currentTarget.getBoundingClientRect(); setCoordinate(String(Math.floor(event.clientX - rect.left))); setSelectedRow(Math.min(page.rows.length - 1, Math.max(0, Math.floor((event.clientY - rect.top) / 18))));
        }} />
      </div>
      <label>当前页行 <select aria-label="全坐标当前页行" value={selectedRow} onChange={(e) => setSelectedRow(Number(e.target.value))} style={{ maxWidth: '100%' }}>{page.rows.map((r, i) => <option key={r.row_index} value={i}>{r.row_index}: {r.label || r.source}</option>)}</select></label>
      <label>物理坐标 <input aria-label="全坐标精确编号" type="number" min="0" max={page.coordinate_count - 1} step="1" value={coordinate} onChange={(e) => setCoordinate(e.target.value)} style={{ width: 80 }} /></label>
      <p style={{ overflowWrap: 'anywhere' }}>原始值 [{selected?.row_index}, {coordinate}] = {value === undefined ? '编号超出范围' : String(value)}<br />{selected?.label || selected?.source}</p>
      <p style={{ color: '#e4c689' }}>{page.display} 图中的导数/方向计数不等于参数本身；实际权重请使用下方单参数查询。不同模型坐标不可按编号对齐。</p>
    </>}
  </details>;
}
