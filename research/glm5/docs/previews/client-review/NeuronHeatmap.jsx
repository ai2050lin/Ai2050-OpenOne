import React, { useEffect, useId, useMemo, useRef, useState } from 'react';
import './neuron-heatmap.css';

const COLUMNS = 128;
const POSITIVE = [31, 95, 191];
const NEGATIVE = [212, 98, 43];
const MISSING = [198, 205, 214];

function measureField(values, width, sharedMax) {
  const length = values?.length ?? 0;
  if (!length) return { empty: '当前读出位置没有归档逐单元激活。' };
  if (width != null && width !== length) {
    return { empty: `归档向量有 ${length} 项，当前模型的 FFN 宽度为 ${width}。请检查快照或采集边界。` };
  }
  let limit = Number.isFinite(sharedMax) && sharedMax > 0 ? sharedMax : 0;
  let missing = 0;
  for (const value of values) {
    if (Number.isFinite(value)) limit = Math.max(limit, Math.abs(value));
    else missing++;
  }
  if (missing === length) return { empty: '此向量没有可读取的有限数值。' };
  const columns = Math.min(COLUMNS, length);
  return { columns, rows: Math.ceil(length / columns), length, limit, missing };
}

function paint(canvas, values, field) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const image = ctx.createImageData(field.columns, field.rows);
  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    const finite = Number.isFinite(value);
    const endpoint = value < 0 ? NEGATIVE : POSITIVE;
    const mix = field.limit > 0 ? (Math.abs(value) / field.limit) ** 0.6 : 0;
    const offset = i * 4;
    for (let c = 0; c < 3; c++) image.data[offset + c] = finite ? 255 + (endpoint[c] - 255) * mix : MISSING[c];
    image.data[offset + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}

function formatLimit(value) {
  return String(Number(value.toPrecision(4)));
}

export default function NeuronHeatmap({ values, index, onPick, block, maxAbs, width, missingReason, children }) {
  const uid = useId();
  const ref = useRef(null);
  const [draft, setDraft] = useState('');
  const field = useMemo(() => measureField(values, width, maxAbs), [values, width, maxAbs]);
  const valid = Number.isInteger(index) && index >= 0 && index < field.length;
  const selected = valid ? values[index] : null;
  const x = valid ? index % field.columns : 0;
  const y = valid ? Math.floor(index / field.columns) : 0;

  useEffect(() => { setDraft(valid ? String(index) : ''); }, [index, valid]);
  useEffect(() => { if (ref.current && !field.empty) paint(ref.current, values, field); }, [values, field]);

  function pickAtPoint(event) {
    const bounds = event.currentTarget.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const column = Math.floor((event.clientX - bounds.left) * field.columns / bounds.width);
    const row = Math.floor((event.clientY - bounds.top) * field.rows / bounds.height);
    const i = row * field.columns + column;
    if (column >= 0 && column < field.columns && row >= 0 && i < field.length) onPick?.(i);
    event.currentTarget.focus({ preventScroll: true });
  }

  function navigate(event) {
    const start = valid ? index : 0;
    const destinations = {
      ArrowLeft: start - 1, ArrowRight: start + 1, ArrowUp: start - field.columns,
      ArrowDown: start + field.columns, Home: 0, End: field.length - 1,
    };
    if (!(event.key in destinations)) return;
    event.preventDefault();
    onPick?.(Math.max(0, Math.min(field.length - 1, destinations[event.key])));
  }

  function jump(event) {
    event.preventDefault();
    const next = Number(draft);
    if (draft.trim() && Number.isInteger(next) && next >= 0 && next < field.length) onPick?.(next);
  }

  return <section className="neuron-heatmap" aria-labelledby={`${uid}-heading`}>
    <header className="nhm-heading">
      <h3 id={`${uid}-heading`}>神经元热力图</h3>
      {Number.isInteger(block) && <span>Block {block} · FFN</span>}
    </header>
    {field.empty ? <div className="nhm-empty">
      <strong>暂无可显示的热图</strong>
      <p>{values?.length ? field.empty : missingReason || field.empty}</p>{children}
    </div> : <>
      <div className="nhm-meta"><span>{field.length.toLocaleString('zh-CN')} 个单元 · 每格一个原生编号</span><span>每行 {field.columns} 个</span></div>
      <div className="nhm-plot">
        <canvas ref={ref} className="nhm-raster" width={field.columns} height={field.rows}
          tabIndex={0} aria-label="FFN 神经元热力图，按原生编号逐行排列" aria-describedby={`${uid}-keys`}
          onClick={pickAtPoint} onKeyDown={navigate}>请使用下方编号框定位神经元并读取原值。</canvas>
        {valid && <span className="nhm-cursor" aria-hidden="true" style={{ left: `${100 * (x + 0.5) / field.columns}%`, top: `${100 * (y + 0.5) / field.rows}%` }} />}
      </div>
      <div className="nhm-legend" aria-label="激活色标">
        {field.limit > 0 ? <><span>负 −{formatLimit(field.limit)}</span><i aria-hidden="true" /><span>0</span><i aria-hidden="true" /><span>+{formatLimit(field.limit)} 正</span></> : <span>色标 0 · 所有已采集单元均为零</span>}
        {field.missing > 0 && <small>灰色为缺值，共 {field.missing} 项</small>}
      </div>
      <div className="nhm-selection">
        <form onSubmit={jump}>
          <label htmlFor={`${uid}-index`}>神经元编号</label>
          <input id={`${uid}-index`} className="nhm-index" type="number" min="0" max={field.length - 1} step="1" required value={draft} onChange={event => setDraft(event.target.value)} />
          <button type="submit">定位</button>
        </form>
        <div className="nhm-readout" aria-live="polite"><span>{valid ? `U${index} 原值` : '原值'}</span>
          <output className="nhm-value" data-index={valid ? index : ''} data-value={Number.isFinite(selected) ? selected : ''}>{valid ? Number.isFinite(selected) ? String(selected) : '未采集' : '请选择单元'}</output>
        </div>
      </div>
      <p className="nhm-help" id={`${uid}-keys`}>点击色块或用方向键移动，Home / End 跳到首尾。色彩按相对幅度的 0.6 次幂映射。</p>
    </>}
  </section>;
}
