import { useEffect, useMemo, useRef, useState } from 'react';
import { Activity, Binary, Boxes, ChevronDown, ChevronLeft, ChevronRight, Microscope, Network, RefreshCw } from 'lucide-react';

import { researchAssetUrl } from '../config/researchAssets';
import { useLanguageEncodingCatalog } from './useLanguageEncodingCatalog';
import './LanguageEncodingExplorer.css';

const PALETTE = {
  negative: [45, 108, 190],
  zero: [22, 29, 39],
  positive: [222, 104, 54],
};

const ROW_WINDOW = 512;

function color(value, scale) {
  const normalized = Math.max(-1, Math.min(1, Number(value || 0) / Math.max(scale, 1e-8)));
  const target = normalized < 0 ? PALETTE.negative : PALETTE.positive;
  const amount = Math.abs(normalized);
  return PALETTE.zero.map((base, index) => Math.round(base + (target[index] - base) * amount));
}

function halfToNumber(bits) {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 0x1f) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

function parseFloat16Npy(buffer, expectedShape) {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 6));
  if (magic !== '\u0093NUMPY') throw new Error('Unsupported activation matrix format');
  const major = view.getUint8(6);
  const headerBytes = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerOffset = major === 1 ? 10 : 12;
  const dataOffset = headerOffset + headerBytes;
  const header = new TextDecoder('latin1').decode(new Uint8Array(buffer, headerOffset, headerBytes));
  if (!header.includes("'descr': '<f2'")) throw new Error('Activation matrix must be little-endian float16');
  const shapeMatch = header.match(/'shape':\s*\(([^)]*)\)/);
  const shape = shapeMatch
    ? shapeMatch[1].split(',').map((item) => Number(item.trim())).filter(Number.isFinite)
    : expectedShape;
  if (shape.length !== 2 || shape.some((value, index) => value !== expectedShape[index])) {
    throw new Error(`Activation matrix shape mismatch: ${shape.join('x')}`);
  }
  return { values: new Uint16Array(buffer, dataOffset, shape[0] * shape[1]), shape };
}

function rowValue(row, coordinate, matrix, coordinateCount) {
  if (row.values) return Number(row.values[coordinate]) || 0;
  if (!matrix || row._matrixIndex == null) return 0;
  return halfToNumber(matrix.values[row._matrixIndex * coordinateCount + coordinate]);
}

function FieldCanvas({ rows, coordinateCount, matrix, onCoordinate }) {
  const canvasRef = useRef(null);
  const scale = useMemo(() => {
    let maximum = 0;
    rows.forEach((row) => {
      for (let coordinate = 0; coordinate < coordinateCount; coordinate += 1) {
        maximum = Math.max(maximum, Math.abs(rowValue(row, coordinate, matrix, coordinateCount)));
      }
    });
    return maximum || 1;
  }, [coordinateCount, matrix, rows]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !rows.length) return;
    canvas.width = coordinateCount;
    canvas.height = rows.length;
    const context = canvas.getContext('2d', { alpha: false });
    const image = context.createImageData(canvas.width, canvas.height);
    rows.forEach((row, y) => {
      for (let x = 0; x < coordinateCount; x += 1) {
        const value = rowValue(row, x, matrix, coordinateCount);
        const [red, green, blue] = color(value, scale);
        const offset = (y * canvas.width + x) * 4;
        image.data[offset] = red;
        image.data[offset + 1] = green;
        image.data[offset + 2] = blue;
        image.data[offset + 3] = 255;
      }
    });
    context.putImageData(image, 0, 0);
  }, [coordinateCount, matrix, rows, scale]);

  return (
    <canvas
      ref={canvasRef}
      className="encoding-field-canvas"
      aria-label={`${rows.length} by ${coordinateCount} activation field`}
      onClick={(event) => {
        const bounds = event.currentTarget.getBoundingClientRect();
        onCoordinate(Math.max(0, Math.min(coordinateCount - 1, Math.floor(((event.clientX - bounds.left) / bounds.width) * coordinateCount))));
      }}
    />
  );
}

function MicroscopeCanvas({ rows, coordinate, coordinateCount, matrix }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !rows.length) return;
    const width = 900;
    const height = 210;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    const context = canvas.getContext('2d');
    context.scale(ratio, ratio);
    context.clearRect(0, 0, width, height);
    const values = rows.map((row) => rowValue(row, coordinate, matrix, coordinateCount));
    const maximum = Math.max(...values.map(Math.abs), 1e-7);
    context.strokeStyle = '#334155';
    context.beginPath();
    context.moveTo(0, height / 2);
    context.lineTo(width, height / 2);
    context.stroke();
    context.strokeStyle = '#f97316';
    context.lineWidth = 2;
    context.beginPath();
    values.forEach((value, index) => {
      const x = values.length === 1 ? width / 2 : (index / (values.length - 1)) * width;
      const y = height / 2 - (value / maximum) * (height * 0.42);
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
    context.stroke();
  }, [coordinate, coordinateCount, matrix, rows]);
  return <canvas ref={canvasRef} className="encoding-microscope-canvas" aria-label={`Coordinate ${coordinate} trace`} />;
}

function FamilyGraph({ families, relations }) {
  return (
    <div className="encoding-family-graph">
      <div className="encoding-family-graph__families">
        {families.map((family) => (
          <div key={family.id}>
            <strong>{family.label}</strong>
            <span>{family.domain}</span>
          </div>
        ))}
      </div>
      <div className="encoding-family-graph__relations">
        {relations.map((relation) => (
          <article key={relation.id} className={`is-${relation.status}`}>
            <span>{relation.id}</span>
            <strong>{relation.source} → {relation.target}</strong>
            <small>{relation.evidence}</small>
          </article>
        ))}
      </div>
    </div>
  );
}

export function LanguageEncodingExplorer() {
  const { catalog, loading, error } = useLanguageEncodingCatalog();
  const [datasetId, setDatasetId] = useState('');
  const [payloadState, setPayloadState] = useState({ datasetId: '', data: null, error: '' });
  const [source, setSource] = useState('all');
  const [role, setRole] = useState('all');
  const [coordinate, setCoordinate] = useState(0);
  const [view, setView] = useState('field');
  const [rowStart, setRowStart] = useState(0);
  const [matrixState, setMatrixState] = useState({ datasetId: '', matrix: null, error: '' });

  const datasets = useMemo(() => catalog?.datasets || [], [catalog]);
  const activeDataset = datasets.find((item) => item.id === datasetId) || datasets[0];
  useEffect(() => {
    if (!activeDataset) return undefined;
    const controller = new AbortController();
    fetch(researchAssetUrl(activeDataset.source_path), { cache: 'no-store', signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`Field dataset ${response.status}`);
        return response.json();
      })
      .then((data) => setPayloadState({ datasetId: activeDataset.id, data, error: '' }))
      .catch((fetchError) => {
        if (fetchError.name !== 'AbortError') {
          setPayloadState({ datasetId: activeDataset.id, data: null, error: fetchError.message });
        }
      });
    return () => controller.abort();
  }, [activeDataset]);

  const payload = payloadState.datasetId === activeDataset?.id ? payloadState.data : null;
  const payloadError = payloadState.datasetId === activeDataset?.id ? payloadState.error : '';

  useEffect(() => {
    if (!payload || !activeDataset) return undefined;
    const binaryUrl = payload.binary_url || activeDataset.binary_path;
    if (!binaryUrl) {
      setMatrixState({ datasetId: activeDataset.id, matrix: null, error: '' });
      return undefined;
    }
    const controller = new AbortController();
    fetch(researchAssetUrl(binaryUrl), { cache: 'no-store', signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`Activation matrix ${response.status}`);
        return response.arrayBuffer();
      })
      .then((buffer) => setMatrixState({
        datasetId: activeDataset.id,
        matrix: parseFloat16Npy(buffer, payload.binary_shape),
        error: '',
      }))
      .catch((matrixError) => {
        if (matrixError.name !== 'AbortError') {
          setMatrixState({ datasetId: activeDataset.id, matrix: null, error: matrixError.message });
        }
      });
    return () => controller.abort();
  }, [activeDataset, payload]);

  const matrix = matrixState.datasetId === activeDataset?.id ? matrixState.matrix : null;
  const matrixError = matrixState.datasetId === activeDataset?.id ? matrixState.error : '';
  const needsMatrix = Boolean(payload?.binary_url || activeDataset?.binary_path);

  const indexedRows = useMemo(() => (payload?.rows || []).map((row, index) => ({
    ...row,
    source: row.source || row.kind || 'activation',
    _matrixIndex: index,
  })), [payload]);
  const sources = useMemo(() => [...new Set(indexedRows.map((row) => row.source))], [indexedRows]);
  const roles = useMemo(() => [...new Set((payload?.rows || []).map((row) => row.role).filter(Boolean))], [payload]);
  const filteredRows = useMemo(() => indexedRows
    .filter((row) => source === 'all' || row.source === source)
    .filter((row) => role === 'all' || row.role === role)
    .sort((left, right) => Number(left.checkpoint || 0) - Number(right.checkpoint || 0)), [indexedRows, role, source]);
  const visibleRows = useMemo(() => filteredRows.slice(rowStart, rowStart + ROW_WINDOW), [filteredRows, rowStart]);
  useEffect(() => setRowStart(0), [activeDataset?.id, role, source]);

  if (loading) return <div className="encoding-explorer__loading"><RefreshCw size={18} /> Loading encoding catalog...</div>;
  if (error || !catalog) return <div className="encoding-explorer__error">{error || 'Encoding catalog unavailable'}</div>;

  return (
    <div className="encoding-explorer">
      <nav className="encoding-explorer__tabs" aria-label="Encoding research views">
        <button className={view === 'field' ? 'is-active' : ''} onClick={() => setView('field')} type="button"><Binary size={15} /> Field</button>
        <button className={view === 'graph' ? 'is-active' : ''} onClick={() => setView('graph')} type="button"><Network size={15} /> Families</button>
        <button className={view === 'evidence' ? 'is-active' : ''} onClick={() => setView('evidence')} type="button"><Boxes size={15} /> Evidence</button>
      </nav>

      {view === 'field' ? (
        <>
          <div className="encoding-explorer__toolbar">
            <label>Dataset <span><select value={activeDataset?.id || ''} onChange={(event) => setDatasetId(event.target.value)}>{datasets.map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}</select><ChevronDown size={14} /></span></label>
            <label>Signal <span><select value={source} onChange={(event) => setSource(event.target.value)}><option value="all">All signals</option>{sources.map((item) => <option key={item} value={item}>{item}</option>)}</select><ChevronDown size={14} /></span></label>
            <label>Role <span><select value={role} onChange={(event) => setRole(event.target.value)}><option value="all">All roles</option>{roles.map((item) => <option key={item} value={item}>{item}</option>)}</select><ChevronDown size={14} /></span></label>
            <label>Coordinate <input type="number" min="0" max={(activeDataset?.coordinate_count || 1) - 1} value={coordinate} onChange={(event) => setCoordinate(Math.max(0, Math.min((activeDataset?.coordinate_count || 1) - 1, Number(event.target.value) || 0)))} /></label>
            <div className="encoding-explorer__pager" aria-label="Activation row window">
              <button type="button" title="Previous rows" disabled={rowStart === 0} onClick={() => setRowStart(Math.max(0, rowStart - ROW_WINDOW))}><ChevronLeft size={16} /></button>
              <span>{filteredRows.length ? `${rowStart + 1}-${Math.min(rowStart + ROW_WINDOW, filteredRows.length)}` : '0'} / {filteredRows.length}</span>
              <button type="button" title="Next rows" disabled={rowStart + ROW_WINDOW >= filteredRows.length} onClick={() => setRowStart(rowStart + ROW_WINDOW)}><ChevronRight size={16} /></button>
            </div>
          </div>
          <div className="encoding-explorer__metrics">
            <div><Activity size={16} /><span>Rows shown</span><strong>{visibleRows.length}</strong></div>
            <div><Binary size={16} /><span>Coordinates</span><strong>{activeDataset?.coordinate_count || 0}</strong></div>
            <div><Microscope size={16} /><span>Selected</span><strong>{coordinate}</strong></div>
            <div><Boxes size={16} /><span>Phase</span><strong>{activeDataset?.phase}</strong></div>
          </div>
          {payloadError || matrixError ? <div className="encoding-explorer__error">{payloadError || matrixError}</div> : null}
          {!payload || (needsMatrix && !matrix) ? <div className="encoding-explorer__loading"><RefreshCw size={18} /> Loading full-coordinate field...</div> : (
            <>
              <section className="encoding-explorer__field">
                <header><div><span>Physical activation field</span><strong>{activeDataset.title}</strong></div><small>{visibleRows.length} × {activeDataset.coordinate_count}</small></header>
                <div className="encoding-explorer__canvas-scroll"><FieldCanvas rows={visibleRows} coordinateCount={activeDataset.coordinate_count} matrix={matrix} onCoordinate={setCoordinate} /></div>
              </section>
              <section className="encoding-explorer__microscope">
                <header><div><span>Coordinate microscope</span><strong>Coordinate {coordinate}</strong></div><small>{source === 'all' ? 'all signals' : source} / {role}</small></header>
                <MicroscopeCanvas rows={visibleRows} coordinate={coordinate} coordinateCount={activeDataset.coordinate_count} matrix={matrix} />
              </section>
              <p className="encoding-explorer__boundary">{activeDataset.boundary}</p>
            </>
          )}
        </>
      ) : null}

      {view === 'graph' ? <FamilyGraph families={catalog.families} relations={catalog.relations} /> : null}
      {view === 'evidence' ? (
        <div className="encoding-evidence-ledger">
          {datasets.map((dataset) => <article key={dataset.id}><span>{dataset.campaign} / Phase {dataset.phase}</span><strong>{dataset.title}</strong><p>{dataset.boundary}</p><small>{dataset.row_count} rows · {dataset.coordinate_count} coordinates · {dataset.claim_level}</small></article>)}
        </div>
      ) : null}
    </div>
  );
}
