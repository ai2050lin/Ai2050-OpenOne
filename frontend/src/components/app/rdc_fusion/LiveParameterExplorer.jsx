import { useEffect, useMemo, useState } from 'react';
import ParamMode from './ParamMode.jsx';
import { LIVE_MODEL_DEMO } from './distributedData.js';

const MAX_SLICE_VALUES = 4096;

const fmtCount = n => {
  const value = Number(n) || 0;
  return value >= 1e9 ? (value / 1e9).toFixed(2) + 'B'
    : value >= 1e6 ? (value / 1e6).toFixed(2) + 'M'
      : value >= 1e3 ? (value / 1e3).toFixed(1) + 'K'
        : value.toLocaleString('en-US');
};

const layerOf = name => {
  const match = String(name || '').match(/(?:^|\.)layers\.(\d+)(?:\.|$)/);
  return match ? Number(match[1]) : null;
};

const defaultSliceSize = shape => shape.map(dim => {
  if (shape.length === 1) return Math.min(dim, 64);
  if (shape.length === 2) return Math.min(dim, 16);
  return Math.min(dim, 4);
});

function TensorSliceValues({ payload }) {
  const shape = payload.actual_shape || [];
  const starts = payload.start || [];
  const values = payload.values;

  if (shape.length === 0) {
    return <pre className="fw-live-slice-json">{String(values)}</pre>;
  }
  if (shape.length === 1) {
    return (
      <div className="fw-live-slice-table-wrap">
        <table className="fw-live-slice-table">
          <thead><tr><th>索引</th><th>权重值</th></tr></thead>
          <tbody>{(values || []).map((value, i) => (
            <tr key={i}><td>{(starts[0] || 0) + i}</td><td>{Number(value).toPrecision(8)}</td></tr>
          ))}</tbody>
        </table>
      </div>
    );
  }
  if (shape.length === 2) {
    const rows = Array.isArray(values) ? values : [];
    const cols = Math.max(0, ...rows.map(row => Array.isArray(row) ? row.length : 1));
    return (
      <div className="fw-live-slice-table-wrap">
        <table className="fw-live-slice-table fw-live-slice-matrix">
          <thead><tr><th>行 / 列</th>{Array.from({ length: cols }, (_, c) => (
            <th key={c}>{(starts[1] || 0) + c}</th>
          ))}</tr></thead>
          <tbody>{rows.map((row, r) => (
            <tr key={r}>
              <th>{(starts[0] || 0) + r}</th>
              {(Array.isArray(row) ? row : [row]).map((value, c) => (
                <td key={c}>{Number(value).toPrecision(7)}</td>
              ))}
            </tr>
          ))}</tbody>
        </table>
      </div>
    );
  }
  return <pre className="fw-live-slice-json">{JSON.stringify(values, null, 2)}</pre>;
}

export default function LiveParameterExplorer({ conn, onSelectLayer, selectedLayer = -1 }) {
  const [query, setQuery] = useState('');
  const [selectedName, setSelectedName] = useState('');
  const [sliceStart, setSliceStart] = useState([]);
  const [sliceSize, setSliceSize] = useState([]);
  const [slice, setSlice] = useState(null);
  const [sliceError, setSliceError] = useState('');
  const [sliceBusy, setSliceBusy] = useState(false);

  const data = conn || LIVE_MODEL_DEMO;
  const tensors = useMemo(() => {
    if (Array.isArray(data.tensors) && data.tensors.length) return data.tensors;
    return (data.groups || []).flatMap(group =>
      (group.tensors || []).map(tensor => ({ ...tensor, group: group.prefix })),
    );
  }, [data]);
  const selected = tensors.find(tensor => tensor.name === selectedName) || null;
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return tensors;
    return tensors.filter(tensor =>
      [tensor.name, tensor.group, tensor.file, tensor.dtype, (tensor.shape || []).join('x')]
        .filter(Boolean).join(' ').toLowerCase().includes(needle),
    );
  }, [query, tensors]);

  useEffect(() => {
    setSelectedName('');
    setSliceStart([]);
    setSliceSize([]);
    setSlice(null);
    setSliceError('');
    onSelectLayer && onSelectLayer(-1);
  }, [conn, onSelectLayer]);

  const chooseTensor = tensor => {
    setSelectedName(tensor.name);
    setSlice(null);
    setSliceError('');
    const shape = tensor.shape || [];
    setSliceStart(shape.map(() => 0));
    setSliceSize(defaultSliceSize(shape));
    const layer = layerOf(tensor.name);
    onSelectLayer && onSelectLayer(layer == null ? -1 : layer);
  };

  const selectedShape = selected?.shape || [];
  const requestedValues = sliceSize.reduce((total, size) => total * Math.max(0, Number(size) || 0), 1);
  const sliceFits = requestedValues > 0 && requestedValues <= MAX_SLICE_VALUES;

  const loadSlice = async () => {
    if (!conn || !selected || !sliceFits || sliceBusy) return;
    setSliceBusy(true);
    setSliceError('');
    try {
      const response = await fetch(API_BASE + '/api/live/tensor/slice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: selected.name,
          start: sliceStart.map(value => Number(value) || 0),
          size: sliceSize.map(value => Number(value) || 0),
        }),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || ('HTTP ' + response.status));
      setSlice(body);
    } catch (error) {
      setSliceError(String(error.message || error).includes('Failed to fetch')
        ? '中心节点不可达（:5001）' : String(error.message || error));
    } finally {
      setSliceBusy(false);
    }
  };

  const setAxisValue = (setter, index, value) => {
    setter(values => values.map((old, i) => i === index ? value : old));
  };
  const totalTensors = tensors.length;
  const totalParams = Number(data.total_params) || tensors.reduce((sum, tensor) => sum + (Number(tensor.params) || 0), 0);

  return (
    <section className="fw-live-browser">
      <header className="fw-live-browser-head">
        <div>
          <h3>完整参数目录</h3>
          <p>{conn ? 'LIVE · safetensors 真实目录' : 'DEMO · 仅展示样例结构；连接本地模型后读取完整清单'}</p>
        </div>
        <div className="fw-live-browser-counts">
          <span><b>{totalTensors.toLocaleString('en-US')}</b> tensors</span>
          <span><b>{fmtCount(totalParams)}</b> parameters</span>
          <span><b>{(data.files || []).length}</b> 个权重分片</span>
        </div>
      </header>

      {data.demo_note && !conn && <div className="fw-live-demo-note">{data.demo_note}</div>}

      <div className="fw-live-browser-main">
        <div className="fw-live-catalog">
          <div className="fw-live-catalog-head">
            <b>Tensor 清单</b>
            <span>{filtered.length.toLocaleString('en-US')} / {totalTensors.toLocaleString('en-US')}</span>
          </div>
          <label className="fw-live-search">
            <span>搜索名称、层、形状、dtype 或分片</span>
            <input
              value={query}
              onChange={event => setQuery(event.target.value)}
              placeholder="如 layers.12 · q_proj · bf16"
            />
          </label>
          <div className="fw-live-table-scroll">
            <table className="fw-live-catalog-table">
              <thead>
                <tr>
                  <th>Tensor 名称</th><th>形状</th><th>dtype</th><th>参数量</th><th>文件分片</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(tensor => (
                  <tr
                    key={tensor.name}
                    className={selectedName === tensor.name ? 'selected' : ''}
                    onClick={() => chooseTensor(tensor)}
                    tabIndex={0}
                    onKeyDown={event => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        chooseTensor(tensor);
                      }
                    }}
                    title={tensor.name}
                  >
                    <td className="mono">{tensor.name}</td>
                    <td className="mono">[{(tensor.shape || []).join(' × ')}]</td>
                    <td>{tensor.dtype || '—'}</td>
                    <td className="mono">{fmtCount(tensor.params)}</td>
                    <td className="mono">{tensor.file || '—'}</td>
                  </tr>
                ))}
                {!filtered.length && <tr><td colSpan="5" className="fw-live-empty">没有匹配的 tensor</td></tr>}
              </tbody>
            </table>
          </div>
        </div>

        <aside className="fw-live-inspector">
          {!selected ? (
            <div className="fw-live-inspector-empty">选择一个 tensor 查看真实统计、分块热图和权重数值切片。</div>
          ) : (
            <>
              <div className="fw-live-selected-head">
                <div>
                  <b className="mono">{selected.name}</b>
                  <span>[{selectedShape.join(' × ')}] · {selected.dtype} · {fmtCount(selected.params)} 参数</span>
                </div>
              </div>

              {conn ? (
                <>
                  <div className="fw-live-heatmap-wrap">
                    <ParamMode
                      key={selected.name}
                      layer={selectedLayer >= 0 ? selectedLayer : 0}
                      live={conn}
                      scan={null}
                      tensorName={selected.name}
                    />
                  </div>
                  <div className="fw-live-slice">
                    <div className="fw-live-slice-head">
                      <b>原始权重切片</b>
                      <span>每次最多读取 {MAX_SLICE_VALUES.toLocaleString('en-US')} 个标量；不会整体载入 tensor</span>
                    </div>
                    {selectedShape.length === 0 ? (
                      <div className="fw-live-slice-controls">
                        <p className="dim2">该 tensor 是标量，可直接读取完整值。</p>
                        <button type="button" className="fw-tbtn primary"
                          disabled={!sliceFits || sliceBusy}
                          onClick={loadSlice}>
                          {sliceBusy ? '读取中…' : '读取标量值'}
                        </button>
                      </div>
                    ) : (
                      <div className="fw-live-slice-controls">
                        {selectedShape.map((dim, index) => (
                          <div className="fw-live-axis" key={index}>
                            <b>轴 {index}</b><span>长度 {dim.toLocaleString('en-US')}</span>
                            <label>起点<input type="number" min="0" max={Math.max(0, dim - 1)}
                              value={sliceStart[index] ?? 0}
                              onChange={event => setAxisValue(setSliceStart, index, event.target.value)} /></label>
                            <label>数量<input type="number" min="1" max={dim}
                              value={sliceSize[index] ?? 1}
                              onChange={event => setAxisValue(setSliceSize, index, event.target.value)} /></label>
                          </div>
                        ))}
                        <button type="button" className="fw-tbtn primary"
                          disabled={!sliceFits || sliceBusy}
                          onClick={loadSlice}>
                          {sliceBusy ? '读取中…' : '读取数值切片'}
                        </button>
                      </div>
                    )}
                    {!sliceFits && <div className="fw-live-slice-error">当前切片超过上限或为空，请缩小范围。</div>}
                    {sliceError && <div className="fw-live-slice-error">{sliceError}</div>}
                    {slice && (
                      <div className="fw-live-slice-result">
                        <div className="fw-live-slice-meta">
                          精确 safetensors 切片 · 起点 [{slice.start.join(', ')}] · 实读形状 [{slice.actual_shape.join(' × ')}] · {slice.loaded_numel.toLocaleString('en-US')} 个值
                        </div>
                        <TensorSliceValues payload={slice} />
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <div className="fw-live-demo-note">DEMO 只提供架构与 tensor 清单样例，不提供真实权重统计或标量值。连接本地 safetensors 模型后可读取。</div>
              )}
            </>
          )}
        </aside>
      </div>
    </section>
  );
}
