import {
  Activity,
  Braces,
  ChevronDown,
  ChevronRight,
  Database,
  GitBranch,
  Layers,
  RefreshCw,
  Search,
  Target,
  X,
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import './MechanismTraceExplorer.css';

const FACTOR_KEYS = ['O', 'R', 'A', 'C', 'F', 'M', 'K', 'S_answer', 'B', 'G', 'N', 'P', 'T'];

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a';
  const num = Number(value);
  if (!Number.isFinite(num)) return 'n/a';
  if (Math.abs(num) >= 100) return num.toFixed(1);
  if (Math.abs(num) >= 10) return num.toFixed(2);
  return num.toFixed(digits);
}

function marginColor(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '#48505a';
  if (value >= 2) return '#1f9d62';
  if (value >= 0) return '#b88918';
  return '#c94b4b';
}

function layerLabel(layer) {
  return layer?.layer === -1 ? 'Emb' : `L${layer?.layer}`;
}

function JsonBlock({ data }) {
  return (
    <pre className="mechanism-json">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}

export default function MechanismTraceExplorer({ onClose }) {
  const [manifest, setManifest] = useState(null);
  const [trace, setTrace] = useState(null);
  const [selectedId, setSelectedId] = useState('');
  const [selectedLayerIdx, setSelectedLayerIdx] = useState(null);
  const [detailTab, setDetailTab] = useState('factors');
  const [expandedPipeline, setExpandedPipeline] = useState(true);
  const [expandedRaw, setExpandedRaw] = useState(false);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const loadManifest = async () => {
    setError('');
    setLoading(true);
    try {
      const res = await fetch('/vis_data/mechanism_trace/manifest.json', { cache: 'no-store' });
      if (!res.ok) throw new Error(`manifest ${res.status}`);
      const data = await res.json();
      setManifest(data);
      const firstId = data?.items?.[0]?.id || '';
      setSelectedId((prev) => prev || firstId);
    } catch (err) {
      setError(`无法读取机制 Trace 数据: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadManifest();
  }, []);

  useEffect(() => {
    const item = manifest?.items?.find((row) => row.id === selectedId);
    if (!item?.path) return;
    let cancelled = false;
    const loadTrace = async () => {
      setError('');
      setLoading(true);
      try {
        const res = await fetch(item.path, { cache: 'no-store' });
        if (!res.ok) throw new Error(`${item.path} ${res.status}`);
        const data = await res.json();
        if (cancelled) return;
        setTrace(data);
        setSelectedLayerIdx(Math.max(0, (data.layers?.length || 1) - 1));
      } catch (err) {
        if (!cancelled) setError(`无法读取 Trace 文件: ${err.message}`);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    loadTrace();
    return () => {
      cancelled = true;
    };
  }, [manifest, selectedId]);

  const selectedLayer = trace?.layers?.[selectedLayerIdx] || null;
  const filteredLayers = useMemo(() => {
    const layers = trace?.layers || [];
    const text = query.trim().toLowerCase();
    if (!text) return layers;
    return layers.filter((layer) => {
      const label = layerLabel(layer).toLowerCase();
      const factorHit = FACTOR_KEYS.some((key) => {
        const factor = layer.factors?.[key];
        return `${key} ${factor?.中文 || ''} ${factor?.label || ''}`.toLowerCase().includes(text);
      });
      return label.includes(text) || factorHit;
    });
  }, [trace, query]);

  const selectedField = selectedLayer?.positions?.answer?.candidate_field || {};
  const factorDefinitions = trace?.factor_definitions || {};
  const selectedItem = manifest?.items?.find((row) => row.id === selectedId);

  return (
    <div className="mechanism-overlay" role="dialog" aria-modal="true">
      <div className="mechanism-shell">
        <header className="mechanism-header">
          <div className="mechanism-title">
            <GitBranch size={20} />
            <div>
              <h2>全链路闭合机制 Trace</h2>
              <p>上文状态 → 条件化路由 → 候选空间 → 知识路径 → 输出边界 → 下一 token</p>
            </div>
          </div>
          <div className="mechanism-header-actions">
            <button className="mechanism-icon-button" onClick={loadManifest} title="刷新数据">
              <RefreshCw size={18} />
            </button>
            <button className="mechanism-icon-button" onClick={onClose} title="关闭">
              <X size={19} />
            </button>
          </div>
        </header>

        <section className="mechanism-toolbar">
          <label>
            <span>测试结果</span>
            <select value={selectedId} onChange={(event) => setSelectedId(event.target.value)}>
              {(manifest?.items || []).map((item) => (
                <option key={item.id} value={item.id}>{item.label || item.id}</option>
              ))}
            </select>
          </label>
          <label className="mechanism-search">
            <Search size={16} />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="筛选 layer / factor"
            />
          </label>
          <div className="mechanism-status">
            {loading ? '读取中...' : error ? error : `${trace?.layers?.length || 0} 层记录`}
          </div>
        </section>

        {trace && (
          <main className="mechanism-main">
            <section className="mechanism-summary-band">
              <div className="mechanism-metric">
                <Target size={18} />
                <span>目标</span>
                <strong>{trace.target?.attribute} / {trace.target?.object}</strong>
              </div>
              <div className="mechanism-metric">
                <Database size={18} />
                <span>全局 rank</span>
                <strong>{trace.summary?.target_global_rank ?? 'n/a'}</strong>
              </div>
              <div className="mechanism-metric">
                <Activity size={18} />
                <span>最终 margin</span>
                <strong>{fmt(trace.summary?.final_margin_vs_color_competitor)}</strong>
              </div>
              <div className="mechanism-metric">
                <Layers size={18} />
                <span>下一 token</span>
                <strong>{trace.summary?.next_token?.token || 'n/a'}</strong>
              </div>
            </section>

            <section className="mechanism-prompt-band">
              <div>
                <span className="mechanism-small-label">Prompt</span>
                <p>{trace.prompt}</p>
              </div>
              <div>
                <span className="mechanism-small-label">数据文件</span>
                <p>{selectedItem?.path}</p>
              </div>
            </section>

            <section className="mechanism-token-strip">
              {(trace.tokens || []).map((token) => (
                <button
                  key={`${token.index}-${token.id}`}
                  className={`mechanism-token ${String(token.role || '').includes('answer') ? 'is-answer' : ''}`}
                  title={`#${token.index} id=${token.id} role=${token.role}`}
                >
                  <span>{token.index}</span>
                  {token.text}
                </button>
              ))}
            </section>

            <section className="mechanism-layout">
              <div className="mechanism-left">
                <button className="mechanism-section-toggle" onClick={() => setExpandedPipeline((prev) => !prev)}>
                  {expandedPipeline ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                  推理阶段
                </button>
                {expandedPipeline && (
                  <div className="mechanism-pipeline">
                    {(trace.pipeline || []).map((step, idx) => (
                      <div key={step.id} className="mechanism-step">
                        <span>{idx + 1}</span>
                        <div>
                          <strong>{step.中文}</strong>
                          <p>{step.name}</p>
                          <small>{(step.factor_keys || []).join(' / ')}</small>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                <div className="mechanism-layer-list" aria-label="layer trace">
                  {filteredLayers.map((layer) => {
                    const margin = layer.positions?.answer?.candidate_field?.margin_vs_competitor;
                    const active = selectedLayer?.layer === layer.layer;
                    return (
                      <button
                        key={layer.name}
                        className={`mechanism-layer-button ${active ? 'active' : ''}`}
                        onClick={() => setSelectedLayerIdx(trace.layers.indexOf(layer))}
                        title={`${layer.name}: margin=${fmt(margin)}`}
                      >
                        <span>{layerLabel(layer)}</span>
                        <i style={{ background: marginColor(margin), width: `${Math.min(100, Math.max(8, Math.abs(Number(margin || 0)) * 14))}%` }} />
                        <em>{fmt(margin)}</em>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="mechanism-center">
                <div className="mechanism-layer-head">
                  <div>
                    <span className="mechanism-small-label">当前层</span>
                    <h3>{selectedLayer ? `${layerLabel(selectedLayer)} · ${selectedLayer.name}` : 'n/a'}</h3>
                  </div>
                  <div className="mechanism-tabs">
                    {['factors', 'candidates', 'components', 'raw'].map((tab) => (
                      <button key={tab} className={detailTab === tab ? 'active' : ''} onClick={() => setDetailTab(tab)}>
                        {tab}
                      </button>
                    ))}
                  </div>
                </div>

                {detailTab === 'factors' && (
                  <div className="mechanism-factor-grid">
                    {FACTOR_KEYS.map((key) => {
                      const factor = selectedLayer?.factors?.[key];
                      const def = factorDefinitions[key];
                      return (
                        <div key={key} className="mechanism-factor-tile">
                          <div>
                            <strong>{key}</strong>
                            <span>{factor?.中文 || def?.中文 || factor?.label}</span>
                          </div>
                          <b>{fmt(factor?.value)}</b>
                          <small>{def?.formula}</small>
                        </div>
                      );
                    })}
                  </div>
                )}

                {detailTab === 'candidates' && (
                  <div className="mechanism-table-wrap">
                    <table className="mechanism-table">
                      <thead>
                        <tr>
                          <th>候选</th>
                          <th>token</th>
                          <th>score</th>
                          <th>probability</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(selectedField.scores || []).map((row) => (
                          <tr key={row.label} className={row.label === trace.target?.attribute ? 'target-row' : ''}>
                            <td>{row.label}</td>
                            <td>{row.token_id}</td>
                            <td>{fmt(row.score)}</td>
                            <td>{fmt((row.probability || 0) * 100, 2)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                {detailTab === 'components' && (
                  <div className="mechanism-component-grid">
                    {['attention', 'mlp'].map((name) => {
                      const component = selectedLayer?.components?.[name];
                      const field = component?.candidate_field || {};
                      return (
                        <div key={name} className="mechanism-component-tile">
                          <h4>{name}</h4>
                          <dl>
                            <dt>norm</dt><dd>{fmt(component?.norm)}</dd>
                            <dt>cosine_to_residual</dt><dd>{fmt(component?.cosine_to_residual)}</dd>
                            <dt>target_score</dt><dd>{fmt(field.target_score)}</dd>
                            <dt>margin</dt><dd>{fmt(field.margin_vs_competitor)}</dd>
                            <dt>competitor</dt><dd>{field.competitor_label || 'n/a'}</dd>
                          </dl>
                        </div>
                      );
                    })}
                  </div>
                )}

                {detailTab === 'raw' && (
                  <JsonBlock data={selectedLayer} />
                )}
              </div>

              <aside className="mechanism-right">
                <h3>详细数据</h3>
                <div className="mechanism-detail-list">
                  <div><span>schema</span><b>{trace.schema_version}</b></div>
                  <div><span>phase</span><b>{trace.phase}</b></div>
                  <div><span>model</span><b>{trace.model}</b></div>
                  <div><span>global_closed</span><b>{trace.summary?.global_closed ? 'true' : 'false'}</b></div>
                  <div><span>candidate_closed</span><b>{trace.summary?.candidate_closed ? 'true' : 'false'}</b></div>
                  <div><span>competitor</span><b>{trace.summary?.final_competitor_label}</b></div>
                  <div><span>gate_open</span><b>{fmt(trace.summary?.final_candidate_gate_open)}</b></div>
                </div>

                <h3>最终 Top Tokens</h3>
                <div className="mechanism-top-token-list">
                  {(trace.top_tokens || []).map((row) => (
                    <div key={`${row.rank}-${row.token_id}`}>
                      <span>{row.rank}. {row.token || '<blank>'}</span>
                      <b>{fmt(row.logit)}</b>
                    </div>
                  ))}
                </div>

                <button className="mechanism-section-toggle raw-toggle" onClick={() => setExpandedRaw((prev) => !prev)}>
                  <Braces size={16} />
                  {expandedRaw ? '收起完整 JSON' : '查看完整 JSON'}
                </button>
                {expandedRaw && <JsonBlock data={trace} />}
              </aside>
            </section>
          </main>
        )}
      </div>
    </div>
  );
}
