import { useEffect, useMemo, useState } from 'react';

const KERNEL_BASE = '/vis_data/research_kernel';
const TRACE_MANIFEST = '/vis_data/real_component_trace/manifest.json';

const MODEL_KEY_MAP = {
  'qwen3-4b': 'qwen3',
  'glm4-9b': 'glm4',
  ds7b: 'deepseek7b',
};

const EVENT_BY_SUBPHASE = {
  input: ['residual_input'],
  ln1: ['norm1'],
  qkv: ['q_projection', 'qkv_projection'],
  // Phase287 does not expose pre-softmax scores or attention probabilities.
  // Keep these empty instead of presenting K/V projections as direct evidence.
  attn_score: [],
  softmax: [],
  attn_out: ['attention_output'],
  residual1: ['residual1'],
  ln2: ['norm2'],
  ffn_up: ['mlp_up', 'mlp_gate_up_merged', 'mlp_gate'],
  ffn_act: ['mlp_product'],
  ffn_down: ['mlp_down'],
  residual2: ['residual2'],
};

async function fetchJson(path) {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${path} ${response.status}`);
  return response.json();
}

async function fetchJsonl(path) {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${path} ${response.status}`);
  const text = await response.text();
  return text
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function eventFor(events, layer, subphaseId) {
  const accepted = EVENT_BY_SUBPHASE[subphaseId] || [];
  return events.find((event) => Number(event.layer) === Number(layer) && accepted.includes(event.event_type)) || null;
}

function buildForwardData(trace) {
  if (!trace?.events?.length) return null;
  const layerCount = Number(trace.model_snapshot?.num_hidden_layers || 0);
  const byLayer = new Map();
  trace.events.forEach((event) => {
    const layer = Number(event.layer);
    if (layer < 0) return;
    if (!byLayer.has(layer)) byLayer.set(layer, {});
    byLayer.get(layer)[event.event_type] = event;
  });
  return {
    schema_version: trace.schema_version,
    model: trace.model,
    model_info: trace.model_snapshot,
    sentence: trace.prompt,
    tokens: trace.tokens || [],
    evidence_level: 'L2',
    source_run_id: trace.run_id,
    layers: Array.from({ length: layerCount }, (_, layer) => {
      const events = byLayer.get(layer) || {};
      const attention = events.attention_output || {};
      const gate = events.mlp_gate || events.mlp_gate_up_merged || {};
      const product = events.mlp_product || {};
      const residual = events.residual2 || {};
      return {
        layer,
        label: `L${layer} real trace`,
        attention: { norm: attention.norm ?? null },
        ffn: {
          gate_activation: gate.norm ?? null,
          norm: product.norm ?? null,
          top_neurons: product.top_units || [],
        },
        residual_norm: residual.norm ?? null,
        candidate_field: residual.candidate_field || null,
      };
    }),
  };
}

export function useResearchKernel(fpModel, fpCurrentLayer, subphaseId) {
  const model = MODEL_KEY_MAP[fpModel] || 'qwen3';
  const [kernelManifest, setKernelManifest] = useState(null);
  const [traceManifest, setTraceManifest] = useState(null);
  const [traceState, setTraceState] = useState({ model: '', payload: null });
  const [unitState, setUnitState] = useState({ model: '', rows: [] });
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    Promise.all([fetchJson(`${KERNEL_BASE}/manifest.json`), fetchJson(TRACE_MANIFEST)])
      .then(([kernel, traces]) => {
        if (!active) return;
        setKernelManifest(kernel);
        setTraceManifest(traces);
        setError('');
      })
      .catch((err) => {
        if (!active) return;
        setError(err?.message || 'research kernel unavailable');
      });
    return () => { active = false; };
  }, []);

  const traceItem = useMemo(
    () => (traceManifest?.items || []).find((item) => item.model === model) || null,
    [model, traceManifest]
  );
  const stableRun = useMemo(
    () => (kernelManifest?.runs || []).find((run) => run.model === model && Number(run.phase) === 286) || null,
    [kernelManifest, model]
  );

  useEffect(() => {
    let active = true;
    if (!traceItem?.path) return () => { active = false; };
    fetchJson(traceItem.path)
      .then((payload) => { if (active) setTraceState({ model, payload }); })
      .catch((err) => { if (active) setError(err?.message || 'trace load failed'); });
    return () => { active = false; };
  }, [model, traceItem]);

  useEffect(() => {
    let active = true;
    if (!stableRun?.unit_path) return () => { active = false; };
    fetchJsonl(`${KERNEL_BASE}/${stableRun.unit_path}`)
      .then((rows) => { if (active) setUnitState({ model, rows }); })
      .catch((err) => { if (active) setError(err?.message || 'unit evidence load failed'); });
    return () => { active = false; };
  }, [model, stableRun]);

  const trace = traceState.model === model ? traceState.payload : null;
  const stableUnits = unitState.model === model ? unitState.rows : [];

  const currentEvent = useMemo(
    () => eventFor(trace?.events || [], fpCurrentLayer, subphaseId),
    [fpCurrentLayer, subphaseId, trace]
  );
  const forwardData = useMemo(() => buildForwardData(trace), [trace]);

  return {
    model,
    kernelManifest,
    traceManifest,
    traceItem,
    stableRun,
    trace,
    stableUnits,
    currentEvent,
    forwardData,
    error,
    ready: Boolean(trace && forwardData),
  };
}

export { MODEL_KEY_MAP };
