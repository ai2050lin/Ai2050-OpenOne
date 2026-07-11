import { useCallback, useEffect, useMemo, useState } from 'react';

import { buildForwardData, eventFor, MODEL_KEY_MAP, useResearchKernel } from './useResearchKernel';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');
const FROZEN_MANIFEST = '/vis_data/real_component_trace/manifest.json';

async function fetchJson(path, options) {
  const response = await fetch(path, { cache: 'no-store', ...options });
  if (!response.ok) {
    let detail = '';
    try {
      const body = await response.json();
      detail = body?.detail ? `: ${body.detail}` : '';
    } catch {
      detail = '';
    }
    throw new Error(`${path} ${response.status}${detail}`);
  }
  return response.json();
}

function normalizeFrozenRuns(manifest) {
  return (manifest?.items || []).map((item) => ({
    ...item,
    source_mode: 'replay',
    status: 'complete',
    validated: true,
  }));
}

export function useResearchWorkspace({
  sourceMode,
  modelKey,
  replayRunId,
  liveRunId,
  currentLayer,
  subphaseId,
}) {
  const model = MODEL_KEY_MAP[modelKey] || modelKey || 'qwen3';
  const kernel = useResearchKernel(modelKey, currentLayer, subphaseId);
  const [runs, setRuns] = useState([]);
  const [apiAvailable, setApiAvailable] = useState(true);
  const [traceState, setTraceState] = useState({ runId: '', payload: null });
  const [liveJob, setLiveJob] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const refreshRuns = useCallback(async () => {
    try {
      const payload = await fetchJson(`${API_BASE}/api/research-trace/runs`);
      setRuns(payload.runs || []);
      setApiAvailable(true);
      return payload.runs || [];
    } catch {
      const manifest = await fetchJson(FROZEN_MANIFEST);
      const frozen = normalizeFrozenRuns(manifest);
      setRuns(frozen);
      setApiAvailable(false);
      return frozen;
    }
  }, []);

  useEffect(() => {
    Promise.resolve()
      .then(() => refreshRuns())
      .catch((reason) => setError(reason?.message || 'Trace 清单读取失败'));
  }, [refreshRuns]);

  const replayRuns = useMemo(
    () => runs.filter((run) => run.source_mode === 'replay' && run.validated !== false),
    [runs]
  );
  const effectiveReplayRun = useMemo(() => {
    const requested = replayRuns.find((run) => run.run_id === replayRunId);
    return requested || replayRuns.find((run) => run.model === model) || null;
  }, [model, replayRunId, replayRuns]);

  useEffect(() => {
    if (sourceMode !== 'replay' || !effectiveReplayRun) return;
    let active = true;
    const path = effectiveReplayRun.path || `${API_BASE}/api/research-trace/runs/${effectiveReplayRun.run_id}/trace`;
    Promise.resolve()
      .then(() => {
        if (!active) return null;
        setLoading(true);
        setError('');
        return fetchJson(path);
      })
      .then((payload) => {
        if (!active || !payload) return;
        setTraceState({ runId: effectiveReplayRun.run_id, payload });
        setError('');
      })
      .catch((reason) => { if (active) setError(reason?.message || '冻结 Trace 读取失败'); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [effectiveReplayRun, sourceMode]);

  useEffect(() => {
    if (sourceMode !== 'live' || !liveRunId) return;
    let active = true;
    let timer = null;
    const poll = async () => {
      try {
        const job = await fetchJson(`${API_BASE}/api/research-trace/runs/${liveRunId}`);
        if (!active) return;
        setLiveJob(job);
        setApiAvailable(true);
        if (job.status === 'complete') {
          const trace = await fetchJson(`${API_BASE}/api/research-trace/runs/${liveRunId}/trace`);
          if (!active) return;
          setTraceState({ runId: liveRunId, payload: trace });
          setLoading(false);
          setError('');
          refreshRuns().catch(() => {});
          return;
        }
        if (['failed', 'cancelled', 'interrupted'].includes(job.status)) {
          setLoading(false);
          setError(job.error || `实时 Trace ${job.status}`);
          return;
        }
        setLoading(true);
        timer = setTimeout(poll, 1500);
      } catch (reason) {
        if (!active) return;
        setLoading(false);
        setApiAvailable(false);
        setError(reason?.message || '实时 Trace 状态读取失败');
      }
    };
    poll();
    return () => {
      active = false;
      if (timer) clearTimeout(timer);
    };
  }, [liveRunId, refreshRuns, sourceMode]);

  const createLiveRun = useCallback(async ({ prompt, targetLabel, topK = 16 }) => {
    setLoading(true);
    setError('');
    try {
      const job = await fetchJson(`${API_BASE}/api/research-trace/runs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          prompt,
          target_label: targetLabel,
          top_k: topK,
          capture_profile: 'full_component',
        }),
      });
      setLiveJob(job);
      setApiAvailable(true);
      return job;
    } catch (reason) {
      setLoading(false);
      setApiAvailable(false);
      setError(reason?.message || '无法启动实时 Trace');
      throw reason;
    }
  }, [model]);

  const cancelLiveRun = useCallback(async () => {
    if (!liveRunId) return null;
    const job = await fetchJson(`${API_BASE}/api/research-trace/runs/${liveRunId}`, { method: 'DELETE' });
    setLiveJob(job);
    setLoading(false);
    return job;
  }, [liveRunId]);

  const activeTrace = traceState.payload;
  const activeRunId = sourceMode === 'live' ? liveRunId : effectiveReplayRun?.run_id || '';
  const trace = traceState.runId === activeRunId ? activeTrace : null;
  const currentEvent = useMemo(
    () => eventFor(trace?.events || [], currentLayer, subphaseId),
    [currentLayer, subphaseId, trace]
  );
  const forwardData = useMemo(() => buildForwardData(trace), [trace]);
  const sourceModel = sourceMode === 'live' ? model : effectiveReplayRun?.model;
  const modelMismatch = Boolean(sourceModel && sourceModel !== model);

  return {
    model,
    sourceMode,
    runs,
    replayRuns,
    effectiveReplayRun,
    activeRunId,
    liveJob,
    trace,
    currentEvent,
    forwardData,
    stableUnits: kernel.stableUnits,
    kernelManifest: kernel.kernelManifest,
    loading,
    error: error || kernel.error,
    ready: Boolean(trace && forwardData),
    apiAvailable,
    modelMismatch,
    refreshRuns,
    createLiveRun,
    cancelLiveRun,
  };
}
