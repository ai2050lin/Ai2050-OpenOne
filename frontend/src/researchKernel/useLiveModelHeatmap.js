import { useCallback, useEffect, useMemo, useState } from 'react';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

const MODEL_KEYS = {
  'qwen3-4b': 'qwen3',
  'glm4-9b': 'glm4',
  ds7b: 'deepseek7b',
};

async function fetchJson(path, options = {}) {
  const response = await fetch(path, { cache: 'no-store', ...options });
  if (!response.ok) {
    let detail = '';
    try {
      const payload = await response.json();
      detail = payload?.detail ? `：${payload.detail}` : '';
    } catch {
      detail = '';
    }
    throw new Error(`请求失败 (${response.status})${detail}`);
  }
  return response.json();
}

export function useLiveModelHeatmap(modelKey) {
  const model = MODEL_KEYS[modelKey] || modelKey || 'qwen3';
  const [runId, setRunId] = useState('');
  const [job, setJob] = useState(null);
  const [liveState, setLiveState] = useState(null);
  const [completedTrace, setCompletedTrace] = useState(null);
  const [error, setError] = useState('');

  const running = ['queued', 'running'].includes(job?.status);

  useEffect(() => {
    if (!runId) return undefined;
    let active = true;
    let timer = null;

    const poll = async () => {
      try {
        const [nextJob, nextState] = await Promise.all([
          fetchJson(`${API_BASE}/api/research-trace/runs/${runId}`),
          fetchJson(`${API_BASE}/api/research-trace/runs/${runId}/live-state`),
        ]);
        if (!active) return;
        setJob(nextJob);
        setLiveState(nextState);
        if (nextJob.status === 'complete') {
          const finalTrace = await fetchJson(`${API_BASE}/api/research-trace/runs/${runId}/trace`);
          if (active) setCompletedTrace(finalTrace);
        }
        if (['failed', 'cancelled', 'interrupted'].includes(nextJob.status)) {
          setError(nextJob.error || `模型运行已${nextJob.status}`);
          return;
        }
        setError('');
        if (['queued', 'running'].includes(nextJob.status)) {
          timer = window.setTimeout(poll, 350);
        }
      } catch (reason) {
        if (!active) return;
        setError(reason?.message || '无法读取模型实时状态');
        timer = window.setTimeout(poll, 1200);
      }
    };

    poll();
    return () => {
      active = false;
      if (timer) window.clearTimeout(timer);
    };
  }, [runId]);

  const start = useCallback(async ({ prompt, topK = 16 }) => {
    const normalizedPrompt = String(prompt || '').trim();
    if (!normalizedPrompt) throw new Error('请输入要分析的文本');
    setError('');
    setLiveState(null);
    setCompletedTrace(null);
    try {
      const nextJob = await fetchJson(`${API_BASE}/api/research-trace/runs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          prompt: normalizedPrompt,
          target_label: 'red',
          top_k: Math.max(4, Math.min(64, Number(topK) || 16)),
          capture_profile: 'full_component',
        }),
      });
      setJob(nextJob);
      setRunId(nextJob.run_id);
      return nextJob;
    } catch (reason) {
      setError(reason?.message || '无法启动本地模型');
      throw reason;
    }
  }, [model]);

  const stop = useCallback(async () => {
    if (!runId) return null;
    const nextJob = await fetchJson(`${API_BASE}/api/research-trace/runs/${runId}`, { method: 'DELETE' });
    setJob(nextJob);
    return nextJob;
  }, [runId]);

  const currentLayer = liveState?.current_layer != null && Number.isFinite(Number(liveState.current_layer))
    ? Number(liveState.current_layer)
    : null;

  const fullStateVectors = useMemo(() => {
    if (!liveState?.run_id || !Array.isArray(liveState?.embedding)) return null;
    const currentKey = currentLayer == null ? null : String(currentLayer);
    const currentHiddenState = currentKey && Array.isArray(liveState?.hidden_state?.[currentKey])
      ? { [currentKey]: liveState.hidden_state[currentKey] }
      : {};
    return {
      schema_version: 'live_state_vectors.v1',
      run_id: liveState.run_id,
      model: liveState.model,
      embedding: liveState.embedding,
      hidden_state: currentHiddenState,
    };
  }, [currentLayer, liveState]);

  const trace = useMemo(() => {
    if (!fullStateVectors) return null;
    if (completedTrace?.run_id === liveState.run_id) return completedTrace;
    return {
      schema_version: 'live_state_heatmap_trace.v1',
      run_id: liveState.run_id,
      status: liveState.status,
      model: liveState.model,
      model_snapshot: liveState.model_snapshot,
      prompt: liveState.prompt,
      target_label: liveState.target_label,
      token_position: liveState.token_position,
      tokens: liveState.tokens || [],
      events: [],
      summary: {
        layer_count: liveState.total_layers,
        completed_layers: liveState.completed_layers,
      },
    };
  }, [completedTrace, fullStateVectors, liveState]);

  return {
    runId,
    job,
    liveState,
    trace: liveState?.model === model ? trace : null,
    fullStateVectors: liveState?.model === model ? fullStateVectors : null,
    currentLayer,
    running,
    ready: Boolean(fullStateVectors),
    error,
    start,
    stop,
  };
}
