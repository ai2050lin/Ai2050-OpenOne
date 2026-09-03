import { useCallback, useEffect, useState } from 'react';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');
const WORKSPACE_URL = `${API_BASE}/api/research-workspace`;

async function readJson(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Research workspace ${response.status}`);
  }
  return payload;
}

export function useResearchWorkspace() {
  const [state, setState] = useState({ snapshot: null, loading: true, saving: false, error: '' });

  const reload = useCallback(async (signal) => {
    setState((current) => ({ ...current, loading: true, error: '' }));
    try {
      const response = await fetch(`${WORKSPACE_URL}/snapshot`, { cache: 'no-store', signal });
      const snapshot = await readJson(response);
      setState({ snapshot, loading: false, saving: false, error: '' });
      return snapshot;
    } catch (error) {
      if (error.name !== 'AbortError') {
        setState((current) => ({ ...current, loading: false, saving: false, error: error.message }));
      }
      return null;
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    reload(controller.signal);
    return () => controller.abort();
  }, [reload]);

  const create = useCallback(async (resource, payload) => {
    setState((current) => ({ ...current, saving: true, error: '' }));
    try {
      const response = await fetch(`${WORKSPACE_URL}/${resource}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      await readJson(response);
      return await reload();
    } catch (error) {
      setState((current) => ({ ...current, saving: false, error: error.message }));
      return null;
    }
  }, [reload]);

  const patch = useCallback(async (resource, payload) => {
    setState((current) => ({ ...current, saving: true, error: '' }));
    try {
      const response = await fetch(`${WORKSPACE_URL}/${resource}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      await readJson(response);
      return await reload();
    } catch (error) {
      setState((current) => ({ ...current, saving: false, error: error.message }));
      return null;
    }
  }, [reload]);

  return { ...state, reload, create, patch };
}
