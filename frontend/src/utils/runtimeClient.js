import { beginBackendRequest, clearBackendUnavailable, isBackendInCooldown, isFetchNetworkError, markBackendUnavailable } from './backendAvailability';

export async function createRunAndFetchEvents(apiBase, runRequest, eventLimit = 20) {
  if (!beginBackendRequest()) {
    throw new Error('backend unavailable cooldown');
  }
  let runRes;
  try {
    runRes = await fetch(`${apiBase}/api/v1/runs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(runRequest),
    });
  } catch (error) {
    if (isFetchNetworkError(error)) markBackendUnavailable();
    throw error;
  }
  if (!runRes.ok) {
    throw new Error(`runtime run create failed: ${runRes.status}`);
  }
  clearBackendUnavailable();
  const runPayload = await runRes.json();
  const runId = runPayload?.run?.run_id;
  if (!runId) {
    throw new Error('runtime run_id missing');
  }

  let eventRes;
  try {
    eventRes = await fetch(`${apiBase}/api/v1/runs/${runId}/events?limit=${eventLimit}`);
  } catch (error) {
    if (isFetchNetworkError(error)) markBackendUnavailable();
    throw error;
  }
  if (!eventRes.ok) {
    throw new Error(`runtime events failed: ${eventRes.status}`);
  }
  clearBackendUnavailable();
  const eventPayload = await eventRes.json();
  return Array.isArray(eventPayload?.events) ? eventPayload.events : [];
}

export async function pollRuntimeWithFallback({
  apiBase,
  runRequest,
  mapRuntimeEvents,
  fetchLegacy,
  eventLimit = 20,
}) {
  try {
    const events = await createRunAndFetchEvents(apiBase, runRequest, eventLimit);
    const runtimeData = mapRuntimeEvents(events);
    if (!runtimeData) {
      throw new Error('runtime event payload incomplete');
    }
    return { source: 'runtime-v1', data: runtimeData };
  } catch (runtimeErr) {
    if (!fetchLegacy) {
      throw runtimeErr;
    }
    if (isBackendInCooldown()) {
      throw runtimeErr;
    }
    const legacyData = await fetchLegacy();
    return { source: 'legacy', data: legacyData };
  }
}

