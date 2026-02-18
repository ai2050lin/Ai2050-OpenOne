export async function createRunAndFetchEvents(apiBase, runRequest, eventLimit = 20) {
  const runRes = await fetch(`${apiBase}/api/v1/runs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(runRequest),
  });
  if (!runRes.ok) {
    throw new Error(`runtime run create failed: ${runRes.status}`);
  }
  const runPayload = await runRes.json();
  const runId = runPayload?.run?.run_id;
  if (!runId) {
    throw new Error('runtime run_id missing');
  }

  const eventRes = await fetch(`${apiBase}/api/v1/runs/${runId}/events?limit=${eventLimit}`);
  if (!eventRes.ok) {
    throw new Error(`runtime events failed: ${eventRes.status}`);
  }
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
    const legacyData = await fetchLegacy();
    return { source: 'legacy', data: legacyData };
  }
}

