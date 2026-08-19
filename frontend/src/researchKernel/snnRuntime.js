function appendLayerSpikes(target, frame) {
  if (!frame || typeof frame !== 'object') return target;
  Object.entries(frame).forEach(([layerName, indices]) => {
    if (!Array.isArray(indices)) return;
    const current = target[layerName] || [];
    target[layerName] = [...new Set([...current, ...indices])].sort((a, b) => a - b);
  });
  return target;
}

export function collectSnnSpikes(payload) {
  if (payload?.spikes && typeof payload.spikes === 'object') {
    return appendLayerSpikes({}, payload.spikes);
  }
  return (Array.isArray(payload?.history) ? payload.history : []).reduce(
    (accumulator, frame) => appendLayerSpikes(accumulator, frame),
    {},
  );
}

