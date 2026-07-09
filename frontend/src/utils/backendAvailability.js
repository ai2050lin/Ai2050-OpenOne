const DEFAULT_COOLDOWN_MS = 30000;
const DEFAULT_PROBE_MS = 1500;

let unavailableUntil = 0;
let probeUntil = 0;

export function isBackendInCooldown() {
  const now = Date.now();
  return now < unavailableUntil || now < probeUntil;
}

export function beginBackendRequest(ms = DEFAULT_PROBE_MS, force = false) {
  if (!force && isBackendInCooldown()) return false;
  probeUntil = Date.now() + ms;
  return true;
}

export function markBackendUnavailable(ms = DEFAULT_COOLDOWN_MS) {
  probeUntil = 0;
  unavailableUntil = Date.now() + ms;
}

export function clearBackendUnavailable() {
  probeUntil = 0;
  unavailableUntil = 0;
}

export function isFetchNetworkError(error) {
  return error instanceof TypeError && /Failed to fetch|NetworkError|Load failed/i.test(error.message || '');
}
