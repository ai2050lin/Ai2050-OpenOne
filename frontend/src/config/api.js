/**
 * API configuration
 * Unified endpoint definitions.
 */

export const API_CONFIG = {
  main: 'http://localhost:5001',
  analysis: 'http://localhost:5001',
  training: 'http://localhost:8000',
  frontend: 'http://localhost:5173',
};

export const API_ENDPOINTS = {
  model: {
    load: `${API_CONFIG.main}/model/load`,
    info: `${API_CONFIG.main}/model/info`,
    generate: `${API_CONFIG.main}/generate`,
  },

  analysis: {
    attention: `${API_CONFIG.main}/nfb/topology`,
    mlp: `${API_CONFIG.main}/nfb/topology`,
    structure: `${API_CONFIG.main}/nfb/topology`,
    topology: `${API_CONFIG.main}/nfb/topology`,
    curvature: `${API_CONFIG.main}/nfb/topology`,
    flow: `${API_CONFIG.main}/nfb/flux`,
  },

  nfb: {
    topology: `${API_CONFIG.main}/nfb/topology`,
    flux: `${API_CONFIG.main}/nfb/flux`,
    alignment: `${API_CONFIG.main}/nfb/alignment`,
    gwtStatus: `${API_CONFIG.main}/nfb/gwt/status`,
    evolutionStatus: `${API_CONFIG.main}/nfb/evolution/status`,
    multimodalAlign: `${API_CONFIG.main}/nfb/multimodal/align`,
  },

  training: {
    status: `${API_CONFIG.training}/training/status`,
    metrics: `${API_CONFIG.main}/toy_experiment/metrics`,
    ricci: `${API_CONFIG.main}/toy_experiment/ricci_metrics`,
  },

  agi: {
    conscious: `${API_CONFIG.main}/nfb_ra/unified_conscious_field`,
    verify: `${API_CONFIG.main}/agi/verify`,
    test: `${API_CONFIG.main}/agi/test`,
  },

  runtime: {
    runs: `${API_CONFIG.main}/api/v1/runs`,
    catalogRoutes: `${API_CONFIG.main}/api/v1/catalog/routes`,
    catalogAnalyses: `${API_CONFIG.main}/api/v1/catalog/analyses`,
    experimentTimeline: (limit = 50, route = null) =>
      `${API_CONFIG.main}/api/v1/experiments/timeline?limit=${limit}${route ? `&route=${encodeURIComponent(route)}` : ''}`,
    weeklyReport: (days = 7, persist = false) =>
      `${API_CONFIG.main}/api/v1/experiments/weekly-report?days=${days}&persist=${persist ? 'true' : 'false'}`,
    runDetail: (runId) => `${API_CONFIG.main}/api/v1/runs/${runId}`,
    runEvents: (runId, limit = 200) => `${API_CONFIG.main}/api/v1/runs/${runId}/events?limit=${limit}`,
  },

  structureExtractor: {
    extract: `${API_CONFIG.main}/structure/extract`,
    compare: `${API_CONFIG.main}/structure/compare`,
  },
};

export async function apiCall(endpoint, options = {}) {
  const defaultOptions = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  const response = await fetch(endpoint, { ...defaultOptions, ...options });
  if (!response.ok) {
    throw new Error(`API Error: ${response.status}`);
  }
  return await response.json();
}

export default API_CONFIG;
