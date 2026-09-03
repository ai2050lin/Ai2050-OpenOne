import { useEffect, useState } from 'react';

import { researchAssetUrl } from '../config/researchAssets';

const SOURCES = {
  models: researchAssetUrl('research_kernel/model_registry.json'),
  atlas: researchAssetUrl('pattern_family_neuron_atlas/v1/manifest.json'),
  kernel: researchAssetUrl('research_kernel/manifest.json'),
  progress: researchAssetUrl('research_kernel/progress.json'),
};

let sharedEvidenceRequest = null;

async function fetchJson(path) {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${path} ${response.status}`);
  return response.json();
}

function loadEvidence() {
  if (!sharedEvidenceRequest) {
    sharedEvidenceRequest = Promise.allSettled(
      Object.entries(SOURCES).map(async ([id, path]) => [id, await fetchJson(path)])
    ).then((results) => {
      const data = {};
      const errors = [];
      results.forEach((result) => {
        if (result.status === 'fulfilled') {
          const [id, value] = result.value;
          data[id] = value;
        } else {
          errors.push(result.reason?.message || 'unknown evidence source error');
        }
      });
      return { data, errors };
    });
  }
  return sharedEvidenceRequest;
}

export function useModelSystemEvidence() {
  const [state, setState] = useState({ loading: true, data: {}, errors: [] });

  useEffect(() => {
    let mounted = true;
    loadEvidence().then((result) => {
      if (mounted) setState({ loading: false, ...result });
    });
    return () => { mounted = false; };
  }, []);

  return state;
}

export function summarizeModelEvidence(atlas, modelId) {
  const partitions = (atlas?.partitions || []).filter((row) => row.model === modelId);
  return {
    familyCount: new Set(partitions.map((row) => row.family_id)).size,
    componentEvents: partitions.reduce((sum, row) => sum + Number(row.phase330_component_event_count || 0), 0),
    pathSignatures: partitions.reduce((sum, row) => sum + Number(row.phase330_path_signature_count || 0), 0),
    unitCandidates: partitions.reduce((sum, row) => sum + Number(row.unique_unit_count || 0), 0),
    localReadoutCandidates: partitions.reduce((sum, row) => sum + Number(row.phase330_local_set_readout_specific_mechanism_count || 0), 0),
    localPropagationPasses: partitions.reduce((sum, row) => sum + Number(row.phase334_local_gate_pass_count || 0), 0),
    singleUnitCausal: partitions.reduce((sum, row) => sum + Number(row.single_unit_causal_count || 0), 0),
    completeChains: partitions.reduce((sum, row) => sum + Number(row.full_natural_chain_pass_count || 0), 0),
  };
}

export { SOURCES as MODEL_SYSTEM_SOURCES };
