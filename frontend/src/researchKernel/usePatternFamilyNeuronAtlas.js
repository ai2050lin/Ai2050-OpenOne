import { useEffect, useMemo, useState } from 'react';

import { researchAssetUrl } from '../config/researchAssets';

const BASE = researchAssetUrl('pattern_family_neuron_atlas/v1');

const MODEL_KEYS = {
  'qwen3-4b': 'qwen3',
  'glm4-9b': 'glm4',
  ds7b: 'deepseek7b',
};

async function fetchJson(path) {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${path} ${response.status}`);
  return response.json();
}

export function usePatternFamilyNeuronAtlas(familyId, modelKey) {
  const model = MODEL_KEYS[modelKey] || modelKey || 'qwen3';
  const [indexState, setIndexState] = useState({ status: 'loading', manifest: null, familyIndex: null, error: '' });
  const [partitionState, setPartitionState] = useState({ key: '', payload: null, error: '' });

  useEffect(() => {
    let active = true;
    Promise.all([
      fetchJson(`${BASE}/manifest.json`),
      fetchJson(`${BASE}/families.json`),
    ])
      .then(([nextManifest, nextFamilies]) => {
        if (!active) return;
        setIndexState({ status: 'ready', manifest: nextManifest, familyIndex: nextFamilies, error: '' });
      })
      .catch((nextError) => {
        if (!active) return;
        setIndexState({
          status: 'error',
          manifest: null,
          familyIndex: null,
          error: nextError?.message || 'pattern-family neuron atlas unavailable',
        });
      });
    return () => { active = false; };
  }, []);

  const { manifest, familyIndex } = indexState;
  const families = useMemo(() => familyIndex?.families || [], [familyIndex]);
  const family = useMemo(
    () => families.find((item) => item.family_id === familyId) || families[0] || null,
    [families, familyId]
  );
  const partitionRef = useMemo(
    () => (manifest?.partitions || []).find((item) => item.family_id === family?.family_id && item.model === model) || null,
    [family?.family_id, manifest?.partitions, model]
  );
  const partitionKey = partitionRef ? `${partitionRef.family_id}:${partitionRef.model}:${partitionRef.path}` : '';

  useEffect(() => {
    let active = true;
    if (!manifest || !family) return () => { active = false; };
    if (!partitionRef) return () => { active = false; };

    fetchJson(`${BASE}/${partitionRef.path}`)
      .then((payload) => {
        if (!active) return;
        setPartitionState({ key: partitionKey, payload, error: '' });
      })
      .catch((nextError) => {
        if (!active) return;
        setPartitionState({
          key: partitionKey,
          payload: null,
          error: nextError?.message || 'neuron atlas partition unavailable',
        });
      });
    return () => { active = false; };
  }, [family, manifest, model, partitionKey, partitionRef]);

  const partition = partitionRef && partitionState.key === partitionKey ? partitionState.payload : null;
  const partitionLoading = Boolean(partitionRef && partitionState.key !== partitionKey);
  const partitionError = partitionState.key === partitionKey ? partitionState.error : '';

  return {
    manifest,
    families,
    family,
    model,
    partition,
    partitionRef,
    mapped: Boolean(partitionRef),
    loading: indexState.status === 'loading' || partitionLoading,
    error: indexState.error || partitionError,
    evidenceBoundary: partition?.evidence_boundary || manifest?.evidence_boundary?.statement || '',
  };
}

export { MODEL_KEYS as PATTERN_ATLAS_MODEL_KEYS };
