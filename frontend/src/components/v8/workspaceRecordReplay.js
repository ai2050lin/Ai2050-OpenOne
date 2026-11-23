const fullSha = /^[a-f0-9]{64}$/;
const identifier = value => typeof value === 'string' && value.length > 0 && value.length <= 160;
const nonnegative = value => Number.isSafeInteger(value) && value >= 0;

export function nativeAddress({layer,module,index}) {
  if(!nonnegative(layer)||!nonnegative(index))return '';
  if(module==='hs')return `L${layer}/D${index}`;
  if(layer===0)return '';
  if(module==='attn')return `B${layer-1}/H${index}`;
  if(module==='mlp')return `B${layer-1}/U${index}`;
  return '';
}

// Only called on a case file whose bytes and SHA-256 matched its verified receipt.
export function readCaseReplay(row,{run,snapshot,artifact}) {
  const ref=row?.native_trace, caseId=String(row?.case_id??row?.id??'');
  if(!ref || ref.schema!=='native_trace_ref.v1' || ref.run_id!==run || ref.case_id!==caseId ||
     ref.model_snapshot!==snapshot || !fullSha.test(snapshot) || !fullSha.test(artifact) || !fullSha.test(ref.trace_id) ||
     !identifier(caseId) || !identifier(ref.trace_run_id) || !identifier(ref.sample_id) || !nonnegative(ref.position))return null;
  const location={layer:ref.layer,module:ref.module,index:ref.index};
  if(!nativeAddress(location))return null;
  return {run,caseId,snapshot,artifact,trace:ref.trace_id,traceRun:ref.trace_run_id,sample:ref.sample_id,
    position:ref.position,location};
}

export function matchesReplayManifest(manifest,reference) {
  return Boolean(manifest && manifest.kind==='real' && manifest.trace_id===reference.trace &&
    manifest.model_snapshot===reference.snapshot && manifest.run_id===reference.traceRun &&
    manifest.sample_id===reference.sample && Array.isArray(manifest.tokens) && reference.position<manifest.tokens.length);
}
