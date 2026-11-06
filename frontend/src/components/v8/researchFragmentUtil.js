export const EVIDENCE_LABEL={hypothesis:'未检验假设',observation:'候选观察',heldout:'留出复现',intervention:'范围内干预',parameter_source:'真实参数来源',negative:'未支持 / 负结果'};

export function orderedStages(stages){
  return Object.entries(stages).sort(([,a],[,b])=>(a.time||0)-(b.time||0));
}

export function fragmentHref(fragment){
  const q=new URLSearchParams({lens:'spatial',view:'parameters',artifact:fragment.sha,feature:fragment.id});
  if(fragment.body.model_binding?.local_model_id)q.set('pmodel',fragment.body.model_binding.local_model_id);
  if(window.location.protocol==='file:')q.set('wb','1');
  // A query-only href stays inside app.asar on Windows and retains desktop routing.
  return `?${q}`;
}

export function fragmentLocation(card,cfg){
  if(!cfg||!Number.isInteger(card.layer)||card.layer<0||card.layer>=cfg.num_hidden_layers)return null;
  if(card.boundary==='mlp.down_proj.input'&&card.index?.space==='mlp'&&Number.isInteger(card.index.unit)&&card.index.unit>=0&&card.index.unit<cfg.intermediate_size)
    return {block:card.layer,kind:'mlp',unit:card.index.unit};
  return null;
}

export function snapshotMatches(expected,actual){
  return typeof expected==='string'&&/^[0-9a-f]{64}$/.test(expected)&&actual?.source==='full_file_sha256'&&expected===actual.sha256;
}

export function responseCompatible(response,card,snapshot,cfg){
  return response?.schema==='research_mlp_response.v1'&&response.kind==='real'&&response.model_snapshot===snapshot
    &&response.run_id===card?.run_id&&response.selected_unit===card?.index?.unit
    &&response.block===card?.layer&&response.boundary===card?.boundary&&response.readout===card?.scope?.readout
    &&Array.isArray(response.values)&&response.values.length===cfg?.intermediate_size&&response.values.every(Number.isFinite);
}
