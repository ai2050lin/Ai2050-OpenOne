export const EVIDENCE_COLORS = { hypothesis:'#94a3b8', observation:'#8b99aa', heldout:'#2d83c4', intervention:'#db7943', parameter_source:'#305a86', negative:'#9c6457' };
export const EVIDENCE_NAMES = { hypothesis:'假设', observation:'观察', heldout:'留出复现', intervention:'范围干预', parameter_source:'参数来源', negative:'负结果' };
const inside=(i,n)=>Number.isInteger(i)&&Number.isInteger(n)&&i>=0&&i<n;
export function researchLocation(object,model) {
  if(object?.type!=='feature'||!model||object.model_snapshot!==model.snapshot)return null;
  const c=object.body||{},i=c.index||{},b=c.layer;
  if(i.space==='embedding'&&inside(i.token_id,model.vocab_size))return {layer:0,module:'embedding',index:i.token_id};
  if(i.space==='residual'&&inside(i.coordinate??i.coord,model.d_model)) {
    if(['embedding','hidden[0]'].includes(c.boundary))return {layer:0,module:'hs',index:i.coordinate??i.coord};
    if(inside(b,model.n_layers))return {layer:b+1,module:'hs',index:i.coordinate??i.coord};
  }
  if(!inside(b,model.n_layers))return null;
  if(i.space==='mlp'&&inside(i.unit,model.d_ff))return {layer:b+1,module:'mlp',index:i.unit};
  if(i.space==='attention'&&inside(i.head,model.n_heads))return {layer:b+1,module:'attn',index:i.head};
  return null;
}
