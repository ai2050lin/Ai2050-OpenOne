/* Native parameter regions; block indices, units and scalar parameters are distinct. */
export function preferredParameter(options,kind){
  const suffix=kind==='head'?'.q_proj.weight':kind==='mlp'?'.gate_proj.weight':'.embed_tokens.weight';
  return options.find(t=>t.name.endsWith(suffix))||options.find(t=>t.id?.endsWith('gate_up_proj.weight:gate'))||options[0];
}

export function parameterRegion(config,kind,index,name){
  const d=config.hidden_size,h=config.head_dim||d/config.num_attention_heads,kv=Math.floor(index/(config.num_attention_heads/config.num_key_value_heads));
  if(kind==='embedding')return {row:0,col:0,rows:config.vocab_size,cols:d};
  if(kind==='head'){
    if(name.includes('_norm'))return {row:0,col:0,rows:h,cols:1,shared:true};
    const group=name.includes('k_proj')||name.includes('v_proj')?kv:index;
    if(name.endsWith('.bias'))return {row:group*h,col:0,rows:h,cols:1,shared:name.includes('k_proj')||name.includes('v_proj')};
    if(name.includes('o_proj'))return {row:0,col:index*h,rows:d,cols:h};
    return {row:group*h,col:0,rows:h,cols:d,shared:name.includes('k_proj')||name.includes('v_proj')};
  }
  if(name.includes('down_proj'))return {row:0,col:index,rows:d,cols:1};
  return {row:index+(name.includes('gate_up_proj')&&name.endsWith(':up')?config.intermediate_size:0),col:0,rows:1,cols:d};
}

