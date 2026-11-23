// Check the identity and axes before displaying a native field for either view.
export function unpackField(data,{traceId,snapshot,block,position,measure,head,headCount,kvHeadCount,width,wholeHeads}){
  if(data.trace_id!==traceId || data.model_snapshot!==snapshot || data.block!==block ||
     data.position!==position || data.boundary!==measure || data.kind!=='real' ||
     data.source!=='native_BF16_bits' || !Array.isArray(data.values))return null;
  const dim=data.shape,v=data.values;
  if(!Array.isArray(dim))return null;
  const allHeads=data.schema==='mechanism_field.v1' && dim.length===2 &&
    dim[0]===headCount && dim[1]===width && v.length===headCount*width;
  if(wholeHeads)return allHeads?v:null;
  if(measure.startsWith('head_')){
    if(data.schema==='mechanism_head_vector.v1'){
      if(data.head!==head || !Number.isInteger(data.kv_group) || data.kv_group<0 ||
         (kvHeadCount>0 && headCount>0 && data.kv_group!==Math.floor(head/(headCount/kvHeadCount))))return null;
    }else if(measure==='head_z' && allHeads){
      return v.slice(head*width,(head+1)*width);
    }else return null;
  }else if(data.schema!=='mechanism_field.v1')return null;
  return dim.length===1 && dim[0]===width && v.length===width ? v : null;
}
