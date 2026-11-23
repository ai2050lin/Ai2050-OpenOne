import { useEffect, useState } from 'react';
import { authJson, getSession, useApi } from '../app/rdc_fusion/api.js';
import { useAuth } from '../app/rdc_fusion/authContext.js';
import { nativeAddress } from './workspaceRecordReplay.js';

const base = '/api/research/workspace';

function usePrivate(path) {
  const {user} = useAuth();
  const [result,setResult] = useState(null);
  const [revision,setRevision] = useState(0);
  const account = user?.role==='admin' ? getSession() : '';
  useEffect(()=>{
    if(!path||!account)return undefined;
    let current=true;
    authJson(base+'/private'+path,{redirect:'error'}).then(data=>{
      if(current)setResult({path,account,data});
    }).catch(error=>{if(current)setResult({path,account,error:error.message});});
    return ()=>{current=false;};
  },[path,account,revision]);
  return {...(account&&result?.path===path&&result.account===account?result:null),reload:()=>setRevision(value=>value+1)};
}

export function useWorkspaceCatalog() {
  const publicIndex = useApi(base+'/catalog');
  const privateIndex = usePrivate('/catalog');
  const catalog = publicIndex.data?.schema==='workspace_catalog.v1' ? publicIndex.data : null;
  return {...publicIndex,catalog,archives:[...(catalog?.archives||[]),...(privateIndex.data?.archives||[])],
    reload:()=>{publicIndex.reload();privateIndex.reload();}};
}

export function useWorkspaceResource(id) {
  return useApi(id ? base+'/resources/'+encodeURIComponent(id) : null);
}

export function archiveShortcuts(archives,model) {
  return archives.filter(entry=>entry.snapshot===model?.model_snapshot)
    .flatMap(entry=>(entry.shortcuts||[entry.location]).filter(location=>location&&location.layer<=model.n_layers &&
      location.index<(location.module==='attn'?model.n_heads:model.d_ff))
      .map(location=>({archive:entry,label:nativeAddress(location),location})));
}

export function observationRow(data,key) {
  return data.rows?.find(item=>item.key===(key||data.defaultRow)) ?? data.rows?.[0] ?? null;
}

export function useObservation(entry,key,block) {
  const endpoint = entry ? '/archives/'+encodeURIComponent(entry.id) : null;
  const isPrivate = entry?.visibility==='private';
  const revision=entry?.revision ? 'revision='+encodeURIComponent(entry.revision) : '';
  const publicMeta = useApi(endpoint&&!isPrivate ? base+endpoint+(revision?'?'+revision:'') : null);
  const privateMeta = usePrivate(isPrivate ? endpoint : null);
  const meta = isPrivate ? privateMeta : publicMeta;
  const doc = meta.data;
  const matches = doc?.schema==='workspace_observation.v1'&&doc.kind==='real'&&doc.id===entry?.id&&
    doc.snapshot===entry.snapshot&&doc.run===entry.run;
  const row = matches ? observationRow(doc,key) : null;
  const fieldEndpoint = row?.blocks?.[block] ? endpoint+'/blocks/'+block+'?row='+encodeURIComponent(row.key) : null;
  const publicField = useApi(fieldEndpoint&&!isPrivate ? base+fieldEndpoint+(revision?'&'+revision:'') : null);
  const privateField = usePrivate(isPrivate ? fieldEndpoint : null);
  const field = isPrivate ? privateField : publicField;
  const values = field.data;
  const bound = values?.schema==='workspace_observation_block.v1'&&values.kind==='real'&&values.id===doc?.id&&
    values.snapshot===doc.snapshot&&values.run===doc.run&&values.row===row?.key&&values.block===block&&
    values.position===row.position&&values.tokenId===row.tokenId;
  return {data:matches?doc:null,row,field:bound?values:null,
    error:meta.error||(doc&&!matches?'归档运行或模型快照不符。':''),
    fieldError:field.error||(values&&!bound?'原场回执与当前模型、对象或 token 不符。':''),
    pending:Boolean(fieldEndpoint&&!values&&!field.error),local:meta.local||field.local};
}

function decodeBf16(bits,offset,count) {
  const raw = atob(bits);
  if(raw.length%2 || (offset+count)*2>raw.length)return null;
  const values = new Float32Array(count), words = new Uint32Array(values.buffer);
  for(let i=0;i<count;i++){
    const byte=(offset+i)*2;
    words[i]=(raw.charCodeAt(byte) | raw.charCodeAt(byte+1)<<8)<<16;
  }
  return values;
}

export function readObservationArchive({data,row,field,fieldError,pending,measure,head,headCount,kvHeadCount,headWidth,size,wholeHeads}) {
  if(!row)return {missing:'归档没有可用的读出位置。'};
  const source = [data.run,row.doc||row.caseId||data.caseId||row.key,'token '+row.position+'（ID '+row.tokenId+'）'].join(' · ');
  if(fieldError||pending)return {source,missing:fieldError||'正在从中心读取此块的原场…'};
  const bits=field?.fields?.[measure];
  if(!bits)return {source,missing:'此块或测量量未随该归档发布。'};
  const partial=field.partialHead;
  if(partial!=null){
    const missing='本题只保存 H'+partial+' 的 Z，其余 Head 未采集。';
    if(!Number.isInteger(partial)||partial<0||partial>=headCount||(!wholeHeads&&head!==partial)||
       bits.length!==4*Math.ceil(headWidth*2/3))return {source,missing};
    return {source,values:decodeBf16(bits,0,headWidth),partialHead:partial,missing,sha:field.sha256};
  }
  const shared=measure==='head_k'||measure==='head_v';
  const group=shared?Math.floor(head/(headCount/kvHeadCount)):head;
  const isHead=measure.startsWith('head_');
  const fieldWidth=isHead?headWidth*(shared?kvHeadCount:headCount):size;
  const offset=isHead&&!wholeHeads?group*headWidth:0;
  const values=bits.length===4*Math.ceil(fieldWidth*2/3)?decodeBf16(bits,offset,size):null;
  return {source:source+(shared?' · 共享 KV 头 '+group+' · 来源位置':''),values,sha:field.sha256};
}
