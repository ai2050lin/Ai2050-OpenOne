import {useMemo} from 'react';
import {useNative} from './MechanismTraceExplorer.jsx';
import {readObservationArchive} from './nativeObservationArchive.js';
import {unpackField} from './nativeTraceField.js';

export function useSceneHeadField({enabled, trace, snapshot, block, position, width, count, bundle, sourceError}) {
  const traceId = trace?.run.id;
  const captured = trace?.run.readout?.includes(position);
  const path = enabled && !sourceError && traceId && captured && trace.base!=null
    ? `/api/traces/${encodeURIComponent(traceId)}/field?block=${block}&position=${position}&boundary=head_z` : null;
  const lookup = useNative(trace?.base ?? null, path);
  return useMemo(() => {
    if (!enabled) return null;
    if (sourceError) return {missing:sourceError};
    let field, partialHead = null, source = '', missing = '';
    if (traceId) {
      source = `轨迹 ${traceId} · token ${position}`;
      if (!captured) return {source, missing:'此 token 未采集 Head 输出。请选择标有「模块读出」的 token。'};
      if (!path) return {source, missing:'缺少保存该轨迹的采集节点。'};
      if (lookup.error) return {source, missing:`节点未提供此层的 Head 输出。${lookup.error}`};
      if (!lookup.data) return {source, missing:'正在读取当前块的 Head 输出…'};
      field = unpackField(lookup.data,{traceId,snapshot,block,position,measure:'head_z',width,headCount:count,wholeHeads:true});
    } else if (bundle) {
      const data = readObservationArchive({...bundle,block,measure:'head_z',head:0,
        headCount:count,headWidth:width,size:count*width,wholeHeads:true});
      ({source, missing, values:field, partialHead = null} = data);
    } else {
      return {missing:'此记录没有 Head 输出通道。请选择含 Z 原场的轨迹或中心归档。'};
    }
    if (field?.length !== (partialHead==null?count*width:width) || !field.every(Number.isFinite))
      return {source,missing:missing || '返回的轨迹身份、Head 维度或原生数值不符。'};
    const vectors = Array.from({length:count}, (_, head) => partialHead==null
      ? field.slice(head*width,(head+1)*width) : head===partialHead ? field : null);
    const strengths = vectors.map(row => row ? Math.sqrt(row.reduce((sum,z)=>sum+z*z,0)/width) : null);
    return {source,missing,vectors,strengths};
  }, [enabled,traceId,snapshot,block,position,width,count,bundle,sourceError,captured,path,lookup.data,lookup.error]);
}
