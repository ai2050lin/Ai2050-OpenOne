/* One bounded JSON entry at a time; completion is required before publishing a cache. */
export async function consumeResearchStream(stream,{id,onEntry=async()=>{},onHeader=async()=>{}}={}){
  const reader=stream.getReader();let buffer='',header=null,footer=null,count=0;
  async function line(text){
    if(!text.trim())return;
    const value=JSON.parse(text);
    if(!header){if(value.schema!=='datapack_extra_stream.v1'||value.id!==id||typeof value.version!=='string')throw Error('不是对应的流式附加包');header=value;await onHeader(value);return}
    if(footer)throw Error('附加包完成标记之后仍有数据');
    if(value.complete===true){if(value.version!==header.version||value.n_entries!==count)throw Error('附加包计数或版本不一致');footer=value;return}
    if(typeof value.path!=='string'||!value.path.startsWith('/api/')||!Object.hasOwn(value,'data'))throw Error('附加包条目无效');
    await onEntry(value.path,value.data);count++;
  }
  try{
    while(true){const {done,value}=await reader.read();if(done)break;buffer+=value;
      let pos;while((pos=buffer.indexOf('\n'))>=0){const text=buffer.slice(0,pos);buffer=buffer.slice(pos+1);await line(text)}
      if(buffer.length>32*1024*1024)throw Error('单条附加包数据过大');
    }
    if(buffer.trim())await line(buffer);
    if(!header||!footer)throw Error('附加包不完整，继续使用原有版本');
    return {header,count};
  }catch(error){await reader.cancel().catch(()=>{});throw error}finally{reader.releaseLock()}
}
