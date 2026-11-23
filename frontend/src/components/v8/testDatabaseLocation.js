export function databaseHref(run,{caseId,fromHead=false,model,fullscreen}={}) {
  const query = new URLSearchParams({lens:'route',rtab:'database',record:run,db:new URLSearchParams({q:run})});
  if(caseId)query.set('case',caseId);
  if(fromHead && caseId){
    query.set('from','head');
    const source=new URLSearchParams(window.location.search);
    const returnModel=model||source.get('mdl');
    if(returnModel)query.set('headModel',returnModel);
    if(fullscreen ?? (source.get('observeFullscreen')==='1'))query.set('headFullscreen','1');
  }
  return window.location.protocol==='file:' ? `?${query}&wb=1` : `/workbench?${query}`;
}

export function readDatabaseLocation() {
  const query = new URLSearchParams(window.location.search);
  const filters = new URLSearchParams(query.get('db') || '');
  const record = query.get('record') || query.get('dbStudy') || '';
  const source = filters.get('source');
  const scope = ['evidence','directions','goodfire','history','center','published'].includes(source) ? source
    : /^(gf-|\d+$)/.test(record) ? 'goodfire' : record ? 'center' : 'evidence';
  return {scope, query:(filters.get('q') || record).slice(0,500), group:filters.get('group') || '',
    kind:filters.get('method') || '', model:filters.get('model') || '', centerFamily:filters.get('family') || '',
    part:filters.get('part') || '', domain:filters.get('domain') || '', state:filters.get('state') || '',
    advanced:Object.fromEntries([...filters].filter(([key])=>key.startsWith('run_')).map(([key,value])=>[key.slice(4),value])),
    record, caseId:(query.get('case')||'').slice(0,120), fromHead:query.get('from')==='head',
    headModel:(query.get('headModel')||'').slice(0,240), headFullscreen:query.get('headFullscreen')==='1', selected:query.getAll('compare')};
}

export function writeDatabaseLocation(filters,record,selected) {
  const url = new URL(window.location.href);
  const encoded = new URLSearchParams(Object.entries(filters).filter(([,value])=>value));
  if(encoded.size)url.searchParams.set('db',encoded);else url.searchParams.delete('db');
  url.searchParams.delete('dbStudy');
  if(url.searchParams.get('record')!==record){
    for(const key of ['case','from','headModel','headFullscreen'])url.searchParams.delete(key);
  }
  if(record)url.searchParams.set('record',record);else url.searchParams.delete('record');
  url.searchParams.delete('compare');
  selected.forEach(id=>url.searchParams.append('compare',id));
  if(url.href!==window.location.href)window.history.replaceState(null,'',url);
}
