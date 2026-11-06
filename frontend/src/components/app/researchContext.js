import {useSyncExternalStore} from 'react';
const KEYS=['run','snapshot','trace','context_trace','pmodel'];
export function publishResearchContext(patch){
  const url=new URL(window.location.href);let changed=false;
  for(const k of KEYS){if(!(k in patch))continue;const value=patch[k];if((url.searchParams.get(k)||'')===(value||''))continue;
    changed=true;if(value)url.searchParams.set(k,value);else url.searchParams.delete(k);
  }
  if(changed){window.history.replaceState(null,'',url);window.dispatchEvent(new Event('research-context'));}
}
export function useResearchContext(){
  const search=useSyncExternalStore(cb=>{window.addEventListener('research-context',cb);window.addEventListener('popstate',cb);return()=>{window.removeEventListener('research-context',cb);window.removeEventListener('popstate',cb)}},()=>window.location.search,()=> '');
  const q=new URLSearchParams(search);return Object.fromEntries(KEYS.map(k=>[k,q.get(k)||'']));
}
