import { useEffect, useState } from 'react';
import { authJson } from './api.js';
import { useAuth } from './authContext.js';
import { Empty } from './ui.jsx';

export default function ManagedNodes(){
  const {user,guest}=useAuth();const loggedIn=!!user&&!guest;const [snapshot,setSnapshot]=useState(null),[error,setError]=useState(''),[busy,setBusy]=useState(false);
  const data=loggedIn&&snapshot?.owner===user?.user_id?snapshot.data:null;
  const setData=data=>setSnapshot({owner:user?.user_id,data});
  const [now,setNow]=useState(()=>Date.now()/1000);
  useEffect(()=>{const timer=setInterval(()=>setNow(Date.now()/1000),5000);return()=>clearInterval(timer)},[]);
  useEffect(()=>{if(!loggedIn)return undefined;let dead=false;const load=()=>authJson('/api/research-nodes').then(d=>{if(!dead){setSnapshot({owner:user.user_id,data:d});setError('')}}).catch(e=>{if(!dead)setError(e.message)});load();const t=setInterval(load,10000);return()=>{dead=true;clearInterval(t)}},[user,loggedIn]);
  async function command(id,action){setBusy(true);try{await authJson(`/api/research-nodes/${encodeURIComponent(id)}/commands`,{method:'POST',body:{action}});setData(await authJson('/api/research-nodes'));setError('命令已排队，等待节点确认；未确认前不视为执行完成')}catch(e){setError(e.message)}finally{setBusy(false)}}
  return <section className="v8-card"><h3>本人账号的远程托管节点</h3><p className="v8-note">须在节点本机选择“远程托管”并开始贡献后授权。只能暂停、恢复、停止；本机暂停/停止立即撤销授权，远端无法重新启动。</p>{!loggedIn?<Empty>登录后查看本人绑定节点。</Empty>:data?.nodes.length?<table className="v6-table"><thead><tr><th>节点</th><th>真实心跳状态</th><th>远程操作</th><th>最近回执</th></tr></thead><tbody>{data.nodes.map(n=>{const enabled=!!n.control?.grant_id&&n.control.updated>now-30;return <tr key={n.node_id}><td>{n.name||n.node_id}</td><td>{n.control?.state||'未授权'} · {enabled?'托管中':'离线/无本机授权'}</td><td>{[['pause','暂停'],['resume','恢复'],['stop','停止']].map(([a,t])=><button key={a} className="fw-tbtn" disabled={!enabled||busy} onClick={()=>command(n.node_id,a)}>{t}</button>)}</td><td>{n.history.slice(0,3).map(c=><div key={c.id}>{c.action} · {c.status}</div>)}</td></tr>})}</tbody></table>:<Empty>{data?'尚无本人绑定的节点。':'读取节点…'}</Empty>}{loggedIn&&error&&<p className="v6-note">{error}</p>}</section>;
}
