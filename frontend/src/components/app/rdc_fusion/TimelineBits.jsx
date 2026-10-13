/* TimelineBits · 泳道与时间线小组件（脉络透镜 / 平台运行面板共用） */
export function Lane({color,label,small,children}){
  return (
    <div className="fw-lane">
      <div className="fw-lane-head"><span className="ln-dot" style={{background:color}}/>{label} <small>{small}</small></div>
      {children}
    </div>
  );
}

export function Timeline({items,tag,onPick}){
  return (
    <div className="fw-tl">
      {items.map(n=>(
        <button key={n.lb} className={'fw-tl-node '+n.st} onClick={()=>onPick?onPick(n):(window.alert(n.lb+'：'+n.tt.replace('\n',' · ')+'（详情接入 '+tag+' 后开放）'))}>
          <div className="fw-tl-dot"/>
          <div className="lb">{n.lb}</div>
          <div className="tt">{n.tt.split('\n').map((t,i)=><span key={i}>{t}<br/></span>)}</div>
        </button>
      ))}
    </div>
  );
}

export function fmtAgo(ts){
  if(!ts) return '—';
  const d = Math.max(0, Math.round(Date.now()/1000 - ts));
  return d<60 ? d+'s 前' : d<3600 ? Math.round(d/60)+'m 前' : Math.round(d/3600)+'h 前';
}
