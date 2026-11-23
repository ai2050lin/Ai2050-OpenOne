const tabs=[['overview','总览'],['database','测试数据库']];

export function OverviewTabs({value,onChange}){
  function navigate(event,index){
    const targets={ArrowLeft:1-index,ArrowRight:1-index,Home:0,End:1};
    if(!Object.hasOwn(targets,event.key))return;
    event.preventDefault();
    const target=targets[event.key];
    event.currentTarget.parentElement.children[target].focus();
    onChange(tabs[target][0]);
  }
  return <div className="wb-subnav" role="tablist" aria-label="总览视图">
    {tabs.map(([id,label],index)=><button key={id} id={'overview-tab-'+id} type="button"
      role="tab" className={id===value?'on':''} aria-selected={id===value} tabIndex={id===value?0:-1}
      aria-controls={'overview-panel-'+id} onKeyDown={event=>navigate(event,index)} onClick={()=>onChange(id)}>{label}</button>)}
  </div>;
}

export function ResearchOverview(){
  const {experimentSources,managementRuns}=window.__evidenceData;
  const models=[...new Set(experimentSources.map(s=>s.model))];
  return <section className="rv-summary" aria-labelledby="rv-title">
    <header><h1 id="rv-title">研究总览</h1><span>本地研究快照</span></header>
    <p>我们在研究语言模型怎样表示知识、推理与语法，以及哪些注意力头和神经元参与了这些计算。</p>
    <dl className="rv-counts"><div><dt>收录测试</dt><dd>{experimentSources.length} 项</dd></div>
      <div><dt>涉及模型</dt><dd>{models.length} 种</dd></div>
      <div><dt>附中心读回回执</dt><dd>{managementRuns.length} 项</dd></div></dl>
    <section className="rv-archives"><h2>最近归档的测试</h2>
      <ul>{managementRuns.map(run=><li key={run.run}><button onClick={()=>window.openTestDatabase({q:String(run.phase)},run.phase)}>
        <span><small>Phase {run.phase} · {run.model}</small><strong>{run.title}</strong></span>
        <span className="rv-action">查看记录 →</span></button></li>)}</ul></section>
  </section>;
}
