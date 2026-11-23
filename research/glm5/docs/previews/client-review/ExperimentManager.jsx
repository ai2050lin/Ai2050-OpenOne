import {useRef, useState} from 'react';

const dateFormat = new Intl.DateTimeFormat('zh-CN', {
  timeZone:'America/Chicago', year:'numeric', month:'2-digit', day:'2-digit'
});

function DraftDialog({dialog}) {
  const [question,setQuestion]=useState('');
  const [model,setModel]=useState('Qwen3-14B');
  const [message,setMessage]=useState('');
  function download(e) {
    e.preventDefault();
    if(!question.trim())return;
    const content=['# 实验草稿','',question.trim(),'','拟用模型：'+model,''].join('\n');
    const a=document.createElement('a');
    a.href='data:text/markdown;charset=utf-8,'+encodeURIComponent(content);
    a.download='AI2050-实验草稿.md';
    a.click();
    setMessage('已生成草稿文件，可在浏览器下载列表中查看。');
  }
  return <dialog className="em-dialog" ref={dialog} aria-labelledby="em-draft-title" onClose={()=>setMessage('')}>
    <form onSubmit={download}>
      <header><h2 id="em-draft-title">新建实验草稿</h2><button type="button" onClick={()=>dialog.current.close()}>关闭</button></header>
      <label>想验证什么问题？<textarea rows={3} required value={question} onChange={e=>{setQuestion(e.target.value);setMessage('');}}/></label>
      <label>拟用模型<select value={model} onChange={e=>{setModel(e.target.value);setMessage('');}}>
        <option>Qwen3-14B</option><option>GLM-4-9B</option><option>Qwen3-4B</option></select></label>
      <footer><small>草稿可下载到本机，正式运行需在中心登记。</small><button className="em-primary" disabled={!question.trim()}>下载草稿</button></footer>
      {message&&<p role="status">{message}</p>}
    </form>
  </dialog>;
}

export default function ExperimentManager({onStation}) {
  const dialog=useRef(null);
  const runs=window.__evidenceData.managementRuns;
  return <section className="em-page" aria-labelledby="em-page-title">
    <div className="em-head"><h1 id="em-page-title">实验管理</h1>
      <button className="em-primary" onClick={()=>dialog.current.showModal()}>新建实验草稿</button></div>
    <section className="em-list" aria-labelledby="em-runs-title">
      <header><h2 id="em-runs-title">已归档实验</h2><small>本地快照 · {runs.length} 项</small></header>
      <ul>{runs.map(r=><li key={r.run}>
        <div className="em-name"><h3>{r.title}</h3><small>Phase {r.phase} · {r.model}</small></div>
        <time dateTime={new Date(r.verifiedAt*1000).toISOString()} title="回执核验日期（美国中部时间）">{dateFormat.format(r.verifiedAt*1000)}</time>
        <button className="em-link" aria-label={'查看 Phase '+r.phase+' 结果'} onClick={()=>window.openTestDatabase({q:r.run},r.phase)}>查看结果 →</button>
      </li>)}</ul>
    </section>
    <details className="em-more">
      <summary>更多工具</summary>
      <div className="em-tools">
        <button onClick={()=>onStation('queue')}>队列与审计</button>
        <button onClick={()=>onStation('catalog')}>材料与模板</button>
        <button onClick={()=>onStation('plans')}>研究计划</button>
      </div>
    </details>
    <p className="em-offline">当前为离线快照，实时运行状态需连接研究中心查看。</p>
    <DraftDialog dialog={dialog}/>
  </section>;
}
