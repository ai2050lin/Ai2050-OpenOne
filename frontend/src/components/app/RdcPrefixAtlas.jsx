import { useEffect, useRef, useState } from 'react';
import './RdcPrefixAtlas.css';

const API='http://127.0.0.1:5001/api/rdc-prefix';
const number=x=>x===null||x===undefined?'—':Number(x).toPrecision(5);
async function json(path){const r=await fetch(API+path);if(!r.ok){let e;try{e=await r.json();}catch{e={detail:r.statusText};}throw new Error(e.detail||r.statusText);}return r.json();}

function NativeField({data,onPick,testId='prefix-field'}){
  const canvas=useRef(null);
  const [colorMode,setColorMode]=useState('signed-log');
  useEffect(()=>{
    if(!data?.values?.length)return;
    const rows=data.values.length,cols=data.values[0].length,el=canvas.current;el.width=cols;el.height=rows;
    const ctx=el.getContext('2d'),img=ctx.createImageData(cols,rows),limit=data.whole_field_absmax||1;
    for(let y=0;y<rows;y++)for(let x=0;x<cols;x++){const raw=data.values[y][x],v=colorMode==='linear'?Math.max(-1,Math.min(1,raw/limit)):Math.sign(raw)*Math.log1p(Math.abs(raw)/.05)/Math.log1p(limit/.05),a=Math.abs(v),i=(y*cols+x)*4;img.data[i]=Math.round(v>=0?244:244-208*a);img.data[i+1]=Math.round(246-153*a);img.data[i+2]=Math.round(v<0?247:247-210*a);img.data[i+3]=255;}
    ctx.putImageData(img,0,0);
  },[data,colorMode]);
  if(!data)return <p>等待可核查数据…</p>;
  return <div className="prefix-field"><p>{data.normalization} · 色标 ±{number(data.whole_field_absmax)} · 全宽 {data.native_width}，当前 [{data.start}, {data.end})</p><label>颜色映射 <select aria-label={testId==='source-field'?'来源颜色映射':'前缀颜色映射'} value={colorMode} onChange={e=>setColorMode(e.target.value)}><option value="signed-log">带符号log1p（尺度0.05，保留低幅值）</option><option value="linear">线性</option></select></label>
    <canvas data-testid={testId} ref={canvas} style={{height:Math.max(150,Math.min(620,data.values.length*12))}} onClick={e=>{const b=e.currentTarget.getBoundingClientRect(),x=Math.min(data.values[0].length-1,Math.floor((e.clientX-b.left)/b.width*data.values[0].length)),y=Math.min(data.values.length-1,Math.floor((e.clientY-b.top)/b.height*data.values.length));onPick({row:data.labels[y],coordinate:data.start+x,value:data.values[y][x]});}}/>
    <p className="prefix-muted">横轴原生编号递增；纵轴{data.labels.length}行：{data.labels[0]} → {data.labels.at(-1)}。缩览会重采样像素；用坐标切片查原值。</p>
    {data.download&&<a href={'http://127.0.0.1:5001'+data.download}>下载完整原数组（BF16位保真 NPZ）</a>}
  </div>;
}

function SourceHistoryStudy(){
  const [summary,setSummary]=useState(null),[scope,setScope]=useState('main'),[rows,setRows]=useState([]),[sample,setSample]=useState(''),[loadedScope,setLoadedScope]=useState('');
  const [mode,setMode]=useState('field'),[rule,setRule]=useState('relative_history'),[anchor,setAnchor]=useState(0),[start,setStart]=useState(0),[count,setCount]=useState(2560),[normalized,setNormalized]=useState(false);
  const [data,setData]=useState(null),[picked,setPicked]=useState(null),[error,setError]=useState('');
  const requestKey=JSON.stringify([scope,sample,mode,rule,anchor,start,count,normalized]);
  useEffect(()=>{let active=true;json('/history/overview').then(d=>{if(active)setSummary(d);}).catch(e=>{if(active)setError(e.message);});return()=>{active=false;};},[]);
  useEffect(()=>{let active=true;json('/history/samples?scope='+scope).then(d=>{if(active){setRows(d);setSample(d[0]?.sample_id||'');setLoadedScope(scope);}}).catch(e=>{if(active)setError(e.message);});return()=>{active=false;};},[scope]);
  useEffect(()=>{if(!sample||loadedScope!==scope)return;let active=true;json(`/history/${mode}?${new URLSearchParams({scope,sample,rule,anchor,start,count,normalized})}`).then(d=>{if(active){setData({...d,requestKey});setPicked(null);setError('');}}).catch(e=>{if(active){setData(null);setError(e.message);}});return()=>{active=false;};},[scope,sample,loadedScope,mode,rule,anchor,start,count,normalized,requestKey]);
  const current=data?.requestKey===requestKey?data:null;
  return <section data-testid="source-history-study"><small>自动续研 · PHASE 2714</small><h2>全部来源H12：平均、绝对位置与相对位置</h2>
    <p>主库512句的每个token均可查询H12；另外64条新来源在规则冻结后才采集。只预测H36，选模目标与上方2712的三层联合预测不同。</p>
    <p>事前验证选择：<strong>{summary?.selected_before_fresh||'尚未冻结'}</strong>。所有规则只使用查询位置之前及当前位置H12，不使用待预测H36或其范数。</p>
    <div className="prefix-table"><table data-testid="source-metrics"><thead><tr>{['材料','规则','H36 MSE','完整词表 KL','argmax一致'].map(x=><th key={x}>{x}</th>)}</tr></thead><tbody>{summary?.metrics.map(r=><tr key={r.scope+r.rule}><td>{r.scope}</td><td>{r.rule}</td><td>{number(r.MSE)}</td><td>{number(r.KL)}</td><td>{number(r.argmax_agreement)}</td></tr>)}</tbody></table></div>
    <div className="prefix-controls"><label>来源集<select aria-label="来源集" value={scope} onChange={e=>{setScope(e.target.value);setMode('field');setStart(0);setCount(2560);}}><option value="main">原主库512句</option><option value="fresh">新冻结确认64句</option></select></label>
      <label className="prefix-wide">来源样本<select aria-label="来源样本" value={sample} onChange={e=>setSample(e.target.value)}>{rows.map(r=><option key={r.sample_id} value={r.sample_id}>{r.sample_id} · {r.split} · {r.genre} · {r.text.slice(0,50)}</option>)}</select></label>
      <button onClick={()=>setMode('field')}>全部来源H12原场</button><button onClick={()=>{setMode('prediction');setNormalized(false);const r=rows.find(r=>['test','confirmation'].includes(r.split));if(r)setSample(r.sample_id);}}>来源规则冻结预测</button><button onClick={()=>{setMode('relations');setNormalized(false);}}>关系条件全坐标</button>
      {mode==='prediction'?<><label>来源规则<select aria-label="来源规则" value={rule} onChange={e=>setRule(e.target.value)}>{['current','mean_history','absolute_history','relative_history'].map(x=><option key={x}>{x}</option>)}</select></label><label>锚点<select aria-label="来源锚点" value={anchor} onChange={e=>setAnchor(Number(e.target.value))}><option value="0">第一锚点</option><option value="1">第二锚点</option></select></label></>:mode==='field'?<label>原场数值<select aria-label="来源归一化" value={String(normalized)} onChange={e=>setNormalized(e.target.value==='true')}><option value="false">原始值</option><option value="true">冻结训练 z-score</option></select></label>:null}
      <label>来源起始坐标<input aria-label="来源起始坐标" type="number" min="0" max="2559" value={start} onChange={e=>setStart(Number(e.target.value))}/></label><label>来源坐标数<input aria-label="来源坐标数" type="number" min="1" max="2560" value={count} onChange={e=>setCount(Number(e.target.value))}/></label>
    </div>
    <p className="prefix-warning">{mode==='relations'?'此图是所选集合按关系聚合的纹理，不是当前单句。标签来自事后UD；新64条在这里是探索性复用。高维余弦bootstrap百分位仅作稳定性参考，不是经校准的显著性证明。':'原场面板是事后完整句的全部位置，含当前锚点之后的位置；预测器严格只使用0..锚点。不要把图中显示的未来位置当作算法可用输入。'}</p>
    <p className="prefix-muted">数字模板审计另发现1条fresh中文句与旧句同构式、仅数字不同；原64条结果保留，去除此类模板的敏感性分析见阶段记录。新来源不等于新构式。</p>
    {error&&<p className="prefix-error" role="alert">{error}</p>}
    <NativeField data={current} onPick={setPicked} testId="source-field"/>{current?.axes&&<p data-testid="source-axes">{current.axes}</p>}{picked&&current&&<output data-testid="source-picked">{picked.row} / 坐标{picked.coordinate} = {number(picked.value)}</output>}
    {current?.available_inputs&&<p data-testid="source-prediction">{current.available_inputs} · MSE {number(current.mse)} · {current.status}</p>}
    <details><summary>来源对齐公式、冻结与适用范围</summary><pre>{JSON.stringify(summary,null,2)}</pre></details>
  </section>;
}

export default function RdcPrefixAtlas(){
  const [overview,setOverview]=useState(null),[samples,setSamples]=useState([]),[run,setRun]=useState('qwen4'),[sample,setSample]=useState('');
  const [loadedRun,setLoadedRun]=useState('');
  const [language,setLanguage]=useState('all'),[split,setSplit]=useState('all'),[genre,setGenre]=useState('all');
  const [anchor,setAnchor]=useState(0),[view,setView]=useState('layers'),[layer,setLayer]=useState(12),[normalized,setNormalized]=useState(false);
  const [start,setStart]=useState(0),[count,setCount]=useState(2560),[model,setModel]=useState('early_linear');
  const [detail,setDetail]=useState(null),[data,setData]=useState(null),[error,setError]=useState(''),[picked,setPicked]=useState(null);
  const [unit,setUnit]=useState(0),[input,setInput]=useState(0),[output,setOutput]=useState(0),[native,setNative]=useState(null);
  const [matrix,setMatrix]=useState(null),[matrixId,setMatrixId]=useState('adjacent_H12'),[matrixRow,setMatrixRow]=useState(0),[matrixCol,setMatrixCol]=useState(0);
  const requestKey=JSON.stringify([run,sample,anchor,view,layer,normalized,start,count,model]);
  useEffect(()=>{let active=true;json('/overview').then(d=>{if(active)setOverview(d);}).catch(e=>{if(active)setError(e.message);});return()=>{active=false;};},[]);
  useEffect(()=>{let active=true;json(`/runs/${run}/samples`).then(d=>{if(active){setSamples(d);setSample(d[0]?.sample_id||'');setLoadedRun(run);setData(null);setNative(null);}}).catch(e=>{if(active)setError(e.message);});return()=>{active=false;};},[run]);
  useEffect(()=>{if(!sample||loadedRun!==run)return;let active=true;
    json(`/runs/${run}/sample/${sample}?anchor=${anchor}`).then(d=>{if(active)setDetail(d);}).catch(e=>{if(active)setError(e.message);});
    const params=new URLSearchParams({anchor,view,layer,normalized,start,count});
    const path=view==='prediction'?`/prediction/${sample}?${new URLSearchParams({run,anchor:Math.floor(anchor/3),model,start,count:Math.min(count,2560)})}`:`/runs/${run}/field/${sample}?${params}`;
    json(path).then(d=>{if(active){setData({...d,requestKey});setError('');setPicked(null);}}).catch(e=>{if(active){setData(null);setError(e.message);}});return()=>{active=false;};
  },[run,loadedRun,sample,anchor,view,layer,normalized,start,count,model,requestKey]);
  const currentData=data?.requestKey===requestKey?data:null;
  const runtime=overview?.runs.find(r=>r.run===run)?.runtime;
  const filtered=samples.filter(r=>(language==='all'||r.language===language)&&(split==='all'||r.split===split)&&(genre==='all'||r.genre===genre));
  const current=samples.find(r=>r.sample_id===sample);
  const selectView=v=>{setView(v);setStart(0);setNormalized(false);setCount(v==='mlp'?9728:runtime?.width||2560);setNative(null);if(v==='tokens'){const p=filtered.find(r=>r.full_panel)||samples.find(r=>r.full_panel);if(p)setSample(p.sample_id);}if(v==='prediction'){const p=filtered.find(r=>['test','confirmation'].includes(r.split))||samples.find(r=>['test','confirmation'].includes(r.split));if(p)setSample(p.sample_id);}};
  async function inspectNative(){try{setNative(await json(`/native/${sample}?${new URLSearchParams({run,anchor,unit,input_coordinate:input,output_coordinate:output})}`));setError('');}catch(e){setError(e.message);}}
  async function inspectMatrix(){try{setMatrix(await json(`/matrix?${new URLSearchParams({matrix_id:matrixId,row:matrixRow,column:matrixCol,count:64})}`));setError('');}catch(e){setError(e.message);}}
  return <main className="prefix-app">
    <header><div><small>RDC · SHARED NATURAL-PREFIX ATLAS · PHASE 2711–2714</small><h1>共同前缀库 · 全坐标图谱</h1><p>观察结构 → 统一规则 → 冻结新材料检查</p></div><nav><a href="/rdc">既有机制与3D图谱 ↗</a><a href="/">原研究客户端 ↗</a></nav></header>
    <section className="prefix-warning"><strong>当前证据：候选结构，尚未闭合</strong><p>英文网络文本与中文维基自然句；前缀图仅有可见线索与顺序，不等于完整词义、角色、共指或作用域图。逐token是给定文本评分，不是自由生成测试。</p><p>{overview?.coverage}</p><p>UTF-8前缀按实际token解码修正；主要数值描述符与冻结模型未变。随机对照另行修复，不能冒充新的独立确认。</p></section>
    <section><div className="prefix-controls">
      <label>运行模型<select aria-label="前缀运行" value={run} onChange={e=>{setRun(e.target.value);setLanguage('all');setSplit('all');setGenre('all');setView('layers');setStart(0);setNormalized(false);setCount(overview?.runs.find(r=>r.run===e.target.value)?.runtime.width||2560);}}>{overview?.runs.map(r=><option key={r.run} value={r.run}>{r.run} · {r.status.completed??0}/{r.status.total??'?'} · {r.runtime.width??'?'}坐标</option>)}</select></label>
      <label>语言<select aria-label="前缀语言" value={language} onChange={e=>setLanguage(e.target.value)}>{['all','en','zh'].map(x=><option key={x}>{x}</option>)}</select></label>
      <label>划分<select aria-label="前缀划分" value={split} onChange={e=>setSplit(e.target.value)}>{['all','train','validation','test','confirmation'].map(x=><option key={x}>{x}</option>)}</select></label>
      <label>体裁<select aria-label="前缀体裁" value={genre} onChange={e=>setGenre(e.target.value)}>{['all',...new Set(samples.map(r=>r.genre))].map(x=><option key={x}>{x}</option>)}</select></label>
      <label className="prefix-wide">样本（筛选 {filtered.length} 条）<select aria-label="前缀样本" value={sample} onChange={e=>setSample(e.target.value)}>{!filtered.some(r=>r.sample_id===sample)&&current&&<option value={sample}>{sample} · 当前项不在筛选内</option>}{filtered.map(r=><option key={r.sample_id} value={r.sample_id}>{r.sample_id} · {r.genre} · {r.split} · {r.text.slice(0,65)}</option>)}</select></label>
      <label>已保存位置<select aria-label="前缀位置" value={anchor} onChange={e=>setAnchor(Number(e.target.value))}>{current?.positions.map((p,i)=><option key={i} value={i}>锚{Math.floor(i/3)+1}+{i%3} · token {p}</option>)}</select></label>
    </div><p data-testid="prefix-identity">{run} · {runtime?.depth} 层 / {runtime?.width} 坐标 · 非量化 {runtime?.dtype} · {current?.token_count} tokens · {current?.full_panel?'完整token面板已保存':'保存6个位置；其他逐token只保留统计'} · 来源 {detail?.source_group}</p>
      <blockquote data-testid="prefix-visible">{detail?.graph.observed_prefix}</blockquote><p className="prefix-muted">上方仅含当前可见token；� 可表示尚未完成的UTF-8字符，不用未来字符补齐。</p>
      <details><summary>原始全文与token身份（全文是事后材料）</summary><p>{detail?.text}</p><pre>{detail?.prompt_ids.map((id,i)=>`${i}\t${id}\t${detail.tokens[i]}`).join('\n')}</pre></details>
    </section>
    <section><nav className="prefix-tabs">{[['layers','全层原生坐标'],['tokens','完整token面板'],['prediction','冻结预测对照'],['mlp','全部MLP单元']].map(([id,label])=><button key={id} className={view===id?'active':''} disabled={(['prediction','mlp'].includes(id)&&!run.startsWith('qwen4'))||(id==='tokens'&&run!=='qwen4')} onClick={()=>selectView(id)}>{label}</button>)}</nav>
      <div className="prefix-controls">{view==='tokens'&&<label>面板层<input aria-label="面板层" type="number" min="0" max={runtime?.depth} value={layer} onChange={e=>setLayer(Number(e.target.value))}/></label>}
        {['layers','tokens'].includes(view)&&<label>数值<select aria-label="前缀归一化" value={String(normalized)} onChange={e=>setNormalized(e.target.value==='true')}><option value="false">原始BF16数值</option><option value="true">本运行训练token逐坐标 z-score</option></select></label>}
        {view==='prediction'&&<label>冻结规则<select aria-label="前缀预测器" value={model} onChange={e=>setModel(e.target.value)}>{['early_linear','full_linear','full_quadratic','graph_interaction','train_mean','copy_H12','temporal_full_linear'].map(x=><option key={x}>{x}</option>)}</select></label>}
        <label>起始坐标<input aria-label="前缀起始坐标" type="number" min="0" max={(view==='mlp'?9728:runtime?.width||2560)-1} value={start} onChange={e=>setStart(Number(e.target.value))}/></label>
        <label>坐标数<input aria-label="前缀坐标数" type="number" min="1" max="16384" value={count} onChange={e=>setCount(Number(e.target.value))}/></label><button onClick={()=>{setStart(0);setCount(view==='mlp'?9728:runtime?.width||2560);}}>恢复全坐标</button></div>
      {error&&<p role="alert" className="prefix-error">{error}</p>}<NativeField data={currentData} onPick={setPicked}/>{picked&&currentData&&<output data-testid="prefix-picked">{picked.row} / 坐标 {picked.coordinate} = {number(picked.value)}</output>}{currentData?.available_inputs&&<p data-testid="prefix-prediction">{currentData.available_inputs} · MSE {number(currentData.mse)} · {currentData.status}</p>}
    </section>
    <section><h2>可见线索的类型化关系图</h2><p>线只表示观测顺序，不是模型真实连接。未知：词义、语义角色、共指、否定作用域、未完成关系。</p><div className="prefix-events" data-testid="prefix-graph">{detail?.graph.events.map((e,i)=><span key={i} title={`字符 ${e.span.join('–')}`}>{i>0?'→ ':''}<b>{e.type}</b> {e.text}</span>)}</div><details><summary>前缀图全部数值与身份</summary><pre>{JSON.stringify(detail?.graph,null,2)}</pre></details></section>
    <section><h2>同一规则库：全部坐标与完整词表</h2><p>test 192锚点，confirmation 256锚点；同句两个锚点不是两个独立来源。MSE越低越好；KL检查151936个词表项；argmax一致不等于任务正确。</p><div className="prefix-table"><table data-testid="prefix-metrics"><thead><tr>{['材料','规则','H36 MSE','全词表 KL','argmax一致','证据状态'].map(h=><th key={h}>{h}</th>)}</tr></thead><tbody>{overview?.metrics.map(r=><tr key={r.scope+r.model}><td>{r.scope}</td><td>{r.model}</td><td>{number(r.H36_mse)}</td><td>{number(r.KL)}</td><td>{number(r.argmax_agreement)}</td><td>{r.status}</td></tr>)}</tbody></table></div><details><summary>三模型匹配材料（原生坐标不一一对应）</summary><pre data-testid="prefix-scale">{JSON.stringify(overview?.scale,null,2)}</pre></details></section>
    <section><h2>坐标 → MLP单元 → 真实标量参数</h2><p>Qwen4 L23全输入点积、全单元写回总和与BF16残差；使用真实观测因子核对，不是提前预测或语义因果门。</p><div className="prefix-controls">{[['原生单元',unit,setUnit,9727],['输入坐标',input,setInput,2559],['输出坐标',output,setOutput,2559]].map(([label,value,setter,max])=><label key={label}>{label}<input aria-label={'前缀'+label} type="number" min="0" max={max} value={value} onChange={e=>setter(Number(e.target.value))}/></label>)}<button disabled={!run.startsWith('qwen4')} onClick={inspectNative}>核对真实参数路径</button></div>{native&&<div data-testid="prefix-native"><pre>{JSON.stringify({...native,all_unit_contributions:undefined},null,2)}</pre><details><summary>全部9728单元写回贡献</summary><pre>{native.all_unit_contributions.map((v,i)=>`${i}: ${v}`).join('\n')}</pre></details></div>}</section>
    <section><h2>完整坐标对关系矩阵</h2><p>每个矩阵2560×2560项；下方读取64×64原始切片。相邻token n=640；UD依存边仅来自16个完整面板，是事后标注。</p><div className="prefix-controls"><select aria-label="前缀关系矩阵" value={matrixId} onChange={e=>setMatrixId(e.target.value)}>{['adjacent_H0','adjacent_H12','adjacent_H24','adjacent_H36','ud_nsubj_H12','ud_nsubj_H24','ud_obj_H12','ud_obj_H24'].map(x=><option key={x}>{x}</option>)}</select>{[['矩阵行',matrixRow,setMatrixRow],['矩阵列',matrixCol,setMatrixCol]].map(([label,value,setter])=><label key={label}>{label}<input aria-label={label} type="number" min="0" max="2559" value={value} onChange={e=>setter(Number(e.target.value))}/></label>)}<button onClick={inspectMatrix}>读取全矩阵切片</button></div>{matrix&&<details open><summary>{matrix.id} · n={matrix.n} · 起点{matrix.row},{matrix.column}</summary><pre data-testid="prefix-matrix">{JSON.stringify(matrix,null,2)}</pre></details>}</section>
    <section><h2>新的自然关系与状态更新研究</h2><p>查看Phase 2715起的新材料、全跨坐标矩阵、真实block接续及自然/自喂生成。</p><a href="/rdc-relation">打开自然关系研究客户端 →</a></section>
    <SourceHistoryStudy/>
    <section><h2>科学图 · 固定原生坐标顺序</h2>{overview?.figures.map(f=><figure key={f.id}><figcaption>{f.title} · <a href={`${API}/figures/${f.id}`} target="_blank" rel="noreferrer">原图</a> · <a href={`${API}/figures/${f.id}?contract=true`}>显示约定与源数组</a></figcaption><img loading="lazy" src={`${API}/figures/${f.id}`} alt={f.title}/></figure>)}</section>
    <section><h2>审查、理论边界与运行计划</h2><details><summary>附件纠错与依据</summary><pre>{JSON.stringify(overview?.review,null,2)}</pre></details><details><summary>整体大方案、资源与续研决定</summary><pre>{JSON.stringify({plan:overview?.plan,continuation:overview?.continuation},null,2)}</pre></details><p>RDC名称保留。尚无新数学定理；高相似度、参数恒等式、局部预测不能直接推出AGI或脑机制。</p></section>
  </main>;
}
