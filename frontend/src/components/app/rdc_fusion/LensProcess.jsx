/* 过程透镜：AI 研发工作台（OpenScience 式三栏：队列/文件树 · 代码+终端 · 结果卡） */
import { useState } from 'react';

const QUEUE=[
  {id:'Q06',name:'C_steer 基座',pill:'RUNNING',cls:'fw-pill-run',on:true},
  {id:'Q05',name:'E_ar(k) 测量',pill:'SEALED',cls:'fw-pill-done'},
  {id:'Q04',name:'E_ar 装置',pill:'SEALED',cls:'fw-pill-done'},
  {id:'Q03',name:'E_read 基线',pill:'SEALED',cls:'fw-pill-done'},
];
const TERM_LINES=[
  [['$ ','dim'],['python q05_ar_sweep/collect_ar.py --all-arms --K 16','']],
  [['[Q05] ','g'],['D4 bridge max|Δrel| = 0.0489 ',''],['<',''],[' 0.05 ',''],['PASS','g']],
  [['[Q05] ','g'],['shape 4b=',''],['flat','y'],[' · 14B=',''],['saturating','y'],[' · 9B=',''],['flat','y']],
  [['[Q05] ','g'],['S_rel min = 0.6858 / 0.6939 / 0.7392','']],
  [['[review] ','g'],['independent TOTAL PASS=47 FAIL=0 → sealed ',''],['1acb1e78','dim']],
  [['$ ','dim'],['_','']],
];

export default function LensProcess({on,onGo}){
  const [tab,setTab]=useState(0);
  return (
    <section className={'fw-view fw-process'+(on?' on':'')}>
      <div className="fw-pr-left">
        <div className="fw-pr-sec">任务队列</div>
        {QUEUE.map(q=>(
          <button key={q.id} className={'fw-q-row'+(q.on?' on':'')}>
            <span className="id">{q.id}</span>{q.name}<span className={'fw-pill '+q.cls}>{q.pill}</span>
          </button>
        ))}
        <div className="fw-pr-sec">工作区 tests/deepseek/</div>
        <span className="fw-file dir">▾ q05_ar_sweep/</span>
        <span className="fw-file">collect_ar.py</span>
        <span className="fw-file">metric_dict_v4.json</span>
        <span className="fw-file">review_report.txt</span>
        <span className="fw-file dir">▸ q06_c_steer/</span>
        <span className="fw-file dir">▸ shared/</span>
        <div className="fw-pr-sec">运行配置</div>
        <div className="fw-q-row" style={{cursor:'default'}}><span className="id" style={{width:'auto'}}>4b bf16 · 14B/9B NF4</span></div>
        <div className="fw-q-row" style={{cursor:'default'}}><span className="id" style={{width:'auto'}}>drift 断言 · 预注册冻结</span></div>
      </div>

      <div className="fw-pr-mid">
        <div className="fw-ed-tabs">
          {['collect_ar.py','result.json','review_report.txt'].map((t,i)=>(
            <button key={t} className={'fw-ed-tab'+(tab===i?' on':'')} onClick={()=>setTab(i)}>{t}</button>
          ))}
        </div>
        <div className="fw-ed-body">
          {tab===0&&(
            <pre style={{margin:0,font:'inherit'}}>
<span className="cm"># Q05 · E_ar(k) 正式测量 — 四臂 738×K16（节选）</span>{'\n'}
<span className="kw">def</span> <span className="fn">sweep_ar</span>(model, K=<span className="num">16</span>, arms=(<span className="st">&quot;is_a&quot;</span>, <span className="st">&quot;attr&quot;</span>, <span className="st">&quot;syntax&quot;</span>, <span className="st">&quot;rand&quot;</span>)):{'\n'}
{'    '}share = <span className="fn">exact_additive_budget</span>(model){'          '}<span className="cm"># 铁律 (a) 精确可加向量预算</span>{'\n'}
{'    '}<span className="kw">for</span> arm <span className="kw">in</span> arms:{'\n'}
{'        '}E = [share.<span className="fn">write</span>(arm, k=k) <span className="kw">for</span> k <span className="kw">in</span> <span className="fn">range</span>(<span className="num">1</span>, K+<span className="num">1</span>)]{'\n'}
{'        '}rel = (E[<span className="num">0</span>] - E) / E[<span className="num">0</span>]{'                  '}<span className="cm"># S_rel 归一化</span>{'\n'}
{'        '}<span className="kw">yield</span> arm, rel{'\n'}
{'\n'}
<span className="cm"># D4 桥（4b bf16 ↔ 4-bit NF4 口径桥）</span>{'\n'}
<span className="kw">assert</span> <span className="fn">max</span>(<span className="fn">abs</span>(rel_bf16 - rel_nf4)) &lt; <span className="num">0.05</span>{'   '}<span className="cm"># PASS: 0.0489</span>
            </pre>
          )}
          {tab===1&&(
            <pre style={{margin:0,font:'inherit'}}>
{'{'}{'\n'}
{'  '}&quot;phase&quot;: <span className="st">&quot;Q05&quot;</span>, <span className="st">&quot;status&quot;</span>: <span className="st">&quot;measured&quot;</span>,{'\n'}
{'  '}&quot;d4_bridge_max_delta_rel&quot;: <span className="num">0.0489</span>,{'\n'}
{'  '}&quot;shape&quot;: {'{'}<span className="st">&quot;4b&quot;</span>: <span className="st">&quot;flat&quot;</span>, <span className="st">&quot;14b&quot;</span>: <span className="st">&quot;saturating&quot;</span>, <span className="st">&quot;9b&quot;</span>: <span className="st">&quot;flat&quot;</span>{'}'},{'\n'}
{'  '}&quot;s_rel_min&quot;: [<span className="num">0.6858</span>, <span className="num">0.6939</span>, <span className="num">0.7392</span>],{'\n'}
{'  '}&quot;sealed_sha&quot;: <span className="st">&quot;1acb1e78&quot;</span>,{'\n'}
{'  '}&quot;review&quot;: {'{'}<span className="st">&quot;pass&quot;</span>: <span className="num">47</span>, <span className="st">&quot;fail&quot;</span>: <span className="num">0</span>{'}'}{'\n'}
{'}'}
            </pre>
          )}
          {tab===2&&(
            <pre style={{margin:0,font:'inherit',whiteSpace:'pre-wrap'}}>
<span className="cm"># 独立复核结论（节选）</span>{'\n'}
装置门 4/4 通过；份额全程使用精确可加向量预算；{'\n'}
冻结锚逐位复现（drift 0.00e+00）；D4 桥在预注册门内。{'\n'}
未支持结论：k→∞ 外推、跨层迁移、权重级因果证明。
            </pre>
          )}
        </div>
        <div className="fw-term">
          {TERM_LINES.map((line,i)=>(
            <div key={i}>{line.map(([txt,cls],j)=><span key={j} className={cls||''}>{txt}</span>)}</div>
          ))}
        </div>
      </div>

      <div className="fw-pr-right">
        <div className="fw-res-card">
          <h5>E_ar(k) 消融曲线 <span className="fw-pill fw-pill-run" style={{background:'#d1fae5'}}>v4 口径</span></h5>
          <svg viewBox="0 0 260 110" style={{width:'100%',display:'block'}}>
            <line x1="30" y1="95" x2="250" y2="95" stroke="#e2e8f0"/>
            <line x1="30" y1="95" x2="30" y2="8" stroke="#e2e8f0"/>
            <text x="10" y="20" fontSize="8" fill="#94a3b8">S_rel</text>
            <text x="225" y="107" fontSize="8" fill="#94a3b8">k</text>
            <path d="M30 28 L60 29 L90 30 L120 31 L150 32 L180 33 L210 34 L240 35" fill="none" stroke="#0284c7" strokeWidth="2"/>
            <path d="M30 28 L62 36 L94 48 L126 60 L158 71 L190 81 L222 88 L240 90" fill="none" stroke="#6366f1" strokeWidth="2" strokeDasharray="4 3"/>
            <path d="M30 24 L60 25 L90 26 L120 27 L150 28 L180 29 L210 30 L240 31" fill="none" stroke="#10b981" strokeWidth="2"/>
            <text x="150" y="24" fontSize="8.5" fill="#0284c7">4b flat</text>
            <text x="150" y="56" fontSize="8.5" fill="#6366f1">14B saturating</text>
            <text x="150" y="45" fontSize="8.5" fill="#059669">9B flat</text>
          </svg>
          <div className="meta">形状 <b>4b flat / 14B saturating / 9B flat</b><br/>含义：4b 中 is-a 关系沿 k <b>无递减</b> → 关系非碎片化存储</div>
        </div>
        <div className="fw-res-card">
          <h5>跨透镜动作</h5>
          <div className="fw-xact">
            <button className="fw-tbtn" onClick={()=>onGo('spatial')}>在 3D 中查看</button>
            <button className="fw-tbtn" onClick={()=>onGo('progress')}>登记到路线图</button>
          </div>
        </div>
        <div className="fw-res-card">
          <h5>证据链</h5>
          <div className="meta">Q03 E_read <b>0.331615</b>（4b 基线）<br/>Q04 装置 <b>device_built</b> → Q05 <b>measured</b><br/>队列 sealed <b>1acb1e78</b> · 复核 47/0</div>
        </div>
      </div>
    </section>
  );
}
