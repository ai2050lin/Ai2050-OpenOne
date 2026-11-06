import {useState} from 'react';

const pct=x=>Number.isFinite(x)?`${(100*x).toFixed(1)}%`:'—';
const LABEL={base:'原模型',target:'旧方向编辑',random_direction:'随机方向',random_columns:'能量分层随机列',acad:'书面要求',colloq:'口语要求'};

function GenerationField({traces}){
  const [selected,setSelected]=useState(0);const trace=traces?.[selected];
  if(!trace)return <p className="v8-note">尚无实际生成轨迹。</p>;
  const ceiling=Math.max(...traces.flatMap(t=>t.hidden_rms.flat()));
  return <><select aria-label="生成轨迹样本" value={selected} onChange={e=>setSelected(Number(e.target.value))}>{traces.map((t,i)=><option key={t.sample_id+t.condition} value={i}>{t.sample_id} · {LABEL[t.condition]}</option>)}</select>
    <p className="v8-note">横轴为实际生成步，纵轴为 hs[0]–hs[36]；颜色为全部 2560 坐标的 RMS，统一色标 log(1+RMS)，范围 0–{ceiling.toFixed(2)}。最终层在归一化后。没有筛掉低幅值背景。</p>
    <svg viewBox={`0 0 ${trace.hidden_rms.length*8+50} 398`} role="img" aria-label="实际自回归全层响应热力图" style={{width:'100%',maxHeight:420}}>
      {Array.from({length:37},(_,l)=><text key={l} x="0" y={l*10+13} fontSize="8">hs[{l}]</text>)}
      {trace.hidden_rms.map((layers,step)=>layers.map((value,l)=>{const a=Math.log1p(value)/Math.log1p(ceiling||1);return <rect key={`${step}:${l}`} x={40+step*8} y={l*10+5} width="8" height="10" fill={`hsl(210 48% ${98-a*68}%)`}><title>{`步${step} hs[${l}] RMS=${value.toPrecision(5)}；预测token ID ${trace.generated_token_ids[step]}`}</title></rect>}))}
      {trace.hidden_rms.map((_,s)=>s%10===0?<text key={s} x={40+s*8} y="390" fontSize="9">{s}</text>:null)}
    </svg><p className="v8-note">步 0 从完整提示预测首个生成 token。生成分叉后，两种条件使用不同历史；同一步的状态差不等于相同前缀下的因果效应。完整原坐标及 9728 单元数组保存在节点，并可从图谱定位入口查看已登记生成步。</p>
  </>;
}

export default function ExternalAiRegisterReport({doc:d}){
  const [tab,setTab]=useState('result'),[page,setPage]=useState(0);const cap=d.capability.result,errors=d.counterexamples;
  return <section aria-label="双AI长文本研究报告"><p className="v8-note">{d.run_id} · Qwen3-4B / CUDA BF16 · 发现校准，未作为确认性机制结论</p>
    <p><strong>{d.effects.find(e=>e.condition==='target')?.qualified_formal_to_spoken.mean===0?'旧方向未观察到合格自然语域切换':d.baseline_gate?'探索行为门通过':'探索行为门未通过'}</strong> · 探索行为门{d.baseline_gate?'通过':'未通过'} · 两个外部 AI 独立评分 · 原有能力监控{cap.status==='noninferior'?'非劣效通过':'尚未通过'}</p>
    <div className="v8-library-stats"><span><b>{d.n_inputs}</b>等长度长文本输入</span><span><b>{d.n_outputs}</b>真实生成结果</span><span><b>{d.n_families}</b>独立内容家族</span></div>
    <div className="v6-toolbar" role="group" aria-label="长文本报告内容">{[['result','结果与边界'],['review','AI评审与校准'],['field','真实生成轨迹'],['errors',`反例与分歧（${errors.length}）`],['audit','预注册与审计']].map(([k,t])=><button key={k} className="fw-tbtn" aria-pressed={tab===k} onClick={()=>setTab(k)}>{t}</button>)}</div>
    {tab==='result'&&<><p className="v8-note">四领域 × 两句式 × 512/1024/2048 有效 token；同一配对只改变语域要求。此轮检验测量与执行闭环，8 家族不作为大样本推广。保义须两位 AI 均判断保留，分歧保守记为未通过。</p>
      <table className="v6-table"><thead><tr><th>条件 / 要求</th><th>整段语域符合</th><th>核心事实保留</th><th>评分分歧</th></tr></thead><tbody>{d.stats.map(s=><tr key={s.condition+s.requested_register}><td>{LABEL[s.condition]} / {LABEL[s.requested_register]}</td><td>{pct(s.style_accuracy)}（n={s.n}）</td><td>{pct(s.meaning_preserved)}</td><td>{s.disagreements}</td></tr>)}</tbody></table>
      <details><summary>合格书面→口语变化 · 发现集描述</summary><table className="v6-table"><thead><tr><th>编辑</th><th>整段切换且保义</th><th>家族区间</th></tr></thead><tbody>{d.effects.map(s=><tr key={s.condition}><td>{LABEL[s.condition]}</td><td>{pct(s.qualified_formal_to_spoken.mean)}</td><td>{s.qualified_formal_to_spoken.zero_event_family_bound?`家族至少一次事件的单侧95%上界 ${pct(s.qualified_formal_to_spoken.zero_event_family_bound.one_sided_upper95)}`:s.qualified_formal_to_spoken.ci95.map(pct).join(' – ')}</td></tr>)}</tbody></table><p className="v8-note">同家族的句式与长度不计独立样本。零事件时重采样会退化为 [0,0]，因此展示家族至少一次事件的精确单侧上界；不是逐句成功率上界。不能据此声称最小编辑或已找到门控。当前只有 α=6；旧固定候选剂量曲线不能当作自然生成曲线。</p></details>
      <details><summary>能力保持 · 精确编辑核对后复用</summary><p className="v8-note">{cap.n} 道已提供答案的固定题，净正确率损失 {pct(cap.net_loss)}，区间 [{cap.loss_interval.map(pct).join(', ')}]，允许损失 {pct(cap.margin)}。8 个分片共 80 个已编辑 BF16 块均与原能力测试逐字节哈希一致，复用原监控结果。这是重复题池，未新增盲测，也不代表开放式生成能力保持；140 道未验证镜像参考题不入门。</p></details>
      <details><summary>限制与下一阶段</summary>{d.limitations.map(t=><p className="v8-note" key={t}>{t}</p>)}</details></>}
    {tab==='review'&&<><p className="v8-note">{d.rater_models.join(' / ')} 独立评阅 {d.unique_blind_records} 条去重文本；评审只看到输出与核心事实，看不到要求的语域、权重条件和另一位的标签。不同模型可能共享偏差，不等于人工金标准。</p><p>语域一致率 {pct(d.unique_style_agreement)} · 保义一致率 {pct(d.unique_meaning_agreement)}</p><table className="v6-table"><thead><tr><th>评分模型</th><th>语义控制正确率</th><th>已知破坏检出率</th></tr></thead><tbody>{d.semantic_controls.map(x=><tr key={x.rater}><td>{x.rater}</td><td>{pct(x.meaning_accuracy)} / {x.n} 条</td><td>{pct(x.corruption_sensitivity)} / {x.known_corruptions} 条</td></tr>)}</tbody></table><p className="v8-note">对照涵盖负责人替换、时间替换及限制反转，不能代表所有语义或语用错误。逐条理由与评分原件进入数据档案，不强行消除分歧。</p></>}
    {tab==='field'&&<><p className="v8-note">已核对 {d.native_summary.verified_files} 个原场文件，共 {(d.native_summary.bytes/2**30).toFixed(2)} GiB；{d.native_summary.valid_generation_steps} 个有效生成步。37 层 HiddenState，B18–27 各层全部 9728 MLP 单元。提示保存 5 个位置，每个实际生成步均保留；不是事后教师强制回放。</p><GenerationField traces={d.traces}/></>}
    {tab==='errors'&&<><p className="v8-note">这里包含未符合输入语域、保义未通过和两位 AI 分歧，可能重叠；它们不是独立机制失败数量。</p>{errors.slice(page*12,(page+1)*12).map((e,i)=><details key={page*12+i}><summary>{e.sample_id} · {LABEL[e.condition]} · {e.rater_disagreement?'AI分歧':!e.consensus_meaning?'保义未通过':'语域未符合'}</summary><pre className="v8-sample-text">{e.text}</pre>{Object.entries(e.ratings).map(([name,r])=><p className="v8-note" key={name}>{name}: {r.style} / {r.meaning} / 自然性{r.naturalness} · {r.reason}</p>)}</details>)}<div className="v6-toolbar"><button className="fw-tbtn" disabled={!page} onClick={()=>setPage(page-1)}>上一页反例</button><span>{page+1} / {Math.max(1,Math.ceil(errors.length/12))}</span><button className="fw-tbtn" disabled={(page+1)*12>=errors.length} onClick={()=>setPage(page+1)}>下一页反例</button></div></>}
    {tab==='audit'&&<><p className="v8-note">8 个真实任务经正式中心派发、CUDA 节点执行及上传；物理节点数为 1，尚未进行第二节点复算。执行审计不自动升级机制证据。</p><details><summary>预注册与冻结</summary><pre className="v8-artifact-json">{JSON.stringify({preregistration:d.preregistration,freeze:d.freeze},null,2)}</pre></details><details><summary>逐格行为门</summary><pre className="v8-artifact-json">{JSON.stringify(d.baseline_cells,null,2)}</pre></details><details><summary>完整报告与产物哈希</summary><pre className="v8-artifact-json">{JSON.stringify(d,null,2)}</pre></details></>}
  </section>;
}
