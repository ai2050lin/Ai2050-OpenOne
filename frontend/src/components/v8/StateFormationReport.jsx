import {useMemo,useState} from 'react';
import {useApi} from '../app/rdc_fusion/api.js';
import {apiUrl} from '../../config.js';
import {NativeArray} from './MechanismScene.jsx';
import './hiddenstate_components.css';

const pct=value=>Number.isFinite(value)?`${(100*value).toFixed(2)}%`:'未定义';
const num=value=>Number.isFinite(value)?value.toPrecision(4):'未定义';
const factorNames={language:'输出语言',color:'查询颜色',color_language:'颜色×语言交互'};
const methodNames={fixed:'固定条件中心',oracle_center:'提供真实颜色后修正中心',decoded_center:'自行读出颜色后修正中心',pooled:'整体线性读出',norm:'仅向量长度',wrong_center:'反向修正中心',permuted_labels:'打乱语言标签'};
function NativeVector({doc,state,factor,kind}){
  const layer=kind==='state'?state:Math.min(35,Math.max(0,Math.floor((state-1)/2)));
  const digest=doc.coordinate_artifacts?.[kind]?.[layer];
  const resource=useApi(digest?`/api/research-library/artifacts/${digest}`:null);
  const [split,setSplit]=useState(1),[coordinate,setCoordinate]=useState(0);
  const decoded=useMemo(()=>{
    const a=resource.data?.array;if(!a||a.encoding!=='base64'||a.dtype!=='<f4'||a.shape?.slice(1).join(',')!=='3,2560'||![1,2].includes(a.shape[0]))return null;
    try{const bytes=Uint8Array.from(atob(a.data),c=>c.charCodeAt(0));if(bytes.length!==a.shape[0]*3*2560*4)return null;const view=new DataView(bytes.buffer);return {values:Array.from({length:bytes.length/4},(_,i)=>view.getFloat32(i*4,true)),splits:a.shape[0]};}catch{return null;}
  },[resource.data]);
  if(!digest)return <p role="status">此版本没有所选坐标数据，请更新研究包或查看原始证据。</p>;
  if(resource.error)return <p role="status">原生坐标读取失败：{resource.error}</p>;
  if(!resource.data)return <p>正在读取全部原生坐标…</p>;
  if(!decoded||resource.data.model_snapshot!==doc.model_snapshot||resource.data.index!==layer||resource.data.field!==kind)return <p role="status">数据身份或形状不匹配，未显示坐标。</p>;
  const source=decoded.splits===1?0:split,k=['language','color','color_language'].indexOf(factor);
  const vector=decoded.values.slice((source*3+k)*2560,(source*3+k+1)*2560);
  return <><p>答案前位置 · {kind==='state'?doc.curves[state].label:`B${layer} ${kind==='attention'?'注意力':'MLP'}实际写入`} · {factorNames[factor]}</p>
    {decoded.splits===2&&<label>数据 <select aria-label="形成坐标来源" value={split} onChange={e=>setSplit(Number(e.target.value))}><option value="0">发现集拟合</option><option value="1">新确认集实测</option></select></label>}
    <p>原生坐标 {coordinate} = {num(vector[coordinate])}</p><label>坐标 <input type="number" aria-label="形成原生坐标" min="0" max="2559" value={coordinate} onChange={e=>setCoordinate(Math.max(0,Math.min(2559,Number(e.target.value)||0)))}/></label>
    <NativeArray values={vector} selected={coordinate} onPick={setCoordinate} label="状态形成完整系数"/>
    <p className="v8-note">保留全部2560坐标。条件系数为发现或确认家族的正交对照均值，不是单神经元标签；当前页面只展示答案前位置，其余三个位置保存在原生证据中。</p>
  </>;
}

function FormationCurve({rows,selected,onSelect}){
  const x=i=>45+i/73*700,y=v=>220-180*v;
  return <div className="hs-curve-scroll" tabIndex="0" aria-label="注意力与MLP形成曲线，可横向滚动"><svg viewBox="0 0 780 270" className="v8-dose-chart" role="img" aria-label="逐模块状态可读出曲线">
    {[.5,.9,1].map(value=><g key={value}><line x1="45" x2="745" y1={y(value)} y2={y(value)} stroke="#d8e0e8" strokeDasharray="4 3"/><text x="3" y={y(value)+4}>{pct(value)}</text></g>)}
    {[[0,'嵌入'],[20,'B9'],[40,'B19'],[60,'B29'],[72,'B35']].map(([i,label])=><text key={i} x={x(i)} y="248" textAnchor="middle">{label}</text>)}
    {[['language_accuracy','#b08045','语言'],['color_accuracy','#246caf','颜色']].map(([key,color,label])=><g key={key}><polyline points={rows.slice(0,73).map(r=>`${x(r.state_index)},${y(r[key])}`).join(' ')} fill="none" stroke={color} strokeWidth="2"/><line x1={x(72)} x2={x(73)} y1={y(rows[72][key])} y2={y(rows[73][key])} stroke={color} strokeDasharray="3 3"/>{rows.map(r=><circle key={r.state_index} cx={x(r.state_index)} cy={y(r[key])} r={r.state_index===selected?5:2} fill={color} onClick={()=>onSelect(r.state_index)}><title>{label} · {r.label} · {pct(r[key])}</title></circle>)}</g>)}
    <text x="390" y="267" textAnchor="middle">每块依次为注意力后、MLP后；末端虚线为最终归一化</text>
  </svg></div>;
}

export default function StateFormationReport({doc}){
  const [tab,setTab]=useState('center'),[state,setState]=useState(68),[factor,setFactor]=useState('language'),[kind,setKind]=useState('state');
  const [candidate,setCandidate]=useState(doc.formation[0]?.id),[dose,setDose]=useState(1),[method,setMethod]=useState('ablation');
  const selected=doc.formation.find(c=>c.id===candidate),effect=doc.effects.find(e=>e.candidate===candidate&&e.dose===dose&&e.method===method);
  const primary=doc.center_rows.find(r=>r.primary),curve=doc.curves[state];
  return <section className="hs-components-report" aria-label="新样本状态形成与作用报告">
    <p><strong>条件中心与语言、颜色状态怎样形成</strong> · Qwen3-4B</p>
    <p>256个发现家族、768个新确认家族；16种新的记录表述与8种新名词。查询和语言指令的框架保持相同。原生BF16、实际模块边界、原模型干预。当前为单节点结果，尚未独立复算。</p>
    <p>行为门：{doc.behavior.passed?'通过':'未全通过，语义机制主张不晋级'}；条件中心主要检验：{!doc.behavior.passed?'未获准入，不能据此接受或否定机制命题':doc.center_primary_supported?'支持冻结命题':'未支持冻结命题'}。</p>
    {doc.diagnostic_only&&<p role="status"><strong>原确认流程已因行为门失败停止。</strong>以下干预按另行冻结的故障诊断协议执行，保留全部预定样本、失败条件和原阈值；数值可供诊断，不算通过的机制确认。{doc.behavior_errors?.length}条四候选行为错误可在“原始证据与失败”中追溯。</p>}
    <div className="v6-toolbar" role="group" aria-label="状态形成分析内容">{[['center','条件中心检验'],['formation','注意力与MLP形成'],['effects','作用与副作用'],['evidence','原始证据与失败']].map(([key,label])=><button className="fw-tbtn" key={key} aria-pressed={tab===key} onClick={()=>setTab(key)}>{label}</button>)}</div>
    {tab==='center'&&<>
      <p>像换一张地图时需要校准坐标原点：这里检验的是，颜色变化后，语言读出的中心是否也要随之改变。修正量全部来自发现集；确认样本不参与调中心或阈值。</p>
      <p>主要边界 hs[34]（B33后）：自行修正中心的语言准确率 {pct(primary.accuracies.decoded_center)}，固定中心 {pct(primary.accuracies.fixed)}，整体线性读出 {pct(primary.accuracies.pooled)}。</p>
      <table className="v6-table" aria-label="条件中心简单基线比较"><thead><tr><th>读出方法</th><th>新确认准确率</th></tr></thead><tbody>{Object.entries(methodNames).map(([key,name])=><tr key={key}><td>{name}</td><td>{pct(primary.accuracies[key])}</td></tr>)}</tbody></table>
      <p>自行修正相对固定中心的平均增益 {pct(primary.gain.mean)}；按句式聚类的区间 {pct(primary.gain.frame_interval[0])} 至 {pct(primary.gain.frame_interval[1])}（α={num(primary.gain.alpha)}）。比某个迁移读出器更好，不等于比整体线性读出更好。</p>
      <p>负对照限制：这里只有一次标签置乱。把其预测标签事后反转，准确率会变为 {pct(1-primary.accuracies.permuted_labels)}；因此低于机会水平不能证明它完全没有携带语言信息，也不能替代多次按家族置乱的零分布。</p>
      <details><summary>如何选择方向及次要边界结果</summary><p>固定中心和自行修正两个方法都先从当前状态读出颜色，再选择在另一颜色上拟合的语言方向；只有明确标出的“提供真实颜色”基线接收正确颜色。尺度来自完整发现集。hs[33]、hs[35]是次要描述，不能用它们替换失败的主要检验。</p><pre className="v8-artifact-json">{JSON.stringify(doc.center_rows,null,2)}</pre></details>
    </>}
    {tab==='formation'&&<>
      <div className="hs-controls"><label>状态边界 <select aria-label="形成状态边界" value={state} onChange={e=>setState(Number(e.target.value))}>{doc.curves.map(r=><option key={r.state_index} value={r.state_index}>{r.label}</option>)}</select></label>
        <label>成分 <select aria-label="形成成分" value={factor} onChange={e=>setFactor(e.target.value)}>{Object.entries(factorNames).map(([key,name])=><option key={key} value={key}>{name}</option>)}</select></label>
        <label>显示对象 <select aria-label="形成坐标对象" value={kind} onChange={e=>setKind(e.target.value)}><option value="state">状态系数</option><option value="attention">实际注意力写入</option><option value="mlp">实际MLP写入</option></select></label></div>
      <FormationCurve rows={doc.curves} selected={state} onSelect={setState}/><p>● <span style={{color:'#b08045'}}>输出语言</span> · ● <span style={{color:'#246caf'}}>查询颜色</span></p>
      <p>本轮在B0注意力后，语言已可读出 {pct(doc.curves[1].language_accuracy)}。四个候选来自预先限定窗口内的变化筛选，不能被命名为信息首次形成的位置。</p>
      <p>{curve.label}：语言可读出 {pct(curve.language_accuracy)}；颜色可读出 {pct(curve.color_accuracy)}；八条件/四状态预测MSE {num(curve.eight_over_four)}。不同边界的曲线各用对应冻结读出器，不能仅凭曲线增量认定模块作用。</p>
      <table className="v6-table" aria-label="状态预测简单基线"><thead><tr><th>发现集拟合方法</th><th>新样本全状态MSE</th></tr></thead><tbody>{Object.entries(curve.state_prediction_mse).map(([key,value])=><tr key={key}><td>{{constant:'常数均值',additive:'查询/记录/语言加法',four_state:'颜色/语言四状态',eight_cell:'八条件完整表'}[key]}</td><td>{num(value)}</td></tr>)}</tbody></table>
      <p>比值低于1表示八条件更好。四状态及八条件预测接收实验条件标签，尚未从输入文本独立算出这些条件；MSE改善不等于已破解相同比例的语言机制。</p>
      {state===73&&<p>这里经过最终RMSNorm；原始B35输出在前一个边界，已单独保存。</p>}
      <details><summary>完整原生坐标与数值</summary><NativeVector doc={doc} state={state} factor={factor} kind={kind}/></details>
      <details><summary>冻结的四个候选及同一读出器的写入对照</summary>{doc.formation.map(c=><p key={c.id}>{c.id}：同一后边界读出器的有符号对照，由 {num(c.same_post_decoder_signed_contrast_before)} 到 {num(c.same_post_decoder_signed_contrast_after)}。发现期准确率增益门{c.selection_support?'通过':'未过，保留为弱候选'}；作用需看原模型干预。</p>)}</details>
    </>}
    {tab==='effects'&&<>
      <div className="hs-controls"><label>候选 <select aria-label="形成作用候选" value={candidate} onChange={e=>{setCandidate(e.target.value);setMethod('ablation');}}>{doc.formation.map(c=><option key={c.id}>{c.id}</option>)}</select></label>
        <label>实验 <select aria-label="形成作用方法" value={method} onChange={e=>{setMethod(e.target.value);if(e.target.value==='source_exclusion')setDose(1);}}><option value="ablation">移除模块写入</option>{selected?.module==='attention'&&<option value="source_exclusion">屏蔽来源文本</option>}</select></label>
        <label>剂量 <select aria-label="形成作用剂量" value={dose} onChange={e=>setDose(Number(e.target.value))} disabled={method==='source_exclusion'}><option value="0.5">移除一半</option><option value="1">完整移除</option></select></label></div>
      {effect?<>
        <p>768个预先选定的新样本，每家族一个条件。目标为{factorNames[effect.factor]}。范围内作用：{doc.diagnostic_only?'仅诊断，不作确认':effect.scope_use_supported?'支持':'未支持'}；作用量预测：{doc.diagnostic_only?'仅报告误差比，不作确认':effect.effect_prediction_supported?'优于强基线':'未优于强基线'}。</p>
        <table className="v6-table" aria-label="状态形成作用与对照"><tbody>{[['目标logit优势平均下降',num(effect.mean_target_margin_loss)],['对照的平均下降',num(effect.mean_control_margin_loss)],['超过对照的句式区间',effect.matched_excess.frame_interval.map(num).join(' 至 ')],['作用预测/最佳简单基线MSE',num(effect.prediction_to_best_baseline)],['原先正确→错误',`${effect.first_correct_to_wrong} / 768`],['原先错误→正确',`${effect.first_wrong_to_correct} / 768`],['另一因子正确→错误',`${effect.other_factor_correct_to_wrong} / 768`],['完整词表平均KL变化',num(effect.mean_KL)]].map(([a,b])=><tr key={a}><td>{a}</td><td>{b}</td></tr>)}</tbody></table>
        <p>{method==='ablation'?`对照匹配实际BF16残差扰动范数，合格比例 ${pct(effect.control_valid_fraction)}。完整移除一个模块可能同时破坏多种计算，不等于选择性删除语言或颜色。`:'语言候选屏蔽指令区，颜色候选屏蔽当前记录区；对照屏蔽等数量背景token。所有头的注意力会重新归一化，尚未分解具体Head或QK参数。'}</p>
        <p>失败与限制：{effect.failures.length?effect.failures.join('；'):'当前冻结门内未记录失败；仍受单节点、任务范围与模块编辑范围限制。'}</p>
        <p>上表正确→错/错→对采用完整词表首词口径，下方分组表采用四候选口径；BF16并列时两者可能不同。因子优势及“另一因子”正确率以另一维取正确值为条件；KL方向为编辑分布到原始分布。作用量预测使用原始模型四个候选分数和已知实验条件，尚不能仅凭输入预测完整输出。</p>
        {method==='source_exclusion'&&<p>只屏蔽所选块最后位置到源区的直接注意力；其他位置、更早层已经传播的信息仍存在。因此弱效应不能证明该来源整体无用。</p>}
        <details><summary>新旧记录冲突与原本答错样本</summary><p>以下是事后分组诊断，各分组重叠，不是新的确认检验。原本答错组样本很少。</p>
          <table className="v6-table" aria-label="作用失败分组诊断"><thead><tr><th>分组 / 样本数</th><th>目标优势下降</th><th>正确→错</th><th>错→对</th></tr></thead><tbody>{(doc.effect_error_diagnosis||[]).filter(e=>e.candidate===candidate&&e.method===method&&e.dose===dose).map(e=><tr key={e.stratum}><td>{{no_old_mapping:'无旧映射',old_agrees:'新旧一致',old_conflicts:'新旧冲突',base_four_correct:'原本四候选正确',base_four_wrong:'原本四候选错误'}[e.stratum]} / {e.n}</td><td>{num(e.mean_margin_loss)}</td><td>{e.lost}</td><td>{e.rescued}</td></tr>)}</tbody></table>
        </details>
        <details><summary>历史能力题副作用与最差作用样本</summary><pre className="v8-artifact-json">{JSON.stringify(doc.capability[candidate],null,2)}</pre>{effect.counterexamples.map(c=><details key={c.family}><summary>家族{c.family} · 超过对照 {num(c.matched_excess)}</summary><pre className="v8-sample-text">{c.prompt}</pre></details>)}</details>
      </>:<p role="status">当前选择没有作用记录，不用其他实验结果代替。</p>}
    </>}
    {tab==='evidence'&&<>
      <p>原始证据保存在C盘研究存储，D盘原逻辑路径由目录链接保留。新结果与旧档案分开计数；文件迁移和重复发布都不增加样本。</p>
      {doc.diagnostic_only&&<><p>行为门失败：{Object.entries(doc.behavior.cells).filter(([,v])=>v.four_choice_accuracy<.9).map(([k,v])=>`记录句式/历史条件 ${k}：${pct(v.four_choice_accuracy)}`).join('；')}。冻结门是逐句式×历史条件至少90%；不能用总体高分代替。</p>
        <p>事后排查：按冻结的四候选评分规则，{doc.behavior_error_diagnosis?.errors_matching_obsolete_color}条判错的颜色与旧记录相同。其中{doc.behavior_error_diagnosis?.four_vs_first_top1_disagreements?.length}条因BF16同分，在四候选排序与完整词表首词排序中判定不同；并列规则未修改。这是行为吻合，尚未证明具体传播路径。</p>
        <details><summary>全部{doc.behavior_errors.length}条行为错误及原始输入</summary>{doc.behavior_errors.map(e=><details key={e.sample.id}><summary>{e.sample.id} · 颜色优势 {num(e.score.color_margin)}</summary><pre className="v8-sample-text">{e.sample.prompt}</pre><pre className="v8-artifact-json">{JSON.stringify(e.score,null,2)}</pre></details>)}</details></>}
      {doc.counterexamples.map(c=><details key={c.family}><summary>中心修正反例 · 家族{c.family} · 增益 {pct(c.gain)}</summary><pre className="v8-sample-text">{c.prompt}</pre><p>显示该家族第一条件，增益汇总八条件；最差样本不必然表示失败。</p></details>)}
      {Object.entries(doc.artifacts||{}).map(([name,digest])=><p key={name}><a href={apiUrl(`/api/research-library/artifacts/${digest}`)} target="_blank" rel="noreferrer">{name}</a></p>)}
      <details><summary>完整边界、负对照与行为门</summary><pre className="v8-artifact-json">{JSON.stringify({limitations:doc.limitations,negative_controls:doc.negative_controls,behavior:doc.behavior},null,2)}</pre></details>
    </>}
  </section>;
}
