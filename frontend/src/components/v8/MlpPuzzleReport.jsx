import {useState} from 'react';
import {apiUrl} from '../../config.js';

const num=x=>Number.isFinite(x)?x.toPrecision(4):'未测';
const pct=x=>Number.isFinite(x)?`${(100*x).toFixed(1)}%`:'未测';
const kind={binding:'对象与属性组合',language:'输出语言条件'};
const criterion={token_predictable:'旧档案：词身份可预测',double_negative:'旧档案：负门控与负值',rare_amplitude:'旧档案：稀有强响应',high_variance:'旧档案：高方差',random:'预抽随机对照'};
const loc=(doc,b,u)=>`?lens=spatial&view=parameters&pkind=mlp&pblock=${b}&punit=${u}&snapshot=${doc.model_snapshot}`;

function ResponseGrid({group,data}) {
  if(!data)return <p>响应表未随当前数据包提供；不以零填补。</p>;
  const extent=Math.max(1e-8,...data.observed.flat().map(Math.abs));
  return <div style={{overflowX:'auto'}}><table className="v6-table" aria-label="八条件原生响应数值与热力图"><thead><tr><th>条件 q / r / L</th>{group.units.map(u=><th key={u}>U{u}</th>)}</tr></thead><tbody>{data.observed.map((row,i)=><tr key={i}><td>{data.conditions[i].join(' / ')}</td>{row.map((v,j)=><td key={j} style={{background:`rgba(${v<0?'211,115,66':'54,128,191'},${.08+.45*Math.abs(v)/extent})`}} title={`实际均值 ${v}；预测 ${data.predicted[i][j]}；基线 ${data.baseline[i][j]}`}>{num(v)}</td>)}</tr>)}</tbody></table><p className="v8-note">512 个留出家族的 A 均值，固定原生单元顺序；共同色标 ±{num(extent)}，颜色不代表语义强弱。悬停查看预测与基线，完整数值见下方。L=+1 英文，−1 中文；q=+1 查询第一个对象；r=+1 第一个对象记录为红色。</p><details><summary>原始均值、冻结预测、加法基线</summary><pre className="v8-artifact-json">{JSON.stringify(data,null,2)}</pre></details></div>;
}

export default function MlpPuzzleReport({doc}) {
  const [tab,setTab]=useState('puzzles'),[id,setId]=useState(doc.groups?.[0]?.id||'');
  const group=doc.groups.find(g=>g.id===id)||doc.groups[0];
  const effects=doc.effects.filter(e=>e.group===group?.id);
  const analysis=doc.audit||{};
  return <section aria-label="原生单元拼图完整验证报告">
    <p><strong>原生单元拼图：解释、反例、预测与作用</strong> · Qwen3-4B · B28—35（从 0 编号）</p>
    <p>48 个旧档案代表单元、16 组新候选；512 个未见记录家族 × 8 条件检验响应，256 个家族检验 4 个冻结对象的作用。</p>
    <p className="v8-note">范围：从文本记录绑定对象与颜色，并按指令选择中英文输出。四种未见句式不等于覆盖全部语言；数学中的 q×r 由实验设计提供，未声称模型自行发现了这个公式。</p>
    {analysis.admissibility&&<p className="v8-note">证据状态：候选观察，未晋级正式证据。运行前二值行为功效准入不完整，原始溯源缺项；详见材料与审计。数值判据通过和正式证据准入分别记录。</p>}
    <div className="v6-toolbar" role="group" aria-label="拼图验证内容">{[['puzzles','能保留的拼图'],['effects','作用与副作用'],['errors','反例与失败'],['units','代表单元'],['evidence','材料与审计']].map(([key,label])=><button key={key} className="fw-tbtn" aria-pressed={tab===key} onClick={()=>setTab(key)}>{label}</button>)}</div>
    {['puzzles','errors'].includes(tab)&&<label>候选单元组 <select aria-label="候选单元组" value={group?.id} onChange={e=>setId(e.target.value)}>{doc.groups.map(g=><option key={g.id} value={g.id}>{kind[g.kind]} · B{g.block} · {g.passed?'响应预测达标':'响应预测未达标'}</option>)}</select></label>}
    {tab==='puzzles'&&group&&<>
      <p><strong>{kind[group.kind]} · B{group.block}</strong>：{group.passed?'通过冻结的响应预测判据':'未通过冻结的响应预测判据'}；{effects.length?(effects.some(e=>e.use_supported)?'有范围内作用证据，详见作用页。':'作用检验未达预设门槛。'):'本组未安排原模型作用检验。'}</p>
      <p>误差 / 基线 <b>{num(group.relative_mse)}</b>；优于基线的家族 {pct(group.family_win_rate)}；同层随机组误差比 {num(group.random_group_relative_mse)}。低于 1 表示改善；随机组也能预测时，不能据此宣称本组独有该功能。</p>
      <p>原生定位：{group.units.map(u=><a style={{marginRight:12}} key={u} href={loc(doc,group.block,u)}>B{group.block}/U{u}</a>)}</p>
      <ResponseGrid group={group} data={doc.response_tables?.[group.id]}/>
      {doc.unit_examples?.[group.id]&&<details><summary>一个原生单元如何响应八种条件</summary><p>家族 {doc.unit_examples[group.id].family} · U{doc.unit_examples[group.id].unit}。G = gate 权重行 · 归一化输入，V = up 权重行 · 同一输入，A = SiLU(G) × V，再沿 down 权重列写回。G 为负仍能产生非零响应；负 G 配负 V 也可写出正 A。这里展示实测值，不把符号直接命名为支持或反对答案。</p><table className="v6-table"><thead><tr><th>q / r / L</th><th>G</th><th>V</th><th>A</th></tr></thead><tbody>{doc.unit_examples[group.id].rows.map((r,i)=><tr key={i}><td>{r.query} / {r.record} / {r.language}</td><td>{num(r.G)}</td><td>{num(r.V)}</td><td>{num(r.A)}</td></tr>)}</tbody></table></details>}
      <p>响应近似：Â = β₀ + βq·q + βr·r + βL·L + βqL·qL + βrL·rL + βqr·qr + βqrL·qrL。参数仅由发现材料拟合。这等价于八条件的均值查表，尚未说明模型内部怎样算出这些条件。</p>
      <p>仅凭输入条件预测下一词分布：KL {num(doc.output_prediction.KL)}，加法基线 KL {num(doc.output_prediction.additive_KL)}。这个预测器直接拟合输出，尚不是由上述单元图谱重建的端到端机制。</p>
      {analysis.baselines&&<p><strong>更强简单基线（事后挑战）：</strong>只按“正确颜色 × 指定语言”查四状态均值表，输出 KL {num(analysis.baselines.output_KL.four_state)}；当前组八条件模型 / 四状态表误差 {num(analysis.baselines.groups[group.id].eight_over_four)}。两者接近时，优先保留更简单的解释；本项不追加盲测成功数。</p>}
    </>}
    {tab==='effects'&&<>
      <p>在答案前最后位置将 A 削弱 50% 或置零，重算原模型后续层。对照包括同层随机单元组和等实际残差范数的随机方向。正的优势下降表示削弱了正确答案；首词改变不必然改变答案类别。</p>
      <div style={{overflowX:'auto'}}><table className="v6-table"><thead><tr><th>对象 / 削弱</th><th>正确优势下降 / 对照</th><th>有效范数对照</th><th>首词改变</th><th>效应预测误差比</th><th>作用判据</th></tr></thead><tbody>{doc.effects.map(e=><tr key={`${e.group}-${e.dose}`}><td>{e.group} / {pct(e.dose)}</td><td>{num(e.mean_margin_loss)} / {num(e.control_mean_margin_loss)} nats</td><td>{e.valid_controls}/{e.n}</td><td>{e.top1_changes}/{e.n}</td><td>{num(e.effect_prediction_mse_ratio)}</td><td>{e.use_supported?'范围内支持':'未支持'}</td></tr>)}</tbody></table></div>
      <p className="v8-note">作用预测使用原始输出加上发现集学到的效应，属于干预预测；不计为独立生成。取消编辑后的 {doc.restoration_checks}/256 次逐位复原只验证实现正确，不能冒充机制救援。</p>
      <p>负的优势下降意味着移除后正确答案反而更占优势。这可保留为反向贡献线索，不能把“正向支持答案”的假设改名后算作原假设通过。两档剂量均展示，结论以完全置零为主要范围；没有首词翻转时不宣称必要性。</p>
      {analysis.causal&&<><p>同一组在不同条件下的作用（置零）：下表是描述性分层；因果子集的语言与句式相关，不能把差异全部归因于语言。</p><table className="v6-table"><thead><tr><th>单元组</th><th>颜色 / 语言</th><th>优势下降</th></tr></thead><tbody>{analysis.causal.filter(e=>e.dose===1&&['binding-B34','language-B28'].includes(e.group)).flatMap(e=>Object.entries(e.strata).map(([key,v])=><tr key={`${e.group}-${key}`}><td>{e.group}</td><td>{key.startsWith('semantic1_')?'红':'蓝'} / {key.endsWith('language1')?'英文':'中文'}</td><td>{num(v.mean_loss)} nats · {v.n} 条</td></tr>))}</tbody></table></>}
      <table className="v6-table"><thead><tr><th>历史能力题库</th><th>原正确</th><th>原对改错</th><th>原错改对</th></tr></thead><tbody>{Object.entries(doc.capability).map(([name,c])=><tr key={name}><td>{name}</td><td>{c.baseline_correct}/{c.n}</td><td>{c.lost}</td><td>{c.gained}</td></tr>)}</tbody></table><p className="v8-note">512 道已暴露的历史选择题，仅作副作用描述；不能宣布通用能力保持。</p>
      <details><summary>全部效应、置信区间与对照</summary><pre className="v8-artifact-json">{JSON.stringify({effects:doc.effects,audit:analysis.causal},null,2)}</pre></details>
    </>}
    {tab==='errors'&&group&&<>
      <p>{analysis.group_failures?.[group.id]?.join('；')||'按冻结阈值逐项检查。以下保留误差改善最差的三个家族；即使均优于基线，也不能把它们标为已经失败。'}</p>
      {group.counterexamples.map(c=><details key={c.family}><summary>家族 {c.family} · 候选误差 {num(c.error)} / 基线 {num(c.baseline)} · {c.error>=c.baseline?'未优于基线':'仍优于基线，属于边界案例'}</summary><pre className="v8-sample-text">{c.sample.prompt}</pre><p>显示该家族第一种条件，误差按其全部八条件聚合。</p></details>)}
      {(analysis.causal_counterexamples?.[group.id]||[]).map(c=><details key={`causal-${c.family}`}><summary>作用边界 · 家族 {c.family} · 优势下降 {num(c.margin_loss)} nats</summary><pre className="v8-sample-text">{c.prompt}</pre><p>{c.margin_loss<0?'移除后正确优势增加，反驳一贯正向支持的解释。':'移除后正确优势下降；仍需对照及整体判据。'}</p><pre className="v8-artifact-json">{JSON.stringify({baseline:c.baseline,edited:c.edited},null,2)}</pre></details>)}
      <p>待解决：从所选单元倒查注意力读入、门控与写出；检验模型是否通过其他单元补偿；使用更丰富的颜色、语言与句式再次冻结。不能沿着失败材料调参后继续称为盲测。</p>
    </>}
    {tab==='units'&&<><p>代表性来自旧档案中的不同响应形态。下列单元的预测误差属于描述性检验，没有逐一通过多重比较，也没有逐一做干预。</p><div style={{overflowX:'auto'}}><table className="v6-table"><thead><tr><th>原生单元</th><th>选取原因</th><th>误差 / 加法基线</th><th>胜出家族</th><th>新样本标准差</th></tr></thead><tbody>{doc.representatives.map(r=><tr key={`${r.block}-${r.unit}`}><td><a href={loc(doc,r.block,r.unit)}>B{r.block}/U{r.unit}</a></td><td>{criterion[r.kind]}</td><td>{num(r.relative_mse)}</td><td>{pct(r.family_win_rate)}</td><td>{num(r.holdout_std)}</td></tr>)}</tbody></table></div></>}
    {tab==='evidence'&&<>
      <p>留出行为门：首词正确 {pct(doc.behavior.first_top1)}；{doc.behavior.passed?'通过':'未通过'}。另有 {doc.generation.length} 条自然生成审查，原文保留。</p>
      <p>运行 {doc.run_id}；模型快照 {doc.model_snapshot}。材料、选择、权重加载审计及哈希可在下列产物逐项回查。</p>
      {Object.entries(doc.artifacts||{}).map(([name,sha])=><p key={name}><a href={apiUrl(`/api/research-library/artifacts/${sha}`)} target="_blank" rel="noreferrer">{name}</a></p>)}
      <details><summary>限制、分析与自然生成原文</summary><pre className="v8-artifact-json">{JSON.stringify({limits:doc.limits,audit:analysis,generation:doc.generation},null,2)}</pre></details>
    </>}
  </section>;
}
