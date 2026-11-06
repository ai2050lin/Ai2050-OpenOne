import {useState} from 'react';
import MlpBankExplorer from './MlpBankExplorer.jsx';
import MlpResponseAnalysis from './MlpResponseAnalysis.jsx';
export default function MlpCodeStudyReport({doc:d}){
  const [tab,setTab]=useState('summary');
  return <section aria-label="全MLP答码解耦研究报告"><p><strong>全350,208个原生MLP单元 · 事实×问法×答案代码</strong></p><p className="v8-note">{d.run_id} · {d.state==='complete'?'正式采集完成':d.state==='calibration_failed'?'行为校准未通过':'研究执行中'}。本轮把稳定事实、问题关系、抽象决定与输出代码分开；单元被采集不等于已破解。</p>
    <div className="v6-toolbar">{[['summary','结论与行为门'],['native','真实3D原场'],...(d.analysis?[['analysis','全部单元检验']]:[]),['evidence','全部判据与来源']].map(([k,t])=><button className="fw-tbtn" aria-pressed={tab===k} disabled={k==='native'&&!d.bank_available} key={k} onClick={()=>setTab(k)}>{t}</button>)}</div>
    {tab==='summary'&&<><p>{d.conclusion||'中性代码的发现集试跑全部条件过门；正式采集后才进行全阵列留出检验。尚无新的机制结论。'}</p><p>正式计划9,984条长输入、416个背景。每条覆盖36块全部9728单元、G/V/A和37层隐态的两个读出；保留完整参数定位。跨代码主判只在发现集A/B上拟合符号。</p><p className="v8-note">第一版反转Yes/No或是/否通常含义的任务没有通过行为门；失败保留。第二版使用A/B、C/D、甲/乙，映射方向独立变化。</p>{d.analysis&&<><table className="v6-table"><thead><tr><th>读出</th><th>响应复现单元</th><th>行为门后可解释</th><th>打乱标签复现</th></tr></thead><tbody>{['完整问题后','assistant边界'].map((label,i)=><tr key={i}><td>{label}</td><td>{d.analysis.response_replicated_by_readout[i].toLocaleString()}</td><td>{d.analysis.eligible_supported_by_readout[i].toLocaleString()}</td><td>{d.analysis.negative_controls.heldout_shuffled_replicated_by_readout[i]}</td></tr>)}</tbody></table><details><summary>全部留出行为条件格与反例入口</summary><pre className="v8-artifact-json">{JSON.stringify({behavior_cells:d.analysis.behavior_cells,errors:d.analysis.behavior_errors,counterexamples:d.analysis.counterexample_entry_points},null,2)}</pre></details></>}<a href="?lens=spatial&view=research&run=MLP-ATLAS-INVENTORY-20261104-v1">查看全部MLP单元证据账本</a></>}
    {tab==='native'&&<MlpBankExplorer run={d.run_id}/>}
    {tab==='analysis'&&d.analysis&&<MlpResponseAnalysis run={d.run_id} analysis={d.analysis}/>}
    {tab==='evidence'&&<><p className="v8-note">完整记录失败、作用范围和未完成项；数组检验只支持范围内响应，不自动晋级个体机制或完整图谱。</p><pre className="v8-artifact-json">{JSON.stringify({prereg:d.preregistration,analysis_plan:d.analysis_plan,pilot:d.pilot,refs:d.artifact_refs,limitations:d.limitations},null,2)}</pre></>}
  </section>;
}
