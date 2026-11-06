import {useState} from 'react';
import {useApi} from '../app/rdc_fusion/api.js';
import {Empty} from '../app/rdc_fusion/ui.jsx';
import {ExtraPackPrompt} from '../open/DataBar.jsx';

export default function DatasetAtlasPlan({sha,implementation=false}){
  const [open,setOpen]=useState(false);
  const data=useApi(open&&sha?`/api/research-library/artifacts/${sha}`:null);
  const d=data.data;
  return <details open={open} onToggle={e=>setOpen(e.currentTarget.open)} aria-label="数据与图谱组织方案">
    <summary>数据与图谱组织方案 · {implementation?'部分已实施，持续更新':'待实施'}</summary>
    {data.error?<Empty>{data.error}<ExtraPackPrompt pack={data.missingPack}/></Empty>:!d?<p className="v8-note">读取方案…</p>:d.schema!=='research_dataset_atlas_plan.v1'?<Empty>方案格式不匹配。</Empty>:<>
      <p><strong>图谱作为主入口，数据集作为证据、筛选与研发管理入口。</strong></p>
      <p className="v8-note">{d.implementation_status==='partial'?'登记、真实任务、AI盲评与生成轨迹已接入；字典发现、第二物理节点复算和图谱预测仍待完成。各阶段的实际状态如下，不代表机制已经证实。':'以下为方案，不表示新数据已经接入、分布式任务已经上线或机制已经证实。已完成的本轮研究见上方真实报告。'}</p>
      {(d.principles||[]).map(x=><details key={x.title}><summary>{x.title}</summary><p className="v8-note">{x.description}</p></details>)}
      <details><summary>客户端一级与二级菜单</summary><table className="v6-table"><thead><tr><th>入口</th><th>二级分组</th><th>信息层级</th></tr></thead><tbody>{(d.client_pages||[]).map(x=><tr key={x.page}><td>{x.page}</td><td>{x.secondary}</td><td>{x.behavior}</td></tr>)}</tbody></table></details>
      <details><summary>数据集选型与使用状态</summary><table className="v6-table"><thead><tr><th>来源</th><th>状态</th><th>用途与边界</th></tr></thead><tbody>{(d.dataset_candidates||[]).map(x=><tr key={x.id}><td><a href={x.primary_source} target="_blank" rel="noreferrer">{x.name}</a></td><td>{x.status||'建议接入，使用状态未登记'}</td><td>{x.role}</td></tr>)}</tbody></table></details>
      <details><summary>阶段任务与验收条件</summary>{(d.stages||[]).map(x=><div key={x.stage}><h4>{x.stage}{x.status?` · ${x.status}`:''}</h4><p className="v8-note">{x.work}</p><p className="v8-note">验收：{x.acceptance}</p></div>)}</details>
      <details><summary>完整契约、身份映射与预测判据</summary><pre className="v8-sample-text">{JSON.stringify(d,null,2)}</pre></details>
    </>}
  </details>;
}
