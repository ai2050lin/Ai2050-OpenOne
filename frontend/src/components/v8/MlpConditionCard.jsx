import {useApi} from '../app/rdc_fusion/api.js';

const number=value=>Number.isFinite(value)?value.toPrecision(4):'未测';
const percent=value=>Number.isFinite(value)?`${(100*value).toFixed(1)}%`:'未测';

export function ConditionDetails({data,unit,block}) {
  const a=data.arrays;
  const study=data.tested_units?.[unit];
  return <section aria-label={`B${block}/U${unit} 条件响应卡`}>
    <p><strong>B{block}/U{unit} · 条件响应卡</strong> · {study?'已有新长文本检验':'已有自然语境观察，受控检验待补'}</p>
    <p className="v8-note">固定参数给出写入方向；这里记录它在不同条件下如何变化。候选关键词保留在原片中，不作为单元名称。</p>
    <div className="v6-toolbar"><span>同 token 内响应方差比：{percent(a.within_token_variance_fraction?.[unit])}</span>
      <span>同时出现正负写入的 token 组：{percent(a.sign_mix?.[unit])}</span>
      <span>词身份预测 FVU：{number(a.token_predictor_fvu?.[unit])}</span></div>
    <details><summary>观察条件、符号组合与限制</summary><p className="v8-note">{data.scope}</p>
      <p>负门控写入的 A² 占比 {percent(a.negative_gate_energy?.[unit])}；负门控与负 value 产生正写入的 A² 占比 {percent(a.double_negative_energy?.[unit])}。</p>
      <p className="v8-note">A² 尚未乘写出参数，也未计算相互抵消。这些数值不是知识量或因果贡献比例。FVU=1 是发现集均值基线，越低越好。</p>
    </details>
    {study&&<details open><summary>新长文本条件与跨层验证</summary>
      <p>{study.role==='source'?'源写入单元':'下游响应单元'} · {study.family==='broad'?'广泛关联候选':'稀有强响应候选'} · 词身份 × 角色 × 位置 × 答案映射。</p>
      <p>条件预测{study.prediction_passed?'通过冻结判据':'未通过冻结判据'}；稀有条件覆盖 {study.rare_families}/512 个家族。{study.rare_powered?'达到稀有分层样本门。':'稀有分层样本不足，只报告观察。'}</p>
      <a href={`?lens=spatial&view=graph&gview=internal&run=${data.run_id}&snapshot=${data.model_snapshot}`}>打开预测、干预、反例和图谱关系</a>
      <p className="v8-note">方向响应、预测、原模型扰动与行为中介分别登记；不自动升级为专属语义。</p>
    </details>}
  </section>;
}

export default function MlpConditionCard({block,unit,snapshot}) {
  const index=useApi('/api/traces/research-space');
  const study=index.data?.studies?.find(s=>s.run_id.startsWith('MLP-CONDITIONAL-PATHS-')&&s.model_snapshot===snapshot);
  const report=useApi(study?.report_sha256?`/api/research-library/artifacts/${study.report_sha256}`:null);
  const sha=report.data?.condition_layers?.[block];
  const layer=useApi(sha?`/api/research-library/artifacts/${sha}`:null);
  if(index.error||report.error||layer.error)return <p className="v8-note">条件响应卡暂时无法从服务器读取：{index.error||report.error||layer.error}</p>;
  if(!index.data)return <p className="v8-note">查询服务器条件响应记录…</p>;
  if(!study)return <p className="v8-note">此服务器尚未发布条件响应卡；下方仍可查看已归档原片和参数。</p>;
  if(!layer.data)return <p className="v8-note">读取 B{block} 条件响应数据…</p>;
  if(layer.data.model_snapshot!==snapshot||layer.data.block!==block)return <p className="v8-note">条件响应卡身份不匹配，停止展示。</p>;
  return <ConditionDetails data={layer.data} unit={unit} block={block}/>;
}
