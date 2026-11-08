/* 编码图谱 · 单元的四级状态与采集维度（Phase 340，ResearchPlan_Final 的特征卡状态）。
   可读 → 可写 → 被下游使用 → 可独立预测，逐级记录；只有登记过的运行才填写，其余写“未测”。
   目前只有注意力头在 MA-H2（Qwen3-4B 快照 6be1034f…）上有观察；注意力去向不等于计算使用。 */
import { DEMO_SNAPSHOT } from './ClientGuide.jsx';
import { FLOW_DATA, rankHeads } from './data/assocFlow.js';
import { CLAIMS } from './data/mainline.js';

const TOP = rankHeads();
const DIMS = [['层', true], ['token 位置', true], ['读出边界', true], ['样本', true], ['生成步', false], ['历史 KV', false]];

export default function UnitStates({ snapshot, block, head }) {
  const matched = (snapshot || '').startsWith(DEMO_SNAPSHOT) && block >= 0 && block < FLOW_DATA.heads.doc.length && head >= 0;
  const d = matched ? FLOW_DATA.heads.doc[block]?.[head] : null, r = matched ? FLOW_DATA.heads.real[block]?.[head] : null;
  const ranked = matched && TOP.some(x => x.l === block && x.h === head);
  const read = d == null ? '这个模型快照还没有登记的观察。'
    : `MA-H2 discovery：答案位置看资料颜色的注意力，资料范围 ${d.toFixed(2)}，现实范围 ${r.toFixed(2)}${ranked ? `；在两者差值最大的 ${TOP.length} 个头里（候选观察，未干预）` : '；不在差值最大的头里'}。`;
  const states = [['可读', read, ranked], ['可写', '未测：修补它能否改变下游读出', false],
    ['被下游使用', '未测：自然计算里是否真的用它', false], ['可独立预测', '未测：只凭图谱预测它在新材料上的响应', false]];
  return <div className="us" aria-label="单元的四级状态">
    <p className="v8-note">B{block} · H{head}{ranked ? ` · 主张 ${CLAIMS[0].id} · 选择与读取` : ''}</p>
    <div className="us-states">{states.map(([name, text, on]) => <div key={name} className={on ? 'on' : ''}><b>{name}</b><span className="v8-note">{text}</span></div>)}</div>
    <div className="us-dims"><span className="v8-note">采集维度</span>{DIMS.map(([name, have]) =>
      <span key={name} className={have && matched ? '' : 'off'}>{name}</span>)}</div>
    <p className="v8-note">虚线 = 还没有数据。{matched ? 'MA-H2 只采了 6 个读出位置的单步前向；生成步与历史 KV 在 H6 采集。' : ''}</p>
  </div>;
}
