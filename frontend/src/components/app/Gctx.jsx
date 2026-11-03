/* 全局上下文条（研究工作台）：数据集 + 被测模型。
   为什么放在这里而不是各面板内部：原来数据集选择是 LangAtlas 的局部 useState，换一页就丢，于是
   「三类功效池 100% 来自 CMMLU，换个数据集结论还成立吗」这件事在工作台上无法回答——而这正是
   当前最要紧的外部效度问题。提升到全局后，所有界面读同一组维度，状态进地址栏（ds / mdl）。
   两个来源都是现成接口，不新增服务：数据集读 /api/lang/atlas/datasets（按 source 真实分组，
   含 testable 与类型映射缺失说明），模型读 /api/lang/tree 的 models。
   纪律：没有数据源的东西不上屏。Xiezhi 从未下载（非商业许可 + 无 HF 直链），所以只列为不可选项并写明原因，
   不假装它在题库里；生成式模板族不在选择题口径内，只给族数，不给题目数。 */
import { useApi } from './rdc_fusion/api.js';
import { shortModel } from '../v8/v8Util.js';

/* source 的真实取值是 'C-Eval val' / 'CMMLU test' / 'CMExam train'；显示用短名，取值仍用原值 */
const SHORT = { 'C-Eval val': 'C-Eval', 'CMMLU test': 'CMMLU', 'CMExam train': 'CMExam' };
const dsLabel = id => SHORT[id] || String(id || '').split(' ')[0] || id;

/* 生成式模板族：六族 ENT-XF / GR-NEG / GR-Q / GR-ROLE / KN-ATTR / KN-ISA，来源 shared/dist_protocol/grammars。
   它们不是选择题，不进题库口径（所以这里只给族数，不给题目数）。 */
const TPL_ID = 'TPL';
const TPL_FAMILIES = 6;

export default function Gctx({ ds, mdl, onDs, onMdl, scope, extra }) {
  const dsApi = useApi('/api/lang/atlas/datasets');
  const treeApi = useApi('/api/lang/tree');
  const list = (dsApi.data?.datasets || []).filter(d => d.n_questions > 0);
  const models = treeApi.data?.models || [];
  const picked = list.find(d => d.id === ds) || null;
  const testableN = picked ? Object.values(picked.by_category || {}).reduce((s, c) => s + (c.n || 0), 0) : 0;

  let status = '';
  if (picked) status = `${picked.n_questions.toLocaleString()} 题 · ${picked.n_subjects} 科 · 可映射三类 ${testableN.toLocaleString()}`;
  else if (ds === TPL_ID) status = `${TPL_FAMILIES} 族生成式模板 · 不在选择题口径内`;
  else if (dsApi.error) status = `数据集清单不可用：${dsApi.error}`;
  else if (!ds) status = list.length ? `自动 · 可测 ${list.filter(d => d.testable).length} / ${list.length} 个题库` : '读取数据集清单…';

  return (
    <div className="wb-gctx">
      <span className="wb-gctx-hd">分析条件</span>
      <label className="wb-gctx-f">
        <span>数据集</span>
        <select value={ds} onChange={e => onDs(e.target.value)} aria-label="数据集">
          <option value="">自动</option>
          {list.map(d => (
            <option key={d.id} value={d.id} disabled={!d.testable}>
              {dsLabel(d.id)} · {d.n_questions.toLocaleString()} 题{!d.testable ? '（无可映射科目）' : ''}
            </option>
          ))}
          <option value={TPL_ID}>生成式模板族 · {TPL_FAMILIES} 族</option>
          <option value="Xiezhi" disabled>Xiezhi · 未接入</option>
        </select>
      </label>
      <label className="wb-gctx-f">
        <span>模型</span>
        <select value={mdl} onChange={e => onMdl(e.target.value)} aria-label="被测模型">
          <option value="">自动</option>
          {models.map(m => <option key={m} value={m} title={m}>{shortModel(m)}</option>)}
        </select>
      </label>
      <span className="wb-gctx-st" title={picked?.reason || ''}>{status}</span>
      {scope && <span className="wb-gctx-scope">{scope}</span>}
      {extra}
    </div>
  );
}
