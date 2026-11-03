/* 全局上下文条（研究工作台）：数据集 + 被测模型，用方块给选项，不用下拉框。
   数据集这一组只在真读题库的视图出现（dsBound，口径见 workbenchUtil.dsBound）：
   语言图谱两处（空间的数据集实测、进展的图谱测试题）。神经元页读自然语料、规则账本按规则记录，
   给它们一排换不动数字的方块，等于给了一个假旋钮。
   为什么是方块：这两维不是「设置项」，而是这一页所有数字的读法前提。下拉框把「有哪些可选」
   藏进一次点击里，而方块把每个选项的题量、科目、普查状态直接印在面上——换哪个会变什么，
   扫一眼就知道。选中态同时写进地址栏（ds / mdl），换页不丢。
   两个来源都是现成接口，不新增服务：数据集读 /api/lang/atlas/datasets（按 source 真实分组，
   含 testable 与类型映射缺失说明），模型读 /api/lang/tree 的 models（普查状态读 /api/model/skeleton）。
   四个数据集：C-Eval / MMLU / M3Exam 三个外部集 + 本地数据集（生成式模板族）。
   纪律：没有数据源的东西不上屏——三个外部集的题量来自题库统计（本机有 questions.v3.jsonl 就从正文数，
   没有就从 meta.v3.json 的 by_dataset 摘要读：发布包只带后者，题目正文按源地址在本机取），
   本地那一项是 shared/dist_protocol/grammars 的六族模板，不进选择题口径，
   所以只给族数，不给题目数。 */
import { useEffect, useMemo } from 'react';

import { useApi } from './rdc_fusion/api.js';
import { defaultModel, shortModel } from '../v8/v8Util.js';

/* source 的真实取值是 'C-Eval val' / 'MMLU test' / 'M3Exam zh test'；显示用短名，取值仍用原值。
   旧题库的 CMMLU / CMExam 仍留在这里：旧链接（?ds=CMMLU test）解析时不会变成空。 */
const SHORT = { 'C-Eval val': 'C-Eval', 'MMLU test': 'MMLU', 'M3Exam zh test': 'M3Exam',
  'CMMLU test': 'CMMLU', 'CMExam train': 'CMExam' };
const dsLabel = id => SHORT[id] || String(id || '').split(' ')[0] || id;

/* 生成式模板族：六族 ENT-XF / GR-NEG / GR-Q / GR-ROLE / KN-ATTR / KN-ISA，来源 shared/dist_protocol/grammars。
   它们不是选择题，不进题库口径（所以这里只给族数，不给题目数）。 */
const TPL_ID = 'TPL';
const TPL_FAMILIES = 6;
/* 方块顺序按固定优先级，不按题量：MMLU 有 14,042 题、C-Eval 只有 169 题，按题量排会把默认值
   顶到 MMLU 上——第一次点「运行实测」要等半小时才出结果。顺序固定成 C-Eval → MMLU → M3Exam，
   默认值取这个顺序里第一个可测的（C-Eval，几分钟跑完），先跑通链路再谈规模。 */
const DS_ORDER = ['C-Eval', 'MMLU', 'M3Exam'];

/* 一个方块：主行是名字，副行是这一项的实际规模 / 状态。off = 有清单但不可选（写明原因在 title）。 */
function Opt({ on, off, onClick, title, main, sub }) {
  return (
    <button type="button" className={'wb-gopt' + (on ? ' on' : '') + (off ? ' off' : '')}
            onClick={off ? undefined : onClick} disabled={off} title={title} aria-pressed={on}>
      <b>{main}</b>
      <small>{sub}</small>
    </button>
  );
}

export default function Gctx({ ds, mdl, onDs, onMdl, scope, extra, dsBound = true }) {
  const dsApi = useApi('/api/lang/atlas/datasets');
  const treeApi = useApi('/api/lang/tree');
  const skApi = useApi('/api/model/skeleton');
  const tree = treeApi.data && !treeApi.data.pending ? treeApi.data : null;
  const skeletons = useMemo(() => Object.fromEntries((skApi.data?.models || []).map(s => [s.model, s])), [skApi.data]);
  const list = (dsApi.data?.datasets || []).filter(d => d.n_questions > 0);
  const rank = id => { const i = DS_ORDER.indexOf(dsLabel(id)); return i < 0 ? 99 : i; };
  const ordered = [...list].sort((a, b) => rank(a.id) - rank(b.id) || b.n_questions - a.n_questions);
  const models = tree?.models || [];

  /* 默认值 = 方块高亮的那一项，而且必须真的写进状态：
     否则会出现「顶部亮着 CMMLU，下面各面板各自用自己默认的数据集」，这种不一致在界面上看不出来。 */
  const knownDs = list.some(d => d.id === ds) || ds === TPL_ID;
  const firstTestable = ordered.find(d => d.testable) || ordered[0] || null;
  const effDs = knownDs ? ds : (firstTestable?.id || '');
  const effMdl = mdl && models.includes(mdl) ? mdl : defaultModel(tree, skeletons);
  const skReady = !!(skApi.data || skApi.error);

  /* 只有真读题库的视图才把默认数据集写进地址栏：神经元页把它写进去，会让人以为这一页按数据集算的。 */
  useEffect(() => { if (dsBound && !ds && effDs) onDs(effDs); }, [dsBound, ds, effDs, onDs]);
  useEffect(() => { if (!mdl && effMdl && tree && skReady) onMdl(effMdl); }, [mdl, effMdl, tree, skReady, onMdl]);

  const picked = list.find(d => d.id === effDs) || null;
  const testableN = picked ? Object.values(picked.by_category || {}).reduce((s, c) => s + (c.n || 0), 0) : 0;

  let status = '';
  if (picked) status = `${picked.n_questions.toLocaleString()} 题 · ${picked.n_subjects} 科 · 可映射三类 ${testableN.toLocaleString()}`;
  else if (effDs === TPL_ID) status = `${TPL_FAMILIES} 族生成式模板 · 不在选择题口径内`;
  else if (dsApi.error) status = `数据集清单不可用：${dsApi.error}`;
  else if (!dsApi.data) status = '读取数据集清单…';

  const dsOpts = [
    ...ordered.map(d => {
      const n = Object.values(d.by_category || {}).reduce((s, c) => s + (c.n || 0), 0);
      return {
        id: d.id, main: dsLabel(d.id), sub: `${d.n_questions.toLocaleString()} 题`, off: !d.testable,
        title: `${d.id} · ${d.n_questions.toLocaleString()} 道选择题 · ${d.n_subjects || 0} 个科目 · 可映射三类 ${n.toLocaleString()} 道`
          + (d.source_repo ? ` · 源 ${d.source_repo}${d.source_split ? ` (${d.source_split})` : ''}${d.license ? ` · ${d.license}` : ''}` : '')
          + (d.stats_source === 'meta' ? ' · 题量为建库时统计（本处不含题目正文）' : '')
          + (d.testable ? '' : ' · 没有可映射到三类的科目，不能用于三类对照')
          + (d.reason ? ` · ${d.reason}` : ''),
      };
    }),
    { id: TPL_ID, main: '本地数据集', sub: `${TPL_FAMILIES} 族生成式`,
      title: '本地生成：ENT-XF / GR-NEG / GR-Q / GR-ROLE / KN-ATTR / KN-ISA 六族模板，共 6,720 种组合。'
        + '是生成式对照材料，不是选择题，没有「正确率」这个口径' },
  ];

  return (
    <div className="wb-gctx">
      {dsBound && (
        <div className="wb-gctx-grp">
          <span className="wb-gctx-hd">数据集</span>
          <div className="wb-gctx-opts" role="group" aria-label="数据集">
            {dsOpts.map(o => (
              <Opt key={o.id} on={effDs === o.id} off={o.off} onClick={() => onDs(o.id)}
                   title={o.title} main={o.main} sub={o.sub} />
            ))}
          </div>
        </div>
      )}
      <div className="wb-gctx-grp">
        <span className="wb-gctx-hd">模型</span>
        <div className="wb-gctx-opts" role="group" aria-label="被测模型">
          {models.map(m => (
            <Opt key={m} on={effMdl === m} onClick={() => onMdl(m)} title={m}
                 main={shortModel(m)} sub={skeletons[m]?.census_run ? '已普查' : '未普查'} />
          ))}
          {!models.length && <span className="wb-gctx-st">{treeApi.error ? `模型清单不可用：${treeApi.error}` : '读取模型清单…'}</span>}
        </div>
      </div>
      {dsBound && status && <span className="wb-gctx-st">{status}</span>}
      {scope && <span className="wb-gctx-scope">{scope}</span>}
      {extra}
    </div>
  );
}
