/* 网站首页：一句话定位 → 研究工作台的两个入口（神经元逆向分析、语言图谱，各带真实计数）→ 参与 → 普查进度与证据边界。
   计数来自本机数据包里的 /api/lang/tree、/api/model/skeleton 与 /api/platform/overview；没有数据时显示「—」，不回退演示数字。 */
import { useApi } from '../app/rdc_fusion/api.js';
import { CensusJobBar } from '../open/OpenHome.jsx';
import { fmtAgo, fmtInt, fmtTokens } from '../open/openUtil.js';
import { defaultModel, itemLevel, levelMeta, shortModel, workbenchHref } from './v8Util.js';

export default function SiteHome({ config, set }) {
  const ov = useApi('/api/platform/overview');
  const treeApi = useApi('/api/lang/tree');
  const skApi = useApi('/api/model/skeleton');
  const tree = treeApi.data && !treeApi.data.pending ? treeApi.data : null;
  const model = defaultModel(tree, Object.fromEntries((skApi.data?.models || []).map(s => [s.model, s])));
  const d = ov.data;
  const items = tree?.items || [];
  const studied = items.filter(it => itemLevel(it, model) > 0).length;
  const census = d?.neurons?.census_runs || [];
  const nTok = census.reduce((s, r) => s + (r.n_tokens || 0), 0);
  const grades = d?.neurons?.explanation_grades || {};
  const nExpl = Object.values(grades).reduce((a, b) => a + b, 0);
  const types = (tree?.types || []).filter(t => t.id !== 'EXT');
  return (
    <div className="v8-home">
      <section className="v8-hero">
        <h1>{config?.title || 'AI2050 开源 LLM 逆向研究平台'}</h1>
        <p>{config?.tagline || '研究大语言模型怎样用有限参数编码语言：每种语言现象在模型内部对应什么结构，由哪些神经元承载。'}</p>
      </section>

      <section className="v8-mainlines" aria-label="研究工作台入口">
        <a className="v8-mainline" href={workbenchHref('lens=spatial&view=neuron')}>
          <span className="n">1</span>
          <h2>神经元逆向分析</h2>
          <p>在研究工作台的「空间」里看模型的全部层，点开一层看它的内部结构与 9,728 个 MLP 神经元，再点开一个神经元看它在自然语料中的激活与对应的语言特征。</p>
          <div className="v8-ml-stats">
            <span><b>{nTok ? fmtTokens(nTok) : '—'}</b>普查 token</span>
            <span><b>{fmtInt(nExpl)}</b>已检验解释</span>
            <span><b>{census.length || '—'}</b>次普查</span>
          </div>
          <small>{census.length ? '普查：自然语料上逐个神经元的激活统计，由参与者的 GPU 分片完成' : '还没有普查结果'}</small>
          <em>进入研究工作台 · 空间 →</em>
        </a>
        <a className="v8-mainline" href={workbenchHref('lens=spatial&view=lang')}>
          <span className="n">2</span>
          <h2>语言图谱</h2>
          <p>按大类选一种语言现象（如疑问、否定、上下位），看它在模型里从哪一层开始被编码、编码长什么样，以及在 LLM 各层里的有效神经元。</p>
          <div className="v8-ml-stats">
            {types.map(t => {
              const its = items.filter(i => i.type === t.id);
              return <span key={t.id}><b>{its.filter(i => itemLevel(i, model) > 0).length}/{its.length}</b>{t.name}</span>;
            })}
          </div>
          <small>{tree ? `${studied} 个语言项已有数据 · ${shortModel(model)} · 状态 ${levelMeta(0).sym} 未研究 → ${levelMeta(4).sym} 跨模型复现` : '读取语言树…'}</small>
          <em>进入研究工作台 · 空间 →</em>
        </a>
      </section>
      <p className="v8-home-route"><a href={workbenchHref('lens=route')}>查看研究进展（研究工作台 · 路线）→</a></p>

      <section className="v8-home-row">
        <div className="v8-home-card">
          <h3>参与研究</h3>
          <p>下载一键包、双击运行，你的 GPU 会自动领取一个分片（模板测试或神经元普查），结果经中心校验与抽样复算后合并。</p>
          <div className="v8-home-stats">
            <span><b>{fmtInt(d?.people?.nodes_online)}</b> / {fmtInt(d?.people?.nodes_total)} 节点在线</span>
            <span><b>{fmtInt(d?.people?.contributors)}</b> 位贡献者</span>
            <span><b>{fmtInt(d?.families?.results_real)}</b> 份模板结果</span>
          </div>
          <button type="button" className="fw-tbtn primary" onClick={() => set({ page: 'join' })}>去参与 →</button>
        </div>
        <div className="v8-home-card">
          <h3>神经元普查进度</h3>
          {(d?.census_jobs || []).length ? (d.census_jobs.map(j => <CensusJobBar key={j.job_id} job={j} />))
            : <p>{ov.error ? `中心服务未接入：${ov.error}` : d ? '目前没有开放的普查作业。' : '读取中…'}</p>}
          <button type="button" className="fw-tbtn" onClick={() => set({ page: 'learn' })}>先了解方法</button>
        </div>
      </section>

      {d && (
        <section className="v8-home-note">
          <b>证据边界</b>
          <p>{d.note}</p>
          <p>快照更新于 {fmtAgo(d.computed_at)}{config?.public_scope?.length ? `；公开内容：${config.public_scope.join('、')}` : ''}。</p>
        </section>
      )}
    </div>
  );
}
