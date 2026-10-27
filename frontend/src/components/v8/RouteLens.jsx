/* 研究工作台 · 进展：内容按 tab 渲染（?rtab=neuron|lang；页签本身在 Workbench 的顶部子导航，
   与空间/研发共用同一个 SubNav 组件与 .wb-subnav 样式）。
   神经元逆向分析进展：普查规模、逐层摘要、分型、解释检验与模板选择性，按块画成热力条（每格一个块）；
   语言图谱进展：大类 × 等级的覆盖、语言项 × 模型的等级矩阵、研究缺口，以及模式族证据门链（记分板快照）。
   各来源分开计数，不拼成转化率；全部读取来自本机数据包。 */
import { useMemo, useState } from 'react';

import RouteBoard from '../app/rdc_fusion/RouteBoard.jsx';
import { useApi } from '../app/rdc_fusion/api.js';
import { Card, Empty } from '../app/rdc_fusion/ui.jsx';
import '../app/workbench.css';
import { LevelChip } from './LangPage.jsx';
import { LEVELS, READOUT_LABEL, defaultModel, itemLevel, levelMeta, seq, shortModel } from './v8Util.js';
import './v8.css';

function fmtTok(v) {
  if (!v) return '—';
  if (v >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  return `${Math.round(v / 1e3)}K`;
}

/* 一行热力条：每格一个块，颜色 = 该块数值 ÷ 本行最大值（单一色相，浅 → 深）；悬停看原始数值 */
function HeatStrip({ label, sub, values, onPick }) {
  const max = Math.max(1e-9, ...values.map(v => v || 0));
  return (
    <div className="v8-strip">
      <div className="v8-strip-lab"><b>{label}</b>{sub && <small>{sub}</small>}</div>
      <div className="v8-strip-cells" role="list">
        {values.map((v, b) => (
          <button key={b} type="button" role="listitem" className={v == null ? 'na' : ''} style={v == null ? undefined : { background: seq((v || 0) / max) }}
                  title={`块 ${b}：${v == null ? '无数据' : Number(v).toLocaleString()}`} onClick={() => onPick?.(b)} />
        ))}
      </div>
      <span className="v8-strip-max">最大 {Number.isFinite(max) && max > 1e-9 ? Number(max).toLocaleString() : '—'}</span>
    </div>
  );
}

function NeuronProgress({ census, skeleton, onOpenBlock }) {
  const tw = useApi(census ? `/api/neurons/tower?model_key=${encodeURIComponent(census)}` : null);
  const idx = useApi('/api/neurons/index');
  const tower = tw.data && !tw.data.pending ? tw.data : null;
  const [ro, setRo] = useState('final');
  if (!census) return <Empty>当前模型还没有自然语料普查，没有神经元逆向分析数据。</Empty>;
  if (tw.error) return <Empty>层塔数据不在本机数据包里：{tw.error}</Empty>;
  if (!tower) return <Empty>读取层塔…</Empty>;
  const run = (idx.data?.census || []).find(r => r.model_key === census && r.run === tower.run);
  const blocks = tower.blocks;
  const grades = {};
  blocks.forEach(b => Object.entries(b.explained || {}).forEach(([g, n]) => { grades[g] = (grades[g] || 0) + n; }));
  const nExpl = Object.values(grades).reduce((a, b) => a + b, 0);
  const typeSum = k => blocks.reduce((s, b) => s + (b.types ? b.types[k] : 0), 0);
  const sel = tower.selectivity || [];
  return (
    <>
      <div className="v8-route-tiles">
        <div><b>{fmtTok(run?.n_tokens)}</b><span>普查 token</span><small>{run ? `${run.sources?.length || 0} 个语料来源 · 每窗口 ${run.seq_len} token` : '—'}</small></div>
        <div><b>{tower.summaries_ready}/{tower.n_layers}</b><span>块已生成逐层摘要</span><small>{(tower.d_ff * tower.n_layers).toLocaleString()} 个 MLP 单元（{tower.n_layers} 块 × {tower.d_ff.toLocaleString()}）</small></div>
        <div><b>{typeSum(0) + typeSum(1)}</b><span>单 token / 上下文检测器（T1 + T2）</span><small>按 L30-v1 规则分型；块 30 以外为描述性应用</small></div>
        <div><b>{nExpl}</b><span>已做解释检验的单元</span><small>{Object.entries(grades).map(([g, n]) => `${g} ${n}`).join(' · ') || '—'}</small></div>
        <div><b>{sel.length}</b><span>模板选择性运行</span><small>{sel.map(s => `${s.item || s.tm_id}·${s.factor}`).join('、') || '—'}</small></div>
      </div>
      <div className="v8-tools">
        <span className="v8-note">每格一个块（块 0 → 块 {tower.n_layers - 1}，从左到右）；颜色按本行最大值归一化。点格子进入该块的神经元。</span>
        <span className="v8-seg">{['final', 'probe_answer'].map(x => <button key={x} type="button" className={ro === x ? 'on' : ''} onClick={() => setRo(x)}>选择性：{READOUT_LABEL[x] || x}读出</button>)}</span>
      </div>
      <div className="v8-strips">
        <HeatStrip label="T1 单 token 检测器" values={blocks.map(b => (b.types ? b.types[0] : null))} onPick={onOpenBlock} />
        <HeatStrip label="T3 稀疏强激活" values={blocks.map(b => (b.types ? b.types[2] : null))} onPick={onOpenBlock} />
        <HeatStrip label="强激活率中位数" values={blocks.map(b => b.dens_median ?? null)} onPick={onOpenBlock} />
        <HeatStrip label="已解释单元" sub="文本可复现 + 语料可识别 + 未解释" values={blocks.map(b => Object.values(b.explained || {}).reduce((a, c) => a + c, 0))} onPick={onOpenBlock} />
        {sel.map(s => (
          <HeatStrip key={s.file} label={`选择性：${s.item || s.tm_id}`} sub={`${s.factor}（${s.hi} 对 ${s.lo}）`}
                     values={(s.by_readout?.[ro] || []).map(v => v ?? 0)} onPick={onOpenBlock} />
        ))}
      </div>
      <p className="v8-note">尚未覆盖：注意力头的逐头分析、跨层计算边（归因图）、多数单元的因果检验。控制轴结论见各单元卡的「干预记录」。{skeleton?.model ? ` 模型：${shortModel(skeleton.model)}。` : ''}</p>
    </>
  );
}

function LangProgress({ tree, model, onOpenItem }) {
  const models = tree.models || [];
  const gaps = tree.items.filter(it => itemLevel(it, model) === 0);
  return (
    <>
      <div className="v8-two">
        <table className="v6-mini-table v8-cov">
          <thead><tr><th>大类</th>{LEVELS.map(l => <th key={l.level}><LevelChip level={l.level} /></th>)}<th className="num">合计</th></tr></thead>
          <tbody>
            {tree.types.map(t => {
              const its = tree.items.filter(it => it.type === t.id);
              return (
                <tr key={t.id}><th>{t.name}</th>
                  {LEVELS.map(l => <td key={l.level} className="num">{its.filter(it => itemLevel(it, model) === l.level).length || ''}</td>)}
                  <td className="num">{its.length}</td></tr>
              );
            })}
          </tbody>
        </table>
        <div>
          <b className="v8-sub">研究缺口（{shortModel(model)} 上还没有数据）</b>
          {gaps.length ? <ul className="v8-gaps">{gaps.map(it => (
            <li key={it.id}><button type="button" className="v6-linkbtn" onClick={() => onOpenItem(it)}>{it.name}</button>
              <small>{tree.types.find(t => t.id === it.type)?.name} · {it.status.planned ? '还没有模式族' : '其他模型上有数据'}</small></li>
          ))}</ul> : <Empty>当前模型上每个语言项都已有数据。</Empty>}
        </div>
      </div>
      <h4 className="v8-h4">语言项 × 模型 <small>读出轴等级；各模型分开记录，方向不跨模型比较</small></h4>
      <div className="v8-matrix-wrap">
        <table className="v6-mini-table v8-matrix">
          <thead><tr><th>语言项</th>{models.map(m => <th key={m} title={m}>{shortModel(m)}</th>)}</tr></thead>
          <tbody>
            {tree.types.map(t => tree.items.filter(it => it.type === t.id).map((it, i) => (
              <tr key={it.id} className={i === 0 ? 'first' : ''}>
                <th><button type="button" className="v6-linkbtn" onClick={() => onOpenItem(it)}>{it.name}</button><small>{t.name}</small></th>
                {models.map(m => {
                  const l = it.status?.by_model?.[m] ?? 0;
                  const meta = levelMeta(l);
                  return <td key={m} title={`${it.name} · ${shortModel(m)}：${meta.label}`}><span className="v8-chip" style={{ background: meta.color, color: meta.ink }}>{meta.sym} {meta.label}</span></td>;
                })}
              </tr>
            )))}
          </tbody>
        </table>
      </div>
      <p className="v8-note">{tree.policy}</p>
    </>
  );
}

/* 页签由工作台顶部的子导航（workbenchUtil.ROUTE_VIEWS）提供，这里只按 tab 渲染内容：
   页签与空间/研发同一层级、同一组件，避免同一页面出现两种页签样式。 */
export default function RouteLens({ tab = 'neuron', onGoSpace, onGo }) {
  const treeApi = useApi('/api/lang/tree');
  const skApi = useApi('/api/model/skeleton');
  const tree = treeApi.data && !treeApi.data.pending ? treeApi.data : null;
  const skeletons = useMemo(() => Object.fromEntries((skApi.data?.models || []).map(s => [s.model, s])), [skApi.data]);
  const model = defaultModel(tree, skeletons);
  const skeleton = skeletons[model];
  const census = skeleton?.census_run ? skeleton.census_key : null;
  return (
    <div className="v8-route">
      {tab === 'neuron' && (
        <Card title="神经元逆向分析进展" sub={`${shortModel(model)} · 自然语料普查 → 逐层摘要与分型 → 解释检验 → 模板选择性（各项分开计数）`}>
          <NeuronProgress census={census} skeleton={skeleton} onOpenBlock={b => onGoSpace('neuron', { block: b, unit: null })} />
        </Card>
      )}
      {tab === 'lang' && (
        <>
          <Card title="语言图谱进展" sub="每个语言项在各模型上的读出轴等级（路线 T：模板对照）；路线 N（神经元经模板确认入树）尚未接入">
            {tree ? <LangProgress tree={tree} model={model} onOpenItem={it => onGoSpace('lang', { item: it.id, type: it.type })} />
              : <Empty>{treeApi.error ? `语言谱图不在本机数据包里：${treeApi.error}` : '读取语言谱图…'}</Empty>}
          </Card>
          <RouteBoard onGo={onGo} />
        </>
      )}
    </div>
  );
}
