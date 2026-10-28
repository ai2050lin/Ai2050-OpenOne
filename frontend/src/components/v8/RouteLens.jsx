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

const EXPL_LABEL = { 文本可复现: '文本可复现', 只在语料中可识别: '语料可识别', 未解释: '未解释' };

function NeuronProgress({ census, skeleton, onOpenBlock }) {
  const tw = useApi(census ? `/api/neurons/tower?model_key=${encodeURIComponent(census)}` : null);
  const idx = useApi('/api/neurons/index');
  const tower = tw.data && !tw.data.pending ? tw.data : null;
  const [ro, setRo] = useState('final');
  const [showSel, setShowSel] = useState(false);
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
  const legend = `linear-gradient(90deg, ${seq(0)}, ${seq(0.5)}, ${seq(1)})`;
  return (
    <>
      <h4 className="v8-h4">已经做到哪一步 <small>对 {shortModel(skeleton?.model)} 全部 MLP 中间单元做自然语料普查，再逐块分型与检验</small></h4>
      <div className="v8-route-tiles">
        <div><b>{fmtTok(run?.n_tokens)}</b><span>普查语料规模</span><small>{run ? `${run.sources?.length || 0} 个语料来源 · 每窗口 ${run.seq_len} token` : '—'}</small></div>
        <div><b>{(tower.d_ff * tower.n_layers).toLocaleString()}</b><span>MLP 单元全部有画像</span><small>{tower.n_layers} 块 × {tower.d_ff.toLocaleString()} · 逐层摘要 {tower.summaries_ready}/{tower.n_layers} 已生成</small></div>
        <div><b>{typeSum(0) + typeSum(1)}</b><span>检测器型单元（T1 + T2）</span><small>T1 单 token {typeSum(0).toLocaleString()} · T2 上下文 {typeSum(1).toLocaleString()} · 按 L30-v1 规则分型</small></div>
        <div><b>{nExpl}</b><span>做过解释检验的单元</span><small>{Object.entries(grades).map(([g, n]) => `${EXPL_LABEL[g] || g} ${n}`).join(' · ') || '尚未开始'}</small></div>
      </div>
      <h4 className="v8-h4">每一层（块）的情况 <small>每格一个块（块 0 → {tower.n_layers - 1}，从左到右）；<i className="v8-legend-bar" style={{ background: legend }} /> 颜色浅 = 少、深 = 多（按本行最大值归一）；点格子进入该块的神经元</small></h4>
      <div className="v8-strips">
        <HeatStrip label="T1 单 token 检测器" sub="激活集中命中同一个词的单元数" values={blocks.map(b => (b.types ? b.types[0] : null))} onPick={onOpenBlock} />
        <HeatStrip label="T3 稀疏强激活" sub="很少激活但一旦激活很强的单元数" values={blocks.map(b => (b.types ? b.types[2] : null))} onPick={onOpenBlock} />
        <HeatStrip label="强激活率中位数" sub="该块单元整体的激活频繁程度" values={blocks.map(b => b.dens_median ?? null)} onPick={onOpenBlock} />
        <HeatStrip label="已解释单元" sub="解释检验覆盖数（三个等级合计）" values={blocks.map(b => Object.values(b.explained || {}).reduce((a, c) => a + c, 0))} onPick={onOpenBlock} />
      </div>
      {sel.length > 0 && (
        <>
          <h4 className="v8-h4">模板选择性 <small>{sel.length} 个运行：对照句式里哪些块的单元能区分两个水平</small>
            <button type="button" className="v6-linkbtn" onClick={() => setShowSel(v => !v)}>{showSel ? '收起' : `展开 ${sel.length} 行`}</button>
          </h4>
          {showSel && (
            <>
              <div className="v8-tools">
                <span className="v8-seg">{['final', 'probe_answer'].map(x => <button key={x} type="button" className={ro === x ? 'on' : ''} onClick={() => setRo(x)}>在「{READOUT_LABEL[x] || x}」位置读出</button>)}</span>
              </div>
              <div className="v8-strips">
                {sel.map(s => (
                  <HeatStrip key={s.file} label={`${s.item || s.tm_id} · ${s.factor}`} sub={`区分「${s.hi}」与「${s.lo}」，入选单元数`}
                             values={(s.by_readout?.[ro] || []).map(v => v ?? 0)} onPick={onOpenBlock} />
                ))}
              </div>
            </>
          )}
        </>
      )}
      <p className="v8-note">边界：分型是描述性分类，不是证据等级；注意力头逐头分析、跨层计算边（归因图）与多数单元的因果检验尚未覆盖——控制轴结论见各单元卡的「干预记录」。{skeleton?.model ? ` 模型：${shortModel(skeleton.model)}。` : ''}</p>
    </>
  );
}

function LangProgress({ tree, model, onOpenItem }) {
  const models = tree.models || [];
  const lv = it => itemLevel(it, model);
  const gaps = tree.items.filter(it => lv(it) === 0);
  const nHas = tree.items.filter(it => lv(it) >= 1).length;
  const nHeld = tree.items.filter(it => lv(it) >= 3).length;
  return (
    <>
      <h4 className="v8-h4">已经做到哪一步 <small>语言项 = 一条可检验的语言规律；等级看「能不能在 {shortModel(model)} 上读出并复现」</small></h4>
      <div className="v8-route-tiles">
        <div><b>{tree.items.length}</b><span>条语言规律（语言项）</span><small>分属 {tree.types.length} 个大类，全部来自模板对照实验</small></div>
        <div><b>{nHas}</b><span>在此模型上有数据</span><small>至少完成过一次读出测量</small></div>
        <div><b>{nHeld}</b><span>留出复现及以上</span><small>证据最强：新数据上也读得出来</small></div>
        <div><b>{gaps.length}</b><span>还没有数据</span><small>{gaps.length ? '见下方研究缺口，可点进条目看怎么补' : '当前模型上每个语言项都已有数据'}</small></div>
      </div>
      <h4 className="v8-h4">覆盖与缺口 <small>左：各大类处在哪个等级；右：还没动手的条目</small></h4>
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
      <h4 className="v8-h4">语言项 × 模型明细 <small>每格一个等级（悬停看全称）；各模型分开记录，方向不跨模型比较</small></h4>
      <div className="v8-levels-legend">
        {LEVELS.slice(1).map(l => <LevelChip key={l.level} level={l.level} />)}
        <LevelChip none="○ 未研究" />
      </div>
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
                  return <td key={m} title={`${it.name} · ${shortModel(m)}：${meta.label}`}><span className="v8-chip" style={{ background: meta.color, color: meta.ink }}>{meta.sym}</span></td>;
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
        <Card title="神经元逆向分析进展" sub={`${shortModel(model)} · 普查 → 分型 → 解释检验 → 模板选择性（各来源分开计数，不拼成转化率）`}>
          <NeuronProgress census={census} skeleton={skeleton} onOpenBlock={b => onGoSpace('neuron', { block: b, unit: null })} />
        </Card>
      )}
      {tab === 'lang' && (
        <>
          <Card title="语言图谱进展" sub={`${shortModel(model)} · 每条语言规律在各模型上的读出等级（模板对照实验）；路线 N（神经元经模板确认入树）尚未接入`}>
            {tree ? <LangProgress tree={tree} model={model} onOpenItem={it => onGoSpace('lang', { item: it.id, type: it.type })} />
              : <Empty>{treeApi.error ? `语言谱图不在本机数据包里：${treeApi.error}` : '读取语言谱图…'}</Empty>}
          </Card>
          <RouteBoard onGo={onGo} />
        </>
      )}
    </div>
  );
}
