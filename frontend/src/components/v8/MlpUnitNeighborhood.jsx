import { useMemo } from 'react';

import { useApi } from '../app/rdc_fusion/api.js';
import { NeuronUnitCardContent } from '../app/rdc_fusion/NeuronUnitCard.jsx';
import { ExtraPackPrompt } from '../open/DataBar.jsx';
import ResearchDisclosure from './ResearchDisclosure.jsx';
import { EVIDENCE_LABEL } from './researchFragmentUtil.js';
import { useB35Unit } from './B35RecursiveReport.jsx';

const TAIL = '/api/research/mlp-tail';
const PREVIEW_COUNT = 3;
const num = value => Number.isFinite(value) ? value.toPrecision(4) : '—';
export const MLP_LINK_COLORS = { upstream: '#a74914', downstream: '#1f6fb0', observation: '#71549a' };

function matchingObservation(data, block, unit, width, direction, snapshot) {
  const source = direction === 'incoming' ? block - 1 : block;
  return data?.kind === 'observation' && data.direction === direction && data.source === source && data.target === source + 1 &&
    data.unit === unit && data.full_units === width && data.values?.length === width &&
    (!data.model_snapshot || data.model_snapshot === snapshot) ? data : null;
}

function observedNeighbors(data) {
  const neighborBlock = data.direction === 'incoming' ? data.source : data.target;
  return data.values.map((score, index) => ({ block: neighborBlock, unit: index, score }))
    .filter(row => Number.isFinite(row.score) && row.score !== 0)
    .sort((a, b) => Math.abs(b.score) - Math.abs(a.score) || a.unit - b.unit)
    .slice(0, PREVIEW_COUNT);
}

export function useTailNeighborhood({ snapshot, block, unit, width, enabled }) {
  const wanted = enabled && Boolean(snapshot) && width === 9728 && block >= 28 && block <= 35;
  const overview = useApi(wanted ? TAIL : null);
  const atlas = wanted && overview.data?.model_snapshot === snapshot && overview.data.blocks?.includes(block) ? overview.data : null;
  const incoming = useApi(atlas?.blocks.includes(block - 1) ? `${TAIL}/relations/${block}/${unit}?direction=incoming&kind=rho-holdout` : null);
  const outgoing = useApi(atlas?.blocks.includes(block + 1) ? `${TAIL}/relations/${block}/${unit}?direction=outgoing&kind=rho-holdout` : null);
  const relations = useMemo(() => atlas ? [
    matchingObservation(incoming.data, block, unit, width, 'incoming', snapshot),
    matchingObservation(outgoing.data, block, unit, width, 'outgoing', snapshot),
  ].filter(Boolean) : [], [atlas, incoming.data, outgoing.data, block, unit, width, snapshot]);
  const observations = useMemo(() => relations.flatMap(observedNeighbors), [relations]);
  return { atlas, observations, relations, loading: wanted && (overview.loading || incoming.loading || outgoing.loading),
    errors: { upstream: incoming.error || overview.error, downstream: outgoing.error || overview.error } };
}

export function collectMlpNeighbors({ block, layers, width, links, paths, observations, recursive }) {
  const neighbors = new Map();
  function add(location, evidence) {
    if (!location || location.module !== 'mlp' || location.layer === block + 1 ||
        !Number.isInteger(location.layer) || location.layer < 1 || location.layer > layers ||
        !Number.isInteger(location.index) || location.index < 0 || location.index >= width) return;
    const key = `${location.layer}:${location.index}`;
    if (!neighbors.has(key)) neighbors.set(key, { key, location, evidence: [] });
    neighbors.get(key).evidence.push(evidence);
  }
  const locate = row => ({ layer: row.block + 1, module: 'mlp', index: row.unit });
  for (const link of links) {
    const level = EVIDENCE_LABEL[link.entry.body.evidence?.level] || '已有登记';
    add(link.location, { kind: 'registered', label: `${link.incoming ? '登记来源' : '登记去向'} · ${level}`, feature: link.other?.label });
  }
  for (const row of observations) add(locate(row), { kind: 'observation', label: '留出共同响应', score: row.score });
  for (const row of [...(paths?.upstream || []), ...(paths?.downstream || [])]) {
    add(locate(row), { kind: 'weight', label: paths.verified ? '相邻层权重候选' : '权重候选 · 快照待核验', score: row.score });
  }
  if (recursive) {
    for (const [field, rows] of Object.entries(recursive.upstream || {})) {
      for (const row of rows) add(locate({ block: 34, unit: row.unit }), {
        kind: 'weight', label: `B35 ${field === 'gate' ? 'Gate' : 'Value'} 权重来源候选`, score: row.cosine,
      });
    }
  }
  const rows = [...neighbors.values()];
  return { upstream: rows.filter(row => row.location.layer < block + 1), downstream: rows.filter(row => row.location.layer > block + 1) };
}

function MlpFeatureSummary({ model, snapshot, location, atlas, labels = [], current, b35Index }) {
  const block = location.layer - 1, unit = location.index;
  const inTail = Boolean(atlas?.blocks.includes(block));
  const tail = useApi(inTail ? `${TAIL}/units/${block}/${unit}` : null);
  const native = tail.data && tail.data.model_snapshot === snapshot && tail.data.block === block && tail.data.unit === unit ? tail.data : null;
  const b35 = useB35Unit(!current && !native && !tail.loading && block === 35 ? b35Index : null, unit);
  const archived = current?.b35Unit?.data || b35.data;
  const run = model?.census_run, key = model?.census_key;
  const shouldReadCard = !current && snapshot && key && run && !native && !tail.loading && !archived && !b35.loading;
  const census = useApi(shouldReadCard ? `/api/neurons/unit?${new URLSearchParams({ model_key: key, run, block, unit })}` : null);
  const result = current?.unit || census;
  const card = current?.card || (census.data?.schema === 'unit_card.v1' && census.data.model_snapshot === snapshot &&
    census.data.layer === block && census.data.unit === unit ? census.data : null);
  const explanation = card?.explanations?.at(-1);
  const stats = native?.statistics?.A || card?.census || archived;

  return <div className="v8-mlp-feature">
    {labels.length > 0 && <p className="v8-note">研究卡 · {labels.join(' / ')}</p>}
    {stats && <p>激活 A 均值 {num(stats.mean)} · 标准差 {num(stats.std)} · 范围 {num(stats.min)} ～ {num(stats.max)}</p>}
    {native?.candidate?.description && <p>{native.candidate.description}</p>}
    {native?.candidate?.repeat?.holdout && <p className="v8-note">候选词在留出正极位置命中 {native.candidate.repeat.holdout.cue_hit_count}/{native.candidate.repeat.holdout.n_pole_positions}，语义待确认。</p>}
    {explanation && <p>解释检验 · {explanation.grade} · {explanation.explanation}</p>}
    {card && !explanation && <p className="v8-note">自然语料普查 · {card.unit_label} · 解释未检验</p>}
    {!native && !card && archived && <p className="v8-note">B35 归档自然语料统计 · {archived.replicated ? '方向复现' : '未达到双留出方向复现'}</p>}
    {!stats && !labels.length && <p className="v8-note">{tail.loading || result.loading || b35.loading || current?.b35Unit?.loading ? '读取单元特征…' : '该单元暂无与当前快照匹配的特征记录。'}</p>}
    {!native && !card && result.missingPack && <ExtraPackPrompt pack={result.missingPack}/>}
    {current && native && <ResearchDisclosure title="自然语境与权重读出">
      {['positive', 'negative'].map(pole => <p key={pole}>{pole === 'positive' ? '正向' : '反向'} · {(native.parameter_readout?.[pole] || []).map(row => row.text).join(' · ')}</p>)}
      {(native.examples || []).slice(0, 2).map(row => <p key={`${row.document_index}:${row.token_position}`}>{row.prefix}</p>)}
      <p className="v8-note">{native.scope}</p>
    </ResearchDisclosure>}
    {current && card && <ResearchDisclosure title="完整普查与解释检验"><NeuronUnitCardContent data={card}/></ResearchDisclosure>}
  </div>;
}

function FullResponseRows({ relations }) {
  return relations.map(row => <div key={row.direction}>
    <h5>B{row.source} ↔ B{row.target} · {row.values.length.toLocaleString()} 个原生单元</h5>
    <p className="v8-note">每个数组位置对应相邻层的原生单元索引。{row.scope}</p>
    <pre className="v8-artifact-json">{JSON.stringify(row.values)}</pre>
  </div>);
}

export default function MlpUnitNeighborhood({ model, snapshot, selected, layers, card, unit, graph, tail, neighbors, candidate, b35Index, b35Unit, onLocate }) {
  const block = selected.layer - 1;
  const missingDirection = (block > 0 && !neighbors.upstream.length) || (block < layers - 1 && !neighbors.downstream.length);
  const labels = [...new Set(graph.cards.map(item => item.body.label).filter(Boolean))].slice(0, PREVIEW_COUNT);
  function rowCard(row) {
    const { location, evidence, key } = row;
    const relatedLabels = [...new Set(evidence.map(entry => entry.feature).filter(Boolean))];
    return <li key={key}>
      <button type="button" className="v6-linkbtn" onClick={() => onLocate(location)}>定位 B{location.layer - 1} · U{location.index}</button>
      <p className="v8-note">{evidence.map(entry => `${entry.label}${Number.isFinite(entry.score) ? ` ${num(entry.score)}` : ''}`).join(' · ')}</p>
      <MlpFeatureSummary model={model} snapshot={snapshot} location={location} atlas={tail.atlas} labels={relatedLabels} b35Index={b35Index}/>
    </li>;
  }
  return <div className="v8-mlp-neighborhood">
    <section aria-label="选中 MLP 单元特征">
      <h4>选中单元 · B{block} · U{selected.index}</h4>
      <MlpFeatureSummary key={`${snapshot}:${block}:${selected.index}`} model={model} snapshot={snapshot} location={selected} atlas={tail.atlas} labels={labels} current={{ card, unit, b35Unit }}/>
    </section>
    <section aria-label="前后层神经元特征">
      <h4>前后层神经元特征</h4>
      <p className="v8-note">留出共同响应按 |r| 每个方向展示前三项。共同响应、权重候选与已登记的干预证据分别标注。</p>
      <div className="v8-structure-neighbors">
        {[['upstream', '前层', block === 0], ['downstream', '后层', block === layers - 1]].map(([direction, label, boundary]) => {
          const rows = neighbors[direction];
          return <section key={direction} aria-label={`${label}单元`}>
            <h5>{label} · {rows.length} 个关联单元</h5>
            {rows.length ? <>
              <ul className="v8-structure-links">{rows.slice(0, PREVIEW_COUNT).map(rowCard)}</ul>
              {rows.length > PREVIEW_COUNT && <ResearchDisclosure title={`其余${label}单元（${rows.length - PREVIEW_COUNT}）`}>
                <ul className="v8-structure-links">{rows.slice(PREVIEW_COUNT).map(rowCard)}</ul>
              </ResearchDisclosure>}
            </> : <p className="v8-note">{boundary ? `该单元没有${label} MLP 层。` : tail.loading || candidate.loading ? '读取跨层研究记录…'
              : tail.errors[direction] || candidate.error || candidate.data?.reason || '当前快照暂无可显示的关联记录。'}</p>}
          </section>;
        })}
      </div>
      {missingDirection && candidate.missingPack && <ExtraPackPrompt pack={candidate.missingPack}/>}
      {tail.relations.length > 0 && <ResearchDisclosure key={`${snapshot}:${block}:${selected.index}`} title="完整留出共同响应数组">
        <FullResponseRows relations={tail.relations}/>
      </ResearchDisclosure>}
    </section>
  </div>;
}
