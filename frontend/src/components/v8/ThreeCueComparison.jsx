import { useState } from 'react';
import {archiveHeadCase, archivedHeadPrompt, selectionHead} from './GraphWorkspaceEvidence.jsx';
import {useWorkspaceResource} from './nativeObservationArchive.js';
import FirstRecordArchive from './FirstRecordArchive.jsx';
import './three_cue_comparison.css';

const RULES = ['姓名', '印刷 ROW 标签', '物理首行'];
const FIRST_RECORD = 2;
const pct = value => (100 * value).toFixed(1) + '%';
const headWinner = ({ mass }) => mass.indexOf(Math.max(...mass));
const isFirstRecordCounterexample = sample => sample.rule === FIRST_RECORD && sample.correct && headWinner(sample) !== sample.gold;

function CausalSummary({ run }) {
  if (run.causal) {
    return (
      <ul>
        {Object.entries(run.causal.branches).map(([key, result]) => (
          <li key={key}>{key.toUpperCase()} · {run.causal.n} 个独立家族 · {result.passes ? '匹配干预达到冻结门槛' : '匹配干预未达到冻结门槛'}。同剂量比例 {pct(result.dose_match)}，原头恢复 {result.head_exact ? '逐位一致' : '有差异'}。</li>
        ))}
      </ul>
    );
  }
  if (run.calibrationFailed) return <p>对照校准未通过，未在独立干预家族执行确认。</p>;
  if (!run.admission.length) return <p>Q/K 预测均未过门，未执行模型干预。两种交换只改题目文本。</p>;
  return <p>暂无已核验的独立干预结果。</p>;
}

export default function ThreeCueComparison(props) {
  const {data,error}=useWorkspaceResource('selection-cases');
  if(!data?.evidence?.runs?.length)return <p role="status">{error||'正在读取中心的三线索记录…'}</p>;
  return <Comparison key={props.initialRunId||data.evidence.runs[0].id} data={data} {...props}/>;
}

function Comparison({data,initialRunId='',initialCaseId='',onOpenCase}) {
  const evidence=data.evidence;
  const [initial] = useState(() => {
    const selectedRun=evidence.runs.find(run=>run.id===initialRunId)||evidence.runs.at(-1);
    const family=initialCaseId?selectedRun.families.find(item=>item.cases.some(row=>row.id===initialCaseId)):null;
    return {runId:selectedRun.id,familyId:family?String(family.id):null,sample:family?.cases.find(row=>row.id===initialCaseId)};
  });
  const [runId, setRunId] = useState(initial.runId);
  const [familyId, setFamilyId] = useState(initial.familyId);
  const [rule, setRule] = useState(initial.sample?.rule??1);
  const [labelSwap, setLabelSwap] = useState(Boolean(initial.sample?.labelSwap));
  const [positionSwap, setPositionSwap] = useState(Boolean(initial.sample?.positionSwap));
  const [counterexampleId, setCounterexampleId] = useState(null);
  const run = evidence.runs.find(item => item.id === runId);
  const counterexamples = run.families.flatMap(family => family.cases.filter(isFirstRecordCounterexample).map(sample => ({ family, sample })));
  const chosen = counterexamples.find(({ sample }) => sample.id === counterexampleId);
  const family = chosen?.family ?? run.families.find(item => String(item.id) === familyId) ?? run.families[0];
  const sample = chosen?.sample ?? family.cases.find(item => item.rule === rule && Boolean(item.labelSwap) === labelSwap && Boolean(item.positionSwap) === positionSwap);
  const nativeCase=run.id===selectionHead(data)?.run&&archiveHeadCase(data,sample.id);
  const originalPrompt=nativeCase?archivedHeadPrompt(data,sample.id):null;
  const headRecord = headWinner(sample);
  const outside = Math.max(0, 1 - sample.mass.reduce((sum, value) => sum + value, 0));
  const ruleCounts = [0, 0, 0];
  const ruleCorrect = [0, 0, 0];
  for (const { cases } of run.families) {
    for (const item of cases) {
      ruleCounts[item.rule]++;
      ruleCorrect[item.rule] += Number(item.correct);
    }
  }

  return (
    <section className="cue-compare" aria-label="三线索选源">
      <header>
        <h4>三线索选源</h4>
        <p className="v8-note">{run.model} {run.precision} · B{run.readout.layer}/H{run.readout.head} · {run.readout.position}。下方回放本机复核的冻结数据，上方 3D 显示当前模型对象。</p>
      </header>
      <div className="cue-controls">
        <label>证据批次<select value={runId} onChange={event => { setRunId(event.target.value); setFamilyId(null); setCounterexampleId(null); }}>
          {evidence.runs.map(item => <option key={item.id} value={item.id}>{item.title}</option>)}
        </select></label>
        <label><input type="checkbox" checked={Boolean(chosen)} disabled={!counterexamples.length}
          onChange={event => setCounterexampleId(event.target.checked ? (isFirstRecordCounterexample(sample) ? sample.id : counterexamples[0].sample.id) : null)} />只看首行反例</label>
        {chosen ? <label>首行反例<select aria-label="首行反例" value={sample.id} onChange={event => setCounterexampleId(event.target.value)}>
          {counterexamples.map(({ family, sample }) => <option key={sample.id} value={sample.id}>{sample.id} · {family.archive}</option>)}
        </select></label> : <label>冻结家族<select value={String(family.id)} onChange={event => setFamilyId(event.target.value)}>
          {run.families.map((item, index) => <option key={item.id} value={String(item.id)}>#{String(index).padStart(3, '0')} · {item.archive}</option>)}
        </select></label>}
      </div>
      {chosen ? <p className="v8-note">{counterexamples.length} 个已存首行反例。模型答对，H36 把其他记录排在首位。当前样本{sample.labelSwap ? '已' : '未'}交换印刷标签，{sample.positionSwap ? '已' : '未'}交换首尾记录。</p> : <>
        <div className="aw-seg cue-rules" role="group" aria-label="题目规则">
          {RULES.map((label, index) => <button key={label} type="button" aria-pressed={rule === index} onClick={() => setRule(index)}>{label}</button>)}
        </div>
        <div className="cue-controls cue-swaps">
          <label><input type="checkbox" checked={labelSwap} onChange={event => setLabelSwap(event.target.checked)} />交换印刷标签</label>
          <label><input type="checkbox" checked={positionSwap} onChange={event => setPositionSwap(event.target.checked)} />交换首尾记录</label>
        </div>
      </>}
      <p className="cue-question">{sample.question}</p>
      <dl className="cue-output" aria-live="polite">
        <div><dt>题目正确答案</dt><dd><code>{family.codes[sample.gold]}</code></dd></div>
        <div><dt>模型实际首 token</dt><dd><code>{sample.output.trim() || JSON.stringify(sample.output)}</code> <small>{sample.correct ? '正确' : '有误'}</small></dd></div>
        <div><dt>头最高的记录</dt><dd><code>{family.codes[headRecord]}</code></dd></div>
      </dl>
      {onOpenCase&&(nativeCase?<>
        <button type="button" className="fw-tbtn primary" onClick={()=>onOpenCase(sample.id)}>在 3D 中观察本题</button>
        {originalPrompt&&<details><summary>查看原始模型输入</summary><pre className="v8-artifact-json" lang="en">{originalPrompt}</pre></details>}
      </>:
        <p className="v8-note">本题提供输出与按记录分组的注意力汇总。同一条件的 Head 通道原场未发布。</p>)}
      <div className="cue-table-wrap" tabIndex={0} role="region" aria-label="记录与注意力表">
        <table>
          <thead><tr><th scope="col">物理行</th><th scope="col">ROW</th><th scope="col">姓名</th><th scope="col">代码</th><th scope="col">指向线索</th><th scope="col">H36 注意力</th></tr></thead>
          <tbody>
            {sample.order.map((record, position) => {
              const cue = sample.targets.indexOf(record);
              const mass = sample.mass[record];
              return (
                <tr key={record} data-gold={record === sample.gold} data-head-first={record === headRecord}>
                  <td>{position + 1}</td><td>{sample.labels[record]}</td><td>{family.names[record]}</td><td><code>{family.codes[record]}</code></td>
                  <td>{cue >= 0 ? RULES[cue] : '—'}</td>
                  <td><span className="cue-mass"><meter aria-label="记录注意力质量" min={0} max={1} value={mass} />{pct(mass)}</span></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="v8-note">表外 token 占 {pct(outside)}。五条记录的注意力按原值相加，未在表内重新归一化。</p>
      <div className="cue-evidence">
        <section>
          <h5>可读出</h5>
          <p>从操作前状态读出题目规则的准确率为 {pct(run.probe)}。各规则下，H36 排对来源与模型答对的比例：</p>
          <ul>{run.natural.map((item, index) => <li key={item.rule}>{RULES[index]} · 头 {pct(item.all_five_gold)} / 答案 {pct(ruleCorrect[index] / ruleCounts[index])}</li>)}</ul>
        </section>
        <section>
          <h5>可预测</h5>
          <p>每支以 512 个原生输入坐标重建选源，与原头五选一比较。通过条件含 ≥90% 一致率及误差对照。</p>
          <ul>{Object.entries(run.predictor).map(([key, result]) => <li key={key}>{key.toUpperCase()} {pct(result.five_way_agreement)} · {result.passed ? '通过' : '未通过'}{run.prior && <small>（同批旧坐标 {pct(run.prior[key])}）</small>}</li>)}</ul>
        </section>
        <section>
          <h5>因果作用</h5>
          <CausalSummary run={run} />
        </section>
      </div>
      <FirstRecordArchive />
      <details className="cue-details">
        <summary>实验范围、坐标与复核来源</summary>
        <p>{run.scope} 本批确认集 {run.families.length} 个家族 / {run.totals.n} 题，模型实际首 token 正确 {run.totals.top1}/{run.totals.n}。</p>
        <p>样本 <code>{sample.id}</code> · 输入 {sample.inputTokens} token · 首输出 token ID {sample.tokenId}</p>
        {Object.entries(run.coordinates).map(([key, columns]) => {
          const scalar = run.scalarRefs[key];
          return <p key={key}>{key.toUpperCase()} 原生输入列 {columns.join(', ')}{scalar && <> · <code>{key}_proj.weight[{scalar.row}, {scalar.col}]</code> = {scalar.weight}</>}</p>;
        })}
        <p>模型快照 <code>{run.snapshot}</code></p>
        <ul>{Object.entries(run.provenance).map(([file, digest]) => <li key={file}><code>{run.id}/{file}</code> · SHA‑256 <code>{digest}</code></li>)}</ul>
      </details>
    </section>
  );
}
