import { useState } from 'react';
import { authJson } from '../app/rdc_fusion/api.js';

const api = (path, opts = {}) => authJson('/api/test-archive' + path, { ...opts, redirect: 'error' });
const allowed = new Set(['script', 'questions', 'protocol', 'case_results', 'results', 'log', 'reproducibility', 'native_field', 'evidence']);
const fileHash = async blob => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', await blob.arrayBuffer())),
  b => b.toString(16).padStart(2, '0')).join('');

function ResearchResult({ summary }) {
  let result;
  try { result = typeof summary === 'string' ? JSON.parse(summary) : summary; } catch { return null; }
  if (result?.stage === 'greedy_color' && result.outcomes?.base && result.outcomes?.cut) {
    return <div><p>真实续写至同异答案词。原模型答对 {result.outcomes.base.correct}/{result.unique_prompts} 题，
      阻断后 {result.outcomes.cut.changed} 题改变答案，{result.outcomes.base.unresolved} 题无法判读。</p></div>;
  }
  if (result?.stage === 'color_reuse' && result.comparison_total && result.color_delta) {
    const group = result.comparison_total;
    return <div><p>按首位置两个小写候选词的得分，Qwen3-14B 在 {group.n} 道留出比较题中答对 {group.correct} 道。跨用途因果判据{result.supported ? '通过' : '未通过'}。</p>
      <table><thead><tr><th>参照物颜色</th><th>同答相对异答的分差变化（logit）</th></tr></thead>
        <tbody>{[['red','红'],['yellow','黄'],['blue','蓝']].map(([key,label]) => <tr key={key}><td>{label}</td><td>{result.color_delta[key]?.mean?.toFixed(3)}</td></tr>)}</tbody></table></div>;
  }
  if (result?.stage === 'diagnostic' && result.cells && result.conflict_pairs) {
    const tasks = [
      ['old_only', '只看旧名单，报旧行'],
      ['old_with_ledger', '加上新表，报旧行'],
      ['explicit_slot', '直接给行号，查代号'],
      ['full_old', '完整题，按原预订取值'],
      ['full_current', '完整题，按当前姓名取值'],
    ];
    const rate = s => s ? s.correct + '/' + s.n + '（' + (100 * s.correct / s.n).toFixed(1) + '%）' : '—';
    return <div>
      <p>Qwen3-14B · {result.families} 个家族，{result.cases} 道题。</p>
      <table><thead><tr><th>任务</th><th>全部</th><th>新旧行一致</th><th>新旧行冲突</th></tr></thead>
        <tbody>{tasks.map(([k, label]) => <tr key={k}><td>{label}</td><td>{rate(result.cells[k]?.all)}</td>
          <td>{rate(result.cells[k]?.agree)}</td><td>{rate(result.cells[k]?.conflict)}</td></tr>)}</tbody></table>
      <p>在 {2 * result.families} 对冲突题中，{result.conflict_pairs.both_correct} 对随指令正确换答。</p>
    </div>;
  }
  if (!result?.gate?.cells) return null;
  const percent = n => (100 * n).toFixed(1) + '%';
  return <div><p>Qwen3-14B · {result.families} 个家族，{result.cases} 道题。{result.gate.passed ? '自然运行通过预注册门槛。' : '自然运行未通过预注册门槛。'}</p>
    <table><thead><tr><th>原行 / 现行</th><th>按原行答对</th><th>按现行答对</th></tr></thead>
    <tbody>{['00','01','10','11'].map(k=><tr key={k}><td>{k[0]==='0'?'上':'下'} / {k[1]==='0'?'上':'下'}</td>
      <td>{percent(result.gate.cells[k+'0'])}</td><td>{percent(result.gate.cells[k+'1'])}</td></tr>)}</tbody></table>
    <p>两行冲突时，B26H36 在同档两行中选对的比例：原行规则 {percent(result.gate.conflict_route['0'])}，现行规则 {percent(result.gate.conflict_route['1'])}。</p>
    {result.confirm && <p>确认家族的双向选择性选源判据{result.confirm.conditional_route_supported?'通过':'未通过'}，输出判据{result.confirm.output_supported?'通过':'未通过'}。</p>}
  </div>;
}

async function putFile(file, sha) {
  const up = await api('/uploads', { method: 'POST', body: { sha256: sha, size: file.size } });
  if (up.available) return;
  const path = '/uploads/' + encodeURIComponent(up.upload_id);
  let offset = up.offset;
  if (!Number.isInteger(offset) || offset < 0 || offset > file.size || !(up.chunk_bytes > 0)) throw Error('上传偏移无效');
  while (offset < file.size) {
    const end = Math.min(file.size, offset + up.chunk_bytes);
    const ack = await api(path + '?offset=' + offset, { method: 'PUT', rawBody: file.slice(offset, end), timeoutMs: 180000 });
    if (ack.offset !== end) throw Error('上传偏移与发送内容不一致');
    offset = end;
  }
  await api(path + '/finish', { method: 'POST', body: {} });
}

export default function RoleRereadArchive({ row }) {
  const [message, setMessage] = useState('');
  const [files, setFiles] = useState(null);
  const [busy, setBusy] = useState(false);
  const [exportText, setExportText] = useState('');
  const [archivedSummary, setArchivedSummary] = useState(null);
  const stage = row.plan.analysis.selection.stage;
  const runPath = '/runs/' + encodeURIComponent(row.id);
  async function action(fn) {
    setBusy(true);
    setMessage('');
    try { await fn(); } catch (error) { setMessage(error.message); }
    finally { setBusy(false); }
  }
  async function exportRegistration() {
    const [run, preflight] = await Promise.all([api(runPath), api('/preflights/' + encodeURIComponent(row.preflightId))]);
    if (run.id !== row.id || run.preflight_id !== preflight.preflight_id || !run.plan) throw Error('登记与预检标识不符');
    setExportText(JSON.stringify({ run, preflight }));
    setMessage('已从中心读回登记和原始查重回执。可保存交接 JSON 到本机研究目录。');
  }
  async function readReceipt(expectedManifest) {
    const receipt = await api(runPath + '/receipt?verify=true', {timeoutMs:180000});
    if (receipt.run_id !== row.id || (expectedManifest && receipt.manifest_sha256 !== expectedManifest) || !receipt.readback_verified)
      throw Error('读回回执与运行或本机清单不一致');
    setExportText(JSON.stringify({ receipt }));
    const stored = await api(runPath);
    setArchivedSummary(stored.final?.summary);
    setMessage('中心已校验复核包。可保存回执以便本机逐项核验。');
  }
  async function saveShown() {
    const suffix = JSON.parse(exportText).receipt ? '-receipt.json' : '-desktop.json';
    const file = await window.showSaveFilePicker({ suggestedName: stage + suffix, types: [{description:'JSON', accept:{'application/json':['.json']}}] });
    const target = await file.createWritable();
    await target.write(exportText);
    await target.close();
    setMessage('已保存读回的中心记录。');
  }
  async function upload() {
    const list = new Map(Array.from(files || [], file => [file.webkitRelativePath.split('/').slice(1).join('/'), file]));
    const get = (name) => { const f = list.get(name); if (!f) throw Error('复核包缺少 ' + name); return f; };
    const spec = JSON.parse(await get('submission.json').text());
    if (spec.run_id !== row.id || spec.preflight_id !== row.preflightId || !Array.isArray(spec.final?.artifacts)
        || !['completed','failed','interrupted'].includes(spec.final.status)) throw Error('所选复核包与当前运行不匹配');
    const review = JSON.parse(await get('reproducibility.json').text());
    const proof = await api('/preflights/' + encodeURIComponent(row.preflightId));
    if (review.server_run_id !== row.id || await fileHash(get('protocol.json')) !== proof.plan_sha256
        || await fileHash(get('test.py')) !== row.plan.entrypoint_sha256
        || await fileHash(get('questions.json')) !== row.plan.dataset.questions_file_sha256) throw Error('方案、题目或入口哈希不符');
    const names = new Set();
    for (const entry of spec.final.artifacts) {
      if (!/^(test\.py|questions\.json|protocol\.json|reproducibility\.json|output\.log|case_results\.jsonl|conclusion\.json|research-scripts\.zip|candidate\.json|native-\d{4}\.npz)$/.test(entry.name)
          || names.has(entry.name) || entry.availability !== 'uploaded' || !allowed.has(entry.role)
          || !['core','important'].includes(entry.retention)) throw Error('复核清单存在未识别或重复的产物');
      names.add(entry.name);
      const file = get(entry.name);
      if (file.size !== entry.size || await fileHash(file) !== entry.sha256) throw Error('产物哈希或长度不符 ' + entry.name);
    }
    let n = 0;
    for (const entry of spec.final.artifacts) {
      setMessage('正在归档 ' + (++n) + ' / ' + names.size + ' · ' + entry.name);
      await putFile(get(entry.name), entry.sha256);
    }
    await api(runPath + '/finalize', {method:'POST',body:spec.final,timeoutMs:180000});
    await readReceipt(spec.manifest_sha256);
  }
  return <details open><summary>{row.plan.family_ids.includes('COLOR-NAMING-COMPARISON') ? '颜色命名与比较实验的执行与归档' : '早期行号与晚期姓名实验的执行与归档'}</summary>
    <ResearchResult summary={archivedSummary || row.result}/>
    <button type="button" disabled={busy} onClick={()=>action(exportRegistration)}>读回本次运行和查重回执</button>
    {(row.archive==='server_verified' || archivedSummary) && <button type="button" disabled={busy}
      onClick={()=>action(()=>readReceipt())}>重新读回归档回执</button>}
    {exportText && <><textarea aria-label="中心交接 JSON" readOnly value={exportText} rows={2}/>
      <button type="button" disabled={busy} onClick={()=>action(saveShown)}>保存中心交接 JSON…</button></>}
    <label>本机复核包目录<input type="file" webkitdirectory="" multiple disabled={busy} onChange={e=>setFiles(e.target.files)}/></label>
    <button type="button" disabled={!files?.length || busy} onClick={()=>action(upload)}>核验并上传本次运行列出的产物</button>
    {message && <p role="status">{message}</p>}
  </details>;
}
