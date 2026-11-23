import { useState } from 'react';
import { API_BASE, getSession } from '../app/rdc_fusion/api.js';
import { readCaseReplay } from './workspaceRecordReplay.js';
import { workbenchHref } from './v8Util.js';

export default function ArchivedCaseResults({artifact,run,snapshot,onReplay,onOpenArchive,archiveEntry}) {
  const [result,setResult] = useState(null);
  const [page,setPage] = useState(0);
  const [message,setMessage] = useState('');
  const [busy,setBusy] = useState(false);
  const size = 12;

  async function load() {
    setMessage('');
    setBusy(true);
    try {
      const session = getSession();
      if (!session) throw Error('请登录中心后读取归档。');
      const response = await fetch(API_BASE+'/api/test-archive/objects/'+encodeURIComponent(artifact.sha256), {
        headers:{Authorization:'Bearer '+session},redirect:'error',cache:'no-store',signal:AbortSignal.timeout(120000),
      });
      if (!response.ok) throw Error('中心未返回逐题产物（HTTP '+response.status+'）。');
      const bytes = await response.arrayBuffer();
      const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),
        byte=>byte.toString(16).padStart(2,'0')).join('');
      if (bytes.byteLength!==artifact.size || hash!==artifact.sha256) throw Error('逐题产物与当前归档清单的长度或 SHA-256 不一致。');
      const lines = new TextDecoder('utf-8',{fatal:true}).decode(bytes).trim().split(/\r?\n/).filter(Boolean);
      const cases = lines.map(line=>JSON.parse(line));
      const groups = new Map();
      for (const row of cases) {
        if (typeof row?.correct!=='boolean') continue;
        const arm = typeof row.arm==='string' ? row.arm : '全部';
        const group = groups.get(arm) || {total:0,correct:0};
        group.total++;
        group.correct+=Number(row.correct);
        groups.set(arm,group);
      }
      setResult({bytes,cases,groups:[...groups]});
      setPage(0);
    } catch (error) {
      setMessage(error.message || '读取逐题产物失败。');
    } finally {
      setBusy(false);
    }
  }

  async function save() {
    try {
      if (window.ai2050Desktop?.saveCaseResults) {
        if (!await window.ai2050Desktop.saveCaseResults(result.bytes,artifact.sha256)) return;
      } else {
        const file = await window.showSaveFilePicker({suggestedName:'case_results.jsonl'});
        const output = await file.createWritable();
        await output.write(result.bytes);
        await output.close();
      }
      setMessage('已保存经哈希核验的逐题文件。');
    } catch (error) {
      if (error.name!=='AbortError') setMessage(error.message);
    }
  }

  return <details><summary>逐题结果与下载</summary>
    <p className="v8-note">清单 SHA-256 <code>{artifact.sha256}</code></p>
    {!result && <button type="button" disabled={busy} onClick={load}>{busy?'正在读取并核验…':'读取逐题结果'}</button>}
    {message && <p role="status">{message}</p>}
    {result && <>
      <p>{result.cases.length} 条逐题观测 · 产物哈希已核验</p>
      {result.groups.length>0 && <ul>{result.groups.map(([arm,value])=><li key={arm}>{arm} · {value.correct} / {value.total} 条判对</li>)}</ul>}
      {(window.ai2050Desktop?.saveCaseResults || window.showSaveFilePicker) && <button type="button" onClick={save}>选择位置保存逐题文件</button>}
      <div className="tr-page">
        <button type="button" disabled={!page} onClick={()=>setPage(page-1)}>上一组题</button>
        <span>{result.cases.length ? page+1 : 0} / {Math.ceil(result.cases.length/size)}</span>
        <button type="button" disabled={(page+1)*size>=result.cases.length} onClick={()=>setPage(page+1)}>下一组题</button>
      </div>
      {result.cases.slice(page*size,(page+1)*size).map((row,i)=>{
        const replay=onReplay&&readCaseReplay(row,{run,snapshot,artifact:artifact.sha256});
        const published=archiveEntry?.caseId===row.case_id&&archiveEntry.run===row.run_id&&
          archiveEntry.snapshot===row.model_snapshot&&archiveEntry.trace===row.trace_id ? archiveEntry : null;
        const locator=published&&new URLSearchParams({lens:'spatial',view:'graph',gview:'structure',snapshot,
          observeMode:'archive',observeArchive:published.id,sblock:published.location.layer,
          smodule:published.location.module,sindex:published.location.index});
        return <details key={i}><summary>
          {String(row?.case_id ?? row?.id ?? ('观测 '+(page*size+i+1)))}{row?.arm && ' · '+row.arm}
        </summary>
          {published?(onOpenArchive?<button type="button" className="fw-tbtn" onClick={()=>onOpenArchive(published.id,{scene:true})}>在 3D 中回放本题</button>:
            <a className="fw-tbtn" href={workbenchHref(locator.toString())}>在 3D 中回放本题</a>):
            replay?<button type="button" className="fw-tbtn" onClick={()=>setMessage(onReplay(replay)||'已打开模型分析。')}>在 3D 中回放本题</button>:
              <p className="v8-note">本题的逐题结果已核验。中心未提供可回放的原场。</p>}
          <pre>{JSON.stringify(row,null,2)}</pre>
        </details>;
      })}
    </>}
  </details>;
}
