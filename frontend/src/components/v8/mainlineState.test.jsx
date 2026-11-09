import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { atMainlineRevision, bundledState, MainlineContext, readSnapshot } from './data/mainlineState.js';
import { LockboxTable, MainlineCard } from './MainlineProgress.jsx';
import { TestsPanel } from './AssocConsole.jsx';
import { MainlineStatus } from './MainlineProvider.jsx';
import release from '../../../../shared/dist_protocol/mainline_release.json';
import MainlineReport from './MainlineReport.jsx';

async function envelope(state) {
  const record = JSON.stringify(state);
  const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(record));
  return { schema: 'mainline_snapshot.v1', sequence: state.sequence, record,
    revision: [...new Uint8Array(hash)].map(x => x.toString(16).padStart(2, '0')).join('') };
}

describe('中心研究状态与同一版本页面', () => {
  it('列表旧版本不能冒充当前版本', () => {
    const resource={data:{mainline_revision:'old',workpackages:[]},reload:()=>{}};
    expect(atMainlineRevision(resource,'new').data).toBeNull();
    expect(atMainlineRevision(resource,'old')).toBe(resource);
    expect(atMainlineRevision(resource,null)).toBe(resource);
  });
  it('H3b真实剂量、负值和限制能在空间内直接阅读', () => {
    const doc = Object.values(release.artifacts).find(d => d.schema === 'mainline_run_report.v1' && d.run_id === 'MA-H3b-v1');
    const html = renderToStaticMarkup(<MainlineReport doc={doc}/>);
    for(const text of ['真实头预算曲线','-0.0223','0.0590','不能排除其他组合','未提供独立图谱预测']) expect(html).toContain(text);
  });
  it('只有保留数据的随机或匹配对照明确显示缺测，不使报告崩溃', () => {
    const doc = structuredClone(Object.values(release.artifacts).find(d => d.schema === 'mainline_run_report.v1' && d.run_id === 'MA-H3b-v1'));
    doc.report.curves.exact = doc.report.curves.random0;
    const html = renderToStaticMarkup(<MainlineReport doc={doc}/>);
    expect(html).toContain('未测');
    expect(html).toContain('未记录');
  });
  it('确认集开封、H3b负结果、尚未测量的独立预测分别显示', () => {
    const html = renderToStaticMarkup(<><MainlineCard/><LockboxTable/><TestsPanel/></>);
    expect(html).toContain('已开封');
    expect(html).toContain('未达 0.6');
    expect(html).toContain('0.9885');
    expect(html).toContain('不是独立预测或破解比例');
    expect(bundledState.view.SCORES.every(s => s.value === null)).toBe(true);
  });
  it('进度、开封清单、方法检验从同一个中心版本切换', () => {
    const state = structuredClone(bundledState);
    state.view.CLAIMS[0].run = '中心新结果一起更新';
    state.view.LOCKBOXES[0].note = '中心新开封记录';
    state.view.METHOD_RESULTS[0].label = '中心新范围检验';
    const html = renderToStaticMarkup(<MainlineContext.Provider value={{...state.view,state,source:'server',revision:'abc',refresh:()=>{}}}>
      <MainlineStatus/><MainlineCard/><LockboxTable/><TestsPanel/>
    </MainlineContext.Provider>);
    for (const text of ['中心研究记录','中心新结果一起更新','中心新开封记录','中心新范围检验']) expect(html).toContain(text);
  });
  it('断网时明确显示随包来源', () => {
    expect(renderToStaticMarkup(<MainlineStatus/>)).toContain('尚未接通中心记录');
  });
  it('校验完整记录哈希和模型身份', async () => {
    const e = await envelope(bundledState);
    expect((await readSnapshot(e)).state).toEqual(bundledState);
    await expect(readSnapshot({...e,record:e.record+' '})).rejects.toThrow('校验失败');
    await expect(readSnapshot({...e,sequence:99})).rejects.toThrow('身份不一致');
  });
  it('拒绝版本倒退和同版本内容冲突', async () => {
    const newer = structuredClone(bundledState);
    newer.sequence += 1;
    const accepted = await readSnapshot(await envelope(newer));
    await expect(readSnapshot(await envelope(bundledState),accepted)).rejects.toThrow('倒退');
    newer.scientific_note = '不能静默改同版本';
    await expect(readSnapshot(await envelope(newer),accepted)).rejects.toThrow('冲突');
  });
  it('即使版本递增也不能将已开封材料重新封存', async () => {
    const previous = await readSnapshot(await envelope(bundledState));
    const next = structuredClone(bundledState);
    next.sequence += 1;
    next.view.LOCKBOXES.find(b => b.id === next.custody[0].id).state = 'sealed';
    next.custody[0].state = 'sealed';
    await expect(readSnapshot(await envelope(next),previous)).rejects.toThrow('不能重新');
  });
});
