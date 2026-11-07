/* 主张卡草稿编辑器（H0）。以 MA-H0 的主张卡为模板，在本机编辑并下载 JSON；不写入服务器。
   正式主张卡要随预注册冻结（scripts/mainline_h*.py 的 prereg 步骤），草稿本身不算冻结。 */
import { useState } from 'react';

import prereg from '../../../../research/mainline/H0/MA-H0-v3.prereg.json';
import './client_guide.css';

const BASE = prereg.claim_card;
const blank = () => ({ id: '', name: '', mechanism: '', behavior: '', later: '' });

export default function ClaimDraft() {
  const [card, setCard] = useState(() => ({
    claim: BASE.claim, model: BASE.model, readout: BASE.readout,
    factors: { ...BASE.factors }, hypotheses: BASE.hypotheses.map(h => ({ ...h })), decisions: BASE.decisions.map(d => [...d]),
  }));
  const [msg, setMsg] = useState('');
  const set = patch => setCard(c => ({ ...c, ...patch }));
  const setHyp = (i, k, v) => set({ hypotheses: card.hypotheses.map((h, j) => j === i ? { ...h, [k]: v } : h) });
  const setDec = (i, k, v) => set({ decisions: card.decisions.map((d, j) => j === i ? d.map((x, m) => m === k ? v : x) : d) });
  const distinct = card.hypotheses.filter(h => h.name.trim() && h.behavior.trim()).length;
  const download = () => {
    const doc = { schema: 'claim_card_draft.v1', created: new Date().toISOString(), based_on: 'MA-H0-v3 主张卡',
      note: '草稿：未冻结、未进入研究记录。正式主张卡随预注册一起冻结。', ...card };
    const url = URL.createObjectURL(new Blob([JSON.stringify(doc, null, 1)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url; a.download = `claim_card_draft_${doc.created.slice(0, 10)}.json`; a.click();
    URL.revokeObjectURL(url);
    setMsg('已下载草稿。');
  };
  return <div aria-label="主张卡草稿">
    <p className="v8-note">从 MA-H0 的主张卡起步，改写成新的问题。至少写两个可区分的假设，并写明每种结果会改变哪项决定。草稿只保存在你的电脑上，不写入服务器，也不算冻结。</p>
    <label className="v8-note">主张<textarea className="h2-text" rows={2} value={card.claim} onChange={e => set({ claim: e.target.value })}/></label>
    <label className="v8-note">模型<textarea className="h2-text" rows={1} value={card.model} onChange={e => set({ model: e.target.value })}/></label>
    <label className="v8-note">读出位置<textarea className="h2-text" rows={2} value={card.readout} onChange={e => set({ readout: e.target.value })}/></label>
    <h4>四类因素</h4>
    {Object.entries(card.factors).map(([k, v]) => <label className="v8-note" key={k}>{k}
      <textarea className="h2-text" rows={1} value={v} onChange={e => set({ factors: { ...card.factors, [k]: e.target.value } })}/></label>)}
    <h4>可区分的假设 <span className={'mp-tag ' + (distinct >= 2 ? 'ok' : 'warn')}>{distinct} 个已填写名称与行为预测</span></h4>
    {card.hypotheses.map((h, i) => <div className="cd-hyp" key={i}>
      <div className="cg-grid2">
        <label className="v8-note">编号<textarea className="h2-text" rows={1} value={h.id} onChange={e => setHyp(i, 'id', e.target.value)}/></label>
        <label className="v8-note">名称<textarea className="h2-text" rows={1} value={h.name} onChange={e => setHyp(i, 'name', e.target.value)}/></label>
      </div>
      <label className="v8-note">机制<textarea className="h2-text" rows={2} value={h.mechanism} onChange={e => setHyp(i, 'mechanism', e.target.value)}/></label>
      <label className="v8-note">行为上的预测<textarea className="h2-text" rows={2} value={h.behavior} onChange={e => setHyp(i, 'behavior', e.target.value)}/></label>
      <label className="v8-note">后续阶段怎样区分<textarea className="h2-text" rows={2} value={h.later} onChange={e => setHyp(i, 'later', e.target.value)}/></label>
      <button type="button" className="v6-linkbtn" onClick={() => set({ hypotheses: card.hypotheses.filter((_, j) => j !== i) })}>删除这个假设</button>
    </div>)}
    <button type="button" className="fw-tbtn" onClick={() => set({ hypotheses: [...card.hypotheses, blank()] })}>添加假设</button>
    <h4>结果 → 决定</h4>
    {card.decisions.map((d, i) => <div className="cg-grid2" key={i}>
      <label className="v8-note">如果<textarea className="h2-text" rows={2} value={d[0]} onChange={e => setDec(i, 0, e.target.value)}/></label>
      <label className="v8-note">那么<textarea className="h2-text" rows={2} value={d[1]} onChange={e => setDec(i, 1, e.target.value)}/></label>
    </div>)}
    <button type="button" className="fw-tbtn" onClick={() => set({ decisions: [...card.decisions, ['', '']] })}>添加一条</button>
    <div className="v6-toolbar"><button type="button" className="fw-tbtn primary" disabled={distinct < 2} onClick={download}>下载主张卡草稿</button>
      <span className="v8-note" role="status">{distinct < 2 ? '至少填写两个假设的名称和行为预测' : msg}</span></div>
  </div>;
}
