/* 研究工作台的「问 AI」侧栏（ai_assist.v1）：只在桌面客户端出现。附带页面登记的上下文（useAiContext），发送前显示
   标题、字符数与目标主机；回答流式显示、可停止，每条都标「AI 回答 · 未检验」。请求由主进程发出，Key 不进页面。 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { aiBridge, contextChars, useCurrentAiContext } from './aiContext.js';
import './ai.css';

let seq = 0;
const newId = () => `q${Date.now().toString(36)}${(seq += 1)}`;

export default function AiChatDrawer({ onSettings }) {
  const bridge = aiBridge();
  const ctx = useCurrentAiContext();
  const [open, setOpen] = useState(false);
  const [cfg, setCfg] = useState(null);
  const [cfgErr, setCfgErr] = useState('');
  const [profileId, setProfileId] = useState('');
  const [attach, setAttach] = useState(true);
  const [msgs, setMsgs] = useState([]);             // {role:'user'|'assistant'|'note', content, meta?, error?, pending?}
  const [input, setInput] = useState('');
  const [busyId, setBusyId] = useState('');
  const lastCtx = useRef(null);
  const listRef = useRef(null);

  const loadCfg = useCallback(() => {
    if (!bridge) return;
    bridge.config().then(c => { setCfg(c); setCfgErr(''); setProfileId(id => (c.profiles.some(p => p.id === id) ? id : c.default_id)); })
      .catch(e => setCfgErr(e.message));
  }, [bridge]);
  useEffect(() => { if (open) loadCfg(); }, [open, loadCfg]);

  // 流式增量：只追加到正在进行的那一条
  useEffect(() => {
    if (!bridge) return undefined;
    return bridge.onEvent(ev => {
      if (ev.type !== 'delta') return;
      setMsgs(ms => ms.map(m => (m.reqId === ev.id && m.pending ? { ...m, content: m.content + ev.text } : m)));
    });
  }, [bridge]);

  // 对话进行中换了页面：插一条提示，让读者知道之后的问题附带的是新页面
  useEffect(() => {
    if (!msgs.length || !ctx || lastCtx.current === null || lastCtx.current === `${ctx.kind}:${ctx.id}`) return;
    setMsgs(ms => [...ms, { role: 'note', content: `上下文已变为：${ctx.title}` }]);
    lastCtx.current = `${ctx.kind}:${ctx.id}`;
  }, [ctx, msgs.length]);

  useEffect(() => { listRef.current?.scrollTo?.({ top: listRef.current.scrollHeight }); }, [msgs]);

  if (!bridge) return null;
  const profile = cfg?.profiles.find(p => p.id === profileId);
  const useCtx = attach && ctx ? ctx : null;
  const chars = contextChars(useCtx);

  const send = async () => {
    const q = input.trim();
    if (!q || busyId) return;
    const id = newId();
    const history = msgs.filter(m => (m.role === 'user' || m.role === 'assistant') && !m.error && m.content).map(m => ({ role: m.role, content: m.content }));
    lastCtx.current = ctx ? `${ctx.kind}:${ctx.id}` : '';
    setMsgs(ms => [...ms, { role: 'user', content: q, ctxTitle: useCtx?.title }, { role: 'assistant', content: '', pending: true, reqId: id }]);
    setInput('');
    setBusyId(id);
    try {
      const r = await bridge.chat({ id, profile_id: profileId, messages: [...history, { role: 'user', content: q }],
        context: useCtx && { kind: useCtx.kind, id: useCtx.id, title: useCtx.title, data: useCtx.data } });
      setMsgs(ms => ms.map(m => (m.reqId === id ? { ...m, content: r.text, pending: false, meta: r } : m)));
      loadCfg();
    } catch (e) {
      setMsgs(ms => ms.map(m => (m.reqId === id ? { ...m, pending: false, error: e.message } : m)));
    }
    setBusyId('');
  };

  return (
    <>
      {!open && <button type="button" className="ai-fab" onClick={() => setOpen(true)} title="问 AI：就当前页面的数据提问">问 AI</button>}
      {open && (
        <aside className="ai-drawer" aria-label="问 AI">
          <header className="ai-dh">
            <b>问 AI</b>
            {cfg && cfg.profiles.length > 0 && (
              <select value={profileId} onChange={e => setProfileId(e.target.value)} aria-label="使用的模型">
                {cfg.profiles.map(p => <option key={p.id} value={p.id} disabled={!p.has_key}>{p.name} · {p.model}{p.has_key ? '' : '（未设 Key）'}</option>)}
              </select>
            )}
            {msgs.length > 0 && !busyId && <button type="button" className="fw-tbtn" onClick={() => { setMsgs([]); lastCtx.current = null; }}>清空</button>}
            <button type="button" className="fw-tbtn" onClick={() => setOpen(false)} aria-label="关闭">×</button>
          </header>

          {cfgErr && <div className="ai-err block">{cfgErr}</div>}
          {cfg && !cfg.profiles.length && (
            <div className="ai-empty">还没有配置 AI 模型。
              <button type="button" className="fw-tbtn primary" onClick={onSettings}>去「研发 → AI 助手」配置</button>
            </div>
          )}

          <div className="ai-list" ref={listRef}>
            {!msgs.length && cfg?.profiles.length > 0 && (
              <div className="ai-hint">
                {ctx ? <>可以问：「这一页的数据说明了什么？」「这个结果可能有哪些简单解释？」「下一步该做什么检验？」</>
                  : <>在空间里打开一个语言项或神经元单元，或在研发 → 神经元测试里打开一份报告，AI 就能看到那一页的数据。</>}
              </div>
            )}
            {msgs.map((m, i) => (
              m.role === 'note' ? <div key={i} className="ai-note-line">{m.content}</div> : (
                <div key={i} className={`ai-msg ${m.role}`}>
                  {m.role === 'user' && m.ctxTitle && <small className="ai-msg-ctx">附带：{m.ctxTitle}</small>}
                  <div className="ai-msg-text">{m.content || (m.pending ? '…' : '')}</div>
                  {m.error && <div className="ai-err">{m.error}</div>}
                  {m.role === 'assistant' && !m.pending && !m.error && m.meta && (
                    <small className="ai-msg-meta">
                      {m.meta.profile} · {m.meta.model} · {(m.meta.usage.input + m.meta.usage.output).toLocaleString()} tokens{m.meta.usage.estimated ? '（估计）' : ''} · {(m.meta.ms / 1000).toFixed(1)} s
                      {m.meta.context_truncated && ' · 页面数据已截断'}
                      <br /><b>AI 回答 · 未检验，不是证据</b>
                    </small>
                  )}
                </div>
              )
            ))}
          </div>

          <div className="ai-compose">
            <label className="ai-ctx">
              <input type="checkbox" checked={attach && Boolean(ctx)} disabled={!ctx} onChange={e => setAttach(e.target.checked)} />
              {ctx ? <span>附带本页数据：<b>{ctx.title}</b> · {chars.toLocaleString()} 字符{chars > 12000 ? '（超过 12,000 会截断）' : ''}
                {profile && <> → 发送给 <span className="mono">{profile.host}</span></>}
                {ctx.local && attach && <em className="ai-warn">本机结果，不在公开数据里，会发送给该主机</em>}</span>
                : <span className="muted">当前页面没有可附带的数据</span>}
            </label>
            <textarea rows={3} value={input} placeholder={profile ? '输入问题，Ctrl+Enter 发送' : '先选择或配置模型'} disabled={!profile}
                      onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); send(); } }} />
            <div className="ai-compose-act">
              {busyId
                ? <button type="button" className="fw-tbtn" onClick={() => bridge.cancel(busyId)}>停止</button>
                : <button type="button" className="fw-tbtn primary" disabled={!profile || !profile.has_key || !input.trim()} onClick={send}>发送</button>}
              {cfg && <small className="muted">今日 {cfg.usage_today.toLocaleString()} / {cfg.daily_limit.toLocaleString()} tokens</small>}
            </div>
          </div>
        </aside>
      )}
    </>
  );
}
