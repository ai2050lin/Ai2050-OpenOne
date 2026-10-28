/* 研发 · AI 助手（ai_assist.v1，设计见 design/ai_assistant_v1.md）：配置自己的模型（地址、模型名、API Key）与每日用量上限。
   配置与 Key 在桌面客户端主进程（Key 用系统加密保存，页面只看到「已设置 · 尾号」）；网页版只显示说明。 */
import { useCallback, useEffect, useState } from 'react';
import { aiBridge } from './aiContext.js';
import { Card, Empty, StatusPill } from './ui.jsx';
import './ai.css';

const PRESETS = [
  { label: 'OpenAI', provider: 'openai', base_url: 'https://api.openai.com/v1', model: 'gpt-4o-mini' },
  { label: 'DeepSeek', provider: 'openai', base_url: 'https://api.deepseek.com', model: 'deepseek-chat' },
  { label: 'Anthropic', provider: 'anthropic', base_url: 'https://api.anthropic.com', model: 'claude-sonnet-5-5' },
  { label: '本机 Ollama', provider: 'openai', base_url: 'http://127.0.0.1:11434/v1', model: 'qwen3:8b' },
];
const EMPTY = { id: '', name: '', provider: 'openai', base_url: '', model: '', key: '', clear_key: false, max_tokens: 2000, temperature: 0.3, has_key: false, key_hint: '' };

function ProfileForm({ init, onSaved, onCancel }) {
  const [f, setF] = useState(init);
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);
  const up = patch => setF(v => ({ ...v, ...patch }));
  const submit = async e => {
    e.preventDefault();
    setBusy(true); setErr('');
    try {
      const body = { id: f.id || undefined, name: f.name, provider: f.provider, base_url: f.base_url, model: f.model,
        max_tokens: Number(f.max_tokens), temperature: Number(f.temperature), clear_key: f.clear_key };
      if (f.key.trim()) body.key = f.key;
      onSaved(await aiBridge().save(body));
    } catch (e2) { setErr(e2.message); }
    setBusy(false);
  };
  const field = (k, props = {}) => <input {...props} value={f[k]} onChange={e => up({ [k]: e.target.value })} />;
  return (
    <form className="ai-form" onSubmit={submit}>
      {!f.id && (
        <div className="ai-presets"><span>常用：</span>
          {PRESETS.map(p => <button key={p.label} type="button" className="fw-tbtn" onClick={() => up({ ...p, name: f.name || p.label })}>{p.label}</button>)}
        </div>
      )}
      <label>名称{field('name', { required: true, maxLength: 40, placeholder: '如 DeepSeek' })}</label>
      <label>接口类型
        <select value={f.provider} onChange={e => up({ provider: e.target.value })}>
          <option value="openai">OpenAI 兼容（OpenAI、DeepSeek、通义、Kimi、Ollama…）</option>
          <option value="anthropic">Anthropic</option>
        </select>
      </label>
      <label className="wide">地址{field('base_url', { required: true, placeholder: 'https://api.deepseek.com', spellCheck: false })}
        <small>只接受 https；http 只允许本机（127.0.0.1 / localhost）。</small></label>
      <label>模型名{field('model', { required: true, placeholder: 'deepseek-chat', spellCheck: false })}</label>
      <label>API Key
        <input type="password" autoComplete="off" value={f.key} disabled={f.clear_key}
               placeholder={f.has_key ? `已设置 · 尾号 ${f.key_hint}（留空不改）` : '粘贴 Key'} onChange={e => up({ key: e.target.value })} />
        {f.has_key && <span className="ai-check"><input type="checkbox" checked={f.clear_key} onChange={e => up({ clear_key: e.target.checked, key: '' })} />清除已保存的 Key</span>}
      </label>
      <label>单次回答上限（tokens）{field('max_tokens', { type: 'number', min: 64, max: 32000, step: 1 })}</label>
      <label>temperature{field('temperature', { type: 'number', min: 0, max: 2, step: 0.1 })}</label>
      <div className="ai-form-act wide">
        <button type="submit" className="fw-tbtn primary" disabled={busy}>{busy ? '保存中…' : '保存'}</button>
        <button type="button" className="fw-tbtn" onClick={onCancel}>取消</button>
        {err && <span className="ai-err" role="alert">{err}</span>}
      </div>
    </form>
  );
}

export default function AiSettingsPanel() {
  const bridge = aiBridge();
  const [cfg, setCfg] = useState(null);
  const [err, setErr] = useState('');
  const [edit, setEdit] = useState(null);                // null | 表单初值
  const [tests, setTests] = useState({});
  const [limit, setLimit] = useState('');
  const [msg, setMsg] = useState('');
  const load = useCallback(() => {
    if (!bridge) return;
    bridge.config().then(c => { setCfg(c); setLimit(String(c.daily_limit)); }).catch(e => setErr(e.message));
  }, [bridge]);
  useEffect(() => { load(); }, [load]);

  if (!bridge) {
    return (
      <Card title="AI 助手">
        <Empty>AI 助手只在桌面客户端可用：在客户端里配置自己的模型（地址、模型名、API Key）后，可以在语言项、神经元和神经元测试页面就地提问。网页版只能浏览，不保存、不使用 Key。</Empty>
      </Card>
    );
  }
  const act = async (fn, okText) => {
    setErr(''); setMsg('');
    try { const c = await fn(); if (c && c.profiles) setCfg(c); if (okText) setMsg(okText); } catch (e) { setErr(e.message); }
  };
  const runTest = async p => {
    setTests(t => ({ ...t, [p.id]: { busy: true } }));
    let result;
    try { result = await bridge.test(p.id); } catch (e) { result = { ok: false, error: e.message }; }
    setTests(t => ({ ...t, [p.id]: { result } }));
  };

  return (
    <div className="ai-settings">
      <Card title="AI 模型" extra={!edit && <button type="button" className="fw-tbtn primary" onClick={() => setEdit({ ...EMPTY })}>＋ 添加模型</button>}>
        {err && <div className="ai-err block" role="alert">{err}</div>}
        {msg && <div className="ai-ok" role="status">{msg}</div>}
        {cfg && !cfg.encryption_available && <div className="ai-err block">本机系统加密不可用，暂时无法保存 API Key。</div>}
        {edit && <ProfileForm key={edit.id || 'new'} init={edit} onCancel={() => setEdit(null)}
                              onSaved={c => { setCfg(c); setEdit(null); setMsg('已保存'); }} />}
        {!cfg && !err && <Empty>读取配置…</Empty>}
        {cfg && !cfg.profiles.length && !edit && <Empty>还没有模型。点「添加模型」，填地址、模型名与 API Key。</Empty>}
        {cfg && cfg.profiles.length > 0 && (
          <table className="ai-table">
            <thead><tr><th>名称</th><th>接口</th><th>模型</th><th>地址</th><th>API Key</th><th>连接</th><th /></tr></thead>
            <tbody>
              {cfg.profiles.map(p => {
                const t = tests[p.id];
                return (
                  <tr key={p.id}>
                    <th scope="row"><b>{p.name}</b>{p.default && <em className="ai-tag">缺省</em>}</th>
                    <td>{p.provider === 'anthropic' ? 'Anthropic' : 'OpenAI 兼容'}</td>
                    <td className="mono">{p.model}</td>
                    <td className="mono">{p.host}</td>
                    <td>{p.has_key ? <StatusPill status="pass" label={`已设置 · ${p.key_hint}`} /> : <StatusPill status="missing" label="未设置" />}</td>
                    <td>
                      {t?.busy && <small>测试中…</small>}
                      {t?.result && (t.result.ok
                        ? <StatusPill status="pass" label={`可用 · ${t.result.ms} ms`} />
                        : <span className="ai-err" title={t.result.error}><StatusPill status="fail" label="失败" /> {t.result.error.slice(0, 80)}</span>)}
                    </td>
                    <td className="ai-act">
                      <button type="button" className="fw-tbtn" disabled={!p.has_key || t?.busy} onClick={() => runTest(p)}>测试连接</button>
                      <button type="button" className="fw-tbtn" onClick={() => setEdit({ ...EMPTY, ...p, key: '', clear_key: false })}>编辑</button>
                      {!p.default && <button type="button" className="fw-tbtn" onClick={() => act(() => bridge.settings({ default_id: p.id }))}>设为缺省</button>}
                      <button type="button" className="fw-tbtn ai-danger" onClick={() => window.confirm(`删除模型配置「${p.name}」（含保存的 Key）？`) && act(() => bridge.remove(p.id), `已删除 ${p.name}`)}>删除</button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </Card>

      {cfg && (
        <Card title="用量与记录">
          <div className="ai-usage">
            <span>今日用量 <b>{cfg.usage_today.toLocaleString()}</b> / {cfg.daily_limit.toLocaleString()} tokens</span>
            <label>每日上限
              <input type="number" min={1000} step={1000} value={limit} onChange={e => setLimit(e.target.value)} />
            </label>
            <button type="button" className="fw-tbtn" disabled={String(cfg.daily_limit) === limit} onClick={() => act(() => bridge.settings({ daily_limit: Number(limit) }), '每日上限已更新')}>保存上限</button>
            <button type="button" className="fw-tbtn" onClick={() => act(() => bridge.openLogs())}>打开记录目录</button>
          </div>
          <p className="ai-note">
            服务没有返回用量时按字符数 ÷ 2 估计。每次提问在记录目录按日期追加一行：模型、主机、附带的页面数据（种类、长度、sha256）、问题与回答，不含 Key。
          </p>
        </Card>
      )}

      <Card title="说明">
        <ul className="ai-note-list">
          <li><b>Key 保存在哪里</b>：只在这台电脑上，用系统加密（Windows DPAPI，绑定当前用户）保存在客户端数据目录；页面、记录与中心都拿不到。</li>
          <li><b>会发送什么</b>：你的问题，以及你勾选附带的「本页数据」（语言项、神经元或本机测试结果的摘要），直接发给你配置的模型地址，不经过 AI2050 中心。</li>
          <li><b>怎么用</b>：在研究工作台右下角点「问 AI」，在语言项、神经元单元或神经元测试报告页面提问，AI 会看到这一页的数据。</li>
          <li><b>证据边界</b>：AI 的回答一律是「未检验」，不是证据；值得验证的想法要写成模板或测试运行。</li>
        </ul>
      </Card>
    </div>
  );
}
