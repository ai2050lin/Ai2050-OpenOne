/**
 * 客户端 AI 助手（ai_assist.v1，设计见 design/ai_assistant_v1.md）：在主进程保存模型配置与 API Key，代页面发起流式请求。
 *
 * - Key 只在主进程：用 safeStorage（Windows DPAPI）加密后写入 <userData>/ai_assistant.json；加密不可用时拒绝保存。
 *   publicConfig() 只给页面「是否已设置 + 尾号」。
 * - 只向配置里的 base_url 发请求：https，或 http 的 localhost / 127.0.0.1 / [::1]（本机模型）。
 * - provider：openai（OpenAI 兼容 /chat/completions）与 anthropic（/v1/messages），都按 SSE 流式解析。
 * - 每日 token 上限；每次请求在 <userData>/ai_logs/<日期>.jsonl 记一行（不含 Key）。
 * 不依赖 Electron：目录、加密器与 fetch 由调用方传入，便于 node:test 用假服务测试。
 */
'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const SCHEMA = 'ai_assist.v1';
const PROVIDERS = ['openai', 'anthropic'];
const DEFAULT_DAILY_LIMIT = 200000;
const CONTEXT_MAX_CHARS = 12000;
const REQUEST_TIMEOUT_MS = 180000;
const ANTHROPIC_VERSION = '2023-06-01';

/* 固定的系统提示词：项目术语与边界。页面数据作为单独一段附在后面。 */
const SYSTEM_PROMPT = [
  '你是 AI2050 研究工作台里的研究助手。AI2050 研究 LLM 内部如何编码语言：用生成式模板（文法）做受控对照，用自然语料普查 MLP 神经元。',
  '术语：hs[0] 是嵌入层，块 b 的输出是 hs[b+1]；读出位置有 target（目标词）、cue（线索词）、final（句末）、probe_answer（答案前）；',
  '行为门 = 模型在该条件格上先要能做对任务；留出复现 = 在未参与拟合的划分（confirm / lockbox）上仍成立；',
  'hs[0] 上就能分开的效应只等于词汇身份，不算上下文计算；读出轴等级 1 有数据、2 confirm 可读出、3 未见词表可读出且超出 hs[0]、4 跨模型或跨模式族。',
  '回答规则：只根据下面给出的页面数据和用户的问题回答；区分「数据显示」与「推测」；推测要写成可检验的形式（用什么模板或测试、预期看到什么）；',
  '数据里没有的数字不要编造；不要把你的判断说成证据——你的回答全部是「未检验」。用中文回答，简洁具体。',
].join('\n');

function today(now) {
  const d = new Date(now());
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

function validateBaseUrl(raw) {
  let u;
  try { u = new URL(String(raw || '').trim()); } catch { throw new Error('地址格式不正确'); }
  if (u.username || u.password) throw new Error('地址里不能带用户名或密码');
  const local = ['localhost', '127.0.0.1', '[::1]'].includes(u.hostname);
  if (u.protocol !== 'https:' && !(u.protocol === 'http:' && local)) throw new Error('只接受 https 地址；http 只允许本机（localhost / 127.0.0.1）');
  if (u.search || u.hash) throw new Error('地址里不要带 ? 或 # 参数');
  return u.toString().replace(/\/+$/, '');
}

function endpoint(p) {
  const base = p.base_url.replace(/\/+$/, '');
  if (p.provider === 'anthropic') return /\/v1$/.test(base) ? `${base}/messages` : `${base}/v1/messages`;
  return `${base}/chat/completions`;
}

function mask(text, key) {
  let s = String(text || '');
  if (key && key.length >= 6) s = s.split(key).join('***');
  return s.replace(/(sk-[A-Za-z0-9_-]{4})[A-Za-z0-9_-]{8,}/g, '$1***');
}

/* SSE：按空行分事件，取每个事件的 data 行 */
async function* sseEvents(body) {
  const dec = new TextDecoder();
  let buf = '';
  for await (const chunk of body) {
    buf += dec.decode(chunk, { stream: true });
    let i;
    while ((i = buf.search(/\r?\n\r?\n/)) >= 0) {
      const block = buf.slice(0, i);
      buf = buf.slice(i).replace(/^\r?\n\r?\n/, '');
      const data = block.split(/\r?\n/).filter(l => l.startsWith('data:')).map(l => l.slice(5).trimStart()).join('\n');
      if (data) yield data;
    }
  }
  const rest = buf.split(/\r?\n/).filter(l => l.startsWith('data:')).map(l => l.slice(5).trimStart()).join('\n');
  if (rest) yield rest;
}

class AiAssistant {
  constructor({ dir, cipher, fetchImpl, now } = {}) {
    this.dir = typeof dir === 'function' ? dir : () => dir;
    this.cipher = cipher;                 // { isEncryptionAvailable(), encryptString(s) -> Buffer, decryptString(Buffer) -> s }
    this.fetch = fetchImpl || globalThis.fetch;
    this.now = now || Date.now;
    this.running = new Map();             // 请求 id → AbortController
  }

  get file() { return path.join(this.dir(), 'ai_assistant.json'); }
  get logDir() { return path.join(this.dir(), 'ai_logs'); }

  _read() {
    try {
      const c = JSON.parse(fs.readFileSync(this.file, 'utf8'));
      return { schema: SCHEMA, profiles: [], default_id: '', daily_limit: DEFAULT_DAILY_LIMIT, usage: {}, ...c };
    } catch {
      return { schema: SCHEMA, profiles: [], default_id: '', daily_limit: DEFAULT_DAILY_LIMIT, usage: {} };
    }
  }

  _write(c) {
    fs.mkdirSync(this.dir(), { recursive: true });
    const tmp = `${this.file}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify(c, null, 1));
    fs.renameSync(tmp, this.file);
  }

  _encAvailable() {
    try { return Boolean(this.cipher && this.cipher.isEncryptionAvailable()); } catch { return false; }
  }

  _usageToday(c) {
    return c.usage && c.usage.date === today(this.now) ? Number(c.usage.tokens) || 0 : 0;
  }

  publicConfig() {
    const c = this._read();
    const def = c.profiles.some(p => p.id === c.default_id) ? c.default_id : (c.profiles[0] || {}).id || '';
    return {
      schema: SCHEMA,
      profiles: c.profiles.map(p => ({ id: p.id, name: p.name, provider: p.provider, base_url: p.base_url, model: p.model,
        max_tokens: p.max_tokens, temperature: p.temperature, has_key: Boolean(p.key_enc), key_hint: p.key_enc ? p.key_hint || '' : '',
        host: (() => { try { return new URL(p.base_url).host; } catch { return ''; } })(), default: p.id === def })),
      default_id: def,
      daily_limit: c.daily_limit,
      usage_today: this._usageToday(c),
      encryption_available: this._encAvailable(),
      log_dir: this.logDir,
    };
  }

  saveProfile(input = {}) {
    const c = this._read();
    const old = c.profiles.find(p => p.id === input.id);
    const provider = String(input.provider || (old && old.provider) || 'openai');
    if (!PROVIDERS.includes(provider)) throw new Error('provider 只能是 openai 或 anthropic');
    const name = String(input.name ?? (old && old.name) ?? '').trim().slice(0, 40);
    if (!name) throw new Error('请填写名称');
    const model = String(input.model ?? (old && old.model) ?? '').trim().slice(0, 120);
    if (!model) throw new Error('请填写模型名');
    const maxTokens = Math.round(Number(input.max_tokens ?? (old && old.max_tokens) ?? 2000));
    if (!(maxTokens >= 64 && maxTokens <= 32000)) throw new Error('单次回答上限应在 64–32000 之间');
    const temperature = Number(input.temperature ?? (old && old.temperature) ?? 0.3);
    if (!(temperature >= 0 && temperature <= 2)) throw new Error('temperature 应在 0–2 之间');
    const p = {
      id: old ? old.id : `p-${crypto.randomBytes(4).toString('hex')}`,
      name, provider, model, max_tokens: maxTokens, temperature,
      base_url: validateBaseUrl(input.base_url ?? (old && old.base_url)),
      key_enc: old ? old.key_enc : '', key_hint: old ? old.key_hint : '',
    };
    if (input.clear_key) { p.key_enc = ''; p.key_hint = ''; }
    const key = typeof input.key === 'string' ? input.key.trim() : '';
    if (key) {
      if (!this._encAvailable()) throw new Error('本机无法加密保存 API Key（系统加密不可用），未保存');
      p.key_enc = this.cipher.encryptString(key).toString('base64');
      p.key_hint = key.slice(-4);
    }
    c.profiles = old ? c.profiles.map(x => (x.id === p.id ? p : x)) : [...c.profiles, p];
    if (!c.default_id || input.make_default) c.default_id = p.id;
    this._write(c);
    return { ...this.publicConfig(), saved_id: p.id };
  }

  removeProfile(id) {
    const c = this._read();
    c.profiles = c.profiles.filter(p => p.id !== id);
    if (c.default_id === id) c.default_id = (c.profiles[0] || {}).id || '';
    this._write(c);
    return this.publicConfig();
  }

  setSettings({ default_id: defaultId, daily_limit: dailyLimit } = {}) {
    const c = this._read();
    if (defaultId !== undefined) {
      if (!c.profiles.some(p => p.id === defaultId)) throw new Error('没有这个配置');
      c.default_id = defaultId;
    }
    if (dailyLimit !== undefined) {
      const n = Math.round(Number(dailyLimit));
      if (!(n >= 1000 && n <= 100000000)) throw new Error('每日上限应在 1,000 – 100,000,000 tokens 之间');
      c.daily_limit = n;
    }
    this._write(c);
    return this.publicConfig();
  }

  _profile(id) {
    const c = this._read();
    const p = c.profiles.find(x => x.id === (id || c.default_id)) || (!id && c.profiles[0]);
    if (!p) throw new Error('还没有 AI 模型配置：在「研发 → AI 助手」里添加');
    if (!p.key_enc) throw new Error(`配置「${p.name}」还没有 API Key`);
    let key;
    try { key = this.cipher.decryptString(Buffer.from(p.key_enc, 'base64')); } catch { throw new Error(`配置「${p.name}」的 Key 无法解密（换了系统用户？），请重新填写`); }
    return { c, p, key };
  }

  _request(p, key, system, messages, maxTokens) {
    const headers = { 'Content-Type': 'application/json', Accept: 'text/event-stream' };
    let body;
    if (p.provider === 'anthropic') {
      headers['x-api-key'] = key;
      headers['anthropic-version'] = ANTHROPIC_VERSION;
      body = { model: p.model, max_tokens: maxTokens, temperature: p.temperature, system, messages, stream: true };
    } else {
      headers.Authorization = `Bearer ${key}`;
      body = { model: p.model, max_tokens: maxTokens, temperature: p.temperature, stream: true,
        messages: [{ role: 'system', content: system }, ...messages] };
    }
    return { url: endpoint(p), headers, body: JSON.stringify(body) };
  }

  /* 一次流式调用：返回 { text, usage: {input, output, estimated}, ms }；onDelta(text) 收到增量 */
  async _stream(p, key, system, messages, maxTokens, signal, onDelta) {
    const t0 = this.now();
    const req = this._request(p, key, system, messages, maxTokens);
    let res;
    try {
      res = await this.fetch(req.url, { method: 'POST', headers: req.headers, body: req.body, signal });
    } catch (e) {
      if (signal.aborted) throw new Error('已停止');
      throw new Error(`连不上 ${new URL(p.base_url).host}：${mask(e.message, key)}`);
    }
    if (!res.ok) {
      const t = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status}：${mask(t, key).slice(0, 400)}`);
    }
    let text = '';
    const usage = { input: 0, output: 0, estimated: false };
    try {
      for await (const data of sseEvents(res.body)) {
        if (data === '[DONE]') break;
        let ev;
        try { ev = JSON.parse(data); } catch { continue; }
        let piece = '';
        if (p.provider === 'anthropic') {
          if (ev.type === 'error') throw new Error(`模型服务报错：${mask(JSON.stringify(ev.error || ev), key).slice(0, 400)}`);
          if (ev.type === 'message_start') usage.input = ev.message?.usage?.input_tokens || 0;
          if (ev.type === 'content_block_delta' && ev.delta?.type === 'text_delta') piece = ev.delta.text || '';
          if (ev.type === 'message_delta' && ev.usage) usage.output = ev.usage.output_tokens || usage.output;
        } else {
          if (ev.error) throw new Error(`模型服务报错：${mask(JSON.stringify(ev.error), key).slice(0, 400)}`);
          piece = ev.choices?.[0]?.delta?.content || '';
          if (ev.usage) { usage.input = ev.usage.prompt_tokens || 0; usage.output = ev.usage.completion_tokens || 0; }
        }
        if (piece) { text += piece; onDelta(piece); }
      }
    } catch (e) {
      if (signal.aborted) { const err = new Error('已停止'); err.partial = text; throw err; }
      throw e;
    }
    if (!usage.input && !usage.output) {          // 服务没有返回用量：按字符数 ÷ 2 估计
      usage.estimated = true;
      usage.input = Math.ceil((system.length + messages.reduce((n, m) => n + String(m.content).length, 0)) / 2);
      usage.output = Math.ceil(text.length / 2);
    }
    return { text, usage, ms: this.now() - t0 };
  }

  _addUsage(tokens) {
    const c = this._read();
    const d = today(this.now);
    c.usage = { date: d, tokens: (c.usage && c.usage.date === d ? Number(c.usage.tokens) || 0 : 0) + tokens };
    this._write(c);
  }

  _log(rec) {
    try {
      fs.mkdirSync(this.logDir, { recursive: true });
      fs.appendFileSync(path.join(this.logDir, `${today(this.now)}.jsonl`), `${JSON.stringify(rec)}\n`);
    } catch { /* 记录失败不影响回答 */ }
  }

  async test(profileId) {
    const { p, key } = this._profile(profileId);
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), 30000);
    try {
      const r = await this._stream(p, key, '只回答“ok”。', [{ role: 'user', content: 'ping' }], 16, ac.signal, () => {});
      return { ok: true, ms: r.ms, model: p.model, host: new URL(p.base_url).host, reply: r.text.slice(0, 80) };
    } catch (e) {
      return { ok: false, model: p.model, host: new URL(p.base_url).host, error: e.message };
    } finally { clearTimeout(timer); }
  }

  /* chat({ id, profile_id, messages:[{role,content}], context:{kind,id,title,data} }, onEvent) */
  async chat(req = {}, onEvent = () => {}) {
    const id = String(req.id || crypto.randomBytes(6).toString('hex'));
    const messages = (Array.isArray(req.messages) ? req.messages : [])
      .filter(m => m && ['user', 'assistant'].includes(m.role) && typeof m.content === 'string' && m.content.trim())
      .slice(-20).map(m => ({ role: m.role, content: m.content.slice(0, 20000) }));
    if (!messages.length || messages[messages.length - 1].role !== 'user') throw new Error('没有要发送的问题');
    const { c, p, key } = this._profile(req.profile_id);
    const used = this._usageToday(c);
    if (used >= c.daily_limit) throw new Error(`今日用量 ${used.toLocaleString()} 已达上限 ${c.daily_limit.toLocaleString()} tokens（设置页可调整）`);

    let ctxText = '';
    let truncated = false;
    const ctx = req.context && typeof req.context === 'object' ? req.context : null;
    if (ctx) {
      ctxText = typeof ctx.data === 'string' ? ctx.data : JSON.stringify(ctx.data ?? null);
      if (ctxText.length > CONTEXT_MAX_CHARS) { ctxText = ctxText.slice(0, CONTEXT_MAX_CHARS); truncated = true; }
    }
    const system = ctx
      ? `${SYSTEM_PROMPT}\n\n当前页面：${String(ctx.title || ctx.kind || '').slice(0, 200)}（${ctx.kind || ''}${ctx.id ? ` · ${ctx.id}` : ''}）\n页面数据（JSON${truncated ? '，已截断' : ''}）：\n${ctxText}`
      : SYSTEM_PROMPT;

    const ac = new AbortController();
    this.running.set(id, ac);
    const timer = setTimeout(() => ac.abort(), REQUEST_TIMEOUT_MS);
    const rec = { ts: new Date(this.now()).toISOString(), id, profile: p.name, provider: p.provider, model: p.model,
      host: new URL(p.base_url).host, context: ctx ? { kind: ctx.kind || '', id: String(ctx.id || ''), chars: ctxText.length, truncated,
        sha256: crypto.createHash('sha256').update(ctxText).digest('hex') } : null,
      question: messages[messages.length - 1].content, turns: messages.length };
    try {
      const r = await this._stream(p, key, system, messages, p.max_tokens, ac.signal, text => onEvent({ type: 'delta', id, text }));
      this._addUsage(r.usage.input + r.usage.output);
      this._log({ ...rec, answer: r.text, usage: r.usage, ms: r.ms });
      return { id, text: r.text, usage: r.usage, ms: r.ms, profile: p.name, model: p.model, host: rec.host, context_truncated: truncated };
    } catch (e) {
      this._log({ ...rec, error: e.message, partial: e.partial || '' });
      throw e;
    } finally {
      clearTimeout(timer);
      this.running.delete(id);
    }
  }

  cancel(id) {
    const ac = this.running.get(String(id));
    if (ac) ac.abort();
    return { cancelled: Boolean(ac) };
  }
}

module.exports = { AiAssistant, SCHEMA, SYSTEM_PROMPT, CONTEXT_MAX_CHARS, DEFAULT_DAILY_LIMIT, validateBaseUrl, endpoint, mask, sseEvents };
