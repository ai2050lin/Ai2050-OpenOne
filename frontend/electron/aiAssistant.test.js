/* node --test frontend/electron/aiAssistant.test.js —— 假模型服务（本机 HTTP）上测试 AI 助手：两种 provider 的流式解析、
   Key 不出现在公开配置与记录里、地址规则、加密不可用时拒绝保存、每日上限、停止、错误信息打码。 */
'use strict';

const assert = require('node:assert/strict');
const fs = require('fs');
const http = require('http');
const os = require('os');
const path = require('path');
const test = require('node:test');
const { AiAssistant, validateBaseUrl, endpoint } = require('./aiAssistant');

const KEY = 'sk-test-0123456789abcdefWXYZ';
const cipher = {       // 假加密：可逆但不是明文（测试里检查文件中找不到 Key）
  isEncryptionAvailable: () => true,
  encryptString: s => Buffer.from([...Buffer.from(s)].map(b => b ^ 0x5a)),
  decryptString: b => Buffer.from([...b].map(x => x ^ 0x5a)).toString(),
};

function tmpDir() { return fs.mkdtempSync(path.join(os.tmpdir(), 'ai-assist-')); }

/* 假服务：/v1/chat/completions（OpenAI 兼容）与 /v1/messages（Anthropic）；记下收到的请求 */
function fakeServer(opts = {}) {
  const seen = [];
  const srv = http.createServer((req, res) => {
    let body = '';
    req.on('data', c => (body += c)).on('end', async () => {
      seen.push({ url: req.url, headers: req.headers, body: body ? JSON.parse(body) : null });
      if (opts.status) { res.writeHead(opts.status, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ error: { message: `bad key ${req.headers.authorization || req.headers['x-api-key']}` } })); return; }
      res.writeHead(200, { 'Content-Type': 'text/event-stream' });
      const send = s => res.write(s);
      const pieces = opts.pieces || ['你好', '，', '世界'];
      if (req.url === '/v1/messages') {
        send(`event: message_start\ndata: ${JSON.stringify({ type: 'message_start', message: { usage: { input_tokens: 11 } } })}\n\n`);
        for (const t of pieces) {
          send(`event: content_block_delta\ndata: ${JSON.stringify({ type: 'content_block_delta', delta: { type: 'text_delta', text: t } })}\n\n`);
          if (opts.slow) await new Promise(r => setTimeout(r, opts.slow));
        }
        send(`event: message_delta\ndata: ${JSON.stringify({ type: 'message_delta', usage: { output_tokens: 7 } })}\n\n`);
        send('event: message_stop\ndata: {"type":"message_stop"}\n\n');
      } else {
        for (const t of pieces) {
          send(`data: ${JSON.stringify({ choices: [{ delta: { content: t } }] })}\r\n\r\n`);   // CRLF 分隔也要能解析
          if (opts.slow) await new Promise(r => setTimeout(r, opts.slow));
        }
        if (opts.usage) send(`data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: 20, completion_tokens: 5 } })}\n\n`);
        send('data: [DONE]\n\n');
      }
      res.end();
    });
  });
  return new Promise(r => srv.listen(0, '127.0.0.1', () => r({ srv, seen, base: `http://127.0.0.1:${srv.address().port}/v1` })));
}

test('地址规则与端点', () => {
  assert.equal(validateBaseUrl('https://api.deepseek.com/v1/'), 'https://api.deepseek.com/v1');
  assert.equal(validateBaseUrl('http://127.0.0.1:11434/v1'), 'http://127.0.0.1:11434/v1');
  assert.throws(() => validateBaseUrl('http://api.example.com/v1'), /https/);
  assert.throws(() => validateBaseUrl('https://user:pw@api.example.com'), /用户名/);
  assert.throws(() => validateBaseUrl('ftp://x'), /https/);
  assert.equal(endpoint({ provider: 'anthropic', base_url: 'https://api.anthropic.com' }), 'https://api.anthropic.com/v1/messages');
  assert.equal(endpoint({ provider: 'anthropic', base_url: 'https://x.test/v1' }), 'https://x.test/v1/messages');
  assert.equal(endpoint({ provider: 'openai', base_url: 'https://x.test/v1' }), 'https://x.test/v1/chat/completions');
});

test('Key 只加密保存，公开配置只给尾号；加密不可用时拒绝保存', () => {
  const dir = tmpDir();
  const ai = new AiAssistant({ dir, cipher });
  const cfg = ai.saveProfile({ name: 'DS', provider: 'openai', base_url: 'https://api.deepseek.com', model: 'deepseek-chat', key: KEY });
  assert.equal(cfg.profiles[0].has_key, true);
  assert.equal(cfg.profiles[0].key_hint, 'WXYZ');
  assert.equal(cfg.default_id, cfg.saved_id);
  assert.ok(!JSON.stringify(cfg).includes(KEY));
  assert.ok(!fs.readFileSync(path.join(dir, 'ai_assistant.json'), 'utf8').includes(KEY));
  // 不带 key 的更新保留原 Key；clear_key 清除
  ai.saveProfile({ id: cfg.saved_id, model: 'deepseek-reasoner' });
  assert.equal(ai.publicConfig().profiles[0].has_key, true);
  assert.equal(ai.publicConfig().profiles[0].model, 'deepseek-reasoner');
  ai.saveProfile({ id: cfg.saved_id, clear_key: true });
  assert.equal(ai.publicConfig().profiles[0].has_key, false);
  const noEnc = new AiAssistant({ dir: tmpDir(), cipher: { isEncryptionAvailable: () => false } });
  assert.throws(() => noEnc.saveProfile({ name: 'x', base_url: 'https://a.test', model: 'm', key: KEY }), /无法加密/);
  assert.throws(() => ai.saveProfile({ name: 'x', base_url: 'http://evil.test', model: 'm' }), /https/);
  assert.throws(() => ai.saveProfile({ name: 'x', provider: 'gemini', base_url: 'https://a.test', model: 'm' }), /provider/);
});

test('OpenAI 兼容：流式拼接、带上下文的系统提示词、用量估计、记录不含 Key', async () => {
  const { srv, seen, base } = await fakeServer();
  const dir = tmpDir();
  const ai = new AiAssistant({ dir, cipher });
  ai.saveProfile({ name: 'local', provider: 'openai', base_url: base, model: 'm1', key: KEY });
  const deltas = [];
  const r = await ai.chat({ id: 'q1', messages: [{ role: 'user', content: '这个语言项说明什么？' }],
    context: { kind: 'lang_item', id: 'GR-NEG', title: '语言项 否定', data: { level: 2, note: 'x'.repeat(20000) } } }, e => deltas.push(e.text));
  srv.close();
  assert.equal(r.text, '你好，世界');
  assert.deepEqual(deltas, ['你好', '，', '世界']);
  assert.equal(r.usage.estimated, true);
  assert.equal(r.context_truncated, true);
  const sent = seen[0];
  assert.equal(sent.url, '/v1/chat/completions');
  assert.equal(sent.headers.authorization, `Bearer ${KEY}`);
  assert.equal(sent.body.stream, true);
  assert.equal(sent.body.messages[0].role, 'system');
  assert.match(sent.body.messages[0].content, /语言项 否定.*lang_item · GR-NEG/s);
  assert.match(sent.body.messages[0].content, /已截断/);
  assert.equal(sent.body.messages[1].content, '这个语言项说明什么？');
  const log = fs.readFileSync(path.join(dir, 'ai_logs', fs.readdirSync(path.join(dir, 'ai_logs'))[0]), 'utf8');
  assert.ok(!log.includes(KEY));
  const rec = JSON.parse(log.trim());
  assert.equal(rec.answer, '你好，世界');
  assert.equal(rec.context.kind, 'lang_item');
  assert.equal(rec.context.chars, 12000);
  assert.equal(rec.context.sha256.length, 64);
  assert.ok(ai.publicConfig().usage_today > 0);
});

test('Anthropic：x-api-key、system 字段与用量', async () => {
  const { srv, seen, base } = await fakeServer();
  const ai = new AiAssistant({ dir: tmpDir(), cipher });
  ai.saveProfile({ name: 'claude', provider: 'anthropic', base_url: base.replace(/\/v1$/, ''), model: 'claude-sonnet-5-5', key: KEY });
  const r = await ai.chat({ messages: [{ role: 'user', content: 'hi' }] });
  srv.close();
  assert.equal(r.text, '你好，世界');
  assert.deepEqual([r.usage.input, r.usage.output, r.usage.estimated], [11, 7, false]);
  assert.equal(seen[0].url, '/v1/messages');
  assert.equal(seen[0].headers['x-api-key'], KEY);
  assert.equal(seen[0].headers['anthropic-version'], '2023-06-01');
  assert.equal(typeof seen[0].body.system, 'string');
  assert.deepEqual(seen[0].body.messages, [{ role: 'user', content: 'hi' }]);
});

test('HTTP 错误信息里的 Key 打码；没有 Key 或配置时给出可操作的提示', async () => {
  const { srv, base } = await fakeServer({ status: 401 });
  const ai = new AiAssistant({ dir: tmpDir(), cipher });
  await assert.rejects(ai.chat({ messages: [{ role: 'user', content: 'x' }] }), /还没有 AI 模型配置/);
  const { saved_id: id } = ai.saveProfile({ name: 'a', base_url: base, model: 'm' });
  await assert.rejects(ai.chat({ messages: [{ role: 'user', content: 'x' }] }), /还没有 API Key/);
  ai.saveProfile({ id, key: KEY });
  const err = await ai.chat({ messages: [{ role: 'user', content: 'x' }] }).catch(e => e);
  const t = await ai.test(id);
  srv.close();
  assert.match(err.message, /HTTP 401/);
  assert.ok(!err.message.includes(KEY));
  assert.equal(t.ok, false);
  assert.ok(!t.error.includes(KEY));
});

test('每日上限、停止与测试连接', async () => {
  const { srv, base } = await fakeServer({ slow: 150, pieces: ['a', 'b', 'c', 'd', 'e'], usage: true });
  let clock = new Date(2026, 9, 28, 10).getTime();
  const ai = new AiAssistant({ dir: tmpDir(), cipher, now: () => clock });
  ai.saveProfile({ name: 'a', base_url: base, model: 'm', key: KEY });
  const t = await ai.test();
  assert.equal(t.ok, true);
  // 停止：第一段到达后取消，返回「已停止」
  const p = ai.chat({ id: 'slow', messages: [{ role: 'user', content: 'x' }] }, () => ai.cancel('slow'));
  await assert.rejects(p, /已停止/);
  assert.equal(ai.cancel('slow').cancelled, false);
  // 上限：设到 1000，再用掉超过它的量后拒绝；第二天恢复
  ai.setSettings({ daily_limit: 1000 });
  const c = JSON.parse(fs.readFileSync(ai.file, 'utf8'));
  c.usage = { date: '2026-10-28', tokens: 1000 };
  fs.writeFileSync(ai.file, JSON.stringify(c));
  await assert.rejects(ai.chat({ messages: [{ role: 'user', content: 'x' }] }), /已达上限/);
  clock += 24 * 3600 * 1000;
  const r = await ai.chat({ messages: [{ role: 'user', content: 'x' }] });
  srv.close();
  assert.equal(r.text, 'abcde');
  assert.deepEqual([r.usage.input, r.usage.output], [20, 5]);
  assert.equal(ai.publicConfig().usage_today, 25);
  assert.throws(() => ai.setSettings({ daily_limit: 5 }), /每日上限/);
});
