/* 请求层超时单测（client_architecture_review P0-5）：postJson 超时给出可读错误，不再永久挂起。 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

describe('postJson 超时', () => {
  beforeEach(() => { vi.stubGlobal('fetch', vi.fn()); });
  afterEach(() => { vi.unstubAllGlobals(); });

  it('挂起的请求在 timeoutMs 后以可读错误结束', async () => {
    const { postJson } = await import('./components/app/rdc_fusion/api.js');
    vi.stubGlobal('fetch', (_url, init) => new Promise((_resolve, reject) => {
      init.signal.addEventListener('abort', () => reject(init.signal.reason));
    }));
    await expect(postJson('/api/grammar/preview', {}, { timeoutMs: 50 }))
      .rejects.toThrow(/超时/);
  }, 10000);

  it('正常响应不受超时影响', async () => {
    const { postJson } = await import('./components/app/rdc_fusion/api.js');
    vi.stubGlobal('fetch', () => Promise.resolve({ ok: true, json: () => Promise.resolve({ ok: 1 }) }));
    await expect(postJson('/api/x', {}, { timeoutMs: 5000 })).resolves.toEqual({ ok: 1 });
  }, 10000);
});
