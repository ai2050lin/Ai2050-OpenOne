// @vitest-environment happy-dom
/* diagnostics.js 单测：环形缓冲、导出格式、全局捕获不抛错。 */
import { describe, it, expect, beforeEach } from 'vitest';
import { recordError, getErrors, clearErrors, errorCount, exportDiagnostics, installGlobalErrorCapture } from './diagnostics.js';

describe('diagnostics 环形缓冲', () => {
  beforeEach(() => clearErrors());

  it('记录并读取错误', () => {
    expect(recordError('error', 'boom', 'stack-line')).toBe(true);
    const all = getErrors();
    expect(all).toHaveLength(1);
    expect(all[0]).toMatchObject({ kind: 'error', message: 'boom', detail: 'stack-line' });
    expect(all[0].ts).toBeTruthy();
    expect(errorCount()).toBe(1);
  });

  it('超过 50 条丢弃最旧的', () => {
    for (let i = 0; i < 55; i++) recordError('manual', `e${i}`);
    const all = getErrors();
    expect(all).toHaveLength(50);
    expect(all[0].message).toBe('e5');
    expect(all[49].message).toBe('e54');
  });

  it('超长消息截断，不抛错', () => {
    expect(recordError('error', 'x'.repeat(5000), 'y'.repeat(9999))).toBe(true);
    expect(getErrors()[0].message.length).toBeLessThanOrEqual(400);
    expect(getErrors()[0].detail.length).toBeLessThanOrEqual(1200);
  });

  it('exportDiagnostics 包含环境与错误明细', () => {
    recordError('error', 'boom', 'at line 1');
    const text = exportDiagnostics();
    expect(text).toContain('AI2050 客户端诊断包');
    expect(text).toContain('boom');
    expect(text).toContain('at line 1');
  });

  it('clearErrors 清空', () => {
    recordError('error', 'x');
    clearErrors();
    expect(errorCount()).toBe(0);
  });
});

describe('installGlobalErrorCapture', () => {
  beforeEach(() => { clearErrors(); });

  it('捕获未处理 rejection 并入账；AbortError/TimeoutError 忽略', () => {
    installGlobalErrorCapture();
    expect(window.__ai2050ErrorCapture).toBe(true);
    window.dispatchEvent(Object.assign(new Event('unhandledrejection'), { reason: new Error('async boom') }));
    expect(getErrors().some(e => e.message === 'async boom')).toBe(true);
    window.dispatchEvent(Object.assign(new Event('unhandledrejection'), { reason: Object.assign(new Error('cancel'), { name: 'AbortError' }) }));
    expect(getErrors().some(e => e.message === 'cancel')).toBe(false);
  });

  it('window error 事件入账', () => {
    installGlobalErrorCapture();
    const ev = new Event('error');
    Object.assign(ev, { message: 'sync boom', filename: 'a.js', lineno: 3, colno: 7 });
    window.dispatchEvent(ev);
    const hit = getErrors().find(e => e.message === 'sync boom');
    expect(hit).toBeTruthy();
    expect(hit.detail).toContain('a.js:3:7');
  });

  it('重复安装幂等', () => {
    installGlobalErrorCapture();
    installGlobalErrorCapture();
    expect(window.__ai2050ErrorCapture).toBe(true);
  });
});
