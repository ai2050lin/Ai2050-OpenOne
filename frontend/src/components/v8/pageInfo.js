/* 当前页面的标题与说明（Phase 338）：页面上不再显示标题和说明，由 PageHead 登记到这里，右上角“?”窗口读取显示。 */
import { useSyncExternalStore } from 'react';

let current = null;
const listeners = new Set();
const emit = () => listeners.forEach(fn => fn());

export function setPageInfo(info) { current = info; emit(); }
export function clearPageInfo(info) { if (current === info) { current = null; emit(); } }
export function usePageInfo() {
  return useSyncExternalStore(fn => { listeners.add(fn); return () => listeners.delete(fn); }, () => current, () => null);
}
