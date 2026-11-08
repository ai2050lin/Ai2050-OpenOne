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

/* 页内页签（Phase 338）：PageTabs 登记当前选中的页签，“?”窗口按它取这一页签的说明 */
let tab = null;
const tabListeners = new Set();
export function setPageTab(id) { tab = id; tabListeners.forEach(fn => fn()); }
export function clearPageTab(id) { if (tab === id) { tab = null; tabListeners.forEach(fn => fn()); } }
export function usePageTab() {
  return useSyncExternalStore(fn => { tabListeners.add(fn); return () => tabListeners.delete(fn); }, () => tab, () => null);
}
