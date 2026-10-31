/* 机制报告的读取 hook（非组件）：全部经 rdc_fusion/api.js 的 useApi（本机数据包优先，离线客户端只读数据包）。 */
import { useApi } from '../rdc_fusion/api.js';
import { isConflict } from './mechUtil.js';

export function useMechIndex() {
  const api = useApi('/api/mechanism/index');
  return { ...api, cases: api.data?.cases || [] };
}

export function useMechCase(id) {
  const api = useApi(id ? `/api/mechanism/case/${encodeURIComponent(id)}` : null);
  return { ...api, report: api.data?.report || null };
}

/* 冲突案例：当前选中（不在列表里时取第一个）及其报告 */
export function useConflictCase(wanted) {
  const index = useMechIndex();
  const cases = index.cases.filter(c => isConflict(c) && !c.error);
  const caseId = cases.some(c => c.id === wanted) ? wanted : (cases[0]?.id || '');
  const api = useMechCase(caseId);
  return { index, cases, caseId, api, report: api.report };
}
