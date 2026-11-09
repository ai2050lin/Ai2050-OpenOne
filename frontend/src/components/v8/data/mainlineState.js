import { createContext, useContext } from 'react';
import release from '../../../../../shared/dist_protocol/mainline_release.json';

export const bundledState = release.state;
export const MainlineContext = createContext({ ...bundledState.view, state: bundledState,
  revision: null, source: 'bundled', error: '', refresh: () => {} });
export const useMainline = () => useContext(MainlineContext);

export function atMainlineRevision(resource, revision) {
  if (!revision || !resource.data || resource.data.mainline_revision === revision) return resource;
  return { ...resource, data: null, error: '研究列表与中心记录尚未同步，请同步研究记录后重试。' };
}

export async function readSnapshot(envelope, previous) {
  if (envelope?.schema !== 'mainline_snapshot.v1' || typeof envelope.record !== 'string') throw new Error('中心研究记录格式不完整');
  const bytes = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(envelope.record));
  const hash = Array.from(new Uint8Array(bytes), b => b.toString(16).padStart(2, '0')).join('');
  if (hash !== envelope.revision) throw new Error('研究记录校验失败');
  const state = JSON.parse(envelope.record);
  if (state.schema !== 'research_mainline.v1' || state.kind !== 'real' || !Number.isInteger(state.sequence) || state.sequence < 1
    || envelope.sequence !== state.sequence || !/^[0-9a-f]{64}$/.test(state.model_snapshot)) throw new Error('研究记录身份不一致');
  for (const [key, value] of Object.entries(bundledState.view)) {
    if (Array.isArray(value) ? !Array.isArray(state.view?.[key]) : typeof state.view?.[key] !== 'object' || !state.view[key]) throw new Error(`研究记录缺少 ${key}`);
  }
  if (previous && (state.sequence < previous.state.sequence || (state.sequence === previous.state.sequence && hash !== previous.revision))) throw new Error('中心记录版本倒退或同版本冲突，保留已校验记录');
  const boxes = new Map(state.view.LOCKBOXES.map(b => [b.id, b.state]));
  for (const box of state.custody || []) if (boxes.get(box.id) !== box.state || (box.state === 'opened' && !box.opened_at)) throw new Error('开封记录与页面状态不一致');
  for (const box of [...bundledState.custody, ...(previous?.state.custody || [])]) if (box.state === 'opened' && boxes.get(box.id) !== 'opened') throw new Error('已开封的材料不能重新显示为封存');
  return { state, revision: hash };
}
