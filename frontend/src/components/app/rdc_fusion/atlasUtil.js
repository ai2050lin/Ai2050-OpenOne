/* 语言谱图共享工具（非组件模块）。 */
import { CATEGORICAL } from './Space3D.jsx';

/* 模式族 → 类别色（按名称排序后固定顺序分配；颜色跟随模式族，不随筛选重排） */
export function familyPalette(families) {
  return Object.fromEntries([...families].sort().map((f, i) => [f, CATEGORICAL[i] || '#94a3b8']));
}

export const HYP_STATUS = { supported: ['pass', '支持'], partial: ['partial', '部分支持'], negative: ['fail', '未支持'] };
