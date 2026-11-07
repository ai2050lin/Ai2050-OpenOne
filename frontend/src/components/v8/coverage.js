/* 覆盖计数钩子：研发页与空间共用，口径与“进展 · 总览”相同。 */
import { useMemo } from 'react';

import { useApi } from '../app/rdc_fusion/api.js';
import { countMeasuredLanguageItems, languageCatalogCount } from './LanguageCatalog.jsx';
import { defaultModel, itemLevel } from './v8Util.js';

/* 语言目录与 MLP 解释检验的覆盖计数，口径与“进展 · 总览”相同。 */
export function useCoverage() {
  const treeApi = useApi('/api/lang/tree');
  const skApi = useApi('/api/model/skeleton');
  const tree = treeApi.data && !treeApi.data.pending ? treeApi.data : null;
  const skeletons = useMemo(() => Object.fromEntries((skApi.data?.models || []).map(s => [s.model, s])), [skApi.data]);
  const model = tree ? defaultModel(tree, skeletons) : '';
  const skeleton = skeletons[model];
  const language = tree?.models?.includes(model)
    ? countMeasuredLanguageItems(new Set(tree.items.filter(item => itemLevel(item, model) >= 1).map(item => item.id))) : null;
  const layers = skeleton?.n_layers, width = skeleton?.d_ff;
  const neuronTotal = layers > 0 && width > 0 ? layers * width : null;
  const censusKey = skeleton?.census_run ? skeleton.census_key : null;
  const checks = useApi(censusKey && neuronTotal ? `/api/neurons/explained?model_key=${encodeURIComponent(censusKey)}` : null);
  const valid = Boolean(censusKey) && checks.data?.model_key === censusKey && Array.isArray(checks.data?.results);
  const checked = valid ? new Set(checks.data.results.filter(row => Number.isInteger(row.block) && row.block >= 0 && row.block < layers &&
    Number.isInteger(row.unit) && row.unit >= 0 && row.unit < width).map(row => `${row.block}:${row.unit}`)).size : null;
  return { model, language, languageTotal: languageCatalogCount, checked, neuronTotal };
}
