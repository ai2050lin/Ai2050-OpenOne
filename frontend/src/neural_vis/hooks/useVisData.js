/**
 * useVisData — 多路线数据加载 Hook
 *
 * 数据源注册表负责声明路线和清单格式；dataSourceAdapters 只把已有证据
 * 转换成统一的 3D 观测图，不提升原始证据等级。
 */
import { useCallback, useState } from 'react';
import {
  normalizeManifestEntries,
  normalizeVisualizationPayload,
} from '../dataSourceAdapters';
import { researchAssetUrl } from '../../config/researchAssets';

const SOURCE_REGISTRY_PATH = researchAssetUrl('source_registry.json');

const LEGACY_SOURCE = {
  id: 'glm5_causal_fiber_atlas',
  route_id: 'glm5',
  route_label: 'GLM5 路线',
  label: '因果纤维历史图谱',
  description: '兼容旧版单清单加载。',
  manifest_path: researchAssetUrl('manifest.json'),
  manifest_schema: 'vis_data_manifest_v1',
  manifest_adapter: 'files',
  payload_adapter: 'atlas_graph',
  data_base_path: researchAssetUrl(),
  models: ['qwen3', 'glm4', 'deepseek7b'],
  evidence_scope: '历史路线图谱；按各阶段原始证据等级解释',
  color: '#a78bfa',
};

const SUPPORTED_SCHEMAS = new Set([
  '1.0',
  '2.0',
  '2.0.0',
  'atlas_graph_v1',
  'real_component_trace.v1',
  'mechanism_trace_v1',
  'mechanism_case.v1',
  'research_kernel_manifest.v1',
  'pattern_family_neuron_atlas.v1',
  'neuron_atlas_partition.v1',
]);

function isSupportedPayload(data, source = null) {
  const version = data?.schema_version || '1.0';
  if (SUPPORTED_SCHEMAS.has(version)) return true;
  return source?.payload_adapter === 'atlas_graph'
    && data?.graph
    && Array.isArray(data.graph.nodes)
    && (Array.isArray(data.graph.edges) || Array.isArray(data.graph.links));
}

async function fetchJson(path) {
  const response = await fetch(researchAssetUrl(path), { cache: 'no-store' });
  if (!response.ok) throw new Error(`${path} 加载失败 (${response.status})`);
  return response.json();
}

export default function useVisData() {
  const [dataSources, setDataSources] = useState([]);
  const [activeSource, setActiveSource] = useState(null);
  const [dataFiles, setDataFiles] = useState([]);
  const [activeData, setActiveData] = useState(null);
  const [activeFileMeta, setActiveFileMeta] = useState(null);
  const [loading, setLoading] = useState(false);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [error, setError] = useState(null);
  const [registryWarning, setRegistryWarning] = useState(null);

  const loadSourceManifest = useCallback(async (source, { clearActive = true } = {}) => {
    setSourceLoading(true);
    setError(null);
    try {
      const manifest = await fetchJson(source.manifest_path);
      if (source.manifest_schema && manifest.schema_version !== source.manifest_schema) {
        throw new Error(
          `${source.label} 清单模式不匹配：期望 ${source.manifest_schema}，实际 ${manifest.schema_version || 'unknown'}`
        );
      }
      const entries = normalizeManifestEntries(source, manifest);
      setActiveSource({ ...source, dataset_count: entries.length });
      setDataFiles(entries);
      if (clearActive) {
        setActiveData(null);
        setActiveFileMeta(null);
      }
      return entries;
    } catch (loadError) {
      setDataFiles([]);
      setError(loadError.message);
      throw loadError;
    } finally {
      setSourceLoading(false);
    }
  }, []);

  const loadDataManifest = useCallback(async () => {
    setSourceLoading(true);
    setError(null);
    try {
      let registry;
      try {
        registry = await fetchJson(SOURCE_REGISTRY_PATH);
        if (registry.schema_version !== 'vis_data_source_registry.v1') {
          throw new Error(`Unsupported source registry: ${registry.schema_version || 'unknown'}`);
        }
        setRegistryWarning(null);
      } catch (registryError) {
        registry = {
          default_source_id: LEGACY_SOURCE.id,
          sources: [LEGACY_SOURCE],
        };
        setRegistryWarning(`多路线注册表不可用，已回退旧 GLM5 清单：${registryError.message}`);
      }

      const sources = Array.isArray(registry.sources) && registry.sources.length
        ? registry.sources
        : [LEGACY_SOURCE];
      setDataSources(sources);
      const initialSource = sources.find((source) => source.id === registry.default_source_id) || sources[0];
      await loadSourceManifest(initialSource);
    } catch (loadError) {
      setError(loadError.message);
    } finally {
      setSourceLoading(false);
    }
  }, [loadSourceManifest]);

  const selectDataSource = useCallback(async (sourceId) => {
    const source = dataSources.find((candidate) => candidate.id === sourceId);
    if (!source || source.id === activeSource?.id) return;
    try {
      await loadSourceManifest(source);
    } catch {
      // loadSourceManifest 已保留可见错误状态。
    }
  }, [activeSource?.id, dataSources, loadSourceManifest]);

  const refreshDataSource = useCallback(async () => {
    if (!activeSource) return;
    try {
      await loadSourceManifest(activeSource, { clearActive: false });
    } catch {
      // loadSourceManifest 已保留可见错误状态。
    }
  }, [activeSource, loadSourceManifest]);

  const loadDataFile = useCallback(async (fileOrPath) => {
    const fileMeta = typeof fileOrPath === 'string'
      ? dataFiles.find((file) => [file.id, file.filename, file.path].includes(fileOrPath)) || {
        id: fileOrPath,
        filename: fileOrPath,
        label: fileOrPath,
        path: researchAssetUrl(fileOrPath),
      }
      : fileOrPath;

    setLoading(true);
    setError(null);
    try {
      const data = await fetchJson(fileMeta.path);
      const version = data.schema_version || '1.0';
      if (!isSupportedPayload(data, activeSource)) throw new Error(`Unsupported schema: ${version}`);
      const normalized = normalizeVisualizationPayload(data, fileMeta, activeSource);
      setActiveData(normalized);
      setActiveFileMeta({
        ...fileMeta,
        source_label: activeSource?.label,
        route_label: activeSource?.route_label,
      });
    } catch (loadError) {
      setError(loadError.message);
    } finally {
      setLoading(false);
    }
  }, [activeSource, dataFiles]);

  const loadLocalFile = useCallback((file) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const data = JSON.parse(event.target.result);
        const version = data.schema_version || '1.0';
        if (!isSupportedPayload(data)) throw new Error(`Unsupported schema: ${version}`);
        const fileMeta = { id: file.name, filename: file.name, label: file.name, source: 'local' };
        setActiveData(normalizeVisualizationPayload(data, fileMeta, null));
        setActiveFileMeta({ ...fileMeta, route_label: '本地文件', source_label: '本地文件' });
        setError(null);
      } catch (loadError) {
        setError(loadError.message);
      }
    };
    reader.onerror = () => setError(`${file.name} 读取失败`);
    reader.readAsText(file);
  }, []);

  return {
    dataSources,
    activeSource,
    dataFiles,
    activeData,
    activeFileMeta,
    loading,
    sourceLoading,
    error,
    registryWarning,
    loadDataManifest,
    selectDataSource,
    refreshDataSource,
    loadDataFile,
    loadLocalFile,
    setActiveData,
    setError,
    setActiveFileMeta,
  };
}
