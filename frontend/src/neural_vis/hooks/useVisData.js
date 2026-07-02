/**
 * useVisData — 数据加载Hook
 * 支持 Schema v1.0、v2.0 和 atlas_graph_v1
 */
import { useState, useCallback } from 'react';

export default function useVisData() {
  const [dataFiles, setDataFiles] = useState([]);
  const [activeData, setActiveData] = useState(null);
  const [activeFileMeta, setActiveFileMeta] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const isSupportedSchema = (version) => ['1.0', '2.0', 'atlas_graph_v1'].includes(version);

  const loadDataManifest = useCallback(async () => {
    try {
      const resp = await fetch('/vis_data/manifest.json');
      const manifest = await resp.json();
      setDataFiles(manifest.files || []);
    } catch {
      setDataFiles([]);
    }
  }, []);

  const loadDataFile = useCallback(async (filepath) => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`/vis_data/${filepath}`);
      const data = await resp.json();
      const version = data.schema_version || '1.0';
      if (!isSupportedSchema(version)) {
        throw new Error(`Unsupported schema: ${version}`);
      }
      setActiveData(data);
      setActiveFileMeta(dataFiles.find((file) => file.filename === filepath) || { filename: filepath });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [dataFiles]);

  const loadLocalFile = useCallback((file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        const version = data.schema_version || '1.0';
        if (!isSupportedSchema(version)) {
          throw new Error(`Unsupported schema: ${version}`);
        }
        setActiveData(data);
        setActiveFileMeta({ filename: file.name, label: file.name, source: 'local' });
      } catch (err) {
        setError(err.message);
      }
    };
    reader.readAsText(file);
  }, []);

  return { dataFiles, activeData, activeFileMeta, loading, error, loadDataManifest, loadDataFile, loadLocalFile, setActiveData, setError, setActiveFileMeta };
}
