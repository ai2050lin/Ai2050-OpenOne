import { API_BASE } from '../config.js';
import { useEffect, useMemo, useState } from 'react';


const panelCardStyle = {
  padding: '12px',
  borderRadius: '12px',
  background: 'rgba(255,255,255,0.04)',
  border: '1px solid rgba(255,255,255,0.08)',
};

const inputStyle = {
  width: '100%',
  background: 'rgba(7, 12, 25, 0.82)',
  border: '1px solid rgba(122, 162, 255, 0.24)',
  borderRadius: 10,
  color: '#e4f0ff',
  fontSize: 12,
  padding: '8px 10px',
  outline: 'none',
  boxSizing: 'border-box',
};

const smallActionButtonStyle = {
  padding: '7px 12px',
  borderRadius: '999px',
  border: '1px solid rgba(79, 172, 254, 0.35)',
  background: 'rgba(79, 172, 254, 0.12)',
  color: '#e7f4ff',
  fontSize: '11px',
  fontWeight: 700,
  cursor: 'pointer',
};

function formatScanOptionLabel(file) {
  if (!file) return '';
  const name = file.name || file.path || '未命名文件';
  const sizeKb = Number(file.size_bytes || 0) / 1024;
  return `${name} | ${sizeKb.toFixed(1)} KB`;
}

export default function BasicEncodingPanel({ workspace }) {
  const {
    queryInput,
    setQueryInput,
    queryCategoryInput,
    setQueryCategoryInput,
    handleGenerateQuery,
    querySets,
    queryFeedback,
    scanImportLimit,
    setScanImportLimit,
    scanImportTopK,
    setScanImportTopK,
    animationMode,
    setAnimationMode,
    animationModes,
    handleImportScanJsonText,
    scanImportSummary,
  } = workspace || {};

  const [assetPanelTab, setAssetPanelTab] = useState('manual');
  const [scanFileOptions, setScanFileOptions] = useState([]);
  const [scanFileLoading, setScanFileLoading] = useState(false);
  const [scanFileImporting, setScanFileImporting] = useState(false);
  const [scanFileError, setScanFileError] = useState('');
  const [selectedScanPath, setSelectedScanPath] = useState('');

  const selectedScanMeta = useMemo(
    () => scanFileOptions.find((item) => item.path === selectedScanPath) || null,
    [scanFileOptions, selectedScanPath]
  );

  const refreshScanFileOptions = async () => {
    setScanFileLoading(true);
    setScanFileError('');
    try {
      const res = await fetch(`${API_BASE}/api/main/scan_files?limit=200`);
      const payload = await res.json();
      if (!res.ok) {
        throw new Error(payload?.detail || '读取扫描文件列表失败');
      }
      const files = Array.isArray(payload?.files) ? payload.files : [];
      setScanFileOptions(files);
      setSelectedScanPath((prev) => {
        if (prev && files.some((file) => file.path === prev)) {
          return prev;
        }
        return files[0]?.path || '';
      });
      if (files.length === 0) {
        setScanFileError('当前没有可导入的研究资产文件。');
      }
    } catch (error) {
      setScanFileOptions([]);
      setSelectedScanPath('');
      setScanFileError(`扫描文件列表加载失败: ${error?.message || error}`);
    } finally {
      setScanFileLoading(false);
    }
  };

  useEffect(() => {
    refreshScanFileOptions();
  }, []);

  const handleImportSelectedScanFile = async () => {
    if (!selectedScanPath || !handleImportScanJsonText) {
      return;
    }
    setScanFileImporting(true);
    setScanFileError('');
    try {
      const res = await fetch(`${API_BASE}/api/main/scan_file?path=${encodeURIComponent(selectedScanPath)}`);
      const payload = await res.json();
      if (!res.ok || !payload?.data) {
        throw new Error(payload?.detail || '研究资产读取失败');
      }
      handleImportScanJsonText(JSON.stringify(payload.data), payload.path || selectedScanPath);
    } catch (error) {
      setScanFileError(`导入失败: ${error?.message || error}`);
    } finally {
      setScanFileImporting(false);
    }
  };

  return (
    <div style={{ display: 'grid', gap: 10 }}>
      <div style={panelCardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#d4e3ff', marginBottom: 10 }}>基础编码窗口</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 8, marginBottom: 10 }}>
          {[
            { id: 'manual', label: '手动输入', desc: '输入名词和类别，生成基础神经元模型' },
            { id: 'artifact', label: '测试数据', desc: '导入研究资产并映射到当前主视图' },
          ].map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setAssetPanelTab(tab.id)}
              style={{
                borderRadius: 10,
                border: `1px solid ${assetPanelTab === tab.id ? 'rgba(126, 224, 255, 0.75)' : 'rgba(122, 162, 255, 0.28)'}`,
                background: assetPanelTab === tab.id ? 'rgba(24, 101, 134, 0.36)' : 'rgba(7, 12, 25, 0.82)',
                color: '#dbe9ff',
                padding: '8px 10px',
                cursor: 'pointer',
                textAlign: 'left',
              }}
            >
              <div style={{ fontSize: 12, fontWeight: 700 }}>{tab.label}</div>
              <div style={{ fontSize: 10, color: '#88a6cf', marginTop: 2 }}>{tab.desc}</div>
            </button>
          ))}
        </div>

        {assetPanelTab === 'manual' ? (
          <div style={{ display: 'grid', gap: 10 }}>
            <div style={{ display: 'grid', gap: 8 }}>
              <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
                <div style={{ fontSize: 12, color: '#9eb4dd' }}>名称</div>
                <input
                  value={queryInput || ''}
                  onChange={(e) => setQueryInput?.(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleGenerateQuery?.();
                  }}
                  placeholder="例如：苹果 / 太阳 / 量子"
                  style={inputStyle}
                />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
                <div style={{ fontSize: 12, color: '#9eb4dd' }}>类别</div>
                <input
                  value={queryCategoryInput || ''}
                  onChange={(e) => setQueryCategoryInput?.(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleGenerateQuery?.();
                  }}
                  placeholder="例如：水果 / 天体 / 抽象概念"
                  style={inputStyle}
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
                <div style={{ fontSize: 11, color: '#7ea2c9' }}>{`已生成概念集 ${querySets?.length || 0} 个`}</div>
                <button type="button" onClick={() => handleGenerateQuery?.()} style={smallActionButtonStyle}>
                  生成 3D 模型
                </button>
              </div>
            </div>
            {queryFeedback ? (
              <div style={{ fontSize: 11, color: '#8fd4ff', lineHeight: 1.6 }}>{queryFeedback}</div>
            ) : null}
          </div>
        ) : (
          <div style={{ display: 'grid', gap: 10 }}>
            <div style={{ display: 'grid', gap: 8 }}>
              <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
                <div style={{ fontSize: 12, color: '#9eb4dd' }}>导入数</div>
                <input
                  type="number"
                  min={1}
                  max={120}
                  value={scanImportLimit ?? 12}
                  onChange={(e) => setScanImportLimit?.(Number(e.target.value))}
                  style={inputStyle}
                />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
                <div style={{ fontSize: 12, color: '#9eb4dd' }}>TopK</div>
                <input
                  type="number"
                  min={4}
                  max={64}
                  value={scanImportTopK ?? 12}
                  onChange={(e) => setScanImportTopK?.(Number(e.target.value))}
                  style={inputStyle}
                />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
                <div style={{ fontSize: 12, color: '#9eb4dd' }}>动画</div>
                <select value={animationMode || ''} onChange={(e) => setAnimationMode?.(e.target.value)} style={inputStyle}>
                  {(animationModes || []).map((opt) => (
                    <option key={`basic-asset-anim-${opt.id}`} value={opt.id}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr', gap: 8, alignItems: 'center' }}>
                <div style={{ fontSize: 12, color: '#9eb4dd' }}>文件</div>
                <select
                  value={selectedScanPath}
                  onChange={(e) => setSelectedScanPath(e.target.value)}
                  style={inputStyle}
                  disabled={scanFileLoading || scanFileOptions.length === 0}
                >
                  {scanFileOptions.length === 0 ? (
                    <option value="">{scanFileLoading ? '扫描中...' : '未发现可导入文件'}</option>
                  ) : (
                    scanFileOptions.map((file) => (
                      <option key={file.path} value={file.path}>
                        {formatScanOptionLabel(file)}
                      </option>
                    ))
                  )}
                </select>
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <button
                  type="button"
                  onClick={refreshScanFileOptions}
                  style={{ ...smallActionButtonStyle, flex: '1 1 120px', minWidth: 120 }}
                  disabled={scanFileLoading}
                >
                  {scanFileLoading ? '刷新中...' : '刷新列表'}
                </button>
                <button
                  type="button"
                  onClick={handleImportSelectedScanFile}
                  style={{ ...smallActionButtonStyle, flex: '1 1 120px', minWidth: 120 }}
                  disabled={scanFileImporting || !selectedScanPath}
                >
                  {scanFileImporting ? '导入中...' : '导入并映射到 3D'}
                </button>
              </div>
            </div>

            {selectedScanMeta ? (
              <div style={{ fontSize: 11, color: '#7ea2c9', lineHeight: 1.6, overflowWrap: 'anywhere' }}>
                {`文件：${selectedScanMeta.name} | ${(Number(selectedScanMeta.size_bytes || 0) / 1024).toFixed(1)} KB | ${selectedScanMeta.mtime_iso || ''}`}
              </div>
            ) : null}
            {scanFileError ? (
              <div style={{ fontSize: 11, color: '#ff9fb0' }}>{scanFileError}</div>
            ) : null}
            {scanImportSummary ? (
              <div style={{ fontSize: 11, color: '#7eb8ff', lineHeight: 1.6 }}>
                {`来源：${scanImportSummary.source} | 导入概念集：${scanImportSummary.importedConcepts} | 类别：${scanImportSummary.importedCategories} | 扫描名词总数：${scanImportSummary.totalNouns}`}
              </div>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}
