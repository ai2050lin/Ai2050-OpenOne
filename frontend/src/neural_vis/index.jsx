/**
 * NeuralVis3DApp — 3D可视化主组件 v3.0
 * 支持 Schema v1.0 + v2.0 + v3.0(多维度视角 + DNN层可视化 + 动画系统)
 * 
 * 可视化类型:
 *   v1.0: trajectory, point_cloud, heatmap_3d, flow, layer_stack
 *   v2.0: subspace_decomposition, force_line, grammar_role_matrix, causal_chain, dark_matter_flow
 *   v3.0: 多维度视角切换, 动画场景演示
 */
import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Stars } from '@react-three/drei';
import { Database, FolderOpen, RefreshCw, Search } from 'lucide-react';
import * as THREE from 'three';

// 模块化导入
import TrajectoryRenderer from './renderers/TrajectoryRenderer';
import PointCloudRenderer from './renderers/PointCloudRenderer';
import LayerStackRenderer from './renderers/LayerStackRenderer';
import Heatmap3DRenderer from './renderers/Heatmap3DRenderer';
import FlowRenderer from './renderers/FlowRenderer';
import SubspaceRenderer from './renderers/SubspaceRenderer';
import ForceLineRenderer from './renderers/ForceLineRenderer';
import GrammarRoleMatrixRenderer from './renderers/GrammarRoleMatrixRenderer';
import CausalChainRenderer from './renderers/CausalChainRenderer';
import DarkMatterFlowRenderer from './renderers/DarkMatterFlowRenderer';
import AtlasGraphRenderer from './renderers/AtlasGraphRenderer';
import SceneHelpers from './components/SceneHelpers';
import HoverTooltip from './components/HoverTooltip';
import useVisData from './hooks/useVisData';
import {
  CATEGORY_COLORS, deltaCosToColor, cosWuToColor, SUBSPACE_COLORS,
  DIMENSION_VIEWS, ANIMATION_SCENARIOS, LAYER_FUNCTIONS,
  layerToFuncColor, layerToFuncLabel,
} from './utils/constants';

// ==================== 样式常量 ====================
const S = {
  panel: { background: '#0f172a', color: '#e2e8f0', fontFamily: 'system-ui, sans-serif' },
  leftPanel: { width: 300, padding: 12, borderRight: '1px solid #1e293b', overflowY: 'auto', flexShrink: 0 },
  section: { marginBottom: 16 },
  sectionTitle: { fontSize: 12, color: '#94a3b8', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.5px' },
  btn: (active, color) => ({
    display: 'block', width: '100%', padding: '5px 8px', marginBottom: 3,
    background: active ? (color ? `${color}22` : '#1e40af') : '#1e293b',
    border: active ? `1px solid ${color || '#3b82f6'}` : '1px solid #334155',
    borderRadius: 4, color: active ? (color || '#bfdbfe') : '#94a3b8',
    cursor: 'pointer', textAlign: 'left', fontSize: 11, transition: 'all 0.15s',
  }),
  btnSmall: (active) => ({
    padding: '3px 6px', background: active ? '#1e40af' : '#1e293b',
    border: active ? '1px solid #3b82f6' : '1px solid #334155',
    borderRadius: 3, color: active ? '#bfdbfe' : '#94a3b8',
    cursor: 'pointer', fontSize: 10,
  }),
  badge: (color) => ({
    display: 'inline-block', width: 8, height: 8, borderRadius: 4,
    background: color, marginRight: 6, verticalAlign: 'middle',
  }),
  progressBg: { width: '100%', height: 4, background: '#1e293b', borderRadius: 2, overflow: 'hidden' },
  progressBar: (pct, color) => ({ width: `${pct}%`, height: '100%', background: color || '#3b82f6', borderRadius: 2, transition: 'width 0.3s' }),
  tag: (bgColor, textColor) => ({
    display: 'inline-block', padding: '1px 6px', borderRadius: 3,
    background: bgColor || '#1e293b', color: textColor || '#94a3b8',
    fontSize: 9, marginRight: 4, marginBottom: 2,
  }),
};

// ==================== 主组件 ====================
export default function NeuralVis3DApp() {
  const {
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
  } = useVisData();
  const fileInputRef = useRef();
  const [dataSearch, setDataSearch] = useState('');

  // ---- 视图状态 ----
  const [viewMode, setViewMode] = useState('all');
  const [activeDimension, setActiveDimension] = useState(null); // 5大维度
  const [activeSubView, setActiveSubView] = useState(null);     // 维度内子视角

  // ---- 动画状态 ----
  const [animProgress, setAnimProgress] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [activeScenario, setActiveScenario] = useState(null);
  const animRef = useRef(null);
  const startTimeRef = useRef(null);

  // ---- 交互状态 ----
  const [hoveredInfo, setHoveredInfo] = useState(null);
  const selectedLayers = null;
  const [leftPanelTab, setLeftPanelTab] = useState('dimension'); // dimension | renderer | animation

  useEffect(() => { loadDataManifest(); }, [loadDataManifest]);

  const filteredDataFiles = useMemo(() => {
    const query = dataSearch.trim().toLocaleLowerCase();
    if (!query) return dataFiles;
    return dataFiles.filter((file) => [
      file.label,
      file.filename,
      file.model,
      file.family_id,
      file.phase,
      file.run_id,
    ].some((value) => String(value ?? '').toLocaleLowerCase().includes(query)));
  }, [dataFiles, dataSearch]);

  const routeCount = useMemo(
    () => new Set(dataSources.map((source) => source.route_id)).size,
    [dataSources]
  );

  const visualizations = activeData?.visualizations || [];
  const schemaVersion = activeData?.schema_version || '1.0';
  const nLayers = activeData?.model_info?.n_layers || 36;
  const isAtlasGraph = schemaVersion === 'atlas_graph_v1';
  const atlasGraph = isAtlasGraph ? activeData?.graph : null;
  const atlasNodeCount = atlasGraph?.nodes?.length || 0;
  const atlasEdgeCount = atlasGraph?.edges?.length || atlasGraph?.links?.length || 0;

  // ---- 按类型分类可视化对象 ----
  const byType = {
    trajectory: visualizations.filter(v => v.type === 'trajectory'),
    point_cloud: visualizations.filter(v => v.type === 'point_cloud'),
    heatmap_3d: visualizations.filter(v => v.type === 'heatmap_3d'),
    flow: visualizations.filter(v => v.type === 'flow'),
    layer_stack: visualizations.filter(v => v.type === 'layer_stack'),
    subspace_decomposition: visualizations.filter(v => v.type === 'subspace_decomposition'),
    force_line: visualizations.filter(v => v.type === 'force_line'),
    grammar_role_matrix: visualizations.filter(v => v.type === 'grammar_role_matrix'),
    causal_chain: visualizations.filter(v => v.type === 'causal_chain'),
    dark_matter_flow: visualizations.filter(v => v.type === 'dark_matter_flow'),
  };

  // ---- 维度视角过滤 ----
  const getActiveRenderers = useCallback(() => {
    if (!activeDimension || !activeSubView) return null;
    const dim = DIMENSION_VIEWS[activeDimension];
    const sub = dim?.subViews[activeSubView];
    return sub?.renderers || null;
  }, [activeDimension, activeSubView]);

  const filterByMode = (type) => {
    if (activeDimension && activeSubView) {
      const activeRenderers = getActiveRenderers();
      if (activeRenderers) return activeRenderers.includes(type);
    }
    return viewMode === 'all' || viewMode === type;
  };

  const trajectories = byType.trajectory.filter(() => filterByMode('trajectory'));
  const pointClouds = byType.point_cloud.filter(() => filterByMode('point_cloud'));
  const heatmaps = byType.heatmap_3d.filter(() => filterByMode('heatmap'));
  const flows = byType.flow.filter(() => filterByMode('flow'));
  const layerStacks = byType.layer_stack;
  const subspaceDecomps = byType.subspace_decomposition.filter(() => filterByMode('subspace'));
  const forceLines = byType.force_line.filter(() => filterByMode('force_line'));
  const grammarMatrices = byType.grammar_role_matrix.filter(() => filterByMode('grammar'));
  const causalChains = byType.causal_chain.filter(() => filterByMode('causal'));
  const darkMatterFlows = byType.dark_matter_flow.filter(() => filterByMode('dark_matter'));

  // ---- 动画场景系统 ----
  const startScenario = useCallback((scenarioKey) => {
    const scenario = ANIMATION_SCENARIOS[scenarioKey];
    if (!scenario) return;
    setActiveScenario(scenarioKey);
    setAnimProgress(0);
    setPlaying(true);
    startTimeRef.current = Date.now();
  }, []);

  const stopScenario = useCallback(() => {
    setPlaying(false);
    setActiveScenario(null);
    setAnimProgress(1);
    if (animRef.current) cancelAnimationFrame(animRef.current);
  }, []);

  useEffect(() => {
    if (!playing || !activeScenario) return;
    const scenario = ANIMATION_SCENARIOS[activeScenario];
    if (!scenario) return;

    const duration = scenario.duration * 1000; // ms
    const tick = () => {
      const elapsed = Date.now() - startTimeRef.current;
      const progress = Math.min(1, elapsed / duration);
      setAnimProgress(progress);
      if (progress < 1) {
        animRef.current = requestAnimationFrame(tick);
      } else {
        setPlaying(false);
      }
    };
    animRef.current = requestAnimationFrame(tick);
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [playing, activeScenario]);

  // 简单播放/暂停 (非场景模式)
  useEffect(() => {
    if (!playing || activeScenario) return;
    const timer = setInterval(() => {
      setAnimProgress(prev => {
        if (prev >= 1) { setPlaying(false); return 1; }
        return prev + 0.01;
      });
    }, 50);
    return () => clearInterval(timer);
  }, [playing, activeScenario]);

  // 获取当前动画阶段信息
  const getCurrentPhase = () => {
    if (!activeScenario) return null;
    const scenario = ANIMATION_SCENARIOS[activeScenario];
    return scenario.phases.find(p => animProgress >= p.start && animProgress < p.end) || scenario.phases[scenario.phases.length - 1];
  };

  const getCount = (key) => {
    if (key === 'all') return visualizations.length;
    if (key === 'heatmap') return byType.heatmap_3d.length;
    if (key === 'subspace') return byType.subspace_decomposition.length;
    if (key === 'force_line') return byType.force_line.length;
    if (key === 'grammar') return byType.grammar_role_matrix.length;
    if (key === 'causal') return byType.causal_chain.length;
    if (key === 'dark_matter') return byType.dark_matter_flow.length;
    return byType[key]?.length || 0;
  };

  // ==================== 渲染器模式按钮 ====================
  const VIEW_MODES = [
    { key: 'all', label: '🔍 全部', icon: '🔍' },
    { key: 'trajectory', label: '📈 轨迹', icon: '📈' },
    { key: 'point_cloud', label: '⚪ 点云', icon: '⚪' },
    { key: 'heatmap', label: '📊 热力图', icon: '📊' },
    { key: 'flow', label: '🔀 信息流', icon: '🔀' },
    { key: 'subspace', label: '🧬 子空间', icon: '🧬' },
    { key: 'force_line', label: '⚡ 力线', icon: '⚡' },
    { key: 'grammar', label: '📝 语法矩阵', icon: '📝' },
    { key: 'causal', label: '🔗 因果链', icon: '🔗' },
    { key: 'dark_matter', label: '🌑 暗物质', icon: '🌑' },
  ];

  return (
    <div style={{ display: 'flex', height: '100vh', ...S.panel }}>
      {/* ==================== 左侧面板 ==================== */}
      <div style={S.leftPanel}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}>
          <h2 style={{ fontSize: 15, margin: 0, color: '#60a5fa', flex: 1 }}>Neural Vis 3D</h2>
          <span style={{ fontSize: 9, color: '#475569' }}>v3.0</span>
        </div>

        {/* ---- 数据源 ---- */}
        <div style={S.section}>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: 7 }}>
            <h3 style={{ ...S.sectionTitle, margin: 0, flex: 1 }}>测试路线数据源</h3>
            <span style={{ fontSize: 9, color: '#64748b' }}>{routeCount} 条路线 · {dataSources.length} 类数据</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 30px', gap: 5, marginBottom: 6 }}>
            <select
              aria-label="测试路线数据源"
              value={activeSource?.id || ''}
              disabled={sourceLoading || dataSources.length === 0}
              onChange={(event) => {
                setDataSearch('');
                selectDataSource(event.target.value);
              }}
              style={{
                minWidth: 0,
                height: 30,
                padding: '0 7px',
                background: '#111c30',
                border: '1px solid #334155',
                borderRadius: 4,
                color: '#dbeafe',
                fontSize: 10,
              }}
            >
              {dataSources.map((source) => (
                <option key={source.id} value={source.id}>
                  {source.route_label} · {source.label}
                </option>
              ))}
            </select>
            <button
              type="button"
              title="刷新当前数据源"
              aria-label="刷新当前数据源"
              disabled={!activeSource || sourceLoading}
              onClick={refreshDataSource}
              style={{
                width: 30,
                height: 30,
                display: 'grid',
                placeItems: 'center',
                background: '#1e293b',
                border: '1px solid #334155',
                borderRadius: 4,
                color: '#93c5fd',
                cursor: sourceLoading ? 'wait' : 'pointer',
              }}
            >
              <RefreshCw size={14} className={sourceLoading ? 'vis-source-spin' : undefined} />
            </button>
          </div>
          {activeSource && (
            <div style={{ padding: '6px 7px', marginBottom: 6, borderLeft: `2px solid ${activeSource.color || '#60a5fa'}`, background: '#111827' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 5, color: '#cbd5e1', fontSize: 10, fontWeight: 700 }}>
                <Database size={12} color={activeSource.color || '#60a5fa'} />
                <span>{activeSource.route_label}</span>
                <span style={{ marginLeft: 'auto', color: '#94a3b8', fontWeight: 500 }}>{dataFiles.length} 组</span>
              </div>
              <div style={{ color: '#64748b', fontSize: 9, lineHeight: 1.45, marginTop: 3 }}>{activeSource.description}</div>
              <div style={{ color: '#475569', fontSize: 8, lineHeight: 1.4, marginTop: 3 }}>
                模型：{activeSource.models?.join(' / ') || '未声明'}
              </div>
            </div>
          )}
          {registryWarning && (
            <div style={{ color: '#fbbf24', fontSize: 9, lineHeight: 1.4, marginBottom: 6 }}>{registryWarning}</div>
          )}
          <button onClick={() => fileInputRef.current?.click()}
            style={{ width: '100%', height: 30, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, background: '#1e293b', border: '1px solid #334155', borderRadius: 4, color: '#e2e8f0', cursor: 'pointer', marginBottom: 6, fontSize: 10 }}>
            <FolderOpen size={14} /> 加载本地 JSON 文件
          </button>
          <input ref={fileInputRef} type="file" accept=".json" style={{ display: 'none' }}
            onChange={(e) => e.target.files[0] && loadLocalFile(e.target.files[0])} />
          {dataFiles.length > 0 && (
            <>
              <label style={{ position: 'relative', display: 'block', marginBottom: 5 }}>
                <Search size={12} style={{ position: 'absolute', left: 7, top: 8, color: '#64748b' }} />
                <input
                  type="search"
                  aria-label="筛选测试数据"
                  value={dataSearch}
                  onChange={(event) => setDataSearch(event.target.value)}
                  placeholder="筛选模式族、模型或阶段"
                  style={{
                    width: '100%',
                    height: 28,
                    padding: '0 7px 0 25px',
                    background: '#0b1220',
                    border: '1px solid #1e293b',
                    borderRadius: 4,
                    color: '#cbd5e1',
                    fontSize: 9,
                    boxSizing: 'border-box',
                  }}
                />
              </label>
              <div style={{ maxHeight: 174, overflowY: 'auto', paddingRight: 2 }} data-testid="route-dataset-list">
                {filteredDataFiles.map((file) => {
                  const isActive = activeFileMeta?.id === file.id || activeFileMeta?.path === file.path;
                  return (
                    <button key={file.id || file.path} onClick={() => loadDataFile(file)}
                      style={{
                        display: 'block', width: '100%', minHeight: 32, padding: '5px 6px',
                        background: isActive ? '#172554' : '#0f172a',
                        border: isActive ? '1px solid #3b82f6' : '1px solid #1e293b',
                        borderRadius: 3, color: isActive ? '#dbeafe' : '#94a3b8',
                        cursor: 'pointer', textAlign: 'left', fontSize: 9, marginBottom: 3,
                      }}>
                      <span style={{ display: 'block', overflowWrap: 'anywhere' }}>{file.label || file.filename}</span>
                      {(file.model || file.phase || file.node_count) && (
                        <span style={{ display: 'block', color: '#475569', fontSize: 8, marginTop: 2 }}>
                          {[file.model, file.phase ? `Phase ${file.phase}` : null, file.node_count ? `${file.node_count} 节点` : null].filter(Boolean).join(' · ')}
                        </span>
                      )}
                    </button>
                  );
                })}
                {filteredDataFiles.length === 0 && (
                  <div style={{ padding: 8, color: '#64748b', fontSize: 9, textAlign: 'center' }}>没有匹配的数据集</div>
                )}
              </div>
            </>
          )}
        </div>

        {/* ---- 左面板Tab切换 ---- */}
        <div style={{ display: 'flex', marginBottom: 12, borderBottom: '1px solid #1e293b' }}>
          {[
            { key: 'dimension', label: '🔬 维度' },
            { key: 'renderer', label: '🎨 渲染器' },
            { key: 'animation', label: '🎬 动画' },
          ].map(t => (
            <button key={t.key} onClick={() => setLeftPanelTab(t.key)}
              style={{
                flex: 1, padding: '6px 2px', background: 'transparent',
                border: 'none', borderBottom: leftPanelTab === t.key ? '2px solid #60a5fa' : '2px solid transparent',
                color: leftPanelTab === t.key ? '#60a5fa' : '#64748b',
                cursor: 'pointer', fontSize: 11, fontWeight: 'bold',
              }}>
              {t.label}
            </button>
          ))}
        </div>

        {/* ========== Tab 1: 维度视角 ========== */}
        {leftPanelTab === 'dimension' && (
          <div>
            {Object.entries(DIMENSION_VIEWS).map(([dimKey, dim]) => (
              <div key={dimKey} style={S.section}>
                <button
                  onClick={() => {
                    if (activeDimension === dimKey) {
                      setActiveDimension(null);
                      setActiveSubView(null);
                    } else {
                      setActiveDimension(dimKey);
                      setActiveSubView(null);
                    }
                  }}
                  style={{
                    ...S.btn(activeDimension === dimKey, dim.color),
                    fontWeight: 'bold', fontSize: 12,
                  }}
                >
                  <span style={S.badge(dim.color)} />
                  {dim.icon} {dim.label}
                  <span style={{ float: 'right', fontSize: 9, color: '#475569' }}>▼</span>
                </button>

                {/* 维度描述 */}
                {activeDimension === dimKey && (
                  <div style={{ fontSize: 10, color: '#64748b', padding: '2px 8px 6px', fontStyle: 'italic' }}>
                    {dim.description}
                  </div>
                )}

                {/* 子视角列表 */}
                {activeDimension === dimKey && Object.entries(dim.subViews).map(([subKey, sub]) => (
                  <button key={subKey}
                    onClick={() => setActiveSubView(activeSubView === subKey ? null : subKey)}
                    style={{
                      ...S.btn(activeSubView === subKey, dim.color),
                      marginLeft: 16, fontSize: 11,
                    }}
                  >
                    {sub.icon} {sub.label}
                    <span style={{ float: 'right', fontSize: 9, color: '#475569' }}>
                      {sub.renderers.length}种
                    </span>
                  </button>
                ))}

                {/* 子视角详情 */}
                {activeDimension === dimKey && activeSubView && dim.subViews[activeSubView] && (
                  <div style={{ marginLeft: 16, padding: '4px 8px', fontSize: 10, color: '#94a3b8', lineHeight: 1.5, background: '#0f172a', borderRadius: 4, marginTop: 4 }}>
                    <div>{dim.subViews[activeSubView].description}</div>
                    <div style={{ marginTop: 4 }}>
                      {dim.subViews[activeSubView].renderers.map(r => (
                        <span key={r} style={S.tag(`${dim.color}33`, dim.color)}>{r}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* 重置维度选择 */}
            {activeDimension && (
              <button onClick={() => { setActiveDimension(null); setActiveSubView(null); }}
                style={{ width: '100%', padding: '6px', background: '#1e293b', border: '1px solid #334155', borderRadius: 4, color: '#94a3b8', cursor: 'pointer', fontSize: 11 }}>
                ✕ 清除维度筛选
              </button>
            )}
          </div>
        )}

        {/* ========== Tab 2: 渲染器选择 ========== */}
        {leftPanelTab === 'renderer' && (
          <div>
            {/* 层功能图例 */}
            <div style={S.section}>
              <h3 style={S.sectionTitle}>层功能分区</h3>
              {Object.entries(LAYER_FUNCTIONS).map(([key, func]) => (
                <div key={key} style={{ display: 'flex', alignItems: 'center', marginBottom: 3, fontSize: 10 }}>
                  <span style={{ ...S.badge(func.color), width: 10, height: 10, borderRadius: 2 }} />
                  <span style={{ color: func.color, width: 60, flexShrink: 0 }}>{func.label}</span>
                  <span style={{ color: '#475569' }}>L{func.range[0]}-L{func.range[1]}</span>
                </div>
              ))}
            </div>

            {/* 渲染器模式 */}
            <div style={S.section}>
              <h3 style={S.sectionTitle}>渲染器模式</h3>
              {VIEW_MODES.map(mode => {
                const count = getCount(mode.key);
                const isActive = viewMode === mode.key && !activeDimension;
                return (
                  <button key={mode.key}
                    onClick={() => { setViewMode(mode.key); setActiveDimension(null); setActiveSubView(null); }}
                    style={S.btn(isActive)}>
                    {mode.label}
                    <span style={{ float: 'right', color: '#475569' }}>{count}</span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* ========== Tab 3: 动画场景 ========== */}
        {leftPanelTab === 'animation' && (
          <div>
            {/* 场景列表 */}
            <div style={S.section}>
              <h3 style={S.sectionTitle}>🎬 预设场景</h3>
              {Object.entries(ANIMATION_SCENARIOS).map(([key, scenario]) => (
                <div key={key} style={{ marginBottom: 6 }}>
                  <button
                    onClick={() => startScenario(key)}
                    style={{
                      ...S.btn(activeScenario === key),
                      display: 'flex', alignItems: 'center', gap: 8,
                    }}
                  >
                    <span style={{ fontSize: 14 }}>{scenario.icon}</span>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 'bold' }}>{scenario.label}</div>
                      <div style={{ fontSize: 9, color: '#475569' }}>{scenario.description}</div>
                    </div>
                    <span style={{ fontSize: 9, color: '#475569' }}>{scenario.duration}s</span>
                  </button>
                </div>
              ))}
            </div>

            {/* 播放控制 */}
            <div style={S.section}>
              <h3 style={S.sectionTitle}>播放控制</h3>
              <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
                <button onClick={() => { if (activeScenario) { setAnimProgress(0); startTimeRef.current = Date.now(); setPlaying(true); } else { setAnimProgress(0); setPlaying(true); } }}
                  style={{ flex: 1, padding: '6px', background: '#1e293b', border: '1px solid #334155', borderRadius: 4, color: '#e2e8f0', cursor: 'pointer', fontSize: 12 }}>
                  ▶ 播放
                </button>
                <button onClick={() => setPlaying(false)}
                  style={{ flex: 1, padding: '6px', background: '#1e293b', border: '1px solid #334155', borderRadius: 4, color: '#e2e8f0', cursor: 'pointer', fontSize: 12 }}>
                  ⏸ 暂停
                </button>
                <button onClick={stopScenario}
                  style={{ flex: 1, padding: '6px', background: '#1e293b', border: '1px solid #334155', borderRadius: 4, color: '#e2e8f0', cursor: 'pointer', fontSize: 12 }}>
                  ⏹ 停止
                </button>
              </div>

              {/* 进度条 */}
              <div style={S.progressBg}>
                <div style={S.progressBar(animProgress * 100, activeScenario ? ANIMATION_SCENARIOS[activeScenario]?.color : '#3b82f6')} />
              </div>
              <div style={{ fontSize: 10, color: '#64748b', marginTop: 4, display: 'flex', justifyContent: 'space-between' }}>
                <span>{(animProgress * 100).toFixed(0)}%</span>
                {activeScenario && <span>{ANIMATION_SCENARIOS[activeScenario].duration}s</span>}
              </div>
            </div>

            {/* 当前动画阶段 */}
            {activeScenario && (() => {
              const phase = getCurrentPhase();
              if (!phase) return null;
              const scenario = ANIMATION_SCENARIOS[activeScenario];
              return (
                <div style={{ ...S.section, padding: 10, background: '#0f172a', borderRadius: 6, border: `1px solid ${scenario.color || '#334155'}` }}>
                  <div style={{ fontSize: 12, color: scenario.color, fontWeight: 'bold', marginBottom: 6 }}>
                    {scenario.icon} {scenario.label}
                  </div>
                  <div style={{ fontSize: 11, color: '#e2e8f0', marginBottom: 4 }}>
                    当前阶段: <span style={{ color: '#60a5fa' }}>{phase.label}</span>
                  </div>
                  <div style={{ fontSize: 10, color: '#64748b' }}>
                    层范围: L{phase.layerRange[0]} → L{phase.layerRange[1]}
                  </div>
                  {/* 阶段进度指示 */}
                  <div style={{ display: 'flex', gap: 2, marginTop: 8 }}>
                    {scenario.phases.map((p, i) => {
                      const isCurrent = p === phase;
                      const isPast = animProgress >= p.end;
                      return (
                        <div key={i} style={{
                          flex: 1, height: 3, borderRadius: 1,
                          background: isCurrent ? scenario.color : isPast ? '#3b82f6' : '#1e293b',
                        }} />
                      );
                    })}
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 2 }}>
                    {scenario.phases.map((p, i) => (
                      <span key={i} style={{ fontSize: 7, color: p === phase ? '#e2e8f0' : '#475569', flex: 1, textAlign: 'center' }}>
                        {p.label}
                      </span>
                    ))}
                  </div>
                </div>
              );
            })()}
          </div>
        )}

        {/* ---- 数据摘要 ---- */}
        {activeData && (
          <div style={{ ...S.section, padding: 8, background: '#0f172a', borderRadius: 6, border: '1px solid #1e293b' }}>
            <h3 style={{ ...S.sectionTitle, marginBottom: 6 }}>数据摘要</h3>
            <div style={{ fontSize: 10, lineHeight: 1.6 }}>
              <div>路线: <span style={{ color: '#c4b5fd' }}>{activeData.source_context?.route_label || activeFileMeta?.route_label || '本地文件'}</span></div>
              <div>数据源: <span style={{ color: '#93c5fd' }}>{activeData.source_context?.source_label || activeFileMeta?.source_label || '-'}</span></div>
              <div>Schema: <span style={{ color: '#60a5fa' }}>{isAtlasGraph ? schemaVersion : `v${schemaVersion}`}</span></div>
              {activeData.source_schema_version && activeData.source_schema_version !== schemaVersion && (
                <div>原始模式: <span style={{ color: '#94a3b8' }}>{activeData.source_schema_version}</span></div>
              )}
              <div>Model: <span style={{ color: '#4ecdc4' }}>{activeData.model || activeData.model_info?.model || 'mixed'}</span></div>
              {isAtlasGraph ? (
                <>
                  <div>Phase: {activeData.model_info?.phase || activeData.phase || '-'}</div>
                  <div>图谱节点: <span style={{ color: '#ffe66d' }}>{atlasNodeCount}</span></div>
                  <div>图谱边: <span style={{ color: '#f97316' }}>{atlasEdgeCount}</span></div>
                </>
              ) : (
                <>
                  <div>Layers: {activeData.model_info?.n_layers} | d_model: {activeData.model_info?.d_model}</div>
                  <div>可视化对象: <span style={{ color: '#ffe66d' }}>{visualizations.length}</span></div>
                </>
              )}
            </div>
          </div>
        )}

        {/* ---- 颜色图例 ---- */}
        <div style={{ marginTop: 12 }}>
          <h3 style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>图例</h3>
          <div style={{ display: 'flex', gap: 3, marginBottom: 6 }}>
            {[1, 0.75, 0.5, 0.25, 0].map(v => (
              <div key={v} style={{ textAlign: 'center' }}>
                <div style={{ width: 20, height: 8, background: deltaCosToColor(v), borderRadius: 2 }} />
                <div style={{ fontSize: 8, color: '#475569' }}>{v.toFixed(1)}</div>
              </div>
            ))}
          </div>
          <div style={{ fontSize: 9, color: '#475569' }}>
            <span style={{ color: SUBSPACE_COLORS.w_u }}>■</span> W_U &nbsp;
            <span style={{ color: SUBSPACE_COLORS.w_u_perp }}>■</span> W_U⊥ &nbsp;
            <span style={{ color: SUBSPACE_COLORS.logic }}>■</span> 逻辑 &nbsp;
            <span style={{ color: SUBSPACE_COLORS.dark_matter }}>■</span> 暗物质
          </div>
        </div>
      </div>

      {/* ==================== 中央3D画布 ==================== */}
      <div style={{ flex: 1, position: 'relative' }}>
        {(loading || sourceLoading) && (
          <div style={{ position: 'absolute', top: 20, left: 20, zIndex: 10, background: '#1e293b', padding: '8px 16px', borderRadius: 6, fontSize: 13 }}>
            {sourceLoading ? '正在加载路线清单...' : '正在加载测试数据...'}
          </div>
        )}
        {error && (
          <div style={{ position: 'absolute', top: 20, left: 20, zIndex: 10, background: '#7f1d1d', padding: '8px 16px', borderRadius: 6, fontSize: 13 }}>
            错误: {error}
          </div>
        )}
        {/* 顶部状态栏 */}
        {(activeDimension || activeScenario) && (
          <div style={{ position: 'absolute', top: 12, left: '50%', transform: 'translateX(-50%)', zIndex: 10, display: 'flex', gap: 8 }}>
            {activeDimension && (
              <div style={{
                padding: '4px 12px', borderRadius: 6,
                background: `${DIMENSION_VIEWS[activeDimension].color}22`,
                border: `1px solid ${DIMENSION_VIEWS[activeDimension].color}`,
                color: DIMENSION_VIEWS[activeDimension].color,
                fontSize: 11,
              }}>
                {DIMENSION_VIEWS[activeDimension].icon} {DIMENSION_VIEWS[activeDimension].label}
                {activeSubView && ` → ${DIMENSION_VIEWS[activeDimension].subViews[activeSubView]?.label}`}
              </div>
            )}
            {activeScenario && (
              <div style={{
                padding: '4px 12px', borderRadius: 6,
                background: '#1e40af', border: '1px solid #3b82f6',
                color: '#bfdbfe', fontSize: 11,
              }}>
                ▶ {ANIMATION_SCENARIOS[activeScenario].label} — {(animProgress * 100).toFixed(0)}%
              </div>
            )}
          </div>
        )}

        <Canvas
          camera={{ position: [25, 30, 25], fov: 50 }}
          gl={{ antialias: true, alpha: true }}
          style={{ background: '#0f172a' }}
        >
          <PerspectiveCamera makeDefault position={[25, 30, 25]} fov={50} />
          <OrbitControls enableDamping dampingFactor={0.1} minDistance={5} />

          {/* 灯光 */}
          <ambientLight intensity={0.5} />
          <directionalLight position={[10, 20, 10]} intensity={0.8} />
          <pointLight position={[-10, -10, -10]} intensity={0.3} />

          {/* 背景 */}
          <Stars radius={100} depth={50} count={1500} factor={4} fade speed={0.5} />

          {/* 场景辅助 */}
          <SceneHelpers nLayers={nLayers} />

          {/* ===== 机制图谱测试结果 ===== */}
          {isAtlasGraph && (
            <AtlasGraphRenderer
              graph={atlasGraph}
              onHoverNode={setHoveredInfo}
            />
          )}

          {/* ===== v1.0 渲染器 ===== */}
          {layerStacks.map(ls => (
            <LayerStackRenderer key={ls.id} layerStack={ls} selectedLayers={selectedLayers} />
          ))}
          {trajectories.map(traj => (
            <TrajectoryRenderer
              key={traj.id}
              trajectory={traj}
              animated={animProgress < 1}
              animationProgress={animProgress}
              onHoverToken={setHoveredInfo}
            />
          ))}
          {pointClouds.map(pc => (
            <PointCloudRenderer key={pc.id} pointCloud={pc} onHoverToken={setHoveredInfo} />
          ))}
          {heatmaps.map(hm => (
            <Heatmap3DRenderer key={hm.id} heatmap={hm} />
          ))}
          {flows.map(fl => (
            <FlowRenderer key={fl.id} flow={fl} animated={animProgress < 1} animationProgress={animProgress} />
          ))}

          {/* ===== v2.0 渲染器 ===== */}
          {subspaceDecomps.map(sd => (
            <SubspaceRenderer
              key={sd.id}
              subspaceDecomp={sd}
              animated={animProgress < 1}
              animationProgress={animProgress}
              onHoverToken={setHoveredInfo}
            />
          ))}
          {forceLines.map(fl => (
            <ForceLineRenderer
              key={fl.id}
              forceLine={fl}
              animated={animProgress < 1}
              animationProgress={animProgress}
              onHoverToken={setHoveredInfo}
            />
          ))}
          {grammarMatrices.map(gm => (
            <GrammarRoleMatrixRenderer
              key={gm.id}
              grammarMatrix={gm}
              onHoverToken={setHoveredInfo}
            />
          ))}
          {causalChains.map(cc => (
            <CausalChainRenderer
              key={cc.id}
              causalChain={cc}
              animated={animProgress < 1}
              animationProgress={animProgress}
              onHoverToken={setHoveredInfo}
            />
          ))}
          {darkMatterFlows.map(dmf => (
            <DarkMatterFlowRenderer
              key={dmf.id}
              darkMatterFlow={dmf}
              animated={animProgress < 1}
              animationProgress={animProgress}
              onHoverToken={setHoveredInfo}
            />
          ))}
        </Canvas>
      </div>

      {/* ==================== 右侧面板 ==================== */}
      <div style={{ width: 320, borderLeft: '1px solid #1e293b', overflowY: 'auto', flexShrink: 0, display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: '8px 4px', background: 'rgba(96, 165, 250, 0.1)', borderBottom: '2px solid #60a5fa', color: '#60a5fa', fontSize: 11, fontWeight: 'bold', textAlign: 'center' }}>
          📋 详情
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: 16 }}>
          <h3 style={{ fontSize: 13, color: '#94a3b8', marginBottom: 12 }}>详情</h3>
            {hoveredInfo ? (
              <div style={{ fontSize: 12, lineHeight: 1.8 }}>
                {hoveredInfo.token && <div style={{ color: '#60a5fa', fontWeight: 'bold', fontSize: 14 }}>{hoveredInfo.token}</div>}
                {hoveredInfo.source && <div><span style={{ color: '#94a3b8' }}>from:</span> {hoveredInfo.source}</div>}
                {hoveredInfo.layer !== undefined && <div><span style={{ color: '#94a3b8' }}>Layer:</span> {hoveredInfo.layer} <span style={{ color: layerToFuncColor(hoveredInfo.layer, nLayers), fontSize: 10 }}>({layerToFuncLabel(hoveredInfo.layer, nLayers)})</span></div>}
                {hoveredInfo.delta_cos !== undefined && (
                  <div><span style={{ color: '#94a3b8' }}>δ_cos:</span> <span style={{ color: deltaCosToColor(hoveredInfo.delta_cos) }}>{hoveredInfo.delta_cos.toFixed(4)}</span></div>
                )}
                {hoveredInfo.cos_with_target !== undefined && <div><span style={{ color: '#94a3b8' }}>cos(target):</span> {hoveredInfo.cos_with_target.toFixed(4)}</div>}
                {hoveredInfo.cos_with_wu !== undefined && (
                  <div><span style={{ color: '#94a3b8' }}>cos(W_U):</span> <span style={{ color: cosWuToColor(hoveredInfo.cos_with_wu) }}>{hoveredInfo.cos_with_wu.toFixed(4)}</span></div>
                )}
                {hoveredInfo.norm !== undefined && <div><span style={{ color: '#94a3b8' }}>norm:</span> {hoveredInfo.norm.toFixed(1)}</div>}
                {hoveredInfo.category && (
                  <div><span style={{ color: '#94a3b8' }}>category:</span> <span style={{ color: CATEGORY_COLORS[hoveredInfo.category] || '#888' }}>{hoveredInfo.category}</span></div>
                )}
                {hoveredInfo.type && <div><span style={{ color: '#94a3b8' }}>type:</span> {hoveredInfo.type}</div>}
                {hoveredInfo.model && <div><span style={{ color: '#94a3b8' }}>model:</span> {hoveredInfo.model}</div>}
                {hoveredInfo.head !== undefined && <div><span style={{ color: '#94a3b8' }}>head:</span> H{hoveredInfo.head}</div>}
                {hoveredInfo.channel !== undefined && <div><span style={{ color: '#94a3b8' }}>channel:</span> C{hoveredInfo.channel}</div>}
                {hoveredInfo.role && <div><span style={{ color: '#94a3b8' }}>role:</span> {hoveredInfo.role}</div>}
                {hoveredInfo.evidence_level && <div><span style={{ color: '#94a3b8' }}>evidence:</span> {hoveredInfo.evidence_level}</div>}
                {hoveredInfo.mean_logprob_delta !== undefined && (
                  <div><span style={{ color: '#94a3b8' }}>mean Δlogp:</span> {Number(hoveredInfo.mean_logprob_delta).toFixed(4)}</div>
                )}
                {hoveredInfo.changed_rate_vs_baseline !== undefined && (
                  <div><span style={{ color: '#94a3b8' }}>changed:</span> {Number(hoveredInfo.changed_rate_vs_baseline).toFixed(3)}</div>
                )}
                {hoveredInfo.hit_drop_rate_vs_baseline !== undefined && (
                  <div><span style={{ color: '#94a3b8' }}>hit drop:</span> {Number(hoveredInfo.hit_drop_rate_vs_baseline).toFixed(3)}</div>
                )}
                {hoveredInfo.subspace && (
                  <div><span style={{ color: '#94a3b8' }}>subspace:</span> <span style={{ color: SUBSPACE_COLORS[hoveredInfo.subspace] || '#888' }}>{hoveredInfo.subspace === 'w_u' ? 'W_U' : 'W_U⊥'}</span></div>
                )}
                {hoveredInfo.isCorrection && <div style={{ color: '#fbbf24', fontWeight: 'bold' }}>⚡ 纠正层</div>}
                {hoveredInfo.growth_rate !== undefined && <div><span style={{ color: '#94a3b8' }}>growth_rate:</span> {hoveredInfo.growth_rate.toFixed(3)}</div>}
                {hoveredInfo.role_pair && <div><span style={{ color: '#94a3b8' }}>角色对:</span> <span style={{ color: '#ffe66d' }}>{hoveredInfo.role_pair}</span></div>}
                {hoveredInfo.cosine !== undefined && <div><span style={{ color: '#94a3b8' }}>cosine:</span> {hoveredInfo.cosine.toFixed(4)}</div>}
                {hoveredInfo.kl_divergence !== undefined && <div><span style={{ color: '#94a3b8' }}>KL:</span> {hoveredInfo.kl_divergence.toFixed(2)}</div>}
                {hoveredInfo.classification_flip !== undefined && <div><span style={{ color: '#94a3b8' }}>flip:</span> {(hoveredInfo.classification_flip * 100).toFixed(1)}%</div>}
                {hoveredInfo.w_u_signal !== undefined && <div><span style={{ color: SUBSPACE_COLORS.w_u }}>W_U:</span> {(hoveredInfo.w_u_signal * 100).toFixed(0)}%</div>}
                {hoveredInfo.w_u_perp_signal !== undefined && <div><span style={{ color: SUBSPACE_COLORS.w_u_perp }}>W_U⊥:</span> {(hoveredInfo.w_u_perp_signal * 100).toFixed(0)}%</div>}
              </div>
            ) : (
              <div style={{ fontSize: 12, color: '#64748b' }}>悬停3D对象查看详情</div>
            )}

            {visualizations.length > 0 && (
              <div style={{ marginTop: 20 }}>
                <h3 style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>可视化对象 ({visualizations.length})</h3>
                {visualizations.map((v, i) => (
                  <div key={i} style={{ padding: '4px 8px', fontSize: 11, color: '#94a3b8', borderBottom: '1px solid #1e293b' }}>
                    <span style={{ color: '#60a5fa' }}>{v.type}</span> {v.label || v.id}
                  </div>
                ))}
              </div>
            )}

            {visualizations.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <h3 style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>类型统计</h3>
                <div style={{ fontSize: 11, lineHeight: 1.8 }}>
                  {Object.entries(byType).filter((entry) => entry[1].length > 0).map(([type, arr]) => (
                    <div key={type}><span style={{ color: '#60a5fa' }}>{type}</span>: {arr.length}</div>
                  ))}
                </div>
              </div>
            )}
          </div>
      </div>
    </div>
  );
}
