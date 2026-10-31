/**
 * 数据展示模板组件
 * 为每种分析模式提供统一的数据展示界面
 */
import React from 'react';
import { ANALYSIS_DATA_TEMPLATES, COLORS } from '../../config/panels';

// 格式化工具函数
const formatValue = (value, format) => {
  if (value === null || value === undefined) return 'N/A';
  switch (format) {
    case 'percent':
      return `${(value * 100).toFixed(1)}%`;
    case 'decimal':
      return typeof value === 'number' ? value.toFixed(4) : value;
    case 'number':
    default:
      return typeof value === 'number' ? value.toLocaleString() : value;
  }
};

// 指标卡片组件
export function MetricCard({ label, value, format, color = COLORS.primary }) {
  return (
    <div style={{
      background: 'rgba(255,255,255,0.05)',
      borderRadius: '6px',
      padding: '8px 12px',
      borderLeft: `3px solid ${color}`,
      flex: 1,
      minWidth: '80px'
    }}>
      <div style={{ fontSize: '10px', color: COLORS.textMuted, marginBottom: '2px' }}>
        {label}
      </div>
      <div style={{ fontSize: '14px', fontWeight: 'bold', color: COLORS.textPrimary }}>
        {formatValue(value, format)}
      </div>
    </div>
  );
}

// 指标行组件
export function MetricsRow({ metrics, data, color }) {
  return (
    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '12px' }}>
      {metrics.map((metric, idx) => (
        <MetricCard
          key={idx}
          label={metric.label}
          value={data?.[metric.key]}
          format={metric.format}
          color={color}
        />
      ))}
    </div>
  );
}

// 层列表组件
export function LayerList({ data, selectedLayer, onLayerSelect, hoveredInfo }) {
  if (!data || !Array.isArray(data)) return null;
  
  return (
    <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
      {data.map((layerData, layerIdx) => {
        const avgConfidence = layerData.reduce((sum, pos) => sum + pos.prob, 0) / layerData.length;
        const isSelected = selectedLayer === layerIdx;
        const isHovered = hoveredInfo?.layer === layerIdx;
        
        return (
          <div
            key={layerIdx}
            onClick={() => onLayerSelect?.(layerIdx)}
            style={{
              padding: '8px',
              marginBottom: '4px',
              background: isSelected ? 'rgba(0, 210, 255, 0.2)' : 
                         isHovered ? 'rgba(0, 210, 255, 0.1)' : 
                         'rgba(255,255,255,0.03)',
              border: isSelected ? '1px solid rgba(0, 210, 255, 0.8)' :
                     isHovered ? '1px solid rgba(0, 210, 255, 0.5)' :
                     '1px solid rgba(255,255,255,0.05)',
              borderRadius: '4px',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontWeight: 'bold', color: COLORS.textPrimary, fontSize: '12px' }}>
                Layer {layerIdx}
              </span>
              <span style={{ 
                color: avgConfidence > 0.5 ? COLORS.success : COLORS.warning,
                fontSize: '11px'
              }}>
                {(avgConfidence * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// 特征表格组件
export function FeatureTable({ features, maxRows = 10 }) {
  if (!features || !Array.isArray(features)) return null;
  
  return (
    <div style={{ 
      maxHeight: '180px', 
      overflowY: 'auto',
      fontSize: '11px'
    }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ color: COLORS.textMuted, borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
            <th style={{ textAlign: 'left', padding: '4px' }}>#</th>
            <th style={{ textAlign: 'left', padding: '4px' }}>特征</th>
            <th style={{ textAlign: 'right', padding: '4px' }}>激活</th>
          </tr>
        </thead>
        <tbody>
          {features.slice(0, maxRows).map((feat, idx) => (
            <tr key={idx} style={{ color: COLORS.textSecondary }}>
              <td style={{ padding: '4px' }}>{idx + 1}</td>
              <td style={{ padding: '4px' }}>{feat.label || `F${feat.idx || idx}`}</td>
              <td style={{ textAlign: 'right', padding: '4px' }}>
                <span style={{ 
                  color: feat.activation > 0.5 ? COLORS.success : COLORS.textMuted 
                }}>
                  {feat.activation?.toFixed(3) || 'N/A'}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// 图结构摘要组件
export function GraphSummary({ graph }) {
  if (!graph) return null;
  
  const nodes = graph.nodes?.length || 0;
  const edges = graph.links?.length || graph.edges?.length || 0;
  const density = nodes > 1 ? (2 * edges / (nodes * (nodes - 1))) : 0;
  
  return (
    <div style={{ fontSize: '11px', color: COLORS.textSecondary }}>
      <div style={{ marginBottom: '6px' }}>
        <strong style={{ color: COLORS.textPrimary }}>节点:</strong> {nodes} 个
      </div>
      <div style={{ marginBottom: '6px' }}>
        <strong style={{ color: COLORS.textPrimary }}>边:</strong> {edges} 条
      </div>
      <div>
        <strong style={{ color: COLORS.textPrimary }}>密度:</strong>{' '}
        <span style={{ color: density > 0.3 ? COLORS.warning : COLORS.success }}>
          {(density * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

// 通用数据展示模板
export function AnalysisDataDisplay({ 
  mode, 
  data, 
  analysisResult, 
  selectedLayer, 
  onLayerSelect,
  hoveredInfo 
}) {
  const template = ANALYSIS_DATA_TEMPLATES[mode];
  
  if (!template) {
    return (
      <div style={{ color: COLORS.textMuted, fontSize: '12px', fontStyle: 'italic' }}>
        选择一个分析模式查看数据
      </div>
    );
  }
  
  // 合并数据源
  const sourceData = {
    ...data,
    ...analysisResult,
    // 计算派生指标
    nodes: analysisResult?.nodes?.length || analysisResult?.graph?.nodes?.length || 0,
    edges: analysisResult?.graph?.links?.length || analysisResult?.graph?.edges?.length || 0,
    density: (() => {
      const n = analysisResult?.nodes?.length || analysisResult?.graph?.nodes?.length || 0;
      const e = analysisResult?.graph?.links?.length || analysisResult?.graph?.edges?.length || 0;
      return n > 1 ? (2 * e / (n * (n - 1))) : 0;
    })(),
    avg_confidence: data?.logit_lens ? 
      data.logit_lens.reduce((sum, layer) => {
        const layerAvg = layer.reduce((s, p) => s + p.prob, 0) / layer.length;
        return sum + layerAvg;
      }, 0) / data.logit_lens.length : null,
  };
  
  return (
    <div>
      {/* 标题 */}
      <div style={{
        paddingBottom: '8px',
        borderBottom: `1px solid ${template.color}40`,
        marginBottom: '12px',
        color: template.color,
        fontWeight: 'bold',
        fontSize: '13px',
        display: 'flex',
        alignItems: 'center',
        gap: '6px'
      }}>
        <span>{template.title}</span>
      </div>
      
      {/* 指标行 */}
      {template.metrics.length > 0 && (
        <MetricsRow 
          metrics={template.metrics} 
          data={sourceData} 
          color={template.color} 
        />
      )}
      
      {/* 各类数据区块 */}
      {template.sections.map((section, idx) => {
        switch (section.type) {
          case 'layer_list':
            return (
              <div key={idx}>
                <div style={{ 
                  fontSize: '11px', 
                  color: COLORS.textMuted, 
                  marginBottom: '6px' 
                }}>
                  {section.title}
                </div>
                <LayerList 
                  data={sourceData[section.source]} 
                  selectedLayer={selectedLayer}
                  onLayerSelect={onLayerSelect}
                  hoveredInfo={hoveredInfo}
                />
              </div>
            );
          case 'feature_table':
            return (
              <div key={idx}>
                <div style={{ 
                  fontSize: '11px', 
                  color: COLORS.textMuted, 
                  marginBottom: '6px' 
                }}>
                  {section.title}
                </div>
                <FeatureTable features={sourceData[section.source]} />
              </div>
            );
          case 'graph_summary':
            return (
              <div key={idx}>
                <div style={{ 
                  fontSize: '11px', 
                  color: COLORS.textMuted, 
                  marginBottom: '6px' 
                }}>
                  {section.title}
                </div>
                <GraphSummary graph={sourceData[section.source]} />
              </div>
            );
          default:
            return null;
        }
      })}
      
      {/* 无数据提示 */}
      {!data && !analysisResult && (
        <div style={{ 
          color: COLORS.textMuted, 
          fontSize: '11px', 
          fontStyle: 'italic',
          textAlign: 'center',
          padding: '20px'
        }}>
          暂无数据，运行分析后查看结果
        </div>
      )}
    </div>
  );
}

export default AnalysisDataDisplay;
