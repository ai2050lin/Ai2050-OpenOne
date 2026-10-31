/**
 * 数据对比视图组件
 * 支持不同分析结果的对比展示
 */
import React, { useState, useCallback } from 'react';
import { COLORS } from '../../config/panels';
import { GitCompare, X, Plus, ChevronLeft, ChevronRight } from 'lucide-react';

// 对比槽位
const MAX_SLOTS = 3;

// 对比卡片组件
function ComparisonSlot({ data, label, onRemove, onClick, isActive }) {
  return (
    <div
      onClick={onClick}
      style={{
        flex: 1,
        minWidth: '120px',
        padding: '10px',
        background: isActive ? 'rgba(0, 210, 255, 0.1)' : 'rgba(255,255,255,0.03)',
        border: isActive ? '1px solid rgba(0, 210, 255, 0.5)' : '1px solid rgba(255,255,255,0.1)',
        borderRadius: '8px',
        cursor: 'pointer',
        transition: 'all 0.2s',
        position: 'relative',
      }}
    >
      {data && onRemove && (
        <button
          onClick={(e) => { e.stopPropagation(); onRemove(); }}
          style={{
            position: 'absolute',
            top: '4px',
            right: '4px',
            background: 'rgba(0,0,0,0.5)',
            border: 'none',
            borderRadius: '50%',
            width: '16px',
            height: '16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
          }}
        >
          <X size={10} color={COLORS.textMuted} />
        </button>
      )}
      
      {data ? (
        <div>
          <div style={{ fontSize: '10px', color: COLORS.textMuted, marginBottom: '4px' }}>
            {label}
          </div>
          <div style={{ fontSize: '12px', color: COLORS.textPrimary, fontWeight: 'bold' }}>
            {data.label || data.mode || '分析结果'}
          </div>
          {data.metrics && (
            <div style={{ marginTop: '6px', display: 'flex', flexDirection: 'column', gap: '2px' }}>
              {Object.entries(data.metrics).slice(0, 3).map(([key, value]) => (
                <div key={key} style={{ fontSize: '10px', color: COLORS.textSecondary }}>
                  {key}: <span style={{ color: COLORS.primary }}>{value}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div style={{ 
          textAlign: 'center', 
          color: COLORS.textMuted,
          padding: '20px 0'
        }}>
          <Plus size={16} style={{ opacity: 0.5 }} />
          <div style={{ fontSize: '10px', marginTop: '4px' }}>点击添加</div>
        </div>
      )}
    </div>
  );
}

// 指标对比行
function MetricComparisonRow({ label, values, colors }) {
  const maxVal = Math.max(...values.filter(v => typeof v === 'number'), 1);
  
  return (
    <div style={{ marginBottom: '12px' }}>
      <div style={{ 
        fontSize: '10px', 
        color: COLORS.textMuted, 
        marginBottom: '4px' 
      }}>
        {label}
      </div>
      <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
        {values.map((val, idx) => (
          <div key={idx} style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '4px' }}>
            <div style={{
              flex: 1,
              height: '8px',
              background: 'rgba(255,255,255,0.1)',
              borderRadius: '4px',
              overflow: 'hidden',
            }}>
              {typeof val === 'number' && (
                <div style={{
                  width: `${(val / maxVal) * 100}%`,
                  height: '100%',
                  background: colors[idx] || COLORS.primary,
                  borderRadius: '4px',
                  transition: 'width 0.3s',
                }} />
              )}
            </div>
            <span style={{ 
              fontSize: '10px', 
              color: COLORS.textSecondary,
              minWidth: '40px',
              textAlign: 'right'
            }}>
              {typeof val === 'number' ? val.toFixed(3) : 'N/A'}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// 差异指示器
function DifferenceIndicator({ base, compare }) {
  if (typeof base !== 'number' || typeof compare !== 'number') return null;
  
  const diff = compare - base;
  const percent = base !== 0 ? ((diff / base) * 100) : 0;
  const isPositive = diff > 0;
  
  return (
    <span style={{
      fontSize: '9px',
      color: isPositive ? COLORS.success : COLORS.danger,
      marginLeft: '4px'
    }}>
      {isPositive ? '↑' : '↓'} {Math.abs(percent).toFixed(1)}%
    </span>
  );
}

// 主对比视图组件
export function DataComparisonView({ 
  currentData, 
  analysisResult,
  mode,
  onCaptureSnapshot 
}) {
  const [slots, setSlots] = useState([null, null, null]);
  const [activeSlot, setActiveSlot] = useState(0);
  const [showComparison, setShowComparison] = useState(false);
  
  // 捕获当前快照
  const captureSnapshot = useCallback((slotIndex) => {
    if (!analysisResult && !currentData) return;
    
    const snapshot = {
      id: Date.now(),
      label: mode || '分析结果',
      mode,
      timestamp: new Date().toISOString(),
      data: currentData,
      result: analysisResult,
      metrics: extractMetrics(analysisResult, currentData, mode),
    };
    
    setSlots(prev => {
      const newSlots = [...prev];
      newSlots[slotIndex] = snapshot;
      return newSlots;
    });
    
    onCaptureSnapshot?.(snapshot);
  }, [analysisResult, currentData, mode, onCaptureSnapshot]);
  
  // 移除快照
  const removeSlot = useCallback((slotIndex) => {
    setSlots(prev => {
      const newSlots = [...prev];
      newSlots[slotIndex] = null;
      return newSlots;
    });
  }, []);
  
  // 提取指标
  function extractMetrics(result, data, currentMode) {
    const metrics = {};
    
    if (currentMode === 'circuit' && result) {
      metrics['节点数'] = result.nodes?.length || 0;
      metrics['边数'] = result.graph?.links?.length || 0;
    }
    if (currentMode === 'features' && result) {
      metrics['特征数'] = result.n_features || 0;
      metrics['稀疏度'] = result.sparsity?.toFixed(4) || 'N/A';
    }
    if (currentMode === 'logit_lens' && data?.logit_lens) {
      const avgConf = data.logit_lens.reduce((sum, layer) => {
        const layerAvg = layer.reduce((s, p) => s + p.prob, 0) / layer.length;
        return sum + layerAvg;
      }, 0) / data.logit_lens.length;
      metrics['平均置信度'] = avgConf.toFixed(3);
    }
    
    return metrics;
  }
  
  // 比较两个槽位的指标
  const getComparisonMetrics = () => {
    const validSlots = slots.filter(s => s !== null);
    if (validSlots.length < 2) return null;
    
    const allMetrics = new Set();
    validSlots.forEach(slot => {
      Object.keys(slot.metrics || {}).forEach(key => allMetrics.add(key));
    });
    
    return Array.from(allMetrics).map(metric => ({
      label: metric,
      values: slots.map(slot => {
        if (!slot?.metrics?.[metric]) return null;
        const val = slot.metrics[metric];
        return typeof val === 'string' ? parseFloat(val) : val;
      }),
    }));
  };
  
  const comparisonMetrics = getComparisonMetrics();
  const slotColors = [COLORS.primary, COLORS.success, COLORS.warning];
  
  return (
    <div>
      {/* 标题栏 */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '12px',
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          color: COLORS.textPrimary,
          fontSize: '12px',
          fontWeight: 'bold'
        }}>
          <GitCompare size={14} color={COLORS.primary} />
          数据对比
        </div>
        <button
          onClick={() => setShowComparison(!showComparison)}
          style={{
            background: 'transparent',
            border: 'none',
            color: COLORS.textMuted,
            cursor: 'pointer',
            fontSize: '10px',
            padding: '2px 6px',
          }}
        >
          {showComparison ? '收起' : '展开'}
        </button>
      </div>
      
      {/* 对比槽位 */}
      <div style={{ 
        display: 'flex', 
        gap: '8px', 
        marginBottom: showComparison ? '12px' : 0 
      }}>
        {slots.map((slot, idx) => (
          <ComparisonSlot
            key={idx}
            data={slot}
            label={`对比 ${idx + 1}`}
            onRemove={() => removeSlot(idx)}
            onClick={() => {
              if (!slot) {
                captureSnapshot(idx);
              } else {
                setActiveSlot(idx);
              }
            }}
            isActive={activeSlot === idx}
          />
        ))}
      </div>
      
      {/* 详细对比 */}
      {showComparison && comparisonMetrics && (
        <div style={{
          padding: '12px',
          background: 'rgba(0,0,0,0.2)',
          borderRadius: '8px',
          marginTop: '8px',
        }}>
          <div style={{ 
            fontSize: '11px', 
            color: COLORS.textMuted,
            marginBottom: '12px',
            display: 'flex',
            gap: '12px'
          }}>
            {slots.filter(s => s).map((slot, idx) => (
              <span key={idx}>
                <span style={{ 
                  display: 'inline-block',
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: slotColors[idx],
                  marginRight: '4px'
                }} />
                {slot.label}
              </span>
            ))}
          </div>
          
          {comparisonMetrics.map((metric, idx) => (
            <MetricComparisonRow
              key={idx}
              label={metric.label}
              values={metric.values}
              colors={slotColors}
            />
          ))}
        </div>
      )}
      
      {/* 提示 */}
      {slots.filter(s => s === null).length === MAX_SLOTS && (
        <div style={{ 
          fontSize: '10px', 
          color: COLORS.textMuted, 
          textAlign: 'center',
          marginTop: '8px',
          fontStyle: 'italic'
        }}>
          运行分析后点击槽位捕获快照进行对比
        </div>
      )}
    </div>
  );
}

export default DataComparisonView;
