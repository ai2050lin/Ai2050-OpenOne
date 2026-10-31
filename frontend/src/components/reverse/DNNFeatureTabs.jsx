/**
 * DNN特征标签页选择器
 * 5大DNN特征维度(权重/激活/因果/信息/动力学) × radio选择
 */
import { Activity, BarChart2, Layers, TrendingUp, Zap } from 'lucide-react';
import { DNN_FEATURES } from '../../config/dnnFeatures';

const ICON_MAP = { Layers, Activity, Zap, BarChart2, TrendingUp };

export default function DNNFeatureTabs({ selectedCategory, selectedFeature, onCategoryChange, onFeatureChange }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
      <span style={{ fontSize: '12px', fontWeight: 700, color: '#dfe8ff' }}>DNN特征选择</span>

      {/* Category tabs */}
      <div style={{ display: 'flex', gap: '3px', flexWrap: 'wrap' }}>
        {Object.values(DNN_FEATURES).map((cat) => {
          const IconComp = ICON_MAP[cat.icon];
          const isActive = selectedCategory === cat.id;
          return (
            <button
              key={cat.id}
              onClick={() => {
                onCategoryChange(cat.id);
                // Auto-select first sub-feature of the category
                if (cat.subFeatures.length > 0) {
                  onFeatureChange(cat.subFeatures[0].id);
                }
              }}
              style={{
                display: 'flex', alignItems: 'center', gap: '3px',
                padding: '4px 8px',
                borderRadius: '4px',
                border: `1px solid ${isActive ? cat.color + '60' : 'rgba(255,255,255,0.1)'}`,
                background: isActive ? cat.color + '20' : 'transparent',
                color: isActive ? cat.color : '#888',
                cursor: 'pointer',
                fontSize: '10px',
                fontWeight: isActive ? 700 : 500,
                transition: 'all 0.15s',
              }}
            >
              {IconComp && <IconComp size={11} />}
              {cat.label}
            </button>
          );
        })}
      </div>

      {/* Sub-feature radio list */}
      {selectedCategory && DNN_FEATURES[selectedCategory] && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gap: '2px',
          padding: '4px 0',
        }}>
          {DNN_FEATURES[selectedCategory].subFeatures.map((sub) => {
            const isSelected = selectedFeature === sub.id;
            const catColor = DNN_FEATURES[selectedCategory].color;
            return (
              <div
                key={sub.id}
                onClick={() => onFeatureChange(sub.id)}
                style={{
                  display: 'flex', alignItems: 'center', gap: '4px',
                  padding: '3px 5px',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  background: isSelected ? catColor + '15' : 'transparent',
                  border: `1px solid ${isSelected ? catColor + '40' : 'transparent'}`,
                  transition: 'all 0.15s',
                  userSelect: 'none',
                }}
              >
                <div style={{
                  width: '8px', height: '8px', borderRadius: '50%',
                  border: `1.5px solid ${isSelected ? catColor : '#555'}`,
                  background: isSelected ? catColor : 'transparent',
                  flexShrink: 0,
                  transition: 'all 0.15s',
                }} />
                <span style={{
                  fontSize: '10px',
                  color: isSelected ? '#eef7ff' : '#888',
                  fontWeight: isSelected ? 600 : 400,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}>
                  {sub.id} {sub.label}
                </span>
              </div>
            );
          })}
        </div>
      )}

      {/* Feature description */}
      {selectedFeature && (() => {
        const cat = DNN_FEATURES[selectedCategory];
        const sub = cat?.subFeatures.find((s) => s.id === selectedFeature);
        if (!sub) return null;
        return (
          <div style={{
            fontSize: '10px', color: '#7f95bb',
            padding: '4px 6px', borderRadius: '4px',
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.06)',
          }}>
            {sub.description}
          </div>
        );
      })()}
    </div>
  );
}
