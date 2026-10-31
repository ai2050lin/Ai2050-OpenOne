/**
 * 语言维度手风琴选择器
 * 5大语言维度（语法/语义/逻辑/语用/形态）× 子维度toggle开关
 */
import { useState } from 'react';
import { Brain, ChevronDown, ChevronRight, GitBranch, Languages, MessageSquare, Type } from 'lucide-react';
import { LANGUAGE_DIMENSIONS } from '../../config/languageDimensions';
import { countSelectedDims } from '../../config/languageDimensions';
import DimensionGroup from './DimensionGroup';

const ICON_MAP = { Type, Brain, GitBranch, MessageSquare, Languages };

export default function LanguageDimensionSelector({ selection, onSelectionChange }) {
  const [expandedGroups, setExpandedGroups] = useState({ syntax: true });

  const toggleGroup = (dimId) => {
    setExpandedGroups((prev) => ({ ...prev, [dimId]: !prev[dimId] }));
  };

  const handleSubDimToggle = (dimId, subId) => {
    const newSelection = {
      ...selection,
      [dimId]: { ...selection[dimId], [subId]: !selection[dimId]?.[subId] },
    };
    onSelectionChange(newSelection);
  };

  const handleGroupToggleAll = (dimId, checked) => {
    const dim = LANGUAGE_DIMENSIONS[dimId];
    if (!dim) return;
    const newGroup = {};
    dim.subDimensions.forEach((sub) => { newGroup[sub.id] = checked; });
    const newSelection = { ...selection, [dimId]: newGroup };
    onSelectionChange(newSelection);
  };

  const selectedCount = countSelectedDims(selection);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
        <span style={{ fontSize: '12px', fontWeight: 700, color: '#dfe8ff' }}>
          语言维度选择
        </span>
        <span style={{ fontSize: '10px', color: '#7f95bb' }}>
          已选 {selectedCount}/35
        </span>
      </div>
      {Object.values(LANGUAGE_DIMENSIONS).map((dim) => {
        const IconComp = ICON_MAP[dim.icon];
        const isExpanded = expandedGroups[dim.id];
        const groupSelected = Object.values(selection[dim.id] || {}).filter(Boolean).length;
        const groupTotal = dim.subDimensions.length;

        return (
          <div key={dim.id} style={{
            borderRadius: '6px',
            border: `1px solid ${groupSelected > 0 ? dim.color + '40' : 'rgba(255,255,255,0.06)'}`,
            background: groupSelected > 0 ? dim.color + '08' : 'rgba(255,255,255,0.02)',
            overflow: 'hidden',
          }}>
            {/* Group header */}
            <div
              onClick={() => toggleGroup(dim.id)}
              style={{
                display: 'flex', alignItems: 'center', gap: '6px',
                padding: '6px 8px', cursor: 'pointer',
                userSelect: 'none',
              }}
            >
              {isExpanded ? <ChevronDown size={12} color="#aaa" /> : <ChevronRight size={12} color="#aaa" />}
              {IconComp && <IconComp size={13} color={dim.color} />}
              <span style={{ fontSize: '11px', fontWeight: 600, color: dim.color, flex: 1 }}>{dim.label}</span>
              <span style={{ fontSize: '10px', color: groupSelected > 0 ? dim.color : '#666' }}>
                {groupSelected}/{groupTotal}
              </span>
              <input
                type="checkbox"
                checked={groupSelected === groupTotal}
                ref={(el) => { if (el) el.indeterminate = groupSelected > 0 && groupSelected < groupTotal; }}
                onChange={(e) => { e.stopPropagation(); handleGroupToggleAll(dim.id, e.target.checked); }}
                onClick={(e) => e.stopPropagation()}
                style={{ cursor: 'pointer', accentColor: dim.color }}
              />
            </div>

            {/* Sub-dimensions */}
            {isExpanded && (
              <DimensionGroup
                dimension={dim}
                selected={selection[dim.id] || {}}
                onToggle={(subId) => handleSubDimToggle(dim.id, subId)}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
