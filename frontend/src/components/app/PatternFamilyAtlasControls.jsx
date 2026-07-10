import { Eye, EyeOff, Network, SlidersHorizontal } from 'lucide-react';
import './PatternFamilyAtlasControls.css';

const FOCUS_OPTIONS = [
  { value: 'key', label: '关键候选' },
  { value: 'natural', label: '自然交叉' },
  { value: 'group', label: '组级支持' },
];

export function PatternFamilyAtlasControls({
  enabled,
  onEnabledChange,
  atlas,
  familyId,
  onFamilyChange,
  modelKey,
  onModelChange,
  evidenceFocus,
  onEvidenceFocusChange,
  maxUnits,
  onMaxUnitsChange,
}) {
  if (!enabled) {
    return (
      <button
        type="button"
        className="pattern-atlas-restore"
        onClick={() => onEnabledChange(true)}
        title="打开模式族神经元脉络"
        aria-label="打开模式族神经元脉络"
      >
        <Network size={18} />
      </button>
    );
  }

  const metrics = atlas.partition?.metrics;
  const mappingStatus = atlas.mapped ? '物理候选已映射' : '真实单元尚未映射';
  const mappingTone = atlas.mapped ? 'mapped' : 'unmapped';

  return (
    <section className="pattern-atlas-controls" aria-label="模式族神经元脉络控制">
      <header className="pattern-atlas-controls__header">
        <div className="pattern-atlas-controls__title">
          <Network size={16} />
          <span>语言模式族神经元脉络</span>
          <span className={`pattern-atlas-status pattern-atlas-status--${mappingTone}`}>{mappingStatus}</span>
        </div>
        <button
          type="button"
          className="pattern-atlas-icon-button"
          onClick={() => onEnabledChange(false)}
          title="返回原工作台"
          aria-label="返回原工作台"
        >
          <EyeOff size={16} />
        </button>
      </header>

      <div className="pattern-atlas-controls__body">
        <label className="pattern-atlas-field">
          <span>模式族</span>
          <select value={familyId} onChange={(event) => onFamilyChange(event.target.value)}>
            {atlas.families.map((family) => {
              const mapped = (family.physical_mapping?.models || []).length > 0;
              return (
                <option key={family.family_id} value={family.family_id}>
                  {family.family_name}{mapped ? '' : ' · 未映射'}
                </option>
              );
            })}
          </select>
        </label>

        <label className="pattern-atlas-field pattern-atlas-field--model">
          <span>模型</span>
          <select value={modelKey} onChange={(event) => onModelChange(event.target.value)} aria-label="模型">
            <option value="qwen3-4b">Qwen3</option>
            <option value="glm4-9b">GLM4</option>
            <option value="ds7b">DS7B</option>
          </select>
        </label>

        <div className="pattern-atlas-segment" aria-label="证据筛选">
          {FOCUS_OPTIONS.map((option) => (
            <button
              key={option.value}
              type="button"
              className={evidenceFocus === option.value ? 'is-active' : ''}
              onClick={() => onEvidenceFocusChange(option.value)}
              disabled={!atlas.mapped}
            >
              {option.label}
            </button>
          ))}
        </div>

        <label className="pattern-atlas-range">
          <SlidersHorizontal size={14} />
          <span>关键单元 {maxUnits}</span>
          <input
            type="range"
            min="12"
            max="96"
            step="12"
            value={maxUnits}
            onChange={(event) => onMaxUnitsChange(Number(event.target.value))}
            disabled={!atlas.mapped}
          />
        </label>

        <button
          type="button"
          className="pattern-atlas-icon-button pattern-atlas-icon-button--visible"
          onClick={() => onEnabledChange(true)}
          title="当前显示模式族脉络"
          aria-label="当前显示模式族脉络"
        >
          <Eye size={16} />
        </button>
      </div>

      <footer className="pattern-atlas-controls__footer">
        {atlas.loading ? (
          <span>正在读取证据分区</span>
        ) : atlas.error ? (
          <span className="is-error">{atlas.error}</span>
        ) : atlas.mapped ? (
          <>
            <span>{atlas.model}</span>
            <span>{metrics?.candidate_layer_count || 0} 个关键层</span>
            <span>{metrics?.unique_unit_count || 0} 个唯一候选</span>
            <span>{metrics?.natural_overlap_count || 0} 个自然交叉</span>
            <span>{metrics?.group_supported_candidate_count || 0} 个组级支持候选</span>
            <span className="pattern-atlas-boundary">单神经元因果 0</span>
          </>
        ) : (
          <span>{atlas.family?.family_name || '当前模式族'} · 等待真实物理映射</span>
        )}
      </footer>
    </section>
  );
}
