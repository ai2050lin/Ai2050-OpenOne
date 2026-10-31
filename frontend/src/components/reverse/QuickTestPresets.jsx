/**
 * 十大核心测试快捷入口
 * T1-T10预设，点击自动配置面板
 */
import { TEST_PRESETS, STATUS_CONFIG } from '../../config/testPresets';

export default function QuickTestPresets({ activePreset, onApplyPreset }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      <span style={{ fontSize: '12px', fontWeight: 700, color: '#dfe8ff' }}>
        核心测试预设
      </span>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
        {TEST_PRESETS.map((preset) => {
          const isActive = activePreset === preset.id;
          const statusInfo = STATUS_CONFIG[preset.status] || STATUS_CONFIG.pending;
          const priorityColor = preset.priority === 'high' ? '#ff6b6b' : preset.priority === 'medium' ? '#fbbf24' : '#6b7280';

          return (
            <div
              key={preset.id}
              onClick={() => onApplyPreset(preset)}
              style={{
                display: 'flex', alignItems: 'center', gap: '6px',
                padding: '5px 7px',
                borderRadius: '5px',
                cursor: 'pointer',
                background: isActive ? 'rgba(79, 172, 254, 0.12)' : 'rgba(255,255,255,0.02)',
                border: `1px solid ${isActive ? 'rgba(79, 172, 254, 0.35)' : 'rgba(255,255,255,0.06)'}`,
                transition: 'all 0.15s',
                userSelect: 'none',
              }}
            >
              <span style={{ fontSize: '10px', width: '20px', fontWeight: 700, color: '#bbb' }}>
                {preset.id}
              </span>
              <span style={{ fontSize: '10px', flex: 1, color: isActive ? '#eef7ff' : '#999', fontWeight: isActive ? 600 : 400 }}>
                {preset.label}
              </span>
              <span style={{ fontSize: '9px', color: statusInfo.color }}>{statusInfo.icon}</span>
              <div style={{
                width: '4px', height: '4px', borderRadius: '50%',
                background: priorityColor, flexShrink: 0,
              }} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
