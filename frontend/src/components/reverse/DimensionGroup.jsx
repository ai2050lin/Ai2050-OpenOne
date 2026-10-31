/**
 * 可折叠维度分组组件
 * 显示单个语言维度下的子维度toggle开关
 */

export default function DimensionGroup({ dimension, selected, onToggle }) {
  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(2, 1fr)',
      gap: '2px',
      padding: '2px 8px 6px 26px',
    }}>
      {dimension.subDimensions.map((sub) => {
        const isActive = selected[sub.id] || false;
        return (
          <div
            key={sub.id}
            onClick={() => onToggle(sub.id)}
            style={{
              display: 'flex', alignItems: 'center', gap: '4px',
              padding: '3px 5px',
              borderRadius: '4px',
              cursor: 'pointer',
              background: isActive ? dimension.color + '18' : 'transparent',
              border: `1px solid ${isActive ? dimension.color + '50' : 'transparent'}`,
              transition: 'all 0.15s',
              userSelect: 'none',
            }}
          >
            <div style={{
              width: '8px', height: '8px', borderRadius: '2px',
              background: isActive ? sub.color : '#444',
              border: `1px solid ${isActive ? sub.color : '#555'}`,
              transition: 'all 0.15s',
              flexShrink: 0,
            }} />
            <span style={{
              fontSize: '10px',
              color: isActive ? '#eef7ff' : '#888',
              fontWeight: isActive ? 600 : 400,
              lineHeight: 1.3,
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
  );
}
