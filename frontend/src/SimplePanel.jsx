import { X } from 'lucide-react';
import React from 'react';

export function SimplePanel({ 
  title, 
  children, 
  onClose, 
  icon,
  style,
  headerStyle = {},
  dragHandleProps, // For draggable handle props
  actions // Optional header actions
}) {
  return (
    <div 
      style={{
        background: 'rgba(20, 20, 25, 0.8)', 
        padding: '16px', 
        borderRadius: '12px',
        backdropFilter: 'blur(10px)', 
        border: '1px solid rgba(255,255,255,0.1)',
        display: 'flex', 
        flexDirection: 'column',
        boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        ...style
      }}
    >
      {/* Header */}
      <div 
        {...dragHandleProps}
        style={{
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center', 
          marginBottom: '16px',
          cursor: dragHandleProps ? 'move' : 'default',
          userSelect: 'none',
          ...headerStyle
        }}
      >
        <h2 style={{ 
          margin: 0, 
          fontSize: '18px', 
          fontWeight: 'bold', 
          background: 'linear-gradient(45deg, #00d2ff, #3a7bd5)', 
          WebkitBackgroundClip: 'text', 
          WebkitTextFillColor: 'transparent',
          display: 'flex', 
          alignItems: 'center', 
          gap: '8px' 
        }}>
          {icon && React.cloneElement(icon, { size: 18, color: '#00d2ff' })} 
          {title}
        </h2>
        {/* Actions & Close */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {actions}
          {onClose && (
             <button 
               onClick={onClose} 
               style={{
                 background: 'transparent', 
                 border: 'none', 
                 color: '#888', 
                 cursor: 'pointer', 
                 padding: '4px',
                 display: 'flex',
                 alignItems: 'center',
                 justifyContent: 'center',
                 transition: 'color 0.2s'
               }}
               onMouseOver={(e) => e.currentTarget.style.color = '#fff'}
               onMouseOut={(e) => e.currentTarget.style.color = '#888'}
             >
               <X size={18} />
             </button>
          )}
        </div>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: 'auto' }}>
        {children}
      </div>
    </div>
  );
}
