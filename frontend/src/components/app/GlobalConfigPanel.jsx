import { RotateCcw, Settings } from 'lucide-react';

import { SimplePanel } from '../../SimplePanel';

export function GlobalConfigPanel({ visibility, onToggle, onClose, onReset, lang, onSetLang, t }) {
  const getLabelFor = (key) => {
    return t(`panels.${key}`) || key;
  };

  return (
    <SimplePanel
      title={t('panels.globalConfig')}
      onClose={onClose}
      icon={<Settings />}
      style={{
        position: 'absolute', top: 60, left: 20, zIndex: 100,
        minWidth: '220px'
      }}
    >
      <div style={{ marginBottom: '16px', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '12px' }}>
        <div style={{ fontSize: '12px', color: '#aaa', marginBottom: '8px' }}>{t('common.language')}</div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={() => onSetLang('zh')}
            style={{
              flex: 1, padding: '4px', borderRadius: '4px',
              border: lang === 'zh' ? '1px solid #4488ff' : '1px solid #444',
              background: lang === 'zh' ? 'rgba(68, 136, 255, 0.2)' : 'transparent',
              color: lang === 'zh' ? '#fff' : '#888',
              cursor: 'pointer', fontSize: '12px'
            }}
          >
            中文
          </button>
          <button
            onClick={() => onSetLang('en')}
            style={{
              flex: 1, padding: '4px', borderRadius: '4px',
              border: lang === 'en' ? '1px solid #4488ff' : '1px solid #444',
              background: lang === 'en' ? 'rgba(68, 136, 255, 0.2)' : 'transparent',
              color: lang === 'en' ? '#fff' : '#888',
              cursor: 'pointer', fontSize: '12px'
            }}
          >
            English
          </button>
        </div>
      </div>

      <div style={{ marginBottom: '16px' }}>
        {Object.entries(visibility).map(([key, isVisible]) => (
          <div key={key} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px', fontSize: '13px', alignItems: 'center' }}>
            <span style={{ color: '#ccc' }}>{getLabelFor(key)}</span>
            <button
              onClick={() => onToggle(key)}
              style={{
                background: isVisible ? '#4488ff' : '#333',
                border: 'none', borderRadius: '12px', width: '36px', height: '20px',
                position: 'relative', cursor: 'pointer', transition: 'background 0.2s'
              }}
            >
              <div style={{
                position: 'absolute', top: '2px', left: isVisible ? '18px' : '2px',
                width: '16px', height: '16px', background: '#fff', borderRadius: '50%',
                transition: 'left 0.2s'
              }} />
            </button>
          </div>
        ))}
      </div>

      <button onClick={onReset} style={{
        width: '100%', padding: '8px', backgroundColor: '#333', color: '#fff', border: 'none',
        borderRadius: '4px', cursor: 'pointer', fontSize: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
        transition: 'background 0.2s', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '12px'
      }}>
        <RotateCcw size={12} /> {t('panels.resetLayout')}
      </button>
    </SimplePanel>
  );
}
