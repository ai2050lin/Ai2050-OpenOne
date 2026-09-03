import { API_CONFIG } from './api';


const ASSET_API_BASE = `${API_CONFIG.main.replace(/\/$/, '')}/api/research-assets/file`;


export function researchAssetUrl(path = '') {
  const value = String(path || '').trim();
  if (!value) return ASSET_API_BASE;
  if (/^https?:\/\//i.test(value)) return value;
  if (value.startsWith('/api/research-assets/')) {
    return `${API_CONFIG.main.replace(/\/$/, '')}${value}`;
  }

  const normalized = value
    .replace(/^\/?vis_data\//, '')
    .replace(/^\/+/, '');
  const encoded = normalized
    .split('/')
    .filter(Boolean)
    .map((segment) => encodeURIComponent(segment))
    .join('/');
  return encoded ? `${ASSET_API_BASE}/${encoded}` : ASSET_API_BASE;
}


export const RESEARCH_ASSET_API_BASE = ASSET_API_BASE;
