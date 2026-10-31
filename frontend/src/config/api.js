/**
 * API configuration
 * Unified endpoint definitions.
 *
 * v2（2026-10-14）：改为转发 src/config.js 统一出口，消除本文件内的独立字面量。
 * 运行形态优先级：window.__AI2050__（Electron 注入）> VITE_*（dev）> 同源/本机默认。
 */
import { API_BASE, ANALYSIS_API_BASE, TRAINING_API_BASE, FRONTEND_BASE } from '../config.js';

export const API_CONFIG = {
  main: API_BASE,
  analysis: ANALYSIS_API_BASE,
  training: TRAINING_API_BASE,
  frontend: FRONTEND_BASE,
};

export default API_CONFIG;
