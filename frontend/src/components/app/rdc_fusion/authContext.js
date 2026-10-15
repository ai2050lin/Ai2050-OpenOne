/* 当前登录账号（AuthGate 提供）。非组件模块：组件文件只导出组件。 */
import { createContext, useContext } from 'react';

export const AuthContext = createContext({ user: null, offline: false, logout: () => {}, refresh: () => {} });
export function useAuth() { return useContext(AuthContext); }
