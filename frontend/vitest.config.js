import { defineConfig } from 'vitest/config';

// 渲染进程单测（client_architecture_review P0-1）：只测纯逻辑与请求层，不测组件渲染。
// VITE_API_BASE 指向 localhost → config.js 判定 RUNTIME_MODE='local'、OFFLINE_CLIENT=false，
// postJson 等在线路径才能在测试里走通（node 环境无 window，相关分支已被 config.js 守护）。
export default defineConfig({
  test: {
    environment: 'node',
    env: { VITE_API_BASE: 'http://localhost:5001' },
    include: ['src/**/*.test.js'],
  },
});
