import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// https://vite.dev/config/
export default defineConfig({
  root: '.',
  /* base:'./' —— 相对资源路径（2026-10-14 部署改造）。
     三种运行形态都要能加载：
       ① vite dev（本仓库根）      → /assets/... 由 dev server 解析，相对路径同样可用
       ② nginx 同源托管（CentOS）   → 站点根目录，相对路径最佳
       ③ Electron file:// 加载 dist → **必须相对**，绝对路径 /assets/... 会指到文件系统根 → 白屏
     改前 dist 实测生成 src="/assets/main-*.js"（绝对），Electron 下必然白屏。 */
  base: './',
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    watch: {
      usePolling: true,
      interval: 1000,
    },
    fs: {
      strict: false,
    },
  },
  build: {
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'index.html'),
        annotation: path.resolve(__dirname, 'annotation.html'),
      },
    },
  },
})
