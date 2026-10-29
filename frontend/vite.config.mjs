import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import path from 'path'
import http from 'http'
import { spawn } from 'child_process'
import { existsSync, mkdirSync, openSync } from 'fs'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

/* ── autoCenter：开发模式自动拉起本机中心（:5001）─────────────
   「实时分析」等能力需要本机前向服务（.venv + server/server.py，权重在本机）。
   vite dev 启动时探测 127.0.0.1:5001；没起就用仓库自带的 .venv 拉起，全程本机、不联网。
   拉起后不随 vite 退出（后台独立进程，日志在 tempdata/center_auto.log），
   找不到本地环境（.venv / server.py）则静默跳过——线上/纯前端开发不受影响。 */
const REPO_ROOT = path.resolve(__dirname, '..')
const sleep = ms => new Promise(r => setTimeout(r, ms))

function probeCenter(timeoutMs = 1500) {
  return new Promise(resolve => {
    const req = http.get('http://127.0.0.1:5001/api/v1/health', { timeout: timeoutMs }, res => {
      res.resume()
      resolve(res.statusCode === 200)
    })
    req.on('timeout', () => { req.destroy(); resolve(false) })
    req.on('error', () => resolve(false))
  })
}

function autoCenter() {
  return {
    name: 'ai2050-auto-center',
    configureServer() {
      const py = path.join(REPO_ROOT, '.venv', 'Scripts', 'python.exe')
      const entry = path.join(REPO_ROOT, 'server', 'server.py')
      if (!existsSync(py) || !existsSync(entry)) {
        console.log('[auto-center] 本地没有 .venv/server.py，跳过自动拉起')
        return
      }
      void (async () => {
        if (await probeCenter()) { console.log('[auto-center] 本机中心 :5001 已在运行'); return }
        try { mkdirSync(path.join(REPO_ROOT, 'tempdata'), { recursive: true }) } catch { /* 已存在 */ }
        const out = openSync(path.join(REPO_ROOT, 'tempdata', 'center_auto.log'), 'a')
        const child = spawn(py, [entry], {
          cwd: REPO_ROOT,
          env: { ...process.env, AI2050_HOST: '127.0.0.1', AI2050_PORT: '5001' },
          stdio: ['ignore', out, out],
        })
        child.on('error', e => console.log('[auto-center] 拉起失败:', e.message))
        console.log(`[auto-center] 已拉起本机中心 (pid ${child.pid})，等待就绪…`)
        for (let i = 0; i < 120; i++) {          // 最多 6 分钟：首启有数据包预热
          if (await probeCenter(1000)) { console.log('[auto-center] 中心就绪 :5001'); return }
          await sleep(3000)
        }
        console.log('[auto-center] 等待超时——中心可能仍在预热，页面会每 5 秒自动重连')
      })()
    },
  }
}

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
  plugins: [react(), autoCenter()],
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
