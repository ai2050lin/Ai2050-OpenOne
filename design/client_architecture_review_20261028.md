# 客户端框架逆向分析 · 使用流程 · 问题与改进路线

> 分析对象：`frontend/`（网站壳 + 研究工作台 + 归档旧界面）+ `frontend/electron/`（桌面客户端主进程）+ `deploy/center_app.py`（中心服务）+ `deploy/node_agent.py`（本机算力节点）
> 分析时间：2026-10-28 22:40 CDT｜代码版本：center 1.0.19｜前端 VERSION 无独立版本号（package.json 1.0.0）
> 方法：读源码 + 读构建产物 + 统计调用面（不看运行中的截图，结论均可在文件行号处复核）

---

## 0. 一句话结论

这个客户端**不是"一个网站 + 一个桌面壳"**，而是**一个 SPA + 一张运行形态布尔矩阵 + 三个外壳 + 五条数据通路**。它真正的复杂度不在 UI，而在"离线优先 / 角色 / 本机算力 / 三形态"这四件事被压进了同一套 `if` 判断里：58 处形态分支、166 处直接 `fetch`、608 个源文件、0 个前端测试、CI 不构建前端。功能是通的，工程上是**靠注释纪律撑住的高耦合单体**。

优先级判断：**先补可验证性（CI + 测试 + 错误可见），再收敛形态矩阵与数据层，最后才谈视觉/移动端。** 反过来做（先改 UI）会把已经散开的判断点再摊一遍。

---

## 1. 逆向：客户端到底是什么

### 1.1 三种运行形态（同一个代码库）

| 形态 | 判定来源 | 入口 | 数据通路 | 能否写 |
|---|---|---|---|---|
| 网页版（同源托管） | `RUNTIME_MODE='same-origin'`，`API_BASE=''` | `/`、`/workbench` | 相对路径 `/api/*` 打到同一个 FastAPI 进程 | 否（`ADMIN_LIVE=false`） |
| 远端网页 / vite dev | `VITE_API_BASE`，`RUNTIME_MODE='remote'/'local'` | 同上 | 跨域打到中心 | dev 本机为 true |
| 桌面客户端（Electron） | preload 注入 `window.__AI2050__`，`mode='electron'` | `dist/index.html` + `?wb=1` | 中心 + **本机节点服务（127.0.0.1 随机端口）** | 是 |

形态判定集中在 `src/config.js:56-109`，一次性在**模块 import 时求值**（`API_BASE` 是常量，运行中不可改），派生 5 个布尔：

```
OFFLINE_CLIENT = CLIENT_MODE!=='online' && RUNTIME_MODE!=='local'   // 只读数据包
ADMIN_LIVE     = !OFFLINE_CLIENT || (DESKTOP && CLIENT_MODE!=='offline')
DESKTOP / REMOTE_CENTER / WEB_URL
```

这三个布尔被 **58 处**代码引用（`OFFLINE_CLIENT` 31、`ADMIN_LIVE` 19、`DESKTOP` 8）。这就是架构的主矛盾：**能力判定没有单一出口**。

### 1.2 四层结构（桌面形态）

```
┌─ 渲染进程（React 19 + Vite 7，单页，无路由库）──────────────┐
│  main.jsx  按 pathname / ?wb=1 字符串分发 4 个入口：          │
│    /            → AtlasApp      网站壳（首页/介绍/参与/管理）  │
│    /workbench   → Workbench     工作台壳（进展/空间/研发/配置）│
│    /legacy      → App.jsx      旧 3D 界面（dev-only，266 KB） │
│    /rdc*        → 10 个专项页（dev-only）                      │
│  基础设施：config.js（形态）→ api.js（getJson/postJson/useApi）│
│           → dataStore.js（IndexedDB 数据包）                   │
└───────────────▲──────────────────────────────────────────────┘
                │ contextBridge（只读快照 + 白名单 IPC）
┌───────────────┴─ preload.js ─────────────────────────────────┐
│  __AI2050__{apiBase,mode}  ai2050Desktop{node,ai,runtime}     │
└───────────────▲──────────────────────────────────────────────┘
                │ ipcMain.handle ×17
┌───────────────┴─ 主进程 main.js ─────────────────────────────┐
│  NodeSupervisor  监护 python node_agent.py serve（退避重启）   │
│  RuntimeInstaller 装 Python3.14.6+torch cu130（sha256 逐包核） │
│  AiAssistant      Key 走 safeStorage 加密，SSE 流式回推页面    │
└──────┬───────────────────────────────┬───────────────────────┘
       │ HTTPS                          │ 127.0.0.1 + X-Local-Token
┌──────▼──────────────┐        ┌────────▼──────────────────────┐
│ 中心 FastAPI (:5010) │◄───────│ 本机节点 node_agent.py         │
│ 6 个 router，~93 路由│  注册/  │ 领任务/跑模型/上传分片          │
│ SQLite + 快照子进程  │  续租/  │ 令牌只在本机 service.json      │
│ 同源托管 frontend/dist│ 上传   └───────────────────────────────┘
└─────────────────────┘
```

### 1.3 数据读取的三级优先（这是整个客户端最核心的机制）

`api.js:94-145` 的 `useApi(path)` 实际上是**一个自制的缓存-回退栈**：

```
1) IndexedDB 主数据包（datapack.v1，键=规范化请求路径）→ 命中即用，不联网
2) IndexedDB 附加包（tm:<模式族> / cards:<模型>）       → 大体积数据按需下载
3) 在线 fetch（no-store）                              → 仅 config、auth、写操作、管理工位
4) 按需缓存（od|<包版本>|<路径>）                       → 合并统计/规律分析首次在线后落本地
```

配套：`dataStore.js` 用单个 KV store 存 `pack` / `x|<path>` / `od|<ver>|<path>` / `xmeta` / `xkeys|<id>`；`canonical()` 与后端 `deploy/datapack.py` 必须逐字一致（靠人工同步，无共享 schema）。

**评价**：这套设计在"研究数据体积大、中心带宽小"的约束下是对的，也是客户端最值钱的部分。问题在于它**没有抽象**——每个面板自己 `useApi` 拼 query，字段名靠约定（`pending`、`route_T`、`readable`…），没有 schema 校验。

### 1.4 Electron 安全边界（这块做得比多数项目好）

- `contextIsolation: true` + `nodeIntegration: false`，preload 只用 `contextBridge` 暴露值（注释里写清了"直接赋值只落在隔离世界"这个坑，`preload.js:9-14`）。
- 页面**拿不到**本机节点令牌与 AI Key：`nodeSupervisor` 白名单转发（`ROUTES` 集合 + 正则），Key 用 `safeStorage`(DPAPI) 加密后仅回传尾号（`aiAssistant.js:115-129`）。
- 计算组件安装是**供应链可验证**的：uv wheel 固定 sha256（`runtimeInstaller.js:23-29`）、`pip --require-hashes`、装完跑 torch/CUDA 探针才写 `runtime.json`。
- 外链一律 `shell.openExternal`，窗口打开一律 deny（`main.js:205-208`）。

### 1.5 规模事实（用于判断"该拆还是该改"）

| 指标 | 数值 | 说明 |
|---|---|---|
| 源文件 | 608（`src/`） | `blueprint/` 占 274，`components/` 233 |
| 最大单文件 | `App.jsx` 266 KB / 1.8 MB 产物 | 已归档为 `/legacy`，仍参与构建 |
| 其他巨文件 | ResearchHeatmapRoute 193 KB、GeminiTab 163 KB、AtlasControlDashboard 118 KB、LensSpatial 56 KB | >1000 行的单文件组件 |
| CSS | `rdc_fusion.css` 103 KB + `rdc_fusion_v6.css` 56 KB，两壳都全量 import | 首屏 CSS 188 KB |
| 直接 `fetch` | 166 处，散布 30+ 文件；`api.js` 只覆盖其中一部分 | 与 `config.js` 注释「唯一真源」纪律冲突 |
| 引用到的 API 路径 | 50 个去重字面量 / 后端 ~93 路由 | 动态拼接的还没算 |
| 前端测试 | **0** | `electron/*.test.js` 有 2 个（node:test） |
| CI 覆盖前端 | **无** | `.github/workflows/checks.yml` 只跑 Python 兼容性 |
| 孤儿组件 | 至少 24 个（无任何 import） | BuildHub、ConceptGallery、GeminiTab、GPT5Tab… |
| 形态分支 | 58 处 | 见 1.1 |

---

## 2. 使用流程（按真实点击顺序还原）

### 流程 A：桌面客户端首次启动

```
双击 start_client.bat / 安装包
 → main.js ensureServer()
     无配置 → 弹系统对话框[默认服务器 / 自定义 / 退出]
        → probe(GET /api/health, 8s) → 失败也放行，只警告"会显示未接入空态"
        → 写 userData/config.json{server,verified,verifiedAt}
 → createWindow(1600×980, minWidth 1280)
 → 加载 dist/index.html（未打包 + AI2050_CLIENT_DEV_URL 时改加载 vite）
 → nodeSup.start(server)：若无存活 service.json → spawn python node_agent.py serve --port 0
     → 轮询 service.json（30s 上限）→ GET /status 200 → state=running
     → 失败：2s→4s→…→60s 退避，10 分钟内 5 次失败即停
 → 渲染：AuthGate（访客直接进）→ AtlasApp 首页
 → 左下角 DataBar：无数据包时自动下载一次主包（datapack.v1, gzip）
```

关键体验点：**服务器地址是"用户输入的信任锚"**，但只校验了 `/api/health` 返回 200，没有强制 HTTPS、没有证书/身份提示（见 P0-3）。

### 流程 B：访客 → 登录 → 浏览空间

```
首页/介绍/参与（只读，不联网）
 → 「研究工作台 ↗」→ /workbench?lens=spatial
 → SpaceLens：左侧 3D 层塔 + 右侧 4 个 tab（神经元 / 语言图谱 / 响应点云 / 谱图关系）
     每个 tab 各自 useApi：/api/lang/tree、/api/model/skeleton、/api/neurons/tower…
     选中层按 tab 分开记（block / lblock / rlayer），全部写回地址栏
 → 右上角账号菜单 → 登录（authJson /api/auth/login → localStorage 存 token+user）
 → 管理员登录后：顶部出现「账号管理 / 数据管理」两个页面
```

### 流程 C：研发工位（能力随形态收放）

`workbenchUtil.visibleStations({live, admin, localTools})` 决定看到哪几个工位：

| 工位 | 网页 | 桌面 | 管理员 | 本机中心 |
|---|---|---|---|---|
| 题库 catalog | ✓ | ✓ | ✓ | ✓ |
| 贡献算力 contribute | ✓（只给下载入口） | ✓ 全功能 | ✓ | ✓ |
| 神经元测试 probe | ✗ | ✓ | ✓ | ✓ |
| AI 助手 | ✗ | ✓ | ✓ | ✓ |
| 模板工作室 / 试跑冻结 / 分发 / 审计 / 普查 | ✗ | ✗ | ✓ | ✓ |
| 自动测试 | ✗ | ✗ | ✓ | 仅 local_tools=1 |

串起来的主链是：`模板工作室 → 试跑与冻结 → 分发与节点 → 审计队列 → 普查作业`，`draft` 通过 `Workbench` 的 `useState` 在两个面板间传递（**刷新即丢**，靠 sessionStorage 没做）。

### 流程 D：贡献算力（唯一"带副作用到本机"的流程）

```
研发 → 贡献算力 → 节点服务状态（running/attached/no-python）
 → 计算组件：装 Python 3.14.6 + torch cu130（约 3 GB 下载 / 6 GB 磁盘，仅 Windows+NVIDIA，驱动≥580）
 → 设备：nvidia-smi 表 + torch 探针 + 磁盘 + 电源
 → 账号与节点：绑定码（/api/auth/node-code 一次性）→ POST /bind
 → 模型：下载与 sha256 核对（每个 1–30 GB）
 → 资源限制 → 开始 / 暂停 / 恢复 / 停止 → 日志与「我的记录」
```

中心侧对应 `deploy/node_agent.py`（769 行）+ `deploy/node_service.py`（1230 行）；上传有分片落盘 + 幂等 finish（最近一次提交 `ec8376669` 就是在补这块）。

---

## 3. 问题清单（按"会不会让研究出错"排序，而不是按好不好看）

### P0 — 会让结论不可信 / 会让用户丢数据或被盗

**P0-1 前端零测试 + CI 不构建前端，等于没有回归网**
证据：`src/**` 下 `*.test.*` 数量为 0；`.github/workflows/checks.yml` 只有 Python 3.9/3.10 兼容性 job，无 `npm ci` / `npm run build` / `eslint`。
后果：QUICKSTART 里写着"仓库仍有历史组件的 lint 技术债"，说明**已知无人守门**。一个面板取错字段（如 P0-2）在评审时看不出来，上线后表现为"空态"，而空态本身是设计允许的，于是错误被静默吸收。
修法：CI 加三关（`vite build` 必须成功 / 新增文件 eslint 零 error / vitest 跑 `canonical`、`packHint`、`visibleStations`、`resolveGo`、`useApi` 纯函数部分）。

**P0-2 数据契约无校验，字段错了不报错，只变空白**
实证：`ConfigView.jsx:45` 读 `cfg?.pending`，但 `useApi` 的返回里**没有** `pending` 字段（只有 `loading`，见 `api.js:137-144`）。结果"读取中…"永远不显示，连接状态在首次加载时就写成"已连接"。这是最小的一例，同类风险遍布：`data.pending`、`route_T`、`skeleton.census_run`、`item.selectivity[].by_readout` 全靠后端与前端人工对齐。
修法：给 `useApi` 加轻量 schema（手写 `check(obj)` 或 zod-lite）；顶层响应统一 `{ok, data, pending, error}` 信封；`pending` 收进 `useApi` 返回值。对 `datapack.v1` 增加逐条 `sha256` + `schema_version`，版本不匹配时明确提示"客户端过旧/数据包过旧"。

**P0-3 服务器地址可指向任意 HTTP 主机，账号密码明文提交**
证据：`main.js:53-58 normalizeServer` 只补 `http://` 前缀，不限制协议；`api.js:186-205 authJson` 把密码/令牌 POST 到该地址。
后果：用户被诱导填一个恶意"自建中心"，客户端会把用户名密码直接交出去。桌面客户端是**用户本机上的可信程序**，这个信任边界必须自己守。
修法：非 localhost 强制 https；首次连接展示主机名 + 证书颁发者并要求二次确认；`config.json` 记录 `pinned_host`，地址变更时重新确认；CSP + `connect-src` 白名单到该 host。

**P0-4 渲染进程崩溃/异常完全不可见**
证据：只有根级 `ErrorBoundary`（`main.jsx:45-46`），无 `window.onerror` / `unhandledrejection`；主进程 `_push` 的 tail 只在内存（`nodeSupervisor.js:108-113`），不落盘；`aiAssistant` 有日志，客户端本身没有。
后果：一个面板抛错 → 整页白屏，用户只能"重新加载"；研究者现场问题无法复现、无法取证。
修法：`ErrorBoundary` 下沉到每个 tab（失败只黑一块）；全局错误 + `unhandledrejection` 落 `userData/logs/renderer-<date>.log` 且滚动清理；「帮助 → 导出诊断包」（版本、config 脱敏、最近 500 行日志、数据包版本）。

**P0-5 只读模式的权限判定是"缓存用户说了算"**
证据：`AuthGate.jsx:102-107`，`OFFLINE_CLIENT` 下不校验令牌，直接用 `localStorage` 里的 user；`useAdminToken`（`api.js:221-222`）据缓存里的 `role==='admin'` 把 session 当管理令牌发出去。
影响：令牌过期/被降权后，界面仍显示管理入口，点击后 401 —— 不是越权（服务端有校验），但会让人误判"平台坏了"。
修法：能力判定改为"服务端下发 capability + 本地乐观缓存带 `exp`"，过期即刻降级为访客并在账号菜单提示"会话已过期，请重新登录"。

### P1 — 会让维护成本指数上升（当前主要痛点）

**P1-1 没有路由库，三个外壳靠字符串判断分发**
证据：`main.jsx:36-54` 用 `pathname` + `?wb=1` 决定渲染哪个 App；`RDC_ROUTES` 手写 10 条 dev-only 路由；`workbenchHref/siteHref` 手工区分 `file:` 协议前后缀（`v8Util.js:24-38`）。
后果：深链、后退、嵌套状态、Electron `file://` 兼容全部要手工补；`/atlas`、`/rdc-fusion`、`?tab=`、`?page=` 四套旧链接靠 `WORKBENCH_REDIRECT` 表兜。
修法：引入 react-router（或 60 行自研 `useRoute`），路由表 `[{path, layout, element, capability}]`；`file://` 只在 `createHashRouter` 层兜一次。

**P1-2 形态矩阵散落 58 处，同一个"能不能写"有三套算法**
证据：`Workbench.jsx:99 isAdmin = !OFFLINE_CLIENT || role==='admin'`（把"非离线"当成管理员）、`visibleStations()`、`postJson` 内的 `ADMIN_LIVE` 抛错、`AccountCard` 的 `mismatch` 判断——彼此独立演化。
修法：单一 `capabilities` 对象由 `config.js` 一次算出（`read / write / admin / localNode / aiChat / mobile`），组件只查 `cap.write`；`isAdmin` 从"形态"里剥离，只由账号角色决定。

**P1-3 API 层形同虚设：166 处直接 fetch**
证据：`grep -c "fetch("` = 166，出现于 30+ 文件；而 `api.js` 注释写「统一 API_BASE」。`postJson` 不带 `AbortSignal`（`api.js:60-78`）→ 请求无法取消、无超时，弱网下按钮永久 busy。
修法：`fetchJson` 统一加 20s 超时 + 一次幂等 GET 重试 + 结构化错误码（`network / http / offline_missing / auth`）；用 eslint `no-restricted-globals`/自定义规则禁止 `src/` 里出现裸 `fetch`；分批迁移，先迁 5 个高频面板。

**P1-4 巨型单体与孤儿文件并存**
证据：`App.jsx` 266 KB（1.8 MB chunk，仍参与构建）、`ResearchHeatmapRoute.jsx` 193 KB、`GeminiTab.jsx` 163 KB；24+ 孤儿组件（`BuildHub`、`ConceptGallery`、`GPT5Tab`、`ResearchAuditTab`…）；`HLAIBlueprint.jsx.bak_mojibake_fix` 这类备份文件还在源码树。
后果：搜索/重构/新人上手成本极高；`v8Util.js` 从 `Space3D.jsx`（组件文件）import 常量 `CATEGORICAL`，把组件模块拉进非组件依赖图 —— 典型的分层泄漏。
修法：`src/{infra,ui,feature,legacy}` 四层；`legacy/`（App.jsx + blueprint 157 文件 + v5/v6 面板）改成**独立构建入口**或移到 `archive` 分支，不进主包；删除孤儿与 .bak（用 `knip`/`ts-prune` 定期跑）。

**P1-5 双壳重复实现同一件事**
证据：`AtlasApp` 与 `Workbench` 各自 import 两份大 CSS、各自的 `<header>`、各自 `AccountMenu`、各自 `useApi('/api/platform/config')`、各自 `document.title` 逻辑；两套"页面状态写地址栏"（`workbenchUtil.writeState` 与 `v8Util.writeState`）各自 delete 自己的键。
修法：抽 `layouts/Shell`（顶栏 + 账号 + 数据条 + 错误横幅）；页面状态统一进 router 的 search params，一个序列化器。

**P1-6 离线优先的复杂度没有被封装，全部外溢到界面**
证据：`api.js:18-32 offlineJson` 里用字符串拼 4 种提示；`DataBar` 自己管"首次自动下载 / 版本比对 / 进度 / 失败"；`ExtraPackPrompt` 又一套；`missingPack` 通过 error 对象上的非标字段回传。
修法：`offline` 成为一等状态机 `{state: fresh|stale|missing|empty, action}`；缺包渲染统一 `<PackGate>` 组件；错误带 `code` 而不是中文句子。

### P2 — 产品与工程的手尾

- **P2-1 桌面端没有自动更新**：`scripts` 里只有 `electron-builder --win`，主进程无 `autoUpdater`。用户靠"重新下载安装包"。建议 `electron-updater` + 代码签名 + 启动时检查协议版本（client ⇄ center）。
- **P2-2 打包声明支持 Linux，实际跑不了贡献算力**：`package.json` 有 `linux: AppImage`，但 `runtimeInstaller.js:178` 明确 `当前只支持 Windows + NVIDIA`，`nodeSupervisor` 还硬编码 `C:\Windows\System32\tar.exe` `/nvidia-smi.exe`。要么去掉 linux target，要么把平台能力写进 `runtime-capabilities` 并在 UI 明示。
- **P2-3 首屏与移动端**：`index.html` `lang="en"`（中文站）、favicon 还是 `vite.svg`、无 CSP、无 `theme-color`、无 noscript；`v8.css` 仅 4 条媒体查询、Electron `minWidth:1280` → 手机打开公开站基本不可用。既然文档把"网页版"当正式形态，这条是产品缺口而非美化项。
- **P2-4 跨域过宽**：`center_app.py:46` `allow_origins=['*']` + `allow_headers=['*']`。token 不自动随请求走，风险有限，但公开中心应收敛到站点自身域名。
- **P2-5 开发服务器暴露到局域网**：`vite.config.mjs` `host:'0.0.0.0'` + `fs.strict:false`。同一 WiFi 下可访问你的源码与调试端点。默认应为 `127.0.0.1`，需要联调时显式开。
- **P2-6 依赖体积与拆分**：`dist` 3.9 MB；`plotly.js` / `three` / `recharts` 全在 `devDependencies` 但被打包（注释已解释原因），无 `manualChunks`。目标是首屏 JS<150 KB gz，3D/图表按 tab 懒加载。
- **P2-7 术语与命名债**：v5/v6/v8、`rdc_fusion`、`atlas`、`fusion`、`blueprint` 并存；`components/app/rdc_fusion/` 里混着 v8 使用的基础设施（`api.js`/`dataStore.js`/`ui.jsx`）与旧面板。建议改名 `infra/`（把 api/store/config 移出），把"v8"这类版本号从目录名里去掉。
- **P2-8 无客户端可观测性**：无错误率、无页面耗时、无版本分布。中心侧建议开一个匿名 `/api/client/telemetry`（只收版本、错误码、耗时直方图）。
- **P2-9 `draft` 只在内存**：模板工作室 → 试跑冻结之间用 `useState` 传 `draft`，刷新/切页即丢。应落 `sessionStorage` 或服务端草稿。
- **P2-10 无 a11y/键盘/对比度审计**：`aria-*` 写得不少（好评），但没有自动化检查；3D 层塔几乎不可键盘操作。

---

## 4. 改进路线（按投入产出排序，不做"大而全重构"）

### 第 1 阶段｜止血（约 1 周，不改架构）
1. CI 三关：`vite build` + 变更文件 eslint + vitest 冒烟（P0-1）。
2. `fetchJson` 加超时/取消/错误码；`postJson` 接受 `signal`（P1-3 的最小切片）。
3. `useApi` 补 `pending`、修 `ConfigView` 误用；给 5 个最高频响应加手写 schema 断言（P0-2）。
4. 强制 https + 主机确认 + CSP + `lang="zh-CN"` + 真 favicon（P0-3、P2-3）。
5. 全局错误落盘 + 「导出诊断包」（P0-4）。
6. 删孤儿组件与 `.bak`（低风险，立刻减少 10% 搜索噪音）。

**验收**：故意改坏一个字段名，CI 或运行时能在 1 秒内指出具体面板与字段，而不是白屏/空态。

### 第 2 阶段｜收敛（约 3–4 周，动结构不动功能）
7. 路由化：`react-router` 接管 `/`、`/workbench`、`/legacy`、`/rdc*`；`file://` 用 hash router 兜一次（P1-1）。
8. 能力矩阵：`capabilities{read,write,admin,localNode,aiChat}` 单点计算，替换 58 处分支（P1-2）。
9. 数据层：`useResource(key)` 统一缓存/去重/超时/陈旧标记；`offline` 变状态机 + `<PackGate>`（P1-4、P1-6）。
10. 分层落地：`src/infra`（config/api/store/auth）、`src/ui`（Card/Pill/Empty/Table 等 20 个原子）、`src/feature/*`、`src/legacy`（独立构建入口，不进主包）；`v8Util` 不再从组件文件 import 常量（P1-4、P1-5）。
11. 外壳合并：`layouts/Shell` 统一顶栏、账号、数据条、错误横幅（P1-5）。

**验收**：新增一个"研发工位"只需加 1 个 `STATIONS` 条目 + 1 个组件文件，不需要改任何 `if (ADMIN_LIVE)`；主包首屏 JS < 150 KB gz。

### 第 3 阶段｜产品化（约 1–2 月）
12. `electron-updater` + 签名 + 协议版本协商（P2-1）。
13. 平台能力声明替代 `process.platform` 硬编码；明确 Linux/macOS 是"只读客户端"（P2-2）。
14. 公开站响应式 + 工作台窄屏只读模式（P2-3）。
15. 匿名遥测 + 中心侧版本/错误看板（P2-8）。
16. Playwright e2e 四条主流程（访客浏览 / 下载包 / 登录 / 工位可见性），跑在 CI 的 `--offline` 假中心上。

---

## 5. 值得保留的设计（不要在重构里丢掉）

1. **"没有数据就显示未接入，不伪造数字"** —— 贯穿 `ui.jsx` 的 `Empty`、`offlineJson` 的四段文案、`ConfigView` 的"每个值都有来源"。这是科研平台的底线，比任何 UI 都重要。
2. **离线优先的 datapack + 附加包分层** —— 在带宽受限下让"读全量图谱"成为可能，且键空间设计（`canonical`）与后端同构。
3. **Electron 侧的三条硬边界**：令牌/Key 不出主进程、IPC 白名单、安装包 sha256 逐包核对。
4. **注释即设计文档** —— 每个关键文件顶部写清"为什么这样"（`preload.js` 的 contextIsolation 坑、`vite.config` 的 `base:'./'` 三形态论证）。重构时这些注释要一起搬走，它们比 README 有效。

---

## 6. 如果只做三件事

1. **让错误可见**（CI + 全局错误落盘 + schema 断言）—— 现在任何回归都表现为"空态"，而空态被设计正当化，这是最危险的一点。
2. **把 58 处形态分支收成 1 个 capabilities 对象** —— 这是所有"网页/桌面行为不一致"类 bug 的根。
3. **把 166 处 fetch 收进一个带超时/取消/错误码的客户端** —— 弱网与离线是常态，不是异常。

三件事都不需要改视觉、不需要重写组件，且都能在两周内完成并立刻降低后续所有改动的成本。

---

### 附：本文引用位置索引

| 结论 | 位置 |
|---|---|
| 形态判定与 5 个布尔 | `frontend/src/config.js:56-109` |
| 入口字符串分发 | `frontend/src/main.jsx:36-54` |
| 数据读取四级优先 | `frontend/src/components/app/rdc_fusion/api.js:36-45,94-145` |
| IndexedDB 键空间 | `frontend/src/components/app/rdc_fusion/dataStore.js:10-11,119-135,202-228` |
| `cfg.pending` 误用 | `frontend/src/components/app/ConfigView.jsx:45` vs `api.js:137-144` |
| 离线不校验会话 | `frontend/src/components/app/rdc_fusion/AuthGate.jsx:102-107` |
| 管理员判定混入形态 | `frontend/src/components/app/Workbench.jsx:99` |
| 地址可指向任意 HTTP | `frontend/electron/main.js:53-58,61-87` |
| 主进程 IPC 与监护 | `frontend/electron/main.js:328-395`、`nodeSupervisor.js:19-23,140-203` |
| 计算组件供应链校验 | `frontend/electron/runtimeInstaller.js:23-36,178-227` |
| 平台硬编码 | `runtimeInstaller.js:178`、`nodeSupervisor.js:42,232` |
| CORS 过宽 | `deploy/center_app.py:46` |
| 开发服务器暴露 | `frontend/vite.config.mjs:20-30` |
| 前端 CI 缺失 | `.github/workflows/checks.yml`（仅 Python job） |

---

## 11. 独立复核记录与合并改进方案（2026-10-28 22:55 CDT）

本节是对上文评审的独立复核（逐条跑命令验证），以及与同日另一轮基础设施侧审计（platform_gap_review_v1 §10）的合并结论。

### 11.1 逐条核验结果（全部属实）

| 评审断言 | 复核方式 | 结果 |
|---|---|---|
| 前端 0 测试、CI 不构建前端 | `find frontend/src -name "*.test.*" -o -name "*.spec.*"` = 0；`.github/workflows/` 仅 Python | ✅ |
| `cfg?.pending` 字段不存在 | `ConfigView.jsx:46` 读 pending；`useApi` state 只有 `{key,data,error,pack}`（api.js:98）→ 加载中假显"已连接" | ✅ |
| 服务器地址可指任意 HTTP、密码明文 | `main.js:53-58 normalizeServer` 无 https 强制；`authJson`（api.js:186-192）把含密码的 body POST 到该地址 | ✅ |
| 渲染进程异常不可见 | 全 src 无 `window.onerror`/`unhandledrejection`；仅 ErrorBoundary 组件 | ✅ |
| 166 处直接 fetch、postJson 无超时/取消 | `grep -c "fetch("` 全 src = 166；getJson/postJson/authJson 均无 AbortController/timeout | ✅ |
| App.jsx 266KB / ResearchHeatmapRoute 193KB / .bak 入树 | 266,096 / 193,232 字节；`src/HLAIBlueprint.jsx.bak_mojibake_fix` | ✅ |
| "非离线即管理员" | `Workbench.jsx:95 isAdmin = !OFFLINE_CLIENT \|\| user?.role==='admin'` | ✅ |

### 11.2 本评审未覆盖、但同日另一轮审计已证实的同等重要问题

1. **`.env.development` 指向线上中心**（VITE_API_BASE=https://server.ai2050.com）——与 README"vite dev 连本机 5001"矛盾，是"本地跑不通"的最大单点，直接影响第 1 节"本机开发"形态的可用性。
2. **默认管理令牌 `'ai2050-admin-dev'`**（`deploy/distributed_service.py:78`）——绕过 center_app 启动即带可猜令牌，与本文"地址可指任意 HTTP"同属安全 P0。
3. **`POST /api/grammar/preview` 无鉴权**、`/api/nodes/register` 无速率限制——公开中心的资源滥用面。
4. **NewbieGuide 新手引导是死代码**（只被已下线的 RdcFusionWorkspace 引用）——本文 §6 的引导缺口由此而来。
5. **中心侧后台 worker 可长时间 pending**（lang_atlas/skeleton 预热 >6min 无进度反馈）——与本文"错误被静默吸收"同根：用户分不清"在算"和"坏了"。

### 11.3 合并后的改进路线（两轮审计取并集，排序不变）

**P0 止血（1 周内）**
1. CI 三关：vite build + eslint + vitest（先给 api.js/config.js/数据层补 10–15 个单测，不追覆盖率）。
2. 错误可见：全局 `window.onerror`/`unhandledrejection` 落盘到本地日志 + 诊断导出；修 `cfg?.pending`（useApi 增加 pending 状态）。
3. 安全三件套：normalizeServer 强制 HTTPS（自签/本地地址白名单豁免）；ADMIN_TOKEN 默认值改空+未设置即拒绝启动（center_app 之外）；grammar/preview 加鉴权或限流。
4. `.env.development` 与 README 对齐（默认 localhost:5001，注释给切线上的一行开关）。
5. 请求层收口第一步：postJson/authJson 加超时（AbortSignal.timeout）与错误码。

**P1 收敛（3–4 周）**
6. `capabilities{read,write,admin,localNode,aiChat}` 单一出口，替换 58 处形态布尔分支；admin 判定并入（会话角色 > 形态推断）。
7. `useResource` 统一数据层：offline/online/附加包做成显式状态机，fetch 全量收口（消灭 166 处散点）。
8. react-router 接管四入口；legacy（App.jsx/rdc-* 专项页）独立 chunk 不进主包；删孤儿文件与 .bak。
9. 中心 worker 进度可见化：快照/预热状态带"进行中步骤 + 上次成功时间"（解决 pending 无限等的判定歧义）。
10. legacy 题处置（归档或迁移 v3 通道，见 platform_gap_review_v1 §10）。

**P2 产品化（1–2 月）**
11. electron-updater 自动更新 + 协议版本协商；Linux 支持或声明收缩到 Windows。
12. 四条主流程 Playwright e2e（浏览/下载包/贡献算力/研发冻结）；响应式公开站。
13. release 大包移出 git；三账本收敛与 atlas verify CLI 打通（平台侧 P0/P1，见 platform_gap_review_v1）。

### 11.4 复核结论

本评审的关键断言**全部经独立命令核验属实**，与同日基础设施侧审计互补：本评审管"客户端工程质量"（可测性/错误可见/形态收敛），后者管"配置矛盾与安全兜底"（.env、默认令牌、无鉴权端点）。合并路线以上表为准；"三件最值得保留"（不伪造数字纪律、离线分层、Electron 硬边界）的判断与另一轮审计一致，共同确认为不可砍的护城河。

### 11.5 P0 执行记录（2026-10-29 00:30 CDT，全部完成并验证）

| P0 项 | 状态 | 实现 |
|---|---|---|
| 修 `cfg?.pending` | ✅ | `useApi` 返回形状补 `pending`（与 loading 同源），加载中显示"读取中" |
| 错误可见 | ✅ | 新增 `src/diagnostics.js`：onerror/unhandledrejection → localStorage 环形缓冲（50 条）；ConfigView 新增「诊断」卡（计数/导出诊断包/清除）；`main.jsx` 挂载 |
| HTTPS 强制 | ✅ | `electron/main.js normalizeServer`：非本机地址（localhost/127.0.0.1/::1 豁免）http 自动升级 https，probe 兜底验证 |
| ADMIN_TOKEN | ✅ | `distributed_service.py`：去掉 `'ai2050-admin-dev'` 默认值；未设环境变量时随机 token_hex(16) 并打印日志（管理员会话仍可用） |
| preview 鉴权 | ✅ | `/api/grammar/preview` 加 `_auth_admin`；TemplateStudio 经 `useAdminToken` 传令牌（管理员会话自动回退） |
| .env 对齐 | ✅ | `.env.development` 默认 `localhost:5001`，注释保留切线上开关 |
| 请求超时 | ✅ | `getJson/postJson/authJson` 加 `withTimeout`（默认 60s，可覆盖，与调用方 AbortController 合并；TimeoutError 转可读文案；数据包下载不受影响） |
| CI 三关 | ✅ | `checks.yml` 新增 frontend job：vitest + vite build 硬门槛，lint 非阻塞（存量 732 问题清零后转硬）；`package.json` 加 `test` 脚本 |
| 单测 | ✅ | vitest + happy-dom：`diagnostics.test.js`（9）+ `api.timeout.test.js`（2）+ `catalogUtil.test.js`（3）= **14/14 通过** |

**验证**：后端 TestClient 冒烟 8/8（随机令牌生效、旧开发令牌被拒 401、正确令牌 200、公开端点无回归）；electron 既有 node --test 全过；vite build 通过；本次改动文件 eslint 0 问题（存量 732 个均为遗留代码）。
