# 可视化客户端快速启动

## 推荐启动方式

在仓库根目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_visualization.ps1
```

启动成功后访问：

- 主客户端：<http://localhost:5173>
- 人工标注工作台：<http://localhost:5173/annotation.html>

也可以在 VS Code 的“运行和调试”中选择 `Frontend` 或 `Full Stack (Backend + Frontend)`。启动配置调用同一个脚本，不再依赖某个用户目录中的固定 npm 路径。

注意：单独运行 `server/server.py` 只会启动 5001 后端，不会自动启动 5173 前端。若
5001 已经有后端运行，`Full Stack (Backend + Frontend)` 会复用它，并让前端继续运行，
不会因为重复后端进程正常退出而连带停止前端。

### 打开工程即自动确保客户端在跑

`.vscode/tasks.json` 里的 `AI2050: Frontend (vite :5173)` 设置了
`runOn: folderOpen`：打开本工程时自动执行一次“确保可用”。该动作**幂等**——
已经在跑就直接复用并打印地址，不会起第二个进程（5173 用 `--strictPort`）。

相关任务：

| 任务 | 作用 |
|---|---|
| `AI2050: Frontend (vite :5173)` | 确保可用（默认脱管，关终端不影响服务） |
| `AI2050: Frontend (foreground, Ctrl+C 停止)` | 前台守护，便于现场观察 |
| `AI2050: Frontend Status (:5173)` | 端口 / 守护进程 / vite PID / 日志路径 |
| `AI2050: Frontend Stop (:5173)` | 停止并等待端口真正释放 |

### 启动链路（2026-10-19 起）

```text
start_visualization.ps1  →  发现 Node.js（版本校验）  →  scripts/dev_server.mjs  →  vite
                                                     ├─ 幂等：健康就复用
                                                     ├─ 回收：本项目僵死 vite 自动清掉
                                                     ├─ 就绪门：HTTP 200 且含 id="root" 才打印地址
                                                     ├─ 自愈：vite 异常退出按 600ms→…→8s 退避重启
                                                     └─ 可观测：.run/devserver.log（带时间戳）
```

改造原因：此前 vite 是**一次性前台进程**，挂在 VSCode 调试会话/终端上——会话重连失败、
终端被回收、窗口 reload 都会把它带走，而 5001 后端是独立任务仍然活着，于是出现
“后端在、页面打不开”的半死状态；并且进程一死终端里的报错就没了，故障无法定位。

## 为什么不直接依赖 `npm run dev`

部分开发环境已经包含 Node.js，但没有把 `node` 和 `npm` 加入终端 PATH。此时会出现：

```text
npm is not recognized
```

启动器会依次检查：

1. 环境变量 `AI2050_NODE_HOME`；
2. 当前 PATH；
3. 标准 Node.js 安装目录；
4. 本机 WorkBuddy Node.js 运行时；
5. 本机 Codex Node.js 运行时。

找到运行时后，它只修改当前启动进程的 PATH，不修改系统环境变量。

## 环境要求

前端使用 Vite 7，需要以下任一版本：

- Node.js 20.19 或更高的 20.x；
- Node.js 22.12 或更高；
- 更新的主版本。

如果 Node.js 安装在自定义目录，可以显式指定：

```powershell
$env:AI2050_NODE_HOME = 'C:\path\to\nodejs'
.\scripts\start_visualization.ps1
```

## 首次安装或修复依赖

```powershell
.\scripts\start_visualization.ps1 -Install
```

该命令使用锁文件执行 `npm ci`，随后启动开发服务器。

## 其他模式

```powershell
# 生产构建
.\scripts\start_visualization.ps1 -Mode build

# 预览生产构建
.\scripts\start_visualization.ps1 -Mode preview

# 使用其他端口
.\scripts\start_visualization.ps1 -Port 5174

# 运行全量 lint
.\scripts\start_visualization.ps1 -Mode lint
```

## 常见问题

### 页面打不开（最常见）

先跑状态，它会直接说清是哪一环坏了（端口 / 守护进程 / vite / 日志）：

```powershell
.\scripts\start_visualization.ps1 -Status
```

- 显示“HTTP 200 · 正常” → 打开 <http://localhost:5173> 即可，问题在浏览器缓存或网址；
- 显示“守护进程未运行、端口空着” → 重新执行一次启动任务（幂等，3 秒内回来）；
- 显示“端口被其它进程占用：xxx” → 那个进程不属于本项目，脚本不会擅自结束它，
  按提示关闭它或显式换端口；
- 仍然异常 → 看 `.run/devserver.log`，里面按时间戳记录了每次拉起 / 退出的原因。

如果终端被关掉、VSCode reload 之后页面就没了，说明当初是以前台方式起的；
现在默认脱管（`-Detach`），不会再随编辑器消失。若所在环境禁止后台进程，
改用前台任务 `AI2050: Frontend (foreground, Ctrl+C 停止)`。

### 端口 5173 被占用

客户端启用了严格端口模式，不会悄悄切换到其他端口（静默换端口会让书签指向空地址）。
如果占用者是本项目遗留的僵死 vite，启动器会**自动回收**；若是别的程序，则会如实报出
进程名与命令行，并要求你显式换端口：

```powershell
.\scripts\start_visualization.ps1 -Port 5174
```

### 找不到 Node.js

安装符合版本要求的 Node.js，或者设置 `AI2050_NODE_HOME`。启动器会输出实际使用的 Node 版本和目录。

### 前端正常但实时数据不可用

前端静态界面可以单独启动；实时模型分析和后端 API 需要另一个终端运行：

```powershell
.\.venv\Scripts\python.exe -m server.server
```

默认后端地址为 <http://localhost:5001>。

### 全量 lint 未通过

仓库仍有历史组件的 lint 技术债。生产构建成功与否应单独检查：

```powershell
.\scripts\start_visualization.ps1 -Mode build
```

不要把已有 lint 警告误判为 Vite 无法启动；新增或修改文件仍应避免增加 lint 错误。
