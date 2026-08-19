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

### 端口 5173 被占用

客户端启用了严格端口模式，不会悄悄切换到其他端口。检查占用：

```powershell
Get-NetTCPConnection -LocalPort 5173 -State Listen
```

确认旧进程可以停止后再结束对应 PID，或者使用：

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
