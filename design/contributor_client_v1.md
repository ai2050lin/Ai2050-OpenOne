# 贡献算力客户端 v1（contributor_client.v1）

状态：设计；P1（本机节点服务）与 P2（Electron 监护 + 贡献算力页）已实现，P3–P4 未开始。范围先只做 **Windows + NVIDIA**。

## 1. 目标与边界

让普通参与者在桌面客户端里一键贡献 GPU 算力：领取中心已冻结的测试任务（`census_shard` 普查分片、`family_test` 模式族分片，之后是交叉复算），在本机运行并上传可合并的统计结果。

不变的部分：

- 合并口径、校验、复算、金丝雀、证据等级全部沿用 dist_protocol v3 / census_job.v1；志愿者节点不需要额外信任。
- 管理操作（模板设计、冻结、审计、普查作业）只在中心同源的网页版研究工作台进行，贡献客户端不做。
- 只读数据包规则（AGENTS.md §10）不变：**界面只和本机节点服务通信**，只有节点服务访问中心，且只做 注册 / 领取 / 续租 / 上传 / 读取站点配置 五类调用，带退避。

浏览器网页版不能调用 GPU 跑 HF 模型并保持与原生 torch 相同的精度与可复现性，因此网页版只展示进度，并提供「下载桌面客户端」与命令行一键包。

## 2. 三层结构

```
桌面客户端（Electron 安装包）
  界面（现有 React）：网站 · 研究工作台 · 研发 → 贡献算力
        │ IPC（Electron 主进程转发）→ 127.0.0.1 本机节点服务
  Electron 主进程：启动 / 监护节点服务、读取本机令牌、升级检查
        │ spawn
  本机节点服务（deploy/node_service.py，Python，仅标准库）
    设备检测 · 资源门 · 模型核对 · 任务循环（复用 node_agent）· 日志 · 历史
        │ 只有：注册 · 领取 · 续租 · 上传 · 站点配置（退避 + 抖动）
  中心 server.ai2050.com
```

## 3. 交付形态

| 组件 | 体积 | 方式 |
| --- | --- | --- |
| Electron + 界面 + `node_agent.py` / `node_service.py` | ~150 MB | 安装包内（P2） |
| 计算组件：Python + torch(CUDA) + transformers | 2–3 GB | 首次「开始贡献」时下载（P3，`uv` + 锁定版本 + sha256，国内镜像可选） |
| 模型权重 | 每个 1–30 GB | 按任务需要下载，按 `model_snapshot` 核对中心要求的快照；先查磁盘 |

命令行一键包（`/api/platform/kit`）继续保留，适合 Linux 服务器；它也带上 `node_service.py`，`node_agent.py serve` 可直接启动本机服务。

## 4. 本机节点服务（P1）

启动：`python node_agent.py serve [--port 5030] [--synthetic]`（`--synthetic` 只用于沙盒中心联调，界面不能打开它）。

### 4.1 本机接口安全

- 只监听 `127.0.0.1`。
- 校验 `Host` 头必须是 `127.0.0.1:<port>` 或 `localhost:<port>`（防 DNS 重绑定）。
- 每次启动生成随机本机令牌，写入 `<home>/node_service/service.json`（端口、令牌、pid）；每个请求必须带 `X-Local-Token`。网页读不到这个文件，所以浏览器里的网站无法操控节点。
- 带 `Origin` 的请求默认拒绝，开发时用 `--allow-origin` 显式放行（如 `http://localhost:5173`）。
- 接口不执行任意命令；任务代码仍按任务包与作业规范的 sha256 核对后才运行。

### 4.2 接口

| 方法 路径 | 作用 |
| --- | --- |
| `GET /status` | 状态（stopped / running / paused / waiting / error）、当前任务、等待原因、计数、绑定信息 |
| `GET /device` | `nvidia-smi` 读 GPU（名称、显存、占用、驱动）；计算组件（torch / CUDA）检测；磁盘剩余；电源状态 |
| `GET /config`、`PUT /config` | 中心地址、任务类型、资源限制、模型目录；按白名单校验 |
| `POST /bind` | 用绑定码向中心注册节点（归属账号） |
| `GET /models` | 中心要求的模型（站点配置缓存）与本机目录核对状态 |
| `POST /models/check` | 为某个快照指定本机目录并在后台核对 sha256 |
| `POST /start`、`/pause`、`/resume`、`/stop` | 开始；暂停（当前任务做完后不再领取）；继续；停止（立即结束运行中的 runner，租约到期后中心重派） |
| `GET /logs?after=<seq>` | 日志环形缓冲（含 runner 进度） |
| `GET /history` | 本机完成记录（任务、结果 sha、中心校验结论） |

### 4.3 资源门（每次领取前检查，不满足则等待并给出原因）

| 门 | 缺省 | 说明 |
| --- | --- | --- |
| 时间窗 `hours` | 空（全天） | `"22:00-08:00"` 这类本地时间段，可跨午夜 |
| 用电池时暂停 `pause_on_battery` | 开 | Windows `GetSystemPowerStatus` |
| GPU 被其他程序占用时让出 `yield_gpu_util` | 30 % | 未运行任务时 `nvidia-smi` 利用率高于阈值则等待 |
| 显存上限 `max_vram_gib` | 0（不限） | 模型权重文件总大小 × 1.2 超过上限的模型不领取（近似估计） |
| 最少剩余磁盘 `min_free_disk_gib` | 5 | 工作目录所在盘 |

### 4.4 退避

- 没有任务：60 s 起，每次 ×2，上限 15 min，±20 % 抖动；领到任务后复位。
- 出错（网络、5xx、429）：30 s 起，每次 ×2，上限 30 min，±20 % 抖动。
- 站点配置最多每 30 min 读一次。

这样即使有大量节点，空闲时每个节点每 15 分钟只访问中心一次。

## 5. 界面与监护（P2）

### 5.1 Electron 主进程（`frontend/electron/nodeSupervisor.js`）

- 启动客户端时运行 `python node_agent.py serve --home <home> --port 0 --server-override <客户端连接的中心>`，按启动时间读 `<home>/service.json` 得到端口与令牌（venv 的 python.exe 在 Windows 上会再启动一个解释器，pid 不一定相同）。`<home>` 与命令行相同（`~/.ai2050/node_service`，`AI2050_NODE_HOME` 可改），所以两者共用节点凭据。
- 已有服务在运行（命令行启动）时直接接管连接，不重复启动；退出客户端时只关自己启动的服务（先 `/shutdown`，5 秒后强制结束进程树）。
- 意外退出时 2 s 起倍增、上限 60 s 重启；10 分钟内连续 5 次失败后停下，界面显示原因与输出尾部。
- Python 查找顺序：界面里选的 → `AI2050_PYTHON` → 客户端计算组件目录（P3）→ 开发仓库 `.venv` → PATH（排除 Windows 商店占位程序）。
- IPC：`ai2050:node-info / node-request / node-restart / node-pick-python` 与推送 `ai2050:node-info`。`node-request` 只转发白名单内的接口（`/shutdown` 不在其中），令牌不进页面。
- `--synthetic` 只能由环境变量 `AI2050_NODE_SYNTHETIC=1` 打开；`AI2050_CLIENT_SERVER` 指定沙盒中心且不写进用户配置。
- 安装包经 electron-builder `extraResources` 带上 `node_agent.py` 与 `node_service.py`（`resources/node/`）。

### 5.2 研发 → 贡献算力（`ContributePanel.jsx`）

所有形态都有这个工位：桌面客户端（有 `window.ai2050Desktop.node`）显示控制面板，网页版只显示下载入口。面板依次为：

- 本机节点服务：状态、Python、重启、选择 Python。
- 运行：开始 / 暂停 / 继续 / 停止、等待原因、当前任务进度、计数。
- 账号与节点：用登录会话向中心取一次性绑定码后注册，或用绑定码注册、匿名注册。
- 设备：GPU、计算组件、磁盘、电源。
- 模型：中心要求的快照、本机目录与核对。
- 资源限制、日志、本机完成记录。

界面每 2 秒读一次本机 `/status` 与新日志，不访问中心；中心只在「读取中心要求」「绑定」时由节点服务或账号接口访问。「试一句话」本机前向是后续可选项。

## 6. 安全与可信（P4）

- runner 随客户端版本内置；中心只下发任务规范与数据。必须由中心下发代码时使用 ed25519 签名，客户端只接受固定公钥签名。
- 结果可信仍靠中心校验 + 跨账号复算 + 金丝雀；按模型快照、精度与平台分桶，不跨桶平均。

## 7. 分期

| 期 | 内容 | 完成标准 |
| --- | --- | --- |
| P1 | 本机节点服务：接口、安全、资源门、退避、日志、历史；保留命令行 | 单元测试；沙盒中心 synthetic 全流程（领取 → 运行 → 上传 → 校验） |
| P2 | Electron 集成：主进程监护节点服务；贡献算力页 | 监护器（真实代码、纯 Node 宿主）+ 页面对沙盒中心走通：绑定账号 → 开始 → 4 个 synthetic 分片校验通过；资源门等待、暂停 / 停止、重启、进程被杀后自动恢复。安装包本身尚未在干净机器上构建验证 |
| P3 | 计算组件安装（uv、镜像）、模型下载与核对 | 干净 Windows 机器得到真实 GPU 结果 |
| P4 | 内置 runner + 签名、自动升级、贡献面板 | 篡改负对照被拒 |
