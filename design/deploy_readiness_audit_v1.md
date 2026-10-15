# 部署就绪度核查报告 — 服务端 CentOS + 客户端安装程序

> 日期：2026-10-14 23:30 · 核查方式：**实测**（非文档推断）
> 结论速览：
> - 服务端 → CentOS：**可以部署**，但**不能整仓直接搬**；有 1 项必须改（路径守卫已容错，可不改）、3 项需环境准备。
> - 客户端 → 安装程序：**可以打包**，但**当前零打包配置**，需新增 Electron/Tauri 壳 + 1 项必改（资源路径 `base`）。

---

## 0 实测证据摘要（本轮真实跑出来的数字）

| 核查项 | 方法 | 结果 |
|---|---|---|
| 服务端可导入性 | `spec.loader.exec_module(server.py)`，无模型模式 | ✅ **5.0 s** 完成，FastAPI app 构建成功 |
| 路由规模 | 展开 FastAPI 0.141 的 `_IncludedRouter.original_router` | ✅ **362 条**路由（atlas 12 / research-assets 23 / rdc-prefix 16 / runs 22 / news 1 / distributed 1） |
| 无 GPU / 无模型启动 | `AI2050_SKIP_MODEL_LOAD=1 CUDA_VISIBLE_DEVICES="" python server/server.py` | ✅ 监听 `0.0.0.0:5099` 成功 |
| 端点实测 | curl 逐个打关键端点 | ✅ **7/7 全 200**（下表） |
| module_space 数据 | `/api/ai-rnd/atlas/tree?schema=v2` | ✅ 29,964 B，`module_space` 34 节点 / 3 桥，`sha8=0d15645a` |
| 前端生产构建 | `npx vite build` | ✅ **5.57 s**，main.js 2.61 MB（gzip 764 KB） |
| Windows 专属调用 | grep `startfile/winreg/pywin32/[A-Z]:\` | ✅ 核心链路**零命中**（仅 3 处 `d:\develop\...` 字面量，见 §2.3） |
| npm registry | `npm ping` | ✅ PONG 995 ms（可安装打包器） |

### 端点实测明细（全部 HTTP 200）

```
200   62 B     /health
200   29,964 B /api/ai-rnd/atlas/tree?schema=v2
200   70,597 B /api/ai-rnd/atlas/registry
200   15,844 B /api/ai-rnd/atlas/gaps
200   1,589 B  /api/distributed/summary
200   4,251 B  /api/news
200   5,893 B  /api/templates
```

---

## 1 服务端：能否部署到 CentOS

### 1.1 判定：**可以**，且已具备生产部署要件

已具备：
- `server/server.py` 自带 `AI2050_HOST`（默认 **0.0.0.0**）与 `AI2050_PORT`（默认 5001）→ 天然对外可访问，无需改代码。
- 自带 **无模型逃生口** `AI2050_SKIP_MODEL_LOAD=1` → CentOS 上**不需要 GPU、不需要下模型权重**即可提供全部图谱/注册表/新闻/分布式接口（已实测）。
- `deploy/` 已有 `ai2050-distributed.service`（systemd）+ `README_CENTOS.md`（CentOS 7/8/9 步骤）作为**可复用范式**。

### 1.2 但**不能整仓直接搬**，缺 4 件东西

| # | 缺口 | 现状 | 影响 | 动作 |
|---|---|---|---|---|
| **G1** | **:5001 没有 systemd 单元** | `deploy/ai2050-distributed.service` 只管 `distributed_service.py`(:5010)，**不管** `server.py`(:5001)——而后者才是客户端真正的数据源 | 部署后需手工 `nohup`，无自动重启/日志管理 | 新增 `deploy/ai2050-server.service`（照抄现有单元结构） |
| **G2** | **无 requirements.txt** | 全仓 0 个依赖清单；实测本机 fastapi 0.141.1 / uvicorn 0.52.1 / pydantic 2.13.4 / numpy 2.5.1 / torch 2.13.0+cu130 / transformers 5.14.1 / scipy 1.18.0，`transformer-lens` **未装**（靠仓库 `transformer_lens/` 源码目录本地导入） | 服务器无法复现环境 | 新增 `requirements-server.txt`（**CPU 版 torch**，见 §1.4） |
| **G3** | **`.gitignore` 排除了 `deploy/`** | 指南已声明「deploy/ 不入 git，clone 后需 scp 补」 | 部署步骤多一次人工拷贝，易漏 | 保持现状（纪律正确），但部署 checklist 要显式写这一步 |
| **G4** | **无 nginx / 静态托管配置** | `frontend/dist` 是纯静态产物（3.8 MB），但**没有**任何 web server 配置 | 前端需另起服务，CORS/同源均未配 | 新增 `deploy/nginx-ai2050.conf`（见 §3.2） |

### 1.3 CentOS 上的最小可运行命令（已验证等价路径）

```bash
sudo useradd -r -s /sbin/nologin ai2050
sudo mkdir -p /opt/ai2050/Ai2050-OpenOne /var/lib/ai2050/distributed
sudo chown -R ai2050:ai2050 /opt/ai2050 /var/lib/ai2050

cd /opt/ai2050/Ai2050-OpenOne
python3.11 -m venv /opt/ai2050/venv
/opt/ai2050/venv/bin/pip install fastapi "uvicorn[standard]" pydantic numpy scipy safetensors transformers
# torch 走 CPU 版（服务器无 GPU 时）：
/opt/ai2050/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

# 关键环境变量：无模型模式（实测 5 s 起服务）
export AI2050_SKIP_MODEL_LOAD=1
export AI2050_HOST=0.0.0.0
export AI2050_PORT=5001
export AI2050_DIST_DIR=/var/lib/ai2050/distributed
/opt/ai2050/venv/bin/python server/server.py
```

### 1.4 依赖上的**重要警告**（易踩）

- **torch 必须换 CPU 版**：本机是 `2.13.0+cu130`，服务器若无 GPU 需装 `--index-url .../whl/cpu`，否则拉 2 GB+ CUDA 包且启动仍回落 CPU。
- **`transformer-lens` 不必 pip 装**：仓库自带 `transformer_lens/` 源码，`server.py` 从仓库根导入即可（本机实测未安装也能 import 成功）。但服务器 clone 后必须**保持在仓库根目录**启动，否则 `import transformer_lens` 失败。
- **磁盘**：不装模型权重则无大文件需求；`research/` + `tests/` 体积大但被 `.gitignore` 排除（`/tests/glm5/result/` 等），clone 后不含实验数据——**图谱类接口会返回空**。若需要 LIVE 图谱数据，须另行同步 `research/gpt5/atlas/`（含 `atlas_tree_v2.json`）与 `research/ledger`。

### 1.5 已知非阻塞瑕疵（Linux 上优雅降级，不会崩）

3 处 Windows 绝对路径字面量，**均带 `os.path.exists` 守卫**，Linux 上走降级分支返回空结构：

| 位置 | 路径 | 端点 | Linux 行为 |
|---|---|---|---|
| `server/server.py:928` | `d:\develop\TransformerLens-main\experiments\toy_experiment` | `/training/status` | 返回 `{"status":"no_training"}` |
| `server/server.py:2339` | 同上 + `training_log.json` | `/toy_experiment/metrics` | 返回空 `{Transformer:[],FiberNet:[]}` |
| `server/server.py:2352` | 同上 + `ricci_flow_data.json` | `/toy_experiment/ricci_metrics` | 返回 `[]` |

**判定：可接受，不阻塞部署**。若要修，改为 `Path(__file__).resolve().parents[1] / "experiments" / "toy_experiment"` 即可（约 3 行）。

---

## 2 客户端：能否打包为安装程序

### 2.1 判定：**可以，但当前配置量为零**

实测：`frontend/package.json` **无 electron / tauri / electron-builder / pkg / nexe 任何依赖**，无 `electron-builder` 配置段。`vite.config.js` 只有 dev server + build，**无 `base` 设置**。

即：**当前只能出「网页 + 本地起服务」形态，出不了安装程序。** 要出安装程序需新增壳工程（下面 §3.1 给出方案）。

### 2.2 两个必改项（不改会白做）

**M1 — 资源路径必须改相对（否则 Electron `file://` 加载必白屏）**

实测 `dist/index.html` 生成的是**绝对路径**：
```html
<script type="module" crossorigin src="/assets/main-DC5X0B6J.js"></script>
<link rel="stylesheet" crossorigin href="/assets/main-BtCe2_Cj.css">
```
`file://` 协议下 `/assets/...` 指向文件系统根目录 → 白屏。**修法**：`vite.config.js` 加 `base: './'`（一行）。

**M2 — 71 处 API 地址硬编码（决定安装包能不能连远端服务）**

实测分布（`frontend/src` 内 `localhost|127.0.0.1`）：

| 端口 | 出现次数 | 性质 |
|---|---|---|
| **5001** | **52** | 多数走 `import.meta.env.VITE_API_BASE`（可配置），但 `RdcBindingAtlas/RdcJointAtlas/RdcLawAtlas/RdcOperatorAtlas/RdcPrefixAtlas/RdcRelationStudy` 等 **6 个文件直接写死 `'http://127.0.0.1:5001/api/...'`** |
| 8000 | 8 | 训练服务 |
| 5004 | 8 | rdc-construction（走 env，可配） |
| 5173 | 1 | 前端自身 |
| 5003 / 5002 | 各 1 | rdc-query / rdc-update（走 env，可配） |

**风险**：装了 Electron 后 `127.0.0.1` 指向**用户本机**，不是服务器 → 52 处 5001 中至少 6 个文件的硬编码会直接连错对象。
**修法**：抽一个 `src/config.js` 统一出口（`export const API_BASE = window.__AI2050_API__ || import.meta.env.VITE_API_BASE || 'http://127.0.0.1:5001'`），安装包首启写 `window.__AI2050_API__`，用户填服务器地址即可。

### 2.3 好消息：打包体量友好

- `dist/` 仅 **3.8 MB**（main.js 2.61 MB + annotation.js 17 KB + css 283 KB + data/）。
- 静态资源无 CDN 依赖（3D 用自研 Canvas + three.js 已打进 bundle）。
- `electron` 壳体本身约 150–250 MB，最终安装包预计 **160–260 MB**，常规量级。

### 2.4 推荐方案：Electron（而非 Tauri）

| 维度 | Electron | Tauri |
|---|---|---|
| 前端兼容性 | ✅ 直接复用现有 vite build 产物，零改动 | 需 Rust 工具链 + 改配置 |
| 与 Python 后端协作 | ✅ 主进程 `spawn` 拉起 `python server/server.py`，天然解决「双击即用」 | 同样可以但更繁琐 |
| 安装包体积 | ~160–260 MB | ~10–30 MB |
| 本机工具链 | Node 22.22.2 已就绪，npm registry 可达（PONG 995ms） | 需装 Rust + WebView2 |

**选 Electron**：本项目是**科研工具**（体积不敏感、可靠性优先），且 Electron 能把 Python 后端一起打包，实现「安装即用、无需预装 Python」。Tauri 的体积优势对科研场景价值有限。

---

## 3 落地清单（按优先级；均**未执行**，等你确认）

### 3.1 服务端 → CentOS（P0，约 4 件）

| # | 动作 | 产出 | 风险 |
|---|---|---|---|
| S1 | 新增 `deploy/ai2050-server.service` | systemd 单元（:5001，`AI2050_SKIP_MODEL_LOAD=1`，`Restart=always`，`ReadWritePaths` 隔离数据目录） | 低（照抄现有单元） |
| S2 | 新增 `requirements-server.txt` | CPU 版 torch + fastapi/uvicorn/pydantic/numpy/scipy/safetensors/transformers 固定版本 | 低 |
| S3 | 新增 `deploy/nginx-ai2050.conf` | 静态托管 `frontend/dist` + `/api` 反代到 127.0.0.1:5001（**顺带解决同源，CORS 可收紧**） | 低 |
| S4 | 新增 `deploy/README_CENTOS_SERVER.md` | :5001 + :5173/dist + node_agent 三段完整步骤（现有 README_CENTOS.md 只覆盖 :5010） | 低 |

### 3.2 客户端 → 安装程序（P1/P2）

| # | 优先级 | 动作 | 说明 |
|---|---|---|---|
| C1 | **P0 必做** | `vite.config.js` 加 `base: './'` | 一行，不做则 Electron 必白屏 |
| C2 | **P0 必做** | 抽 `src/config.js` 统一 API 基址出口，替换 6 个硬编码文件 | 决定安装包连远端还是本机 |
| C3 | P1 | 新增 `frontend/electron/main.js` + `preload.js` + `package.json` 段 | 主进程拉起 Python 后端、开 BrowserWindow、加载 dist |
| C4 | P1 | 装 `electron` + `electron-builder`，配 `win.target=nsis` | 产出 `.exe` 安装程序 |
| C5 | P2 | 首启「服务器地址」配置对话框，写入 `window.__AI2050_API__` | 提升多机可用性 |

### 3.3 可选修复（P2，不阻塞）

- `server/server.py` 3 处 `d:\develop\...` → 改为仓库相对路径（约 3 行，Linux 上目前只是返回空，不崩）。

---

## 4 结论（严格口径）

1. **服务端可以部署到 CentOS** —— 已实测「无 GPU + 无模型」5 秒起服务、7/7 端点 200、362 路由齐备。**但**目前只做到「能跑」，离「一键部署」还差 4 件产物（systemd 单元 / 依赖清单 / nginx 配置 / :5001 部署指南）。
2. **客户端可以打包为安装程序** —— 但**当前零打包配置**，且有 **2 个必改项**（`base: './'` 与 71 处 API 地址）。改完这两项，Electron 路线技术风险低、工具链齐备（Node 22.22.2 + npm 可达）。
3. **两个「不能」要说清**：
   - 不能整仓 `git clone` 后直接跑 —— `deploy/` 被 gitignore，必须 scp 补；`transformer_lens` 靠源码目录，**必须在仓库根启动**。
   - 不能指望 CentOS 上有图谱数据 —— 实验数据（`tests/*/result/`、`research/` 大部）被 gitignore 排除，须单独同步，否则图谱接口返回空（这是设计好的诚实空态，不是故障）。

---

## 附：本轮实测脚本（可复跑）

- `tests/gpt5_temp/_deploy_probe.py` —— 无模型导入探测
- `tests/gpt5_temp/_server_routes3.py` —— 362 路由展开盘点（注意 FastAPI 0.141 的 `_IncludedRouter` 需读 `original_router`）
- 端点验证：`AI2050_SKIP_MODEL_LOAD=1 AI2050_PORT=5099 python server/server.py` + curl 7 连
