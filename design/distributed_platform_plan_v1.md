# 分布式研发平台优化方案 v1（提案）

> 定位：本文件是**设计提案**，不是正式事实源。总体架构唯一权威 = `ai2050_research_os/README.md`。
> 本方案 = 其「十六、实施路线」阶段四（分布式执行）+ 阶段五（多人协作）的落地设计，
> 全程遵守「十三、分布式执行边界」与「十七、开发禁止项」（尤其 #9：单机闭环未稳定前不得优先开发分布式调度）。
> 状态：待用户裁决。裁决后按治理规则同步更新 research_os README。

## 一、目标重述

平台三层角色（对应总架构图）：

| 角色 | 想做什么 |
| --- | --- |
| 中心节点（服务器） | 维护模板任务库、节点目录、结果库；核验与聚合；对外公开进度/新闻/下载 |
| Worker 节点（贡献者） | 连接服务器领取逆向分析任务 → 本地 AI 执行测试 → 上传结果；全程本地 AI 配置不出机 |
| 访客 | 免登录查看整体进度、行业动态、下载他人结果 |

最终科学产出：综合大量模板结果，分析 LLM 语言编码机制（语言/风格/逻辑/位置条件效应图 → 机制地图）。

## 二、现状 → 缺口映射（改动落在真实资产上）

| 用户愿景 | 已有资产（直接复用） | 缺口（本方案要补） |
| --- | --- | --- |
| 查看整体进度 | 融合页路线透镜三 tab（业界新闻/平台进度/当前机器进度）、静态官网 fusion-theme | `/api/distributed/summary` 未实现；前端三 tab 全是 demo 数据；官网无进度页 |
| 领取逆向分析任务 | `phase_queue`、`metric_dict`（口径版本化）、`contracts/*.json`（13 份冻结合同）、Q06 预注册纪律 | 模板包规范、任务队列/调度/租约 API |
| 本地 AI 完成测试 | `server/ai_rnd_service.py`（多模型配置+五证据门+SSE，1575 行）、`tests/deepseek` 采集框架、63 条装置纪律 | 节点 Agent（runner）、任务收件箱 UI |
| 上传到服务器 | `collect.npz` 指纹纪律、execution.json drift 断言 | 上传协议、内容寻址存储、哈希核验、去重 |
| 其他人下载 | `atlas_ledger`（n=306）、snapshots 机制 | 公共下载 API、下载中心 UI |
| 综合分析编码机制 | Q05 多条件分解方案（η²/cos 管线已设计）、D4 桥口径可比方法 | 聚合管线 AGG（跨节点分桶汇总） |
| 行业动态 | 路线透镜 7 条静态新闻 | 新闻聚合源（RSS/arXiv/官方博客） |

## 三、服务器部分（7 个工作包）

### S1 单机闭环收口（前置红线，对应阶段一/三）
- `researchctl validate` 全绿；`snapshots/current/snapshot.json` 确定性可重建；
- 融合页/客户端当前状态改走 `useResearchSnapshot()`，清掉 JSX 硬编码 Phase 与硬编码服务地址（禁止项 #1/#6）；
- **此步不过，不开任何分布式开发**（禁止项 #9）。

### S2 模板包规范 Template Bundle
- 字段：`tm_id / version / design_sha（预注册冻结）/ metric_dict 版本 pin / 模型 revision pin / 输入语料清单 / 期望产物 schema / 许可`；
- 来源：把现有 `contracts/` 13 份冻结合同 + Q06 预注册设计直接升级为 TM-01…TM-12（多条件分解网格：语言最小对、风格最小对、逻辑连接词、距离扫描、实体置换、纯位置响应…）；
- 模板包 = 可分发 zip：语料 + 执行脚本 + 产物 schema + 契约 JSON。

### S3 节点目录与调度 API（对应 13.1/13.2）
- `POST /api/nodes/register`：登记 GPU/RAM/已有模型/dtype/存储/可信等级 → 发 `node_token`；
- `POST /api/tasks/claim`：按能力匹配领取模板任务（租约制，领走即锁定）；
- `PUT /api/tasks/{id}/heartbeat`：心跳续租；超时自动回收重派（13.1 超时恢复）。

### S4 结果库（内容寻址）
- `POST /api/results/upload`：manifest + 每文件 sha256 + 分块续传；服务端哈希核验、重复去重；
- 按 13.3 纪律：**敏感原始数据只留本地**，上传物 = 统计量（η²/cos/读出分数）+ 产物引用 + execution 证明，不收大张量；
- 结果必须附 `execution.json`（design_sha 与模板一致，否则拒收——预注册纪律上服务器）。

### S5 公开只读 API（访客免登录）
- `GET /api/distributed/summary`：在线节点/模板进度/累计 cells/下载计数/聚合版本；
- `GET /api/templates`、`GET /api/results/{sha}`、`GET /api/news`。

### S6 聚合管线 AGG v1
- 分桶：模板 × 模型 × 种子 × 环境（**禁止跨桶简单平均**，13.1 红线）；
- 桶内汇总 → η² 条件效应热图 → 方向 cos 矩阵 → 条件维度机制地图（复用 Q05 五步管线）；
- 产出版本化 AGG 报告（AGG-vN），口径跨版本可比走 D4 桥方法。

### S7 信任与纪律
- 不可信节点抽样复算（13.3）；节点最小权限；node_token 与 AI API key 严格分离（**API key 永不出本机**）；
- sealed 结果自动登记 `atlas_ledger`；新闻源：arXiv cs.CL/官方博客 RSS 定时抓取 + 人工审核位。

## 四、客户端部分（6 个工作包）

### C1 节点 Agent（核心新增）
- 常驻服务（复用 `ai_rnd_service` 会话骨架）：`claim → 下载模板包 → 调用本地执行框架（tests/deepseek 采集脚本骨架）→ 守卫 → 上传 → 汇报`；
- 支持「服务器任务」与「本地研究目标」两种驱动（现有 ai-rnd 循环保留）。

### C2 融合页去 demo 接线
- 路线透镜三 tab 接真实端点：平台进度←summary、业界新闻←/api/news、当前机器进度←本机 Agent 状态（GPU/当前任务/进度条/上传数）；
- 当前 Phase/队列状态来自 snapshot（禁止项 #1）。

### C3 任务收件箱
- 研发透镜左栏新增「服务器任务」区：浏览模板 → 领取 → 执行进度直读；与本地 AI 配置（主模型+分析模型）打通。

### C4 下载中心
- 按模板/模型/指纹检索他人结果 → 一键导入本地分析（对象卡 F#3734 接真实结果索引）。

### C5 执行守卫产品化（63 坑纪律内置）
- identity 对照臂逐位恒等断言、显存守卫（Qwen3/GLM4/DS7B 串行）、drift 断言、SMOKE 必做必看——不满足自动不上传。

### C6 断点续跑与离线队列
- 领取的任务本地持久化、checkpoint 恢复；断网缓存上传队列，恢复后补传（对应阶段四租约/检查点/恢复）。

## 五、关键接口草案

| 端点 | 方法 | 要点 |
| --- | --- | --- |
| `/api/nodes/register` | POST | 能力目录 → node_token |
| `/api/tasks/claim` | POST | 能力匹配 + 租约 |
| `/api/tasks/{id}/heartbeat` | PUT | 续租；超时回收 |
| `/api/templates` | GET/POST | 列表 / 注册（POST 需审核） |
| `/api/templates/{id}/bundle` | GET | 模板包下载 |
| `/api/results/upload` | POST | manifest+sha256+分块 |
| `/api/results/{sha}` | GET | 公开下载 |
| `/api/distributed/summary` | GET | 进度总览（公开） |
| `/api/news` | GET | 聚合新闻（公开） |

## 六、分期路线（对齐阶段一~五）

| 期 | 内容 | 验收 |
| --- | --- | --- |
| **M0** 单机闭环收口（前置） | S1 + 禁止项 #1/#6 清理 | snapshot 可重建、前端硬编码=0、validate 全绿 |
| **M1** 分布式最小闭环 | S2/S3/S4(简)/S5 + C1(CLI 版) + C2(接 summary) | 1 台服务器 + 2 台节点端到端跑通 TM-01，结果可下载 |
| **M2** 平台化 | S6/S7 全量 + C3/C4/C5/C6 + 新闻源 | 聚合 v1 出第一张跨节点 η² 热图；抽样复算上线 |
| **M3** 协作与公开 | 阶段五（角色/贡献记录/复现包）+ 官网公开进度页 | 访客免登录看进度；贡献可归属 |

## 七、风险与红线

1. 结果可信度 → sha256 核验 + execution 证明 + 不可信节点抽样复算；
2. 口径漂移 → metric_dict 版本 pin，跨版本不可比（D4 桥是唯一合法桥）；
3. 隐私 → AI API key 只在本机，服务器只见 node_token；敏感原始数据不上传；
4. 复现性 → 模板强制模型 revision pin（避免"同名模型不同权重"）；
5. 单点 → 先单中心，接口按多中心 federation 预留（幂等 + 内容寻址天然友好）；
6. 治理 → 裁决后同步 research_os README，不建平行事实源。
