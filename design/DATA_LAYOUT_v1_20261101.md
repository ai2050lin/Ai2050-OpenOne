# 本地数据布局整理方案 v1 [2026-11-01 10:00]

## 0 结论

- **唯一本地数据根：`D:\ai2050_native`**（不新建目录）。它是项目 20+ 脚本 `AI2050_NATIVE_DIR` 的缺省值，结构已规范，对应服务器 `/var/lib/ai2050/native`，改动成本为零。
- **服务器为权威的层**：分片结果（objects）、任务/模板库（sqlite）、节点任务产物 —— 本地全部可删。
- **服务器上没有、也不能删的层**：原生全场（逐层 float16 激活）。协议明确不上传（`shared/dist_protocol/runner_v3.py:36`），本地是唯一副本。
- 按本方案执行，C 盘可释放约 **48G**、D 盘约 **340G**（D 盘当前仅剩 9G）。

## 1 三条硬约束（方案的依据）

| 约束 | 证据 | 含义 |
|---|---|---|
| 原生全场**不上传** | `runner_v3.py:36` 注释「原生全场（不上传）」；`:455`「原生全场只存于节点本地（不上传）；按 manifest 的 sha256 校验后可按需拉取或用同一语料与模型重算」 | `AI2050_NATIVE_DIR` 下的数据**在服务器上没有副本**，删除即丢失，重算需 GPU 小时级 |
| 分片结果**上传中心** | `node_service/history.jsonl` 每条任务记录 `status:done` + `sha`；中心 `objects/` 内容寻址 | 节点任务产物（items/suffstats npz）已在服务器，本地可删 |
| 服务器已同步 | `GET https://server.ai2050.com/api/catalog` 返回 12 个 items，与本地 `distributed_data/snapshots/catalog.json` **12/12 完全一致**（0 缺 0 多）；`/api/health` → `data_dir_in_release:false`（正规部署） | 本地 `distributed_data` 是服务器的一份过期镜像，不是权威 |

## 2 三层归属模型

| 层 | 定义 | 内容 | 服务器有副本 | 处置 |
|---|---|---|---|---|
| **L1 服务器权威** | 已上传、可重新下载 | `distributed_data/`（14G）、`~/.ai2050_agent/tasks/*/*.npz`（13.3G）、`~/.ai2050/node_service/work/`（11G）、`C:\ai2050_sim`（6.8G） | 是 | **本地删除** |
| **L2 本地唯一** | 协议不上传，重算昂贵 | `D:\ai2050_native`（69G）、`~/.ai2050/native_fields` 中 D 侧缺失的部分（7.2G） | 否 | **合并进 `D:\ai2050_native` 后统一保留** |
| **L3 结论与代码** | 论文级证据链 | `research/`（933M）、`AGENTS.md`、`scripts/`、`shared/`、`deploy/`、`.workbuddy/memory/` | 否 | **全量保留** |

## 3 目标布局

```
D:\ai2050_native\                     # 唯一本地数据根（AI2050_NATIVE_DIR）
  ├─ _census\         2.2G  普查语料与 38 层 MLP 统计（现役）
  ├─ _transcoder\      11G  transcoder 采集/训练（现役）
  ├─ _entity_atlas\   133M  实体图谱（今天仍在写）
  ├─ _weights\        4.0G  权重切片（o_proj 等）
  ├─ ENT-XF\           31G
  ├─ GR-Q\             13G
  ├─ GR-NEG\          2.8G
  ├─ KN-ISA\          1.8G
  ├─ KN-ATTR\         748M
  ├─ GR-ROLE\          迁移自 native_fields（独有 118M）
  └─ _behavior\        新建：从旧节点 tasks 抽出的 behavior.json（约 12.3M）

D:\AI2050\Ai2050-OpenOne\             # 代码仓，无大数据
  ├─ research\        933M  结论级 json + MEMO（必须留）
  └─ tests\          见 §5 批 4（glm5/result 326G 是主要清理对象）

（删除）distributed_data\、C:\ai2050_sim\、~\.ai2050\native_fields\、~\.ai2050_agent\tasks\*npz
```

环境变量固化（一次性）：

```powershell
setx AI2050_NATIVE_DIR "D:\ai2050_native"
setx AI2050_DIST_DIR   "D:\AI2050\Ai2050-OpenOne\distributed_data"   # 仅本地调试用；正式走 server.ai2050.com
setx AI2050_OFFLOAD_DIR "D:\ai2050_native\_offload"
```

## 4 清理清单

### 批 1｜零风险（纯重复 / 零引用）— 共 19.5G

| 路径 | 体积 | 依据 |
|---|---|---|
| `D:\ai2050_native\_checks\_weights` | 3.3G | 与 `_weights` 两组各 40 个文件**大小完全一致**；`scripts/` `server/` 检索无路径引用 |
| `D:\ai2050_runtime_p3test` | 3.1G | uv 独立运行时（py3.14.6/torch2.13+cu130）；全项目（含 tests）检索 `runtime_p3test` **零命中** |
| `C:\ai2050_sim` | 6.8G | 本地模拟中心快照：72 个 objects 全部在 `distributed_data/objects` 有同名同大小副本；72 个 task_id 全部在本地主库；无环境变量/代码指向 |
| `~\.ai2050\native_fields\ENT-XF\TG-ENT-XF-v2` | 6.3G | 与 `D:\ai2050_native\ENT-XF\TG-ENT-XF-v2\64925…\6be…` **300 文件逐一比对 0 独有** |

### 批 2｜低风险（可重建 / 已上传）— 共 29.8G

| 路径 | 体积 | 依据 | 代价 |
|---|---|---|---|
| `~\.ai2050_agent\tasks\*\outputs\*.npz` | 13.3G | 13.33G 全为 items/suffstats 分片；结论级 `behavior.json` 仅 12.3M | 无（**必先抽出 json**，见批 2 前置） |
| `~\.ai2050\node_service\work\tasks` | 11G | 172 个已完成任务，`history.jsonl` 全部 `done`+sha 已上传 | 重跑需重新下载任务包 |
| `D:\ai2050_offload` | 5.2G | accelerate 卸载缓存，原始权重 `models/hf/Qwen3-14B`（62G，8 shard）完整 | 下次 14b 低显存干预时重新生成（一次模型加载） |
| `D:\AI2050\Ai2050-OpenOne\distributed_data` | 14G | 服务器 catalog 12/12 一致；`data_dir_in_release:false` 说明服务器数据在程序外独立存放 | 本地中心调服务时需重新 export |

**批 2 前置（不可跳过）**：`make_ent_lexicon_v2.py`、`estimate_ent_gate.py` 直读 `~/.ai2050_agent/tasks/*/outputs/behavior.json`，必须先把 209 个 behavior.json（12.3M）抽到 `D:\ai2050_native\_behavior\` 并改这两个脚本的默认路径，再删 npz。

### 批 3｜大件：历史运行产物 — 315G（**需你确认**）

| 路径 | 体积 | 事实 |
|---|---|---|
| `tests\glm5\result\**\*.npz` | 258.4G / 96,747 个 | 时间分布：**97,322 个落在 2026-09**，10 月仅 168 个，8 月 206 个 ⇒ 9 月批量采集的一次性产物 |
| `tests\glm5\result\**\*.npy` | 56.3G / 949 个 | 同上 |
| `tests\glm5\result\**\*.json` | 9.1G / 175,567 个 | **不删**。<100KB 的 172,882 个是结论/verdict/prereg，属证据链 |

⚠️ `tests/glm5/` 本身是 **G 线现役脚本主目录**（137 个 10 月修改的 `.py`，含 `phase3280_g5h43_consumer_attribution.py`、`phase3281_g5h44_consumer_attribution_norm.py`）。**只能删 `result/` 下的 npz/npy，绝不能碰同级 `.py`。**

### 批 4｜需你决策

| 路径 | 体积 | 问题 |
|---|---|---|
| `~\.ai2050\native_fields\ENT-XF\TG-ENT-XF-v1` | 6.3G | D 侧不存在；`research/family_runs/entangle/` 两条 v1 结果 json 引用过它。v2 已取代 v1 |
| `~\.ai2050\native_fields\ENT-XF\TG-ENT-XF-P-v1/-v2` | 858M | D 侧只有 P-v3/v4/v5 |
| `~\.ai2050\node_service`（保留服务本体，仅考虑 work） | 11G | 节点服务仍在配置中（`server.ai2050.com`），当前无进程 |

## 5 执行顺序（**先腾后迁**，重要）

D 盘当前仅剩 9G，而 §2/§4 有 7.2G 需要迁入，必须先释放：

1. 批 1（D 盘 +3.3G +3.1G → 剩 ~15G）
2. 迁移：`~\.ai2050\native_fields\ENT-XF\TG-ENT-XF-v1`、`P-v1`、`P-v2` → `D:\ai2050_native\ENT-XF\`（+7.2G → 剩 ~8G）
3. 批 1 的 C 盘项 + 批 2 前置（抽 behavior.json）
4. 批 2（D 盘 +19.5G → 剩 ~27G）
5. 批 3（需确认；D 盘 +315G → 剩 ~340G）

## 6 验收标准

- `https://server.ai2050.com/api/health` 正常，`/api/catalog` 仍返回 12 items
- `python -c "import os;print(os.environ.get('AI2050_NATIVE_DIR'))"` 指向 `D:\ai2050_native`
- 抽查 3 个脚本的 `NATIVE` 解析路径存在：`run_mlp_census.py`、`entity_atlas_run.py`、`run_layer_panorama.py`
- `research/` 933M 完整，`.workbuddy/memory/` 完整
- 各研究线最近 phase（3280/3281）的 res json 与 MEMO 可读

## 7 风险与未决

1. **原生全场无服务器副本**：`D:\ai2050_native` 69G 删了就是丢了，只能 GPU 重算。这是本方案唯一的硬底线。
2. **批 3 不可逆**：326G 里 9 月 npz 若将来要复算某个 phase 的二次分析，需重跑采集。建议先导出「phase → 文件」清单存档，再删。
3. **两个 GPU 任务正在跑**（`entity_atlas_v4.py` PID 50416、`pat_q8b.py` PID 44752），它们在写 `_entity_atlas`。D 盘操作应等其结束。
4. `_checks/_weights` 虽无代码引用，但来源未查明（疑为某次校验脚本的临时副本）；建议先改名观察 3 天再删。
