# 科研测试登记、查重、归档与清理

接口随中心程序升级包交用户手动部署，不会运行模型，也不自动公开原始题目或数据。现有库不会自动转换成“已经查全”的档案：尚未回填的历史测试仍需检索本地备忘录和结果。接口的技术核验不等于科学审核，科学状态始终从 `unreviewed` 起步。

## 升级与验收

将本次发布包、同名 `.sha256` 和 `upgrade_center.sh` 上传到服务器同一目录，按既有服务名运行：

```sh
sudo bash upgrade_center.sh ai2050-center-1.0.42.tar.gz
curl -fsS https://server.ai2050.com/api/health
curl -fsS https://server.ai2050.com/api/test-archive/capabilities
```

capabilities 应返回 `research_test_archive_api.v1`、`private: true`、`runs_models: false`。部署保留当前 `AI2050_DIST_DIR`；私有档案位于该目录的 `research_library/test_archive/`，包含 SQLite、不可变对象和续传暂存。必须将这个子目录纳入服务器备份。该目录不由公开研究库索引，不可把它挂成 nginx 静态目录。

除 capabilities 外，所有接口均要求现有管理员会话或 `X-Admin-Token`。客户端凭据从 `AI2050_RESEARCH_SESSION`、`AI2050_RESEARCH_ADMIN_TOKEN` / `AI2050_DIST_ADMIN_TOKEN`，或 `--token-file` 读取；不要把令牌写进 URL、脚本、包或终端参数。使用 HTTPS，只有本机 loopback 允许 HTTP；客户端拒绝重定向，防止凭据被转发到其他站点。

日常测试通过归档协议完成上传。桌面客户端读取服务器记录。桌面登录授权仅对客户端会话生效，命令行使用受控环境注入的管理员凭据，或从仅授权账户可读的管理令牌文件取得凭据。使用 `deploy/test_archive_client.py` 时把 `--token-file` 放在 `query`、`check`、`start` 或 `submit` 之前。可先用只读的 `query` 验证命令行权限。

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 deploy/test_archive_client.py --token-file $archiveTokenPath query --q $archiveRunId
```

## 接口

| 接口 | 用途 |
| --- | --- |
| `POST /api/test-archive/preflight` | 冻结方案并查询精确/相似测试，保存查重回执 |
| `GET /api/test-archive/runs?q=&domain=&model=&state=&family=&limit=&offset=` | 按关键词、大类、模型、状态、模式族查询 |
| `GET /api/test-archive/runs/{run_id}` | 获取目标、原理、方案、状态、清单与结果摘要 |
| `POST /api/test-archive/runs` | 用 preflight_id 登记运行及复用/新增理由 |
| `POST /api/test-archive/uploads` | 用 SHA256、size 初始化或恢复产物上传 |
| `GET /api/test-archive/uploads/{id}` | 获取持久化上传偏移 |
| `PUT /api/test-archive/uploads/{id}?offset=N` | 提交不超过 2 MiB 的二进制分块 |
| `POST /api/test-archive/uploads/{id}/finish` | 校验全文件 SHA256、原子存入私有 CAS；可重放 |
| `POST /api/test-archive/runs/{run_id}/finalize` | 保存 completed/failed/interrupted、摘要和产物清单 |
| `GET /api/test-archive/runs/{run_id}/receipt?verify=true` | 重新读取服务端文件核对哈希，返回可用性与清理门 |
| `GET/POST /api/test-archive/runs/{run_id}/revisions` | 查产物恢复历史，或用当前父清单哈希追加恢复版本 |
| `GET /api/test-archive/objects/{sha256}` | 认证下载原始文件 |

请求元数据上限 512 KiB。默认单对象上限 64 GiB，可通过 `AI2050_TEST_ARCHIVE_MAX_OBJECT_BYTES` 调整；对象通过分块上传，不需要提高 nginx 的单请求上限。上传前对并发尚未写完的总量预留空间，继续写入时重查。未完成上传超过 24 小时后在下一次初始化时回收；已完成对象不随暂存过期删除。磁盘保留线沿用 storage_policy；空间不足返回 507。大文件核验可能较慢，客户端超时按实际带宽与磁盘能力评估；校验打印进度，不运行 GPU。归档库写事务串行，忙时返回 503，可安全重试。

## 从模型分析检索与回放

在 3D 空间选中 Head、MLP 单元或 HiddenState 坐标，可直接查询相关测试。中心运行查询在既有参数外支持 `snapshot`（完整模型快照）、`target`（冻结方案的 `acquisition.capture.targets` 中明确登记的规范地址）、`method`（精确方法版本）、`archive_state`、`artifact_role`（有已上传且已登记大小的产物）、`decision`。地址格式为 `B26/H36`、`B23/U7708` 或 `L27/D8`，块和层沿用客户端编号。过滤条件按交集，响应的 `applied_filters` 可用于确认中心版本确实执行了所选筛选。旧记录没有明确地址时，使用关键词入口查阅其元数据。

带有原生轨迹的逐题结果可以在 `case_results.jsonl` 的对应行嵌入显式 `native_trace` 引用：

```json
{
  "schema": "native_trace_ref.v1",
  "run_id": "中心归档运行 ID",
  "case_id": "本行 case_id",
  "model_snapshot": "64 位 SHA-256",
  "trace_id": "64 位轨迹 ID",
  "trace_run_id": "轨迹清单的 run_id",
  "sample_id": "轨迹清单的 sample_id",
  "position": 120,
  "layer": 27,
  "module": "attn",
  "index": 36
}
```

位置 `position` 是输入或当前生成轨迹的绝对 token 索引。层 `layer=0` 对应嵌入边界，只支持 `hs`。块 B26 的 `layer=27`，模块使用 `attn`、`mlp` 或 `hs`。客户端先核对逐题文件长度和 SHA-256，再核对轨迹清单的真实类型、模型、运行、样本与位置。原生轨迹须已在可连接的节点上登记。仅有逐题统计或摘要时，保留对应结果和缺测原因。筛选到原生场文件不能单独确定某一题可回放。

## 先查重，再开始

`research_test_plan.v1` 必须包含：

- `kind: real`；title、objective、principle、hypothesis、alternatives、criteria。
- `domains: [KN/RS/GR]`、稳定 `family_ids`。
- `dataset`：id、revision、content_sha256、split_manifest_sha256、case_count、exposure。
- `acquisition`：model_id、完整 model_snapshot、tokenizer_sha256、实际 inputs_sha256、code_sha256、environment_sha256、precision、seed、generation，以及 capture 的 layers/readouts/boundary。
- `analysis`：method_version、code_sha256、scoring_sha256、selection（冻结范围/划分/暴露等完整配置）、required_artifact_roles。

不能只用题目名称或模型简称判断重复。采集键对完整 acquisition 取规范化哈希；分析键对采集键、analysis、材料版本与判定标准取哈希。只改统计或评分不会改变采集键。模型编辑、数值环境、随机种子、生成配置和读出边界影响真实采集时必须体现在 acquisition 中；分析输入文件哈希和选择范围放进 analysis，不得隐去这些差别。

```sh
python deploy/test_archive_client.py check --plan plan.json --output preflight.json
python deploy/test_archive_client.py query --domain KN --q 来源
python deploy/test_archive_client.py start --preflight preflight.json \
  --run-id source-20261119-01 --decision new_acquisition --reason '新增未见句式，旧材料已暴露'
```

查重返回 exact analysis、same acquisition、shared family/domain 的候选和匹配理由。相似候选只是元数据检索，不是语义模型判断，也不能单凭“同大类”拒绝新实验。查询覆盖仅是 registered_records_only。运行登记在同一事务内重新查重，防止两个代理同时看到“没有结果”后重复采集。

`decision` 可取 `new_acquisition`、`analysis_only`、`independent_replication`、`recapture`。已有可用结果/采集或正在进行的同采集会阻止普通 new_acquisition；新增复现或重采须显式给原因。analysis_only 还必须给 source_run_id，且原采集键一致、需要的原始产物可用。开封/暴露范围不会随分析改变而恢复。

## 完成后提交

在数据目录中创建显式清单 `files.json`，不自动扫描整个盘。每一项都写清用途与保留级别：

```json
{
  "schema": "research_test_files.v1",
  "files": [
    {"path":"test.py","role":"script","retention":"core"},
    {"path":"questions.jsonl","role":"questions","retention":"core"},
    {"path":"protocol.json","role":"protocol","retention":"core"},
    {"path":"case_results.jsonl","role":"case_results","retention":"core"},
    {"path":"conclusion.json","role":"results","retention":"core"},
    {"path":"output.log","role":"log","retention":"core"},
    {"path":"reproducibility.json","role":"reproducibility","retention":"core"},
    {"path":"native_field.bin","role":"native_field","retention":"important"},
    {"path":"projection.tmp","role":"derived","retention":"scratch","in_use":false,"rebuild":"从已归档原场按analysis-v1重新派生"}
  ]
}
```

题目和每题结果须保留 case_id、family_id、实际输入、期望集、模型实际输出、评分/错误及 run_id；不得只交总分。中断时保存已完成部分，未执行部分明确标记，不补造模型输出。原生全场须记录 dtype、shape、层、读出与生成步等重建信息；将解码说明作为协议或复核文件上传。文件存在且校验通过只是字节完整性检查；是否覆盖冻结题目、身份是否准确、指标和因果证据是否充分仍由实验审计判断。

```sh
python deploy/test_archive_client.py submit --run-id source-20261119-01 \
  --manifest /数据目录/files.json --status completed --summary '实际结果、反例与适用范围' --clean-scratch
```

失败用 failed，中止用 interrupted。核心产物缺失时仍可登记，但回执为 missing_artifacts，清理不放行。重要产物未上传会拒绝 finalize。最终清单不可覆盖；缺件后来找回时，submit 额外传 `--parent-manifest-sha <当前清单哈希>`，服务端追加恢复版本并保留旧清单和回执，执行状态不变，不把恢复文件算成新运行。需要新的分析结果时建立明确的新分析运行引用原采集。连接失败时本地文件留存，用同一清单续传，不重跑模型。普通网络错误、429 与 5xx 有有限重试；4xx 必须修正原因。

补传运行器留下的结果时，从 `review.json` 确认 `server_run_id` 和本轮工作目录，使用其中的 `archive-files.json`。向 `submit` 传原运行 ID、实际结束状态、原摘要和同一清单，协议会复用已上传的文件或从服务端已有偏移续传。检查清单目录里的 `server-receipt.json`，核对运行 ID、最终清单哈希、归档状态和 `readback_verified`。完整复核包的归档状态为 `server_verified`。

`--clean-scratch` 只逐个删除清单中 retention=scratch、in_use=false、有重建依据的文件，先写 cleanup-record.json；不会递归删除目录，不删除未列出的文件，也不删除保留文件的本机副本。只有刚读回的有效服务器回执允许清理；上传路径越界、符号链接越界、保留文件哈希不匹配、缺件、损坏或在用都会停止清理。原场退役须另行核对复用与依赖，并用服务端 manifest 的 retired_not_archived + reason 登记，不能把未上传原场写成远程可用。

## 日常测试入口

```sh
python tests/glm5/run_routine_test.py --script tests/glm5/your_test.py \
  --questions /题目路径/questions.jsonl --peak-gib 8 \
  --archive-plan /方案路径/plan.json --archive-reason '本轮新增价值' -- 参数
```

除前述字段，方案增加 entrypoint_sha256、dataset.questions_file_sha256、execution_args；同时在 acquisition 中加入实际 entrypoint_sha256、questions_file_sha256，在 acquisition.generation.script_args 中记录完整脚本参数，确保真实文件或参数改变也改变采集键。acquisition.environment_sha256 应等于 `shared.dist_protocol.test_archive.digest(deploy.test_archive_client.runtime_identity())`，入口会核对当前 Python/平台/安装依赖版本的指纹，并保存原始运行环境。CLI 的模型快照、tokenizer 与种子如另行指定必须与冻结方案一致。测试脚本将产物写入 AI2050_TEST_WORK_DIR，必须输出 conclusion.json、case_results.jsonl；额外产物和临时缓存以 artifacts.json（research_test_files.v1）列明。入口在启动子进程前查询并登记，在结束后提交完整日志、输出和额外重要产物；归档失败或不完整均保留本地文件、返回非零，不再直接递归删除整个临时目录。该入口拒绝 analysis_only；重新分析应下载并引用已有采集，使用独立分析工具，不能借分析登记启动模型脚本。

## 范围说明

本协议供日常运行器和归档命令行使用。历史记录按原始产物分批登记，历史缺件不得从备忘录推造。单元测试使用隔离库与小文件工程夹具，不向正式中心发送伪造实验。部署后首次真实科研测试前，先检查 capabilities，再使用管理员认证查询已有记录。

## 中心模型目录与原值回放

中心与客户端一起升级。客户端从 `GET /api/research/workspace/catalog` 读取公开快照、资源和回放覆盖。首次同步后可离线读取此中心已取得的公开目录和切片。私有目录和切片沿用测试归档管理员会话，访问时从中心读取。

部署包内 `deploy/workspace_data/catalog.json` 提供已登记记录。新增或更新条目时，把 `workspace_publication.v1` 清单及所引用的 JSON 置于 `AI2050_WORKSPACE_DIR`（默认中心数据目录下的 `workspace`）。发布文件用同目录暂存后原子替换。模型由完整 `model_snapshot` 定位，归档和资源按 `id` 覆盖，`withdrawn: true` 撤下对应条目。发布目录可用 `goodfire-records.json` 更新公开测试索引，用 `history-records.json` 更新受限历史索引。客户端点「同步中心目录」即可取得更新。

公开原场按已公开的快照、运行、题目、读出位置、模型块和测量量登记。私有归档登记现有 `run`、`caseId`、`trace`、`artifact` 和 `source_block`，中心在返回原值前校验上传对象与逐题回执的哈希、快照及读出位置。Phase 3075 的原场沿这个入口读取已有 `native-0000.npz`。其它归档若没有可映射的原场，会提示范围或缺测原因。技术归档状态和科学复核状态分别显示。

客户端回放只需中心地址。实时模式还需安装兼容的本机采集服务，并将 `AI2050_LIVE_MODEL_DIR` 指向实际权重目录。界面会在加载与运行时提交所选快照供本机核验。目前此工作台的原生实时采集适配器覆盖已登记的 Qwen3 快照。

## 统一证据检索与历史回填

「总览 → 测试数据库」默认打开统一证据检索。访客通过 `GET /api/research/workspace/evidence` 查看公开研究、运行和原题。管理员使用相同筛选条件查询 `GET /api/research/workspace/private/evidence`，同时检索受限历史索引、中心运行和可读取的归档原题。受限结果不进入公开缓存。

可通过 `q` 查机制、输入、输出及反例，可叠加 `source`、`coverage`、`snapshot`、`family`、`target`、`position`、`token_id`、`counter` 和分页。部件用 `B26/H36` 或 `B26/U17407` 等原生编号。指定 Token 位置或 ID 时，仅返回相同快照与原题上已绑定该位置的原值入口。携带 `targets` 的冻结方案可按声明对象找到，逐题可回放覆盖以实际归档为准。私有逐题索引支持 JSONL 与包含题目数组的 JSON，读取前校验上传清单的 SHA-256。单个文件限 16 MiB，一次索引的私有文件读取预算为 64 MiB、解析范围为 20,000 行。超限或旧格式文件仍可从原运行按权限下载，检索页面会说明未纳入的范围。

各覆盖数字按对象类型分别计数。历史出处只有原文路径、Phase 和段落 SHA-256 完全相符时才绑定到唯一的 Goodfire 记录。逐题回放要求同一运行、模型快照和题号，私有记录还核对轨迹 ID。已发布 B26 两次实验的逐题评分直接索引原发布数据。能读取原值的少数题带有单独的原场入口。现有 `goodfire-evidence.json` 补充已核对的出处、缺测理由和复测增量。后续可在中心工作区目录更新同名文件，新增注记须使用原记录的段落哈希。

升级时保留中心数据目录和现有归档库。先使用已有真实记录核对搜索、逐题文件和同一部件、Token 的原值。批量补录沿用已有运行 ID 和回执。只有历史出处的条目按出处浏览，找回真实材料后再追加归档版本。可回放的记录在资料齐全时允许建立「基于此原场新建分析」草稿，草稿仍走既有查重和运行登记流程。
