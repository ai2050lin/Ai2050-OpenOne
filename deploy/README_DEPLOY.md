# 开放平台中心 · 部署指南

中心服务是一个不需要 GPU 的 Python 进程（`deploy/center_app.py`），同时托管 API 和客户端网页。参与者的节点从中心领取冻结的任务包，在自己的 GPU 上运行，再上传统计结果。

```
浏览器 ──https──> nginx :80/443 ──> center_app 127.0.0.1:5010 ──> /var/lib/ai2050/center   SQLite、结果、普查作业
                                                              └─> /var/lib/ai2050/native   普查语料、合并后的普查
参与者节点（node_agent.py participate）──> 同一个域名的 /api/*
```

## 1. 生成发布包（在开发机上）

```bash
cd frontend && npx vite build && cd ..
python scripts/build_center_release.py          # → release/ai2050-center-<版本>.tar.gz 与 .sha256
```

发布包包含：中心与节点程序、协议代码、客户端构建，以及公开研究数据（`research/interventions`、`research/neurons`）。研究记录、模型权重、普查语料和原生全场都不在包里。

## 2. 安装（在服务器上）

```bash
scp release/ai2050-center-1.0.0.tar.gz root@<服务器>:/root/
ssh root@<服务器>
tar xzf ai2050-center-1.0.0.tar.gz && cd ai2050-center-1.0.0
DOMAIN=platform.example.org bash deploy/install_center.sh --yes --https
```

安装脚本会创建系统用户 `ai2050`、Python 环境（依赖见 `deploy/requirements-center.txt`）、systemd 服务 `ai2050-center` 和 nginx 站点，并生成随机管理令牌 `/etc/ai2050/admin_token`。脚本可以重复运行：升级时覆盖程序，保留数据、令牌和站点配置。

安装后先在浏览器打开站点并注册**第一个账号**，它会自动成为管理员。之后的注册方式由 `AI2050_REGISTRATION` 控制：`open` 表示任何人都能注册，`closed` 表示只能由管理员创建账号。修改位置是 `/etc/systemd/system/ai2050-center.service`。

## 3. 站点配置

站点配置在 `/etc/ai2050/platform_config.json`，模板见 `deploy/platform_config.example.json`：

| 字段 | 作用 |
| --- | --- |
| `title` / `tagline` / `contact` / `repo_url` | 首页与页脚显示的文字 |
| `task_kinds` | 开放给参与者的任务类型：`census_shard`、`family_test` |
| `models` | 参与者需要准备的模型：`name`、`hf_repo`、`hf_revision`、`snapshot_sha256`；开放的普查作业会自动补进这里 |
| `ui.tabs` | 客户端显示哪些页面：`home`、`learn`、`atlas`、`join` |

修改后立即生效，不需要重启。

## 4. 开放神经元普查

1. **上传语料**。在开发机上用 `scripts/prepare_census_corpus.py` 生成语料，然后上传：

   ```bash
   rsync -av D:/ai2050_native/_census/corpus/<指纹>/ root@<服务器>:/var/lib/ai2050/native/_census/corpus/<指纹>/
   ```

2. **准备模型目录**。中心至少需要 `config.json` 和 `tokenizer.json`，不需要权重：

   ```bash
   mkdir -p /opt/ai2050/models/qwen3-4b && cp config.json tokenizer.json tokenizer_config.json /opt/ai2050/models/qwen3-4b/
   ```

   中心没有权重时，要在持有权重的机器上计算快照：

   ```bash
   python deploy/admin_tool.py snapshot <模型目录> --out qwen3-4b.snapshot.json
   ```

3. **建作业**。每个分片的规模是「切片 token 数 × 层组」：

   ```bash
   sudo /opt/ai2050/venv/bin/python /opt/ai2050/app/deploy/admin_tool.py census-create \
       --job-id CJ-qwen3-4b-20M --model-dir /opt/ai2050/models/qwen3-4b --snapshot-file qwen3-4b.snapshot.json \
       --corpus-fp <指纹> --tokens-per-slice 2000000 --group-size 4 --replica-rate 0.1 \
       --model-name qwen3-4b --hf-repo Qwen/Qwen3-4B
   ```

   在 RTX 5080 上实测：Qwen3-4B 一个 4 层分片约 4300–5300 token/s，显存峰值约 10 GB，上传约 30 MB，与切片长度基本无关。
   以 2M token/切片为例：2000 万 token 共 10 个切片 × 9 个层组 = 90 个分片，每个约 7 分钟。

4. **进度与合并**。中心每 5 分钟把新完成的切片合并进 `/var/lib/ai2050/native/_census/<model_key>/<job_id>/`，「图谱 · 神经元单元卡」随即可查。也可以手动合并：

   ```bash
   sudo .../admin_tool.py census-list
   sudo .../admin_tool.py census-merge CJ-qwen3-4b-20M
   ```

5. **复算不一致**。按 `replica_rate` 抽样的分片会由两个不同账号的节点各跑一次。两份结果不一致时都标为 `disputed`，暂不合并：

   ```bash
   sudo .../admin_tool.py census-review                     # 看比较指标
   sudo .../admin_tool.py census-resolve <result_id> accept # 接受一份，另一份驳回
   ```

## 5. 防作弊与复核

- **任务包冻结**：runner 和 `census.py` 在建作业时复制并记录 sha256。节点逐个核对文件后才执行，结果里带 runner 哈希，中心再核对一次。
- **中心逐项校验**：
  - 每层计入的 token 数必须等于中心从切片重算的值；
  - 直方图每个单元的总数必须等于 n；
  - 平方和与和必须自洽，极值列表要有序且首项等于 max / min；
  - 极值的 token 位置必须落在该切片实际计入的位置里；
  - 模型快照必须一致。
- **复算**：抽样分片派给另一个账号，按均值 z 值、max 相对差、top-1 位置一致率和直方图 L1 距离比较。
- **边界**：这些检查只说明产物可信、可以合并，不是证据晋级。合成（synthetic）结果只有沙盒中心（`AI2050_DIST_ALLOW_SYNTHETIC=1`）才接受。

## 6. 迁移已有数据（可选）

在开发机上已有的中心数据（`distributed_data/`，包括模式族模板、结果和账号）可以整目录同步到 `/var/lib/ai2050/center/`。同步前先停服务：`systemctl stop ai2050-center`，同步后执行 `chown -R ai2050:ai2050`。

已有的本机普查结果（`<native>/_census/<model_key>/<run>/`，Qwen3-4B 2000 万 token 约 1.1 GB）与对应语料一起同步到 `/var/lib/ai2050/native/_census/` 后，单元卡就能直接读取。

## 7. 运维

| 事项 | 命令 |
| --- | --- |
| 日志 | `journalctl -u ai2050-center -f` |
| 重启 | `systemctl restart ai2050-center` |
| 健康检查 | `curl http://127.0.0.1:5010/api/health` |
| 备份 | `/var/lib/ai2050/center`（SQLite 用 `sqlite3 dist.sqlite3 ".backup b.sqlite3"`）与 `/etc/ai2050` |
| 首次启动 | 后台会计算谱图与总览快照（可能需要几分钟），公开页面先显示「计算中」 |
