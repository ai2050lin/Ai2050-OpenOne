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
python scripts/build_center_release.py          # 递增 VERSION → release/ai2050-center-<新版本>.tar.gz 与 .sha256
```

发布包包含：中心与节点程序、协议代码、客户端构建，以及公开研究数据（`research/interventions`、`research/neurons`）。研究记录、模型权重、普查语料和原生全场都不在包里。

## 2. 安装（在服务器上）

```bash
scp release/ai2050-center-1.0.0.tar.gz root@<服务器>:/root/
ssh root@<服务器>
tar xzf ai2050-center-1.0.0.tar.gz && cd ai2050-center-1.0.0
DOMAIN=platform.example.org bash deploy/install_center.sh --yes --https
```

安装脚本重复运行时会保留已有的 nginx HTTPS 配置（certbot 写入的 443 段）；`--reset-nginx` 按模板重写，服务器上已有证书时会自动装回。客户端默认连接 `https://`，HTTPS 不通时客户端会显示「连不上中心服务」。

安装脚本会创建系统用户 `ai2050`、Python 环境（依赖见 `deploy/requirements-center.txt`）、systemd 服务 `ai2050-center` 和 nginx 站点，并生成随机管理令牌 `/etc/ai2050/admin_token`。脚本可以重复运行：升级时覆盖程序，保留数据、令牌和站点配置。

安装后先在浏览器打开站点并注册**第一个账号**，它会自动成为管理员。之后的注册方式由 `AI2050_REGISTRATION` 控制：`open` 表示任何人都能注册，`closed` 表示只能由管理员创建账号。修改位置是 `/etc/systemd/system/ai2050-center.service`。

## 2.1 版本与升级

- 每次打包都会递增版本号：仓库根目录 `VERSION`，缺省递增补丁号。发布包和解压目录都带版本号，如 `ai2050-center-1.0.1/`。
- 版本号可以在三处核对：站点页脚、`/api/health` 和 `/api/platform/config` 的 `version` 字段；包内的 `RELEASE_MANIFEST.json` 还记录了对应的 git 提交。
- **数据不要放在带版本号的目录里。** 中心数据由 `AI2050_DIST_DIR` 决定；没有设置时会落在程序目录下的 `distributed_data/`，换用新版本目录时就「丢」了。`/api/health` 的 `data_dir_in_release` 为 `true` 时，说明当前就是这种情况。

**推荐的服务器布局（以 `/data` 为例）：**

```
/data/ai2050-center-1.0.0/      旧版本程序（解压目录，可保留以便回退）
/data/ai2050-center-1.0.1/      新版本程序
/data/ai2050-data/center/       中心数据：dist.sqlite3、结果对象、普查作业
/data/ai2050-data/native/       普查语料与合并后的普查结果
/etc/ai2050/                    admin_token、platform_config.json
```

用安装脚本部署时，程序会复制到 `/opt/ai2050/app`，数据默认在 `/var/lib/ai2050/`，两者都与解压目录无关。直接在解压目录里运行 `center_app.py` 时，必须在 systemd 里设置 `AI2050_DIST_DIR` 与 `AI2050_NATIVE_DIR`，让它们指向版本目录之外。

**升级步骤（直接从解压目录运行的情况）：**

```bash
# 1. 备份数据库
sqlite3 /data/ai2050-data/center/dist.sqlite3 ".backup /root/dist-$(date +%Y%m%d).sqlite3"
# 2. 解压新版本，与旧版本并排存放
cd /data && tar xzf ai2050-center-1.0.1.tar.gz
# 3. 把 systemd 中的 WorkingDirectory / ExecStart 改到新目录，数据环境变量保持不变
sudo systemctl edit --full ai2050-center
sudo systemctl daemon-reload && sudo systemctl restart ai2050-center
curl -s http://127.0.0.1:5010/api/health        # version 应为 1.0.1，data_dir_in_release 应为 false
```

需要回退时，把 systemd 指回旧目录后重启即可；数据库没有被改动过。

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

## 6. 导入研究数据（让各页面有内容）

新中心刚装好时没有研究数据，首页、谱图、实验响应、路线、神经元卡都会显示空状态。用数据包把研发机上的研究数据**合并**进来。合并只新增记录，不覆盖已有记录，也不碰账号。

1. **在研发机上导出：**

   ```bash
   python scripts/export_center_data.py      # → release/ai2050-data-<日期>.tar（约 6 GB）与 .sha256
   ```

   数据包包含：当前模板版本的 real 结果与模板包、规律分析缓存、最大的一次普查（含语料与 tokenizer）、模型快照。不包含账号、会话、旧版本结果和原生全场。

2. **传到服务器后合并导入：**

   ```bash
   systemctl cat ai2050-center | grep -E "AI2050_(DIST|NATIVE)_DIR"        # 确认数据目录
   systemctl stop ai2050-center
   python3 /data/ai2050-center-<版本>/deploy/import_center_data.py ai2050-data-<日期>.tar        --data-dir <AI2050_DIST_DIR> --native-dir <AI2050_NATIVE_DIR>     # 可先加 --dry-run 看计划
   chown -R ai2050:ai2050 <AI2050_DIST_DIR> <AI2050_NATIVE_DIR>
   systemctl start ai2050-center
   ```

   导入前会自动备份数据库（`dist.sqlite3.before-import-<时间>`）。导入是幂等的，可以重复执行。

3. **开放普查试点作业**（参与页随即出现可领取的任务）。数据包里带了模型快照与 tokenizer 目录：

   ```bash
   sudo python3 /data/ai2050-center-<版本>/deploy/admin_tool.py census-create --job-id CJ-qwen3-4b-pilot        --model-dir <AI2050_NATIVE_DIR>/_census/6be1034f0b55a6df-bf16/2ca5f0a756d36c8a-20000000/tokenizer        --snapshot-file <AI2050_NATIVE_DIR>/_census/snapshots/6be1034f0b55a6df-bf16.json        --corpus-fp 2ca5f0a756d36c8a --tokens-per-slice 2000000 --max-slices 2        --model-name qwen3-4b --hf-repo Qwen/Qwen3-4B --hf-revision 1cfa9a7208912126459214e8b04321603b3df60c
   ```

中心启动时会拉起一个低优先级的后台计算子进程（`deploy/snapshot_worker.py`），负责计算总览、谱图与记分板的快照，并预热实验响应页的规律分析；结果写到 `<AI2050_DIST_DIR>/snapshots/`，网站进程只读这些文件，计算再久也不影响访问。每份快照都带数据指纹，重启后直接可用，只有数据变化时才重算。数据包里已有缓存的模板，打开即时显示；没有缓存的大模板首次需要较长时间，页面会提示「后台计算中」，算完自动刷新。

同一个子进程还会在数据变化时生成**客户端离线数据包**（`<AI2050_DIST_DIR>/datapack/datapack.json.gz`，约 1–2 MB）。客户端平时只读取站点配置；页面左下角显示本地数据时间，服务器有新数据时出现「更新数据」按钮，由用户手动下载，不会定时访问服务器。单元卡与管理类页面（审计、分发、普查作业、自动测试）打开时在线读取，用页面上的「刷新」按钮更新。

## 7. 运维

| 事项 | 命令 |
| --- | --- |
| 日志 | `journalctl -u ai2050-center -f` |
| 重启 | `systemctl restart ai2050-center` |
| 健康检查 | `curl http://127.0.0.1:5010/api/health` |
| 备份 | `/var/lib/ai2050/center`（SQLite 用 `sqlite3 dist.sqlite3 ".backup b.sqlite3"`）与 `/etc/ai2050` |
| 首次启动 | 后台会计算谱图与总览快照（可能需要几分钟），公开页面先显示「计算中」 |
