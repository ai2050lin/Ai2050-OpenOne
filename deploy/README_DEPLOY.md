# 开放平台中心 · 部署指南

中心服务是一个不需要 GPU 的 Python 进程（`deploy/center_app.py`），同时托管 API 和客户端网页。它不执行任何模型前向：只保存数据、提供下载、转发命令、回收结果；实测与普查都在本机或参与者节点上跑（职责表见 2.2）。参与者的节点从中心领取冻结的任务包，在自己的 GPU 上运行，再上传统计结果。

```
浏览器 ──https──> nginx :80/443 ──> center_app 127.0.0.1:5010 ──> /var/lib/ai2050/center   SQLite、结果、普查作业
                                                              └─> /var/lib/ai2050/native   普查语料、合并后的普查
参与者节点（node_agent.py participate；桌面客户端经本机节点服务 node_agent.py serve）──> 同一个域名的 /api/*
```

## 1. 生成发布包（在开发机上）

```bash
cd frontend && npx vite build && cd ..
python scripts/build_center_release.py          # 递增 VERSION → release/ai2050-center-<新版本>.tar.gz 与 .sha256
```

发布包包含：中心与节点程序、协议代码、客户端构建、公开研究数据（`research/interventions`、`research/neurons`）与数据集清单摘要（`data/lang_bench/meta.v3.json`）。研究记录、模型权重、普查语料、原生全场和**题库正文**都不在包里——题库正文按源地址现取，见 2.3。

需要一份「题目齐全」的离线包（例如断网机房）时可以显式带上：

```bash
python scripts/build_center_release.py --with-questions
```

这时 C-Eval / M3Exam（CC BY-NC-SA 4.0）会随包分发，接收方同样受非商业与相同方式共享的约束。

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

**账号管理**：管理员登录后，在右上角账号菜单进入「账号管理」（`https://<域名>/?page=accounts`）。这一页可以管理本中心的全部账号：

- 创建账号、改角色和资料。
- 为他人重置密码。
- 强制下线、停用或启用账号。
- 删除账号：要输入用户名确认，名下节点改为无主，已上传的结果保留。
- 开关自助注册：页面上的设置优先于 `AI2050_REGISTRATION`。

系统至少保留一个可用的管理员。

**忘记密码**：页面上没有找回功能，需要在中心服务器上重置。脚本直接修改数据库，不需要管理令牌。新密码要在终端输入两次，输入时不回显。重置后，该账号已有的登录全部失效，锁定和停用状态也会一起解除。

- **找数据目录**：用 `systemctl show ai2050-center -p Environment` 查看 `AI2050_DIST_DIR`。用安装脚本部署的中心，数据目录默认是 `/var/lib/ai2050/center`。
- **管理员账号全部丢失时**：加 `--make-admin`，重置的同时把账号设为管理员。

```bash
sudo -u ai2050 /opt/ai2050/venv/bin/python /opt/ai2050/app/deploy/reset_password.py --data-dir /var/lib/ai2050/center --list
sudo -u ai2050 /opt/ai2050/venv/bin/python /opt/ai2050/app/deploy/reset_password.py --data-dir /var/lib/ai2050/center <用户名>
```

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

**一键升级（推荐）**：打包时会在 `release/` 里同时放一份 `upgrade_center.sh`。把发布包、它的 `.sha256` 和这个脚本传到服务器的同一目录，以 root 运行：

```bash
scp release/ai2050-center-<版本>.tar.gz release/ai2050-center-<版本>.tar.gz.sha256 release/upgrade_center.sh root@<服务器>:/data/
ssh root@<服务器> 'cd /data && bash upgrade_center.sh'
```

脚本依次完成以下步骤：

1. 核对发布包的 sha256。
2. 从 systemd 读出当前的程序目录、数据目录和端口。
3. 把数据库在线备份到 `/root/ai2050-backups/`，并检查备份的完整性。
4. 部署新版本并重启，等 `/api/health` 报告新版本号。

两种布局都支持：

- **版本目录**（如 `/data/ai2050-center-1.0.13`）：新版本解压到同一父目录，systemd 改指新目录，旧目录保留。
- **固定目录**（`/opt/ai2050/app`）：先把旧程序复制为 `app.bak-<旧版本>`，再原地同步新程序。

脚本在动手前先做检查，以下情况直接拒绝，不改任何东西：数据目录在程序目录之内、sha256 不一致、当前已经是新版本。重启失败或 health 不对时，脚本会自动恢复旧版本的程序与 systemd 配置，并打印启动日志。

**手动升级步骤（直接从解压目录运行的情况）：**

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

## 2.2 中心的职责边界：不碰 GPU

中心进程只做四件事：**数据保存、数据下载、命令转发、结果回收**。`deploy/requirements-center.txt` 里没有 torch / transformers，中心也没有任何加载模型权重的代码路径；所有涉及前向、训练、消融的功能都在本机或参与节点上执行。

| 功能 | 在哪执行 | 中心做什么 |
| --- | --- | --- |
| 图谱测试题实测 `/api/lang/bench/run` | 本机（有模型的机器） | **不提供该端点**；`/api/lang/bench/overview` 的 `model_status` 恒为 `unavailable` |
| 数据集三类实测 `/api/lang/atlas/run` | 本机 | 同上；只保留历史的读数与逐头结果 |
| 自动测试批次 `/api/autotest/*` | 本机中心（`AI2050_LOCAL_TOOLS=1`） | 公开中心一律 `403` + `local_only: true`，**不给管理令牌后门** |
| 神经元普查分片 | 参与节点 | 派发作业、校验分片、合并结果 |
| 单元卡生成 | 数据快照子进程（CPU） | 生成、缓存、分发 |

这张表也通过接口下发给客户端：`GET /api/platform/config` 的 `compute` 字段（`gpu: false`、`role`、`local_execution`、`note`）。界面据此把「运行实测」换成「本机执行」的说明，而不是放一个点了必然失败的按钮。

## 2.3 数据集：按源地址引用，不随包分发

数据集清单（`/api/lang/atlas/datasets`、`/api/lang/bench/overview`）只需要 `data/lang_bench/meta.v3.json` 里的 `by_dataset` 摘要：题量、科目、三类可映射题量、许可、源地址。**题目正文 `questions.v3.jsonl` 不在发布包里**，原因是许可（C-Eval / M3Exam 为 CC BY-NC-SA 4.0，MMLU 为 MIT）与体积；正文按下面的地址在本机取。

| 数据集 | 源地址 | 拆分 | 许可 | 题量 / 可映射三类 |
| --- | --- | --- | --- | --- |
| C-Eval | `huggingface.co/datasets/ceval/ceval-exam` | val | CC BY-NC-SA 4.0 | 169 / 169 |
| MMLU | `huggingface.co/datasets/cais/mmlu` | test | MIT | 14,042 / 2,504 |
| M3Exam | `huggingface.co/datasets/chiayewken/m3exam` | chinese/test | CC BY-NC-SA 4.0 | 682 / 565 |

源地址与许可是唯一真源 `shared/dist_protocol/bench_sources.py`，接口的 `sources` 字段与界面上的「原始数据」链接都从这里来。在本机（有 GPU、要跑实测的那台）取题库：

```bash
python scripts/download_lang_bench_v3.py                 # 先按源地址下载 parquet，再构建 questions.v3.jsonl
python scripts/download_lang_bench_v3.py --meta-only     # 只重算 meta 摘要（改了映射表以后用）
```

本机有正文时接口 `source.mode` 为 `questions`（逐行数出来），没有正文时为 `meta`（读建库时的摘要）；两种模式给出的题量、科目、三类分布完全相同，界面在 `meta` 模式下标注「题量为建库时统计」。

**许可**：本平台是非商业研究项目。使用 C-Eval / M3Exam 数据需保留署名、非商业使用、相同方式共享；MMLU 为 MIT，只需保留版权与许可声明。若将来转商业部署，必须先把 C-Eval / M3Exam 从任何分发渠道移除。

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

同一个子进程还会在数据变化时生成**客户端离线数据包**：主数据包（`<AI2050_DIST_DIR>/datapack/datapack.json.gz`，约 1–2 MB）与附加包（`datapack/extra/`：每个模式族一个实验响应包，0.3–30 MB；每个有普查的模型一个单元卡包，约 10 MB）。客户端（网页版、连接本中心的 vite dev 与安装包）都是**只读数据包模式**：使用中只读取本机数据包，不实时访问其他接口；连接服务器的只有登录 / 注销 / 账号操作、读取站点配置、下载数据包与附加包、下载一键包。第一次打开自动下载一次主数据包，页面左下角显示本地数据时间，服务器有新数据时出现「更新数据」按钮；打开实验响应或单元卡时提示下载（或更新）对应的附加包。审计、冻结、分发、普查管理在研究工作台（`/workbench`，实时连接中心，用「刷新」按钮更新）进行。第一次生成单元卡包约需 20 分钟，之后只在研究文件或普查结果变化时重建；缓存包里带了预先生成的数据包（见 6.1）。

规律分析预热之后，子进程还会生成**语言谱图**（`snapshots/lang_atlas.json`，pieces.v1）：把各模式族的规律分析挂到语言树上，供 `/api/lang/tree`、`/api/lang/item/{id}`、`/api/model/skeleton`、`/api/space/overlay` 读取，并随主数据包下发。每个规律分析文件只解析一次（大文件峰值约 300 MB），抽取结果缓存在 `<AI2050_DIST_DIR>/lang_cache/`；缓存包里已带好。同一子进程还为每个有普查的模型生成**逐层神经元摘要**（`<AI2050_NATIVE_DIR>/_census/<模型>/<运行>/neuron_layers/`，每块约 0.3 MB、0.3 秒，36 块约 15 秒），供神经元主线的层塔与单元阵列使用。v8 客户端在 `https://<域名>/atlas`（两条主线：语言图谱、神经元），默认首页不变。

### 6.1 小内存服务器（2 GB 及以下）

一个大模式族的分片统计量展开后要占 0.5–0.9 GB 内存，第一次合并时小内存、无 swap 的服务器会被系统强制结束进程（`journalctl` 里出现 `code=killed, status=9/KILL`）。中心为此做了三层保护：

- 合并统计（`agg_cache/`）与规律分析（`analysis_cache/`）都落盘，算过一次之后网站进程只读文件；规律分析结果原样发送文件，不在网站进程里解析。
- 重计算只在后台子进程里做，且同一进程一次只合并一个桶；子进程把自己的 OOM 优先级调到最高，内存耗尽时系统先结束它，网站不受影响。
- 某一步计算中被结束后，同一数据版本下不再自动重试（记录在 `snapshots/_skipped.json`），不会每次重启都把机器拖垮；数据变化后会再试一次。

建议：

1. 加 2–4 GB swap（只需一次）：
   ```bash
   sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile && echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```
2. 在开发机上用同一份数据算好缓存，打成缓存包上传（`ai2050-cache-<版本>.tar.gz`，含 `agg_cache/`、`analysis_cache/`、`lang_cache/` 与 `datapack/`），服务器上直接解开，不用在服务器上合并（`.tar.gz` 用 `tar -xzf`）：
   ```bash
   sudo tar -xf ai2050-cache-<版本>.tar -C /var/lib/ai2050/center && sudo chown -R ai2050:ai2050 /var/lib/ai2050/center && sudo systemctl restart ai2050-center
   ```
   缓存键含结果 sha、审计状态与代码哈希，只有与服务器数据、代码版本一致时才命中；不一致的部分由后台子进程照常补算。

## 7. 运维

| 事项 | 命令 |
| --- | --- |
| 日志 | `journalctl -u ai2050-center -f` |
| 重启 | `systemctl restart ai2050-center` |
| 健康检查 | `curl http://127.0.0.1:5010/api/health` |
| 备份 | `/var/lib/ai2050/center`（SQLite 用 `sqlite3 dist.sqlite3 ".backup b.sqlite3"`）与 `/etc/ai2050` |
| 首次启动 | 后台会计算谱图与总览快照（可能需要几分钟），公开页面先显示「计算中」 |


## 8. 持续研究库与节点托管（2026-11-03）

更新中心时一并部署 research_library.py、node_control.py、parameter_service.py、parameter_reader.py；build_center_release.py 已包含这些文件。桌面 extraResources 与命令行一键包也包含 parameter_reader.py，读取参数不依赖中心保有权重。Windows x64 客户端 1.0.1 已打包并在本机启动验收，远端中心尚未部署；旧节点需更新 node_service.py 才能托管。

持久库位于 `<AI2050_DIST_DIR>/research_library/`，随中心数据库、objects/、census/ 一起备份，不能在升级程序目录时清空。后台 snapshot_worker 自动索引公开研究报告、真实分片和审计 / 复核变化；默认每轮结束后等待300秒。新实验输出 `research/**/report.json` 或 `*.research.json`；成员可从 Goodfire 报告主动提交，管理员可登记显式图谱。可用 AI2050_RESEARCH_DIR 指定研究根目录。主包与 research-library 附加包供离线客户端更新，schema 与证据边界见 design/platform_spec_v1.md 第9节。

远程托管只接受本人账号对已在本机授权的节点进行暂停 / 恢复 / 停止；公开中心仍不加载 GPU 模型。公开中心不要设置 AI2050_LOCAL_TOOLS=1。参数浏览默认走桌面本机 IPC。没有完整快照或显式证据的旧报告保留原貌和版本，不自动成为机制结论。

完整本机服务 `server/server.py` 与独立中心都随生命周期启动研究快照后台，退出时停止；已有旧快照也必须继续刷新，不能只在首次缺文件时运行。沿用 `AI2050_SNAPSHOT_DISABLE` 与 `AI2050_SNAPSHOT_SECONDS`。客户端升级不会自动升级远端中心；新增研究库需中心代码和离线数据包同时更新。1.0.1 安装包、blockmap 和 SHA256SUMS.txt 位于 `frontend/dist-installer/1.0.1/`。本轮验收启动的是同包解出的程序，未运行安装向导、未覆盖已有安装。

1.0.2 新增工作包、最近拼图修订与样本完整MLP响应回查。更新 research_library.py、parameter_reader.py、parameter_service.py、node_service.py、datapack.py、语言树与对应前端；保留原有持久库。桌面主进程新增本机 `/parameters/identity` 白名单和完整哈希核验超时，旧节点服务需要重启加载新代码。安装包在 `frontend/dist-installer/1.0.2/`。公开中心仍不能开放本机权重工具；普通客户端只读取研究数据包，核验模型通过本机节点完成。升级后先更新主数据，再按提示下载研究库附加包。
