# CentOS 部署指南 — AI2050 分布式研发平台（中心节点）

对应 `design/distributed_platform_plan_v1.md` S2–S6（M1 最小闭环）。
服务本体 `deploy/distributed_service.py`，依赖 fastapi/uvicorn/pydantic/numpy（无 GPU 依赖）；协议 v3 代码在仓库 `shared/dist_protocol/`，deploy/ 必须放在仓库目录内（`<repo>/deploy/`）。

> **⚠️ deploy/ 不入 git**：`.gitignore` 已忽略整个 `deploy/`（中心节点服务、种子模板、
> runner、节点 Agent 均属服务器侧代码，不进公开仓库）。`git clone` 得到的仓库**不含**本目录，
> 部署前需从本机把整个 `deploy/` 拷贝到服务器（`scp`/`rsync` 均可）。

## 1. 基础环境（CentOS 7/8/9 Stream）

```bash
sudo dnf install -y python3.11 python3.11-pip git   # 7 用 python3 + EPEL 亦可
sudo useradd -r -s /sbin/nologin ai2050
sudo mkdir -p /opt/ai2050 /var/lib/ai2050/distributed
sudo chown ai2050:ai2050 /opt/ai2050 /var/lib/ai2050/distributed
```

## 2. 代码与依赖

```bash
# deploy/ 不随 git 分发，先从本机拷贝（在本地机器执行）：
scp -r deploy/ user@<服务器IP>:/opt/ai2050/Ai2050-OpenOne/deploy/

sudo -u ai2050 git clone https://github.com/ai2050lin/Ai2050-OpenOne.git /opt/ai2050/Ai2050-OpenOne
cd /opt/ai2050/Ai2050-OpenOne
python3.11 -m venv /opt/ai2050/venv
/opt/ai2050/venv/bin/pip install --upgrade pip
/opt/ai2050/venv/bin/pip install fastapi "uvicorn[standard]" pydantic numpy
# numpy：dist_protocol v3（shared/dist_protocol，随 git 分发）合并充分统计量所需；缺失时 v3 端点返回 503，legacy 端点不受影响
```

## 3. 配置

```bash
# 管理令牌（注册模板用），替换随机值
export ADMIN_TOKEN=$(python3 -c "import secrets;print(secrets.token_hex(24))")
sudo sed -i "s/change-me-to-random-hex/${ADMIN_TOKEN}/" deploy/ai2050-distributed.service
```

## 4. systemd 安装

```bash
sudo cp deploy/ai2050-distributed.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai2050-distributed
systemctl status ai2050-distributed
journalctl -u ai2050-distributed -f
```

首次启动自动播种 TM-01/02/03 三个模板（多条件分解：语言/风格/逻辑最小对，
见 `deploy/dist_templates_seed.py`）。日志出现 `seeded 3 templates` 即就绪。

## 5. 防火墙 / SELinux

```bash
sudo firewall-cmd --permanent --add-port=5010/tcp && sudo firewall-cmd --reload
# SELinux enforcing 时端口策略（9200 语义,按需）：
sudo semanage port -a -t http_port_t -p tcp 5010 || true
```

对外仅建议开放 5010（或经 nginx 反代 443 + TLS；Agent 与访客 API 同源）。

## 6. 验证（服务器本机）

```bash
curl -s http://127.0.0.1:5010/                          # {"service":"ai2050-distributed",...}
curl -s http://127.0.0.1:5010/api/distributed/summary    # 节点/模板/结果总览
curl -s http://127.0.0.1:5010/api/news                   # arXiv 抓取或内置兜底
```

## 7. Worker 节点接入（任意机器，Windows/Linux 皆可）

节点机器只需 `deploy/node_agent.py`（仅标准库）+ Python 3.10+：

```bash
python deploy/node_agent.py --server http://<中心IP>:5010 register --name gpu-box-1 --gpu "RTX 5080"
python deploy/node_agent.py --server http://<中心IP>:5010 run --once          # 联调（冒烟模式）
# 真实采集（需 torch+transformers 与模型权重）：
python deploy/node_agent.py --server http://<中心IP>:5010 run \
    --model-path /data/models/qwen3-4b --model-id qwen3-4b --gpu "RTX 5080" --dtype bf16
# 常驻：建议 systemd --user 或 tmux；--interval 60 轮询
```

凭据存 `~/.ai2050_agent/node_credentials.json`（node_token ≠ AI API key，后者永不出本机）。
冒烟结果 kind=smoke，与真实结果在 `/api/results` 中可辨——诚实口径，smoke 不冒充实测。

## 8. 升级与运维

- 模板内容变更 → `POST /api/templates` 升 version（design_sha 变化，旧结果不可跨版混比）；
- 备份 = 打包 `/var/lib/ai2050/distributed`（sqlite + objects 内容寻址，可增量）；
- 数据库 WAL 模式，可直接 `sqlite3 dist.sqlite3` 只读查询；
- 汇总大屏：`GET /api/distributed/summary`（本仓库融合页「平台进度」tab 已接入）。
