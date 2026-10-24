# CentOS 部署指南 — AI2050 平台（域名 platform.ai2050.com）

对应服务：`server/server.py`（:5001，客户端**唯一数据源**）+ nginx 静态托管 + 可选分布式平台（:5010）。

> **⚠️ `deploy/` 不入 git**：`.gitignore` 忽略整个 `deploy/`（中心服务、种子模板、runner、节点 Agent、
> nginx 配置、部署脚本均属服务器侧代码）。`git clone` 得到的仓库**不含**本目录，
> 部署前需从本机把整个 `deploy/` 拷贝过去（`scp` / `rsync` 均可）。
>
> **一键部署（推荐）**：`sudo bash deploy/install_centos.sh --yes`
> 本文档同时给出逐步手动命令，便于排障。

---

## 0. 一键部署（推荐）

### 0.1 前置

```bash
# 1) 域名 A 记录：platform.ai2050.com → 本机公网 IP（先确认 IP）
curl -fsS https://api.ipify.org

# 2) 本机准备前端产物（服务器不装 node，直接传 dist）
cd frontend && npx vite build        # 产出 frontend/dist/
```

### 0.2 传输

```bash
# 本机执行（把整个仓库传过去；exclude 掉大目录）
rsync -avz --exclude 'node_modules' --exclude '.venv' --exclude 'tests' \
      --exclude '.git' --exclude 'ai2050_research_os/generated' \
      ./  root@<服务器IP>:/opt/ai2050/Ai2050-OpenOne/
```

### 0.3 执行

```bash
ssh root@<服务器IP>
cd /opt/ai2050/Ai2050-OpenOne
sudo bash deploy/install_centos.sh --yes
```

脚本会依次完成：用户/目录 → 前端 dist → venv+依赖（torch 走 CPU wheel）→ systemd（:5001 + 可选 :5010）
→ nginx 站点 → 防火墙 → certbot HTTPS → 7 项验证。

**可用参数**：

| 参数 | 作用 |
|---|---|
| `--yes` | 无人值守（跳过交互提问） |
| `--no-https` | 跳过 certbot |
| `--skip-dist` | 跳过 :5010 分布式平台 |
| `--fresh` | 重建 venv（换依赖大版本时用） |
| `--reset-admin-token` | 重置管理令牌 |

**幂等**：可重复执行。目录/venv/unit 只更新不重建；nginx 配置先备份（`.bak.<时间戳>`）后覆盖。
环境变量可覆盖默认值：`DOMAIN`、`INSTALL_ROOT`、`DATA_DIR`、`WEB_ROOT`、`SERVER_PORT`、`PYBIN`。

### 0.4 部署后

```
站点      https://platform.ai2050.com
中心服务  http://127.0.0.1:5001        (systemd: ai2050-server)
前端文件  /var/www/ai2050
数据目录  /var/lib/ai2050/distributed
管理令牌  /etc/ai2050/admin_token
日志      journalctl -u ai2050-server -f
```

---

## 1. 手动部署（排障用）

### 1.1 基础环境（CentOS 7/8/9 Stream）

```bash
dnf install -y python3.11 python3.11-pip nginx rsync
# CentOS 7 需先启用 EPEL 并用 python3
sudo useradd -r -s /sbin/nologin ai2050
sudo mkdir -p /opt/ai2050/Ai2050-OpenOne /var/lib/ai2050/distributed /var/www/ai2050
sudo chown -R ai2050:ai2050 /opt/ai2050 /var/lib/ai2050
```

### 1.2 代码与依赖

```bash
cd /opt/ai2050/Ai2050-OpenOne
python3.11 -m venv /opt/ai2050/venv
/opt/ai2050/venv/bin/pip install --upgrade pip
# torch 单独装 CPU wheel（服务器通常无 GPU）
/opt/ai2050/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/opt/ai2050/venv/bin/pip install -r requirements-server.txt

# 校验（必须在仓库根执行）
cd /opt/ai2050/Ai2050-OpenOne && /opt/ai2050/venv/bin/python -c \
  "import transformer_lens, fastapi, torch; print('OK')"
```

> **`transformer-lens` 无需 pip 安装** —— 仓库自带 `transformer_lens/` 源码目录，
> `server.py` 从仓库根导入即可。**必须在仓库根目录启动**，否则 `import transformer_lens` 失败。

### 1.3 systemd

```bash
export ADMIN_TOKEN=$(python3 -c "import secrets;print(secrets.token_hex(24))")
sudo sed "s/change-me-to-random-hex/${ADMIN_TOKEN}/" \
     deploy/ai2050-server.service > /etc/systemd/system/ai2050-server.service
sudo systemctl daemon-reload
sudo systemctl enable --now ai2050-server
systemctl status ai2050-server
journalctl -u ai2050-server -f
```

单元关键项（已在文件内固化）：
- `AI2050_SKIP_MODEL_LOAD=1` —— **无 GPU / 无模型权重**即可提供全部图谱接口（实测 import 5.0 s、362 路由）。
- `AI2050_HOST=0.0.0.0` / `AI2050_PORT=5001`
- `ProtectSystem=full` + `ReadWritePaths` 放行 `/var/lib/ai2050/distributed` 与
  `ai2050_research_os/generated`（research_workspace sqlite 需要写）。
- 改端口只需同时改 `Environment=AI2050_PORT` 与 `deploy/nginx-ai2050.conf` 的 upstream。

### 1.4 nginx

```bash
sudo cp deploy/nginx-ai2050.conf /etc/nginx/conf.d/ai2050.conf
sudo rsync -a frontend/dist/ /var/www/ai2050/     # 若服务器上已 build 可跳过
sudo nginx -t && sudo systemctl reload nginx
```

站点配置要点：
- `root /var/www/ai2050` + `try_files` 回退；`/assets/` 长缓存（文件名含内容哈希）。
- `/api/` 反代 `127.0.0.1:5001`，**同源 → 无 CORS**，`client_max_body_size 2048m`，
  `proxy_read_timeout 600s`（registry 实时派生较慢）。
- `= /healthz` 供部署脚本探活。

### 1.5 防火墙 / SELinux

```bash
sudo firewall-cmd --permanent --add-port=80/tcp
sudo firewall-cmd --permanent --add-port=443/tcp
sudo firewall-cmd --reload
sudo semanage port -a -t http_port_t -p tcp 80 || true
```

### 1.6 HTTPS（certbot）

```bash
sudo dnf install -y certbot python3-certbot-nginx
sudo certbot --nginx -d platform.ai2050.com --redirect
# 证书续期定时器：systemctl list-timers | grep certbot
```

---

## 2. 验证

```bash
# 本机
curl -s http://127.0.0.1:5001/health                        # 中心服务
curl -s http://127.0.0.1/healthz                            # nginx 探活
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1/    # 前端 200
curl -s http://127.0.0.1/api/ai-rnd/atlas/registry | head -c 200

# 公网
curl -s -o /dev/null -w '%{http_code}\n' https://platform.ai2050.com/
```

---

## 3. 分布式平台（可选，:5010）

```bash
sudo sed "s/change-me-to-random-hex/${ADMIN_TOKEN}/" \
     deploy/ai2050-distributed.service > /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now ai2050-distributed
```

节点接入（任意机器，Windows/Linux 皆可；`node_agent.py` 仅需标准库）：

```bash
python deploy/node_agent.py --server http://platform.ai2050.com:5010 register --name gpu-box-1 --gpu "RTX 5080"
python deploy/node_agent.py --server http://platform.ai2050.com:5010 run --once   # 冒烟
```

> 冒烟结果 `kind=smoke`，与真实结果在 `/api/results` 中可辨 —— 诚实口径，不冒充实测。

---

## 4. 已知限制（诚实口径）

| # | 限制 | 表现 | 说明 |
|---|---|---|---|
| L1 | **图谱数据默认为空** | `/api/ai-rnd/atlas/registry` 返回空 registry | 实验数据被 `.gitignore` 排除（`/tests/glm5/result/` 等），clone 后不含。**需另行 rsync** `research/gpt5/atlas/`。这是设计好的诚实空态，**不是故障**。 |
| L2 | 三个 toy 端点在 Linux 无数据 | `/training/status`、`/toy_experiment/metrics`、`/toy_experiment/ricci_metrics` 返回空结构 | `server.py:928/2339/2352` 硬编码 `d:\develop\...`，但**有 `os.path.exists` 守卫**，Linux 上优雅降级不崩溃。 |
| L3 | 5002/5003/5004/8000 不在本部署内 | 前端对应面板显示「未接入」 | RDC 分片研究服务与训练服务是独立开发进程；nginx 未反代 → 404 → 前端显示空态，不伪造数据。 |
| L4 | 无 GPU 加载路径 | 模型类面板不可用 | `AI2050_SKIP_MODEL_LOAD=1` 关闭本地模型加载。要在服务器跑模型需设 `AI2050_SKIP_MODEL_LOAD=0` 并挂载权重。 |
| L5 | main.js 2.45 MB（gzip 715 KB） | 首屏偏大 | 已分 chunk（annotation/x/main），但未做路由级懒加载；后续可 `manualChunks` 优化。 |

---

## 5. 运维

```bash
systemctl status ai2050-server
systemctl restart ai2050-server
journalctl -u ai2050-server -f                # 实时日志
journalctl -u ai2050-server --since "1h ago" -p err   # 只看错误

# 前端更新（无需重启后端）
cd /opt/ai2050/Ai2050-OpenOne/frontend && npx vite build
rsync -a frontend/dist/ /var/www/ai2050/ && nginx -t && systemctl reload nginx

# 备份 = 打包数据目录（sqlite + 内容寻址对象库，可增量）
tar czf /backup/ai2050-dist-$(date +%F).tar.gz -C /var/lib/ai2050 distributed
```
