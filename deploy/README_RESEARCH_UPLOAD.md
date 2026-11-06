# 正式中心研究库升级与数据上传

`server.ai2050.com` 的程序升级和研究数据登记分开进行。发布包采用现有中心的版本化升级脚本，保留中心数据库、数据目录和管理配置。已登记成果可在 3D 工作台公开回查。

把发布包、同名 `.sha256` 和 `upgrade_center.sh` 上传到服务器同一目录。若中心沿用默认的 `ai2050-center` systemd 单元，在该目录运行以下命令。使用不同单元名时，将实际单元名通过 `SERVICE` 传入。

```bash
sudo bash upgrade_center.sh ai2050-center-1.0.28.tar.gz
curl -fsS https://server.ai2050.com/api/health
curl -fsS https://server.ai2050.com/api/traces/research-space
```

第一条公开检查应返回 1.0.28，第二条应返回 `research_space_index.v1`。打开站点时应由该中心提供同源客户端。若 nginx 另外托管静态前端，按现有配置把新程序目录中的 `frontend/dist` 发布到该站点的静态目录。

管理员登录新版客户端后，在「研发 → 实验与队列 → 上传研究包」选择 `ai2050-research-*.tar.gz`，看到校验通过后点击「登记到中心」。客户端沿用现有研究库 API，按清单顺序提交原始 JSON。中断后可重试，已登记对象按内容哈希跳过。公开索引和报告可读后，页面会给出 3D 研究成果入口。

需要用脚本发布时，在已获授权的机器上从当前服务的程序目录运行 `deploy/research_ingest_bundle.py`，使用数据包的实际路径。

```bash
# 从服务器读取已有管理令牌文件。端口和令牌路径以当前部署配置为准。
sudo python3 deploy/research_ingest_bundle.py /上传目录/ai2050-research-factor-steer.tar.gz \
  --center http://127.0.0.1:5010 --admin-token-file /etc/ai2050/admin_token

# 在已将管理会话注入 AI2050_RESEARCH_SESSION 的环境执行。
python3 deploy/research_ingest_bundle.py ai2050-research-factor-steer.tar.gz \
  --center https://server.ai2050.com
```

上传器先核对完整包的 SHA256 对象，再按引用顺序上传并复查公开索引。网络中断后可用同一命令重试。新版客户端打开 3D 研究成果时直接读取中心索引和证据。已下载的研究库附加包供离线回查，由后台快照任务刷新。
