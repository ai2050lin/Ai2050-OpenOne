# 正式中心 502 排查与主数据包更新

适用于 `server.ai2050.com` 的 nginx 与 `ai2050-center` systemd 部署。项目模板将 nginx 转发到 `127.0.0.1:5010`。本次外部检查中，正式中心首页、健康接口和主数据包均由 nginx 返回 502。具体原因由服务器日志和环回探测确定。

## 一、上传并核对文件

把 `ai2050-center-1.0.32.tar.gz`、对应 `.sha256`、`upgrade_center.sh` 和 `recover_center_lang_map.sh` 上传到服务器同一个目录。Windows 1.0.11 安装器供桌面用户下载。

```bash
sha256sum -c ai2050-center-1.0.32.tar.gz.sha256
sudo bash recover_center_lang_map.sh diagnose
```

如果实际服务名称或环回端口有差异，按服务器上核实的值传入 `SERVICE` 和 `PORT`，例如 `sudo SERVICE=ai2050-center PORT=5010 bash recover_center_lang_map.sh diagnose`。诊断只读，日志留在服务器查看。

| 环回 `/api/health` | 域名 `/api/health` | 查哪里 |
| --- | --- | --- |
| 无连接，且对应端口无人监听 | 502 | 看 `journalctl -u ai2050-center` 最早的失败行。检查服务进程、Python 路径、缺失文件、磁盘或 OOM。 |
| HTTP 200 | 502 | 比较 nginx 错误日志的 `upstream` 地址与实际监听端口，再检查生效的 `server_name` 和 `proxy_pass`。 |
| 本机 nginx HTTPS 为 200 | 公网为 502 | 核对域名解析、外部代理或负载均衡是否指到这台主机。 |

若 nginx 错误行是 `connect() failed (111: Connection refused)`，先核对错误中显示的 upstream 对应哪个服务。确认配置需要修改后，先运行 `sudo nginx -t`，通过再执行 `sudo systemctl reload nginx`。保留已有 TLS 配置。

## 二、升级中心并刷新主数据包

确认当前服务使用已有持久数据目录，且应用故障可以通过升级解决，或服务本身健康、需要部署本版时，在上传目录执行：

```bash
sudo bash upgrade_center.sh ./ai2050-center-1.0.32.tar.gz
curl -fsS http://127.0.0.1:5010/api/health
sudo bash recover_center_lang_map.sh refresh
```

升级脚本备份现有数据库和 systemd 配置，重启后用环回健康接口核对版本。若已运行 1.0.32，可从健康检查开始。

刷新脚本读取运行中服务使用的数据目录和 Python，短暂停止中心，备份现有主数据包，再以服务用户重建。它校验包内图谱与词层、版本与元数据一致，随后启动中心。重建或校验失败时恢复这次备份的主包并尝试启动服务。附加包和现有账号、研究数据目录继续使用。输出中的 `Existing main package backup` 给出备份位置。

## 三、验收

```bash
curl -fsS https://server.ai2050.com/api/health
curl -fsS https://server.ai2050.com/api/lang/map | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["schema"], d["stats"])'
curl -fsS https://server.ai2050.com/api/platform/config | python3 -c 'import json,sys; print(json.load(sys.stdin)["datapack"]["version"])'
```

本版图谱应有 3 个领域、80 个语言项、477 个词。客户端连接正式中心后，在左下角点「更新数据」，进入「空间 → 语言图谱 → 结构图谱」，搜索「名词语义类别」并打开词层。更新后的数据可在离线时查看。
