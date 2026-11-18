# AI2050 正式环境上传包（2026-11-08）

版本组合：Windows 客户端 **1.0.8**，中心服务/网页客户端 **1.0.29**，研究主线记录 **序列4**。

## 文件与用途

- `AI2050-Setup-1.0.8-x64.exe`：Windows x64 客户端安装程序，上传到客户端下载位置。不是 Linux 服务器安装程序。
- `ai2050-center-1.0.29.tar.gz`、对应 `.sha256`、`upgrade_center.sh`：中心程序与同源网页客户端。保留正式中心已有的数据目录、账号和配置。
- `ai2050-data-20261108-mainline4.tar`、对应 `.sha256`：客户端主数据包、15个附加包、H3/H3b记录及两份已发布研究上传包。数据源本来存在的旧快照/报告保留原口径，只有主线相关索引刷新；不要把缓存生成时间当成新实验时间。
- `apply_data.py`：校验并合并客户端缓存，默认只验不写；不会替换研究库或账号数据库。新主线记录由新版中心首次读取时幂等归档。
- `SHA256SUMS.txt`、`RELEASE_MANIFEST.json`：所有交付文件的大小、哈希和版本映射。

本包不含模型权重、私钥/令牌、账号/会话、未开封确认材料正文，或约85 GiB的本机原生张量归档。正式中心已有的原生数据继续使用；新建空中心需要另行迁移原生数据，客户端缓存不能代替这些原场。

Windows 安装器未配置代码签名；可使用本目录 SHA256 校验其完整性。本数据包属于展示缓存与已发布研究包，不是数据库和原场的全量迁移包。

## 上传后按顺序操作

先把同一发布目录里的文件上传到服务器同一个目录，进入该目录：

```sh
sha256sum -c SHA256SUMS.txt
sudo bash upgrade_center.sh
```

确认中心版本，查出**现有**数据目录（不要新建空目录替代）：

```sh
curl -fsS https://server.ai2050.com/api/health
systemctl show ai2050-center -p Environment --value | python3 -c 'import shlex,sys; print(next((v.split("=",1)[1] for v in shlex.split(sys.stdin.read()) if v.startswith("AI2050_DIST_DIR=")), "未找到：请核对服务配置中的数据目录"))'
```

把下面 `<现有AI2050_DIST_DIR>` 换成上一步实际路径。脚本先完整校验；加 `--apply` 才写缓存。替换的旧缓存会移动到数据目录下的 `client-data-backup-...`，用于回滚；账号和研究数据库不动。服务停机可避免后台缓存重建与部署写入竞争。

```sh
python3 apply_data.py ai2050-data-20261108-mainline4.tar --data-dir '<现有AI2050_DIST_DIR>'
sudo systemctl stop ai2050-center
sudo python3 apply_data.py ai2050-data-20261108-mainline4.tar --data-dir '<现有AI2050_DIST_DIR>' --apply
sudo systemctl start ai2050-center
```

若合并失败，脚本自动恢复本次替换的缓存；确认服务已重新启动。服务端更高版本的主线记录会阻止旧缓存回退。

## 验收

```sh
curl -fsS https://server.ai2050.com/api/research/mainline
curl -fsS https://server.ai2050.com/api/research-library/workpackages
curl -fsS https://server.ai2050.com/api/traces/research-space
```

主线 `revision` 与后两个接口的 `mainline_revision` 应一致。客户端先更新主数据包，再按需下载附加包；当前研发应显示 confirm 已开封、H3b 当前判据未支持，3D 可回查报告、曲线和候选组成员。界面升级不改变科学结论。

两份既有研究上传包位于数据包 `research_uploads/`，合并后也在 `<AI2050_DIST_DIR>/research_uploads/`。正式中心已入库时不需要重复上传；确实缺件时用客户端“上传研究包”协议补入，重复上传会去重。

Windows 安装器只打包生成和核验，本次不运行安装向导覆盖已有安装，也不自动发布 GitHub Release 或修改正式下载链接。请上传该安装器并更新你管理的下载入口。
