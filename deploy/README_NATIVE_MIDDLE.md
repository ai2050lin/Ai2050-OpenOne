# Qwen3-4B / Qwen3-14B B18–B28 原生单元档案

本轮范围为零基 B18–B28，包含两端。4B 为 107,008 个单元，14B 为 191,488 个单元。全量观察、响应预测和有限写入干预分别显示；正式语义确认数仍为 0。报告保留失败预测、无效范数对照及副作用。

运行版本：

- `QWEN4-GOODFIRE-B18-B28-20261113-v2`
- `QWEN14-GOODFIRE-B18-B28-20261113-v2`

新运行采用与原始 Hugging Face 加载一致的 CPU float32 RoPE 频率初始化再送入 CUDA。旧运行的 CUDA 初始化是另一执行版本，不因模型权重相同而逐位合并。具体误差和修正记录见研究报告。

## 本机查看

从项目目录启动 `tests/glm5/goodfire_middle_server.py`，浏览 `http://127.0.0.1:5035/workbench`。成果库选择对应运行后打开完整报告，或检索 `B28:U9727` / `B28:U17407` 并定位到当前空间。可查全量层阵列、两端语境、完整 gate/up/down 参数、相邻层全部关系、预测对照和逐样本干预。

这个本机入口只读真实归档，禁用模型运行工具，不自动加载 GPU。尚未审计完成的研究包返回缺数据提示。

## 手动部署

本轮不自动更新公网。中心代码、研究登记包、原生包必须配套部署。安装包中的客户端版本为 1.0.15；实际中心版本与包哈希以交付清单为准。

1. 更新本轮中心发布包，保留现有 `AI2050_DIST_DIR` 和旧研究数据。
2. 为每个模型导入原生包。文件名以交付清单为准，例如：

```bash
python deploy/native_mlp_pack.py verify --archive qwen4-b18-b28-native.tar.gz
python deploy/native_mlp_pack.py import --archive qwen4-b18-b28-native.tar.gz --data-dir /srv/ai2050-data
python deploy/native_mlp_pack.py verify --archive qwen14-b18-b28-native.tar.gz
python deploy/native_mlp_pack.py import --archive qwen14-b18-b28-native.tar.gz --data-dir /srv/ai2050-data
```

3. 按 `README_RESEARCH_UPLOAD.md` 导入对应的 `qwen4-b18-b28-research-upload.tar.gz` 和 `qwen14-b18-b28-research-upload.tar.gz`。登记包只含研究对象，不能代替原生包。
4. 设置中心 `AI2050_DIST_DIR=/srv/ai2050-data`，重启中心。逐一读取 `/api/research/native-mlp?model=qwen4`、`?model=qwen14`，确认运行 ID，然后从客户端检索两模型最后一个单元、参数末坐标和干预末样本。

原生包安装到 `AI2050_DIST_DIR/native_atlases/<RUN>`，逐文件验证 SHA256，拒绝覆盖不同版本或损坏文件。失败暂存自动清理；已有相同且完整版本可幂等重试。重试前保留错误日志，损坏的正式目录须显式修复。磁盘需容纳压缩包、解包数据及至少 20 GiB 空余；Windows 项目数据禁止默认回落 C 盘。

源数据在 `tests/glm5/result/goodfire_middle_20261113_v2/{qwen4,qwen14}`，可用以下命令重建传输包，执行前会检查实际空间：

```bash
python deploy/native_mlp_pack.py build --source tests/glm5/result/goodfire_middle_20261113_v2/qwen4 --out qwen4-b18-b28-native.tar.gz
```

14B 同理。原生数据作为可复用档案保留，复查日期 2026-12-13。安装后查询不需要模型权重或 GPU；返回的权重向量已经包含在研究包中。
