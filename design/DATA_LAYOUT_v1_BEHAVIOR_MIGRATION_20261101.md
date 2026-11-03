# 旧节点结论级文件迁移记录 [2026-11-01]

## 背景

`make_ent_lexicon_v2.py`（第 117 行）与 `estimate_ent_gate.py` 直接读取
`~/.ai2050_agent/tasks/*/outputs/behavior.json`，这是 ENT 词表 v2 与行为门估计的输入。
该目录属于**已退役**的旧节点客户端（最后活动 2026-10-19），其中 13.33G 为 npz 分片。

## 操作

- 源：`C:/Users/Admin/.ai2050_agent/tasks`
- 目标：`D:/ai2050_native/_behavior/_legacy_agent/tasks`（保留 `<task_id>/outputs/` 相对结构）
- 保留类型：`.json`（结论级记录）、`.py`（task 脚本，13MB）
- 不迁移：`.npz`（逐 token 分片，13.33G，服务器已有上传副本）

## 结果

- 迁移文件：**1285 个，116.1 MB**（其中 behavior.json **209** 个）
- 跳过（npz）：418 个
- 逐文件大小校验失败：**0** 个
- 复制异常：0 个
- 耗时：3.2s

## 受影响脚本的路径变更

| 脚本 | 原路径 | 新路径 | 方式 |
|---|---|---|---|
| `scripts/make_ent_lexicon_v2.py` | `~/.ai2050_agent/tasks/{t}/outputs/behavior.json` | `D:/ai2050_native/_behavior/_legacy_agent/tasks/{t}/outputs/behavior.json` | 新增模块级常量 `BEHAVIOR_ROOT`（可用环境变量 `AI2050_BEHAVIOR_DIR` 覆盖），旧路径作为回退 |
| `scripts/estimate_ent_gate.py` | `--behavior` 命令行参数 | 同上（调用时传新 glob） | 无需改代码，仅更新文档字符串 |

生成时间：2026-11-01 10:48:28
