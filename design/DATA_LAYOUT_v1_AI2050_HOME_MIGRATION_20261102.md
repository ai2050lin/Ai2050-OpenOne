# .ai2050 迁移记录：C 盘 → D:\ai2050_native [2026-11-02 06:44–06:50]

## 背景

`C:\Users\Admin\.ai2050`（21G）是节点服务与原生全场在 C 盘的工作目录。按「正式数据上服务器、
本地唯一数据根为 `D:\ai2050_native`」的布局原则，用户要求全量迁移并删除该目录。

## 路径机制（迁移可行性的依据）

- `deploy/node_agent.py:390`：`_home()` = 环境变量 `AI2050_HOME`，缺省 `~/.ai2050`
- `deploy/node_service.py:1204`：`default_home()` = `AI2050_HOME/node_service`，且支持 `--home` 参数
- `deploy/node_agent.py:301`：`snapshot_cache.json` 位于 `AI2050_HOME/`
- `shared/dist_protocol/runner_v3.py:155`：`native_root()` = 环境变量 `AI2050_NATIVE_DIR`，缺省 `~/.ai2050/native_fields`

## 迁移布局

| 源（C 盘） | 目标（D 盘） | 体积 |
|---|---|---|
| `.ai2050\native_fields\<家族>\...` | `D:\ai2050_native\<家族>\...`（与主库合并） | 9.6G |
| `.ai2050\node_service\...`（凭据/配置/历史/work 11G） | `D:\ai2050_native\_agent_home\node_service\...` | 10.2G |
| `.ai2050\snapshot_cache.json` | `D:\ai2050_native\_agent_home\snapshot_cache.json` | 30K |

## 环境变量（用户级，已写入注册表并回读确认）

```
AI2050_HOME      = D:\ai2050_native\_agent_home
AI2050_NATIVE_DIR = D:\ai2050_native
```

## 执行与校验

- dry-run：3608 文件 / 20.03GB，**0 个大小冲突**（36 个目标已存在且同大小 → 判定为真重复跳过）
- 复制：3559 文件 / 19.11GB 复制 + 逐文件大小校验，0 失败
- 终检：native_fields 源 2178 文件在 D 侧 **0 缺失 0 差异**；node_service 源 1380 文件 = 目标 1380 文件（10.22GB 完全一致）
- 删除：`C:\Users\Admin\.ai2050` 整体删除，路径已不存在，**C 盘 35.51 → 55.33 GB（+19.8G，真实释放）**

⚠️ 说明：本次删除未进回收站而是直接释放空间（与批 1 行为不同，钩子机制不完全一致）。因全部内容
已先复制到 D 盘并逐文件校验，无数据损失。若需恢复原目录，从 `D:\ai2050_native` 反向复制即可。

## 注意事项

1. 环境变量只对**新进程**生效。若节点客户端以旧会话/服务方式自启，会在 C 盘原位重建 `.ai2050`——
   重启会话或重新登录后即使用新路径。
2. `node_service\work` 的 11G 历史任务工作副本现随迁至 D 盘；节点下次接任务会在新 home 下继续工作。
3. 节点服务最后一次任务活动为 2026-10-28 17:34（T-cc7cb47711，done），迁移时无进程占用。
