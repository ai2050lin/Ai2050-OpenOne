# Qwen3-14B 临界分差题的屏蔽与恢复

在原四种中文问句模板上生成新队名、人名和名单。正式名单明确有效，临近问题加入标为过时或无效的草稿、错误示范或另一来源的提示，给错误候选增加竞争。原问句末段与 `Answer:` 保持固定。每题的四个答案均是一个 token。

`screen` 对全部新题执行原模型。`freeze` 在读取任何干预结果前，保留原运行四选一正确、正确答案比分数最高的错误选项高 (0<m\leq1) logit 的全部题。恰好平分的题排除。全词表第一名另外记录。

| 操作 | 含义 |
| --- | --- |
| `mask_key` / `mask_adjacent` | 只屏蔽上轮冻结的位置或末段的循环后继位置。若冻结 token 是末尾问号和换行，后继按上轮规则回到末段第一个 token。 |
| `mask_all` | 屏蔽最后一个名字之后到问句结束的所有位置。 |
| `reopen_key` / `reopen_adjacent` | 在全段注意力屏蔽中，只重新开放关键位置或对照位置。 |
| `zero_all` | 把末段各边的 V 设为零，位置仍参加注意力权重计算。 |
| `restore_key` / `restore_adjacent` | 从全段 V 清零恢复一个原生位置，保留其他位置清零。 |
| `restore_all` / `self_all` | 全段原生 V 恢复与独立本身复制控制，两者逐条完全相等。 |

所有干预只触及原固定 16 个头在第 23–27 块从作答位置发出的边。保留模型原来的文字输入、名字读取与其他计算。对临界题保留无操作、自交换与完全恢复对照，另外报告这些对照一致的子集。`report.json` 中可以逐题追踪原选项、改错和恢复。

## 从仓库根目录复算

```powershell
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_low_margin.py design
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_low_margin.py screen
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_low_margin_analysis.py freeze
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_low_margin.py test
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_low_margin_analysis.py analyze
```

实际提示、候选、冻结队列、逐条分数、哈希及日志保存在 `tests/glm5/result/source_binding_qkv_followup_v1/low-margin/`。已完成分片可续用。`design` 用于空结果目录。

第二批增强干扰材料使用与第一批分开的名字、种子和干扰模板。它在只查看第一批原模型分数后的低入选率后冻结。依次用 `source_binding_qkv_low_margin_expanded.py` 的 `design`、`screen`、`freeze`、`test`、`analyze` 命令运行，产物在同级的 `low-margin-expanded/`。其 `wrapper.json` 记录实际组数、干扰模板和种子。

第三批在前两批低入选率明确后冻结，给同一队伍同时展示正式名单和互相冲突的作废名单。使用 `source_binding_qkv_low_margin_conflict.py` 的同一组命令，输出到 `low-margin-conflict/`。

用 `source_binding_qkv_low_margin_flip_replay.py low-margin`（或传另外两个目录名）复跑冻结队列。这个对照改用物理相邻位置，优先右侧，关键位置在问句末尾时用紧邻左侧。同时单独重新开放此前唯一被屏蔽的关键边，检查因屏蔽改错的题能否逐条恢复。跨批汇总用 `source_binding_qkv_low_margin_compare.py --with-conflict`。
