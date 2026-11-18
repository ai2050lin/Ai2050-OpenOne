# 来源绑定的 Q、K、V 与后续读出对照

从仓库根目录运行以下命令。模型为本机 Qwen3-14B BF16，CPU 读取原权重，CUDA 按块执行。留出运行可复用已完成的分片。

```powershell
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_followup.py design
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_followup.py pilot
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_followup.py holdout
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_followup_analysis.py
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_history.py
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_gap_positions.py
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_readout.py design
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_readout.py run
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_readout.py analysis
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_suffix_sites.py design
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_suffix_sites.py run
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_suffix_sites.py analysis
.\.venv\Scripts\python.exe -B -u tests/glm5/source_binding_qkv_suffix_compare.py
```

结果目录 `tests/glm5/result/source_binding_qkv_followup_v1/` 保存设计、实际提示 token、候选、供体索引、逐条四选一分数、每层注意力摘要、分片哈希、日志和汇总。英文颜色与中文动物各从 Phase 426 旧材料选择与 Phase 2921 提示对照不交叠的家族。材料在 Phase 426 已开放，本轮的 Q、K、V 组合为新干预。两种提示共享完全相同的供体、名字、候选和位置。

| 字段 | 操作 |
| --- | --- |
| `W` | 在第 23、24 或 25 块入口替换问题中来源名、实体名或两者的完整残差。 |
| `H` | 仅固定的 H16 在作答位置读取供体名字 K、V。 |
| `K` / `V` | 同一名字供体，只替换 K 或只替换 V。未编辑分量仍随本分支自然计算。 |
| `Q` | 只把 H16 的作答位置 Q 换为相反提示的原前向 Q。 |
| `QH` | 与 H 相同的供体 K/V，再仅将 H16 的 Q 换成相反提示 H 分支的 Q。两方向均测。 |
| `D` | 第 28 块入口给当前提示的原状态叠加相反提示下 H 减原状态的差，后续按当前提示计算。 |
| `Dself` | 同提示回放上述差，检查能否精确回到 H。 |
| `ALL` | 同一窗口内的所有头读供体名字。 |
| `Hpos` | 在 H 基础上只把供体名字 K 的旋转位置距离统一为 22。各名字内部 token 间距不变。 |
| `Hback` | 在 `Answer:` 的 H 基础上，让相同的头额外读取问题后缀的供体 K/V，排除名字与答案提示。 |

候选分差使用同一新记录减原记录。Q 的跨提示替换只改 Q 投影，保留当前提示残差。名字 K/V 的跨提示不变性逐位核对。第 28 块的跨提示回放固定的是有限差值，接收者残差和前缀按当前提示计算，所以比较包含后续计算和接收状态的交互。不同干预的作用可交互，报告的百分点和 logit 变化不解释成互斥的贡献比例。

`readout/` 另存每任务前 12 个家族、三个冻结编辑条件的后续计算对照。第 28 块入口交换完整的作答位置状态时，同时交换无编辑、H16 编辑两种状态。两种状态的分数差可用来检查在两份作答状态完全固定时，只换接收提示前缀的影响。另复算仅交换有限差值的结果。

`suffix/` 保存同一批 `Answer:` 条件下的问句区域对照。以名字 K/V 编辑为共同起点，分别加两个名字之间、最后一个名字之后的问句、仅回答格式要求、整个去名问句、全部后缀。评分使用主实验的宽编辑目标，另记录不同批形状下共同对照的选择差异。`manifest.json` 记录已完成的六个分片，`report.json` 汇总各臂，`region-comparison.json` 复算三段的直接配对比较和跨任务差距。`partial-coverage.json` 保留续跑前的进度快照。

运行依赖共享 `GpuLease`。如果提示设备已由其他研究使用，保留已完成的分片，待租约可用后从同一命令恢复。分析步骤只读已验证完成的清单。
