# Qwen3-14B 来源绑定迁移对照

用本机 Qwen3-14B 原始 BF16 快照执行。CPU 读取权重、分组保留中间状态，CUDA 逐块执行。所有运行复用第 5 版冻结的 16 个读取头。

## 四因素材料

`tests/glm5/result/source_binding_transfer_factors_v1/design.json` 保存实验设计、模型快照、头索引、代码与材料哈希。`holdout-materials.json` 保存 64 个预生成家族。实际运行取前 24 个家族，`holdout-gate.json` 与 `holdout-edits.json` 均登记执行范围。

同一家族的四条记录和答案 token 固定。每个相邻条件只切换一个因素。

- 正文英语／中文。共用同一份符号表、拉丁字母名字与颜色答案；名字位置、前缀总长和答案提示匹配。
- 问题名字一／两个 token。保持可解码拼写不变，以词表中的两段替代单 token。两段使用同一 RoPE 位置，其他 token 的位置保持固定。顺序因果掩码仍区分两段。
- 名字近／远。把同一条十 token 说明句移到问题前或后，整条输入的 token 集合、总长与提示保持固定。
- `Answer:`／`答案：`。最后两个 token 改变，正文与候选保持固定。

每个条件每家族有 8 个基础题。每题按 3 个替换位点、3 个入口块和 4 个供体关系做 36 次宽编辑。第 4 个供体关系固定问题并改变表格布局，另外 3 个关系同时测窗口全头和同数随机头。

## 原题答案提示复测

`tests/glm5/result/source_binding_transfer_cue_v1/design.json` 固定历史材料哈希与供体重算入口。两种原题各取 16 个已开封家族。每题评估 `Answer:`、`Answer：`、`答案:`、`答案：`，候选答案、正文、名字位置和供体保持固定。每种题、每种提示有 256 个基础题与 9,216 次宽编辑。

报告的 `same_target` 只使用两个条件下宽编辑都改变原答案、且选中同一新记录的配对。`gain` 使用全部编辑，表示 H16 总一致率减去「答案保持原样」基线。两个口径共同展示。历史复算差异单列，提示对照统一使用同一运行器。

## 运行与复核

从仓库根目录用 `./.venv/Scripts/python.exe -B -u` 执行以下命令的参数部分。已有产物可直接进入分析，`design` 只用于空结果目录。

```text
tests/glm5/source_binding_transfer_factors.py design
tests/glm5/source_binding_transfer_factors.py gate --split pilot
tests/glm5/source_binding_transfer_factors.py edits --split pilot
tests/glm5/source_binding_transfer_factor_campaign.py equivalence --group 1 --families 24
tests/glm5/source_binding_transfer_factor_campaign.py gate --group 8 --families 24
tests/glm5/source_binding_transfer_factor_campaign.py edits --group 2 --families 24
tests/glm5/source_binding_transfer_factor_analysis.py
tests/glm5/source_binding_transfer_cue.py design
tests/glm5/source_binding_transfer_cue.py gate
tests/glm5/source_binding_transfer_cue.py run
tests/glm5/source_binding_transfer_cue_analysis.py
tests/glm5/source_binding_transfer_original_distance_analysis.py
```

报告由已保存的候选分数、逐条预测及供体索引复算。文件哈希和样本 ID 在汇总前核对。`progress.jsonl` 和运行日志保留数值检查、显存与内存、异常及重试信息。
