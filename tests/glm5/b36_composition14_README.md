# Qwen3-14B B36 固定32单元：措辞、状态与局部传播

本阶段沿用 `branch_reuse14_20261113_v3/frozen-selection.json` 的 B36 参数候选32单元、匹配32单元和 B39 八个根，没有在新确认集重新选择。实际结论以结果目录 `b36_composition14_20261113_v3` 的报告及准入记录为准。所有模型调用使用本地原始 BF16 Qwen3-14B、CUDA 逐块执行及共享 `GpuLease`；没有部署正式服务。

块号为零基B0–B39：h37是B36输出的残差，h40是B39输出、最终RMSNorm之前的残差，hs0是输入embedding。MLP单元索引与残差坐标索引分属17,408维和5,120维空间，不可混称。

## 材料与范围

- 原创虚构材料，生成文法、填充家族、实际字符串、token ID 与读出位置全部保存。
- 同一正文中同时给出操作步骤和具体记录值；请求来源、技术/普通标题、操作序列和选项符号分别控制。标题与请求来源的 token 长度配平。
- 发现集只拟合 A/B 原子操作；新家族与新句式的确认材料包含原子、AB/BA 和 ABA/BAB。两步顺序对照词语多重集相同；三步也改变操作次数。
- v1、v2 试跑失败均保留。v3 在读取确认前缩小范围：仅数值原子计算和两个分支的记录读取进入候选内部采集。程序组合失败仍需行为复测，不进入内部机制正例；记录题中的复杂背景指令不代表模型执行了组合。
- 80个确认家族/分支，共160个。标题、格式、顺序和干预臂复用家族，不能增加独立 N。格式仅20家族/分支。

## 冻结与分析

1. `trial-plan.json` 绑定采集源码、快照与材料；`scope-plan.json` 记录 v2→v3 的事前部分准入；v3 只读复用 v2 试跑。
2. `analysis-plan.json` 规定行为门、指标、比较和功效。`selection.json` 绑定仅由发现集拟合的14个岭回归预测器。
3. `execution-freeze.json` 绑定实际前向、数学算子、统计和输出评价源码及数值试跑。冻结后不得改这些文件继续确认。
4. 每一来源/分支/深度须两个标题都通过试跑和确认行为门，才进行内部分析。缺格不能由成功子集补过联合门。
5. 自然状态预测分别使用组 A、完整 h37、h37 与前缀均值、词汇基线及平均 embedding。全词表预测全部带入实测 h37，属于条件化晚层预测。
6. 13个原模型干预臂包括0/1/2倍、匹配单元组、等实际写入范数对照、注意力/MLP路径固定、B39根及匹配根恢复。组作用不等于单成员必要性。
7. 局部算子使用自然状态和权重预测有限改动的传播；编辑后的下游状态只用于评分。FP32平滑导数与BF16舍入分别记账，不能把已知数学恒等式称为新语义规律。

## 命令与恢复

在仓库根目录用 `.venv/Scripts/python.exe -B -X utf8 -u` 执行脚本。历史 v1/v2 以各自源码归档及日志为准。已完成冻结或选择不应再次执行。

```text
tests/glm5/b36_composition14_scope_v3.py
tests/glm5/b36_composition14_collect.py freeze
tests/glm5/b36_composition14_analysis.py plan
tests/glm5/b36_composition14_campaign.py collect --split discovery
tests/glm5/b36_composition14_analysis.py select
tests/glm5/b36_composition14_campaign.py tail-pilot
tests/glm5/b36_composition14_tail.py freeze
tests/glm5/b36_composition14_campaign.py collect --split confirm
tests/glm5/b36_composition14_analysis.py scope
tests/glm5/b36_composition14_analysis.py analyze
tests/glm5/b36_composition14_campaign.py collect --split format
tests/glm5/b36_composition14_analysis.py scope --split format
tests/glm5/b36_composition14_campaign.py tail --split confirm
tests/glm5/b36_composition14_tail_analysis.py
tests/glm5/b36_composition14_campaign.py tail --split format
tests/glm5/b36_composition14_tail_analysis.py --split format
tests/glm5/b36_composition14_campaign.py output --split confirm
tests/glm5/b36_composition14_behavior_only.py run
tests/glm5/b36_composition14_behavior_only.py finish
tests/glm5/b36_composition14_readout_diagnostic.py plan
tests/glm5/b36_composition14_readout_diagnostic.py run
tests/glm5/b36_composition14_path_report.py
tests/glm5/b36_composition14_report.py heatmaps
tests/glm5/b36_composition14_cards.py
tests/glm5/b36_composition14_report.py report
tests/glm5/b36_composition14_audit.py audit
node tests/glm5/b36_composition14_browser.cjs
tests/glm5/b36_composition14_closeout.py archive
tests/glm5/b36_composition14_closeout.py cleanup
tests/glm5/b36_composition14_memo.py
tests/glm5/b36_composition14_audit.py manifest
tests/glm5/b36_composition14_delivery_verify.py
```

格式干预仅在其准入清单非空时运行。采集与干预按已完成分片的行 ID 与 SHA 恢复，不删除结果重跑；GPU租约冲突只排队，不强行结束其他研究。日志 `progress.jsonl`、逐阶段 `.log` 与 `pipeline.log` 可查看进度、预计时间、CUDA和DXGI共享显存。原生字段是保存BF16位模式的无损压缩uint16，必须用 `unbits` 解码。

被排除条件的 `behavior-only` 评分引擎在阶段末尾统一保存，没有原生采集的分片恢复接口；若中断，应保留日志并核对冻结版本后重新执行该评分阶段，不能将重复计算计作额外样本。

本轮协议回归曾因默认扫描真实研究库而在隔离目录复制约2.93 GiB，触发磁盘门并中断确认采集。只删除已核对归属的测试复制件，864条已提交确认输入保留。后续通过现有 `AI2050_RESEARCH_DIR` 使用空测试目录，在允许本机回环的环境中运行相同315项测试；314通过，既有SOURCE-BINDING语言映射失败保留。这不等于完成真实全库打包验收。相关停止、脱敏、清理、剩余空间预算和续跑回执均在结果目录；科研源码与阈值未改变。

## 复核入口

同题ABC/XYZ比较只使用格式面板对应的20个原家族，不能与ABC全80家族直接比较。`readout-diagnostic` 是看到反向后新建的事后代数诊断，复用现有终态，并非新增确认或有效自然题面上的交叉行为。对角BF16评分须逐位复现，对称分摊恒等式只解释代码读出与状态变化的数值关系，不认定唯一因果中介。

`report.html` 展示行为失败题、主干预、状态预测、局部效应预测、路径对照及原生全坐标图。`feature-cards.json` 与 `mechanism-graph.json` 分开标记观察、计算来源、预测、近似归因和原模型组干预。报告中的数值始终回指逐条产物，不作正式中心晋级。

`audit.json` 核对模型文件、分词、家族划分、原生数组、逐位无操作及前缀一致性；技术审计不等于独立科学复现。冻结源码、交付源码和所有SHA清单保留。v2试跑仍是v3的原生数据来源，不可清理。只清理本轮已结束且可重建的临时目录；项目数据写入D盘，始终保留20 GiB。
