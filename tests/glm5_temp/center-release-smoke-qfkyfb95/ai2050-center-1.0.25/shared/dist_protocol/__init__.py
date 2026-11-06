# -*- coding: utf-8 -*-
"""AI2050 分布式测试协议 v3（dist_protocol）。

把「大量语言模板 → 分布式测试 → 合并统计 → 候选结构 → 留出预测」做成可执行协议：

- ``grammar``    生成式模板：文法框架 × 填充词表 × 因子水平 → 语料；按填充词/框架
                 确定性切分 discovery / confirm / lockbox / lockbox_frame；功效近似估算。
- ``suffstats``  可合并充分统计量（每 split×条件格×层×维 的 n、Σx、Σx²），跨节点精确合并，
                 派生 η² 分解、条件方向、跨划分复现与 RDM 一致性。
- ``canary``     固定金丝雀输入的 HiddenState 摘要与漂移判定（跨 GPU / dtype / 快照一致性）。
- ``pipeline``   中心节点侧：结果文件解析、按桶合并、覆盖矩阵、审计自动检查、总览漏斗、
                 预测记分板。只读结果库，不创建 Evidence。
- ``runner_v3``  Worker 端自包含 runner（随模板包下发，不 import 本包）。

本包只依赖 Python 标准库与 numpy；中心节点（deploy/distributed_service.py）与测试共用。
"""

PROTOCOL_VERSION = 'dist_protocol.v3'
