# -*- coding: utf-8 -*-
"""2026-10-07 日志追加：分布式平台 M1 落地（append-only）"""
import io

LOG = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-07.md'
text = u"""

## 分布式平台 M1 落地：服务器+Agent+前端接线（15:0x）

- 用户裁决：按 design/distributed_platform_plan_v1.md 完成修改，服务器部署 CentOS。
- **服务器**（`server/distributed_service.py`，24.7KB）：sqlite(WAL)+内容寻址 objects/；S2 模板注册（design_sha=规范化 corpus+runner sha，预注册冻结）+S3 注册/claim/租约 6h/心跳+心跳+超时回收+S4 init/chunk(1.5MB b64)/finish 逐文件 sha256 核验（design_sha 不符 422；单文件 64MB 上限防大张量）+S5 公开 summary/templates/results/news+S6 AGG-v0 分桶回显（禁跨桶平均）+S7 token 哈希存库。双重运行：挂载 server.py(:5001, 2 行 patch) 或独立 `python -m server.distributed_service`(:5010)。
- **种子模板**（dist_templates_seed.py + dist_runner_tm.py）：TM-01 语言最小对 24 项/TM-02 风格 16 项/TM-03 逻辑连接词 16 项；runner 末层残差流目标 token 采集→逐维 η²→条件方向 cos 矩阵→means.npz；无 torch 诚实降级 smoke。
- **节点 Agent**（`server/node_agent.py`，stdlib only）：register→run[claim→落盘 bundle→subprocess runner→init/chunks/finish→complete]+心跳线程；冒烟/真实（--model-path）两模式。
- **CentOS 部署件**（deploy/）：ai2050-distributed.service（systemd，ProtectSystem 硬化，AI2050_DIST_DIR=/var/lib/ai2050/distributed）+ README_CENTOS.md（venv/firewall/SELinux/Agent 接入/运维）。
- **前端**（LensProgress v4）：平台进度 tab←summary（LIVE/DEMO 徽标 .fw-src-chip）、新闻 tab←/api/news（30s 轮询、点击开原文）、离线回退 demo；API_BASE 沿用 VITE_API_BASE 惯例。
- **测试**：e2e 冒烟 16/16（claim→上传→下载 roundtrip→summary/agg/news→负路径 design_sha 422）；真实模式 24/24（qwen3-4b GPU 实采 16.3s，kind=real，`qwen3-4b|real|seed0` 桶）；Agent CLI 对 :5001 全链通（内容寻址去重生效：同内容同 sha）。playwright 前端 7/7、0 错误：LIVE 徽标+真实统计+节点行+TM-01..03 矩阵；机器/研发透镜回归不变。
- 踩坑：agent 忘 import os；相对 --data-dir 在 subprocess cwd 下拼双路径→main 里 resolve()；Edit 把两行拼一行→AST 抓住修复。
- 运行态：:5001 后端（含分布式路由）/:5173 前端/:5010 独立实例（tests/dist_test_data 库）均在跑；news=arxiv 代理不通→builtin-fallback（按设计）。
- 未做（留 M2/M3）：AGG-v1 跨节点 η² 汇总、下载中心 UI、M0 snapshot 收口、多中心 federation。
"""
with io.open(LOG, 'r', encoding='utf-8') as f:
    old = f.read()
with io.open(LOG, 'a', encoding='utf-8', newline='') as f:
    f.write(text)
with io.open(LOG, 'r', encoding='utf-8') as f:
    new = f.read()
assert new.startswith(old) and len(new) > len(old), 'append failed'
print('LOG APPENDED OK')
