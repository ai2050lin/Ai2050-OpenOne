# 工程集成指南

## 一、推荐放置位置

将整个目录放入现有工程根目录：

```text
research/
└── ai2050_research_os/
```

若现有工程已经有 `tests/gpt5/`、`tests/glm5/` 和结果目录，不要移动原实验。只在 `phases.json` 和 `evidence.json` 中保存相对路径与摘要。

## 二、与现有 Phase 体系的关系

保留原命名：

```text
tests/glm5/phase1235_*.py
tests/glm5/result/phase1235_*/
```

新增的机器账本只引用它们：

```json
{
  "phase": 1235,
  "campaign_id": "LEGACY",
  "phase_type": "confirmation",
  "evidence_refs": ["K210"],
  "artifact_paths": ["tests/glm5/phase1235_*.py"]
}
```

未来 Phase 应额外绑定：`campaign_id`、`work_package_id`、`puzzle_ids`、`hypothesis_ids` 和 `test_battery_ids`。

## 三、第一次迁移

1. 复制本框架目录；
2. 修改 `registry/project.json` 中的真实工程路径和模型清单；
3. 将 K1–K210 从原备忘录逐条迁移到 `evidence.json`，不要根据总结重写等级；
4. 给每条 K 添加原 Phase、结果目录、适用域和摘要；
5. 将历史理论路线映射到 `hypotheses.json`，区分“全局强版本关闭”和“局部版本仍可能”；
6. 执行 `validate`；
7. 执行 `build`，检查自动看板；
8. 冻结 C001 的首个正式合同。

## 四、持续集成建议

在自动化流程中加入：

```bash
.venv/Scripts/python.exe research/ai2050_research_os/scripts/researchctl.py validate
.venv/Scripts/python.exe research/ai2050_research_os/scripts/researchctl.py build --check-clean
.venv/Scripts/python.exe research/ai2050_research_os/scripts/researchctl.py verify-manifest manifests/EXP-C001-WP01-001.manifest.json
```

第一条阻止无效引用和非法状态进入主分支；第二条阻止机器账本与生成看板不一致。

## 五、大型结果文件

原始张量、模型权重和逐样本结果不要放入 JSON 账本。账本只保存：

- 相对路径；
- 内容摘要；
- 形状、类型和样本数；
- 生成程序与环境摘要；
- 关键聚合指标；
- 审计状态。

建议大型产物布局：

```text
artifacts/
└── C001/
    └── WP03/
        └── RUN-20260811-001/
            ├── contract.json
            ├── raw/
            ├── derived/
            ├── audit/
            └── manifest.json
```

## 六、提交粒度

每个正式 Phase 最好分成三次提交：

1. 合同冻结；
2. 原始结果和审计；
3. 证据、裁决和自动看板更新。

这样可以从版本历史上确认指标和门是否在揭盲前存在。
