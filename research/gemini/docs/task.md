# Phase 266: Shift to Causal Physics & Empirical Hypotheses

## 任务分解

- [ ] **1. 理论文档的全面降级与修正**
  - [ ] 修正 `llm_first_principles_mathematics.md`，将“公理”降级为“经验假说”，增加维度诅咒和缺乏预测性的免责声明。
  - [ ] 修正 `neuron_attribute_mapping.md`，标注 DS7B 低维的退化可能，以及“架构免疫”的局限性。
  - [ ] 同步更新到用户的 `docs` 目录。
- [ ] **2. 开发因果切除 (Ablation) 测试脚本**
  - [ ] 编写 `tests/claude/266_causal_ablation.py`。
  - [ ] 目标：在 Qwen3-4B 的 L35 切除已知的“动词神经元”或“名词神经元”。
  - [ ] 测试语料：`I need to book a flight` vs `I read a book`。
  - [ ] 观察切除后输出概率分布（Logits）的崩塌情况，建立严格的因果链。
- [ ] **3. 执行与分析**
  - [ ] 运行脚本并收集对比数据。
  - [ ] 验证因果干预是否符合理论预期。
- [ ] **4. 制作结案汇报展示**
  - [ ] 产出 `walkthrough.md`。
