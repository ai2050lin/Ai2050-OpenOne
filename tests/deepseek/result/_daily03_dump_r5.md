# 工作日志 2026-10-03

## Phase 21 之后（R3 续研：验收函数落地，非 Phase）

- **把 I1 操作化为可计算量**：`tests/deepseek/_review/classify_phases_r3.py` 扫描 gpt5 MEMO，启发式判定「共享指标变化」⇒ **397 个 Phase 节中仅 26 个（6.55%）报告了「指标名 + 小数 A→B」**（窄词表 18，区间 [18,26]）⇒ **93.45% 是目录条目**。
- **两份治理交付件**：
  - `research/gpt5/docs/RDC_RESEARCH_CONSTITUTION_v1.md`（10,388 B / sha8 `01df6398` / LF-only）：I1–I11 全部落为判据（KPI 三数、行为判决层、死线双轨、可识别性门、未识别变量程序、单一真源、冻结队列、外部评审规则、工程去摩擦）。
  - `research/gpt5/atlas/phase_queue_v1.json`（9,669 B / sha8 `675836fd` / 30 项 Q01–Q30 / 唯一议程来源）。
- **生成与复核**：`gen_constitution_r3.py`（数据驱动，0 占位残留）；`disk_verify_continue_r3.py` 独立复核 **PASS 44 / FAIL 0**（源文件级重算：Phase 节 397、宽词表 advance 精确复现 26、ledger 自声明 `41d65a13` ≠ 实际 `bbda63df`、MEMO 中 E_ar/E_read/C_steer 各 0 次）。
- **一处工程修正**：生成器文本模式写盘产生 CRLF（10388 vs 10555 差 166 行）⇒ 改 `newline="\n"` 强制 LF，使 hash 跨平台可复现。
- **未改动**：MEMO 原文、既有判决、Ledger 均未触碰；K1 改判仍为待 seal 的决策点（Q08）。

## R4：A 闸门 Q01 + Q02（元层单一真源 + KPI 口径冻结，零 GPU）
- **Q01**：五处元层不自洽全定位根因（原 3 + 新 2：MEMO 记 `proposition_ledger`=`add57ba7`≠实际 `9a3c6ff4`；`measurements` 两套 schema 且 `evidence_level` 304/304 恒为常量）。口径量纲混淆（55%/63% = 分量数÷条数）；哈希自指结构性失效（自洽值 `0dc6e57a`）。产物 `research/gpt5/docs/META_SINGLE_SOURCE_Q01.md`(`95366ca9`) + `research/gpt5/atlas/meta_single_source_v4.json`(`f9d7ede6`)；独立复核 **44/0 ALL_PASS**。
- **Q02**：`metric_dict.json` v1→v2（`469c0ad1`→`03887e51`；备份 `metric_dict_v1_backup.json`），新增 `global_kpis`（E_read=`b4_rel_readout_mean3seed` 0.33162/0.39860/0.38984，**0/3 过 5% 门**；E_ar/C_steer 口径冻结未测）+ 登记规则；`metrics`7 + `meta_rules` 原样 ⇒ 兼容 P3150 历史断言。报告 `research/gpt5/docs/METRIC_DICT_Q02.md`(`ba518fea`)；复核 **43/0 ALL_PASS**。
- 汇报页 `tests/deepseek_temp/_review/q01q02_gate_r4.html`(`d5fbf42b`)。
- 脚本入 `tests/deepseek/_review/`（probe_q01/q01b/q02/q02b/q02c、gen_q01/gen_q02/gen_q01q02_html、disk_verify_q01/q02），报告入 `tests/deepseek_temp/_review/`。
- **未改动**任何 MEMO 原文 / TESTPLAN / Ledger / 3152 产物；6 条更正 C1–C6 **待 seal**。
- 技能 `rdc-phase-closeout` 新增**教训 37**（元层单一真源审计 5 条）；MEMORY.md 重写压缩并纳入本节。
