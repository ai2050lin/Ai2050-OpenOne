import os, time

root = r'D:\AI2050\Ai2050-OpenOne'
dst = os.path.join(root, '.workbuddy', 'memory', '2026-10-01.md')

section = """
## 复核 R2：确认 R1 复核 + 产物迁移与登记约定变更（21:10）

- **R1 复核裁决**：独立对源核验，「复核 R1」**10 条全部成立**（仅 2 处数字口径需微修：qwen2.5 N3 总结行实为 PASS；"rank-1 11–18%" 应引 memo §4.5 表 0.7–26%）。
  - R1-P1（P1 级）逐条对上：seal `N3_design_seal.json` K_d = "同号且 B/A≥0.70→解耦，否则判极性-内容纠缠"；实测 qwen3-4b 0.526 / glm4 0.333 / qwen2.5 7.380（A_full=+0.012 退化）⇒ 原判 FAIL；memo Phase 7 §3.1.5 头条"内容与极性可分离（3/3）"属 seal 漂移。
  - R1-P5 印证：`MAIN_AXIS_VERDICT_v1.md` L69 自述 gemma 度量不适用，L20/L94 却计入"6/6" ⇒ 口径应为 5/5 + 1 排除。
  - R1-P7 印证：`n1v2_report_qwen3-4b.txt` L44 每词 dS 峰层 苹果/小米/病毒@L16、杜鹃@L10 ⇒ E2 窗实为 L10–L16。
- **纠错落账**：1 条 P1（K_d 降级为 post-hoc 假设 P-N3b）+ 4 处表述降级（维数定律、必要性倍数 170–6000×、E_same、绑定 6/6），append-only 写入 deepseek 备忘录 `## 复核 R1 确认 + 登记约定变更` 节（127,959 → 136,771 B，sha8 a0c4e066→895ca478，前缀逐字节未变）。
- **产物迁移（137 项，零损失）**：Phase 1–7 全部产物移入新约定落点——`tests/deepseek/`（42 脚本）、`tests/deepseek_temp/`（95 文件 + `memo_review_20261001/` 8 文件）。逐文件 sha256 前后一致、源目录零残留、0 错误。清单 `gpt5_temp/move_manifest_phase1_7.{txt,json}` + `_supplement.txt`。
- **登记约定变更（即日生效）**：deepseek/N 线脚本 → `tests/deepseek/`；报告 `*_report_*.txt`、seal `*_design_seal.json`、校验 `verify_*.txt`、memo/wlog 节源码 → `tests/deepseek_temp/`；记录仍只写 `research/deepseek/docs/AGI_DEEPSEEK_MEMO.md`。旧 `tests/gpt5_temp/` 引用按"`.py`→deepseek、其余→deepseek_temp"换算。
- **技能更新**：`rdc-main-axis-probe` 补入 seal 漂移防护纪律 + 产物落点约定 + 3 条新坑（21 G−1 同义反复 / 22 必要性对照须能量匹配 / 23 E_same 构造性近零），20→23 条。
- **下一步**（按 GPU 优先级）：① N2h1-α 权重级归因（最高）；② N3-δ 确认集 + P-N3b 同轮，前置修否定臂行为基线；③ 对照补强（范数匹配/top-5 PC/同类跨上下文/rank sweep 迁承诺层）；④ qwen3-14b 接入；⑤ K4 处置；⑥ N 线补登 atlas_ledger。
"""

b0 = os.path.getsize(dst)
with open(dst, 'a', encoding='utf-8') as f:
    f.write(section)
b1 = os.path.getsize(dst)
T = open(dst, encoding='utf-8').read()

out = []
out.append('wlog before %d after %d delta %d' % (b0, b1, b1 - b0))
out.append('has_section %s' % ('## 复核 R2' in T))
out.append('tail_ok %s' % T.rstrip().endswith('⑥ N 线补登 atlas_ledger。'))
open(os.path.join(root, 'tests', 'deepseek_temp', 'verify_wlog_r2.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('\n'.join(out))
