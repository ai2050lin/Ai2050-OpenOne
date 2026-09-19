# -*- coding: utf-8 -*-
"""Phase 2904 MEMO append (append-only) + on-disk re-verify."""
import io

MEMO = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md"
REPORT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\memo_append_2904_report.txt"

SECTION = u"""
## Phase 2904: B 行空间结构分析与检测器代数审计 [2026-09-19 05:31]

### 原理
L14 精化（2903）判定 margin 必栖身于 B 行空间的类别相关结构而非标量平均量。本 Phase 对 2902/2903 npz 的 B 矩阵（glm4 78x12 / qwen 57x10，行=词、列=窗口层）做零前向纯矩阵分析，按 2809 制度先过构造代数审计再进主判读：主分解把 margin 承载结构分为一阶（类均值移位，margin_from_mean）与高阶（within-class centering 后存活的 margin_within）；阳性组集合 P 由 2902/2903 冻结的 stored p95 判定。

### 预注册（冻结于 execution.json，脚本 SHA256-8 2bdd1097）
- 锚 a1：4 组（glm4/qwen x mlp/attn）margin 与 acc 从 float32 npz 重算须与 stored 值 abs 差 < 2e-5，否则 anchor_fail_all_void。
- 代数审计（合成矩阵）：audit_1st_order（类均值广播 + N(0,1)，delta=2.0*e1，split 22/35）期望 margin_full > p95_full 且 margin_within <= p95_within；audit_2nd_order（零均值移位 + 类相关协方差 4x std 沿随机正交 v1/v2）期望 margin_within > p95_within，失败 => detector_insensitive_all_void（不进主判读）；audit_negative（N(0,I)）期望双指标均低于 p95；from_mean 恒等式（margin(广播矩阵) == 1 - cos(mu0h, mu1h) < 1e-9）逐组内建核对。
- null：SEED=2904，1000 次 label 置换/组（单一共享流，组序 glm4-mlp, glm4-attn, qwen-mlp, qwen-attn）；p95_within 在置换下重算 within-center。
- 判决映射（冻结）：锚败 => all_void；阴性/一阶审计败 => algebraic_audit_fail_all_void；二阶审计败 => detector_insensitive_all_void；P 空 => structure_not_established；P 全体 margin_within <= p95_within => margin_carried_by_class_mean_shift；P 全体 > => margin_carried_by_higher_order_structure；混合 => structure_mixed_across_carriers。

### 结果（零前向，秒级）
- **判决：detector_insensitive_all_void**——主判读未发生。
- **锚 4/4 全过**：glm4 mlp 0.01979/0.67949、glm4 attn 0.12126/0.73077、qwen mlp 0.17991/0.84211、qwen attn -0.01946/0.61404——float32 npz 数据链完好。
- audit_1st_order PASS（full 0.176 > p95 0.031，within -0.034 <= p95 -0.031）；audit_negative PASS（iid 3% 假阳性基线）。
- **audit_2nd_order FAIL**：协方差差异构造下 margin_within = -0.032 未超 p95 -0.026。
- from_mean 恒等式 4/4 过（主分析段已进 gate 前完成计算——注：恒等式核对在 gate 内，gate 拦截后主统计未输出）。
- 阳性组集合（stored 判据）：P = {glm4 attn (0.1213 > 0.0426), qwen mlp (0.1799 > 0.0400)}；glm4 mlp (0.0198 < 0.0402) 与 qwen attn (-0.0195 < 0.0339) 为阴性组。

### 解析诊断（2904b 合成诊断，临时脚本 diag_2904b.py）
审计 2 失败的根因是**构造的数学缺陷而非检测器缺陷**：
1. 对均值零、类相关协方差的行分布，E[cos|same] - E[cos|diff] = 0（一阶矩恒等）——协方差各向异性对余弦相似度的均值 margin 无期望贡献。实测：cov-4x 构造 pairwise gap = -0.001（40 万独立对）；n=57 完整管线检出率 6/100（= 假阳性基线）。
2. margin 家族的正确灵敏域：一阶（类均值移位）与**类内偏度/非对称子簇**结构（均值恰为零但质量单侧分布，90% +2w / 10% -18w）：pairwise gap = +0.092，n=57 管线检出率 **74/100**；iid null 3/100。
3. 结论：margin_within 检测器对二阶矩（协方差各向异性）**原理性盲**，对三阶矩（偏度）灵敏。

### 硬伤与教训
- 审计 2 构造参数未做事前解析期望核对就冻结——合成审计构造本身必须先做解析推导或数值预检（本 Phase 用一轮 all_void 买到此教训，制度化入 N11 reopen_condition：二阶矩敏感的 margin 变体须自带新代数审计）。
- gate 设计按纪律把主判读拦在审计后，主统计（P 组 margin_within 读数）在 2904 从未被观测——2905 将是首次判读，无污染。

### 结论
1. **N11 边界定律**：2896 族 margin 指标（含 within 变体）检测域 = {类均值移位（一阶矩）, 类内偏度/非对称子簇（三阶矩）}；协方差各向异性（二阶矩）数学不可见。
2. 锚 4/4 证明零前向矩阵分析的数据链可靠；审计 gate 机制按设计工作（构造缺陷在主判读前被拦截并如实登记 all_void）。
3. B 行空间结构假说的可检验形式收敛为：P 组的 margin 是否存活于 within-class centering——须用 skew 审计过的检测器（2905）。

### 接续
- 2905（已开工）：audit_2nd_order 换为 skew 构造（big=2.0/tail=18.0/pf=0.9，w0 垂直 w1 随机正交；20 独立实例检出率 >= 12/20 为 pass），其余协议不变——P={glm4 attn, qwen mlp} 的 margin_within 首次判读。

### 文件
- 脚本 tests/glm5/phase2904_b_row_space_structure.py（2bdd1097）
- 产物 phase2904/b_row_space_structure/：execution.json 81e46f99 / result.json 3a782d18 / b_row_space_structure.npz 8739c76e
- 诊断 tests/gpt5_temp/diag_2904b.py（临时探针，结果 diag_2904b.txt）
- Ledger：M2904_b_row_space_structure_audit_gate + N11_covariance_anisotropy_invisible_to_margin + L14 再精化（notes + connects 11）（measurements 43 / errata 8 / negatives 11 / growth 26 / linkage 14，ledger b89607b1）
"""

def main():
    out = []
    with io.open(MEMO, "r", encoding="utf-8") as f:
        body = f.read()
    out.append("before_chars=%d" % len(body))
    if u"## Phase 2904:" in body:
        out.append("already_present=True (skip append)")
    else:
        if not body.endswith(u"\n"):
            body += u"\n"
        body += SECTION
        with io.open(MEMO, "w", encoding="utf-8", newline="") as f:
            f.write(body)
        out.append("appended=True")
    with io.open(MEMO, "r", encoding="utf-8") as f:
        body2 = f.read()
    out.append("after_chars=%d" % len(body2))
    out.append("title_ok=%s" % (u"## Phase 2904: B 行空间结构分析与检测器代数审计 [2026-09-19 05:31]" in body2))
    idx = body2.find(u"## Phase 2904:")
    out.append("line_2904=%d" % (body2.count(u"\n", 0, idx) + 1))
    with io.open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
