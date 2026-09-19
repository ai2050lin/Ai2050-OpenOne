# -*- coding: utf-8 -*-
"""Phase 2905 MEMO append (append-only) + on-disk re-verify."""
import io

MEMO = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md"
REPORT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\memo_append_2905_report.txt"

SECTION = u"""
## Phase 2905: skew 审计检测器下 margin 结构主判读 [2026-09-19 05:40]

### 原理
2904 判决 detector_insensitive_all_void 并诊断出 margin 检测域边界（N11：一阶均值 + 三阶偏度灵敏、二阶协方差数学盲）。本 Phase 按 2904 收尾预案把 audit_2nd_order 换为诊断验证过的 skew 构造（20 独立实例检出率 >= 12/20 为 pass；2904b 诊断点估计 74%），其余协议 verbatim 2904，对阳性组集合 P={glm4 attn, qwen mlp} 执行 margin_within 的**首次主判读**——检验"margin 栖身于 B 行空间类别相关结构"假说的可检验形式：移除类均值移位后 margin 是否存活。

### 预注册（冻结于 execution.json，脚本 SHA256-8 ea1ac473）
- 锚 a1 / audit_1st_order / audit_negative / from_mean 恒等式：verbatim 2904。
- audit_2nd_order_skew：均值零行、类相关偏度（90% 质量 +2w、10% 质量 -18w，w0 垂直 w1 随机正交，n=57 d=10 split 22/35），20 独立实例（rng 子流 [2905,2,k]），每个实例 1000 perm p95，pass 当且仅当检出率 >= 12/20；失败 => detector_insensitive_all_void。
- null：SEED=2905，1000 次 label 置换/组（共享流，组序 glm4-mlp, glm4-attn, qwen-mlp, qwen-attn），p95_within 在置换下重算 within-center。
- 判决映射（冻结）：与 2904 相同（P 全体 margin_within <= p95_within => margin_carried_by_class_mean_shift；全体 > => higher_order；混合 => mixed）。

### 结果（零前向，秒级）
- **判决：margin_carried_by_class_mean_shift**
- 审计全过：锚 4/4（glm4 mlp 0.01979/0.67949、glm4 attn 0.12126/0.73077、qwen mlp 0.17991/0.84211、qwen attn -0.01946/0.61404）；audit_1st PASS（full 0.166 > p95 0.030）；**audit_2nd_skew 检出 20/20**（中位 margin_within +0.0849）——skew 检测器全实例灵敏，2904 的失败确证为构造缺陷；audit_negative PASS；from_mean 恒等式 4/4。
- **主判读（P 两阳性组 margin_within 均不超置换 p95_within）**：
  | 组 | margin_full | margin_from_mean | margin_within | p95_within | within 超阈 |
  |---|---|---|---|---|---|
  | glm4 attn | +0.1213 | 0.3852 | -0.0229 | -0.0149 | 否 |
  | qwen mlp | +0.1799 | 1.4726 | -0.0332 | -0.0195 | 否 |
- **Fisher F / Hotelling T2 与 margin 阳性/阴性 4/4 一致**：qwen mlp F=5.121 > p95 2.286、T2=51.1 > 23.1（rho1=0.694）；glm4 attn F=6.685 > 2.701、T2=54.9 > 27.1（rho1=0.648）；阴性组 glm4 mlp（F=1.73 < 2.35）、qwen attn（F=0.63 < 2.38）均不超——B 行空间语言均值移位的存在性恰好按通道复现 margin 谱系。
- delta 层剖面：qwen mlp delta_norm=0.2557，集中 L27/L30/L34（-0.173/-0.118/-0.107）；glm4 attn delta_norm=0.0151（更弱但同侧富集 L34 0.0119）。
- 定量图像：margin_from_mean >> margin_full（qwen mlp 1.47 vs 0.18；glm4 attn 0.39 vs 0.12）——纯类均值结构的相似度优势被类内散布稀释成观测 margin；within-centering 移除均值后结构完全消失（margin_within 落到置换基线下）。

### 硬伤与混杂
- margin_within 的 p95_within 为负值区（-0.015/-0.020）："不超阈"即 margin_within 在基线以下，结论是结构消失而非弱信号。
- F/T2 与 margin 排序不完全同序（F: glm4 attn 6.69 > qwen mlp 5.12，margin: qwen mlp > glm4 attn）——存在性判据（超阈与否）与幅值排序是不同的泛函；margin 幅值 = f(delta 强度, 类内形状, 类大小)，属 L14 幅值定律的未决精化。
- 57/78 词、10/12 层窗口的单一数据集；labels 仅 en/fr 二分类（margin 的原始定义），concept 结构未在本 Phase 判据内。

### 结论
1. **margin 的结构根源已定位（一阶）**：2896 族 margin 阳性 ⟺ B 行空间存在超置换基线的类（语言）均值移位（F/T2 判别 4/4 一致）；移除类均值后 margin 存活为 0——margin_carried_by_class_mean_shift。
2. **谱系闭环**：2902/2903 三个标量 Jacobian 假设出局（E8/N9/H2）→ 2904 N11 划定检测域 → 2905 定位一阶类均值移位。L14 谱系从"margin 是类别相关结构"精化为"margin 是类均值移位 x 类内散布稀释"：qwen mlp delta_norm 0.256（最大）> qwen attn 0.024 ≈ glm4 attn 0.015（但 glm4 attn 类内形状更聚，稀释更少）——幅值差异来源留待分解。
3. **方法论闭环**：skew 审计检测器 20/20 检出 + 阴性审计 3% 假阳性——构造代数审计（2809 制度化）在 2904 拦截构造缺陷、在 2905 放行有效检测，全流程按设计工作。

### 接续
- 2906 候选：A（主选）**margin 幅值定律分解**——delta_norm x 类内散布几何如何映射到 margin 幅值（qwen mlp delta 最大但 F 不是最大；解析 + 合成标定 + 真实数据拟合），解释谱系幅值排序 qwen mlp 0.18 > glm4 attn 0.12 > glm4 mlp 0.02；B（备选）delta 层剖面的通道归因（qwen mlp L27/L30/L34 vs glm4 attn L34）；C 换 concept 标签重跑分解（结构对标签类的特异性）。

### 文件
- 脚本 tests/glm5/phase2905_margin_structure_skew_audited.py（ea1ac473）
- 产物 phase2905/margin_structure_skew_audited/：execution.json 619e02ee / result.json 05a42b4d / margin_structure_skew_audited.npz 2846e50b
- Ledger：M2905_margin_structure_skew_audited + L14 再精化（2905 main reading；connects 12）（measurements 44 / errata 8 / negatives 11 / growth 26 / linkage 14，ledger f9e9db55）
"""

def main():
    out = []
    with io.open(MEMO, "r", encoding="utf-8") as f:
        body = f.read()
    out.append("before_chars=%d" % len(body))
    if u"## Phase 2905:" in body:
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
    out.append("title_ok=%s" % (u"## Phase 2905: skew 审计检测器下 margin 结构主判读 [2026-09-19 05:40]" in body2))
    idx = body2.find(u"## Phase 2905:")
    out.append("line_2905=%d" % (body2.count(u"\n", 0, idx) + 1))
    with io.open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
