# -*- coding: utf-8 -*-
"""Append Phase 2907 section to AGI_GPT5_MEMO.md (append-only)."""
import hashlib

MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\memo_append_2907.txt')

SECTION = """
## Phase 2907: attn 形状修正阶梯检验与 2906 sigma 定义勘误 [2026-09-19 06:55]

### 原理
2906 公布候选 A：把 M1 各向同性 summary 升级为形状修正阶梯，检验 attn 通道"低于 M1 下界"是否需要超越各向同性的类内形状来还原。阶梯：M1（mu_c + sigma_c*I，标量）-> M2a（mu_c + diag(Sigma_c)^{1/2}，层方差剖面）-> M2b（mu_c + chol(Sigma_c)，完整协方差，Sigma_c = 同类行样本协方差，n>=d+1）。还原逻辑：若 M1 区间已含真实 margin，则形状修正不必要；若 M1 排除而 M2a 包含，则层方差剖面（对角）承载形状效应；若需 M2b，则跨层相关也承载。sigma 定义按 2906 prereg 文本统一为 RMS sqrt(tr(Sigma_c)/d)。

### 预注册（冻结于 execution.json，脚本 SHA256-8 87376b53）
- 锚 a1：margin/acc 4 组复现（abs 2e-5）；锚 a2：Delta_B 重算 == 2905 delta_per_layer（abs 1e-4）。
- 覆盖率审计：rng [2907,0] M2a（真 diag Sigma，层方差 0.09-1.0）与 [2907,3] M2b（真全 PSD Sigma = A A^T + 0.1I），各 40 重复，pass 当且仅当覆盖 [32,40]/40；失败 => audit_coverage_fail_all_void。
- 合成：每组每级 400 抽样，共享流 SEED=2907，组序 glm4-mlp/glm4-attn/qwen-mlp/qwen-attn，级序 M1/M2a/M2b，类大小固定。
- 判决映射（冻结）：attn 两组 both M2a => shape_correction_diagonal；both M2b => shape_correction_full_covariance；both none => shape_correction_failed；split => shape_correction_mixed；mlp 组仅作 sanity。**映射未枚举 both-M1 分支**（预注册时先验认为 attn 不会落入 M1，依据 2906 观测）。

### 结果（零前向，1.2s）
- 守卫全过：锚 4/4；覆盖率审计 M2a 39/40、M2b 39/40 双过。
- 阶梯逐组（sigma 用 RMS）：
  | 组 | margin_full | M1 95% 区间 | M2a 区间 | M2b 区间 | 级 | 层方差比 S0 | 跨层 |corr| 均值 |
  |---|---|---|---|---|---|---|---|
  | glm4 mlp | 0.0198 | [-0.0035, 0.0692] | [-0.0144, 0.0981] | [-0.0125, 0.1024] | **M1** | 31.0 | 0.183 |
  | glm4 attn | 0.1213 | [0.0915, 0.2262] | [0.0521, 0.2748] | [0.0400, 0.2656] | **M1** | 64.4 | 0.130 |
  | qwen mlp | 0.1799 | [0.0970, 0.2911] | [0.0975, 0.2984] | [0.0865, 0.3395] | **M1** | 9.3 | 0.366 |
  | qwen attn | -0.0195 | [-0.0198, 0.0631] | [-0.0224, 0.0936] | [-0.0233, 0.0808] | **M1** | 21.9 | 0.152 |
- **四组全部 level=M1**：RMS 定义下各向同性 summary 对全部四通道充分，形状修正不必要。qwen attn 富余仅 0.0004（MC 噪声级 borderline）；glm4 attn 从 2906 的"低于下界 0.044"翻转为区间内。
- 判决映射缺口：both-M1 未被冻结映射枚举，落入 else 分支机械输出 shape_correction_mixed——**该标签与实际结果（both-M1，形状修正不必要）不符**，按纪律保留冻结标签原样、以本节为准解读。

### diag_2907b 归因诊断（post-hoc，2x2x2 网格）
因素：sigma 定义（mean-std=2906 实现 vs rms=2907 实现/2906 prereg 文本）x seed（2906/2907）x 流结构（mode2906 每组仅 M1 段 vs mode2907 M1->M2a->M2b 完整阶梯流）。复现锚：mean_std|2906|mode2906 与 rms|2907|mode2907 分别复现 2906/2907 登记区间至 max|diff| 9e-7（round6 存储容差）——诊断器可信。
- sigma 比值 rms/mean-std（Jensen 下界 1）：glm4 mlp 1.26/1.19，glm4 attn 1.19/1.34，qwen mlp 1.08/1.06，qwen attn 1.15/1.30。
- **glm4_attn 判定 100% 由 sigma 定义驱动**：mean-std 下 4/4 组合 below（下界 0.157-0.165 vs full 0.1213）；rms 下 4/4 组合 inside（下界 0.091-0.100）——与 seed、流结构完全无关。
- **qwen_attn 为 MC 噪声级 borderline**：mean-std 下 4/4 below（差 0.004-0.008）；rms 下 inside 2/4、below 2/4（下界 -0.017 至 -0.021 vs full -0.0195）。
- **2906 prereg 文本-实现 drift（E9）**：2906 prereg 冻结文本写 sqrt(tr(Sigma_c)/d)（RMS），实现却用 B[m].std(0).mean()（mean-std，向下偏代理）；2906 覆盖率审计与实现共享同一定义故自洽通过、drift 未被察觉。按 2906 自己的 prereg 文本定义（RMS）重算，attn 判定即非 BOTH_BELOW。

### 硬伤与混杂
- 判决标签 shape_correction_mixed 名不副实（映射缺口），Ledger M2907 verdict 内已注明。
- qwen attn 的 M1 判定富余 0.0004，400 抽样 MC 误差（区间端点 ~0.002-0.004）与判定同量级——qwen attn "inside" 结论需增样或多 seed 确认。
- 2906 result.json immutable 不改；其 attn-below 读法由 E9 勘误 + M2907 修正，非删改。
- M2a/M2b 区间端点跨组无单调关系（glm4 attn M2b 下界低于 M2a，qwen mlp M2b 上界高于 M2a），chol 旋转既可增宽也可移位区间——阶梯只在"M1 排除后逐级还原"语义下有效。

### 结论
1. **形状修正不必要（RMS 定义下）**：四通道（含两 attn）的 margin 幅值均由各向同性 summary（类均值移位 x RMS 类内标量方差）定量还原——2906"attn 类内形状主动压低 margin"作为幅值事实被撤回，幸存结论仅为 qwen attn 位于 M1 下缘（MC 分辨率内）。
2. **2906 勘误 E9 入账**：prereg 文本-实现 sigma drift 使 2906 的 attn 失配读法不稳健；"amplitude_law_not_established" 冻结判决保留在案，机制读法以 M2907 为准。
3. **谱系链条闭环（2903->2907）**：margin = 类均值移位（2905 一阶载体）x 类内散布稀释（2906 mlp 定量 + 2907 全通道 RMS 确认）；读出侧 mlp concentrated / attn distributed（2906）与幅值谱系 4/4 一致；sigma 定义敏感性（E9）是本轮唯一修正。
4. 方法论：prereg 文本-实现 drift 不能靠共享该定义的审计自检——需要独立审计通读实现（本次由 2907 跨 phase 对照暴露）；2x2x2 stream-faithful 复现网格（双锚 9e-7）是归因此类跨 phase 翻转的廉价强协议。

### 接续
- 2908 候选：A（主选）**qwen_attn M1 边界判定加密**——400 -> 10000 抽样 + 多 seed 网格（零前向，秒级），判定 qwen attn margin 在 M1 内/低于下界/恰在下缘，消解当前谱系最脆弱一环；B（备选）头级分解：o_proj 列字典 -> per-head W_VO 子空间，定位 attn 通道承载/抵消类移位的头；C 前向 SwiGLU 激活级归因（非零前向，eps=1.0 协议预算）。

### 文件
- 脚本 tests/glm5/phase2907_shape_correction_law.py（87376b53）
- 产物 phase2907/shape_correction_law/：execution.json bd3d5322 / result.json 390f27c9 / shape_correction_law.npz 9bd10e2d
- 诊断 tests/gpt5_temp/diag_2907b.py -> diag_2907b.txt（2x2x2 网格 + 双锚复现）
- Ledger：M2907_shape_correction_law + E9（corrects M2906）+ L14 再精化（2907 shape ladder；connects 14）（measurements 46 / errata 9 / negatives 11 / growth 26 / linkage 14，ledger a25a7122）
"""

before = hashlib.sha256(open(MEMO, 'rb').read()).hexdigest()[:8]
with open(MEMO, 'a', encoding='utf-8') as f:
    f.write(SECTION)

# verify
tail = open(MEMO, encoding='utf-8').read()
ok_title = '## Phase 2907: attn 形状修正阶梯检验与 2906 sigma 定义勘误 [2026-09-19 06:55]' in tail
n2907 = tail.count('## Phase 2907')
after = hashlib.sha256(tail.encode('utf-8')).hexdigest()[:8]
rep = ('before %s / after %s / title_ok=%s / n2907=%d / tail_len=%d'
       % (before, after, ok_title, n2907, len(tail)))
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(rep + '\n')
print('OK', OUT)
