# -*- coding: utf-8 -*-
"""Phase 2906 MEMO append (append-only) + on-disk re-verify."""
import io

MEMO = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md"
REPORT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\memo_append_2906_report.txt"

SECTION = u"""
## Phase 2906: margin 幅值定律检验与神经元读出归因 [2026-09-19 05:53]

### 原理
2905 定位 margin 载体为 B 行空间一阶类均值移位（F/T2 与 margin 阳性 4/4 一致），遗留两问：(a) 幅值——各向同性 summary（类均值 mu_c + 每类标量方差 sigma_c）能否**定量还原**谱系幅值排序 qwen mlp 0.180 > glm4 attn 0.121 > glm4 mlp 0.020 > qwen attn -0.019；(b) 粒度——用户指令要求检查/推进神经元级：语言均值移位 delta 在响应空间由哪些输出神经元读出方向承载。响应空间的类均值差 Delta_r（r_same 类均值差，2560/4096 维）可从 2902/2903 npz 直接算出；其分解字典取 down_proj/o_proj 列（列 k = 输出神经元 k 把信号写入残差流的读出方向）——激活值不可零前向恢复，但读出方向分解可以。权重经 safetensors safe_open 直读（不加载模型、零前向）。

### 预注册（冻结于 execution.json，脚本 SHA256-8 00b72977）
- 锚 a1：margin/acc 4 组复现（abs 2e-5）；锚 a2：Delta_B 重算 == 2905 delta_per_layer（abs 1e-4，2905 存 round4）。
- 覆盖率审计（rng [2906,0]）：40 次重复，真参数 -> 抽样 -> 估计 (mu_hat, sigma_hat) -> 400 合成 margin 的 95% 区间覆盖真 margin；pass 当且仅当覆盖 [32,40]/40（Binomial(40,0.95) 下尾 ~2%）。
- 主判据（冻结）：每组 M1 各向同性合成（实测类均值 + sqrt(tr(Sigma_c)/d) 标量方差，类大小固定，400 抽样，共享流，组序 glm4-mlp/glm4-attn/qwen-mlp/qwen-attn）95% 区间；真实 margin_full 落入 4/4 => amplitude_law_confirmed_isotropic；3/4 => amplitude_law_partial；否则 amplitude_law_not_established。
- 神经元归因（描述性，不进主判决）：Delta_r[q] = mean(r_same|lab1,q) - mean(r_same|lab0,q)；c = pinv(W) @ Delta_r（W 行满秩，精确重构）；集中度 = top-64 坐标能量占比；null = 200 次 label 置换 Delta_r（保留真实响应协方差，rng [2906,1]）逐层 p95；高斯 null（rng [2906,2]）诊断对照；组内层中位数 > 置换 p95 中位数 => readout_concentrated。

### 结果（零前向，14m14s）
- **判决：amplitude_law_not_established（2/4），失败模式按通道分裂且有方向**
- 守卫全过：锚 4/4（dLB max ~4.9e-5 < 1e-4）；覆盖率审计 **40/40**。
- M1 还原逐组：
  | 组 | margin_full | M1 95% 区间 | 落入 | SNR | 稀释比 |
  |---|---|---|---|---|---|
  | glm4 mlp | 0.0198 | [-0.0005, 0.0828] | 是 | 0.046 | 0.053 |
  | glm4 attn | 0.1213 | [0.1649, 0.3004] | **否（低于下界）** | 0.154 | 0.315 |
  | qwen mlp | 0.1799 | [0.1128, 0.2984] | 是 | 0.189 | 0.122 |
  | qwen attn | -0.0195 | [-0.0116, 0.0871] | **否（低于下界）** | 0.025 | -0.066 |
- **通道分裂规律：mlp 通道各向同性 summary 充分（幅值定律成立），attn 通道双双低于各自合成下界——attn 类内形状（超越各向同性）主动压低 margin**（qwen attn 被压至负值）。N11 断言协方差各向异性对 margin 期望一阶盲，但真实 attn 数据的形状结构经合成分布比较在幅值上可检——二阶矩痕迹在此显形。
- 神经元读出归因（描述性）：
  | 组 | top-64 占比中位 | 置换 p95 中位 | 判读 |
  |---|---|---|---|
  | glm4 mlp | 0.195 | 0.170 | **concentrated**（10/12 层超，L34 0.310 最强） |
  | qwen mlp | 0.199 | 0.193 | **concentrated**（L29/30/31/34 超，与 delta 剖面 L27/30/34 部分重合） |
  | glm4 attn | 0.149 | 0.162 | distributed |
  | qwen attn | 0.243 | 0.288 | distributed |
- 读出集中度与 margin 阳性/阴性 **4/4 一致**：mlp 通道的语言均值移位由 ~0.7% 输出神经元（top-64/9728 或 /13696）承载 ~20% 能量写入语言轴；attn 通道在置换基线水平（o_proj 列字典无特异集中）。高斯 null 全部 ~0.05-0.12 远低于实测——置换 null（保留响应协方差）才是有效对照。

### 硬伤与混杂
- Delta_r 取 same 条件响应（r_func/r_null 未存盘，对比响应不可重构）——归因对象是 same 上下文的类均值移位，与 B 的对比构造存在 0.5(Delta_func+Delta_null) 成分差（B 锚 a2 已覆盖 B 空间一致性）。
- o_proj 列字典是头聚合读出方向，非严格头级/神经元级；attn 通道的"concentrated 检验"功效受 o_proj 列间强相关影响（置换 p95 与实测几乎重合）。
- M1 的 mu_hat/sigma_hat 与真实 margin 同数据估计（summary 充分性检验，非独立预测）；区间内 = summary 充分，区间外方向 = 形状效应符号。
- 40 覆盖率重复的参数域（mu 范数 0.3、delta 0.5、sigma 0.3-1.0，d=10）为设计选择，未覆盖极端 SNR。

### 结论
1. **幅值定律按通道分裂**：mlp 通道 margin 幅值由各向同性 summary（类均值移位 x 标量类内方差）定量决定；attn 通道需要超越各向同性的类内形状（压低方向）——"类均值移位 x 类内散布稀释"的 L14 精化在 mlp 成立、在 attn 需升级为"形状修正"。
2. **神经元级首证（读出侧）**：mlp 通道（margin 载体）的语言均值移位在 down_proj 列字典上显著集中（~0.7% 神经元承载 ~20% 能量），attn 通道分布化——粒度推进到输出神经元读出级，且与 margin 谱系 4/4 一致。
3. 方法论：pinv 列字典 + 置换标签 null 是零前向神经元归因的可行协议；高斯 null 在真实响应协方差下严重失效（ underestimate p95 ~4 倍），不可用作对照。

### 接续
- 2907 候选：A（主选）**attn 通道形状修正定律**——把 M1 升级为 M2（保留实测 Sigma_c 谱或对 attn 引入形状参数），恢复幅值还原并定位压低 margin 的形状成分（零前向，纯矩阵）；B（备选）头级分解（把 o_proj 列字典换成 per-head W_VO 子空间，定位承载 attn 类移位的头）；C 2907 用前向抓取 SwiGLU 中间激活，把读出侧 concentrated 结论推进到门控神经元激活级（非零前向，须按 eps=1.0 协议预算前向）。

### 文件
- 脚本 tests/glm5/phase2906_amplitude_law_neuron_attribution.py（00b72977）
- 产物 phase2906/amplitude_law_neuron_attribution/：execution.json 4054ffeb / result.json 277bff9e / amplitude_law_neuron_attribution.npz 8b31ea68
- Ledger：M2906_amplitude_law_neuron_attribution + L14 再精化（2906 amplitude law channel-split；connects 13）（measurements 45 / errata 8 / negatives 11 / growth 26 / linkage 14，ledger 52383c5b）
"""

def main():
    out = []
    with io.open(MEMO, "r", encoding="utf-8") as f:
        body = f.read()
    out.append("before_chars=%d" % len(body))
    if u"## Phase 2906:" in body:
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
    out.append("title_ok=%s" % (u"## Phase 2906: margin 幅值定律检验与神经元读出归因 [2026-09-19 05:53]" in body2))
    idx = body2.find(u"## Phase 2906:")
    out.append("line_2906=%d" % (body2.count(u"\n", 0, idx) + 1))
    with io.open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
