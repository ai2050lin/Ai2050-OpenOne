# -*- coding: utf-8 -*-
"""Phase 2903 MEMO append (append-only) + on-disk re-verify."""
import io, os, sys

MEMO = r"D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md"
REPORT = r"D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\memo_append_2903_report.txt"

SECTION = u"""
## Phase 2903: qwen 通道 Jacobian 分解——语言内容密度跨模型检验 [2026-09-19 03:34]

### 原理
M2902 将 glm4 margin 层级（attn/mlp 比 6.13）归因于落地方向语言内容密度（G_language_content_density_mechanism），其跨模型预测：qwen mlp 通道（margin ~0.18，谱系最大载体）落地方向的语言对齐 A 应显著高于 glm4 mlp（A 密度最低载体）。本 Phase 将 2902 协议 verbatim 译至 qwen3-4b（窗口 [26,36)，57 词），执行同一三假设分解（H1 范数增益 / H2 方向保持 / H3 语言对齐），并以 qwen 自身 20 随机方向 null 校准 A 指标的噪声底线。若 qwen 也证实则 G 升格为跨模型机制；若 qwen 否定则须回查 glm4 的 A 是否本就未超噪声。

### 预注册（冻结于 execution.json，脚本 SHA256-8 f05e5644）
- 窗口 W=[26,36)（2896 冻结）；dirs 从 2886 S_last 重算（labels i%2 断言，2896 verbatim）；words/labels 逐字取 2887 npz（57 词）；null_tids 按 2896 rng 顺序（SEED=2896）bit-exact。
- 响应：r_ch = (module(x+eps*d)-ref)/eps @ pos1，eps=1.0，57 词 x 3 条件（same/func/null），机制 verbatim 2896/2898。
- v1：模块重算 rel err < 1e-6 且 hook-vs-call 噪声 < 1e-4；v2 锚：B'=(r.same-0.5(r.func+r.null)).d 须与 2896 npz B_eigen rel err < 1e-3 否则 all void。
- tie=True：unembed = embed_tokens（2902 glm4 为 untied lm_head，协议差异已登记）。
- null 对照：20 随机单位方向（SEED=2903）@ 层 {26,29,32}，双通道。
- 判决映射（冻结）：锚败=>v2_anchor_fail_all_void；D_q_mlp >= 2*max(D_g_mlp,D_g_attn) 且 D_q_mlp > 2*q_null_p95 => language_content_density_confirmed_cross_model；elif D_q_mlp > 2*q_null_p95 => density_elevated_below_2x_glm4max；else density_not_confirmed。
- **Amendment（run 1 后、判决前）**：run 1 v2 锚失配 2.48e-1——AutoModelForCausalLM+device_map（混合 cpu/cuda）前向上下文数值与 2896 load_native 全 GPU 不同，即 2877 教训的设备级版本。修正：load_native("qwen4") verbatim 2896。run 1 无任何判决输出，未进入判读。

### 结果（run 4，运行 125.9s）
- **判决：density_not_confirmed**
- 守卫全过：v1 = 0.0（双通道）、hook 噪声 0.0；**v2 锚 rel err = 2.32e-8**（verbatim 加载后 bit-exact 复现 2896 上下文）。
- margin 层级跨模型复现：qwen mlp margin = +0.1799（acc 0.8421，超自身 perm p95 0.0400 的 4.5 倍）；**qwen attn eigen 首测阴性：margin = -0.0195**（acc 0.6140，p95 0.0339 内）——与 glm4 attn 阳性（0.1213）相反，attn 通道非普适载体。
- **H1 否定（N9 再证）**：R_gamma=0.2166——qwen mlp gamma（1.54-3.83）比 attn（0.29-1.01）大 ~4.6 倍，与 2902 glm4 同向：mlp 大响应非 margin 源。
- **H2 否定**：R_rho=0.52，两通道 rho 近零（-0.16~0.15）——落地方向完全重定向，与 2902 一致。
- **H3 判决关键**：D_q_mlp = 0.00165 < 2*q_null_p95 = 0.00928（自身 null p95 = 0.00464，D 值低于噪声底线）→ 未达 elevated 门槛 → density_not_confirmed。R_A=1.16（<2902 的 1.438）。
- 权重描述性：qwen attn 零前向响应增益随层增长（L26 2.5 -> L34 37.4），W_VO 复合 PR ~676-800；mlp 零前向 SwiGLU 直通近零（0.002-0.05）而实测 gamma 1.0-4.2——context 交互主导，同 2902 结论。

### E8 勘误（corrects M2902，后验校准）
qwen 自身 null 底线（0.00464）促使回查 2902 null 条目：g_null_p95 mlp = 0.00506 / attn = 0.01094。而 D_g_mlp = 0.00102、D_g_attn = 0.00429——**glm4 双通道的 A 值同样全在自身噪声内**。故 M2902 的 H3"确认"（R_A=1.438）系未做 null 校准的假阳性；A（词 unembed 行平均 |cos|）作为标量指标在两模型全通道均不携带超噪声信号。H3 降级 not-established；2902 的"结构根源 = 语言内容密度"机制解释（结论 1/2）撤回；入 E8 勘误与 N10_cross_model_density_prediction（refuted）。

### 硬伤与混杂
- 57 词（<2902 的 78 词）与 null 方向仅 20x3 层：标量 D 的功效有限，不排除子空间结构信号被中位数抹平。
- A 指标以 embed_tokens 行为 unembed 代理（tie=True），未含 ln_f/softmax 归一——但 null 校准已包含同一代理，底线结论不受代理选择影响。
- E8 为后验（qwen 结果触发回查），非预注册判决——按纪律如实登记为勘误而非 silently 修订 M2902。

### 结论
1. **margin 层级根源重新开放**：qwen mlp margin 阳性（+0.180）但语言对齐在噪声内、glm4 attn margin 阳性同样语言对齐在噪声内——三个标量 Jacobian 假设（H1 范数增益 / H2 方向保持 / H3 语言对齐）全部出局。margin 必然栖身于 B 行空间的**类别相关结构**（class-correlated STRUCTURE）而非任何标量平均量。
2. **qwen attn 通道非普适载体**：同方向族下 qwen attn eigen margin 首测阴性，与 glm4 attn 相反——margin 层级是每（模型,通道）的特异性质，非跨模型通道角色。
3. **方法论再证**：锚复现必须 verbatim 加载方式（run1 device_map 失配 2.5e-1 -> run4 load_native 2.3e-8，2877 教训设备级推广）；null 校准必须在主判决前完成（E8 教训：A 指标 2902 未校准、2903 补校准推翻主判决）。

### 接续
- 2904 候选：A（主选）**B 行空间类别相关结构分析**——对已有 2902/2903 npz 的 B 矩阵（57/78 词 x 方向族）做类内/类间主方向分解、有效秩/PR、与 label 的典型相关，直接检验"margin 栖身于结构"假说（零前向，纯矩阵分析）；B（备选）attn 响应的注意力再分配分量分离（K-shift vs V-path）；C glm4 attn eigen 格弱层归因。

### 文件
- 脚本 tests/glm5/phase2903_qwen_channel_jacobian_decomposition.py（f05e5644）
- 产物 phase2903/qwen_channel_jacobian_decomposition/：execution.json c6a5bc3c / result.json 8e6d009b / qwen_channel_jacobian_decomposition.npz d99eb685
- Ledger：M2903_qwen_channel_jacobian_decomposition + E8（corrects M2902）+ N10_cross_model_density_prediction(refuted) + G_language_content_density_mechanism 修订（claim withdrawn）+ L14 再精化（measurements 42 / errata 8 / negatives 10 / growth 26 / linkage L14=10，ledger 7389e8c6）
"""

def main():
    out = []
    with io.open(MEMO, "r", encoding="utf-8") as f:
        body = f.read()
    out.append("before_chars=%d" % len(body))
    if u"## Phase 2903:" in body:
        out.append("already_present=True (skip append)")
    else:
        if not body.endswith(u"\n"):
            body += u"\n"
        body += SECTION
        with io.open(MEMO, "w", encoding="utf-8", newline="") as f:
            f.write(body)
        out.append("appended=True")
    # on-disk re-verify (fresh read)
    with io.open(MEMO, "r", encoding="utf-8") as f:
        body2 = f.read()
    out.append("after_chars=%d" % len(body2))
    out.append("title_ok=%s" % (u"## Phase 2903: qwen 通道 Jacobian 分解——语言内容密度跨模型检验 [2026-09-19 03:34]" in body2))
    out.append("tail_lines=%d" % len(body2.splitlines()))
    idx = body2.find(u"## Phase 2903:")
    out.append("line_2903=%d" % (body2.count(u"\n", 0, idx) + 1))
    with io.open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
