# -*- coding: utf-8 -*-
"""memo_2916_append.py -- append Phase 2916 section (append-only)."""
import io

MP = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'

section = """
## Phase 2916: 前段载体选择校正与（头,层）事件分解（early_carriers_selection_confirmed；载体 margin 由单层事件驱动） [2026-09-19 10:07]

### 原理
2915 确定 {7,8} 是层段绑定载体（后段稳定、前段换头 {27,31}/{7,27}），但前段载体只过了 family max-null gate，未做 2913 P3 口径的选择校正检验；且 V4 中 h7 margin 1.1797（vs V1 0.1771）的巨大值来源未知。本 Phase（2915 接续候选 A）：零前向产物域，对 2915 npz B_heads 做 (1) 前段载体选择校正检验、(2) 逐层 leave-one-out margin 分解定位 h7 双角色来源。

### 预注册（execution.json 先落盘冻结 e7661a8b）
- 数据：2915 npz B_heads_V1..V4（与 2913 rel 2.73e-08 已证）+ margins_V；标签 verbatim 2887。零前向，runtime 0.6 s。
- 锚（冻结）：a1 margins_V[V1 行] vs 2913 margins_h：Spearman >= 0.9999 且 max rel < 1e-5；a2 |margin({7,8} from B_heads_V1) - 0.28036| < 5e-3。V1 的 p3 复现检查（vs 2913 的 0.014925）登记为 replicate check 不作锚（bf16 级数据差可能移动边界置换）。
- 探针（冻结）：P1 选择校正贪婪检验（2913 P3 口径 verbatim：排序 + 贪婪 top-k + obs=max_k；null = 200 SEED=2896 置换下全流程重跑）对 V1/V2/V4；P2 h7/h8（V1、V4）与 V2 {27,31}、V4 {27} 的 leave-one-layer-out 剖面（Delta_j = 去层 j 后 margin - full，负 = 层 j 正贡献）+ 2910 口径 sign-margin 序列（辅助）；P3 每变体 top sign-margin（头,层）事件表。
- 判决映射（冻结）：anchor fail => anchor_fail_all_void；p3(V2) <= 0.05 且 p3(V4) <= 0.05 => early_carriers_selection_confirmed；恰一者 => early_carriers_partially_confirmed；否则 => early_carriers_not_confirmed。

### 结果（一次通过，0.6 s）
锚 2/2：a1 Spearman 1.000000 / **rel 0.00e+00**（2915 npz 的 margins_V[V1] 与 2913 margins_h 在 fp32 存储精度下逐位相同）；a2 m78 0.280360。

| 检验 | k_best | heads | obs margin | null p95 | p3 |
|---|---|---|---|---|---|
| V1 [26,36) | 2 | {7,8} | 0.28036 | 0.20998 | 0.014925（**与 2913 absdev 0.0 精确复现**） |
| V2 [20,26) | 1 | {27} | 0.78360 | 0.21966 | 0.004975 |
| V4 [16,26) | 1 | {7} | 1.17970 | 0.21222 | 0.004975 |

判决：p3(V2)、p3(V4) 双双 <= 0.05 => **early_carriers_selection_confirmed**。前段载体统计真实，且**单头即足**（k=1）。

### P2 逐层分解（核心发现：载体 = （头,层）事件）

| 载体 | full margin | 主驱动层 | loo delta（去该层后） | sign peak |
|---|---|---|---|---|
| V4 h7 | 1.17970 | **L19** | -1.059（去后仅 0.121） | L19 1.3463 |
| V2 h27 | 0.78360 | **L24** | -0.773 | L24 0.8106 |
| V4 h27 | 0.78650 | **L24** | -0.661 | L24 0.8106 |
| V2 h31 | 0.44323 | **L22** | -0.381 | L22 0.4561 |
| V4 h8 | 0.21920 | L23 | -0.143 | L20 0.4017 |
| V1 h7 | 0.17710 | **L34** | -0.157 | L34 0.1870 |
| V1 h8 | 0.13426 | **L34** | -0.071 | L34 0.2973 |

每个载体头的聚合 margin 几乎完全由**单一（头,层）事件**驱动（去层后 margin 崩 60-90%）。

P3 全变体 top sign 事件：V1 = h17@L28 (0.568)、h10@L34 (0.493)、h8@L34；V4 = h7@L19 (1.346)、h24@L23 (1.016)、h13@L22 (0.823)。

### 判决
early_carriers_selection_confirmed。

### 发现（结构读法）
1. **"载体头"实际是"（头,层）事件"**：h7 的跨段双角色 = 两个不同事件（h7@L19 前段强分离、h7@L34 后段抵消结构）；头身份跨窗口变化是因为不同窗口包含不同事件。
2. **2913 P3 的 {7,8} 显著性 = 同层（L34）双事件之和**：h7@L34 + h8@L34 都在 L34——提示是"层 34 机制招募两个头"，而非"跨层头对机制"。
3. 前段通道正 margin（+0.16）由 h7@L19 单事件主导；后段通道负 margin 的抵消结构由 L34 双事件 + 其余头负贡献构成。
4. 选择校正下前段载体单头 p=0.005（比后段 {7,8} 集合 p=0.015 更显著）——前段分离更强、更集中。

### 硬伤
- （头,层）事件表基于冻结窗口集合（16-36 层段内 4 窗口），未覆盖全层（0-15 层未扫描）；事件显著性用选择校正（窗口内 32 头族），未做跨窗口多重校正。
- leave-one-layer-out 量化贡献但未建立因果（单层 margin 高不等于该层因果必需；perturb 验证未做）。
- sign-margin 是 2910 口径（sign 外积 Gram），与聚合 row-norm margin 不同度量——两者一致性是定性观察。

### 结论
前段载体选择校正确认（h27、h7 单头 p=0.005）；载体 margin 由单层（头,层）事件驱动——h7@L19（前段 1.06 贡献）、h7@L34 + h8@L34（后段）、h27@L24、h31@L22。载体概念的最终形态：**（头,层）事件图谱**，头身份是事件在窗口内的投影。2913-2916 载体谱系收敛：通道 margin（2903）-> 头级分解（2913）-> 头身份复现（2914）-> 层段绑定（2915）-> 单层事件（2916）。

### 文件
- 脚本 tests/glm5/phase2916_early_carrier_selection.py（a2053a30）
- 产物 phase2916/early_carrier_selection/：execution.json e7661a8b、result.json fc0a8b7a、early_carrier_selection.npz 89300ad9（sign_seq_V1..V4 全 32 头逐层、margins_V、margins_h_2913 对照）
- Ledger：M2916_early_carrier_selection + L14 再精化（measurements 55、L14 connects 23、ledger sha 2005b213）
- 工作区日志 2026-09-19.md 追加 2916 节

### 接续（2917 候选）
- A（主选）：（头,层）事件图谱全层扫描——前向捕获扩展到全部 36 层（每层 2886 类间差方向），构建 32x36 单层 sign-margin 矩阵 + 选择校正事件显著性（family = 全头全层），定位全部显著事件；检验 L19/L22/L23/L24/L34 之外是否还有未发现事件。
- B：h7@L19 与 h7@L34 事件关系——同头两层事件的 r_c 响应相关（同一头机制还是头内不同子空间）。
- C：跨模型（glm4-9b）同型（头,层）事件检验（载体定律的跨模型形态）。
"""

with io.open(MP, 'a', encoding='utf-8') as f:
    f.write(section)
print('OK memo 2916 appended')
