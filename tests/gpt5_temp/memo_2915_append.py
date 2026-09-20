# -*- coding: utf-8 -*-
"""memo_2915_append.py -- append Phase 2915 section (append-only)."""
import io

MP = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'

section = """
## Phase 2915: 载体身份鲁棒性域——窗口与方向族扫描（head_identity_broadly_stable；{7,8} 是层段绑定载体） [2026-09-19 10:00]

### 原理
2914 确认 h7/h8 头身份跨 run 精确复现且无静态谱身份（载体身份由运行时响应决定）。遗留问题（2914 接续候选 A）：身份对协议参数（窗口、方向族）的稳定域是什么？由于身份活在数据依赖的响应里，换窗口/换方向族就是换响应——直接重问载体选择。

### 预注册（execution.json 先落盘冻结 46bf692e）
- 5 变体：V1 [26,36) + 2886 层匹配类间差方向（2913 参照口径）；V2 [20,26)；V3 [30,36)（V1 后半子窗口）；V4 [16,26)；V5 = V1 窗口 + 2887 固定全局 lang_dir（stale 方向族对照，描述性不进判决轴）。
- 协议 per 变体：SEED=2896、eps=1.0、pos 1、57 词 verbatim 2887、conds same/func/null、null_tids rng 顺序、o_proj 输入捕获、200 SEED=2896 标签置换（Sm 固定 + mask 置换，2903 口径）family gate。一次前向捕获共享（attnin 与方向无关），每变体独立投影分解。
- 锚（冻结）：a1 全变体块恒等 max < 1e-9；a2 V1 margin vs 2903 < 5e-3 且 acc 差 <= 2 词；a3 V1 margins_h vs 2913 npz：Spearman >= 0.9999 且 max rel < 1e-5。
- 判决映射（冻结）：anchor fail => anchor_fail_all_void；n_top2（{7,8} 同时进 top2 的窗口数，V1-V4）== 4 => head_identity_protocol_general；>= 2 => head_identity_broadly_stable；== 1 => head_identity_partially_local；0 => head_identity_window_local；n_top5 并行登记。

### 结果（一次通过，32.1 s）
锚 3/3：a1 max 1.42e-15（V1 1.029e-15 / V2 1.051e-15 / V3 1.029e-15 / V4 1.051e-15 / V5 1.419e-15）；a2 V1 margin -0.02139（stored -0.01946，差 1 词）；a3 Spearman 1.000000 / rel 2.73e-08（**第三次连续精确复现**，与 2914 relB 3.16e-8 同量级）；m78_V1 0.280360 = 2914 ref。

| 变体 | top2 | h7 (rank) | h8 (rank) | gate | 通道 margin / acc |
|---|---|---|---|---|---|
| V1 [26,36) | {7,8} | 0.17710 (#1) | 0.13426 (#2) | True (p95 0.17474) | -0.02139 / 0.632 |
| V2 [20,26) | {27,31} | 0.12731 (#14) | 0.20459 (#8) | True (p95 0.21765) | +0.03671 / 0.789 |
| V3 [30,36) | {8,7} | 0.15076 (#2) | 0.16431 (#1) | False (p95 0.20731) | -0.02295 / 0.632 |
| V4 [16,26) | {7,27} | 1.17970 (#1) | 0.21920 (#10) | True (p95 0.20486) | +0.16005 / 0.842 |
| V5 stale | {4,20} | 0.01172 (#14) | -0.01160 (#26) | False (p95 0.15109) | +0.00265 / 0.579 |

判决轴：n_top2 = 2/4（V1+V3），n_top5 = 2/4 => **head_identity_broadly_stable**。

跨变体 Spearman（margins_h）：V1-V3 0.9091；V2-V4 0.8658；V1-V2 0.0257；V1-V4 0.0762；V1-V5 0.3039——**排序跨层段近正交、段内一致**。

### 判决
head_identity_broadly_stable。

### 发现（结构读法）
1. **{7,8} 身份是层段绑定的，不是协议全局的**：后段窗口 [26,36) 与其子窗口 [30,36) 中 {7,8} 稳定占据 top2（V3 内部 1/2 名互换，top5 与 V1 重叠 4/5）；前段窗口（<=26 层）载体换头（V2 {27,31}；V4 {7,27}）。
2. **通道 margin 符号是层段属性**：前段 attn 通道正且 acc 高（V2 +0.037/0.789、V4 +0.160/0.842），后段负（V1 -0.021/0.632）。2903 登记的负 attn margin 是 [26,36) 口径——前段 attn 通道本身"工作良好"。因此 **{7,8} 承载的是后段抵消结构，不是全局语言载体**。
3. **h7 是唯一跨段头**（V1 #1 / V3 #2 / V4 #1，V4 margin 1.1797 量级远超其他头），h8 严格后段（V2 rank8 / V4 rank10）。
4. **头级分离要求层匹配方向族**：V5 stale 固定 lang_dir 下载体消失（gate False、通道 margin ~0）——2903 读出谱系（qwen subspace-tolerant，stale 方向弱）在头级别复现。
5. V3 的 6 层窗口 family null 更宽（p95 0.207 vs V1 0.175），{7,8} 虽居 top2 但不超自身 family p95——短窗口的家族显著性下降是窗口长度效应，非载体消失。

### 硬伤
- 判决轴只覆盖窗口几何（16-36 层段），未扫描方向族全空间（V5 仅 1 个 stale 对照）；前段载体 {27,31}/{7,27} 的统计显著性只有 family gate（V2/V4 present），未做 2913 P3 口径的选择校正重构检验。
- V4 中 h7 margin 1.1797 的巨大值未做逐层分解归因（哪几层贡献主导未知）。
- n_top2=2/4 落在 broadly_stable 档的边界解释依赖窗口集合选择（若加更多后段子窗口，n_top2 只会增；若加更多前段窗口则减）——窗口集合是预注册冻结的，但结论应读作"该冻结集合上的 2/4"。

### 结论
{7,8} 头身份的适用域确定：**后段窗口（26-36 层）稳定的抵消结构载体**（后段内窗口几何鲁棒，含子窗口与名次互换），前段窗口换载体且通道符号翻转，stale 方向族载体消失。2913/2914 的"h7/h8 是 attn 负 margin 载体"结论应限定为后段口径——这与 2903 通道 margin 本身就是 [26,36) 口径自洽。载体概念是（窗口, 方向族）相对的，"头身份"在冻结协议内精确可复现，跨协议则迁移为"层段 + 头"的联合结构。

### 文件
- 脚本 tests/glm5/phase2915_carrier_robustness_domain.py（acc490a5）
- 产物 phase2915/carrier_robustness_domain/：execution.json 46bf692e、result.json 07314fa6、carrier_robustness_domain.npz d4b85302（B_heads_V1..V5、margins_V 5x32、margins_h_2913 对照）
- Ledger：M2915_carrier_robustness_domain + L14 再精化（measurements 54、L14 connects 22、ledger sha 9e830db5）
- 工作区日志 2026-09-19.md 追加 2915 节

### 接续（2916 候选）
- A（主选）：前段载体形式化——对 V2/V4 的 top 载体（{27,31}/{7,27}）跑 2913 P3 口径的选择校正重构检验 + V4 中 h7 margin 1.1797 的逐层贡献分解（h7 跨段双角色解剖：前段 1.18 vs 后段 0.177 的来源层）。
- B：层段符号结构的通道级确认——前段/后段通道 margin 的窗口滑窗扫描（逐层 leave-one-out margin），定位符号翻转的边界层。
- C：2914 候选 B 遗留——{7,8} 与 B3_mlp 载体的逐词相关（同一语言信号还是独立信号）。
"""

with io.open(MP, 'a', encoding='utf-8') as f:
    f.write(section)
print('OK memo 2915 appended')
