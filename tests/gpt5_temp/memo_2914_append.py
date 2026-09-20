# -*- coding: utf-8 -*-
"""memo_2914_append.py -- append Phase 2914 section (append-only)."""
import io

MP = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'

section = """
## Phase 2914: h7/h8 头身份跨 run 复现与头级 W_VO 谱（head_identity_reproduced；载体无静态谱身份） [2026-09-19 09:50]

### 原理
2913 发现 qwen_attn 负 margin 是头级结构：h7（0.1771）/h8（0.1343）两个头承载 mlp 量级的正语言分离，其余 30 头净抵消（{7,8} 贪婪重构 0.28036 vs 全通道 -0.02139，选择校正 p=0.01493），但 2913 硬伤条款要求头身份跨 run 确认后才能写入正式结论。本 Phase 两个任务：(1) Run B 按 2913 v2 协议 verbatim 重跑前向，检验头身份/排序/重构的精确复现；(2) 零前向权重域头级 W_VO 谱分析——载体头是否有静态谱特殊性（若 margin 载体身份可由权重谱预测，则响应结构是权重结构的投影；若不能，则身份只活在数据依赖的响应里）。

### 预注册（execution.json 先落盘冻结 ee279e84）
- Run B：SEED=2896、eps=1.0、pos 1、window [26,36)、57 词 verbatim 2887、dirs 自 2886 S_last 重算、conds same/func/null、null_tids rng 顺序、o_proj 输入 pre-hook 捕获——全部 verbatim 2913 v2。
- 权重域：每头 W_VO_h = W_O_h @ Wv_kv[kv(h)]（GQA 8kv rep4）；瘦 SVD 技巧——A(m,128)·B(128,n) 的奇异值 = 中间 128x128 矩阵 S_A(V_A^T U_B)S_B 的奇异值；composite W_VO = Wo@M 全 SVD（2903 口径 verbatim：M 用 fp64 zeros、Wo(fp32)@M(fp64)->fp64 matmul、fp64 SVD）；zf 头响应 va_h = W_O_h @ (Wv_kv[kv(h)] @ (g_attn*d_q))。
- 锚（冻结）：a1 块=整体恒等 < 1e-9；a2 margin vs 2903 < 5e-3 且 acc 差 <= 2 词；a3 composite 谱 vs 2903 weights_descriptive 全 10 层（t12 绝对 1e-4、PR/zf_gain 相对 1e-4、zf_cos 绝对 1e-4）；a4a sum_h va_h == composite va < 1e-9；a4b 瘦 SVD 自检 < 1e-10。
- 判决映射（冻结）：anchor fail => anchor_fail_all_void；spearman>=0.999 且 top2=={7,8} 且 family gate 复现 且 {7,8} 重构差<5e-3 => head_identity_reproduced；elif spearman>=0.9 => margin_spectrum_reproduced_identity_shifted；elif >=0.5 => head_ordering_partially_reproduced；else => head_ordering_not_reproduced；P3/P4 描述性。

### 执行史
run1 作废：a4b 自检广播 bug——sv_mid 返回 128 个奇异值而 svd(A@Bm)（300x200）返回 200 个（含 72 个精确零尾），(128,) vs (200,) 崩溃；修复为 leading-128 比较。run2（正式，66.3 s）。

### 结果
锚 5/5：a1=1.22e-15；a2 margin -0.02139（stored -0.01946，acc 差 1 词）；a3 10 层全过 worst 4.992e-06（L26 t12 1.018649 vs 1.01865、PR 764.72 vs 764.71987；L35 t12 1.527215 vs 1.52721、PR 791.66 vs 791.65641）；a4a=3.08e-15；a4b=2.96e-15。

| 探针 | 结果 |
|---|---|
| P1 relB | B_heads 跨 run 相对差 3.16e-08（fp32 捕获域，与 e2run 3.88e-8 同量级）；rel_agg 3.66e-08 |
| P1 排序 | Spearman=1.000000、Kendall=1.000000（32 头逐位一致） |
| P1 top2 | {7,8} 相同；top5 数字逐位相同 0.1771/0.13426/0.07888/0.06009/0.05537 |
| P1 gate | 复现：h7 0.17710 > p95_max 0.17474（与 2913 同数） |
| P1 重构 | {7,8} margin 0.280360 vs ref 0.28036，absdiff 0.0 |
| P2 | flips_h 逐位相等；p_flip_min 0.089844（交替仍无单头载体） |
| P3 | k=2 heads [7,8] margin 0.28036 p=0.01493（与 2913 完全一致） |
| P4 | t12 排名 h7=30/32、h8=4/32；PR h7=18、h8=32；zf_gain h7=14、h8=30；zf_cos h7=16、h8=18 |

P4 相关性（谱统计 vs margins_h 的 Spearman）：t12 0.0、PR -0.0114、zf_gain -0.3046、zf_cos -0.1785——全部无预测力。

### 判决
head_identity_reproduced（四条件全过）。

### 硬伤
- Spearman=1.0 是同协议同 seed 复现的必然：本 Phase 确认的是"2913 数字非单次运行伪影、无未记录随机性泄漏"，不证明 {7,8} 身份对窗口/词表/方向族选择的鲁棒性（换窗口协议才能回答）。
- P4 仅覆盖 [26,36) 窗口与单一语言方向族（2886 S_last 类间差方向）；谱-响应关系外推有限。
- run1 广播 bug 已按纪律作废重跑（产物先删后跑）。

### 结论
h7/h8 头身份正式入账（2913 硬伤条款解除）：attn 通道的正语言分离由两个特定、精确可复现的头承载，其身份在静态权重谱中不可见（谱排名 4-32 名散布、相关性 ~0），只存在于数据依赖的响应中——"结构在响应不在权重"在头级别再次成立（与 L14 谱系读法一致）。margin 载体的可复现性链完整：通道 margin（2903/2913 三方锚）-> 头级分解（2913）-> 头身份复现（2914）。

### 文件
- 脚本 tests/glm5/phase2914_head_identity_replication.py（7c9db568）
- 产物 phase2914/head_identity_replication/：execution.json ee279e84、result.json a0142e60、head_identity_replication.npz 1fb12b0a（B_heads_runB 32x57x10、B_agg_runB、B_out_runB、margins_h_runB、flips_h_runB、spec_h 10x32x4、margins_h_2913/flips_h_2913 对照）
- Ledger：M2914_head_identity_replication + L14 confirmed（measurements 53、L14 connects 21、ledger sha e0b55261）
- 工作区日志 2026-09-19.md 追加 2914 节

### 接续（2915 候选）
- A（主选）：载体身份鲁棒性域——换窗口（如 [20,26) 或 [36,42)）与/或换方向族重跑头级分解，检验 {7,8} 是跨协议参数稳定还是窗口局域现象；直接回答 P4 遗留的"runtime-response 决定论的适用范围"。
- B：{7,8} 响应结构解剖——r_c 的逐词/逐层分解 + 与 B3_mlp 通道响应的逐词相关（两头承载的语言分离与 mlp 载体是同一信号还是独立信号）。
- C：glm4-9b 同型头级分解（跨模型：glm4 attn eigen+orth 层级是否同样由少数头承载）。
"""

with io.open(MP, 'a', encoding='utf-8') as f:
    f.write(section)
print('OK memo 2914 appended')
