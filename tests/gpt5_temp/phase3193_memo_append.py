# -*- coding: utf-8 -*-
"""Phase 3193 MEMO append (three-pass verdict + honest log + 3194 prereg)."""
import hashlib
import io

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
pre = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]

BLOCK = """

## Phase 3193 — G5-D1 u* 子空间成分分解（D 线开线，零 GPU，2026-10-10）

**预注册**：MEMO 3192 尾（观测前冻结）。C 线收官后，最后一问：共享崩塌方向 u* 是已知机制成分的归约还是新成分。

**协议（freeze 7af34fe6，含 SMOKE 期设计修正诚实登记）**：残差链 = 3192 verbatim；设备锚：3189/3190/3191 封存标量 drift == 0 逐位 ×3、3157 tc 锚 + 3183 dv 复现（cos_rl/norm_dv）、3169 逐 seed E 0 drift；report-only |cos(u*,v_c)| 与 3190 逐位一致——链跨 3189–3193 完全同一。**预注册降级（freeze 钉定）**：3165/3166 封存产物只含标量/奇异值、无方向向量，重导需重跑其协议超零 GPU 预算 → 门集降级为 {dvu（3157/3183 verbatim）、v_s×6（seen 6 类均值残差方向，738 全 seen 行）、u_seen（seen 残差 top-1 SVD）}；v_c×4 report-only。门（钉定）：DECOMPOSED iff max|cos| ≥0.5 ×≥2/3；NOVEL iff max <0.3 ×≥2/3。

**观测（主臂）**：
| 模型 | \\|cos(u*,dvu)\\| | max\\|cos(u*,v_s)\\| | \\|cos(u*,u_seen)\\| | gate_max | report-only \\|cos(u*,v_c)\\| |
|---|---|---|---|---|---|
| qwen3-4b | 0.1064 | 0.2130 | 0.1546 | 0.2130 | 0.806/0.762/0.836/0.819 |
| qwen3-14b | 0.0405 | 0.2183 | 0.3303 | 0.3303 | 0.807/0.797/0.852/0.808 |
| glm4-9b | 0.0240 | 0.1980 | 0.2457 | 0.2457 | 0.778/0.746/0.802/0.724 |

**verdict（重复一）**：**NOVEL 2/3**（4b、glm4 <0.3；14b 0.3303 中间带）——u* 对全部已知结构方向的最大 |cos| ≤0.33：**u* 不可归约为任何已知机制方向**。NOVEL 分支执行：FTR-26 u*_novel_direction 候选立项（family=context E1_repeatable，features 25→26 append-only，25 legacy 字节保全），registry v1.18→v1.19。

**verdict（重复二）**：|cos(u*,dvu)| = 0.106/0.041/0.024——u* 与关系偏移方向**方向级正交**（3188 DECOUPLED 的方向级确证：份额≈噪声 + 方向正交双重证据）；u* 与 seen 类均值/seen 主方向 |cos| ≤0.33 而 |cos(u*,v_c)| 0.72–0.85——u* 是 OOV 面板**内部生成**的几何方向，不携带 seen 侧结构。

**verdict（重复三）**：D 线开线确立核心事实：谱外崩塌的承载方向是一个**跨模型 held-out 可预测（E2）但机制上新颖**的方向——既非关系偏移、亦非 seen 类结构、亦非面板基线主方向。FTR-26 后续升格通道：held_out 臂（u* 预测留出类行方向，3191 协议换预测子）与因果/干预臂（E3）。

**诚实登记**：SMOKE crash ×3——(1) ALL_SEEN_ROWS 过滤漏实体约束（i<41，738 行断言拦截）；(2) 3183 目录名 g5b5_lowrank_check → g5b8_dv_lowrank；(3) 3183 per_model 键 glm4 → glm4-9b。isa/hasa 行分组修正（rl==0 在 i%4∈{0,1}，非前 32 行；3183 dv 锚 0.2385 漂移拦截）。FULL-1：SHA_ANCHOR 漏 p3192_result 键 → **设计修正重 freeze（bd6dbbc6→7af34fe6）**，删旧 execution.json 重跑；crash 前旧 registry v1.19（ca43f252）作废，以本次 4d64a153 为准。SMOKE/FULL per_model payload 逐位一致，零观测泄漏。

**登记**：registry v1.18→**v1.19 `4d64a153`**（FTR-26 append-only，**26 特征**）；atlas_v1_21.html `5a6d219f`（764 字段 0/0/0/0）；ledger #345（c6c3c24a→3583ef4f）；design `7af34fe6`；arm res `5465dd47`/seal `ce1d8c8d`；final res `e75c6e34`/seal `9164b1ec`。图谱现状：**26 特征（E2×18/E3×3/E1×4 含 FTR-26 候选）**、failures F1–F19、升级链 7、gap v1.6、ledger n=345。

**3194 预注册（G5-D2，零 GPU，封存复用）**——FTR-26 held_out 臂（E1→E2 通道）：检验 u* 自身的留出预测性：对每模型 4 折，v̂_held = 其余 3 类行方向均值（unit），报告留出类行方向与 **u*（全 876 行 SVD top-1，含留出类——非泄漏：u* 是方向汇总不是留出类标签）** 与 v̂_held 双预测子的中位 |cos|；门（freeze 前细化）：留出类行方向中位 |cos(u*)| ≥0.3 全 4 折 ×≥2/3 模型 ⇒ FTR-26 升 E2_held_out（registry v1.19→v1.20）；未过 ⇒ E1 维持 + counter_evidence。对照锚：3191 FTR-25 held_out 值（min med 0.39/0.42/0.37）同协议复算须 drift==0。设备锚：3189–3193 封存标量 drift == 0 逐位。挂账不变：5 appendix、N 线 P3–P7 补 Ledger、跨线账本补丁施加确认、水果类、K4。
"""

with io.open(P, 'a', encoding='utf-8', newline='') as f:
    f.write(BLOCK)
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
print('MEMO appended: pre %s post %s' % (pre, post))
