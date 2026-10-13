# -*- coding: utf-8 -*-
"""Phase 3188 MEMO append: Phase 3188 section, triple verdict, 3189 prereg."""
import hashlib
import io

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
PRE = 'b49c04d3'
raw = open(P, 'rb').read()
got = hashlib.sha256(raw).hexdigest()[:8]
assert got == PRE, ('pre sha', got, PRE)
old = raw.decode('utf-8')

SEC = """

## Phase 3188 — G5-C1 GAP-4 残留 1.8× 的 Δv 方向归因（C 线开线，零 GPU，2026-10-10）

**预注册**：MEMO 3187 尾（观测前冻结）。以 3183 verbatim Δv 方向对 3169 谱外面板封存逐细胞崩塌残差做投影归因——勘察确认 3169 有逐细胞 H（730 细胞×37 层），无需降级为聚合检验；协议细节 freeze 前从 3169/3175/3183 脚本 verbatim 导出。

**协议（freeze 977bb327，先于观测钉定）**：
- Δv：3157 collect（dC=H[ctx1]−H[ctx0] @ KOUT=NL−1，dCn 行归一，v_rl=unit_mean(rel 32)，Δv=v_isa−v_hasa）。层兼容性钉定：3157 readout 与 3169 readout 三模型同层（35/39/39），同一残差流槽位，跨封存投影合法。
- 主臂：3169 B 口径 verbatim（seeds 7/8/9、split_s1_246、ridge λ=1e-3、cols 780），逐 (row,seed) 残差方向 q=unit(Y−XW)，投影 proj=q·dvu；行级聚合 OOV 876 行（3 seed 均值）vs seen S1-test 行（出现 seed 均值）。门（钉定）：COUPLED iff KS≥0.15 且 perm_p<0.01 ×≥2/3 模型 且中位差符号一致 ≥2/3（置换 n_perm=20000，rng 20260388）。
- 附录臂（report-only）：3175 out-arm k=8 verbatim 复现（ne=105 扩展列、out_triples 装配、te_oov=876−96=780 行），锚=sealed out_curve['8'] per-seed E <1e-9。

**设备门（全 0.0 drift）**：3157 tc 三锚 <1e-9；3183 dv 复现（cos_rl/‖Δv‖/frac_full/pc_share/med_cos）drift=0；3169 逐 seed E_seen/E_oov 复现 drift=0；3175 out-arm k=8 锚 drift=0 ×3。SMOKE/FULL 臂数字逐位一致。

**观测（主臂）**：
| 模型 | KS | perm_p | 中位差 | share_OOV | share_seen | share_ctrl(≈1/D) | 门 |
|---|---|---|---|---|---|---|---|
| qwen3-4b | 0.3344 | 5e-5 | +0.0621 | 0.00883 | 0.00995 | 3.6e-4 (1/D=3.9e-4) | PASS |
| qwen3-14b | 0.1003 | 0.01050 | −0.0031 | 0.00905 | 0.01419 | 1.2e-4 (1/D=2.0e-4) | FAIL |
| glm4-9b | 0.1913 | 5e-5 | −0.0104 | 0.00596 | 0.01934 | 2.4e-4 (1/D=2.4e-4) | PASS |

**verdict（重复一）**：dv_direction_decoupled_from_oov_collapse——2/3 模型过 KS+p 门但**中位差符号相反**（4b +0.0621 vs glm4 −0.0104，14b ≈0），符号一致性门 FAIL ⇒ DECOUPLED。

**verdict（重复二）**：GAP-4 残留 1.8× 与关系偏移方向 Δv **在方向级解耦**——方向份额读数（report-only）三模型全部 OOV≤seen（0.88%≤1.00%、0.91%≤1.42%、0.60%≤1.93%）且都仅 ≈1–2%（对照 ≈1/D），Δv 方向不承载谱外崩塌残差的主向。

**verdict（重复三）**：counter_evidence 已 append 到 FTR-24（statement/anchors 字节保全）；FTR-25 候选**不立**。T2 链条（3177–3187）结论进一步巩固：R_rel=关系级方向 Δv 不与谱外泛化崩塌共享机制通道，GAP-4 残留需在 Δv 之外找机制（端口残差幅度级、类内子空间、或类间共享崩塌方向）。

**登记**：registry v1.14→v1.15（`0a55d2e7`，features 24 不变、failures 19、FTR-24 counter_evidence append-only）；atlas_v1_16.html（`384d0834`，734 字段 0/0/0/0）；ledger #340（da29d7d6→13f416bd）；final res `8c1d94a3`/seal `ad0444ad`；design `977bb327`。无 SMOKE/FULL 修正、无 crash。

**3189 预注册（G5-C2，零 GPU，封存 3169 复用）**——谱外残差的类级/共享方向结构归因：对 3169 逐细胞崩塌残差矩阵（OOV 876×D per model）做 (a) 类级分解——4 个 OOV 类间残差方向均值 cos 矩阵与类内/类间均值比（类特异性 vs 类间共享）；(b) 残差方向 SVD 谱——top-k 能量占比与 r80/r90（低秩共享崩塌方向假设）；(c) 门（freeze 前细化）：若存在跨类共享方向（top1 占比 ≥0.2 ×≥2/3 模型 且类间中位 cos ≥0.2 ×≥2/3）⇒ 谱外崩塌由少数共享方向承载，GAP-4 残留获首条结构线索并登记 FTR 候选；否则崩塌方向类特异性弥散 ⇒ counter_evidence。设备锚：3169 逐 seed E 复现 <1e-9、残差方向与 3188 主臂逐位一致（抽样 cos=1 断言）。
"""

new = old + SEC
with io.open(P, 'w', encoding='utf-8', newline='') as f:
    f.write(new)
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
back = open(P, 'rb').read().decode('utf-8')
assert back.startswith(old), 'prefix not preserved'
assert back.count('verdict（重复一）') >= 1 and back.count('verdict（重复二）') >= 1 \
    and back.count('verdict（重复三）') >= 1
print('MEMO appended: pre %s post %s' % (PRE, post))
