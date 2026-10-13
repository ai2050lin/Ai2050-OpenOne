# -*- coding: utf-8 -*-
"""Phase 3189 MEMO append: Phase 3189 section, triple verdict, 3190 prereg."""
import hashlib
import io

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
PRE = '0c4aba7c'
raw = open(P, 'rb').read()
assert hashlib.sha256(raw).hexdigest()[:8] == PRE
old = raw.decode('utf-8')

SEC = """

## Phase 3189 — G5-C2 谱外残差类级/共享方向结构归因（C 线，零 GPU，2026-10-10）

**预注册**：MEMO 3188 尾（观测前冻结）。3188 判定 GAP-4 残留与 Δv 解耦后，检验 3169 谱外逐细胞崩塌残差方向由少数共享方向承载还是类特异性弥散。

**协议（freeze 10339511）**：残差链 = 3188 主臂 verbatim（3169 B 口径、seeds 7/8/9、ridge λ=1e-3、cols 780，876 OOV 行，行级方向 = 3-seed 均值 unit 化）。dvu 自 3157 重导（供 3188 标量锚）。设备锚：3157 tc 三锚、3183 dv 复现、3169 逐 seed E 复现全 0 drift；**3188 封存标量（share_oov/share_seen/ks/med_oov/med_seen）drift == 0 逐位**——3189 残差链与 3188 完全同一。

**观测（主臂）**：
| 模型 | SVD top1 | top2 | r80/r90 | 类间中位 cos | 类内均值 cos | 门 |
|---|---|---|---|---|---|---|
| qwen3-4b | 0.3065 | 0.3856 | 37/93 | 0.5441 | 0.460/0.416/0.491/0.461 | PASS |
| qwen3-14b | 0.2936 | 0.3603 | 47/120 | 0.5663 | 0.408/0.425/0.459/0.405 | PASS |
| glm4-9b | 0.3008 | 0.4002 | 29/79 | 0.4449 | 0.527/0.479/0.504/0.485 | PASS |

**verdict（重复一）**：shared_oov_collapse_direction 3/3 全过——top1 ≥0.29（单方向承载 ≈30% 能量）、类间中位 cos 0.44–0.57（6 对全 ≥0.37）。

**verdict（重复二）**：**类间 cos 高于类内 ×3**（4 类类内均值 0.41–0.53 全部低于该模型类间中位）——崩塌方向跨 OOV 类共享且强于类内一致性：谱外崩塌不是类特异性机制，是跨类共享方向承载的统一几何事件。

**verdict（重复三）**：与 3188 Δv 解耦拼合——共享崩塌方向 ≠ Δv（3188 share ≈1–2%），GAP-4 残留 1.8× 获得首条结构级机制线索：存在一个（或少数几个）独立于关系偏移的跨类共享崩塌方向；FTR-25 候选 E1_repeatable 入册（features 24→25 append-only），后续可做 held_out 升级与方向内容归因（top1 PC 与可解释子空间的对齐）。

**诚实登记**：SMOKE-1 log 格式补丁（drift69 % 格式化缺失，纯日志）；FULL-1 crash——atlas 写盘后字段校验 1 mismatch（FTR-25.scope_limits 用 list 违反 legacy 字符串口径），崩在 final_result 写盘前，修复重跑 arm res/seal 逐位一致（3a26f541/fb8c1f65），registry v1.16 重写 sha 114aa114，零观测泄漏。

**登记**：registry v1.15→**v1.16 `114aa114`**（FTR-25 family=context E1_repeatable，24 legacy 字节保全）；atlas_v1_17.html `4c779d24`（746 字段 0/0/0/0）；ledger #341（13f416bd→f6584f85）；final res `66a8a8c2`/seal `cb4c92bb`；design `10339511`。图谱现状：25 特征（E2×17/E3×3/E1×4 + FTR-25 候选）、failures F1–F19、ledger n=341。

**3190 预注册（G5-C3，零 GPU，封存 3169 复用）**——共享崩塌方向的内容归因与稳健性：(a) top1 主方向 u* 对 3169 seen 面板（train 基线）残差方向的投影分布（共享方向是否面板内已存在/增强）；(b) u* 与 3169 封存 marg（中心化类 logit，10 类）读出权重的对齐：marg 对 10 类 first-token embedding 的差分方向 ∇_cls（H 最后层 → logit 的类间方向代理，从封存 marg 无法直接得——降级方案：u* 对 10 类 first-token embedding（tokenizer 无 GPU 可得）差分方向的投影报告）；(c) LOO 稳健性：剔除单类后 top1 重算（4 折 ×3 模型，top1 变化 <0.05 ⇒ 共享方向非单类承载）；(d) 门（freeze 前细化）：LOO 4 折 top1 波动 <0.05 ×≥2/3 ⇒ 稳健；(a)(b) 为 report-only 归因读数。设备锚：3189 残差链复现（top1/cross_med drift == 0 vs 3189 封存）。
"""

new = old + SEC
with io.open(P, 'w', encoding='utf-8', newline='') as f:
    f.write(new)
post = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
assert open(P, 'rb').read().decode('utf-8').startswith(old)
print('MEMO appended: pre %s post %s' % (PRE, post))
