# -*- coding: utf-8 -*-
"""Phase 3180 closeout: MEMO 尾部 append（Phase 3180 + 预注册 3181）"""
import io

P = r"D:/AI2050/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md"
txt = """

---

## Phase 3180: G5-B5 FTR-15 held_out 实体留出臂——E1→E2 升级 PASS（零 GPU，registry v1.6→v1.7 + atlas_v1_8.html） 05:15

**verdict: ftr15_upgraded_E2_predictive——FTR-15（输出等价类边界：局部薄邻域等价成立、子空间商结构被拒）按实体 LOO 留出臂完成 E1→E2 升级：16 锚经 3157 collect ei 标签核实恰为 16 实体每实体一行，16 fold 全部与全量同 quotient class（ratio_train < 3.0）且 max |log(ratio_train/ratio_full)| = 0.0300/0.0278/0.0182（4b/14b/glm4）全部远低于 0.25 预注册 bar；held 锚入表，upgrade_log #6，FTR-15 成为第 17 个 E2_predictive 特征（E2×17/E3×3/E1×3）。scope_limits 明确保留：「局部等价」不外推为全局商结构（F1 商结构拒绝不变）。**

**verdict（重复二）：装置全绿——10 源文件 sha 断言通过（3158 collect c4d9b9fb/7f53f3a0/2ea72cfe + per-model result 90e734fd/1f532016/c49614fc + summary 7ed85b36 + registry/gap/3162 基座）；G1 装置门：由 BUD 按封存 ratio 定义重算全量值 1.028348/1.331677/1.247260 对封存 3158 result drift 全部 0.00e+00；G3 方向性描述：held 锚 null/rand 中值比 >1 的计数 10/16、16/16、15/16（report only，不设门——4b 的 6 个反向锚与 ratio≈1.03 的弱信号一致，诚实披露）；SMOKE（3 fold）与 FULL（16 fold）共享 fold 数值逐位一致（max_band 0.0300/0.0200/0.0181），装置模式无关；execution.json design 8e38ff2f；FULL res e97867f4 / seal ce9f80d4（arm res f1689e94 / a3c1b53e）。**

**verdict（重复三）：独立复核 ALL PASS（独立进程，tests/gpt5_temp/phase3180_verify_ledger.py）——sha 断言（reg17 f3ad070f / html 1dc09043 / final seal ce9f80d4）；22 条 legacy 特征 + failures 逐条字节保全；FTR-15 anchors 2→3（held 锚 src=p3180 带 tags cross_model+held_out）；upgrade_log 5→6（node=FTR-15）；ratio 三模型 drift ≤1e-9 现场重断言；ledger append-only（measurements 331→332，#332=3180，前缀逐条等价，pre 54b10689 / post 8d03febc，ledger_sha256_8 按 Phase 8 先例 stale 不重算）。编写期修正 1 次（先于 FULL 观测）：smoke 标志从 design dict 移出（属运行模式非设计内容，消除 SMOKE/FULL 冻结 sha 漂移）；1 处源路径修正（3162 registry 文件名）；均不影响数值。**

### 升级读数（result 现场渲染）

| 读数 | 4b | 14b | glm4 | 门/判定 |
|---|---|---|---|---|
| ratio 重算（G1） | 1.028348 | 1.331677 | 1.247260 | vs 封存 drift 0 |
| max log band（G2，16 fold） | 0.0300 | 0.0278 | 0.0182 | <0.25 全过 |
| 全 fold 同 class <3.0 | 是 | 是 | 是 | G2 全过 |
| c(e)>1 锚计数（G3 描述） | 10/16 | 16/16 | 15/16 | report only |

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3180/g5b5_ftr15_heldout/` {execution.json design 8e38ff2f, registry v1.7 f3ad070f, atlas_v1_8.html 1dc09043（704 字段 0/0/0/0）, result res e97867f4/seal ce9f80d4, final_result, smoke res 518425f7/8d7ebb87}。脚本 `tests/glm5/phase3180_g5b5_ftr15_heldout.py`（渲染器 verbatim 复用 phase3179 模块）。E1 剩余：FTR-04、FTR-23（held_out 对关系特异特征为测量构造，留设计）。

### 接续：预注册 3181

**3181 = G5-B6 FTR-04 glm4 弱特异性归因（零 GPU，全维上界 vs S_class 曲线定判）**——FTR-04（S_class 类子空间重建）在 glm4 的读数弱于 qwen 双模型（3162 audit 登记），归因二选一：(a) glm4 类质心差分方向有效维数更低（S_class 行空间更病态，3165 补记已证 float64/32 敏感带 0.27°）；(b) 类端口读出谱更分散（top64 能量占比更低）。协议：3169/3166 封存 npz 复用，逐模型计算 S_class 行空间奇异值谱（条件数、有效秩 e90/e99）+ 谱外行投影能量，门（freeze 前细化）：若 glm4 有效秩 ≤ qwen 双模型最小值 ×0.8 => (a) 确证（registry counter_evidence 量化追加）；若条件数同量级而投影能量显著更低 => (b)；两者皆否 => 弱特异性归于采样噪声（FTR-04 直接升 E2 候选）。协议细节 freeze 前从 3166/3169 脚本 verbatim 导出。挂账不变：FTR-23 held_out 设计、GAP-4 残留 1.8×、5 appendix、N 线 P3–P7 补 Ledger、跨线账本补丁施加确认、水果类、K4、UI v1.6 客户端改造。
"""

with io.open(P, "ab") as f:
    f.write(txt.encode("utf-8"))
t = open(P, "rb").read().decode("utf-8")
assert t.count("## Phase 3180:") == 1 and t.find("## Phase 3180:") > t.find("## Phase 3179:")
assert "预注册 3181" in t
print("MEMO appended + read-back OK")
