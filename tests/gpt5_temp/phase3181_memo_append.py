# -*- coding: utf-8 -*-
"""Phase 3181 closeout: AGI_GPT5_MEMO.md tail append（Phase 3181 记录 + 预注册 3182）"""
import io, hashlib

P = r"D:/AI2050/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md"
pre8 = hashlib.sha256(open(P, "rb").read()).hexdigest()[:8]

BLOCK = """

---

## Phase 3181: G5-B6 FTR-04 glm4 弱特异性归因——双臂定判 noise_level（零 GPU，registry v1.7→v1.8 + atlas_v1_9.html） 06:25

**verdict: noise_level_not_attributable——FTR-04 glm4 弱特异性（LOEO acc 0.4800 < 0.5 门）经几何臂 vs 读出臂双臂定判，两门皆 FALSE：几何臂（S_class 行空间奇异谱，3166 配方 verbatim，per-model W_U）kappa 4.81e7/4.60e7/4.50e7（glm4 未 ≥1.25× max qwen）、e90 8/8/8（无降）、PR 8.78/8.90/8.92（glm4 未 ≤0.8× min）=> G-a FALSE；读出臂（封存 3169 seen rows）proj_ratio 0.0763/0.0328/0.0294（glm4 0.0294 > 0.75×min qwen 0.0246，未过门）、LOEO mean_margin 2.6614/1.4164/5.2326（glm4 最大而非最小）=> G-b FALSE。判词：弱特异性既不归于 S_class 几何也不归于读出通路；E 级不变（FTR-04 维持 E1_repeatable），held_out 路线维持 3176 判闭。方向性诚实披露：glm4 proj_ratio 三模型最低（0.0294），与「读出通路略弱」方向一致但未过预注册门——不升格、不作独立机制挂账。**

**verdict（重复二）：装置全绿——8 源文件 sha 断言通过（3169 collect npz ×3 36eb4ff0/97f98575/9cc3f8b4 + p3169_result 5b51c2c1 + p2806_exec b291ea35 + registry v1.7 f3ad070f + gap v1.5 4fb3f37d + 3162 registry 00f15e98）；封存内容双锁（p3169 49430a39/018c6024 + ratio_B 锚 2.5388114997805062）；保真锚：LOEO acc 重跑 0.6019/0.5981/0.4800 与封存 3176 arm_a 逐位一致（判决式 verbatim）；SMOKE 与 FULL 数值逐位一致（装置模式无关）；execution.json design 7c567972；FULL res 1e95a19d / seal 4ad066e0；registry v1.8 d08e9064（23 条 legacy 特征字节保全，FTR-04 counter_evidence 量化追加 + 归因锚 append）；atlas_v1_9.html 79acbb6c（708 字段 0/0/0/0）。**

**verdict（重复三）：独立复核 ALL PASS（独立进程，tests/gpt5_temp/phase3181_verify_independent.py）——sha 断言（reg18 d08e9064 / html 79acbb6c / final seal 4ad066e0 / ledger post 72567879）；23 条 legacy 特征逐条字节保全；FTR-04 counter_evidence 仅追加（长度 3162版+1）；归因锚 asserts 数值与 result 一致；atlas data-k 708 字段回读 0/0/0/0；ledger append-only（measurements 332→333，#333=3181，前缀逐条等价，pre 8d03febc / post 72567879，ledger_sha256_8 按 Phase 8 先例 stale 不重算）。编写期修正 2 次（均先于 FULL 观测）：① 语法残留 placeholder 块删除；② cat_c69 构造键序修正（3176 原文为英文值→中文键查 CLS69_IDX，SMOKE-1 拦截 KeyError 'fruit'），均不触及数值。**

### 判词细节与诚实登记

- **双臂读数**：kappa（s1/s10）三模型同量级 ~4.5–4.8e7，e90 全 8（10 维 S_class 满有效秩），PR glm4 最高（8.92）——几何完全健康；proj_ratio glm4 最低但未过 0.75 门；margin glm4 5.23 反而最大（排除读出退化）。弱特异性（LOEO 0.48）不属于几何/读出二臂任一可解释机制，按预注册判 noise_level。
- **预注册口径差异登记**：MEMO 3180 tail 预注册文本写「两者皆否 => FTR-04 直接升 E2 候选」，而 freeze 钉定的 design gates 为「所有分支 E 级不变（counter_evidence 量化追加）」。以 freeze 为准（先于观测钉定），预注册文本偏差在此诚实登记；E2 升格需 held_out 构造，按 3176 判闭维持不动。
- **产物**：`tests/glm5/result/rdc_query_construction_20260913/phase3181/g5b6_ftr04_glm4_attribution/` {execution.json design 7c567972, registry v1.8 d08e9064, atlas_v1_9.html 79acbb6c, result res 1e95a19d/seal 4ad066e0, final_result, smoke res 08f6d9db/68bddf29}。脚本 `tests/glm5/phase3181_g5b6_ftr04_glm4_attribution.py`（渲染器 verbatim 复用 phase3179 模块）。E1 剩余：FTR-04（归因已闭合，held_out 构造留设计）、FTR-23。

### 接续：预注册 3182

**3182 = G5-B7 FTR-23 held_out 关系留出臂（零 GPU，封存 3157 collect npz 复用）**——E1 剩余两特征中 FTR-23（关系特异残差）的 held_out 对「关系」而言即留出构造：以 3157 collect 的 isa/hasa 双关系面板做 leave-one-relation-out fold（null 标定用全部关系重建、留出关系读数），r_held 与 r_full 差 <0.1 ×3 模型 => held 锚入表、FTR-23 升 E2_predictive；fail => F17 入失败账本（归因预注册下一 phase，同 3177→3178 先例）。协议细节 freeze 前从 3178/3157 脚本 verbatim 导出。挂账不变：GAP-4 残留 ~1.8× 结构性成分、5 appendix、N 线 P3–P7 补 Ledger、跨线账本补丁施加确认、水果类、K4、客户端图谱透镜端到端验证。
"""

with io.open(P, "r", encoding="utf-8") as f:
    txt = f.read()
assert "Phase 3181: G5-B6" not in txt, "already appended"
txt = txt.rstrip("\n") + BLOCK
with io.open(P, "w", encoding="utf-8", newline="") as f:
    f.write(txt)
post = hashlib.sha256(open(P, "rb").read()).hexdigest()[:8]
print("MEMO appended pre=%s post=%s" % (pre8, post))
