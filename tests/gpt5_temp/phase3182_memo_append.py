# -*- coding: utf-8 -*-
"""Phase 3182 closeout: AGI_GPT5_MEMO.md tail append（Phase 3182 记录 + 预注册 3183）"""
import io, hashlib

P = r"D:/AI2050/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md"
pre8 = hashlib.sha256(open(P, "rb").read()).hexdigest()[:8]

BLOCK = """

---

## Phase 3182: G5-B7 FTR-23 held_out 关系留出臂——LORO cross-null 定判 FAIL（F17 入账，零 GPU，registry v1.8→v1.9 + atlas_v1_10.html） 11:05

**verdict: ftr23_held_out_gate_fail——FTR-23（关系特异残差）held_out 升级被预注册门拒绝：r_full 按 3178 verbatim 配方重算 drift 0.00e+00 ×3 模型（0.0918/0.0951/0.1289 保真锚）；新增跨关系 null（标定=另一关系 30 细胞实体留出、读数=留出关系 2 细胞）下 p_x(m,rel)=0.1299/0.1342（4b isa/hasa）、0.1507/0.1035（14b）、0.1579/0.1564（glm4）——5–8 倍于同关系内采样罚（~0.02），把关系缺口 g 整个吸收；r_held=−0.0153/−0.0097/−0.0079，delta=0.1070/0.1048/0.1368 全部 ≥0.1 门 ⇒ FAIL：held_out 升级拒绝，F17 入失败账本，FTR-23 保持 E1_repeatable，counter_evidence 量化追加。**

**verdict（重复二）：装置全绿——12 源文件 sha 断言通过；3157 三模型+summary 双锁；3177（d451030f/7b746eb2）+3178（4f521d9b/1f166645）封存 result 双锁、数字从钉定文件加载零转录；96/96 同关系 null sanity + 96/96 cross-null sanity（finite、|p|<0.5）全过；装置 drift <1e-9；3177 fold 复制 drift <1e-6；SMOKE 与 FULL 数值逐位一致；execution.json design ea0a1e7a；FULL res 5f08426f / seal b6a57f07；registry v1.9 59a91512（22 条 legacy 特征字节保全，FTR-23 counter_evidence 追加 + failures F17 追加，upgrade_log 6 不变）；atlas_v1_10.html 0f24b739（713 字段 0/0/0/0）。**

**verdict（重复三）：实质读数（本轮最有价值产出）——关系特异残差在跨关系标定下不作为「残差」复现，而是以对称跨关系读出罚重现：p_x 的 isa/hasa 两读同模型差 ≤0.047（isa↔hasa 方向对称），且三模型 p_x 同量级（0.10–0.16）。这与「关系偏移结构」（relation-offset，Δv_isa↔hasa 系统性方向差）一致而与「关系非对称噪声」不一致——FTR-23 的 statement（T_C 按关系分解、跨关系系统性偏移）获得第三条独立量化证据（p_x 直接读出偏移量级），但 held_out 形式判 FAIL（残差本体在留出标定下被吸收）。诚实登记 1 次 pinning 偏差：预注册文本写「null 标定用全部关系重建」，freeze 钉定非泄漏版（标定=另一关系 only），偏差入 MEMO。独立复核 ALL PASS（tests/gpt5_temp/phase3182_verify_independent.py）。**

### 判词细节

- **为什么 FAIL 是正确判**：门问的是「r_held ≈ r_full」（残差在留出标定下复现）。实测 r_held ≈ −0.01：跨关系标定罚 p_x 把 g 全额吃掉。残差没有作为残差复现 ⇒ 按预注册 FAIL，不升 E2。但 p_x 本身（0.10–0.16 ×3 模型 ×2 关系）是「跨关系偏移」的直接量化——三原则之「失败必入图谱」的正面产出。
- **产物**：`tests/glm5/result/rdc_query_construction_20260913/phase3182/g5b7_ftr23_heldout_loro/` {execution.json design ea0a1e7a, registry v1.9 59a91512, atlas_v1_10.html 0f24b739, result res 5f08426f/seal b6a57f07, final_result, smoke res 42e77329/6a02861e}。脚本 `tests/glm5/phase3182_g5b7_ftr23_heldout_loro.py`（arm 从 3178 verbatim 导出；渲染复用 phase3179 模块）。ledger append（closeout 本轮）：measurements 333→334（#334=3182），pre 72567879 / post 73778c8e，前缀逐条等价，ledger_sha256_8 按 Phase 8 先例 stale 不重算。E1 现状：FTR-04（归因闭合 3181 噪声级）、FTR-23（held_out F17，p_x 路线开放）。

### 接续：预注册 3183

**3183 = G5-B8 跨关系偏移向量低秩检验（T2 前置臂，零 GPU，封存 3157 collect 复用）**——综合报告 T2（关系残差参数化，最高优先）的第一步：对每模型以 3157 dC 面板分别取 v_isa（hasa 32 细胞 unit_mean）与 v_hasa（isa 32 细胞 unit_mean），检验 Δv = v_isa − v_hasa 的结构：(a) ‖Δv‖ 量级（对照 g 与 p_x）；(b) Δv 在「共享子空间 vs 正交补」的能量分解（对 v_full 正交化）；(c) 实体级投影稳定性（16 实体两两 Δv_e cos 谱）。判读门（freeze 前细化）：若 Δv ≈ 单一低秩方向（主成分占比高、实体级 Δv_e cos 谱集中于正值带）⇒ ΔM_rel = T_C + R_rel 低秩假设获得首条直接证据，登记 FTR-24 候选（E1_repeatable 判据：3 模型同向 + 量级一致）；若 Δv 弥散 ⇒ R_rel 非低秩，T2 路线修正。协议细节 freeze 前从 3157/3178 脚本 verbatim 导出。挂账不变：GAP-4 残留 1.8×、5 appendix、N 线 P3–P7 补 Ledger、跨线账本补丁施加确认、水果类、K4、客户端图谱透镜端到端验证。
"""

with io.open(P, "r", encoding="utf-8") as f:
    txt = f.read()
assert "Phase 3182: G5-B7" not in txt, "already appended"
txt = txt.rstrip("\n") + BLOCK
with io.open(P, "w", encoding="utf-8", newline="") as f:
    f.write(txt)
post = hashlib.sha256(open(P, "rb").read()).hexdigest()[:8]
print("MEMO appended pre=%s post=%s" % (pre8, post))
