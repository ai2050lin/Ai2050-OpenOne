# -*- coding: utf-8 -*-
"""Phase 3185 closeout: AGI_GPT5_MEMO.md tail append（Phase 3185 节 + 预注册 3186）"""
import hashlib

P = r"D:/AI2050/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md"

def sha8(b): return hashlib.sha256(b.encode('utf-8')).hexdigest()[:8]

t = open(P, encoding='utf-8').read()
pre8 = sha8(t)
assert pre8 == "56f55b82", "pre-append sha changed: %s" % pre8
assert t.rstrip().endswith("K4、客户端图谱透镜端到端验证。") or "预注册 3185" in t

ADD = """
---

## Phase 3185: G5-B10 R_rel 低秩模型定量检验——线性叠加拒绝 FAIL（F19 入账，零 GPU，registry v1.11→v1.12 + atlas_v1_13.html） 12:45

**verdict: rrel_linear_superposition_rejected——FTR-24 低秩几何的预测力检验 FAIL：以 Δv（3183 verbatim）为 R_rel 参数化的免拟合解析预测，对 (a) 3177 关系缺口实体级 g_e = cos_train − u_e@v_c（g_pred = ḡ − κ·γ_e，γ_e = (u_e−ṽ_h)@dvu，恒等式 mean(g_e)=ḡ、mean(γ_e)=0 精确成立）与 (b) 3182 LORO cross-null 罚 p_x 实体级分布（p_pred = b_e − γ_e·κ_x(e)）双双未过门：r_g 0.5571/−0.0353/0.6162（4b/14b/glm4，14b hasa fold r=−0.2185）、R²_px 0.2241/−0.0692/0.2858 ×3 全部 <0.5。宽容 OLS 重拟合读法（r²）4b 0.39–0.40、glm4 0.33–0.47、14b 仍为负，同样不过门——结论对拟合自由度稳健。⇒ F19（rrel_linear_superposition_rejected）入失败账本；FTR-24 counter_evidence 定量追加，维持 E1；T2 路线修正：R_rel 低秩方向存在（FTR-24 结构证据不变）但叠加为非线性——Δv 分量与缺口/罚呈符号相关（r_B 最高 0.69）却只解释 <30% 方差，实体级偏离谱不由单一关系偏移方向承载。**

**verdict（重复二）：装置全绿——15 源文件 sha 断言；3157 双锁 + 3177（d451030f/7b746eb2）+ 3178（4f521d9b/1f166645）+ 3182（33d73196/ed51ca22）+ 3183（c55e4e5e/7196fa8a）+ 3184（54e2b9d9/55706292）封存 result 五重双锁；复制门全 0.00e+00（3177 folds cos_train/cos_held、3178 r_full、3182 p_x_mean ×2 关系 ×3 模型、3183 dv 五量 cos_rl/norm_dv/frac_full/pc_share/med_cos）；装置 tc_full drift <1e-9；SMOKE/FULL 数值逐位一致；execution.json design 65ee0dcc；FULL res df958cc9 / seal 7559b27b；registry v1.12 269a37c0（F19 failures 追加 + FTR-24 counter_evidence 追加，23 条 legacy 字节保全，upgrade_log 6 不动）；atlas_v1_13.html a3bd2206（736 字段 0/0/0/0）。SMOKE-1 修正诚实登记：恒等式断言容差 1e-12→1e-6（float32 累积误差），FULL 观测前修正。协议注记：预注册「3177 关系缺口 g_e」实体级操作化为 g_e = cos_train − u_e@v_c（3177 verbatim 分量），freeze 钉定。**

**verdict（重复三）：实质读数——(a) 14b 是唯一 r_g 负号模型，其 hasa fold r=−0.2185 且残差谱 max 实体 e=12（与 3184 离群实体精确重合）、isa fold 残差 max e=15；(b) 4b/glm4 残差 max 同样集中于 e15（×2 模型 ×2 臂）——e15 是面板级公共离群实体，低秩预测的失败部分由少数实体承载而非均匀失配；(c) 机制结论：ΔM_rel = T_C + R_rel 中 R_rel 的实体级变异只有 <30% 落在 Δv 单方向上——关系偏移的「方向」低秩（3183/3184）与「逐实体幅度叠加」非线性（3185）并存，T2 参数化需放弃 αΔv 线性形式（下一 phase 走 per-entity 幅度 or 二阶项）。独立复核 tests/gpt5_temp/phase3185_verify_independent.py。**

### 产物与挂账

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3185/g5b10_rrel_predictive/` {execution.json design 65ee0dcc, registry v1.12 269a37c0, atlas_v1_13.html a3bd2206, result res df958cc9/seal 7559b27b, final_result, smoke res 28701d27/fb10a81b}。脚本 `tests/glm5/phase3185_g5b10_rrel_predictive.py`。ledger append：measurements 336→337（#337=3185），pre ad566a5c / post 4fff0f4a，前缀逐条等价。失败账本 F1–F19；E1：FTR-04/23/24。ledger_sha256_8 NOT recomputed（recipe unknown，stale since Phase 8）。

### 接续：预注册 3186

**3186 = G5-B11 关系偏移二阶结构检验（T2 修正臂，零 GPU，封存 3157 collect 复用）**——F19 的归因检验：Δv 单方向解释 <30% 方差、残差集中于少数实体（e15 公共离群 + 14b e12）。检验实体级偏离 δ_e = u_e^h − ṽ_h 的谱结构：(a) δ_e（×2 关系 ×16 实体）SVD 谱——偏离是否低秩多方向（秩 2–3 即可解释残差）还是满秩扩散；(b) e15/e12 离群归因——剔除后 r_g/R²_px 是否过门（LOO 复用 3184 协议）；(c) 若秩-2 子空间解释率 ≥50% ×3 ⇒ R_rel 修正为平面参数化（T2'：ΔM_rel = T_C + αΔv + βw₂）；若满秩扩散 ⇒ 关系偏移的实体变异是个体噪声，T2 收缩为均值级参数化（只保留关系级 Δv、放弃实体级预测）。判读门（freeze 前细化）：SVD 前 2 主成分能量占比 ≥0.5 ×3 模型 ⇒ 平面参数化立项候选；否则均值级收缩。协议细节 freeze 前从 3183/3185 脚本 verbatim 导出。挂账不变：GAP-4 残留 1.8×、5 appendix、N 线 P3–P7 补 Ledger、跨线账本补丁施加确认、水果类、K4、客户端图谱透镜端到端验证。
"""
t2 = t.rstrip() + ADD
open(P, 'w', encoding='utf-8', newline='').write(t2)
print("MEMO appended. pre", pre8, "post", sha8(t2))
