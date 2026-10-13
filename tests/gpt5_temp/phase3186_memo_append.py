# -*- coding: utf-8 -*-
"""Phase 3186 closeout: AGI_GPT5_MEMO.md tail append（Phase 3186 节 + 预注册 3187）"""
import hashlib

P = r"D:/AI2050/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md"

def sha8(b): return hashlib.sha256(b.encode('utf-8')).hexdigest()[:8]

t = open(P, encoding='utf-8').read()
pre8 = sha8(t)
assert pre8 == "ec73b057", "pre-append sha changed: %s" % pre8

ADD = """
---

## Phase 3186: G5-B11 关系偏移二阶结构检验——满秩扩散 DIFFUSE（均值级收缩定判，零 GPU，registry v1.12→v1.13 + atlas_v1_14.html） 13:05

**verdict: rrel_meanlevel_contraction——F19 归因闭合：实体级偏离 δ_e = u_e − ṽ_h（和零 15 维子空间，恒等式 <1e-6 验证）的 SVD 谱满秩扩散——top2 能量占比 0.2458/0.3197/0.3004（4b/14b/glm4，isa/hasa 均值；top1 仅 0.126–0.232），80% 能量需 9–10 个方向、90% 需 12–13 个，×3 全部 <0.5 门 ⇒ 平面参数化 T2′ 被拒，均值级收缩定判：R_rel 只保留关系级 Δv（FTR-24 方向陈述不变），实体级变异为个体扩散噪声，放弃实体级预测主张。离群 LOO（3184 协议）：flip 实体（剔除后 r_g≥0.5 且 R²_px≥0.5）= none ×3 模型 ×2 关系——4b isa 剔 e15 把 r_g 0.47→0.62 但 R² 仍 ≤0.33；14b hasa 剔 e12 后 r_g LOO 上界 −0.04 仍负（3184 离群实体不是本处失败原因）——F19 的失败是系统性而非离群承载。**

**verdict（重复二）：装置全绿——16 源文件 sha 断言；3157 双锁 + 3177/3178/3182/3183/3185 封存结果五重双锁；复制门全 <1e-9（含新增 3185 全面板 r_A/R²_B 重算 drift 0.00e+00）；装置 tc drift <1e-9；SMOKE/FULL 数值逐位一致；execution.json design 2c4906b7；FULL res 1bd47cee / seal 731169a0；registry v1.13 730729f5（FTR-24 counter_evidence 追加 + 收缩 statement 追加，双 append-only，anchors 3 不变；23 条 legacy 字节保全；failures 19 / upgrade_log 6 不动；按预注册钉定 DIFFUSE 不新增失败 id——F19 已覆盖实体级线性拒绝）；atlas_v1_14.html 56c45e50（737 字段 0/0/0/0）。**

**verdict（重复三）：实质读数——(a) 聚合低秩 ≠ 实体级低秩：关系偏移方向（Δv，聚合 32 细胞 unit_mean）与实体级偏离谱解耦——3183/3184 的 Δv 低秩是聚合量属性，3185/3186 证明它不延伸到逐实体幅度；(b) T2 主臂（G5-B10/B11）收官结论：ΔM_rel = T_C + αΔv 在均值级成立性保留（α 拟合待做），实体级预测主张正式撤销；(c) 至此 isa/hasa 面板的关系偏移链条 3177→3178→3182→3183→3184→3185→3186 七层完整闭环，每个可证伪假设都有定判。独立复核 tests/gpt5_temp/phase3186_verify_independent.py。**

### 产物与挂账

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3186/g5b11_second_order/` {execution.json design 2c4906b7, registry v1.13 730729f5, atlas_v1_14.html 56c45e50, result res 1bd47cee/seal 731169a0, final_result, smoke res 966885e0/36316d43}。脚本 `tests/glm5/phase3186_g5b11_second_order.py`。ledger append：measurements 337→338（#338=3186），pre 4fff0f4a / post 17c4f3e6，前缀逐条等价。失败账本 F1–F19 不变；E1：FTR-04/23/24。

### 接续：预注册 3187

**3187 = G5-B12 R_rel 均值级 α 拟合定标（T2 收官臂，零 GPU，封存 3157 collect 复用）**——均值级收缩后的最后一块：以 3183 verbatim Δv 与 3178 关系缺口 g（3177 同 v_shared 配方）做单参数最小二乘 α̂ = ⟨Δv 残差投影, g 分量⟩/‖·‖²（per model per relation，2 fold ×3 模型），登记 α̂ 数值、解释率（α̂Δv 对 g 的方差占比）、跨 fold/跨模型稳定性；判读门（freeze 前细化）：α̂ 同模型两 fold 符号一致且跨模型量级同带（max/min ≤3×）⇒ α 稳定可定标，T2 均值级参数化正式定稿入 FTR-24 statement（α̂ ×3 数值入锚）；否则 α 不稳定 ⇒ T2 降级为方向级陈述（只保留 Δv 方向、撤销幅度主张），FTR-24 statement 相应收缩。协议细节 freeze 前从 3177/3183 脚本 verbatim 导出。挂账不变：GAP-4 残留 1.8×、5 appendix、N 线 P3–P7 补 Ledger、跨线账本补丁施加确认、水果类、K4、客户端图谱透镜端到端验证（server 版本链登记 v1.9–v1.13 + 重启）。
"""
t2 = t.rstrip() + ADD
open(P, 'w', encoding='utf-8', newline='').write(t2)
print("MEMO appended. pre", pre8, "post", sha8(t2))
