# -*- coding: utf-8 -*-
"""Phase 3187 closeout: AGI_GPT5_MEMO.md tail append（Phase 3187 节 + 预注册 3188）"""
import hashlib

P = r"D:/AI2050/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md"

def sha8(b): return hashlib.sha256(b.encode('utf-8')).hexdigest()[:8]

t = open(P, encoding='utf-8').read()
pre8 = sha8(t)
assert pre8 == "ef1e4839", "pre-append sha changed: %s" % pre8

ADD = """
---

## Phase 3187: G5-B12 R_rel 均值级 α 拟合定标——α 不稳定 UNSTABLE（T2 降级方向级陈述，零 GPU，registry v1.13→v1.14 + atlas_v1_15.html） 13:15

**verdict: rrel_alpha_unstable——单参数最小二乘 α̂（3185 prediction-A 形式 g_e = ḡ − κ·α·γ_e，per model per fold，2×3=6 拟合）：4b +1.1348/+1.1997（解释率 0.2235/0.4113）、glm4 +2.1793/+1.3102（0.4195/0.3420）、14b +0.7602/−0.7864——14b 两 fold 符号翻转，α_m = −0.0131，跨模型带比 133.28（门 ≤3）⇒ GATE UNSTABLE：T2 降级为方向级陈述——FTR-24 只保留 Δv 方向主张，幅度参数 α 撤销（实体级主张已于 3186 撤销）。LS 代数恒等式解释率 = 3185 封存 r_A² ×6 <1e-9（经独立路径复制 3185）。p_x 侧 α̂_px report-only 同样呈 14b 不一致（+0.59/−1.03）。**

**verdict（重复二）：装置全绿——15 源文件 sha 断言；3157 双锁 + 3177/3178/3182/3183/3185 五重双锁；复制门全 <1e-9（3177 folds、3178 r_full、3182 p_x、3183 dv）；装置 tc drift <1e-9；SMOKE/FULL per_model 逐位一致；execution.json design 760b886d；FULL res 10140b94 / seal cb1bce5b；registry v1.14 95fe55f5（FTR-24 statement append only：方向级降级，anchors 3 与 counter_evidence 不变；23 条 legacy 字节保全；failures 19 / upgrade_log 6 不动）；atlas_v1_15.html a29022d5（736 字段 0/0/0/0）。FULL-1 crash 诚实登记：finish_registry FAIL 分支 format 误引 summary['band_ratio']（应为 summary['gates']['band_ratio']），在任何 registry 写盘前崩溃；修复后重跑 FULL，arm res/seal 跨次逐位一致（10140b94/cb1bce5b）——无观测泄漏。**

**verdict（重复三）：实质读数——(a) 4b 与 glm4 的 α̂ ~1.2–1.7 一致且量级合理（Δv 幅度与关系缺口同尺度，均值级线性形式对 2/3 模型是有意义的）；但 14b 的反对称 fold 结构（与其 3185 hasa r_A=−0.2185 一致）破坏跨模型定标，钉定门要求 ×3 ⇒ 方向级降级是诚实处置——α 或可按模型族分别定标，但跨模型标量不可主张；(b) T2 主臂（G5-B10/B11/B12）三连收官：实体级线性拒绝（F19）→ 谱满秩扩散（均值级收缩）→ α 跨模型不稳定（方向级降级）——ΔM_rel = T_C + R_rel 的最终形态：R_rel = 方向级 Δv（关系级、聚合量、无幅度/实体级主张）；(c) isa/hasa 面板关系偏移链条 3177–3187 八层完整闭环。独立复核 tests/gpt5_temp/phase3187_verify_independent.py。**

### 产物与挂账

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3187/g5b12_alpha_calibration/` {execution.json design 760b886d, registry v1.14 95fe55f5, atlas_v1_15.html a29022d5, result res 10140b94/seal cb1bce5b, final_result, smoke res f7df8d39/97a69d1d}。脚本 `tests/glm5/phase3187_g5b12_alpha_calibration.py`。ledger append：measurements 338→339（#339=3187），pre 17c4f3e6 / post da29d7d6，前缀逐条等价。失败账本 F1–F19 不变；E1：FTR-04/23/24。

### 接续：预注册 3188

**3188 = G5-C1 GAP-4 残留 1.8× 的 Δv 方向归因（T2 后新开 C 线，零 GPU，封存 3169/3175 复用）**——T2 收官后回到最大挂账 GAP-4：谱外崩塌 2.54× 已由端口校准解释 29%（3172），残留 1.8× 机制未明。面板内检验：以 3183 verbatim Δv 方向对 3169 谱外面板封存 dC/崩塌量做投影归因——(a) 谱外细胞 dCn 与 Δv 的投影分布 vs 面板内（isa/hasa）基线；(b) 谱外崩塌读数中 Δv 方向可解释份额（方向级，无幅度主张）；(c) 若投影分布显著偏移（谱外 vs 面板内 KS 或中位差，门 freeze 前细化）⇒ R_rel 方向与谱外崩塌耦合，GAP-4 残留获得首条方向级机制线索；否则 GAP-4 残留与关系偏移解耦，登记 counter_evidence。协议细节 freeze 前从 3169/3175/3183 脚本 verbatim 导出；若 3169 封存缺逐细胞 dC（仅聚合读数），降级为聚合投影检验并诚实登记。挂账不变：5 appendix、N 线 P3–P7 补 Ledger、跨线账本补丁施加确认、水果类、K4、客户端图谱透镜端到端验证（server 版本链登记 v1.9–v1.14 + 重启）。
"""
t2 = t.rstrip() + ADD
open(P, 'w', encoding='utf-8', newline='').write(t2)
print("MEMO appended. pre", pre8, "post", sha8(t2))
