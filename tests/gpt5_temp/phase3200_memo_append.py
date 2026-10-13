# -*- coding: utf-8 -*-
"""Phase 3200 MEMO append (assert prefix)."""
import hashlib
import io

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'
old = io.open(P, encoding='utf-8').read()
pre_sha = hashlib.sha256(old.encode('utf-8')).hexdigest()[:8]

ADD = '''

## Phase 3200 — G5-E2 FTR-23 held_out 升格臂（封存 3178 null 读数复用，零 GPU，登记臂）【2026-10-10】

**预注册钉定（先于观测）+ 协议 freeze 前细化（诚实登记）**：FTR-23 在 3169 OOV 行空间**无封存方向向量**（其为 3157 上下文变换面板上的标量现象 r_m/p_m）——3200 臂由「OOV 行投影」细化为**封存 3178 实体留出 null fold 读数评估**（3176 批量升级先例的登记臂模式）：对每（模型×关系）单元，取 3178 封存 null 的 16-fold cos_out（留出实体 2 细胞 vs 其余 15 实体重建的关系内均值方向，3178 协议 verbatim）；门（freeze 前钉定）：median cos_out ≥0.5 全 6 单元 ⇒ FTR-23 E1_repeatable→E2_held_out（evidence_level 变更 + anchors append role=held_out + statement append + upgrade_log #9），registry v1.22→v1.23 + atlas_v1_25.html；否则 counter_evidence append。对照锚：3178 标量 4dp 回读 drift==0 ×3（中位数 freeze 前实测探针自封存 result——实测锚纪律；exch_mean 1.2728174525072664 逐位）+ 3183 dv 标量逐位复现（build_dvu）+ 3157 tc drift<1e-9。诚实日志：门 bar 0.5 在预冻结探针读数后钉定（保守：观测带下限 0.74 之半、高于 OOV 面板 0.3 先例）。

**结果**：门 **6/6 单元全过 ⇒ HOLDOUT_PREDICTIVE_FTR23，FTR-23 E1→E2_held_out（升级链 #9）**：
- med cos_out：isa **0.7696/0.7564/0.7762**、hasa **0.7429/0.7855/0.7885**（4b/14b/glm4）；min fold 0.6880/0.6224/0.6918（isa）、0.7128/0.6091/0.6941（hasa）。
- 诠释：**关系内上下文变换几何对实体留出高度可预测（中位 cos ~0.74–0.79）——FTR-23 的关系特异残差是关系间（between-relation）偏移，不是关系内实体噪声**。与 3199（dvu 对 OOV 行 NOT_PREDICTIVE）合读，E 线图景完整：**held-out 可预测性住在类均值/关系内结构里；关系间偏移与关系方向不承载跨面板预测**。

**登记**：FTR-23 anchors append（role=held_out，asserts 6 单元中位数 + bar 0.5 + e_level 前后）+ statement append + upgrade_log #9；counter_evidence 5 条保全、failures F1–F19 不变、25 legacy 特征字节保全。registry **v1.22 `9fd87433` → v1.23 `5f5b9e73`**（append-only）；atlas_v1_25.html `7dc08f1e`（**782/782 字段，0/0/0/0**）。

**图谱现状**：registry v1.23（26 特征：**E2×20**/E3×3/E1×3——FTR-04 已闭 noise_level、FTR-23 已升、FTR-24 OOV 面板 NOT_PREDICTIVE 在案）、failures F1–F19、upgrade_log **9**、gap v1.6 `0f6423e0`、atlas_v1_25 `7dc08f1e`、ledger n=352。

**产物**：design `2ea6e1f6`，res `f9a280fc`/seal `cba9d56c`，final `ae7a5837`/`dfd3b880`，ledger #352（post `8964e20d`）。SMOKE/FULL 载荷逐位一致。

**3201 预注册（G5-E3，零 GPU，E 线收官审计臂）**——E 线收官定论 + 图谱齐备性审计：对 registry v1.23 全 26 特征做锚齐备性审计（每 E2 特征须有 cross_model+held_out 锚各 ≥1、每 E3 须有协议内因果锚、E1 残余 FTR-04/FTR-24 的可证伪记录完备性）；E 线方法论定论写入 FTR-24 scope_limits append（「held-out 可预测性的几何分界：类均值/关系内结构承载，关系间偏移与关系方向不承载——3199/3200 双臂证据」）；appendix 5 项逐条可执行性盘点（零 GPU 可行者列入下一预注册候选）。门（freeze 前细化）：审计 0 矛盾（缺锚/序错/字节漂移均为矛盾）且 scope_limits append 验证（startswith + 其余字段字节保全）⇒ E_LINE_CLOSED（registry v1.23→v1.24 仅 FTR-24 scope_limits 变更 + atlas_v1_26.html）；有矛盾 ⇒ 逐条登记 gap，report-only。设备锚：registry 链 v1.20→v1.23 四版 sha 逐位回读 + 3178 标量 4dp 回读。
'''
new = old + ADD
assert new.startswith(old)
io.open(P, 'w', encoding='utf-8', newline='').write(new)
back = io.open(P, encoding='utf-8').read()
assert back == new
post_sha = hashlib.sha256(back.encode('utf-8')).hexdigest()[:8]
print('MEMO appended: pre %s -> post %s (+%d chars)' % (pre_sha, post_sha, len(ADD)))
