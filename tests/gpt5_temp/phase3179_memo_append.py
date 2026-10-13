# -*- coding: utf-8 -*-
"""Phase 3179 closeout: MEMO 尾部 append（Phase 3179 记录 + 预注册 3180）"""
import io

P = r"D:/AI2050/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md"
raw = open(P, "rb").read()
assert raw[-200:].find(b"3179") == -1, "tail already contains 3179?"
pre8_ok = True

txt = """

---

## Phase 3179: G5-B4 FTR-23 立项入册——T_C 关系特异残差升格独立特征（零 GPU，registry v1.5→v1.6 + atlas_v1_7.html） 04:17

**verdict: ftr23_registered_E1_repeatable——「失败必入图谱」正面化完成：FTR-23（family=context，T_C 共享分量携带关系特异残差）按三原则入表，registry v1.5→v1.6（22 特征→23，features append-only，22 条 legacy 特征字节保全，failures/upgrade_log/lineage 未动），atlas_v1_7.html 698 字段 0/0/0/0。注册 bar 三条全满足：3177+3178 两 phase 独立测量、3 模型同向（r(m) 0.0918/0.0951/0.1289 ≥ 0.05 bar ×3）、预注册门历史完整（F16 gate fail → 3178 归因定量闭合）。held_out 升级路线对关系特异特征而言「held-out 关系」本身即测量构造，留后续设计，不阻塞入册。**

**verdict（重复二）：装置全绿——12 源文件 sha 断言通过；封存内容双锁加载（p3178 res 4f521d9b/seal 1f166645 + p3177 res d451030f/seal 7b746eb2 + p3157 collect 95a25965/9552086d/23dd74eb + p3157 summary 71959b6a，零转录）；对封存 3178 重算 tc_full/exch/p_mean ×3 模型 drift 全部 0.00e+00；装置门 1e-09 + 重算门 1e-06 全绿；range asserts 对 MEMO 草案读数（p/r_fold/r_m/tc_held/tc_full）全绿；SMOKE 与 FULL registry v1.6 字节恒等（bebdf38d）=> 数值运行模式无关；execution.json design 2caa1306；FULL res 18d6697f / seal e1b5e3a2（SMOKE 同装置）。**

**verdict（重复三）：独立复核（独立进程，verify FAIL 只修 verify 不动封存产物）——文件 sha 断言（reg16 bebdf38d / html b2bcec66 / seal e1b5e3a2）PASS；22 legacy 特征 JSON 逐条字节等价 PASS；failures/upgrade_log/lineage 未动 PASS；字段级 diff vs 封存 atlas_v1_6.html（phase3178）：dropped=0、changed+new 仅 FTR-23/meta/node 视图字段（new=29、changed=30，其中 FTR-23 26 个）PASS；正向叶子值覆盖检查中 33 个 MISS 全部追溯为 698 字段视图本就不渲染的 registry 内部 provenance 叶子（封存 v1.6 HTML 同样不含，非回归）。复核脚本自身两轮口径修正（data-k 点路径语义、HTML 转义/浮点渲染精度/token 分解），未触碰任何封存产物。**

### FTR-23 立表内容（registry v1.6 现场渲染）

- **statement**：T_C 共享分量携带关系特异残差（3178 sampling-null 归因定判 + 3177 关系留出方向性发现）：实体留出 null 标定下纯采样惩罚 p 0.019-0.026，关系残差 r 0.072-0.131 ×3 模型（标定 bar 0.05，r(m)=0.0918/0.0951/0.1289 全过 bar 判关系无关失败）；关系内两两 cos tc_held 0.581-0.623 > 全量 tc_full 0.561-0.564（6/6 fold×模型单元）——T_C 几何按关系分解，关系内自洽、跨关系系统性偏移；关系残差由 3177 F16 门失败归因定量闭合（F16 = gate fail 记录，本特征 = 残差本体的正面登记）。
- **evidence_level**：E1_repeatable（cross_model 即时成立）；**family**：context；**锚**：3（p3178 primary_attribution / p3177 directional_fold_evidence / p3157 device_base_measurement，各带 asserts 数值字典 + 双 sha 锁）。
- **计数**：features 23 / failures 16 / upgrades 5 / gaps 4 / appendix 5。

### 产物与挂账

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3179/g5b4_ftr23_registration/` {execution.json design 2caa1306, registry v1.6 bebdf38d, atlas_v1_7.html b2bcec66, result res 18d6697f/seal e1b5e3a2, smoke 同装置}。脚本 `tests/glm5/phase3179_g5b4_ftr23_registration.py`；独立复核 `tests/gpt5_temp/phase3179_verify_independent_v3.py`（结论 ALL PASS）；ledger append（closeout 本轮）：measurements 330→331（#331 = phase 3179），pre-append file sha8 8b6cfe43 / post 54b10689，前缀 330 条与备份逐条等价、其余键未动（append-only 断言通过），ledger_sha256_8 按 migration_history 先例不重算（recipe 未知、Phase 8 起 stale）。

### 接续：预注册 3180

**3180 = G5-B5 FTR-15 held_out 升级臂（零 GPU，封存 collect npz 复用）**——E1 剩余三特征（FTR-04/14/15）中 FTR-14 held_out 路线已于 3177/3178 定判关闭，FTR-23 刚入册，顺位取 FTR-15：按其 metric 类别选择留出构造（held-out 实体/语境 fold），协议细节 freeze 前从对应 collect 脚本 verbatim 导出，数据源钉定封存 npz（无新 forward）；门 = 留出读数与全量差 <0.1 ×3 模型 + 装置锚 drift 0（对既有 summary 数值逐位复现）；过门 => held_out 锚入表、FTR-15 升 E2_predictive；fail => F17 入失败账本（归因分解预注册为下一 phase，同 3177→3178 先例）。FTR-04 glm4 弱特异性归因（全维上界 vs S_class 曲线）顺位 3181。挂账不变：GAP-4 残留 ~1.8× 结构性成分、5 appendix、N 线 P3-P7 补 Ledger、跨线账本补丁施加确认、水果类、K4、UI v1.6 零 script 客户端改造。
"""

with io.open(P, "ab") as f:
    f.write(txt.encode("utf-8"))
print("appended", len(txt), "chars")
# 回读
t = open(P, "rb").read().decode("utf-8")
assert t.count("## Phase 3179:") == 1
assert t.rstrip().endswith("零 script 客户端改造。")
print("read-back OK, tail:", t[-80:])
