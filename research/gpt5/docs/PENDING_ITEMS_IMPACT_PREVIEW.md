# 待决项影响预览（Phase 3271，report-only）

生成：3271 G5-H34 `pending_impact_preview_registered`（freeze `12d3194a` / res `6cf92c2b`）。
全部数值磁盘实测（含 op4 零写入 DRY-RUN 链计算）；5 项裁决状态均未改变。

## ① 水果类/K4 MEMO 挂账删除
- 对象：`research/deepseek/docs/AGI_DEEPSEEK_MEMO.md`（基线 sha8 `d3f62d35`）。
- 行位实测：「水果类」26 处、「K4」26 处；其中挂账清单行（7153/7257/7360/7409/2279/2530/2834/3124 等）与科学发现段（1106/1160/1318/3097 等）须区别对待。
- 机制：**append-only 勘误注记**（MEMO 尾部追加 dated resolution note + 新 dated 小节逐行注记），不重写历史字节；若所有者要求物理删除则属破坏性操作须二次确认。
- 批准后触及：约 52 行的注记 + 1 段新注记；账本 +1 收据。

## ② lcorr op4 施加（A-gate 治理登记）
- payload（lcorr v2 `ef72c733`，`READY_PENDING_CONFIRMATION`）：phase="A-gate"、name=rdc_a_gate_closure_q01_q02_q08_q09_q12、verdict=k1_fired_at_readout__kspecific__k2_k3_not_measurable__A_gate_closed、evidence_level=bit_anchored、note=Q08=甲 seal 后登记；属治理记录非科学测量。
- **DRY-RUN（零写入）链计算：批准并施加后 post = `329b1398`（n 427→428）**；机制同 3258（append + recompute + revert-proof）。
- 联动：批准本项即结清 **APPX-5 的最后开放路径**（见 ⑤）。

## ③ APPX-1 量化口径（14b R×K_entity 定位）
- 定义（gap v1.13）：eff_ge05=2 仅 14b；定位共享方向并解释其来源；status=appendix_open。
- 语境事实：Qwen3-14B-bnb-nf4 在盘（nf4_on_disk=True）；RTX 5080 16GB 无法 bf16 装载 14B。
- 裁决空间：批准 nf4 基线执行 / 拒绝量化（转未来硬件）/ 维持 open。批准后 = 新 GPU 执行相（3267 协议模式），封存账目不动。

## ④ APPX-2/3 执行相
- 协议：`appx23_protocol_v2.json`（`ce73984d`，PREREGISTERED_DRAFT_V2）。
- 前置复查（本轮实测）：5 候选模型 tokenizer+config+weights 全在盘。
- GPU 成本声明（3262 readiness note，非排期）：APPX-2 约 15–60 min/模型（S1–S3）；APPX-3 基于现有 3 模型 collects，GPU-light。

## ⑤ APPX-4/5 跨线治理
- **APPX-4 = STALE_STATUS（本轮新定谳）**：N-P3..P7 自 3258 起实测在账（ledger 5 条 N-P 条目），gap v1.13 的 status=open_crossline 系滞后注记；处置 = gap v1.14 appendix 注记 RESOLVED（registry 臂），v1.13 字节不动。
- **APPX-5 = 已收敛至 op4**：ledger_corrections_v1.json（`beb775bd`）四操作 3260 全处置（op1 作废/op2 不迁移/op3 被实践超越/op4 READY）——唯一活口即 ②。

## 裁决矩阵（owner 视角）
| 项 | 动作 | 触及 | 风险 |
|---|---|---|---|
| ① 水果类/K4 | 批准→append-only 注记 | deepseek MEMO ~52 行注记 | 低（不重写历史） |
| ② lcorr op4 | 批准→ledger append | atlas_ledger +1（post `329b1398`） | 低（revert-proof） |
| ③ APPX-1 | 三选一 | 新 GPU 相或无 | 中（量化保真度） |
| ④ APPX-2/3 | 点名启动 | GPU 相 ×N + gap/registry 落册 | 中（跨模型口径） |
| ⑤ APPX-4/5 | 随②自动结清+注记 | gap v1.14（registry 臂） | 低 |


<!-- maintenance-addendum-3272 -->
## 维护注记（Phase 3272，2026-10-21）

第 ② 节的 DRY-RUN post `329b1398` 系基于 n=427 基线计算，因 3271 closeout append（#423，n=428）**已过期**（SUPERSEDED_BY_MAINTENANCE）。对现役头（#423，post `3384cfbd`）刷新：**op4 批准并施加后 post = `d165ee30`**（n 428→429）。本注记为 append-only 追加，上文基线字节未改动。


<!-- addendum-3274 -->
## 施加回执（Phase 3274，2026-10-21）

② lcorr op4 已施加：owner 裁决批准（2026-10-21），实际 post = `498a80f6`（n 430→431，现场重算；本文件记载的 329b1398/d165ee30 均已过期，仅作历史参考）。⑤ APPX-4/5 已结清：gap v1.14 `c6d97133` + registry v1.37 `bfb7d7e5` 注记 RESOLVED。剩余待决 3 项：① 水果类/K4、③ APPX-1 三选一、④ APPX-2/3 点名。

<!-- addendum-3275 -->
① 水果类/K4 已结清：owner 裁决批准（2026-10-21），Phase 3275 以 append-only 勘误注记落地（deepseek MEMO 6d0ed4fb→fb813f57，35 行 52 处逐行注记，前缀字节不动；账本 #427 post 923806e0）。剩余待决 2 项：③ APPX-1 三选一、④ APPX-2/3 点名。


<!-- addendum-3276 -->
④ APPX-2/3 执行相已实施（Phase 3276，2026-10-21，owner 点名）：APPX-2 全 5 模型 S 族轴×K_readout 判定 separable（守恒）；APPX-3 K3 死线 3/3 触发（弃单坐标机制目标）、K2 3/3 不触发。gap/registry 落册留待 3277（协议 G5/H5）。剩余待决 1 项：③ APPX-1 量化三选一。

<!-- addendum-3277 -->
④ APPX-2/3 已落册：Phase 3276 执行相 （res 8d5bba36）+ Phase 3277 registry 相（G5/H5，gap v1.15 58094dab APPX-8 + registry v1.38 f8772d65；ledger #429）。剩余待决 1 项：③ APPX-1 三选一。

<!-- addendum-3278 -->
③ APPX-1 已执行（Phase 3278，2026-10-21，owner 批准 nf4 基线）：定位+解释完成（两共享维 0.894/0.715，末段层新生，话语/真值语义族在案）；nf4 定量守恒 top1 差 4.15° 超预注册容差 3.0°（量化扰动 R_logic 侧，eff=2 与 cos diff≤0.035 守恒）→ verdict appx1_blocked（res 588d36c9，ledger #430）。处置（按定位/解释关闭 vs 留待全精度硬件复测）留 owner。剩余待决：APPX-1 关闭口径一项。
<!-- addendum-3279 -->
③ APPX-1 已落册：Phase 3279（G5-H42）将 3278 执行结果 append-only 入账（gap v1.16 12961b7b APPX-9 + registry v1.39 ab14e586；blocked verdict 逐字入账，F 账本原则）。APPX-1 挂账状态不在落册中关闭——处置（关闭 vs 留待全精度硬件复测）仍留 owner。剩余待决 1 项。
