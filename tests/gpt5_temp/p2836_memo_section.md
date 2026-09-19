

## Phase 2836: 双通道逐层增益分解 + 归纳复制与属性写头关系 [2026-09-17 14:55]

### 1. 原理与设计

2836 候选 a+b。臂 (a)：分离通道逐层增益分解——残差恒等式 `inc(l) = hs(l+1)−hs(l) = attn(l)+mlp(l)`，分离增量 `inc_spec(l) = inc_same(l) − 0.5(inc_func(l)+inc_null(l))`，逐层分解为 attn/mlp 载体，方向分辨口径 `cls_spec_inc(l) = |inc_spec(l)@cdir| / ||inc_spec(l)||`。臂 (b)：归纳复制探针——greedy decode "The apple is a fruit. The"，在**发出复制 token 的那次前向**（输入以 "The" 结尾）捕获 L35 o_proj 输入，按 32 头分解到 `c_apple` 方向，检验 2824/2833 属性写头簇 {h20,h22,h23,h26} 是否参与归纳检索。

预注册：A1 闭合 <0.02；A2 cls_spec_inc 峰值层 ≥20；A3 载体比 f_attn（描述性）；B1 前 3 步内生成 'apple'；B2 写头簇 ≥3/4 进 top16。判决 = A1∧A2 / B1∧B2。

### 2. 判决：A1=true / A2=true → dual_channel_gain_resolved；B1=true / B2=true → induction_via_write_heads（Gen3 复现一致）

| 判据 | 读数 | 结论 |
|---|---|---|
| A1 闭合 | rel 0.0129（20 词均值） | 增益分解闭合，attn/mlp 载体分解可信 |
| A2 层分辨 | cls_spec_inc 峰 L30（0.0943） | 语义调制增益集中中晚层（与 2833 S4 峰 L33 呼应） |
| A3 载体比 | f_attn = 0.1424 | 分离增量范数 MLP 占 86%、attn 仅 14%（描述性） |
| B1 归纳复制 | step 0 生成 " apple"（其后 " is a fruit that is commonly used"） | 归纳式实体复制确认 |
| B2 写头参与 | h20 排名 1（−0.0301）、h22=3（0.0181）、h26=4（0.0151）、h23=5（0.0144），**4/4 全部进 top16** | 归纳复制经由属性写头簇 |

### 3. 机制发现与 Gen1 诊断（重要环境新知识）

1. **HF hidden_states 尾元素陷阱（新钉死）**：`output_hidden_states` 的第 37 项（hidden_states[36]）是**过 final RMS norm 之后的状态**，不是 L35 层原始输出。Gen1 用 `hs[-1]` 锚定 Δ_spec 导致 A1 闭合失败（rel 5.56）；残差恒等式在 L0–L34 成立（探针 dev 相对 <1%），L35 处 hs[35]+attn[35]+mlp[35] ≠ hs[36]（dev 667）。**凡逐层分解/增量分析必须用 hs[35]+attn+mlp 重构原始末态。**
2. **L6 massive-activation 通道**：裸 spec_gain（范数口径）峰值在 L6（524.9）——null 随机 token 在 L6 产生巨大增量（massive activation），方向非特异（cdir 投影份额同样被摊薄），与语义调制的中晚层峰（L30）完全解耦。**双通道实为三层结构：L6 通用 massive-activation 通道（范数大、方向盲）+ 中晚层语义调制通道（方向特异、cls 0.094）+ 末层读出通道。**corr(spec_gain, gen_gain)=0.62。
3. **归纳复制 = 属性写头硬件复用（B2 4/4）**：2834 true 臂归纳式复读的检索-写入步由同一批属性写头簇执行（h20 也在 2833 头分解中为拮抗头，此处 −0.0301 仍为最强）。**2824 属性写头簇不是"属性专用头"，而是实体-档案读写通用头**——归纳复制（n-gram 级）与属性调制（语义级）共享同一写入硬件，差别在读端检索信号。此发现把"归纳头"与"属性写头"两个文献概念在 Qwen3-4B 上统一为一个硬件簇。
4. c_fruit top5 = h20(−0.0351)/h22/h23/h26——写头簇对 fruit 方向同样最强；c_food 无簇集中（top5 分散，h0/h5）——**写头簇编码的是实体-类别轴，不是 food 语义本身**，与 2828/2830/2834 链传递三否证一致（food 不在运行时通道）。

### 4. 硬伤

1. f_attn=0.14 是范数口径，被 L6 massive activation 污染；方向分辨的 attn/mlp 载体比未做（需逐层 cls 投影 × 载体分解二维）。
2. 归纳探针单句单实体；B2 是相关性（写头参与）非因果（消融写头看复制是否消失）——2837 候选：写头消融下归纳复制与属性调制的双重因果检验。
3. null token 的 massive activation 未系统刻画（L6 通道只出现在随机 token 条件，需要专门扫描）。

### 5. 产物登记（immutable + SHA256，Gen3 为准）

- 脚本 `tests/glm5/phase2836_dual_gain_induction.py` sha256 = `7e83b705def0abea…`
- `…/phase2836/dual_gain_induction/execution.json` sha256 = `d6f4cb699efe5150…`
- `result.json` sha256 = `33f789d804740842…`；`gain_layers.npz` sha256 = `c8de741b5d0dace7…`
- Gen 记录：Gen1（hs[-1] 陷阱，A1 假阳性失败+dsf 锚定错误）、Gen2（修正 dsf/口径/B2 前缀，成功）、Gen3（PREREG 文本对齐重跑，判决复现一致）。Gen1/Gen2 产物已按纪律删除。

### 6. 纪律制度化：numpy 混合高级索引禁令（2835 承接）

自本 Phase 起预审清单新增两条硬规则：①凡 `(slice, list, …)` 混合高级索引一律改写为逐行标量索引或先花式索引后切片，禁止依赖布局推断；②凡逐层分解/增量分析禁止使用 `hidden_states[-1]`（post-final-norm），一律 `hs[L]+attn[L]+mlp[L]` 重构原始末态。

### 7. 接续（2837 候选）

1. 属性写头消融双重因果检验：消融 {h20,h22,h23,h26}（L35），同时测归纳复制存活率与属性调制幅度——若两者同损，硬件统一成立；若只损属性，归纳走并行通路。
2. attn/mlp × 方向分辨二维载体谱（f_attn 的干净版本）。
3. L6 massive-activation 通道系统扫描（哪些 token 触发、是否 null-token 特有）。
