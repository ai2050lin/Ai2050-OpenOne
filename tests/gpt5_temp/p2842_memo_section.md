

## Phase 2842: QK 边来源回溯——写头接口的源位置分解 [2026-09-17 20:15]

### 1. 原理与设计

承接 2841（5 层因果 top1 头：L22 h28 / L23 h29 / L26 h17 / L28 h4 / L33 h25）。用 eager attention 精确权重 + 目标层 v_proj 值向量，把每头对 cls_spec 对齐的贡献精确分解到两个 QK 源位置：pos0=条件词位（same/func/null 词）、pos1=目标词位（apple 自身）。恒等式 head_out(pos1)=Σ_j A[h,1,j]·OV_h V[j]（分解精确性验证 err=6.2e-15）；C_j = same 项 − 0.5(func+null 项)，signed share_j = C_j·cdir/||d_spec_full||。20 目标词 × 4 前向，15.3s。Gen 历史：3 次崩溃（self_attn 关键字参数 hook、bf16 dtype、reshape 维度）后 Gen4 一次成功；execution.json 每次重跑覆盖，磁盘终态自洽（run4 exec+result 配对）。

### 2. 判决：Q1=true(5/5) / write_interface = **self**（4/5 self_dominant，L33 h25 mixed）

| 头 | share_pos0 | share_pos1 | attn 同条件 pos0/pos1 | attn func pos0 | 判决 |
|---|---|---|---|---|---|
| L22 h28 | 0.00000 | **0.00010** | 0.690 / 0.310 | 0.987 | self_source |
| L23 h29 | 0.00000 | **0.00118** | 0.536 / 0.464 | 0.956 | self_source |
| L26 h17 | 0.00001 | **0.00224** | 0.721 / 0.279 | 0.935 | self_source |
| L28 h4 | 0.00001 | **0.00122** | 0.667 / 0.333 | 0.735 | self_source |
| L33 h25 | −0.00000 | −0.00001 | 0.947 / 0.053 | 0.988 | mixed（总量≈0） |

### 3. 机制发现：写头接口 = QK 自绑定门（self-binding gate），非 OV 内容复制

1. **pos0（条件词位）贡献恒 ≈0，全部对齐贡献来自 pos1（目标词自身）**：语义调制不是"从条件词复制内容"，而是**注意力质量重分配**——same 条件使目标词对该头的自注意力份额大增（L22 h28：same 0.310 vs func 0.013 / null 0.128），条件词存在时目标 token **更多读自己、更少读前文**。func/null 条件下注意力质量涌向 pos0（0.87-0.99）但其 OV·cdir 贡献经 ctrl 相消归零。
2. **与前序结论闭环**：① 2824-2825"实体条件化写入=token 私有 emb 注入"——语义内容早已在目标词自身表征（来自 embedding 与浅层），L22+ 写头只做**门控读出**；② 2834"无生成式链传播"——因为接口根本不是跨 token 内容传递；③ 2837 decoupled——消融单头输出不破坏调制，因为调制量（注意力质量）本身是 QK 侧的、由更早层状态决定。
3. **头贡献量级与 2841 drop 自洽**：top1 头 share_head≈0.0001-0.0022（占 ||d_spec_full|| 口径），换算占 cdir 分量（≈0.084·||d_spec_full||）约 0.1-2.7%——与 2841 单头消融 drop 1-5.5% 同量级（消融 drop 含下游传播放大）。
4. L33 h25 总贡献 ≈0（−1e-5）：其 1.14% 消融 drop 全部来自下游传播，自身直写可忽略——2841 的"指数尾"在最末端层与"零直写+纯中继"一致。

### 4. 硬伤

1. 序列长 2，无 BOS——来源只有两个位置，"self vs cond"判决受限于此窗口；更长语境（"我喜欢吃苹果"式）下 pos0 可能是多 token，须扩展。
2. share 是 cdir 单方向口径；QK 门的域选择性（颜色/大小/类别差异）未测。
3. bf16 前向 + fp64 分解的数值误差 6.2e-15（恒等式内），但 sain 捕获本身经 float() 截断，与真实 bf16 链有微小偏差。

### 5. 产物登记（immutable + SHA256，Gen4 为准）

- 脚本 `tests/glm5/phase2842_qk_source_backtrace.py` sha256 = `93f8552beceabe5a…`
- `…/phase2842/qk_source_backtrace/execution.json` sha256 = `063ee300adeb6f0b…`
- `result.json` sha256 = `31dbad089f5429a4…`；`source_shares.npz` sha256 = `463e684829f54b3d…`
- Gen 记录：Gen1-3 崩溃（hook kwargs / dtype / reshape），Gen4 成功（15.3s）。预注册判据全程未动。

### 6. 接续（2843 候选）

1. **自绑定门的时间-内容双重验证**：same 条件下 pos1 自注意力增量（Δgate = A_s[1,1] − 0.5(A_f+A_n)[1,1]）与该头 drop 跨词相关；并测 QK 侧干预（把 A[1,1] 钳到 func 水平）能否消除调制——门的因果收口。
2. **多 token 语境扩展**：把来源分解扩展到 ≥4 token 窗口（"我喜欢吃[red]苹果"），测 pos0 群体贡献与 2831 优先级表对接。
3. **QK 门的域选择性**：8 方向组（颜色/大小/类别/速度）下 Δgate 对比——写头门是否域通用。
