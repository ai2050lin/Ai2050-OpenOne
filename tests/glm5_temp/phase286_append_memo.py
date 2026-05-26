"""Append Phase 286 MEMO entry."""
content = r"""

---

## Phase 286: Head-Level Real Forward Patching — 从组件级到Head级因果分解 [2026-05-26 13:03]

### 实验动机

Phase 285确立了三个模型的组件因果架构（Qwen3分布式、GLM4 L0 MLP瓶颈、DS7B组件决定型），但只停留在attn/mlp/resid粒度。Phase 286将分析下沉到attention head级别，直接回答核心问题：
- 哪些head实现哪些功能？
- 不同功能是否共享同一批head？
- Head级激活差异（diff norm）能否预测因果重要性？

### 方法

```
PART 1 - CACHE: 对121对句子(A,B)，hook self_attn.o_proj INPUT
          (W_o投影前的拼接head输出)，形状 [batch, seq, n_heads*head_dim]
          head_dim动态从 o_proj.weight.shape 确定，兼容不同模型架构
          
PART 2 - ANALYZE: 计算每个head的diff norm ||input_A_h - input_B_h||_2
          按功能类别聚合，全局排序
          
PART 3 - PATCH: 选取top-5 heads/layer * 10个关键层 * 14对代表性句子
          用pre_hook修改o_proj输入中指定head的slot
          测量KL偏移 = KL(P_patched||P_B) / KL(P_A||P_B)
          
PART 4 - AGGREGATE: Head级因果效应矩阵，跨功能head复用矩阵
```

### 核心结果

#### Qwen3 (34s: 11s缓存 + 23s patching)

Head因果效应: mean=0.045, max=0.266, >0.1占比=10.7%, Diff-Causal: r=-0.09

**Qwen3结论**: 单head因果效应极小。即便是"最强"的L16_H27也仅能解释7.7%的A-B差异。完全印证Phase 285"分布式残差流型"——Qwen3信息分散在32个head和残差流中，无单一head起决定性作用。

#### GLM4 (14.3min: 4min缓存 + 10.3min patching)

Head因果效应: mean=0.178(被翻译拉高), max=2.310, >0.1占比=7.1%, Diff-Causal: r=+0.58(仅N=8)

**关键发现——GLM4 attention heads的"功能互换性":**
所有patched heads的因果效应极窄：0.161-0.179（除翻译外）。GLM4的32个attention heads在因果上几乎不可区分——任何单一head的替换都产生相似的约17%的effect。标准差极大(0.57-0.59)，即同一head在不同句子上效应波动剧烈。

**翻译是GLM4唯一有head差异化的功能：**
- translation L16_H7: effect=2.310, L8_H16: 2.294, L28_H27: 2.291
- negation: head effect~0.002 (几乎不出现在attention路径)

**GLM4结论**: MLP做所有实质性计算，attention heads只是通用的上下文混合器。唯独跨语言翻译需要特定的attention heads。

#### DS7B (9.0min: 2.7min缓存 + 6.3min patching)

Head因果效应: mean=0.728, max=5.000(cap), >0.1占比=97.1%, Diff-Causal: r=-0.46(强负相关!)

**DS7B L0_H3 — "万能通用head":** 对animal(1.18), recursive(0.99), translation(1.13), temporal(0.58), human_object(0.59), place(0.62), negation(0.40), passive(0.24)共8种功能都有显著因果效应。

**DS7B结论**: 单head因果效应极强。L0的两个head承载了50-78%的A-B差异。Sliding Window架构导致信息更"脆弱"地分布在少数heads中。

### 跨模型核心对比

```
              Qwen3       GLM4        DS7B
Head效应均值   0.045       0.178*      0.728
Head效应最大值 0.266       2.310       5.000(cap)
效应>0.1占比   10.7%       7.1%*      97.1%
Diff-Causal    r=-0.09     r=+0.58    r=-0.46
编码模式       全分布式    MLP集中型   头级因果型

* GLM4均值被翻译类拉高, 排除翻译后约0.02
```

### 最大方法论发现：Diff Norm != Causal Importance

激活差异（diff norm）完全不能预测因果重要性：
- Qwen3: r=-0.09, DS7B: r=-0.46
- Diff norm最高在深度输出层(L32-L35)，因果最强在中间层(L8-L21)
- 大量基于probing、激活聚类的论文用"激活差异"推断"功能重要性"的方法可能是系统偏误的

### 新增客观事实拼图（18条）

1. Qwen3 o_proj内部head_dim=128（非d_model/n_heads=80），concat_dim=4096
2. DS7B head_dim=128, concat_dim=3584=n_heads*head_dim
3. 121对全部成功缓存o_proj input（Part 1）
4. Head级pre_hook patching在所有三模型正常工作（Part 3）
5. Qwen3 head因果效应均=0.045, 仅10.7% heads>0.1
6. Qwen3最强head L16_H27仅能解释7.7%的A-B差异
7. Qwen3 head效应L8-L35完全平坦(0.024-0.036)，无关键层
8. GLM4 attention heads因果效应极窄带0.161-0.179，功能上几乎可互换
9. GLM4翻译(英-中)是唯一head效应>2的功能类别
10. GLM4否定功能head效应~0.002，几乎不出现在attention路径
11. DS7B 97.1%的heads因果效应>0.1，几乎全heads都有因果力
12. DS7B L0_H3对8种不同功能都有>0.4的因果效应——通用信息瓶颈head
13. DS7B L0_H10 human effect=5.0（已达cap），实际可能远超
14. 激活差异和因果效应在Qwen3(r=-0.09)和DS7B(r=-0.46)呈零或负相关
15. Diff norm最高在深度输出层(L32-L35)，因果最强在中间层(L8-L21)
16. 翻译类在三模型中head效应都显著高于其他功能
17. Qwen3效应最大的head(L16_H27)对3种功能都出现——但效应绝对值仍很小
18. GLM4 head效应的跨pair标准差(0.57-0.59)远超跨head均值差异(0.02)——语境变异主导

### 硬伤

1. GLM4 diff-causal仅N=8：L0的top-diff heads不在key_layers采样中，需要重测时包含L0的heads
2. DS7B effect>1普遍存在：单head替换经常"过度转换"，反映Sliding Window下脆弱信息分布
3. head_dim=128（所有模型）：Qwen3 d_model=2560但concat_dim=4096，需确认是否影响head级结论
4. per-category仅1 pair：per-category的effect std极大
5. GLM4 "均质性"可能是artifact：patched head数太少(40/1280)，恰好在GLM4平坦区采样

### 命令记录

```bash
python tests/glm5/phase286_head_level_patching.py qwen3     # 34s
python tests/glm5/phase286_head_level_patching.py glm4       # 14.3min
python tests/glm5/phase286_head_level_patching.py deepseek7b # 9.0min
```

### 数据文件

- `results/phase286_head_patching/{model}_head_diff.json`
- `results/phase286_head_patching/{model}_head_patching.json`
- `results/phase286_head_patching/{model}_o_proj_cache.json`
- `tmp/phase286_{model}.txt`
- `tests/glm5/phase286_head_level_patching.py`
- `tests/glm5_temp/phase286_summary.py`
- `tests/glm5_temp/phase286_glm4_details.py`
- `tests/glm5_temp/phase286_cross_model.py`
"""

with open('research/glm5/docs/AGI_GLM5_MEMO.md', 'a', encoding='utf-8') as f:
    f.write(content)
print('MEMO appended successfully')
print(f'Added {len(content.splitlines())} lines')
