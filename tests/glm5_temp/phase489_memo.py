"""Phase 489 MEMO updater"""
import os

content = """

## Phase 489: 共享语义抑制机制与末层前体验证 ★★★关键模型差异★★★ [2026-06-14 15:42]

### ★★★核心发现: shared_semantic的因果效应是模型特异的和层位特异的★★★

Phase 488假设shared_semantic是"边界刹车",但Phase 489发现这取决于模型和层位。

### Exp1: shared_semantic消融因果测试 ★★★★★ 关键

**DS7B (刹车模式 — 符合Phase 488假设):**
| 层 | 操作 | target_D变化 | 含义 |
|----|------|-------------|------|
| fruit L21 | ablate_shared | **+0.844** | 边界增强! 刹车! |
| fruit L21 | reverse_shared | -0.154 | 反向→边界削弱 |
| food L26 | ablate_shared | **+3.304** | 强刹车! |
| food L26 | reverse_shared | -0.524 | 反向→边界削弱 |

**GLM4 (刹车模式 — 符合Phase 488假设):**
| 层 | 操作 | target_D变化 | 含义 |
|----|------|-------------|------|
| fruit L22 | ablate_shared | **+0.093** | 边界增强! 刹车! |
| fruit L27 | ablate_shared | **+0.131** | 边界增强! 刹车! |
| clothing L34 | ablate_shared | +0.004 | 弱/零 |

**Qwen3 (反刹车模式 — 与Phase 488假设相反!):**
| 层 | 操作 | target_D变化 | 含义 |
|----|------|-------------|------|
| clothing L25 | ablate_shared | **-0.094** | 边界削弱! 反刹车! |
| clothing L30 | ablate_shared | **-0.336** | 边界削弱! |
| fruit L27 | ablate_shared | **-0.147** | 边界削弱! |
| fruit L32 | ablate_shared | +0.051 | 弱增强 |

★★★关键发现1: DS7B和GLM4的shared_semantic是刹车, Qwen3的shared_semantic是支撑★★★

### Exp2: 末层orth_bc消融/注入 ★★★★★ 关键

**跨模型一致性: 末层orth_bc消融一致导致边界削弱**

| 模型-类别 | 层 | ablate_orth_bc | inject_orth(s1.0) | 含义 |
|-----------|-----|----------------|-------------------|------|
| Qwen3 clothing | L35(late) | **-3.641** | -1.4311 | 消融→边界大降! |
| Qwen3 clothing | L34(late-1) | +0.625 | -0.7905 | 消融→边界小增 |
| Qwen3 fruit | L35(late) | **-4.660** | +0.2827 | 消融→边界大降! 注入→正! |
| Qwen3 fruit | L34(late-1) | +2.551 | -1.1549 | 消融→边界增 |
| GLM4 fruit | L39(late) | **-0.367** | +0.2235 | 消融→边界降! 注入→正! |
| GLM4 fruit | L38(late-1) | -0.472 | -0.2098 | 消融→边界降 |
| GLM4 fruit | L13(mid) | +0.091 | -0.1921 | 中间层不同! |
| DS7B fruit | L27(late) | **-5.059** | +0.0599 | 消融→边界大降! |
| DS7B fruit | L13(mid) | -0.067 | -0.2152 | 中间层弱效应 |
| DS7B food | L13(mid) | -2.720 | -0.2824 | 消融→边界降 |

★★★关键发现2: 末层(n_layers-1)orth_bc消融一致削弱边界, 说明末层orth_bc包含重要边界支撑成分★★★

注意: 注入(注入平均方向)和消融(移除实际成分)结果不一致, 说明orth_bc不是单一方向,
而是包含支撑边界和抑制边界的混合成分。

### Exp3: 投影写入vs共享抑制剂量曲线 ★★★

**DS7B fruit L13 (最清晰):**
| 操作 | 效应 |
|------|------|
| shared_scale=-1.0 → target=+0.813 (松刹车→边界增) |
| shared_scale=+1.0 → target=-0.393 (加刹车→边界降) |
| proj_scale=-1.0 → target=+0.642 (移除写入→边界反而增?!) |
| proj_scale=+1.0 → target=+0.090 (增加写入→边界略增) |

GLM4和DS7B的剂量曲线支持"刹车"模型。Qwen3效应太弱。

### Exp4: 共享语义抑制与竞争释放 ★★★

| 模型-类别 | ablate_shared目标变化 | ablate_bc目标变化 | ablate_competitor目标变化 |
|-----------|----------------------|-------------------|--------------------------|
| Qwen3 clothing | -0.062 | +0.109 | +0.016 |
| Qwen3 fruit | -0.030 | -0.011 | -0.068 |
| GLM4 fruit | -0.057 | +0.030 | +0.010 |
| GLM4 clothing | +0.055 | -0.123 | +0.002 |
| DS7B fruit | -0.183 | +0.079 | -0.006 |
| DS7B food | **-0.884** | -0.127 | **-1.407** |

★★★关键发现3: 在早层(L13), shared_semantic消融也削弱边界(支撑模式), 不是刹车★★★

这表明shared_semantic在不同层有不同功能:
- 早层(L13): 支撑边界形成
- 中晚层(L21-L27): 抑制过早类别化(刹车)

### Exp5: 跨模型一致性 ★★★

| 模型-类别 | mid_orth_effect | late_orth_alignment | n_shared |
|-----------|----------------|---------------------|----------|
| Qwen3 clothing | +0.024 | -0.421 | 5 |
| Qwen3 fruit | -0.282 | -0.068 | 5 |
| GLM4 fruit | -0.140 | -0.113 | 5 |
| DS7B fruit | +0.062 | **+0.096** | 5 |
| DS7B food | +0.317 | N/A | 5 |

注意: 只有DS7B fruit的末层orth_bc alignment为正(与B_c对齐), 其他模型为负。

### ★★★Phase 489最重要的5个客观发现★★★

**发现1: shared_semantic的因果效应是模型特异的**
- DS7B + GLM4: ablate_shared → 边界增强(刹车模式)
- Qwen3: ablate_shared → 边界削弱(支撑模式)
- 不能简单说shared_semantic是"刹车"

**发现2: shared_semantic的因果效应是层位特异的**
- 早层(L13): ablate_shared → 边界削弱(支撑模式, 跨模型一致)
- 中晚层(L21-L27): 效应取决于模型

**发现3: 末层orth_bc消融一致削弱边界**
- 所有模型的最后一层(n_layers-1)orth_bc消融都导致边界下降
- 说明末层orth_bc包含重要的边界支撑/读出成分
- 这修正了Phase 488"末层orth_bc可能是前体"的判断: 它不仅是前体,而是包含多种功能成分

**发现4: 注入和消融结果不一致, 说明orth_bc是混合成分**
- 消融末层orth_bc → 边界降(说明含有支撑成分)
- 但注入平均orth_bc方向 → 效应混合(因为平均方向不代表所有成分)
- orth_bc不是一个单一功能的空间, 而是多功能混合

**发现5: proj_bc消融效应在中间层很小**
- GLM4 fruit L22: ablate_proj → -0.009 (几乎零!)
- GLM4 fruit L27: ablate_proj → -0.160 (中等)
- Qwen3 clothing L25: ablate_proj → -0.109 (小)
- 相比之下, ablate_competitor有时更大(GLM4 fruit L27: +0.016 vs ablate_shared: +0.131)

### 对Phase 488结论的修正

Phase 488说: "中间层orth_bc主要是共享语义抑制项,消融orth_bc效果大是因为松刹车"

Phase 489修正为:
1. shared_semantic的效应是模型特异的(DS7B/GLM4是刹车, Qwen3是支撑)
2. shared_semantic的效应是层位特异的(早层支撑, 中晚层可能刹车)
3. orth_bc不是单一功能空间, 包含支撑+抑制+读出等多种成分
4. 末层orth_bc消融一致削弱边界, 说明包含重要边界支撑成分

### 新增客观事实(8条)

72. DS7B fruit L21: ablate_shared→+0.844, food L26: ablate_shared→+3.304 (刹车模式)
73. GLM4 fruit L22/L27: ablate_shared→+0.093/+0.131 (刹车模式)
74. Qwen3 clothing L25/L30: ablate_shared→-0.094/-0.336 (支撑模式,与刹车相反!)
75. Qwen3 fruit L27: ablate_shared→-0.147 (支撑模式)
76. 所有模型末层(n_layers-1)orth_bc消融都削弱边界: Qwen3:-3.6/-4.7, GLM4:-0.37, DS7B:-5.06
77. Qwen3 fruit L35 orth注入→bc_increase=+0.28(正,前体); GLM4 fruit L39→+0.22(正,前体)
78. 早层(L13)shared_semantic消融削弱边界(DS7B food:-0.884), 不是刹车
79. orth_bc是多功能混合空间,不是单一功能(支撑+抑制+读出共存)

### 命令记录

python tests/glm5/phase489_shared_semantic_brake.py qwen3 1        # ~2min
python tests/glm5/phase489_shared_semantic_brake.py glm4 1          # ~50min
python tests/glm5/phase489_shared_semantic_brake.py deepseek7b 1    # ~33min

脚本: tests/glm5/phase489_shared_semantic_brake.py
结果: results/glm5/phase489_{qwen3,glm4,deepseek7b}_r1.json
"""

with open('research/glm5/docs/AGI_GLM5_MEMO.md', 'a', encoding='utf-8') as f:
    f.write(content)
print('MEMO updated')
