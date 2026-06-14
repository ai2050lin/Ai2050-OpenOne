"""Phase 490 MEMO更新脚本"""
import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')

memo_path = "research/glm5/docs/AGI_GLM5_MEMO.md"

timestamp = time.strftime("%Y-%m-%d %H:%M")

memo_text = f"""

## Phase 490: 功能子空间层位连续曲线与末层分解 ★★★核心层位分化发现★★★ [2026-06-14 {timestamp}]

### ★★★核心发现: shared_semantic消融效果随层位先增后剧变反转, 末层-1和末层功能完全不同★★★

Phase 489发现shared_semantic效应模型特异和层位特异。Phase 490通过全层扫描精确揭示了层位功能分化。

### Exp1: shared_semantic消融全层扫描 ★★★★★ 最关键

**ablate_shared target_delta随层位变化:**

| 层位 | Qwen3 fruit | Qwen3 clothing | GLM4 fruit | GLM4 clothing | DS7B fruit | DS7B food |
|------|------------|----------------|-----------|----------------|-----------|----------|
| L0 | -0.046 | -0.102 | +0.012 | +0.009 | +0.187 | -0.055 |
| L6 | +0.183 | +0.257 | +0.017 | +0.027 | +2.882 | +1.317 |
| L12 | +1.030 | +0.111 | +0.054 | +0.092 | +4.318 | +4.290 |
| L18 | +2.561 | +1.003 | +0.195 | +0.239 | +13.096 | +9.425 |
| L24 | +6.437 | +2.270 | +0.858 | +0.756 | +63.763 | +30.054 |
| L30 | +17.792 | +7.829 | +2.935 | +2.381 | — | — |
| L34 | +23.133 | +11.608 | +7.343 | +5.323 | — | — |
| L(n-2) | +23.1 | +11.6 | +16.0 | +10.2 | +93.5 | +51.6 |
| **L(n-1)** | **-18.8** | **-25.6** | **+8.9** | **+3.3** | **-121.1** | **-71.8** |

★★★关键发现1: 所有模型在L(n-2)都是强的刹车(ablate_shared→边界增强),
但L(n-1)发生剧烈反转, 变成支撑(Qwen3/DS7B)或明显减弱(GLM4)★★★

★★★关键发现2: DS7B的效应极端放大, L26→+93.5, L27→-121.1, 这不是正常尺度★★★

### Exp2: 末层orth_bc分解为支撑/抑制/中性 ★★★★★

**DS7B (效应最大, 最清晰):**
| 层 | 支撑n | 支撑delta | 抑制n | 抑制delta | 中性n |
|----|-------|----------|-------|----------|-------|
| L27(末层) | 6 | **-54.3** | 2 | +3.7 | 0 |
| L26(末-1) | 1 | -0.07 | 6 | **+47.4** | 1 |

**Qwen3:**
| 层 | 支撑n | 支撑delta | 抑制n | 抑制delta | 中性n |
|----|-------|----------|-------|----------|-------|
| L35(末层) | 4 | **-7.3** | 4 | +5.9 | 0 |
| L34(末-1) | 3 | -1.7 | 5 | +10.1 | 0 |

★★★关键发现3: 末层(n-1)以支撑成分为主, 末层-1(n-2)以抑制成分为主
这精确解释了为什么ablate_shared在末层-1是刹车, 在末层反转为支撑★★★

### Exp3: 竞争类别边界因果 ★★★

**DS7B (最关键):**
| 层 | fruit→all_comp | food→all_comp |
|----|---------------|--------------|
| L7 | -0.379 | -0.581 |
| L14 | +0.013 | +0.018 |
| L21 | +0.245 | +0.044 |
| L27 | **-34.0** | **-15.5** |

★★★关键发现4: 竞争类别方向在末层有巨大因果效应(DS7B fruit L27: -34.0)
但在中间层效应很小(L14: +0.013), 说明竞争控制在末层才集中生效★★★

### Exp4: 早层vs中晚层shared_semantic ★★★

**Qwen3 fruit (跨层完整):**
- L4(early): shared_abl=-0.255 (弱支撑)
- L6(early): shared_abl=+0.183 (弱刹车)
- L18(mid): shared_abl=+2.561 (刹车)
- L27(late): shared_abl=+12.878 (强刹车)
- L34(late): shared_abl=+23.133 (极强刹车)
- L35(late): shared_abl=-18.782 (反转!支撑!)

★★★关键发现5: shared_semantic从早层的弱混合效应, 到中层的刹车,
到末层-1的极强刹车, 再到末层的剧烈反转为支撑, 形成完整的层位功能梯度★★★

### 对Phase 489结论的重要补充

Phase 489说: "shared_semantic效应是模型特异和层位特异的"

Phase 490更精确地发现:
1. 层位分化有精确结构: L(n-2)是刹车峰, L(n-1)是支撑峰
2. 这种分化跨模型一致(Qwen3/DS7B最清晰, GLM4末层也是减弱)
3. DS7B效应极端放大(可能是其架构特征: 小模型大效应)
4. 竞争类别控制在末层才集中生效
5. 末层orth_bc分解确认: 支撑vs抑制成分的层位分化

### 新增客观事实(6条)

80. Qwen3 fruit: ablate_shared从L0(-0.05)到L34(+23.1)递增(刹车), L35反转为-18.8(支撑)
81. GLM4 fruit: ablate_shared从L0(+0.01)到L38(+16.0)递增(刹车), L39降到+8.9
82. DS7B fruit: ablate_shared从L0(+0.19)到L26(+93.5)极增(刹车), L27反转为-121.1(支撑)
83. DS7B food: ablate_shared从L0(-0.06)到L26(+51.6)递增(刹车), L27反转为-71.8(支撑)
84. 末层orth_bc分解: DS7B L27支撑6个方向(delta=-54.3), L26抑制6个方向(delta=+47.4)
85. 竞争类别方向在末层集中生效: DS7B fruit L27 all_comp=-34.0, L14 only +0.013

### 命令记录

python tests/glm5/phase490_layer_sweep_decomposition.py qwen3 1       # ~30s
python tests/glm5/phase490_layer_sweep_decomposition.py glm4 1          # ~10min
python tests/glm5/phase490_layer_sweep_decomposition.py deepseek7b 1   # ~5min
"""

with open(memo_path, "a", encoding="utf-8") as f:
    f.write(memo_text)

print(f"MEMO updated at {timestamp}")
