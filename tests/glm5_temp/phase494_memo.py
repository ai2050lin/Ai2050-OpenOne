"""Phase 494 MEMO updater"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

memo_text = """

## Phase 494: 跨层因果干预、shared主轴logit分解与异常类别机制 [2026-06-14 20:42]

### 核心突破: 首次完成真正的跨层因果干预

在L(n-2)修改hidden state → 继续forward到L(n-1) → 追踪L(n-1)实际变化。
这是从相关性到因果性的关键一步。

---

### Exp1: 真正跨层因果干预结果

**Qwen3 (L34→L35)**:
| 类别 | ablate ratio | ΔD_n1 | 含义 |
|------|-------------|-------|------|
| fruit | 1.159 | -30.27 | L34消融shared→L35 DCF下降(释放) |
| clothing | 1.319 | -29.00 | 同上 |
| animal | 1.132 | -0.71 | 弱效应 |
| vehicle | 1.534 | -29.72 | 强传递 |
| container | 1.119 | +18.40 | **正!** L34消融→L35 DCF上升(仍刹车) |
| emotion | 1.129 | +90.80 | **强正!** 消融→L35 DCF大幅上升 |

**GLM4 (L38→L39)**:
| 类别 | ablate ratio | ΔD_n1 | 含义 |
|------|-------------|-------|------|
| fruit | 2.472 | +55.28 | L38消融→L39 DCF上升(刹车) |
| clothing | 1.016 | +7.70 | 同上 |
| animal | 1.896 | +34.81 | 同上 |
| emotion | 1.234 | +125.68 | 强刹车 |

**DS7B (L26→L27)**:
| 类别 | ablate ratio | ΔD_n1 | 含义 |
|------|-------------|-------|------|
| fruit | 0.871 | -38.26 | L26消融→L27 DCF下降(释放) |
| clothing | 0.683 | -17.58 | 同上 |
| animal | 1.059 | +90.51 | **异常正!** |
| emotion | 0.982 | -18.49 | 释放 |

**关键发现**:
1. **因果传递成功验证**: L(n-2)的shared修改确实因果传递到L(n-1)
2. **三模型策略差异在因果层面确认**: Qwen3/DS7B消融→ΔD为负(释放), GLM4消融→ΔD为正(刹车)
3. **ablate ratio≠regression slope**: 因果ratio(0.7-2.5)与相关slope(Qwen3约2.7, GLM4约0.9)不同
4. **异常类别有因果层面解释**: container/emotion的ΔD_n1为正, 说明L(n-2)消融后L(n-1)仍然刹车

---

### Exp2: shared_semantic主轴logit分解

**核心发现: shared方向对目标类别logit的贡献在L(n-2)→L(n-1)发生符号翻转**

Qwen3:
- fruit: L34 contrib=+0.0349 → L35 contrib=-0.0233 (正→负=反转!)
- clothing: L34 contrib=+0.0170 → L35 contrib=-0.0304
- emotion: L34 contrib=+0.0680 → L35 contrib=-0.0054 (微弱负!)
- container: L34 contrib=+0.0563 → L35 contrib=-0.0012 (几乎零!)
- action: L34 contrib=-0.0475 → L35 contrib=+0.0138 (负→正=不同方向!)

**这直接解释了为什么Qwen3的释放信号在shared主轴上而非orth子空间**:
- shared方向在L(n-2)是正贡献(刹车方向), 在L(n-1)变为负贡献(释放方向)
- 这不是orth子空间的support/inhibit竞争, 而是shared方向本身功能反转

GLM4: target_contrib始终为负(从-0.04到-0.07), 没有翻转
DS7B: target_contrib从负变正(L26 -0.08 → L27 +0.06), 反转更强

---

### Exp3: 异常类别多层轨迹

**Qwen3 (最后6层)**:
- fruit: L30=+16.3 → L35=-18.5 (L35反转)
- action: L30=+18.1 → L35=-9.96 (**L35反转!** 之前以为是例外,实际反转了)
- container: L30=+22.3 → L35=+0.95 (**接近0, 几乎中性**)
- emotion: L30=+29.3 → L35=+4.07 (**仍正, 弱刹车**)

**GLM4**: 全部类别全部层都为正, 无反转

**DS7B**:
- container: L22=+45 → L27=-100.01 (**L27反转!**)
- emotion: L22=+61 → L27=-166.67 (**L27反转!**)
- action: L22=+49.6 → L27=+7.05 (**不反转, 接近0**)

**关键**: DS7B的container和emotion反转了! Qwen3的action也反转了! 
唯一不反转的: Qwen3的container(中性)和emotion(弱刹车), DS7B的action(弱刹车)

---

### Exp4: 语义类型分组统计

| 组别 | Qwen3 | GLM4 | DS7B |
|------|-------|------|------|
| natural_entity | 4/4 (100%) | 0/4 (0%) | 4/4 (100%) |
| artifact | 5/7 (71%) | 0/7 (0%) | 7/7 (100%) |
| abstract | 0/1 (0%) | 0/1 (0%) | 1/1 (100%) |
| action | 1/1 (100%) | 0/1 (0%) | 0/1 (0%) |
| substance | 2/2 (100%) | 0/2 (0%) | 2/2 (100%) |
| location | 1/1 (100%) | 0/1 (0%) | 1/1 (100%) |

**关键发现**: 
- DS7B几乎全反转(15/16), 唯一例外是action
- Qwen3大部分反转(13/16), 例外是artifact中的vehicle/container和abstract的emotion
- GLM4全不反转(0/16)
- **action是唯一三模型都较难反转的类别**

---

### 五大核心客观发现

1. **跨层因果传递成功验证**: L(n-2)消融shared→L(n-1) DCF变化, ratio=0.7-2.5, 证明因果链存在
2. **shared方向的logit贡献发生符号翻转**: Qwen3的shared_dir在L(n-2)对target是正贡献, L(n-1)变负, 这是释放的真正机制
3. **三模型因果策略差异确认**: Qwen3/DS7B消融→DCF下降(释放), GLM4消融→DCF上升(刹车)
4. **action是唯一跨模型难以释放的类别**: 三模型action的末层ablate_shared都接近0或弱正
5. **DS7B的container/emotion在R1测试中反转了**: 之前Phase 493的判断需要修正

---

### 硬伤与瓶颈

1. **因果干预只用了均值替换**: 用h_target_mean替换L(n-2)整个hidden state, 这丢失了逐样本变异性
2. **hook替换方法有局限**: 只替换一个token位置的hidden state, 没考虑其他位置
3. **action的机制仍未解释**: 为什么action跨模型都不走释放路径?
4. **ablate ratio和double ratio不对称**: Qwen3 ablate ratio=1.159, double ratio=0.976, 说明传递非线性
5. **DS7B animal的ΔD_n1=+90.51异常**: 与fruit/clothing符号相反, 未解释

---

### 下一步核心任务

1. **逐样本因果干预**: 不用均值替换, 对每个样本分别修改L(n-2)再forward
2. **action类别专项研究**: 为什么action不走释放路径? 是否用不同的编码机制?
3. **shared方向功能反转的层内机制**: L(n-1)的Attention/MLP如何实现shared从刹车到支撑的反转?
4. **非线性传递分析**: ablate和double的ratio不对称, 需要理解非线性效应
"""

with open("research/glm5/docs/AGI_GLM5_MEMO.md", "a", encoding="utf-8") as f:
    f.write(memo_text)

print("MEMO updated successfully!")
