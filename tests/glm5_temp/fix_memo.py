#!/usr/bin/env python3
"""Fix Phase 145 garbled text in AGI_GLM5_MEMO.md"""

with open('research/glm5/docs/AGI_GLM5_MEMO.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print(f"Total lines: {len(lines)}")

# Find the line that contains "Phase 145" (may be garbled around it)
phase145_start = None
for i, line in enumerate(lines):
    if 'Phase 145' in line:
        phase145_start = i
        print(f"Found Phase 145 at line {i+1}")
        break
    # Also check for garbled version
    raw = line.encode('utf-8')
    if b'Phase 145' in raw:
        phase145_start = i
        print(f"Found Phase 145 (bytes) at line {i+1}")
        break

if phase145_start is None:
    print("Phase 145 not found! Searching around line 63355...")
    for i in range(63350, min(63360, len(lines))):
        try:
            print(f"Line {i+1}: {repr(lines[i][:100])}")
        except:
            print(f"Line {i+1}: [encoding error]")
else:
    # Replace from phase145_start to end of file
    correct_text = """
## Phase 145: 吸引子动力学 — 扰动不被纠正的发现 [2026-05-12 22:30]

### 直接检验用户第三次批评的核心假说

用户提出四大高价值实验:
1. 吸引子恢复实验(最高优先级): 扰动后系统是否回归原轨道?
2. 稳定/不稳定模态谱: 哪些方向被压制,哪些方向持续存在?
3. 约束修复时间: 不同约束类型在哪层被修复?
4. Activation causal graph: 哪个head负责哪个约束?

### 四大实验结果 (Qwen3, n_layers=36, d_model=2560)

#### Exp A: 吸引子恢复实验 (最关键!)

方法: 对hidden state注入扰动(eps=0.5-5.0),观察后续层是否回归原轨道

关键数据 (eps=2.0, 10个句子平均, 归一化到首次扰动出现层):

| 注入层 | random peak | random final | semantic peak | semantic final | constraint peak | constraint final |
|--------|------------|-------------|--------------|---------------|----------------|-----------------|
| L0 | 7.38x | 1.90x | 6.93x | 1.72x | 7.33x | 1.84x |
| L9 | 4.83x | 1.27x | 5.46x | 1.41x | 5.31x | 1.37x |
| L18 | 4.72x | 1.22x | 5.05x | 1.29x | 4.81x | 1.21x |
| L27 | 2.74x | 0.80x | 3.10x | 0.87x | 2.99x | 0.84x |

扰动轨迹三阶段 (L0注入, eps=2.0):
- L1-L17: 微弱波动 (0.94-1.12x), 扰动基本保持
- L18-L35: 持续增长 (1.02->8.58x), 每层平均放大~1.1x
- L35->L36: 骤降至2.55x (压缩比=0.297)

核心发现:
1. 早/中层注入: 扰动最终被弱放大(1.2-1.9x), 不是吸引子!
2. 晚层注入: 扰动最终被部分纠正(0.80x), 但可能是末层效应
3. 三种扰动类型(随机/语义/约束)的行为几乎相同
4. 语义扰动恢复更好的比例仅60%(6/10), 统计不显著

不同eps下的末层骤降比:
| eps | final/peak |
|-----|-----------|
| 0.5 | 0.261 |
| 1.0 | 0.249 |
| 2.0 | 0.257 |
| 5.0 | 0.246 |

末层骤降比与eps无关(~0.25), 强烈支持LayerNorm归一化效应!

#### Exp B: 稳定/不稳定模态谱

方法: 计算每层Jacobian的奇异值谱(80维采样)

| 层 | Top-5 SV | Bottom-5 SV | PR | contract(<0.5) | expand(>1.5) |
|----|---------|------------|-----|---------------|-------------|
| L0 | 1.002 | 1.001 | 80.0 | 0/80 | 0/80 |
| L18 | 1.005 | 1.000 | 80.0 | 0/80 | 0/80 |
| L35 | 0.434 | 0.216 | 77.9 | 79/80 | 0/80 |

核心发现:
1. 早/中间层: Jacobian近似单位矩阵(SV~1.002), 所有方向被同等保持
2. 末层: 所有多方向强收缩(SV<0.5), 无选择性
3. 没有发现语义方向被保留,非语义方向被修正的证据
4. 所有方向被同等对待,不存在方向选择性

#### Exp C: 约束修复动力学

| 约束类型 | init | peak@L35 | final@L36 | L35->L36 ratio | decay% |
|---------|------|----------|-----------|---------------|--------|
| SVA | 0.000 | 116.56 | 25.23 | 0.216 | 78% |
| TENSE | 0.000 | 124.57 | 28.05 | 0.225 | 77% |
| SCOPE | 1.563 | 377.47 | 118.25 | 0.313 | 69% |
| LOGIC | 0.316 | 222.97 | 60.96 | 0.273 | 73% |
| SEMANTIC | 0.000 | 189.48 | 45.74 | 0.241 | 76% |

核心发现:
1. 所有约束类型的L35->L36骤降比几乎相同(0.21-0.31)
2. 骤降比与eps无关(~0.25), 是LayerNorm效应
3. 不同约束类型的行为高度一致, 无特异性修复

#### Exp D: 语义vs随机扰动恢复
问题: W_U的SVD计算失败(init_gesdd failed init), 语义方向向量全为0
结果: 仅random方向有有效数据, 语义方向无法测试

### Phase 145最重要的发现

发现1: Transformer不是吸引子系统
扰动不被纠正! 早层注入,扰动最终被弱放大(1.2-1.9x)。不存在轨道回归现象。

发现2: 末层骤降是LayerNorm归一化
证据: L35->L36压缩比与eps无关(~0.25); 不同约束类型的压缩比几乎相同。

发现3: 中间层是信号放大器,不是约束路由器
证据: Jacobian SV~1.002; 无expand模式(>1.5); 无contract模式(<0.5)。

发现4: 语义vs随机扰动无显著差异
语义扰动恢复更好的比例仅60%(6/10),远低于约束稳定传播假说预测的100%。

### 对用户理论框架的系统检验

| 用户论点 | Phase 145检验 | 结论 |
|---------|-------------|------|
| Transformer是约束稳定传播系统 | 扰动不被纠正,反而被放大 | 被否定 |
| 存在吸引子结构 | 轨道不回归原点 | 被否定 |
| 语义方向被保留,非语义方向被修正 | 所有方向被同等对待(SV~1.0) | 被否定 |
| 中间层是约束混合器 | 中间层是信号放大器 | 需修正:混合->放大 |
| MLP是约束修正器 | alignment~0.5 | 需弱化:部分修正 |
| Attention是约束路由 | 功能分化弱 | 证据不足 |
| 末层有特殊结构 | LayerNorm统一收缩 | 确认 |
| 语言能力来自约束传播 | 约束被放大不被修正 | 需修正:传播->放大 |

### 修正后的理论框架

Phase 144旧框架: 语言 = 局部光滑传播 x 语义分层边界 x 中等修正 x 低秩解码

Phase 145修正后: 语言 = 信号分层放大 + 末层归一化 + 低秩解码几何

1. 信号分层放大: 中间层近乎保持所有方向(SV~1.0),但累积效应使扰动增长到3-9x
2. 末层归一化: LayerNorm统一压缩所有方向(~0.25x),无选择性
3. 低秩解码几何: W_U的低秩结构决定了哪些信号影响输出
4. 语言能力来自W_U几何, 不是来自中间层的约束路由

### 为什么与Phase 144的Jacobian一致性不矛盾?

Phase 144发现: cos(J_s1*v, J_s2*v) 随语义距离递减
Phase 145发现: 所有方向的SV~1.0,无选择性放大/压制

不矛盾,因为:
- Phase 144测的是方向一致性: 相似语义点的Jacobian方向相似
- Phase 145测的是幅度行为: 所有方向的SV~1.0
- 方向一致但幅度无选择性: 语义通过方向编码,不通过幅度选择性编码

### 严峻的问题和瓶颈

1. 只有Qwen3数据: GLM4和DS7B的8bit模式hook不工作
2. Jacobian采样不足: 80/2560维采样可能遗漏重要的放大方向
3. Exp D失败: W_U SVD计算失败
4. 中间层放大的来源: 单层Jacobian~I但累积放大8x,需理解非线性的累积效应
5. 方向vs幅度的分离: 扰动幅度放大8x,但方向是否保持?如果方向保持,输出可能仍然正确

### 破解语言数学原理的第一性原理

Phase 140-145的完整图景:

语言能力的数学结构 = 低维信号空间 + 方向一致性传播 + 统一归一化 + 低秩解码

1. 低维信号空间(Phase 140): PR~70, hidden states集中在低维壳层
2. 方向一致性传播(Phase 144): 相似语义点的Jacobian方向一致(cos~0.8)
3. 统一归一化(Phase 145): 所有方向被同等放大/压制,无选择性
4. 低秩解码(Phase 140-142): W_U的低秩结构决定输出

核心洞察: 语言能力不来自约束如何被路由和修正,而来自信号如何在低维空间中被方向一致地传播,最终通过低秩几何被解码。

方向一致性(cos~0.8 for semantic neighbors)是关键:它使得语义近邻的信号被相似地处理,从而产生连贯的输出。这不是约束修正,而是方向惯性——系统倾向于保持相似的处理方式。

### 下一步方向

Phase 146 (关键实验): 方向一致性vs幅度行为的分离
- 同时测量扰动传播的方向变化和幅度变化
- 如果方向被保持(方向一致性高),即使幅度放大,输出可能仍然正确

Phase 147: 低秩解码几何的深入分析
- W_U的SVD结构和信号方向的关系
- 即使扰动放大8x,如果放大方向在W_U的null space中,输出不受影响

Phase 148: 非线性累积放大效应
- 为什么单层Jacobian~I,但累积放大8x?
- 需要理解每层的微小偏差如何通过非线性累积

### 测试脚本和结果
- 主脚本: tests/glm5/phase145_attractor_dynamics.py
- Qwen3结果: tests/glm5_temp/phase145_qwen3_attractor_20260512_2119.json
- GLM4/DS7B: 8bit模式下hook不工作
- 综合分析: tests/glm5_temp/phase145_comprehensive.py
- 最终分析: tests/glm5_temp/phase145_final_analysis.py
"""

    # Keep lines before Phase 145, replace from phase145_start to end
    new_lines = lines[:phase145_start] + [correct_text]
    
    with open('research/glm5/docs/AGI_GLM5_MEMO.md', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"Fixed! Old lines: {len(lines)}, New lines: {len(new_lines)}")
    print(f"Phase 145 starts at line {phase145_start+1}")
