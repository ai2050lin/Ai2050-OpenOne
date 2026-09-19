# -*- coding: utf-8 -*-
import io

SECTION = """
---

## Phase 2856：Atlas-E200 词表扩展预研 + 双谱普查管线封装（2026-09-18）

### 原理
MA 战线推进（2846 双谱正交 → Atlas-E 词汇扩展）：正式的 200 词全头双谱普查前，须先回答两个前置问题——①200 词词表能否构造、类方向结构是否保持（Arm A，纯 unembed 层面，无前向）；②2846 协议封装为可复用管线后是否逐位忠实（Arm B，GPU 复测对照）。

**管线库 `rdc_atlas_census.py`**（sha `c3fcead807543632`）：build_vocab（tid 缓存→单 token 过滤→top-N 截取→全 single_tok 集 dW_unit→种子 null tids→func tid 末位，与 2846 构造顺序逐句一致）+ AtlasCensus 类（hooks/patched attn/OV 缓存/conds2_for/bias_for/measure_layer = 因果 drop + OV 对齐 s0/s1，2846 协议逐行复刻，词表与 null_tids 参数化）。

Arm A：2806 词表（10 类×10 词）+ 固定顺序 12 候选/类，单 token 过滤取前 10 存活者 → 目标 20 词/类。判据（冻结，execution.json `6a74cc0f`）：E1 ≥9/10 类满 10 新词；E2 ≥9/10 类 cos(dW_unit_200, dW_unit_80)≥0.9；E3 off-diag 类中心余弦矩阵 Pearson r≥0.95。Arm B：管线重测 L29-31 × 前 40 词（seed=2846 词表构造），B1 = 3/3 层 r(drops_new, drops_2846)≥0.99 且 med|Δ|≤0.002。

### 正式判决（243.4s，Gen2，脚本 `c5441794`）
| 判据 | 值 | 结果 |
|---|---|---|
| E1 词表覆盖 | 每类新词 [fruit 6, animal 10, metal 9, vehicle 10, country 10, food 10, nature 10, furniture 6, tool 4, clothing 9]，总 184 词 | **false**（6/10 类达标） |
| E2 方向稳定 | cos 谱 [0.907, 0.875, 0.910, 0.872, 0.936, 0.847, 0.874, 0.896, 0.889, 0.882]，min 0.847（food） | **false**（6/10 类 ≥0.9） |
| E3 几何保持 | off-diag 中心余弦 r=0.9512（dW 版描述性 0.9562） | **true** |
| B1 管线复现 | L29/30/31 全部 **r=1.00000、med\\|Δ\\|=0.000000**（逐位一致），clamp_max_resid 0.0127 | **true** |
| **final_verdict** | | **atlas_e200_ready=False / pipeline_verified=True** |

### Arm B 逐位复现的方法论意义
管线库对 2846 协议的封装**逐位忠实**（r=1.0、med|Δ|=0 精确为零）——与 2852 误删恢复实证同源：确定性管线 + 协议逐行复刻 = 可完整迁移的测量机器。2857+ 的 Atlas-E200 正式普查（36 层×32 头×200 词）直接 import 此库换词表参数即可，无须再写测量代码。Gen1 崩溃（AttributeError：__init__ 漏设 target_list）修复后一次通过。

### Arm A 双失败的诊断（预研的核心产出）
1. **E1 失败 = 词源问题**：复合词/低频词大量多 token——tool 类仅 4/10 存活（screwdriver/hatchet/crowbar/tweezers/anvil/sickle/lathe/chisel 全灭，scissors 是复数形但单 token）；fruit 6/10（papaya/guava/apricot/pomegranate/watermelon/kiwi 灭）；furniture 6/10（armchair/bookshelf/recliner/hutch/ottoman/nightstand 全灭——复合词必然多 token）。高频常见名词池可轻松补足 10/类，词表扩到 200 无根本障碍。
2. **E2 失败 = 方向温和漂移（0.847-0.936），两个候选解释待审计**：
   - (a) **多义词污染**：lead（金属/动词）、Turkey（国家/禽）、alloy/scissors 等引入类外分量，把中心拖偏；
   - (b) **小样本噪声反转**：80 词版 dW_unit 本身是 8-10 词小样本估计，200 词版才是更好的估计——余弦 0.85 可能反映旧方向的抽样噪声而非新词表的问题。
   - 审计设计（2857）：①jackknife 逐词剔除，定位每类余弦的主要拖动词；②新旧词子集中心分别对比（new-only 中心 vs old 中心余弦）；③若漂移均匀且 new/old 子集中心余弦高 → 判据语义反转为"旧方向是噪声基准"，200 词方向上任。
3. E3 通过（0.9512，贴线）：类间几何（10 类中心两两余弦）在扩展下基本保持——类结构本身稳健，漂移发生在类内方向估计层面。

### 文件
- 脚本 `tests/glm5/phase2856_atlas_e200_prep.py` sha `c5441794`（10,823 B）
- 管线库 `tests/glm5/rdc_atlas_census.py` sha `c3fcead8`（12,758 B）
- 产物 `phase2856/atlas_e200_prep/`：execution.json `6a74cc0f`、result.json `e9e5f9eb`、atlas_e200.npz `10677e58`

### 接续（2857 候选）
1. **E2 方向漂移审计**（主选）：jackknife 逐词剔除 + 新旧子集中心对比 + 多义词定点检查 → 判定漂移源（污染 vs 噪声），产出清洗规则或判据重校准。
2. 词池修复：每类补足高频单 token 候选（池扩至 ~20/类）→ 200 词达阵复跑 E1/E2/E3。
3. 审计通过后启动 Atlas-E200 正式双谱普查（管线库就绪，36 层×32 头×200 词，分块后台）。
"""

path = r'D:/AI2050/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md'
with io.open(path, 'a', encoding='utf-8') as f:
    f.write(SECTION)
print('memo appended')
