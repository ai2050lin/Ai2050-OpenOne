import io

MEMO = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs\AGI_GPT5_MEMO.md'

section = '''

## Phase 2872（2026-09-18）：属性轴因果 census —— 双轴机制侧完全分离（33.2 min，28 词 × 36 层全谱）

**设计**：2846 census 机制移植到属性轴——`AttrCensus(AtlasCensus)` 子类仅覆盖 `conds2_for`（same = 同轴其他词按 tid 序确定性选取，极性无关，局限已登记）；10 轴 / 28 单 token 极性词（2870 对表合并：size6/speed3/temp4/age3/weight2/strength2/brightness2/moisture2/height2/fullness2）；cdir = 轴方向（对差方向平均，2870 同款）；null tids SEED=2872。全谱 drops_attr (28,36,32)。钳制校验 max_cr ≤ 0.0192 全通过。

**产物**：`tests/glm5/phase2872_attr_census.py`（sha 2551a824 [execution.json] / 282182af [result.json] / 72eb6982 [attr_census.npz]），目录 `phase2872/attr_census/`。

**判决**：

| 判据 | 观测 | 结果 |
|---|---|---|
| **X1** 跨轴共享前缘 | max 轴重叠 = 5/10（无头进 ≥6 轴 top-32；null p95 = 0） | **false → attr_frontedge_absent** |
| **X2** 机制侧分离 | ρ(attr 谱, 类谱) = **0.055**；ρ(attr 谱, η²) = −0.039 | **true → mechanism_side_separate** |
| **X3** 头群独立性 | \\|attr_top64 ∩ class_top64\\| = 5，超几何 p = **0.280**（不富集） | **independent_populations** |

**描述**：各轴前缘分散于 L0-L34、各自为政（size: L21H2/L2H7…；temp: L0H15/L1H27…）；仅个别轴前缘触及 core2 头（brightness/moisture 含 L13H30）。

### 核心发现（重复三遍）

**双轴从表示到机制完全独立：2870 证 unembed 几何独立（属性×类 cos < 随机基线），2872 证机制空间独立（ρ=0.055、头群不富集 p=0.28）——类别轴有集中前缘（top-64 = 51.6% 负载），属性轴无跨轴共享前缘、由轴特异小头群承载。**

机制图像合并：类 = 范畴级特征（集中前缘 + 三重一致核心 2871）；属性 = 词汇级特征（轴特异分散小群）。二者与 2869 密度门控、2866 词级特异、2863 类均值失败七线互证——**机制空间的组织粒度与特征的语义粒度对齐：粒度越细（词/属性 < 类），机制载体越分散**。

### 阶段小结（2870-2872，双轴关联样例链闭合）

1. 2870：属性轴几何前提（独立 + 再现）
2. 2871：显著头三重一致核心（类轴机制实体化）
3. 2872：双轴机制侧分离（属性轴普查第一版）

**2873 候选**：A 门控融合增长率曲线第三点（属性轴 B_attr 接入 2869 协议——每加一轴的增量可测）；B 属性轴 top-64 头的 OV/角色解剖（attr 版 2862）；C 语法轴探针立项。
'''

with io.open(MEMO, 'r', encoding='utf-8') as f:
    before = f.read()
with io.open(MEMO, 'a', encoding='utf-8') as f:
    f.write(section)
print('APPENDED: %d -> %d lines' % (before.count(chr(10)), (before + section).count(chr(10))))
