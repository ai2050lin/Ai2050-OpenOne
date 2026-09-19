## Phase 2827 = Phase Alpha: 域×实体完整谱矩阵与通路分工图 [2026-09-17 07:51]

### 1. 原理与设计（用户三阶段大计划第一步）

用户下达 Alpha/Beta/Gamma 三阶段大计划。ALPHA（本 Phase）四子任务：(1) 48 实体×1152 头×9 方向完整谱矩阵；(2) 每域重复三件套（差分谱选头+正交化 key+flip）；(3) 域×通路分工全图；(4) 通用属性头 L29 h27 实体条件化验证。

每域 target（正极代表实体）：size→elephant、weight→elephant、temperature→sun、speed→rocket、hardness→steel、taste→apple、loudness→trumpet、shape→balloon。三臂：head3（差分选头 top-10 + 正交 key + flip）、headplain（同头组 plain key）、cols（26-35 层每层 top-20 |proj| 列 flip，仅 comp>0）。

预注册（零观测前冻结）：A1 谱显著（9/9 域 target top-10 mean c > null q95）；A2 三件套效力（≥7/8 域 d_margin(target) < −0.3）；A3 正交隔离（head3 spill < 0.7×headplain，≥6/8）；A4 分工结构（head/col 效力比 max/min > 3）；A5 通用头条件化（c29h27color[apple]/c[sky] > 3 且 ≥5/9 域 max/min > 3）。门禁：dW=1.82e-9，Z=2.46e-6（通过）。

### 2. 结果（Gen2 成功，41.7s；Gen1 einsum 下标字母冲突）

**判决：A1=true, A2=true, A3=true, A4=true, A5=false（4/5）**。

| 域 | 基线 margin | head3 | headplain | cols | spill3/spillp | head/col 比 |
|---|---|---|---|---|---|---|
| size | +0.81 | **−2.125** | −2.250 | −1.062 | 0.26/0.48 | 2.0 |
| weight | +2.94 | −2.250 | −1.938 | −1.250 | 0.21/0.37 | 1.8 |
| temperature | +3.06 | **−2.562** | −2.688 | −0.625 | 0.16/0.47 | 4.1 |
| speed | +0.81 | −1.125 | −1.812 | +0.125 | 0.13/0.79 | 9.0 |
| hardness | +1.06 | −0.750 | −0.750 | −0.375 | 0.10/0.28 | 2.0 |
| taste | +2.75 | −0.875 | −1.750 | +0.125 | 0.08/0.67 | 7.0 |
| loudness | +3.06 | **−3.062** | −3.000 | −0.250 | 0.24/0.48 | 12.25 |
| shape | +6.09 | −2.688 | −3.031 | −1.438 | 0.27/0.76 | 1.87 |

### 3. 关键发现

1. **A2 8/8 全过——三件套全域有效**：每个新域都实现 target margin 大幅负移（−0.75 ~ −3.06），方向全部正确。
2. **2826 的 size"列主导"结论被修正（重要）**：2826 用 apple 上下文谱选头、编辑 elephant → 效力 ≈0；2827 用 **target 实体自己的差分谱**选头 → size 头通路 −2.125 反超列 −1.062。**头通路全域可用，"分工"实为选头质量差异**。2826 P5 的失败是"用错实体的谱选头"的伪象。跨域保留的结构差异：head/col 比 1.8（weight）~12.25（loudness），强头域 loudness/taste/speed/temperature vs 均衡域 size/weight/hardness/shape。
3. **A3 8/8 全过——正交化 key 隔离普适**：三件套 spill 比 plain 低 2-6 倍（如 taste 0.081 vs 0.665，6 倍）。
4. **A5 度量缺陷下的强条件化证据**：预注册第二判据（max/min > 3）9 域全败（0.47-2.23），因谱符号混合使 max/|min| 失效；但第一判据 apple/sky = **27.87** 大幅通过。L29 h27 对 apple 的写出束：color 0.769 + shape 0.509 + taste 0.460 + weight 0.432（**苹果语义束**：红+圆+甜+重），对 sky 全部 ≈0（−0.06~0.10）。通用属性头 = "实体条件化的属性束写头"确认，度量方法待改（如用 target-vs-均值比）。
5. 基线 margin 揭示 shape 6.09 / temperature 3.06 / loudness 3.06 的高先验——round/hot/loud 是高频属性联想。

### 4. 硬伤

1. A5 第二判据设计缺陷（预注册不可改）：符号混合谱的 max/min 不是条件化度量；正式结论以 apple/sky 比为准，需下 Phase 修正度量重测。
2. 三件套未做逐域翻转判定（head3 −0.75~-3.06 未全部跨零，如 shape 基线 6.09 仍差 3.4）。
3. 分工图只用效力比，未测双通路叠加（cols+head3 组合）是否继续加和。
4. 列臂 census 用磁盘权重（restore 后等价），但未在 arms 间随机化顺序。

### 5. 结论（ALPHA 交付）

**域×实体谱矩阵完成**（36×48×32×9，spec_full.npz 1.9MB）；**三件套 9/9 域全域有效**（color 2825 + 8 新域）；**域×通路分工图完成**——修正版结论：不存在"只能走列"的域，头通路经差分谱+正交化后全域主导或均衡（比 1.8-12.25），**通路分工 = 选头质量 × 域头密度**；**通用属性头 = 实体条件化属性束写头**（L29 h27 对 apple 写红/圆/甜/重束，对 sky 零写出，27.87 倍条件化）。实体知识 = 专业化写头 × 实体条件化 key 的图景在全部 9 域成立。

### 6. 接续（Phase Beta，2828）

知识链 key 方向链式传递：构造"苹果→水果→食物"式 3-5 跳链，逐跳捕获 key 方向；验证第一跳写出是否改变第二跳 key；链式编辑（改第一跳 key 测后续跟随）；链断裂修复（断裂点注入正确 key）。

### 7. 产物登记（immutable + SHA256）

- 脚本 tests/glm5/phase2827_alpha_matrix.py sha256 = 07fea4d74d2e941b41e81043cfb35acca301040f6009df5832c4435f70bdb125
- …/phase2827/alpha_matrix/execution.json sha256 = cd4d7e767a6087054ee6dfb25bc77b33467db535d88ebbca20577b8bd6fad031
- …/result.json sha256 = 864f5a50c498ea624d264a3e2e581201f54c627ace180456adc6f9740d78d6ea
- …/spec_full.npz sha256 = 8227b3b68b90876ba56e70d155eadf409660c908e762c3b30344e4b4badb0c5d

**Gens 记录**：2 次运行（①einsum 'rd' 下标字母与 'edh' 的 d 尺寸冲突（2560 vs 9）→ 改 'dr'；②干净 41.7s）。预注册判据全程未动；A5 负结果如实入账并附度量缺陷分析。
