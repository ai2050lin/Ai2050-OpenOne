## Phase 2826: 九方向头级功能图谱 [2026-09-17 07:21]

### 1. 原理与设计

2825 交付了 apple→red 的头级电路与实体条件化真翻转。核心开放问题：**写头硬件是颜色专属还是跨属性域共享？** 本 Phase 把同一批捕获（apple 上下文、全 1152 头实测写出 delta）投影到 9 个方向：8 属性域（2823 词对 size/weight/temperature/speed/hardness/taste/loudness/shape）+ color_red（12 色对比方向），产出 1152×9 头级功能图谱。

配套（2809 纪律）：
- **随机 null 对照**：100 个随机单位方向同 delta 投影 → 池 q95 = 0.0625。
- **跨域行为编辑**：size 域 top-10 头（elephant 上下文 key）rank-1 flip（−2·c·dD，2821 验证过的真翻转算子），测 48 实体 size margin 位移。

预注册判据（零观测前冻结）：
- P1 头显著：9/9 域 top-20 头 mean c > null q95
- P2 晚层质量：9/9 域正写出质量 L≥30 占比 > 0.5
- P3 专业化：9 域 top-20 头集两两 Jaccard 均值 < 0.35
- P4 通用写头：L35 h0 在 9 域全部 c > 0
- P5 size 头编辑：d_margin_size(elephant) < −0.5

门禁：gate_dW = 1.82e-9，gate_Z = 2.46e-6（通过）。

### 2. 结果（一次运行成功，11.5s）

**判决：P1=true, P2=false, P3=true, P4=false, P5=false（2/5）**。

各域 top-20 mean c：taste 0.287 / shape 0.272 / speed 0.221 / temperature 0.214 / weight 0.203 / loudness 0.186 / size 0.182 / hardness 0.174 / color_red 0.367 —— 全部超 null q95（0.0625）3-6 倍。

Jaccard 均值 0.134（36 对，最高 temperature~color_red 仅 0.25，最低 size~weight 0.026）。**头集高度专业化**。

### 3. 关键发现

1. **P1：9 域写出全部真实**（超随机 null 显著）——头谱是域结构不是伪投影。
2. **P3：写头专业化，属性簇泛亲缘**。L29 h27 出现在 weight/hardness/taste/shape/color_red 五域 top（"通用属性头"）；L35 h28 在 temperature/speed/hardness/taste/loudness 五域；L34 h15 在 hardness/taste/shape 三域。物理属性簇共享部分硬件，但 top-20 集合分离。
3. **P4 否证通用写头**：L35 h0 九域谱 = color_red **1.301**、speed 0.61、temperature 0.28、taste 0.22、hardness 0.15、shape 0.08、size 0.04、loudness −0.05、weight **−0.735**。它是**颜色/速度/温度写头 + 强抑重**——专业化且跨域拮抗（与 2820 列拮抗同构，拮抗出现在头级）。
4. **P2 否证末层尖峰**：正写出质量 L≥30 仅占 0.38-0.46，即 **~60% 写出质量在 L22-29 中晚层宽带**。2824 的"L35 聚集"是峰值视角；质量视角写出是宽带。
5. **P5 失败揭示域×通路分工（本 Phase 最大发现）**：size top-10 头 flip 对行为几乎零效果（全实体 |Δ|≤0.188，elephant +0.125），而 2822 列编辑同实体 −1.50 真翻转。合并两 Phase 证据：**颜色 = OV 头通路主导（头编辑 2.7 vs 列 1.75）；大小 = MLP 列通路主导（列 −1.50 vs 头 ~0）**。不同属性域走不同硬件通路。

### 4. 硬伤

1. size 头编辑无效是"判据失败"，但未直接测 size 域的正交 key/差分谱变体（头通路 size 效力小可能因 key 条件化差而非无承载）。
2. 晚层阈值 0.5 是拍脑袋预注册值；宽带结构本身是新认知而非失败。
3. 只测 apple 上下文的谱（域×实体谱矩阵未做）；size 行为编辑只做了 flip 单算子。
4. Jaccard 用 top-20 截断，对阈值敏感。

### 5. 结论

**"具体机制"最终图景（2820→2826 六连）**：LLM 属性知识 = ①每属性域一组专业化写头（Jaccard 0.134，头集近乎不重叠）+ 属性簇共享的通用属性头（L29 h27 型）；②**域×通路分工**——颜色走 OV 头、大小走 MLP 列（双通路证据闭环）；③写出端拮抗跨域存在（L35 h0 抑重）；④写出质量呈 L22-29 中晚层宽带 + L30-35 收束；⑤实体条件化在 key 端（差分谱+正交化 key 可实现实体专属编辑，2825）。参数经济性：少量专业化头/列 × 复用的通用属性头 × 每实体一个条件化 key 方向。

### 6. 接续（Phase 2827 候选）

1. **域×实体谱矩阵**（9 方向 × 48 实体捕获全展开）：检验"通用属性头"（L29 h27）是否实体条件化，即其五域写出是否都依赖实体。
2. size 域头通路复活：差分谱 + 正交化 key（2825 三件套）套用到 size，验证头通路是否被条件化方法救活。
3. 阈值门定位（emb β 细扫 + 逐层 logit-lens）。
4. Δh 通道分离主线（2810 承接）不变。

### 7. 产物登记（immutable + SHA256）

- 脚本 tests/glm5/phase2826_domain_spectrum.py sha256 = 686e2cec878eba2826d46d8c5dedad59079611fb638e8398f90515a05fe2cf6f
- …/phase2826/domain_spectrum/execution.json sha256 = 2ce0a5705b75352e153e3620e2e3b26ee7cd5f972d7d1f2bb4a15c0bee4801fe
- …/result.json sha256 = 261e39da9b35f6c284f1cf64217bd210b20afd73f471899bb4e4eaca8c5e7670
- …/spec9.npz sha256 = 8d1a78582c370d205dd8e345b0800b7bfe8162f4057290dbfc876ba7fd8e4663

**Gens 记录**：1 次运行（11.5s，零崩溃）。预注册判据全程未动；P2/P4/P5 负结果如实入账。
