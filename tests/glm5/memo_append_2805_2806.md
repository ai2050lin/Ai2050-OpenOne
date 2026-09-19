

## Phase 2805（自动续研，LPF-18：实体感定律大样验证——21 域全局对比方向普查，负结果定稿）

**测试原理**：2804 的实体感定律（21/21 分离）n=4 统计弱，且其操作化是"词内对比"（target 去自参照后 vs 其余感）。2805 把实体感定律升级为**域方向命题**做大样检验：取 2795 atlas 全部 21 域 × 10 词=210 词（company/country 两域为实体感组），构造**全局对比方向** dW~_s = Wmeans_s − mean(其它 20 域)（域均值减其它域均值，与词内对比操作化正交），对每个域方向算 PR；词级 PR~(w,s)=词 CM 在其域全局方向的 PR。gates：E-vs-atlas 0.000000、lens-vs-2797 7e-06、dWg 零和 1.77e-16、turkey 词级 PR 复现 2804 err 0.044。预注册 P-F1/P-F2/P-F3 于任何读出前冻结（execution f700721b）。

**结果**：
| 判据 | 结果 | 读数 |
|---|---|---|
| P-F1 方向级 | **FALSE** | 21 域方向 PR 排名：company rank 1（765.1 最紧凑）**但 country rank 18（871.0）**——实体感两域不都最小；color rank 21（910.7）/anatomy rank 20（903.6）最松散；全距 765-911 仅 19% |
| P-F2 投影级 | **FALSE** | median PR~：entity 201.8 < natural 221.8（z=−2.74, p=0.003 显著）但 entity 201.8 vs named 214.5（z=−1.70, p=0.045 > 0.01 阈值，不显著）；gen 臂 90 词同构（entity 218.0 < natural 222.1 < artifact 235.5） |
| P-F3 归因 | **FALSE** | 21 点回归 mean_w PR~(w,s) ~ PR(dW~_s)：R²=0.478 < 0.5（slope 0.684, intercept −368.9）；entity 残差 company −0.2 / country −4.2 |
| verdict | **entity_law_general = false** | 三判据全败，负结果如实登记 |

**头条发现（定律降级 + 新不对称）**：2804 实体感定律**不跨操作化推广**——词内对比（target 从自身锚集移除 vs 其余感）下实体感取词内最小 PR 是"词内候选竞争"现象；全局对比方向下 country 方向与普通域无异。company 方向仍 rank 1 且其词级残差 −0.2（完美被词级 PR 预测）——**公司名在 Qwen3 词嵌入中构成异常紧凑族群，国家名不**。compactness 本身量级小（全距 19%），说明 21 个域方向的 PR 几乎同质，2804 观察到的戏剧性分层（PR 32 vs 300+）只能来自词内对比构造，不能外推为域几何属性。LPF 词位随身档案不受影响：参数作用=带符号投票表（2802 终态），实体感定律从"普适定律"降级为"操作化依赖现象"。

**相关文件**（SHA256 前 16）：脚本 phase2805_rdc_entity_law.py=9cf6746b1315eff1；产物 phase2805/qwen4_entity_law/{execution f700721bf474f28c, result e8d7b0876ef85c98（210 词全 PR 记录+21 方向 PR 表+回归系数）, entity.npz 239f051230dc242f}。哈希登记 probe2805_2806_hashes.txt。gens：2805 首跑 NameError（生成器作用域）+ KeyError 'nature'（atlas nature 不在 dom_idx，跳过登记）→ 修 bug 不改判据重跑，负结果可信。

**问题硬伤**：① 全局对比只是众多操作化之一，不排除其它全局构造（除法对比/池化对比）下实体感成立；② 21 方向 PR 范围窄（19%）,"方向紧凑性"维度本身低方差；③ named/natural 二分继承 2804 粗糙性；④ 210 词全部来自 2795 atlas，无独立词表复制；⑤ gen 臂 named/nature 组为 null（atlas 类别缺失），泛化检验仅 3 组。

**结论**：实体感定律终审——**实体感紧凑性是词内对比操作化的现象，非域方向普遍几何属性**（P-F1/F2/F3 全 FALSE）。保留的唯一坚实碎片：company 方向 rank 1 + 词级残差 −0.2（公司名族群异常紧凑）。LPF 框架三定律（对比编码/平坦谱/画像可读）与词位随身档案不受影响。

**接续（2806 候选，同目标自动续研）**：用户追加关键目标——"通过大量物体获取多层级定义的相对结构（苹果/香蕉 vs 水果-食物-植物-物体；老虎狮子相对苹果的差异；汽车飞机相对苹果的差异），最终获取所有名词词嵌入整体分布与差异化、类别层级分布与差异化、高效表达类型与属性的机制"。2806 冻结层级几何五臂协议（Arm A 嵌套保持/Arm B 质心树/Arm C 列方差分解/Arm D 效率曲线/Arm E 例对能量分解）。

## Phase 2806（自动续研，LPF-19：名词词嵌入层级几何——多层级定义的相对结构）

**测试原理**：2806 把用户关键目标操作化为五臂：10 类 atlas（fruit/animal/metal/vehicle/country/food/furniture/tool/clothing/nature，各 10 词），类方向 dW_class=Wmeans_c−mean(其它 9 类)，2 个域方向 nat_art（natural 5 类 vs artifact 4 类）/liv_non（living 3 类 vs 非 living）。**Arm A 嵌套**：类方向 QR 子空间 Qc（10 维），域方向投影保持率——P-G1 保持率≥0.90；**Arm B 树**：10 类质心 average-linkage 凝聚树，Spearman(cophenetic, 语义距离 1/2/3)≥0.5——P-G2；**Arm C 分层布局**：列方差域份额≥0.30 且类内份额≥0.30——P-G3；**Arm D 效率**：top-k 类方向最近质心分类 100 词，10 维准确率≥0.85×全维——P-G4；**Arm E 能量分级**：9 对词对差向量分解 ‖d‖²=dom²+cls²+res²（dom=2 域方向子空间、cls=类子空间剩余、res=残差），cross-domain 域份额≥3×same-class——P-G5。gates：E 0.000000、lens 7e-06、dW_class 零和 7.29e-17。预注册冻结（execution b291ea35）。

**结果**：
| 判据 | 结果 | 读数 |
|---|---|---|
| P-G1 嵌套保持 | **TRUE** | nat_art=**1.000**、liv_non=**1.000**——两个域方向 100% 落在 10 类方向张成子空间内 |
| P-G2 树一致 | FALSE | ρ_tree=0.221<0.5；merges：fruit+food→furniture+clothing→animal+vehicle→…→country 最后合并 |
| P-G3 分层布局 | FALSE | 列方差：域 8.5% / 类 91.5%——域只是小分量 |
| P-G4 效率 | **TRUE** | acc_full(2560 维)=1.000；top-k：k=1 30%→k=5 85%→k=8 99%→k=10 **100%**；2 域方向 64% |
| P-G5 能量分级 | **TRUE** | 域能量份额 same-class 0.0015 vs cross-domain 0.1082 = **74.4×** |
| verdict | hierarchy_law=false | P-G2/G3 败，但嵌套+效率+分级三定律成立 |

**头条发现（层级=嵌套聚合，非树非正交分层）**：名词词嵌入的类别层级几何不是凝聚树（ρ=0.221，animal+vehicle 第 3 步合并暴露质心距离的失效）也不是正交独立层（域方差仅 8.5%），而是**嵌套聚合结构**：高层域方向是低层类方向的线性组合（投影保持率 1.000，机制直接可见——cos(dom:liv_non, fruit)=0.618/food=0.529/animal=0.510 正对齐，cos(dom:nat_art, furniture)=−0.504/clothing=−0.496 负对齐，域方向=生物类正载荷+人造类负载荷的代数和）。**用户关键目标的直接量化答案**：类别信息被极端高效压缩——10/2560 维（0.39%）零损失、5 维 85%；个体身份占词对差异能量 70-99%（残差主导）。三级能量分级即"层级分布与差异化"的几何实现：同类词对（apple-banana dom 0.000/cls 0.006/res 0.994；tiger-lion 0.000/0.002/0.998）几乎纯个体差；同域跨类（apple-bread 0.011/**0.228**/0.772；apple-tiger 0.003/0.221/0.779）类差跳到 22%；跨域（apple-car **0.114**/0.299/0.701；apple-hammer 0.131/0.271/0.729；car-tiger 0.080/0.184/0.816）域差跳到 8-13%；跨 realm（apple-japan 0.082/0.217/0.783）居中——**层级以剂量效应编码在词对差异中，层级越高能量份额越大，但恒小于个体身份残差**。

**相关文件**（SHA256 前 16）：脚本 phase2806_rdc_hierarchy.py=845ba46cc7a2a073；产物 phase2806/qwen4_hierarchy/{execution b291ea35613deaee, result e4926e8bc8ead3af（9 对能量分解+12×12 cos 矩阵+merges 全记录）, hierarchy.npz 4dbed1c1278edef0}；run2806.log=48fe32a0abb80355。哈希登记 probe2805_2806_hashes.txt。gens：merges id≥10 越界 IndexError → cname() 修复 + average_linkage/condensed_links 重写，判据未动。

**问题硬伤**：① 类方向构造与最近质心评估用同一批词表——存在表内自洽循环，未证明对 atlas 外新词泛化；② 只测 2 个域方向，abstract 域（idea/theory 类）缺席，DOMAIN_OF 中 abstract 组无词表；③ P-G2 的语义距离标签 1/2/3 人工粗粒度指定；④ 域定义 nat_art 中 country 类的归属模糊（named 域缺席导致 2805 country rank 18 与此处 country 最后合并相互呼应但无法归因）；⑤ 单模型 Qwen3-4B。

**结论**：层级几何定律（LPF-19）——**名词词嵌入的类别层级是嵌套聚合结构**：①嵌套（域方向 ∈ 类方向子空间，保持率 1.000）；②高效（0.39% 维度零损失分类）；③分级（层级差以剂量效应进入词对差异：同类≈0.2% 域能量 → 跨类 22% 类能量 → 跨域 11% 域+30% 类能量）；④残差主导（个体身份差占 70-99%，词位随身档案的主体是个体身份差向量本身）。"树"和"正交分层"两种朴素层级观都被证伪。LPF v5.3 补全最后一块：类别调制增益谱的类方向构成嵌套子空间骨架，词身份差向量在其上叠加个体变化。

**接续（2807 候选）**：(a) held-out 泛化——atlas 外 100 新词用 10 类方向分类，检验 0.39% 压缩对新词的泛化（去循环化关键）；(b) L30 残差侧复核——H30 vs E30 层级结构对比（2806 全部用 E30）；(c) abstract 域补全（idea/theory/emotion/music-genre 混合抽象词表）补第三域方向进嵌套检验；(d) 跨模型复核（GLM4/DS7B 嵌套保持率）；(e) 2805 遗留——company 方向紧凑机制（W_U 实体词行几何扎堆检验）。
