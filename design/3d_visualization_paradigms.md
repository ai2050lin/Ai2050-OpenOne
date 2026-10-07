# 3D 空间研究可视化：九领域范式对比与平台 Part① 技术选型

> 用途：三部分平台"可视化研究（3D 神经元空间）"落地 frontend 的技术选型依据。
> 依据：2026-10-05 第二轮 9 领域调研（全部经 GitHub 核实）+ 第一轮（FlyWire/Neuronpedia/Isaac/napari）。
> 关联效果图：`design/mockups/platform_v3_refined.html`（v3）。

## 一、九领域 3D 研究项目对比

| 领域 | 代表项目（许可证） | 3D 空间定义 | 分析能力 | 对本项目的可借鉴点 |
|---|---|---|---|---|
| 3D 重建研究 | `nerfstudio-project/gsplat`（Apache-2.0） | 3D 高斯泼溅场 | CUDA 可微光栅化 | GPU 可微渲染管线 |
| 分子生物学 | `3dmol/3Dmol.js`（BSD）、`arose/ngl`（MIT） | 原子坐标 | 浏览器 WebGL 体数据、残基级选取 | 轻量 WebGL 嵌入式查看器 |
| 计算神经解剖 | **BrainGlobe** 生态（BSD-3） | Allen 脑图谱坐标 | brainrender + cellfinder + brainreg | **atlas 统一坐标系范式** |
| 地理空间 | `CesiumGS/cesium`（Apache-2.0）、`visgl/deck.gl`（MIT） | WGS84 / 3D Tiles | GPU 百万点、体渲染、轨迹 | deck.gl 大规模点云图层 |
| 天文 | **OpenSpace**（MIT，AMNH+NASA） | 行星→宇宙连续尺度 | 体渲染、十亿星表 | **多尺度连续 zoom** |
| 高能物理 | `HSF/phoenix`（Apache-2.0，ATLAS 官方） | 探测器几何 | three.js 事件显示 | **experiment-agnostic 架构** |
| 地质建模 | `gempy-project/gempy`（EUPL） | 隐式地质场 | PyTorch 自动微分 + 贝叶斯不确定性 | 3D 概率建模 |
| 科学可视化底座 | `Kitware/ParaView`（BSD-3） | 任意体数据 | exa 级、Python 管线 | 服务端大数据处理 |
| 点云 / AI 研究 | `isl-org/Open3D`（Apache-2.0）、`Unity-Technologies/ml-agents`（Apache-2.0） | 点云 / 游戏引擎 | PyTorch 算子、RL 环境 | 点云算法库 |

第一轮补充（与本项目最直接相关）：

| 项目 | 许可证 | 要点 |
|---|---|---|
| FlyWire / `google/neuroglancer` | Apache-2.0 | WebGL 体数据分块/LOD/sharded 架构；"AI 分割→3D 人工校对→版本化（CAVE）"闭环 |
| **Neuronpedia** `hijohnnylin/neuronpedia` | MIT | SAE 特征平台，支持 Qwen3-4B；steering 实时干预；已本地部署（D:\AI2050\Neuronpedia） |
| Apple **Embedding Atlas** | MIT | WebGPU 百万点浏览器 60fps 全本地 |
| TensorBoard **Embedding Projector** | Apache-2.0 | PCA/t-SNE/**custom projection**（双词定义语义轴） |
| ai-to-lab-orchestrator | 开源 | dashboard 只读渲染永不执行；SQLite 单一事实源；失败分类学 |

## 二、五个值得直接抄的交互范式

1. **experiment-agnostic（phoenix）**：渲染内核与"事件加载器"分离。平台实现 = 客户端只认统一的 `(layer, coord, activation)` schema，Qwen3-4B / GLM4 / DS7B 只是不同的数据加载器，模型切换不换界面。
2. **custom projection（Projector）**：用户输入两个词/两组 token，实时把高维空间投影到这两个语义方向张成的平面上。对应 P4–P7 的 W↔unembed 对齐读出（G−1=5 维读出位槽），是"绑机制的交互"而非装饰。
3. **atlas 统一坐标系（BrainGlobe）**：所有采集数据先配准到公共 `(layer × d_model)` atlas 空间，再叠加比较。内部响应图谱据此可跨 Phase、跨模型叠加。
4. **多尺度连续 zoom（OpenSpace）**：token → 神经元 → 层 → 模型做成连续缩放，不做离散页面切换。这是"高科技感"的核心体验来源。
5. **AI 产出候选 + 3D 界面人工把关（FlyWire）**：与"AI 自动研发 → 3D 观察 → seal 账本"三段闭环同构；CAVE 版本化 = 四级证据标尺的工程化先例。

## 三、Part① 渲染技术选型（决策）

**结论：three.js 起步 → WebGPU 分阶段演进，数据层 zarr 分块流式。**

| 阶段 | 渲染 | 数据管道 | 理由 |
|---|---|---|---|
| M1（落地 v3） | three.js `InstancedMesh` + Points | NPZ 快照直载（≤50 万点） | 生态最成熟；v3 效果图交互可复用；本机 Qwen3-4B d_model=2560×36 层≈9 万坐标，无压力 |
| M2 | 同上 + LOD | 服务端 NPZ→zarr 分块（neuroglancer precomputed 模式），按视野流式取块 | 支撑 14B/Gemma4 级全场；保持坐标顺序可回查 |
| M3 | WebGPU（three.js TSL 或 Apple Atlas 路线） | 全量 instanced + GPU 侧过滤 | 百万点 60fps（Atlas 已验证）；按需再启动，不预做 |

**不做的事**：不引入 deck.gl/Cesium（地理空间特化过重）；不用 ParaView（桌面级，不符浏览器交付）；不自研渲染引擎。

## 四、v3 效果图 → 实现组件映射

| v3 效果图元素 | 实现方案 | 范式来源 |
|---|---|---|
| 3D 点云（X=神经元/Y=激活/Z=层） | three.js InstancedMesh，色=激活、径=参数范数 | v3 既有设计 |
| 数据源切换 DEMO/NPZ/本地结果 | `sourceLoader` 接口：demo 生成器 / collect.npz 解析 / snapshot API | experiment-agnostic |
| 3D↔2D 联动抽屉 | 点选 → 2D 热力图/轨迹/时序（复用现有 App.jsx 图表组件） | FlyWire 点选校对 |
| W↔unembed 对齐 gauge | custom projection：选双 token → unembed 方向投影 | Projector |
| 全层→单层连续缩放 | 相机连续 zoom + LOD 切换 | OpenSpace |
| 条件齿轮候选高亮 | 高激活点白心高亮 + 查询过滤 | Neuronpedia steering 思路 |

## 五、风险与边界

- **数据规模**：Qwen3-14B 全场采集单样本 ≈ 40 层 × 5120 维 ≈ 20 万坐标，NPZ 直载可行；多样本对比必须走 M2 分块，M1 阶段限制单样本/单层切片。
- **Windows 本机代理**：所有在线数据源加载需走禁代理白名单（本机已知 127.0.0.1:61422 劫持问题）。
- **与 Neuronpedia 的关系**：Neuronpedia（已本地部署）提供 SAE 特征视角，平台 Part① 提供原生坐标视角；两者通过"特征↔坐标"映射表互补，不重复造轮子。
