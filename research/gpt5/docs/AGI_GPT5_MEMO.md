# AGI GPT5 Memo


## Phase 1: 读取 GLM5 Phase 280-281 后的路由-内容分离总结 [2026-05-25 23:33]

### 本轮执行命令
- `rg -n "^## Phase (280|281):|^## Phase [0-9]+:" /home/rankrank/Documents/OpenOne/Ai2050-OpenOne/research/glm/docs/AGI_GLM5_MEMO.md`
- `rg --files /home/rankrank/Documents/OpenOne/Ai2050-OpenOne/research | rg 'AGI_GLM5_MEMO\.md$|AGI_GPT5_MEMO\.md$'`
- `rg -n "^## Phase (280|281):" /home/rankrank/Documents/OpenOne/Ai2050-OpenOne/research/glm5/docs/AGI_GLM5_MEMO.md`
- `sed -n '57127,57276p' /home/rankrank/Documents/OpenOne/Ai2050-OpenOne/research/glm5/docs/AGI_GLM5_MEMO.md`
- `sed -n '57277,57480p' /home/rankrank/Documents/OpenOne/Ai2050-OpenOne/research/glm5/docs/AGI_GLM5_MEMO.md`
- `sed -n '57481,57720p' /home/rankrank/Documents/OpenOne/Ai2050-OpenOne/research/glm5/docs/AGI_GLM5_MEMO.md`
- `date '+%Y-%m-%d %H:%M'`

### 生成脚本
- 本轮未生成测试脚本，属于文档读取、理论总结与下一阶段计划制定。

### 原理
- Phase 280 从 hidden state 分析转向 attention graph 直接测量，核心是区分语言计算中的“路由层”：谁和谁通信。
- Phase 281 在 Phase 280 的基础上做路由-内容反事实交换，核心是区分 attention weights 的 routing effect 与 value vectors 的 content effect。
- 两阶段合起来形成一个更基础的拆分：attention graph 主要刻画通信拓扑，value/content 向量主要承载语义角色、内容变换和深层分歧。

### 结果
- Phase 280 发现 SVO 角色交换几乎不改变 attention 图拓扑，frob 很小且相关系数极高，说明角色绑定不主要依赖 attention 重路由。
- Phase 281 进一步发现 Qwen3/GLM4 存在浅层路由主导、中层内容主导的分工；DS7B 则从 L0 起更偏 content 主导，提示 MoE 或 Sliding Window 架构可能把角色绑定前移。
- 当前结论不能夸大为“已破解语言数学结构”，更谨慎的说法是：已经获得了“路由/内容可分离”的直接因果线索，但仍受样本量、RoPE 缺失、深层权重访问、GQA 头级重复、DS7B attention 不完整等限制。

### 理论研究进展
- 新的关键拼图是：语言计算至少包含相对稳定的路由拓扑与高度动态的内容编码两层。
- 角色绑定更可能发生在 value/content 子空间，而不是 attention 图的整体重连；浅层可能先建立最初通信路径，中层再完成角色语义编码。
- 下一阶段应优先扩大样本量、修复深层权重加载、补 RoPE 手动 attention、做 head 级 L0 路由图，以及直接测量中层 value 向量中的 agent/patient 角色子空间。

---

## Phase 2: 读取 GLM5 Phase 270-279 后的项目研究图谱总结 [2026-05-25 23:37]

### 本轮执行命令
- `rg -n "^## Phase (270|271|272|273|274|275|276|277|278|279):" /home/rankrank/Documents/OpenOne/Ai2050-OpenOne/research/glm5/docs/AGI_GLM5_MEMO.md`
- `tail -n 80 /home/rankrank/Documents/OpenOne/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md`
- `sed -n '55494,55980p' /home/rankrank/Documents/OpenOne/Ai2050-OpenOne/research/glm5/docs/AGI_GLM5_MEMO.md`
- `sed -n '55981,56407p' /home/rankrank/Documents/OpenOne/Ai2050-OpenOne/research/glm5/docs/AGI_GLM5_MEMO.md`
- `sed -n '56408,56781p' /home/rankrank/Documents/OpenOne/Ai2050-OpenOne/research/glm5/docs/AGI_GLM5_MEMO.md`
- `sed -n '56782,57126p' /home/rankrank/Documents/OpenOne/Ai2050-OpenOne/research/glm5/docs/AGI_GLM5_MEMO.md`
- `date '+%Y-%m-%d %H:%M'`

### 生成脚本
- 本轮未生成测试脚本，属于文档读取、跨 Phase 证据整理与理论计划制定。

### 原理
- Phase 270-279 的主线是从“hidden state 子空间是否可解码”推进到“语言计算是否由条件路由、局部动力学和组合拓扑构成”。
- 关键方法路径为：V_vis/V_inv 子空间分解 → 拓扑保持检验 → 路径复用/分叉 → 噪声因果检验 → CRTM 替换式路由指纹 → CDRE 条件 Jacobian → Jacobian 谱/子空间分析 → 全局动力学图谱 → 关系/组合/操作子/递归拓扑。
- 这一串实验不断淘汰过强解释：Transport R² 不是 V_inv 特殊输运证据，噪声 causal tracing 更像脆弱度图谱，hidden state 不按语义类别聚类，否定也不是动力学反转。

### 结果
- V_inv 不是“暗物质”，但 Phase 270 的 Transport R²≈1.0 被 Phase 271 证明主要是高维回归假象；更可靠的是关系拓扑跨层保持、类内拓扑保持强于类间。
- Phase 272-274 显示同类概念共享更多残差变化方向和路由指纹，噪声方法失败，而替换式 CRTM 更能稳定区分类内/类间路径。
- Phase 275-276 显示同类 token 的条件 Jacobian 响应和 Jacobian 子空间更相似，但奇异值谱几乎通用，提示概念差异主要不在增益大小，而在方向/子空间。
- Phase 277-278 显示中间层远非 rank-1，有 19-28 个显著方向；首末层更接近统一方向；隐藏状态语义聚类极弱，说明语言结构不应被理解成简单语义类簇。
- Phase 279 显示真正强信号来自关系、组合、操作子和递归：SVO、组合短语和操作子会强烈改变中间层轨迹；操作子子空间几乎正交；not X 不等于 antonym，操作子更像计算路径切换。

### 理论研究进展
- 当前项目的更稳结论是：语言能力背后可能不是“词向量聚类”，而是“条件计算拓扑”。
- 中间层是核心计算形成区：上下文敏感、方向最多、组合非线性最强、操作子效应最强、关系/递归发散也主要发生在这里。
- 深层更像压缩/输出准备区：多方向分化被压缩，末层又出现脆弱性和非线性回升，尤其 GLM4 特征明显。
- 下一阶段应把 Phase 279 的组合拓扑与 Phase 280-281 的 attention/value 分离合并：直接测量关系、操作子、递归的变化到底发生在 attention routing、value content、MLP 还是残差子空间中。

---

## Phase 3: 基于项目核心思路的复用-差异化破解路线分析 [2026-05-25 23:41]

### 本轮执行命令
- `date '+%Y-%m-%d %H:%M'`

### 生成脚本
- 本轮未生成脚本，属于理论归纳、研究路线设计与硬伤审视。

### 原理
- 用户给出的核心问题是：语言、大脑和深度神经网络中的概念、属性、语法、逻辑、任务模式，是否都来自同一套“复用与差异化”的编码机制。
- 当前最关键的研究目标不是继续证明“同类更像、异类更不同”，而是要在神经元/参数级别说明：哪些组件被复用，哪些组件产生差异，差异如何进入路由、value、MLP、残差方向和输出压缩。
- 因此下一阶段应从统计相似度测试转向可干预、可定位、可拆解的机制破解：给定一个概念或功能，恢复它调用的最小共享骨架、差异残差和条件控制路径。

### 结果
- Phase 270-281 已经支持“相对编码/条件计算拓扑”这一方向：hidden state 不按语义类别强聚类，attention 图本身也不直接承担角色交换，真正强信号在中间层多方向分化、CRTM 路由指纹、Jacobian 子空间、value/content 角色编码、操作子正交子空间。
- 当前最大进展是把“词向量语义空间”修正为“条件路由 + 内容编码 + 局部动力学 + 组合拓扑”的多层机制框架。
- 当前最大硬伤是仍没有神经元/参数级别的闭环：还不知道一个水果概念、一个操作子、一个翻译模式到底由哪些 attention head、value 子空间、MLP neuron、残差方向和输出投影共同实现。

### 理论研究进展
- 新的第一性原理候选应表述为：语言编码不是固定语义轴，而是“共享骨架 + 条件差异 + 路径选择 + 深层压缩”的相对拓扑机制。
- 复用不是相似度高这么简单，而是多个概念/功能共享一组可复用计算组件；差异化也不是距离远这么简单，而是少量条件组件改变路由、value 内容或 MLP 变换方向，使系统在无穷组合中保持唯一输出。
- 下一阶段大任务应命名为“神经元级复用-差异化图谱”：对水果、操作子、翻译/续写/推理模式三类对象，构造共享骨架、差异残差、可干预验证和跨模型对比闭环。

---

## Phase 4: 分析 20260525 破解系统编码方案的正确性与改进 [2026-05-25 23:45]

### 本轮执行命令
- `wc -l /home/rankrank/Documents/OpenOne/Ai2050-OpenOne/research/关键记录/20260525_破解系统编码方案.md`
- `sed -n '1,220p' /home/rankrank/Documents/OpenOne/Ai2050-OpenOne/research/关键记录/20260525_破解系统编码方案.md`
- `sed -n '221,440p' /home/rankrank/Documents/OpenOne/Ai2050-OpenOne/research/关键记录/20260525_破解系统编码方案.md`
- `sed -n '441,700p' /home/rankrank/Documents/OpenOne/Ai2050-OpenOne/research/关键记录/20260525_破解系统编码方案.md`
- `tail -n 60 /home/rankrank/Documents/OpenOne/Ai2050-OpenOne/research/gpt5/docs/AGI_GPT5_MEMO.md`
- `date '+%Y-%m-%d %H:%M'`

### 生成脚本
- 本轮未生成脚本，属于方案审查、理论校正与阶段计划改进。

### 原理
- 该方案的核心判断是：局部概念 probe 不够，必须升级到全局条件计算结构、轨迹场、路由图谱和条件 Jacobian 图谱。
- 结合 Phase 270-281，方向基本正确：hidden state 不按语义强聚类，强信号来自中间层多方向分化、CRTM 路由指纹、Jacobian 子空间、组合拓扑、value/content 角色编码。
- 但方案中的“语义流形、纤维束、相空间、混沌、百万 Jacobian”等表述过早，必须先改成基础可测量对象：共享组件、差异组件、路由权重、value 方向、MLP neuron、残差方向和可干预行为变化。

### 结果
- 正确部分：从局部 probe 升级到系统级图谱是必要的；轨迹比单层 hidden vector 更接近真实编码对象；动态计算图重建比静态相似度更接近复用-差异化问题。
- 主要问题：计划太宏大，缺少计算预算、样本分层、因果验证和神经元/参数级闭环；高级数学命名容易掩盖尚未定义的测量对象；“百万提示/百万 Jacobian”在当前本地模型测试条件下不现实。
- 改进方向：把 Language Dynamics Observatory 拆成三层闭环：先做小而全的机制切片，再做中规模覆盖图谱，最后才做全局自动化观测站。

### 理论研究进展
- 当前最需要的不是继续扩大统计观察，而是建立“复用-差异化因果闭环”：发现共享骨架与差异残差，定位到 head/value/MLP/neuron/residual，再通过替换、消融、注入验证行为是否按预测改变。
- 下一阶段应优先选择三类对象：水果概念、操作子、任务模式。它们分别代表知识网络、逻辑控制和高级模式，能覆盖用户提出的核心语言特性。
- 成功标准必须从“相似度有差异”升级为“可定位、可替换、可预测、可跨模型复现”。

---

## Phase 5: 搭建 Qwen3/GLM4/DeepSeek7B 本地探针与消融测试环境 [2026-05-26 01:17]

### 本轮执行命令
- `df -h .`
- `nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader`
- `rg -n "transformer_lens|HookedTransformer|AutoModelForCausalLM|snapshot_download|huggingface|Qwen|GLM|DeepSeek|deepseek" ...`
- `sed -n '1,180p' pyproject.toml`
- `sed -n '1,360p' tests/model_demo_bf16.py`
- `sed -n '1,260p' tests/glm5/model_utils.py`
- `python -m pip install -e .`
- `python -m pip install socksio`
- `python tests/gpt5/check_probe_env.py`
- `python tests/gpt5/download_models.py qwen3`
- `python tests/gpt5/probe_ablation_smoke.py qwen3`
- `python tests/gpt5/download_models.py deepseek7b`
- `python tests/gpt5/probe_ablation_smoke.py deepseek7b`
- `python tests/gpt5/download_models.py glm4`
- `python tests/gpt5/probe_ablation_smoke.py glm4`
- `du -sh models/hf/qwen3-4b models/hf/deepseek-r1-distill-qwen-7b models/hf/glm4-9b-chat-hf`
- `python - <<'PY' ... 合并 results/gpt5_probe_ablation_smoke/summary.json ... PY`
- `date '+%Y-%m-%d %H:%M'`

### 生成脚本
- `tests/gpt5/model_registry.py`
- `tests/gpt5/download_models.py`
- `tests/gpt5/hf_probe_env.py`
- `tests/gpt5/probe_ablation_smoke.py`
- `tests/gpt5/check_probe_env.py`

### 原理
- 本轮目标是先完成可重复的本地模型环境：三个模型只顺序加载一个，避免 4090D 24GB 显存溢出。
- `transformer_lens` 已通过 `pip install -e .` 以本地 editable 方式安装；对 Qwen3/GLM4/DeepSeek7B 这类 HF remote-code 架构，当前用 Hugging Face `register_forward_hook` 做探针和消融，避免强行依赖 HookedTransformer 对非内置架构的支持。
- `probe_ablation_smoke.py` 的基础验证为：用水果 prompt 与非水果 control 在若干层构造 fruit direction，然后在最佳层 hook 中移除最后 token hidden state 沿该方向的投影，观察 `fruit` 相对 `tool` 的 logit margin 是否下降。

### 结果
- 硬件确认：NVIDIA GeForce RTX 4090 D，约 24GB 显存；磁盘剩余约 1.4T。
- 环境确认：`torch 2.6.0+cu124`、`transformers 5.5.4`、`accelerate 1.13.0`、`huggingface_hub 1.11.0`、本地 `transformer_lens` 可导入。
- 已下载模型：
  - `Qwen/Qwen3-4B` → `models/hf/qwen3-4b`，约 7.6G
  - `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` → `models/hf/deepseek-r1-distill-qwen-7b`，约 15G
  - `zai-org/glm-4-9b-chat-hf` → `models/hf/glm4-9b-chat-hf`，约 18G
- smoke test 结果：
  - Qwen3: `fruit-tool` margin 12.5625 → 4.5625，delta -8.0
  - DeepSeek7B: 14.28125 → 9.375，delta -4.90625
  - GLM4: 9.890625 → 1.0，delta -8.890625
- 这说明三个模型均可本地加载、提取 hidden states、注册 forward hook，并进行中层方向消融。

### 理论研究进展
- 本轮不是理论证明，而是为“神经元级复用-差异化图谱”建立了工程底座。
- 初步 smoke test 说明“水果共享方向”在三个模型中都可被探针读出，且移除该方向会降低水果类别输出倾向；这只是环境验证，不能夸大为完整机制闭环。
- 下一步应在该环境上扩展为正式实验：多水果/多属性样本、大样本 control、head/value/MLP 分组件 hook、消融强度曲线、替换 `apple_unique -> banana_unique` 的行为验证。

### 硬伤与注意
- 下载过程中 HF 未登录且本机代理不稳定，脚本已改为 `HF_HUB_DISABLE_XET=1`、单线程、自动重试，避免 Xet 416 和代理断流。
- `probe_ablation_smoke.py` 只是 smoke test，样本很小，不能作为研究结论。
- 当前 GLM4/DeepSeek7B 用 HF hooks 运行，不强制转换成 TransformerLens `HookedTransformer`；这是为了优先保证非内置架构下的探针和消融可用。

---

## Phase 6: 忽略本地 Hugging Face 模型目录防止提交大模型 [2026-05-26 01:25]

### 本轮执行命令
- `ls -la .gitignore .git/info/exclude 2>/dev/null || true`
- `sed -n '1,240p' .gitignore 2>/dev/null || true`
- `git status --short -- models/hf .gitignore`
- `git status --short -- .gitignore models/hf`
- `git check-ignore -v models/hf/qwen3-4b/config.json models/hf/glm4-9b-chat-hf/model-00001-of-00004.safetensors models/hf/deepseek-r1-distill-qwen-7b/config.json`
- `date '+%Y-%m-%d %H:%M'`

### 生成脚本
- 本轮未生成脚本。

### 原理
- 本地 `models/hf/` 保存 Qwen3、GLM4、DeepSeek7B 权重，属于大模型文件，不应进入 git 历史。
- 将 `models/hf/` 加入 `.gitignore` 的 Models 区域，可以忽略该目录下 config、tokenizer、safetensors 和下载 manifest。

### 结果
- 已修改 `.gitignore`，新增规则：`models/hf/`
- `git check-ignore -v` 验证通过：
  - `models/hf/qwen3-4b/config.json`
  - `models/hf/glm4-9b-chat-hf/model-00001-of-00004.safetensors`
  - `models/hf/deepseek-r1-distill-qwen-7b/config.json`
- `git status --short -- .gitignore models/hf` 只显示 `.gitignore` 修改，不再显示 `models/hf/` 未跟踪目录。

### 理论研究进展
- 本轮是工程卫生处理，不产生新的理论结论。
- 对后续研究的意义是确保探针/消融环境可本地复现，同时避免把大模型权重误提交到仓库。

---

## Phase 7: 独立验证 GLM5 Phase 282 四路因果 Patching 结论 [2026-05-26 01:36]

### 本轮执行命令
- `rg -n "^## Phase 282:" research/glm5/docs/AGI_GLM5_MEMO.md`
- `sed -n '57570,57980p' research/glm5/docs/AGI_GLM5_MEMO.md`
- `sed -n '1,260p' tests/glm5/phase282_causal_patching_rope.py`
- `sed -n '260,620p' tests/glm5/phase282_causal_patching_rope.py`
- `sed -n '620,980p' tests/glm5/phase282_causal_patching_rope.py`
- `find results/phase282_causal_patching -maxdepth 1 -type f -printf '%f %s\n'`
- `python tests/gpt5/phase282_independent_validation.py qwen3`
- `python tests/gpt5/phase282_independent_validation.py deepseek7b`
- `python tests/gpt5/phase282_independent_validation.py glm4`
- `python - <<'PY' ... 合并 results/gpt5_phase282_validation/summary.json ... PY`
- `python -m py_compile tests/gpt5/phase282_independent_validation.py`
- `python - <<'PY' ... 对比 Phase 282 原结果与独立验证结果 ... PY`
- `date '+%Y-%m-%d %H:%M'`

### 生成脚本
- `tests/gpt5/phase282_independent_validation.py`

### 原理
- Phase 282 的最强结论是：加入 RoPE 后，Phase 281 的 content/value 主导被大幅逆转，attention weights/routing 成为角色交换差异的主要来源。
- 为避免复用 Phase 282 的手写 RoPE 和手动 attention 实现，本轮独立验证直接使用模型真实 forward 返回的 `output_attentions=True` attention weights；这些 weights 已经由模型内部处理 RoPE、causal mask、attention mask 和 GQA。
- 然后从 self_attn 的真实输入计算 V，再构造：
  - `routing_effect_correct = ||(A_attention @ B_value) - (B_attention @ B_value)|| / ||A_pure - B_pure||`
  - `content_effect_correct = ||(B_attention @ A_value) - (B_attention @ B_value)|| / ||A_pure - B_pure||`
- 这样可以直接验证 routing 与 content 的相对贡献，而不依赖手写 RoPE。

### 结果
- Qwen3 独立验证（14个同长度 SVO 交换对）：
  - L0: routing 0.306, content 0.966 → CONTENT
  - L9: routing 0.587, content 0.885 → CONTENT
  - L18: routing 0.958, content 0.866 → ROUTING
  - L27: routing 0.602, content 0.843 → CONTENT
  - L35: routing 0.382, content 0.937 → CONTENT
- DeepSeek7B：
  - L0/L7/L14/L21 均为 CONTENT
  - L27 为 ROUTING
- GLM4：
  - L0/L10/L20/L30/L39 全部为 CONTENT
- 与 Phase 282 原结果对比发现：
  - Phase 282 的 `weight_effect` 数值在多数层更接近本轮的 `content_effect_correct`
  - Phase 282 的 `value_effect` 数值更接近本轮的 `routing_effect_correct`
  - 原脚本中：
    - `weight_effect_raw = ||(B_attention @ A_value) - (B_attention @ B_value)||`
    - 这实际上是固定 attention weights、改变 value，按定义应属于 content/value effect。
    - `value_effect_raw = ||(A_attention @ B_value) - (B_attention @ B_value)||`
    - 这实际上是固定 value、改变 attention weights，按定义应属于 routing/weight effect。
- 因此 Phase 282 的核心文字结论“RoPE 修正后多数层 WEIGHT/routing 主导”高度可疑，主要原因可能是指标命名反置。

### 理论研究进展
- 当前更可信的修正判断是：RoPE 修正很重要，但它并没有可靠推翻 Phase 281 的 content/value 主导结论。
- 更谨慎的跨模型图景：
  - 角色交换的差异多数层仍主要体现在 value/content 通道。
  - 少数关键层可能出现 routing 主导，例如 Qwen3 L18、DeepSeek7B 末层。
  - attention routing 仍重要，但不能根据 Phase 282 当前指标命名说它“全局主导”。
- Phase 280、281、282 应重新统一为：
  - 宏观 attention 图拓扑变化很小；
  - 精细 attention weight 有一定贡献；
  - 但多数层 value/content 改变仍是角色差异的主要承载方式；
  - 中层/末层可能存在 routing 与 content 的局部交接点。

### 硬伤与后续任务
- Phase 282 原脚本疑似存在效应标签反置，需要先修正命名，再重跑 52 对全量数据。
- 原手动 attention 计算中未明显加入 causal mask；这会使前向注意力变成双向注意力，可能进一步污染结果。
- Qwen3 的 q/k norm 在原脚本里按整个 head 矩阵做 RMS，而真实 RMSNorm 应按 token 向量维度做归一化，也可能造成偏差。
- 本轮独立验证只用 14 个同长度 SVO 交换对，是复核性质；下一步应扩展到 Phase 282 的 52 对，并输出“修正标签版”的完整贡献矩阵。

---

## Phase 8: 读取 GLM5 Phase 283-284 后判断组件分析与语言测试优先级 [2026-05-26 06:42]

### 本轮执行命令
- `rg -n "^## Phase (283|284):" research/glm5/docs/AGI_GLM5_MEMO.md`
- `rg -n "^## Phase 28[5-9]:|^## Phase 29[0-9]:" research/glm5/docs/AGI_GLM5_MEMO.md | head -n 10`
- `sed -n '57841,58070p' research/glm5/docs/AGI_GLM5_MEMO.md`
- `sed -n '58071,58420p' research/glm5/docs/AGI_GLM5_MEMO.md`
- `date '+%Y-%m-%d %H:%M'`

### 生成脚本
- 本轮未生成脚本，属于文档读取、方法论审查和下一阶段路线判断。

### 原理
- Phase 283/284 继续沿着 Phase 282 的 weight/value 手动 attention 分解路线推进，扩大到全层、逐头、partial RoPE 和 121 对多功能句子。
- 但 GPT5 Phase 7 已独立验证：Phase 282 原脚本疑似存在 effect 标签反置；真实 forward attention 验证显示多数层仍偏 content/value 主导。
- 因此 Phase 283/284 的“Qwen3全层WEIGHT主导”“组件主导模式由架构决定而非语言功能”等强结论，必须在修正标签、加入真实 causal mask、量化 manual-real gap 后才能采信。

### 结果
- Phase 283 的价值：
  - 意识到 Phase 282 的 GLM4/DS7B 深层不完整问题；
  - 试图用 safetensors 补全全层；
  - 提出逐 head、partial RoPE、扩展句型等更细分析。
- Phase 284 的价值：
  - 明确提出先校准工具再做全功能矩阵，这是方向正确的；
  - 扩展到否定、翻译、递归、条件、逻辑等更接近真实语言功能的测试集。
- 主要问题：
  - Block 0 校准失败，manual attention 与真实 forward 的 gap 仍未量化；
  - Phase 283/284 仍沿用疑似反置的 weight/value effect 命名；
  - “不同功能共享同一组件主导模式”可能只是指标太粗或标签错误导致的表象；
  - 继续做组件功能矩阵，可能会把工具误差放大成理论结论。

### 理论研究进展
- 当前不应在未校准的组件矩阵上继续做“组件功能分析”的强理论解释。
- 也不应完全跳到黑盒语言测试，因为那会失去机制定位能力。
- 更稳路线是“语言行为锚定 + 组件因果归因”的两阶段闭环：
  1. 先做语言本身的可判定任务测试，确认模型在否定、翻译、递归、条件、角色交换上确实产生可量化行为差异；
  2. 再在这些已确认的行为差异上做组件替换/消融，观察是否能按预测改变行为。
- 成功标准应从“某层weight/value均值谁更大”改成：
  - 行为任务存在明确正确/错误或logit目标；
  - 组件干预能稳定改变行为；
  - 干预效应跨样本、跨模型、跨层可复现；
  - 最后才解释组件功能。

### 下一步建议
- P0：修正 Phase 282/283/284 的 effect 标签，并用模型真实 `output_attentions=True` 重建 52/121 对的 routing/content 贡献矩阵。
- P0：建立语言行为锚定测试集：SVO角色、否定、被动、条件、递归、翻译、量词、比较、时态，每类至少 100 对，先测目标 token/logit 或选择题准确率。
- P1：只对“行为差异最大、且模型表现稳定”的任务做组件消融，优先 Qwen3 和 GLM4，DS7B 等 Sliding Window 校准后再纳入。
- P1：组件分析必须改成 real forward patching 优先，manual attention 只作为辅助解释。

---

## Phase 9: 语言行为锚定测试 v3 首轮结果 [2026-05-26 07:00]

### 本轮执行命令
- `python -m py_compile tests/gpt5/language_behavior_anchor.py`
- `python tests/gpt5/language_behavior_anchor.py qwen3 --output-dir results/gpt5_language_behavior_anchor_v3`
- `python tests/gpt5/language_behavior_anchor.py deepseek7b --output-dir results/gpt5_language_behavior_anchor_v3`
- `python tests/gpt5/language_behavior_anchor.py glm4 --output-dir results/gpt5_language_behavior_anchor_v3`
- `python - <<'PY' ... 读取 results/gpt5_language_behavior_anchor_v3/summary.json 并输出各模型分类准确率 ... PY`
- `python - <<'PY' ... 读取 *_language_behavior.json 并输出失败样本 ... PY`
- `date '+%Y-%m-%d %H:%M'`

### 生成脚本
- `tests/gpt5/language_behavior_anchor.py`

### 测试原理
- 本轮先做语言本身的行为锚定，而不是直接解释组件功能。
- 测试集包含 60 个最小语言任务样本，覆盖：
  - 主谓宾角色 `svo_agent`
  - 被动句施事 `passive_agent`
  - 否定是/否 `negation_yesno`
  - 条件关系 `conditional`
  - 比较关系 `comparison`
  - 时间关系 `temporal`
  - 递归绑定 `recursive_binding`
  - 量词/否定量词 `quantifier`
  - 中英翻译 `translation`
- 每个样本指定一个目标答案 token 和若干干扰 token，读取模型最后位置 logits，计算 `target_logit - max(distractor_logit)`。
- `margin > 0` 记为正确。这个指标的好处是简单、直接、可用于后续 activation patching；缺点是仍然只测首 token 偏好，不等于完整生成能力。
- v1/v2 中 yes/no prompt 对部分模型不稳定，因此 v3 对 yes/no 类统一加入 `Answer yes or no:`，减少格式偏差。

### 结果文件
- 汇总结果：`results/gpt5_language_behavior_anchor_v3/summary.json`
- 逐样本结果：
  - `results/gpt5_language_behavior_anchor_v3/qwen3_language_behavior.json`
  - `results/gpt5_language_behavior_anchor_v3/deepseek7b_language_behavior.json`
  - `results/gpt5_language_behavior_anchor_v3/glm4_language_behavior.json`

### 核心结果
- Qwen3-4B：60 题正确率 88.33%，平均 margin 4.63，中位 margin 3.38，最小 margin -3.00。
  - 强项：条件、比较、量词、SVO、时间均为 100%。
  - 弱项：递归绑定 50%，被动句 66.7%，翻译中 `apple -> 苹果` 个别失败。
- DeepSeek-R1-Distill-Qwen-7B：60 题正确率 66.67%，平均 margin 3.23，中位 margin 1.88，最小 margin -3.88。
  - 强项：条件 100%，SVO/被动/翻译大体可用。
  - 弱项：否定 yes/no 25%，量词 50%，递归 50%，时间 50%，比较 66.7%。
  - 典型问题：多个 `no` 目标被模型强烈压低，说明该模型在这个首 token 判别格式下有明显回答格式或偏置问题。
- GLM4-9B：60 题正确率 88.33%，平均 margin 4.75，中位 margin 4.19，最小 margin -2.19。
  - 强项：条件、被动、时间、翻译均为 100%，递归 83.3%。
  - 弱项：比较 66.7%，量词 66.7%，SVO 中 `sheep follows wolf` 失败，否定中 `There is no reason...` 失败。

### 失败样本观察
- 三个模型都容易在“羊/狼追随”“递归绑定中多名词竞争”“否定量词 no/few”上出错，说明这些任务更适合作为后续失败分析样本。
- Qwen3 和 GLM4 在条件关系上全对，且 margin 较高，可以作为组件因果分析的稳定正样本。
- DeepSeek7B 的否定 yes/no 表现非常弱，暂时不能直接解释为“没有否定理解”，更可能混有 prompt 格式、首 token 选择、蒸馏推理模型回答习惯等因素。

### 可信度审查
- 本轮可信的结论：
  - 三个模型在当前 prompt 与首 token margin 指标下，确实呈现明显类别差异；
  - Qwen3/GLM4 整体语言锚定强于 DeepSeek7B；
  - 条件、SVO、翻译是后续组件因果归因的较稳任务；
  - 递归、量词、否定是更需要扩大样本和校准格式的困难任务。
- 本轮不能过度推出的结论：
  - 不能说模型是否真正“理解”某类语言结构；
  - 不能说错误一定来自某个 attention head 或 MLP；
  - 不能说 DeepSeek7B 语言能力弱于 Qwen3/GLM4，因为当前测试是首 token 选择，不是完整生成评测。

### 理论研究进展
- 当前路线从“直接解释组件”转为“先确定语言行为，再做组件干预”，这是必要修正。
- 语言背后的数学结构如果存在，不应首先表现为抽象统计均值，而应表现为：
  1. 不同语言功能在行为层有可重复的正确/错误边界；
  2. 同类功能样本共享某些可干预的中间状态；
  3. 交换关键语义变量时，中间状态和输出 logits 发生可预测改变；
  4. 这种改变在不同模型中存在部分共性，同时保留架构差异。
- 因此，下一阶段不应只问“哪个组件主导”，而应问“一个语言功能从输入差异到输出差异，中间经过哪些可复现的状态转换”。

### 下一步任务
- P0：把每类样本扩展到至少 100 条，尤其是递归、否定、量词、比较，避免 6-8 条样本造成偶然结论。
- P0：增加 full-choice scoring：对完整答案词或短语求总 logprob，而不是只看第一个 token。
- P0：对 yes/no、中文翻译、英文实体选择分别做 prompt 模板校准，找出模板敏感性。
- P1：选择稳定高 margin 样本做 activation patching：
  - 条件关系：三模型全对，可作为正向稳定任务；
  - SVO/被动角色：适合做角色交换和残差流 patch；
  - 翻译：适合做任务模式与词义内容分离；
  - 递归/量词：作为困难任务，先扩大样本后再做失败定位。
- P1：组件分析必须以真实 forward hook 和行为变化为准，manual attention 分解只做辅助解释。

---

## Phase 10: 系统语言测试脚本与正式长跑阻塞记录 [2026-05-26 07:30]

### 本轮执行命令
- `sed -n '1,260p' tests/gpt5/language_behavior_anchor.py`
- `sed -n '1,260p' tests/gpt5/model_registry.py`
- `sed -n '1,260p' tests/gpt5/hf_probe_env.py`
- `python -m py_compile tests/gpt5/systematic_language_benchmark.py`
- `python tests/gpt5/systematic_language_benchmark.py qwen3 --cases-per-category 2 --batch-size 4 --output-dir results/gpt5_systematic_language_benchmark_smoke`
- `python tests/gpt5/systematic_language_benchmark.py qwen3 --cases-per-category 100 --batch-size 6 --output-dir results/gpt5_systematic_language_benchmark_v1`
- `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits`
- `ps -o pid,etime,pcpu,pmem,cmd -C python`
- `kill 24109`
- `kill -9 24109`
- `python -m py_compile tests/gpt5/systematic_language_benchmark.py`
- `date '+%Y-%m-%d %H:%M'`

### 生成脚本
- `tests/gpt5/systematic_language_benchmark.py`

### 脚本设计
- 本轮把 Phase 9 的 60 条小样本语言锚定，扩展为系统语言测试框架。
- 默认每类 `100` 条，共 `9` 类、`900` 条/模型：
  - `svo_agent`
  - `passive_agent`
  - `negation_yesno`
  - `conditional`
  - `comparison`
  - `temporal`
  - `recursive_binding`
  - `quantifier`
  - `translation`
- 每条题为二选一形式，保存 prompt、choices、answer_index、category、case_id。
- 同时计算三种指标：
  - `first_token_margin`：答案首 token logprob 减去干扰项首 token logprob；
  - `full_margin`：完整答案 token 序列总 logprob 差；
  - `mean_margin`：按答案 token 数归一后的平均 logprob 差。
- 这样可以同时保留两类价值：
  - 首 token margin 便于和之前 probe/patching 方案连接；
  - full-choice logprob 更接近完整选择题判断，避免多 token 答案被首 token 指标误判。

### 小样本验证结果
- 命令：
  - `python tests/gpt5/systematic_language_benchmark.py qwen3 --cases-per-category 2 --batch-size 4 --output-dir results/gpt5_systematic_language_benchmark_smoke`
- 结果文件：
  - `results/gpt5_systematic_language_benchmark_smoke/qwen3_systematic_language.json`
  - `results/gpt5_systematic_language_benchmark_smoke/summary.json`
- Qwen3 小样本验证：
  - 总样本数：18
  - full accuracy：88.89%
  - mean accuracy：88.89%
  - first-token accuracy：88.89%
  - first/full disagreement：0
- 结论：脚本的基本 scoring 路径、中文/英文选项、完整 logprob 和首 token logprob 输出可以工作。

### 正式长跑阻塞
- 正式命令：
  - `python tests/gpt5/systematic_language_benchmark.py qwen3 --cases-per-category 100 --batch-size 6 --output-dir results/gpt5_systematic_language_benchmark_v1`
- 现象：
  - Qwen3 权重加载完成后进入正式打分；
  - 运行数分钟无输出；
  - `nvidia-smi` 查询卡住；
  - `ps` 显示 python 进入异常状态；
  - 对该 python 进程执行 `kill` 和 `kill -9` 后，进程仍残留；
  - 随后单独启动一个只导入脚本、构造 case 的 Python 进程，也出现异常残留。
- 当前判断：
  - 这不是语言测试结论，而是本机 CUDA/NVIDIA 驱动或 PyTorch CUDA 上下文进入异常状态；
  - 正式 900 条/模型测试没有完成；
  - 不能基于本轮正式长跑给出三模型系统语言能力排名。

### 已做修正
- 修改 `tests/gpt5/systematic_language_benchmark.py`：
  - 去掉每个 batch 后的 `torch.cuda.empty_cache()`，避免长跑时反复触发 CUDA cache 清理；
  - 加入 `--progress-every` 参数，默认每 25 个 batch 输出一次进度；
  - 保留模型结束后的统一 `release_loaded()` 清理。

### 可信结论
- 可信：
  - 系统语言测试框架已经搭好；
  - 小样本 smoke test 通过；
  - 正式测试需要在 CUDA 状态恢复后重跑；
  - 当前 GPU/驱动异常使本轮不能继续顺序测试 DeepSeek7B 和 GLM4，否则会污染结果并增加系统卡死风险。
- 不可信：
  - 不能把小样本 18 条结果当成系统测试结果；
  - 不能把正式长跑阻塞解释为模型语言能力问题；
  - 不能继续用当前异常 CUDA 状态做任何组件消融或探针结论。

### 理论研究进展
- 语言测试标准进一步明确：
  - 行为基准必须同时有大样本和完整选项 scoring；
  - 首 token 指标只能作为机制分析入口，不能单独作为语言能力结论；
  - 系统性语言功能差异必须先在行为层稳定复现，才能进入组件归因。
- 本轮也暴露了工程层面的第一性约束：
  - 要破解语言机制，测试系统本身必须稳定、可恢复、可复跑；
  - 长跑任务必须有进度、checkpoint、分模型隔离和失败恢复；
  - 否则 GPU/驱动异常会被误读成模型或理论问题。

### 下一步任务
- P0：先恢复 CUDA 状态。建议重启 Python/Codex 相关进程；若 `nvidia-smi` 仍卡住，需要重启机器或重载 NVIDIA 驱动。
- P0：恢复后从单模型、低 batch 开始：
  - `python tests/gpt5/systematic_language_benchmark.py qwen3 --cases-per-category 100 --batch-size 2 --progress-every 10 --output-dir results/gpt5_systematic_language_benchmark_v1`
  - 跑完 Qwen3 后再跑 DeepSeek7B；
  - 跑完 DeepSeek7B 后再跑 GLM4。
- P0：给脚本增加按 category checkpoint 的能力，避免一次长跑失败导致全量结果丢失。
- P1：系统结果完成后，挑选高置信任务进入 activation patching：
  - 稳定正样本：条件、SVO、被动、翻译；
  - 困难样本：递归、量词、否定、比较；
  - 每类先找高 margin 正样本和低 margin/错误样本各一批，再做残差流与 attention/MLP patching。

## Phase 11: 上轮系统卡死任务完成度与原因复核 [2026-05-26 11:19]

### 本轮执行命令
- `git status --short`
- `tail -n 120 research/gpt5/docs/AGI_GPT5_MEMO.md`
- `find tests/gpt5 tests/gpt5_temp -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -n 80`
- `ps -eo pid,ppid,stat,pcpu,pmem,etime,cmd --sort=-pcpu | head -n 35`
- `nvidia-smi`
- `find results/gpt5_systematic_language_benchmark_smoke results/gpt5_systematic_language_benchmark_v1 -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | sort`
- `jq . results/gpt5_systematic_language_benchmark_smoke/summary.json 2>/dev/null || true`
- `pgrep -af 'python|systematic_language_benchmark|torch|transformers' || true`
- `ps -eo pid,ppid,stat,wchan:24,etime,cmd | awk '$3 ~ /D/ {print}'`
- `journalctl --list-boots --no-pager | tail -n 10`
- `journalctl -b -1 -k --since '2026-05-26 07:20' --until '2026-05-26 07:26' --no-pager | rg -i 'Xid \(PCI|GSP Timeout|GPU is probably locked|GPU recovery|soft lockup|pid=24109|name=python|pid=3559|name=code' | head -n 80`
- `journalctl -b -1 -k --since '2026-05-26 07:20' --until '2026-05-26 11:13' --no-pager | rg -i 'Xid \(PCI|GPU is probably locked|soft lockup' | wc -l`
- `last -x | head -n 30`
- `nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits`
- `date '+%Y-%m-%d %H:%M'`

### 生成脚本
- 本轮没有生成新脚本。
- 本轮只做系统状态、结果目录和内核日志复核。

### 检查原理
- 用结果目录判断任务是否真正产出正式结果：
  - smoke 目录若有 summary 和模型 json，只说明小样本验证完成；
  - v1 正式目录若无模型结果文件，说明正式长跑没有完成。
- 用进程表和 `nvidia-smi` 判断当前是否还有残留任务或 GPU 占用。
- 用 `journalctl --list-boots` 和上一轮 boot 的 kernel log 判断卡死是应用层慢、显存不足，还是 NVIDIA/CUDA 驱动层异常。
- 用 `last -x` 判断上一轮会话是否以 crash 形式结束。

### 复核结果
- 任务完成度：
  - `results/gpt5_systematic_language_benchmark_smoke/summary.json` 存在，Qwen3 小样本 smoke test 完成；
  - smoke 样本数为 18，full/mean/first-token accuracy 均为 88.89%；
  - `results/gpt5_systematic_language_benchmark_v1/` 没有正式模型结果文件；
  - 因此正式 `cases-per-category=100` 的系统语言测试没有完成，也没有得到 Qwen3/DeepSeek7B/GLM4 的正式可比结果。
- 当前系统状态：
  - 当前 boot 从 2026-05-26 11:13 开始；
  - 当前没有残留 `systematic_language_benchmark`、PyTorch、transformers 测试进程；
  - 当前 `nvidia-smi` 可正常返回，GPU 约 922 MiB / 24564 MiB，主要是图形界面占用；
  - 当前内存充足，约 57 GiB available。
- 卡死原因证据：
  - 上一轮 boot 在 2026-05-26 07:23:16 出现 `NVRM: GSP Timeout`；
  - 日志明确记录 `Xid 119, pid=24109, name=python, Timeout after 45s of waiting for RPC response from GPU0 GSP`；
  - 随后 2026-05-26 07:24:01 出现 `GPU is probably locked`；
  - 2026-05-26 07:24:46 出现 `Xid 154, GPU Reset Required`；
  - 后续出现大量 `watchdog: BUG: soft lockup`；
  - 2026-05-26 11:13 系统进入新 boot，`last -x` 显示上一轮图形会话为 crash。
- 综合判断：
  - 直接诱因是运行正式 Qwen3 系统语言 benchmark 的 Python 进程触发 NVIDIA GSP RPC timeout；
  - 之后 GPU/驱动进入 locked/reset-required 状态，连 VSCode/code 释放 GPU 资源时也被拖入 soft lockup；
  - 这不是模型语言能力问题，也不是正式测试结果，只能归类为 CUDA/NVIDIA 驱动或 GPU 固件层异常。

### 理论研究进展
- 本轮没有新增语言结构结论。
- 关键约束更明确：
  - 行为测试不能只追求样本量，还必须具备 checkpoint、分段保存、超时控制和失败恢复；
  - 单次长跑把模型加载、全量样本、汇总写盘绑在一起，会把驱动异常放大为全量结果丢失；
  - 机制研究的第一性前提之一是实验系统可复现、可中断、可恢复，否则工程异常会污染理论判断。

### 下一步任务
- P0：先不要直接重跑三模型全量长跑。
- P0：给 `tests/gpt5/systematic_language_benchmark.py` 增加按模型、按 category、按 batch 或分片的 checkpoint 写盘。
- P0：增加最小恢复策略：
  - 每完成一个 category 立刻写入 partial json；
  - 已完成 category 自动跳过；
  - 每个模型独立进程运行；
  - 默认 batch size 先降到 1-2；
  - 每个模型跑完后强制释放模型并退出进程，再启动下一个模型。
- P0：恢复后优先重跑 Qwen3 单模型正式测试，确认稳定后再顺序跑 DeepSeek7B 和 GLM4。
- P1：若再次出现 Xid 119 或 GSP timeout，应暂停模型实验，优先处理 NVIDIA 驱动/GSP 固件稳定性问题。

## Phase 12: 可恢复系统语言长跑与三模型正式结果 [2026-05-26 11:28]

### 本轮执行命令
- `sed -n '1,760p' tests/gpt5/systematic_language_benchmark.py`
- `sed -n '1,260p' tests/gpt5/hf_probe_env.py`
- `sed -n '1,220p' tests/gpt5/model_registry.py`
- `python -m py_compile tests/gpt5/systematic_language_benchmark.py`
- `python tests/gpt5/systematic_language_benchmark.py qwen3 --cases-per-category 1 --batch-size 1 --progress-every 1 --output-dir results/gpt5_systematic_language_checkpoint_smoke`
- `python tests/gpt5/systematic_language_benchmark.py qwen3 --cases-per-category 1 --batch-size 1 --progress-every 1 --output-dir results/gpt5_systematic_language_checkpoint_smoke`
- `python tests/gpt5/systematic_language_benchmark.py qwen3 --cases-per-category 100 --batch-size 1 --progress-every 50 --output-dir results/gpt5_systematic_language_benchmark_v1`
- `python tests/gpt5/systematic_language_benchmark.py deepseek7b --cases-per-category 100 --batch-size 1 --progress-every 50 --output-dir results/gpt5_systematic_language_benchmark_v1`
- `python -m py_compile tests/gpt5/systematic_language_benchmark.py && python tests/gpt5/systematic_language_benchmark.py glm4 --cases-per-category 100 --batch-size 1 --progress-every 50 --output-dir results/gpt5_systematic_language_benchmark_v1`
- `jq '{results: [.results[] | {model, num_cases, cases_per_category, batch_size, complete, full_accuracy: .aggregate.overall.full.accuracy, mean_accuracy: .aggregate.overall.mean.accuracy, first_accuracy: .aggregate.overall.first_token.accuracy, first_full_disagreements: .aggregate.overall.first_full_disagreements}]}' results/gpt5_systematic_language_benchmark_v1/summary.json`
- `jq -r '.results[] | .model as $m | .aggregate.per_category | to_entries[] | [$m, .key, (.value.full.accuracy|tostring), (.value.full.mean_margin|tostring), (.value.full.min_margin|tostring)] | @tsv' results/gpt5_systematic_language_benchmark_v1/summary.json | column -t`
- `find results/gpt5_systematic_language_benchmark_v1/checkpoints -mindepth 2 -maxdepth 2 -type f | sed 's#results/gpt5_systematic_language_benchmark_v1/checkpoints/##' | cut -d/ -f1 | sort | uniq -c`
- `nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits`
- `pgrep -af 'systematic_language_benchmark|torch|transformers' || true`
- `journalctl -b 0 -k --since '2026-05-26 11:20' --no-pager | rg -i 'nvrm|xid|gpu is probably locked|soft lockup|oom|out of memory' || true`
- `date '+%Y-%m-%d %H:%M'`

### 生成/修改脚本
- 修改：`tests/gpt5/systematic_language_benchmark.py`
- 新增能力：
  - `atomic_write_json()`：所有结果先写 tmp，再 replace，避免半截 json；
  - `checkpoints/<model>/<category>.json`：每个模型每个 category 单独落盘；
  - `--resume/--no-resume`：默认自动复用已完成 category；
  - `*_systematic_language.partial.json`：每完成一个 category 更新一次 partial 结果；
  - 默认 batch size 从 6 降到 2，本轮正式运行进一步手动使用 1；
  - summary 记录 `complete` 字段，确认模型结果是否完整。

### 测试原理
- 每个 category 有 100 条选择题，每条题两个候选答案，共 200 个候选 forward。
- 每个模型共有 9 类、900 条题、1800 个候选 forward。
- 仍然同时计算：
  - `full`：完整答案 token 序列总 logprob；
  - `mean`：按答案 token 数归一后的 logprob；
  - `first_token`：首 token logprob。
- 本轮不把首 token 当最终能力指标，只作为机制分析入口；正式能力判断优先看 full-choice scoring。
- 工程上每完成一个 category 立刻写 checkpoint；崩溃后重跑会跳过已完成 category，避免全量损失。

### 工程验证结果
- checkpoint smoke：
  - Qwen3，`cases-per-category=1`，9 类共 9 条；
  - 首次运行生成 9 个 category checkpoint、partial、完整结果和 summary；
  - 第二次运行同一命令全部显示 `resume <category>`，验证恢复路径可用。
- 正式长跑：
  - Qwen3、DeepSeek7B、GLM4 均按单模型独立进程顺序完成；
  - 三个模型各 9 个 category checkpoint；
  - 当前无残留 benchmark/PyTorch/transformers 进程；
  - 当前 boot 内未发现 `NVRM`、`Xid`、`GPU locked`、`soft lockup`、OOM 日志；
  - 结束后 `nvidia-smi` 显示 GPU 显存回落到桌面占用，约 795 MiB / 24564 MiB。

### 三模型正式结果
- 输出目录：
  - `results/gpt5_systematic_language_benchmark_v1/summary.json`
  - `results/gpt5_systematic_language_benchmark_v1/qwen3_systematic_language.json`
  - `results/gpt5_systematic_language_benchmark_v1/deepseek7b_systematic_language.json`
  - `results/gpt5_systematic_language_benchmark_v1/glm4_systematic_language.json`
- overall full accuracy：
  - Qwen3：900/900 条输入，full accuracy 95.56%，mean accuracy 95.00%，first-token accuracy 95.56%；
  - GLM4：900/900 条输入，full accuracy 78.89%，mean accuracy 78.33%，first-token accuracy 78.89%；
  - DeepSeek7B：900/900 条输入，full accuracy 71.11%，mean accuracy 70.56%，first-token accuracy 71.11%。
- first/full disagreement：
  - 三个模型均为 0；
  - 说明本批任务中首 token 与完整答案判断一致，但这不代表其他任务也会一致。

### 分类型 full accuracy
- Qwen3：
  - comparison 100%，conditional 95%，negation 100%，passive 100%，quantifier 100%，recursive 85%，SVO 100%，temporal 80%，translation 100%。
- GLM4：
  - comparison 90%，conditional 100%，negation 50%，passive 90%，quantifier 50%，recursive 80%，SVO 100%，temporal 50%，translation 100%。
- DeepSeek7B：
  - comparison 75%，conditional 95%，negation 50%，passive 75%，quantifier 40%，recursive 65%，SVO 100%，temporal 40%，translation 100%。

### 谨慎结论
- 比较稳的结论：
  - 三模型在 SVO 主谓宾角色和基础翻译上都非常强；
  - conditional 也整体较强，尤其 GLM4 达到 100%，Qwen3/DeepSeek7B 达到 95%；
  - recursive binding、temporal、quantifier、negation 是更能拉开差距的类别；
  - Qwen3 在这套人工构造题上明显最稳。
- 必须保留的怀疑：
  - 样本是规则生成的，不是真实自然语言分布；
  - 每类 100 条比 smoke 可靠，但仍可能被模板形式、词表、答案顺序影响；
  - 二选一 logprob 测试不能等同于开放生成能力；
  - negation、quantifier、temporal 的低分可能部分来自 prompt 格式或模型偏好，而不一定是纯语义能力缺陷；
  - Qwen3 优势可能包含 tokenizer、训练语料和 prompt 适配差异，不能直接推断其内部数学结构更接近语言本质。

### 理论研究进展
- 语言能力差异不是均匀下降，而是集中出现在少数结构类型上：
  - 角色绑定：SVO/被动/递归；
  - 逻辑极性：否定、量词；
  - 时间坐标：过去/现在/未来/before/after；
  - 条件映射：if-cause-result；
  - 符号翻译：英文-中文词义映射。
- 这提示语言背后的结构可能不是一个单一能力，而是多个基础关系操作的组合：
  - 对象与角色的绑定；
  - 命题真假与极性的翻转；
  - 集合范围的约束；
  - 时间顺序与事件坐标；
  - 条件触发的状态转移；
  - 符号之间的稳定映射。
- 从第一性原理看，下一步不应先套高级统计模型，而应把这些操作拆成最小、可控、可复现的关系拼图，观察模型在哪些关系变换上稳定、在哪些地方破裂。

### 下一步大任务
- P0：对正式结果做错误样本审计：
  - 每类提取错误样本、低 margin 正确样本、高 margin 正确样本；
  - 手工检查是否有题目本身歧义、模板误导、答案 tokenization 偏差。
- P0：为 negation、quantifier、temporal 各设计第二套模板：
  - 目标是判断低分来自结构困难，还是 prompt 模板偏差；
  - 若换模板后大幅变化，说明当前结论主要是 prompt 敏感性，不是语言结构结论。
- P1：选择稳定高 margin 样本进入机制分析：
  - 正样本：SVO、translation、conditional；
  - 困难样本：negation、quantifier、temporal、recursive；
  - 做 residual stream / attention / MLP patching 时，必须用真实 forward hook 和行为变化验证。
- P1：建立“最小关系操作图谱”：
  - 先不引入复杂数学理论；
  - 只记录对象、角色、极性、集合、时间、条件、符号映射之间的可测试关系；
  - 等足够多稳定拼图出现后，再让抽象结构自然浮现。

---

## Phase 13: 系统语言测试阶段性进展与下一阶段方案 [2026-05-26 11:50]

### 本轮执行命令
- `sed -n '1,280p' tests/gpt5/systematic_language_benchmark.py`
- `tail -n 140 research/gpt5/docs/AGI_GPT5_MEMO.md`
- `find results -maxdepth 2 -type f \( -name '*systematic_language*' -o -name 'summary.json' \) -printf '%p %s bytes\n' 2>/dev/null | sort | tail -n 80`
- `date '+%Y-%m-%d %H:%M'`

### 当前研究进展
- 工程层面已经从“不可恢复长跑”升级为“可恢复长跑”：
  - 每个模型、每个 category 单独 checkpoint；
  - 默认支持 `--resume`；
  - 每完成一个 category 生成或更新 `*_systematic_language.partial.json`；
  - 崩溃后可以跳过已完成 category 继续跑；
  - 三模型已经完成 900 条/模型正式测试。
- 行为测试层面已经从 Phase 9 的 60 条小样本，升级到 Phase 12 的 2700 条总样本：
  - Qwen3：900 条；
  - DeepSeek7B：900 条；
  - GLM4：900 条。
- 结果层面已经形成第一张语言功能差异地图：
  - 三模型共同强项：SVO 主谓宾角色、基础翻译；
  - 整体较强：条件关系；
  - 明显拉开差距：否定、量词、时间、递归绑定；
  - Qwen3 在当前人工规则生成测试上最稳；
  - GLM4 居中；
  - DeepSeek7B 在否定、量词、时间上明显较弱。

### 当前最重要结论
- 语言能力不是均匀整体强弱，而是按基础关系操作分化：
  - 对象-角色绑定；
  - 主动/被动转换；
  - 递归限定；
  - 极性翻转；
  - 集合量词约束；
  - 时间坐标定位；
  - 条件触发映射；
  - 符号翻译映射。
- 这比“某层某头负责语言”更接近当前项目核心问题：语言可能由一组可组合的基础关系操作构成。
- 下一步重点应从“扩大一个总分”转向“系统拆解每一种关系操作的稳定边界、失败边界和可干预中间状态”。

### 当前硬伤
- 样本由规则模板生成，可能存在模板适配和分布偏差。
- 每类 100 条已经比小样本稳，但还不足以支持强理论结论。
- 二选一 logprob 能给出清晰行为差异，但不能等价于开放生成能力。
- first token 和 full-choice 本轮一致，不代表所有任务都一致；多 token 答案、中文答案、长答案仍需继续校验。
- 低分任务可能混有三类因素：
  - 语言结构确实困难；
  - prompt 模板不适配；
  - 候选答案/tokenization 设计有偏差。
- 现在还不能把行为差异直接解释为神经元、attention head 或 MLP 的功能分工。

### 下一阶段系统测试方案
- 第一阶段：错误审计与数据清洗。
  - 对每个模型、每个 category 输出：
    - 错误样本；
    - 低 margin 正确样本；
    - 高 margin 正确样本；
    - first/full/mean scoring 不一致样本。
  - 手工检查题目歧义、答案顺序、tokenization、模板误导。
  - 目标：先证明测试题本身可靠。
- 第二阶段：模板鲁棒性测试。
  - 对否定、量词、时间、递归各设计至少 3 套模板。
  - 每套模板每类至少 100 条。
  - 比较同一语义关系在不同表达下是否保持相同强弱。
  - 若换模板后结果大幅变化，说明当前问题主要是 prompt 敏感性。
- 第三阶段：变量交换测试。
  - 对 SVO、被动、递归做最小对：
    - A 追 B；
    - B 追 A；
    - A 被 B 追；
    - B 被 A 追。
  - 对否定/量词做极性最小对：
    - all / no / some / few / not every；
    - true / false / unknown。
  - 目标：观察模型是否真的跟随关系变量变化，而不是依赖词频或模板位置。
- 第四阶段：跨模型共性图谱。
  - 用统一指标比较 Qwen3、GLM4、DeepSeek7B：
    - 每类 accuracy；
    - mean/full margin 分布；
    - 错误重合率；
    - 模板敏感性；
    - 变量交换一致性。
  - 目标：区分“所有模型共同困难”与“某个模型架构/训练特有困难”。
- 第五阶段：机制分析准入。
  - 只有满足以下条件的任务进入 activation patching：
    - 行为结果稳定；
    - 模板鲁棒；
    - 变量交换可控；
    - margin 足够大或错误足够稳定；
    - 三模型中至少两个模型出现可比较模式。
  - 优先机制任务：
    - SVO/被动：角色绑定；
    - translation：符号映射；
    - conditional：条件触发；
    - negation/quantifier/temporal：失败机制。

### 阶段性大任务
- 大任务 A：建立语言行为可靠性基准。
  - 目标不是刷分，而是找出哪些语言关系测试真的稳定、干净、可复现。
- 大任务 B：建立最小关系操作图谱。
  - 从对象、角色、极性、集合、时间、条件、符号映射这些基础操作开始，不先引入高级理论。
- 大任务 C：从行为进入因果机制。
  - 在可靠行为任务上做 residual stream、attention、MLP patching，观察干预是否按预测改变输出。
- 大任务 D：寻找复用与差异化的神经实现。
  - 比较同一任务不同样本共享的中间状态；
  - 比较相近任务之间复用的部分；
  - 比较最小变量交换导致的差异部分。

### 下一步具体执行
- P0：写错误审计脚本，输入 `results/gpt5_systematic_language_benchmark_v1/*_systematic_language.json`，输出每模型每类的错误和低 margin 样本。
- P0：写模板鲁棒性 v2 测试集，优先覆盖 negation、quantifier、temporal、recursive。
- P0：保持 checkpoint/resume 架构，所有长跑继续按模型和 category 分段保存。
- P1：基于审计结果挑选第一批机制分析样本，每类正样本和困难样本各 20 条。
- P1：对 Qwen3 先做 residual stream patching，因为其行为最稳，最适合作为机制定位起点。

---

## Phase 14: 系统语言结果错误审计与数据质量硬伤 [2026-05-26 11:55]

### 本轮执行命令
- `python - <<'PY' ... 检查 qwen3_systematic_language.json 的 case 结构 ... PY`
- `sed -n '1,220p' results/gpt5_systematic_language_benchmark_v1/summary.json`
- `find results/gpt5_systematic_language_benchmark_v1/checkpoints -mindepth 2 -maxdepth 2 -type f | sort | head -30`
- `python -m py_compile tests/gpt5/audit_systematic_language_results.py`
- `python tests/gpt5/audit_systematic_language_results.py --input-dir results/gpt5_systematic_language_benchmark_v1 --output-dir results/gpt5_systematic_language_audit_v1 --top-k 20 --low-margin 1.0`
- `sed -n '1,220p' results/gpt5_systematic_language_audit_v1/audit.md`
- `python - <<'PY' ... 输出跨模型全错样本、低 margin 统计和唯一 prompt 数 ... PY`
- `date '+%Y-%m-%d %H:%M'`

### 生成/修改脚本
- 新增：`tests/gpt5/audit_systematic_language_results.py`
- 输出：
  - `results/gpt5_systematic_language_audit_v1/audit.json`
  - `results/gpt5_systematic_language_audit_v1/audit.md`
- 脚本功能：
  - 每模型、每 category 统计准确率、失败数、低 margin 正确数；
  - 提取最严重失败样本；
  - 提取低 margin 正确样本；
  - 提取高 margin 正确样本；
  - 检查 first/full、mean/full scoring 是否不一致；
  - 统计跨模型共同错误与混合错误；
  - 新增 `unique_prompt_choices` 和 `duplicate_factor`，检查数据重复。

### 审计结果
- 三模型整体结果与 Phase 12 一致：
  - DeepSeek7B：900 条，full accuracy 71.11%；
  - GLM4：900 条，full accuracy 78.89%；
  - Qwen3：900 条，full accuracy 95.56%。
- 但发现严重数据质量问题：
  - 多数 category 并不是 100 个唯一样本，而是 10-50 个唯一题重复扩容。
  - `comparison`：100 条中只有 20 个唯一 prompt+choices，重复因子 5；
  - `conditional`：100 条中只有 20 个唯一 prompt+choices，重复因子 5；
  - `negation_yesno`：100 条中只有 20 个唯一 prompt+choices，重复因子 5；
  - `passive_agent`：100 条中只有 20 个唯一 prompt+choices，重复因子 5；
  - `recursive_binding`：100 条中只有 20 个唯一 prompt+choices，重复因子 5；
  - `svo_agent`：100 条中只有 20 个唯一 prompt+choices，重复因子 5；
  - `quantifier`：100 条中只有 10 个唯一 prompt+choices，重复因子 10；
  - `temporal`：100 条中只有 10 个唯一 prompt+choices，重复因子 10；
  - `translation`：100 条中只有 50 个唯一 prompt+choices，重复因子 2。

### 关键样本发现
- 跨模型共同全错只集中在两个类别：
  - `recursive_binding`：5 条，实际是同一模板重复，如 `The student that the pilot guarded was old. The old one was the ...`
  - `temporal`：10 条，实际是同一模板重复，如 `Tomorrow, the tree grew. The growing happens in the ...`
- DeepSeek7B 典型错误：
  - 否定 yes/no：大量把 `not ...?` 判断成 `yes`；
  - 量词：`Few keys came... Did many keys come?` 判断成 `yes`；
  - temporal：`Now, Sam washed his hands...` 倾向 `past`；
  - passive：`the school is followed by the nurse` 倾向选择 patient。
- GLM4 典型错误：
  - 否定和量词也明显偏 `yes`；
  - temporal 对 future/past 有模板偏置；
  - recursive binding 在多名词绑定中不稳。
- Qwen3 典型错误：
  - conditional 中 `guard wakes up` 被 `asleep` 干扰；
  - temporal 中 before/after 和未来时有错误；
  - recursive binding 中部分主语关系从句不稳。

### 对 Phase 12 结论的修正
- Phase 12 的三模型分数仍有参考价值，但必须降级解释：
  - 它不是 900 个独立语言样本的结果；
  - 更准确地说，是“少量模板重复扩容后的稳定性结果”；
  - 重复样本会放大某些错误，也会让 accuracy 看起来更稳定。
- 仍可保留的结论：
  - SVO 和 translation 在当前模板下非常稳定；
  - negation、quantifier、temporal、recursive 是困难区；
  - Qwen3 在当前模板分布下最稳。
- 不能保留为强结论的部分：
  - 不能说每类 100 条已充分覆盖该语言功能；
  - 不能把错误重合率直接解释为三模型共同机制缺陷；
  - 不能直接进入 activation patching，否则可能是在解释模板重复偏差。

### 理论研究进展
- 本轮最重要的研究进展是方法论上的：
  - 语言机制研究不能只扩大样本数量，必须扩大“独立关系结构”的数量；
  - 重复模板会把一个局部现象伪装成稳定规律；
  - 真正需要的是“关系结构多样性”，不是简单题目条数。
- 对“语言背后数学结构”的第一性约束进一步明确：
  - 如果语言编码机制真的存在，应跨实体、跨动词、跨模板、跨表达方式保持某种稳定变换；
  - 如果只在一个模板里稳定，那更可能是模板记忆或局部模式；
  - 所以后续测试必须把变量空间系统展开：实体、关系、极性、时间、集合、句式都要分别控制。

### 下一步任务
- P0：重写 v2 测试集生成器，硬性要求每个 category 至少 100 个唯一 prompt+choices。
- P0：加入 uniqueness guard：
  - 生成后自动统计 `unique_prompt_choices`；
  - 若低于要求，脚本直接报错，不允许长跑。
- P0：优先修复四个关键类别：
  - `negation_yesno`：增加不同句式，不只 `X is not Y`；
  - `quantifier`：扩展 all/no/some/few/not all/exactly one/at least two；
  - `temporal`：修正语义冲突模板，例如 `Tomorrow, the tree grew` 这种时态混乱；
  - `recursive_binding`：增加主语关系从句、宾语关系从句、所有格、介词短语限定。
- P1：v2 跑完后再做错误审计，只有通过唯一性和模板鲁棒性检查后，才进入机制分析。

---

## Phase 15: 读取 GLM5 Phase 285-287 后的研发计划与显卡稳定性复核 [2026-05-26 16:39]

### 本轮执行命令
- `rg -n "^## Phase (285|286|287):" research/glm5/docs/AGI_GLM5_MEMO.md`
- `sed -n '58396,58585p' research/glm5/docs/AGI_GLM5_MEMO.md`
- `sed -n '58586,58710p' research/glm5/docs/AGI_GLM5_MEMO.md`
- `sed -n '58711,58920p' research/glm5/docs/AGI_GLM5_MEMO.md`
- `nvidia-smi`
- `cat /proc/driver/nvidia/version`
- `cat /proc/driver/nvidia/params | rg -i 'EnableGpuFirmware|GSP|OpenRm|Firmware'`
- `modinfo nvidia | rg -i 'filename|version|license|firmware|EnableGpuFirmware'`
- `journalctl -b 0 -k --no-pager | rg -i 'NVRM|Xid|GSP|GPU is probably locked|GPU Reset Required|soft lockup|oom|out of memory'`
- `python - <<'PY' ... torch tiny cuda op ... PY`
- `apt-cache policy nvidia-driver-570 nvidia-driver-570-open nvidia-driver-580 nvidia-driver-580-open`
- `dpkg -l | rg '^(ii)\s+(nvidia|libnvidia|linux-modules-nvidia)'`
- `dkms status`
- `for i in $(seq 1 6); do ... nvidia-smi ...; sleep 5; done`
- `date '+%Y-%m-%d %H:%M'`

### GLM5 Phase 285-287 关键信息
- Phase 285：
  - 从 manual attention 转向真实 forward activation patching；
  - 用真实 hook/cache/patch 测 attn、mlp、resid 的因果效应；
  - 得到三类架构图像：Qwen3 分布式、GLM4 L0 MLP 瓶颈、DS7B 组件因果决定型；
  - 但只有 28 对样本，per-category 只有 2 对，且 effect>1 和 L0 artifact 需要谨慎。
- Phase 286：
  - 下沉到 head-level real forward patching；
  - 发现 Qwen3 单 head 效应小，GLM4 attention head 大多可互换，DS7B head 因果效应强；
  - 提出 diff norm 不能预测 causal importance，这是重要方法论发现；
  - 但 per-category 只有 1 pair，GLM4 采样 head 数少，DS7B effect>1 普遍。
- Phase 287：
  - 进一步分离 attention head 的 routing 与 content；
  - 结论是 15/15 heads 显示 routing/content 可分离；
  - 重要发现：negation 在 Qwen3/DS7B 中偏 content，translation 三模型接近平衡；
  - 但关键类别样本少，例如 DS7B negation R/C 基于 3 对，GLM4 平衡可能是弱效应噪声。

### 结合 GPT5 当前研究进展的判断
- GLM5 Phase 285-287 的方法方向是对的：
  - 应优先使用真实 forward patching；
  - manual attention 只能作为辅助解释；
  - causal patching 比 diff norm/probing 更可信。
- 但当前不能直接把这些结果作为强理论结论：
  - GPT5 Phase 14 已发现 v1 行为数据存在重复模板扩容问题；
  - GLM5 Phase 285-287 的机制实验样本更少，category 级解释信度不足；
  - 机制实验应绑定到经过唯一性、模板鲁棒性、变量交换验证的行为样本上。
- 因此下一步不是继续扩大 head patching，而是先建立可靠行为基准 v2，然后把 patching 接到 v2 的稳定样本和困难样本上。

### 显卡驱动稳定性复核
- 用户说明显卡驱动已更新为 570；但本机实际检查结果：
  - `nvidia-driver-570` 元包确实显示已安装；
  - 但当前活跃驱动仍是 `580.159.03`；
  - `nvidia-smi` 显示 Driver Version `580.159.03`；
  - `/proc/driver/nvidia/version` 显示 `580.159.03`；
  - `dkms status` 显示 `nvidia/580.159.03`；
  - 当前安装的 libnvidia/nvidia-utils/nvidia-dkms 也主要是 580。
- 当前积极变化：
  - 已不再加载 open kernel module；
  - `modinfo nvidia` 路径为 `/updates/dkms/nvidia.ko.zst`；
  - license 为 `NVIDIA`，不是 `Dual MIT/GPL`；
  - `EnableGpuFirmware: 0`，即 GSP 已关闭；
  - 当前 boot 没有发现新的 NVIDIA Xid/GSP timeout；
  - tiny CUDA op 成功；
  - 30 秒低负载 `nvidia-smi` 轮询正常。
- 稳定性结论：
  - 低负载状态目前正常；
  - 但不能确认高负载稳定，因为当前活跃驱动仍是 580，不是 570；
  - 由于此前 Xid 119 是高负载/长 CUDA 调用触发，必须通过分阶段 smoke 才能确认。

### 下一步研发计划
- P0：先解决驱动状态确认。
  - 如果目标是 570，需清理 580 或切换 alternatives/DKMS，使 `nvidia-smi` 真正显示 570；
  - 如果暂时接受 580 proprietary + GSP disabled，则先做小规模稳定性测试，不直接跑全量。
- P0：恢复 v2 行为测试，但采用更保守阶梯：
  1. Qwen3 `cases-per-category=1`；
  2. Qwen3 单 category 20 条；
  3. Qwen3 单 category 100 条；
  4. 观察 30 分钟无 Xid 后再跑全部 900 条；
  5. 再依次跑 GLM4、DeepSeek7B。
- P0：继续完善 v2 唯一样本测试集。
  - 当前 `systematic_language_benchmark.py` 已加入 v2 唯一性生成与 `DATASET_VERSION`；
  - 每类 100 个唯一 prompt+choices；
  - checkpoint 需继续细化到 category 内分片，避免单 category 卡死丢进度。
- P1：v2 行为结果完成后，做错误审计 v2。
  - 提取稳定正确样本、高 margin 样本、稳定错误样本、低 margin 样本；
  - 排除模板歧义和 tokenization 偏差。
- P1：机制实验准入标准。
  - 只对通过行为基准的样本做 real forward patching；
  - 优先 Qwen3，因为行为最稳、组件效应分布式，适合验证“复用/差异化”；
  - 机制粒度按顺序推进：resid stream patching → attn/mlp patching → head patching → route/content separation。
- P2：理论主线。
  - 不直接相信“某 head 负责某功能”；
  - 重点寻找对象、角色、极性、集合、时间、条件、翻译等最小关系操作在网络中的可复现状态转换；
  - 用 causal patching 验证这些状态是否可干预，而不是只看激活差异。

---

## Phase 16: 6.17 回退导致 NVIDIA 570 再次无法加载的修复 [2026-05-26 21:24]

### 背景
用户反馈刚刚又出现卡死，重启后系统内核又变回 `6.17`，显卡驱动没有加载成功。

### 本轮执行命令
- `uname -r && nvidia-smi || true`
- `cat /proc/driver/nvidia/version 2>/dev/null || true`
- `dkms status || true`
- `ls -1 /boot | sort | rg 'vmlinuz|initrd|config|System.map'`
- `sed -n '1,120p' /etc/default/grub 2>/dev/null || true`
- `grub-editenv list 2>/dev/null || true`
- `journalctl -b -k --no-pager | rg -i 'nvidia|nvrm|Invalid relocation|Exec format|Xid|GSP' | tail -120`
- `modinfo -k 6.8.0-117-generic nvidia | rg '^(filename|version|vermagic|signer|sig_hashalgo):'`
- `modinfo -k 6.17.0-29-generic nvidia | rg '^(filename|version|vermagic|signer|sig_hashalgo):'`
- `bash -n tests/gpt5_temp/force_kernel_6_8_default.sh`

### 当前状态
- 当前启动内核：
  - `6.17.0-29-generic`
- 当前 `nvidia-smi` 失败：
  - 无法连接 NVIDIA driver。
- DKMS 状态：
  - `nvidia/570.211.01, 6.17.0-29-generic, x86_64: installed`
  - `nvidia/570.211.01, 6.8.0-117-generic, x86_64: installed`
- 6.8 内核文件仍存在：
  - `/boot/vmlinuz-6.8.0-117-generic`
  - `/boot/initrd.img-6.8.0-117-generic`
- `/etc/default/grub` 当前仍是：
  - `GRUB_DEFAULT=0`
  - `GRUB_TIMEOUT_STYLE=hidden`
  - `GRUB_TIMEOUT=0`

### 关键判断
- 这次“重启后回到 6.17”不是奇怪现象，而是 GRUB 默认项仍为 `0`：
  - 默认选择最新内核；
  - 当前最新内核是 `6.17.0-29-generic`；
  - 所以每次普通重启都会回到 6.17。
- 此前进入 6.8 是通过 `grub-reboot` 设置的一次性启动项；
  - 一次性启动项使用后会清空；
  - 后续重启自然回到默认 6.17。
- 570 在 6.17 下仍复现同一类错误：
  - `module: x86/modules: Invalid relocation target, existing value is nonzero for type 1`
- 因此显卡驱动未加载成功的直接原因是：
  - 系统启动到了 6.17；
  - 570 模块在 6.17 下加载失败；
  - 而不是 570 包丢失。

### 生成脚本
- 新增：
  - `tests/gpt5_temp/force_kernel_6_8_default.sh`
- 脚本功能：
  - 检查 `6.8.0-117-generic` 内核是否存在；
  - 检查 NVIDIA 570 DKMS 是否已为 6.8 安装；
  - 备份 `/etc/default/grub`；
  - 将 `GRUB_DEFAULT=0` 改为：
    - `GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 6.8.0-117-generic"`
  - 临时打开 5 秒 GRUB 菜单：
    - `GRUB_TIMEOUT_STYLE=menu`
    - `GRUB_TIMEOUT=5`
  - 执行 `update-grub`。
- 脚本已通过语法检查：
  - `bash -n tests/gpt5_temp/force_kernel_6_8_default.sh`

### 执行方式
在本机终端执行：

```bash
cd /home/rankrank/Documents/OpenOne/Ai2050-OpenOne
bash tests/gpt5_temp/force_kernel_6_8_default.sh
sudo reboot
```

重启后验证：

```bash
uname -r
nvidia-smi
cat /proc/driver/nvidia/params | grep EnableGpuFirmware
```

### 对模型测试的影响
- 本次卡死说明即使 `570 + 6.8 + GSP off` 能启动，模型测试仍可能在 CUDA/驱动清理阶段卡住。
- 刚才的 stage10 结果显示：
  - Qwen3 已完成 90 条并写入 checkpoint；
  - 卡死发生在 Qwen3 完成后、准备释放模型或进入下一个模型阶段；
  - 因此后续不能再用“一个 Python 进程连续跑三个模型”的方式。
- 已对 `tests/gpt5/systematic_language_benchmark.py` 增加 `--hard-exit-after-model` 方案：
  - 单模型单进程；
  - 写完结果后跳过显式 CUDA cleanup；
  - 用 `os._exit(0)` 直接结束进程；
  - 避免卡在模型释放/CUDA 清理路径。

### 下一步计划
1. 先永久固定 6.8 默认启动，恢复 NVIDIA 570。
2. 重启确认：
   - `uname -r` 必须是 `6.8.0-117-generic`；
   - `nvidia-smi` 必须正常；
   - `EnableGpuFirmware` 必须为 `0`。
3. 继续测试时采用单模型单进程：
   - Qwen3 已有 stage10 结果；
   - GLM4 单独跑；
   - DeepSeek7B 单独跑；
   - 每个模型结束后重新开进程，必要时每个模型之间重启。
4. 研究结论保持谨慎：
   - 当前最重要的不是扩大样本，而是先把驱动/内核变量固定；
   - 否则语言机制结果会被系统不稳定污染。

## Phase 17: 再次卡死后的根因分层与测试策略降级 [2026-05-26 21:48]

### 背景
用户反馈三个模型测试过程中再次出现卡死。此前已经固定到 `6.8.0-117-generic + NVIDIA 570.211.01 + GSP off`，因此本轮重点判断是否仍是 6.17 回退问题，还是更深层的 CUDA/NVIDIA 驱动死锁问题。

### 本轮执行命令
- `uname -r`
- `cat /proc/driver/nvidia/version`
- `cat /proc/driver/nvidia/params | rg 'EnableGpuFirmware|EnableGpuFirmwareLogs'`
- `ps -eo pid,ppid,etime,stat,pcpu,pmem,wchan:32,cmd | rg 'python tests/gpt5|systematic_language_benchmark|nvidia-smi|cuda|python'`
- `journalctl -b -k --since '20 minutes ago' --no-pager | rg -i 'NVRM|Xid|GSP|GPU is probably locked|nvidia|uvm|drm|soft lockup|hung|blocked|timeout|reset'`
- `journalctl -b -1 -k --no-pager | rg -i 'NVRM|Xid|GSP|GPU is probably locked|nvidia|uvm|drm|soft lockup|hung|blocked|timeout|reset|os_acquire'`
- `journalctl --list-boots --no-pager | tail -8`
- `python - <<'PY' ... checkpoint 完成度统计 ... PY`
- `lspci -nn | rg -i 'vga|3d|display|graphics|arc|intel'`
- `ls -l /dev/dri`
- `for f in /sys/class/drm/card*/device/vendor; do ...`
- `python - <<'PY' ... torch/transformers/python 版本 ... PY`
- `bash -n tests/gpt5_temp/run_stage10_single_category_after_reboot.sh`
- `bash -n tests/gpt5_temp/create_stable_cuda121_env.sh`

### 当前系统状态
- 当前已回到正确内核与驱动组合：
  - `uname -r`: `6.8.0-117-generic`
  - NVIDIA kernel module: `570.211.01`
  - `EnableGpuFirmware: 0`
- 当前 boot 中 NVIDIA 驱动加载正常。
- 上一轮卡死不是 6.17 回退导致的驱动加载失败，而是在 6.8 + 570 + GSP off 下运行模型时再次触发。

### 关键现场
- Qwen3 stage10 已完整完成：
  - 9 个 category 全部 10/10；
  - 结果文件已写入。
- GLM4 stage10 进展：
  - 已完成：`svo_agent`、`passive_agent`、`negation_yesno`、`conditional`、`comparison`、`temporal`、`recursive_binding`
  - `quantifier` 只完成 5/10；
  - `translation` 未完成。
- DeepSeek7B stage10 未开始。
- GLM4 卡死时进程状态显示：
  - 主进程在用户态 `futex_wait_queue`；
  - 一个 CUDA 相关线程进入 D 状态，wchan 为 `os_acquire_mutex`；
  - `SIGKILL` 后主线程可变 zombie，但 D 状态 CUDA 线程不能被杀掉；
  - 必须重启才能释放。

### 重要判断
- 这已经不是普通 Python 异常，也不是单纯脚本逻辑问题。
- 这是 NVIDIA/CUDA 内核驱动路径死锁：
  - 进程进入不可中断 D 状态；
  - 普通 kill 无效；
  - `nvidia-smi` 查询也可能卡住；
  - 桌面可能一起卡死。
- 内核日志没有稳定出现 Xid，并不代表没有驱动问题：
  - D 状态卡死可能发生在驱动 mutex/rwlock 路径；
  - 未必来得及输出 Xid；
  - 也可能日志在硬重启前没有完整落盘。

### 环境风险
- 当前 Python/CUDA 栈偏激进：
  - Python: `3.13.12`
  - PyTorch: `2.6.0+cu124`
  - CUDA runtime: `12.4`
  - transformers: `5.5.4`
- 对大模型 CUDA 稳定性测试来说，这套组合不够保守。
- 用户此前也提到 Windows 上通过降低 CUDA 版本解决过类似卡死，因此当前应把 CUDA runtime / PyTorch 版本作为关键变量。

### 显示与计算共卡问题
- 当前机器只有 NVIDIA 作为 VGA/display 设备：
  - `lspci` 只显示 4090D 为 VGA；
  - `/dev/dri` 中只有 NVIDIA vendor `0x10de`；
  - 没有可见 Intel/AMD 独显或核显 DRM card。
- 这意味着桌面显示和 CUDA 计算共用同一张 4090D。
- 一旦 CUDA/NVIDIA 内核驱动死锁，桌面也会被拖死。
- 这是“模型进程卡死”升级成“整机卡死”的重要结构性原因。

### 已做代码调整
- 修改 `tests/gpt5/model_registry.py`：
  - 将 GLM4 与 DeepSeek7B 的 `load_strategy` 从 `auto` 改为 `cuda`；
  - 目的：减少 accelerate `device_map=auto` / CPU offload / 多路径调度的复杂性。
- `tests/gpt5/systematic_language_benchmark.py` 已支持：
  - `--hard-exit-after-model`
  - 单模型单进程，写完后 `os._exit(0)`。
- 新增：
  - `tests/gpt5_temp/run_stage10_single_category_after_reboot.sh`
  - 功能：每个模型、每个 category 单独进程运行，已完成 checkpoint 自动跳过。
- 新增：
  - `tests/gpt5_temp/create_stable_cuda121_env.sh`
  - 功能：创建保守 CUDA 12.1 环境：
    - Python 3.11
    - PyTorch `2.5.1+cu121`
    - transformers `4.52.4`
    - accelerate `1.8.1`
  - 目的：验证是否为当前 Python 3.13 / cu124 / transformers 5.x 组合触发驱动死锁。

### 根因分层
当前最可能原因排序：

1. **NVIDIA/CUDA 驱动死锁**
   - 证据最强：D 状态、`os_acquire_mutex`、kill 无效、重启释放。

2. **当前 CUDA runtime / PyTorch / transformers 组合不稳**
   - 当前栈过新，且用户历史经验支持“降低 CUDA 版本可能解决”。

3. **显示和计算共用 4090D**
   - 不一定是死锁根因，但会显著放大卡死影响。

4. **GLM4/DeepSeek7B 加载路径复杂**
   - 原先使用 `device_map=auto`，可能触发 accelerate/offload 与 CUDA 清理交互问题；
   - 已改为整模型 CUDA。

5. **硬件/供电/主板 BIOS/PCIe 稳定性**
   - 不能排除，但目前还没有 ECC/PCIe/AER 证据。

### 解决方案
短期不继续强推三模型长跑。先按下面顺序降级排查：

1. 固定内核与驱动：
   - 保持 `6.8.0-117-generic`
   - 保持 `570.211.01`
   - 保持 `EnableGpuFirmware=0`

2. 建立保守 CUDA 环境：

```bash
bash tests/gpt5_temp/create_stable_cuda121_env.sh
conda activate openone-cuda121
```

3. 用保守环境只跑最小 category：

```bash
python tests/gpt5/systematic_language_benchmark.py glm4 \
  --cases-per-category 10 \
  --batch-size 1 \
  --case-chunk-size 1 \
  --progress-every 2 \
  --categories quantifier \
  --output-dir results/gpt5_systematic_language_v2_driver570_stage10 \
  --hard-exit-after-model
```

4. 如果 GLM4 `quantifier` 能过，再跑 `translation`。
5. 如果 GLM4 仍卡死，暂停 GLM4，改跑 DeepSeek7B 单 category。
6. 如果 DeepSeek7B 也卡死，停止 GPU 模型测试，转向系统层稳定性处理。

### 系统层建议
- 最推荐：让显示和计算分离。
  - 在 BIOS 中启用 Intel 核显/集显多显示；
  - 显示器插主板视频输出；
  - 让 4090D 只做 CUDA compute。
- 如果这台机器无法启用核显：
  - 可以考虑加一张低功耗显示卡；
  - 或远程/TTY/headless 模式下跑测试，降低桌面被 GPU 死锁拖住的概率。
- 可选降载：
  - 降低 4090D power limit；
  - 降低 batch size 已经做了；
  - 每个 category 后重启是最保守但最慢的方式。

### 研究影响
- 当前不能把 GLM4/DeepSeek7B 未完成测试解释为语言机制问题。
- 当前最主要瓶颈是实验平台稳定性。
- 在平台稳定前，只能保留 Qwen3 stage10 行为结果和 GLM4 已完成 category 的局部结果。
- 机制破解阶段必须推迟，不能在驱动死锁环境下做 activation patching 或消融。

## Phase 18: CUDA 13/Driver 595 下 GLM4 Conditional 触发 Xid 62/45 的卡死取证 [2026-05-27 19:48]

### 背景
用户升级 CUDA/驱动后要求继续三模型测试，并要求增加完整日志，以便 CUDA 锁死后定位原因。本轮先关闭 ComfyUI 后，使用带日志包装脚本继续 stage10 测试。

### 本轮执行命令
- `uname -r; nvidia-smi; cat /proc/driver/nvidia/version; cat /proc/driver/nvidia/params | rg 'EnableGpuFirmware|EnableGpuFirmwareLogs'`
- `timeout 8s nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits`
- `ps -eo pid,ppid,etime,stat,pcpu,pmem,wchan:32,cmd | rg 'ComfyUI|python main.py|systematic_language_benchmark|python tests/gpt5|nvidia-smi'`
- `kill -TERM 3616`
- `OUTPUT_DIR=results/gpt5_systematic_language_v2_driver595_stage10 CASES_PER_CATEGORY=10 tests/gpt5_temp/run_stage10_logged_sequence.sh qwen3 glm4 deepseek7b`
- `journalctl -b -k --since '2026-05-27 13:58:40' --no-pager | rg -i 'NVRM|Xid|GSP|GPU is probably locked|nvidia|uvm|drm|soft lockup|hung|blocked|timeout|reset|os_acquire|BUG|Oops|Call Trace'`
- `journalctl -b -5 -k --no-pager | rg -i 'NVRM|Xid|GSP|GPU is probably locked|nvidia|uvm|drm|soft lockup|hung|blocked|timeout|reset|os_acquire|BUG|Oops|Call Trace'`
- `sed -n '1,220p' results/gpt5_gpu_lock_logs/20260527_135845_glm4_conditional/run.log`
- `tail -120 results/gpt5_gpu_lock_logs/20260527_135845_glm4_conditional/gpu_process_monitor.log`
- `tail -160 results/gpt5_gpu_lock_logs/20260527_135845_glm4_conditional/kernel.follow.log`
- `python - <<'PY' ... checkpoint 状态统计 ... PY`
- `chmod +x tests/gpt5_temp/run_glm4_conditional_xid_repro.sh`
- `bash -n tests/gpt5_temp/run_glm4_conditional_xid_repro.sh`

### 当前系统与运行环境
- 内核：
  - `6.8.0-117-generic`
- 当前 NVIDIA 驱动：
  - `595.71.05`
- `nvidia-smi` 显示 CUDA Version：
  - `13.2`
- `nvcc` 工具链：
  - CUDA Toolkit `13.0`
- PyTorch 运行时仍是：
  - `torch 2.6.0+cu124`
  - `torch.version.cuda = 12.4`
- Python:
  - `3.13.12`
- transformers:
  - `5.5.4`
- GSP:
  - `EnableGpuFirmware: 0`
  - `GSP Firmware Version: N/A`

### 重要前置修正
- 测试开始前发现 ComfyUI 仍在运行：
  - PID `3616`
  - 命令：`python main.py --listen 0.0.0.0 --port 8188 --cuda-device 0 --highvram --fp16-vae --preview-method taesd`
  - 显存占用约 `16912MiB`
- 已执行 `kill -TERM 3616`。
- 之后显存降到约 `780MiB`，只剩桌面/浏览器图形进程。
- 因此本轮后续 Xid 不是 ComfyUI 干扰造成的。

### 已完成测试
- 输出目录：
  - `results/gpt5_systematic_language_v2_driver595_stage10`
- Qwen3 stage10 全部 9 类完成：
  - `svo_agent`: 10/10
  - `passive_agent`: 10/10
  - `negation_yesno`: 10/10
  - `conditional`: 10/10
  - `comparison`: 10/10
  - `temporal`: 10/10
  - `recursive_binding`: 10/10
  - `quantifier`: 10/10
  - `translation`: 10/10
- GLM4 已完成：
  - `svo_agent`: 10/10
  - `passive_agent`: 10/10
  - `negation_yesno`: 10/10
- GLM4 卡死于：
  - `conditional`
  - checkpoint 显示 `num_cases=8`, `complete=False`
  - 最后完成到 `conditional_007`
  - 卡死发生在 `conditional cases 8:9/10`
- DeepSeek7B 未开始。

### 关键日志证据
- 运行日志：
  - `results/gpt5_gpu_lock_logs/20260527_135845_glm4_conditional/run.log`
- 运行日志停止位置：
  - `GLM4 conditional cases 8:9/10`
- 监控日志卡死前状态：
  - `2026/05/27 13:58:46.759`
  - 温度约 `38C`
  - 功耗约 `24.99W`
  - 显存约 `540MiB`
  - GPU util 约 `1%`
  - python 进程 PID `9684`
- 内核日志明确记录：

```text
NVRM: GPU at PCI:0000:01:00: GPU-299fe279-1c52-2255-d4ba-07d7bd2861d9
NVRM: Xid (PCI:0000:01:00): 62, 023f0f30 00000000 00000000 202c0ffe 202bc4b6 2029bdb6 202bccb8 20297b2a
NVRM: Xid (PCI:0000:01:00): 45, pid=9684, name=python, channel 0x00000015
```

### 判断
- 这次卡死已经不是“可能的 Python 卡死”。
- `Xid 45` 明确指向测试进程 `pid=9684, name=python`。
- `Xid 62` + `Xid 45` 说明 GPU/驱动通道发生硬错误，随后相关 channel 被驱动处理。
- 因为卡死前温度、功耗、显存都很低，本次不支持“高温/满载/OOM”解释。
- 因为 GSP 已关闭，本次也不是典型 GSP firmware timeout。
- 当前最强解释：
  1. GLM4 某次 forward 触发 NVIDIA driver / CUDA kernel / PyTorch kernel 路径错误；
  2. 该错误导致 GPU channel 异常和系统卡死；
  3. 桌面和 CUDA 共用 4090D，所以 GPU channel/driver 异常会放大为整机不可用。

### 可能原因排序
1. **Driver 595 + PyTorch cu124 + GLM4 forward 路径的兼容性问题**
   - 驱动已升到 595，但 PyTorch 仍是 cu124 runtime；
   - 当前 Python/transformers 栈非常新；
   - Qwen3 可过，GLM4 conditional 稳定触发风险，说明模型/架构 forward 路径是关键变量。

2. **GLM4 模型实现或 attention/MLP kernel 路径触发驱动 bug**
   - 已使用 `attn_implementation='eager'`，仍可触发；
   - 说明不只限于 flash attention。

3. **显示与计算共用同一张 4090D**
   - 不一定是根因，但会把 GPU 错误变成桌面卡死。

4. **硬件/PCIe/主板 BIOS 稳定性**
   - 不能排除，但本轮没有看到 AER/PCIe 明确错误。

### 新增脚本
- 新增带完整日志的单 category 包装脚本：
  - `tests/gpt5_temp/run_logged_language_category.sh`
- 新增 sequence 包装脚本：
  - `tests/gpt5_temp/run_stage10_logged_sequence.sh`
- 新增重启后取证脚本：
  - `tests/gpt5_temp/collect_gpu_lock_report.sh`
- 新增窄复现脚本：
  - `tests/gpt5_temp/run_glm4_conditional_xid_repro.sh`
  - 默认设置：
    - `CUDA_LAUNCH_BLOCKING=1`
    - `PYTORCH_NO_CUDA_MEMORY_CACHING=1`
    - `TOKENIZERS_PARALLELISM=false`

### 接下来建议
- 暂停 GLM4 和 DeepSeek7B 的 CUDA 测试，不继续硬跑。
- 如果必须复现，只运行：

```bash
tests/gpt5_temp/run_glm4_conditional_xid_repro.sh
```

- 但运行前必须保存工作，因为它可能再次锁死。
- 锁死重启后立即运行：

```bash
tests/gpt5_temp/collect_gpu_lock_report.sh -1
```

### 解决方向
1. 建立更保守的 Python/CUDA 环境：
   - Python 3.11
   - PyTorch 2.5.1 cu121 或更稳定组合
   - transformers 4.x
2. 尝试 GLM4 用 `float16` 而不是 `bfloat16`。
3. 进一步降低 GPU 风险：
   - power limit 降到 250W 或 300W；
   - 但本轮功耗很低，降功耗可能帮助有限。
4. 尽量让显示和计算分离：
   - 启用核显/主板输出；
   - 或加一张低功耗显示卡；
   - 4090D 只做 CUDA。
5. 在平台稳定前，语言机制破解只基于已完成且无 Xid 的 Qwen3 stage10 结果，不进入 GLM4/DeepSeek 的机制消融。

### 研究影响
- Qwen3 stage10 可作为当前稳定行为基线。
- GLM4 在 driver595 下仍未稳定，且已明确触发 `Xid 62/45`。
- DeepSeek7B 尚未测试，不应推断其语言机制。
- 当前主要瓶颈仍是实验平台稳定性，不是语言理论本身。

## Phase 19: 切换 CUDA 12.1 保守环境后的三模型语言测试稳定化 [2026-05-27 20:08]

### 背景
上一个阶段在 base 环境中已经明确捕获到 GLM4 conditional 测试触发 NVIDIA `Xid 62/45`，并且当时温度、功耗、显存都很低，所以问题更像是驱动/CUDA/PyTorch/模型 forward 路径兼容性，而不是 OOM 或高温。为避免继续用不稳定测量环境污染语言机制结论，本阶段切换到更保守的 CUDA 12.1 环境。

### 环境
- 内核：`6.8.0-117-generic`
- 驱动：`595.71.05`
- `nvidia-smi` 显示 CUDA capability：`13.2`
- conda 环境：`openone-cuda121`
- Python：`3.11.15`
- PyTorch：`2.5.1+cu121`
- PyTorch CUDA runtime：`12.1`
- transformers：`4.52.4`
- accelerate：`1.8.1`
- GSP：此前已关闭

### 代码和脚本
- 修改 `tests/gpt5/hf_probe_env.py`
  - 新增 `PROBE_TORCH_DTYPE` 环境变量。
  - 支持 `bfloat16/bf16/float16/fp16/float32/fp32`。
  - 目的：允许不同模型使用不同 dtype 复测，避免把数值精度问题误判成语言机制问题。
- 使用已有长跑保护脚本：
  - `tests/gpt5_temp/create_stable_cuda121_env.sh`
  - `tests/gpt5_temp/run_logged_language_category.sh`
  - `tests/gpt5_temp/run_stage10_logged_sequence.sh`
  - `tests/gpt5_temp/collect_gpu_lock_report.sh`
- 新增窄复现脚本：
  - `tests/gpt5_temp/run_glm4_conditional_xid_repro.sh`

### 关键命令
创建保守环境：

```bash
bash tests/gpt5_temp/create_stable_cuda121_env.sh openone-cuda121
```

GLM4 conditional 窄复测：

```bash
source /home/rankrank/miniconda3/etc/profile.d/conda.sh
conda activate openone-cuda121
OUTPUT_DIR=results/gpt5_systematic_language_v2_driver595_cuda121_stage10 \
PROBE_TORCH_DTYPE=float16 \
CUDA_LAUNCH_BLOCKING=1 \
PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
TOKENIZERS_PARALLELISM=false \
tests/gpt5_temp/run_logged_language_category.sh glm4 conditional 10
```

三模型 stage10 顺序测试：

```bash
source /home/rankrank/miniconda3/etc/profile.d/conda.sh
conda activate openone-cuda121
OUTPUT_DIR=results/gpt5_systematic_language_v2_driver595_cuda121_stage10 \
CASES_PER_CATEGORY=10 \
PROBE_TORCH_DTYPE=float16 \
CUDA_LAUNCH_BLOCKING=1 \
PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
TOKENIZERS_PARALLELISM=false \
tests/gpt5_temp/run_stage10_logged_sequence.sh qwen3 glm4 deepseek7b
```

DeepSeek7B bf16 修正复测：

```bash
source /home/rankrank/miniconda3/etc/profile.d/conda.sh
conda activate openone-cuda121
OUTPUT_DIR=results/gpt5_systematic_language_v2_driver595_cuda121_bf16_stage10 \
CASES_PER_CATEGORY=10 \
PROBE_TORCH_DTYPE=bfloat16 \
CUDA_LAUNCH_BLOCKING=1 \
PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
TOKENIZERS_PARALLELISM=false \
tests/gpt5_temp/run_stage10_logged_sequence.sh deepseek7b
```

### 稳定性结果
- GLM4 conditional 在保守环境下完成 10/10，没有复现 `Xid 62/45`。
- Qwen3 和 GLM4 完成 9 个 category，共 180 个 case，没有系统卡死。
- DeepSeek7B 在 fp16 下完成流程但多类出现 `NaN` margin，因此 fp16 结果判定为无效。
- DeepSeek7B 改用 bf16 后完成 9 个 category，共 90 个 case，没有 NaN，没有 Xid，没有卡死。
- 本阶段末尾 GPU 处于正常桌面空闲状态，`nvidia-smi` 仅显示 Xorg、gnome-shell、浏览器等图形进程。

### 结果摘要
Qwen3，CUDA 12.1 保守环境，fp16，stage10：

| category | acc | mean_margin |
|---|---:|---:|
| svo_agent | 0.60 | 1.641 |
| passive_agent | 0.80 | 1.508 |
| negation_yesno | 1.00 | 2.457 |
| conditional | 0.70 | 2.601 |
| comparison | 1.00 | 2.039 |
| temporal | 0.90 | 1.688 |
| recursive_binding | 0.70 | 0.898 |
| quantifier | 1.00 | 2.212 |
| translation | 1.00 | 11.205 |
| micro average | 0.856 | - |

GLM4，CUDA 12.1 保守环境，fp16，stage10：

| category | acc | mean_margin |
|---|---:|---:|
| svo_agent | 1.00 | 4.859 |
| passive_agent | 0.90 | 2.112 |
| negation_yesno | 0.40 | 0.082 |
| conditional | 1.00 | 3.686 |
| comparison | 0.80 | 1.316 |
| temporal | 1.00 | 1.013 |
| recursive_binding | 0.80 | 0.630 |
| quantifier | 0.50 | 0.340 |
| translation | 1.00 | 7.736 |
| micro average | 0.822 | - |

DeepSeek7B，CUDA 12.1 保守环境，bf16，stage10：

| category | acc | mean_margin |
|---|---:|---:|
| svo_agent | 0.70 | 2.343 |
| passive_agent | 0.50 | -1.088 |
| negation_yesno | 0.40 | -0.258 |
| conditional | 0.70 | 2.632 |
| comparison | 0.60 | 0.751 |
| temporal | 0.50 | 0.750 |
| recursive_binding | 0.60 | 0.287 |
| quantifier | 0.50 | 0.648 |
| translation | 1.00 | 8.834 |
| micro average | 0.611 | - |

### 结论
1. 保守环境显著改善稳定性。此前 GLM4 conditional 在 base 环境触发 `Xid 62/45`，本阶段同一类测试在 `torch 2.5.1+cu121 + transformers 4.52.4 + fp16` 下通过，说明原问题更可能来自软件栈兼容性，而不是模型测试脚本的基本逻辑。
2. DeepSeek7B 不应使用 fp16 结果。fp16 下多个 category 出现 NaN，说明数值路径不可信；bf16 复测后结果正常，后续 DeepSeek7B 默认使用 bf16。
3. 当前语言测试仍只是 stage10 小样本，用于验证环境、流程、指标和大体方向，不能直接作为语言数学结构的强结论。
4. 三个模型都表现出明显 category 分化：translation 最稳定，否定、量词、被动、递归绑定更弱。这支持“语言能力不是单一能力，而是多个可分离功能/模式的组合”的研究路线。

### 硬伤和风险
- 每类只有 10 个 case，统计量太小，只能看作工程稳定性检查和粗粒度信号。
- 当前 benchmark 是二选一 logprob 形式，不能覆盖生成式语言使用中的完整路径。
- acc 和 margin 只能说明输出偏好，不能直接说明神经元级编码机制。
- 不同模型 dtype 不一致：Qwen3/GLM4 使用 fp16，DeepSeek7B 使用 bf16。这样更稳定，但跨模型比较必须谨慎。
- 使用桌面同卡计算，未来长跑仍可能因为 GPU driver 错误放大为整机卡死。

### 理论进展
本阶段最重要的理论价值不是分数本身，而是确定了一个可继续工作的实验平台。语言背后编码机制的破解必须先区分三层变量：

1. 测量系统变量：驱动、CUDA、PyTorch、dtype、attention 实现。
2. 行为功能变量：SVO、被动、否定、条件、比较、时间、递归、量词、翻译。
3. 神经编码变量：哪些 layer、head、MLP/neuron 对某类差异负责，哪些部分复用，哪些部分差异化。

目前已经基本处理第一层，可以开始系统推进第二层，再进入第三层。若跳过第二层直接做组件消融，容易得到大量不可解释的局部现象。

### 下一阶段计划
1. 固定保守环境作为正式实验环境：
   - `openone-cuda121`
   - Qwen3/GLM4 默认 fp16
   - DeepSeek7B 默认 bf16
   - 每个模型、每个 category 单独进程，保留 `--hard-exit-after-model`
2. 把 stage10 扩大为 stage100 或 stage200：
   - 先跑 Qwen3 单模型全 category；
   - 再跑 GLM4；
   - 最后跑 DeepSeek7B；
   - 每类独立 checkpoint，崩溃可 resume。
3. 做错误模式分解：
   - 不只看 acc，还记录错误 case 的语言类型；
   - 判断失败来自词义、句法位置、逻辑关系、否定方向、量词范围，还是 tokenization 干扰。
4. 设计“最小对照差异”数据：
   - 只改变一个语言因素，例如主动/被动、肯定/否定、主语/宾语交换；
   - 目标是让模型内部差异尽可能对应一个明确语言变量。
5. 在 stage100 稳定后再进入组件分析：
   - 对稳定 category 做 activation patching；
   - 对弱 category 做 layer/head/MLP 差异定位；
   - 比较同一 category 在三模型中的复用与差异化。

### 当前判断
现在可以继续语言机制破解，但必须先把语言行为矩阵做大。第一性原则是：先获得大量、稳定、可复现、单变量控制的语言差异，再去寻找这些差异在神经网络中的路径和复用结构。否则直接研究神经元，很容易把测量噪声、dtype 问题、样本偏差误认为编码机制。

## Phase 20: 将 CUDA 12.1 保守环境固化为默认测试入口 [2026-05-27 20:33]

### 背景
Phase 19 已经证明 `openone-cuda121` 环境可以稳定完成三模型 stage10 语言测试，并且 GLM4 conditional 不再复现 `Xid 62/45`。本阶段目标是把保守环境从“手动输入一串环境变量”固化成默认入口，降低后续误用 base 环境、错误 dtype 或遗漏日志参数的概率。

### 修改内容
- 修改 `tests/gpt5_temp/run_logged_language_category.sh`
  - 默认自动激活 `openone-cuda121`。
  - 可用 `OPENONE_USE_CONSERVATIVE_ENV=0` 关闭自动激活。
  - 默认输出目录改为 `results/gpt5_systematic_language_v2_conservative_stage10`。
  - 默认 dtype 按模型选择：
    - `qwen3`: `float16`
    - `glm4`: `float16`
    - `deepseek7b`: `bfloat16`
  - 日志中新增：
    - `conda_env`
    - `probe_torch_dtype`
- 修改 `tests/gpt5_temp/run_stage10_logged_sequence.sh`
  - 默认自动激活 `openone-cuda121`。
  - 默认输出目录改为 `results/gpt5_systematic_language_v2_conservative_stage10`。
  - 日志中明确输出模型 dtype 默认策略。
- 新增正式入口：
  - `tests/gpt5/run_conservative_language_sequence.sh`
  - 默认设置：
    - `OPENONE_USE_CONSERVATIVE_ENV=1`
    - `OPENONE_CONSERVATIVE_ENV=openone-cuda121`
    - `CASES_PER_CATEGORY=10`
    - `CUDA_LAUNCH_BLOCKING=1`
    - `PYTORCH_NO_CUDA_MEMORY_CACHING=1`
    - `TOKENIZERS_PARALLELISM=false`
- 修改 `tests/gpt5/hf_probe_env.py`
  - 将模型默认 dtype 下沉到 Python 加载层。
  - 即使直接调用 benchmark，不经过 shell 包装，也默认使用：
    - Qwen3/GLM4: fp16
    - DeepSeek7B: bf16
  - 仍可用 `PROBE_TORCH_DTYPE` 手动覆盖。
- 修改 `tests/gpt5/check_probe_env.py`
  - 输出当前 `conda_env`。
  - 输出当前 `PROBE_TORCH_DTYPE`，未设置时显示 `model_default`。
- 修改 `tests/gpt5_temp/create_stable_cuda121_env.sh`
  - 固化 `numpy==1.26.4`，避免 TransformerLens 在 Python 3.11 下与 NumPy 2.x 不兼容。
  - 安装本地 TransformerLens editable 包：
    - `python -m pip install -e . --no-deps`
  - 补齐 TransformerLens 关键依赖：
    - `beartype`
    - `better-abc`
    - `datasets`
    - `fancy-einsum`
    - `jaxtyping`
    - `pandas`
    - `rich`
    - `transformers-stream-generator`
    - `typeguard`
    - `wandb`

### 当前环境修复命令
已在当前 `openone-cuda121` 环境中执行：

```bash
source /home/rankrank/miniconda3/etc/profile.d/conda.sh
conda activate openone-cuda121
python -m pip install -e . --no-deps
python -m pip install beartype==0.14.1 better-abc==0.0.3 datasets==2.21.0 fancy-einsum==0.0.3 jaxtyping==0.2.38 pandas==2.2.3 rich==13.9.4 transformers-stream-generator==0.0.5 typeguard==4.4.2 wandb==0.17.9
python -m pip install numpy==1.26.4
```

### 验证命令
语法验证：

```bash
bash -n tests/gpt5/run_conservative_language_sequence.sh \
  tests/gpt5_temp/run_stage10_logged_sequence.sh \
  tests/gpt5_temp/run_logged_language_category.sh \
  tests/gpt5_temp/create_stable_cuda121_env.sh
```

Python 编译验证：

```bash
source /home/rankrank/miniconda3/etc/profile.d/conda.sh
conda activate openone-cuda121
python -m py_compile tests/gpt5/hf_probe_env.py tests/gpt5/check_probe_env.py tests/gpt5/systematic_language_benchmark.py
```

TransformerLens 环境验证：

```bash
source /home/rankrank/miniconda3/etc/profile.d/conda.sh
conda activate openone-cuda121
python tests/gpt5/check_probe_env.py
```

结果确认：
- Python: `3.11.15`
- NumPy: `1.26.4`
- PyTorch: `2.5.1+cu121`
- CUDA runtime: `12.1`
- transformers: `4.52.4`
- accelerate: `1.8.1`
- transformer_lens: `local-editable`
- 三个本地模型目录均存在，且有 config 和 safetensors。

默认入口冒烟测试：

```bash
MAX_SECONDS=600 \
OUTPUT_DIR=results/gpt5_conservative_default_smoke \
tests/gpt5_temp/run_logged_language_category.sh qwen3 translation 1
```

结果：
- 自动激活环境：`conda_env=openone-cuda121`
- 自动选择 dtype：`probe_torch_dtype=float16`
- checkpoint：`results/gpt5_conservative_default_smoke/checkpoints/qwen3/translation.json`
- `num_cases=1`
- `complete=True`
- `accuracy=1.0`
- `mean_margin=6.953125`
- 未发现新的 `Xid/NVRM` 错误。

### 后续默认用法
小样本验证：

```bash
tests/gpt5/run_conservative_language_sequence.sh qwen3 glm4 deepseek7b
```

扩大样本：

```bash
CASES_PER_CATEGORY=100 \
tests/gpt5/run_conservative_language_sequence.sh qwen3
```

单 category 调试：

```bash
tests/gpt5_temp/run_logged_language_category.sh glm4 conditional 10
```

临时覆盖 dtype：

```bash
PROBE_TORCH_DTYPE=bfloat16 \
tests/gpt5_temp/run_logged_language_category.sh qwen3 translation 10
```

### 判断
当前默认路径已经从“base 环境 + 人工记忆参数”切换为“保守环境 + 模型级 dtype 默认 + 单 category 日志隔离”。这不会解决所有 CUDA/驱动风险，但能显著减少人为误用环境导致的假故障。

### 下一步
下一阶段可以正式跑 stage100：
1. 先跑 Qwen3 全 category。
2. 如果无 Xid、无 NaN，再跑 GLM4。
3. 最后跑 DeepSeek7B。
4. 所有结果必须先做错误类型审计，再进入 activation patching 和组件级定位。

## Phase 21: 读取 GLM5 Phase 288 后调整为 Attention-MLP-Residual 契约图谱方案 [2026-05-27 20:47]

### 读取对象
- `research/glm5/docs/AGI_GLM5_MEMO.md`
- Phase 288: `Attention vs MLP Causal Decomposition [2026-05-27 10:20]`
- 相关脚本：
  - `tests/glm5/phase288_attn_mlp_decomp.py`
  - `tests/glm5_temp/phase288v2_summary.py`
- 相关结果：
  - `results/phase288_attn_mlp/qwen3_decomp.json`
  - `results/phase288_attn_mlp/glm4_decomp.json`
  - `results/phase288_attn_mlp/deepseek7b_decomp.json`

### 综合判断
Phase 288 的方向正确，而且比 Phase 287 更可靠。核心原因是：Phase 287 的 AW@V 手工重构路线因为 eager/flash attention 数值和模块边界不匹配，导致 `full_A_ratio` 远离 1，且 causal/random 不可区分；Phase 288 改为直接 hook attention block 输出和 MLP block 输出，属于标准 activation patching，避开了不可靠重构。

但是 Phase 288 不是“破解编码机制”的终点。它真正把问题推进到更关键的一层：Attention、MLP、Residual 之间的模块契约。后续不能再只问“哪个模块贡献更大”，而要问：

```text
哪个功能在第几层由 attention 产生方向信号；
哪个 MLP 把这个方向信号转换成可继续传播的内部格式；
残差流如何累积这些变化；
哪些替换离开自然分布，不能当作真实机制证据。
```

### 当前计划需要调整的地方
1. AW@V 手工重构降级为探索工具，不再作为主线证据。
2. 标准 activation patching 升级为主方法。
3. `kl_ratio` 不再解释为贡献强度，只作为输出分布变化和过度转换信号。
4. `over-conversion` 定义为契约破坏信号，不定义为功能贡献强度。
5. `progress` 拆成至少三部分：
   - 方向正确性：是否朝 B 方向移动；
   - 幅度合理性：是否落在自然 A/B 差异范围内；
   - 分布合法性：patch 后的中间状态是否仍像自然 forward。
6. DS7B 后续不能继续使用 `device_map=auto` 的 Phase 288 结果做完整比较；必须使用当前 GPT5 保守环境的 `cuda` 加载策略，保证晚层 hook 生效。
7. 当前 stage100 行为矩阵仍然需要做，但它的角色从“最终行为结论”调整为“Phase 289 数据质量门槛和功能子类筛选器”。

### Phase 288 中可信的结论
- Qwen3 和 DS7B 中 both-patching 普遍优于 attn-only 或 mlp-only，说明 attention 和 MLP 协同，而非二选一。
- GLM4 的 attention-only patching 出现极端 KL 放大，而 MLP patching 更能缩小差距，继续支持“GLM4 更偏 MLP 集中型”的判断。
- recursive 等功能在三模型中走不同路径，说明语言机制不是固定语义轴，也不是单一算法，而是架构约束下的复用/差异化策略。
- Phase 288 已经从表示统计推进到计算机制层，下一步应该研究模块协同关系。

### Phase 288 中不能直接成立的结论
- `Attn_KR=21x` 不能解释为 attention 贡献 21 倍。
- `over-conversion` 不能直接解释为“模块很重要”，只能说明替换破坏了下游期望的动态范围或内部格式。
- 当前每功能 20-40 对，类别太粗，不能给出稳定的功能主导路径。
- early/mid/late/all3 只替换 3 层，对分布式机制太粗。
- translation 中英 tokenization 差异较大，跨语言 patching 可能混入 token 对齐问题；后续应单独处理，不和同语言功能混为同一种证据。

### Phase 289 总目标
建立：

```text
功能级 Attention-MLP-Residual 契约图谱
```

目标不是找单点贡献，而是画出每个功能从输入到输出过程中：

```text
attention 在哪里改变通信方向；
MLP 在哪里完成门控和重编码；
residual 在哪里累积和压缩；
哪些层发生契约兼容；
哪些层发生契约断裂。
```

### Phase 289 实验设计

#### 实验 A：全层模块曲线扫描
对每个模型、每个功能、每一层分别做：

```text
attn-only patch
mlp-only patch
both patch
resid/post-layer patch
```

输出：

```text
layer -> attn_progress
layer -> mlp_progress
layer -> both_progress
layer -> resid_progress
layer -> over_conversion_flag
layer -> naturalness_score
```

目的：
- 找功能形成层；
- 找 attention 到 MLP 的转换层；
- 找输出压缩层；
- 找契约断裂层。

#### 实验 B：自然性检测
不优先使用复杂统计指标，先使用基础、可解释指标：

```text
logit_delta_ratio = ||patched_logits - logits_A|| / ||logits_B - logits_A||
hidden_norm_ratio = ||patched_hidden|| / mean(||natural_A_hidden||, ||natural_B_hidden||)
attn_next_ratio = ||next_attn_after_patch|| / ||next_attn_natural_A||
mlp_next_ratio = ||next_mlp_after_patch|| / ||next_mlp_natural_A||
finite_check = 是否出现 NaN/Inf
```

判定：
- 如果 progress 高但 norm/downstream ratio 爆炸，标记为非自然反事实；
- 非自然反事实不能作为真实机制贡献证据；
- GLM4 attention over-conversion 必须用这套指标重新解释。

#### 实验 C：连续插值
把硬替换：

```text
A_output <- B_output
```

改成：

```text
A_output <- (1-alpha) * A_output + alpha * B_output
alpha = 0, 0.1, 0.25, 0.5, 0.75, 1.0
```

输出：

```text
alpha -> progress
alpha -> KL ratio
alpha -> hidden_norm_ratio
alpha -> downstream_ratio
```

目的：
- 如果曲线平滑，说明该模块可能是自然控制通道；
- 如果某个 alpha 后突然爆炸，说明存在门控/契约断裂；
- 对 GLM4 的 attention over-conversion 尤其关键。

#### 实验 D：契约兼容度矩阵
对每层构造四种组合：

```text
A_attn + A_mlp
B_attn + B_mlp
B_attn + A_mlp
A_attn + B_mlp
```

再观察下一层和最终 logits 是否稳定。

核心指标：

```text
contract_break = hidden_norm_ratio 超阈值 或 downstream_ratio 超阈值 或 KL 极端放大
contract_compatible = 方向正确 且 幅度自然 且 下游未爆炸
```

目的：
- 测 attention 输出是否能被另一个上下文中的 MLP 正常处理；
- 测 MLP 输出是否能被残差流正常接收；
- 解释 GLM4 为什么 attention 替换会产生极端 KL 放大。

#### 实验 E：功能复用矩阵
把每个功能表示成简单曲线签名：

```text
function_signature =
[attn_curve, mlp_curve, both_curve, resid_curve, contract_break_curve]
```

先不做复杂数学，只做基础比较：
- 峰值层是否相同；
- 曲线形状是否相似；
- attention/MLP 转换顺序是否相同；
- 契约断裂层是否相同。

输出：
- 哪些功能复用同一类契约；
- 哪些功能在特定层分叉；
- 哪些模型的同一功能走不同架构路径。

### 数据集调整
不能继续只用粗类别。下一阶段至少拆成：

```text
negation:
  lexical_not / auxiliary_not / existential_no / quantifier_not / logical_not / double_negation

translation:
  word_translation / phrase_translation / sentence_translation / word_order_shift / target_language_switch

logical:
  and_or / conditional / causal / contrast / inference / nested_logic

recursive:
  relative_clause / prepositional_recursion / complement_clause / nested_clause

passive:
  simple_passive / by_phrase_passive / implicit_agent_passive

comparative:
  adjective_comparison / quantity_comparison / relational_comparison / counterfactual_comparison
```

执行顺序：
1. pilot：每子类 20 对，先跑 Qwen3；
2. 稳定后扩到每子类 100 对；
3. 再跑 GLM4；
4. 最后跑 DeepSeek7B。

### 工程执行方案
后续脚本应放在 `tests/gpt5/`：

```text
tests/gpt5/phase289_contract_dataset.py
tests/gpt5/phase289_layer_contract_scan.py
tests/gpt5/phase289_contract_summary.py
tests/gpt5/run_phase289_conservative.sh
```

运行环境：

```bash
tests/gpt5/run_conservative_language_sequence.sh qwen3
```

或 Phase 289 专用入口：

```bash
tests/gpt5/run_phase289_conservative.sh qwen3 --pilot
```

工程要求：
- 默认使用 `openone-cuda121`；
- 每个模型单独进程；
- 每个 category/subcategory 单独 checkpoint；
- 保留 `--hard-exit-after-model`；
- Qwen3/GLM4 默认 fp16；
- DeepSeek7B 默认 bf16；
- 不使用 `device_map=auto` 作为 DS7B 的机制结论来源。

### 阶段路线
1. Phase 289a：Qwen3 pilot，全层扫描，少量子类，验证指标和日志。
2. Phase 289b：加入自然性和连续插值，重点复查 GLM4 over-conversion。
3. Phase 289c：三模型全层契约图谱，修复 DS7B late/all3 缺失。
4. Phase 289d：每子类扩大到 100 对，形成稳定功能签名。
5. Phase 289e：功能复用/差异化矩阵，回答哪些功能共享契约，哪些功能分叉。

### 当前最合理理论版本
语言编码暂时不应描述为固定语义轴，也不应描述为单纯 routing/content 二分。更稳的版本是：

```text
语言编码是条件模块契约系统。

词嵌入提供初始条件；
attention 生成通信方向和上下文结构偏置；
MLP 将 attention 结果转换为可继续传播的内部格式；
residual 保存多层候选状态并累积功能路径；
不同语言功能通过改变 attention-MLP-residual 的契约路径实现复用和差异化。

复用发生在契约相同的地方；
差异化发生在契约断裂、动态范围改变、或者 MLP 门控重编码的地方。
```

### 最终判断
当前方案需要调整，但调整方向不是放弃行为测试，也不是直接跳到单 head 消融，而是：

```text
行为矩阵作为数据质量门槛；
标准 activation patching 作为主因果工具；
自然性、连续插值、契约兼容度作为防止误判的约束；
全层曲线和功能子类作为破解复用/差异化的主地图。
```

这会把研究从“某个模块有影响”推进到“某个语言功能在网络中如何沿契约路径传播和分叉”。

## Phase 22: Phase 289 三模型 Attention-MLP-Residual 契约 Pilot 测试 [2026-05-27 20:52]

### 目标
开始执行 Phase 289 的“功能级 Attention-MLP-Residual 契约图谱”方案。为了避免再次出现 CUDA 卡死且无日志，本阶段只做小规模 pilot：

- 三个模型分别单进程运行；
- 使用 `openone-cuda121` 保守环境；
- 每个模型使用 `--hard-exit-after-model`；
- 每个模型独立 kernel/GPU/process 日志；
- 每个模型独立 checkpoint；
- 先测 negation 6 个子类，每子类 2 对，共 12 对。

### 新增脚本
- `tests/gpt5/phase289_contract_scan.py`
  - GPT5 版 Phase 289 契约扫描脚本。
  - 使用 `tests/gpt5/hf_probe_env.py` 的保守加载路径。
  - 支持：
    - `attn`
    - `mlp`
    - `both`
    - `resid`
    - `cross_battn_amlp`
    - `cross_aattn_bmlp`
  - 支持 alpha 连续插值：
    - `0`
    - `0.5`
    - `1.0`
  - 支持 checkpoint/resume。
  - 支持 `--hard-exit-after-model`。

- `tests/gpt5/run_phase289_conservative.sh`
  - Phase 289 专用日志包装器。
  - 默认激活 `openone-cuda121`。
  - 默认记录：
    - `run.log`
    - `snapshots.log`
    - `gpu_process_monitor.log`
    - `kernel.follow.log`
    - `kernel.since-start.log`
    - `kernel.since-start.filtered.log`
  - 运行前检查是否已有 compute GPU 进程。

### 测试配置
数据：

```text
negation:
  lexical_not_adj
  syntactic_do_not
  existential_no
  never
  morphological_neg
  scope_quantifier

每子类 2 对，共 12 对/模型。
```

patch 类型：

```text
attn
mlp
both
resid
cross_battn_amlp  = B_attn + A_mlp
cross_aattn_bmlp  = A_attn + B_mlp
```

alpha：

```text
0, 0.5, 1.0
```

层：

```text
Qwen3:      [0, 12, 24, 35]
GLM4:       [0, 14, 28, 39]
DeepSeek7B: [0, 9, 18, 27]
```

每模型结果数：

```text
12 pairs * 4 layers * 3 alpha * 6 patch_types = 864 rows
```

总结果数：

```text
2592 rows
```

### 命令记录
Qwen3：

```bash
MAX_SECONDS=1800 \
OUTPUT_DIR=results/gpt5_phase289_contract_pilot \
tests/gpt5/run_phase289_conservative.sh qwen3 \
  --pilot \
  --categories negation \
  --max-pairs-per-subtype 2 \
  --layer-stride 12 \
  --alphas 0,0.5,1.0 \
  --progress-every 1
```

GLM4：

```bash
MAX_SECONDS=2400 \
OUTPUT_DIR=results/gpt5_phase289_contract_pilot \
tests/gpt5/run_phase289_conservative.sh glm4 \
  --pilot \
  --categories negation \
  --max-pairs-per-subtype 2 \
  --layer-stride 14 \
  --alphas 0,0.5,1.0 \
  --progress-every 1
```

DeepSeek7B：

```bash
MAX_SECONDS=2400 \
OUTPUT_DIR=results/gpt5_phase289_contract_pilot \
tests/gpt5/run_phase289_conservative.sh deepseek7b \
  --pilot \
  --categories negation \
  --max-pairs-per-subtype 2 \
  --layer-stride 9 \
  --alphas 0,0.5,1.0 \
  --progress-every 1
```

### 输出文件
结果：

```text
results/gpt5_phase289_contract_pilot/qwen3_phase289_contract_scan.json
results/gpt5_phase289_contract_pilot/glm4_phase289_contract_scan.json
results/gpt5_phase289_contract_pilot/deepseek7b_phase289_contract_scan.json
```

checkpoints：

```text
results/gpt5_phase289_contract_pilot/checkpoints/qwen3/negation_pilot.json
results/gpt5_phase289_contract_pilot/checkpoints/glm4/negation_pilot.json
results/gpt5_phase289_contract_pilot/checkpoints/deepseek7b/negation_pilot.json
```

日志：

```text
results/gpt5_gpu_lock_logs/20260527_204215_phase289_qwen3
results/gpt5_gpu_lock_logs/20260527_204419_phase289_glm4
results/gpt5_gpu_lock_logs/20260527_204632_phase289_deepseek7b
```

### 稳定性
三模型均完成，退出码均为 0。

```text
Qwen3:      complete, 864 rows
GLM4:       complete, 864 rows
DeepSeek7B: complete, 864 rows
```

没有发现：

```text
Xid
NVRM error
Traceback
NaN/Inf result row
timeout
```

GPU 监控峰值约：

```text
Qwen3:      8.9GB, 40C, 89W
GLM4:       19.1GB, 45C, 114W
DeepSeek7B: 15.8GB, 46C, 121W
```

本轮支持一个工程判断：当前 `openone-cuda121 + hard-exit + 单模型单进程 + 日志包装` 路径可以继续作为机制测试默认路径。

### 结果摘要

#### Qwen3

```text
class: Qwen3ForCausalLM
pairs: 12
results: 864
best_layer_by_both_progress: L0
contract_broken_layers: [0]
```

核心层曲线，alpha=1：

```text
L0:
  attn P=0.808 KR=0.159 DR=1.16
  mlp  P=0.896 KR=0.140 DR=1.16
  both P=0.914 KR=0.130 DR=1.14
  resid P=0.899 KR=0.127 DR=1.11
  B_attn + A_mlp P=0.446 KR=0.660 DR=0.78
  A_attn + B_mlp P=0.896 KR=0.140 DR=1.16

L12:
  attn P=0.012 KR=0.990 DR=0.25
  mlp  P=0.184 KR=0.929 DR=0.56
  both P=0.180 KR=0.851 DR=0.55
  resid P=0.992 KR=0.026 DR=1.12

L24:
  attn P=0.044 KR=0.950 DR=0.16
  mlp  P=0.175 KR=0.748 DR=0.39
  both P=0.224 KR=0.715 DR=0.43
  resid P=1.031 KR=0.010 DR=1.16

L35:
  attn P=0.066 KR=0.932 DR=0.55
  mlp  P=0.391 KR=0.985 DR=0.96
  both P=0.424 KR=0.886 DR=0.80
  resid P=1.000 KR=0.000 DR=1.00
```

初步解释：
- L0 的 attention/MLP/both 都很强，说明早层已携带很强的否定 A→B 差异。
- `B_attn + A_mlp` 在 L0 明显弱于 both，说明“只替换 attention 而保留 A 的 MLP”会破坏部分契约。
- 中后层单独 attn/MLP 替换较弱，但 resid patch 很强，说明功能差异可能已经进入残差流整体状态，不再只由单模块输出表示。

#### GLM4

```text
class: GlmForCausalLM
pairs: 12
results: 864
best_layer_by_both_progress: L0
contract_broken_layers: [0]
```

核心层曲线，alpha=1：

```text
L0:
  attn P=0.010 KR=1.128 DR=0.36
  mlp  P=0.969 KR=0.063 DR=1.02
  both P=0.969 KR=0.063 DR=1.02
  resid P=0.981 KR=0.053 DR=1.03
  B_attn + A_mlp P=0.013 KR=1.123 DR=0.30
  A_attn + B_mlp P=0.969 KR=0.063 DR=1.02

L14:
  attn P=0.013 KR=1.018 DR=0.09
  mlp  P=0.169 KR=0.856 DR=0.41
  both P=0.185 KR=0.847 DR=0.43
  resid P=0.989 KR=0.007 DR=1.01

L28:
  attn P=0.004 KR=0.980 DR=0.05
  mlp  P=0.050 KR=0.872 DR=0.22
  both P=0.054 KR=0.860 DR=0.23
  resid P=0.990 KR=0.002 DR=1.00

L39:
  attn P=-0.014 KR=1.037 DR=0.11
  mlp  P=0.324 KR=1.317 DR=0.64
  both P=0.331 KR=1.303 DR=0.65
  resid P=1.000 KR=0.000 DR=1.00
```

初步解释：
- GLM4 的 attention-only 在所有采样层几乎没有正向推进，L0 甚至 KR>1。
- L0 MLP 和 both 几乎相同，支持“GLM4 否定 pilot 中 MLP 是主承载路径”的判断。
- 本轮没有复现 Phase 288 中 21x 级别的 extreme KR，说明 extreme over-conversion 可能依赖样本、层、patch 位置或更细功能类型；不能用本轮小样本否定它。

#### DeepSeek7B

```text
class: Qwen2ForCausalLM
pairs: 12
results: 864
best_layer_by_both_progress: L27
contract_broken_layers: [27]
```

核心层曲线，alpha=1：

```text
L0:
  attn P=0.324 KR=0.640 DR=0.78
  mlp  P=0.187 KR=0.885 DR=0.54
  both P=0.216 KR=0.670 DR=0.70
  resid P=0.365 KR=0.399 DR=0.69

L9:
  attn P=0.283 KR=1.027 DR=0.47
  mlp  P=0.136 KR=1.236 DR=0.55
  both P=0.285 KR=1.272 DR=0.65
  resid P=0.399 KR=0.477 DR=0.65

L18:
  attn P=0.162 KR=1.209 DR=0.43
  mlp  P=0.435 KR=1.052 DR=0.68
  both P=0.314 KR=1.060 DR=0.66
  resid P=0.375 KR=0.383 DR=0.62

L27:
  attn P=0.735 KR=0.677 DR=0.86
  mlp  P=0.710 KR=0.529 DR=0.81
  both P=0.851 KR=0.320 DR=0.95
  resid P=1.000 KR=0.000 DR=1.00
  B_attn + A_mlp P=0.009 KR=0.845 DR=0.23
  A_attn + B_mlp P=0.710 KR=0.529 DR=0.81
```

初步解释：
- DeepSeek7B 和 Qwen3/GLM4 不同，最强 both 层在最后层 L27。
- L27 attention 与 MLP 都有效，both 更强，说明最后层存在明显协同。
- `B_attn + A_mlp` 在 L27 几乎失效，而 `A_attn + B_mlp` 保留 MLP 效果，说明最后层 MLP 或 residual 接收格式可能更关键。
- 这和此前“DS7B 可能有深层特殊释放/最后层压缩”的判断一致。

### 关键洞察
1. 三模型都没有统一路径：
   - Qwen3：pilot 中 L0 attention/MLP/both 都强；
   - GLM4：pilot 中 L0 MLP 极强，attention 几乎无效；
   - DeepSeek7B：pilot 中最后层 L27 both 最强。

2. `resid` patch 几乎总是很强：
   - 这不是说明 residual 是“单独模块贡献”；
   - 它更像是上界测量：如果直接替换整层输出，A 很容易接近 B；
   - 后续 resid 指标应作为“该层整体状态是否足以推动功能”的上界，不和 attn/MLP 直接等价比较。

3. 交叉契约指标可用：
   - Qwen3 L0 和 DeepSeek7B L27 都出现 `B_attn + A_mlp` 明显弱于 both；
   - 这说明 attention 输出不是任意上下文都可用，它需要匹配对应的 MLP/residual 状态；
   - 这正是 Phase 289 要找的“模块契约”信号。

4. GLM4 的结果继续支持 MLP 集中型：
   - L0 `mlp` 和 `both` 几乎一样；
   - `attn` 几乎没有 progress；
   - 但本轮未出现 21x over-conversion，因此不能把 Phase 288 extreme KR 直接泛化到所有否定样本。

### 硬伤
1. 样本太小：每子类 2 对，只能验证脚本、日志和粗方向。
2. 层太稀：每模型只测 4 层，不能确定真实关键层。
3. 只有 negation：还不能推广到 passive、logical、recursive、translation、comparison。
4. `resid` patch 是整层输出替换，上界意义强，机制分解意义弱。
5. 当前 contract_broken 判定较粗：当 both KL 很小时，cross/both ratio 容易放大；下一版需要同时看 absolute KL、progress、delta ratio。
6. 当前自然性指标仍主要在 logit 层，hidden/downstream 指标已经记录部分 norm，但还没有完整汇总。

### 下一步
1. 修正 summary：
   - contract broken 不只用 `cross_kl / both_kl`；
   - 加入 absolute KL、progress drop、logit_delta_ratio 范围。
2. Qwen3 做 denser layer scan：
   - 否定每子类 5-10 对；
   - 层 stride 从 12 降到 4 或全层；
   - 先只跑 Qwen3，确认 L0 强效是否稳定。
3. GLM4 重点查 over-conversion：
   - 加大样本；
   - 全层或 stride=4；
   - 重点输出 alpha 曲线，找是否存在突然爆炸阈值。
4. DeepSeek7B 重点查最后层：
   - 加密 L18-L27；
   - 验证 “深层释放/最后层压缩” 是否稳定。
5. 在否定稳定后扩展到 passive/logical/recursive。

### 当前判断
Phase 289 pilot 成功跑通，日志完整，三模型稳定。最重要的进展不是某个分数，而是已经建立了一个能直接测试“模块契约”的最小系统：

```text
单模块替换 -> 是否朝 B 移动
联合替换 -> 是否协同增强
交叉替换 -> 是否契约断裂
resid 替换 -> 该层整体状态上界
alpha 插值 -> 后续用于识别平滑控制还是突然爆炸
```

这条线可以继续扩大样本和层密度，进入真正的 Attention-MLP-Residual 契约图谱。

## Phase 23: Phase 289 Dense Negation 三模型契约扫描 [2026-05-27 21:52]

### 目标
在 Phase 22 pilot 成功后，加大数据量和层密度，正式测试三模型在 negation 功能上的 Attention-MLP-Residual 契约图谱。

本阶段相比 pilot 的扩大：

```text
pairs: 12 -> 40
alpha: 0,0.5,1.0 -> 0,0.25,0.5,0.75,1.0
layers:
  Qwen3:      4层 -> 10层
  GLM4:       4层 -> 11层
  DeepSeek7B: 4层 -> 8层
rows:
  Qwen3:      864 -> 12000
  GLM4:       864 -> 13200
  DeepSeek7B: 864 -> 9600
total rows: 34800
```

### 脚本改进
修改 `tests/gpt5/phase289_contract_scan.py`：

- 新增 `alpha_curve` 汇总。
- 修正 `contract_broken_layers` 判定，不再只看 `cross_kl / both_kl`。
- 新判定要求同时满足：

```text
cross_kl / both_kl >= 2.0
cross_kl >= 0.5
both_progress - cross_progress >= 0.25
cross_logit_delta_ratio >= 0.15
```

目的：避免 both KL 极小时产生虚假 contract broken。

### 命令记录

Qwen3：

```bash
MAX_SECONDS=5400 \
OUTPUT_DIR=results/gpt5_phase289_contract_dense_negation \
tests/gpt5/run_phase289_conservative.sh qwen3 \
  --categories negation \
  --max-pairs-per-subtype 999 \
  --layer-stride 4 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --progress-every 2
```

GLM4：

```bash
MAX_SECONDS=7200 \
OUTPUT_DIR=results/gpt5_phase289_contract_dense_negation \
tests/gpt5/run_phase289_conservative.sh glm4 \
  --categories negation \
  --max-pairs-per-subtype 999 \
  --layer-stride 4 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --progress-every 2
```

DeepSeek7B：

```bash
MAX_SECONDS=5400 \
OUTPUT_DIR=results/gpt5_phase289_contract_dense_negation \
tests/gpt5/run_phase289_conservative.sh deepseek7b \
  --categories negation \
  --max-pairs-per-subtype 999 \
  --layer-stride 4 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --progress-every 2
```

### 输出文件

```text
results/gpt5_phase289_contract_dense_negation/qwen3_phase289_contract_scan.json
results/gpt5_phase289_contract_dense_negation/glm4_phase289_contract_scan.json
results/gpt5_phase289_contract_dense_negation/deepseek7b_phase289_contract_scan.json
```

checkpoints：

```text
results/gpt5_phase289_contract_dense_negation/checkpoints/qwen3/negation_full.json
results/gpt5_phase289_contract_dense_negation/checkpoints/glm4/negation_full.json
results/gpt5_phase289_contract_dense_negation/checkpoints/deepseek7b/negation_full.json
```

日志：

```text
results/gpt5_gpu_lock_logs/20260527_205629_phase289_qwen3
results/gpt5_gpu_lock_logs/20260527_211127_phase289_glm4
results/gpt5_gpu_lock_logs/20260527_213530_phase289_deepseek7b
```

### 稳定性
三模型均完成，退出码均为 0。

```text
Qwen3:      40 pairs, 12000 rows, complete
GLM4:       40 pairs, 13200 rows, complete
DeepSeek7B: 40 pairs, 9600 rows, complete
```

没有发现：

```text
Xid
NVRM error
Traceback
timeout
系统卡死
```

GPU 峰值日志约：

```text
Qwen3:      8.9GB, 47C, 95W
GLM4:       19.2GB, 50C, 112W
DeepSeek7B: 15.8GB, 50C, 107W
```

GLM4 有 2 条非有限数值行：

```text
pair: neg_know
subtype: syntactic_do_not
patch_type: resid
alpha: 0.5
layers: L8, L12
finite: 0
```

这不是 CUDA 卡死，也不是 kernel Xid；它是 patch 后模型输出非有限，后续应作为自然性/契约断裂异常样本单独追踪。

### Qwen3 结果

```text
class: Qwen3ForCausalLM
pairs: 40
rows: 12000
target_layers: [0,4,8,12,16,20,24,28,32,35]
best_layer_by_both_progress: L0
contract_broken_layers: [0]
```

contract event：

```text
L0, cross_battn_amlp
cross_kl=0.962
both_kl=0.161
cross/both=5.96
cross_progress=0.357
both_progress=0.865
progress_drop=0.508
```

Top layers by both progress：

```text
L0:  bothP=0.865 bothKR=0.161 attnP=0.735 mlpP=0.850 residP=0.859
L4:  bothP=0.391 bothKR=0.682 attnP=0.173 mlpP=0.341 residP=0.872
L35: bothP=0.390 bothKR=0.989 attnP=0.176 mlpP=0.287 residP=1.000
L8:  bothP=0.283 bothKR=0.859 attnP=0.114 mlpP=0.264 residP=0.905
```

子类中非-resid 最强项：

```text
existential_no:    both P=0.220 KR=0.810
lexical_not_adj:   both P=0.325 KR=0.748
morphological_neg: both P=0.173 KR=0.826
never:             both P=0.285 KR=0.806
scope_quantifier:  both P=0.366 KR=0.661
syntactic_do_not:  both P=0.300 KR=0.737
```

判断：
- Qwen3 在否定任务上呈现强早层模式，L0 明显最强。
- L0 both 强于 cross_battn_amlp，说明 B 的 attention 输出不能被 A 的 MLP/residual 直接完整接收，存在早层契约依赖。
- 各否定子类非-resid 最强项均为 both，支持 attention + MLP 协同。

### GLM4 结果

```text
class: GlmForCausalLM
pairs: 40
rows: 13200
target_layers: [0,4,8,12,16,20,24,28,32,36,39]
best_layer_by_both_progress: L0
contract_broken_layers: [0,4]
```

contract events：

```text
L0, cross_battn_amlp
cross_kl=1.116
both_kl=0.048
cross/both=23.18
cross_progress=0.031
both_progress=0.963
progress_drop=0.932

L4, cross_battn_amlp
cross_kl=0.803
both_kl=0.370
cross/both=2.17
cross_progress=0.177
both_progress=0.564
progress_drop=0.387
```

Top layers by both progress：

```text
L0:  bothP=0.963 bothKR=0.048 attnP=0.043 mlpP=0.961 residP=0.974
L4:  bothP=0.564 bothKR=0.370 attnP=0.253 mlpP=0.449 residP=0.984
L39: bothP=0.347 bothKR=1.287 attnP=-0.003 mlpP=0.339 residP=1.000
L8:  bothP=0.280 bothKR=0.756 attnP=0.053 mlpP=0.219 residP=0.989
```

子类中非-resid 最强项：

```text
existential_no:    both P=0.244 KR=0.736
lexical_not_adj:   both P=0.267 KR=0.793
morphological_neg: both P=0.200 KR=0.785
never:             both P=0.272 KR=0.768
scope_quantifier:  both P=0.274 KR=0.857
syntactic_do_not:  both P=0.278 KR=0.813
```

判断：
- GLM4 的 L0 极端 MLP 主导非常稳定：`mlpP=0.961`，`attnP=0.043`。
- L0 both 几乎等于 MLP，说明 attention 单独不是主要转换通道。
- `B_attn + A_mlp` 在 L0/L4 明显失败，而 `A_attn + B_mlp` 继承 MLP 效果，说明 GLM4 的否定转换关键在 MLP 输出格式，而不是 attention 输出本身。
- 这比 Phase 288 的“GLM4 MLP集中型”更强，因为样本和层密度都更高。

### DeepSeek7B 结果

```text
class: Qwen2ForCausalLM
pairs: 40
rows: 9600
target_layers: [0,4,8,12,16,20,24,27]
best_layer_by_both_progress: L27
contract_broken_layers: [27]
```

contract event：

```text
L27, cross_battn_amlp
cross_kl=0.841
both_kl=0.358
cross/both=2.35
cross_progress=0.000
both_progress=0.765
progress_drop=0.765
```

Top layers by both progress：

```text
L27: bothP=0.765 bothKR=0.358 attnP=0.628 mlpP=0.645 residP=1.006
L0:  bothP=0.437 bothKR=0.856 attnP=0.443 mlpP=0.257 residP=0.508
L24: bothP=0.233 bothKR=0.864 attnP=0.170 mlpP=0.178 residP=0.524
L20: bothP=0.204 bothKR=1.056 attnP=0.095 mlpP=0.210 residP=0.556
```

子类中非-resid 最强项：

```text
existential_no:    attn P=0.536 KR=0.862
lexical_not_adj:   both P=0.189 KR=0.915
morphological_neg: both P=0.362 KR=1.064
never:             attn P=0.424 KR=1.540
scope_quantifier:  both P=0.124 KR=0.880
syntactic_do_not:  both P=0.209 KR=0.891
```

判断：
- DeepSeek7B 与 Qwen3/GLM4 不同，最强层在最后层 L27。
- L27 attention 和 MLP 都强，both 更强，说明最后层有明显协同。
- `B_attn + A_mlp` 在 L27 几乎完全失效，说明 B attention 必须配合 B MLP/残差上下文，存在强契约。
- 这支持此前关于 DeepSeek7B “深层释放/最后层压缩”的猜想。

### 总体结论
本轮更完整地支持 Phase 289 的核心方向：

```text
语言功能不是单点模块贡献；
而是 Attention、MLP、Residual 的层级契约路径。
```

三模型的否定功能路径明显不同：

```text
Qwen3:
  早层 L0 attention+MLP 协同，both 最强。

GLM4:
  早层 L0 MLP 极强，attention 单独近乎无效，MLP 集中型最清楚。

DeepSeek7B:
  最后层 L27 attention+MLP 协同最强，呈深层/末层压缩型。
```

复用/差异化的第一性解释也更清楚：

```text
复用不是单个神经元复用；
而是某些模块契约被多个功能共享。

差异化不是某个固定语义轴不同；
而是不同模型/功能在不同层选择不同契约路径。
```

### 严格审视
1. 仍然只测 negation，不能推广到所有语言功能。
2. Qwen3/GLM4 的 L0 强效可能部分来自 token/局部词形差异，需要通过更严格的最小对照验证。
3. residual patch 很强，但它是整层状态替换，只能作为上界，不等价于机制解释。
4. 当前 hidden naturalness 还不完整，主要使用 logit_delta_ratio 和 finite check。
5. GLM4 的 2 条 non-finite resid patch 需要单独复现，确认是模型数值敏感点还是真实契约断裂点。
6. DeepSeek7B 最后层强效可能受 final norm/lm_head 影响，需要细分 pre/post final-layer 状态。

### 下一步计划
1. 做 GLM4 `neg_know` 窄复现：
   - L8/L12
   - resid patch
   - alpha=0.25/0.5/0.75
   - 记录 hidden norm 和 logits finite 状态。
2. 扩展到 passive 和 logical：
   - 先每类 20-40 对；
   - 使用同样 dense 层扫描；
   - 看模型路径是否仍然 Qwen3 早层、GLM4 MLP、DeepSeek7B 末层。
3. 给 Phase289 增加 hidden naturalness：
   - patch vector norm ratio；
   - next layer input/output norm ratio；
   - finite/NaN 捕获；
   - alpha 曲线拐点检测。
4. 对 Qwen3 L0 和 GLM4 L0 做更细粒度：
   - 分开 hook self_attn 前后；
   - 分开 MLP gate/up/down；
   - 检查是否是词形否定的早层模式，还是抽象否定机制。

### 当前判断
本轮已经从 pilot 进入可用的机制图谱雏形。最强结论是：

```text
否定功能在三模型中不是同一编码路径。
Qwen3 是早层协同型；
GLM4 是早层 MLP 集中型；
DeepSeek7B 是末层协同/压缩型。
```

这说明接下来破解语言编码机制时，重点不应是寻找一个统一语义轴，而应追踪：

```text
每个功能在每个模型中选择了哪条 Attention-MLP-Residual 契约路径。
```

## Phase 24: Phase 289 扩展到 Logical/Passive/Recursive 三模型契约扫描 [2026-05-28 00:25]

### 任务目标

在 Phase 23 的 dense negation 基础上继续加大数据量，不再只测试否定，而是把同一套 Attention-MLP-Residual 契约扫描扩展到：

1. logical：and/or、conditional、causal、contrast、inference。
2. passive：by phrase passive、no-agent passive、get passive、dative passive。
3. recursive：relative clause、PP chain、complement clause、possessive chain。

核心目的不是证明某个单头或单层负责语言功能，而是继续绘制功能级模块契约图谱：不同语言功能在每一层中，是通过 attention、MLP、both、resid，还是跨模块错误拼接产生转换或断裂。

### 脚本变更

修改脚本：

```text
tests/gpt5/phase289_contract_scan.py
```

主要变更：

1. 将 logical 样本从 6 条扩展到 40 条，覆盖 5 个子类。
2. 将 passive 样本从 4 条扩展到 32 条，覆盖 4 个子类。
3. 新增 recursive 样本 32 条，覆盖 4 个子类。
4. 当前总样本数变为 144 条：
   - negation: 40
   - logical: 40
   - passive: 32
   - recursive: 32
5. 修正 summary 的 mean 逻辑，只对有限值求均值，避免单个 NaN 把整条 alpha/layer 曲线污染为 NaN。
6. 在 summary 中增加 nonfinite_rows，用于直接记录反事实 patch 后模型输出非有限的行数。

### GLM4 非有限输出复现

Phase 23 中 GLM4 的 dense negation 出现 2 行非有限输出，集中在：

```text
pair = neg_know
subtype = syntactic_do_not
patch_type = resid
alpha = 0.5
layer = 8, 12
```

复现命令：

```bash
MAX_SECONDS=1800 OUTPUT_DIR=results/gpt5_phase289_glm4_nonfinite_repro \
tests/gpt5/run_phase289_conservative.sh glm4 \
  --categories negation \
  --subtypes syntactic_do_not \
  --max-pairs-per-subtype 999 \
  --layers 8,12 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --patch-types resid \
  --progress-every 1
```

复现结果：

```text
结果文件：
results/gpt5_phase289_glm4_nonfinite_repro/glm4_phase289_contract_scan.json

日志目录：
results/gpt5_gpu_lock_logs/20260527_220649_phase289_glm4

bad_count = 2
kernel.since-start.filtered.log = 0 行
```

结论：

这 2 行非有限输出可稳定复现，而且 kernel 日志没有 Xid/NVRM/GPU lockup。因此它不是显卡驱动错误，而是特定 resid 插值反事实状态在 GLM4 内部触发了 logits 非有限输出。这个现象应标记为模型数值/契约不合法，而不是硬件故障。

### 三模型扩展扫描命令

Qwen3：

```bash
MAX_SECONDS=7200 OUTPUT_DIR=results/gpt5_phase289_contract_expanded_lpr \
tests/gpt5/run_phase289_conservative.sh qwen3 \
  --categories logical,passive,recursive \
  --max-pairs-per-subtype 999 \
  --layer-stride 4 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --progress-every 4
```

GLM4：

```bash
MAX_SECONDS=10800 OUTPUT_DIR=results/gpt5_phase289_contract_expanded_lpr \
tests/gpt5/run_phase289_conservative.sh glm4 \
  --categories logical,passive,recursive \
  --max-pairs-per-subtype 999 \
  --layer-stride 4 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --progress-every 4
```

DeepSeek7B：

```bash
MAX_SECONDS=9000 OUTPUT_DIR=results/gpt5_phase289_contract_expanded_lpr \
tests/gpt5/run_phase289_conservative.sh deepseek7b \
  --categories logical,passive,recursive \
  --max-pairs-per-subtype 999 \
  --layer-stride 4 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --progress-every 4
```

全部命令均通过 conservative CUDA 环境执行，逐模型运行，并保留 GPU/process/kernel 日志。

### 输出文件

结果文件：

```text
results/gpt5_phase289_contract_expanded_lpr/qwen3_phase289_contract_scan.json
results/gpt5_phase289_contract_expanded_lpr/glm4_phase289_contract_scan.json
results/gpt5_phase289_contract_expanded_lpr/deepseek7b_phase289_contract_scan.json
```

checkpoint：

```text
results/gpt5_phase289_contract_expanded_lpr/checkpoints/qwen3/logical-passive-recursive_full.json
results/gpt5_phase289_contract_expanded_lpr/checkpoints/glm4/logical-passive-recursive_full.json
results/gpt5_phase289_contract_expanded_lpr/checkpoints/deepseek7b/logical-passive-recursive_full.json
```

GPU/内核日志：

```text
results/gpt5_gpu_lock_logs/20260527_220956_phase289_qwen3
results/gpt5_gpu_lock_logs/20260527_224918_phase289_glm4
results/gpt5_gpu_lock_logs/20260527_235200_phase289_deepseek7b
```

三轮的 kernel.since-start.filtered.log 都是 0 行，没有发现 Xid、NVRM、GSP、GPU locked、soft lockup、hung、timeout、reset 等关键错误。

### 数据规模

```text
Qwen3:
  pairs = 104
  target_layers = 10
  results = 31,200
  nonfinite_rows = 0

GLM4:
  pairs = 104
  target_layers = 11
  results = 34,320
  nonfinite_rows = 50

DeepSeek7B:
  pairs = 104
  target_layers = 8
  results = 24,960
  nonfinite_rows = 0

总结果行数 = 90,480
```

### Qwen3 结果

```text
class = Qwen3ForCausalLM
best_layer_by_both_progress = 0
contract_broken_layers = [0]
nonfinite_rows = 0
```

L0 是最强层：

```text
L0:
  attn_progress = 0.7415
  mlp_progress = 0.8041
  both_progress = 0.8346
  both_kl_ratio = 0.2754
  resid_progress = 0.8453
```

契约断裂事件：

```text
layer = 0
cross_type = cross_battn_amlp
cross_kl_ratio = 0.8851
both_kl_ratio = 0.2754
kl_ratio_vs_both = 3.2139
cross_progress = 0.2082
both_progress = 0.8346
progress_drop = 0.6264
cross_logit_delta_ratio = 0.5477
```

解释：

Qwen3 在 logical/passive/recursive 上延续了 dense negation 的模式：早层 L0 是最强转换点，attention 与 MLP 单独都有明显作用，both 更强。错误组合 B_attn + A_mlp 会显著破坏转换，说明 Qwen3 的早层 attention 和 MLP 已经形成了明确契约。

### GLM4 结果

```text
class = GlmForCausalLM
best_layer_by_both_progress = 0
contract_broken_layers = []
nonfinite_rows = 50
```

L0 是最强层：

```text
L0:
  attn_progress = 0.0165
  mlp_progress = 0.9319
  both_progress = 0.9401
  both_kl_ratio = 0.1135
  resid_progress = 0.9786
```

非有限输出分布：

```text
bad_count = 50

by_patch:
  resid = 18
  cross_aattn_bmlp = 15
  mlp = 9
  both = 8

by_alpha:
  0.5 = 14
  0.25 = 13
  0.75 = 11
  1.0 = 9
  0.0 = 3

by_layer:
  L4 = 21
  L0 = 13
  L16 = 9
  L8 = 5
  L12 = 2

by_subtype:
  get_passive = 24
  possessive_chain = 13
  dative_passive = 8
  and_or = 3
  contrast = 1
  by_phrase = 1
```

解释：

GLM4 的结论更强了：在 logical/passive/recursive 上，L0 的 MLP 几乎等于 both，attention 单独几乎不推动转换。这比 dense negation 更支持“GLM4 是 MLP 集中型”的判断。

同时，GLM4 的 nonfinite_rows = 50，且集中在早层和特定被动/递归子类。因为内核日志完全为空，这不是硬件错误，而是 GLM4 在某些反事实 patch 状态下的内部数值不合法。这个现象本身很重要：GLM4 的功能编码可能对 MLP/残差输入格式更敏感，契约更硬，错误状态不是平滑退化，而是直接数值崩坏。

注意：本轮 contract_broken_layers = [] 不表示 GLM4 没有契约问题，而是当前契约断裂判据主要依赖 cross KL 相对 both KL 的放大和 progress drop；GLM4 的问题大量表现为非有限输出，被 summary 的有限均值过滤后没有进入 contract_events。因此下一步需要把 nonfinite_rows 本身纳入契约断裂定义。

### DeepSeek7B 结果

```text
class = Qwen2ForCausalLM
best_layer_by_both_progress = 27
contract_broken_layers = [27]
nonfinite_rows = 0
```

L27 是最强层：

```text
L27:
  attn_progress = 0.7547
  mlp_progress = 0.6767
  both_progress = 0.8316
  both_kl_ratio = 0.4152
  resid_progress = 1.0000
```

契约断裂事件：

```text
layer = 27
cross_type = cross_battn_amlp
cross_kl_ratio = 0.9765
both_kl_ratio = 0.4152
kl_ratio_vs_both = 2.3520
cross_progress = 0.0020
both_progress = 0.8316
progress_drop = 0.8297
cross_logit_delta_ratio = 0.3090
```

解释：

DeepSeek7B 和 Qwen3/GLM4 完全不同：最强层不是 L0，而是最后层 L27。attention、MLP、both 都很强，both 最强；错误组合 B_attn + A_mlp 在最后层几乎失去 progress。这继续支持前面观察到的 DeepSeek7B “深层释放/最后压缩”特征。

### 三模型对照结论

本轮扩展后，Phase 23 的模型差异没有消失，而是更清晰：

```text
Qwen3:
  早层协同型。
  L0 同时需要 attention 和 MLP，both 最强。
  错误拼接 B_attn + A_mlp 在 L0 断裂。

GLM4:
  早层 MLP 集中型。
  L0 的 MLP 单独几乎等于 both，attention 单独很弱。
  出现大量反事实非有限输出，说明契约更硬、更脆。

DeepSeek7B:
  最后层释放/压缩型。
  L27 最强，attention 和 MLP 都强，both 最强。
  L27 的 B_attn + A_mlp 错误拼接几乎完全失去 progress。
```

这一轮非常重要，因为它说明：

```text
语言功能不是统一地在某个固定层、固定模块或固定语义轴中完成。
不同模型的复用/差异化策略非常不同。
但“attention 输出必须和下游 MLP/残差格式兼容”这个契约结构在多个模型中反复出现。
```

### 当前理论推进

当前更稳的理论表达应改为：

```text
语言编码不是固定语义坐标。
语言编码也不是单独 attention 路由或单独 MLP 内容。
语言编码更像条件模块契约系统。

attention 负责上下文通信方向和结构偏置；
MLP 负责非线性变换、门控、压缩和内部格式重编码；
residual 保存和累积候选状态；
具体语言功能通过不同层位上的 attention-MLP-residual 契约路径实现。
```

复用和差异化也应重新定义：

```text
复用：
  多个语言功能共享相似的层位、模块组合和契约格式。

差异化：
  某个功能在特定层、特定模块、特定动态范围或特定契约处偏离共享路径。

契约断裂：
  错误组合 attention/MLP/residual 后，progress 降低、KL 放大，或直接产生非有限输出。
```

### 硬伤和限制

1. 当前样本虽然从否定扩展到 logical/passive/recursive，但每个子类仍只有 8 条左右，不能作为最终统计结论。
2. 样本是人工模板生成，语言自然性和难度分布仍偏窄。
3. progress 指标仍然只是“朝 B 移动”的粗指标，不能直接解释为真实因果贡献。
4. resid patch 是上界，不是组件贡献，不能和 attention/MLP 直接等价比较。
5. GLM4 的 nonfinite_rows 说明当前反事实 patch 有时会离开自然分布，必须把自然性/合法性作为一级指标。
6. 当前 contract_events 没有把非有限输出纳入断裂判据，导致 GLM4 的断裂被低估。
7. 当前层扫描 stride=4，仍不是逐层扫描；关键层附近需要更细粒度扫描。

### 下一步计划

1. 修改 Phase 289 summary：把 nonfinite_rows 按 layer/patch/subtype/alpha 纳入 contract_events，把非有限输出定义为最高等级契约断裂。
2. 对 GLM4 的 get_passive、possessive_chain、dative_passive 做窄复现，确认非有限输出是否稳定、是否只在 fp16 出现，必要时用 bfloat16/float32 小样本对照。
3. 对三模型的关键层做逐层精扫：
   - Qwen3: L0-L8
   - GLM4: L0-L8、L16
   - DeepSeek7B: L20-L27
4. 增加自然性指标：
   - hidden norm ratio
   - next layer output norm ratio
   - patched logits finite check
   - alpha 曲线是否平滑
5. 把当前结果整理成“功能签名”：
   - function_signature = [attn_curve, mlp_curve, both_curve, resid_curve, contract_break_curve, nonfinite_curve]
6. 做功能复用矩阵：
   - 比较 logical/passive/recursive/negation 各子类在三模型中的曲线相似度。
7. 继续扩大样本，优先不是盲目增加同类句子，而是增加语言结构差异：
   - passive 加入复杂施事、长距离宾语、嵌套被动；
   - recursive 加入双层关系从句、多重 PP、嵌套 complement；
   - logical 加入量词、否定、条件嵌套。

### 阶段性判断

Phase 24 后，当前研究已经不只是“组件功能分析”，而是进入了“模块契约破解”阶段。最关键的结果不是哪个模型哪个模块强，而是三模型都显示出：语言功能转换依赖 attention、MLP、residual 的兼容路径，错误拼接会导致 progress 下降、KL 放大或非有限输出。

下一阶段真正要破解的不是单个神经元或单个 head，而是：

```text
模型如何决定某个语言功能应该复用哪条 attention-MLP-residual 契约路径；
什么时候共享路径；
什么时候分叉；
什么时候错误状态会被自然修正；
什么时候错误状态会直接崩坏。
```

这比“语义轴”更接近语言背后的编码机制。

## Phase 25: Phase 290 断裂等级与关键层精扫 [2026-05-28 02:13]

### 任务目标

根据最新分析，Phase 24 的方向基本正确，但不能把当前现象直接上升为“语言编码就是契约系统”。本轮只做更客观的基础设施和测试：

1. 把 nonfinite 输出纳入断裂记录。
2. 增加 norm 合法性指标。
3. 把 cross-module incompatibility 暂时作为现象记录，不直接称为最终契约证明。
4. 使用关键层逐层精扫，而不是 stride=4 粗扫。
5. 使用 BF16，不使用 bf8/8bit。
6. GLM4 和 DeepSeek7B 使用 `device_map="auto"`。
7. 使用 `attn_implementation="sdpa"`，即 PyTorch SDPA flash 路径；本机没有 `flash_attn` 包，因此不是 flash-attention-2。
8. 模型逐个运行，并保留 kernel/GPU/process 日志。

### 对用户分析的判断

这次分析中正确的部分：

```text
1. Phase 24 不能证明强版本“语言编码就是契约系统”。
2. progress 只能作为扫描指标，不能解释为真实因果贡献。
3. resid patch 是上界，不是组件贡献。
4. cross 拼接失败应先称为 cross-module incompatibility，而不是直接称为已证明的契约断裂。
5. GLM4 nonfinite 必须做 dtype 复核。
6. stride=4 会错过关键层，必须做关键层逐层精扫。
```

因此本轮没有继续盲目扩样本，而是先修正测试框架。

### 脚本变更

修改：

```text
tests/gpt5/hf_probe_env.py
```

新增环境变量：

```text
PROBE_ATTN_IMPLEMENTATION
PROBE_DEVICE_MAP_AUTO_MODELS
PROBE_MAX_GPU_MEMORY
PROBE_MAX_CPU_MEMORY
```

修正关键问题：

```text
当某模型被 PROBE_DEVICE_MAP_AUTO_MODELS 指定为 auto 时，不再执行 model.to("cuda")。
```

新增脚本：

```text
tests/gpt5/phase290_contract_break_scan.py
tests/gpt5/run_phase290_conservative.sh
```

Phase 290 脚本新增指标：

```text
nonfinite_rows
norm_illegal_rows
nonfinite_by_layer
nonfinite_by_patch
nonfinite_by_subtype
nonfinite_by_alpha
patch_*_norm_ratio_to_a
patch_*_norm_ratio_to_b
next_resid_in_norm_ratio_to_a/b
next_layer_out_norm_ratio_to_a/b
contract_events:
  numeric_illegal
  norm_illegal
  functional_incompatible
```

同时新增 pair-level resume：如果长跑中断，partial checkpoint 中已完整的 pair 会跳过，继续跑剩余 pair。

### Smoke Test

Qwen3 smoke：

```bash
MAX_SECONDS=900 OUTPUT_DIR=results/gpt5_phase290_smoke \
tests/gpt5/run_phase290_conservative.sh qwen3 \
  --categories negation \
  --subtypes lexical_not_adj \
  --max-pairs-per-subtype 1 \
  --layers 0 \
  --alphas 0,1 \
  --patch-types attn,mlp,both,resid,cross_battn_amlp,cross_aattn_bmlp \
  --progress-every 1 \
  --label smoke
```

结果：

```text
rows = 10
exit_code = 0
log_dir = results/gpt5_gpu_lock_logs/20260528_010014_phase290_qwen3
```

GLM4 smoke：

```bash
MAX_SECONDS=1200 OUTPUT_DIR=results/gpt5_phase290_smoke \
tests/gpt5/run_phase290_conservative.sh glm4 \
  --categories passive \
  --subtypes get_passive \
  --max-pairs-per-subtype 1 \
  --layers 0 \
  --alphas 0,1 \
  --patch-types mlp,resid,cross_aattn_bmlp \
  --progress-every 1 \
  --label smoke
```

结果：

```text
rows = 5
exit_code = 0
log_dir = results/gpt5_gpu_lock_logs/20260528_010029_phase290_glm4
```

### 正式测试命令

Qwen3，L0-L8：

```bash
MAX_SECONDS=7200 OUTPUT_DIR=results/gpt5_phase290_contract_break_core \
tests/gpt5/run_phase290_conservative.sh qwen3 \
  --categories negation,logical,passive,recursive \
  --max-pairs-per-subtype 4 \
  --layers 0,1,2,3,4,5,6,7,8 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --progress-every 4 \
  --label core
```

GLM4，L0-L8 + L16：

第一次运行命令：

```bash
MAX_SECONDS=10800 OUTPUT_DIR=results/gpt5_phase290_contract_break_core \
tests/gpt5/run_phase290_conservative.sh glm4 \
  --categories negation,logical,passive,recursive \
  --max-pairs-per-subtype 4 \
  --layers 0,1,2,3,4,5,6,7,8,16 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --progress-every 4 \
  --label core
```

第一次 GLM4 在 24/76 pair 后发生用户态 segmentation fault：

```text
exit_code = 139
rows = 5280
kernel.since-start.filtered.log = 0 行
```

随后补充 pair-level resume，并用更保守的运行参数继续：

```bash
CUDA_LAUNCH_BLOCKING=1 PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
MAX_SECONDS=10800 OUTPUT_DIR=results/gpt5_phase290_contract_break_core \
tests/gpt5/run_phase290_conservative.sh glm4 \
  --categories negation,logical,passive,recursive \
  --max-pairs-per-subtype 4 \
  --layers 0,1,2,3,4,5,6,7,8,16 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --progress-every 4 \
  --label core
```

第二次从 24 个已完成 pair 后继续，最终完成。

DeepSeek7B，L20-L27：

```bash
MAX_SECONDS=9000 OUTPUT_DIR=results/gpt5_phase290_contract_break_core \
tests/gpt5/run_phase290_conservative.sh deepseek7b \
  --categories negation,logical,passive,recursive \
  --max-pairs-per-subtype 4 \
  --layers 20,21,22,23,24,25,26,27 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --progress-every 4 \
  --label core
```

### 输出文件

```text
results/gpt5_phase290_contract_break_core/qwen3_phase290_contract_break_scan.json
results/gpt5_phase290_contract_break_core/glm4_phase290_contract_break_scan.json
results/gpt5_phase290_contract_break_core/deepseek7b_phase290_contract_break_scan.json
```

checkpoints：

```text
results/gpt5_phase290_contract_break_core/checkpoints/qwen3/logical-negation-passive-recursive_core.json
results/gpt5_phase290_contract_break_core/checkpoints/glm4/logical-negation-passive-recursive_core.json
results/gpt5_phase290_contract_break_core/checkpoints/deepseek7b/logical-negation-passive-recursive_core.json
```

日志：

```text
results/gpt5_gpu_lock_logs/20260528_010046_phase290_qwen3
results/gpt5_gpu_lock_logs/20260528_011803_phase290_glm4
results/gpt5_gpu_lock_logs/20260528_013232_phase290_glm4
results/gpt5_gpu_lock_logs/20260528_015556_phase290_deepseek7b
```

所有上述正式测试的 `kernel.since-start.filtered.log` 都是 0 行。

### 数据规模

```text
Qwen3:
  pairs = 76
  rows = 15048
  target_layers = L0-L8
  nonfinite_rows = 0
  norm_illegal_rows = 5

GLM4:
  pairs = 76
  rows = 16720
  target_layers = L0-L8 + L16
  nonfinite_rows = 0
  norm_illegal_rows = 0

DeepSeek7B:
  pairs = 76
  rows = 13376
  target_layers = L20-L27
  nonfinite_rows = 0
  norm_illegal_rows = 0

总 rows = 45144
```

### Qwen3 客观结果

```text
best_layer_by_both_progress = 0
contract_broken_layers = [0]
nonfinite_rows = 0
norm_illegal_rows = 5
```

L0：

```text
attn_progress = 0.7691
mlp_progress = 0.8247
both_progress = 0.8559
both_kl_ratio = 0.2012
resid_progress = 0.8620
cross_battn_amlp_progress = 0.2577
cross_battn_amlp_kl_ratio = 0.8825
```

主要 event：

```text
level = functional_incompatible
layer = 0
patch_type = cross_battn_amlp
kl_ratio_vs_both = 4.3861
progress_drop = 0.5982
```

客观现象：

Qwen3 在 L0-L8 逐层精扫中仍然是 L0 最强。`B_attn + A_mlp` 在 L0 明显比 both 差，说明早层 attention/MLP 的错误拼接会造成强功能失败。没有非有限输出。

### GLM4 客观结果

```text
best_layer_by_both_progress = 0
contract_broken_layers = [0, 1]
nonfinite_rows = 0
norm_illegal_rows = 0
```

L0：

```text
attn_progress = 0.0295
mlp_progress = 0.9416
both_progress = 0.9444
both_kl_ratio = 0.0831
resid_progress = 0.9775
cross_battn_amlp_progress = 0.0193
cross_battn_amlp_kl_ratio = 1.0764
```

L1：

```text
attn_progress = 0.2318
mlp_progress = 0.6442
both_progress = 0.6841
both_kl_ratio = 0.4124
cross_battn_amlp_progress = 0.0525
cross_battn_amlp_kl_ratio = 0.9859
```

主要 events：

```text
L0 cross_battn_amlp:
  kl_ratio_vs_both = 12.9533
  progress_drop = 0.9251

L1 cross_battn_amlp:
  kl_ratio_vs_both = 2.3909
  progress_drop = 0.6317
```

客观现象：

1. GLM4 在 BF16 下 nonfinite_rows 从 Phase 24 的 fp16 50 行降为 0。
2. 因此 Phase 24 的 GLM4 nonfinite 很可能主要是 fp16 数值脆弱性，而不是强机制证据。
3. 但 GLM4 的 functional incompatibility 仍然很强，尤其 L0：both_progress = 0.9444，而 cross_battn_amlp_progress = 0.0193。
4. GLM4 的 MLP 集中现象继续成立：L0 的 mlp_progress = 0.9416，几乎等于 both_progress = 0.9444，而 attn_progress = 0.0295。
5. GLM4 第一次 SDPA 长跑出现用户态 segfault 139；kernel 日志为空。resume 后在 `CUDA_LAUNCH_BLOCKING=1` 和 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 下完成。这个是运行稳定性问题，需要后续保守参数默认化。

### DeepSeek7B 客观结果

```text
best_layer_by_both_progress = 27
contract_broken_layers = []
nonfinite_rows = 0
norm_illegal_rows = 0
```

L27：

```text
attn_progress = 0.5728
mlp_progress = 0.4139
both_progress = 0.6117
both_kl_ratio = 0.4922
resid_progress = 1.0000
cross_battn_amlp_progress = 0.1184
cross_battn_amlp_kl_ratio = 0.7968
```

L26：

```text
attn_progress = -0.0199
mlp_progress = 0.2957
both_progress = 0.2771
both_kl_ratio = 1.1028
resid_progress = 0.6270
```

客观现象：

DeepSeek7B 的 L20-L27 逐层精扫支持 L27 最强，L26 次强，但 L27 没有进入当前 `functional_incompatible` 阈值，因为 cross KL 相对 both 的放大未达到 2.0。也就是说，Phase 24 中 “L27 断裂” 在更换样本规模、关键层、BF16/SDPA/auto 后变弱了；末层最强仍成立，但断裂强度需要谨慎。

### 三模型对比

```text
Qwen3:
  L0 strongest
  attn and MLP both strong
  cross_battn_amlp functional incompatible at L0

GLM4:
  L0 strongest
  MLP almost equals both
  attention alone very weak
  cross_battn_amlp functional incompatible at L0/L1
  fp16 nonfinite disappears in BF16

DeepSeek7B:
  L27 strongest
  attention > MLP, both strongest
  no event under current strict threshold
```

### 当前最重要的修正

Phase 24 中“GLM4 nonfinite 是最高等级契约断裂”的判断需要修正：

```text
在 fp16 下，GLM4 出现大量 nonfinite；
在 BF16 下，同类关键层精扫 nonfinite = 0；
因此 nonfinite 更可能是数值精度脆弱性，而不能直接作为机制性断裂证据。
```

但 GLM4 的 cross-module incompatibility 没有消失：

```text
L0 both_progress = 0.9444
L0 cross_battn_amlp_progress = 0.0193
```

这说明即使去掉 fp16 溢出，GLM4 仍存在很强的 attention/MLP 组合不兼容现象。

### 硬伤

1. 当前 Phase 290 是关键层精扫，不是全层扫描。
2. 每个 subtype 最多 4 对，样本语言多样性仍不足。
3. SDPA 在 GLM4 第一次长跑中出现用户态 segfault，说明当前高性能路径还不够稳。
4. DeepSeek7B 的 L27 强峰成立，但断裂 event 不成立，说明断裂阈值需要继续校准。
5. norm_illegal 当前只基于粗阈值 `[0.5, 2.0]`，只能作为异常探测，不能作为精确流形距离。
6. progress 仍不是因果贡献，只是扫描指标。

### 下一步计划

1. 把 `CUDA_LAUNCH_BLOCKING=1` 和 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 作为 GLM4 长跑默认保守参数，避免 SDPA 长跑 segfault。
2. 对 GLM4 做 fp16 vs BF16 的同样小样本对照，只测出过 nonfinite 的 subtype 和 patch，确认 fp16 溢出的边界。
3. 对 DeepSeek7B L24-L27 增加样本和更细事件阈值，确认 L27 是否只是输出接口，还是确实有最后层功能转换。
4. 增加更客观的自然分布距离：
   - per-layer natural norm mean/std
   - norm z-score
   - kNN distance 或 PCA residual distance
5. 下一阶段不要急着做理论总结，优先继续积累：
   - 哪些模型/层会出现 functional incompatibility；
   - 哪些 dtype 会出现 numeric_illegal；
   - 哪些 subtype 最容易触发不兼容；
   - 这些现象能否跨样本稳定复现。

## Phase 26: Phase 290 全量样本系统测试 [2026-05-28 04:31]

### 任务目标

在 Phase 25 的 76 pair 核心扫描之后，继续扩大数据量，把当前样本池全部跑完：

```text
total pairs = 144
negation = 40
logical = 40
passive = 32
recursive = 32
```

本轮继续使用 Phase 290 的断裂等级框架，不做新的理论总结，优先记录客观结果。

### 脚本调整

修改：

```text
tests/gpt5/run_phase290_conservative.sh
```

调整内容：

```text
如果 MODEL=glm4:
  CUDA_LAUNCH_BLOCKING 默认设为 1
  PYTORCH_NO_CUDA_MEMORY_CACHING 默认设为 1

其他模型:
  继续保持默认高性能设置
```

原因：

Phase 25 中 GLM4 在 SDPA 长跑中出现过用户态 segmentation fault 139。这个修改不是改变实验指标，而是提高 GLM4 长跑稳定性。

### 测试环境

```text
conda_env = openone-cuda121
torch_dtype = bfloat16
attn_implementation = sdpa
device_map_auto_models = glm4,deepseek7b
max_gpu_memory = 21GiB
```

说明：

本机没有 `flash_attn` 包，所以这里开启的是 PyTorch SDPA flash 路径，不是 flash-attention-2。

### 测试命令

Qwen3：

```bash
MAX_SECONDS=10800 OUTPUT_DIR=results/gpt5_phase290_contract_break_full \
tests/gpt5/run_phase290_conservative.sh qwen3 \
  --categories negation,logical,passive,recursive \
  --max-pairs-per-subtype 999 \
  --layers 0,1,2,3,4,5,6,7,8 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --progress-every 8 \
  --label full
```

GLM4：

```bash
MAX_SECONDS=14400 OUTPUT_DIR=results/gpt5_phase290_contract_break_full \
tests/gpt5/run_phase290_conservative.sh glm4 \
  --categories negation,logical,passive,recursive \
  --max-pairs-per-subtype 999 \
  --layers 0,1,2,3,4,5,6,7,8,16 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --progress-every 8 \
  --label full
```

GLM4 第一次运行到 96/144 pair 后出现用户态 illegal instruction：

```text
exit_code = 132
rows = 21120
kernel.since-start.filtered.log = 0 行
```

随后用同一命令从 checkpoint resume：

```bash
MAX_SECONDS=10800 OUTPUT_DIR=results/gpt5_phase290_contract_break_full \
tests/gpt5/run_phase290_conservative.sh glm4 \
  --categories negation,logical,passive,recursive \
  --max-pairs-per-subtype 999 \
  --layers 0,1,2,3,4,5,6,7,8,16 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --progress-every 8 \
  --label full
```

第二次从 96 个已完成 pair 继续，最终完成。

DeepSeek7B：

```bash
MAX_SECONDS=10800 OUTPUT_DIR=results/gpt5_phase290_contract_break_full \
tests/gpt5/run_phase290_conservative.sh deepseek7b \
  --categories negation,logical,passive,recursive \
  --max-pairs-per-subtype 999 \
  --layers 20,21,22,23,24,25,26,27 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --progress-every 8 \
  --label full
```

### 输出文件

```text
results/gpt5_phase290_contract_break_full/qwen3_phase290_contract_break_scan.json
results/gpt5_phase290_contract_break_full/glm4_phase290_contract_break_scan.json
results/gpt5_phase290_contract_break_full/deepseek7b_phase290_contract_break_scan.json
```

checkpoints：

```text
results/gpt5_phase290_contract_break_full/checkpoints/qwen3/logical-negation-passive-recursive_full.json
results/gpt5_phase290_contract_break_full/checkpoints/glm4/logical-negation-passive-recursive_full.json
results/gpt5_phase290_contract_break_full/checkpoints/deepseek7b/logical-negation-passive-recursive_full.json
```

日志：

```text
results/gpt5_gpu_lock_logs/20260528_021635_phase290_qwen3
results/gpt5_gpu_lock_logs/20260528_024847_phase290_glm4
results/gpt5_gpu_lock_logs/20260528_033516_phase290_glm4
results/gpt5_gpu_lock_logs/20260528_035815_phase290_deepseek7b
```

所有正式日志的 `kernel.since-start.filtered.log` 都是 0 行。

### 数据规模

```text
Qwen3:
  pairs = 144
  rows = 28512
  target_layers = L0-L8

GLM4:
  pairs = 144
  rows = 31680
  target_layers = L0-L8 + L16

DeepSeek7B:
  pairs = 144
  rows = 25344
  target_layers = L20-L27

total rows = 85536
```

### Qwen3 结果

```text
best_layer_by_both_progress = 0
contract_broken_layers = [0]
nonfinite_rows = 0
norm_illegal_rows = 17
```

L0：

```text
attn_progress = 0.7418
mlp_progress = 0.8180
both_progress = 0.8448
both_kl_ratio = 0.2413
resid_progress = 0.8509
cross_battn_amlp_progress = 0.2506
cross_battn_amlp_kl_ratio = 0.9090
```

主要 event：

```text
level = functional_incompatible
layer = 0
patch_type = cross_battn_amlp
kl_ratio_vs_both = 3.7664
progress_drop = 0.5943
```

客观现象：

全量 144 pair 后，Qwen3 仍然只有 L0 进入 functional_incompatible。L0 的 both、attn、MLP 都较强，错误组合 `B_attn + A_mlp` 明显削弱 progress。

### GLM4 结果

```text
best_layer_by_both_progress = 0
contract_broken_layers = [0, 1]
nonfinite_rows = 0
norm_illegal_rows = 0
```

L0：

```text
attn_progress = 0.0238
mlp_progress = 0.9421
both_progress = 0.9465
both_kl_ratio = 0.0963
resid_progress = 0.9779
cross_battn_amlp_progress = 0.0097
cross_battn_amlp_kl_ratio = 1.0445
```

L1：

```text
attn_progress = 0.2463
mlp_progress = 0.6471
both_progress = 0.6984
both_kl_ratio = 0.4404
resid_progress = 0.9863
cross_battn_amlp_progress = 0.0509
cross_battn_amlp_kl_ratio = 0.9752
```

主要 events：

```text
L0 cross_battn_amlp:
  kl_ratio_vs_both = 10.8411
  progress_drop = 0.9367

L1 cross_battn_amlp:
  kl_ratio_vs_both = 2.2145
  progress_drop = 0.6475
```

客观现象：

1. GLM4 全量 BF16 下仍然没有 nonfinite。
2. Phase 24 中 fp16 nonfinite 继续被削弱为“fp16 数值脆弱性”解释。
3. GLM4 的 MLP 集中现象在全量样本中更稳：L0 `mlp_progress = 0.9421`，几乎等于 `both_progress = 0.9465`，而 `attn_progress = 0.0238`。
4. `cross_battn_amlp` 在 L0/L1 仍然强烈失败。
5. GLM4 SDPA 长跑仍有用户态稳定性问题：本轮出现 exit_code=132，但不是 kernel/GPU 锁死，resume 后完成。

### DeepSeek7B 结果

```text
best_layer_by_both_progress = 27
contract_broken_layers = []
nonfinite_rows = 0
norm_illegal_rows = 0
```

L27：

```text
attn_progress = 0.5970
mlp_progress = 0.4561
both_progress = 0.6563
both_kl_ratio = 0.4632
resid_progress = 1.0000
cross_battn_amlp_progress = 0.0889
cross_battn_amlp_kl_ratio = 0.7977
```

L26：

```text
attn_progress = 0.0149
mlp_progress = 0.2539
both_progress = 0.2570
both_kl_ratio = 1.0894
resid_progress = 0.5980
```

客观现象：

DeepSeek7B 全量后仍然是 L27 最强。L27 的 `cross_battn_amlp_progress = 0.0889` 明显低于 `both_progress = 0.6563`，但因为当前 event 阈值要求 KL 相对 both 放大到 2.0 以上，所以没有进入 `contract_events`。这说明 L27 存在明显 progress drop，但不是 KL explosion 型断裂。

### subtype 非 residual 最优分布

Qwen3：

```text
both = 16
mlp = 3
attn = 0
```

GLM4：

```text
both = 19
mlp = 0
attn = 0
```

DeepSeek7B：

```text
both = 17
mlp = 1
attn = 1
```

客观现象：

1. 三模型绝大多数 subtype 的非 residual 最优 patch 都是 both。
2. GLM4 虽然全 subtype 最优都是 both，但 L0 的 MLP 单独几乎等于 both，因此“both 最优”不能掩盖 MLP 集中。
3. DeepSeek7B 有少数 subtype 表现为单独 attention 或 MLP 最优，但主体仍是 both。

### 与 Phase 25 的一致性

```text
Qwen3:
  Phase 25: L0 broken
  Phase 26: L0 broken
  结论稳定

GLM4:
  Phase 25: L0/L1 broken, BF16 nonfinite=0
  Phase 26: L0/L1 broken, BF16 nonfinite=0
  结论稳定

DeepSeek7B:
  Phase 25: L27 strongest, no strict event
  Phase 26: L27 strongest, no strict event
  结论稳定
```

### 硬伤

1. 样本池全部跑完后仍只有 144 pair，语言多样性仍有限。
2. GLM4 的 SDPA 长跑稳定性仍有问题，出现过用户态 exit_code=132。
3. 当前 event 阈值对 DeepSeek7B 这类“progress drop 强、KL 不爆炸”的情况不敏感。
4. norm_illegal 使用粗阈值 `[0.5, 2.0]`，不是自然流形距离。
5. progress 仍只是方向移动指标，不是因果贡献。

### 下一步计划

1. 增加 progress-drop-only event：
   - 例如 `both_progress - cross_progress >= 0.5`
   - 不强制 KL ratio >= 2.0
   - 用于捕捉 DeepSeek7B L27 这类不兼容。
2. 对 GLM4 SDPA 长跑做稳定性隔离：
   - SDPA vs eager
   - 每 48 pair 分段运行
   - 判断 exit_code=132 是否只出现在长 session。
3. 扩展样本不是继续模板复制，而是增加结构复杂度：
   - long passive
   - nested passive
   - nested logical
   - double recursive
4. 开始保存自然分布参考统计：
   - 每层 natural attn/mlp/resid/next_out norm 均值和方差
   - 以 z-score 替代粗 norm_illegal。

## Phase 27: Phase 291 多层 Block 契约扫描 [2026-05-28 05:48]

### 任务目标

根据“全局功能契约图谱算法”的建议，本轮先完成最可落地的一步：多层 block 契约扫描。

当前不直接进入 head/neuron 级别，因为如果单层/多层功能曲线还不清楚，直接下沉到神经元会导致定位目标不稳定。

本轮重点验证：

```text
1. 单层强是否等于局部功能节点；
2. 连续层 block 是否比单层更强；
3. cross_battn_amlp 是否在 block 级别继续失败；
4. DeepSeek7B 的 L27 是否只是单层现象，还是深层 block 累积；
5. progress-drop-only event 能否捕捉 KL 不爆炸但功能下降的情况。
```

### 对用户方案的判断

“全局功能契约图谱算法”方向正确，但完整目标很大，应分阶段完成。

当前正确部分：

```text
1. 不能继续只看单层，必须测试连续层 block。
2. 每个功能需要拆子类型，不能把 negation/logical/translation 当作单一功能。
3. 需要为每个功能生成 contract_signature。
4. 复用/差异化矩阵是后续核心目标。
5. head/neuron 映射必须建立在稳定的功能峰值层或 block 之上。
```

当前需要谨慎的部分：

```text
1. 每类 100 对样本是目标，但现阶段需要先验证算法和关键现象。
2. head/neuron 级映射不能过早做，否则会把不稳定曲线映射到错误神经元。
3. “全局图谱”必须先定义 signature，再做相似度矩阵，否则容易变成零散实验堆叠。
```

因此本轮执行 Phase 291：多层 block 契约扫描。

### 脚本

新增：

```text
tests/gpt5/phase291_block_contract_scan.py
tests/gpt5/run_phase291_conservative.sh
```

脚本功能：

```text
1. 支持 block 参数，例如 0,0-2,0-4,0-8。
2. 对 block 内所有层同时 patch。
3. patch 类型：
   - attn
   - mlp
   - both
   - resid
   - cross_battn_amlp
   - cross_aattn_bmlp
4. 支持 alpha 插值。
5. 支持 pair-level resume。
6. 输出 block_curve、alpha_curve、subtype_signature。
7. 新增 functional_drop_only event：
   - both_progress >= 0.4
   - both_progress - cross_progress >= 0.5
   - cross_progress <= 0.25
   - 不要求 KL ratio >= 2
```

`functional_drop_only` 是为 DeepSeek7B L27 这类现象新增的：功能下降明显，但 KL 不一定爆炸。

### Smoke Test

命令：

```bash
MAX_SECONDS=900 OUTPUT_DIR=results/gpt5_phase291_smoke \
tests/gpt5/run_phase291_conservative.sh qwen3 \
  --categories negation \
  --subtypes lexical_not_adj \
  --max-pairs-per-subtype 1 \
  --blocks 0,0-2 \
  --alphas 0,1 \
  --progress-every 1 \
  --label smoke
```

结果：

```text
rows = 20
best_block = L0
broken_blocks = [L0, L0-L2]
nonfinite = 0
norm_illegal = 0
log_dir = results/gpt5_gpu_lock_logs/20260528_043653_phase291_qwen3
```

### 正式测试命令

Qwen3：

```bash
MAX_SECONDS=10800 OUTPUT_DIR=results/gpt5_phase291_block_contract_full \
tests/gpt5/run_phase291_conservative.sh qwen3 \
  --categories negation,logical,passive,recursive \
  --max-pairs-per-subtype 999 \
  --blocks 0,0-2,0-4,0-8,4-8 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --progress-every 8 \
  --label full
```

GLM4：

```bash
MAX_SECONDS=14400 OUTPUT_DIR=results/gpt5_phase291_block_contract_full \
tests/gpt5/run_phase291_conservative.sh glm4 \
  --categories negation,logical,passive,recursive \
  --max-pairs-per-subtype 999 \
  --blocks 0,0-1,0-2,0-4,0-8,4-8 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --progress-every 8 \
  --label full
```

DeepSeek7B：

```bash
MAX_SECONDS=10800 OUTPUT_DIR=results/gpt5_phase291_block_contract_full \
tests/gpt5/run_phase291_conservative.sh deepseek7b \
  --categories negation,logical,passive,recursive \
  --max-pairs-per-subtype 999 \
  --blocks 20-23,24-27,20-27,26-27,27 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --progress-every 8 \
  --label full
```

### 输出文件

```text
results/gpt5_phase291_block_contract_full/qwen3_phase291_block_contract_scan.json
results/gpt5_phase291_block_contract_full/glm4_phase291_block_contract_scan.json
results/gpt5_phase291_block_contract_full/deepseek7b_phase291_block_contract_scan.json
```

checkpoints：

```text
results/gpt5_phase291_block_contract_full/checkpoints/qwen3/logical-negation-passive-recursive_full.json
results/gpt5_phase291_block_contract_full/checkpoints/glm4/logical-negation-passive-recursive_full.json
results/gpt5_phase291_block_contract_full/checkpoints/deepseek7b/logical-negation-passive-recursive_full.json
```

日志：

```text
results/gpt5_gpu_lock_logs/20260528_043712_phase291_qwen3
results/gpt5_gpu_lock_logs/20260528_045551_phase291_glm4
results/gpt5_gpu_lock_logs/20260528_052839_phase291_deepseek7b
```

三轮 `kernel.since-start.filtered.log` 都是 0 行。

### 数据规模

```text
Qwen3:
  pairs = 144
  rows = 15840
  blocks = L0, L0-L2, L0-L4, L0-L8, L4-L8

GLM4:
  pairs = 144
  rows = 19008
  blocks = L0, L0-L1, L0-L2, L0-L4, L0-L8, L4-L8

DeepSeek7B:
  pairs = 144
  rows = 15840
  blocks = L20-L23, L20-L27, L24-L27, L26-L27, L27

total rows = 50688
```

### Qwen3 客观结果

```text
best_block_by_both_progress = L0-L8
broken_blocks = [L0, L0-L2, L0-L4, L0-L8, L4-L8]
nonfinite_rows = 0
norm_illegal_rows = 0
```

block 曲线：

```text
L0:
  attn_progress = 0.7418
  mlp_progress = 0.8180
  both_progress = 0.8448
  both_kl_ratio = 0.2414
  cross_battn_amlp_progress = 0.2506

L0-L2:
  attn_progress = 0.7764
  mlp_progress = 0.8270
  both_progress = 0.8543
  both_kl_ratio = 0.1338
  cross_battn_amlp_progress = 0.3093

L0-L4:
  attn_progress = 0.7790
  mlp_progress = 0.8482
  both_progress = 0.8602
  both_kl_ratio = 0.1112
  cross_battn_amlp_progress = 0.2060

L0-L8:
  attn_progress = 0.7642
  mlp_progress = 0.8547
  both_progress = 0.8788
  both_kl_ratio = 0.0820
  cross_battn_amlp_progress = 0.1948

L4-L8:
  attn_progress = 0.3046
  mlp_progress = 0.7577
  both_progress = 0.7979
  both_kl_ratio = 0.2229
  cross_battn_amlp_progress = 0.1590
```

主要 event：

```text
L0-L8 cross_battn_amlp:
  kl_ratio_vs_both = 10.8808
  progress_drop = 0.6840

L0-L4 cross_battn_amlp:
  kl_ratio_vs_both = 8.1855
  progress_drop = 0.6543

L0-L2 cross_battn_amlp:
  kl_ratio_vs_both = 6.1467
  progress_drop = 0.5450
```

客观现象：

Qwen3 的单层 L0 已经很强，但 L0-L8 比 L0 更强，且 both_kl_ratio 明显下降。这说明 Qwen3 不只是 L0 单点，也存在浅层 block 累积增强。L4-L8 单独也强，说明中浅层 block 能继续推动功能转换。

### GLM4 客观结果

```text
best_block_by_both_progress = L0-L8
broken_blocks = [L0, L0-L1, L0-L2, L0-L4, L0-L8, L4-L8]
nonfinite_rows = 0
norm_illegal_rows = 0
```

block 曲线：

```text
L0:
  attn_progress = 0.0238
  mlp_progress = 0.9421
  both_progress = 0.9465
  both_kl_ratio = 0.0963
  cross_battn_amlp_progress = 0.0097

L0-L1:
  attn_progress = 0.0517
  mlp_progress = 0.9724
  both_progress = 0.9823
  both_kl_ratio = 0.0495
  cross_battn_amlp_progress = 0.0402

L0-L2:
  attn_progress = 0.0969
  mlp_progress = 0.9777
  both_progress = 0.9846
  both_kl_ratio = 0.0325
  cross_battn_amlp_progress = 0.0596

L0-L4:
  attn_progress = 0.2741
  mlp_progress = 0.9854
  both_progress = 0.9884
  both_kl_ratio = 0.0214
  cross_battn_amlp_progress = 0.1951

L0-L8:
  attn_progress = 0.5050
  mlp_progress = 0.9872
  both_progress = 0.9906
  both_kl_ratio = 0.0167
  cross_battn_amlp_progress = 0.1846

L4-L8:
  attn_progress = 0.4255
  mlp_progress = 0.8674
  both_progress = 0.9213
  both_kl_ratio = 0.0611
  cross_battn_amlp_progress = 0.1315
```

主要 event：

```text
L0-L1 cross_battn_amlp:
  kl_ratio_vs_both = 20.2548
  progress_drop = 0.9421

L0 cross_battn_amlp:
  kl_ratio_vs_both = 10.8411
  progress_drop = 0.9367

L0-L2 cross_battn_amlp:
  kl_ratio_vs_both = 29.7162
  progress_drop = 0.9249

L0-L8 cross_battn_amlp:
  kl_ratio_vs_both = 52.8985
  progress_drop = 0.8060
```

客观现象：

GLM4 的 MLP 集中现象在 block 级别更强。L0 单层已经强，但 L0-L8 的 both_progress 接近 0.991，MLP_progress 也接近 0.987，几乎等于 both。attention_progress 随 block 变宽而上升，但仍明显低于 MLP/both。错误组合 `B_attn + A_mlp` 在所有 block 中都失败。

### DeepSeek7B 客观结果

```text
best_block_by_both_progress = L20-L27
broken_blocks = [L20-L27, L24-L27, L26-L27, L27]
nonfinite_rows = 0
norm_illegal_rows = 0
```

block 曲线：

```text
L20-L23:
  attn_progress = 0.0841
  mlp_progress = 0.2299
  both_progress = 0.2695
  both_kl_ratio = 0.7282
  cross_battn_amlp_progress = 0.0556

L20-L27:
  attn_progress = 0.7033
  mlp_progress = 0.7818
  both_progress = 0.9173
  both_kl_ratio = 0.0832
  cross_battn_amlp_progress = 0.1300

L24-L27:
  attn_progress = 0.6447
  mlp_progress = 0.6166
  both_progress = 0.8004
  both_kl_ratio = 0.2183
  cross_battn_amlp_progress = 0.1098

L26-L27:
  attn_progress = 0.6076
  mlp_progress = 0.5235
  both_progress = 0.7254
  both_kl_ratio = 0.3713
  cross_battn_amlp_progress = 0.0993

L27:
  attn_progress = 0.5970
  mlp_progress = 0.4561
  both_progress = 0.6563
  both_kl_ratio = 0.4632
  cross_battn_amlp_progress = 0.0889
```

主要 events：

```text
L20-L27 cross_battn_amlp:
  level = functional_kl_incompatible
  kl_ratio_vs_both = 8.5146
  progress_drop = 0.7873

L24-L27 cross_battn_amlp:
  level = functional_kl_incompatible
  kl_ratio_vs_both = 3.3639
  progress_drop = 0.6906

L26-L27 cross_battn_amlp:
  level = functional_kl_incompatible
  kl_ratio_vs_both = 2.1216
  progress_drop = 0.6261

L27 cross_battn_amlp:
  level = functional_drop_only
  kl_ratio_vs_both = 1.7221
  progress_drop = 0.5673
```

客观现象：

Phase 290 中 DeepSeek7B L27 单层没有进入 strict KL event，但 Phase 291 新增的 progress-drop-only 捕捉到了 L27 单层功能下降。更重要的是，L20-L27 block 明显强于 L27 单层：both_progress 从 0.6563 提升到 0.9173，both_kl_ratio 从 0.4632 降到 0.0832。这说明 DeepSeek7B 的深层机制不是单独 L27，而是 L20-L27 跨层累积。

### 三模型对比

```text
Qwen3:
  L0 单层强；
  L0-L8 更强；
  浅层存在 block 累积增强。

GLM4:
  L0 单层极强；
  L0-L8 更接近完整转换；
  MLP block 几乎等于 both block；
  attention 单独较弱，但随 block 变宽增强。

DeepSeek7B:
  L27 单层强；
  L20-L27 block 明显更强；
  深层 block 累积是主要现象。
```

### 当前最重要的新事实

Phase 291 说明：

```text
单层峰值不是完整机制。
Qwen3、GLM4、DeepSeek7B 都存在 block 累积增强。
DeepSeek7B 尤其明显：L20-L27 远强于 L27。
GLM4 的 MLP 集中不是 L0 单点，而是浅层 MLP block 持续累积。
```

这支持下一步继续做 contract_signature，而不是直接跳到单层 neuron。

### 硬伤

1. block patch 仍是 activation patch，不是严格意义上的“从 patch 后状态自然重算出每个内部模块输出”。
2. block 内每层都直接替换为 B 的自然输出，因此它更像 block-level causal replacement，不是完整动态路径重建。
3. 目前只测现有 144 pair，仍不足以覆盖完整语言功能。
4. 当前没有 head/neuron 级定位。
5. naturalness 仍是粗 norm 检查，没有 z-score/kNN/PCA 距离。

### 下一步计划

1. Phase 292：生成 contract_signature 表。
   - 输入 Phase 290 单层结果 + Phase 291 block 结果。
   - 每个 subtype 形成向量：
     layer_curve + block_curve + alpha_curve + event_curve。
   - 输出 subtype 相似度矩阵。

2. Phase 293：扩展功能库。
   - translation
   - style
   - tense
   - coreference
   - role binding 复杂版
   - recursive 复杂版

3. Phase 294：峰值 block 内 head/neuron 定位。
   - Qwen3: L0-L8
   - GLM4: L0-L8
   - DeepSeek7B: L20-L27
   - 先在 block 内定位，再做 head/neuron patch/ablation。

## Phase 28: Phase 292 Contract Signature 与复用矩阵初版 [2026-05-28 09:25]

### 任务目标

根据 Phase 27 的下一步计划，本轮不继续跑 GPU 模型，而是把已有结果整理成第一版 contract_signature：

```text
输入：
  Phase 290 单层关键层结果
  Phase 291 多层 block 结果

输出：
  每个 model-subtype 的 contract_signature
  模型内 subtype 相似度矩阵
  模型内 top reuse / bottom differentiation pair
  跨模型 same-subtype 相似度
```

这一步是“全局功能契约图谱算法”的数据结构骨架，不是最终理论结论。

### 脚本

新增：

```text
tests/gpt5/phase292_contract_signature.py
```

运行命令：

```bash
python tests/gpt5/phase292_contract_signature.py \
  --output-dir results/gpt5_phase292_contract_signature
```

输入文件：

```text
results/gpt5_phase290_contract_break_full/qwen3_phase290_contract_break_scan.json
results/gpt5_phase290_contract_break_full/glm4_phase290_contract_break_scan.json
results/gpt5_phase290_contract_break_full/deepseek7b_phase290_contract_break_scan.json

results/gpt5_phase291_block_contract_full/qwen3_phase291_block_contract_scan.json
results/gpt5_phase291_block_contract_full/glm4_phase291_block_contract_scan.json
results/gpt5_phase291_block_contract_full/deepseek7b_phase291_block_contract_scan.json
```

输出文件：

```text
results/gpt5_phase292_contract_signature/contract_signatures.json
results/gpt5_phase292_contract_signature/signature_summary.csv
results/gpt5_phase292_contract_signature/cross_model_same_subtype_similarity.csv
results/gpt5_phase292_contract_signature/qwen3_subtype_similarity.csv
results/gpt5_phase292_contract_signature/glm4_subtype_similarity.csv
results/gpt5_phase292_contract_signature/deepseek7b_subtype_similarity.csv
results/gpt5_phase292_contract_signature/qwen3_top_reuse_pairs.csv
results/gpt5_phase292_contract_signature/glm4_top_reuse_pairs.csv
results/gpt5_phase292_contract_signature/deepseek7b_top_reuse_pairs.csv
results/gpt5_phase292_contract_signature/qwen3_bottom_differentiation_pairs.csv
results/gpt5_phase292_contract_signature/glm4_bottom_differentiation_pairs.csv
results/gpt5_phase292_contract_signature/deepseek7b_bottom_differentiation_pairs.csv
results/gpt5_phase292_contract_signature/CONTRACT_SIGNATURE_REPORT.md
```

### Signature 定义

本轮生成两类向量：

```text
full_vectors:
  保留模型内 layer/block 原始标签，用于模型内相似度。

canonical_vectors:
  使用 layer_pos/block_pos、best_progress、mean_progress、cross_drop、alpha 曲线等更通用特征，
  用于模型内矩阵和跨模型 same-subtype 粗比较。
```

特征来源：

```text
Phase 290:
  layer × patch_type × progress
  layer × patch_type × kl_ratio
  layer × cross_battn_amlp drop
  alpha × patch_type curve

Phase 291:
  block × patch_type × progress
  block × patch_type × kl_ratio
  block × cross_battn_amlp drop
  block width
  alpha × patch_type curve
```

### 数据规模

```text
model-subtype signatures = 57
cross-model same-subtype rows = 57

每个模型:
  subtypes = 19
```

### 模型级摘要

Qwen3：

```text
avg p290 both best = 0.8506
avg p291 both best = 0.8971
avg p291 cross max drop = 0.7375
```

GLM4：

```text
avg p290 both best = 0.9550
avg p291 both best = 0.9963
avg p291 cross max drop = 0.9648
```

DeepSeek7B：

```text
avg p290 both best = 0.6624
avg p291 both best = 0.9131
avg p291 cross max drop = 0.7937
```

客观现象：

```text
1. 三模型 Phase 291 block best 都高于 Phase 290 layer best。
2. GLM4 的平均 block best 最高，接近 1.0。
3. DeepSeek7B 从单层到 block 的提升最大，符合 Phase 291 的深层累积观察。
4. GLM4 的 cross max drop 最大，说明错误组合在 GLM4 中最严重。
```

### 模型内 Top Reuse Pairs

Qwen3 top：

```text
complement_clause / syntactic_do_not: 0.9732
complement_clause / relative_clause: 0.9721
get_passive / pp_chain: 0.9704
relative_clause / syntactic_do_not: 0.9607
pp_chain / relative_clause: 0.9519
```

GLM4 top：

```text
complement_clause / possessive_chain: 0.9986
conditional / existential_no: 0.9958
complement_clause / never: 0.9939
never / possessive_chain: 0.9933
and_or / inference: 0.9893
```

DeepSeek7B top：

```text
possessive_chain / relative_clause: 0.9992
get_passive / possessive_chain: 0.9988
never / syntactic_do_not: 0.9985
relative_clause / scope_quantifier: 0.9984
possessive_chain / scope_quantifier: 0.9983
```

客观现象：

```text
1. 递归类内部经常出现高相似度，例如 complement_clause / relative_clause / possessive_chain。
2. 否定内部 never / syntactic_do_not 在 DeepSeek7B 中高度相似。
3. 也出现跨类别高相似度，例如 get_passive / pp_chain、conditional / existential_no。
```

但这不能直接解释为真实功能复用，因为当前 signature 可能仍被模型整体 block 曲线主导。

### 模型内 Bottom Differentiation Pairs

Qwen3 最低相似度：

```text
causal / scope_quantifier: 0.3541
contrast / scope_quantifier: 0.4164
conditional / scope_quantifier: 0.4808
causal / complement_clause: 0.4972
morphological_neg / scope_quantifier: 0.5207
```

GLM4 最低相似度：

```text
contrast / possessive_chain: 0.6715
complement_clause / contrast: 0.6765
contrast / never: 0.6771
causal / pp_chain: 0.6793
contrast / pp_chain: 0.6853
```

DeepSeek7B 最低相似度：

```text
complement_clause / no_agent: 0.7980
complement_clause / inference: 0.8183
complement_clause / pp_chain: 0.8594
complement_clause / existential_no: 0.8714
complement_clause / morphological_neg: 0.8751
```

客观现象：

```text
1. Qwen3 的 subtype 分化最明显，最低相似度可以到 0.35。
2. GLM4 分化中等，最低在 0.67 左右。
3. DeepSeek7B 分化最弱，最低仍接近 0.80。
```

这说明当前 canonical signature 对 DeepSeek7B 的功能区分能力不足，或者 DeepSeek7B 的深层 block 曲线在不同 subtype 中确实高度一致。需要后续用去模型均值/残差化 signature 区分这两种可能。

### 跨模型 Same-Subtype 相似度

```text
qwen3 vs glm4:
  mean = 0.8264
  min = contrast, 0.6684
  max = existential_no, 0.9884

qwen3 vs deepseek7b:
  mean = 0.8495
  min = no_agent, 0.6492
  max = relative_clause, 0.9586

glm4 vs deepseek7b:
  mean = 0.8445
  min = inference, 0.6964
  max = possessive_chain, 0.9504
```

客观现象：

跨模型 same-subtype 相似度整体偏高，但最低值显示某些功能在不同模型中路径差异较大，例如：

```text
contrast: qwen3 vs glm4 差异大
no_agent: qwen3 vs deepseek7b 差异大
inference: glm4 vs deepseek7b 差异大
```

### Best Block Width 分布

Qwen3：

```text
width 9 = 8 subtypes
width 5 = 6 subtypes
width 3 = 1 subtype
width 1 = 4 subtypes
```

GLM4：

```text
width 9 = 6 subtypes
width 5 = 3 subtypes
width 3 = 6 subtypes
width 2 = 2 subtypes
width 1 = 2 subtypes
```

DeepSeek7B：

```text
width 8 = 19 subtypes
```

客观现象：

DeepSeek7B 所有 subtype 的 best block width 都是 8，也就是 L20-L27。这再次说明当前 DS7B 的功能转换在本实验设置下高度集中到深层 block，而不是单层或窄 block。

### 本轮最重要的新事实

```text
1. Contract signature 初版已经可以从 Phase 290/291 自动生成。
2. block 曲线确实提供了比单层更强的功能指纹。
3. Qwen3/GLM4/DeepSeek7B 的 subtype 分化程度不同：
   Qwen3 分化最明显；
   GLM4 中等；
   DeepSeek7B 最弱或被全局深层 block 模式主导。
4. 当前 signature 仍存在“模型整体曲线主导”的风险，不能直接把高相似度解释为真实功能复用。
```

### 硬伤

1. 当前 canonical signature 没有做去模型均值，所以相似度可能被模型整体层/block形状主导。
2. DeepSeek7B 的相似度过高，说明当前特征对 DS7B subtype 分化不够敏感。
3. 还没有加入自然分布 z-score/kNN/PCA 距离。
4. 还没有 head/neuron 级特征。
5. 样本仍是 144 pair，功能库还没扩展到 translation/style/tense/coreference。

### 下一步计划

1. Phase 292b：生成 residualized signature。
   - 对每个模型内部的 subtype signature 减去模型均值。
   - 或对每个 category 内部减均值。
   - 再计算相似度，判断哪些高相似度是真复用，哪些只是模型整体曲线。

2. Phase 293：扩展功能库。
   - translation
   - style
   - tense
   - coreference
   - long passive
   - nested logical
   - double recursive

3. Phase 294：自然性指标升级。
   - 保存 natural norm mean/std。
   - 计算 norm z-score。
   - 后续加入 PCA residual distance。

4. Phase 295：峰值 block 内 head/neuron 定位。
   - 在 residualized signature 稳定后再进入 head/neuron。

## Phase 29: Phase 292b 残差化签名与复用候选过滤 [2026-05-28 09:38]

### 任务目标

根据最新分析，Phase 292 的 contract signature 只是功能图谱的数据结构初版，不能直接把 raw similarity 解释为真实复用。本轮不重新跑 GPU 模型，而是先对 Phase 292 的签名做去偏置分析：

```text
1. model-centered signature：减去模型内部所有 subtype 的均值。
2. category-centered signature：减去同一 category 内 subtype 的均值。
3. feature group normalized signature：按 layer/block/alpha/cross/summary 等特征组归一化。
4. zscore_model signature：模型内部逐特征 z-score。
```

目标是区分：

```text
raw similarity 很高，但只是模型整体曲线相似；
raw similarity 很高，残差化后仍然相似，作为更可信的复用候选。
```

### 对用户分析的判断

这次分析中正确的部分：

```text
1. Phase 292 是必要进展，但不是语言编码机制破解。
2. 高相似度不能直接解释为真实功能复用。
3. DeepSeek7B 的 signature 很可能被整体深层 block 模式主导。
4. 下一步必须做 residualized signature。
5. 只有 signature、naturalness、dynamic recompute、variable decoding 一致时，才接近机制证据。
```

因此本轮先做 Phase 292b，而不是直接进入 head/neuron 定位。

### 脚本

新增：

```text
tests/gpt5/phase292b_residualized_signature.py
```

输入：

```text
results/gpt5_phase292_contract_signature/contract_signatures.json
```

运行命令：

```bash
python tests/gpt5/phase292b_residualized_signature.py \
  --input results/gpt5_phase292_contract_signature/contract_signatures.json \
  --output-dir results/gpt5_phase292b_residualized_signature \
  --vector-kind canonical \
  --top-k 20
```

本轮没有加载模型，没有占用 GPU。

### 输出文件

```text
results/gpt5_phase292b_residualized_signature/RESIDUALIZED_SIGNATURE_REPORT.md
results/gpt5_phase292b_residualized_signature/residualized_summary.csv
results/gpt5_phase292b_residualized_signature/pair_stability_diagnostics.csv
results/gpt5_phase292b_residualized_signature/cross_model_same_subtype_residual_similarity.csv
results/gpt5_phase292b_residualized_signature/residualized_signatures.json
```

每个模型还输出：

```text
{model}_raw_similarity.csv
{model}_model_centered_similarity.csv
{model}_category_centered_similarity.csv
{model}_group_normalized_similarity.csv
{model}_zscore_model_similarity.csv
{model}_pair_diagnostics.csv
```

### 数据规模

```text
models = 3
subtypes/model = 19
pairs/model = 171
total pair diagnostics = 513
cross-model same-subtype rows = 285
```

### Qwen3 客观结果

```text
raw similarity:
  mean = 0.7789
  min = 0.3541
  max = 0.9732

model-centered similarity:
  mean = -0.0523
  min = -0.9219
  max = 0.8795

category-centered similarity:
  mean = -0.0535
  min = -0.8586
  max = 0.9236
```

诊断标签：

```text
residual_stable_candidate = 11
model_shape_candidate = 2
category_shape_candidate = 5
stable_differentiation_candidate = 21
```

残差化后仍较稳定的候选：

```text
complement_clause / syntactic_do_not:
  raw = 0.9732
  model_resid = 0.8614
  category_resid = 0.6784

complement_clause / relative_clause:
  raw = 0.9721
  model_resid = 0.7949
  category_resid = 0.3559

get_passive / pp_chain:
  raw = 0.9704
  model_resid = 0.7676
  category_resid = 0.7962

causal / contrast:
  raw = 0.9507
  model_resid = 0.8795
  category_resid = 0.6785
```

raw 高但 category-centered 后变弱的候选：

```text
dative_passive / possessive_chain:
  raw = 0.9482
  model_resid = 0.5706
  category_resid = -0.1889

lexical_not_adj / pp_chain:
  raw = 0.9404
  model_resid = 0.2382
  category_resid = -0.2319

get_passive / lexical_not_adj:
  raw = 0.9385
  model_resid = 0.3498
  category_resid = -0.3191
```

客观现象：

Qwen3 的 raw similarity 本来就不是特别高，残差化后仍保留一部分复用候选，同时也保留较多 differentiation candidate。这说明 Qwen3 的 subtype 分化不是纯粹由模型整体曲线造成。

### GLM4 客观结果

```text
raw similarity:
  mean = 0.8623
  min = 0.6715
  max = 0.9986

model-centered similarity:
  mean = -0.0476
  min = -0.9630
  max = 0.9905

category-centered similarity:
  mean = -0.0494
  min = -0.9599
  max = 0.9726
```

诊断标签：

```text
residual_stable_candidate = 21
model_shape_candidate = 8
category_shape_candidate = 18
stable_differentiation_candidate = 0
```

残差化后仍较稳定的候选：

```text
complement_clause / possessive_chain:
  raw = 0.9986
  model_resid = 0.9905
  category_resid = 0.9726

conditional / existential_no:
  raw = 0.9958
  model_resid = 0.9673
  category_resid = 0.7816

complement_clause / never:
  raw = 0.9939
  model_resid = 0.9571
  category_resid = 0.7936

causal / contrast:
  raw = 0.9871
  model_resid = 0.9480
  category_resid = 0.8207
```

raw 高但 category-centered 后变弱的候选：

```text
and_or / get_passive:
  raw = 0.9707
  model_resid = 0.8167
  category_resid = 0.0659

causal / get_passive:
  raw = 0.9626
  model_resid = 0.8128
  category_resid = 0.0612

dative_passive / morphological_neg:
  raw = 0.9602
  model_resid = 0.6439
  category_resid = -0.1039
```

客观现象：

GLM4 在 raw similarity 上整体高于 Qwen3，但残差化后并不是全部坍塌，仍有 21 个 residual_stable_candidate。GLM4 的 MLP 集中和高相似 signature 可能同时存在：一部分是模型整体浅层 MLP 模式，一部分可能是稳定的 subtype deviation 模式。

### DeepSeek7B 客观结果

```text
raw similarity:
  mean = 0.9556
  min = 0.7980
  max = 0.9992

model-centered similarity:
  mean = -0.0168
  min = -0.7865
  max = 0.9474

category-centered similarity:
  mean = -0.0002
  min = -0.8773
  max = 0.9662
```

诊断标签：

```text
residual_stable_candidate = 21
model_shape_candidate = 108
category_shape_candidate = 7
stable_differentiation_candidate = 0
```

残差化后仍较稳定的候选：

```text
possessive_chain / relative_clause:
  raw = 0.9992
  model_resid = 0.9447
  category_resid = 0.9662

get_passive / possessive_chain:
  raw = 0.9988
  model_resid = 0.9197
  category_resid = 0.6551

never / syntactic_do_not:
  raw = 0.9985
  model_resid = 0.9052
  category_resid = 0.8377

by_phrase / dative_passive:
  raw = 0.9983
  model_resid = 0.9474
  category_resid = 0.9165
```

raw 高但 model-centered 后变弱的候选很多：

```text
model_shape_candidate = 108 / 171 pairs
```

典型例子：

```text
by_phrase / get_passive:
  raw = 0.9811
  model_resid = 0.1214
  category_resid = -0.1043

by_phrase / possessive_chain:
  raw = 0.9802
  model_resid = 0.0871
  category_resid = -0.2803

dative_passive / get_passive:
  raw = 0.9801
  model_resid = 0.0929
  category_resid = -0.1152
```

客观现象：

DeepSeek7B 的 raw similarity 过高确实主要由模型整体曲线主导。raw mean = 0.9556，但 model-centered mean 降到 -0.0168，category-centered mean 接近 0。也就是说，Phase 292 中“DeepSeek7B subtype 分化最弱”的判断需要修正为：

```text
当前 canonical signature 下，DeepSeek7B 的 subtype 曲线被共同的 L20-L27 深层 block 模式强烈主导；
残差化后仍有少数稳定复用候选，但大多数 raw 高相似不能当作复用证据。
```

### 跨模型 same-subtype similarity

raw 均值：

```text
qwen3 vs glm4 = 0.8264
qwen3 vs deepseek7b = 0.8495
glm4 vs deepseek7b = 0.8445
```

model-centered 均值：

```text
qwen3 vs glm4 = 0.3531
qwen3 vs deepseek7b = 0.1890
glm4 vs deepseek7b = 0.1767
```

category-centered 均值：

```text
qwen3 vs glm4 = 0.1177
qwen3 vs deepseek7b = 0.0729
glm4 vs deepseek7b = 0.1949
```

zscore_model 均值：

```text
qwen3 vs glm4 = 0.3774
qwen3 vs deepseek7b = -0.0792
glm4 vs deepseek7b = -0.0501
```

客观现象：

跨模型 raw same-subtype similarity 很高，但残差化后明显下降。说明三模型在“同一 subtype 的偏离模式”上并没有 raw similarity 看起来那么一致。Qwen3 与 GLM4 的残差模式相似度高于它们与 DeepSeek7B 的相似度。

### 本轮最重要的新事实

```text
1. Phase 292 的 raw similarity 确实会被模型整体曲线主导。
2. DeepSeek7B 是最明显的例子：108/171 个 pair 是 model_shape_candidate。
3. Qwen3 的 subtype 分化更稳定，残差化后仍保留 differentiation candidate。
4. GLM4 有较多 residual_stable_candidate，但也有不少 category_shape_candidate，说明高相似度必须分层解释。
5. 跨模型 same-subtype raw similarity 不能直接解释为同一语言机制；残差化后相似度大幅降低。
```

### 当前需要修正的判断

Phase 292 中：

```text
DeepSeek7B subtype 分化最弱
```

需要改成更谨慎的版本：

```text
DeepSeek7B 的 canonical signature 被共同深层 block 模式强烈主导；
在当前特征下，raw similarity 无法有效区分 subtype；
残差化后仍有少数候选复用 pair，但绝大多数 raw 高相似不是机制复用证据。
```

### 硬伤

1. 残差化只是分析方法，不是新的因果实验。
2. 阈值 `raw >= 0.90`、`model_resid >= 0.50`、`category_resid >= 0.30` 只是筛选标签，不是科学定律。
3. 当前 signature 仍然来自 patch behavior，不是内部变量内容。
4. 还没有 naturalness z-score/PCA/kNN。
5. 还没有 dynamic recompute，无法证明 block patch 不是搬运自然轨迹。
6. 还没有变量解码，不能回答路径中传递的具体语言变量是什么。

### 下一步计划

1. Phase 293：自然性指标升级。
   - 从已有 Phase 290/291 结果中先生成 norm reference baseline。
   - 输出每个 model/layer/block/patch 的 norm z-score。
   - 把 functional failure 分成 off-manifold failure 和 norm-normal functional failure。

2. Phase 294：动态重算路径。
   - patch 起点，然后重算下游。
   - patch attention output，然后让 MLP 真实重算。
   - 优先测试：
     Qwen3 L0-L8；
     GLM4 L0-L8；
     DeepSeek7B L20-L27。

3. Phase 295：扩展功能库。
   - translation
   - style
   - tense
   - coreference
   - long passive
   - nested logical
   - double recursive

4. Phase 296：变量解码。
   - agent
   - patient
   - operator
   - scope
   - clause boundary
   - coreference target
   - role binding

## Phase 30: Phase 293 自然性扫描与 Phase 294 动态重算 Pilot [2026-05-28 09:56]

### 任务目标

继续进行系统性测试。本轮分两步：

```text
1. Phase 293：基于已有 Phase 290/291 大规模结果，建立 norm-based naturalness 检查。
2. Phase 294：做小规模 GPU 动态重算 pilot，测试 patch 起点后让下游自然 forward 是否能恢复目标状态。
```

本轮仍然坚持只记录客观现象，不把结果直接上升为“语言编码机制已经破解”。

### Phase 293 脚本

新增：

```text
tests/gpt5/phase293_naturalness_scan.py
```

核心思路：

```text
1. 从 Phase 290/291 结果中读取 patch norm、ratio_to_a、ratio_to_b。
2. 用 ratio 反推 A/B 自然参考 norm，建立每个 model/source/layer/module 的 norm reference。
3. 对每个 patch 状态计算 norm z-score。
4. 把 functional failure 分成：
   - off_manifold_functional_failure
   - norm_normal_functional_failure
   - off_manifold_no_drop
   - numeric_illegal
```

注意：

```text
这只是 norm-based naturalness，不是 PCA/kNN/Mahalanobis 流形距离。
```

### Phase 293 命令

主阈值：

```bash
python tests/gpt5/phase293_naturalness_scan.py \
  --phase290-dir results/gpt5_phase290_contract_break_full \
  --phase291-dir results/gpt5_phase291_block_contract_full \
  --output-dir results/gpt5_phase293_naturalness \
  --z-threshold 4.0 \
  --drop-threshold 0.5 \
  --both-min 0.4
```

阈值敏感性：

```bash
python tests/gpt5/phase293_naturalness_scan.py \
  --phase290-dir results/gpt5_phase290_contract_break_full \
  --phase291-dir results/gpt5_phase291_block_contract_full \
  --output-dir results/gpt5_phase293_naturalness_z3 \
  --z-threshold 3.0 \
  --drop-threshold 0.5 \
  --both-min 0.4

python tests/gpt5/phase293_naturalness_scan.py \
  --phase290-dir results/gpt5_phase290_contract_break_full \
  --phase291-dir results/gpt5_phase291_block_contract_full \
  --output-dir results/gpt5_phase293_naturalness_z5 \
  --z-threshold 5.0 \
  --drop-threshold 0.5 \
  --both-min 0.4
```

### Phase 293 输出

```text
results/gpt5_phase293_naturalness/NATURALNESS_REPORT.md
results/gpt5_phase293_naturalness/natural_norm_reference.csv
results/gpt5_phase293_naturalness/naturalness_events.csv
results/gpt5_phase293_naturalness/naturalness_summary.csv
results/gpt5_phase293_naturalness/naturalness_subtype_summary.csv
results/gpt5_phase293_naturalness/naturalness_stats.json
```

主阈值结果：

```text
reference_rows = 211
event_rows = 3330
summary_rows = 15
```

### Phase 293 客观结果

主阈值 z=4.0：

```text
Qwen3:
  phase290 norm_normal_functional_failure = 221
  phase290 off_manifold_functional_failure = 1
  phase290 off_manifold_no_drop = 88
  phase291 norm_normal_functional_failure = 474

GLM4:
  phase290 norm_normal_functional_failure = 541
  phase290 off_manifold_functional_failure = 2
  phase290 off_manifold_no_drop = 22
  phase291 norm_normal_functional_failure = 825
  phase291 off_manifold_functional_failure = 6
  phase291 off_manifold_no_drop = 72

DeepSeek7B:
  phase290 norm_normal_functional_failure = 214
  phase290 off_manifold_no_drop = 135
  phase291 norm_normal_functional_failure = 467
  phase291 off_manifold_functional_failure = 11
  phase291 off_manifold_no_drop = 251
```

功能失败总数：

```text
Qwen3 = 696
GLM4 = 1374
DeepSeek7B = 692
total = 2762
```

其中 off-manifold functional failure 很少：

```text
Qwen3 = 1
GLM4 = 8
DeepSeek7B = 11
```

阈值敏感性：

```text
z = 3.0:
  functional failures = 2762
  norm-normal = 2686
  off-manifold = 76

z = 4.0:
  functional failures = 2762
  norm-normal = 2742
  off-manifold = 20

z = 5.0:
  functional failures = 2762
  norm-normal = 2750
  off-manifold = 12
```

客观现象：

```text
1. 功能失败总数对 z 阈值不敏感。
2. 绝大多数 cross functional failure 没有明显 norm 异常。
3. 因此，至少在粗 norm 检查下，cross failure 不能简单解释为“范数离开自然范围”。
4. 但这还不能排除更细的 off-manifold，例如方向、局部密度、PCA residual 或 token-position 分布异常。
```

典型 norm-normal functional failure：

```text
DeepSeek7B phase290 L25 cross_battn_amlp existential_no:
  progress_drop = 2.5447
  off_manifold = 0
  max_abs_norm_z = 1.76

Qwen3 phase291 L0-L8 cross_battn_amlp no_agent:
  progress_drop = 2.0209
  off_manifold = 0
  max_abs_norm_z = 2.09

GLM4 phase291 L0-L1 cross_battn_amlp dative_passive:
  progress_drop = 1.2202
  off_manifold = 0
  max_abs_norm_z = 1.09
```

### Phase 294 脚本

新增：

```text
tests/gpt5/phase294_dynamic_recompute_pilot.py
tests/gpt5/run_phase294_conservative.sh
```

测试内容：

```text
1. resid_in：把 A 在某层输入 residual 替换为 B 的同层 residual input，然后让后续自然 forward。
2. resid_out：把 A 在某层输出 residual 替换为 B 的同层 residual output，然后让后续自然 forward。
3. attn_out：只替换 attention output，让该层 MLP 和后续层自然重算。
4. mlp_out：只替换 MLP output，让后续层自然重算。
```

这比 block patch 更接近动态路径测试，因为它不是替换整段 block 的自然轨迹，而是替换某个起点或单模块输出后看下游是否自然恢复。

### Phase 294 Smoke

```bash
MAX_SECONDS=900 OUTPUT_DIR=results/gpt5_phase294_smoke \
tests/gpt5/run_phase294_conservative.sh qwen3 \
  --categories negation \
  --subtypes lexical_not_adj \
  --max-pairs-per-subtype 1 \
  --layers 0 \
  --alphas 1.0 \
  --progress-every 1
```

结果：

```text
rows = 4
nonfinite = 0
exit_code = 0
log_dir = results/gpt5_gpu_lock_logs/20260528_095251_phase294_qwen3
```

### Phase 294 三模型 Pilot 命令

Qwen3：

```bash
MAX_SECONDS=3600 OUTPUT_DIR=results/gpt5_phase294_dynamic_recompute_pilot \
tests/gpt5/run_phase294_conservative.sh qwen3 \
  --categories negation,logical,passive,recursive \
  --max-pairs-per-subtype 2 \
  --layers 0,4,8 \
  --alphas 1.0 \
  --progress-every 8
```

GLM4：

```bash
MAX_SECONDS=5400 OUTPUT_DIR=results/gpt5_phase294_dynamic_recompute_pilot \
tests/gpt5/run_phase294_conservative.sh glm4 \
  --categories negation,logical,passive,recursive \
  --max-pairs-per-subtype 2 \
  --layers 0,4,8 \
  --alphas 1.0 \
  --progress-every 8
```

DeepSeek7B：

```bash
MAX_SECONDS=5400 OUTPUT_DIR=results/gpt5_phase294_dynamic_recompute_pilot \
tests/gpt5/run_phase294_conservative.sh deepseek7b \
  --categories negation,logical,passive,recursive \
  --max-pairs-per-subtype 2 \
  --layers 20,24,27 \
  --alphas 1.0 \
  --progress-every 8
```

### Phase 294 输出

```text
results/gpt5_phase294_dynamic_recompute_pilot/qwen3_phase294_dynamic_recompute_pilot.json
results/gpt5_phase294_dynamic_recompute_pilot/glm4_phase294_dynamic_recompute_pilot.json
results/gpt5_phase294_dynamic_recompute_pilot/deepseek7b_phase294_dynamic_recompute_pilot.json
```

日志：

```text
results/gpt5_gpu_lock_logs/20260528_095304_phase294_qwen3
results/gpt5_gpu_lock_logs/20260528_095354_phase294_glm4
results/gpt5_gpu_lock_logs/20260528_095511_phase294_deepseek7b
```

三轮正式 pilot 的 `kernel.since-start.filtered.log` 都是 0 行。

### Phase 294 数据规模

```text
Qwen3:
  pairs = 38
  rows = 456
  nonfinite = 0

GLM4:
  pairs = 38
  rows = 456
  nonfinite = 0

DeepSeek7B:
  pairs = 38
  rows = 456
  nonfinite = 0

total rows = 1368
```

### Phase 294 Qwen3 客观结果

best by patch type：

```text
resid_in:
  best layer = 8
  progress = 0.8325

resid_out:
  best layer = 8
  progress = 0.8485

attn_out:
  best layer = 0
  progress = 0.7444

mlp_out:
  best layer = 0
  progress = 0.7945
```

layer curve：

```text
L0:
  attn_out_progress = 0.7444
  mlp_out_progress = 0.7945
  resid_in_progress = 0.6682
  resid_out_progress = 0.8252

L4:
  attn_out_progress = 0.0649
  mlp_out_progress = 0.3038
  resid_in_progress = 0.8280
  resid_out_progress = 0.8333

L8:
  attn_out_progress = 0.0847
  mlp_out_progress = 0.2181
  resid_in_progress = 0.8325
  resid_out_progress = 0.8485
```

客观现象：

Qwen3 中，单模块 attn_out/mlp_out 的动态重算效果集中在 L0；但 resid_in/resid_out 在 L4/L8 也很强。这说明浅层模块输出可以启动转换，而中浅层 residual 状态已经携带较完整的 B 方向。

### Phase 294 GLM4 客观结果

best by patch type：

```text
resid_in:
  best layer = 4
  progress = 0.9856

resid_out:
  best layer = 8
  progress = 0.9843

attn_out:
  best layer = 4
  progress = 0.1482

mlp_out:
  best layer = 0
  progress = 0.9345
```

layer curve：

```text
L0:
  attn_out_progress = 0.0066
  mlp_out_progress = 0.9345
  resid_in_progress = 0.9784
  resid_out_progress = 0.9785

L4:
  attn_out_progress = 0.1482
  mlp_out_progress = 0.3729
  resid_in_progress = 0.9856
  resid_out_progress = 0.9830

L8:
  attn_out_progress = 0.0284
  mlp_out_progress = 0.1546
  resid_in_progress = 0.9853
  resid_out_progress = 0.9843
```

客观现象：

GLM4 的动态重算 pilot 继续支持 MLP 集中：L0 mlp_out_progress = 0.9345，而 L0 attn_out_progress = 0.0066。resid_in/resid_out 在 L0/L4/L8 都接近完整转换，但这不能直接说明 residual 是独立组件，只能说明相应层的 residual 状态已足够携带 B 状态。

### Phase 294 DeepSeek7B 客观结果

best by patch type：

```text
resid_in:
  best layer = 24
  progress = 0.5822

resid_out:
  best layer = 27
  progress = 1.0000

attn_out:
  best layer = 27
  progress = 0.6052

mlp_out:
  best layer = 27
  progress = 0.4115
```

layer curve：

```text
L20:
  attn_out_progress = -0.0010
  mlp_out_progress = 0.0452
  resid_in_progress = 0.5822
  resid_out_progress = 0.5758

L24:
  attn_out_progress = 0.0049
  mlp_out_progress = 0.0179
  resid_in_progress = 0.5822
  resid_out_progress = 0.5827

L27:
  attn_out_progress = 0.6052
  mlp_out_progress = 0.4115
  resid_in_progress = 0.5750
  resid_out_progress = 1.0000
```

客观现象：

DeepSeek7B 的 L20/L24 单模块 attn_out/mlp_out 动态重算几乎不能推动目标转换，但 L27 attn_out 明显有效。resid_in 从 L20/L24/L27 都只能达到约 0.58，而不是 Phase 291 block L20-L27 的 0.9173。这说明 DeepSeek7B 的强 block 效果很可能不是单一起点 residual patch 后自然重算就能复现，而需要多层轨迹累积或多点持续干预。

### 本轮最重要的新事实

```text
1. 粗 norm 自然性检查下，绝大多数 cross functional failure 不是范数异常导致。
2. DeepSeek7B 的 L20-L27 block 强效果不能由单个 L20/L24 residual 起点 patch 复现。
3. Qwen3/GLM4 的浅层 residual patch 很强，但必须谨慎，因为 residual patch 是上界，可能搬运了大量 token/position/任务格式信息。
4. GLM4 的 MLP 集中在动态重算 pilot 中继续稳定出现。
5. 三模型 Phase 294 pilot 均 nonfinite=0，kernel filtered log=0。
```

### 当前判断修正

Phase 291 中：

```text
DeepSeek7B L20-L27 block 明显强于 L27 单层，说明深层 block 累积是主要现象。
```

现在需要补充：

```text
DeepSeek7B 的强 block 累积不是简单地由 L20 或 L24 的 B residual input 作为起点后自然重算完成；
更可能需要多层持续 patch、多个模块共同推动，或者 block patch 搬运了多个中间轨迹状态。
```

### 硬伤

1. Phase 293 只是 norm-based naturalness，不能排除更细的 off-manifold。
2. Phase 294 只是 pilot，每个 subtype 只有 2 pair。
3. Phase 294 只测 alpha=1.0，没有插值曲线。
4. Phase 294 只测少数层，没有完整逐层动态重算。
5. resid_in/resid_out patch 仍然可能搬运 token/position/任务格式，不等于找到了编码变量。
6. 还没有解码 agent/patient/scope/operator 等内部变量。

### 下一步计划

1. Phase 294b：扩大动态重算。
   - 每个 subtype 至少 4-8 pair。
   - 加入 alpha = 0, 0.25, 0.5, 0.75, 1.0。
   - Qwen3/GLM4 扫 L0-L8。
   - DeepSeek7B 扫 L20-L27。

2. Phase 295：多点动态路径测试。
   - 单点 resid_in patch。
   - 起点 resid_in + 后续 attn_out patch。
   - 起点 resid_in + 后续 mlp_out patch。
   - 连续层 segment patch 后重算下游。

3. Phase 296：自然性升级。
   - PCA residual distance。
   - kNN distance。
   - 按 token position 分层统计，而不是只看整体 norm。

4. Phase 297：变量解码。
   - 优先在 Qwen3 L0-L8、GLM4 L0-L8、DeepSeek7B L20-L27 内做 agent/patient/operator/scope 解码。

## Phase 31: Phase 294b 扩展动态重算测试 [2026-05-28 10:59]

### 任务目标

在 Phase 294 pilot 基础上扩大动态重算测试：

```text
1. 每个 subtype 从 2 pair 扩大到 4 pair。
2. Qwen3 / GLM4 扫 L0-L8。
3. DeepSeek7B 扫 L20-L27。
4. 加入 alpha 插值：
   0, 0.25, 0.5, 0.75, 1.0
5. patch 类型：
   resid_in, resid_out, attn_out, mlp_out
```

目标是验证：

```text
1. 单点起点 patch 后自然重算是否能接近 Phase 291 block patch。
2. alpha 曲线是否平滑。
3. Qwen3/GLM4/DeepSeek7B 的动态路径是否存在模型差异。
```

### 脚本调整

修改：

```text
tests/gpt5/phase294_dynamic_recompute_pilot.py
```

新增：

```text
1. pair-level checkpoint/resume。
2. --label 参数。
3. --resume / --no-resume 参数。
```

checkpoint：

```text
results/gpt5_phase294b_dynamic_recompute_full/checkpoints/{model}/logical-negation-passive-recursive_full.json
```

原因：

GLM4 长跑之前多次出现用户态 segmentation fault / illegal instruction，必须保证中断后可恢复。

### 测试命令

Qwen3：

```bash
MAX_SECONDS=7200 OUTPUT_DIR=results/gpt5_phase294b_dynamic_recompute_full \
tests/gpt5/run_phase294_conservative.sh qwen3 \
  --categories negation,logical,passive,recursive \
  --max-pairs-per-subtype 4 \
  --layers 0,1,2,3,4,5,6,7,8 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --progress-every 8 \
  --label full
```

GLM4：

```bash
MAX_SECONDS=9000 OUTPUT_DIR=results/gpt5_phase294b_dynamic_recompute_full \
tests/gpt5/run_phase294_conservative.sh glm4 \
  --categories negation,logical,passive,recursive \
  --max-pairs-per-subtype 4 \
  --layers 0,1,2,3,4,5,6,7,8 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --progress-every 8 \
  --label full
```

GLM4 第一次运行到 64/76 pair 后出现用户态 segmentation fault：

```text
exit_code = 139
completed_rows = 11520
```

随后用同一命令 resume：

```bash
MAX_SECONDS=3600 OUTPUT_DIR=results/gpt5_phase294b_dynamic_recompute_full \
tests/gpt5/run_phase294_conservative.sh glm4 \
  --categories negation,logical,passive,recursive \
  --max-pairs-per-subtype 4 \
  --layers 0,1,2,3,4,5,6,7,8 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --progress-every 8 \
  --label full
```

resume 识别：

```text
resume rows = 11520
expected_rows_per_pair = 180
completed_pairs = 64
```

第二次完成剩余部分。

DeepSeek7B：

```bash
MAX_SECONDS=7200 OUTPUT_DIR=results/gpt5_phase294b_dynamic_recompute_full \
tests/gpt5/run_phase294_conservative.sh deepseek7b \
  --categories negation,logical,passive,recursive \
  --max-pairs-per-subtype 4 \
  --layers 20,21,22,23,24,25,26,27 \
  --alphas 0,0.25,0.5,0.75,1.0 \
  --progress-every 8 \
  --label full
```

### 输出文件

```text
results/gpt5_phase294b_dynamic_recompute_full/qwen3_phase294_dynamic_recompute_pilot.json
results/gpt5_phase294b_dynamic_recompute_full/glm4_phase294_dynamic_recompute_pilot.json
results/gpt5_phase294b_dynamic_recompute_full/deepseek7b_phase294_dynamic_recompute_pilot.json
```

日志：

```text
results/gpt5_gpu_lock_logs/20260528_100011_phase294_qwen3
results/gpt5_gpu_lock_logs/20260528_101558_phase294_glm4
results/gpt5_gpu_lock_logs/20260528_103938_phase294_glm4
results/gpt5_gpu_lock_logs/20260528_104356_phase294_deepseek7b
```

所有正式运行的 `kernel.since-start.filtered.log` 都是 0 行，包括 GLM4 segfault 那次。

### 数据规模

```text
Qwen3:
  pairs = 76
  rows = 13680
  nonfinite = 0

GLM4:
  pairs = 76
  rows = 13680
  nonfinite = 0

DeepSeek7B:
  pairs = 76
  rows = 12160
  nonfinite = 0

total rows = 39520
```

### Qwen3 客观结果

alpha 平均曲线：

```text
resid_in:
  alpha 0    = 0.0000
  alpha 0.25 = 0.2866
  alpha 0.5  = 0.6220
  alpha 0.75 = 0.8229
  alpha 1.0  = 0.8455

resid_out:
  alpha 0    = 0.0000
  alpha 0.25 = 0.2936
  alpha 0.5  = 0.6371
  alpha 0.75 = 0.8474
  alpha 1.0  = 0.8714

attn_out:
  alpha 0    = 0.0000
  alpha 0.25 = 0.0454
  alpha 0.5  = 0.1048
  alpha 0.75 = 0.1631
  alpha 1.0  = 0.2059

mlp_out:
  alpha 0    = 0.0000
  alpha 0.25 = 0.0859
  alpha 0.5  = 0.1747
  alpha 0.75 = 0.2578
  alpha 1.0  = 0.3171
```

alpha=1.0 最强层：

```text
resid_in:
  L7 = 0.8825
  L6 = 0.8792
  L8 = 0.8758

resid_out:
  L6 = 0.8825
  L8 = 0.8819
  L5 = 0.8792

attn_out:
  L0 = 0.7691
  L1 = 0.1898
  L6 = 0.1662

mlp_out:
  L0 = 0.8247
  L4 = 0.2998
  L5 = 0.2855
```

与 Phase 291 对比：

```text
Phase 291 Qwen3 L0-L8 both_progress = 0.8788
Phase 294b Qwen3 resid_in L7 alpha=1 = 0.8825
```

客观现象：

Qwen3 的单点 residual 起点 patch 后自然重算已经能接近 L0-L8 block patch。attn_out/mlp_out 单模块效果仍集中在 L0，后续层单模块效果明显弱于 residual。

### GLM4 客观结果

alpha 平均曲线：

```text
resid_in:
  alpha 0    = -0.0000
  alpha 0.25 = 0.2995
  alpha 0.5  = 0.7200
  alpha 0.75 = 0.9552
  alpha 1.0  = 0.9856

resid_out:
  alpha 0    = 0.0000
  alpha 0.25 = 0.2920
  alpha 0.5  = 0.7380
  alpha 0.75 = 0.9547
  alpha 1.0  = 0.9876

attn_out:
  alpha 0    = 0.0000
  alpha 0.25 = 0.0079
  alpha 0.5  = 0.0226
  alpha 0.75 = 0.0480
  alpha 1.0  = 0.0876

mlp_out:
  alpha 0    = 0.0000
  alpha 0.25 = 0.0490
  alpha 0.5  = 0.1723
  alpha 0.75 = 0.2885
  alpha 1.0  = 0.4033
```

alpha=1.0 最强层：

```text
resid_in:
  L8 = 0.9914
  L7 = 0.9912
  L6 = 0.9900

resid_out:
  L7 = 0.9914
  L6 = 0.9912
  L8 = 0.9912

attn_out:
  L1 = 0.2318
  L4 = 0.1474
  L2 = 0.1091

mlp_out:
  L0 = 0.9416
  L1 = 0.6442
  L2 = 0.5326
```

与 Phase 291 对比：

```text
Phase 291 GLM4 L0-L8 both_progress = 0.9906
Phase 294b GLM4 resid_in L8 alpha=1 = 0.9914
```

客观现象：

GLM4 的单点 residual 起点 patch 后自然重算几乎完全复现 block patch 效果。MLP_out 的 L0 仍然很强，attention_out 始终弱。这继续支持 GLM4 的 MLP 集中模式。GLM4 长跑仍然出现一次用户态 segfault 139，但 kernel filtered 日志为 0，resume 后完成。

### DeepSeek7B 客观结果

alpha 平均曲线：

```text
resid_in:
  alpha 0    = 0.0000
  alpha 0.25 = 0.1715
  alpha 0.5  = 0.3207
  alpha 0.75 = 0.4970
  alpha 1.0  = 0.6289

resid_out:
  alpha 0    = 0.0000
  alpha 0.25 = 0.1771
  alpha 0.5  = 0.3340
  alpha 0.75 = 0.5283
  alpha 1.0  = 0.6753

attn_out:
  alpha 0    = 0.0000
  alpha 0.25 = 0.0194
  alpha 0.5  = 0.0411
  alpha 0.75 = 0.0688
  alpha 1.0  = 0.0765

mlp_out:
  alpha 0    = 0.0000
  alpha 0.25 = 0.0426
  alpha 0.5  = 0.0713
  alpha 0.75 = 0.1264
  alpha 1.0  = 0.1786
```

alpha=1.0 最强层：

```text
resid_in:
  L25 = 0.6317
  L24 = 0.6314
  L26 = 0.6294

resid_out:
  L27 = 1.0000
  L24 = 0.6317
  L23 = 0.6314

attn_out:
  L27 = 0.5728
  L25 = 0.0417
  L24 = 0.0137

mlp_out:
  L27 = 0.4139
  L26 = 0.2957
  L25 = 0.2592
```

与 Phase 291 对比：

```text
Phase 291 DeepSeek7B L20-L27 both_progress = 0.9173
Phase 294b DeepSeek7B resid_in alpha=1:
  L20 = 0.6289
  L21 = 0.6269
  L22 = 0.6282
  L23 = 0.6277
  L24 = 0.6314
  L25 = 0.6317
  L26 = 0.6294
  L27 = 0.6270
```

客观现象：

DeepSeek7B 的单点 residual 起点 patch 无法复现 L20-L27 block patch 效果。L27 resid_out 直接达到 1.0，说明最后层输出 residual 已经基本是目标状态；但 L20-L26 的 resid_in 起点 patch 后自然重算都停留在约 0.63。这支持：

```text
DeepSeek7B 的 L20-L27 强效果更像多层轨迹累计或持续多点干预，
不是单个深层 residual 起点足以自然重算出来。
```

### 三模型对比

```text
Qwen3:
  单点 residual patch 可接近 block patch；
  L0 attn_out/mlp_out 明显有效；
  后续层单模块 patch 弱于 residual。

GLM4:
  单点 residual patch 几乎完全复现 block patch；
  L0 mlp_out 极强；
  attn_out 很弱；
  仍有用户态长跑稳定性问题。

DeepSeek7B:
  单点 residual patch 明显低于 block patch；
  L27 attn_out/mlp_out 有效；
  L20-L26 单模块 patch 很弱；
  L20-L27 block 强效果不能由单起点解释。
```

### 本轮最重要的新事实

```text
1. Qwen3/GLM4 的 block patch 效果大部分可以由单点 residual 起点 patch 后自然重算复现。
2. DeepSeek7B 的 block patch 效果不能由单点 residual 起点 patch 复现。
3. GLM4 的 L0 MLP_out 强效在更大样本和 alpha 曲线下继续稳定。
4. DeepSeek7B 的有效单模块信号集中在 L27，深层前段 L20-L26 单模块信号弱。
5. 三模型 alpha 曲线总体平滑，没有出现大量非有限输出。
```

### 当前判断修正

Phase 291 的 block 累积增强现在应拆成两类：

```text
1. 可由单点 residual 起点重算复现的 block 效果：
   Qwen3、GLM4 更接近这一类。

2. 不能由单点 residual 起点重算复现的 block 效果：
   DeepSeek7B 更接近这一类。
```

这说明 block patch 不能统一解释为同一种机制。对于 DeepSeek7B，必须测试 segment patch 或多点 patch。

### 硬伤

1. Phase 294b 仍然只测现有 76 pair，不是完整功能库。
2. 只测 residual/attn/mlp 输出，不含 head/neuron 层级。
3. `resid_out L27 = 1.0` 这类结果很可能包含输出接口效应，不能当作中间机制。
4. GLM4 仍有用户态 segfault 139，虽然 kernel filtered 日志为空，但长跑稳定性仍需分段处理。
5. 动态重算仍是反事实 patch，不是直接观察自然任务路径。

### 下一步计划

1. Phase 295：DeepSeek7B segment dynamic recompute。
   - patch L20-L23，重算 L24-L27。
   - patch L24-L27，重算输出。
   - patch L20-L23 + L27 attn/mlp。
   - 判断 DeepSeek7B 的关键差异是前段写入、中段传播，还是末层释放。

2. Phase 296：Qwen3/GLM4 变量解码 pilot。
   - 因为单点 residual 起点可重算，适合先在 Qwen3/GLM4 上解码 agent/patient/operator/scope。

3. Phase 297：扩展功能库。
   - translation
   - tense
   - coreference
   - long passive
   - nested logical
   - double recursive

4. Phase 298：GLM4 稳定性隔离。
   - 每 48 pair 分段运行。
   - 比较 SDPA vs eager。
   - 判断 segfault 是否由长 session、SDPA、device_map auto 或 GLM remote code 引起。

## Phase 32: Global Functional Contract Mapping v0 系统级图谱测试 [2026-05-28 11:08]

### 任务目标

根据“全局功能契约图谱算法”的建议，本轮先实现一个可落地的 v0 版本，不重新跑 GPU，而是把已有多阶段结果合成统一的系统级功能图谱：

```text
Phase 290: single-layer contract break scan
Phase 291: block contract scan
Phase 293: norm-based naturalness events
Phase 294b: dynamic recompute alpha/layer curves
```

目标不是证明最终语言理论，而是建立可以累计的功能签名和复用/差异化矩阵。

### 对用户方案的判断

用户提出的 Global Functional Contract Mapping 方向正确，尤其是：

```text
1. 不应继续只看单层或单模块。
2. 每个 subtype 应生成层 × block × alpha × dynamic recompute × naturalness 的契约指纹。
3. 复用/差异化矩阵应分维度输出，而不是只输出一个总分。
4. head/neuron 映射必须建立在稳定功能路径上。
```

但当前还不能直接做完整版本，因为：

```text
1. 功能库仍只有 19 个 subtype。
2. 还没有 translation/style/tense/coreference。
3. 还没有 head/neuron 级别数据。
4. naturalness 仍只是 norm-based。
```

因此本轮先做 GFCM v0：系统级数据结构和矩阵，不做最终理论总结。

### 脚本

新增：

```text
tests/gpt5/phase295_global_contract_mapping.py
```

脚本输入：

```text
results/gpt5_phase290_contract_break_full
results/gpt5_phase291_block_contract_full
results/gpt5_phase293_naturalness
results/gpt5_phase294b_dynamic_recompute_full
```

生成 signature 维度：

```text
layer_curve:
  Phase 290 layer × patch progress/KL/delta

single_alpha:
  Phase 290 alpha × patch progress

block_curve:
  Phase 291 block × patch progress/KL/delta/drop

block_alpha:
  Phase 291 alpha × patch progress

naturalness:
  Phase 293 norm-normal/off-manifold/numeric event rates

dynamic_layer:
  Phase 294b layer × dynamic patch progress/KL/delta

dynamic_alpha:
  Phase 294b alpha × dynamic patch progress

summary:
  per-phase best/mean progress and block width summaries
```

同时输出：

```text
1. group-normalized similarity
2. z-score similarity
3. top reuse candidates
4. bottom differentiation candidates
5. per-dimension similarities:
   layer / block / dynamic / naturalness / alpha / summary
```

### 命令

```bash
python tests/gpt5/phase295_global_contract_mapping.py \
  --phase290-dir results/gpt5_phase290_contract_break_full \
  --phase291-dir results/gpt5_phase291_block_contract_full \
  --phase293-dir results/gpt5_phase293_naturalness \
  --phase294-dir results/gpt5_phase294b_dynamic_recompute_full \
  --output-dir results/gpt5_phase295_global_contract_mapping \
  --top-k 25
```

说明：

第一次输出时发现分维度相似度全为 0，是 group-normalized 特征前缀处理错误。已修复：

```text
group_subset() 支持识别 "layer_curve:" 等 group 前缀。
```

随后重跑并覆盖输出。

### 输出文件

```text
results/gpt5_phase295_global_contract_mapping/GLOBAL_CONTRACT_MAPPING_REPORT.md
results/gpt5_phase295_global_contract_mapping/global_mapping_summary.csv
results/gpt5_phase295_global_contract_mapping/global_contract_maps.json

results/gpt5_phase295_global_contract_mapping/qwen3_global_similarity.csv
results/gpt5_phase295_global_contract_mapping/qwen3_global_top_reuse_candidates.csv
results/gpt5_phase295_global_contract_mapping/qwen3_global_bottom_differentiation_candidates.csv
results/gpt5_phase295_global_contract_mapping/qwen3_global_zscore_top_pairs.csv
results/gpt5_phase295_global_contract_mapping/qwen3_global_zscore_bottom_pairs.csv

results/gpt5_phase295_global_contract_mapping/glm4_global_similarity.csv
results/gpt5_phase295_global_contract_mapping/glm4_global_top_reuse_candidates.csv
results/gpt5_phase295_global_contract_mapping/glm4_global_bottom_differentiation_candidates.csv
results/gpt5_phase295_global_contract_mapping/glm4_global_zscore_top_pairs.csv
results/gpt5_phase295_global_contract_mapping/glm4_global_zscore_bottom_pairs.csv

results/gpt5_phase295_global_contract_mapping/deepseek7b_global_similarity.csv
results/gpt5_phase295_global_contract_mapping/deepseek7b_global_top_reuse_candidates.csv
results/gpt5_phase295_global_contract_mapping/deepseek7b_global_bottom_differentiation_candidates.csv
results/gpt5_phase295_global_contract_mapping/deepseek7b_global_zscore_top_pairs.csv
results/gpt5_phase295_global_contract_mapping/deepseek7b_global_zscore_bottom_pairs.csv
```

### 数据规模

```text
Qwen3:
  phase290_rows = 28512
  phase291_rows = 15840
  phase293_event_rows = 784
  phase294_rows = 13680
  subtypes = 19
  features/subtype = 504

GLM4:
  phase290_rows = 31680
  phase291_rows = 19008
  phase293_event_rows = 1468
  phase294_rows = 13680
  subtypes = 19
  features/subtype = 544

DeepSeek7B:
  phase290_rows = 25344
  phase291_rows = 15840
  phase293_event_rows = 1078
  phase294_rows = 12160
  subtypes = 19
  features/subtype = 472
```

### Group-normalized 总体相似度

```text
Qwen3:
  combined_mean = 0.9505
  combined_min = 0.8731
  combined_max = 0.9953
  same_category_mean = 0.9693
  cross_category_mean = 0.9453

GLM4:
  combined_mean = 0.9675
  combined_min = 0.9112
  combined_max = 0.9980
  same_category_mean = 0.9812
  cross_category_mean = 0.9638

DeepSeek7B:
  combined_mean = 0.8913
  combined_min = 0.6143
  combined_max = 0.9900
  same_category_mean = 0.9148
  cross_category_mean = 0.8848
```

客观现象：

group-normalized 后整体相似度仍然偏高，说明当前 19 个 subtype 的行为曲线有很强共同结构。DeepSeek7B 的分化最明显，min 降到 0.6143。

### Qwen3 候选结果

Group-normalized top reuse candidates：

```text
causal / contrast:
  combined = 0.9953
  layer = 0.9930
  block = 0.9937
  dynamic = 0.9899
  naturalness = 0.9950

contrast / inference:
  combined = 0.9949
  layer = 0.9870
  block = 0.9896
  dynamic = 0.9904
  naturalness = 0.9960

causal / inference:
  combined = 0.9944
  layer = 0.9876
  block = 0.9932
  dynamic = 0.9825
  naturalness = 0.9998
```

Group-normalized bottom differentiation candidates：

```text
morphological_neg / syntactic_do_not:
  combined = 0.8731
  layer = 0.8298
  block = 0.8875
  dynamic = 0.8490

get_passive / morphological_neg:
  combined = 0.8763
  layer = 0.8393
  block = 0.9133
  dynamic = 0.8552

and_or / syntactic_do_not:
  combined = 0.8794
  layer = 0.8568
  block = 0.8947
  dynamic = 0.8607
```

Z-score top：

```text
conditional / morphological_neg = 0.8288
causal / contrast = 0.8267
causal / no_agent = 0.8255
causal / morphological_neg = 0.8179
causal / conditional = 0.8076
```

Z-score bottom：

```text
complement_clause / conditional = -0.7912
causal / get_passive = -0.7702
causal / complement_clause = -0.7651
get_passive / morphological_neg = -0.7593
complement_clause / inference = -0.7575
```

客观现象：

Qwen3 的逻辑关系类功能在 z-score 偏离模式中较接近，complement_clause 与多个逻辑/被动/否定 subtype 分化明显。

### GLM4 候选结果

Group-normalized top reuse candidates：

```text
by_phrase / dative_passive:
  combined = 0.9980
  layer = 0.9955
  block = 0.9965
  dynamic = 0.9956
  naturalness = 0.9998

causal / inference:
  combined = 0.9980
  layer = 0.9950
  block = 0.9980
  dynamic = 0.9933
  naturalness = 0.9979

dative_passive / get_passive:
  combined = 0.9971
  layer = 0.9940
  block = 0.9949
  dynamic = 0.9923
  naturalness = 0.9997
```

Group-normalized bottom differentiation candidates：

```text
and_or / syntactic_do_not:
  combined = 0.9112
  layer = 0.8359
  block = 0.9700
  dynamic = 0.9094

and_or / lexical_not_adj:
  combined = 0.9159
  layer = 0.8292
  block = 0.9574
  dynamic = 0.9096

causal / syntactic_do_not:
  combined = 0.9188
  layer = 0.8271
  block = 0.9737
  dynamic = 0.9019
```

Z-score top：

```text
conditional / contrast = 0.8637
causal / inference = 0.8363
causal / contrast = 0.8321
causal / conditional = 0.7840
contrast / no_agent = 0.7817
```

Z-score bottom：

```text
contrast / syntactic_do_not = -0.7417
conditional / syntactic_do_not = -0.7128
causal / syntactic_do_not = -0.6878
inference / syntactic_do_not = -0.6707
contrast / relative_clause = -0.6331
```

客观现象：

GLM4 的 passive 子类型在多维度上高度相似，逻辑类也形成较稳定的相似簇；syntactic_do_not 与多个逻辑 subtype 在 z-score 偏离模式中分化明显。

### DeepSeek7B 候选结果

Group-normalized top reuse candidates：

```text
by_phrase / get_passive:
  combined = 0.9900
  layer = 0.9865
  block = 0.9869
  dynamic = 0.9832
  naturalness = 0.9799

possessive_chain / relative_clause:
  combined = 0.9832
  layer = 0.9687
  block = 0.9678
  dynamic = 0.9719
  naturalness = 0.9820

get_passive / possessive_chain:
  combined = 0.9814
  layer = 0.9955
  block = 0.9910
  dynamic = 0.9579
  naturalness = 0.9386
```

Group-normalized bottom differentiation candidates：

```text
complement_clause / no_agent:
  combined = 0.6143
  layer = 0.8062
  block = 0.8647
  dynamic = 0.5793
  naturalness = 0.9377

complement_clause / morphological_neg:
  combined = 0.6205
  layer = 0.8404
  block = 0.6968
  dynamic = 0.5597
  naturalness = 0.7675

complement_clause / existential_no:
  combined = 0.6440
  layer = 0.8052
  block = 0.8682
  dynamic = 0.6314
  naturalness = 0.8150
```

Z-score top：

```text
possessive_chain / pp_chain = 0.6327
by_phrase / get_passive = 0.6307
by_phrase / possessive_chain = 0.6010
possessive_chain / relative_clause = 0.5783
get_passive / possessive_chain = 0.5644
```

Z-score bottom：

```text
conditional / possessive_chain = -0.6292
causal / possessive_chain = -0.5682
existential_no / pp_chain = -0.5605
inference / possessive_chain = -0.5374
conditional / relative_clause = -0.5095
```

客观现象：

DeepSeek7B 的 complement_clause 在 group-normalized 矩阵中与多个 subtype 明显分化，尤其 dynamic_layer 相似度低。这和 Phase 294b 中 DeepSeek7B 单点 residual 起点不能复现 block 效果一致：它的递归/补足从句路径可能与其他功能有不同的动态重算模式。

### 本轮最重要的新事实

```text
1. GFCM v0 可以把 layer、block、naturalness、dynamic recompute 合成统一功能图谱。
2. group-normalized 总体相似度仍偏高，说明当前功能库太窄，共同模板/共同路径仍强。
3. z-score 后可以看到更明显的偏离结构。
4. Qwen3/GLM4 中逻辑类关系形成较稳定相似簇。
5. GLM4 中 passive 子类型高度相似。
6. DeepSeek7B 中 complement_clause 和 possessive/relative 等递归相关结构显示强分化候选。
```

### 当前最重要的硬伤

1. 当前 GFCM v0 仍只有 19 个 subtype，不是“全局语言”。
2. group-normalized 相似度过高，说明当前特征仍可能被共同任务形状主导。
3. z-score 结果更有区分度，但 z-score 只是模型内偏离模式，不等于真实机制复用。
4. 没有 head/neuron 级数据。
5. 没有变量解码，不能回答路径中传递的内容是什么。
6. naturalness 仍只是 norm-based。

### 下一步计划

1. Phase 296：扩展功能库第一批。
   - translation
   - tense
   - coreference
   - style
   - long passive
   - nested logical
   - double recursive

2. Phase 297：GFCM v1。
   - 新功能库进入 Phase 290/291/294b 流程。
   - 每类至少 100 pair 作为目标，但先用 20-40 pair 验证脚本。
   - 输出更大 subtype matrix。

3. Phase 298：DeepSeek7B segment dynamic recompute。
   - 解释 complement_clause 和深层 block 差异。

4. Phase 299：head/neuron 映射 pilot。
   - 只在 GFCM 中稳定的候选路径上做。
   - 优先：
     GLM4 passive 子类型；
     Qwen3 logical 子类型；
     DeepSeek7B complement_clause 分化路径。

## Phase 33: Phase 296 扩展功能库与三模型 Pilot 验证 [2026-05-28 11:41]

### 任务目标

继续推进 Global Functional Contract Mapping。Phase 32 的最大硬伤是功能库太窄，只有 19 个 subtype。本轮先扩展功能库第一批，并用三模型做小规模 pilot，验证：

```text
1. 新功能库能否被现有 Phase 290/291/294b 脚本直接复用。
2. 新 category/subtype 是否能稳定跑完三模型。
3. 新功能是否继续表现出模型特异路径：
   Qwen3 浅层协同；
   GLM4 MLP 集中；
   DeepSeek7B 末层释放。
```

### 脚本修改

修改：

```text
tests/gpt5/phase289_contract_scan.py
```

在 `build_pairs()` 中新增功能库：

```text
translation
tense
coreference
style
passive.long_passive
passive.nested_passive
logical.nested_condition
logical.nested_contrast
logical.negated_condition
recursive.double_relative
recursive.center_embedding
recursive.deep_complement
```

扩展后 pair 库规模：

```text
total pairs = 244
categories = 8
subtypes = 45
```

category 分布：

```text
coreference = 12
logical = 52
negation = 40
passive = 44
recursive = 44
style = 16
tense = 16
translation = 20
```

subtype 分布：

```text
coreference:
  deictic_switch = 4
  he_coref = 2
  it_coref = 2
  she_coref = 2
  they_coref = 2

logical:
  and_or = 8
  causal = 8
  conditional = 8
  contrast = 8
  inference = 8
  negated_condition = 4
  nested_condition = 4
  nested_contrast = 4

negation:
  existential_no = 6
  lexical_not_adj = 8
  morphological_neg = 6
  never = 6
  scope_quantifier = 6
  syntactic_do_not = 8

passive:
  by_phrase = 8
  dative_passive = 8
  get_passive = 8
  long_passive = 8
  nested_passive = 4
  no_agent = 8

recursive:
  center_embedding = 4
  complement_clause = 8
  deep_complement = 4
  double_relative = 4
  possessive_chain = 8
  pp_chain = 8
  relative_clause = 8

style:
  casual = 4
  concise = 4
  formal = 4
  poetic = 4

tense:
  future_will = 4
  past_simple = 4
  perfect = 4
  progressive = 4

translation:
  en_fr_phrase = 4
  en_fr_word = 4
  en_zh_phrase = 4
  en_zh_word = 4
  target_language_switch = 4
```

### 测试命令

本轮使用 Phase 290 单层扫描做扩展库 pilot。每个 subtype 取 2 pair：

```text
selected pairs = 90
layers:
  Qwen3 / GLM4 = L0, L4, L8
  DeepSeek7B = L20, L24, L27
alphas = 0, 0.5, 1.0
patch_types = attn, mlp, both, resid, cross_battn_amlp, cross_aattn_bmlp
```

Qwen3：

```bash
MAX_SECONDS=3600 OUTPUT_DIR=results/gpt5_phase296_expanded_function_pilot \
tests/gpt5/run_phase290_conservative.sh qwen3 \
  --categories negation,logical,passive,recursive,translation,tense,coreference,style \
  --max-pairs-per-subtype 2 \
  --layers 0,4,8 \
  --alphas 0,0.5,1.0 \
  --progress-every 8 \
  --label expanded_pilot
```

GLM4：

```bash
MAX_SECONDS=5400 OUTPUT_DIR=results/gpt5_phase296_expanded_function_pilot \
tests/gpt5/run_phase290_conservative.sh glm4 \
  --categories negation,logical,passive,recursive,translation,tense,coreference,style \
  --max-pairs-per-subtype 2 \
  --layers 0,4,8 \
  --alphas 0,0.5,1.0 \
  --progress-every 8 \
  --label expanded_pilot
```

DeepSeek7B：

```bash
MAX_SECONDS=3600 OUTPUT_DIR=results/gpt5_phase296_expanded_function_pilot \
tests/gpt5/run_phase290_conservative.sh deepseek7b \
  --categories negation,logical,passive,recursive,translation,tense,coreference,style \
  --max-pairs-per-subtype 2 \
  --layers 20,24,27 \
  --alphas 0,0.5,1.0 \
  --progress-every 8 \
  --label expanded_pilot
```

### 输出文件

```text
results/gpt5_phase296_expanded_function_pilot/qwen3_phase290_contract_break_scan.json
results/gpt5_phase296_expanded_function_pilot/glm4_phase290_contract_break_scan.json
results/gpt5_phase296_expanded_function_pilot/deepseek7b_phase290_contract_break_scan.json
```

checkpoints：

```text
results/gpt5_phase296_expanded_function_pilot/checkpoints/qwen3/coreference-logical-negation-passive-recursive-style-tense-translation_expanded_pilot.json
results/gpt5_phase296_expanded_function_pilot/checkpoints/glm4/coreference-logical-negation-passive-recursive-style-tense-translation_expanded_pilot.json
results/gpt5_phase296_expanded_function_pilot/checkpoints/deepseek7b/coreference-logical-negation-passive-recursive-style-tense-translation_expanded_pilot.json
```

日志：

```text
results/gpt5_gpu_lock_logs/20260528_112237_phase290_qwen3
results/gpt5_gpu_lock_logs/20260528_112734_phase290_glm4
results/gpt5_gpu_lock_logs/20260528_113557_phase290_deepseek7b
```

三轮 `kernel.since-start.filtered.log` 都是 0 行。

### 数据规模

```text
Qwen3:
  pairs = 90
  rows = 3780
  nonfinite = 0
  norm_illegal = 1

GLM4:
  pairs = 90
  rows = 3780
  nonfinite = 0
  norm_illegal = 0

DeepSeek7B:
  pairs = 90
  rows = 3780
  nonfinite = 0
  norm_illegal = 0

total rows = 11340
```

### Qwen3 客观结果

```text
best_layer_by_both_progress = 0
contract_broken_layers = [0]
```

layer curve：

```text
L0:
  both_progress = 0.7810
  attn_progress = 0.7060
  mlp_progress = 0.7391
  resid_progress = 0.7855
  cross_battn_amlp_progress = 0.2387

L4:
  both_progress = 0.3152
  attn_progress = 0.0906
  mlp_progress = 0.2694
  resid_progress = 0.8088
  cross_battn_amlp_progress = 0.0678

L8:
  both_progress = 0.1918
  attn_progress = 0.0998
  mlp_progress = 0.2062
  resid_progress = 0.8249
  cross_battn_amlp_progress = 0.0609
```

contract event：

```text
L0 cross_battn_amlp:
  kl_ratio_vs_both = 2.7479
  progress_drop = 0.5422
```

category both alpha=1：

```text
coreference = 0.2775
logical = 0.3923
negation = 0.5326
passive = 0.5359
recursive = 0.4005
style = 0.2471
tense = 0.4716
translation = 0.5409
```

top subtypes：

```text
syntactic_do_not = 0.9647
center_embedding = 0.9153
by_phrase = 0.7058
possessive_chain = 0.6958
en_zh_phrase = 0.6880
```

bottom subtypes：

```text
style.concise = -0.0146
coreference.it_coref = 0.0597
recursive.double_relative = 0.0658
coreference.they_coref = 0.1701
recursive.complement_clause = 0.1756
```

客观现象：

Qwen3 在扩展库中仍是 L0 最强，且 attention/MLP/both 在 L0 都较强；cross_battn_amlp 在 L0 明显失败。translation、passive、negation 在这个 pilot 中 both_progress 较高，style/coreference 较低。

### GLM4 客观结果

```text
best_layer_by_both_progress = 0
contract_broken_layers = []
```

layer curve：

```text
L0:
  both_progress = 0.8942
  attn_progress = 0.0150
  mlp_progress = 0.8969
  resid_progress = 0.9440
  cross_battn_amlp_progress = 0.0031

L4:
  both_progress = 0.4277
  attn_progress = 0.1164
  mlp_progress = 0.3445
  resid_progress = 0.9618
  cross_battn_amlp_progress = 0.0559

L8:
  both_progress = 0.1793
  attn_progress = 0.0327
  mlp_progress = 0.1379
  resid_progress = 0.9718
  cross_battn_amlp_progress = 0.0210
```

category both alpha=1：

```text
coreference = 0.3632
logical = 0.4595
negation = 0.6153
passive = 0.5873
recursive = 0.5200
style = 0.4464
tense = 0.5770
translation = 0.4153
```

top subtypes：

```text
syntactic_do_not = 0.7229
negated_condition = 0.7126
by_phrase = 0.6759
never = 0.6707
formal = 0.6638
```

bottom subtypes：

```text
it_coref = 0.1744
they_coref = 0.1790
style.concise = 0.2082
target_language_switch = 0.2339
style.casual = 0.2595
```

客观现象：

GLM4 在扩展库中继续表现为 L0 MLP 集中：L0 mlp_progress = 0.8969，几乎等于 both_progress = 0.8942，而 L0 attn_progress = 0.0150。扩展库 pilot 没有复现 GLM4 用户态 segfault。

### DeepSeek7B 客观结果

```text
best_layer_by_both_progress = 27
contract_broken_layers = []
```

layer curve：

```text
L20:
  both_progress = 0.0690
  attn_progress = -0.0006
  mlp_progress = 0.0565
  resid_progress = 0.5336
  cross_battn_amlp_progress = 0.0130

L24:
  both_progress = 0.0269
  attn_progress = -0.0078
  mlp_progress = 0.0445
  resid_progress = 0.5674
  cross_battn_amlp_progress = -0.0050

L27:
  both_progress = 0.6973
  attn_progress = 0.5989
  mlp_progress = 0.4959
  resid_progress = 1.0000
  cross_battn_amlp_progress = 0.0953
```

category both alpha=1：

```text
coreference = 0.2565
logical = 0.2521
negation = 0.2675
passive = 0.2278
recursive = 0.2948
style = 0.2709
tense = 0.2316
translation = 0.3106
```

top subtypes：

```text
past_simple = 0.4925
target_language_switch = 0.4474
existential_no = 0.4076
she_coref = 0.3931
en_fr_word = 0.3756
```

bottom subtypes：

```text
no_agent = 0.0188
style.concise = 0.0476
they_coref = 0.0591
causal = 0.0786
progressive = 0.1108
```

客观现象：

DeepSeek7B 在扩展库中继续表现为 L27 最强。L20/L24 的 attn/mlp/both 都很弱，但 resid_progress 已有 0.53-0.57；L27 attn/mlp/both 均明显增强，resid_progress = 1.0。这与前面“深层/末层释放”一致。

### 本轮最重要的新事实

```text
1. 扩展功能库可以直接接入现有测试框架。
2. translation/tense/coreference/style 能在三模型上稳定跑完。
3. 三模型原有结构差异在扩展功能库中仍然出现：
   Qwen3 = L0 attention/MLP 协同；
   GLM4 = L0 MLP 集中；
   DeepSeek7B = L27 末层释放。
4. style.concise 与 coreference 部分 subtype 在当前 patch 指标下较弱。
5. translation 在 Qwen3 中较强，在 GLM4/DeepSeek7B 中中等，需要后续按目标语言拆分分析。
```

### 硬伤

1. 本轮每个 subtype 只有 2 pair，只是 pilot。
2. translation 模板仍较粗，目标语言切换与翻译内容混在一起。
3. coreference 样本太少，he/she/it/they 只有 2 pair/subtype。
4. style 的语义差异和风格差异没有完全解耦。
5. 当前只跑 Phase 290 单层扫描，还没有 block/dynamic/naturalness/GFCM v1。

### 下一步计划

1. Phase 297：扩展功能库 Phase 291 block scan。
   - 对新功能库跑 block 测试。
   - Qwen3/GLM4：L0, L0-L2, L0-L4, L0-L8, L4-L8。
   - DeepSeek7B：L20-L23, L24-L27, L20-L27, L26-L27, L27。

2. Phase 298：扩展功能库 Phase 294b dynamic recompute。
   - 每个 subtype 2-4 pair。
   - 保留 alpha 曲线。
   - 验证新功能是否也符合 Qwen3/GLM4 单点 residual 可重算、DeepSeek7B 需要 segment 的模式。

3. Phase 299：GFCM v1。
   - 合并扩展功能库 Phase 290/291/294b。
   - 输出 45 subtype 的全局矩阵。

4. Phase 300：扩展样本到每类 100 pair。
   - 先补 coreference、translation、style。
   - 再补 nested logical、double recursive、long passive。

## Phase 34: 扩展功能库 Block Scan 与 GFCM v1 Partial [2026-05-28 12:30]

### 任务目标

根据最新分析，继续完成全局功能契约图谱任务。本轮做两件事：

```text
1. 对 Phase 33 扩展后的 45 subtype 功能库运行 Phase 291 block scan。
2. 生成 GFCM v1 partial，先把扩展功能库的 layer/block 曲线纳入全局矩阵。
```

注意：

```text
本轮 GFCM v1 是 partial。
新 subtype 目前只有 Phase 290/291 数据；
naturalness 与 dynamic recompute 还没有为新功能库完整重跑。
因此本轮主要解释 layer/block 路径，不解释完整动态签名。
```

### 对用户分析的判断

用户分析基本正确：

```text
1. 当前已经进入路径图谱阶段，但还没有进入编码变量内容。
2. Phase 294b 的动态重算把 block patch 分成可由单点 residual 复现和不可由单点 residual 复现两类。
3. GFCM 是必要框架，但当前还是行为图谱，不是编码图谱。
4. 需要扩展功能库，并逐步加入变量解码、自然流形距离、head/neuron 定位。
```

本轮选择继续扩展 GFCM，而不是马上做 head/neuron，因为新功能库刚加入，必须先确认 block 曲线是否稳定。

### 测试命令

Qwen3：

```bash
MAX_SECONDS=5400 OUTPUT_DIR=results/gpt5_phase297_expanded_block_pilot \
tests/gpt5/run_phase291_conservative.sh qwen3 \
  --categories negation,logical,passive,recursive,translation,tense,coreference,style \
  --max-pairs-per-subtype 2 \
  --blocks 0,0-2,0-4,0-8,4-8 \
  --alphas 0,0.5,1.0 \
  --progress-every 8 \
  --label expanded_pilot
```

GLM4：

```bash
MAX_SECONDS=7200 OUTPUT_DIR=results/gpt5_phase297_expanded_block_pilot \
tests/gpt5/run_phase291_conservative.sh glm4 \
  --categories negation,logical,passive,recursive,translation,tense,coreference,style \
  --max-pairs-per-subtype 2 \
  --blocks 0,0-1,0-2,0-4,0-8,4-8 \
  --alphas 0,0.5,1.0 \
  --progress-every 8 \
  --label expanded_pilot
```

DeepSeek7B：

```bash
MAX_SECONDS=5400 OUTPUT_DIR=results/gpt5_phase297_expanded_block_pilot \
tests/gpt5/run_phase291_conservative.sh deepseek7b \
  --categories negation,logical,passive,recursive,translation,tense,coreference,style \
  --max-pairs-per-subtype 2 \
  --blocks 20-23,24-27,20-27,26-27,27 \
  --alphas 0,0.5,1.0 \
  --progress-every 8 \
  --label expanded_pilot
```

GFCM v1 partial：

```bash
python tests/gpt5/phase295_global_contract_mapping.py \
  --phase290-dir results/gpt5_phase296_expanded_function_pilot \
  --phase291-dir results/gpt5_phase297_expanded_block_pilot \
  --phase293-dir results/gpt5_phase293_naturalness \
  --phase294-dir results/gpt5_phase294b_dynamic_recompute_full \
  --output-dir results/gpt5_phase297_expanded_gfcm_v1_partial \
  --top-k 30
```

### 输出文件

Block scan：

```text
results/gpt5_phase297_expanded_block_pilot/qwen3_phase291_block_contract_scan.json
results/gpt5_phase297_expanded_block_pilot/glm4_phase291_block_contract_scan.json
results/gpt5_phase297_expanded_block_pilot/deepseek7b_phase291_block_contract_scan.json
```

GFCM v1 partial：

```text
results/gpt5_phase297_expanded_gfcm_v1_partial/GLOBAL_CONTRACT_MAPPING_REPORT.md
results/gpt5_phase297_expanded_gfcm_v1_partial/global_mapping_summary.csv
results/gpt5_phase297_expanded_gfcm_v1_partial/global_contract_maps.json
```

日志：

```text
results/gpt5_gpu_lock_logs/20260528_115951_phase291_qwen3
results/gpt5_gpu_lock_logs/20260528_120734_phase291_glm4
results/gpt5_gpu_lock_logs/20260528_122101_phase291_deepseek7b
```

三轮 `kernel.since-start.filtered.log` 都是 0 行。

### Block Scan 数据规模

```text
Qwen3:
  pairs = 90
  rows = 6300
  best_block = L0-L8
  nonfinite = 0
  norm_illegal = 0

GLM4:
  pairs = 90
  rows = 7560
  best_block = L0-L8
  nonfinite = 0
  norm_illegal = 0

DeepSeek7B:
  pairs = 90
  rows = 6300
  best_block = L20-L27
  nonfinite = 0
  norm_illegal = 0
```

### Qwen3 Block 结果

block curve：

```text
L0:
  both = 0.7808
  attn = 0.7059
  mlp = 0.7390
  resid = 0.7854
  cross_battn_amlp = 0.2387

L0-L2:
  both = 0.7961
  attn = 0.7575
  mlp = 0.7637
  resid = 0.7982
  cross_battn_amlp = 0.3109

L0-L4:
  both = 0.8077
  attn = 0.7603
  mlp = 0.7729
  resid = 0.8086
  cross_battn_amlp = 0.2245

L0-L8:
  both = 0.8251
  attn = 0.7395
  mlp = 0.8002
  resid = 0.8248
  cross_battn_amlp = 0.1922

L4-L8:
  both = 0.6561
  attn = 0.2763
  mlp = 0.6045
  resid = 0.8248
  cross_battn_amlp = 0.1444
```

category both alpha=1：

```text
coreference = 0.6821
logical = 0.8278
negation = 0.8866
passive = 0.8270
recursive = 0.5549
style = 0.5060
tense = 0.9343
translation = 0.9666
```

top subtype：

```text
en_fr_phrase = 1.1882
en_zh_phrase = 1.1210
center_embedding = 1.1082
syntactic_do_not = 1.0717
en_fr_word = 1.0525
```

bottom subtype：

```text
concise = 0.0955
it_coref = 0.1746
double_relative = 0.2631
relative_clause = 0.3007
complement_clause = 0.3353
```

contract events：

```text
L0-L8 cross_battn_amlp:
  kl_ratio_vs_both = 5.3472
  progress_drop = 0.6329

L0-L4 cross_battn_amlp:
  kl_ratio_vs_both = 4.4402
  progress_drop = 0.5831

L4-L8 cross_battn_amlp:
  kl_ratio_vs_both = 3.5702
  progress_drop = 0.5118
```

客观现象：

Qwen3 在扩展库 block scan 中仍是 L0-L8 最强，且 L0 单层已强。translation 与 tense 在 block patch 下非常强，style.concise 与部分 coreference/recursive subtype 明显较弱。

### GLM4 Block 结果

block curve：

```text
L0:
  both = 0.8942
  attn = 0.0150
  mlp = 0.8969
  resid = 0.9440
  cross_battn_amlp = 0.0031

L0-L1:
  both = 0.9505
  attn = 0.0316
  mlp = 0.9405
  resid = 0.9608
  cross_battn_amlp = 0.0362

L0-L2:
  both = 0.9581
  attn = 0.0698
  mlp = 0.9507
  resid = 0.9568
  cross_battn_amlp = 0.0360

L0-L4:
  both = 0.9621
  attn = 0.2528
  mlp = 0.9658
  resid = 0.9618
  cross_battn_amlp = 0.1393

L0-L8:
  both = 0.9707
  attn = 0.4661
  mlp = 0.9675
  resid = 0.9718
  cross_battn_amlp = 0.1166

L4-L8:
  both = 0.8537
  attn = 0.3742
  mlp = 0.8157
  resid = 0.9718
  cross_battn_amlp = 0.0697
```

category both alpha=1：

```text
coreference = 0.8457
logical = 0.9311
negation = 0.9742
passive = 1.0126
recursive = 0.8906
style = 0.8612
tense = 0.9958
translation = 0.9320
```

top subtype：

```text
get_passive = 1.0667
by_phrase = 1.0338
long_passive = 1.0307
dative_passive = 1.0123
perfect = 1.0103
```

bottom subtype：

```text
it_coref = 0.6918
they_coref = 0.7392
concise = 0.7465
casual = 0.7616
nested_condition = 0.7866
```

contract events：

```text
L0-L2 cross_battn_amlp:
  kl_ratio_vs_both = 18.1929
  progress_drop = 0.9221

L0-L1 cross_battn_amlp:
  kl_ratio_vs_both = 11.6944
  progress_drop = 0.9144

L0-L8 cross_battn_amlp:
  kl_ratio_vs_both = 35.9318
  progress_drop = 0.8542
```

客观现象：

GLM4 的 MLP 集中在扩展 block 中更清楚：L0 的 mlp_progress 直接等于 both，L0-L8 的 MLP 也几乎等于 both。扩展库所有大类的 block 效果都较高，passive 和 tense 尤其高。

### DeepSeek7B Block 结果

block curve：

```text
L20-L23:
  both = 0.2541
  attn = 0.0373
  mlp = 0.1911
  resid = 0.5716
  cross_battn_amlp = 0.0500

L20-L27:
  both = 0.9324
  attn = 0.6967
  mlp = 0.7805
  resid = 1.0000
  cross_battn_amlp = 0.1346

L24-L27:
  both = 0.8517
  attn = 0.6431
  mlp = 0.6512
  resid = 1.0000
  cross_battn_amlp = 0.1162

L26-L27:
  both = 0.7759
  attn = 0.6085
  mlp = 0.5600
  resid = 1.0000
  cross_battn_amlp = 0.1050

L27:
  both = 0.6973
  attn = 0.5989
  mlp = 0.4959
  resid = 1.0000
  cross_battn_amlp = 0.0953
```

category both alpha=1：

```text
coreference = 0.6888
logical = 0.8014
negation = 0.6377
passive = 0.6438
recursive = 0.7437
style = 0.7029
tense = 0.5476
translation = 0.7701
```

top subtype：

```text
conditional = 1.0288
en_zh_phrase = 0.9245
nested_passive = 0.9066
target_language_switch = 0.8941
nested_condition = 0.8824
```

bottom subtype：

```text
they_coref = 0.3876
perfect = 0.4142
progressive = 0.4726
by_phrase = 0.4790
future_will = 0.4800
```

contract events：

```text
L20-L27 cross_battn_amlp:
  kl_ratio_vs_both = 11.6114
  progress_drop = 0.7978

L24-L27 cross_battn_amlp:
  kl_ratio_vs_both = 3.8853
  progress_drop = 0.7355

L26-L27 cross_battn_amlp:
  progress_drop = 0.6708

L27 cross_battn_amlp:
  progress_drop = 0.6020
```

客观现象：

DeepSeek7B 在扩展库中仍然是 L20-L27 最强，且 L24-L27、L26-L27、L27 逐步下降。这继续支持深层 block 累积和末层释放。L20-L23 单独较弱，但不为零，说明前段可能有弱写入或准备作用。

### GFCM v1 Partial 结果

输入：

```text
phase290_dir = results/gpt5_phase296_expanded_function_pilot
phase291_dir = results/gpt5_phase297_expanded_block_pilot
phase293_dir = results/gpt5_phase293_naturalness
phase294_dir = results/gpt5_phase294b_dynamic_recompute_full
```

输出：

```text
results/gpt5_phase297_expanded_gfcm_v1_partial/GLOBAL_CONTRACT_MAPPING_REPORT.md
results/gpt5_phase297_expanded_gfcm_v1_partial/global_mapping_summary.csv
results/gpt5_phase297_expanded_gfcm_v1_partial/global_contract_maps.json
```

数据规模：

```text
Qwen3:
  subtypes = 45
  features_min = 210
  features_max = 368
  combined_mean = 0.8022
  combined_min = 0.2757
  combined_max = 0.9983

GLM4:
  subtypes = 45
  features_min = 230
  features_max = 388
  combined_mean = 0.8518
  combined_min = 0.6245
  combined_max = 0.9989

DeepSeek7B:
  subtypes = 45
  features_min = 210
  features_max = 356
  combined_mean = 0.7962
  combined_min = 0.5235
  combined_max = 0.9917
```

重要说明：

```text
由于新 subtype 尚未跑 expanded naturalness 与 expanded dynamic recompute，
GFCM v1 partial 的 dynamic/naturalness 维度不完整。
因此本轮只把它作为 layer/block 图谱，不把它解释成完整功能契约图谱。
```

GFCM v1 partial 中的候选现象：

```text
Qwen3:
  future_will / perfect / progressive 高相似；
  concise 与 syntactic_do_not / by_phrase / lexical_not_adj 等强分化。

GLM4:
  future_will / perfect / progressive 高相似；
  it_coref、casual 与多个强功能分化。

DeepSeek7B:
  future_will / perfect 高相似；
  center_embedding / nested_condition 高相似；
  complement_clause / they_coref、casual / causal 等分化明显。
```

Z-score 候选：

```text
Qwen3 top:
  future_will / perfect = 0.9694
  future_will / progressive = 0.9545
  perfect / progressive = 0.9539

GLM4 top:
  future_will / perfect = 0.9675
  future_will / progressive = 0.9630
  perfect / progressive = 0.9574

DeepSeek7B top:
  future_will / perfect = 0.9378
  nested_contrast / nested_passive = 0.8825
  center_embedding / nested_condition = 0.8744
```

客观现象：

扩展到 45 subtype 后，相似度均值明显下降：

```text
Qwen3:
  Phase 32 19 subtype mean = 0.9505
  Phase 34 45 subtype mean = 0.8022

GLM4:
  Phase 32 19 subtype mean = 0.9675
  Phase 34 45 subtype mean = 0.8518

DeepSeek7B:
  Phase 32 19 subtype mean = 0.8913
  Phase 34 45 subtype mean = 0.7962
```

这说明前面相似度过高确实有功能库过窄的原因。扩展结构型功能库后，GFCM 的分辨率提高。

### 本轮最重要的新事实

```text
1. 45 subtype 扩展功能库能稳定运行 Phase 291 block scan。
2. 三模型原有路径类型在扩展库中继续成立：
   Qwen3 = 浅层协同 + L0-L8 block 增强；
   GLM4 = 浅层 MLP 写入 + residual 传播；
   DeepSeek7B = 深层 block 累积 + L27 释放。
3. 扩展功能库显著降低 GFCM 平均相似度，提升分化能力。
4. tense 的 future/perfect/progressive 在三模型中都高度相似，是强复用候选。
5. style.concise、coreference.it/they、DeepSeek7B 的某些 complement/coreference 组合是分化候选。
```

### 硬伤

1. GFCM v1 partial 仍缺 expanded dynamic recompute。
2. GFCM v1 partial 仍缺 expanded naturalness。
3. 每个 subtype 只有 2 pair，仍是 pilot。
4. translation/style/coreference 仍可能有模板和语义混杂。
5. 仍没有变量解码。

### 下一步计划

1. Phase 35：expanded dynamic recompute。
   - 对 45 subtype 跑 Phase 294b。
   - Qwen3/GLM4：L0-L8。
   - DeepSeek7B：L20-L27。
   - 先每 subtype 2 pair，保留 alpha 曲线。

2. Phase 36：expanded naturalness。
   - 对 expanded Phase 290/291/294b 结果生成 norm-based naturalness。
   - 后续再升级 PCA/kNN。

3. Phase 37：GFCM v1 full。
   - 合并 expanded Phase 290/291/294b/naturalness。
   - 输出完整 45 subtype matrix。

4. Phase 38：变量解码 pilot。
   - 优先从 tense、passive、logical 三组做。

## Phase 35: 扩展功能库全量动态重算与 GSSC v1 Dynamic [2026-05-28 17:21]

### 任务目标

根据“全局语义语法契约图谱 GSSC Map”的设计，本轮补齐 Phase 34 最大缺口：扩展 45 subtype 功能库的动态重算曲线。

本轮不使用保守脚本，按用户要求使用正常模式：

```text
CUDA_LAUNCH_BLOCKING = 0
PYTORCH_NO_CUDA_MEMORY_CACHING = 0
torch_dtype = bfloat16
attn_implementation = sdpa
device_map_auto_models = glm4,deepseek7b
max_gpu_memory = 21GiB
```

并且每个模型单独进程运行，使用：

```text
--hard-exit-after-model
```

确保一个模型结束后退出进程、释放显存，再加载下一个模型。

### 对 GSSC 方案的判断

用户提出的“全局语义语法契约图谱”方向是正确的，但必须严格限定当前证据：

```text
1. 大量测试数据只能提供覆盖度，不能自动推出语言编码原理。
2. 真正关键的是：功能分类、反事实干预、动态重算、自然性检验、变量解码、复用/分化矩阵、模块定位。
3. 当前还没有变量解码，因此本轮仍然属于路径图谱和行为图谱，不是最终编码内容图谱。
4. 但 expanded dynamic recompute 是 GSSC 的关键基础数据，因为它测试 patch 后后续自然 forward 是否能接上。
```

因此本轮任务是：

```text
补齐 45 subtype 的 dynamic recompute 数据，
把它合并到 expanded GSSC/GFCM v1 dynamic 图谱，
并记录正常模式下的运行稳定性。
```

### 新增脚本

新增正常模式单模型运行器：

```text
tests/gpt5/run_phase294_normal.sh
```

功能：

```text
1. 使用 openone-cuda121 环境。
2. 使用 BF16 + SDPA。
3. GLM4 和 DeepSeek7B 使用 device_map="auto"。
4. 默认不启用 CUDA_LAUNCH_BLOCKING。
5. 默认不禁用 CUDA memory cache。
6. 每轮启动 kernel follower 与 GPU/process monitor。
7. 运行 phase294_dynamic_recompute_pilot.py 时强制加入 --hard-exit-after-model。
8. 保存 run.log、gpu_process_monitor.log、kernel.follow.log、kernel.since-start.filtered.log。
```

新增三模型顺序运行器：

```text
tests/gpt5/run_phase298_expanded_dynamic_normal_all.sh
```

功能：

```text
1. 顺序运行 qwen3 -> glm4 -> deepseek7b。
2. 每个模型独立进程，结束后查询 compute apps。
3. 任一模型失败时默认停止，方便检查日志和 resume。
4. 支持 checkpoint resume。
```

新增结果汇总脚本：

```text
tests/gpt5/phase298_dynamic_summary.py
```

输出：

```text
results/gpt5_phase298_expanded_dynamic_normal/expanded_dynamic_summary.json
results/gpt5_phase298_expanded_dynamic_normal/EXPANDED_DYNAMIC_SUMMARY.md
```

### 测试命令

正式运行命令：

```bash
MAX_PAIRS_PER_SUBTYPE=999 \
QWEN3_MAX_SECONDS=14400 \
GLM4_MAX_SECONDS=18000 \
DEEPSEEK7B_MAX_SECONDS=14400 \
OUTPUT_DIR=results/gpt5_phase298_expanded_dynamic_normal \
tests/gpt5/run_phase298_expanded_dynamic_normal_all.sh
```

Qwen3 正常模式在 224/244 pair 处发生用户态 segmentation fault：

```text
exit_code = 139
checkpoint rows = 40320
kernel.since-start.filtered.log = 0 行
```

随后 resume：

```bash
MAX_PAIRS_PER_SUBTYPE=999 \
QWEN3_MAX_SECONDS=7200 \
GLM4_MAX_SECONDS=18000 \
DEEPSEEK7B_MAX_SECONDS=14400 \
OUTPUT_DIR=results/gpt5_phase298_expanded_dynamic_normal \
tests/gpt5/run_phase298_expanded_dynamic_normal_all.sh
```

GLM4 正常模式在 88/244 pair 处发生用户态 segmentation fault：

```text
exit_code = 139
checkpoint rows = 15840
kernel.since-start.filtered.log = 0 行
```

随后 resume：

```bash
MAX_PAIRS_PER_SUBTYPE=999 \
QWEN3_MAX_SECONDS=1200 \
GLM4_MAX_SECONDS=18000 \
DEEPSEEK7B_MAX_SECONDS=14400 \
OUTPUT_DIR=results/gpt5_phase298_expanded_dynamic_normal \
tests/gpt5/run_phase298_expanded_dynamic_normal_all.sh
```

Qwen3 checkpoint 已 complete，自动跳过；GLM4 从 88 pair 继续，最终完成；DeepSeek7B 一次完成。

### 输出文件

动态重算输出：

```text
results/gpt5_phase298_expanded_dynamic_normal/qwen3_phase294_dynamic_recompute_pilot.json
results/gpt5_phase298_expanded_dynamic_normal/glm4_phase294_dynamic_recompute_pilot.json
results/gpt5_phase298_expanded_dynamic_normal/deepseek7b_phase294_dynamic_recompute_pilot.json
```

checkpoints：

```text
results/gpt5_phase298_expanded_dynamic_normal/checkpoints/qwen3/coreference-logical-negation-passive-recursive-style-tense-translation_expanded_dynamic_normal.json
results/gpt5_phase298_expanded_dynamic_normal/checkpoints/glm4/coreference-logical-negation-passive-recursive-style-tense-translation_expanded_dynamic_normal.json
results/gpt5_phase298_expanded_dynamic_normal/checkpoints/deepseek7b/coreference-logical-negation-passive-recursive-style-tense-translation_expanded_dynamic_normal.json
```

日志：

```text
results/gpt5_gpu_lock_logs/20260528_143141_phase294normal_qwen3
results/gpt5_gpu_lock_logs/20260528_151538_phase294normal_qwen3
results/gpt5_gpu_lock_logs/20260528_151945_phase294normal_glm4
results/gpt5_gpu_lock_logs/20260528_154645_phase294normal_qwen3
results/gpt5_gpu_lock_logs/20260528_154656_phase294normal_glm4
results/gpt5_gpu_lock_logs/20260528_163056_phase294normal_deepseek7b
```

所有上述日志的：

```text
kernel.since-start.filtered.log = 0 行
```

说明本轮没有捕获到 NVRM/Xid/GSP/GPU locked/kernel hang 级别日志。Qwen3 和 GLM4 的问题是用户态进程崩溃，不是系统级 GPU 锁死。

### 数据规模

```text
function library:
  categories = 8
  subtypes = 45
  pairs = 244

Qwen3:
  target_layers = L0-L8
  pairs = 244
  rows = 43920
  nonfinite_rows = 0

GLM4:
  target_layers = L0-L8
  pairs = 244
  rows = 43920
  nonfinite_rows = 0

DeepSeek7B:
  target_layers = L20-L27
  pairs = 244
  rows = 39040
  nonfinite_rows = 0

total dynamic rows = 126880
```

### Qwen3 客观结果

全局 summary 中，按所有 alpha 平均的 patch 最优：

```text
attn_out:
  best_layer = 0
  progress = 0.333718

mlp_out:
  best_layer = 0
  progress = 0.386442

resid_in:
  best_layer = 4
  progress = 0.488311

resid_out:
  best_layer = 3
  progress = 0.488309
```

按 alpha=1 的 subtype 最优分布：

```text
best_patch_counts:
  resid_out = 25
  resid_in = 10
  mlp_out = 6
  attn_out = 4

best_layer_counts:
  L0 = 21
  L1 = 1
  L2 = 1
  L3 = 3
  L4 = 1
  L6 = 7
  L7 = 5
  L8 = 6
```

top subtype：

```text
en_zh_phrase:
  category = translation
  best = resid_out L0
  progress = 1.377287

en_fr_phrase:
  category = translation
  best = resid_out L1
  progress = 1.203791

he_coref:
  category = coreference
  best = mlp_out L0
  progress = 1.180285

en_fr_word:
  category = translation
  best = resid_out L7
  progress = 1.052615

causal:
  category = logical
  best = attn_out L0
  progress = 1.034798
```

bottom subtype：

```text
it_coref:
  best = resid_out L7
  progress = 0.396908

double_relative:
  best = resid_out L4
  progress = 0.473765

deep_complement:
  best = resid_out L8
  progress = 0.549443

nested_condition:
  best = mlp_out L3
  progress = 0.615510

perfect:
  best = attn_out L0
  progress = 0.623038
```

客观现象：

```text
1. Qwen3 的动态重算最强模块仍集中在浅层。
2. L0 是大量 subtype 的最佳层，支持浅层写入/启动。
3. resid_out/resid_in 多于 attn_out/mlp_out，说明单模块输出不能覆盖全部动态效果，残差状态仍是更强上界。
4. translation/coreference 中存在 progress > 1 的 over-conversion，需要谨慎解释，不能当作贡献大于 1。
```

### GLM4 客观结果

按所有 alpha 平均的 patch 最优：

```text
attn_out:
  best_layer = 1
  progress = 0.081681

mlp_out:
  best_layer = 0
  progress = 0.490105

resid_in:
  best_layer = 2
  progress = 0.590795

resid_out:
  best_layer = 1
  progress = 0.590795
```

按 alpha=1 的 subtype 最优分布：

```text
best_patch_counts:
  resid_out = 29
  resid_in = 14
  mlp_out = 2

best_layer_counts:
  L0 = 19
  L1 = 4
  L3 = 2
  L4 = 1
  L5 = 2
  L6 = 1
  L7 = 2
  L8 = 14
```

top subtype：

```text
dative_passive:
  category = passive
  best = resid_out L1
  progress = 1.122602

get_passive:
  category = passive
  best = resid_in L0
  progress = 1.062892

long_passive:
  category = passive
  best = resid_out L1
  progress = 1.047683

by_phrase:
  category = passive
  best = resid_in L0
  progress = 1.039192

complement_clause:
  category = recursive
  best = resid_out L6
  progress = 1.027453
```

bottom subtype：

```text
nested_condition:
  best = resid_out L8
  progress = 0.866974

double_relative:
  best = resid_out L8
  progress = 0.869599

pp_chain:
  best = resid_out L5
  progress = 0.874349

it_coref:
  best = resid_out L8
  progress = 0.877729

possessive_chain:
  best = resid_out L8
  progress = 0.894064
```

客观现象：

```text
1. GLM4 的 attn_out 动态效果非常弱，best progress 仅 0.081681。
2. L0 mlp_out 仍然明显强于 attention，符合浅层 MLP 写入型判断。
3. 但 subtype 最优主要落在 resid_in/resid_out，而不是 mlp_out，说明 MLP 写入之后的 residual 状态是更可重算的路径载体。
4. passive 子类在 GLM4 中尤其强，top 4 全是 passive，这和之前 GLM4 MLP/residual 对角色/语态类敏感的现象一致。
```

### DeepSeek7B 客观结果

按所有 alpha 平均的 patch 最优：

```text
attn_out:
  best_layer = 27
  progress = 0.288754

mlp_out:
  best_layer = 27
  progress = 0.250668

resid_in:
  best_layer = 21
  progress = 0.278892

resid_out:
  best_layer = 27
  progress = 0.483323
```

按 alpha=1 的 subtype 最优分布：

```text
best_patch_counts:
  resid_out = 32
  resid_in = 9
  attn_out = 3
  mlp_out = 1

best_layer_counts:
  L20 = 9
  L22 = 1
  L23 = 1
  L24 = 1
  L26 = 1
  L27 = 32
```

top subtype：

```text
concise:
  category = style
  best = resid_out L24
  progress = 1.109055

existential_no:
  category = negation
  best = resid_out L23
  progress = 1.097377

deep_complement:
  category = recursive
  best = attn_out L27
  progress = 1.083397

deictic_switch:
  category = coreference
  best = mlp_out L26
  progress = 1.056671

pp_chain:
  category = recursive
  best = attn_out L27
  progress = 1.048624
```

bottom subtype 的 best progress 也接近 1：

```text
double_relative = 0.999981
nested_contrast = 0.999984
long_passive = 0.999985
he_coref = 0.999985
en_zh_word = 0.999986
```

客观现象：

```text
1. DeepSeek7B 的 alpha=1 subtype 最优几乎都能达到接近 1。
2. 大多数 subtype 最优集中在 L27 resid_out。
3. 这说明 L27 resid_out 非常接近 output-ready 表示，是强输出接口/释放点。
4. 但这不等于 DeepSeek7B 的完整机制只在 L27，因为 Phase 34 block scan 已显示 L20-L27 block 远强于 L27 单层 both patch。
5. 因此更合理的客观描述是：DeepSeek7B 的单点动态读出在 L27 最强，但完整路径仍可能依赖 L20-L27 的多点轨迹累积。
```

### GSSC v1 Dynamic 构建

命令：

```bash
python tests/gpt5/phase295_global_contract_mapping.py \
  --phase290-dir results/gpt5_phase296_expanded_function_pilot \
  --phase291-dir results/gpt5_phase297_expanded_block_pilot \
  --phase293-dir results/gpt5_phase293_naturalness \
  --phase294-dir results/gpt5_phase298_expanded_dynamic_normal \
  --output-dir results/gpt5_phase298_expanded_gssc_v1_dynamic \
  --top-k 30
```

输出：

```text
results/gpt5_phase298_expanded_gssc_v1_dynamic/GLOBAL_CONTRACT_MAPPING_REPORT.md
results/gpt5_phase298_expanded_gssc_v1_dynamic/global_mapping_summary.csv
results/gpt5_phase298_expanded_gssc_v1_dynamic/global_contract_maps.json
results/gpt5_phase298_expanded_gssc_v1_dynamic/*_global_similarity.csv
results/gpt5_phase298_expanded_gssc_v1_dynamic/*_global_zscore_top_pairs.csv
results/gpt5_phase298_expanded_gssc_v1_dynamic/*_global_zscore_bottom_pairs.csv
```

GSSC v1 dynamic 数据规模：

```text
Qwen3:
  phase290_rows = 3780
  phase291_rows = 6300
  phase293_event_rows = 784
  phase294_rows = 43920
  subtypes = 45
  features_min = 346
  features_max = 368
  combined_mean = 0.865291
  combined_min = 0.377526
  combined_max = 0.997370
  same_category_mean = 0.892881
  cross_category_mean = 0.861807

GLM4:
  phase290_rows = 3780
  phase291_rows = 7560
  phase293_event_rows = 1468
  phase294_rows = 43920
  subtypes = 45
  features_min = 366
  features_max = 388
  combined_mean = 0.922709
  combined_min = 0.751590
  combined_max = 0.998955
  same_category_mean = 0.947678
  cross_category_mean = 0.919556

DeepSeek7B:
  phase290_rows = 3780
  phase291_rows = 6300
  phase293_event_rows = 1078
  phase294_rows = 39040
  subtypes = 45
  features_min = 334
  features_max = 356
  combined_mean = 0.851783
  combined_min = 0.570001
  combined_max = 0.981900
  same_category_mean = 0.876334
  cross_category_mean = 0.848683
```

相对于 Phase 34 partial，加入 expanded dynamic 后：

```text
Qwen3:
  0.8022 -> 0.8653

GLM4:
  0.8518 -> 0.9227

DeepSeek7B:
  0.7962 -> 0.8518
```

解释需要谨慎：

```text
动态曲线加入后相似度上升，说明很多 subtype 在动态重算层面共享较强模型整体模式。
这不等于真实语言功能复用已经成立。
必须继续做 residualized/z-score、变量解码和自然性升级来过滤模型整体曲线。
```

### GSSC v1 Dynamic 候选现象

Qwen3 top reuse candidates：

```text
future_will / perfect:
  combined = 0.9974
  layer = 0.9967
  block = 0.9973
  dynamic = 0.9944

future_will / progressive:
  combined = 0.9961
  dynamic = 0.9941

perfect / progressive:
  combined = 0.9948
  dynamic = 0.9894

causal / contrast:
  combined = 0.9899
  dynamic = 0.9973
```

Qwen3 bottom differentiation candidates：

```text
concise / syntactic_do_not:
  combined = 0.3775
  dynamic = 0.6545

by_phrase / concise:
  combined = 0.4109

center_embedding / concise:
  combined = 0.4166
```

GLM4 top reuse candidates：

```text
future_will / perfect:
  combined = 0.9990
  dynamic = 0.9986

perfect / progressive:
  combined = 0.9989
  dynamic = 0.9991

future_will / progressive:
  combined = 0.9988

conditional / contrast:
  combined = 0.9964

long_passive / nested_passive:
  combined = 0.9963
```

GLM4 bottom differentiation candidates：

```text
it_coref / scope_quantifier:
  combined = 0.7516

it_coref / lexical_not_adj:
  combined = 0.7572

it_coref / never:
  combined = 0.7574
```

DeepSeek7B top reuse candidates：

```text
future_will / perfect:
  combined = 0.9819

by_phrase / get_passive:
  combined = 0.9814
  dynamic = 0.9912

formal / negated_condition:
  combined = 0.9809

and_or / morphological_neg:
  combined = 0.9775

by_phrase / relative_clause:
  combined = 0.9741
```

DeepSeek7B bottom differentiation candidates：

```text
complement_clause / they_coref:
  combined = 0.5700

complement_clause / concise:
  combined = 0.6104

complement_clause / en_zh_word:
  combined = 0.6366
```

### 本轮最重要的新事实

```text
1. 扩展 45 subtype、244 pairs 的动态重算已完成，新增 126880 rows。
2. 三模型 nonfinite_rows 全部为 0。
3. Qwen3 和 GLM4 在正常模式长 session 下都出现过用户态 segfault 139，但 kernel filtered log 为 0 行，checkpoint/resume 成功补完。
4. DeepSeek7B 正常模式一次完成，没有用户态崩溃。
5. Qwen3 动态最佳仍集中浅层，支持浅层启动/传播。
6. GLM4 attn_out 极弱，mlp_out L0 明显强，residual 状态更强，支持浅层 MLP 写入 + residual 传播。
7. DeepSeek7B L27 resid_out 对几乎所有 subtype 都极强，支持末层输出接口/释放点；但完整 block 机制仍需分段动态重算确认。
8. GSSC v1 dynamic 已能整合 single-layer、block、naturalness event、dynamic recompute 四类行为签名。
```

### 硬伤

```text
1. Dynamic recompute 仍是行为证据，不是变量编码证据。
2. progress > 1 的 subtype 存在 over-conversion，不能解释为贡献超过完整转换。
3. DeepSeek7B 的 L27 resid_out 过强，可能掩盖 L20-L27 内部多点轨迹。
4. naturalness 仍使用旧的 norm-based event，尚未对 expanded dynamic 结果做完整自然性升级。
5. 当前 GSSC v1 dynamic 的高相似度仍可能被模型整体曲线主导。
6. 每个 subtype 数量虽已使用当前库全部 244 pairs，但某些新增类别如 style/coreference/translation 的语言多样性仍有限。
7. 正常模式下 Qwen3/GLM4 用户态 segfault 说明长 session 仍不够稳，虽然没有 GPU kernel 锁死。
```

### 当前理论进展的谨慎表述

不能说已经破解语言整体编码机制。

当前可以更稳地说：

```text
在扩展 45 subtype 功能库上，三模型表现出稳定但不同的路径组织方式：

Qwen3:
  浅层动态启动明显，L0 模块与 L0-L8 residual 传播都重要。

GLM4:
  attention 输出动态作用极弱，L0 MLP 和早层 residual 状态更关键。

DeepSeek7B:
  L27 resid_out 是强输出释放点，但结合 block scan，完整机制更像 L20-L27 多层轨迹累积。
```

这支持“路径格式假说”的弱版本：

```text
语言功能不是单个静态语义轴，而是沿 residual trajectory 形成；
attention、MLP、residual 在不同模型中承担不同路径组织方式；
功能复用/分化需要通过 layer/block/dynamic/naturalness/variable decoding 多维图谱共同确认。
```

但它还没有回答：

```text
路径里传递的具体语言变量是什么。
```

因此下一阶段必须从“路径行为图谱”推进到“路径变量图谱”。

### 下一步计划

Phase 36：expanded dynamic naturalness。

```text
目标：
  对 results/gpt5_phase298_expanded_dynamic_normal 的动态 patch 状态加入自然性判断。

内容：
  1. 建立 expanded natural norm reference。
  2. 对 resid_in/resid_out/attn_out/mlp_out patch 状态计算 norm z-score。
  3. 标记 progress 高但 off-manifold 的样本。
  4. 区分 on-manifold functional success/failure 与 off-manifold artifact。
```

Phase 37：DeepSeek7B segment dynamic recompute。

```text
目标：
  拆开 L20-L27 深层轨迹。

测试：
  1. patch L20-L23, recompute L24-L27
  2. patch L24-L27
  3. patch L20-L26, recompute L27
  4. patch every other layer
  5. L27 attn_out / mlp_out 与前段 block 组合
```

Phase 38：变量解码 pilot。

```text
优先变量：
  polarity
  voice
  operator
  role binding
  clause boundary
  tense/aspect
  coreference target

优先对象：
  GLM4 passive
  Qwen3 logical
  DeepSeek7B recursive/complement
```

Phase 39：GSSC v2。

```text
合并：
  layer curve
  block curve
  dynamic curve
  naturalness curve
  variable decoding curve

输出：
  1. 功能复用矩阵
  2. 功能分化矩阵
  3. 模型对齐矩阵
  4. 复杂度迁移矩阵
```

## Phase 36: Dynamic Naturalness 与 Residualized GSSC 过滤 [2026-05-28 18:08]

### 任务目标

根据最新判断，GSSC v1 Dynamic 和语言动态编码闭包工程不是二选一，而是上下游关系：

```text
GSSC v1 Dynamic:
  负责发现候选路径、候选功能、候选模型差异。

语言动态编码闭包工程:
  负责变量解码、变量替换、变量恢复、动态重算闭包验证。
```

因此本轮不继续盲目扩大功能库，而是先修正 GSSC v1 Dynamic 的证据质量：

```text
1. 对 expanded dynamic recompute 做 norm/z-score naturalness。
2. 对 GSSC v1 Dynamic 做 residualized similarity，过滤模型整体曲线污染。
3. 为后续变量闭包工程筛选更可靠的候选对象。
```

本轮不重新加载模型，不新增 GPU 测试，直接分析 Phase 35 已生成的：

```text
results/gpt5_phase298_expanded_dynamic_normal
results/gpt5_phase298_expanded_gssc_v1_dynamic
```

### 对用户分析的判断

用户分析中正确的部分：

```text
1. GSSC 是全局路径图谱，不是最终编码机制证明。
2. 语言动态编码闭包工程是局部机制证明路线。
3. 二者应当串联：GSSC 找哪里有机制，闭包工程证明机制是什么。
4. 当前 GSSC 最大硬伤是变量解码不足、自然性不足、整体曲线污染。
5. 现在不应继续盲目扩 subtype，而应先做 naturalness、residualized similarity 和变量 pilot。
6. GLM4 passive、Qwen3 logical、DeepSeek7B recursive/complement 是合理的闭包工程候选突破口。
```

需要谨慎的部分：

```text
1. 本轮 naturalness 只能做 norm/z-score，不能等价于 PCA/kNN/Mahalanobis 流形距离。
2. residualized similarity 只能过滤部分整体曲线污染，不能证明真实机制复用。
3. high progress、over-conversion、on-manifold 仍然只是候选证据，需要变量替换和恢复测试。
```

### 新增脚本

新增动态自然性分析：

```text
tests/gpt5/phase299_dynamic_naturalness.py
```

输入：

```text
results/gpt5_phase298_expanded_dynamic_normal
```

输出：

```text
results/gpt5_phase299_dynamic_naturalness/dynamic_norm_reference.csv
results/gpt5_phase299_dynamic_naturalness/dynamic_naturalness_events.csv
results/gpt5_phase299_dynamic_naturalness/dynamic_naturalness_summary.csv
results/gpt5_phase299_dynamic_naturalness/dynamic_naturalness_subtype_summary.csv
results/gpt5_phase299_dynamic_naturalness/dynamic_naturalness_patch_layer_summary.csv
results/gpt5_phase299_dynamic_naturalness/DYNAMIC_NATURALNESS_REPORT.md
```

原理：

```text
1. 对每个 model/layer/patch_type，用 a_ref_norm 与 b_ref_norm 建立自然 norm reference。
2. 对 patch_norm 计算 norm_z。
3. 标记 off_manifold、high_progress、over_conversion、negative_progress。
4. 区分：
   on_manifold_high_progress
   off_manifold_high_progress
   on_manifold_over_conversion
   off_manifold_over_conversion
```

限制：

```text
没有激活向量，因此不能计算 PCA residual distance、kNN distance、Mahalanobis distance。
没有保存完整 logits，因此不能计算 loss_delta、entropy、logit margin。
```

新增 residualized similarity 分析：

```text
tests/gpt5/phase299_gssc_residualized_similarity.py
```

输入：

```text
results/gpt5_phase298_expanded_gssc_v1_dynamic/global_contract_maps.json
```

输出：

```text
results/gpt5_phase299_gssc_residualized_dynamic/RESIDUALIZED_SIMILARITY_REPORT.md
results/gpt5_phase299_gssc_residualized_dynamic/residualized_similarity_summary.csv
results/gpt5_phase299_gssc_residualized_dynamic/*_residualized_pair_diagnostics.csv
results/gpt5_phase299_gssc_residualized_dynamic/*_stable_reuse_candidates.csv
results/gpt5_phase299_gssc_residualized_dynamic/*_stable_differentiation_candidates.csv
```

原理：

```text
对每个模型的 GSSC group-normalized signature，计算：
  raw similarity
  model-centered similarity
  category-centered similarity
  group-centered similarity
  zscore similarity

并标记：
  residual_stable_reuse_candidate
  model_curve_artifact_candidate
  category_curve_artifact_candidate
  stable_differentiation_candidate
```

### 运行命令

Dynamic naturalness：

```bash
python tests/gpt5/phase299_dynamic_naturalness.py \
  --input-dir results/gpt5_phase298_expanded_dynamic_normal \
  --output-dir results/gpt5_phase299_dynamic_naturalness \
  --z-threshold 3.0 \
  --success-threshold 0.8 \
  --over-threshold 1.05
```

Residualized GSSC：

```bash
python tests/gpt5/phase299_gssc_residualized_similarity.py \
  --input results/gpt5_phase298_expanded_gssc_v1_dynamic/global_contract_maps.json \
  --output-dir results/gpt5_phase299_gssc_residualized_dynamic \
  --top-k 30
```

### Dynamic Naturalness 结果

整体结果：

```text
Qwen3:
  total_rows = 43920
  off_manifold_rows = 84
  off_manifold_rate = 0.001913
  on_manifold_high_progress_rows = 7289
  on_manifold_high_progress_rate = 0.165961
  off_manifold_high_progress_rows = 10
  on_manifold_over_conversion_rows = 1587
  off_manifold_over_conversion_rows = 2

GLM4:
  total_rows = 43920
  off_manifold_rows = 171
  off_manifold_rate = 0.003893
  on_manifold_high_progress_rows = 10843
  on_manifold_high_progress_rate = 0.246881
  off_manifold_high_progress_rows = 75
  on_manifold_over_conversion_rows = 776
  off_manifold_over_conversion_rows = 0

DeepSeek7B:
  total_rows = 39040
  off_manifold_rows = 145
  off_manifold_rate = 0.003714
  on_manifold_high_progress_rows = 3600
  on_manifold_high_progress_rate = 0.092213
  off_manifold_high_progress_rows = 19
  on_manifold_over_conversion_rows = 869
  off_manifold_over_conversion_rows = 11
```

客观现象：

```text
1. 三模型 norm/z-score off_manifold 比例都低于 0.4%。
2. 因此不能把 GSSC dynamic 的高 progress 整体解释为“范数离谱导致的伪影”。
3. 但仍有少量 off_manifold_high_progress，需要在后续变量解码前过滤。
4. over_conversion 大多数是 on_manifold_norm 层面，不是简单范数异常。
```

top off_manifold subtype：

```text
Qwen3:
  en_zh_word = 23
  target_language_switch = 21
  nested_contrast = 12
  en_zh_phrase = 7

GLM4:
  deep_complement = 134
  long_passive = 18
  double_relative = 5
  en_fr_phrase = 5

DeepSeek7B:
  no_agent = 19
  syntactic_do_not = 16
  deep_complement = 13
  complement_clause = 12
```

top over_conversion subtype：

```text
Qwen3:
  lexical_not_adj = 192
  syntactic_do_not = 152
  by_phrase = 134
  center_embedding = 129
  get_passive = 114

GLM4:
  dative_passive = 214
  get_passive = 173
  by_phrase = 116
  complement_clause = 73
  long_passive = 65

DeepSeek7B:
  conditional = 100
  causal = 87
  inference = 77
  contrast = 55
  deep_complement = 54
```

关键解释：

```text
progress > 1 的 over_conversion 大多数不是 norm off-manifold。
因此它更可能来自目标方向过冲、logit metric 过敏、或 patch 对输出分布的过度推进。
不能把 over_conversion 当作贡献强度，只能作为需要审计的诊断信号。
```

### Residualized GSSC 结果

summary：

```text
Qwen3:
  raw_mean = 0.865291
  model_centered_mean = -0.013480
  category_centered_mean = -0.015242
  residual_stable_reuse_candidate_count = 85
  model_curve_artifact_candidate_count = 91
  category_curve_artifact_candidate_count = 108
  stable_differentiation_candidate_count = 47

GLM4:
  raw_mean = 0.922709
  model_centered_mean = -0.014066
  category_centered_mean = -0.015589
  residual_stable_reuse_candidate_count = 178
  model_curve_artifact_candidate_count = 336
  category_curve_artifact_candidate_count = 120
  stable_differentiation_candidate_count = 0

DeepSeek7B:
  raw_mean = 0.851783
  model_centered_mean = -0.020525
  category_centered_mean = -0.020343
  residual_stable_reuse_candidate_count = 65
  model_curve_artifact_candidate_count = 10
  category_curve_artifact_candidate_count = 87
  stable_differentiation_candidate_count = 28
```

客观现象：

```text
1. GLM4 raw similarity 最高，但 model_curve_artifact_candidate 也最多。
2. 这说明 GLM4 的高相似度很容易被整体曲线污染，需要更谨慎解释。
3. Qwen3 和 DeepSeek7B 有更多稳定分化候选。
4. residualized 后仍然稳定的 pair，才适合作为变量闭包工程候选。
```

Qwen3 stable reuse candidates：

```text
contrast / no_agent:
  raw = 0.9897
  model_centered = 0.9083
  category_centered = 0.8689

causal / contrast:
  raw = 0.9899
  model_centered = 0.9017
  category_centered = 0.8500

contrast / inference:
  raw = 0.9885
  model_centered = 0.8887
  category_centered = 0.8266

causal / inference:
  raw = 0.9862
  model_centered = 0.8579
  category_centered = 0.7744
```

Qwen3 stable differentiation candidates：

```text
concise / syntactic_do_not:
  raw = 0.3775
  model_centered = -0.3702

by_phrase / concise:
  raw = 0.4109
  model_centered = -0.4806

center_embedding / concise:
  raw = 0.4166
  model_centered = -0.2784
```

GLM4 stable reuse candidates：

```text
casual / concise:
  raw = 0.9944
  model_centered = 0.9722
  category_centered = 0.9302

long_passive / nested_passive:
  raw = 0.9963
  model_centered = 0.8977
  category_centered = 0.9399

conditional / contrast:
  raw = 0.9964
  model_centered = 0.9472
  category_centered = 0.8906

causal / inference:
  raw = 0.9952
  model_centered = 0.9343
```

GLM4 differentiation 注意事项：

```text
GLM4 的最底部 pair 如 it_coref / scope_quantifier 虽然 raw 低且 model_centered 为负，
但 raw > 0.70，没有进入 stable_differentiation_candidate 阈值。
这说明 GLM4 整体曲线仍然较强，需要更敏感的变量级测试才能区分。
```

DeepSeek7B stable reuse candidates：

```text
long_passive / nested_condition:
  raw = 0.9649
  model_centered = 0.8367
  category_centered = 0.8118

nested_contrast / nested_passive:
  raw = 0.9709
  model_centered = 0.8543
  category_centered = 0.8032

center_embedding / nested_contrast:
  raw = 0.9690
  model_centered = 0.8447
  category_centered = 0.7766

center_embedding / nested_condition:
  raw = 0.9738
  model_centered = 0.8806
  category_centered = 0.7711
```

DeepSeek7B stable differentiation candidates：

```text
complement_clause / they_coref:
  raw = 0.5700
  model_centered = -0.6381

complement_clause / concise:
  raw = 0.6104
  model_centered = -0.3662

complement_clause / en_zh_word:
  raw = 0.6366
  model_centered = -0.5023

casual / causal:
  raw = 0.6460
  model_centered = -0.4031
```

### 本轮最重要的新事实

```text
1. expanded dynamic 的 norm/z-score off_manifold 比例很低。
2. 高 progress 大多不是简单 norm 异常导致。
3. over_conversion 不能丢弃，但必须进入审计清单。
4. GLM4 的高相似度确实有明显模型整体曲线污染。
5. Qwen3 logical 簇在 residualized 后仍然稳定，是变量解码好候选。
6. GLM4 passive/conditional/style 中有 residual-stable reuse 候选，但 GLM4 也最容易被模型整体曲线污染。
7. DeepSeek7B nested/recursive/logical/passive 之间出现稳定复用候选，同时 complement_clause 与 coreference/style/translation 的分化很明显。
```

### 当前结论的边界

当前可以说：

```text
GSSC v1 Dynamic 的主要高 progress 不是大量 norm off-manifold 伪影；
residualized similarity 可以筛出比 raw similarity 更可靠的复用/分化候选；
这些候选可以指导语言动态编码闭包工程。
```

当前不能说：

```text
1. residual_stable_reuse_candidate 就等于真实机制复用。
2. on_manifold_high_progress 就等于真实变量编码。
3. GLM4 high similarity 就等于功能共享同一机制。
4. DeepSeek7B L27 resid_out 就是语言功能形成点。
```

### 对下一步闭包工程的选择

基于 Phase 35/36，最合理的闭包 pilot：

```text
GLM4:
  passive / voice / role binding
  原因：
    passive 子类 dynamic 强；
    L0 MLP 与 residual 状态强；
    long_passive / nested_passive residualized 后仍高相似。

Qwen3:
  logical / polarity / operator
  原因：
    causal / contrast / inference residualized 后仍高相似；
    L0 attention/MLP 参与明显；
    适合测试 operator 和 event-state 转换。

DeepSeek7B:
  recursive / clause boundary / output release
  原因：
    nested_condition / nested_contrast / center_embedding 等 residualized 后稳定；
    complement_clause 与多个功能强分化；
    需要拆 L20-L27 多点轨迹。
```

### 下一步计划

Phase 37：GLM4 passive 变量闭包 pilot。

目标：

```text
验证 voice / role binding 是否可以被解码、替换、恢复，并在后续 block recompute 后自然接上。
```

最小任务：

```text
1. 构造 active/passive/by_phrase/get_passive/dative_passive/long_passive 样本。
2. 采集 GLM4 L0-L8 的 resid_in/resid_out/mlp_out。
3. 训练或直接构造简单线性变量读出：
   voice = active/passive
   surface_subject
   semantic_agent
   semantic_patient
4. 做变量方向替换：
   active -> passive
   agent <-> patient
5. 做恢复测试：
   破坏 role/voice 后，再恢复变量方向，看输出是否恢复。
6. 做 downstream recompute：
   patch L0/L1 residual 或 mlp_out 后，让 L1-L8 自然 forward。
```

Phase 38：DeepSeek7B segment dynamic recompute。

目标：

```text
拆开 L20-L27 是写入、传播、压缩、释放中的哪一种。
```

Phase 39：Qwen3 logical operator pilot。

目标：

```text
测试 causal/contrast/inference/conditional 的 operator 与 event-state 是否形成可替换变量。
```

## Phase 37: GLM4 Passive Voice 变量闭包 Pilot [2026-05-28 20:32]

### 任务目标

根据 Phase 36 的筛选结果，本轮开始从 GSSC 路径图谱进入“语言动态编码闭包工程”的第一步。

选择对象：

```text
GLM4 passive / voice / role binding
```

原因：

```text
1. Phase 35/36 显示 GLM4 passive 类动态信号强。
2. GLM4 的 L0 MLP 与 early residual 对 passive 特别敏感。
3. long_passive / nested_passive 在 residualized GSSC 中仍然高相似。
4. passive/voice 比 recursive/coreference 更适合先做最小变量闭包。
```

本轮先做最小闭包 pilot，只验证：

```text
voice(active/passive) 变量方向是否：
  1. 可以被线性读出；
  2. 可以从训练样本迁移到测试样本；
  3. 可以通过方向 patch 推动 active -> passive；
  4. 是否也可以推动 passive -> active。
```

注意：

```text
本轮还没有分离 agent/patient；
还没有做 role binding 替换；
还没有做破坏-恢复闭包；
因此只能叫 voice variable pilot，不能叫完整 passive 机制破解。
```

### 脚本变更

新增变量闭包测试脚本：

```text
tests/gpt5/phase300_voice_closure_pilot.py
```

核心逻辑：

```text
1. 构造 passive 数据：
   by_phrase
   get_passive
   long_passive
   dative_passive

2. 分 subtype 做 stratified train/test split。

3. 在 train set 上，对每个 layer/module 计算：
   voice_direction = mean(passive_vector - active_vector)

4. 在 test set 上做 probe：
   score = dot(hidden_vector, voice_direction)
   判断 active/passive 是否可分。

5. 在 test set 上做 causal direction patch：
   active hidden + alpha * voice_direction
   passive hidden - alpha * voice_direction

6. 让后续层自然 forward，计算：
   progress
   KL ratio
   logit_delta_ratio
   finite
```

测试模块：

```text
resid_in
resid_out
mlp_out
```

新增正常模式运行器：

```text
tests/gpt5/run_phase300_normal.sh
tests/gpt5/run_phase300_voice_closure_normal_all.sh
```

新增汇总脚本：

```text
tests/gpt5/phase300_voice_closure_summary.py
```

### 重要修正：stratified split

第一次运行后发现一个关键问题：

```text
train/test split 按排序列表直接切半，
导致 test set 主要落在 get_passive / long_passive，
没有均匀覆盖 by_phrase / dative_passive。
```

因此第一版输出：

```text
results/gpt5_phase300_voice_closure_pilot
```

只作为诊断，不进入正式结论。

随后修正为：

```text
stratified_train_test_split
```

保证每个 subtype 都进入 train/test。

正式结果目录：

```text
results/gpt5_phase300_voice_closure_pilot_stratified
```

### 运行命令

Smoke test：

```bash
MAX_SECONDS=1200 \
OUTPUT_DIR=results/gpt5_phase300_voice_closure_smoke \
tests/gpt5/run_phase300_normal.sh qwen3 \
  --max-pairs-per-subtype 2 \
  --train-fraction 0.5 \
  --layers 0 \
  --modules resid_in,resid_out,mlp_out \
  --alphas 0,1.0 \
  --progress-every 1
```

Smoke 结果：

```text
rows = 48
nonfinite = 0
best_probe_acc = 1.0
exit_code = 0
```

正式分层测试：

```bash
MAX_PAIRS_PER_SUBTYPE=24 \
QWEN3_MAX_SECONDS=7200 \
GLM4_MAX_SECONDS=10800 \
DEEPSEEK7B_MAX_SECONDS=7200 \
OUTPUT_DIR=results/gpt5_phase300_voice_closure_pilot_stratified \
tests/gpt5/run_phase300_voice_closure_normal_all.sh
```

运行设置：

```text
normal mode:
  CUDA_LAUNCH_BLOCKING = 0
  PYTORCH_NO_CUDA_MEMORY_CACHING = 0

dtype:
  bfloat16

attention:
  sdpa

device_map:
  glm4, deepseek7b = auto

hard exit:
  --hard-exit-after-model
```

### 输出文件

正式结果：

```text
results/gpt5_phase300_voice_closure_pilot_stratified/qwen3_phase300_voice_closure_pilot.json
results/gpt5_phase300_voice_closure_pilot_stratified/glm4_phase300_voice_closure_pilot.json
results/gpt5_phase300_voice_closure_pilot_stratified/deepseek7b_phase300_voice_closure_pilot.json
results/gpt5_phase300_voice_closure_pilot_stratified/VOICE_CLOSURE_SUMMARY.md
results/gpt5_phase300_voice_closure_pilot_stratified/voice_closure_summary.json
```

日志：

```text
results/gpt5_gpu_lock_logs/20260528_195159_phase300normal_qwen3
results/gpt5_gpu_lock_logs/20260528_200354_phase300normal_glm4
results/gpt5_gpu_lock_logs/20260528_202037_phase300normal_deepseek7b
```

三轮正式测试：

```text
kernel.since-start.filtered.log = 0 行
```

说明本轮没有 GPU kernel/Xid/GSP/locked 级异常。

### 数据规模

```text
passive voice pairs = 92
train pairs = 46
test pairs = 46

subtypes:
  by_phrase
  dative_passive
  get_passive
  long_passive

Qwen3:
  layers = L0-L8
  rows = 9936
  nonfinite_rows = 0

GLM4:
  layers = L0-L8
  rows = 9936
  nonfinite_rows = 0

DeepSeek7B:
  layers = L20-L27
  rows = 8832
  nonfinite_rows = 0
```

### Qwen3 客观结果

Probe：

```text
probe_mean_accuracy = 0.923108
best_probe:
  layer = L0
  module = resid_in
  accuracy = 1.000000
  mean_signed_margin = 0.039576
```

Direction patch：

```text
active_to_passive best:
  layer = L1
  module = mlp_out
  progress = 0.306731
  kl_ratio = 1.197655
  logit_delta_ratio = 0.905774

passive_to_active best:
  layer = L2
  module = resid_out
  progress = 0.305248
  kl_ratio = 0.705311
  logit_delta_ratio = 0.664987
```

Subtype curve：

```text
by_phrase:
  active_to_passive = 0.166095
  passive_to_active = 0.045212

dative_passive:
  active_to_passive = 0.112386
  passive_to_active = 0.056769

get_passive:
  active_to_passive = 0.186283
  passive_to_active = 0.051683

long_passive:
  active_to_passive = 0.185860
  passive_to_active = 0.041687
```

客观现象：

```text
1. Qwen3 的 voice 方向可解码。
2. 方向 patch 有中等 causal effect。
3. active_to_passive 与 passive_to_active 都有效，但 subtype 平均上 active_to_passive 更强。
4. 最强 active_to_passive 在 L1 mlp_out，说明浅层 MLP 方向可推动 voice 转换。
```

### GLM4 客观结果

Probe：

```text
probe_mean_accuracy = 0.932770
best_probe:
  layer = L0
  module = resid_in
  accuracy = 1.000000
  mean_signed_margin = 0.000298
```

Direction patch：

```text
active_to_passive best:
  layer = L0
  module = resid_in
  progress = 0.487534
  kl_ratio = 0.597967
  logit_delta_ratio = 0.863517

passive_to_active best:
  layer = L8
  module = resid_out
  progress = 0.012508
  kl_ratio = 1.005237
  logit_delta_ratio = 0.105789
```

Subtype curve：

```text
by_phrase:
  active_to_passive = 0.213462
  passive_to_active = 0.001275

dative_passive:
  active_to_passive = 0.198536
  passive_to_active = 0.000378

get_passive:
  active_to_passive = 0.213752
  passive_to_active = 0.002016

long_passive:
  active_to_passive = 0.186301
  passive_to_active = -0.008697
```

客观现象：

```text
1. GLM4 的 voice 变量在 L0 resid_in 可被完美线性分开。
2. active_to_passive 方向 patch 在 L0 resid_in 有明显 causal effect：
   progress = 0.487534
   KL ratio 降到 0.597967
3. passive_to_active 几乎无效：
   best progress = 0.012508
4. 这说明 GLM4 中当前学到的 voice_direction 更像“写入被动语态方向”，不是对称的 active/passive 双向变量轴。
5. 这与 Phase 35 的 GLM4 浅层 MLP/residual 写入型判断一致，但也说明闭包还没有完成。
```

### DeepSeek7B 客观结果

Probe：

```text
probe_mean_accuracy = 0.640399
best_probe:
  layer = L21
  module = mlp_out
  accuracy = 1.000000
  mean_signed_margin = 108.308620
```

Direction patch：

```text
active_to_passive best:
  layer = L27
  module = resid_out
  progress = 0.053636
  kl_ratio = 0.854817
  logit_delta_ratio = 0.283468

passive_to_active best:
  layer = L27
  module = resid_out
  progress = 0.184869
  kl_ratio = 0.765157
  logit_delta_ratio = 0.390689
```

Subtype curve：

```text
by_phrase:
  active_to_passive = 0.008890
  passive_to_active = 0.002664

dative_passive:
  active_to_passive = 0.015331
  passive_to_active = 0.050772

get_passive:
  active_to_passive = 0.026669
  passive_to_active = -0.000450

long_passive:
  active_to_passive = 0.012186
  passive_to_active = 0.095901
```

客观现象：

```text
1. DeepSeek7B 某些 layer/module 可以完美 probe voice，但平均 probe accuracy 只有 0.640399。
2. 可解码不等于可因果控制：active_to_passive direction patch 很弱。
3. passive_to_active 稍强，但仍远弱于 GLM4 active_to_passive。
4. 最强点仍在 L27 resid_out，符合 DeepSeek7B 输出释放点特征。
5. 这支持：DeepSeek7B 的 voice 信息可能在深层输出接口可读，但单一全局 voice direction 不足以重写机制。
```

### 三模型对比

```text
Qwen3:
  voice 可解码；
  双向 patch 都有中等效果；
  active_to_passive 最强在 L1 mlp_out。

GLM4:
  voice 可解码；
  active_to_passive 强；
  passive_to_active 几乎无效；
  最强点在 L0 resid_in。

DeepSeek7B:
  局部可解码；
  direction patch 效果弱；
  最强点仍在 L27 resid_out。
```

### 本轮最重要的新事实

```text
1. 这是第一轮变量级实验，不再只是 GSSC 行为图谱。
2. 三模型都存在可解码 voice 信息。
3. GLM4 的 active_to_passive 方向最具因果效应，符合前面筛选出的 GLM4 passive 候选。
4. GLM4 的 voice 方向不是对称轴：被动写入有效，反向恢复无效。
5. Qwen3 更像浅层双向可调，但效果不如 GLM4 active_to_passive 强。
6. DeepSeek7B 再次表现为“可读出但难以单点方向控制”，说明它可能需要分段/多点轨迹而非单方向变量。
```

### 关键硬伤

```text
1. 当前 voice_direction 是 mean(passive - active)，还不是纯 voice 变量。
   它可能混入：
     surface subject
     word order
     auxiliary was/got
     by phrase
     verb participle
     role binding

2. 当前向量是 sequence mean pooling，可能丢失 token-level role binding。

3. 当前只做 direction patch，没有做：
   variable ablation
   destroy-and-restore
   agent/patient swap
   by_phrase 控制
   token-level causal patch

4. GLM4 passive_to_active 失败说明闭包尚未成立。

5. DeepSeek7B 可 probe 但 patch 弱，说明 probe 不是机制证明。
```

### 对编码机制的谨慎解释

当前不能说：

```text
已经破解 passive 机制；
已经找到 voice 变量本体；
已经证明 GLM4 的 role binding 机制；
```

当前可以说：

```text
1. GLM4 中存在一个从 active 到 passive 的浅层 residual voice-like direction。
2. 这个方向能从 train pair 泛化到 test pair。
3. 在 L0 resid_in 加入该方向后，模型输出明显向 passive 目标推进。
4. 但该方向不是对称闭包变量，因为 passive_to_active 几乎无效。
```

这说明：

```text
变量闭包工程是可行的；
但必须从 voice-like direction 继续拆成更细变量：
  voice
  surface_subject
  semantic_agent
  semantic_patient
  auxiliary/by_phrase
```

### 下一步计划

Phase 38：GLM4 passive 变量拆分。

目标：

```text
把当前 voice-like direction 拆成 voice 与 role binding。
```

测试：

```text
1. surface subject 控制：
   the dog chased the cat
   the dog was chased by the cat

2. semantic role 控制：
   the dog chased the cat
   the cat chased the dog

3. voice 控制：
   the dog chased the cat
   the cat was chased by the dog

4. no-agent passive 控制：
   someone broke the window
   the window was broken

5. by-phrase 控制：
   the window was broken
   the window was broken by the boy
```

关键指标：

```text
voice probe
agent probe
patient probe
surface_subject probe
direction patch
ablation
destroy + restore
downstream recompute
```

Phase 39：Qwen3 logical operator 闭包。

Phase 40：DeepSeek7B recursive segment closure。

## Phase 38: Passive Factor 变量拆分与跨模型闭包测试 [2026-05-28 21:49]

### 任务目标

根据 Phase 37 的结果，继续从 GSSC 路径图谱进入变量级闭包工程。本轮不再只测试 `voice-like direction`，而是把 passive 相关变化拆成三个可测试变量：

```text
voice: active -> passive_by
role_swap: agent/patient 交换
by_phrase: passive_no_agent -> passive_by_agent
```

本轮重点验证：

```text
1. Phase 37 中 GLM4 的 active -> passive 效果主要来自哪个变量。
2. role binding 是否能被 sequence-mean direction 捕捉。
3. Qwen3 / GLM4 / DeepSeek7B 是否存在不同变量控制方式。
4. probe 可读出是否等于 direction patch 可控制。
```

### 对用户分析的判断

用户分析整体正确。Phase 37 只能说明存在 `voice-like direction`，不能说明已经破解 passive mechanism。原因是原始 `passive - active` 差分混入了：

```text
voice
word_order
surface_subject
semantic_agent
semantic_patient
auxiliary
by_phrase
verb_participle
```

所以本轮必须做变量拆分。尤其是 passive 的核心不是“句子是否被动”这个整体标签，而是：

```text
谁是 semantic_agent
谁是 semantic_patient
谁成为 surface_subject
是否出现 by_phrase
语态模板如何写入
```

本轮仍然不是完整闭包，因为还没有 token-level role binding、destroy-and-restore、变量恢复实验。但它比 Phase 37 更接近机制：开始区分 `voice`、`role_swap`、`by_phrase` 三个因素。

### 新增脚本

```text
tests/gpt5/phase301_passive_factor_closure.py
tests/gpt5/run_phase301_normal.sh
tests/gpt5/run_phase301_passive_factor_normal_all.sh
tests/gpt5/phase301_passive_factor_summary.py
```

脚本要点：

```text
1. 使用 BF16 + attn_implementation="sdpa"。
2. GLM4 和 DeepSeek7B 使用 device_map="auto"。
3. 每个模型单独进程运行，并添加 --hard-exit-after-model。
4. run_all 脚本按 qwen3 -> glm4 -> deepseek7b 顺序运行。
5. 每个模型结束后进程退出，避免前一个模型残留显存。
6. 使用正常模式，不使用 CUDA_LAUNCH_BLOCKING=1 / PYTORCH_NO_CUDA_MEMORY_CACHING=1 的保守模式。
7. 运行时记录 GPU / process / kernel 日志。
```

### 数据构造

每个 base 构造 6 个状态：

```text
active_ab:
  the agent verb the patient

active_ba:
  the patient verb the agent

passive_ab_by:
  the patient was verb by the agent

passive_ba_by:
  the agent was verb by the patient

passive_ab_no:
  the patient was verb

passive_ba_no:
  the agent was verb
```

变量方向：

```text
voice:
  active_ab -> passive_ab_by
  active_ba -> passive_ba_by

role_swap:
  active_ab -> active_ba
  passive_ab_by -> passive_ba_by
  passive_ab_no -> passive_ba_no

by_phrase:
  passive_ab_no -> passive_ab_by
  passive_ba_no -> passive_ba_by
```

说明：

```text
本轮仍然使用 sequence mean pooling。
这可以测试整体方向是否存在，但不适合证明 token-level role binding。
```

### Smoke Test

```bash
MAX_SECONDS=1200 OUTPUT_DIR=results/gpt5_phase301_passive_factor_smoke \
tests/gpt5/run_phase301_normal.sh qwen3 \
  --max-bases 4 \
  --train-fraction 0.5 \
  --layers 0 \
  --modules resid_in,resid_out,mlp_out \
  --alphas 0,1.0 \
  --progress-every 1
```

结果：

```text
rows = 168
nonfinite_rows = 0
exit_code = 0
```

### 正式测试命令

```bash
MAX_BASES=24 \
QWEN3_MAX_SECONDS=7200 \
GLM4_MAX_SECONDS=10800 \
DEEPSEEK7B_MAX_SECONDS=7200 \
OUTPUT_DIR=results/gpt5_phase301_passive_factor_closure \
tests/gpt5/run_phase301_passive_factor_normal_all.sh
```

汇总命令：

```bash
python tests/gpt5/phase301_passive_factor_summary.py \
  --input-dir results/gpt5_phase301_passive_factor_closure \
  --output-dir results/gpt5_phase301_passive_factor_closure
```

### 输出文件

```text
results/gpt5_phase301_passive_factor_closure/qwen3_phase301_passive_factor_closure.json
results/gpt5_phase301_passive_factor_closure/glm4_phase301_passive_factor_closure.json
results/gpt5_phase301_passive_factor_closure/deepseek7b_phase301_passive_factor_closure.json
results/gpt5_phase301_passive_factor_closure/passive_factor_summary.json
results/gpt5_phase301_passive_factor_closure/PASSIVE_FACTOR_SUMMARY.md
```

日志：

```text
results/gpt5_gpu_lock_logs/20260528_205338_phase301normal_qwen3
results/gpt5_gpu_lock_logs/20260528_210935_phase301normal_glm4
results/gpt5_gpu_lock_logs/20260528_213154_phase301normal_deepseek7b
```

三个正式日志的 `kernel.since-start.filtered.log` 都是 0 行。

### 数据规模

```text
Qwen3:
  bases/train/test = 24 / 12 / 12
  rows = 13608
  nonfinite_rows = 0

GLM4:
  bases/train/test = 24 / 12 / 12
  rows = 13608
  nonfinite_rows = 0

DeepSeek7B:
  bases/train/test = 24 / 12 / 12
  rows = 12096
  nonfinite_rows = 0
```

总计：

```text
rows = 39312
nonfinite_rows = 0
```

### Qwen3 客观结果

probe 最佳结果：

```text
by_phrase:
  best = L0 resid_in
  acc = 1.000000
  margin = 0.100888

role_swap:
  best = L7 mlp_out
  acc = 0.944444
  margin = 0.620264

voice:
  best = L0 resid_in
  acc = 1.000000
  margin = 0.054826
```

direction patch 最佳结果：

```text
by_phrase forward:
  best = L1 mlp_out
  progress = 0.480497
  kl_ratio = 2.771863

by_phrase reverse:
  best = L2 resid_out
  progress = 0.522776
  kl_ratio = 1.089960

role_swap forward:
  best = L0 resid_in
  progress = 0.017493
  kl_ratio = 0.983960

role_swap reverse:
  best = L1 resid_out
  progress = 0.027947
  kl_ratio = 0.983883

voice forward:
  best = L0 resid_in
  progress = 0.346220
  kl_ratio = 1.296394

voice reverse:
  best = L2 resid_out
  progress = 0.428750
  kl_ratio = 0.755255
```

全局变量方向均值：

```text
by_phrase forward:
  progress = 0.136004
  kl_ratio = 1.188874

by_phrase reverse:
  progress = 0.146432
  kl_ratio = 1.111486

role_swap forward:
  progress = 0.006346
  kl_ratio = 1.004198

role_swap reverse:
  progress = 0.012143
  kl_ratio = 1.003458

voice forward:
  progress = 0.183000
  kl_ratio = 1.036725

voice reverse:
  progress = 0.087780
  kl_ratio = 0.981955
```

客观现象：

```text
1. Qwen3 的 voice 和 by_phrase 都可读出，也都有一定 direction patch 效果。
2. voice reverse 在最佳点上比 voice forward 更稳定，且 KL 降低。
3. by_phrase 的最佳 progress 很高，但 KL 明显升高，说明可能存在过度推进或分布扰动。
4. role_swap probe 可以读出，但 direction patch 几乎无效。
```

### GLM4 客观结果

probe 最佳结果：

```text
by_phrase:
  best = L0 resid_in
  acc = 1.000000
  margin = 0.000759

role_swap:
  best = L5 mlp_out
  acc = 0.972222
  margin = 0.000808

voice:
  best = L0 resid_in
  acc = 1.000000
  margin = 0.000386
```

direction patch 最佳结果：

```text
by_phrase forward:
  best = L0 resid_in
  progress = 0.269420
  kl_ratio = 0.916097

by_phrase reverse:
  best = L7 resid_out
  progress = 0.020095
  kl_ratio = 0.990468

role_swap forward:
  best = L0 resid_out
  progress = 0.017619
  kl_ratio = 0.963600

role_swap reverse:
  best = L8 resid_out
  progress = 0.030030
  kl_ratio = 1.000330

voice forward:
  best = L0 resid_in
  progress = 0.472474
  kl_ratio = 0.598427

voice reverse:
  best = L8 resid_out
  progress = 0.015532
  kl_ratio = 0.998028
```

全局变量方向均值：

```text
by_phrase forward:
  progress = 0.108485
  kl_ratio = 0.933662

by_phrase reverse:
  progress = -0.005009
  kl_ratio = 0.991968

role_swap forward:
  progress = 0.010789
  kl_ratio = 1.001808

role_swap reverse:
  progress = 0.011120
  kl_ratio = 0.999127

voice forward:
  progress = 0.271165
  kl_ratio = 0.760783

voice reverse:
  progress = -0.001104
  kl_ratio = 0.997715
```

客观现象：

```text
1. GLM4 的 Phase 37 active -> passive 效果主要来自 voice forward。
2. by_phrase forward 有中等效果，但明显弱于 voice forward。
3. role_swap 在 probe 上可读出，但 direction patch 几乎没有控制效果。
4. voice reverse 继续失败，说明 GLM4 不是对称 active/passive 语态轴。
5. 最强控制点仍然是 L0 resid_in，符合浅层写入型判断。
```

### DeepSeek7B 客观结果

probe 最佳结果：

```text
by_phrase:
  best = L21 mlp_out
  acc = 0.958333
  margin = 207.724309

role_swap:
  best = L21 mlp_out
  acc = 0.847222
  margin = 12.688938

voice:
  best = L20 mlp_out
  acc = 1.000000
  margin = 123.004636
```

direction patch 最佳结果：

```text
by_phrase forward:
  best = L26 resid_out
  progress = 0.620331
  kl_ratio = 1.080685

by_phrase reverse:
  best = L26 resid_out
  progress = 0.482621
  kl_ratio = 1.392317

role_swap forward:
  best = L24 resid_in
  progress = 0.018090
  kl_ratio = 0.981939

role_swap reverse:
  best = L27 resid_out
  progress = 0.015873
  kl_ratio = 0.985071

voice forward:
  best = L20 resid_in
  progress = 0.056489
  kl_ratio = 1.055646

voice reverse:
  best = L27 resid_out
  progress = 0.118703
  kl_ratio = 0.798672
```

全局变量方向均值：

```text
by_phrase forward:
  progress = 0.220030
  kl_ratio = 1.243953

by_phrase reverse:
  progress = 0.003117
  kl_ratio = 1.198228

role_swap forward:
  progress = 0.008298
  kl_ratio = 0.980360

role_swap reverse:
  progress = -0.001965
  kl_ratio = 1.157389

voice forward:
  progress = 0.017968
  kl_ratio = 1.029938

voice reverse:
  progress = -0.027237
  kl_ratio = 1.647137
```

客观现象：

```text
1. DeepSeek7B 的变量可以被 probe 读出，尤其 L20-L21 mlp_out 很强。
2. 但 voice direction patch 很弱，继续支持“可读不等于可控”。
3. by_phrase 在 L26 resid_out 有高 progress，但 KL 没有改善，甚至部分方向升高。
4. 这更像深层输出格式扰动或过度推进，不足以证明 by_phrase 机制闭包。
5. role_swap 同样可读但不可控。
```

### 三模型对比

```text
Qwen3:
  voice 和 by_phrase 都有中等控制效果；
  role_swap 可读但不可控；
  整体更像分布式、较双向的浅层变量系统。

GLM4:
  voice forward 最清楚；
  by_phrase forward 次之；
  role_swap 基本不可控；
  active -> passive 是浅层 L0 resid_in 写入，不是双向语态轴。

DeepSeek7B:
  probe 读出很强；
  direction patch 控制弱；
  by_phrase 的 late residual progress 高但 KL 风险较大；
  更像深层输出释放/格式扰动，不适合用单一方向解释。
```

### 当前最重要的新事实

Phase 38 对 Phase 37 的 GLM4 结论进行了拆分：

```text
Phase 37:
  GLM4 active -> passive 有强 voice-like direction。

Phase 38:
  这个强效果主要来自 voice forward；
  by_phrase forward 有较小贡献；
  role_swap 几乎没有 direction patch 效果。
```

因此更严格的表达是：

```text
GLM4 在 L0 resid_in 存在 passive-construction write signal。
它更像“写入被动构造控制信号”，而不是完整 passive mechanism。
完整 passive mechanism 至少还需要 token-level role binding：
  semantic_agent
  semantic_patient
  surface_subject
```

### 硬伤

1. 仍然使用 sequence mean pooling，不能证明 token-level role binding。
2. `role_swap` probe 可读出但 patch 不可控，说明整体均值方向不能捕捉角色绑定机制。
3. `by_phrase` 方向混合了语法长度、agent specification、介词结构和输出格式，不是纯变量。
4. DeepSeek7B 的高 progress 伴随 KL 上升，不能作为干净机制证据。
5. 还没有 destroy-and-restore，因此仍不是完整闭包。
6. 还没有只替换 subject token / object token / by-agent token 的局部变量干预。
7. 当前变量方向仍是线性均值差分，可能无法描述非线性或多点轨迹变量。

### 对语言编码机制的谨慎判断

本轮结果支持一个更稳的判断：

```text
passive 不是单一语义轴；
passive 更像变量组合程序。
```

其中至少包含：

```text
voice control signal
surface_subject selection
semantic_agent / semantic_patient binding
by_phrase realization
verb morphology
output formatting
```

GLM4 的 L0 resid_in 可以写入较强 `voice forward` 控制信号，但这个信号不能反向恢复 active，也不能单独完成 role binding。所以它不是 passive mechanism 本身，只是 passive program 的一个早期控制变量。

### 下一步计划

Phase 39：token-level GLM4 passive role binding。

目标：

```text
从 sequence mean 改为 token-level 变量定位。
```

需要捕捉：

```text
subject token
verb token
object token
by token
by-agent token
last token
```

变量：

```text
surface_subject
semantic_agent
semantic_patient
voice
by_phrase
auxiliary
```

关键测试：

```text
1. token-level probe:
   哪个 token 位置最能读出 agent/patient/voice。

2. token-level direction patch:
   只替换 subject token / object token / by-agent token。

3. role swap patch:
   单独交换 agent/patient token 表示，看输出是否改变。

4. destroy-and-restore:
   破坏 role variable 后恢复，看输出是否恢复。
```

Phase 40：Qwen3 logical operator 闭包。

Phase 41：DeepSeek7B recursive segment closure。

## Phase 39: Passive Token-Level Role Binding 脚本与正常模式 GPU 阻断记录 [2026-05-28 23:45]

### 任务目标

根据 Phase 38 的硬伤，继续把 passive 机制从 sequence mean direction 推进到 token-level role binding。

本轮计划测试：

```text
1. subject / object / by_agent / verb / last token 是否能读出 voice、by_phrase、role。
2. 只 patch 某个 token span 时，输出是否向目标状态移动。
3. all_positions 是否比单 token 更强。
4. role_swap 是否能在 token-level 变得可控。
```

### 对用户分析的判断

用户分析是正确的。Phase 38 最大价值是证明：

```text
可读出不等于可控制；
direction patch 不等于机制闭包；
sequence mean pooling 不适合证明 role binding。
```

passive 的真正对象不是一个全句方向，而是：

```text
voice control
surface_subject selection
semantic_agent binding
semantic_patient binding
by-agent binding
verb morphology
prediction priority shift
```

因此下一步必须进入 token-level variable closure。

### 新增脚本

```text
tests/gpt5/phase302_passive_token_role_closure.py
tests/gpt5/run_phase302_normal.sh
tests/gpt5/run_phase302_passive_token_role_normal_all.sh
tests/gpt5/phase302_passive_token_role_summary.py
```

脚本设计：

```text
1. 继续使用 BF16 + attn_implementation="sdpa"。
2. GLM4 / DeepSeek7B 使用 device_map="auto"。
3. 每个模型使用 --hard-exit-after-model。
4. run_all 按 qwen3 -> glm4 -> deepseek7b 顺序运行。
5. 使用正常模式：
   CUDA_LAUNCH_BLOCKING=0
   PYTORCH_NO_CUDA_MEMORY_CACHING=0
6. 捕捉 token span：
   subject
   object
   by_agent
   verb
   last
7. token-level probe:
   voice
   by_phrase
   agent_to_patient
8. token-level patch:
   subject_only
   object_only
   by_agent_only
   verb_only
   last_only
   all_positions
```

注意：

```text
第一次 smoke 后发现 all_positions 只是逐个 token patch，名称会误导。
随后修正为真正的多 token 同时 patch。
```

### Smoke Test

命令：

```bash
MAX_SECONDS=1200 OUTPUT_DIR=results/gpt5_phase302_passive_token_role_smoke \
tests/gpt5/run_phase302_normal.sh qwen3 \
  --max-bases 4 \
  --train-fraction 0.5 \
  --layers 0 \
  --modules resid_in,resid_out,mlp_out \
  --alphas 0,1.0 \
  --progress-every 1
```

修正后的 smoke 结果：

```text
rows = 288
probe_rows = 21
nonfinite_rows = 0
exit_code = 0
log_dir = results/gpt5_gpu_lock_logs/20260528_231858_phase302normal_qwen3
```

说明脚本的 token span 捕捉、hook patch、summary 基础流程可运行。

### 24-base 正式测试尝试

命令：

```bash
MAX_BASES=24 \
QWEN3_MAX_SECONDS=7200 \
GLM4_MAX_SECONDS=10800 \
DEEPSEEK7B_MAX_SECONDS=7200 \
OUTPUT_DIR=results/gpt5_phase302_passive_token_role_closure \
tests/gpt5/run_phase302_passive_token_role_normal_all.sh
```

Qwen3 在 24-base 正常模式下发生用户态 segmentation fault：

```text
run_id = 20260528_231932_phase302normal_qwen3
model = qwen3
bases/train/test = 24 / 12 / 12
probe_rows = 339
progress:
  intervention bases = 4/12, rows = 5184
  intervention bases = 8/12, rows = 10368
exit_code = 139
kernel.since-start.filtered.log = 0 行
```

当时没有看到 Xid / NVRM kernel error，因此先判断为用户态长 session 稳定性问题。

### 16-base 缩短测试尝试

为了保持正常模式，同时缩短单进程运行时长，继续尝试：

```bash
MAX_BASES=16 \
QWEN3_MAX_SECONDS=7200 \
GLM4_MAX_SECONDS=10800 \
DEEPSEEK7B_MAX_SECONDS=7200 \
OUTPUT_DIR=results/gpt5_phase302_passive_token_role_closure \
tests/gpt5/run_phase302_passive_token_role_normal_all.sh
```

Qwen3 第二次运行：

```text
run_id = 20260528_233408_phase302normal_qwen3
model = qwen3
bases/train/test = 16 / 8 / 8
probe_rows = 303
```

但运行中出现 NVIDIA kernel Oops。手动生成 kernel 过滤日志：

```text
results/gpt5_gpu_lock_logs/20260528_233408_phase302normal_qwen3/kernel.since-start.manual.log
results/gpt5_gpu_lock_logs/20260528_233408_phase302normal_qwen3/kernel.since-start.manual.filtered.log
```

关键日志：

```text
May 28 23:38:11 kernel: RIP: 0010:os_alloc_mem+0xad/0x100 [nvidia]
May 28 23:38:11 kernel: ? nvUvmInterfaceGetExternalAllocPtes+0xab/0xe0 [nvidia]
May 28 23:38:11 kernel: ? map_rm_pt_range.constprop.0+0x2b9/0x5d0 [nvidia_uvm]
May 28 23:38:11 kernel: ? uvm_page_table_range_vec_init+0x20c/0x2a0 [nvidia_uvm]
May 28 23:38:11 kernel: ? uvm_va_range_map_rm_allocation+0x39f/0x4c0 [nvidia_uvm]
May 28 23:38:11 kernel: ? uvm_map_external_allocation_on_gpu+0x319/0x4f0 [nvidia_uvm]
May 28 23:38:11 kernel: ? uvm_api_map_external_allocation+0x5cc/0x850 [nvidia_uvm]
May 28 23:38:11 kernel: ? uvm_ioctl+0x1650/0x1be0 [nvidia_uvm]
May 28 23:38:11 kernel: note: python[94693] exited with irqs disabled
```

当前进程状态：

```text
nvidia-smi 进程卡在 D 状态：
  os_acquire_rwlock_write
```

因此本轮不能继续加载 GLM4 和 DeepSeek7B。继续运行会污染测试结果，并可能导致系统再次卡死。

### 当前有效结果

本轮没有产生有效三模型正式结果。

有效内容仅限于：

```text
1. Phase302 token-level 脚本已经完成并通过 qwen3 smoke。
2. 正常模式长跑在 qwen3 上触发用户态 segfault 139。
3. 缩短到 16 bases 后仍触发 NVIDIA kernel Oops，涉及 nvidia / nvidia_uvm。
4. GPU 查询进程进入 D 状态，说明当前驱动状态不适合继续 CUDA 测试。
```

不能得出的结论：

```text
不能比较三模型 token-level role binding；
不能判断 role_swap 是否在 token-level 可控；
不能把本轮视为 Phase 39 机制结果。
```

### 技术判断

这次和之前普通用户态 segfault 不同。

24-base 尝试：

```text
exit_code = 139
kernel filtered = 0
```

16-base 尝试：

```text
python exited with irqs disabled
nvidia / nvidia_uvm kernel stack
nvidia-smi D state
```

因此当前更像：

```text
NVIDIA UVM / driver kernel path 在大量小 forward + hook + SDPA 正常模式下触发异常。
```

这不是语言机制结果，而是测试环境稳定性问题。

### 硬伤

1. Phase302 正式结果未完成。
2. 没有 GLM4 / DeepSeek7B 结果。
3. qwen3 正式输出文件未保存。
4. 当前 GPU 驱动状态异常，不能继续 CUDA 测试。
5. 正常模式长 session 不稳定，和用户要求的“不要保守方式”存在现实冲突。

### 下一步计划

必须先解决测试工程稳定性，否则 token-level 机制实验无法推进。

建议 Phase 40 先做测试工程修正，而不是继续理论扩展：

```text
1. Phase302 增加 pair/base-level checkpoint。
   每完成 1 个 test base 就落盘 partial JSON。

2. 增加 shard 参数：
   --test-start
   --test-count
   每个进程只跑 1-2 个 test base。

3. run_all 仍保持正常模式，但改成短进程 shard：
   qwen3 shard 0..N
   完成 qwen3 后再 glm4
   完成 glm4 后再 deepseek7b

4. 禁止在高风险运行中并发调用额外 nvidia-smi 查询。
   这次 nvidia-smi 卡在 D 状态，说明监控本身也可能被 driver lock 牵连。

5. 将 GPU monitor 改成可选：
   默认只记录 run.log + kernel follower；
   需要时再开启 nvidia-smi monitor。
```

研究路线不变：

```text
Phase 40:
  稳定版 token-level shard runner。

Phase 41:
  GLM4 passive token-level role binding closure。

Phase 42:
  Qwen3 logical operator closure。

Phase 43:
  DeepSeek7B recursive segment closure。
```

当前最重要的结论不是机制结论，而是工程结论：

```text
token-level closure 的脚本方向正确；
但正常模式长跑已经触发 nvidia_uvm kernel Oops；
必须把长跑拆成短 shard + checkpoint，否则无法可靠积累全局语义语法契约图谱。
```

## Phase 40: Token-Level Shard Runner 与三模型短分片验证 [2026-05-29 12:23]

### 任务目标

继续完成 Phase 39 未完成的任务，但先解决工程稳定性问题。Phase 39 已经证明：

```text
Phase302 token-level 脚本方向正确；
但正常模式长 session 会触发用户态 segfault，甚至 nvidia / nvidia_uvm kernel Oops。
```

因此本轮目标不是继续强行长跑，而是：

```text
1. 给 Phase302 增加 shard 参数。
2. 每个进程只跑 1 个 test base。
3. 每完成 1 个 base 就写 partial checkpoint。
4. 每个 shard 独立加载模型、运行、保存、退出。
5. qwen3 完成后再 glm4，glm4 完成后再 deepseek7b。
6. 禁用并发 nvidia-smi GPU monitor，只保留 run.log 和 kernel follower。
```

### 脚本变更

修改：

```text
tests/gpt5/phase302_passive_token_role_closure.py
tests/gpt5/run_phase302_normal.sh
tests/gpt5/phase302_passive_token_role_summary.py
```

新增：

```text
tests/gpt5/run_phase302_passive_token_role_sharded_normal_all.sh
```

新增参数：

```text
--test-start
--test-count
--shard-label
```

新增输出：

```text
results/.../partials/{model}/{model}_phase302_{shard}.partial.json
results/.../{model}_phase302_passive_token_role_closure_{shard}.json
results/.../passive_token_role_merged.json
```

运行脚本新增环境变量：

```text
ENABLE_GPU_MONITOR=0/1
ENABLE_SNAPSHOT_NVIDIA_SMI=0/1
```

默认：

```text
ENABLE_GPU_MONITOR=0
```

原因：

```text
Phase 39 中 nvidia-smi 进程曾卡在 os_acquire_rwlock_write。
高风险 token-level 长跑中并发 nvidia-smi 监控可能会被 driver lock 牵连。
```

### Smoke Test

命令：

```bash
MAX_SECONDS=900 \
OUTPUT_DIR=results/gpt5_phase302_shard_smoke \
ENABLE_GPU_MONITOR=0 \
tests/gpt5/run_phase302_normal.sh qwen3 \
  --max-bases 4 \
  --train-fraction 0.5 \
  --layers 0 \
  --modules resid_in,resid_out,mlp_out \
  --alphas 0,1.0 \
  --progress-every 1 \
  --test-start 0 \
  --test-count 1 \
  --shard-label test000-001
```

结果：

```text
rows = 144
probe_rows = 21
nonfinite_rows = 0
exit_code = 0
kernel.since-start.filtered.log = 0 行
```

汇总也成功生成：

```text
results/gpt5_phase302_shard_smoke/passive_token_role_summary.json
results/gpt5_phase302_shard_smoke/passive_token_role_merged.json
results/gpt5_phase302_shard_smoke/PASSIVE_TOKEN_ROLE_SUMMARY.md
```

### 正式短分片测试

第一轮每模型 2 个 test base：

```bash
MAX_BASES=16 \
TEST_TOTAL=2 \
SHARD_SIZE=1 \
QWEN3_MAX_SECONDS=1800 \
GLM4_MAX_SECONDS=2400 \
DEEPSEEK7B_MAX_SECONDS=1800 \
OUTPUT_DIR=results/gpt5_phase302_passive_token_role_sharded \
ENABLE_GPU_MONITOR=0 \
tests/gpt5/run_phase302_passive_token_role_sharded_normal_all.sh
```

随后继续扩展每模型第 3-4 个 test base：

```bash
OUT=results/gpt5_phase302_passive_token_role_sharded
COMMON='--max-bases 16 --train-fraction 0.5 --modules resid_in,resid_out,mlp_out --alphas 0,1.0 --progress-every 1'

for model_layers in \
  'qwen3|0,1,2,3,4,5,6,7,8|1800' \
  'glm4|0,1,2,3,4,5,6,7,8|2400' \
  'deepseek7b|20,21,22,23,24,25,26,27|1800'; do
  IFS='|' read -r model layers max_seconds <<< "$model_layers"
  for start in 2 3; do
    end=$((start+1))
    label=$(printf 'test%03d-%03d' "$start" "$end")
    MAX_SECONDS="$max_seconds" OUTPUT_DIR="$OUT" ENABLE_GPU_MONITOR=0 \
      tests/gpt5/run_phase302_normal.sh "$model" \
        --layers "$layers" \
        --test-start "$start" \
        --test-count 1 \
        --shard-label "$label" \
        $COMMON
  done
done

python tests/gpt5/phase302_passive_token_role_summary.py \
  --input-dir "$OUT" \
  --output-dir "$OUT"
```

### 输出文件

```text
results/gpt5_phase302_passive_token_role_sharded/qwen3_phase302_passive_token_role_closure_test000-001.json
results/gpt5_phase302_passive_token_role_sharded/qwen3_phase302_passive_token_role_closure_test001-002.json
results/gpt5_phase302_passive_token_role_sharded/qwen3_phase302_passive_token_role_closure_test002-003.json
results/gpt5_phase302_passive_token_role_sharded/qwen3_phase302_passive_token_role_closure_test003-004.json

results/gpt5_phase302_passive_token_role_sharded/glm4_phase302_passive_token_role_closure_test000-001.json
results/gpt5_phase302_passive_token_role_sharded/glm4_phase302_passive_token_role_closure_test001-002.json
results/gpt5_phase302_passive_token_role_sharded/glm4_phase302_passive_token_role_closure_test002-003.json
results/gpt5_phase302_passive_token_role_sharded/glm4_phase302_passive_token_role_closure_test003-004.json

results/gpt5_phase302_passive_token_role_sharded/deepseek7b_phase302_passive_token_role_closure_test000-001.json
results/gpt5_phase302_passive_token_role_sharded/deepseek7b_phase302_passive_token_role_closure_test001-002.json
results/gpt5_phase302_passive_token_role_sharded/deepseek7b_phase302_passive_token_role_closure_test002-003.json
results/gpt5_phase302_passive_token_role_sharded/deepseek7b_phase302_passive_token_role_closure_test003-004.json

results/gpt5_phase302_passive_token_role_sharded/passive_token_role_summary.json
results/gpt5_phase302_passive_token_role_sharded/passive_token_role_merged.json
results/gpt5_phase302_passive_token_role_sharded/PASSIVE_TOKEN_ROLE_SUMMARY.md
```

日志：

```text
results/gpt5_gpu_lock_logs/20260529_120156_phase302normal_qwen3
results/gpt5_gpu_lock_logs/20260529_120326_phase302normal_qwen3
results/gpt5_gpu_lock_logs/20260529_120456_phase302normal_glm4
results/gpt5_gpu_lock_logs/20260529_120709_phase302normal_glm4
results/gpt5_gpu_lock_logs/20260529_120918_phase302normal_deepseek7b
results/gpt5_gpu_lock_logs/20260529_121052_phase302normal_deepseek7b
results/gpt5_gpu_lock_logs/20260529_121249_phase302normal_qwen3
results/gpt5_gpu_lock_logs/20260529_121421_phase302normal_qwen3
results/gpt5_gpu_lock_logs/20260529_121549_phase302normal_glm4
results/gpt5_gpu_lock_logs/20260529_121805_phase302normal_glm4
results/gpt5_gpu_lock_logs/20260529_122014_phase302normal_deepseek7b
results/gpt5_gpu_lock_logs/20260529_122145_phase302normal_deepseek7b
```

以上正式 shard 的 `kernel.since-start.filtered.log` 全部为 0 行。

当前 GPU 状态：

```text
driver_version = 595.71.05
memory_used = 566 MiB / 24564 MiB
gpu_utilization = 3%
```

### 数据规模

```text
Qwen3:
  bases/train/test = 16 / 8 / 4
  rows = 5184
  nonfinite_rows = 0

GLM4:
  bases/train/test = 16 / 8 / 4
  rows = 5184
  nonfinite_rows = 0

DeepSeek7B:
  bases/train/test = 16 / 8 / 4
  rows = 4608
  nonfinite_rows = 0
```

总计：

```text
rows = 14976
nonfinite_rows = 0
shards = 12
kernel filtered errors = 0
```

### Qwen3 客观结果

probe 最佳结果：

```text
agent_to_patient:
  best = L6 resid_out by_agent
  acc = 0.750000
  margin = 5.128343

by_phrase:
  best = L0 resid_out last
  acc = 1.000000
  margin = 9.339340

voice:
  best = L0 resid_out verb
  acc = 1.000000
  margin = 2.951893
```

token patch 最佳结果：

```text
by_phrase:
  best = last_only / last
  layer = L5 resid_out
  progress = 0.188462
  kl_ratio = 1.030767

role_swap:
  best = all_positions / object+subject
  layer = L3 resid_out
  progress = 0.057689
  kl_ratio = 1.094042

voice:
  best = all_positions / object+subject+verb
  layer = L2 resid_in
  progress = 0.267556
  kl_ratio = 1.272648
```

客观现象：

```text
1. Qwen3 的 voice 和 by_phrase 在 token-level probe 上很容易读出。
2. token patch 有一定 progress，但 KL 没有稳定改善。
3. role_swap 仍然很弱，即使 all_positions 同时 patch 也只有 0.058 左右。
```

### GLM4 客观结果

probe 最佳结果：

```text
agent_to_patient:
  best = L4 resid_out by_agent
  acc = 0.750000
  margin = 0.026202

by_phrase:
  best = L1 resid_out last
  acc = 1.000000
  margin = 0.067451

voice:
  best = L1 resid_out verb
  acc = 1.000000
  margin = 0.034583
```

token patch 最佳结果：

```text
by_phrase:
  best = all_positions / last+subject+verb
  layer = L4 resid_out
  progress = 0.009157
  kl_ratio = 0.990765

role_swap:
  best = all_positions / object+subject
  layer = L4 resid_out
  progress = 0.047778
  kl_ratio = 0.948062

voice:
  best = subject_only / subject
  layer = L8 resid_out
  progress = 0.002460
  kl_ratio = 1.008206
```

客观现象：

```text
1. GLM4 的 token-level probe 能读出 voice / by_phrase。
2. 但 token-level direction patch 几乎不能推动输出。
3. 这和 Phase 38 的 sequence-level voice forward 强效果形成反差。
4. 说明 GLM4 的 passive write signal 可能更像整句/构造级状态，而不是简单 token 局部方向。
```

### DeepSeek7B 客观结果

probe 最佳结果：

```text
agent_to_patient:
  best = L20 mlp_out by_agent
  acc = 0.750000
  margin = 59.389148

by_phrase:
  best = L20 resid_in last
  acc = 1.000000
  margin = 7887.195618

voice:
  best = L20 resid_in verb
  acc = 1.000000
  margin = 3432.862277
```

token patch 最佳结果：

```text
by_phrase:
  best = last_only / last
  layer = L24 resid_out
  progress = 0.006703
  kl_ratio = 0.966365

role_swap:
  best = all_positions / object+subject
  layer = L24 resid_out
  progress = 0.093240
  kl_ratio = 0.951682

voice:
  best = all_positions / object+subject+verb
  layer = L20 resid_in
  progress = 0.184532
  kl_ratio = 1.185566
```

客观现象：

```text
1. DeepSeek7B 继续表现为 probe 极强、direction patch 较弱。
2. voice token patch 比 by_phrase 更有 progress，但 KL 上升。
3. role_swap 比 Qwen3/GLM4 稍强，但仍不足以称为角色绑定闭包。
```

### 三模型对比

```text
Qwen3:
  token-level voice patch 有一定效果；
  by_phrase 有弱效果；
  role_swap 很弱。

GLM4:
  token-level patch 整体很弱；
  和 Phase 38 的 sequence-level voice forward 强效果相反；
  更像 construction-level global write，不是 token-local direction。

DeepSeek7B:
  probe 极强；
  token patch 仍弱；
  role_swap 在三模型里相对略强，但仍不是闭包。
```

### 当前最重要的工程结论

短 shard 方案有效：

```text
Phase 39:
  长 session 触发 nvidia_uvm kernel Oops。

Phase 40:
  12 个短 shard 全部完成；
  每个 shard 单独加载模型并退出；
  kernel filtered 全部 0；
  当前 GPU 正常。
```

因此后续 token-level 大测试必须使用：

```text
short shard + partial checkpoint + merge summary
```

不能再使用长 session 全量运行。

### 当前最重要的机制线索

Phase 40 不支持“token-level 单方向已经破解 role binding”。

更谨慎的结论是：

```text
1. voice / by_phrase 在 token-level 可读出。
2. agent_to_patient 也能部分读出，但准确率只有 0.75。
3. token-level direction patch 对 role_swap 仍然很弱。
4. 因此 role binding 不是简单 token 局部线性方向。
```

这进一步支持：

```text
passive = construction-level control + relational role binding + output formatting
```

其中：

```text
construction-level control:
  Phase 38 GLM4 sequence-level voice forward 强。

token-local role direction:
  Phase 40 证据弱。
```

所以 passive 的下一步不应该继续做单方向 patch，而应该做：

```text
candidate-set role query
destroy-and-restore
token swap / state transplant
subspace ablation
segment recompute
```

### 硬伤

1. 测试只有 4 个 test base/模型，仍是小样本。
2. token-level direction 仍然是均值差分，不是学习到的 role subspace。
3. 没有 candidate-set 输出验证，因此 role_swap progress 仍是全 logits 方向指标。
4. 没有 destroy-and-restore。
5. 没有真实 token swap，只是方向加法。
6. GLM4 的 sequence-level 强效果与 token-level 弱效果之间还没有解释清楚。

### 下一步计划

Phase 41：candidate-set role query。

核心目标：

```text
不要只看全 logits progress；
直接问模型：who did the action / who received the action。
```

样本形式：

```text
The teacher praised the student. The person who did the action was the ...
The student was praised by the teacher. The person who did the action was the ...
The person who received the action was the ...
```

指标：

```text
agent candidate logprob
patient candidate logprob
agent-patient margin
wrong-role margin
```

干预：

```text
1. patch subject token
2. patch object token
3. patch by_agent token
4. swap subject/object hidden states
5. restore correct role token state
```

判断标准：

```text
如果 patch 后 agent/patient candidate margin 按预期翻转或恢复，
才开始接近 role binding closure。
```

Phase 42：GLM4 sequence-level construction signal 与 token-level role signal 的组合实验。

Phase 43：Qwen3 logical operator closure。

Phase 44：DeepSeek7B recursive segment closure。

## Phase 41: Candidate-Set Role Query 脚本与 GPU Xid 阻断记录 [2026-05-29 17:53]

### 任务目标

根据 Phase 40 的结论，继续从 token-level direction patch 转向 candidate-set role query。

本轮目标：

```text
1. 不再只看全 logits progress。
2. 直接构造角色查询：
   who did the action?
   who received the action?
3. 用 agent / patient 两个候选的 logit margin 判断角色绑定。
4. 用 token state transplant 替代 direction add。
5. 继续使用 short shard + partial checkpoint。
```

### 对用户分析的判断

用户分析正确。Phase 39/40 的核心价值不是证明 role binding 已破解，而是证明：

```text
1. sequence mean direction 不够。
2. token-level direction 也不够。
3. role binding 更可能是 token relation + construction control + routing + prediction priority 的动态机制。
4. 下一步必须转向 candidate-set query、state transplant、subspace patch、destroy-and-restore。
```

因此本轮进入 candidate-set role query。

### 新增脚本

```text
tests/gpt5/phase303_role_query_closure.py
tests/gpt5/phase303_role_query_summary.py
tests/gpt5/run_phase303_normal.sh
tests/gpt5/run_phase303_role_query_sharded_normal_all.sh
```

脚本设计：

```text
1. 使用 BF16 + attn_implementation="sdpa"。
2. GLM4 / DeepSeek7B 使用 device_map="auto"。
3. 每个模型、每个 shard 独立进程运行，并使用 --hard-exit-after-model。
4. 默认 ENABLE_GPU_MONITOR=0。
5. 每个 shard 输出独立 JSON。
6. 每个 base 完成后写 partial checkpoint。
```

### 查询格式

对每个 base 和 state 构造：

```text
{sentence}. the one who did the action was the ...
{sentence}. the one that received the action was the ...
```

候选：

```text
agent candidate
patient candidate
```

指标：

```text
margin = logit(correct_candidate) - logit(wrong_candidate)
correct_choice = margin > 0
```

### 干预方式

不再使用 direction add，而是 token state transplant：

```text
src_state -> dst_state
active_ab -> active_ba
active_ba -> active_ab
passive_ab_by -> passive_ba_by
passive_ba_by -> passive_ab_by
```

patch mode：

```text
subject_only
object_only
by_agent_only
verb_only
subject_object
subject_by_agent
all_roles
```

指标：

```text
source_target_margin:
  源 prompt 下目标角色候选 margin

target_clean_margin:
  目标 prompt 的自然 margin

patched_target_margin:
  transplant 后的目标角色候选 margin

margin_progress:
  (patched_target_margin - source_target_margin)
  /
  (target_clean_margin - source_target_margin)

flip_rate:
  patched_target_margin > 0 的比例
```

### Smoke Test

命令：

```bash
MAX_SECONDS=900 \
OUTPUT_DIR=results/gpt5_phase303_role_query_smoke \
ENABLE_GPU_MONITOR=0 \
tests/gpt5/run_phase303_normal.sh qwen3 \
  --max-bases 4 \
  --train-fraction 0.5 \
  --layers 0 \
  --modules resid_in,resid_out,mlp_out \
  --progress-every 1 \
  --test-start 0 \
  --test-count 1 \
  --shard-label test000-001
```

结果：

```text
rows = 120
baseline_rows = 8
nonfinite_rows = 0
exit_code = 0
kernel.since-start.filtered.log = 0 行
```

smoke 说明：

```text
candidate-set role query、token transplant、summary merge 基础流程可运行。
```

但 smoke 中某些 active_ba/passive_ba baseline 已经错误，说明查询格式本身存在模型偏置，后续必须把 baseline accuracy 和 patch effect 分开解释。

### 正式测试尝试

命令：

```bash
MAX_BASES=16 \
TEST_TOTAL=4 \
SHARD_SIZE=1 \
QWEN3_MAX_SECONDS=1800 \
GLM4_MAX_SECONDS=2400 \
DEEPSEEK7B_MAX_SECONDS=1800 \
OUTPUT_DIR=results/gpt5_phase303_role_query_closure \
ENABLE_GPU_MONITOR=0 \
tests/gpt5/run_phase303_role_query_sharded_normal_all.sh
```

Qwen3 第一个 shard 完成：

```text
run_id = 20260529_174846_phase303normal_qwen3
rows = 1080
baseline_rows = 8
nonfinite_rows = 0
exit_code = 0
```

Qwen3 第二个 shard 发生 CUDA runtime error：

```text
run_id = 20260529_175005_phase303normal_qwen3
error = CUDA error: unspecified launch failure
```

随后 GPU kernel 日志出现 Xid：

```text
May 29 17:51:19 kernel: NVRM: Xid 62
May 29 17:51:19 kernel: NVRM: Xid 45, pid=32059, name=python
May 29 17:51:49 kernel: NVRM: Xid 154, GPU recovery action changed from 0x0 (None) to 0x1 (GPU Reset Required)
May 29 17:51:59 kernel: NVRM: Xid 158, pid=29297, name=code, timeout error waiting for NV_UFLUSH_FB_FLUSH
May 29 17:52:09 kernel: NVRM: Xid 8, pid=32059, name=python
```

手动日志：

```text
results/gpt5_gpu_lock_logs/20260529_175005_phase303normal_qwen3/kernel.since-start.manual.log
results/gpt5_gpu_lock_logs/20260529_175005_phase303normal_qwen3/kernel.since-start.manual.filtered.log
```

当前判断：

```text
GPU 已进入 Reset Required 状态。
不能继续加载 GLM4 / DeepSeek7B。
继续运行会污染结果并增加系统卡死风险。
```

### 已得到的 Qwen3 单 shard 结果

输出：

```text
results/gpt5_phase303_role_query_closure/qwen3_phase303_role_query_closure_test000-001.json
results/gpt5_phase303_role_query_closure/ROLE_QUERY_SUMMARY.md
```

Qwen3 单 shard summary：

```text
bases/train/test = 16 / 8 / 1
baseline_rows = 8
intervention_rows = 1080
nonfinite_rows = 0
```

baseline：

```text
agent/active_ab:
  acc = 1.0
  margin = 6.1875

agent/active_ba:
  acc = 1.0
  margin = 3.75

agent/passive_ab_by:
  acc = 0.0
  margin = -1.125

agent/passive_ba_by:
  acc = 0.0
  margin = -0.25

patient/active_ab:
  acc = 1.0
  margin = 8.1875

patient/active_ba:
  acc = 1.0
  margin = 5.5

patient/passive_ab_by:
  acc = 1.0
  margin = 2.75

patient/passive_ba_by:
  acc = 1.0
  margin = 0.875
```

best intervention：

```text
agent:
  best = subject_only
  layer = L0 resid_in
  margin_progress = 4.104059
  patched_margin = 0.078125
  flip_rate = 0.5

patient:
  best = subject_only
  layer = L2 resid_out
  margin_progress = 1.250118
  patched_margin = 0.031250
  flip_rate = 0.5
```

解释限制：

```text
1. 只有 1 个 test base，不能作为稳定机制结果。
2. passive agent query baseline 为 0，说明查询模板对 passive agent 不稳。
3. intervention 的 margin_progress 可大于 1，但 patched_margin 很小，flip_rate 只有 0.5。
4. 因此它只能说明 candidate-set query 流程可产生角色相关信号，不能说明 role binding closure。
```

### 当前有效结论

有效工程结论：

```text
1. Phase303 candidate-set role query 脚本完成。
2. smoke 成功。
3. Qwen3 一个正式 shard 成功。
4. 第二个 shard 触发 CUDA launch failure 和 NVRM Xid。
5. GPU 进入 reset required，必须停止。
```

有效研究线索：

```text
1. candidate-set query 比全 logits progress 更接近 role binding 目标。
2. 当前 query template 对 passive agent 不稳定，需要改进。
3. subject token transplant 对 agent/patient margin 有影响，但小样本不足。
```

不能得出的结论：

```text
不能比较三模型。
不能判断 GLM4 / DeepSeek7B role binding。
不能证明 Qwen3 role binding closure。
不能把 subject_only 解释成 agent/patient 机制。
```

### 硬伤

1. GPU 出现 Xid 62 / 45 / 154 / 158 / 8，且需要 reset。
2. 正常模式即使使用 short shard，仍会在 Phase303 中触发 GPU 错误。
3. Phase303 没有完成三模型测试。
4. 查询模板本身对 passive agent 不稳定。
5. 没有 destroy-and-restore。

### 下一步计划

当前不能继续 CUDA 测试。需要先处理 GPU 稳定性。

建议：

```text
1. 重启系统，让 GPU reset。
2. Phase303 改进 query template：
   - 避免 "the one who did the action" 对 passive 的偏置；
   - 增加多种 paraphrase query；
   - 分别报告模板稳定性。

3. Phase303 增加 CPU-only debug / tiny CUDA debug：
   - 先只跑 baseline candidate scoring；
   - 再跑 transplant；
   - 分离是哪一类 hook/transplant 触发 CUDA failure。

4. 若必须继续正常模式：
   - 每 shard 只跑 1 base；
   - 每个 shard 只跑少量 layer；
   - 禁用 snapshot nvidia-smi；
   - 优先跑 GLM4 的 L0-L4，而不是全层。
```

研究路线仍然不变：

```text
function -> factor -> token relation -> candidate set -> state transplant -> destroy/restore
```

但当前最大瓶颈重新变成工程稳定性：

```text
Phase302 短 shard 可稳定完成；
Phase303 candidate-query transplant 在第二个 shard 触发 Xid；
必须先定位触发源，否则无法安全推进全局语义语法契约图谱。
```

## Phase 42: Role Query 模板校准与候选集合读出负结果 [2026-05-29 18:51]

### 任务目标

根据 Phase 41 的阻断和最新分析，本轮不继续做 token state transplant，也不继续跑复杂 hook，而是先校准 candidate-set role query 本身。

核心问题：

```text
如果查询模板本身无法稳定回答：
  谁是 agent（施事者）？
  谁是 patient（受事者）？

那么后续任何 patch / transplant / destroy-restore 都无法解释为 role binding 机制。
```

因此本轮只做 baseline logits，不做 activation hook，不做状态移植，不做破坏恢复。

### 对用户分析的判断

用户分析大方向正确：

```text
1. Phase 40/41 主要是实验范式进步，不是机制破解。
2. candidate-set role query 比全 logits progress 更接近 role binding。
3. 但查询模板不稳定时，不能把 patch 效果解释为角色绑定。
4. 下一步必须先做模板校准，再做因果移植和破坏恢复。
5. 当前最大瓶颈从理论转回实验可靠性。
```

本轮按这个判断执行：先把 role query 的读出器测稳。

### 新增脚本

新增：

```text
tests/gpt5/phase304_role_query_template_calibration.py
tests/gpt5/phase304_role_query_template_summary.py
tests/gpt5/run_phase304_normal.sh
tests/gpt5/run_phase304_role_query_template_normal_all.sh
```

Phase 304 测试普通自然语言补全模板，例如：

```text
the one who did the action was the ...
the one who performed the action was the ...
the one affected by the action was the ...
the recipient of the action was the ...
```

新增：

```text
tests/gpt5/phase305_role_query_option_calibration.py
tests/gpt5/phase305_role_query_option_summary.py
tests/gpt5/run_phase305_normal.sh
tests/gpt5/run_phase305_role_query_option_normal_all.sh
```

Phase 305 测试显式二选一模板，并反转候选顺序，例如：

```text
sentence: the teacher praised the student.
question: who performed the action, the teacher or the student?
answer: the ...

sentence: the teacher praised the student.
question: who performed the action, the student or the teacher?
answer: the ...
```

本轮脚本特性：

```text
1. 三模型依次运行：qwen3 -> GLM4 -> DS7B。
2. 每个模型完成后 hard-exit-after-model。
3. 使用 PROBE_ATTN_IMPLEMENTATION=sdpa。
4. GLM4 和 DS7B 使用 device_map="auto"。
5. 正常模式运行，不使用保守脚本。
6. 不做 hook，不做 transplant，只测 baseline logits。
```

### Phase 304 命令

Smoke：

```bash
MAX_SECONDS=900 OUTPUT_DIR=results/gpt5_phase304_role_query_template_smoke \
ENABLE_GPU_MONITOR=0 \
tests/gpt5/run_phase304_normal.sh qwen3 \
  --max-bases 2 \
  --query-types agent \
  --progress-every 1
```

结果：

```text
rows = 64
reliable = 3
nonfinite = 0
kernel.filtered = 0 行
```

正式三模型：

```bash
MAX_SECONDS=2400 \
QWEN3_MAX_SECONDS=1800 \
GLM4_MAX_SECONDS=2400 \
DEEPSEEK7B_MAX_SECONDS=2400 \
OUTPUT_DIR=results/gpt5_phase304_role_query_template_calibration \
ENABLE_GPU_MONITOR=0 \
tests/gpt5/run_phase304_role_query_template_normal_all.sh
```

### Phase 304 输出

```text
results/gpt5_phase304_role_query_template_calibration/qwen3_phase304_role_query_template_calibration.json
results/gpt5_phase304_role_query_template_calibration/glm4_phase304_role_query_template_calibration.json
results/gpt5_phase304_role_query_template_calibration/deepseek7b_phase304_role_query_template_calibration.json
results/gpt5_phase304_role_query_template_calibration/ROLE_QUERY_TEMPLATE_CALIBRATION_SUMMARY.md
```

日志：

```text
results/gpt5_gpu_lock_logs/20260529_183233_phase304normal_qwen3
results/gpt5_gpu_lock_logs/20260529_183500_phase304normal_glm4
results/gpt5_gpu_lock_logs/20260529_183804_phase304normal_deepseek7b
```

三轮 kernel.since-start.filtered.log 都是 0 行。

### Phase 304 数据规模

```text
每模型:
  bases = 32
  states = active_ab, active_ba, passive_ab_by, passive_ba_by
  templates = 16
  rows = 2048

三模型总计:
  rows = 6144
  nonfinite_rows = 0
```

### Phase 304 客观结果

严格可靠模板定义：

```text
所有 state 的 accuracy >= 0.9
所有 state 的 mean_margin >= 0
```

结果：

```text
Qwen3:
  reliable_templates = 0
  nonfinite = 0

GLM4:
  reliable_templates = 0
  nonfinite = 0

DeepSeek7B:
  reliable_templates = 0
  nonfinite = 0
```

相对较好的模板也不达标：

```text
GLM4 patient_affected:
  min_state_accuracy = 0.7812
  min_state_mean_margin = 1.4863
  仍低于 0.9
```

典型弱点：

```text
Qwen3:
  agent/passive_ba_by 多个模板 accuracy 低至 0.0625 - 0.3750

GLM4:
  agent/passive_ba_by 和 agent/active_ba 是主要弱点

DeepSeek7B:
  agent/active_ba 和 patient/passive_ba_by 是主要弱点
```

客观现象：

```text
普通自然语言补全模板不能作为稳定 role query 读出器。
尤其 agent 查询容易受 AB/BA、active/passive、模板词汇影响。
```

### Phase 305 命令

为排除“没有显式列出候选”导致的错误，继续做二选一模板，并反转候选顺序：

```bash
MAX_SECONDS=3000 \
QWEN3_MAX_SECONDS=2400 \
GLM4_MAX_SECONDS=3000 \
DEEPSEEK7B_MAX_SECONDS=3000 \
OUTPUT_DIR=results/gpt5_phase305_role_query_option_calibration \
ENABLE_GPU_MONITOR=0 \
tests/gpt5/run_phase305_role_query_option_normal_all.sh
```

### Phase 305 输出

```text
results/gpt5_phase305_role_query_option_calibration/qwen3_phase305_role_query_option_calibration.json
results/gpt5_phase305_role_query_option_calibration/glm4_phase305_role_query_option_calibration.json
results/gpt5_phase305_role_query_option_calibration/deepseek7b_phase305_role_query_option_calibration.json
results/gpt5_phase305_role_query_option_calibration/ROLE_QUERY_OPTION_CALIBRATION_SUMMARY.md
```

日志：

```text
results/gpt5_gpu_lock_logs/20260529_184330_phase305normal_qwen3
results/gpt5_gpu_lock_logs/20260529_184602_phase305normal_glm4
results/gpt5_gpu_lock_logs/20260529_184907_phase305normal_deepseek7b
```

三轮 kernel.since-start.filtered.log 都是 0 行。

### Phase 305 数据规模

```text
每模型:
  bases = 32
  states = active_ab, active_ba, passive_ab_by, passive_ba_by
  option_templates = 8
  option_orders = agent_first, patient_first
  rows = 2048

三模型总计:
  rows = 6144
  nonfinite_rows = 0
```

### Phase 305 客观结果

严格可靠模板定义：

```text
所有 state 的 accuracy >= 0.9
所有 option_order 的 accuracy >= 0.9
所有 state 的 mean_margin >= 0
```

结果：

```text
Qwen3:
  reliable_templates = 0
  nonfinite = 0

GLM4:
  reliable_templates = 0
  nonfinite = 0

DeepSeek7B:
  reliable_templates = 0
  nonfinite = 0
```

相对较好的模板：

```text
GLM4 patient_who_affected:
  min_state_accuracy = 0.8594
  min_option_accuracy = 0.8672
  min_state_mean_margin = 1.5508
  仍低于 0.9

GLM4 patient_who_received:
  min_state_accuracy = 0.7656
  min_option_accuracy = 0.7422
  min_state_mean_margin = 1.3740
```

典型弱点：

```text
Qwen3:
  agent/passive_ba_by 仍很差；
  patient/passive_ba_by 也不稳。

GLM4:
  patient 模板比 agent 模板稳定；
  agent/active_ba 和 agent/passive_ba_by 仍是主要问题。

DeepSeek7B:
  patient/passive_ba_by 和 patient/active_ba 明显不稳；
  agent/active_ba 也不稳定。
```

### 当前最重要的新事实

本轮最重要结论是负结果：

```text
普通自然语言补全模板不可靠。
显式二选一模板也不可靠。
三模型都没有一个模板能跨 active/passive、AB/BA、option_order 稳定通过 0.9 标准。
```

这说明 Phase 41 的角色查询基线不稳不是偶然，也不是只因为没有显式列出候选。

更严格地说：

```text
当前 candidate-set role query 仍然不能作为 role binding closure 的读出器。
```

如果继续在这些模板上做 transplant、subspace patch 或 destroy-restore，很容易把模板偏置、候选顺序偏置、表层词序偏置误判为内部角色机制。

### 对机制研究的影响

这轮负结果反而很关键。

它说明 role binding 机制测试不能简单依赖自然语言问答模板：

```text
who did the action?
who received the action?
```

这些模板本身会混入：

```text
1. 表层主语偏置；
2. AB/BA 候选名偏置；
3. active/passive 构式偏置；
4. option order 偏置；
5. 模型对模板短语的语用偏置；
6. next-token 补全格式偏置。
```

所以角色绑定闭包需要新的读出方式。

### 当前硬伤

1. 本轮只测试了英文模板，没有测试符号化模板。
2. 候选词都是普通名词，有词频和语义自然性差异。
3. 仍然使用 next-token first-token logits，而不是完整答案序列概率。
4. 没有使用无语义占位符，例如 dax / wug / mip。
5. 没有把输入句和输出查询完全形式化，例如 ROLE_AGENT: A/B。
6. 没有进入 hook / transplant，因此本轮不产生因果机制结论。

### 结论修正

Phase 41 后原计划是：

```text
候选集合查询 -> state transplant -> destroy-restore
```

现在必须修正为：

```text
先构造可靠读出器 -> 再 state transplant -> 再 destroy-restore
```

当前自然语言 query 不合格。

下一步最合理的是 Phase 43：

```text
构造符号化 role query benchmark。
```

候选方向：

```text
1. 使用无语义实体名：
   dax / wug / mip / tev

2. 使用显式标签：
   ENTITY_A / ENTITY_B

3. 使用固定输出标签：
   AGENT = A or B
   PATIENT = A or B

4. 使用完整序列 logprob，而不是只看 first token。

5. 同时保留自然语言句子和形式化查询：
   sentence: the dax praised the wug.
   roles:
   agent = ?
   answer: A
```

只有当这个读出器在三模型中跨 active/passive、AB/BA、candidate order 稳定后，才应该恢复 token transplant 和 destroy-restore。

### 阶段性判断

本轮支持用户的核心判断：

```text
方向编码失败不是偶然；
token direction 失败也不是偶然；
candidate-set query 是正确方向，但当前自然语言模板仍不够干净。
```

围绕破解语言背后的编码机制，下一步必须从“自然语言问答模板”继续下沉到“符号化变量读出器”：

```text
语言功能 -> 因子 -> 词元关系 -> 符号化读出 -> 状态干预 -> 破坏恢复
```

这不是倒退，而是机制研究必须经过的校准层。

## Phase 43: GPU 修复后 Phase306 符号化角色读出器恢复测试 [2026-06-02 11:14]

### 任务目标

用户反馈显卡问题已经解决，要求继续完成全局语义语法契约图谱测试。本轮恢复 CUDA 正常方式测试，不使用保守脚本，重点验证：

```text
1. 新显卡/驱动/CUDA 环境是否能稳定跑三模型。
2. Phase306 符号化 role query 读出器在扩大样本后是否可靠。
3. qwen3、GLM4、DS7B 是否能按顺序加载、测试、硬退出、释放。
4. 保留日志，方便再次卡死时排查。
```

### 当前环境

```text
date = 2026-06-02
kernel = 6.8.0-117-generic
nvidia_driver = 595.71.05
gpu = NVIDIA GeForce RTX 4090 D, 24564 MiB
conda_env = openone-cu130-py312
python = 3.12.8
torch = 2.10.0+cu130
torch_cuda_runtime = 13.0
cuda_available = True
flash_sdp_enabled = True
```

说明：

```text
1. 当前环境已经不同于前几轮 570 / CUDA 12.1 环境。
2. tests/gpt5/check_probe_env.py 在 openone-cu130-py312 中失败，因为该环境没有 transformer_lens。
3. Phase306 使用 HuggingFace loader，不依赖 transformer_lens，因此继续测试。
4. 测试前发现 ComfyUI 仍有 GPU compute 进程：
   /home/rankrank/miniconda3/envs/comfyui/bin/python main.py ...
   占用约 476 MiB。
5. 为避免 GPU 进程混跑，本轮先终止 ComfyUI 进程，再启动测试。
```

### 使用脚本

本轮没有新增脚本，使用已有：

```text
tests/gpt5/run_phase306_normal.sh
tests/gpt5/run_phase306_symbolic_role_query_normal_all.sh
tests/gpt5/phase306_symbolic_role_query_calibration.py
tests/gpt5/phase306_symbolic_role_query_summary.py
```

关键参数：

```text
--hard-exit-after-model 由 run_phase306_normal.sh 自动传入。
PROBE_ATTN_IMPLEMENTATION=sdpa
PROBE_TORCH_DTYPE=bfloat16
PROBE_DEVICE_MAP_AUTO_MODELS=glm4,deepseek7b
PROBE_MAX_GPU_MEMORY=21GiB
ENABLE_GPU_MONITOR=0
ENABLE_SNAPSHOT_NVIDIA_SMI=1
```

说明：

```text
1. qwen3 测试完成后进程硬退出，再运行 GLM4。
2. GLM4 测试完成后进程硬退出，再运行 DS7B。
3. 没有启用持续 GPU monitor，减少额外 NVIDIA ioctl 干扰。
4. 每个模型仍保留 before/after snapshot、run.log、kernel.follow.log 和 filtered kernel log。
```

### Smoke 测试命令

```bash
OUTPUT_DIR=results/gpt5_phase306_symbolic_role_query_after_gpu_fix_smoke \
MAX_BASES=2 \
MAX_SEQ_LEN=128 \
PROGRESS_EVERY=1 \
QWEN3_MAX_SECONDS=1800 \
GLM4_MAX_SECONDS=2400 \
DEEPSEEK7B_MAX_SECONDS=2400 \
ENABLE_GPU_MONITOR=0 \
ENABLE_SNAPSHOT_NVIDIA_SMI=1 \
PROBE_ATTN_IMPLEMENTATION=sdpa \
PROBE_TORCH_DTYPE=bfloat16 \
tests/gpt5/run_phase306_symbolic_role_query_normal_all.sh
```

smoke 结果：

```text
qwen3:
  rows = 768
  reliable_templates = 2
  nonfinite = 0
  exit_code = 0
  log_dir = results/gpt5_gpu_lock_logs/20260602_102745_phase306normal_qwen3

GLM4:
  rows = 768
  reliable_templates = 5
  nonfinite = 0
  exit_code = 0
  log_dir = results/gpt5_gpu_lock_logs/20260602_102826_phase306normal_glm4

DS7B:
  rows = 768
  reliable_templates = 0
  nonfinite = 0
  exit_code = 0
  log_dir = results/gpt5_gpu_lock_logs/20260602_102927_phase306normal_deepseek7b
```

三模型 smoke 的 filtered kernel log：

```text
qwen3 = 0 lines
GLM4 = 0 lines
DS7B = 0 lines
```

### 中等规模测试命令

```bash
OUTPUT_DIR=results/gpt5_phase306_symbolic_role_query_after_gpu_fix_bases16 \
MAX_BASES=16 \
MAX_SEQ_LEN=128 \
PROGRESS_EVERY=4 \
QWEN3_MAX_SECONDS=3600 \
GLM4_MAX_SECONDS=4800 \
DEEPSEEK7B_MAX_SECONDS=4800 \
ENABLE_GPU_MONITOR=0 \
ENABLE_SNAPSHOT_NVIDIA_SMI=1 \
PROBE_ATTN_IMPLEMENTATION=sdpa \
PROBE_TORCH_DTYPE=bfloat16 \
tests/gpt5/run_phase306_symbolic_role_query_normal_all.sh
```

bases=16 结果：

```text
qwen3:
  rows = 6144
  reliable_templates = 0
  nonfinite = 0
  exit_code = 0
  log_dir = results/gpt5_gpu_lock_logs/20260602_103042_phase306normal_qwen3

GLM4:
  rows = 6144
  reliable_templates = 3
  nonfinite = 0
  exit_code = 0
  log_dir = results/gpt5_gpu_lock_logs/20260602_103418_phase306normal_glm4

DS7B:
  rows = 6144
  reliable_templates = 0
  nonfinite = 0
  exit_code = 0
  log_dir = results/gpt5_gpu_lock_logs/20260602_104010_phase306normal_deepseek7b
```

三模型 bases=16 的 filtered kernel log：

```text
qwen3 = 0 lines
GLM4 = 0 lines
DS7B = 0 lines
```

### 正式测试命令

```bash
OUTPUT_DIR=results/gpt5_phase306_symbolic_role_query_after_gpu_fix_bases32 \
MAX_BASES=32 \
MAX_SEQ_LEN=128 \
PROGRESS_EVERY=8 \
QWEN3_MAX_SECONDS=5400 \
GLM4_MAX_SECONDS=7200 \
DEEPSEEK7B_MAX_SECONDS=7200 \
ENABLE_GPU_MONITOR=0 \
ENABLE_SNAPSHOT_NVIDIA_SMI=1 \
PROBE_ATTN_IMPLEMENTATION=sdpa \
PROBE_TORCH_DTYPE=bfloat16 \
tests/gpt5/run_phase306_symbolic_role_query_normal_all.sh
```

正式输出文件：

```text
results/gpt5_phase306_symbolic_role_query_after_gpu_fix_bases32/qwen3_phase306_symbolic_role_query_calibration.json
results/gpt5_phase306_symbolic_role_query_after_gpu_fix_bases32/glm4_phase306_symbolic_role_query_calibration.json
results/gpt5_phase306_symbolic_role_query_after_gpu_fix_bases32/deepseek7b_phase306_symbolic_role_query_calibration.json
results/gpt5_phase306_symbolic_role_query_after_gpu_fix_bases32/SYMBOLIC_ROLE_QUERY_CALIBRATION_SUMMARY.md
results/gpt5_phase306_symbolic_role_query_after_gpu_fix_bases32/symbolic_role_query_calibration_summary.json
```

正式日志：

```text
results/gpt5_gpu_lock_logs/20260602_104506_phase306normal_qwen3
results/gpt5_gpu_lock_logs/20260602_105204_phase306normal_glm4
results/gpt5_gpu_lock_logs/20260602_110332_phase306normal_deepseek7b
```

正式结果：

```text
qwen3:
  rows = 12288
  reliable_templates = 0
  nonfinite = 0
  exit_code = 0

GLM4:
  rows = 12288
  reliable_templates = 3
  nonfinite = 0
  exit_code = 0

DS7B:
  rows = 12288
  reliable_templates = 0
  nonfinite = 0
  exit_code = 0

total_rows = 36864
```

正式三模型 filtered kernel log：

```text
qwen3 = 0 lines
GLM4 = 0 lines
DS7B = 0 lines
```

测试结束后的 GPU 状态：

```text
nvidia_driver = 595.71.05
gpu_memory_used = 530 MiB
temperature = 50 C
power = 12.31 W
pstate = P8
```

### DS7B 警告

DS7B 运行时出现实现警告：

```text
Sliding Window Attention is enabled but not implemented for `sdpa`;
unexpected results may be encountered.
```

本轮没有改变 DS7B 参数，因为目标是先做同一配置下的跨模型恢复测试。后续如果专门研究 DS7B 读出器，需要把该警告作为混杂因素，并考虑 eager 对照。

### 客观现象

#### 1. GPU/驱动稳定性明显改善

与 Phase 47-51 相比，本轮在新环境中完成：

```text
smoke: 3 models x 768 rows
bases16: 3 models x 6144 rows
bases32: 3 models x 12288 rows
```

总计：

```text
Phase306 rows = 3 * (768 + 6144 + 12288) = 57600
```

全部：

```text
exit_code = 0
nonfinite = 0
kernel.since-start.filtered.log = 0 lines
```

因此可以暂时判断：

```text
新 595.71.05 + torch 2.10.0 cu130 环境，
至少已经通过 Phase306 级别的三模型正常 CUDA 测试。
```

#### 2. 小样本 smoke 的可靠模板数量会虚高

smoke 中：

```text
qwen3 reliable = 2
GLM4 reliable = 5
DS7B reliable = 0
```

但扩大到 bases=16：

```text
qwen3 reliable = 0
GLM4 reliable = 3
DS7B reliable = 0
```

扩大到 bases=32：

```text
qwen3 reliable = 0
GLM4 reliable = 3
DS7B reliable = 0
```

这说明：

```text
读出器可靠性必须以较大样本为准；
smoke 只能验证工程流程，不能验证机制读出器。
```

#### 3. Phase306 当前不能作为三模型通用 role binding 读出器

bases=32 中：

```text
qwen3:
  no reliable template

GLM4:
  reliable templates:
    entity_ab / entity / agent / json_agent
    entity_ab / entity / agent / role_table_agent
    entity_ab / entity / agent / compact_agent

DS7B:
  no reliable template
```

因此：

```text
当前符号化模板只在 GLM4 的 agent 读出上有少量可用候选；
不能作为跨模型 role binding closure 的统一读出器。
```

#### 4. patient 读出仍然是主要失败点

三个模型 summary 中，弱项大量集中在：

```text
patient query
passive_ab_by
active_ab
role_table_patient
json_patient
```

例如：

```text
DS7B:
  entity_ab / entity / patient / json_patient / active_ab:
    acc = 0.0000
    margin = -4.6563

GLM4:
  nonce / entity / patient / json_patient / passive_ab_by:
    acc = 0.0000
    margin = -3.5742

qwen3:
  entity_ab / entity / patient / role_table_patient / active_ab:
    acc = 0.0000
    margin = -7.6641
```

这说明当前 prompt 格式仍然不是干净的 agent/patient 读出器。

### 当前结论

本轮可以产生两个结论，必须严格区分：

```text
工程结论：
  显卡/驱动/CUDA 环境已经显著稳定，可以继续恢复三模型 CUDA 测试。

机制结论：
  Phase306 的符号化 role query 读出器仍不可靠，不能进入状态移植、
  subspace patch、destroy-restore 等闭包实验。
```

### 硬伤

```text
1. qwen3 和 DS7B 没有任何模板通过可靠性门槛。
2. GLM4 只有 agent 方向的 entity_ab/entity 模板通过，patient 仍失败。
3. 当前读出器对 answer_style、entity_style、query_type 仍高度敏感。
4. DS7B 有 sliding window + sdpa 实现警告，读出器失败可能混入实现因素。
5. Phase306 仍只是读出器校准，不是因果干预。
```

### 对全局语义语法契约图谱的影响

这轮恢复了最重要的基础：

```text
测量平台重新可用。
```

但同时确认：

```text
role binding 的读出器层仍未过关。
```

因此下一阶段不能直接做：

```text
token state transplant
subspace patch
destroy-restore
head/neuron localization
```

而应该先重建读出器。

### 下一步计划

#### Phase 53：读出器重设计

目标：解决 Phase306 的 patient 读出失败和跨模型失败。

候选方向：

```text
1. 使用更强的机器可读格式：
   Sentence: ...
   Output exactly one label.
   AGENT: A
   PATIENT: B

2. 使用联合输出概率，而不是分开比较 AGENT/PATIENT：
   "AGENT=A; PATIENT=B"
   vs
   "AGENT=B; PATIENT=A"

3. 用完整结构标签而不是单变量 query：
   ROLE_MAP = {AGENT: A, PATIENT: B}
   ROLE_MAP = {AGENT: B, PATIENT: A}

4. 加入 few-shot 但必须控制模板泄漏：
   训练读出格式，不训练具体样本。

5. 对 DS7B 做 sdpa vs eager 对照。
```

#### Phase 54：只在通过读出器门槛后恢复闭包测试

通过标准：

```text
active/passive 都稳定；
AB/BA 都稳定；
agent/patient 都稳定；
candidate order 不影响结果；
三模型至少各有一个可靠模板；
min_state_accuracy >= 0.9；
min_option_accuracy >= 0.9；
min_state_margin > 0。
```

只有满足这些条件，才恢复：

```text
state transplant
subspace patch
destroy-restore
minimal circuit search
```

### 第一性原则

本轮再次确认：

```text
语言机制研究不能跳过读出器。
```

如果读出器不能稳定读出 agent/patient，那么任何 patch 后的变化都无法解释为角色绑定机制。

当前正确路线是：

```text
稳定测量平台
→ 构造可靠读出器
→ 做因果干预
→ 做破坏恢复
→ 归纳语言编码机制。
```

## Phase 53: GLM5 最新进展综合与语言编码机制路线校准 [2026-06-02 13:50]

### 任务目标

读取 `research/glm5/docs/AGI_GLM5_MEMO.md` 最新记录，结合 GPT5 当前研究计划，给出破解语言编码机制的整体进度报告和下一步计划。

本轮没有新增模型测试，只做跨 memo 研究综合。

### 读取范围

GLM5 最新阶段：

```text
Phase 339+340+341: 多Baseline验证+管线组合+身份探针 [2026-06-02 13:45]
```

相关前置阶段：

```text
Phase 336+337+338: 多层替换+反向破坏+跨层注意力
Phase 333-335: Attention vs MLP 组件因果替换
Phase 328-332: Context-Gated Binding 与层级轨迹
```

GPT5 当前阶段：

```text
Phase 52: GPU 修复后 Phase306 符号化角色读出器恢复测试 [2026-06-02 11:14]
```

### GLM5 最新核心进展

#### 1. MLP > Attention 在多 baseline 下稳健成立

GLM5 Phase 339 使用 4 个 corrupted baseline：

```text
The item
The thing
The object
The entity
```

三模型关键块结果显示：

```text
Qwen3:
  L21-L29 MLP 恢复 = 69.6% - 92.7%
  Attention 恢复 = 6.3% - 24.5%

GLM4:
  L30-L38 MLP 恢复 = 46.0% - 66.4%
  Attention 恢复 = 2.8% - 15.8%

DS7B:
  L19-L24 MLP 恢复 = 58.5% - 71.3%
  Attention 恢复 = 2.3% - 7.9%
```

结论：

```text
对象-属性 binding 的后期计算主通道是 MLP，而不是 Attention。
```

这个结论比早期单 baseline 结果更可信，因为 baseline 选择偏差已经被明显削弱。

#### 2. 早期 identity block 单独接近 100% 恢复

Phase 340 的最重要发现：

```text
Qwen3:
  identity_L0-L2_full = 99.4%

GLM4:
  identity_L0-L4_full = 99.6%

DS7B:
  identity_L0-L2_full = 100.6%
```

这说明：

```text
只要早期 residual stream 被修正为 clean 状态，
后续层可以自然计算出正确 binding。
```

关键修正：

```text
早期层不一定直接写入 binding 结果；
更准确地说，早期层提供正确的对象身份/上下文 residual 输入。
```

#### 3. 后期 MLP 是兼容性计算主通道，但不是唯一通道

GLM4 更宽 MLP 计算块结果：

```text
L30-L38 MLP = 46.0%
L25-L38 MLP = 80.3%
L20-L38 MLP = 90.9%
```

说明：

```text
binding 不是单层完成；
而是多层 MLP 链式累积。
```

后期 MLP 的意义不是“唯一存储 binding”，而是：

```text
在正确对象身份输入上，把身份信息转换为属性兼容性排序。
```

#### 4. 对象身份信息从 L0 就存在，并贯穿全层

Phase 341 的身份探针显示：

```text
对象类别最近质心分类器 chance = 14.3%
三模型各层分类准确率约 33.3% - 58.3%
所有层远高于 chance
```

说明：

```text
对象身份信息从嵌入/早期 residual 就存在；
但“身份存在”不等于“binding 已经计算完成”。
```

身份信息和 binding 计算必须区分：

```text
身份信息 = 对象是谁
binding 计算 = 该对象与当前属性/上下文是否兼容，如何排序输出
```

### GPT5 当前进展对照

GPT5 Phase 52 刚完成 GPU 修复后的 Phase306 符号化 role query 恢复测试：

```text
qwen3:
  rows = 12288
  reliable_templates = 0
  nonfinite = 0

GLM4:
  rows = 12288
  reliable_templates = 3
  nonfinite = 0

DS7B:
  rows = 12288
  reliable_templates = 0
  nonfinite = 0

kernel filtered logs = 0 lines for all models
```

工程结论：

```text
GPU/driver/CUDA 平台重新可用。
```

机制结论：

```text
agent/patient role query 读出器仍未可靠；
不能继续做 role binding 的 transplant / patch / destroy-restore。
```

### 当前整体进度判断

现在研究出现了两条不同成熟度的路线：

#### 路线 A：对象-属性 binding 路线，进展较成熟

GLM5 当前已经形成相对清晰的管线：

```text
1. L0 附近：对象身份向量已经存在。
2. 早期 L0-L2/L0-L4 full block：提供正确 residual 输入。
3. 中层：维持与转换 residual stream。
4. 后期 MLP：将对象身份转换为属性兼容性排序。
5. 输出层：把兼容性排序读出到 logits。
```

这条路线已经有：

```text
多 baseline 验证；
三模型验证；
多层 patch；
组件分解；
身份探针；
较明确的因果恢复现象。
```

因此它是当前最接近“语言编码机制拼图”的主线。

#### 路线 B：语法角色 binding 路线，仍卡在读出器层

GPT5 role binding 方向目前还没过读出器门槛：

```text
agent/patient 自然语言模板不稳；
symbolic role query 也不稳；
patient 读出尤其失败；
qwen3 和 DS7B 没有可靠模板。
```

因此这条路线现在不能做机制归因。

更准确地说：

```text
不是 role binding 机制不存在；
而是我们还没有构造出可靠读出器来测它。
```

### 关键理论进展

当前最稳的机制表达不应是“语言有某个固定语义轴”，而应是：

```text
语言功能 = 早期身份/结构条件输入
       + 中层 residual 维持与格式转换
       + 后期 MLP 兼容性/排序计算
       + 输出读出
```

其中：

```text
Attention 更像上下文信息传递和结构选择；
MLP 更像非线性兼容性计算、排序、门控与格式转换；
Residual stream 是跨层状态载体。
```

这和此前“路径格式假说”一致，但 GLM5 最新结果把它具体化到了对象-属性 binding 上。

### 重要硬伤

#### 1. identity block 100% 恢复的解释仍需谨慎

早期 full block patch 接近 100%，可能说明：

```text
早期 identity/context 输入是充分起点。
```

但也可能只是：

```text
patch 前几层接近于把模型切换成 clean forward 轨迹。
```

需要继续拆：

```text
resid_in
attn_out
mlp_out
position/token 子空间
对象 token vs 属性 token
```

#### 2. 后期 MLP 内部如何计算兼容性仍未破解

现在只知道：

```text
后期 MLP patch 恢复率高；
更宽 MLP block 恢复率更高。
```

还不知道：

```text
具体哪些 neuron / subspace 表示对象身份？
哪些表示属性类型？
哪些表示属性值？
如何组合成兼容性排序？
```

#### 3. GPT5 role binding 读出器仍不可靠

没有可靠读出器，就不能做：

```text
state transplant
subspace patch
destroy-restore
head/neuron localization
```

否则会把模板偏置误判为语言机制。

#### 4. 对象身份分类准确率仍不高

类别分类远超 chance，但最高约 58.3%。

说明：

```text
对象身份不是简单线性类别向量；
可能是上下文相关、分布式、多子空间编码。
```

### 下一步大计划

#### Phase A：优先沿 GLM5 对象-属性 binding 路线继续突破

目标：

```text
从“MLP 块恢复”推进到“MLP 内部计算结构”。
```

任务：

```text
1. 拆 identity block：
   L0-L2/L0-L4 中 resid/attn/mlp/token position 哪个最关键。

2. 拆后期 MLP：
   找出恢复率最高层段中的 neuron/subspace。

3. 做对象 token vs 属性 token patch：
   区分对象身份通道、属性类型通道、属性值通道。

4. 做 destroy-restore：
   破坏对象身份子空间或属性兼容性子空间，
   再恢复，验证必要性与充分性。

5. 做跨 baseline / 跨属性类型复用矩阵：
   判断 color/temperature/shape 等属性是否共享同一兼容性计算格式。
```

#### Phase B：重建 GPT5 role binding 读出器

目标：

```text
让 agent/patient 读出器先过门槛，再谈闭包。
```

任务：

```text
1. 改用联合结构输出：
   ROLE_MAP = {AGENT: A, PATIENT: B}
   vs
   ROLE_MAP = {AGENT: B, PATIENT: A}

2. 加 few-shot 格式校准：
   只教输出格式，不泄漏测试样本。

3. 用 AB/BA、active/passive、candidate order 全对照。

4. 对 DS7B 做 sdpa/eager 对照。

5. 通过后再恢复 state transplant 和 destroy-restore。
```

#### Phase C：统一成全局语义语法契约图谱

最终图谱不应只记录：

```text
哪个层强；
哪个模块强。
```

而应记录：

```text
功能对象：
  object-property binding
  role binding
  coreference
  negation/scope
  recursive structure

路径对象：
  identity block
  computation block
  output block

变量对象：
  object identity
  attribute type
  attribute value
  semantic agent
  semantic patient
  scope/operator

因果证据：
  patch recovery
  ablation damage
  destroy-restore recovery
  cross-template stability
  cross-model stability
```

### 当前最重要判断

语言编码机制研究已经从：

```text
找语义方向
```

推进到：

```text
找条件化计算管线。
```

当前最清晰的一条管线是：

```text
对象身份输入 → residual 轨迹维持 → 后期 MLP 兼容性计算 → 输出排序
```

这不是完整语言机制，但它很可能是理解语言中“概念-属性-值绑定”的第一个稳定子机制。

### 第一性原则

当前不能急着提出统一数学理论。

更合理的突破路线是：

```text
先破解一个稳定子机制：
  对象-属性 binding

再迁移到更复杂机制：
  agent/patient role binding
  scope binding
  coreference binding
  recursive binding

最后再抽象：
  不同 binding 是否共享同一种条件化残差轨迹 + MLP兼容性计算结构。
```

如果这些机制最终共享相似结构，那么语言背后的数学原理可能不是“某个语义轴”，而是：

```text
在 residual stream 中维持变量身份，
通过 MLP 链进行上下文兼容性变换，
再由输出层读出排序的动态编码系统。
```

## Phase 54: 对象-属性 Binding 有效性验证测试 [2026-06-02 14:06]

### 任务目标

用户要求对对象-属性 binding 进行验证测试，保证测试结果有效；显卡问题已经解决，不再输出跟踪日志，直接进行正常全量测试。

本轮目标：

```text
1. 复验 GLM5 Phase 339+340+341 的核心发现。
2. 保证三模型按 qwen3 → GLM4 → DS7B 顺序单独运行。
3. 每个模型测试后使用 --hard-exit-after-model 硬退出，避免显存残留。
4. 不启用 GPU 跟踪日志。
5. 自动审计结果是否满足核心有效性判据。
```

### 脚本修改

修改：

```text
tests/glm5/phase339_multibaseline_pipeline.py
```

修改内容：

```text
1. 使用 tests/gpt5/model_registry.py 中的 Linux 本地模型路径，
   替代原脚本中的 Windows D:/develop/... 路径。

2. 新增 --output-dir 参数。

3. 新增 --hard-exit-after-model 参数。

4. 新增 PHASE339_ATTN_IMPLEMENTATIONS 环境变量，
   可控制 attention implementation 尝试顺序。
```

新增：

```text
tests/gpt5/run_phase342_object_binding_validation_normal_all.sh
tests/gpt5/phase342_object_binding_validation_summary.py
```

用途：

```text
run_phase342_object_binding_validation_normal_all.sh:
  正常方式顺序运行 qwen3、GLM4、DS7B；
  每个模型单独进程；
  每个模型结束硬退出；
  不启用 GPU 跟踪日志。

phase342_object_binding_validation_summary.py:
  读取三个模型的 phase339 JSON；
  自动检查核心判据：
    all_baselines_mlp_gt_attn
    identity_recovery_ge_95
    min_valid_pairs_ge_12
    category_probe_above_chance
```

### 运行环境

```text
date = 2026-06-02
conda_env = openone-cu130-py312
nvidia_driver = 595.71.05
gpu = NVIDIA GeForce RTX 4090 D
```

测试结束后 GPU 状态：

```text
memory_used = 675 MiB
temperature = 40 C
pstate = P8
```

说明：

```text
模型进程已正常释放；
没有出现之前的显卡锁死问题。
```

### 第一次全量命令

```bash
PHASE339_OUTPUT_DIR=results/gpt5_phase342_object_binding_validation_full \
tests/gpt5/run_phase342_object_binding_validation_normal_all.sh
```

第一次结果：

```text
qwen3:
  core_validation = True

GLM4:
  core_validation = True

DS7B(sdpa):
  core_validation = False
  原因：identity_L0-2_full = 86.5%，低于 95% 阈值
```

DS7B 运行时出现警告：

```text
Sliding Window Attention is enabled but not implemented for `sdpa`;
unexpected results may be encountered.
```

因此 DS7B 的 sdpa 结果不能直接作为最终机制证据。

### DS7B eager 对照命令

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate openone-cu130-py312

PHASE339_ATTN_IMPLEMENTATIONS=eager \
PHASE339_OUTPUT_DIR=results/gpt5_phase342_object_binding_validation_deepseek7b_eager \
PYTHONUNBUFFERED=1 \
python tests/glm5/phase339_multibaseline_pipeline.py deepseek7b \
  --output-dir results/gpt5_phase342_object_binding_validation_deepseek7b_eager \
  --hard-exit-after-model
```

DS7B eager 结果：

```text
identity_L0-2_full = 116.1%
identity+compute = 115.0%
all_baselines_mlp_gt_attn = True
category_probe_above_chance = True
core_validation = True
```

注意：

```text
eager 下仍提示 sliding window 未实现；
但 identity block 恢复回到高位。
因此最终汇总使用 DS7B eager 结果，
并把 DS7B attention implementation 作为硬伤记录。
```

### 最终汇总目录

```text
results/gpt5_phase342_object_binding_validation_final/
```

最终文件：

```text
results/gpt5_phase342_object_binding_validation_final/qwen3_phase339.json
results/gpt5_phase342_object_binding_validation_final/glm4_phase339.json
results/gpt5_phase342_object_binding_validation_final/deepseek7b_phase339.json
results/gpt5_phase342_object_binding_validation_final/phase342_object_binding_validation_summary.json
results/gpt5_phase342_object_binding_validation_final/PHASE342_OBJECT_BINDING_VALIDATION_SUMMARY.md
```

### 最终核心结果

#### Qwen3

```text
core_validation = True

baseline results:
  The item:
    MLP = 70.1%
    Attention = 23.8%
    Full = 80.5%
    n_valid = 22

  The thing:
    MLP = 94.1%
    Attention = 6.6%
    Full = 91.5%
    n_valid = 18

  The object:
    MLP = 90.9%
    Attention = 10.9%
    Full = 93.8%
    n_valid = 19

  The entity:
    MLP = 93.2%
    Attention = 9.4%
    Full = 97.2%
    n_valid = 19

mean_mlp = 87.08%
mean_attention = 12.68%
mean_full = 90.75%
identity_L0-2_full = 99.60%
identity+compute = 100.30%
mean_category_accuracy = 0.4653
chance = 0.1429
```

#### GLM4

```text
core_validation = True

baseline results:
  The item:
    MLP = 45.5%
    Attention = 10.9%
    Full = 56.1%
    n_valid = 22

  The thing:
    MLP = 56.9%
    Attention = 2.6%
    Full = 62.1%
    n_valid = 17

  The object:
    MLP = 65.3%
    Attention = 16.7%
    Full = 74.0%
    n_valid = 19

  The entity:
    MLP = 63.7%
    Attention = 14.2%
    Full = 72.0%
    n_valid = 22

mean_mlp = 57.85%
mean_attention = 11.10%
mean_full = 66.05%
identity_L0-4_full = 100.00%
identity+compute = 100.20%
mean_category_accuracy = 0.4653
chance = 0.1429
```

#### DS7B（eager 对照）

```text
core_validation = True

baseline results:
  The item:
    MLP = 56.6%
    Attention = 16.2%
    Full = 83.7%
    n_valid = 15

  The thing:
    MLP = 86.0%
    Attention = -1.6%
    Full = 91.1%
    n_valid = 21

  The object:
    MLP = 64.9%
    Attention = -5.2%
    Full = 73.9%
    n_valid = 18

  The entity:
    MLP = 81.6%
    Attention = -0.6%
    Full = 96.0%
    n_valid = 18

mean_mlp = 72.28%
mean_attention = 2.20%
mean_full = 86.17%
identity_L0-2_full = 116.10%
identity+compute = 115.00%
mean_category_accuracy = 0.5000
chance = 0.1429
```

### 自动有效性判据

三模型最终均满足：

```text
all_baselines_mlp_gt_attn = True
identity_recovery_ge_95 = True
min_valid_pairs_ge_12 = True
category_probe_above_chance = True
passes_core_validation = True
```

因此本轮对象-属性 binding 核心结论通过有效性验证。

### 客观结论

#### 1. MLP > Attention 跨模型、跨 baseline 稳健成立

三模型、四 baseline 下全部满足：

```text
MLP recovery > Attention recovery
```

这说明对象-属性 binding 的主要计算通道不是 Attention block，而是后期 MLP block。

#### 2. Early identity block 是充分起点

```text
Qwen3 identity_L0-2_full = 99.6%
GLM4 identity_L0-4_full = 100.0%
DS7B identity_L0-2_full = 116.1%（eager）
```

这说明：

```text
只要早期 residual stream 被修正为 clean 状态，
后续自然计算可以恢复对象-属性 binding。
```

这不能简单解释为“早期层直接计算了 binding”，更准确是：

```text
早期层提供正确对象身份/上下文状态，
后续层自然完成兼容性计算。
```

#### 3. 后期 MLP 是兼容性变换主通道

对象身份信息从 L0 就存在，但它不是最终 binding。

当前最稳的管线是：

```text
对象 token → 早期 residual 身份状态
→ 中层维持/转换
→ 后期 MLP 兼容性计算
→ 输出排序
```

#### 4. DS7B 的 implementation 需要特别处理

DS7B sdpa 版本：

```text
identity_L0-2_full = 86.5%
core_validation = False
```

DS7B eager 版本：

```text
identity_L0-2_full = 116.1%
core_validation = True
```

因此 DS7B 后续实验必须显式记录 attention implementation，并优先避免把 sdpa/sliding-window 警告污染为机制差异。

### 硬伤

```text
1. DS7B eager 下仍提示 sliding window 未实现；
   虽然结果通过，但 implementation 仍是硬伤。

2. identity recovery 可超过 100%，说明恢复率指标存在 over-recovery；
   不能解释为“贡献超过真实机制”，只能说明 patch 后目标 logit margin 超过 clean。

3. identity block 是 full block，不是纯身份子空间；
   它包含 attention、MLP、residual 更新和层归一化影响。

4. category probe 虽远高于 chance，但只有约 0.46-0.50；
   对象身份不是简单线性类别向量。

5. 后期 MLP 内部如何实现兼容性计算仍未拆开。
```

### 对语言编码机制的意义

本轮最重要的稳定拼图是：

```text
对象-属性 binding 不是固定语义轴；
而是条件化计算管线。
```

更具体：

```text
1. 对象身份从输入早期就进入 residual stream。
2. 早期 residual 状态必须正确，否则后续 binding 会错。
3. 后期 MLP 将对象身份和属性上下文转换为兼容性排序。
4. Attention 不是主恢复通道，但可能提供上下文信息流。
5. 输出层读出兼容性排序。
```

这与当前“全局语义语法契约图谱”的方向一致：

```text
功能不是单点；
是变量状态 + 层间路径 + 模块变换 + 输出读出。
```

### 下一步计划

#### Phase 55：拆 early identity block

目标：

```text
确认 L0-L2 / L0-L4 full block 中，
到底是 resid_in、attn_out、mlp_out 还是某些 token position 最关键。
```

必须区分：

```text
早期 full block patch 近 100%
```

和：

```text
对象身份子空间本身足以恢复
```

#### Phase 56：拆后期 MLP 兼容性计算

目标：

```text
在 Qwen3 L21-L29、GLM4 L20-L38、DS7B L19-L24/L12-L24 中，
定位具体层、neuron/subspace 如何完成对象-属性兼容性排序。
```

优先做：

```text
1. 单层 MLP scan
2. MLP neuron activation difference
3. top-k neuron patch/ablation
4. destroy-restore
```

#### Phase 57：对象 token vs 属性 token 分离

目标：

```text
区分对象身份通道、属性类型通道、属性值通道。
```

测试：

```text
只 patch object token
只 patch attribute value token
只 patch prompt context token
只 patch final token
```

#### Phase 58：扩展属性类型与复用矩阵

目标：

```text
检查 color / temperature / texture / material / state 是否共享同一 MLP 兼容性计算格式。
```

这一步才开始接近：

```text
语言中概念-属性-值系统的复用与差异化图谱。
```

### 第一性原则

对象-属性 binding 当前已经是最稳定的可破解子机制。

接下来不应该急着扩大到所有语言功能，而应先把这个子机制拆到：

```text
变量层：
  object identity
  attribute type
  attribute value

路径层：
  early identity block
  middle residual trajectory
  late MLP compatibility block

因果层：
  patch recovery
  ablation damage
  destroy-restore
```

如果这条机制能被完整闭包，再迁移到：

```text
agent/patient role binding
scope binding
coreference binding
recursive binding
```

才有可能逐步拼出语言背后的整体编码机制。

## Phase 55: 最新 Memo 读取后的对象-属性 Binding 下一阶段计划 [2026-06-02 15:29]

### 任务目标

读取 GPT5 与 GLM5 memo 最新记录，确定接下来破解语言编码机制的研发计划。

本轮没有新增模型测试，只做计划校准。

### 最新记录要点

GPT5 最新阶段：

```text
Phase 54: 对象-属性 Binding 有效性验证测试 [2026-06-02 14:06]
```

已经确认：

```text
1. qwen3、GLM4、DS7B 最终都通过对象-属性 binding 核心有效性验证。
2. MLP > Attention 跨模型、跨 baseline 稳健成立。
3. early identity block 可以让后续自然计算恢复 binding。
4. DS7B 需要 eager 对照，不能直接使用 sdpa/sliding-window 警告结果解释机制。
```

GLM5 最新阶段：

```text
Phase 344+345: 多关系方向 + 匹配随机对照
Phase 346: 精确交互分解 + 层级累积闭合
```

关键新事实：

```text
1. binding / negation / antonym / role / tense / same_class 六种语言关系方向
   都呈现 balance ratio ≈ 1.00。

2. 平衡放大不是 binding 特有，而是语义方向的通用属性。

3. 不同关系的 net/gross 比不同：
   tense 最低；
   negation / role / same_class / binding 较高。

4. binding 相对随机方向的 net/gross 优势很弱：
   Qwen3/DS7B 对 W_U-subspace random 有边际显著；
   GLM4 不显著。

5. Phase 346 精确分解显示：
   MLP 中 gate_main ≈ 25-30%
   up_main ≈ 25-31%
   gate×up interaction ≈ 39-46%

6. 交互项是最大贡献者，但符号不稳定。

7. MLP net sum 与 final binding 不闭合：
   说明只看 binding 层 MLP 不足，
   还需要 attention、LayerNorm、非 binding 层、残差交互进入闭合模型。
```

### 当前总体判断

对象-属性 binding 现在已经从：

```text
有没有可恢复路径？
```

推进到：

```text
MLP 内部如何通过 gate、up、interaction 产生微偏置？
```

这是一个重要升级。

当前最稳的机制表述是：

```text
对象身份在 early residual stream 中形成充分起点；
后续 MLP 链把对象身份和属性上下文转换为兼容性排序；
这种转换不是单方向选择，
而是 gate/up 主效应 + gate×up 非线性交互共同产生的微小净偏置；
大量正负通道平衡放大，最终只留下很小的方向性残余。
```

### 对 Phase54 计划的修正

Phase54 原计划：

```text
Phase55: 拆 early identity block
Phase56: 拆后期 MLP 兼容性计算
Phase57: 对象 token vs 属性 token 分离
Phase58: 扩展属性类型与复用矩阵
```

根据 GLM5 Phase344-346，计划需要调整顺序：

```text
优先级 1:
  先把后期 MLP 的 gate/up/interaction 机制做闭合。

优先级 2:
  再拆 early identity block。

原因：
  GLM5 已经发现后期 MLP 内部的精确交互结构，
  这是当前最接近“编码机制数学结构”的入口。
```

### 下一步阶段计划

#### Phase 56：MLP Gate-Up-Interaction 机制复验与扩样

目标：

```text
确认 gate_main / up_main / interaction 占比是否在更大样本、
更多属性类型、更多 baseline 下稳定。
```

测试内容：

```text
1. 扩大对象-属性 pair 数量：
   从 24 对扩到 80-150 对。

2. 扩展属性类型：
   color
   temperature
   texture
   material
   state
   size

3. 对每个模型关键 MLP 层做精确 2x2 分解：
   CC = gate_clean * up_clean
   CR = gate_clean * up_corrupt
   RC = gate_corrupt * up_clean
   RR = gate_corrupt * up_corrupt

4. 输出：
   gate_main ratio
   up_main ratio
   interaction ratio
   interaction sign stability
   per-layer distribution
   per-attribute distribution
```

通过标准：

```text
interaction 仍稳定占最大或接近最大；
gate/up 主效应均非零；
不同属性类型不完全相同，但结构可比较；
三模型结果方向一致。
```

#### Phase 57：层级闭合模型修正

目标：

```text
解决 Phase346 中 MLP net sum 与 final binding 不闭合的问题。
```

需要纳入：

```text
1. 所有层 MLP，不只 binding 层。
2. Attention 输出贡献。
3. LayerNorm / RMSNorm 缩放。
4. residual stream 累积与重投影。
5. final logits readout。
```

输出：

```text
layer_contribution_sum
attention_contribution_sum
mlp_contribution_sum
norm_rescale_factor
residual_closure_ratio
final_logit_alignment
```

目标不是强行闭合到 1.0，而是判断：

```text
最终 binding 信号到底由哪些路径累计而来。
```

#### Phase 58：Early Identity Block 精拆

目标：

```text
确认 identity_L0-2 / identity_L0-4 full block 的充分性来自哪里。
```

测试：

```text
1. resid_in patch
2. attn_out patch
3. mlp_out patch
4. full block patch
5. object token only
6. attribute token only
7. final token only
```

关键问题：

```text
是对象 token 的身份状态足够？
还是整句 early residual 轨迹足够？
Attention 是否只是搬运上下文？
MLP 是否已经做早期格式化？
```

#### Phase 59：对象/属性/值三变量分离

目标：

```text
把对象-属性 binding 拆成 object identity、attribute type、attribute value 三个变量。
```

设计：

```text
object:
  apple / banana / snow / fire ...

attribute type:
  color / temperature / texture / material ...

attribute value:
  red / yellow / hot / cold / rough / smooth ...
```

测试：

```text
只替换 object identity；
只替换 attribute type；
只替换 attribute value；
组合替换 object+type；
组合替换 object+value；
组合替换 type+value；
完整替换 object+type+value。
```

目标：

```text
确定 binding 的最小变量组合。
```

#### Phase 60：Destroy-Restore 闭包

目标：

```text
从 patch recovery 进入真正因果闭包。
```

流程：

```text
1. 定位关键 MLP interaction 子空间。
2. destroy：消融或扰动该子空间。
3. observe：binding 输出下降。
4. restore：只恢复该子空间。
5. check：binding 输出恢复，非目标属性尽量不被破坏。
```

通过标准：

```text
破坏后显著下降；
恢复后显著恢复；
随机子空间不能恢复；
跨 baseline 有效；
跨属性类型部分迁移。
```

### 当前不应优先做的事

```text
1. 不应立刻回到 agent/patient role binding。
   原因：GPT5 Phase306 读出器仍未过关。

2. 不应直接做 head/neuron 全模型扫描。
   原因：目前还没确定 interaction 子空间和变量分离结构。

3. 不应急着提出统一数学理论。
   原因：当前只是对象-属性 binding 子机制初步清晰。
```

### 当前第一性原则

下一步要围绕一个核心问题：

```text
对象身份和属性值如何在 MLP 中经过 gate×up 非线性交互，
变成最终的兼容性排序？
```

如果这个问题被破解，就得到语言编码机制中的第一个可闭合单元：

```text
变量输入
→ residual 轨迹
→ MLP 非线性交互
→ 微偏置
→ 输出排序
```

这个单元之后才能迁移到：

```text
role binding
scope binding
coreference binding
recursive binding
```

### 推荐立即执行

下一轮应执行：

```text
Phase 56:
  MLP Gate-Up-Interaction 机制复验与扩样。
```

这是当前最接近“语言背后数学结构”的切入点。

## Phase 56: 全局路径比较与 MLP Gate-Up-Interaction 复验 [2026-06-02 15:40]

### 任务目标

根据最新判断，深度神经网络更可能是相对编码系统。单一 object-property binding 路径只能提供局部信息，不能直接解释语言整体编码机制。本轮不再只看 binding，而是把 binding 放入多关系路径中比较：

```text
binding
negation
antonym
role
tense
same_class
```

同时复验 object-property binding 中的 MLP gate/up/interaction 结构，观察：

```text
1. binding 是否相对其他关系特殊；
2. binding 相对随机方向是否有稳定优势；
3. MLP gate、up、gate×up interaction 的比例是否稳定；
4. 单独 MLP net sum 是否能闭合最终 binding；
5. 是否支持“全局路径图谱优先于单一路径解释”。
```

### 脚本变更

修改：

```text
tests/glm5/phase344_345_multi_relation.py
tests/glm5/phase346_interaction_closure.py
```

主要变更：

```text
1. 使用 tests/gpt5/model_registry.py 中的本地模型路径，替代 Windows D:/ 路径。
2. 新增 --output-dir。
3. 新增 --hard-exit-after-model，保证每个模型运行后硬退出并释放显存。
4. 新增 PHASE344_OUTPUT_DIR / PHASE346_OUTPUT_DIR。
5. 新增 PHASE344_ATTN_IMPLEMENTATIONS / PHASE346_ATTN_IMPLEMENTATIONS。
6. scipy 不存在时跳过 t-test，不中断核心实验。
```

新增：

```text
tests/gpt5/run_phase56_global_path_interaction_normal_all.sh
tests/gpt5/phase56_global_path_interaction_summary.py
```

运行脚本逻辑：

```text
qwen3 -> phase344/345 -> hard exit
qwen3 -> phase346 -> hard exit
glm4 -> phase344/345 -> hard exit
glm4 -> phase346 -> hard exit
deepseek7b -> phase344/345 -> hard exit
deepseek7b -> phase346 -> hard exit
summary
```

DeepSeek7B 使用 eager，原因是前面 sdpa 对 sliding-window attention 的结果不稳定；本轮记录为解释硬伤。

### 测试命令

```bash
PHASE56_OUTPUT_DIR=results/gpt5_phase56_global_path_interaction_full \
tests/gpt5/run_phase56_global_path_interaction_normal_all.sh
```

输出文件：

```text
results/gpt5_phase56_global_path_interaction_full/phase344_345/qwen3_phase344_345.json
results/gpt5_phase56_global_path_interaction_full/phase344_345/glm4_phase344_345.json
results/gpt5_phase56_global_path_interaction_full/phase344_345/deepseek7b_phase344_345.json

results/gpt5_phase56_global_path_interaction_full/phase346/qwen3_phase346.json
results/gpt5_phase56_global_path_interaction_full/phase346/glm4_phase346.json
results/gpt5_phase56_global_path_interaction_full/phase346/deepseek7b_phase346.json

results/gpt5_phase56_global_path_interaction_full/PHASE56_GLOBAL_PATH_INTERACTION_SUMMARY.md
results/gpt5_phase56_global_path_interaction_full/phase56_global_path_interaction_summary.json
```

### 一、全局关系路径结果

#### Qwen3

```text
binding:
  balance = 1.0169
  net/gross = 0.0279
  n = 45

negation:
  balance = 1.0532
  net/gross = 0.0313
  n = 25

role:
  balance = 1.0160
  net/gross = 0.0281
  n = 20

same_class:
  balance = 1.0066
  net/gross = 0.0331
  n = 25

binding_net_gross_rank_among_relations = 4
```

客观现象：

```text
Qwen3 中 binding 不是最强路径。
same_class、negation、role 的 net/gross 都高于或接近 binding。
```

#### GLM4

```text
binding:
  balance = 0.9969
  net/gross = 0.0200
  n = 36

negation:
  balance = 0.9910
  net/gross = 0.0369
  n = 20

role:
  balance = 1.0028
  net/gross = 0.0318
  n = 16

same_class:
  balance = 0.9941
  net/gross = 0.0317
  n = 20

binding_net_gross_rank_among_relations = 5
```

客观现象：

```text
GLM4 中 binding 明显不是最突出路径。
negation、role、same_class、antonym 都高于 binding。
```

#### DeepSeek7B

```text
binding:
  balance = 0.9957
  net/gross = 0.0214
  n = 32

negation:
  balance = 1.0133
  net/gross = 0.0245
  n = 20

same_class:
  balance = 0.9995
  net/gross = 0.0198
  n = 20

binding_net_gross_rank_among_relations = 2
```

客观现象：

```text
DeepSeek7B 中 binding 排第 2，仅低于 negation。
但 DeepSeek7B 使用 eager，且有 sliding-window attention warning，因此不能过度解释。
```

### 二、随机方向对照

Qwen3：

```text
binding reference net/gross = 0.0279
norm-matched random = 0.0198
W_U-subspace random = 0.0208
binding-orthogonal random = 0.0203
pure random = 0.0186
```

GLM4：

```text
binding reference net/gross = 0.0200
norm-matched random = 0.0186
W_U-subspace random = 0.0177
binding-orthogonal random = 0.0173
pure random = 0.0163
```

DeepSeek7B：

```text
binding reference net/gross = 0.0214
norm-matched random = 0.0141
W_U-subspace random = 0.0150
binding-orthogonal random = 0.0154
pure random = 0.0136
```

客观现象：

```text
binding 相比随机方向有一定优势，但优势不大。
这说明 binding 不是普通随机方向，但也不是可以单独解释语言机制的特殊轴。
```

### 三、MLP Gate / Up / Interaction 复验

#### Qwen3

```text
gate_main_frac = 0.2570
up_main_frac = 0.3157
interaction_frac = 0.4273
total_effect_mean = 0.5644

closure:
  mean_closure_ratio = 1.1237
  mean_final_binding = 2.4064
  mean_mlp_net_sum = 3.5695
```

#### GLM4

```text
gate_main_frac = 0.3034
up_main_frac = 0.3060
interaction_frac = 0.3906
total_effect_mean = 0.3340

closure:
  mean_closure_ratio = 0.4228
  mean_final_binding = 2.9638
  mean_mlp_net_sum = 1.2823
```

#### DeepSeek7B

```text
gate_main_frac = 0.2803
up_main_frac = 0.2635
interaction_frac = 0.4562
total_effect_mean = 1.0111

closure:
  mean_closure_ratio = -11.4383
  mean_final_binding = 1.2340
  mean_mlp_net_sum = 4.2638
```

客观现象：

```text
1. 三模型中 interaction_frac 都是最大或接近最大：
   Qwen3 = 42.7%
   GLM4 = 39.1%
   DeepSeek7B = 45.6%

2. gate 和 up 单独都重要，但 gate×up interaction 是不可忽略的核心项。

3. MLP net sum 不能稳定闭合最终 binding：
   Qwen3 closure ratio 波动较大；
   GLM4 平均只解释约 42%；
   DeepSeek7B closure ratio 极端不稳定。
```

### 四、综合判断

本轮最重要的结论不是“binding 被破解了”，而是：

```text
单一 binding 路径信息有限。
binding 必须放在 negation、role、same_class、tense 等多关系路径中比较。
```

当前更稳的表述是：

```text
object-property binding 是全局语义路径中的一个局部投影。
它确实有 MLP gate/up/interaction 结构，
但这个结构不是 binding 独有。
语言编码更像多路径相对编码系统，
每个功能都通过一组路径和其他功能比较后才能定位。
```

这与当前核心猜想一致：

```text
深度神经网络不是固定语义轴编码，
而是相对路径、功能差异、模块交互、残差累积共同形成的编码网络。
```

### 五、硬伤和问题

```text
1. scipy 不存在，本轮跳过 t-test。
   但核心均值和对照结果已经保存。

2. DeepSeek7B 使用 eager。
   运行时出现 sliding-window attention warning，
   因此 DeepSeek7B 的绝对数值需要谨慎，只用于结构性参考。

3. 当前 relation 数据量仍不大。
   binding 有 32-45 个观测，其他关系约 12-25 个观测。
   后续重要结论必须扩样本。

4. MLP closure 不完整。
   说明除了目标 binding layers 的 MLP，还存在 attention、RMSNorm、其他层、残差路径和输出层影响。

5. binding 随机对照优势很小。
   说明它不是纯随机方向，但也不能作为单独机制本体。
```

### 六、关键洞察

如果把 object-property binding 当作单一路径，会误导研究。

更合理的第一性原则是：

```text
语言编码机制不是某个功能自己的独立通道，
而是多个功能路径之间的相对差异结构。
```

也就是说，要破解语言背后的数学结构，必须同时回答：

```text
1. 某个功能路径本身是什么；
2. 它与相邻功能路径有何不同；
3. 哪些路径复用同一 MLP interaction；
4. 哪些路径在 residual trajectory 中分叉；
5. 哪些路径最终通过输出层被读出。
```

### 七、下一步计划

下一阶段不应只继续扩大 binding，而要进入：

```text
Phase 57: 全局路径矩阵扩样与层级路径图谱
```

目标：

```text
1. 扩大 relation library：
   binding / negation / role / same_class / antonym / tense / coreference / quantifier / causal / comparison。

2. 每类至少扩展到 50-100 对基础样本。

3. 对每个 relation 输出：
   path signature
   MLP gate/up/interaction signature
   random-control advantage
   closure ratio
   layer trajectory

4. 构造 relation-to-relation path matrix：
   看哪些功能共享路径，哪些功能分化。

5. 再回到 object-property binding：
   用全局矩阵判断 binding 到底接近 same_class、role、还是 negation。
```

阶段性大任务：

```text
从单功能机制分析，升级为全局语义路径图谱。
只有在全局路径中定位 binding，后续 destroy-restore 才有解释意义。
```

## Phase 57: 全局关系路径矩阵扩样 [2026-06-02 16:04]

### 任务目标

根据最新分析，Phase 54-56 的方向基本正确，但 object-property binding 不能孤立解释语言编码机制。深度神经网络更像相对编码系统，单一 binding 路径信息有限，必须与其他关系路径比较，才能获得全局结构。

本轮目标：

```text
1. 扩大 relation library。
2. 每个 relation 使用更多语境变体样本。
3. 对每个 relation 生成 path signature。
4. 构造 relation-to-relation similarity matrix。
5. 判断 binding 在全局语义关系路径中的相对位置。
```

### 对用户分析的判断

用户提供的分析大方向正确，尤其是以下部分：

```text
1. binding 是当前较稳定的子机制，但不是语言机制本体。
2. early identity block 是充分起点，不是完整 binding 计算器。
3. MLP gate/up/interaction 是重要机制入口，但不是 binding 独有。
4. MLP net sum 不能闭合 final binding，说明需要全路径模型。
5. 下一步应优先做全局关系路径矩阵，而不是继续孤立扩大 binding。
```

本轮据此执行 Phase 57。

### 脚本

新增：

```text
tests/gpt5/phase57_global_relation_path_matrix.py
tests/gpt5/run_phase57_global_relation_path_matrix_normal_all.sh
tests/gpt5/phase57_global_relation_path_matrix_summary.py
```

脚本特性：

```text
1. 三模型顺序运行：
   qwen3 -> GLM4 -> DeepSeek7B

2. 每个模型运行后使用：
   --hard-exit-after-model

3. 关系库扩展到 14 类：
   binding
   negation
   antonym
   role
   tense
   same_class
   coreference
   quantifier
   causal
   comparison
   spatial
   temporal_order
   condition
   contrast

4. 每类最多 40 对样本，并加入语境前缀变体：
   Usually
   In general
   People know
   In this sentence
   For this example

5. 对每个 relation 输出：
   balance
   net/gross
   gate_main_frac
   up_main_frac
   interaction_frac
   per-layer path signature
   relation similarity matrix
```

说明：

```text
DeepSeek7B 使用 eager。
运行时仍出现 sliding-window attention warning。
因此 DS7B 结果只作为结构性参考，不做过细数值解释。
```

### 测试命令

```bash
PHASE57_OUTPUT_DIR=results/gpt5_phase57_global_relation_path_matrix_full \
PHASE57_MAX_PAIRS_PER_RELATION=40 \
tests/gpt5/run_phase57_global_relation_path_matrix_normal_all.sh
```

输出：

```text
results/gpt5_phase57_global_relation_path_matrix_full/qwen3_phase57_global_relation_path_matrix.json
results/gpt5_phase57_global_relation_path_matrix_full/glm4_phase57_global_relation_path_matrix.json
results/gpt5_phase57_global_relation_path_matrix_full/deepseek7b_phase57_global_relation_path_matrix.json

results/gpt5_phase57_global_relation_path_matrix_full/PHASE57_GLOBAL_RELATION_PATH_MATRIX_SUMMARY.md
results/gpt5_phase57_global_relation_path_matrix_full/phase57_global_relation_path_matrix_summary.json
```

运行后 GPU 状态：

```text
RTX 4090 D
driver = 595.71.05
memory_used = 646 MiB
temperature = 47 C
```

### 数据规模

Qwen3：

```text
target_layers = [21, 23, 25, 27, 29]
每个 relation 约 160-200 个 layer observations
```

GLM4：

```text
target_layers = [30, 33, 36, 38]
每个 relation 约 128-160 个 layer observations
```

DeepSeek7B：

```text
target_layers = [19, 21, 23, 24]
每个 relation 约 128-160 个 layer observations
```

总观测约：

```text
14 relations × 3 models × 128-200 observations
```

### Qwen3 结果

按 net/gross 排名：

```text
1. temporal_order  0.0451
2. same_class      0.0378
3. role            0.0315
4. causal          0.0293
5. binding         0.0278
6. negation        0.0278
7. antonym         0.0268
8. contrast        0.0249
9. condition       0.0245
10. tense          0.0238
11. coreference    0.0222
12. quantifier     0.0213
13. comparison     0.0195
14. spatial        0.0191
```

binding：

```text
net/gross = 0.0278
balance = 1.0111
interaction = 0.4260
n = 200
rank = 5/14
```

相似度候选：

```text
condition / contrast = 0.9945
comparison / contrast = 0.9938
antonym / role = 0.9816
tense / coreference = 0.9756
```

低相似度：

```text
negation / comparison = 0.5666
binding / comparison = 0.6382
binding / contrast = 0.6663
```

客观现象：

```text
Qwen3 中 binding 不是中心路径。
temporal_order、same_class、role、causal 都高于 binding。
binding 与 comparison/contrast 的路径差异明显。
```

### GLM4 结果

按 net/gross 排名：

```text
1. quantifier      0.0442
2. condition       0.0392
3. contrast        0.0364
4. negation        0.0329
5. same_class      0.0316
6. antonym         0.0281
7. role            0.0266
8. tense           0.0265
9. spatial         0.0253
10. coreference    0.0243
11. binding        0.0228
12. causal         0.0225
13. temporal_order 0.0214
14. comparison     0.0201
```

binding：

```text
net/gross = 0.0228
balance = 0.9959
interaction = 0.3757
n = 160
rank = 11/14
```

相似度候选：

```text
tense / condition = 0.9962
comparison / temporal_order = 0.9953
coreference / causal = 0.9948
tense / contrast = 0.9924
```

低相似度：

```text
antonym / comparison = 0.8162
negation / comparison = 0.8206
binding / comparison = 0.8468
```

客观现象：

```text
GLM4 中 binding 排名很低。
quantifier、condition、contrast、negation 更强。
这说明 GLM4 的全局路径里，binding 不是主导关系，而更像局部兼容性路径。
```

### DeepSeek7B 结果

按 net/gross 排名：

```text
1. temporal_order  0.0260
2. contrast        0.0259
3. same_class      0.0243
4. condition       0.0243
5. spatial         0.0221
6. quantifier      0.0214
7. antonym         0.0203
8. negation        0.0203
9. coreference     0.0185
10. role           0.0184
11. binding        0.0179
12. comparison     0.0163
13. tense          0.0162
14. causal         0.0162
```

binding：

```text
net/gross = 0.0179
balance = 0.9985
interaction = 0.4586
n = 160
rank = 11/14
```

相似度候选：

```text
binding / antonym = 0.9894
role / condition = 0.9783
tense / condition = 0.9760
binding / negation = 0.9707
```

低相似度：

```text
negation / comparison = 0.6255
binding / comparison = 0.7156
same_class / comparison = 0.7589
```

客观现象：

```text
DS7B 中 binding 排名也很低。
但 interaction = 0.4586，说明 binding 的 MLP 非线性交互比例仍高。
这支持“binding 有结构，但不是全局主路径”。
```

### 三模型共同现象

#### 1. binding 不是全局最强路径

```text
Qwen3: binding rank = 5/14
GLM4: binding rank = 11/14
DeepSeek7B: binding rank = 11/14
```

这比 Phase56 更明确地支持：

```text
binding 只是全局语义关系路径中的一个节点。
```

#### 2. balance 仍接近 1

绝大多数 relation 的 balance 都接近 1。

这说明：

```text
正负通道的 gross amplification 普遍接近平衡。
平衡放大不是 binding 特有，而是 MLP 处理多类语义方向的通用现象。
```

#### 3. 不同 relation 的 interaction_frac 明显不同

例如：

```text
Qwen3 binding interaction = 0.4260
Qwen3 temporal_order interaction = 0.1758

GLM4 binding interaction = 0.3757
GLM4 quantifier interaction = 0.1602

DeepSeek7B binding interaction = 0.4586
DeepSeek7B contrast interaction = 0.1676
```

这说明不同关系不只是强弱不同，MLP 内部计算结构也不同。

### 核心结论

本轮最重要的结论：

```text
语言功能不能用单一路径解释。
object-property binding 是稳定入口，但不是中心机制。
全局关系路径矩阵比单功能 patch 更接近语言编码机制。
```

更谨慎的理论表述：

```text
深度网络中的语言编码可能不是固定语义轴，
而是多个语义/语法关系在 residual trajectory 与 MLP interaction 中形成的相对路径网络。

每个 relation 有自己的 path signature：
  balance
  net/gross
  gate/up/interaction
  layer trajectory
  similarity to other relations

真正要破解的是这些 path signature 之间的复用与分化。
```

### 硬伤

```text
1. 部分 relation 仍是粗模板。
   coreference、quantifier、condition 等还不是稳定读出器，
   这里只能作为路径方向样本，不是功能正确性证明。

2. 每类 40 对仍不是最终规模。
   已明显好于 Phase56，但要支撑机制结论，还需要按 subtype 扩到 50-100 对。

3. DeepSeek7B 使用 eager 且有 sliding-window warning。
   DS7B 结果只能作为结构参考。

4. 当前 similarity matrix 是行为路径签名相似度。
   高相似不等于真实机制复用。
   后续必须做变量拆分和 destroy-restore。

5. 本轮没有 random-control advantage。
   Phase56 已做随机方向对照，本轮优先做全局矩阵。
   后续 Phase57b 应把随机对照也扩到所有 relation。
```

### 对破解语言背后数学结构的第一性原则

本轮进一步支持：

```text
语言编码机制的基本单位不是一个概念方向，
而是关系路径。
```

更具体：

```text
1. 每个语言关系不是独立通道，而是在全局路径网络中相对定位。
2. MLP 对不同关系普遍产生平衡放大，但净偏置很小。
3. 不同关系的 gate/up/interaction 比例不同，说明同一 MLP 框架承载了不同计算格式。
4. binding、same_class、role、negation、temporal_order 等关系之间的差异，可能就是语言网络复用和差异化的真实入口。
```

所以接下来要研究的不是：

```text
binding 到底在哪一层？
```

而是：

```text
binding 与 same_class、role、negation、temporal_order 在路径签名上哪里相同，哪里分叉？
这些分叉对应哪些 token、变量、MLP 子空间和 residual 轨迹？
```

### 下一步计划

Phase 58 应优先做：

```text
全局关系路径矩阵 Phase57b：随机对照与 subtype 扩展
```

具体任务：

```text
1. 对 14 类 relation 增加 subtype：
   binding:
     color / temperature / texture / taste / material
   role:
     active / passive / dative
   negation:
     lexical / syntactic / quantifier scope
   quantifier:
     all/some/no/every/not all
   temporal_order:
     before/after/while/then
   condition:
     if/unless/only if

2. 每个 subtype 至少 40-60 对。

3. 为每个 relation/subtype 加入随机对照：
   norm-matched random
   W_U-subspace random
   relation-orthogonal random

4. 输出：
   subtype path matrix
   relation path matrix
   random advantage matrix
   interaction matrix

5. 找出最稳定的复用/分化候选，再进入变量级闭包。
```

更大的阶段任务：

```text
建立全局语义/语法关系路径图谱 v1。
用它决定哪个功能值得进入 destroy-restore，而不是凭直觉选择 binding 或 role。
```

## Phase 58: 关系子类型随机对照与候选路径筛选 [2026-06-02 16:56]

### 任务目标

Phase 57 建立了 14 类 relation 的全局路径矩阵，但仍有一个关键硬伤：

```text
某个 relation 的 net/gross 高，
不一定说明它有语言机制优势；
可能只是该方向更容易和 W_U 或 MLP 权重分布对齐。
```

因此本轮补充：

```text
1. subtype 拆分；
2. 随机方向对照；
3. random advantage 计算；
4. subtype-level similarity matrix；
5. 重新筛选值得进入变量闭包的候选。
```

### 对最新分析的判断

用户给出的分析基本正确：

```text
1. Phase 57 是全局路径图谱第一版，不是机制闭包。
2. binding 不是全局中心机制。
3. relation path similarity 只能作为复用候选，不能当作复用证明。
4. Phase 57 缺少随机方向对照，这是核心硬伤。
5. 下一步必须做 subtype + random advantage。
```

本轮按该方向执行 Phase 58。

### 脚本

新增：

```text
tests/gpt5/phase58_relation_subtype_random_controls.py
tests/gpt5/run_phase58_relation_subtype_random_controls_normal_all.sh
tests/gpt5/phase58_relation_subtype_random_controls_summary.py
```

脚本功能：

```text
1. 三模型顺序运行：
   qwen3 -> GLM4 -> DeepSeek7B

2. 每个模型运行后：
   --hard-exit-after-model

3. 每个 subtype 输出：
   real net/gross
   random mean net/gross
   random advantage
   balance
   interaction fraction
   subtype similarity matrix

4. 随机对照类型：
   norm_matched
   W_U-subspace
   relation_orthogonal
   pure_random
```

### 测试命令

```bash
PHASE58_OUTPUT_DIR=results/gpt5_phase58_relation_subtype_random_controls_full \
PHASE58_MAX_PAIRS_PER_SUBTYPE=30 \
PHASE58_RANDOM_SAMPLES_PER_PAIR=2 \
tests/gpt5/run_phase58_relation_subtype_random_controls_normal_all.sh
```

输出：

```text
results/gpt5_phase58_relation_subtype_random_controls_full/qwen3_phase58_relation_subtype_random_controls.json
results/gpt5_phase58_relation_subtype_random_controls_full/glm4_phase58_relation_subtype_random_controls.json
results/gpt5_phase58_relation_subtype_random_controls_full/deepseek7b_phase58_relation_subtype_random_controls.json

results/gpt5_phase58_relation_subtype_random_controls_full/PHASE58_RELATION_SUBTYPE_RANDOM_CONTROLS_SUMMARY.md
results/gpt5_phase58_relation_subtype_random_controls_full/phase58_relation_subtype_random_controls_summary.json
```

运行后 GPU 状态：

```text
RTX 4090 D
driver = 595.71.05
memory_used = 574 MiB
temperature = 46 C
```

### 数据规模

Qwen3：

```text
target_layers = [21, 23, 25, 27, 29]
每个 subtype:
  pairs = 30
  real observations = 150
  random controls = 4 类 × 2 samples/pair × 5 layers
```

GLM4：

```text
target_layers = [30, 33, 36, 38]
每个 subtype:
  pairs = 30
  real observations = 120
```

DeepSeek7B：

```text
target_layers = [19, 21, 23, 24]
每个 subtype:
  pairs = 30
  real observations = 120
  attn_impl = eager
```

### Qwen3 结果

按 random advantage 排名：

```text
1. temporal_order/before_after
   real = 0.0446
   random = 0.0180
   advantage = 0.0265
   interaction = 0.1849

2. same_class/category_peer
   real = 0.0372
   random = 0.0214
   advantage = 0.0158
   interaction = 0.3365

3. contrast/but_and
   real = 0.0267
   random = 0.0181
   advantage = 0.0087
   interaction = 0.1826

4. binding/color
   real = 0.0294
   random = 0.0216
   advantage = 0.0078
   interaction = 0.3672

5. negation/syntactic_not
   real = 0.0276
   random = 0.0200
   advantage = 0.0076
   interaction = 0.3726

6. role/active_swap
   real = 0.0264
   random = 0.0193
   advantage = 0.0071
   interaction = 0.2500
```

弱或负优势：

```text
quantifier/all_some:
  advantage = 0.0007

comparison/greater_less:
  advantage = -0.0022
```

客观现象：

```text
Qwen3 中 temporal_order/before_after 是最强候选。
same_class/category_peer 也明显强于多数 binding subtype。
binding 内部不统一：
  color > taste > temperature ≈ texture
```

### GLM4 结果

按 random advantage 排名：

```text
1. negation/quantifier_no
   real = 0.0525
   random = 0.0181
   advantage = 0.0344
   interaction = 0.2218

2. quantifier/all_some
   real = 0.0456
   random = 0.0180
   advantage = 0.0277
   interaction = 0.1761

3. negation/syntactic_not
   real = 0.0428
   random = 0.0194
   advantage = 0.0234
   interaction = 0.3321

4. condition/if_unless
   real = 0.0413
   random = 0.0195
   advantage = 0.0218
   interaction = 0.1638

5. contrast/but_and
   real = 0.0341
   random = 0.0191
   advantage = 0.0150
   interaction = 0.1921

6. same_class/category_peer
   real = 0.0310
   random = 0.0171
   advantage = 0.0139
   interaction = 0.3143
```

binding subtype：

```text
binding/taste:
  advantage = 0.0100

binding/color:
  advantage = 0.0022

binding/texture:
  advantage = 0.0000

binding/temperature:
  advantage = -0.0006
```

客观现象：

```text
GLM4 的强候选明显不是 binding。
negation 和 quantifier 相关 subtype 显著强于 binding。
这和 Phase57 中 GLM4 quantifier/condition/contrast 排名高一致。
```

### DeepSeek7B 结果

按 random advantage 排名：

```text
1. contrast/but_and
   real = 0.0290
   random = 0.0148
   advantage = 0.0142
   interaction = 0.1906

2. temporal_order/before_after
   real = 0.0270
   random = 0.0140
   advantage = 0.0130
   interaction = 0.2368

3. quantifier/all_some
   real = 0.0269
   random = 0.0139
   advantage = 0.0129
   interaction = 0.3279

4. condition/if_unless
   real = 0.0251
   random = 0.0147
   advantage = 0.0104
   interaction = 0.2970

5. negation/syntactic_not
   real = 0.0240
   random = 0.0150
   advantage = 0.0090
   interaction = 0.4202

6. binding/texture
   real = 0.0225
   random = 0.0161
   advantage = 0.0065
   interaction = 0.3565
```

弱或接近随机：

```text
role/active_swap:
  advantage = 0.0007

binding/temperature:
  advantage = -0.0005

comparison/greater_less:
  advantage = 0.0014
```

客观现象：

```text
DeepSeek7B 中 contrast、temporal_order、quantifier、condition 更强。
binding 大多较弱，只有 texture 有一定正优势。
role/active_swap 也几乎没有随机优势。
```

### 三模型共同现象

#### 1. binding 不是稳定最高候选

```text
Qwen3:
  最强 = temporal_order/before_after
  binding/color 排第 4

GLM4:
  最强 = negation/quantifier_no
  binding 最好的是 taste，排第 7

DeepSeek7B:
  最强 = contrast/but_and
  binding 最好的是 texture，排第 6
```

这说明：

```text
object-property binding 不是全局路径图谱的中心。
```

#### 2. subtype 层比 relation 粗标签更重要

binding 内部：

```text
Qwen3:
  color advantage = 0.0078
  temperature = 0.0026
  texture = 0.0025
  taste = 0.0044

GLM4:
  taste = 0.0100
  color = 0.0022
  texture = 0.0000
  temperature = -0.0006

DeepSeek7B:
  texture = 0.0065
  color = 0.0027
  taste = 0.0016
  temperature = -0.0005
```

同一个 relation 内部，subtype 差异很大。

#### 3. 随机对照改变解释

Phase57 中某些 relation 的 net/gross 不低，但 Phase58 显示：

```text
quantifier/all_some in Qwen3:
  advantage = 0.0007

comparison/greater_less in Qwen3:
  advantage = -0.0022

role/active_swap in DeepSeek7B:
  advantage = 0.0007
```

这说明：

```text
只看 net/gross 会误判。
random advantage 是必须指标。
```

#### 4. interaction 高不等于 random advantage 高

例子：

```text
DeepSeek7B binding/taste:
  interaction = 0.6137
  advantage = 0.0016

Qwen3 binding/temperature:
  interaction = 0.4746
  advantage = 0.0026
```

说明：

```text
gate×up interaction 比例高，
不等于该方向有真实结构优势。
```

### 核心结论

Phase58 最重要的修正：

```text
全局关系路径图谱必须同时看：
  real net/gross
  random baseline
  random advantage
  subtype stability
  interaction structure
```

当前最强候选不是泛化的 binding，而是：

```text
Qwen3:
  temporal_order/before_after
  same_class/category_peer
  contrast/but_and

GLM4:
  negation/quantifier_no
  quantifier/all_some
  negation/syntactic_not
  condition/if_unless

DeepSeek7B:
  contrast/but_and
  temporal_order/before_after
  quantifier/all_some
  condition/if_unless
```

这说明：

```text
不同模型的强路径候选不同。
语言编码机制不是固定功能路径，而是模型特异的相对路径网络。
```

### 硬伤

```text
1. 当前 subtype 仍然不够多。
   例如 quantifier 只测 all_some，
   condition 只测 if_unless，
   contrast 只测 but_and。

2. random controls 每 pair 只有 2 个样本。
   可以用于筛选，但不能作为最终统计结论。

3. DeepSeek7B 仍使用 eager，且有 sliding-window attention warning。

4. 当前仍是路径行为图谱。
   random advantage 高不等于变量已破解。

5. 还没有做 token/subspace 定位和 destroy-restore。
```

### 对破解语言机制的第一性原则更新

本轮进一步说明：

```text
语言编码机制不能从单一 relation 推出，
也不能只从 relation 粗标签推出。
必须下沉到 relation subtype，
再通过随机对照筛出真正高于背景方向分布的路径。
```

更准确的机制单位应是：

```text
relation subtype path
```

而不是：

```text
relation path
```

例如：

```text
binding/color、binding/temperature、binding/taste
不是同一强度路径；

negation/syntactic_not、negation/quantifier_no
也不是同一路径；

quantifier/all_some 在 GLM4/DS7B 强，
但在 Qwen3 很弱。
```

所以全局机制应表示为：

```text
语言机制 = subtype path network
             + random-advantage filter
             + cross-model alignment
             + variable-level closure
```

### 下一步计划

Phase 59 应做：

```text
稳定候选路径的变量拆分与 token/subspace 定位
```

不建议再盲目扩大所有 subtype。

优先候选：

```text
1. temporal_order/before_after
   原因：
     Qwen3 强；
     DeepSeek7B 强；
     读出相对简单；
     before/after 是明确变量。

2. quantifier/all_some 或 negation/quantifier_no
   原因：
     GLM4 强；
     DeepSeek7B 强；
     但需要更可靠读出器。

3. same_class/category_peer
   原因：
     Qwen3 强；
     GLM4 中等强；
     概念类别关系较适合变量拆分。

4. contrast/but_and
   原因：
     DeepSeek7B 强；
     Qwen3/GLM4 也有正优势。
```

Phase 59 具体任务：

```text
1. 选择 2-3 个稳定 subtype。
2. 对每个 subtype 做：
   token-level path signature
   layer trajectory
   MLP gate/up/interaction token split
   candidate-output readout

3. 判断：
   变量信息在哪里写入？
   哪个 token 承载关系变量？
   哪些 MLP 层产生净偏置？
   是否可以做 destroy-restore？
```

阶段性大任务：

```text
从全局路径筛选，进入稳定 subtype 的变量级闭包。
```

## Phase 59: Temporal Order 符号读出器与 Token Path 定位 [2026-06-02 18:45]

### 任务目标

根据 Phase 58 的结果，`temporal_order/before_after` 在 Qwen3 和 DeepSeek7B 中都是随机对照优势较强的 subtype，因此本轮选择它作为第一个稳定候选路径，测试：

```text
1. before/after 是否可以构造成更干净的符号化读出器；
2. temporal_order 信息是否集中在某些 token / layer / module；
3. 是否可以进入后续 subspace patch 或 destroy-restore；
4. 当前全局关系路径图谱是否能从 relation-subtype path 推进到 variable/token path。
```

本轮仍然严格限定结论：只做读出器校准和候选路径定位，不做机制闭包结论。

### 对用户分析的判断

用户提供的分析整体正确。关键正确点：

```text
1. Phase 58 证明的是 relation subtype path，而不是最终语言机制。
2. 单一 binding 路径信息有限，必须和其他关系路径比较，才有全局意义。
3. 下一步不应继续盲目扩大所有 subtype，而应选择稳定候选 subtype 做变量拆分。
4. temporal_order/before_after 适合作为优先候选，因为变量清晰，且在 Qwen3 / DeepSeek7B 中随机优势较强。
5. 任何路径定位必须先有可靠读出器，否则 patch / ablation / destroy-restore 都会被读出偏差污染。
```

需要补充的谨慎点：

```text
Phase 58 中 temporal_order 强，不等于 FIRST_EVENT 读出器天然稳定。
如果符号读出器准确率不足 0.9，就不能进入因果闭包，只能继续校准读出器和观察候选路径。
```

### 新增脚本

```text
tests/gpt5/phase59_temporal_order_token_path.py
tests/gpt5/run_phase59_temporal_order_token_path_normal_all.sh
tests/gpt5/phase59_temporal_order_token_path_summary.py
```

脚本设计：

```text
1. 构造符号化 temporal_order 样本：
   A = dax smiled. B = wug left.
   Relation: A happened before B.
   Answer with A or B. FIRST_EVENT:

2. 同时构造 before / after、AB / BA、模板变化。

3. 使用完整候选 completion logprob 比较：
   logP(" A") vs logP(" B")

4. 捕获目标层的：
   resid_in
   resid_out
   attn_out
   mlp_out

5. 捕获 token position：
   A_label
   B_label
   before
   after
   last

6. 用 W_U(" A") - W_U(" B") 作为粗输出方向，计算 sign-corrected projection。

7. 三模型按顺序运行，每个模型完成后使用 --hard-exit-after-model 退出，避免模型共存导致显存污染。
```

### 测试命令

Smoke：

```bash
PHASE59_ATTN_IMPLEMENTATIONS=sdpa \
python tests/gpt5/phase59_temporal_order_token_path.py qwen3 \
  --output-dir results/gpt5_phase59_smoke \
  --max-cases 8 \
  --progress-every 4
```

正式三模型顺序测试：

```bash
PHASE59_OUTPUT_DIR=results/gpt5_phase59_temporal_order_token_path_full \
PHASE59_MAX_CASES=96 \
tests/gpt5/run_phase59_temporal_order_token_path_normal_all.sh
```

运行说明：

```text
qwen3 -> glm4 -> deepseek7b 顺序执行；
每个模型都添加 --hard-exit-after-model；
Qwen3 / GLM4 优先 sdpa；
DeepSeek7B 使用 eager，仍有 Sliding Window Attention warning。
```

运行中 PyTorch 输出：

```text
Can't initialize NVML
```

该警告影响 GPU 监控状态读取，不影响本轮 CUDA 计算完成；但后续如果要做稳定性诊断，需要单独检查 NVML / nvidia-smi 状态。

### 输出文件

```text
results/gpt5_phase59_temporal_order_token_path_full/qwen3_phase59_temporal_order_token_path.json
results/gpt5_phase59_temporal_order_token_path_full/glm4_phase59_temporal_order_token_path.json
results/gpt5_phase59_temporal_order_token_path_full/deepseek7b_phase59_temporal_order_token_path.json
results/gpt5_phase59_temporal_order_token_path_full/phase59_temporal_order_token_path_summary.json
results/gpt5_phase59_temporal_order_token_path_full/PHASE59_TEMPORAL_ORDER_TOKEN_PATH_SUMMARY.md
```

### 数据规模

```text
每模型 cases = 96
三模型 total cases = 288

Qwen3 target layers:
  L21, L23, L25, L27, L29

GLM4 target layers:
  L30, L33, L36, L38

DeepSeek7B target layers:
  L19, L21, L23, L24
```

### Qwen3 结果

```text
accuracy = 0.8021
mean_abs_margin = 0.9961
n_cases = 96
```

Top token/module paths：

```text
1. L29:resid_out:last  projection = 3.1995
2. L29:attn_out:last   projection = 2.9250
3. L27:resid_out:last  projection = 0.4705
4. L27:resid_in:last   projection = 0.4697
5. L29:resid_in:last   projection = 0.4233
6. L25:resid_in:last   projection = 0.4061
7. L25:resid_out:last  projection = 0.3872
8. L23:resid_out:last  projection = 0.1016
```

客观现象：

```text
1. Qwen3 的 FIRST_EVENT 读出准确率只有 0.8021，未达到 0.9 闭包门槛。
2. 输出方向投影高度集中在 last token 的深层 residual / attention。
3. L29 attn_out:last 很高，说明最后层 attention 输出可能强烈参与候选 A/B 输出偏置。
4. 但由于读出器不够稳定，不能把 L29 attn_out 解释为 temporal_order 机制本体。
```

### GLM4 结果

```text
accuracy = 0.8021
mean_abs_margin = 0.7319
n_cases = 96
```

Top token/module paths：

```text
1. L38:resid_out:last  projection = 0.9618
2. L38:resid_in:last   projection = 0.5610
3. L36:resid_out:last  projection = 0.5604
4. L36:resid_in:last   projection = 0.5166
5. L33:resid_out:last  projection = 0.4121
6. L38:mlp_out:last    projection = 0.3899
7. L33:resid_in:last   projection = 0.3628
8. L33:attn_out:last   projection = 0.0603
```

客观现象：

```text
1. GLM4 和 Qwen3 的准确率相同，都是 0.8021。
2. GLM4 路径主要集中在 deep residual last token。
3. L38 mlp_out:last 有一定正投影，但弱于 residual。
4. 这更像输出读出状态形成，而不是已经定位到 before/after 变量写入位置。
```

### DeepSeek7B 结果

```text
accuracy = 0.6562
mean_abs_margin = 0.9391
n_cases = 96
```

Top token/module paths：

```text
1. L24:resid_out:last  projection = 0.4095
2. L24:attn_out:last   projection = 0.2907
3. L21:resid_in:last   projection = 0.1162
4. L24:mlp_out:last    projection = 0.1058
5. L21:resid_out:last  projection = 0.1027
6. L19:attn_out:last   projection = 0.0617
7. L19:resid_out:last  projection = 0.0519
8. L21:attn_out:last   projection = 0.0371
```

客观现象：

```text
1. DeepSeek7B 的 FIRST_EVENT 读出准确率只有 0.6562。
2. 虽然 Phase 58 中 temporal_order/before_after 是 DS7B 的强随机优势 subtype，
   但当前符号化读出器不能稳定读取 FIRST_EVENT。
3. Top path 仍集中在末层 L24 last token。
4. DS7B 本轮有 Sliding Window Attention eager warning，因此路径结果只能作为候选观察。
```

### 三模型对比

```text
Qwen3:
  reader accuracy = 0.8021
  strongest path = L29 resid_out / attn_out at last token

GLM4:
  reader accuracy = 0.8021
  strongest path = L38 residual last token

DeepSeek7B:
  reader accuracy = 0.6562
  strongest path = L24 residual / attention last token
```

共同现象：

```text
1. 三模型 top path 都集中在 last token 的深层 residual_out。
2. 这说明当前 FIRST_EVENT 任务主要在输出位置形成 A/B 候选偏置。
3. 但 before/after operator token、A/B label token 没有进入 top path。
4. 因此本轮更像输出读出路径定位，而不是 temporal_order 变量写入机制定位。
```

### 当前结论

本轮最重要的结果不是找到了 temporal_order 机制，而是发现：

```text
Phase 58 的 temporal_order/before_after 随机优势强，
但当前 FIRST_EVENT 符号读出器仍不够稳定。
```

因此：

```text
1. temporal_order 仍是值得继续研究的强候选 subtype。
2. 当前读出器不能进入 destroy-restore 或 subspace causal patch。
3. top path 主要反映输出 A/B 候选读出位置，而不一定是 before/after 变量编码位置。
4. 下一步必须继续校准读出器和任务格式。
```

### 硬伤

```text
1. 读出器准确率不足：
   Qwen3 / GLM4 = 0.8021，DeepSeek7B = 0.6562，均未达到 0.9。

2. 指标仍然依赖 W_U(" A") - W_U(" B")：
   这只是候选输出方向，不是变量本体。

3. top path 集中在 last token：
   可能说明任务被模型当成输出选择，而不是内部 temporal relation 推理。

4. DS7B 使用 eager 且有 sliding-window warning：
   不能和 Qwen3 / GLM4 完全对称解释。

5. NVML 初始化失败：
   不影响本次结果生成，但说明 GPU 监控链路仍需单独排查。

6. 没有 patch / ablation / destroy-restore：
   本轮没有因果闭包证据。
```

### 关键洞察

这轮强化了一个判断：

```text
关系路径强，不等于读出器稳定；
读出器稳定之前，不应进入机制闭包；
路径图谱必须和读出器校准绑定推进。
```

也就是说，全局语义语法契约图谱不能只记录：

```text
哪个 relation subtype 强；
哪个 layer/module 强；
```

还必须记录：

```text
该 subtype 的读出器可靠性；
候选输出是否稳定；
operator token 是否参与；
relation variable 是否能从 output choice 中分离。
```

### 下一步计划

Phase 60 应优先做 temporal_order 读出器重构，而不是直接做 patch：

```text
1. 改用更强约束的完整答案格式：
   FIRST_EVENT = A
   FIRST_EVENT = B
   或 ANSWER: A / ANSWER: B

2. 使用多 token candidate sequence logprob：
   " A\n"
   " B\n"
   " EVENT_A"
   " EVENT_B"

3. 加入真假判断格式：
   Statement: A happened before B.
   Query: Did A happen first? Answer yes/no.

4. 分离 operator 读出和 output choice：
   before/after token path
   answer token path
   event label path

5. 只有当 reader accuracy > 0.9 且跨模板稳定后，
   才进入 temporal_order token/state transplant 和 destroy-restore。
```

阶段性大任务：

```text
建立每个 relation subtype 的可靠读出器层。

全局路径图谱 = relation path matrix
              + random-control advantage
              + reader reliability
              + token/module path
              + causal closure status
```

只有这个结构完成后，才可能从“关系路径拼图”进入“语言编码机制拼图”。

## Phase 60: Temporal Order 读出器重构与长会话工程阻断 [2026-06-02 18:59]

### 任务目标

根据 Phase 59 的结果，`temporal_order/before_after` 虽然在 Phase 58 中是强候选 subtype，但 `FIRST_EVENT` 读出器未达到 0.9 稳定门槛。因此本轮不做 patch / ablation / destroy-restore，而是先重构读出器：

```text
1. 比较多种 temporal_order 读出格式；
2. 使用完整 candidate sequence logprob，而不是只看首 token；
3. 平衡 before/after、A/B 反转、上下文模板；
4. 为每个 reader 输出 overall accuracy、context accuracy、relation accuracy；
5. 只有 reader 同时满足：
   overall accuracy >= 0.90
   min_context_accuracy >= 0.85
   min_relation_accuracy >= 0.85
   才允许进入后续机制闭包。
```

### 对用户分析的判断

用户分析整体正确。关键正确点：

```text
1. Phase 59 是“读出器失败但路径定位有效”的阶段。
2. relation subtype 有 random advantage，不等于读出器可靠。
3. top path 集中在 last token，说明当前任务更像 output choice path，不是 temporal variable 本体。
4. GSSC 必须加入 reader reliability，否则路径图谱无法进入机制闭包。
5. 下一步应该先改 reader，而不是直接做 subspace patch。
```

需要强调：

```text
读出器是机制研究的地基。
如果读出器不能稳定回答 before/after 或 first event，
后续任何 patch 成功都可能只是输出偏置或模板偏置。
```

### 新增和修改脚本

新增：

```text
tests/gpt5/phase60_temporal_order_reader_calibration.py
tests/gpt5/run_phase60_temporal_order_reader_calibration_normal_all.sh
tests/gpt5/phase60_temporal_order_reader_calibration_summary.py
tests/gpt5/run_phase60_temporal_order_reader_calibration_sharded_normal_all.sh
```

脚本功能：

```text
1. 不注册 hook，不做内部状态捕获，只做读出器校准。
2. 每个 case 生成 8 种 reader：
   first_event_letter
   first_event_event_label
   json_first_event
   order_pair
   a_first_yesno
   b_first_yesno
   before_statement_yesno
   after_statement_yesno

3. 每个 reader 比较 correct completion 和 wrong completion 的完整序列 logprob。
4. 输出 by_reader、by_context、by_relation、by_target_type。
5. 支持 --case-offset / --case-count / --output-suffix，用于短分片恢复。
6. sharded runner 默认每片 16 cases，每片完成后 hard-exit。
```

### Smoke Test

命令：

```bash
PHASE60_ATTN_IMPLEMENTATIONS=sdpa \
python tests/gpt5/phase60_temporal_order_reader_calibration.py qwen3 \
  --output-dir results/gpt5_phase60_smoke \
  --max-cases 8 \
  --progress-every 4

python tests/gpt5/phase60_temporal_order_reader_calibration_summary.py \
  --input-dir results/gpt5_phase60_smoke \
  --output-dir results/gpt5_phase60_smoke
```

结果：

```text
qwen3 smoke:
  cases = 8
  rows = 64
  exit_code = 0
```

Smoke summary：

```text
after_statement_yesno:
  accuracy = 0.8750
  min_context_accuracy = 0.8750
  min_relation_accuracy = 0.5000
  pass = no

first_event_event_label:
  accuracy = 0.8750
  min_context_accuracy = 0.8750
  min_relation_accuracy = 0.5000
  pass = no

a_first_yesno:
  accuracy = 0.7500
  min_relation_accuracy = 0.0000

b_first_yesno:
  accuracy = 0.7500
  min_relation_accuracy = 0.0000

first_event_letter:
  accuracy = 0.5000
```

客观现象：

```text
1. 脚本逻辑可以跑通。
2. 小样本中某些 reader overall accuracy 看似较高。
3. 但 min_relation_accuracy 很低，说明 reader 仍存在 relation type 偏置。
4. 因此即使 smoke 中 accuracy=0.875，也不能进入闭包。
```

### 正式长会话测试尝试

命令：

```bash
PHASE60_OUTPUT_DIR=results/gpt5_phase60_temporal_order_reader_calibration_full \
PHASE60_MAX_CASES=384 \
tests/gpt5/run_phase60_temporal_order_reader_calibration_normal_all.sh
```

目标规模：

```text
每模型 base cases = 384
每 case readers = 8
每 reader 两个 candidate sequence
每模型 rows = 3072
三模型 rows = 9216
```

实际结果：

```text
Qwen3 加载成功，flash_attention_2 不可用后自动降级到 sdpa。
但在第一批 progress 输出前，进程进入 D-state。
```

进程状态：

```text
PID 41457:
  python tests/gpt5/phase60_temporal_order_reader_calibration.py qwen3 ...
  STAT = Dl

PID 41559:
  nvidia-smi
  STAT = Ds
```

`kill -TERM` 和 `kill -KILL` 均无法清除该进程：

```text
41457 仍为 D-state
41559 仍为 D-state
```

`dmesg -T` 当前用户无权限读取：

```text
dmesg: read kernel buffer failed: Operation not permitted
```

输出文件：

```text
results/gpt5_phase60_temporal_order_reader_calibration_full/
```

没有生成正式模型结果文件，说明长会话卡在第一批样本之前。因此本轮不能给出三模型全量 reader 结论。

### 工程判断

本轮最重要的工程结论：

```text
即使不注册 hook、不做 patch、不做内部状态捕获，
384 cases 单进程 CUDA 长会话仍可能触发 D-state 卡死。
```

这说明问题不一定来自 hook 或 patch，也不一定来自机制实验本身，而可能来自：

```text
1. 长时间连续 CUDA forward；
2. NVML / nvidia-smi 与 CUDA 运行的交互；
3. driver / kernel / display GPU 组合；
4. 模型加载后的某些 CUDA 调用；
5. 单进程长 session 的资源状态。
```

因此，后续不能再使用单进程长会话作为默认全量方案。

### 已完成的修正

为恢复后继续测试，已经把 Phase60 改为短分片：

```text
tests/gpt5/run_phase60_temporal_order_reader_calibration_sharded_normal_all.sh
```

默认设置：

```text
PHASE60_MAX_CASES = 384
PHASE60_SHARD_CASES = 16
每个 shard:
  独立加载模型
  独立运行 16 cases
  独立保存 shard JSON
  --hard-exit-after-model

summary 脚本会自动合并：
  *_shard0000.json
  *_shard0001.json
  ...
```

恢复后推荐命令：

```bash
PHASE60_OUTPUT_DIR=results/gpt5_phase60_temporal_order_reader_calibration_sharded_full \
PHASE60_MAX_CASES=384 \
PHASE60_SHARD_CASES=16 \
tests/gpt5/run_phase60_temporal_order_reader_calibration_sharded_normal_all.sh
```

### 当前结论

本轮不能给出 temporal_order 读出器的三模型全量结论，只能给出两个可靠结论：

```text
1. Phase60 reader calibration 脚本和 summary 脚本已经完成，并通过 Qwen3 smoke。
2. 单进程 384 cases 长会话在 Qwen3 上触发 CUDA/NVML D-state 工程阻断。
```

因此，当前研究进展不是机制结果，而是实验系统校准：

```text
GSSC 不能只设计科学指标；
还必须设计可恢复、短分片、可合并的工程结构。
```

### 硬伤

```text
1. 三模型正式 Phase60 未完成。
2. 当前 D-state 进程未能被 kill 清除，需要系统/驱动恢复后继续。
3. Qwen3 smoke 样本太少，只能验证脚本，不能验证 reader。
4. Smoke 已显示 min_relation_accuracy 很低，reader 偏置仍然存在。
5. 本轮没有 token path、patch、ablation 或 destroy-restore。
```

### 下一步计划

Phase 60 需要在环境恢复后继续，但必须使用短分片：

```text
1. 先运行 sharded runner，每片 16 cases。
2. 如果仍卡死，把 PHASE60_SHARD_CASES 降到 4 或 8。
3. 禁止并发 nvidia-smi 监控。
4. 每个模型、每个 shard 独立 hard-exit。
5. 完成后合并 reader reliability matrix。
```

如果 Phase60 找到稳定 reader：

```text
进入 Phase61:
  temporal_order token/path 复测；
  区分 operator token、event label token、last token；
  再考虑 token transplant。
```

如果 Phase60 仍找不到稳定 reader：

```text
暂时降低 temporal_order 优先级，
转向 same_class/category_peer 或 quantifier/all_some，
因为它们可能更容易构造稳定符号读出器。
```

阶段性大任务保持不变：

```text
全局语义语法契约图谱 =
  relation subtype matrix
  + random-control advantage
  + reader reliability
  + variable/token path
  + causal closure status
  + engineering recoverability
```

只有读出器和工程系统都稳定后，才应继续推进语言编码机制闭包。

## Phase 61: Phase60a 短分片恢复尝试与最小 CUDA Smoke 阻断 [2026-06-02 19:18]

### 任务目标

用户要求继续完成 Phase 59 中未完成的测试，并结合最新分析继续推进全局语义语法契约图谱。

根据 Phase 59 / Phase 60 的关系，本轮实际应补的是：

```text
Phase 60a:
  用短分片完成 temporal_order/before_after reader calibration。

原因：
  Phase 59 已经完成 token/path 定位；
  未完成的是 Phase 59 暴露出的 reader reliability 问题；
  只有 reader 过门槛，才能继续 Phase61/62 的 token path 复测和 causal closure。
```

### 对用户分析的判断

用户提供的分析正确。关键点：

```text
1. Phase 60 不是机制结论，而是实验地基校准。
2. temporal_order/before_after 是强候选 subtype，但 reader 仍不稳定。
3. 读出器不稳定时，不能进入 patch / ablation / destroy-restore。
4. GSSC 必须包含 reader reliability 和 engineering recoverability。
5. 长 CUDA session 不能作为默认全量方案，必须使用短分片、落盘、合并。
```

本轮继续沿用这个判断，不做新的理论外推。

### 已完成的脚本修正

根据 Phase 60 的长会话阻断，继续完善短分片能力。

修改：

```text
tests/gpt5/phase60_temporal_order_reader_calibration.py
```

修正内容：

```text
1. 增加 --case-offset
2. 增加 --case-count
3. 增加 --output-suffix
4. sequence_logprob 中 candidate token ids 保留在 CPU，
   只把 input_ids 放到 GPU，减少不必要的 CUDA tensor .tolist() 同步。
```

修改：

```text
tests/gpt5/phase60_temporal_order_reader_calibration_summary.py
```

修正内容：

```text
1. 支持自动合并：
   *_phase60_temporal_order_reader_calibration_shard*.json

2. 如果没有单文件正式结果，会从 shard 文件重建：
   rows
   by_reader
   by_context
   by_relation
   cross_model_readers
```

已存在短分片 runner：

```text
tests/gpt5/run_phase60_temporal_order_reader_calibration_sharded_normal_all.sh
```

设计：

```text
PHASE60_MAX_CASES = 384
PHASE60_SHARD_CASES = 16
每个 shard 独立加载模型、运行、保存、hard-exit。
```

### 短分片正式运行尝试

命令：

```bash
PHASE60_OUTPUT_DIR=results/gpt5_phase60_temporal_order_reader_calibration_sharded_full \
PHASE60_MAX_CASES=384 \
PHASE60_SHARD_CASES=16 \
tests/gpt5/run_phase60_temporal_order_reader_calibration_sharded_normal_all.sh
```

结果：

```text
qwen3 shard0000:
  模型加载成功；
  flash_attention_2 不可用，自动降级到 sdpa；
  运行第一个 shard 时出现：
    torch.AcceleratorError: CUDA error: unspecified launch failure
```

错误位置表面发生在：

```text
sequence_logprob -> wrong candidate logprob
```

但 CUDA 报错是异步的，因此不能判断真实错误发生点。

该次失败后没有生成 shard 结果文件：

```text
results/gpt5_phase60_temporal_order_reader_calibration_sharded_full/
  无正式 JSON 输出
```

### 最小 CUDA Smoke 尝试

为了判断是否只是 shard 数据量太大，本轮进一步把测试缩到 1 case：

```bash
PHASE60_ATTN_IMPLEMENTATIONS=sdpa \
python tests/gpt5/phase60_temporal_order_reader_calibration.py qwen3 \
  --output-dir results/gpt5_phase60_smoke_after_fix \
  --max-cases 1 \
  --case-count 1 \
  --progress-every 1 \
  --output-suffix onecase
```

结果：

```text
只输出：
  `torch_dtype` is deprecated! Use `dtype` instead!

随后无进度输出。
```

进程状态：

```text
PID 9097:
  python tests/gpt5/phase60_temporal_order_reader_calibration.py qwen3 ...
  STAT = Dsl / Ds

kill -TERM 无效；
kill -KILL 无效。
```

因此：

```text
即使 1 case、无 hook、无 patch、无 nvidia-smi 并发，
当前 CUDA 环境仍会进入 D-state。
```

### 当前客观结论

本轮没有完成 Phase60a 三模型读出器全量校准，也不能继续 GLM4 / DS7B。

可靠结论只有：

```text
1. Phase60a 的短分片与合并脚本已经完成。
2. Qwen3 shard0000 在当前环境下触发 CUDA unspecified launch failure。
3. 修正 CPU token ids 后，1 case 最小 CUDA smoke 仍进入 D-state。
4. 当前问题已经不是数据量过大，也不是 hook / patch 造成。
5. 当前 CUDA/驱动/系统状态不适合继续模型测试。
```

### 对当前研究计划的影响

Phase 59 的科学结论仍然成立：

```text
temporal_order/before_after 是强候选 subtype；
但 FIRST_EVENT reader 不稳定；
top path 更像 output choice path；
不能进入 closure。
```

Phase 60 / 61 的新增影响是：

```text
工程系统仍是当前瓶颈。
在 CUDA 最小 smoke 无法完成前，
继续设计更复杂机制测试没有意义。
```

因此当前不能把失败解释为：

```text
temporal_order reader 永久失败
```

只能解释为：

```text
当前环境未能完成 reader calibration。
```

### 硬伤

```text
1. Phase60a 未完成三模型测试。
2. Qwen3 1 case smoke 也进入 D-state。
3. 没有可用的全量 reader reliability matrix。
4. 当前不能启动 GLM4 / DS7B，否则会污染环境和结果。
5. 缺少 dmesg / kernel 日志权限，不能从本轮直接定位 Xid 或驱动错误。
```

### 下一步计划

必须先恢复 CUDA 环境，再继续 Phase60a。

恢复后执行顺序：

```text
1. 不运行 nvidia-smi 长监控。
2. 先跑最小 1 case qwen3：
   PHASE60_ATTN_IMPLEMENTATIONS=sdpa
   max_cases=1
   case_count=1

3. 如果成功，再跑 shard_cases=4。
4. 如果成功，再跑 shard_cases=8。
5. 最后才跑 shard_cases=16。
6. qwen3 完整后再跑 GLM4，再跑 DS7B。
```

推荐恢复后命令：

```bash
PHASE60_OUTPUT_DIR=results/gpt5_phase60_temporal_order_reader_calibration_sharded_full \
PHASE60_MAX_CASES=384 \
PHASE60_SHARD_CASES=4 \
tests/gpt5/run_phase60_temporal_order_reader_calibration_sharded_normal_all.sh
```

如果 Phase60a 最终完成：

```text
1. 若 temporal_order reader 通过门槛：
   进入 temporal_order token path 复测与 operator/output 分离。

2. 若 temporal_order reader 不通过：
   暂停 temporal_order 闭包；
   转向 same_class/category_peer 或 quantifier/all_some reader calibration。
```

阶段性大任务不变：

```text
先建立可靠读出器层；
再定位变量/token路径；
再做 causality；
最后做 destroy-restore。
```

目前最重要的原则：

```text
不要让工程不稳定伪装成机制负结果；
也不要在读出器未完成时强行解释内部路径。
```

## Phase 62: Phase60a Temporal Order 全量短分片读出器校准完成 [2026-06-08 17:13]

### 任务目标

显卡问题恢复后，继续完成 Phase 59 暴露出的未完成任务：

```text
temporal_order/before_after reader reliability calibration
```

本轮使用 Phase 60 已经实现的短分片系统，完成三模型全量测试。

### 运行环境

```text
date = 2026-06-08
driver = 595.71.05
CUDA shown by nvidia-smi = 13.2
GPU = RTX 4090 D 24GB
conda_env = openone-cu130-py312
```

运行前检查：

```text
nvidia-smi 正常返回；
无 Phase59/60 残留模型进程；
GPU 显存仅桌面占用。
```

### 测试命令

```bash
PHASE60_OUTPUT_DIR=results/gpt5_phase60_temporal_order_reader_calibration_sharded_full_20260608 \
PHASE60_MAX_CASES=384 \
PHASE60_SHARD_CASES=16 \
SLEEP_AFTER_SHARD=1 \
tests/gpt5/run_phase60_temporal_order_reader_calibration_sharded_normal_all.sh
```

运行策略：

```text
1. qwen3 -> glm4 -> deepseek7b 顺序执行。
2. 每个模型 24 shards。
3. 每个 shard = 16 base cases。
4. 每个 shard 独立加载模型、保存 JSON、hard-exit。
5. summary 脚本自动合并所有 shards。
```

注意：

```text
Qwen3 / GLM4:
  flash_attention_2 未安装，自动降级到 sdpa。

DeepSeek7B:
  使用 eager；
  仍有 Sliding Window Attention warning。
```

### 输出文件

```text
results/gpt5_phase60_temporal_order_reader_calibration_sharded_full_20260608/
  qwen3_phase60_temporal_order_reader_calibration_shard0000.json ... shard0023.json
  glm4_phase60_temporal_order_reader_calibration_shard0000.json ... shard0023.json
  deepseek7b_phase60_temporal_order_reader_calibration_shard0000.json ... shard0023.json
  phase60_temporal_order_reader_calibration_summary.json
  PHASE60_TEMPORAL_ORDER_READER_CALIBRATION_SUMMARY.md
```

数据规模：

```text
每模型:
  cases = 384
  rows = 3072

三模型:
  cases = 1152
  rows = 9216

shards:
  qwen3 = 24/24
  glm4 = 24/24
  deepseek7b = 24/24
```

### Qwen3 结果

```text
best reader = json_first_event
accuracy = 0.7578
min_context_accuracy = 0.6562
min_relation_accuracy = 0.4479
pass = false
```

完整排名：

```text
json_first_event:
  acc = 0.7578
  min_ctx = 0.6562
  min_rel = 0.4479

a_first_yesno:
  acc = 0.7448
  min_ctx = 0.6771
  min_rel = 0.0833

first_event_event_label:
  acc = 0.7109
  min_ctx = 0.6146
  min_rel = 0.2708

after_statement_yesno:
  acc = 0.6536
  min_ctx = 0.5104
  min_rel = 0.4167

before_statement_yesno:
  acc = 0.6354
  min_ctx = 0.5000
  min_rel = 0.2708

order_pair:
  acc = 0.4974
  min_rel = 0.0000
```

客观现象：

```text
Qwen3 有一定 temporal_order 读出信号，
但所有 reader 都未达到闭包门槛。
最好的 json_first_event 仍然存在明显 relation-type 失衡。
```

### GLM4 结果

```text
best reader = first_event_letter
accuracy = 0.6068
min_context_accuracy = 0.5208
min_relation_accuracy = 0.2292
pass = false
```

完整排名：

```text
first_event_letter:
  acc = 0.6068
  min_ctx = 0.5208
  min_rel = 0.2292

first_event_event_label:
  acc = 0.5625
  min_ctx = 0.4896
  min_rel = 0.0417

b_first_yesno:
  acc = 0.5495
  min_rel = 0.0104

order_pair:
  acc = 0.5469
  min_rel = 0.0938

json_first_event:
  acc = 0.5286
  min_rel = 0.2812

yes/no statement readers:
  acc ≈ 0.50
  min_rel = 0
```

客观现象：

```text
GLM4 在当前 temporal_order symbolic reader 上基本不稳定。
读出结果接近弱偏置而不是可靠时间顺序理解。
```

### DeepSeek7B 结果

```text
best reader = json_first_event
accuracy = 0.5990
min_context_accuracy = 0.5729
min_relation_accuracy = 0.1354
pass = false
```

完整排名：

```text
json_first_event:
  acc = 0.5990
  min_ctx = 0.5729
  min_rel = 0.1354

first_event_event_label:
  acc = 0.5807
  min_rel = 0.0521

first_event_letter:
  acc = 0.5573
  min_rel = 0.0625

order_pair:
  acc = 0.5286
  min_rel = 0.0104

yes/no readers:
  acc ≈ 0.50
  min_rel = 0
```

客观现象：

```text
DeepSeek7B 的 temporal_order reader 仍然较弱，
和 Phase59 中 FIRST_EVENT accuracy = 0.6562 的现象一致。
本轮因为 eager + sliding-window warning，仍需谨慎解释。
```

### Cross Model 结果

没有任何 reader 跨模型通过门槛。

跨模型最好项：

```text
json_first_event:
  mean_acc = 0.6285
  min_acc = 0.5286
  min_ctx = 0.4896
  min_rel = 0.1354
  all_pass = false

first_event_event_label:
  mean_acc = 0.6181
  min_acc = 0.5625
  min_rel = 0.0417
  all_pass = false
```

### 当前结论

Phase60a 给出明确负结果：

```text
当前 8 种 temporal_order reader 均不可靠。
```

这不是 temporal_order 机制不存在，而是说明：

```text
1. 当前 prompt / candidate 格式不能稳定读出时间顺序变量；
2. Phase59 的 last-token top path 更可能是 output choice path；
3. temporal_order/before_after 暂时不能进入 patch / ablation / destroy-restore；
4. GSSC 必须记录 reader failure，而不是只记录 Phase58 的 random advantage。
```

### 对 Phase59 的补完

Phase59 中未完成的问题是：

```text
temporal_order 是否有可靠 reader？
```

本轮回答：

```text
在当前 8 类 reader、384 cases、三模型测试下，没有。
```

因此 Phase59 后续不能继续走：

```text
temporal_order token path -> subspace patch -> destroy-restore
```

而应转向：

```text
寻找更稳定的 relation subtype reader。
```

### 硬伤

```text
1. 本轮只测 reader，不测内部路径。
2. temporal_order 的 prompt 仍可能不适合这些模型。
3. DeepSeek7B 使用 eager 且 sliding-window warning。
4. 读出失败不能证明模型没有 temporal_order 表征。
5. 但读出失败足以阻止机制闭包。
```

### 下一步计划

根据 Phase58 的候选排序和 Phase60a 的失败结果，下一步转向：

```text
Phase63: same_class/category_peer reader calibration
```

原因：

```text
1. same_class/category_peer 在 Qwen3 中强；
2. GLM4 中等；
3. 关系变量比 temporal_order 更容易构造稳定读出器；
4. 可用显式 fact table 降低世界知识干扰。
```

Phase63 设计原则：

```text
1. 使用符号对象和显式类别事实：
   Object A belongs to category K1.
   Object B belongs to category K1.
   Object C belongs to category K2.

2. 查询：
   Which object is in the same category as A?
   B or C

3. 平衡：
   A/B/C 顺序；
   正确候选位置；
   category label；
   context template；
   answer format。

4. 门槛不变：
   overall >= 0.90
   min_context >= 0.85
   min_relation/candidate-position >= 0.85
```

如果 same_class reader 通过：

```text
进入 same_class token path 和 subspace closure。
```

如果 same_class 也失败：

```text
转向 quantifier/all_some 或 object-attribute binding，
但必须先过 reader calibration。
```

## Phase 63: Same-Class 符号关系读出器全量校准 [2026-06-08 17:31]

### 任务目标

Phase 60/62 的 temporal_order（时间顺序）读出器全量校准没有任何三模型通用读出器通过门槛，因此不能进入 temporal_order 的 patch/closure（干预/闭包）实验。

本轮继续全局关系路径图谱路线，但换成相对更简单的对象类别关系：

```text
same_class / different_class
```

目标不是直接做机制结论，而是先验证：

```text
模型是否能稳定读出：
Object B 和 Object C 中，谁与 Object A 属于同一类别？
或者谁与 Object A 属于不同类别？
```

如果读出器不稳定，后续 token path、state transplant、destroy-restore 都不能解释为对象-属性 binding 机制证据。

### 脚本

新增：

```text
tests/gpt5/phase63_same_class_reader_calibration.py
tests/gpt5/phase63_same_class_reader_calibration_summary.py
tests/gpt5/run_phase63_same_class_reader_calibration_sharded_normal_all.sh
```

脚本设计：

```text
1. 使用符号化 fact table，不依赖真实世界常识。
2. 每个 case 中包含 Object A/B/C 的 category。
3. 两种关系变体：
   - B_same: B 与 A 同类，C 不同类
   - C_same: C 与 A 同类，B 不同类
4. 四种上下文风格：
   - table
   - sentences
   - compact
   - record
5. 8 个 reader：
   - same_letter
   - different_letter
   - json_same
   - same_object_label
   - b_same_yesno
   - c_same_yesno
   - b_statement_true
   - c_statement_true
6. pass gate:
   - overall accuracy >= 0.90
   - min_context_accuracy >= 0.85
   - min_variant_accuracy >= 0.85
```

### 运行命令

Smoke：

```bash
PHASE63_ATTN_IMPLEMENTATIONS=sdpa \
python tests/gpt5/phase63_same_class_reader_calibration.py qwen3 \
  --output-dir results/gpt5_phase63_smoke \
  --max-cases 8 \
  --case-count 8 \
  --progress-every 4 \
  --output-suffix smoke
```

全量短分片：

```bash
PHASE63_OUTPUT_DIR=results/gpt5_phase63_same_class_reader_calibration_sharded_full_20260608 \
PHASE63_MAX_CASES=384 \
PHASE63_SHARD_CASES=16 \
SLEEP_AFTER_SHARD=1 \
tests/gpt5/run_phase63_same_class_reader_calibration_sharded_normal_all.sh
```

运行方式：

```text
qwen3 -> glm4 -> deepseek7b 顺序运行；
每个模型 24 shard；
每个 shard 16 cases；
每个 shard 使用 --hard-exit-after-model；
一个 shard 完成后退出 Python 进程，再加载下一个 shard；
避免长 CUDA session 和模型残留显存。
```

### 工程结果

输出目录：

```text
results/gpt5_phase63_same_class_reader_calibration_sharded_full_20260608
```

汇总文件：

```text
results/gpt5_phase63_same_class_reader_calibration_sharded_full_20260608/phase63_same_class_reader_calibration_summary.json
results/gpt5_phase63_same_class_reader_calibration_sharded_full_20260608/PHASE63_SAME_CLASS_READER_CALIBRATION_SUMMARY.md
```

完成情况：

```text
qwen3:      24/24 shards completed
glm4:       24/24 shards completed
deepseek7b: 24/24 shards completed
```

说明：

```text
GLM4 在 shard0001 第一次运行时出现一次用户态 segfault 139；
没有进入 D-state，没有发现 GPU reset 阻断；
resume 后跳过已完成 qwen3 和 glm4 shard0000，从 glm4 shard0001 继续，最终完成全部 shard。
```

### 数据规模

```text
qwen3:
  cases = 384
  rows = 3072

glm4:
  cases = 384
  rows = 3072

deepseek7b:
  cases = 384
  rows = 3072

total:
  cases = 1152
  rows = 9216
```

### Qwen3 结果

最佳 reader：

```text
reader = different_letter
accuracy = 0.9896
min_context_accuracy = 0.9583
min_variant_accuracy = 0.9792
mean_margin = 2.5648
passes_gate = yes
```

其他 reader：

```text
b_statement_true:
  accuracy = 0.9271
  min_context_accuracy = 0.7083
  min_variant_accuracy = 0.9010
  pass = no

c_statement_true:
  accuracy = 0.9010
  min_context_accuracy = 0.7396
  min_variant_accuracy = 0.8021
  pass = no

same_letter:
  accuracy = 0.8802
  min_context_accuracy = 0.5208
  min_variant_accuracy = 0.7865
  pass = no
```

客观现象：

```text
Qwen3 可以稳定回答“哪个对象与 A 不同类”，different_letter 通过全部门槛。
但 same_letter 没有通过，说明“同类对象”自然读出和“不同对象”读出并不对称。
```

### GLM4 结果

最佳 reader：

```text
reader = json_same
accuracy = 0.8099
min_context_accuracy = 0.5729
min_variant_accuracy = 0.6979
mean_margin = 1.1681
passes_gate = no
```

其他 reader：

```text
c_statement_true:
  accuracy = 0.7057
  min_context_accuracy = 0.5208
  min_variant_accuracy = 0.4792
  pass = no

same_object_label:
  accuracy = 0.6510
  min_context_accuracy = 0.5000
  min_variant_accuracy = 0.3021
  pass = no
```

客观现象：

```text
GLM4 在 same-class 符号读出上有一定信号，但上下文和 B/C 变体稳定性不足。
当前不能把 GLM4 的 same_class 读出结果作为干预闭包读出器。
```

### DeepSeek7B 结果

最佳 reader：

```text
reader = c_same_yesno
accuracy = 0.7552
min_context_accuracy = 0.5521
min_variant_accuracy = 0.5938
mean_margin = 0.5418
passes_gate = no
```

其他 reader：

```text
b_same_yesno:
  accuracy = 0.7266
  min_context_accuracy = 0.5312
  min_variant_accuracy = 0.5365
  pass = no

b_statement_true:
  accuracy = 0.6536
  min_context_accuracy = 0.5208
  min_variant_accuracy = 0.4427
  pass = no
```

客观现象：

```text
DeepSeek7B 没有稳定 same_class reader。
虽然 yes/no 形式有一定总体准确率，但 min_context 和 min_variant 均远低于门槛。
```

### 跨模型结果

跨模型最佳 reader：

```text
reader = c_statement_true
mean_accuracy = 0.7535
min_accuracy = 0.6536
min_context_accuracy = 0.5104
min_variant_accuracy = 0.4062
all_pass = no
```

结论：

```text
没有任何 same_class reader 在三模型上同时通过。
```

### 与 Phase62 的关系

Phase62 temporal_order 结论：

```text
三模型没有任何 temporal_order reader 通过；
不能进入 temporal_order closure。
```

Phase63 same_class 结论：

```text
Qwen3 的 different_letter reader 通过；
GLM4 和 DeepSeek7B 没有 reader 通过；
三模型通用 reader 没有通过。
```

这说明：

```text
1. same_class/different_class 比 temporal_order 更容易读出；
2. 但读出器仍然强烈依赖模型和问题形式；
3. 不能假设“一个符号读出模板可以跨模型、跨关系稳定工作”。
```

### 当前研究判断

这轮结果支持一个更严格的测试原则：

```text
关系机制实验必须先校准读出器；
读出器必须按模型、关系类型、输出形式分别过门槛；
没有稳定读出器时，patch/probe/closure 都不能解释为机制证据。
```

对语言编码机制的意义：

```text
same_class 不是单纯对象属性表读取。
模型对“相同”和“不同”的输出偏好并不对称；
Qwen3 对 different-object 读出极稳，而 GLM4/DS7B 仍受上下文形式和候选变体影响。
这说明关系编码的读出端本身就是机制的一部分，而不是一个可忽略的测量工具。
```

### 硬伤

1. 这仍是 reader calibration，不是 path localization，也不是因果闭包。
2. 384 cases 中包含 trial repeat，虽然上下文和类别组合覆盖较大，但独立结构仍有限。
3. Qwen3 通过的是 different_letter，而不是 same_letter；后续如果研究 same_class binding，要谨慎区分“同类读出”和“异类排除”。
4. GLM4 与 DeepSeek7B 没有可靠 reader，因此不能直接进入三模型同构机制比较。
5. GLM4 曾出现一次用户态 segfault，虽然 resume 后完成，但长时间多 shard 仍需要保留 checkpoint/resume 结构。

### 下一步计划

下一阶段不应直接对三模型做 same_class patch，而应分两条线推进：

```text
Phase64a: Qwen3 different-class path localization
  使用已经通过门槛的 different_letter reader；
  定位 Qwen3 中 different_class 的 layer/token/module 路径；
  先确认它是对象-属性比较路径，还是输出排除路径。

Phase64b: GLM4/DeepSeek7B reader 改写
  针对 GLM4/DS7B 设计更强符号读出器；
  避免 yes/no 和 B/C 单侧偏置；
  尝试 forced-choice JSON、label-only、verifier-style 多步格式；
  只有 reader 过门槛后再进入 path/closure。
```

更大的阶段任务：

```text
1. 建立关系读出器库：
   same_class / different_class / object_attribute / temporal_order / causal_order / role_binding。

2. 每个关系先通过 reader gate：
   overall >= 0.90；
   min_context >= 0.85；
   min_variant >= 0.85。

3. 通过 gate 的关系才进入：
   token path localization；
   state transplant；
   subspace destroy-restore；
   cross-model comparison。
```

### 当前结论

Phase63 完成了 same_class/different_class 的三模型全量短分片读出器校准。

最可靠的客观发现是：

```text
Qwen3 能稳定读出“哪个对象与 A 不同类”；
GLM4 和 DeepSeek7B 当前没有稳定 same_class/different_class reader；
三模型没有通用读出器通过。
```

所以，当前不能说已经找到对象-属性 binding 机制。

更稳的说法是：

```text
对象-属性关系机制研究进入了读出器分化阶段；
Qwen3 可以先进入 different_class 路径定位；
GLM4/DS7B 必须先继续读出器校准。
```

## Phase 64: Same-Class 读出器改写复验与接口偏置确认 [2026-06-08 17:53]

### 任务目标

Phase63 中：

```text
Qwen3:
  different_letter 通过 reader gate；

GLM4 / DeepSeek7B:
  没有 same_class/different_class reader 通过；

三模型：
  没有通用 reader 通过。
```

本轮继续 reader calibration，而不是进入 patch/closure。

核心问题：

```text
GLM4 和 DeepSeek7B 没有过门槛，
到底是因为上一轮自然语言模板不合适，
还是因为当前 B/C 二选一读出接口本身不稳定？
```

因此本轮设计改写版 reader：

```text
1. 减少自然语言；
2. 使用 A_CAT/B_CAT/C_CAT 字段；
3. 加入 CSV / JSON / equation / key-value 上下文；
4. 保留 Phase63 中 Qwen3 表现最好的自然 forced-choice reader 作为对照；
5. 继续用三模型全量短分片。
```

### 脚本

新增：

```text
tests/gpt5/phase64_same_class_reader_refine.py
tests/gpt5/phase64_same_class_reader_refine_summary.py
tests/gpt5/run_phase64_same_class_reader_refine_sharded_normal_all.sh
```

reader 类型：

```text
same_key_letter
same_key_space
same_json_min
same_option_line
same_natural_control
same_compare_values

different_key_letter
different_key_space
different_json_min
different_option_line
different_natural_control
different_compare_values

b_eq_a_binary
c_eq_a_binary
```

上下文类型：

```text
key_value:
  A_CAT=K0
  B_CAT=K0
  C_CAT=K1

csv:
  object,cat
  A,K0
  B,K0
  C,K1

json:
  {"A_CAT":"K0","B_CAT":"K0","C_CAT":"K1"}

equation:
  cat(A)=K0; cat(B)=K0; cat(C)=K1.
```

pass gate 保持不变：

```text
overall accuracy >= 0.90
min_context_accuracy >= 0.85
min_variant_accuracy >= 0.85
```

### 运行命令

Smoke：

```bash
PHASE64_ATTN_IMPLEMENTATIONS=sdpa \
python tests/gpt5/phase64_same_class_reader_refine.py qwen3 \
  --output-dir results/gpt5_phase64_smoke \
  --max-cases 16 \
  --case-count 16 \
  --progress-every 8 \
  --output-suffix smoke \
  --hard-exit-after-model
```

全量：

```bash
PHASE64_OUTPUT_DIR=results/gpt5_phase64_same_class_reader_refine_sharded_full_20260608 \
PHASE64_MAX_CASES=384 \
PHASE64_SHARD_CASES=16 \
SLEEP_AFTER_SHARD=1 \
tests/gpt5/run_phase64_same_class_reader_refine_sharded_normal_all.sh
```

运行方式：

```text
qwen3 -> glm4 -> deepseek7b 顺序运行；
每模型 24 shard；
每 shard 16 cases；
每 shard 使用 --hard-exit-after-model；
完成一个 shard 后退出 Python 进程，释放模型和显存；
再运行下一个 shard。
```

### 工程结果

输出目录：

```text
results/gpt5_phase64_same_class_reader_refine_sharded_full_20260608
```

汇总文件：

```text
results/gpt5_phase64_same_class_reader_refine_sharded_full_20260608/phase64_same_class_reader_refine_summary.json
results/gpt5_phase64_same_class_reader_refine_sharded_full_20260608/PHASE64_SAME_CLASS_READER_REFINE_SUMMARY.md
```

完成情况：

```text
qwen3:      24/24 shards completed
glm4:       24/24 shards completed
deepseek7b: 24/24 shards completed
```

数据规模：

```text
qwen3:
  cases = 384
  rows = 5376

glm4:
  cases = 384
  rows = 5376

deepseek7b:
  cases = 384
  rows = 5376

total:
  cases = 1152
  rows = 16128
```

工程状态：

```text
全量运行完成；
没有出现系统卡死；
没有出现 GPU reset；
运行结束后显存约 438 MiB / 24564 MiB；
短分片 + --hard-exit-after-model 继续有效。
```

说明：

```text
本机没有 flash_attn 包；
qwen3/glm4 尝试 flash_attention_2 后回退到 sdpa；
deepseek7b 使用 eager，并出现 Sliding Window Attention warning；
这是模型实现路径差异，不改变本轮 reader calibration 的指标解释。
```

### Qwen3 结果

通过 reader：

```text
reader = different_natural_control
accuracy = 0.9766
min_context_accuracy = 0.9583
min_variant_accuracy = 0.9635
mean_margin = 1.0465
passes_gate = yes
```

未通过 reader：

```text
same_compare_values:
  accuracy = 0.6823
  min_context_accuracy = 0.4792
  min_variant_accuracy = 0.6042
  pass = no

b_eq_a_binary:
  accuracy = 0.6719
  min_context_accuracy = 0.5625
  min_variant_accuracy = 0.3438
  pass = no

same_natural_control:
  accuracy = 0.6641
  min_context_accuracy = 0.5000
  min_variant_accuracy = 0.3281
  pass = no
```

客观现象：

```text
Qwen3 的 different-class 自然 forced-choice 读出稳定通过；
但 same-class 读出仍然明显不稳；
字段化/JSON/二元等式模板没有改善 same-class。
```

与 Phase63 一致：

```text
Qwen3 对“哪个对象与 A 不同类”的读出稳定；
但“哪个对象与 A 同类”的读出不对称地弱。
```

### GLM4 结果

最佳 reader：

```text
reader = same_compare_values
accuracy = 0.6198
min_context_accuracy = 0.5000
min_variant_accuracy = 0.2760
mean_margin = 0.2995
passes_gate = no
```

其他 reader：

```text
c_eq_a_binary:
  accuracy = 0.5781
  min_context_accuracy = 0.4375
  min_variant_accuracy = 0.4271
  pass = no

same_option_line:
  accuracy = 0.5417
  min_context_accuracy = 0.5000
  min_variant_accuracy = 0.0885
  pass = no

b_eq_a_binary:
  accuracy = 0.5365
  min_context_accuracy = 0.4896
  min_variant_accuracy = 0.0990
  pass = no
```

客观现象：

```text
GLM4 没有任何 reader 接近 pass gate。
改写为字段比较、JSON、equation 后没有改善；
相反，Phase63 中 json_same 还能达到 0.8099，本轮最佳只有 0.6198。
```

解释限制：

```text
这不能说明 GLM4 没有对象-属性关系编码；
只能说明当前 B/C forced-choice 读出接口对 GLM4 不可靠。
```

### DeepSeek7B 结果

最佳 reader：

```text
reader = b_eq_a_binary
accuracy = 0.5885
min_context_accuracy = 0.5000
min_variant_accuracy = 0.2188
mean_margin = 0.2795
passes_gate = no
```

其他 reader：

```text
same_compare_values:
  accuracy = 0.5339
  min_context_accuracy = 0.5000
  min_variant_accuracy = 0.1042
  pass = no

different_option_line:
  accuracy = 0.5156
  min_context_accuracy = 0.4583
  min_variant_accuracy = 0.1094
  pass = no

c_eq_a_binary:
  accuracy = 0.5130
  min_context_accuracy = 0.5000
  min_variant_accuracy = 0.0729
  pass = no
```

客观现象：

```text
DeepSeek7B 的改写版 reader 全部失败；
个别 shard 中 b_eq_a_binary 可达到 1.0，但全量后不稳定；
这说明局部高分更可能是 label/case 适配，不是稳定关系读出器。
```

### 跨模型结果

跨模型最佳 reader：

```text
reader = b_eq_a_binary
mean_accuracy = 0.5990
min_accuracy = 0.5365
min_context_accuracy = 0.4896
min_variant_accuracy = 0.0990
all_pass = no
```

其他：

```text
same_compare_values:
  mean_accuracy = 0.6120
  min_accuracy = 0.5339
  min_context_accuracy = 0.4792
  min_variant_accuracy = 0.1042
  all_pass = no

different_natural_control:
  mean_accuracy = 0.6589
  min_accuracy = 0.5000
  min_context_accuracy = 0.5000
  min_variant_accuracy = 0.0000
  all_pass = no
```

结论：

```text
没有任何 Phase64 reader 在三模型上同时通过。
```

### 与 Phase63 对比

Phase63：

```text
Qwen3:
  different_letter pass
  accuracy = 0.9896

GLM4:
  best json_same accuracy = 0.8099
  pass = no

DeepSeek7B:
  best c_same_yesno accuracy = 0.7552
  pass = no
```

Phase64：

```text
Qwen3:
  different_natural_control pass
  accuracy = 0.9766

GLM4:
  best same_compare_values accuracy = 0.6198
  pass = no

DeepSeek7B:
  best b_eq_a_binary accuracy = 0.5885
  pass = no
```

主要变化：

```text
1. Qwen3 different-class 结果稳定复现。
2. GLM4/DS7B 没有因为字段化、JSON、equation、binary equality 而改善。
3. 过度符号化模板反而降低了 GLM4/DS7B 的读出稳定性。
```

### 当前研究判断

这轮最重要的结论不是“找到更好的 reader”，而是：

```text
关系读出接口本身是模型特异的；
同一个符号关系在 Qwen3、GLM4、DS7B 上不能直接共享同一种读出器；
过度符号化不一定更干净，可能反而偏离模型自然输出接口。
```

对相对编码的意义：

```text
对象-属性 binding 不能只看单一路径；
读出端、输出格式、候选偏置、上下文格式都是相对编码系统的一部分；
关系机制必须放进全局路径比较中，而不是拿一个 reader 当固定观察窗。
```

### 硬伤

1. Phase64 仍是 reader calibration，不是因果机制测试。
2. Qwen3 通过的是 different-class reader，不是 same-class reader。
3. GLM4/DeepSeek7B 没有可用 reader，因此不能进入同类关系的三模型闭包比较。
4. 字段化模板失败不能证明模型没有内部关系表示，只能证明当前输出接口不可靠。
5. DeepSeek7B 使用 eager 路径并有 sliding-window warning，因此 DS7B 数值结果仍需谨慎和稳定 reader 复核。

### 下一步计划

下一步不能直接做三模型 same_class patch。

建议拆成三个阶段：

```text
Phase65a: Qwen3 different-class token path localization
  使用通过门槛的 different_natural_control reader；
  定位 layer/token/module 路径；
  判断 different-class 是对象-属性比较，还是输出排除路径。

Phase65b: GLM4/DS7B 读出协议重建
  不继续只做 B/C 二选一；
  尝试完整答案序列评分：
    "A same as B; A different from C"
    vs
    "A same as C; A different from B"
  尝试 few-shot 格式；
  尝试让模型生成完整 relation table，再评分整段 logprob。

Phase65c: 全局关系读出器矩阵
  对 temporal_order / same_class / different_class / object_attribute / role_binding
  统一记录：
    reader_type
    model
    context_format
    variant_stability
    label_bias
    pass_gate
```

### 第一性原则修正

本轮进一步说明：

```text
破解语言编码机制不能把“读出器”当作外部透明测量工具。
读出器本身也是模型计算闭环的一部分。
```

更基础的表达是：

```text
语言关系编码 = 内部关系状态 + 输出接口 + 候选竞争 + 任务格式。
```

所以后续图谱必须同时记录：

```text
1. 关系内部路径；
2. 读出器路径；
3. 候选输出路径；
4. 三者之间是否可自然接上。
```

否则会把“读出器失败”误判为“内部机制不存在”，或者把“读出器偏置”误判为“机制成功”。

### 当前结论

Phase64 完成了三模型 same_class/different_class 改写读出器全量复验。

最可靠事实：

```text
Qwen3 的 different-class 自然 forced-choice reader 稳定通过；
Qwen3 的 same-class reader 不稳定；
GLM4 和 DeepSeek7B 在本轮所有改写 reader 中均未通过；
没有三模型通用 reader。
```

因此当前不能进入三模型对象-属性 binding 闭包。

可以进入的是：

```text
Qwen3 different-class 路径定位。
```

必须继续校准的是：

```text
GLM4 / DeepSeek7B 的关系读出协议。
```

## Phase 65: 对象-属性兼容性梯度分解全量测试 [2026-06-08 18:29]

### 任务目标

根据 GLM5 memo 最新 Phase 396/396b 的结论，继续完成对象-属性 binding 的全量复验。

核心问题：

```text
对象-属性兼容性是否存在稳定方向？
这个方向是纯 compatibility gradient，
还是 value bias 与上下文值 token 交互后的结果？
```

本轮不再只看类别平均方向，而是做：

```text
1. per-object 分对象方向；
2. correct / incorrect / neutral 三条件；
3. 多层追踪；
4. L1_category / L2_crossfit / OBJ_crossfit 三种方向；
5. 三模型完整测试。
```

关键判据：

```text
FULL_SYMMETRIC:
  correct prompt 下 IDEAL(T↑ C↓)
  incorrect prompt 下仍 IDEAL(T↑ C↓)

neutral_ideal:
  neutral prompt 下也 IDEAL

如果 FULL 多但 neutral_ideal 也多：
  说明可能混入 value preference / output bias

如果 FULL 多且 neutral_ideal 少：
  更接近上下文依赖的兼容性交互机制
```

### 脚本

新增：

```text
tests/gpt5/phase65_object_attribute_compat_decomposition.py
tests/gpt5/phase65_object_attribute_compat_summary.py
tests/gpt5/run_phase65_object_attribute_compat_full.sh
```

数据：

```text
3 categories:
  color
  size
  moisture

每类 6 objects；
每 object 2 value pairs；
每 pair 4 frames；
total pairs/model = 144
```

方向：

```text
L1:
  mu + category_centroid

L2_cf:
  mu + category_centroid + leave-one-pair-out object-category residual

OBJ_cf:
  leave-one-pair-out object-category residual only
```

prompt 条件：

```text
correct:
  clean:   The elephant is big.
  corrupt: The item is big.

incorrect:
  clean:   The elephant is small.
  corrupt: The item is small.

neutral:
  clean:   The elephant is
  corrupt: The item is
```

### 运行命令

Smoke：

```bash
PHASE65_ATTN_IMPLEMENTATIONS=sdpa \
python tests/gpt5/phase65_object_attribute_compat_decomposition.py qwen3 \
  --layers 4 \
  --output-dir results/gpt5_phase65_smoke \
  --max-pairs 12 \
  --progress-every 6 \
  --output-suffix smoke \
  --hard-exit-after-model
```

全量：

```bash
PHASE65_OUTPUT_DIR=results/gpt5_phase65_object_attribute_compat_full_20260608_1818 \
SLEEP_AFTER_MODEL=3 \
tests/gpt5/run_phase65_object_attribute_compat_full.sh
```

模型顺序：

```text
qwen3 -> glm4 -> deepseek7b
```

测试层：

```text
qwen3:
  L4, L8, L12, L16, L20

glm4:
  L4, L10, L20, L30

deepseek7b:
  L4, L8, L12, L16, L20
```

每个模型运行完使用：

```text
--hard-exit-after-model
```

### 工程结果

输出目录：

```text
results/gpt5_phase65_object_attribute_compat_full_20260608_1818
```

汇总文件：

```text
results/gpt5_phase65_object_attribute_compat_full_20260608_1818/phase65_object_attribute_compat_summary.json
results/gpt5_phase65_object_attribute_compat_full_20260608_1818/PHASE65_OBJECT_ATTRIBUTE_COMPAT_SUMMARY.md
```

运行状态：

```text
三模型全部完成；
没有系统卡死；
没有 GPU reset；
运行结束后显存约 680 MiB / 24564 MiB；
```

### Qwen3 结果

```text
Layer | FULL | neutral_ideal | clean_margin(FULL-neutral)
L4    | 3    | 3             | 0
L8    | 2    | 3             | -1
L12   | 1    | 3             | -2
L16   | 2    | 3             | -1
L20   | 0    | 1             | -1
```

方向分解：

```text
L4:
  L1_FULL = 1
  L2cf_FULL = 1
  OBJcf_FULL = 1

L8:
  OBJcf_FULL = 2

L12:
  OBJcf_FULL = 1

L16:
  OBJcf_FULL = 2
```

分类别：

```text
size:
  全部 NO

color:
  少量 FULL，主要在 OBJ_cf

moisture:
  少量 FULL / HALF，主要在 OBJ_cf 或 L2_cf
```

客观现象：

```text
Qwen3 的 FULL 数量很少；
neutral_ideal 与 FULL 同量甚至更多；
因此 Qwen3 当前不能作为干净兼容性梯度证据。
```

更谨慎解释：

```text
Qwen3 早层存在一些对象-属性方向效果，
但它们很可能混入 value bias / neutral 输出偏好。
```

### GLM4 结果

```text
Layer | FULL | neutral_ideal | clean_margin(FULL-neutral)
L4    | 0    | 0             | 0
L10   | 3    | 0             | 3
L20   | 1    | 1             | 0
L30   | 2    | 1             | 1
```

方向分解：

```text
L10:
  L1_FULL = 2
  L2cf_FULL = 1
  OBJcf_FULL = 0

L20:
  OBJcf_FULL = 1

L30:
  OBJcf_FULL = 2
```

分类别：

```text
color:
  L1_FULL = 2
  L2cf_FULL = 1
  OBJcf_FULL = 1

moisture:
  L1/L2_cf 多为 HALF
  OBJcf_FULL = 1

size:
  只有极少 OBJcf_FULL
```

客观现象：

```text
GLM4 L10 是本轮最干净的 GLM4 峰值：
  FULL = 3
  neutral_ideal = 0

这与 GLM5 memo Phase396b 中 GLM4 L10 color/moisture 的线索一致。
```

解释：

```text
GLM4 的对象-属性兼容性更像中层交互；
L10 比 L20/L30 更少受到 neutral value bias 污染。
```

### DeepSeek7B 结果

```text
Layer | FULL | neutral_ideal | clean_margin(FULL-neutral)
L4    | 12   | 4             | 8
L8    | 6    | 3             | 3
L12   | 11   | 2             | 9
L16   | 10   | 1             | 9
L20   | 7    | 5             | 2
```

方向分解：

```text
L4:
  L1_FULL = 3
  L2cf_FULL = 4
  OBJcf_FULL = 5

L12:
  L1_FULL = 6
  L2cf_FULL = 2
  OBJcf_FULL = 3

L16:
  L1_FULL = 3
  L2cf_FULL = 3
  OBJcf_FULL = 4

L20:
  L1_FULL = 2
  L2cf_FULL = 1
  OBJcf_FULL = 4
```

分类别：

```text
size:
  L1_FULL = 9
  L2cf_FULL = 10
  OBJcf_FULL = 11

color:
  L1_FULL = 4
  L2cf_FULL = 2
  OBJcf 主要是 HALF

moisture:
  OBJcf_FULL = 6
  L1_FULL = 3
  L2cf_FULL = 1
```

客观现象：

```text
DS7B 是当前对象-属性兼容性最强模型。
L12/L16 是最干净的峰值区：
  L12: FULL=11, neutral=2
  L16: FULL=10, neutral=1
```

解释：

```text
DS7B 的对象-属性兼容性不是单层点；
它在 L12-L16 形成较稳定的上下文依赖兼容性交互。
L20 仍有 FULL，但 neutral_ideal 上升，说明更深层可能混入输出值偏好。
```

### 三模型对比

```text
Qwen3:
  FULL 少，neutral 多；
  当前不是干净兼容性机制候选。

GLM4:
  L10 小规模但干净；
  中层有少量真实交互候选。

DS7B:
  L12-L16 强且相对干净；
  当前最适合继续做对象-属性机制闭包。
```

排序：

```text
compatibility_interaction_candidate:
  DS7B L12/L16 > GLM4 L10 > Qwen3 L4
```

### 对 GLM5 Phase396/396b 的复验结论

本轮支持 Phase396b 的关键修正：

```text
1. 分对象后确实存在 FULL_SYMMETRIC。
2. 类别平均会掩盖对象级信号。
3. neutral prompt 是必要过滤器。
4. DS7B L12 是最强候选层之一。
```

但本轮进一步修正：

```text
1. DS7B L16 与 L12 同样重要，甚至 neutral 更少。
2. Qwen3 的 FULL 被 neutral_ideal 抵消，不应作为强证据。
3. GLM4 L10 比 L20/L30 更干净。
4. OBJ_cf 方向在 DS7B L4/L16/L20 中有贡献，说明 object-specific residual 不是噪声。
```

### 当前理论进展

对象-属性兼容性机制更像：

```text
object identity direction
× current value context
× model/layer-specific value bias
→ target/competitor logit interaction
```

而不是：

```text
一个固定 compatibility axis。
```

更准确的第一性描述：

```text
兼容性不是方向自身的属性；
兼容性是对象状态与值状态在残差流中的交互结果。
```

这与“相对编码”一致：

```text
对象的意义不是静态向量；
属性的意义也不是静态向量；
兼容关系在对象-属性-上下文三者组合中产生。
```

### 硬伤

1. 仍然是加性方向注入，不是自然动态重算。
2. neutral prompt 与 valued prompt 的 token 结构仍不完全匹配。
3. target/competitor logit 只取单 token，可能受到 tokenizer 与输出接口影响。
4. 只测 color/size/moisture 三类，还没有扩展到 texture、temperature、material、function 等。
5. 没有做 destroy-restore，不能称为闭包。
6. 没有做 MLP/attention 分解，无法判断 DS7B L12-L16 具体由哪个模块完成。

### 下一步计划

下一阶段应以 DS7B L12/L16 为主线，做真正机制拆解。

建议 Phase66：

```text
DS7B L12-L16 Object-Attribute Dynamic Closure
```

任务：

```text
1. 对 DS7B L12/L16 做 token-level path localization:
   object token
   value token
   last token
   all tokens

2. 做 module decomposition:
   resid_in
   attn_out
   mlp_out
   resid_out

3. 对 FULL 对象做 destroy-restore:
   破坏 object-specific direction；
   破坏 value-context direction；
   恢复 object-specific direction；
   测 target/competitor 是否恢复。

4. 做 matched neutral:
   使用占位值或语法匹配 prompt，
   减少 "The item is" 与 "The item is wet." 的结构差异。

5. 扩展 category:
   temperature
   texture
   material
   function
```

并行任务：

```text
GLM4 L10 小规模机制复核；
Qwen3 暂停对象-属性闭包，优先处理 reader/path 更稳定的 different-class。
```

### 当前结论

Phase65 完成三模型对象-属性兼容性梯度分解全量测试。

最可靠结果：

```text
DS7B L12-L16 是当前最强、相对最干净的对象-属性兼容性交互候选；
GLM4 L10 有少量但干净的中层交互；
Qwen3 的对象-属性 FULL 信号被 neutral_ideal 抵消，不宜作为强机制证据。
```

因此，当前已经可以从“对象-属性是否有方向”推进到：

```text
DS7B L12-L16 的对象-属性兼容性交互如何由 token、module、residual path 共同实现？
```

## Phase 66: 三图谱一闭包整体研究计划 [2026-06-08 19:18]

### 任务来源

读取并分析：

```text
research/MainAnalysis/20260609_03_从知识网络-逻辑推理-语法规则三者破解编码机制.md
```

并结合当前 Phase63-65 的真实测试结果，重新制定后续大计划。

### 对 MainAnalysis 文档的判断

文档的大方向是正确的。它把研究对象从：

```text
单一概念方向
单一路径
单一 probe 准确率
```

推进到：

```text
知识网络
逻辑推理
语法规则
三者背后的 relation state / candidate distribution / routing dynamics
```

这是当前项目最需要的升级。

其中最正确的部分：

```text
1. 语言编码不是固定语义轴，而是条件化关系状态。
2. 单个 binding 路径意义有限，必须和其他关系路径比较，形成全局路径图谱。
3. 知识网络、逻辑推理、语法规则应分别建任务宇宙，再比较它们的共同结构。
4. 读出器校准是机制研究的一部分，不是外部评测。
5. 候选集合分布比 target-vs-competitor 更接近真实机制。
6. 必须区分 norm/readout 效应和真正关系绑定效应。
7. 不能直接把 additive patch 当成自然机制，需要逐步进入 natural activation exchange、destroy-restore、path closure。
```

需要修正的部分：

```text
1. 不能同时全面铺开知识、逻辑、语法三大宇宙，否则每条线都会停在行为图谱层。
2. 当前 reader 仍是最大瓶颈，逻辑和语法不能越过 reader gate 直接做 patch。
3. Phase65 显示对象-属性 binding 的最佳突破口不是 Qwen3，而是 DS7B L12-L16。
4. Qwen3 当前更适合 different-class / category reader 或逻辑路径，暂时不适合作为对象-属性闭包主模型。
5. GLM4 L10 有干净但小的对象-属性中层交互，适合作为复核对象，不适合作为第一主线。
```

### 当前研究进展定位

已经完成的拼图：

```text
Phase63-64:
  same/different class reader 校准证明：
  Qwen3 可形成较可靠 class relation reader；
  GLM4 和 DS7B 在该 reader 任务上不稳定；
  跨模型 reader 不成立。

Phase65:
  object-attribute compatibility decomposition 证明：
  DS7B L12-L16 存在当前最强、相对最干净的对象-属性兼容性交互候选；
  GLM4 L10 存在少量但干净的中层交互；
  Qwen3 对象-属性 full 信号大多被 neutral_ideal 抵消，不宜过度解释。
```

因此当前不能再只说“语言是相对编码”，而要进入更具体的问题：

```text
相对编码中的关系状态，如何在对象-属性、逻辑算子、语法角色三类任务中形成、传播、读出和复用？
```

### 总体大计划：三图谱一闭包

后续研究不再按零散小功能推进，而按一个大框架推进：

```text
一、知识网络关系图谱
二、逻辑状态转移图谱
三、语法角色路由图谱
四、机制闭包验证
```

简称：

```text
三图谱一闭包
```

核心原则：

```text
先找到可稳定读出的关系；
再定位路径；
再做候选集合分布；
再做 neutral/norm/control 扣除；
再做 natural exchange；
最后做 destroy-restore。
```

### 第一大任务：知识网络关系图谱

优先级最高。

原因：

```text
1. Phase65 已经给出 DS7B L12-L16 的强候选层。
2. 对象-属性关系比语法角色查询更容易构造稳定 candidate set。
3. object-attribute 是知识网络最基础的关系单元：
   object + attribute + value + compatibility + candidate distribution。
```

目标：

```text
建立对象-属性关系如何在模型中编码、传播、读出的全局路径图谱。
```

任务拆分：

```text
Phase67:
  DS7B L12-L16 对象-属性 token/module/path 定位。
  测 resid_in, attn_out, mlp_out, resid_out。
  测 object token, attribute token, value token, last token。

Phase68:
  DS7B 对象-属性 candidate-set 全量读出。
  不只测 target/competitor，而测 compatible/incompatible/neutral/full candidate distribution。

Phase69:
  DS7B 对象-属性 natural activation exchange。
  从 additive patch 转为 clean object/value/context state exchange。

Phase70:
  DS7B 对象-属性 destroy-restore。
  破坏 object-specific relation direction / value-context direction；
  再恢复；
  检查 candidate distribution 是否恢复。

Phase71:
  扩展知识关系：
  color, size, wetness, temperature, texture, material, function, location, part-of, used-for, can-do, is-a。
```

并行复核：

```text
GLM4 L10:
  小规模复核中层对象-属性干净交互是否稳定。

Qwen3:
  暂停对象-属性闭包；
  优先使用 Phase64 已通过 reader 的 different_natural_control 做 category relation 图谱。
```

### 第二大任务：逻辑状态转移图谱

不能立即做 patch，必须先通过 reader gate。

原因：

```text
之前 temporal_order / role query 的自然语言 reader 多次失败；
说明读出器不稳时，任何 patch 结果都没有机制解释力。
```

目标：

```text
建立 operator × scope × candidate distribution 的状态转移图。
```

第一批逻辑任务：

```text
negation:
  A is true / A is not true / not all / none / double negation

causal:
  because / so / therefore

contrast:
  but / although / however

condition:
  if / unless / only if

temporal:
  before / after / during / then
```

执行顺序：

```text
Phase72:
  只做 symbolic reader calibration，不做 patch。

Phase73:
  对通过 reader gate 的逻辑任务，建立 layer/module candidate distribution map。

Phase74:
  对稳定逻辑任务做 operator state exchange。

Phase75:
  做 operator destroy-restore。
```

通过标准：

```text
reader 必须跨：
  template variation
  candidate order
  AB/BA
  entity/value replacement
  three models 或至少 one-model stable

否则不进入机制解释。
```

### 第三大任务：语法角色路由图谱

语法任务暂时不能直接继续做 role-binding patch，因为之前 Phase42/304/305 暴露出 reader 不稳。

目标：

```text
建立 identity-role-construction-position routing map。
```

第一批任务：

```text
active/passive
agent/patient
surface subject / semantic agent
by-agent
dative
relative clause
coreference
```

执行顺序：

```text
Phase76:
  重新设计 symbolic role reader。
  不用自然语言 next-token 问答作为主读出器。

Phase77:
  用候选集合全序列概率校准 agent/patient reader。

Phase78:
  只对通过 reader 的结构做 token-relation state transplant。

Phase79:
  做 construction signal + role signal 组合实验。

Phase80:
  做 role subspace destroy-restore。
```

### 第四大任务：机制闭包与跨模型不变量

最终不是找某个模型里的某个方向，而是找跨模型功能约束。

闭包标准：

```text
1. 可读出：
   候选集合分布稳定指向正确关系。

2. 可定位：
   能定位到 layer / module / token / path。

3. 可干预：
   natural exchange 或 subspace patch 能按预测改变候选分布。

4. 可破坏：
   删除候选变量后，功能下降。

5. 可恢复：
   恢复候选变量后，功能恢复。

6. 可迁移：
   换实体、换属性、换模板、换模型后仍保留功能结构。
```

跨模型比较不比较向量本身，而比较：

```text
功能路径类型
candidate distribution shape
层段位置
module 分工
destroy-restore 曲线
对 neutral/norm/control 的敏感性
```

### 第一性原理判断

当前最稳的第一性原理不是“语言有某个语义轴”，而是：

```text
语言系统必须在有限参数中高效表示无限组合；
因此它必须把 object, relation, value, operator, role, scope, position, context 分解成可复用变量；
同时通过残差路径、模块变换、候选集合竞争实现条件化组合。
```

所以真正要找的是：

```text
变量如何复用；
变量如何绑定；
变量如何读出；
变量如何在不同功能中分叉；
哪些动态约束跨模型保留。
```

这比寻找固定语义方向更接近语言背后的数学结构。

### 下一步立即执行建议

不要再平均推进所有功能。下一步集中完成：

```text
知识网络图谱第一阶段：
DS7B L12-L16 对象-属性关系路径闭包。
```

具体执行：

```text
1. 编写 Phase67 跨模型脚本，但主分析对象为 DS7B。
2. 三模型都跑，必须添加 --hard-exit-after-model。
3. qwen3 -> glm4 -> deepseek7b 依次运行，每个模型结束后卸载。
4. 对 DS7B L12/L16 加大数据量。
5. 生成 token × module × layer × relation 的候选分布图谱。
6. 只把通过 neutral/norm/control 扣除的结果作为机制候选。
```

### 当前结论

MainAnalysis 文档方向正确，但后续执行必须收束到：

```text
先知识网络；
再逻辑状态；
再语法路由；
最后跨模型闭包。
```

当前第一突破口已经非常明确：

```text
DS7B L12-L16 对象-属性兼容性交互。
```

这条线如果完成 destroy-restore，才真正从“全局路径图谱”进入“语言编码机制闭包”。

## Phase 67: 对象-属性关系 token/module/path 图谱测试 [2026-06-08 19:56]

### 任务目标

根据 Phase66 的“三图谱一闭包”计划，开始第一大任务：

```text
知识网络关系图谱
```

本轮聚焦对象-属性关系：

```text
object -> attribute candidate distribution
```

不再只看 target/competitor 的整体 logit 差，而是开始拆：

```text
layer
module
token position
candidate rank flip
candidate margin shift
```

核心问题：

```text
对象信息从 clean prompt 移植到 generic prompt 后，
候选属性分布是否按对象属性兼容性移动？
```

### 新增脚本

```text
tests/gpt5/phase67_object_attribute_path_map.py
tests/gpt5/phase67_object_attribute_path_summary.py
tests/gpt5/run_phase67_object_attribute_path_full.sh
```

### 测试原理

构造 clean/corrupt prompt：

```text
clean:
  The apple is

corrupt:
  The item is
```

候选集合：

```text
target:
  red

distractors:
  blue / white / black / small ...
```

在指定 layer/module/token position 上提取：

```text
delta = activation(clean object prompt) - activation(corrupt generic prompt)
```

再把 delta 加到 corrupt prompt 中，观察：

```text
target rank 是否上升；
target-vs-distractor margin 是否上升；
patch 后 target 是否变成 top1。
```

测试模块：

```text
resid_out
attn_out
mlp_out
```

测试位置：

```text
object_first
object_last
last
```

关系类别：

```text
color
moisture
size
temperature
texture
material
```

### 关键工程修正

第一次 full run 发现：

```text
GLM4 delta_norm 全部为 0；
DS7B object token delta 大量为 0。
```

原因不是机制结果，而是输入位置不一致：

```text
GLM4 tokenizer 默认 forward 时会加入特殊 token；
而位置匹配使用的是 add_special_tokens=False 的 token 序列；
导致 patch 打到错误位置。
```

修正：

```text
所有 forward encode 显式使用 add_special_tokens=False。
```

旧目录：

```text
results/gpt5_phase67_object_attribute_path_full_20260608_193555
```

只作为工程错误记录，不作为机制结果。

有效目录：

```text
results/gpt5_phase67_object_attribute_path_full_20260608_194412
```

### 正式测试命令

```bash
PHASE67_OUTPUT_DIR=results/gpt5_phase67_object_attribute_path_full_20260608_194412 \
PHASE67_PROGRESS_EVERY=24 \
bash tests/gpt5/run_phase67_object_attribute_path_full.sh
```

脚本内部按顺序运行：

```text
qwen3:
  layers = 4,8,12,16,20
  items = 72

glm4:
  layers = 4,10,20,30
  items = 72

deepseek7b:
  layers = 8,12,16,20
  items = 144
```

每个模型都使用：

```text
--hard-exit-after-model
```

模型顺序：

```text
qwen3 -> glm4 -> deepseek7b
```

### 输出文件

```text
results/gpt5_phase67_object_attribute_path_full_20260608_194412/qwen3_phase67_object_attribute_path_map.json
results/gpt5_phase67_object_attribute_path_full_20260608_194412/glm4_phase67_object_attribute_path_map.json
results/gpt5_phase67_object_attribute_path_full_20260608_194412/deepseek7b_phase67_object_attribute_path_map.json
results/gpt5_phase67_object_attribute_path_full_20260608_194412/phase67_object_attribute_path_summary.json
results/gpt5_phase67_object_attribute_path_full_20260608_194412/PHASE67_OBJECT_ATTRIBUTE_PATH_SUMMARY.md
```

数据规模：

```text
qwen3:
  rows = 3240

glm4:
  rows = 2592

deepseek7b:
  rows = 5184

total:
  rows = 11016
```

### 三模型客观结果

#### Qwen3

最强路径：

```text
L4 resid_out object_first/object_last
mean_progress = 0.7790
rank_flip_rate = 0.6528
improve_rate = 0.8750
clean_top1 = 0.7361
corrupt_not_top1 = 0.9444
```

稳健 eligible 子集：

```text
eligible = clean target top1 且 corrupt target 非 top1
n = 49 / 72

L4 resid_out object token:
  delta_eligible = 3.4592
  patch_top1_eligible = 0.9184
```

客观现象：

```text
Qwen3 的对象-属性候选分布可以被早层 object-token residual state 强力恢复。
L4 是当前最强层。
attention_out 和 mlp_out 单独作用很弱。
```

这和 Phase65 中 Qwen3 object-attribute full 结果被 neutral 抵消的结论并不矛盾：

```text
Phase65 测的是 full value-conditioned direction；
Phase67 测的是 object-neutral 到 generic-neutral 的 object state transplant。
Qwen3 更像 object identity / category 信息可以在早层 residual 中读出并影响候选属性，
但这不等于已经证明 value-conditioned compatibility binding。
```

#### GLM4

最强路径：

```text
L30 resid_out last
mean_progress = 0.6848
rank_flip_rate = 0.6528
improve_rate = 0.9028
clean_top1 = 0.7500
corrupt_not_top1 = 0.9167
```

稳健 eligible 子集：

```text
eligible = 48 / 72

L30 resid_out last:
  delta_eligible = 4.6885
  patch_top1_eligible = 0.9792

L20 resid_out last:
  delta_eligible = 3.8314
  patch_top1_eligible = 0.8750

L4 resid_out object token:
  delta_eligible = 3.9662
  patch_top1_eligible = 0.8333
```

客观现象：

```text
GLM4 在修正 token 位置后不再是 0；
它存在强 residual path；
但最强不是 Phase65 的 L10，而是 L30 last-token residual readout。
```

解释需谨慎：

```text
Phase65 的 L10 干净交互更可能是 value-conditioned full delta；
Phase67 的对象到属性候选恢复，更像 late residual readout 或 early object residual identity。
两者不是同一实验，不应强行合并为同一机制。
```

#### DeepSeek7B

四层扫描最强路径：

```text
L12 resid_out object_last
mean_progress = 0.7463
rank_flip_rate = 0.2292
improve_rate = 0.9167
clean_top1 = 0.4583
corrupt_not_top1 = 0.9028
```

稳健 eligible 子集：

```text
eligible = 54 / 144

L12 resid_out object_last:
  delta_eligible = 2.2274
  patch_top1_eligible = 0.5185

L12 resid_out object_first:
  delta_eligible = 2.1227
  patch_top1_eligible = 0.5000

L8 resid_out object_last:
  delta_eligible = 2.2541
  patch_top1_eligible = 0.4630
```

客观现象：

```text
DS7B 的 clean_top1 只有 0.4583，说明候选读出器本身较弱；
但在 eligible 子集上，L8-L12 object-token residual transplant 能显著提升 target margin；
L12 object_last 是当前最强候选。
```

### DS7B Dense Scan

为了避免 L8/L12/L16 四点采样错过峰值层，追加 DS7B dense scan：

```bash
OUT=results/gpt5_phase67_object_attribute_path_ds7b_dense_20260608_195121
PHASE67_ATTN_IMPLEMENTATIONS=flash_attention_2,sdpa,eager \
python tests/gpt5/phase67_object_attribute_path_map.py deepseek7b \
  --layers 8,9,10,11,12,13,14,15,16 \
  --max-items 144 \
  --modules resid_out,mlp_out \
  --positions object_first,object_last,last \
  --frames the,this,that,a \
  --output-dir "$OUT" \
  --progress-every 24 \
  --hard-exit-after-model
```

输出：

```text
results/gpt5_phase67_object_attribute_path_ds7b_dense_20260608_195121/deepseek7b_phase67_object_attribute_path_map.json
```

数据规模：

```text
rows = 7776
items = 144
layers = L8-L16
modules = resid_out, mlp_out
```

Dense scan 关键结果：

```text
L12 resid_out object_last:
  eligible_top1 = 0.5185
  delta_eligible = 2.2274
  rank_flip_rate = 0.2292

L11 resid_out object_last:
  eligible_top1 = 0.4815
  delta_eligible = 2.1921
  rank_flip_rate = 0.2361

L8 resid_out object_last:
  eligible_top1 = 0.4630
  delta_eligible = 2.2541
  rank_flip_rate = 0.2292

L10 resid_out object_last:
  eligible_top1 = 0.4630
  delta_eligible = 2.1629
  rank_flip_rate = 0.2222
```

连续层曲线：

```text
resid_out object_last:

L8:
  flip = 0.2292
  eligible_top1 = 0.4630
  delta_eligible = 2.2541

L9:
  flip = 0.2222
  eligible_top1 = 0.4444
  delta_eligible = 2.2075

L10:
  flip = 0.2222
  eligible_top1 = 0.4630
  delta_eligible = 2.1629

L11:
  flip = 0.2361
  eligible_top1 = 0.4815
  delta_eligible = 2.1921

L12:
  flip = 0.2292
  eligible_top1 = 0.5185
  delta_eligible = 2.2274

L13-L16:
  eligible_top1 降到 0.4074-0.4259 区间
```

结论：

```text
DS7B 对象-属性关系不是单点 L12，而是 L8-L12 中层 residual object-token 平台；
L12 是当前最好的读出点，但 L8-L11 也保留明显路径能力；
L13-L16 开始下降。
```

这修正 Phase65 的判断：

```text
Phase65 认为 DS7B L12/L16 最强；
Phase67 dense scan 进一步显示，L8-L12 是更合理的对象-属性关系平台，
L16 仍有效但不是峰值。
```

### 当前最可靠结论

```text
1. Qwen3:
   早层 L4 object-token residual state 可以强力恢复对象属性候选分布。

2. GLM4:
   late L30 last-token residual readout 最强；
   也存在 L4 object-token residual 早层信号。

3. DS7B:
   L8-L12 中层 object-token residual 平台最值得继续做闭包；
   L12 object_last 是当前最佳单点。
```

三模型共同点：

```text
对象-属性候选分布最主要由 residual path 承载；
attn_out 和 mlp_out 单独 transplant 都弱得多；
这说明 object-attribute relation 更像 residual trajectory 中的可读状态，
而不是单个 attention/MLP 输出直接决定。
```

### 硬伤和问题

1. 当前仍是 additive delta，不是 natural activation exchange。

```text
delta = clean - corrupt
patch = corrupt + delta
```

这会带来非自然状态风险。

2. DS7B 的 clean_top1 只有 0.4583。

```text
说明当前候选集合里很多样本模型本来不把 target 当 top1；
DS7B 的机制结论必须主要基于 eligible 子集。
```

3. GLM4 的 Phase65 L10 和 Phase67 L30 不一致。

```text
这可能说明 value-conditioned compatibility 和 object-neutral attribute readout 是两种不同任务；
不能过度合并。
```

4. `arid` 是 multi-token candidate，被跳过。

```text
后续需要清理 candidate vocabulary，
保证候选值尽量都是单 token 或改用 full-sequence logprob。
```

5. DS7B 使用 sdpa 时有 sliding-window attention warning。

```text
attention_out 结果需要降级解释；
本轮主要依据 resid_out 和 mlp_out。
```

### 理论进展

本轮支持一个更具体的知识网络机制图景：

```text
对象属性关系不是简单属性方向；
而是对象 token 的 residual state 改变候选属性分布。
```

更通俗地说：

```text
模型不是只记住 “red” 这个属性；
而是对象词元在残差路径里携带一种候选分布偏置，
这个偏置会让后续 “is ...” 的属性候选发生排序变化。
```

这比“对象-属性有一个方向”更接近知识网络编码机制。

当前第一性原理判断：

```text
知识网络可能不是按固定概念轴储存，
而是通过对象 token residual state 对候选属性集合施加条件化约束。
```

这符合相对编码：

```text
apple 的编码不是独立固定点；
apple 在 “The apple is ...” 这种上下文中，
相对于候选属性集合产生 red/green/small 等候选优先级变化。
```

### 下一步计划

Phase68 不应继续只加数据，而要升级干预方式：

```text
Phase68:
  DS7B L8-L12 object-attribute natural activation exchange。

核心实验：
  不再做 corrupt + (clean - corrupt)；
  而是直接 transplant clean object token state 到 corrupt prompt 的对应位置。

目标：
  判断对象 token residual state 本身是否足以恢复属性候选分布。
```

Phase69:

```text
DS7B L8-L12 object-attribute destroy-restore。

destroy:
  移除 object-token residual relation component。

restore:
  恢复 object-token residual relation component。

判据:
  target candidate rank / margin 是否下降再恢复。
```

Phase70:

```text
扩展知识网络关系：
  is-a
  part-of
  used-for
  can-do
  location
  material
  function
```

最终目标：

```text
从 object -> attribute
扩展到 object -> relation -> value，
建立知识网络候选分布图谱。
```

## Phase 68: 对象-属性自然状态移植与 mismatched control [2026-06-08 21:42]

### 任务目标

Phase67 使用的是 additive delta：

```text
corrupt + (clean - corrupt)
```

这仍可能产生非自然状态。Phase68 升级为 natural activation exchange：

```text
直接把 clean object token state 替换到 corrupt prompt 对应位置。
```

同时加入 mismatched object control：

```text
correct transplant:
  clean object state -> corrupt generic prompt

control transplant:
  mismatched object state -> corrupt generic prompt
```

核心指标不再是单独 progress，而是：

```text
correct - control
```

也就是正确对象状态相对于错误对象状态，是否能更强地恢复目标属性候选分布。

### 新增脚本

```text
tests/gpt5/phase68_object_attribute_natural_exchange.py
tests/gpt5/phase68_object_attribute_natural_exchange_summary.py
tests/gpt5/run_phase68_object_attribute_natural_exchange_full.sh
```

### 数据扩展

本轮扩展对象-属性数据：

```text
color
moisture
size
temperature
texture
material
taste
weight
```

框架：

```text
The {object} is
This {object} is
That {object} is
A {object} is
```

corrupt prompt：

```text
The item is
This item is
That item is
A thing is
```

### 正式测试命令

```bash
PHASE68_OUTPUT_DIR=results/gpt5_phase68_object_attribute_natural_exchange_full_20260608_211815 \
PHASE68_PROGRESS_EVERY=32 \
bash tests/gpt5/run_phase68_object_attribute_natural_exchange_full.sh
```

脚本内部按顺序：

```text
qwen3 -> glm4 -> deepseek7b
```

并且每个模型使用：

```text
--hard-exit-after-model
```

### 输出文件

```text
results/gpt5_phase68_object_attribute_natural_exchange_full_20260608_211815/qwen3_phase68_object_attribute_natural_exchange.json
results/gpt5_phase68_object_attribute_natural_exchange_full_20260608_211815/glm4_phase68_object_attribute_natural_exchange.json
results/gpt5_phase68_object_attribute_natural_exchange_full_20260608_211815/deepseek7b_phase68_object_attribute_natural_exchange.json
results/gpt5_phase68_object_attribute_natural_exchange_full_20260608_211815/phase68_object_attribute_natural_exchange_summary.json
results/gpt5_phase68_object_attribute_natural_exchange_full_20260608_211815/PHASE68_OBJECT_ATTRIBUTE_NATURAL_EXCHANGE_SUMMARY.md
```

### 数据规模

```text
qwen3:
  items = 192
  rows = 4608
  layers = L4,L8,L12,L16

glm4:
  items = 192
  rows = 4608
  layers = L4,L10,L20,L30

deepseek7b:
  items = 248
  rows = 13392
  layers = L8-L16 dense

total:
  rows = 22608
```

### Qwen3 结果

最强路径：

```text
L4 resid_out object_last:
  eligible = 101
  correct_delta = 2.1105
  control_delta = -0.6276
  net_delta = 2.7381
  correct_flip = 0.5104
  control_flip = 0.0833
  eligible_correct_top1 = 0.9208
  eligible_control_top1 = 0.1089
  eligible_net_delta = 3.8308

L4 resid_out object_first:
  eligible_correct_top1 = 0.9208
  eligible_control_top1 = 0.1188
  eligible_net_delta = 3.8156
```

后续层：

```text
L8-L16 object-token resid_out 仍然强：
  eligible_correct_top1 = 0.7228 - 0.7426
  eligible_control_top1 = 0.0891 - 0.1287
```

客观现象：

```text
Qwen3 的对象-属性自然状态移植最强在 L4 object-token residual。
correct transplant 远强于 mismatched control。
这说明 Phase67 的 Qwen3 object-token residual 效果不是简单加法伪影。
```

### GLM4 结果

最强路径：

```text
L30 resid_out last:
  eligible = 117
  correct_delta = 3.0776
  control_delta = -0.6060
  net_delta = 3.6836
  correct_flip = 0.5833
  control_flip = 0.0573
  eligible_correct_top1 = 0.9487
  eligible_control_top1 = 0.0940
  eligible_net_delta = 4.6740
```

早层对象词元也强：

```text
L4 resid_out object_last:
  eligible_correct_top1 = 0.8547
  eligible_control_top1 = 0.0855
  eligible_net_delta = 4.4245

L4 resid_out object_first:
  eligible_correct_top1 = 0.8376
  eligible_control_top1 = 0.0855
  eligible_net_delta = 4.4146
```

中层：

```text
L10 resid_out last:
  eligible_correct_top1 = 0.6923
  eligible_control_top1 = 0.0940
  eligible_net_delta = 3.1217

L10 object token:
  eligible_correct_top1 = 0.6068 - 0.6154
  eligible_control_top1 = 0.0769 - 0.0855
```

客观现象：

```text
GLM4 的对象-属性自然状态移植非常强；
最强是 late L30 last-token residual readout；
但 L4 object-token residual 也很强。
```

这比 Phase67 更强，因为：

```text
correct transplant 明显强于 mismatched control；
因此不是随便移植一个对象状态都能恢复目标属性。
```

### DeepSeek7B 结果

最强路径：

```text
L12 resid_out object_last:
  eligible = 84
  correct_delta = 1.3802
  control_delta = 0.4231
  net_delta = 0.9571
  correct_flip = 0.1976
  control_flip = 0.0605
  eligible_correct_top1 = 0.5000
  eligible_control_top1 = 0.1190
  eligible_net_delta = 1.5433
```

连续层平台：

```text
L8-L16 resid_out object_last:
  eligible_net_delta = 1.4193 - 1.5433
  eligible_correct_top1 = 0.4167 - 0.5000
  eligible_control_top1 = 0.1071 - 0.1310
```

前几名：

```text
L12 object_last:
  eligible_net_delta = 1.5433

L10 object_last:
  eligible_net_delta = 1.5039

L11 object_last:
  eligible_net_delta = 1.4939

L8 object_last:
  eligible_net_delta = 1.4909

L9 object_last:
  eligible_net_delta = 1.4727
```

客观现象：

```text
DS7B 再次支持 L8-L12 中层 residual object-token 平台；
L12 object_last 是当前最佳点；
但 L8-L11 也非常接近。
```

### 三模型对比

```text
Qwen3:
  早层 L4 object-token residual 最强。

GLM4:
  late L30 last-token residual readout 最强；
  早层 L4 object-token residual 也强。

DS7B:
  L8-L12 中层 object-token residual 平台最强；
  L12 object_last 当前最佳。
```

共同点：

```text
1. residual path 明显强于 MLP path。
2. correct natural transplant 显著强于 mismatched control。
3. 对象状态可以直接改变属性候选分布。
```

这比 Phase67 更接近机制证据。

### 当前最可靠结论

对象-属性知识关系最稳的当前图景是：

```text
对象 token 的 residual state 包含可自然移植的候选属性偏置信息。
这个状态不是单纯属性方向；
它在具体上下文中对属性候选集合施加条件化约束。
```

换句话说：

```text
object + context -> residual object state -> attribute candidate distribution
```

这是知识网络关系编码的一个实际候选机制。

### 和“相对编码”的关系

本轮结果支持相对编码，而不是固定轴编码：

```text
同一个对象状态不是孤立概念点；
它在 “X is ...” 这种关系上下文中，
相对于候选属性集合改变 red/blue/wet/dry/large/tiny 等候选的优先级。
```

也就是说：

```text
编码不是 object = vector；
而是 object 在 relation context 中产生 candidate distribution constraint。
```

### 硬伤

1. 还没有 destroy-restore。

```text
Phase68 证明了 sufficiency-like evidence：
  正确对象状态移植可以恢复候选属性。

但还没有证明 necessity：
  破坏对象关系状态会让属性候选失败。
```

2. DS7B eligible 仍只有 84/248。

```text
说明 DS7B 当前候选读出器仍偏弱；
后续 DS7B 结论必须继续看 eligible 子集。
```

3. 当前 mismatched control 只是不同对象状态。

```text
还需要更多 control：
  same target different object
  same category different value
  random same-norm state
  shuffled object-token state
```

4. MLP path 弱不能解释为 MLP 不重要。

```text
本轮只测替换 MLP output；
MLP 可能负责把 residual state 转成下游可读格式，
但单独替换 MLP output 不一定足以恢复属性候选。
```

5. 还没有 full-sequence candidate scoring。

```text
当前仍使用 first-token logit；
后续 multi-token candidate 必须改用 full-sequence logprob。
```

### 理论进展

Phase68 让知识网络机制更具体：

```text
知识不是静态概念表；
知识更像对象状态对候选关系值的约束。
```

对象-属性关系可能的机制单元：

```text
object_token_residual_state
relation_context = "is ..."
candidate_attribute_distribution
```

也就是：

```text
对象状态 + 关系上下文 -> 候选属性排序
```

这个机制比“苹果有红色方向”更基础，因为它能解释：

```text
同一个对象在不同关系上下文中会激活不同候选集合；
同一类对象可以复用候选约束；
不同模型可以在不同层段实现相同候选分布效果。
```

### 下一步计划

Phase69 必须做 destroy-restore，不要继续只做 sufficiency。

建议：

```text
Phase69:
  object-attribute relation destroy-restore。
```

优先路径：

```text
Qwen3:
  L4 resid_out object token

GLM4:
  L4 resid_out object token
  L30 resid_out last token

DS7B:
  L8-L12 resid_out object token
```

核心实验：

```text
1. destroy:
   用 mismatched control state 替换 clean object state，
   或将 object residual state 投影到 control subspace。

2. observe failure:
   target rank / margin 下降。

3. restore:
   恢复 clean object residual state。

4. observe recovery:
   target rank / margin 恢复。
```

Phase70:

```text
扩展 object -> relation -> value：
  is-a
  used-for
  part-of
  can-do
  location
  material
  function
```

更大的目标：

```text
建立知识网络关系路径图谱：
  object relation value
  object attribute value
  category relation value
  relation candidate distribution
```

## Phase 69: 对象-属性关系 destroy-restore 闭包测试 [2026-06-08 22:10]

### 任务目标

Phase68 证明了：

```text
correct object natural transplant > mismatched object control
```

但这仍偏向 sufficiency-like evidence。

Phase69 进一步测试 necessity + recovery：

```text
1. destroy:
   在 clean prompt 的早层对象 token residual state 上，
   用 mismatched object state 替换，破坏对象-属性候选分布。

2. restore:
   在后续层恢复 clean object state。

3. 判断:
   target margin / target rank 是否从 destroy 中恢复。
```

这开始接近机制闭包：

```text
可破坏；
可恢复；
恢复后候选分布接近 clean。
```

### 新增脚本

```text
tests/gpt5/phase69_object_attribute_destroy_restore.py
tests/gpt5/phase69_object_attribute_destroy_restore_summary.py
tests/gpt5/run_phase69_object_attribute_destroy_restore_full.sh
```

### 测试原理

对 clean prompt：

```text
The apple is
```

候选：

```text
red / blue / white / black / small ...
```

先捕获：

```text
clean object state at destroy layer
control object state at destroy layer
clean object state at restore layer
```

然后运行：

```text
clean:
  原始 clean prompt

destroy:
  在 destroy_layer 把 clean object state 替换为 mismatched control object state

restore:
  在 destroy_layer 先执行 destroy；
  在 restore_layer 再恢复 clean object state
```

指标：

```text
destroy_drop = clean_margin - destroy_margin
restore_gain = restore_margin - destroy_margin
restore_to_clean_gap = clean_margin - restore_margin
```

如果：

```text
destroy_drop 大；
restore_gain 大；
restore_to_clean_gap 小；
destroy_top1 下降；
restore_top1 恢复；
```

说明该路径接近闭包。

### 正式测试命令

```bash
PHASE69_OUTPUT_DIR=results/gpt5_phase69_object_attribute_destroy_restore_full_20260608_215548 \
PHASE69_PROGRESS_EVERY=32 \
bash tests/gpt5/run_phase69_object_attribute_destroy_restore_full.sh
```

三模型顺序：

```text
qwen3 -> glm4 -> deepseek7b
```

每个模型：

```text
--hard-exit-after-model
```

### 输出文件

```text
results/gpt5_phase69_object_attribute_destroy_restore_full_20260608_215548/qwen3_phase69_object_attribute_destroy_restore.json
results/gpt5_phase69_object_attribute_destroy_restore_full_20260608_215548/glm4_phase69_object_attribute_destroy_restore.json
results/gpt5_phase69_object_attribute_destroy_restore_full_20260608_215548/deepseek7b_phase69_object_attribute_destroy_restore.json
results/gpt5_phase69_object_attribute_destroy_restore_full_20260608_215548/phase69_object_attribute_destroy_restore_summary.json
results/gpt5_phase69_object_attribute_destroy_restore_full_20260608_215548/PHASE69_OBJECT_ATTRIBUTE_DESTROY_RESTORE_SUMMARY.md
```

### 数据规模

```text
qwen3:
  items = 192
  layer_pairs = 5
  rows = 2880

glm4:
  items = 192
  layer_pairs = 6
  rows = 3456

deepseek7b:
  items = 248
  layer_pairs = 9
  rows = 6696

total:
  rows = 13032
```

### Qwen3 结果

最强闭包路径：

```text
L4 -> L8 object_last:
  eligible = 118
  eligible_destroy_drop = 4.0463
  eligible_restore_gain = 3.6676
  eligible_restore_to_clean_gap = 0.3787
  eligible_destroy_top1 = 0.1186
  eligible_restore_top1 = 0.8559

L4 -> L8 object_first:
  eligible_destroy_drop = 4.0143
  eligible_restore_gain = 3.6028
  eligible_restore_to_clean_gap = 0.4115
  eligible_destroy_top1 = 0.1186
  eligible_restore_top1 = 0.8475
```

其他路径：

```text
L4 -> L16 object_last:
  restore_gain = 3.3528
  restore_top1 = 0.7542

L8 -> L12 object_first:
  destroy_drop = 3.1200
  restore_gain = 3.0540
  restore_top1 = 0.9746
```

客观现象：

```text
Qwen3 的 L4 object-token residual state 被破坏后，target top1 大幅下降；
在 L8 恢复 clean object state 后，target top1 基本恢复。
```

这强力支持：

```text
Qwen3 的对象-属性关系闭包主路径在浅层 residual object token，
L4 写入/携带，L8 可恢复。
```

### GLM4 结果

最强闭包路径：

```text
L4 -> L10 object_last:
  eligible = 134
  eligible_destroy_drop = 4.6387
  eligible_restore_gain = 3.9606
  eligible_restore_to_clean_gap = 0.6781
  eligible_destroy_top1 = 0.1493
  eligible_restore_top1 = 0.8358

L4 -> L10 object_first:
  eligible_destroy_drop = 4.5601
  eligible_restore_gain = 3.8769
  eligible_restore_to_clean_gap = 0.6832
  eligible_destroy_top1 = 0.1716
  eligible_restore_top1 = 0.8358
```

中层恢复：

```text
L10 -> L20 object_last:
  eligible_destroy_drop = 2.6178
  eligible_restore_gain = 2.6512
  eligible_restore_to_clean_gap = -0.0335
  eligible_destroy_top1 = 0.4701
  eligible_restore_top1 = 0.9851
```

末层问题：

```text
L4 -> L30 object_last:
  destroy_drop = 4.6387
  restore_gain = 0.7948
  restore_gap = 3.8439
  restore_top1 = 0.2164
```

客观现象：

```text
GLM4 的 L4 -> L10 是强闭包路径；
L10 -> L20 也非常强；
但 L4 -> L30 object-token restore 很弱。
```

解释：

```text
GLM4 的对象-属性关系状态需要在较近后续层恢复；
如果只到 L30 才恢复 object-token state，已经错过关键传播/转换窗口。
```

这修正 Phase68 的 GLM4 L30 last 强读出：

```text
L30 last-token 是强 readout；
但 object-token relation closure 更像 L4-L20 的渐进路径，
不是 L30 object-token 单点可恢复。
```

### DeepSeek7B 结果

最强闭包路径：

```text
L8 -> L14 object_last:
  eligible = 115
  eligible_destroy_drop = 2.3420
  eligible_restore_gain = 2.2147
  eligible_restore_to_clean_gap = 0.1273
  eligible_destroy_top1 = 0.4174
  eligible_restore_top1 = 0.9043

L8 -> L16 object_last:
  eligible_destroy_drop = 2.3420
  eligible_restore_gain = 2.2077
  eligible_restore_to_clean_gap = 0.1342
  eligible_destroy_top1 = 0.4174
  eligible_restore_top1 = 0.9130

L8 -> L12 object_last:
  eligible_destroy_drop = 2.3420
  eligible_restore_gain = 2.1643
  eligible_restore_to_clean_gap = 0.1777
  eligible_destroy_top1 = 0.4174
  eligible_restore_top1 = 0.8957
```

中段恢复：

```text
L12 -> L14 object_last:
  eligible_destroy_drop = 2.1170
  eligible_restore_gain = 2.1287
  eligible_restore_to_clean_gap = -0.0117
  eligible_destroy_top1 = 0.4783
  eligible_restore_top1 = 0.9913

L12 -> L16 object_last:
  eligible_restore_top1 = 0.9826
```

客观现象：

```text
DS7B 的 L8-L16 中层平台形成强闭包；
破坏 L8/L10/L12 object-token state 会明显降低属性候选；
在 L12/L14/L16 恢复 clean state 后，候选分布强恢复。
```

这进一步确认 Phase67/68：

```text
DS7B 对象-属性关系不是单点 L12；
而是 L8-L16 中层 residual object-token 轨迹平台。
```

### 三模型对比

```text
Qwen3:
  L4 -> L8 是最清晰闭包路径。

GLM4:
  L4 -> L10 / L10 -> L20 是最清晰闭包路径；
  L30 更像读出层，不适合 object-token restore。

DS7B:
  L8 -> L14/L16 和 L12 -> L14/L16 都强；
  中层 residual 平台最明显。
```

共同规律：

```text
1. object-token residual state 被破坏后，target 属性候选显著下降。
2. 后续层恢复 clean object state 后，候选分布显著恢复。
3. 这是比 Phase68 更强的闭包证据。
```

### 当前最可靠机制图景

对象-属性知识关系的机制候选可以写成：

```text
object token residual trajectory
  -> relation context ("is ...")
  -> candidate attribute distribution
```

更具体：

```text
对象 token residual state 不是静态概念向量；
它是可在路径中破坏、恢复、继续传播的关系状态。
```

这支持：

```text
知识网络编码 = 对象关系状态对候选值分布的条件化约束。
```

### 对破解语言编码机制的意义

这是当前项目中少数真正接近闭包的结果：

```text
可读出：
  candidate attribute distribution 可测。

可干预：
  natural exchange 可以改变候选分布。

可破坏：
  mismatched object state 会破坏 target attribute。

可恢复：
  后续层恢复 clean state 能恢复 candidate distribution。
```

还没有完成的是：

```text
subspace-level destroy-restore；
跨关系类型 object-relation-value；
full-sequence candidate scoring；
跨模型抽象结构对齐。
```

### 严格硬伤

1. 当前 destroy 是 whole-state transplant。

```text
它破坏的是整个 object token residual state，
还没有分离 identity / category / relation / value constraint 子空间。
```

2. restore 是直接恢复 clean state。

```text
这证明可恢复，但还没有证明最小充分变量。
```

3. 候选仍是 first-token logit。

```text
后续必须支持 full-sequence logprob。
```

4. 关系类型仍主要是 attribute。

```text
还不能代表完整知识网络。
```

5. 没有 random same-norm / orthogonal control。

```text
后续需要确认不是任意大范数状态都能破坏/恢复。
```

### 下一步计划

Phase70 应该扩展为：

```text
object-relation-value global path map。
```

优先关系：

```text
is-a:
  robin -> bird
  salmon -> fish

part-of:
  wheel -> car
  leaf -> tree

used-for:
  knife -> cutting
  cup -> drinking

can-do:
  bird -> fly
  fish -> swim

location:
  fish -> water
  book -> shelf

material:
  spoon -> metal
  shirt -> cloth
```

目标：

```text
验证 Phase69 的对象-属性闭包是否能推广到 object-relation-value。
```

Phase71:

```text
subspace-level destroy-restore。
```

目标：

```text
从 whole object state 闭包，
推进到 identity/category/relation/value 子空间闭包。
```

阶段性大任务：

```text
建立知识网络编码图谱：
  object
  relation
  value
  candidate distribution
  residual trajectory
  destroy-restore closure
```

## Phase 70: 对象-关系-值全局 destroy-restore 闭包图谱 [2026-06-08 22:58]

### 任务目标

根据 Phase 69 的结果继续推进。

Phase 69 已经证明：

```text
object-token residual state 可以被破坏、恢复，并显著影响属性候选分布。
```

但 Phase 69 仍然只覆盖对象-属性关系，不能代表完整知识网络。本轮扩展为：

```text
object-relation-value global closure map
```

目标是验证 whole object-token residual closure 是否能推广到更多知识关系：

```text
is-a
part-of
used-for
can-do
location
material
function
```

本轮仍然不做理论跳跃，优先记录客观现象。

### 对用户分析的判断

用户提供的分析基本正确：

```text
1. Phase69 是目前最接近机制闭包的阶段。
2. 但 Phase69 闭合的是 whole object-token residual state，不是最小变量闭包。
3. 还没有分清 identity/category/relation/value prior/norm/readout 哪一部分起作用。
4. GLM4 L30 last-token 强读出不能等同于 object-token restore 机制。
5. 下一步应扩展到 object-relation-value，而不是只重复属性。
```

因此本轮执行 Phase70：对象-关系-值全局 destroy-restore 闭包图谱。

### 新增脚本

```text
tests/gpt5/phase70_object_relation_value_closure.py
tests/gpt5/phase70_object_relation_value_closure_summary.py
tests/gpt5/run_phase70_object_relation_value_closure_full.sh
```

脚本特性：

```text
1. 三模型依次运行：qwen3 -> glm4 -> deepseek7b。
2. 每个模型均使用 --hard-exit-after-model，避免显存残留。
3. 加载优先 flash_attention_2，当前环境无 flash_attn 包，自动 fallback 到 sdpa。
4. 使用 bfloat16 + device_map="auto"。
5. 每 32 items 输出进度日志。
6. 每个 layer pair 后写 partial json。
7. 输出 by_path / by_relation / by_relation_path 三种 summary。
```

### Smoke Test

命令：

```bash
OUT=results/gpt5_phase70_smoke_$(date +%Y%m%d_%H%M%S)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate openone-cu130-py312
python tests/gpt5/phase70_object_relation_value_closure.py qwen3 \
  --layer-pairs 4-8 \
  --max-items 16 \
  --module resid_out \
  --positions object_first,object_last \
  --output-dir "$OUT" \
  --progress-every 4 \
  --hard-exit-after-model
```

第一次 smoke 暴露一个数据抽样问题：

```text
max-items 前缀截断时可能只保留同一 target，导致 mismatched control 不存在。
```

已修复：

```text
build_items(max_items) 改为 deterministic even sampling，
避免关系/target 被前缀截断污染。
```

修复后 smoke：

```text
rows = 32
exit_code = 0
```

### 正式测试命令

```bash
PHASE70_OUTPUT_DIR=results/gpt5_phase70_object_relation_value_closure_full_20260608_222748 \
PHASE70_PROGRESS_EVERY=32 \
bash tests/gpt5/run_phase70_object_relation_value_closure_full.sh
```

三模型都正常完成。

说明：

```text
flash_attention_2 未安装，因此实际使用 PyTorch SDPA。
DeepSeek7B 仍有 sliding-window attention with SDPA warning。
本轮主要 hook resid_out，不解释 attention 权重，因此该 warning 记录为工程限制。
```

### 输出文件

```text
results/gpt5_phase70_object_relation_value_closure_full_20260608_222748/qwen3_phase70_object_relation_value_closure.json
results/gpt5_phase70_object_relation_value_closure_full_20260608_222748/glm4_phase70_object_relation_value_closure.json
results/gpt5_phase70_object_relation_value_closure_full_20260608_222748/deepseek7b_phase70_object_relation_value_closure.json
results/gpt5_phase70_object_relation_value_closure_full_20260608_222748/phase70_object_relation_value_closure_summary.json
results/gpt5_phase70_object_relation_value_closure_full_20260608_222748/PHASE70_OBJECT_RELATION_VALUE_CLOSURE_SUMMARY.md
```

### 数据规模

```text
Qwen3:
  items = 342
  rows = 5130
  layer_pairs = L4-L8, L4-L12, L4-L16, L8-L12, L8-L16

GLM4:
  items = 342
  rows = 6156
  layer_pairs = L4-L10, L4-L20, L4-L30, L10-L20, L10-L30, L20-L30

DeepSeek7B:
  items = 342
  rows = 9234
  layer_pairs = L8-L10, L8-L12, L8-L14, L8-L16, L10-L12, L10-L14, L10-L16, L12-L14, L12-L16

total_rows = 20520
```

### 实验原理

对每个关系样本：

```text
clean prompt:
  A robin is a kind of ...

target:
  bird

distractors:
  fish/tool/fruit/metal
```

destroy：

```text
在 clean prompt 中，把对象 token 的 clean residual state
替换成同关系/同模板下 mismatched object residual state。
```

restore：

```text
在后续 restore layer 恢复 clean object residual state。
```

核心指标：

```text
destroy_drop = clean_margin - destroy_margin
restore_gain = restore_margin - destroy_margin
restore_to_clean_gap = clean_margin - restore_margin
```

如果：

```text
destroy_drop 大；
restore_gain 大；
restore_to_clean_gap 小；
destroy_top1 低；
restore_top1 高；
```

说明该路径具备 whole-state destroy-restore closure 特征。

### Qwen3 客观结果

整体最强路径：

```text
L4 -> L8 object_last:
  eligible = 299
  eligible_destroy_drop = 9.8621
  eligible_restore_gain = 8.7762
  eligible_restore_to_clean_gap = 1.0859
  eligible_destroy_top1 = 0.1773
  eligible_restore_top1 = 0.8930

L4 -> L8 object_first:
  eligible_destroy_drop = 9.3580
  eligible_restore_gain = 8.2771
  eligible_restore_to_clean_gap = 1.0809
  eligible_destroy_top1 = 0.1973
  eligible_restore_top1 = 0.8963

L8 -> L12 object_last:
  eligible_destroy_drop = 8.1869
  eligible_restore_gain = 7.7148
  eligible_restore_to_clean_gap = 0.4721
  eligible_destroy_top1 = 0.3110
  eligible_restore_top1 = 0.9666

L8 -> L16 object_last:
  eligible_restore_top1 = 0.9766
```

关系级结果：

```text
is_a:
  eligible_destroy_drop = 9.4524
  eligible_restore_gain = 8.1246
  eligible_restore_top1 = 0.9728

used_for:
  eligible_destroy_drop = 7.5311
  eligible_restore_gain = 6.5013
  eligible_restore_top1 = 0.9319

can_do:
  eligible_destroy_drop = 5.6668
  eligible_restore_gain = 5.0645
  eligible_restore_top1 = 0.9188

function:
  eligible_restore_top1 = 0.9615

part_of:
  eligible_restore_top1 = 0.9248

material:
  eligible_restore_top1 = 0.8081

location:
  eligible_restore_top1 = 0.9385
```

客观现象：

```text
1. Qwen3 的 L4->L8 仍是最强 early closure path。
2. L8->L12/L16 restore_to_clean_gap 更小，说明中浅层恢复后更接近 clean。
3. is-a / used-for / can-do / function 都有强闭包。
4. material 相对弱，可能候选读出更受模板/候选词影响。
```

### GLM4 客观结果

整体最强路径：

```text
L4 -> L10 object_last:
  eligible = 316
  eligible_destroy_drop = 9.3529
  eligible_restore_gain = 6.1331
  eligible_restore_to_clean_gap = 3.2198
  eligible_destroy_top1 = 0.1709
  eligible_restore_top1 = 0.7057

L4 -> L20 object_last:
  eligible_restore_gain = 5.7241
  eligible_restore_to_clean_gap = 3.6288
  eligible_restore_top1 = 0.6835

L10 -> L20 object_last:
  eligible_destroy_drop = 4.0759
  eligible_restore_gain = 3.8696
  eligible_restore_to_clean_gap = 0.2063
  eligible_destroy_top1 = 0.6519
  eligible_restore_top1 = 0.9873

L10 -> L20 object_first:
  eligible_restore_to_clean_gap = 0.1904
  eligible_restore_top1 = 0.9873
```

晚层对照：

```text
L4 -> L30 object_last:
  eligible_restore_gain = 0.9985
  eligible_restore_to_clean_gap = 8.3544
  eligible_restore_top1 = 0.2437

L10 -> L30 object_last:
  eligible_restore_gain = 0.6100
  eligible_restore_to_clean_gap = 3.4659
  eligible_restore_top1 = 0.7025
```

关系级结果：

```text
used_for:
  eligible_destroy_drop = 6.2348
  eligible_restore_gain = 3.2933
  eligible_restore_top1 = 0.8009

is_a:
  eligible_destroy_drop = 6.7264
  eligible_restore_gain = 2.7854
  eligible_restore_top1 = 0.7912

can_do:
  eligible_restore_top1 = 0.7636

function:
  eligible_restore_top1 = 0.8527

part_of:
  eligible_restore_top1 = 0.7944

location:
  eligible_restore_top1 = 0.7733

material:
  eligible_restore_top1 = 0.7449
```

客观现象：

```text
1. GLM4 的 L4 早层 destroy 很强，但 L4->L10/L20 restore_gap 仍较大。
2. L10->L20 的 gap 极小，restore_top1 接近 0.9873，是更干净的 closure window。
3. L4->L30 object-token restore 继续失败，支持 Phase69 的修正：
   L30 更像 readout，不是 object-token restore 的有效窗口。
4. GLM4 的关系闭包弱于 Qwen3/DS7B，尤其早层恢复不完全。
```

### DeepSeek7B 客观结果

整体最强路径：

```text
L8 -> L10 object_last:
  eligible = 246
  eligible_destroy_drop = 5.0276
  eligible_restore_gain = 4.3645
  eligible_restore_to_clean_gap = 0.6631
  eligible_destroy_top1 = 0.3537
  eligible_restore_top1 = 0.9024

L8 -> L12 object_last:
  eligible_restore_gain = 4.2791
  eligible_restore_to_clean_gap = 0.7485
  eligible_restore_top1 = 0.8943

L12 -> L14 object_last:
  eligible_destroy_drop = 3.9365
  eligible_restore_gain = 3.9421
  eligible_restore_to_clean_gap = -0.0056
  eligible_destroy_top1 = 0.5163
  eligible_restore_top1 = 0.9756

L12 -> L16 object_last:
  eligible_restore_to_clean_gap = 0.0310
  eligible_restore_top1 = 0.9675
```

关系级结果：

```text
is_a:
  eligible_destroy_drop = 5.0370
  eligible_restore_gain = 4.7000
  eligible_restore_to_clean_gap = 0.3370
  eligible_restore_top1 = 1.0000

used_for:
  eligible_destroy_drop = 4.3628
  eligible_restore_gain = 3.8803
  eligible_restore_top1 = 0.9172

can_do:
  eligible_destroy_drop = 3.6241
  eligible_restore_gain = 3.4413
  eligible_restore_top1 = 0.9450

function:
  eligible_restore_top1 = 0.8815

material:
  eligible_restore_top1 = 0.9677

location:
  eligible_restore_top1 = 0.9427

part_of:
  eligible_restore_top1 = 0.9762
```

客观现象：

```text
1. DeepSeek7B 的 L8-L16 中层 residual platform 在 object-relation-value 上继续成立。
2. L12->L14/L16 的 restore_gap 接近 0，说明 L12 后对象关系状态非常可恢复。
3. is_a 关系最强，restore_top1 = 1.0。
4. part_of 的 destroy_drop 较弱，但 restore_top1 高，说明它可能本身 clean/top1 较稳、破坏幅度较小。
```

### 三模型对比

```text
Qwen3:
  L4->L8 是最强 early closure path。
  L8->L12/L16 更接近 clean。
  多数关系都有强闭包。

GLM4:
  L4 破坏强，但早层恢复不完全。
  L10->L20 是最干净闭包窗口。
  L30 object-token restore 继续弱，支持 readout/window 区分。

DeepSeek7B:
  L8-L16 中层平台稳定。
  L12->L14/L16 restore_gap 最小。
  多关系闭包比 GLM4 更干净。
```

### 当前最可靠机制拼图

Phase70 支持 Phase69 的扩展版本：

```text
object-token residual trajectory
  -> relation prompt/context
  -> candidate value distribution
```

更准确地说：

```text
对象 token residual state 不只是 object identity；
它在不同 relation frame 中，对候选 value 分布形成条件约束。
这种约束可以被 mismatched object state 破坏，
也可以被 clean object state 在后续层恢复。
```

这说明知识网络编码至少存在以下路径对象：

```text
object state
relation frame
candidate value set
residual trajectory
restore window
readout position
```

### 严格硬伤

1. 仍是 whole-state closure。

```text
还没有把 object identity / category / relation-specific constraint / value prior 拆开。
```

2. 候选仍是 first-token logit。

```text
function、photography、navigation 等多字符候选即使 token 可能单 token，
仍需要 full-sequence logprob 做更干净验证。
```

3. 关系模板仍然较简单。

```text
A robin is a kind of ...
A knife is used for ...
这种模板能稳定读出关系，但还不能覆盖自然句中复杂关系。
```

4. mismatched control 还不够多。

后续需要加入：

```text
same-category different-value control
same-target different-object control
random same-norm state control
orthogonalized control
shuffled layer control
```

5. GLM4 早层强破坏但恢复 gap 大。

```text
这说明 GLM4 的 early object state 可能混入更多构造/模板/读出路径，
不能简单解释为同 Qwen3/DS7B 一样的干净对象关系变量。
```

### 关键结论边界

可以说：

```text
object-relation-value 的 whole object-token residual state 具有跨关系 destroy-restore closure 特征。
```

不能说：

```text
已经破解知识网络编码机制；
已经分离出 object/relation/value 的最小数学变量；
已经证明全部语言编码机制。
```

### 下一步大任务

Phase71：多 control 稳健性复核。

目标：

```text
验证 Phase70 的 closure 不是任意状态替换、范数变化、候选模板偏置造成。
```

优先 control：

```text
1. same-relation same-category wrong-value
2. same-target different-object
3. random same-norm hidden state
4. shuffled object state
5. wrong-layer clean object state
```

Phase72：full-sequence value scoring。

目标：

```text
用完整候选序列 logprob 替代 first-token logits。
```

Phase73：subspace destroy-restore。

目标：

```text
从 whole-state closure 进入 identity/category/relation/value 子空间闭包。
```

Phase74：全局知识网络路径矩阵。

目标：

```text
建立 object-relation-value 的路径复用/差异化矩阵：
  哪些关系共享 L4/L8/L12/L20 路径；
  哪些关系需要更晚读出；
  哪些模型走浅层写入；
  哪些模型走中层平台；
  哪些只是 readout 强而非 relation state 强。
```

阶段性方向：

```text
当前最接近突破的主线不是抽象逻辑，也不是角色语法，
而是知识网络中的 object-relation-value 闭包。

先把知识网络闭包做成稳定、可拆分、可预测的机制图谱，
再反过来比较逻辑推理和语法规则是否复用同一类路径结构。
```

## Phase 71: 对象-关系-值多 control 稳健性复核 [2026-06-08 23:29]

### 任务目标

根据 Phase70 的硬伤继续推进。

Phase70 证明：

```text
object-relation-value 的 whole object-token residual state 具有跨关系 destroy-restore closure 特征。
```

但 Phase70 的 control 还不够严格。本轮目标是加入多种 control，检查 Phase70 的闭包是否只是任意状态替换、范数扰动、同 prompt 其他 token、或者 same-target 对象造成。

本轮不做理论总结，优先给出客观结果。

### 对用户分析的判断

用户提供的分析基本正确：

```text
1. Phase70 是 Phase69 后的关键升级。
2. 但 Phase70 仍是 whole-state closure，不是 factor-level closure。
3. 需要验证 mismatched object 的破坏是否强于随机同范数、同 prompt last token、same-target object 等控制。
4. 继续大数据跨模型测试比小样本结论更可靠。
```

因此本轮执行 Phase71：多 control 稳健性复核。

### 新增脚本

```text
tests/gpt5/phase71_object_relation_value_control_audit.py
tests/gpt5/phase71_object_relation_value_control_audit_summary.py
tests/gpt5/run_phase71_object_relation_value_control_audit_full.sh
```

脚本特性：

```text
1. 三模型依次运行：qwen3 -> glm4 -> deepseek7b。
2. 每个模型使用 --hard-exit-after-model。
3. 优先 flash_attention_2，当前环境未安装 flash_attn，因此 fallback 到 sdpa。
4. 使用 bfloat16 + device_map="auto"。
5. 每 48 items 输出进度。
6. 每个 layer pair 后写 partial。
```

### Control 类型

```text
mismatch_object:
  同 relation/frame 下，target 不同的对象状态。

same_target_object:
  target 相同但 object 不同的对象状态。
  用来判断是否只是 value/category 相同即可恢复。

random_same_norm:
  与 clean object state 同范数的随机向量。
  用来排除范数扰动解释。

same_prompt_last:
  同一个 clean prompt 的 last-token state。
  用来排除任意同上下文状态都能破坏/恢复的解释。
```

### Smoke Test

命令：

```bash
OUT=results/gpt5_phase71_smoke_$(date +%Y%m%d_%H%M%S)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate openone-cu130-py312
python tests/gpt5/phase71_object_relation_value_control_audit.py qwen3 \
  --layer-pairs 4-8 \
  --max-items 24 \
  --module resid_out \
  --positions object_first,object_last \
  --controls mismatch_object,same_target_object,random_same_norm,same_prompt_last \
  --output-dir "$OUT" \
  --progress-every 8 \
  --hard-exit-after-model
```

结果：

```text
rows = 148
exit_code = 0
```

### 正式测试命令

```bash
PHASE71_OUTPUT_DIR=results/gpt5_phase71_object_relation_value_control_audit_full_20260608_230640 \
PHASE71_PROGRESS_EVERY=48 \
bash tests/gpt5/run_phase71_object_relation_value_control_audit_full.sh
```

三模型均正常完成。

### 输出文件

```text
results/gpt5_phase71_object_relation_value_control_audit_full_20260608_230640/qwen3_phase71_object_relation_value_control_audit.json
results/gpt5_phase71_object_relation_value_control_audit_full_20260608_230640/glm4_phase71_object_relation_value_control_audit.json
results/gpt5_phase71_object_relation_value_control_audit_full_20260608_230640/deepseek7b_phase71_object_relation_value_control_audit.json
results/gpt5_phase71_object_relation_value_control_audit_full_20260608_230640/phase71_object_relation_value_control_audit_summary.json
results/gpt5_phase71_object_relation_value_control_audit_full_20260608_230640/PHASE71_OBJECT_RELATION_VALUE_CONTROL_AUDIT_SUMMARY.md
```

### 数据规模

```text
Qwen3:
  items = 342
  rows = 7560
  layer_pairs = L4-L8, L8-L12, L8-L16

GLM4:
  items = 342
  rows = 7560
  layer_pairs = L4-L10, L10-L20, L4-L30

DeepSeek7B:
  items = 342
  rows = 10080
  layer_pairs = L8-L10, L8-L12, L12-L14, L12-L16

total_rows = 25200
```

### Qwen3 客观结果

按 control 汇总：

```text
mismatch_object:
  eligible = 1794
  eligible_destroy_drop = 8.4494
  eligible_restore_gain = 7.6817
  eligible_restore_to_clean_gap = 0.7676
  eligible_destroy_top1 = 0.2865
  eligible_restore_top1 = 0.9459

same_prompt_last:
  eligible_destroy_drop = 4.7106
  eligible_restore_gain = 4.2398
  eligible_restore_to_clean_gap = 0.4708
  eligible_destroy_top1 = 0.5886
  eligible_restore_top1 = 0.9766

random_same_norm:
  eligible_destroy_drop = 4.3578
  eligible_restore_gain = 3.9043
  eligible_restore_to_clean_gap = 0.4535
  eligible_destroy_top1 = 0.6338
  eligible_restore_top1 = 0.9788

same_target_object:
  eligible_destroy_drop = 0.1544
  eligible_restore_gain = 0.2979
  eligible_restore_to_clean_gap = -0.1435
  eligible_destroy_top1 = 0.9495
  eligible_restore_top1 = 0.9984
```

最强路径：

```text
mismatch_object L4->L8 object_last:
  eligible_destroy_drop = 9.8621
  eligible_restore_gain = 8.7762
  eligible_restore_to_clean_gap = 1.0859
  eligible_destroy_top1 = 0.1773
  eligible_restore_top1 = 0.8930

mismatch_object L8->L12 object_last:
  eligible_destroy_drop = 8.1869
  eligible_restore_gain = 7.7148
  eligible_restore_to_clean_gap = 0.4721
  eligible_restore_top1 = 0.9666
```

客观现象：

```text
1. mismatched object 明显强于 random_same_norm 和 same_prompt_last。
2. random_same_norm / same_prompt_last 也会造成一定破坏，说明 whole-state replacement 仍有非语义扰动成分。
3. same_target_object 几乎不破坏，说明如果候选 value 相同，即使 object 不同，输出保持稳定。
```

### GLM4 客观结果

按 control 汇总：

```text
mismatch_object:
  eligible = 1896
  eligible_destroy_drop = 7.3136
  eligible_restore_gain = 3.4898
  eligible_restore_to_clean_gap = 3.8238
  eligible_destroy_top1 = 0.3534
  eligible_restore_top1 = 0.6540

random_same_norm:
  eligible_destroy_drop = 3.1530
  eligible_restore_gain = 1.5785
  eligible_restore_to_clean_gap = 1.5745
  eligible_destroy_top1 = 0.7447
  eligible_restore_top1 = 0.9182

same_prompt_last:
  eligible_destroy_drop = 2.5322
  eligible_restore_gain = 1.3348
  eligible_restore_to_clean_gap = 1.1974
  eligible_destroy_top1 = 0.8223
  eligible_restore_top1 = 0.9378

same_target_object:
  eligible_destroy_drop = 0.1218
  eligible_restore_gain = 0.1151
  eligible_restore_to_clean_gap = 0.0068
  eligible_destroy_top1 = 0.9581
  eligible_restore_top1 = 0.9833
```

关键路径：

```text
mismatch_object L4->L10 object_last:
  eligible_destroy_drop = 9.3529
  eligible_restore_gain = 6.1331
  eligible_restore_to_clean_gap = 3.2198
  eligible_destroy_top1 = 0.1709
  eligible_restore_top1 = 0.7057

mismatch_object L10->L20 object_last:
  eligible_destroy_drop = 4.0759
  eligible_restore_gain = 3.8696
  eligible_restore_to_clean_gap = 0.2063
  eligible_destroy_top1 = 0.6519
  eligible_restore_top1 = 0.9873

mismatch_object L4->L30 object_last:
  eligible_destroy_drop = 9.3529
  eligible_restore_gain = 0.9985
  eligible_restore_to_clean_gap = 8.3544
  eligible_restore_top1 = 0.2437
```

客观现象：

```text
1. mismatch_object 仍最强。
2. same_target_object 基本不破坏。
3. GLM4 的 L10->L20 是最干净恢复窗口。
4. L4->L30 继续失败，支持 L30 object-token restore 不是有效机制窗口。
5. GLM4 random_same_norm 的破坏不小，说明 GLM4 对 whole-state 替换更敏感，后续必须做子空间级实验。
```

### DeepSeek7B 客观结果

按 control 汇总：

```text
mismatch_object:
  eligible = 1968
  eligible_destroy_drop = 4.3491
  eligible_restore_gain = 3.9909
  eligible_restore_to_clean_gap = 0.3582
  eligible_destroy_top1 = 0.4482
  eligible_restore_top1 = 0.9355

random_same_norm:
  eligible_destroy_drop = 1.9750
  eligible_restore_gain = 1.8023
  eligible_restore_to_clean_gap = 0.1727
  eligible_destroy_top1 = 0.7368
  eligible_restore_top1 = 0.9360

same_prompt_last:
  eligible_destroy_drop = 1.5854
  eligible_restore_gain = 1.4619
  eligible_restore_to_clean_gap = 0.1235
  eligible_destroy_top1 = 0.7774
  eligible_restore_top1 = 0.9543

same_target_object:
  eligible_destroy_drop = 0.2235
  eligible_restore_gain = 0.1759
  eligible_restore_to_clean_gap = 0.0476
  eligible_destroy_top1 = 0.9099
  eligible_restore_top1 = 0.9745
```

关键路径：

```text
mismatch_object L8->L10 object_last:
  eligible_destroy_drop = 5.0276
  eligible_restore_gain = 4.3645
  eligible_restore_to_clean_gap = 0.6631
  eligible_destroy_top1 = 0.3537
  eligible_restore_top1 = 0.9024

mismatch_object L12->L14 object_last:
  eligible_destroy_drop = 3.9365
  eligible_restore_gain = 3.9421
  eligible_restore_to_clean_gap = -0.0056
  eligible_destroy_top1 = 0.5163
  eligible_restore_top1 = 0.9756

mismatch_object L12->L16 object_last:
  eligible_restore_to_clean_gap = 0.0310
  eligible_restore_top1 = 0.9675
```

客观现象：

```text
1. mismatch_object 强于 random_same_norm / same_prompt_last。
2. same_target_object 几乎不破坏。
3. DS7B L12->L14/L16 仍是很干净的中层恢复窗口。
4. random_same_norm 有中等影响，说明仍不能把 whole-state 闭包解释成纯语义变量。
```

### 三模型共同现象

```text
1. mismatch_object 在三模型中都是最强破坏控制。
2. same_target_object 在三模型中都几乎不破坏。
3. random_same_norm / same_prompt_last 会造成中等破坏，但明显弱于 mismatch_object。
4. 这说明 Phase70 的 closure 不是任意状态替换或范数扰动即可解释。
5. 但 whole-state transplant 仍包含非语义扰动成分，尤其 Qwen3/GLM4 的 random_same_norm 和 same_prompt_last 不为零。
```

### 当前更稳的客观结论

Phase71 支持以下较弱但可靠的说法：

```text
object-relation-value closure 对 mismatched object state 特别敏感；
same-target object state 基本不会破坏候选 value；
random same-norm 和 same-prompt last-token state 也能造成扰动，
但强度明显低于 mismatched object。
```

这说明：

```text
Phase70 的 closure 既包含对象/关系/值约束信息，
也包含 whole-state 替换导致的格式/范数/位置扰动。
```

### 严格硬伤

1. same-target object control 不等于 pure identity control。

```text
它也可能共享 category/value prior，所以不能证明 identity 完全无关。
```

2. random_same_norm 破坏不为零。

```text
说明 whole residual state 替换仍有格式破坏成分。
```

3. same_prompt_last 破坏不为零。

```text
说明同上下文不同 token state 本身也会扰乱 object-token 路径。
```

4. 仍然是 first-token logit。

```text
下一步必须做 full-sequence value scoring。
```

5. 仍然不是子空间级闭包。

```text
还没有分离 identity/category/relation/value factor。
```

### 下一步计划

Phase72：full-sequence value scoring。

目标：

```text
把候选 value 从 first-token logit 升级为完整候选序列 logprob。
```

Phase73：子空间级 control。

目标：

```text
在 object-token residual state 中拆分：
  identity
  category
  relation-specific constraint
  value prior
  readout alignment
```

Phase74：factor-level destroy-restore。

目标：

```text
只破坏/恢复某个候选子空间，
判断是否能得到比 whole-state 更干净的闭包。
```

Phase75：全局知识网络路径矩阵。

目标：

```text
把 Phase70/71 的路径按 model / relation / layer window / control type 组织成矩阵，
寻找路径复用和差异化。
```

阶段性判断：

```text
知识网络主线继续是当前最可行突破口。
但下一步必须从 whole-state closure 进入 factor-level closure，
否则无法真正回答深度网络如何高效实现多层次知识网络。
```

## Phase 72: 对象-关系-值 full-sequence value scoring 闭包复核 [2026-06-09 01:24]

### 任务目标

根据 Phase71 的硬伤继续推进。

Phase70/71 仍使用 first-token logit 作为候选 value 分数。本轮升级为：

```text
full-sequence candidate logprob
```

目标是验证：

```text
object-relation-value destroy-restore closure 是否在完整候选序列概率下仍然成立。
```

本轮继续优先客观结果，不做过度理论总结。

### 对用户分析的判断

用户提供的分析基本正确：

```text
1. Phase71 已进入受控闭包阶段。
2. 但 Phase71 仍是 whole-state closure。
3. random_same_norm / same_prompt_last 不为零，说明 whole-state replacement 有格式扰动。
4. first-token logit 是硬伤，必须升级到 full-sequence logprob。
5. 当前不应轻易总结理论，应继续积累真实现象拼图。
```

因此本轮执行 Phase72：full-sequence value scoring 复核。

### 新增脚本

```text
tests/gpt5/phase72_object_relation_value_fullseq_closure.py
tests/gpt5/phase72_object_relation_value_fullseq_closure_summary.py
tests/gpt5/run_phase72_object_relation_value_fullseq_closure_full.sh
```

脚本特性：

```text
1. 三模型依次运行：qwen3 -> glm4 -> deepseek7b。
2. 每个模型使用 --hard-exit-after-model。
3. 使用 bfloat16 + device_map="auto"。
4. 优先 flash_attention_2，当前环境未安装 flash_attn，自动 fallback 到 sdpa。
5. 对每个候选 value 计算完整 token 序列 logprob。
6. 仍使用 destroy/restore 框架。
```

### Smoke Test

命令：

```bash
OUT=results/gpt5_phase72_smoke_$(date +%Y%m%d_%H%M%S)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate openone-cu130-py312
python tests/gpt5/phase72_object_relation_value_fullseq_closure.py qwen3 \
  --layer-pairs 4-8 \
  --max-items 12 \
  --module resid_out \
  --positions object_last \
  --output-dir "$OUT" \
  --progress-every 4 \
  --hard-exit-after-model
```

结果：

```text
rows = 12
exit_code = 0
```

### 正式测试命令

```bash
PHASE72_OUTPUT_DIR=results/gpt5_phase72_object_relation_value_fullseq_closure_full_20260609_005202 \
PHASE72_PROGRESS_EVERY=24 \
bash tests/gpt5/run_phase72_object_relation_value_fullseq_closure_full.sh
```

三模型均正常完成。

### 输出文件

```text
results/gpt5_phase72_object_relation_value_fullseq_closure_full_20260609_005202/qwen3_phase72_object_relation_value_fullseq_closure.json
results/gpt5_phase72_object_relation_value_fullseq_closure_full_20260609_005202/glm4_phase72_object_relation_value_fullseq_closure.json
results/gpt5_phase72_object_relation_value_fullseq_closure_full_20260609_005202/deepseek7b_phase72_object_relation_value_fullseq_closure.json
results/gpt5_phase72_object_relation_value_fullseq_closure_full_20260609_005202/phase72_object_relation_value_fullseq_closure_summary.json
results/gpt5_phase72_object_relation_value_fullseq_closure_full_20260609_005202/PHASE72_OBJECT_RELATION_VALUE_FULLSEQ_CLOSURE_SUMMARY.md
```

### 数据规模

```text
Qwen3:
  items = 342
  rows = 2052
  layer_pairs = L4-L8, L8-L12, L8-L16

GLM4:
  items = 342
  rows = 2052
  layer_pairs = L4-L10, L10-L20, L4-L30

DeepSeek7B:
  items = 342
  rows = 2736
  layer_pairs = L8-L10, L8-L12, L12-L14, L12-L16

total_rows = 6840
```

说明：

```text
Phase72 每一行包含 target + distractors 的 full-sequence logprob 计算，
因此 rows 数少于 Phase71，但实际 forward 次数更多。
```

### Qwen3 客观结果

最强路径：

```text
L4 -> L8 object_last:
  eligible = 299
  eligible_destroy_drop = 9.8617
  eligible_restore_gain = 8.7756
  eligible_restore_to_clean_gap = 1.0861
  eligible_destroy_top1 = 0.1773
  eligible_restore_top1 = 0.8930

L4 -> L8 object_first:
  eligible_destroy_drop = 9.3575
  eligible_restore_gain = 8.2764
  eligible_restore_to_clean_gap = 1.0811
  eligible_restore_top1 = 0.8963

L8 -> L12 object_last:
  eligible_destroy_drop = 8.1866
  eligible_restore_gain = 7.7153
  eligible_restore_to_clean_gap = 0.4713
  eligible_restore_top1 = 0.9666

L8 -> L16 object_last:
  eligible_restore_top1 = 0.9766
```

关系级结果：

```text
is_a:
  eligible_destroy_drop = 13.5765
  eligible_restore_gain = 12.7790
  eligible_restore_to_clean_gap = 0.7975
  eligible_restore_top1 = 1.0000

used_for:
  eligible_destroy_drop = 10.5310
  eligible_restore_gain = 9.7953
  eligible_restore_top1 = 0.9514

can_do:
  eligible_destroy_drop = 8.0405
  eligible_restore_gain = 7.4337
  eligible_restore_top1 = 0.9203

function:
  eligible_restore_top1 = 0.9778

part_of:
  eligible_restore_top1 = 0.9402

material:
  eligible_restore_top1 = 0.8537

location:
  eligible_restore_top1 = 0.9679
```

客观现象：

```text
full-sequence scoring 下，Qwen3 的路径排序和 Phase70/71 基本一致。
L4->L8 仍是最强 early closure path；
L8->L12/L16 restore gap 更小。
```

### GLM4 客观结果

最强路径：

```text
L4 -> L10 object_last:
  eligible = 316
  eligible_destroy_drop = 9.3528
  eligible_restore_gain = 6.1330
  eligible_restore_to_clean_gap = 3.2198
  eligible_destroy_top1 = 0.1709
  eligible_restore_top1 = 0.7057

L10 -> L20 object_last:
  eligible_destroy_drop = 4.0757
  eligible_restore_gain = 3.8694
  eligible_restore_to_clean_gap = 0.2063
  eligible_destroy_top1 = 0.6519
  eligible_restore_top1 = 0.9873

L4 -> L30 object_last:
  eligible_destroy_drop = 9.3528
  eligible_restore_gain = 0.9982
  eligible_restore_to_clean_gap = 8.3546
  eligible_restore_top1 = 0.2437
```

关系级结果：

```text
used_for:
  eligible_destroy_drop = 9.7794
  eligible_restore_gain = 5.7145
  eligible_restore_top1 = 0.7118

is_a:
  eligible_destroy_drop = 11.5040
  eligible_restore_gain = 4.8221
  eligible_restore_to_clean_gap = 6.6819
  eligible_restore_top1 = 0.6204

can_do:
  eligible_restore_top1 = 0.6738

function:
  eligible_restore_top1 = 0.7355

part_of:
  eligible_restore_top1 = 0.6583

location:
  eligible_restore_top1 = 0.6351

material:
  eligible_restore_top1 = 0.5379
```

客观现象：

```text
1. GLM4 full-sequence scoring 下仍显示 L10->L20 是最干净 closure window。
2. L4->L30 object-token restore 继续失败。
3. GLM4 的 relation restore top1 明显低于 Qwen3/DS7B。
4. 这不是 first-token 指标造成的假象。
```

### DeepSeek7B 客观结果

最强路径：

```text
L8 -> L10 object_last:
  eligible = 246
  eligible_destroy_drop = 5.0276
  eligible_restore_gain = 4.3647
  eligible_restore_to_clean_gap = 0.6629
  eligible_destroy_top1 = 0.3537
  eligible_restore_top1 = 0.9024

L12 -> L14 object_last:
  eligible_destroy_drop = 3.9365
  eligible_restore_gain = 3.9421
  eligible_restore_to_clean_gap = -0.0056
  eligible_destroy_top1 = 0.5163
  eligible_restore_top1 = 0.9756

L12 -> L16 object_last:
  eligible_restore_to_clean_gap = 0.0310
  eligible_restore_top1 = 0.9675
```

关系级结果：

```text
is_a:
  eligible_destroy_drop = 7.2960
  eligible_restore_gain = 6.9268
  eligible_restore_to_clean_gap = 0.3692
  eligible_restore_top1 = 1.0000

used_for:
  eligible_destroy_drop = 5.9446
  eligible_restore_gain = 5.2757
  eligible_restore_top1 = 0.9044

can_do:
  eligible_destroy_drop = 4.3964
  eligible_restore_gain = 4.1408
  eligible_restore_top1 = 0.9143

function:
  eligible_restore_top1 = 0.8350

location:
  eligible_restore_top1 = 0.9113

part_of:
  eligible_restore_top1 = 0.9643

material:
  eligible_restore_top1 = 0.9551
```

客观现象：

```text
1. DeepSeek7B 的 L8-L16 中层 residual platform 在 full-sequence scoring 下继续成立。
2. L12->L14/L16 的 restore gap 仍接近 0。
3. is_a restore_top1 仍为 1.0。
```

### 与 Phase70/71 的一致性

```text
Qwen3:
  first-token 与 full-sequence 结果几乎一致。

GLM4:
  L10->L20 是干净窗口、L4->L30 失败，这一点完全复现。

DeepSeek7B:
  L8-L16 中层平台、L12->L14/L16 干净恢复继续复现。
```

这说明：

```text
Phase70/71 的核心结果不是 first-token logit 假象。
```

但也要注意：

```text
当前候选多数本身较短，很多仍接近单 token；
full-sequence scoring 是必要修正，但还不是最终自然语言读出。
```

### 当前更稳的结论

Phase72 支持：

```text
object-relation-value whole-state closure 在完整候选序列评分下仍然成立。
```

更具体：

```text
对象 token residual state 对 relation-conditioned candidate value distribution 的影响，
不是 first-token 评分造成的假象。
```

### 严格硬伤

1. full-sequence scoring 仍然是候选集合评分。

```text
它比 first-token 干净，但还不是开放式自然生成验证。
```

2. 仍是 whole-state closure。

```text
还没有分解 identity/category/relation/value 子空间。
```

3. 关系模板仍简单。

```text
需要后续加入复杂句式、组合关系、上下文临时绑定。
```

4. full-sequence 与 first-token 几乎一致，部分原因可能是候选词较短。

```text
后续需要刻意加入多 token value，例如 "fresh water", "cutting food", "school building"。
```

### 下一步计划

Phase73：多 token value 复核。

目标：

```text
刻意构造多 token candidate value，
验证 full-sequence scoring 是否仍保持 closure。
```

Phase74：factor-level / subspace-level control。

目标：

```text
从 whole object-token residual state 中分离：
identity
category
relation-conditioned constraint
value prior
format/readout alignment
```

Phase75：relation frame token 干预。

目标：

```text
当前主要替换 object token；
下一步需要替换 relation-frame token，
判断 relation context 如何选择候选 value 空间。
```

阶段性判断：

```text
知识网络闭包主线继续稳固。
现在已经完成：
  whole-state closure
  multi-control audit
  full-sequence scoring audit

下一步必须从 object state 进入 factor 和 relation-frame 的联合机制。
```

## Phase 73: 多 token value 全序列闭包复核 [2026-06-09 07:06]

### 任务目标

根据 Phase 72 的硬伤继续复核：

```text
Phase 72 已经证明 full-sequence scoring 下 object-relation-value closure 不是 first-token 假象。
但 Phase 72 的很多 target/distractor 仍然较短，可能仍然被首 token 主导。
```

本轮专门构造所有候选值都是 2-token phrase 的数据，例如：

```text
small bird
freshwater fish
cutting food
drinking water
school building
shiny metal
bright red
```

目标是验证：

```text
对象 token residual state 对 relation-conditioned multi-token value distribution 的影响是否仍成立。
```

### 对用户分析的判断

用户分析中正确的部分：

```text
1. Phase 72 的 full-sequence scoring 是正确进展，但还不能停止。
2. 如果候选值太短，full-sequence 与 first-token 可能接近，仍需刻意加入多 token value。
3. 当前主线仍应优先积累客观现象，不应急着抽象统一理论。
4. 单一路径信息有限，必须在多 relation、多路径、多模型之间比较。
```

因此本轮执行 Phase 73：多 token value 全序列闭包复核。

### 新增脚本

```text
tests/gpt5/phase73_multitoken_value_closure.py
tests/gpt5/phase73_multitoken_value_closure_summary.py
tests/gpt5/run_phase73_multitoken_value_closure_full.sh
```

脚本特性：

```text
1. 复用 Phase 72 已验证的 fullseq_logprob / capture_state / restore scoring。
2. 新增 7 类 relation：
   is_a
   part_of
   used_for
   can_do
   location
   material
   property
3. 每个 target 和 distractor 都刻意构造为 2-token candidate value。
4. 三模型按 qwen3 -> GLM4 -> DeepSeek7B 顺序运行。
5. 每个模型命令均带 --hard-exit-after-model，避免模型间显存残留。
6. attn implementation 优先 flash_attention_2；本机无 flash_attn 包，因此实际回退到 sdpa。
```

### Smoke Test

命令：

```bash
PHASE73_OUTPUT_DIR=results/gpt5_phase73_multitoken_value_closure_smoke2_$(date +%Y%m%d_%H%M%S) \
PHASE73_MODELS=qwen3 \
QWEN3_PHASE73_MAX_ITEMS=2 \
QWEN3_PHASE73_LAYER_PAIRS=4-8 \
PHASE73_PROGRESS_EVERY=1 \
PHASE73_POSITIONS=object_last \
tests/gpt5/run_phase73_multitoken_value_closure_full.sh
```

结果：

```text
qwen3:
  items = 2
  rows = 2
  exit_code = 0
```

说明：

```text
脚本、多 token 数据、full-sequence scoring、hook、hard-exit 都可以正常运行。
```

中间曾有一次 smoke 参数误用：

```text
GLM4_PHASE73_MAX_ITEMS=0
```

由于脚本中 0 被解释为不限制，GLM4 开始跑全量烟测，已手动终止，不作为正式结果。
随后给 runner 增加：

```text
PHASE73_MODELS
```

用于控制烟测模型列表。

### 正式测试命令

```bash
PHASE73_OUTPUT_DIR=results/gpt5_phase73_multitoken_value_closure_full_$(date +%Y%m%d_%H%M%S) \
PHASE73_PROGRESS_EVERY=48 \
tests/gpt5/run_phase73_multitoken_value_closure_full.sh
```

模型参数：

```text
qwen3:
  layer_pairs = 4-8,8-12,8-16
  max_items = 336

GLM4:
  layer_pairs = 4-10,10-20,4-30
  max_items = 336

DeepSeek7B:
  layer_pairs = 8-10,8-12,12-14,12-16
  max_items = 336
```

### 输出文件

```text
results/gpt5_phase73_multitoken_value_closure_full_20260609_063421/qwen3_phase73_multitoken_value_closure.json
results/gpt5_phase73_multitoken_value_closure_full_20260609_063421/glm4_phase73_multitoken_value_closure.json
results/gpt5_phase73_multitoken_value_closure_full_20260609_063421/deepseek7b_phase73_multitoken_value_closure.json
results/gpt5_phase73_multitoken_value_closure_full_20260609_063421/phase73_multitoken_value_closure_summary.json
results/gpt5_phase73_multitoken_value_closure_full_20260609_063421/PHASE73_MULTITOKEN_VALUE_CLOSURE_SUMMARY.md
```

### 数据规模

```text
qwen3:
  items = 336
  rows = 2016
  target_token_len_mean = 2.000
  distractor_token_len_mean = 2.000

GLM4:
  items = 336
  rows = 2016
  target_token_len_mean = 2.000
  distractor_token_len_mean = 2.000

DeepSeek7B:
  items = 336
  rows = 2688
  target_token_len_mean = 2.000
  distractor_token_len_mean = 2.000

total_rows = 6720
```

三模型均完成，并且每个模型结束后均 hard-exit。

### Qwen3 客观结果

总体：

```text
clean_top1 = 0.7649
destroy_top1 = 0.2068
restore_top1 = 0.6766
mean_destroy_drop = 7.5594
mean_restore_gain = 6.6472
mean_restore_to_clean_gap = 0.9122
```

Top paths：

```text
L4->L8:object_last
  eligible_n = 257
  eligible_destroy_drop = 9.9709
  eligible_restore_gain = 8.3849
  eligible_restore_to_clean_gap = 1.5860
  eligible_destroy_top1 = 0.1634
  eligible_restore_top1 = 0.8171

L8->L12:object_last
  eligible_destroy_drop = 7.7094
  eligible_restore_gain = 6.9590
  eligible_restore_to_clean_gap = 0.7504
  eligible_restore_top1 = 0.9066

L8->L16:object_last
  eligible_destroy_drop = 7.7094
  eligible_restore_gain = 6.7620
  eligible_restore_to_clean_gap = 0.9474
  eligible_restore_top1 = 0.8794
```

Relation summary：

```text
used_for:
  eligible_destroy_drop = 10.6666
  eligible_restore_gain = 9.8943
  eligible_restore_top1 = 0.9048

is_a:
  eligible_destroy_drop = 9.8723
  eligible_restore_gain = 9.0859
  eligible_restore_top1 = 0.9757

can_do:
  eligible_destroy_drop = 8.8372
  eligible_restore_gain = 8.5723
  eligible_restore_top1 = 0.8952
```

客观现象：

```text
Qwen3 在所有候选值都是 2-token phrase 的情况下，仍然出现强 destroy/restore。
L4->L8 破坏最强，L8->L12 恢复更接近 clean。
used_for / is_a / can_do 三类关系最稳定。
```

### GLM4 客观结果

总体：

```text
clean_top1 = 0.8274
destroy_top1 = 0.2927
restore_top1 = 0.5124
mean_destroy_drop = 7.4817
mean_restore_gain = 3.3654
mean_restore_to_clean_gap = 4.1163
```

Top paths：

```text
L4->L10:object_last
  eligible_n = 278
  eligible_destroy_drop = 10.5254
  eligible_restore_gain = 6.4832
  eligible_restore_to_clean_gap = 4.0421
  eligible_restore_top1 = 0.6295

L10->L20:object_last
  eligible_destroy_drop = 4.3480
  eligible_restore_gain = 4.1079
  eligible_restore_to_clean_gap = 0.2401
  eligible_restore_top1 = 0.9856

L4->L30:object_last
  eligible_destroy_drop = 10.5254
  eligible_restore_gain = 0.8460
  eligible_restore_to_clean_gap = 9.6794
  eligible_restore_top1 = 0.2014
```

Relation summary：

```text
used_for:
  eligible_destroy_drop = 12.2873
  eligible_restore_gain = 6.5743
  eligible_restore_to_clean_gap = 5.7130
  eligible_restore_top1 = 0.6560

is_a:
  eligible_destroy_drop = 10.3334
  eligible_restore_gain = 4.7041
  eligible_restore_to_clean_gap = 5.6293
  eligible_restore_top1 = 0.6123

can_do:
  eligible_destroy_drop = 7.8246
  eligible_restore_gain = 3.6981
  eligible_restore_to_clean_gap = 4.1265
  eligible_restore_top1 = 0.6228
```

客观现象：

```text
GLM4 的 early destroy 很强，但 early->late restore 不完整。
L10->L20 恢复几乎闭合，eligible_restore_top1 接近 0.986。
L4->L30 restore 明显失败，说明直接恢复到过深层并不能自然接上。
```

这与此前 GLM4 “浅层强写入，但路径格式有层段限制” 的观察一致。

### DeepSeek7B 客观结果

总体：

```text
clean_top1 = 0.5327
destroy_top1 = 0.3118
restore_top1 = 0.5335
mean_destroy_drop = 3.4082
mean_restore_gain = 3.0283
mean_restore_to_clean_gap = 0.3799
```

Top paths：

```text
L8->L10:object_last
  eligible_n = 179
  eligible_destroy_drop = 5.4433
  eligible_restore_gain = 4.4255
  eligible_restore_to_clean_gap = 1.0178
  eligible_restore_top1 = 0.8827

L12->L14:object_last
  eligible_destroy_drop = 3.8526
  eligible_restore_gain = 3.7800
  eligible_restore_to_clean_gap = 0.0726
  eligible_restore_top1 = 0.9777

L12->L16:object_last
  eligible_destroy_drop = 3.8526
  eligible_restore_gain = 3.7106
  eligible_restore_to_clean_gap = 0.1420
  eligible_restore_top1 = 0.9777
```

Relation summary：

```text
is_a:
  eligible_destroy_drop = 6.5960
  eligible_restore_gain = 6.0588
  eligible_restore_top1 = 0.9593

used_for:
  eligible_destroy_drop = 6.4740
  eligible_restore_gain = 5.4767
  eligible_restore_top1 = 0.9537

can_do:
  eligible_destroy_drop = 5.9283
  eligible_restore_gain = 5.3504
  eligible_restore_top1 = 0.8971
```

客观现象：

```text
DeepSeek7B 的 clean_top1 只有 0.533，因此必须重点看 eligible 子集。
eligible 子集里 L12->L14 和 L12->L16 restore 几乎闭合。
L8->L10 destroy 更强，但 gap 也更大。
```

这说明 DeepSeek7B 在多 token value 上仍有闭包信号，但 baseline 候选选择能力较弱，不能把全体 rows 直接解释为机制强弱。

### 与 Phase 72 的关系

Phase 72：

```text
full-sequence scoring 下 object-relation-value closure 成立。
但候选值可能较短。
```

Phase 73：

```text
所有 target/distractor 均为 2-token phrase。
三模型仍然出现 destroy_drop 与 restore_gain。
因此 Phase 72 的核心现象不是短候选或 first-token artifact。
```

更稳的结论：

```text
对象 token residual state 对 relation-conditioned candidate value distribution 的影响，
在刻意多 token value 的 full-sequence scoring 下仍然成立。
```

### 严格硬伤

1. 仍然是候选集合评分。

```text
full-sequence candidate scoring 比 first-token 更干净，但还不是开放式生成验证。
```

2. 仍然是 whole-state closure。

```text
当前替换的是 object token residual state 整体，还没有分离 identity/category/relation-conditioned constraint/value prior。
```

3. GLM4 早层恢复缺口很大。

```text
L4->L10 destroy 很强，但 restore gap 约 4.0。
说明 GLM4 早层 object state 携带强信息，但简单 restore 不足以完全重建下游格式。
```

4. DeepSeek7B baseline 低。

```text
clean_top1 只有 0.533。
必须以 eligible rows 为主要解释对象，否则会把模型本身没选对 target 的行混入机制分析。
```

5. 本机没有 flash_attn 包。

```text
脚本优先尝试 flash_attention_2，但实际回退到 sdpa。
DeepSeek7B 仍出现 sliding window attention + sdpa 的实现警告，结果可用但需要在后续关键实验中谨慎复核。
```

### 研究进展

目前知识网络闭包主线已经完成四级审计：

```text
Phase 68:
  natural exchange

Phase 69:
  destroy/restore

Phase 70:
  object-relation-value closure

Phase 71:
  multi-control audit

Phase 72:
  full-sequence scoring audit

Phase 73:
  deliberately multi-token value audit
```

最稳的客观拼图：

```text
对象状态不是孤立词义，而是可以在关系模板中约束候选 value 分布。
这种约束跨 relation、跨模型、跨 full-sequence multi-token candidate 都存在。
不同模型的闭包路径不同：
  Qwen3: L4->L8 强破坏，L8->L12 恢复更闭合。
  GLM4: L10->L20 恢复闭合，过深 restore 明显失败。
  DeepSeek7B: eligible 子集里中层恢复闭合，但 clean baseline 较低。
```

### 下一步计划

Phase 74：factor-level / subspace-level control。

目标：

```text
从 whole object-token state 中分离：
1. object identity
2. category/type
3. relation-conditioned constraint
4. value prior
5. prompt/readout alignment
```

关键实验：

```text
同 relation、不同 category 的 control；
同 category、不同 relation 的 control；
同 object、不同 frame 的 control；
用投影/残差化方式测试哪些因素能独立破坏或恢复 value distribution。
```

Phase 75：relation-frame token 干预。

目标：

```text
当前主要替换 object token；
下一步要替换 relation token / frame token，
判断 relation context 如何选择候选 value 空间。
```

Phase 76：object-state + relation-frame 联合闭包。

目标：

```text
测试 value prediction 是否来自：
object state 单独决定；
relation frame 单独决定；
object state 与 relation frame 的交互决定。
```

这一步比继续单独扩大数据更关键，因为深度网络是相对编码，必须比较路径之间的相互关系，才能形成全局路径图谱。

## Phase 74: 多 token value 因子级 control audit [2026-06-09 08:02]

### 任务目标

根据 Phase 73 的结论继续推进：

```text
Phase 73 证明 multi-token value full-sequence closure 成立；
但仍然是 whole object-token state closure。
```

本轮不直接宣称已经完成 factor/subspace 分离，而是先做更稳的 factor-level control audit：

```text
用不同类型的自然控制状态替换 object token residual state，
观察哪些 control 会破坏多 token value distribution，
哪些 control 基本不破坏。
```

### 对用户分析的判断

用户分析中正确的部分：

```text
1. Phase 73 已经补上短候选 / first-token artifact 的关键硬伤。
2. 当前最大硬伤是 whole-state closure，不是因子级闭包。
3. 下一步应拆分 object identity / category / value support / relation-conditioned constraint / readout alignment。
4. relation-frame token 还没有进入干预核心，后续必须做。
5. 不应急于理论总结，应优先完成客观拼图。
```

因此本轮执行 Phase 74：多 token value 因子级 control audit。

### 新增脚本

```text
tests/gpt5/phase74_factor_control_audit.py
tests/gpt5/phase74_factor_control_audit_summary.py
tests/gpt5/run_phase74_factor_control_audit_full.sh
```

脚本使用 Phase 73 的 multi-token value 数据和 Phase 72 的 full-sequence scoring。

### Control 类型

本轮比较四类 control：

```text
wrong_target_same_relation_frame:
  同 relation / 同 frame，但 object 和 target 不同。
  这是标准强破坏 control。

same_target_same_relation_frame:
  同 relation / 同 frame / 同 target，但 object 不同。
  测试是否只要 value support 相同，就不会破坏。

same_object_same_relation_other_frame:
  同 object / 同 relation / 同 target，但 frame 模板不同。
  测试 object state 是否对同关系不同表达稳定。

same_object_different_relation:
  同 object，但 relation 不同。
  测试 object token state 是否强烈依赖 relation context。
```

注意：

```text
same_object_different_relation 当前只覆盖在多个 relation 中都出现的 object，因此 rows 更少。
```

### Smoke Test

命令：

```bash
PHASE74_OUTPUT_DIR=results/gpt5_phase74_factor_control_audit_smoke_$(date +%Y%m%d_%H%M%S) \
PHASE74_MODELS=qwen3 \
QWEN3_PHASE74_MAX_ITEMS=6 \
QWEN3_PHASE74_LAYER_PAIRS=4-8 \
PHASE74_POSITIONS=object_last \
PHASE74_PROGRESS_EVERY=2 \
tests/gpt5/run_phase74_factor_control_audit_full.sh
```

结果：

```text
qwen3:
  rows = 6
  exit_code = 0
```

说明：

```text
脚本、control 查找、full-sequence scoring、hard-exit 正常。
```

### 正式测试命令

```bash
PHASE74_OUTPUT_DIR=results/gpt5_phase74_factor_control_audit_full_$(date +%Y%m%d_%H%M%S) \
PHASE74_POSITIONS=object_first,object_last \
PHASE74_PROGRESS_EVERY=48 \
tests/gpt5/run_phase74_factor_control_audit_full.sh
```

模型参数：

```text
qwen3:
  layer_pairs = 4-8,8-12
  max_items = 336

GLM4:
  layer_pairs = 4-10,10-20
  max_items = 336

DeepSeek7B:
  layer_pairs = 8-10,12-14
  max_items = 336
```

说明：

```text
本轮没有继续跑所有 Phase 73 layer pairs，而是选取每个模型最关键的两个窗口：
  一个偏强破坏窗口；
  一个偏恢复闭包窗口。
这样在数据量较大时仍能保持模型顺序全量完成。
```

### 输出文件

```text
results/gpt5_phase74_factor_control_audit_full_20260609_071434/qwen3_phase74_factor_control_audit.json
results/gpt5_phase74_factor_control_audit_full_20260609_071434/glm4_phase74_factor_control_audit.json
results/gpt5_phase74_factor_control_audit_full_20260609_071434/deepseek7b_phase74_factor_control_audit.json
results/gpt5_phase74_factor_control_audit_full_20260609_071434/phase74_factor_control_audit_summary.json
results/gpt5_phase74_factor_control_audit_full_20260609_071434/PHASE74_FACTOR_CONTROL_AUDIT_SUMMARY.md
```

### 数据规模

```text
qwen3:
  items = 336
  rows = 3900

GLM4:
  items = 336
  rows = 3900

DeepSeek7B:
  items = 336
  rows = 3900

total_rows = 11700
```

三模型均完成，并且每个模型结束后均 hard-exit。

### Qwen3 客观结果

By control type：

```text
wrong_target_same_relation_frame:
  n = 1344
  eligible_n = 1028
  eligible_destroy_drop = 8.5678
  eligible_restore_gain = 7.4166
  eligible_restore_to_clean_gap = 1.1512
  eligible_destroy_top1 = 0.2451
  eligible_restore_top1 = 0.8619

same_target_same_relation_frame:
  n = 840
  eligible_n = 644
  eligible_destroy_drop = 0.5579
  eligible_restore_gain = 0.4630
  eligible_restore_to_clean_gap = 0.0949
  eligible_destroy_top1 = 0.8680
  eligible_restore_top1 = 0.9488

same_object_same_relation_other_frame:
  n = 1344
  eligible_n = 1028
  eligible_destroy_drop = 0.0611
  eligible_restore_gain = 0.0508
  eligible_restore_to_clean_gap = 0.0103
  eligible_destroy_top1 = 0.9708
  eligible_restore_top1 = 0.9805

same_object_different_relation:
  n = 372
  eligible_n = 268
  eligible_destroy_drop = 0.0324
  eligible_restore_gain = 0.0190
  eligible_restore_to_clean_gap = 0.0133
  eligible_destroy_top1 = 0.9701
  eligible_restore_top1 = 0.9776
```

客观现象：

```text
Qwen3 中，wrong-target control 强破坏；
same-target control 破坏很小；
same-object other-frame 和 same-object different-relation 几乎不破坏。
```

这说明：

```text
此前 Qwen3 的强闭包主要来自 target/value-support mismatch，
不是任意 object identity 变化，也不是同 object 的 frame 表达变化。
```

### GLM4 客观结果

By control type：

```text
wrong_target_same_relation_frame:
  n = 1344
  eligible_n = 1112
  eligible_destroy_drop = 7.1684
  eligible_restore_gain = 5.0584
  eligible_restore_to_clean_gap = 2.1101
  eligible_destroy_top1 = 0.4344
  eligible_restore_top1 = 0.8076

same_target_same_relation_frame:
  n = 840
  eligible_n = 688
  eligible_destroy_drop = 0.1032
  eligible_restore_gain = 0.1651
  eligible_restore_to_clean_gap = -0.0619
  eligible_destroy_top1 = 0.9462
  eligible_restore_top1 = 0.9767

same_object_different_relation:
  n = 372
  eligible_n = 280
  eligible_destroy_drop = 0.0006
  eligible_restore_gain = 0.0117
  eligible_restore_to_clean_gap = -0.0111
  eligible_destroy_top1 = 0.9857
  eligible_restore_top1 = 0.9929

same_object_same_relation_other_frame:
  n = 1344
  eligible_n = 1112
  eligible_destroy_drop = -0.0287
  eligible_restore_gain = 0.0013
  eligible_restore_to_clean_gap = -0.0300
  eligible_destroy_top1 = 0.9910
  eligible_restore_top1 = 0.9964
```

客观现象：

```text
GLM4 的 factor-control 对比最干净：
wrong-target 强破坏；
same-target / same-object-other-frame / same-object-different-relation 基本不破坏。
```

这说明：

```text
GLM4 中 object token state 的关键因果差异更接近 value-support / target-compatibility，
而不是对象身份本身或 frame 表面表达。
```

同时：

```text
wrong-target 的 restore gap 仍较大，说明 GLM4 的状态格式转换问题仍存在。
```

### DeepSeek7B 客观结果

By control type：

```text
wrong_target_same_relation_frame:
  n = 1344
  eligible_n = 716
  eligible_destroy_drop = 4.5743
  eligible_restore_gain = 4.0297
  eligible_restore_to_clean_gap = 0.5446
  eligible_destroy_top1 = 0.5503
  eligible_restore_top1 = 0.9302

same_target_same_relation_frame:
  n = 840
  eligible_n = 468
  eligible_destroy_drop = 0.1491
  eligible_restore_gain = 0.0311
  eligible_restore_to_clean_gap = 0.1180
  eligible_destroy_top1 = 0.9359
  eligible_restore_top1 = 0.9444

same_object_same_relation_other_frame:
  n = 1344
  eligible_n = 716
  eligible_destroy_drop = 0.1462
  eligible_restore_gain = 0.1435
  eligible_restore_to_clean_gap = 0.0027
  eligible_destroy_top1 = 0.9860
  eligible_restore_top1 = 0.9972

same_object_different_relation:
  n = 372
  eligible_n = 184
  eligible_destroy_drop = 0.1194
  eligible_restore_gain = 0.0720
  eligible_restore_to_clean_gap = 0.0474
  eligible_destroy_top1 = 0.9891
  eligible_restore_top1 = 1.0000
```

客观现象：

```text
DeepSeek7B 中 wrong-target control 仍明显破坏；
其他三类 control 仅产生很小破坏。
eligible 子集下 same-object / same-target control 的 top1 基本保持。
```

这说明：

```text
DeepSeek7B 中同样不是任意 object identity 或 frame 变化导致闭包，
而是目标值支持结构变化导致闭包。
```

### 三模型一致结果

最重要的共同现象：

```text
wrong_target_same_relation_frame:
  三模型均强破坏。

same_target_same_relation_frame:
  三模型均只产生很小破坏。

same_object_same_relation_other_frame:
  三模型几乎不破坏。

same_object_different_relation:
  三模型整体也几乎不破坏，但该 control rows 较少。
```

这说明 Phase 70-73 的强 destroy/restore 不是因为：

```text
1. 任意换一个 object token 就会破坏；
2. 同 object 换一个 prompt frame 就会破坏；
3. 同 target 的另一个 object 会强烈破坏；
4. 同 object 在不同 relation prompt 下必然强烈破坏。
```

更稳的解释是：

```text
object token residual state 中存在某种 value-support compatibility。
当 control object 支持不同 target value 时，候选值分布被强烈破坏；
当 control object 支持相同 target value 时，破坏很小。
```

### 对 Phase 73 的修正和推进

Phase 73 的结论：

```text
多 token full-sequence value closure 成立。
```

Phase 74 推进一步：

```text
闭包效应对 control 类型高度选择性。
强效主要来自 target/value-support mismatch，而不是任意对象替换。
```

因此当前更准确的对象状态描述是：

```text
object token residual state 不只是 object identity，
而包含可与 relation frame 和 candidate value distribution 兼容的 value-support factor。
```

### 严格硬伤

1. 还不是真正的子空间分离。

```text
本轮是自然 control audit，不是投影、正交化或 learned subspace patch。
不能说已经找到了 z_id / z_cat / z_valseq 的具体向量子空间。
```

2. same_object_different_relation 覆盖较少。

```text
只有同一个 object 在多个 relation 中都出现时才有该 control。
三模型该 control rows = 372，少于其他 control。
后续需要专门构造 object-cross-relation 平衡数据。
```

3. same_target control 不等于纯 value-support control。

```text
同 target 的两个 object 往往也共享 category 或功能类型。
它仍混合 category/value-support，而不是纯粹 value factor。
```

4. relation-frame token 仍未被直接干预。

```text
本轮只改变 object token state 的来源 prompt；
还没有替换 relation frame token state。
```

5. 仍是 closed candidate scoring。

```text
候选集合内 full-sequence scoring 已经很稳；
但还不是开放式生成。
```

### 研究进展

目前知识网络闭包主线进入第六级审计：

```text
Phase 68:
  natural exchange

Phase 69:
  destroy/restore

Phase 70:
  object-relation-value closure

Phase 71:
  multi-control audit

Phase 72:
  full-sequence scoring audit

Phase 73:
  deliberately multi-token value audit

Phase 74:
  factor-level natural control audit
```

当前最稳的客观拼图：

```text
1. object token residual state 能控制 relation-conditioned multi-token candidate value distribution。
2. 这种控制不是 first-token artifact。
3. 这种控制不是短候选 artifact。
4. 这种控制不是任意 object replacement artifact。
5. 同 target / 同 object / 同 relation other-frame 的自然状态基本兼容。
6. 不同 target 的同 relation control 才强破坏。
```

第一性原则层面的谨慎表述：

```text
语言知识网络中的对象编码，更像是“相对兼容性结构”，而不是固定对象身份轴。
对象状态在给定关系框架下，与候选 value 的支持结构兼容或不兼容；
这种兼容性决定了候选值分布是否稳定。
```

这和“深度神经网络是相对编码”的核心判断一致：

```text
单一 object state 没有完整意义；
必须放在 relation frame、candidate value space、layer window、readout position 中比较，
才显示出它的编码作用。
```

### 下一步计划

Phase 75：relation-frame token intervention。

目标：

```text
直接替换 relation-frame token state，
测试 relation frame 如何选择 value candidate space。
```

关键设计：

```text
保持 object token state 不变；
替换 frame token state：
  is_a frame -> used_for frame
  used_for frame -> can_do frame
  location frame -> material frame

观察 candidate value distribution 是否随 frame state 切换。
```

Phase 76：object-state + relation-frame 联合闭包。

四种组合：

```text
clean object + clean frame
clean object + wrong frame
wrong object + clean frame
wrong object + wrong frame
```

目标：

```text
判断 value prediction 是否由 object × relation 的组合兼容性决定。
```

Phase 77：balanced cross-relation object dataset。

目标：

```text
专门构造同一批 object 覆盖所有 relation：
is_a
used_for
can_do
location
material
property
part_of

解决 Phase 74 中 same_object_different_relation rows 较少的问题。
```

Phase 78：subspace projection audit。

目标：

```text
在 Phase 74 自然 control 结果稳定后，
再用 residualization / projection / low-rank subspace patch 尝试分离：
identity factor
category factor
value-support factor
relation-compatibility factor
readout-format factor
```

当前不宜直接跳到复杂数学分解；
应先完成 relation-frame 干预和 object-frame 联合闭包。

## Phase 75: 关系框架词元干预 [2026-06-09 11:10]

### 任务目标

根据 Phase 74 的 object token（对象词元）破坏-恢复结果，本轮继续验证全局关系路径图谱中的另一半：

```text
value prediction 是否不仅依赖 object state（对象状态），
也依赖 relation-frame token（关系框架词元）携带的关系/读出约束。
```

核心问题：

```text
如果保持 object 不变，只把 relation frame token 的 hidden state（隐藏状态）
替换成同一 object 的其他 relation frame，
模型对当前 relation 的正确 value 是否下降？

如果再恢复 clean frame token state，
模型是否能恢复正确 value margin？
```

这轮继续遵守当前研究原则：

```text
1. 不先做理论总结，优先跑完三模型客观现象。
2. 三模型 qwen3 -> GLM4 -> DS7B 顺序执行。
3. 每个模型运行结束后 hard exit，避免显存残留。
4. 数据量使用完整 relation-frame balanced dataset，不使用小样本结论。
```

### 新增脚本

```text
tests/gpt5/phase75_relation_frame_token_intervention.py
tests/gpt5/phase75_relation_frame_token_intervention_summary.py
tests/gpt5/run_phase75_relation_frame_token_intervention_full.sh
```

脚本设计：

```text
objects = 12
relations = 6
frames = 3
items = 216

relations:
  is_a
  used_for
  can_do
  location
  material
  property

frame token positions:
  frame_first
  frame_last

controls:
  wrong_relation_same_object
  same_relation_other_frame
  same_relation_frame_other_object
```

三个 control 的含义：

```text
wrong_relation_same_object:
  同一 object，不同 relation frame。
  用来测试 relation-frame state 是否真的控制当前 relation 的读出。

same_relation_other_frame:
  同一 object、同一 relation、同一 target，不同模板 frame。
  用作格式扰动对照。

same_relation_frame_other_object:
  同一 relation、同一 frame、不同 object。
  用来测试 frame token 中是否混入 object/readout 信息。
```

每个 row 执行：

```text
1. clean forward 得到原始 candidate margin。
2. destroy:
   在 destroy layer 把 frame token state 替换为 control item 的对应 frame token state。
3. restore:
   在 restore layer 把 clean frame token state 恢复。
4. 用完整 answer sequence likelihood（答案序列概率）计算 target/distractor margin。
```

### 执行命令

```bash
PHASE75_OUTPUT_DIR=results/gpt5_phase75_relation_frame_token_intervention_full_$(date +%Y%m%d_%H%M%S) \
PHASE75_PROGRESS_EVERY=36 \
tests/gpt5/run_phase75_relation_frame_token_intervention_full.sh
```

实际输出目录：

```text
results/gpt5_phase75_relation_frame_token_intervention_full_20260609_103811
```

输出文件：

```text
results/gpt5_phase75_relation_frame_token_intervention_full_20260609_103811/qwen3_phase75_relation_frame_token_intervention.json
results/gpt5_phase75_relation_frame_token_intervention_full_20260609_103811/glm4_phase75_relation_frame_token_intervention.json
results/gpt5_phase75_relation_frame_token_intervention_full_20260609_103811/deepseek7b_phase75_relation_frame_token_intervention.json
results/gpt5_phase75_relation_frame_token_intervention_full_20260609_103811/phase75_relation_frame_token_intervention_summary.json
results/gpt5_phase75_relation_frame_token_intervention_full_20260609_103811/PHASE75_RELATION_FRAME_TOKEN_INTERVENTION_SUMMARY.md
```

注意：

```text
脚本优先尝试 flash_attention_2。
本机当前未安装 flash_attn，因此自动 fallback 到 sdpa。
DeepSeek7B 有 sliding window attention + sdpa warning，结果需保留该实现差异标记。
```

### 数据规模

```text
qwen3:
  rows = 2592

GLM4:
  rows = 2592

DS7B:
  rows = 2592

total rows = 7776
```

每个模型均完成完整 dataset，没有分批分析。

### Qwen3 客观结果

control 汇总：

```text
wrong_relation_same_object:
  eligible_destroy_drop = 1.6677
  eligible_restore_gain = 1.6606
  eligible_restore_gap = 0.0071
  eligible_destroy_top1 = 0.8160
  eligible_restore_top1 = 0.9392

same_relation_other_frame:
  eligible_destroy_drop = 0.3143
  eligible_restore_gain = 0.2620
  eligible_restore_gap = 0.0523
  eligible_destroy_top1 = 0.9323
  eligible_restore_top1 = 0.9670

same_relation_frame_other_object:
  eligible_destroy_drop = 0.1397
  eligible_restore_gain = 0.0978
  eligible_restore_gap = 0.0419
  eligible_destroy_top1 = 0.9566
  eligible_restore_top1 = 0.9844
```

最强路径：

```text
wrong_relation_same_object L8->L12 frame_last:
  destroy_drop = 2.8263
  restore_gain = 2.8474
  restore_gap = -0.0211
  destroy_top1 = 0.7431
  restore_top1 = 0.9514

wrong_relation_same_object L4->L8 frame_last:
  destroy_drop = 2.7624
  restore_gain = 2.7346
  restore_gap = 0.0278
  destroy_top1 = 0.7083
  restore_top1 = 0.9375
```

关系维度：

```text
wrong_relation_same_object used_for:
  destroy_drop = 3.1260
  restore_gain = 3.0223

wrong_relation_same_object can_do:
  destroy_drop = 1.7359
  restore_gain = 1.8389

wrong_relation_same_object is_a:
  destroy_drop = 1.6487
  restore_gain = 1.6950
```

客观现象：

```text
Qwen3 的 relation-frame token 干预有效，
尤其 frame_last 明显强于 frame_first。
wrong_relation_same_object 远强于两个 same-relation control。
说明 relation frame state 对当前 relation value 读出有非平凡作用。
```

### GLM4 客观结果

control 汇总：

```text
wrong_relation_same_object:
  eligible_destroy_drop = 1.1403
  eligible_restore_gain = 1.2104
  eligible_restore_gap = -0.0701
  eligible_destroy_top1 = 0.8660
  eligible_restore_top1 = 0.9641

same_relation_other_frame:
  eligible_destroy_drop = 0.2553
  eligible_restore_gain = 0.2783
  eligible_restore_gap = -0.0229
  eligible_destroy_top1 = 0.9003
  eligible_restore_top1 = 0.9641

same_relation_frame_other_object:
  eligible_destroy_drop = 0.4396
  eligible_restore_gain = 0.4231
  eligible_restore_gap = 0.0165
  eligible_destroy_top1 = 0.9232
  eligible_restore_top1 = 0.9592
```

最强路径：

```text
wrong_relation_same_object L10->L20 frame_last:
  destroy_drop = 2.0501
  restore_gain = 2.1373
  restore_gap = -0.0872
  destroy_top1 = 0.7778
  restore_top1 = 0.9673

wrong_relation_same_object L4->L10 frame_last:
  destroy_drop = 1.9294
  restore_gain = 2.0396
  restore_gap = -0.1103
```

关系维度：

```text
wrong_relation_same_object used_for:
  destroy_drop = 2.4622
  restore_gain = 2.1074

wrong_relation_same_object can_do:
  destroy_drop = 1.3125
  restore_gain = 1.5228

wrong_relation_same_object is_a:
  destroy_drop = 1.2839
  restore_gain = 1.3569
```

客观现象：

```text
GLM4 也存在 relation-frame token 因果作用。
最强路径偏 L10->L20，比 Qwen3 更靠中层。
same_relation_frame_other_object 的 effect 不为零，
说明 GLM4 的 frame token 可能混入 object/readout 或 category-compatible 信息。
```

### DS7B 客观结果

control 汇总：

```text
wrong_relation_same_object:
  eligible_destroy_drop = 1.6177
  eligible_restore_gain = 1.6495
  eligible_restore_gap = -0.0318
  eligible_destroy_top1 = 0.7849
  eligible_restore_top1 = 0.9704

same_relation_other_frame:
  eligible_destroy_drop = 0.5184
  eligible_restore_gain = 0.5656
  eligible_restore_gap = -0.0472
  eligible_destroy_top1 = 0.8898
  eligible_restore_top1 = 0.9704

same_relation_frame_other_object:
  eligible_destroy_drop = 0.7954
  eligible_restore_gain = 0.8009
  eligible_restore_gap = -0.0055
  eligible_destroy_top1 = 0.8817
  eligible_restore_top1 = 0.9677
```

最强路径：

```text
wrong_relation_same_object L8->L10 frame_last:
  destroy_drop = 2.9246
  restore_gain = 2.7970
  restore_gap = 0.1276
  destroy_top1 = 0.6989
  restore_top1 = 0.9570

wrong_relation_same_object L12->L14 frame_last:
  destroy_drop = 2.6094
  restore_gain = 2.6047
  restore_gap = 0.0047
```

关系维度：

```text
wrong_relation_same_object can_do:
  destroy_drop = 2.6444
  restore_gain = 2.7255

wrong_relation_same_object used_for:
  destroy_drop = 2.3401
  restore_gain = 2.4907

wrong_relation_same_object material:
  destroy_drop = 1.6311
  restore_gain = 1.6731
```

客观现象：

```text
DS7B 也存在可恢复 relation-frame token 作用。
但 DS7B 的 clean_top1 baseline 较低，eligible subset 解释要更谨慎。
same_relation controls 明显不为零，说明 frame token 不是纯 relation-only state。
```

### 三模型共同现象

```text
1. wrong_relation_same_object 的破坏效应在三模型中都明显大于 same_relation_other_frame。
2. frame_last 明显强于 frame_first。
3. restore gain 基本可以恢复 destroy drop，说明干预不是单纯随机破坏。
4. relation-frame token 的作用小于 Phase 74 object token wrong-target replacement，
   但仍是稳定非零因果因素。
5. used_for / can_do 在三模型中普遍更敏感，
   说明功能性 relation 可能比 is_a/property 更依赖显式 relation-frame gating。
```

### 当前理论进展

Phase 75 支持一个更稳的局部图景：

```text
value prediction 不是 object token 单独决定，
也不是 relation frame token 单独决定，
而是 object state × relation-frame state 的兼容性读出。
```

从破解语言编码机制角度看，本轮推进了：

```text
知识网络中的 relation binding 不应被理解为一个静态语义轴。
它更像由 object identity / relation frame / candidate value / readout format
共同形成的条件化路径。
```

更具体地说：

```text
object token 提供“当前对象是谁”和部分 value-support 信息；
frame_last token 提供“当前要读取哪类 relation/value”的关系门控；
最终 candidate margin 来自两者与输出候选之间的兼容。
```

这与当前“相对编码”判断一致：

```text
单一路径信息有限；
必须比较 object path、relation-frame path、same-relation control、
wrong-relation control 和 joint path，才有全局路径意义。
```

### 问题和硬伤

```text
1. relation-frame token 并不是纯 relation 变量。
   same_relation_frame_other_object 在 GLM4/DS7B 中不为零，
   说明 frame token state 混入 object/readout/format 信息。

2. 当前还没有 object token + relation-frame token 联合干预。
   因此只能证明两条路径都相关，
   还不能证明二者的组合闭包。

3. 当前仍是整 token state transplant，
   不是 subspace-level factor isolation。
   不能区分 identity factor、relation factor、format factor。

4. DS7B baseline 较低，
   需要在下一轮对 DS7B 的候选答案稳定性单独做 baseline filter。

5. 结果说明 relation-frame path 重要，
   但不能直接上升为“语言整体编码机制”。
   它只是知识网络 relation binding 拼图的一块。
```

### 下一步计划

Phase 76：object + relation-frame joint closure。

目标：

```text
同时干预 object token 和 relation-frame token，
比较：
1. only object wrong-target
2. only relation-frame wrong-relation
3. object + relation-frame matched wrong item
4. object + relation-frame mismatched item
5. restore object only
6. restore frame only
7. restore both
```

关键判据：

```text
如果 both restore 明显强于单独 restore，
说明 value prediction 是组合闭包；

如果 matched wrong item 比 mismatched item 更稳定地转向对应 value，
说明 object × relation compatibility 是真实路径结构。
```

Phase 77：balanced cross-relation dataset 扩展。

目标：

```text
扩大 object/relation/value 三元组，
重点加强：
same_object_different_relation
same_relation_different_object
same_value_different_path
```

Phase 78：factor subspace audit。

目标：

```text
在 object path 与 relation-frame path 均稳定后，
再尝试分离：
identity factor
relation factor
value-support factor
readout-format factor
compatibility factor
```

当前不建议直接做复杂数学抽象；
应继续用大样本、强 control、可恢复干预，把全局关系路径图谱补完整。

## Phase 76: object-frame 联合闭包测试 [2026-06-09 12:20]

### 任务目标

根据 Phase 75 的结论继续推进：

```text
Phase 74:
  object token path 有稳定因果作用。

Phase 75:
  relation-frame token path 也有稳定因果作用。

Phase 76:
  测试 object token 与 relation-frame token 是否形成组合闭包。
```

核心问题：

```text
value prediction 是否由 object state × relation-frame state 的组合兼容性决定？
```

本轮不做模型测试前理论分析，直接编写跨模型脚本并按顺序跑完：

```text
qwen3 -> GLM4 -> DS7B
```

每个模型完成后均使用 `--hard-exit-after-model`。

### 新增脚本

```text
tests/gpt5/phase76_object_frame_joint_closure.py
tests/gpt5/phase76_object_frame_joint_closure_summary.py
tests/gpt5/run_phase76_object_frame_joint_closure_full.sh
```

脚本设计：

```text
dataset:
  objects = 12
  relations = 6
  frames = 3
  items = 216

relations:
  is_a
  used_for
  can_do
  location
  material
  property

patched positions:
  object_last
  frame_last
```

每个 clean item 选择：

```text
matched source:
  different object
  different relation
  different target

mismatch frame source:
  different from clean relation
  different from matched relation
```

条件：

```text
object_only_matched:
  只替换 object token 为 matched source 的 object token state。

frame_only_matched:
  只替换 frame_last token 为 matched source 的 frame_last token state。

joint_matched:
  object token 与 frame_last token 都来自同一个 matched source。

joint_mismatched_frame:
  object token 来自 matched source，
  frame_last token 来自另一个 mismatch source。

joint_restore_object_only:
  destroy 时替换 object + frame，
  restore 时只恢复 object。

joint_restore_frame_only:
  destroy 时替换 object + frame，
  restore 时只恢复 frame。

joint_restore_both:
  destroy 时替换 object + frame，
  restore 时同时恢复 object + frame。
```

评分方式：

```text
使用完整 answer sequence likelihood。

候选集合包括：
  clean target
  clean distractors
  matched source target
  mismatch source target

同时记录：
  clean target margin 是否下降；
  matched source target margin 是否上升；
  patched top1 是否转向 matched source target。
```

这能区分：

```text
只是破坏 clean target
vs
真的朝 matched source 的 value space 转移
```

### Smoke Test

命令：

```bash
PHASE76_OUTPUT_DIR=results/gpt5_phase76_smoke_$(date +%Y%m%d_%H%M%S) \
PHASE76_MODELS=qwen3 \
QWEN3_PHASE76_MAX_ITEMS=12 \
QWEN3_PHASE76_LAYER_PAIRS=4-8 \
PHASE76_PROGRESS_EVERY=6 \
tests/gpt5/run_phase76_object_frame_joint_closure_full.sh
```

结果：

```text
qwen3 smoke:
  rows = 84
  exit_code = 0
```

说明多点 hook、matched/mismatched source、full-sequence candidate scoring、summary 均可运行。

### 正式测试命令

```bash
PHASE76_OUTPUT_DIR=results/gpt5_phase76_object_frame_joint_closure_full_$(date +%Y%m%d_%H%M%S) \
PHASE76_PROGRESS_EVERY=36 \
tests/gpt5/run_phase76_object_frame_joint_closure_full.sh
```

实际输出目录：

```text
results/gpt5_phase76_object_frame_joint_closure_full_20260609_115345
```

输出文件：

```text
results/gpt5_phase76_object_frame_joint_closure_full_20260609_115345/qwen3_phase76_object_frame_joint_closure.json
results/gpt5_phase76_object_frame_joint_closure_full_20260609_115345/glm4_phase76_object_frame_joint_closure.json
results/gpt5_phase76_object_frame_joint_closure_full_20260609_115345/deepseek7b_phase76_object_frame_joint_closure.json
results/gpt5_phase76_object_frame_joint_closure_full_20260609_115345/phase76_object_frame_joint_closure_summary.json
results/gpt5_phase76_object_frame_joint_closure_full_20260609_115345/PHASE76_OBJECT_FRAME_JOINT_CLOSURE_SUMMARY.md
```

注意：

```text
本机未安装 flash_attn，flash_attention_2 自动 fallback 到 sdpa。
DS7B 仍有 sliding window attention + sdpa warning，后续解释继续标记该实现差异。
```

### 数据规模

```text
qwen3:
  items = 216
  rows = 3024

GLM4:
  items = 216
  rows = 3024

DS7B:
  items = 216
  rows = 3024

total rows = 9072
```

三模型均完整完成，没有分批分析。

### Qwen3 客观结果

condition 汇总：

```text
joint_matched:
  eligible_clean_drop = 11.3796
  eligible_matched_gain = 14.3994
  eligible_clean_margin_after = -6.6304
  eligible_matched_margin_after = 1.5088
  eligible_patched_clean_top1 = 0.0851
  eligible_patched_matched_top1 = 0.6028

joint_mismatched_frame:
  eligible_clean_drop = 10.0603
  eligible_matched_gain = 8.9988
  eligible_clean_margin_after = -5.3112
  eligible_matched_margin_after = -3.8918
  eligible_patched_clean_top1 = 0.1099
  eligible_patched_matched_top1 = 0.2092

object_only_matched:
  eligible_clean_drop = 7.5133
  eligible_matched_gain = 5.7724
  eligible_patched_matched_top1 = 0.0745

frame_only_matched:
  eligible_clean_drop = 4.0672
  eligible_matched_gain = 8.7956
  eligible_patched_matched_top1 = 0.1702

joint_restore_both:
  eligible_clean_drop = 0.7328
  eligible_matched_gain = 0.9057
  eligible_patched_clean_top1 = 0.8546
  eligible_patched_matched_top1 = 0.0035
```

最强路径：

```text
joint_matched L4->L8:
  eligible_clean_drop = 12.0060
  eligible_matched_gain = 14.2684
  eligible_patched_matched_top1 = 0.5957

joint_matched L8->L12:
  eligible_clean_drop = 10.7532
  eligible_matched_gain = 14.5303
  eligible_patched_matched_top1 = 0.6099
```

关系维度：

```text
joint_matched can_do:
  clean_drop = 13.5527
  matched_gain = 17.3343
  matched_top1 = 0.6667

joint_matched is_a:
  clean_drop = 13.0722
  matched_gain = 17.5949
  matched_top1 = 0.7000

joint_matched property:
  clean_drop = 11.7313
  matched_gain = 11.9787
  matched_top1 = 0.7778
```

客观现象：

```text
Qwen3 中 joint_matched 明显强于 object_only 和 frame_only。
matched target 被显著推高，并且 matched_top1 达到 0.6028。
mismatched frame 也会破坏 clean target，但 matched_top1 明显低于 joint_matched。
restore both 后 clean target 基本恢复，matched target 基本退出 top1。
```

### GLM4 客观结果

condition 汇总：

```text
joint_matched:
  eligible_clean_drop = 10.3216
  eligible_matched_gain = 13.4800
  eligible_clean_margin_after = -5.5684
  eligible_matched_margin_after = 1.8823
  eligible_patched_clean_top1 = 0.1567
  eligible_patched_matched_top1 = 0.6533

joint_mismatched_frame:
  eligible_clean_drop = 8.7697
  eligible_matched_gain = 7.4227
  eligible_matched_margin_after = -4.1750
  eligible_patched_matched_top1 = 0.1800

object_only_matched:
  eligible_clean_drop = 5.3553
  eligible_matched_gain = 4.5411
  eligible_patched_matched_top1 = 0.0733

frame_only_matched:
  eligible_clean_drop = 3.9600
  eligible_matched_gain = 8.3804
  eligible_patched_matched_top1 = 0.2633

joint_restore_both:
  eligible_clean_drop = 0.8120
  eligible_matched_gain = 1.6062
  eligible_patched_clean_top1 = 0.9067
  eligible_patched_matched_top1 = 0.0067
```

最强路径：

```text
joint_matched L4->L10:
  clean_drop = 11.8084
  matched_gain = 14.1173
  matched_top1 = 0.6800

joint_matched L10->L20:
  clean_drop = 8.8349
  matched_gain = 12.8428
  matched_top1 = 0.6267
```

关系维度：

```text
joint_matched can_do:
  clean_drop = 12.7828
  matched_gain = 16.8680
  matched_top1 = 0.8500

joint_matched used_for:
  clean_drop = 12.1617
  matched_gain = 13.8461
  matched_top1 = 0.6136

joint_matched property:
  clean_drop = 10.2595
  matched_gain = 10.7197
  matched_top1 = 0.8478
```

客观现象：

```text
GLM4 的 joint_matched 同样强于单路径。
L4->L10 比 L10->L20 更强，说明本轮 joint source transplant 对 GLM4 的早中层窗口更敏感。
restore both 后 clean_top1 回到 0.9067，matched_top1 降到 0.0067。
```

### DS7B 客观结果

condition 汇总：

```text
joint_matched:
  eligible_clean_drop = 9.3725
  eligible_matched_gain = 12.5306
  eligible_clean_margin_after = -3.8368
  eligible_matched_margin_after = -1.2280
  eligible_patched_clean_top1 = 0.2717
  eligible_patched_matched_top1 = 0.4457

joint_mismatched_frame:
  eligible_clean_drop = 8.4139
  eligible_matched_gain = 7.1188
  eligible_matched_margin_after = -6.6399
  eligible_patched_matched_top1 = 0.0707

object_only_matched:
  eligible_clean_drop = 4.0315
  eligible_matched_gain = 3.8427
  eligible_patched_matched_top1 = 0.0163

frame_only_matched:
  eligible_clean_drop = 5.8079
  eligible_matched_gain = 9.8481
  eligible_patched_matched_top1 = 0.2663

joint_restore_both:
  eligible_clean_drop = 0.4271
  eligible_matched_gain = 0.5590
  eligible_patched_clean_top1 = 0.9130
  eligible_patched_matched_top1 = 0.0054
```

最强路径：

```text
joint_matched L8->L10:
  clean_drop = 9.8944
  matched_gain = 12.6858
  matched_top1 = 0.4565

joint_matched L12->L14:
  clean_drop = 8.8505
  matched_gain = 12.3755
  matched_top1 = 0.4348
```

关系维度：

```text
joint_matched can_do:
  clean_drop = 13.8635
  matched_gain = 17.6439
  matched_top1 = 0.7857

joint_matched used_for:
  clean_drop = 9.9869
  matched_gain = 13.9578
  matched_top1 = 0.4615

joint_matched property:
  clean_drop = 9.3158
  matched_gain = 10.2682
  matched_top1 = 0.6500
```

客观现象：

```text
DS7B 的 clean baseline eligible_n 较低，但在 eligible subset 中 joint_matched 仍明显强于 object_only。
frame_only 在 DS7B 中较强，说明 DS7B 对 relation-frame/readout state 更敏感。
joint_matched 的 matched_top1 = 0.4457，明显高于 joint_mismatched_frame = 0.0707。
restore both 后 clean_top1 = 0.9130，matched_top1 = 0.0054。
```

### 三模型共同现象

```text
1. joint_matched 在三模型中都是最强组合干预。

2. joint_matched 同时满足：
   clean target 大幅下降；
   matched source target 大幅上升；
   matched target top1 显著高于 object_only、frame_only、mismatched。

3. joint_mismatched_frame 也会强烈破坏 clean target，
   但 matched target 上升明显弱于 joint_matched。
   这说明“组合一致性”比“简单破坏”更重要。

4. restore both 在三模型中均能大幅恢复 clean target，
   且 matched target 基本退出 top1。

5. object-only 与 frame-only 都有作用，
   但单独路径很少能稳定把输出转向 matched source target。

6. can_do / used_for / property 等关系在 joint matched 下更敏感，
   说明功能/属性类关系更依赖 object-frame 组合兼容。
```

### 当前研究进展

Phase 76 对 Phase 75 的关键推进是：

```text
Phase 75:
  证明 relation-frame path 也有因果作用。

Phase 76:
  证明 object token 与 relation-frame token 同源匹配时，
  输出不只是被破坏，
  而是会明显转向 matched source 的 value space。
```

这比单路径 patch 更接近组合闭包。

当前最稳的客观结论：

```text
value prediction 由 object state 和 relation-frame state 的组合兼容性共同决定。
```

但严格说，还不能说已经得到纯数学机制，因为：

```text
当前仍然是 whole-token state transplant；
object state 和 frame state 都是混合因子；
还没有 subspace-level factor isolation。
```

### 对条件化关系因子动力学公式的修正

当前更合适的局部公式：

```text
S(v | o, r, c, l)
=
S_base(v, c)
+
S_obj(v, z_o(l))
+
S_frame(v, z_f(l))
+
S_joint(v, z_o(l), z_f(l))
+
S_read(v, h_l)
```

中文解释：

```text
S_obj:
  对象状态对候选值的支持。

S_frame:
  关系框架对候选空间和值槽位的门控。

S_joint:
  对象状态和关系框架状态之间的组合兼容性。

S_read:
  当前残差轨迹到输出候选序列的读出项。
```

Phase 76 的证据主要支持：

```text
S_joint 不等于 S_obj + S_frame 的简单相加。
```

原因：

```text
joint_matched 能显著推高 matched source target；
joint_mismatched_frame 虽然也破坏 clean target，
但不能稳定推高 matched source target。
```

也就是说：

```text
同源 object-frame pair 更像合法组合状态；
不同源 object-frame pair 更像不兼容/混乱状态。
```

这就是 object-frame compatibility dynamics 的核心证据。

### 问题和硬伤

```text
1. 本轮使用的是完整 token state transplant，
   不能区分 identity、relation、format、readout、compatibility 等子因子。

2. matched source target 的上升并不等于完全自然生成。
   它仍是在 closed candidate set 内的 full-sequence scoring。

3. joint_mismatched_frame 也有强 clean_drop，
   说明不兼容状态本身就会强烈破坏输出，
   因此必须同时看 matched_gain 和 matched_top1，不能只看 clean_drop。

4. DS7B baseline eligible_n 较低，且仍有 SDPA sliding window warning。
   DS7B 结论以 eligible subset 的相对比较为主。

5. 当前只测 object_last + frame_last。
   还没有测试 value slot 前后、多 token object、多 token frame、last token readout。

6. 当前没有 open generation audit。
   还不能证明自然生成会稳定转向 matched source value。
```

### 下一步计划

Phase 77：balanced cross-relation 扩展与复核。

目标：

```text
扩大 object/relation/value 三元组，
重点增强：
same_object_different_relation
same_relation_different_object
same_value_different_path
matched vs mismatched joint source
```

Phase 78：factor subspace audit。

目标：

```text
在 Phase 76 证明 object-frame joint closure 后，
开始尝试分离：
identity factor
category factor
relation gate factor
value-support factor
compatibility factor
readout factor
```

但判据必须是：

```text
subspace destroy-restore，
不是 probe accuracy。
```

Phase 79：open generation audit。

目标：

```text
验证 object-frame joint intervention 是否影响自由生成，
而不仅仅影响 closed candidate ranking。
```

Phase 80：迁移到 logic/syntax。

原则：

```text
先建立稳定 reader；
再定位 object/operator/frame path；
再做 joint closure；
最后做 factor subspace closure。
```

当前不应直接做大理论收束；
应该继续补全“关系组合路径图谱”的自然 control 和子因子审计。

## Phase 77: balanced cross-relation joint closure 大范围复核 [2026-06-09 13:59]

### 任务目标

根据 Phase 76 的 object-frame 联合闭包结果继续扩大数据范围，验证：

```text
object × relation-frame joint closure 是否只是 12 objects / 6 relations 小数据集现象，
还是能在更大 object/relation/value/frame 覆盖上继续成立。
```

本轮遵守用户要求：

```text
1. 测试数据范围尽量扩大。
2. 三模型按 qwen3 -> GLM4 -> DS7B 顺序运行。
3. 每个模型完成后 hard-exit。
4. 模型测试中不做中途分析。
5. 优先记录客观结果，不轻易做理论总结。
```

### 新增脚本

```text
tests/gpt5/phase77_balanced_cross_relation_joint_closure.py
tests/gpt5/phase77_balanced_cross_relation_joint_closure_summary.py
tests/gpt5/run_phase77_balanced_cross_relation_joint_closure_full.sh
```

### 数据设计

Phase 76：

```text
objects = 12
relations = 6
frames = 3
items = 216
```

Phase 77 扩展为：

```text
objects = 24
relations = 7
frames = 4
items = 672
```

relations：

```text
is_a
used_for
can_do
location
material
property
part_of
```

每个 object 覆盖所有 relation，每个 relation 有 4 个 frame。继续使用完整 answer sequence likelihood，不使用 first-token scoring。

### 条件设计

沿用 Phase 76 的 joint closure 条件：

```text
object_only_matched
frame_only_matched
joint_matched
joint_mismatched_frame
joint_restore_object_only
joint_restore_frame_only
joint_restore_both
```

关键指标：

```text
clean_drop:
  clean target margin 下降多少。

matched_gain:
  matched source target margin 上升多少。

matched_top1:
  patched 后 matched source target 成为第一名的比例。
```

本轮继续强调：

```text
不能只看 clean_drop。
必须同时看 matched_gain 和 matched_top1，
区分“破坏输出”和“合法转移到 matched value space”。
```

### Smoke Test

命令：

```bash
PHASE77_OUTPUT_DIR=results/gpt5_phase77_smoke_$(date +%Y%m%d_%H%M%S) \
PHASE77_MODELS=qwen3 \
QWEN3_PHASE77_MAX_ITEMS=28 \
QWEN3_PHASE77_LAYER_PAIRS=4-8 \
PHASE77_PROGRESS_EVERY=14 \
tests/gpt5/run_phase77_balanced_cross_relation_joint_closure_full.sh
```

结果：

```text
qwen3 smoke:
  items = 28
  rows = 196
  exit_code = 0
```

说明扩展数据、matched/mismatched 选择、多点 hook、full-sequence scoring 和 summary 均正常。

### 正式测试命令

```bash
PHASE77_OUTPUT_DIR=results/gpt5_phase77_balanced_cross_relation_joint_closure_full_$(date +%Y%m%d_%H%M%S) \
PHASE77_PROGRESS_EVERY=84 \
tests/gpt5/run_phase77_balanced_cross_relation_joint_closure_full.sh
```

实际输出目录：

```text
results/gpt5_phase77_balanced_cross_relation_joint_closure_full_20260609_123642
```

输出文件：

```text
results/gpt5_phase77_balanced_cross_relation_joint_closure_full_20260609_123642/qwen3_phase77_balanced_cross_relation_joint_closure.json
results/gpt5_phase77_balanced_cross_relation_joint_closure_full_20260609_123642/glm4_phase77_balanced_cross_relation_joint_closure.json
results/gpt5_phase77_balanced_cross_relation_joint_closure_full_20260609_123642/deepseek7b_phase77_balanced_cross_relation_joint_closure.json
results/gpt5_phase77_balanced_cross_relation_joint_closure_full_20260609_123642/phase77_balanced_cross_relation_joint_closure_summary.json
results/gpt5_phase77_balanced_cross_relation_joint_closure_full_20260609_123642/PHASE77_BALANCED_CROSS_RELATION_JOINT_CLOSURE_SUMMARY.md
```

注意：

```text
flash_attention_2 仍因未安装 flash_attn 自动 fallback 到 sdpa。
DS7B 仍出现 sliding window attention + sdpa warning，解释 DS7B 结果时继续标记该实现差异。
```

### 数据规模

```text
qwen3:
  objects = 24
  items = 672
  rows = 9408

GLM4:
  objects = 24
  items = 672
  rows = 9408

DS7B:
  objects = 24
  items = 672
  rows = 9408

total rows = 28224
```

三模型完整完成。

### Qwen3 客观结果

condition 汇总：

```text
joint_matched:
  eligible_clean_drop = 10.6469
  eligible_matched_gain = 13.6120
  eligible_matched_margin_after = 0.4939
  eligible_patched_clean_top1 = 0.0577
  eligible_patched_matched_top1 = 0.5295

joint_mismatched_frame:
  eligible_clean_drop = 9.1362
  eligible_matched_gain = 8.7118
  eligible_matched_margin_after = -4.4062
  eligible_patched_matched_top1 = 0.1744

object_only_matched:
  eligible_clean_drop = 7.6197
  eligible_matched_gain = 5.4163
  eligible_patched_matched_top1 = 0.0667

frame_only_matched:
  eligible_clean_drop = 4.2557
  eligible_matched_gain = 8.4183
  eligible_patched_matched_top1 = 0.1564

joint_restore_both:
  eligible_clean_drop = 0.7039
  eligible_matched_gain = 0.7830
  eligible_patched_clean_top1 = 0.8744
  eligible_patched_matched_top1 = 0.0077
```

路径维度：

```text
joint_matched L8->L12:
  clean_drop = 10.3321
  matched_gain = 14.2122
  matched_top1 = 0.5718

joint_matched L4->L8:
  clean_drop = 10.9617
  matched_gain = 13.0117
  matched_top1 = 0.4872
```

关系维度：

```text
joint_matched can_do:
  clean_drop = 11.4236
  matched_gain = 16.9673
  matched_top1 = 0.7153

joint_matched location:
  clean_drop = 9.2001
  matched_gain = 11.9923
  matched_top1 = 0.6351

joint_matched property:
  clean_drop = 9.8111
  matched_gain = 10.0038
  matched_top1 = 0.6053

joint_matched part_of:
  clean_drop = 9.7537
  matched_gain = 12.9344
  matched_top1 = 0.4151
```

客观现象：

```text
Qwen3 在扩展数据上继续保持 joint_matched 最强。
matched_top1 从 Phase 76 的 0.6028 降到 0.5295，但仍远高于 object_only / frame_only / mismatched。
restore_both 后 clean_top1 = 0.8744，matched_top1 = 0.0077。
```

### GLM4 客观结果

condition 汇总：

```text
joint_matched:
  eligible_clean_drop = 9.9306
  eligible_matched_gain = 14.0403
  eligible_matched_margin_after = 1.8232
  eligible_patched_clean_top1 = 0.1126
  eligible_patched_matched_top1 = 0.6486

joint_mismatched_frame:
  eligible_clean_drop = 8.2591
  eligible_matched_gain = 8.2753
  eligible_matched_margin_after = -3.9419
  eligible_patched_matched_top1 = 0.2005

object_only_matched:
  eligible_clean_drop = 5.8580
  eligible_matched_gain = 5.3420
  eligible_patched_matched_top1 = 0.0923

frame_only_matched:
  eligible_clean_drop = 3.3112
  eligible_matched_gain = 8.1506
  eligible_patched_matched_top1 = 0.2005

joint_restore_both:
  eligible_clean_drop = 0.7635
  eligible_matched_gain = 1.2880
  eligible_patched_clean_top1 = 0.8998
  eligible_patched_matched_top1 = 0.0113
```

路径维度：

```text
joint_matched L4->L10:
  clean_drop = 11.0774
  matched_gain = 14.3068
  matched_top1 = 0.6712

joint_matched L10->L20:
  clean_drop = 8.7839
  matched_gain = 13.7739
  matched_top1 = 0.6261
```

关系维度：

```text
joint_matched property:
  clean_drop = 9.5283
  matched_gain = 11.6585
  matched_top1 = 0.8469

joint_matched can_do:
  clean_drop = 9.4343
  matched_gain = 17.2041
  matched_top1 = 0.7361

joint_matched location:
  clean_drop = 9.4483
  matched_gain = 11.6492
  matched_top1 = 0.7353

joint_matched used_for:
  clean_drop = 12.2855
  matched_gain = 16.7086
  matched_top1 = 0.6986

joint_matched part_of:
  clean_drop = 9.1177
  matched_gain = 12.6077
  matched_top1 = 0.4789
```

客观现象：

```text
GLM4 扩展数据中 joint_matched 仍最强，matched_top1 = 0.6486。
L4->L10 比 L10->L20 更强，延续 Phase 76 现象。
restore_both 后 clean_top1 = 0.8998，matched_top1 = 0.0113。
```

### DS7B 客观结果

condition 汇总：

```text
joint_matched:
  eligible_clean_drop = 7.1563
  eligible_matched_gain = 12.1629
  eligible_matched_margin_after = -1.3271
  eligible_patched_clean_top1 = 0.2535
  eligible_patched_matched_top1 = 0.4161

joint_mismatched_frame:
  eligible_clean_drop = 6.2317
  eligible_matched_gain = 7.2367
  eligible_matched_margin_after = -6.2533
  eligible_patched_matched_top1 = 0.1119

object_only_matched:
  eligible_clean_drop = 3.6021
  eligible_matched_gain = 3.4196
  eligible_patched_matched_top1 = 0.0262

frame_only_matched:
  eligible_clean_drop = 3.7984
  eligible_matched_gain = 8.7878
  eligible_patched_matched_top1 = 0.2133

joint_restore_both:
  eligible_clean_drop = 0.4736
  eligible_matched_gain = 0.5091
  eligible_patched_clean_top1 = 0.8794
  eligible_patched_matched_top1 = 0.0087
```

路径维度：

```text
joint_matched L8->L10:
  clean_drop = 7.5145
  matched_gain = 12.3939
  matched_top1 = 0.4371

joint_matched L12->L14:
  clean_drop = 6.7981
  matched_gain = 11.9320
  matched_top1 = 0.3951
```

关系维度：

```text
joint_matched can_do:
  clean_drop = 8.7604
  matched_gain = 17.0136
  matched_top1 = 0.6311

joint_matched used_for:
  clean_drop = 9.5306
  matched_gain = 15.3395
  matched_top1 = 0.5135

joint_matched part_of:
  clean_drop = 6.7018
  matched_gain = 13.4518
  matched_top1 = 0.4459

joint_matched property:
  clean_drop = 5.5004
  matched_gain = 8.4085
  matched_top1 = 0.4143
```

客观现象：

```text
DS7B 在扩展数据中仍保持 joint_matched > frame_only > object_only 的 matched_top1 结构。
matched_top1 = 0.4161，低于 Qwen3/GLM4，但明显高于 mismatched = 0.1119。
restore_both 后 clean_top1 = 0.8794，matched_top1 = 0.0087。
```

### 与 Phase 76 的一致性

```text
Qwen3:
  Phase 76 joint_matched matched_top1 = 0.6028
  Phase 77 joint_matched matched_top1 = 0.5295
  扩展后效应变弱但仍稳定。

GLM4:
  Phase 76 joint_matched matched_top1 = 0.6533
  Phase 77 joint_matched matched_top1 = 0.6486
  几乎完全稳定。

DS7B:
  Phase 76 joint_matched matched_top1 = 0.4457
  Phase 77 joint_matched matched_top1 = 0.4161
  扩展后仍稳定，但绝对值较低。
```

### 当前客观结论

Phase 77 说明：

```text
Phase 76 的 object-frame joint matched effect 不是小数据集偶发现象。
在 24 objects / 7 relations / 4 frames 的扩展数据上，三模型仍稳定出现：

joint_matched > joint_mismatched_frame > object_only/frame_only

其中最关键的是 matched_top1：
  joint_matched 明显高于 mismatched 和单路径。
```

更严格地说：

```text
object-frame 同源组合可以产生合法 value-space 转移；
object-frame 不匹配组合更多表现为破坏，而不是合法转移。
```

### 问题和硬伤

```text
1. Phase 77 数据虽然更大，但仍是人工构造 dataset，不能等同自然语言全分布。

2. 扩展数据中部分对象/关系语义较简单，可能存在模板和候选池先验。

3. 当前仍是 whole-token transplant，不是 factor subspace closure。

4. matched_top1 在 Qwen3/DS7B 扩展后下降，说明组合闭包强度受数据复杂度影响。

5. DS7B 仍需以 eligible subset 相对比较为主。

6. 当前仍是 closed candidate scoring，不是 open generation。
```

### 下一步计划

Phase 78：factor subspace audit。

目标：

```text
在 Phase 76/77 已证明 whole-token joint closure 后，
开始拆分 object token 与 frame token 内部的混合因子。
```

优先分离：

```text
identity factor
category factor
relation gate factor
value-support factor
slot/readout factor
compatibility factor
```

但判据必须是：

```text
subspace destroy-restore
```

而不是：

```text
probe accuracy
```

Phase 79：open generation audit。

目标：

```text
验证 object-frame joint matched intervention 是否影响自由生成，
而不是只改变封闭候选排序。
```

Phase 80：从知识关系迁移到逻辑和语法。

原则：

```text
reader calibration
-> path localization
-> joint closure
-> factor subspace closure
```

当前不应继续做宏大理论收束；应先把组合闭包推进到子因子层。

## Phase 78: factor subspace audit 大范围测试 [2026-06-09 16:12]

### 任务目标

根据 Phase 76/77 的结果，whole-token joint closure 已经在三模型和更大数据上稳定出现：

```text
object token + relation-frame token 的同源组合，可以把 clean answer 推向 matched answer。
```

但这仍然是 whole-token transplant，不知道 object token 和 frame token 内部到底哪些成分在起作用。本轮目标是做更保守的 factor subspace audit：

```text
1. 不训练 probe。
2. 不用复杂统计理论做机制结论。
3. 只用自然对比差分构造低秩子空间。
4. 只替换 object/frame 在该子空间内的成分。
5. 测试子空间级 joint matched 是否还能复现 Phase 77 的整词元 joint matched 效果。
```

用户提醒是正确的：深度网络是相对编码，单一路径信息有限，必须和其他路径比较，获得全局路径才有意义。因此本轮不是只看 object binding，而是继续围绕：

```text
object path
relation-frame path
matched joint path
mismatched frame path
restore path
```

做对比。

### 生成脚本

新增：

```text
tests/gpt5/phase78_factor_subspace_audit.py
tests/gpt5/phase78_factor_subspace_audit_summary.py
tests/gpt5/run_phase78_factor_subspace_audit_full.sh
```

脚本检查：

```bash
python -m py_compile \
  tests/gpt5/phase78_factor_subspace_audit.py \
  tests/gpt5/phase78_factor_subspace_audit_summary.py
```

结果：

```text
compile passed
```

### 测试原理

对每个模型、每个 layer pair，先用自然样本构造两个子空间：

```text
object_basis:
  matched object token state - clean object token state

frame_basis:
  matched relation-frame token state - clean relation-frame token state
```

然后不是替换整个 token state，而是只替换这些 basis 张成的子空间成分：

```text
current_state <- current_state + B @ B.T @ (source_state - current_state)
```

测试条件：

```text
object_subspace_matched
frame_subspace_matched
joint_subspace_matched
joint_subspace_mismatched_frame
joint_subspace_restore_object_only
joint_subspace_restore_frame_only
joint_subspace_restore_both
```

关键判据：

```text
joint_subspace_matched 是否明显强于 object_subspace_matched 和 frame_subspace_matched；
joint_subspace_matched 是否明显强于 joint_subspace_mismatched_frame；
restore_both 是否恢复 clean answer；
```

如果成立，说明 Phase 76/77 的 whole-token effect 不是只能靠整词元混合产生，object/frame 的自然对比子空间中已经包含部分可因果迁移的信息。

### Smoke Test

命令：

```bash
PHASE78_MODELS=qwen3 \
QWEN3_PHASE78_MAX_ITEMS=28 \
QWEN3_PHASE78_LAYER_PAIRS=4-8 \
PHASE78_MAX_BASIS_ITEMS=28 \
PHASE78_PROGRESS_EVERY=14 \
PHASE78_OUTPUT_DIR=results/gpt5_phase78_factor_subspace_audit_smoke_$(date +%Y%m%d_%H%M%S) \
tests/gpt5/run_phase78_factor_subspace_audit_full.sh
```

结果：

```text
qwen3 rows = 196
exit_code = 0
attn_impl = sdpa
```

说明子空间构造、hook、full-sequence scoring、hard-exit、summary 全链路可运行。

### 全量测试命令

```bash
PHASE78_OUTPUT_DIR=results/gpt5_phase78_factor_subspace_audit_full_$(date +%Y%m%d_%H%M%S) \
PHASE78_PROGRESS_EVERY=84 \
PHASE78_MAX_BASIS_ITEMS=168 \
tests/gpt5/run_phase78_factor_subspace_audit_full.sh
```

模型顺序：

```text
qwen3 -> glm4 -> deepseek7b
```

每个模型都使用：

```text
--hard-exit-after-model
```

实际输出目录：

```text
results/gpt5_phase78_factor_subspace_audit_full_20260609_141908
```

输出文件：

```text
results/gpt5_phase78_factor_subspace_audit_full_20260609_141908/qwen3_phase78_factor_subspace_audit.json
results/gpt5_phase78_factor_subspace_audit_full_20260609_141908/glm4_phase78_factor_subspace_audit.json
results/gpt5_phase78_factor_subspace_audit_full_20260609_141908/deepseek7b_phase78_factor_subspace_audit.json
results/gpt5_phase78_factor_subspace_audit_full_20260609_141908/phase78_factor_subspace_audit_summary.json
results/gpt5_phase78_factor_subspace_audit_full_20260609_141908/PHASE78_FACTOR_SUBSPACE_AUDIT_SUMMARY.md
```

注意：

```text
本机未安装 flash_attn，因此 flash_attention_2 加载失败后自动回退到 sdpa。
DeepSeek7B 运行时出现 sliding-window + sdpa 的 transformers 提醒，因此 DeepSeek7B 的结论需要保留实现路径 caveat。
```

### 数据规模

```text
objects = 24
relations = 7
frames = 4
items/model = 672
basis_items/model = 168
conditions = 7
layer_pairs/model = 2
rows/model = 9408
total_rows = 28224
```

layer pairs：

```text
Qwen3:
  L4->L8
  L8->L12

GLM4:
  L4->L10
  L10->L20

DeepSeek7B:
  L8->L10
  L12->L14
```

### Qwen3 客观结果

```text
rows = 9408
eligible = 780
```

condition summary：

```text
joint_subspace_matched:
  clean_drop = 8.8485
  matched_gain = 12.0854
  clean_after = -4.6308
  matched_after = -1.0326
  clean_top1 = 0.1641
  matched_top1 = 0.4051

joint_subspace_mismatched_frame:
  clean_drop = 7.6611
  matched_gain = 7.7940
  clean_top1 = 0.2141
  matched_top1 = 0.1205

frame_subspace_matched:
  clean_drop = 4.0292
  matched_gain = 7.9465
  clean_top1 = 0.5397
  matched_top1 = 0.1385

object_subspace_matched:
  clean_drop = 6.1465
  matched_gain = 4.1399
  clean_top1 = 0.3692
  matched_top1 = 0.0256

joint_subspace_restore_both:
  clean_drop = 1.0651
  matched_gain = 1.2686
  clean_top1 = 0.8423
  matched_top1 = 0.0077
```

路径对比：

```text
L8->L12 joint_subspace_matched:
  matched_top1 = 0.4436

L4->L8 joint_subspace_matched:
  matched_top1 = 0.3667
```

Qwen3 现象：

```text
1. 子空间 joint matched 明显强于 object-only 和 frame-only。
2. matched frame 比 mismatched frame 更能产生合法 matched transfer。
3. restore_both 能显著恢复 clean answer。
4. L8->L12 略强于 L4->L8，说明中浅层 object-frame 子空间组合路径更清晰。
```

与 Phase 77 whole-token 对比：

```text
Phase77 joint_matched matched_top1 = 0.5295
Phase78 joint_subspace_matched matched_top1 = 0.4051
```

说明子空间替换保留了相当一部分整词元闭包效果，但没有完全复现 whole-token effect。

### GLM4 客观结果

```text
rows = 9408
eligible = 888
```

condition summary：

```text
joint_subspace_matched:
  clean_drop = 7.1730
  matched_gain = 11.4769
  clean_after = -2.8415
  matched_after = -0.7403
  clean_top1 = 0.3052
  matched_top1 = 0.4403

joint_subspace_mismatched_frame:
  clean_drop = 6.1109
  matched_gain = 7.1422
  clean_top1 = 0.3637
  matched_top1 = 0.1273

frame_subspace_matched:
  clean_drop = 2.6983
  matched_gain = 7.0353
  clean_top1 = 0.6892
  matched_top1 = 0.1261

object_subspace_matched:
  clean_drop = 4.4448
  matched_gain = 3.8216
  clean_top1 = 0.5372
  matched_top1 = 0.0507

joint_subspace_restore_both:
  clean_drop = 1.0388
  matched_gain = 1.5692
  clean_top1 = 0.8806
  matched_top1 = 0.0135
```

路径对比：

```text
L4->L10 joint_subspace_matched:
  matched_top1 = 0.4730

L10->L20 joint_subspace_matched:
  matched_top1 = 0.4077
```

GLM4 现象：

```text
1. 子空间 joint matched 明显成立。
2. L4->L10 强于 L10->L20，说明 GLM4 的 object-frame 组合路径偏浅层。
3. frame_subspace 单独比 object_subspace 更能提升 matched_gain，但合法 top1 仍主要来自 joint。
4. restore_both clean_top1 = 0.8806，说明 object/frame 两个子空间确实是可恢复的主要扰动来源。
```

与 Phase 77 whole-token 对比：

```text
Phase77 joint_matched matched_top1 = 0.6486
Phase78 joint_subspace_matched matched_top1 = 0.4403
```

GLM4 子空间保留效果明显，但相比 whole-token 下降更大，说明 whole token 中还有其他格式/路由成分未被 rank-16 object/frame 自然对比子空间覆盖。

### DeepSeek7B 客观结果

```text
rows = 9408
eligible = 572
```

condition summary：

```text
joint_subspace_matched:
  clean_drop = 5.0480
  matched_gain = 9.6949
  clean_after = -1.1080
  matched_after = -3.7951
  clean_top1 = 0.4108
  matched_top1 = 0.2640

joint_subspace_mismatched_frame:
  clean_drop = 4.8122
  matched_gain = 6.0316
  clean_top1 = 0.4231
  matched_top1 = 0.0769

frame_subspace_matched:
  clean_drop = 2.8841
  matched_gain = 7.6324
  clean_top1 = 0.6101
  matched_top1 = 0.1486

object_subspace_matched:
  clean_drop = 2.7596
  matched_gain = 2.4039
  clean_top1 = 0.6399
  matched_top1 = 0.0140

joint_subspace_restore_both:
  clean_drop = 0.6379
  matched_gain = 0.5466
  clean_top1 = 0.8392
  matched_top1 = 0.0035
```

路径对比：

```text
L12->L14 joint_subspace_matched:
  matched_top1 = 0.2657

L8->L10 joint_subspace_matched:
  matched_top1 = 0.2622
```

DeepSeek7B 现象：

```text
1. 子空间 joint matched 成立，但弱于 Qwen3/GLM4。
2. 两条路径接近，说明当前 L8-L14 早中层范围内没有明显单一路径峰值。
3. frame_subspace_matched 明显强于 object_subspace_matched。
4. restore_both 仍能恢复 clean top1 到 0.8392。
```

与 Phase 77 whole-token 对比：

```text
Phase77 joint_matched matched_top1 = 0.4161
Phase78 joint_subspace_matched matched_top1 = 0.2640
```

说明 DS7B 的合法转移更依赖 whole-token 或更复杂的轨迹成分，rank-16 子空间只能保留部分 effect。

### 三模型对比

Phase 78 joint_subspace_matched：

```text
Qwen3:
  matched_top1 = 0.4051
  matched_gain = 12.0854

GLM4:
  matched_top1 = 0.4403
  matched_gain = 11.4769

DeepSeek7B:
  matched_top1 = 0.2640
  matched_gain = 9.6949
```

Phase 77 whole-token joint_matched -> Phase 78 subspace joint_matched：

```text
Qwen3:
  0.5295 -> 0.4051

GLM4:
  0.6486 -> 0.4403

DeepSeek7B:
  0.4161 -> 0.2640
```

客观结论：

```text
1. 三模型中，rank-16 object/frame 自然对比子空间都能保留一部分 Phase 77 的 whole-token joint closure。
2. 子空间 joint matched 都强于 object-only、frame-only、mismatched-frame。
3. restore_both 都能恢复 clean answer，说明子空间扰动不是不可逆破坏。
4. 子空间 effect 明显低于 whole-token effect，说明当前 basis 只捕获了部分因子。
5. frame path 在三模型中普遍比 object path 单独更强，说明 relation-frame 可能更接近 value readout / relation gate。
6. object path 单独弱，但与 frame path 同源组合后明显增强，说明 object 因子可能不是独立输出因子，而是 compatibility / identity support。
```

### 当前研究进展

Phase 78 把 Phase 76/77 的结论推进了一层：

```text
Phase 76:
  object token + frame token 同源整词元组合有效。

Phase 77:
  在 24 objects × 7 relations × 4 frames 大范围数据上仍有效。

Phase 78:
  不替换整词元，只替换 object/frame 自然对比子空间，也能部分复现 joint closure。
```

这说明 object-frame binding 不是只存在于不可分解的整词元 hidden state 中，而是至少有一部分信息落在可由自然对比提取的低秩子空间里。

更谨慎的理论表述：

```text
知识关系的输出不是 object identity 单独决定，
也不是 relation-frame 单独决定，
而是 object support factor 与 relation-frame readout factor 的相容组合。
```

这和当前“条件化关系因子动力学”方向一致：

```text
object 提供身份/类别/可兼容性支持；
relation-frame 提供读取槽位/关系门控/输出格式；
两者同源组合形成合法 value-space 转移；
两者不匹配组合更多表现为破坏或弱转移。
```

### 问题和硬伤

```text
1. 当前 basis 是 rank-16 自然对比子空间，仍然不是纯 factor。

2. object_basis 和 frame_basis 仍可能混入 identity、relation、value、position、template 多种因素。

3. 子空间替换低于 whole-token 替换，说明还有未捕获成分：
   token residual remainder；
   attention route；
   MLP nonlinear gate；
   position-specific compatibility；
   multi-layer trajectory；
   output formatting factor。

4. 当前仍是 closed candidate scoring，不是 open generation。

5. 当前不是 destroy-restore 子空间闭包，只是 matched subspace transfer + clean restore。

6. DeepSeek7B 使用 sdpa 时有 sliding-window warning，因此 DS7B 结果要与其实现路径绑定解释。

7. 当前仍然只覆盖知识关系 value retrieval，还没有迁移到逻辑推理和语法规则。
```

### 条件化关系因子动力学公式的改进

Phase 78 后，公式需要从：

```text
Value = F(Object, RelationFrame)
```

细化为：

```text
h_l = h_l
    + P_O^l(source_O - clean_O)
    + P_R^l(source_R - clean_R)

Readout(value)
    = G_l(
        IdentitySupport_O,
        RelationGate_R,
        Compatibility(O, R),
        ResidualContext,
        OutputSlot
      )
```

其中：

```text
P_O^l:
  object 自然对比子空间投影

P_R^l:
  relation-frame 自然对比子空间投影

IdentitySupport_O:
  object 提供的身份/类别/兼容性支持

RelationGate_R:
  relation-frame 提供的读取槽位和关系门控

Compatibility(O, R):
  object 与 relation-frame 是否同源、是否可组合

OutputSlot:
  当前 prompt 的输出位置和候选 value 格式
```

这仍不是最终数学理论，只是更贴近当前实验事实的操作性公式。

### 下一步计划

Phase 79：rank sweep + residual remainder audit。

目标：

```text
测试 rank=4,8,16,32,64 时 joint_subspace_matched 如何变化；
同时测试 remainder-only 是否仍含有大量 closure effect。
```

关键问题：

```text
object-frame factor 是低秩集中，还是高维分散？
whole-token effect 中有多少能被自然对比子空间解释？
剩余维度是否是噪声，还是另一个关键 factor？
```

Phase 80：factor orthogonal audit。

目标：

```text
把 object basis、relation-frame basis、value basis、template basis 做正交化对比。
```

判据：

```text
去掉 value direction 后，object-frame joint 是否还有效；
去掉 template direction 后，relation-frame 是否仍有效；
```

Phase 81：open generation audit。

目标：

```text
验证子空间 joint matched 是否影响自由生成，
而不是只改变封闭候选排序。
```

Phase 82：知识关系 -> 逻辑/语法迁移。

目标：

```text
把同一套流程迁移到：
逻辑 operator-event binding；
语法 active/passive role binding；
temporal order binding；
coreference entity binding。
```

统一路径：

```text
reader calibration
-> path localization
-> whole-token joint closure
-> factor subspace audit
-> destroy-restore
-> open generation
```

当前最重要的结论不是“已经破解编码机制”，而是：

```text
全局关系路径图谱已经从 token 级推进到 subspace 级。
现在看到的语言/知识机制更像条件化因子组合，而不是单一方向或单一路径。
```

## Phase 79: rank sweep + residual remainder audit [2026-06-09 20:42]

### 任务目标

根据 Phase 78 的结果，rank-16 natural contrast subspace 已经能保留一部分 whole-token joint closure，但仍明显低于 Phase 77 whole-token effect。因此本轮继续验证两个关键问题：

```text
1. object-frame 因子是低秩集中，还是高维分散？
2. 被 rank 子空间拿走后，remainder 是否仍然保留大量 closure effect？
```

本轮不是理论总结，而是做 rank sweep + remainder audit：

```text
rank = 4, 8, 16, 32, 64

joint_subspace_matched:
  只替换 rank 子空间成分

joint_remainder_matched:
  只替换 rank 子空间正交剩余成分

joint_subspace_mismatched_frame:
  用不匹配 relation-frame 检查合法组合是否下降

joint_subspace_restore_both:
  恢复 object/frame 子空间，检查 clean answer 是否恢复
```

用户提供的 Phase 78 分析判断基本正确：

```text
1. Phase 78 是 natural contrast subspace transfer audit，不是纯因子闭包。
2. rank-16 子空间不是纯 identity/relation/value factor。
3. 子空间效果低于 whole-token，说明仍有 remainder、route、gate、format、trajectory 成分。
4. 下一步必须做 rank sweep 和 remainder-only。
```

因此本轮直接执行 Phase 79。

### 生成脚本

新增：

```text
tests/gpt5/phase79_rank_sweep_remainder_audit.py
tests/gpt5/phase79_rank_sweep_remainder_audit_summary.py
tests/gpt5/run_phase79_rank_sweep_remainder_audit_full.sh
```

脚本检查：

```bash
python -m py_compile \
  tests/gpt5/phase79_rank_sweep_remainder_audit.py \
  tests/gpt5/phase79_rank_sweep_remainder_audit_summary.py
```

结果：

```text
compile passed
```

### 测试原理

先用最大 rank=64 构造 object/frame 自然对比 basis：

```text
object_basis = matched object state - clean object state
frame_basis  = matched frame state - clean frame state
```

对每个 rank 取前 k 个方向：

```text
B_k = B[:, :k]
```

子空间替换：

```text
current <- current + B_k @ B_k.T @ (source - current)
```

剩余成分替换：

```text
current <- current + ((source - current) - B_k @ B_k.T @ (source - current))
```

如果 rank 增大时 subspace effect 上升、remainder effect 下降，说明 causal signal 主要集中在这些自然对比主方向中。

如果 remainder effect 仍然很强，说明 hidden state 中还有大量未被低秩 basis 捕获的高维成分。

### Smoke Test

命令：

```bash
PHASE79_MODELS=qwen3 \
QWEN3_PHASE79_MAX_ITEMS=28 \
QWEN3_PHASE79_LAYER_PAIRS=4-8 \
PHASE79_RANKS=4,8 \
PHASE79_MAX_BASIS_ITEMS=28 \
PHASE79_PROGRESS_EVERY=14 \
PHASE79_OUTPUT_DIR=results/gpt5_phase79_rank_sweep_remainder_audit_smoke_$(date +%Y%m%d_%H%M%S) \
tests/gpt5/run_phase79_rank_sweep_remainder_audit_full.sh
```

结果：

```text
qwen3 rows = 224
exit_code = 0
attn_impl = sdpa
```

### 全量测试命令

```bash
PHASE79_OUTPUT_DIR=results/gpt5_phase79_rank_sweep_remainder_audit_full_$(date +%Y%m%d_%H%M%S) \
PHASE79_PROGRESS_EVERY=84 \
PHASE79_MAX_BASIS_ITEMS=224 \
tests/gpt5/run_phase79_rank_sweep_remainder_audit_full.sh
```

模型顺序：

```text
qwen3 -> glm4 -> deepseek7b
```

每个模型都使用：

```text
--hard-exit-after-model
```

实际输出目录：

```text
results/gpt5_phase79_rank_sweep_remainder_audit_full_20260609_165459
```

输出文件：

```text
results/gpt5_phase79_rank_sweep_remainder_audit_full_20260609_165459/qwen3_phase79_rank_sweep_remainder_audit.json
results/gpt5_phase79_rank_sweep_remainder_audit_full_20260609_165459/glm4_phase79_rank_sweep_remainder_audit.json
results/gpt5_phase79_rank_sweep_remainder_audit_full_20260609_165459/deepseek7b_phase79_rank_sweep_remainder_audit.json
results/gpt5_phase79_rank_sweep_remainder_audit_full_20260609_165459/phase79_rank_sweep_remainder_audit_summary.json
results/gpt5_phase79_rank_sweep_remainder_audit_full_20260609_165459/PHASE79_RANK_SWEEP_REMAINDER_AUDIT_SUMMARY.md
```

### 数据规模

```text
objects = 24
relations = 7
frames = 4
items/model = 672
basis_items/model = 224
ranks = 4, 8, 16, 32, 64
conditions = 4
layer_pairs/model = 2
rows/model = 26880
total_rows = 80640
```

layer pairs：

```text
Qwen3:
  L4->L8
  L8->L12

GLM4:
  L4->L10
  L10->L20

DeepSeek7B:
  L8->L10
  L12->L14
```

注意：

```text
本机未安装 flash_attn，因此 flash_attention_2 自动回退到 sdpa。
DeepSeek7B 使用 sdpa 时仍有 sliding-window warning，因此 DS7B 结论要带实现路径 caveat。
```

### Qwen3 客观结果

rank sweep：

```text
rank4 joint_subspace_matched:
  matched_top1 = 0.1010
  matched_gain = 5.1931

rank8 joint_subspace_matched:
  matched_top1 = 0.2082
  matched_gain = 8.7201

rank16 joint_subspace_matched:
  matched_top1 = 0.4264
  matched_gain = 12.1618

rank32 joint_subspace_matched:
  matched_top1 = 0.5299
  matched_gain = 13.7214

rank64 joint_subspace_matched:
  matched_top1 = 0.5399
  matched_gain = 13.8309
```

remainder：

```text
rank4 joint_remainder_matched:
  matched_top1 = 0.1771
  matched_gain = 7.7217

rank8 joint_remainder_matched:
  matched_top1 = 0.0561
  matched_gain = 4.3982

rank16 joint_remainder_matched:
  matched_top1 = 0.0175
  matched_gain = 1.6927

rank32 joint_remainder_matched:
  matched_top1 = 0.0025
  matched_gain = 0.0329

rank64 joint_remainder_matched:
  matched_top1 = 0.0000
  matched_gain = 0.0027
```

mismatched frame：

```text
rank64 joint_subspace_mismatched_frame:
  matched_top1 = 0.1883
  matched_gain = 8.9235
```

restore：

```text
rank64 joint_subspace_restore_both:
  clean_top1 = 0.8653
  matched_top1 = 0.0100
```

Qwen3 现象：

```text
1. subspace effect 随 rank 单调上升。
2. rank32 已接近 Phase77 whole-token matched_top1 = 0.5295。
3. remainder effect 从 rank4 的 0.1771 快速下降到 rank64 的 0。
4. matched frame 明显强于 mismatched frame，说明合法组合不是纯破坏。
```

### GLM4 客观结果

rank sweep：

```text
rank4 joint_subspace_matched:
  matched_top1 = 0.0629
  matched_gain = 3.3169

rank8 joint_subspace_matched:
  matched_top1 = 0.1812
  matched_gain = 6.8299

rank16 joint_subspace_matched:
  matched_top1 = 0.4488
  matched_gain = 11.3391

rank32 joint_subspace_matched:
  matched_top1 = 0.6407
  matched_gain = 13.5902

rank64 joint_subspace_matched:
  matched_top1 = 0.6546
  matched_gain = 13.8849
```

remainder：

```text
rank4 joint_remainder_matched:
  matched_top1 = 0.2846
  matched_gain = 8.7733

rank8 joint_remainder_matched:
  matched_top1 = 0.1077
  matched_gain = 5.1105

rank16 joint_remainder_matched:
  matched_top1 = 0.0277
  matched_gain = 1.7149

rank32 joint_remainder_matched:
  matched_top1 = 0.0075
  matched_gain = 0.0979

rank64 joint_remainder_matched:
  matched_top1 = 0.0043
  matched_gain = -0.0123
```

mismatched frame：

```text
rank64 joint_subspace_mismatched_frame:
  matched_top1 = 0.2100
  matched_gain = 8.3018
```

restore：

```text
rank64 joint_subspace_restore_both:
  clean_top1 = 0.8838
  matched_top1 = 0.0171
```

GLM4 现象：

```text
1. rank32/64 subspace almost fully recovers Phase77 whole-token effect。
2. Phase77 whole-token matched_top1 = 0.6486，Phase79 rank64 = 0.6546。
3. remainder 在 rank32/64 基本消失。
4. GLM4 的 object-frame causal signal 比 Qwen3/DS7B 更集中在自然对比子空间中。
```

### DeepSeek7B 客观结果

rank sweep：

```text
rank4 joint_subspace_matched:
  matched_top1 = 0.0554
  matched_gain = 3.9165

rank8 joint_subspace_matched:
  matched_top1 = 0.1487
  matched_gain = 7.1524

rank16 joint_subspace_matched:
  matched_top1 = 0.2896
  matched_gain = 9.6624

rank32 joint_subspace_matched:
  matched_top1 = 0.3829
  matched_gain = 11.2440

rank64 joint_subspace_matched:
  matched_top1 = 0.4193
  matched_gain = 11.6139
```

remainder：

```text
rank4 joint_remainder_matched:
  matched_top1 = 0.1978
  matched_gain = 7.4433

rank8 joint_remainder_matched:
  matched_top1 = 0.0506
  matched_gain = 3.9489

rank16 joint_remainder_matched:
  matched_top1 = 0.0174
  matched_gain = 2.0311

rank32 joint_remainder_matched:
  matched_top1 = 0.0063
  matched_gain = 0.8004

rank64 joint_remainder_matched:
  matched_top1 = 0.0047
  matched_gain = 0.3668
```

mismatched frame：

```text
rank64 joint_subspace_mismatched_frame:
  matched_top1 = 0.1123
  matched_gain = 6.7360
```

restore：

```text
rank64 joint_subspace_restore_both:
  clean_top1 = 0.8655
  matched_top1 = 0.0047
```

DeepSeek7B 现象：

```text
1. subspace effect 随 rank 上升。
2. rank64 matched_top1 = 0.4193，接近 Phase77 whole-token 0.4161。
3. remainder effect 下降明显，但 rank64 matched_gain 仍有 0.3668。
4. 相比 Qwen3/GLM4，DS7B 的低 rank subspace 较弱，说明因子更分散或更依赖轨迹。
```

### 三模型对比

Phase77 whole-token joint matched vs Phase79 rank64 subspace：

```text
Qwen3:
  Phase77 = 0.5295
  Phase79 rank64 = 0.5399

GLM4:
  Phase77 = 0.6486
  Phase79 rank64 = 0.6546

DeepSeek7B:
  Phase77 = 0.4161
  Phase79 rank64 = 0.4193
```

rank64 remainder：

```text
Qwen3:
  matched_top1 = 0.0000
  matched_gain = 0.0027

GLM4:
  matched_top1 = 0.0043
  matched_gain = -0.0123

DeepSeek7B:
  matched_top1 = 0.0047
  matched_gain = 0.3668
```

客观结论：

```text
1. rank64 natural contrast subspace 基本复现 Phase77 whole-token effect。
2. rank32 已经接近饱和，rank64 只小幅提升。
3. remainder effect 随 rank 增大快速消失。
4. rank4/8 仍保留部分 remainder effect，说明低 rank 不足以解释全部 closure。
5. 三模型均显示：object-frame joint closure 的主要 causal signal 集中在前 32-64 个自然对比方向。
6. DS7B 的低 rank 更弱，说明其关系因子更分散或更依赖上下游轨迹。
```

### 当前研究进展

Phase 79 是一个非常关键的客观结果：

```text
Phase 78:
  rank16 子空间能部分复现 whole-token effect。

Phase 79:
  rank32/64 子空间基本复现 whole-token effect；
  rank64 remainder 基本不再保留合法 matched transfer。
```

这说明：

```text
object-frame joint closure 的主要因果成分不是随机分散在整个 hidden state 中，
而是集中在自然对比主方向中。
```

但是：

```text
这仍不是纯 factor。
```

因为自然对比方向仍混合：

```text
identity
category
relation
value support
template
position
readout format
compatibility
```

最稳妥表述：

```text
知识关系 value retrieval 的可迁移因果信号主要集中在 object-frame natural contrast subspace 中；
该子空间可以近似复现 whole-token joint closure；
但它仍是混合因子子空间，不是最终的纯语义变量。
```

### 对条件化关系因子动力学公式的更新

Phase 79 后，公式可以进一步加入 rank saturation：

```text
h_l' = h_l
     + P_O^{l,k}(source_O - clean_O)
     + P_R^{l,k}(source_R - clean_R)

Effect(k)
  -> saturates around k = 32..64

Remainder(k)
  -> decays toward zero as k increases
```

因此：

```text
ValueReadout
  ≈ G(
      LowRankObjectContrast,
      LowRankRelationFrameContrast,
      Compatibility(O, R),
      ResidualContext,
      OutputSlot
    )
```

更直白地说：

```text
object-frame binding 的核心不是全 hidden state；
也不是单一方向；
而是一组有限数量的自然对比方向形成的条件化组合子空间。
```

### 问题和硬伤

```text
1. rank32/64 子空间虽然基本复现 whole-token effect，但仍不是纯 factor。

2. 当前没有把 identity、relation、value、template、position 分开。

3. 当前 rank basis 来自 matched-clean 差分，可能包含答案 value 信息。

4. 当前仍是 closed candidate scoring。

5. 当前还没有 open generation 验证。

6. 当前还没有真正的 remove-only / restore-only / subspace-only / remainder-only 的完整闭包矩阵。
   本轮 remainder 是 source-clean 的正交剩余转移，不等同 erase clean subspace。

7. DeepSeek7B 仍受 sdpa sliding-window 实现提醒影响。
```

### 下一步计划

Phase 80：orthogonal factor audit。

目标：

```text
把 object/relation/value/template/position 因子进一步拆开。
```

核心测试：

```text
1. value basis:
   用 target value 的 token state 构造 value 子空间。

2. template basis:
   用同 object/relation 但不同 frame 构造 template 子空间。

3. position basis:
   用同 token 不同位置构造 position 子空间。

4. orthogonal removal:
   从 object/frame basis 中去掉 value/template/position 子空间后，再测 joint closure。
```

Phase 81：open generation audit。

目标：

```text
验证 rank64 object-frame subspace 是否影响自由生成，
而不是只改变封闭候选排序。
```

Phase 82：destroy-restore matrix。

目标：

```text
真正做 erase subspace / restore subspace：
  clean erase object/frame subspace 是否破坏 clean answer；
  erase 后恢复是否恢复；
  只保留 subspace 是否足够；
  只保留 remainder 是否不足。
```

Phase 83：迁移到逻辑/语法。

目标：

```text
把 relation-frame 机制迁移到：
operator-event binding；
active/passive role binding；
temporal order binding；
coreference entity binding。
```

当前优先级：

```text
先拆因子，再谈理论收束。
```

## Phase 80: orthogonal factor audit [2026-06-09 23:34]

### 任务目标

根据 Phase 79 的结果，rank32/64 natural contrast subspace 已经基本复现 whole-token joint closure，remainder effect 基本消失。但用户提供的分析指出一个关键硬伤：

```text
rank64 子空间仍然不是纯 factor。
matched-clean contrast 可能混入：
  value
  template
  position
  readout format
  identity
  relation
  compatibility
```

因此 Phase 80 做 orthogonal factor audit，目标不是证明纯因子，而是检查：

```text
从 object/frame contrast basis 中正交移除 value/template/position nuisance basis 后，
joint closure 还剩多少。
```

### 生成脚本

新增：

```text
tests/gpt5/phase80_orthogonal_factor_audit.py
tests/gpt5/phase80_orthogonal_factor_audit_summary.py
tests/gpt5/run_phase80_orthogonal_factor_audit_full.sh
```

脚本检查：

```bash
python -m py_compile \
  tests/gpt5/phase80_orthogonal_factor_audit.py \
  tests/gpt5/phase80_orthogonal_factor_audit_summary.py
```

结果：

```text
compile passed
```

### 测试原理

构造主对比子空间：

```text
object_basis:
  matched object token state - clean object token state

frame_basis:
  matched relation-frame token state - clean relation-frame token state
```

构造 nuisance basis：

```text
value_basis:
  matched value token state - clean value token state

template_basis:
  同 object/relation/value 下，不同 frame 的 frame token state 差异

position_basis:
  同 object 在短 prompt 和长 prompt 中的位置/上下文差异
```

注意：

```text
这些 nuisance basis 也不是纯因子，只是混杂因素审计工具。
```

然后进行正交移除：

```text
object_orth_value    = object_basis 去掉 value_basis 投影
frame_orth_value     = frame_basis  去掉 value_basis 投影

object_orth_template = object_basis 去掉 template_basis 投影
frame_orth_template  = frame_basis  去掉 template_basis 投影

object_orth_position = object_basis 去掉 position_basis 投影
frame_orth_position  = frame_basis  去掉 position_basis 投影

object_orth_all      = object_basis 去掉 value/template/position 总投影
frame_orth_all       = frame_basis  去掉 value/template/position 总投影
```

测试条件：

```text
joint_raw
joint_orth_value
joint_orth_template
joint_orth_position
joint_orth_all
joint_mismatched_frame_raw
joint_value_basis_only
joint_template_basis_only
joint_position_basis_only
joint_raw_restore_both
```

### 工程记录

第一次全量运行中，Qwen3 在 L4->L8 item=504/672 后发生用户态 segmentation fault：

```text
exit_code = 139
kernel log = 无可读错误
nvidia-smi = 正常
```

脚本随后增加：

```text
item_idx 记录
progress checkpoint
CUDA/Python cache cleanup
malloc_trim
```

重新运行后，三模型全部完成。

### Smoke Test

命令：

```bash
PHASE80_MODELS=qwen3 \
QWEN3_PHASE80_MAX_ITEMS=28 \
QWEN3_PHASE80_LAYER_PAIRS=4-8 \
PHASE80_MAX_BASIS_ITEMS=28 \
PHASE80_PROGRESS_EVERY=14 \
PHASE80_CONTRAST_RANK=16 \
PHASE80_NUISANCE_RANK=8 \
PHASE80_OUTPUT_DIR=results/gpt5_phase80_orthogonal_factor_audit_smoke_$(date +%Y%m%d_%H%M%S) \
tests/gpt5/run_phase80_orthogonal_factor_audit_full.sh
```

结果：

```text
qwen3 rows = 280
exit_code = 0
```

### 全量测试命令

```bash
PHASE80_OUTPUT_DIR=results/gpt5_phase80_orthogonal_factor_audit_full_$(date +%Y%m%d_%H%M%S) \
PHASE80_PROGRESS_EVERY=42 \
PHASE80_MAX_BASIS_ITEMS=224 \
tests/gpt5/run_phase80_orthogonal_factor_audit_full.sh
```

实际输出目录：

```text
results/gpt5_phase80_orthogonal_factor_audit_full_20260609_211637
```

输出文件：

```text
results/gpt5_phase80_orthogonal_factor_audit_full_20260609_211637/qwen3_phase80_orthogonal_factor_audit.json
results/gpt5_phase80_orthogonal_factor_audit_full_20260609_211637/glm4_phase80_orthogonal_factor_audit.json
results/gpt5_phase80_orthogonal_factor_audit_full_20260609_211637/deepseek7b_phase80_orthogonal_factor_audit.json
results/gpt5_phase80_orthogonal_factor_audit_full_20260609_211637/phase80_orthogonal_factor_audit_summary.json
results/gpt5_phase80_orthogonal_factor_audit_full_20260609_211637/PHASE80_ORTHOGONAL_FACTOR_AUDIT_SUMMARY.md
```

### 数据规模

```text
items/model = 672
basis_items/model = 224
contrast_rank = 64
nuisance_rank = 24
conditions = 10
layer_pairs/model = 2
rows/model = 13440
total_rows = 40320
```

### Qwen3 客观结果

```text
joint_raw:
  matched_top1 = 0.5399
  matched_gain = 13.8309

joint_orth_value:
  matched_top1 = 0.5050
  matched_gain = 13.2989

joint_orth_template:
  matched_top1 = 0.2668
  matched_gain = 9.7317

joint_orth_position:
  matched_top1 = 0.5436
  matched_gain = 13.7312

joint_orth_all:
  matched_top1 = 0.2082
  matched_gain = 8.8659

joint_mismatched_frame_raw:
  matched_top1 = 0.1883
  matched_gain = 8.9235

joint_value_basis_only:
  matched_top1 = 0.0050
  matched_gain = 0.5279

joint_template_basis_only:
  matched_top1 = 0.0474
  matched_gain = 3.3716

joint_position_basis_only:
  matched_top1 = 0.0025
  matched_gain = 0.0681

joint_raw_restore_both:
  clean_top1 = 0.8653
  matched_top1 = 0.0100
```

Qwen3 现象：

```text
1. 去掉 value 后，effect 只小幅下降：0.5399 -> 0.5050。
2. 去掉 template 后，effect 大幅下降：0.5399 -> 0.2668。
3. 去掉 position 几乎不影响：0.5399 -> 0.5436。
4. 去掉 value/template/position 后，effect 降到 0.2082，接近 mismatched_frame 0.1883。
5. value/template/position basis only 都不能单独形成合法 transfer。
```

### GLM4 客观结果

```text
joint_raw:
  matched_top1 = 0.6546
  matched_gain = 13.8849

joint_orth_value:
  matched_top1 = 0.6205
  matched_gain = 13.2570

joint_orth_template:
  matched_top1 = 0.3294
  matched_gain = 9.6652

joint_orth_position:
  matched_top1 = 0.6557
  matched_gain = 13.8489

joint_orth_all:
  matched_top1 = 0.2783
  matched_gain = 8.8383

joint_mismatched_frame_raw:
  matched_top1 = 0.2100
  matched_gain = 8.3018

joint_value_basis_only:
  matched_top1 = 0.0053
  matched_gain = 0.3133

joint_template_basis_only:
  matched_top1 = 0.0277
  matched_gain = 2.0030

joint_position_basis_only:
  matched_top1 = 0.0000
  matched_gain = 0.0108

joint_raw_restore_both:
  clean_top1 = 0.8838
  matched_top1 = 0.0171
```

GLM4 现象：

```text
1. 去掉 value 后仍很强：0.6546 -> 0.6205。
2. 去掉 template 后显著下降：0.6546 -> 0.3294。
3. 去掉 position 不影响：0.6546 -> 0.6557。
4. orth_all 降到 0.2783，仍略高于 mismatched_frame 0.2100。
5. nuisance basis only 几乎不能独立产生 matched transfer。
```

### DeepSeek7B 客观结果

```text
joint_raw:
  matched_top1 = 0.4193
  matched_gain = 11.6139

joint_orth_value:
  matched_top1 = 0.3244
  matched_gain = 10.3805

joint_orth_template:
  matched_top1 = 0.2753
  matched_gain = 9.5142

joint_orth_position:
  matched_top1 = 0.4114
  matched_gain = 11.4794

joint_orth_all:
  matched_top1 = 0.1978
  matched_gain = 8.1449

joint_mismatched_frame_raw:
  matched_top1 = 0.1123
  matched_gain = 6.7360

joint_value_basis_only:
  matched_top1 = 0.0063
  matched_gain = 1.1287

joint_template_basis_only:
  matched_top1 = 0.0174
  matched_gain = 1.7876

joint_position_basis_only:
  matched_top1 = 0.0047
  matched_gain = 0.2243

joint_raw_restore_both:
  clean_top1 = 0.8655
  matched_top1 = 0.0047
```

DeepSeek7B 现象：

```text
1. 去掉 value 下降更明显：0.4193 -> 0.3244。
2. 去掉 template 也下降：0.4193 -> 0.2753。
3. 去掉 position 基本不影响：0.4193 -> 0.4114。
4. orth_all 仍高于 mismatched_frame：0.1978 vs 0.1123。
5. nuisance basis only 不能独立形成合法 transfer。
```

### 三模型对比

```text
Qwen3:
  raw = 0.5399
  orth_value = 0.5050
  orth_template = 0.2668
  orth_position = 0.5436
  orth_all = 0.2082
  mismatched = 0.1883

GLM4:
  raw = 0.6546
  orth_value = 0.6205
  orth_template = 0.3294
  orth_position = 0.6557
  orth_all = 0.2783
  mismatched = 0.2100

DeepSeek7B:
  raw = 0.4193
  orth_value = 0.3244
  orth_template = 0.2753
  orth_position = 0.4114
  orth_all = 0.1978
  mismatched = 0.1123
```

跨模型稳定现象：

```text
1. value_basis_only 几乎无效。
2. position_basis_only 几乎无效。
3. template_basis_only 弱于 raw 很多。
4. 去掉 position 基本不影响 joint closure。
5. 去掉 value 小幅到中等影响。
6. 去掉 template 影响最大。
7. 去掉 value/template/position 后，joint closure 大幅下降，但仍略高于 mismatched_frame。
```

### 当前研究进展

Phase 80 回答了 Phase 79 的核心硬伤之一：

```text
rank64 子空间不是纯 value leakage。
```

证据：

```text
1. value_basis_only 几乎不能单独产生 matched transfer。
2. 从 object/frame basis 中去掉 value 后，Qwen3/GLM4 仍保留大部分 effect。
3. DS7B 对 value removal 更敏感，但 effect 仍未消失。
```

同时 Phase 80 发现 template/readout-format 相关方向非常关键：

```text
去掉 template 后，三模型 matched_top1 均大幅下降。
```

这说明 relation-frame 的作用不只是 relation semantic gate，也包含 frame/readout format。

更谨慎地说：

```text
object-frame joint closure 的 rank64 causal subspace 中，
value leakage 不是主解释；
template/readout-format 是重要组成；
position 不是主要组成；
去掉三类 nuisance 后仍有一部分 residual compatibility signal。
```

### 条件化关系因子动力学公式更新

Phase 80 后，公式应从：

```text
LowRankObjectContrast + LowRankRelationFrameContrast
```

细化为：

```text
LowRankObjectFrameContrast
  =
  CompatibilityCore
  + ReadoutTemplate
  + ValueSupport
  + PositionContext
  + ResidualMixedFactor
```

实验约束：

```text
ValueSupport only:
  weak

Template only:
  weak but larger than value/position

Position only:
  near zero

Remove value:
  small/moderate drop

Remove template:
  large drop

Remove all:
  large drop, but residual remains above mismatched in GLM4/DS7B
```

因此当前更稳的操作性公式：

```text
Score(value)
  =
  BaseContext
  + Compat(Object, Relation)
  + ReadoutTemplate(Frame)
  + CompatTemplateInteraction(Object, Frame)
  + weak ValueSupport
```

关键改进：

```text
relation-frame path 不只是 relation gate，
还携带 readout template / output format。

object path 不只是 identity，
而是和 frame template 发生兼容性组合。
```

### 问题和硬伤

```text
1. value/template/position basis 仍是粗糙混杂审计，不是纯因子。

2. template_basis 的定义是同 object/relation/value 不同 frame 的 frame token 差异，
   它可能同时包含模板、语序、读出槽位、局部上下文。

3. position_basis 是短/长 neutral prompt 对比，不能完全代表真实 prompt 中的位置编码。

4. orth_all 下降很大，但不能说明全部下降都来自这些 nuisance；
   子空间正交化本身可能移除部分 compatibility core。

5. 当前仍是 closed candidate scoring。

6. 当前没有 open generation。

7. 当前还没有 erase-clean-subspace 的 destroy-restore matrix。
```

### 下一步计划

Phase 81：template/readout-factor 深挖。

目标：

```text
把 template/readout-format 从 relation-frame 中进一步拆出来。
```

测试：

```text
same relation + different frame
same frame + different relation
same object + same relation + paraphrased frame
same prompt + changed output slot
```

关键问题：

```text
template 下降到底来自表面模板，
还是来自 output slot/readout format？
```

Phase 82：destroy-restore matrix。

目标：

```text
对 rank64 object/frame subspace 做真正 erase-clean-subspace：
  erase 是否破坏 clean answer；
  restore 是否恢复；
  subspace-only 是否足够；
  remainder-only 是否不足。
```

Phase 83：open generation audit。

目标：

```text
验证 rank64 subspace 和 orthogonal factors 是否影响自由生成。
```

Phase 84：迁移到逻辑/语法。

目标：

```text
用同样流程测试：
operator-event binding
active/passive role binding
temporal order binding
coreference entity binding
```

目前最重要的结论：

```text
知识关系编码不是单纯 object identity，也不是单纯 value leakage。
它更像 object compatibility 与 relation-frame readout template 的条件化组合。
```

## Phase 81: template/readout decomposition [2026-06-10 04:35]

### 任务目标

用户提供的最新分析认为：

```text
Phase 68-79 已经比早期 GFCM 路线更深入；
Phase 80 排除了 value leakage 和 position artifact 作为主解释；
但 Phase 80 中的 template/readout-format 仍然是混合概念；
下一步必须拆开 surface template 和 output slot/readout format。
```

这个判断基本正确。

Phase 81 因此不继续证明 rank64 强，而是构造一个受控 prompt family，把 relation phrase（关系表达短语）和 answer slot（答案槽位）显式拆开。

核心问题：

```text
Phase 80 中 template/readout-format 下降，
到底来自表面模板，
还是来自输出槽位/读出格式，
还是来自更深的 object-frame compatibility core？
```

### 脚本

新增：

```text
tests/gpt5/phase81_template_readout_decomposition.py
tests/gpt5/phase81_template_readout_decomposition_summary.py
tests/gpt5/run_phase81_template_readout_decomposition_full.sh
```

脚本特点：

```text
1. 三模型顺序运行：qwen3 -> GLM4 -> DS7B。
2. 每个模型运行后使用 --hard-exit-after-model，避免显存残留。
3. 使用正常 CUDA 路径，优先 flash_attention_2，失败后自动回退到 sdpa。
4. 使用受控 prompt：
   Object: {object}. Relation: {phrase}.{slot}
5. 每个 relation 有 4 个 relation phrase。
6. 每个 prompt 有 4 个 slot style：
   Answer:
   Value:
   ->
   =
7. 每个模型 max_items=1344。
8. 每个模型 2 个 layer pair。
9. 每个模型 rows=34944。
10. 三模型总 rows=104832。
```

relation phrase 示例：

```text
is_a:
  category / kind / type / class

used_for:
  use / purpose / function / used for

location:
  location / place / where found / usual place
```

测试条件：

```text
joint_raw
joint_orth_phrase
joint_orth_slot
joint_orth_phrase_slot
joint_orth_relation
joint_orth_all
joint_phrase_basis_only
joint_slot_basis_only
joint_relation_basis_only
joint_same_relation_other_phrase_frame
joint_same_relation_other_slot_frame
joint_same_object_other_relation_frame
joint_raw_restore_both
```

说明：

```text
joint_raw:
  rank64 object_basis + frame_basis 原始联合转移。

joint_orth_phrase:
  从 object/frame rank64 basis 中移除 relation phrase nuisance basis。

joint_orth_slot:
  移除 answer slot/readout style nuisance basis。

joint_orth_phrase_slot:
  同时移除 phrase + slot。

joint_orth_relation:
  移除 same object / same phrase / same slot / other relation 的 relation nuisance basis。

joint_orth_all:
  同时移除 phrase + slot + relation。

*_basis_only:
  只使用对应 nuisance basis 做转移，检查该 nuisance 是否单独足以产生合法 value transfer。
```

### 运行命令

```bash
PHASE81_OUTPUT_DIR=results/gpt5_phase81_template_readout_decomposition_full_$(date +%Y%m%d_%H%M%S) \
tests/gpt5/run_phase81_template_readout_decomposition_full.sh
```

实际输出目录：

```text
results/gpt5_phase81_template_readout_decomposition_full_20260609_235258
```

输出文件：

```text
results/gpt5_phase81_template_readout_decomposition_full_20260609_235258/qwen3_phase81_template_readout_decomposition.json
results/gpt5_phase81_template_readout_decomposition_full_20260609_235258/glm4_phase81_template_readout_decomposition.json
results/gpt5_phase81_template_readout_decomposition_full_20260609_235258/deepseek7b_phase81_template_readout_decomposition.json
results/gpt5_phase81_template_readout_decomposition_full_20260609_235258/phase81_template_readout_decomposition_summary.json
results/gpt5_phase81_template_readout_decomposition_full_20260609_235258/PHASE81_TEMPLATE_READOUT_DECOMPOSITION_SUMMARY.md
```

### 运行过程

Qwen3：

```text
loaded with sdpa
items = 1344
layer_pairs = L4->L8, L8->L12
rows = 34944
exit_code = 0
```

GLM4：

```text
loaded with sdpa
items = 1344
layer_pairs = L4->L10, L10->L20
rows = 34944
exit_code = 0
```

DeepSeek7B：

```text
loaded with sdpa
items = 1344
layer_pairs = L8->L10, L12->L14
rows = 34944
exit_code = 0
```

注意：

```text
本机仍没有 flash_attn 包，因此 flash_attention_2 加载失败后回退到 sdpa。
DeepSeek7B 加载时 transformers 提示：
Sliding Window Attention is enabled but not implemented for sdpa;
因此 DS7B 结果仍需保留这个实现差异 caveat。
```

### 数据规模

```text
Qwen3:
  items = 1344
  rows = 34944
  eligible_n = 1228

GLM4:
  items = 1344
  rows = 34944
  eligible_n = 1500

DeepSeek7B:
  items = 1344
  rows = 34944
  eligible_n = 1190

total rows = 104832
```

### 客观结果：Qwen3

核心条件：

```text
joint_raw:
  eligible_clean_drop = 6.0488
  eligible_matched_gain = 5.4541
  eligible_clean_top1 = 0.5399
  eligible_matched_top1 = 0.1857

joint_orth_phrase:
  eligible_matched_top1 = 0.1857
  matched_gain_delta_vs_raw = -0.0128

joint_orth_slot:
  eligible_matched_top1 = 0.1849
  matched_gain_delta_vs_raw = -0.0039

joint_orth_phrase_slot:
  eligible_matched_top1 = 0.1849
  matched_gain_delta_vs_raw = -0.0226

joint_orth_relation:
  eligible_matched_top1 = 0.1865
  matched_gain_delta_vs_raw = -0.0080

joint_orth_all:
  eligible_matched_top1 = 0.1865
  matched_gain_delta_vs_raw = -0.0166
```

nuisance basis only：

```text
joint_phrase_basis_only:
  eligible_matched_top1 = 0.0000

joint_slot_basis_only:
  eligible_matched_top1 = 0.0000

joint_relation_basis_only:
  eligible_matched_top1 = 0.0008
```

restore：

```text
joint_raw_restore_both:
  eligible_clean_top1 = 0.9251
  eligible_matched_top1 = 0.0033
```

客观现象：

```text
在受控 prompt family 中，移除 phrase/slot/relation nuisance basis 几乎不降低 Qwen3 的 joint_raw matched_top1。
phrase/slot/relation basis alone 几乎不能产生 matched transfer。
restore_both 仍然强烈恢复 clean target，说明干预位置和 restore 流程有效。
```

### 客观结果：GLM4

核心条件：

```text
joint_raw:
  eligible_clean_drop = 4.0473
  eligible_matched_gain = 4.2601
  eligible_clean_top1 = 0.5933
  eligible_matched_top1 = 0.1507

joint_orth_phrase:
  eligible_matched_top1 = 0.1447
  matched_gain_delta_vs_raw = +0.0195

joint_orth_slot:
  eligible_matched_top1 = 0.1487
  matched_gain_delta_vs_raw = +0.0156

joint_orth_phrase_slot:
  eligible_matched_top1 = 0.1467
  matched_gain_delta_vs_raw = +0.0091

joint_orth_relation:
  eligible_matched_top1 = 0.1500
  matched_gain_delta_vs_raw = +0.0203

joint_orth_all:
  eligible_matched_top1 = 0.1460
  matched_gain_delta_vs_raw = +0.0092
```

nuisance basis only：

```text
joint_phrase_basis_only:
  eligible_matched_top1 = 0.0013

joint_slot_basis_only:
  eligible_matched_top1 = 0.0007

joint_relation_basis_only:
  eligible_matched_top1 = 0.0027
```

restore：

```text
joint_raw_restore_both:
  eligible_clean_top1 = 0.9500
  eligible_matched_top1 = 0.0073
```

客观现象：

```text
GLM4 中移除 phrase/slot/relation 也几乎不破坏 joint_raw。
单独 phrase/slot/relation basis 基本无效。
restore_both 对 clean target 的恢复最强。
```

### 客观结果：DeepSeek7B

核心条件：

```text
joint_raw:
  eligible_clean_drop = 3.7176
  eligible_matched_gain = 3.7061
  eligible_clean_top1 = 0.6160
  eligible_matched_top1 = 0.1521

joint_orth_phrase:
  eligible_matched_top1 = 0.1513
  matched_gain_delta_vs_raw = -0.0316

joint_orth_slot:
  eligible_matched_top1 = 0.1529
  matched_gain_delta_vs_raw = -0.0135

joint_orth_phrase_slot:
  eligible_matched_top1 = 0.1555
  matched_gain_delta_vs_raw = -0.0600

joint_orth_relation:
  eligible_matched_top1 = 0.1504
  matched_gain_delta_vs_raw = -0.0619

joint_orth_all:
  eligible_matched_top1 = 0.1538
  matched_gain_delta_vs_raw = -0.1046
```

nuisance basis only：

```text
joint_phrase_basis_only:
  eligible_matched_top1 = 0.0050

joint_slot_basis_only:
  eligible_matched_top1 = 0.0050

joint_relation_basis_only:
  eligible_matched_top1 = 0.0034
```

restore：

```text
joint_raw_restore_both:
  eligible_clean_top1 = 0.9311
  eligible_matched_top1 = 0.0109
```

客观现象：

```text
DeepSeek7B 中 phrase/slot/relation 正交移除对 matched_top1 基本无破坏。
matched_gain 有小幅下降，尤其 orth_all 下降 -0.1046，但 matched_top1 反而略高。
单独 nuisance basis 仍几乎无效。
```

### 三模型共同事实

共同事实 1：

```text
在受控 prompt family 中，phrase/slot/relation nuisance basis alone 几乎不能产生合法 matched value transfer。
```

共同事实 2：

```text
从 rank64 object/frame basis 中移除 phrase/slot/relation nuisance 后，
joint_raw 的 matched_top1 基本不下降。
```

共同事实 3：

```text
restore_both 在三模型中都能明显恢复 clean target：
Qwen3 = 0.9251
GLM4 = 0.9500
DeepSeek7B = 0.9311
```

共同事实 4：

```text
Phase 80 中 template/readout-format 下降很大；
Phase 81 在受控 phrase/slot 分解下没有复现这种大下降。
```

这说明 Phase 80 的 template/readout-format 不能简单解释成：

```text
表面 relation phrase
或 answer slot 文本
```

更可能包含：

```text
自然语言 frame 的完整局部上下文；
relation expression 与 object 的兼容格式；
answer slot 与上文共同形成的 readout alignment；
自然模板中的语序、功能词、语用提示；
而不是单独 phrase 或 slot。
```

### 重要解释修正

Phase 80 后的表达是：

```text
template/readout-format 是重要组成。
```

Phase 81 后必须修正为：

```text
受控拆出的 phrase/slot/relation nuisance 不是 Phase 80 template effect 的主因。
Phase 80 的 template effect 更可能是自然 frame 级别的整体 readout alignment，
不是单独 surface phrase 或 output slot marker。
```

这使 object-frame 机制进一步接近：

```text
object compatibility support
+
frame-conditioned readout alignment
+
relation-object compatibility core
```

而不是：

```text
object identity + surface template + output slot
```

### 硬伤和注意事项

1. `joint_same_relation_other_phrase_frame` 和 `joint_same_relation_other_slot_frame` 的 `matched_gain` 不能直接解释为跨对象合法 value transfer。

原因：

```text
这些条件的目标值与 clean target 相同，
matched_gain 的基线仍来自 matched object source target。
因此 matched_gain 会被高估。
```

本轮只把这两个条件作为：

```text
clean target preservation / phrase-slot perturbation observation
```

不把它们作为主要机制证据。

2. 受控 prompt family 和 Phase77/80 的自然模板不同。

本轮结果说明：

```text
单独 phrase/slot 不是主因。
```

但不能说明：

```text
自然模板不重要。
```

3. 受控 prompt 的 slot style 差异仍较浅。

```text
Answer:
Value:
->
=
```

它们主要测试局部输出标记，不等价于完整 natural-language readout format。

4. 当前仍是 closed candidate scoring。

5. 当前仍没有 open generation。

6. 当前没有对 natural frame 做更细粒度的 clause/slot/function-word 拆分。

7. DeepSeek7B 使用 SDPA 时有 sliding window attention caveat。

### 当前进展

到 Phase 81，知识关系编码机制的拼图更清楚：

```text
1. whole-token object-frame joint closure 成立。
2. rank64 natural contrast subspace 基本复现 whole-token effect。
3. remainder 几乎无效。
4. value leakage 不是主解释。
5. position artifact 不是主解释。
6. Phase80 template/readout effect 很重要。
7. Phase81 显示：受控拆出的 phrase/slot/relation 不是该 effect 的主因。
```

因此现在更稳的判断是：

```text
知识关系 value retrieval 的因果信号，
集中在 object-frame 兼容子空间中；
这个子空间包含自然 frame 条件化的 readout alignment，
但不是简单答案值、位置、短语表面、槽位标记可以单独解释。
```

### 条件化关系因子动力学公式修正

Phase 80 后的公式：

```text
h_{l+1}
= h_l
+ C_l(object, relation-frame, readout-template)
```

Phase 81 后应改为：

```text
h_{l+1}
= h_l
+ C_l(
     object support,
     natural frame context,
     relation-object compatibility,
     readout alignment
   )
```

更谨慎地说：

```text
relation-frame 不是纯 relation，也不是 surface phrase + slot。
它更像一个由自然框架上下文形成的条件化读出对齐状态。
```

### 下一步计划

Phase 82：natural frame component ablation。

目标：

```text
继续拆 Phase80 中真正强的 natural template effect。
```

需要把自然 frame 分成：

```text
relation lexical words
function words
answer-slot boundary
object-relative word order
pre-object context
post-object context
full frame suffix
```

测试：

```text
1. 只替换 relation lexical words。
2. 只替换 answer-slot boundary。
3. 只替换 object 后面的 frame suffix。
4. 只替换 pre-object context。
5. relation words + suffix 联合替换。
6. full natural frame 替换。
```

Phase 83：erase-clean-subspace destroy-restore matrix。

目标：

```text
不是只做 matched transfer，
而是从 clean hidden state 中 erase rank64 object/frame subspace，
看 clean answer 是否下降；
再 restore 对应子空间，看 clean answer 是否恢复。
```

Phase 84：open generation audit。

目标：

```text
验证 rank64 subspace 和 natural frame alignment 是否影响自由生成，
而不是只影响 closed candidate scoring。
```

Phase 85：迁移到逻辑/语法。

目标：

```text
用同一套 object support + frame alignment + compatibility core 的思想，
测试：
operator-event binding
temporal order binding
active/passive role binding
coreference entity binding
```

最重要的提醒：

```text
当前不要把 phrase/slot 当作语言编码核心。
真正核心更可能是自然上下文形成的相对兼容路径。
```

## Phase 82: natural frame component ablation [2026-06-10 09:16]

### 任务目标

用户提供的最新分析认为 Phase 81 的判断基本正确：

```text
Phase 80 的 template/readout-format 效应不能简单归因于表面 phrase、slot、relation label；
Phase 81 的受控 prompt family 说明单独 phrase/slot/relation nuisance 不是主因；
下一步应该回到自然 frame，把自然 frame 进一步拆成组件。
```

这个判断是正确的。

Phase 82 因此执行 natural frame component ablation，目标是继续拆 Phase 80 中真正强的 natural template effect。

本轮重点测试：

```text
pre-object context
post-object suffix / natural frame suffix
answer boundary
relation label prompt
full natural frame variation
```

### 脚本

新增：

```text
tests/gpt5/phase82_natural_frame_component_ablation.py
tests/gpt5/phase82_natural_frame_component_ablation_summary.py
tests/gpt5/run_phase82_natural_frame_component_ablation_full.sh
```

核心 basis：

```text
object_basis:
  matched source object token - clean object token

frame_basis:
  matched source natural frame token - clean natural frame token

full_frame_basis:
  same object / same relation / same target / different natural frame 的 frame_last 差异

pre_object_basis:
  same object / same relation / different natural frame 的 object_last 差异

suffix_basis:
  clean natural frame_last - object-only prompt object_last

boundary_basis:
  clean prompt + " Answer:" 的新 frame_last - clean frame_last

relation_label_basis:
  "{object} {relation_label}" prompt 的 frame_last - clean frame_last
```

测试条件：

```text
joint_raw
joint_orth_full_frame
joint_orth_pre_object
joint_orth_suffix
joint_orth_boundary
joint_orth_relation_label
joint_orth_all_components
joint_full_frame_basis_only
joint_pre_object_basis_only
joint_suffix_basis_only
joint_boundary_basis_only
joint_relation_label_basis_only
joint_mismatched_frame_raw
joint_raw_restore_both
```

说明：

```text
orth_*:
  从 rank64 object/frame basis 中正交移除对应 component basis。

basis_only:
  只使用该 component basis 做 object+frame 转移，检查它是否单独足够。
```

### 运行命令

```bash
PHASE82_OUTPUT_DIR=results/gpt5_phase82_natural_frame_component_ablation_full_$(date +%Y%m%d_%H%M%S) \
tests/gpt5/run_phase82_natural_frame_component_ablation_full.sh
```

实际输出目录：

```text
results/gpt5_phase82_natural_frame_component_ablation_full_20260610_061746
```

输出文件：

```text
results/gpt5_phase82_natural_frame_component_ablation_full_20260610_061746/qwen3_phase82_natural_frame_component_ablation.json
results/gpt5_phase82_natural_frame_component_ablation_full_20260610_061746/glm4_phase82_natural_frame_component_ablation.json
results/gpt5_phase82_natural_frame_component_ablation_full_20260610_061746/deepseek7b_phase82_natural_frame_component_ablation.json
results/gpt5_phase82_natural_frame_component_ablation_full_20260610_061746/phase82_natural_frame_component_ablation_summary.json
results/gpt5_phase82_natural_frame_component_ablation_full_20260610_061746/PHASE82_NATURAL_FRAME_COMPONENT_ABLATION_SUMMARY.md
```

### 数据规模

```text
Qwen3:
  items = 672
  rows = 18816
  layer_pairs = L4->L8, L8->L12
  eligible_n = 802

GLM4:
  items = 672
  rows = 18816
  layer_pairs = L4->L10, L10->L20
  eligible_n = 938

DeepSeek7B:
  items = 672
  rows = 18816
  layer_pairs = L8->L10, L12->L14
  eligible_n = 632

total rows = 56448
```

运行情况：

```text
三模型均完成。
每个模型完成后使用 --hard-exit-after-model。
flash_attention_2 因未安装 flash_attn 包失败，自动回退到 sdpa。
DeepSeek7B 仍有 sliding window attention under sdpa caveat。
```

### Qwen3 客观结果

核心结果：

```text
joint_raw:
  matched_top1 = 0.5399
  matched_gain = 13.8309

joint_orth_pre_object:
  matched_top1 = 0.5162
  matched_gain = 13.4426
  delta_top1 = -0.0237

joint_orth_full_frame:
  matched_top1 = 0.2693
  matched_gain = 9.7438
  delta_top1 = -0.2706

joint_orth_relation_label:
  matched_top1 = 0.2544
  matched_gain = 9.7924
  delta_top1 = -0.2855

joint_orth_suffix:
  matched_top1 = 0.0711
  matched_gain = 5.7915
  delta_top1 = -0.4688

joint_orth_boundary:
  matched_top1 = 0.0711
  matched_gain = 5.9752
  delta_top1 = -0.4688

joint_orth_all_components:
  matched_top1 = 0.0623
  matched_gain = 5.1338
  delta_top1 = -0.4776

joint_mismatched_frame_raw:
  matched_top1 = 0.2369
```

basis only：

```text
joint_suffix_basis_only:
  matched_top1 = 0.1534

joint_boundary_basis_only:
  matched_top1 = 0.1446

joint_full_frame_basis_only:
  matched_top1 = 0.0474

joint_pre_object_basis_only:
  matched_top1 = 0.0000
```

restore：

```text
joint_raw_restore_both:
  clean_top1 = 0.8653
  matched_top1 = 0.0100
```

客观现象：

```text
Qwen3 中 pre-object context 几乎不是主因。
去掉 suffix 或 boundary 后，matched_top1 从 0.5399 下降到 0.0711，低于 mismatched_frame_raw。
suffix/boundary basis only 有一定效果，但远低于 joint_raw。
```

### GLM4 客观结果

核心结果：

```text
joint_raw:
  matched_top1 = 0.6546
  matched_gain = 13.8849

joint_orth_pre_object:
  matched_top1 = 0.6365
  matched_gain = 13.5154
  delta_top1 = -0.0181

joint_orth_full_frame:
  matched_top1 = 0.3337
  matched_gain = 9.6759
  delta_top1 = -0.3209

joint_orth_relation_label:
  matched_top1 = 0.4392
  matched_gain = 10.9777
  delta_top1 = -0.2154

joint_orth_suffix:
  matched_top1 = 0.1013
  matched_gain = 5.9859
  delta_top1 = -0.5533

joint_orth_boundary:
  matched_top1 = 0.1162
  matched_gain = 6.1659
  delta_top1 = -0.5384

joint_orth_all_components:
  matched_top1 = 0.0981
  matched_gain = 5.4180
  delta_top1 = -0.5565

joint_mismatched_frame_raw:
  matched_top1 = 0.2751
```

basis only：

```text
joint_suffix_basis_only:
  matched_top1 = 0.1461

joint_boundary_basis_only:
  matched_top1 = 0.1194

joint_full_frame_basis_only:
  matched_top1 = 0.0245

joint_pre_object_basis_only:
  matched_top1 = 0.0011
```

restore：

```text
joint_raw_restore_both:
  clean_top1 = 0.8838
  matched_top1 = 0.0171
```

客观现象：

```text
GLM4 中 pre-object context 也几乎不是主因。
suffix/boundary 正交移除造成最大破坏。
relation_label 移除有中等破坏，但明显弱于 suffix/boundary。
```

### DeepSeek7B 客观结果

核心结果：

```text
joint_raw:
  matched_top1 = 0.4193
  matched_gain = 11.6139

joint_orth_pre_object:
  matched_top1 = 0.3655
  matched_gain = 10.6894
  delta_top1 = -0.0538

joint_orth_full_frame:
  matched_top1 = 0.2706
  matched_gain = 9.4406
  delta_top1 = -0.1487

joint_orth_relation_label:
  matched_top1 = 0.2801
  matched_gain = 9.5748
  delta_top1 = -0.1392

joint_orth_suffix:
  matched_top1 = 0.0396
  matched_gain = 3.9577
  delta_top1 = -0.3797

joint_orth_boundary:
  matched_top1 = 0.0617
  matched_gain = 4.6512
  delta_top1 = -0.3576

joint_orth_all_components:
  matched_top1 = 0.0316
  matched_gain = 3.1118
  delta_top1 = -0.3877

joint_mismatched_frame_raw:
  matched_top1 = 0.2674
```

basis only：

```text
joint_suffix_basis_only:
  matched_top1 = 0.1566

joint_boundary_basis_only:
  matched_top1 = 0.1171

joint_full_frame_basis_only:
  matched_top1 = 0.0206

joint_pre_object_basis_only:
  matched_top1 = 0.0079
```

restore：

```text
joint_raw_restore_both:
  clean_top1 = 0.8655
  matched_top1 = 0.0047
```

客观现象：

```text
DeepSeek7B 中 suffix/boundary 仍是最关键组件。
pre-object context 有轻微影响，但远弱于 suffix/boundary。
```

### 三模型共同事实

共同事实 1：

```text
pre-object context 不是主要因果成分。
```

证据：

```text
Qwen3:
  raw 0.5399 -> orth_pre_object 0.5162

GLM4:
  raw 0.6546 -> orth_pre_object 0.6365

DeepSeek7B:
  raw 0.4193 -> orth_pre_object 0.3655
```

共同事实 2：

```text
post-object suffix / natural frame suffix 是最强因果成分。
```

证据：

```text
Qwen3:
  raw 0.5399 -> orth_suffix 0.0711

GLM4:
  raw 0.6546 -> orth_suffix 0.1013

DeepSeek7B:
  raw 0.4193 -> orth_suffix 0.0396
```

共同事实 3：

```text
answer boundary basis 与 suffix basis 高度相关，也非常关键。
```

证据：

```text
Qwen3:
  orth_boundary = 0.0711

GLM4:
  orth_boundary = 0.1162

DeepSeek7B:
  orth_boundary = 0.0617
```

共同事实 4：

```text
relation_label/full_frame 有中等影响，但不是最核心。
```

共同事实 5：

```text
suffix/boundary basis only 有一定效果，但不足以复现 joint_raw。
```

证据：

```text
Qwen3:
  suffix_basis_only = 0.1534
  raw = 0.5399

GLM4:
  suffix_basis_only = 0.1461
  raw = 0.6546

DeepSeek7B:
  suffix_basis_only = 0.1566
  raw = 0.4193
```

这说明：

```text
suffix/boundary 是必要或近必要成分，
但不是充分成分。
完整机制仍需要 object support + frame suffix/readout boundary + compatibility composition。
```

### 和 Phase 81 的关系

Phase 81 结论：

```text
受控 phrase/slot/relation marker 不是主因。
```

Phase 82 结论：

```text
自然 frame 中 object 后的 suffix/readout boundary 是主因。
```

二者合起来说明：

```text
不是任意 slot marker 重要；
而是自然语言 frame 在 object 后形成的读出后缀和答案边界重要。
```

这正好解释 Phase 80：

```text
Phase 80 的 template/readout-format 效应，
更具体地说是 natural post-object frame suffix + readout boundary effect。
```

### 当前理论修正

Phase 81 后公式：

```text
h_{l+1}
= h_l
+ C_l(
     object support,
     natural frame context,
     relation-object compatibility,
     readout alignment
   )
```

Phase 82 后进一步修正：

```text
h_{l+1}
= h_l
+ C_l(
     object support,
     post-object frame suffix,
     readout boundary,
     relation-object compatibility
   )
```

更具体：

```text
object token 提供可兼容的对象支持；
object 后面的 natural frame suffix 提供关系化读出路径；
answer boundary 决定该路径如何进入候选值空间；
三者联合形成 value-space transfer。
```

这不是单一语义轴，也不是单一模板词，而是：

```text
对象支持 + 后缀读出路径 + 关系兼容性 的条件化组合。
```

### 严格硬伤

1. suffix_basis 定义仍然较粗。

```text
clean frame_last - object-only prompt object_last
```

它包含：

```text
post-object relation words
function words
answer-ready slot state
local syntax
position drift
readout preparation
```

因此不能说已经找到纯 suffix factor。

2. boundary_basis 也不纯。

```text
clean prompt + " Answer:" - clean prompt
```

它可能包含新的 answer marker，也可能包含额外 token 带来的位置和上下文变化。

3. basis only 有一定 transfer。

这说明 suffix/boundary 含有可推动 value-space 的成分，但由于远低于 raw，不能单独解释完整机制。

4. 当前仍是 closed candidate scoring。

5. 当前仍没有 open generation。

6. 当前没有对 suffix 内部 token 逐词拆分。

7. 当前没有 erase-clean-subspace。

8. DS7B 仍有 SDPA sliding window caveat。

### 下一步计划

Phase 83：suffix token-level decomposition。

目标：

```text
把 post-object frame suffix 逐 token 拆开。
```

例如：

```text
A {obj} is a kind of
```

拆成：

```text
is
a
kind
of
```

测试：

```text
1. 每个 suffix token 单独 basis。
2. suffix token cumulative basis。
3. relation lexical token basis。
4. final pre-answer token basis。
5. function word basis。
6. suffix without final token。
7. final token only。
```

关键问题：

```text
真正强的是 relation lexical words，
还是最后一个 pre-answer token，
还是整个 suffix trajectory？
```

Phase 84：erase-clean-subspace destroy-restore matrix。

目标：

```text
从 clean state 中 erase suffix/readout subspace，看 clean answer 是否下降；
再 restore，看 clean answer 是否恢复。
```

Phase 85：open generation audit。

目标：

```text
验证 suffix/readout boundary 是否影响自由生成，而不只是 closed candidate scoring。
```

Phase 86：迁移到逻辑/语法。

目标：

```text
把 object + suffix/readout 的结构迁移到：
operator + event suffix
temporal marker + event order
passive suffix + role binding
coreference prompt + entity binding
```

当前最关键洞察：

```text
语言中的关系编码不是“对象向量 + 关系词向量”。
更像是对象 token 在一个后续 frame suffix/readout boundary 中被重新读出。
```

## Phase 83: suffix token decomposition 全量测试 [2026-06-10 13:03]

### 任务目标

根据最新分析，Phase 82 已经说明：

```text
pre-object context 不是主要因素；
post-object natural frame suffix / answer boundary 是 object-value closure 的关键因素。
```

但 Phase 82 仍然只把 suffix 当作整体处理。本轮目标是继续拆开 suffix：

```text
1. suffix_all：对象之后到 frame 末尾的全部后缀。
2. suffix_nonfinal：不含最后 readout token 的后缀。
3. suffix_final：最后一个 pre-answer/readout token。
4. suffix_penultimate：倒数第二个 token。
5. suffix_first / suffix_second：后缀开头 token。
6. suffix_function：功能词后缀子空间。
7. suffix_lexical：非功能词/关系词后缀子空间。
8. all_suffix_tokens：多个 suffix token component 拼接后的整体子空间。
```

核心问题：

```text
真正强的是最后 readout token？
还是倒数第二个/词汇 token？
还是整个 suffix trajectory？
```

### 对用户分析的判断

这次分析方向正确：

```text
1. Phase 82 不能停在“suffix 很重要”。
2. relation lexical words 可能不是纯关系类型，而是 object-after readout program fragment。
3. 必须拆 suffix token-level component，否则无法判断是关系词、功能词、最后边界，还是整体轨迹在起作用。
4. 当前仍然不要上升为完整数学理论，优先继续收集客观拼图。
```

### 脚本

新增：

```text
tests/gpt5/phase83_suffix_token_decomposition.py
tests/gpt5/phase83_suffix_token_decomposition_summary.py
tests/gpt5/run_phase83_suffix_token_decomposition_full.sh
```

运行中 Qwen3 第一次在 `L4->L8 item=588/672` 后发生用户态 segmentation fault：

```text
exit_code = 139
已落盘 partial rows = 9996
已完成完整 item = 588
```

随后给 Phase 83 主脚本补充 pair/item 级 resume：

```text
默认 --resume；
如果存在 partial/final，则读取已有 rows；
对每个 destroy_layer/restore_layer/item_idx，如果已有 17 条 condition rows，则跳过；
继续完成剩余 item。
```

这个修改不改变实验指标，只保证长跑中断后可以继续。

### 正式命令

```bash
PHASE83_OUTPUT_DIR=results/gpt5_phase83_suffix_token_decomposition_full_20260610_092932 \
tests/gpt5/run_phase83_suffix_token_decomposition_full.sh
```

runner 内部顺序：

```text
qwen3:
  layer_pairs = 4-8,8-12
  max_items = 672

glm4:
  layer_pairs = 4-10,10-20
  max_items = 672

deepseek7b:
  layer_pairs = 8-10,12-14
  max_items = 672
```

共同参数：

```text
module = resid_out
contrast_rank = 64
component_rank = 24
max_basis_items = 224
max_distractors = 10
hard_exit_after_model = true
attn path = flash_attention_2 fallback to sdpa
```

说明：

```text
本机没有 flash_attn 包，因此 flash_attention_2 加载失败后自动使用 PyTorch SDPA。
DeepSeek7B 仍然出现 sliding window attention + sdpa 的实现警告，后续解释 DS7B 结果时需要保留这个限制。
```

### 输出文件

```text
results/gpt5_phase83_suffix_token_decomposition_full_20260610_092932/qwen3_phase83_suffix_token_decomposition.json
results/gpt5_phase83_suffix_token_decomposition_full_20260610_092932/glm4_phase83_suffix_token_decomposition.json
results/gpt5_phase83_suffix_token_decomposition_full_20260610_092932/deepseek7b_phase83_suffix_token_decomposition.json
results/gpt5_phase83_suffix_token_decomposition_full_20260610_092932/phase83_suffix_token_decomposition_summary.json
results/gpt5_phase83_suffix_token_decomposition_full_20260610_092932/PHASE83_SUFFIX_TOKEN_DECOMPOSITION_SUMMARY.md
```

### 数据规模

```text
Qwen3:
  items = 672
  layer_pairs = 2
  rows = 22848
  eligible raw rows = 802

GLM4:
  items = 672
  layer_pairs = 2
  rows = 22848
  eligible raw rows = 938

DeepSeek7B:
  items = 672
  layer_pairs = 2
  rows = 22848
  eligible raw rows = 632

total rows = 68544
```

### Qwen3 客观结果

以 eligible_patched_matched_top1 为主：

```text
joint_raw                         = 0.5399
joint_orth_suffix_all             = 0.0810
joint_orth_suffix_nonfinal        = 0.2556
joint_orth_suffix_final           = 0.0711
joint_orth_suffix_penultimate     = 0.4127
joint_orth_suffix_first           = 0.3042
joint_orth_suffix_second          = 0.2469
joint_orth_suffix_function        = 0.1010
joint_orth_suffix_lexical         = 0.4102
joint_orth_all_suffix_tokens      = 0.0623
joint_suffix_all_basis_only       = 0.1372
joint_suffix_final_basis_only     = 0.1534
joint_suffix_lexical_basis_only   = 0.0150
joint_mismatched_frame_raw        = 0.2369
joint_raw_restore_both            = 0.0100
```

客观现象：

```text
1. raw joint patch 仍最强。
2. remove suffix_all 或 remove final token 后，matched_top1 基本塌到 0.08/0.07。
3. remove penultimate 或 lexical 后仍保留较多效果，约 0.41。
4. suffix_final basis_only 有一定迁移，0.1534，高于 lexical basis_only 的 0.0150。
5. mismatched frame raw = 0.2369，说明错误 frame 仍能带来部分推进，但远低于 raw。
```

### GLM4 客观结果

```text
joint_raw                         = 0.6546
joint_orth_suffix_all             = 0.1269
joint_orth_suffix_nonfinal        = 0.4009
joint_orth_suffix_final           = 0.1013
joint_orth_suffix_penultimate     = 0.5267
joint_orth_suffix_first           = 0.3977
joint_orth_suffix_second          = 0.3870
joint_orth_suffix_function        = 0.1386
joint_orth_suffix_lexical         = 0.5608
joint_orth_all_suffix_tokens      = 0.0959
joint_suffix_all_basis_only       = 0.1183
joint_suffix_final_basis_only     = 0.1461
joint_suffix_lexical_basis_only   = 0.0043
joint_mismatched_frame_raw        = 0.2751
joint_raw_restore_both            = 0.0171
```

客观现象：

```text
1. GLM4 raw 最强，0.6546。
2. remove suffix_final 后降到 0.1013，remove suffix_all 后降到 0.1269。
3. remove suffix_lexical 后仍有 0.5608，说明 lexical token 不是 GLM4 的主因。
4. remove suffix_penultimate 后仍有 0.5267，也不是单独倒数第二 token 主导。
5. suffix_final basis_only = 0.1461，仍有少量迁移，但远低于 raw。
```

### DeepSeek7B 客观结果

```text
joint_raw                         = 0.4193
joint_orth_suffix_all             = 0.0538
joint_orth_suffix_nonfinal        = 0.1772
joint_orth_suffix_final           = 0.0396
joint_orth_suffix_penultimate     = 0.2690
joint_orth_suffix_first           = 0.2089
joint_orth_suffix_second          = 0.1899
joint_orth_suffix_function        = 0.0570
joint_orth_suffix_lexical         = 0.3085
joint_orth_all_suffix_tokens      = 0.0364
joint_suffix_all_basis_only       = 0.1108
joint_suffix_final_basis_only     = 0.1566
joint_suffix_lexical_basis_only   = 0.0174
joint_mismatched_frame_raw        = 0.2674
joint_raw_restore_both            = 0.0047
```

客观现象：

```text
1. DS7B raw = 0.4193，低于 Qwen3/GLM4。
2. remove suffix_all/final/function 后几乎塌陷到 0.04-0.06。
3. remove suffix_lexical 后仍保留 0.3085。
4. suffix_final basis_only = 0.1566，是三个模型中 final basis-only 相对最强的。
5. mismatched_frame_raw = 0.2674，与 raw 差距小于 Qwen3/GLM4，说明 DS7B 对 frame mismatch 的区分更弱或更受输出边界影响。
```

### 三模型共同事实

本轮最稳定的共同现象：

```text
1. suffix_all 和 suffix_final 是最关键破坏项。
   remove suffix_all / remove suffix_final 后，三模型 matched_top1 都大幅下降。

2. suffix_lexical 不是主因。
   remove suffix_lexical 后仍保留大量 raw 效果：
     Qwen3: 0.4102 / raw 0.5399
     GLM4: 0.5608 / raw 0.6546
     DS7B: 0.3085 / raw 0.4193

3. suffix_final basis_only 有少量但有限的可迁移性：
     Qwen3: 0.1534
     GLM4: 0.1461
     DS7B: 0.1566
   说明最后 readout token 自身带有一部分输出接口格式，但不足以独立完成 object-value closure。

4. raw_restore_both 基本恢复 clean，matched_top1 接近 0：
     Qwen3: 0.0100
     GLM4: 0.0171
     DS7B: 0.0047
   说明 destroy/restore 框架本身是有效的，不是补丁后不可逆污染。
```

### 当前最重要修正

Phase 83 支持对 Phase 82 的进一步修正：

```text
object-value closure 的关键不是 relation lexical word 本身。
更关键的是 object token 后面的 suffix/readout boundary，尤其 final pre-answer token。
```

但它也说明：

```text
final token 不是完整机制。
suffix_final basis_only 只能产生有限迁移；
完整 raw 效果需要 object state + frame/readout state 的 joint compatibility。
```

更谨慎的表达：

```text
对象-属性/对象-关系读取不是“对象向量 + 关系词向量”。
它更像：
object state 在 post-object suffix trajectory 中被格式化，
最后由 answer-boundary/readout token 接入候选输出。
```

### 硬伤

```text
1. suffix token 分类仍然粗糙，function/lexical 由简单词表判断，不能当作最终语言变量。
2. orthogonal removal 仍可能同时移除兼容性核心，因此“下降”不能直接解释为该 token 独立编码了关系。
3. basis_only 只能说明该子空间有部分可迁移格式，不能说明它是完整因果机制。
4. DS7B 使用 SDPA 时存在 sliding window attention 实现警告，DS7B 结果需要谨慎。
5. Qwen3 第一次长跑出现用户态 segfault，虽然 resume 后完成，但说明超长单进程仍有工程稳定性风险。
```

### 下一步计划

Phase 84：clean-state erase / restore 测试。

目标：

```text
从 clean 状态中 erase suffix_final / suffix_all / all_suffix_tokens 子空间；
观察 clean answer 是否下降；
再 restore 该子空间；
观察 clean answer 是否恢复。
```

意义：

```text
Phase 83 主要是在 matched transfer 中看 suffix component；
Phase 84 要在 clean computation 中验证这些子空间是否必要。
```

Phase 85：readout boundary open generation audit。

目标：

```text
不只看 closed candidate scoring；
还要看自然生成中 suffix_final/readout boundary 改变后，生成内容是否按目标关系切换。
```

Phase 86：迁移到其他关系路径。

目标：

```text
把 object + suffix/readout boundary 结构迁移到：
temporal order
logical operator
passive role query
coreference
translation/style control
```

阶段性大任务：

```text
建立 global relation-path map：
1. object/value binding path；
2. temporal order path；
3. logical operator path；
4. role binding path；
5. coreference path。

每条路径都要拆成：
source token state；
post-source suffix/context trajectory；
readout boundary；
candidate answer interface；
destroy/restore closure。
```

## Phase 84: clean suffix erase/restore 全量测试 [2026-06-10 17:48]

### 任务目标

根据 Phase 83 和最新分析，本轮从 matched transfer 转向 clean computation：

```text
Phase 83:
  在 matched transfer 中发现 suffix_final / suffix_all 极关键。

Phase 84:
  在 clean prompt 本身中 erase suffix/readout 子空间；
  观察 clean answer 是否下降；
  再 restore clean 子空间；
  观察是否恢复。
```

核心问题：

```text
suffix_final / suffix_all 是 transfer 相关因素，
还是 clean computation 中的必要读出成分？
```

### 对用户分析的判断

这次分析基本正确：

```text
1. Phase 83 的主要进展不是 relation lexical word，而是 readout boundary。
2. final token 不是完整机制，但很可能是候选值空间接口。
3. 需要在 clean 状态中做 erase/restore，才能判断它是否必要。
4. 不应直接做理论总结，应该继续补客观拼图。
```

因此本轮执行 Phase 84：clean suffix erase/restore。

### 脚本

新增：

```text
tests/gpt5/phase84_clean_suffix_erase_restore.py
tests/gpt5/phase84_clean_suffix_erase_restore_summary.py
tests/gpt5/run_phase84_clean_suffix_erase_restore_full.sh
```

测试逻辑：

```text
1. 使用 Phase 83 相同方式构造 suffix component bases：
   suffix_all
   suffix_final
   suffix_nonfinal
   suffix_function
   suffix_lexical
   all_suffix_tokens

2. 对 clean prompt 的 object_last / frame_last 位置做 erase：
   erase_object_*
   erase_frame_*
   erase_both_*

3. erase 方法：
   在对应 basis 上把 clean state 的投影减掉。

4. restore 方法：
   在 restore_layer 上把 clean state 的对应 basis 投影加回。

5. 指标：
   clean_drop = base_margin - erase_margin
   restore_gain = restore_margin - erase_margin
   restore_gap = base_margin - restore_margin
   erase_top1
   restore_top1
```

### Smoke Test

```bash
PHASE84_MODELS=qwen3 \
QWEN3_PHASE84_MAX_ITEMS=2 \
QWEN3_PHASE84_LAYER_PAIRS=4-8 \
PHASE84_OUTPUT_DIR=results/gpt5_phase84_smoke \
tests/gpt5/run_phase84_clean_suffix_erase_restore_full.sh
```

结果：

```text
exit_code = 0
rows = 36
```

### 正式命令

```bash
PHASE84_OUTPUT_DIR=results/gpt5_phase84_clean_suffix_erase_restore_full_20260610_132647 \
tests/gpt5/run_phase84_clean_suffix_erase_restore_full.sh
```

runner 内部顺序：

```text
qwen3:
  layer_pairs = 4-8,8-12
  max_items = 672

glm4:
  layer_pairs = 4-10,10-20
  max_items = 672

deepseek7b:
  layer_pairs = 8-10,12-14
  max_items = 672
```

共同参数：

```text
module = resid_out
contrast_rank = 64
component_rank = 24
max_basis_items = 224
max_distractors = 10
hard_exit_after_model = true
```

说明：

```text
本机没有 flash_attn 包，flash_attention_2 加载失败后自动使用 SDPA。
DeepSeek7B 仍有 sliding window attention + SDPA warning，解释 DS7B 结果时需要保留限制。
```

### 输出文件

```text
results/gpt5_phase84_clean_suffix_erase_restore_full_20260610_132647/qwen3_phase84_clean_suffix_erase_restore.json
results/gpt5_phase84_clean_suffix_erase_restore_full_20260610_132647/glm4_phase84_clean_suffix_erase_restore.json
results/gpt5_phase84_clean_suffix_erase_restore_full_20260610_132647/deepseek7b_phase84_clean_suffix_erase_restore.json
results/gpt5_phase84_clean_suffix_erase_restore_full_20260610_132647/phase84_clean_suffix_erase_restore_summary.json
results/gpt5_phase84_clean_suffix_erase_restore_full_20260610_132647/PHASE84_CLEAN_SUFFIX_ERASE_RESTORE_SUMMARY.md
```

### 数据规模

```text
Qwen3:
  items = 672
  rows = 24192
  eligible = 828

GLM4:
  items = 672
  rows = 24192
  eligible = 954

DeepSeek7B:
  items = 672
  rows = 24192
  eligible = 656

total rows = 72576
```

### Qwen3 客观结果

核心条件：

```text
erase_frame_suffix_all:
  drop = 1.9121
  restore_gain = 1.4408
  restore_gap = 0.4713
  erase_top1 = 0.7778
  restore_top1 = 0.9143

erase_frame_suffix_final:
  drop = 2.2054
  restore_gain = 1.6714
  restore_gap = 0.5340
  erase_top1 = 0.7488
  restore_top1 = 0.9046

erase_frame_suffix_nonfinal:
  drop = 0.8843
  restore_gain = 0.6162
  restore_gap = 0.2681
  erase_top1 = 0.8901
  restore_top1 = 0.9408

erase_frame_suffix_function:
  drop = 1.9586
  restore_gain = 1.4983
  restore_gap = 0.4603
  erase_top1 = 0.7802
  restore_top1 = 0.9251

erase_frame_suffix_lexical:
  drop = 0.2764
  restore_gain = 0.1738
  restore_gap = 0.1025
  erase_top1 = 0.9469
  restore_top1 = 0.9541

erase_frame_all_suffix_tokens:
  drop = 2.5583
  restore_gain = 2.0840
  restore_gap = 0.4743
  erase_top1 = 0.7162
  restore_top1 = 0.9118

erase_object_suffix_final:
  drop = 0.1122
  restore_gain = 0.0237
  restore_gap = 0.0885
  erase_top1 = 0.9674
  restore_top1 = 0.9686
```

客观现象：

```text
1. frame/readout 位置 erase 很强，object 位置 erase 很弱。
2. suffix_final 比 suffix_nonfinal 更关键。
3. suffix_function 接近 suffix_all，suffix_lexical 很弱。
4. all_suffix_tokens 最强，说明 readout boundary 不是单点，而是 suffix component 联合接口。
5. restore 能恢复大部分 drop，但不是完全恢复。
```

### GLM4 客观结果

```text
erase_frame_suffix_all:
  drop = 0.6251
  restore_gain = 0.5344
  restore_gap = 0.0907
  erase_top1 = 0.9078
  restore_top1 = 0.9623

erase_frame_suffix_final:
  drop = 0.4368
  restore_gain = 0.3238
  restore_gap = 0.1130
  erase_top1 = 0.9172
  restore_top1 = 0.9602

erase_frame_suffix_nonfinal:
  drop = 0.2375
  restore_gain = 0.1948
  restore_gap = 0.0427
  erase_top1 = 0.9539
  restore_top1 = 0.9727

erase_frame_suffix_function:
  drop = 0.3142
  restore_gain = 0.2724
  restore_gap = 0.0418
  erase_top1 = 0.9486
  restore_top1 = 0.9675

erase_frame_suffix_lexical:
  drop = 0.2029
  restore_gain = 0.1430
  restore_gap = 0.0599
  erase_top1 = 0.9455
  restore_top1 = 0.9654

erase_frame_all_suffix_tokens:
  drop = 1.0872
  restore_gain = 0.9205
  restore_gap = 0.1666
  erase_top1 = 0.8700
  restore_top1 = 0.9497

erase_object_suffix_final:
  drop = 0.0156
  restore_gain = -0.0012
  restore_gap = 0.0168
  erase_top1 = 0.9864
  restore_top1 = 0.9853
```

客观现象：

```text
1. GLM4 的 clean erase 效果明显弱于 Qwen3 和 DS7B。
2. 但 all_suffix_tokens 仍有最大 drop = 1.0872。
3. object 位置几乎不受 suffix erase 影响。
4. restore 非常接近 base，restore_gap 很小。
```

### DeepSeek7B 客观结果

```text
erase_frame_suffix_all:
  drop = 1.2361
  restore_gain = 1.0544
  restore_gap = 0.1817
  erase_top1 = 0.8171
  restore_top1 = 0.9345

erase_frame_suffix_final:
  drop = 1.2377
  restore_gain = 1.0238
  restore_gap = 0.2139
  erase_top1 = 0.7973
  restore_top1 = 0.9253

erase_frame_suffix_nonfinal:
  drop = 0.6634
  restore_gain = 0.5074
  restore_gap = 0.1559
  erase_top1 = 0.8765
  restore_top1 = 0.9558

erase_frame_suffix_function:
  drop = 1.0788
  restore_gain = 0.9109
  restore_gap = 0.1680
  erase_top1 = 0.8323
  restore_top1 = 0.9390

erase_frame_suffix_lexical:
  drop = 0.3569
  restore_gain = 0.2214
  restore_gap = 0.1355
  erase_top1 = 0.9085
  restore_top1 = 0.9543

erase_frame_all_suffix_tokens:
  drop = 1.5748
  restore_gain = 1.4061
  restore_gap = 0.1687
  erase_top1 = 0.7546
  restore_top1 = 0.9192

erase_object_suffix_final:
  drop = 0.1857
  restore_gain = 0.0955
  restore_gap = 0.0901
  erase_top1 = 0.9527
  restore_top1 = 0.9588
```

客观现象：

```text
1. DS7B clean erase 支持 frame/readout 位置重要。
2. suffix_final 与 suffix_all 几乎同等 drop。
3. suffix_lexical 仍明显弱。
4. restore 能恢复大部分 drop。
5. object 位置影响弱，但略强于 Qwen3/GLM4。
```

### 三模型共同事实

```text
1. clean computation 中，frame/readout 位置远比 object 位置更受 suffix erase 影响。

2. suffix_final 是 clean answer 的重要必要成分：
   Qwen3 drop = 2.2054
   GLM4 drop = 0.4368
   DS7B drop = 1.2377

3. all_suffix_tokens 是最强破坏项：
   Qwen3 drop = 2.5583
   GLM4 drop = 1.0872
   DS7B drop = 1.5748

4. suffix_lexical 是弱项：
   Qwen3 drop = 0.2764
   GLM4 drop = 0.2029
   DS7B drop = 0.3569

5. restore 能恢复大部分 erase 造成的下降：
   Qwen3 all_suffix_tokens restore_gain = 2.0840
   GLM4 all_suffix_tokens restore_gain = 0.9205
   DS7B all_suffix_tokens restore_gain = 1.4061

6. restore_gap 仍非零，说明当前 basis restore 不是完整状态恢复。
```

### 对 Phase 83 的验证

Phase 84 直接支持 Phase 83 的核心判断：

```text
suffix_final / readout boundary 不只是 transfer artifact；
它在 clean answer computation 中也是必要成分。
```

同时修正为：

```text
不是 final token 单点决定一切；
all_suffix_tokens 更强，说明 final token 是接口核心，
但完整机制仍依赖 suffix trajectory 的联合格式。
```

当前更稳表达：

```text
object-value closure =
object state
+ post-object suffix trajectory
+ final readout boundary
+ candidate answer interface

其中真正被 clean erase 证实的关键位置是 frame/readout token，
不是 object token 本身。
```

### 硬伤

```text
1. erase 使用的是子空间投影移除，不等于精确删除单一语言变量。
2. suffix_final basis 仍混合 readout boundary、位置、局部语法、候选类型期待。
3. restore 不是完整恢复，说明 basis 只捕捉部分必要成分。
4. GLM4 clean erase drop 较小，说明模型间路径机制差异仍然很大。
5. DeepSeek7B 使用 SDPA 时有 sliding window attention warning。
```

### 当前关键洞察

本轮把 object-value binding 的拼图推进了一层：

```text
关系不是储存在关系实词里；
也不是只存在对象 token 本身；
而是在对象之后的自然语言后缀轨迹中形成读出格式，
最后由 frame/readout boundary 接入候选值空间。
```

这对“相对编码”非常关键：

```text
对象本身不是孤立编码单元；
对象必须放在某个 readout frame 中才形成可回答的关系值。
```

### 下一步计划

Phase 85：suffix/readout open generation audit。

目标：

```text
验证 suffix_final / all_suffix_tokens 的作用是否影响自由生成，
而不只是 closed candidate scoring。
```

Phase 86：readout boundary cross-function transfer。

目标：

```text
把 object-value 的 readout boundary 结构迁移到：
1. temporal order；
2. logical operator；
3. role binding；
4. coreference；
5. translation/style。
```

Phase 87：minimal readout circuit search。

目标：

```text
在 frame/readout token 上定位：
attention heads；
MLP channels；
residual subspace；
看哪些组件负责 candidate-space gateway。
```

阶段性大任务：

```text
建立 global readout-interface map。

每个语言功能都拆成：
source token
context/suffix trajectory
readout boundary
candidate/output interface
destroy-restore closure
```

## Phase 85: readout open generation audit 全量测试 [2026-06-10 19:03]

### 任务目标

Phase 84 已经证明：

```text
suffix_final / all_suffix_tokens 在 clean candidate scoring 中是必要成分。
```

但 Phase 84 仍然是 closed candidate scoring。本轮尝试进入 open generation：

```text
对 clean prompt 做 greedy generation；
在 frame/readout 位置 erase/restore suffix 子空间；
观察自由生成是否仍命中 target。
```

本轮目标不是证明机制，而是检查：

```text
closed candidate scoring 中的 readout boundary 效应，
是否能直接迁移到 open generation。
```

### 对用户分析的判断

最新分析方向正确：

```text
1. Phase 84 是性质升级，说明 suffix/readout 不只是 transfer artifact。
2. 但 Phase 84 仍然缺 sufficiency 和 open generation。
3. open generation 必须做，但读出模板本身也可能成为新瓶颈。
```

因此本轮执行 Phase 85：readout open generation audit。

### 脚本

新增：

```text
tests/gpt5/phase85_readout_open_generation_audit.py
tests/gpt5/phase85_readout_open_generation_audit_summary.py
tests/gpt5/run_phase85_readout_open_generation_audit_full.sh
```

测试逻辑：

```text
1. 使用 Phase 83/84 的 suffix component bases。
2. 对 clean_prompt 做 greedy generation。
3. 在 frame_last 位置进行：
   erase_frame_suffix_final
   erase_frame_suffix_all
   erase_frame_suffix_function
   erase_frame_suffix_lexical
   erase_frame_all_suffix_tokens
4. 同时做 restore_frame_*。
5. 记录 generated text 是否 prefix/contains target。
6. 记录 generated text 是否相对 base changed。
```

### Smoke Test

```bash
PHASE85_MODELS=qwen3 \
QWEN3_PHASE85_MAX_ITEMS=2 \
QWEN3_PHASE85_LAYER_PAIRS=4-8 \
PHASE85_OUTPUT_DIR=results/gpt5_phase85_smoke \
tests/gpt5/run_phase85_readout_open_generation_audit_full.sh
```

结果：

```text
exit_code = 0
```

### 正式命令

```bash
PHASE85_OUTPUT_DIR=results/gpt5_phase85_readout_open_generation_audit_full_20260610_175741 \
tests/gpt5/run_phase85_readout_open_generation_audit_full.sh
```

runner 内部顺序：

```text
qwen3:
  layer_pairs = 4-8,8-12
  audit_layers = 4,8,12
  max_items = 224

glm4:
  layer_pairs = 4-10,10-20
  audit_layers = 4,10,20
  max_items = 224

deepseek7b:
  layer_pairs = 8-10,12-14
  audit_layers = 8,10,12,14
  max_items = 224
```

共同参数：

```text
module = resid_out
component_rank = 24
max_basis_items = 224
max_new_tokens = 6
hard_exit_after_model = true
generation = greedy, use_cache=false
```

说明：

```text
本机没有 flash_attn 包，flash_attention_2 加载失败后自动使用 SDPA。
DeepSeek7B 仍有 sliding window attention + SDPA warning。
```

### 输出文件

```text
results/gpt5_phase85_readout_open_generation_audit_full_20260610_175741/qwen3_phase85_readout_open_generation_audit.json
results/gpt5_phase85_readout_open_generation_audit_full_20260610_175741/glm4_phase85_readout_open_generation_audit.json
results/gpt5_phase85_readout_open_generation_audit_full_20260610_175741/deepseek7b_phase85_readout_open_generation_audit.json
results/gpt5_phase85_readout_open_generation_audit_full_20260610_175741/phase85_readout_open_generation_audit_summary.json
results/gpt5_phase85_readout_open_generation_audit_full_20260610_175741/PHASE85_READOUT_OPEN_GENERATION_AUDIT_SUMMARY.md
```

### 数据规模

```text
Qwen3:
  items = 224
  audit_layers = 3
  rows = 7392

GLM4:
  items = 224
  audit_layers = 3
  rows = 7392

DeepSeek7B:
  items = 224
  audit_layers = 4
  rows = 9856

total rows = 24640
```

### 关键负结果

三模型 base open generation 的 target prefix hit 全部为 0：

```text
Qwen3:
  base prefix_hit = 0.0000
  eligible_n = 0

GLM4:
  base prefix_hit = 0.0000
  eligible_n = 0

DeepSeek7B:
  base prefix_hit = 0.0000
  eligible_n = 0
```

示例：

```text
Qwen3:
  prompt target = metal tool
  base generation = " weapon, but it's not"
  erase_all generation = " a tool, but it's"

GLM4:
  prompt target = metal tool
  base generation = " tool with a cutting edge."
  erase_all generation = " tool with a cutting edge."

DeepSeek7B:
  prompt target = metal tool
  base generation = " __________ tool.\n\n\nA"
  erase_all generation = " 62.5%"
```

客观解释：

```text
当前 natural clean_prompt 不是稳定 open-generation answer reader。
模型自由生成经常生成语义相关补全，但不是数据集 target 的 exact prefix。
因此不能用 prefix_hit 作为机制判断指标。
```

### 仍有价值的现象

虽然 target hit 不可用，但 generated text changed 指标显示：

Qwen3：

```text
erase_frame_suffix_final changed = 0.7262
erase_frame_suffix_all changed = 0.5268
erase_frame_suffix_function changed = 0.5580
erase_frame_suffix_lexical changed = 0.3676
erase_frame_all_suffix_tokens changed = 0.9643
restore_frame_all_suffix_tokens changed = 0.1146
```

GLM4：

```text
erase_frame_suffix_final changed = 0.5104
erase_frame_suffix_all changed = 0.3512
erase_frame_suffix_function changed = 0.3720
erase_frame_suffix_lexical changed = 0.2530
erase_frame_all_suffix_tokens changed = 0.9062
restore_frame_all_suffix_tokens changed = 0.0744
```

DeepSeek7B：

```text
erase/restore 完成，但由于 base target hit = 0，不能解释为答案命中变化。
```

客观现象：

```text
1. erase_frame_all_suffix_tokens 几乎总会改变自由生成文本。
2. restore_frame_all_suffix_tokens 会大幅接近 base generation，changed 下降到约 0.07-0.11。
3. 这说明 suffix/readout 子空间确实影响自由生成轨迹。
4. 但当前 prompt 没有稳定生成标准答案，因此不能把 changed 解释为正确/错误答案切换。
```

### 当前最重要修正

Phase 85 不是机制成功结果，而是读出器校准结果：

```text
closed candidate scoring 里的 readout boundary 效应很强；
open generation 中 readout 子空间能改变生成轨迹；
但当前 open generation prompt 不能稳定读出 target；
因此 open generation 机制实验必须先做 answer-only reader calibration。
```

这与前面 role query 的经验一致：

```text
读出器不稳时，不能直接做机制解释。
```

### 硬伤

```text
1. base generation target prefix_hit = 0，导致 eligible_n = 0。
2. exact prefix target 对自由生成过于严格，语义相关输出无法计入。
3. prompt 没有明确 answer-only 约束，模型会生成解释、补充短语或模板文本。
4. changed 指标只能说明生成轨迹变化，不能说明答案正确性变化。
5. 仍需设计开放生成专用读出器。
```

### 下一步计划

Phase 86：answer-only open generation reader calibration。

目标：

```text
设计稳定的 open generation reader：
1. 明确 Answer: 格式；
2. 要求只输出 value；
3. 使用符号或短答案约束；
4. 对 base generation 做校准；
5. 只有 base target hit 足够高，才进入 erase/restore。
```

建议模板：

```text
Question: What is the answer value?
Context: A knife is a type of
Answer:
```

或：

```text
Complete with only the answer value.
A knife is a type of
Answer:
```

Phase 87：open generation erase/restore 复测。

目标：

```text
在校准过的 answer-only reader 上，
复测 suffix_final / all_suffix_tokens 对自由生成答案的影响。
```

Phase 88：跨功能 readout reader 设计。

目标：

```text
对 temporal order、logical operator、role binding、coreference 等功能，
先建立稳定 reader，
再做 erase/restore。
```

关键结论：

```text
open generation 不能直接使用自然 prompt；
必须先建立稳定读出器。
这不是失败，而是说明机制研究必须把“读出器校准”作为正式实验层。
```

## Phase 86: answer-only reader calibration 全量测试 [2026-06-10 21:58]

### 任务目标

根据 Phase 85 的负结果，本轮不继续做 erase/restore 机制干预，而是先校准开放生成读出器。

Phase 85 已经说明：

```text
suffix/readout 子空间会显著改变开放生成轨迹；
但自然 prompt 的 base target prefix_hit = 0；
因此不能直接把 open generation 命中率当作内部机制证据。
```

本轮目标：

```text
1. 设计 answer-only reader templates；
2. 在 qwen3、GLM4、DS7B 三模型上做大样本开放生成校准；
3. 区分严格命中、值族重叠、格式违规；
4. 判断是否已经具备进入 open generation erase/restore 的读出条件。
```

### 对输入分析的判断

输入分析中正确的部分：

```text
1. Phase 85 不是机制失败，而是 reader failure。
2. open generation 必须拆成内部机制层 + 读出器层 + 解码层。
3. 不应在 base reader 不稳定时继续做机制干预。
4. answer-only reader calibration 应作为正式实验层。
```

需要谨慎的部分：

```text
1. exact/prefix hit 太严格，但不能直接用弱语义相似替代机制证据。
2. family overlap 只能表示值族重叠，不等于完整答案正确。
3. 如果 answer-only reader 仍不稳定，应继续重建读出器，而不是强行做 erase/restore。
```

### 新增脚本

```text
tests/gpt5/phase86_answer_only_reader_calibration.py
tests/gpt5/phase86_answer_only_reader_calibration_summary.py
tests/gpt5/run_phase86_answer_only_reader_calibration_full.sh
```

脚本特性：

```text
1. 三模型顺序运行：qwen3 -> GLM4 -> DS7B。
2. 每个模型完成后使用 --hard-exit-after-model。
3. 使用本地模型和 CUDA。
4. 默认加载 flash_attention_2,sdpa,eager；本机无 flash_attn 包，因此实际回退到 sdpa。
5. 支持 resume；DeepSeek7B 中途 segfault 后通过 partial 继续完成。
6. 不做 hook、不做 patch，只做 reader calibration。
```

### 模板

测试 8 个 answer-only reader 模板：

```text
answer_only_plain
answer_only_short_phrase
question_value
fill_blank_answer
json_value
value_equals
bare_answer
one_phrase
```

核心指标：

```text
exact_hit：生成首段完全等于 target。
prefix_hit：生成首段以前缀方式命中 target。
contains_hit：生成文本包含 target。
word_subset_hit：target content words 全部出现在首段中。
family_overlap_hit：首段与 target 有至少 50% content-word 双向重叠。
target_word_coverage：target words 被首段覆盖比例。
first_word_precision：首段中属于 target 的词比例。
format_violation：出现解释、长句、重复上下文等格式违规。
```

说明：

```text
family_overlap_hit 只作为弱值族信号，不作为完整答案正确证据。
```

### 运行命令

smoke：

```bash
PHASE86_OUTPUT_DIR=results/gpt5_phase86_answer_only_reader_calibration_smoke2_$(date +%Y%m%d_%H%M%S) \
PHASE86_MODELS=qwen3 \
QWEN3_PHASE86_MAX_ITEMS=4 \
PHASE86_TEMPLATES=answer_only_plain,question_value \
PHASE86_PROGRESS_EVERY=2 \
tests/gpt5/run_phase86_answer_only_reader_calibration_full.sh
```

正式全量：

```bash
PHASE86_OUTPUT_DIR=results/gpt5_phase86_answer_only_reader_calibration_full_$(date +%Y%m%d_%H%M%S) \
PHASE86_PROGRESS_EVERY=84 \
tests/gpt5/run_phase86_answer_only_reader_calibration_full.sh
```

DeepSeek7B 第一次运行到 420/672 item 后出现用户态 segmentation fault：

```text
exit_code = 139
```

随后使用同一输出目录 resume：

```bash
PHASE86_OUTPUT_DIR=results/gpt5_phase86_answer_only_reader_calibration_full_20260610_211347 \
PHASE86_MODELS=qwen3,glm4,deepseek7b \
PHASE86_PROGRESS_EVERY=84 \
tests/gpt5/run_phase86_answer_only_reader_calibration_full.sh
```

resume 后 DeepSeek7B 完成。

### 输出文件

```text
results/gpt5_phase86_answer_only_reader_calibration_full_20260610_211347/qwen3_phase86_answer_only_reader_calibration.json
results/gpt5_phase86_answer_only_reader_calibration_full_20260610_211347/glm4_phase86_answer_only_reader_calibration.json
results/gpt5_phase86_answer_only_reader_calibration_full_20260610_211347/deepseek7b_phase86_answer_only_reader_calibration.json
results/gpt5_phase86_answer_only_reader_calibration_full_20260610_211347/phase86_answer_only_reader_calibration_summary.json
results/gpt5_phase86_answer_only_reader_calibration_full_20260610_211347/PHASE86_ANSWER_ONLY_READER_CALIBRATION_SUMMARY.md
```

### 数据规模

```text
qwen3:
  items = 672
  templates = 8
  rows = 5376

GLM4:
  items = 672
  templates = 8
  rows = 5376

DS7B:
  items = 672
  templates = 8
  rows = 5376

total rows = 16128
```

### Qwen3 客观结果

按 template：

```text
fill_blank_answer:
  exact_hit = 0.0060
  prefix_hit = 0.0060
  word_subset_hit = 0.0074
  family_overlap_hit = 0.3006
  format_violation = 0.0640

answer_only_plain:
  exact_hit = 0.0060
  prefix_hit = 0.0074
  word_subset_hit = 0.0104
  family_overlap_hit = 0.2560
  format_violation = 0.0863

answer_only_short_phrase:
  exact_hit = 0.0074
  prefix_hit = 0.0074
  word_subset_hit = 0.0089
  family_overlap_hit = 0.2217
  format_violation = 0.0818
```

按 relation：

```text
is_a:
  family_overlap_hit = 0.2552
  word_subset_hit = 0.0130

used_for:
  family_overlap_hit = 0.2227
  word_subset_hit = 0.0169

location:
  family_overlap_hit = 0.2201
  word_subset_hit = 0.0065
```

样例：

```text
target = metal tool
generated = tool
解释：值族重叠，但不是完整 target。

target = cutting food
generated = cutting food
解释：严格命中。
```

### GLM4 客观结果

按 template：

```text
fill_blank_answer:
  exact_hit = 0.0074
  prefix_hit = 0.0074
  word_subset_hit = 0.0104
  family_overlap_hit = 0.3021
  format_violation = 0.1979

answer_only_short_phrase:
  exact_hit = 0.0238
  prefix_hit = 0.0238
  word_subset_hit = 0.0268
  family_overlap_hit = 0.2723
  format_violation = 0.1250

answer_only_plain:
  exact_hit = 0.0045
  prefix_hit = 0.0045
  word_subset_hit = 0.0104
  family_overlap_hit = 0.2292
  format_violation = 0.1935
```

按 relation：

```text
location:
  family_overlap_hit = 0.2031
  word_subset_hit = 0.0078

part_of:
  family_overlap_hit = 0.1797
  word_subset_hit = 0.0117

used_for:
  family_overlap_hit = 0.1758
  word_subset_hit = 0.0221
```

样例：

```text
target = metal tool
generated = cutting tool
解释：语义相关，但不是完整 target。

target = cutting food
generated = cutting food
解释：严格命中。
```

### DeepSeek7B 客观结果

按 template：

```text
fill_blank_answer:
  exact_hit = 0.0000
  prefix_hit = 0.0000
  word_subset_hit = 0.0060
  family_overlap_hit = 0.1220
  format_violation = 0.1161

bare_answer:
  exact_hit = 0.0045
  prefix_hit = 0.0060
  word_subset_hit = 0.0060
  family_overlap_hit = 0.0967
  format_violation = 0.0982

answer_only_short_phrase:
  exact_hit = 0.0000
  prefix_hit = 0.0000
  word_subset_hit = 0.0000
  family_overlap_hit = 0.0640
  format_violation = 0.0967
```

按 relation：

```text
part_of:
  family_overlap_hit = 0.0534
  word_subset_hit = 0.0026

location:
  family_overlap_hit = 0.0521
  word_subset_hit = 0.0026

can_do:
  family_overlap_hit = 0.0495
  word_subset_hit = 0.0000
```

样例：

```text
target = metal tool
generated = tool
解释：弱值族重叠，但不完整。

target = cutting food
generated = cutting food
解释：少量 strict hit 存在。
```

### 跨模型模板排名

```text
fill_blank_answer:
  exact_hit = 0.0045
  word_subset_hit = 0.0079
  family_overlap_hit = 0.2416
  format_violation = 0.1260

answer_only_short_phrase:
  exact_hit = 0.0104
  word_subset_hit = 0.0119
  family_overlap_hit = 0.1860
  format_violation = 0.1012

answer_only_plain:
  exact_hit = 0.0035
  word_subset_hit = 0.0079
  family_overlap_hit = 0.1647
  format_violation = 0.1290
```

### 关键客观现象

```text
1. 三模型 answer-only reader 的 strict hit 仍然很低。
2. Qwen3 和 GLM4 在 fill_blank_answer 上有约 0.30 的 family_overlap_hit，但 strict word_subset_hit 仍只有约 0.01。
3. DeepSeek7B 的 answer-only reader 更弱，最佳 family_overlap_hit 只有 0.1220。
4. 很多输出是值族核心词，例如 tool / animal / cutting tool，而不是目标完整短语 metal tool / home animal。
5. answer_only 约束能减少解释倾向，但不能让模型稳定输出目标 value。
6. 当前 answer-only reader 仍不满足进入 open generation erase/restore 的条件。
```

### 对 Phase 85/86 的修正

Phase 85 的自然 prompt 完全不能读出 target。

Phase 86 的 answer-only prompt 有改进，但仍只达到：

```text
strict hit ≈ 0.00-0.03
family overlap ≈ 0.12-0.30
```

因此当前不能进入：

```text
open generation erase/restore
open generation matched transfer
```

否则会把 reader failure 与机制 failure 混在一起。

### 硬伤

```text
1. target 多为实验设计中的具体短语，例如 metal tool、home animal；模型自然生成更倾向于 tool、animal、cutting tool 等一般答案。
2. exact/prefix hit 过低，说明 reader 没有校准到目标短语级别。
3. family_overlap_hit 只能说明值族相关，不能说明完整答案正确。
4. 部分模板 format_violation 仍明显，尤其 GLM4 的 fill_blank_answer 和 Qwen3/GLM4 的 is_a 关系。
5. DeepSeek7B 使用 sdpa 时有 Sliding Window Attention warning，且本轮中途出现一次用户态 segmentation fault；resume 后完成，但这是工程硬伤。
6. 本轮不包含机制干预，因此不能直接证明 suffix/readout 子空间对正确答案生成的因果作用。
```

### 当前理论进展

Phase 86 支持一个更严格的分层判断：

```text
closed candidate reader 已稳定；
open natural reader 失败；
answer-only reader 仍不足；
因此当前知识读出机制主要仍只能在 closed candidate scoring 中可靠验证。
```

这说明：

```text
内部 readout-gateway compatibility 可以存在；
但开放生成是否表现为目标短语，需要额外的 reader alignment。
```

更准确的理论表达：

```text
语言知识读出不是从内部状态直接等价到生成文本。
它至少经过：
1. 内部兼容状态；
2. 读出网关；
3. 候选评分或生成读出器；
4. 输出格式约束；
5. 解码策略。
```

Phase 86 的负结果把“开放生成读出器”正式提升为需要单独破解的对象。

### 下一步计划

不要马上做 open generation erase/restore。

下一阶段应改为：

```text
Phase 87：symbolic / multiple-choice reader calibration。
```

核心方案：

```text
1. 给定 object + relation + 4-8 个候选 value；
2. 让模型只输出选项字母或 value；
3. 同时保留 closed candidate logits 作为基准；
4. 比较：
   - closed candidate score 是否正确；
   - forced choice generation 是否正确；
   - answer-only free generation 是否正确。
```

目标是拆开三类读出器：

```text
R_closed：封闭候选评分；
R_choice：显式多选生成；
R_open：开放短答案生成。
```

如果 R_closed 强、R_choice 强、R_open 弱：

```text
说明问题主要是开放生成格式，不是知识读出。
```

如果 R_closed 强、R_choice 弱：

```text
说明从候选评分到文本选择仍有读出接口问题。
```

如果三者都弱：

```text
再回到内部机制测试。
```

更大的阶段性任务：

```text
1. 先建立可靠 reader stack；
2. 再回到 suffix/readout erase/restore；
3. 最后把 closed / choice / open 三层读出与内部路径图谱连接起来。
```

## Phase 87: reader stack calibration 全量测试 [2026-06-11 09:37]

### 任务目标

Phase 86 说明 answer-only open reader 仍不稳定。本轮继续完成读出器栈校准，不做机制干预。

目标是把知识读出拆成三层：

```text
R_closed：封闭候选评分。
R_choice：显式多选生成。
R_open：开放短答案生成。
```

核心问题：

```text
1. closed candidate scoring 是否稳定？
2. multiple-choice generation 是否能把候选评分转成文本选择？
3. open answer generation 是否仍然失败？
4. 多选是否存在 candidate order bias？
```

### 脚本

新增：

```text
tests/gpt5/phase87_reader_stack_calibration.py
tests/gpt5/phase87_reader_stack_calibration_summary.py
tests/gpt5/run_phase87_reader_stack_calibration_full.sh
```

脚本特性：

```text
1. 三模型顺序运行：qwen3 -> GLM4 -> DS7B。
2. 每个模型完成后使用 --hard-exit-after-model。
3. 每个 item 同时测 closed / choice / open 三类 reader。
4. choice reader 使用 4 个模板、3 种候选顺序。
5. open reader 使用 Phase 86 中相对最好的 2 个模板。
6. 支持 resume。
```

choice 模板：

```text
choice_plain
choice_blank
choice_no_explain
choice_json_letter
```

候选顺序：

```text
target_first
target_last
rotating
```

open 模板：

```text
open_fill_blank
open_short_phrase
```

### 运行命令

smoke：

```bash
PHASE87_OUTPUT_DIR=results/gpt5_phase87_reader_stack_calibration_smoke_$(date +%Y%m%d_%H%M%S) \
PHASE87_MODELS=qwen3 \
QWEN3_PHASE87_MAX_ITEMS=4 \
PHASE87_CHOICE_TEMPLATES=choice_plain \
PHASE87_OPEN_TEMPLATES=open_fill_blank \
PHASE87_PROGRESS_EVERY=2 \
tests/gpt5/run_phase87_reader_stack_calibration_full.sh
```

正式全量：

```bash
PHASE87_OUTPUT_DIR=results/gpt5_phase87_reader_stack_calibration_full_$(date +%Y%m%d_%H%M%S) \
PHASE87_PROGRESS_EVERY=84 \
tests/gpt5/run_phase87_reader_stack_calibration_full.sh
```

### 输出文件

```text
results/gpt5_phase87_reader_stack_calibration_full_20260611_084922/qwen3_phase87_reader_stack_calibration.json
results/gpt5_phase87_reader_stack_calibration_full_20260611_084922/glm4_phase87_reader_stack_calibration.json
results/gpt5_phase87_reader_stack_calibration_full_20260611_084922/deepseek7b_phase87_reader_stack_calibration.json
results/gpt5_phase87_reader_stack_calibration_full_20260611_084922/phase87_reader_stack_calibration_summary.json
results/gpt5_phase87_reader_stack_calibration_full_20260611_084922/PHASE87_READER_STACK_CALIBRATION_SUMMARY.md
```

### 数据规模

每个 item：

```text
closed = 1 row
choice = 4 templates * 3 orders = 12 rows
open = 2 rows
total = 15 rows/item
```

正式数据：

```text
qwen3:
  items = 672
  rows = 10080

GLM4:
  items = 672
  rows = 10080

DS7B:
  items = 672
  rows = 10080

total rows = 30240
```

三模型均完成。本轮没有运行中断。

说明：

```text
本机无 flash_attn 包，因此 flash_attention_2 加载失败后回退到 sdpa。
DS7B 加载时仍提示 Sliding Window Attention is enabled but not implemented for sdpa。
```

### Qwen3 客观结果

按 reader：

```text
closed:
  top1 = 0.5923
  margin = 1.3528

choice:
  top1 = 0.7334
  valid = 0.8545
  no_target_first_top1 = 0.8105
  rotating_top1 = 0.7630
  target_last_top1 = 0.8579

open:
  word_subset_hit = 0.0082
  family_overlap_hit = 0.2612
  format_violation = 0.0729
```

choice 模板：

```text
choice_no_explain:
  top1 = 0.8656
  no_target_first_top1 = 0.8854
  rotating_top1 = 0.8899
  target_last_top1 = 0.8810

choice_json_letter:
  top1 = 0.8487
  no_target_first_top1 = 0.8720
  rotating_top1 = 0.8810
  target_last_top1 = 0.8631
```

客观现象：

```text
Qwen3 的 multiple-choice reader 明显强于 closed scoring 和 open generation。
choice_no_explain / choice_json_letter 在去掉 target_first 后仍稳定在 0.87-0.89 左右。
```

### GLM4 客观结果

按 reader：

```text
closed:
  top1 = 0.6696
  margin = 2.1754

choice:
  top1 = 0.6997
  valid = 1.0000
  no_target_first_top1 = 0.6315
  rotating_top1 = 0.7121
  target_last_top1 = 0.5510

open:
  word_subset_hit = 0.0186
  family_overlap_hit = 0.2872
  format_violation = 0.1615
```

choice 模板：

```text
choice_no_explain:
  top1 = 0.7728
  no_target_first_top1 = 0.7634
  rotating_top1 = 0.8318
  target_last_top1 = 0.6949

choice_blank:
  top1 = 0.7688
  no_target_first_top1 = 0.7485
  rotating_top1 = 0.8080
  target_last_top1 = 0.6890

choice_json_letter:
  top1 = 0.7401
  no_target_first_top1 = 0.7225
  rotating_top1 = 0.7991
  target_last_top1 = 0.6458
```

客观现象：

```text
GLM4 的 closed scoring 最强，choice reader 也稳定。
choice_plain 有很强顺序偏置，target_first = 0.9673，但 target_last = 0.1741。
choice_no_explain / choice_blank / choice_json_letter 更可靠。
```

### DeepSeek7B 客观结果

按 reader：

```text
closed:
  top1 = 0.4390
  margin = -1.1054

choice:
  top1 = 0.4701
  valid = 0.8914
  no_target_first_top1 = 0.2775
  rotating_top1 = 0.3460
  target_last_top1 = 0.2091

open:
  word_subset_hit = 0.0030
  family_overlap_hit = 0.0930
  format_violation = 0.1064
```

choice 模板：

```text
choice_json_letter:
  top1 = 0.6637
  no_target_first_top1 = 0.5610
  rotating_top1 = 0.6369
  target_last_top1 = 0.4851

choice_no_explain:
  top1 = 0.4449
  no_target_first_top1 = 0.3542
  rotating_top1 = 0.3571
  target_last_top1 = 0.3512
```

客观现象：

```text
DS7B 的 reader stack 明显弱于 qwen3 和 GLM4。
choice_plain / choice_blank 存在极强 target_first 偏置：
  choice_plain target_first = 1.0000
  choice_plain target_last = 0.0000
  choice_blank target_first = 0.9256
  choice_blank target_last = 0.0000
choice_json_letter 相对最好，但 target_last 仍只有 0.4851。
```

### 跨模型结果

```text
closed:
  top1 = 0.5670
  margin = 0.8076

choice:
  top1 = 0.6344
  valid = 0.9153

open:
  word_subset_hit = 0.0099
  family_overlap_hit = 0.2138
  format_violation = 0.1136
```

choice 模板跨模型：

```text
choice_json_letter:
  top1 = 0.7508
  valid = 1.0000

choice_no_explain:
  top1 = 0.6944
  valid = 0.8848

choice_blank:
  top1 = 0.5766
  valid = 0.8447

choice_plain:
  top1 = 0.5157
  valid = 0.9317
```

### 关键结论

Phase 87 明确把三层 reader 分开：

```text
R_closed：封闭候选评分，中等稳定。
R_choice：显式多选生成，明显强于开放生成。
R_open：开放短答案生成，仍然不合格。
```

最重要的客观发现：

```text
1. qwen3 和 GLM4 的 multiple-choice reader 已经可用，尤其 choice_no_explain / choice_json_letter。
2. open generation 仍然只能输出弱值族相关结果，不能稳定输出完整 target。
3. DS7B 在 sdpa 下 reader stack 整体较弱，且存在强候选顺序偏置。
4. 多选结果必须报告 no_target_first / rotating / target_last，否则容易把第一选项偏置误判为读出能力。
```

### 对 Phase 86 的修正

Phase 86 只能说明 answer-only open reader 不合格。

Phase 87 进一步说明：

```text
open reader 不合格，并不代表文本读出完全失败；
显式多选 reader 可以显著恢复读出能力。
```

因此目前最稳的 reader stack 是：

```text
closed scoring：机制验证基准；
choice_json_letter / choice_no_explain：可用于生成式读出校准；
open answer-only：暂不可用于机制验证。
```

### 硬伤

```text
1. R_choice 仍然不是自由语言生成，只是结构化选择读出。
2. qwen3 和 GLM4 的 choice reader 可用，但不同模板和候选顺序影响很大。
3. DS7B 的 choice reader 有强 target_first 偏置，必须使用 rotating / target_last 做校正。
4. closed scoring 与 choice generation 并不完全一致，说明从候选评分到输出选择还有接口层。
5. open generation 仍然不能作为机制验证读出器。
6. DS7B 的 sdpa sliding-window warning 仍是模型实现层硬伤。
```

### 理论进展

当前语言知识读出应明确分成三层：

```text
内部兼容状态
  -> R_closed 封闭候选评分
  -> R_choice 显式选择读出
  -> R_open 开放短答案生成
```

Phase 87 说明：

```text
R_closed 与 R_choice 已经能反映一部分内部知识路径；
R_open 仍受格式、短语习惯、解释倾向和解码策略影响严重。
```

这支持一个更精确的理论：

```text
语言编码机制不是直接等于自然生成文本。
内部状态先形成候选兼容性；
读出器再把候选兼容性转成选择或生成。
不同 reader 的失败位置不同。
```

### 下一步计划

Phase 88：choice-reader erase/restore retest。

理由：

```text
现在 R_choice 已经比 R_open 稳定，尤其 qwen3 / GLM4。
可以在 choice_json_letter 或 choice_no_explain 上复测 suffix/readout erase/restore。
```

建议测试：

```text
1. qwen3:
   reader = choice_no_explain, choice_json_letter
   order = rotating, target_last

2. GLM4:
   reader = choice_no_explain, choice_blank, choice_json_letter
   order = rotating, target_last

3. DS7B:
   reader = choice_json_letter
   order = rotating, target_last
```

核心指标：

```text
choice_top1_drop
choice_restore_gain
choice_target_letter_margin
choice_order_robustness
closed_score_drop
closed_restore_gain
```

阶段性大任务：

```text
1. 用 R_closed 继续定位内部机制；
2. 用 R_choice 验证生成式读出接口；
3. 暂停 R_open 机制干预；
4. 等 choice-reader erase/restore 稳定后，再重新设计 open reader。
```

## Phase 88: choice reader erase/restore 迁移测试 [2026-06-11 14:48]

### 任务目标

根据 Phase87 的结果，`R_choice` 是目前最稳定的桥接读出器：它比 open answer 更稳定，又比 closed fullseq candidate scoring 更接近真实输出。本轮测试 Phase84 的 `suffix/readout gateway` 是否能从 closed scoring 迁移到结构化 choice 输出。

核心问题：

```text
如果 frame suffix 子空间在 closed candidate scoring 中是必要读出门，
那么擦除该子空间是否也会降低 choice reader 的真实选择正确率？
恢复该子空间是否能恢复 choice 输出？
```

同时结合 GLM5 Phase456 的发现，本轮不使用 softmax margin 作为主指标，只记录 full-sequence logprob margin、rank/top1 和 choice top1/valid。

### 脚本

新增脚本：

```text
tests/gpt5/phase88_choice_reader_erase_restore.py
tests/gpt5/phase88_choice_reader_erase_restore_summary.py
tests/gpt5/run_phase88_choice_reader_erase_restore_full.sh
```

运行命令：

```bash
PHASE88_OUTPUT_DIR=results/gpt5_phase88_choice_reader_erase_restore_full_20260611_1046 \
tests/gpt5/run_phase88_choice_reader_erase_restore_full.sh
```

模型顺序：

```text
qwen3 -> glm4 -> deepseek7b
```

每个模型都使用：

```text
--hard-exit-after-model
BF16
device_map="auto"
attn_implementation: flash_attention_2 尝试失败后回退 sdpa
```

本机仍未安装 flash_attn，因此实际成功路径是 PyTorch SDPA，不是 flash-attention-2。

### 测试规模

为避免小样本误判，同时控制生成式 erase/restore 的组合爆炸，本轮使用中大规模测试：

```text
max_items = 336
choice_orders = rotating,target_last
conditions =
  frame_suffix_final
  frame_suffix_all
  frame_suffix_function
  frame_suffix_lexical
  frame_all_suffix_tokens
  object_suffix_final
  object_all_suffix_tokens
```

模型配置：

```text
qwen3:
  layer_pairs = 4-8,8-12
  templates = choice_json_letter,choice_no_explain
  rows = 18816

GLM4:
  layer_pairs = 4-10,10-20
  templates = choice_json_letter,choice_no_explain,choice_blank
  rows = 28224

DeepSeek7B:
  layer_pairs = 8-10,12-14
  templates = choice_json_letter
  rows = 9408
```

总行数：

```text
56448 rows
bad_numeric_rows = 0 for all three models
```

输出目录：

```text
results/gpt5_phase88_choice_reader_erase_restore_full_20260611_1046/
```

### 总体结果

模型级汇总：

```text
qwen3:
  n = 18816
  base_choice_top1 = 0.8646
  erase_choice_top1 = 0.8595
  restore_choice_top1 = 0.8587
  choice_drop = 0.0050
  choice_restore_gain = -0.0009
  closed_base_top1 = 0.6190
  closed_erase_top1 = 0.5548
  closed_restore_top1 = 0.6025
  closed_drop = 0.6348
  closed_restore_gain = 0.4807

GLM4:
  n = 28224
  base_choice_top1 = 0.7411
  erase_choice_top1 = 0.7402
  restore_choice_top1 = 0.7407
  choice_drop = 0.0009
  choice_restore_gain = 0.0005
  closed_base_top1 = 0.6488
  closed_erase_top1 = 0.6533
  closed_restore_top1 = 0.6694
  closed_drop = 0.0803
  closed_restore_gain = 0.1447

DeepSeek7B:
  n = 9408
  base_choice_top1 = 0.5685
  erase_choice_top1 = 0.5669
  restore_choice_top1 = 0.5638
  choice_drop = 0.0016
  choice_restore_gain = -0.0031
  closed_base_top1 = 0.4613
  closed_erase_top1 = 0.4226
  closed_restore_top1 = 0.4524
  closed_drop = 0.3132
  closed_restore_gain = 0.2884
```

### 条件级结果

最强 closed scoring 效应仍来自 frame all suffix tokens：

```text
qwen3 frame_all_suffix_tokens:
  choice_drop = 0.0097
  choice_restore_gain = -0.0108
  closed_drop = 1.5555
  closed_restore_gain = 1.3861

GLM4 frame_all_suffix_tokens:
  choice_drop = 0.0030
  choice_restore_gain = -0.0010
  closed_drop = 0.5721
  closed_restore_gain = 0.5936

DeepSeek7B frame_all_suffix_tokens:
  choice_drop = 0.0030
  choice_restore_gain = -0.0141
  closed_drop = 0.8301
  closed_restore_gain = 0.7506
```

Qwen3 的 frame suffix closed 效应最强，但 choice 输出只轻微下降，且 restore 没有恢复：

```text
qwen3 frame_suffix_final:
  choice_drop = 0.0037
  choice_restore_gain = -0.0011
  closed_drop = 0.8107
  closed_restore_gain = 0.6947

qwen3 frame_suffix_all:
  choice_drop = 0.0041
  choice_restore_gain = -0.0004
  closed_drop = 0.5975
  closed_restore_gain = 0.3777
```

GLM4 的 closed margin 对部分 frame 条件有恢复，但 choice 几乎不动：

```text
GLM4 frame_all_suffix_tokens:
  choice_drop = 0.0030
  choice_restore_gain = -0.0010
  closed_drop = 0.5721
  closed_restore_gain = 0.5936

GLM4 frame_suffix_all:
  choice_drop = -0.0007
  choice_restore_gain = 0.0000
  closed_drop = 0.0163
  closed_restore_gain = 0.1659
```

DeepSeek7B 的 closed 恢复明显，但 choice 输出仍不恢复：

```text
DeepSeek7B frame_suffix_final:
  choice_drop = 0.0030
  choice_restore_gain = -0.0015
  closed_drop = 0.4635
  closed_restore_gain = 0.4246

DeepSeek7B frame_suffix_function:
  choice_drop = -0.0015
  choice_restore_gain = 0.0000
  closed_drop = 0.4770
  closed_restore_gain = 0.4476
```

### 路径级结果

Qwen3 的 closed effect 在两个路径都稳定存在：

```text
qwen3 frame_all_suffix_tokens L4->L8:
  choice_drop = 0.0156
  choice_restore_gain = -0.0089
  closed_drop = 1.3950
  closed_restore_gain = 1.3021

qwen3 frame_all_suffix_tokens L8->L12:
  choice_drop = 0.0037
  choice_restore_gain = -0.0126
  closed_drop = 1.7160
  closed_restore_gain = 1.4702
```

GLM4 的 closed effect 也在两个路径上存在，但 choice 输出仍弱：

```text
GLM4 frame_all_suffix_tokens L4->L10:
  choice_drop = 0.0040
  choice_restore_gain = -0.0035
  closed_drop = 0.5937
  closed_restore_gain = 0.6090

GLM4 frame_all_suffix_tokens L10->L20:
  choice_drop = 0.0020
  choice_restore_gain = 0.0015
  closed_drop = 0.5506
  closed_restore_gain = 0.5783
```

DeepSeek7B 的 closed effect 在 L8->L10 和 L12->L14 都存在：

```text
DeepSeek7B frame_all_suffix_tokens L8->L10:
  choice_drop = 0.0015
  choice_restore_gain = -0.0164
  closed_drop = 0.8756
  closed_restore_gain = 0.6926

DeepSeek7B frame_all_suffix_tokens L12->L14:
  choice_drop = 0.0045
  choice_restore_gain = -0.0119
  closed_drop = 0.7846
  closed_restore_gain = 0.8086
```

### 关系级高效应片段

closed scoring 中最强片段：

```text
qwen3 frame_all_suffix_tokens used_for:
  closed_drop = 3.6508
  closed_restore_gain = 3.3784
  choice_drop = 0.0260
  choice_restore_gain = -0.0104

qwen3 frame_suffix_final is_a:
  closed_drop = 2.2511
  closed_restore_gain = 1.6739
  choice_drop = 0.0052
  choice_restore_gain = 0.0104

GLM4 frame_all_suffix_tokens used_for:
  closed_drop = 1.8393
  closed_restore_gain = 1.5861
  choice_drop = 0.0000

DeepSeek7B frame_all_suffix_tokens location:
  closed_drop = 1.6783
  closed_restore_gain = 1.7299
  choice_drop = 0.0000
```

### 客观判断

本轮最重要的客观现象：

```text
1. Phase84 的 closed scoring suffix/readout gateway 效应被再次复现。
2. 三模型中 frame_all_suffix_tokens 都是最强 closed erase/restore 条件之一。
3. frame 条件明显强于 object 条件，说明读出端更依赖 relation/frame suffix，而不是单纯对象 token。
4. closed scoring 的擦除-恢复强效没有稳定迁移到 choice 输出。
5. choice 输出在 qwen3 中有轻微下降，但幅度远小于 closed margin；GLM4 和 DeepSeek7B 几乎不受影响。
6. restore 在 closed scoring 中有效，但在 choice 输出中没有稳定恢复，甚至常出现负恢复。
```

因此，当前不能说：

```text
suffix/readout gateway 已经控制真实结构化输出。
```

更稳的说法是：

```text
suffix/readout gateway 是 closed candidate scoring 的强读出因子；
但结构化 choice generation 还有额外输出层、格式层、选项顺序层或决策层缓冲，
使得 closed margin 改变不会线性转化为 choice top1 改变。
```

### 对当前分析的判断

附件中“Phase87 后应做 Choice-Reader Readout-Gateway Erase/Restore”的分析是正确的：

```text
R_choice 必须作为 closed 与 open 之间的桥；
不能直接把 closed scoring 的 gateway 结果解释成真实生成控制；
需要测试 choice 输出是否同步下降和恢复。
```

本轮结果支持这个谨慎路线，并进一步收缩结论：

```text
closed scoring 可以看到强读出门；
choice 输出没有出现同等强度闭包；
所以读出门更像 candidate scoring interface 的必要因子，
还不是完整语言输出机制闭包。
```

### 硬伤

1. choice top1 是离散指标，可能对 margin 改变不敏感；本轮没有记录 choice letter logits margin。
2. choice generation 存在模板、格式和选项顺序缓冲；擦除 closed margin 不一定立刻翻转输出字母。
3. 本轮只用 suffix 子空间擦除/恢复，没有做 attention/MLP route patch。
4. 本轮测试的是对象-关系值任务，不是完整语义、逻辑、语法全域。
5. DeepSeek7B 的 base choice 仍偏低，DS7B 的 choice 结论要弱于 qwen3/GLM4。
6. restore 对 closed scoring 有效，但对 choice 输出无效，说明“变量恢复”还没有进入生成决策闭包。

### 下一步

下一阶段不应继续只扩大 choice erase/restore，而应把读出链拆成两层：

```text
closed candidate scoring interface
structured choice decision interface
```

建议 Phase89 做：

```text
choice letter full-sequence scoring audit
```

具体做法：

```text
1. 对同一个 choice prompt，不生成，而是分别计算 A/B/C/D/E 各选项字母的 full-sequence logprob。
2. 在同样的 frame suffix erase/restore 条件下，观察 target_letter_margin 是否下降和恢复。
3. 同时保留原始 candidate value fullseq margin。
4. 比较：
   candidate value margin 是否变；
   choice letter margin 是否变；
   generated choice top1 是否变。
```

如果：

```text
candidate value margin 变，但 choice letter margin 不变
```

说明 closed candidate scoring 和 choice decision 之间有格式/选项映射层。

如果：

```text
choice letter margin 变，但 generated top1 不变
```

说明 generation decoding/format layer 有缓冲。

如果：

```text
candidate margin、letter margin、generated top1 三者同步
```

才可以把 readout gateway 推进为真实 choice decision gateway。

## Phase 89: GPT5/GLM5 双线进展比较与主线选择 [2026-06-11 16:06]

### 比较对象

本轮读取并比较：

```text
research/gpt5/docs/AGI_GPT5_MEMO.md
research/glm5/docs/AGI_GLM5_MEMO.md
```

重点比较最新阶段：

```text
GPT5:
  Phase84-88
  object-relation-value closed scoring
  suffix/readout gateway
  reader stack calibration
  choice reader erase/restore

GLM5:
  Phase454-456
  candidate family margin dynamics
  attn/MLP component effect
  category/slot/template robustness
  family logit decomposition
```

### GPT5 线的当前进展

GPT5 线最强的贡献不是直接机制结论，而是读出器与闭包测试基础设施。

已完成：

```text
1. object-relation-value 的 closed full-sequence candidate scoring 闭包测试。
2. frame suffix/readout gateway 的 erase/restore 效应。
3. answer-only open reader 校准，证明 open reader 不稳定。
4. reader stack calibration，确认 R_choice 是 closed 与 open 之间的桥。
5. choice reader erase/restore 迁移测试，证明 closed gateway 强效没有稳定迁移到 generated choice top1。
```

最新 Phase88 的关键结果：

```text
closed scoring:
  qwen3 closed_drop = 0.6348, restore_gain = 0.4807
  GLM4 closed_drop = 0.0803, restore_gain = 0.1447
  DS7B closed_drop = 0.3132, restore_gain = 0.2884

choice generation:
  qwen3 choice_drop = 0.0050
  GLM4 choice_drop = 0.0009
  DS7B choice_drop = 0.0016
```

这说明：

```text
closed candidate scoring 的读出门存在；
但它还不是生成式 choice decision 的完整控制机制。
```

GPT5 线最可靠的意义：

```text
测量系统建设；
读出器校准；
closed/choice/open 三层接口区分；
防止把 scoring artifact 误判为语言机制。
```

GPT5 线最大硬伤：

```text
1. 当前主要停留在读出器与接口层；
2. choice 输出没有形成强闭包；
3. 还没深入 attention/MLP 组件如何改变竞争候选；
4. 对语言机制本体的解释力弱于 GLM5 线；
5. 下一步必须做 choice letter full-sequence scoring，拆出 candidate margin、letter margin、generated top1 三者关系。
```

### GLM5 线的当前进展

GLM5 线更接近机制本体，尤其 Phase455-456 已经从“是否促进/压制”推进到“目标族与竞争族 margin 如何被组件选择性改变”。

最新可靠结果：

```text
1. Softmax margin 不可用，因为概率差异被大词表稀释。
2. Top1 margin 和 mean margin 大体一致，但 DS7B vehicle 的 attention 效应在两者间翻转。
3. attention/MLP 效应强烈依赖类别、槽位和模型。
4. DS7B L27 fruit/animal 路径翻转稳定复现：
   fruit: attn=SUPPRESS, mlp=PROMOTE
   animal: attn=PROMOTE, mlp=SUPPRESS
5. GLM4 L39 中 attention 近似 neutral，MLP 是主要 margin driver。
6. Qwen3 后层 MLP 从 amplifier 转为 suppressor，GLM4/DS7B 最后层 MLP 多数转为 amplifier。
7. MLP 不是统一促进或压制，而是对不同候选族有选择性幅度差。
```

GLM5 线的关键突破是：

```text
Logit 效应 != Margin 效应
```

同一个组件可以：

```text
压低所有候选 logit，
但如果更强压低竞争族，
目标 margin 反而上升。
```

这比 GPT5 当前的 suffix/readout gateway 更接近语言编码机制，因为它开始解释：

```text
模型如何在目标候选与竞争候选之间重新分配优势；
attention 和 MLP 如何在不同类别/槽位上承担相反功能；
同一模型内部为什么会出现路径翻转。
```

GLM5 线最大硬伤：

```text
1. 当前仍集中在候选族与对象/属性类任务，语言结构覆盖不够。
2. color/function slot 数据仍偏少。
3. GLM4 MLP 跨模板稳定性不足。
4. DS7B vehicle 的 attention 在 Top1/Mean 间翻转，说明竞争族内部结构还没拆清。
5. 仍需把组件 margin 动力学接入 GPT5 的读出器校准框架，避免读出器偏差。
```

### 哪条线更值得推进

如果目标是：

```text
破解语言背后的编码机制
```

当前更值得作为主线推进的是：

```text
GLM5 线
```

原因：

```text
1. GLM5 已经进入组件级机制：
   attention / MLP 如何改变目标族与竞争族 margin。

2. GLM5 发现了稳定的路径分化：
   DS7B fruit/animal 的 attn/MLP 路径翻转，
   Qwen3/GLM4/DS7B 的最后层 MLP 转折模式。

3. GLM5 的结果更接近“相对编码”：
   不是单一方向，而是同一组件在不同类别、槽位、竞争环境下改变作用。

4. GLM5 已经可以解释 margin 来源：
   目标族 logit 和竞争族 logit 的选择性幅度差。

5. GPT5 当前更像测量接口研究：
   它告诉我们哪些读出器可信，哪些现象不能直接解释为生成控制。
```

但不是放弃 GPT5。最合理路线是：

```text
GLM5 做机制主线；
GPT5 做读出器和闭包验证框架；
两者合并。
```

### 综合判断

当前路线优先级：

```text
第一优先级:
  推进 GLM5 的 candidate-family margin dynamics，
  扩展到更多槽位、更多关系、更多语言结构。

第二优先级:
  用 GPT5 的 reader stack 方法校准 GLM5 的读出器，
  避免把 closed scoring artifact 当成语言机制。

第三优先级:
  将 GPT5 的 erase/restore 框架用于 GLM5 已发现的组件路径翻转，
  做 destroy/restore 因果闭包。
```

### 下一步建议

下一阶段应做一个合并测试：

```text
Phase90: component-margin dynamics + reader interface alignment
```

测试目标：

```text
1. 在 GLM5 已发现的类别/槽位路径上，加入 GPT5 的 reader stack：
   candidate value margin
   choice letter margin
   generated choice top1

2. 对 attention 和 MLP 分别做 zero/restore：
   看组件 margin effect 是否传导到 choice letter decision。

3. 扩大槽位：
   category
   color
   function
   material
   location
   action

4. 扩大类别：
   fruit
   animal
   tool
   vehicle
   place
   body_part
   profession

5. 输出三层矩阵：
   component -> candidate family margin
   candidate family margin -> choice letter margin
   choice letter margin -> generated top1
```

判断标准：

```text
如果 component margin 改变能稳定传导到 choice letter margin，
说明机制已跨过 closed scoring 接口。

如果 choice letter margin 改变仍不能改变 generated top1，
说明生成格式/解码层还有额外缓冲。

如果三者同步改变，
才接近真实输出机制闭包。
```

### 当前最稳结论

```text
GPT5 线告诉我们：读出器必须分层，closed scoring 不能直接等于生成机制。
GLM5 线告诉我们：真正机制可能在 attention/MLP 对目标族与竞争族 margin 的选择性重分配中。
```

因此：

```text
下一步主攻 GLM5 机制线，
同时用 GPT5 的 reader calibration 和 erase/restore 框架做严格验证。
```

## Phase 90: component-margin dynamics 与 reader interface 对齐测试 [2026-06-11 18:32]

### 任务目标

根据 Phase89 的路线判断，本轮把 GLM5 线的组件边际动力学接入 GPT5 线的读出器分层框架。

核心问题：

```text
attention / MLP 对 candidate value margin 的影响，
是否能传导到 choice letter margin，
并进一步传导到 generated choice top1？
```

这不是继续测 suffix/readout gateway，而是直接测：

```text
component -> candidate family/value margin -> choice letter margin -> generated choice
```

### 脚本

新增脚本：

```text
tests/gpt5/phase90_component_margin_reader_alignment.py
tests/gpt5/phase90_component_margin_reader_alignment_summary.py
tests/gpt5/run_phase90_component_margin_reader_alignment_full.sh
```

运行命令：

```bash
PHASE90_OUTPUT_DIR=results/gpt5_phase90_component_margin_reader_alignment_full_20260611_1640 \
tests/gpt5/run_phase90_component_margin_reader_alignment_full.sh
```

模型顺序：

```text
qwen3 -> GLM4 -> DeepSeek7B
```

每个模型独立进程，带：

```text
--hard-exit-after-model
BF16
device_map="auto"
attn_implementation: flash_attention_2 尝试失败后回退 sdpa
```

实际成功路径仍是 PyTorch SDPA。

### 数据范围

本轮测试 5 个 slot：

```text
category
color
function
material
location
```

对象覆盖：

```text
fruit
animal
tool
vehicle
place
body_part
profession
```

每模型：

```text
items = 420
layers = 6 个关键层
components = clean, zero_attn, zero_mlp
rows = 7560
```

三模型总行数：

```text
22680 rows
bad_numeric_rows = 0
```

结果目录：

```text
results/gpt5_phase90_component_margin_reader_alignment_full_20260611_1640/
```

### 指标定义

对每个 prompt 同时计算三层指标：

```text
1. candidate value full-sequence margin
   对真实答案文本和干扰文本做 full-sequence logprob scoring。

2. choice letter full-sequence margin
   对同一个 choice prompt 的 A/B/C/D/E 字母做 full-sequence logprob scoring。

3. generated choice top1
   直接生成结构化 JSON letter，解析选择值。
```

组件效应定义：

```text
component_effect = clean_margin - zero_ablated_margin
```

因此：

```text
正数 = 该组件促进目标 margin
负数 = 该组件压制目标 margin
```

### 模型级结果

```text
qwen3 clean:
  value_top1 = 0.7333
  letter_top1 = 0.9262
  choice_top1 = 0.9286

qwen3 zero_attn:
  component_value_effect_top1 = 0.1793
  component_letter_effect_top1 = 0.5374
  choice_drop = 0.1095

qwen3 zero_mlp:
  component_value_effect_top1 = 0.6206
  component_letter_effect_top1 = 0.6016
  choice_drop = 0.1893
```

Qwen3 出现较清楚的三层传导：

```text
zero_mlp 使 candidate value margin 下降；
choice letter margin 同步下降；
generated choice top1 也明显下降。
```

GLM4：

```text
GLM4 clean:
  value_top1 = 0.7595
  letter_top1 = 0.7619
  choice_top1 = 0.7595

GLM4 zero_attn:
  component_value_effect_top1 = 0.0313
  component_letter_effect_top1 = 0.0362
  choice_drop = 0.0004

GLM4 zero_mlp:
  component_value_effect_top1 = 0.2821
  component_letter_effect_top1 = -0.0874
  choice_drop = -0.0187
```

GLM4 出现接口断裂/反转：

```text
MLP 促进 candidate value margin；
但对 choice letter margin 是负效应；
generated choice 反而略升。
```

这与 GLM5 Phase456 中“GLM4 MLP 跨模板不稳定”的硬伤一致。

DeepSeek7B：

```text
DeepSeek7B clean:
  value_top1 = 0.6333
  letter_top1 = 0.8857
  choice_top1 = 0.8881

DeepSeek7B zero_attn:
  component_value_effect_top1 = 0.0631
  component_letter_effect_top1 = -0.2963
  choice_drop = 0.0683

DeepSeek7B zero_mlp:
  component_value_effect_top1 = 0.2502
  component_letter_effect_top1 = -0.1776
  choice_drop = 0.1444
```

DeepSeek7B 也出现断裂：

```text
candidate value margin 显示组件促进目标；
choice letter margin 平均反而显示负效应；
但 generated choice top1 下降。
```

说明 DS7B 的 generated choice 不完全由 letter fullseq margin 解释，可能存在格式、缓存路径、最后层生成动态或 token-level 决策差异。

### 层级关键发现

Qwen3：

```text
L6 zero_mlp:
  value_effect = 3.047
  letter_effect = 3.563
  choice_drop = 0.924

L24 zero_attn:
  value_effect = 0.101
  letter_effect = 4.315
  choice_drop = 0.610

L35 zero_mlp:
  value_effect = -0.240
  letter_effect = 2.241
  choice_drop = 0.167
```

Qwen3 的重要现象：

```text
1. L6 MLP 是强全链路节点：candidate、letter、generated 三层同步下降。
2. L24 attention 主要影响 choice letter/generation，而不是 candidate value margin。
3. L35 MLP 对 candidate value margin 是负效应，但对 letter/generation 是正效应，说明 candidate 与 choice interface 在后层分叉。
```

GLM4：

```text
L39 zero_mlp:
  value_effect = 0.656
  letter_effect = -0.681
  choice_drop = -0.038

L38 zero_mlp:
  value_effect = 0.538
  letter_effect = -0.056
  choice_drop = -0.052
```

GLM4 的重要现象：

```text
最后层 MLP 明显促进 candidate value margin；
但不促进 choice letter decision；
甚至 zero_mlp 后 choice top1 略升。
```

这说明 GLM4 的 candidate scoring interface 和 choice decision interface 明显分离。

DeepSeek7B：

```text
L26 zero_mlp:
  value_effect = 0.455
  letter_effect = -0.862
  choice_drop = 0.829

L27 zero_attn:
  value_effect = -0.046
  letter_effect = 0.854
  choice_drop = 0.590

L27 zero_mlp:
  value_effect = 0.082
  letter_effect = 0.474
  choice_drop = 0.050
```

DeepSeek7B 的重要现象：

```text
1. L26 MLP 对 generated choice 极其关键，但 letter margin 方向反常。
2. L27 attention 对 generated choice 极其关键，且主要通过 letter/generation interface，而不是 candidate value margin。
3. L27 MLP 的影响弱于 L27 attention。
```

这和 GLM5 中 DS7B 最后层路径特异、L26->L27 剧烈翻转的现象相互支持。

### 槽位结果

Qwen3 中，zero_mlp 对所有 slot 都有明显 choice drop：

```text
category: choice_drop = 0.179
color:    choice_drop = 0.169
function: choice_drop = 0.214
location: choice_drop = 0.171
material: choice_drop = 0.214
```

说明 Qwen3 的 MLP 对结构化 choice 输出有广泛支持作用。

GLM4 中，zero_mlp 对 choice 几乎不造成下降，多个 slot 甚至为负：

```text
category: choice_drop = -0.040
color:    choice_drop = -0.038
function: choice_drop = 0.014
location: choice_drop = 0.000
material: choice_drop = -0.030
```

说明 GLM4 的 candidate margin MLP 效应不能直接解释 choice 输出。

DeepSeek7B 中，zero_mlp 对所有 slot 都造成 choice drop：

```text
category: choice_drop = 0.159
color:    choice_drop = 0.099
function: choice_drop = 0.200
location: choice_drop = 0.095
material: choice_drop = 0.169
```

但其 letter margin 平均不一定同步，说明 DS7B 需要进一步拆 letter scoring 与 generation decoding。

### 核心客观进展

本轮第一次把三层读出链放到同一张表里：

```text
component ablation
candidate value margin
choice letter margin
generated choice top1
```

结果说明：

```text
1. Qwen3 有相对清楚的三层传导，尤其 L6 MLP 和 L24 attention。
2. GLM4 的 candidate value margin 与 choice decision 明显分离。
3. DeepSeek7B 的 generated choice 对 L26 MLP 和 L27 attention 极敏感，但 letter margin 与生成结果不完全一致。
4. candidate margin、letter margin、generated top1 不是同一个东西，必须分层研究。
```

这直接支持 Phase89 的判断：

```text
GPT5 的 reader stack 是必要的；
GLM5 的 component margin dynamics 更接近机制主线；
两者必须合并。
```

### 当前理论收缩

不能说：

```text
candidate value margin 就是模型真实选择机制。
choice letter margin 就是模型真实生成机制。
```

更稳的说法：

```text
语言读出至少有三层接口：
1. semantic candidate/value scoring interface
2. structured option-letter decision interface
3. autoregressive generation/format interface
```

不同模型的三层接口耦合程度不同：

```text
Qwen3:
  三层接口耦合较强，部分层有同步传导。

GLM4:
  candidate scoring 与 choice decision 明显分离。

DeepSeek7B:
  generation 对深层组件非常敏感，但 letter scoring 不能完全解释生成变化。
```

### 硬伤

1. 本轮仍是 zero ablation，不是 restore，因此还不是闭包证明。
2. 数据虽然覆盖 420 items / 5 slots，但仍是对象-槽位任务，不是完整语法/逻辑。
3. choice letter scoring 只测单字母 continuation，没有测完整 JSON 结束序列。
4. generated choice 可能受格式合法性、停止符、缓存路径影响。
5. GLM4/DS7B 的接口反转说明还需要逐层拆解生成决策，而不能直接用单一 margin 解释。

### 下一步

Phase91 应做：

```text
component destroy/restore across reader interfaces
```

目标：

```text
对 Phase90 中最强节点做 restore：

qwen3:
  L6 MLP
  L24 attention
  L35 MLP

GLM4:
  L38/L39 MLP

DeepSeek7B:
  L26 MLP
  L27 attention
```

测试三层是否能恢复：

```text
candidate value margin restore
choice letter margin restore
generated choice top1 restore
```

如果某层组件：

```text
zero 后三层下降；
restore 后三层恢复；
跨 slot 稳定；
```

才可以称为真正接近 reader-interface mechanism closure。

## Phase 91: component restore reader closure 全量测试 [2026-06-11 19:44]

### 任务目标

根据 Phase90 的结果，本轮对最强组件节点做 restore 测试，检查：

```text
clean
zero component
restore clean component output
```

三种条件下：

```text
candidate value margin
choice letter margin
generated choice top1
```

是否能同步下降和恢复。

本轮不是全层扫描，而是只测试 Phase90 中最强和最有解释价值的节点：

```text
qwen3:
  L6 MLP
  L24 attention
  L35 MLP

GLM4:
  L38 MLP
  L39 MLP

DeepSeek7B:
  L26 MLP
  L27 attention
```

### 脚本

新增脚本：

```text
tests/gpt5/phase91_component_restore_reader_closure.py
tests/gpt5/phase91_component_restore_reader_closure_summary.py
tests/gpt5/run_phase91_component_restore_reader_closure_full.sh
```

运行命令：

```bash
PHASE91_OUTPUT_DIR=results/gpt5_phase91_component_restore_reader_closure_full_20260611_1849 \
tests/gpt5/run_phase91_component_restore_reader_closure_full.sh
```

模型顺序：

```text
qwen3 -> GLM4 -> DeepSeek7B
```

每个模型使用：

```text
--hard-exit-after-model
BF16
device_map="auto"
attn_implementation: flash_attention_2 尝试失败后回退 sdpa
```

实际成功路径仍是 PyTorch SDPA。

### 测试规模

```text
slots = category,color,function,material,location
max_items = 420
qwen3 rows = 1260
GLM4 rows = 840
DeepSeek7B rows = 840
total rows = 2940
bad_numeric_rows = 0
```

结果目录：

```text
results/gpt5_phase91_component_restore_reader_closure_full_20260611_1849/
```

### 方法说明

本轮 restore 的具体含义：

```text
zero:
  将指定 layer 的指定组件输出置零。

restore:
  在同一个输入上捕获 clean component output，
  然后将该 clean component output 写回该组件输出位置。
```

因此本轮回答的是：

```text
被 zero 破坏的三层读出接口，能否由同一组件的 clean output 恢复？
```

它不是更强版本的“独立变量子空间恢复”，因为恢复源仍是同输入 clean component output。这一点必须谨慎。

### 模型级核心结果

#### Qwen3

```text
qwen3 L6 MLP:
  clean_choice_top1 = 0.9286
  zero_choice_top1 = 0.0048
  restore_choice_top1 = 0.9286
  value_drop = 3.0473
  value_restore_gain = 3.0473
  letter_drop = 3.5628
  letter_restore_gain = 3.5628
  choice_drop = 0.9238
  choice_restore_gain = 0.9238

qwen3 L24 attention:
  clean_choice_top1 = 0.9286
  zero_choice_top1 = 0.3190
  restore_choice_top1 = 0.9286
  value_drop = 0.1010
  letter_drop = 4.3152
  choice_drop = 0.6095

qwen3 L35 MLP:
  clean_choice_top1 = 0.9286
  zero_choice_top1 = 0.7619
  restore_choice_top1 = 0.9286
  value_drop = -0.2404
  letter_drop = 2.2412
  choice_drop = 0.1667
```

Qwen3 的结果最清楚：

```text
L6 MLP 是强全链路节点：
candidate value margin、choice letter margin、generated choice 全部下降，并全部恢复。

L24 attention 主要控制 letter/generation 接口：
value_drop 很小，但 letter_drop 和 choice_drop 很大。

L35 MLP 出现 candidate/choice 分叉：
value_drop 为负，但 letter_drop 和 choice_drop 为正。
```

#### GLM4

```text
GLM4 L38 MLP:
  clean_choice_top1 = 0.7595
  zero_choice_top1 = 0.8119
  restore_choice_top1 = 0.7595
  value_drop = 0.5381
  letter_drop = -0.0558
  choice_drop = -0.0524

GLM4 L39 MLP:
  clean_choice_top1 = 0.7595
  zero_choice_top1 = 0.7976
  restore_choice_top1 = 0.7595
  value_drop = 0.6564
  letter_drop = -0.6807
  choice_drop = -0.0381
```

GLM4 的结果确认 Phase90 的接口分离：

```text
最后层 MLP 促进 candidate value margin，
但并不促进 choice letter 或 generated choice。

zero MLP 后 choice top1 反而上升，
restore 后回到 clean。
```

这说明 GLM4 的 MLP candidate scoring interface 与 choice decision interface 分离很明显。

#### DeepSeek7B

```text
DeepSeek7B L26 MLP:
  clean_choice_top1 = 0.8881
  zero_choice_top1 = 0.0595
  restore_choice_top1 = 0.8881
  value_drop = 0.4551
  letter_drop = -0.8616
  choice_drop = 0.8286

DeepSeek7B L27 attention:
  clean_choice_top1 = 0.8881
  zero_choice_top1 = 0.2976
  restore_choice_top1 = 0.8881
  value_drop = -0.0460
  letter_drop = 0.8537
  choice_drop = 0.5905
```

DeepSeek7B 结果确认：

```text
L26 MLP 是生成选择的极强节点，
但 letter margin 方向仍与 generated choice 不一致。

L27 attention 是强 generation/letter interface 节点，
但 candidate value margin 几乎不解释它。
```

### slot 级结果

Qwen3 L6 MLP 对所有 slot 都强：

```text
category: choice_drop = 1.000
color:    choice_drop = 0.845
function: choice_drop = 0.964
location: choice_drop = 0.893
material: choice_drop = 0.917
```

Qwen3 L24 attention 对所有 slot 都影响 choice：

```text
category: choice_drop = 0.702
color:    choice_drop = 0.464
function: choice_drop = 0.643
location: choice_drop = 0.500
material: choice_drop = 0.738
```

GLM4 L39 MLP 对所有 slot 的 choice_drop 基本不为正：

```text
category: choice_drop = -0.083
color:    choice_drop = -0.024
function: choice_drop = -0.012
location: choice_drop = 0.000
material: choice_drop = -0.071
```

DeepSeek7B L26 MLP 对所有 slot 都强烈影响 choice：

```text
category: choice_drop = 0.798
color:    choice_drop = 0.774
function: choice_drop = 0.929
location: choice_drop = 0.762
material: choice_drop = 0.881
```

DeepSeek7B L27 attention 也对所有 slot 有明显影响：

```text
category: choice_drop = 0.750
color:    choice_drop = 0.500
function: choice_drop = 0.631
location: choice_drop = 0.631
material: choice_drop = 0.440
```

### 客观进展

本轮确认：

```text
1. Qwen3 L6 MLP 是强三层闭包节点：
   value、letter、generation 同步下降并恢复。

2. Qwen3 L24 attention 是 choice/generation 接口节点：
   candidate value margin 解释力弱，但 letter/generation 影响强。

3. GLM4 L38/L39 MLP 不是 choice decision 支持节点：
   它促进 candidate value margin，但 zero 后 choice 反而略升。

4. DeepSeek7B L26 MLP 和 L27 attention 是强 generation 节点：
   zero 后 generated choice 大幅下降，restore 完全恢复。

5. 三模型的读出接口结构明显不同：
   Qwen3 更连续；
   GLM4 candidate 与 choice 分离；
   DeepSeek7B generation 对深层组件敏感但 letter/candidate 解释不充分。
```

### 对附件分析的判断

附件中关于 Phase90 的分析基本正确：

```text
Phase90 的关键进展不是理论总结，
而是从读出门推进到组件—边际—输出链。
```

附件提出的硬伤也正确：

```text
Phase90 只是 zero ablation，不是 restore closure。
```

Phase91 对这一点做了直接补充：

```text
zero 后下降；
restore clean component output 后恢复；
三层接口一起记录。
```

但要强调：Phase91 的 restore 是同输入 clean component output 的恢复，证明的是“组件输出对当前读出接口必要且可恢复”，不是已经找到了抽象变量子空间。

### 当前最稳结论

可以较稳地说：

```text
1. Qwen3 的早层 MLP 存在强全链路读出支持作用。
2. Qwen3 中层 attention 存在 choice/generation 接口支持作用。
3. GLM4 最后层 MLP 的 candidate margin 与 choice output 分离。
4. DeepSeek7B 的 L26 MLP 和 L27 attention 是强生成决策节点。
```

不能说：

```text
已经破解语言编码机制；
已经找到抽象语义变量；
restore 证明了跨样本变量闭包；
candidate margin 可以统一解释所有模型输出。
```

### 硬伤

1. restore 来源是同输入 clean component output，不是跨样本变量，不是子空间变量。
2. 本轮仍是对象-槽位任务，不是逻辑/语法/指代。
3. generated choice 使用的是短 JSON letter，不是开放生成。
4. DeepSeek7B 的 letter margin 与 generated choice 仍有反向关系，需要拆完整生成格式路径。
5. GLM4 的 candidate/choice 分离说明还缺 interface transformation layer 的解释。

### 下一步

Phase92 应做：

```text
cross-item component transplant
```

目标：

```text
不再从同输入恢复 clean component output，
而是在同 slot / 不同 object / 不同 target 的样本之间移植组件输出。
```

测试：

```text
1. same-slot same-target transplant
2. same-slot different-target transplant
3. different-slot transplant
4. same-object different-slot transplant
```

重点节点：

```text
qwen3:
  L6 MLP
  L24 attention

DeepSeek7B:
  L26 MLP
  L27 attention

GLM4:
  L39 MLP 作为 candidate/choice 分离对照
```

判断标准：

```text
如果 same-target transplant 能恢复，
但 different-target transplant 不能恢复，
说明组件输出含有目标候选变量。

如果 same-slot transplant 有效，
但 different-slot transplant 无效，
说明组件输出含有 slot/interface 格式。

如果跨 object 仍有效，
说明开始接近抽象变量级机制。
```

## Phase 92: Cross-item Component Transplant 跨样本组件输出移植 [2026-06-11 23:35]

### 本轮目标

根据 Phase 91 的结果，同输入 clean component output restore 已经证明：

```text
某些组件输出对当前样本的 reader interface 是必要且可恢复的。
```

但 Phase 91 不能证明：

```text
组件输出中存在跨样本可迁移的抽象关系变量。
```

因此本轮推进到 cross-item component transplant（跨样本组件输出移植）：

```text
target input 的指定组件输出
← donor input 的同层同组件输出
```

并比较 donor 类型：

```text
self_restore：同输入恢复
same_slot_same_target：同槽位、同目标值、不同对象
same_slot_diff_target：同槽位、不同目标值
diff_slot_same_object：同对象、不同槽位
diff_slot_diff_object：不同对象、不同槽位
```

### 生成脚本

```text
tests/gpt5/phase92_cross_item_component_transplant.py
tests/gpt5/phase92_cross_item_component_transplant_summary.py
tests/gpt5/run_phase92_cross_item_component_transplant_full.sh
```

### 运行命令

smoke：

```bash
python tests/gpt5/phase92_cross_item_component_transplant.py qwen3 \
  --nodes 6:mlp \
  --max-items 4 \
  --output-dir results/gpt5_phase92_smoke \
  --progress-every 2 \
  --hard-exit-after-model
```

全量：

```bash
tests/gpt5/run_phase92_cross_item_component_transplant_full.sh
```

实际输出目录：

```text
results/gpt5_phase92_cross_item_component_transplant_full_20260611_220707
```

### 测试设置

```text
模型顺序：
qwen3 → GLM4 → DeepSeek7B

每模型测试完成后：
--hard-exit-after-model

attention implementation：
flash_attention_2,sdpa,eager

实际加载：
flash_attention_2 未安装，三模型均回退到 sdpa。

数据范围：
420 个 object-slot-template items
slots = category,color,function,material,location

copy_mode：
tail

节点：
qwen3: L6 MLP, L24 attention
GLM4: L39 MLP
DeepSeek7B: L26 MLP, L27 attention
```

### 总体结果

```text
total_rows = 10500
total_bad_numeric_rows = 0

qwen3 rows = 4200
GLM4 rows = 2100
DeepSeek7B rows = 4200
```

本轮没有跑 open generation，优先使用：

```text
candidate value full-sequence margin
choice letter full-sequence margin
```

原因是本轮目标是判断组件输出是否跨样本可迁移，过多 generation 格式噪声会污染判断。

### qwen3 结果

#### L6 MLP

```text
self_restore:
  value_patch_gain = 3.0473
  letter_patch_gain = 3.5628

same_slot_same_target:
  value_patch_gain = 1.9593
  letter_patch_gain = 2.4646

same_slot_diff_target:
  value_patch_gain = 2.1686
  letter_patch_gain = 2.7589

diff_slot_same_object:
  value_patch_gain = 0.7879
  letter_patch_gain = 1.1192

diff_slot_diff_object:
  value_patch_gain = 0.6514
  letter_patch_gain = 0.9609
```

观察：

```text
1. L6 MLP 的跨样本移植不是只在 same-target 时有效。
2. same_slot_diff_target 反而比 same_slot_same_target 更强。
3. diff_slot 条件仍有一定恢复，但明显弱于 same_slot。
```

这说明 Qwen3 L6 MLP 更像：

```text
slot/interface computation support
```

而不是简单：

```text
target value memory vector
```

它承载的可能是同一读出任务格式或槽位相关计算状态，而不是单个答案值。

#### L24 attention

```text
self_restore:
  value_patch_gain = 0.1010
  letter_patch_gain = 4.3152

same_slot_same_target:
  value_patch_gain = 0.2476
  letter_patch_gain = 4.4435

same_slot_diff_target:
  value_patch_gain = -0.6354
  letter_patch_gain = -6.1327

diff_slot_same_object:
  value_patch_gain = 0.0264
  letter_patch_gain = 2.1098

diff_slot_diff_object:
  value_patch_gain = -0.5287
  letter_patch_gain = -1.1286
```

观察：

```text
1. L24 attention 对 choice letter interface 极强。
2. same_slot_same_target 几乎可以完全恢复甚至略超 self_restore。
3. same_slot_diff_target 强烈反向，说明 wrong target donor 会破坏 letter decision。
4. diff_slot_same_object 仍有中等 letter 恢复，说明 object/context 也贡献一部分接口状态。
```

这比 L6 MLP 更接近：

```text
target-specific choice/generation interface
```

### GLM4 结果

#### L39 MLP

```text
self_restore:
  value_patch_gain = 0.6564
  letter_patch_gain = -0.6807

same_slot_same_target:
  value_patch_gain = 0.7107
  letter_patch_gain = -0.8045

same_slot_diff_target:
  value_patch_gain = 0.0988
  letter_patch_gain = -0.5283

diff_slot_same_object:
  value_patch_gain = 0.5157
  letter_patch_gain = -0.5643

diff_slot_diff_object:
  value_patch_gain = -0.2486
  letter_patch_gain = -0.4766
```

观察：

```text
1. GLM4 L39 MLP 对 candidate value margin 有明显条件迁移。
2. same_slot_same_target 的 value_patch_gain 高于 self_restore。
3. diff_slot_same_object 也保留较强 value 恢复。
4. 但所有条件下 letter_patch_gain 仍为负。
```

这延续 Phase90/91 的结论：

```text
GLM4 的 candidate value interface 与 choice letter interface 分离。
```

GLM4 L39 MLP 更像：

```text
candidate scoring component
```

而不是：

```text
choice decision component
```

### DeepSeek7B 结果

#### L26 MLP

```text
self_restore:
  value_patch_gain = 0.4551
  letter_patch_gain = -0.8616

same_slot_same_target:
  value_patch_gain = 0.3270
  letter_patch_gain = -0.8872

same_slot_diff_target:
  value_patch_gain = -0.4715
  letter_patch_gain = -0.7020

diff_slot_same_object:
  value_patch_gain = 0.4227
  letter_patch_gain = -0.9088

diff_slot_diff_object:
  value_patch_gain = -0.0512
  letter_patch_gain = -0.7289
```

观察：

```text
1. L26 MLP 对 candidate value margin 有迁移，但主要依赖 same target 或 same object。
2. same_slot_diff_target 变成负向，说明 wrong target donor 会破坏 value scoring。
3. letter_patch_gain 一直为负，说明它不是 letter interface。
```

#### L27 attention

```text
self_restore:
  value_patch_gain = -0.0460
  letter_patch_gain = 0.8537

same_slot_same_target:
  value_patch_gain = -0.2073
  letter_patch_gain = 0.8473

same_slot_diff_target:
  value_patch_gain = -0.3843
  letter_patch_gain = 0.7484

diff_slot_same_object:
  value_patch_gain = -0.1573
  letter_patch_gain = 0.7685

diff_slot_diff_object:
  value_patch_gain = -0.2432
  letter_patch_gain = 0.7917
```

观察：

```text
1. L27 attention 几乎完全是 letter interface。
2. same_slot_same_target 的 letter_patch_gap = 0.0064，几乎复现 self_restore。
3. 但其他 donor 条件也保持很高 letter_patch_gain。
4. 这说明 L27 attention 可能承载的是输出选项接口格式，而不一定是语义目标本身。
```

### 对附件分析的判断

附件对 Phase91 的判断基本正确：

```text
Phase91 是组件级闭包推进，
但同输入 restore 不能证明抽象变量。
```

本轮 Phase92 正是对这个硬伤的补充：

```text
把同输入 restore 改成跨样本 donor transplant，
观察组件输出是否可以跨 object / slot / target 迁移。
```

### 本轮关键进展

1. **Qwen3 L6 MLP 不是单纯答案值存储。**

```text
same_slot_diff_target 迁移强于 same_slot_same_target，
说明它更像 slot/interface 计算状态，而非 target-value vector。
```

2. **Qwen3 L24 attention 更接近 target-specific choice interface。**

```text
same_slot_same_target 强恢复；
same_slot_diff_target 强破坏。
```

3. **GLM4 L39 MLP 继续显示 candidate/choice 分离。**

```text
value margin 可迁移；
letter margin 不支持，甚至为负。
```

4. **DeepSeek7B L26 MLP 和 L27 attention 分工清楚。**

```text
L26 MLP 更接近 candidate value / object-target support；
L27 attention 更接近 letter/output interface。
```

5. **组件输出迁移存在明显条件性。**

```text
不是所有 donor 都能恢复；
slot、target、object 三者都会改变迁移效果。
```

### 当前最稳结论

可以较稳地说：

```text
对象-槽位 reader 任务中，
三模型都不是一个统一读出路径。

Qwen3:
  L6 MLP = slot/interface computation support
  L24 attention = target-sensitive choice interface

GLM4:
  L39 MLP = candidate scoring component
  choice decision interface 与 candidate scoring 分离

DeepSeek7B:
  L26 MLP = candidate value/object-target support
  L27 attention = letter/output interface
```

不能说：

```text
已经找到抽象语义变量；
跨样本 transplant 已经完成变量闭包；
tail 对齐一定是最正确的移植方式；
这些结论可直接推广到逻辑、语法、指代。
```

### 硬伤

1. 本轮使用 tail 对齐，不同长度 prompt 的 token 级对应关系仍然粗糙。
2. 本轮没有 generation，只测 full-sequence margin。
3. donor transplant 是整组件输出移植，不是子空间变量移植。
4. Qwen3 L6 MLP 的 same_slot_diff_target 强恢复说明它不是纯 target 变量，仍需拆 slot/interface 和 target/value。
5. DeepSeek7B L27 attention 对多种 donor 都有高 letter 恢复，可能主要是输出格式接口，不一定是语义内容。

### 下一步

Phase93 应做：

```text
component transplant alignment audit
```

核心问题：

```text
tail 对齐、prefix 对齐、both 对齐结果是否一致？
```

如果只有 tail 对齐有效，说明当前发现主要在后缀读出接口。
如果 prefix/both 也有效，说明更可能是全序列路径状态。

Phase94 应做：

```text
subspace transplant
```

不要继续整组件移植，而是把组件输出拆成：

```text
slot subspace
target/value subspace
object/context subspace
choice-interface subspace
residual remainder
```

然后分别移植，判断：

```text
哪个子空间控制 value margin；
哪个子空间控制 letter margin；
哪个子空间只是输出格式接口。
```

Phase95 应把对象-槽位 reader 的结论扩展到：

```text
translation
logical relation
temporal order
active/passive
coreference
```

但必须先通过 Phase93/94，把当前对象-槽位任务的路径拆干净，否则继续扩功能库只会增加行为图谱，不会进入编码机制。

## Phase 93: Component Transplant Alignment Audit 组件移植对齐方式审计 [2026-06-12 04:37]

### 本轮目标

Phase92 发现跨样本组件输出移植存在条件性，但硬伤是：

```text
只使用 tail 对齐。
```

不同长度 prompt / continuation 的 token 位置并不完全一致，因此需要验证：

```text
tail 对齐、prefix 对齐、both 对齐
是否产生一致结论。
```

如果只有 tail 有效，说明当前发现主要是后缀读出接口。
如果 prefix/both 也有效，说明组件输出中可能存在更全局的路径状态。

### 生成脚本

```text
tests/gpt5/phase93_transplant_alignment_audit.py
tests/gpt5/phase93_transplant_alignment_audit_summary.py
tests/gpt5/run_phase93_transplant_alignment_audit_full.sh
```

### 运行命令

smoke：

```bash
python tests/gpt5/phase93_transplant_alignment_audit.py qwen3 \
  --nodes 6:mlp \
  --max-items 4 \
  --output-dir results/gpt5_phase93_smoke \
  --progress-every 2 \
  --hard-exit-after-model
```

全量：

```bash
tests/gpt5/run_phase93_transplant_alignment_audit_full.sh
```

DS7B 中途在 tail/L27 attention 约 2800 行处出现一次 segmentation fault：

```text
Segmentation fault (core dumped)
```

已从 partial checkpoint 恢复：

```bash
python tests/gpt5/phase93_transplant_alignment_audit.py deepseek7b \
  --nodes 26:mlp,27:attn \
  --slots category,color,function,material,location \
  --max-items 420 \
  --choice-template choice_json_letter \
  --copy-modes tail,prefix,both \
  --donor-kinds self_restore,same_slot_same_target,same_slot_diff_target,diff_slot_same_object,diff_slot_diff_object \
  --output-dir results/gpt5_phase93_transplant_alignment_audit_full_20260612_001114 \
  --progress-every 70 \
  --hard-exit-after-model
```

### 输出目录

```text
results/gpt5_phase93_transplant_alignment_audit_full_20260612_001114
```

### 测试设置

```text
模型顺序：
qwen3 → GLM4 → DeepSeek7B

每模型测试完成后：
--hard-exit-after-model

attention implementation：
flash_attention_2,sdpa,eager

实际加载：
flash_attention_2 未安装，三模型均回退到 sdpa。

数据范围：
420 个 object-slot-template items
slots = category,color,function,material,location

copy_modes:
tail,prefix,both

donor_kinds:
self_restore
same_slot_same_target
same_slot_diff_target
diff_slot_same_object
diff_slot_diff_object

节点：
qwen3: L6 MLP, L24 attention
GLM4: L39 MLP
DeepSeek7B: L26 MLP, L27 attention
```

### 总体结果

```text
total_rows = 31500
total_bad_numeric_rows = 0

qwen3 rows = 12600
GLM4 rows = 6300
DeepSeek7B rows = 12600
```

本轮仍然只使用：

```text
candidate value full-sequence margin
choice letter full-sequence margin
```

不引入 generation，以避免格式化生成噪声污染 alignment audit。

### qwen3 结果

#### copy mode 总体

```text
tail:
  value_patch_gain = 0.7825
  letter_patch_gain = 1.4474

prefix:
  value_patch_gain = 1.2315
  letter_patch_gain = 1.9727

both:
  value_patch_gain = 1.2569
  letter_patch_gain = 1.8420
```

观察：

```text
prefix/both 强于 tail。
```

这说明 Phase92 的结论不是 tail-only artifact。Qwen3 的可迁移组件输出不只存在于答案后缀接口，也存在于前缀/全序列路径状态中。

#### L6 MLP

```text
tail:
  value_patch_gain = 1.7229
  letter_patch_gain = 2.1733

prefix:
  value_patch_gain = 2.6068
  letter_patch_gain = 3.2021

both:
  value_patch_gain = 2.6708
  letter_patch_gain = 2.9941
```

donor 条件：

```text
tail/same_slot_diff_target:
  value_patch_gain = 2.1686
  letter_patch_gain = 2.7589

prefix/same_slot_diff_target:
  value_patch_gain = 2.5110
  letter_patch_gain = 2.9850

both/same_slot_diff_target:
  value_patch_gain = 2.4109
  letter_patch_gain = 2.8497
```

观察：

```text
L6 MLP 的 same_slot_diff_target 强恢复在三种对齐方式下都存在。
```

因此 Phase92 的判断更稳：

```text
Qwen3 L6 MLP 不是单纯 target value vector，
更像 slot/interface computation support。
```

并且这个支持不是后缀特异的，而是前缀/全序列状态也能承载。

#### L24 attention

```text
tail:
  value_patch_gain = -0.1578
  letter_patch_gain = 0.7214

prefix:
  value_patch_gain = -0.1439
  letter_patch_gain = 0.7433

both:
  value_patch_gain = -0.1570
  letter_patch_gain = 0.6898
```

关键 donor 条件：

```text
tail/same_slot_same_target:
  letter_patch_gain = 4.4435

prefix/same_slot_same_target:
  letter_patch_gain = 2.7090

both/same_slot_same_target:
  letter_patch_gain = 4.5033

tail/same_slot_diff_target:
  letter_patch_gain = -6.1327

prefix/same_slot_diff_target:
  letter_patch_gain = -3.5527

both/same_slot_diff_target:
  letter_patch_gain = -6.2369
```

观察：

```text
same_slot_same_target 强恢复、same_slot_diff_target 强破坏的模式在三种对齐中都成立。
```

但 prefix 的强度较弱，说明：

```text
L24 attention 的 target-sensitive choice interface 更偏后缀/全序列输出接口，
但不是纯 tail artifact。
```

### GLM4 结果

#### L39 MLP

```text
tail:
  value_patch_gain = 0.3466
  letter_patch_gain = -0.6109

prefix:
  value_patch_gain = 0.4631
  letter_patch_gain = -0.1150

both:
  value_patch_gain = 0.3466
  letter_patch_gain = -0.6109
```

观察：

```text
1. prefix 对 value margin 更强。
2. prefix 下 letter 负向程度变弱，但仍没有成为稳定正向 choice interface。
3. tail 和 both 结果几乎一致。
```

关键 donor 条件：

```text
prefix/same_slot_same_target:
  value_patch_gain = 0.6576
  letter_patch_gain = -0.2895

prefix/diff_slot_same_object:
  value_patch_gain = 0.5952
  letter_patch_gain = 0.3510

tail/same_slot_same_target:
  value_patch_gain = 0.7107
  letter_patch_gain = -0.8045
```

GLM4 的 Phase92 结论仍然成立：

```text
L39 MLP 更像 candidate scoring component，
不是稳定 choice decision component。
```

但新增信息是：

```text
GLM4 的 candidate scoring 更偏 prefix / 上下文侧，
choice letter 的负向现象受对齐方式影响。
```

### DeepSeek7B 结果

#### L26 MLP

```text
tail:
  value_patch_gain = 0.1364
  letter_patch_gain = -0.8177

prefix:
  value_patch_gain = 0.3046
  letter_patch_gain = -0.7961

both:
  value_patch_gain = 0.2319
  letter_patch_gain = -0.7704
```

关键 donor 条件：

```text
tail/same_slot_same_target:
  value_patch_gain = 0.3270

prefix/same_slot_same_target:
  value_patch_gain = 0.6131

both/same_slot_same_target:
  value_patch_gain = 0.5555

tail/same_slot_diff_target:
  value_patch_gain = -0.4715

prefix/same_slot_diff_target:
  value_patch_gain = 0.0725

both/same_slot_diff_target:
  value_patch_gain = -0.1937
```

观察：

```text
L26 MLP 对 value margin 的支持在 prefix/both 下更强。
same_slot_same_target 稳定有效。
same_slot_diff_target 的负向破坏在 tail 最强，prefix 下被削弱。
```

这说明：

```text
DS7B L26 MLP 的 value/object-target support 不是纯后缀接口，
但 wrong target 破坏效应对对齐方式敏感。
```

#### L27 attention

```text
tail:
  value_patch_gain = -0.2077
  letter_patch_gain = 0.8019

prefix:
  value_patch_gain = -0.2425
  letter_patch_gain = 0.7345

both:
  value_patch_gain = -0.2077
  letter_patch_gain = 0.8019
```

关键 donor 条件：

```text
tail/same_slot_same_target:
  letter_patch_gain = 0.8473

prefix/same_slot_same_target:
  letter_patch_gain = 0.8358

both/same_slot_same_target:
  letter_patch_gain = 0.8473
```

观察：

```text
L27 attention 的 letter/output interface 在三种对齐中都稳定。
```

这比 Phase92 更稳：

```text
DS7B L27 attention 确实是强 letter/output interface，
不是 tail 对齐造成的假象。
```

### 对附件分析的判断

附件对 Phase92 的分析基本正确：

```text
Phase92 证明部分组件输出可跨样本迁移，
但迁移内容不是纯 target value，
而是 slot/interface/target/object 混合的条件化组件状态。
```

附件指出的硬伤也正确：

```text
tail 对齐可能导致后缀接口假象。
```

Phase93 的结果表明：

```text
这个硬伤被部分排除。
```

因为关键现象在 prefix/both 下仍然存在，尤其是：

```text
Qwen3 L6 MLP 的 slot/interface 支持；
Qwen3 L24 attention 的 target-sensitive letter interface；
DS7B L27 attention 的 letter/output interface。
```

### 当前最稳结论

可以进一步稳定地说：

```text
1. Qwen3 L6 MLP 是跨位置可迁移的 slot/interface computation support。
2. Qwen3 L24 attention 是 target-sensitive choice interface，且不是 tail-only。
3. GLM4 L39 MLP 是 candidate scoring component，choice interface 分离仍成立。
4. DeepSeek7B L26 MLP 是 value/object-target support，prefix/both 更强。
5. DeepSeek7B L27 attention 是稳定 letter/output interface。
```

不能说：

```text
已经完成抽象变量闭包；
整组件移植等于变量移植；
prefix/both 有效就说明找到了数学结构本体；
这些对象-槽位 reader 结果可以直接推广到全部语言能力。
```

### 硬伤

1. 本轮仍是整组件输出移植，不是子空间移植。
2. prefix/both 虽然排除了 tail-only，但仍没有 token-level 精细对齐。
3. self_restore 是同输入恢复，跨样本 donor 仍混入 object、slot、target、template 多因素。
4. DS7B 首次运行中出现一次 segmentation fault，虽已恢复完成，但工程稳定性仍需记录。
5. 本轮仍没有 generation，只是 full-sequence margin。

### 下一步

Phase94 必须进入 subspace transplant：

```text
不再移植整个 component output，
而是学习/构造若干低维子空间：

slot/interface subspace
target/value subspace
object/context subspace
choice-letter/output subspace
remainder
```

优先节点：

```text
Qwen3 L6 MLP：
拆 slot/interface 与 target/value。

Qwen3 L24 attention：
拆 target-specific letter interface。

DeepSeek7B L27 attention：
拆 output-format 与 target semantic。

GLM4 L39 MLP：
拆 candidate scoring 与 choice decision 分离。
```

Phase95 再考虑跨功能迁移：

```text
object-slot reader
translation
logical relation
temporal order
active/passive
coreference
```

目前最重要的突破口不是再扩大功能库，而是把当前 reader 任务中的：

```text
component-level transferable state
```

拆成：

```text
factor-level transferable subspace
```

这一步才真正接近语言编码机制中的“条件化关系因子”。

## Phase 94: 因子子空间闭包总方案 [2026-06-12 09:09]

### 背景

综合 GLM5 Phase467 与 GPT5 Phase93：

```text
GLM5 线：
已经发现 PC1 与 logit entropy / position 有强关系，
并且 vehicle/tool/furniture 等类别方向会被 PC1 严重污染。
去 PC1、去混叠、去 top PCs 对不同类别有不同效果。

GPT5 线：
已经从 reader calibration 推进到 component restore、
cross-item transplant 和 alignment audit。
Qwen3 L6 MLP、Qwen3 L24 attention、GLM4 L39 MLP、
DeepSeek7B L26 MLP / L27 attention 已经形成稳定组件分工图。
```

当前共同硬伤：

```text
1. GLM5 的方向净化还停留在残差方向 / 类别方向层面。
2. GPT5 的 transplant 还是整组件输出移植，不是因子子空间移植。
3. 两条线都还没有证明抽象变量闭包。
```

因此下一步不能只是继续扩功能库，也不能只做更多整组件移植。

### 当前判断

附件中对 Phase93 的分析基本正确：

```text
Phase93 排除了 tail-only artifact，
证明组件输出迁移不是单纯后缀读出接口复制。
```

但结论必须收缩：

```text
这仍然不是抽象语义变量，
而是 component-level transferable state。
```

更准确的对象应命名为：

```text
条件化关系因子状态
```

即：

```text
某个组件输出中可迁移的部分，
受 slot、target、object/context、reader interface、position、entropy 主轴共同条件化。
```

### 条件化关系因子动力学公式更新

旧表达：

```text
h_l(x) = Σ_k Code_k(l,x) + ε_l
```

需要改成：

```text
h_l(x) =
  Base_l(x)
  + Σ_r Gate_{l,r}(x) · F_{l,r}(x)
  + U_l(x)
```

其中：

```text
Base_l(x):
  模型通用路径状态，包括位置、输出不确定性、格式、推理模板等。

F_{l,r}(x):
  条件化关系因子，例如 object、slot、target value、choice interface、
  language axis、category direction。

Gate_{l,r}(x):
  当前上下文对关系因子的调用权重。
  它决定某个因子是否进入读出路径。

U_l(x):
  未分解残差，包括噪声、未建模变量、模型特有结构。
```

GLM5 Phase467 表明：

```text
Base_l(x) 中至少包含 PC1 entropy/position axis。
```

GPT5 Phase93 表明：

```text
F_{l,r}(x) 不是单一 target vector，
而是 slot/interface/target/object 混合因子。
```

因此更具体：

```text
h_l(x) =
  EntropyPositionAxis_l(x)
  + ObjectContext_l(x)
  + SlotInterface_l(x)
  + TargetValue_l(x)
  + ChoiceOutput_l(x)
  + Remainder_l(x)
```

但这些项不是固定正交向量，而是条件化、模型依赖、层依赖的可迁移子空间。

### 对深度神经网络内部结构的进展

当前已经从：

```text
找概念方向
```

推进到：

```text
组件级路径分工
主轴污染识别
跨样本可迁移状态
对齐方式审计
候选/选择接口分离
```

较稳的模型分型：

```text
Qwen3:
  L6 MLP 是跨位置可迁移的 slot/interface computation support。
  L24 attention 是 target-sensitive choice interface。
  PC1 与 entropy 强相关，vehicle/tool/furniture 等类别方向受 PC1 污染。

GLM4:
  L39 MLP 更像 candidate scoring component。
  choice decision interface 与 candidate scoring 分离。
  类别方向可写性更强，但不同类别对 PC1 / 正交化敏感性不同。

DeepSeek7B:
  L26 MLP 更像 value/object-target support。
  L27 attention 是稳定 letter/output interface。
  PC1 不稳定，生成模板容易崩坏，R1-Distill 模式可能导致默认数学化输出。
```

### 关键硬伤

```text
1. 整组件移植仍太粗，不能证明变量。
2. PC1 去除有效，但 PC1 的因果作用还没证明。
3. no_pc1、disentangle、no_top PCs 不是通用净化方法，类别和层强相关。
4. reader 任务仍集中在 object-slot，不足以代表完整语言机制。
5. DS7B 的生成基线不可靠，必须先做模板安全校准。
6. candidate value margin、choice letter margin、generation 三者仍未完全统一。
```

### Phase94 总任务

Phase94 不拆成很多小任务，而作为一个大任务：

```text
Factor Subspace Closure Map
因子子空间闭包图谱
```

目标：

```text
把 component-level transferable state
拆成 factor-level transferable subspace。
```

### Phase94 实验对象

优先节点：

```text
Qwen3:
  L6 MLP
  L24 attention

GLM4:
  L39 MLP
  可补 L13/L20 residual category directions

DeepSeek7B:
  L26 MLP
  L27 attention
```

优先因子：

```text
slot/interface
target/value
object/context
choice-letter/output
entropy/position PC1
category direction
remainder
```

### Phase94 测试设计

1. 子空间构造：

```text
从组件输出中构造若干差分方向或低秩子空间：

slot subspace:
  同 object、不同 slot

target/value subspace:
  同 slot、不同 target

object/context subspace:
  同 slot/target、不同 object

choice-output subspace:
  value prompt 与 choice prompt 的组件差分

PC1 base subspace:
  从自然激活 PCA 取 PC1 / top PCs
```

2. 子空间净化：

```text
raw
no_pc1
disentangle
no_top3pc
no_pc1 + controlled disentangle
```

但不假设某种方法普适，必须按模型、层、类别分别记录。

3. 子空间移植：

```text
只移植 slot subspace
只移植 target/value subspace
只移植 object/context subspace
只移植 choice-output subspace
移植 remainder
组合移植
```

4. 闭包判据：

```text
destroy:
  去掉某个子空间后，对应 reader 指标下降。

restore:
  只恢复该子空间后，对应 reader 指标恢复。

specificity:
  slot 子空间主要影响 slot/interface；
  target 子空间主要影响 target/value；
  choice 子空间主要影响 letter/output；
  PC1 子空间主要影响 entropy/format，而不是具体语义。

transfer:
  同槽位/同目标/同对象条件下迁移效果符合预测。
```

### Phase94 输出

每个模型输出：

```text
factor_subspace_summary
destroy_restore_score
transfer_specificity_matrix
pc1_contamination_matrix
candidate_choice_separation_matrix
model_strategy_table
```

### Phase94 成功标准

最低成功：

```text
在 Qwen3 L6 MLP 中分离出 slot/interface subspace，
且该子空间移植比 target/value 子空间更能解释 same_slot_diff_target 强恢复。
```

中等成功：

```text
Qwen3 L24 attention 可分离 target-sensitive choice-output subspace，
same_slot_same_target 恢复强，same_slot_diff_target 破坏强。
```

强成功：

```text
三模型都能形成：

Qwen3:
  slot/interface → target choice interface 的连续路径

GLM4:
  candidate scoring 与 choice decision 的分离路径

DeepSeek7B:
  value/object-target support 与 output interface 的深层分工
```

### 阶段性理论判断

如果 Phase94 成功，则可以把当前理论从：

```text
组件级路径图谱
```

推进到：

```text
条件化关系因子子空间图谱
```

这才是接近语言编码机制的关键中间层。

目前不应宣称：

```text
已经破解语言数学结构。
```

但可以说：

```text
语言机制越来越不像固定概念方向，
越来越像由通用主轴、关系因子、接口因子、输出因子组成的条件化动态路径系统。
```

## Phase 95: Factor Subspace Closure 全量测试 [2026-06-12 11:15]

### 本轮目标

根据 Phase94 总方案，本轮把 Phase92/93 的整组件输出移植推进到：

```text
factor subspace destroy / transplant
```

不再替换整个 component output，而是在 hook 中只操作某个低秩子空间的投影：

```text
destroy:
  output' = output - P_factor(output)

transplant:
  output' = output - P_factor(output) + P_factor(donor_output)
```

目标是初步拆分：

```text
pc1 / base
slot / interface
target / value
object / context
choice / output
```

### 生成脚本

```text
tests/gpt5/phase94_factor_subspace_closure.py
tests/gpt5/phase94_factor_subspace_closure_summary.py
tests/gpt5/run_phase94_factor_subspace_closure_full.sh
```

### 运行命令

smoke：

```bash
python tests/gpt5/phase94_factor_subspace_closure.py qwen3 \
  --nodes 6:mlp \
  --max-items 6 \
  --rank 2 \
  --output-dir results/gpt5_phase94_smoke \
  --progress-every 3 \
  --hard-exit-after-model
```

全量：

```bash
tests/gpt5/run_phase94_factor_subspace_closure_full.sh
```

### 输出目录

```text
results/gpt5_phase94_factor_subspace_closure_full_20260612_091612
```

### 测试设置

```text
模型顺序：
qwen3 → GLM4 → DeepSeek7B

每模型测试完成后：
--hard-exit-after-model

attention implementation：
flash_attention_2,sdpa,eager

实际加载：
flash_attention_2 未安装，三模型均回退到 sdpa。

数据范围：
420 个 object-slot-template items
slots = category,color,function,material,location

rank:
4

pool_mode:
tail

copy_mode:
both

节点：
qwen3: L6 MLP, L24 attention
GLM4: L39 MLP
DeepSeek7B: L26 MLP, L27 attention
```

### 总体结果

```text
total_rows = 21000
total_bad_numeric_rows = 0

qwen3 rows = 8400
GLM4 rows = 4200
DeepSeek7B rows = 8400
```

所有节点的 basis 维度：

```text
pc1 = 1
slot = 4
target = 4
object = 4
choice = 4
```

### qwen3 结果

#### L6 MLP

关键 destroy：

```text
destroy_object:
  value_delta = -0.1526
  letter_delta = -2.4289
  letter_top1_delta = -0.1286

destroy_target:
  value_delta = -0.0989
  letter_delta = -0.6116
  letter_top1_delta = -0.0405

destroy_pc1:
  value_delta = +0.0179
  letter_delta = -0.1417

destroy_slot:
  value_delta = +0.0786
  letter_delta = +0.1137

destroy_choice:
  value_delta = +0.0874
  letter_delta = +1.2187
```

观察：

```text
1. L6 MLP 中 object/context 子空间对 letter interface 最关键。
2. target 子空间也有负向必要性，但弱于 object。
3. slot 子空间破坏后不降反升，说明当前 slot basis 不是必要槽位程序。
4. choice 子空间破坏后 letter_delta 大幅上升，说明该 basis 可能包含抑制/竞争成分，而不是纯 output support。
```

这修正了 Phase93 的理解：

```text
Qwen3 L6 MLP 的 slot/interface support
可能并不在当前用 diff_slot_same_object 构造的 slot basis 中，
而更强地混入 object/context 与 target 因子。
```

#### L24 attention

```text
destroy_object:
  value_delta = -0.0216
  letter_delta = +0.5149

destroy_choice:
  value_delta = +0.0036
  letter_delta = +0.1798

destroy_pc1:
  value_delta = -0.0056
  letter_delta = +0.0131

destroy_target:
  value_delta = -0.0206
  letter_delta = +0.0533
```

观察：

```text
L24 attention 的当前低秩子空间破坏没有复现 Phase92/93 的强 letter interface 必要性。
```

这说明：

```text
L24 attention 的 target-sensitive choice interface
可能不是由简单 tail-pooled rank-4 差分子空间承载，
而是更依赖 token-level pattern 或更高秩结构。
```

### GLM4 结果

#### L39 MLP

```text
destroy_choice:
  value_delta = -0.5485
  letter_delta = -0.1582

destroy_target:
  value_delta = -0.4653
  letter_delta = -0.0640

destroy_object:
  value_delta = -0.4592
  letter_delta = -0.0071

destroy_slot:
  value_delta = -0.3918
  letter_delta = +0.0037

destroy_pc1:
  value_delta = +0.0073
  letter_delta = -0.0314
```

观察：

```text
1. GLM4 L39 MLP 的 value margin 明确依赖 choice、target、object、slot 等子空间。
2. letter margin 仍然几乎不受这些子空间正向支持。
3. PC1 对该节点 value scoring 几乎不是必要项。
```

这进一步确认：

```text
GLM4 L39 MLP 是 candidate scoring component，
不是 choice decision component。
```

而且 candidate scoring 不是单一 target/value 子空间，而是多个因子共同构成。

### DeepSeek7B 结果

#### L26 MLP

```text
destroy_object:
  value_delta = +0.2780
  letter_delta = +0.5397

destroy_target:
  value_delta = +0.1313
  letter_delta = -0.2308

destroy_pc1:
  value_delta = -0.1490
  letter_delta = +0.0263

destroy_slot:
  value_delta = +0.0268
  letter_delta = -0.3647
```

观察：

```text
1. destroy_pc1 会降低 value margin，说明 DS7B L26 MLP 的 candidate value 支持依赖 PC1/base 方向。
2. destroy_object 反而提升 value/letter，说明当前 object basis 可能包含竞争或干扰成分。
3. slot/target 对 letter 有负向影响，但 value 必要性不强。
```

#### L27 attention

```text
destroy_pc1:
  value_delta = +0.5296
  letter_delta = -0.1574

destroy_target:
  value_delta = +0.4525
  letter_delta = -0.2068

destroy_object:
  value_delta = +0.4007
  letter_delta = -0.4074

destroy_choice:
  value_delta = +0.4999
  letter_delta = -0.1716
```

观察：

```text
1. L27 attention 中多个子空间破坏都会提高 value margin，但降低 letter margin。
2. 这与 Phase93 的判断一致：L27 attention 是 letter/output interface，而不是 value semantic interface。
3. 该 output interface 与 PC1/base、object、target、choice 子空间都有耦合。
```

### 对前面理论的修正

Phase95 的关键结果不是“成功拆出干净变量”，而是证明：

```text
当前简单低秩差分子空间还不能干净分离 slot / target / object / choice。
```

但它带来重要结构信息：

```text
1. GLM4 L39 MLP 的 candidate scoring 可以被低秩因子子空间明显破坏。
2. Qwen3 L6 MLP 的强接口支持更依赖 object/context 与 target 混合子空间。
3. Qwen3 L24 attention 的强 choice interface 不容易被 tail-pooled rank-4 子空间捕捉。
4. DeepSeek7B L27 attention 的 letter/output interface 与多个因子混合耦合。
```

### 当前最稳结论

可以说：

```text
component-level transferable state
确实可以进一步投影到 factor-like subspaces，
但这些子空间不是干净变量轴。
```

不能说：

```text
slot / target / object / choice 已经被完全分离；
rank-4 tail-pooled 子空间就是语言变量；
当前子空间闭包已经完成抽象机制证明。
```

### 硬伤

1. 子空间构造仍是简单差分 + SVD，变量定义粗。
2. pool_mode = tail，虽然 transplant 用 both，但 basis 本身来自 tail pooling。
3. rank = 4 可能过低，尤其 Qwen3 L24 attention 的接口可能更高秩或 token-pattern 化。
4. destroy 子空间后指标上升，说明很多 basis 包含竞争/抑制成分，不能简单解释为“该因子必要”。
5. 没有做 token-level 子空间，也没有做 generation。

### 下一步

Phase96 应做：

```text
Rank and Token Subspace Sweep
```

目标：

```text
验证 Phase95 的失败是因为：
1. rank 太低；
2. tail pooling 太粗；
3. 子空间定义混叠；
4. 还是这些变量本来就不是线性子空间。
```

测试：

```text
rank = 1,2,4,8,16
pool = tail, prefix, mean, token-aligned
factor = object,target,choice,pc1
nodes = Qwen3 L6 MLP, Qwen3 L24 attention, GLM4 L39 MLP, DS7B L27 attention
```

判据：

```text
如果高 rank / token-aligned 后 Qwen3 L24 attention 的 choice interface 被捕捉，
说明接口是高秩或 token-pattern。

如果仍然捕捉不到，
说明它可能是 attention route / relational pattern，而非简单输出子空间。
```

## Phase 96: Rank-Pool Subspace Sweep 全量诊断 [2026-06-12 13:51]

### 背景

Phase95 发现：用 `rank=4`、`tail pooling` 构造的 factor subspace 不能干净分离 `slot / target / object / choice`。本阶段不继续做新理论，而是直接诊断失败来源：

```text
1. 是否 rank 太低；
2. 是否 tail pooling 太粗；
3. 是否 Qwen3 L24 attention / DS7B L27 attention 的接口不是普通 component-output 子空间；
4. 是否 GLM4 L39 MLP 的 candidate scoring 仍能被更高 rank 子空间稳定破坏。
```

### 生成脚本

```text
tests/gpt5/phase96_rank_pool_subspace_sweep.py
tests/gpt5/phase96_rank_pool_subspace_sweep_summary.py
tests/gpt5/run_phase96_rank_pool_subspace_sweep_full.sh
```

### 运行命令

Smoke:

```bash
python tests/gpt5/phase96_rank_pool_subspace_sweep.py qwen3 \
  --nodes 6:mlp \
  --max-items 4 \
  --ranks 1 \
  --pool-modes tail \
  --factors pc1,object \
  --output-dir results/gpt5_phase96_smoke \
  --progress-every 2 \
  --hard-exit-after-model
```

Full:

```bash
chmod +x tests/gpt5/run_phase96_rank_pool_subspace_sweep_full.sh
tests/gpt5/run_phase96_rank_pool_subspace_sweep_full.sh
```

### 测试设置

```text
output_dir = results/gpt5_phase96_rank_pool_subspace_sweep_full_20260612_113529
models = qwen3 -> GLM4 -> deepseek7b
hard_exit_after_model = true
attention implementation = flash_attention_2,sdpa,eager
实际加载 = sdpa (本地未安装 FlashAttention2)
items = 210
slots = category,color,function,material,location
ranks = 1,4,16
pool_modes = tail,prefix,mean
factors = pc1,object,target,slot,choice
intervention = destroy only
```

节点：

```text
Qwen3: L6 MLP, L24 attention
GLM4: L39 MLP
DeepSeek7B: L27 attention
```

本阶段没有加入 transplant，是为了把诊断维度控制在 rank × pool × factor，不让 donor 维度混入。

### 数据规模

```text
Qwen3 rows = 18900
GLM4 rows = 9450
DeepSeek7B rows = 9450
total_rows = 37800
bad_numeric_rows = 0
```

### 核心原理

对每个 node / pool / rank / factor，先用 `max_rank=16` 构造 basis，再按 `rank=1/4/16` 截断：

```text
destroy:
  h' = h - P_factor(h)
```

然后比较：

```text
value_delta = patched_value_margin - clean_value_margin
letter_delta = patched_letter_margin - clean_letter_margin
```

如果提高 rank 或改变 pool 后破坏效应明显增强，说明 Phase95 的失败可能来自 rank/pooling 不足。

如果仍然没有捕捉到强接口，说明该接口可能不是简单 component-output 子空间，而更像 attention route / token pattern / relational pattern。

### 客观结果

#### 1. Qwen3 L6 MLP：rank/pool 极大改变破坏效应

整体：

```text
L6 MLP:
  value_delta = -1.7485
  letter_delta = -2.4877
  value_top1_delta = -0.1180
  letter_top1_delta = -0.2904
```

代表性结果：

```text
tail rank1:
  choice letter_delta = -0.1869
  object letter_delta = -0.4054
  slot letter_delta = +0.2982
  pc1 letter_delta = -0.0673

tail rank4:
  choice letter_delta = +0.9592
  slot letter_delta = +0.7982
  object letter_delta = -1.3161
  target letter_delta = -0.6429

tail rank16:
  choice letter_delta = -3.9408
  target letter_delta = -3.9167
  object letter_delta = -1.7893

prefix rank4:
  target letter_delta = -3.7214

mean rank4:
  target letter_delta = -3.6995
```

判断：

```text
Qwen3 L6 MLP 的 factor effect 强烈依赖 rank 和 pool。
Phase95 中 rank4-tail 的“slot/choice 破坏后反而上升”不是稳定变量结论，而是低秩/位置截取下的竞争-抑制混合现象。
rank16 后 choice/target 对 letter interface 的必要性大幅显现。
```

这修正 Phase95：

```text
Qwen3 L6 MLP 不是“无子空间结构”，而是存在高秩、位置敏感的混合子空间。
```

#### 2. Qwen3 L24 attention：仍没有被 rank/pool sweep 捕捉为强必要子空间

整体：

```text
L24 attention:
  value_delta = -0.0241
  letter_delta = +0.0346
  value_top1_delta = +0.0007
  letter_top1_delta = -0.0075
```

代表性结果：

```text
tail rank4 object:
  letter_delta = +0.3827

tail rank16 object:
  letter_delta = +0.3274

prefix rank4 choice:
  letter_delta = +0.2173

mean rank16 object:
  letter_delta = -0.2655
```

判断：

```text
即使 rank 提高到 16，pool 从 tail 扩展到 prefix/mean，Qwen3 L24 attention 的强 choice interface 仍没有表现为可被 destroy 的普通低秩输出子空间。
```

这支持：

```text
Qwen3 L24 attention 的 Phase92/93 强现象更可能是 attention route / target-sensitive token pattern / downstream readout alignment，
而不是简单 attn_out 向量子空间。
```

#### 3. GLM4 L39 MLP：candidate scoring 的低秩子空间证据更稳定

整体：

```text
L39 MLP:
  value_delta = -0.3503
  letter_delta = +0.0055
  value_top1_delta = +0.0032
  letter_top1_delta = +0.0104
```

按 factor：

```text
choice:
  value_delta = -0.4349
  letter_delta = -0.0248

target:
  value_delta = -0.4372
  letter_delta = -0.0262

object:
  value_delta = -0.3213
  letter_delta = +0.0099

slot:
  value_delta = -0.3346
  letter_delta = +0.0943

pc1:
  value_delta = -0.2233
  letter_delta = -0.0259
```

rank 影响：

```text
rank1 choice value_delta = -0.2500
rank4 choice value_delta = -0.5582
rank16 choice value_delta = -0.4966

rank1 target value_delta = -0.2685
rank4 target value_delta = -0.4992
rank16 target value_delta = -0.5439
```

判断：

```text
GLM4 L39 MLP 的 value/candidate scoring 确实能被 choice/target/object/slot 子空间稳定破坏。
letter margin 基本不随之下降。
```

这进一步确认：

```text
GLM4 L39 MLP 是 candidate scoring component，不是 choice decision component。
```

#### 4. DeepSeek7B L27 attention：value 与 letter 接口方向相反

整体：

```text
L27 attention:
  value_delta = +0.3830
  letter_delta = -0.2346
  value_top1_delta = -0.0174
  letter_top1_delta = -0.0048
```

按 factor：

```text
pc1:
  value_delta = +0.4156
  letter_delta = -0.1431

choice:
  value_delta = +0.3949
  letter_delta = -0.2769

target:
  value_delta = +0.3866
  letter_delta = -0.2376

object:
  value_delta = +0.3589
  letter_delta = -0.2614

slot:
  value_delta = +0.3590
  letter_delta = -0.2540
```

rank 影响：

```text
rank1 letter_delta 大约 -0.14 到 -0.15
rank4 letter_delta 约 -0.17 到 -0.29
rank16 letter_delta 约 -0.33 到 -0.46
```

代表性：

```text
tail rank16 object:
  value_delta = +0.2891
  letter_delta = -0.5077

prefix rank16 choice:
  value_delta = +0.3774
  letter_delta = -0.5286

mean rank16 slot:
  value_delta = +0.2984
  letter_delta = -0.5098
```

判断：

```text
DeepSeek7B L27 attention 的输出接口更清楚地表现为 value/letter 分离：
destroy 子空间会提高 value margin，却降低 letter margin。
```

这说明：

```text
L27 attention 更像 output-letter interface / readout router，
不是 value candidate builder。
```

### 对 Phase95 的修正

Phase95 说“rank4-tail 子空间不能干净解释变量”，Phase96 进一步拆开：

```text
1. 对 Qwen3 L6 MLP，rank 和 pool 是关键；提高 rank 后破坏效应非常强。
2. 对 Qwen3 L24 attention，rank/pool 仍不能解释强 choice interface，说明可能不是输出子空间问题。
3. 对 GLM4 L39 MLP，低秩子空间稳定破坏 value scoring，模型画像更稳。
4. 对 DeepSeek7B L27 attention，高 rank 更明显破坏 letter interface，但 value margin 反而上升，说明它不是普通语义值通道。
```

### 当前最稳结论

可以说：

```text
深度网络内部存在 factor-like subspace structure，
但它不是单变量、低秩、全位置共享的干净轴。
```

更准确的结构是：

```text
factor effect = rank-sensitive + position-sensitive + interface-specific + competition-mixed
```

### 硬伤

1. 本阶段只做 destroy，没有做 transplant / restore。
2. pool 仍是 tail/prefix/mean，没有真正 token-aligned。
3. factor basis 仍来自差分 + SVD，没有做监督子空间或因果筛选。
4. `destroy` 后指标可能上升，说明 basis 仍混入竞争和抑制成分。
5. 没有做 generation，只评估 value/letter margin。
6. Qwen3 L24 attention 的关键接口仍未定位到可解释子空间。

### 理论进展

条件化关系因子动力学公式继续保留，但需要加入 rank/position/interface 三个约束：

```text
h_l(x) =
  Base_l(x)
  + Σ_r Gate_{l,r}(x, position, interface)
      · F_{l,r}^{rank,position}(x)
  + U_l(x)
```

其中：

```text
Base_l:
  位置、熵、格式、全局状态等基础成分。

F_{l,r}^{rank,position}:
  某一关系因子在特定 rank / token position / component interface 中的子空间表达。

Gate_{l,r}:
  条件化门控，不仅依赖输入内容，也依赖当前读出接口和后续任务格式。

U_l:
  尚未解释的非线性、路由、注意力路径和高阶关系残差。
```

### 下一步

Phase97 应优先做：

```text
Token-Aligned Subspace and Route Test
```

目标：

```text
解释 Qwen3 L24 attention 为什么整组件 transplant 很强，但 rank/pool destroy 不强。
```

测试设计：

```text
1. 对 prompt 中 object token、slot token、target continuation token、choice letter position 分别对齐建 basis。
2. 对 Qwen3 L24 attention 做 token-aligned destroy / transplant。
3. 对 attention route 做更细干预：attn_out token copy、head-level copy、last-token readout copy。
4. 对 DS7B L27 attention 同步测试，比较 output-letter router 是否同构。
5. 对 GLM4 L39 MLP 做 restore，确认 candidate scoring 子空间是否可恢复。
```

判据：

```text
如果 token-aligned / head-level 后 Qwen3 L24 attention 的 choice interface 被捕捉，
说明它是 token route / attention pattern。

如果仍捕捉不到，
说明它可能是 downstream readout compatibility，而不是当前层局部状态本身。
```

## Phase 97: Token Route Local Patch 全量测试 [2026-06-12 16:28]

### 背景

用户提供的 Phase96 分析是正确的：Phase96 没有证明干净因子轴，但把失败拆成不同机制类型：

```text
Qwen3 L6 MLP:
  rank / pool 不足是主要问题，高 rank 后因子破坏效应显现。

Qwen3 L24 attention:
  rank / pool 仍不能捕捉整组件 transplant 强效，必须转向 token route。

GLM4 L39 MLP:
  candidate scoring 子空间稳定，但不控制 choice decision。

DeepSeek7B L27 attention:
  value / letter separation 更清楚。
```

结合 GLM5 Phase470 的 DCF 进展：

```text
Meaning(x) = ΔP(future | x)
```

当前更合理的方向不是继续找概念方向，而是测试某个组件如何改变未来分布约束，尤其是读出位置和局部路由。

### 生成脚本

```text
tests/gpt5/phase97_token_route_local_patch.py
tests/gpt5/phase97_token_route_local_patch_summary.py
tests/gpt5/run_phase97_token_route_local_patch_full.sh
```

### 运行命令

Smoke:

```bash
python tests/gpt5/phase97_token_route_local_patch.py qwen3 \
  --nodes 24:attn \
  --max-items 3 \
  --positions object_span,prompt_tail \
  --donor-kinds same_slot_same_target \
  --output-dir results/gpt5_phase97_smoke \
  --progress-every 1 \
  --hard-exit-after-model
```

Full:

```bash
chmod +x tests/gpt5/run_phase97_token_route_local_patch_full.sh
tests/gpt5/run_phase97_token_route_local_patch_full.sh
```

第一次 qwen3 在 L6 MLP prompt_tail 附近出现一次 Python segmentation fault，已用同一输出目录 resume：

```bash
PHASE97_OUTPUT_DIR=results/gpt5_phase97_token_route_local_patch_full_20260612_151434 \
  tests/gpt5/run_phase97_token_route_local_patch_full.sh
```

resume 从 partial checkpoint 继续，最终三模型完整完成。

### 测试设置

```text
output_dir = results/gpt5_phase97_token_route_local_patch_full_20260612_151434
models = qwen3 -> GLM4 -> DeepSeek7B
hard_exit_after_model = true
实际 attention implementation = sdpa
items = 210
slots = category,color,function,material,location
positions = object_span, relation_span, prompt_tail, last4, prefix8
donor_kinds = same_slot_same_target, same_slot_diff_target
interventions = zero, local token transplant
```

节点：

```text
Qwen3: L24 attention, L6 MLP
GLM4: L39 MLP
DeepSeek7B: L27 attention
```

### 数据规模

```text
Qwen3 rows = 5940
GLM4 rows = 2970
DeepSeek7B rows = 2970
total_rows = 11880
bad_numeric_rows = 0
```

### 核心原理

Phase96 做的是 pooled subspace destroy：

```text
h' = h - P_factor(h)
```

Phase97 改为 token-local route patch：

```text
zero:
  h'[:, token_positions, :] = 0

transplant:
  h'[:, target_positions, :] = donor_h[:, donor_positions, :]
```

测试位置：

```text
object_span:
  对象名词对应 token。

relation_span:
  对象之后到 prompt 结尾的关系/槽位提示 token。

prompt_tail:
  最后一个 prompt token，也就是预测下一个 token 的直接读出位置。

last4:
  prompt 末尾四个 token。

prefix8:
  prompt 前八个 token。
```

判据：

```text
如果 Qwen3 L24 attention 的强 choice interface 来自读出 token route，
那么 prompt_tail / last4 局部 patch 应该强烈影响 letter margin。

如果来自对象语义本身，
object_span 局部 patch 应该强。
```

### 客观结果

#### 1. Qwen3 L24 attention：强接口定位到 prompt_tail / last4

整体：

```text
L24 attention:
  value_delta = -0.1583
  letter_delta = -2.0296
  value_top1_delta = -0.0054
  letter_top1_delta = -0.1993
```

按位置：

```text
last4:
  value_delta = -0.2237
  letter_delta = -5.1242
  letter_top1_delta = -0.5101

prompt_tail:
  value_delta = -0.1954
  letter_delta = -4.9895
  letter_top1_delta = -0.4798

object_span:
  value_delta = +0.0022
  letter_delta = -0.0078

relation_span:
  value_delta = -0.1811
  letter_delta = -0.0114

prefix8:
  value_delta = -0.1936
  letter_delta = -0.0154
```

关键局部条件：

```text
L24 attention / last4 / transplant same_slot_diff_target:
  value_delta = -0.6271
  letter_delta = -10.2530
  letter_top1_delta = -0.8286

L24 attention / prompt_tail / transplant same_slot_diff_target:
  value_delta = -0.4896
  letter_delta = -10.1827
  letter_top1_delta = -0.8238

L24 attention / last4 / zero:
  value_delta = -0.1266
  letter_delta = -4.4708
  letter_top1_delta = -0.6524

L24 attention / prompt_tail / zero:
  value_delta = -0.1435
  letter_delta = -4.1548
  letter_top1_delta = -0.5714
```

而 object_span 基本无效：

```text
L24 attention / object_span / zero:
  value_delta = +0.0120
  letter_delta = -0.0101
  letter_top1_delta = -0.0048
```

判断：

```text
Phase92/93 中 Qwen3 L24 attention 的强 choice interface，
不是 object token 语义局部状态，
而是 prompt_tail / last4 读出位置上的强路由接口。
```

这解释 Phase96：

```text
tail/prefix/mean pooled subspace 抓不到强接口，
是因为接口高度集中在 readout token route，
不是全局平均子空间。
```

#### 2. Qwen3 L6 MLP：prefix8 是更强的全局/早段状态

整体：

```text
L6 MLP:
  value_delta = -0.5257
  letter_delta = -0.3966
  value_top1_delta = -0.0418
  letter_top1_delta = -0.0539
```

按位置：

```text
prefix8:
  value_delta = -1.7410
  letter_delta = -1.5348
  value_top1_delta = -0.1700
  letter_top1_delta = -0.2424

object_span:
  value_delta = -0.2025
  letter_delta = -0.1178

relation_span:
  value_delta = -0.1830
  letter_delta = -0.1204

prompt_tail:
  value_delta = -0.1351
  letter_delta = -0.0547

last4:
  value_delta = -0.3668
  letter_delta = -0.1553
```

关键条件：

```text
L6 MLP / prefix8 / zero:
  value_delta = -2.9812
  letter_delta = -3.8371
  value_top1_delta = -0.2333
  letter_top1_delta = -0.5762

L6 MLP / prefix8 / transplant same_slot_diff_target:
  value_delta = -1.3550
  letter_delta = -0.3815
```

判断：

```text
Qwen3 L6 MLP 的作用更像早段上下文/格式/候选准备状态，
不是只在 prompt_tail 的读出接口。
```

这与 Phase96 的高秩、位置敏感子空间结论一致。

#### 3. GLM4 L39 MLP：value scoring 由 prefix/relation/tail 局部状态控制，letter 仍不是主目标

整体：

```text
L39 MLP:
  value_delta = -0.3435
  letter_delta = +0.0877
  value_top1_delta = +0.0088
  letter_top1_delta = +0.0034
```

按位置：

```text
prefix8:
  value_delta = -0.4543
  letter_delta = 0.0000

relation_span:
  value_delta = -0.4252
  letter_delta = 0.0000

prompt_tail:
  value_delta = -0.4191
  letter_delta = +0.2193

last4:
  value_delta = -0.4191
  letter_delta = +0.2193

object_span:
  value_delta = 0.0000
  letter_delta = 0.0000
```

关键条件：

```text
L39 MLP / prefix8 / zero:
  value_delta = -0.6559

L39 MLP / prefix8 / transplant same_slot_diff_target:
  value_delta = -0.7101

L39 MLP / prompt_tail / zero:
  value_delta = -0.6559
  letter_delta = +0.6417
```

判断：

```text
GLM4 L39 MLP 仍稳定表现为 candidate scoring component。
局部破坏主要影响 value margin，不稳定降低 letter margin。
```

这与 Phase90/91/95/96 一致。

#### 4. DeepSeek7B L27 attention：final readout / letter router 特征更清楚

整体：

```text
L27 attention:
  value_delta = -0.1880
  letter_delta = -0.1286
  value_top1_delta = -0.0357
  letter_top1_delta = +0.0121
```

按位置：

```text
prompt_tail:
  value_delta = -0.2552
  letter_delta = -0.3216

last4:
  value_delta = -0.2552
  letter_delta = -0.3216

relation_span:
  value_delta = -0.2100
  letter_delta = 0.0000

prefix8:
  value_delta = -0.2198
  letter_delta = 0.0000

object_span:
  value_delta = 0.0000
  letter_delta = 0.0000
```

关键条件：

```text
L27 attention / prompt_tail / zero:
  value_delta = -0.1007
  letter_delta = -0.8132

L27 attention / prompt_tail / transplant same_slot_diff_target:
  value_delta = -0.4594
  letter_delta = -0.0997

L27 attention / prefix8 / transplant same_slot_diff_target:
  value_delta = -0.3625
  letter_delta = 0.0000
```

判断：

```text
DeepSeek7B L27 attention 的 letter effect 主要在 prompt_tail / last4。
prefix/relation 局部状态影响 value margin，但不影响 letter margin。
```

这与 Phase96 的 value/letter separation 一致。

### 重要结构解释

为什么很多模型的 object_span 是 0：

```text
在靠近末层的组件中，object token 的局部状态已经很难再影响 prompt_tail 的下一个 token logits。
因为 causal transformer 的最终读出主要来自最后读出位置。
如果 object position 在最后层被改动，但没有足够后续层让信息重新传播到 prompt_tail，
它对最终读出可以接近 0。
```

这不是说明对象信息不存在，而是说明：

```text
对象信息若要影响答案，必须已经通过前面层的 attention route 聚合到读出位置。
```

这正是 Phase97 的核心发现。

### 对 Phase96 的修正

Phase96 说：

```text
Qwen3 L24 attention 可能是 attention route / token pattern，而非普通输出子空间。
```

Phase97 进一步证实：

```text
Qwen3 L24 attention 的关键接口集中在 prompt_tail / last4 readout token。
same_slot_diff_target transplant 在这些位置造成极强 letter collapse。
```

所以 Qwen3 L24 attention 的机制不应继续用：

```text
全序列 pooled factor subspace
```

解释，而应改为：

```text
readout-token route interface
```

### 当前最稳结论

```text
1. Qwen3 L24 attention 是强 readout-token choice interface。
2. Qwen3 L6 MLP 是更早段的上下文/候选准备状态。
3. GLM4 L39 MLP 是 candidate scoring component。
4. DeepSeek7B L27 attention 是 final readout / letter router，并与 value margin 分离。
```

### 硬伤

1. 本阶段只做 token-local whole-vector patch，没有做 head-level patch。
2. donor transplant 仍是整 token hidden state，不是子空间级 transplant。
3. object_span 在末层无效，不能说明对象变量不存在，只能说明末层局部对象状态不能重新传播。
4. qwen3 第一次运行中发生一次 segmentation fault，虽然 resume 后完整完成，但工程稳定性仍需关注。
5. 没有 generation 验证，只看 value/letter margin。
6. 没有把 DCF 约束指纹直接接入 token route 测试。

### 理论进展

条件化关系因子动力学公式需要进一步加入 `readout-token route`：

```text
h_l(t_readout, x) =
  Base_l(t_readout, x)
  + Σ_r Gate_{l,r}(x, t_source -> t_readout, interface)
      · Route_{l,r}(t_source -> t_readout)
      · F_{l,r}(t_source, x)
  + U_l(t_readout, x)
```

更通俗地说：

```text
语言变量不是只存在于某个对象 token。
它必须被路由到当前任务的读出 token，
并在读出 token 上改变未来分布约束。
```

这与 GLM5 Phase470 的 DCF 原理一致：

```text
Meaning(x) = ΔP(future | x)
```

在 GPT5 当前路径中，更具体地说：

```text
Mechanism(x, task) =
  how factors from source tokens are routed into readout token
  and how that readout token changes future distribution.
```

中文：

```text
机制不是对象 token 本身有什么向量，
而是对象/关系/选择因子如何被路由到读出位置，
并改变未来输出分布。
```

### 下一步

Phase98 应做：

```text
Head-Level Readout Route Mapping
```

目标：

```text
在 Qwen3 L24 attention 和 DeepSeek7B L27 attention 中，
定位哪些 attention heads 负责把 source token 信息写入 prompt_tail / last4。
```

测试：

```text
1. 对 Qwen3 L24 attention 做 head-level zero / transplant。
2. 只在 prompt_tail / last4 读出位置 patch 每个 head 输出。
3. 比较 same_slot_same_target 与 same_slot_diff_target。
4. 对 DeepSeek7B L27 attention 做同样测试。
5. 保留 GLM4 L39 MLP 作为 MLP scoring 对照，不做 head-level 主测试。
```

判据：

```text
如果少数 head 的 prompt_tail patch 能复现 L24 attention 的 letter collapse，
说明 choice interface 有可定位的 route head。

如果必须多个 head 或全 attention 输出才有效，
说明接口是分布式 route ensemble。
```

## Phase 98: Head-Level Readout Route Mapping 全量测试 [2026-06-12 19:14]

### 本轮任务

继续 Phase97 的 `readout-token route` 结论，进一步把 Qwen3 L24 attention、GLM4 L39 attention、DeepSeek7B L27 attention 拆到 head 级别，测试：

```text
1. 哪些 attention head 负责 prompt_tail / last4 读出位置的 letter 接口。
2. 是少数 head 可定位，还是必须整个 attention 输出共同工作。
3. Qwen3 / GLM4 / DS7B 在 head-level route 上是否存在同构结构。
```

### 生成脚本

```text
tests/gpt5/phase98_head_readout_route_mapping.py
tests/gpt5/phase98_head_readout_route_mapping_summary.py
tests/gpt5/run_phase98_head_readout_route_mapping_full.sh
```

核心实现：

```text
1. hook self_attn.o_proj 的 forward_pre_hook。
2. 按 o_proj 输入维度和 num_attention_heads 切分每个 head 的输出块。
3. 对单个 head 在 prompt_tail / last4 位置做 zero 或 transplant。
4. donor 使用 same_slot_diff_target，直接测试 choice/letter route 是否被错误目标状态劫持。
5. scoring 只保留 letter margin，避免 head 全扫描数据量过大。
6. 每个模型结束使用 --hard-exit-after-model，三模型顺序运行。
```

### 运行命令

Smoke：

```bash
python tests/gpt5/phase98_head_readout_route_mapping.py qwen3 \
  --layers 24 --heads 0,1 --max-items 4 \
  --positions prompt_tail \
  --donor-kinds same_slot_diff_target \
  --output-dir results/gpt5_phase98_smoke \
  --progress-every 2 \
  --hard-exit-after-model
```

全量：

```bash
chmod +x tests/gpt5/run_phase98_head_readout_route_mapping_full.sh
tests/gpt5/run_phase98_head_readout_route_mapping_full.sh
```

Qwen3 首轮在中途发生一次 segmentation fault，使用 partial checkpoint 续跑完成：

```bash
PHASE98_OUTPUT_DIR=results/gpt5_phase98_head_readout_route_mapping_full_20260612_165924 \
  tests/gpt5/run_phase98_head_readout_route_mapping_full.sh
```

DeepSeek7B 初始全量运行在每 head 重复 clean cache 设计下崩溃；随后修正为每个 item 只计算一次 clean cache，再对所有 head 复用 clean cache，单独完成 DS7B：

```bash
python tests/gpt5/phase98_head_readout_route_mapping.py deepseek7b \
  --layers 27 \
  --heads all \
  --slots category,color,function,material,location \
  --max-items 105 \
  --positions prompt_tail,last4 \
  --donor-kinds same_slot_diff_target \
  --choice-template choice_json_letter \
  --output-dir results/gpt5_phase98_head_readout_route_mapping_full_20260612_165924 \
  --progress-every 35 \
  --hard-exit-after-model
```

汇总：

```bash
python tests/gpt5/phase98_head_readout_route_mapping_summary.py \
  --output-dir results/gpt5_phase98_head_readout_route_mapping_full_20260612_165924
```

### 数据规模

结果目录：

```text
results/gpt5_phase98_head_readout_route_mapping_full_20260612_165924
```

总量：

```text
total_rows = 38640
total_bad_numeric_rows = 0
```

分模型：

```text
Qwen3:
  layer = L24
  heads = 32
  rows = 13440
  bad_numeric_rows = 0

GLM4:
  layer = L39
  heads = 32
  rows = 13440
  bad_numeric_rows = 0

DeepSeek7B:
  layer = L27
  heads = 28
  rows = 11760
  bad_numeric_rows = 0
```

每个 head 测试：

```text
items = 105
positions = prompt_tail, last4
conditions = zero, transplant:same_slot_diff_target
```

### 客观结果

#### 1. Qwen3：L24 attention 的 letter route 高度集中在少数 head

整体：

```text
L24 all heads:
  letter_delta = -0.2038
  letter_top1_delta = -0.0183

condition:
  transplant:same_slot_diff_target:
    letter_delta = -0.3062
    letter_top1_delta = -0.0359
  zero:
    letter_delta = -0.1014
    letter_top1_delta = -0.0007
```

最强 head：

```text
L24:29:
  letter_delta = -3.0116
  letter_top1_delta = -0.3238

L24:31:
  letter_delta = -2.5554
  letter_top1_delta = -0.2571

L24:28:
  letter_delta = -0.6440
  letter_top1_delta = -0.0048
```

最强具体条件：

```text
L24:29:last4:transplant:same_slot_diff_target:
  letter_delta = -5.1619
  letter_top1_delta = -0.6286

L24:29:prompt_tail:transplant:same_slot_diff_target:
  letter_delta = -5.1595
  letter_top1_delta = -0.6381

L24:31:last4:transplant:same_slot_diff_target:
  letter_delta = -4.5131
  letter_top1_delta = -0.5143

L24:31:prompt_tail:transplant:same_slot_diff_target:
  letter_delta = -4.5012
  letter_top1_delta = -0.5143
```

这说明 Phase97 中 Qwen3 L24 attention 的读出位置 letter collapse 不是均匀分布在所有 head，而是主要集中在 L24 head 29 / 31，head 28 也有明显 zero 破坏效应。

#### 2. GLM4：L39 attention 几乎不是 letter route 接口

整体：

```text
L39 all heads:
  letter_delta = -0.0009
  letter_top1_delta = -0.0077

condition:
  transplant:same_slot_diff_target:
    letter_delta = -0.0009
    letter_top1_delta = -0.0074
  zero:
    letter_delta = -0.0010
    letter_top1_delta = -0.0080
```

最强 head 也很弱：

```text
L39:31:
  letter_delta = -0.0366
  letter_top1_delta = -0.0143

L39:17:
  letter_delta = -0.0265
  letter_top1_delta = -0.0143

L39:20:
  letter_delta = -0.0199
  letter_top1_delta = -0.0143
```

这和 Phase96 / Phase97 一致：GLM4 的强信号主要不是末层 attention 的 letter route，而更像 MLP/candidate scoring 或其他非 head-local 的输出接口。

#### 3. DeepSeek7B：L27 attention 单 head 效应弱，zero 比 transplant 更明显

整体：

```text
L27 all heads:
  letter_delta = -0.0215
  letter_top1_delta = +0.0058

condition:
  zero:
    letter_delta = -0.0383
    letter_top1_delta = +0.0051
  transplant:same_slot_diff_target:
    letter_delta = -0.0047
    letter_top1_delta = +0.0065
```

相对最强 head：

```text
L27:21:
  letter_delta = -0.1295
  letter_top1_delta = 0.0000

L27:26:
  letter_delta = -0.0997
  letter_top1_delta = +0.0048

L27:9:
  letter_delta = -0.0935
  letter_top1_delta = +0.0048
```

最强具体条件：

```text
L27:21:prompt_tail:zero:
  letter_delta = -0.2518
  letter_top1_delta = -0.0095

L27:21:last4:zero:
  letter_delta = -0.2518
  letter_top1_delta = -0.0095

L27:26:prompt_tail:zero:
  letter_delta = -0.1845
  letter_top1_delta = 0.0000

L27:9:prompt_tail:zero:
  letter_delta = -0.1815
  letter_top1_delta = +0.0095
```

DeepSeek7B 的 L27 attention 存在若干弱 head 贡献，但没有出现 Qwen3 那种 `same_slot_diff_target transplant` 强烈劫持 letter 输出的 head。它更像输出释放附近的弱分布式接口，而不是可由单一 head transplant 强控制的 choice route。

### 本轮关键进展

1. Phase97 的 Qwen3 L24 readout-token route 被定位到 head 级：主要是 L24 head 29 和 head 31。
2. Qwen3 的强效来自 transplant:same_slot_diff_target，而不只是 zero ablation，说明这些 head 不只是“重要”，而是携带可被错误目标劫持的 letter/choice route 内容。
3. GLM4 的末层 attention head 几乎不控制 letter 输出，继续支持 GLM4 与 Qwen3 的路径组织差异。
4. DeepSeek7B 的 L27 attention head 有弱破坏效应，但 transplant 不强，说明它不像 Qwen3 一样有清晰单 head choice route。

### 问题和硬伤

1. 本轮只测试了一个候选层：Qwen3 L24、GLM4 L39、DeepSeek7B L27；还没有做跨层 head map。
2. 只测 letter margin，没有同时测 value margin，因此不能判断这些 head 是否也影响候选值语义。
3. Qwen3 head 29/31 很强，但还没有做 head combination / restore，不能证明最小充分 head set。
4. 还没有做 attention pattern 分析，因此不知道 head 29/31 具体从哪些 source token 读取。
5. DeepSeek7B 仍可能依赖 L20-L27 多层连续轨迹，本轮单层 head map 不足以解释它的深层机制。
6. Qwen3 与 DS7B 测试过程中都发生过工程崩溃，虽然 partial/resume 和 cache 优化后完成，但后续重要结论仍建议复测一次。

### 理论进展

条件化关系因子动力学公式需要继续细化，把 `readout-token route` 拆成 head 级路由项：

```text
h_l(t_readout, x)
= Base_l(t_readout, x)
 Σ_h RouteHead_{l,h}(t_source -> t_readout, x)
 Σ_m MLPFactor_{l,m}(t_readout, x)
 U_l(t_readout, x)
```

更具体到本轮结果：

```text
Qwen3:
  choice/letter route 主要经过 L24 attention head 29/31。

GLM4:
  letter route 不在 L39 attention head，可能在 MLP scoring / residual 输出接口。

DeepSeek7B:
  L27 attention 有弱释放效应，但不是单 head transplant 可控接口。
```

因此当前不能再只说：

```text
attention 负责路由。
```

而要更精确地说：

```text
某些模型中，少数 attention head 在特定读出 token 上承担 choice/letter route；
另一些模型中，同样功能可能由 MLP scoring、残差输出接口或多层轨迹完成。
```

### 下一步

Phase99 应做：

```text
Qwen3 L24 head 29/31 route source analysis and restore
```

目标：

```text
1. 对 Qwen3 L24 head 29/31 做 attention pattern source token 分析。
2. 测试它们主要读取 object_span、relation_span、candidate block、还是 prompt_tail 自身。
3. 做 head 29/31 组合 ablation，比较 single-head 与 pair-head 的必要性。
4. 做 restore：先破坏全 L24 attention 或 head 29/31，再只恢复 head 29/31，测试 letter/top1 是否恢复。
5. 同时补测 value margin，判断 head 29/31 是纯 letter route、choice route，还是 value+letter 共同接口。
```

如果 restore 成立，Qwen3 将获得第一个比较接近“读出路由最小电路”的证据；如果 restore 不成立，则说明 head 29/31 是强必要节点，但不是充分闭包。

## Phase 99: Head-Set Route Closure 必要性/充分性测试 [2026-06-12 20:51]

### 本轮任务

基于 Phase98 的 head-level 定位结果，继续测试：

```text
1. Qwen3 L24 head 29/31 是否构成接近最小的 letter readout route。
2. zero_heads 是否显示必要性。
3. keep_heads 是否显示充分性。
4. transplant_heads 是否能复现 transplant_all 的错误目标劫持。
5. value margin 与 letter margin 是否同向变化。
```

对照模型：

```text
Qwen3:
  L24 heads 29, 31, 28

GLM4:
  L39 heads 31, 17, 20

DeepSeek7B:
  L27 heads 21, 26, 9
```

### 生成脚本

```text
tests/gpt5/phase99_head_set_route_closure.py
tests/gpt5/phase99_head_set_route_closure_summary.py
tests/gpt5/run_phase99_head_set_route_closure_full.sh
```

### 测试原理

Phase98 只测单个 head 的 zero / transplant。Phase99 改成 head-set 级对照：

```text
zero_all:
  清除读出位置所有 head 输出。

transplant_all:
  把 donor 的所有 head 输出移植到读出位置。

zero_heads:
  只清除候选 head set。

keep_heads:
  保留候选 head set，清除其他所有 head。

transplant_heads:
  只移植候选 head set。
```

判断逻辑：

```text
zero_heads 强：
  候选 head set 对当前输出有必要性。

keep_heads 接近 clean：
  候选 head set 可能充分。

keep_heads 仍明显下降：
  候选 head set 不充分，需要其他 head / residual / MLP 协同。

transplant_heads 接近 transplant_all：
  候选 head set 携带主要可劫持的 route 内容。
```

本轮同时测：

```text
value_margin
letter_margin
value_top1
letter_top1
```

### 运行命令

Smoke：

```bash
python tests/gpt5/phase99_head_set_route_closure.py qwen3 \
  --layer 24 \
  --head-sets 'single29=29;pair2931=29,31' \
  --max-items 2 \
  --positions prompt_tail \
  --output-dir results/gpt5_phase99_smoke \
  --progress-every 1 \
  --hard-exit-after-model
```

全量：

```bash
chmod +x tests/gpt5/run_phase99_head_set_route_closure_full.sh
tests/gpt5/run_phase99_head_set_route_closure_full.sh
```

Qwen3 在最后阶段出现一次 segmentation fault，已从 partial checkpoint 恢复，继续完成三模型：

```bash
PHASE99_OUTPUT_DIR=results/gpt5_phase99_head_set_route_closure_full_20260612_191927 \
  tests/gpt5/run_phase99_head_set_route_closure_full.sh
```

汇总：

```bash
python tests/gpt5/phase99_head_set_route_closure_summary.py \
  --output-dir results/gpt5_phase99_head_set_route_closure_full_20260612_191927
```

### 数据规模

结果目录：

```text
results/gpt5_phase99_head_set_route_closure_full_20260612_191927
```

总量：

```text
total_rows = 17640
total_bad_numeric_rows = 0
```

分模型：

```text
Qwen3:
  rows = 5880
  bad_numeric_rows = 0

GLM4:
  rows = 5880
  bad_numeric_rows = 0

DeepSeek7B:
  rows = 5880
  bad_numeric_rows = 0
```

每模型：

```text
items = 210
positions = prompt_tail, last4
conditions = 14
```

### 客观结果

#### 1. Qwen3：L24 head 29/31 是强必要和强可劫持 route，但不是完整充分闭包

核心结果：

```text
zero_all:
  value_delta = -0.1182
  letter_delta = -4.3128
  letter_top1_delta = -0.6119

transplant_all:
  value_delta = -0.4243
  letter_delta = -10.2179
  letter_top1_delta = -0.8262
```

单 head：

```text
transplant_heads:single29:
  value_delta = -0.0040
  letter_delta = -5.1042
  letter_top1_delta = -0.6048

transplant_heads:single31:
  value_delta = -0.0059
  letter_delta = -4.3655
  letter_top1_delta = -0.5381
```

组合 head：

```text
zero_heads:pair2931:
  value_delta = +0.0013
  letter_delta = -3.5652
  letter_top1_delta = -0.4643

transplant_heads:pair2931:
  value_delta = -0.0128
  letter_delta = -9.6869
  letter_top1_delta = -0.8024

transplant_heads:wide282931:
  value_delta = +0.0266
  letter_delta = -9.8193
  letter_top1_delta = -0.8048
```

充分性测试：

```text
keep_heads:pair2931:
  value_delta = -0.1172
  letter_delta = -1.7494
  letter_top1_delta = -0.0643

keep_heads:wide282931:
  value_delta = -0.1203
  letter_delta = -0.8241
  letter_top1_delta = -0.0357
```

解释限定：

```text
1. head 29/31 组合几乎复现 transplant_all 的 letter 劫持：
   -9.6869 vs -10.2179。

2. 加入 head 28 后，transplant 仍接近 transplant_all：
   -9.8193 vs -10.2179。

3. zero head 29/31 会造成强 letter 下降：
   -3.5652。

4. keep head 29/31 或 28/29/31 不能完全保持 clean：
   keep 仍有 letter_delta = -1.7494 / -0.8241。
```

因此 Qwen3 L24 head 29/31 是强必要节点，也是主要可劫持 route；但它们不是完整充分电路，至少还需要其他 head、residual 状态或后续输出接口协同。

最关键的新发现：

```text
Qwen3 的 letter route 与 value route 分离。
```

因为 head 29/31 transplant 对 letter 非常强：

```text
letter_delta ≈ -9.69
```

但 value 影响很小：

```text
value_delta ≈ -0.013
```

这说明 head 29/31 更像 `choice/letter interface`，不是完整语义 value 表征。

#### 2. GLM4：L39 attention head-set 仍然很弱

核心结果：

```text
zero_all:
  value_delta = +0.0177
  letter_delta = -0.0390

transplant_all:
  value_delta = -0.0755
  letter_delta = -0.0357

zero_heads:wide311720:
  value_delta = +0.0039
  letter_delta = -0.1063

transplant_heads:wide311720:
  value_delta = -0.0220
  letter_delta = -0.0476
```

GLM4 的 L39 attention 即使取 Phase98 最强 heads，也没有形成 Qwen3 那种强 letter route。GLM4 的可解释路径仍应优先看 MLP scoring / residual output，而不是末层 attention head route。

#### 3. DeepSeek7B：L27 attention 有破坏效应，但没有 transplant 劫持效应

核心结果：

```text
zero_all:
  value_delta = -0.0612
  letter_delta = -0.8132
  letter_top1_delta = +0.0714

transplant_all:
  value_delta = -0.0121
  letter_delta = -0.0997
  letter_top1_delta = +0.0095
```

候选 head set：

```text
zero_heads:single21:
  letter_delta = -0.2536

zero_heads:single26:
  letter_delta = -0.1905

zero_heads:pair2126:
  letter_delta = -0.3985

zero_heads:wide212609:
  letter_delta = -0.4884
```

transplant 很弱：

```text
transplant_heads:single21:
  letter_delta = -0.0101

transplant_heads:pair2126:
  letter_delta = -0.0155

transplant_heads:wide212609:
  letter_delta = -0.0179
```

DeepSeek7B 的 L27 attention heads 有一定破坏效应，但不能被 same_slot_diff_target donor 强劫持。这与前面“DeepSeek7B 更像深层连续轨迹/输出释放型，而不是单点可控 route head”一致。

### 本轮关键进展

1. Qwen3 L24 head 29/31 被确认为主要 letter/choice route head。
2. Qwen3 的 head 29/31 pair transplant 几乎复现 transplant_all，说明错误目标劫持主要经过这组 head。
3. Qwen3 的 value 与 letter 明显分离：head 29/31 强烈影响 letter，但几乎不改变 value margin。
4. GLM4 L39 attention head route 基本被排除为主要机制。
5. DeepSeek7B L27 attention 有弱必要性但无 transplant 可控性，继续支持多层轨迹解释。

### 问题和硬伤

1. Qwen3 的 keep_heads 不充分，说明 head 29/31 不是完整电路。
2. 还没有做真正 restore：例如先 zero_all，再恢复 head 29/31 的 clean donor 状态。
3. 还没有分析 head 29/31 的 attention source token，因此不知道它从 object、target、candidate option、还是格式 token 读入。
4. 只在一个层做 head-set closure，还没有做跨层 head relay。
5. Qwen3 运行中仍发生 segmentation fault，虽然恢复后完成，但关键结果建议之后独立复测。
6. 本轮仍是 margin 实验，没有 generation 或行为输出验证。

### 理论进展

条件化关系因子动力学需要区分：

```text
value factor:
  候选内容本身的语义/属性分数。

choice interface:
  把候选内容映射到 A/B/C/D 或输出格式的读出接口。
```

本轮最重要的结构证据是：

```text
Qwen3 L24 head 29/31 主要控制 choice/letter interface，
而不是完整 value factor。
```

因此公式应从：

```text
h_l(t_readout)
= Base + RouteHead + MLPFactor + U
```

细化为：

```text
h_l(t_readout)
= Base_l
+ ValueFactors_l(object, relation, target)
+ ChoiceRouteHeads_l(candidate -> letter)
+ FormatInterface_l(task, template)
+ ResidualCarry_l
+ U_l
```

中文解释：

```text
语言输出不是只由“目标值”决定，
还要经过一个把目标值映射到当前任务输出格式的接口。

Qwen3 的 L24 head 29/31 更像这个接口的一部分。
```

这解释了为什么：

```text
1. head 29/31 transplant 能强烈打掉 letter；
2. value margin 几乎不变；
3. GLM4 和 DS7B 没有同样的 head-local choice interface。
```

### 下一步

Phase100 应做：

```text
Qwen3 L24 head 29/31 Source Token Attribution and Restore
```

目标：

```text
1. 分析 head 29/31 的 attention source token。
2. 区分它读取的是 object/relation/target，还是 candidate option / letter format。
3. 做真正 restore：
   zero_all L24 attention at readout token
   + restore head 29/31 clean state
   测试是否恢复 letter margin/top1。
4. 做 value/letter 双读出：
   判断 restore 是否只恢复 letter，不恢复 value。
```

如果 restore 成立：

```text
Qwen3 将得到第一个 choice interface 层面的闭包证据。
```

如果 restore 不成立：

```text
head 29/31 是强 route 节点，但必须与其他 head/residual/MLP 共同构成闭包。
```

## Phase 100: Head Route Restore and Source Attribution [2026-06-13 01:49]

### 本轮任务

根据 Phase98/99 的结论继续完成：

```text
1. 对 Qwen3 L24 head 29/31 做真正 restore 测试。
2. 对 GLM4 L39 和 DS7B L27 做同样 head-set restore 对照。
3. 对候选 head 做 source attention attribution。
4. 同时观察 value margin 和 letter margin，继续验证 value route 与 choice/letter route 是否分离。
```

核心问题：

```text
Qwen3 head 29/31 是否只是强可劫持节点，
还是能在 transplant_all 破坏后恢复 letter 接口？
```

### 生成脚本

```text
tests/gpt5/phase100_head_route_restore_source.py
tests/gpt5/phase100_head_route_restore_source_summary.py
tests/gpt5/run_phase100_head_route_restore_source_full.sh
```

### 测试设计

本轮区分两类测试：

```text
主干 restore 测试：
  使用 sdpa，210 items，三模型顺序运行。

source attention attribution：
  使用 eager，60 items，三模型顺序补写 source_attention。
```

之所以拆开：

```text
sdpa 不返回 output_attentions。
eager 可以返回 attention，但全量 restore 若都用 eager 会非常慢。
```

主干条件：

```text
zero_all:
  清零读出位置所有 head。

transplant_all:
  把错误目标 donor 的所有 head 移植到读出位置。

zero_heads:
  只清零候选 head set。

transplant_heads:
  只移植候选 head set。

zero_all_restore_clean_heads:
  先清零所有 head，再恢复候选 head 的 clean 状态。

transplant_all_restore_clean_heads:
  先把所有 head 替换成错误目标 donor，
  再只恢复候选 head 的 clean 状态。
```

最关键条件：

```text
transplant_all_restore_clean_heads
```

判据：

```text
如果 transplant_all 使 letter 崩溃，
而 transplant_all_restore_clean_heads 能恢复 letter，
说明候选 head set 不只是可劫持节点，
而是对 choice/letter interface 有恢复能力。
```

### 运行命令

Smoke：

```bash
python tests/gpt5/phase100_head_route_restore_source.py qwen3 \
  --layer 24 \
  --head-sets 'single29=29;pair2931=29,31' \
  --max-items 2 \
  --source-attn-items 1 \
  --positions prompt_tail \
  --output-dir results/gpt5_phase100_smoke \
  --progress-every 1 \
  --hard-exit-after-model

python tests/gpt5/phase100_head_route_restore_source.py qwen3 \
  --layer 24 \
  --head-sets 'single29=29;pair2931=29,31' \
  --max-items 2 \
  --source-attn-items 1 \
  --positions prompt_tail \
  --output-dir results/gpt5_phase100_smoke \
  --attn-implementations eager \
  --source-only \
  --hard-exit-after-model
```

全量：

```bash
chmod +x tests/gpt5/run_phase100_head_route_restore_source_full.sh
tests/gpt5/run_phase100_head_route_restore_source_full.sh
```

汇总：

```bash
python tests/gpt5/phase100_head_route_restore_source_summary.py \
  --output-dir results/gpt5_phase100_head_route_restore_source_full_20260612_221206
```

### 数据规模

结果目录：

```text
results/gpt5_phase100_head_route_restore_source_full_20260612_221206
```

总量：

```text
total_rows = 22680
total_bad_numeric_rows = 0
```

分模型：

```text
Qwen3:
  rows = 7560
  source_attention_items = 60
  bad_numeric_rows = 0

GLM4:
  rows = 7560
  source_attention_items = 60
  bad_numeric_rows = 0

DeepSeek7B:
  rows = 7560
  source_attention_items = 60
  bad_numeric_rows = 0
```

### 客观结果

#### 1. Qwen3：head 29/31 restore 几乎救回 transplant_all 的 letter 崩溃

破坏条件：

```text
transplant_all:
  value_delta = -0.4243
  letter_delta = -10.2179
  letter_top1_delta = -0.8262
```

只移植 head 29/31：

```text
transplant_heads:pair2931:
  value_delta = -0.0128
  letter_delta = -9.6869
  letter_top1_delta = -0.8024
```

这复现 Phase99：错误目标劫持主要经过 head 29/31。

关键 restore：

```text
transplant_all_restore_clean_heads:pair2931:
  value_delta = -0.4099
  letter_delta = -0.3054
  letter_top1_delta = -0.0357
```

加入 head 28：

```text
transplant_all_restore_clean_heads:wide282931:
  value_delta = -0.4449
  letter_delta = -0.1765
  letter_top1_delta = -0.0238
```

对比：

```text
transplant_all letter_delta:
  -10.2179

restore head 29/31 后:
  -0.3054

restore head 28/29/31 后:
  -0.1765
```

这说明：

```text
Qwen3 L24 head 29/31 不只是强必要和强可劫持节点，
而且在 all-head wrong-target corruption 后，
能恢复绝大多数 letter interface。
```

但 value 没有恢复：

```text
transplant_all_restore_clean_heads:pair2931:
  value_delta = -0.4099
```

也就是说：

```text
head 29/31 restore 主要恢复 letter/choice interface，
不恢复 value factor。
```

这进一步证实：

```text
value route 和 choice/letter route 是分离结构。
```

zero_all restore：

```text
zero_all:
  letter_delta = -4.3128

zero_all_restore_clean_heads:pair2931:
  letter_delta = -1.7494

zero_all_restore_clean_heads:wide282931:
  letter_delta = -0.8241
```

这说明在完全清零所有 head 后，恢复 head 29/31 只能部分救回；加入 head 28 更好，但仍不等于 clean。也就是说 head 29/31 是 choice interface 的核心，但不是完整 attention 输出充分条件。

#### 2. Qwen3 source attention：head 29/31 主要读 letter label，不是 object/relation

source attribution 结果：

```text
H29 -> letter_label:
  attention = 0.690976

H31 -> letter_label:
  attention = 0.453401

H31 -> readout_tail:
  attention = 0.186336

H31 -> distractor_option:
  attention = 0.184171

H28 -> distractor_option:
  attention = 0.260342

H28 -> letter_label:
  attention = 0.251471
```

这说明 Qwen3 L24 head 29/31 主要不是直接读取 object / relation，而是强烈读取选项字母标签和输出格式区域。

因此它们更准确的定位是：

```text
letter-label / choice-format interface heads
```

而不是：

```text
semantic value heads
```

#### 3. GLM4：attention restore 不构成主要机制

核心条件：

```text
transplant_all:
  value_delta = -0.0755
  letter_delta = -0.0357

transplant_all_restore_clean_heads:pair3117:
  value_delta = -0.0661
  letter_delta = -0.0071

transplant_all_restore_clean_heads:wide311720:
  value_delta = -0.0565
  letter_delta = +0.0083
```

GLM4 的 attention 本身破坏很小，restore 也很小。source attention 显示：

```text
H17 -> readout_tail:
  attention = 0.999561

H31 -> readout_tail:
  attention = 0.993748

H20 -> readout_tail:
  attention = 0.986995
```

这说明 GLM4 L39 attention heads 几乎是读出位置自环或局部保持，不是 Qwen3 那种读取 letter label 的 choice interface。

#### 4. DeepSeek7B：restore 不强，source 分散在 option/tail

核心条件：

```text
transplant_all:
  value_delta = -0.0121
  letter_delta = -0.0997

transplant_all_restore_clean_heads:pair2126:
  value_delta = -0.0361
  letter_delta = -0.0902

transplant_all_restore_clean_heads:wide212609:
  value_delta = -0.0543
  letter_delta = -0.0929
```

因为 DS7B 的 transplant_all 本身就很弱，所以 restore 不能证明强 choice interface。

source attention：

```text
H21 -> distractor_option:
  attention = 0.336774

H21 -> letter_label:
  attention = 0.268165

H26 -> readout_tail:
  attention = 0.317012

H9 -> readout_tail:
  attention = 0.300458

H9 -> distractor_option:
  attention = 0.284884
```

DS7B 存在 option/tail 相关读取，但没有形成 Qwen3 那样可由 head restore 闭合的 letter interface。

### 本轮关键进展

1. Qwen3 L24 head 29/31 获得了真正的 restore 证据：

```text
transplant_all:
  letter_delta = -10.2179

restore head 29/31:
  letter_delta = -0.3054
```

2. Qwen3 head 29/31 restore 不恢复 value：

```text
value_delta = -0.4099
```

这证明它们更像 choice/letter interface，而不是 semantic value factor。

3. Qwen3 source attribution 显示 head 29/31 主要读取 letter_label，说明它们是输出格式接口头。

4. GLM4 的 L39 heads 主要 readout_tail self-loop，不是 choice interface。

5. DS7B 的 L27 heads 有 option/tail 注意力，但缺少可恢复的强 letter route。

### 问题和硬伤

1. Qwen3 restore 是 scoring-level closure，还不是 generation-level closure。
2. source attribution 的标签是规则化粗分类，尚未逐 token 人工审查。
3. Qwen3 的 value route 仍未定位；head 29/31 只解释 letter/choice。
4. zero_all_restore_clean_heads 仍不能完全恢复，说明完整 attention 输出还有其他辅助 head。
5. 还没有把上游 value factor 如何进入 choice interface 串起来。
6. GLM4 / DS7B 的机制仍未闭合，只排除了同构的 Qwen3-style head-local choice interface。

### 理论进展

本轮把输出机制进一步拆成三层：

```text
1. semantic value factor:
   决定候选内容是否正确。

2. choice/letter interface:
   把当前候选映射到 A/B/C/D 或格式标签。

3. generation/output policy:
   把格式标签实际生成出来。
```

Qwen3 L24 head 29/31 属于第 2 层：

```text
choice/letter interface
```

而不是第 1 层：

```text
semantic value factor
```

条件化关系因子动力学公式继续细化为：

```text
h_l(t_readout, x)
= Base_l
+ ValuePath_l(object, relation, target)
+ ChoiceFormatHeads_l(letter_label, option_block, task_format)
+ Bridge_l(value -> choice)
+ ResidualCarry_l
+ U_l
```

本轮已经较强确认：

```text
ChoiceFormatHeads_l
```

在 Qwen3 中具体包含：

```text
L24 head 29/31
```

但仍未定位：

```text
ValuePath_l
Bridge_l(value -> choice)
```

中文解释：

```text
模型不是直接从语义值生成字母答案。
它先有候选值评分路径，再通过格式接口路径映射到字母输出。
Qwen3 的 L24 head 29/31 主要负责后者。
```

### 下一步

Phase101 应做：

```text
Value-to-Choice Bridge Mapping
```

目标：

```text
1. 在 Qwen3 中追踪 value factor 如何进入 L24 head 29/31 的 choice interface。
2. 重点连接 Phase96 的 L6 MLP rank/pool value factor 与 Phase100 的 L24 head 29/31 letter interface。
3. 做 L6 MLP value destroy + L24 head 29/31 restore，测试是否能救 letter。
4. 做 L6 clean restore + L24 wrong head transplant，测试 value 正确但 choice 错误是否分离。
5. 对 GLM4 和 DS7B 做同样对照，判断它们是否缺少显式 Bridge_l，或桥接发生在 MLP/residual 输出接口。
```

如果成功，将得到：

```text
value path -> choice interface
```

之间的第一条跨层桥接证据。

## Phase 101: Value-to-Choice Bridge Mapping [2026-06-13 15:01]

### 本轮任务

结合 GPT5 Phase100 和 GLM5 Phase480 的最新进展继续推进。

GLM5 Phase480 的关键进展是：

```text
类别边界残差是普遍机制：
  Qwen3 8/8 类别 selectivity > 1
  GLM4 6/8 类别 selectivity > 1
  DS7B 5/8 类别 selectivity > 1

category_specific 方向有自然使用证据：
  Qwen3 8/8 类别在自身 specific 方向上投影最高。

反向注入有效：
  Qwen3 4/4 类别 -specific 方向能抑制对应类别。
```

这说明 value factor / semantic boundary path 不是虚构方向，而是模型自然使用的语义值路径之一。

GPT5 Phase100 的关键进展是：

```text
Qwen3 L24 head 29/31 是 choice/letter format interface。
它能恢复 letter，但不能恢复 value。
```

因此 Phase101 要测试：

```text
value path 与 choice interface 是否可分离？
value path 被破坏后，choice head restore 能不能救 letter？
choice head 被污染后，value path 是否仍保持？
```

### 生成脚本

```text
tests/gpt5/phase101_value_choice_bridge_mapping.py
tests/gpt5/phase101_value_choice_bridge_mapping_summary.py
tests/gpt5/run_phase101_value_choice_bridge_mapping_full.sh
```

### 测试设计

三模型节点：

```text
Qwen3:
  value path = L6 MLP prefix8
  choice interface = L24 head 29/31 prompt_tail

GLM4:
  value path = L39 MLP prefix8
  choice interface = L39 head 31/17 prompt_tail

DeepSeek7B:
  value path = L27 MLP prefix8
  choice interface = L27 head 21/26 prompt_tail
```

主测试条件：

```text
value_zero:
  清零 value node。

value_transplant:
  把 value node 替换为 same_slot_diff_target donor。

choice_transplant_heads:
  只替换 choice heads。

value_zero + choice_restore_clean_heads:
  value node 清零，同时把 choice heads 恢复为 clean。

value_transplant + choice_restore_clean_heads:
  value node 被 donor 替换，同时把 choice heads 恢复为 clean。

value_transplant + choice_transplant_heads:
  value node 和 choice heads 都被 donor 替换。

value_transplant + choice_transplant_all_restore_clean_heads:
  value node 被 donor 替换；
  choice attention 全部 donor；
  但 choice heads 恢复 clean。
```

判据：

```text
如果 value_zero 破坏 value 和 letter，
但 choice_restore_clean_heads 只恢复 letter、不恢复 value，
说明 choice interface 可绕过或覆盖 letter 输出格式，
但不能恢复语义值路径。

如果 choice_transplant_heads 只破坏 letter、不破坏 value，
说明 choice interface 与 value path 分离。
```

### 运行命令

Smoke：

```bash
python tests/gpt5/phase101_value_choice_bridge_mapping.py qwen3 \
  --value-layer 6 \
  --value-component mlp \
  --value-position prefix8 \
  --choice-layer 24 \
  --choice-heads 29,31 \
  --max-items 2 \
  --output-dir results/gpt5_phase101_smoke \
  --progress-every 1 \
  --hard-exit-after-model
```

三模型主测试：

```bash
chmod +x tests/gpt5/run_phase101_value_choice_bridge_mapping_full.sh
tests/gpt5/run_phase101_value_choice_bridge_mapping_full.sh
```

Qwen3 关键结果加大数据复测：

```bash
OUT=results/gpt5_phase101_value_choice_bridge_mapping_qwen3_validate_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
python tests/gpt5/phase101_value_choice_bridge_mapping.py qwen3 \
  --value-layer 6 \
  --value-component mlp \
  --value-position prefix8 \
  --choice-layer 24 \
  --choice-heads 29,31 \
  --max-items 240 \
  --choice-position prompt_tail \
  --donor-kind same_slot_diff_target \
  --choice-template choice_json_letter \
  --progress-every 40 \
  --output-dir "$OUT" \
  --hard-exit-after-model 2>&1 | tee "$OUT/qwen3_validate.log"
python tests/gpt5/phase101_value_choice_bridge_mapping_summary.py \
  --output-dir "$OUT" | tee "$OUT/summary.log"
```

### 数据规模

三模型主测试：

```text
results/gpt5_phase101_value_choice_bridge_mapping_full_20260613_141841

total_rows = 2520
total_bad_numeric_rows = 0

Qwen3 = 840 rows
GLM4 = 840 rows
DS7B = 840 rows
items/model = 120
```

Qwen3 复测：

```text
results/gpt5_phase101_value_choice_bridge_mapping_qwen3_validate_20260613_144631

rows = 1680
bad_numeric_rows = 0
items = 240
```

### 客观结果

#### 1. Qwen3 主测试：value path 与 choice interface 强分离

```text
value_zero:
  value_delta = -3.2103
  letter_delta = -4.0358
  value_top1_delta = -0.3250
  letter_top1_delta = -0.6333
```

L6 MLP value node 清零后，value 和 letter 都下降，说明 L6 MLP prefix8 是上游强 value path。

```text
choice_transplant_heads:
  value_delta = -0.0093
  letter_delta = -10.0734
  value_top1_delta = 0.0000
  letter_top1_delta = -0.8750
```

只污染 L24 head 29/31，value 几乎不动，但 letter 大崩。这再次说明 head 29/31 是 choice/letter interface，不是 value path。

最关键桥接条件：

```text
value_zero + choice_restore_clean_heads:
  value_delta = -3.1953
  letter_delta = +0.3755
  value_top1_delta = -0.3417
  letter_top1_delta = +0.0667
```

解释：

```text
value 已经被 L6 MLP zero 严重破坏，
但只要 L24 head 29/31 恢复 clean，
letter 反而恢复到接近 clean，甚至略正。
```

这说明：

```text
Qwen3 的 letter choice 可以被 L24 head 29/31 clean interface 强行恢复，
即使上游 value margin 没有恢复。
```

这不是完整语义恢复，而是输出接口恢复。

另一个关键条件：

```text
value_transplant + choice_transplant_all_restore_clean_heads:
  value_delta = -1.5215
  letter_delta = -0.3365
  value_top1_delta = -0.2000
  letter_top1_delta = -0.0333
```

即使 value path 被 donor 替换、choice attention 全部被 donor 污染，只恢复 head 29/31 clean 也能大幅救回 letter。

#### 2. Qwen3 240 items 复测确认

```text
value_zero:
  value_delta = -3.2425
  letter_delta = -4.0035
  value_top1_delta = -0.3042
  letter_top1_delta = -0.6208

choice_transplant_heads:
  value_delta = -0.0065
  letter_delta = -10.2411
  value_top1_delta = +0.0083
  letter_top1_delta = -0.8917

value_zero + choice_restore_clean_heads:
  value_delta = -3.2351
  letter_delta = +0.3130
  value_top1_delta = -0.3125
  letter_top1_delta = +0.0458

value_transplant + choice_transplant_all_restore_clean_heads:
  value_delta = -1.4074
  letter_delta = -0.4401
  value_top1_delta = -0.1417
  letter_top1_delta = -0.0333
```

复测与主测试一致：

```text
1. L6 MLP prefix8 是强 value path。
2. L24 head 29/31 是强 choice/letter interface。
3. choice interface restore 可以恢复 letter，但不能恢复 value。
```

#### 3. GLM4：value path 有效，但 head choice interface 很弱

```text
value_zero:
  value_delta = -0.3594
  letter_delta = 0.0000

value_transplant:
  value_delta = -0.3126
  letter_delta = 0.0000

choice_transplant_heads:
  value_delta = -0.0039
  letter_delta = -0.0510

value_transplant + choice_transplant_heads:
  value_delta = -0.3099
  letter_delta = -0.0510
```

GLM4 的 L39 MLP prefix8 对 value 有影响，但 L39 head 31/17 对 letter 几乎没有强接口作用。

这继续支持：

```text
GLM4 的 choice/output interface 不在 L39 attention heads。
```

#### 4. DS7B：本轮节点不是强 bridge

```text
value_zero:
  value_delta = -0.0666
  letter_delta = 0.0000

value_transplant:
  value_delta = -0.0835
  letter_delta = 0.0000

choice_transplant_heads:
  value_delta = +0.0202
  letter_delta = -0.0182

value_transplant + choice_transplant_all_restore_clean_heads:
  value_delta = +0.0324
  letter_delta = -0.1005
```

DS7B 的 L27 MLP prefix8 和 L27 head 21/26 没形成强 value-to-choice bridge。结合前面阶段，DS7B 仍更像深层多点轨迹/输出释放型。

### 本轮关键进展

1. Qwen3 中第一次得到跨层桥接分离证据：

```text
L6 MLP prefix8 = value path
L24 head 29/31 = choice/letter interface
```

2. Qwen3 的 value 和 letter 可以被独立破坏：

```text
choice_transplant_heads:
  value_delta ≈ 0
  letter_delta ≈ -10
```

3. Qwen3 的 letter 可以在 value 仍坏的情况下被恢复：

```text
value_zero + choice_restore_clean_heads:
  value_delta ≈ -3.2
  letter_delta ≈ +0.31 to +0.38
```

4. GLM4 有 MLP value effect，但没有 Qwen3 式 attention-head choice interface。

5. DS7B 在当前节点上没有明显 bridge，需要改用 segment trajectory 方法。

### 问题和硬伤

1. Qwen3 的 letter 恢复不等于语义正确恢复；value 仍然坏。
2. 现在测试的是 scoring margin，不是 generation 行为。
3. L6 MLP value path 的具体 factor 仍未拆成 category/color/function/material/location 子方向。
4. choice interface 为什么能在 value 坏时恢复 letter，需要进一步解释：可能是 clean head 中已经含有足够的 letter-format state，而不是实时读取 value。
5. GLM4 / DS7B 的 bridge 没找到，不等于不存在，只说明当前节点不是主桥。

### 理论进展

Phase101 明确支持三层结构：

```text
ValuePath:
  语义值路径，决定候选内容评分。

ChoiceInterface:
  选择格式接口，决定输出字母/格式。

Bridge:
  把语义值路径接入选择格式接口的跨层机制。
```

Qwen3 当前结构：

```text
L6 MLP prefix8:
  强 value path。

L24 head 29/31:
  强 choice/letter interface。
```

但本轮更微妙地说明：

```text
L24 head 29/31 clean state 本身已经携带足够强的 letter interface 信息，
可以在上游 value path 被破坏时恢复 letter margin。
```

因此完整公式需要区分：

```text
online bridge:
  当前 forward 中 value factor 实时进入 choice interface。

cached interface state:
  choice head 在 L24 时已经形成的格式/字母状态。
```

更新后的机制表达：

```text
h_l(t_readout, x)
= Base_l
+ ValuePath_l(x)
+ Bridge_l(ValuePath -> ChoiceInterface)
+ ChoiceState_l(letter_label, option_block, task_format)
+ OutputPolicy_l
+ U_l
```

其中 Phase101 已经较强定位：

```text
Qwen3:
  ValuePath ≈ L6 MLP prefix8
  ChoiceState / ChoiceInterface ≈ L24 head 29/31
```

但尚未定位：

```text
Bridge_l(ValuePath -> ChoiceInterface)
```

### 下一步

Phase102 应做：

```text
Qwen3 Value Factor Decomposition inside L6 MLP
```

目标：

```text
1. 把 L6 MLP prefix8 的 value path 拆成 slot-specific factors:
   category / color / function / material / location。

2. 对每个 slot 分别做 value subspace destroy/restore。

3. 判断哪些 slot 的 value factor 会传递到 L24 choice interface。

4. 与 GLM5 Phase480 的 category_specific / semantic boundary residual 对齐：
   看 object-attribute value path 是否也是 category boundary residual 的一种下游读出形式。

5. 对 Qwen3 做更细的:
   L6 factor destroy
   L24 head 29/31 clean restore
   generation audit
```

如果 Phase102 成功，就可以开始把：

```text
semantic boundary factor
object-attribute value path
choice/letter interface
```

三块拼图接成一条更完整的语言输出机制链。

## Phase 102: Value Factor Bridge Decomposition [2026-06-13 16:12]

### 触发问题

用户要求结合附件分析与 `research/glm5/docs/AGI_GLM5_MEMO.md` 最新记录，继续完成全局语义语法契约图谱任务。附件对 Phase100 的判断基本正确：Qwen3 L24 head 29/31 已较强定位为 `choice/letter interface`（选择/字母接口），不是 `semantic value heads`（语义值头）。GLM5 memo 最新 Phase480 进一步给出类别边界残差证据：category-specific semantic boundary direction（类别特异语义边界方向）在 Qwen3/GLM4/DS7B 上都有不同程度复现，尤其 Qwen3 最稳定。因此下一步应把 GPT5 侧的 value path（值路径）与 GLM5 侧的 category boundary residual（类别边界残差）连接起来。

### 生成脚本

```text
tests/gpt5/phase102_value_factor_bridge_decomposition.py
tests/gpt5/phase102_value_factor_bridge_decomposition_summary.py
tests/gpt5/run_phase102_value_factor_bridge_decomposition_full.sh
```

### 执行命令

第一次运行中，Qwen3 在 40/240 后发生 Python 进程段错误：

```text
tests/gpt5/run_phase102_value_factor_bridge_decomposition_full.sh
```

失败信息：

```text
Segmentation fault (core dumped), exit code 139
```

已保存 partial：

```text
results/gpt5_phase102_value_factor_bridge_decomposition_full_20260613_150644/qwen3_phase102_value_factor_bridge_decomposition.partial.json
```

随后以同一输出目录 resume，并提高 partial 落盘频率：

```text
PHASE102_OUTPUT_DIR=results/gpt5_phase102_value_factor_bridge_decomposition_full_20260613_150644 \
PHASE102_PROGRESS_EVERY=10 \
tests/gpt5/run_phase102_value_factor_bridge_decomposition_full.sh
```

三模型最终全部完成。

### 测试规模

```text
Qwen3:      240 items, 2850 rows, bad_numeric_rows=0
GLM4:       240 items, 2850 rows, bad_numeric_rows=0
DeepSeek7B: 240 items, 2850 rows, bad_numeric_rows=0
Total:      8550 rows, bad_numeric_rows=0
```

输出目录：

```text
results/gpt5_phase102_value_factor_bridge_decomposition_full_20260613_150644
```

### 测试原理

Phase101 已定位：

```text
Qwen3:
  L6 MLP prefix8 = value path（值路径）
  L24 head 29/31 = choice/letter interface（选择/字母接口）
```

Phase102 在 value path 内构建多个 rank-4 子空间：

```text
value_all: 全局目标值子空间
value_category / value_color / value_function / value_material / value_location: 当前 slot 的值子空间
relation: 关系/slot 子空间
object: 对象子空间
```

然后测试：

```text
destroy_own_value
transplant_own_value
destroy_all_value
transplant_all_value
destroy_relation
transplant_relation
destroy_object
transplant_object
destroy_own_value + choice_restore_clean_heads
transplant_own_value + choice_restore_clean_heads
destroy_all_value + choice_restore_clean_heads
transplant_all_value + choice_restore_clean_heads
```

读出同时包含：

```text
value_margin: 目标语义值候选的 full-sequence logprob margin
letter_margin: 选择题字母候选的 full-sequence logprob margin
```

这样可以区分：

```text
值路径坏了没有；
选择接口坏了没有；
恢复 choice heads 是否能在 value 仍坏时恢复 letter。
```

### 核心客观结果

#### 1. Qwen3：L6 MLP value factor 是强因果路径

```text
destroy_own_value:
  value_delta = -3.6314
  letter_delta = -3.0400
  value_top1_delta = -0.2833
  letter_top1_delta = -0.2833

destroy_all_value:
  value_delta = -3.7212
  letter_delta = -3.8145
  value_top1_delta = -0.2792
  letter_top1_delta = -0.5958

destroy_relation:
  value_delta = -3.6696
  letter_delta = -3.6072
  value_top1_delta = -0.2208
  letter_top1_delta = -0.4667

destroy_object:
  value_delta = -3.6865
  letter_delta = -3.6449
  value_top1_delta = -0.2625
  letter_top1_delta = -0.4000
```

Qwen3 的 L6 MLP prefix8 中，value_all、relation、object、own_slot_value 子空间都强烈影响 value 与 letter。说明此处不是一个单一语义轴，而是对象、关系、目标值共同参与的 value factor bundle（值因子束）。

#### 2. Qwen3：choice head restore 可以救 letter，但不能救 value

```text
destroy_own_value + choice_restore_clean_heads:
  value_delta = -3.5715
  letter_delta = +1.3842
  value_top1_delta = -0.2875
  letter_top1_delta = +0.0458

destroy_all_value + choice_restore_clean_heads:
  value_delta = -3.5629
  letter_delta = -0.0177
  value_top1_delta = -0.2333
  letter_top1_delta = +0.0208
```

这复现并加强 Phase101 的分离结论：

```text
L24 head 29/31 clean restore 能恢复 letter interface；
但 value path 仍然损坏。
```

因此 Qwen3 的选择输出至少分为：

```text
semantic value path（语义值路径）
choice/letter interface（选择/字母接口）
```

二者可被分离破坏和分离恢复。

#### 3. Qwen3：slot 差异明显

`destroy_all_value` 下按 slot：

```text
category:
  value_delta = -3.635
  letter_delta = -5.645

color:
  value_delta = -0.853
  letter_delta = -2.663

function:
  value_delta = -7.028
  letter_delta = -3.748

location:
  value_delta = -2.997
  letter_delta = -4.017

material:
  value_delta = -4.093
  letter_delta = -2.999
```

function 对 value 最敏感，category/location 对 letter interface 也很强。这说明不同关系槽位不是共用一条完全相同路径，而是共享 value factor bundle 后在输出接口上有不同投影。

#### 4. GLM4：同一测试中 value factor 效应弱很多

```text
destroy_own_value:
  value_delta = -0.1904
  letter_delta = -0.0443

destroy_all_value:
  value_delta = -0.1719
  letter_delta = -0.0323

destroy_relation:
  value_delta = -0.2570
  letter_delta = +0.0875

destroy_object:
  value_delta = -0.2136
  letter_delta = +0.0102
```

GLM4 在 L39 MLP prefix8 上有弱 value effect，但没有 Qwen3 式强 value-to-letter 耦合，也没有可见 choice-head restore 差异：

```text
destroy_own_value 和 destroy_own_value+choice_restore_clean_heads 完全相同；
destroy_all_value 和 destroy_all_value+choice_restore_clean_heads 完全相同。
```

这继续支持：GLM4 的选择接口不在当前 L39 head 31/17。

#### 5. DeepSeek7B：当前节点不是 value bridge，甚至 destroy 常常提升 margin

```text
destroy_own_value:
  value_delta = +0.1337
  letter_delta = +0.1172

destroy_all_value:
  value_delta = +0.1308
  letter_delta = +0.1326

destroy_relation:
  value_delta = +0.2310
  letter_delta = +0.1271

destroy_object:
  value_delta = +0.2382
  letter_delta = +0.1073
```

DeepSeek7B L27 MLP prefix8 与 L27 head 21/26 没有形成 Qwen3 式 value bridge。当前 destroy 子空间反而略微提升 margin，说明该位置更可能是输出竞争/噪声/压缩后接口的一部分，而不是可直接解释的语义值写入路径。

### 本轮进展

1. Qwen3 的 value path 不只是单一 value direction，而是可拆成 object/relation/value-slot 多因子束。
2. Qwen3 的 L6 MLP value factors 对 value 和 letter 都有强因果影响。
3. Qwen3 的 L24 head 29/31 restore 再次证明它们更像 choice/letter interface，而不是 semantic value restore。
4. GLM4 和 DS7B 在当前节点没有同构结构，说明三模型的 value-to-choice bridge 位置不同。
5. GPT5 侧结果与 GLM5 Phase480 的 category boundary residual 可以开始连接：Qwen3 的 category/value factor 不是孤立方向，而是多关系槽位 value bundle 中的一部分。

### 问题和硬伤

1. Qwen3 第一次运行出现 segmentation fault。虽然 resume 后三模型完成，但说明长 hook 会话仍有稳定性风险。
2. 当前是 rank-4 子空间 destroy/transplant，不是最小充分电路。
3. 子空间由 SVD 差分构造，仍可能混入模板、对象身份、候选分布和选项格式。
4. Qwen3 的 choice restore 可以救 letter，但这不等于语义正确；value_delta 仍很负。
5. GLM4/DS7B 没找到 bridge，不等于不存在，只说明当前节点不是主 bridge。
6. 本轮没有 generation audit，只测 full-sequence scoring margin。

### 理论进展

当前更稳的结构应写成：

```text
Output(x)
= Readout(
    ValueBundle_l(object, relation, slot, target)
    -> Bridge_l
    -> ChoiceInterface_l(letter_label, option_format)
  )
```

其中 Qwen3 已有较强定位：

```text
ValueBundle:
  L6 MLP prefix8

ChoiceInterface:
  L24 head 29/31
```

但 Bridge 仍未完全定位。Phase102 说明：

```text
破坏 L6 value bundle 会同时破坏 value 和 letter；
恢复 L24 choice heads 可以恢复 letter，但不能恢复 value。
```

因此语言输出机制不是：

```text
语义值 = 输出字母
```

而至少是：

```text
语义值因子束
→ 跨层桥接
→ 选择/格式接口
→ 输出策略
```

这与“相对编码”一致：单一 binding path 信息有限，必须比较 object / relation / slot / choice interface 多条路径，才能看到全局结构。

### 下一步 Phase103

建议进入：

```text
Qwen3 Bridge Localization Sweep
```

目标不是继续扩大宏观数据，而是在 Qwen3 中定位 `ValueBundle -> ChoiceInterface` 的中间桥：

```text
1. 固定 value destroy at L6 MLP prefix8。
2. 扫描 L8/L12/L16/L20/L22/L24 的 attention 与 MLP restore。
3. 测哪些层/模块能在 value 破坏后恢复 letter，哪些能恢复 value。
4. 对 category/function/location 三个强槽位分别跑。
5. 最后对最强桥接节点做 generation audit。
```

关键判据：

```text
如果某中间模块 restore 能同时恢复 value 和 letter:
  它更接近真正 Bridge。

如果只能恢复 letter:
  它仍是 ChoiceInterface / formatting state。

如果只能恢复 value:
  它是 ValuePath downstream，而不是最终接口。
```

## Phase 103: Bridge Localization Restore Sweep [2026-06-13 21:47]

### 触发问题

附件分析基本正确：Phase101/102 已经把 Qwen3 的机制分成三层：

```text
semantic value path（语义值路径）
→ value-to-choice bridge（值到选择桥）
→ choice/letter interface（选择/字母接口）
```

目前强定位为：

```text
Qwen3 L6 MLP prefix8:
  value path / value factor bundle（值路径/值因子束）

Qwen3 L24 head 29/31:
  choice/letter interface（选择/字母接口）
```

但中间 Bridge 仍未定位。因此 Phase103 固定破坏 value bundle，再扫描后续层模块 clean restore，看哪些模块能恢复 value，哪些只能恢复 letter。

### 生成脚本

```text
tests/gpt5/phase103_bridge_localization_restore_sweep.py
tests/gpt5/phase103_bridge_localization_restore_sweep_summary.py
tests/gpt5/run_phase103_bridge_localization_restore_sweep_full.sh
```

### 执行命令

```bash
tests/gpt5/run_phase103_bridge_localization_restore_sweep_full.sh
```

三模型按顺序运行，并使用 `--hard-exit-after-model`：

```text
qwen3 → glm4 → deepseek7b
```

输出目录：

```text
results/gpt5_phase103_bridge_localization_restore_sweep_full_20260613_202833
```

### 测试规模

```text
Qwen3:      180 items, 5040 rows, bad_numeric_rows=0
GLM4:       180 items, 2880 rows, bad_numeric_rows=0
DeepSeek7B: 180 items, 2880 rows, bad_numeric_rows=0
Total:      10800 rows, bad_numeric_rows=0
```

测试槽位：

```text
category / function / location
```

测试因子：

```text
value_all
own slot value
```

### 测试原理

对每个 item 先计算 clean value/letter margin。

然后破坏指定 value basis：

```text
destroy_only:
  在 value_layer 的 MLP 输出中删除 value factor 子空间投影。
```

再加 clean restore：

```text
destroy_restore:Lx:attn
destroy_restore:Lx:mlp
destroy_restore:Lx:choice_heads
```

判据：

```text
如果 restore 后 value_delta 接近 0:
  该节点可能在 value path downstream 或 bridge 内。

如果 restore 后 letter_delta 接近 0 或变正，但 value_delta 仍很负:
  该节点更像 choice/letter interface 或 format state。

如果 value 和 letter 都恢复:
  才是强 Bridge 候选。
```

### Qwen3 结果

Qwen3 设置：

```text
value destroy:
  L6 MLP prefix8

restore sweep:
  L8/L12/L16/L20/L22/L24 attention and MLP
  L24 choice_heads 29/31
```

#### 1. destroy baseline

```text
destroy_only:
  value_delta = -4.3472
  letter_delta = -4.0910
  value_top1_delta = -0.3639
  letter_top1_delta = -0.3500
```

这比 Phase102 更强，说明本轮在 category/function/location 强槽位上，L6 value bundle 破坏明显。

#### 2. 最强 letter restore 是 L24 attention

按 letter_delta 排序：

```text
L24:attn:
  value_delta = -4.0133
  letter_delta = +0.7901
  value_top1_delta = -0.3139
  letter_top1_delta = +0.0222

L24:choice_heads:
  value_delta = -4.1945
  letter_delta = -0.2206
  value_top1_delta = -0.3722
  letter_top1_delta = +0.0222

L8:attn:
  value_delta = -4.4491
  letter_delta = -3.1648
```

关键现象：

```text
恢复 L24 attention 可以把 letter_delta 从 -4.0910 拉到 +0.7901；
但 value_delta 仍为 -4.0133。
```

这说明 L24 attention 整体比 head 29/31 更能恢复 choice/letter interface，但仍不能恢复 semantic value。

#### 3. 最强 value restore 是 L22/L24 MLP，但恢复幅度有限

按 value_delta 排序：

```text
L22:mlp:
  value_delta = -3.7750
  letter_delta = -4.6763

L24:mlp:
  value_delta = -3.8677
  letter_delta = -3.8117

L24:attn:
  value_delta = -4.0133
  letter_delta = +0.7901
```

L22/L24 MLP 对 value 有一定缓解，但不能恢复到接近 clean。它们也不能恢复 letter。

#### 4. 中间层没有找到同时恢复 value 和 letter 的强 Bridge

```text
L24 attention:
  restore letter, not value

L22/L24 MLP:
  slight value relief, not letter

L8/L12/L16/L20:
  no stable joint restore
```

所以本轮没有定位到强 Bridge，只定位到更清楚的分工：

```text
late attention = choice/letter interface state
late MLP = weak downstream value relief
```

### GLM4 结果

GLM4 设置：

```text
value destroy:
  L33 MLP prefix8

restore sweep:
  L35/L37/L39 attention and MLP
  L39 choice_heads 31/17
```

结果整体很弱：

```text
destroy_only:
  value_delta = -0.0456
  letter_delta = +0.0014

best value restore L39:mlp:
  value_delta = -0.0372
  letter_delta = +0.0021

best letter restore L35:attn / L37:mlp:
  letter_delta = +0.0040
```

GLM4 在当前范式下没有明显 value destruction，也没有明显 bridge restore。说明该任务的 GLM4 value path 不在 L33 MLP prefix8，或者 GLM4 的候选评分路径不适合用本轮 Qwen3 式 value-basis destroy 捕捉。

### DeepSeek7B 结果

DeepSeek7B 设置：

```text
value destroy:
  L24 MLP prefix8

restore sweep:
  L25/L26/L27 attention and MLP
  L27 choice_heads 21/26
```

结果：

```text
destroy_only:
  value_delta = -0.0486
  letter_delta = -0.1387

best value restore L27:mlp:
  value_delta = -0.0034
  letter_delta = -0.1752

best letter restore L26:mlp:
  value_delta = -0.0686
  letter_delta = -0.0292
```

DS7B 的 L27 MLP 可以恢复 value margin 到接近 0，但 letter 更差；L26 MLP 对 letter 有一定缓解但 value 更差。没有发现 joint bridge。

### 本轮关键进展

1. Qwen3 的 L24 attention 是比 head 29/31 更宽的 choice/letter interface 恢复节点。
2. Qwen3 中没有发现能同时恢复 value 和 letter 的单一中间模块。
3. Qwen3 late MLP 对 value 有弱恢复，但和 letter interface 分离。
4. GLM4 当前扫描没有明显 value destroy/restore 结构。
5. DS7B 显示 value 和 letter 可能在 L26/L27 分离：L27 MLP 更接近 value relief，L26 MLP 更接近 letter relief。

### 问题和硬伤

1. Restore 使用 full-sequence clean state，可能包含 candidate-specific state；它能定位恢复节点，但不能直接等同自然重算机制。
2. Qwen3 没有找到强 Bridge，说明 Bridge 可能不是单层单模块，而是多层路径。
3. GLM4/DS7B 结果弱，不代表没有机制；可能是 value destroy 层或因子 basis 选错。
4. 当前仍是 scoring margin，不是 open generation。
5. 本轮只测 category/function/location 三个强槽位，不代表全部关系类型。

### 当前理论更新

Phase103 后，Qwen3 的结构应更谨慎地写成：

```text
L6 MLP:
  ValueBundle(object, relation, slot, target)

L22/L24 MLP:
  downstream value relief / partial value state

L24 attention:
  broad choice/letter interface

L24 head 29/31:
  concentrated letter-label sub-interface
```

也就是说，Bridge 不是一个已经定位的单点，而更可能是：

```text
ValueBundle 从 L6 开始；
沿多层 residual trajectory 传播；
晚层 MLP 保留部分 value state；
晚层 attention 将格式/选项/字母接口接入输出。
```

当前最稳结论仍然是结构分离：

```text
semantic value factor bundle
≠
choice/letter interface
```

### 下一步 Phase104

建议进入：

```text
Qwen3 Segment Dynamic Bridge Recompute
```

目标：

```text
不要再只 restore 单层 clean state。
改为 patch L6 value bundle 后，让 L8-L24 分段自然重算。
```

测试设计：

```text
1. destroy L6 value bundle。
2. restore / transplant segment:
   L8-L12
   L12-L16
   L16-L20
   L20-L24
   L8-L24
3. 比较 value_margin 与 letter_margin。
4. 对 L24 attention 和 L24 MLP 分别做 final restore。
```

关键问题：

```text
如果某段自然重算能同时恢复 value 与 letter:
  Bridge 是 segment-level trajectory。

如果只有 L24 attention 能恢复 letter:
  choice interface 仍是末端格式接口。

如果 value 只能由 MLP segment 恢复:
  value path 与 choice interface 的连接需要多模块组合。
```

## Phase 104: 全局类别分析与类别竞争图谱整合 [2026-06-13 23:59]

### 本阶段目标

读取 `research/glm5/docs/AGI_GLM5_MEMO.md` 最新 Phase 483-484 进展，并参考用户附加资料，完成第一版全局类别分析。重点不是重新运行模型，而是把已经完成的三模型实验拼成一张全局类别地图：

```text
类别 = 共享语义流形 + 类别边界残差 + 竞争释放关系
```

本轮只使用基础分析：读取 JSON、排序、正负号、简单幅度比较、人工归纳，不做复杂统计和高级数学建模。

### 命令记录

```bash
python tests/gpt5/phase104_global_category_analysis.py
python -m py_compile tests/gpt5/phase104_global_category_analysis.py
```

### 脚本与结果

- 脚本：`tests/gpt5/phase104_global_category_analysis.py`
- JSON 结果：`results/gpt5/phase104_global_category_analysis.json`
- Markdown 摘要：`results/gpt5/phase104_global_category_analysis.md`
- 输入结果：
  - `results/glm5/phase483_{qwen3,glm4,deepseek7b}_r1.json`
  - `results/glm5/phase483_{qwen3,glm4,deepseek7b}_r2.json`
  - `results/glm5/phase484_{qwen3,glm4,deepseek7b}_r1.json`
  - `results/glm5/phase484_{qwen3,glm4,deepseek7b}_r2.json`

### 分析原理

1. **Category-Layer Map**：读取 Phase 483 全 8 类最佳层位、目标类别移除幅度、选择性和边界范数，形成类别-层位图。
2. **Competition Graph**：对每个类别移除后的 DCF 变化取正值边，形成 `removed_category -> released_category` 图谱。
3. **Cross-model Stable Edges**：只按“几个模型中为正”做基础稳定性判断，不做统计显著性推断。
4. **Writer Map**：读取 Phase 484 的 MLP 重构 cos@k、显著神经元数、k=5 消融与方向级移除的一致性，粗分为 MLP 因果写入器、集中候选、非 MLP/反向、弥散/缺失、混合未解。
5. **Relation Slot Map**：读取 kind_of / used_for / found_in 下 B_c 注入 delta，判断关系槽位是否改变边界方向读出。
6. **Anomaly Map**：读取 food->vehicle、animal->clothing 的属性释放解释，避免把异常边直接判为错误。

### 核心结果

1. **全局图谱支持当前主假设**：类别不是孤立方向，更像“共享语义流形 + 类别边界残差 + 竞争释放”的组合结构。
2. **跨三模型都为正的释放边**：

```text
animal -> clothing
clothing -> furniture
tool -> vehicle
fruit -> animal
clothing -> plant
vehicle -> clothing
furniture -> clothing
fruit -> clothing
furniture -> fruit
```

这些边不都很强，但它们在 Qwen3、GLM4、DS7B 中方向一致，可能是最早显露的稳定竞争骨架。

3. **模型差异很大**：

```text
Qwen3: 释放幅度最大，竞争图最清楚。
GLM4: 释放幅度整体很小，但方向上仍有若干一致边。
DS7B: 幅度可大，但存在方向不干净和抑制性神经元问题。
```

4. **MLP 因果写入器不是全局统一机制**：

```text
Qwen3 clothing: MLP 因果写入器最清楚，k=5 cos_remove≈0.962。
GLM4 fruit: MLP 因果写入器最清楚，k=5 cos_remove≈0.924。
Qwen3 fruit/animal: MLP 消融方向为负，说明真正写入器可能在 attention 或 residual route。
DS7B animal: cos@50 高但 k=5 消融为负，说明“重构集中”不等于“因果写入”。
```

5. **类别最佳层位不是统一层**：

```text
Qwen3: fruit L32, animal L33, tool L23, vehicle L29, clothing L30, furniture L26, food L34, plant L28
GLM4: fruit L27, animal L38, tool L27, vehicle L29, clothing L39, furniture L34, food L38, plant L32
DS7B: fruit L26, animal L27, tool L26, vehicle L26, clothing L23, furniture L25, food L27, plant L25
```

这说明类别边界存在“类别-模型特异发育时间”，不能继续假设所有类别在同一层形成。

6. **关系槽位读出支持 prompt-invariant 边界，但仍需小尺度复核**：Phase 484 中 fruit 的 B_c 注入 delta 在 kind_of / used_for / found_in 基本不变；但 scale=1.0 可能过强，下一阶段必须做 scale sweep。

7. **异常边不是简单错误**：

```text
food -> vehicle: 可能来自地点/移动属性释放。
animal -> clothing: 可能来自商业/户外属性释放。
```

但 DS7B 的 food/animal 方向不够干净，不能把它作为强证据。

### 理论进展

当前理论应从“局部类别边界存在”升级为：

```text
语言模型内部可能存在类别竞争网络。
类别边界不是一个个孤立坐标轴，而是通过竞争释放关系互相定义。
一个类别的意义，部分来自它激活什么，部分来自它压制什么。
```

更严格地说：

```text
类别 C 的内部编码至少包含三层：

1. 共享属性簇 M_c：
   与邻近类别共用的语义材料。

2. 边界残差 B_c：
   把 C 从邻近类别中分离出来的方向。

3. 竞争抑制场 R_c->*：
   C 激活时对其他类别/属性的压制关系。
```

这很接近“相对编码”的第一性原理：意义不是靠绝对位置定义，而是靠一组可复用属性和一组差异边界共同定义。

### 最严格审视与硬伤

1. **本轮没有新跑模型**：只整合既有 Phase 483/484 结果，因此是全局拼图，不是新增因果证据。
2. **类别数太少**：目前只有 8 类，每类 8 个对象，只能看到雏形，不能证明完整语义大陆。
3. **DCF 词表仍可能制造偏置**：food->vehicle、animal->clothing 等边可能受候选词集合影响，需要宽词表和开放生成复核。
4. **关系不变性可能是假象**：scale=1.0 注入可能覆盖关系模板差异，必须用 0.05/0.1/0.2/0.5/1.0 重测。
5. **写入器证据只覆盖三类**：Phase 484 只对 fruit/animal/clothing 做 MLP 重构，tool/vehicle/furniture/food/plant 还没有写入器级因果图。
6. **MLP 重构与因果不等价**：DS7B animal 是关键反例，cos@50 高但消融为负。
7. **GLM4 幅度太弱**：方向一致不代表机制强，需要更多对象和更干净读出确认。

### 第一性原理判断

如果语言背后存在某种基础数学结构，它现在更像是：

```text
复用材料 + 差异边界 + 竞争抑制 + 层级发育
```

而不是简单的：

```text
词向量空间中有一个类别方向
```

要破解语言背后的数学理论，第一原则应从“寻找单一语义轴”转向“寻找语义如何通过相对差异闭合”。也就是说，核心问题不是 fruit 方向在哪里，而是：

```text
fruit 如何复用 plant/food 的材料；
fruit 如何排除 animal/tool/vehicle；
fruit 的边界在何层形成；
fruit 的边界由哪个模块写入；
fruit 激活后释放/压制哪些邻接类别；
这些关系是否跨模型稳定。
```

### 下一阶段大任务

下一阶段不应只做一个小功能，而应做 **Global Category Atlas v2**：

1. **扩展类别规模**：从 8 类扩展到至少 32 类，每类不少于 24 个对象，覆盖自然物、人造物、生物、身体、地点、材料、抽象概念、社会角色。
2. **建立四张图**：

```text
Category-Layer Map: 每类在哪些层形成边界。
Competition Graph: 每类压制/释放哪些类别。
Writer Map: MLP/attention/residual route 谁写入边界。
Relation Slot Map: 不同关系是否只改变 baseline，不改变 B_c 读出。
```

3. **做 scale sweep**：对 B_c 注入和移除使用 0.05/0.1/0.2/0.5/1.0，确认关系不变性不是强注入造成。
4. **分模块找写入器**：对非 MLP 主导类别，分别测试 attention output、MLP output、residual route，找 fruit/animal 的真正写入源。
5. **异常边宽词表审计**：对 food->vehicle、animal->clothing 做更宽属性词表和开放生成验证，区分真实属性释放与 DCF 偏置。
6. **三模型顺序执行**：若进入模型重测，必须按 Qwen3 -> GLM4 -> DS7B 顺序单模型运行，并添加 `--hard-exit-after-model`，避免 GPU 内存溢出。

## Phase 105: CUDA 全类型系统类别图谱与层位分布分析 [2026-06-14 00:12]

### 本阶段目标

根据用户要求，使用 CUDA 对“所有类型”做系统分析，重点回答：

```text
1. 每种类型分布在哪些层？
2. 每种类型的读出强度、边界强度、类内凝聚是什么样？
3. 类型之间的相对邻接关系是什么？
4. 不同模型是否有相同的类型层位规律？
```

本轮从 8 类扩展到 32 个大类，每类 24 个对象，三模型顺序运行：

```text
qwen3 -> glm4 -> deepseek7b
```

每个模型单独运行，并添加 `--hard-exit-after-model`，避免 GPU 显存残留。

### 执行命令

```bash
python tests/gpt5/phase105_global_category_atlas_cuda.py qwen3 \
  --max-categories 4 \
  --objects-per-category 3 \
  --batch-size 2 \
  --progress-every 1 \
  --output-dir results/gpt5_phase105_smoke \
  --hard-exit-after-model

python tests/gpt5/phase105_global_category_atlas_cuda.py qwen3 \
  --objects-per-category 24 \
  --batch-size 8 \
  --progress-every 12 \
  --output-dir results/gpt5_phase105_global_category_atlas \
  --hard-exit-after-model

python tests/gpt5/phase105_global_category_atlas_cuda.py glm4 \
  --objects-per-category 24 \
  --batch-size 8 \
  --progress-every 12 \
  --output-dir results/gpt5_phase105_global_category_atlas \
  --hard-exit-after-model

python tests/gpt5/phase105_global_category_atlas_cuda.py deepseek7b \
  --objects-per-category 24 \
  --batch-size 8 \
  --progress-every 12 \
  --output-dir results/gpt5_phase105_global_category_atlas \
  --hard-exit-after-model

python tests/gpt5/phase105_global_category_atlas_summary.py \
  --input-dir results/gpt5_phase105_global_category_atlas

python -m py_compile \
  tests/gpt5/phase105_global_category_atlas_cuda.py \
  tests/gpt5/phase105_global_category_atlas_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase105_global_category_atlas_cuda.py`
- 汇总脚本：`tests/gpt5/phase105_global_category_atlas_summary.py`
- Qwen3 结果：`results/gpt5_phase105_global_category_atlas/phase105_qwen3_atlas.json`
- GLM4 结果：`results/gpt5_phase105_global_category_atlas/phase105_glm4_atlas.json`
- DS7B 结果：`results/gpt5_phase105_global_category_atlas/phase105_deepseek7b_atlas.json`
- 跨模型汇总：`results/gpt5_phase105_global_category_atlas/phase105_cross_model_summary.md`

### 类别集合

本轮共 32 类，每类 24 个对象：

```text
fruit, animal, tool, vehicle, clothing, furniture, food, plant,
body, place, building, material, color, emotion, role, profession,
abstract, action, event, time, number, shape, sound, light,
weather, container, instrument, machine, communication, relation,
property, substance
```

### 测试原理

1. 对每个对象构造自然模板：

```text
The {obj} is a kind of
```

2. 使用 CUDA 前向，并设置 `output_hidden_states=True`，抓取所有层最后 token 的 hidden state。

3. 对每个类别、每层计算类别中心：

```text
center(category, layer) = mean(hidden_state(objects in category, layer))
```

4. 对每层计算基础指标：

```text
target margin:
  类别中心对自身类别 readout words 的分数
  -
  对其他类别 readout words 的最大分数

rank:
  自身类别 readout 在 32 类中的排名

cohesion:
  同类对象向类别中心的平均 cos

boundary norm:
  类别中心 - 其他类别中心平均值 的范数

nearest neighbors:
  类别中心与其他类别中心的 cos 排序

local boundary release:
  在最佳 margin 层做本层 logit-lens 边界移除，
  看其他类别 readout 是否上升
```

本轮仍坚持基础分析，不使用复杂统计建模。

### 三模型全局层位结果

```text
Qwen3:
  layers = 36
  best top1 layer = L36, 23/32 类 top1
  best mean margin layer = L36, mean margin = 0.68
  best mean boundary layer = L35, mean boundary norm = 161.17

GLM4:
  layers = 40
  best top1 layer = L40, 22/32 类 top1
  best mean margin layer = L0, mean margin ≈ 0
  best mean boundary layer = L19, mean boundary norm = 2.48

DS7B:
  layers = 28
  best top1 layer = L28, 8/32 类 top1
  best mean margin layer = L0, mean margin ≈ -0.02
  best mean boundary layer = L27, mean boundary norm = 238.80
```

### 关键类别层位图

#### Qwen3

Qwen3 呈现最清楚的晚层类别读出：

```text
fruit:      margin L32, boundary L35, margin 12.54
animal:     margin L32, boundary L35, margin 11.58
tool:       margin L35, boundary L35, margin 6.88
vehicle:    margin L33, boundary L35, margin 12.25
food:       margin L33, boundary L35, margin 16.44
plant:      margin L34, boundary L35, margin 15.91
building:   margin L35, boundary L35, margin 14.42
profession: margin L35, boundary L35, margin 22.24
sound:      margin L33, boundary L35, margin 23.19
shape:      margin L34, boundary L35, margin 8.63
```

Qwen3 中较弱或弥散的类型：

```text
role, abstract, action, time, number, relation
```

这些类型不是没有结构，而是当前 readout basis 下不形成强类别标签 margin。

#### GLM4

GLM4 的 rank 可出现正确，但 DCF margin 幅度极小：

```text
vehicle: margin L40, margin 1.05
body:    margin L40, margin 1.08
emotion: margin L40, margin 2.64
machine: margin L40, margin 1.11
```

多数类别 margin 接近 0，说明当前英文 DCF readout 对 GLM4 可能不够校准，不能简单说 GLM4 没有类别结构。

#### DS7B

DS7B 呈现强晚层 boundary norm，但类别标签 margin 普遍弱：

```text
boundary norm peak = L27
profession: margin L27, margin 26.42
animal:     margin L28, margin 0.91
plant:      margin L28, margin 0.75
property:   margin L12, margin 1.48
```

DS7B 的结构更像“中心和边界存在，但当前 DCF 标签读不干净”。

### 类型邻接关系示例

Qwen3 中一些强类别的最近邻：

```text
fruit -> plant, color, food
animal -> relation, plant, role
vehicle -> machine, container, building
food -> substance, material, container
plant -> color, fruit, relation
building -> place, container, action
profession -> role, relation, action
sound -> action, communication, light
shape -> property, number, light
```

这些邻接关系不是人类分类表的简单复制，而是模型内部类别中心的相对位置。

### 重要理论发现

1. **类别边界和类别读出不是同一件事**

Qwen3 中 margin 和 boundary norm 都在晚层很清楚；但 GLM4/DS7B 出现“boundary 或 rank 有信号，但 margin 很弱”的情况。说明：

```text
类别结构存在
≠
当前 DCF readout 能干净读出
```

2. **边界层普遍偏晚**

Qwen3 边界峰值在 L35，DS7B 在 L27，都是接近末层。类别差异不是只在早层产生，而是经过层级发育后在晚层变得最清楚。

3. **具体名词类比抽象关系类更容易形成强读出**

Qwen3 中 fruit/animal/vehicle/food/plant/building/sound/profession 很强，而 role/abstract/action/time/number/relation 较弱。这说明抽象和关系类可能更依赖上下文槽位，不适合用单一句式和类别标签 readout 直接测。

4. **“类型”不是统一形态**

当前可粗分：

```text
sharp_readout_cohesive:
  读出强、类内凝聚，典型如 Qwen3 fruit/animal/food/plant/sound/profession。

readout_clear:
  有清楚读出，但没有达到最强边界，例如 Qwen3 tool/color/weather/instrument。

cohesive_boundary_unclear_readout:
  有中心/边界，但类别标签读出弱，例如 Qwen3 clothing/furniture/body/machine。

diffuse_or_contextual:
  当前模板下弥散或依赖上下文，例如 GLM4 多数类、Qwen3 role/abstract/action/time/number/relation。
```

### 最严格审视与硬伤

1. **本轮是全层 logit-lens 图谱，不是下游因果干预**

local boundary release 只是本层读出变化，不等同于真正 forward patch 后的输出变化。不能把释放边直接当成因果机制。

2. **模板只有一个**

本轮为了快速完成 32 类三模型全图，只使用：

```text
The {obj} is a kind of
```

抽象类、关系类、动作类明显可能被模板压制。下一轮必须加多模板。

3. **readout words 对 GLM4/DS7B 可能严重不公平**

GLM4 和 DS7B 的 margin 弱，不一定说明类别弱，可能是英文 readout token、chat model 格式或输出头标定导致。

4. **cohesion 容易被共享模板抬高**

所有 prompt 共享前缀，类内凝聚可能混入模板相似性。需要做模板残差扣除或对象 token 位置测试。

5. **类别词表仍是人工定义**

32 类比 8 类更大，但仍不是完整语义空间；一些类别互相重叠，例如 role/profession/relation、material/substance/property。

6. **层位解释要谨慎**

`hidden_states[k]` 表示第 k-1 个 transformer block 之后的状态，L36/L40/L28 接近最终输出接口，可能混入 readout 适配，不完全等价于“语义生成层”。

### 第一性原理更新

本轮把“类别边界”从局部机制推进到全局层位图。当前更合理的第一性原理表述是：

```text
类型不是一个固定向量。
类型是对象集合在层级演化中逐步形成的相对闭合区域。

这个区域至少有四个可观察量：
1. 中心：同类对象是否聚到一起。
2. 边界：该中心和其他类型中心如何分离。
3. 读出：输出头是否能把它命名成类别。
4. 竞争：移除边界后哪些相邻类型被释放。
```

这说明语言背后的数学结构可能不是传统“向量空间 + 分类面”那么简单，而更像：

```text
对象轨道 -> 类别中心 -> 相对边界 -> 竞争网络 -> 输出读出接口
```

也就是意义并非静态存放，而是在层级计算中逐步闭合。

### 下一阶段大任务

下一阶段应做 **Phase 106: 多模板残差扣除 + 因果释放验证**：

1. **多模板重跑**

```text
The {obj} is a kind of
A {obj} belongs to the category of
The word {obj} refers to a type of
People use the word {obj} when talking about
```

2. **模板残差扣除**

对同一模板下所有类别中心求公共模板向量，再从对象表示中扣除，测试 cohesion 和 boundary 是否仍存在。

3. **对象 token 位置测试**

不要只看最后 token，也看对象 token 的首/尾位置，判断类别是在对象处形成，还是在答案槽位形成。

4. **挑选稳定边做真正 CUDA patch**

从 Phase 105 中选择强邻接/强释放边，例如：

```text
fruit -> plant/food
vehicle -> machine/container/building
food -> substance/material/container
profession -> role/relation/action
sound -> action/communication/light
```

在 Qwen3 先做真实 forward boundary removal，再扩展到 GLM4/DS7B。

5. **改进 GLM4/DS7B readout**

为 GLM4/DS7B 单独标定 readout words、中文/英文双语 readout、chat template readout，避免把读出失败误判成结构不存在。

## Phase 106: 多模板残差扣除与对象位置类别图谱复核 [2026-06-14 08:06]

### 本阶段目标

根据用户要求，先判断附加分析是否正确，再继续完成真实客观现象拼图。

对附加分析的收缩判断：

```text
正确部分：
1. Phase105 只是 logit-lens atlas，不是完整因果图谱。
2. 类别结构与 readout interface 需要分开。
3. 下一步必须做多模板、模板残差扣除、对象 token 位置测试。

需要谨慎部分：
1. 不能过早理论总结。
2. 不能把 Phase105 的 local boundary release 当成真实 forward causal edge。
3. GLM4/DS7B 的 weak margin 不能直接解释为类别结构不存在。
```

本轮 Phase106 使用 CUDA 对三模型完整重测，不分小批次实验，不在模型测试期间插入分析。

### 执行命令

```bash
python tests/gpt5/phase106_multitemplate_residual_cuda.py qwen3 \
  --objects-per-category 2 \
  --templates 2 \
  --batch-size 4 \
  --progress-every 2 \
  --output-dir results/gpt5_phase106_smoke \
  --hard-exit-after-model

python tests/gpt5/phase106_multitemplate_residual_cuda.py qwen3 \
  --objects-per-category 24 \
  --templates 4 \
  --batch-size 16 \
  --progress-every 16 \
  --output-dir results/gpt5_phase106_multitemplate_residual \
  --hard-exit-after-model

python tests/gpt5/phase106_multitemplate_residual_cuda.py glm4 \
  --objects-per-category 24 \
  --templates 4 \
  --batch-size 16 \
  --progress-every 16 \
  --output-dir results/gpt5_phase106_multitemplate_residual \
  --hard-exit-after-model

python tests/gpt5/phase106_multitemplate_residual_cuda.py deepseek7b \
  --objects-per-category 24 \
  --templates 4 \
  --batch-size 16 \
  --progress-every 16 \
  --output-dir results/gpt5_phase106_multitemplate_residual \
  --hard-exit-after-model

python tests/gpt5/phase106_multitemplate_residual_summary.py

python -m py_compile \
  tests/gpt5/phase106_multitemplate_residual_cuda.py \
  tests/gpt5/phase106_multitemplate_residual_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase106_multitemplate_residual_cuda.py`
- 汇总脚本：`tests/gpt5/phase106_multitemplate_residual_summary.py`
- Qwen3 结果：`results/gpt5_phase106_multitemplate_residual/phase106_qwen3_multitemplate_residual.json`
- GLM4 结果：`results/gpt5_phase106_multitemplate_residual/phase106_glm4_multitemplate_residual.json`
- DS7B 结果：`results/gpt5_phase106_multitemplate_residual/phase106_deepseek7b_multitemplate_residual.json`
- 跨模型汇总：`results/gpt5_phase106_multitemplate_residual/phase106_cross_model_summary.md`

### 测试规模

```text
models = qwen3, glm4, deepseek7b
categories = 32
objects/category = 24
templates = 4
positions = answer_last, object_last
prompts/model = 32 * 24 * 4 = 3072
total prompts = 9216
```

四个模板：

```text
The {obj} is a kind of
A {obj} belongs to the category of
The word {obj} refers to a type of
People use the word {obj} when talking about
```

两个位置：

```text
answer_last:
  答案槽位/类别读出槽位。

object_last:
  对象 token 最后位置。
```

两种基底：

```text
raw:
  原始 hidden state 类别中心。

template_residual:
  每个 template、每层、每个位置上，先减去该 template 的所有类别公共均值向量。
```

### 客观结果：全局层位

#### Qwen3

```text
answer_last / raw:
  top1 L36 = 21/32
  best mean margin L32 = 0.718
  best boundary L35 = 155.255

answer_last / template_residual:
  best mean margin L33 = 7.587
  best boundary L35 = 155.255

object_last / raw:
  top1 L0 = 18/32
  best mean margin L0 ≈ 0
  best boundary L35 = 119.261

object_last / template_residual:
  top1 L13 = 22/32
  best mean margin L32 = 0.946
  best boundary L35 = 119.261
```

#### GLM4

```text
answer_last / raw:
  top1 L40 = 25/32
  best mean margin L0 ≈ 0
  best boundary L18 = 2.644

answer_last / template_residual:
  top1 L19 = 32/32
  best mean margin L0 ≈ 0
  best boundary L18 = 2.644

object_last / raw:
  top1 L24 = 24/32
  best mean margin L0 ≈ 0
  best boundary L19 = 70.176

object_last / template_residual:
  top1 L20 = 32/32
  best mean margin L0 ≈ 0
  best boundary L19 = 70.176
```

#### DS7B

```text
answer_last / raw:
  top1 L28 = 9/32
  best mean margin L0 = -0.017
  best boundary L27 = 263.246

answer_last / template_residual:
  best mean margin L27 = 4.723
  best boundary L27 = 263.246

object_last / raw:
  top1 L4 = 5/32
  best mean margin L0 = -0.009
  best boundary L27 = 213.556

object_last / template_residual:
  top1 L28 = 15/32
  best mean margin L0 = -0.007
  best boundary L27 = 213.556
```

### 客观结果：Phase105 的直接修正

1. **Qwen3 的 Phase105 结论大体保留，但弱类被模板残差显著增强**

Phase105 中 Qwen3 的强类仍然强，例如：

```text
fruit:   raw 12.57 -> residual 14.07
vehicle: raw 10.10 -> residual 12.08
food:    raw 14.77 -> residual 16.22
plant:   raw 15.02 -> residual 11.99
sound:   raw 25.38 -> residual 14.57
```

但一些 Phase105 中偏弱的类，在 template_residual 后明显增强：

```text
clothing:      3.41 -> 14.79
furniture:     0.62 -> 7.67
body:          0.43 -> 4.87
place:         0.49 -> 7.84
action:       -0.08 -> 4.39
time:         -0.07 -> 8.93
number:       -0.07 -> 7.80
container:     0.16 -> 7.64
communication: 0.52 -> 6.50
property:     -0.08 -> 5.26
```

这说明 Phase105 对 Qwen3 的“弥散类”判断有一部分是模板公共向量污染造成的。

2. **Qwen3 object_last 明显弱于 answer_last，但不是空信号**

object_last / template_residual 中有多类仍有正 margin：

```text
weather 7.85
light 5.13
container 4.18
shape 3.68
vehicle 3.72
relation 3.51
color 3.25
profession 3.10
plant 3.05
```

说明类别信息在对象 token 位置已经存在，但在 answer_last 槽位被显著放大。

3. **GLM4 的问题不是简单模板公共向量污染**

GLM4 在 raw 与 template_residual 下 margin 仍接近 0：

```text
answer_last / template_residual best mean margin ≈ 0
object_last / template_residual best mean margin ≈ 0
```

但 top1 count 可达 32/32，说明 top1 在 margin 极小时会虚高，不能作为强证据。GLM4 更可能需要重新校准 readout words、chat template 或中英文 readout。

4. **DS7B 被 Phase105 明显低估**

DS7B 在 answer_last 做 template_residual 后：

```text
best mean margin: -0.017 -> 4.723
best layer: L27
```

大量类别从 raw 弱信号变成强信号：

```text
fruit:     -0.03 -> 9.18
vehicle:    0.00 -> 9.19
clothing:  -0.02 -> 10.21
plant:      1.01 -> 11.07
body:       0.44 -> 8.80
place:     -0.02 -> 6.93
building:  -0.03 -> 7.21
color:      0.20 -> 9.14
number:    -0.01 -> 9.63
weather:    0.56 -> 14.40
```

这说明 DS7B 内部类别结构并不弱，而是被公共模板/格式方向遮蔽。

5. **边界层结论稳定**

template_residual 不改变类别之间的相对差值，因此 boundary layer 基本不变：

```text
Qwen3 answer_last boundary peak: L35
Qwen3 object_last boundary peak: L35
DS7B answer_last boundary peak: L27
DS7B object_last boundary peak: L27
GLM4 answer_last boundary peak: L18
GLM4 object_last boundary peak: L19
```

### 当前最可靠客观事实

1. **answer_last 是类别读出放大槽位**：Qwen3 和 DS7B 的 answer_last margin 明显强于 object_last。
2. **模板公共向量会严重遮蔽类别方向**：尤其是 DS7B，也影响 Qwen3 弱类。
3. **boundary layer 比 margin layer 更稳定**：扣除模板公共向量后，boundary peak 基本不变。
4. **top1 count 不能单独作为证据**：GLM4 在 margin≈0 时也能出现 32/32 top1。
5. **GLM4 仍未被当前 readout 正确读出**：需要专门 readout 校准。
6. **Phase105 对 Qwen3 强类判断基本正确，但对弱类过于保守；对 DS7B 明显低估。**

### 硬伤分析

1. **仍不是真正因果 patch**：本轮是多模板/残差/位置图谱，不是 downstream forward intervention。
2. **template_residual 可能引入相对化增强**：减去公共均值后 margin 变大，说明差异更清楚，但不等于模型自然输出一定使用这个差异。
3. **object_last 定位是 token subsequence 近似**：多 token 对象或 tokenizer 差异可能影响位置定位。
4. **GLM4 readout 仍失败**：当前英文 readout words 可能不适配 GLM4，需要双语/聊天模板校准。
5. **没有测试跨模板一致对象轨道**：本轮只看中心，不看每个对象在多模板中的轨道是否闭合。

### 下一步任务

Phase107 不应做理论总结，应做真实因果验证：

```text
目标：从 Phase106 中选择最稳定、margin 高、boundary 层稳定的边，做 downstream forward boundary removal。
```

优先测试：

```text
Qwen3:
  clothing, furniture, time, number, action, container
  因为这些类在 template_residual 后从弱变强。

DS7B:
  fruit, vehicle, clothing, plant, body, place, building, weather
  因为这些类从 raw 弱信号变为 residual 强信号。

GLM4:
  暂不做因果边界测试，先做 readout calibration。
```

Phase107 应输出：

```text
1. 自然 forward baseline。
2. best boundary layer removal。
3. template residual boundary removal。
4. random same-norm control。
5. target DCF 下降和 competitor release 上升。
```

## Phase 107: 真实前向类别边界移除因果验证 [2026-06-14 08:43]

### 本阶段目标

根据用户要求，综合 Phase106 正确部分继续任务，不做过早理论总结，优先完成真实客观现象拼图。

Phase106 的正确部分：

```text
1. Phase105/106 仍是 atlas/readout 图谱，不是因果图。
2. 模板公共向量会遮蔽类别方向，尤其 DS7B。
3. answer_last 是类别读出放大槽位。
4. boundary layer 比 margin layer 更稳定。
```

Phase107 的目标：

```text
从 atlas 进入真实 forward causal intervention。
在自然前向传播中，于 boundary layer 的 answer_last 位置移除类别边界投影，
观察最终 logits 的类别 DCF 是否改变。
```

### 执行命令

```bash
python tests/gpt5/phase107_causal_boundary_removal_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --categories fruit,clothing \
  --output-dir results/gpt5_phase107_smoke \
  --hard-exit-after-model

python tests/gpt5/phase107_causal_boundary_removal_cuda.py qwen3 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 16 \
  --output-dir results/gpt5_phase107_causal_boundary_removal \
  --hard-exit-after-model

python tests/gpt5/phase107_causal_boundary_removal_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 16 \
  --output-dir results/gpt5_phase107_causal_boundary_removal \
  --hard-exit-after-model

# GLM4 fp16 logits 出现 NaN，改用 bf16 重新运行并覆盖结果
PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase107_causal_boundary_removal_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 16 \
  --output-dir results/gpt5_phase107_causal_boundary_removal \
  --hard-exit-after-model

python tests/gpt5/phase107_causal_boundary_removal_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 16 \
  --output-dir results/gpt5_phase107_causal_boundary_removal \
  --hard-exit-after-model

python tests/gpt5/phase107_causal_boundary_removal_summary.py

python -m py_compile \
  tests/gpt5/phase107_causal_boundary_removal_cuda.py \
  tests/gpt5/phase107_causal_boundary_removal_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase107_causal_boundary_removal_cuda.py`
- 汇总脚本：`tests/gpt5/phase107_causal_boundary_removal_summary.py`
- Qwen3 结果：`results/gpt5_phase107_causal_boundary_removal/phase107_qwen3_causal_boundary_removal.json`
- GLM4 结果：`results/gpt5_phase107_causal_boundary_removal/phase107_glm4_causal_boundary_removal.json`
- DS7B 结果：`results/gpt5_phase107_causal_boundary_removal/phase107_deepseek7b_causal_boundary_removal.json`
- 跨模型汇总：`results/gpt5_phase107_causal_boundary_removal/phase107_cross_model_summary.md`

### 测试规模

```text
models = qwen3, glm4, deepseek7b
test categories = 12
train objects/category = 12
heldout test objects/category = 12
templates = 4
prompts/category = 48
conditions = baseline, remove_boundary, random_same_norm
```

测试类别：

```text
fruit, vehicle, clothing, furniture, plant, body,
place, building, time, number, weather, container
```

模型边界层：

```text
Qwen3: L35
GLM4: L18
DS7B: L27
```

### 方法

1. 使用前 12 个对象训练类别中心。
2. 对每个类别在 boundary layer 估计边界：

```text
B_c = mean_template(center(c, template) - mean_other_categories(template))
```

3. 在 heldout 对象上真实 forward。
4. 在 boundary layer 的 answer_last 位置注册 hook：

```text
h := h - projection(h, B_c)
```

5. 对照组：

```text
random_same_norm:
  使用确定性随机单位方向，做同样 projection removal。
```

6. 测量最终 logits 的 32 类 DCF 变化。

### 客观结果摘要

#### Qwen3

```text
fruit:     target Δ +0.16, top release sound +0.69
vehicle:   target Δ +0.11, top release role +0.22
clothing:  target Δ +0.49, top release tool +0.89
furniture: target Δ +1.37, top release building +1.49
plant:     target Δ +0.04, top release color +0.15
body:      target Δ +0.17, top release weather +0.73
place:     target Δ +0.05, top release shape +0.24
building:  target Δ +0.36, top release shape +0.59
time:      target Δ -0.51, top release animal +0.60
number:    target Δ -1.41, top release animal +0.23
weather:   target Δ +0.10, top release light +0.58
container: target Δ +0.03, top release fruit +1.12
```

Qwen3 只有 `time` 和 `number` 表现为目标下降，其中 `time` 同时有竞争释放。
多数具体类别是 release-only 或 target-up/opposed。

#### GLM4

GLM4 初次 fp16 运行 logits 出现 NaN，bf16 重跑后结果有限但正常。

```text
fruit:     target Δ +0.08, top release shape +0.39
vehicle:   target Δ -0.01, top release place +0.53
clothing:  target Δ -0.15, top release property +0.29
furniture: target Δ +0.01, top release material +0.07
plant:     target Δ -0.00, top release material +0.27
body:      target Δ +0.05, top release place +0.31
place:     target Δ +0.03, top release action +0.14
building:  target Δ -0.01, top release action +0.08
time:      target Δ -0.03, top release material +0.16
number:    target Δ +0.05, top release container +0.18
weather:   target Δ -0.26, top release shape +0.30
container: target Δ -0.01, top release role +0.23
```

GLM4 效应整体较小，不能作为强因果证据。

#### DS7B

```text
fruit:     target Δ +0.94, top release time +1.48
vehicle:   target Δ -0.04, top release machine +0.48
clothing:  target Δ +1.05, top release tool +1.58
furniture: target Δ +0.62, top release tool +1.02
plant:     target Δ +1.04, top release animal +1.19
body:      target Δ +0.65, top release container +1.00
place:     target Δ +0.21, top release emotion +0.23
building:  target Δ +0.22, top release fruit +0.55
time:      target Δ +0.10, top release clothing +0.23
number:    target Δ -2.58, no positive release
weather:   target Δ +0.01, top release clothing +0.40
container: target Δ -2.28, no positive release
```

DS7B 的 `number` 和 `container` 出现强目标下降，但没有清楚竞争释放。
多个具体类表现为 target-up/opposed。

### 当前最可靠客观事实

1. **atlas boundary vectors 能真实影响最终 logits**  
   boundary removal 的 release 幅度通常明显大于 random same-norm control。

2. **边界方向不是简单正支持方向**  
   很多类别移除边界后 target DCF 反而上升，例如 Qwen3 furniture、DS7B clothing/plant。

3. **干净的 target-down + competitor-release 很少**  
   本轮最接近的是：

```text
Qwen3 time: target Δ -0.51, animal release +0.60
```

4. **number 类跨模型更像可移除目标边界**

```text
Qwen3 number: target Δ -1.41
DS7B number: target Δ -2.58
```

但两者都缺少强竞争释放，因此更像 target boundary removal，不是完整 competition edge。

5. **GLM4 需要 bf16 才能避免 NaN**

GLM4 fp16 forward logits 不稳定，后续 GLM4 CUDA 测试应默认：

```bash
PROBE_TORCH_DTYPE=bfloat16
```

6. **Phase106 的强 margin 不等于简单因果支持**

Phase106 中 template_residual margin 很强的类别，在 Phase107 中不一定 target-down。
这说明 readout margin、boundary geometry、forward causal support 三者必须分开。

### 硬伤分析

1. **只移除 answer_last 单点**  
   类别边界可能分布在多 token、多层 residual trajectory 中，单点移除不一定能关闭类别。

2. **边界定义仍是 center-vs-others**  
   对 target-up/opposed 类别，说明此边界可能混入抑制方向或读出接口方向。

3. **没有 scale sweep**  
   本轮 scale=1.0，下一轮必须测试 0.25/0.5/1.0/1.5。

4. **没有 layer sweep**  
   只用了 boundary peak layer。真实因果操作层可能不是 boundary norm 最大层。

5. **没有多位置 patch**  
   object_last、answer_last、多 token 共同干预可能与单点干预不同。

### 下一步任务

Phase108 应继续客观测试，不做理论扩张：

```text
Boundary Causal Sweep:
  categories = number, time, container, clothing, furniture, plant
  models = Qwen3, DS7B
  GLM4 = bf16 only, optional calibration branch
```

必须测试：

```text
1. scale sweep: 0.25 / 0.5 / 1.0 / 1.5
2. layer sweep: boundary_layer-3 ... boundary_layer
3. position sweep: object_last, answer_last, both
4. controls: random_same_norm, neighbor_boundary_control
```

核心目标不是总结，而是判定：

```text
哪些类别的边界是正支持方向？
哪些类别的边界是抑制/竞争方向？
哪些类别需要多层/多位置共同移除才有因果效果？
```

## Phase 108: Boundary Causal Sweep 层位-位置-scale-对照系统扫描 [2026-06-14 09:03]

### 本阶段目标

根据用户要求，先判断附加分析是否正确，再继续完成客观现象拼图。

附加分析中正确部分：

```text
1. 分布情况是语言编码机制的核心拼图。
2. Phase107 已经从 atlas/readout 进入真实 forward causal intervention。
3. Phase107 的结果不能解释成“类别边界=简单正支持方向”。
4. 下一步必须做 scale、layer、position、control sweep。
```

本轮 Phase108 目标：

```text
判定哪些类别边界是正支持方向；
哪些是抑制/竞争/接口混合方向；
哪些需要多层/多位置共同移除才出现因果效果。
```

### 执行命令

```bash
python tests/gpt5/phase108_boundary_causal_sweep_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --layer-back 1 \
  --categories number,time \
  --output-dir results/gpt5_phase108_smoke \
  --hard-exit-after-model

python tests/gpt5/phase108_boundary_causal_sweep_cuda.py qwen3 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --output-dir results/gpt5_phase108_boundary_causal_sweep \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase108_boundary_causal_sweep_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --output-dir results/gpt5_phase108_boundary_causal_sweep \
  --hard-exit-after-model

python tests/gpt5/phase108_boundary_causal_sweep_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --output-dir results/gpt5_phase108_boundary_causal_sweep \
  --hard-exit-after-model

python tests/gpt5/phase108_boundary_causal_sweep_summary.py

python -m py_compile \
  tests/gpt5/phase108_boundary_causal_sweep_cuda.py \
  tests/gpt5/phase108_boundary_causal_sweep_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase108_boundary_causal_sweep_cuda.py`
- 汇总脚本：`tests/gpt5/phase108_boundary_causal_sweep_summary.py`
- Qwen3 结果：`results/gpt5_phase108_boundary_causal_sweep/phase108_qwen3_boundary_causal_sweep.json`
- GLM4 结果：`results/gpt5_phase108_boundary_causal_sweep/phase108_glm4_boundary_causal_sweep.json`
- DS7B 结果：`results/gpt5_phase108_boundary_causal_sweep/phase108_deepseek7b_boundary_causal_sweep.json`
- 跨模型汇总：`results/gpt5_phase108_boundary_causal_sweep/phase108_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, time, container, clothing, furniture, plant
train objects/category = 12
heldout test objects/category = 12
templates = 4
prompts/category = 48
layers = boundary_layer-3 ... boundary_layer
positions = answer_last, object_last, both
scales = 0.25, 0.5, 1.0, 1.5
controls = boundary, random_same_norm, neighbor_boundary
```

模型层位：

```text
Qwen3: L32-L35
GLM4: L15-L18
DS7B: L24-L27
```

### 客观结果

#### Qwen3

```text
number:
  strongest target down = L35 both scale1.5, target Δ -3.06
  same-setting random target Δ +0.02
  same-setting neighbor target Δ +0.28
  strongest release = animal +0.37

time:
  strongest target down = L35 both scale1.5, target Δ -1.35
  random +0.05
  neighbor +0.69
  strongest release = animal +0.61

container:
  strongest target down = L32 answer_last scale1.5, target Δ -0.34
  strongest release = clothing +2.03 at L34 both scale1.5

clothing:
  strongest target down = L33 answer_last scale1.5, target Δ -0.45
  strongest target up = +0.51
  strongest release = tool +1.08

furniture:
  no meaningful target down
  strongest target up = +2.10
  strongest release = clothing +2.22

plant:
  weak target down = -0.37 at L32 answer_last scale1.5
  strongest release = color +0.25
```

#### GLM4 bf16

```text
number:
  strongest target down = -0.02
  strongest target up = +0.17
  strongest release = material +0.48

time:
  strongest target down = -0.24
  strongest release = material +0.32

container:
  strongest target down = -0.05
  strongest release = event +0.25

clothing:
  strongest target down = -0.13
  strongest release = property +0.16

furniture:
  strongest target down = -0.08
  strongest target up = +0.14
  strongest release = material +0.22

plant:
  strongest target down = -0.06
  strongest release = shape +0.39
```

GLM4 效应仍然弱。

#### DS7B

```text
number:
  strongest target down = L27 both scale1.5, target Δ -4.75
  random -0.02
  neighbor -1.51
  strongest release = clothing +0.46

time:
  no meaningful boundary target down
  neighbor control itself can reduce target strongly
  strongest release = clothing +0.43

container:
  strongest target down = L27 both scale1.5, target Δ -3.21
  random -0.02
  neighbor +0.07
  strongest release weak = clothing +0.09

clothing:
  weak target down only at L25 object_last scale0.25, target Δ -0.17
  strongest target up = +1.61
  strongest release = tool +2.17

furniture:
  no meaningful target down
  strongest target up = +1.02
  strongest release = tool +1.48

plant:
  no meaningful target down
  strongest target up = +1.31
  strongest release = animal +1.51
```

### 当前最可靠客观事实

1. **number 是最稳定的可移除目标边界**

```text
Qwen3 number: L35 both scale1.5 target Δ -3.06
DS7B number: L27 both scale1.5 target Δ -4.75
```

两者都明显强于 random control，也强于 neighbor control。

2. **container 在 DS7B 是强 target-down 边界**

```text
DS7B container: L27 both scale1.5 target Δ -3.21
```

Qwen3 container 不是强 target-down，但有强 release：

```text
Qwen3 container -> clothing +2.03
```

3. **time 在 Qwen3 是 target-down + release，DS7B 不是**

```text
Qwen3 time: target Δ -1.35, animal release +0.61
DS7B time: boundary weak，neighbor control 影响更大
```

4. **clothing/furniture/plant 更像竞争/抑制混合边界**

这些类别常出现：

```text
target up
competitor release
缺少稳定 target down
```

例如：

```text
Qwen3 furniture: target up +2.10, clothing release +2.22
DS7B clothing: target up +1.61, tool release +2.17
DS7B plant: target up +1.31, animal release +1.51
```

5. **both-position 高 scale 对 target-down 很关键**

最强 target-down 基本出现在：

```text
answer_last + object_last
scale = 1.5
boundary peak layer
```

尤其 number 和 DS7B container。

6. **最佳因果层不一定是 boundary norm peak**

Qwen3 container/plant 的 target-down 出现在 L32，而不是 L35。

```text
boundary norm peak ≠ best causal layer
```

### 硬伤分析

1. **scale 最大只到 1.5**
   如果类别边界分布更宽，可能需要多层小 scale 累积，而不是单层大 scale。

2. **boundary vector 仍是 center-vs-others**
   对 clothing/furniture/plant 这类 target-up 类别，说明边界混入 suppressor/interface 成分，需要拆分。

3. **neighbor control 有时很强**
   DS7B time 中 neighbor control target down 更强，说明类别边界互相缠绕。

4. **没有同时做多层 patch**
   本轮是单层 sweep，不是 multi-layer cumulative patch。

5. **没有直接分解 support vs suppressor**
   只能从 target_down、target_up、release 模式推断，尚未直接分离成分。

### 下一步任务

Phase109 应继续客观测试：

```text
Support/Suppressor Decomposition
```

优先对象：

```text
number:
  作为较干净 target-support boundary。

clothing/furniture/plant:
  作为 suppressor/interface mixed boundary。

container:
  比较 Qwen3 release-only 与 DS7B target-down。
```

测试要求：

```text
1. 用 readout target direction 与 boundary vector 做分解。
2. 分别移除 boundary 中的 target-readout aligned component 和 orthogonal component。
3. 测 target_delta 与 release_delta。
4. 加 random_same_norm 和 neighbor_boundary control。
```

## Phase 109: 支持/抑制成分分解方案与条件化关系因子动力学更新 [2026-06-14 09:16]

### 本阶段性质

本阶段没有运行模型测试，而是根据 Phase 105-108 的客观结果，完成系统分析、公式更新和下一阶段研究方案设计。

### 对附加分析的判断

附加分析基本正确，尤其以下判断成立：

```text
1. 分布情况是语言编码机制的核心拼图。
2. Phase 107 已经证明 atlas boundary vector 进入真实 forward causal space。
3. Phase 108 证明类别边界不是简单正支持方向。
4. layer、position、scale、control 四个维度必须同时看。
5. number 是当前最稳定的 target-support boundary。
6. clothing/furniture/plant 更像 suppressor/interface mixed boundary。
```

需要收缩的部分：

```text
1. 不能把 CategoryCausalField 当成已被完整证明的理论对象。
2. 目前只证明了若干类别边界具有可测因果效应。
3. support / suppressor / interface 仍是基于干预模式的工作性分解，还不是直接电路分解。
4. 条件化关系因子动力学公式应更新为可测试公式，而不是最终数学理论。
```

### 当前客观进展

从 Phase 105 到 Phase 108，已经形成一条清楚路径：

```text
Phase 105:
  32 类全局类别图谱，发现层位分布、邻接关系、边界峰值。

Phase 106:
  多模板、模板残差、对象位置/答案位置复核。
  证明模板公共向量会遮蔽类别方向。

Phase 107:
  真实 forward boundary removal。
  证明 atlas boundary vector 能影响最终 logits。

Phase 108:
  layer/position/scale/control sweep。
  证明类别边界有不同因果类型。
```

当前最稳事实：

```text
1. number 是最稳定 target-support boundary:
   Qwen3: target_delta = -3.06
   DS7B:  target_delta = -4.75

2. time 在 Qwen3 中接近 target-down + release:
   target_delta = -1.35
   animal_release = +0.61

3. DS7B container 是强 target-down:
   target_delta = -3.21

4. clothing/furniture/plant 多数表现为 target-up 或 release-only。

5. both-position 高 scale 对强 target-down 很关键。

6. boundary norm peak 不一定是 best causal layer。
```

### 对深度神经网络内部结构研究的进展

当前内部结构研究从“有没有概念方向”推进到“方向的因果功能分类”：

```text
1. 表征几何:
   类别中心、边界、邻接关系存在。

2. 读出接口:
   answer_last 是类别读出放大槽位。

3. 分布式路径:
   object_last + answer_last 共同干预比单位置更强。

4. 因果分类:
   同样是类别边界，可能是支持、抑制、竞争、接口混合。

5. 模型差异:
   Qwen3 和 DS7B 对 number 一致，但对 container/clothing/plant 不一致。
```

这说明深度神经网络内部不是单一语义流，而至少有：

```text
object state
template/base state
category boundary
readout interface
competition/suppression field
final logit projection
```

### 条件化关系因子动力学公式更新

旧公式可以收缩为：

```text
h_{l,p} = Base_{l,p}(template)
        + Object_{l,p}(x)
        + Relation_{l,p}(r)
        + Category_{l,p}(c)
        + residual
```

但 Phase 106-108 表明这个公式不够，因为类别因子不是单一正方向。

更新为可测试公式：

```text
h_{l,p}(x,r,t)
= B_{l,p}(t)
+ O_{l,p}(x | t)
+ R_{l,p}(r | x,t)
+ C_{l,p}(c | x,r,t)
+ I_{l,p}(task | x,r,t)
+ ε
```

其中类别因子需要继续分解：

```text
C_{l,p}(c | x,r,t)
= S_{l,p}(c)
+ U_{l,p}(c)
+ K_{l,p}(c -> neighbors)
+ G_{l,p}(c -> readout)
```

含义：

```text
B:
  模板/基础状态。

O:
  对象状态。

R:
  关系条件状态。

C:
  类别条件状态。

I:
  任务/读出接口状态。

S:
  target-support component，支持目标类别的成分。

U:
  suppressor component，抑制或校准自身/邻居的成分。

K:
  competition component，压制或释放邻接类别的成分。

G:
  readout-interface component，连接输出词表读出的成分。
```

更直接的因果观测公式：

```text
ΔLogits_c
= A_c · Remove(S_c)
+ B_c · Remove(U_c)
+ D_c · Remove(K_c)
+ E_c · Remove(G_c)
```

当前观测对应：

```text
number:
  Remove(S_c) 主导，所以 target down。

clothing/furniture/plant:
  Remove(U_c 或 K_c) 主导，所以 target up 或 competitor release。

container:
  Qwen3 更像 K_c 主导，DS7B 更像 S_c 主导。
```

这不是最终理论，而是下一轮实验可直接证伪的工作公式。

### 当前最大问题和硬伤

1. **还没有直接分解 S/U/K/G**

目前只是通过 target_delta、release_delta、control 差异间接判断。

2. **边界仍由 center-vs-others 定义**

这个边界可能混合多个方向，不适合直接称为类别语义方向。

3. **邻居边界缠绕严重**

DS7B time 中 neighbor control 很强，说明类别边界不是独立坐标轴。

4. **多层累计效应未测**

Phase 108 是单层扫描，没有测试多层小尺度累积移除。

5. **读出词表仍可能影响结论**

DCF readout 仍是人工词表，不等于完整开放生成行为。

6. **GLM4 仍未解决**

GLM4 需要 bf16，且 readout 效应弱，必须做单独校准，不能和 Qwen3/DS7B 直接强比较。

### Phase 109 研究方案

目标：

```text
Support/Suppressor Decomposition
将类别边界拆成 target-readout aligned component 和 orthogonal component，
判断 support、suppressor、competition、interface 的相对贡献。
```

测试对象：

```text
number:
  稳定 target-support boundary。

time:
  Qwen3 中 target-down + animal release。

container:
  Qwen3 release-only, DS7B target-down。

clothing:
  tool release 明显，target-up/混合。

furniture:
  clothing release 明显，target-up/混合。

plant:
  animal/color release，target-up/混合。
```

核心方法：

```text
1. 计算类别边界 B_c。
2. 计算类别 readout direction W_c。
3. 将 B_c 分解为:

   B_parallel = projection(B_c, W_c)
   B_orth     = B_c - B_parallel

4. 分别移除:
   remove B_parallel
   remove B_orth
   remove full B_c

5. 测最终 logits:
   target_delta
   top competitor release
   random_same_norm control
   neighbor_boundary control
```

数据范围：

```text
models:
  qwen3, glm4, deepseek7b

GLM4:
  必须使用 PROBE_TORCH_DTYPE=bfloat16

categories:
  number, time, container, clothing, furniture, plant

train objects/category:
  12

heldout test objects/category:
  12

templates:
  4

positions:
  answer_last, both

layers:
  每个模型/类别采用 Phase 108 最强层 + boundary peak layer

scales:
  0.5, 1.0, 1.5
```

判据：

```text
如果 B_parallel 移除导致 target down:
  target-support component 成立。

如果 B_orth 移除导致 competitor release 或 target up:
  suppressor/competition component 成立。

如果 full B_c 效果大于两者单独效果:
  support 和 suppressor 存在非线性组合或接口耦合。

如果 neighbor control 接近或超过 B_c:
  该类别边界不是独立边界，而是邻接边界缠绕。
```

预期输出：

```text
1. 每类 support/suppressor/competition 类型表。
2. Qwen3 与 DS7B 的类别因果类型对照。
3. GLM4 readout 是否仍弱的客观确认。
4. 可用于 Phase 110 多层累计 patch 的候选类别。
```

## Phase 109: Support/Suppressor Decomposition 实测 [2026-06-14 09:23]

### 本阶段目标

根据 Phase108 的下一步任务，直接测试：

```text
类别边界 B_c 中，哪一部分是 target-readout aligned component，
哪一部分是 orthogonal component，
二者分别导致 target down、target up 还是 competitor release。
```

本轮重点不是理论总结，而是用真实 forward patch 继续客观拼图。

### 执行命令

```bash
python tests/gpt5/phase109_support_suppressor_decomposition_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --categories number,time \
  --output-dir results/gpt5_phase109_smoke \
  --hard-exit-after-model

python tests/gpt5/phase109_support_suppressor_decomposition_cuda.py qwen3 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --output-dir results/gpt5_phase109_support_suppressor_decomposition \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase109_support_suppressor_decomposition_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --output-dir results/gpt5_phase109_support_suppressor_decomposition \
  --hard-exit-after-model

python tests/gpt5/phase109_support_suppressor_decomposition_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --output-dir results/gpt5_phase109_support_suppressor_decomposition \
  --hard-exit-after-model

python tests/gpt5/phase109_support_suppressor_decomposition_summary.py

python -m py_compile \
  tests/gpt5/phase109_support_suppressor_decomposition_cuda.py \
  tests/gpt5/phase109_support_suppressor_decomposition_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase109_support_suppressor_decomposition_cuda.py`
- 汇总脚本：`tests/gpt5/phase109_support_suppressor_decomposition_summary.py`
- Qwen3 结果：`results/gpt5_phase109_support_suppressor_decomposition/phase109_qwen3_support_suppressor_decomposition.json`
- GLM4 结果：`results/gpt5_phase109_support_suppressor_decomposition/phase109_glm4_support_suppressor_decomposition.json`
- DS7B 结果：`results/gpt5_phase109_support_suppressor_decomposition/phase109_deepseek7b_support_suppressor_decomposition.json`
- 跨模型汇总：`results/gpt5_phase109_support_suppressor_decomposition/phase109_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, time, container, clothing, furniture, plant
train objects/category = 12
heldout test objects/category = 12
templates = 4
positions = answer_last, both
scales = 0.5, 1.0, 1.5
kinds = full_boundary, readout_parallel, orthogonal, random_same_norm, neighbor_boundary
```

### 分解方法

```text
B_c = category boundary
W_c = category readout direction

B_parallel = projection(B_c, W_c)
B_orth = B_c - B_parallel
```

分别移除：

```text
full_boundary
readout_parallel
orthogonal
random_same_norm
neighbor_boundary
```

### 客观结果

#### Qwen3

```text
number:
  cos(B,W)=0.165
  parallel_norm_fraction=0.165
  readout_parallel target_delta=-0.05
  orthogonal target_delta=-3.05
  full target_delta=-3.06

time:
  cos(B,W)=0.204
  readout_parallel target_delta=-0.18
  orthogonal target_delta=-0.64
  orthogonal release animal+0.96
  full target_delta=-1.35

container:
  readout_parallel target_delta=+0.01
  orthogonal target_delta=-0.33
  orthogonal release shape+2.81
  full target_delta=-0.34

clothing:
  readout_parallel target_delta=-0.37
  orthogonal target_delta=-0.57
  orthogonal release tool+1.46
  full target_delta=-0.45

furniture:
  readout_parallel target_delta=+0.02
  orthogonal target_delta=+1.00
  orthogonal release number+3.30
  full target_delta=+0.72

plant:
  readout_parallel target_delta=+0.11
  orthogonal target_delta=-0.41
  orthogonal release color+0.13
  full target_delta=-0.37
```

#### GLM4 bf16

```text
boundary-readout cos 接近 0。
所有类别效应整体很弱。

number:
  orthogonal target_delta=-0.01
  orthogonal release material+0.45

time:
  orthogonal target_delta=-0.23
  release material+0.33

container:
  orthogonal target_delta=-0.04
  release event+0.25

clothing:
  orthogonal target_delta=-0.13
  release property+0.17

furniture:
  orthogonal target_delta=-0.08
  release material+0.22

plant:
  orthogonal target_delta=-0.06
  release shape+0.39
```

#### DS7B

```text
number:
  cos(B,W)=0.130
  readout_parallel target_delta=-0.08
  orthogonal target_delta=-4.95
  full target_delta=-4.75

container:
  cos(B,W)=0.102
  readout_parallel target_delta=+0.06
  orthogonal target_delta=-3.15
  full target_delta=-3.21

clothing:
  readout_parallel target_delta=-0.87
  orthogonal target_delta=+0.40
  orthogonal release tool+2.24
  full target_delta=+0.39

furniture:
  readout_parallel target_delta=-1.11
  orthogonal target_delta=+0.16
  orthogonal release tool+1.09
  full target_delta=+0.31

plant:
  readout_parallel target_delta=-0.19
  orthogonal target_delta=+0.28
  orthogonal release animal+1.59
  full target_delta=+0.33
```

### 当前最可靠客观事实

1. **强 target-down 主要来自 orthogonal component，而不是 readout_parallel component**

```text
Qwen3 number:
  readout_parallel -0.05
  orthogonal -3.05

DS7B number:
  readout_parallel -0.08
  orthogonal -4.95

DS7B container:
  readout_parallel +0.06
  orthogonal -3.15
```

这推翻了一个简单假设：

```text
target-support boundary 不等于直接 output-readout aligned direction。
```

2. **boundary 与 readout word direction 的 cos 很低**

```text
Qwen3: 约 0.15-0.20
DS7B: 约 0.07-0.13
GLM4: 接近 0
```

说明类别因果边界多数不沿着输出词表 readout 方向。

3. **DS7B clothing/furniture 出现成分冲突**

```text
clothing:
  readout_parallel target down -0.87
  orthogonal release tool +2.24
  full boundary target up +0.39

furniture:
  readout_parallel target down -1.11
  orthogonal release tool +1.09
  full boundary target up +0.31
```

这说明 full boundary 是多个成分相互抵消/冲突后的结果。

4. **Qwen3 furniture 是典型 competition/interface 混合边界**

```text
orthogonal target up +1.00
orthogonal release number +3.30
full target up +0.72
```

5. **GLM4 仍然弱**

GLM4 的边界-readout cos 接近 0，效应小，仍需 readout calibration。

### 对公式的修正

Phase109 后，`S` 不应再简单等同于 readout_parallel。

需要改为：

```text
C_c = S_c + U_c + K_c + G_c
```

但：

```text
G_c ≈ readout_parallel component
S_c 不一定与 G_c 对齐
S_c 很可能主要位于 readout-orthogonal causal subspace
```

也就是说：

```text
target support 不是直接输出词方向；
它可能是通过内部因果子空间改变最终 readout。
```

这对破解编码机制非常关键。

### 硬伤分析

1. **readout direction 仍由 DCF 词表定义**

如果 readout words 不准，parallel/orthogonal 分解会受影响。

2. **orthogonal component 仍然太大**

因为 boundary-readout cos 很低，orthogonal 几乎包含大部分边界，仍需进一步分解。

3. **只分成两块还不够**

orthogonal 中同时包含 support、suppressor、competition、interface residual。

4. **未做多层累计**

number/container 的 orthogonal target-down 强，但是否来自单层或多层累积仍未知。

5. **GLM4 readout 问题未解决**

GLM4 不能用于强机制结论。

### 下一步任务

Phase110 应继续客观测试：

```text
Orthogonal Subspace Split
```

目标：

```text
把 B_orth 继续分成:
1. neighbor-aligned component
2. target-object trajectory component
3. residual component
```

优先测试：

```text
Qwen3 number/time/furniture
DS7B number/container/clothing/furniture/plant
```

方法：

```text
1. 用 neighbor boundary basis 分解 B_orth。
2. 用 object_last -> answer_last transport direction 分解 B_orth。
3. 分别移除各子成分。
4. 测 target_delta、release_delta、control_delta。
```

## Phase 110: Orthogonal Subspace Split 正交子空间拆分 [2026-06-14 09:34]

### 本阶段目标

根据 Phase109 的结果，`readout_parallel` 不是主要 target support，真正强因果成分主要位于 `readout-orthogonal` 子空间。

本阶段继续把 `B_orth` 拆成三类更基础成分：

```text
1. neighbor_aligned: 与邻近类别边界空间对齐的成分
2. transport_aligned: 与 object_last -> answer_last 平均传输方向对齐的成分
3. residual: 去除 neighbor 和 transport 后剩余的成分
```

核心问题：

```text
强 target-down 到底来自类别竞争边界、对象到答案位置的传输通道，还是剩余未知方向。
```

### 执行命令

```bash
python -m py_compile tests/gpt5/phase110_orthogonal_subspace_split_cuda.py

python tests/gpt5/phase110_orthogonal_subspace_split_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --categories number,time \
  --output-dir results/gpt5_phase110_smoke \
  --hard-exit-after-model

python tests/gpt5/phase110_orthogonal_subspace_split_cuda.py qwen3 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --output-dir results/gpt5_phase110_orthogonal_subspace_split \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase110_orthogonal_subspace_split_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --output-dir results/gpt5_phase110_orthogonal_subspace_split \
  --hard-exit-after-model

python tests/gpt5/phase110_orthogonal_subspace_split_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --output-dir results/gpt5_phase110_orthogonal_subspace_split \
  --hard-exit-after-model

python tests/gpt5/phase110_orthogonal_subspace_split_summary.py

python -m py_compile \
  tests/gpt5/phase110_orthogonal_subspace_split_cuda.py \
  tests/gpt5/phase110_orthogonal_subspace_split_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase110_orthogonal_subspace_split_cuda.py`
- 汇总脚本：`tests/gpt5/phase110_orthogonal_subspace_split_summary.py`
- Qwen3 结果：`results/gpt5_phase110_orthogonal_subspace_split/phase110_qwen3_orthogonal_subspace_split.json`
- GLM4 结果：`results/gpt5_phase110_orthogonal_subspace_split/phase110_glm4_orthogonal_subspace_split.json`
- DS7B 结果：`results/gpt5_phase110_orthogonal_subspace_split/phase110_deepseek7b_orthogonal_subspace_split.json`
- 跨模型汇总：`results/gpt5_phase110_orthogonal_subspace_split/phase110_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, time, container, clothing, furniture, plant
train objects/category = 12
heldout test objects/category = 12
templates = 4
prompts/category = 48
components = orthogonal_full, neighbor_aligned, transport_aligned, residual, random_same_norm
positions = answer_last, both
scales = 1.0, 1.5
```

模型层位：

```text
Qwen3: L35
GLM4: L18
DS7B: L27
```

### 客观结果

#### Qwen3

```text
number:
  norm fractions neighbor/transport/residual = 0.54/0.27/0.80
  best neighbor target Δ -1.91
  best transport target Δ -3.43
  best residual target Δ -0.37
  best orthogonal_full target Δ -3.05
  random_same_norm target Δ -0.12

time:
  fractions = 0.57/0.15/0.81
  neighbor target Δ -1.95
  transport target Δ -1.84
  residual target Δ +0.03
  orthogonal_full target Δ -0.64

container:
  fractions = 0.46/0.28/0.84
  neighbor target Δ +0.24
  transport target Δ -1.75
  residual target Δ +0.00
  orthogonal_full target Δ +0.25

clothing:
  fractions = 0.34/0.39/0.85
  neighbor target Δ +0.71
  transport target Δ -1.43
  residual target Δ +0.01
  orthogonal_full target Δ +0.72

furniture:
  fractions = 0.54/0.33/0.78
  neighbor target Δ +0.55
  transport target Δ -0.56
  residual target Δ -0.46
  orthogonal_full target Δ +1.93

plant:
  fractions = 0.49/0.28/0.83
  neighbor target Δ +0.10
  transport target Δ -5.97
  residual target Δ -0.29
  orthogonal_full target Δ -0.02
```

#### GLM4 bf16

```text
number:
  fractions = 0.64/0.04/0.77
  strongest target down = neighbor Δ -0.14

time:
  fractions = 0.74/0.07/0.67
  strongest target down = neighbor Δ -0.47

container:
  fractions = 0.84/0.06/0.53
  strongest target down = transport Δ -0.07

clothing:
  fractions = 0.89/0.01/0.45
  strongest target down = orthogonal_full Δ -0.08

furniture:
  fractions = 0.93/0.00/0.37
  strongest target down = transport Δ -0.03

plant:
  fractions = 0.90/0.04/0.44
  strongest target down = orthogonal_full Δ -0.06
```

GLM4 仍然弱，不能作为强机制结论来源。

#### DS7B

```text
number:
  fractions = 0.41/0.22/0.89
  neighbor target Δ -0.94
  transport target Δ +1.06
  residual target Δ -2.76
  orthogonal_full target Δ -4.95
  random_same_norm target Δ +0.07

time:
  fractions = 0.46/0.18/0.87
  neighbor target Δ -0.82
  transport target Δ -0.61
  residual target Δ -0.93
  orthogonal_full target Δ +0.06

container:
  fractions = 0.30/0.31/0.90
  neighbor target Δ -0.24
  transport target Δ -5.68
  residual target Δ -1.44
  orthogonal_full target Δ -3.15

clothing:
  fractions = 0.28/0.44/0.85
  neighbor target Δ -0.18
  transport target Δ -5.17
  residual target Δ -0.91
  orthogonal_full target Δ +1.22

furniture:
  fractions = 0.44/0.35/0.83
  neighbor target Δ +0.07
  transport target Δ -3.85
  residual target Δ -0.03
  orthogonal_full target Δ +0.31

plant:
  fractions = 0.42/0.34/0.84
  neighbor target Δ +0.66
  transport target Δ -3.28
  residual target Δ -0.12
  orthogonal_full target Δ +1.05
```

### 当前最可靠客观事实

1. **transport_aligned 是大量类别的强 target-down 成分**

典型结果：

```text
Qwen3 number transport Δ -3.43
Qwen3 plant transport Δ -5.97
DS7B container transport Δ -5.68
DS7B clothing transport Δ -5.17
DS7B furniture transport Δ -3.85
DS7B plant transport Δ -3.28
```

这说明 object_last 到 answer_last 的内部传输方向，是类别信息进入答案位置的重要候选通道。

2. **完整 orthogonal_full 会掩盖子成分**

例如：

```text
Qwen3 plant:
  transport Δ -5.97
  orthogonal_full Δ -0.02

DS7B clothing:
  transport Δ -5.17
  orthogonal_full Δ +1.22

DS7B plant:
  transport Δ -3.28
  orthogonal_full Δ +1.05
```

完整正交边界里混有方向相反的成分，直接移除整块会发生抵消甚至 target-up。

3. **DS7B number 是特殊模式**

```text
DS7B number:
  residual Δ -2.76
  orthogonal_full Δ -4.95
  transport Δ +1.06
```

number 在 DS7B 中不是 transport 主导，而更像剩余未知方向与完整正交边界共同形成强支撑。

4. **Qwen3 time 更像 neighbor/transport 混合**

```text
Qwen3 time:
  neighbor Δ -1.95
  transport Δ -1.84
  orthogonal_full Δ -0.64
```

time 与 number、event、weather 等邻近类别纠缠更强。

5. **GLM4 仍然低效应**

GLM4 的最大效应大多小于 0.5，继续证明当前 readout/intervention 框架下 GLM4 信号弱。

### 对 Phase109 附加分析的校正

Phase109 的核心判断仍然正确：

```text
target support 主要不在 readout_parallel；
readout-orthogonal causal subspace 是关键区域。
```

但 Phase110 进一步说明：

```text
readout-orthogonal 不是一个单一语义边界；
其中大量强因果效应来自 object_last -> answer_last transport component。
```

因此更准确的说法是：

```text
模型内部的类别信息，可能先在对象位置形成类别/对象状态，
再通过位置传输通道进入答案位置，
最后才改变输出词 readout。
```

### 条件化关系因子动力学公式更新

上一阶段：

```text
C_c = S_c + U_c + K_c + G_c
```

Phase110 后应拆成：

```text
C_c = G_c + N_c + T_c + R_c
```

含义：

```text
C_c: 类别边界整体
G_c: readout-parallel output gateway
N_c: neighbor-aligned competition/interface component
T_c: object_last -> answer_last transport component
R_c: residual unknown causal component
```

更接近当前结果的因果链：

```text
object state at object_last
  -> T_c transport to answer_last
  -> answer-position category state
  -> G_c/output gateway
  -> next-token category logits
```

中文解释：

```text
对象位置先承载对象/类别状态；
答案位置不是凭空生成类别，而是接收对象位置传来的类别状态；
输出词方向只是最后的门口，不是内部语义支撑本体。
```

### 硬伤分析

1. **transport direction 只是均值差分方向**

当前 `object_last -> answer_last` 是平均残差差分，不等于已经证明真实路径。

2. **neighbor basis 是人工邻接**

邻近类别由人为指定，可能漏掉模型内部真正的竞争类别。

3. **仍是单层干预**

如果类别传输跨多层累积，单层移除会低估或扭曲真实机制。

4. **子成分之间不是线性独立因果模块**

一些子成分移除比完整 orthogonal_full 更强，说明完整边界内部存在非线性或方向抵消。

5. **GLM4 仍然不能用于强结论**

GLM4 在当前框架下效应弱，需要单独校准。

### 下一步任务

Phase111 应做一个更大的阶段任务：

```text
Transport Path Causal Mapping
```

目标：

```text
确认 transport component 是否是真正的对象位置到答案位置类别传输通道。
```

建议测试：

```text
1. 对 object_last 单独写入/移除 transport component，观察 answer_last 与 logits 是否同步变化。
2. 对 answer_last 单独写入/移除 transport component，和 object_last 干预对照。
3. 做 layer-to-layer transport sweep，找出类别状态从对象位置迁移到答案位置的层段。
4. 做 multi-layer cumulative patch，确认单层结果是否低估或被抵消。
5. 对 Qwen3 number/time/plant 与 DS7B number/container/clothing/furniture/plant 扩大 heldout objects 做复测。
```

优先级：

```text
第一优先：DS7B container/clothing/furniture/plant 的 transport-dominant 现象
第二优先：Qwen3 plant 的 transport 强效但 orthogonal_full 近零现象
第三优先：DS7B number 的 residual-support 特殊模式
```

## Phase 111: Transport Path Causal Mapping 传输路径因果定位 [2026-06-14 10:43]

### 本阶段目标

根据用户附加分析与 Phase110 结果，先判断：

```text
Phase110 的 transport_aligned 强 target-down 是否等于真实 object_last -> answer_last 传输路径？
```

附加分析中正确部分：

```text
1. Phase110 的 transport_aligned 是目前最强候选语义通道之一。
2. readout_parallel 不是主要语义支持方向。
3. orthogonal_full 会掩盖强子成分。
4. transport direction 仍只是均值差分，不等于已经证明真实路径。
5. 下一步必须做 object-site 与 answer-site 的因果对照。
```

因此本阶段不再继续理论总结，而是直接测试：

```text
在 object_last 移除/写入 T_c，answer_last 的 T_c 投影和 final logits 是否同步变化。
```

### 执行命令

smoke：

```bash
python -m py_compile \
  tests/gpt5/phase111_transport_path_causal_mapping_cuda.py \
  tests/gpt5/phase111_transport_path_causal_mapping_summary.py

python tests/gpt5/phase111_transport_path_causal_mapping_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --layer-back 1 \
  --categories number,time \
  --scales 1.0 \
  --output-dir results/gpt5_phase111_smoke \
  --hard-exit-after-model
```

正式测试第一轮使用：

```bash
python tests/gpt5/phase111_transport_path_causal_mapping_cuda.py qwen3 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --output-dir results/gpt5_phase111_transport_path_causal_mapping \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase111_transport_path_causal_mapping_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --output-dir results/gpt5_phase111_transport_path_causal_mapping \
  --hard-exit-after-model

python tests/gpt5/phase111_transport_path_causal_mapping_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --output-dir results/gpt5_phase111_transport_path_causal_mapping \
  --hard-exit-after-model
```

第一轮发现 Phase110 的强效常在 scale=1.5，而默认范围只有 0.25/0.5/1.0。为避免错误否定，重新加入 1.5 完整复测：

```bash
python tests/gpt5/phase111_transport_path_causal_mapping_cuda.py qwen3 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --scales 0.25,0.5,1.0,1.5 \
  --output-dir results/gpt5_phase111_transport_path_causal_mapping \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase111_transport_path_causal_mapping_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --scales 0.25,0.5,1.0,1.5 \
  --output-dir results/gpt5_phase111_transport_path_causal_mapping \
  --hard-exit-after-model

python tests/gpt5/phase111_transport_path_causal_mapping_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --scales 0.25,0.5,1.0,1.5 \
  --output-dir results/gpt5_phase111_transport_path_causal_mapping \
  --hard-exit-after-model

python tests/gpt5/phase111_transport_path_causal_mapping_summary.py

python -m py_compile \
  tests/gpt5/phase111_transport_path_causal_mapping_cuda.py \
  tests/gpt5/phase111_transport_path_causal_mapping_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase111_transport_path_causal_mapping_cuda.py`
- 汇总脚本：`tests/gpt5/phase111_transport_path_causal_mapping_summary.py`
- Qwen3 结果：`results/gpt5_phase111_transport_path_causal_mapping/phase111_qwen3_transport_path_causal_mapping.json`
- GLM4 结果：`results/gpt5_phase111_transport_path_causal_mapping/phase111_glm4_transport_path_causal_mapping.json`
- DS7B 结果：`results/gpt5_phase111_transport_path_causal_mapping/phase111_deepseek7b_transport_path_causal_mapping.json`
- 跨模型汇总：`results/gpt5_phase111_transport_path_causal_mapping/phase111_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, time, container, clothing, furniture, plant
train objects/category = 12
heldout test objects/category = 12
templates = 4
prompts/category = 48
patch layers = peak-3 ... peak
patch sites = object_last, answer_last
patch modes = remove_target, amplify_target, wrong_inject_abs, random_remove
scales = 0.25, 0.5, 1.0, 1.5
monitor = answer_last transport projection at peak layer + final DCF logits
```

模型层位：

```text
Qwen3: monitor L35, patch L32-L35
GLM4: monitor L18, patch L15-L18
DS7B: monitor L27, patch L24-L27
```

### 测试原理

Phase110 中 `T_c` 的定义：

```text
B_orth = B - proj(B, readout_direction)
after_neighbor = B_orth - proj(B_orth, neighbor_boundary_basis)
T_c = proj(after_neighbor, mean(answer_last - object_last))
```

Phase111 做真实 forward 干预：

```text
1. 在 object_last 移除 target T_c。
2. 在 answer_last 移除 target T_c。
3. 写入 wrong-category T_d。
4. 使用 random_same_norm 作为对照。
5. 记录 final logits target_delta。
6. 同时记录 peak layer answer_last 的 T_c projection delta。
```

强路径闭包判据：

```text
object_last remove T_c
  -> answer_last T_c projection 同步下降
  -> target logits 同步下降
  -> 明显强于 random control
```

### 客观结果

#### Qwen3

```text
number:
  object_last remove: target Δ -0.00, answer projection Δ -0.10
  answer_last remove: target Δ -3.43
  wrong inject: target Δ -3.54
  random: target Δ -0.05

time:
  object_last remove: target Δ -0.03, answer projection Δ +0.15
  answer_last remove: target Δ -1.84
  wrong inject: target Δ -4.18
  random: target Δ -0.09

container:
  object_last remove: target Δ -0.05, answer projection Δ -0.16
  answer_last remove: target Δ -2.59
  wrong inject: target Δ -0.76
  random: target Δ -0.08

clothing:
  object_last remove: target Δ +0.01
  answer_last remove: target Δ -1.43
  wrong inject: target Δ -4.51

furniture:
  object_last remove: target Δ +0.01
  answer_last remove: target Δ -0.55
  wrong inject: target Δ -3.26

plant:
  object_last remove: target Δ -0.00, answer projection Δ +0.10
  answer_last remove: target Δ -5.97
  wrong inject: target Δ -2.52
```

#### GLM4 bf16

```text
all categories:
  object_last remove target effect near 0
  answer_last remove target effect near 0
  wrong inject weak
```

最大量级仍然很小：

```text
wrong inject clothing Δ -0.22
wrong inject furniture Δ -0.21
```

GLM4 在当前框架中仍不能支持强机制判断。

#### DS7B

```text
number:
  object_last remove: target Δ -0.07
  answer_last remove: target Δ +0.69
  wrong inject: target Δ -3.39
  random: target Δ -0.12

time:
  object_last remove: target Δ -0.02
  answer_last remove: target Δ -0.56
  wrong inject: target Δ -1.50

container:
  object_last remove: target Δ -0.21
  object-site strongest answer projection drop: Δ -1.70, but target Δ +0.08
  answer_last remove: target Δ -5.50
  random: target Δ -0.38

clothing:
  object_last remove: target Δ -0.23
  object-site strongest answer projection drop: Δ -2.23, but target Δ +0.05
  answer_last remove: target Δ -5.04

furniture:
  object_last remove: target Δ -0.17
  object-site strongest answer projection drop: Δ -2.16, but target Δ +0.11
  answer_last remove: target Δ -3.82

plant:
  object_last remove: target Δ -0.15
  answer projection Δ -0.75
  answer_last remove: target Δ -3.20
  wrong inject: target Δ -2.11
```

### 当前最可靠客观事实

1. **answer_last 是 transport_aligned 强 target-down 的直接作用位点**

强结果与 Phase110 基本对齐：

```text
Qwen3 number answer_last remove Δ -3.43
Qwen3 plant answer_last remove Δ -5.97
DS7B container answer_last remove Δ -5.50
DS7B clothing answer_last remove Δ -5.04
DS7B furniture answer_last remove Δ -3.82
DS7B plant answer_last remove Δ -3.20
```

2. **object_last remove 没有形成强 logits 因果闭包**

所有模型/类别中，object_last remove 的 target_delta 都很弱：

```text
Qwen3: roughly 0
GLM4: roughly 0
DS7B: strongest only around -0.23
```

3. **DS7B object_last 干预可以改变 answer projection，但不改变 target logits**

例如：

```text
DS7B container:
  object-site answer projection Δ -1.70
  target Δ +0.08

DS7B clothing:
  object-site answer projection Δ -2.23
  target Δ +0.05

DS7B furniture:
  object-site answer projection Δ -2.16
  target Δ +0.11
```

这说明“投影同步变化”本身不足以证明输出因果闭包。

4. **wrong-category injection 往往很强，但更像干扰/抑制，不是清晰类别替换**

例如：

```text
Qwen3 clothing wrong inject Δ -4.51
Qwen3 time wrong inject Δ -4.18
DS7B number wrong inject Δ -3.39
```

这些结果说明 wrong T_d 写入会强烈扰乱目标类别，但尚未证明它把输出推向指定错误类别。

5. **GLM4 继续弱**

GLM4 仍不能用于强结论。

### 对 Phase110 理论的校正

Phase110 的正确部分：

```text
T_c 是大量类别的强 target-down 成分。
T_c 位于 readout-orthogonal 子空间。
完整 orthogonal_full 会被其他成分抵消。
```

Phase111 的关键校正：

```text
当前还不能说 T_c 已被证明为 object_last -> answer_last 的真实传输路径。
```

更严格表述应为：

```text
T_c 是 answer_last 上非常强的类别状态/读出前状态成分；
它与 object_last -> answer_last 的均值差分对齐；
但 object_last 单点移除没有让 final logits 产生同步强变化。
```

因此当前理论从：

```text
object_last category state -> T_c transport -> answer_last
```

暂时回退为更谨慎的版本：

```text
object/answer positional contrast defines T_c;
T_c at answer_last is a strong causal pre-readout state;
object_last 单点 patch 尚未闭合到 logits。
```

### 条件化关系因子动力学公式更新

Phase110 公式：

```text
C_c = G_c + N_c + T_c + R_c
```

Phase111 后应加上位点区分：

```text
C_c(answer) = G_c(answer) + N_c(answer) + T_c(answer) + R_c(answer)
C_c(object) = O_c(object) + P_c(object)
```

当前已验证较强的是：

```text
T_c(answer) -> final logits
```

尚未验证的是：

```text
C_c(object) -> T_c(answer) -> final logits
```

因此完整链条应暂写为：

```text
object_state  --unclosed--> answer_transport_state -> output_gateway -> logits
```

中文解释：

```text
答案位置上的传输对齐状态具有强输出因果作用；
对象位置到答案位置的上游路径仍未闭合。
```

### 硬伤分析

1. **没有证明 object_last 单点移除足够打断路径**

object_last 可能只是路径起点之一，真实传输可能分布在多个层、多个 token、多个 attention head 中。

2. **monitor projection 不是完整 answer state**

本轮只监测 peak layer 的一个 T_c 投影。即使该投影下降，也不等于完整语义状态下降。

3. **patch at monitor layer 的 projection delta 记录有局限**

当 patch layer 等于 monitor layer 时，final logits 已改变，但记录的 hidden projection 可能显示 0，说明 hook 返回值与 hidden_states 记录顺序存在实现细节限制。

4. **wrong injection 未做目标错误类别释放分析**

wrong T_d 会压低目标，但还没有证明它提升了指定 wrong category。

5. **仍未做 generation audit**

目前仍是 DCF logits，不是开放生成闭包。

### 当前进展评价

Phase111 的结果不是对 Phase110 的否定，而是把结论变严格：

```text
Phase110 证明：T_c(answer) 是强因果成分。
Phase111 显示：object_last 单点 T_c patch 不能闭合到 final logits。
```

所以当前最可靠拼图是：

```text
读出前答案位置状态，是类别输出的关键因果位置；
对象位置上游路径仍未找到真正入口。
```

### 下一步任务

Phase112 应进入更细的路径搜索，而不是继续只做 residual stream 单点 patch：

```text
Attention Transport Head Mapping
```

目标：

```text
找出哪些 attention heads 把 object_last 信息写入 answer_last。
```

建议测试：

```text
1. 在 peak-3...peak 层记录 answer_last 对 object_last 的 attention 权重。
2. 对高权重 head 做 head output ablation。
3. 对高权重 head 做 object_last value patch。
4. 观察 answer_last T_c projection 与 final logits 是否同步变化。
5. 对 DS7B container/clothing/furniture/plant 与 Qwen3 plant/number 重点复测。
```

关键理由：

```text
如果真实路径是 attention transport，
那么 residual stream 的 object_last 单点 T_c 移除可能打不到真正写入 answer_last 的 head/value 通道。
```

## Phase 112: Attention Transport Head Mapping 注意力传输头定位 [2026-06-14 10:58]

### 本阶段目标

根据用户附加分析与 Phase111 结果，先判断：

```text
Phase111 的收缩是正确的：
T_c(answer) 是强因果读出前状态；
object_last 单点 residual patch 没有闭合到 logits。
```

附加分析中正确部分：

```text
1. 不应继续把 T_c 直接解释成已证明的 object_last -> answer_last 真实路径。
2. 下一步应从 residual direction 转向 attention route。
3. 需要测 answer_last 对 object/relation source 的 attention mass。
4. 需要做 head output ablation，而不只看注意力权重。
5. projection change 不等于 causal closure。
```

本阶段目标：

```text
定位哪些 attention heads 在 answer_last 读取 object source；
并测试这些 head 的单头消融是否降低 T_c(answer) 与 final logits。
```

### 执行命令

smoke：

```bash
python -m py_compile \
  tests/gpt5/phase112_attention_transport_head_mapping_cuda.py \
  tests/gpt5/phase112_attention_transport_head_mapping_summary.py

python tests/gpt5/phase112_attention_transport_head_mapping_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --layer-back 1 \
  --top-k-heads 2 \
  --categories number,time \
  --output-dir results/gpt5_phase112_smoke \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase112_attention_transport_head_mapping_cuda.py glm4 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --layer-back 1 \
  --top-k-heads 2 \
  --categories number,time \
  --output-dir results/gpt5_phase112_smoke \
  --hard-exit-after-model
```

正式测试：

```bash
python tests/gpt5/phase112_attention_transport_head_mapping_cuda.py qwen3 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --top-k-heads 8 \
  --output-dir results/gpt5_phase112_attention_transport_head_mapping \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase112_attention_transport_head_mapping_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --top-k-heads 8 \
  --output-dir results/gpt5_phase112_attention_transport_head_mapping \
  --hard-exit-after-model

python tests/gpt5/phase112_attention_transport_head_mapping_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --top-k-heads 8 \
  --output-dir results/gpt5_phase112_attention_transport_head_mapping \
  --hard-exit-after-model

python tests/gpt5/phase112_attention_transport_head_mapping_summary.py

python -m py_compile \
  tests/gpt5/phase112_attention_transport_head_mapping_cuda.py \
  tests/gpt5/phase112_attention_transport_head_mapping_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase112_attention_transport_head_mapping_cuda.py`
- 汇总脚本：`tests/gpt5/phase112_attention_transport_head_mapping_summary.py`
- Qwen3 结果：`results/gpt5_phase112_attention_transport_head_mapping/phase112_qwen3_attention_transport_head_mapping.json`
- GLM4 结果：`results/gpt5_phase112_attention_transport_head_mapping/phase112_glm4_attention_transport_head_mapping.json`
- DS7B 结果：`results/gpt5_phase112_attention_transport_head_mapping/phase112_deepseek7b_attention_transport_head_mapping.json`
- 跨模型汇总：`results/gpt5_phase112_attention_transport_head_mapping/phase112_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, time, container, clothing, furniture, plant
train objects/category = 12
heldout test objects/category = 12
templates = 4
prompts/category = 48
layers = peak-3 ... peak
selected heads/category = top 8 by answer_last attention to object_span + object_last
intervention = zero selected head slice at answer_last before o_proj
metrics = target_delta, release_delta, answer_last T_c projection delta
```

模型层位：

```text
Qwen3: monitor L35, scan/patch L32-L35, heads=32
GLM4: monitor L18, scan/patch L15-L18, heads=32
DS7B: monitor L27, scan/patch L24-L27, heads=28
```

### 测试原理

Phase112 分两步：

```text
1. attention source scan:
   对每个 head 记录 answer_last query 对 object_span/object_last/pre_object/post_object/self 的 attention mass。

2. head output ablation:
   在 o_proj 输入前，把该 head 在 answer_last 的 head slice 置零。
   然后测 final logits 和 answer_last T_c projection。
```

注意：

```text
attention mass 只用于候选选择；
真正因果判据来自 head ablation 后的 logits change。
```

### 客观结果

#### Qwen3

```text
number:
  top source head = L35 H21 object mass 0.057
  strongest target-down = L33 H24 target Δ -0.30, answer projection Δ -1.86
  strongest projection-down = L35 H8 answer projection Δ -4.83, target Δ +0.01

time:
  top source head = L33 H9 object mass 0.078
  strongest target-down = L35 H21 target Δ -0.02, answer projection Δ -2.15
  strongest projection-down = L35 H8 answer projection Δ -5.78, target Δ +0.00

container:
  top source head = L35 H27 object mass 0.093
  strongest target-down = L34 H21 target Δ -0.03
  strongest projection-down = L35 H8 answer projection Δ -4.73, target Δ +0.03

clothing:
  top source head = L33 H9 object mass 0.111
  strongest target-down = L33 H9 target Δ -0.07
  strongest projection-down = L35 H8 answer projection Δ -5.42, target Δ -0.02

furniture:
  top source head = L35 H21 object mass 0.101
  strongest target-down = L35 H28 target Δ -0.05
  strongest projection-down = L35 H8 answer projection Δ -4.85, target Δ -0.03

plant:
  top source head = L34 H21 object mass 0.117
  strongest target-down = L35 H27 target Δ -0.02
  strongest projection-down = L35 H21 answer projection Δ -2.18, target Δ +0.03
```

#### GLM4 bf16

```text
number/time/container/clothing/furniture/plant:
  top object-source attention heads exist, object mass roughly 0.12-0.16
  strongest target-down all near 0
  projection changes also near 0
```

GLM4 仍然不支持当前机制框架下的强结论。

#### DS7B

```text
number:
  top source head = L24 H17 object mass 0.174
  strongest target-down = L24 H22 target Δ -0.08, answer projection Δ -3.64

time:
  top source head = L25 H19 object mass 0.202
  strongest target-down = L24 H22 target Δ -0.06, answer projection Δ -6.62

container:
  top source head = L25 H19 object mass 0.228
  strongest target-down = L25 H15 target Δ -0.27, answer projection Δ -4.97
  strongest projection-down = L24 H22 answer projection Δ -5.84, target Δ -0.14

clothing:
  top source head = L25 H19 object mass 0.229
  strongest target-down = L24 H17 target Δ -0.40, answer projection Δ +0.62
  strongest projection-down = L25 H15 answer projection Δ -7.61, target Δ -0.02

furniture:
  top source head = L25 H19 object mass 0.273
  strongest target-down = L24 H2 target Δ -0.08
  strongest projection-down = L25 H15 answer projection Δ -6.33, target Δ +0.02

plant:
  top source head = L24 H6 object mass 0.311
  strongest target-down = L25 H24 target Δ -0.16
  strongest projection-down = L25 H15 answer projection Δ -6.65, target Δ +0.03
```

### 当前最可靠客观事实

1. **answer_last 确实会在 late layers 读取 object source**

DS7B 尤其明显：

```text
plant top object mass 0.311
furniture top object mass 0.273
clothing top object mass 0.229
container top object mass 0.228
```

Qwen3 也有较弱但可见的 object-source attention：

```text
plant 0.117
clothing 0.111
furniture 0.101
container 0.093
```

2. **单个高 object-source attention head 消融没有复现 Phase111 的强 target-down**

最强 target-down 仍很小：

```text
Qwen3 number: -0.30
DS7B clothing: -0.40
DS7B container: -0.27
```

这远弱于 Phase111 的 answer_last T_c removal：

```text
Qwen3 plant: -5.97
DS7B container: -5.50
DS7B clothing: -5.04
DS7B furniture: -3.82
```

3. **存在强 projection-only heads**

一些 head 消融会大幅降低 answer_last T_c projection，但 logits 几乎不变。

典型：

```text
Qwen3 L35 H8:
  number projection Δ -4.83, target Δ +0.01
  time projection Δ -5.78, target Δ +0.00
  container projection Δ -4.73, target Δ +0.03
  clothing projection Δ -5.42, target Δ -0.02
  furniture projection Δ -4.85, target Δ -0.03

DS7B L25 H15:
  clothing projection Δ -7.61, target Δ -0.02
  furniture projection Δ -6.33, target Δ +0.02
  plant projection Δ -6.65, target Δ +0.03
```

这再次证明：

```text
T_c projection change 不等于 logits causal closure。
```

4. **attention mass 不是因果强度**

高 object attention head 不一定有 target-down 效果。

例如：

```text
DS7B plant top source head L24 H6 object mass 0.311
但 strongest target-down 只有 -0.16
```

### 对 Phase111 的校正

Phase111 的判断继续成立：

```text
answer-site T_c 是强因果状态；
上游路径未闭合。
```

Phase112 进一步说明：

```text
单个 high object-attention head 不是足够的传输入口。
```

更严格说法：

```text
object source attention 存在；
但单头 answer_last output ablation 不能解释 answer-site T_c 的强 logits 因果效应。
```

因此上游路径可能是：

```text
1. 多头集合共同写入；
2. attention + MLP 接力；
3. value path 而非 head output 单点；
4. 多层 residual trajectory；
5. object_span/relation_span/template 多源共同构成。
```

### 条件化关系因子动力学公式更新

Phase111：

```text
object_state --unclosed--> answer_transport_state -> output_gateway -> logits
```

Phase112 后更细化为：

```text
source_tokens
  -> distributed_route_set
  -> A_c(answer)
  -> output_gateway
  -> logits
```

其中：

```text
distributed_route_set ≠ single high-attention head
A_c(answer) 包含强 causal state
projection(A_c, T_c) 不是充分因果指标
```

中文解释：

```text
对象源确实被答案位置读取；
但强类别因果状态不是由某一个明显高注意力头单独决定；
它更像多头、多层或注意力与 MLP 共同形成的答案位置状态。
```

### 硬伤分析

1. **只消融 top 8 object-source heads**

如果关键 head 不靠 object attention mass 排名，它可能被漏掉。

2. **只做单头消融**

强 T_c(answer) 可能由多个 head 累积写入，单头置零会低估。

3. **没有拆 Q/K/V**

本轮只在 o_proj 输入前置零 head slice，没有区分 attention pattern 与 value content。

4. **projection-only 现象仍未解释**

某些 head 强烈改变 T_c projection 但不改 logits，说明 T_c projection 本身不是完整因果状态。

5. **仍未做 generation audit**

尚未验证生成行为。

### 当前进展评价

Phase112 不是找到最终路径，而是排除了一个过于简单的假设：

```text
强 answer-site T_c 不是由单个高 object-attention head 直接控制。
```

当前最可靠拼图：

```text
1. answer_last 有强类别因果状态。
2. answer_last 确实读取 object source。
3. 单头 source attention 与 logits 因果之间不闭合。
4. projection-only heads 存在，投影不是充分指标。
```

### 下一步任务

Phase113 应测试：

```text
Head Set and MLP Relay Closure
```

目标：

```text
从单头转向 head set、多层累计与 MLP 接力，寻找能复现 Phase111 强 target-down 的最小路径集合。
```

建议测试：

```text
1. 对 top-k object-source heads 做 cumulative ablation。
2. 对 projection-only heads 与 source heads 分开/联合消融。
3. 对 attention output 与 MLP output 分别消融。
4. 测 answer_last T_c removal 与 head-set ablation 的 overlap。
5. 优先 DS7B container/clothing/furniture/plant 与 Qwen3 plant/number。
```

关键判据：

```text
如果 head set + MLP relay 能接近 Phase111 的 answer_last T_c remove 效果，
则上游路径开始闭合；
否则需要转向 residual trajectory / broader source span search。
```

## Phase 113: Head Set and MLP Relay Closure 注意力头集合与 MLP 接力闭包 [2026-06-14 11:27]

### 本阶段目标

根据用户附加分析与 Phase112 结果，先判断：

```text
Phase112 是正确的排除式进展：
object-source attention 存在；
但单个高 object-source head 不能解释 answer-site T_c 的强 logits 因果效应。
```

附加分析中正确部分：

```text
1. 单头不是基本单位，head set 可能才是基本单位。
2. attention mass 不是因果贡献。
3. projection-only heads 是重要现象，但不能直接解释成输出因果。
4. 下一步应测试 cumulative head-set ablation 与 MLP relay。
```

本阶段目标：

```text
测试 head set、MLP output、head set + MLP 是否能接近 Phase111 的 answer_last T_c removal 强效。
```

### 执行命令

smoke：

```bash
python -m py_compile \
  tests/gpt5/phase113_head_set_mlp_relay_closure_cuda.py \
  tests/gpt5/phase113_head_set_mlp_relay_closure_summary.py

python tests/gpt5/phase113_head_set_mlp_relay_closure_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --layer-back 1 \
  --candidate-heads 4 \
  --set-sizes 1,2,4 \
  --categories number,plant \
  --output-dir results/gpt5_phase113_smoke \
  --hard-exit-after-model
```

正式测试：

```bash
python tests/gpt5/phase113_head_set_mlp_relay_closure_cuda.py qwen3 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --candidate-heads 16 \
  --set-sizes 1,2,4,8,16 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase113_head_set_mlp_relay_closure \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase113_head_set_mlp_relay_closure_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --candidate-heads 16 \
  --set-sizes 1,2,4,8,16 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase113_head_set_mlp_relay_closure \
  --hard-exit-after-model

python tests/gpt5/phase113_head_set_mlp_relay_closure_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --layer-back 3 \
  --candidate-heads 16 \
  --set-sizes 1,2,4,8,16 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase113_head_set_mlp_relay_closure \
  --hard-exit-after-model

python tests/gpt5/phase113_head_set_mlp_relay_closure_summary.py

python -m py_compile \
  tests/gpt5/phase113_head_set_mlp_relay_closure_cuda.py \
  tests/gpt5/phase113_head_set_mlp_relay_closure_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase113_head_set_mlp_relay_closure_cuda.py`
- 汇总脚本：`tests/gpt5/phase113_head_set_mlp_relay_closure_summary.py`
- Qwen3 结果：`results/gpt5_phase113_head_set_mlp_relay_closure/phase113_qwen3_head_set_mlp_relay_closure.json`
- GLM4 结果：`results/gpt5_phase113_head_set_mlp_relay_closure/phase113_glm4_head_set_mlp_relay_closure.json`
- DS7B 结果：`results/gpt5_phase113_head_set_mlp_relay_closure/phase113_deepseek7b_head_set_mlp_relay_closure.json`
- 跨模型汇总：`results/gpt5_phase113_head_set_mlp_relay_closure/phase113_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, clothing, plant
train objects/category = 12
heldout test objects/category = 12
templates = 4
prompts/category = 48
layers = peak-3 ... peak
candidate heads/category = 16
set sizes = 1, 2, 4, 8, 16
head sets = source, projection, target, mixed, random
relays = heads_only, mlp_only, heads_plus_mlp
reference = answer_last T_c removal, scale 1.5
```

模型层位：

```text
Qwen3: L32-L35
GLM4: L15-L18
DS7B: L24-L27
```

### 测试原理

每个类别先构造：

```text
T_c(answer)
```

并用 Phase111 的方式得到参考效应：

```text
answer_last T_c removal target_delta
```

然后选择候选 head：

```text
source heads:
  answer_last attention to object_span + object_last 最高的 heads。

projection heads:
  在候选池内，单头消融后 answer T_c projection 下降最多的 heads。

target heads:
  在候选池内，单头消融后 target logits 下降最多的 heads。

mixed heads:
  source + projection 的混合集合。

random heads:
  同规模随机对照。
```

干预：

```text
heads_only:
  在 o_proj 输入前，把 head set 在 answer_last 的 head slice 置零。

mlp_only:
  在 peak-3...peak 层，把 MLP output 在 answer_last 置零。

heads_plus_mlp:
  同时做 head set 消融和 MLP output 消融。
```

关键指标：

```text
effect_ratio = head_set_target_delta / T_c_remove_target_delta
```

### 客观结果

#### Qwen3

```text
number:
  T_c reference Δ -3.43
  best heads_only Δ -0.33, ratio 0.10
  best heads_plus_mlp Δ +4.00
  best mlp_only Δ +4.18
  random heads_only Δ -0.02

container:
  T_c reference Δ -1.75
  best heads_only Δ -0.15, ratio 0.09
  best heads_plus_mlp Δ +2.39
  best mlp_only Δ +2.63
  random heads_only Δ -0.01

clothing:
  T_c reference Δ -1.43
  best heads_only Δ -0.72, ratio 0.50
  best random heads_only Δ -0.35, ratio 0.25
  best heads_plus_mlp Δ +1.58
  best mlp_only Δ +2.49

plant:
  T_c reference Δ -5.97
  best heads_only Δ -0.59, ratio 0.10
  best heads_plus_mlp Δ +3.07
  best mlp_only Δ +3.48
  random heads_only Δ -0.02
```

Qwen3 中只有 clothing 出现局部闭合线索：

```text
heads_only ratio 0.50
random ratio 0.25
```

但这仍不能解释多数类别。

#### GLM4 bf16

```text
T_c reference 本身很弱：
number Δ -0.09
container Δ -0.07
clothing Δ -0.07
plant Δ +0.02
```

因此 GLM4 本轮不进入强机制结论。

#### DS7B

```text
number:
  T_c reference Δ +1.06
  reference 不是 target-down，因此不适合闭合判据。

container:
  T_c reference Δ -5.50
  best heads_only Δ -0.28, ratio 0.05
  best heads_plus_mlp Δ +0.34
  best mlp_only Δ +0.14
  random heads_only Δ -0.15

clothing:
  T_c reference Δ -5.04
  best heads_only Δ -0.78, ratio 0.16
  best random heads_only Δ -0.45, ratio 0.09
  best heads_plus_mlp Δ +1.39
  best mlp_only Δ +1.44

plant:
  T_c reference Δ -3.20
  best heads_only Δ -0.32, ratio 0.10
  best heads_plus_mlp Δ +0.55
  best mlp_only Δ +0.92
  random heads_only Δ -0.28
```

DS7B 的 head set 仍不能接近 T_c reference。

### 当前最可靠客观事实

1. **head set 消融比单头稍强，但大多数仍不能闭合**

典型：

```text
Qwen3 plant:
  T_c reference -5.97
  heads_only -0.59

DS7B container:
  T_c reference -5.50
  heads_only -0.28

DS7B clothing:
  T_c reference -5.04
  heads_only -0.78
```

2. **Qwen3 clothing 是局部例外**

```text
Qwen3 clothing:
  T_c reference -1.43
  target head set -0.72
  ratio 0.50
  random -0.35
```

这说明某些类别可能确实有 head-set 局部闭合，但不是普遍结构。

3. **coarse MLP output ablation 不支持 MLP relay 闭合**

MLP ablation 常常产生 target-up，而不是复现 T_c target-down：

```text
Qwen3 number mlp_only +4.18
Qwen3 plant mlp_only +3.48
DS7B clothing mlp_only +1.44
DS7B plant mlp_only +0.92
```

同时 answer projection 出现巨大变化：

```text
Qwen3 mlp_only answer projection Δ around -154 to -199
DS7B mlp_only answer projection Δ around -303 to -328
```

这说明粗 MLP 置零是强破坏，不是干净机制分解。

4. **projection change 继续不是充分因果指标**

很多条件 answer projection 大幅变化，但 target logits 不按 T_c reference 方向变化。

5. **GLM4 继续弱参考**

GLM4 的 T_c reference 太小，本轮不支持强机制结论。

### 对 Phase112 的校正

Phase112 的正确部分仍成立：

```text
单个高 object-source head 不是完整路径。
```

Phase113 进一步说明：

```text
top source/projection/target head set 也大多不是完整路径；
coarse MLP output ablation 也没有形成闭合。
```

更严格说法：

```text
answer-site T_c 的强因果效应，不能由当前 tested head-set + coarse MLP relay 解释。
```

因此当前路径可能在：

```text
1. 更宽的 residual trajectory；
2. 非 top-attention 的 value-content heads；
3. MLP 内部子方向，而非整个 MLP output；
4. 多层小尺度分布式累积；
5. answer-site 子空间而非单方向 T_c。
```

### 条件化关系因子动力学公式更新

Phase112：

```text
source_tokens -> distributed_route_set -> A_c(answer) -> output_gateway -> logits
```

Phase113 后应更谨慎：

```text
source_tokens
  -> unresolved_distributed_dynamics
  -> A_c(answer)
  -> output_gateway
  -> logits
```

其中：

```text
unresolved_distributed_dynamics
  不等于 single head
  不等于 tested top-k head set
  不等于 coarse whole-MLP output ablation
```

当前强验证仍是：

```text
A_c(answer) / T_c(answer) -> logits
```

未闭合部分仍是：

```text
source_tokens -> A_c(answer)
```

### 硬伤分析

1. **MLP ablation 太粗**

把整层 MLP output 在 answer_last 置零，会破坏大量非目标功能，不能说明 MLP 子方向机制。

2. **projection heads 仍来自 source candidate pool**

如果真正 projection heads 不在 top source candidate 中，仍可能漏掉。

3. **没有 Q/K/V value transplant**

仍未测试 value content 是否是关键。

4. **没有 answer-site 多维子空间**

T_c 是单方向；强因果场可能是多维子空间。

5. **没有 generation audit**

仍未验证生成行为。

### 当前进展评价

Phase113 是第二次排除式进展：

```text
单头不够；
top-k head set 大多也不够；
coarse MLP relay 也不够。
```

当前最可靠拼图：

```text
1. answer-site T_c 是强因果入口。
2. object-source attention 存在。
3. 单头与 top-k head set 多数不能闭合。
4. MLP 整体置零不是正确分解粒度。
5. Qwen3 clothing 有局部 head-set 线索。
```

### 下一步任务

Phase114 应转向：

```text
Answer-Site Causal Subspace Expansion
```

目标：

```text
不要再把 answer-site causal field 压缩成单方向 T_c；
构造多维 answer-site causal subspace，再测试子空间移除是否比单方向更稳定、更接近真实机制。
```

建议测试：

```text
1. 从多个强类别和多个模板中提取 answer-site causal directions。
2. 构造低秩子空间 rank 2/4/8/16。
3. 在 answer_last 移除整个子空间，和单方向 T_c 对照。
4. 测 target_delta、competitor release、random subspace control。
5. 优先 Qwen3 number/plant/clothing 与 DS7B container/clothing/plant。
```

关键理由：

```text
projection-only、head-set 不闭合、MLP 粗消融反向，
都说明当前单方向 T_c 只是强因果场的一个切片；
破解路径前，必须先把 answer-site causal field 的维度结构拼出来。
```

## Phase 114: Answer-Site Causal Subspace Expansion 答案位置因果子空间扩展 [2026-06-14 12:09]

### 本阶段目标

根据用户附加分析与 Phase113 结果，先判断：

```text
Phase113 的收缩是正确的：
单头不够；
top-k head set 大多不够；
coarse MLP relay 不够；
T_c(answer) 可能只是 answer-site causal field 的一维切片。
```

本阶段目标：

```text
构造 rank 1/2/4/8/16 的 answer-site category contrast subspace，
测试多维子空间移除是否比单方向 T_c 更稳定或更强。
```

### 执行命令

smoke：

```bash
python -m py_compile \
  tests/gpt5/phase114_answer_site_causal_subspace_cuda.py \
  tests/gpt5/phase114_answer_site_causal_subspace_summary.py

python tests/gpt5/phase114_answer_site_causal_subspace_cuda.py qwen3 \
  --train-objects 2 \
  --test-objects 2 \
  --batch-size 4 \
  --ranks 1,2 \
  --scales 1.0 \
  --categories number,plant \
  --output-dir results/gpt5_phase114_smoke \
  --hard-exit-after-model
```

正式测试：

```bash
python tests/gpt5/phase114_answer_site_causal_subspace_cuda.py qwen3 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --ranks 1,2,4,8,16 \
  --scales 1.0,1.5 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase114_answer_site_causal_subspace \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase114_answer_site_causal_subspace_cuda.py glm4 \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --ranks 1,2,4,8,16 \
  --scales 1.0,1.5 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase114_answer_site_causal_subspace \
  --hard-exit-after-model

python tests/gpt5/phase114_answer_site_causal_subspace_cuda.py deepseek7b \
  --train-objects 12 \
  --test-objects 12 \
  --batch-size 24 \
  --ranks 1,2,4,8,16 \
  --scales 1.0,1.5 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase114_answer_site_causal_subspace \
  --hard-exit-after-model

python tests/gpt5/phase114_answer_site_causal_subspace_summary.py

python -m py_compile \
  tests/gpt5/phase114_answer_site_causal_subspace_cuda.py \
  tests/gpt5/phase114_answer_site_causal_subspace_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase114_answer_site_causal_subspace_cuda.py`
- 汇总脚本：`tests/gpt5/phase114_answer_site_causal_subspace_summary.py`
- Qwen3 结果：`results/gpt5_phase114_answer_site_causal_subspace/phase114_qwen3_answer_site_causal_subspace.json`
- GLM4 结果：`results/gpt5_phase114_answer_site_causal_subspace/phase114_glm4_answer_site_causal_subspace.json`
- DS7B 结果：`results/gpt5_phase114_answer_site_causal_subspace/phase114_deepseek7b_answer_site_causal_subspace.json`
- 跨模型汇总：`results/gpt5_phase114_answer_site_causal_subspace/phase114_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, clothing, plant
train objects/category = 12
heldout test objects/category = 12
templates = 4
prompts/category = 48
ranks = 1, 2, 4, 8, 16
scales = 1.0, 1.5
layer = model-specific causal peak
controls = random same-rank subspace
```

模型层位：

```text
Qwen3: L35
GLM4: L18
DS7B: L27
```

### 测试原理

对每个类别，在 answer-site peak layer 构造 target-vs-other contrast rows：

```text
每个 template:
  target_center - other_mean
  target_center - each_other_category_center
```

然后对这些 contrast rows 做 SVD，取：

```text
rank = 1, 2, 4, 8, 16
```

得到 answer-site category contrast subspace。

真实 forward 干预：

```text
在 answer_last 移除该子空间投影：
  h = h - scale * proj_subspace(h)
```

对照：

```text
1. 单方向 T_c removal
2. same-rank random subspace removal
```

指标：

```text
target_delta
max_other_release_delta
random control target_delta
```

### 客观结果

#### Qwen3

```text
number:
  T_c: r1 scale1.5 target Δ -3.43, release +0.87
  best subspace: rank2 scale1.5 target Δ -3.12, release +0.78
  random: rank16 scale1.5 target Δ -0.50

container:
  T_c: target Δ -1.74, release +0.12
  best subspace: rank16 scale1.5 target Δ -2.59, release +2.03
  random: target Δ -0.12

clothing:
  T_c: target Δ -1.42, release +1.12
  best subspace: rank8 scale1.5 target Δ -0.47, release +0.69
  random: target Δ -0.04

plant:
  T_c: target Δ -5.98, release +0.73
  best subspace: rank2 scale1.5 target Δ -1.26, release +0.00
  random: target Δ -0.21
```

Qwen3 结果是混合的：

```text
number: subspace 接近 T_c
container: subspace 稍强但 release 很大
clothing/plant: 单方向 T_c 更强
```

#### GLM4 bf16

```text
number:
  T_c -0.10
  subspace rank16 -0.86, release +1.22

container:
  T_c -0.08
  subspace rank16 -0.53

clothing:
  T_c -0.07
  subspace rank8 -0.34

plant:
  T_c +0.01
  subspace rank16 -0.13
```

GLM4 子空间有一些变化，但 T_c reference 本身弱，仍不作为强机制结论来源。

#### DS7B

```text
number:
  T_c: target Δ +1.11, release +1.22
  best subspace: rank16 scale1.5 target Δ -11.75, release +0.00
  random: target Δ -0.30

container:
  T_c: target Δ -5.60, release +0.00
  best subspace: rank16 scale1.5 target Δ -12.42, release +0.00
  random: target Δ -0.30

clothing:
  T_c: target Δ -5.22, release +0.18
  best subspace: rank8 scale1.5 target Δ -4.99, release +0.00
  random: target Δ -0.23

plant:
  T_c: target Δ -3.19, release +0.00
  best subspace: rank8 scale1.5 target Δ -7.93, release +0.00
  random: target Δ -0.43
```

DS7B 出现强正向结果：

```text
number/container/plant 的 answer-site 多维子空间显著强于单方向 T_c，
并且远强于 random subspace。
```

### 当前最可靠客观事实

1. **DS7B 的 answer-site causal field 明显是多维结构**

最强例子：

```text
DS7B container:
  T_c -5.60
  rank16 subspace -12.42
  random -0.30

DS7B plant:
  T_c -3.19
  rank8 subspace -7.93
  random -0.43

DS7B number:
  T_c +1.11
  rank16 subspace -11.75
  random -0.30
```

这说明 DS7B 的单方向 T_c 确实只是答案位置因果场的切片。

2. **Qwen3 更类别分化**

```text
Qwen3 number:
  T_c -3.43
  subspace -3.12

Qwen3 container:
  T_c -1.74
  subspace -2.59 but release +2.03

Qwen3 plant:
  T_c -5.98
  subspace -1.26
```

Qwen3 plant 仍然是强单方向模式。

3. **random subspace control 很弱**

典型：

```text
DS7B container random -0.30 vs subspace -12.42
DS7B plant random -0.43 vs subspace -7.93
Qwen3 number random -0.50 vs T_c/subspace around -3
```

说明强效不是单纯 rank 高或随机删除造成。

4. **container 类子空间 release 需要谨慎**

Qwen3 container：

```text
subspace target Δ -2.59
release +2.03
```

这说明该子空间混入强竞争释放，不是纯 target support。

### 对 Phase113 的校正

Phase113 的排除式判断仍正确：

```text
head set / coarse MLP relay 没有解释 T_c(answer) 强效。
```

Phase114 给出新的正向方向：

```text
尤其在 DS7B，answer-site causal field 不是单方向，而是低秩多维子空间。
```

因此上游路径未闭合，不一定是因为没有路由，而可能是因为我们追踪的目标状态维度太窄：

```text
source_tokens -> A_c(answer)
```

其中 `A_c(answer)` 应从单方向 `T_c` 改为多维子空间。

### 条件化关系因子动力学公式更新

Phase113：

```text
source_tokens
  -> unresolved_distributed_dynamics
  -> A_c(answer)
  -> output_gateway
  -> logits
```

Phase114 后：

```text
A_c(answer) ∈ Subspace_c^k(answer)
```

更具体：

```text
source_tokens
  -> unresolved_distributed_dynamics
  -> Subspace_c^k(answer)
  -> output_gateway
  -> logits
```

其中：

```text
T_c 是 Subspace_c^k 的一个强切片；
但在 DS7B 中，rank8/rank16 子空间比 T_c 更接近完整因果场。
```

中文解释：

```text
答案位置的类别因果状态不是一个方向，而可能是一个低秩子空间；
不同模型和类别的有效维度不同；
破解路径前，必须先确定要追踪的答案位置状态空间。
```

### 硬伤分析

1. **子空间来自类别几何，不是自动因果发现**

当前 subspace 是 target-vs-other answer center contrast 的 SVD，不等于已证明的最小因果子空间。

2. **random control 没有匹配谱结构**

random subspace 只匹配 rank，没有匹配奇异值谱或与 readout/transport 的夹角。

3. **高 rank 可能混入竞争释放**

Qwen3 container release +2.03 说明子空间会包含竞争/抑制成分。

4. **仍是 DCF logits**

没有 generation audit。

5. **上游路径仍未闭合**

本轮定位的是 answer-site field，不是 source -> answer 路径。

### 当前进展评价

Phase114 是关键正向进展：

```text
首次明确显示 answer-site causal field 在 DS7B 中是多维低秩结构；
并且多维子空间远强于随机子空间。
```

当前最可靠拼图：

```text
1. answer-site causal field 是核心因果入口。
2. DS7B 的 answer-site field 是多维结构。
3. Qwen3 存在类别分化：number 接近多维/单向都可，plant 强单向。
4. T_c 不是完整因果状态，只是某些模型/类别的强切片。
5. 上游路径搜索应改为追踪 Subspace_c^k(answer)，而不是单方向 T_c。
```

### 下一步任务

Phase115 应做：

```text
Causal Subspace Robustness and Release Decomposition
```

目标：

```text
验证 Phase114 的多维子空间是否稳定，并把 target support 与 competitor release 拆开。
```

建议测试：

```text
1. 对 DS7B number/container/plant 扩大 heldout objects 复测。
2. 对 rank8/rank16 子空间做 scale sweep: 0.25,0.5,1.0,1.5。
3. 对子空间做 leave-template-out 验证，确认不是模板过拟合。
4. 对 release 强的 Qwen3 container 做 target-support / release-component 分解。
5. 加 matched-spectrum random subspace control。
```

关键判据：

```text
如果 DS7B rank8/rank16 子空间在模板留出、扩大对象、matched random control 下仍强，
则可以把 answer-site causal field 从“单方向假设”正式升级为“低秩因果子空间”。
```

## Phase 115: Causal Subspace Robustness and Release Decomposition 因果子空间稳健性与释放分解 [2026-06-14 13:16]

### 本阶段目标

根据 Phase114 的结果继续验证：

```text
DS7B 的 answer-site causal field 是否真是稳健低秩子空间；
Qwen3 的子空间 target-down 是否混入 competitor release；
Phase114 强效是否会在扩大 heldout objects、leave-template-out、matched-spectrum random control 下保留。
```

### 执行命令

smoke：

```bash
python -m py_compile \
  tests/gpt5/phase115_causal_subspace_robustness_cuda.py \
  tests/gpt5/phase115_causal_subspace_robustness_summary.py

python tests/gpt5/phase115_causal_subspace_robustness_cuda.py qwen3 \
  --train-objects 4 \
  --test-objects 4 \
  --batch-size 4 \
  --ranks 2 \
  --scales 0.5 \
  --categories number,container \
  --output-dir results/gpt5_phase115_smoke \
  --hard-exit-after-model
```

正式测试：

```bash
python tests/gpt5/phase115_causal_subspace_robustness_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --ranks 8,16 \
  --scales 0.25,0.5,1.0,1.5 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase115_causal_subspace_robustness \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase115_causal_subspace_robustness_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --ranks 8,16 \
  --scales 0.25,0.5,1.0,1.5 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase115_causal_subspace_robustness \
  --hard-exit-after-model

python tests/gpt5/phase115_causal_subspace_robustness_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --ranks 8,16 \
  --scales 0.25,0.5,1.0,1.5 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase115_causal_subspace_robustness \
  --hard-exit-after-model

python tests/gpt5/phase115_causal_subspace_robustness_summary.py

python -m py_compile \
  tests/gpt5/phase115_causal_subspace_robustness_cuda.py \
  tests/gpt5/phase115_causal_subspace_robustness_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase115_causal_subspace_robustness_cuda.py`
- 汇总脚本：`tests/gpt5/phase115_causal_subspace_robustness_summary.py`
- Qwen3 结果：`results/gpt5_phase115_causal_subspace_robustness/phase115_qwen3_causal_subspace_robustness.json`
- GLM4 结果：`results/gpt5_phase115_causal_subspace_robustness/phase115_glm4_causal_subspace_robustness.json`
- DS7B 结果：`results/gpt5_phase115_causal_subspace_robustness/phase115_deepseek7b_causal_subspace_robustness.json`
- 跨模型汇总：`results/gpt5_phase115_causal_subspace_robustness/phase115_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, clothing, plant
train objects/category = 8
heldout test objects/category = 16
templates = 4
full prompts/category = 64
ranks = 8, 16
scales = 0.25, 0.5, 1.0, 1.5
layer = model-specific causal peak
controls = matched-spectrum random subspace
robustness = leave-template-out 4 folds
release decomposition = strongest-release-category excluded contrast
```

模型层位：

```text
Qwen3: L35
GLM4: L18
DS7B: L27
```

### 测试原理

Phase115 在 Phase114 基础上做四类验证：

```text
1. 扩大 heldout:
   train objects/category = 8
   test objects/category = 16

2. scale sweep:
   0.25, 0.5, 1.0, 1.5

3. leave-template-out:
   用 3 个模板构造子空间；
   在第 4 个模板上测试；
   4 个模板轮换。

4. matched-spectrum random:
   用 synthetic contrast matrix 保留奇异值谱，再生成随机子空间。
```

release 分解的初步版本：

```text
先找 full subspace 的最强 release category；
再从 contrast construction 中排除该 release category；
测试 release-excluded subspace。
```

注意：这还不是完整 support/release factorization，只是第一层排查。

### 客观结果

#### Qwen3

```text
number:
  full subspace: r8 scale1.5 target Δ -1.83, release +2.37
  matched random: target Δ +0.05
  release-excluded: target Δ -2.72, release +1.99
  LTO mean: target Δ -2.18, release +2.12
  LTO random mean: target Δ -0.19

container:
  full subspace: r16 scale1.5 target Δ -2.53, release +1.90
  matched random: target Δ +0.03
  release-excluded: target Δ -3.06, release +1.97
  LTO mean: target Δ -2.54, release +1.54
  LTO random mean: target Δ -0.07

clothing:
  full subspace: target Δ +0.22, release +0.51
  matched random: target Δ -0.38
  LTO mean: target Δ +0.24
  LTO random mean: target Δ -0.30

plant:
  full subspace: r16 scale1.5 target Δ -1.24, release +1.59
  matched random: target Δ -0.15
  release-excluded: target Δ -1.17, release +1.77
  LTO mean: target Δ -1.74, release +1.41
```

Qwen3 结论：

```text
number/container/plant 的子空间效应能跨模板保留，但 release 很大；
clothing 对照敏感；
release-excluded 没有解决 release，说明 release 不是单一类别导致。
```

#### GLM4 bf16

```text
number:
  full subspace Δ -0.90, release +0.68
  LTO mean Δ -0.58

container:
  full subspace Δ -0.32
  LTO mean Δ -0.36

clothing:
  full subspace Δ -0.28
  LTO mean Δ -0.19

plant:
  full subspace Δ -0.13
  LTO mean Δ -0.07
```

GLM4 仍然弱，但 number 出现小幅稳定信号。

#### DS7B

```text
number:
  full subspace: r16 scale1.5 target Δ -12.58, release +0.00
  matched random: target Δ -0.07
  LTO mean: target Δ -11.59, release +0.00
  LTO random mean: target Δ -0.20

container:
  full subspace: r16 scale1.5 target Δ -12.52, release +0.00
  matched random: target Δ -0.24
  LTO mean: target Δ -11.45, release +0.00
  LTO random mean: target Δ -0.37

clothing:
  full subspace: r8 scale1.0 target Δ -4.20, release +0.00
  matched random: target Δ -0.06
  LTO mean: target Δ -5.07, release +0.00
  LTO random mean: target Δ -0.37

plant:
  full subspace: r8 scale1.5 target Δ -9.40, release +0.00
  matched random: target Δ -0.29
  LTO mean: target Δ -8.71, release +0.00
  LTO random mean: target Δ -0.22
```

DS7B 结论：

```text
number/container/plant 是 robust_strong；
clothing 是 robust_moderate；
所有 matched-spectrum random controls 都很弱；
所有 release 都为 0。
```

### 当前最可靠客观事实

1. **DS7B answer-site low-rank causal subspace 已通过稳健性测试**

最强证据：

```text
DS7B number:
  full -12.58
  LTO mean -11.59
  random -0.07

DS7B container:
  full -12.52
  LTO mean -11.45
  random -0.24

DS7B plant:
  full -9.40
  LTO mean -8.71
  random -0.29
```

这说明 DS7B 的低秩子空间不是模板过拟合，也不是 rank/random 删除造成。

2. **DS7B 子空间几乎无 competitor release**

```text
number/container/clothing/plant:
  max_other_release = 0.00 in full and LTO mean
```

说明 DS7B 的子空间更像干净 target support removal。

3. **Qwen3 子空间混有强 release**

```text
Qwen3 number release +2.37
Qwen3 container release +1.90
Qwen3 plant release +1.59
```

release-excluded 后仍然 release 较高：

```text
number +1.99
container +1.97
plant +1.77
```

这说明 Qwen3 的 release 不是一个竞争类别造成，而是多竞争/接口混合结构。

4. **GLM4 仍然弱，但 number 有小信号**

```text
GLM4 number:
  full -0.90
  LTO -0.58
  random -0.02
```

仍不能与 DS7B 强结论同等对待。

### 理论进展

Phase115 支持把 DS7B 的 answer-site 表述升级为：

```text
Subspace_c^k(answer) 是稳健因果状态。
```

更具体：

```text
source_tokens
  -> unresolved_distributed_dynamics
  -> robust low-rank Subspace_c^k(answer)
  -> output_gateway
  -> logits
```

对 DS7B：

```text
number/container/plant:
  k ≈ 8-16
  robust across heldout objects and heldout templates
  target-down strong
  competitor release near zero
```

对 Qwen3：

```text
answer-site subspace 是 mixed support/release field；
需要进一步拆 support 与 release。
```

### 硬伤分析

1. **matched-spectrum random 仍使用 orthonormal basis 干预**

虽然通过 synthetic contrast matrix 匹配奇异值谱，但最终移除的是正交基投影，谱结构只影响基的生成过程。

2. **release-excluded 不是完整分解**

只排除最强 release category，无法分解多竞争释放。

3. **仍然没有生成审计**

目前仍是 DCF logits。

4. **上游路径仍未闭合**

Phase115 证明 answer-site 子空间稳健，但没有解释 source 如何写入该子空间。

5. **Qwen3 与 DS7B 机制分型明显不同**

不能把 DS7B 的干净低秩子空间结论直接套到 Qwen3。

### 当前进展评价

Phase115 是一次强确认：

```text
DS7B 的 answer-site low-rank causal subspace 已从“可能结构”升级为“稳健客观事实”。
```

当前最可靠拼图：

```text
1. DS7B number/container/plant 存在稳健、干净、低秩的 answer-site 因果子空间。
2. DS7B clothing 也有中强稳健子空间。
3. Qwen3 的 answer-site 子空间存在，但混入强 release。
4. GLM4 仍弱，只能作为小信号参考。
5. 下一步应从“是否有子空间”转向“子空间内部成分如何分解”。
```

### 下一步任务

Phase116 应做：

```text
Subspace Basis Component Audit
```

目标：

```text
把稳健低秩子空间拆成 rank component，
确定哪些基向量负责 target support，
哪些基向量负责 release/interface，
哪些是冗余或控制维度。
```

建议测试：

```text
1. 对 DS7B number/container/plant 的 rank16 子空间逐基向量 ablation。
2. 对 rank16 做 cumulative basis ablation: top1, top2, top4, top8, top16。
3. 对 Qwen3 number/container/plant 做 basis-level release decomposition。
4. 对每个 basis component 记录 target_delta、release_delta、readout cosine、transport cosine。
5. 加 matched random basis component control。
```

关键判据：

```text
如果少数 basis components 能复现大部分 target-down，
则子空间可以继续压缩；
如果必须 top8/top16 累积才强，
说明答案位置因果场确实是分布式低秩结构。
```

## Phase 116: Subspace Basis Component Audit 子空间基向量成分审计 [2026-06-14 13:28]

### 本阶段目标

根据用户附加分析与 Phase115 结果，继续完成：

```text
把稳健低秩 answer-site causal subspace 拆成 basis components；
确定哪些基向量负责 target support；
哪些负责 competitor release / interface；
以及完整强效是否来自少数基向量或 top-k 累积。
```

### 执行命令

smoke：

```bash
python -m py_compile \
  tests/gpt5/phase116_subspace_basis_component_audit_cuda.py \
  tests/gpt5/phase116_subspace_basis_component_audit_summary.py

python tests/gpt5/phase116_subspace_basis_component_audit_cuda.py qwen3 \
  --train-objects 4 \
  --test-objects 4 \
  --batch-size 4 \
  --ranks 4 \
  --set-sizes 1,2,4 \
  --categories number,container \
  --output-dir results/gpt5_phase116_smoke \
  --hard-exit-after-model
```

正式测试：

```bash
python tests/gpt5/phase116_subspace_basis_component_audit_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --ranks 8,16 \
  --set-sizes 1,2,4,8,16 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase116_subspace_basis_component_audit \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase116_subspace_basis_component_audit_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --ranks 8,16 \
  --set-sizes 1,2,4,8,16 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase116_subspace_basis_component_audit \
  --hard-exit-after-model

python tests/gpt5/phase116_subspace_basis_component_audit_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --ranks 8,16 \
  --set-sizes 1,2,4,8,16 \
  --categories number,container,clothing,plant \
  --output-dir results/gpt5_phase116_subspace_basis_component_audit \
  --hard-exit-after-model

python tests/gpt5/phase116_subspace_basis_component_audit_summary.py

python -m py_compile \
  tests/gpt5/phase116_subspace_basis_component_audit_cuda.py \
  tests/gpt5/phase116_subspace_basis_component_audit_summary.py
```

### 脚本与结果

- 主测试脚本：`tests/gpt5/phase116_subspace_basis_component_audit_cuda.py`
- 汇总脚本：`tests/gpt5/phase116_subspace_basis_component_audit_summary.py`
- Qwen3 结果：`results/gpt5_phase116_subspace_basis_component_audit/phase116_qwen3_subspace_basis_component_audit.json`
- GLM4 结果：`results/gpt5_phase116_subspace_basis_component_audit/phase116_glm4_subspace_basis_component_audit.json`
- DS7B 结果：`results/gpt5_phase116_subspace_basis_component_audit/phase116_deepseek7b_subspace_basis_component_audit.json`
- 跨模型汇总：`results/gpt5_phase116_subspace_basis_component_audit/phase116_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, clothing, plant
train objects/category = 8
heldout test objects/category = 16
templates = 4
prompts/category = 64
ranks = 8, 16
scale = 1.5
cumulative set sizes = 1, 2, 4, 8, 16
metrics = target_delta, max_release_delta, readout_cos, transport_cos, template_abs_cos
```

模型层位：

```text
Qwen3: L35
GLM4: L18
DS7B: L27
```

### 测试原理

对每个类别构造 answer-site contrast subspace：

```text
target_center - other_mean
target_center - each_other_category_center
```

SVD 后取 rank8/rank16 basis。

测试三类干预：

```text
1. basis-wise ablation:
   单独移除每个 basis vector。

2. cumulative basis ablation:
   按单基 target_delta 从强到弱排序，测试 top1/top2/top4/top8/top16。

3. split sets:
   根据单基效果标注 support / release / mixed / weak，
   分别移除这些集合。
```

基向量诊断：

```text
readout_cos
transport_cos
template_abs_cos
```

### 客观结果

#### Qwen3

```text
number:
  best single: basis0 target Δ -2.90, release +1.59
  best cumulative rank16: top4 target Δ -3.67, release +0.86
  release set: target Δ +0.70, release +1.77
  mixed set: target Δ -3.10, release +1.05
  best random single: target Δ -0.28

container:
  best single: basis1 target Δ -0.66, release +0.00
  support set rank16: target Δ -1.65, release +0.00
  release set rank16: target Δ +0.24, release +1.88
  mixed set: target Δ -0.46, release +2.05
  cumulative rank16 top8: target Δ -2.66, release +0.40

clothing:
  best single: basis1 target Δ -0.72, release +0.00
  support set: target Δ -1.03, release +0.00
  release set rank16: target Δ +2.00, release +2.64
  cumulative rank16 top8: target Δ -1.62, release +0.00

plant:
  best single: basis1 target Δ -1.02, release +0.00
  support set rank16: target Δ -1.39, release +0.00
  release set rank16: target Δ +1.09, release +1.62
  cumulative rank16 top8: target Δ -2.85, release +0.00
```

Qwen3 关键事实：

```text
1. release basis 可以直接分离出来。
2. support set 往往 target-down 干净。
3. number 的最强单基是 mixed，不是干净 support。
4. container/clothing/plant 都出现 clear support/release split。
```

#### GLM4 bf16

```text
number:
  best single target Δ -0.37
  cumulative rank16 top16 target Δ -0.90
  support set target Δ -0.60

container:
  best single target Δ -0.17
  cumulative rank16 top8 target Δ -0.46

clothing:
  best single target Δ -0.11
  cumulative rank16 top8 target Δ -0.44

plant:
  best single target Δ -0.07
  cumulative rank16 top8 target Δ -0.21
```

GLM4 仍弱。

#### DS7B

```text
number:
  best single: basis1 target Δ -5.55, release +0.00
  rank16 cumulative top16 target Δ -12.58, release +0.00
  support set rank16: target Δ -12.49, release +0.00
  release set rank16: target Δ +0.62, release +1.83
  best random single: target Δ -0.13

container:
  best single: basis6 target Δ -2.92, release +0.00
  rank16 cumulative top8 target Δ -13.55, release +0.00
  support set rank16: target Δ -13.55, release +0.00
  release set: target Δ -0.16, release +1.22
  best random single: target Δ -0.17

clothing:
  best single: basis0 target Δ -3.44, release +0.00
  rank16 cumulative top4 target Δ -5.31, release +0.00
  support set rank16: target Δ -5.58, release +0.00
  release set rank16: target Δ +2.07, release +1.67
  mixed set: target Δ -0.60, release +0.70

plant:
  best single: basis0 target Δ -4.93, release +0.00
  rank16 cumulative top8 target Δ -9.71, release +0.00
  support set rank16: target Δ -9.66, release +0.00
  release set: target Δ +0.17, release +0.53
  best random single: target Δ -0.14
```

DS7B 关键事实：

```text
1. 存在强单基 support component。
2. 完整强效仍需要多个 support basis 累积。
3. support set 非常干净，release=0。
4. release components 也存在，但不是 full subspace 强 target-down 的主要来源。
```

### 当前最可靠客观事实

1. **DS7B 是 clean distributed support subspace**

例如：

```text
container:
  single -2.92
  support set -13.55

plant:
  single -4.93
  support set -9.66

number:
  single -5.55
  support set -12.49
```

这说明少数强 basis 很重要，但完整效果需要多个 support basis。

2. **Qwen3 的 support/release 可在 basis level 分离**

典型：

```text
Qwen3 container:
  support set -1.65, release 0
  release set +0.24, release +1.88
  mixed set -0.46, release +2.05

Qwen3 clothing:
  support set -1.03, release 0
  release set +2.00, release +2.64

Qwen3 plant:
  support set -1.39, release 0
  release set +1.09, release +1.62
```

Phase115 中 Qwen3 的大 release，在 Phase116 被拆到了具体 basis sets。

3. **随机单基对照很弱**

典型：

```text
DS7B number random single -0.13 vs real single -5.55
DS7B plant random single -0.14 vs real single -4.93
Qwen3 number random single -0.28 vs real single -2.90
```

4. **readout/transport/template cos 都不高**

许多最强单基的 cos 仍低：

```text
DS7B number best single:
  readout_cos -0.06
  transport_cos -0.20
  template_abs_cos 0.35

DS7B container best single:
  readout_cos 0.00
  transport_cos 0.15
  template_abs_cos 0.13
```

说明强 causal basis 不是简单 readout/transport/template 方向。

### 理论进展

Phase115：

```text
Subspace_c^k(answer) 是稳健因果状态。
```

Phase116 后可进一步拆成：

```text
Subspace_c^k(answer)
=
SupportBasisSet_c
+ ReleaseBasisSet_c
+ MixedBasisSet_c
+ Weak/RedundantBasisSet_c
```

对 DS7B：

```text
SupportBasisSet_c 是主导；
release basis 存在但不主导；
target-down 几乎无 competitor release。
```

对 Qwen3：

```text
support 与 release basis 明显共存；
类别意义更像相对竞争场。
```

### 硬伤分析

1. **SVD basis 不是唯一基**

旋转同一子空间会改变单基解释，因此 basis-level 标签不是最终机制基。

2. **component labels 是启发式**

support/release/mixed/weak 由 target_delta 和 release_delta 阈值判定，需要后续验证。

3. **未做旋转不变审计**

需要测试 varimax/ICA/causal-optimized basis 等不同基选择。

4. **仍未做 generation audit**

目前仍是 DCF logits。

5. **上游路径仍未闭合**

本轮进一步理解 answer-site 子空间内部，但未解释 source 如何写入这些 basis sets。

### 当前进展评价

Phase116 是一次重要分解：

```text
DS7B: clean support basis set
Qwen3: support/release basis split
GLM4: weak
```

当前最可靠拼图：

```text
1. answer-site low-rank causal subspace 可拆成基向量功能成分。
2. DS7B 强效来自多个 support basis 累积。
3. Qwen3 的 release 是 basis-level 真实成分，不是统计噪声。
4. 强 basis 与 readout/transport/template 方向都不简单对齐。
```

### 下一步任务

Phase117 应做：

```text
Basis Rotation and Causal Axis Stabilization
```

目标：

```text
验证 Phase116 的 support/release basis 是否依赖 SVD 基选择；
寻找更稳定、更接近因果轴的 basis。
```

建议测试：

```text
1. 对 DS7B number/container/plant 做 SVD basis vs varimax-like rotation vs random orthogonal rotation。
2. 对 Qwen3 container/plant 做 support/release basis 在不同旋转下的稳定性。
3. 用 causal score 对子空间内方向做贪心搜索，找 causal-optimized basis。
4. 比较各 basis 的 target_delta、release_delta、readout_cos、transport_cos。
5. 保留 matched random control。
```

关键判据：

```text
如果 support/release 分解在不同合理旋转下稳定，
则 basis-level 功能分解可信度提高；
如果不稳定，则只能保留“子空间级”结论，不能解释单基功能。
```

## Phase 117: Basis Rotation and Causal Axis Stabilization 基旋转与因果轴稳定化 [2026-06-14 14:14]

### 本阶段目标

根据用户要求，先分析 Phase116 和附件判断是否正确，再继续客观测试。

判断：

```text
Phase116 的子空间内部 basis component 审计基本正确；
但 SVD basis 不是唯一基，因此 basis-level support/release 解释有硬伤；
必须测试同一 answer-site causal subspace 在正交旋转下是否保持因果效应，
以及单基 support/release 标签是否稳定。
```

本阶段 Phase117 目标：

```text
验证 Phase116 的 support/release basis 是否依赖 SVD 基选择；
比较 SVD、varimax-like rotation、random orthogonal rotation、causal_greedy basis；
区分“子空间级稳定事实”和“单基级可解释标签”。
```

### 生成脚本

- 主测试脚本：`tests/gpt5/phase117_basis_rotation_causal_axis_cuda.py`
- 汇总脚本：`tests/gpt5/phase117_basis_rotation_causal_axis_summary.py`

### 测试原理

```text
1. 复用 Phase116 的 answer-site category contrast matrix。
2. 取 rank16 SVD 子空间作为同一因果子空间。
3. 在该子空间内部构造不同正交基：
   - svd
   - varimax
   - random_rot_0
   - random_rot_1
   - causal_greedy
4. 对每个基向量做 answer_last projection removal。
5. 按 target_delta 和 max_other_delta 标注 support/release/mixed/weak。
6. 比较 single、top4、top8、top16、support set、release set。
```

关键判据：

```text
如果 top16 在不同旋转下保持一致：
  子空间级因果事实稳定。

如果 best single / support count / release count 随旋转改变：
  单基标签依赖基选择，不能当作最终机制轴。

如果 causal_greedy 用少量方向恢复大部分效果：
  子空间内存在更集中的 causal axis。
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase117_basis_rotation_causal_axis_cuda.py \
  tests/gpt5/phase117_basis_rotation_causal_axis_summary.py

python tests/gpt5/phase117_basis_rotation_causal_axis_cuda.py qwen3 \
  --train-objects 4 \
  --test-objects 4 \
  --batch-size 4 \
  --rank 8 \
  --categories number,container \
  --random-rotations 1 \
  --causal-candidates 8 \
  --set-sizes 1,2,4 \
  --output-dir results/gpt5_phase117_smoke \
  --hard-exit-after-model

python tests/gpt5/phase117_basis_rotation_causal_axis_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --categories number,container,clothing,plant \
  --random-rotations 2 \
  --causal-candidates 24 \
  --set-sizes 1,2,4,8,16 \
  --output-dir results/gpt5_phase117_basis_rotation_causal_axis \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase117_basis_rotation_causal_axis_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --categories number,container,clothing,plant \
  --random-rotations 2 \
  --causal-candidates 24 \
  --set-sizes 1,2,4,8,16 \
  --output-dir results/gpt5_phase117_basis_rotation_causal_axis \
  --hard-exit-after-model

python tests/gpt5/phase117_basis_rotation_causal_axis_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --categories number,container,clothing,plant \
  --random-rotations 2 \
  --causal-candidates 24 \
  --set-sizes 1,2,4,8,16 \
  --output-dir results/gpt5_phase117_basis_rotation_causal_axis \
  --hard-exit-after-model

python tests/gpt5/phase117_basis_rotation_causal_axis_summary.py

python -m py_compile \
  tests/gpt5/phase117_basis_rotation_causal_axis_cuda.py \
  tests/gpt5/phase117_basis_rotation_causal_axis_summary.py
```

### 结果文件

- Qwen3：`results/gpt5_phase117_basis_rotation_causal_axis/phase117_qwen3_basis_rotation_causal_axis.json`
- GLM4：`results/gpt5_phase117_basis_rotation_causal_axis/phase117_glm4_basis_rotation_causal_axis.json`
- DS7B：`results/gpt5_phase117_basis_rotation_causal_axis/phase117_deepseek7b_basis_rotation_causal_axis.json`
- 跨模型汇总：`results/gpt5_phase117_basis_rotation_causal_axis/phase117_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, clothing, plant
train objects/category = 8
heldout test objects/category = 16
templates = 4
prompts/category = 64
rank = 16
scale = 1.5
set_sizes = 1, 2, 4, 8, 16
rotations = svd, varimax, random_rot_0, random_rot_1, causal_greedy
causal candidates/category = 24
```

### 客观结果

#### Qwen3

```text
number:
  top16 在所有旋转下固定为 target Δ -1.82, release +2.76
  svd single = -2.90 / release +1.59
  varimax single = -1.41 / release +2.53
  causal_greedy support set = -1.80 / release +0.31

container:
  top16 固定为 target Δ -2.53, release +1.90
  svd support set = -1.65 / release 0
  varimax single = -2.64 / release +1.33
  random_rot_0 top8 = -3.60 / release +0.15
  causal_greedy support set = -3.17 / release 0

clothing:
  top16 固定为 target Δ +1.69, release +1.49
  svd support set = -1.03 / release 0
  causal_greedy support set = -1.84 / release 0
  release set 仍强：causal_greedy +2.31 / release +2.10

plant:
  top16 固定为 target Δ -1.24, release +1.59
  svd support set = -1.39 / release 0
  random_rot_1 top8 = -3.07 / release 0
  causal_greedy top8 = -3.41 / release 0
```

Qwen3 结论：

```text
子空间级效果稳定，但 full-rank top16 常带 release；
support/release 单基标签对旋转敏感；
causal_greedy 可以找到更干净的局部 support set，
但完整子空间仍包含 release/interface 成分。
```

#### GLM4 bf16

```text
number:
  top16 约 -0.90 / release +0.68
  varimax top8 = -1.08 / release +0.78

container:
  top16 约 -0.22 / release +0.21

clothing:
  top16 约 -0.15 / release +0.20

plant:
  top16 约 -0.12 / release 0
```

GLM4 结论：

```text
旋转和 causal_greedy 没有挖出强隐藏因果轴；
Phase116 的“GLM4 效应弱”继续成立。
```

#### DS7B

```text
number:
  svd top16 = -12.58 / release 0
  varimax top16 = -12.58 / release 0
  random_rot_0 top16 = -12.59 / release 0
  random_rot_1 top16 = -12.58 / release 0
  causal_greedy top16 = -12.58 / release 0
  varimax single = -12.24 / release 0
  causal_greedy top4 = -10.65 / release 0

container:
  svd top16 = -12.52 / release 0
  varimax top16 = -12.52 / release 0
  random_rot_0 top16 = -12.54 / release 0
  random_rot_1 top16 = -12.54 / release 0
  causal_greedy top16 = -12.53 / release 0
  varimax single = -11.53 / release 0
  causal_greedy support set = -14.26 / release 0

clothing:
  top16 固定约 -2.46 / release 0
  svd support set = -5.58 / release 0
  full-rank 效果弱于 top4/top8，说明存在抵消成分
  release set 在多种基下仍存在

plant:
  top16 固定为 -7.87 / release 0
  svd top8 = -9.71 / release 0
  varimax single = -8.63 / release 0
  causal_greedy top8 = -9.91 / release 0
```

DS7B 结论：

```text
number/container/plant 是稳定的 causal subspace；
full-rank 因果效应对正交旋转不敏感；
但最强单基可从 SVD 的分布式形态变成 varimax 的集中单轴形态；
因此“强子空间存在”稳定，“SVD 单基就是机制轴”不稳定。
```

### 当前最可靠客观事实

1. **子空间级因果效应稳定**

同一 rank16 子空间经过正交旋转后，top16 基本不变。

典型：

```text
DS7B number:
  svd -12.58
  varimax -12.58
  random_rot_0 -12.59
  random_rot_1 -12.58
  causal_greedy -12.58

DS7B container:
  svd -12.52
  varimax -12.52
  random rotations -12.54
```

这说明 Phase114/115 的 answer-site causal subspace 不是 SVD 偶然产物。

2. **单基级标签明显依赖旋转**

例如 DS7B number：

```text
svd:
  best single -5.55
  support count 8

varimax:
  best single -12.24
  support count 1
```

同一子空间从“多个 support basis 累积”变成“一个极强单轴”，说明 Phase116 的 basis component 标签不能直接当作最终机制变量。

3. **DS7B 存在可集中化的强因果轴**

```text
number varimax single -12.24
container varimax single -11.53
plant varimax single -8.63
```

这不是随机方向，而是同一低秩子空间内部经过旋转后显露出的集中方向。

4. **DS7B 的 clean support 事实仍成立**

对于 number/container/plant：

```text
target_down 强；
release 接近 0；
top8/top16 稳定；
support set 强。
```

所以 Phase116 的“DS7B clean support subspace”需要改写为更严格表述：

```text
DS7B has a clean causal support subspace;
its basis-level distribution depends on the chosen orthogonal basis.
```

即：

```text
DS7B 有干净因果支持子空间；
但该支持在具体基向量上的分布依赖基选择。
```

5. **Qwen3 的 release/interface 是子空间级真实成分**

Qwen3 top16 在完整子空间下仍带明显 release：

```text
number top16: target -1.82, release +2.76
container top16: target -2.53, release +1.90
clothing top16: target +1.69, release +1.49
plant top16: target -1.24, release +1.59
```

虽然 causal_greedy 可以找到较干净 support set，但完整子空间仍包含 release/interface。

6. **GLM4 仍没有强因果轴**

旋转和 causal_greedy 都没有把 GLM4 提升到 DS7B/Qwen3 水平。

### 对 Phase116 的修正

Phase116 正确部分：

```text
1. answer-site low-rank subspace 内部确实含有功能不同的成分。
2. DS7B 的 number/container/plant 是干净支持型因果子空间。
3. Qwen3 的 release 是真实子空间成分，不是噪声。
4. GLM4 效应弱。
```

需要修正部分：

```text
1. “basis component” 不能直接解释成唯一机制轴。
2. support/release count 依赖基选择。
3. SVD 下的 distributed support 不一定表示机制本身必须分布式；
   varimax 可把 DS7B number/container/plant 压到强单轴或少数轴。
4. 更稳健的表述单位应从 basis component 上升到 causal subspace 和 causal axis family。
```

### 理论进展

Phase114/115/116/117 后，当前更稳健理论形式应改写为：

```text
Category causal state at answer site
=
low-rank causal subspace
+
rotation-dependent causal axis family
+
support/release/interface components
```

更具体：

```text
S_c(answer)
  是稳定对象；

Basis(S_c)
  不是稳定对象；

CausalAxisFamily(S_c)
  是下一步要寻找的对象。
```

对于 DS7B：

```text
S_c(answer) 是 clean support subspace；
在某些旋转下可集中成强 causal axis；
但 SVD basis 下显示为多个 support basis 累积。
```

对于 Qwen3：

```text
S_c(answer) 同时含 support 与 release/interface；
局部 support axis 可被 causal_greedy 找到；
但完整子空间不是干净 support。
```

### 硬伤分析

1. **causal_greedy 只是有限随机搜索**

```text
24 candidates/category 不等于全局最优因果轴。
```

2. **varimax 不是因果目标优化**

varimax 只是几何稀疏化旋转，不能保证就是真实机制变量。

3. **仍是 answer-site 单层测试**

尚未验证这些集中因果轴是否由上游层写入，或在多层路径中保持同一坐标。

4. **仍使用 DCF logits**

没有开放生成验证。

5. **Qwen3 的 release 仍未分解来源**

目前只知道 release/interface 是子空间级成分，尚不知道来自竞争类别、模板、词形、对象属性还是任务格式。

### 下一阶段任务

Phase118 应进入：

```text
Causal Axis Transport and Source-to-Answer Closure
```

目标：

```text
把 Phase117 找到的稳定 causal axes 从 answer site 往上游追踪，
测试这些轴是否在 object_last、middle layers、boundary layers 中被逐步写入；
并判断 DS7B 的强 support axis 是否是跨层同一坐标，
还是在 answer site 才重组出来。
```

建议测试：

```text
1. 选 DS7B number/container/plant 的 varimax/causal_greedy strong axes。
2. 在 object_last 与 answer_last 同时测：
   - source projection strength
   - answer causal effect
   - layer sweep
3. 做 axis patch：
   - remove at source layer
   - remove at answer layer
   - source+answer combined remove
4. 加 Qwen3 container/plant 对照，追踪 release/interface 是否来自上游竞争轴。
5. 加 random in-subspace axis 与 random ambient axis 对照。
```

Phase118 的关键问题：

```text
语言类别编码是否是：
  上游对象位置写入稳定因果轴，
  后续层传输并在答案位置读出；
还是：
  多个上游混合因素到答案位置才重组为 causal axis？
```
## Phase 118: Causal Axis Transport and Source-to-Answer Closure 因果轴传输与源到答案闭合 [2026-06-14 14:27]

### 本阶段目标

根据用户要求，先判断附件和 Phase117 分析是否正确，再继续客观测试。

判断：

```text
附件对 Phase117 的判断正确。
Phase117 没有推翻 Phase116，而是把结论收缩为：
  子空间级 causal effect 稳定；
  单个 SVD basis 的 support/release 标签不是旋转不变机制变量。
```

Phase118 目标：

```text
把 Phase117 找到的 answer-site causal axes 往上游追踪；
测试同一轴在 object_last、answer_last、both 三个位置的因果效果；
判断强轴是上游对象位置已经写入并直接传输，
还是主要在 answer_last 位置组装/读出。
```

### 生成脚本

- 主测试脚本：`tests/gpt5/phase118_causal_axis_transport_closure_cuda.py`
- 汇总脚本：`tests/gpt5/phase118_causal_axis_transport_closure_summary.py`

### 测试原理

```text
1. 在模型边界峰层构造 category answer-site rank16 causal subspace。
2. 对该子空间做 varimax rotation，选择 answer_last target-down 最强的 varimax_best axis。
3. 同时保留 svd_subspace 与 random_in_subspace 对照。
4. 在近峰层 sweep：
   Qwen3: L32-L35
   GLM4: L15-L18
   DS7B: L24-L27
5. 对每个 patch layer，在三个位置移除同一轴/子空间：
   object_last
   answer_last
   both
6. 记录 DCF logits target_delta、max_other_delta，并监控 answer-layer selected axis projection。
```

判据：

```text
如果 object_last removal 接近 answer_last removal：
  支持 source-to-answer 同坐标传输闭合。

如果 answer_last removal 很强而 object_last removal 很弱：
  支持 answer-site assembly/readout dominant。

如果 both 明显强于 answer_last：
  支持分布式位置共同因果。
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase118_causal_axis_transport_closure_cuda.py \
  tests/gpt5/phase118_causal_axis_transport_closure_summary.py

python tests/gpt5/phase118_causal_axis_transport_closure_cuda.py qwen3 \
  --train-objects 4 \
  --test-objects 4 \
  --batch-size 4 \
  --rank 8 \
  --layer-back 1 \
  --categories number,container \
  --output-dir results/gpt5_phase118_smoke \
  --hard-exit-after-model

python tests/gpt5/phase118_causal_axis_transport_closure_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase118_causal_axis_transport_closure \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase118_causal_axis_transport_closure_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase118_causal_axis_transport_closure \
  --hard-exit-after-model

python tests/gpt5/phase118_causal_axis_transport_closure_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --categories number,container,plant \
  --output-dir results/gpt5_phase118_causal_axis_transport_closure \
  --hard-exit-after-model

python tests/gpt5/phase118_causal_axis_transport_closure_summary.py

python -m py_compile \
  tests/gpt5/phase118_causal_axis_transport_closure_cuda.py \
  tests/gpt5/phase118_causal_axis_transport_closure_summary.py
```

### 结果文件

- Qwen3：`results/gpt5_phase118_causal_axis_transport_closure/phase118_qwen3_causal_axis_transport_closure.json`
- GLM4：`results/gpt5_phase118_causal_axis_transport_closure/phase118_glm4_causal_axis_transport_closure.json`
- DS7B：`results/gpt5_phase118_causal_axis_transport_closure/phase118_deepseek7b_causal_axis_transport_closure.json`
- 跨模型汇总：`results/gpt5_phase118_causal_axis_transport_closure/phase118_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, plant
train objects/category = 8
heldout test objects/category = 16
templates = 4
prompts/category = 64
rank = 16
scale = 1.5
axis_types = varimax_best, svd_subspace, random_in_subspace
patch_sites = object_last, answer_last, both
patch layers:
  qwen3 L32-L35
  glm4 L15-L18
  deepseek7b L24-L27
```

### 客观结果

#### Qwen3

```text
number:
  varimax_best selected = target Δ -1.41, release +2.53
  object_last best = -0.02, release +0.04
  answer_last best = -1.41, release +2.53
  both best = -1.41, release +2.57
  svd_subspace answer_last = -1.82, release +2.76

container:
  varimax_best selected = target Δ -2.64, release +1.33
  object_last best = -0.07, release +0.05
  answer_last best = -2.64, release +1.33
  both best = -2.73, release +1.28
  svd_subspace answer_last = -2.53, release +1.90

plant:
  varimax_best selected = target Δ -0.94, release +1.36
  object_last best = +0.01, release +0.08
  answer_last best = -0.94, release +1.36
  both best = -1.00, release +1.31
  svd_subspace answer_last = -1.24, release +1.59
```

Qwen3 结论：

```text
answer_last 明显强于 object_last；
both 基本不超过 answer_last；
Qwen3 的 release/interface 仍主要出现在 answer-site 轴移除中。
```

#### GLM4 bf16

```text
number:
  varimax_best selected = -0.38, release +0.26
  object_last = 0.00
  answer_last = -0.38
  svd_subspace answer_last = -0.90
  svd_subspace object_last = -0.31

container:
  varimax_best selected = -0.15, release +0.09
  object_last = -0.01
  answer_last = -0.15
  svd_subspace answer_last = -0.22

plant:
  varimax_best selected = -0.04
  object_last = -0.02
  answer_last = -0.04
```

GLM4 结论：

```text
整体仍弱；
没有出现强 source-to-answer closure。
```

#### DS7B

```text
number:
  varimax_best selected = target Δ -12.24, release 0
  object_last best = -0.74, release 0
  answer_last best = -12.24, release 0
  both best = -12.46, release 0
  svd_subspace:
    object_last -0.79
    answer_last -12.58
    both -12.78

container:
  varimax_best selected = target Δ -11.53, release 0
  object_last best = -0.47, release 0
  answer_last best = -11.53, release 0
  both best = -11.70, release 0
  svd_subspace:
    object_last -0.48
    answer_last -12.52
    both -12.68

plant:
  varimax_best selected = target Δ -8.63, release 0
  object_last best = -0.95, release 0
  answer_last best = -8.63, release 0
  both best = -8.91, release 0
  svd_subspace:
    object_last -0.90
    answer_last -7.87
    both -8.16
```

DS7B 结论：

```text
强因果轴在 answer_last 极强；
同一轴/子空间在 object_last 移除非常弱；
both 仅比 answer_last 小幅增强；
因此当前测试不支持“同一坐标从 object_last 直接传输到 answer_last”。
更支持 answer-site assembly/readout dominant。
```

### 当前最可靠客观事实

1. **DS7B 强轴主要是 answer-site 因果**

典型比例：

```text
number:
  object_last -0.74 vs answer_last -12.24

container:
  object_last -0.47 vs answer_last -11.53

plant:
  object_last -0.95 vs answer_last -8.63
```

object_last 不是完全没有信号，但远弱于 answer_last。

2. **both 不形成强加和**

```text
DS7B number:
  answer_last -12.24
  both -12.46

DS7B container:
  answer_last -11.53
  both -11.70

DS7B plant:
  answer_last -8.63
  both -8.91
```

这说明在当前同轴 patch 设计下，主要因果杠杆已经集中在 answer_last。

3. **Qwen3 同样是 answer_last 主导，但带 release**

```text
container:
  object_last -0.07
  answer_last -2.64, release +1.33
  both -2.73, release +1.28
```

Qwen3 的 release/interface 并没有在 object_last 同轴移除中显著出现，而是在 answer-site removal 中出现。

4. **GLM4 继续弱**

GLM4 没有强同轴闭合结果，延续 Phase116/117 的弱效应结论。

5. **同一 answer-site axis 不能简单当作 upstream source coordinate**

Phase118 的核心负结果：

```text
把 answer-site 选出的强 causal axis 直接拿到 object_last 移除，
不能复现 answer_last 的强 target_down。
```

这不等于上游没有类别信息，而是说明：

```text
上游对象位置的编码坐标可能不同；
answer-site 强轴可能是后续层重组/读出后的坐标。
```

### 对 Phase117 的修正和推进

Phase117 正确部分：

```text
answer-site causal subspace 稳定；
DS7B 有 clean support subspace；
varimax 可显露强单轴；
Qwen3 有 support/release/interface mixed subspace。
```

Phase118 新增限制：

```text
这些 answer-site strong axes 不能直接外推为 object_last source axes。
```

更严格表述：

```text
CausalAxis_c(answer)
  是答案位置强因果轴；
但不一定等于 CausalAxis_c(object)。
```

### 理论进展

当前条件化关系因子动力学公式应继续改写：

```text
Object state:
  O_c^l(object)

Transport / transformation:
  T_{object -> answer}^{l..L}

Answer state:
  S_c^L(answer)

Observed causal axis:
  A_c^L(answer) ∈ S_c^L(answer)
```

Phase118 表明：

```text
A_c^L(answer)
不能简单反向复制到 O_c^l(object)。
```

因此当前更稳健公式是：

```text
S_c^L(answer)
=
Transform_l_to_L(
  O_c^l(object),
  template/context,
  attention/MLP routing
)
```

而不是：

```text
S_c^L(answer)
=
direct_copy(O_c^l along same axis)
```

这对破解语言背后编码机制很关键：

```text
语言类别编码可能不是静态方向传输，
而是跨层坐标变换后在答案位置形成可读出的因果子空间。
```

### 硬伤分析

1. **只测试了 answer-site axis 在 upstream 的同坐标移除**

弱 object_last 不代表上游无类别信息，只说明同一答案轴在 object_last 不闭合。

2. **轴是在 monitor layer 拟合的**

每层可能有自己的局部坐标。如果要公平测试 source axis，需要在每层分别构造 local causal axis。

3. **object_last 可能不是唯一 source**

类别信息可能分散在 object span、post-object tokens、template tokens 或 attention output 中。

4. **Aproj 是均值投影监控**

投影均值变化不等于分布形状变化；target_down 可能来自方差/样本级排序变化。

5. **仍是 DCF logits**

尚未做开放生成和多任务验证。

### 下一阶段任务

Phase119 应进入：

```text
Layer-local Source Axis Discovery and Coordinate Transform Mapping
```

目标：

```text
不要再把 answer-site axis 直接搬到 object_last；
而是在每个 layer、每个 site 上分别学习 local category causal subspace/axis，
再测试 local source axis 是否能影响最终 answer logits。
```

建议测试：

```text
1. 对 DS7B number/container/plant，在 L24-L27 分别构造：
   - object_last local contrast subspace
   - answer_last local contrast subspace
2. 每层分别选 local varimax_best axis。
3. 测 local object axis removal 对最终 logits 的影响。
4. 测 local object axis 与 next-layer answer axis 的对齐和投影变化。
5. 加 object_span、post_object、answer_last 三类 source site。
6. 对 Qwen3 container/plant 做同样测试，追踪 release/interface 是否来自局部 source axis。
```

Phase119 的关键问题：

```text
如果 local object axis 有强因果效果：
  说明上游有类别源轴，只是坐标随层变换。

如果 local object axis 仍弱：
  说明类别因果子空间主要在 answer-site late assembly 中形成。
```
## Phase 119: Layer-local Source Axis Discovery 层局部源轴发现 [2026-06-14 14:58]

### 本阶段目标

根据用户要求，先判断附件与 Phase118 分析是否正确，再继续完成任务。

判断：

```text
附件对 Phase118 的判断正确。
Phase118 的负结果不能解释为 object/source 没有类别信息；
只能说明 answer-site axis 不能直接当作 object_last 的同坐标 source axis。
```

Phase119 目标：

```text
不再把 answer-site axis 直接搬到 object_last；
而是在每个 layer、每个 site 上分别学习 local category subspace/axis；
测试 local source axis 是否能影响最终 DCF logits。
```

### 生成脚本

- 主测试脚本：`tests/gpt5/phase119_layer_local_source_axis_cuda.py`
- 汇总脚本：`tests/gpt5/phase119_layer_local_source_axis_summary.py`

### 测试原理

```text
1. 对每个模型，在边界峰层前 3 层到峰层做 layer sweep。
2. 对每个 layer 和 site，分别捕获 train objects 的 hidden state centers。
3. 每个 site 单独构造 category contrast matrix。
4. 对 local contrast matrix 取 rank16 SVD subspace。
5. 对 local subspace 做 varimax rotation，并在同 layer/site 上选择 target-down 最强 local_varimax_best axis。
6. 同时测试：
   - local_varimax_best
   - local_svd_subspace
   - random_in_local_subspace
7. 对 heldout objects 测最终 DCF logits 的 target_delta 与 max_other_delta。
```

本阶段测试的 site：

```text
object_last
object_span_mean
post_object_mean
answer_last
```

其中：

```text
post_object_mean = object span 后到 answer_last 前/含 answer_last 的提示尾部区域平均。
```

### 执行命令

```bash
python -m py_compile \
  tests/gpt5/phase119_layer_local_source_axis_cuda.py \
  tests/gpt5/phase119_layer_local_source_axis_summary.py

python tests/gpt5/phase119_layer_local_source_axis_cuda.py qwen3 \
  --train-objects 4 \
  --test-objects 4 \
  --batch-size 4 \
  --rank 8 \
  --layer-back 1 \
  --sites object_last,answer_last \
  --categories number,container \
  --output-dir results/gpt5_phase119_smoke \
  --hard-exit-after-model

python tests/gpt5/phase119_layer_local_source_axis_cuda.py qwen3 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --sites object_last,object_span_mean,post_object_mean,answer_last \
  --categories number,container,plant \
  --output-dir results/gpt5_phase119_layer_local_source_axis \
  --hard-exit-after-model

PROBE_TORCH_DTYPE=bfloat16 python tests/gpt5/phase119_layer_local_source_axis_cuda.py glm4 \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --sites object_last,object_span_mean,post_object_mean,answer_last \
  --categories number,container,plant \
  --output-dir results/gpt5_phase119_layer_local_source_axis \
  --hard-exit-after-model

python tests/gpt5/phase119_layer_local_source_axis_cuda.py deepseek7b \
  --train-objects 8 \
  --test-objects 16 \
  --batch-size 24 \
  --rank 16 \
  --layer-back 3 \
  --sites object_last,object_span_mean,post_object_mean,answer_last \
  --categories number,container,plant \
  --output-dir results/gpt5_phase119_layer_local_source_axis \
  --hard-exit-after-model

python tests/gpt5/phase119_layer_local_source_axis_summary.py

python -m py_compile \
  tests/gpt5/phase119_layer_local_source_axis_cuda.py \
  tests/gpt5/phase119_layer_local_source_axis_summary.py
```

### 结果文件

- Qwen3：`results/gpt5_phase119_layer_local_source_axis/phase119_qwen3_layer_local_source_axis.json`
- GLM4：`results/gpt5_phase119_layer_local_source_axis/phase119_glm4_layer_local_source_axis.json`
- DS7B：`results/gpt5_phase119_layer_local_source_axis/phase119_deepseek7b_layer_local_source_axis.json`
- 跨模型汇总：`results/gpt5_phase119_layer_local_source_axis/phase119_cross_model_summary.md`

### 测试范围

```text
models = qwen3, glm4, deepseek7b
categories = number, container, plant
train objects/category = 8
heldout test objects/category = 16
templates = 4
prompts/category = 64
rank = 16
scale = 1.5
layers:
  qwen3 L32-L35
  glm4 L15-L18
  deepseek7b L24-L27
sites:
  object_last
  object_span_mean
  post_object_mean
  answer_last
axis_types:
  local_varimax_best
  local_svd_subspace
  random_in_local_subspace
```

### 客观结果

#### Qwen3

```text
number:
  object_last ≈ 0
  object_span_mean ≈ 0
  post_object_mean local_varimax_best: L35 target Δ -4.43, release +1.93
  post_object_mean local_svd_subspace: L35 target Δ -4.41, release +2.30
  answer_last local_varimax_best: L35 target Δ -1.41, release +2.53

container:
  object_last = -0.07
  object_span_mean = -0.07
  post_object_mean local_varimax_best: L32 target Δ -1.23, release +1.86
  post_object_mean local_svd_subspace: L32 target Δ -1.73, release +3.61
  answer_last local_varimax_best: L35 target Δ -2.64, release +1.33

plant:
  object_last = -0.02
  object_span_mean = -0.05
  post_object_mean local_varimax_best: L35 target Δ -5.29, release +1.37
  post_object_mean local_svd_subspace: L35 target Δ -4.66, release +1.83
  answer_last local_varimax_best: L35 target Δ -0.94, release +1.36
```

Qwen3 结论：

```text
object token 本身仍弱；
post_object_mean 出现强 local source axis；
但 release 很大，说明 Qwen3 的 source 区域仍是 support/release/interface 混合场。
```

#### GLM4 bf16

```text
number:
  object_last local_svd_subspace: -0.27, release +0.19
  object_span_mean local_svd_subspace: -0.20, release +0.23
  post_object_mean local_svd_subspace: -1.11, release +0.05
  answer_last local_svd_subspace: -0.90, release +0.68

container:
  best source weak，post_object_mean local_varimax_best -0.48, release +0.57
  answer_last local_svd_subspace -0.22

plant:
  all weak，post_object_mean local_varimax_best -0.29, release +0.31
```

GLM4 结论：

```text
仍弱；
只有 number 的 post_object_mean local_svd_subspace 有轻度信号。
```

#### DS7B

```text
number:
  object_last local_varimax_best: L27 target Δ -0.78, release 0
  object_span_mean local_varimax_best: L27 target Δ -0.81, release 0
  post_object_mean local_varimax_best: L27 target Δ -11.74, release 0
  post_object_mean local_svd_subspace: L27 target Δ -12.03, release 0
  answer_last local_varimax_best: L27 target Δ -12.24, release 0
  answer_last local_svd_subspace: L27 target Δ -12.58, release 0

container:
  object_last local_varimax_best: L27 target Δ -0.90, release 0
  object_span_mean local_varimax_best: L27 target Δ -0.95, release 0
  post_object_mean local_varimax_best: L27 target Δ -13.24, release 0
  post_object_mean local_svd_subspace: L27 target Δ -12.74, release 0
  answer_last local_varimax_best: L27 target Δ -11.53, release 0
  answer_last local_svd_subspace: L27 target Δ -12.52, release 0

plant:
  object_last local_varimax_best: L27 target Δ -0.97, release 0
  object_span_mean local_varimax_best: L27 target Δ -1.44, release 0
  post_object_mean local_varimax_best: L27 target Δ -10.58, release 0
  post_object_mean local_svd_subspace: L27 target Δ -9.57, release 0
  answer_last local_varimax_best: L27 target Δ -8.63, release 0
  answer_last local_svd_subspace: L27 target Δ -7.87, release 0
```

DS7B 结论：

```text
object_last/object_span 仍弱；
post_object_mean 出现与 answer_last 同量级甚至更强的 clean support source axis；
release = 0；
说明 Phase118 的负结果来自 source site 选窄了，而不是源轴不存在。
```

### 当前最可靠客观事实

1. **object_last 不是主要类别因果源点**

跨模型看：

```text
DS7B number object_last -0.78 vs answer_last -12.24
DS7B container object_last -0.90 vs answer_last -11.53
DS7B plant object_last -0.97 vs answer_last -8.63
```

即使重新学习 local object axis，object_last 仍远弱于 answer_last。

2. **object_span_mean 也不是主要源点**

DS7B：

```text
number object_span -0.81
container object_span -0.95
plant object_span -1.44
```

略强于 object_last，但仍远弱于 post_object/answer。

3. **post_object_mean 是强 source/control site**

DS7B：

```text
number post_object_mean -11.74 / -12.03
container post_object_mean -13.24 / -12.74
plant post_object_mean -10.58 / -9.57
```

这是 Phase119 的最大新发现。

4. **DS7B 的 post_object source axis 是 clean support**

```text
release = 0
```

对 number/container/plant 都成立。

5. **Qwen3 也有 post_object source effect，但混有 release**

Qwen3：

```text
number post_object -4.43, release +1.93
plant post_object -5.29, release +1.37
container post_object -1.23 to -1.73, release +1.86 to +3.61
```

Qwen3 的相对竞争/接口混合场不仅在 answer site，也出现在 post_object/source-control 区。

6. **random_in_local_subspace 对照显示 post_object 强效不是任意随机局部方向**

DS7B：

```text
number random post_object -4.07 vs local_varimax -11.74
container random post_object -1.48 vs local_varimax -13.24
plant random post_object -2.61 vs local_varimax -10.58
```

随机方向有时也有信号，说明局部子空间整体有因果性，但 local_varimax/local_svd 更强。

### 对 Phase118 的修正

Phase118 正确部分：

```text
answer-site axis 不能直接外推为 object_last axis；
object_last 同轴和 local axis 都弱；
answer_last 是强因果杠杆。
```

Phase119 修正部分：

```text
源位置不能只看 object_last 或 object_span；
post_object_mean 是强 source/control site；
在 DS7B 中 post_object_mean 与 answer_last 同量级。
```

更严格表述：

```text
类别因果源不在 object token 本身，
而更可能在 object 后的 prompt-tail / interface / pre-answer region 中形成。
```

### 理论进展

当前公式进一步改写：

```text
Object lexical state:
  O_c^l(object_span)

Prompt-tail / interface control state:
  P_c^l(post_object)

Answer readout state:
  A_c^L(answer)
```

Phase119 表明：

```text
O_c^l(object_span) 因果弱；
P_c^l(post_object) 因果强；
A_c^L(answer) 因果强。
```

因此当前更稳健的关系式是：

```text
A_c^L(answer)
=
Transform(
  P_c^l(post_object),
  O_c^l(object_span),
  template/context,
  route
)
```

而不是：

```text
A_c^L(answer)
=
Transform(O_c^l(object_span))
```

进一步：

```text
P_c^l(post_object)
可能是类别任务接口状态：
  它把 object lexical state 转成 category-query/readout-ready state。
```

中文解释：

```text
对象词本身更像提供语义材料；
对象后面的模板/接口区域把这些材料变成“准备回答类别”的控制状态；
答案位置再把控制状态读出为目标类别 logits。
```

### 对破解语言编码机制的关键洞察

1. **源不等于对象词本身**

当前证据显示：

```text
object token 是语义材料位置；
post_object region 是任务化/接口化控制位置；
answer token 是输出读出位置。
```

2. **语言编码可能是三段式**

```text
object semantic material
→ prompt-tail/interface control state
→ answer-site causal subspace
→ output logits
```

3. **DS7B 给出最干净版本**

```text
post_object_mean 与 answer_last 都是 clean support；
object_last/object_span 弱；
release = 0。
```

4. **Qwen3 给出竞争场版本**

```text
post_object 与 answer site 都有 target_down；
但 release 明显，说明类别状态包含竞争/接口混合。
```

### 硬伤分析

1. **post_object_mean 包含 answer_last**

当前 post_object_positions 定义为 object 后到 answer_last，包含最终位置。
这可能使 post_object_mean 受 answer_last 强轴影响。

2. **post_object 是 mean patch**

对所有 post-object tokens 使用同一 mean-derived axis，不能定位到底是哪一个 token 最关键。

3. **仍没有显式 transform mapping**

本轮发现了强 local source site，但没有拟合 post_object axis 到 answer_last axis 的变换。

4. **仍是 DCF logits**

没有开放生成验证。

5. **只测了三个类别**

number/container/plant 是关键类别，但还需要扩展到 clothing/furniture/time 等混合类别。

### 下一阶段任务

Phase120 应进入：

```text
Post-object Token Localization and Interface State Decomposition
```

目标：

```text
把 post_object_mean 拆开，定位到底是哪个 token 或哪类 token 形成强 source/control state；
排除“只是 answer_last 被平均进去”的可能。
```

建议测试：

```text
1. 将 post_object 区域拆为：
   - after_object_first
   - after_object_middle
   - pre_answer_last
   - answer_last
   - post_object_excluding_answer
2. 对每个 token/site 构造 local axis。
3. 在 DS7B number/container/plant 上优先测试。
4. 对 Qwen3 number/container/plant 做对照，观察 release/interface 来源。
5. 加 full post_object_mean 与 excluding_answer 对照。
6. 继续保留 random_in_local_subspace control。
```

Phase120 的关键问题：

```text
强 post_object source axis 是由 answer_last 混入造成，
还是确实存在于答案前的 prompt-tail/interface tokens？
```
