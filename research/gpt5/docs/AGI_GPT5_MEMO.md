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
