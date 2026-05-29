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
