/**
 * AI研发 默认配置数据
 */

// 默认AI主模型配置
export const DEFAULT_MASTER_MODEL = {
  name: 'GLM-4-Flash',
  model_type: 'master',
  api_type: 'zhipu',
  api_base: 'https://open.bigmodel.cn/api/paas/v4',
  api_key: '',
  model_id: 'glm-4-flash',
  analysis_prompt: `你是一位深度神经网络逆向工程专家。请分析以下研究数据，找出关键模式、异常和突破点。

当前研究轮次: {round}
累积研究发现:
{findings}

最新测试结果:
{test_results}

请从以下角度分析:
1. 数据中是否出现了新的模式或规律？
2. 之前的假设是否被验证或推翻？
3. 当前最大的瓶颈是什么？
4. 有哪些异常值需要特别关注？

请给出简洁、有洞察力的分析。`,
  planning_prompt: `你是AI研究主管，负责制定下一步研究方案。

当前研究轮次: {round}
研究上下文:
{research_context}

各分析师的发现:
{analyst_reports}

已验证的结论:
{verified_conclusions}

当前瓶颈:
{bottlenecks}

请制定下一步研究方案，包括:
1. 需要验证的核心假设
2. 具体实验设计
3. 预期结果和判据
4. 优先级排序

方案要具体、可执行，能直接转化为Python测试代码。`,
  code_gen_prompt: `你是一位精通Transformer模型逆向工程的Python开发者。请根据研究方案生成测试代码。

研究方案:
{plan}

可用的本地模型和工具:
- 模型: Qwen3-4B (4B参数, bf16), GLM4-9B (8bit量化), DeepSeek-R1-7B (8bit量化)
- 框架: PyTorch, TransformerLens, HuggingFace Transformers
- GPU: RTX 5080 16GB VRAM

已有工具函数 (tests/glm5/model_utils.py):
- load_model_bf16(model_name) - 加载模型
- get_layers(model) - 获取transformer层
- get_W_U(model, model_name) - 获取unembedding矩阵
- get_model_info(model, model_name) - 获取模型信息
- release_model(model) - 释放模型
- MODEL_CONFIGS - 模型路径配置

代码要求:
1. 输出文件保存到 results/glm5/ 目录
2. 临时文件保存到 tests/glm5_temp/ 目录
3. 使用 if __name__ == "__main__" 入口
4. 包含计时和内存监控
5. 测试完一个模型再测试下一个，避免GPU OOM
6. 8bit量化加载大模型(GLM4/DS7B)

请只输出Python代码，不要解释。`,
  summary_prompt: `你是研究总结专家。请总结本轮研究的关键发现。

研究轮次: {round}
执行结果:
{execution_results}

关键数据:
{key_data}

请总结:
1. 本轮最重要的发现（1-3条）
2. 被验证的结论
3. 被推翻的假设
4. 新出现的疑问
5. 对下一轮研究的建议

用简洁的学术语言总结。`,
};

// 默认分析师模型配置
export const DEFAULT_ANALYST_MODELS = [
  {
    name: 'DeepSeek-Chat',
    model_type: 'analyst',
    api_type: 'deepseek',
    api_base: 'https://api.deepseek.com/v1',
    api_key: '',
    model_id: 'deepseek-chat',
    analysis_prompt: `你是一位擅长数学分析的AI研究助手。请从数学结构的角度分析以下实验数据。

当前研究轮次: {round}
研究发现:
{findings}

测试结果:
{test_results}

请重点关注:
1. 数据中是否隐含某种代数或几何结构？
2. 不同模型的结果是否有共同的数学模式？
3. 是否可以用更简洁的数学语言描述观察到的现象？
4. 结果是否与已知理论（如流形假设、正交分解）一致？

请给出数学直觉和初步形式化。`,
  },
  {
    name: 'Qwen-Max',
    model_type: 'analyst',
    api_type: 'dashscope',
    api_base: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key: '',
    model_id: 'qwen-max',
    analysis_prompt: `你是一位擅长统计分析和模式识别的AI研究助手。请从实证角度分析以下实验数据。

当前研究轮次: {round}
研究发现:
{findings}

测试结果:
{test_results}

请重点关注:
1. 数据中是否有统计显著的规律？
2. 异常值是否暗示了某种被忽略的机制？
3. 跨模型一致性是否足够强以支持通用结论？
4. 样本量是否足够？是否需要更大规模的验证？

请给出谨慎、有统计支撑的分析。`,
  },
];

// 本地测试模型配置
export const LOCAL_TEST_MODELS = [
  {
    id: 'qwen3',
    name: 'Qwen3-4B',
    params: '4B',
    load_mode: 'bf16',
    vram: '~8GB',
  },
  {
    id: 'glm4',
    name: 'GLM4-9B-Chat',
    params: '9B',
    load_mode: '8bit',
    vram: '~11GB',
  },
  {
    id: 'deepseek7b',
    name: 'DeepSeek-R1-7B',
    params: '7B',
    load_mode: '8bit',
    vram: '~9GB',
  },
];

// 研究阶段定义
export const RESEARCH_PHASES = [
  { id: 'analyze', label: '分析', icon: '🔍', color: '#4488ff' },
  { id: 'plan', label: '规划', icon: '📋', color: '#ffaa00' },
  { id: 'generate', label: '生成', icon: '⚡', color: '#00ff88' },
  { id: 'execute', label: '执行', icon: '🚀', color: '#ff4444' },
  { id: 'summarize', label: '总结', icon: '📝', color: '#aa44ff' },
];

// API类型配置
export const API_TYPES = [
  { id: 'openai', label: 'OpenAI 兼容', prefix: 'https://api.openai.com/v1' },
  { id: 'zhipu', label: '智谱 AI', prefix: 'https://open.bigmodel.cn/api/paas/v4' },
  { id: 'deepseek', label: 'DeepSeek', prefix: 'https://api.deepseek.com/v1' },
  { id: 'dashscope', label: '阿里 DashScope', prefix: 'https://dashscope.aliyuncs.com/compatible-mode/v1' },
  { id: 'siliconflow', label: 'SiliconFlow', prefix: 'https://api.siliconflow.cn/v1' },
];
