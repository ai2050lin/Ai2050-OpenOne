# -*- coding: utf-8 -*-
"""AI2050 分布式平台 —— 内置种子模板（TM-01..05）

来源：design/distributed_platform_plan_v1.md S2（模板包 = contracts/ 预注册纪律升级）。
语料为受控最小对网格（多条件分解，见 2026-10-05 方案：语言/风格/逻辑/位置条件），
每次只动一个条件；corpus 变更必须升 version（design_sha 随内容变化，结果不可跨版混比）。

design_sha 由 distributed_service 在注册时对 corpus+runner 规范化 JSON 计算，
预注册语义：模板一旦发布，语料与口径冻结；Worker 上传结果的 design_sha 必须一致，否则拒收。

v2（2026-10-07 协议）：新增两张因子网格模板——
    TM-04 句式×知识（topic×frame）：句式从噪声翻转为免费因子操纵；
         特征 ID=反词嵌入指纹（类内全 token 行投影结构等价=真特征；单 token 峰值=伪特征）；
         推理/组合特征预期落在交互项（Cmp 候选竞争的领地）。
    TM-05 实体×位置等变（entity×position）：固定权重+位置滑动+产出稳定 ⇒
         内容寻址解引用；预注册预测：位置 η² 早层高、随深度衰减、末层≈0；
         实体 η² 各层均高且 ≫ 位置 η²——若违反则登记为新发现（可证伪）。
元法则：变化=因子，不变=指纹；η² 定编码，干预定因果（升格须过迁移干预门）。
"""
from pathlib import Path

RUNNER_SOURCE = Path(__file__).with_name('dist_runner_tm.py').read_text(encoding='utf-8')

METRIC_PIN = 'metric_dict v4'
MODEL_PIN = 'Qwen/Qwen3-4B（revision 由节点环境 pin，AI2050_RUNNER_MODEL 指定）'

TM01_CORPUS = {
    'tm_id': 'TM-01',
    'name': '语言最小对 · 中/英（多条件分解网格）',
    'dim': '语言 lang ∈ {zh, en}；风格/逻辑/内容受控',
    'description': '同一命题的中英最小对，目标 token 为命题主词；'
                   '分离语言条件效应（多条件分解管线 step1-3：采集 + 坐标级 η² + 方向 cos）。',
    'items': [
        {'id': 'lang-fruit-zh', 'cond': {'lang': 'zh'}, 'text': '苹果富含维生素。', 'target': '苹果'},
        {'id': 'lang-fruit-en', 'cond': {'lang': 'en'}, 'text': 'Apples are rich in vitamins.', 'target': 'Apples'},
        {'id': 'lang-dog-zh', 'cond': {'lang': 'zh'}, 'text': '狗是常见的哺乳动物。', 'target': '狗'},
        {'id': 'lang-dog-en', 'cond': {'lang': 'en'}, 'text': 'Dogs are common mammals.', 'target': 'Dogs'},
        {'id': 'lang-iron-zh', 'cond': {'lang': 'zh'}, 'text': '铁是一种金属元素。', 'target': '铁'},
        {'id': 'lang-iron-en', 'cond': {'lang': 'en'}, 'text': 'Iron is a metallic element.', 'target': 'Iron'},
        {'id': 'lang-whale-zh', 'cond': {'lang': 'zh'}, 'text': '鲸鱼生活在海洋里。', 'target': '鲸鱼'},
        {'id': 'lang-whale-en', 'cond': {'lang': 'en'}, 'text': 'Whales live in the ocean.', 'target': 'Whales'},
        {'id': 'lang-photon-zh', 'cond': {'lang': 'zh'}, 'text': '光子是光的基本粒子。', 'target': '光子'},
        {'id': 'lang-photon-en', 'cond': {'lang': 'en'}, 'text': 'Photons are the basic particles of light.', 'target': 'Photons'},
        {'id': 'lang-enzyme-zh', 'cond': {'lang': 'zh'}, 'text': '酶能加速化学反应。', 'target': '酶'},
        {'id': 'lang-enzyme-en', 'cond': {'lang': 'en'}, 'text': 'Enzymes can speed up chemical reactions.', 'target': 'Enzymes'},
        {'id': 'lang-glacier-zh', 'cond': {'lang': 'zh'}, 'text': '冰川由常年不化的积雪形成。', 'target': '冰川'},
        {'id': 'lang-glacier-en', 'cond': {'lang': 'en'}, 'text': 'Glaciers form from perennial snow.', 'target': 'Glaciers'},
        {'id': 'lang-volcano-zh', 'cond': {'lang': 'zh'}, 'text': '火山喷发出炽热的岩浆。', 'target': '火山'},
        {'id': 'lang-volcano-en', 'cond': {'lang': 'en'}, 'text': 'Volcanoes erupt hot magma.', 'target': 'Volcanoes'},
        {'id': 'lang-neuron-zh', 'cond': {'lang': 'zh'}, 'text': '神经元通过电信号传递信息。', 'target': '神经元'},
        {'id': 'lang-neuron-en', 'cond': {'lang': 'en'}, 'text': 'Neurons transmit information via electrical signals.', 'target': 'Neurons'},
        {'id': 'lang-antibio-zh', 'cond': {'lang': 'zh'}, 'text': '抗生素能抑制细菌生长。', 'target': '抗生素'},
        {'id': 'lang-antibio-en', 'cond': {'lang': 'en'}, 'text': 'Antibiotics inhibit bacterial growth.', 'target': 'Antibiotics'},
        {'id': 'lang-satellite-zh', 'cond': {'lang': 'zh'}, 'text': '卫星沿轨道绕地球运行。', 'target': '卫星'},
        {'id': 'lang-satellite-en', 'cond': {'lang': 'en'}, 'text': 'Satellites orbit the Earth along trajectories.', 'target': 'Satellites'},
        {'id': 'lang-chloro-zh', 'cond': {'lang': 'zh'}, 'text': '叶绿素参与植物的光合作用。', 'target': '叶绿素'},
        {'id': 'lang-chloro-en', 'cond': {'lang': 'en'}, 'text': 'Chlorophyll drives photosynthesis in plants.', 'target': 'Chlorophyll'},
    ],
}

TM02_CORPUS = {
    'tm_id': 'TM-02',
    'name': '风格最小对 · 正式/口语（同内容同语言）',
    'dim': '风格 style ∈ {formal, casual}；语言固定 zh',
    'description': '同一命题的正式/口语最小对，目标 token 为共享内容词；分离风格条件效应。',
    'items': [
        {'id': 'sty-medic-f', 'cond': {'style': 'formal'}, 'text': '该药物疗效显著。', 'target': '药物'},
        {'id': 'sty-medic-c', 'cond': {'style': 'casual'}, 'text': '这药效果是真不错。', 'target': '药'},
        {'id': 'sty-data-f', 'cond': {'style': 'formal'}, 'text': '数据显示上升趋势明显。', 'target': '数据'},
        {'id': 'sty-data-c', 'cond': {'style': 'casual'}, 'text': '数据看起来涨了不少。', 'target': '数据'},
        {'id': 'sty-exp-f', 'cond': {'style': 'formal'}, 'text': '实验结果支持该假设。', 'target': '实验'},
        {'id': 'sty-exp-c', 'cond': {'style': 'casual'}, 'text': '实验做下来基本靠谱。', 'target': '实验'},
        {'id': 'sty-model-f', 'cond': {'style': 'formal'}, 'text': '模型性能显著提升。', 'target': '模型'},
        {'id': 'sty-model-c', 'cond': {'style': 'casual'}, 'text': '模型这下厉害多了。', 'target': '模型'},
        {'id': 'sty-method-f', 'cond': {'style': 'formal'}, 'text': '该方法效率优于基线。', 'target': '方法'},
        {'id': 'sty-method-c', 'cond': {'style': 'casual'}, 'text': '这法子比老办法强。', 'target': '法子'},
        {'id': 'sty-concl-f', 'cond': {'style': 'formal'}, 'text': '结论具有重要意义。', 'target': '结论'},
        {'id': 'sty-concl-c', 'cond': {'style': 'casual'}, 'text': '结论挺有意思的。', 'target': '结论'},
        {'id': 'sty-error-f', 'cond': {'style': 'formal'}, 'text': '误差控制在合理范围。', 'target': '误差'},
        {'id': 'sty-error-c', 'cond': {'style': 'casual'}, 'text': '误差没超太多。', 'target': '误差'},
        {'id': 'sty-chart-f', 'cond': {'style': 'formal'}, 'text': '图表清晰呈现趋势。', 'target': '图表'},
        {'id': 'sty-chart-c', 'cond': {'style': 'casual'}, 'text': '图表一眼就能看懂。', 'target': '图表'},
    ],
}

TM03_CORPUS = {
    'tm_id': 'TM-03',
    'name': '逻辑连接词 · 因果/转折/递进/平叙',
    'dim': '逻辑 logic ∈ {because, however, moreover, none}；语言固定 zh',
    'description': '同话题四句框架，仅逻辑连接词不同，目标 token 为连接词本身；分离逻辑条件效应。',
    'items': [
        {'id': 'lg-fruit-bec', 'cond': {'logic': 'because'}, 'text': '今年水果丰收，因为雨水充足。', 'target': '因为'},
        {'id': 'lg-fruit-how', 'cond': {'logic': 'however'}, 'text': '今年水果丰收，但是价格偏低。', 'target': '但是'},
        {'id': 'lg-fruit-mor', 'cond': {'logic': 'moreover'}, 'text': '今年水果丰收，而且品质更佳。', 'target': '而且'},
        {'id': 'lg-fruit-non', 'cond': {'logic': 'none'}, 'text': '今年水果丰收，行情总体平稳。', 'target': '水果'},
        {'id': 'lg-exp-bec', 'cond': {'logic': 'because'}, 'text': '实验进度加快，因为流程简化了。', 'target': '因为'},
        {'id': 'lg-exp-how', 'cond': {'logic': 'however'}, 'text': '实验进度加快，但是准确率下降。', 'target': '但是'},
        {'id': 'lg-exp-mor', 'cond': {'logic': 'moreover'}, 'text': '实验进度加快，而且成本降低。', 'target': '而且'},
        {'id': 'lg-exp-non', 'cond': {'logic': 'none'}, 'text': '实验进度加快，团队士气高涨。', 'target': '实验'},
        {'id': 'lg-model-bec', 'cond': {'logic': 'because'}, 'text': '模型收敛提速，因为学习率合适。', 'target': '因为'},
        {'id': 'lg-model-how', 'cond': {'logic': 'however'}, 'text': '模型收敛提速，但是内存占用高。', 'target': '但是'},
        {'id': 'lg-model-mor', 'cond': {'logic': 'moreover'}, 'text': '模型收敛提速，而且精度提升。', 'target': '而且'},
        {'id': 'lg-model-non', 'cond': {'logic': 'none'}, 'text': '模型收敛提速，训练进入正轨。', 'target': '模型'},
        {'id': 'lg-weather-bec', 'cond': {'logic': 'because'}, 'text': '气温骤然下降，因为寒潮南下。', 'target': '因为'},
        {'id': 'lg-weather-how', 'cond': {'logic': 'however'}, 'text': '气温骤然下降，但是降水稀少。', 'target': '但是'},
        {'id': 'lg-weather-mor', 'cond': {'logic': 'moreover'}, 'text': '气温骤然下降，而且风力增强。', 'target': '而且'},
        {'id': 'lg-weather-non', 'cond': {'logic': 'none'}, 'text': '气温骤然下降，冬意渐浓。', 'target': '气温'},
    ],
}

TM04_CORPUS = {
    'tm_id': 'TM-04',
    'analysis': 'factorial',
    'factors': ['topic', 'frame'],
    'layers': ['last'],
    'name': '句式×知识 因子网格（多条件分解·内容轴）',
    'dim': '知识 topic ∈ {fruit, metal, whale, glacier} × 句式 frame ∈ {decl, interrog, concess, condit}',
    'description': '同一知识条目嵌入四种句式（陈述/疑问/让步/条件），balanced 无重复网格；'
                   '逐维 two-way 分解出 η²(知识)/η²(句式)/η²(交互)——句法特征跟句子走、'
                   '知识特征跟内容走、推理组合特征只活在交互项（Cmp 候选竞争领地）；'
                   'fingerprint 为反词嵌入指纹参照集：类内全 token 行投影结构等价（spread 小）=真特征，'
                   '单 token 峰值直指=伪特征（logit-lens 回声）。',
    'items': [
        {'id': 'sf-fruit-decl', 'cond': {'topic': 'fruit', 'frame': 'decl'}, 'text': '苹果富含维生素。', 'target': '苹果'},
        {'id': 'sf-fruit-inte', 'cond': {'topic': 'fruit', 'frame': 'interrog'}, 'text': '苹果富含维生素吗？', 'target': '苹果'},
        {'id': 'sf-fruit-conc', 'cond': {'topic': 'fruit', 'frame': 'concess'}, 'text': '虽然苹果很常见，它富含维生素。', 'target': '苹果'},
        {'id': 'sf-fruit-cond', 'cond': {'topic': 'fruit', 'frame': 'condit'}, 'text': '如果说起苹果，人们会想到维生素。', 'target': '苹果'},
        {'id': 'sf-metal-decl', 'cond': {'topic': 'metal', 'frame': 'decl'}, 'text': '铁是常见的金属元素。', 'target': '铁'},
        {'id': 'sf-metal-inte', 'cond': {'topic': 'metal', 'frame': 'interrog'}, 'text': '铁是常见的金属元素吗？', 'target': '铁'},
        {'id': 'sf-metal-conc', 'cond': {'topic': 'metal', 'frame': 'concess'}, 'text': '虽然铁很普通，它是重要的金属元素。', 'target': '铁'},
        {'id': 'sf-metal-cond', 'cond': {'topic': 'metal', 'frame': 'condit'}, 'text': '如果说起铁，人们会想到金属元素。', 'target': '铁'},
        {'id': 'sf-whale-decl', 'cond': {'topic': 'whale', 'frame': 'decl'}, 'text': '鲸鱼生活在海洋里。', 'target': '鲸鱼'},
        {'id': 'sf-whale-inte', 'cond': {'topic': 'whale', 'frame': 'interrog'}, 'text': '鲸鱼生活在海洋里吗？', 'target': '鲸鱼'},
        {'id': 'sf-whale-conc', 'cond': {'topic': 'whale', 'frame': 'concess'}, 'text': '虽然鲸鱼体型巨大，它生活在海洋里。', 'target': '鲸鱼'},
        {'id': 'sf-whale-cond', 'cond': {'topic': 'whale', 'frame': 'condit'}, 'text': '如果说起鲸鱼，人们会想到海洋。', 'target': '鲸鱼'},
        {'id': 'sf-glacier-decl', 'cond': {'topic': 'glacier', 'frame': 'decl'}, 'text': '冰川由常年不化的积雪形成。', 'target': '冰川'},
        {'id': 'sf-glacier-inte', 'cond': {'topic': 'glacier', 'frame': 'interrog'}, 'text': '冰川由常年不化的积雪形成吗？', 'target': '冰川'},
        {'id': 'sf-glacier-conc', 'cond': {'topic': 'glacier', 'frame': 'concess'}, 'text': '虽然冰川移动缓慢，它由常年不化的积雪形成。', 'target': '冰川'},
        {'id': 'sf-glacier-cond', 'cond': {'topic': 'glacier', 'frame': 'condit'}, 'text': '如果说起冰川，人们会想到积雪。', 'target': '冰川'},
    ],
    'fingerprint': {
        'fruit':   {'within': ['苹果', '水果'], 'out': ['铁', '鲸鱼', '冰川']},
        'metal':   {'within': ['铁', '金属'],   'out': ['苹果', '鲸鱼', '冰川']},
        'whale':   {'within': ['鲸鱼', '海洋'], 'out': ['苹果', '铁', '冰川']},
        'glacier': {'within': ['冰川', '积雪'], 'out': ['苹果', '铁', '鲸鱼']},
    },
}

TM05_CORPUS = {
    'tm_id': 'TM-05',
    'analysis': 'factorial',
    'factors': ['entity', 'position'],
    'layers': ['first', 'mid', 'last'],
    'name': '实体×位置 等变网格（多条件分解·位置轴）',
    'dim': '实体 entity ∈ {ming, hong} × 绝对位置 position ∈ {early, mid, late}；层位 first/mid/last',
    'description': '同一实体在不同句式框架中出现在句首/句中/句末——权重固定而位置滑动、产出稳定，'
                   '检验「内容寻址解引用」机制：读位按内容（key）回取实体，不按绝对地址。'
                   '预注册预测（可证伪）：η²(实体) ≫ η²(位置) 且各层均高；'
                   'η²(位置) 早层（first）显著、随深度衰减、末层（last）≈0——'
                   '若末层位置 η² 仍显著，说明位置信息泄漏进语义通道，登记为新发现；'
                   '干预门（移位/换名实验，效应跟内容走）为升格 mechanism_evidence 的前置条件。',
    'items': [
        {'id': 'ep-ming-early', 'cond': {'entity': 'ming', 'position': 'early'}, 'text': '小明在图书馆看书。', 'target': '小明'},
        {'id': 'ep-ming-mid', 'cond': {'entity': 'ming', 'position': 'mid'}, 'text': '昨天小明在图书馆安静看书。', 'target': '小明'},
        {'id': 'ep-ming-late', 'cond': {'entity': 'ming', 'position': 'late'}, 'text': '昨天在图书馆安静看书的正是小明。', 'target': '小明'},
        {'id': 'ep-hong-early', 'cond': {'entity': 'hong', 'position': 'early'}, 'text': '小红在实验室写报告。', 'target': '小红'},
        {'id': 'ep-hong-mid', 'cond': {'entity': 'hong', 'position': 'mid'}, 'text': '傍晚小红在实验室写报告。', 'target': '小红'},
        {'id': 'ep-hong-late', 'cond': {'entity': 'hong', 'position': 'late'}, 'text': '傍晚在实验室写报告的正是小红。', 'target': '小红'},
    ],
}

TM06_CORPUS = {
    'tm_id': 'TM-06',
    'version': 4,  # v1 attentions 索引 bug；v2 SDPA 空 attentions；v3 want_attn 未定义——均冻结，升版重注册
    'analysis': 'heads',
    'cond_factor': 'prefix',
    'layers': ['last'],
    'topk': 32,
    'name': '前缀扰动 × 目标词 head 稳定性（苹果用例）',
    'dim': '前缀 prefix ∈ {none, space, filler} × 内容 content ∈ {apple, book}；analysis=heads',
    'description': '用户用例（2026-10-07）：「我喜欢吃红色的苹果。」前面加空格或无关信息，'
                   '看关注「苹果」的 attention head 是否变化。预注册统计量：'
                   '①attn_target=每(层,头)对目标词位置的平均注意力份额；'
                   '②条件内排名 top-32 head 集合 → 跨条件 Jaccard（head 集合指纹，尺度无关）——'
                   'same-content 跨前缀 Jaccard 高 = head 级内容寻址；same-prefix 跨内容（apple~book）'
                   '作对照，应显著更低；'
                   '③attn_pull=target 对无关前缀区段的注意力份额（item.prefix_chars 预注册前缀字符数）'
                   '——无关信息吸引度；'
                   '④last 层残差流条件方向 cos / η²——语义读出稳定性。'
                   '预测：无关前缀不改变目标词的 head 指纹与读出方向（产出稳定）；'
                   '若 Jaccard 大幅下降或 cos 崩塌 = 前缀敏感性，登记为新发现（可证伪）。',
    'items': [
        {'id': 'pp-apple-none', 'cond': {'content': 'apple', 'prefix': 'none'}, 'text': '我喜欢吃红色的苹果。', 'target': '苹果', 'prefix_chars': 0},
        {'id': 'pp-apple-space', 'cond': {'content': 'apple', 'prefix': 'space'}, 'text': '　　我喜欢吃红色的苹果。', 'target': '苹果', 'prefix_chars': 2},
        {'id': 'pp-apple-filler', 'cond': {'content': 'apple', 'prefix': 'filler'}, 'text': '昨天下午在拥挤的地铁上，我忽然想起，我喜欢吃红色的苹果。', 'target': '苹果', 'prefix_chars': 18},
        {'id': 'pp-book-none', 'cond': {'content': 'book', 'prefix': 'none'}, 'text': '我喜欢读厚厚的书。', 'target': '书', 'prefix_chars': 0},
        {'id': 'pp-book-space', 'cond': {'content': 'book', 'prefix': 'space'}, 'text': '　　我喜欢读厚厚的书。', 'target': '书', 'prefix_chars': 2},
        {'id': 'pp-book-filler', 'cond': {'content': 'book', 'prefix': 'filler'}, 'text': '昨天下午在拥挤的地铁上，我忽然想起，我喜欢读厚厚的书。', 'target': '书', 'prefix_chars': 18},
    ],
}

NEWS_FALLBACK = [
    {'date': '2025-05', 'title': 'Circuit Tracing（Anthropy 团队）— 归因图方法', 'p': '用替换模型与跨层编码边把激活图编译为可读电路，行业主流量化归因路线。', 'src': 'anthropic.com', 'tag': '方法', 'url': 'https://www.anthropic.com/research/circuit-tracing'},
    {'date': '2024-09', 'title': 'Gemma Scope 开放 SAE 全套', 'p': '开源 Gemma-2 全层宽 SAE 权重与在线演示，公开特征词典参考系。', 'src': 'deepmind.google', 'tag': '资源', 'url': 'https://deepmind.google/technologies/gemma-scope/'},
    {'date': '2024-08', 'title': 'InterPLM：蛋白质语言模型的 SAE 对齐', 'p': 'ESM-2 SAE 特征与 UniProt 注释对齐，跨域同构旁证。', 'src': 'biorxiv', 'tag': '论文', 'url': 'https://www.biorxiv.org/'},
    {'date': '2024-03', 'title': 'Neuronpedia 开放特征百科', 'p': '聚合多模型 SAE 特征与激活示例，对象路由范式来源。', 'src': 'neuronpedia.org', 'tag': '资源', 'url': 'https://neuronpedia.org/'},
    {'date': '2023-10', 'title': 'SAEBench 发布', 'p': 'SAE 系列评测基准：稀疏性-可解释性-下游影响多维刻度。', 'src': 'github.com', 'tag': '基准', 'url': 'https://github.com/'},
    {'date': '2022-09', 'title': 'Toy Models of Superposition', 'p': '叠加与相变：特征数超维度的表征压缩机制，本项目核心理论参照。', 'src': 'anthropic.com', 'tag': '理论', 'url': 'https://transformer-circuits.pub/2022/toy_model/index.html'},
    {'date': '2020-03', 'title': 'OpenAI Microscope', 'p': '神经元与电路可视化图鉴，对象→多视图范式的先行者。', 'src': 'openai.com', 'tag': '资源', 'url': 'https://microscope.openai.com/'},
]


def seed_templates() -> list:
    """返回待注册模板列表（service 启动时对齐播种：已有 tm_id 跳过，缺的补注册）。"""
    out = []
    for tm_id, corpus, dim_note in (
        ('TM-01', TM01_CORPUS, '语言条件（多条件分解·语言轴）'),
        ('TM-02', TM02_CORPUS, '风格条件（多条件分解·风格轴）'),
        ('TM-03', TM03_CORPUS, '逻辑条件（多条件分解·逻辑轴）'),
        ('TM-04', TM04_CORPUS, '句式×知识因子网格（多条件分解·内容轴，反词嵌入指纹）'),
        ('TM-05', TM05_CORPUS, '实体×位置等变网格（多条件分解·位置轴，内容寻址解引用）'),
        ('TM-06', TM06_CORPUS, '前缀扰动×目标词 head 稳定性（head 级指纹·苹果用例）'),
    ):
        out.append({
            'tm_id': tm_id,
            'version': corpus.get('version', 1),
            'name': corpus['name'],
            'dim': corpus['dim'],
            'description': corpus['description'],
            'metric_pin': METRIC_PIN,
            'model_pin': MODEL_PIN,
            'corpus': corpus,
            'runner': RUNNER_SOURCE,
        })
    return out
