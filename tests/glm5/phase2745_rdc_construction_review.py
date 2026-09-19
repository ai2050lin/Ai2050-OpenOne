"""Audit the three current attachments and register one continuous programme.

Descriptions are not experimental outcomes. Prior artifacts remain immutable.
"""
from rdc_construction_common import *


def main():
    out = BASE / 'review'
    if (out / 'integrated_plan.json').exists():
        return
    attachments = [
        Path('C:/Users/Admin/.codex/attachments/7bf676b1-4b92-46b4-9458-0c5059036c1f/pasted-text.txt'),
        Path('C:/Users/Admin/.codex/attachments/63e11001-5dc7-4af2-bef2-f5a3f9557f30/pasted-text.txt'),
        Path('C:/Users/Admin/.codex/attachments/876bb8f8-2cb3-4281-8239-f28a64ba8960/pasted-text.txt')]
    evidence = ['analysis/phase2741.json', 'analysis/phase2742.json', 'followup/result.json',
        'identifiability/analysis/result.json', 'identifiability/analysis/pair_change_control.json',
        'formation/result.json', 'theory_snapshot.json', 'delivery_manifest.json']
    for p in evidence:
        assert (OLD / p).exists()
    # Explicit judgments distinguish architecture identities from new evidence.
    judgments = [
        ('R01', [1, 2], '百万查询图谱', 'retain_scoped', '10000前缀、2777文档、100固定查询；端点全坐标不等于百万查询全部token全部层永久保存。固定后缀并非100个正交语义操作。', evidence[-1]),
        ('R02', [1, 2], '来源反对称量完整求和抵消', 'retain_identity_reject_overreach', '只否定该反对称分账总和独立驱动输出的说法；不能推出MLP无法保留关系方向或激活不含方向信息。', evidence[1]),
        ('R03', [1, 2], '整体MSE与全词表KL', 'retain_scoped', 'MSE改善不保证KL改善；不等于所有线性映射不可能，unembedding本身是线性且softmax非线性。', evidence[0]),
        ('R04', [1, 2], '跨表达映射全失败', 'correct', '八个方向/划分条件有一个mixed-holdout有限阳性；预定test未通过，不能说全部无效，也不能称语言代码同构。', evidence[0]),
        ('R05', [1, 2, 3], '真实与乱序标签训练', 'retain_scoped', '四条32步受限续训不是预训练史；乱序原始NLL改善更多，但累计BF16位移未匹配，不能说所有微调只是伪学习。', evidence[5]),
        ('R06', [1, 2, 3], '概率校准', 'retain_scoped', 'T大于1使概率变平缓；附件二称温度降低会平缓需纠正。校准残差不是纯语义，alpha零只限制实际测试的先验。', evidence[3]),
        ('R07', [1], '弱词频先验', 'retain_algebraic_limit', '576次计数对151936词项各加1，平滑主导；需更大训练侧计数和总浓度而非每token固定1的新对照。', evidence[3]),
        ('R08', [1, 2, 3], '实际查询H12补充信息', 'retain_scoped', '更晚信息条件有明显收益，支持研究查询构造；不是与旧前缀输入相同，也未证明唯一缺口。第一块Q在attention之前尚未读取前缀，不能说每层Q必然都受此前缀改变。', evidence[2]),
        ('R09', [1, 2], '严格词袋控制', 'retain_scoped', '160对完整实际token多重集一致；排除纯词袋函数，但仍有位置、局部序列、模板与题义混杂。', evidence[3]),
        ('R10', [1, 2], '属性绑定与知识链', 'retain_behavior_reject_mechanism_name', '属性绑定能力强、此知识链规则任务弱是实际行为差异；浅深任务标签不是内部层定位，滑齿/流形旋转/测地线断裂均未测量。', evidence[3]),
        ('R11', [1, 2], '关系变化预测', 'retain_negative_result', '旧五规则整体拟合有增益，但未见查询的配对变化未稳定胜零；差值是统计目标，不是激活搬运。', evidence[4]),
        ('R12', [2], '仅静态HiddenState永远无用', 'reject_universal_claim', '参数、状态及动态响应互补；已测末状态与有限未来查询存在统计联系。有限探针不自动是完整预测状态。', evidence[0]),
        ('R13', [2], '主特征向量就是纯语义基', 'reject_identification', '谱统计可作非裁剪数值诊断，但特征向量没有自动语义身份；不采用主成分截断定义齿轮。', evidence[4]),
        ('R14', [2], '晚期logit偏置修复逻辑', 'test_only_without_gold', '同类别共同偏置保持类别内部相对概率；可测试时机及后续历史，不预写修复成功，不使用正确答案选择偏置。', evidence[1]),
        ('R15', [2], '剥离全部格式得到纯语义且参数空间正交', 'reject_guarantee_test_finite_constraints', '有限校准/格式约束不覆盖全部非语义因素；局部投影不保证有限训练格式不变，已有非正交证据保留。', evidence[6]),
        ('R16', [2], '跨模态完美等距与消除所有幻觉', 'reject_guarantee_test_scoped_transfer', '文字、代码和数学文本均为token序列，不因此形成跨模态证据；只检验明确域内映射、配对置乱、读出与自身历史收益。', evidence[0]),
        ('R17', [1, 2, 3], '需要/不需要新数学', 'open_question', '两种断言均未被证明；先记录新增对象、运算、规律及相对基线预测增量。RDC名称保留。', evidence[6]),
        ('R18', [3], '固定参数不同输入有不同输出', 'retain_with_limit', '一般函数也如此；语言泛化与复用才是需要解释的组织规律，不能推出最优或严格模块。', evidence[6]),
        ('R19', [1, 3], 'MLP全参数Gamma因式张量', 'retain_architecture_identity', '保留gate/up/down所有向量可精确查询任意乘积元素；实数恒等式与BF16残差分开，不是新语言定律。', evidence[6]),
        ('R20', [1, 3], 'QK和OV参数骨架', 'retain_native_implementation_required', '包含实际QK归一化、bias、GQA、RoPE及残差边界；简化矩阵式不能冒充所有模型原生实现。', evidence[6]),
        ('R21', [1, 3], '条件ANOVA交互', 'prospective_test', '平衡网格中逐原生坐标双中心化；交互是采样分布下的统计项，不自动等于语义。', evidence[4]),
        ('R22', [1, 3], '一般与原生参数约束查询预测', 'prospective_test', '只用预测时可见信息；全部真实目标和未来token不能作修补项，分别评价整体、配对、词表和自身历史。', evidence[2]),
        ('R23', [3], '跨层读写兼容性和真实链式依赖', 'prospective_test', '固定向量内积只是骨架；全链式JVP/VJP须含attention变化、RMS分母、双门和残差，局部导数不等于全局语义。', evidence[6]),
        ('R24', [3], '训练形成四环', 'prospective_test', '连接语言更新相容性、真实参数变化、条件运行变化与未见输出；原checkpoint不足以唯一恢复训练史。', evidence[5])]
    items = [{'id': i, 'attachments': a, 'claim': c, 'decision': d, 'correction_or_scope': s,
              'evidence_path': str(OLD / e), 'evidence_sha256': sha(OLD / e)} for i, a, c, d, s, e in judgments]
    references = [
        {'url': 'https://arxiv.org/abs/2002.05202', 'supports': 'GLU/SwiGLU门控乘法是已有架构，不是新发现的语义律。'},
        {'url': 'https://arxiv.org/abs/2104.09864', 'supports': 'RoPE在attention匹配中引入相对位置；各本地实现仍单独核对。'},
        {'url': 'https://transformer-circuits.pub/2025/attribution-graphs/methods.html', 'supports': '原文Limitations: Missing Attention Circuits明确冻结attention模式不解释QK模式如何形成。'},
        {'url': 'https://papers.neurips.cc/paper/1983-predictive-representations-of-state.pdf', 'supports': '有限测试响应成为预测状态需对所有测试构成充分统计量；本项目未证明。'}]
    immutable(out / 'claim_audit.json', {'timestamp': stamp(), 'source': snapshot(__file__),
        'attachments': [{'number': i+1, **snapshot(p)} for i, p in enumerate(attachments)],
        'judgments': items, 'external_primary_references_checked': references,
        'scope': 'Current three attachments read in full and checked against preserved local scientific records and selected primary references; historical experiments are not all rerun by this review.'})
    phases = [
        {'phase': 2745, 'question': '语境如何构造查询，现有参数训练收益如何区分方向与幅值？',
         'tasks': ['三非量化模型320表达×100查询原生全坐标构造场及来源attention',
                   '全部坐标条件ANOVA、原始/RMS视图、五族双语成功失败对照',
                   '实际H1与可用前缀候选的全跨坐标算子、同输入逐坐标/置乱/零基线',
                   '预测HE经原生Q投影与postnorm经完整词表编译',
                   '23个原生/旧训练/匹配BF16位移/坐标置乱/反向条件的自然NLL与自身历史',
                   '只读图谱、完整证据和MEMO连续追加'],
         'state_at_registration': 'frozen_and_partly_captured; no formal extraction outcome inspected'},
        {'phase': 2746, 'question': '固定参数关系怎样在自然语言条件下被选择并跨层接续？',
         'tasks': ['全层全单元参数骨架目录、Gamma及QK/OV精确查询，不重复存模型',
                   '统一自然语料与既有五族网格的gate/up/写回/来源运行谱',
                   '复用10000自然前缀百万端点作全坐标交互分析；逐层扩展先核对增量，再流式覆盖，记录各层实际覆盖',
                   '一般来源条件查询更新与原生参数库约束版本，同输入同材料比较',
                   '完整中间网络JVP/VJP和有限变化核对，固定读写兼容性只作对照',
                   '冻结规则后自然文档、表达/实体与组合确认及自身历史连续误差',
                   '参数索引—条件交互—跨层效应—输出四者同ID关联'],
         'state_at_registration': 'planned_not_executed'},
        {'phase': 2747, 'question': '候选条件规律能否通过真实参数学习形成并推广？',
         'tasks': ['真实/置乱监督与明确有限格式约束方向的真实训练检查点，不预设纯语义',
                   '累计FP32/BF16位移匹配、全参数梯度相容性和方向/半径控制',
                   '更充分训练侧词频计数、温度/总浓度先验及保留语义残差的谨慎解释',
                   '预测参数变化如何改变查询、gate/up、跨层场及未见输出',
                   '原有文字/代码匹配材料的映射/置乱/无gold读出对照，不预设同构',
                   '较长自身历史、内容/格式/停止与历史分叉；三个原生模型同能力条件复查',
                   '全理论拼图及公式审计后，同目标且有信息价值则自动下一完整阶段'],
         'state_at_registration': 'planned_not_executed'}]
    immutable(out / 'integrated_plan.json', {'timestamp': stamp(), 'source': snapshot(__file__),
        'goal': '破解有限参数如何形成可复用的语言关系计算；未知科学结论不预写为必然成功。',
        'phases': phases, 'resource_policy': read(BASE / 'protocol.json')['resource_policy'],
        'ordering': 'Observe, extract, test generalization, then discriminate mechanisms. Three-phase programme is one authorized task, not three confirmation requests.',
        'coverage_policy': 'Each task keeps explicit planned/implemented/executed/verified states and actual sample coverage. A failed prediction is an informative result, not a completed universal mechanism.',
        'review_timing': 'This integration follows initial2745 protocol and partial capture. It does not retroactively change its frozen primary tests or claim all new plans were previously frozen.',
        'claim_audit_sha256': sha(out / 'claim_audit.json'),
        'no_new_markdown_except_append_to_existing_memo': True,
        'model_concurrency': 1, 'goal_complete': False})
    print('CONSTRUCTION_ATTACHMENT_REVIEW', len(items), 'judgments; integrated phases2745-2747', flush=True)


if __name__ == '__main__':
    main()
