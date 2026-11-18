"""Append the completed study and bounded conclusion without rewriting prior records."""
import hashlib
import json
import re
import time
from pathlib import Path
import branch_reuse14 as B


def f(x):
    return '—' if x is None else f'{x:.6g}'


def ci(x):
    return '['+', '.join(f(v) for v in x)+']'


def table(headers, rows):
    return '\n'+' | '.join(headers)+'\n'+' | '.join(['---']*len(headers))+'\n'+'\n'.join(' | '.join(str(x) for x in row) for row in rows)+'\n'


def append(path, make_text):
    before=path.read_bytes()
    text=before.decode('utf8')
    assert B.RUN not in text, 'This run already has a memo entry; do not duplicate it.'
    phases=[int(n) for n in re.findall(r'^## Phase (\d+):',text,re.M)]
    phase=max(phases)+1
    stamp=time.strftime('%Y-%m-%d %H:%M')
    addition=make_text(phase,stamp).encode('utf8')
    assert addition.startswith(b'\n\n## Phase ')
    assert path.read_bytes()==before, 'Concurrent memo change; retry from current tail.'
    with path.open('ab') as out:
        out.write(addition)
        out.flush()
    after=path.read_bytes()
    assert after[:len(before)]==before and after[len(before):]==addition
    return dict(path=str(path),phase=phase,timestamp_local=stamp,line=before.count(b'\n')+3,
                prefix_bytes=len(before),prefix_sha256=hashlib.sha256(before).hexdigest(),
                append_bytes=len(addition),append_sha256=hashlib.sha256(addition).hexdigest(),
                entire_sha256=hashlib.sha256(after).hexdigest(),prefix_preserved=True)


def run():
    B.configure()
    audit=B.read(B.OUT/'audit.json');assert audit['passed']
    assert B.read(B.OUT/'browser-check.json')['passed']
    assert B.read(B.OUT/'qualified-pipeline-state.json')['status']=='completed'
    response=B.read(B.OUT/'response-and-graph-report.json')
    output=B.read(B.OUT/'output-fidelity.json')
    intervention=B.read(B.OUT/'intervention-report.json')
    controls=B.read(B.OUT/'control-comparisons.json')
    verdict=B.read(B.OUT/'conclusion.json')
    cleanup=B.read(B.OUT/'cleanup.json')
    native=B.read(B.OUT/'native-manifest.json')
    joint=B.read(B.OUT/'joint-state-summary.json')
    groups=B.read(B.OUT/'group-response-profiles.json')
    progress=[]
    for line in (B.OUT/'progress.jsonl').read_text(encoding='utf8').splitlines():
        try:progress.append(json.loads(line))
        except json.JSONDecodeError:pass
    peak=lambda key:max((r.get(key,0) or 0 for r in progress),default=0)
    strict={(r['kind'],r['branch'],r['arm']):r for r in controls['strict_accuracies']}

    def main_text(phase,stamp):
        parts=[f'\n\n## Phase {phase}: Qwen3-14B 跨分支共同响应与三层组合的联合验收结论 [{stamp}]\n',
        f'''**任务、范围与结论。** 用户明确将“泛化递归”限定为“同一单元或单元组跨语义分支复用共同结构，并通过层间组合形成更复杂特征”。本轮运行 `{B.RUN}`，只在本机原始BF16 Qwen3-14B上串行执行，快照 `1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce`。结论分层：发现阶段预选的B19组7/8、B39根组6/8在数值和标签留出分支都有正向A配对响应，说明共同响应可以跨这些材料复现；历史B19/U64的统一V/A主解释和两条三段稀疏预测链均未通过冻结联合门，不能将它们认定为已确认的泛化递归机制。按权重关系预选的同一B36单元组对两分支答案间隔有可测作用，部分输出预测收益也保留。这里否定的是具体候选、测量位置和预测算法的联合命题，不是证明14B不存在任何跨分支复用。材料没有系统改变组合深度或提供复杂特征真值，因此“形成更复杂特征”仍未得到直接确认。不能把有限实验的阴性结果写成全模型不存在机制的确定结论。

**从旧结果到新冻结。** 上轮全部留出已经开封，本轮复用旧原场比较直接A、G/V乘积和完整残差状态，只作为开发诊断。补足残差输入的改善更明显，提示旧失败可能包含状态不足。随后独立建立新材料，冻结样本、组、两个链结构、96个拟合文件、指标、阈值、科学源码与执行准入，再读取确认结果。旧字段不被覆盖，新确认不用于重选组或重拟合。模型、tokenizer、依赖、源码及冻结产物哈希见trial-plan、analysis-plan、frozen-selection、execution-freeze与qualified-execution。

**材料、行为校准与排除。** 对照“对未列输入仍有效的规则”和“仅列有限输入输出的记录”，交叉技术/普通标题，每家族四版本。有限记录也有关系，不能当作关系语义绝对缺席。v1问题设计失败后只用发现试跑修订；v2序列分支在扩大试跑失败，同一题轮换正确选项位置后A/B各32/32、C只有4/32，说明选择题接口存在明显位置依赖，不是模型不会重排的充分证据。v3保留数值、坐标、标签三分支，家族切分和材料身份保持。标签完全不参与拟合，两个拟合分支为数值/坐标；确认使用新词池和新表达，但数值可重叠，不声称训练未见。

v3总计1,536条、384家族：试跑192条、发现384条、确认960条。修正argmax同分偏好后的严格行为为191/192、376/384、928/960。原始分数与旧计分保留；新门只要求正确分数严格高于对照，阈值仍为每格至少90%。坐标规则两种标题68/80、70/80未通过，停止其内部确认与干预。数值318/320，标签312/320，所有分格合格；剩余640条、160家族，仅作两分支部分验收。独立适配器在内部确认分析前冻结排除范围，原三分支联合门保持不可通过，原六检验校正不降低。序列和坐标的缺失不能被平均准确率或合格子集掩盖。

**采集与计算对象。** 逐块CUDA运行、batch=4、padding长度256，真实子进程持GPU互斥租约。两读出为问题/选项出现前的正文共同末token和作答提示末token。保存hs[0]及40块输出的全部5,120坐标，40块×17,408单元的G/V/A，两个历史单元另存每token轨迹；不是保存所有token的所有单元全场。BF16以uint16位模式无损压缩。41个残差边界与最终归一化边界分开。采集核对实际tokenizer、选项单token边界、G/V乘积及HF参考数值。全量984个NPZ、1,536条分词和14个模型/tokenizer文件已审计。覆盖量不是机制破解进度。

对块l、位置t、单元j，z是MLP实际归一化输入，g/u是gate/up权重行，d是down列。以下是架构计算定义：
''',
        r'''
$$G_{ltj}=g_{lj}^{\mathsf T}z_{lt},\qquad V_{ltj}=u_{lj}^{\mathsf T}z_{lt},\qquad A_{ltj}=\operatorname{SiLU}(G_{ltj})V_{ltj},\qquad \Delta h_{ltj}=A_{ltj}d_{lj}.$$

**历史单元与预选组的留出结果。** 正文末位主检验为B19/U64的V/A规则减记录配对差，每分支80家族，要求正均值、dz≥0.5、双侧p<0.05/6、两个标题均正，且“前缀平均embedding＋位置/标题＋已知规则标签”的MSE≤同输入去除规则标签基线的0.9倍。标签是实验者信息，不是已提取的无标注机制。家族四变体先配对聚合，置信区间用原六项校正。
''']
        parts.append(table(['分支','量','配对差','dz','校正区间','p','组合/嵌入MSE','主门'],[
            [r['branch'],r['field'],f(r['contrast']['mean']),f(r['contrast']['dz']),ci(r['contrast']['ci']),f(r['contrast']['p_two_sided']),f(r['combined_over_embedding']),'通过' if r['passed'] else '未通过']
            for r in response['responses'] if r['primary']]))
        parts.append('''
数值A的校正区间含零，不能称为显著负向；补充效应量区间排除了预设+0.5下限，但不能排除任意小效应。标签V/A均有强正向响应，加入已知标签没有进一步胜过嵌入基线；嵌入本身可能已含相同信息，因此基线增益失败不能单独说明没有语义。B26/U13099为次级候选，在数值A正向、标签接近零，没有普遍复用证据。

预选B19组为[64,16877,4731,8515,12952,8000,5976,15647]；除64外七单元在数值/标签两个分支都复现正向A，两标题同向。预选B39根为[15961,4545,13298,7693,4684,2182,1845,4087]，除4545/2182外六单元同样复现。它们是预先选索引的次级描述，按组内多项区间核对，不补救主门。相同角色措辞和任务格式仍是替代解释；A正向也不意味着down列写入方向一致。17张原生单元卡加1张B36组功能卡、极端正负配对原文、完整结果均保留，正文与当时尚未出现的后续题目分开显示。

**三层组合预测。** 从B39八根向B38/B37/B36各选32父单元，分别采用最大联合响应相关和真实权重读写系数先验。后者省略注意力和完整归一化导数，并不是精确因果边。每层使用发现家族三折选择岭系数1/10/100/1000；两个读出各拟合。比较均值、512维词/双词哈希、前缀平均embedding、三组幅值匹配随机32单元，以及完整同位置残差状态。完整残差的信息预算更大，单独标明。
''')
        parts.append(r'''
$$\widehat Y_{l+1}=F_l(A_{l,P_l}),\quad Y=[G,V,A];\qquad \widehat A^{\rm direct}=\widehat Y_A,\quad\widehat A^{\rm composed}=\operatorname{SiLU}(\widehat Y_G)\widehat Y_V.$$

F是拟合预测器，不是新架构定律。自由递推只用真实B36选定A起步，后面只传预测值。局部边每格MSE须≤均值、词汇、三随机组平均损失的0.9倍；根自由递推须≤均值0.5倍、词汇与embedding各0.9倍。以下为自由根G/V/A误差：
''')
        parts.append(table(['路线','方式','读出','分支','MSE/均值','MSE/词汇','MSE/嵌入','门'],[
            [r['method'],r['mode'],r['readout'],r['branch'],f(r['ratios']['mean']['ratio']),f(r['ratios']['lexical']['ratio']),f(r['ratios']['embedding']['ratio']),'通过' if r['passed'] else '未通过'] for r in response['rollouts']]))
        parts.append(f"\n局部边{sum(r['passed'] for r in response['edges'])}/{len(response['edges'])}格通过，自由递推{sum(r['passed'] for r in response['rollouts'])}/{len(response['rollouts'])}格通过；没有整条链满足全部条件。每层拟合的是不同F，不能把迭代计算本身叫作同一规则递归，也没有证据显示层数等于语义复杂度。\n")
        parts.append('''
**条件化输出预测的正向增量。** 用上述真实B36种子递推根A，再以发现集拟合原始h40读出器，接原模型最终RMSNorm与完整词表head。KL为KL(p原模型||p预测)。这不是仅凭文本的图谱输出保真度。下表“全残差”仅为同一作答位置h37的5,120坐标，不含完整前缀KV；“真实根”是测得B39根A经同一拟合读出器。
''')
        parts.append(table(['分支','方法','全词表KL','选项准确率（原argmax口径）'],[
            [r['branch'],r['method'],f(r['mean_KL']),f(r['accuracy'])] for r in output['summary']]))
        parts.append('''
四种链预测×两个合格分支的8格均胜过均值与embedding冻结KL门，但远不及完整同位置h37。根预测有时优于真实根，可能来自收缩/平滑和读出器分布误差，不是预测比实测更接近机制。选项准确率与KL可以不同向。输出选项准确率仍展示原argmax口径，不用于严格行为准入；严格原模型及干预准确率另存。

**原模型干预与反事实效应。** 每分支固定前40确认家族、每家族四版本。B19操作正文位置的单元/8单元组A为0/2倍；另设同层匹配组和实际BF16写入范数随机方向。B36只操作作答末位的两条链32父单元及匹配组。两个实验各320输入、10臂，共6,400条逐臂记录，但共用同一80家族；不能将臂或位置计成独立N。1倍重算只是数值无操作，不是独立救援。B19单元/组0倍主检验保持alpha=.05/6，其他行为对比探索性。每分支40家族对dz=.5的名义功效约.89，六检验校正后约.70，不能排除小效应。
''')
        effects=[r for r in intervention['effects'] if r['kind']=='B19' and r['arm'] in ['single_0','group_0','matched_0']]
        parts.append(table(['分支','B19操作','间隔变化','区间','p','严格准确率 原→操作','全词表KL'],[
            [r['branch'],r['arm'],f(r['margin']['mean']),ci(r['margin']['ci']),f(r['margin']['p_two_sided']),
             f(strict['B19',r['branch'],r['arm']]['strict_baseline_accuracy'])+'→'+f(strict['B19',r['branch'],r['arm']]['strict_accuracy']),f(r['mean_KL'])] for r in effects]))
        parts.append('''
单个历史单元和整个预选组的0倍操作均未显著损伤答案间隔；共同响应尚未转为这套操作下的共同功能证据。这不证明没有作用、没有冗余或其他条件没有影响。单元随机方向严格范数合格数值34/160、标签25/160；组随机方向均160/160。资格不足不能声称控制已合格，合格子集也可能选择偏倚。匹配组差异与规则减记录作用差异分别留存，不把对照本身的显著性当作目标组特异性。

B36干预验收检验真正的“改动后会怎样”：由冻结F预测父A设0/2倍后的根G/V/A变化，与原模型实测变化比较，损失按发现尺度标准化。只有实际变化RMS≥.05且预测MSE≤零效应预测MSE的.9倍才通过；效果太小记为检验不足。
''')
        parts.append(table(['路线','方式','剂量','分支','实测变化RMS','MSE/零效应','95%家族区间','可检验','门'],[
            [r['method'],r['mode'],r['dose'],r['branch'],f(r['standardized_actual_RMS']),f(r['ratio_to_zero_effect']['ratio']),ci(r['ratio_to_zero_effect']['ci95']),'是' if r['informative'] else '不足','通过' if r['passed'] else '未通过'] for r in intervention['causal_predictions']]))
        parts.append(f"\n反事实预测共{len(intervention['causal_predictions'])}格，{verdict['causal_cells_informative']}格达到实际效应下限，{verdict['causal_cells_passed']}格通过预测门。相关路线误差为零效应基线212–471倍；权重路线2.49–4.40倍，且所有95%家族区间均在1以上。这是可测效应下的预测失败，不是所有干预无效。\n")
        parts.append('''
必须单独保留B36的正向计算线索：按真实权重关系预选的同一32单元组，0倍在数值/标签降低答案间隔，2倍提高；相对同层匹配组也有同向差异。它们是预定操作的次级结果，不应被失败的稀疏预测器抹掉。下表区间为探索性95%，严格准确率同分算错。
''')
        native36=[r for r in intervention['effects'] if r['kind']=='B36' and r['arm'] in ['parameter_0','parameter_2']]
        parts.append(table(['分支','B36操作','间隔变化','95%区间','p','严格准确率 原→操作','全词表KL'],[
            [r['branch'],r['arm'],f(r['margin']['mean']),ci(r['margin']['ci']),f(r['margin']['p_two_sided']),f(strict['B36',r['branch'],r['arm']]['strict_baseline_accuracy'])+'→'+f(strict['B36',r['branch'],r['arm']]['strict_accuracy']),f(r['mean_KL'])] for r in native36]))
        parts.append('''
0倍减匹配组0倍的间隔差：数值−0.3265625（95%区间[−0.364901,−0.288224]）、标签−0.21015625（[−0.240435,−0.179877]）。但0倍的“规则减记录”效应为+0.1546875/+0.134375（区间均正），即有限记录的损伤更大，不是规则条件特异损伤；2倍则为−0.16875/−0.15625。答案字母在家族间近似配平，但有限记录的正确语义固定为Not specified，缺少其他答案词域和自由输出控制。没有等实际写入范数、广泛任务副作用或中介阻断证据，不能认定该组编码共同抽象结构，不能认定输出作用一定经过所拟合B39根。这个结果具体支持“同一预选组在两类材料中实际参与输出”，同时反驳“冻结F准确解释这条作用链”；两件事分别记账。
''')
        parts.append('''
**严格审视与三图谱增量。** 外部图谱新增规则/有限记录、标题、数值/坐标/标签及测量失败的序列分支；内部图谱保存固定原生索引、两个读出位置全坐标响应、17张原生单元卡、1张组功能卡与反例；关联边区分响应复现、条件化输出预测、原生参数来源与干预验收，没有正式机制晋级。发现集全40×17,408未排序G/V/A均值、配对差与dz数组保留；概览连续16单元平均可能抵消相反作用，逐坐标PNG与数组保留背景和未截断值，色标与截色比例登记。可视化没有用于确认后再选点。

主要硬伤：两个分支未过行为测量；只有数值与标签两个合格分支，其中只有标签是完全未参与拟合的语义分支；词与构式重复使80家族不等于80种独立表达系统；规则/记录共同措辞尚未排除；正文读出可能错过早期局部计算；32个同位置A省略残差、注意力、归一化和历史位置；未直接操纵组合深度或复杂特征真值；未与同预算SAE/transcoder比较；未做自然语料、自由生成、独立节点复算或广泛任务副作用。当前BF16微小行为差未额外高精度重测。统计显著或审计通过均不消除这些限制。固定快照没有训练过程证据。

**智能理论与第一性原理。** 有限参数的可组合能力需要区分三个对象：外部输入关系、内部可复用状态/运算、输出实际使用。共同响应只是第一步映射；预测器若不能预测原模型对干预的响应，就不能直接当作该模型的生成机制。若所选状态未包含足够信息，递推失败可以是状态不足，不能直接归结为没有规律。G与V的联合分布、SiLU非线性以及down写入必须分别审查。下式仅为期望分解恒等式，实际原生BF16多一个舍入残差，不是本轮发现的新语义定律：
''')
        parts.append(r'''
$$\mathbb E[A\mid C]=\mathbb E[\operatorname{SiLU}(G)\mid C]\mathbb E[V\mid C]+\operatorname{Cov}(\operatorname{SiLU}(G),V\mid C).$$

RDC名称、原有部分映射与类型化三图谱接口保持不变：

$$\mathcal T^{\mathcal D}_{\ell,\tau}:(L(p),W_\ell(p),X_\ell(p))\rightharpoonup W_{\ell+1}(p),\qquad \mathrm{Atlas}_{\mathcal D}=(G_{\rm external},G_{\rm internal},E_{\rm association}^{\mathcal D}).$$

这里沿用历史定义：L为材料/程序p的外部语言描述，W为指定实验域内的类型化析因响应束，X为按层/位置/坐标组织的观察场，D为受检域、τ为条件/角色；本轮的原生H与拟合F不自动等同于历史W或已完成的T。图边须带条件、算法、可用输入、划分、预测结果与证据状态。没有引入新的全局统一公式，也没有从本轮推出人脑机制或AGI。

**下一阶段作为一个大任务。** 围绕“共同结构的识别、组合与实际使用”推进：第一阶段建立同一关系的多种措辞、同措辞不同关系、关系缺席/否定、位置长度配平，先做选项轮换与自由输出行为校准，将新语义分支和新组合深度独立封存。第二阶段用同一材料比较原生单元/组、稀疏字典和包含残差、注意力来源的状态，显式定义复杂特征真值及组合操作，以未见组合预测检验状态是否足够；不只提高节点数。第三阶段只对通过者逐段追踪QK选择、V/O写入、MLP联合门控与真实参数，进行匹配剂量、恢复及副作用检验，并比较反事实预测和最终输出KL。局部图先验收再连接扩展。训练期分支形成需独立的合法检查点或受控训练研究。
''')
        parts.append(f'''
**工程复核、恢复与资源。** 完整原生审计及下游执行审计通过，所有原始科学源码冻结哈希保持。严格同分门为保守准入补充，坐标失败时原流水线在内部分析前停止；暂停登记器超时和原回执恢复全过程保存。首个B19干预因Windows保存路径分隔符登记失败，原生块与重放块逐字节相同；只修改独立适配器的路径规范化，未改前向或拟合。所有最终基线重放与1倍臂逐位一致，KL最大0。算法测试5项通过；完整协议315项中314通过，唯一既有SOURCE-BINDING语言映射缺失保留，0错误。第一次协议测试在沙盒Windows本地socket等待，停止自有测试后以相同测试范围在允许回环网络环境完成；未改共享协议代码。一次性隔离测试凭据日志已脱敏并登记。离线报告桌面/手机、图像、证据链接和正文/未来题目隔离检查通过。

进度逐行刷新，记录步骤/数量/时间/预计余时、CUDA与DXGI共享显存。本轮日志记录CUDA分配峰值{peak('cuda_peak')/2**30:.4f}GiB、共享显存峰值{peak('shared_gpu_bytes')/2**20:.1f}MiB。B19/B36干预分别耗时{B.read(B.OUT/'intervention-B19/result.json')['seconds']:.2f}/{B.read(B.OUT/'intervention-B36/result.json')['seconds']:.2f}秒。所有新增文件在D盘并通过storage_policy，预留20GiB；原生压缩数组{native['total_bytes']:,}字节，保留用于局部窗口/状态充分性/跨层复核，复查日期2026-12-13。仅清理本轮专属临时目录与已核对的重复失败块，移除文件字节{cleanup['removed_file_bytes']:,}；v2/v3相同试跑123文件经逐项SHA及行ID核对后以NTFS硬链接保留两个历史路径，减少重复物理内容{cleanup['deduplicated_file_bytes']:,}字节。文件字节之和不等于并行工作期间全盘空闲增量。旧约61GiB字段、全部独有原场、失败版本、日志、冻结/交付源码与当前逐臂数据保留。

**实际入口与交付。** 脚本全部位于tests/glm5，临时脚本在tests/glm5_temp；结果在tests/glm5/result/branch_reuse14_20261113_v3。命令依次为branch_reuse14_materials.py、collect.py freeze/collect、analysis.py plan/select、execution.py、qualified.py pipeline、group_cards.py、fieldmaps.py、audit.py native/final、report.py、node branch_reuse14_browser.cjs、closeout.py archive/cleanup、memo.py及closeout.py manifest。逐项参数、恢复入口、证据边界见branch_reuse14_README.md；冻结归档与delivery-source.zip可追溯真实版本。没有客户端/公网部署、正式中心登记或新GPU模型测试。

通俗说明：已经找到一批在不同题型中反复“亮起来”的单元，但它们是否在做同一种抽象工作，仍不能只凭亮度确认。把所选单元连成三段预测链后，正常输出能猜对一部分，内部数值和改动后的变化却未同时通过检验。这张图还不能当成模型真正使用的通用加工流程。下一步关键是补齐必要状态并控制措辞，再证明同一组连接能处理未见组合，而不是继续扩大未经验证的相关图。
\n''')
        return ''.join(parts)

    main=append(B.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md',main_text)

    def index_text(phase,stamp):
        return f'''\n\n## Phase {phase}: Qwen3-14B 跨分支复用与层间组合验收索引 [{stamp}]

详见GLM5 Phase {main['phase']}，RUN `{B.RUN}`。1,536条v3原始采集；坐标分支确认行为门失败后，按冻结部分准入只分析数值/标签640条、160家族，原三分支联合门不可补过。预选B19组7/8、B39根6/8单元复现共同A响应，但历史B19/U64主解释与两条三层稀疏链未过完整联合门。8格条件化输出KL均胜过均值/embedding，明显不及完整同位置h37；B19单元与组0倍没有显著答案间隔损伤；同一B36权重组0倍/2倍则对两分支产生同向可测损伤/提升，尚未确证共享语义。反事实根G/V/A预测{verdict['causal_cells_passed']}/{len(intervention['causal_predictions'])}格通过，具体效应下限与失败见主记录。更复杂特征组合与全模型机制存在性仍未确定，不能以候选失败否定整个方向。6,400逐臂结果来自同一80家族，未扩大独立N。原生11,038,909,213字节保留；审计/浏览器通过，协议314/315含既有映射失败。报告tests/glm5/result/branch_reuse14_20261113_v3/report.html；未部署或正式晋级。RDC公式不变，下一阶段聚焦措辞控制、足够状态、未见组合及反事实预测的一体化验收。
'''
    index=append(B.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md',index_text)
    B.save('memo-append.json',dict(entries=[main,index],source_sha256=B.file_sha(__file__),append_only=True))
    print(json.dumps(dict(main_phase=main['phase'],main_line=main['line'],index_phase=index['phase'],prefixes_preserved=True),ensure_ascii=False),flush=True)


if __name__=='__main__':
    run()
