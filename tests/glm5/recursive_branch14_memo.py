"""Append the completed campaign and a unified-index cross reference, never rewrite history."""
import hashlib
import re
import time
import numpy as np
import recursive_branch14 as R


NAMES={'structure':'B26/U13099','procedure':'B19/U64'}
SPLITS={'new_words':'新词','new_frames':'新词＋新句式','new_domains':'新词＋新领域背景'}


def number(v):return '不适用' if v is None else f'{v:.6g}'


def append(path,body):
    before=path.read_bytes();encoded=body.encode('utf8')
    with path.open('ab') as stream:
        stream.write(encoded);stream.flush()
    after=path.read_bytes()
    assert after==before+encoded,'Concurrent edit or unexpected non-append change'
    return dict(path=str(path.relative_to(R.ROOT)),old_bytes=len(before),appended_bytes=len(encoded),
        before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(after).hexdigest(),
        preserved_existing_bytes=True)


def run():
    R.configure();R.admission()
    assert not (R.OUT/'memo-append.json').exists(),'Campaign already appended'
    audit=R.read(R.OUT/'audit.json');assert audit['passed']
    assert R.read(R.OUT/'browser-check.json')['passed']
    response=R.read(R.OUT/'response-report.json');effects=R.read(R.OUT/'intervention-report.json')
    controls=R.read(R.OUT/'control-comparisons.json');native=R.read(R.OUT/'native-manifest.json')
    plan=R.read(R.OUT/'plan.json');selection=R.read(R.OUT/'selection.json')
    memo=R.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
    phase=max(map(int,re.findall(r'^## Phase (\d+):',memo.read_text(encoding='utf8'),re.M)))+1
    stamp=time.strftime('%Y-%m-%d %H:%M')
    text=f'\n\n## Phase {phase}: Qwen3-14B 关系分支的三类留出、原生单元干预与末层回溯验收 [{stamp}]\n\n'
    text+='**任务边界与结论。** 按用户要求仅在本机Qwen3-14B上串行测试。前轮六个解释候选中，只有B26/U13099与B19/U64属于14B，本轮直接验证这两个；四个4B候选保持待检，没有把原生编号跨模型迁移。两个冻结的“共同关系在正文末位表现为正向G响应”的解释都未通过三类留出的联合门；两组B39向B36回溯链也未通过所有基线与泛化门。这个结果否定本次具体解释、位置与预测算法的联合命题，不否定更宽的形式对象、排序或条件齿轮假设，更不能从固定快照推断训练时如何形成递归分支。当前不应优先扩展为全模型逐层相关图，应先让局部候选同时通过更强预测基线与原模型作用检验。\n\n'
    text+=f'运行ID `{R.RUN}`；模型原始BF16快照 `{R.SNAPSHOT}`，CPU读取safetensors并初始化RoPE，RTX 5080 CUDA逐块执行，SDPA，batch=4、padding长度192。最终重新核对模型、tokenizer全套文件与冻结时哈希、依赖版本一致。版本见plan.json。所有新增数据写D盘并通过storage_policy，冻结预计新增峰值72GiB、保留20GiB；真实子进程持GPU租约，未加载4B或GLM。\n\n'
    text+='**先校准行为，再冻结。** v5发现集试跑的“引用但未生效”条件失败，部分排序表达也失败。只在发现材料上做诊断，显式排序修复了表述问题，但生效判断没有整体通过。v6改为指令式与参考描述均提供可按假设执行的关系，校准379/384；这不是旧命题复活。沿用原家族ID及其切分，冻结v6后才采集三类留出。早期准备、元数据失败、v5原结果、诊断及修订原因保留。调度器只把连续空闲等待30秒改为5秒并记录前后哈希，GPU互斥和冻结科学源码未改。\n\n'
    text+='**材料、观测与统计。** 正式6,144条输入、1,024个家族，每家族三种表达（指令、参考描述、条目清单）×两种标题。每候选每划分128家族、768条；发现集与三类留出各1,536条。新词是发现阶段未用的常用填充词，不能声称模型训练未见；新句式同时含新词，新领域只是受控背景迁移。正文结束共同token位于问题和选项之前，第二读出位于答案提示末尾。保存hs[0]加40层全部5,120坐标，以及40块全部17,408单元的G/V/A，均为原生BF16位模式；两候选另存每token G/V/A。40×17,408=696,320个单元是覆盖量，不是破解进度。输入embedding、前缀平均embedding、全部原文、分词、位置及行ID均保留。18,432个选项续接通过实际tokenizer边界核对；236组相同前缀、不同后续问题的旧新试跑在全部40块G/V/A逐位相同。\n\n'
    text+='| 划分 | 关系配对行为题 | 次序变换行为题 | 逐格行为门 |\n| --- | ---: | ---: | --- |\n'
    for split in R.SPLITS:
        data=R.read(R.OUT/split/'collection.json');rows=R.materials();counts=[]
        for h in R.CANDIDATES:
            ids=[i for i,g in enumerate(data['row_indices']) if rows[g]['hypothesis']==h]
            correct=sum(int(np.argmax(data['scores'][i]))==rows[data['row_indices'][i]]['gold'] for i in ids)
            counts.append(f'{correct}/{len(ids)}')
        text+=f'| {split} | {counts[0]} | {counts[1]} | '+('通过' if data['behavior_passed'] else '未通过')+' |\n'
    text+='\n**计算对象与冻结判据。** 对块l、位置t、单元j，z是实际MLP归一化输入，g/u为gate/up权重行，d为down列；下式是原模型计算定义，不是新经验定律：\n\n'
    text+=r'$$G_{ltj}=g_{lj}^{T}z_{lt},\quad V_{ltj}=u_{lj}^{T}z_{lt},\quad A_{ltj}=\operatorname{SiLU}(G_{ltj})V_{ltj},\quad \Delta h_{ltj}=A_{ltj}d_{lj}.$$'+'\n\n'
    text+='每家族先在标题版本内配对，再平均。关系差为(G指令＋G参考)/2−G清单。预期正向，dz≥0.35、六项主检验p<0.05/6，两种表达各自差值为正，且词汇＋关系元数据的G预测误差≤词汇基线0.9倍；三类留出全部通过才保留该解释。以128家族、dz=.35做近似功效规划为0.907。元数据为实验者已知标签，不是可从未标注文本自动提取的机制。比较512维词/双词哈希、位置长度、前缀平均embedding、语义元数据和组合模型；拟合与组选择只用发现集。\n\n'
    text+='| 候选 | 留出 | G关系差 | 家族dz | 双侧p | 组合/词汇G误差 | 冻结门 |\n| --- | --- | ---: | ---: | ---: | ---: | --- |\n'
    for r in response['results']:
        c=r['contrasts']['G']['relation']
        text+=f"| {NAMES[r['hypothesis']]} | {SPLITS[r['split']]} | {number(c['mean'])} | {number(c['dz'])} | {number(c['p_two_sided'])} | {number(r['combined_to_lexical_G_MSE'])} | "+('通过' if r['primary_passed'] else '未通过')+' |\n'
    text+='\nB26只有新句式一格通过；B19在新词与新领域条件的方向与预期相反。六项G预测全部不如前缀平均embedding，说明胜过当前较弱词汇哈希基线不能当作机制提取成功。G、V、A方向和直接/组合A预测误差全部分开保留。事后逐token诊断发现，两单元句中常有短暂正G，清单对照也会出现；B19的正文G峰值在三类留出均更偏关系描述，但峰值依赖句长、词语和极值选择，只是下一轮位置假设。B26正文末位G全负不等于关闭，SiLU负半轴仍非零。清单也提到“没有关系/过程”，不能把它当作语义概念绝对缺席。反例与对应原文收录在counterexamples.json和报告。\n\n'
    text+='保留的正向次级线索：B19/U64的V关系差在新词/新句式/新领域依次为+0.20663/+0.23759/+0.25881，A为+0.018252/+0.010302/+0.018385；各自探索性95%家族区间均为正。它们是预定次级对比，不是正向G主命题通过，也没有排除词汇、措辞和任务形式。后续可预注册联合V/A写入与位置假设，不能在本轮开封后换指标补门。B26的V与A随句式翻转，缺少同样的跨条件方向稳定性。\n\n'
    text+='**原模型单元与单元组干预。** 每候选固定192个目标家族＋48个另一任务家族、每家族六变体，共1,440条件、10臂；两项合计28,800条逐臂结果。将正文全部位置的原生A设为0倍或2倍，再走真实down投影及后续层。8单元组和同层匹配组只在发现集选择，另有实际BF16写入范数随机方向控制。名为restore的1倍重算臂是无操作数值一致性，不是独立救援。主指标m=正确选项logit−最强错误选项logit，效应δ=m干预−m基线；在家族内平均六变体，再在家族间推断。四项主检验（两候选×单元/组0倍）alpha=.05/4；其余为探索。192家族针对dz=.35的近似功效0.991；单一指令/中性标题准确率8点变化的规划功效0.843，不覆盖任意小效应。\n\n'
    text+='| 候选 | 操作 | 目标家族间隔变化 | 校正区间 | p | 原准确率→操作后 | 平均全词表KL |\n| --- | --- | ---: | --- | ---: | ---: | ---: |\n'
    for r in effects['results']:
        if r['scope']=='target' and r['role']=='all' and r['arm'] in ['single_0','group_0','matched_group_0']:
            c=r['margin_change']
            text+=f"| {NAMES[r['hypothesis']]} | {r['arm']} | {number(c['mean'])} | [{number(c['ci'][0])}, {number(c['ci'][1])}] | {number(c['p_two_sided'])} | {r['baseline_accuracy']:.4%}→{r['accuracy']:.4%} | {number(r['mean_KL'])} |\n"
    text+='\n匹配组行采用探索性95%区间；主检验行采用98.75%区间。\n\n'
    text+='两个单元0倍的目标效应均未显著。B26八单元组0倍的主检验达到显著，但方向为间隔小幅改善（约+0.00825 logit），并非损伤；其2倍也略改善，不能据此建立简单的单调必要作用。相对同层匹配组的差异区间含零，两任务关系减清单的干预效应差也均未显著。B19单元与8单元组0倍的正确率不变，间隔变化接近零；这限定了本材料、位置范围和剂量下的主张，不证明候选在其他语境完全无作用。当前分数沿用原始BF16计算，没有额外高精度复现小效应。\n\n'
    for c in controls['results']:
        if c['contrast']=='group0 minus matched-group0':
            v=c['comparison'];text+=f"{NAMES[c['hypothesis']]}的选组0倍减匹配组0倍效应={number(v['mean'])}，探索性95%区间[{number(v['ci'][0])},{number(v['ci'][1])}]。"
    text+='\n\n'
    for h in R.CANDIDATES:
        for arm in ['single_norm_control','group_norm_control']:
            r=next(x for x in effects['results'] if x['hypothesis']==h and x['scope']=='target' and x['role']=='all' and x['arm']==arm)
            c=r['strict_matched_native_minus_control']
            text+=f"{NAMES[h]} {arm}逐token全部满足2%容差的目标行{r['strict_norm_matched_rows']}/{r['rows']}；"
            text+=f"合格子集原生减随机方向效应{number(c['mean'])}，涉及{c['n_families']}家族。\n\n" if c else '没有有效的严格范数匹配比较。\n\n'
    text+='严格范数覆盖不足时不能宣称控制合格；合格子集也可能发生选择偏差。另一任务只有48家族，不能当作广泛能力保持；0倍与2倍结果、错误转移、完整词表KL和所有不合格控制均保留。损伤、反向提升和无变化都不能单独证明关系语义、充分性、必要性或唯一性。\n\n'
    text+='**从最后一层回溯。** 每主题在B39发现集选8个G响应根，向B38/B37/B36各选32父单元，联合预测子G/V/A；岭回归比较均值、词汇及同数随机父组。以真实B36 A为起点，后续层只用预测G/V组合成A向前递推。这不是只凭文本的图谱到完整下一词分布预测。三类留出要求每段MSE同时≤对应基线0.9倍，且自由递推仍优于均值/词汇。\n\n'
    text+='| 主题 | 新词递推MSE/均值 | 新句式 | 新领域 | 三类留出联合门 |\n| --- | ---: | ---: | ---: | --- |\n'
    for h in R.CANDIDATES:
        rr=[next(x for x in response['backward_chains'] if x['hypothesis']==h and x['split']==s) for s in R.SPLITS[1:]]
        text+='| '+NAMES[h]+' | '+' | '.join(number(x['free_rollout']['ratio_to_mean']) for x in rr)+' | 未通过 |\n'
    text+='\n新词上的显著预测增量值得保留，但没有一段在三类留出中同时满足全部门，主要瓶颈包括随机父组优势不足和跨条件崩溃。全部链条虽胜过本次弱词汇模型，不能据此晋级。全40层未排序G/V/A原始差与配对dz热力图、完整17,408坐标数组已经生成；显示固定色标、截色比例、坐标与分辨率限制随图保存。只用发现集绘图，不根据留出重选父组。全层采集是后续复用资源，当前只完成最后三段预测链验收。\n\n'
    text+='**严格审视与三图谱增量。** 外部图谱新增的是材料维度“关系呈现方式、技术标题、句式”，不能据此建立神经模块。内部图谱保存两个有反例的原生特征卡、全层响应与候选预测边。关联机制明确记录未过冻结门、范围内行为扰动和未知上游，而非给相关热图加因果箭头。更强上下文基线、同预算SAE/transcoder、自然语料复现、跨节点独立复算、自由生成、完整概率分布预测、广泛副作用和训练检查点均缺失。材料词池和句式数量有限；题目固定查询第一来源，可能用固定槽位策略；模板家族统计独立性仍受重复词/框架影响。统计p值不消除这些设计限制。正文末位检验的是保持到该位置的响应，不能否定早期局部触发。\n\n'
    text+='**第一性原理与理论进展。** 要解释有限参数如何组合，首先要区分输入关系、条件状态、乘法写入与下一词使用。单一G均值不是完整机制；对任意条件C，以下是期望分解恒等式，不是本轮发现的新定律：\n\n'
    text+=r'$$\mathbb E[A\mid C]=\mathbb E[\operatorname{SiLU}(G)\mid C]\mathbb E[V\mid C]+\operatorname{Cov}(\operatorname{SiLU}(G),V\mid C).$$'+'\n\n'
    joint=R.read(R.OUT/'joint-state-summary.json')['summary']
    ex=next(x for x in joint if x['hypothesis']=='procedure' and x['split']=='new_words' and x['role']=='operative')
    qr=next(x for x in joint if x['hypothesis']=='procedure' and x['split']=='new_words' and x['role']=='quoted')
    text+=f"事后联合描述中，B19/U64新词指令条件的原生平均A={number(ex['mean_native_A'])}，平均SiLU(G)×平均V={number(ex['product_of_mean_siluG_and_meanV'])}，协变项={number(ex['conditional_covariance'])}，原生BF16舍入残差={number(ex['native_mean_minus_fp64'])}。新词参考描述条件中，G/V双负样本占绝对A总量{qr['quadrants']['--']['native_abs_A_share']:.2%}。这是联合状态信息的具体例子，不能据此追认失败的正向G假设或单元语义；所有划分/表达的分解与象限份额保留在joint-state-summary.json。\n\n"
    text+='此外SiLU(E[G])通常不等于E[SiLU(G)]。所以把两个预测均值相乘可能漏掉非线性与条件协变；联合G/V/A和真实写入必须分别预测、分别审查。只用上一层少数A预测下一层，省略了残差、注意力、归一化及历史位置，未必构成足够状态；本次递推失败不能区分“没有规律”和“输入状态不充分”。优先检验可在不同表达下复用的条件转移，而不是把图形分枝形状当作递归形成证明。RDC名称与既有统一机制公式不变；本轮没有建立新的全局机制公式，也未完成图谱输出保真度。\n\n'
    text+='**下一阶段作为一个大任务推进。** 目标是“共同关系怎样形成可推广且有行为作用的原生写入”。第一阶段补充自然表达、概念否定与无关关系、位置/长度/局部词控制，按新实体、新句式和新组合独立封存，先校准行为；将本轮正文峰值改成明确位置窗口假设再冻结。四个4B分支主题须在14B重新发现原生候选，不能直接搬编号。第二阶段在同一发现/留出材料上比较单元/单元组、强上下文embedding、稀疏字典及包含残差与注意力来源的联合状态，先验收G/V/A与未见组合预测，并计算真正图谱输出KL。第三阶段仅围绕通过者，从输出读出向前追踪小范围计算连接，分离输入选择、门值、载荷和写入，做匹配剂量、随机组、独立恢复及跨任务副作用；每完成一个有效局部图再扩展层数。训练期递归来源另行需要合法训练检查点或受控训练，不从本轮静态数据外推。\n\n'
    text+='**工程复核、资源和保存。** '
    for h in R.CANDIDATES:
        d=R.read(R.OUT/f'intervention-{h}.json')
        text+=f"{NAMES[h]}干预{d['cases']}/1440，耗时{d['seconds']/60:.2f}分钟，原始重放最大KL={number(max(c['replay_KL_max'] for c in d['checks']))}，1倍重算逐位一致={all(c['restore_bitwise'] for c in d['checks'])}；"
    lines=[R.json.loads(line) for line in (R.OUT/'progress.jsonl').read_text(encoding='utf8').splitlines() if line.startswith('{')]
    text+=f"进程CUDA分配峰值{max(x.get('cuda_peak',0) for x in lines)/2**30:.3f}GiB，共享显存峰值{max(x.get('shared_gpu_bytes',0) for x in lines)/2**20:.1f}MiB。全机可用RAM最低{min(x['ram_available_bytes'] for x in lines if 'ram_available_bytes' in x)/2**30:.3f}GiB，不能全部归因于本进程。"
    repeat=audit['pilot_discovery_replay']
    text+=f"原生数组形状、有限值、入口缓存、逐项哈希通过；试跑与正式发现重复采集最大相对L2={number(max(x['relative_L2'] for x in repeat))}。"
    text+='算法5项通过，完整协议315项中314通过；唯一失败为此前已有SOURCE-BINDING-20261110.source映射缺失，本轮96项RG14维度映射均通过，未修改无关冻结文法掩盖失败。协议测试的环境隔离与兼容修正、失败日志及临时凭据行脱敏记录保留。离线报告浏览器检查通过，桌面、手机、图片、原文/后续问题分离和证据链接均核验。没有新增客户端发布或正式服务登记。\n\n'
    text+='浏览器首次登记回执时，一次性包装命令漏掉R.configure，Git信任目录环境未设置导致provenance断言失败；页面检查本身已完成。初始化同一研究环境后重新登记既有检查，原失败元数据与恢复说明保留在browser-receipt-recovery.json；模型、统计分析与页面均未因此改动。\n\n'
    cleanup=R.read(R.OUT/'final-cleanup.json')
    total=cleanup['total_bytes']+R.read(R.OUT/'test-cleanup.json')['total_bytes']
    text+=f"核心原生数组{native['total_bytes']:,}字节，登记用于后续位置/转移研究及复核，复查日期{plan['retention']['review_date']}。已按归属和解析路径清理本轮独占的协议测试目录、可重建缓存与已逐条核对的重复分片，共{total:,}字节；文件字节合计不等于同时间全盘空闲变化。保留失败证据、冻结源码、当前交付源码、日志、原场、完整逐臂结果、反例、报告及SHA256清单。\n\n"
    text+='实际入口：recursive_branch14_materials.py / revise、recursive_branch14.py freeze/pilot/collect、recursive_branch14_pipeline.py、recursive_branch14_analysis.py select/analyze、recursive_branch14_intervene.py run/summarize、controls/cards/audit/fieldmap/position_diagnostic/report/browser/closeout/memo。脚本位于tests/glm5，临时脚本位于tests/glm5_temp，结果与可读报告位于tests/glm5/result/recursive_branch14_20261113_v6。详细恢复步骤见recursive_branch14_README.md。\n\n'
    text+='通俗说明：先前给两个神经元取的“共同关系”名字，没有经受住完整的新材料检验；某些条件下仍有响应和预测能力，不能简单丢弃。现在已经把模型整条40层的原始测量保留下来，但从末层倒着画出一棵树，并不等于找到了语言的生成规则。下一步应先找出几段能预测新表达、改动后又确实按预期影响行为的连接，再逐段接起来。\n'
    receipts=[append(memo,text)]
    unified=R.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md'
    phase2=max(map(int,re.findall(r'^## Phase (\d+):',unified.read_text(encoding='utf8'),re.M)))+1
    cross=f'\n\n## Phase {phase2}: Qwen3-14B 关系分支验证与末层回溯索引 [{stamp}]\n\n'
    cross+=f'详见GLM5 Phase {phase}（research/glm5/docs/AGI_GLM5_MEMO.md），RUN `{R.RUN}`。只测试原始BF16 Qwen3-14B；两个历史14B候选在6,144条/1,024家族材料上完成三类留出与单元/8单元组干预，另四个4B候选待检。两项正文末位正向G解释与两组B39→B36回溯链均未过全部冻结门；六项G预测不如前缀平均embedding。全40层原生G/V/A和hidden场可复用，干预、匹配控制、反例与限制分开记录。报告：tests/glm5/result/recursive_branch14_20261113_v6/report.html；审计audit.json通过，协议315项中314通过，既有SOURCE-BINDING映射失败保留。本轮不建立训练期递归形成或完整图谱结论，未登记正式服务。优先推进更强状态预测和输出锚定的局部机制图，不铺开未验收的全层相关图。\n'
    receipts.append(append(unified,cross))
    R.save('memo-append.json',dict(phase=phase,unified_index_phase=phase2,local_time=stamp,receipts=receipts,
        source_sha256=R.file_sha(__file__)))
    print(f'Appended GLM5 Phase {phase}; unified index Phase {phase2}',flush=True)


if __name__=='__main__':run()
