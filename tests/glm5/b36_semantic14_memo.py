"""Append this completed stage to both research records, verifying old prefixes."""
import hashlib,re
import b36_semantic14 as S

def append(path,title,body):
    original=path.read_bytes();text=original.decode('utf-8-sig');phases=[int(s) for s in re.findall(r'^## Phase (\d+):',text,re.M)];phase=max(phases)+1
    stamp=S.now()[:16].replace('T',' ');addition=f'\n\n## Phase {phase}: {title} [{stamp}]\n\n'+body.strip()+'\n'
    with path.open('ab') as f:f.write(addition.encode('utf8'))
    after=path.read_bytes();assert after[:len(original)]==original and after[len(original):]==addition.encode('utf8')
    return dict(path=str(path.relative_to(S.ROOT)),phase=phase,original_bytes=len(original),appended_bytes=len(after)-len(original),original_sha256=hashlib.sha256(original).hexdigest(),prefix_unchanged=True,after_sha256=hashlib.sha256(after).hexdigest())

def run():
    S.configure();assert not (S.OUT/'memo-append-receipt.json').exists()
    a=S.read(S.OUT/'audit.json');decision=S.read(S.OUT/'stage-decision.json');assert decision['status']=='behavior_gate_failed'
    expanded=next(s for s in a['stages'] if s['stage']=='expanded');env=S.read(S.OUT/'execution-environment.json')
    lines=['步数 | 格式 | 候选严格正确 | 全词表首token正确 | 描述性Wilson95%区间 | 准入','--- | --- | --- | --- | --- | ---']
    for c in expanded['cells']:lines.append(f"{c['depth']} | {c['format']} | {c['strict_correct']}/{c['n']} | {c['top1_correct']}/{c['n']} | [{c['top1_wilson95'][0]:.3f},{c['top1_wilson95'][1]:.3f}] | {'通过' if c['passed'] else '失败'}")
    ledger=['材料 / 接口 | 已提交 / 计划 | 候选严格正确 | 全词表首token正确','--- | --- | --- | ---']
    for s in a['stages']:
        for f in s['frames']:ledger.append(f"{s['stage']} / {f['frame']} | {f['n']}/{f['planned']} | {f['strict']}/{f['n']} | {f['top1']}/{f['n']}")
    body=f'''**目标与实际完成状态。** 用户要求先在独立新材料校准直接答案、答案代码轮换与两三步组合，再结合中间语义真值追踪B36固定32单元的实际输出路径，区分共同语义与格式效应、检验复杂特征逐层形成。本轮完成新材料、真实CUDA行为校准、失败诊断、独立HF调度核对及交付审计；联合行为门失败，因此依预注册停止受影响的机制正例步骤。没有新增B36干预、特征拟合、路径选择或机制原场采集，没有声称已完成用户要求的后半段。运行ID：{S.RUN}。

**模型与边界。** 主模型Qwen3-14B原始BF16，快照`{S.SNAPSHOT}`。14个权重/tokenizer文件重新逐一哈希，并与本地HF缓存树revision `40c069824f4251a91eefaf281ebe4c544efd3e18`一致；没有新远程来源认证。CPU读取原safetensors，40块逐块送CUDA后释放，SDPA、batch4、右padding至256、实际末位读出、无量化、普通评分无KV缓存。软件版本：Python {env['python']}、torch {env['packages']['torch']}、transformers {env['packages']['transformers']}。另串行运行Qwen3-4B原始BF16参考检查，快照`6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad`；不把14B的B36索引迁移到4B，不合并证据。前一模型进程结束后才运行另一模型，没有GLM4新测试。

**问题、算法与数学定义。** 对外部任务定义初始状态s₀、真实操作Tₖ与中间状态sₖ=Tₖ(sₖ₋₁)，最终答案a=π(s_d)，π是状态到答案代码的映射，直接答案时为恒等。这些是可独立计算的任务真值，不预设模型内部逐步运行同一算法。中间真值只存为元数据，不把正确中间答案塞进提示。实际模型作答末位的logits记z，候选严格正确当且仅当m=z[a]−max(z[b],b∈候选且b≠a)>0；无约束下一token正确当且仅当argmax_v z[v]=token(a)。候选均在真实续写边界校验为单token，无打分长度偏差。候选选择只覆盖列出的竞争项，不能替代全词表或自由生成。程序长度d不是神经层数；模型也可能并行求值或构造全局排序。

**材料与划分。** 原创规则图按全6!状态重命名与两操作交换规范化，得到256个不同操作对；冻结选取12试跑、48后续发现、96确认与32锁箱图。两组词表属于同一逻辑图的配对视图，不能计作两倍家族。12试跑家族计划1152条，448条提交后，两候选句式均已出现至少一格即使余下全答对也无法达到90%，故停止其余704条；受中断且未提交分片不计结果。该停止证明不需推断总体准确率。随后只复用其中3个完整暴露家族改接口，另用6个新算术发现程序、12个新排序家族校准，检查普通chat、Answer前缀、函数式/文字式表达与两条固定示范。排序题查询链内部节点，仍可能由全局排名回答，并不能确定实际推理路径。

Python chat在3个旧普通算术程序的两步直接/两种代码题都答对后，固定接口，在已有算术文法的discovery池选择48个新完整程序家族。显式排除其1/2/3步语义前缀与旧6程序的重合，也排除新家族之间的重复前缀；不只按完整三步程序ID切分。每个家族9个视图：深度1/2/3×direct/code0/code2，共432条。代码字母保持A/B/C/D正常顺序，只轮换对应数字。这是新发现校准，未升级成盲确认；旧接口诊断已参与选择。确认、锁箱模型响应开封数为0。完整阶段2176条已提交14B测量来自71个构造家族（5个已测规则图、54个算术程序、12个排序家族），不能称2176个独立样本；工程HF复核4题及4B的18次参考生成另记。

**冻结的扩大校准准入。** 48家族每格须两指标均≥90%，即至少44/48；9格全部满足才进入联合两/三步B36机制实验。不通过后禁止改阈值、只选答对题或只保留成功格补过。本轮不是消融正确率效应检验，未借用McNemar功效作为过门证明；n48是发现校准筛查，在p=.9时准确率标准差约.0433。逐格Wilson区间是描述性95%，不是9格同时区间，准入按冻结点估计执行。原拟96家族确认尚未运行。

'''+ '\n'.join(lines)+'''

联合门仅3/9格通过：单步code0、单步direct及两步direct。两步直接作答46/48，代码映射后分别39/48、43/48；三步直接33/48、code0为32/48、code2为41/48。不能笼统说代码更难或模型完全不会组合。事后描述性逐家族配对：两步direct→code0有9题从对变错、2题从错变对，code2对应5与2；三步direct→code2则3题从对变错、11题从错变对。改变代码也改变题面、选项上下文和输出读出；这不是固定内部状态上的纯输出头中介实验，不能指定其因果来源。

**全部诊断账本。** 下表为复现与排错，不用汇总平均替代逐格准入。小诊断格有的只有3个家族，不据此估计一般能力。

'''+ '\n'.join(ledger)+f'''

**实际反例和运行器核对。** 新家族`aad47bca25ee332f:3:direct:51`给Python x=2; x+=5; x-=2; x-=5; print(x)，真值序列2→7→5→0，原模型首token为2，候选严格选择也错。另有x=2; x*=2的单步题首token为8，但在四候选内正确项4严格领先，说明窄候选分数不等于完整答对。独立工程复核对4条明确题走标准HF自动构造的掩码/位置，HiddenState逐位相同、全词表最大logit差0；1+1与2+3答对，而((1+2)−2)+3直接回答2，标准HF不padding自回归输出也为“2<|im_end|>”。这支持特定错误不是本运行器掩码/位置调度差异，却不排除库、模型快照或题面自身的全部问题。

4B参考对同一三步算式能给4；部分题先解释，思考模式常在256token上限前没有给最终答案。保存完整生成文本、截断和exact_final字段；中途出现正确值不算最终成功。小样本跨模型差异不支持能力排名，也不证明14B内部不存在相关算法。原阶段主要测单token直接作答，广泛自然自由生成与停止行为仍缺失。

**严格审视。** 多次接口调整属于探索，所有失败都保留。重要扩大样本48家族虽比3家族更可信，仍只覆盖数字0–9、有限加减乘程序、两种代码轮换和单快照。未见完整程序及短前缀不等于未见基础运算。原规则图和排序题的词表/拓扑共享，算术表达式与文字描述也不是独立语义领域。早期只按完整程序划分可能共享短前缀，该局限已记录，后续确认必须对全部本轮暴露前缀去重。格式遵循差不能直接标成语义不会；先输出解释可能后来答对，本轮没有大规模生成证据。改变代码时无同长度、同上下文的全部正交控制。没有必要性、充分性、净中介、广泛任务副作用、同预算SAE/transcoder比较或独立节点复算。无原场、无干预是行为门失败后的预定停止，不是机制实验得到零效应。

**核心拼图保留与理论进展。** Phase2928的原生局部传播预测、ABC特定格式下组干预、XYZ格式反例和B39八根净恢复近零的边界全部保留；本轮既没有推翻也没有提升它们。Phase2929通俗解读保持。外部图谱新增可复算的操作、逐步真值、答案映射及行为失败格；内部图谱无新验证单元，固定B36的32个单元和匹配组仅作为待检目标保存在stage-decision.json；关联图不新增语义机制正例边。RDC主体与全局图谱公式没有修改，没有新统一数学定律、脑机制或AGI结论。上述sₖ、π、margin都是实验定义，不包装为内部编码新定律。

**第一性原理与下一大阶段。** 有限参数实现无限组合，需要说明外部关系的复合在内部对应什么可复用运算，以及语义运算如何与答案代码转换结合。一个特征在某层可读出不等于模型已用它完成任务；三步题也不预设模型沿三个层构造三步中间值。当前首要瓶颈是把“求值”“输出表达”“停止”分开验收，并区别沿网络深度的计算与沿自回归历史的计算。下一大阶段应作为完整“自产中间状态与答案映射联合验收”推进：①先在参考模型建立自然生成评分和截断分类，再在14B上用新家族校准直接答案、模型自行生成的简洁中间状态及完整代码轮换；中间结果不能由实验者预先提供，超时单独记账。②按实测每token成本冻结有限资源、样本功效与读出时间，封存程序前缀、句式和映射，分别记录语义终答、格式、停止；失败模式决定材料与状态定义，不继续无约束换提示词。③行为通过后冻结共同样本，比较全HiddenState、B36原生组、历史状态与可比预算稀疏特征；预测逐步真值的可读出时刻及未来响应，排除身份/位置基线。④最后按发现集的真实净输出作用选择B36写入到后三层attention/MLP的路径，在留出上分别阻断、恢复并检验代码等变性及副作用，确认复杂特征是否逐层或跨生成步形成。以上为未执行计划，不能写入本轮成绩。

**工程、资源与交付。** 3项材料工程测试通过；audit.py复算2880条计划材料（含704未执行行）的独立规则/算术/排序真值、token边界与2176条成绩，所有已执行来源/计划哈希及分片身份一致。原生与HF复核按以上有限范围通过。共享协议、生产runner与客户端未改，未重复全套协议回归。源码编译及源码包逐字节回读、报告链接检查通过；图表人工查看，未声称浏览器桌面/手机渲染验收。新增文件全部D盘，按storage_policy保持20GiB余量；交付时约{env['free_disk_bytes']/2**30:.2f}GiB空闲。14B CUDA分配峰值{env['peaks']['qwen14b']['cuda_peak_bytes']/2**30:.3f}GiB，4B参考{env['peaks']['qwen4b_reference']['cuda_peak_bytes']/2**30:.3f}GiB；两者DXGI进程共享峰值均82MiB。系统提交峰值含其他并行进程，不能当作本任务独占用量。原场预算未动用，收尾仅清理本轮独占字体缓存/临时目录，实际清理字节与保留结果容量见cleanup-receipt.json；失败材料、日志、冻结计划与源码均保留，无历史证据删除。无正式服务器发布或证据登记。

脚本：tests/glm5/b36_semantic14*及test_b36_semantic14_materials.py；实际命令与复核入口见b36_semantic14_README.md。报告：tests/glm5/result/b36_semantic14_20261113_v1/report.html；同目录audit.json、stage-decision.json、expanded-plan.json、paired-format-description.json、execution-environment.json、delivery-source.zip与manifest.json可逐项复核。记录采用尾部追加并验证原字节前缀不变。

通俗说明：现在看到“会算”和“会按指定编号回答”并不总同步。48个新程序中，两步直接写数字做对46个，换成一种字母编号只做对39个；三步换另一种编号却从33个提高到41个。由于这些条件没有一起达到可靠水平，继续给32个单元贴上“共同复杂语义齿轮”标签会越过证据。本轮完成了可靠性检查，尚未完成内部路径验证。
'''
    first=append(S.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md','B36新材料行为校准：直接答案、代码轮换与两三步组合的准入边界',body)
    index=f'''本阶段详见GLM5 Phase {first['phase']}，运行{S.RUN}。主模型Qwen3-14B原始BF16串行CUDA执行，另有独立Qwen3-4B参考检查。完成2176条发现/诊断测量，来自71个构造家族；确认/锁箱响应未开封。扩大Python校准48个新程序家族、432个配对视图，显式排除暴露和重复的短程序前缀。两步直接作答46/48，代码映射39/48与43/48；三步为33/48、32/48、41/48。冻结联合门仅3/9格通过，因此B36路径干预、特征形成验证及机制原场未执行。不能用成功格或答对题子集补过，也不能推出模型普遍没有组合能力。

3项材料工程测试及2880行材料/2176条成绩审计通过，标准HF四题HiddenState与logits对齐，失败不是该有限范围的调度差异；技术审计不升级科学结论。保留全部失败、4B截断与格式差异，RDC无新公式，Phase2928既有传播拼图边界不变。下一大阶段先联合验收模型自产中间状态、语义终答、格式和停止，过行为门后再比较层内与生成历史状态并追踪B36净输出中介。报告tests/glm5/result/b36_semantic14_20261113_v1/report.html，入口tests/glm5/b36_semantic14_README.md。全部新增数据D盘，清理仅本轮临时缓存，无发布。'''
    second=append(S.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md','Qwen3-14B B36组合与答案格式行为准入索引',index)
    S.save('memo-append-receipt.json',dict(records=[first,second]));print('Appended GLM5',first['phase'],'Claude',second['phase'],flush=True)

if __name__=='__main__':run()
