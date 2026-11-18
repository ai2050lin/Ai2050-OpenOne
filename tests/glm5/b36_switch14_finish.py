"""Readable delivery, source/identity audit, append-only research record and owned cleanup."""
import html,json,re,zipfile,shutil,hashlib,sys
from pathlib import Path
import numpy as np
import b36_switch14_design as D
from b36_semantic14_memo import append
M=D.M;S=D.S


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=[]
    with np.load(S.OUT/'native-heatmaps.npz') as z:
        for site in D.SITES:
            arrays=[z[site+'_'+name] for name in ['first2','first3','paired_difference']]
            for view in ['raw','asinh']:
                vv=arrays if view=='raw' else [np.arcsinh(20*x/np.maximum(z[site+'_rms'][:,None],1e-12)) for x in arrays]
                lim=max(float(np.abs(x).max()) for x in vv);fig,axs=plt.subplots(3,1,figsize=(25.6,8),sharex=True,layout='constrained')
                for ax,x,name in zip(axs,vv,['First2','First3','Paired First3-First2']):
                    im=ax.imshow(x,aspect='auto',origin='lower',interpolation='nearest',cmap='RdBu_r',vmin=-lim,vmax=lim,extent=[-.5,5119.5,-.5,40.5]);ax.set_ylabel(name+'\nResidual boundary')
                axs[-1].set_xlabel('Native coordinate (all5120; no sorting or TopK)');fig.colorbar(im,ax=axs,label='BF16 values, averaged as float32' if view=='raw' else 'asinh(20 * mean / pooled layer RMS)')
                fig.suptitle(site+' | '+view+' | observed paired responses, not causal edges');name='field-'+site+'-'+view+'.png';fig.savefig(S.OUT/name,dpi=140);plt.close(fig);result.append(name)
    return result


def behavior_figure(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=['step_count_correct','steps_correct','value_correct','joint'];x=np.arange(4)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for j,(cell,color,label) in enumerate(zip(data['cells'],['#277da8','#e08b31'],['Execute first 2 updates','Execute first 3 updates'])):
        values=[cell[k] for k in keys];bars=ax.bar(x+(j-.5)*.34,values,.34,label=label,color=color)
        ax.bar_label(bars,labels=[str(v)+'/96' for v in values],padding=3,fontsize=10)
    ax.set_xticks(x,['Printed step count','Printed step values','Final Value','Joint success']);ax.set_ylim(0,108);ax.set_ylabel('Correct paired views (96 families per instruction)');ax.legend(frameon=False)
    ax.spines[['top','right']].set_visible(False);ax.set_title('One instruction-token change; identical programs and output fields')
    fig.savefig(S.OUT/'behavior-results.png',dpi=160);plt.close(fig);return ['behavior-results.png']


def run():
    S.configure(64*2**20);assert not (S.OUT/'memo-append-receipt.json').exists()
    calibration=S.read(S.OUT/'calibration.json');cp=S.read(S.OUT/'calibration-plan.json');M.admit(cp)
    plan=S.read(S.OUT/'plan.json') if (S.OUT/'plan.json').exists() else None
    behavior=S.read(S.OUT/'behavior-analysis.json') if (S.OUT/'behavior-analysis.json').exists() else None
    native=S.read(S.OUT/'native-summary.json') if (S.OUT/'native-summary.json').exists() else None
    analysis=S.read(S.OUT/'native-analysis.json') if (S.OUT/'native-analysis.json').exists() else None
    review=S.read(S.OUT/'output-review.json') if (S.OUT/'output-review.json').exists() else None
    assert S.read(S.OUT/'calibration-independent-audit.json')['passed']
    assert S.read(S.OUT/'independent-score-audit.json')['passed']
    if calibration['passed']:assert behavior is not None,'Complete the new-family behavior study before closeout'
    if behavior and behavior['passed']:assert native is not None,'Complete the authorized native path study before closeout'
    if native and native['passed']:assert analysis and analysis['admitted']
    if plan:M.admit(plan)
    if behavior:
        for name,h in behavior['source_hashes'].items():assert S.file_sha(S.OUT/(name+'.json'))==h
    if native:
        for r in native['records']:assert S.file_sha(S.OUT/r['path'])==r['sha256']
    if analysis:assert analysis['source_sha256']==S.file_sha(S.ROOT/'tests/glm5/b36_switch14_analyze.py')
    if review:assert review['source_sha256']==S.file_sha(S.ROOT/'tests/glm5/b36_switch14_review.py')
    plots=(behavior_figure(behavior) if behavior else [])+(figures() if analysis else [])
    sources=set((S.ROOT/'tests/glm5').glob('b36_switch14*.py'))|{S.ROOT/'tests/glm5/test_b36_switch14.py',S.ROOT/'tests/glm5/b36_semantic14_memo.py',S.ROOT/'tools/provenance.py',S.ROOT/'tools/power_analysis.py',S.ROOT/'deploy/storage_policy.py',S.ROOT/'deploy/storage_policy.json',S.ROOT/'shared/dist_protocol/mechanism_trace.py'}
    sources|={S.ROOT/n for n in cp['source_hashes']}
    if plan:sources|={S.ROOT/n for n in plan['source_hashes']}
    with zipfile.ZipFile(S.OUT/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(sources):
            if path.suffix=='.py':compile(path.read_text(encoding='utf-8-sig'),str(path),'exec')
            z.write(path,str(path.relative_to(S.ROOT)).replace('\\','/'))
    with zipfile.ZipFile(S.OUT/'source.zip') as z:
        assert z.testzip() is None
        for path in sources:assert z.read(str(path.relative_to(S.ROOT)).replace('\\','/'))==path.read_bytes()
    progress=[json.loads(x) for x in (S.OUT/'progress.jsonl').read_text(encoding='utf8').splitlines() if x.strip()]
    stats=dict(cuda_peak_bytes=max(r.get('cuda_peak',0) for r in progress),shared_gpu_sample_max_bytes=max(r.get('shared_gpu_bytes',0) for r in progress),
        dedicated_gpu_sample_max_bytes=max(r.get('dedicated_gpu_bytes',0) for r in progress),system_commit_sample_max_bytes=max(r.get('commit_used_bytes',0) for r in progress),ram_available_sample_min_bytes=min(r['ram_available_bytes'] for r in progress),
        calibration_seconds=calibration['seconds'],behavior_generation_seconds=sum(S.read(S.OUT/(arm+'.json'))['seconds'] for arm in D.ARMS),native_execution_seconds=0,
        free_bytes=shutil.disk_usage(S.ROOT).free)
    caltext=f"从同一次标准HF生成中，在已生成8个token、缓存含前7个生成token时保存CPU缓存副本、掩码、位置和待处理token，再继续生成。原样回放逐token一致{calibration.get('exact',0)}/96，首次续写全词表logits逐位一致={calibration.get('first_logits_bitwise',False)}，CPU缓存哈希未变={calibration.get('cache_immutable',False)}。冻结门{'通过' if calibration['passed'] else '失败'}。96题均为上一阶段已暴露材料，只计校准，不计独立确认；原生成与回放的所有token、首次logits和逐层缓存哈希保留，完整多GiB缓存仅在两分支比较期间驻留内存，结束后释放。"
    headline=caltext
    if behavior:headline+=' 新家族联合成功为'+str({c['arm']:str(c['joint'])+'/96' for c in behavior['cells']})+'；行为门'+('通过。' if behavior['passed'] else '失败，停止机制确认；B36及后续路径本轮未验证。')
    if native:headline+=' 同缓存原生路径数值准入'+('通过。' if native['passed'] else '失败；产物不能作确认机制证据。')
    table='';native_table='';comparisons='六项冻结原生主比较未执行，未产生p值；计划功效不代表已取得这些干预样本。';examples='';effect='没有新家族机制证据。'
    if behavior:
        table='条件 | N | 步数正确 | 步骤全对 | Value正确 | 格式 | EOS | 联合\n--- | --- | --- | --- | --- | --- | --- | ---\n'+'\n'.join(' | '.join(str(c[k]) for k in ['arm','n','step_count_correct','steps_correct','value_correct','format_correct','eos','joint']) for c in behavior['cells'])
        examples='\n\n'.join('`'+r['id']+'`：`'+r['text'].replace('\n',' / ')+'`；'+str(r['parsed']['step_pattern']) for r in behavior['failures'][:8]) or '本轮冻结行为口径下未观察到失败；有限样本全对不能证明总体无错误。'
    if analysis:
        native_table='接口 | 位置 | 条件 | N | 类别正确 | 首选token变化 | 目标logP平均变化 | 平均KL\n--- | --- | --- | --- | --- | --- | --- | ---\n'+'\n'.join(' | '.join(str(c[k]) if k not in ['target_logp_delta_mean','kl_mean'] else f'{c[k]:.7g}' for k in ['interface','site','condition','n','correct','changed_token','target_logp_delta_mean','kl_mean']) for c in analysis['cells'])
        comparisons='\n\n'.join(f"{r['left']}→{r['right']}：N=96，新增正确{r['gained']}、丢失正确{r['lost']}，差{r['delta']:.6f}，精确p={r['p_exact']:.7g}，Holm p={r['p_holm']:.7g}，描述性家族95%区间{r['family95']}；{r['affine_pair_clusters']}组双终态仿射函数聚类敏感性区间{r['affine_pair_cluster95']}。" for r in analysis['primary'])
        changes={arm:sum(c['changed_token'] for c in analysis['cells'] if c['condition']==arm) for arm in plan['native_conditions']}
        effect=f"六位置联合类别正确的家族数为{analysis['family_joint_correct']}；各条件改变首选token次数（576个条件视图，并非576个独立家族）为{changes}。自然首选与第二名间隔{analysis['baseline_top2_gap']}。这些数字分别描述正确性和响应变化；不论显著与否，都不能自行证明唯一计算模块。"
    unified=r'''$$\widetilde H_l=H_l+A_l(N_l(H_l),KV_l,\mathrm{position}),\quad H_{l+1}=\widetilde H_l+M_l(N'_l(\widetilde H_l)),\quad p_{t+1}=\mathrm{softmax}(W_U N_f(H_L)_{t}).$$'''
    body=f'''**任务、冻结范围与实际状态。** 承接Phase2933，先修复无编辑历史回放一致性，再研究继续计算、停止计算、报告数值的转换，定位固定B36单元与后续路径。运行{S.RUN}。{headline}

**校准原理与证据。** {caltext} 相同字符串重新编码与相同生成状态并不相同：本轮比较的是相同前缀x、缓存K_t、位置p_t、掩码A_t及待处理token在相同批次形状下的确定性续写Fθ(x,K_t,p_t,A_t)。等价性是本运行器/快照/批次下的测量准入，不是语言机制发现，也不能倒过来替上一阶段的失配结果补过。

**新家族材料与控制。** {'本轮已运行96个新家族，每家族同一组三条可见运算，只把指令执行前2条改为执行前3条，192个自然生成视图。所有输入113token；逐对仅一个指令token不同，长度、程序及输出字段完全一致。每题只输出Steps与Value，没有选项表。全三操作序列、实际前二操作序列和初始值+首操作前缀排除登记旧材料；不声称预训练未见。初始30–49、三次真值均为10–99且互异，有限材料分布不能代表自然语言全域，也不能把与前阶段初始10–29的差异作因果比较。第三条操作在first2条件未执行但仍可见，始终纳入家族身份，执行ops与visible_ops分别保存。' if plan else '缓存校准尚未通过，本轮没有执行新家族模型测试；已编写的新材料与路径入口仅为后续准备。'}

外部真值定义为s₀=x，s_k=T_k(s_(k−1))，要求执行d∈{{2,3}}条，报告v=s_d。T_k是实际加、减、乘操作，d是输入中唯一改变的指令数字。这些是任务定义，不是神经语言规律。正确打印中间值不保证内部推理忠实；提示已含d，因此后续输出要求对早期状态的影响不是未来token逆向影响过去。

**行为与读出准入。** first2与first3分别至少88/96联合成功（步骤内容、Value、格式、EOS均合格）；192条全部拥有3个可定位读出才进入机制确认，不按答对与否筛选。after_first在首个打印整数后，下一个逗号/换行token之前；after_second在第二个打印整数后的对应位置，d=2应换行、d=3应逗号；report在Value首个数字token之前。定位使用自然分支下一token判断边界，但该token绝不输入受干预分支；因此属于事后定位的自然路径条件实验，不是可预知未来的停算检测器。分隔符之前的空格保留。类别是token解码文本以逗号、换行或相应数字开头，不等同于整个后续数字或完整答案。

{table}

**失败、反例与缺件。** {examples}

完整可见错误账本（描述性，不新增确认检验）：{review['cells'] if review else '未执行新家族输出审查'}。报告首数字类别正确、但完整Value错误的数量：{review.get('first_digit_correct_despite_wrong_full_value') if review else '不适用'}。与被要求忽略的第三步结果相同不证明内部执行了它；第三步与重复末操作的数值可能重合，相关标记允许重叠。

两种指令下完整输出token完全相同的配对数为{review.get('identical_full_token_sequences_across_instructions') if review else '未执行'}。本轮first3的54个失败均只打印两个正确的前缀状态，其中17个Value仍正确、37个Value停留在第二步值；first2有3个步骤正确而Value等于被要求忽略的第三步值。这些可见反例说明完整打印步骤与正确终答不能互相替代，不能用它们断言内部实际执行了几次运算。该指令比较配平了输入长度、可见程序与输出字段，却没有把所需运算深度、范围限制表述及输出计划的作用完全分开；不能将本次first3失败概括为模型普遍不会三步计算。行为配对描述：{behavior.get('paired_descriptive') if behavior else '未执行'}，不是六项原生机制主检验。

缺失读出：{behavior.get('missing_sites') if behavior else '尚未执行新家族生成'}。原行为门不因缓存校准成功而放宽；若门失败，未执行内部实验不记为零效应。

**拟定内部对象、公式与路径。** 本轮因行为门失败没有执行以下神经干预，相关代码仅经CPU辅助测试，未完成真实模型分支验收。B36指零起始block索引36；固定32单元和预先匹配32单元来自旧冻结候选，本轮不据新结果选层、挑坐标或调强度。z=RMSNorm(r)，G=W_gate z，V=W_up z，a=SiLU(G)⊙V，MLP=W_down a；a是MLP中间单元，不是残差坐标或权重本身。拟定单元集合J的消融为a'_j=0(j∈J)，增强为a'_j=2a_j(j∈J)；实数理想写入差Δr=Σ_(j∈J)(a'_j−a_j)W_down[:,j]，实际BF16须重新投影并保留舍入，因此不要求浮点结果满足精确实数加法。拟定G/V/a原生乘积逐位核对属于模型计算审计，不是新理论，也不是本轮已执行结果。

实际原生状态：{native['status'] if native else '未执行'}。{'第二次自然生成仍由原标准HF调用完成，在同一次活前向的B36输入处取状态及B36–39各层的原有KV，固定batch96、当前位置、掩码与旋转位置嵌入，复制到独立单步分支。仅当前token施加消融/增强，过去KV与原始生成都不编辑。另在消融分支中分别恢复B37–39自然attention、MLP或两者的当前位置输出；这是一种混合路径干预，既不证明唯一中介，也不等于自由生成修改后的历史。每个事件要求无编辑分支全96条完整词表logits逐位一致，所有192条重跑自然序列要求逐token复现行为阶段；缓存副本哈希不变。全部41个当前位置残差边界、完整G/V/a、真实64列down参数、各分支完整词表logits及B36之后原生状态保留。logits用与基线异或后ZIP无损压缩，可逐位还原。' if native else '没有新增通过本轮准入的内部图谱节点或关联机制因果边。'}

{effect}

实际B36残差位移核对：{review.get('actual_residual_displacements') if review else '未执行'}。位移为零时不能把输出不变当作有效扰动下的负结果；位移非零而输出不变也只限定本次范围与强度。

{native_table}

**主要检验。** 每个家族的六个读出（三位置×两种指令）全对才记联合类别正确；96家族是独立性近似单位，不能把576个视图当成独立N。设新增正确g、丢失正确l、D=g+l，精确McNemar p=min(1,2Σ_(k=0)^min(g,l) C(D,k)/2^D)，D=0时p=1。六项比较使用Holm校正，所有失败或未准入项不作确认。规划20个百分点效应、30%不一致率、α=.05/6，96家族；小效应功效不足。连续评分、KL与逐位置分解均为描述，不追加显著性检验。家族bootstrap是描述性95%区间；零不一致时退化到[0,0]，不能证明总体等价。以执行前二和前三操作的双仿射终态函数聚类只作敏感性检查，不能保证完全消除相关性。

{comparisons}

**可视化与身份基线。** 本轮图表仅展示真实行为计数，没有采集新家族原生HiddenState全场，也没有绘制其热力图。预定原生视图须保留全部坐标、不排序、不删背景，不以TopK或PCA定义主干，并分别保留原值及明确标准化的视图；这些绘图代码本轮未验收为机制结果。相同token的嵌入身份与中间层上下文作用仍须分开，相同坐标索引跨层不直接代表同一功能，平均图也不能单凭相似提升为因果边。

**严格审视与理论增量。** 无编辑分支一致性解决测量可比性，单步消融定位当前响应参与，二者都不等于找到了语义首次形成的位置。即使停算指令只有一个token不同，它仍在整个前缀计算中影响多层状态；不能把末端B36局部作用等同于完整停止策略。这里的停止首先指停止列出状态，不能直接判断内部是否继续运算。目标是单token类别而非完整数值或后续自由生成，报告首数字正确不代表Value全对。固定匹配组不能自动排除写入范数差；实际残差位移范数单独报告。某比较显著、另一比较不显著，不能单凭这一点认定两种效应存在显著差异。没有广泛无关任务损伤测试、同预算SAE/transcoder对比、跨模型确认、独立节点复算或留出状态预测器。本轮只跑本地Qwen3-14B原始BF16、CUDA逐层装载、SDPA、禁用thinking；不量化，不并行装载4B或GLM4。所有真实结果按准入范围保留，失败不删除，数值小或不显著不代表作用不存在。RDC主体及全局图谱公式未改，不发布新统一语言数学理论或AGI结论。

**核心拼图与统一框架的接续。** 与本轮直接相关的旧拼图保留原证据等级，不在本轮重新升级：Phase2694–2698区分词族可读性、真实MLP来源、条件/位置失效与next-token读出；Phase2896提供Qwen14原生单元普查及可复用采集入口；Phase2928保留局部传播预测、ABC接口组干预、XYZ反例和B39八根净恢复近零的边界；Phase2930–2932保留语义求值、输出映射及多打印一步的行为范围；Phase2933保留明确计数接口过门、历史回放92/96未准入和Code读出正确率无翻转的结果。本轮只按实际测量增加同缓存一致性、单指令token配对材料及通过准入的当前决策局部路径证据；不把这些拼成已闭合的通用语义算法。

沿用Phase2698列出的可核查统一计算接口（显示省略浮点余项，实际舍入单独计入），这里H_l是原生残差状态，A_l/M_l是原模型attention/MLP，N_l/N'_l/N_f是相应归一化，KV_l是已有缓存，W_U是输出权重：

{unified}

这是原模型计算定义，而非新提取的语言理论。RDC的类型化部分转移仍是带适用域的候选框架；全局图谱主线仍为“语言程序与词嵌入初态→自然状态与历史→条件响应→类型化部分转移→尚未闭合的输出竞争”。外部图谱本轮增加配对停止指令和真实错误；内部图谱只增加实际采集的原生对象；关联图谱只按数值/行为准入后的范围内干预与反例记账，未登记正式中心，也没有新增经证实的全局图谱闭合公式。

**第一性原理与下一大阶段。** 有限参数的组合能力需要把任务条件d、状态更新T、终态保持和符号读出连接成能预测的转移规则；必须同时跟踪神经层l和生成时间t的h_(l,t)，以及过去状态对当前位置的缓存影响。下一大阶段围绕“执行范围绑定—打印计划—终态报告”联合研究：①保留本轮全部失败，在独立发现材料上正交控制仅执行前d条/执行全部、可见操作数与实际忽略条数、输出行数和明确Steps项数，检验first3为何经常只列两步，不把单纯换提示后成功当作旧协议过门；发现后冻结新家族，当前96家族已暴露，不能再作为改版盲确认。②将正确终态与被忽略后续运算结果作为明确竞争项，完整多token Value及自由续写单独验收；先检验行为和读出可用性，再进入原生路径。③在同一冻结留出材料上比较原生坐标/单元、简单词汇与位置基线和同预算稀疏特征，预测继续/停止条件下的全场及输出，而不是只重跑原模型；④在已校准同缓存分支上，区分题目来源、历史来源、当前位置及更早生成时刻，测完整自由续写与局部单步效应，并检验后续头/MLP的中介和副作用。按预测残差与反例扩展三图谱，只在未见程序、未见句式及有限内部改动预测都成立时保留可复用机制规则。未执行部分只作计划，不计本轮进展。

**工程、资源与交付。** 五项CPU工程检查通过；全部192条新材料的输入token、可见程序独立执行真值、输出token解码、评分和读出重新核对；独立splitlines/int评分器再次核对192条，96对均只改变零起始第8个输入token。另独立CPU检查96条缓存回放前缀、待处理token、掩码导出的位置、40层缓存形状和完整logits数组。完整KV字节未长期保存，运行结束后的CPU审计不能重新哈希已释放的完整缓存；哈希不变是运行时核对结果。标准HF源码核查表明本次未显式传入旧缓存的自然generate调用会创建新的DynamicCache；没有另用独立模型实现复算全部192条。科研源码与运行器SHA256冻结并归档。未改共享协议、正式中心或生产客户端，因此未运行整套dist_protocol回归；没有发布或登记正式中心。日志逐token换行刷新并记录GPU共享显存，CUDA分配峰值{stats['cuda_peak_bytes']/2**30:.3f}GiB，共享显存采样最大{stats['shared_gpu_sample_max_bytes']/2**20:.1f}MiB，不冒称连续共享峰值；驱动专用显存采样最大{stats['dedicated_gpu_sample_max_bytes']/2**30:.3f}GiB与CUDA已分配量分别记账。校准耗时{stats['calibration_seconds']:.1f}秒，两组新家族生成累计{stats['behavior_generation_seconds']:.1f}秒。系统提交内存含其他进程，不视为本任务独占用量，完整采样统计见delivery.json。全部新增数据D盘，按storage_policy解析路径并保留20GiB；交付前空闲{stats['free_bytes']/2**30:.2f}GiB。预定原生批次的RAM压缩及磁盘写入方案未进入真实执行。只清理本轮独占临时目录，释放字节见cleanup.json；所有原始输出、失败、冻结计划、缓存哈希、复算依据及源码保留。HTML仅作静态内容/链接检查，图表另作人工图像检查，不声称桌面/手机浏览器验收。

入口tests/glm5/b36_switch14*.py及test_b36_switch14.py；实际执行范围由plan、calibration、behavior-analysis、native-summary和analysis文件共同决定，存在脚本不表示已执行。命令采用.venv/Scripts/python.exe -B -X utf8 -u；calibrate.py prepare/run，工程test，design.py冻结，behavior.py run，paths.py，analyze.py，finish.py。结果目录tests/glm5/result/b36_switch14_20261114_v1。记录只尾部追加并核对旧内容字节前缀不变。

通俗说明：先确认把模型暂停后原样接着算，它会写出完全相同的内容。再给同一道三步题，只改“算前两步”或“算前三步”，观察第二步之后它怎样选择继续还是停下。最后只动固定的一小组单元，看当前选择是否改变；这能检验局部作用，还不能把整个推理能力归于这组单元。
'''
    title='同缓存回放通过与条件化停算行为门失败：步骤列举和终态报告分离'
    r1=append(S.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md',title,body)
    r2=append(S.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md','B36同缓存回放与停算转换验收索引',f"详见GLM5 Phase {r1['phase']}，运行{S.RUN}。{headline} 具体效应、失败、数学对象、资源和下一大阶段见tests/glm5/result/b36_switch14_20261114_v1/report.html。未完成项不作研究结论，无正式中心登记或发布。")
    S.save('memo-append-receipt.json',dict(records=[r1,r2]))
    # Plain HTML uses the exact record content and images; no Markdown renderer dependency.
    text='<!doctype html><html lang="zh"><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>body{max-width:1200px;margin:40px auto;padding:0 24px;background:#101722;color:#e5e9f2;font:16px/1.75 system-ui}a{color:#83cfff}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{width:100%;background:white}h1{font-size:28px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #40516c;text-align:left}summary{cursor:pointer;font-size:20px}</style><h1>'+html.escape(title)+'</h1><p>'+html.escape(headline)+'</p>'
    if behavior:
        text+='<h2>当前结果</h2><table><tr><th>条件</th><th>步数正确</th><th>步骤内容正确</th><th>最终数值正确</th><th>联合成功</th></tr>'
        for c in behavior['cells']:text+='<tr><td>'+('只执行前两步' if c['arm']=='first2' else '执行前三步')+'</td>'+''.join('<td>'+str(c[k])+'/96</td>' for k in ['step_count_correct','steps_correct','value_correct','joint'])+'</tr>'
        text+='</table><p><strong>核心发现：</strong>3题步骤正确而终值错误；另17题少写一步但终值正确。打印步骤与最终答案不能互相替代。两种指令有39对完整输出完全相同。</p><p><strong>证据边界：</strong>缓存回放已修复；行为门仅一组通过，B36干预未启动，尚未定位其在停算与报告中的作用。</p><p><strong>下一步：</strong>在新的发现与确认家族上，联合控制执行范围、被忽略的语句和输出项数，再检验终态保持与完整答案路径。</p>'
    text+=''.join('<p><a href="'+name+'">'+name+'</a></p><img src="'+name+'" alt="'+name+'">' for name in plots)
    text+='<details><summary>完整研究记录、公式与限制</summary><pre>'+html.escape(body)+'</pre></details>'
    links=['calibration-plan.json','calibration.json','cache-original.json','cache-replay.json','source.zip']+(['plan.json','behavior-analysis.json'] if plan else [])+(['native-summary.json','native-analysis.json'] if analysis else [])
    text+='<p>'+' · '.join('<a href="'+n+'">'+n+'</a>' for n in links if (S.OUT/n).exists())+'</p></html>'
    (S.OUT/'report.html').write_text(text,encoding='utf8')
    for rel in re.findall(r'(?:src|href)="([^"]+)"',text):assert (S.OUT/rel).is_file(),rel
    for r in [r1,r2]:assert hashlib.sha256((S.ROOT/r['path']).read_bytes()[:r['original_bytes']]).hexdigest()==r['original_sha256']
    scratch=S.TEMP.resolve();assert scratch==(S.ROOT/'tests/glm5_temp/b36_switch14_20261114_v1').resolve() and scratch.drive.upper()=='D:' and scratch.is_relative_to(S.ROOT.resolve())
    assert not S.TEMP.is_symlink() and not S.TEMP.is_junction()
    paths=list(scratch.rglob('*'));assert all(not p.is_symlink() and not p.is_junction() and p.resolve().is_relative_to(scratch) for p in [scratch]+paths)
    released=sum(p.stat().st_size for p in paths if p.is_file());shutil.rmtree(scratch)
    S.save('cleanup.json',dict(path=str(scratch),released_bytes=released,scope='Only this run scratch; all raw evidence and frozen artifacts retained.'))
    S.save('delivery.json',dict(resource=stats,phase=r1['phase'],source_sha256=S.file_sha(__file__),source_archive_sha256=S.file_sha(S.OUT/'source.zip'),calibration_passed=calibration['passed'],behavior_passed=behavior['passed'] if behavior else None,native_passed=native['passed'] if native else None,plots=plots))
    files=[dict(path=str(p.relative_to(S.OUT)).replace('\\','/'),bytes=p.stat().st_size,sha256=S.file_sha(p)) for p in sorted(S.OUT.rglob('*')) if p.is_file() and p.name!='manifest.json']
    S.save('manifest.json',dict(files=files))
    print('Delivered Phase',r1['phase'],len(files),'artifacts; own scratch removed',released,'bytes',flush=True)


if __name__=='__main__':run()
