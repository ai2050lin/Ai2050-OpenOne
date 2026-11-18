"""Reproducible source bundle, explicit execution status and append-only journal."""
import argparse,hashlib,json,re,shutil,sys,zipfile
from pathlib import Path
from datetime import datetime
sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as D

def bundle():
    root=D.OUT;assert D.read(root/'audit.json')['passed'];paths=sorted((D.ROOT/'tests/glm5').glob('source_binding*.py'))
    paths+=sorted((D.ROOT/'tests/glm5').glob('test_source_binding*.py'))
    paths+=[D.ROOT/'shared/dist_protocol/grammars/SOURCE-BINDING-20261110.json',D.ROOT/'shared/dist_protocol/mechanism_trace.py',
        D.ROOT/'shared/dist_protocol/natural_register_runner.py',D.ROOT/'deploy/storage_policy.py',D.ROOT/'deploy/storage_policy.json']
    for name in ['plan.json','internal-code-freeze.json','native-v2-code-freeze.json','causal-admission-v2.json','conditional-update-addendum.json','runtime-environment.json']:paths.append(root/name)
    archive=root/'sources.zip';assert not archive.exists();D.check_write(archive,8<<20,operation='来源绑定冻结源代码归档')
    manifest={str(p.relative_to(D.ROOT)):D.sha(p) for p in paths}
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.relative_to(D.ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        for name,h in manifest.items():assert hashlib.sha256(z.read(name.replace('\\','/'))).hexdigest()==h
    D.save('source-bundle.json',{'archive_sha256':D.sha(archive),'files':manifest,'purpose':'retain exact frozen and recovery sources without copying model weights'})
    D.save('execution-status.json',{'qwen14':{'unique_material_rows':2048,'internal_rows':2048,'primary_splits':['discovery','confirm','lockbox_frame'],
        'capture_runner':'source_binding_recover.py run qwen14 <split> -> native_v2','causal_runner':'source_binding_causal_v2.py',
        'causal_MLP_status':'skipped before execution due failed frozen norm matching','conditional_update_status':'exploratory after Qwen component holdouts opened'},
        'glm9':{'pilot_rows':128,'pilot_passed':D.read(root/'glm9/pilot-gate.json')['passed'],'internal_rows':0,
            'status':'pilot behavior failed; no candidate extraction, no internal collection, no heldout mechanism replication',
            'supplement':'natural generation of same128 already exposed pilot rows, behavior-only; gate remains failed'},
        'native_v2_freeze_sha256':D.sha(root/'native-v2-code-freeze.json'),'causal_v2_freeze_sha256':D.sha(root/'causal-admission-v2.json'),
        'formal_evidence_promotion':0,'review_native_retention_by':'2026-12-10'})

def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in rows)

def memo_text(phase,stamp):
    root=D.OUT;q=root/'qwen14';g=root/'glm9';report=D.read(root/'report.json');audit=D.read(root/'audit.json')
    plan=D.read(root/'plan.json');candidate=D.read(q/'candidates.json');glm=D.read(g/'pilot-gate.json');diag=D.read(g/'pilot-output-diagnostic.json')
    predictions={s:D.read(q/f'{s}-prediction.json') for s in ['confirm','lockbox_frame']}
    components=[]
    for i,x in enumerate(predictions['confirm']['candidates']):
        key='selected32' if x['kind']=='head' else 'selected256';random='random32' if x['kind']=='head' else 'random256';y=predictions['lockbox_frame']['candidates'][i]
        components.append([x['id'],f"{x['methods'][key]['improvement']:.2%}",f"{y['methods'][key]['improvement']:.2%}",f"{x['methods'][random]['improvement']:.2%} / {y['methods'][random]['improvement']:.2%}",
            f"{x['methods'][key].get('binding_record_accuracy',0):.2%} / {y['methods'][key].get('binding_record_accuracy',0):.2%}" if x['kind']=='head' else '不适用'])
    causal=[]
    for arm in ['correct_edge','wrong_edge','sparse_qk','random_qk']:
        a=D.read(q/'confirm-causal.json')['summary'][arm];b=D.read(q/'lockbox_frame-causal.json')['summary'][arm]
        causal.append([arm,f"{a['forced_accuracy']:.2%} / {b['forced_accuracy']:.2%}",f"{a['gold_margin_change']:.3f} / {b['gold_margin_change']:.3f}",f"{a['kl_mean']:.6f} / {b['kl_mean']:.6f}"])
    rollout=[]
    for split in ['confirm','lockbox_frame']:
        s=D.read(q/f'{split}-conditional-update.json')['steps'][-1]['methods'];o=D.read(q/f'{split}-conditional-output.json')['summary']
        for name in ['layer_mean','h32','h16_prefix16']:rollout.append([split,name,f"{s[name]['rollout_whole_state_relative_error']:.2%}",f"{o[name]['KL_mean']:.3f}",f"{o[name]['top1_agreement']:.2%}",f"{o[name]['forced_accuracy']:.2%}"])
    body=INTRO.format(phase=phase,stamp=stamp,qwen_sha=plan['models']['qwen14'],glm_sha=plan['models']['glm9'])
    body+=METHODS
    body+='\n\n**局部组件的冻结留出结果。** 以下改善率为相对发现侧均值预测的平方误差降低，不是全模型破解比例；所有原生输出坐标均参与计算。\n\n'
    body+=table(['候选（零基索引）','confirm 改善','新句式改善','随机同规模改善（两个面板）','所选QK记录命中（两个面板）'],components)
    body+=QWEN_ANALYSIS
    body+='\n\n'+table(['原生干预','四选一正确率（confirm / 新句式）','正确词margin变化','KL(原模型‖干预)'],causal)
    body+=CAUSAL_ANALYSIS
    body+='\n\n**完整自然输出诊断。** 三个Qwen面板各取冻结的前8家族/64条，贪心最多4个新token；分别有59/64、57/64、64/64严格只输出代码词，全部正常结束。前两个面板的5条和7条不合规均多了句号，原代码词正确。未因诊断改判原首token行为门；这也说明当前门不等于严格格式或一般多token生成门。\n'
    body+=f'\n**GLM条件复现的停止结果。** Qwen的两个面板满足原方案的复制准入后，顺序运行GLM发现侧前16家族/128条试跑。四选一{glm["forced_accuracy"]:.2%}、全词表首token{glm["top1_accuracy"]:.2%}；S0E1O0的top1为81.25%，S1E0O1的强制/首token为93.75%/87.5%，S1E1O1为87.5%/87.5%，句式1强制为93.75%、句式2首token81.25%，故完整逐格门未通过。没有采集GLM内部场、选择GLM候选或进入确认材料。\n'
    body+=f'\n仅为区分绑定与输出格式，随后在同一批已暴露128条上做最多8token的自然输出诊断：严格代码词输出{diag["exact_code_only"]:.2%}，忽略大小写及末尾句号的事后描述为{diag["case_and_terminal_period_normalized"]:.2%}，正常结束{diag["terminated_fraction"]:.2%}。该宽松描述不覆盖原门，也不提供机制复现。具体输出及反例保存在`glm9/pilot-output-diagnostic.json`。'
    body+=GLM_LIMITS
    body+='\n\n**条件更新与递推诊断。** Qwen局部留出已查看后新增并冻结此诊断，明确作为探索；其代码在GLM试跑前固定，但因GLM门失败没有在GLM执行。起点只提供一次真实hs27，随后使用自身预测状态递推至原始hs40。三方法的结果为：\n\n'
    body+=table(['面板','方法','整状态相对Frobenius误差','全词表KL','模型top1一致','四选一正确率'],rollout)
    body+=ROLLOUT_ANALYSIS
    body+=LIMITATIONS
    body+=NEXT_PHASE
    resources=report['resources']
    body+=f'\n\n**执行、审计、清理和交付。** 11项工程单测通过；另用随机初始化的微型Qwen/GLM架构检查原生QKV、SwiGLU、编辑局部性、零操作与撤销，均通过，均只属工程材料。最终审计{audit["count"]}项通过并重哈希所有Qwen原生数组。专用CUDA峰值{resources["max_cuda_peak_GiB"]:.2f}GiB；DXGI进程非本地显存峰值{resources["max_shared_GPU_MiB"]:.1f}MiB，包含驱动驻留，不能全部解释为溢出。GPU进程已记录的累计运行约{resources["GPU_process_logged_wall_minutes"]:.1f}分钟；所有模型按顺序执行。字段、冻结代码、候选、预测、干预、输出、图、特征卡及审计在`tests/glm5/result/source_binding_v1/`，源代码归档`sources.zip`和`source-bundle.json`可复原本次实现。\n'
    body+=CLOSEOUT
    return body

INTRO='''

## Phase {phase}: 来源绑定的原生QK/VO与MLP提取、条件更新和跨模型准入边界 [{stamp}]

**目标、版本与结论范围。** 接续Phase2905/Claude392，在用户要求继续完成来源绑定与条件更新大阶段的授权下执行`SOURCE-BINDING-20261110-v1`。2026-11-10 05:14冻结材料及行为协议，05:25冻结内部提取、预测和干预实现；后续存储恢复、失配对照停步和探索性递推均另留版本，未覆盖原冻结源码。主模型Qwen3-14B原BF16，模型/tokenizer快照`{qwen_sha}`；GLM-4-9B原BF16快照`{glm_sha}`，只在Qwen预设准入成立后串行试跑。RTX5080 CUDA执行原HF块，CPU逐层读取权重；没有量化或答案方向注入。

本轮得到Qwen指定任务内的三个可压缩头写入候选、两个MLP原生单元组的预测结果及头组范围内干预；GLM试跑失败，未形成跨模型机制复现。末token简化递推也未闭合。正式证据自动晋级数为0；下面的公式分别是原架构计算式、提取算法、拟合预测器或度量定义，均不是新发现的普适语言定律。

**材料、家族与行为先行。** 新建两份冲突档案Orion/Vega，同一对虚构实体在两档案中各有不同代码词。每家族交叉查询来源2×查询实体2×档案顺序2=8条；实体行序随家族与档案交替。发现64家族/512条，确认96家族/768条，新句式96家族/768条；三个划分的128/192/192个实体名字互不重合。发现使用16个代码词，留出使用另8个词，两留出面板共用这8词但实体不同。每面板4个句式，新句式同时改变记录与提问形式。名字和词汇“未见”只指本实验发现拟合边界，不对预训练暴露作主张。

材料为本项目构造的受控自然语言输入，结果均来自真实本地模型。生成文法保存在`shared/dist_protocol/grammars/SOURCE-BINDING-20261110.json`并通过现有validate；实际执行的是显式冻结家族清单，不是通用runner的整个笛卡尔展开，也未冒充标准分布式审计分片。两个模型分别冻结chat模板、完整输入、token IDs、字符/词槽跨度及单token答案。所有来源/实体区分信息均在最终查询读出之前；档案中有四个候选值，答案选择需要绑定来源与实体。

行为门要求每个S×E×O格和每个句式n≥16、四选一≥95%、全词表首token≥90%，不剔除实体或错误行。每个完整split先独立跑完behavior，落盘哈希绑定的gate，再由admission重算、核对后才导入内部采集模块和创建数组。Qwen发现集两指标100%；confirm四选一100%、首token99.87%；新句式两者100%，逐格均通过。全部行进入既定内部分析，没有按答案正确筛选。行为试跑128条只是发现集子集，不增加样本。文本解析器从真实文本同时解析档案和实体，三个面板均100%；只取最后记录25%，只识别来源或实体分别50%。这些基线说明任务结构清楚，不意味着模型的高行为准确率本身就是机制发现。

''' 
METHODS=r'''
**采集对象与数值校准。** 保留embedding和40个原始decoder输出边界，在两档案标题末、查询来源末、查询实体末和最终提示位置采集全部原生坐标，最终RMSNorm后状态另存。发现侧记录40块的完整前缀K/V、末查询Q、全部头context/attention，以及全部MLP G/V/A与输入输出；留出保存全hidden场和冻结候选块的详细组件。原生值以BF16的uint16位模式保存，注意力复算和区域概率为FP32诊断量；没有以PCA或Top-K定义全场。

原生SwiGLU为

$$A_{l,j}=operatorname{SiLU}(G_{l,j})V_{l,j},qquad m_l=W_l^{down}A_l.$$

逐批核对A与真实down_proj输入逐比特一致，并核对Down重放。QK复算含实际Q/K归一化、RoPE、GQA复用和因果/填充掩码，末查询的全128维attention×V与真实o_proj输入相对误差须小于2%。采集后的自然输出与此前行为输出top1完全一致、四类logits最大差0。另从存储的原始输入状态重新执行原参数投影，核对Q/K/V及G/V/A；两个留出面板均在2%容差内，confirm最大约0.81%。独立矩阵批形状下的BF16舍入差与原同形状逐比特重放分开记录。

**原生提取算法。** 发现侧在块8–35，按正确记录在四记录注意力质量中的占比、再按绝对质量排序，最多每块取一头，共4头；不使用目标logits选头。冻结为B28H21、B27H3、B30H13、B29H10。每头在所有有效前缀位置上计算

$$E_j=mathbb E_{i,t}[(q_{i,j}k_{i,t,j})^2],quad J=operatorname{Top32}(E).$$

Top32仅定义受检验的局部点积近似，不取代全坐标观察。候选预测为

$$hatalpha_t=operatorname{softmax}_t!left(rac{sum_{jin J}q_jk_{t,j}}{sqrt{128}}ight),qquad

hat d=W^O_hsum_thatalpha_tv_t.$$

softmax仍覆盖整个有效前缀；比较同为32项的固定种子随机坐标、发现侧平均head context和原128项校准。这里输入的是当前层真实Q/K/V，不能称为从文本预测Q/K、独立推断答案或完整模型压缩。Qwen的Q/K RMSNorm分母依赖全部128维，RoPE又混合配对坐标，32个点积项不等于只执行32行权重。

MLP按发现侧$sum_joperatorname{Var}(A_j)|W^{down}_{:,j}|^2$选择2块，再取每块最大的256项，得到B39/B38，采用

$$hat m=ar m+sum_{jin S}W^{down}_{:,j}(A_j-ar A_j).$$

均值仅来自发现侧，原Down列固定；比较随机256单元和预设范数匹配组。该算法有原生参数来源，仍依赖同层真实归一化输入及其G/V计算；不把读出部分A后重建写入包装成已预测了A的形成。两种方法都用全部输出坐标计算$R=1-sum_i|y_i-hat y_i|^2/sum_i|y_i-ar y_D|^2$。冻结门为R≥0.10，头还要求正确记录质量argmax命中≥0.65；家族bootstrap2000次另给误差改善区间，点估计门与区间分别记账。
'''
QWEN_ANALYSIS='''

B28H21/B30H13/B29H10和两组MLP在两个面板均通过预设局部预测门。B27H3明确失败：选对记录近乎100%，却比均值基线更难重建写入。发现侧该头正确记录的平均质量从全点积0.656降到32项的0.334，仍常为四记录中最大。这说明记录排序、整前缀归一化概率与写入内容不能混为一个指标。原128项近乎完整重现只是架构代数校准，不计为新提取规律。

两MLP组的发现侧范数匹配比仅0.399/0.568，不满足事先10%容差。另冻结`causal-admission-v2.json`，在任何留出干预前跳过MLP selected/matched两臂；保留其预测结果，未用这些失配对照声称MLP语义特异性或必要性。失败限定于本次固定匹配方法，不证明任何可能的匹配控制都不存在。

**原模型干预。** 仅在最终查询位置，四个冻结头联合执行正确记录attention边排除、另一来源同实体记录排除、保留所选32个QK点积项或保留随机32项。其余位置与头保持原计算，下游原模型真实重算；没有给隐藏态加答案均值差，也没有把任务改成要求模型输出人为目标词。正确/错误来源记录由任务结构指定，这是范围内干预定位，不是模型自己预测干预位置。零操作与撤销校准、自然基线输出重放均通过。
'''
CAUSAL_ANALYSIS='''

两个面板的正确边排除分别降低四选一25.52和21.74个百分点，按家族重抽样的95%区间分别约[−28.52,−22.92]、[−24.22,−19.27]个百分点。仍有约75%–78%原答案保留，不支持四头为唯一完整路径。错误来源对照匹配记录角色/范围，但并未匹配写入范数：confirm每头局部W_O写入变化范数均值约30.66与1.33，相差很大，不能据这组对照独立证明精确语义特异性。

所选32项保留答案且全词表KL很小，但随机32项也保留99.87%四选一正确；所以“答案保持”不能证明这32项特别必要。随机项的局部写入误差明显较大，却没有同比例的输出损害，说明本任务有足够余量、其他计算路径或下游补偿；当前实验没有区分这些替代解释。confirm中所选/随机局部写入变化范数约8.30/29.33，剂量也不同。正确边排除与随机项的pooled范数接近仅为事后描述，未成为预注册逐条匹配对照。
'''
GLM_LIMITS='''

三条强制选择失败中，两条为同档案另一实体的值，一条为同实体另一档案的值。例如要求Vega中Lubuvo的cat，却选择Tegivi的stone；要求Vega中Mifoki的metal，却选择Orion中的fish。自然输出确认9条严格失败：3条错值、2条只回复实体名、3条带实体标签和句号、1条仅大小写不同，全部正常终止。该诊断只用于辨别这些现象，不能把已失败的口径替换成更宽松口径后宣布过门。

尤其要看到试跑的局限：每因子格只有16条，95%阈值在该样本数下实际上要求16/16；存在明显离散性和较高误停概率。总体97.66%不能覆盖失败格，失败也不能被解释为GLM没有来源绑定能力或必定采用不同机制。下一版本应先以足够家族数校准句式和完整答案口径，再冻结新的材料与准入；本轮没有扩大GLM或换提示补过门。
'''
ROLLOUT_ANALYSIS=r'''

逐层预测器均只在发现侧拟合。h32用当前末状态的32维固定随机投影；混合方法在四个候选头所在块用16维末状态投影+16维前缀K/V投影，其余块与h32相同；二者每输出坐标均为32个系数加截距，ridge参数固定1。前缀投影保留逻辑位置的固定随机权重，严格排除末查询自己的K/V与填充位；排除规则通过植入工程检查。目标状态、未来token、答案标签不作预测输入。

$$hat h_{l+1}=hat h_l+f_l(hat h_l,mathcal K_{l,<t},mathcal V_{l,<t}),qquad hat h_{27}=h_{27}.$$

原前缀各层K/V可以独立于最后查询计算，但本诊断仍承担原模型前缀预计算成本，不是全场或文本到输出的自主模型。每步喂真实末状态的one-step结果与使用自身预测的递推结果均保存。confirm的h32递推相对均值增量的优势在预测hs36时消失；新句式在hs29已消失。混合摘要没有稳定改善：最终相对h32的平方误差收益confirm为−7683（家族区间约[−10421,−5109]），新句式−1986（约[−5214,+1185]）。真实原始hs40重放logits与行为记录完全一致；简化模型却全部top1一致率为0，四选一近机会水平。

这个负结果不证明“前缀无用”或末token必然不含所需信息。16维随机KV摘要会丢失内容，32维线性模型和发现/留出词汇分布变化也会造成失败。它否定的是当前具体压缩与线性更新方案，不能升级为不存在有限状态闭合的定理。局部QK/VO近似使用原生V和原权重，信息与计算预算远大于这个低维全状态预测器，二者不能直接合并成单一解释率。
'''
LIMITATIONS='''

**严格审视、硬伤和未解决问题。**

1. 外部覆盖仅为两来源、两实体、四代码的受控检索；没有本轮的开放知识、一般语法、长逻辑链、自然语料或多步关系组合确认。固定档案名没有留出；句式与实体行序的家族奇偶规则存在混杂，4个句式也不能当作大量独立构式。代码词分布有限；家族bootstrap只对应当前词池和句式范围。96家族的最坏伯努利标准误约0.051只是敏感性描述，不是已证明各机制阈值具有充分功效。
2. 原生坐标、MLP单元与标量参数已能追溯，但没有独立干预单个标量参数，没有为每个单元赋予语义名称。MLP选择分数没有完整处理单元间协方差；大方差和大Down列可能偏向末层输出变化，不能直接等同于来源绑定功能。字典学习/SAE/transcoder对比本轮未执行。
3. QK坐标是后归一化、后RoPE索引，跨层同号索引和跨模型索引没有功能同一性保证。每候选只用一个冻结随机坐标/单元集合，没有对随机集合分布做充分重复。随机项与所选项能量不同，不能只据删除项数相同断言因果剂量匹配。
4. 四头联合干预没有定位每一头的独立必要性、上游Q/K形成过程或单参数意义，也没有完整救援、无关任务保持或长期自然生成结果。正确与错误记录控制的扰动范数差距很大；全词表KL测量当前任务分布扰动，不代替一般能力保持。模型在别处存在替代路径的解释尚未排除。
5. 原生计算恒等式与完整128维重放只证明采集/索引正确。局部32项或256单元的留出预测确有增量，但仍消费原模型当前层状态。真正从外部语言关系预测内部Q/K、G/V的形成，以及自主生成的整体保真，仍缺证据。探索性递推的明确失败正限制了“闭合”的说法。
6. Qwen确认与新句式面板现已开封。任何由本轮反例重新选头、坐标、文法或阈值的候选，都不能再用它们包装成新盲测。GLM试跑也已暴露；新版本须登记暴露范围。两个模型虽有独立快照，使用的是同一受控材料设计，不能把它们当作独立语言分布复制。

**三图谱、RDC与智能理论增量。** 外部图谱补入同实体跨来源冲突绑定的冻结族、句式及失败样本；内部图谱补入全坐标背景、完整QK/V与G/V/A，及六张含原生索引、典型上下文、残差反例、留出指标和干预范围的特征卡；关联图谱补入限定留出写入预测和头组范围内因果作用。图上的观察关系、原生计算来源、预测、干预与失败分别记账，不能由邻近布局或高亮自动升为因果边。

RDC名称与主体保留，本轮没有新增已证AGI公式。第一性原理上的实质线索是：选择“哪一条关系记录”和传递“该记录的内容”必须分别建模；B27H3选对却写不准、随机QK写入误差大却保留答案，表明来源排序、归一化竞争、内容写入和最终决策有不同保真要求。Transformer的共享参数在不同条件下可以产生可复用的局部算子，但这次任务并未证明无限语言组合来自某个已找到的统一最小齿轮。
'''
NEXT_PHASE='''

**下一阶段的大任务：解释查询—键的形成，并把路由与原生内容保留接入可预测的更新模型。**

1. **重建行为与功效协议。** 用新的家族和档案名，增加来源数、实体数、记录顺序、距离、关系类型及组合深度；先用发现侧足够家族数校准完整答案和格式，控制小格门槛的离散性。对GLM区分绑定错误、复述实体、大小写和句号，而不是只调整一个阈值让当前试跑通过。正式新确认材料在候选冻结前保持隔离。
2. **从“已有Q/K可压缩”推进到“Q/K如何产生”。** 在候选周围追踪原生输入归一化、Q/K行与可见源位置，比较来源、实体、位置的加法与交互模型，先预测未见来源名/实体组合的点积分布、全前缀softmax质量和VO写入，再做局部关闭与恢复。使用同信息预算的文本解析、位置、简单状态预测和适用字典/电路基线；不把正确记录argmax代替连续写入预测。
3. **修复因果控制和MLP解释。** 在发现侧预先建立按实际移除写入范数、层、位置和影响范围匹配的头/单元对照，记录匹配失败；不要沿用本轮失配集合或只匹配数量。围绕自然G/V符号与Down来源检验条件化单元组，做独立组与个体层面的关闭/恢复及其他任务损害检查；明确保留协方差、替代路径与负结果。
4. **建立保留内容的条件更新模型。** 优先保留原生V/内容通道，提取来源与实体控制变量，比较末状态、带类型的源位置状态以及前缀记忆的方案。共享规则和层特异规则须在明确的参数、输入、存储与计算预算下比较，报告一步、递推、多token的状态误差和全词表KL，以及首次误差扩大的层。当前16维随机摘要失败不应驱动盲目扩大PCA/压缩；需要先确定它丢失了哪类必要内容，再设计可证伪的增量。

这些任务应在一个共同冻结问题下推进“形成—选择—写入—读出”的预测链，完成新确认和跨模型准入后再讨论更一般的数学结构，不以增加任务量、图节点数或干预成功率代替破解进度。
'''
CLOSEOUT='''

运行环境为Python3.14.6、PyTorch2.13.0+cu130、Transformers5.14.1、NumPy2.5.1；实际Qwen/GLM架构和SDPA库源码哈希在`runtime-environment.json`。共享显存监测改用实际[DXGI QueryVideoMemoryInfo](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/nf-dxgi1_4-idxgiadapter3-queryvideomemoryinfo)，不再把不可用计数器记成0。

首版发现侧采集在19200/20480个“行×块”进度附近触及2GiB可用主存保护后退出；模型CUDA峰值约1.73GiB、共享段约82MiB，未发生GPU OOM。原因是跨批长期持有完整memmap映射。保留首版源码、冻结、日志、各部分数组哈希和已完成样本后，另建native_v2，仅在每128行批次间flush并关闭/重开映射；恢复后主存保持约10GiB以上，保存样本逐比特一致。旧未完成数组13,271,779,200字节在恢复前已按归属核对后删除；科学阈值、输入和计算保持不变。

关键命令为`source_binding_design.py prepare`，`source_binding_run.py behavior --model ... --split ...`，`source_binding_recover.py run ...`，`source_binding_analysis.py fit/evaluate/replication`，`source_binding_causal_v2.py`，`source_binding_generation.py`，`source_binding_parameter_audit.py`，`source_binding_update.py prereg/fit/evaluate/output`，`source_binding_failed_pilot.py`，`source_binding_report.py`、`source_binding_cards.py`、`source_binding_audit.py`和`source_binding_closeout.py`。均使用项目`.venv\\Scripts\\python.exe -B`，长任务加`-u`，逐批日志含已完成/总数、用时、ETA、专用与共享显存。代码全部在`tests/glm5/`；阶段原生场和结果在`tests/glm5/result/source_binding_v1/`，临时材料只在`tests/glm5_temp/`。

图包括原尺度全坐标图、另存的log(1+RMS)视图与逐坐标标准化视图、所有头的记录质量图、留出组件误差、原生干预和递推失败图，坐标未重排，未以投影替换原场。已目视检查布局与标注。原生场有后续Q/K形成、原参数来源和反例复算用途，保留至2026-12-10复查，而不是长期默认永久保留。没有修改客户端、中心服务或部署；此处结论不等同于正式证据登记。
'''

def memo():
    assert D.read(D.OUT/'audit.json')['passed'] and (D.OUT/'source-bundle.json').exists()
    assert not (D.OUT/'memo-append-started.json').exists(),'Append already started; inspect before recovery'
    main=D.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';claude=D.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md'
    before=main.read_bytes();cbefore=claude.read_bytes();stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    phase=max(map(int,re.findall(r'^## Phase (\d+):',before.decode('utf-8-sig'),re.M)))+1
    cphase=max(map(int,re.findall(r'^## Phase (\d+):',cbefore.decode('utf-8-sig'),re.M)))+1
    body=memo_text(phase,stamp)
    # Only task-owned, reproducible local cache and completed temporary checkpoint.
    cleaned=[];tempbase=(D.ROOT/'tests/glm5_temp').resolve()
    for name in ['source_binding_mpl','source_binding_checkpoint.txt']:
        path=tempbase/name
        if not path.exists():continue
        actual=path.resolve();assert actual.is_relative_to(tempbase) and actual.name==name
        size=sum(p.stat().st_size for p in actual.rglob('*') if p.is_file()) if actual.is_dir() else actual.stat().st_size
        if actual.is_dir():shutil.rmtree(actual)
        else:actual.unlink()
        cleaned.append({'path':str(actual),'bytes':size})
    D.save('cleanup.json',{'removed':cleaned,'bytes_freed':sum(x['bytes'] for x in cleaned),'earlier_failed_partial_bytes_freed':13271779200,
        'preserved':'all valid native evidence, failures and completed sample, frozen sources, model files, unrelated tasks'})
    size=sum(p.stat().st_size for p in D.OUT.rglob('*') if p.is_file())
    body+=f'\n\n收尾仅清理本轮matplotlib字体缓存与已完成临时检查点，释放{sum(x["bytes"] for x in cleaned):,}字节；有效产物约{size/2**30:.2f}GiB，在32GiB预算内。其他任务文件、模型及原生证据未清理。追加后逐字节核验两个研究记录的原有前缀保持不变。\n'
    cross=f'\n\n## Phase {cphase}: 来源绑定的原生组件预测与条件更新边界 [{stamp}]\n\n完整原理、公式、结果、严格审视和下一阶段任务见`research/glm5/docs/AGI_GLM5_MEMO.md` Phase {phase}。运行`{D.RUN}`完成Qwen3-14B BF16的256家族/2048条新受控材料，两个768条留出均在内部采集前通过逐格行为门。三头的32项QK/VO近似在两个面板改善局部写入误差76.6%–97.7%，B27H3选对记录却写入预测失败。两个256单元MLP组改善40.7%–51.3%，但范数匹配对照失败，因果臂在执行前停止。联合删除正确来源记录边使正确率降至74.48%/78.26%；随机32项也几乎保留全部答案，不能认定唯一或必要坐标。\n\nQwen完整生成抽查为59/64、57/64、64/64严格合规，前两面板有多余句号。GLM-4-9B条件试跑125/128四选一正确、119/128首token正确，逐格门失败，未做内部采集或机制复现；同128条自然输出仅作失败诊断。探索性条件递推即使加入16维前缀KV摘要仍未闭合，模型top1一致率0%，全部状态输入与暴露边界已说明。原生字段、六张特征卡、图、冻结代码归档及全文件哈希审计在`tests/glm5/result/source_binding_v1/`。首次memmap驻留触及主存保护后以存储修订版恢复，保留失败证据且复算逐比特一致。没有自动晋级正式证据或部署。下一阶段重点为Q/K来源—实体控制的形成、保留原生内容的更新模型、合格因果控制与新的独立确认。\n'
    D.save('memo-append-started.json',{'phase':phase,'claude_phase':cphase,'timestamp':stamp,'main_before_sha256':hashlib.sha256(before).hexdigest(),
        'claude_before_sha256':hashlib.sha256(cbefore).hexdigest(),'main_append_sha256':hashlib.sha256(body.encode()).hexdigest()})
    with main.open('ab') as f:f.write(body.encode('utf-8'))
    with claude.open('ab') as f:f.write(cross.encode('utf-8'))
    after=main.read_bytes();cafter=claude.read_bytes();assert after[:len(before)]==before and cafter[:len(cbefore)]==cbefore
    assert after[len(before):]==body.encode() and cafter[len(cbefore):]==cross.encode()
    line=before.count(b'\n')+body[:body.index('## Phase')].count('\n')+1
    D.save('closeout.json',{'complete':True,'phase':phase,'claude_phase':cphase,'timestamp':stamp,'main_memo':str(main),'main_line':line,
        'prefix_bytes_preserved':True,'main_after_sha256':D.sha(main),'claude_after_sha256':D.sha(claude),'audit_sha256':D.sha(D.OUT/'audit.json'),
        'source_bundle_sha256':D.sha(D.OUT/'sources.zip'),'artifact_bytes_before_memo':size,'formal_evidence_promotion':0})
    print(json.dumps({'phase':phase,'claude_phase':cphase,'line':line,'timestamp':stamp},ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['bundle','memo']);a=ap.parse_args();globals()[a.stage]()
