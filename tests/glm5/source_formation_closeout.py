"""Append-only research closeout after real artifacts, figures and audits complete."""
import argparse,hashlib,json,re,sys,zipfile
from datetime import datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import source_formation_design as D

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def pct(x):return f'{100*x:.2f}%'
def num(x):return f'{x:.4f}'
def ci(x):return '['+', '.join(num(v) for v in x)+']'

def archive():
    files=set((D.ROOT/'tests/glm5').glob('source_formation*.py'))
    files.update((D.ROOT/'tests/glm5').glob('test_source_formation*.py'))
    for p in ['tests/glm5/source_binding_design.py','tests/glm5/source_binding_runtime.py','tests/glm5/source_binding_native_v2.py','tests/glm5/source_binding_resources.py',
              'shared/dist_protocol/mechanism_trace.py','shared/dist_protocol/natural_register_runner.py','shared/dist_protocol/storage_policy.py','deploy/storage_policy.json']:
        path=D.ROOT/p
        if path.exists():files.add(path)
    for p in D.OUT.glob('*.json'):files.add(p)
    path=D.OUT/'source-and-protocol.zip';assert not path.exists();D.check_write(path,8*2**20,operation='Reproduction source bundle')
    entries={}
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            name=p.relative_to(D.ROOT).as_posix();z.write(p,name);entries[name]=D.sha(p)
    with zipfile.ZipFile(path) as z:
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in entries.items())
    D.save('source-archive.json',{'sha256':D.sha(path),'files':entries,'scope':'Exact scientific, orchestration, reporting and audit sources plus frozen root protocols. Original pre-index-fix archive separately preserved.'})

def main():
    r=D.read(D.OUT/'report.json');audit=D.read(D.OUT/'audit.json');assert audit['passed']
    assert D.read(D.OUT/'supplement-complete.json')['completed'];D.read(D.OUT/'figure-manifest.json');D.read(D.OUT/'feature-cards.json')
    archive();models=r['models'];lines=[]
    memo=D.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';before=memo.read_bytes();text=before.decode('utf-8-sig')
    phase=max(map(int,re.findall(r'^## Phase (\d+):',text,re.M)))+1;stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    lines.append(f'\n\n## Phase {phase}: 早层Q/K形成、保留内容的条件递推与匹配干预——局部推进和闭合失败 [{stamp}]\n')
    lines.append('本轮核心结果（操作性过门与完整机制结论分开）：\n')
    for model,data in models.items():
        outcome=data['primary_outcome'];names=', '.join(outcome['common_passing_heads']) or '无'
        lines.append(f"- {model}：两份必需内部留出面板齐备={outcome['both_required_panels_completed']}；共同通过冻结局部写入门的头={names}。\n")
        for split,p in data['panels'].items():
            if 'update-output' in p:
                a=p['update-output']['summary'];routed=a['routed_content'];state=a['state128']
                lines.append(f"  {split}：路由内容/纯状态递推的六选一={pct(routed['forced_accuracy'])}/{pct(state['forced_accuracy'])}，路由递推与原模型全词表首选一致率={pct(routed['top1_agreement'])}。\n")
    lines.append(r'''### 任务、证据范围与一句话结论

承接 Phase 2906，继续检验三个问题：后续头的 Q/K 能否由较早状态预测；保留内容 V 是否改善连续更新；强度匹配后，来源路由与所选 MLP 单元是否仍表现出差异。主测试使用本机 Qwen3-14B 原生 BF16、CUDA；达到冻结准入条件后才串行测试本机 GLM4-9B。没有改动正式中心、没有发布或提交代码，也没有把本轮结果自动登记为正式机制证据。

本轮推进的是**受控三来源、双实体、单词值检索中的局部机制预测**。完整语言编码机制、跨任务语法/知识网络及自主输出闭合仍未完成。3072 条是每个模型的冻结材料总量，实际执行量以下表为准；材料是自行构造的受控句子，前向和干预是真实本地模型计算，不是 synthetic runner 的工程数据。

### 冻结材料、切分与计算边界

- 发现 64 个家族×12查询=768条；确认与新句式各96家族×12=1152条。每个家族包含3来源×2实体×2来源呈现顺序。来源名、实体名跨三个划分完全不重合，并排除了上一轮已用名字；发现/留出值词分开，名字和值词的“未见”只相对于本轮拟合，不代表模型预训练或项目历史未见。
- 来源排列与实体排列独立随机化；每个面板4句式，远近条件与 code/tag 表述交叉。query 从文本可恢复来源/实体信息，读取发生在完整查询之后；固定 `Answer:\n` 助手前缀、一次最多4个生成token，遇EOS终止。完整答案归一化仅去首尾空白、末尾句点和大小写；严格格式准确率另记。
- 冻结行为门：逐主条件格至少64条，六选一与归一化且终止的生成准确率各≥90%，合并各≥95%。这是预定操作性点估计门，不是置信下界过门；家族自助法2000次，不能将同家族12查询视为独立重复；所报95%区间为逐比较描述性区间，未作全阶段多重比较校正。
- 按每模型行为门先行。通过后保存原生全坐标：6位置×41边界（含 hs[0]）、完整前缀 hs20、块27–30的全部头 Q/K/V、注意力与写入、块38–39的 SwiGLU G/V/A 与 Down 输入输出。BF16按uint16原位保存；读取、标准化与图像从原场派生。跨层同坐标不自动同功能。
- 发现侧在固定块27、28、29、30各选一个头，按正确记录质量/六记录质量选；块窗口来自上一轮，不是无先验全模型发现。MLP每块选256个 `Var(A_j)||Down[:,j]||²` 最大单元，并取同块256个不重合随机单元。

### 数学对象与算法（定义、恒等式、拟合模型分别记账）

**拟合预测器**：令 \(h_{20,t}\) 为第20块之前、位置t的原生残差状态。发现侧按标准化输入与四头Q或K全坐标的协方差平方和选128个原生输入坐标；对照为128随机坐标及128 embedding坐标。三个方法同样用标准化岭回归（\(\lambda=1\)，截距不惩罚）：

$$
\widehat q_{b,t}=a^q_b+B^q_b\,z(h_{20,t}[I_q]),\qquad
\widehat k_{b,s}=a^k_b+B^k_b\,z(h_{20,s}[I_k]).
$$

\(I_q,I_k\)、均值/尺度与系数均只由发现侧确定；K拟合每句最多24个均匀位置。Q/K对象是原 QK-norm 之后、RoPE之前的全部头坐标。该预测器的系数是学习得到的回归参数，不能冒充从原生权重直接推导出的语言定律。

**原生注意力计算来源**：用模型实际RoPE算子 \(R_t,R_s\)，保持BF16旋转边界，随后

$$
\widehat p_s=\operatorname{softmax}_s((R_t\widehat q)^\top R_s\widehat k_s/\sqrt d),\qquad
\widehat w=W_O^{(b,h)}\sum_s\widehat p_s\,v_s.
$$

\(v_s\) 和 \(W_O^{(b,h)}\) 是原模型该层实际V及该头的原生O列块。**局部写入预测仍依赖真实当层V**；它不是从文本独立执行的替代模型。全真Q/K/V复算仅作数值校准。均匀token、发现侧32位置密度、六记录等质量均分三种基线共享同一V/WO，防止把保留V本身误计成路由预测的全部收益。

**误差改善率的定义**：对于一个留出面板H，

$$
R=1-\frac{\sum_{i\in H}\|w_i-\widehat w_i\|_2^2}
{\sum_{i\in H}\|w_i-\overline w_D\|_2^2}.
$$

\(\overline w_D\) 为发现侧组件均值。R可以为负；它是相对均值的局部平方误差下降，不是“解释了整个模型的百分比”。冻结主门为每个头在两留出上各满足R≥0.1且优于embedding；至少同一个头通过才准入GLM行为复查。均匀、位置等附加对照不改主门。

**原生标量参数来源（局部代数，不是早层预测）**：另一支从真实当层完整RMSNorm输入 \(x\) 中，按 \(\operatorname{Var}_D(x_j)\|W_{:,j}\|^2\) 选128坐标：

$$
\widetilde u=W\overline x_D+b+W_{:,I}(x_I-\overline x_{D,I}),\qquad
\widetilde q\text{或}\widetilde k=N_{Q/K}(\widetilde u).
$$

W、b、Q/K归一化均来自原生参数；每个Q或K块保留128×128=16384个标量连接，完整输入仍用于RMSNorm，未选输入的发现均值贡献保留。 Q在末查询位置评估，K在五个已存的非末位置评估；K均值基线按这五个位置分别估计，而输入均值在这些位置合并估计。它与主预测器在完整前缀上的K误差指标覆盖不同，不能直接混为一项。此分解定位参数来源，不能称为只读128原始坐标的独立预测，更不能称为单参数因果证明。

**条件递推拟合模型**：从真实 \(h_{20,t}\) 起步，后续末位置状态只输入自身预测值；在块27–30加回预测路由的历史V写入，并拟合其余增量：

$$
\widehat h_{b+1,t}=\widehat h_{b,t}+f_b(\widehat h_{b,t}[J_b])+\widehat w_{b,<t}.
$$

\(f_b\) 是发现侧128坐标岭回归；非目标块无额外写入。历史V严格排除末查询自己的V并重归一化，但其余位置真实未来层V仍是外供记忆。路由组比state128多用了Q/K预测器、原WO及真实V，信息/算力预算不等。循环移动六记录V段的对照同时移动其语境信息，不是只改答案词。最后用原norm/unembedding比较全词表KL、首选token一致率与六选一，不能仅按隐藏状态误差宣称闭合。

**干预**：正确记录排除 vs 另一来源同实体记录排除；所选MLP组 vs 随机组抑制。仅编辑末位置，重算真实下游，参数不改。每步两分支当前状态上计算完整扰动范数，目标为二者较小值的0.5；18步二分寻找BF16实际扰动，不放大。冻结匹配门为99%扰动在各自目标5%内。没有删除失败样本。头四块共同编辑、MLP两块共同编辑，效应不能分摊成单头/单元必要性。

**误差恒等式与诊断**：旋转后的向量记Q、K，预测误差为δQ、δK，则

$$
\delta s_s=(\delta Q^\top K_s+Q^\top\delta K_s+\delta Q^\top\delta K_s)/\sqrt d.
$$

逐句对有效token中心化三项，移除softmax不敏感的公共偏移；保存完整3×3二阶矩，包括正负交叉项。此为代数误差账本，不是三个因果贡献比例。用真Q+预测K、预测Q+真K定位瓶颈：Qwen确认侧为看过主结果后的诊断，同一配方在新句式及GLM结果之前冻结。它们使用了真实当层信息，不计入可执行早层预测成绩。
''')
    lines.append('\n模型与tokenizer联合快照：\n'+table(['本机模型','快照SHA256'],[[str(D.MODELS[m]),x['model_snapshot']] for m,x in models.items()]))
    lines.append('\n本轮是自定义家族文法的本地实验，使用共享资源租约/快照工具，但不是dist_protocol v3正式分片，不能冒充该协议的正式合并或独立节点复算。六个候选值都已出现在提示词中，本任务测来源与实体的联合检索，不检验模型新增世界知识。\n')
    lines.append('### 实际行为门与失败样本\n')
    behavior_rows=[]
    for model,data in models.items():
        for split,p in data['panels'].items():
            b=p['behavior'];est=b['estimates'];g=b['gate']
            material=D.read(D.OUT/f'{model}-materials.json')['splits'][split]
            behavior_rows.append([model,split,len(material),pct(est['forced_correct']['accuracy']),pct(est['normalized_correct']['accuracy']),pct(est['strict_correct']['accuracy']),str(g['passed']),json.dumps(b['failure_categories'],ensure_ascii=False)])
    lines.append(table(['模型','面板','N','六选一','完整归一化+终止','严格+终止','行为门','失败分类'],behavior_rows))
    lines.append('\n每条错误的原输入、输出与正确值见 report.json 的 behavior.failures；所有失败都保留在主统计中。GLM是否进入内部测量按实际行为门执行，未采集部分明确记为未检，不能从Qwen外推。\n')
    gb=models['glm9']['panels'].get('lockbox_frame',{}).get('behavior')
    if gb and not gb['gate']['passed']:
        categories=gb['failure_categories'];lines.append(f"\nGLM新句式的{len(gb['failures'])}条完整答案错误中，{categories.get('wrong_entity_same_source',0)}条是在正确来源内选错实体；这不是单纯的输出格式问题。当前头干预的负对照为“另一来源、同一实体”，未检验“同来源、另一实体”的独立对照，因此不能把来源干预结果直接当作这些实体混淆的解释。后续应对来源与实体两条路由轴分别设计对照。\n")
    for model,data in models.items():
        panels=data['panels'];lines.append(f'### {model}：局部预测、参数来源、干预与递推\n')
        rows=[]
        for split,p in panels.items():
            if 'formation' not in p:continue
            for c in p['formation']['results']['candidates']:
                m=c['methods'];rows.append([split,c['id'],num(m['native']['improvement']),num(m['random']['improvement']),num(m['embedding']['improvement']),pct(m['native']['record_accuracy']),num(m['native']['query']['improvement']),num(m['native']['key_improvement']),str(c['passes'])])
        if not rows:
            lines.append('行为门未准入内部实验；无该模型Q/K、MLP、干预或递推结论。\n');continue
        lines.append(table(['面板','头','写入R','随机R','embedding R','预测路由记录准确率','Q向量R','K向量R','冻结主门'],rows))
        lines.append('\n发现侧为拟合内成绩；另外两面板才是本轮冻结后的留出预测。记录路由准确率不等于原模型最终答案准确率。\n')
        held=[(s,p) for s,p in panels.items() if s!='discovery' and 'formation' in p]
        if len(held)==2:
            passed=[{c['id'] for c in p['formation']['results']['candidates'] if c['passes']} for s,p in held]
            lines.append('\n两留出共同通过的头：'+', '.join(sorted(passed[0]&passed[1]))+'。这是预注册点估计准入，完整配对误差置信区间仍应结合查看。\n')
        for split,p in held:
            cs=p['formation']['results']['candidates'];negative=[c['id'] for c in cs if c['methods']['native']['query']['improvement']<0]
            if negative:lines.append(f'\n{split}的Q向量预测劣于发现均值的头：'+', '.join(negative)+'。局部写入R与Q向量R不可互换，不能称为全部Q/K形成已被稳定解释。\n')
        extra=[];causal_rows=[];outputs=[];params=[];pairnotes=[]
        for split,p in panels.items():
            if split=='discovery' or 'formation' not in p:continue
            for c in p['error-split']['results']:
                extra.append([split,c['id'],num(c['methods']['true_Q_pred_K']['improvement']),num(c['methods']['pred_Q_true_K']['improvement']),num(c['query_cosine_mean']),f"{c['algebra_identity_max_abs_error']:.2e}"])
            c=p['causal']
            for kind,pair in c['paired_comparisons'].items():
                causal_rows.append([split,kind,pct(c['matching'][kind]['fraction_within_5pct']),str(pair['frozen_matching_admitted']),num(pair['margin_difference']),ci(pair['margin_difference_ci_family']),pct(pair['accuracy_difference']),num(pair['actual_pair_relative_norm_error_p95']),num(pair['actual_pair_relative_norm_error_max'])])
                failed=pair['matching_failed_rows'];pairnotes.append(f"{split}/{kind}：有至少一次局部匹配失败的查询={failed['n']}条；这些行对全部样本平均margin差的贡献={num(failed['contribution_to_full_sample_mean_margin_difference'])}。所有行保留，这只是数值失败账本，未另设筛选后的估计。")
            fq=c['summary']['forecast_qk'];pairnotes.append(f"{split}：四头预测Q/K替代后的六选一={pct(fq['forced_accuracy'])}，相对基线变化={pct(fq['accuracy_change'])}，margin变化={num(fq['margin_change'])}，全词表KL={num(fq['KL_mean'])}。")
            for method,x in p['update-output']['summary'].items():outputs.append([split,method,pct(x['forced_accuracy']),pct(x['top1_agreement']),num(x['KL_mean'])])
            for c in p['native-parameters']['results']:params.append([split,c['id'],num(c['methods']['selected']['improvement']),num(c['methods']['random']['improvement']),f"{c['methods']['full']['max_relative_error']:.3e}"])
        if extra:
            lines.append('\n**特权输入诊断**\n'+table(['面板','头','真Q+预测K的写入R','预测Q+真K的写入R','Q余弦','恒等式最大误差'],extra))
            lines.append('\n**原生参数局部分解**\n'+table(['面板','Q/K块','128选定坐标R','128随机坐标R','完整原生复算最大相对误差'],params))
            lines.append('\n**匹配干预（正确来源减其他来源；选定MLP减随机MLP）**\n'+table(['面板','组','在5%目标内比例','匹配准入','margin差','家族95%区间','正确率差','两分支范数差P95/目标','最大/目标'],causal_rows))
            lines.append('\n'+'\n\n'.join(pairnotes)+'\n')
            lines.append('匹配失败的MLP行即使置信区间不跨零，也不能支持匹配特异性。匹配通过的头组仍有允许范围内少数失败及很小的正确来源抑制剂量，必须保留最大误差和剂量。无准确率变化不证明组件不参与；小margin变化也不证明必要性。\n')
            lines.append('\n**连续更新与全词表输出**\n'+table(['面板','方法','六选一','原模型首选token一致率','KL(P原||P预测)'],outputs))
            paired=[]
            for split,p in panels.items():
                if 'rollout_comparisons' not in p:continue
                for method,x in p['rollout_comparisons']['comparisons'].items():
                    paired.append([split,method,pct(x['routed_accuracy_gain']),ci(x['accuracy_gain_ci_family']),num(x['routed_KL_reduction']),ci(x['KL_reduction_ci_family'])])
            lines.append('\n路由递推相对各对照的配对差（准确率区间为比例单位，不是百分数）：\n'+table(['面板','对照','准确率增量','家族95%区间','KL下降','家族95%区间'],paired))
        controls=[]
        for split,p in panels.items():
            if split=='discovery' or 'content-controls' not in p:continue
            for c in p['content-controls']['results']:
                controls.append([split,c['id'],*[num(c['methods'][m]['improvement']) for m in ['uniform','position32','equal_records']]])
        if controls:lines.append('\n**同V的简单路由基线**\n'+table(['面板','头','均匀token R','发现位置密度R','六记录等质量R'],controls)+'\n同V基线的逐家族配对误差差及区间见report.json；不只比较点估计。\n')
        relations=[]
        for split,p in panels.items():
            if split=='discovery' or 'formation' not in p:continue
            for c in p['formation']['results']['candidates']:
                a=c['relation']['additive'];b=c['relation']['interaction']
                relations.append([split,c['id'],num(a['conditional_KL']),num(b['conditional_KL']),pct(a['record_accuracy']),pct(b['record_accuracy'])])
        if relations:lines.append('\n**外部关系模型**\n'+table(['面板','头','加法条件KL','交互条件KL','加法记录准确率','交互记录准确率'],relations)+'\n外部特征已含来源匹配与实体匹配，乘积直接标明目标记录；因此记录准确率不能计为新发现的内部算法。与真实六记录条件质量的KL比较才是响应拟合指标，且仍不能代替全前缀注意力/写入预测。\n')
        for split,p in held:
            cs=p['formation']['results']['candidates'];better=sum(c['relation']['interaction']['conditional_KL']<c['relation']['additive']['conditional_KL'] for c in cs)
            lines.append(f'\n{split}上额外来源×实体交互项的条件KL优于加法模型：{better}/{len(cs)}头；未改善的结果作为该附加特征假设的负结果保留。\n')
    qconfirm=models['qwen14']['panels'].get('confirm',{})
    if 'native39-matching-numerics' in qconfirm:
        nd=qconfirm['native39-matching-numerics'];ss=nd['results']['mlp_selected'];lineage=qconfirm['matching-coordinate-lineage']
        actual=qconfirm['causal']['paired_comparisons']['mlp']['matching_by_block_arm']
        lines.append('\n### 事后数值诊断：Qwen MLP匹配失败的位置与刻度\n')
        lines.append('这项诊断是在两个Qwen留出结果已暴露后新增的CPU分析，不改任何原主实验代码、阈值或结论。原GPU记录的分块匹配比例：'+json.dumps(actual,ensure_ascii=False)+'。\n')
        lines.append(f"对确认侧全部{nd['n']}条的未编辑第39块自然状态作算术对照，选定组扰动的最大坐标能量占比平均{pct(ss['largest_coordinate_energy_fraction']['mean'])}；最常见输出行={lineage['modal_dominant_output_row']}，覆盖{pct(lineage['modal_fraction_of_samples'])}样本，属于残差输出坐标而非MLP单元。此行附近BF16相邻刻度平均{num(ss['nearest_output_spacing']['mean'])}。18步和28步搜索的5%匹配比例分别为{pct(ss['bf16_18']['fraction_within_5pct'])}、{pct(ss['bf16_28']['fraction_within_5pct'])}，连续FP32算术为{pct(ss['continuous_fp32']['fraction_within_5pct'])}。\n")
        lines.append(f"实际干预的第39块输入已经受第38块编辑影响，因此自然状态对照不是该分支的精确重放；两者目标范数的最大相对差为{pct(nd['cpu_vs_gpu_target_relative_error_max'])}。该结果支持“输出方向高度集中加上BF16离散刻度是数值瓶颈”的解释，不能补过匹配门，也不能把坐标731命名为语义概念。后续应在发现侧先检查方向支撑、可实现剂量与离散刻度；单纯增加二分次数并未解决这里的问题。原始数组、逐样本索引和源码哈希见matching-numerics、native39-matching-numerics与matching-coordinate-lineage产物。\n")
    lines.append(r'''### 严格审视：成立的增量与仍然存在的硬伤

1. **可重复的局部关系，尚非整体理论。** hs20到若干后续Q/K的预测及原生参数局部分解分别增加了响应边与计算来源边；未证明128坐标是唯一/最小机制，也未证明一个坐标对应一个概念。共同窗口、同一个模型、同一套模板和单节点测量限制了推广。
2. **Q/K均方误差与下游作用不等价。** K整体向量R较好，仍可能在与查询点积、相对token竞争有关的细节上出错。Qwen第一面板真K混合诊断改善明显较大；Qwen新句式上该优势缩小，恢复真Q也有明显改善，而四个Q向量R全部为负。因此不能将瓶颈概括为只在K。这不是键的因果必要性证明，也不保证换一种目标函数即可解决；GLM按自己的结果单独判断。
3. **输出闭合未通过。** 保留真实历史V会改善部分六选一成绩，但全词表首选一致率及KL给出更严厉的失败证据。没有把真实未来末状态喂回递推，却仍使用真实未来前缀V；其中也包括问题与助手前缀的较早token，而不仅是账本原始内容，可能已经携带上下文计算信息。排除最后一个token不等于消除了所有近答案信息；即便成绩更高，也只能称为条件部分模型。
4. **匹配干预的数值门必须独立记账。** BF16扰动存在离散刻度，二分缩放不能保证任意范数可达。失败的MLP配对不升级；头组通过操作门也不等于每个样本完全等范数。修改干预剂量、精度或门槛后需要新版本、新留出，不能在已看结果上补过门。
5. **覆盖仍狭窄。** 只覆盖英文三账本、两实体、短单token值输出；没有自然语料字典/SAE对照，没有跨语言、长推理、多token编辑生成、其他任务副作用或独立节点复算。关系特征由模板标签/可恢复文本关系提供，未建立可推广自然语言解析器。
6. **预注册层级明确。** 主Q/K、更新和干预先于新内部测量冻结；原生参数附加支先于确认内部采集；同V基线先于打开确认机制结果；误差拆分对确认是事后诊断，对新句式是提前固定配方。不得将各支统称成一次完全盲测。已执行面板今后均视为已暴露。

### 智能理论的工作洞察与下一阶段大任务

有限参数支持多种关系组合，可能需要稳定的“内容表示—条件选择—写入更新”运算，而不是为每个句子存储一个独立齿轮。本轮的具体启发是：衡量表示误差，应同时考虑**它在当前查询及竞争记录下怎样改变下一步计算**。这仍是待证候选，不是AGI或人脑机制的结论。

联合行为也不自动要求乘法特征：在本材料每个来源/实体只有一个对应记录的条件下，假设简单分数为“α×来源匹配 + β×实体匹配”，且α、β均为正，则同时匹配的记录得到α+β，高于只匹配一项的记录。这个可计算替代解释已足以产生联合选中，不能因行为像AND便宣称模型内部发现了独立乘法齿轮。本轮加法/交互条件KL对照正是为限制这类过度解释。

在V与WO固定、score扰动足够小的条件下，由原注意力计算可推导一阶局部近似（不是本轮已验证新规律，大误差时还需检验高阶项）：令V按token为行，\(S(p)=\operatorname{diag}(p)-pp^\top\)，则

$$
\delta w\approx W_O V^\top S(p)\,\delta s,\qquad
\|\delta w\|^2\approx\delta s^\top S(p)V W_O^\top W_O V^\top S(p)\delta s.
$$

它指出一个可证伪方向：只降低K的欧氏误差，可能不如降低查询条件下、经过softmax竞争与V/WO后放大的相对分数误差。定义这个条件度量并不等于发现了新数学结构；必须与简单标量点积预测、等预算原生坐标、随机坐标、SAE/低秩表示在同一新留出上比较。

下一大阶段建议围绕“**键的形成与可执行的关系路由**”组织完整闭环，而非再添加一个热力图：

- A：冻结新的来源/实体/句式与更深组合；先校准行为，保留一个从未用于选法的最终面板。将查询条件下的相对score及写入误差列为预定目标，与当前向量岭回归作等预算比较；同时测信息来源和计算成本。
- B：沿hs20到目标头的真实K投影，追踪小但高作用的坐标组合、MLP单元及原生标量连接。区分Q读方向、K承载关系、V保存内容，给出可执行计算与残差；完整坐标视图始终保留，低维表示只作候选。
- C：在发现侧先验证BF16可实现的干预剂量与对照，再冻结可匹配的头/神经元/参数干预，测正确来源、无关来源与其他任务副作用。不能把本轮MLP失败通过改阈值消掉。
- D：先区分账本记录、问题token与助手前缀的V输入收益，再逐级移除特权信息：真V→从较早状态预测V→前缀记忆自身更新；同时评估状态重建、全词表KL、自然生成与更深组合。至少有一条跨句式预测和原模型干预链成立后再扩展规模，跨模型先过行为门。

### 复算、工程异常、清理与证据保留

行为初次热身从64驻留样本改为128，模型微批仍为8，评分与材料不变；原启动/中断记录保留，不计独立样本。原内部分析曾因多维K输入索引写法失败，旧源码完整保存在pre-index-fix-sources.zip，新执行v2仅修索引 `x[...,idx]` 并保持拟合系数身份；不是按留出结果调参。确认行为第一次遇其他任务GPU租约而等待，队列保留共享租约并串行执行，没有结束其他任务。

复算入口：`source_formation_design.py`（冻结/身份）、`source_formation_behavior_v2.py`（行为）、`source_formation_native.py`（一次采集）、`source_formation_analysis.py`（Q/K和关系模型）、`source_formation_update.py`（递推）、`source_formation_causal.py`（真实干预）、三项附加分析脚本、`source_formation_report.py`、`source_formation_audit.py`、`source_formation_artifacts.py`。运行器均有不可覆盖保护；复算须使用新输出目录并核对原冻结哈希，不直接覆盖本轮产物。

所有脚本位于tests/glm5；结果位于tests/glm5/result/source_formation_v1。source-and-protocol.zip保存实际代码/冻结配置，source-archive.json逐文件哈希；原生manifest保留模型、材料与数组哈希。feature-cards.json区分原生索引、拟合预测、参数代数、组干预、反例与未测项；typed-graph.json不同类型的边不互相替代。图中无PCA/Top-K删坐标，原值与固定坐标RMS缩放分开，全数组可重算。

本轮原生全场有键形成后续研究用途，按manifest保留至2026-12-10复审，不为减小目录删关键证据。仅清理本轮可重建绘图库缓存与临时进度文件；清理路径与字节另存cleanup.json。其他任务文件、原始日志、错误记录和全部真实结果保留。技术审计通过仅说明身份、产物和执行口径符合检查，不等于独立科学复现或正式证据晋级。
''')
    res=r['resources']
    lines.append('\n本轮资源实测：CUDA峰值 '+num(res['max_cuda_GiB'])+' GiB；DXGI进程共享显存峰值 '+num(res['max_shared_MiB'])+' MiB；最低可用RAM '+num(res['min_available_RAM_GiB'])+' GiB。逐步时间、已完成/总数、ETA和内存见progress.jsonl；不用多阶段复用PID的最大时长冒充累计GPU时间。\n')
    lines.append(f"\n审计：{audit['count']}项身份/数据/执行检查通过；正式证据自动晋级数=0。工程单元测试15项通过，命令为 `.venv/Scripts/python.exe -B -m unittest discover -s tests/glm5 -p test_source_formation.py -v`，记录见engineering-calibration.json、engineering-tests-index-fix.log及本轮收尾验证日志。\n")
    lines.append('\n通俗说明：可以把这次结果理解为，已经从较早的“阅读状态”预测出部分后续“查哪条记录、取出什么内容”的动作；但把这些动作串起来，还不能重现模型的完整回答。下一步要找清关键关系在“索引键”中如何形成，并逐步减少对真实内部记忆的依赖。\n')
    addition='\n'.join(lines).encode('utf-8')
    assert memo.read_bytes()==before,'Memo changed concurrently; reread rather than overwrite'
    with memo.open('ab') as f:f.write(addition)
    after=memo.read_bytes();assert after[:len(before)]==before and after[len(before):]==addition
    claude=D.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md';cbefore=claude.read_bytes()
    ref=(f'\n\n### 本机Qwen3-14B / GLM4来源形成阶段索引 [{stamp}]\n\n'
         f'本轮完整研究记录已append至 `research/glm5/docs/AGI_GLM5_MEMO.md` Phase {phase}。研究早层Q/K预测、真实参数来源、保留V的条件递推与匹配干预；局部写入预测与完整输出保真分别报告，匹配失败的MLP对照不作特异性结论。真实产物、原生数组manifest、冻结代码、审计与特征卡位于 `tests/glm5/result/source_formation_v1/`。GLM是否准入与实际行为门见report.json。没有发布，也没有自动晋级正式证据。\n').encode('utf-8')
    with claude.open('ab') as f:f.write(ref)
    assert claude.read_bytes()[:len(cbefore)]==cbefore
    D.save('memo-append.json',{'phase':phase,'timestamp':stamp,'main_memo':str(memo),'prefix_bytes':len(before),'prefix_sha256':hashlib.sha256(before).hexdigest(),
        'addition_sha256':hashlib.sha256(addition).hexdigest(),'append_only_verified':True,'claude_prefix_bytes':len(cbefore),'claude_prefix_sha256':hashlib.sha256(cbefore).hexdigest()})
    print('CLOSEOUT_APPENDED_PHASE',phase,flush=True)

if __name__=='__main__':main()
