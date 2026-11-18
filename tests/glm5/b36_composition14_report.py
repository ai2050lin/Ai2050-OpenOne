"""Auditable local scientific report and native-coordinate discovery views."""
import argparse,html,json,time
from pathlib import Path
import numpy as np
import b36_composition14 as B
import b36_composition14_analysis as A


def heatmaps():
    B.configure(50*2**20)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows,meta=A.load('discovery');arrays=[]
    for chunk in meta['chunks']:
        with np.load(B.OUT/'discovery'/chunk['name']/'states.npz') as d:arrays.append(B.unbits(d['hidden']))
    h=np.concatenate(arrays).astype(np.float32);by={}
    for i,r in enumerate(rows):by.setdefault((r['family'],r['sequence'],r['mode']),{})[r['title']]=i
    title_pairs=[(p[1],p[0]) for p in by.values() if set(p)=={0,1}]
    bymode={}
    for i,r in enumerate(rows):
        if r['branch']=='number':bymode.setdefault((r['family'],r['sequence'],r['title']),{})[r['mode']]=i
    mode_pairs=[(p['program'],p['archive']) for p in bymode.values() if set(p)=={'program','archive'}]
    dest=B.OUT/'figures';dest.mkdir(exist_ok=True);views=[];numbers={}
    def export(name,matrix,meaning):
        matrix=np.asarray(matrix,np.float32);limit=max(float(np.max(np.abs(matrix))),1e-8)
        path=dest/(name+'.png');plt.imsave(path,np.repeat(matrix,12 if len(matrix)>10 else 40,axis=0),cmap='RdBu_r',vmin=-limit,vmax=limit)
        numbers[name]=matrix;views.append(dict(file='figures/'+path.name,name=name,shape=list(matrix.shape),range=[-limit,limit],meaning=meaning,coordinate_order='native, unchanged',sorting=False,projection=False,top_k=False))
    for name,pairs in [('source',mode_pairs),('title',title_pairs)]:
        plus,minus=np.array(pairs).T;diff=(h[plus]-h[minus]).mean(0);sd=np.maximum(h.std(0),1e-6)
        for readout in [0,1]:
            export(f'hidden-{name}-r{readout}-raw',diff[:,readout],('number program minus archive' if name=='source' else 'technical minus plain cover')+'; layers0..40, all5120 coordinates')
            export(f'hidden-{name}-r{readout}-standard',diff[:,readout]/sd[:,readout],'Paired mean difference / discovery population coordinate SD; no sorting or coordinate removal')
        for field in 'GVA':
            raw=[];std=[]
            for block in [36,39]:
                x=A.field('discovery',block,field);effect=(x[plus]-x[minus]).mean(0)
                raw.append(effect);std.append(effect/np.maximum(x.std(0),1e-6))
            export(f'mlp-{field}-{name}-raw',raw,'Rows B36 and B39; answer readout; all17408 native MLP units')
            export(f'mlp-{field}-{name}-standard',std,'Same rows; paired mean effect / discovery unit SD')
    B.arrays(dest/'full-coordinate-values.npz',numbers)
    B.save('heatmap-manifest.json',dict(kind='discovery_observation_only',views=views,native_values_sha256=B.file_sha(dest/'full-coordinate-values.npz'),
        source_pairs=len(mode_pairs),title_pairs=len(title_pairs),normalization='Separate raw and standardized views. Linear symmetric full-extrema color scale per image; no quantile clipping. hs0 same readout token is an identity baseline.',
        scope='No claim that similar color patterns imply common semantics or causal connections. B37/B38 full MLP fields were not stored; all41 hidden boundaries retained.'))


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'


def fmt(x):return '—' if x is None else f'{x:.4g}' if isinstance(x,(int,float)) else str(x)


def report():
    B.configure(30*2**20);scope=B.read(B.OUT/'scope-plan.json');behavior=B.read(B.OUT/'full-behavior-confirmation.json')
    state=B.read(B.OUT/'confirm-state-report.json');tail=B.read(B.OUT/'confirm-tail-report.json');out=B.read(B.OUT/'confirm-output-report.json');admit=B.read(B.OUT/'confirm-admissibility.json')
    format_tail=B.read(B.OUT/'format-tail-report.json') if (B.OUT/'format-tail-report.json').exists() else None
    maps=B.read(B.OUT/'heatmap-manifest.json');selection=B.read(B.OUT/'selection.json');allrows=B.read(B.OUT/'materials.json')['rows'];byid={r['id']:r for r in allrows}
    path_report=B.read(B.OUT/'path-output-report.json');paths=path_report['reports']['confirm']
    diagnostic=B.read(B.OUT/'readout-diagnostic.json')
    primary=tail['primary_matched'];causal=tail['counterfactual_prediction'];natural=[r for r in state['results'] if r['target']=='roots' and r['method'] in ['hidden','hidden_prefix']]
    summary=dict(fixed_group=B.group(),admitted_rows=len(admit['eligible_row_ids']),admitted_strata=admit['eligible_strata'],
        full_behavior_joint_passed=behavior['reports']['confirm']['passed'],matched_primary_passed=sum(r['passed'] for r in primary),matched_primary_tested=len(primary),
        local_counterfactual_passed=sum(r['passed'] for r in causal),local_counterfactual_tested=len(causal),state_completion_passed=sum(r['state_completion_passed'] for r in natural),state_completion_tested=len(natural),
        composition_mechanism_status='not_admitted_after_pilot_failure',unique_confirm_families=160,formal_promotion=False)
    B.save('research-summary.json',dict(kind='real_scoped_summary',**summary,source_sha256=B.file_sha(__file__)))
    blocks=[]
    blocks.append('<p class="kicker">Qwen3-14B · 原始 BF16 · CUDA · 固定 B36 32 单元组</p><h1>措辞、必要状态与层间传播</h1><p>本阶段同时检查三件事：旧单元组是否仍影响新题目；补入残差和历史信息能否改善预测；真实权重计算能否解释干预如何向后传播。</p>')
    blocks.append('<p>B0–B39使用零基块号；h37是B36输出残差，h40是最后一块输出、最终RMSNorm之前的残差。MLP单元与残差坐标属于不同空间。</p>')
    blocks.append('<div class="notice"><b>组合机制尚未获得准入。</b> 原两来源联合试跑失败；标签操作与两、三步计算不作为机制正例。新版本只对合格的数值原子计算、数值记录与文本记录做内部确认。独立确认仍保留所有失败条件的行为测试。</div>')
    blocks.append('<div class="stats">'+''.join(f'<div><strong>{v}</strong><span>{k}</span></div>' for k,v in [('确认家族',160),('合格内部输入',summary['admitted_rows']),('匹配组主检验通过',f"{summary['matched_primary_passed']}/{summary['matched_primary_tested']}"),('局部效应预测通过',f"{summary['local_counterfactual_passed']}/{summary['local_counterfactual_tested']}")])+'</div>')
    blocks.append('<p><b>具体结论：</b>主面板ABC中六个记录读取条件的匹配组检验通过；换成XYZ后，同题数值分支出现方向反转，因此作用依赖答案编码。完整原生局部算子解释了本轮有限改动的主要传播，但此前选定的B39八个根并未表现为主要净输出中介。与编码无关的共享语义、最小充分状态和更复杂特征形成仍未建立。</p>')
    blocks.append('<h2>1. 行为门与不通过的组合</h2><p>严格要求正确选项得分大于全部对照；同分算失败。每个正式主条件有80个家族；标题、符号、操作顺序和干预臂均不是新增独立样本。</p>')
    cells=behavior['reports']['confirm']['cells'];blocks.append(table(['分支','步骤','来源','标题','正确/总数','准确率','行为门'],[[r['branch'],r['depth'],r['mode'],r['title'],f"{r['correct']}/{r['n']}",fmt(r['accuracy']),'通过' if r['passed'] else '失败'] for r in cells]))
    blocks.append('<details><summary>原始全词表首选与候选比较</summary><p>这里只核对未编辑输入的下一token首选。干预后的完整续写尚未测试；不能用原始首选结果替代干预后自由生成。</p>'+table(['分支','步骤','来源','N','全词表首选正确','首选在候选中','候选比较正确'],[[r['branch'],r['depth'],r['mode'],r['n'],r['full_vocab_top1_correct'],r['full_vocab_top1_in_choices'],r['forced_correct']] for r in path_report['baseline_next_token'] if r['split']=='confirm'])+'</details>')
    blocks.append('<details><summary>相同输入与选项的操作顺序对照</summary><p>AB/BA 只换顺序，词语多重集相同；ABA/BAB 也改变操作次数，不能称作纯顺序对照。失败试跑条件即使确认行为改善，也不会追认内部准入。</p>'+table(['分支','步骤','序列','N','准确率'],[[r['branch'],r['depth'],r['sequence'],r['n'],fmt(r['accuracy'])] for r in behavior['paired_order']])+'</details>')
    errors=[r for r in behavior['rows'] if r['split']=='confirm' and r['scores'][r['gold']]<=max(v for k,v in enumerate(r['scores']) if k!=r['gold'])]
    blocks.append('<details><summary>真实失败题（前12条，不用于后选特征）</summary>'+''.join('<article><b>'+html.escape(r['id'])+'</b><pre>'+html.escape(byid[r['id']]['prompt'])+'</pre><p>得分 '+html.escape(str(r['scores']))+'；正确选项 '+str(r['gold'])+'（零基）</p></article>' for r in errors[:12])+'</details>')
    blocks.append('<h2>2. 同一单元组是否参与输出</h2><p>同一固定32单元组在作答末位设为0或2倍。主检验是0倍效应减去同层匹配32单元组的0倍效应；按12个预定检验校正，包括未准入项。所有旁路、范数控制和格式变化另记。</p>')
    blocks.append(table(['分支','步骤','来源','目标−匹配 Δmargin','校正区间','通过'],[[r['branch'],r['depth'],r['mode'],fmt(r['stats']['mean']),str([round(x,4) for x in r['stats']['ci']]),r['passed']] for r in primary]))
    blocks.append('<details><summary>所有原生干预臂、严格准确率与全词表 KL</summary>'+table(['分支','步骤','来源','干预臂','Δmargin','准确率','KL'],[[r['branch'],r['depth'],r['mode'],r['arm'],fmt(r['margin']['mean']),fmt(r['strict_accuracy']),fmt(r['KL'])] for r in tail['summaries']])+'</details>')
    blocks.append('<details><summary>范数控制资格与格式面板</summary>'+table(['分支','对照','通过/总数','最大相对误差'],[[r['branch'],r['arm'],f"{r['passed']}/{r['n']}",fmt(r['max_relative_error'])] for r in tail['norm_control_qualification']])+('<p>XYZ格式面板已完成；复用主确认家族，详见 format-tail-report.json。</p>' if format_tail else '<p>格式内部面板未完成或未获准入，不能声称通过。</p>')+'</details>')
    blocks.append('<details><summary>同剂量随机方向的配对对照</summary><p>这里是目标组0倍减随机方向，按同一输入配对。区间为描述性95%区间，不新增确认通过条件；单个固定随机方向也不覆盖所有可能的无关方向。</p>'+table(['分支','步骤','来源','对照全部合格','Δmargin差','95%区间'],[[r['branch'],r['depth'],r['mode'],r['all_controls_qualified'],fmt(r['statistics']['mean']),str([round(x,4) for x in r['statistics']['ci']])] for r in paths['random_norm_paired']])+'</details>')
    blocks.append('<details open><summary>同题ABC → XYZ：格式反例</summary><p>只比较格式面板对应的20个原家族、普通标题、相同题目与语义选项；不把ABC全80家族与XYZ20家族直接比较。这里报告描述性配对95%区间。</p>'+table(['分支','步骤','来源','干预','ABC Δmargin','XYZ Δmargin','XYZ−ABC','95%区间'],[[r['branch'],r['depth'],r['mode'],r['arm'],fmt(r['ABC']['mean']),fmt(r['XYZ']['mean']),fmt(r['XYZ_minus_ABC']['mean']),str([round(x,4) for x in r['XYZ_minus_ABC']['ci']])] for r in path_report['paired_format']])+'</details>')
    blocks.append('<details><summary>标题措辞与实际选错</summary><p>标题对照为同家族的技术标题减普通标题；以下区间为描述性95%区间，没有将未显著视作等价。准确率主比较只取每家族一条预定的记录题。</p>'+table(['分支','步骤','来源','标题效应差','95%区间'],[[r['branch'],r['depth'],r['mode'],fmt(r['technical_minus_plain']['mean']),str([round(x,4) for x in r['technical_minus_plain']['ci']])] for r in tail['title_control']])+'<pre>'+html.escape(json.dumps(tail['paired_binary'],ensure_ascii=False,indent=2))+'</pre></details>')
    blocks.append('<details><summary>事后诊断：代码读出与状态变化</summary><p>看到格式反向后才设计，复用已暴露的140对题目，没有新增独立样本。将既存终态交叉用ABC/XYZ读出；交叉项不是有效题面的自然答题行为，更不是留出预测。代码项和状态项采用对称分摊恒等式，两者之和等于总变化；这种分摊不代表唯一因果中介。原生对角读出'+str(diagnostic['diagonal_replays'])+'次逐位复现。</p>'+table(['分支','步骤','来源','操作','代码项','状态项','总变化','交互项'],[[r['branch'],r['depth'],r['mode'],r['arm'],*[fmt(r['statistics'][k]['mean']) for k in ['code_component','state_component','total','interaction']]] for r in diagnostic['summary']])+'<p>状态项包含基线归一化、B36响应和全部后续变化，不能据此定位形成代码依赖的具体层。公式、逐题结果及区间见readout-diagnostic.json。</p></details>')
    blocks.append('<h2>3. 补全状态与条件化预测</h2><p>只用发现集中的原子操作拟合。输入分别为B36组A、完整h37、h37加历史前缀均值、词汇基线和正文结束前的平均embedding；不输入待预测状态或正确答案标签。h37历史均值只在作答位置使用，包含此前题目与选项；embedding均值不包含后续题目和选项。完整残差具有更大的信息预算。当前岭回归的预测收益不证明找到了最小充分状态，也不证明所有新增坐标都必要。</p>')
    blocks.append(table(['分支','步骤','来源','读出','输入','MSE/组A','MSE/词汇','MSE/均值','MSE/h37','状态门'],[[r['branch'],r['depth'],r['mode'],r['readout'],r['method'],fmt(r['ratios']['selectedA']['ratio']),fmt(r['ratios']['lexical']['ratio']),fmt(r['ratios']['mean']['ratio']),fmt(r['ratios'].get('hidden',{}).get('ratio')),r['state_completion_passed']] for r in natural]))
    prefix=[r for r in natural if r['method']=='hidden_prefix']
    blocks.append('<p>前缀均值相对单独h37至少改善10%的条件：'+str(sum(r['ratios']['hidden']['ratio']<=.9 for r in prefix))+'/'+str(len(prefix))+'。这是对这一均值摘要的检查，不能推出原始历史状态不重要。</p>')
    blocks.append('<details><summary>条件化全词表预测</summary><p>所有预测都带入真实h37，比较预测后续更新与直接保留h37。它不等于仅凭外部文本重建模型。</p>'+table(['分支','步骤','来源','预测输入','KL','选项准确率'],[[r['branch'],r['depth'],r['mode'],r['method'],fmt(r['KL']),fmt(r['strict_accuracy'])] for r in out['summary']])+'</details>')
    blocks.append('<h2>4. 改动如何传播到后三层</h2><p>已知权重与自然状态给出条件化局部算子：δh40 ≈ J39 J38 J37 δh37，其中 Jℓ=(I+JMLP,ℓ)(I+JAttn,ℓ)。实际干预后的下游状态只用于验收，不输入预测器。前缀K/V来自自然前向，因编辑发生在最后一个位置而保持不变。算子使用后三块完整权重，尚未提取小型可复用语义电路。</p><p>根G/V/A的预测误差须不超过零效应基线的25%、不超过仅残差预测的90%；实测标准化RMS至少0.05。BF16舍入误差独立保留。</p>')
    blocks.append(table(['分支','步骤','来源','剂量','实测RMS','MSE/零效应','MSE/残差','通过'],[[r['branch'],r['depth'],r['mode'],r['dose'],fmt(r['rms_actual_standardized']),fmt(r['ratios']['zero']['ratio']),fmt(r['ratios']['residual']['ratio']),r['passed']] for r in causal]))
    blocks.append('<details><summary>同一批干预上的拟合预测器对照</summary><p>所有拟合系数在确认前固定。组A预测器只得到该组改动；h37预测器与原生局部算子得到完全相同的实际残差改变量。局部算子另使用自然下游状态及参数，信息预算更大。</p>'+table(['分支','步骤','来源','剂量','预测器','拟合MSE/零效应','原生局部MSE/拟合MSE'],[[r['branch'],r['depth'],r['mode'],r['dose'],name,fmt(v['mse_over_zero']['ratio']),fmt(v['native_jvp_over_learned']['ratio'])] for r in causal for name,v in r['learned_baselines'].items()])+'</details>')
    blocks.append('<details><summary>最终HiddenState与编辑后全词表预测</summary><p>目标是原模型真实编辑后的输出，比较局部预测与保持自然输出。这里没有新增通过阈值；根G/V/A的预设门仍独立判断。</p>'+table(['分支','步骤','来源','剂量','h40 MSE/零效应','h40 MSE/残差'],[[r['branch'],r['depth'],r['mode'],r['dose'],fmt(r['hidden_ratios']['zero']['ratio']),fmt(r['hidden_ratios']['residual']['ratio'])] for r in causal])+table(['分支','步骤','来源','剂量','算子','编辑目标KL','保持自然输出KL','KL比'],[[r['branch'],r['depth'],r['mode'],r['dose'],r['operator'],fmt(r['mean_KL']),fmt(r['mean_zero_effect_KL']),fmt(r['ratio']['ratio'] if r['ratio'] else None)] for r in paths['edited_output']])+'</details>')
    blocks.append('<details><summary>根单元的门、值、联合项与舍入</summary><p>ΔA=V·ΔSiLU(G)+SiLU(G)·ΔV+ΔSiLU(G)·ΔV+BF16舍入剩余。下表为逐题根向量RMS的均值；项之间可抵消，不能相加为百分比。</p>'+table(['分支','步骤','来源','剂量','实际ΔA','门项','值项','联合项','舍入'],[[r['branch'],r['depth'],r['mode'],r['dose'],*[fmt(r['mean_per_row_RMS'][f]) for f in ['actual_A_rms','gate_rms','value_rms','cross_rms','rounding_rms']]] for r in paths['gate_value_terms']])+'<p>成员线性写入之和与原生组残差改变量的相对差：平均 '+fmt(paths['unit_seed_rounding_relative']['mean'])+'，最大 '+fmt(paths['unit_seed_rounding_relative']['maximum'])+'。</p></details>')
    blocks.append('<details><summary>固定路径与根恢复的真实干预</summary><p>下表是相对未固定路径的0倍干预，答案间隔恢复量；不自动等同于唯一中介比例。根恢复只把B39选定八单元的A（down输入）恢复为自然值，保留其他单元及状态。</p>'+table(['分支','步骤','来源','操作','间隔恢复','95%区间'],[[r['branch'],r['depth'],r['mode'],r['arm'],fmt(r['margin_recovery']['mean']),str([round(x,4) for x in r['margin_recovery']['ci']])] for r in tail['path_clamps']])+'</details>')
    blocks.append('<details><summary>32个成员的线性化贡献</summary><p>φj≈读出导数·J39J38J37·(−Aj dj)。这是基线附近的计算来源分解，并非32项单元独立因果实验，也不是语义标签。</p>'+table(['分支','B36单元','平均预测间隔贡献'],[[r['branch'],r['unit'],fmt(r['mean_contribution'])] for r in tail['unit_linearized_attribution']])+'</details>')
    blocks.append('<h2>5. 原生坐标全景</h2><p>以下视图保持原生坐标顺序，无Top-K、PCA或重新排序。原始差值和按发现集坐标标准差缩放的视图分别保存；每张图使用自己的完整极值对称色标，图间颜色不能直接比较。</p>')
    for v in maps['views']:
        blocks.append('<details><summary>'+html.escape(v['name'])+'</summary><p>'+html.escape(v['meaning'])+'；色标 '+str(v['range'])+'</p><a href="'+v['file']+'"><img loading="lazy" src="'+v['file']+'" alt="'+html.escape(v['meaning'])+'"></a></details>')
    blocks.append('<h2>6. 可以说什么，仍缺什么</h2><ul><li>共享响应、真实输出作用、局部传播预测与共享语义是不同结论。只能按各自通过的检验陈述。</li><li>本阶段未获得可靠的两、三步操作内部准入；记录任务中存在复杂的背景指令，不等于模型实际执行了这些指令。</li><li>局部算子使用自然下游基线状态与已知参数；它说明条件化传播，不能代替从文本提取共享语义规则。</li><li>只编辑最后一个位置，不涉及整个历史计算、训练中如何形成分支、其他模型或自由生成。</li><li>32个成员的单独贡献是近似归因；组消融不证明每个成员必要或共享同一个语义。</li><li>标题与来源长度配平，仍不能排除其他措辞、选项或通用答题功能。</li><li>事后代码分解存在状态与读出交互；margin的最大竞争项也会改变，不能将交互量直接视为神经元交互。该诊断未新增独立确认样本。</li></ul><p><b>下一大阶段：</b>先在独立材料上校准直接语义答案、自由生成和代码轮换，冻结未见组合；再比较含历史来源的状态与同预算特征方法，并用实际净输出中介定位路径。需要验证跨答案写法的关系规律，不能继续只沿着单一代码的高响应根扩图。</p>')
    links=['research-summary.json','trial-plan.json','scope-plan.json','analysis-plan.json','execution-freeze.json','full-behavior-confirmation.json','confirm-state-report.json','confirm-output-report.json','confirm-tail-report.json','path-output-report.json','heatmap-manifest.json','feature-cards.json','mechanism-graph.json','audit.json']
    if format_tail:links.append('format-tail-report.json')
    links+=['readout-diagnostic-plan.json','readout-diagnostic.json']
    blocks.append('<h2>7. 复核入口</h2><p>模型快照 '+B.SNAPSHOT+'。所有指标对应冻结材料、源代码哈希和原生产物；没有部署或晋级正式中心。</p><div class="links">'+''.join('<a href="'+n+'">'+n+'</a>' for n in links)+'</div>')
    css='body{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;background:#f1f3f5;color:#1d2939;margin:0}main{max-width:1160px;margin:auto;padding:36px 24px 90px}h1{font-size:38px;line-height:1.2}h2{margin-top:42px;font-size:25px}.kicker{color:#45617e;font-size:14px}.notice{background:#fff5dd;border-left:4px solid #b47a22;padding:20px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}.stats div{padding:16px;background:white;border-radius:8px}.stats strong{display:block;font-size:28px}.stats span{font-size:13px;color:#526174}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{border-bottom:1px solid #dce1e7;padding:9px 12px;text-align:left;white-space:nowrap}th{background:#e6edf5}details{margin:14px 0;border:1px solid #d1d9e4;background:white;border-radius:7px;padding:14px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;background:#f3f5f8;padding:14px}img{display:block;width:100%;height:auto}article{border-top:1px solid #ddd;padding:12px 0}.links{display:flex;flex-wrap:wrap;gap:14px}a{color:#15588b}@media(max-width:650px){main{padding:22px 14px}h1{font-size:28px}.stats{grid-template-columns:repeat(2,1fr)}body{font-size:15px}}'
    css+='main{overflow-wrap:anywhere;min-width:0}.scroll{max-width:100%}'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B36固定单元组研究</title><style>'+css+'</style><main>'+''.join(blocks)+'<footer><p>生成于 '+html.escape(B.now())+' · '+B.RUN+'</p></footer></main></html>'
    (B.OUT/'report.html').write_text(page,encoding='utf8')
    B.save('report-build.json',dict(report_sha256=B.file_sha(B.OUT/'report.html'),source_sha256=B.file_sha(__file__),inputs={n:B.file_sha(B.OUT/n) for n in links if (B.OUT/n).exists()}))
    print('Report written',B.OUT/'report.html',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['heatmaps','report']);a=p.parse_args();globals()[a.stage]()
