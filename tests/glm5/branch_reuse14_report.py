"""Readable scientific report; keep observations, prediction and causal tests separate."""
import html
import time
import numpy as np
import branch_reuse14 as B


def esc(x):return html.escape(str(x))
def fmt(x):return '—' if x is None else f'{x:.5g}'
def pct(x):return f'{100*x:.2f}%'
def interval(x):return '['+', '.join(fmt(v) for v in x)+']'
def yes(x):return '<span class="yes">通过</span>' if x else '<span class="no">未通过</span>'
def table(headers,rows):return '<div class="table"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def link(path,label):return f'<a href="{esc(path)}">{esc(label)}</a>'


def plots(response,output,intervention):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,field in zip(axes,'VA'):
        rows=[r for r in response['responses'] if r['primary'] and r['field']==field]
        m=np.array([r['contrast']['mean'] for r in rows]);ci=np.array([r['contrast']['ci'] for r in rows])
        ax.errorbar(range(len(rows)),m,yerr=np.array([m-ci[:,0],ci[:,1]-m]),fmt='o',capsize=5,color='#235c8b')
        ax.axhline(0,color='#888',lw=1);ax.set_xticks(range(len(rows)),[r['branch'] for r in rows]);ax.set_title('B19/U64 '+field);ax.set_ylabel('Paired rule - record difference; corrected CI')
    fig.tight_layout();fig.savefig(B.OUT/'primary-response.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,readout in zip(axes,[0,1]):
        names=[(m,mode) for m in ['correlation','parameter'] for mode in ['direct','composed']]
        for j,branch in enumerate(['number','label']):
            vals=[next(r['ratios']['mean']['ratio'] for r in response['rollouts'] if (r['method'],r['mode'],r['readout'],r['branch'])==(m,mode,readout,branch)) for m,mode in names]
            ax.bar(np.arange(4)+(j-.5)*.34,vals,.34,label=branch)
        ax.axhline(.5,color='#a13434',ls='--',label='Frozen ceiling .5');ax.axhline(1,color='#888',lw=.8)
        ax.set_xticks(range(4),[m[:4]+' '+mode for m,mode in names],rotation=20);ax.set_ylabel('Free root G/V/A MSE / mean baseline')
        ax.set_title('Content end' if readout==0 else 'Answer prompt');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(B.OUT/'graph-rollout.png',dpi=150);plt.close(fig)
    names=['correlation-direct','correlation-composed','parameter-direct','parameter-composed','oracle','embedding','lexical','residual','mean']
    fig,ax=plt.subplots(figsize=(13,4.8))
    for j,branch in enumerate(['number','label']):
        vals=[next(r['mean_KL'] for r in output['summary'] if (r['branch'],r['method'])==(branch,m)) for m in names]
        ax.bar(np.arange(len(names))+(j-.5)*.36,vals,.36,label=branch)
    ax.set_yscale('log');ax.set_ylabel('Teacher KL over full vocabulary (lower is better)');ax.set_xticks(range(len(names)),names,rotation=25,ha='right');ax.legend()
    ax.set_title('Conditional late-state prediction; residual baseline uses all 5120 h37 coordinates')
    fig.tight_layout();fig.savefig(B.OUT/'output-kl.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for ax,branch in zip(axes,['number','label']):
        arms=['single_0','group_0','matched_0'];rr=[next(r for r in intervention['effects'] if (r['kind'],r['branch'],r['arm'])==('B19',branch,a)) for a in arms]
        mean=np.array([r['margin']['mean'] for r in rr]);ci=np.array([r['margin']['ci'] for r in rr])
        ax.errorbar(range(3),mean,yerr=np.array([mean-ci[:,0],ci[:,1]-mean]),fmt='o',capsize=5,color='#235c8b');ax.axhline(0,color='#888',lw=1)
        ax.set_xticks(range(3),arms);ax.set_ylabel('Correct-option margin change');ax.set_title(branch+'; corrected primary / 95% control CI')
    fig.tight_layout();fig.savefig(B.OUT/'native-intervention.png',dpi=150);plt.close(fig)
    return ['primary-response.png','graph-rollout.png','output-kl.png','native-intervention.png']


def run():
    B.configure();response=B.read(B.OUT/'response-and-graph-report.json');output=B.read(B.OUT/'output-fidelity.json')
    intervention=B.read(B.OUT/'intervention-report.json');controls=B.read(B.OUT/'control-comparisons.json');groups=B.read(B.OUT/'group-response-profiles.json')
    behavior=B.read(B.OUT/'strict-behavior-confirm.json');maps=B.read(B.OUT/'fieldmap-manifest.json');examples=B.read(B.OUT/'counterexamples.json')['examples']
    figure_files=plots(response,output,intervention);scope=B.read(B.OUT/'qualified-execution.json');audit=B.read(B.OUT/'audit.json')
    count19=sum(r['positive_both_eligible_branches'] for r in groups['results'] if r['group']=='B19-preselected')
    count39=sum(r['positive_both_eligible_branches'] for r in groups['results'] if r['group']=='B39-roots')
    causal_pass=sum(r['passed'] for r in intervention['causal_predictions']);causal_info=sum(r['informative'] for r in intervention['causal_predictions'])
    graph_status=[]
    for method in ['correlation','parameter']:
      for mode in ['direct','composed']:
        er=[r for r in response['edges'] if r['method']==method];rr=[r for r in response['rollouts'] if r['method']==method and r['mode']==mode]
        oo=[r for r in output['summary'] if r['method']==method+'-'+mode];cc=[r for r in intervention['causal_predictions'] if r['method']==method and r['mode']==mode]
        graph_status.append(dict(method=method,mode=mode,eligible_observation_passed=all(r['passed'] for r in er+rr),
            eligible_output_passed=all(r['passed'] for r in oo),eligible_causal_prediction_passed=all(r['passed'] for r in cc),
            full_required_passed=False,excluded_branches=scope['excluded_branches']))
    verdict=dict(definition='同一单元/单元组跨语义分支复用共同结构，并通过层间组合形成更复杂特征',
        common_response=dict(B19_group_positive=count19,total19=8,B39_roots_positive=count39,total39=8,
            status='Fixed preselected units show common responses across eligible heldout number/label branches. Semantic common structure is not uniquely identified against wording/task-role alternatives.'),
        historical_unit_primary_passed=False,graphs=graph_status,causal_cells_passed=causal_pass,causal_cells_informative=causal_info,
        native_B36_parameter_group=dict(status='Same fixed 32-unit group affects correct-option margins in both eligible branches; exploratory secondary effects, with matched-group comparisons. Common semantic content and mediation by the fitted chain are not established.',
            effects=[r for r in intervention['effects'] if r['kind']=='B36' and r['arm'] in ['parameter_0','parameter_2']],
            index_source='frozen-selection.json / graphs.parameter.edges[-1].parents'),
        required_branches=scope['required_branches'],eligible_branches=scope['eligible_branches'],
        conclusion='No tested candidate reaches the full frozen joint mechanism gate. Shared responses and partial output information are retained; neither universal absence of branch reuse nor formation of higher-complexity features has been established.',
        source_sha256=B.file_sha(__file__))
    B.save('conclusion.json',verdict)
    selection=B.read(B.OUT/'frozen-selection.json')
    B.save('native-group-effect-card.json',dict(
        feature_id=B.SNAPSHOT+':B36:mlp_down_input:parameter_group32:BR14v3',
        units=selection['graphs']['parameter']['edges'][-1]['parents'],
        selection='Frozen discovery-time weight read-write coefficient prior, 32 parents of the B37 target group; no confirmation-based reselection.',
        readout='Final answer-prompt token only; selection and state prediction evaluated separately at both declared readouts.',
        natural_corpus='Not measured for this candidate group.',
        template_scope='Rule versus finite records; eligible number and unseen-by-fit label branches. Coordinate confirmation excluded after failed behavior gate.',
        original_model_effects=verdict['native_B36_parameter_group']['effects'],
        matched_comparisons=[r for r in controls['comparisons'] if r['kind']=='B36' and r['native'].startswith('parameter')],
        output_predictions='output-fidelity.json; group-state conditional prediction, not text-only atlas fidelity.',
        downstream_effect_predictions=[r for r in intervention['causal_predictions'] if r['method']=='parameter'],
        counterexamples='All causal-effect predictions fail; finite records show larger margin effects than rules; label strict accuracy decreases only 1.25 percentage points under zeroing.',
        limits='No matched realized-norm control for B36, mediator necessity test, natural-corpus/free-output or broad-task replication. Individual members are not independently established.',
        evidence_grade='Range-limited original-model group intervention with exploratory output effects; no confirmed shared semantic code or fitted causal chain.',
        source_sha256=B.file_sha(__file__)))
    body=[]
    body.append('<header><p class="eyebrow">原始 BF16 · CUDA · 本机研究 · '+esc(B.RUN)+'</p><h1>Qwen3-14B：跨分支复用与层间组合</h1><p>同一批原生单元是否复用共同结构，并经层间组合参与更复杂的预测？</p></header>')
    body.append(f'<div class="notice"><strong>本轮结论：已观察到跨分支共同响应，尚未确认完整的“泛化递归”机制。</strong><p>预选 B19 组有 {count19}/8 个单元、B39 根组有 {count39}/8 个单元，在数值与标签留出材料上都出现正向 A 响应。按权重关系选出的同一 B36 单元组还对两个分支的答案间隔有可测作用。但历史 B19/U64 的通用 V/A 解释，以及两条三层递推模型，均未通过完整验收。不能把失败扩展成“14B 内不存在这种机制”。</p></div>')
    body.append('<p>范围：1,536 条 v3 输入保存了 40 块全部 G/V/A、41 个残差边界的完整坐标。正式确认原有三个分支、每分支 80 家族；坐标分支未通过行为门，停止其内部确认与干预。以下内部结果仅含数值与标签的 640 条、160 家族。标签分支未参与拟合；两分支确认都使用新表达。数值可能与发现集重叠。</p>')
    body.append('<nav>'+''.join(link('#'+x,y) for x,y in [('behavior','行为与范围'),('response','共同响应'),('graph','层间预测'),('output','输出保真'),('causal','原模型干预'),('maps','全坐标视图'),('cases','原文反例'),('limits','结论边界')])+'</nav>')
    body.append('<section id="behavior"><h2>先检查模型是否完成任务</h2><p>通用规则可应用于未列出的输入；有限记录明确没有为其他输入规定输出。两者仍都含关系。正确选项分数必须严格大于所有错误选项，同分不算对。所有阈值保持冻结值。</p>')
    body.append(table(['分支','条件','标题','正确 / 数量','准确率','行为门'],[[r['branch'],r['role'],r['title'],f"{r['correct']}/{r['n']}",pct(r['accuracy']),yes(r['passed'])] for r in behavior['cells']]))
    body.append('<p>早期序列试跑也未通过：同一组规则题轮换选项后，正确答案放在 A/B 时均答对，放在 C 时只有 12.5%。它是选择题接口的反例，不能证明模型不会重排。v1/v2 失败、严格计分修正和坐标分支缺失均保留，未调低行为门。</p></section>')
    body.append('<section id="response"><h2>共同响应：有复现，但语义解释仍需排除替代原因</h2><p>G 是 gate 投影，V 是 up 投影，A 是实际 MLP 中间写入系数。计算关系是原架构定义：<code>G=gᵀz，V=uᵀz，A=SiLU(G)V，写入=A·d</code>。A 的正向差异不等于多个单元的写入方向一致。</p>')
    body.append(table(['历史候选','分支','量','配对差','家族 dz','校正区间','p','组合/嵌入基线 MSE','主门'],[[f"B{r['block']}/U{r['unit']}",r['branch'],r['field'],fmt(r['contrast']['mean']),fmt(r['contrast']['dz']),interval(r['contrast']['ci']),fmt(r['contrast']['p_two_sided']),fmt(r['combined_over_embedding']),yes(r['passed'])] for r in response['responses'] if r['primary']]))
    body.append('<img src="primary-response.png" alt="历史候选的留出 V 与 A 配对差及校正置信区间"><p>数值 A 的校正区间含零，不能称为显著负向。它没有达到预设的正向效应下限。标签 V/A 的正向响应很强，但加入已知关系标签未改善强嵌入基线；后者也可能已包含相同信息，不能单独据此断言没有语义。</p>')
    body.append(table(['预选组','单元','数值 A差 / dz','标签 A差 / dz','两分支共同正向响应'],[[r['group'],f"B{r['block']}/U{r['unit']}",*[fmt(z['contrast']['mean'])+' / '+fmt(z['contrast']['dz']) for z in r['A_results']],yes(r['positive_both_eligible_branches'])] for r in groups['results']]))
    body.append('<p>上表是发现阶段选定单元的次级留出描述，不能替代失败的主门。共同响应仍可能来自文档角色措辞、任务格式或词汇共现。每家族四个版本配对分析，不能算四倍独立样本。</p></section>')
    body.append('<section id="graph"><h2>三层递推：原生索引相连，预测仍不闭合</h2><p>从 B39 选 8 个根单元，向 B38/B37/B36 各选 32 个父单元。相关性路线与真实权重读写系数路线分别冻结；真实权重评分省略注意力与完整归一化导数，因此也只是选点先验。每层拟合不同的转移，不是证明同一算法反复递归执行。</p><p><code>预测 G/V/A = Fₗ(上一块选定 A)</code>；自由递推比较直接预测 A 和 <code>SiLU(预测 G)×预测 V</code>。仅 B36 种子来自真实模型，后面不读取真实父值。</p>')
    body.append(table(['路线','方式','读出','分支','MSE/均值','MSE/词汇','MSE/嵌入','递推门'],[[r['method'],r['mode'],'正文末位' if r['readout']==0 else '作答末位',r['branch'],fmt(r['ratios']['mean']['ratio']),fmt(r['ratios']['lexical']['ratio']),fmt(r['ratios']['embedding']['ratio']),yes(r['passed'])] for r in response['rollouts']]))
    edge_pass=sum(r['passed'] for r in response['edges'])
    body.append(f'<img src="graph-rollout.png" alt="三层自由递推的误差与冻结门槛"><p>局部边 {edge_pass}/{len(response["edges"])} 个条件通过，但没有整条链满足全部推广与基线要求。完整结果保留逐项家族 bootstrap 区间。<strong>原生编号、权重来源、观察预测与因果边分别记账。</strong></p><p>这些材料主要检验一步规则是否可推广，未系统改变组合深度，也没有独立标注“更复杂特征”的真值。经过三层不等于形成了三层更复杂的概念；本轮不能对这一部分给出肯定证明。</p></section>')
    body.append('<section id="output"><h2>输出保真：有信息，不能替代内部机制验收</h2><p>由真实 B36 的选定 A 起步，递推到 B39 根 A，再用发现集拟合的读出器预测原始 h40，经原模型最终 RMSNorm 与完整词表 unembedding 输出。它使用真实的晚层状态，不是仅凭文本的图谱解码。</p>')
    body.append(table(['分支','预测器','全词表 KL','选项准确率（原 argmax 口径）'],[[r['branch'],r['method'],fmt(r['mean_KL']),pct(r['accuracy'])] for r in output['summary']]))
    body.append('<img src="output-kl.png" alt="条件化完整词表 KL 与不同信息预算基线"><p>四个链式预测器在两个合格分支都优于均值与前缀平均 embedding 的 KL，但远不及同一作答位置全部 5,120 个 h37 坐标。这个强基线也不包含完整历史 KV，且信息预算更大。预测根有时比真实根经同一拟合读出器更好，可能来自平滑或读出器的分布偏差，不能解释为“更真实的内部机制”。KL 与选项准确率也不总同向。</p></section>')
    body.append('<section id="causal"><h2>原模型干预与反事实预测</h2><p>每个合格分支预选 40 个家族、保留四个版本。B19 操作正文各位置的原生 A，B36 仅操作作答末位。两组各 320 条输入、10 臂，共 6,400 条逐臂结果，仍是同一批 80 家族。0/2 倍操作不搬运差分方向；1 倍臂只是数值无操作，不是独立救援。</p>')
    strict={(r['kind'],r['branch'],r['arm']):r for r in controls['strict_accuracies']}
    effects=[r for r in intervention['effects'] if (r['kind']=='B19' and r['arm'] in ['single_0','group_0','matched_0','group_norm']) or (r['kind']=='B36' and r['arm'] in ['parameter_0','parameter_2','parameter_matched_0','parameter_matched_2'])]
    body.append(table(['位置/操作','分支','间隔变化','区间','p','严格准确率 原→操作','平均KL'],[[r['kind']+'/'+r['arm'],r['branch'],fmt(r['margin']['mean']),interval(r['margin']['ci']),fmt(r['margin']['p_two_sided']),pct(strict[r['kind'],r['branch'],r['arm']]['strict_baseline_accuracy'])+' → '+pct(strict[r['kind'],r['branch'],r['arm']]['strict_accuracy']),fmt(r['mean_KL'])] for r in effects]))
    body.append('<img src="native-intervention.png" alt="原模型 B19 单元与组干预的配对效应"><p>单元／组 0 倍主检验维持原六检验校正；匹配组及其差异为探索性。显著损伤不自动等于共同语义、唯一性或必要性。严格随机方向范数控制只有全部被操作位置误差≤2%才合格，资格不足和选中子集偏差均保留。</p>')
    body.append('<p><strong>B36 的正向证据：</strong>权重关系选出的同一 32 单元组，在数值／标签上设 0 倍使正确答案间隔平均降低 0.3242／0.2078 logit，2 倍提高 0.3375／0.1953。相对同层匹配组的差异也有相同方向。它们是预定操作的次级结果，说明该组在本材料中有可测作用，不能归结为“都没有影响”。不过 0 倍对有限记录的损伤更大，规则减记录效应为 +0.1547／+0.1344；它并不表现为规则条件的选择性损伤。没有广泛副作用、答案词域与等实际剂量控制，不能认定它承载共同抽象语义，或效应一定通过所拟合 B39 根中介。</p>')
    body.append(table(['路线','方式','剂量','分支','实际效应标准化RMS','误差/零效应','95%区间','有检验力','因果预测门'],[[r['method'],r['mode'],r['dose'],r['branch'],fmt(r['standardized_actual_RMS']),fmt(r['ratio_to_zero_effect']['ratio']),interval(r['ratio_to_zero_effect']['ci95']),'是' if r['informative'] else '不足',yes(r['passed'])] for r in intervention['causal_predictions']]))
    body.append('<p>“因果预测门”检验 B36 改动能否按冻结模型预测 B39 根 G/V/A 的实际变化，不是只看输出是否被扰动。实际效应小于预设标准化 RMS 0.05 时记为检验不足，不算证明没有连接。</p></section>')
    body.append('<p>本轮 16 格均达到预设实测效应下限，但没有一格通过反事实预测。相关路线误差为零效应基线的 212–471 倍，权重路线为 2.49–4.40 倍；不是因所有干预都无变化而无法检验。权重组选点与拟合转移必须分开评价。</p>')
    body.append('<section id="maps"><h2>全原生坐标观察图</h2><p>只使用发现材料。原生顺序不排序，不用 PCA 或 Top-K 定义主干。概览按连续 16 单元求均值，可能抵消相反响应；逐坐标 PNG 与数组保留每个单元及低幅背景。各图显示对称色标，约 0.5% 数值超过显示范围，原始数组未截断。</p>')
    for row in maps['plots']:
        body.append('<figure><img src="'+esc(row['overview'])+'" alt="'+esc(row['field']+' '+row['statistic']+'全坐标观察概览')+'"><figcaption>'+esc(row['field']+' / '+row['statistic']+'，色标 '+str(row['color_limits']))+' · '+link(row['native'],'打开逐原生坐标图片')+'</figcaption></figure>')
    body.append('</section><section id="cases"><h2>代表性反例与原文</h2><p>每个历史候选、每个合格分支选出 A 配对差最小与最大的家族，属于验收后的描述性展示；全部材料与统计同时保留。</p>')
    for example in examples:
        body.append('<details class="case"><summary>'+esc(example['candidate']+' · '+example['family']+' · A配对差='+fmt(example['A_difference']))+'</summary>')
        for row in example['variants']:
            body.append('<p>'+esc(row['id']+f" | G={row['G']:.6g}, V={row['V']:.6g}, A={row['A']:.6g}")+'</p><pre>'+esc(row['body'])+'</pre><details><summary>后续问题与选项：正文读出当时尚未出现</summary><pre>'+esc(row['future_probe'])+'</pre></details>')
        body.append('</details>')
    body.append('</section><section id="limits"><h2>结论边界与下一阶段</h2><p><strong>可以确定：</strong>这组预选单元存在跨合格分支复现的共同响应；历史 B19/U64 不能作为已经通过本轮强检验的通用解释；当前两类三段预测链没有达到冻结的联合机制门。输出预测的部分收益与内部响应／因果预测的失败必须同时保留。</p><p><strong>仍未确定：</strong>共同响应是否主要对应语义关系而非共同措辞；哪些残差、注意力和历史信息构成足够的转移状态；更复杂特征是否由这些特定单元组合形成；以及模型训练时如何形成分支。全模型不存在此机制并未得到证明。</p><p>主要硬伤：序列与坐标行为测量失败；只有少量构式与人工材料；标签是唯一未参与拟合的合格语义分支；同一家族之外也共享词与句式；没有自然语料复现、独立节点复算、同预算 SAE/transcoder、自由生成和广泛副作用面板。置信区间按家族计算，不涵盖所有构式与领域不确定性。</p><p>下一大阶段应围绕“共同结构的识别、组合与实际使用”整体推进：先以选项轮换和自由输出校准测量，再冻结能排除局部关键词／角色措辞的跨分支材料及未见组合；在同一材料上比较原生单元组、稀疏特征与包含残差／注意力来源的状态；仅对通过者追踪局部读写、匹配干预、恢复及组合复杂度。不要把画得像树的响应图当作层层形成复杂特征的证据。</p><p>RDC 名称、历史部分映射与类型化三图谱接口保持不变。本轮的 SiLU 乘积、岭回归、KL 和期望分解均非新数学定理。</p></section>')
    evidence=['conclusion.json','analysis-plan.json','trial-plan.json','qualified-execution.json','qualification-runtime-recovery.json','strict-behavior-confirm.json',
        'response-and-graph-report.json','group-response-profiles.json','feature-cards.json','native-group-effect-card.json','output-fidelity.json','intervention-report.json','control-comparisons.json',
        'joint-state-summary.json','counterexamples.json','native-manifest.json','audit.json','protocol-result.json','full-native-fieldmaps.npz']
    body.append('<section><h2>复核入口</h2><p>'+ ' · '.join(link(p,p) for p in evidence)+'</p><p>数据按 D 盘策略保存。原生数据 '+f"{B.read(B.OUT/'native-manifest.json')['total_bytes']:,}"+' 字节，复查日期 2026-12-13。执行审计通过不等于机制通过。协议回归 314/315，唯一既有 SOURCE-BINDING 映射问题保留。未发布客户端或公网。</p></section>')
    css='body{margin:0;background:#f4f6fa;color:#182b3b;font:16px/1.7 system-ui,Segoe UI,sans-serif}main{max-width:1180px;margin:auto;padding:32px 24px}h1{font-size:32px;line-height:1.3}h2{font-size:24px;margin-top:0}section{margin:26px 0;padding:25px;background:white;border:1px solid #dce3eb;border-radius:12px}header{padding:20px 0}.eyebrow{font-size:13px;color:#60788c}.notice{padding:22px;background:#fff4d9;border:1px solid #dec689;border-radius:12px}nav{display:flex;gap:15px;flex-wrap:wrap;padding:15px 0}a{color:#145b8d;overflow-wrap:anywhere}.table{overflow:auto}table{border-collapse:collapse;min-width:600px;width:100%;font-size:14px}td,th{padding:9px 10px;text-align:left;border-bottom:1px solid #dde4eb;vertical-align:top}th{background:#eaf1f7}.yes{color:#21674c}.no{color:#9c3935}img{display:block;width:100%;height:auto;margin:20px 0}figure{margin:24px 0}figcaption{font-size:13px;color:#50687a}pre{padding:12px;background:#edf2f6;white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}details{margin:15px 0;border:1px solid #dce3eb;border-radius:7px;padding:12px}summary{cursor:pointer;font-weight:600}code{overflow-wrap:anywhere}section{scroll-margin-top:15px}@media(max-width:600px){main{padding:15px 10px}h1{font-size:26px}section{padding:16px}h2{font-size:21px}}'
    page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>14B 跨分支复用与层间组合</title><style>'+css+'</style></head><body><main>'+''.join(body)+'</main></body></html>'
    (B.OUT/'report.html').write_text(page,encoding='utf8')
    B.save('report-build.json',dict(report_sha256=B.file_sha(B.OUT/'report.html'),figures=figure_files,
        conclusion_sha256=B.file_sha(B.OUT/'conclusion.json'),source_sha256=B.file_sha(__file__),created_local=time.strftime('%Y-%m-%d %H:%M'),
        missing_branch_never_counted_as_pass=True,local_only=True))


if __name__=='__main__':run()
