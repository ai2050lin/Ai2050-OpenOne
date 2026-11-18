"""Append-only scientific record, generated from completed real artifacts."""
import hashlib,json,re
from pathlib import Path
import numpy as np
import b36_composition14 as B
import b36_composition14_tail as T


def f(v):return '—' if v is None else f'{v:.6g}' if isinstance(v,(float,int)) else str(v)
def ci(s):return '['+','.join(f(v) for v in s)+']'
def table(headers,rows):return '\n'.join([' | '.join(headers),' | '.join(['---']*len(headers))]+[' | '.join(f(v) for v in r) for r in rows])


def append(path,make):
    old=path.read_bytes();text=old.decode('utf8');phase=max(map(int,re.findall(r'^## Phase (\d+):',text,re.M)))+1
    addition=('\n\n'+make(phase)+'\n').encode('utf8')
    assert path.read_bytes()==old,'Concurrent memo modification; retry append from current end'
    with path.open('ab') as out:out.write(addition);out.flush()
    new=path.read_bytes();assert new[:len(old)]==old and new[len(old):len(old)+len(addition)]==addition
    return dict(path=path.relative_to(B.ROOT).as_posix(),phase=phase,old_bytes=len(old),appended_bytes=len(addition),
        before_sha256=hashlib.sha256(old).hexdigest(),append_sha256=hashlib.sha256(addition).hexdigest(),prefix_after_sha256=hashlib.sha256(new[:len(old)+len(addition)]).hexdigest())


def run():
    B.configure();T.admit();assert not (B.OUT/'memo-append.json').exists()
    d={n:B.read(B.OUT/n) for n in ['research-summary.json','full-behavior-confirmation.json','confirm-admissibility.json','confirm-state-report.json','confirm-tail-report.json',
        'confirm-output-report.json','path-output-report.json','readout-diagnostic.json','audit.json','browser-check.json','cleanup.json','protocol-regression.json','delivery-source.json']}
    assert d['audit.json']['passed'] and d['browser-check.json']['passed']
    summary=d['research-summary.json'];behavior=d['full-behavior-confirmation.json'];tail=d['confirm-tail-report.json'];state=d['confirm-state-report.json'];output=d['confirm-output-report.json']
    paths=d['path-output-report.json'];diagnostic=d['readout-diagnostic.json']
    stamp=B.now()[:16].replace('T',' ');basename=B.OUT.relative_to(B.ROOT).as_posix()
    summaries={(r['branch'],r['depth'],r['mode'],r['arm']):r for r in tail['summaries']}
    natural=[r for r in state['results'] if r['target']=='roots' and r['method'] in ['hidden','hidden_prefix']]
    outby={(r['branch'],r['depth'],r['mode'],r['method']):r for r in output['summary']}
    stages=[]
    for line in (B.OUT/'progress.jsonl').read_text(encoding='utf8').splitlines():
        try:stages.append(json.loads(line))
        except json.JSONDecodeError:pass
    cuda=max((r.get('cuda_peak',0) for r in stages),default=0);shared=max((r.get('shared_gpu_bytes',0) for r in stages),default=0)
    native=d['audit.json']['native_bytes_counting_referenced_pilot_once'];protocol=d['protocol-regression.json']
    def main(phase):
        p=[f'## Phase {phase}: Qwen3-14B B36固定32单元的措辞控制、状态补全与原生层间传播 [ABBR]',
           f'**目标与证据范围。** RUN `{B.RUN}`，本地Qwen3-14B原始BF16快照 `{B.SNAPSHOT}`，RTX 5080 CUDA逐块执行。接续Phase2924同一B36的32单元、同层匹配32单元与B39八个根，全部索引在本轮确认前固定。检验新措辞与具体答案下的输出作用、补充残差/历史状态的预测收益、真实改动的层间传播，以及未参与拟合的操作组合。执行、技术审计与科学主张分别登记；正式晋级0。时间为America/Chicago（UTC−06:00）。',
           '**行为校准、版本与样本。** v1两来源试跑288条未过门，随后多种接口诊断仅用已经暴露的试跑家族；没有读取确认材料选模板。v2改用明确数值运算和整串大写/追加后缀，试跑237/288；标签操作以及两、三步程序组合不合格。v3在确认前保留相同正文、划分和旧单元组，按已冻结规则缩小内部范围；没有降低阈值。试跑原生数据只读复用v2，不算新样本。原标题、格式、顺序和失败条件均保留。',
           f'材料共3,536条：试跑288；原始发现768，其中准入576完成采集（96家族，565/576严格答对），未准入发现192未执行；确认1,920、格式240、顺序320。确认80家族/分支，共160家族。v3预先采集候选内部确认1,120条、格式140条；其余确认800、格式100、顺序320共1,220条只测行为。最终确认内部准入{summary["admitted_rows"]}条、{len(summary["admitted_strata"])}个分支×深度×来源条件。家族内标题、格式、顺序及13干预臂均不增加独立N。标签字符串首字母在发现/确认分开；数值可能重复，不能把家族切分称作全部数值未见。',
           '**设计与因果可见性。** 原创虚构材料带生成文法和合法来源声明。同一正文同时给出操作程序和具体记录值，两来源均有具体答案，消除旧版记录条件总是“未指定”的混杂。技术/普通标题与来源请求token长度配平；框架0/1用于拟合、2/3用于确认。发现只有A或B，确认含AB/BA和ABA/BAB；两步词语多重集相同，三步操作次数也变，不能叫纯顺序。body_end与answer_end均在题面相关信息出现后、答案token出现前；内容末token和答题末token身份分别固定。行为采用正确选项分数严格超过所有竞争选项，同分失败；试跑每标题格≥0.85、确认≥0.9，两个标题都通过才内部准入。被试跑排除者即使确认提高，也不追认准入。',
           '**完整确认行为（含被排除的内部条件）。**',
           table(['分支','深度','请求','标题','正确/N','准确率','过门'],[[r['branch'],r['depth'],r['mode'],r['title'],f"{r['correct']}/{r['n']}",r['accuracy'],r['passed']] for r in behavior['reports']['confirm']['cells']]),
           '完整确认1,595/1,920严格答对（83.07%）。标签原子程序复测达96.25%，但试跑失败的既定排除不撤销；两步程序标签62.5%/66.25%、数值38.75%/57.5%，三步标签61.25%/55%、数值20%/47.5%（普通/技术标题），均未到90%门。顺序对照与主确认合并的标签BAB为92.5%，同深度ABA为38.75%；这些已暴露子格只作反例分析，不能事后择优升级组合机制。',
           '**测量、拟合与输入边界。** 保存两个读出位置的全部41个残差边界×5,120坐标；作答位置B36/B39全部17,408单元G/V/A；B37/B38作答位置及四层正文位置仅保存指定72单元联合索引，并非完整MLP场。另保存两个位置的注意力/MLP输出、B36组的全有效token轨迹与embedding/h37前缀均值。BF16位模式以uint16无损压缩。14个岭回归预测器只用发现原子操作，三折家族交叉验证选择α∈{1,10,100,1000}，冻结后不重拟合。自然预测输入分别为32个A、完整h37、h37+历史均值、词汇哈希基线和平均embedding；目标为B39根G/V/A及h40−h37。完整残差/历史具有更大的信息预算。全词表KL预测都带入实测h37，不是仅凭文本的图谱保真度。',
           '**原生算子与公式。** 以下是已有网络计算定义及平滑局部近似，并非新发现的语言定律。令hℓ为作答末位第ℓ块入口残差，P为该位置之前的自然状态，aℓ为含Q/K归一化、RoPE、softmax及V/O的注意力写入，Nℓ为MLP前RMSNorm，Dℓ为down投影；Gℓ、Vℓ分别是gate/up投影，Aℓ=SiLU(Gℓ)⊙Vℓ。',
           r'$$r_\ell=h_\ell+a_\ell(h_\ell,P),\qquad h_{\ell+1}=r_\ell+D_\ell[\operatorname{SiLU}(G_\ell(N_\ell(r_\ell)))\odot V_\ell(N_\ell(r_\ell))].$$',
           r'$$J_\ell=(I+J_{\mathrm{MLP},\ell})(I+J_{\mathrm{Attn},\ell}),\qquad \widehat{\delta h_{40}}=J_{39}J_{38}J_{37}\delta h_{37}.$$',
           'δh37取实际BF16组改动后的残差位移；导数在自然未编辑基线上用FP32计算。已知自然下游状态和真实参数是局部算子的合法条件输入，编辑后下游目标不输入预测器。各层根G/V/A另沿相同链在对应计算边界读出。最终token编辑不能影响先前token，前缀K/V与整条早期状态逐位核对。对照包括零效应、只保留残差，以及同样冻结的F32(A36)、Fh(h37)、Fhp(h37,前缀均值)对相同干预的预测。',
           r'$$\Delta A=V\,\Delta\operatorname{SiLU}(G)+\operatorname{SiLU}(G)\,\Delta V+\Delta\operatorname{SiLU}(G)\,\Delta V+\epsilon_{\mathrm{BF16}}.$$',
           '该有限差恒等分解区分门、值、交叉项和原生舍入剩余；各项RMS可相互抵消，不能当作相加至100%的贡献率。32成员的φj≈∇margin·J39J38J37(−A36,jD36,:,j)为局部近似归因，竞争选项固定在基线；没有逐成员单独干预，也不证明每个成员必要。',
           '**同组原模型输出作用。** 在作答末位改变B36 down输入，13臂包括基线/1倍无操作、0/2倍、匹配组0/2倍、等实际残差写入范数的随机/负径向对照、后三层注意力/MLP/两者固定和B39根/匹配根恢复。主检验为“目标组0倍−匹配组0倍”的答案间隔变化，按家族聚合，配对t区间、α=0.05/12（未准入项也占原多重比较预算）。预测MSE比另按家族2,000次重抽样给95%区间；各比值门用冻结的点估计判据，不能说每项都通过了显著性检验。',
           table(['分支','深度','来源','0倍Δmargin','2倍Δmargin','0倍减匹配组','校正区间','通过'],[[r['branch'],r['depth'],r['mode'],summaries[r['branch'],r['depth'],r['mode'],'zero']['margin']['mean'],summaries[r['branch'],r['depth'],r['mode'],'double']['margin']['mean'],r['stats']['mean'],ci(r['stats']['ci']),r['passed']] for r in tail['primary_matched']]),
           f'主检验通过{summary["matched_primary_passed"]}/{summary["matched_primary_tested"]}个实际可检条件，不能用成功条件代替全部12个原定条件。预定每家族一条原子记录题的准确率比较为：{json.dumps(tail["paired_binary"],ensure_ascii=False)}。小分支准确率效应没有充分功效，分数改变不自动等于选错。标题效应、格式面板和所有臂的KL/严格准确率分别保留；标题差不显著不等于措辞等价。',
           table(['分支','范数对照','合格/N','最大相对误差'],[[r['branch'],r['arm'],f"{r['passed']}/{r['n']}",r['max_relative_error']] for r in tail['norm_control_qualification']]),
           'ABC下六个记录条件均通过匹配组主检验，数值原子程序未通过。目标组0倍的记录间隔下降约0.076–0.186 logit；预定160家族的原子记录准确率比较没有任何得失，不能据此认定该组对选对答案必要。随机等实际范数对照1,120/1,120合格，负径向只有1,078/1,120合格，最大范数误差2.397%超过2%门；未调整参数补过。数值三步背景下随机方向的间隔下降还大于目标组，不能把匹配组优势推广为对所有无关方向均特异。部分标题差的描述性区间排除零，不能声称措辞不变。',
           '**关键反例：相同题目改用XYZ答案代码。** 140条格式内部面板来自每分支20个既有确认家族，只取普通标题。正文、语义选项、正确项、长度和作答位置均配对相同，改变答案代码及其提示。该面板原方向的匹配组主检验0/7通过，但局部根效应预测仍14/14通过。下表严格配对ABC的同20家族，区间为描述性95%，不增加独立N。',
           table(['分支','深度','来源','操作','ABC效应','XYZ效应','配对变化','95%区间'],[[r['branch'],r['depth'],r['mode'],r['arm'],r['ABC']['mean'],r['XYZ']['mean'],r['XYZ_minus_ABC']['mean'],ci(r['XYZ_minus_ABC']['ci'])] for r in paths['paired_format']]),
           '数值记录0倍由ABC负效应变为XYZ正效应，两倍效应也反向；标签效应则减弱或变化。两种格式的基线答题均合格。因此本轮不支持把该组称作答案格式无关的共同语义运算。格式变化会同时改变输入状态与输出读出方向，不能仅凭反向就归因于lm_head。',
           '**事后输出读出分解。** 看到格式反向后另存计划及源码哈希，复用已暴露的140对题目，在同一BF16最终状态上交叉使用ABC和XYZ读出。这是代数诊断，交叉组合并非有效题面的自然行为，使用编辑后的终态，不是新的留出预测。令E(s,c)为状态来源s、代码读出c的干预后间隔减原间隔；s,c∈{A,X}。按对称分摊约定，代码项C=[E(A,X)−E(A,A)+E(X,X)−E(X,A)]/2，状态项S=[E(X,A)−E(A,A)+E(X,X)−E(A,X)]/2，有C+S=E(X,X)−E(A,A)。这是恒等式而非唯一因果中介。',
           table(['分支','深度','来源','操作','代码项','状态项','总变化','交互项'],[[r['branch'],r['depth'],r['mode'],r['arm'],*[r['statistics'][k]['mean'] for k in ['code_component','state_component','total','interaction']]] for r in diagnostic['summary']]),
           f'原始对角读出{diagnostic["diagonal_replays"]}次逐位复现，最大候选分数差{diagnostic["diagonal_max_abs_error"]}；280行两项之和逐条等于总变化。状态项包含基线归一化、B36响应和后续传播，不能据此确定哪一层形成代码依赖。详见readout-diagnostic.json，独立新增样本为0。',
           '例：数值原子记录0倍的总变化+0.2875，被对称分为代码项+0.13125（描述性95%区间[0.01361,0.24889]）及状态项+0.15625（[0.06559,0.24691]）；数值两、三步背景的状态项也为正且名义区间排除零。不能只把反向归结为输出头换了方向。标签0倍的2×2交互为−0.2531至−0.3250，说明读出变化与状态变化不可视作普遍独立的加法贡献；交互同时包含margin最大竞争项选择的非线性，不能直接解释为一个神经元交互机制。',
           '**状态补全的留出结果。** 根预测门：MSE≤0.9×组A、≤0.9×词汇基线、≤0.5×均值。各读出/方法单独记账，不能把表中多个方法当作独立样本。',
           table(['分支','深度','来源','读出','输入','MSE/组A','MSE/词汇','MSE/均值','过门'],[[r['branch'],r['depth'],r['mode'],r['readout'],r['method'],r['ratios']['selectedA']['ratio'],r['ratios']['lexical']['ratio'],r['ratios']['mean']['ratio'],r['state_completion_passed']] for r in natural]),
           '完整h37通过13/14个根预测条件；失败的是数值记录、三步背景的作答位置。h37+前缀均值通过6/7个条件，但相对单独h37达到至少10%改善的只有数值三步背景1/7。标签三步背景反而从通过变为略超均值门（MSE/均值约0.501）。这不说明历史状态无用，只说明这个均值摘要没有普遍增量。对h40−h37更新，组A、h37、h37+前缀均值各7/7过门，词汇0/7、embedding3/7；不同目标的结果不能互相替代。',
           '**真实改动的局部预测。** 在同一材料、同一组干预上，根效应的误差按发现集根坐标SD标准化；要求实测RMS≥0.05，局部MSE≤0.25×零效应MSE且≤0.9×仅残差MSE。未达效应下限标为不可充分检验。',
           table(['分支','深度','来源','剂量','标准化实测RMS','MSE/零效应','MSE/残差','过门'],[[r['branch'],r['depth'],r['mode'],r['dose'],r['rms_actual_standardized'],r['ratios']['zero']['ratio'],r['ratios']['residual']['ratio'],r['passed']] for r in tail['counterfactual_prediction']]),
           f'局部反事实门通过{summary["local_counterfactual_passed"]}/{summary["local_counterfactual_tested"]}。完整逐格区间、拟合基线、最终hidden误差、门/值分解与路径固定后的间隔恢复见confirm-tail-report.json；正常状态预测与反事实预测不能互换。根恢复如果改变结果，只证明所操纵路径在该有限设计中的作用，不证明独有中介或完整复杂语义。',
           '本轮14/14局部预测通过，根G/V/A的MSE为零效应基线3.69%–5.31%，为仅残差基线4.81%–9.99%；最终h40效应MSE为零效应2.73%–4.05%。此前只由自然状态拟合的32单元预测器反事实MSE却为零效应的10.44–19.75倍，h37预测器为0.590–0.761倍。原生局部算子使用完整后三层参数和自然状态，信息预算不同；改善说明补入计算来源后可以解释传播，不能包装成同预算方法胜出或新的语言编码定律。',
           '以实际编辑后的全词表分布为目标，完整局部算子的平均KL为0.000612–0.001156，是保持自然输出基线KL的8.09%–14.54%；该指标单独报告，没有事后增设科学通过门。它依赖真实基线状态、实际起始残差位移与完整参数，不等同于从外部语言图谱独立生成模型输出。',
           '**路径与根中介的负结果。** 将B37–B39注意力和MLP的输出都固定为自然值时，记录条件的组0倍间隔效应仅余−0.0203至+0.0203，原效应为−0.0758至−0.1859，提示主要分差变化需要下游处理。单独固定路径可能过度恢复，各操作不能按比例相加。把B39八根的A恢复成自然值，平均间隔恢复仅−0.000781至+0.002344，所有95%区间含零，且与匹配根接近；没有证据表明这八根承载了主要净输出作用。可预测的根响应与实际中介不能画等号，也不能由本次末位A恢复否认它们在所有语境的作用。',
           '**条件化输出保真度（KL越小越好）。**',
           table(['分支','深度','来源','identity','均值更新','组A','h37','h37+前缀','词汇','embedding'],[[*k,*[outby[*k,m]['KL'] for m in ['identity','mean','selectedA','hidden','hidden_prefix','lexical','embedding']]] for k in sorted({(r['branch'],r['depth'],r['mode']) for r in output['summary']})]),
           '**严格审视。** 本轮最主要硬伤是“执行程序组合”的行为未通过预设校准，因此不能完成更复杂特征形成的联合检验。记录读取中遇到未见背景组合，只检验干扰条件推广。强制选项分数不能替代自然自由生成；没有声称模型本身普遍不会组合。技术标题控制只排除两组特定等长词语，来源词仍不同，分词长度和模板有限；不同分支共享答题格式、选项与来源判断仍是替代解释。只编辑最后位置、单一快照、两个领域，未测试全历史必要性、其他模型、训练中形成、广泛任务副作用或独立节点。完整h37和局部算子享有额外状态/参数，不与稀疏组具有相同信息预算；本轮尚无同预算SAE/transcoder比较。有限剂量线性化有非线性和BF16误差，等范数不合格者不得充作合格负控制。主检验及其失败不允许确认后选单位、阈值或条件补过。',
           '**三图谱增量与理论判断。** 外部图谱登记来源选择、标题、程序深度与顺序及行为失败格；内部图谱保留原生索引、两位置全HiddenState视图、32张原生单元卡与1张组卡；关联图的6类边分开标记观察、原生参数来源、留出状态预测、条件化局部传播、组干预和路径恢复。特征卡中自然语料统计与成员独立因果状态明确待检，不贴未经证实的共同语义标签。RDC既有部分映射与类型化三图谱定义保持不变，没有新统一数学定律、脑机制或AGI结论。',
           '**第一性原理洞察与下一大阶段。** 要解释有限参数的可组合语言能力，需要同时确定外部关系、足够内部状态、可复用运算以及输出实际使用。当前结果把两类问题分开：沿真实网络预测扰动传播已有进展；把这条传播解释成特定共享语义仍受格式反例与行为失败限制。即使语义答案不变，输出代码改变也能改变同组干预的符号，所以应检验跨读出的关系等变性，而不能用一个logit方向定义语义。下一大阶段应作为完整“可执行语义组合验收”推进：①在独立新家族上校准简化且平衡的程序执行、直接答案与自由生成，将答案代码随机轮换并独立封存未见格式和未见关系组合；②在同一材料和可比预算下比较原生组、包含历史来源的状态和稀疏字典，显式给出每步中间语义真值，检验T_B∘T_A及顺序反例、跨读出对应关系，而不把深度效应当作复杂特征；③只有行为及留出预测过门后，分别操纵B36写入、QK选择、V/O写入与后三层MLP，以实际净输出中介而非根响应大小选择后续路径，并预测有限改动、恢复和其他任务副作用。局部链验收后才连接更大机制图；训练期形成另需合法检查点或受控训练。',
           f'**工程验收、资源与复算。** 数学工程测试5项、准入测试2项通过，另有原始HF/BF16对齐和数值试跑。完整协议{protocol["tests"]}项，失败{len(protocol["failures"])}、错误{len(protocol["errors"])}，详情protocol-regression.json；沙盒本机接口停滞的自有测试已停止并保留脱敏日志，之后同套测试在允许回环接口环境执行。原生技术审计及本地报告桌面/手机检查通过，不等于独立复现。CUDA分配日志峰值{cuda/2**30:.4f}GiB，DXGI进程共享峰值{shared/2**20:.2f}MiB；所有新增路径在D盘，通过统一磁盘门并保留20GiB。原生采集数据（复用试跑只计一次）{native:,}字节，后续干预数组另留存。保留用途为局部状态/路径复核，复查日2026-12-13。本轮收尾清理{d["cleanup.json"]["removed_file_bytes"]:,}文件字节，早期v1重复/无后续用途原场清理另见predecessor-cleanup-receipt.json；唯一正式原场、v2只读试跑、全部失败成绩与日志保留。并行其他任务使文件字节与全盘空闲变化不等。',
           '**资源中断与恢复披露。** 完整协议第一次在沙盒本机接口停滞；允许回环后，快照测试将真实research目录复制进隔离research_library，预估200MiB不足，实际临时复制3,146,889,259字节，触及20GiB保留线。确认在第864输入分片、第14块被统一磁盘门停止；864条已提交分片未改动。停止自有测试并删除已核对归属的复制件，正式原件不动，恢复约22.53GiB；重新核算剩余模型/分析/测试合计峰值1,300MiB，通过门后续跑。协议使用现有AI2050_RESEARCH_DIR指向空测试目录，315项测试和生产代码未修改，并补上测试监控线程触发磁盘门时终止整个测试的保护。原冻结科研源码、材料、阈值与预测器未改变。此为资源预估不足及恢复记录，不能只写最终测试通过。',
           f'脚本全部位于tests/glm5/b36_composition14*，入口及实际命令见b36_composition14_README.md；结果 `{basename}/report.html`，原理与逐项产物见同目录各冻结计划、audit.json、feature-cards.json、mechanism-graph.json、delivery-source.zip与manifest.json。采用逐行刷新进度，GPU租约冲突仅串行等待。没有新模型、服务器发布或正式中心登记。',
           '通俗说明：沿着真实权重和状态，已经能较准确地算出“关掉这批单元后，变化怎样传下去”。但只是把答案编号ABC改成XYZ，同一操作就可能从压低正确答案变成抬高它；而恢复原先选的八个末层根几乎救不回分差。因此找到了可计算的传播过程，还没有找到与答案写法无关的共同语义齿轮。程序组合的行为校准也未通过，不能宣称已证明神经元逐层组成复杂语言特征。']
        return '\n\n'.join(p).replace('[ABBR]',f'[{stamp}]')
    primary=append(B.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md',main)
    def index(phase):
        return f'## Phase {phase}: Qwen3-14B B36固定32单元状态与传播验收索引 [{stamp}]\n\n详见GLM5 Phase {primary["phase"]}，RUN `{B.RUN}`。固定旧B36组，在两来源具体答案、等长标题与未参与拟合的操作序列上检验；v2组合行为试跑失败，v3确认前缩小内部范围，其余条件保留行为复测。160个确认家族，内部最终准入{summary["admitted_rows"]}条。ABC匹配组输出主检验{summary["matched_primary_passed"]}/{summary["matched_primary_tested"]}、局部根效应预测{summary["local_counterfactual_passed"]}/{summary["local_counterfactual_tested"]}通过。关键反例：同题换成XYZ，数值记录组干预效应反向，原主检验0/7通过；事后交叉读出分解同时发现状态与代码作用，仅为暴露材料上的代数诊断。B39八根A恢复近零，不能认定主要净输出中介。完整残差/历史输入和自然下游条件化局部算子，与共享语义及复杂特征形成分开记账；组合机制未获准入，没有正式晋级。原生审计和报告检查通过，协议{protocol["tests"]-len(protocol["failures"])-len(protocol["errors"])}/{protocol["tests"]}。报告`{basename}/report.html`，32张单元卡、组卡、6类关联边及反例保留。RDC公式不变，下一阶段优先校准语义组合、答案代码轮换和中间真值，再对足够状态、有限改动及副作用一体化验收。'
    secondary=append(B.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md',index)
    B.save('memo-append.json',dict(receipts=[primary,secondary],phase=primary['phase'],index_phase=secondary['phase'],source_sha256=B.file_sha(__file__),inputs={n:B.file_sha(B.OUT/n) for n in d}))
    print('Append-only memo',primary['phase'],'index',secondary['phase'],flush=True)


if __name__=='__main__':run()
