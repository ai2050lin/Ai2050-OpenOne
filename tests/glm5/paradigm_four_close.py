"""Verify and archive the four-paradigm phase with exact source and parameter identities."""

from hashlib import sha256
from pathlib import Path
from shutil import rmtree
import os
import time

import numpy as np

import paradigm_four_run as X
import paradigm_four_report as V
import paradigm_four_discovery as D
import paradigm_four_material as M
import paradigm_routes_audit as prior_audit
from b36_semantic14_memo import append
from deploy.storage_policy import checked_dir,check_write


R,C,OUT=X.R,X.C,X.OUT
TITLES=('语言','风格','逻辑取值','括号语法')


def figures(confirmation):
    os.environ['MPLCONFIGDIR']=str(checked_dir(X.TEMP/'matplotlib',2**20,operation='四范式图表缓存'))
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with np.load(OUT/'all-native-ranking.npz',allow_pickle=False) as z:
        first=np.stack((z['gate_fraction'],z['up_fraction']))
        second=np.concatenate((z['gate_link'],z['up_link']))
    fig,axes=plt.subplots(2,1,figsize=(12,4.8),constrained_layout=True)
    for ax,value,labels,title in zip(axes,(first,second),(('G','U'),('G-H25','G-H18','U-H25','U-H18')),
                                      ('Conditional response at all B30 units','Original normalized read-weight connection')):
        visual=ax.imshow(value,aspect='auto',interpolation='none',cmap='magma',vmin=0)
        ax.set_yticks(range(len(labels)),labels)
        ax.set_title(title)
        ax.set_xlabel('Original B30 neuron index')
        fig.colorbar(visual,ax=ax)
    fig.savefig(check_write(OUT/'b30-native-map.png',3*2**20,operation='全坐标范式结构图'),dpi=150)
    plt.close(fig)
    values=confirmation['fusion']['per_cell_KL']
    fig,axis=plt.subplots(figsize=(8,4.3),constrained_layout=True)
    positions=np.arange(len(values))
    for method,i,shift in [('Baseline',0,-.24),('Additive',1,0),('Pairwise',2,.24)]:
        axis.bar(positions+shift,[row[i] for row in values],.22,label=method)
    axis.set_xticks(positions,[''.join(map(str,M.LEVELS[i])) for i in X.COMBINED])
    axis.set_ylabel('Full-vocabulary KL')
    axis.set_xlabel('Language / format / factual selection / delimiter')
    axis.legend(frameon=False)
    fig.savefig(check_write(OUT/'heldout-fusion.png',2**20,operation='未见组合融合图'),dpi=150)
    plt.close(fig)


def gate_interaction(confirmation,parameters,candidate):
    from scipy.special import expit

    samples=[]
    for name in confirmation['parts']:
        with np.load(OUT/name,allow_pickle=False) as file:
            if 'native_gua' in file.files:
                samples.append(file['native_gua'])
    counts=np.concatenate(samples)
    g=counts[...,0].astype(np.float64)
    up=counts[...,1].astype(np.float64)
    store=counts[...,2].astype(np.float64)
    gate=g*expit(g)
    contrast=lambda v:v[:,4]-v[:,2]-v[:,3]+v[:,0]
    g_delta=gate-gate[:,[0]]
    u_delta=up-up[:,[0]]
    terms=np.stack((up[:,0]*contrast(gate),gate[:,0]*contrast(up),contrast(g_delta*u_delta)),-1)
    interaction=contrast(gate*up)
    assert np.max(np.abs(terms.sum(-1)-interaction))<1.e-10
    written=V.restore_bits(parameters['down_columns']).astype(np.float64)
    output=(V.restore_bits(parameters['readout.print'])-V.restore_bits(parameters['readout.The'])).astype(np.float64)
    write_projection=written@output
    chosen=terms[:,:32]
    projected=(chosen*write_projection[None,:32,None]).sum(1)
    report=dict(rows=len(counts),native_unit_ids=candidate['selected'],discovery_frozen_units=True,
                ablation_endpoints=['reference Value','request Value','joint'],
                mean_abs_per_unit=abs(chosen).mean(axis=(0,1)).tolist(),
                mean_abs_projected_group=abs(projected).mean(0).tolist(),
                bf16_interaction_absolute_error=float(abs(contrast(store)-interaction).mean()),
                bf16_interaction_max_absolute_error=float(abs(contrast(store)-interaction).max()),
                first_unit_term_means=terms[:,0].mean(0).tolist(),
                local_scope='Descriptive native G/U path decomposition over active interventions, including technically excluded samples. Prints mathematical real-valued SwiGLU terms and BF16 residual separately. Not final normalized output effects.')
    R.write('gate-interaction.json',report)
    return report


def noun_sensitivity(confirmation):
    from transformers import AutoTokenizer

    token=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    material=list(M.expanded(token,'confirm'))
    variants=V.data_load(list(confirmation['parts']))
    assert list(variants['ids'])==[row['id'] for row in material]
    nouns=np.asarray([row['noun'] for row in material[::16]])
    raw=variants['fusion']
    difference=(raw[:,:,1,0]-raw[:,:,2,0]).mean(1)
    by_noun={word:float(difference[nouns==word].mean()) for word in np.unique(nouns)}
    leave_one_out={word:float(difference[nouns!=word].mean()) for word in by_noun}
    measures=dict(nouns=len(by_noun),noun_balanced=V.summary(list(by_noun.values())),
                  leave_one_noun_out_range=[min(leave_one_out.values()),max(leave_one_out.values())],
                  per_noun=by_noun)
    margins=variants['margin'].reshape(-1,16,4)
    means={}
    from itertools import combinations

    for i,j in combinations(range(4),2):
        a,b=8>>i,8>>j
        coefficient=margins[:,a+b]-margins[:,a]-margins[:,b]+margins[:,0]
        means[f'{X.FACTORS[i]}×{X.FACTORS[j]}']=[
            V.summary(coefficient[:,k]) for k in range(4)]
    measures['paired_margin_coefficients']=means
    # Each gain summand uses identical family weight. Negative entries can
    # occur when some combinations favor the additive model.
    cell_gains=(raw[:,:,1,0]-raw[:,:,2,0]).mean(0)
    measures['per_high_order_cell_gain']=cell_gains.tolist()
    measures['largest_gain_cell']=list(M.LEVELS[X.COMBINED[int(cell_gains.argmax())]])
    measures['largest_gain_cell_share']=float(cell_gains.max()/max(cell_gains.sum(),1e-9))
    R.write('noun-sensitivity.json',measures)
    return measures


def body(plan,candidate,old,pilot,new,parameter_hash,progress,interaction,redundancy,sensitivity):
    from record_update_report import table,show

    report=[]
    for i,axis in enumerate(X.FACTORS):
        p=pilot['behavior'][axis];h=new['behavior'][axis]
        report.append([TITLES[i],f"{p['minimum_stratum']:.1%}",p['passed'],f"{h['minimum_stratum']:.1%}",
                       f"{h['absolute']:.1%}",show(h['paired'])])
    errors=[[k,show(new['fusion'][k]['kl']),show(new['fusion'][k]['top1'])] for k in ('base','additive','pairwise')]
    forecast=[]
    for k in ('text','source','native','random'):
        forecast.append([k,*[show(new['forecast'][k][axis]) for axis in X.FACTORS]])
    ca=new['causal'];lead=candidate['cards'][0];gain=new['fusion']['gain']
    fmt=new['behavior']['format']
    peak=max((row.get('cuda_peak_bytes',0) for row in progress))/2**30
    shared=max((row.get('shared_gpu_bytes',0) for row in progress))/2**20
    old_cell=old['recorded']['cells']
    rand_unique=old['draws_better_than_selected']
    exceptions=new['fusion_counterexamples'][:3]
    best_path=new['path']
    cleanup='本轮临时字体配置在保存图片后清理，原有全场继续保留。'
    redundant=[]
    for name,group in redundancy['results'].items():
        metrics=group['metrics']
        redundant.append([name,group['residual_effective_rank'],*[show(metrics[k]['gain_over_text_and_head']) for k in
              ('original','only_cell_centroids','remove_training_cell_centroids')]])
    return f'''### 主张、样本与冻结顺序

运行 {M.RUN}，Qwen3-14B 原始 BF16，快照 {plan['snapshot']['snapshot']}。复用 Phase 2964 的发现、已开封旧确认原生场及冻结预测，不把本轮的旧材料复算计入新样本。以个人可复核的动作连接本轮两条路线：先按真实注意力来源、G/U 的条件变化、原生读写权重拟合候选，再在从未运行的 128 个独立新家族验证。每个家族包含16个组合，共2048个视图，另有16个试跑家族。128个联合锁箱家族保持封存。新家族用新名词、数量配对和独立表述，四因子是输出英文／中文、论文式／Python、从初始／修订计数取值、圆／方括号。复测中归档示例分文体平衡。四类信息在同一自由输出首词的读出前已出现。自由首词过后另使用预先指定的前缀，测量消息语言、首个左括号及有效计数。这些位置的结果只反映给定前缀后的选择。逐家族核对首词的共同 token 与源位置，并对32视图保存首词及计数两个位置的41层BF16原生状态。

初始预算以独立家族二值率差14个百分点、失谐0.20、双侧 α=.01 规划，128家族的标称功效为 {plan['power']['achieved_power']:.3f}。KL、亚组项和小剂量原生效应独立报告实际区间。字段、权重、候选、代码及材料 SHA-256 绑定见 plan.json 和 candidate.json。试跑线性预测器在新确认开始前落盘。

### 随机单元为什么更善于预测

旧面板精确首词全词表 KL 为强写出 {old['recorded']['mlp_selected_KL']:.4f}、原同层随机 {old['recorded']['random_KL']:.4f}。强写出减随机 {show(old['recorded']['all'])}，按旧八格差序为 {', '.join(f'{x:.3f}' for x in old_cell)}。随机优势主要集中在部分条件，最后一格方向反转。固定相同文本和头协变量重拟合，25套同层32维随机集合中 {rand_unique} 套的终层归一状态 MSE 优于强写出组。强写出组与原随机组中，按格式、示例和语言八格解释的激活方差分别为 {old['explained_by_factor_cells']['selected']:.1%} 和 {old['explained_by_factor_cells']['random']:.1%}，按标准化奇异值估计的有效秩为 {old['activation_effective_rank']['selected']:.2f} 和 {old['activation_effective_rank']['random']:.2f}。旧筛选最大化单一 print−The 局部写出，得到高度共变的单元，重复了文本已知的格信息。随机组包含较多剩余变化。这是由多随机重采样与表示统计支持的解释。正交化、等信息量和其他重建目标仍须单独试验其因果贡献。

旧资料复算中，用发现集均值将每份激活拆为格均值与格内余量。下表的三列为各方案相比文本加头的终层状态 MSE 改善，正值表示误差降低。格均值只由发现集拟合。

{table(['旧单元组','格内有效秩','原激活','只保留格均值','只保留格内余量'],redundant)}

只保留格均值后两组改善接近。随机格内余量单独使用接近零改善，强写出格内余量单独使用使 MSE 明显上升。随机组优势因此与分布更宽、原始激活中条件信号和格内信号的组合有关。以格内余量单独支撑随机组优势的解释缺少证据。格均值移除也改变了输入协方差和岭回归惩罚作用，细分原因仍需独立预注册。

### 原生路径筛选与公式

候选范围固定 B30。源 R 的注意力读量 m_R=Σ_(i∈R) softmax(q·k_i/√128)，写入 w_R=W_O^h Σ_(i∈R) α_i v_i。本轮先保留 H25，再按发现集请求／示例的 QK 质量比变化选不同 GQA 组的 H{candidate['heads'][1]}。对单元 j 保留 G_j=g_jᵀN(x)、U_j=u_jᵀN(x)、A_j=SiLU(G_j)U_j 和写出坐标 W_down[k,j]A_j。头到单元的候选连接用原参数乘积 g_jᵀdiag(γ)W_O^h 和 u_jᵀdiag(γ)W_O^h 的行归一范数，其中 γ 是注意力后 RMSNorm 的尺度。这项只刻画权重连接，RMSNorm 分母及后续非线性在原生干预重算。联合候选分数为 G/U 条件选择性、多次未混用家族半集的输出残差信号一致度和原生连接强度的秩乘积，全17,408单位打分保留。

排名首位 B30/U{lead['unit']}，最大模 gate/up 标量分别位于列 {lead['primary_gate_coordinate']}、{lead['primary_up_coordinate']}。两头 gate 连接 {lead['gate_links']}，up 连接 {lead['up_links']}，原 BF16 行、列与参数索引可查 native-parameters.npz，SHA {parameter_hash}。确认时两头的请求/示例质量对数比分别按四因子测量。风格主对比 {', '.join(show(x) for x in best_path['ratios']['format'])}。这些结果定位候选的条件读写。

### 真机行为与未见组合预测

{table(['范式','试跑最低配对格','试跑门','确认最低配对格','确认绝对分差方向正确','确认配对分差变化 [99% CI]'],report)}

Python 独立首词 print 命中 {fmt['first_token_hit_1']:.1%}，论文侧非print率 {fmt['first_token_hit_0']:.1%}。括号及计数比较从已给定的答案前缀开始，记录模型对两个竞争答案的 logit 差。这个设计保留全部模型反应和失败。

令 z(S) 为家族只开启 S 中因子的实测首词 logits。只读零因子及单因子，预测 z_add(S)=z(∅)+Σ_i[z(i)−z(∅)]。再加每对的 z(ij)−z(i)−z(j)+z(∅) 得 z_pair(S)。四个三因子格和一个四因子格的目标 logits 均被留出，评估 KL(p_model∥p_hat) 和全词表首选。预测前拥有同家族低阶条件的真实前向，因此属于已测单轴与双轴如何组合的预测。逐家族等权聚合。

{table(['模型','全词表 KL [99% CI]','首选命中 [99% CI]'],errors)}

加入双因子交互后的相对 KL 改善 {gain['relative']:.2%}，家族差 {show(gain)}，冻结门 {gain['passed']}。代表性反例 {exceptions}。另用试跑训练固定岭回归，在确认集对比文本、两头源质量、源质量加32个筛选单元及源质量加32个同块随机单元。这是等单元数比较，四幅量的目标尺度各异。

家族复用 {sensitivity['nouns']} 个确认名词。按词等权汇总 KL 改善为 {show(sensitivity['noun_balanced'])}。每次整词剔除一类后的家族均值范围为 {sensitivity['leave_one_noun_out_range']}，用于检验词汇重复对结果的影响。新表述仅覆盖本轮冻结的确认句式。五个目标格的 KL 改善分别为 {sensitivity['per_high_order_cell_gain']}，其中 {sensitivity['largest_gain_cell']} 格贡献总改善的 {sensitivity['largest_gain_cell_share']:.1%}。同一材料上的六个双因素原生分差交互在 noun-sensitivity.json。

{table(['预测输入',*TITLES],forecast)}

### 原模型竞争与下游接收

64个预先指定的确认家族，每家族16视图，共1024视图。只改 B30 所选两头在自由输出首位的来源地址或 Value，保持其余前缀。原生分支重算后续层，并以同位 sham 核对。有 {ca['families']}/{ca['total_families']} 家族的16视图首选完全与未改模型对齐，{ca['sham_views']} 视图对齐。所有比较仅取完整对齐家族。归档与请求格式冲突时，阻断归档朝请求方向的变化 {show(ca['archive_removed_toward_requested'])}，阻断请求的分差损失 {show(ca['request_removal_loss'])}，联合效应相对两个单项之和的偏离 {show(ca['joint_nonadditivity'])}。实际首选变化数 {ca['counts']}，全原生恢复核验 {ca['native_full_restore_equal']}。冻结的功能竞争判据 {ca['functional_competition_passed']}。

单查询分支与整段前向有6个首选失配视图，分布在6个家族，导致完整技术样本低于60。完整原生恢复在单查询形状内部逐词表核验一致。两种运行形状的差异原因尚需逐个核对。对照保留原生源写入量，对同头的两个来源并未强制等幅，分差方向只解释原剂量下的作用。

联合阻断后，选中组的平均 |ΔG|、|ΔU|、|ΔA| 为 {ca['receiver_abs_GUA_change']['selected']}，随机组为 {ca['receiver_abs_GUA_change']['random']}。原生恢复32个 A 标量后与恢复随机组的分差之差 {show(ca['selected_rescue_vs_random'])}，写入剂量误差 P95 为 {ca['rescue_dose_p95']}，匹配门 {ca['matched_rescue']}。只有剂量合格才可评估等幅恢复优越性。局部连接和真实接收变化提供结构候选，中间态路径特异性还需要更精确的对照。

对门 s=SiLU(G)、值 u 以及每种来源阻断下相对原生 sham 的改变量 δs、δu，有 A−A₀=u₀δs+s₀δu+δsδu。将三项各自取联合减两个单项加基线，得精确实数版双源交互，另对账 BF16 舍入。选中组的单元平均绝对交互量按门、值、乘积分别为 {interaction['mean_abs_per_unit']}，与存盘 BF16 交互差的平均绝对值为 {interaction['bf16_interaction_absolute_error']:.6f}。真实单位序号和投影结果在 gate-interaction.json。

### 理论进展、限制与阶段任务

将外部图谱的范式变量和源 token、内部图谱的原生 QK 与 G/U 支路以及关联边的预测和干预结果落到同一组材料。可检验的第一性原理是固定权重允许多个来源通过共同残差进入可复用的条件计算。主效应、归一化竞争、V 写入和 SwiGLU 交互对输出的贡献可以分别检验。双因子近似的有效范围由实际留出指标决定，RDC 保持为有条件和适用域的候选转移框架。原生架构参见 [QK/OV 分解](https://transformer-circuits.pub/2021/framework/index.html) 和 [Goodfire 神经元研究](https://www.goodfire.com/research/neurons-as-an-interpretable-basis)。

本轮的 E 读出只覆盖生成入口，语法是括号槽位，逻辑是初始和修订事实取值，文体也受固定答案格式影响。确认选择用时序明确的早期信号，完整残差仅保存固定两家族两位置。候选集中在单块。两个来源编辑联合变化，只能界定该支路在测得的任务中的行为。下一大阶段应完成连接的端到端验收。在未开封材料建立词义、角色、逻辑关系及句法作用域四层转移，包含词汇同义改写、跨句式、未见范式深组合及自然生成。先用同预算、去冗余的原生特征与随机、稀疏字典或相同宽度探针对账，寻找跨层真正被重复使用的读写关系；再用更细的单坐标、单神经元、单参数版本和独立恢复检验交换来源是否产生预言的反事实。按原模型完整输出误差、其他任务副作用和独立节点复算记录证据等级。

### 实际执行、复算与磁盘

入口 tests/glm5/paradigm_four_discovery.py、paradigm_four_run.py freeze、all、paradigm_four_close.py，结果 tests/glm5/result/{OUT.name}。真实前向采用 CUDA 串行单模型，最大 CUDA 分配 {peak:.2f} GiB、DXGI 共享峰值 {shared:.0f} MiB，日志逐批写盘。{cleanup}'''


def main():
    R.OUT=OUT
    X.base.OUT=OUT
    plan=X.verify()
    new=R.read('confirm-summary.json')
    pilot=R.read('pilot-summary.json')
    old=R.read('retrospective.json')
    candidate=R.read('candidate.json')
    assert not list(OUT.glob('lockbox-*'))
    for split in (pilot,new):
        assert split['plan']==C.A.digest(OUT/'plan.json')
        for name,digest in split['parts'].items():
            assert C.A.digest(OUT/name)==digest
    for name,digest in candidate['discovery'].items():
        assert C.A.digest(D.OLD/name)==digest
    assert (OUT/'pilot-predictor.json').stat().st_mtime_ns<min(q.stat().st_mtime_ns for q in OUT.glob('confirm-??.json'))
    q=candidate['block'];h,k=candidate['heads']
    parameters=dict(units=[[q,u] for u in candidate['selected']],random_units=[[q,u] for u in candidate['random']],head=[q,h],control_head=k)
    arrays,param_hash=prior_audit.extract_parameters(parameters)
    assert arrays['gate_rows'].shape==(64,5120)
    figures(new)
    import json

    logs=[json.loads(row) for row in (OUT/'progress.jsonl').read_text(encoding='utf-8').splitlines()]
    interaction=gate_interaction(new,arrays,candidate)
    sensitivity=noun_sensitivity(new)
    text=body(plan,candidate,old,pilot,new,param_hash,logs,interaction,R.read('redundancy-diagnostic.json'),sensitivity)
    if (OUT/'campaign-first-lease.log').exists():
        text+='\n\n启动时 GPU 被项目另一项运行占用，互斥记录保留在 campaign-first-lease.log。释放后沿用相同冻结计划串行运行。'
    check_write(OUT/'report.md',len(text.encode())+4096,operation='四范式报告').write_text('# '+V.TITLE+'\n\n'+text,encoding='utf-8')
    record=R.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
    assert V.TITLE not in record.read_text(encoding='utf-8-sig')
    added=append(check_write(record,len(text.encode())+4096,operation='追加四范式阶段密记'),V.TITLE,text)
    cleaned=0
    if X.TEMP.is_dir():
        target=X.TEMP
        assert target.resolve()==target and target.parent==(R.ROOT/'tests/glm5_temp').resolve()
        children=[target,*target.rglob('*')]
        assert all(not p.is_symlink() and not p.is_junction() and p.resolve().is_relative_to(target) for p in children)
        cleaned=sum(p.stat().st_size for p in children if p.is_file())
        rmtree(target)
    R.write('memo-receipt.json',dict(record=added,cleanup={'path':str(X.TEMP),'bytes':cleaned},parameters=param_hash))
    sources=plan['sources'].copy()
    for file in (Path(__file__),Path(prior_audit.__file__),Path(__file__).with_name('paradigm_four_redundancy.py')):
        sources[file.relative_to(R.ROOT).as_posix()]=sha256(file.read_bytes()).hexdigest()
    R.write('manifest.json',dict(sources=sources,files={p.name:dict(bytes=p.stat().st_size,sha256=C.A.digest(p))
             for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}))
    print('Appended and verified Phase',added['phase'],flush=True)


if __name__=='__main__':
    main()
