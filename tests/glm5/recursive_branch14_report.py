"""Readable report and scientific plots from immutable real result files."""
import html
from pathlib import Path
import time
import numpy as np
import recursive_branch14 as R

NAMES={'structure':'关系配对 · B26/U13099','procedure':'次序变换 · B19/U64'}
SPLITS={'new_words':'新词','new_frames':'新词＋新句式','new_domains':'新词＋新领域背景'}


def f(v):return '—' if v is None else f'{v:.4g}'
def esc(v):return html.escape(str(v))


def table(headers,rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def plots(report,interventions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    for ax,(h,name) in zip(axes,R.CANDIDATES.items()):
        rr=[r for r in report['results'] if r['hypothesis']==h and 'contrasts' in r]
        for offset,(contrast,color,label) in enumerate([('relation','#1565c0','Relation'),('title','#bf4e16','Technical title'),('operative','#14805e','Instruction vs reference')]):
            x=np.arange(len(rr))+(offset-1)*.17;y=[r['contrasts']['G'][contrast]['mean'] for r in rr]
            err=np.array([[r['contrasts']['G'][contrast]['mean']-r['contrasts']['G'][contrast]['ci'][0] for r in rr],
                          [r['contrasts']['G'][contrast]['ci'][1]-r['contrasts']['G'][contrast]['mean'] for r in rr]])
            if rr:ax.errorbar(x,y,yerr=err,fmt='o',capsize=3,color=color,label=label)
        ax.axhline(0,color='#777',lw=.7);ax.set_xticks(np.arange(len(rr)),['Words','Words + frame','Words + domain'][:len(rr)])
        ax.set_title(f'{h}: B{name[0]}/U{name[1]}');ax.set_ylabel('Raw gate G paired family contrast');ax.legend(fontsize=8)
    fig.suptitle('Content-end readout, before question/options; intervals are family-based')
    fig.savefig(R.OUT/'response-contrasts.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    for ax,h in zip(axes,R.CANDIDATES):
        rr=[r for r in report['backward_chains'] if r['hypothesis']==h and 'free_rollout' in r]
        x=np.arange(len(rr));ax.bar(x,[r['free_rollout']['ratio_to_mean'] for r in rr],color='#526fa4')
        ax.axhline(1,color='#bf4e16',ls='--',label='Mean baseline');ax.axhline(.9,color='#14805e',ls=':',label='Frozen 0.9 threshold')
        ax.set_xticks(x,['Words','Words + frame','Words + domain'][:len(rr)]);ax.set_title(h);ax.set_ylabel('Free B36 to B39 standardized MSE / mean MSE');ax.legend(fontsize=8)
    fig.savefig(R.OUT/'backward-rollout.png',dpi=170);plt.close(fig)
    if interventions:
        fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
        for ax,h in zip(axes,R.CANDIDATES):
            rr=[r for r in interventions['results'] if r['hypothesis']==h and r['scope']=='target' and r['role']=='all' and r['arm'] in ['single_0','group_0','matched_group_0']]
            x=np.arange(len(rr));y=[r['margin_change']['mean'] for r in rr]
            error=np.array([[r['margin_change']['mean']-r['margin_change']['ci'][0] for r in rr],[r['margin_change']['ci'][1]-r['margin_change']['mean'] for r in rr]])
            ax.errorbar(x,y,yerr=error,fmt='o',capsize=4,color='#1565c0');ax.axhline(0,color='#777',lw=.8)
            ax.set_xticks(x,[r['arm'] for r in rr]);ax.set_title(h);ax.set_ylabel('Change in correct-vs-best-rival logit margin')
        fig.savefig(R.OUT/'intervention-margin.png',dpi=170);plt.close(fig)
    allrows=R.materials();meta=R.read(R.OUT/'new_frames/collection.json')
    rows=[allrows[i] for i in meta['row_indices']]
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    trajectory_cases=[]
    for rownum,h in enumerate(R.CANDIDATES):
        j=next(i for i,r in enumerate(rows) if r['hypothesis']==h and r['role']=='operative' and r['lexical']==0)
        r=rows[j];z=R.unbits(np.load(R.OUT/'new_frames'/f'{h}-all-token-GVA.npy',mmap_mode='r')[j,:len(r['tokens'])])
        trajectory_cases.append(dict(hypothesis=h,row=r,selection='First archived material-order operative neutral-title new-frame case; order is split/family/role/title, not ranked by response'))
        for col,field in enumerate(['G','V','A']):
            ax=axes[rownum,col];ax.plot(np.arange(len(z)),z[:,col],color=['#1565c0','#bf4e16','#14805e'][col],lw=1)
            ax.axvline(r['content_end'],color='#333',ls='--',lw=.8);ax.axhline(0,color='#999',lw=.6)
            ax.set_title(f'{h}: {field}');ax.set_xlabel('Native token position');ax.set_ylabel('Native BF16 value')
    fig.suptitle('Two fixed cases: dashed line is content-end; later states see the question/options')
    fig.savefig(R.OUT/'joint-token-trajectories.png',dpi=170);plt.close(fig)
    R.save('trajectory-cases.json',dict(cases=trajectory_cases))


def build():
    R.configure();p=R.read(R.OUT/'plan.json');sel=R.read(R.OUT/'selection.json');report=R.read(R.OUT/'response-report.json')
    interventions=R.read(R.OUT/'intervention-report.json') if (R.OUT/'intervention-report.json').exists() else None
    plots(report,interventions)
    accepted=sum(report['hypothesis_passed'].values());chains_accepted=sum(report['backward_passed'].values())
    evaluable=[r for r in report['results'] if 'prediction' in r]
    embedding_better=sum(r['prediction']['embedding']['per_field_MSE'][0]<r['prediction']['combined']['per_field_MSE'][0] for r in evaluable)
    priority=('可以对通过预测门的局部连接继续做机制追踪，完整图谱仍须逐段验收。' if chains_accepted else
              '当前不宜优先铺开完整的逐层回溯图谱；应先解决局部预测与原模型作用的验证。')
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen14 分支验证与末层回溯</title>',
        '<style>body{font:16px/1.7 system-ui,sans-serif;margin:0;background:#f4f6f9;color:#172234}main{max-width:1150px;margin:auto;padding:28px 20px}h1{font-size:30px;line-height:1.3}h2{margin-top:36px}section,details{background:white;border:1px solid #d9e1ed;border-radius:9px;padding:16px 20px;margin:14px 0}p{max-width:94ch}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:9px 12px;border-bottom:1px solid #e1e6ef;white-space:nowrap}th{background:#eff3f9}img{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace;background:#f3f5f8;padding:12px}a{color:#145caa}.small{font-size:13px;color:#526172}.formula{font-family:ui-monospace,monospace}.note{border-left:4px solid #bf822a;padding-left:14px}summary{cursor:pointer;font-weight:600}@media(max-width:600px){main{padding:15px 10px}section,details{padding:13px}h1{font-size:24px}}</style><main>',
        '<p class="small">真实本地 BF16 / CUDA · 原生单元编号零基 · '+esc(R.RUN)+'</p><h1>Qwen3-14B：关系分支验证与末层回溯</h1>',
        '<section><p><strong>结果：'+str(accepted)+'/2 个固定单元解释通过三类留出的联合门，'+str(chains_accepted)+'/2 组末层回溯链通过全部留出预测门。</strong></p><p>'+priority+'</p><p>未通过只否定本次冻结的解释、读出和预测算法，不证明模型没有关系计算或递归式特征形成。</p></section>',
        '<p class="note">简单的前缀平均embedding基线在'+str(embedding_better)+'/'+str(len(evaluable))+'个留出比较中的G预测误差低于“词汇＋关系元数据”组合模型。仅胜过本次词汇模型不够；完整基线结果保留在下表和response-report.json中。</p>',
        '<section><p>本轮直接验证两个属于 14B 的历史候选：B26/U13099 与 B19/U64。另四个候选来自 4B，保持待检，不按编号映射到 14B。</p>',
        '<p>共 6,144 条材料，1,024 个家族。每家族有指令式规则、参考书描述、只列条目三种表达，交叉有／无技术词标题。每个候选在每种留出条件有 128 个家族。三选一行为题含真实错误映射或错误顺序。</p>',
        '<p>协议开发时，“引用但未生效”的原任务没有通过行为门。发现集诊断后，任务收窄为描述关系的假设执行；校准为379/384。该数字是开发结果，不计入独立确认，原来的生效判断失败保留。所有家族划分保持不变，未把旧发现材料移入留出集。</p>',
        '<p class="note">这是固定快照中响应规律和有限干预的检验。它不能确定这些响应分支怎样在训练中形成，也不能凭静态关联证明训练中的“递归泛化”。</p></section>',
        '<h2>候选验证</h2>']
    summary=[]
    for r in report['results']:
        c=r.get('contrasts',{}).get('G',{}).get('relation',{})
        summary.append([NAMES[r['hypothesis']],SPLITS[r['split']],'通过' if r['behavior_passed'] else '未通过',f(c.get('mean')),f(c.get('dz')),f(c.get('p_two_sided')),f(r.get('combined_to_lexical_G_MSE')),'通过' if r['primary_passed'] else '未通过'])
    parts.append(table(['对象','留出条件','行为门','G 关系差','配对 dz','p','组合/词汇 G MSE','预注册联合门'],summary))
    parts+=['<section><details><summary>冻结的原生单元组与末层根单元</summary>']
    for h in R.CANDIDATES:
        group=sel['groups'][h];chain=sel['backward'][h]
        parts+=['<p><strong>'+esc(NAMES[h])+'</strong><br>干预组：'+esc(group['units'])+
                '<br>匹配组：'+esc(group['matched_units'])+'<br>B39 根单元：'+esc(chain['roots'])+'</p>']
        for edge in chain['edges']:
            parts+=['<p class="small">B'+str(edge['parent_block'])+' 预测父组：'+esc(edge['parents'])+'</p>']
    parts+=['<p class="small">选择只使用发现集。这里的“父组”表示预测输入，尚不是已证明的原生因果父节点。</p></details></section>']
    joint=[]
    for r in report['results']:
        if 'contrasts' not in r:continue
        joint.append([NAMES[r['hypothesis']],SPLITS[r['split']]]+
                     [f(r['contrasts'][field]['relation']['mean']) for field in ['G','V','A']]+
                     [f(r['prediction'][method]['per_field_MSE'][0]) for method in ['combined','embedding']]+
                     [f(r['prediction'][method]['per_field_MSE'][2]) for method in ['combined','combined_composed','embedding']])
    parts+=['<details><summary>联合核对 G、V、A 与写入预测误差</summary>',
            table(['对象','留出','ΔG','ΔV','ΔA','组合模型G误差','平均embedding G误差','直接预测A误差','G/V组合A误差','平均embedding A误差'],joint),
            '<p class="small">误差按对应字段的发现集标准差标准化。由两个预测均值计算 SiLU(G)×V 不能自动恢复 G/V 联合分布；不能把它与真实单次前向的恒等式混淆。</p></details>']
    parts+=['<p><strong>保留的次级线索：</strong>B19/U64的平均关系差在三类留出中，V均为正（约0.207、0.238、0.259），A也均为正（约0.01825、0.01030、0.01839）。这些是预定的次级响应对比，不能替换失败的正向G主检验；它们提示下一轮应冻结联合写入假设。当前未排除措辞和词汇因素，也未建立对应行为必要性。B26的V/A则随句式出现方向翻转。</p>']
    if (R.OUT/'joint-state-summary.json').exists():
        jointstats=R.read(R.OUT/'joint-state-summary.json')['summary']
        rows=[[NAMES[r['hypothesis']],SPLITS[r['split']],r['role'],
               f(r['mean_native_A']),f(r['product_of_mean_siluG_and_meanV']),
               f(r['conditional_covariance']),f(r['native_mean_minus_fp64'])] for r in jointstats]
        parts+=['<details><summary>联合状态的事后分解：平均值与协变</summary>',
                table(['对象','留出','表达','原生平均A','平均SiLU(G)×平均V','SiLU(G)、V协变','原生BF16与FP64分解的残差'],rows),
                '<p class="small">这是同一批原始数据的描述性恒等分解，没有新的确认检验。使用FP64重新计算SiLU(G)时，与原生BF16激活存在舍入残差；该残差单独报告。正负象限与绝对A份额见joint-state-summary.json。</p></details>']
        example=next(x for x in jointstats if x['hypothesis']=='procedure' and x['split']=='new_words' and x['role']=='quoted')
        parts+=['<p>B19/U64在新词参考描述条件中，G、V双负样本贡献了该条件绝对A总量的'+f(100*example['quadrants']['--']['native_abs_A_share'])+
                '%。这个描述性例子说明负G仍可与负V形成正A，不能按单一门值正负判断功能；它不改变正向G冻结假设失败的结果。</p>']
    parts+=['<p class="small">联合门要求：三类留出分别通过行为门；G 关系差为正、dz≥0.35、p&lt;0.05/6，且指令与参考描述各自相对清单的均值均为正；加入关系元数据后的 G 预测 MSE 至少比词汇模型低10%。关系响应与超过词汇基线的预测增量分别保留。</p>',
        '<img src="response-contrasts.png" alt="关系、标题和指令参考句式的原始G对比"><p class="small">关系误差条使用六次主检验校正区间；标题和指令／参考句式为探索性95%区间，不据此自动升级语义解释。</p>',
        '<section><h2>数学对象与读出</h2><p class="formula">G = gᵀz；V = uᵀz；A = SiLU(G)V；单元写入 = A·d。</p>',
        '<p>z 是实际归一化后的 MLP 输入；g、u 是 gate/up 权重行，d 是 down 权重列。正文结束位置在问题与选项之前，当前 token 对所有材料相同。G、V、A 是不同对象：共同高 G 不保证 A 同号或写出功能相同。</p>',
        '<p class="formula">Δ关系 = (G指令 + G参考描述)/2 − G条目；Δ标题 = G技术标题 − G中性标题。</p>',
        '<p>拟合只使用发现集。比较词身份/双词组哈希、长度与位置、前缀平均 embedding、已知角色元数据以及两者组合。角色元数据来自实验材料，不是模型自己发现的解析器。预测 A 同时比较直接拟合与 SiLU(预测G)×预测V。</p></section>',
        '<img src="joint-token-trajectories.png" alt="两个固定案例的逐token G、V、A原始曲线"><p class="small">按材料顺序固定取例，没有按响应强弱挑选。虚线以后的位置已经看见后续问题或选项；不能反过来解释虚线之前的状态。原文见 trajectory-cases.json。</p>',
        ]
    if (R.OUT/'position-diagnostic.json').exists():
        diag=R.read(R.OUT/'position-diagnostic.json')
        parts+=['<section><h2>事后位置诊断：句中响应不等于末位保持</h2><p>这项诊断在冻结主判定完成后才计算，使用已有逐token数据，不重新选择单元或改变干预。下列峰值只取正文，排除标题、后续问题和选项；最大值受句长和词汇影响，没有独立确认含义。</p>']
        for h in R.CANDIDATES:
            parts+=['<details><summary>'+esc(NAMES[h])+'：正文G峰值与清单对照</summary>']
            for split in R.SPLITS[1:]:
                rows=[next(x for x in diag['summary'] if x['hypothesis']==h and x['split']==split and x['role']==role) for role in ['operative','quoted','inventory']]
                parts+=['<p>'+esc(SPLITS[split])+'，指令／参考描述／清单的平均正文G峰值：'+esc(' / '.join(f(x['mean_body_max_G']) for x in rows))+
                        '；正文至少一次G&gt;0的条数：'+esc(' / '.join(str(x['any_body_G_positive'])+'/'+str(x['rows']) for x in rows))+'。</p>']
            parts+=['</details>']
        parts+=['<p>清单中关于“未提供关系或过程”的否定描述仍可能唤起关系概念，因此这不是“概念绝对不存在”的对照。G为负也不等于单元关闭：SiLU(G)通常仍非零，须结合V和实际A判断写入。</p></section>']
    parts+=['<h2>从最后一层向前的可行性试验</h2>']
    chainrows=[]
    for r in report['backward_chains']:
        free=r.get('free_rollout',{})
        chainrows.append([NAMES[r['hypothesis']],SPLITS[r['split']],f(free.get('ratio_to_mean')),'通过' if r['passed'] else '未通过'])
    parts.append(table(['候选主题','留出条件','B36→B39 自由递推 MSE / 均值','链条联合门'],chainrows))
    edgerows=[]
    for r in report['backward_chains']:
        for edge in r.get('edges',[]):
            errors=edge['errors'];mse=errors['parent']['standardized_MSE']
            edgerows.append([NAMES[r['hypothesis']],SPLITS[r['split']],
                f"B{edge['parent_block']}→B{edge['child_block']}",f(errors['parent']['ratio_to_mean']),
                f(mse/errors['lexical']['standardized_MSE']),f(mse/errors['random']['standardized_MSE'])])
    parts+=['<details><summary>逐段预测：均值、词汇和随机父组对照</summary>',
            table(['主题','留出','边','父组/均值 MSE','父组/词汇 MSE','父组/随机父组 MSE'],edgerows),
            '<p class="small">各比值越低越好，冻结门要求≤0.9。响应预测边不是原模型因果边；随机父组同样可能携带广泛的任务状态。</p></details>']
    parts+=['<img src="backward-rollout.png" alt="末层回溯链的留出自由递推误差">',
        '<p>每个主题从 B39 发现集选 8 个响应根单元，再为 B38、B37、B36 各选32个父单元。父层 A 预测子层 G/V/A，比较词汇模型和同数随机父单元。自由递推仅以真实 B36 A 起步，之后使用预测值。通过也只得到有留出预测支持的响应边。</p>',
        '<p>全部40层的原生场已经保存，不等于40层都已有解释。完整回溯的优先级应由预测链是否保留未见条件信息决定；继续逐层筛选必须使用新的冻结确认材料。</p>']
    if (R.OUT/'full-field-maps.json').exists():
        parts+=['<section><h2>从末层到首层：完整坐标观察</h2><p>下面只使用发现集，保持每层17,408个原生单元的编号顺序，分别显示G、V、A。B39在上、B0在下；没有按强度排序或只留下强响应。不同层相同编号不代表相同功能。</p>',
                '<p class="small">显示采用每种场、每种视图跨所有层固定的对称色标，绝对值99.5%分位以外截色；阈值与截色数量可查full-field-maps.json。屏幕像素不足以逐个展示全部单元，精确数组保存在NPZ中。原始差值与家族配对dz分开展示，图形不构成语义或因果证据。</p>']
        for h in R.CANDIDATES:
            parts+=['<details><summary>'+esc(NAMES[h])+'：全坐标原始值与标准化视图</summary>']
            for view,label in [('raw','原始关系差'),('standardized','家族配对标准化差')]:
                name=f'full-field-{h}-{view}.png'
                parts+=['<a href="'+name+'"><img src="'+name+'" alt="'+esc(NAMES[h]+' '+label)+'"></a>']
            parts+=['<p><a href="full-field-'+h+'.npz">下载未排序的完整数组</a></p></details>']
        parts+=['</section>']
    if interventions:
        parts+=['<h2>原模型单元和单元组干预</h2><p>每个候选在192个目标家族和48个另一任务家族上测试。只改变正文部分选定单元的真实 A，再走原生 down 投影与后续层；0倍与2倍，配有同层匹配单元组、写入范数控制和1倍重算检查。名为restore的臂直接重算原始A，是无操作数值检查，不是独立因果救援。</p>']
        parts+=['<p><strong>本轮未检出单独移除两个候选造成的目标答案间隔下降。</strong> B26八单元组移除后间隔小幅上升约0.00825 logit（四项主检验校正后仍显著），但相对匹配组的差异区间含零，关系相对清单的差异也未显著。B19组移除没有显著间隔变化。这不构成共同关系的必要性或选择性证据，也不能把未显著当作所有条件均无作用。</p>']
        rows=[]
        for r in interventions['results']:
            if r['role']=='all' and r['arm'] in ['single_0','group_0','matched_group_0']:
                c=r['margin_change'];rows.append([NAMES[r['hypothesis']],r['scope'],r['arm'],c['n_families'],f(c['mean']),f(c['p_two_sided']),f(r['baseline_accuracy']),f(r['accuracy']),f(r['mean_KL'])])
        parts.append(table(['对象','范围','操作','家族','答案间隔变化','p','原正确率','操作后正确率','全词表KL'],rows))
        parts+=['<img src="intervention-margin.png" alt="单元与单元组移除对答案间隔的影响">',
            '<p class="small">四项主检验为目标任务的单元0倍和单元组0倍，alpha=0.05/4。其他剂量、角色分层及副作用为探索性结果。全词表KL衡量原模型干预变化，不是图谱预测保真度。所有范数不匹配行保留；严格控制比较仅用全部受改 token 均满足2%容差的行。</p>']
        normrows=[]
        for r in interventions['results']:
            if r['role']=='all' and r['arm'].endswith('norm_control'):
                c=r['strict_matched_native_minus_control']
                normrows.append([NAMES[r['hypothesis']],r['scope'],r['arm'],
                    f"{r['strict_norm_matched_rows']}/{r['rows']}",
                    str(c['n_families']) if c else '0',f(c['mean']) if c else '—',
                    str([round(v,5) for v in c['ci']]) if c else '无有效比较'])
        parts+=['<details><summary>实际写入范数控制：合格覆盖与配对比较</summary>',
            table(['对象','范围','控制','全部token合格行/总行','涉及家族','原生−随机方向间隔变化','探索性95%区间'],normrows),
            '<p class="small">合格子集可能偏向写入较大的样本；不能把其结果推广到全部材料。这里按有合格行的家族聚合，六个变体并非六个独立家族。</p></details>']
        if (R.OUT/'control-comparisons.json').exists():
            controls=R.read(R.OUT/'control-comparisons.json')['results']
            cr=[[NAMES[c['hypothesis']],c.get('arm','group0 − matched group0'),c['contrast'],
                 f(c['comparison']['mean']),str([round(x,5) for x in c['comparison']['ci']])] for c in controls]
            parts+=['<section><h3>匹配对照与条件差异</h3>',table(['对象','操作','比较','均值','探索性95%区间'],cr),
                    '<p class="small">这些对照用于检查一般扰动和条件差异，不增加新的通过门槛。标题、句长、任务难度与措辞的替代解释仍须保留；显著损伤不自动证明关系特异性。</p></section>']
    parts+=['<h2>原文与反例</h2><p>下面同时保留每个角色的低响应与高响应。标记的数值只对应正文结束，未使用后面的答案提示解释当时的状态。</p>']
    examples=R.read(R.OUT/'counterexamples.json')['examples']
    for e in examples:
        r=e['row'];label=f'{NAMES[e["hypothesis"]]} · {SPLITS[e["split"]]} · {e["role"]}/{e["pole"]} · G={f(e["values"]["G"])} V={f(e["values"]["V"])} A={f(e["values"]["A"])}'
        parts+=['<details><summary>'+esc(label)+'</summary><p class="small">'+esc(r['id'])+'</p><pre>'+esc(r['body'])+'</pre><details><summary>后续行为题（不属于上述读出可见前缀）</summary><pre>'+esc(r['prompt'][len(r['body']):])+'</pre></details></details>']
    parts+=['<h2>解释边界与下一阶段</h2><section><p>这些新领域是受控的背景替换，共享配对或排序框架；它们不足以代表完整自然领域的泛化。新词指发现集未使用过的常用填充词，不是模型训练时未见的词。新句式同时使用新词，不能把失败单独归因于句式。配对问题固定查询第一来源，尚未排除固定槽位策略。不同留出条件共享部分词池，重复词和模板的依赖也未完全消除。</p>',
        '<p>主检验只针对共同正文末位，检验该位置是否保留可推广的关系响应；失败不能排除候选在正文其他位置短暂起作用。逐token的G/V/A已经保留，但图上事后发现的峰值不能替代冻结主检验，需要新的位置假设与留出验证。</p>',
        '<p>同一固定数学规则可以在多个语境复用，但需要分别证明：外部关系可以预测响应、G/V联合条件确定写入、写入参与行为、以及训练时怎样形成。这四层证据不能互相替代。本轮没有训练轨迹、独立节点复算、同预算字典方法、自由自然生成或通用能力保持结论。</p>',
        '<p>下一阶段围绕“共同关系如何被读入并形成有用写入”组织一个大任务：增加自然表达和语义反例，控制关系与局部措辞；冻结原生单元组与更强上下文/字典基线；检验未见组合、完整输出预测与匹配干预；仅对通过的候选继续回溯注意力来源、归一化及MLP参数。训练起源另需合法检查点或受控训练。</p></section>',
        '<h2>可追溯文件</h2><ul>']
    for name in ['plan.json','materials-audit.json','material-revision.json','scheduler-revision.json','choice-boundary-audit.json','future-prefix-audit.json','model-identity.json','selection.json','response-report.json','joint-state-summary.json','position-diagnostic.json','intervention-report.json','control-comparisons.json','feature-cards.json','full-field-maps.json','counterexamples.json','trajectory-cases.json','pilot.json','audit.json','protocol-result.json','language-compat-audit.json','algorithm-checks.log','progress.jsonl','frozen-source.zip']:
        if (R.OUT/name).exists():parts.append('<li><a href="'+name+'">'+name+'</a></li>')
    for name in ['pilot.json','diagnosis/report.json','neutral/report.json']:
        href='../recursive_branch14_20261113_v5/'+name
        parts.append('<li><a href="'+href+'">v5 开发失败与诊断：'+name+'</a></li>')
    parts+=['</ul><p class="small">生成时间 '+time.strftime('%Y-%m-%d %H:%M')+'。协议、原始场、日志、原文与反例保存在同一结果目录；未向公网发布。</p></main></html>']
    (R.OUT/'report.html').write_text('\n'.join(parts),encoding='utf8')
    R.save('report-build.json',dict(source_sha256=R.file_sha(__file__),report_sha256=R.file_sha(R.OUT/'report.html'),
        inputs={n:R.file_sha(R.OUT/n) for n in ['plan.json','selection.json','response-report.json','counterexamples.json',
            'intervention-report.json','control-comparisons.json','joint-state-summary.json','position-diagnostic.json','full-field-maps.json']
            if (R.OUT/n).exists()},images=[p.name for p in R.OUT.glob('*.png')]))


if __name__=='__main__':build()
