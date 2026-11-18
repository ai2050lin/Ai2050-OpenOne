"""Audit position-only heldout forecasts and native residual interventions."""

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

import paradigm_position_field as S
import paradigm_position_native as N
from b36_semantic14_memo import append
from deploy.storage_policy import check_write


R,C,M,P,OUT=S.R,S.C,S.M,S.P,S.OUT
TITLE='固定措辞下的档案与请求位置、B32 前残差写入及联合阻断'
MEMO=R.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'


def stat(v):
    a=np.asarray(v)
    return S.X.V.summary(a) if len(a) else dict(mean=None,ci=[None,None],n=0)


def show(statistic):
    if not statistic['n']:
        return '缺有效对照'
    mean=statistic['mean']
    lo,hi=statistic['ci']
    return f'{mean:+.4f} [{lo:+.4f}, {hi:+.4f}]'


def direction(s):
    if not s['n']:
        return 0
    if s['ci'][0]>0:
        return 1
    return -1 if s['ci'][1]<0 else 0


def pairing(data,key):
    assert (data['archive'].reshape(-1,2)==[0,1]).all()
    return data[key][1::2]-data[key][::2]


def location_effect(data,candidate,tok=None):
    n=len(data['ids'])//16
    assert n==128 and (data['layout'][::2].reshape(n,2,4)==np.arange(4)).all()
    assert (data['cell'][::2].reshape(n,2,4)==np.array(M.TARGETS)[None,:,None]).all()
    true=pairing(data,'margin')[:,1].reshape(n,2,4)
    reads=pairing(data,'project')[:,candidate['selected']]
    k=(data['cell'][::2]==15).astype(int)*4+data['layout'][::2]
    base=np.asarray(candidate['baseline'])[k]
    centered=(reads-np.asarray(candidate['feature_means'])[k])/candidate['scales']
    fitted=base+centered@candidate['beta']
    base=base.reshape(n,2,4);fitted=fitted.reshape(n,2,4)
    records={}
    for i,cell in enumerate(M.TARGETS):
        y=true[:,i]
        baseline_err=np.abs(base[:,i]-y)
        fit_err=np.abs(fitted[:,i]-y)
        estimates=[stat(y[:,a]) for a in range(4)]
        edges={}
        for a,b in S.LAYOUT_EDGES:
            slope=stat(y[:,b]-y[:,a])
            key=f'{a}-{b}'
            reversed_rows=np.flatnonzero(y[:,a]*y[:,b]<0)
            favored=reversed_rows[np.argsort(-np.abs(y[reversed_rows,b]-y[reversed_rows,a]))][:10]
            examples=[]
            for f in favored:
                origin=int(f)*16+i*8
                positions=[origin+int(step)*2+arc for step in (a,b) for arc in range(2)]
                tokens=[int(data['top'][j,0]) for j in positions]
                examples.append(dict(ids=[str(data['ids'][j]) for j in positions],family=str(data['family'][origin]),
                    effects=[float(y[f,a]),float(y[f,b])],first_tokens=tok.batch_decode([[t] for t in tokens]) if tok else tokens))
            edges[key]=dict(move='archive' if a^b==2 else 'request',difference=slope,
                           replicated_reversal=direction(estimates[a])*direction(estimates[b])==-1,
                           same_positive=direction(estimates[a])==direction(estimates[b])==1,
                           individual_reversal=len(reversed_rows),counterexamples=examples)
        records[f'{cell:04b}']=dict(layouts=estimates,edges=edges,
            archive_main=stat((y[:,2:]-y[:,:2]).mean(1)),
            request_main=stat((y[:,[1,3]]-y[:,[0,2]]).mean(1)),
            interaction=stat(y[:,3]-y[:,2]-y[:,1]+y[:,0]),
            baseline_MAE=float(baseline_err.mean()),residual_MAE=float(fit_err.mean()),
            MSE_gain=stat((baseline_err**2-fit_err**2).mean(1)))
    return records


def field(data,tok):
    trace=data['trace']
    fields=data['field']
    assert trace.shape==(256,33,160,6) and fields.shape[0:2]==(2,41) and fields.shape[-1]==5120
    materials=S.M.material(tok,'discovery')[0]
    assert [x['id'] for x in materials]==data['ids'].tolist()
    one=materials[0]
    start=one['source'][0]
    assert np.array_equal(fields[0,:,:start],fields[1,:,:start])
    out={}
    heat=[]
    for layout in range(4):
        choices=np.flatnonzero(data['layout'][::2]==layout)
        rows=[materials[2*i] for i in choices]
        rel=72
        sum_rms=np.zeros((33,rel))
        count=np.zeros(rel,int)
        first={}
        for i,row in zip(choices,rows,strict=True):
            source=row['source']
            e=row['query'][0]
            width=min(rel,e-source[-1])
            values=trace[i,:,source[-1]+1:source[-1]+width+1]
            prefix=trace[i,:,:source[0],[0,1,2]]
            assert (prefix==0).all()
            count[:width]+=1
            sum_rms[:,:width]+=values[:,:,2]
        mean=sum_rms/count.clip(1)
        mean[:,count==0]=np.nan
        heat.append(mean)
        for site,name in enumerate(M.ANCHORS):
            ps=data['sites'][::2][choices,site]
            s=data['source'][::2,-1][choices]
            keep=ps>s
            if not keep.any():
                continue
            r=trace[choices[keep],:,ps[keep],:]
            after=r[:,:,1]
            any_after=after>=S.RAW_THRESHOLD
            layers=np.where(any_after.any(1),any_after.argmax(1),-1)
            any_exact=after>0
            exact=np.where(any_exact.any(1),any_exact.argmax(1),-1)
            displaced=r[np.arange(len(r)),layers.clip(0),0]
            hits=layers>=0
            first[name]=dict(n=len(ps[keep]),strong_n=int(hits.sum()),
                first_hist={str(l):int((layers==l).sum()) for l in sorted(set(layers))},
                exact_hist={str(l):int((exact==l).sum()) for l in sorted(set(exact))},
                direct_attention_first=int(((displaced<S.RAW_THRESHOLD)&hits).sum()),
                B0_attention_mean=float(after[:,0].mean()),B0_after_mean=float(r[:,0,2].mean()),
                token_examples=sorted(set(tok.decode([row['token']['ids'][row['sites'][site]]]) for row in rows))[:3],
                position_range=[int(ps.min()),int(ps.max())])
        peaks=[]
        for k in [i for i in np.argsort(-mean[0]) if count[i]>=32][:8]:
            actual=[row['source'][-1]+1+int(k) for row in rows]
            labels=[tok.decode([row['token']['ids'][pos]]) for row,pos in zip(rows,actual) if pos<=row['query'][0]]
            peaks.append(dict(offset=int(k+1),mean_B0=float(mean[0,k]),n=int(count[k]),
                         tokens=sorted(set(labels),key=lambda w:(-labels.count(w),w))[:3]))
        out[str(layout)]=dict(by_anchor=first,B0_largest=peaks)
    return dict(full_ids=data['field_ids'].tolist(),shape=list(fields.shape),positional_prefix_verified=True,
                field_shape=list(trace.shape),layouts=out),heat


def fusion(data,tok):
    scores=data['pred']
    assert scores.shape==(64,2,len(S.A.RULES))
    y=scores[:,:,S.A.RULES.index('add')]
    z=scores[:,:,S.A.RULES.index('pair')]
    r=scores[:,:,S.A.RULES.index('target_1011')]
    results={}
    for i,cell in enumerate(M.TARGETS):
        original=cell+np.arange(len(scores))*16
        rows=[]
        for n in np.argsort(-(z[:,i]-y[:,i]))[:12]:
            row=original[n]
            rows.append(dict(id=str(data['ids'][row]),layout=int(data['layout'][row]),base=float(y[n,i]),
                pair=float(z[n,i]),rule=float(r[n,i]),
                top=tok.batch_decode([[int(j)] for j in data['pred_top'][n,i,:4]])))
        results[f'{cell:04b}']=dict(add=stat(y[:,i]),pair=stat(z[:,i]),rule=stat(r[:,i]),
            harmed=int((z[:,i]>y[:,i]).sum()),rule_gain_over_add=stat(y[:,i]-r[:,i]),counterexamples=rows)
    results['aggregate']=dict(add=stat(y.mean(1)),pair=stat(z.mean(1)),rule=stat(r.mean(1)),
        gain_vs_add=stat((y-r).mean(1)),gain_vs_pair=stat((z-r).mean(1)))
    return results


def causal(data,c,tok):
    assert len(data['ids'])==256 and len(set(data['family']))==64
    assert (data['kl'][:,[0,8]]==0).all()
    margins={name:data['edit'][:,j] for j,name in enumerate(N.BRANCHES)}
    original=data['natural'][:,1]
    contrast=original-data['natural'][:,0]
    sign=np.sign(contrast)
    delta={name:sign*(original-m) for name,m in margins.items()}
    both=data['dose'][:,0];recv=data['dose'][:,1]
    weights=dict(layouts={},placement_level={})
    include=('single_0','single_1','single_2','joint','alt','joint_at_E','skip_at_E','mlp_at_E')
    for index,cell in enumerate(M.TARGETS):
        layouts=c['layouts']
        use=[(data['cell']==cell)&(data['layout']==layout) for layout in layouts]
        assert all(x.sum()==64 for x in use)
        assert np.array_equal(data['family'][use[0]],data['family'][use[1]])
        key=f'{cell:04b}'
        weights['layouts'][key]={}
        placement=contrast[use[1]]-contrast[use[0]]
        expected=np.sign(c['baseline'][index*4+layouts[1]]-c['baseline'][index*4+layouts[0]])
        def closure(op):
            loss=original-margins[op]
            return expected*(loss[use[1]]-loss[use[0]])
        cut={label:stat(closure(label)) for label in include}
        match=(both[use[0],5]>0)&(both[use[1],5]>0)
        compare_ctrl=stat((closure('joint_match')-closure('alt_match'))[match])
        single_advantage=stat(closure('joint')-closure('single_0'))
        global_effect=stat(placement)
        mean=global_effect['mean']
        joint_share=cut['joint']['mean']/max(.000001,abs(mean))
        weights['placement_level'][key]=dict(natural=global_effect,closure=cut,
            matched_joint_vs_alt=compare_ctrl,joint_vs_strongest=single_advantage,
            dose_coverage=float(match.mean()),discovery_orientation=float(expected),
            explained=joint_share,passed=bool(joint_share>=.10 and direction(cut['joint'])==1 and
                direction(compare_ctrl)==1 and direction(single_advantage)==1 and match.mean()>=.98 and expected*mean>0))
        for layout,keep in zip(layouts,use,strict=True):
            n_mask=keep&both[:,5].astype(bool)
            g_mask=keep&recv[:,5].astype(bool)
            described={name:stat(delta[name][keep]) for name in include}
            abs_natural=float(np.abs(contrast[keep]).mean())
            abs_joint=np.abs(delta['joint'])
            recovery=stat((abs_joint-np.abs(delta['gu_match']))[g_mask])
            restore_control=stat((np.abs(delta['random_match'])-np.abs(delta['gu_match']))[g_mask])
            gu_pass=bool(g_mask.sum()/keep.sum()>=.98 and direction(recovery)==direction(restore_control)==1)
            chosen=np.flatnonzero(keep)
            worst=chosen[np.argsort(delta['joint'][keep])[:4]]
            bad=chosen[~(n_mask&g_mask)[chosen]]
            cases=[]
            for i in dict.fromkeys([*worst,*bad]):
                cases.append(dict(id=str(data['ids'][i]),delta=float(contrast[i]),joint=float(delta['joint'][i]),
                    single=float(delta['single_0'][i]),dose=data['dose'][i].tolist(),
                    tokens=tok.batch_decode([[int(j)] for j in [*data['original'][i],data['tops'][i,4]]])))
            archived=data['gua'][keep,1]-data['gua'][keep,0]
            weights['layouts'][key][str(layout)]=dict(natural=stat(contrast[keep]),effects=described,
                mlp_skip_nonadd=stat((delta['joint_at_E']-delta['skip_at_E']-delta['mlp_at_E'])[keep]),
                extra_positions=stat((delta['joint']-delta['joint_at_E'])[keep]),
                joint_minus_singles=stat((delta['joint']-delta['single_0']-delta['single_1']-delta['single_2'])[keep]),
                joint_share=described['joint']['mean']/max(.00001,abs_natural),
                matched_joint_vs_alt=stat((delta['joint_match']-delta['alt_match'])[n_mask]),
                source_equal=P.dose_percentiles(both[keep]),receiver_equal=P.dose_percentiles(recv[keep]),
                recovery=recovery,recovery_vs_random=restore_control,receiver_passed=gu_pass,
                native_g_and_u={arm:stat((abs_joint-np.abs(delta[arm]))[keep]) for arm in ('freeze_g','freeze_u','freeze_gu','freeze_random')},
                gua_abs_selected=np.abs(archived[:,:32]).mean((0,1)).tolist(),
                gua_abs_random=np.abs(archived[:,32:]).mean((0,1)).tolist(),counterexamples=cases)
    return weights


def picture(heat,c):
    os.environ['MPLCONFIGDIR']=str(S.TMP/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig,axes=plt.subplots(2,2,figsize=(10.2,6.8),layout='constrained',sharex=True,sharey=True)
    for i,ax in enumerate(axes.flat):
        h=np.log10(np.maximum(heat[i],1e-6))
        image=ax.imshow(h,extent=(.5,len(h[0])+.5,-.5,h.shape[0]-.5),aspect='auto',origin='lower',
                        cmap='viridis',vmin=-4,vmax=1,interpolation='nearest')
        a,r=M.LAYOUTS[i]
        a=('early','late')[a];r=('early','late')[r]
        ax.set_title(f'Layout {i}   ·   archive {a} / request {r}')
        ax.axhline(c['block'],color='white',alpha=.65,ls=':',lw=1)
        ax.set_xlabel('Tokens downstream from archive')
        ax.set_ylabel('Qwen3-14B block')
    fig.colorbar(image,ax=axes.ravel().tolist(),label='log10 paired residual RMS',shrink=.9)
    fig.suptitle('Where the residual changes after a matched archive swap')
    p=check_write(OUT/'residual-first-write.png',5*2**20,operation='档案首次写入热图')
    fig.savefig(p,dpi=165)
    plt.close(fig)


def cross_runs(regular,native):
    indexed={k:i for i,k in enumerate(regular['ids'])}
    alignment=[[indexed[k[:-1]+'0'],indexed[k]] for k in native['ids']]
    top=int((regular['top'][alignment,0]!=native['original']).sum())
    margin=float(np.max(np.abs(regular['margin'][alignment,1]-native['natural'])))
    assert top==0 and margin<=1e-5
    return dict(views=2*len(alignment),top_disagreements=top,margin_max=margin)


def md(result):
    d=result['design'];c=result['candidate'];g=result['gate'];f=result['field']
    k=result['location'];n=result['native'];fusion_stats=result['fusion'];prev=result['previous']
    names='、'.join(M.ANCHORS[i] for i in c['selected'])
    ref=tuple(c['layouts'])
    key=f'{ref[0]}-{ref[1]}'
    text=[
        f"- 本地 Qwen3-14B 原 BF16，CPU 读取 safetensors、CUDA 按 40 块顺序执行，宽 160、批 8。运行 {M.RUN}。设计在 {d['at']} 冻结，候选在 {c['at']} 冻结，原生方法在 {result['holdout']['at']} 冻结。交叉 16 试跑、32 发现、128 独立确认家族，128 锁箱家族只哈希。Qwen 模型与 tokenizer 快照见冻结数据。",
        f"- Phase 2973 的 1011 两套句式出现 {prev['1011_first']:+.4f} / {prev['1011_second']:+.4f} 的方向反转，"
        f"1111 平均 {prev['1111_mean']:+.4f}。本轮单移对照检验位置是否足以复现；新模板改用固定双槽，控制填充语体和其他词位。",
        '- 固定整句和词汇。档案使用一对同载荷存档样例，单个样例占 6 token；要求格式占 5 token。用各自等长无效标记作交换。每次移动验证两个槽的 token 精确对调，其余 token、输入长度和四个输出读出位置逐位不变。位置 0–3 分别是档案/请求 早早、早晚、晚早、晚晚。比较档案内容时，同一布局只换活动档案槽。',
        f"- 试跑四因子最低成对行为率 {'、'.join(f'{key} {value["minimum_stratum"]:.0%}' for key,value in g['factors'].items())}。四布局各自档案代码/论文两格的格式对照最小 {min(min(row) for row in g['position_style']):.0%}。格式在自由首位评，其他因子使用已知续写。行为门不要求换档案后出现方向反转，因此没有筛掉反例。",
        f"- 全场覆盖 B0–B32 的 {f['field_shape'][0]} 条成对样本 × {f['field_shape'][2]} 个槽位，分析只用有因果可见性的实际 token。保留 {f['full_ids']} 的 41 层 × 有效 token × 5120 原生坐标。档案前位置逐位一致。用 .0005 RMS 作写入的稳健诊断线，同时保存原始零值与极小值。",
        r'$$D_{b,i}=|r^{mathrm{code}}_{b,i}-r^{mathrm{prose}}_{b,i}|_2/sqrt{5120},quad A_{b,i,u}=mathrm{SiLU}(G_{b,i,u})U_{b,i,u}.$$',
        '- 独立计算 attention 输出投影和 down_proj 的实际写入；首次非零以实际数值为准，进入注意力后的差异和残差输出的差异各有字段。早期强写入相对活动档案最后一个 token 的距离以第一条下游 token 为 1。下表为 B0 整体 RMS 排前的可解释 token，完整位置和层矩阵在发现分片。图见 residual-first-write.png。',
        '',
        '| 布局 | 最强 B0 下游写入（偏移、示例 token、RMS） |',
        '| --- | --- |']
    for l,entry in f['layouts'].items():
        examples='；'.join(f"+{p['offset']} {p['tokens'][0]!r} {p['mean_B0']:.4f}" for p in entry['B0_largest'][:4])
        text.append(f"| {l} | {examples} |")
    firsts=[v for i in f['layouts'].values() for v in i['by_anchor'].values()]
    total=sum(x['n'] for x in firsts)
    b0=sum(x['first_hist'].get('0',0) for x in firsts)
    text.append('')
    text.append(f"- 有效的命名位置×配对共 {total} 次，{b0} 次在 B0 注意力后的残差即超过诊断线。"
                "最强响应集中在紧邻档案的 Form / slot / 冒号，远处命名词位同层也可检测到变化。"
                "因此“最早写入”具有同层并行和强弱梯度，不能按词序理解成模型逐个搬运。")
    text.extend(['',
        f"- 冻结 B{c['block']} 输出边界、{names} 三个不同区域及其同层复位对照。发现阶段以固定幅度差而非留出结果选择只移动{('档案' if c['layout_move']=='archive' else '请求')}的布局 {ref[0]}→{ref[1]}。B32 自由起始位使用 32 个 G/U 候选及 32 个冻结随机单元。留出先跑相同计算形状的早期技术份额。",
        '',
        '| 组合 | 布局 0 | 布局 1 | 布局 2 | 布局 3 | 档案位 × 请求位 |',
        '| --- | --- | --- | --- | --- | --- |'])
    notes=[]
    for name,x in k.items():
        cols=[show(z) for z in x['layouts']]
        text.append(f"| {name} | "+' | '.join(cols)+f" | {show(x['interaction'])} |")
        edges='；'.join(f"{edge} {show(value['difference'])}" for edge,value in x['edges'].items())
        tags=x['edges'][key]
        notes.append(f"- {name} 单移对照 {edges}。冻结的 {key} 对照扩展到 128 家族后，"
                    f"群体方向反转 {'通过' if tags['replicated_reversal'] else '未通过'}，"
                    f"同向正效应 {'通过' if tags['same_positive'] else '未通过'}。"
                    f"布局基线和残差预测 MAE {x['baseline_MAE']:.4f} / {x['residual_MAE']:.4f}，MSE 改善 {show(x['MSE_gain'])}。")
    text.extend(['',*notes,''])
    counter=k['1111']['edges']['0-2']
    if counter['replicated_reversal']:
        c0=counter['counterexamples'][0]
        text.append(f"- 1111 新反例是固定早请求、档案从早挪到晚。群体换文效应 {show(k['1111']['layouts'][0])} → "
                    f"{show(k['1111']['layouts'][2])}，{counter['individual_reversal']}/128 家族反转。"
                    f"{c0['family']} 的配对间隔 {c0['effects'][0]:+.4f} → {c0['effects'][1]:+.4f}。"
                    f"该对照不在冻结的 {key} 原生布局内，路径机制留作后续检验。")
    if [direction(x) for x in k['1111']['layouts']]==[1,1,-1,1]:
        text.append('- 1111 的三个“先档案、后请求”布局都推高 print 相对分数，唯一“先请求、后档案”的布局压低分数。'
                    '把晚档案固定住，只把请求后移，也会恢复正效应。这个位置关系适用于当前双槽模板。')
    text.extend(['',
        '- 原生干预每格 64 个独立家族，256 个代码档案接收样本、256 个配对原生供体输入。单点与联合替换用同一家族、同一布局的论文档案残差，当前输入的词和其余位置保持代码档案端。正效应表示朝该家族原生换文方向移动。整套样本的无操作和原生复位全词表逐位一致。',
        f"- 常规留出与干预基线共有 {result['cross_runs']['views']} 个输入，首 token 分歧 {result['cross_runs']['top_disagreements']}，"
        f"格式间隔最大差 {result['cross_runs']['margin_max']:.3g}。",
        '- 选中与三个下游对照位置按联合实际 BF16 范数对齐；B32 后单独对齐选中与随机 G/U 恢复。正剂量下限 .003、相对差≤3%、对原生方向余弦≥.90。B32 之后单路实验将非起始位置恢复为原值，分别带走改写后的残差或 MLP 写回。联合 E 臂用于区分该位置与其他位置。',
        '',
        '| 组合、布局 | 单点最强 | 三点联合 | B32 残差单路 / MLP 单路 | E 以外剩余 | 等幅联合优势 | G/U 等幅恢复优势 |',
        '| --- | --- | --- | --- | --- | --- | --- |'])
    for cell,layouts in n['layouts'].items():
        for layout,x in layouts.items():
            effects=x['effects']
            text.append(f"| {cell} {layout} | {show(effects['single_0'])} | {show(effects['joint'])} | "
              f"{show(effects['skip_at_E'])} / {show(effects['mlp_at_E'])} | {show(x['extra_positions'])} | "
              f"{show(x['matched_joint_vs_alt'])} | {show(x['recovery_vs_random'])} |")
    text.append('')
    for cell,x in n['placement_level'].items():
        text.append(f"- {cell} 冻结布局的联合闭合比例 {x['explained']:+.1%}，同向联合效应 {show(x['closure']['joint'])}，"
            f"对比最强单点 {show(x['joint_vs_strongest'])}，跨布局等幅对照优势 {show(x['matched_joint_vs_alt'])}，"
            f"位置机制门 {'通过' if x['passed'] else '未通过'}。")
    for cell,layouts in n['layouts'].items():
        for layout,x in layouts.items():
            case=x['counterexamples'][0]
            text.append(f"- {cell} 布局 {layout} 的实际等幅覆盖 {x['source_equal']['coverage']:.0%} / "
                f"{x['receiver_equal']['coverage']:.0%}，三点相对三次单点之和 {show(x['joint_minus_singles'])}。"
                f"反向端 {case['id']}，自然差 {case['delta']:+.3f}，联合效应 {case['joint']:+.3f}。")
    comparisons=sum(direction(v['matched_joint_vs_alt'])<0 for layouts in n['layouts'].values() for v in layouts.values())
    text.append(f"- {comparisons}/4 个原生条件里，位置对照的等幅作用显著大于候选。对照横跨语言、括号与关闭附近，"
                "它们距离生成起点与候选不同，不能将优势直接解释为位置的特异性。当前结果足以限制三个候选的主干主张。")
    late=n['layouts']['1011'][str(ref[1])]['effects']
    early=n['layouts']['1111'][str(ref[0])]['effects']
    text.append(f"- 后移请求时 1011 的 B32 残差单路 {show(late['skip_at_E'])}，MLP 单路 {show(late['mlp_at_E'])}。"
                f"早请求的 1111 对应 {show(early['skip_at_E'])} 和 {show(early['mlp_at_E'])}。"
                "局部两个写入方向可以相反，但两臂在该边界重新装回未改前缀。证据只覆盖 B32 自由起始位的局部分解。")
    y=fusion_stats['1111'];t=y['counterexamples'][0]
    text.extend(['',
        f"- 64 个独立家族上使用上一轮冻结的只给 1011 加二阶规则，平均 KL "
        f"{fusion_stats['aggregate']['rule']['mean']:.5f}，全加法 {fusion_stats['aggregate']['add']['mean']:.5f}，"
        f"总用二阶 {fusion_stats['aggregate']['pair']['mean']:.5f}。1111 总用二阶有 {y['harmed']}/64 条变差；"
        f"最坏 {t['id']} 的加法→二阶 KL {t['base']:.6g}→{t['pair']:.6g}，"
        f"真实、加法、二阶首 token {t['top'][:3]!r}。全体原始含 1111 反例保存在 fusion 分片。",
        '- 第一性原理是因果位置限制。不同位置的词仅凭 attention 能改变后续位置，逐位置 MLP 只改变已携带该信号的位置。若多个位置的联合删除依然保留大量输出差，剩余项可能来自未处理的中间位置、直接反复读取档案、层间读取历史或非线性交互。本轮结论限定于三个冻结位置、一个早期边界和一个晚期位置。',
        '- 三图谱增量分别是等长槽位换序的可重算外部边、原生前缀各层写入与救援的内部边、及带未解释残差的预测和干预联系。下一阶段可换用不同无效标记复查位置边，并在等幅之外给位置对照匹配距档案与输出的距离。冻结分层位置截断及恢复预算后，再检验能否在自然生成中预测格式与内容的变化。',
        f"- 所有冻结、分片、反例和日志存于 tests/glm5/result/{OUT.name}。临时目录单独清理。"])
    return '\n'.join(text)+'\n'


def run(close):
    from transformers import AutoTokenizer
    design=R.read('design.json');candidate=R.read('candidate.json');plan=R.read(N.PLAN)
    gate=R.read('behavior.json')
    assert gate['passed'] and design['code']==S.registry() and design['previous']==S.bind_previous()
    assert plan['code']==N.source_id() and plan['candidate']==S.sha(OUT/'candidate.json')
    assert (OUT/'candidate.json').stat().st_mtime_ns<(OUT/N.PLAN).stat().st_mtime_ns
    for name in ('confirm','native','fusion'):
        assert (OUT/N.PLAN).stat().st_mtime_ns<(OUT/(name+'-files.json')).stat().st_mtime_ns
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
    first,heat=field(S.load('discover'),tok)
    raw=S.load('native',N.binding())
    regular=S.load('confirm',N.binding())
    native=causal(raw,candidate,tok)
    location=location_effect(regular,candidate,tok)
    retain=fusion(S.load('fusion',N.binding()),tok)
    old=json.loads((S.A.OUT/'analysis.json').read_bytes())['archives']
    previous={'1011_first':old['1011']['frame0']['format']['mean'],
              '1011_second':old['1011']['frame1']['format']['mean'],
              '1111_mean':old['1111']['all']['format']['mean']}
    data=dict(at=C.stamp(),design=design,candidate=candidate,holdout=plan,gate=gate,field=first,
              native=native,location=location,fusion=retain,previous=previous,cross_runs=cross_runs(regular,raw))
    R.write('analysis.json',data)
    picture(heat,candidate)
    body=md(data)
    check_write(OUT/'report.md',len(body.encode())+4096,operation='位置研究报告').write_text('# '+TITLE+'\n\n'+body,encoding='utf-8')
    if not close:
        return
    assert not (OUT/'memo-receipt.json').exists() and TITLE not in MEMO.read_text(encoding='utf-8-sig')
    verified=append(check_write(MEMO,len(body.encode())+4096,operation='追加 GLM5 研究记录'),TITLE,body)
    cleanup=[]
    if S.TMP.is_dir():
        folder=S.TMP.resolve()
        assert folder==S.TMP and folder.parent==(R.ROOT/'tests/glm5_temp').resolve()
        items=[folder,*folder.rglob('*')]
        assert all(not p.is_symlink() and not p.is_junction() and p.resolve().is_relative_to(folder) for p in items)
        removed=sum(p.stat().st_size for p in items if p.is_file())
        shutil.rmtree(folder)
        cleanup.append(dict(path=str(folder),freed_bytes=removed))
    R.write('memo-receipt.json',dict(memo=verified,cleanup=cleanup))
    files={p.name:dict(sha256=S.sha(p),bytes=p.stat().st_size) for p in sorted(OUT.iterdir())
           if p.is_file() and p.name!='manifest.json'}
    extra={p.relative_to(R.ROOT).as_posix():S.sha(p) for p in (Path(__file__).resolve(),Path(N.__file__).resolve())}
    R.write('manifest.json',dict(run=M.RUN,files=files,sources=design['code']|extra))
    print('GLM5 Phase',verified['phase'],'appended',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--close',action='store_true')
    args=parser.parse_args()
    S.setup()
    run(args.close)
